#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# Real llama.cpp integration regression for the v4 path.
#
# This hardware test intentionally checks two different facts:
#
# 1. A real llama.cpp workload can run on the local POLARIS backend when the
#    available llama.cpp binary exposes POLARIS0.
# 2. The LD_PRELOAD shim can bootstrap RM/UVM, register a Polaris VA-space,
#    route a real llama.cpp CUDA allocation through Polaris static RM backing,
#    and then stop cleanly at the current unsupported host-copy surface.
#
# Set POLARIS_LLAMA_STRICT_SHIM_FAULT_PASS=1 to turn the shim probe into the
# future M5 gate: the llama.cpp command must complete and uvm_hook_calls plus
# uvm_handled must increase.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/wano/workspace/llama.cpp}"
NVIDIA_KO_DIR="${NVIDIA_KO_DIR:-$ROOT_DIR/third_party/open-gpu-kernel-modules}"
SHIM_SO="${SHIM_SO:-$ROOT_DIR/libpolaris-shim/libpolaris-shim.so}"
STATS_PATH="${STATS_PATH:-/sys/kernel/polaris/stats}"

die() {
    echo "error: $*" >&2
    exit 1
}

note() {
    echo "==> $*" >&2
}

first_existing_executable() {
    local path
    for path in "$@"; do
        if [[ -f "$path" && -x "$path" ]]; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    return 1
}

first_existing_file() {
    local path
    for path in "$@"; do
        if [[ -f "$path" ]]; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    return 1
}

stat_value() {
    local key="$1"
    awk -v key="${key}:" '$1 == key { print $2; found = 1 } END { if (!found) print "0" }' "$STATS_PATH"
}

extract_last_metric() {
    local key="$1"
    local log="$2"
    sed -n "s/.*${key}=\([0-9][0-9]*\).*/\1/p" "$log" | tail -n 1
}

require_file() {
    local path="$1"
    local what="$2"
    [[ -f "$path" ]] || die "$what not found: $path"
}

require_executable() {
    local path="$1"
    local what="$2"
    [[ -f "$path" && -x "$path" ]] || die "$what is not an executable file: $path"
}

LLAMA_CPP_BIN="${LLAMA_CPP_BIN:-}"
if [[ -z "$LLAMA_CPP_BIN" ]]; then
    LLAMA_CPP_BIN="$(first_existing_executable \
        "$LLAMA_CPP_DIR/build-polaris/tools/llama-bench" \
        "$LLAMA_CPP_DIR/build-polaris/bin/llama-bench" \
        "$LLAMA_CPP_DIR/build/tools/llama-bench" \
        "$LLAMA_CPP_DIR/build/bin/llama-bench" \
    )" || die "could not find llama-bench; set LLAMA_CPP_BIN=/path/to/llama-bench"
fi

LLAMA_CPP_MODEL="${LLAMA_CPP_MODEL:-}"
if [[ -z "$LLAMA_CPP_MODEL" ]]; then
    LLAMA_CPP_MODEL="$(first_existing_file \
        "/home/wano/workspace/models/SmolLM2-135M-Instruct-F16.gguf" \
        "$LLAMA_CPP_DIR/models/tinyllamas/stories260K.gguf" \
    )" || die "could not find a runnable GGUF model; set LLAMA_CPP_MODEL=/path/to/model.gguf"
fi

require_executable "$LLAMA_CPP_BIN" "llama.cpp test binary"
require_file "$LLAMA_CPP_MODEL" "GGUF model"
require_file "$SHIM_SO" "libpolaris-shim.so"
require_file "$NVIDIA_KO_DIR/kernel-open/nvidia-uvm.ko" "patched nvidia-uvm.ko"
require_file "$NVIDIA_KO_DIR/kernel-open/Module.symvers" "patched Module.symvers"

if [[ "$(basename "$LLAMA_CPP_BIN")" != "llama-bench" ]]; then
    die "this integration regression requires llama-bench; got $LLAMA_CPP_BIN"
fi

if [[ "${POLARIS_LLAMA_LOAD_MODULE:-0}" == "1" ]]; then
    note "loading kernel/polaris.ko"
    rmmod polaris 2>/dev/null || true
    insmod "$ROOT_DIR/kernel/polaris.ko"
    chmod 666 /dev/polaris 2>/dev/null || true
fi

[[ -r /dev/polaris && -w /dev/polaris ]] ||
    die "/dev/polaris is not accessible; load polaris.ko and run as root or adjust device permissions"
[[ -r /dev/nvidiactl && -w /dev/nvidiactl ]] || die "/dev/nvidiactl is not accessible"
[[ -r /dev/nvidia-uvm && -w /dev/nvidia-uvm ]] || die "/dev/nvidia-uvm is not accessible"
[[ -r "$STATS_PATH" ]] || die "$STATS_PATH is not readable"

if [[ -r /proc/kallsyms ]]; then
    grep -q 'uvm_polaris_map_external_allocation' /proc/kallsyms ||
        die "loaded nvidia-uvm.ko does not expose uvm_polaris_map_external_allocation"
    grep -q 'uvm_polaris_register_hook' /proc/kallsyms ||
        die "loaded nvidia-uvm.ko does not expose uvm_polaris_register_hook"
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/polaris-llama-e2e.XXXXXX")"
device_log="$tmpdir/list-devices.log"
polaris_log="$tmpdir/llama-polaris-backend.log"
shim_log="$tmpdir/llama-shim-probe.log"
stats_before="$tmpdir/stats.before"
stats_after="$tmpdir/stats.after"
cp "$STATS_PATH" "$stats_before"

base_args=(
    -m "$LLAMA_CPP_MODEL"
    -p "${LLAMA_CPP_PROMPT_TOKENS:-32}"
    -n "${LLAMA_CPP_GEN_TOKENS:-4}"
    -r "${LLAMA_CPP_REPETITIONS:-1}"
    --no-warmup
    -ngl "${LLAMA_CPP_GPU_LAYERS:-1}"
    -fa 0
    -o json
)

if [[ -n "${LLAMA_CPP_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=( ${LLAMA_CPP_EXTRA_ARGS} )
    base_args+=( "${extra_args[@]}" )
fi

note "llama.cpp binary: $LLAMA_CPP_BIN"
note "model: $LLAMA_CPP_MODEL"
note "logs: $tmpdir"

if [[ "${POLARIS_LLAMA_RUN_POLARIS_BACKEND:-1}" != "0" ]]; then
    note "checking llama.cpp device list"
    "$LLAMA_CPP_BIN" --list-devices >"$device_log" 2>&1 ||
        die "llama.cpp --list-devices failed; see $device_log"
    grep -q "${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}:" "$device_log" ||
        die "llama.cpp binary did not list ${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}; see $device_log"

    note "running real llama.cpp workload on ${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}"
    "$LLAMA_CPP_BIN" "${base_args[@]}" -dev "${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}" \
        >"$polaris_log" 2>&1 ||
        { tail -n 120 "$polaris_log" >&2 || true; die "llama.cpp POLARIS backend run failed; see $polaris_log"; }
    grep -q "\"devices\": \"${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}\"" "$polaris_log" ||
        die "llama.cpp POLARIS backend output did not report devices=${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}; see $polaris_log"
    note "PASS: real llama.cpp workload completed on ${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}"
fi

if [[ "${POLARIS_LLAMA_RUN_SHIM_PROBE:-1}" == "0" ]]; then
    note "shim probe disabled"
    note "logs kept in $tmpdir"
    exit 0
fi

before_hook_calls="$(stat_value uvm_hook_calls)"
before_handled="$(stat_value uvm_handled)"
before_no_pte="$(stat_value uvm_no_pte)"
before_errors="$(stat_value uvm_errors)"

note "running LD_PRELOAD shim probe on ${LLAMA_CPP_SHIM_DEVICE:-CUDA0}"
run_shim_probe() {
    env \
        POLARIS_SHIM_BOOTSTRAP_RM_UVM=1 \
        POLARIS_SHIM_STATIC_RM_BACKEND=1 \
        POLARIS_SHIM_STRICT_MANAGED_ALLOC=1 \
        POLARIS_SHIM_REPORT_STATS=1 \
        POLARIS_SHIM_TRANSIENT_GPU=1 \
        POLARIS_SHIM_GPU_ID="${POLARIS_SHIM_GPU_ID:-0}" \
        POLARIS_SHIM_CUDA_ORDINAL="${POLARIS_SHIM_CUDA_ORDINAL:-0}" \
        POLARIS_SHIM_BLOCK_SIZE="${POLARIS_SHIM_BLOCK_SIZE:-0x200000}" \
        POLARIS_SHIM_MANAGED_LENGTH_CAP="${POLARIS_SHIM_MANAGED_LENGTH_CAP:-17179869184}" \
        POLARIS_SHIM_MIN_MANAGED_ALLOC="${POLARIS_SHIM_MIN_MANAGED_ALLOC:-1048576}" \
        POLARIS_SHIM_MAX_MANAGED_ALLOC="${POLARIS_SHIM_MAX_MANAGED_ALLOC:-0}" \
        LD_PRELOAD="$SHIM_SO${LD_PRELOAD:+:$LD_PRELOAD}" \
        "$LLAMA_CPP_BIN" "${base_args[@]}" -dev "${LLAMA_CPP_SHIM_DEVICE:-CUDA0}"
}

set +e
run_shim_probe >"$shim_log" 2>&1
rc=$?
set -e

cp "$STATS_PATH" "$stats_after"

grep -q '\[polaris-shim\] RM/UVM bootstrap ready' "$shim_log" ||
    die "shim did not bootstrap RM/UVM VA-space; see $shim_log"
grep -q '\[polaris-shim\] registered VA-space' "$shim_log" ||
    die "shim did not register the VA-space with polaris.ko; see $shim_log"
grep -q '\[polaris-shim\] static RM backend block' "$shim_log" ||
    die "no shim-managed allocation registered static RM backing; see $shim_log"
grep -q '\[polaris-shim\] managed allocation' "$shim_log" ||
    die "no llama.cpp allocation was routed through Polaris; see $shim_log"

after_hook_calls="$(stat_value uvm_hook_calls)"
after_handled="$(stat_value uvm_handled)"
after_no_pte="$(stat_value uvm_no_pte)"
after_errors="$(stat_value uvm_errors)"

if [[ "$rc" -ne 0 ]]; then
    if [[ "${POLARIS_LLAMA_STRICT_SHIM_FAULT_PASS:-0}" == "1" ]]; then
        tail -n 160 "$shim_log" >&2 || true
        die "strict shim fault-path run failed with exit code $rc; see $shim_log"
    fi

    if grep -Eq '\[polaris-shim\] (cudaMemcpy|cuMemcpy).*Polaris pointer is not wired yet' "$shim_log" &&
       ! grep -Eq 'Segmentation fault|SIGSEGV' "$shim_log"; then
        note "PASS: shim probe reached the current llama.cpp host-copy blocker cleanly"
        note "blocker: llama.cpp uploads model tensors with cudaMemcpyAsync into the first intercepted Polaris allocation"
        note "uvm_hook_calls: $before_hook_calls -> $after_hook_calls"
        note "uvm_handled:    $before_handled -> $after_handled"
        note "logs kept in $tmpdir"
        exit 0
    fi

    tail -n 160 "$shim_log" >&2 || true
    die "shim probe failed before the known guarded host-copy blocker (exit code $rc); see $shim_log"
fi

managed_success="$(extract_last_metric managed_success_calls "$shim_log")"
strict_failures="$(extract_last_metric strict_failure_calls "$shim_log")"
managed_success="${managed_success:-0}"
strict_failures="${strict_failures:-0}"
if [[ "$managed_success" -le 0 ]]; then
    die "shim stats reported no successful managed allocations; see $shim_log"
fi
if [[ "$strict_failures" -ne 0 ]]; then
    die "shim stats reported strict allocation failures=$strict_failures; see $shim_log"
fi

if [[ "$after_hook_calls" -le "$before_hook_calls" ]]; then
    die "kernel UVM hook call counter did not increase ($before_hook_calls -> $after_hook_calls)"
fi
if [[ "$after_handled" -le "$before_handled" ]]; then
    die "kernel UVM handled counter did not increase ($before_handled -> $after_handled)"
fi
if [[ "$after_no_pte" -gt "$before_no_pte" ]]; then
    die "kernel reported unserviceable Polaris faults ($before_no_pte -> $after_no_pte)"
fi
if [[ "$after_errors" -gt "$before_errors" ]]; then
    die "kernel reported Polaris UVM hook errors ($before_errors -> $after_errors)"
fi

note "PASS: llama.cpp shim allocation and GPU fault path reached Polaris"
note "managed_success_calls=$managed_success"
note "uvm_hook_calls: $before_hook_calls -> $after_hook_calls"
note "uvm_handled:    $before_handled -> $after_handled"
note "logs kept in $tmpdir"
