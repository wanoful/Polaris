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
#    route real llama.cpp CUDA KV-cache allocations through daemon-backed
#    Polaris logical blocks, service GPU replayable faults through polarisd
#    RM backing, and clean up before exit.
#
# Set POLARIS_LLAMA_STRICT_SHIM_FAULT_PASS=1 to require the shim probe to
# complete and increase both uvm_hook_calls and uvm_handled.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/wano/workspace/llama.cpp}"
NVIDIA_KO_DIR="${NVIDIA_KO_DIR:-/home/wano/workspace/open-gpu-kernel-modules}"
SHIM_SO="${SHIM_SO:-$ROOT_DIR/libpolaris-shim/libpolaris-shim.so}"
STATS_PATH="${STATS_PATH:-/sys/kernel/polaris/stats}"
POLARISD_BIN="${POLARISD_BIN:-$ROOT_DIR/target/debug/polarisd}"

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
    require_file "$ROOT_DIR/kernel/polaris.ko" "polaris.ko"
    note "loading patched nvidia-uvm.ko and kernel/polaris.ko"
    rmmod polaris 2>/dev/null || true
    rmmod nvidia_uvm 2>/dev/null || rmmod nvidia-uvm 2>/dev/null || true
    insmod "$NVIDIA_KO_DIR/kernel-open/nvidia-uvm.ko" uvm_enable_builtin_tests=1
    insmod "$ROOT_DIR/kernel/polaris.ko"
    chmod 666 /dev/polaris /dev/nvidiactl /dev/nvidia-uvm 2>/dev/null || true
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
managed_shim_log="$tmpdir/llama-shim-managed-probe.log"
dynamic_shim_log="$tmpdir/llama-shim-dynamic-window.log"
polarisd_log="$tmpdir/polarisd.log"
stats_before="$tmpdir/stats.before"
stats_after="$tmpdir/stats.after"
cp "$STATS_PATH" "$stats_before"

polarisd_pid=""

cleanup() {
    if [[ -n "$polarisd_pid" ]]; then
        kill "$polarisd_pid" 2>/dev/null || true
        wait "$polarisd_pid" 2>/dev/null || true
        polarisd_pid=""
    fi
}
trap cleanup EXIT

base_args=(
    -m "$LLAMA_CPP_MODEL"
    -p "${LLAMA_CPP_PROMPT_TOKENS:-32}"
    -n "${LLAMA_CPP_GEN_TOKENS:-4}"
    -r "${LLAMA_CPP_REPETITIONS:-1}"
    --no-warmup
    -ngl "${LLAMA_CPP_GPU_LAYERS:-999}"
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

if [[ "${POLARIS_LLAMA_RUN_POLARIS_BACKEND:-auto}" != "0" ]]; then
    note "checking llama.cpp device list"
    "$LLAMA_CPP_BIN" --list-devices >"$device_log" 2>&1 ||
        die "llama.cpp --list-devices failed; see $device_log"
    if grep -q "${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}:" "$device_log"; then
        note "running real llama.cpp workload on ${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}"
        "$LLAMA_CPP_BIN" "${base_args[@]}" -dev "${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}" \
            >"$polaris_log" 2>&1 ||
            { tail -n 120 "$polaris_log" >&2 || true; die "llama.cpp POLARIS backend run failed; see $polaris_log"; }
        grep -q "\"devices\": \"${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}\"" "$polaris_log" ||
            die "llama.cpp POLARIS backend output did not report devices=${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}; see $polaris_log"
        note "PASS: real llama.cpp workload completed on ${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}"
    elif [[ "${POLARIS_LLAMA_RUN_POLARIS_BACKEND:-auto}" == "1" ]]; then
        die "llama.cpp binary did not list ${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0}; see $device_log"
    else
        note "skipping POLARIS backend run; ${LLAMA_CPP_POLARIS_DEVICE:-POLARIS0} not listed by this llama.cpp binary"
    fi
fi

if [[ "${POLARIS_LLAMA_RUN_SHIM_PROBE:-1}" == "0" ]]; then
    note "shim probe disabled"
    note "logs kept in $tmpdir"
    exit 0
fi

if [[ "${POLARIS_LLAMA_START_POLARISD:-1}" == "1" ]]; then
    require_executable "$POLARISD_BIN" "polarisd binary"
    note "starting polarisd with daemon-owned RM backing"
    POLARISD_RM_BACKING=1 "$POLARISD_BIN" >"$polarisd_log" 2>&1 &
    polarisd_pid=$!
    for _ in $(seq 1 200); do
        if ! kill -0 "$polarisd_pid" 2>/dev/null; then
            tail -n 160 "$polarisd_log" >&2 || true
            die "polarisd exited during startup; see $polarisd_log"
        fi
        if [[ "$(stat_value daemon)" -ge 1 && "$(stat_value gpus)" -ge 1 ]]; then
            break
        fi
        sleep 0.05
    done
    if [[ "$(stat_value daemon)" -lt 1 || "$(stat_value gpus)" -lt 1 ]]; then
        tail -n 160 "$polarisd_log" >&2 || true
        die "polarisd did not register with polaris.ko; see $polarisd_log"
    fi
fi

run_shim_probe() {
    local unified_memory="$1"
    local dynamic_window="${2:-0}"
    local probe_env=(
        GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}"
        GGML_CUDA_PDL="${GGML_CUDA_PDL:-0}"
        POLARIS_SHIM_BOOTSTRAP_RM_UVM=1
        POLARIS_SHIM_STRICT_MANAGED_ALLOC=1
        POLARIS_SHIM_REPORT_STATS=1
        POLARIS_SHIM_REQUIRE_KV_SCOPE="${POLARIS_SHIM_REQUIRE_KV_SCOPE:-1}"
        POLARIS_SHIM_ALLOW_ZERO_MEMSET="${POLARIS_SHIM_ALLOW_ZERO_MEMSET:-1}"
        POLARIS_SHIM_TRANSIENT_GPU="${POLARIS_SHIM_TRANSIENT_GPU:-0}"
        POLARIS_SHIM_GPU_ID="${POLARIS_SHIM_GPU_ID:-0}"
        POLARIS_SHIM_CUDA_ORDINAL="${POLARIS_SHIM_CUDA_ORDINAL:-0}"
        POLARIS_SHIM_BLOCK_SIZE="${POLARIS_SHIM_BLOCK_SIZE:-0x200000}"
        POLARIS_SHIM_MANAGED_LENGTH_CAP="${POLARIS_SHIM_MANAGED_LENGTH_CAP:-17179869184}"
        POLARIS_SHIM_MIN_MANAGED_ALLOC="${POLARIS_SHIM_MIN_MANAGED_ALLOC:-1}"
        POLARIS_SHIM_MAX_MANAGED_ALLOC="${POLARIS_SHIM_MAX_MANAGED_ALLOC:-0}"
        LD_PRELOAD="$SHIM_SO${LD_PRELOAD:+:$LD_PRELOAD}"
    )

    if [[ "$dynamic_window" == "1" ]]; then
        probe_env+=(
            POLARIS_SHIM_MANAGED_BLOCKS="${POLARIS_LLAMA_DYNAMIC_MANAGED_BLOCKS:-256}"
            POLARIS_SHIM_MANAGED_INITIAL_BLOCKS="${POLARIS_LLAMA_DYNAMIC_INITIAL_BLOCKS:-1}"
            POLARIS_SHIM_MANAGED_GROW_BLOCKS="${POLARIS_LLAMA_DYNAMIC_GROW_BLOCKS:-1}"
        )
    fi

    if [[ "$unified_memory" == "1" ]]; then
        probe_env+=(GGML_CUDA_ENABLE_UNIFIED_MEMORY=1)
    fi

    env "${probe_env[@]}" \
        "$LLAMA_CPP_BIN" "${base_args[@]}" -dev "${LLAMA_CPP_SHIM_DEVICE:-CUDA0}"
}

verify_shim_probe() {
    local label="$1"
    local log="$2"
    local before_hook="$3"
    local before_handled="$4"
    local before_no_pte="$5"
    local before_errors="$6"
    local expected_selected_key="$7"
    local rc="$8"
    local before_blocks="$9"
    local require_dynamic="${10:-0}"
    local after_hook
    local after_handled
    local after_no_pte
    local after_errors
    local managed_success
    local strict_failures
    local expected_selected
    local grow_calls
    local shrink_calls
    local blocks_before
    local blocks_after

    grep -q '\[polaris-shim\] RM/UVM bootstrap ready' "$log" ||
        die "$label shim did not bootstrap RM/UVM VA-space; see $log"
    grep -q '\[polaris-shim\] registered VA-space' "$log" ||
        die "$label shim did not register the VA-space with polaris.ko; see $log"
    if grep -q '\[polaris-shim\] static RM backend block' "$log"; then
        die "$label unexpectedly used shim static RM backing; see $log"
    fi
    grep -q '\[polaris-shim\] managed allocation' "$log" ||
        die "$label no llama.cpp allocation was routed through Polaris; see $log"
    grep -q 'polarisd: RM ALLOC block' "$polarisd_log" ||
        die "$label polarisd did not publish daemon-owned RM backing; see $polarisd_log"

    after_hook="$(stat_value uvm_hook_calls)"
    after_handled="$(stat_value uvm_handled)"
    after_no_pte="$(stat_value uvm_no_pte)"
    after_errors="$(stat_value uvm_errors)"

    if [[ "$rc" -ne 0 ]]; then
        if [[ "${POLARIS_LLAMA_STRICT_SHIM_FAULT_PASS:-0}" == "1" ]]; then
            tail -n 160 "$log" >&2 || true
            die "$label strict shim fault-path run failed with exit code $rc; see $log"
        fi

        if grep -Eq '\[polaris-shim\] (cudaMemcpy|cuMemcpy).*Polaris pointer is not wired yet' "$log" &&
           ! grep -Eq 'Segmentation fault|SIGSEGV' "$log"; then
            tail -n 160 "$log" >&2 || true
            note "PASS: $label shim selected a copied buffer and stopped at the guarded host-copy surface"
            note "exit code: $rc"
            note "uvm_hook_calls: $before_hook -> $after_hook"
            note "uvm_handled:    $before_handled -> $after_handled"
            note "logs kept in $tmpdir"
            exit 0
        fi

        tail -n 160 "$log" >&2 || true
        die "$label shim probe failed after bootstrap but before the strict fault-path gate (exit code $rc); see $log"
    fi

    managed_success="$(extract_last_metric managed_success_calls "$log")"
    strict_failures="$(extract_last_metric strict_failure_calls "$log")"
    expected_selected="$(extract_last_metric "$expected_selected_key" "$log")"
    grow_calls="$(extract_last_metric managed_window_grow_calls "$log")"
    shrink_calls="$(extract_last_metric managed_window_shrink_calls "$log")"
    managed_success="${managed_success:-0}"
    strict_failures="${strict_failures:-0}"
    expected_selected="${expected_selected:-0}"
    grow_calls="${grow_calls:-0}"
    shrink_calls="${shrink_calls:-0}"
    if [[ "$managed_success" -le 0 ]]; then
        die "$label shim stats reported no successful managed allocations; see $log"
    fi
    if [[ "$strict_failures" -ne 0 ]]; then
        die "$label shim stats reported strict allocation failures=$strict_failures; see $log"
    fi
    if [[ "$expected_selected" -le 0 ]]; then
        die "$label shim stats reported ${expected_selected_key}=0; see $log"
    fi
    if [[ "$require_dynamic" == "1" ]]; then
        grep -q '\[polaris-shim\] grew registered VA-space window' "$log" ||
            die "$label did not grow the registered v4 fault window; see $log"
        grep -q '\[polaris-shim\] shrank registered VA-space window' "$log" ||
            die "$label did not shrink the registered v4 fault window; see $log"
        if [[ "$grow_calls" -le 0 || "$shrink_calls" -le 0 ]]; then
            die "$label dynamic-window stats missing grow/shrink calls grow=$grow_calls shrink=$shrink_calls; see $log"
        fi
    fi

    if [[ "$after_hook" -le "$before_hook" ]]; then
        die "$label kernel UVM hook call counter did not increase ($before_hook -> $after_hook)"
    fi
    if [[ "$after_handled" -le "$before_handled" ]]; then
        die "$label kernel UVM handled counter did not increase ($before_handled -> $after_handled)"
    fi
    if [[ "$after_no_pte" -gt "$before_no_pte" ]]; then
        die "$label kernel reported unserviceable Polaris faults ($before_no_pte -> $after_no_pte)"
    fi
    if [[ "$after_errors" -gt "$before_errors" ]]; then
        die "$label kernel reported Polaris UVM hook errors ($before_errors -> $after_errors)"
    fi
    blocks_before="$before_blocks"
    for _ in $(seq 1 100); do
        blocks_after="$(stat_value blocks)"
        if [[ "$blocks_after" -le "$blocks_before" ]]; then
            break
        fi
        sleep 0.05
    done
    if [[ "$blocks_after" -gt "$blocks_before" ]]; then
        die "$label leaked Polaris blocks ($blocks_before -> $blocks_after)"
    fi

    note "PASS: $label llama.cpp shim allocation and GPU fault path reached Polaris"
    note "managed_success_calls=$managed_success"
    note "$expected_selected_key=$expected_selected"
    if [[ "$grow_calls" -gt 0 || "$shrink_calls" -gt 0 ]]; then
        note "managed_window_grow_calls=$grow_calls"
        note "managed_window_shrink_calls=$shrink_calls"
    fi
    note "uvm_hook_calls: $before_hook -> $after_hook"
    note "uvm_handled:    $before_handled -> $after_handled"
}

before_hook_calls="$(stat_value uvm_hook_calls)"
before_handled="$(stat_value uvm_handled)"
before_no_pte="$(stat_value uvm_no_pte)"
before_errors="$(stat_value uvm_errors)"
before_blocks="$(stat_value blocks)"

note "running LD_PRELOAD shim probe on ${LLAMA_CPP_SHIM_DEVICE:-CUDA0}"
set +e
run_shim_probe "" "0" >"$shim_log" 2>&1
rc=$?
set -e
verify_shim_probe "default" "$shim_log" "$before_hook_calls" "$before_handled" \
    "$before_no_pte" "$before_errors" api_runtime_alloc_selected "$rc" "$before_blocks" 0

before_hook_calls="$(stat_value uvm_hook_calls)"
before_handled="$(stat_value uvm_handled)"
before_no_pte="$(stat_value uvm_no_pte)"
before_errors="$(stat_value uvm_errors)"
before_blocks="$(stat_value blocks)"

note "running LD_PRELOAD unified-memory shim probe on ${LLAMA_CPP_SHIM_DEVICE:-CUDA0}"
set +e
run_shim_probe "1" "0" >"$managed_shim_log" 2>&1
rc=$?
set -e
verify_shim_probe "unified-memory" "$managed_shim_log" "$before_hook_calls" "$before_handled" \
    "$before_no_pte" "$before_errors" api_runtime_managed_alloc_selected "$rc" "$before_blocks" 0

if [[ "${POLARIS_LLAMA_RUN_DYNAMIC_WINDOW_PROBE:-0}" == "1" ]]; then
    before_hook_calls="$(stat_value uvm_hook_calls)"
    before_handled="$(stat_value uvm_handled)"
    before_no_pte="$(stat_value uvm_no_pte)"
    before_errors="$(stat_value uvm_errors)"
    before_blocks="$(stat_value blocks)"

    note "running LD_PRELOAD dynamic-window shim probe on ${LLAMA_CPP_SHIM_DEVICE:-CUDA0}"
    set +e
    run_shim_probe "" "1" >"$dynamic_shim_log" 2>&1
    rc=$?
    set -e
    verify_shim_probe "dynamic-window" "$dynamic_shim_log" "$before_hook_calls" "$before_handled" \
        "$before_no_pte" "$before_errors" api_runtime_alloc_selected "$rc" "$before_blocks" 1
fi

cp "$STATS_PATH" "$stats_after"
note "logs kept in $tmpdir"
