#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# M6 daemon-backed RM soak runner.
# This composes the focused daemon-backed M2/M6 gates into repeated rounds on
# real polaris.ko + real polarisd with POLARISD_RM_BACKING=1. Static RM
# registration is not used by any mode run here.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NVIDIA_KO_DIR="${NVIDIA_KO_DIR:-/home/wano/workspace/open-gpu-kernel-modules}"
NVIDIA_UVM_KO="${NVIDIA_UVM_KO:-$NVIDIA_KO_DIR/kernel-open/nvidia-uvm.ko}"
POLARIS_KO="${POLARIS_KO:-$ROOT_DIR/kernel/polaris.ko}"
M2_BIN="${M2_BIN:-$ROOT_DIR/tests/m2/m2_static_block_setup}"
POLARISD_BIN="${POLARISD_BIN:-$ROOT_DIR/target/debug/polarisd}"
STATS_PATH="${STATS_PATH:-/sys/kernel/polaris/stats}"
POLARIS_SOAK_ITERS="${POLARIS_SOAK_ITERS:-2}"
POLARIS_SOAK_NEAR_CAPACITY_ITERS="${POLARIS_SOAK_NEAR_CAPACITY_ITERS:-1}"
POLARIS_SOAK_MICROBENCH_ITERS="${POLARIS_SOAK_MICROBENCH_ITERS:-1}"
POLARIS_SOAK_RELOAD_MODULE="${POLARIS_SOAK_RELOAD_MODULE:-1}"
POLARIS_NEAR_CAPACITY_BUDGET_BYTES="${POLARIS_NEAR_CAPACITY_BUDGET_BYTES:-4194304}"
POLARIS_MICROBENCH_BUDGET_BYTES="${POLARIS_MICROBENCH_BUDGET_BYTES:-4194304}"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/polaris-daemon-rm-soak.XXXXXX")"
polarisd_pid=""
phase_index=0

note() {
    echo "==> $*" >&2
}

die() {
    echo "error: $*" >&2
    exit 1
}

stat_value() {
    local key="$1"
    if [[ ! -r "$STATS_PATH" ]]; then
        echo "0"
        return 0
    fi
    awk -F':[[:space:]]*' -v key="$key" '$1 == key { print $2; found = 1; exit } END { if (!found) print "0" }' "$STATS_PATH"
}

require_file() {
    local path="$1"
    local what="$2"
    [[ -f "$path" ]] || die "$what not found: $path"
}

require_executable() {
    local path="$1"
    local what="$2"
    [[ -f "$path" && -x "$path" ]] || die "$what is not executable: $path"
}

wait_for_idle() {
    for _ in $(seq 1 100); do
        if ! pgrep -x polarisd >/dev/null && ! sudo fuser /dev/polaris >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

stop_polarisd() {
    if [[ -n "${polarisd_pid:-}" ]] && kill -0 "$polarisd_pid" 2>/dev/null; then
        sudo kill "$polarisd_pid" 2>/dev/null || true
        wait "$polarisd_pid" 2>/dev/null || true
    fi
    polarisd_pid=""
}

stop_all_polarisd() {
    local pids

    stop_polarisd
    pids="$(pgrep -x polarisd || true)"
    if [[ -n "$pids" ]]; then
        sudo kill $pids 2>/dev/null || true
    fi
}

cleanup() {
    stop_all_polarisd
    wait_for_idle >/dev/null 2>&1 || true
    rm -rf "$tmpdir"
}
trap cleanup EXIT

reload_module() {
    stop_all_polarisd
    wait_for_idle || die "/dev/polaris stayed busy before module reload"
    sudo rmmod polaris 2>/dev/null || true
    verify_patched_uvm
    sudo insmod "$POLARIS_KO"
    sudo chmod 666 /dev/polaris /dev/nvidiactl /dev/nvidia-uvm 2>/dev/null || true
    [[ -r "$STATS_PATH" ]] || die "$STATS_PATH not readable after insmod"
}

load_patched_modules() {
    stop_all_polarisd
    wait_for_idle || die "/dev/polaris stayed busy before module reload"
    note "loading patched nvidia-uvm.ko and polaris.ko"
    sudo rmmod polaris 2>/dev/null || true
    sudo rmmod nvidia_uvm 2>/dev/null || sudo rmmod nvidia-uvm 2>/dev/null || true
    sudo insmod "$NVIDIA_UVM_KO" uvm_enable_builtin_tests=1
    verify_patched_uvm
    sudo insmod "$POLARIS_KO"
    sudo chmod 666 /dev/polaris /dev/nvidiactl /dev/nvidia-uvm 2>/dev/null || true
    [[ -r "$STATS_PATH" ]] || die "$STATS_PATH not readable after insmod"
}

verify_patched_uvm() {
    if [[ -r /proc/kallsyms ]]; then
        grep -q 'uvm_polaris_register_hook' /proc/kallsyms ||
            die "loaded nvidia-uvm.ko does not expose uvm_polaris_register_hook"
        grep -q 'uvm_polaris_map_external_allocation' /proc/kallsyms ||
            die "loaded nvidia-uvm.ko does not expose uvm_polaris_map_external_allocation"
        grep -q 'uvm_polaris_copy_external_allocation' /proc/kallsyms ||
            die "loaded nvidia-uvm.ko does not expose uvm_polaris_copy_external_allocation"
    fi
}

wait_for_stat_at_least() {
    local key="$1"
    local expected_min="$2"
    local label="$3"
    for _ in $(seq 1 200); do
        if [[ "$(stat_value "$key")" -ge "$expected_min" ]]; then
            note "$label: $key=$(stat_value "$key")"
            return 0
        fi
        sleep 0.05
    done
    sed -n '1,220p' "$STATS_PATH" >&2 || true
    die "timeout waiting for $label: $key >= $expected_min"
}

wait_for_stat_eq() {
    local key="$1"
    local expected="$2"
    local label="$3"
    for _ in $(seq 1 200); do
        if [[ "$(stat_value "$key")" == "$expected" ]]; then
            note "$label: $key=$expected"
            return 0
        fi
        sleep 0.05
    done
    sed -n '1,220p' "$STATS_PATH" >&2 || true
    die "timeout waiting for $label: $key=$expected"
}

start_polarisd() {
    local label="$1"
    local budget_bytes="${2:-}"
    local log="$tmpdir/polarisd-${label}.log"
    local env_args=(POLARISD_RM_BACKING=1)

    if [[ -n "$budget_bytes" ]]; then
        env_args+=(POLARISD_GPU_BUDGET_BYTES="$budget_bytes")
    fi

    rm -f "$log"
    sudo env "${env_args[@]}" "$POLARISD_BIN" >"$log" 2>&1 &
    polarisd_pid=$!
    for _ in $(seq 1 200); do
        if ! kill -0 "$polarisd_pid" 2>/dev/null; then
            tail -n 200 "$log" >&2 || true
            die "polarisd exited during $label startup"
        fi
        if [[ "$(stat_value daemon)" -ge 1 && "$(stat_value gpus)" -ge 1 ]]; then
            note "$label: polarisd registered"
            return 0
        fi
        sleep 0.05
    done
    tail -n 200 "$log" >&2 || true
    die "timeout waiting for polarisd registration during $label"
}

assert_clean_kernel_state() {
    local label="$1"
    wait_for_stat_eq sessions 0 "$label cleanup"
    wait_for_stat_eq blocks 0 "$label cleanup"
    wait_for_stat_eq pending_decs 0 "$label cleanup"
    wait_for_stat_eq static_blocks 0 "$label cleanup"
    wait_for_stat_eq block_mappings 0 "$label cleanup"
    wait_for_stat_eq v4_va_spaces 0 "$label cleanup"
    wait_for_stat_eq v4_worker_pids 0 "$label cleanup"
}

assert_no_gpu_accounting() {
    local label="$1"
    wait_for_stat_eq daemon 0 "$label daemon stop"
    wait_for_stat_eq gpus 0 "$label GPU cleanup"
    wait_for_stat_eq gpu_total_mib 0 "$label GPU total cleanup"
    wait_for_stat_eq gpu_budget_mib 0 "$label GPU budget cleanup"
    wait_for_stat_eq cpu_pool_mib 0 "$label CPU pool cleanup"
}

run_gate() {
    local label="$1"
    shift
    local errors_before
    local errors_after
    local bridge_before
    local bridge_after
    local rejected_before
    local rejected_after

    phase_index=$((phase_index + 1))
    errors_before="$(stat_value uvm_errors)"
    bridge_before="$(stat_value uvm_bridge_map_calls)"
    rejected_before="$(stat_value uvm_rejected)"
    note "phase $phase_index: $label"
    sudo "$M2_BIN" "$@"
    errors_after="$(stat_value uvm_errors)"
    bridge_after="$(stat_value uvm_bridge_map_calls)"
    rejected_after="$(stat_value uvm_rejected)"
    if [[ "$errors_after" != "$errors_before" ]]; then
        sed -n '1,220p' "$STATS_PATH" >&2 || true
        die "$label changed uvm_errors: $errors_before -> $errors_after"
    fi
    if [[ "$bridge_after" -le "$bridge_before" ]]; then
        sed -n '1,220p' "$STATS_PATH" >&2 || true
        die "$label did not increase uvm_bridge_map_calls: $bridge_before -> $bridge_after"
    fi
    if [[ "$rejected_after" -le "$rejected_before" ]]; then
        sed -n '1,220p' "$STATS_PATH" >&2 || true
        die "$label did not increase uvm_rejected: $rejected_before -> $rejected_after"
    fi
    assert_clean_kernel_state "$label"
}

require_file "$POLARIS_KO" "polaris.ko"
require_executable "$M2_BIN" "M2 harness"
require_executable "$POLARISD_BIN" "polarisd"
require_file "$NVIDIA_UVM_KO" "patched nvidia-uvm.ko"
require_file "$NVIDIA_KO_DIR/kernel-open/Module.symvers" "patched NVIDIA Module.symvers"

if [[ "$POLARIS_SOAK_ITERS" -lt 1 ]]; then
    die "POLARIS_SOAK_ITERS must be >= 1"
fi
if [[ "$POLARIS_SOAK_NEAR_CAPACITY_ITERS" -lt 0 ]]; then
    die "POLARIS_SOAK_NEAR_CAPACITY_ITERS must be >= 0"
fi
if [[ "$POLARIS_SOAK_MICROBENCH_ITERS" -lt 0 ]]; then
    die "POLARIS_SOAK_MICROBENCH_ITERS must be >= 0"
fi

if [[ "$POLARIS_SOAK_RELOAD_MODULE" != "0" ]]; then
    load_patched_modules
else
    verify_patched_uvm
fi

start_polarisd "normal"
for iter in $(seq 1 "$POLARIS_SOAK_ITERS"); do
    note "normal-budget soak iteration $iter/$POLARIS_SOAK_ITERS"
    run_gate "single-block spill/reload stress iter $iter" --daemon-rm-spill-reload-stress
    run_gate "multi-block stress iter $iter" --daemon-rm-multi-block-stress
    run_gate "dynamic fragmentation stress iter $iter" --daemon-rm-dynamic-fragmentation-stress
    run_gate "overwrite COW roundtrip iter $iter" --daemon-rm-cow-roundtrip
    run_gate "observed mapping key isolation iter $iter" --daemon-rm-observed-mapping-key-isolation
done
stop_polarisd
assert_no_gpu_accounting "normal-budget"

if [[ "$POLARIS_SOAK_MICROBENCH_ITERS" -gt 0 ]]; then
    note "reloading polaris.ko for single-worker microbench"
    reload_module
    start_polarisd "microbench" "$POLARIS_MICROBENCH_BUDGET_BYTES"
    wait_for_stat_at_least gpu_budget_mib 4 "microbench daemon budget"
    for iter in $(seq 1 "$POLARIS_SOAK_MICROBENCH_ITERS"); do
        note "single-worker microbench iteration $iter/$POLARIS_SOAK_MICROBENCH_ITERS"
        run_gate "single-worker microbench iter $iter" --daemon-rm-single-worker-microbench
    done
    stop_polarisd
    assert_no_gpu_accounting "microbench"
fi

if [[ "$POLARIS_SOAK_NEAR_CAPACITY_ITERS" -gt 0 ]]; then
    note "reloading polaris.ko for near-capacity budget soak"
    reload_module
    start_polarisd "near-capacity" "$POLARIS_NEAR_CAPACITY_BUDGET_BYTES"
    wait_for_stat_at_least gpu_budget_mib 4 "near-capacity daemon budget"
    for iter in $(seq 1 "$POLARIS_SOAK_NEAR_CAPACITY_ITERS"); do
        note "near-capacity soak iteration $iter/$POLARIS_SOAK_NEAR_CAPACITY_ITERS"
        run_gate "near-capacity soak iter $iter" --daemon-rm-near-capacity-soak
    done
    stop_polarisd
    assert_no_gpu_accounting "near-capacity"
fi

assert_clean_kernel_state "daemon RM soak final"
assert_no_gpu_accounting "daemon RM soak final"
wait_for_stat_at_least uvm_bridge_map_calls 1 "daemon RM soak bridge telemetry"
note "M6 daemon-backed RM soak passed"
