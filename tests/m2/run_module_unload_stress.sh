#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# M6 module unload/reload stress for the v4 worker-registration path.
# The gate deliberately keeps a real RM/UVM-registered Polaris worker fd open,
# verifies normal rmmod is refused while that fd pins polaris.ko, then verifies
# the module can unload/reload cleanly after the worker exits. A final
# daemon-backed RM spill/reload roundtrip proves the reloaded module still
# services the production-shaped path.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NVIDIA_KO_DIR="${NVIDIA_KO_DIR:-/home/wano/workspace/open-gpu-kernel-modules}"
NVIDIA_UVM_KO="${NVIDIA_UVM_KO:-$NVIDIA_KO_DIR/kernel-open/nvidia-uvm.ko}"
POLARIS_KO="${POLARIS_KO:-$ROOT_DIR/kernel/polaris.ko}"
M2_BIN="${M2_BIN:-$ROOT_DIR/tests/m2/m2_static_block_setup}"
POLARISD_BIN="${POLARISD_BIN:-$ROOT_DIR/target/debug/polarisd}"
STATS_PATH="${STATS_PATH:-/sys/kernel/polaris/stats}"

holder_pid=""
polarisd_pid=""
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/polaris-unload-stress.XXXXXX")"
holder_log="$tmpdir/holder.log"
polarisd_log="$tmpdir/polarisd.log"
ready_path="$tmpdir/holder.ready"
rmmod_log="$tmpdir/rmmod-live-worker.log"

note() {
    echo "==> $*" >&2
}

die() {
    echo "error: $*" >&2
    exit 1
}

stat_value() {
    local key="$1"
    awk -v key="${key}:" '$1 == key { print $2; found = 1 } END { if (!found) print "0" }' "$STATS_PATH"
}

cleanup_processes() {
    if [[ -n "${holder_pid:-}" ]] && kill -0 "$holder_pid" 2>/dev/null; then
        kill "$holder_pid" 2>/dev/null || true
        wait "$holder_pid" 2>/dev/null || true
    fi
    if [[ -n "${polarisd_pid:-}" ]] && kill -0 "$polarisd_pid" 2>/dev/null; then
        sudo kill "$polarisd_pid" 2>/dev/null || true
        wait "$polarisd_pid" 2>/dev/null || true
    fi
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

cleanup() {
    cleanup_processes
    wait_for_idle >/dev/null 2>&1 || true
    rm -rf "$tmpdir"
}
trap cleanup EXIT

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

reload_module() {
    cleanup_processes
    wait_for_idle || die "/dev/polaris stayed busy before module reload"
    sudo rmmod polaris 2>/dev/null || true
    verify_patched_uvm
    sudo insmod "$POLARIS_KO"
    sudo chmod 666 /dev/polaris /dev/nvidiactl /dev/nvidia-uvm 2>/dev/null || true
    [[ -r "$STATS_PATH" ]] || die "$STATS_PATH not readable after insmod"
}

load_patched_modules() {
    cleanup_processes
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

assert_clean_kernel_state() {
    local label="$1"
    wait_for_stat_eq sessions 0 "$label cleanup"
    wait_for_stat_eq blocks 0 "$label cleanup"
    wait_for_stat_eq pending_decs 0 "$label cleanup"
    wait_for_stat_eq static_blocks 0 "$label cleanup"
    wait_for_stat_eq block_mappings 0 "$label cleanup"
    wait_for_stat_eq v4_va_spaces 0 "$label cleanup"
}

assert_no_gpu_accounting() {
    local label="$1"
    wait_for_stat_eq daemon 0 "$label daemon stop"
    wait_for_stat_eq gpus 0 "$label GPU cleanup"
    wait_for_stat_eq gpu_total_mib 0 "$label GPU total cleanup"
    wait_for_stat_eq gpu_budget_mib 0 "$label GPU budget cleanup"
    wait_for_stat_eq cpu_pool_mib 0 "$label CPU pool cleanup"
}

start_polarisd() {
    rm -f "$polarisd_log"
    sudo env POLARISD_RM_BACKING=1 "$POLARISD_BIN" >"$polarisd_log" 2>&1 &
    polarisd_pid=$!
    for _ in $(seq 1 200); do
        if ! kill -0 "$polarisd_pid" 2>/dev/null; then
            tail -n 160 "$polarisd_log" >&2 || true
            die "polarisd exited during startup"
        fi
        if [[ "$(stat_value daemon)" -ge 1 && "$(stat_value gpus)" -ge 1 ]]; then
            note "polarisd registered"
            return 0
        fi
        sleep 0.05
    done
    tail -n 160 "$polarisd_log" >&2 || true
    die "timeout waiting for polarisd registration"
}

stop_polarisd() {
    if [[ -n "${polarisd_pid:-}" ]] && kill -0 "$polarisd_pid" 2>/dev/null; then
        sudo kill "$polarisd_pid" 2>/dev/null || true
        wait "$polarisd_pid" 2>/dev/null || true
    fi
    polarisd_pid=""
}

require_file "$POLARIS_KO" "polaris.ko"
require_executable "$M2_BIN" "M2 harness"
require_executable "$POLARISD_BIN" "polarisd"
require_file "$NVIDIA_UVM_KO" "patched nvidia-uvm.ko"
require_file "$NVIDIA_KO_DIR/kernel-open/Module.symvers" "patched NVIDIA Module.symvers"

load_patched_modules

note "starting real RM/UVM registered worker holder"
rm -f "$ready_path" "$holder_log"
sudo env POLARIS_HOLD_READY_PATH="$ready_path" "$M2_BIN" --hold-registered-worker \
    >"$holder_log" 2>&1 &
holder_pid=$!
for _ in $(seq 1 200); do
    if [[ -f "$ready_path" ]]; then
        break
    fi
    if ! kill -0 "$holder_pid" 2>/dev/null; then
        tail -n 160 "$holder_log" >&2 || true
        die "registered worker holder exited before ready"
    fi
    sleep 0.05
done
[[ -f "$ready_path" ]] || {
    tail -n 160 "$holder_log" >&2 || true
    die "timeout waiting for registered worker holder"
}
wait_for_stat_eq v4_va_spaces 1 "registered worker holder"
wait_for_stat_eq block_mappings 1 "registered worker holder"
wait_for_stat_eq static_blocks 0 "static RM guard"

note "checking rmmod is refused while registered worker fd is live"
set +e
sudo rmmod polaris >"$rmmod_log" 2>&1
rmmod_rc=$?
set -e
if [[ "$rmmod_rc" -eq 0 ]]; then
    cat "$rmmod_log" >&2 || true
    die "rmmod unexpectedly succeeded while registered worker was live"
fi
cat "$rmmod_log"

note "stopping registered worker and verifying fd-close cleanup"
kill "$holder_pid" 2>/dev/null || true
wait "$holder_pid"
holder_pid=""
wait_for_stat_eq v4_va_spaces 0 "registered worker cleanup"
wait_for_stat_eq block_mappings 0 "registered worker cleanup"
assert_no_gpu_accounting "registered worker cleanup"

note "unloading module after worker cleanup"
sudo rmmod polaris
if [[ -e "$STATS_PATH" ]]; then
    die "$STATS_PATH still exists after rmmod"
fi

note "reloading module after unload"
sudo insmod "$POLARIS_KO"
sudo chmod 666 /dev/polaris /dev/nvidiactl /dev/nvidia-uvm 2>/dev/null || true
[[ -r "$STATS_PATH" ]] || die "$STATS_PATH not readable after reload"
assert_clean_kernel_state "post-reload"
assert_no_gpu_accounting "post-reload"

note "running daemon-backed RM spill/reload on reloaded module"
start_polarisd
sudo "$M2_BIN" --daemon-rm-spill-reload-roundtrip
stop_polarisd
assert_clean_kernel_state "post-regression"
assert_no_gpu_accounting "post-regression"

note "M6 module unload/reload stress passed"
