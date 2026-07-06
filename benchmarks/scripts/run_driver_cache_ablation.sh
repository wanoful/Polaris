#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# driver_cache ablation wrapper.
#
# Runs the existing llama.cpp/POLARIS KV benchmark twice: once with the
# in-kernel "already mapped to this gpu_va_space" short-circuit enabled
# (default), once with it bypassed via /sys/kernel/polaris/driver_cache=0.
# Emits a side-by-side summary so the contribution of that short-circuit
# is measurable rather than just counted.
#
# Requires polaris.ko built from the tree that exposes the
# /sys/kernel/polaris/driver_cache attribute (see kernel/polaris.rs).
#
# Defaults run the short 512x128 SmolLM2 path; override via env if you want
# Qwen14B 16k. Keep prompt small for the OFF arm — it will be slow.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVER_CACHE_PATH="${DRIVER_CACHE_PATH:-/sys/kernel/polaris/driver_cache}"
RESULTS_ROOT="${POLARIS_BENCH_RESULTS_DIR:-$ROOT_DIR/benchmarks/results/llama_cpp}"
RUN_STAMP="${POLARIS_BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ABLATION_DIR="$RESULTS_ROOT/driver-cache-ablation-$RUN_STAMP"
SUMMARY_MD="$ABLATION_DIR/ablation_summary.md"

die() { echo "error: $*" >&2; exit 1; }
note() { echo "==> $*" >&2; }

[[ -w "$DRIVER_CACHE_PATH" ]] || die "$DRIVER_CACHE_PATH is not writable (need polaris.ko with driver_cache attr; run as root)"

mkdir -p "$ABLATION_DIR"

set_cache() {
    local val="$1"
    echo "$val" >"$DRIVER_CACHE_PATH" || die "failed to set $DRIVER_CACHE_PATH=$val"
    local got
    got="$(cat "$DRIVER_CACHE_PATH")"
    [[ "$got" == "$val" ]] || die "expected driver_cache=$val, got $got"
    note "driver_cache = $val (verified)"
}

# Allow caller to override the underlying bench parameters; defaults match
# the existing 512x128 driver-cache final report.
export POLARIS_BENCH_PROMPTS="${POLARIS_BENCH_PROMPTS:-512}"
export POLARIS_BENCH_GENS="${POLARIS_BENCH_GENS:-128}"
export POLARIS_BENCH_REPETITIONS="${POLARIS_BENCH_REPETITIONS:-3}"
export POLARIS_BENCH_MODES="${POLARIS_BENCH_MODES:-native_cuda,polaris_no_pressure,polaris_pressure}"

restore_on_exit() {
    set_cache 1 || true
}
trap restore_on_exit EXIT

run_arm() {
    local arm="$1"
    local cache_val="$2"
    local out_dir="$ABLATION_DIR/$arm"
    note "=== arm: $arm (driver_cache=$cache_val) ==="
    set_cache "$cache_val"
    POLARIS_BENCH_OUT_DIR="$out_dir" \
    POLARIS_BENCH_RUN_ID="$RUN_STAMP-$arm" \
        bash "$ROOT_DIR/benchmarks/scripts/run_llama_kv_bench.sh"
    [[ -f "$out_dir/runs.jsonl" ]] || die "$arm run did not produce runs.jsonl"
}

run_arm "cache_on"  1
run_arm "cache_off" 0

note "generating side-by-side ablation summary"
python3 - "$ABLATION_DIR/cache_on/runs.jsonl" "$ABLATION_DIR/cache_off/runs.jsonl" "$SUMMARY_MD" <<'PY'
import json
import sys
from pathlib import Path

on_path, off_path, out_path = sys.argv[1:]

def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows

def key(r):
    w = r.get("workload", {})
    return (r.get("mode"), w.get("prompt_tokens"), w.get("gen_tokens"))

def metric(r, k):
    delta = r.get("stats_delta", {}) or {}
    return delta.get(k, 0)

on_rows  = {key(r): r for r in load(on_path)  if r.get("source") == "polaris"}
off_rows = {key(r): r for r in load(off_path) if r.get("source") == "polaris"}

cols = [
    ("avg_ts",                "llama_avg_ts"),
    ("bridge_maps",           ("delta", "uvm_bridge_map_calls")),
    ("cached_map_hits",       ("delta", "uvm_cached_map_hits")),
    ("hook_calls",            ("delta", "uvm_hook_calls")),
    ("offloads",              ("delta", "offloads")),
    ("reloads",               ("delta", "reloads")),
    ("uvm_bridge_map_avg_ns", ("delta", "uvm_bridge_map_avg_ns")),
    ("uvm_errors",            ("delta", "uvm_errors")),
]

def get(r, spec):
    if spec == "llama_avg_ts":
        v = r.get("llama_avg_ts")
        return v
    if isinstance(spec, tuple):
        bucket, k = spec
        return (r.get(f"stats_{bucket}") or {}).get(k, 0)
    return None

def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)

lines = ["# driver_cache ablation", ""]
lines.append("| mode | prompt | gen | metric | cache_on | cache_off | delta |")
lines.append("|---|---:|---:|---|---:|---:|---:|")

for k in sorted(set(on_rows) | set(off_rows)):
    on  = on_rows.get(k, {})
    off = off_rows.get(k, {})
    mode, p, g = k
    for label, spec in cols:
        v_on  = get(on,  spec)
        v_off = get(off, spec)
        if v_on is None and v_off is None:
            continue
        try:
            d = (v_off or 0) - (v_on or 0)
        except TypeError:
            d = ""
        lines.append(f"| {mode} | {p} | {g} | {label} | {fmt(v_on)} | {fmt(v_off)} | {fmt(d)} |")
    lines.append("")

Path(out_path).write_text("\n".join(lines) + "\n")
print(f"wrote {out_path}")
PY

note "ablation summary: $SUMMARY_MD"
note "raw runs: $ABLATION_DIR/cache_on/ and $ABLATION_DIR/cache_off/"
