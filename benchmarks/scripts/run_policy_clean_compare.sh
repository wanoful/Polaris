#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# Clean-module policy comparison.
#
# Existing policy-compare report has a confounder: FIFO 114.97 tok/s came
# from a fresh module, the LRU/Phase rows came from a same-module replay,
# and the FIFO rerun on the same module collapsed to 69.73 tok/s. This
# wrapper removes that confounder by rmmod+insmod between every single
# benchmark invocation, then runs each policy N times for mean+stddev.
#
# Default workload mirrors the Qwen14B 16k pressure run at 2560 MiB budget
# (the existing policy-compare config). Override via env if you want a
# different shape.
#
# Requires root (sudo). Uses the existing run_llama_kv_bench.sh path so the
# runs.jsonl format stays compatible with compare_kv_results.py.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLARIS_KO="${POLARIS_KO:-$ROOT_DIR/kernel/polaris.ko}"
STATS_PATH="${STATS_PATH:-/sys/kernel/polaris/stats}"
RESULTS_ROOT="${POLARIS_BENCH_RESULTS_DIR:-$ROOT_DIR/benchmarks/results/llama_cpp}"
RUN_STAMP="${POLARIS_BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_DIR="$RESULTS_ROOT/policy-clean-compare-$RUN_STAMP"
SUMMARY_MD="$SWEEP_DIR/policy_clean_summary.md"

POLICIES_DEFAULT="fifo,lru,phase_aware,attention_stream"
IFS=',' read -r -a POLICIES <<<"${POLARIS_BENCH_POLICIES:-$POLICIES_DEFAULT}"
TRIALS="${POLARIS_BENCH_TRIALS:-3}"

die() { echo "error: $*" >&2; exit 1; }
note() { echo "==> $*" >&2; }

[[ -f "$POLARIS_KO" ]] || die "polaris.ko not found at $POLARIS_KO (set POLARIS_KO=)"
command -v sudo >/dev/null || die "sudo required for module reload"

# Workload defaults — Qwen14B 16k pressure at 2560 MiB budget, single rep
# per run (we get repetitions from outer trial loop, not -r).
export LLAMA_CPP_MODEL="${LLAMA_CPP_MODEL:-/home/wano/workspace/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf}"
export POLARIS_BENCH_PROMPTS="${POLARIS_BENCH_PROMPTS:-16384}"
export POLARIS_BENCH_GENS="${POLARIS_BENCH_GENS:-32}"
export POLARIS_BENCH_REPETITIONS="${POLARIS_BENCH_REPETITIONS:-1}"
export POLARIS_BENCH_MODES="${POLARIS_BENCH_MODES:-polaris_pressure}"
export POLARIS_BENCH_PRESSURE_BUDGET_BYTES="${POLARIS_BENCH_PRESSURE_BUDGET_BYTES:-2684354560}"  # 2560 MiB
export POLARIS_BENCH_CPU_POOL_BYTES="${POLARIS_BENCH_CPU_POOL_BYTES:-8589934592}"               # 8 GiB
export POLARIS_LLAMA_KV_HINTS="${POLARIS_LLAMA_KV_HINTS:-1}"

mkdir -p "$SWEEP_DIR"

wait_for_idle() {
    for _ in $(seq 1 200); do
        if ! fuser /dev/polaris >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.05
    done
    return 1
}

reload_module() {
    pkill -x polarisd 2>/dev/null || true
    wait_for_idle || die "/dev/polaris stayed busy before reload"
    sudo rmmod polaris 2>/dev/null || true
    sudo insmod "$POLARIS_KO"
    sudo chmod 666 /dev/polaris /dev/nvidiactl /dev/nvidia-uvm 2>/dev/null || true
    [[ -r "$STATS_PATH" ]] || die "$STATS_PATH not readable after insmod"
}

run_trial() {
    local policy="$1"
    local trial="$2"
    local out_dir="$SWEEP_DIR/$policy/trial-$trial"
    note "--- policy=$policy trial=$trial/$TRIALS (clean module) ---"
    reload_module
    POLARIS_BENCH_OUT_DIR="$out_dir" \
    POLARIS_BENCH_RUN_ID="$RUN_STAMP-$policy-t$trial" \
    POLARIS_BENCH_EVICTION_POLICY="$policy" \
        bash "$ROOT_DIR/benchmarks/scripts/run_llama_kv_bench.sh"
    [[ -f "$out_dir/runs.jsonl" ]] || die "policy=$policy trial=$trial produced no runs.jsonl"
}

for policy in "${POLICIES[@]}"; do
    for trial in $(seq 1 "$TRIALS"); do
        run_trial "$policy" "$trial"
    done
done

note "aggregating mean ± stddev"
python3 - "$SWEEP_DIR" "$SUMMARY_MD" "${POLICIES[*]}" "$TRIALS" <<'PY'
import json
import math
import sys
from pathlib import Path

sweep_dir = Path(sys.argv[1])
out_path  = Path(sys.argv[2])
policies  = sys.argv[3].split()
trials    = int(sys.argv[4])

KEYS = [
    ("avg_ts",       "llama_avg_ts"),
    ("offloads",     ("delta", "offloads")),
    ("reloads",      ("delta", "reloads")),
    ("bridge_maps",  ("delta", "uvm_bridge_map_calls")),
    ("cached_hits",  ("delta", "uvm_cached_map_hits")),
    ("acct_calls",   ("delta", "uvm_acct_calls")),
    ("acct_touches", ("delta", "uvm_acct_touches")),
    ("uvm_errors",   ("delta", "uvm_errors")),
]

def value(rec, spec):
    if spec == "llama_avg_ts":
        return rec.get("llama_avg_ts")
    if isinstance(spec, tuple):
        bucket, k = spec
        return (rec.get(f"stats_{bucket}") or {}).get(k, 0)
    return None

def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None)
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return (mu, 0.0)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return (mu, math.sqrt(var))

def fmt(mu, sd):
    if mu is None:
        return ""
    if isinstance(mu, float):
        return f"{mu:.2f} ± {sd:.2f}"
    return f"{mu} ± {sd:.2f}"

lines = [
    "# Clean-module policy comparison",
    "",
    f"Trials per policy: {trials}. Every trial uses a fresh `rmmod polaris && insmod polaris.ko`.",
    "",
    "| policy | metric | mean ± stddev | n |",
    "|---|---|---|---:|",
]

for policy in policies:
    samples = {label: [] for label, _ in KEYS}
    n_runs = 0
    for trial in range(1, trials + 1):
        rj = sweep_dir / policy / f"trial-{trial}" / "runs.jsonl"
        if not rj.exists():
            continue
        for line in rj.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("source") != "polaris":
                continue
            if rec.get("mode", "").startswith("native"):
                continue
            n_runs += 1
            for label, spec in KEYS:
                samples[label].append(value(rec, spec))
    for label, _ in KEYS:
        mu, sd = stats(samples[label])
        lines.append(f"| {policy} | {label} | {fmt(mu, sd)} | {n_runs} |")
    lines.append("")

out_path.write_text("\n".join(lines) + "\n")
print(f"wrote {out_path}")
PY

note "done — summary at $SUMMARY_MD"
