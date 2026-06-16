#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# Run a llama.cpp/POLARIS KV-cache GPU-budget sweep.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_ROOT="${POLARIS_SWEEP_RESULTS_DIR:-$ROOT_DIR/benchmarks/results/llama_cpp}"
REPORTS_ROOT="${POLARIS_SWEEP_REPORTS_DIR:-$ROOT_DIR/benchmarks/reports}"
RUN_ID="${POLARIS_SWEEP_RUN_ID:-pressure-sweep-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="$RESULTS_ROOT/$RUN_ID"
REPORT_PATH="${POLARIS_SWEEP_REPORT:-$REPORTS_ROOT/$RUN_ID.md}"

PROMPTS="${POLARIS_SWEEP_PROMPTS:-512}"
GENS="${POLARIS_SWEEP_GENS:-128}"
REPETITIONS="${POLARIS_SWEEP_REPETITIONS:-3}"
BUDGETS_MIB="${POLARIS_SWEEP_BUDGETS_MIB:-4,16,64,256}"
MODES="${POLARIS_SWEEP_MODES:-polaris_pressure}"

note() {
    echo "==> $*" >&2
}

mkdir -p "$OUT_ROOT" "$(dirname "$REPORT_PATH")"

IFS=',' read -r -a budget_matrix <<<"$BUDGETS_MIB"
report_args=()

for budget_mib in "${budget_matrix[@]}"; do
    budget_mib="${budget_mib//[[:space:]]/}"
    [[ -n "$budget_mib" ]] || continue
    budget_bytes=$((budget_mib * 1024 * 1024))
    run_name="${RUN_ID}-${budget_mib}mib"
    out_dir="$OUT_ROOT/$budget_mib-mib"
    note "running budget=${budget_mib} MiB out=$out_dir"
    sudo env \
        NVIDIA_KO_DIR="${NVIDIA_KO_DIR:-../open-gpu-kernel-modules}" \
        LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-../llama.cpp}" \
        LLAMA_CPP_MODEL="${LLAMA_CPP_MODEL:-/home/wano/workspace/models/SmolLM2-135M-Instruct-F16.gguf}" \
        POLARIS_BENCH_RUN_ID="$run_name" \
        POLARIS_BENCH_OUT_DIR="$out_dir" \
        POLARIS_BENCH_PROMPTS="$PROMPTS" \
        POLARIS_BENCH_GENS="$GENS" \
        POLARIS_BENCH_REPETITIONS="$REPETITIONS" \
        POLARIS_BENCH_MODES="$MODES" \
        POLARIS_BENCH_PRESSURE_BUDGET_BYTES="$budget_bytes" \
        POLARIS_BENCH_SUSTAINED_BUDGET_BYTES="$budget_bytes" \
        "$ROOT_DIR/benchmarks/scripts/run_llama_kv_bench.sh"
    report_args+=(--run "${budget_mib} MiB:$out_dir")
done

python3 "$ROOT_DIR/benchmarks/scripts/generate_pressure_sweep_report.py" \
    --title "llama.cpp / POLARIS Pressure Budget Sweep - $RUN_ID" \
    "${report_args[@]}" \
    --output "$REPORT_PATH" >/dev/null

note "wrote $REPORT_PATH"
