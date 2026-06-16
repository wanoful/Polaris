#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# Run real-model vLLM and SGLang KV-cache benchmarks with POLARIS trace patches.
#
# This runner compares framework KV allocator behavior and framework-level
# offline throughput for the same synthetic token workload. It does not make
# vLLM/SGLang allocate KV cache from POLARIS.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "$ROOT_DIR/.." && pwd)"
MODEL_PATH="${FRAMEWORK_BENCH_MODEL:-$WORKSPACE_DIR/models/SmolLM2-135M-Instruct}"
VLLM_DIR="${VLLM_DIR:-$WORKSPACE_DIR/vllm}"
SGLANG_DIR="${SGLANG_DIR:-$WORKSPACE_DIR/sglang}"
RESULTS_ROOT="${FRAMEWORK_BENCH_RESULTS_DIR:-$ROOT_DIR/benchmarks/results/frameworks}"
RUN_ID="${FRAMEWORK_BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${FRAMEWORK_BENCH_OUT_DIR:-$RESULTS_ROOT/$RUN_ID}"
LOG_DIR="$OUT_DIR/logs"
RUNS_JSONL="$OUT_DIR/runs.jsonl"
SUMMARY_MD="$OUT_DIR/summary.md"

INPUT_LEN="${FRAMEWORK_BENCH_INPUT_LEN:-128}"
OUTPUT_LEN="${FRAMEWORK_BENCH_OUTPUT_LEN:-32}"
NUM_PROMPTS="${FRAMEWORK_BENCH_NUM_PROMPTS:-16}"
MAX_MODEL_LEN="${FRAMEWORK_BENCH_MAX_MODEL_LEN:-256}"
DTYPE="${FRAMEWORK_BENCH_DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${FRAMEWORK_BENCH_GPU_MEMORY_UTILIZATION:-0.45}"
MODES="${FRAMEWORK_BENCH_MODES:-vllm,sglang}"
VLLM_RANDOM_RANGE_RATIO="${FRAMEWORK_BENCH_VLLM_RANDOM_RANGE_RATIO:-0.0}"
SGLANG_RANDOM_RANGE_RATIO="${FRAMEWORK_BENCH_SGLANG_RANDOM_RANGE_RATIO:-1.0}"

VLLM_VENV="${VLLM_VENV:-$WORKSPACE_DIR/.bench-venvs/vllm}"
SGLANG_VENV="${SGLANG_VENV:-$WORKSPACE_DIR/.bench-venvs/sglang}"
VLLM_BIN="${VLLM_BIN:-}"
VLLM_PYTHON="${VLLM_PYTHON:-}"
SGLANG_PYTHON="${SGLANG_PYTHON:-}"

die() {
    echo "error: $*" >&2
    exit 1
}

note() {
    echo "==> $*" >&2
}

contains_mode() {
    local needle="$1"
    [[ ",$MODES," == *",$needle,"* ]]
}

require_file() {
    [[ -f "$1" ]] || die "$2 not found: $1"
}

require_dir() {
    [[ -d "$1" ]] || die "$2 not found: $1"
}

require_executable() {
    [[ -x "$1" ]] || die "$2 is not executable: $1"
}

find_vllm() {
    if [[ -n "$VLLM_BIN" ]]; then
        require_executable "$VLLM_BIN" "vLLM binary"
    elif [[ -x "$VLLM_VENV/bin/vllm" ]]; then
        VLLM_BIN="$VLLM_VENV/bin/vllm"
    elif [[ -x "/tmp/polaris-bench-vllm/bin/vllm" ]]; then
        VLLM_BIN="/tmp/polaris-bench-vllm/bin/vllm"
    elif command -v vllm >/dev/null 2>&1; then
        VLLM_BIN="$(command -v vllm)"
    else
        die "could not find vLLM. Set VLLM_BIN=/path/to/vllm or create $VLLM_VENV"
    fi

    if [[ -n "$VLLM_PYTHON" ]]; then
        require_executable "$VLLM_PYTHON" "vLLM Python"
    elif [[ -x "$(dirname "$VLLM_BIN")/python" ]]; then
        VLLM_PYTHON="$(dirname "$VLLM_BIN")/python"
    else
        die "could not find Python for vLLM. Set VLLM_PYTHON=/path/to/python"
    fi
}

find_sglang() {
    if [[ -n "$SGLANG_PYTHON" ]]; then
        require_executable "$SGLANG_PYTHON" "SGLang Python"
    elif [[ -x "$SGLANG_VENV/bin/python" ]]; then
        SGLANG_PYTHON="$SGLANG_VENV/bin/python"
    elif command -v python >/dev/null 2>&1; then
        SGLANG_PYTHON="$(command -v python)"
    else
        die "could not find SGLang Python. Set SGLANG_PYTHON=/path/to/python or create $SGLANG_VENV"
    fi
}

apply_vllm_trace_patch() {
    local patch_file="$ROOT_DIR/benchmarks/patches/vllm/trace_kv_cache.patch"
    require_file "$patch_file" "vLLM trace patch"

    local site_root
    site_root="$("$VLLM_PYTHON" - <<'PY'
import inspect
from pathlib import Path
import vllm.v1.core.kv_cache_manager as manager
print(Path(inspect.getfile(manager)).parents[3])
PY
)"

    local target="$site_root/vllm/v1/core/kv_cache_manager.py"
    require_file "$target" "vLLM KV cache manager"
    if grep -q "_POLARIS_TRACE_PATH" "$target"; then
        note "vLLM trace patch already applied in $site_root"
        return 0
    fi
    note "applying vLLM trace patch in $site_root"
    (cd "$site_root" && patch -p1 < "$patch_file")
}

apply_sglang_trace_patch() {
    local patch_file="$ROOT_DIR/benchmarks/patches/sglang/trace_alloc.patch"
    require_file "$patch_file" "SGLang trace patch"
    require_dir "$SGLANG_DIR" "SGLang repository"

    local trace_target="$SGLANG_DIR/python/sglang/srt/mem_cache/common.py"
    local bench_target="$SGLANG_DIR/python/sglang/bench_offline_throughput.py"
    require_file "$trace_target" "SGLang mem_cache common.py"
    require_file "$bench_target" "SGLang offline benchmark"
    if grep -q "_POLARIS_TRACE_PATH" "$trace_target" &&
        grep -q '"random-ids"' "$bench_target" &&
        grep -q -- "--tokenize-prompt" "$bench_target"; then
        note "SGLang trace patch already applied in $SGLANG_DIR"
        return 0
    fi
    note "applying SGLang trace patch in $SGLANG_DIR"
    git -C "$SGLANG_DIR" apply "$patch_file"
}

append_trace_summary() {
    local source="$1"
    local trace="$2"
    python3 "$ROOT_DIR/benchmarks/scripts/kv_trace_summary.py" \
        --trace "$trace" \
        --source "$source" \
        --jsonl >> "$RUNS_JSONL"
}

run_vllm_bench() {
    find_vllm
    apply_vllm_trace_patch

    local dir="$OUT_DIR/vllm"
    local trace="$dir/bench_${INPUT_LEN}x${OUTPUT_LEN}.trace.csv"
    local result="$dir/bench_${INPUT_LEN}x${OUTPUT_LEN}.json"
    local log="$LOG_DIR/vllm_${INPUT_LEN}x${OUTPUT_LEN}.log"
    mkdir -p "$dir"

    note "running vLLM real-model benchmark"
    rm -f "$trace" "$result"
    PATH="$(dirname "$VLLM_BIN"):$PATH" \
    POLARIS_TRACE="$trace" \
        "$VLLM_BIN" bench throughput \
        --model "$MODEL_PATH" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --random-range-ratio "$VLLM_RANDOM_RANGE_RATIO" \
        --num-prompts "$NUM_PROMPTS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --dtype "$DTYPE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --enforce-eager \
        --output-json "$result" \
        2>&1 | tee "$log"

    append_trace_summary "vllm" "$trace"
}

run_sglang_bench() {
    find_sglang
    apply_sglang_trace_patch

    local dir="$OUT_DIR/sglang"
    local trace="$dir/bench_${INPUT_LEN}x${OUTPUT_LEN}.trace.csv"
    local result="$dir/bench_${INPUT_LEN}x${OUTPUT_LEN}.jsonl"
    local log="$LOG_DIR/sglang_${INPUT_LEN}x${OUTPUT_LEN}.log"
    mkdir -p "$dir"

    note "running SGLang real-model benchmark"
    rm -f "$trace" "$result"
    PATH="$(dirname "$SGLANG_PYTHON"):$PATH" \
    POLARIS_TRACE="$trace" \
        "$SGLANG_PYTHON" -m sglang.bench_offline_throughput \
        --model-path "$MODEL_PATH" \
        --dataset-name random-ids \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --random-range-ratio "$SGLANG_RANDOM_RANGE_RATIO" \
        --num-prompts "$NUM_PROMPTS" \
        --context-length "$MAX_MODEL_LEN" \
        --dtype "$DTYPE" \
        --mem-fraction-static "$GPU_MEMORY_UTILIZATION" \
        --attention-backend flashinfer \
        --sampling-backend pytorch \
        --cuda-graph-backend-decode disabled \
        --cuda-graph-backend-prefill disabled \
        --skip-warmup \
        --tokenize-prompt \
        --result-filename "$result" \
        2>&1 | tee "$log"

    append_trace_summary "sglang" "$trace"
}

write_summary() {
    python3 - "$OUT_DIR" "$RUNS_JSONL" "$SUMMARY_MD" "$INPUT_LEN" "$OUTPUT_LEN" "$NUM_PROMPTS" "$VLLM_RANDOM_RANGE_RATIO" "$SGLANG_RANDOM_RANGE_RATIO" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
runs_jsonl = Path(sys.argv[2])
summary = Path(sys.argv[3])
input_len, output_len, num_prompts = map(int, sys.argv[4:7])
vllm_range_ratio = sys.argv[7]
sglang_range_ratio = sys.argv[8]

trace_records = []
if runs_jsonl.exists():
    trace_records = [json.loads(line) for line in runs_jsonl.read_text().splitlines() if line.strip()]

def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None

def load_jsonl_last(path):
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[-1] if rows else None

vllm = load_json(out_dir / "vllm" / f"bench_{input_len}x{output_len}.json")
sglang = load_jsonl_last(out_dir / "sglang" / f"bench_{input_len}x{output_len}.jsonl")
trace_by_source = {rec.get("source"): rec for rec in trace_records}

lines = [
    "# vLLM / SGLang KV Benchmark Summary",
    "",
    f"- workload: {num_prompts} prompts, input={input_len} tokens, output={output_len} tokens",
    f"- vLLM random range ratio: {vllm_range_ratio}",
    f"- SGLang random range ratio: {sglang_range_ratio}",
    "- scope: real-model framework throughput plus KV allocator trace",
    "- note: vLLM/SGLang traces do not route KV cache through POLARIS yet",
    "",
    "| source | requests/s | output tok/s | total tok/s | total reserved blocks | peak live blocks | total reserved tokens |",
    "|---|---:|---:|---:|---:|---:|---:|",
]

if vllm:
    elapsed = float(vllm.get("elapsed_time", 0.0) or 0.0)
    output_tok_s = (num_prompts * output_len / elapsed) if elapsed else 0.0
    rec = trace_by_source.get("vllm", {})
    lines.append(
        "| vllm | {rps:.2f} | {out:.2f} | {total:.2f} | {reserved} | {peak} | {tokens} |".format(
            rps=float(vllm.get("requests_per_second", 0.0)),
            out=output_tok_s,
            total=float(vllm.get("tokens_per_second", 0.0)),
            reserved=rec.get("total_reserved_blocks", ""),
            peak=rec.get("peak_live_blocks", ""),
            tokens=rec.get("total_reserved_tokens", ""),
        )
    )

if sglang:
    rec = trace_by_source.get("sglang", {})
    lines.append(
        "| sglang | {rps:.2f} | {out:.2f} | {total:.2f} | {reserved} | {peak} | {tokens} |".format(
            rps=float(sglang.get("request_throughput", 0.0)),
            out=float(sglang.get("output_throughput", 0.0)),
            total=float(sglang.get("total_throughput", 0.0)),
            reserved=rec.get("total_reserved_blocks", ""),
            peak=rec.get("peak_live_blocks", ""),
            tokens=rec.get("total_reserved_tokens", ""),
        )
    )

lines.extend([
    "",
    "Artifacts:",
    f"- `{runs_jsonl}`",
    f"- `{out_dir}`",
])
summary.write_text("\n".join(lines) + "\n")
print(summary)
PY
}

main() {
    require_dir "$(dirname "$MODEL_PATH")" "model parent directory"
    [[ -e "$MODEL_PATH" ]] || die "model path not found: $MODEL_PATH"
    mkdir -p "$LOG_DIR"
    : > "$RUNS_JSONL"

    contains_mode vllm && run_vllm_bench
    contains_mode sglang && run_sglang_bench
    write_summary

    note "results written to $OUT_DIR"
}

main "$@"
