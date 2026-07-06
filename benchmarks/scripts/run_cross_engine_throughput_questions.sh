#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# Answer two cross-engine questions with explicit metric scopes:
#
# 1. Who has higher aggregate throughput for a 16-request fixed-shape batch?
# 2. Who has faster single-request decode for one fixed-shape request?
#
# vLLM/SGLang rows are native framework baselines. They do not route KV cache
# through POLARIS. The default llama.cpp rows are native llama.cpp rows so this
# script can run without root. Set CROSS_ENGINE_INCLUDE_POLARIS=1 to also run
# the existing llama.cpp + POLARIS harness for the single-request question.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "$ROOT_DIR/.." && pwd)"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$WORKSPACE_DIR/llama.cpp}"
LLAMA_CPP_MODEL="${LLAMA_CPP_MODEL:-$WORKSPACE_DIR/models/SmolLM2-135M-Instruct-F16.gguf}"
FRAMEWORK_BENCH_MODEL="${FRAMEWORK_BENCH_MODEL:-$WORKSPACE_DIR/models/SmolLM2-135M-Instruct}"

LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-$LLAMA_CPP_DIR/build/bin/llama-bench}"
LLAMA_BATCHED_BENCH_BIN="${LLAMA_BATCHED_BENCH_BIN:-$LLAMA_CPP_DIR/build/bin/llama-batched-bench}"

RESULTS_ROOT="${CROSS_ENGINE_RESULTS_DIR:-$ROOT_DIR/benchmarks/results/cross_engine}"
RUN_ID="${CROSS_ENGINE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${CROSS_ENGINE_OUT_DIR:-$RESULTS_ROOT/$RUN_ID}"
LOG_DIR="$OUT_DIR/logs"
SUMMARY_MD="$OUT_DIR/summary.md"

PROMPT_TOKENS="${CROSS_ENGINE_PROMPT_TOKENS:-512}"
GEN_TOKENS="${CROSS_ENGINE_GEN_TOKENS:-128}"
AGGREGATE_PROMPTS="${CROSS_ENGINE_AGGREGATE_PROMPTS:-16}"
REPETITIONS="${CROSS_ENGINE_REPETITIONS:-3}"
WARMUP_REPETITIONS="${CROSS_ENGINE_WARMUP_REPETITIONS:-1}"
MAX_MODEL_LEN="${CROSS_ENGINE_MAX_MODEL_LEN:-$((PROMPT_TOKENS + GEN_TOKENS + 128))}"
GPU_MEMORY_UTILIZATION="${CROSS_ENGINE_GPU_MEMORY_UTILIZATION:-0.45}"
FRAMEWORK_MODES="${CROSS_ENGINE_FRAMEWORK_MODES:-vllm,sglang}"

RUN_SINGLE="${CROSS_ENGINE_RUN_SINGLE:-1}"
RUN_AGGREGATE="${CROSS_ENGINE_RUN_AGGREGATE:-1}"
INCLUDE_POLARIS="${CROSS_ENGINE_INCLUDE_POLARIS:-0}"

LLAMA_CPP_GPU_LAYERS="${LLAMA_CPP_GPU_LAYERS:-999}"
LLAMA_CPP_FLASH_ATTN="${LLAMA_CPP_FLASH_ATTN:-0}"
LLAMA_CPP_DEVICE="${LLAMA_CPP_DEVICE:-CUDA0}"
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-2048}"
LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-512}"

die() {
    echo "error: $*" >&2
    exit 1
}

note() {
    echo "==> $*" >&2
}

require_file() {
    [[ -f "$1" ]] || die "$2 not found: $1"
}

require_executable() {
    [[ -x "$1" ]] || die "$2 is not executable: $1"
}

run_llama_single_native() {
    local dir="$OUT_DIR/single_request/llama_cpp_native"
    local json="$dir/llama-bench-p${PROMPT_TOKENS}-n${GEN_TOKENS}.json"
    local log="$LOG_DIR/llama_single_native.log"
    mkdir -p "$dir" "$LOG_DIR"

    note "running llama.cpp native single-request llama-bench"
    "$LLAMA_BENCH_BIN" \
        -m "$LLAMA_CPP_MODEL" \
        -p "$PROMPT_TOKENS" \
        -n "$GEN_TOKENS" \
        -r "$REPETITIONS" \
        --no-warmup \
        -ngl "$LLAMA_CPP_GPU_LAYERS" \
        -fa "$LLAMA_CPP_FLASH_ATTN" \
        -o json \
        -dev "$LLAMA_CPP_DEVICE" \
        >"$json" 2>"$log"
}

run_llama_single_polaris_optional() {
    if [[ "$INCLUDE_POLARIS" != "1" ]]; then
        return 0
    fi

    local dir="$OUT_DIR/single_request/llama_cpp_polaris"
    note "running optional llama.cpp + POLARIS single-request harness"
    POLARIS_BENCH_OUT_DIR="$dir" \
    POLARIS_BENCH_RUN_ID="$RUN_ID-single-polaris" \
    POLARIS_BENCH_PROMPTS="$PROMPT_TOKENS" \
    POLARIS_BENCH_GENS="$GEN_TOKENS" \
    POLARIS_BENCH_REPETITIONS="$REPETITIONS" \
    POLARIS_BENCH_MODES="${POLARIS_BENCH_MODES:-polaris_no_pressure,polaris_pressure}" \
    LLAMA_CPP_BIN="$LLAMA_BENCH_BIN" \
    LLAMA_CPP_MODEL="$LLAMA_CPP_MODEL" \
        "$ROOT_DIR/benchmarks/scripts/run_llama_kv_bench.sh"
}

run_framework_single() {
    local dir="$OUT_DIR/single_request/frameworks"
    note "running native vLLM/SGLang single-request framework benchmark"
    FRAMEWORK_BENCH_OUT_DIR="$dir" \
    FRAMEWORK_BENCH_RUN_ID="$RUN_ID-single-frameworks" \
    FRAMEWORK_BENCH_MODEL="$FRAMEWORK_BENCH_MODEL" \
    FRAMEWORK_BENCH_INPUT_LEN="$PROMPT_TOKENS" \
    FRAMEWORK_BENCH_OUTPUT_LEN="$GEN_TOKENS" \
    FRAMEWORK_BENCH_NUM_PROMPTS=1 \
    FRAMEWORK_BENCH_REPETITIONS="$REPETITIONS" \
    FRAMEWORK_BENCH_WARMUP_REPETITIONS="$WARMUP_REPETITIONS" \
    FRAMEWORK_BENCH_MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    FRAMEWORK_BENCH_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    FRAMEWORK_BENCH_MODES="$FRAMEWORK_MODES" \
    FRAMEWORK_BENCH_TRACE=0 \
        "$ROOT_DIR/benchmarks/scripts/run_framework_kv_bench.sh"
}

ensure_llama_batched_bench() {
    if [[ -x "$LLAMA_BATCHED_BENCH_BIN" ]]; then
        return 0
    fi
    note "building llama-batched-bench target"
    cmake --build "$LLAMA_CPP_DIR/build" --target llama-batched-bench -j"$(nproc)"
    require_executable "$LLAMA_BATCHED_BENCH_BIN" "llama-batched-bench"
}

run_llama_aggregate_native() {
    ensure_llama_batched_bench

    local dir="$OUT_DIR/aggregate_batch/llama_cpp_native"
    local jsonl="$dir/llama-batched-bench-p${PROMPT_TOKENS}-n${GEN_TOKENS}-b${AGGREGATE_PROMPTS}.jsonl"
    local log="$LOG_DIR/llama_aggregate_native.log"
    mkdir -p "$dir" "$LOG_DIR"

    local ctx_size=$((AGGREGATE_PROMPTS * (PROMPT_TOKENS + GEN_TOKENS)))
    if [[ "$ctx_size" -lt "$MAX_MODEL_LEN" ]]; then
        ctx_size="$MAX_MODEL_LEN"
    fi

    note "running llama.cpp native aggregate batch llama-batched-bench"
    "$LLAMA_BATCHED_BENCH_BIN" \
        -m "$LLAMA_CPP_MODEL" \
        -c "$ctx_size" \
        -b "$LLAMA_BATCH_SIZE" \
        -ub "$LLAMA_UBATCH_SIZE" \
        -ngl "$LLAMA_CPP_GPU_LAYERS" \
        -fa "$LLAMA_CPP_FLASH_ATTN" \
        -npp "$PROMPT_TOKENS" \
        -ntg "$GEN_TOKENS" \
        -npl "$AGGREGATE_PROMPTS" \
        --output-format jsonl \
        >"$jsonl" 2>"$log"
}

run_framework_aggregate() {
    local dir="$OUT_DIR/aggregate_batch/frameworks"
    note "running native vLLM/SGLang aggregate framework benchmark"
    FRAMEWORK_BENCH_OUT_DIR="$dir" \
    FRAMEWORK_BENCH_RUN_ID="$RUN_ID-aggregate-frameworks" \
    FRAMEWORK_BENCH_MODEL="$FRAMEWORK_BENCH_MODEL" \
    FRAMEWORK_BENCH_INPUT_LEN="$PROMPT_TOKENS" \
    FRAMEWORK_BENCH_OUTPUT_LEN="$GEN_TOKENS" \
    FRAMEWORK_BENCH_NUM_PROMPTS="$AGGREGATE_PROMPTS" \
    FRAMEWORK_BENCH_REPETITIONS="$REPETITIONS" \
    FRAMEWORK_BENCH_WARMUP_REPETITIONS="$WARMUP_REPETITIONS" \
    FRAMEWORK_BENCH_MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    FRAMEWORK_BENCH_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    FRAMEWORK_BENCH_MODES="$FRAMEWORK_MODES" \
    FRAMEWORK_BENCH_TRACE=0 \
        "$ROOT_DIR/benchmarks/scripts/run_framework_kv_bench.sh"
}

write_summary() {
    python3 - "$OUT_DIR" "$SUMMARY_MD" "$PROMPT_TOKENS" "$GEN_TOKENS" "$AGGREGATE_PROMPTS" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
summary = Path(sys.argv[2])
prompt = int(sys.argv[3])
gen = int(sys.argv[4])
aggregate_prompts = int(sys.argv[5])

def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None

def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip().startswith("{")]

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)

def llama_rows(path: Path):
    rows = load_json(path)
    if not rows:
        return None
    prompt_row = next((r for r in rows if r.get("n_prompt", 0) > 0), {})
    gen_row = next((r for r in rows if r.get("n_gen", 0) > 0), {})
    values = [r.get("avg_ts") for r in rows if isinstance(r.get("avg_ts"), (int, float))]
    return {
        "prompt_ts": prompt_row.get("avg_ts"),
        "decode_ts": gen_row.get("avg_ts"),
        "aggregate": sum(values) / len(values) if values else None,
    }

def framework_row(path: Path):
    rec = load_json(path)
    if not rec:
        return None
    return {
        "requests": rec.get("num_requests"),
        "input_ts": rec.get("input_tokens_per_second"),
        "output_ts": rec.get("output_tokens_per_second"),
        "total_ts": rec.get("total_tokens_per_second"),
        "elapsed": rec.get("elapsed_time"),
    }

single_llama = llama_rows(out_dir / "single_request" / "llama_cpp_native" / f"llama-bench-p{prompt}-n{gen}.json")
single_vllm = framework_row(out_dir / "single_request" / "frameworks" / "vllm" / f"bench_{prompt}x{gen}.json")
single_sglang = framework_row(out_dir / "single_request" / "frameworks" / "sglang" / f"bench_{prompt}x{gen}.json")

aggregate_llama_rows = load_jsonl(out_dir / "aggregate_batch" / "llama_cpp_native" / f"llama-batched-bench-p{prompt}-n{gen}-b{aggregate_prompts}.jsonl")
aggregate_llama = next((r for r in aggregate_llama_rows if r.get("pp") == prompt and r.get("tg") == gen and r.get("pl") == aggregate_prompts), None)
aggregate_vllm = framework_row(out_dir / "aggregate_batch" / "frameworks" / "vllm" / f"bench_{prompt}x{gen}.json")
aggregate_sglang = framework_row(out_dir / "aggregate_batch" / "frameworks" / "sglang" / f"bench_{prompt}x{gen}.json")

lines = [
    "# Cross-Engine Throughput Questions",
    "",
    f"- workload: {prompt}/{gen}",
    f"- aggregate batch size: {aggregate_prompts} requests",
    "- vLLM/SGLang rows are native framework baselines; they do not route KV through POLARIS.",
    "- llama.cpp aggregate row uses llama-batched-bench, not llama-bench.",
    "",
    "## Question 1: Who has faster single-request decode?",
    "",
    "| system | request shape | prompt/input tok/s | decode/output tok/s | aggregate metric | aggregate metric name |",
    "|---|---:|---:|---:|---:|---|",
]

if single_llama:
    lines.append("| llama.cpp native | 1 | {p} | {d} | {a} | llama_avg_ts |".format(
        p=fmt(single_llama["prompt_ts"]),
        d=fmt(single_llama["decode_ts"]),
        a=fmt(single_llama["aggregate"]),
    ))
if single_vllm:
    lines.append("| vLLM native | {n} | {p} | {d} | {a} | total tok/s |".format(
        n=single_vllm["requests"],
        p=fmt(single_vllm["input_ts"]),
        d=fmt(single_vllm["output_ts"]),
        a=fmt(single_vllm["total_ts"]),
    ))
if single_sglang:
    lines.append("| SGLang native | {n} | {p} | {d} | {a} | total tok/s |".format(
        n=single_sglang["requests"],
        p=fmt(single_sglang["input_ts"]),
        d=fmt(single_sglang["output_ts"]),
        a=fmt(single_sglang["total_ts"]),
    ))

lines.extend([
    "",
    "For this question, compare the `decode/output tok/s` column. Each row is a single request.",
    "",
    "## Question 2: Who has higher aggregate throughput?",
    "",
    "| system | request shape | aggregate prompt/input tok/s | aggregate decode/output tok/s | aggregate total tok/s | elapsed s |",
    "|---|---:|---:|---:|---:|---:|",
])

if aggregate_llama:
    lines.append("| llama.cpp native batched | {n} | {p} | {d} | {a} | {elapsed} |".format(
        n=aggregate_llama.get("pl"),
        p=fmt(aggregate_llama.get("speed_pp")),
        d=fmt(aggregate_llama.get("speed_tg")),
        a=fmt(aggregate_llama.get("speed")),
        elapsed=fmt(aggregate_llama.get("t")),
    ))
if aggregate_vllm:
    lines.append("| vLLM native | {n} | {p} | {d} | {a} | {elapsed} |".format(
        n=aggregate_vllm["requests"],
        p=fmt(aggregate_vllm["input_ts"]),
        d=fmt(aggregate_vllm["output_ts"]),
        a=fmt(aggregate_vllm["total_ts"]),
        elapsed=fmt(aggregate_vllm["elapsed"]),
    ))
if aggregate_sglang:
    lines.append("| SGLang native | {n} | {p} | {d} | {a} | {elapsed} |".format(
        n=aggregate_sglang["requests"],
        p=fmt(aggregate_sglang["input_ts"]),
        d=fmt(aggregate_sglang["output_ts"]),
        a=fmt(aggregate_sglang["total_ts"]),
        elapsed=fmt(aggregate_sglang["elapsed"]),
    ))

lines.extend([
    "",
    "For this question, compare aggregate throughput columns. This answers a batch-throughput question, not a per-request latency question.",
    "",
    "## Caveats",
    "",
    "- llama.cpp and vLLM/SGLang still use different engines, model formats, batching implementations, and kernels.",
    "- This script improves metric alignment, but it is still not a same-engine KV-backend A/B.",
    "- Use the single-request section for decode-rate claims and the aggregate section for throughput-capacity claims.",
    "- POLARIS rows are optional and should be interpreted through the llama.cpp same-engine A/B lens.",
    "",
    "## Artifacts",
    "",
    f"- `{out_dir}`",
])

summary.write_text("\n".join(lines) + "\n")
print(summary)
PY
}

main() {
    require_executable "$LLAMA_BENCH_BIN" "llama-bench"
    require_file "$LLAMA_CPP_MODEL" "llama.cpp GGUF model"
    [[ -e "$FRAMEWORK_BENCH_MODEL" ]] || die "framework model path not found: $FRAMEWORK_BENCH_MODEL"
    mkdir -p "$OUT_DIR" "$LOG_DIR"

    if [[ "$RUN_SINGLE" == "1" ]]; then
        run_llama_single_native
        run_llama_single_polaris_optional
        run_framework_single
    fi

    if [[ "$RUN_AGGREGATE" == "1" ]]; then
        run_llama_aggregate_native
        run_framework_aggregate
    fi

    write_summary
    note "summary written to $SUMMARY_MD"
}

main "$@"
