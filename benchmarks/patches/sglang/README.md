# SGLang Trace Collection Patch

## What it does

Adds KV cache token allocation/free tracing to SGLang for POLARIS Phase 4b benchmark comparison.
It also extends `bench_offline_throughput.py` so the real-model benchmark can
use `random-ids` with `--tokenize-prompt`, which keeps prompt lengths exact
instead of depending on text re-tokenization.

Set the environment variable `POLARIS_TRACE=<output_path>` before launching SGLang, and every KV cache allocation event will be written to the specified CSV file.

## How it differs from the vLLM patch

SGLang allocates KV cache at **token granularity** (not block granularity). The trace helpers aggregate token-level allocations into POLARIS block-sized (16 token) events to match the common trace format.

The trace portion targets `python/sglang/srt/mem_cache/common.py` — the central
orchestration module where all KV cache allocation flows converge:
- `alloc_for_extend()` — prefill allocation (batch level)
- `alloc_for_decode()` — decode allocation (batch level)
- `release_kv_cache()` — request completion / preemption

The benchmark portion targets `python/sglang/bench_offline_throughput.py`:
- adds `random-ids` to the accepted dataset choices;
- adds `--tokenize-prompt`;
- passes generated token ids as `input_ids` when requested.

## Trace format

```csv
timestamp_ns,op,session_id,token_start,token_count
```

| Op | When emitted |
|----|-------------|
| `SESSION_CREATE` | First `alloc_for_extend()` for a request (using `req.rid` as session_id) |
| `BLOCK_RESERVE` | One row per 16-token logical block reserved during extend or decode |
| `BLOCK_RELEASE` | One row per 16-token block released in `release_kv_cache()` |
| `SESSION_DESTROY` | `release_kv_cache()` called |

## Which SGLang version/commit

Tested against SGLang commit: `12ebb35439` (`0.5.13.post2.dev348+g12ebb3543`).

## How to apply

```bash
# From the SGLang repository root:
patch -p1 < /path/to/Polaris/benchmarks/patches/sglang/trace_alloc.patch

# Or with git:
git apply /path/to/Polaris/benchmarks/patches/sglang/trace_alloc.patch
```

## How to revert

```bash
patch -R -p1 < /path/to/Polaris/benchmarks/patches/sglang/trace_alloc.patch
```

## Usage

```bash
# Collect trace from an SGLang run:
POLARIS_TRACE=/tmp/sglang_trace.csv \
  python -m sglang.launch_server --model meta-llama/Llama-2-7b-hf

# Real-model offline benchmark with exact random token lengths:
POLARIS_TRACE=/tmp/sglang_trace.csv \
  python -m sglang.bench_offline_throughput \
    --model-path /path/to/model \
    --dataset-name random-ids \
    --random-input-len 128 \
    --random-output-len 32 \
    --random-range-ratio 1.0 \
    --num-prompts 16 \
    --context-length 256 \
    --dtype float16 \
    --mem-fraction-static 0.45 \
    --attention-backend flashinfer \
    --sampling-backend pytorch \
    --cuda-graph-backend-decode disabled \
    --cuda-graph-backend-prefill disabled \
    --skip-warmup \
    --tokenize-prompt \
    --result-filename /tmp/sglang_result.jsonl

# Copy the trace to POLARIS for replay:
cp /tmp/sglang_trace.csv /path/to/Polaris/benchmarks/traces/

# Replay through POLARIS (Phase 4b):
polaris-workload trace-replay --input benchmarks/traces/sglang_trace.csv
```

## Design notes

- **Token-to-block mapping**: SGLang allocates at token granularity (1 token per slot with `page_size=1`). The trace aggregates into POLARIS's 16-token blocks.
- **Zero behavioral change**: the patch only adds logging, never alters allocation logic.
- **Batch allocation**: SGLang allocates KV cache per batch, not per request. The trace helpers iterate over the batch to emit per-request events.
- **Prefix caching**: SGLang's RadixCache handles prefix sharing internally. Block-level events are emitted for logical reservations only — cached prefix hits don't generate BLOCK_RESERVE events (consistent with vLLM behavior).
- **Preemption**: SGLang handles preemption through `release_kv_cache()` which also inserts committed tokens into the radix tree. The trace captures this as BLOCK_RELEASE + SESSION_DESTROY, same as vLLM.
- **Benchmark token lengths**: `random` text prompts can re-tokenize to lengths
  that drift from the configured random input length. Use `random-ids`,
  `--tokenize-prompt`, and `--random-range-ratio 1.0` for controlled KV-cache
  comparisons.
