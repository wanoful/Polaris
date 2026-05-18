# vLLM Trace Collection Patch

## What it does

Adds KV cache block allocation/free tracing to vLLM for POLARIS Phase 4b benchmark comparison.

Set the environment variable `POLARIS_TRACE=<output_path>` before launching vLLM, and every KV cache block event will be written to the specified CSV file.

## Trace format

```csv
timestamp_ns,op,session_id,token_start,token_count
```

| Op | When emitted |
|----|-------------|
| `SESSION_CREATE` | First `allocate_slots()` call for a request_id |
| `BLOCK_RESERVE` | One row per newly reserved logical block (includes token_start and block_size) |
| `BLOCK_RELEASE` | One row per block released |
| `SESSION_DESTROY` | `KVCacheManager.free()` called (request completion or preemption) |

## Which vLLM version/commit

Tested against vLLM commit: `132765e3560659ff63ebd236203672e991b70e08`

The patch targets `vllm/v1/core/kv_cache_manager.py` — the V1 engine's single entry point for all KV cache block management.

## How to apply

```bash
# From the vLLM repository root:
patch -p1 < /path/to/Polaris/benchmarks/patches/vllm/trace_kv_cache.patch

# Or with git:
git apply /path/to/Polaris/benchmarks/patches/vllm/trace_kv_cache.patch
```

## How to revert

```bash
patch -R -p1 < /path/to/Polaris/benchmarks/patches/vllm/trace_kv_cache.patch
```

## Usage

```bash
# Collect trace from a vLLM run:
POLARIS_TRACE=/tmp/vllm_trace.csv vllm serve meta-llama/Llama-2-7b-hf

# Then send requests to the server normally.
# The trace file is written in real-time (flushed after each event).

# Copy the trace to POLARIS for replay:
cp /tmp/vllm_trace.csv /path/to/Polaris/benchmarks/traces/

# Replay through POLARIS (Phase 4b):
polaris-workload trace-replay --input benchmarks/traces/vllm_trace.csv --csv benchmarks/traces/polaris_replay.csv
```

## Design notes

- **Zero behavioral change**: the patch only adds logging, never alters allocation logic.
- **V1 engine targeting**: vLLM v1 replaced `BlockSpaceManager` (v0) with `KVCacheManager`. All KV cache block management flows through `allocate_slots()` and `free()`.
- **Preemption handling**: vLLM preemption calls `free()` then later re-allocates for the same request_id. The trace captures this as SESSION_DESTROY + SESSION_CREATE, which POLARIS replays as sequential create/destroy cycles.
- **Prefix caching**: blocks shared via prefix cache have `ref_cnt > 1` but appear as a single BLOCK_RESERVE. The trace faithfully records the logical reservation pattern without needing to understand prefix cache internals.
