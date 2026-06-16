# KV Cache Comparison Method

This comparison is about KV-cache management behavior, not a single blended
throughput number across unrelated engines.

## What Is Comparable Today

POLARIS currently has a live llama.cpp KV-only path:

- model weights stay on the normal CUDA path;
- ggml KV-cache allocations are routed through POLARIS;
- daemon-owned RM backing is materialized by UVM replayable faults;
- pressure can trigger `OFFLOAD` / `RELOAD` through `POLARIS_RM_COPY`.

vLLM and SGLang do not yet allocate KV cache from POLARIS. They are compared
through allocator traces collected by the patches under `benchmarks/patches/`.
Those traces expose logical KV lifecycle events in a common format:

```csv
timestamp_ns,op,session_id,token_start,token_count
```

## Metrics

End-to-end llama.cpp / POLARIS records include:

- `llama_avg_ts`: average `llama-bench` tokens/s across prompt and generation
  rows;
- `stats_delta.offloads`;
- `stats_delta.reloads`;
- `stats_delta.uvm_hook_calls`;
- `stats_delta.uvm_handled`;
- `stats_delta.uvm_bridge_map_calls`;
- `stats_delta.uvm_no_pte`;
- `stats_delta.uvm_errors`.

vLLM / SGLang trace records include:

- `unique_sessions`;
- `peak_active_sessions`;
- `total_reserved_blocks`;
- `total_reserved_tokens`;
- `peak_live_blocks`;
- `peak_live_tokens`;
- `implicit_destroy_released_blocks`.

Use the trace metrics to compare KV allocation pressure, block churn, and live
KV footprint. Do not present trace-only vLLM/SGLang rows as end-to-end serving
throughput.

## First Benchmark Matrix

Recommended initial POLARIS matrix:

```sh
sudo env \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules \
  LLAMA_CPP_DIR=../llama.cpp \
  POLARIS_BENCH_PROMPTS=128,512,1024 \
  POLARIS_BENCH_GENS=32,128 \
  POLARIS_BENCH_REPETITIONS=3 \
  POLARIS_BENCH_MODES=native_cuda,polaris_no_pressure,polaris_pressure \
  benchmarks/scripts/run_llama_kv_bench.sh
```

Recommended pressure follow-up:

```sh
sudo env \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules \
  LLAMA_CPP_DIR=../llama.cpp \
  POLARIS_BENCH_PROMPTS=512,1024 \
  POLARIS_BENCH_GENS=128 \
  POLARIS_BENCH_REPETITIONS=3 \
  POLARIS_BENCH_MODES=polaris_sustained_pressure \
  POLARIS_BENCH_PRESSURE_BUDGET_BYTES=4194304 \
  benchmarks/scripts/run_llama_kv_bench.sh
```

For vLLM/SGLang trace comparison, pass collected traces:

```sh
sudo env \
  VLLM_TRACE=/tmp/vllm_trace.csv \
  SGLANG_TRACE=/tmp/sglang_trace.csv \
  benchmarks/scripts/run_llama_kv_bench.sh
```

Or run a real-model framework benchmark that collects both traces:

```sh
FRAMEWORK_BENCH_MODEL=/home/wano/workspace/models/SmolLM2-135M-Instruct \
FRAMEWORK_BENCH_NUM_PROMPTS=16 \
FRAMEWORK_BENCH_INPUT_LEN=128 \
FRAMEWORK_BENCH_OUTPUT_LEN=32 \
FRAMEWORK_BENCH_MAX_MODEL_LEN=256 \
benchmarks/scripts/run_framework_kv_bench.sh
```

The first recorded real-model run is documented in
`benchmarks/reports/framework-kv-real-model-20260616.md`.

For the 16 x 128-input / 32-output run, the KV trace summary was:

| source | total reserved blocks | peak live blocks | total reserved tokens | peak live tokens |
|---|---:|---:|---:|---:|
| vLLM | 144 | 121 | 2304 | 1936 |
| SGLang | 640 | 640 | 2560 | 2560 |

## Interpretation

- `native_cuda` is the normal llama.cpp CUDA baseline.
- `polaris_no_pressure` estimates shim/fault path overhead when the resident
  budget is not binding.
- `polaris_pressure` estimates the cost and behavior of real KV offload/reload.
- vLLM/SGLang trace rows show how many logical KV blocks their allocators would
  reserve and keep live for the same request pattern, assuming the trace was
  collected from a comparable workload.

The next integration milestone for vLLM/SGLang is an explicit KV allocator
backend that routes their KV cache blocks through POLARIS. Until that exists,
KV trace replay is the honest comparison layer.
