# POLARIS Benchmarks

This directory contains the current KV-cache benchmark harness.

The benchmark scope is deliberately narrow:

- llama.cpp + CUDA baseline is measured end to end with `llama-bench`.
- llama.cpp + POLARIS is measured end to end with the KV-only LD_PRELOAD shim.
- vLLM and SGLang are compared at the KV allocator trace layer using the trace
  patches under `benchmarks/patches/`; they are not yet live POLARIS backends.

## Run llama.cpp / POLARIS

From the POLARIS repository:

```sh
sudo env \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules \
  LLAMA_CPP_DIR=../llama.cpp \
  POLARIS_BENCH_PROMPTS=128,512 \
  POLARIS_BENCH_GENS=32 \
  POLARIS_BENCH_REPETITIONS=3 \
  POLARIS_BENCH_MODES=native_cuda,polaris_no_pressure,polaris_pressure \
  benchmarks/scripts/run_llama_kv_bench.sh
```

Outputs are written under:

```text
benchmarks/results/llama_cpp/<timestamp>/
  runs.jsonl
  summary.md
  logs/
```

Each POLARIS record includes the raw `llama-bench` JSON, `/sys/kernel/polaris`
stats before/after, and derived deltas such as `offloads`, `reloads`,
`uvm_hook_calls`, `uvm_handled`, `uvm_bridge_map_calls`, and `uvm_errors`.

## Modes

- `native_cuda`: normal llama.cpp CUDA path without the shim.
- `polaris_no_pressure`: KV-only shim path with a large default GPU budget.
- `polaris_pressure`: KV-only shim path with a small default GPU budget.
- `polaris_sustained_pressure`: same as pressure, intended for larger prompt
  and generation matrices.

Useful knobs:

```sh
POLARIS_BENCH_PROMPTS=128,512,1024
POLARIS_BENCH_GENS=32,128
POLARIS_BENCH_REPETITIONS=3
POLARIS_BENCH_PRESSURE_BUDGET_BYTES=4194304
POLARIS_BENCH_NO_PRESSURE_BUDGET_BYTES=17179869184
POLARIS_BENCH_CPU_POOL_BYTES=4294967296
LLAMA_CPP_MODEL=/path/to/model.gguf
LLAMA_CPP_BIN=/path/to/llama-bench
```

## vLLM / SGLang KV Trace Comparison

Apply the trace patches from:

```text
benchmarks/patches/vllm/trace_kv_cache.patch
benchmarks/patches/sglang/trace_alloc.patch
```

Then collect traces with:

```sh
POLARIS_TRACE=/tmp/vllm_trace.csv vllm serve <model>
POLARIS_TRACE=/tmp/sglang_trace.csv python -m sglang.launch_server --model <model>
```

Append their KV-level summaries to a llama.cpp/POLARIS benchmark run:

```sh
sudo env \
  VLLM_TRACE=/tmp/vllm_trace.csv \
  SGLANG_TRACE=/tmp/sglang_trace.csv \
  benchmarks/scripts/run_llama_kv_bench.sh
```

Or summarize a trace directly:

```sh
python3 benchmarks/scripts/kv_trace_summary.py \
  --trace /tmp/vllm_trace.csv \
  --source vllm \
  --jsonl
```

The trace summary reports allocator-level metrics:

- `unique_sessions`
- `peak_active_sessions`
- `total_reserved_blocks`
- `total_reserved_tokens`
- `peak_live_blocks`
- `peak_live_tokens`
- release/session event counts

This is the right comparison layer until vLLM/SGLang get explicit POLARIS KV
allocator backends. End-to-end throughput comparisons across different engines
should be labeled separately from these KV allocator metrics.
