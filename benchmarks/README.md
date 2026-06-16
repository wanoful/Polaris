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
POLARIS_BENCH_DAEMON_STARTUP_TIMEOUT_SEC=60
POLARIS_BENCH_RUN_TIMEOUT_SEC=240
POLARIS_BENCH_RUN_TIMEOUT_KILL_AFTER_SEC=15
POLARIS_BENCH_ALLOW_FAILURES=1
LLAMA_CPP_MODEL=/path/to/model.gguf
LLAMA_CPP_BIN=/path/to/llama-bench
```

Run a POLARIS pressure-budget sweep for one workload:

```sh
sudo -v
POLARIS_SWEEP_RUN_ID=pressure-sweep-512x128-$(date -u +%Y%m%dT%H%M%SZ) \
POLARIS_SWEEP_PROMPTS=512 \
POLARIS_SWEEP_GENS=128 \
POLARIS_SWEEP_REPETITIONS=3 \
POLARIS_SWEEP_BUDGETS_MIB=4,16,64,256 \
NVIDIA_KO_DIR=../open-gpu-kernel-modules \
LLAMA_CPP_DIR=../llama.cpp \
LLAMA_CPP_MODEL=/path/to/model.gguf \
benchmarks/scripts/run_polaris_pressure_sweep.sh
```

The sweep writes one benchmark artifact directory per GPU budget plus a
markdown report under `benchmarks/reports/`.

## vLLM / SGLang KV Trace Comparison

For a real-model offline benchmark against unmodified vLLM/SGLang, use:

```sh
FRAMEWORK_BENCH_TRACE=0 \
FRAMEWORK_BENCH_MODEL=/home/wano/workspace/models/SmolLM2-135M-Instruct \
FRAMEWORK_BENCH_NUM_PROMPTS=16 \
FRAMEWORK_BENCH_REPETITIONS=1 \
FRAMEWORK_BENCH_WARMUP_REPETITIONS=1 \
FRAMEWORK_BENCH_INPUT_LEN=128 \
FRAMEWORK_BENCH_OUTPUT_LEN=32 \
FRAMEWORK_BENCH_MAX_MODEL_LEN=256 \
benchmarks/scripts/run_framework_kv_bench.sh
```

The runner:

- runs unmodified vLLM/SGLang by default through their public offline APIs;
- uses fixed token-id prompts so the input/output token counts are controlled;
- writes framework throughput artifacts and `summary.md` under
  `benchmarks/results/frameworks/<timestamp>/`.

Set `FRAMEWORK_BENCH_TRACE=1` to apply the trace patches and collect KV
allocator lifecycle CSVs. Trace mode is for KV allocator inspection, not for
strict "original framework" throughput reporting.

Useful knobs:

```sh
FRAMEWORK_BENCH_MODES=vllm,sglang
FRAMEWORK_BENCH_TRACE=0
FRAMEWORK_BENCH_MODEL=/path/to/hf/model
FRAMEWORK_BENCH_NUM_PROMPTS=16
FRAMEWORK_BENCH_REPETITIONS=1
FRAMEWORK_BENCH_WARMUP_REPETITIONS=1
FRAMEWORK_BENCH_INPUT_LEN=128
FRAMEWORK_BENCH_OUTPUT_LEN=32
FRAMEWORK_BENCH_MAX_MODEL_LEN=256
FRAMEWORK_BENCH_GPU_MEMORY_UTILIZATION=0.45
FRAMEWORK_BENCH_VLLM_RANDOM_RANGE_RATIO=0.0
FRAMEWORK_BENCH_SGLANG_RANDOM_RANGE_RATIO=1.0
VLLM_BIN=/path/to/vllm
VLLM_PYTHON=/path/to/python
SGLANG_PYTHON=/path/to/python
SGLANG_DIR=/path/to/sglang
```

The vLLM and SGLang random dataset flags do not use the same range-ratio
semantics in the tested versions. The runner defaults to vLLM `0.0` and SGLang
`1.0` for controlled 128/32 token-length runs.

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

Recorded real-model results:

- `benchmarks/reports/framework-kv-real-model-20260616.md`
- `benchmarks/reports/llama-polaris-vs-original-frameworks-20260616.md`
- `benchmarks/reports/llama-polaris-vs-original-frameworks-trend-20260616.md`
- `benchmarks/reports/pressure-sweep-512x128-20260616T1420Z.md`
- `benchmarks/reports/qwen14b-polaris-kv-16k-20260616.md`
- `benchmarks/reports/qwen14b-polaris-pressure-sweep-20260616.md`

Regenerate a cross-engine report from existing artifacts:

```sh
python3 benchmarks/scripts/generate_framework_comparison_report.py \
  --title "llama.cpp + POLARIS vs Original Framework Trend - 2026-06-16" \
  --llama-run 128/32:benchmarks/results/llama_cpp/llama-polaris-128x32-20260616T1325Z \
  --llama-run 512/128:benchmarks/results/llama_cpp/llama-polaris-512x128-r3-20260616T1336Z \
  --llama-run 1024/128:benchmarks/results/llama_cpp/llama-polaris-1024x128-r3-20260616T1340Z \
  --framework-run 128/32:vllm:benchmarks/results/frameworks/original-vllm-clean-128x32-20260616T1327Z \
  --framework-run 128/32:sglang:benchmarks/results/frameworks/original-128x32-20260616T1324Z \
  --framework-run 512/128:vllm:benchmarks/results/frameworks/original-512x128-r3-20260616T1334Z \
  --framework-run 512/128:sglang:benchmarks/results/frameworks/original-512x128-r3-20260616T1334Z \
  --framework-run 1024/128:vllm:benchmarks/results/frameworks/original-1024x128-r3-20260616T1338Z \
  --framework-run 1024/128:sglang:benchmarks/results/frameworks/original-1024x128-r3-20260616T1338Z \
  --output /tmp/polaris-framework-comparison.md
```
