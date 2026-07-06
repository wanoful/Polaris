# Cross-Engine Throughput Questions - 2026-07-06

This report records the first run of
`benchmarks/scripts/run_cross_engine_throughput_questions.sh`, which separates
two questions that were previously conflated:

1. Who has faster single-request decode?
2. Who has higher fixed-shape aggregate batch throughput?

The run completed for llama.cpp native and vLLM native. SGLang was not run
because no usable SGLang Python environment was found on this machine during
the run.

## Scope

- Workload: 512 prompt tokens, 128 generated tokens.
- Aggregate batch size: 16 requests.
- llama.cpp single-request path: `llama-bench`.
- llama.cpp aggregate path: `llama-batched-bench`.
- vLLM path: native framework baseline through `original_framework_bench.py`.
- POLARIS: not enabled in this run. This run answers native cross-engine
  throughput questions only.

The framework rows do not route KV cache through POLARIS.

## Question 1: Who Has Faster Single-Request Decode?

| system | request shape | prompt/input tok/s | decode/output tok/s | aggregate metric | aggregate metric name |
|---|---:|---:|---:|---:|---|
| llama.cpp native | 1 | 36188.02 | 929.22 | 18558.62 | llama_avg_ts |
| vLLM native | 1 | 550.92 | 137.73 | 688.65 | total tok/s |

For this run, llama.cpp native has higher single-request decode throughput:

- llama.cpp native: 929.22 decode tok/s.
- vLLM native: 137.73 output tok/s.

This is a single-request comparison. It should not be mixed with 16-request
aggregate framework output throughput.

## Question 2: Who Has Higher Aggregate Throughput?

| system | request shape | aggregate prompt/input tok/s | aggregate decode/output tok/s | aggregate total tok/s | elapsed s |
|---|---:|---:|---:|---:|---:|
| llama.cpp native batched | 16 | 34581.01 | 5827.27 | 17404.79 | 0.59 |
| vLLM native | 16 | 8202.20 | 2050.55 | 10252.75 | 1.00 |

For this run, llama.cpp native batched has higher 16-request aggregate
throughput:

- llama.cpp native batched: 5827.27 aggregate decode/output tok/s.
- vLLM native: 2050.55 aggregate output tok/s.

This result is a batch-throughput comparison, not a per-request latency
comparison.

## Missing SGLang Row

SGLang did not run because the current machine did not expose a usable
`SGLANG_PYTHON` or SGLang benchmark virtual environment. To add the missing
row, rerun with a valid SGLang interpreter, for example:

```sh
CROSS_ENGINE_RUN_ID=cross-engine-questions-sglang-$(date -u +%Y%m%dT%H%M%SZ) \
CROSS_ENGINE_FRAMEWORK_MODES=sglang \
SGLANG_PYTHON=/path/to/sglang-venv/bin/python \
benchmarks/scripts/run_cross_engine_throughput_questions.sh
```

To run both vLLM and SGLang:

```sh
CROSS_ENGINE_FRAMEWORK_MODES=vllm,sglang \
VLLM_BIN=/path/to/vllm-venv/bin/vllm \
VLLM_PYTHON=/path/to/vllm-venv/bin/python \
SGLANG_PYTHON=/path/to/sglang-venv/bin/python \
benchmarks/scripts/run_cross_engine_throughput_questions.sh
```

## Caveats

- This is still a cross-engine comparison: llama.cpp, vLLM, and SGLang use
  different runtimes, model formats, batching implementations, and kernels.
- `llama-batched-bench` gives a better aggregate-throughput llama.cpp baseline
  than single-stream `llama-bench`, but it is still not identical to vLLM's
  scheduler.
- This run does not evaluate POLARIS. POLARIS remains evaluated through the
  llama.cpp same-engine native-vs-POLARIS benchmark path.
- SGLang remains pending until the environment is restored or explicitly
  provided.

## Artifacts

Raw artifacts were written under the ignored results tree:

`benchmarks/results/cross_engine/cross-engine-questions-vllm-20260706T000000Z/`
