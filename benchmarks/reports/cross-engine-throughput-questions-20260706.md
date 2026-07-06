# Cross-Engine Throughput Questions - 2026-07-06

This report records the first run of
`benchmarks/scripts/run_cross_engine_throughput_questions.sh`, which separates
two questions that were previously conflated:

1. Who has faster single-request decode?
2. Who has higher fixed-shape aggregate batch throughput?

The run completed for llama.cpp native, vLLM native, and SGLang native.
vLLM and SGLang were run as separate same-shape invocations of
`run_cross_engine_throughput_questions.sh`; each invocation also reran the
llama.cpp native rows as a local reference. The tables below use the llama.cpp
native rows from the corresponding invocation.

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
| llama.cpp native, vLLM run | 1 | 36188.02 | 929.22 | 18558.62 | llama_avg_ts |
| vLLM native | 1 | 550.92 | 137.73 | 688.65 | total tok/s |
| llama.cpp native, SGLang run | 1 | 36497.81 | 928.40 | 18713.11 | llama_avg_ts |
| SGLang native | 1 | 587.83 | 146.96 | 734.79 | total tok/s |

For these runs, llama.cpp native has higher single-request decode throughput:

- llama.cpp native: 928.40-929.22 decode tok/s across the two invocations.
- vLLM native: 137.73 output tok/s.
- SGLang native: 146.96 output tok/s.

This is a single-request comparison. It should not be mixed with 16-request
aggregate framework output throughput.

## Question 2: Who Has Higher Aggregate Throughput?

| system | request shape | aggregate prompt/input tok/s | aggregate decode/output tok/s | aggregate total tok/s | elapsed s |
|---|---:|---:|---:|---:|---:|
| llama.cpp native batched, vLLM run | 16 | 34581.01 | 5827.27 | 17404.79 | 0.59 |
| vLLM native | 16 | 8202.20 | 2050.55 | 10252.75 | 1.00 |
| llama.cpp native batched, SGLang run | 16 | 37303.68 | 5805.63 | 17890.71 | 0.57 |
| SGLang native | 16 | 8173.87 | 2043.47 | 10217.33 | 1.00 |

For these runs, llama.cpp native batched has higher 16-request aggregate
throughput:

- llama.cpp native batched: 5805.63-5827.27 aggregate decode/output tok/s.
- vLLM native: 2050.55 aggregate output tok/s.
- SGLang native: 2043.47 aggregate output tok/s.

This result is a batch-throughput comparison, not a per-request latency
comparison.

## Reproduce

This machine has SGLang at `/home/wano/workspace/.bench-venvs/sglang/bin/python`.
The runner also auto-detects the repo-local vLLM environment at
`.venv-vllm/bin/vllm` when present.

To run both vLLM and SGLang in one invocation:

```sh
CROSS_ENGINE_FRAMEWORK_MODES=vllm,sglang \
SGLANG_PYTHON=/home/wano/workspace/.bench-venvs/sglang/bin/python \
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
- vLLM and SGLang were run in separate invocations for this report, so their
  paired llama.cpp native rows differ slightly due to normal run-to-run
  variation.

## Artifacts

Raw artifacts were written under the ignored results tree:

- `benchmarks/results/cross_engine/cross-engine-questions-vllm-20260706T000000Z/`
- `benchmarks/results/cross_engine/cross-engine-questions-sglang-20260706T000000Z/`
