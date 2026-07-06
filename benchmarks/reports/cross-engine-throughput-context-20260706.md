# Cross-Engine Throughput Context - 2026-07-06

This report repackages existing benchmark artifacts into a stricter
cross-engine presentation. It does not add new measurements.

## Claim Boundary

The controlled POLARIS comparison is the llama.cpp same-engine A/B:

- llama.cpp native CUDA
- llama.cpp + POLARIS KV-only shim, no pressure
- llama.cpp + POLARIS KV-only shim, pressure

The vLLM and SGLang rows are native framework baselines. They do not route KV
cache through POLARIS. They are included only as cross-engine throughput
context, not as a same-engine KV-backend A/B.

Do not interpret this report as:

- vLLM native vs vLLM + POLARIS
- SGLang native vs SGLang + POLARIS
- an isolated comparison of POLARIS KV management against vLLM/SGLang KV
  allocators

The fair statement is:

> We evaluate POLARIS in a controlled llama.cpp native-vs-POLARIS setup, and
> place those results next to native vLLM/SGLang throughput baselines to show
> the broader serving-stack performance context on the same GPU.

## Setup

- Model family: SmolLM2-135M-Instruct.
- llama.cpp model: `/home/wano/workspace/models/SmolLM2-135M-Instruct-F16.gguf`.
- vLLM/SGLang model: `/home/wano/workspace/models/SmolLM2-135M-Instruct`.
- GPU: NVIDIA GeForce RTX 5070 Ti.
- Driver: 610.43.02.
- POLARIS branch: `v4-fault-hook`.
- Framework baselines: native/original runtime paths, trace disabled.

Important differences remain:

- llama.cpp uses GGUF and llama-bench.
- vLLM/SGLang use their native model/runtime paths and offline batch serving.
- llama.cpp rows are one llama-bench workload shape.
- vLLM/SGLang rows use 16 prompts per batch.
- Framework rows are not POLARIS execution.

## Metric Definitions

For llama.cpp rows:

- `prompt tok/s` is the llama-bench prompt/prefill row (`n_prompt > 0`).
- `gen tok/s` is the llama-bench generation/decode row (`n_gen > 0`).
- `llama_avg_ts` is the arithmetic mean of llama-bench `avg_ts` rows:
  `(prompt tok/s + gen tok/s) / 2` for these two-row runs.

For vLLM/SGLang rows:

- `input tok/s` is framework input-token throughput.
- `output tok/s` is framework generated-token throughput.
- `total tok/s` is input plus output token throughput.

`llama_avg_ts` and framework `total tok/s` are not the same metric. For
cross-engine discussion, `gen tok/s` vs `output tok/s` is the closest available
decode-oriented comparison, while still not being a same-engine A/B.

## Controlled llama.cpp A/B

This is the primary POLARIS evidence because the engine is held constant.

| workload | mode | prompt tok/s | gen tok/s | llama_avg_ts | offloads | reloads | bridge maps | uvm_errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128/32 | native_cuda | 1041.51 | 815.75 | 928.63 | 0 | 0 | 0 | 0 |
| 128/32 | polaris_no_pressure | 599.83 | 724.73 | 662.28 | 0 | 0 | 1807 | 0 |
| 128/32 | polaris_pressure | 941.25 | 177.71 | 559.48 | 95 | 93 | 782 | 0 |
| 512/128 | native_cuda | 36281.44 | 930.33 | 18605.88 | 0 | 0 | 0 | 0 |
| 512/128 | polaris_no_pressure | 48155.98 | 757.68 | 24456.83 | 0 | 0 | 806 | 0 |
| 512/128 | polaris_pressure | 7416.72 | 184.08 | 3800.40 | 1166 | 1161 | 8172 | 0 |
| 1024/128 | native_cuda | 42377.27 | 930.74 | 21654.00 | 0 | 0 | 0 | 0 |
| 1024/128 | polaris_no_pressure | 41424.41 | 756.35 | 21090.38 | 0 | 0 | 1856 | 0 |
| 1024/128 | polaris_pressure | 2408.31 | 182.69 | 1295.50 | 1220 | 1209 | 43931 | 0 |

Interpretation:

- POLARIS no-pressure exercises fault-backed KV mapping without offload/reload.
- POLARIS pressure exercises real KV spill/reload under constrained budget.
- All rows complete with `uvm_errors = 0`.
- Pressure-mode decode throughput drops sharply on these early SmolLM2 runs
  because offload/reload churn is on the hot path.

## Cross-Engine Context

This table should be read as system-level context, not as an allocator A/B.

| workload | system | benchmark path | POLARIS KV? | request shape | prompt/input tok/s | decode/output tok/s | aggregate metric | aggregate metric name |
|---:|---|---|---|---:|---:|---:|---:|---|
| 128/32 | llama.cpp native_cuda | llama-bench | no | 1 bench workload | 1041.51 | 815.75 | 928.63 | llama_avg_ts |
| 128/32 | llama.cpp polaris_no_pressure | llama-bench | yes | 1 bench workload | 599.83 | 724.73 | 662.28 | llama_avg_ts |
| 128/32 | llama.cpp polaris_pressure | llama-bench | yes | 1 bench workload | 941.25 | 177.71 | 559.48 | llama_avg_ts |
| 128/32 | vLLM original/clean | offline batch | no | 16 | 3949.64 | 987.41 | 4937.05 | total tok/s |
| 128/32 | SGLang original | offline batch | no | 16 | 1591.40 | 397.85 | 1989.25 | total tok/s |
| 512/128 | llama.cpp native_cuda | llama-bench | no | 1 bench workload | 36281.44 | 930.33 | 18605.88 | llama_avg_ts |
| 512/128 | llama.cpp polaris_no_pressure | llama-bench | yes | 1 bench workload | 48155.98 | 757.68 | 24456.83 | llama_avg_ts |
| 512/128 | llama.cpp polaris_pressure | llama-bench | yes | 1 bench workload | 7416.72 | 184.08 | 3800.40 | llama_avg_ts |
| 512/128 | vLLM original | offline batch | no | 16 | 8286.50 | 2071.63 | 10358.13 | total tok/s |
| 512/128 | SGLang original | offline batch | no | 16 | 8333.23 | 2083.31 | 10416.54 | total tok/s |
| 1024/128 | llama.cpp native_cuda | llama-bench | no | 1 bench workload | 42377.27 | 930.74 | 21654.00 | llama_avg_ts |
| 1024/128 | llama.cpp polaris_no_pressure | llama-bench | yes | 1 bench workload | 41424.41 | 756.35 | 21090.38 | llama_avg_ts |
| 1024/128 | llama.cpp polaris_pressure | llama-bench | yes | 1 bench workload | 2408.31 | 182.69 | 1295.50 | llama_avg_ts |
| 1024/128 | vLLM original | offline batch | no | 16 | 15988.28 | 1998.53 | 17986.81 | total tok/s |
| 1024/128 | SGLang original | offline batch | no | 16 | 16039.16 | 2004.89 | 18044.05 | total tok/s |

Interpretation:

- Native vLLM/SGLang provide a throughput envelope for common serving stacks on
  the same hardware and similar token shapes.
- The framework rows use native batching and native KV allocators; they should
  not be used to isolate POLARIS KV overhead.
- The llama.cpp POLARIS rows show that the injected KV-only path is functional
  and can be evaluated against native llama.cpp under the same engine.
- Decode/output throughput is the most useful cross-engine column, but still
  remains affected by different batching and runtime behavior.

## Recommended Wording

Use this wording in project reports:

> POLARIS is evaluated in a controlled same-engine llama.cpp native-vs-POLARIS
> configuration. To contextualize end-to-end throughput against widely used
> serving stacks, we also report native vLLM and native SGLang baselines on the
> same GPU and similar token shapes. These framework baselines do not route KV
> cache through POLARIS and are not same-engine KV-backend A/B comparisons.

Avoid this wording:

> POLARIS outperforms/underperforms vLLM or SGLang.

That statement is not supported by the current artifacts because vLLM and
SGLang do not yet have live POLARIS allocator/backend integration.

## Artifacts

- llama.cpp 128/32:
  `benchmarks/results/llama_cpp/llama-polaris-128x32-20260616T1325Z/`
- llama.cpp 512/128:
  `benchmarks/results/llama_cpp/llama-polaris-512x128-r3-20260616T1336Z/`
- llama.cpp 1024/128:
  `benchmarks/results/llama_cpp/llama-polaris-1024x128-r3-20260616T1340Z/`
- vLLM clean 128/32:
  `benchmarks/results/frameworks/original-vllm-clean-128x32-20260616T1327Z/`
- vLLM/SGLang 128/32:
  `benchmarks/results/frameworks/original-128x32-20260616T1324Z/`
- vLLM/SGLang 512/128:
  `benchmarks/results/frameworks/original-512x128-r3-20260616T1334Z/`
- vLLM/SGLang 1024/128:
  `benchmarks/results/frameworks/original-1024x128-r3-20260616T1338Z/`
