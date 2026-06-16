# llama.cpp + POLARIS vs Original Framework Trend - 2026-06-16

This report extends the first `128/32` comparison with longer contexts and
repeat samples.

## Scope

- Model family: SmolLM2-135M-Instruct
- llama.cpp model: `/home/wano/workspace/models/SmolLM2-135M-Instruct-F16.gguf`
- vLLM/SGLang model: `/home/wano/workspace/models/SmolLM2-135M-Instruct`
- GPU: NVIDIA GeForce RTX 5070 Ti, driver 610.43.02
- POLARIS branch: `v4-fault-hook`
- Framework baselines: original/unmodified runtime paths, `FRAMEWORK_BENCH_TRACE=0`
- Framework long-context runs: one warmup repetition, three measured repetitions
- llama.cpp long-context runs: `llama-bench -r 3`

This is an end-to-end system comparison across different engines and model
formats. It should not be interpreted as a pure KV-manager A/B test.

## Original Framework Baselines

| workload | framework | requests/s | total tok/s mean | total tok/s stddev | measured reps | warmup reps |
|---|---|---:|---:|---:|---:|---:|
| 128/32 | vLLM | 30.86 | 4937.05 | 0.00 | 1 | 0 |
| 128/32 | SGLang | 12.43 | 1989.25 | 0.00 | 1 | 0 |
| 512/128 | vLLM | 16.18 | 10358.13 | 83.44 | 3 | 1 |
| 512/128 | SGLang | 16.28 | 10416.54 | 22.10 | 3 | 1 |
| 1024/128 | vLLM | 15.61 | 17986.81 | 298.39 | 3 | 1 |
| 1024/128 | SGLang | 15.66 | 18044.05 | 33.12 | 3 | 1 |

The 512/128 and 1024/128 framework rows are stable after one in-process warmup
batch. The 128/32 row is retained from the first pass and was not re-run with
repetitions.

## llama.cpp / POLARIS

| workload | mode | prompt tok/s | prompt stddev | gen tok/s | gen stddev | llama avg tok/s |
|---|---|---:|---:|---:|---:|---:|
| 128/32 | native_cuda | 1041.51 | 0.00 | 815.75 | 0.00 | 928.63 |
| 128/32 | polaris_no_pressure | 599.83 | 0.00 | 724.73 | 0.00 | 662.28 |
| 128/32 | polaris_pressure | 941.25 | 0.00 | 177.71 | 0.00 | 559.48 |
| 512/128 | native_cuda | 36281.44 | 33183.94 | 930.33 | 6.64 | 18605.88 |
| 512/128 | polaris_no_pressure | 48155.98 | 38930.22 | 757.68 | 4.24 | 24456.83 |
| 512/128 | polaris_pressure | 7416.72 | 3677.40 | 184.08 | 0.95 | 3800.40 |
| 1024/128 | native_cuda | 42377.27 | 29842.49 | 930.74 | 11.62 | 21654.00 |
| 1024/128 | polaris_no_pressure | 41424.41 | 31891.37 | 756.35 | 2.83 | 21090.38 |
| 1024/128 | polaris_pressure | 2408.31 | 1469.53 | 182.69 | 1.51 | 1295.50 |

The llama-bench prompt rows show high variance on longer prompts. Generation
tokens/s is more stable in these runs and better reflects decode behavior.

## POLARIS Counters

| workload | mode | offloads | reloads | UVM bridge maps | UVM handled | UVM errors |
|---|---|---:|---:|---:|---:|---:|
| 128/32 | native_cuda | 0 | 0 | 0 | 0 | 0 |
| 128/32 | polaris_no_pressure | 0 | 0 | 1807 | 1804 | 0 |
| 128/32 | polaris_pressure | 95 | 93 | 782 | 779 | 0 |
| 512/128 | native_cuda | 0 | 0 | 0 | 0 | 0 |
| 512/128 | polaris_no_pressure | 0 | 0 | 806 | 800 | 0 |
| 512/128 | polaris_pressure | 1166 | 1161 | 8172 | 8166 | 0 |
| 1024/128 | native_cuda | 0 | 0 | 0 | 0 | 0 |
| 1024/128 | polaris_no_pressure | 0 | 0 | 1856 | 1844 | 0 |
| 1024/128 | polaris_pressure | 1220 | 1209 | 43931 | 43919 | 0 |

The pressure rows confirm that POLARIS is actively managing llama.cpp KV cache
under constrained GPU budget. The no-pressure rows exercise fault-backed
mapping without offload/reload. All recorded rows completed with `uvm_errors=0`.

## Artifacts

- `benchmarks/results/frameworks/original-vllm-clean-128x32-20260616T1327Z/`
- `benchmarks/results/frameworks/original-128x32-20260616T1324Z/`
- `benchmarks/results/frameworks/original-512x128-r3-20260616T1334Z/`
- `benchmarks/results/frameworks/original-1024x128-r3-20260616T1338Z/`
- `benchmarks/results/llama_cpp/llama-polaris-128x32-20260616T1325Z/`
- `benchmarks/results/llama_cpp/llama-polaris-512x128-r3-20260616T1336Z/`
- `benchmarks/results/llama_cpp/llama-polaris-1024x128-r3-20260616T1340Z/`

## Interpretation

- POLARIS correctness path is intact through 1024/128: no UVM errors, clean
  teardown, and pressure-mode offload/reload activity.
- Current pressure policy heavily impacts decode throughput: generation is
  around 178-184 tok/s under pressure for these runs.
- No-pressure POLARIS still has overhead versus native CUDA on decode, but it
  does not trigger offloads/reloads.
- vLLM/SGLang original baselines are very close on the longer token-id batches
  after warmup.

## Next Steps

- Add a single comparison report generator so these tables are derived directly
  from JSON artifacts instead of hand-curated.
- Run a pressure-budget sweep, e.g. 4 MiB, 16 MiB, 64 MiB, 256 MiB.
- Re-run 128/32 with repetitions/warmup for consistency.
- Add latency distribution or per-sample timing for framework rows beyond
  aggregate throughput.
