# LLM Serving Capability With and Without POLARIS Injection - 2026-06-19

This report compares llama.cpp serving-shaped inference with and without
POLARIS injection on models already present on this machine.

The comparison is intentionally scoped to the production-shaped POLARIS path:
llama.cpp KV-cache allocations are selected by the LD_PRELOAD shim and routed
through POLARIS-managed, fault-capable GPU virtual addresses. Model weights,
upload buffers, CUDA workspaces, and ordinary CUDA allocations remain on the
normal CUDA path. These numbers should not be read as a general transparent CUDA
memory-paging benchmark.

## Hardware and Software

- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB VRAM
- NVIDIA driver: 610.43.02
- Kernel: Linux 7.0.0-22-generic
- POLARIS commit: `ade73cd`
- llama.cpp commit: `0470e05`
- llama.cpp binary: `/home/wano/workspace/llama.cpp/build-polaris/bin/llama-bench`
- POLARIS eviction policy: `phase_aware`
- UVM builtin tests parameter: enabled

## Models Used

The benchmark used existing local models, with no downloads:

| model | path | file size | benchmark role |
|---|---|---:|---|
| SmolLM2-135M-Instruct F16 | `/home/wano/workspace/models/SmolLM2-135M-Instruct-F16.gguf` | 259 MiB | repeatable small-model control |
| Qwen2.5-14B-Instruct Q4_K_M | `/home/wano/workspace/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf` | 8.4 GiB | more realistic local serving base |

## Modes Compared

| mode | meaning |
|---|---|
| `native_cuda` | llama.cpp on CUDA with no POLARIS shim injection |
| `polaris_no_pressure` | LD_PRELOAD shim active; llama.cpp KV cache goes through POLARIS with a large GPU budget, so no offload/reload is expected |
| `polaris_pressure` | LD_PRELOAD shim active; llama.cpp KV cache goes through POLARIS with a 4 MiB GPU budget, forcing daemon-owned RM offload/reload |

The workload was `llama-bench -p 128 -n 32`. SmolLM2 used 3 repetitions; Qwen
used 1 repetition to keep the larger-model run short. `llama_avg_ts` is the
simple average of llama-bench prompt-processing tokens/sec and generation
tokens/sec. For serving capability, decode/generation tokens/sec is usually the
more meaningful number.

## End-to-End Results

### SmolLM2-135M-Instruct F16

| mode | prompt tok/s | decode tok/s | llama_avg_ts | decode vs native | POLARIS offloads | POLARIS reloads | UVM bridge maps | UVM errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native CUDA | 19,236.492 | 921.765 | 10,079.128 | 100.0% | 0 | 0 | 0 | 0 |
| POLARIS no pressure | 23,063.365 | 750.732 | 11,907.049 | 81.4% | 0 | 0 | 9 | 0 |
| POLARIS pressure | 12,822.422 | 196.377 | 6,509.400 | 21.3% | 293 | 291 | 300 | 0 |

Interpretation:

- POLARIS injection without pressure completed successfully and exercised the
  KV mapping path without offload/reload. Decode throughput was about 81% of
  native CUDA on this small model.
- The pressure run forced active offload/reload and remained correct at the
  kernel counter level: 293 offloads, 291 reloads, 300 bridge-map calls, and 0
  UVM errors. Decode throughput dropped to about 21% of native CUDA.
- Prompt throughput is noisy at this size because the first sample includes
  setup and warmup effects. Decode throughput is much more stable across
  samples and is the better serving-side signal.

### Qwen2.5-14B-Instruct Q4_K_M

| mode | prompt tok/s | decode tok/s | llama_avg_ts | decode vs native | POLARIS offloads | POLARIS reloads | UVM bridge maps | UVM errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native CUDA | 816.202 | 76.810 | 446.506 | 100.0% | 0 | 0 | 0 | 0 |
| POLARIS no pressure | 826.936 | 75.646 | 451.291 | 98.5% | 0 | 0 | 72 | 0 |
| POLARIS pressure | 774.684 | 22.829 | 398.757 | 29.7% | 788 | 744 | 816 | 0 |

Interpretation:

- On the larger local model, POLARIS injection without pressure was effectively
  at native decode speed for this short run: 75.646 decode tok/s versus 76.810
  native decode tok/s.
- Under a deliberately tiny 4 MiB GPU budget, the run exercised heavy
  offload/reload: 788 offloads, 744 reloads, 816 bridge-map calls, and 0 UVM
  errors. Decode throughput dropped to 22.829 tok/s, about 30% of native.
- The pressure result is a capability result more than a tuned performance
  result. It confirms that llama.cpp KV execution can continue with POLARIS
  daemon-owned RM backing under severe KV residency pressure, but the current
  copy/reload path is the throughput limiter.

## Capability Summary

POLARIS injection is functional for the supported llama.cpp KV-cache-only path
on both downloaded local models. In no-pressure mode, the shim routes KV-cache
allocation through POLARIS while leaving model weights and ordinary CUDA buffers
on CUDA, and the serving-shaped workload completes with no UVM errors.

The serving cost depends strongly on whether the workload actually pressures
the POLARIS KV residency budget:

- Without pressure, the larger Qwen run showed near-native decode throughput
  for this short workload, while the smaller SmolLM2 run showed measurable
  overhead in decode throughput.
- With a 4 MiB pressure budget, both models continued to run and exercised
  offload/reload successfully, but decode throughput fell sharply. This is
  expected for the current implementation because pressure runs spend much of
  their time in daemon-driven RM copy, offload, and reload stages.

The current conclusion is therefore:

POLARIS currently demonstrates LLM serving viability for llama.cpp KV-cache
injection, especially in the no-pressure case. Under forced memory pressure, it
demonstrates correctness and continuity of service with daemon-owned RM backing,
but not native-like decode throughput. Further serving-performance work should
focus on reducing offload/reload frequency and RM copy overhead before treating
pressure mode as a throughput-competitive serving configuration.

## Raw Artifacts

Fresh benchmark artifacts:

- SmolLM2 run:
  `benchmarks/results/llama_cpp/serving-injection-smollm2-20260619T0835Z/`
- Qwen2.5-14B run:
  `benchmarks/results/llama_cpp/serving-injection-qwen14b-20260619T0915Z/`

Related prior long-context Qwen artifacts:

- `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-polaris-20260616T1435Z/`
- `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-pressure4g-20260616T1438Z/`

## Reproduction

SmolLM2:

```sh
sudo env \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules \
  LLAMA_CPP_DIR=../llama.cpp \
  LLAMA_CPP_BIN=/home/wano/workspace/llama.cpp/build-polaris/bin/llama-bench \
  LLAMA_CPP_MODEL=/home/wano/workspace/models/SmolLM2-135M-Instruct-F16.gguf \
  POLARIS_BENCH_RUN_ID=serving-injection-smollm2-20260619T0835Z \
  POLARIS_BENCH_PROMPTS=128 \
  POLARIS_BENCH_GENS=32 \
  POLARIS_BENCH_REPETITIONS=3 \
  POLARIS_BENCH_MODES=native_cuda,polaris_no_pressure,polaris_pressure \
  POLARIS_BENCH_EVICTION_POLICY=phase_aware \
  POLARIS_LLAMA_KV_HINTS=1 \
  POLARIS_BENCH_RUN_TIMEOUT_SEC=180 \
  benchmarks/scripts/run_llama_kv_bench.sh
```

Qwen2.5-14B:

```sh
sudo env \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules \
  LLAMA_CPP_DIR=../llama.cpp \
  LLAMA_CPP_BIN=/home/wano/workspace/llama.cpp/build-polaris/bin/llama-bench \
  LLAMA_CPP_MODEL=/home/wano/workspace/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf \
  POLARIS_BENCH_RUN_ID=serving-injection-qwen14b-20260619T0915Z \
  POLARIS_BENCH_PROMPTS=128 \
  POLARIS_BENCH_GENS=32 \
  POLARIS_BENCH_REPETITIONS=1 \
  POLARIS_BENCH_MODES=native_cuda,polaris_no_pressure,polaris_pressure \
  POLARIS_BENCH_EVICTION_POLICY=phase_aware \
  POLARIS_LLAMA_KV_HINTS=1 \
  POLARIS_BENCH_RUN_TIMEOUT_SEC=600 \
  benchmarks/scripts/run_llama_kv_bench.sh
```
