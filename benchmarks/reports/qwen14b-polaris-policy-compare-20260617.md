# Qwen2.5 14B POLARIS KV Policy Comparison - 2026-06-17

## Scope

- Engine: llama.cpp with the POLARIS KV-only shim.
- Model: `/home/wano/workspace/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf`.
- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB total VRAM.
- Workload: `llama-bench -p 16384 -n 32 -r 1 --no-warmup -ngl 999 -fa 0`.
- POLARIS mode: `polaris_pressure`, GPU KV budget 2,560 MiB, CPU pool 8,192 MiB.
- Model weights stay on llama.cpp's normal CUDA allocation path. The shim selected two KV allocations totaling 3,271,557,120 bytes and passed through 9,986,195,456 bytes of non-KV allocations in the 2026-06-17 policy runs.
- Runtime requirement for these runs: `nvidia_uvm` was loaded with `uvm_enable_builtin_tests=1` so the shim could query the real UVM dispatch key for the process VA space.
- Important metric note: the kernel `/sys/kernel/polaris/stats` offload/reload counters are global module counters and can produce misleading per-run deltas when benchmark runs are chained without reloading the module. For policy comparison, offload/reload counts below are counted from each run's `polarisd.log` decision stream. `benchmarks/scripts/run_llama_kv_bench.sh` now records these as `polarisd_decisions`.

## Results

| policy | prompt tok/s | gen tok/s | avg tok/s | offloads | reloads | bridge maps | UVM rejected | UVM errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIFO historical | 154.97 | 74.77 | 114.87 | 5308 | 5052 | 2166058 | 19 | 0 |
| LRU | 68.52 | 72.59 | 70.55 | 5308 | 5052 | 2466276 | 19 | 0 |
| Phase-aware old | 69.64 | 71.95 | 70.79 | 5315 | 5059 | 2406843 | 22 | 0 |
| Phase window experiment | 119.68 | 72.47 | 96.07 | 3119 | 2863 | 1258723 | 13 | 0 |
| Phase locality experiment | 70.40 | 71.90 | 71.15 | 5308 | 5052 | 2388584 | 17 | 0 |
| Phase FIFO fast-path | 67.48 | 71.79 | 69.63 | 5308 | 5052 | 2508724 | 15 | 0 |
| FIFO rerun, same module | 67.54 | 71.92 | 69.73 | 5308 | 5052 | 2449474 | 16 | 0 |

## Current Code State

- `phase_aware` now has a conservative fast-path for the current llama.cpp shim shape: if all eligible victims are private `Prefill` KV chunks, it delegates victim selection to FIFO map-time ordering. This avoids treating `token_start` as semantic LLM token position; in the current shim, `token_start` is a Polaris chunk index.
- If the workload later exposes real decode/shared/multi-session signals, `phase_aware` falls back to the scoring path: active chunk neighborhood protection, decode recency, sharing/refcount protection, session priority, and pressure.
- The experimental active-window policy did reduce offload/reload and bridge-map counts, but it did not beat FIFO throughput on this Qwen14B prefill-heavy benchmark. It remains useful evidence that fewer migrations alone is not enough; the stall placement matters.
- The same-module FIFO rerun and phase FIFO fast-path run are effectively equivalent in throughput and decision counts. The earlier 154.97 tok/s FIFO historical result should be treated as a previous-run baseline, not as a guaranteed reproducible value for every module/runtime state.

## Artifacts

- FIFO: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget2560mib-20260616T1450Z`
- LRU: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget2560mib-lru-20260617T0310Z`
- Phase-aware: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget2560mib-phase-aware-20260617T0318Z`
- Phase window experiment: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget2560mib-phase-window-20260617T0325Z`
- Phase locality experiment: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget2560mib-phase-locality-20260617T0335Z`
- Phase FIFO fast-path: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget2560mib-phase-fifo-fastpath-20260617T0345Z`
- FIFO rerun, same module: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget2560mib-fifo-rerun-20260617T0355Z`
- Small-model policy smokes:
  - LRU: `benchmarks/results/llama_cpp/policy-smoke-lru-20260617T0305Z`
  - Phase-aware: `benchmarks/results/llama_cpp/policy-smoke-phase-aware-20260617T0306Z`
  - Phase-aware fast-path and runner decision counts: `benchmarks/results/llama_cpp/policy-runner-decision-count-smoke-20260617T0405Z`

## Interpretation

- The non-FIFO policies now run through the real llama.cpp KV fault pipeline with clean teardown and `uvm_errors=0`.
- Model weights are still not POLARIS-managed. The shim's selected bytes correspond to KV allocations; the roughly 9.99 GB of non-KV/model allocations are passed through to llama.cpp's original CUDA path.
- The current llama.cpp integration is KV-only and operational: Qwen2.5-14B Q4_K_M runs end-to-end with 3.0 GiB KV reserved, 2.5 GiB/ GPU POLARIS residency pressure, daemon RM backing, offload/reload, and `uvm_errors=0`.
- `token_start` currently means Polaris chunk index, not model token index. A true workload-aware policy needs either better shim/runtime hints or a separate access-tracking surface before it can reason about prefill/decode token reuse precisely.
- The next optimization target should stay KV-specific: expose real KV access phase/range hints from the llama.cpp shim, then use those hints to protect the active prefill/decode working set. This is not a reason to broaden POLARIS into arbitrary CUDA buffer transparent paging.
