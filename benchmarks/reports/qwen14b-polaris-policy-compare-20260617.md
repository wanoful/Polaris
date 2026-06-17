# Qwen2.5 14B POLARIS KV Policy Comparison - 2026-06-17

## Scope

- Engine: llama.cpp with the POLARIS KV-only shim.
- Model: `/home/wano/workspace/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf`.
- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB total VRAM.
- Workload: `llama-bench -p 16384 -n 32 -r 1 --no-warmup -ngl 999 -fa 0`.
- POLARIS mode: `polaris_pressure`, GPU KV budget 2,560 MiB, CPU pool 8,192 MiB.
- Model weights stay on llama.cpp's normal CUDA allocation path. The shim selected two KV allocations totaling 3,271,557,120 bytes and passed through 9,986,195,456 bytes of non-KV allocations in the 2026-06-17 policy runs.
- Runtime requirement for these runs: `nvidia_uvm` was loaded with `uvm_enable_builtin_tests=1` so the shim could query the real UVM dispatch key for the process VA space.

## Results

| policy | prompt tok/s | gen tok/s | avg tok/s | offloads | reloads | bridge maps | UVM rejected | UVM errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 154.97 | 74.77 | 114.87 | 5308 | 5052 | 2167594 | 19 | 0 |
| LRU | 68.52 | 72.59 | 70.55 | 5308 | 5052 | 2467812 | 19 | 0 |
| Phase-aware | 69.64 | 71.95 | 70.79 | 5315 | 5059 | 2408379 | 22 | 0 |

## Artifacts

- FIFO: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget2560mib-20260616T1450Z`
- LRU: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget2560mib-lru-20260617T0310Z`
- Phase-aware: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget2560mib-phase-aware-20260617T0318Z`
- Small-model policy smokes:
  - LRU: `benchmarks/results/llama_cpp/policy-smoke-lru-20260617T0305Z`
  - Phase-aware: `benchmarks/results/llama_cpp/policy-smoke-phase-aware-20260617T0306Z`

## Interpretation

- The non-FIFO policies now run through the real llama.cpp KV fault pipeline with clean teardown and `uvm_errors=0`.
- The 2026-06-17 LRU and phase-aware runs did not improve the 2,560 MiB pressure point. They produced almost the same offload/reload counts as FIFO, with lower prompt throughput in this single run.
- This means the current selectable policies are operational but not yet the final workload-aware VA residency strategy. The present LRU path is mostly timestamp ordering, and phase-aware uses a simple static score. Neither policy understands the prefill access wave well enough to protect soon-to-be-reused KV blocks.
- The next optimization target should be token/phase-aware residency, for example protecting the active prefill frontier and decode-recent window, rather than broadening POLARIS into arbitrary CUDA buffer paging.
