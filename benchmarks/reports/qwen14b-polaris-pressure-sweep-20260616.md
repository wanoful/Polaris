# Qwen2.5 14B POLARIS KV Pressure Sweep - 2026-06-16

## Scope

- Engine: llama.cpp with the POLARIS KV-only shim.
- Model: `/home/wano/workspace/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf`.
- Model size in llama-bench: 8,982,142,976 bytes; parameters: 14,770,033,664.
- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB total VRAM.
- Workload: `llama-bench -p 16384 -n 32 -r 1 --no-warmup -ngl 999 -fa 0`.
- Model weights stay on llama.cpp's normal CUDA loading path.
- This report isolates GPU budget sensitivity for POLARIS-managed KV cache.

## Results

| workload | mode | GPU budget | prompt tok/s | gen tok/s | avg tok/s | offloads | reloads | bridge maps | UVM handled | UVM errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16384/32 | polaris_pressure | 2048 MiB | 71.76 | 75.24 | 73.50 | 12956 | 12444 | 4925437 | 4923901 | 0 |
| 16384/32 | polaris_pressure | 2560 MiB | 154.97 | 74.77 | 114.87 | 5308 | 5052 | 2167594 | 2166058 | 0 |
| 16384/32 | polaris_pressure | 3072 MiB | 832.67 | 75.63 | 454.15 | 0 | 0 | 145883 | 144347 | 0 |
| 16384/32 | polaris_pressure | 4096 MiB | 825.18 | 75.42 | 450.30 | 0 | 0 | 148278 | 146742 | 0 |

## Fault Path Detail

| workload | mode | GPU budget | UVM hook calls | UVM deferred | UVM rejected | bridge retries | bridge errors | bridge avg ns delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 16384/32 | polaris_pressure | 2048 MiB | 8443651 | 3519725 | 25 | 1536 | 1536 | 51 |
| 16384/32 | polaris_pressure | 2560 MiB | 3668454 | 1502377 | 19 | 1536 | 1536 | -5913 |
| 16384/32 | polaris_pressure | 3072 MiB | 189339 | 44989 | 3 | 1536 | 1536 | -19077 |
| 16384/32 | polaris_pressure | 4096 MiB | 201978 | 55228 | 8 | 1536 | 1536 | -486 |

## Artifacts

- `4096 MiB`: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-pressure4g-20260616T1438Z`
- `3072 MiB`: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget3072mib-20260616T1450Z`
- `2560 MiB`: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget2560mib-20260616T1450Z`
- `2048 MiB`: `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-budget2048mib-20260616T1450Z`

## Interpretation

- All rows completed with `uvm_errors=0` and clean teardown.
- The 14B/16k single-sequence working-set threshold is between 2560 MiB and 3072 MiB for the current FIFO policy: 3072 MiB has no offload/reload, while 2560 MiB triggers 5308 offloads and 5052 reloads.
- Prompt throughput is highly sensitive to spill/reload churn: 3072 MiB reaches 832.67 tok/s, 2560 MiB drops to 154.97 tok/s, and 2048 MiB drops to 71.76 tok/s.
- Decode throughput stays around 75 tok/s across this sweep because the generation segment is short and the benchmark is dominated by the long prefill pressure.
- This confirms that POLARIS can run the larger 14B model with real KV offload/reload at 2.0-2.5 GiB budgets, but the current FIFO/bounded-window strategy creates heavy prefill thrashing once the budget falls below the working set.
- The next optimization target is workload-aware residency policy around prefill/decode phase behavior, not arbitrary CUDA buffer paging.
