# Qwen2.5 14B llama.cpp / POLARIS KV Benchmark - 2026-06-16

## Scope

- Model: `/home/wano/workspace/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf`
- Model SHA256: `e47ad95dad6ff848b431053b375adb5d39321290ea2c638682577dafca87c008`
- Model size in llama-bench: 8,982,142,976 bytes
- Parameters: 14,770,033,664
- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB total VRAM
- Workload: `llama-bench -p 16384 -n 32 -r 1 --no-warmup -ngl 999 -fa 0`
- POLARIS scope: KV-only shim; model weights remain on llama.cpp normal CUDA loading path

The 16k prompt is a high-VRAM workload for this 16 GiB card. It is not a direct vLLM/SGLang comparison.

## Results

| mode | POLARIS GPU budget | prompt tok/s | gen tok/s | avg tok/s | offloads | reloads | bridge maps | UVM handled | UVM errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native_cuda | native | 1310.95 | 76.48 | 693.72 | 0 | 0 | 0 | 0 | 0 |
| polaris_no_pressure | 16384 MiB | 820.32 | 75.49 | 447.91 | 0 | 0 | 147027 | 145491 | 0 |
| polaris_pressure | 8192 MiB | 832.34 | 75.52 | 453.93 | 0 | 0 | 143383 | 141847 | 0 |
| polaris_pressure | 4096 MiB | 825.18 | 75.42 | 450.30 | 0 | 0 | 148278 | 146742 | 0 |

## Fault Path Detail

| mode | POLARIS GPU budget | UVM hook calls | UVM deferred | UVM rejected | bridge retries | bridge errors |
|---|---:|---:|---:|---:|---:|---:|
| native_cuda | native | 0 | 0 | 0 | 0 | 0 |
| polaris_no_pressure | 16384 MiB | 201998 | 56504 | 3 | 1536 | 1536 |
| polaris_pressure | 8192 MiB | 197564 | 55712 | 5 | 1536 | 1536 |
| polaris_pressure | 4096 MiB | 201978 | 55228 | 8 | 1536 | 1536 |

## Artifacts

- `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-20260616T1430Z`
- `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-polaris-20260616T1435Z`
- `benchmarks/results/llama_cpp/qwen14b-q4km-16k32-pressure4g-20260616T1438Z`

## Interpretation

- The 14B Q4_K_M model runs successfully through native llama.cpp CUDA and through the POLARIS KV-only path at 16k prompt length.
- POLARIS handled a large number of UVM bridge mappings with `uvm_errors=0`, so the fault-backed KV path remains correct under this larger model/workload.
- Decode throughput is essentially unchanged in this single-repetition run: native is 76.48 tok/s, POLARIS rows are about 75.4-75.5 tok/s.
- Prompt throughput is lower on the POLARIS path: about 820-832 tok/s versus native 1310.95 tok/s.
- Even at a 4 GiB POLARIS GPU budget, this workload did not trigger offload/reload. That indicates the budget is currently applying only to POLARIS-managed KV pages and the resident KV working set for this run stayed within 4 GiB; model weights are not counted because they intentionally stay on llama.cpp normal CUDA allocations.
- A follow-up 1 GiB POLARIS budget run was attempted separately. It entered severe thrashing rather than completing in a reasonable time: after about four minutes it had exceeded 27k offloads and 26k reloads, had not emitted llama-bench JSON, and was manually stopped. That interrupted run is not included in the success table above.
- After stopping the 1 GiB stress attempt, `polaris.ko` was unloaded/reloaded to clear the intentionally interrupted session state; the module stats returned to zero.
- For a successful offload/reload benchmark on this 14B model, the next test should use a less extreme budget between 1 GiB and 4 GiB, or a shorter prompt, so the run completes while still exercising spill/reload.
