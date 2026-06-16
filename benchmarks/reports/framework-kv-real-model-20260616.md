# Framework KV Real-Model Benchmark - 2026-06-16

This report records the first real-model vLLM/SGLang benchmark collected for
POLARIS KV-cache allocator comparison.

## Scope

- Model: `/home/wano/workspace/models/SmolLM2-135M-Instruct`
- Workload: 16 requests, 128 input tokens, 32 output tokens, 256 max context
- Dtype: `float16`
- GPU memory fraction/utilization: `0.45`
- vLLM random range ratio: default `0.0`
- SGLang random range ratio: `1.0`
- GPU: NVIDIA GeForce RTX 5070 Ti, driver 610.43.02
- vLLM: `0.22.0`, source tree HEAD `3f53e2138`
- SGLang: `0.5.13.post2.dev348+g12ebb3543`, source tree HEAD `12ebb35439`

These are real-model framework throughput runs plus KV allocator traces. They
do not route vLLM or SGLang KV cache through POLARIS. POLARIS live paging is
currently implemented for the llama.cpp KV-only path.

## Throughput

| source | requests/s | input tok/s | output tok/s | total tok/s | total latency/s |
|---|---:|---:|---:|---:|---:|
| vLLM | 30.51 | 3905.84 | 976.46 | 4882.29 | 0.524 |
| SGLang | 12.58 | 1609.67 | 402.42 | 2012.09 | 1.272 |

vLLM's benchmark JSON reports total token throughput and elapsed time, not
separate input/output throughput. The input/output rows above are derived from
the fixed 2048 input tokens and 512 output tokens divided by elapsed time.

## KV Allocator Trace

| source | events | sessions | peak active sessions | reserve events | release events | total reserved blocks | peak live blocks | total reserved tokens | peak live tokens | final live blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vLLM | 320 | 16 | 16 | 144 | 144 | 144 | 121 | 2304 | 1936 | 0 |
| SGLang | 832 | 16 | 16 | 640 | 160 | 640 | 640 | 2560 | 2560 | 0 |

The trace format uses 16-token logical comparison blocks. vLLM's native cache
manager already works in block units. SGLang allocates token slots, so the patch
emits 16-token logical chunks for prefill/decode reservations and releases.

## Commands

The vLLM run used:

```sh
PATH=/tmp/polaris-bench-vllm/bin:$PATH \
POLARIS_TRACE=/home/wano/workspace/Polaris/benchmarks/results/frameworks/manual-20260616T1143Z/vllm/bench_128x32_trace.csv \
/tmp/polaris-bench-vllm/bin/vllm bench throughput \
  --model /home/wano/workspace/models/SmolLM2-135M-Instruct \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 32 \
  --num-prompts 16 \
  --max-model-len 256 \
  --dtype float16 \
  --gpu-memory-utilization 0.45 \
  --enforce-eager \
  --output-json benchmarks/results/frameworks/manual-20260616T1143Z/vllm/bench_128x32_result.json
```

The SGLang run used:

```sh
PATH=/home/wano/workspace/.bench-venvs/sglang/bin:$PATH \
POLARIS_TRACE=/home/wano/workspace/Polaris/benchmarks/results/frameworks/manual-20260616T1143Z/sglang/bench_128x32_trace.csv \
/home/wano/workspace/.bench-venvs/sglang/bin/python -m sglang.bench_offline_throughput \
  --model-path /home/wano/workspace/models/SmolLM2-135M-Instruct \
  --dataset-name random-ids \
  --random-input-len 128 \
  --random-output-len 32 \
  --random-range-ratio 1.0 \
  --num-prompts 16 \
  --context-length 256 \
  --dtype float16 \
  --mem-fraction-static 0.45 \
  --attention-backend flashinfer \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --skip-warmup \
  --tokenize-prompt \
  --result-filename benchmarks/results/frameworks/manual-20260616T1143Z/sglang/bench_128x32_result.jsonl
```

For new runs, prefer `benchmarks/scripts/run_framework_kv_bench.sh`. The script
uses the non-deprecated SGLang CUDA graph flags:
`--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled`.

## Notes

- SGLang's plain `random` dataset can produce text whose tokenizer length drifts
  from the requested length. The benchmark patch adds `random-ids` to the
  offline benchmark CLI and `--tokenize-prompt` so generated ids are passed as
  `input_ids`.
- The tested vLLM `--random-range-ratio` semantics differ from SGLang. vLLM
  requires values in `[0, 1)` and defaults to `0.0`; SGLang uses `1.0` for the
  fixed-length `random-ids` run.
- vLLM/SGLang rows are allocator trace comparisons. They are not POLARIS live
  KV backends and should not be presented as POLARIS offload/reload results.
- Raw artifacts remain under
  `benchmarks/results/frameworks/manual-20260616T1143Z/` in this workspace.
