#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Run unmodified vLLM/SGLang offline throughput on fixed token-id prompts."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


def build_token_prompts(
    num_prompts: int, input_len: int, vocab_size: int, seed_offset: int = 100
) -> list[list[int]]:
    if input_len <= 0:
        raise SystemExit("--input-len must be positive")
    if vocab_size <= 1024:
        raise SystemExit(f"vocab_size is too small for random token prompts: {vocab_size}")

    usable = max(1, vocab_size - seed_offset - 1)
    prompts: list[list[int]] = []
    for i in range(num_prompts):
        base = seed_offset + (i * 9973) % usable
        prompts.append([seed_offset + ((base + i + j) % usable) for j in range(input_len)])
    return prompts


def count_sglang_output_tokens(row: dict[str, Any]) -> int:
    meta = row.get("meta_info") or {}
    if isinstance(meta.get("completion_tokens"), int):
        return int(meta["completion_tokens"])
    if isinstance(row.get("output_ids"), list):
        return len(row["output_ids"])
    if isinstance(row.get("token_ids"), list):
        return len(row["token_ids"])
    return 0


def make_sample_result(
    args: argparse.Namespace,
    source: str,
    repetition: int,
    elapsed: float,
    output_tokens: int,
) -> dict[str, Any]:
    input_tokens = args.num_prompts * args.input_len
    return {
        "source": source,
        "comparison_scope": "original_framework_e2e",
        "model": args.model,
        "repetition": repetition,
        "num_requests": args.num_prompts,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "elapsed_time": elapsed,
        "requests_per_second": args.num_prompts / elapsed,
        "input_tokens_per_second": input_tokens / elapsed,
        "output_tokens_per_second": output_tokens / elapsed,
        "total_tokens_per_second": (input_tokens + output_tokens) / elapsed,
    }


def aggregate_samples(
    args: argparse.Namespace, source: str, samples: list[dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": source,
        "comparison_scope": "original_framework_e2e",
        "model": args.model,
        "num_requests": args.num_prompts,
        "repetitions": args.repetitions,
        "warmup_repetitions": args.warmup_repetitions,
        "total_input_tokens": args.num_prompts * args.input_len,
        "total_output_tokens": samples[-1]["total_output_tokens"] if samples else 0,
        "samples": samples,
    }

    metric_names = [
        "elapsed_time",
        "requests_per_second",
        "input_tokens_per_second",
        "output_tokens_per_second",
        "total_tokens_per_second",
    ]
    for name in metric_names:
        values = [float(sample[name]) for sample in samples]
        result[name] = statistics.fmean(values)
        result[f"{name}_stddev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result[f"{name}_min"] = min(values)
        result[f"{name}_max"] = max(values)
    return result


def run_vllm(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_len,
        ignore_eos=True,
    )
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )

    samples: list[dict[str, Any]] = []
    total_runs = args.warmup_repetitions + args.repetitions
    for run_idx in range(total_runs):
        prompts = build_token_prompts(
            args.num_prompts,
            args.input_len,
            args.vocab_size,
            seed_offset=100 + run_idx * 4099,
        )
        prompt_payload = [{"prompt_token_ids": p} for p in prompts]

        start = time.perf_counter()
        outputs = llm.generate(
            prompt_payload, sampling_params=sampling_params, use_tqdm=False
        )
        elapsed = time.perf_counter() - start

        output_tokens = 0
        for item in outputs:
            if item.outputs:
                output_tokens += len(item.outputs[0].token_ids)
        if run_idx < args.warmup_repetitions:
            continue
        repetition = run_idx - args.warmup_repetitions
        samples.append(
            make_sample_result(args, "vllm", repetition, elapsed, output_tokens)
        )

    return aggregate_samples(args, "vllm", samples)


def run_sglang(args: argparse.Namespace) -> dict[str, Any]:
    from sglang.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path=args.model,
        tokenizer_path=args.model,
        context_length=args.max_model_len,
        dtype=args.dtype,
        mem_fraction_static=args.gpu_memory_utilization,
        attention_backend=args.sglang_attention_backend,
        sampling_backend=args.sglang_sampling_backend,
        cuda_graph_backend_decode=args.sglang_cuda_graph_backend,
        cuda_graph_backend_prefill=args.sglang_cuda_graph_backend,
        log_level="error",
    )

    try:
        samples: list[dict[str, Any]] = []
        total_runs = args.warmup_repetitions + args.repetitions
        for run_idx in range(total_runs):
            prompts = build_token_prompts(
                args.num_prompts,
                args.input_len,
                args.vocab_size,
                seed_offset=100 + run_idx * 4099,
            )
            sampling_params = [
                {
                    "temperature": 0,
                    "max_new_tokens": args.output_len,
                    "ignore_eos": True,
                }
                for _ in prompts
            ]

            start = time.perf_counter()
            outputs = engine.generate(input_ids=prompts, sampling_params=sampling_params)
            elapsed = time.perf_counter() - start
            if isinstance(outputs, dict):
                outputs = [outputs]
            output_tokens = sum(count_sglang_output_tokens(row) for row in outputs)
            if run_idx < args.warmup_repetitions:
                continue
            repetition = run_idx - args.warmup_repetitions
            samples.append(
                make_sample_result(args, "sglang", repetition, elapsed, output_tokens)
            )
        server_info = engine.get_server_info()
    finally:
        engine.shutdown()

    result = aggregate_samples(args, "sglang", samples)
    try:
        result["last_gen_throughput"] = float(
            server_info["internal_states"][0]["last_gen_throughput"]
        )
    except Exception:
        pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=["vllm", "sglang"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--vocab-size", type=int, default=49152)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--sglang-attention-backend", default="flashinfer")
    parser.add_argument("--sglang-sampling-backend", default="pytorch")
    parser.add_argument("--sglang-cuda-graph-backend", default="disabled")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    if args.warmup_repetitions < 0:
        raise SystemExit("--warmup-repetitions must be non-negative")

    if args.source == "vllm":
        result = run_vllm(args)
    else:
        result = run_sglang(args)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
