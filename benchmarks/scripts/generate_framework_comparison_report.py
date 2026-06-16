#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Generate llama.cpp/POLARIS vs original framework benchmark reports.

The input is existing benchmark artifact directories. This script does not run
benchmarks and does not patch vLLM/SGLang. It keeps the comparison label honest:
llama.cpp/POLARIS is compared end to end against original vLLM/SGLang runtime
paths, not as a same-engine KV-manager A/B test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workload:
    prompt: int
    gen: int

    @property
    def label(self) -> str:
        return f"{self.prompt}/{self.gen}"

    @property
    def file_label(self) -> str:
        return f"{self.prompt}x{self.gen}"


@dataclass(frozen=True)
class LlamaRun:
    workload: Workload
    path: Path


@dataclass(frozen=True)
class FrameworkRun:
    workload: Workload
    source: str
    path: Path


def parse_workload(value: str) -> Workload:
    if "/" in value:
        prompt_s, gen_s = value.split("/", 1)
    elif "x" in value:
        prompt_s, gen_s = value.split("x", 1)
    else:
        raise argparse.ArgumentTypeError(
            f"{value!r} must be formatted as PROMPT/GEN or PROMPTxGEN"
        )
    try:
        prompt = int(prompt_s)
        gen = int(gen_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} contains a non-integer token count"
        ) from exc
    if prompt <= 0 or gen <= 0:
        raise argparse.ArgumentTypeError(
            f"{value!r} must contain positive token counts"
        )
    return Workload(prompt=prompt, gen=gen)


def parse_llama_run(value: str) -> LlamaRun:
    try:
        workload_s, path_s = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--llama-run must be formatted as WORKLOAD:DIR"
        ) from exc
    return LlamaRun(workload=parse_workload(workload_s), path=Path(path_s))


def parse_framework_run(value: str) -> FrameworkRun:
    try:
        workload_s, source, path_s = value.split(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--framework-run must be formatted as WORKLOAD:SOURCE:DIR"
        ) from exc
    source = source.strip().lower()
    if source not in {"vllm", "sglang"}:
        raise argparse.ArgumentTypeError("SOURCE must be vllm or sglang")
    return FrameworkRun(
        workload=parse_workload(workload_s),
        source=source,
        path=Path(path_s),
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"missing JSON artifact: {path}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError as exc:
        raise SystemExit(f"missing JSONL artifact: {path}") from exc
    return [json.loads(line) for line in lines if line.strip()]


def fmt_float(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def stat_delta(record: dict[str, Any], key: str) -> int:
    stats = record.get("stats_delta")
    if isinstance(stats, dict):
        value = stats.get(key, 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def llama_result_for(record: dict[str, Any], *, prompt: bool) -> dict[str, Any]:
    for row in record.get("llama_results", []):
        if not isinstance(row, dict):
            continue
        n_prompt = int(row.get("n_prompt") or 0)
        n_gen = int(row.get("n_gen") or 0)
        if prompt and n_prompt > 0:
            return row
        if not prompt and n_gen > 0:
            return row
    return {}


def load_llama_records(run: LlamaRun) -> list[dict[str, Any]]:
    records = load_jsonl(run.path / "runs.jsonl")
    out: list[dict[str, Any]] = []
    for record in records:
        workload = record.get("workload", {})
        if not isinstance(workload, dict):
            continue
        if (
            int(workload.get("prompt_tokens") or 0) == run.workload.prompt
            and int(workload.get("gen_tokens") or 0) == run.workload.gen
        ):
            out.append(record)
    if not out:
        raise SystemExit(
            f"{run.path}: no llama records for workload {run.workload.label}"
        )
    return out


def load_framework_record(run: FrameworkRun) -> dict[str, Any]:
    path = run.path / run.source / f"bench_{run.workload.file_label}.json"
    record = load_json(path)
    if record.get("source") and record.get("source") != run.source:
        raise SystemExit(f"{path}: source is {record.get('source')!r}, expected {run.source!r}")
    return record


def artifact_path(path: Path) -> str:
    return str(path)


def infer_scope(framework_records: list[dict[str, Any]]) -> list[str]:
    warmups = [
        int(record.get("warmup_repetitions") or 0)
        for record in framework_records
        if "warmup_repetitions" in record
    ]
    reps = [
        int(record.get("repetitions") or 1)
        for record in framework_records
        if "repetitions" in record
    ]
    lines = [
        "This is an end-to-end system comparison across different engines and model formats.",
        "It should not be interpreted as a pure KV-manager A/B test.",
    ]
    if reps:
        lines.append(
            "Framework repeated runs report the mean and sample standard deviation from the JSON artifacts."
        )
    if warmups:
        lines.append(
            "Framework warmup repetitions are initialization-local batches and are excluded from measured means."
        )
    return lines


def render_report(
    title: str,
    llama_runs: list[LlamaRun],
    framework_runs: list[FrameworkRun],
) -> str:
    llama_by_workload = {run.workload: load_llama_records(run) for run in llama_runs}
    framework_records: dict[tuple[Workload, str], tuple[FrameworkRun, dict[str, Any]]] = {
        (run.workload, run.source): (run, load_framework_record(run))
        for run in framework_runs
    }
    workloads = sorted(
        set(llama_by_workload) | {run.workload for run in framework_runs},
        key=lambda workload: (workload.prompt, workload.gen),
    )

    flat_framework_records = [record for _, record in framework_records.values()]
    lines: list[str] = [
        f"# {title}",
        "",
        "## Scope",
        "",
    ]

    model_names = sorted(
        {
            str(record.get("model"))
            for record in flat_framework_records
            if record.get("model")
        }
    )
    llama_models = sorted(
        {
            str(record.get("model"))
            for records in llama_by_workload.values()
            for record in records
            if record.get("model")
        }
    )
    if llama_models:
        lines.append(f"- llama.cpp model: `{llama_models[0]}`")
    if model_names:
        lines.append(f"- vLLM/SGLang model: `{model_names[0]}`")

    gpu_info = ""
    for records in llama_by_workload.values():
        for record in records:
            for row in record.get("llama_results", []):
                if isinstance(row, dict) and row.get("gpu_info"):
                    gpu_info = str(row["gpu_info"])
                    break
            if gpu_info:
                break
        if gpu_info:
            break
    if gpu_info:
        lines.append(f"- GPU: {gpu_info}")
    lines.extend(
        [
            "- Framework baselines: original/unmodified runtime paths, `FRAMEWORK_BENCH_TRACE=0`",
            "- llama.cpp/POLARIS path: KV-only shim, model weights use llama.cpp's normal loading path",
            "",
        ]
    )
    lines.extend(infer_scope(flat_framework_records))

    lines.extend(
        [
            "",
            "## Original Framework Baselines",
            "",
            "| workload | framework | requests/s | total tok/s mean | total tok/s stddev | measured reps | warmup reps |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for workload in workloads:
        for source in ("vllm", "sglang"):
            item = framework_records.get((workload, source))
            if not item:
                continue
            _, record = item
            lines.append(
                "| {workload} | {source} | {rps} | {total} | {stddev} | {reps} | {warmup} |".format(
                    workload=workload.label,
                    source="vLLM" if source == "vllm" else "SGLang",
                    rps=fmt_float(record.get("requests_per_second")),
                    total=fmt_float(record.get("total_tokens_per_second")),
                    stddev=fmt_float(record.get("total_tokens_per_second_stddev", 0.0)),
                    reps=fmt_int(record.get("repetitions", 1)),
                    warmup=fmt_int(record.get("warmup_repetitions", 0)),
                )
            )

    lines.extend(
        [
            "",
            "## llama.cpp / POLARIS",
            "",
            "| workload | mode | prompt tok/s | prompt stddev | gen tok/s | gen stddev | llama avg tok/s |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for workload in workloads:
        for record in llama_by_workload.get(workload, []):
            prompt_row = llama_result_for(record, prompt=True)
            gen_row = llama_result_for(record, prompt=False)
            lines.append(
                "| {workload} | {mode} | {prompt_ts} | {prompt_stddev} | {gen_ts} | {gen_stddev} | {avg_ts} |".format(
                    workload=workload.label,
                    mode=record.get("mode", ""),
                    prompt_ts=fmt_float(prompt_row.get("avg_ts")),
                    prompt_stddev=fmt_float(prompt_row.get("stddev_ts")),
                    gen_ts=fmt_float(gen_row.get("avg_ts")),
                    gen_stddev=fmt_float(gen_row.get("stddev_ts")),
                    avg_ts=fmt_float(record.get("llama_avg_ts")),
                )
            )

    lines.extend(
        [
            "",
            "## POLARIS Counters",
            "",
            "| workload | mode | offloads | reloads | UVM bridge maps | UVM handled | UVM errors |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    all_errors = 0
    pressure_has_activity = False
    for workload in workloads:
        for record in llama_by_workload.get(workload, []):
            errors = stat_delta(record, "uvm_errors")
            all_errors += errors
            offloads = stat_delta(record, "offloads")
            reloads = stat_delta(record, "reloads")
            if record.get("mode") == "polaris_pressure" and (offloads or reloads):
                pressure_has_activity = True
            lines.append(
                "| {workload} | {mode} | {offloads} | {reloads} | {bridge} | {handled} | {errors} |".format(
                    workload=workload.label,
                    mode=record.get("mode", ""),
                    offloads=offloads,
                    reloads=reloads,
                    bridge=stat_delta(record, "uvm_bridge_map_calls"),
                    handled=stat_delta(record, "uvm_handled"),
                    errors=errors,
                )
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    artifact_dirs = []
    artifact_dirs.extend(run.path for run in llama_runs)
    artifact_dirs.extend(run.path for run in framework_runs)
    for path in sorted({artifact_path(path) for path in artifact_dirs}):
        lines.append(f"- `{path}`")

    lines.extend(["", "## Interpretation", ""])
    if all_errors == 0:
        lines.append("- POLARIS correctness counters are clean in these rows: `uvm_errors=0`.")
    else:
        lines.append(f"- POLARIS reported {all_errors} UVM error(s); inspect the run logs.")
    if pressure_has_activity:
        lines.append(
            "- Pressure-mode POLARIS rows triggered real offload/reload activity."
        )
    lines.extend(
        [
            "- No-pressure POLARIS rows exercise fault-backed mapping without intentional GPU-budget pressure.",
            "- Framework rows are original vLLM/SGLang baselines; they do not route KV cache through POLARIS.",
            "- Cross-engine throughput is useful for system context, but it is not a direct KV backend speedup claim.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="llama.cpp + POLARIS vs Original Frameworks")
    parser.add_argument(
        "--llama-run",
        action="append",
        type=parse_llama_run,
        required=True,
        help="Existing llama.cpp/POLARIS artifact as WORKLOAD:DIR",
    )
    parser.add_argument(
        "--framework-run",
        action="append",
        type=parse_framework_run,
        required=True,
        help="Existing vLLM/SGLang artifact as WORKLOAD:SOURCE:DIR",
    )
    parser.add_argument("--output", type=Path, help="Optional markdown output path")
    args = parser.parse_args()

    rendered = render_report(args.title, args.llama_run, args.framework_run)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
