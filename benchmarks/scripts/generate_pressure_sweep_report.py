#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Generate a llama.cpp/POLARIS pressure-budget sweep report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunInput:
    label: str
    path: Path


def parse_run_input(value: str) -> RunInput:
    try:
        label, path_s = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--run must be formatted as LABEL:DIR") from exc
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("LABEL must not be empty")
    return RunInput(label=label, path=Path(path_s))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError as exc:
        raise SystemExit(f"missing JSONL artifact: {path}") from exc
    return [json.loads(line) for line in lines if line.strip()]


def stat_delta(record: dict[str, Any], key: str) -> int:
    stats = record.get("stats_delta")
    if not isinstance(stats, dict):
        return 0
    try:
        return int(stats.get(key, 0))
    except (TypeError, ValueError):
        return 0


def policy_bytes(record: dict[str, Any], key: str) -> int:
    policy = record.get("policy")
    if not isinstance(policy, dict):
        return 0
    try:
        return int(policy.get(key, 0))
    except (TypeError, ValueError):
        return 0


def workload_value(record: dict[str, Any], key: str) -> int:
    workload = record.get("workload")
    if not isinstance(workload, dict):
        return 0
    try:
        return int(workload.get(key, 0))
    except (TypeError, ValueError):
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


def budget_label(record: dict[str, Any], fallback: str) -> str:
    budget = policy_bytes(record, "gpu_budget_bytes")
    if budget <= 0:
        return fallback
    mib = budget / 1024 / 1024
    if mib.is_integer():
        return f"{int(mib)} MiB"
    return f"{mib:.1f} MiB"


def sort_key(item: tuple[RunInput, dict[str, Any]]) -> tuple[int, int, int, str]:
    _, record = item
    return (
        workload_value(record, "prompt_tokens"),
        workload_value(record, "gen_tokens"),
        policy_bytes(record, "gpu_budget_bytes"),
        str(record.get("mode", "")),
    )


def collect_records(runs: list[RunInput]) -> list[tuple[RunInput, dict[str, Any]]]:
    out: list[tuple[RunInput, dict[str, Any]]] = []
    for run in runs:
        for record in load_jsonl(run.path / "runs.jsonl"):
            if record.get("source") != "polaris":
                continue
            out.append((run, record))
    if not out:
        raise SystemExit("no POLARIS records found in sweep inputs")
    return sorted(out, key=sort_key)


def render_report(title: str, runs: list[RunInput]) -> str:
    records = collect_records(runs)
    lines: list[str] = [
        f"# {title}",
        "",
        "## Scope",
        "",
        "- Engine: llama.cpp with the POLARIS KV-only shim.",
        "- Model weights stay on llama.cpp's normal CUDA loading path.",
        "- This report isolates GPU budget sensitivity for POLARIS-managed KV cache.",
        "",
        "## Results",
        "",
        "| workload | mode | GPU budget | prompt tok/s | gen tok/s | avg tok/s | offloads | reloads | bridge maps | UVM handled | UVM errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    total_errors = 0
    for run, record in records:
        prompt = workload_value(record, "prompt_tokens")
        gen = workload_value(record, "gen_tokens")
        prompt_row = llama_result_for(record, prompt=True)
        gen_row = llama_result_for(record, prompt=False)
        errors = stat_delta(record, "uvm_errors")
        total_errors += errors
        lines.append(
            "| {workload} | {mode} | {budget} | {prompt_ts} | {gen_ts} | {avg_ts} | {offloads} | {reloads} | {bridge} | {handled} | {errors} |".format(
                workload=f"{prompt}/{gen}",
                mode=record.get("mode", ""),
                budget=budget_label(record, run.label),
                prompt_ts=fmt_float(prompt_row.get("avg_ts")),
                gen_ts=fmt_float(gen_row.get("avg_ts")),
                avg_ts=fmt_float(record.get("llama_avg_ts")),
                offloads=fmt_int(stat_delta(record, "offloads")),
                reloads=fmt_int(stat_delta(record, "reloads")),
                bridge=fmt_int(stat_delta(record, "uvm_bridge_map_calls")),
                handled=fmt_int(stat_delta(record, "uvm_handled")),
                errors=fmt_int(errors),
            )
        )

    lines.extend(
        [
            "",
            "## Fault Path Detail",
            "",
            "| workload | mode | GPU budget | UVM hook calls | UVM deferred | UVM rejected | bridge retries | bridge errors | bridge avg ns delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run, record in records:
        prompt = workload_value(record, "prompt_tokens")
        gen = workload_value(record, "gen_tokens")
        lines.append(
            "| {workload} | {mode} | {budget} | {hooks} | {deferred} | {rejected} | {retries} | {bridge_err} | {bridge_avg} |".format(
                workload=f"{prompt}/{gen}",
                mode=record.get("mode", ""),
                budget=budget_label(record, run.label),
                hooks=fmt_int(stat_delta(record, "uvm_hook_calls")),
                deferred=fmt_int(stat_delta(record, "uvm_deferred")),
                rejected=fmt_int(stat_delta(record, "uvm_rejected")),
                retries=fmt_int(stat_delta(record, "uvm_bridge_map_retry")),
                bridge_err=fmt_int(stat_delta(record, "uvm_bridge_map_err")),
                bridge_avg=fmt_int(stat_delta(record, "uvm_bridge_map_avg_ns")),
            )
        )

    lines.extend(["", "## Artifacts", ""])
    for run in runs:
        lines.append(f"- `{run.label}`: `{run.path}`")

    lines.extend(["", "## Interpretation", ""])
    if total_errors == 0:
        lines.append("- All rows completed with `uvm_errors=0`.")
    else:
        lines.append(f"- POLARIS reported {total_errors} UVM error(s); inspect logs before comparing throughput.")
    lines.extend(
        [
            "- Higher GPU budgets should reduce offload/reload churn if the working set fits.",
            "- Decode `gen tok/s` is the most useful single throughput number for this sweep.",
            "- If offloads/reloads stay high at larger budgets, the next target is policy/workload-awareness rather than raw budget size.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="llama.cpp / POLARIS Pressure Budget Sweep")
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_input,
        required=True,
        help="Existing sweep run as LABEL:DIR",
    )
    parser.add_argument("--output", type=Path, help="Optional markdown output path")
    args = parser.parse_args()

    rendered = render_report(args.title, args.run)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
