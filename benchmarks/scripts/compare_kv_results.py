#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Build a compact KV-cache comparison table from benchmark JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def get(record: dict[str, Any], key: str, default: Any = 0) -> Any:
    cur: Any = record
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def first_present(record: dict[str, Any], keys: list[str], default: Any = 0) -> Any:
    for key in keys:
        value = get(record, key, None)
        if value is not None:
            return value
    return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Benchmark JSONL file")
    parser.add_argument("--output", type=Path, help="Optional markdown output path")
    args = parser.parse_args()

    records = load_records(args.jsonl)
    rows = [
        "| source | mode | scope | prompt | gen | avg_ts | offloads | reloads | bridge_maps | peak_live_blocks | total_reserved_blocks | uvm_errors |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rec in records:
        source = rec.get("source", "polaris")
        mode = rec.get("mode", "-")
        scope = rec.get("comparison_scope", "end_to_end")
        prompt = get(rec, "workload.prompt_tokens", rec.get("n_prompt", 0))
        gen = get(rec, "workload.gen_tokens", rec.get("n_gen", 0))
        avg_ts = rec.get("llama_avg_ts", "")
        if isinstance(avg_ts, float):
            avg_ts_s = f"{avg_ts:.3f}"
        else:
            avg_ts_s = str(avg_ts)
        rows.append(
            "| {source} | {mode} | {scope} | {prompt} | {gen} | {avg_ts} | {offloads} | {reloads} | {bridge} | {peak} | {reserved} | {errors} |".format(
                source=source,
                mode=mode,
                scope=scope,
                prompt=prompt,
                gen=gen,
                avg_ts=avg_ts_s,
                offloads=first_present(rec, ["polarisd_decisions.offloads", "stats_delta.offloads"], rec.get("offloads", 0)),
                reloads=first_present(rec, ["polarisd_decisions.reloads", "stats_delta.reloads"], rec.get("reloads", 0)),
                bridge=get(rec, "stats_delta.uvm_bridge_map_calls", rec.get("uvm_bridge_map_calls", 0)),
                peak=rec.get("peak_live_blocks", ""),
                reserved=rec.get("total_reserved_blocks", ""),
                errors=get(rec, "stats_delta.uvm_errors", rec.get("uvm_errors", 0)),
            )
        )

    rendered = "\n".join(rows) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
