#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Analyze POLARIS policy behavior from benchmark daemon logs.

The daemon log records every ALLOC/OFFLOAD/RELOAD/FREE decision with a block
id. That is enough to reconstruct block-level residency, offload/reload misses,
VRAM usage, and victim order without adding instrumentation to the fault hot
path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DECISION_RE = re.compile(
    r"polarisd: executing decision (?P<decision>\d+) op=(?P<op>[A-Z_]+)"
    r".*?\bblock_id=(?P<block>\d+)\b.*?\bsession_id=(?P<session>\d+)\b"
    r".*?\bsize=(?P<size>\d+)\b.*?\bsrc_vaddr=(?P<src>0x[0-9a-fA-F]+|\d+)"
    r".*?\bdst_vaddr=(?P<dst>0x[0-9a-fA-F]+|\d+)"
)


@dataclass
class Event:
    seq: int
    decision: int
    op: str
    block_id: int
    session_id: int
    size: int
    src_vaddr: int
    dst_vaddr: int
    resident_blocks_before: int
    resident_bytes_before: int
    cpu_blocks_before: int
    cpu_bytes_before: int
    state_before: str
    resident_blocks_after: int
    resident_bytes_after: int
    cpu_blocks_after: int
    cpu_bytes_after: int
    state_after: str
    is_vram_miss: int
    is_victim: int


def load_runs(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/trial-*/runs.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["_runs_path"] = str(path)
            records.append(rec)
    return records


def parse_events(log_path: Path) -> list[Event]:
    resident: dict[int, int] = {}
    cpu: dict[int, int] = {}
    events: list[Event] = []

    def resident_bytes() -> int:
        return sum(resident.values())

    def cpu_bytes() -> int:
        return sum(cpu.values())

    def state(block_id: int) -> str:
        if block_id in resident:
            return "resident"
        if block_id in cpu:
            return "cpu"
        return "absent"

    for line in log_path.read_text(errors="replace").splitlines():
        m = DECISION_RE.search(line)
        if not m:
            continue

        op = m.group("op")
        block_id = int(m.group("block"))
        size = int(m.group("size"))
        before_state = state(block_id)
        before_resident_blocks = len(resident)
        before_resident_bytes = resident_bytes()
        before_cpu_blocks = len(cpu)
        before_cpu_bytes = cpu_bytes()
        miss = 0
        victim = 0

        if op == "ALLOC":
            resident[block_id] = size
            cpu.pop(block_id, None)
        elif op == "OFFLOAD":
            victim = 1
            resident.pop(block_id, None)
            cpu[block_id] = size
        elif op == "RELOAD":
            miss = 1
            cpu.pop(block_id, None)
            resident[block_id] = size
        elif op == "FREE":
            resident.pop(block_id, None)
            cpu.pop(block_id, None)

        events.append(
            Event(
                seq=len(events) + 1,
                decision=int(m.group("decision")),
                op=op,
                block_id=block_id,
                session_id=int(m.group("session")),
                size=size,
                src_vaddr=int(m.group("src"), 0),
                dst_vaddr=int(m.group("dst"), 0),
                resident_blocks_before=before_resident_blocks,
                resident_bytes_before=before_resident_bytes,
                cpu_blocks_before=before_cpu_blocks,
                cpu_bytes_before=before_cpu_bytes,
                state_before=before_state,
                resident_blocks_after=len(resident),
                resident_bytes_after=resident_bytes(),
                cpu_blocks_after=len(cpu),
                cpu_bytes_after=cpu_bytes(),
                state_after=state(block_id),
                is_vram_miss=miss,
                is_victim=victim,
            )
        )

    return events


def write_event_csv(path: Path, events: list[Event]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Event.__dataclass_fields__))
        writer.writeheader()
        for event in events:
            writer.writerow(event.__dict__)


def summarize_record(rec: dict[str, Any], events: list[Event], policy: str) -> dict[str, Any]:
    stats_delta = rec.get("stats_delta") or {}
    decisions = rec.get("polarisd_decisions") or {}
    offloads = sum(e.is_victim for e in events)
    reloads = sum(e.is_vram_miss for e in events)
    allocs = sum(1 for e in events if e.op == "ALLOC")
    frees = sum(1 for e in events if e.op == "FREE")
    peak_resident_bytes = max((e.resident_bytes_after for e in events), default=0)
    peak_cpu_bytes = max((e.cpu_bytes_after for e in events), default=0)
    peak_resident_blocks = max((e.resident_blocks_after for e in events), default=0)
    peak_cpu_blocks = max((e.cpu_blocks_after for e in events), default=0)
    logical_accesses = stats_delta.get("uvm_handled", 0) + stats_delta.get("uvm_deferred", 0)
    cached_hits = stats_delta.get("uvm_cached_map_hits", 0)
    bridge_maps = stats_delta.get("uvm_bridge_map_calls", 0)
    hit_den = cached_hits + bridge_maps + reloads
    return {
        "policy": policy,
        "run_id": rec.get("run_id", ""),
        "return_code": rec.get("return_code"),
        "avg_ts": rec.get("llama_avg_ts"),
        "prompt_ts": next((r.get("avg_ts") for r in rec.get("llama_results", []) if r.get("n_prompt", 0)), None),
        "gen_ts": next((r.get("avg_ts") for r in rec.get("llama_results", []) if r.get("n_gen", 0)), None),
        "events": len(events),
        "allocs": allocs,
        "offloads": offloads,
        "reloads": reloads,
        "frees": frees,
        "daemon_allocs": decisions.get("allocs", 0),
        "daemon_offloads": decisions.get("offloads", 0),
        "daemon_reloads": decisions.get("reloads", 0),
        "daemon_frees": decisions.get("frees", 0),
        "peak_resident_blocks": peak_resident_blocks,
        "peak_resident_mib": peak_resident_bytes / 1048576,
        "peak_cpu_blocks": peak_cpu_blocks,
        "peak_cpu_mib": peak_cpu_bytes / 1048576,
        "uvm_cached_map_hits": cached_hits,
        "uvm_bridge_map_calls": bridge_maps,
        "uvm_deferred": stats_delta.get("uvm_deferred", 0),
        "uvm_handled": stats_delta.get("uvm_handled", 0),
        "uvm_hook_calls": stats_delta.get("uvm_hook_calls", 0),
        "uvm_errors": stats_delta.get("uvm_errors", 0),
        "kv_active_epoch": (rec.get("stats_after") or {}).get("kv_active_epoch", 0),
        "approx_cache_hit_rate": cached_hits / hit_den if hit_den else None,
        "reloads_per_offload": reloads / offloads if offloads else None,
        "logical_accesses": logical_accesses,
    }


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def aggregate(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float]:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None, 0.0
    return mean(vals), stdev(vals) if len(vals) > 1 else 0.0


def victim_sequence(events: list[Event]) -> list[int]:
    return [e.block_id for e in events if e.is_victim]


def reload_sequence(events: list[Event]) -> list[int]:
    return [e.block_id for e in events if e.is_vram_miss]


def sequence_digest(values: list[int]) -> str:
    payload = ",".join(str(v) for v in values).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def sequence_diff(a: list[int], b: list[int]) -> int:
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def jaccard(a: list[int], b: list[int]) -> float | None:
    left = set(a)
    right = set(b)
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def per_block_rows(events: list[Event]) -> list[dict[str, Any]]:
    blocks: dict[int, dict[str, Any]] = {}
    for event in events:
        row = blocks.setdefault(
            event.block_id,
            {
                "block_id": event.block_id,
                "session_id": event.session_id,
                "size": event.size,
                "allocs": 0,
                "offloads": 0,
                "reloads": 0,
                "frees": 0,
                "first_seq": event.seq,
                "last_seq": event.seq,
                "first_offload_seq": "",
                "last_offload_seq": "",
                "first_reload_seq": "",
                "last_reload_seq": "",
            },
        )
        row["last_seq"] = event.seq
        if event.op == "ALLOC":
            row["allocs"] += 1
        elif event.op == "OFFLOAD":
            row["offloads"] += 1
            if row["first_offload_seq"] == "":
                row["first_offload_seq"] = event.seq
            row["last_offload_seq"] = event.seq
        elif event.op == "RELOAD":
            row["reloads"] += 1
            if row["first_reload_seq"] == "":
                row["first_reload_seq"] = event.seq
            row["last_reload_seq"] = event.seq
        elif event.op == "FREE":
            row["frees"] += 1
    return [blocks[k] for k in sorted(blocks)]


def write_per_block_csv(path: Path, events: list[Event]) -> None:
    rows = per_block_rows(events)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "block_id",
        "session_id",
        "size",
        "allocs",
        "offloads",
        "reloads",
        "frees",
        "first_seq",
        "last_seq",
        "first_offload_seq",
        "last_offload_seq",
        "first_reload_seq",
        "last_reload_seq",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_pairwise_rows(
    summaries: list[dict[str, Any]],
    event_by_run: dict[str, list[Event]],
) -> list[dict[str, Any]]:
    by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        run_id = str(row["run_id"])
        trial = run_id.rsplit("-t", 1)[-1] if "-t" in run_id else run_id
        by_trial[trial].append(row)

    rows: list[dict[str, Any]] = []
    for trial, trial_rows in sorted(by_trial.items()):
        for left, right in combinations(sorted(trial_rows, key=lambda r: r["policy"]), 2):
            left_events = event_by_run[left["run_id"]]
            right_events = event_by_run[right["run_id"]]
            left_offloads = victim_sequence(left_events)
            right_offloads = victim_sequence(right_events)
            left_reloads = reload_sequence(left_events)
            right_reloads = reload_sequence(right_events)
            rows.append(
                {
                    "trial": trial,
                    "left_policy": left["policy"],
                    "right_policy": right["policy"],
                    "left_run_id": left["run_id"],
                    "right_run_id": right["run_id"],
                    "offload_count_left": len(left_offloads),
                    "offload_count_right": len(right_offloads),
                    "offload_order_diffs": sequence_diff(left_offloads, right_offloads),
                    "offload_set_jaccard": jaccard(left_offloads, right_offloads),
                    "offload_digest_left": sequence_digest(left_offloads),
                    "offload_digest_right": sequence_digest(right_offloads),
                    "reload_count_left": len(left_reloads),
                    "reload_count_right": len(right_reloads),
                    "reload_order_diffs": sequence_diff(left_reloads, right_reloads),
                    "reload_set_jaccard": jaccard(left_reloads, right_reloads),
                    "reload_digest_left": sequence_digest(left_reloads),
                    "reload_digest_right": sequence_digest(right_reloads),
                    "peak_resident_mib_left": left["peak_resident_mib"],
                    "peak_resident_mib_right": right["peak_resident_mib"],
                    "peak_cpu_mib_left": left["peak_cpu_mib"],
                    "peak_cpu_mib_right": right["peak_cpu_mib"],
                }
            )

    return rows


def write_pairwise_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_report(summaries: list[dict[str, Any]], event_by_run: dict[str, list[Event]]) -> str:
    lines: list[str] = []
    lines.append("# POLARIS Policy Trace Analysis")
    lines.append("")
    lines.append("This report reconstructs block-level residency from daemon decisions.")
    lines.append("`OFFLOAD` is counted as a VRAM eviction/victim; `RELOAD` is counted as a VRAM miss.")
    lines.append("Per-GPU-access hit attribution is not present in current logs, so cache-hit rate uses aggregate `uvm_cached_map_hits` plus bridge/reload events.")
    lines.append("The generated `*-events.csv` files are the detailed block residency logs; `*-blocks.csv` files roll those events up by block.")
    lines.append("")

    policies = sorted({s["policy"] for s in summaries})
    keys = [
        ("avg_ts", "avg_ts"),
        ("offloads", "offloads"),
        ("reloads", "reloads"),
        ("peak_resident_mib", "peak VRAM MiB"),
        ("peak_cpu_mib", "peak CPU MiB"),
        ("uvm_cached_map_hits", "cached hits"),
        ("uvm_bridge_map_calls", "bridge maps"),
        ("approx_cache_hit_rate", "approx hit rate"),
        ("kv_active_epoch", "KV hint epoch"),
    ]
    lines.append("| policy | metric | mean +/- stddev | n |")
    lines.append("|---|---|---:|---:|")
    for policy in policies:
        rows = [s for s in summaries if s["policy"] == policy]
        for key, label in keys:
            mu, sd = aggregate(rows, key)
            if mu is None:
                value = ""
            elif key == "approx_cache_hit_rate":
                value = f"{mu:.4f} +/- {sd:.4f}"
            else:
                value = f"{mu:.2f} +/- {sd:.2f}"
            lines.append(f"| {policy} | {label} | {value} | {len(rows)} |")
        lines.append("")

    lines.append("## Per-Run Summary")
    lines.append("")
    lines.append("| policy | run | avg_ts | offloads | reloads | peak VRAM MiB | peak CPU MiB | cached hits | bridge maps | approx hit rate | KV epoch |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        lines.append(
            "| {policy} | {run} | {avg} | {offloads} | {reloads} | {vram} | {cpu} | {hits} | {bridge} | {hit_rate} | {epoch} |".format(
                policy=row["policy"],
                run=row["run_id"],
                avg=fmt_num(row["avg_ts"]),
                offloads=row["offloads"],
                reloads=row["reloads"],
                vram=fmt_num(row["peak_resident_mib"]),
                cpu=fmt_num(row["peak_cpu_mib"]),
                hits=row["uvm_cached_map_hits"],
                bridge=row["uvm_bridge_map_calls"],
                hit_rate=fmt_num(row["approx_cache_hit_rate"], 4),
                epoch=row["kv_active_epoch"],
            )
        )

    lines.append("")
    lines.append("## Sequence Checks")
    lines.append("")
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_policy[row["policy"]].append(row)
    lines.append("| policy | first run offload sequence hash basis | first 12 victims | first 12 reload misses |")
    lines.append("|---|---|---|---|")
    for policy in policies:
        row = by_policy[policy][0]
        events = event_by_run[row["run_id"]]
        victims = victim_sequence(events)
        reloads = reload_sequence(events)
        basis = f"len={len(victims)} sum={sum(victims)} first={victims[:3]} last={victims[-3:]}"
        lines.append(
            f"| {policy} | `{basis}` | `{victims[:12]}` | `{reloads[:12]}` |"
        )

    lines.append("")
    lines.append("## Pairwise Working-Set Checks")
    lines.append("")
    lines.append("| trial | left | right | offload order diffs | offload set jaccard | reload order diffs | reload set jaccard |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    pairwise = build_pairwise_rows(summaries, event_by_run)
    for row in pairwise:
        lines.append(
            "| {trial} | {left} | {right} | {off_diffs} | {off_j:.6f} | {rel_diffs} | {rel_j:.6f} |".format(
                trial=row["trial"],
                left=row["left_policy"],
                right=row["right_policy"],
                off_diffs=row["offload_order_diffs"],
                off_j=row["offload_set_jaccard"] or 0.0,
                rel_diffs=row["reload_order_diffs"],
                rel_j=row["reload_set_jaccard"] or 0.0,
            )
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.output_dir or run_dir / "policy_trace"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    event_by_run: dict[str, list[Event]] = {}

    for rec in load_runs(run_dir):
        runs_path = Path(rec["_runs_path"])
        policy = runs_path.parts[-3]
        log_path = Path(rec.get("polarisd_log") or "")
        events = parse_events(log_path) if log_path.exists() else []
        run_id = str(rec.get("run_id", runs_path.parent.name))
        event_by_run[run_id] = events
        write_event_csv(out_dir / policy / f"{runs_path.parent.name}-events.csv", events)
        write_per_block_csv(out_dir / policy / f"{runs_path.parent.name}-blocks.csv", events)
        summaries.append(summarize_record(rec, events, policy))

    summaries.sort(key=lambda r: (r["policy"], r["run_id"]))
    with (out_dir / "summary.csv").open("w", newline="") as f:
        fieldnames = list(summaries[0].keys()) if summaries else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    write_pairwise_csv(out_dir / "pairwise.csv", build_pairwise_rows(summaries, event_by_run))
    report = render_report(summaries, event_by_run)
    (out_dir / "summary.md").write_text(report)
    print(out_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
