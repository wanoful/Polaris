#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Summarize POLARIS/vLLM/SGLang KV-cache trace CSV files.

Input format:
    timestamp_ns,op,session_id,token_start,token_count

The vLLM and SGLang trace patches in benchmarks/patches/ emit this shape.
This script intentionally reports KV-level allocator metrics only; it does not
claim end-to-end serving throughput for vLLM/SGLang.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RESERVE_OPS = {"BLOCK_RESERVE", "ALLOC", "RESERVE"}
RELEASE_OPS = {"BLOCK_RELEASE", "FREE", "RELEASE"}
SESSION_CREATE_OPS = {"SESSION_CREATE"}
SESSION_DESTROY_OPS = {"SESSION_DESTROY"}


def as_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value, 0)


def summarize_trace(path: Path, source: str, block_tokens: int) -> dict[str, Any]:
    live_blocks: dict[str, int] = defaultdict(int)
    live_tokens: dict[str, int] = defaultdict(int)
    seen_sessions: set[str] = set()
    active_sessions: set[str] = set()

    event_count = 0
    reserve_events = 0
    release_events = 0
    session_create_events = 0
    session_destroy_events = 0
    total_reserved_blocks = 0
    total_released_blocks = 0
    implicit_destroy_released_blocks = 0
    total_reserved_tokens = 0
    total_released_tokens = 0
    implicit_destroy_released_tokens = 0
    peak_live_blocks = 0
    peak_live_tokens = 0
    peak_active_sessions = 0
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp_ns", "op", "session_id", "token_start", "token_count"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing required column(s): {', '.join(sorted(missing))}")

        for row in reader:
            event_count += 1
            ts = as_int(row.get("timestamp_ns"))
            op = (row.get("op") or "").strip()
            sid = str(row.get("session_id") or "")
            token_count = as_int(row.get("token_count"))
            blocks = max(1, (token_count + block_tokens - 1) // block_tokens) if token_count > 0 else 1

            if first_timestamp_ns is None:
                first_timestamp_ns = ts
            last_timestamp_ns = ts

            if op in SESSION_CREATE_OPS:
                session_create_events += 1
                seen_sessions.add(sid)
                active_sessions.add(sid)
            elif op in SESSION_DESTROY_OPS:
                session_destroy_events += 1
                active_sessions.discard(sid)
                if live_blocks[sid] or live_tokens[sid]:
                    implicit_destroy_released_blocks += live_blocks[sid]
                    implicit_destroy_released_tokens += live_tokens[sid]
                    total_released_blocks += live_blocks[sid]
                    total_released_tokens += live_tokens[sid]
                    live_blocks[sid] = 0
                    live_tokens[sid] = 0
            elif op in RESERVE_OPS:
                reserve_events += 1
                seen_sessions.add(sid)
                active_sessions.add(sid)
                live_blocks[sid] += blocks
                live_tokens[sid] += token_count
                total_reserved_blocks += blocks
                total_reserved_tokens += token_count
            elif op in RELEASE_OPS:
                release_events += 1
                release_blocks = min(blocks, live_blocks[sid]) if live_blocks[sid] > 0 else blocks
                if token_count > 0:
                    release_tokens = min(token_count, live_tokens[sid])
                else:
                    release_tokens = min(release_blocks * block_tokens, live_tokens[sid])
                live_blocks[sid] = max(0, live_blocks[sid] - release_blocks)
                live_tokens[sid] = max(0, live_tokens[sid] - release_tokens)
                total_released_blocks += release_blocks
                total_released_tokens += release_tokens

            current_live_blocks = sum(live_blocks.values())
            current_live_tokens = sum(live_tokens.values())
            peak_live_blocks = max(peak_live_blocks, current_live_blocks)
            peak_live_tokens = max(peak_live_tokens, current_live_tokens)
            peak_active_sessions = max(peak_active_sessions, len(active_sessions))

    duration_ns = 0
    if first_timestamp_ns is not None and last_timestamp_ns is not None:
        duration_ns = max(0, last_timestamp_ns - first_timestamp_ns)

    live_blocks_final = sum(live_blocks.values())
    live_tokens_final = sum(live_tokens.values())

    return {
        "source": source,
        "trace_path": str(path),
        "comparison_scope": "kv_cache_allocator_trace",
        "block_tokens": block_tokens,
        "event_count": event_count,
        "session_create_events": session_create_events,
        "session_destroy_events": session_destroy_events,
        "reserve_events": reserve_events,
        "release_events": release_events,
        "unique_sessions": len(seen_sessions),
        "peak_active_sessions": peak_active_sessions,
        "total_reserved_blocks": total_reserved_blocks,
        "total_released_blocks": total_released_blocks,
        "implicit_destroy_released_blocks": implicit_destroy_released_blocks,
        "total_reserved_tokens": total_reserved_tokens,
        "total_released_tokens": total_released_tokens,
        "implicit_destroy_released_tokens": implicit_destroy_released_tokens,
        "peak_live_blocks": peak_live_blocks,
        "peak_live_tokens": peak_live_tokens,
        "live_blocks_final": live_blocks_final,
        "live_tokens_final": live_tokens_final,
        "duration_ns": duration_ns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path, help="Trace CSV path")
    parser.add_argument("--source", required=True, help="Trace source label, e.g. vllm or sglang")
    parser.add_argument("--block-tokens", type=int, default=16, help="KV tokens per comparison block")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--jsonl", action="store_true", help="Emit one-line JSON")
    args = parser.parse_args()

    summary = summarize_trace(args.trace, args.source, args.block_tokens)
    rendered = json.dumps(summary, sort_keys=True if args.jsonl else False, indent=None if args.jsonl else 2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
