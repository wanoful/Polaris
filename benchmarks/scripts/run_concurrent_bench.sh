#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# Multi-session concurrent KV pressure benchmark.
#
# Exercises the OS-level session/eviction logic that the llama.cpp single-
# session bench does not reach. Uses the existing polaris-workload concurrent
# subcommand to drive N sessions through prefill+decode against polaris.ko,
# captures /sys/kernel/polaris/stats deltas, and emits runs.jsonl +
# summary.md.
#
# This is NOT an end-to-end LLM benchmark — there is no real attention or
# real KV byte movement; it is a pure KV-scheduler stress test that
# measures policy behavior under concurrent multi-session pressure.
#
# Requires polaris.ko loaded and patched nvidia-uvm.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATS_PATH="${STATS_PATH:-/sys/kernel/polaris/stats}"
POLARISD_BIN="${POLARISD_BIN:-$ROOT_DIR/target/debug/polarisd}"
POLARISCTL_BIN="${POLARISCTL_BIN:-$ROOT_DIR/target/debug/polarisctl}"
WORKLOAD_BIN="${WORKLOAD_BIN:-$ROOT_DIR/target/debug/polaris-workload}"
RESULTS_ROOT="${POLARIS_BENCH_RESULTS_DIR:-$ROOT_DIR/benchmarks/results/llama_cpp}"
RUN_STAMP="${POLARIS_BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${POLARIS_BENCH_OUT_DIR:-$RESULTS_ROOT/concurrent-$RUN_STAMP}"
LOG_DIR="$OUT_DIR/logs"
RUNS_JSONL="$OUT_DIR/runs.jsonl"
SUMMARY_MD="$OUT_DIR/summary.md"

# Defaults from benchmarks/configs/concurrent.toml. Override via env.
NUM_SESSIONS_LIST="${POLARIS_BENCH_NUM_SESSIONS:-4,8,16,32}"
PROMPT_TOKENS="${POLARIS_BENCH_PROMPT_TOKENS:-256}"
OUTPUT_TOKENS="${POLARIS_BENCH_OUTPUT_TOKENS:-64}"
BLOCK_TOKENS="${POLARIS_BENCH_BLOCK_TOKENS:-16}"
GPU_BUDGET_BYTES="${POLARIS_BENCH_PRESSURE_BUDGET_BYTES:-8589934592}"  # 8 GiB
CPU_POOL_BYTES="${POLARIS_BENCH_CPU_POOL_BYTES:-4294967296}"           # 4 GiB
POLICIES_LIST="${POLARIS_BENCH_POLICIES:-fifo,lru,phase_aware,attention_stream}"

die() { echo "error: $*" >&2; exit 1; }
note() { echo "==> $*" >&2; }

for bin in "$POLARISD_BIN" "$POLARISCTL_BIN" "$WORKLOAD_BIN"; do
    [[ -x "$bin" ]] || die "missing executable: $bin"
done
[[ -r "$STATS_PATH" ]] || die "$STATS_PATH not readable (polaris.ko loaded?)"

mkdir -p "$LOG_DIR"
: >"$RUNS_JSONL"

polarisd_pid=""
cleanup() {
    if [[ -n "$polarisd_pid" ]]; then
        kill "$polarisd_pid" 2>/dev/null || true
        wait "$polarisd_pid" 2>/dev/null || true
        polarisd_pid=""
    fi
}
trap cleanup EXIT

stat_value() {
    local key="$1"
    awk -v k="${key}:" '
        $1 == k { print $2; found=1; exit }
        index($0, k) == 1 {
            v = substr($0, length(k) + 1)
            sub(/^[ \t]*/, "", v); sub(/[ \t].*/, "", v)
            print v; found=1; exit
        }
        END { if (!found) print "0" }
    ' "$STATS_PATH"
}

start_polarisd() {
    local log="$1"
    note "starting polarisd (budget=$((GPU_BUDGET_BYTES/1024/1024))MiB)"
    env POLARISD_RM_BACKING=1 \
        POLARISD_GPU_BUDGET_BYTES="$GPU_BUDGET_BYTES" \
        POLARISD_CPU_POOL_BYTES="$CPU_POOL_BYTES" \
        "$POLARISD_BIN" >"$log" 2>&1 &
    polarisd_pid=$!
    for _ in $(seq 1 200); do
        if ! kill -0 "$polarisd_pid" 2>/dev/null; then
            tail -n 80 "$log" >&2; die "polarisd exited; see $log"
        fi
        if [[ "$(stat_value daemon)" -ge 1 && "$(stat_value gpus)" -ge 1 ]]; then
            return 0
        fi
        sleep 0.05
    done
    tail -n 80 "$log" >&2
    die "polarisd did not register; see $log"
}

set_policy() {
    local pol="$1"
    local id
    case "$pol" in
        fifo|0)                                      id=0 ;;
        lru|1)                                       id=1 ;;
        phase_aware|phase-aware|PhaseAware|2)        id=2 ;;
        attention_stream|attention-stream|AttentionStream|3) id=3 ;;
        *) die "unknown policy: $pol" ;;
    esac
    "$POLARISCTL_BIN" set-policy "$id" >/dev/null
}

IFS=',' read -r -a NUM_SESSIONS_ARR <<<"$NUM_SESSIONS_LIST"
IFS=',' read -r -a POLICIES_ARR <<<"$POLICIES_LIST"

run_one() {
    local policy="$1"
    local n="$2"
    local tag="${policy}-N${n}"
    local stats_before="$LOG_DIR/$tag.stats.before"
    local stats_after="$LOG_DIR/$tag.stats.after"
    local workload_log="$LOG_DIR/$tag.log"
    local polarisd_log="$LOG_DIR/$tag.polarisd.log"

    note "--- policy=$policy num_sessions=$n ---"
    start_polarisd "$polarisd_log"
    set_policy "$policy"
    cp "$STATS_PATH" "$stats_before"

    local start_ns end_ns
    start_ns=$(date +%s%N)
    set +e
    "$WORKLOAD_BIN" concurrent \
        --num-sessions "$n" \
        --prompt-tokens "$PROMPT_TOKENS" \
        --output-tokens "$OUTPUT_TOKENS" \
        --tokens-per-block "$BLOCK_TOKENS" \
        >"$workload_log" 2>&1
    local rc=$?
    set -e
    end_ns=$(date +%s%N)

    cp "$STATS_PATH" "$stats_after"
    cleanup

    python3 - "$tag" "$policy" "$n" "$rc" "$start_ns" "$end_ns" \
        "$PROMPT_TOKENS" "$OUTPUT_TOKENS" "$BLOCK_TOKENS" \
        "$GPU_BUDGET_BYTES" "$CPU_POOL_BYTES" \
        "$stats_before" "$stats_after" "$workload_log" "$polarisd_log" \
        "$RUN_STAMP" >>"$RUNS_JSONL" <<'PY'
import json, sys
from pathlib import Path
(tag, policy, n, rc, start_ns, end_ns, prompt, out, blk, budget, cpu_pool,
 before, after, wlog, plog, run_id) = sys.argv[1:]

def parse_stats(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line: continue
        k, v = line.split(":", 1)
        tok = v.strip().split()[0] if v.strip() else "0"
        try: out[k.strip()] = int(tok, 0)
        except ValueError: pass
    return out

b = parse_stats(before)
a = parse_stats(after)
keys = ["offloads","reloads","evictions","cow_breaks","blocks",
        "uvm_hook_calls","uvm_bridge_map_calls","uvm_errors"]
delta = {k: a.get(k,0) - b.get(k,0) for k in keys}

elapsed_ns = int(end_ns) - int(start_ns)
record = {
    "source": "polaris",
    "mode": "polaris_concurrent",
    "comparison_scope": "kv_scheduler_only",
    "run_id": run_id,
    "tag": tag,
    "policy": {"eviction": policy,
               "gpu_budget_bytes": int(budget),
               "cpu_pool_bytes": int(cpu_pool)},
    "workload": {"num_sessions": int(n),
                 "prompt_tokens": int(prompt),
                 "output_tokens": int(out),
                 "tokens_per_block": int(blk)},
    "return_code": int(rc),
    "elapsed_ns": elapsed_ns,
    "elapsed_sec": elapsed_ns / 1e9,
    "stats_before": b,
    "stats_after":  a,
    "stats_delta":  delta,
    "workload_log": wlog,
    "polarisd_log": plog,
}
print(json.dumps(record, sort_keys=True))
PY
}

for policy in "${POLICIES_ARR[@]}"; do
    for n in "${NUM_SESSIONS_ARR[@]}"; do
        run_one "$policy" "$n"
    done
done

note "writing summary"
python3 - "$RUNS_JSONL" "$SUMMARY_MD" <<'PY'
import json, sys
from pathlib import Path
src, dst = sys.argv[1:]
rows = []
for line in Path(src).read_text().splitlines():
    line = line.strip()
    if not line: continue
    try: rows.append(json.loads(line))
    except json.JSONDecodeError: pass

lines = ["# Concurrent KV-scheduler benchmark", ""]
lines.append("| policy | N | sec | offloads | reloads | cow_breaks | bridge_maps | uvm_errors | rc |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in sorted(rows, key=lambda x: (x.get("policy",{}).get("eviction",""),
                                    x.get("workload",{}).get("num_sessions",0))):
    d = r.get("stats_delta") or {}
    lines.append(
        f"| {r['policy']['eviction']} | {r['workload']['num_sessions']} | "
        f"{r['elapsed_sec']:.2f} | {d.get('offloads',0)} | {d.get('reloads',0)} | "
        f"{d.get('cow_breaks',0)} | {d.get('uvm_bridge_map_calls',0)} | "
        f"{d.get('uvm_errors',0)} | {r['return_code']} |"
    )
Path(dst).write_text("\n".join(lines) + "\n")
print("wrote", dst)
PY

note "results: $OUT_DIR"
