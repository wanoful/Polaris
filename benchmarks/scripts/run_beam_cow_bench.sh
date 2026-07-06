#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# Beam-search COW memory benchmark.
#
# Compares two arms with the same logical shape (1 parent + N-1 children
# each generating `decode_steps` tokens from a shared 512-token prefix):
#
#   arm A: no-cow → N independent sessions, each re-prefills the 512-token
#                   prefix from scratch (via `polaris-workload concurrent`).
#                   No shared blocks.
#   arm B: cow    → 1 parent session prefills, N-1 branch via SESSION_BRANCH
#                   (refcount sharing). Then each child decodes.
#                   (via `polaris-workload beam-search`).
#
# Metric of interest: shared_gpu_bytes / private_gpu_bytes / cow_break_count
# from /sys/kernel/polaris/stats, plus elapsed time.
#
# Requires polaris.ko + polarisd. This is a scheduler-level benchmark, not
# an end-to-end LLM run.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATS_PATH="${STATS_PATH:-/sys/kernel/polaris/stats}"
POLARISD_BIN="${POLARISD_BIN:-$ROOT_DIR/target/debug/polarisd}"
POLARISCTL_BIN="${POLARISCTL_BIN:-$ROOT_DIR/target/debug/polarisctl}"
WORKLOAD_BIN="${WORKLOAD_BIN:-$ROOT_DIR/target/debug/polaris-workload}"
RESULTS_ROOT="${POLARIS_BENCH_RESULTS_DIR:-$ROOT_DIR/benchmarks/results/llama_cpp}"
RUN_STAMP="${POLARIS_BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${POLARIS_BENCH_OUT_DIR:-$RESULTS_ROOT/beam-cow-$RUN_STAMP}"
LOG_DIR="$OUT_DIR/logs"
RUNS_JSONL="$OUT_DIR/runs.jsonl"
SUMMARY_MD="$OUT_DIR/summary.md"

BEAM_WIDTH="${POLARIS_BENCH_BEAM_WIDTH:-8}"
PROMPT_TOKENS="${POLARIS_BENCH_PROMPT_TOKENS:-512}"
DECODE_STEPS="${POLARIS_BENCH_DECODE_STEPS:-64}"
BLOCK_TOKENS="${POLARIS_BENCH_BLOCK_TOKENS:-16}"
GPU_BUDGET_BYTES="${POLARIS_BENCH_PRESSURE_BUDGET_BYTES:-8589934592}"
CPU_POOL_BYTES="${POLARIS_BENCH_CPU_POOL_BYTES:-4294967296}"

die() { echo "error: $*" >&2; exit 1; }
note() { echo "==> $*" >&2; }

for bin in "$POLARISD_BIN" "$WORKLOAD_BIN"; do
    [[ -x "$bin" ]] || die "missing executable: $bin"
done
[[ -r "$STATS_PATH" ]] || die "$STATS_PATH not readable"

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
    env POLARISD_RM_BACKING=1 \
        POLARISD_GPU_BUDGET_BYTES="$GPU_BUDGET_BYTES" \
        POLARISD_CPU_POOL_BYTES="$CPU_POOL_BYTES" \
        "$POLARISD_BIN" >"$log" 2>&1 &
    polarisd_pid=$!
    for _ in $(seq 1 200); do
        if ! kill -0 "$polarisd_pid" 2>/dev/null; then
            tail -n 80 "$log" >&2; die "polarisd exited"
        fi
        if [[ "$(stat_value daemon)" -ge 1 && "$(stat_value gpus)" -ge 1 ]]; then
            return 0
        fi
        sleep 0.05
    done
    die "polarisd did not register"
}

run_arm() {
    local arm="$1"; shift
    local cmd=( "$@" )
    local stats_before="$LOG_DIR/$arm.stats.before"
    local stats_after="$LOG_DIR/$arm.stats.after"
    local wlog="$LOG_DIR/$arm.log"
    local plog="$LOG_DIR/$arm.polarisd.log"

    note "--- arm=$arm ---"
    start_polarisd "$plog"
    cp "$STATS_PATH" "$stats_before"

    local start_ns end_ns
    start_ns=$(date +%s%N)
    set +e
    "${cmd[@]}" >"$wlog" 2>&1
    local rc=$?
    set -e
    end_ns=$(date +%s%N)

    cp "$STATS_PATH" "$stats_after"
    cleanup

    python3 - "$arm" "$rc" "$start_ns" "$end_ns" \
        "$BEAM_WIDTH" "$PROMPT_TOKENS" "$DECODE_STEPS" "$BLOCK_TOKENS" \
        "$GPU_BUDGET_BYTES" "$CPU_POOL_BYTES" \
        "$stats_before" "$stats_after" "$wlog" "$plog" "$RUN_STAMP" \
        >>"$RUNS_JSONL" <<'PY'
import json, sys
from pathlib import Path
(arm, rc, start_ns, end_ns, beam_width, prompt, decode, blk,
 budget, cpu_pool, before, after, wlog, plog, run_id) = sys.argv[1:]

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

# We want absolute peak-ish residency at "after" — concurrent and beam
# workloads both teardown sessions at end, so by `after` the resident-set
# is mostly drained. The deltas in cow_break_count and offloads/reloads
# still reflect what happened during the run.
delta_keys = ["offloads","reloads","cow_breaks","blocks","uvm_errors"]
delta = {k: a.get(k,0) - b.get(k,0) for k in delta_keys}

elapsed_ns = int(end_ns) - int(start_ns)
record = {
    "source": "polaris",
    "mode":   f"beam_cow_{arm}",
    "comparison_scope": "kv_scheduler_cow",
    "run_id": run_id,
    "workload": {"beam_width": int(beam_width),
                 "prompt_tokens": int(prompt),
                 "decode_steps": int(decode),
                 "tokens_per_block": int(blk)},
    "policy": {"gpu_budget_bytes": int(budget),
               "cpu_pool_bytes": int(cpu_pool)},
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

# Arm A: no-cow — N independent sessions each re-prefills the prompt
N_INDEPENDENT_SESSIONS="$BEAM_WIDTH"
run_arm "no_cow" \
    "$WORKLOAD_BIN" concurrent \
    --num-sessions "$N_INDEPENDENT_SESSIONS" \
    --prompt-tokens "$PROMPT_TOKENS" \
    --output-tokens "$DECODE_STEPS" \
    --tokens-per-block "$BLOCK_TOKENS"

# Arm B: cow — 1 parent prefill + N-1 branched children
run_arm "cow" \
    "$WORKLOAD_BIN" beam-search \
    --beam-width "$BEAM_WIDTH" \
    --prompt-tokens "$PROMPT_TOKENS" \
    --decode-steps "$DECODE_STEPS" \
    --tokens-per-block "$BLOCK_TOKENS"

note "writing summary"
python3 - "$RUNS_JSONL" "$SUMMARY_MD" "$LOG_DIR" <<'PY'
import json, re, sys
from pathlib import Path
src, dst, log_dir = sys.argv[1:]

rows = []
for line in Path(src).read_text().splitlines():
    line = line.strip()
    if not line: continue
    try: rows.append(json.loads(line))
    except json.JSONDecodeError: pass

# Try to extract per-arm "shared/private MiB" from the workload stdout —
# beam-search prints "After branch: shared=X MiB  private=Y MiB  cow_breaks=Z"
# concurrent does not, so we read the workload log for each row.
SHARE_RE = re.compile(r"shared=(\d+)\s*MiB\s+private=(\d+)\s*MiB\s+cow_breaks=(\d+)")
for r in rows:
    log = Path(r.get("workload_log") or "")
    shared = private = cow = 0
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            m = SHARE_RE.search(line)
            if m:
                shared, private, cow = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    r["peak_shared_mib"] = shared
    r["peak_private_mib"] = private
    r["peak_cow_breaks"] = cow

lines = ["# Beam-search COW vs no-COW", ""]
lines.append("Logical shape per arm: 1 parent of `prompt_tokens` + `beam_width-1` children, each generating `decode_steps`.")
lines.append("- no_cow: N independent sessions, each re-prefills (worst-case memory)")
lines.append("- cow:    1 parent prefill + branched children (refcount sharing)")
lines.append("")
lines.append("| arm | sec | offloads | reloads | cow_breaks (delta) | peak_shared MiB | peak_private MiB | rc |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in sorted(rows, key=lambda x: x.get("mode","")):
    d = r.get("stats_delta") or {}
    lines.append(
        f"| {r['mode']} | {r['elapsed_sec']:.2f} | {d.get('offloads',0)} | {d.get('reloads',0)} | "
        f"{d.get('cow_breaks',0)} | {r['peak_shared_mib']} | {r['peak_private_mib']} | {r['return_code']} |"
    )
Path(dst).write_text("\n".join(lines) + "\n")
print("wrote", dst)
PY

note "results: $OUT_DIR"
