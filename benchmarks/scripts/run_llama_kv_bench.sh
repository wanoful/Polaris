#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# llama.cpp/POLARIS KV-cache benchmark runner.
#
# This is an engineering benchmark harness, not a correctness gate. It records
# end-to-end llama-bench JSON plus POLARIS KV fault/offload/reload counters.
# Optional vLLM/SGLang CSV traces are summarized at the KV allocator layer and
# appended to the same JSONL file for block-level comparison.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/wano/workspace/llama.cpp}"
NVIDIA_KO_DIR="${NVIDIA_KO_DIR:-/home/wano/workspace/open-gpu-kernel-modules}"
SHIM_SO="${SHIM_SO:-$ROOT_DIR/libpolaris-shim/libpolaris-shim.so}"
STATS_PATH="${STATS_PATH:-/sys/kernel/polaris/stats}"
POLARISD_BIN="${POLARISD_BIN:-$ROOT_DIR/target/debug/polarisd}"
POLARISCTL_BIN="${POLARISCTL_BIN:-$ROOT_DIR/target/debug/polarisctl}"
NVIDIA_UVM_TESTS_PARAM="${NVIDIA_UVM_TESTS_PARAM:-/sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests}"
RESULTS_ROOT="${POLARIS_BENCH_RESULTS_DIR:-$ROOT_DIR/benchmarks/results/llama_cpp}"
RUN_ID="${POLARIS_BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${POLARIS_BENCH_OUT_DIR:-$RESULTS_ROOT/$RUN_ID}"
LOG_DIR="$OUT_DIR/logs"
RUNS_JSONL="$OUT_DIR/runs.jsonl"
SUMMARY_MD="$OUT_DIR/summary.md"

die() {
    echo "error: $*" >&2
    exit 1
}

note() {
    echo "==> $*" >&2
}

first_existing_executable() {
    local path
    for path in "$@"; do
        if [[ -f "$path" && -x "$path" ]]; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    return 1
}

first_existing_file() {
    local path
    for path in "$@"; do
        if [[ -f "$path" ]]; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    return 1
}

require_file() {
    local path="$1"
    local what="$2"
    [[ -f "$path" ]] || die "$what not found: $path"
}

require_executable() {
    local path="$1"
    local what="$2"
    [[ -f "$path" && -x "$path" ]] || die "$what is not an executable file: $path"
}

stat_value_from_file() {
    local file="$1"
    local key="$2"
    awk -v key="${key}:" '
        $1 == key {
            print $2
            found = 1
            exit
        }
        index($0, key) == 1 {
            value = substr($0, length(key) + 1)
            sub(/^[ \t]*/, "", value)
            sub(/[ \t].*/, "", value)
            print value
            found = 1
            exit
        }
        END { if (!found) print "0" }
    ' "$file"
}

stat_value() {
    stat_value_from_file "$STATS_PATH" "$1"
}

json_escape() {
    python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

LLAMA_CPP_BIN="${LLAMA_CPP_BIN:-}"
if [[ -z "$LLAMA_CPP_BIN" ]]; then
    LLAMA_CPP_BIN="$(first_existing_executable \
        "$LLAMA_CPP_DIR/build-polaris/tools/llama-bench" \
        "$LLAMA_CPP_DIR/build-polaris/bin/llama-bench" \
        "$LLAMA_CPP_DIR/build/tools/llama-bench" \
        "$LLAMA_CPP_DIR/build/bin/llama-bench" \
    )" || die "could not find llama-bench; set LLAMA_CPP_BIN=/path/to/llama-bench"
fi

LLAMA_CPP_MODEL="${LLAMA_CPP_MODEL:-}"
if [[ -z "$LLAMA_CPP_MODEL" ]]; then
    LLAMA_CPP_MODEL="$(first_existing_file \
        "/home/wano/workspace/models/SmolLM2-135M-Instruct-F16.gguf" \
        "$LLAMA_CPP_DIR/models/tinyllamas/stories260K.gguf" \
    )" || die "could not find a runnable GGUF model; set LLAMA_CPP_MODEL=/path/to/model.gguf"
fi

require_executable "$LLAMA_CPP_BIN" "llama.cpp benchmark binary"
require_file "$LLAMA_CPP_MODEL" "GGUF model"
require_file "$SHIM_SO" "libpolaris-shim.so"
require_executable "$POLARISD_BIN" "polarisd binary"
require_executable "$POLARISCTL_BIN" "polarisctl binary"
[[ -r "$STATS_PATH" ]] || die "$STATS_PATH is not readable"

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

wait_for_kernel_cleanup() {
    local label="$1"
    local baseline_gpus="$2"
    for _ in $(seq 1 120); do
        if [[ "$(stat_value daemon)" == "0" &&
              "$(stat_value sessions)" == "0" &&
              "$(stat_value blocks)" == "0" &&
              "$(stat_value pending_decs)" == "0" &&
              "$(stat_value static_blocks)" == "0" &&
              "$(stat_value block_mappings)" == "0" &&
              "$(stat_value v4_va_spaces)" == "0" &&
              "$(stat_value v4_worker_pids)" == "0" &&
              "$(stat_value gpus)" == "$baseline_gpus" ]]; then
            return 0
        fi
        sleep 0.05
    done
    sed -n '1,140p' "$STATS_PATH" >&2 || true
    die "$label cleanup did not return to baseline gpus=$baseline_gpus"
}

start_polarisd() {
    local log="$1"
    shift
    local timeout_sec="${POLARIS_BENCH_DAEMON_STARTUP_TIMEOUT_SEC:-10}"
    local loops=$((timeout_sec * 20))
    if [[ "$loops" -lt 1 ]]; then
        loops=1
    fi
    note "starting polarisd for benchmark"
    env POLARISD_RM_BACKING=1 "$@" "$POLARISD_BIN" >"$log" 2>&1 &
    polarisd_pid=$!
    for _ in $(seq 1 "$loops"); do
        if ! kill -0 "$polarisd_pid" 2>/dev/null; then
            tail -n 160 "$log" >&2 || true
            die "polarisd exited during startup; see $log"
        fi
        if [[ "$(stat_value daemon)" -ge 1 && "$(stat_value gpus)" -ge 1 ]]; then
            return 0
        fi
        sleep 0.05
    done
    tail -n 160 "$log" >&2 || true
    die "polarisd did not register with polaris.ko within ${timeout_sec}s; see $log"
}

require_uvm_dispatch_key_probe() {
    local enabled=""
    if [[ -r "$NVIDIA_UVM_TESTS_PARAM" ]]; then
        enabled="$(tr -d '[:space:]' <"$NVIDIA_UVM_TESTS_PARAM")"
    fi

    case "$enabled" in
        1|Y|y)
            return 0
            ;;
    esac

    die "nvidia_uvm must be loaded with uvm_enable_builtin_tests=1 for POLARIS RM/UVM bootstrap; current $NVIDIA_UVM_TESTS_PARAM=${enabled:-unavailable}"
}

set_eviction_policy() {
    local policy="${POLARIS_BENCH_EVICTION_POLICY:-fifo}"

    local policy_id
    case "$policy" in
        0|fifo|FIFO)
            policy_id=0
            ;;
        1|lru|LRU)
            policy_id=1
            ;;
        2|phase_aware|phase-aware|PhaseAware|PHASE_AWARE)
            policy_id=2
            ;;
        *)
            die "unknown POLARIS_BENCH_EVICTION_POLICY=$policy"
            ;;
    esac

    note "setting eviction policy=$policy_id"
    "$POLARISCTL_BIN" set-policy "$policy_id" >/dev/null
}

stop_polarisd() {
    cleanup
}

build_llama_args() {
    local -n out_args="$1"
    local prompt_tokens="$2"
    local gen_tokens="$3"
    local repetitions="$4"

    out_args=(
        -m "$LLAMA_CPP_MODEL"
        -p "$prompt_tokens"
        -n "$gen_tokens"
        -r "$repetitions"
        --no-warmup
        -ngl "${LLAMA_CPP_GPU_LAYERS:-999}"
        -fa "${LLAMA_CPP_FLASH_ATTN:-0}"
        -o json
        -dev "${LLAMA_CPP_DEVICE:-CUDA0}"
    )

    if [[ -n "${LLAMA_CPP_EXTRA_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        extra_args=( ${LLAMA_CPP_EXTRA_ARGS} )
        out_args+=( "${extra_args[@]}" )
    fi
}

write_record() {
    local mode="$1"
    local comparison_scope="$2"
    local prompt_tokens="$3"
    local gen_tokens="$4"
    local repetitions="$5"
    local budget_bytes="$6"
    local cpu_pool_bytes="$7"
    local stdout_json="$8"
    local stderr_log="$9"
    local stats_before="${10}"
    local stats_after="${11}"
    local rc="${12}"

    python3 - "$mode" "$comparison_scope" "$prompt_tokens" "$gen_tokens" "$repetitions" \
        "$budget_bytes" "$cpu_pool_bytes" "$stdout_json" "$stderr_log" "$stats_before" "$stats_after" "$rc" \
        "$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || true)" \
        "$LLAMA_CPP_BIN" "$LLAMA_CPP_MODEL" "$RUN_ID" >>"$RUNS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

mode, scope, prompt, gen, reps, budget, cpu_pool, stdout_json, stderr_log, before, after, rc, commit, llama_bin, model, run_id = sys.argv[1:]
keys = [
    "sessions", "blocks", "gpus", "daemon", "offloads", "reloads", "evictions",
    "cow_breaks", "pending_decs", "v4_va_spaces", "static_blocks", "block_mappings",
    "uvm_hook_calls", "uvm_handled", "uvm_deferred", "uvm_rejected", "uvm_no_pte",
    "uvm_errors", "uvm_bridge_map_calls", "uvm_bridge_map_ok", "uvm_bridge_map_err",
    "uvm_bridge_map_retry", "uvm_bridge_map_avg_ns",
]

def parse_stats(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        token = v.strip().split()[0] if v.strip() else "0"
        try:
            out[k] = int(token, 0)
        except ValueError:
            pass
    return out

def load_llama(path):
    text = Path(path).read_text()
    if not text.strip():
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []

before_s = parse_stats(before)
after_s = parse_stats(after)
delta = {k: after_s.get(k, 0) - before_s.get(k, 0) for k in keys}
llama = load_llama(stdout_json)
avg_ts_values = [x.get("avg_ts") for x in llama if isinstance(x, dict) and isinstance(x.get("avg_ts"), (int, float))]
avg_ts = sum(avg_ts_values) / len(avg_ts_values) if avg_ts_values else None
record = {
    "source": "polaris",
    "mode": mode,
    "comparison_scope": scope,
    "run_id": run_id,
    "git_commit": commit,
    "llama_cpp_bin": llama_bin,
    "model": model,
    "workload": {
        "prompt_tokens": int(prompt),
        "gen_tokens": int(gen),
        "repetitions": int(reps),
    },
    "policy": {
        "gpu_budget_bytes": int(budget),
        "cpu_pool_bytes": int(cpu_pool),
    },
    "return_code": int(rc),
    "llama_avg_ts": avg_ts,
    "llama_results": llama,
    "stats_before": before_s,
    "stats_after": after_s,
    "stats_delta": delta,
    "stdout_json": stdout_json,
    "stderr_log": stderr_log,
}
print(json.dumps(record, sort_keys=True))
PY
}

run_bench_command() {
    local stdout_json="$1"
    local stderr_log="$2"
    shift 2
    local timeout_sec="${POLARIS_BENCH_RUN_TIMEOUT_SEC:-0}"
    local kill_after_sec="${POLARIS_BENCH_RUN_TIMEOUT_KILL_AFTER_SEC:-30}"

    if [[ "$timeout_sec" != "0" && -n "$timeout_sec" ]]; then
        timeout --kill-after="${kill_after_sec}s" "${timeout_sec}s" "$@" >"$stdout_json" 2>"$stderr_log"
    else
        "$@" >"$stdout_json" 2>"$stderr_log"
    fi
}

run_one() {
    local mode="$1"
    local prompt_tokens="$2"
    local gen_tokens="$3"
    local repetitions="$4"
    local budget_bytes="$5"
    local cpu_pool_bytes="$6"
    local baseline_gpus
    local stdout_json="$LOG_DIR/$mode-p${prompt_tokens}-n${gen_tokens}.json"
    local stderr_log="$LOG_DIR/$mode-p${prompt_tokens}-n${gen_tokens}.log"
    local stats_before="$LOG_DIR/$mode-p${prompt_tokens}-n${gen_tokens}.stats.before"
    local stats_after="$LOG_DIR/$mode-p${prompt_tokens}-n${gen_tokens}.stats.after"
    local polarisd_log="$LOG_DIR/$mode-p${prompt_tokens}-n${gen_tokens}.polarisd.log"
    local args=()
    local rc=0

    note "running mode=$mode prompt=$prompt_tokens gen=$gen_tokens reps=$repetitions"
    cp "$STATS_PATH" "$stats_before"
    baseline_gpus="$(stat_value gpus)"
    build_llama_args args "$prompt_tokens" "$gen_tokens" "$repetitions"

    case "$mode" in
        native_cuda)
            set +e
            run_bench_command "$stdout_json" "$stderr_log" "$LLAMA_CPP_BIN" "${args[@]}"
            rc=$?
            set -e
            ;;
        polaris_no_pressure|polaris_pressure|polaris_sustained_pressure)
            require_uvm_dispatch_key_probe
            start_polarisd "$polarisd_log" \
                POLARISD_GPU_BUDGET_BYTES="$budget_bytes" \
                POLARISD_CPU_POOL_BYTES="$cpu_pool_bytes"
            set_eviction_policy
            set +e
            run_bench_command "$stdout_json" "$stderr_log" \
                env \
                GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}" \
                GGML_CUDA_PDL="${GGML_CUDA_PDL:-0}" \
                POLARIS_SHIM_BOOTSTRAP_RM_UVM=1 \
                POLARIS_SHIM_STRICT_MANAGED_ALLOC=1 \
                POLARIS_SHIM_REPORT_STATS=1 \
                POLARIS_SHIM_REQUIRE_KV_SCOPE="${POLARIS_SHIM_REQUIRE_KV_SCOPE:-1}" \
                POLARIS_SHIM_ALLOW_ZERO_MEMSET="${POLARIS_SHIM_ALLOW_ZERO_MEMSET:-1}" \
                POLARIS_SHIM_TRANSIENT_GPU="${POLARIS_SHIM_TRANSIENT_GPU:-1}" \
                POLARIS_SHIM_GPU_ID="${POLARIS_SHIM_GPU_ID:-0}" \
                POLARIS_SHIM_CUDA_ORDINAL="${POLARIS_SHIM_CUDA_ORDINAL:-0}" \
                POLARIS_SHIM_BLOCK_SIZE="${POLARIS_SHIM_BLOCK_SIZE:-0x200000}" \
                POLARIS_SHIM_MANAGED_LENGTH_CAP="${POLARIS_SHIM_MANAGED_LENGTH_CAP:-17179869184}" \
                POLARIS_SHIM_MIN_MANAGED_ALLOC="${POLARIS_SHIM_MIN_MANAGED_ALLOC:-1}" \
                POLARIS_SHIM_MAX_MANAGED_ALLOC="${POLARIS_SHIM_MAX_MANAGED_ALLOC:-0}" \
                POLARIS_SHIM_GPU_BUDGET_BYTES="$budget_bytes" \
                POLARIS_SHIM_CPU_POOL_BYTES="$cpu_pool_bytes" \
                LD_PRELOAD="$SHIM_SO${LD_PRELOAD:+:$LD_PRELOAD}" \
                "$LLAMA_CPP_BIN" "${args[@]}"
            rc=$?
            set -e
            stop_polarisd
            wait_for_kernel_cleanup "$mode" "$baseline_gpus"
            ;;
        *)
            die "unknown benchmark mode: $mode"
            ;;
    esac

    cp "$STATS_PATH" "$stats_after"
    write_record "$mode" "llama_cpp_end_to_end" "$prompt_tokens" "$gen_tokens" "$repetitions" \
        "$budget_bytes" "$cpu_pool_bytes" "$stdout_json" "$stderr_log" "$stats_before" "$stats_after" "$rc"

    if [[ "$rc" -ne 0 ]]; then
        tail -n 160 "$stderr_log" >&2 || true
        if [[ "${POLARIS_BENCH_ALLOW_FAILURES:-0}" == "1" ]]; then
            note "benchmark mode=$mode failed with rc=$rc; continuing because POLARIS_BENCH_ALLOW_FAILURES=1"
        else
            die "benchmark mode=$mode failed with rc=$rc; see $stderr_log"
        fi
    fi
}

summarize_external_trace() {
    local source="$1"
    local trace="$2"
    [[ -n "$trace" ]] || return 0
    require_file "$trace" "$source trace"
    python3 "$ROOT_DIR/benchmarks/scripts/kv_trace_summary.py" \
        --trace "$trace" \
        --source "$source" \
        --block-tokens "${POLARIS_BENCH_BLOCK_TOKENS:-16}" \
        --jsonl >>"$RUNS_JSONL"
}

IFS=',' read -r -a prompt_matrix <<<"${POLARIS_BENCH_PROMPTS:-128,512}"
IFS=',' read -r -a gen_matrix <<<"${POLARIS_BENCH_GENS:-32}"
IFS=',' read -r -a mode_matrix <<<"${POLARIS_BENCH_MODES:-native_cuda,polaris_no_pressure,polaris_pressure}"

repetitions="${POLARIS_BENCH_REPETITIONS:-3}"
no_pressure_budget="${POLARIS_BENCH_NO_PRESSURE_BUDGET_BYTES:-17179869184}"
pressure_budget="${POLARIS_BENCH_PRESSURE_BUDGET_BYTES:-4194304}"
sustained_budget="${POLARIS_BENCH_SUSTAINED_BUDGET_BYTES:-$pressure_budget}"
cpu_pool_bytes="${POLARIS_BENCH_CPU_POOL_BYTES:-4294967296}"

note "results: $OUT_DIR"
note "modes: ${mode_matrix[*]}"
note "prompts: ${prompt_matrix[*]} gens: ${gen_matrix[*]} reps=$repetitions"

for prompt in "${prompt_matrix[@]}"; do
    for gen in "${gen_matrix[@]}"; do
        for mode in "${mode_matrix[@]}"; do
            case "$mode" in
                native_cuda)
                    run_one "$mode" "$prompt" "$gen" "$repetitions" 0 0
                    ;;
                polaris_no_pressure)
                    run_one "$mode" "$prompt" "$gen" "$repetitions" "$no_pressure_budget" "$cpu_pool_bytes"
                    ;;
                polaris_pressure)
                    run_one "$mode" "$prompt" "$gen" "$repetitions" "$pressure_budget" "$cpu_pool_bytes"
                    ;;
                polaris_sustained_pressure)
                    run_one "$mode" "$prompt" "$gen" "$repetitions" "$sustained_budget" "$cpu_pool_bytes"
                    ;;
                *)
                    die "unknown POLARIS_BENCH_MODES entry: $mode"
                    ;;
            esac
        done
    done
done

summarize_external_trace "vllm" "${VLLM_TRACE:-}"
summarize_external_trace "sglang" "${SGLANG_TRACE:-}"

python3 "$ROOT_DIR/benchmarks/scripts/compare_kv_results.py" "$RUNS_JSONL" --output "$SUMMARY_MD" >/dev/null
note "wrote $RUNS_JSONL"
note "wrote $SUMMARY_MD"
