#!/usr/bin/env bash
# POLARIS Full Benchmark Suite
# Runs all configured benchmarks and produces comparison results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results"

mkdir -p "$RESULTS_DIR"

echo "=== POLARIS Benchmark Suite ==="
echo "Results: $RESULTS_DIR"

exec "$SCRIPT_DIR/run_llama_kv_bench.sh"
