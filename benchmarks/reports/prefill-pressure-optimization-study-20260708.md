# Prefill-Under-Pressure Optimization Study — 2026-07-08

Qwen2.5-14B-Q4_K_M, 16384/32, POLARIS budget 2560 MiB, RTX 5070 Ti, baseline
`nvidia-uvm` (`BEC2C67`) + `polaris.ko` from `v4-prefill-fault-path-opt`. Every
run reloads `polaris.ko` fresh (stale kernel block/session state otherwise
causes intermittent setup-time `BLOCK_RESERVE` failures — see note).

## Motivation

Under memory pressure POLARIS prefill runs at ~736 tok/s vs ~1317 native
llama.cpp (−44%). (No-pressure prefill is already ~1268 vs 1317 = −3.7%, so the
pressure case is the real target.) This study characterizes *what* the
bottleneck is by attacking it from four independent angles.

## Results

| change | mechanism attacked | prompt tok/s | migrations | verdict |
|---|---|---:|---:|---|
| baseline (sync, 2 MiB) | — | 736 | 21232 | reference |
| **async offloads** | offload serialization | **851 (n=3)** | 21232 | **+15.6% ✓ shipped** |
| 4 MiB blocks (async) | fault/copy count | 867.6 (n=3) | 11816 | +1.9% ✓ marginal |
| 8 MiB blocks (async) | fault/copy count | 709 | 9856 | **−17% ✗** |
| UVM staging cache | per-copy overhead | (hangs) | — | **✗ net regression** |
| eager reserve (async) | fault avoidance | (732 uvm_errors) | — | **✗ fails under pressure** |

## Key finding: the bottleneck is round-trip latency, not copy cost

Four experiments triangulate the same conclusion — **reload copy cost (count or
speed) is not the dominant factor; the serialized GPU-fault round-trip is.**

- **Async offloads (+15.6%)** helped most, by moving offloads *off* the serial
  fault→wait→copy→complete critical path. This is the one clear win.
- **Bigger blocks** cut migration count ~2× (2→4 MiB) but gave only +1.9%, and
  4× (8 MiB) *regressed* −17%: fewer-but-fatter reloads each stall the GPU
  longer on the critical path. Fault *count* is not the lever.
- **UVM staging cache** made each copy ~1.8× faster (817→456 µs) yet regressed
  overall — profiling showed the daemon is ~90% idle and copies are ~8% of wall
  time, so copy *speed* is not the lever either. (Reverted; WIP at
  `/tmp/regimeB-staging-cache-final.patch`.)
- **Eager reserve** (materialize at reserve time to avoid faults) fails hard
  under pressure (732 `uvm_errors`): forcing the full KV set resident when only
  2560 MiB fits creates unresolvable contention. It is a no-pressure-only
  optimization.

The arithmetic confirms the diagnosis: reload traffic is ~40 GB over a ~22 s
prefill = ~1.8 GB/s, against ~25 GB/s of PCIe Gen4. Transfers *should* hide
behind compute; they don't only because reloads are **reactive** (fault-blocked
one at a time), not prefetched.

## Recommendation

The remaining ~−35% pressure gap is a **latency-hiding** problem, not a
bandwidth or copy-efficiency one. The highest-value next step is **speculative
reload prefetch**: prefill reads KV blocks in near-sequential order, so the
kernel (or shim) can reload blocks *ahead* of the fault so the GPU never stalls
— the same principle that made async offloads work, applied to the reload side.
This is a kernel fault-path change and is scoped as separate follow-up work.

4 MiB blocks (`POLARIS_SHIM_BLOCK_SIZE=0x400000`) are a safe, workload-specific
+1.9% and can be set per-run; not changed as a global default because 8 MiB
regresses and the effect is workload-dependent.

## Note: benchmarking discipline

Reliable pressure benchmarking requires reloading `polaris.ko` before every run.
The bench harness reloads `polarisd` but not the kernel module, so block/session
state accumulates across runs and causes intermittent setup-time
`BLOCK_RESERVE: Invalid argument` failures that are easily mistaken for code
regressions.
