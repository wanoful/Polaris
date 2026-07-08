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
| **prefetch (async, 4 MiB)** | **fault-round-trip latency** | **1042 (n=3)** | 14526 | **+20.2% ✓ shipped** |

## Full stack vs native

| config | prompt tok/s | vs native (1317) | cumulative |
|---|---:|---:|---:|
| baseline (sync, 2 MiB) | 736 | −44% | — |
| + async offloads | 851 | −35% | +15.6% |
| + 4 MiB blocks | 867 | −34% | +17.8% |
| + prefetch | **1042** | **−21%** | **+41.5%** |

The pressure prefill gap went from −44% to −21% vs native — **82% of the
no-pressure ceiling (~1268)**, which is the practical target since native
llama.cpp cannot run this oversubscribed case at all (its KV would not fit).

## Prefetch broader validation

Prefetch validated across the pressure spectrum and workload shapes (Qwen14B,
async offloads on, 4 MiB blocks, fresh module per run):

| case | budget | pf=off | pf=on | delta | note |
|---|---|---:|---:|---:|---|
| no-pressure 16k/32 | 16 GiB | 1285.4 | 1282.8 | −0.2% | no-op: 0 reloads |
| pressure 16k/32 | 3072 MiB | 1285.1 | 1282.3 | −0.2% | no-op: working set fits |
| pressure 16k/32 | 2560 MiB | 864.5 | 1036.2 | **+19.9%** | headline case |
| pressure 16k/512 | 2560 MiB | 861.0 | 1038.9 | **+20.7%** | decode-heavy |
| pressure 16k/32 | 2048 MiB | — | — | — | thrash floor: neither arm completes |

- **Clean no-op when the working set fits** (no-pressure and 3072 MiB): prefetch
  correctly does nothing — 0 reloads, throughput within noise. No regression.
- **+20% holds under long decode** (16k/512): the prefill gain is unchanged and
  **decode throughput (gen) is identical (~75.8) with or without prefetch** — the
  decode tail stays mostly resident, so prefetch neither helps nor harms it.
- **2048 MiB is a pre-existing thrash floor** (>14k reloads, never completes)
  for both arms — prefetch does not change that boundary.
- `uvm_errors = 0` in every completing run. Note: budget-marginal setup has an
  intermittent `BLOCK_RESERVE: Invalid argument` flake at token ≈18 (before any
  prefetch fires — prefetch triggers only on reload *completion*); it hit both
  arms and clean retries succeed, so it is unrelated to prefetch.

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

## Outcome

The diagnosis (latency-hiding, not bandwidth or copy-efficiency) was confirmed
and then acted on: **speculative reload prefetch** exploits the ~73% sequential
access to materialize block N+1 before the GPU faults on it, turning a blocking
round-trip into a cheap bridge-map. Gated behind `/sys/kernel/polaris/prefetch`,
it delivered **+20.2%** (867 → 1042 tok/s) — the single largest lever, exactly
because it attacks the diagnosed bottleneck. Combined with async offloads it
closes the pressure prefill gap from −44% to −21% vs native (82% of the
no-pressure ceiling).

The arithmetic that predicted this: reload traffic is ~40 GB over a ~22 s
prefill = ~1.8 GB/s, against ~25 GB/s of PCIe Gen4 — transfers *can* hide behind
compute; they only failed to because reloads were **reactive** (fault-blocked
one at a time). Prefetch makes them proactive.

4 MiB blocks (`POLARIS_SHIM_BLOCK_SIZE=0x400000`) are a safe, workload-specific
+1.9% and can be set per-run; not changed as a global default because 8 MiB
regresses and the effect is workload-dependent.

## Note: benchmarking discipline

Reliable pressure benchmarking requires reloading `polaris.ko` before every run.
The bench harness reloads `polarisd` but not the kernel module, so block/session
state accumulates across runs and causes intermittent setup-time
`BLOCK_RESERVE: Invalid argument` failures that are easily mistaken for code
regressions.
