# POLARIS — Results Summary

*Concise synthesis of the POLARIS v4 experiments. Last data: 2026-07-08.*

## What POLARIS is

OS-level **KV-cache paging** for LLM inference on NVIDIA GPUs: a kernel module
(`polaris.ko`) + patched NVIDIA UVM fault hook + `LD_PRELOAD` shim + userspace
daemon (`polarisd`). Model weights and ordinary CUDA buffers stay on the normal
CUDA path; only the **llama.cpp KV-cache** is routed through Polaris-managed,
fault-capable GPU virtual addresses that can spill/reload under VRAM pressure.

Reference hardware: RTX 5070 Ti (16 GiB), driver 610.43.02, `v4-fault-hook`.

## Headline findings

1. **Correctness holds under real pressure.** `uvm_errors = 0` across every
   trial — including 9 clean-module Qwen2.5-14B 16k runs with active
   spill/reload (5308 offloads / 5052 reloads each). The KV-only fault-driven
   path works end-to-end.

2. **Prefill-under-pressure closed from −44% to −21% vs native** (2026-07-08).
   Two opt-in optimizations, +41.5% cumulative on Qwen14B 16k @ 2560 MiB:
   **async offloads** (run eviction copies off the fault-critical path, +15.6%)
   and **speculative reload prefetch** (exploit ~73% sequential KV access to
   materialize block N+1 before the GPU faults on it, +20.2%). The bottleneck
   was the *serialized GPU-fault round-trip latency*, not copy cost — proven by
   four experiments (a staging cache that made copies 1.8× faster *regressed*).

3. **OS-level COW for beam search works and is 38% faster** than independent
   re-prefill (0.18 s vs 0.29 s), sharing 256 MiB across 7 children at ~37 µs
   per branch.

4. **Multi-session scheduler scales cleanly** to 32 concurrent sessions
   sub-second; evictions are exactly `20 × N`.

## Throughput at a glance (Qwen2.5-14B Q4_K_M, 16384×32)

| mode | prompt tok/s | gen tok/s | notes |
|---|---:|---:|---|
| native CUDA | ~1310 | 76.5 | baseline, no paging |
| POLARIS no-pressure (16 GiB) | **~1285** | ~75.7 | near-native; fault-mapping overhead only |
| POLARIS pressure 2560 MiB — sync baseline | 736 | ~75 | pre-optimization (2026-06-30) |
| POLARIS pressure 2560 MiB — + async offloads | 851 | ~75 | +15.6% |
| POLARIS pressure 2560 MiB — + prefetch (4 MiB blocks) | **1042** | ~75.7 | **+41.5% cumulative; −21% vs native** |

- **No-pressure prefill is already near-native (~1285, −2%).** The old "~820 /
  −37%" figure was pre-06-30 data; the `driver_cache` short-circuit + eager shim
  reserve closed that gap. The remaining prefill penalty is only under *pressure*.
- **The pressure penalty is a latency-hiding problem, not a copy-cost one.**
  Reload traffic is ~40 GB over a ~22 s prefill = ~1.8 GB/s vs ~25 GB/s of PCIe;
  transfers hide behind compute once reloads are made proactive (prefetch)
  instead of fault-blocked. Both optimizations are now **on by default**
  (disable with `POLARISD_ASYNC_OFFLOAD=0` / writing `0` to
  `/sys/kernel/polaris/prefetch`); prefetch is a clean no-op when the working
  set fits (verified no-pressure and 3072 MiB).
- **Decode throughput is free only while the active KV fits the budget**
  (~75 vs 76 native at low pressure), unchanged by these prefill optimizations.
  Once decode itself must page, it falls to ~21–30% of native (see the pressure
  rows in "Compared with plain llama.cpp" below) — this case is not yet addressed.

## Compared with plain llama.cpp: gains and losses

POLARIS is **not** a speedup over stock llama.cpp. On any workload that fits in
VRAM, plain llama.cpp is faster on both prefill and decode. The comparison below
is same-engine (llama.cpp on both sides); only the KV allocator differs.

### Honest scoping of the "gains"

- **Oversubscription is UVM's, not POLARIS's.** The ability to back a GPU VA
  with off-VRAM memory and remap on fault is provided by the patched NVIDIA UVM
  fault + external-map bridge. A plain `cudaMallocManaged` KV allocation would
  also let the KV set exceed VRAM. **No report here benchmarks POLARIS against a
  plain-UVM-managed-KV baseline**, so we cannot claim POLARIS oversubscribes
  *better* than UVM — only that it does so under explicit KV-aware control.
- What POLARIS genuinely adds *on top of* UVM:
  - **KV-block-granular, daemon-controlled eviction** to a CPU pool (policy +
    telemetry), instead of UVM's opaque page-level migration.
  - **Cross-session COW / refcount sharing** of prefix KV — something stock UVM
    managed memory does not do. Beam search shares 256 MiB across 7 children,
    **38% faster** than re-prefill (baseline is POLARIS re-prefill, not UVM).
  - **KV-only selective scope** via the shim — weights stay on the normal CUDA
    path, with no source changes to llama.cpp.
- **Correctness under real spill** is demonstrated: Qwen14B @ 16k on a 16 GiB
  card under a 2560 MiB budget, 5308 offloads / 5052 reloads, `uvm_errors = 0`.

### What you lose

Budget context matters: the prefill rows below are at a **2560 MiB** budget
(n=3); the **decode-under-pressure** rows are at a deliberately pathological
**4 MiB** budget, `llama-bench -p 128 -n 32` capability probe (SmolLM2 n=3;
Qwen effectively n=1) — a "does it survive extreme oversubscription" result,
not a tuned-performance result. The two "under pressure" regimes are not
comparable.

| metric | plain llama.cpp | POLARIS | delta |
|---|---:|---:|---:|
| Qwen14B prefill, no-pressure (16 GiB) | ~1310 | ~1285 | **−2%** |
| Qwen14B prefill, under spill (2560 MiB, async+prefetch, n=3) | ~1310 | ~1042 | **−21%** |
| Qwen14B **decode, no-pressure** | 76.81 | 75.65 | −1.5% |
| Qwen14B **decode, 4 MiB budget capability probe (n=1)** | 76.81 | 22.83 | **−70% (→30% of native)** |
| SmolLM2 **decode, no-pressure** | 921.77 | 750.73 | −19% |
| SmolLM2 **decode, 4 MiB budget capability probe (n=3)** | 921.77 | 196.38 | **−79% (→21% of native)** |

- **Prefill is ~1.27× slower under spill** (was ~1.6–1.8× before the 2026-07-08
  optimizations), and near-native (−2%) when the KV fits. The gap that remains
  under pressure is the residual fault-servicing cost the sequential prefetch
  cannot fully hide.
- **Decode is not improved — and collapses under pressure.** It is near-native
  *only when the active KV fits the residency budget* (no spill). Once decode
  itself must page, throughput falls to **~21–30% of native**. The prefill
  optimizations do not touch this — decode-under-pressure remains open work.
- **Long-context *generation* under pressure is essentially untested.** The
  near-native decode numbers come from decode tails where most KV stays
  resident. Sustained long-generation with a spilled KV history — the
  thrashing-prone case — is listed as open work in the README.
- **There is a thrashing floor.** Too small a budget (1 GiB on the 14B model,
  or 2048 MiB with these optimizations) drives runaway offload/reload that never
  completes — the prefetch/async wins apply in the *productive* paging regime,
  not the pathological one.
- **Operational cost:** patched NVIDIA UVM, root, an out-of-tree kernel module +
  daemon, and `uvm_enable_builtin_tests=1`. Plain llama.cpp needs none of this.

**Net:** POLARIS buys *KV-aware, shareable, controllable* off-VRAM KV residency —
not speed. Prefill is near-native when the KV fits (−2%) and ~1.27× under spill
after the 2026-07-08 optimizations; decode is near-native only while the working
set fits the budget and degrades to ~1/3–1/5 native once decode itself pages.
Whether this beats plain UVM-managed KV is not yet measured.

## Policy comparison (attention-stream trace, n=3, clean module)

| policy | avg_ts | offloads | reloads | peak VRAM | approx hit rate |
|---|---:|---:|---:|---:|---:|
| attention_stream | 383.57 | 5308 | 5052 | 2564 MiB | 0.788 |
| phase_aware | 383.06 | 5308 | 5052 | 2564 MiB | 0.784 |
| fifo | 384.59 | 5308 | 5052 | 2564 MiB | 0.777 |
| lru | 384.89 | 5308 | 5052 | 2564 MiB | 0.786 |

All four are statistically indistinguishable. The prefill-dominated workload
carries no victim-selection signal the kernel can currently act on:
`token_start` is a Polaris chunk index, not a model token position, and no KV
access-phase hint is wired through yet (`KV hint epoch = 0`).

## Bottom line

The KV-only paging path is **correct and functional** for the llama.cpp
integration: a 14B model runs at 16k context on a 16 GiB card under forced
spill/reload with no correctness errors. Prefill is near-native when the KV fits
and, after the 2026-07-08 optimizations, ~1.27× under spill (gap −44% → −21% vs
native); decode stays near-native **as long as the active KV fits the residency
budget** and still collapses to ~1/3–1/5 native once decode itself pages. The
oversubscription capability itself is UVM's; POLARIS's contribution is the
KV-aware control, sharing, and eviction layered on top — whose value over plain
UVM-managed KV is **not yet benchmarked**.

The optimization study **corrected the earlier forward-look**: cutting RM copy
*cost* is **not** the lever — a UVM staging cache that made each copy 1.8× faster
regressed overall, because the daemon is ~90% idle and copies are ~8% of wall
time. The bottleneck is the **serialized GPU-fault round-trip latency**, and the
wins came from hiding migration behind compute: **async offloads** (off the
critical path) and **sequential-access prefetch** (a concrete realization of the
workload-aware-hint direction). Remaining open work: **decode-under-pressure**
(untouched by these prefill optimizations), a "don't-prefetch-when-budget-
critical" guard, and the intermittent setup-time `BLOCK_RESERVE` flake.

## Sources

- `benchmarks/reports/prefill-pressure-optimization-study-20260708.md` (async
  offloads, block-size sweep, staging cache, eager reserve, prefetch, and the
  broader validation across budgets/shapes)
- `benchmarks/reports/2026-06-30-final-experiments.md` (driver_cache ablation,
  clean policy compare, concurrent scheduler, beam COW)
- `benchmarks/analysis/policy-trace-attention-stream-r3-20260702T113904Z/summary.md`
- `benchmarks/reports/qwen14b-polaris-policy-compare-20260617.md`
- `benchmarks/reports/qwen14b-polaris-kv-16k-20260616.md`
- `benchmarks/reports/llama-polaris-vs-original-frameworks-20260616.md`
- `docs/llm-serving-polaris-injection-comparison-20260619.md` (decode-under-
  pressure 4 MiB capability probe numbers in the "What you lose" table)
