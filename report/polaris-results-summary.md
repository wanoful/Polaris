# POLARIS — Results Summary

*Concise synthesis of the POLARIS v4 experiments. Last data: 2026-07-02.*

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

2. **~5.9× prompt-throughput gain from hot-path optimizations** (2026-06-17 →
   2026-06-30) on Qwen14B 16k @ 2560 MiB budget. The `driver_cache` in-kernel
   fault short-circuit + eager shim reserve cut redundant bridge maps from
   **2.17 M → 8,148**, taking prompt throughput from ~154 to **~736 tok/s**.

3. **Eviction policy is not the right optimization axis** for this workload.
   FIFO / LRU / PhaseAware / attention-stream converge within **0.5%** once the
   module is reloaded between trials. Migration counts (offloads, reloads,
   bridge maps) show **zero variance** across policies; working-set Jaccard =
   1.000. The old "FIFO 115 vs LRU 70 tok/s" gap was a **module-state
   confounder**, now eliminated.

4. **OS-level COW for beam search works and is 38% faster** than independent
   re-prefill (0.18 s vs 0.29 s), sharing 256 MiB across 7 children at ~37 µs
   per branch.

5. **Multi-session scheduler scales cleanly** to 32 concurrent sessions
   sub-second; evictions are exactly `20 × N`.

## Throughput at a glance (Qwen2.5-14B Q4_K_M, 16384×32)

| mode | prompt tok/s | gen tok/s | notes |
|---|---:|---:|---|
| native CUDA | 1310.95 | 76.48 | baseline, no paging |
| POLARIS no-pressure (16 GiB) | ~820 | ~75.5 | fault-mapping overhead only |
| POLARIS pressure (2560 MiB, **optimized 06-30**) | **~736** | ~75.0 | real spill/reload |
| POLARIS pressure (2560 MiB, pre-opt 06-17) | ~154 (FIFO) | ~74 | before short-circuit |

- **Decode throughput is free only while the active KV fits the budget**
  (~75 vs 76 native at low pressure) — the paging cost is paid in prefill. Once
  decode itself must page, it falls to ~21–30% of native (see the pressure rows
  in "Compared with plain llama.cpp" below).
- Prefill overhead vs native reflects the fault-driven mapping path; the
  short-circuit fires ~46–47k times/run on long prompts (it is dead code on
  short-prompt SmolLM2 workloads, where UVM's own PTE cache absorbs the faults).

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

### What you losez

| metric | plain llama.cpp | POLARIS | delta |
|---|---:|---:|---:|
| Qwen14B prefill, no-pressure (16 GiB) | 1310.95 | 820.32 | **−37%** |
| Qwen14B prefill, under spill (2560 MiB, optimized) | 1310.95 | ~736 | **−44%** |
| Qwen14B **decode, no-pressure** | 76.81 | 75.65 | −1.5% |
| Qwen14B **decode, under budget pressure** | 76.81 | 22.83 | **−70% (→30% of native)** |
| SmolLM2 **decode, no-pressure** | 921.77 | 750.73 | −19% |
| SmolLM2 **decode, under budget pressure** | 921.77 | 196.38 | **−79% (→21% of native)** |

- **Prefill is 1.6–1.8× slower** — fault-driven mapping + UVM bridge crossings
  on first touch of each KV chunk. Even the ~266× bridge-map reduction
  (§ headline #2) does not close this vs a pure in-VRAM allocator.
- **Decode is not improved — and collapses under pressure.** It is near-native
  *only when the active KV fits the residency budget* (no spill). Once decode
  itself must page, throughput falls to **~21–30% of native**, because the run
  spends most of its time in daemon RM copy / offload / reload. The RM copy path
  is the throughput limiter, not the policy.
- **Long-context *generation* under pressure is essentially untested.** The
  near-native decode numbers come from 32-token decode tails where most KV stays
  resident. Sustained long-generation with a spilled KV history — the
  thrashing-prone case — is listed as open work in the README.
- **There is a thrashing floor.** Too small a budget (1 GiB on the 14B model)
  drives runaway offload/reload — >27k migrations in ~4 min, never completing.
- **Operational cost:** patched NVIDIA UVM, root, an out-of-tree kernel module +
  daemon, and `uvm_enable_builtin_tests=1`. Plain llama.cpp needs none of this.

**Net:** POLARIS buys *KV-aware, shareable, controllable* off-VRAM KV residency —
not speed. Prefill costs ~1.6–1.8×; decode is near-native only while the working
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
spill/reload with no correctness errors, and decode stays near-native **as long
as the active KV fits the residency budget**. It does not make llama.cpp faster
— prefill costs ~1.6–1.8× and decode collapses to ~1/3–1/5 native once decode
itself pages. The oversubscription capability itself is UVM's; POLARIS's
contribution is the KV-aware control, sharing, and eviction layered on top —
whose value over plain UVM-managed KV is **not yet benchmarked**. The next gains
are **not** in eviction policy but in cutting RM copy / offload cost and in
exposing **workload-aware KV access hints** (prefill/decode phase, active
window) so the policy can protect the true working set — see the
[Vidmem Access Bit Buffer] direction.

## Sources

- `benchmarks/reports/2026-06-30-final-experiments.md` (driver_cache ablation,
  clean policy compare, concurrent scheduler, beam COW)
- `benchmarks/analysis/policy-trace-attention-stream-r3-20260702T113904Z/summary.md`
- `benchmarks/reports/qwen14b-polaris-policy-compare-20260617.md`
- `benchmarks/reports/qwen14b-polaris-kv-16k-20260616.md`
- `benchmarks/reports/llama-polaris-vs-original-frameworks-20260616.md`

[Vidmem Access Bit Buffer]: ../benchmarks/README.md
