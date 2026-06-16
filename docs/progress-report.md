# POLARIS Progress Report

**Generated:** 2026-06-02

> Current status note: this report predates the v4 UVM external-range path and
> is retained as historical context. The current production direction is
> documented in `docs/roadmap-v4.md`: llama.cpp KV-cache-only fault-driven
> paging through a UVM-registered fault-capable VA-space, with model weights
> and ordinary CUDA buffers left on the normal CUDA path. Do not use the older
> raw-CUDA-VMM limitation language below as the current v4 status.

## Project Summary

POLARIS (**P**aged **O**perating **L**ayer for **A**ccelerated **R**outing and **I**nference **S**ystems) is a Linux kernel module + CUDA Virtual Memory Management (VMM) system that provides OS-level paged KV Cache management for LLM inference. It implements kernel-directed page fault decisions via CUDA VMM, CPU offload/reload under memory pressure, and reference-counted copy-on-write (COW) for beam search, benchmarked against vLLM and SGLang on real NVIDIA GPU hardware.

**Core insight:** KV Cache pressure is an OS-level paging and sharing problem, not merely a per-process allocation problem. POLARIS elevates KV Cache management from in-process Python classes to an OS-level service with a universal ioctl protocol.

---

## Completed Work

### Phase 0: Research Claim Freeze **[DONE]**
- Design document and hardware setup frozen.
- Research claim defined: OS-level paging for KV Cache with kernel-authoritative block table, COW, and eviction policies.

### Phase 1a: NVIDIA UVM Fault Hook Spike **[DONE]**
- Patched `nvidia-uvm.ko` is tracked as the `third_party/open-gpu-kernel-modules`
  submodule on the `polaris-v4` branch.
- The hook (`polaris_uvm_handle_gpu_fault`, exported from `kernel/polaris_export.c`) is callable from the UVM replayable-fault bottom half via C symbol.
- `kernel/polaris_uvm.h` defines the C header for the hook.
- **Diagnostic result:** A CUDA kernel touching a raw unmapped CUDA VMM VA produces `Xid 31`, `FAULT_PDE`, `cudaErrorIllegalAddress` — raw CUDA VMM holes are *not* automatically serviced as UVM replayable faults on driver 610.43.02. The hook works but a fully transparent fault-driven path is not achievable without a different VA substrate or a deeper driver integration. The explicit prefetch/offload path remains the benchmarkable route. See `docs/fault-driven-analysis.md` for full evidence.

### Phase 1b: Kernel Module Skeleton **[DONE]**
- **File:** `kernel/polaris.rs` (2112 lines) — main kernel module in Rust-for-Linux.
- **ABI types:** `kernel/polaris_abi.rs` (289 lines) — single source of truth shared between kernel and userspace.
- **Kernel types:** `kernel/polaris_types.rs` (256 lines) — ioctl codes, flag macros, internal structs.
- **Eviction policy engine:** `kernel/polaris_policy.rs` (304 lines) — FIFO, LRU, PhaseAware.
- **C export:** `kernel/polaris_export.c` — UVM fault hook entry point.
- **Build:** `kernel/Kbuild`, `kernel/Makefile` — Kbuild integration.
- **Binaries:** `polaris.ko` built and present.

**IOCTLs implemented (15 commands):**
| Command | Purpose |
|---------|---------|
| `REGISTER_GPU` | Register a GPU with its total memory |
| `REGISTER_VA_RANGE` | Register a CUDA VMM VA range for a process |
| `SESSION_CREATE` | Create a new inference session |
| `SESSION_DESTROY` | Destroy a session and release all blocks |
| `SESSION_BRANCH` | Clone a session for beam search (increments refcounts) |
| `SESSION_GET_STATS` | Query per-session statistics |
| `BLOCK_RESERVE` | Reserve a KV block in a session |
| `BLOCK_RELEASE` | Release a KV block |
| `BLOCK_TOUCH` | Update LRU timestamp for a block |
| `BLOCK_GET_STATE` | Query block state |
| `GET_DECISION` | Poll kernel for pending executor decisions |
| `COMPLETE_OPERATION` | Report executor completion back to kernel |
| `GET_GLOBAL_STATS` | Read global statistics |
| `LIST_SESSIONS` | Enumerate active sessions |
| `SET_POLICY` | Switch eviction policy at runtime (0=FIFO, 1=LRU, 2=PhaseAware) |

**Block states (9):** Unmapped, CowPending, AllocPending, ReloadPending, OffloadPending, FreePending, Resident, CpuOffloaded, Evicted.

**Decision operations (7):** Alloc, Free, MapExisting, Unmap, Offload, Reload, CowBreak.

**Kernel module internals:**
- Single global mutex protecting all state (blocks, sessions, GPUs, decisions, faults).
- COW logic with two-phase overlap detection, refcount management, and CowPending state.
- Decision queue with retry (max 3 retries), GPU health flag, eviction on fatal errors.
- sysfs interface: `/sys/kernel/polaris/stats`.
- `PinnedDrop` for `PolarisDevice` handles `rmmod -f` gracefully.

### Phase 1c: In-Process CUDA VMM Runtime **[DONE]**
- **Crate:** `polaris-runtime/` — builds as rlib, staticlib (C FFI), and cdylib.
- **Core runtime:** `runtime.rs` (1305 lines) — decision poller thread, explicit KV block allocate/map/unmap/offload/reload/free.
- **CUDA VMM wrapper:** `cuda_vmm.rs` (220 lines) — wraps cuMemCreate, cuMemMap, cuMemSetAccess, cuMemUnmap, cuMemRelease, cuMemAllocHost, cuMemcpyHtoD/DtoH/DtoD, cuMemAddressReserve, cuMemAddressFree.
- **C FFI:** `lib.rs` (352 lines) — 13 `#[no_mangle]` exports: `polaris_runtime_create`, `start`, `stop`, `destroy`, `poll_once`, `alloc_kv`, `map_kv_all`, `map_kv_block`, `unmap_kv_block`, `offload_kv_block`, `reload_kv_block`, `unmap_kv`, `free_kv`, `last_error`.
- **C header:** `include/polaris_runtime.h` (64 lines) — for integration into inference engines.

### Phase 1d: Runtime Lifecycle & Resilience **[DONE]**
- **Daemon:** `polarisd/` (7 source files) — optional userspace control-plane daemon.
  - `main.rs` (248 lines): CUDA init, NVML GPU discovery, GPU+VA registration, decision loop.
  - `decision.rs` (462 lines): Decision dispatcher for all 7 ops.
  - `gpu.rs` (271 lines): GPU state tracking, first-fit VA pool allocator with coalescing.
  - `offload.rs` (325 lines): CPU pinned memory pool, offload/reload execution.
  - `lifecycle.rs` (224 lines): State reconciliation from sysfs, systemd notify.
  - `cuda_vmm.rs` (237 lines): CUDA VMM wrapper functions.
  - `nvml.rs` (48 lines): NVML GPU discovery.
- **Systemd unit file:** `polarisd/polarisd.service`.

### Phase 2a: GPU↔CPU Offload/Reload **[DONE]**
- `polarisd/src/offload.rs`: Pre-allocated CPU pinned memory pool. Block data copied GPU→CPU (offload), physical handle released, VA reserved for future reload. Reload allocates new physical, maps, copies CPU→GPU, frees CPU buffer.
- Kernel-side eviction trigger selects victims, queues `OFFLOAD`/`RELOAD` decisions.
- End-to-end data preservation validated by `polaris-runtime/tests/kv_smoke.c`.

### Phase 2b: Eviction Policies **[DONE]**
- Three policies implemented in `kernel/polaris_policy.rs` (304 lines):
  - **FIFO:** Victim = oldest `map_time_ns`.
  - **LRU:** Victim = oldest `last_touch_ns`.
  - **PhaseAware:** Scoring function with weights for age, prefill phase, GPU pressure, session priority, COW sharing, decode recency. Two-pass search (other sessions first, then own).
- Runtime policy switching via `SET_POLICY` ioctl.
- Per-policy counters reset on switch.

### Phase 3: Copy-on-Write for Beam Search **[DONE]**
- `SESSION_BRANCH` ioctl clones a parent session, increments block refcounts atomically.
- COW break triggered on write (`BLOCK_RESERVE` with `RESERVE_FLAG_OVERWRITE`).
- COW break: new physical allocation + GPU→GPU copy of shared data.
- Validated by `workloads/src/main.rs` with 7 comprehensive COW break tests (`polaris-workload cow-break`).

### Phase 4a: llama.cpp Runtime Integration **[DONE]**
- Integration plan documented at `integrations/llama.cpp/README.md`.
- Minimal patch plan: CMake `LLAMA_POLARIS=ON` option, link `libpolaris_runtime`, reserve POLARIS VA span for KV tensors.
- External patches exist (referenced from `docs/roadmap.md`).
- Opt-in environment variables: `LLAMA_POLARIS=1`, `LLAMA_POLARIS_KV=1`, `LLAMA_POLARIS_KV_BLOCK_API=1`, `LLAMA_POLARIS_FAULT_WORKER=1`.

### CLI Tool **[DONE]**
- `polarisctl/` — 5 subcommands: `stats`, `create-session`, `destroy-session`, `list-sessions`, `set-policy`.

### Workload Generator **[DONE]**
- `workloads/` — 4 subcommands:
  - `synthetic-kv`: Single-session KV reserve/touch (flat or decode-loop).
  - `beam-search`: Beam search with parent/child sessions and COW break.
  - `concurrent`: Multi-session round-robin allocation.
  - `cow-break`: 7-test comprehensive COW break validation.

### Userspace Library **[DONE]**
- `libpolaris/` — ioctl wrappers, auto-copied ABI types from kernel source via `build.rs`.

### Benchmark Infrastructure **[PARTIALLY DONE]**
- **Configs:** `benchmarks/configs/` — beam_search.toml, concurrent.toml, long_context.toml, memory_pressure.toml.
- **Trace patches:** `benchmarks/patches/vllm/` and `benchmarks/patches/sglang/` — monkey-patches for trace collection.
- **Validation script:** `benchmarks/scripts/validate_1b.sh` — Phase 1b GPU memory accounting validation.

### Documentation **[DONE]**
- `docs/roadmap.md` (1600 lines) — comprehensive engineering roadmap, architecture decisions, milestones, weekly schedule.
- `docs/fault-driven-analysis.md` (233 lines) — UVM replayable fault feasibility evidence.
- `docs/nvidia-memory-allocation-api-report.md` (501 lines) — NVIDIA API requirements analysis.
- `integrations/llama.cpp/README.md` (77 lines) — llama.cpp integration plan.

### Tests
- `polaris-runtime/tests/fault_smoke.rs` — Rust integration test (requires hardware, `#[ignore]`).
- `polaris-runtime/tests/kv_smoke.c` — C end-to-end test: explicit KV block lifecycle.
- `polaris-runtime/tests/uvm_fault_smoke.cu` — CUDA diagnostic: raw VMM fault behavior.

---

## In Progress

| Item | Status | Detail |
|------|--------|--------|
| **Phase 4b — Benchmark suite** | Stub | `benchmarks/scripts/run_all.sh` prints "not yet implemented". Workload generator exists and COW tests exist, but the automated benchmark runner with CSV/plot output is not built. |
| **M3 — Block-level offload/reload policy integration** | Implementation present, end-to-end integration variable | The runtime API for explicit per-block map/unmap/offload/reload is complete and smoke-tested. Real policy integration (deciding *which* llama KV blocks are cold, offloading only those, ensuring attention-required blocks are resident) is partially integrated via the `LLAMA_POLARIS_KV_BLOCK_API` mode. |
| **M4 — Fault-driven transparent paging** | Known limitation | The fault-decision protocol (kernel→runtime→CUDA VMM) is verified working. However, raw CUDA VMM VA holes do not produce replayable UVM faults on driver 610.43.02. A true transparent fault-driven flow requires a different VA substrate or deeper driver hook. This is documented as an architectural limitation, not a bug. |

---

## Not Yet Implemented

| Phase | Item | Priority |
|-------|------|----------|
| 4b | **Benchmark runner** (`benchmarks/scripts/run_all.sh`) — automated execution of all scenarios with metric collection, CSV output, and plotting (`benchmarks/scripts/plot.py` referenced but not present). | **High** |
| 4c | **vLLM trace replay** — replay vLLM traces through POLARIS workload engine for comparison. Trace collection patches exist. | **High** |
| 4d | **SGLang trace replay** — replay SGLang traces through POLARIS workload engine. Trace collection patches exist. | **High** |
| 4e | **Head-to-head comparison matrix** — compare POLARIS vs. vLLM vs. SGLang across 5 scenarios (long context, concurrent, beam search, memory pressure, synthetic). Metrics: fragmentation, peak memory, offload/reload counts, COW savings. | **High** |
| 4f | **vLLM integration adapter** — drop-in `polaris_block_manager.py` replacing vLLM's `BlockSpaceManager`. ~100 lines of Python in a single file. Optional/stretch goal. | Low |
| 5 | **eBPF network offload** — XDP eBPF program for zero-copy inference request ingestion. Optional/stretch. | Low |

### Deferred Directories
- **`adapter/`** — contains only `.gitkeep`. Reserved for Phase 4f vLLM adapter.
- **`ebpf/`** — contains only `.gitkeep`. Reserved for Phase 5 eBPF XDP program.

### Referenced but Missing Files (from roadmap)
- `benchmarks/scripts/collect_vllm_trace.py` — referenced in README and roadmap but not present in repo. Trace patches exist as `.patch` files.
- `benchmarks/scripts/collect_sglang_trace.py` — same.
- `benchmarks/scripts/plot.py` — referenced but not present.
- `polarisctl/src/stats.rs`, `session.rs`, `debug.rs` — referenced in roadmap but all logic is consolidated into single `main.rs`.
- `workloads/src/trace_replay.rs` — referenced in README but trace replay is not yet implemented as a separate module.

---

## Repository Structure (Actual)

```
/home/wano/workspace/Polaris/
├── kernel/                    # polaris.ko — Linux kernel module (Rust-for-Linux)
│   ├── polaris.rs             # Main module (2112 lines)
│   ├── polaris_abi.rs         # ABI types shared with userspace (289 lines)
│   ├── polaris_types.rs       # Kernel-internal types, ioctl codes (256 lines)
│   ├── polaris_policy.rs      # Eviction policies: FIFO, LRU, PhaseAware (304 lines)
│   ├── polaris_export.c       # C export for UVM fault hook symbol
│   ├── polaris_uvm.h          # C header for UVM fault hook
│   ├── polaris_rust.rs        # Includes polaris.rs
│   ├── Kbuild / Makefile
│   └── polaris.ko             # Built module binary
├── libpolaris/                # Userspace client library (Rust)
│   ├── src/ioctl.rs           # IOCTL command codes and low-level wrappers
│   ├── src/types.rs           # include! of generated ABI types
│   ├── src/lib.rs
│   └── build.rs               # Copies kernel/polaris_abi.rs → userspace
├── polaris-runtime/           # In-process CUDA VMM executor with C FFI
│   ├── src/runtime.rs         # Core runtime (1305 lines)
│   ├── src/cuda_vmm.rs        # CUDA VMM wrapper (220 lines)
│   ├── src/lib.rs             # C FFI exports (352 lines)
│   ├── include/polaris_runtime.h
│   ├── tests/fault_smoke.rs   # Integration test (ignored)
│   ├── tests/kv_smoke.c       # C smoke test
│   └── tests/uvm_fault_smoke.cu  # CUDA diagnostic
├── polarisd/                  # Userspace daemon (Rust binary, optional)
│   ├── src/main.rs            # (248 lines)
│   ├── src/decision.rs        # (462 lines)
│   ├── src/gpu.rs             # (271 lines)
│   ├── src/offload.rs         # (325 lines)
│   ├── src/lifecycle.rs       # (224 lines)
│   ├── src/cuda_vmm.rs        # (237 lines)
│   ├── src/nvml.rs            # (48 lines)
│   └── polarisd.service
├── polarisctl/                # CLI tool (Rust binary)
│   └── src/main.rs            # (198 lines, 5 subcommands)
├── workloads/                 # Synthetic workload generator (Rust binary)
│   └── src/main.rs            # (1087 lines, 4 subcommands)
├── integrations/
│   └── llama.cpp/README.md    # Integration plan (77 lines)
├── benchmarks/
│   ├── configs/               # 4 TOML workload configs
│   ├── patches/
│   │   ├── vllm/              # trace_kv_cache.patch + README
│   │   └── sglang/            # trace_alloc.patch + README
│   └── scripts/
│       ├── run_all.sh         # STUB — prints "not yet implemented"
│       └── validate_1b.sh     # Phase 1b validation script
├── adapter/                   # .gitkeep only — deferred
├── ebpf/                      # .gitkeep only — deferred
├── scripts/                   # Build tooling (gen-rust-project.sh, etc.)
├── third_party/
│   └── open-gpu-kernel-modules/  # Git submodule: patched nvidia-uvm.ko
├── docs/
│   ├── roadmap.md             # (1600 lines)
│   ├── fault-driven-analysis.md  # (233 lines)
│   ├── nvidia-memory-allocation-api-report.md  # (501 lines)
│   └── progress-report.md     # This file
├── Cargo.toml                 # Workspace: 5 members
├── Cargo.lock
├── Makefile                   # kernel + userspace builds
├── .gitignore
├── .gitmodules
└── README.md                  # (95 lines)
```

---

## Build Instructions

```bash
# Build kernel module
make kernel
sudo insmod kernel/polaris.ko

# Build all userspace
make userspace    # runs `cargo build --release`

# Build specific components
cargo build -p polarisd
cargo build -p polaris-runtime
cargo build -p polarisctl
cargo build -p workloads

# Run daemon
sudo systemctl start polarisd

# CLI
polarisctl stats
polarisctl create-session --gpu 0 --vas-bytes 1073741824
polarisctl list-sessions
polarisctl set-policy 2   # PhaseAware
```

**Prerequisites:**
- Linux kernel with `CONFIG_RUST=y`
- NVIDIA GPU (Turing+ / compute 7.5+) with 610.43.02 driver
- CUDA Toolkit 13.2
- Patched `nvidia-uvm.ko` from `third_party/open-gpu-kernel-modules`
- Rust toolchain matching kernel's built-with version (Arch: pacman rustc preferred)

---

## Key Technical Decisions

1. **Single global mutex** protects all kernel module state — sufficient for single-GPU prototype.
2. **Single source of truth** for ABI types in `kernel/polaris_abi.rs`, auto-copied to userspace by `libpolaris/build.rs`.
3. **In-process CUDA VMM executor** (not cross-process daemon) because CUDA VMM mappings are context-local. The daemon (`polarisd`) is optional, for control-plane tasks only.
4. **Explicit prefetch/offload** is the benchmarkable path. True fault-driven UVM replay remains a research item due to raw CUDA VMM hole limitations.
5. **COW for beam search** uses refcount-based sharing; COW break creates new physical allocation + GPU→GPU copy.
6. **CPU offload** uses pre-allocated pinned memory pool (currently hardcoded 4 GiB in `polarisd`).
7. **Multi-GPU routing** is explicitly out of scope.

---

## Known Limitations

1. **No transparent fault-driven paging:** Raw CUDA VMM VA holes produce `Xid 31` / `cudaErrorIllegalAddress`, not replayable UVM faults. This is a driver-level limitation documented in `docs/fault-driven-analysis.md`.
2. **Single-GPU only:** GPU 0 is hardcoded in `polarisd/src/main.rs:35`.
3. **CPU pool hardcoded:** 4 GiB fixed size in `polarisd/src/main.rs:70`.
4. **No cross-process CUDA VMM mapping:** Each process needs its own in-process runtime because CUDA VMM is context-local.
5. **No multi-address-space COW:** COW sharing works within a single process's sessions. Cross-process COW is not yet addressed.

---

## Next Steps (Priority Order)

1. **Implement benchmark runner** — `benchmarks/scripts/run_all.sh` needs to:
   - Load configs from `benchmarks/configs/*.toml`.
   - Run `polaris-workload` subcommands with appropriate parameters.
   - Capture metrics (GPU memory, offload/reload counts, COW stats) via `polarisctl stats`.
   - Output CSV results to `benchmarks/results/`.

2. **Implement trace replay** — `workloads/src/trace_replay.rs` module that reads vLLM/SGLang trace files and replays allocation decisions through POLARIS ioctls.

3. **Run comparison benchmarks** — Head-to-head: llama.cpp baseline vs. llama.cpp+POLARIS, and POLARIS trace replay vs. vLLM/SGLang native on the 4 configured scenarios (or 5th synthetic).

4. **Generate plots and final report** — From CSV results, produce fragmentation/comparison plots and write the final evaluation report.

5. **Stretch:** vLLM adapter (`adapter/polaris_block_manager.py`) and eBPF XDP program.

---

## Code Quality Notes

- Zero `todo!()`, `unimplemented!()`, or stubbed function bodies in any source file.
- Zero commented-out code blocks.
- All `_ => {}` match arms in the kernel module are correctly handling remaining/edge cases with explicit behavior or safe fallbacks, not stubs.
- One outdated comment: `polarisctl/src/main.rs:8` says "Phase 1a skeleton" but all 5 subcommands are fully implemented.
- Test that requires hardware is `#[ignore]`-annotated with clear rationale.
