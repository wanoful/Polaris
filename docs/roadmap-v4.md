# POLARIS Roadmap v4

Date: 2026-06-10

This roadmap supersedes v3. v3 adopted Method A (explicit-lease CUDA VMM
paging) because raw CUDA VMM VA holes produce fatal MMU faults, not
serviceable UVM replayable faults. v4 keeps that diagnosis but takes a
different exit: instead of pushing residency decisions up into the framework
via leases, host workers in a fault-capable VA-space owned by UVM and let
faults flow through a hook UVM calls before its own managed-fault servicing.

The worker is oblivious. polarisd decides. polaris.ko executes.

## Direction

> Transparent kernel-directed KV paging on a UVM-registered fault-capable
> VA-space. A small LD_PRELOAD shim hides VA-space setup. polaris.ko owns
> a copy-engine channel, a VRAM partition, and all PTE installs. polarisd
> owns the block table and policy. Workers run unmodified.

The lease API from v3 is not part of the production path. v3 lease code may
remain as a fallback for diagnostics and for non-fault-capable contexts but
should not be the integration target.

## Why this works where raw CUDA VMM did not

- Raw `cuMemCreate`/`cuMemMap` VA-spaces are not fault-capable. Unmapped
  access is fatal.
- A VA-space created with `NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING |
  IS_EXTERNALLY_OWNED` and registered through `UvmRegisterGpuVaSpace` *is*
  fault-capable. UVM's replayable-fault ISR services it.
- Adding one hook in UVM's replayable-fault bottom half lets polaris.ko
  intercept faults belonging to Polaris-managed ranges before UVM's managed
  VA lookup runs.

## Component Layout

```
worker process                shim (LD_PRELOAD)         polaris.ko             polarisd
─────────────────             ─────────────────         ──────────             ────────
CUDA app code                 intercept allocator,      block→worker map,      logical block
(unmodified)                  steer into Polaris        VRAM partition,        table, policy,
                              fault-capable VA-space    CE channel,            pinned host pool
                                                        UVM fault hook         orchestration
```

| Layer | Owns | Does not own |
|-------|------|--------------|
| worker | its CUDA context, the VA-space object, its own kernels | mapping, unmapping, block IDs, eviction |
| shim | per-process VA-space creation, allocator interception, ioctl registration with polaris.ko | data movement, policy, block state |
| polaris.ko | VRAM partition, CE channel, PTE installs into all registered worker VA-spaces, block→worker mapping, fault hot path | logical block lifecycle, eviction decisions, host buffer slot assignment |
| polarisd | block table, eviction/spill policy, host pinned pool slot assignment, COW arbitration | PTE writes, CE submissions, fault servicing |

## Driver-side changes (open-gpu-kernel-modules)

Rebase from `polaris-610.43.02`. Discard the `pvm-driver` branch entirely.
Land three small commits against stock UVM, no RM changes.

1. `uvm: export polaris fault hook registration`
   - Add `polaris_uvm_register_hook(struct polaris_uvm_ops *)`,
     `polaris_uvm_unregister_hook()`. Hook ops vector carries one function:
     `int handle_gpu_fault(u32 gpu_id, u64 va_space_token, u64 addr, u32 access)`.
   - Weak default returns `POLARIS_UVM_FAULT_NOT_MINE`.
   - `EXPORT_SYMBOL_GPL`.

2. `uvm: call polaris hook in replayable fault service path`
   - In `uvm_gpu_replayable_faults.c`, inside `service_fault_batch_dispatch`
     (called from `service_fault_batch`), after the per-fault VA-space has
     been resolved by `uvm_parent_gpu_fault_entry_to_va_space` and before
     the managed/ATS branch dispatches, call the registered hook with the
     decoded `(gpu, va_space_token, addr, access)`.
   - `HANDLED` → mark fault for replay and skip UVM managed/ATS servicing.
   - `NOT_MINE` → fall through to stock UVM path.
   - `ERROR` → route through existing UVM fatal/cancel path.
   - Define `va_space_token` as a stable kernel handle the shim and
     polaris.ko both reference (RM VA-space handle is the natural choice).

3. `uvm: expose pmm reservation entry point` (only when needed)
   - `EXPORT_SYMBOL_GPL` wrappers for `uvm_pmm_gpu_alloc_kernel` /
     `uvm_pmm_gpu_free` so polaris.ko can carve a VRAM partition at module
     init.
   - Defer until commit 1+2 are working; option A (use PMM directly) first.

4. `uvm: export channel manager and push primitives for polaris`
   - `EXPORT_SYMBOL_GPL` a minimal CE-channel surface that lets polaris.ko
     run spill/reload copies through UVM's existing channel infrastructure
     instead of standing up its own: channel-manager create/destroy,
     `uvm_push_begin`/`_end`, the CE memcopy helper, and tracker wait.
   - Defer until M3 actually needs it; land commits 1+2 first.

Total expected diff against stock UVM: ~150–200 LOC, almost all in the
export wrappers. Survives driver bumps with small rebases — the export
surface is the only ABI we commit to, and it stays small. No RM changes,
no PVM neutral bridge, no backend arbitration, no access-counter
translator. Delete the entire pvm-driver branch contents.

## Polaris-side components

### polaris.ko

New in v4. Replaces the v3 "kernel as lease bookkeeper" role.

- **VA-space registry**: ioctl `POLARIS_REGISTER_VASPACE(gpu, va_space_fd,
  managed_range)` from the shim. Kernel takes a refcount on the RM VA-space
  object, records the worker pid, stores the `(gpu, va_space_token, range)`
  tuple for hook lookups.
- **Block→worker map**: `block_id → {(va_space_token, va)}`. Refcounted.
  Drives spill teardown across all workers that share a block.
- **VRAM partition**: option A first (call `uvm_pmm_gpu_alloc` per chunk),
  option B (carve a contiguous reservation) only if PMM policy fights us.
- **CE channel**: one per GPU, allocated through UVM's channel manager via
  the GPL-exported entry points (commit 4). Used exclusively for
  spill/reload copies. polaris.ko does not reimplement channel state — it
  borrows UVM's, so RM channel conventions, pushbuffer layout, and tracker
  semantics stay UVM's problem across driver bumps.
- **Host pinned pool DMA mapping**: ioctl to ingest a polarisd-allocated
  pinned region, build sg-list, hand to CE.
- **Fault hook implementation**:
  1. `(va_space_token, addr)` → block_id via registry.
  2. Block state from kernel-side policy mirror.
  3. Resident in VRAM → install PTE in this worker's VA-space → `HANDLED`.
  4. Resident on host → allocate VRAM chunk (recursive eviction if pool
     full, per kernel-side policy mirror), CE-copy host→VRAM, install
     PTE, `HANDLED`.
  5. No block known for this VA → `NOT_MINE` (UVM falls through to
     managed/fatal as appropriate).
- **Policy mirror**: shadow of polarisd's per-block flags (pinned, movable,
  priority, COW group). Refreshed by ioctl from polarisd. Read-only on
  the fault path. No userspace round-trip in hot path.
- **Slow-path upcalls**: netlink or chardev read queue for events polarisd
  must arbitrate (new block on cold fault, COW split on write-to-shared).

### libpolaris-shim.so

New in v4. The transparency layer that v3 did not need.

- `LD_PRELOAD` (or installed as a CUDA driver-API hook layer).
- On `cuInit` / first CUDA call: create a fault-capable externally-owned
  VA-space, register with UVM via `UvmRegisterGpuVaSpace`, hand the token
  to polaris.ko.
- Intercept `cuMemAlloc`, `cudaMalloc`, and the PyTorch/llama.cpp
  allocator-backend entry points used by target workloads. Reserve VA
  inside the Polaris VA-space, return the pointer, **do not map**.
- Tell polaris.ko which VA range backs which logical Polaris allocation
  (so the fault hook can resolve `addr → block_id`).
- Intercept `cuMemFree` / `cudaFree` to deregister the range.
- Out-of-scope for v1: capturing `cuMemAllocAsync`, stream-ordered
  allocators, multi-context apps. Add only when a target workload needs
  them.

### polarisd

Keep most of v3's identity; change the control surface.

- **Keep**: block table, CPU pinned pool allocator, policy engine (FIFO/
  LRU/phase-aware), COW refcount logic, CLI, tracing.
- **Replace**: the lease API surface and the explicit residency scheduler.
  v4 does not require the framework to declare a required set before
  launch.
- **Add**:
  - Pinned host pool registration into polaris.ko at startup.
  - Policy-mirror push to polaris.ko on every block-state change.
  - Slow-path upcall handler (cold-fault arbitration, COW split).
  - Spill orchestration: choose victim → ioctl `POLARIS_SPILL(block_id)` →
    polaris.ko tears down PTEs in all workers, CE-copies VRAM→host,
    returns slot id → polarisd updates block table.
- **Remove from production path**:
  - The "require_resident → lease → launch → release" cycle. Keep code
    behind a feature flag for diagnostics only.

### polaris-runtime

- Keep CUDA VMM wrappers for diagnostic and v3-fallback paths.
- Add a minimal init/shutdown surface the shim can call.
- The framework-visible runtime is mostly *not* the v4 integration point —
  the shim is. polaris-runtime becomes a library the shim links, not a
  thing frameworks call directly.

### Integrations

- **llama.cpp**: rebuild around the shim. No source changes to llama.cpp
  beyond linker flags / `LD_PRELOAD` invocation. Remove
  `LLAMA_POLARIS_FAULT_WORKER` and the v2 fault-worker mode.
- **vLLM / SGLang**: same shim, no source changes. Confirm PyTorch caching
  allocator's backend entry point is interceptable; if not, add a
  `PYTORCH_CUDA_ALLOC_CONF` backend.

## Data Flows

### Reload on fault (hot path)

```
worker GPU dereferences ptr in Polaris range
  → MMU miss → replayable fault → UVM ISR
  → polaris_uvm_handle_gpu_fault(gpu, va_space_token, addr, READ)
  → polaris.ko: (token, addr) → block B
  → B.location = host:slot H
  → allocate VRAM chunk C (evict victim per mirror policy if pool full)
  → CE copy: H → C, wait semaphore
  → write PTE: VA → C in this worker's VA-space
  → return HANDLED
  → UVM replays the faulting work
```

No IPC to polarisd on this path. Budget: tens of microseconds.

### Spill (policy-driven)

```
polarisd: pick block B per LRU/phase
  → ioctl POLARIS_SPILL(B, prefer_slot=H)
  → polaris.ko: for each (worker, va) in B.mappings:
       unmap PTE in worker's VA-space, enqueue TLB invalidate
     CE copy: C → H, wait semaphore
     free chunk C back to pool
  → reply: B at host:H
  → polarisd updates block table
```

Workers are not informed. Next access on any worker will fault and reload.

### Cold fault (new logical block, slow path)

```
fault → polaris.ko: no block known for (token, addr)
  → if VA in shim-registered range that has no block yet:
       upcall to polarisd, block until reply
       polarisd: create block, push state, reply
       polaris.ko: allocate VRAM, optionally zero, install PTE, HANDLED
  → else: NOT_MINE → UVM fatal
```

Cold faults are rare and tolerate millisecond latency.

### COW split (slow path)

```
fault with access=WRITE on block B with refcount>1
  → polaris.ko sees mirror flag B.cow=true, refcount>1
  → upcall to polarisd
  → polarisd: allocate new block B', record split, push state
  → polaris.ko: CE copy B→B', install PTE for B' in this worker only,
                leave other workers mapped to B, return HANDLED
```

## Correctness Invariants

1. PTE installs happen only from polaris.ko, only against VA-spaces
   registered by a shim instance whose pid still exists.
2. CE submissions and the resulting PTE update for a single fault are
   ordered: copy completion is observed before PTE write.
3. Spill tears down PTEs in every worker holding the block before the CE
   copy starts, and TLB invalidation completes before VRAM is freed.
4. Policy mirror is read-only on the fault path; updates from polarisd
   are applied under a sequence number the fault path snapshots.
5. The fault hook never blocks on polarisd except on cold-fault and COW
   slow paths, and those paths must be reachable only when the fault
   would otherwise be fatal anyway.
6. Worker process death drops the VA-space refcount; polaris.ko reaps
   all PTE state for that worker before allowing block table reuse.

## Non-Goals

- Kernel-side shared page tables across worker VA-spaces. Each worker
  keeps a private page table. Sharing is at the *physical block* level,
  not the PTE level.
- A neutral RM bridge ("PVM"). polaris.ko talks to UVM and RM through
  existing exported APIs and the one new hook.
- Replacing UVM. UVM continues to own managed memory, ATS, fault
  arbitration, and the ISR. Polaris is a tenant.
- Userspace residency leases as a required framework API.
- Access-counter-driven residency policy in v1. Add only when measurement
  shows stock policy is the bottleneck.

## Milestones

### M1: Driver hook lands

- UVM patches 1 and 2 above, against `polaris-610.43.02`.
- Stub polaris.ko registers the hook, returns `NOT_MINE` for everything.
- CI: build the driver, load, register, unregister, unload cleanly.

### M2: Fault-capable VA-space end-to-end

- Shim creates a fault-capable VA-space, registers with UVM, registers
  with polaris.ko.
- polaris.ko hot path returns `HANDLED` for one statically-pre-mapped
  block in a microbenchmark process.
- Verifies the fault → hook → PTE-install → replay path on real
  hardware.

### M3: CE-driven spill/reload

- UVM commit 4 lands the channel-manager / push GPL exports.
- polaris.ko allocates a CE channel through UVM's channel manager and owns
  the VRAM partition (option A, via PMM).
- Spill ioctl and reload-on-fault paths implemented.
- Single-worker microbenchmark: allocate > VRAM partition, dereference
  pattern that forces spill/reload, validate data integrity.

### M4: Multi-worker block sharing

- Refcounted `block → {worker, va}` map.
- Spill tears down across all workers.
- COW split slow path (read-only beam-search workload).

### M5: llama.cpp via shim

- Shim intercepts llama.cpp's CUDA allocator.
- Run llama.cpp end-to-end against shim+polaris.ko+polarisd, no source
  changes to llama.cpp.
- Compare throughput vs v3-lease path and vs vLLM/SGLang baselines.

### M6: Hardening

- Worker crash reaping.
- Module unload with workers still attached (refcount drains).
- Tracing for: faults serviced, faults rejected, spills, reloads, CE
  occupancy, policy-mirror sequence drift.
- Stress: dynamic KV growth, fragmentation pressure, OOM behavior.

### M7: PyTorch / vLLM integration

- Confirm or add PyTorch allocator backend interception.
- Run vLLM under shim+polaris.ko+polarisd.

## Out-of-Scope For v4

- eBPF observability beyond v3's existing hooks.
- Migration across GPUs (single-GPU first).
- ATS interactions.
- Multi-CE-channel parallelism within polaris.ko.
- Production access-counter feedback into policy.

These re-enter scope only after M5 demonstrates the fault-driven path
beats the v3 lease path on a real workload.

## Migration From v3

- v3 explicit-lease API: keep behind `--legacy-lease` for diagnostics.
- v3 kernel block-state machine: keep, polaris.ko v4 builds on it.
- v3 offload/reload mechanics in polaris-runtime: keep for diagnostic
  shell. Production path moves to polaris.ko's CE channel.
- v3 llama.cpp fault-worker mode: delete.
- v3 README claims about production replayable-fault paging: rewrite
  once M5 lands.

## Open Questions

1. Can the shim reliably intercept PyTorch's caching allocator without
   a custom backend? If not, M7 needs a `PYTORCH_CUDA_ALLOC_CONF`
   backend ship.
2. PMM partition (option B) vs per-chunk PMM calls (option A): need a
   measurement on M3 to decide.
3. Cold-fault upcall mechanism: netlink, chardev, or io_uring-style
   submission queue? Pick when M2 lands.
4. Multi-GPU: one polaris.ko CE channel per GPU is mechanical; the
   polarisd block table needs explicit GPU affinity. Decide before M4.
5. UVM rebases: how often do the call sites in
   `uvm_gpu_replayable_faults.c` (`service_fault_batch_dispatch`,
   `uvm_parent_gpu_fault_entry_to_va_space`) and the channel-manager
   export surface shift across driver versions? Track across 535 / 545 /
   550 / 555 / 560.
