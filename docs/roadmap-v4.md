# POLARIS Roadmap v4

Date: 2026-06-10 (revised 2026-06-12 after polaris-v4 driver branch landed)

This roadmap supersedes v3. v3 adopted Method A (explicit-lease CUDA VMM
paging) because raw CUDA VMM VA holes produce fatal MMU faults, not
serviceable UVM replayable faults. v4 keeps that diagnosis but takes a
different exit: instead of pushing residency decisions up into the framework
via leases, host workers in a fault-capable VA-space owned by UVM and let
faults flow through a hook UVM calls before its own managed-fault servicing.

The worker is oblivious. polarisd decides. polaris.ko executes.

## Direction

> Transparent kernel-directed KV paging on a UVM-registered fault-capable
> VA-space. A small LD_PRELOAD shim hides VA-space setup and registers each
> KV range as a UVM external range. polaris.ko owns a VRAM partition (RM
> allocations whose handles UVM can map) and the policy mirror that drives
> fault servicing; PTE installs go through UVM's existing external-range map
> path via a new GPL-exported bridge. polarisd owns the block table and
> policy. Workers run with a CUDA-driver-visible managed/external pointer
> and require no source changes for the basic `cudaMalloc` / `cudaMallocManaged`
> + kernel access
> + free path.

The lease API from v3 is not part of the production path. v3 lease code may
remain as a fallback for diagnostics and for non-fault-capable contexts but
should not be the integration target.

## What changed in this revision

This revision reconciles the original 2026-06-10 plan with the three commits
that actually landed on the `polaris-v4` driver branch
(`eaed5057`, `30e9c9a5`, `00debab0`). The revisions:

- The driver-side hook is the **only** kernel ABI Polaris commits to. PMM and
  channel-manager exports (the original commits 3 and 4) are dropped from the
  plan: PMM stays internal to UVM, and PTE installs go through a single new
  bridge `uvm_polaris_map_external_allocation()` that reuses UVM's external-
  range path instead of giving polaris.ko its own CE channel.
- polaris.ko's VRAM does not come from carving a PMA partition. polaris.ko
  allocates RM memory objects (one per block, page-size-aligned) and hands
  `(h_client, h_memory)` to UVM through the bridge. UVM does the actual PTE
  programming and TLB invalidate.
- The hook contract is single-fault (`gpu_id, rm_client_token,
  va_space_token, gpu_va_space_ptr, fault_address, access_type`),
  RCU-published, module-owner refcounted. UVM dispatches under
  `service_lock` + va_space read lock already held; the hook may sleep but
  should not block userspace IPC.
- `va_space_token` is the **user RM VA-space handle** that the shim sees
  through `UvmRegisterGpuVaSpace`, not the duped handle UVM keeps.
  `rm_client_token` is the user RM client handle from the same registration.
  The shim must pass both handles to polaris.ko because RM object handles are
  scoped to an RM client.
- Application transparency is downgraded from "no source changes ever" to:
  llama.cpp allocator interception is the first target, but transparent
  execution still requires either avoiding host copy/fill APIs on intercepted
  buffers or implementing a real host/device copy path for Polaris external
  VA. PyTorch / vLLM still require an allocator backend in M7 because their
  caching allocators bypass driver-API interception.

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
| shim | per-process VA-space creation, allocator interception, UVM external-range registration for KV VA, ioctl registration with polaris.ko | data movement, policy, block state, PTE programming |
| polaris.ko | block→worker map, policy mirror, fault hook implementation, RM allocation pool for KV chunks, host pinned pool DMA mapping, scheduling slow-path upcalls to polarisd | direct PTE writes (delegated to UVM through the bridge), the fault buffer, CE channel state |
| polarisd | block table, eviction/spill policy, host pinned pool slot assignment, COW arbitration, mirror push to polaris.ko | PTE writes, fault servicing, hot-path decisions |

## Driver-side changes (open-gpu-kernel-modules)

Status: landed on branch `polaris-v4` (commits `eaed5057`, `30e9c9a5`,
`00debab0` on top of `polaris-610.43.02`). Total UVM diff is ~280 LOC.
RM is untouched. The `pvm-driver` branch is frozen as a research artifact
and is not part of the v4 production path.

The three commits, as actually landed:

1. `uvm: export polaris fault hook registration` (`eaed5057`)
   - Adds `kernel-open/nvidia-uvm/uvm_polaris.{c,h}`: a single-slot
     RCU-protected `uvm_polaris_ops` pointer with `EXPORT_SYMBOL_GPL`'d
     `uvm_polaris_register_hook()` / `uvm_polaris_unregister_hook()`.
   - Registration uses `cmpxchg` to enforce a single owner;
     unregister waits via `synchronize_rcu()` so in-flight dispatchers
     finish before the caller is allowed to free its storage.
   - `uvm_polaris_dispatch_fault()` resolves `(gpu_id, rm_client_token,
     va_space_token, gpu_va_space_ptr, fault_address, access_type)` from a
     `uvm_gpu_va_space_t *` and calls the hook under
     `try_module_get(ops->owner)` so polaris.ko cannot be rmmod'd while
     a fault is mid-flight.
   - `va_space_token` is the **user RM VA-space handle**
     (`gpu_va_space->user_rm_va_space`) and `rm_client_token` is the user RM
     client handle (`gpu_va_space->user_rm_client`). Both are populated from
     `UvmRegisterGpuVaSpace`; polaris.ko uses the pair as the stable
     per-worker key in its block→worker map.

2. `uvm: call polaris hook in replayable fault service path` (`30e9c9a5`)
   - Inserts a call to `uvm_polaris_dispatch_fault()` at the top of
     `service_fault_batch_dispatch` in
     `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`, immediately
     after the per-fault VA-space is resolved by `service_fault_batch()`
     and before any managed-VA or ATS lookup runs.
   - `HANDLED`: set `current_entry->filtered = true`, `*block_faults = 1`,
     return `NV_OK`. UVM's outer loop then advances past this entry and
     issues its normal batch replay; polaris.ko has already serviced the
     fault end-to-end (RM allocation, host→VRAM copy if needed, PTE install
     through the bridge in commit 3 below).
   - `ERROR`: route through the existing `service_fault_batch_fatal_notify`
     path with `NV_ERR_OPERATING_SYSTEM` and `UVM_FAULT_CANCEL_VA_MODE_ALL`.
   - `NOT_MINE`: fall through unchanged to the stock managed/ATS dispatch.
   - The dispatcher uses RCU internally, so this adds a
     `rcu_read_lock`/`rcu_read_unlock` pair per fault on the hot path.

3. `uvm: add polaris PTE bridge and fault dispatch test` (`00debab0`)
   - Adds `uvm_polaris_map_external_allocation()` (in
     `uvm_map_external.c`, GPL-exported): polaris.ko passes the
     `gpu_va_space_ptr` it received in `handle_gpu_fault`, the VA range
     `(base, length, offset)`, and the RM allocation handle
     `(rm_control_fd, h_client, h_memory)`; UVM looks up the
     `uvm_va_range_external_t` covering the range and calls
     `uvm_map_external_allocation_on_gpu()` to install the PTEs. PTE
     programming and TLB invalidate stay UVM's responsibility.
   - This replaces the original v4 commit-4 plan (export the channel
     manager and PMM so polaris.ko could push CE work itself). Reusing
     UVM's existing external-range path is significantly less surface
     area and avoids re-implementing channel/tracker bookkeeping.
   - Adds `UVM_TEST_POLARIS_DISPATCH_FAULT` ioctl so the hook can be
     end-to-end tested by synthesizing a fault from userspace, without
     real GPU hardware-faulting traffic. Enabled only with builtin tests.

What this means for polaris.ko's VRAM model: polaris.ko does **not** carve
a PMA partition or own a CE channel. It allocates per-block RM memory
objects (page-size-aligned device memory), records `(h_client, h_memory)`
in the block table, and when a fault arrives calls the bridge to map the
right block into the faulting worker's VA range. Spill / reload is then
"unmap the external range across all workers holding it" + "free the RM
allocation" + "the next access re-faults and gets a fresh allocation".

What is **not** done in the driver tree and stays deferred:

- PMM and channel-manager GPL exports — out of scope after the bridge
  approach landed.
- A real RM ISR/bottom-half handoff into polaris-owned fault servicing —
  not needed for v4; UVM keeps interrupt ownership.
- The full PVM neutral RM bridge that lived on `pvm-driver` — explicitly
  abandoned. The branch stays for reference only.

## Polaris-side components

### polaris.ko

New in v4. Replaces the v3 "kernel as lease bookkeeper" role.

- **VA-space registry**: ioctl
  `POLARIS_REGISTER_VASPACE(gpu, rm_client_token, user_rm_va_space,
  managed_range)` from the shim. The `rm_client_token` and
  `user_rm_va_space` fields are the same RM handles the shim passed to
  `UvmRegisterGpuVaSpace`; they match what UVM passes on the fault hot path.
  polaris.ko records `(gpu, rm_client_token, va_space_token, range,
  worker pid, owning gpu_va_space_ptr-on-first-fault)` for hook lookups.
- **Block→worker map**: `block_id → {(rm_client_token, va_space_token, va)}`.
  Refcounted.
  Drives spill teardown across all workers that share a block.
- **VRAM block pool**: per-block RM allocations (one per logical block,
  page-size-aligned device memory) obtained through RM's standard
  allocation path. polaris.ko keeps the `(h_client, h_memory)` tuple for
  each resident block and reuses it across re-mappings. **No PMA carve-out,
  no CE channel ownership.**
- **PTE install through UVM bridge**: on fault, after picking which block
  to materialize, polaris.ko calls
  `uvm_polaris_map_external_allocation(gpu_va_space_ptr, base, length,
  offset, rm_control_fd, h_client, h_memory)`. UVM does the actual PTE
  programming, TLB invalidate, and tracker bookkeeping on its existing
  external-range path.
- **Host pinned pool DMA mapping**: ioctl to ingest a polarisd-allocated
  pinned region, build sg-list, and hand to UVM-side copy infrastructure
  for spill/reload. (Data movement on cold→hot transitions still goes
  through the same UVM-owned channel polaris.ko piggybacks on via the
  external-range path; for spill, polaris.ko unmaps and lets the next
  fault re-materialize with fresh content from host.)
- **Fault hook implementation**:
  1. `(rm_client_token, va_space_token, fault_address)` → block_id via registry.
  2. Block state from kernel-side policy mirror.
  3. Block resident in VRAM (RM alloc still held) → call
     `uvm_polaris_map_external_allocation` to (re)install PTE → `HANDLED`.
  4. Block offloaded to host pinned pool → allocate fresh RM device alloc,
     copy host → device through whatever copy path is wired (initial slice
     can use `cuMemcpyHtoDAsync` from a polaris-owned worker thread that
     has the right context; later optimizations may push directly), call
     bridge, `HANDLED`.
  5. No block known for this VA → `NOT_MINE`.
  6. Hard failure (OOM, RM alloc fails, bridge returns error) → `ERROR`
     so UVM cancels the fault through `service_fault_batch_fatal_notify`.
- **Policy mirror**: shadow of polarisd's per-block flags (pinned, movable,
  priority, COW group). Refreshed by ioctl from polarisd. Read-only on
  the fault path. No userspace round-trip in hot path.
- **Slow-path upcalls**: netlink or chardev read queue for events polarisd
  must arbitrate (cold fault on a VA range that has no logical block yet,
  COW split on write-to-shared).
- **Hook lifecycle**: at module init, publish ops vector via
  `uvm_polaris_register_hook()`; at exit, call
  `uvm_polaris_unregister_hook()` which waits via `synchronize_rcu()` so no
  in-flight dispatcher is still inside polaris.ko before the module unloads.
  UVM additionally pins the module via `try_module_get(ops->owner)` for
  each dispatch.

### libpolaris-shim.so

New in v4. The transparency layer that v3 did not need.

- `LD_PRELOAD` (or installed as a CUDA driver-API hook layer).
- On `cuInit` / first CUDA call: create a fault-capable externally-owned
  GPU VA-space (via the existing RM path that already sets
  `NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING |
  IS_EXTERNALLY_OWNED`), register it with UVM via `UvmRegisterGpuVaSpace`,
  hand the **same RM client and RM VA-space handles** to polaris.ko via
  `POLARIS_REGISTER_VASPACE`. These handles are what UVM stores into
  `gpu_va_space->user_rm_client` / `gpu_va_space->user_rm_va_space` and
  replays back on the hook, so the shim and polaris.ko must agree on them
  byte-for-byte.
- Intercept `cuMemAlloc`, `cudaMalloc`, and the PyTorch/llama.cpp
  allocator-backend entry points used by target workloads. For each
  allocation:
  1. Reserve VA inside the Polaris VA-space (return the pointer).
  2. Register the VA range with UVM as an **external range**
     (`UvmCreateExternalRange` or equivalent) — required by the bridge,
     because `uvm_polaris_map_external_allocation` looks up an existing
     `uvm_va_range_external_t` covering the fault address.
  3. Tell polaris.ko which VA range backs which logical Polaris allocation
     via `POLARIS_REGISTER_RANGE` (so the fault hook can resolve
     `rm_client_token + va_space_token + addr → block_id`).
  4. Do **not** map any pages. Faults populate them on demand.
- Intercept `cuMemFree` / `cudaFree` to unregister the range, drop the
  external range, and let polaris.ko reclaim block table state.
- Application transparency limits (see also "Application transparency"
  below):
  - The basic `cudaMalloc` + kernel-deref + `cudaMemcpy` + `cudaFree` path
    is transparent.
  - CUDA IPC (`cuIpcGetMemHandle`) and CUDA Graph capture of KV
    accesses are not transparent; KV ranges in those APIs are out of
    scope for v1 and must either fall back to the v3 lease path or be
    documented as unsupported.
  - PyTorch caching allocator ownership of KV memory remains out of scope
    for M5. The shim now covers runtime stream-ordered symbols for smoke
    tests, but M7 still needs an explicit allocator backend to make
    framework-owned KV allocations intentional.

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
  → MMU miss → replayable fault → UVM ISR / bottom half
  → service_fault_batch() resolves (gpu_va_space, fault_address)
  → service_fault_batch_dispatch() calls uvm_polaris_dispatch_fault()
  → polaris.ko's handle_gpu_fault(gpu_id, rm_client_token,
                                  va_space_token, gpu_va_space_ptr,
                                  addr, access)
  → (rm_client_token, va_space_token, addr) → block B
  → B's state = host-resident, slot H
  → ensure a device-side RM allocation D for B (allocate if missing)
  → copy H → D (initial slice uses async memcpy; later replaced with
    a UVM-borrowed copy path)
  → call uvm_polaris_map_external_allocation(gpu_va_space_ptr,
        B's VA base, B's length, 0,
        D.rm_control_fd, D.h_client, D.h_memory)
  → UVM programs the PTEs, batches TLB invalidate, returns
  → polaris.ko returns HANDLED
  → UVM marks current_entry filtered, advances *block_faults
  → outer loop replays the fault batch → GPU re-issues the access
```

No IPC to polarisd on this path. Hot-path budget: tens to a few hundred
microseconds; the bridge call itself walks UVM's external-range map and
PTE batch, which is bounded by block size, not VA-space size.

### Spill (policy-driven)

```
polarisd: pick block B per LRU/phase
  → ioctl POLARIS_SPILL(B, prefer_slot=H)
  → polaris.ko: for each (worker, va) in B.mappings:
       unmap PTE in worker's VA-space (via the same external-range path)
       enqueue TLB invalidate
     copy D → H (sync wait on the copy)
     release D's RM allocation back to the pool
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
       polaris.ko: allocate RM device memory, optionally zero,
                   call bridge to install PTE, HANDLED
  → else: NOT_MINE → UVM fatal
```

Cold faults are rare and tolerate millisecond latency.

### COW split (slow path)

```
fault with access=WRITE on block B with refcount>1
  → polaris.ko sees mirror flag B.cow=true, refcount>1
  → upcall to polarisd
  → polarisd: allocate new block B', record split, push state
  → polaris.ko: allocate new RM device alloc D' for B',
                copy D → D' (async),
                call bridge to install PTE for B' in this worker only,
                leave other workers mapped to D, return HANDLED
```

## Correctness Invariants

1. PTE installs happen only through `uvm_polaris_map_external_allocation`,
   only against `gpu_va_space_ptr` values polaris.ko received in
   `handle_gpu_fault` from a still-live worker, only against VA ranges
   the shim registered as UVM external ranges.
2. The RM allocation D backing a block must remain alive for the full
   span from "bridge call begins" through "UVM completes its TLB invalidate
   and the worker has replayed the faulting access". polaris.ko refcounts
   D against active mappings and pending hook returns.
3. Spill tears down PTEs in every worker holding the block before the
   data is reused for a different block. The unmap and the
   reallocation/reload of D for a different block must not overlap.
4. Policy mirror is read-only on the fault path; updates from polarisd
   are applied under a sequence number the fault path snapshots.
5. The fault hook never blocks on polarisd except on cold-fault and COW
   slow paths, and those paths must be reachable only when the fault
   would otherwise be fatal anyway.
6. Worker process death drops the VA-space refcount; polaris.ko reaps
   all block→worker mapping entries for that worker before allowing
   block-table reuse. The `try_module_get(ops->owner)` UVM holds across
   each dispatch prevents polaris.ko from unloading while a hook call is
   in flight; `uvm_polaris_unregister_hook()` waits via
   `synchronize_rcu()` before returning to the caller's module exit.

## Application transparency

This section replaces the original v4 claim of unconditional zero source
changes. The honest position:

**Transparent through shim, no source changes**:
- `cudaMalloc` / `cuMemAlloc_v2` / `cudaFree` paths
- Direct kernel dereference (`kernel<<<...>>>(p)` then `p[i]`)
- Pointer arithmetic and sub-range arguments

**Requires Polaris allocations to be registered as external ranges**:
- `cuPointerGetAttribute` queries for `CU_POINTER_ATTRIBUTE_MEMORY_TYPE`
  and `CU_POINTER_ATTRIBUTE_DEVICE_POINTER` — driver must recognize the
  VA. The shim's external-range registration is what makes this work.

**Not transparent in v1**:
- `cudaMemcpy` / `cudaMemset` involving Polaris pointers. The current shim
  guards these calls with `cudaErrorNotSupported`; llama.cpp's first
  intercepted CUDA model-buffer allocation currently hits this path during
  tensor upload before any replayable GPU fault can reach Polaris.
- `cuIpcGetMemHandle` and IPC-based sharing.
- CUDA Graph capture of accesses that depend on faulting in KV pages.
- PyTorch caching allocator ownership of KV memory — even though the shim can
  interpose runtime allocation symbols such as `cudaMallocAsync`, M7 still
  needs an allocator backend to force PyTorch KV allocations through Polaris
  deliberately.

The course / paper claim should be: **the UVM hook/PTE bridge and shim
allocator registration path are wired; fully transparent llama.cpp execution
still needs the host-copy gap closed or a KV-only allocation selection that
does not use CUDA host copy/fill APIs**. vLLM and PyTorch require an explicit
backend.

## Non-Goals

- Kernel-side shared page tables across worker VA-spaces. Each worker
  keeps a private page table. Sharing is at the *RM allocation* level,
  not the PTE level.
- A neutral RM bridge ("PVM"). polaris.ko talks to UVM through
  `uvm_polaris_*` and to RM only via standard allocation APIs.
- Replacing UVM. UVM continues to own managed memory, ATS, fault
  arbitration, the ISR, and external-range PTE programming on
  Polaris's behalf.
- Userspace residency leases as a required framework API.
- Access-counter-driven residency policy in v1. Add only when measurement
  shows stock policy is the bottleneck.
- PMA partition carving in polaris.ko.
- A polaris-owned CE channel.

## Milestones

### M1: Driver hook lands ✅ (driver tree)

Done on `polaris-v4`: commits `eaed5057` + `30e9c9a5` + `00debab0`.
The UVM side of M1/M2/M3 is materially complete:

- `uvm_polaris_register_hook` / `uvm_polaris_unregister_hook` exported.
- Hook dispatched per fault in `service_fault_batch_dispatch` with
  HANDLED / ERROR / NOT_MINE semantics.
- `uvm_polaris_map_external_allocation` bridge available so polaris.ko
  does not need its own CE channel.
- `UVM_TEST_POLARIS_DISPATCH_FAULT` ioctl for synthetic-fault e2e tests
  without GPU hardware faulting.

Polaris-side status:

- polaris.ko publishes the ops vector at init and unregisters it at exit.
- The M2 static-block fault path is wired: registered VA-spaces and static
  RM allocation blocks are mirrored into a small fast lookup table, and a
  matching synthetic UVM fault calls `uvm_polaris_map_external_allocation`
  before returning `HANDLED`.
- The M3 unmap/refault diagnostic path is wired: UVM exports
  `uvm_polaris_unmap_external_allocation`, polaris.ko exposes
  `POLARIS_UNMAP_STATIC_BLOCK` plus the logical-block
  `POLARIS_REGISTER_BLOCK_MAPPING` / `POLARIS_UNMAP_BLOCK_MAPPINGS`
  teardown pair, and the M2 harness validates fault-map → unmap →
  refault-map on real hardware through both the static and block-mapping
  paths.
- VA-space lookup is keyed by `(gpu_id, rm_client_token, va_space_token)`.
  `va_space_token` is an RM object handle and is only unique inside the RM
  client. The UVM hook now dispatches both user RM handles, and the builtin
  test ioctl reports both so concurrent harnesses do not cross-match.
- `make kernel` now consumes the patched UVM `Module.symvers` from the
  `third_party/open-gpu-kernel-modules` submodule by default.

Still to do on the Polaris side for M1/M2:

- CI: build the driver, load polaris.ko, register / unregister, unload
  cleanly. Use the test ioctl to confirm dispatch returns expected codes
  with and without a registered hook.

### M2: Fault-capable VA-space end-to-end ✅

- M2 harness creates a fault-capable VA-space, registers it with UVM,
  probes the exact `(gpu_id, rm_client_token, user_rm_va_space)` key UVM
  will dispatch, and registers that key with polaris.ko.
- M2 harness registers one external range and one static RM allocation block
  so the bridge can resolve it.
- polaris.ko hot path returns `HANDLED` for that statically-pre-allocated
  block by calling `uvm_polaris_map_external_allocation` directly.
- Hardware validation passed on an RTX 5070 Ti / driver 610.43.02 with
  patched `nvidia-uvm.ko` loaded and builtin UVM tests enabled:
  `tests/m2/m2_static_block_setup`, `--dispatch-fault`, and the M3
  `--unmap-refault`, `--block-unmap-refault`, and
  `--logical-backed-refault` diagnostics all pass.
- Production shim still needs in-shim RM/UVM VA-space creation and external-
  range registration; current shim only bootstraps a harness-created
  VA-space through environment variables.

### M3: External-allocation-driven spill/reload

- polaris.ko owns a pool of RM device allocations (one per logical block,
  page-size-aligned). `nvidia-uvm.ko`'s commit-3/4 PMM/CE exports are
  **not** needed; the bridge replaces them.
- First slice complete: `uvm_polaris_unmap_external_allocation` tears down an
  external mapping for the UVM GPU VA-space observed on a previous fault, and
  `POLARIS_UNMAP_STATIC_BLOCK` proves the same block can refault and remap.
- Second slice complete: polaris.ko has a logical block-mapping registry
  keyed by `block_id`; the fault hook records the observed
  `gpu_va_space_ptr` for registered block mappings; and
  `POLARIS_UNMAP_BLOCK_MAPPINGS` unmaps all observed UVM mappings for that
  block. The M2 harness's `--block-unmap-refault` mode creates a real
  session/block, registers its mapping, fault-maps it, unmaps by `block_id`,
  verifies `unmapped_count == 1`, and refaults successfully.
- Logical-backed bridge slice complete: `POLARIS_REGISTER_BLOCK_BACKING`
  attaches an RM allocation tuple
  `(rm_control_fd, h_client, h_memory, length, offset)` to an existing
  logical block. The UVM hook now first tries the static diagnostic fast path,
  then resolves a registered logical block mapping with attached RM backing
  and maps it through `uvm_polaris_map_external_allocation`. The M2 harness's
  `--logical-backed-refault` mode skips `POLARIS_REGISTER_STATIC_BLOCK`,
  registers only the logical mapping plus block backing, fault-maps it, unmaps
  by `block_id`, verifies `unmapped_count == 1`, and refaults successfully.
  This removes the static-block table as a requirement for the live UVM bridge
  diagnostic, but it still uses harness/shim-created RM backing rather than
  daemon-created spill/reload backing.
- Real CUDA-kernel compatibility slice wired: exact UVM hook lookup remains
  keyed by `(gpu_id, rm_client_token, va_space_token)`, but when a real
  runtime CUDA kernel faults on a Polaris VA from CUDA's own registered
  VA-space, polaris.ko can service it only if exactly one logical block mapping
  with registered RM backing covers that `(gpu_id, fault_address)`. The hook
  creates the missing external range in the observed faulting UVM VA-space,
  retries the bridge internally, records the observed `gpu_va_space_ptr`, and
  returns `HANDLED`. Ambiguous address matches still return `NOT_MINE`.
- Third slice complete: `POLARIS_SPILL_BLOCK` is in the ABI. It first tears
  down observed UVM mappings with the same block-level unmap helper, then
  queues the existing `OFFLOAD` decision for resident logical blocks so
  polarisd performs device → host copy and physical-handle release through
  the normal completion path. The M2 harness validates the ioctl's unresident
  block rejection (`--spill-validation`), but the static RM harness cannot
  positively execute copy/release because it does not create a daemon-owned
  CUDA VMM resident block.
- Positive daemon/runtime spill test wired: the ignored root/GPU
  `polaris-runtime` fault-smoke creates a resident logical block through the
  runtime decision worker, writes a CUDA-visible pattern, calls
  `POLARIS_SPILL_BLOCK`, verifies the OFFLOAD completion moves the block to
  `CpuOffloaded`, reloads the same VA through overwrite reserve, and checks
  the bytes survive the device → host → device cycle. A separate long-running
  `polarisd` process soak remains useful, but the production decision path is
  no longer covered only by fake executor state-machine tests.
- Kernel state-machine coverage added: `libpolaris` has an ignored
  root-only `kernel_spill_state` test that drives
  `ALLOC → POLARIS_SPILL_BLOCK/OFFLOAD → BLOCK_RESERVE/RELOAD` through
  `GET_DECISION` / `COMPLETE_OPERATION` without invoking CUDA. This catches
  the M3 control-plane contract even when the CUDA/UVM channel path is not
  safe to run. It is build-covered by `cargo test`; execution requires a
  freshly loaded `polaris.ko`.
- Reload state-machine fix: overwriting an existing single-ref block now
  triggers synchronous fault resolution when the block is not resident,
  so `CpuOffloaded` blocks queue `RELOAD` instead of returning an unmapped
  VA.
- Reload on fault: re-allocate RM device memory, copy host → device, call
  bridge, return HANDLED.
- Single-worker microbenchmark: register external range larger than the
  block pool, drive a deref pattern that forces spill/reload, validate
  data integrity.

### M4: Multi-worker block sharing + COW

- Refcounted `block → {(rm_client_token, va_space_token, va)}` map.
- Spill tears down across all workers.
- COW split slow path (read-only beam-search workload): polaris.ko sees
  write fault on refcount>1 block, upcalls polarisd, allocates fresh
  device alloc for the writer, calls bridge for the new mapping, leaves
  other workers mapped to the original.
- First control-plane slice wired: `SESSION_BRANCH` refcounts parent
  blocks, overwrite reserve on a shared child block creates a private block
  and queues `COW_BREAK`, and the COW decision now carries the writer's
  destination VA instead of `0` so userspace maps the private copy at the
  faulting session VA. The child session's inherited block reference is
  replaced with the private block during the split, so destroying the child
  cannot later drop the parent's still-live block. `libpolaris/tests/kernel_spill_state.rs`
  includes an ignored root-only fake-executor test for this branch/overwrite
  path and validates the child block table after the split.
- Multi-worker cleanup slice wired: the same ignored test file registers two
  v4 worker VA-spaces for one logical block, registers one block mapping per
  worker, then verifies explicit `POLARIS_UNREGISTER_VASPACE` and fd-close
  cleanup remove only the matching worker's mapping before all v4 state
  returns to zero. This covers the control-plane ownership rules for
  block-to-worker mappings without invoking CUDA/UVM channel registration.
- Block-lifetime cleanup slice wired: logical block mappings are reaped when
  the logical block is removed by `BLOCK_RELEASE`, direct `SESSION_DESTROY`,
  or successful daemon `FREE` completion. Ignored non-CUDA tests cover the
  release and deferred-block session-destroy paths; execution requires a
  freshly loaded `polaris.ko` because the currently pinned module may not
  contain this cleanup fix.
- COW-shared release cleanup wired: `BLOCK_RELEASE` now resolves blocks
  through the session's accessible block-id list, so a child can release an
  inherited shared block directly and decrement the shared refcount without
  waiting for session teardown. When a shared release only decrements the
  refcount, mappings in the releasing session's VA interval are reaped while
  mappings for surviving sharers remain registered.

### M5: llama.cpp via shim

- Shim intercepts llama.cpp's CUDA allocator for KV tensors.
- First allocator-interposition slice wired: `libpolaris-shim.so` now exports
  `cuMemAlloc`, `cuMemAlloc_v2`, `cudaMalloc`, `cudaMallocManaged`,
  `cuMemFree`, `cuMemFree_v2`, and `cudaFree`. In opt-in harness mode
  (`POLARIS_SHIM_MANAGE_ALLOCATIONS=1`)
  the shim creates a Polaris session over the provided managed VA window,
  reserves deferred logical blocks, registers block-to-worker mappings, and
  returns Polaris VAs without mapping pages. This is build-covered by
  `make -C libpolaris-shim all tests`; execution against live faults still
  requires a freshly loaded module and a real UVM external range.
- External-range slice wired for bootstrapper-owned UVM VA-spaces:
  `POLARIS_SHIM_CREATE_EXTERNAL_RANGES=1` plus `POLARIS_SHIM_UVM_FD=<fd>`
  makes each shim-managed allocation call `UVM_CREATE_EXTERNAL_RANGE` on the
  same initialized UVM VA-space fd used for `UVM_REGISTER_GPU_VASPACE`, and
  explicit frees call `UVM_FREE` for that range. This moves allocator-span
  external-range lifecycle into the shim while full in-shim RM/UVM VA-space
  creation is still pending.
- In-shim RM/UVM bootstrap slice wired behind
  `POLARIS_SHIM_BOOTSTRAP_RM_UVM=1`: the shim allocates an RM root client,
  device, subdevice, and a fault-capable externally-owned `FERMI_VASPACE_A`,
  initializes `/dev/nvidia-uvm`, calls `UVM_REGISTER_GPU` and
  `UVM_REGISTER_GPU_VASPACE`, registers those same RM handles with
  polaris.ko, and feeds the registered UVM fd into the external-range helper.
  This removes the M2 harness requirement for VA-space and range creation,
  but still needs live llama.cpp validation and production allocator-window
  policy.
- Shim allocator reuse wired: freed logical allocation spans are coalesced and
  reused for later `cuMemAlloc` / `cudaMalloc` calls instead of permanently
  advancing a one-way token cursor. The managed window is still fixed-size,
  but ordinary alloc/free churn no longer exhausts it after successful frees.
- Pointer-query compatibility slice wired: `cuMemGetAddressRange` /
  `cuMemGetAddressRange_v2`, `cuPointerGetAttribute`,
  `cuPointerGetAttributes`, and runtime `cudaPointerGetAttributes` now answer
  metadata for shim-managed Polaris pointers, including allocator-probe
  attributes (`HOST_POINTER`, `IS_MANAGED`, `DEVICE_ORDINAL`, and
  `MEMPOOL_HANDLE`), so allocator/runtime code that probes CUDA pointer
  metadata can proceed before a real GPU dereference.
- Runtime allocator smoke wired: the non-fault `managed_alloc` harness can
  drive the shim through `cudaMalloc` / `cudaFree` with
  `POLARIS_SHIM_TEST_RUNTIME_ALLOC=1`, covering the allocator entry points
  expected from llama.cpp before attempting a real GPU dereference.
- Runtime managed-allocation smoke wired: the shim now exports
  `cudaMallocManaged` and routes in-policy calls through the same Polaris
  allocator table, with fallback to real CUDA for out-of-policy or non-strict
  failures. The harness covers this with `POLARIS_SHIM_TEST_MANAGED_ALLOC=1`,
  matching llama.cpp's `GGML_CUDA_ENABLE_UNIFIED_MEMORY` allocation branch.
  Polaris-serviced pointers continue to report as device memory rather than
  CUDA managed memory because they are external-range VAs owned by Polaris.
- Stream-ordered allocation smoke wired: the shim now exports runtime
  `cudaMallocAsync` / `cudaFreeAsync` and driver
  `cuMemAllocAsync` / `cuMemAllocAsync_v2` /
  `cuMemFreeAsync` / `cuMemFreeAsync_v2`, routes successful allocations
  through the same fixed managed-window allocator, and the harness can
  exercise these surfaces with `POLARIS_SHIM_TEST_ASYNC_ALLOC=1` and
  `POLARIS_SHIM_TEST_DRIVER_ASYNC_ALLOC=1`.
- Strict managed-allocation mode wired:
  `POLARIS_SHIM_STRICT_MANAGED_ALLOC=1` makes exhausted or failed Polaris
  allocations return CUDA allocation errors instead of silently falling back
  to non-Polaris CUDA memory. This gives M5 KV-only experiments a way to
  prove the allocation path stayed inside the managed window.
- Size-policy allocator filter wired:
  `POLARIS_SHIM_MIN_MANAGED_ALLOC=<bytes>` and
  `POLARIS_SHIM_MAX_MANAGED_ALLOC=<bytes>` restrict Polaris interception to
  selected allocation sizes. Out-of-policy allocations intentionally fall
  through to the real CUDA allocator even in strict mode, while in-policy
  failures keep strict-mode CUDA error behavior. The `managed_alloc` harness
  covers this with `POLARIS_SHIM_TEST_FILTER=1` so M5 can target KV-sized
  allocations without hijacking small CUDA runtime/control allocations.
- Allocator observability wired:
  `POLARIS_SHIM_REPORT_STATS=1` prints an exit-time summary of intercepted
  allocation/free calls, size-policy pass-throughs, Polaris successes and
  failures, real CUDA fallback calls/results, live managed bytes, peak live
  managed bytes, and per-allocation-API selected counters. This gives
  llama.cpp/KV experiments a low-friction way to confirm the size filter
  selected the intended allocations before running the riskier kernel-deref
  fault path.
- Bootstrapped managed-window default improved: in
  `POLARIS_SHIM_BOOTSTRAP_RM_UVM=1` mode, the shim now derives the managed
  allocator window from the RM-reported fault-capable VA-space instead of the
  old 4 MiB harness default, with `POLARIS_SHIM_MANAGED_LENGTH_CAP` limiting
  the default bring-up window. Explicit `POLARIS_SHIM_MANAGED_BASE` /
  `POLARIS_SHIM_MANAGED_LENGTH` overrides still work for targeted smoke
  tests.
- Runtime setup compatibility slice wired: the shim forwards common runtime
  setup calls (`cudaSetDevice`, `cudaSetDeviceFlags`, `cudaGetDeviceFlags`,
  `cudaGetDevice`, `cudaGetDeviceCount`, `cudaGetDeviceProperties`,
  `cudaDeviceGetAttribute`, `cudaDeviceCanAccessPeer`,
  `cudaDeviceEnablePeerAccess`, `cudaRuntimeGetVersion`,
  `cudaDriverGetVersion`, and `cudaDeviceSynchronize`) and records runtime
  device selection before RM/UVM bootstrap when `POLARIS_SHIM_CUDA_ORDINAL` is
  not explicit. It also forwards memory/error query calls commonly used by CUDA allocator probes
  (`cudaMemGetInfo`, `cuMemGetInfo_v2`, runtime/driver error-name and
  error-string helpers, `cudaGetLastError`, and `cudaPeekAtLastError`) plus
  ordinary runtime stream/event lifecycle calls including
  `cudaStreamWaitEvent` and `cudaStreamIsCapturing`. The `managed_alloc`
  harness covers these with `POLARIS_SHIM_TEST_RUNTIME_SETUP=1`.
- Managed-memory advice compatibility wired: `cudaMemAdvise` on a
  shim-managed Polaris pointer returns success as a no-op instead of handing
  the external-range VA to CUDA's managed-memory subsystem. Non-Polaris
  pointers still fall through to the real runtime. The harness covers this
  with `POLARIS_SHIM_TEST_ADVISE=1`.
- llama.cpp host/query compatibility slice wired: the shim forwards pinned
  host memory APIs (`cudaMallocHost`, `cudaHostAlloc`,
  `cudaHostGetDevicePointer`, `cudaFreeHost`, `cudaHostRegister`, and
  `cudaHostUnregister`) plus kernel query/tuning helpers
  (`cudaFuncSetAttribute`, `cudaFuncGetAttributes`, and
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor`). The harness covers the
  host-memory subset with `POLARIS_SHIM_TEST_HOST_APIS=1`; function helpers
  are build-covered and left as pass-throughs because meaningful execution
  needs real kernel symbols.
- Memory-operation guard slice wired: the shim interposes common runtime and
  driver copy/fill calls (`cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`,
  `cudaMemsetAsync`, `cudaMemcpy2DAsync`, `cudaMemcpyPeerAsync`,
  `cudaMemcpy3DPeerAsync`, `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`, generic
  `cuMemcpy_v2`, their async variants, and
  `cuMemsetD8` / `cuMemsetD16` / `cuMemsetD32` variants), classifies whether
  a shim-managed Polaris pointer is involved, and fails such operations with
  `cudaErrorNotSupported` instead of handing unmapped Polaris VA to CUDA.
  This is intentionally a guard/scaffold; true transparent host copies and
  fills still need the host/device data path from the reload/offload
  machinery.
- IPC guard slice wired: the shim interposes `cuIpcGetMemHandle` and
  `cudaIpcGetMemHandle` and rejects shim-managed Polaris pointers with
  `cudaErrorNotSupported`, matching v1's no-IPC contract while allowing
  non-Polaris IPC calls to fall through.
- CUDA Graph guard/pass-through slice wired: the shim interposes runtime
  `cudaStreamBeginCapture`, `cudaStreamEndCapture`, and `cudaGraphLaunch`
  plus driver `cuStreamBeginCapture` / `cuStreamBeginCapture_v2`, returning
  the stream-capture unsupported error while shim-managed Polaris allocations
  are live. Runtime graph lifecycle/update calls (`cudaGraphInstantiate`,
  `cudaGraphExecUpdate`, `cudaGraphDestroy`, and `cudaGraphExecDestroy`) pass
  through for non-Polaris graph management. The `managed_alloc` harness covers
  the runtime and driver guards with `POLARIS_SHIM_TEST_GRAPH=1`.
- CUDA VMM compatibility slice wired: the shim interposes and forwards the
  driver VMM pool primitives used by llama.cpp's CUDA backend
  (`cuMemAddressReserve`, `cuMemAddressFree`, `cuMemCreate`, `cuMemRelease`,
  `cuMemMap`, `cuMemUnmap`, `cuMemSetAccess`, and
  `cuMemGetAllocationGranularity`). These remain pass-through diagnostics and
  workload-compatibility surfaces, not the v4 production path, because raw CUDA
  VMM VA is still not fault-capable. The `managed_alloc` harness can validate
  symbol coverage and driver pass-through with a non-deref invalid-argument
  probe with
  `POLARIS_SHIM_TEST_VMM=1`.
- Kernel-launch compatibility slice wired: the shim interposes and forwards
  runtime `cudaLaunchKernel` / `cudaLaunchKernelExC` and driver
  `cuLaunchKernel` / `cuLaunchKernelEx` so llama.cpp's ordinary `<<<...>>>`
  launches and CUDA 11.8+ PDL launch path stay visible through the shim. These
  are pass-throughs rather than guards because the production v4 path requires
  real kernels to fault on Polaris-managed VA. The `managed_alloc` harness
  covers symbol routing without executing kernels via
  `POLARIS_SHIM_TEST_LAUNCH=1`; intentionally invalid launch calls are not
  used because some CUDA runtime paths dereference launch metadata before
  returning an error.
- Remaining llama.cpp helper compatibility wired: the shim forwards driver
  device/context setup (`cuDeviceGet`, `cuDeviceGetAttribute`,
  `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`), runtime peer/PCI helpers
  (`cudaDeviceDisablePeerAccess`, `cudaDeviceGetPCIBusId`), host/cooperative
  launch helpers (`cudaLaunchHostFunc`, `cudaLaunchHostFunc_v2`,
  `cudaLaunchCooperativeKernel`), `cudaOccupancyMaxPotentialBlockSize`, and
  graph node helpers (`cudaGraphGetNodes`, `cudaGraphNodeGetType`,
  `cudaGraphKernelNodeGetParams`, `cudaGraphKernelNodeSetParams`). The
  harness exercises safe query/host-callback calls and otherwise validates
  symbol coverage without running kernels.
- Real llama.cpp regression added:
  `tests/llama_cpp/run_llama_shim_e2e.sh` / `make llama-e2e` defaults to
  `/home/wano/workspace/llama.cpp` and runs two checks. First, if the local
  llama.cpp binary exposes `POLARIS0`, it runs a real `llama-bench` workload
  on that device. Second, it runs unmodified CUDA `llama-bench` workloads under
  `LD_PRELOAD` for both the ordinary `cudaMalloc` path and the
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` / `cudaMallocManaged` path, and asserts
  that the shim bootstraps RM/UVM, registers a Polaris VA-space, routes real
  llama allocations through Polaris, creates UVM external ranges, registers
  static RM backing, records the expected per-API selected allocation counter,
  and in strict mode increments both `uvm_hook_calls` and `uvm_handled`.
- Static RM backend wired for integration testing only:
  `POLARIS_SHIM_STATIC_RM_BACKEND=1` requires in-shim RM/UVM bootstrap,
  allocates/frees RM `NV01_MEMORY_LOCAL_USER` objects per shim-managed
  allocation, registers them with `POLARIS_REGISTER_STATIC_BLOCK`, and also
  attaches the same RM tuple to the logical block with
  `POLARIS_REGISTER_BLOCK_BACKING`. The static registration preserves the
  existing strict llama gate, while the logical backing registration exercises
  the newer block-mapping bridge path. This is not the daemon-backed
  production spill/reload path, and it does not make CUDA runtime host copies
  into Polaris external VA safe.
- KV-only selection slice wired and validated against the local llama.cpp
  binary: `POLARIS_SHIM_REQUIRE_KV_SCOPE=1` uses the ggml allocation-scope hook
  to leave the copied model-buffer allocation on real CUDA memory while routing
  KV-cache allocations through Polaris. `POLARIS_SHIM_ALLOW_ZERO_MEMSET=1`
  accepts base-address zero-fill initialization within a selected allocation as
  a no-op declaration for these selected ranges; nonzero host copy/fill remains
  guarded. A temporary
  UVM-map-plus-CUDA-copy experiment was rejected after `cuMemcpyHtoD_v2`
  returned `CUDA_ERROR_INVALID_CONTEXT` without a current context and segfaulted
  inside `libcuda` with a current runtime context.
- Strict static-RM shim fault-path gate passes on the local SmolLM2
  `llama-bench` run for both llama.cpp allocator branches: the default probe
  selects KV allocations through runtime `cudaMalloc`, and the
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` probe selects KV allocations through
  runtime `cudaMallocManaged`. In both cases the shim passes copied model/init
  allocations through to real CUDA, accepts KV zero-fill initialization,
  disables CUDA Graph capture and CUDA PDL launch selection for the shim probe,
  and completes with `uvm_hook_calls` and `uvm_handled` increasing and no
  `uvm_no_pte` or `uvm_errors` increments. This validates the
  no-source-change KV-only path through the UVM bridge for the integration-test
  backend.
- Run llama.cpp end-to-end against shim+polaris.ko+polarisd using the
  daemon-backed spill/reload path. CUDA Graph mode may need to be disabled
  (or KV ranges excluded from graph capture); document the decision per
  integration option.
- Remaining production shim work: replace the static RM test backend with the
  daemon-backed spill/reload path, replace the fixed managed-window reservation
  model with workload-appropriate VA management, and document or disable CUDA
  Graph interactions.
- Compare throughput vs v3-lease path and vs vLLM/SGLang baselines.

### M6: Hardening

- Worker crash reaping (drop block→worker map entries when the
  `gpu_va_space_ptr` goes away under us).
- First worker-lifetime control-plane slice wired: v4 VA-space registrations
  record the registering process pid for diagnostics, `/sys/kernel/polaris/stats`
  reports `v4_worker_pids`, and an ignored non-CUDA test spawns a worker
  process that registers a VA-space plus block mapping and verifies process
  exit closes the fd and reaps both entries. This validates the current
  chardev ownership hook; true UVM `gpu_va_space_ptr` invalidation callbacks
  remain deferred.
- Module unload with workers still attached: rely on UVM's
  `try_module_get` + `synchronize_rcu` ordering; verify under stress.
- Tracing for: faults serviced, faults rejected, spills, reloads, bridge
  call latency, policy-mirror sequence drift.
- Stress: dynamic KV growth, fragmentation pressure, OOM behavior on
  both VRAM (RM alloc fails) and host pinned pool sides.

### M7: PyTorch / vLLM integration

- PyTorch caching allocator backend (`PYTORCH_CUDA_ALLOC_CONF`-driven or
  custom backend) — LD_PRELOAD is not sufficient.
- Run vLLM under shim+polaris.ko+polarisd. Document remaining gaps
  (CUDA Graphs, IPC, paged-attention kernels that bypass the allocator).

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
  shell. Production reload now goes through `uvm_polaris_map_external_allocation`.
- v3 llama.cpp fault-worker mode: delete.
- v3 README claims about production replayable-fault paging: rewrite
  once M5 lands.

## Open Questions

1. Can the shim reliably intercept PyTorch's caching allocator without
   a custom backend? Confirmed **no** — M7 needs a
   `PYTORCH_CUDA_ALLOC_CONF` backend ship. M5 stays llama.cpp-only.
2. ~~PMM partition (option B) vs per-chunk PMM calls (option A)~~ — moot.
   polaris.ko allocates per-block RM memory objects and lets the
   `uvm_polaris_map_external_allocation` bridge install PTEs through
   UVM's external-range path. PMM is not exported.
3. Cold-fault upcall mechanism: netlink, chardev, or io_uring-style
   submission queue? Pick when M2 lands.
4. Multi-GPU: one polaris.ko block pool per GPU is mechanical; the
   polarisd block table needs explicit GPU affinity. Decide before M4.
5. UVM rebases: how often does `service_fault_batch_dispatch`'s prologue
   and the `uvm_va_range_external_t` lookup surface shift across driver
   versions? Track across 535 / 545 / 550 / 555 / 560 / next.
6. Host→device copy path for cold reload: the bridge handles PTE install,
   but moving bytes from the pinned host pool into the freshly allocated
   RM device memory still needs a concrete path. Initial slice uses an
   in-context `cuMemcpyHtoDAsync` on a polaris-owned helper thread; M3
   should evaluate whether UVM exposes a cheaper kernel-side copy
   primitive without needing the dropped channel-manager export.
7. Lock-budget audit on the hook hot path: the dispatcher runs with
   va_space read lock + service_lock held; the bridge re-acquires them.
   Measure bridge call latency under contention before M5.
8. Stable VA-space key: today it is
   `(gpu_id, gpu_va_space->user_rm_client, gpu_va_space->user_rm_va_space)`.
   RM object handles are scoped to an RM client, so `va_space_token` alone is
   not globally unique.
