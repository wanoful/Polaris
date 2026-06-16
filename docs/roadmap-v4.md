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

> KV-cache-only, kernel-directed paging on a UVM-registered fault-capable
> VA-space. A small LD_PRELOAD shim hides VA-space setup, identifies
> llama.cpp KV-cache allocations, and registers only those KV ranges as UVM
> external ranges. Model weights, tensor upload buffers, CUDA workspaces, and
> unrelated CUDA allocations stay on the application's normal CUDA path.
> polaris.ko owns the logical KV block registry, RM-backed residency metadata,
> and fault hook implementation; PTE installs go through UVM's existing
> external-range map path via GPL-exported bridges. polarisd owns daemon RM
> allocation, pinned-host spill slots, copy execution, and policy.

v4 is **not** a general-purpose transparent CUDA memory pager. The production
claim is narrower and more useful: no-source-change llama.cpp execution where
KV cache storage is Polaris-managed and model weights are not.

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
- Application transparency is downgraded from "no source changes ever" to a
  KV-only contract. llama.cpp allocator interception is the first target:
  `POLARIS_SHIM_REQUIRE_KV_SCOPE=1` must leave copied model/init buffers on
  real CUDA memory and route KV-cache allocations through Polaris. PyTorch /
  vLLM still require an allocator backend in M7 because their caching
  allocators bypass driver-API interception.

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
(unmodified)                  steer KV cache into       RM-backed residency,   table, policy,
                              fault-capable VA-space    UVM fault hook         pinned host pool
                                                        UVM fault hook         orchestration
```

| Layer | Owns | Does not own |
|-------|------|--------------|
| worker | its CUDA context, the VA-space object, its own kernels | mapping, unmapping, block IDs, eviction |
| shim | per-process VA-space creation, allocator interception, UVM external-range registration for KV VA, ioctl registration with polaris.ko | data movement, policy, block state, PTE programming |
| polaris.ko | block→worker map, policy mirror, fault hook implementation, RM-backed residency metadata for KV chunks, scheduling slow-path upcalls to polarisd | direct PTE writes (delegated to UVM through the bridge), the fault buffer, CE channel state, model-weight ownership |
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
  1. Reserve a contiguous VA range inside the Polaris VA-space and return
     that pointer to the worker.
  2. Register the VA range with UVM as an **external range**
     (`UvmCreateExternalRange` or equivalent) — required by the bridge,
     because `uvm_polaris_map_external_allocation` looks up an existing
     `uvm_va_range_external_t` covering the fault address.
  3. Split the range into Polaris-sized KV chunks and tell polaris.ko which
     VA chunk backs each logical Polaris block via
     `POLARIS_REGISTER_BLOCK_MAPPING` (so the fault hook can resolve
     `rm_client_token + va_space_token + addr → block_id`).
  4. Do **not** map any pages. Faults populate them on demand.
- Intercept `cuMemFree` / `cudaFree` to unregister the range, drop the
  external range, and let polaris.ko reclaim block table state.
- KV-only transparency limits (see also "KV-Only Transparency Contract"
  below):
  - The supported llama.cpp path is KV allocation + direct kernel dereference
    + free. Model-weight upload and unrelated CUDA buffers intentionally pass
    through to CUDA.
  - General CUDA host copies/fills involving Polaris pointers are unsupported
    except for the explicit KV zero-fill contract used by llama.cpp.
  - CUDA IPC (`cuIpcGetMemHandle`) and CUDA Graph capture of KV accesses are
    out of scope for v4 and documented as unsupported.
  - PyTorch caching allocator ownership of KV memory remains out of scope for
    M5. The shim now covers runtime stream-ordered symbols for smoke tests,
    but M7 still needs an explicit allocator backend to make framework-owned
    KV allocations intentional.

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

## KV-Only Transparency Contract

This section replaces the original v4 claim of unconditional zero source
changes. The production contract is deliberately KV-cache-only:

**Transparent through shim, no source changes for llama.cpp KV cache**:
- The shim interposes llama.cpp's runtime/driver allocation APIs
  (`cudaMalloc`, `cudaMallocManaged`, stream-ordered allocation variants, and
  `cuMemAlloc*`) but selects an allocation only when the ggml KV-scope hook
  marks the current allocation context as KV cache.
- Selected KV allocations return stable Polaris VA pointers. Their physical
  RM backing is materialized on first GPU access through UVM replayable faults
  and later can be spilled/reloaded without changing the worker pointer.
- Model weights, tensor upload buffers, CUDA runtime/control allocations,
  temporary workspaces, and copied model/init buffers remain on the normal
  CUDA allocator path.
- KV zero-fill initialization is accepted only under the explicit
  `POLARIS_SHIM_ALLOW_ZERO_MEMSET=1` contract. Nonzero host copy/fill into a
  Polaris pointer remains guarded.

**Requires Polaris KV allocations to be registered as external ranges**:
- Every selected KV VA range must be a UVM external range and must have a
  matching `POLARIS_REGISTER_BLOCK_MAPPING` entry so the fault hook can
  resolve `(worker key, fault address) -> logical block`.
- Pointer metadata queries for selected KV pointers are answered by the shim;
  the driver must still see the VA as an external range for the bridge path to
  install PTEs.

**Explicitly not transparent in v4**:
- Arbitrary CUDA buffers. POLARIS does not try to page model weights,
  activations, cuBLAS/cuDNN workspaces, temporary upload buffers, or general
  application allocations.
- General `cudaMemcpy` / `cudaMemset` involving Polaris pointers. Only the
  KV-specific zero-fill contract is allowed today; daemon-backed spill/reload
  data movement uses `POLARIS_RM_COPY` through UVM-owned staging instead.
- `cuIpcGetMemHandle` / CUDA IPC for Polaris pointers.
- CUDA Graph capture/replay and CUDA PDL while Polaris allocations are live.
- PyTorch caching allocator ownership of KV memory. M7 needs an explicit
  allocator backend to force PyTorch/vLLM KV allocations through Polaris
  deliberately.

The course / paper claim should be: **the UVM hook/PTE bridge, daemon-owned
RM backing path, and llama.cpp KV-only shim registration path are wired; the
remaining work is KV workload hardening and pressure benchmarking, not
general CUDA-buffer transparent paging**.

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
- Completion-backed bridge slice complete: successful
  `POLARIS_COMPLETE_OPERATION` replies can now carry optional RM backing
  metadata for resident logical blocks (`rm_control_fd`, `h_client`,
  `h_memory`, `length`). polaris.ko stores that metadata on `ALLOC`, `RELOAD`,
  and `COW_BREAK` completions, clears it on offload/free/error transitions, and
  uses it for the same logical block fault mapping path as
  `POLARIS_REGISTER_BLOCK_BACKING`. The M2 harness's
  `--complete-backed-refault` mode reserves a real logical block without
  `DEFER_FAULT`, completes the queued `ALLOC` decision with diagnostic RM
  metadata, registers the worker mapping, fault-maps, unmaps by `block_id`, and
  refaults successfully. This closes the kernel ABI gap for daemon/runtime
  executors to publish UVM-bridge-mapable residency, but the executor still
  needs real daemon-owned RM allocation and host/device copy wiring.
- Daemon-owned RM allocation/free slice wired: `polarisd` now has an opt-in
  `POLARISD_RM_BACKING=1` backend that opens RM, creates a root client/device/
  subdevice, allocates `NV01_MEMORY_LOCAL_USER` per `ALLOC` decision, and
  returns the real daemon-owned `(rm_control_fd, h_client, h_memory, length)`
  tuple through `POLARIS_COMPLETE_OPERATION`. `FREE` decisions release the
  daemon-owned RM object by block id, so normal `BLOCK_RELEASE` /
  `SESSION_DESTROY` cleanup now has a real RM owner on the daemon path.
- CUDA-copy visibility probe wired: `tests/m2/m2_static_block_setup
  --cuda-copy-probe` maps a harness-created RM vidmem allocation through the
  public UVM external-allocation ioctl, creates a normal CUDA primary context
  for the same device, and attempts `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2`
  against the external VA in an isolated child process. The probe is diagnostic:
  success would make daemon-side CUDA copy worth integrating, while CUDA errors
  or a child crash point the production RM-backed spill/reload path toward a
  UVM/kernel copy helper. Local validation on 2026-06-14 reached
  `cuMemcpyHtoD_v2` after successful RM allocation and public UVM mapping, then
  the isolated child terminated with `SIGSEGV`, so the current implementation
  should not rely on ordinary CUDA copy APIs for RM-backed external VA.
- RM CPU-map visibility probe wired: the same M2 harness has
  `--rm-cpu-map-probe`, which calls `NV_ESC_RM_MAP_MEMORY` for the harness RM
  vidmem object and then touches the returned CPU address in an isolated child.
  Local validation on 2026-06-14 showed the RM map ioctl returned a CPU address,
  but touching it terminated the child with `SIGSEGV`, so a simple daemon-side
  `memcpy` path is also not viable for this backing shape.
- RM physical-address visibility probe wired: UVM now exports the diagnostic
  `uvm_polaris_probe_external_allocation`, polaris.ko exposes
  `POLARIS_PROBE_RM_PHYS`, and the M2 harness's `--rm-phys-probe` mode
  fault-maps a logical RM-backed block, asks UVM/RM for page size plus
  GPU-visible physical addresses, and then verifies the same block still
  unmaps/refaults through the bridge. This proves only that the backing shape
  can expose geometry a later UVM/kernel copy helper needs; it does **not** copy
  bytes, validate CE programming, or remove the RM-backed spill/reload guard.
- RM CE-copy probe wired: UVM now exports the diagnostic
  `uvm_polaris_probe_external_copy`, polaris.ko exposes
  `POLARIS_PROBE_RM_COPY`, and the M2 harness's `--rm-copy-probe` mode
  follows the completion-backed resident path, then stages a deterministic CPU pattern in
  UVM-owned sysmem DMA memory, CE-copies it into the RM allocation's
  GPU-visible physical address, CE-copies it back to sysmem, and verifies byte
  integrity. Local validation on 2026-06-14 copied and verified the full 2 MiB
  contiguous vidmem harness allocation (`page=0x200000 count=1
  mismatch=0xffffffffffffffff`) and then unmap/refaulted the block through the
  bridge. This proves the narrow local RM-backed shape can move bytes through
  UVM's CE path; it is still diagnostic, not the daemon integration itself.
- RM user-buffer copy primitive wired: UVM now exports
  `uvm_polaris_copy_external_allocation`, polaris.ko exposes `POLARIS_RM_COPY`,
  and the M2 harness's `--rm-copy-roundtrip` mode copies a deterministic
  userspace CPU buffer into completion-backed RM backing, copies it back into a
  second userspace buffer, and verifies byte integrity in userspace. This keeps
  the same narrow bring-up limit as the probe (contiguous vidmem reached through
  UVM CE staging), but changes the endpoint from an internal diagnostic pattern
  to the user CPU pointer shape needed by daemon-backed `OFFLOAD` and `RELOAD`.
- RM-backed spill/reload execution wired: `POLARIS_SPILL_BLOCK` validates the
  logical block, tears down observed UVM mappings, and queues `OFFLOAD` for
  RM-backed bridge-resident blocks. `polarisd` uses `POLARIS_RM_COPY` to copy
  daemon-owned RM backing to the pinned CPU pool, frees the RM allocation, then
  later services `RELOAD` by allocating fresh daemon-owned RM backing and
  copying the CPU buffer back.
- RM-backed overwrite COW execution wired: COW decisions now carry the source
  `block_id` in the reserved decision metadata when no legacy CUDA physical
  handle exists. `polarisd` services RM-backed `COW_BREAK` by staging source RM
  backing through the pinned CPU pool with `POLARIS_RM_COPY`, allocating fresh
  daemon-owned destination RM backing, copying the staged bytes into it, and
  returning the destination RM tuple through `POLARIS_COMPLETE_OPERATION`. The
  M2 `--rm-cow-roundtrip` mode validates parent and child byte preservation for
  the current `SESSION_BRANCH` + overwrite-reserve COW surface with a harness
  executor, and `--daemon-rm-cow-roundtrip` validates the same byte preservation
  with a real `polarisd` running `POLARISD_RM_BACKING=1` and daemon-owned
  parent/child RM backing. Full write-fault permission-split COW remains future
  M4 hardening.
- Live-backing FREE lifetime slice wired: `BLOCK_RELEASE` and
  `SESSION_DESTROY` now queue daemon `FREE` decisions for resident legacy
  physical handles, RM-backed bridge-resident logical blocks, and CPU-offloaded
  backing instead of silently dropping kernel metadata. Static RM diagnostics
  and the shim's integration-test backend opt into
  `POLARIS_RELEASE_FLAG_CALLER_OWNS_BACKING` so harness-owned RM handles remain
  caller-cleaned; production/default release keeps daemon-owned backing on the
  explicit `FREE` path. `libpolaris/tests/kernel_spill_state.rs` covers legacy
  resident release, RM-backed release without `gpu_phys_handle`, and RM-backed
  session-destroy cleanup.
- Deferred logical-block materialization wired: when the UVM hook sees a
  registered logical block mapping with no RM backing yet, it can enter the
  existing bounded `ALLOC` decision path, wait for userspace to complete the
  operation with RM backing metadata, and bridge-map the same fault before
  returning `HANDLED`. The M2
  `--deferred-complete-fault` diagnostic reserves a block with
  `DEFER_FAULT`, registers only the worker mapping, completes the first
  synthetic fault through `POLARIS_COMPLETE_OPERATION`, and verifies
  unmap/refault still uses the completed logical backing. This closes the
  kernel-side gap between the shim's deferred allocation shape and the
  completion-backed RM residency contract.
- UVM-hook reload hardening wired: when a replayable UVM fault hits a
  `CpuOffloaded` daemon-backed RM block, polaris.ko now queues one real daemon
  `RELOAD` decision and returns `DEFERRED` instead of waiting inside the
  hook or reporting the fault serviced before a PTE exists. Replayed faults that arrive
  while the block is already `ReloadPending` / `AllocPending` /
  `OffloadPending` / `FreePending` are also treated as queued rather than
  enqueueing duplicate decisions or reporting a fatal error. Process-context
  synthetic fault paths still use the bounded wait mode. The production reload
  remains real daemon-backed RM allocation plus `POLARIS_RM_COPY`; static RM is
  still diagnostic/integration-only.
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
  polarisd performs device → host copy and backing release through the normal
  completion path. The M2 harness validates the ioctl's unresident block
  rejection (`--spill-validation`); positive RM-backed copy coverage now lives
  in the RM roundtrip and llama gates below because real RM copy requires an
  observed UVM VA-space context.
- Positive daemon/runtime spill test wired: the ignored root/GPU
  `polaris-runtime` fault-smoke creates a resident logical block through the
  runtime decision worker, writes a CUDA-visible pattern, calls
  `POLARIS_SPILL_BLOCK`, verifies the OFFLOAD completion moves the block to
  `CpuOffloaded`, reloads the same VA through overwrite reserve, and checks
  the bytes survive the device → host → device cycle. A separate long-running
  `polarisd` process soak remains useful, but the production decision path is
  no longer covered only by fake executor state-machine tests.
- Positive RM harness spill/reload test wired:
  `tests/m2/m2_static_block_setup --rm-spill-reload-roundtrip` writes a
  deterministic userspace pattern into RM backing with `POLARIS_RM_COPY`,
  spills to a CPU buffer, reloads into fresh RM backing, refaults through the
  bridge, and verifies byte integrity. This exercises the same UVM copy helper
  used by the daemon path without requiring static RM registration.
- Focused daemon-backed RM spill/reload gate wired:
  `tests/m2/m2_static_block_setup --daemon-rm-spill-reload-roundtrip` requires a
  real `polarisd` already running with `POLARISD_RM_BACKING=1`, reserves a
  deferred logical block without static RM registration or harness-owned
  logical backing, lets the daemon handle `ALLOC`, writes/verifies bytes through
  `POLARIS_RM_COPY`, queues `POLARIS_SPILL_BLOCK`, waits for daemon `OFFLOAD`,
  forces daemon `RELOAD`, refaults, and verifies cleanup drains daemon `FREE`.
  This closes the gap between the harness-owned RM byte roundtrip and the
  llama.cpp gate.
- Focused daemon-backed RM spill/reload stress wired:
  `tests/m2/m2_static_block_setup --daemon-rm-spill-reload-stress` runs repeated
  real-daemon `POLARIS_SPILL_BLOCK` / `OFFLOAD` / `RELOAD` cycles on one deferred
  logical block with daemon-owned RM backing, writes a different deterministic
  pattern through `POLARIS_RM_COPY` each cycle, refaults after reload, verifies
  byte integrity, and checks daemon decision-queue drain between cycles. This is
  a focused single-block stress gate; fragmentation, dynamic KV growth, and OOM
  pressure remain broader M6 work.
- Daemon-backed RM multi-block stress wired:
  `tests/m2/m2_static_block_setup --daemon-rm-multi-block-stress` reserves
  several deferred logical blocks in one Polaris session, materializes each block
  through real daemon-backed `ALLOC`, writes unique byte patterns into
  daemon-owned RM backing, spills all blocks, waits for daemon `OFFLOAD`,
  reloads them in reverse order, refaults, verifies byte integrity for every
  block, and drains daemon `FREE` cleanup. This broadens coverage beyond a
  single block while keeping static RM diagnostic-only; dynamic KV growth,
  fragmentation, and OOM pressure remain broader M6 work.
- Daemon-backed RM dynamic fragmentation stress wired:
  `tests/m2/m2_static_block_setup --daemon-rm-dynamic-fragmentation-stress`
  reserves five deferred logical blocks in one Polaris session, materializes
  them through real daemon-backed `ALLOC`, writes deterministic bytes into
  daemon-owned RM backing, releases alternating blocks to create holes, waits
  for daemon `FREE` cleanup and kernel block removal, regrows those token ranges
  as fresh deferred blocks, writes new deterministic bytes, spills all live
  blocks, waits for daemon `OFFLOAD`, reloads/refaults them, verifies byte
  integrity for survivor and regrown blocks, and checks `pending_decs=0` plus
  `static_blocks=0`. This covers the first dynamic-growth / fragmentation M6
  slice on the production daemon-backed RM path; host-pool and RM allocation
  OOM pressure are now covered separately.
- Daemon-backed RM host-pool OOM pressure wired:
  `tests/m2/m2_static_block_setup --daemon-rm-host-pool-oom-pressure` runs
  against a real `polarisd` started with
  `POLARISD_RM_BACKING=1 POLARISD_CPU_POOL_BYTES=2097152`, reserves two
  deferred logical blocks, materializes both through daemon-backed `ALLOC`,
  writes deterministic bytes into daemon-owned RM backing, spills the first
  block into the one-block CPU pool, then spills the second block and observes
  real daemon `-ENOMEM`. The kernel retries the RM-backed `OFFLOAD` decision,
  marks the second block `Evicted` after the bounded retry limit, and cleanup
  releases both the CPU-offloaded block and the evicted block through daemon
  `FREE`, with `blocks=0`, `pending_decs=0`, `static_blocks=0`, and
  `block_mappings=0`. This covers the host pinned-pool side of the M6 OOM
  requirement on the production daemon-backed RM path without static RM or a
  fake `POLARIS_TEST_ERROR`.
- Daemon-backed RM allocation OOM pressure wired:
  `tests/m2/m2_static_block_setup --daemon-rm-alloc-oom-pressure` runs against
  a real `polarisd` started with
  `POLARISD_RM_BACKING=1 POLARISD_RM_PRESSURE_OUTSIDE_FB_RANGE=1
  POLARISD_GPU_BUDGET_BYTES=68719476736`, registers a large UVM external range,
  reserves one 32 GiB deferred logical block, and dispatches a synthetic fault
  that queues a real daemon-backed `ALLOC`. Because the kernel GPU budget is
  intentionally above the request, the decision reaches the daemon; because
  the daemon constrains the real `NV01_MEMORY_LOCAL_USER` request to a physical
  FB range outside local memory with `NVOS32_ALLOC_FLAGS_USE_BEGIN_END`, RM
  returns real `NV_ERR_NO_MEMORY` (`0x51`) before bridge mapping instead of
  using static RM or `POLARIS_TEST_ERROR`. `polarisd` now treats non-warning
  `NV_STATUS` values as failures rather than using the old high-bit heuristic,
  so the daemon completes the decision with `-ENOMEM`. The kernel retries,
  marks the block `Evicted`, reports UVM `ERROR` rather than `HANDLED`,
  verifies `uvm_errors` increments while `uvm_handled` does not, verifies
  `uvm_last_map_ret` is unchanged so the bridge was not the failure source, and
  cleans up with `blocks=0`, `pending_decs=0`, `static_blocks=0`, and
  `block_mappings=0`. This covers the VRAM/RM allocation side of the M6 OOM
  requirement with deterministic RM allocation pressure.
- Daemon-backed RM near-capacity soak wired and verified on 2026-06-15:
  `tests/m2/m2_static_block_setup --daemon-rm-near-capacity-soak` runs against
  a real `polarisd` started with
  `POLARISD_RM_BACKING=1 POLARISD_GPU_BUDGET_BYTES=4194304`, reserves four
  deferred 2 MiB logical blocks in one Polaris session, materializes them
  through daemon-backed `ALLOC`, writes deterministic bytes with
  `POLARIS_RM_COPY`, and verifies the two-block/4 MiB resident-set cap via
  `resident=2`, `gpu_used_mib=4`, and `offloaded=2`. The reload/refault phase
  exercises the async UVM-hook reload path: the first fault on a CPU-offloaded
  block queues daemon `RELOAD` and returns `DEFERRED`, the harness waits for
  daemon completion to publish fresh RM backing, and a second refault returns
  `HANDLED` after mapping the daemon-owned backing before byte-integrity
  verification. Cleanup checks
  `pending_decs=0`, `blocks=0`, and `static_blocks=0`. The gate now also
  validates M6 bridge telemetry by checking `uvm_bridge_map_calls`,
  `uvm_bridge_map_ok`, `uvm_bridge_map_err`, `uvm_bridge_map_last_ns`, and
  `uvm_bridge_map_avg_ns` around the same daemon-backed faults. Local
  validation used the patched NVIDIA module, real `polaris.ko`, and real
  daemon-backed `polarisd`; daemon logs showed real `RM ALLOC` / `RM OFFLOAD` /
  `RM RELOAD` / `RM FREE` operations with no timeout or stale-completion lines.
- Focused daemon-backed RM COW gate wired:
  `tests/m2/m2_static_block_setup --daemon-rm-cow-roundtrip` requires the same
  real daemon-backed RM path, reserves a deferred parent block without static RM
  registration or harness-owned logical backing, lets `polarisd` handle parent
  `ALLOC`, writes deterministic parent bytes with `POLARIS_RM_COPY`, branches the
  session, triggers overwrite-reserve `COW_BREAK`, lets `polarisd` publish fresh
  daemon-owned child RM backing, fault-maps the child, verifies parent and child
  bytes, and waits for daemon `FREE` cleanup to drain. This removes the remaining
  harness-executor dependency from the RM-backed overwrite COW gate; permission-
  based write-fault COW and broader stress remain M4/M6 work.
- Focused daemon-backed observed mapping key isolation wired:
  `tests/m2/m2_static_block_setup --daemon-rm-observed-mapping-key-isolation`
  requires a real `polarisd` with `POLARISD_RM_BACKING=1`, reserves the primary
  logical block with `DEFER_FAULT`, lets daemon `ALLOC` publish the RM backing,
  registers an alternate same-address mapping under a different
  `(rm_client_token, va_space_token)` key, and verifies only the truly faulted
  key records an observed UVM `gpu_va_space_ptr`. The alternate block unmap
  returns `unmapped_count=0`, the primary returns `unmapped_count=1`, the primary
  refault maps daemon-owned RM backing again, and cleanup drains daemon `FREE`
  with `static_blocks=0`.
- Kernel state-machine coverage added: `libpolaris` has ignored root-only
  `kernel_spill_state` tests that drive legacy
  `ALLOC → POLARIS_SPILL_BLOCK/OFFLOAD → BLOCK_RESERVE/RELOAD` through
  `GET_DECISION` / `COMPLETE_OPERATION` without invoking CUDA, and a separate
  real-`polarisd` RM-backed `ALLOC`/`FREE` lifetime test. They catch the
  control-plane contract when the CUDA/UVM data path is not safe to run, but
  they do not prove RM-backed byte movement because `POLARIS_RM_COPY` needs an
  observed UVM `gpu_va_space_ptr`. It is build-covered by `cargo test`;
  execution requires a freshly loaded `polaris.ko`.
- Reload state-machine fix: overwriting an existing single-ref block now
  triggers synchronous fault resolution when the block is not resident,
  so `CpuOffloaded` blocks queue `RELOAD` instead of returning an unmapped
  VA.
- Reload on fault: process-context callers still wait for fresh RM backing,
  copy host → device, bridge-map, and return `HANDLED`; UVM-hook callers queue
  daemon `RELOAD`, return `DEFERRED` while no PTE has been installed, and map
  on a later refault once daemon-published backing is resident.
- Single-worker microbenchmark wired:
  `tests/m2/m2_static_block_setup --daemon-rm-single-worker-microbench` runs
  against a real `polarisd` started with
  `POLARISD_RM_BACKING=1 POLARISD_GPU_BUDGET_BYTES=4194304`, registers a
  six-block external range over deferred logical blocks, drives a real CUDA
  one-thread write kernel in a strided pattern over a two-block resident
  budget, and validates the final bytes through `POLARIS_RM_COPY`. The gate
  requires daemon offload/reload counters and UVM bridge-map counters to move,
  `uvm_errors` to remain stable, `static_blocks=0`, and cleanup through
  daemon `FREE`. This closes the M3 single-worker microbenchmark item without
  using static RM or fake errors. Local focused validation on 2026-06-15 used
  the patched NVIDIA module, real `polaris.ko`, and real daemon-backed
  `polarisd`; the composed reduced soak below also exercises this gate.

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
- RM-backed overwrite COW data path wired for the same
  `SESSION_BRANCH` + overwrite-reserve surface: `polarisd` stages source RM
  backing through the pinned CPU pool, publishes fresh daemon-owned RM backing
  for the child block, the M2 `--rm-cow-roundtrip` diagnostic verifies both
  parent and child bytes with a harness executor, and
  `--daemon-rm-cow-roundtrip` verifies the same split with a real daemon-backed
  parent allocation and daemon-executed `COW_BREAK`. This does not yet implement
  permission-based write-fault COW on already-mapped read-mostly pages.
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
- Block-count managed-window cap wired: `POLARIS_SHIM_MANAGED_BLOCKS=<count>`
  caps the bootstrapped managed window to `count * POLARIS_SHIM_BLOCK_SIZE`
  when no explicit `POLARIS_SHIM_MANAGED_LENGTH` override is present. The
  existing byte cap still applies, so M5 workload runs can bound the registered
  external-range VA by either bytes or logical KV block count. The
  `managed_alloc` harness covers this with `POLARIS_SHIM_TEST_BLOCK_WINDOW=1`,
  which admits two one-block allocations and rejects the third in strict mode.
- In-place registered-window growth wired: the shim can start with an initial
  v4 fault window (`POLARIS_SHIM_MANAGED_INITIAL_BLOCKS` or
  `POLARIS_SHIM_MANAGED_INITIAL_LENGTH`) smaller than the allocator capacity
  and grow the same `POLARIS_REGISTER_VASPACE` registration before handing out
  later token spans. The kernel treats same-fd/same-key
  `POLARIS_REGISTER_VASPACE` calls as monotonic managed-length updates and
  refreshes the fast UVM-hook lookup entry; different keys remain rejected.
  `POLARIS_SHIM_TEST_GROW_WINDOW=1` covers three one-block allocations through
  an initially one-block window without static RM. This narrows the fixed
  window gap, but full workload-specific VA reclamation is still future work.
- Tail registered-window reclamation wired: successful frees return token
  spans, collapse contiguous free spans at the allocation high-water mark, and
  shrink the same v4 `POLARIS_REGISTER_VASPACE` registration back to the
  highest live token. The kernel permits same-key shrink only after no block or
  static mapping remains beyond the new end, so live fault coverage cannot be
  truncated under an existing allocation. `POLARIS_SHIM_TEST_RECLAIM_WINDOW=1`
  covers grow to three one-block allocations, tail free/shrink, and a later
  reallocation that regrows the registered window without static RM. Interior
  holes are still reused but do not compact the bounded capacity.
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
  plus driver `cuStreamBeginCapture` / `cuStreamBeginCapture_v2` and
  `cuStreamEndCapture` / `cuStreamEndCapture_v2`, returning the stream-capture
  unsupported error while shim-managed Polaris allocations are live. If capture
  starts before any Polaris allocation exists, selected-size
  `cudaMallocAsync` / `cuMemAllocAsync_v2` calls pass through to CUDA instead
  of returning Polaris VA, so captured graphs do not record work against
  unmapped Polaris external ranges. Runtime graph lifecycle/update calls
  (`cudaGraphInstantiate`,
  `cudaGraphExecUpdate`, `cudaGraphDestroy`, and `cudaGraphExecDestroy`) pass
  through for non-Polaris graph management. The `managed_alloc` harness covers
  the runtime and driver live-allocation guards with `POLARIS_SHIM_TEST_GRAPH=1`
  and the capture-before-allocation fallback with
  `POLARIS_SHIM_TEST_GRAPH_CAPTURE_ALLOC=1`.
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
  launches stay visible through the shim. Ordinary launches remain
  pass-throughs because the production v4 path requires real kernels to fault
  on Polaris-managed VA. CUDA Programmatic Dependent Launch remains unsupported
  for live Polaris allocations: when `cudaLaunchKernelExC` /
  `cuLaunchKernelEx` configs carry
  `ProgrammaticStreamSerialization` or `ProgrammaticEvent` attributes and a
  shim-managed allocation is live, the shim returns `cudaErrorNotSupported`
  before forwarding to CUDA. The `managed_alloc` harness covers symbol routing
  without executing kernels via `POLARIS_SHIM_TEST_LAUNCH=1` and covers the PDL
  guard with `POLARIS_SHIM_TEST_PDL_GUARD=1`.
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
  llama allocations through Polaris, creates UVM external ranges, starts
  `polarisd` with daemon-owned RM backing, observes daemon RM allocation for
  the deferred logical blocks, records the expected per-API selected allocation
  counter, and in strict mode increments both `uvm_hook_calls` and
  `uvm_handled` without static RM registration.
- Per-chunk llama.cpp KV residency wired: selected llama.cpp KV allocations
  still return one contiguous Polaris VA range to ggml, but the shim now
  reserves/registers one logical Polaris block per `POLARIS_SHIM_BLOCK_SIZE`
  chunk inside that range. This keeps llama.cpp's pointer/range semantics
  intact while giving polaris.ko multiple independently evictable resident
  units for a single KV allocation.
- Dynamic-window llama.cpp regression variant added:
  `POLARIS_LLAMA_RUN_DYNAMIC_WINDOW_PROBE=1` runs an additional unmodified
  CUDA `llama-bench` probe with a one-block initial registered v4 fault window
  and a larger block-count capacity, requiring the shim to report both
  `managed_window_grow_calls` and `managed_window_shrink_calls` while the
  workload still reaches daemon-published RM backing and UVM handled faults.
  This validates the grow/reclaim allocator policy against the real
  no-source-change llama KV path without static RM.
- Small-budget llama.cpp KV pressure gate added and validated on 2026-06-16:
  `POLARIS_LLAMA_RUN_PRESSURE_PROBE=1` restarts `polarisd` with
  `POLARISD_RM_BACKING=1` and a 4 MiB default budget
  (`POLARIS_LLAMA_PRESSURE_BUDGET_BYTES`), mirrors that budget and the CPU pool
  into the shim-registered transient GPU policy state, runs an unmodified CUDA
  `llama-bench` workload, and requires daemon-backed `offloads`, `reloads`,
  `uvm_bridge_map_calls`, and `uvm_bridge_map_ok` to increase with
  `uvm_no_pte` and `uvm_errors` stable. Local validation with the SmolLM2 model
  observed `offloads: 34 -> 45`, `reloads: 28 -> 37`, and bridge maps
  `calls 2644 -> 2838` / `ok 2594 -> 2785`.
- Sustained llama.cpp KV pressure gate added and validated on 2026-06-16:
  `POLARIS_LLAMA_RUN_SUSTAINED_PRESSURE_PROBE=1` reuses the same KV-only
  daemon-backed pressure machinery with a longer `llama-bench` profile
  (`POLARIS_LLAMA_SUSTAINED_PROMPT_TOKENS`, `POLARIS_LLAMA_SUSTAINED_GEN_TOKENS`,
  `POLARIS_LLAMA_SUSTAINED_REPETITIONS`) so short smoke coverage and longer
  local pressure coverage share one verification path. Local SmolLM2 validation
  observed `offloads: 12 -> 206`, `reloads: 10 -> 202`, and bridge maps
  `calls 930 -> 5903` / `ok 912 -> 5882`.
- Static RM backend wired for integration testing only:
  `POLARIS_SHIM_STATIC_RM_BACKEND=1` requires in-shim RM/UVM bootstrap,
  allocates/frees RM `NV01_MEMORY_LOCAL_USER` objects per shim-managed
  allocation, registers them with `POLARIS_REGISTER_STATIC_BLOCK`, and also
  attaches the same RM tuple to the logical block with
  `POLARIS_REGISTER_BLOCK_BACKING`. The static registration is now diagnostic
  and integration-only; the strict llama gate below uses daemon-published RM
  backing instead. Static RM remains outside the daemon-backed production
  spill/reload path, and it does not make CUDA runtime host copies into Polaris
  external VA safe.
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
- Strict daemon-backed shim fault-path gate verified on 2026-06-15 with the
  local SmolLM2 `llama-bench` run for both llama.cpp allocator branches. The
  default probe selected KV allocations through runtime `cudaMalloc`; the
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` probe selected KV allocations through
  runtime `cudaMallocManaged`. In both cases the shim passed copied model/init
  allocations through to real CUDA, accepted KV zero-fill initialization,
  disabled CUDA Graph capture and CUDA PDL launch selection for the shim probe,
  used daemon-published RM backing (`polarisd: RM ALLOC block` / `RM FREE`
  observed), and completed with `uvm_hook_calls` and `uvm_handled` increasing
  and no `uvm_no_pte` or `uvm_errors` increments. This validates the
  no-source-change KV-only llama.cpp path through daemon-published RM backing
  without static RM registration for the strict gate. If a workload enables PDL
  while Polaris allocations are live, the shim now rejects the PDL-specific
  extended launch attributes instead of allowing an unaudited launch ordering.
- Remaining production shim work: replace the bounded managed-window capacity
  model with workload-appropriate VA reclamation/rebalancing for long-running
  server churn, and, if needed for performance, explicitly validate and enable
  CUDA PDL launch behavior with live Polaris allocations.
- Initial llama.cpp/POLARIS KV benchmark harness wired:
  `benchmarks/scripts/run_llama_kv_bench.sh` runs `native_cuda`,
  `polaris_no_pressure`, `polaris_pressure`, and `polaris_sustained_pressure`
  modes over a prompt/generation matrix, captures raw `llama-bench` JSON,
  `/sys/kernel/polaris/stats` before/after snapshots, and derived fault,
  bridge-map, offload, and reload deltas into `runs.jsonl`. It also writes a
  compact `summary.md` table for quick inspection. A local smoke run with
  prompt=32/gen=4 verified native CUDA and POLARIS pressure records, including
  nonzero daemon-backed offload/reload and bridge-map deltas.
- vLLM/SGLang comparison support is currently KV allocator trace-level:
  `benchmarks/scripts/kv_trace_summary.py` consumes the existing trace patch
  CSV format and reports logical reservation, release, peak-live-block, token,
  and session metrics. This is the honest comparison layer until M7 adds live
  POLARIS KV allocator backends for those frameworks.

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
- First module unload/reload stress gate wired and verified:
  `tests/m2/run_module_unload_stress.sh` reloads real `polaris.ko`, starts a
  real RM/UVM fault-capable worker holder through
  `m2_static_block_setup --hold-registered-worker`, verifies normal
  `rmmod polaris` is refused while that worker fd pins the module, then stops
  the worker, verifies `v4_va_spaces=0` and `block_mappings=0`, unloads/reloads
  the module, and runs the daemon-backed RM spill/reload roundtrip on the
  reloaded module. Static RM is not registered in the holder or the final
  daemon-backed regression.
- Tracing for: faults serviced, faults rejected, spills, reloads, bridge
  call latency, policy-mirror sequence drift.
- Bridge map latency telemetry slice wired: `polaris.ko` now times each
  `uvm_polaris_map_external_allocation` attempt and reports
  `uvm_bridge_map_calls`, `uvm_bridge_map_ok`, `uvm_bridge_map_err`,
  `uvm_bridge_map_retry`, `uvm_bridge_map_last_ns`, and
  `uvm_bridge_map_avg_ns` in `/sys/kernel/polaris/stats`. The daemon-backed
  near-capacity soak validates these counters on the production RM-backed path
  with static RM disabled.
- Stress: dynamic KV growth, fragmentation pressure, OOM behavior on
  both VRAM (RM alloc fails) and host pinned pool sides. The host pinned-pool
  side and the RM allocation-failure side now have deterministic daemon-backed
  gates, and the focused near-capacity daemon-backed RM soak covers resident-set
  budget pressure plus async UVM-hook reload/refault behavior. Longer duration
  soak and module-unload stress remain useful before M7.
- Composed daemon-backed RM soak gate wired and verified on 2026-06-15:
  `tests/m2/run_daemon_rm_soak.sh` reloads real `polaris.ko`, runs repeated
  normal-budget daemon-backed single-block spill/reload, multi-block,
  dynamic-fragmentation, overwrite-COW, and observed mapping key isolation gates
  against a real `polarisd` with `POLARISD_RM_BACKING=1`, then reloads the module
  for the CUDA-kernel
  single-worker microbench and near-capacity budget soak with
  `POLARISD_GPU_BUDGET_BYTES=4194304`. Each phase
  requires `uvm_errors` to remain stable, `uvm_bridge_map_calls` to increase,
  and cleanup to drain `sessions`, `blocks`, `pending_decs`, `static_blocks`,
  `block_mappings`, and `v4_va_spaces`. Static RM registration and fake error
  injection are not used. Local validation used the Makefile target with the
  patched NVIDIA tree at `/home/wano/workspace/open-gpu-kernel-modules`. After
  the UVM `DEFERRED` contract change, local validation completed a reduced
  live run with `POLARIS_SOAK_ITERS=1`,
  `POLARIS_SOAK_MICROBENCH_ITERS=1`, and
  `POLARIS_SOAK_NEAR_CAPACITY_ITERS=1`; the earlier pre-microbench default run
  completed `POLARIS_SOAK_ITERS=2` plus
  `POLARIS_SOAK_NEAR_CAPACITY_ITERS=1`. The observed mapping key isolation gate
  was later added to the normal-budget loop and validated in a reduced live run
  with `POLARIS_SOAK_ITERS=1`, `POLARIS_SOAK_MICROBENCH_ITERS=0`, and
  `POLARIS_SOAK_NEAR_CAPACITY_ITERS=0`.

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
