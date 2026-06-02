# POLARIS Roadmap v3

Date: 2026-06-02

This roadmap replaces v2. It is intentionally a forward delta, not a history
of completed work. The implementation direction is now Method A:

> Explicit CUDA VMM KV paging with framework-visible safe points, resident
> leases, and a POLARIS scheduler. Raw CUDA VMM GPU page faults are not a
> production path on the current NVIDIA driver stack.

## Current Decision

POLARIS should become an explicit, OS-style KV residency manager:

1. Frameworks reserve stable CUDA virtual addresses for KV cache.
2. Frameworks tell POLARIS which KV blocks the next GPU work may touch.
3. POLARIS maps or reloads those blocks before kernel launch.
4. POLARIS returns a residency lease that pins those blocks.
5. Framework kernels run using pointers covered by the lease.
6. POLARIS releases the lease after a CUDA stream/event proves the work is
   complete.
7. POLARIS may offload, unmap, or release blocks only when no active lease or
   in-flight event protects them.

Do not implement production behavior that depends on a CUDA kernel touching an
unmapped raw CUDA VMM VA and NVIDIA UVM replaying the work. The diagnostic
evidence in `docs/fault-driven-analysis.md` and the API audit in
`docs/nvidia-memory-allocation-api-report.md` show that current raw CUDA VMM
holes become fatal RM/MMU faults, not serviceable UVM replayable faults.

## Audit Result

### Good: Keep And Build On

- CUDA VMM wrappers in `polaris-runtime/src/cuda_vmm.rs`.
- Explicit KV lifecycle in `polaris-runtime`: reserve VA, map block, unmap,
  offload, reload, free.
- Data-preservation smoke test in `polaris-runtime/tests/kv_smoke.c`.
- Kernel ABI source-of-truth pattern in `kernel/polaris_abi.rs`.
- Kernel block states, decision queue, completion reporting, stats, and ioctl
  library.
- Existing offload/reload mechanics that genuinely `cuMemRelease` physical
  VRAM after copying data to pinned CPU memory.
- COW/refcount concepts and COW workload tests.
- FIFO/LRU/phase-aware policy code as a starting victim selector.
- `BLOCK_GET_STATE` as a diagnostic/planning API.
- `docs/fault-driven-analysis.md` and
  `docs/nvidia-memory-allocation-api-report.md`.

### Rewrite

- Replace the public runtime access model `query/map/read` with
  `require_resident -> lease -> launch -> release`.
- Replace "eviction policy" as a standalone victim selector with a residency
  scheduler that plans required, optional, evictable, and offloadable blocks.
- Unify explicit runtime KV allocations with the kernel-owned logical block
  table. The current explicit KV path tracks `ExplicitKvBlock` in the process
  runtime; that is useful for smoke tests but is not enough for POLARIS's
  kernel-authoritative claim.
- Rewrite llama.cpp integration around residency requests before graph/kernel
  execution, not around fault workers or raw unmapped access.
- Rewrite benchmark infrastructure around explicit residency/offload behavior,
  not transparent fault latency.
- Update README and integration docs after the lease API lands so they stop
  claiming production replayable page-fault paging.

### Remove Or Deprecate From The Production Path

- The v2 production architecture based on patched UVM replayable faults.
- `POLARIS_RESERVE_FLAG_DEFER_FAULT` except as a negative diagnostic.
- Any success criterion requiring raw CUDA VMM faults to reach UVM.
- `LLAMA_POLARIS_FAULT_WORKER` as a production integration mode.
- Statements that `polarisd` can map another process's CUDA VMM VA. CUDA VMM
  execution for a live framework must happen in that framework's CUDA context.
- Missing-script references in docs and benchmark plans unless the scripts are
  actually added.
- eBPF and a live vLLM adapter from the core path. They are deferred until the
  lease-based llama.cpp and trace replay flows work.

## Terminology

Use these terms consistently:

| Term | Meaning |
|------|---------|
| Logical allocation | A framework-visible KV allocation with stable CUDA VAs. Does not imply physical VRAM. |
| Block | Fixed-size subdivision of a logical allocation. The block is the scheduling and residency unit. |
| Resident | Physical GPU memory is mapped at the block's VA and access permissions are set. |
| Offloaded | Block data is stored in CPU pinned memory and no physical GPU backing is mapped. |
| Unmapped | The block has logical metadata but no valid current contents and no physical mapping. |
| Lease | Runtime/kernel promise that a set of resident blocks will not be unmapped/offloaded until release conditions are met. |
| Required set | Blocks the next GPU work may touch. Must be resident before launch. |
| Optional set | Blocks useful to prefetch but not required for correctness. |
| Evictable set | Resident blocks with no active lease, no pending operation, and safe stream/event status. |

Do not use "allocated" as a synonym for "safe to access". Use "resident" or
"leased" when describing GPU-accessible memory.

## Correctness Invariants

Implementations must preserve these invariants:

1. A device pointer may be given to framework GPU code only for blocks covered
   by an active residency lease or by an explicitly permanent resident mode.
2. `BLOCK_GET_STATE` is not a correctness primitive. State query is only for
   diagnostics, assertions, and scheduler planning.
3. `require_resident` must be idempotent. Resident blocks become leased;
   unmapped blocks are allocated and mapped; offloaded blocks are reloaded.
4. No block with `lease_count > 0` may be selected for offload, unmap, release,
   COW source destruction, or physical handle reuse.
5. No block protected by a not-yet-completed CUDA event may be reclaimed.
6. CUDA VMM calls for live framework allocations must execute in the owning
   process's CUDA context.
7. A logical block that does not exist is an error. POLARIS may allocate new
   physical backing only for an existing logical block.
8. Read of an unwritten logical block is an error unless the allocation was
   explicitly created with a zero-fill policy.
9. Offload/reload must preserve data for written blocks.
10. The scheduler must enforce a resident physical VRAM budget. VA reservation
    size is not a memory budget.

## Target Architecture

```
llama.cpp / workload / trace replay
        |
        |  require_resident(required_blocks, stream)
        v
polaris-runtime in the same process
        |
        |  ioctl: acquire lease / update block plan
        v
polaris.ko
        |
        |  plan: keep, map, reload, evict, offload
        v
polaris-runtime executes CUDA VMM in owning context
        |
        |  lease returned only after required blocks are safe
        v
framework launches CUDA work
        |
        |  release lease after stream/event completion
        v
POLARIS may reclaim unleased blocks
```

`polarisd` remains useful for monitoring, policy updates, synthetic tests, and
possibly ownerless cleanup. It must not be the mapper for live framework CUDA
VAs unless the allocation was created inside `polarisd` itself.

## API And ABI Work

### Runtime C API

Add a lease-oriented API to `polaris-runtime/include/polaris_runtime.h`.

Initial v3 C ABI:

```c
typedef enum polaris_access_mode {
    POLARIS_ACCESS_READ = 1,
    POLARIS_ACCESS_WRITE = 2,
    POLARIS_ACCESS_READ_WRITE = 3,
} polaris_access_mode_t;

typedef enum polaris_residency_flags {
    POLARIS_RESIDENCY_REQUIRED = 1u << 0,
    POLARIS_RESIDENCY_PREFETCH = 1u << 1,
    POLARIS_RESIDENCY_ZERO_FILL_UNWRITTEN = 1u << 2,
} polaris_residency_flags_t;

typedef struct polaris_residency_range {
    uint64_t va;
    uint64_t offset;
    uint64_t length;
    uint32_t access;
    uint32_t flags;
} polaris_residency_range_t;

typedef struct polaris_residency_request {
    const polaris_residency_range_t *ranges;
    uint32_t range_count;
    uint32_t phase;
    uint32_t priority;
    uint32_t reserved;
    uintptr_t cuda_stream;
} polaris_residency_request_t;

typedef struct polaris_residency_lease {
    uint64_t lease_id;
    uint64_t bytes_resident;
    uint32_t block_count;
    uint32_t reserved;
} polaris_residency_lease_t;

int polaris_runtime_require_resident(
    polaris_runtime_t *runtime,
    const polaris_residency_request_t *request,
    polaris_residency_lease_t *out_lease);

int polaris_runtime_release_lease(
    polaris_runtime_t *runtime,
    uint64_t lease_id);

int polaris_runtime_release_lease_on_stream(
    polaris_runtime_t *runtime,
    uint64_t lease_id,
    uintptr_t cuda_stream);

int polaris_runtime_get_kv_block_state(
    polaris_runtime_t *runtime,
    uint64_t va,
    uint64_t block_index,
    uint32_t *out_state);
```

Rules:

- `require_resident` is the only normal way to prepare blocks for GPU access.
- `get_kv_block_state` is diagnostic and must not be documented as required
  before reads/writes.
- `release_lease` is valid only when the caller knows all dependent GPU work is
  complete.
- `release_lease_on_stream` records or waits on CUDA event state so release is
  delayed until work submitted to the stream is complete.
- `map_kv_block`, `reload_kv_block`, and `offload_kv_block` may remain for
  tests and low-level diagnostics, but production integrations must use leases.

### Kernel ABI

Add kernel-visible ownership, allocation, and lease concepts. Exact struct
layout may change during implementation, but the semantics must be preserved.

Required additions:

```rust
pub struct PolarisRuntimeRegisterArg {
    pub runtime_id: u64,
    pub pid: u32,
    pub gpu_id: u32,
    pub va_base: u64,
    pub va_length: u64,
    pub block_size: u64,
}

pub struct PolarisAllocationCreateArg {
    pub allocation_id: u64,
    pub runtime_id: u64,
    pub session_id: u64,
    pub va: u64,
    pub size: u64,
    pub block_size: u64,
    pub flags: u32,
}

pub struct PolarisLeaseAcquireArg {
    pub lease_id: u64,
    pub runtime_id: u64,
    pub session_id: u64,
    pub range_count: u32,
    pub phase: u32,
    pub priority: u32,
    pub ranges_user_ptr: u64,
}

pub struct PolarisLeaseReleaseArg {
    pub lease_id: u64,
    pub runtime_id: u64,
    pub flags: u32,
}
```

Required internal fields:

```rust
PolarisBlock {
    allocation_id: u64,
    runtime_id: u64,
    block_index: u64,
    lease_count: u32,
    in_flight_count: u32,
    dirty: bool,
    ever_written: bool,
    last_required_ns: u64,
    last_release_ns: u64,
}

PolarisLease {
    lease_id: u64,
    runtime_id: u64,
    session_id: u64,
    block_ids: Vec<u64>,
    access: u32,
    state: Active | ReleasePending | Released,
}
```

The kernel scheduler must treat `lease_count > 0` and `ReleasePending` as hard
eviction barriers.

## State Machine

The v3 state machine is:

```
NoBlock
  -> Unmapped                  allocation creates logical block

Unmapped
  -> AllocPending              require_resident for first write/zero-fill read
  -> Resident                  VMM allocation/map/access complete

Resident
  -> Leased                    lease acquired
  -> OffloadPending            scheduler selects unleased victim
  -> FreePending               allocation/session destroy

Leased
  -> ReleasePending            release_lease_on_stream
  -> Resident                  event complete or immediate safe release

OffloadPending
  -> CpuOffloaded              GPU->CPU copy, unmap, release physical handle
  -> Resident                  offload failed and recovery kept mapping
  -> Evicted                   unrecoverable failure

CpuOffloaded
  -> ReloadPending             require_resident
  -> Resident                  map new physical handle, CPU->GPU copy complete

ReloadPending
  -> Resident
  -> Evicted

FreePending
  -> Evicted/removed
```

`Leased` may be represented as `Resident` plus `lease_count > 0` in code. The
important invariant is that eviction ignores leased and release-pending blocks.

## Scheduler Work

### Scheduler Responsibilities

Implement a resident-set scheduler, not just a victim selector.

Inputs:

- Required ranges for the next GPU operation.
- Optional prefetch ranges.
- Resident VRAM budget.
- CPU offload capacity.
- Block phase: prefill or decode.
- Session/request priority.
- Lease and in-flight event state.
- COW refcount/share state.
- Estimated reload cost.

Outputs:

- Blocks already resident and leased.
- Blocks to allocate and map.
- Blocks to reload from CPU.
- Unleased victim blocks to offload.
- Optional prefetch work if budget allows.
- Admission/backpressure result if required blocks cannot be made resident.

### Built-In Policies

Keep these policies:

- LRU baseline.
- FIFO baseline for debugging.
- Phase-aware policy.

Extend policy scoring with:

- Hard exclusion for leased and release-pending blocks.
- Hard exclusion for blocks with pending operations.
- Protection for high-refcount shared blocks.
- Decode recency protection.
- Priority/deadline protection.
- CPU pool pressure awareness.
- Reload-cost awareness.

The initial scheduler may be synchronous. Async prefetch/offload can be added
after correctness is covered by tests.

### Admission Control

If required ranges cannot be made resident under budget, POLARIS must return a
clean error before kernel launch. It must not allow the framework to launch
against unmapped memory.

Required failure modes:

- `-ENOMEM`: resident budget cannot satisfy required set even after eviction.
- `-ENOSPC`: CPU offload pool cannot hold selected victims.
- `-EAGAIN`: required blocks are pending conflicting work.
- `-EINVAL`: request targets unknown logical allocation or invalid range.

## Runtime Work

### R1: Local Lease Implementation

First implement leases entirely inside `polaris-runtime` for the explicit KV
path. This validates the API before kernel ABI expansion.

Tasks:

- Add `lease_id` allocation and `lease_count` to local explicit KV blocks.
- Implement `require_resident` over local `ExplicitKvBlock` state.
- Make `require_resident` call existing map/reload helpers:
  - `Resident` -> increment lease count.
  - `Unmapped` -> `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, mark resident,
    increment lease count.
  - `Offloaded` -> reload, then increment lease count.
- Add immediate `release_lease`.
- Add stream/event-based release.
- Add `get_kv_block_state` for diagnostics.
- Keep existing low-level map/offload/reload APIs for tests.

Acceptance tests:

- Allocated logical KV starts unmapped.
- `require_resident` maps an unmapped block.
- Second `require_resident` for the same block is a no-op except lease count.
- Offload of leased block fails or is deferred.
- Release permits offload.
- Offloaded block reloads and preserves data.
- Stream release does not permit offload before event completion.

### R2: Kernel-Owned Logical Allocations

Move from process-local explicit KV bookkeeping to kernel-visible logical
allocations.

Tasks:

- Runtime registers itself with the kernel and gets a `runtime_id`.
- Runtime registers each `alloc_kv` as a kernel allocation with block records.
- Kernel block IDs and runtime block indexes must be cross-referenced.
- Kernel global stats must include explicit runtime KV blocks.
- `BLOCK_GET_STATE` must work for explicit runtime allocations.
- Existing session/token block APIs must either map to the same allocation
  model or be clearly marked as legacy/synthetic.

Acceptance tests:

- `polarisctl stats` reflects runtime-created KV blocks.
- `BLOCK_GET_STATE` or replacement query returns correct state for runtime KV
  blocks.
- Destroying a runtime or allocation releases kernel metadata and physical
  mappings safely.

### R3: Kernel Lease ABI

Add kernel-tracked leases after R1 and R2 are stable.

Tasks:

- Add lease acquire/release ioctls.
- Kernel increments block lease counts for all required blocks.
- Scheduler refuses to offload leased blocks.
- Lease release handles immediate and event-delayed release.
- Runtime mirrors kernel lease state and rolls back on partial failures.

Acceptance tests:

- Kernel policy never selects leased blocks as victims.
- Concurrent leases on overlapping ranges are reference-counted.
- Partial failure during acquire releases all blocks already pinned.
- Runtime crash marks leases release-pending or evicted without dangling
  completion pointers.

## llama.cpp Integration

Rewrite llama.cpp integration around leases.

Required behavior:

1. Initialize `polaris-runtime` after CUDA device/context selection.
2. Allocate KV tensors from a POLARIS logical VA allocation.
3. Before prefill/decode graph or kernel launch, compute the KV block ranges
   the operation may read/write.
4. Call `polaris_runtime_require_resident`.
5. Use returned stable pointers for CUDA work.
6. Release the lease on the CUDA stream after the work is enqueued.
7. Allow POLARIS to offload only after release.

Do not use:

- Raw unmapped VA access.
- `LLAMA_POLARIS_FAULT_WORKER`.
- `map_kv_all` except as a baseline/debug mode.

Required integration modes:

- `LLAMA_POLARIS=1`: enable runtime initialization.
- `LLAMA_POLARIS_KV=1`: allocate KV from POLARIS VA.
- `LLAMA_POLARIS_LEASES=1`: use lease API before attention work.
- `LLAMA_POLARIS_PIN_ALL=1`: debug baseline that maps all KV and disables
  offload.

Acceptance tests:

- Short prompt/decode runs with `PIN_ALL`.
- Short prompt/decode runs with leases and a budget equal to full KV demand.
- Constrained budget run forces offload/reload without illegal CUDA access.
- Long-context run shows resident physical bytes below logical KV bytes.
- CUDA graph mode either works with stable leased pointers or is explicitly
  disabled with a clear message.

## polarisd Direction

Keep `polarisd` as:

- Control-plane daemon.
- Stats and health monitor.
- Policy updater.
- Synthetic executor for allocations it owns.
- Compatibility harness for decision-queue tests.

Do not rely on `polarisd` to map live llama.cpp/vLLM/SGLang CUDA VAs. The
in-process runtime owns live framework VMM operations.

Tasks:

- Rename or document decision executor paths as synthetic/diagnostic unless
  they operate on daemon-owned allocations.
- Add warnings if users try to use daemon mapping for a foreign runtime.
- Keep lifecycle reconciliation and sysfs reporting.

## UVM Hook Direction

Keep the patched UVM hook only as research/diagnostic infrastructure.

Tasks:

- Keep `uvm_fault_smoke.cu` as a negative test documenting raw VMM behavior.
- Remove UVM hook from success criteria.
- Do not add new production features depending on `POLARIS_RESERVE_FLAG_DEFER_FAULT`.
- Keep the code buildable if already present, but feature-gate any runtime path
  that starts fault workers.

## Benchmark Work

The benchmark goal changes from "transparent fault latency" to "explicit
residency scheduler effectiveness."

Required metrics:

- Logical KV bytes.
- Resident physical GPU KV bytes.
- Peak physical GPU KV bytes.
- VA reservation bytes.
- Offload count.
- Reload count.
- Offload bytes.
- Reload bytes.
- Lease acquire latency.
- Lease release latency.
- Scheduler decision latency.
- Kernel throughput/tokens per second.
- CUDA illegal access count, expected to be zero.
- COW shared bytes and COW break count.
- CPU pinned pool usage.

Required scenarios:

- Full-resident baseline.
- Budgeted decode with offload/reload.
- Long context.
- Concurrent sessions.
- Beam search/COW.
- Memory pressure with admission control.
- Trace replay for vLLM and SGLang allocator traces.

Tasks:

- Implement `benchmarks/scripts/run_all.sh`.
- Add `benchmarks/scripts/plot.py`.
- Add trace replay if missing.
- Ensure scripts fail loudly when required traces or models are absent.
- Stop referencing missing files unless they are added.

## vLLM And SGLang

Do not start with live adapters.

Priority order:

1. Trace collection.
2. Trace replay through POLARIS scheduler.
3. Comparison report.
4. Optional live adapter after lease semantics are stable.

Live adapter constraints:

- Adapter must call `require_resident` at framework safe points.
- Adapter must not rely on GPU page faults.
- Adapter must not expose raw POLARIS pointers outside a lease lifetime.

## Documentation Cleanup

After R1 lease API lands:

- Update `README.md` to remove production "kernel-directed page faults" claims.
- Update `integrations/llama.cpp/README.md` to describe leases.
- Mark `docs/progress-report.md` as a historical snapshot if retained.
- Keep `docs/fault-driven-analysis.md` and
  `docs/nvidia-memory-allocation-api-report.md` as rationale documents.

Required README claim:

> POLARIS is a Linux kernel module plus in-process CUDA VMM runtime for
> block-granular KV cache residency. It reserves stable GPU virtual addresses,
> maps/reloads required blocks before kernel execution, offloads cold blocks
> under a resident VRAM budget, and uses leases to prevent unsafe reclamation
> while GPU work is in flight.

Forbidden README claim:

> POLARIS transparently handles raw CUDA VMM GPU page faults in production.

## Implementation Order

### Milestone 1: Runtime Lease API

Deliver:

- C header additions.
- Runtime local lease implementation.
- Immediate and stream/event release.
- Diagnostic block state query.
- Tests for leased offload rejection/deferment.

Exit criteria:

- `kv_smoke` or successor proves write -> offload -> reload -> read works
  through `require_resident`.
- A leased block cannot be offloaded until release.

### Milestone 2: Kernel Allocation Registration

Deliver:

- Runtime registration ioctl.
- Logical allocation registration ioctl.
- Kernel block records for runtime-created KV allocations.
- Stats and state query include runtime allocations.

Exit criteria:

- `polarisctl stats` reflects runtime KV blocks and states.
- Destroying a runtime cleans up kernel records.

### Milestone 3: Kernel Lease Tracking

Deliver:

- Kernel lease acquire/release.
- Policy hard-filters for leased/release-pending blocks.
- Rollback on partial acquire failure.

Exit criteria:

- Synthetic pressure test cannot evict leased blocks.
- Concurrent lease tests pass.

### Milestone 4: Scheduler

Deliver:

- Required/optional/evictable planning.
- Budget enforcement.
- Admission/backpressure.
- Phase-aware policy updated for leases and reload cost.

Exit criteria:

- Budgeted workload stays under resident VRAM limit.
- No illegal CUDA access occurs under stress.

### Milestone 5: llama.cpp Leases

Deliver:

- Lease-based KV residency around attention work.
- Pin-all baseline.
- Budgeted offload/reload mode.

Exit criteria:

- llama.cpp runs short and long-context tests without illegal CUDA access.
- Budgeted run shows physical resident bytes below logical KV bytes.

### Milestone 6: Benchmarks

Deliver:

- Runnable benchmark suite.
- CSV output and plots.
- Trace replay for vLLM/SGLang.

Exit criteria:

- One command runs all available scenarios.
- Missing external dependencies fail with clear instructions.
- Report compares POLARIS full-resident, POLARIS budgeted, and trace baselines.

## Non-Goals For v3

- Transparent raw CUDA VMM fault handling.
- Firmware modification.
- Live vLLM adapter before llama.cpp lease mode works.
- eBPF networking.
- Proving compile-time safety for arbitrary CUDA code.
- Cross-process VMM mapping by a daemon into another process's CUDA context.

## Final Success Criteria

POLARIS v3 is successful when:

- Framework code accesses POLARIS KV memory only through active leases or a
  documented pin-all debug mode.
- POLARIS keeps resident physical KV bytes under a configured budget.
- Offload/reload preserves data.
- Leased/in-flight blocks are never reclaimed.
- llama.cpp can run with budgeted POLARIS KV residency without CUDA illegal
  address failures.
- Benchmarks show logical KV capacity can exceed resident physical VRAM while
  maintaining correct inference execution.
