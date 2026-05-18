# POLARIS Roadmap v2

**POLARIS: Paged Operating Layer for Accelerated Routing and Inference Systems**

This roadmap aligns the project with the core assignment requirements:
kernel-level PagedAttention with page-fault-driven GPU memory allocation,
copy-on-write for beam search, and benchmark comparisons against vLLM and
SGLang. Multi-GPU routing is out of scope. eBPF network offload is deferred
to an optional final phase.

---

## Architecture: Patched NVIDIA UVM Fault Hook + POLARIS Kernel Service + CUDA VMM Daemon

```
Inference layer (PyTorch / vLLM / synthetic workload)
      │
      │  CUDA kernel touches an unmapped KV GPU VA
      ▼
NVIDIA GPU MMU
      │
      │  replayable page fault interrupt
      ▼
┌──────────────────────────────┐
│ patched nvidia-uvm.ko        │  ← owns replayable page-fault interrupt,
│ (UVM fault bottom half)      │    parses fault buffer, filters POLARIS VA
└──────────────┬───────────────┘
               │  internal kernel call / exported symbol
      ▼
┌──────────────┐
│  polaris.ko  │  ← kernel service: authoritative block table (page table),
│  (kernel)    │    page-fault decisions, COW refcount, LRU state,
│              │    victim selection, eviction policy, fault wait queues
└──────┬───────┘
       │  ioctl wakeup ("Execute: cuMemMap block 42 on GPU 0 at VA 0x7f...")
       ▼
┌──────────────┐
│   polarisd   │  ← userspace daemon: executes CUDA VMM operations —
│  (userspace) │    cuMemCreate, cuMemMap, cuMemUnmap, cuMemSetAccess,
│              │    cudaMemcpy (GPU↔CPU), cuMemRelease
└──────────────┘
       │
       ▼
┌──────────────┐
│  NVIDIA GPU  │  ← real hardware, managed through CUDA driver
└──────────────┘
```

**Core justification:** GPU hardware page tables are proprietary and
controlled by NVIDIA firmware. POLARIS still cannot edit them directly, but
the open NVIDIA kernel driver exposes the page-fault interrupt path used by
UVM. RM owns the physical interrupt, UVM calls
`nvUvmInterfaceOwnPageFaultIntr(..., NV_TRUE)` to take replayable page-fault
ownership, RM calls UVM's registered `isrTopHalf`, and UVM's bottom half
parses the replayable fault buffer before issuing a GPU replay. POLARIS will
take over this path by patching `nvidia-uvm.ko`, not by polling from the
workload.

CUDA Virtual Memory Management (VMM) remains the OS-equivalent mapping
primitive: `cuMemMap` maps physical GPU memory into a reserved virtual
address range; `cuMemUnmap` removes the mapping. The kernel service is the
**sole decision authority** - it maintains the global block table, tracks
per-block state, and issues all map/unmap/reclaim decisions. The daemon is a
blind executor. The critical change is that the event source is a real GPU
page-fault interrupt in the NVIDIA driver rather than a manual workload
growth ioctl.

**Driver takeover finding:** A standalone third-party module cannot cleanly
preempt UVM after it has claimed replayable page faults. The practical path
is to build a patched `nvidia-uvm.ko` from
`/home/wano/workspace/os/open-gpu-kernel-modules` and add a narrow POLARIS
hook in the UVM replayable fault service path. Non-POLARIS faults must fall
through to NVIDIA's existing UVM handler unchanged.

### NVIDIA UVM Fault Path To Hook

The open driver shows the control points POLARIS needs:

- UVM registers its interrupt callbacks in
  `kernel-open/nvidia-uvm/uvm_global.c` by assigning
  `g_exported_uvm_events.isrTopHalf = uvm_isr_top_half_entry` and passing the
  callback table to `nvUvmInterfaceRegisterUvmEvents`.
- UVM takes replayable page-fault interrupt ownership in
  `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c` by calling
  `nvUvmInterfaceOwnPageFaultIntr(parent_gpu->rm_device, NV_TRUE)`.
- The RM-facing wrapper is in `kernel-open/nvidia/nv_uvm_interface.c`;
  it exports `nvUvmInterfaceOwnPageFaultIntr`,
  `nvUvmInterfaceInitFaultInfo`, `nvUvmInterfaceFlushReplayableFaultBuffer`,
  `nvUvmInterfaceHasPendingNonReplayableFaults`, and
  `nvUvmInterfaceGetNonReplayableFaults`.
- The RM implementation in
  `src/nvidia/arch/nvalloc/unix/src/rm-gpu-ops.c` calls
  `nvGpuOpsOwnPageFaultIntr(device, bOwnInterrupts)`. Its comments state
  that when UVM owns replayable faults, RM will not service, enable, or
  disable that interrupt.
- The interrupt top half is `uvm_isr_top_half_entry` in
  `kernel-open/nvidia-uvm/uvm_gpu_isr.c`. It schedules
  `replayable_faults_isr_bottom_half_entry` when replayable faults are
  pending.
- The replayable bottom half calls
  `uvm_parent_gpu_service_replayable_faults(parent_gpu)`, which fetches fault
  buffer entries, services a batch, and issues replay through UVM's existing
  replay machinery.
- Fault packets are parsed into `uvm_fault_buffer_entry_t`; the parsed entry
  exposes `fault_address`, `fault_access_type`, `is_virtual`, `instance_ptr`,
  and `fault_source`. POLARIS should filter on a registered GPU VA range and
  let all other entries follow the stock UVM path.

**Hook strategy:** add a small POLARIS dispatcher immediately after
replayable fault entries are parsed and before UVM attempts normal managed
memory servicing. If `fault_address` is outside a POLARIS registered VA
range, return `NOT_MINE` and preserve stock behavior. If it is inside a
POLARIS range, the dispatcher:

1. Coalesces faults to the KV block granularity.
2. Calls into `polaris.ko` with `(gpu_uuid, fault_address, access_type,
   instance_ptr, fault_source)`.
3. Queues the required decision (`ALLOC`, `RELOAD`, `COW_BREAK`, or
   `MAP_EXISTING`) and wakes `polarisd`.
4. Waits for the daemon to complete the VMM operation or time out.
5. Returns success to the UVM fault loop so UVM can update the fault buffer
   GET pointer, clear/rearm the interrupt, and replay the faulting work.

The hook must never perform CUDA VMM operations in interrupt context. All
CUDA calls stay in `polarisd`; the UVM bottom half only blocks on a bounded
kernel wait queue while the GPU channel is already fault-stalled.

### Why a kernel module is necessary

| Concern | vLLM / SGLang | POLARIS |
|---------|---------------|---------|
| Global visibility across processes | None — each process sees only itself | Kernel service sees all sessions, all GPUs, all blocks |
| State persistence | Lost on process crash | Kernel structures survive, allowing graceful reclaim |
| OS integration | Isolated Python process | Standard interfaces: patched `nvidia-uvm.ko`, `/dev/polaris`, `/sys/kernel/polaris/stats`, ioctls |
| COW refcount authority | None (Python dict) | Atomic kernel-side refcount, crash-safe |
| Policy pluggability | Hard-coded in Python | Kernel module exposes tunable policies, switchable at runtime |

### Motivation: There Is No Universal KV Cache Interface

Current inference systems handle KV Cache management entirely within the
application process:

```
                            vLLM                                        SGLang
                              │                                            │
                              │  self.block_manager.allocate()             │  self.radix_cache.insert()
                              ▼                                            ▼
 ┌───────────────────────────────────────┐    ┌──────────────────────────────────────┐
 │  vLLM (Python inference service)      │    │  SGLang (Python inference service)   │
 │  ┌─────────────────────────────────┐  │    │  ┌────────────────────────────────┐  │
 │  │ BlockSpaceManager               │  │    │  │ RadixCache                     │  │
 │  │  .allocate(seq, n)  │ cudaMalloc│  │    │  .insert(key, value) │ cudaMalloc │  │
 │  │  .free(seq)                     │  │    │  │  .match_prefix(key)            │  │
 │  │  .fork(parent, child)  ← beam   │  │    │  │  .evict()                      │  │
 │  │  .swap_out(seq)  │ cudaMemcpy   │  │    │  │                                │  │
 │  │  .swap_in(seq)                  │  │    │  └────────────────────────────────┘  │
 │  └─────────────────────────────────┘  │    └──────────────────────────────────────┘
 └────────────────────┬──────────────────┘
                      │  torch.matmul(Q, K^T)
                      ▼
 ┌───────────────────────────────────────┐
 │  PyTorch (tensor computation library) │
 │  Doesn't know what KV Cache is.       │
 │  Doesn't know about blocks, sessions, │
 │  or scheduling. Just does the math.   │
 │  ┌────────────────────────────────┐   │
 │  │ CUDA / cuBLAS                  │   │
 │  └────────────────────────────────┘   │
 └───────────────────────────────────────┘
```

vLLM's `BlockSpaceManager` and SGLang's `RadixCache` are **internal Python
classes**, not external APIs. PyTorch sits below them — it provides tensor
operations but has no concept of KV Cache blocks, sessions, or scheduling.
Neither vLLM nor SGLang was designed to expose a reusable OS-level interface
for KV Cache management. Each framework reimplements the same functionality
inside its own process, in its own way, with no portability.

POLARIS fills this gap: **it elevates KV Cache management from an in-process
Python class to an OS-level service with a universal ioctl protocol.** Any
inference framework can link against the same protocol and immediately gain
paged memory, COW sharing, and automatic offload, without writing its own
block manager.

```
          Before POLARIS                          With POLARIS
 ════════════════════════════           ═══════════════════════════

   vLLM              SGLang                vLLM         SGLang       Your framework
    │                  │                     │             │             │
    │ self.alloc()     │ self.insert()       │ ioctl()     │ ioctl()     │ ioctl()
    ▼                  ▼                     └──────┬──────┘             │
 BlockSpaceMgr     RadixCache                       ▼                    │
    │                  │                     ┌──────────────┐            │
    │ cudaMalloc       │ cudaMalloc          │  polaris.ko  │ ←──────────┘
    ▼                  ▼                     │   (kernel)   │
  GPU mem            GPU mem                 └──────┬───────┘
                                                    │
                                           cuMemMap / cuMemUnmap
                                                    │
                                                    ▼
                                                GPU memory
```

The ioctl protocol defined in Phase 1b plus the UVM fault hook in Phase 1a
are POLARIS's proposal for this universal interface. No such standard exists
today.

---

## Core Abstractions

### KV Block (GPU Page)

A KV block is the smallest scheduling and paging unit. It is the GPU
equivalent of a physical memory page.

```
bytes_per_token = num_layers × 2 (K+V) × num_kv_heads × head_dim × dtype_bytes
```

- **Default configuration:** 16 tokens per block, FP16 dtype
- **Target model:** LLaMA-2-7B (32 layers, 32 KV heads, head_dim=128)
- **Result:** ~512 KB per token → ~8 MB per block (16 tokens)
- **Block size is configurable** for sensitivity experiments

### Session

A session represents one LLM inference request from creation to completion.
It is the unit of GPU virtual address space reservation and owns a linked
list of KV blocks in token order.

### Page Table (Block Table)

The kernel module maintains the authoritative mapping from
`(session_id, token_range)` to `(gpu_phys_handle, gpu_vaddr, state)`.
This is POLARIS's equivalent of a page table.

---

## Phase 0: Freeze Research Claim & Assumptions

### Claim

> LLM inference KV Cache memory pressure is not just an allocation problem
> but an OS-level paging and sharing problem. POLARIS implements in-kernel
> PagedAttention semantics: page-fault-driven block allocation via CUDA VMM,
> GPU↔CPU offload/reload, and reference-counted copy-on-write for beam
> search. Under constrained GPU memory, POLARIS delivers higher memory
> utilization and sharing efficiency compared to vLLM and SGLang.

### Questions to be answered

- Can CUDA VMM's `cuMemMap`/`cuMemUnmap` be composed into a kernel-directed
  page-fault mechanism driven by NVIDIA UVM replayable-fault interrupts?
- Can a patched `nvidia-uvm.ko` safely intercept only POLARIS KV-cache fault
  addresses while preserving stock UVM handling for all other faults?
- What is the latency from GPU page-fault interrupt to daemon VMM mapping to
  UVM replay?
- How much GPU memory is saved when the kernel module applies phase-aware
  victim selection with CPU offload?
- What is the COW sharing efficiency for beam search compared to full
  prefix duplication?
- What is the overhead (ioctl latency, daemon dispatch latency) introduced
  by the kernel-user split architecture?
- How does POLARIS compare to vLLM's PagedAttention block allocator and
  SGLang's RadixAttention under identical memory budgets?

### Assumptions

| Parameter | Value |
|-----------|-------|
| Model | LLaMA-2-7B (32 layers, 32 KV heads, head_dim=128, FP16) |
| KV bytes per token | 524,288 (~512 KB) |
| Block size | 16 tokens (~8 MB) |
| GPU | 1× NVIDIA GPU, CUDA 12.x+, VMM capable (Turing+) |
| OS | Linux with loadable kernel module support |

### Metrics

- Peak GPU memory usage (validated against `nvidia-smi`)
- External fragmentation: ratio of allocated bytes to useful bytes
- Internal fragmentation: unused token slots within allocated blocks
- COW sharing: shared GPU bytes vs. private GPU bytes
- Page-fault count, offload count, reload count
- Driver-fault count, interrupt-to-map latency, map-to-replay latency
- Offload latency, reload latency
- End-to-end throughput: tokens per second

---

## Phase 1: Kernel Module + CUDA VMM Single-GPU PagedAttention

This is the **core contribution** of the project. Everything else depends
on this working correctly.

### Phase 1a: NVIDIA UVM Fault Hook Spike

**Goal:** Replace manual workload notifications with real GPU page-fault
interrupts from the NVIDIA driver.

- Build a patched `nvidia-uvm.ko` from the open driver tree and load it with
  the matching open NVIDIA kernel modules.
- Add a small POLARIS hook in the replayable fault service path after fault
  entries are parsed into `uvm_fault_buffer_entry_t`.
- Register POLARIS GPU VA ranges in a lookup table keyed by GPU UUID and VA
  range.
- For each replayable fault:
  - If `fault_address` is not in a POLARIS range, return immediately to the
    stock UVM handler.
  - If it is in a POLARIS range, coalesce to the KV block, call
    `polaris_resolve_gpu_fault()`, and wait for completion.
  - On success, let UVM replay the faulting work.
  - On timeout or daemon failure, cancel or fail the fault using the existing
    UVM fatal-fault path rather than spinning forever.
- Preserve non-replayable fault handling. UVM/RM split ownership of
  non-replayable faults through a shadow buffer; POLARIS should not use that
  path for the primary KV-cache demand-paging mechanism.

**Validation experiments:**

- Trigger a deliberate unmapped access inside a registered POLARIS VA range
  and verify the patched UVM hook sees the `fault_address`.
- Verify a normal Unified Memory workload still follows the stock UVM handler
  and passes before/after checks.
- Measure interrupt-to-hook latency and hook-to-replay latency with a no-op
  handler before enabling daemon VMM operations.

**Success criterion:**

- A CUDA kernel touching an unmapped POLARIS KV-cache VA stalls, enters the
  patched UVM replayable-fault path, is mapped by `polarisd`, and then resumes
  after UVM replay without a manual workload-triggered ioctl.

### Phase 1b: Kernel Module Skeleton

**Goal:** Loadable kernel module with session and block metadata management,
plus the complete decision protocol interface between kernel and daemon.

**Implement:**

- Character device `/dev/polaris` with `open`, `release`, `unlocked_ioctl`
- Kernel-facing API used by patched `nvidia-uvm.ko`:
  `polaris_register_fault_range()` and `polaris_resolve_gpu_fault()`
- `sysfs` stats exposure (`/sys/kernel/polaris/stats`)
- GPU registration structure
- Session table (create, lookup, destroy)
- Block table — the authoritative page table
- Monotonic block IDs, per-GPU memory counters, per-block state machine

**Key kernel data structures:**

```c
enum polaris_block_state {
    POLARIS_BLOCK_PHYS_FREE_PENDING, // daemon is releasing the phys handle
    POLARIS_BLOCK_RESIDENT,         // mapped to GPU, accessible
    POLARIS_BLOCK_ALLOC_PENDING,    // daemon is creating + mapping
    POLARIS_BLOCK_UNMAPPED,         // unmapped from GPU VA, phys handle exists
    POLARIS_BLOCK_RECLAIMABLE,      // logical release done, waiting for safe physical reclaim
    POLARIS_BLOCK_CPU_OFFLOADED,    // unmapped, data on CPU pinned memory
    POLARIS_BLOCK_OFFLOAD_PENDING,  // daemon is copying GPU→CPU
    POLARIS_BLOCK_RELOAD_PENDING,   // daemon is allocating + copying CPU→GPU
    POLARIS_BLOCK_COW_PENDING,      // daemon is allocating + copying for COW break
    POLARIS_BLOCK_EVICTED,          // no storage, must recompute
};

#define POLARIS_BLOCK_FLAG_SHARED   (1 << 0)  // refcount > 1, COW shared

#define POLARIS_PHASE_PREFILL       1
#define POLARIS_PHASE_DECODE        2

struct polaris_block {
    u64 block_id;
    u64 session_id;
    u32 token_start, token_count;
    u32 home_gpu;
    u64 gpu_vaddr;           // session-local GPU VA from registered POLARIS pool
    u64 gpu_phys_handle;     // opaque CUDA VMM allocation handle
    u64 cpu_buf_addr;        // CPU pinned buffer address (valid if CPU_OFFLOADED)
    u64 size_bytes;
    atomic_t refcount;       // COW: how many sessions share this physical block
    enum polaris_block_state state;
    u32 flags;               // POLARIS_BLOCK_FLAG_*
    u32 phase;               // POLARIS_PHASE_PREFILL or POLARIS_PHASE_DECODE
    u64 last_touch_ns;       // for LRU
    u64 map_time_ns;         // when this block was last mapped
    u64 fault_count;         // number of driver interrupts resolved for this block
};

struct polaris_session {
    u64 session_id;
    u32 home_gpu;
    u64 gpu_vas_base;        // session-local subrange inside daemon-reserved VA pool
    u64 gpu_vas_size;        // total VA metadata budget for this session
    u32 beam_width;
    u64 parent_session_id;   // for COW: 0 if root
    struct list_head blocks; // linked list in token order
};

struct polaris_gpu {
    u32 gpu_id;
    u8 gpu_uuid[16];           // NVIDIA UUID used by UVM/RM fault callbacks
    u64 total_bytes;
    u64 used_bytes;
    u64 budget_bytes;
    u64 pressure_score;
    u64 cpu_pool_total_bytes;  // total CPU pinned memory for offload
    u64 cpu_pool_used_bytes;   // currently in-use CPU offload buffer space
};

struct polaris_fault {
    u64 fault_id;
    u64 generation;          // prevents stale daemon completions after timeout/reuse
    u32 gpu_id;
    u64 fault_address;       // parsed from uvm_fault_buffer_entry_t
    u64 block_id;            // resolved by POLARIS VA range lookup
    u32 access_type;         // read/write/atomic/prefetch
    u32 state;               // QUEUED, WAITING_DAEMON, RESOLVED, FAILED
    u64 enqueue_ns;
    u64 deadline_ns;         // bounded wait in UVM bottom half
    u64 resolved_ns;
};
```

**Required ioctls, grouped by owner:**

```
/* daemon/admin-facing */
POLARIS_DAEMON_ATTACH            // daemon announces executor availability
POLARIS_DAEMON_HEARTBEAT         // daemon liveness and generation tracking
POLARIS_REGISTER_GPU             // daemon reports GPU capacity, CPU pool size
POLARIS_REGISTER_VA_RANGE        // daemon-only: register CUDA-reserved KV VA range
POLARIS_GET_DECISION             // blocking wait: "what should I do next?"
POLARIS_COMPLETE_OPERATION       // daemon reports: "I did the thing" (with result code)
POLARIS_SET_POLICY               // admin/debug: switch eviction policy
POLARIS_GET_GLOBAL_STATS

/* workload/framework-facing */
POLARIS_SESSION_CREATE           // metadata-only session creation
POLARIS_SESSION_DESTROY          // release all logical blocks and VA bindings
POLARIS_SESSION_GET_STATS
POLARIS_SESSION_BRANCH           // COW: fork session, share physical handles
POLARIS_BLOCK_RESERVE            // logical token range + session-local GPU VA
POLARIS_BLOCK_RELEASE            // logical release; physical reclaim is async/safe
POLARIS_BLOCK_TOUCH              // advisory access timestamp for LRU
POLARIS_BLOCK_GET_STATE          // query a block's location and mapping

/* debug/test only */
// no legacy manual-growth ioctl; tests use BLOCK_RESERVE plus the fault path
```

### Kernel ↔ Daemon Decision Protocol

The kernel module queues decisions that the daemon executes. This protocol
is the critical interface between policy (kernel) and execution (daemon).

**Decision types (opcodes):**

```
POLARIS_DEC_ALLOC       // cuMemCreate + cuMemMap → return phys handle
                        //   (used when a real UVM replayable fault lands on an unmapped block)
POLARIS_DEC_FREE        // cuMemUnmap + cuMemRelease
POLARIS_DEC_MAP         // cuMemMap an existing phys handle into a VA range
                        //   (used for COW sharing: map parent's handle into child's VA)
POLARIS_DEC_UNMAP       // cuMemUnmap but keep phys handle
                        //   (metadata-only detach or post-copy cleanup; never before copy)
POLARIS_DEC_OFFLOAD     // cudaMemcpy GPU VA→CPU + cuMemUnmap after copy
POLARIS_DEC_RELOAD      // cuMemCreate + cuMemMap + cudaMemcpy CPU→GPU VA
POLARIS_DEC_COW_BREAK   // cuMemCreate + cuMemMap + cudaMemcpy old VA→new VA
```

**Wire format:**

```c
#define POLARIS_MAX_DECISIONS_PER_GET  16

struct polaris_decision {
    __u64 decision_id;          // unique, assigned by kernel
    __u64 fault_id;             // 0 for non-fault decisions
    __u64 generation;           // fault/session generation for stale completion rejection
    __u32 op;                   // POLARIS_DEC_*
    __u32 gpu_id;
    __u64 block_id;             // kernel block ID (for traceability)
    __u64 session_id;

    // Inputs (kernel → daemon)
    __u64 src_handle;           // physical backing identity for map/release/debug
    __u64 dst_handle;           // optional existing destination handle
    __u64 src_vaddr;            // CUDA-readable GPU VA for copy/unmap
    __u64 dst_vaddr;            // CUDA-map/write GPU VA for map/reload/COW
    __u64 size_bytes;
    __u64 cpu_addr;             // CPU pinned buffer address (offload/reload)
    __u32 access_flags;         // CUDA VMM access mode for cuMemSetAccess
    __u32 timeout_ms;           // bounded daemon execution budget

    __u64 __reserved[4];
};
```

**Opcode semantics:**

```
ALLOC:
  input:  dst_vaddr, size_bytes, access_flags
  action: cuMemCreate → cuMemMap(dst_vaddr) → cuMemSetAccess
  output: output_handle

MAP:
  input:  src_handle, dst_vaddr, size_bytes, access_flags
  action: cuMemMap(dst_vaddr, src_handle) → cuMemSetAccess

UNMAP:
  input:  src_vaddr, size_bytes
  action: cuMemUnmap(src_vaddr)

FREE:
  input:  src_vaddr if mapped, src_handle, size_bytes
  action: cuMemUnmap(src_vaddr) if still mapped → cuMemRelease(src_handle)

OFFLOAD:
  input:  src_vaddr, cpu_addr, size_bytes
  action: cudaMemcpy(cpu_addr ← src_vaddr), then cuMemUnmap(src_vaddr)
  output: output_cpu_addr

RELOAD:
  input:  dst_vaddr, cpu_addr, size_bytes, access_flags
  action: cuMemCreate → cuMemMap(dst_vaddr) → cuMemSetAccess
          → cudaMemcpy(dst_vaddr ← cpu_addr)
  output: output_handle

COW_BREAK:
  input:  src_vaddr, dst_vaddr, size_bytes, access_flags
  action: cuMemCreate → cuMemMap(dst_vaddr) → cuMemSetAccess
          → cudaMemcpy(dst_vaddr ← src_vaddr)
  output: output_handle
```

CUDA VMM handles identify physical backing allocations. CUDA kernels and
`cudaMemcpy` operate on mapped GPU virtual addresses, so copy-related
decisions must carry explicit source and destination VAs in addition to
handles.

**GET_DECISION argument:**

```c
struct polaris_get_decision_arg {
    __u32 count;                // out: decisions returned
    __u32 __reserved;
    struct polaris_decision decisions[POLARIS_MAX_DECISIONS_PER_GET];
};
```

**COMPLETE_OPERATION argument:**

```c
struct polaris_complete_operation_arg {
    __u64 decision_id;
    __u64 generation;           // must match current decision/fault generation
    __s32 result;               // 0 = success; negative = errno on failure
    __u32 __reserved;

    // Outputs (daemon → kernel, filled only on success)
    __u64 output_handle;        // ALLOC/RELOAD/COW_BREAK: new phys handle
    __u64 output_cpu_addr;      // OFFLOAD: CPU buffer address where data resides

    __u64 __reserved2[2];
};
```

**Error handling contract (G4):** When daemon reports `result != 0` in
`COMPLETE_OPERATION`, the kernel must:

| Error code | Daemon meaning | Kernel response |
|------------|---------------|-----------------|
| `-ENOMEM` | `cuMemCreate` returned OUT_OF_MEMORY | Retry with smaller block, queue eviction/offload, or fail the pending fault/fallback ioctl |
| `-ENODEV` | GPU lost / driver error | Mark GPU as unhealthy, reject all sessions on that GPU |
| `-EINVAL` | Invalid parameter (driver rejected) | Kernel log + mark block FAILED + return error to caller |
| `-EFAULT` | Copy failed (page fault on CPU buffer) | Retry once, then evict the block |
| Any other | Unknown failure | Kernel log, mark decision FAILED, continue with next decision |

The kernel maintains a per-decision retry counter to prevent infinite retry
loops. After 3 consecutive failures for the same block, the block is
marked as `POLARIS_BLOCK_EVICTED` and the pending fault or fallback caller
receives an error. Late `COMPLETE_OPERATION` calls whose generation no
longer matches the active decision are ignored.

**Implementation note:** `POLARIS_GET_DECISION` is a blocking ioctl with a
timeout, and `/dev/polaris` should also support `poll`/`epoll`. The daemon
must process every returned decision before waiting for more work, to prevent
starvation. The `(decision_id, generation)` pair ties each
`COMPLETE_OPERATION` to exactly one queued decision.

### ioctl Argument Structs

```c
/* --- GPU registration (daemon → kernel) --- */
struct polaris_daemon_attach_arg {
    __u32 daemon_version;
    __u32 flags;
    __u64 heartbeat_interval_ms;
    __u64 __reserved[4];
};

struct polaris_register_gpu_arg {
    __u32 gpu_id;
    __u8  gpu_uuid[16];          // NVIDIA UUID used by UVM fault hook
    __u64 total_bytes;
    __u64 budget_bytes;          // kernel enforces this limit
    __u64 cpu_pool_bytes;        // daemon's pre-allocated CPU pinned memory size
    __u32 numa_node;
    __u32 __reserved;
    __u64 __reserved2[2];
};

/* --- POLARIS GPU VA range registration (daemon-only → kernel/driver hook) --- */
struct polaris_register_va_range_arg {
    __u64 range_id;               // out/in: kernel-assigned or daemon-supplied pool ID
    __u32 gpu_id;
    __u32 flags;                  // global KV pool, read-only capable, etc.
    __u64 base;                  // base returned by cuMemAddressReserve
    __u64 length;                // reserved KV cache VA span
    __u64 block_size;            // KV block granularity
    __u64 __reserved2[4];
};

/* --- Session create (workload → kernel metadata only) --- */
struct polaris_session_create_arg {
    __u64 session_id;             // out: kernel-assigned
    __u32 home_gpu;               // in: preferred GPU (kernel may override)
    __u32 beam_width;             // in
    __u64 __reserved[4];
};

#define POLARIS_RESERVE_FLAG_OVERWRITE (1 << 0) // allow overlap with shared blocks
#define POLARIS_RESERVE_FLAG_READ_MOSTLY (1 << 1)
#define POLARIS_RESERVE_FLAG_WRITE_NEW   (1 << 2)
#define POLARIS_RESERVE_FLAG_FULL_OVERWRITE_NO_PRESERVE (1 << 3)
#define POLARIS_RELEASE_FLAG_STREAM_QUIESCED (1 << 0)

/* --- Block reserve (workload → kernel metadata, no GPU allocation) --- */
struct polaris_block_reserve_arg {
    __u64 session_id;             // in
    __u32 token_start;            // in
    __u32 token_count;            // in
    __u32 phase;                  // in: PREFILL or DECODE
    __u32 flags;                  // in: POLARIS_RESERVE_FLAG_* and overwrite mode
    __u64 block_id;               // out: kernel-assigned logical block ID
    __u64 gpu_vaddr;              // out: session-local VA to be touched by CUDA kernel
    __u64 __reserved[3];
};

/* --- Block grow test path (workload → kernel) --- */
struct polaris_block_grow_arg {
    __u64 session_id;             // in
    __u32 token_start;            // in
    __u32 token_count;            // in
    __u32 flags;                  // in: debug compatibility with POLARIS_RESERVE_FLAG_*
    __u32 __reserved;
    __u64 block_id;               // out: kernel-assigned block ID
    __s32 ret_code;               // out: 0 = queued, -ENOMEM, -ENODEV
    __u32 __reserved2;
    __u64 __reserved3[2];
};

/* --- Block release (workload → kernel) --- */
struct polaris_block_release_arg {
    __u64 session_id;             // in
    __u32 token_start;            // in
    __u32 token_count;            // in
    __u32 flags;                  // caller guarantees stream quiescence if set
    __u32 __reserved2;
    __u64 __reserved3[4];
};

/* --- Block touch (workload → kernel, updates LRU) --- */
struct polaris_block_touch_arg {
    __u64 session_id;             // in
    __u64 token_start;            // in: start of the range to timestamp
    __u64 token_count;            // in
    __u64 __reserved[4];
};

/* --- Session branch / COW fork (workload → kernel) --- */
struct polaris_session_branch_arg {
    __u64 parent_session_id;      // in
    __u64 child_session_id;       // out: kernel-assigned
    __u64 __reserved[4];
};
```

Production page faults do not enter through a workload growth ioctl. They
enter through the patched UVM fault hook, which calls an internal
`polaris_resolve_gpu_fault()` helper with the parsed `uvm_fault_buffer_entry_t`
fields. Synthetic tests and trace replay use `POLARIS_BLOCK_RESERVE` to create
logical KV ranges; physical mapping is still driven by the fault decision path.

`POLARIS_BLOCK_RESERVE` is the normal allocation-facing API after the
interrupt change. It creates only logical metadata and returns a session-local
GPU VA. It does not allocate physical GPU memory. The first GPU access to
that VA triggers the UVM replayable fault path, and POLARIS then decides
whether to allocate, reload, map an existing shared handle, or COW break.
Overlapping reservations return `-EEXIST` unless `POLARIS_RESERVE_FLAG_OVERWRITE`
is set.

`POLARIS_BLOCK_RELEASE` is a logical release, not an immediate unsafe
`cuMemRelease`. The caller must either guarantee stream quiescence with
`POLARIS_RELEASE_FLAG_STREAM_QUIESCED`, or POLARIS marks the block
reclaimable and delays physical unmap/release until a safe point such as
session teardown or a daemon-observed CUDA event in later integrations.

### Phase 1c: CUDA VMM Backend (Daemon)

**Goal:** Real GPU memory allocation driven by kernel page-fault decisions.

This is the implementation of kernel-level PagedAttention. The key insight:
`cuMemMap`/`cuMemUnmap` at GPU virtual address granularity is semantically
equivalent to OS page fault handling.

**Daemon responsibilities:**

- Discover GPU via CUDA and NVML
- Report GPU memory capacity, free memory to the kernel module
- Reserve a single global GPU VA pool at startup (`cuMemAddressReserve`)
  — all session-local KV VAs are carved from this pool. Sessions receive
  distinct, non-overlapping VA slices for isolation and COW correctness, but
  the daemon avoids per-session CUDA reservation round-trips. GPU VA is vast
  (40+ bits), so a single 64 GiB reservation is safe.
- Register the global KV VA pool with POLARIS through privileged
  `POLARIS_REGISTER_VA_RANGE`; workloads do not register fault ranges
- Execute kernel decisions:
  - `cuMemCreate`: allocate physical GPU memory handle
  - `cuMemMap`: map a physical handle into a sub-range of the global VA pool
  - `cuMemUnmap`: unmap the sub-range — equivalent of page-out
  - `cuMemSetAccess`: set read/write access for a GPU on a VA range
  - `cuMemRelease`: free physical memory handle
- Block in `POLARIS_GET_DECISION` and execute the returned operation list
- Report completion via `POLARIS_COMPLETE_OPERATION`

**Page-fault flow (real driver interrupt):**

```
 CUDA kernel
      │  read/write GPU VA 0x7f... inside session 7 KV range
      ▼
 NVIDIA UVM fault BH          POLARIS kernel service             Daemon
 ┌──────────────────────┐     ┌──────────────────────┐        ┌─────────────────┐
 │ 1. Parse fault entry │     │                      │        │                 │
 │ 2. Match POLARIS VA  │────▶│ 3. Resolve session   │        │                 │
 │    range             │     │    and block         │        │                 │
 │                      │     │ 4. Check GPU budget  │        │                 │
 │                      │     │ 5. Create/mark block │        │                 │
 │                      │     │    ALLOC_PENDING     │        │                 │
 │                      │     │ 6. Queue decision:   │─WAKE──▶│ 7. Read decision│
 │                      │     │    {op=ALLOC,        │        │ 8. cuMemCreate  │
 │                      │     │     fault_id=F,      │        │ 9. cuMemMap     │
 │                      │     │     dst_vaddr=0x7f}  │◀DONE──│10. COMPLETE_OP  │
 │                      │     │11. Update block:     │        │                 │
 │                      │◀────│    state=RESIDENT    │        │                 │
 │12. Advance fault GET │     │                      │        │                 │
 │13. UVM replay        │     │                      │        │                 │
 └──────────────────────┘     └──────────────────────┘        └─────────────────┘
```

When GPU memory budget is exceeded at step 4, the kernel fault resolver does
not return `-ENOMEM` to the workload directly because the workload is stalled
inside a GPU fault. Instead it either queues eviction/offload work (Phase 2)
or fails the fault through the UVM cancel/fatal path so the CUDA work receives
an ordinary CUDA failure.

**Error path (step 8 fails):** If `cuMemCreate` returns OUT_OF_MEMORY,
daemon reports `result=-ENOMEM` via `COMPLETE_OPERATION`. Kernel retries
once (recomputes the decision, possibly selecting a different victim). If
the retry also fails, the block entry is removed and the fault is completed
as failed; the UVM hook cancels/fails the replayable fault rather than
spinning in the bottom half.

**Deliverables:**

- `insmod polaris.ko` works
- `polarisd` daemon compiles and runs
- Real `cuMemCreate`/`cuMemMap`/`cuMemUnmap` path functional
- Per-GPU used/free bytes reported to the kernel and match `nvidia-smi`
- Synthetic KV workload: reserve VA → launch kernel touching unmapped KV
  blocks → driver fault hook maps them → touch/free them

**Success criterion:**

- A workload allocates KV blocks by touching unmapped GPU VAs, `nvidia-smi`
  visibly shows memory consumption rising, and POLARIS stats match the real
  GPU memory usage within a 5% margin.

### Phase 1d: Daemon Lifecycle & Resilience

**Daemon startup:** `polarisd` runs as a systemd service, started before any
workload. On init:

1. Opens `/dev/polaris`
2. Calls `POLARIS_DAEMON_ATTACH`
3. Discovers GPUs via CUDA/NVML
4. Calls `POLARIS_REGISTER_GPU` for each GPU (including CPU pool capacity)
5. Reserves and registers the global KV GPU VA pool with
   `POLARIS_REGISTER_VA_RANGE`
6. Pre-allocates the CPU pinned memory pool (`cudaMallocHost`)
7. Enters the blocking decision loop:
   `POLARIS_GET_DECISION` → execute → `COMPLETE_OPERATION`
8. Sends `POLARIS_DAEMON_HEARTBEAT` periodically so pending faults can fail
   quickly if the executor dies

**Daemon crash recovery:** If the daemon process dies:

- Kernel marks all blocks in `*_PENDING` states as `EVICTED` (the pending
  operations cannot complete)
- All new `POLARIS_GET_DECISION` calls return an empty list (no daemon attached)
- All pending fault waiters fail when their deadline expires or the daemon
  generation changes
- Any unsupported legacy workload ioctl path is rejected with `-ENODEV`
- All new driver faults in POLARIS VA ranges fail through the UVM cancel/fatal
  path after a bounded timeout
- Workloads receive an ordinary CUDA error and may retry after the daemon
  restarts

**Daemon restart:** On restart, the daemon queries `/sys/kernel/polaris/stats`
to reconcile its internal state with the kernel's block table. The daemon
then resumes the decision loop. The kernel begins queuing new decisions.

---

## Phase 2: CPU Offload, Reload, and Eviction Policies

**Goal:** Make POLARIS continue operating when GPU memory is over-subscribed.

### Phase 2a: GPU ↔ CPU Offload/Reload Path

- Daemon pre-allocates a pinned CPU memory pool (`cudaMallocHost`) at init
  time, reports total size via `POLARIS_REGISTER_GPU` → kernel stores in
  `polaris_gpu.cpu_pool_total_bytes`
- Kernel tracks utilization in `polaris_gpu.cpu_pool_used_bytes`
- Kernel selects victim blocks → queues `POLARIS_DEC_OFFLOAD` decision →
  daemon copies GPU→CPU (`cudaMemcpy`) → `cuMemUnmap` → reports
  `output_cpu_addr` → kernel updates block state to `CPU_OFFLOADED` and
  stores `cpu_buf_addr`

**Decode fault handling (reload trigger):** During decode, the attention
kernel reads **every** historical KV block. If any historical block is
offloaded, the first access to its unmapped GPU VA triggers the patched UVM
fault hook. The POLARIS fault resolver must map or reload the block before
allowing UVM to replay the faulting work.

```
CUDA decode kernel accesses block 5 for session 7
  → GPU MMU faults because block 5 is unmapped
  → patched UVM fault hook identifies block 5 in POLARIS VA range
  → POLARIS sees block 5 is CPU_OFFLOADED
  → kernel queues RELOAD for block 5
  → daemon maps/copies CPU→GPU and completes the decision
  → POLARIS marks block 5 RESIDENT
  → UVM replays the faulting work
```

**Reload execution:** `POLARIS_DEC_RELOAD` → daemon `cuMemCreate` →
`cuMemMap` → `cudaMemcpy` CPU→GPU → reports `output_handle` →
kernel updates state to `RESIDENT`.

**BLOCK_TOUCH semantics:** `BLOCK_TOUCH` updates `last_touch_ns` for LRU
**only**. It does NOT trigger reload. Reloads are triggered by real GPU page
faults, plus optional proactive prefetch decisions if the adapter knows the
next decode step will read a range.

**CPU pool exhaustion:** When `cpu_pool_used_bytes + victim_block_size >
cpu_pool_total_bytes`, offload is not possible. The kernel must either:
1. Select a **different** victim that fits in the remaining CPU pool, or
2. **Evict** the current victim: mark state `EVICTED`, release phys
   handle. Data is lost and must be recomputed if needed later.

The phase-aware policy should account for CPU pool pressure in its victim
scoring: prefer evicting prefill blocks over offloading when CPU pool is
nearly full.

### Phase 2b: Eviction Policies

Implement three policies, selectable via module parameter or ioctl:

**FIFO (First-In, First-Out):**
- Victim = block with the oldest `map_time_ns`
- Baseline policy, no intelligence

**LRU (Least Recently Used):**
- Victim = block with the oldest `last_touch_ns`
- Updated on every `BLOCK_TOUCH` call from the inference layer

**Phase-Aware (POLARIS-specific):**
- Prefill blocks are preferred victims (needed once, then "cold")
- Recent decode blocks are **protected** (needed every decode step)
- Shared blocks (refcount > 1) are **protected** (COW saving is valuable)
- Low-priority sessions' blocks are preferred victims

Victim scoring function:

```
victim_score =
    age_weight       × normalized_age
  + prefill_weight   × is_prefill_block
  + pressure_weight  × gpu_pressure
  + priority_weight  × inverse_session_priority
  - sharing_weight   × refcount
  - decode_weight    × recent_decode_access
```

**Deliverables:**
- Real `cudaMemcpy` GPU↔CPU path
- Real reload path: offloaded block accessed → transparently remapped
- Runtime-switchable policy: `fifo`, `lru`, `phase_aware`
- Per-policy statistics (offload count, reload count, offload latency)

**Success criterion:**
- With a GPU memory budget set to 60% of total demand, POLARIS offloads
  and reloads blocks transparently instead of failing allocation. The
  phase-aware policy shows measurably lower offload/reload churn than FIFO.

---

## Phase 3: Copy-on-Write for Beam Search

**Goal:** Efficient prefix sharing for branching decode workloads.

**Key insight:** In standard beam search, each branch only **appends** new
tokens after the fork point — they never write into the shared prefix
blocks. This means the COW **break** rarely triggers in normal operation.
The memory benefit comes from **refcount-based sharing**: child sessions do
not duplicate the parent's prompt blocks. COW break is a safety net for
advanced use cases (speculative decoding, tree attention, prompt editing)
where a branch may deliberately overwrite a shared block.

### Implementation

**Branching (POLARIS_SESSION_BRANCH):**

```
parent_session_id=3       →   child_session_id=7
```

1. Child session is created, linked to parent via `parent_session_id`
2. All parent blocks get `refcount` atomically incremented and `flags`
   set to `POLARIS_BLOCK_FLAG_SHARED`
3. Child session's block table entries point to the same physical handles
   as the parent, but use child-owned GPU virtual addresses carved from the
   POLARIS VA pool
4. The daemon maps the shared physical handles into the child's VA slots with
   restrictive access permissions where supported
5. No new GPU memory allocated — only metadata and additional VA mappings

**COW Break trigger:** COW is preferably triggered by the same driver fault
path used for demand paging. Shared prefix blocks are mapped with the most
restrictive access mode available for the target CUDA VMM mapping. If a
branch writes or atomically updates a shared block, the replayable fault entry
arrives with a write/atomic `fault_access_type`; the POLARIS fault resolver
checks the authoritative refcount and queues `COW_BREAK` before replay.

If the CUDA VMM path cannot create a read-only GPU mapping for the tested
driver/GPU combination, COW falls back to the KV block API boundary. In that
fallback, the workload or framework adapter expresses a semantic operation
through ioctl - append, branch, or overwrite - and the kernel decides whether
COW is required from the authoritative block table. The caller must not decide
COW itself and must not inspect refcounts.

In practice:

- **Standard beam search:** child reserves a new decode token range beyond
  the shared prefix. The first GPU write to that new block faults, POLARIS
  allocates a fresh block, and no COW break is needed.
- **Range conflict without overwrite:** if the adapter asks POLARIS to reserve
  a token range that already overlaps an existing block and
  `POLARIS_RESERVE_FLAG_OVERWRITE` is not set, the kernel must reject the ioctl.
  This prevents accidental silent corruption of existing KV contents. The
  preferred error is `-EEXIST` when available in the kernel binding;
  otherwise return `-EINVAL` and log the overlapping
  `(session_id, token_start, token_count)`.
- **Overwrite path:** a real GPU write/atomic fault on a shared protected
  block, or the fallback `POLARIS_RESERVE_FLAG_OVERWRITE` path, requests a write
  to an existing logical block. The flag means "this operation overwrites an
  existing logical block"; it does **not** mean "force COW." The kernel then
  checks the overlapping block:
  - `refcount == 1`: the block is private, so overwrite may proceed in place.
  - `refcount > 1`: the block is shared, so the kernel queues `COW_BREAK`.

This mirrors vLLM's design at a different layer: vLLM performs refcount/COW
checks inside its user-space block manager, while POLARIS performs the same
decision inside the kernel module and delegates the copy/map operation to
`polarisd`.

**COW Break execution (POLARIS_DEC_COW_BREAK):**

```
Session 7 writes token range 0..15 in a shared mapped block
  → GPU write fault enters patched UVM replayable-fault path
  → POLARIS finds block[0..15] has refcount > 1 (shared with parent)
  → kernel queues COW_BREAK decision
  → daemon:
       cuMemCreate (new physical handle)
       cuMemMap (new handle into session 7's private VA slot for tokens 0..15)
       cudaMemcpy (old shared VA → new private VA, copies content)
       cuMemSetAccess (new mapping, read/write)
  → daemon reports completion: result=0, output_handle=<new_phys>
  → kernel: old block refcount--, new block refcount=1
  → kernel: session 7's block[0..15] now points to the new physical handle
  → UVM replays the faulting write
```

The old block remains mapped for all other sessions that still share it.

### Optimisation: Skip Copy On Full-Block Overwrite

The default COW break must copy old contents into the new physical block.
This is required for partial overwrites: if only token 10 inside a 16-token
block is modified, tokens 0..9 and 11..15 must still preserve their old KV
values for the writing session.

The kernel may skip the old→new copy only when the caller explicitly declares
that the operation overwrites the **entire** block and old contents do not
need to be preserved. In that case the kernel can allocate a private block,
decrement the old block's refcount, and let the caller fill the new block
from scratch.

Important safety rule: tracking the session's token cursor is not
sufficient to prove that a block is stale. In normal LLM decode, historical
KV blocks remain semantically live because every decode attention step reads
the prefix. Therefore, skip-copy is a narrow full-overwrite/no-preserve
optimization, not the default beam-search path.

### Tracking

- `shared_gpu_bytes`: total bytes of blocks with `POLARIS_BLOCK_FLAG_SHARED`
- `private_gpu_bytes`: total bytes of blocks without the shared flag
- `cow_break_count`: number of COW breaks triggered
- `cow_copy_bytes`: total bytes copied during COW breaks
- `memory_saved_vs_naive`: `(beam_width - 1) × shared_bytes - cow_copy_bytes`

### Workload

- Phase 1: standard beam search — beam_width=8, verify that child sessions
  allocate zero new blocks for the prompt (all shared), then allocate fresh
  for decode. Measure `shared_gpu_bytes`.
- Phase 2: COW break test - write to a shared protected block and verify the
  driver fault queues `COW_BREAK`. Also test the fallback
  `POLARIS_RESERVE_FLAG_OVERWRITE` path when read-only VMM mappings are not
  available.

**Deliverables:**
- `POLARIS_SESSION_BRANCH` ioctl
- Write/atomic fault handling for shared blocks; fallback
  `POLARIS_RESERVE_FLAG_OVERWRITE` flag for `BLOCK_RESERVE`
- COW break detection, `POLARIS_DEC_COW_BREAK` execution path
- Real GPU copy for COW break
- Sharing and COW statistics

**Success criterion:**
- Beam width 8 workload uses **>70% less real GPU memory** for the prompt
  portion than naive full prefix duplication (8 separate copies of the
  prompt blocks).

---

## Phase 4: Benchmark Suite and vLLM/SGLang Comparison

**Goal:** Quantitatively demonstrate POLARIS's advantages over existing
systems.

### Phase 4a: Synthetic Inference Workload Runtime

Build a Rust-based workload generator that:

- Creates sessions with configurable prompt/output lengths
- Reserves POLARIS GPU VA ranges and allocates real GPU-backed KV blocks by
  launching kernels that touch unmapped KV addresses
- Simulates decode loop by launching a tiny CUDA kernel that reads the
  historical KV addresses and writes the newly appended block; in fallback
  mode only, it calls `BLOCK_RESERVE`/`BLOCK_TOUCH` to exercise the logical
  reservation path without the patched NVIDIA driver
- Supports beam search workload: `SESSION_BRANCH` → COW testing
- Emits CSV traces: timestamp, operation, session_id, block_id, GPU,
  latency

  Note: this format includes `block_id` and `latency` for POLARIS internal
  debugging. The cross-system trace format in Phase 4b uses a simpler
  schema (`timestamp_ns,op,session_id,token_start,token_count`) for
  replayability across vLLM, SGLang, and POLARIS.

This is **not a real model** - it only exercises the KV cache management
path with real CUDA memory and real NVIDIA replayable page faults. This is
sufficient for the OS-level evaluation.

### Phase 4b: vLLM Trace Replay

#### Trace Collection Methodology

**What to collect:** A CSV file of every KV Cache block reservation and
release event during a vLLM inference run.

**Trace format (`trace.csv`):**

```csv
timestamp_ns,op,session_id,token_start,token_count
123456789,SESSION_CREATE,1,0,0
124000000,BLOCK_RESERVE,1,0,16
124010000,BLOCK_RESERVE,1,16,16
124020000,BLOCK_RESERVE,1,32,16
...
145000000,BLOCK_RESERVE,1,512,16
145500000,BLOCK_RESERVE,1,528,16
146000000,SESSION_BRANCH,1,0,2
146010000,BLOCK_RESERVE,2,544,16
...
220000000,SESSION_DESTROY,1,0,0
220000001,SESSION_DESTROY,2,0,0
```

**Opcodes:** `SESSION_CREATE`, `BLOCK_RESERVE`, `BLOCK_RELEASE`,
`SESSION_BRANCH` (parent→child), `SESSION_DESTROY`.

**How to collect (monkey-patch vLLM, ~20 lines):**

Insert a logging call at the entry of `BlockSpaceManager.allocate()` and
`BlockSpaceManager.free()`. vLLM's `BlockSpaceManager` is a single Python
class in `vllm/core/block_manager.py` — all GPU memory allocation and
deallocation flows through it. No eBPF, no kernel tracing needed.

```python
# vllm/core/block_manager_v1.py  (or v2 for newer vLLM)
import time

# Note: vLLM 0.6+ uses BlockSpaceManagerV2 with prefix caching.
# The monkey-patch logic is identical — only the class name changes.
# For v2: vllm/core/block_manager_v2.py, class BlockSpaceManagerV2.
TRACE_FD = open("/tmp/vllm_trace.csv", "w")
TRACE_FD.write("timestamp_ns,op,session_id,token_start,token_count\n")

class BlockSpaceManager:
    def allocate(self, seq_group: SequenceGroup) -> None:
        # ---- ADD THIS BLOCK ----
        ts = time.time_ns()
        seq = seq_group.get_seqs()[0]
        start = seq.get_len()
        count = self.block_size  # tokens per block
        TRACE_FD.write(f"{ts},BLOCK_RESERVE,{seq_group.request_id},{start},{count}\n")
        # ---- END ADD ----
        # ... original logic unchanged ...

    def free(self, seq: Sequence) -> None:
        ts = time.time_ns()
        TRACE_FD.write(f"{ts},BLOCK_RELEASE,{seq.seq_id},0,0\n")
        # ... original logic unchanged ...

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        ts = time.time_ns()
        TRACE_FD.write(f"{ts},SESSION_BRANCH,{parent_seq.seq_id},0,{child_seq.seq_id}\n")
        # ... original logic unchanged ...
```

**Key point:** This is purely additive — zero changes to vLLM's behavior.
The trace file is then consumed by the POLARIS trace replay runner.

#### Trace Replay

1. Run vLLM with LLaMA-2-7B on standard benchmarks (ShareGPT, LongBench)
   with the monkey-patch active → produces `trace.csv`
2. Replay the **exact same trace** through POLARIS: the trace replay runner
   reads each line and calls the corresponding ioctl. GPU memory budget is
   set identically to what vLLM had.
3. During replay, POLARIS records its own metrics (peak memory, offload
   count, etc.)
4. Compare:
   - Peak GPU memory usage
   - External fragmentation (% of allocated bytes not storing useful KV data)
   - Internal fragmentation (% of block capacity wasted, averaged)
   - Number of allocation/free calls

### Phase 4c: SGLang Trace Replay

Same procedure as 4b, applied to SGLang's RadixAttention. SGLang shares
KV Cache prefixes across requests via a radix tree — this is the closest
existing comparison to POLARIS's COW mechanism.

### Phase 4d: Head-to-Head Comparison Matrix

**Methodology: trace replay ensures fair comparison.** vLLM and SGLang are
run first with the monkey-patch to produce traces. POLARIS replays the
identical alloc/free sequence under the identical GPU memory budget. The
only variable is the memory management policy — how blocks are placed,
when they are evicted, and how sharing is handled.

**What we compare:** Not end-to-end QPS (latency depends on model execution,
which POLARIS does not touch), but **memory management efficiency** under a
fixed, realistic allocation workload.

#### Comparison Scenarios

| Scenario | vLLM | SGLang | POLARIS | Key metric |
|----------|------|--------|---------|-------------|
| Single long context (32K prompt) | ✓ | ✓ | ✓ | Peak memory, internal/external fragmentation |
| 100 concurrent short requests | ✓ | ✓ | ✓ | Peak memory under concurrency, fairness of eviction |
| Beam search (width=8, 64 output tokens) | ✓ | ✓ | ✓ | COW sharing %, memory saved vs. naive duplication |
| Memory budget = 50% of demand | ✓ | ✓ | ✓ | Offload/reload churn, sustained throughput |
| Radix tree prefix sharing (SGLang pattern) | — | ✓ | ✓ | COW vs. radix tree sharing efficiency |

#### Per-Scenario Metrics Collection

For each run (vLLM-native, SGLang-native, POLARIS-replay), collect:

| Metric | Source | Validation |
|--------|--------|------------|
| Peak GPU memory (bytes) | Trace + `nvidia-smi` snapshots | Cross-check POLARIS accounting vs. `nvidia-smi` |
| Reserved bytes total | Sum of all `BLOCK_RESERVE` sizes | Must match across all three runs for fairness |
| Useful bytes | `∑ token_count × bytes_per_token` | Same for all three (identical trace) |
| **External fragmentation** | `1 − useful / allocated` | Lower is better |
| **Internal fragmentation** | `1 − useful / (num_blocks × block_capacity_bytes)` | Lower is better |
| Offload count | `POLARIS_OFFLOAD` decisions | Count for POLARIS; vLLM swap count equivalent |
| Reload count | `POLARIS_RELOAD` decisions | Count for POLARIS; vLLM swap-in count equivalent |
| COW break count | `POLARIS_COW_BREAK` decisions | POLARIS only; vLLM has no COW — naive fork cost = `num_blocks × beam_width` |
| Shared GPU bytes | Sum of block sizes with refcount > 1 | POLARIS only; SGLang radix tree cache hit bytes |

#### Deliverables

- Synthetic workload runtime (Rust, calls POLARIS ioctls directly)
- Trace collection scripts for vLLM and SGLang (Python monkey-patches)
- Trace replay runtime (Rust, reads `trace.csv`, calls POLARIS ioctls)
- Automated benchmark script: for each scenario, run original + replay →
  collect metrics → produce CSV and plots (`benchmarks/plots/`)

**Success criterion:**
- Under identical memory budgets and traces, POLARIS shows less
  fragmentation and/or lower peak memory than at least one of
  vLLM/SGLang on 3 out of 5 scenarios.

### Phase 4e: Optional vLLM Integration Adapter

**Goal:** Replace vLLM's `BlockSpaceManager` with POLARIS ioctls so that a
real vLLM instance uses POLARIS for KV Cache management during live
inference.

This is **optional** — the trace replay methodology in 4b/4d already
provides a fair comparison. Integration provides an end-to-end smoke test.

**What to change:** Only `BlockSpaceManager`. vLLM's scheduler, model
runner, and worker are untouched. Five methods are affected:

```python
# vllm/core/block_manager.py

class PolarisBlockManager(BlockSpaceManager):
    def __init__(self, polaris_fd, ...):
        self.fd = polaris_fd  # file descriptor for /dev/polaris

    def allocate(self, seq_group):
        # Reserve logical KV metadata; the CUDA kernel's first touch maps it.
        arg = polaris_block_reserve_arg(
            session_id=seq_group.request_id,
            token_start=seq_group.get_token_count(),
            token_count=self.block_size,
        )
        fcntl.ioctl(self.fd, POLARIS_BLOCK_RESERVE, arg)

    def free(self, seq):
        # Replace cudaFree with POLARIS metadata release.
        fcntl.ioctl(self.fd, POLARIS_BLOCK_RELEASE, ...)

    def fork(self, parent, child):
        # Replace Python copy with kernel-side refcount sharing.
        arg = polaris_session_branch_arg(parent_session_id=parent.id)
        fcntl.ioctl(self.fd, POLARIS_SESSION_BRANCH, arg)

    def swap_out(self, seq):
        # No-op: POLARIS decides offload from pressure and access faults.

    def swap_in(self, seq):
        # No-op: a later GPU access triggers reload through the UVM fault hook.
```

`swap_out` and `swap_in` become no-ops because POLARIS's kernel module
handles offload/reload transparently — the inference layer never sees it.

**Scope of changes:** Approximately 100 lines of Python in a single file.
The rest of vLLM (scheduler, model runner, worker, engine) is unaware of
the change.

**Deliverables:**
- `polaris_block_manager.py` — drop-in replacement for vLLM's BlockSpaceManager
- Smoke test: launch vLLM with POLARIS backend, run 1 short request, verify
  output correctness and `nvidia-smi` memory reflects POLARIS control

**Success criterion:**
- vLLM runs a real inference request end-to-end with POLARIS managing all
  KV Cache blocks.

**Note:** Only attempt after Phase 4a–4d are complete. The trace replay
comparison is the primary evaluation method.

---

## Phase 5: eBPF Network Offload (Optional / Stretch)

**Goal:** Kernel-level ingress path that bypasses socket buffers for
inference request delivery.

This phase is **decoupled** from the core PagedAttention, COW, and
benchmarking work — it is a self-contained add-on.

**Implementation:**

1. Write an XDP eBPF program attached to the NIC ingress path
2. Parse inference request headers (request ID, prompt length, priority)
   from raw packet data
3. Write parsed request metadata into a BPF ring buffer map
4. `polaris.ko` reads from the ring buffer and calls `SESSION_CREATE`
   without the request ever touching userspace socket buffers (zero-copy)

**Deliverables:**
- eBPF XDP program source
- BPF ring buffer communication with `polaris.ko`
- Benchmarked packet-to-session latency comparison:
  socket path vs. eBPF path

**Success criterion:**
- eBPF path shows measurably lower request ingestion latency than the
  standard socket path. Only attempt after Phase 4 is complete.

---

## Implementation Order

```
Week 1–2:   Phase 0     — Freeze design document, hardware setup
Week 3–4:   Phase 1a    — Patched NVIDIA UVM replayable-fault hook spike
Week 5–6:   Phase 1b    — Kernel module skeleton, device, ioctls, decision protocol
Week 7–9:   Phase 1c    — CUDA VMM daemon, real GPU allocation, driver-fault flow
Week 10:    Phase 1d    — Daemon lifecycle, systemd integration, crash recovery
Week 11–12: Phase 2a    — GPU↔CPU offload/reload
Week 13–14: Phase 2b    — FIFO, LRU, phase-aware eviction policies
Week 15–17: Phase 3     — COW for beam search (padded to 3 weeks for refcount debugging)
Week 18–21: Phase 4a–d — Benchmark suite, vLLM/SGLang trace replay and comparison
Week 22:     Phase 4e    — Optional: vLLM integration adapter (if time permits)
Week 23:     Phase 5     — eBPF network offload (if time permits)
Week 24:    Buffer      — Integration testing, final report
```

---

## Repository Structure

```
polaris/
  nvidia-uvm-patch/
    README.md               # files and symbols changed in open-gpu-kernel-modules
    polaris_uvm_hook.c      # UVM replayable fault dispatcher glue
    polaris_uvm_hook.h      # hook declarations shared with patched UVM files
    patches/                # git-format patches against NVIDIA open driver

  kernel/
    polaris.c              # main module
    polaris.h              # shared types
    polaris_ioctl.h        # ioctl command codes and argument structs
    polaris_block.c        # block table management
    polaris_session.c      # session lifecycle
    polaris_policy.c       # eviction/paging policy implementations
    Makefile
    Kbuild

  polarisd/
    Cargo.toml
    src/
      main.rs              # daemon entry point, decision loop
      ioctl.rs             # ioctl wrappers for kernel communication
      cuda_vmm.rs          # cuMemCreate/Map/Unmap/Release wrappers
      nvml.rs              # GPU discovery and telemetry
      gpu.rs               # GPU state tracking
      offload.rs           # GPU↔CPU copy operations
      cow.rs               # COW break execution
      decision.rs          # GET_DECISION parsing and execution
      lifecycle.rs         # systemd integration, crash recovery, state reconciliation

  polarisctl/
    Cargo.toml
    src/
      main.rs              # CLI entry point
      stats.rs             # read and display /sys/kernel/polaris/stats
      session.rs           # session create/destroy commands
      debug.rs             # block table dump, state inspection

  workloads/
    Cargo.toml
    src/
      synthetic_kv.rs      # single-session KV allocation stress
      beam_search.rs       # COW beam search workload
      concurrent.rs        # multi-session concurrent workload
      trace_replay.rs      # vLLM/SGLang trace file replay

  benchmarks/
    configs/
      long_context.toml
      concurrent.toml
      beam_search.toml
      memory_pressure.toml
    scripts/
      collect_vllm_trace.py   # vLLM monkey-patch for trace collection
      collect_sglang_trace.py # SGLang trace collection
      run_all.sh              # full benchmark suite
      plot.py                 # matplotlib plots from CSV results
    results/                  # per-run CSV output

  adapter/
    polaris_block_manager.py    # vLLM BlockSpaceManager replacement (optional)

  ebpf/
    polaris_xdp.c             # XDP eBPF program
    polaris_xdp_loader.c      # loader and map setup

  docs/
    design.md                 # system design document
    ioctl-api.md              # ioctl reference
    evaluation.md             # benchmark methodology and results

  report/
    final-report.md           # final course report
    figures/                  # generated plots
```

---

## MVP Definition

The minimum viable POLARIS must include:

- [x] Loadable Linux kernel module `polaris.ko`
- [x] Patched `nvidia-uvm.ko` hook for replayable GPU page faults in POLARIS
  VA ranges
- [x] Decision protocol: `GET_DECISION` / `COMPLETE_OPERATION` with error handling
- [x] Rust userspace daemon `polarisd` with systemd integration and crash recovery
- [x] CUDA VMM backend: real `cuMemCreate`, `cuMemMap`, `cuMemUnmap`
- [x] Page-fault flow: GPU replayable fault enters patched UVM → POLARIS
  queues daemon decision → daemon maps → UVM replays faulting work
- [x] Block-level logical release: `BLOCK_RELEASE` for safe async reclaim
- [x] Per-GPU memory accounting (GPU + CPU pool), validated against `nvidia-smi`
- [x] Block-based KV Cache abstraction with token-range granularity
- [x] GPU-to-CPU offload and CPU-to-GPU reload, with CPU pool exhaustion handling
- [x] Three eviction policies: FIFO, LRU, phase-aware
- [x] COW for beam search with `POLARIS_SESSION_BRANCH` and `POLARIS_RESERVE_FLAG_OVERWRITE`
- [x] Benchmark results comparing POLARIS to vLLM and SGLang on at least 3 scenarios
  (trace replay is the primary evaluation method; vLLM adapter integration is a
  stretch goal)

---

## Final Report Claims

Avoid:
- "We replaced the entire NVIDIA GPU driver"
- "We directly edited GPU hardware page tables"
- "We implemented full production vLLM/SGLang"
- "We solved distributed multi-node inference"

Claim instead:
- "We implemented POLARIS, a Linux kernel service plus a narrow patch to
  NVIDIA's open UVM module that orchestrates CUDA VMM to provide
  kernel-level PagedAttention for LLM KV Cache management."
- "POLARIS treats KV Cache blocks as OS-paged resources: a real NVIDIA UVM
  replayable page fault on a missing block queues daemon mapping work, then
  the faulting GPU work is replayed."
- "POLARIS implements in-kernel copy-on-write with atomic reference counting
  to enable memory-efficient beam search."
- "POLARIS is evaluated against vLLM and SGLang on real KV Cache allocation
  traces, demonstrating lower fragmentation under constrained GPU memory."

---

## One-Sentence Summary

**POLARIS is a patched NVIDIA UVM fault hook, Linux kernel service, and CUDA
VMM daemon that provides OS-level paged KV Cache management for LLM
inference: real GPU page-fault-triggered allocation, CPU offload/reload, and
reference-counted COW for beam search, validated against vLLM and SGLang on
real GPU hardware.**
