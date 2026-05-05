# POLARIS Roadmap v2

**POLARIS: Paged Operating Layer for Accelerated Routing and Inference Systems**

This roadmap aligns the project with the core assignment requirements:
kernel-level PagedAttention with page-fault-driven GPU memory allocation,
copy-on-write for beam search, and benchmark comparisons against vLLM and
SGLang. Multi-GPU routing is out of scope. eBPF network offload is deferred
to an optional final phase.

---

## Architecture: Kernel Module + CUDA VMM Daemon

```
Inference layer (PyTorch / vLLM / synthetic workload)
      │
      │  ioctl  ("I need a new KV block")
      ▼
┌──────────────┐
│  polaris.ko  │  ← kernel module: authoritative block table (page table),
│  (kernel)    │    page-fault decisions, COW refcount, LRU state,
│              │    victim selection, eviction policy
└──────┬───────┘
       │  ioctl  ("Execute: cuMemMap block 42 on GPU 0 at VA 0x7f...")
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
controlled by NVIDIA firmware. POLARIS cannot directly edit them. Instead,
CUDA Virtual Memory Management (VMM) provides the OS-equivalent primitives:
`cuMemMap` maps physical GPU memory into a reserved virtual address range;
`cuMemUnmap` removes the mapping. The kernel module is the **sole decision
authority** — it maintains the global block table, tracks per-block state,
and issues all map/unmap/reclaim commands. The daemon is a blind executor.
This split is structurally identical to how a CPU OS handles page faults:
the kernel decides, the MMU executes. The difference is only that GPU MMU
manipulation flows through the NVIDIA driver rather than kernel code
directly.

### Why a kernel module is necessary

| Concern | vLLM / SGLang | POLARIS |
|---------|---------------|---------|
| Global visibility across processes | None — each process sees only itself | Kernel module sees all sessions, all GPUs, all blocks |
| State persistence | Lost on process crash | Kernel structures survive, allowing graceful reclaim |
| OS integration | Isolated Python process | Standard interfaces: `/dev/polaris`, `/sys/kernel/polaris/stats`, ioctls |
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

The ioctl protocol defined in Phase 1a is POLARIS's proposal for this
universal interface. No such standard exists today.

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
  page-fault mechanism that behaves like OS paging for KV Cache?
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
- Offload latency, reload latency
- End-to-end throughput: tokens per second

---

## Phase 1: Kernel Module + CUDA VMM Single-GPU PagedAttention

This is the **core contribution** of the project. Everything else depends
on this working correctly.

### Phase 1a: Kernel Module Skeleton

**Goal:** Loadable kernel module with session and block metadata management,
plus the complete decision protocol interface between kernel and daemon.

**Implement:**

- Character device `/dev/polaris` with `open`, `release`, `unlocked_ioctl`
- `sysfs` stats exposure (`/sys/kernel/polaris/stats`)
- GPU registration structure
- Session table (create, lookup, destroy)
- Block table — the authoritative page table
- Monotonic block IDs, per-GPU memory counters, per-block state machine

**Key kernel data structures:**

```c
enum polaris_block_state {
    POLARIS_BLOCK_FREE_PENDING,     // daemon is releasing the phys handle
    POLARIS_BLOCK_RESIDENT,         // mapped to GPU, accessible
    POLARIS_BLOCK_ALLOC_PENDING,    // daemon is creating + mapping
    POLARIS_BLOCK_UNMAPPED,         // unmapped from GPU VA, phys handle exists
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
    u64 gpu_vaddr;           // GPU virtual address (reserved via cuMemAddressReserve)
    u64 gpu_phys_handle;     // opaque CUDA VMM allocation handle
    u64 cpu_buf_addr;        // CPU pinned buffer address (valid if CPU_OFFLOADED)
    u64 size_bytes;
    atomic_t refcount;       // COW: how many sessions share this physical block
    enum polaris_block_state state;
    u32 flags;               // POLARIS_BLOCK_FLAG_*
    u32 phase;               // POLARIS_PHASE_PREFILL or POLARIS_PHASE_DECODE
    u64 last_touch_ns;       // for LRU
    u64 map_time_ns;         // when this block was last mapped
};

struct polaris_session {
    u64 session_id;
    u32 home_gpu;
    u64 gpu_vas_base;        // reserved GPU virtual address space start
    u64 gpu_vas_size;        // total reserved VA space for this session
    u32 beam_width;
    u64 parent_session_id;   // for COW: 0 if root
    struct list_head blocks; // linked list in token order
};

struct polaris_gpu {
    u32 gpu_id;
    u64 total_bytes;
    u64 used_bytes;
    u64 budget_bytes;
    u64 pressure_score;
    u64 cpu_pool_total_bytes;  // total CPU pinned memory for offload
    u64 cpu_pool_used_bytes;   // currently in-use CPU offload buffer space
};
```

**Required ioctls:**

```
POLARIS_REGISTER_GPU            // daemon reports GPU capacity, CPU pool size
POLARIS_SESSION_CREATE           // allocate session, reserve GPU VA space
POLARIS_SESSION_DESTROY          // release all blocks, free VA space
POLARIS_SESSION_GET_STATS
POLARIS_SESSION_BRANCH           // COW: fork session, share all existing blocks
POLARIS_BLOCK_GROW               // page-fault entry: request new block(s)
POLARIS_BLOCK_FREE               // release a specific token range (decrements refcount)
POLARIS_BLOCK_TOUCH              // update access timestamp for LRU
POLARIS_BLOCK_GET_STATE          // query a block's location and mapping
POLARIS_GET_DECISION             // daemon polls: "what should I do next?"
POLARIS_COMPLETE_OPERATION       // daemon reports: "I did the thing" (with result code)
POLARIS_GET_GLOBAL_STATS
```

### Kernel ↔ Daemon Decision Protocol

The kernel module queues decisions that the daemon executes. This protocol
is the critical interface between policy (kernel) and execution (daemon).

**Decision types (opcodes):**

```
POLARIS_DEC_ALLOC       // cuMemCreate + cuMemMap → return phys handle
POLARIS_DEC_FREE        // cuMemUnmap + cuMemRelease
POLARIS_DEC_MAP         // cuMemMap an existing phys handle into a VA range
                        //   (used for COW sharing: map parent's handle into child's VA)
POLARIS_DEC_UNMAP       // cuMemUnmap but keep phys handle
                        //   (used before offload: detach from VA before cudaMemcpy)
POLARIS_DEC_OFFLOAD     // cudaMemcpy GPU→CPU + cuMemUnmap
POLARIS_DEC_RELOAD      // cuMemCreate + cuMemMap + cudaMemcpy CPU→GPU
POLARIS_DEC_COW_BREAK   // cuMemCreate + cuMemMap + cudaMemcpy old→new
```

**Wire format:**

```c
#define POLARIS_MAX_DECISIONS_PER_POLL  16

struct polaris_decision {
    __u64 decision_id;          // unique, assigned by kernel
    __u32 op;                   // POLARIS_DEC_*
    __u32 gpu_id;
    __u64 block_id;             // kernel block ID (for traceability)
    __u64 session_id;

    // Inputs (kernel → daemon)
    __u64 src_handle;           // phys handle to operate on
    __u64 dst_vaddr;            // target GPU VA (for map)
    __u64 size_bytes;
    __u64 cpu_addr;             // CPU pinned buffer address (offload/reload)

    __u64 __reserved[4];
};
```

**GET_DECISION argument:**

```c
struct polaris_get_decision_arg {
    __u32 count;                // out: decisions returned
    __u32 __reserved;
    struct polaris_decision decisions[POLARIS_MAX_DECISIONS_PER_POLL];
};
```

**COMPLETE_OPERATION argument:**

```c
struct polaris_complete_operation_arg {
    __u64 decision_id;
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
| `-ENOMEM` | `cuMemCreate` returned OUT_OF_MEMORY | Retry with smaller block, or fail the original BLOCK_GROW ioctl |
| `-ENODEV` | GPU lost / driver error | Mark GPU as unhealthy, reject all sessions on that GPU |
| `-EINVAL` | Invalid parameter (driver rejected) | Kernel log + mark block FAILED + return error to caller |
| `-EFAULT` | Copy failed (page fault on CPU buffer) | Retry once, then evict the block |
| Any other | Unknown failure | Kernel log, mark decision FAILED, continue with next decision |

The kernel maintains a per-decision retry counter to prevent infinite retry
loops. After 3 consecutive failures for the same block, the block is
marked as `POLARIS_BLOCK_EVICTED` and the original caller receives an
error.

**Implementation note:** The daemon must process every pending decision
before polling `GET_DECISION` again, to prevent starvation. The
`decision_id` ties each `COMPLETE_OPERATION` to exactly one queued decision.

### Workload-Facing ioctl Argument Structs

```c
/* --- GPU registration (daemon → kernel) --- */
struct polaris_register_gpu_arg {
    __u32 gpu_id;
    __u64 total_bytes;
    __u64 budget_bytes;          // kernel enforces this limit
    __u64 cpu_pool_bytes;        // daemon's pre-allocated CPU pinned memory size
    __u32 numa_node;
    __u32 __reserved;
    __u64 __reserved2[2];
};

/* --- Session create (workload → kernel) --- */
struct polaris_session_create_arg {
    __u64 session_id;             // out: kernel-assigned
    __u32 home_gpu;               // in: preferred GPU (kernel may override)
    __u32 beam_width;             // in
    __u64 gpu_vas_bytes;          // in: how much GPU VA to reserve
    __u64 __reserved[4];
};

/* --- Block grow / page-fault (workload → kernel) --- */
#define POLARIS_GROW_FLAG_OVERWRITE  (1 << 0)   // allow overlap with shared blocks

struct polaris_block_grow_arg {
    __u64 session_id;             // in
    __u32 token_start;            // in
    __u32 token_count;            // in
    __u32 flags;                  // in: POLARIS_GROW_FLAG_*
    __u32 __reserved;
    __u64 block_id;               // out: kernel-assigned block ID
    __s32 ret_code;               // out: 0 = queued, -ENOMEM, -ENODEV
    __u32 __reserved2;
    __u64 __reserved3[2];
};

/* --- Block free (workload → kernel) --- */
struct polaris_block_free_arg {
    __u64 session_id;             // in
    __u32 token_start;            // in
    __u32 token_count;            // in
    __u64 __reserved[4];
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

### Phase 1b: CUDA VMM Backend (Daemon)

**Goal:** Real GPU memory allocation driven by kernel page-fault decisions.

This is the implementation of kernel-level PagedAttention. The key insight:
`cuMemMap`/`cuMemUnmap` at GPU virtual address granularity is semantically
equivalent to OS page fault handling.

**Daemon responsibilities:**

- Discover GPU via CUDA and NVML
- Report GPU memory capacity, free memory to the kernel module
- Reserve a single global GPU VA pool at startup (`cuMemAddressReserve`)
  — all blocks across all sessions are mapped into this pool. Per-session
  reservation adds unnecessary daemon round-trips and fragments VA space.
  GPU VA is vast (40+ bits), so a single 64 GiB reservation is safe.
- Execute kernel decisions:
  - `cuMemCreate`: allocate physical GPU memory handle
  - `cuMemMap`: map a physical handle into a sub-range of the global VA pool
  - `cuMemUnmap`: unmap the sub-range — equivalent of page-out
  - `cuMemSetAccess`: set read/write access for a GPU on a VA range
  - `cuMemRelease`: free physical memory handle
- Poll `POLARIS_GET_DECISION` and execute the returned operation list
- Report completion via `POLARIS_COMPLETE_OPERATION`

**Page-fault flow (BLOCK_GROW):**

```
 Inference layer
      │  ioctl(POLARIS_BLOCK_GROW, session=7, tokens=1..16, flags=0)
      ▼
 Kernel module                                Daemon
 ┌──────────────────────┐                  ┌─────────────────┐
 │ 1. Look up session 7 │                  │                 │
 │ 2. Find gap in block │                  │                 │
 │    table at tokens   │                  │                 │
 │    1..16 → page fault│                  │                 │
 │ 3. Check GPU budget  │                  │                 │
 │    → has free space   │                  │                 │
 │ 4. Create block entry │                  │                 │
 │    state=ALLOC_PENDING│                  │                 │
 │ 5. Queue decision:   │───GET_DECISION──▶│ 6. Read decision│
 │    {op=ALLOC,         │                  │ 7. cuMemCreate  │
 │     session=7,        │                  │ 8. cuMemMap     │
 │     gpu=0,            │                  │ 9. Report done  │
 │     size=8MB}         │◀─COMPLETE_OP────│    {result=0,    │
 │                       │                  │     handle=X}   │
 │ 10. Update block:    │                  │                 │
 │     phys_handle=X,   │                  │                 │
 │     state=RESIDENT   │                  │                 │
 │ 11. Return to user   │                  │                 │
 │     ret_code=0        │                  │                 │
 └──────────────────────┘                  └─────────────────┘
```

When GPU memory budget is exceeded at step 3, the kernel returns `-ENOMEM`
to the caller.  Full victim selection, offload, and eviction under memory
pressure is implemented in Phase 2.

**Error path (step 7 fails):** If `cuMemCreate` returns OUT_OF_MEMORY,
daemon reports `result=-ENOMEM` via `COMPLETE_OPERATION`. Kernel retries
once (recomputes the decision, possibly selecting a different victim). If
the retry also fails, the block entry is removed and the `BLOCK_GROW`
ioctl returns `-ENOMEM` to the calling process.

**Deliverables:**

- `insmod polaris.ko` works
- `polarisd` daemon compiles and runs
- Real `cuMemCreate`/`cuMemMap`/`cuMemUnmap` path functional
- Per-GPU used/free bytes reported to the kernel and match `nvidia-smi`
- Synthetic KV workload: allocate blocks → touch them → free them

**Success criterion:**

- A workload allocates KV blocks through `BLOCK_GROW`, `nvidia-smi` visibly
  shows memory consumption rising, and POLARIS stats match the real GPU
  memory usage within a 5% margin.

### Phase 1c: Daemon Lifecycle & Resilience

**Daemon startup:** `polarisd` runs as a systemd service, started before any
workload. On init:

1. Opens `/dev/polaris`
2. Discovers GPUs via CUDA/NVML
3. Calls `POLARIS_REGISTER_GPU` for each GPU (including CPU pool capacity)
4. Pre-allocates the CPU pinned memory pool (`cudaMallocHost`)
5. Enters the decision loop: `POLARIS_GET_DECISION` → execute → `COMPLETE_OPERATION`

**Daemon crash recovery:** If the daemon process dies:

- Kernel marks all blocks in `*_PENDING` states as `EVICTED` (the pending
  operations cannot complete)
- All new `POLARIS_GET_DECISION` calls return an empty list (no daemon attached)
- All new `BLOCK_GROW` calls from workloads immediately return `-ENODEV`
- Workloads receive an error and may retry after the daemon restarts

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

**Pre-decode residency check (reload trigger):** When the inference layer
calls `BLOCK_GROW` during the decode phase, the kernel must ensure all
existing blocks of that session are GPU-resident before returning. The
attention computation reads **every** historical KV block — a single
offloaded block causes a GPU fault.

```
Inference layer calls BLOCK_GROW(tokens=512, count=16) for session 7
  → kernel queues ALLOC for the new block
  → kernel scans session 7's block table: finds block 5 is CPU_OFFLOADED
  → kernel queues RELOAD for block 5
  → kernel waits for both ALLOC and RELOAD to complete (via daemon)
  → kernel returns from BLOCK_GROW
  → inference layer can now safely compute attention over all blocks
```

**Reload execution:** `POLARIS_DEC_RELOAD` → daemon `cuMemCreate` →
`cuMemMap` → `cudaMemcpy` CPU→GPU → reports `output_handle` →
kernel updates state to `RESIDENT`.

**BLOCK_TOUCH semantics:** `BLOCK_TOUCH` updates `last_touch_ns` for LRU
**only**. It does NOT trigger reload. The kernel relies on the `BLOCK_GROW`
handler to reload offloaded blocks proactively before computation.

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
3. Child session's block table entries **point to the same physical
   handles and GPU virtual addresses** as the parent
4. No new GPU memory allocated — only metadata in the kernel

**COW Break trigger:** COW is triggered at the KV block API boundary, not by
GPU hardware write faults. The workload or framework adapter expresses a
semantic operation through ioctl — append, branch, or overwrite — and the
kernel decides whether COW is required from the authoritative block table.
The caller must not decide COW itself and must not inspect refcounts.

In practice:

- **Standard beam search:** child calls `BLOCK_GROW(tokens=512, count=16)`.
  This token range is **beyond** the shared prefix → it's a new block, not
  a modification → no COW break. The child allocates fresh.
- **Range conflict without overwrite:** if `BLOCK_GROW` targets a token range
  that already overlaps an existing block and `POLARIS_GROW_FLAG_OVERWRITE`
  is not set, the kernel must reject the ioctl. This prevents accidental
  silent corruption of existing KV contents. The preferred error is
  `-EEXIST` when available in the kernel binding; otherwise return
  `-EINVAL` and log the overlapping `(session_id, token_start, token_count)`.
- **Overwrite path:** the workload or vLLM/SGLang adapter passes
  `POLARIS_GROW_FLAG_OVERWRITE` to `BLOCK_GROW`, requesting a write to an
  existing token range. This flag means "this operation overwrites an
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
Session 7 calls BLOCK_GROW(tokens=0..15, flags=OVERWRITE)
  → kernel finds block[0..15] has refcount > 1 (shared with parent)
  → kernel queues COW_BREAK decision
  → daemon:
       cuMemCreate (new physical handle)
       cuMemMap (into session 7's VA space for tokens 0..15)
       cudaMemcpy (old_phys → new_phys, copies content)
       cuMemSetAccess (new mapping, read/write)
  → daemon reports completion: result=0, output_handle=<new_phys>
  → kernel: old block refcount--, new block refcount=1
  → kernel: session 7's block[0..15] now points to the new physical handle
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
- Phase 2: COW break test — use `POLARIS_GROW_FLAG_OVERWRITE` on a shared
  block in one branch, verify the copy occurs and the other branches are
  unaffected.

**Deliverables:**
- `POLARIS_SESSION_BRANCH` ioctl
- `POLARIS_GROW_FLAG_OVERWRITE` flag for `BLOCK_GROW`
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
- Allocates real GPU-backed KV blocks via POLARIS ioctls
- Simulates decode loop: calls `BLOCK_GROW` for each new token,
  `BLOCK_TOUCH` for existing blocks
- Supports beam search workload: `SESSION_BRANCH` → COW testing
- Emits CSV traces: timestamp, operation, session_id, block_id, GPU,
  latency

  Note: this format includes `block_id` and `latency` for POLARIS internal
  debugging. The cross-system trace format in Phase 4b uses a simpler
  schema (`timestamp_ns,op,session_id,token_start,token_count`) for
  replayability across vLLM, SGLang, and POLARIS.

This is **not a real model** — it only exercises the KV cache management
path with real CUDA memory. This is sufficient for the OS-level evaluation.

### Phase 4b: vLLM Trace Replay

#### Trace Collection Methodology

**What to collect:** A CSV file of every KV Cache block allocation and
deallocation event during a vLLM inference run.

**Trace format (`trace.csv`):**

```csv
timestamp_ns,op,session_id,token_start,token_count
123456789,SESSION_CREATE,1,0,0
124000000,BLOCK_ALLOC,1,0,16
124010000,BLOCK_ALLOC,1,16,16
124020000,BLOCK_ALLOC,1,32,16
...
145000000,BLOCK_ALLOC,1,512,16
145500000,BLOCK_ALLOC,1,528,16
146000000,SESSION_BRANCH,1,0,2
146010000,BLOCK_ALLOC,2,544,16
...
220000000,SESSION_DESTROY,1,0,0
220000001,SESSION_DESTROY,2,0,0
```

**Opcodes:** `SESSION_CREATE`, `BLOCK_ALLOC`, `BLOCK_FREE`,
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
        TRACE_FD.write(f"{ts},BLOCK_ALLOC,{seq_group.request_id},{start},{count}\n")
        # ---- END ADD ----
        # ... original logic unchanged ...

    def free(self, seq: Sequence) -> None:
        ts = time.time_ns()
        TRACE_FD.write(f"{ts},BLOCK_FREE,{seq.seq_id},0,0\n")
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
| Allocated bytes total | Sum of all `BLOCK_ALLOC` sizes | Must match across all three runs for fairness |
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
        # 替换 cudaMalloc → ioctl(POLARIS_BLOCK_GROW)
        arg = polaris_block_grow_arg(
            session_id=seq_group.request_id,
            token_start=seq_group.get_token_count(),
            token_count=self.block_size,
        )
        fcntl.ioctl(self.fd, POLARIS_BLOCK_GROW, arg)

    def free(self, seq):
        # 替换 cudaFree → 可以 no-op（POLARIS 内核自行回收）

    def fork(self, parent, child):
        # 替换 Python copy → ioctl(POLARIS_SESSION_BRANCH)
        arg = polaris_session_branch_arg(parent_session_id=parent.id)
        fcntl.ioctl(self.fd, POLARIS_SESSION_BRANCH, arg)

    def swap_out(self, seq):
        # 替换 cudaMemcpy → no-op（内核自动决策换出）

    def swap_in(self, seq):
        # 替换 cudaMemcpy → no-op（内核自动决策换入）
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
Week 3–5:   Phase 1a    — Kernel module skeleton, device, ioctls, decision protocol
Week 6–8:   Phase 1b    — CUDA VMM daemon, real GPU allocation, page-fault flow
Week 9:     Phase 1c    — Daemon lifecycle, systemd integration, crash recovery
Week 10–11: Phase 2a    — GPU↔CPU offload/reload
Week 12–13: Phase 2b    — FIFO, LRU, phase-aware eviction policies
Week 14–16: Phase 3     — COW for beam search (padded to 3 weeks for refcount debugging)
Week 17–20: Phase 4a–d — Benchmark suite, vLLM/SGLang trace replay and comparison
Week 21:     Phase 4e    — Optional: vLLM integration adapter (if time permits)
Week 22:     Phase 5     — eBPF network offload (if time permits)
Week 23–24: Buffer      — Integration testing, final report
```

---

## Repository Structure

```
polaris/
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
- [x] Decision protocol: `GET_DECISION` / `COMPLETE_OPERATION` with error handling
- [x] Rust userspace daemon `polarisd` with systemd integration and crash recovery
- [x] CUDA VMM backend: real `cuMemCreate`, `cuMemMap`, `cuMemUnmap`
- [x] Page-fault flow: `BLOCK_GROW` triggers kernel decision → daemon maps → accessible to inference layer
- [x] Block-level free: `BLOCK_FREE` for mid-lifecycle block release
- [x] Per-GPU memory accounting (GPU + CPU pool), validated against `nvidia-smi`
- [x] Block-based KV Cache abstraction with token-range granularity
- [x] GPU-to-CPU offload and CPU-to-GPU reload, with CPU pool exhaustion handling
- [x] Three eviction policies: FIFO, LRU, phase-aware
- [x] COW for beam search with `POLARIS_SESSION_BRANCH` and `POLARIS_GROW_FLAG_OVERWRITE`
- [x] Benchmark results comparing POLARIS to vLLM and SGLang on at least 3 scenarios
  (trace replay is the primary evaluation method; vLLM adapter integration is a
  stretch goal)

---

## Final Report Claims

Avoid:
- "We replaced the NVIDIA GPU driver"
- "We directly edited GPU hardware page tables"
- "We implemented full production vLLM/SGLang"
- "We solved distributed multi-node inference"

Claim instead:
- "We implemented POLARIS, a Linux kernel module that orchestrates CUDA VMM
  to provide kernel-level PagedAttention for LLM KV Cache management."
- "POLARIS treats KV Cache blocks as OS-paged resources: page-fault on
  missing blocks, evict under memory pressure, reload on access."
- "POLARIS implements in-kernel copy-on-write with atomic reference counting
  to enable memory-efficient beam search."
- "POLARIS is evaluated against vLLM and SGLang on real KV Cache allocation
  traces, demonstrating lower fragmentation under constrained GPU memory."

---

## One-Sentence Summary

**POLARIS is a Linux kernel module and CUDA VMM daemon that provides
OS-level paged KV Cache management for LLM inference: kernel-directed page
faults, CPU offload/reload, and reference-counted COW for beam search,
validated against vLLM and SGLang on real GPU hardware.**
