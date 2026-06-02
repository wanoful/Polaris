# NVIDIA API Requirements for POLARIS KV Memory Allocation

Date: 2026-06-02

## Executive conclusion

For the current POLARIS repo to work as a manually controlled KV-cache pager
for LLM inference frameworks, the required NVIDIA API is already public:
CUDA Driver API Virtual Memory Management (VMM), specifically
`cuMemAddressReserve`, `cuMemGetAllocationGranularity`, `cuMemCreate`,
`cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`, `cuMemRelease`, plus host/device
copy APIs for offload and reload. The current implementation is aligned with
that surface. `polaris-runtime/src/cuda_vmm.rs` wraps exactly those calls,
and `polarisd/src/decision.rs` turns kernel decisions into VMM operations.

For the stronger goal of transparent GPU page-fault-driven allocation of
unmapped CUDA VMM addresses, the needed NVIDIA API does not exist in the
public CUDA, NVML, or open kernel-driver surfaces inspected here. NVIDIA would
need to expose a recoverable GPU virtual-address fault registration and replay
API for CUDA VMM ranges, or implement an equivalent GSP/RM/UVM firmware-driver
path. Without that, a CUDA kernel touching an unmapped `cuMemAddressReserve`
hole is not a recoverable POLARIS allocation event; on the observed
610.43.02 stack it becomes an RM/MMU fault (`Xid 31`, `FAULT_PDE`,
`ACCESS_TYPE_VIRT_WRITE`) and the channel reports `cudaErrorIllegalAddress`.

Therefore the correct near-term route is explicit/manual KV residency control
inside each inference framework process. The transparent fault-driven design is
only achievable if NVIDIA adds a new recoverable fault API or changes firmware
and driver behavior so CUDA VMM holes can be registered as serviceable,
replayable virtual-memory faults.

## Sources and version grounding

Online sources consulted:

- NVIDIA open GPU kernel modules repository:
  https://github.com/NVIDIA/open-gpu-kernel-modules
- Latest local and online-referenced NVIDIA open kernel module version:
  `610.43.02`. The local checkout at
  `/home/wano/workspace/open-gpu-kernel-modules` has
  `README.md`, `version.mk`, and `src/common/inc/nvUnixVersion.h` all showing
  `610.43.02`.
- NVIDIA 610.43.02 kernel-open README:
  https://us.download.nvidia.com/XFree86/Linux-x86_64/610.43.02/README/kernel_open.html
- NVIDIA CUDA Driver API, virtual memory management:
  https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
- NVIDIA CUDA Driver API, memory pools:
  https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html

Local source paths inspected:

- `/home/wano/workspace/Polaris`
- `/home/wano/workspace/open-gpu-kernel-modules`

The local NVIDIA checkout is branch `polaris-610.43.02` at commit
`00ba0831`, whose top commit is the POLARIS UVM replayable-fault hook.

## What POLARIS currently needs from NVIDIA

The current repo has two separable designs:

1. Explicit/manual block residency.
2. Transparent fault-driven residency.

The explicit/manual design only needs stable CUDA VMM. That is the path
which should be treated as the production/benchmark path.

The transparent fault-driven design needs an NVIDIA capability not currently
available to users.

### Public API sufficient for manual allocation

The current explicit design requires:

- `cuInit`
- `cuDeviceGet`
- `cuDevicePrimaryCtxRetain`
- `cuCtxPushCurrent` / `cuCtxPopCurrent`
- `cuMemGetAllocationGranularity`
- `cuMemAddressReserve`
- `cuMemAddressFree`
- `cuMemCreate`
- `cuMemMap`
- `cuMemSetAccess`
- `cuMemUnmap`
- `cuMemRelease`
- `cuMemAllocHost` / `cuMemFreeHost`
- `cuMemcpyDtoH`, `cuMemcpyHtoD`, `cuMemcpyDtoD`

`polaris-runtime/src/cuda_vmm.rs` maps directly onto these calls:

- lines 67-87: VMM allocation granularity query
- lines 89-96: VA reservation with `cuMemAddressReserve`
- lines 112-125: physical allocation with `cuMemCreate`
- lines 128-147: `cuMemMap` plus `cuMemSetAccess`
- lines 150-164: unmap/release
- lines 166-220: pinned CPU allocation and GPU/CPU/GPU copies

`polarisd/src/decision.rs` then uses those primitives:

- lines 98-168: `ALLOC` creates physical memory, maps it at a chosen VA,
  sets access, and tracks the handle
- lines 171-210: `FREE` unmaps and releases
- lines 213-228: `MAP_EXISTING` maps a shared physical handle at another VA
- lines 231-250: `UNMAP`
- lines 253 onward: offload/reload/COW are built from the same primitives

This is the correct public-NVIDIA-API architecture for manual allocation in
LLM frameworks: reserve stable virtual KV-cache addresses, map physical GPU
blocks into those addresses when needed, unmap/offload cold blocks, and reload
before kernels touch them.

### API missing for transparent faults

To make the originally envisioned "CUDA kernel touches unmapped POLARIS VA,
kernel resolves fault, runtime maps memory, GPU replays" path robust, NVIDIA
would need to expose at least one of the following:

- A CUDA Driver API registration for replayable CUDA VMM VA ranges:
  `cuMemRegisterFaultHandler`-style semantics, scoped to a CUDA context or
  process VA space.
- A UVM kernel API for external serviceable VA ranges, letting a third-party
  kernel module register a range, receive replayable fault packets, block the
  faulting work, and return "mapped; replay now".
- An RM/GSP firmware service that classifies missing PDE/PTE faults on
  registered CUDA VMM ranges as recoverable replayable faults instead of
  fatal channel errors.

The minimum event payload must include:

- GPU identity.
- Faulting virtual address.
- Access type.
- Faulting GPU VA space or context identity.
- Process/PASID or equivalent attribution.
- Channel/TSG identity if replay or cancellation is channel-scoped.
- A way to distinguish registered external pager ranges from invalid pointers.

The minimum completion operation must allow:

- The owning process or trusted runtime to map backing memory into the faulting
  CUDA context.
- Required MMU/TLB invalidation.
- Safe replay/resume of the faulting work.
- Precise cancellation and error reporting when mapping fails or times out.

No inspected public NVIDIA API provides this for raw CUDA VMM holes today.

## Why the current UVM hook is not enough

The patched local driver inserts POLARIS into UVM replayable-fault servicing:

- `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c:2898` defines
  `uvm_parent_gpu_service_replayable_faults`.
- Lines 2923-2932 fetch and preprocess replayable fault-buffer entries.
- Lines 2941-2958 call `uvm_polaris_filter_replayable_faults`.
- If POLARIS consumed all faults, lines 2950-2956 push a replay and continue.

That hook is structurally reasonable for genuine UVM replayable faults. The
problem is earlier: raw CUDA VMM holes do not reliably enter this path as
serviceable UVM faults.

UVM preprocessing expects an NVIDIA UVM VA-space association before normal
service:

- `uvm_gpu_replayable_faults.c:1110-1150` sorts entries and translates
  `instance_ptr` values into UVM `va_space` objects.
- `uvm_gpu_replayable_faults.c:1938-2035` services only known UVM GMMU-mappable
  VA ranges, HMM-backed CPU ranges, or ATS-serviceable ranges. Otherwise it
  emits fatal notification/cancellation behavior.

The result is exactly what `docs/fault-driven-analysis.md` records: touching
an unmapped raw CUDA VMM address does not queue a POLARIS decision and instead
falls into RM/MMU fault handling.

## RM/PRI path evidence

The relevant RM path in the 610.43.02 open source is
`src/nvidia/src/kernel/gpu/mmu/arch/volta/kern_gmmu_gv100.c`.

`kgmmuServicePriFaults_GV100` is documented at lines 2429-2431 as handling
BAR1/BAR2, physical faults, and faults captured when fault buffers are
disabled. It is not a general user-extensible replayable pager.

Important constraints in that path:

- Lines 2459-2471: BAR1 faults are treated as incorrect mappings and trigger
  fault logging plus `krcBreakpoint`.
- Lines 2473-2484: BAR2 faults are serviced only for BAR2-specific behavior and
  still call `krcBreakpoint`.
- Lines 2486-2498: physical MMU faults are treated as fatal/notifiable faults.
- Lines 2511-2518: when fault buffers are disabled, non-replayable faults are
  serviced by channel RC, and replayable faults snapped in PRI are cancelled
  because RM does not support replaying them from that path.
- Lines 2543-2547: if fault buffers are enabled but the fault reached this PRI
  path, RM prints the fault and triggers `krcBreakpoint`.
- Lines 2558-2562: BAR2 fatal conditions can set the GPU fatal-error property
  and recover all channels.

This is not a viable place to implement transparent POLARIS allocation as an
out-of-tree patch. It can observe or log failures, but it does not expose the
"map backing and replay the original work" contract that POLARIS needs.

## NVIDIA firmware/GSP constraints

The latest open NVIDIA driver source does not mean users can modify the full
GPU memory-management stack.

The 610.43.02 README says the open kernel modules must be used with matching
GSP firmware and user-space NVIDIA driver components from the same driver
release. It also explains that `nvidia.ko` and `nvidia-modeset.ko` historically
have OS-agnostic components, while `nvidia-drm.ko` and `nvidia-uvm.ko` do not.
The same README describes firmware binary images used by Nouveau to load and
communicate with GSP firmware.

Practical consequences:

- Users can patch and build the open kernel modules, including UVM, and can
  modify much of the published RM source.
- Users cannot rewrite or rebuild NVIDIA's GSP firmware from this source tree.
- Users cannot assume that changing open host code can alter every GPU
  microcontroller decision, fault classification, replay mechanism, security
  policy, or channel recovery path.
- The kernel modules must version-match the GSP firmware and user-space driver.
  A custom kernel module against mismatched firmware/userspace is not a valid
  target for this project.

So "achieve it in NVIDIA firmware" means "NVIDIA implements it in GSP/RM/UVM
and exposes a supported interface." It does not mean POLARIS can safely patch a
firmware binary or depend on undocumented microcontroller behavior.

## How NVIDIA could implement the needed firmware/driver path

The clean firmware-backed design would look like this:

1. A process reserves a CUDA VMM VA span for KV cache.
2. The process registers that span as an external pageable CUDA VMM range with
   the driver, including block size, fault policy, and a user-mode completion
   queue or file descriptor.
3. GSP/RM records that missing PDE/PTE faults inside the span are recoverable,
   not ordinary illegal-address faults.
4. On a GPU fault, GSP/RM/UVM emits a replayable external fault packet with
   GPU/context/process/address/access metadata.
5. The NVIDIA driver stalls the faulting channel/TSG in a bounded way.
6. POLARIS or the linked runtime receives the event, chooses an action, and
   maps backing memory using public VMM or an NVIDIA-provided fault-completion
   call.
7. The driver invalidates relevant MMU/TLB state.
8. GSP/RM/UVM replays the faulting work.
9. If completion fails, the driver cancels precisely and reports a recoverable
   CUDA error instead of corrupting unrelated channels.

The key firmware change is classification: a missing PDE/PTE in a registered
external range must be treated like a recoverable replayable fault, not like a
generic illegal GPU VA. The key driver change is completion: third-party policy
must be able to install or request backing and ask NVIDIA to replay.

Security requirements are non-negotiable. NVIDIA would need to enforce that a
handler can only service its own process/context ranges unless privileged, that
it cannot map memory across security domains, and that timeouts cannot hang the
GPU indefinitely.

## Methods and tradeoffs

### Method A: Explicit CUDA VMM pager

This is the recommended path for POLARIS now.

Mechanism:

- Framework reserves a stable KV VA range.
- POLARIS kernel owns logical block table, session state, COW refcounts, and
  eviction policy.
- In-process runtime executes CUDA VMM operations in the owning CUDA context.
- Framework calls POLARIS before kernels so all needed blocks are resident.
- Cold blocks are copied to pinned CPU memory and unmapped/released.

Pros:

- Uses supported public NVIDIA APIs.
- Already matches POLARIS code.
- Works with raw CUDA VMM.
- Does not require GSP firmware changes.
- Keeps stable tensor pointers, which llama.cpp/ggml and similar frameworks
  need.
- Can be benchmarked honestly against vLLM/SGLang allocation policies.

Cons:

- Not transparent page faulting.
- The framework must integrate residency points before GPU kernels.
- Incorrect residency prediction causes explicit errors or illegal access.
- Multi-process global policy is possible, but VMM execution must remain
  process/context-local.

Best fit:

- llama.cpp first, because it owns CUDA buffer allocation.
- vLLM/SGLang via trace replay or deeper integration at their KV/block-manager
  boundaries.

### Method B: Managed memory / UVM / HMM substrate

Mechanism:

- Allocate KV on `cudaMallocManaged`, HMM, or pageable CPU memory exposed to
  the GPU.
- Let NVIDIA UVM generate replayable faults.
- Add POLARIS policy hooks around genuine UVM fault service.

Pros:

- Genuine UVM replayable faults can reach UVM machinery.
- Useful to prove the hook works for real UVM-managed faults.
- Less risk of immediate Xid for supported ranges.

Cons:

- UVM owns VA ranges, residency metadata, migration policy, and page tables.
- `cuMemMap` cannot simply back arbitrary managed-memory VA with POLARIS
  handles.
- HMM/pageable behavior is constrained by Linux VMAs, ATS support, page sizes,
  and UVM's policy model.
- Performance and control are likely worse for KV cache than explicit VMM.
- It does not deliver POLARIS's desired "own the physical KV block allocator"
  semantics without invasive private UVM work.

Best fit:

- Diagnostic or research proof, not the production POLARIS allocator.

### Method C: Patched UVM replayable-fault filter

Mechanism:

- Keep the local patch in `nvidia-uvm.ko`.
- Filter replayable fault batches after UVM parses and translates them.
- Call `polaris_uvm_handle_gpu_fault` for POLARIS ranges.
- On handled faults, let UVM issue replay.

Pros:

- Good fit for real UVM replayable faults.
- Reuses NVIDIA's replay machinery rather than inventing one.
- The hook is narrow and lets non-POLARIS faults fall through.

Cons:

- It only sees faults that already reached UVM replayable-fault service.
- Raw CUDA VMM holes do not currently show up there as serviceable faults.
- The hook has only address/GPU/access today; a correct multi-process design
  needs process/context identity.
- It is an unsupported driver patch.

Best fit:

- Keep as a research branch and diagnostic hook. Do not base current claims on
  transparent raw-CUDA-VMM replay.

### Method D: RM/PRI fault hook

Mechanism:

- Patch lower RM MMU fault handling such as `kgmmuServicePriFaults_*`.
- Detect POLARIS VA faults before channel recovery.

Pros:

- May observe faults that bypass UVM.
- Useful for logging and confirming fault classification.

Cons:

- The path is already fatal or cancellation-oriented.
- Source comments explicitly say RM does not support replaying replayable PRI
  faults when fault buffers are disabled.
- No safe public continuation exists to map and resume arbitrary compute work.
- High risk of channel resets, Xids, and version-specific breakage.

Best fit:

- Diagnostics only.

### Method E: NVIDIA-supported external replayable VMM pager API

Mechanism:

- NVIDIA adds a CUDA/UVM/RM API for external pageable CUDA VMM ranges.
- POLARIS registers ranges and receives recoverable replayable faults.
- Completion maps memory and asks NVIDIA to replay.

Pros:

- This is the correct transparent design.
- Preserves CUDA VMM's stable virtual addressing and explicit backing model.
- Lets NVIDIA enforce security, context ownership, replay safety, and
  timeouts.
- Could support multiple LLM frameworks without private driver patches.

Cons:

- Requires NVIDIA implementation.
- Requires careful ABI design across CUDA user-mode driver, kernel UVM/RM, GSP,
  and possibly hardware fault-buffer behavior.
- Not available today.

Best fit:

- Long-term request to NVIDIA.

### Method F: CUDA memory pools / stream-ordered allocator

Mechanism:

- Use `cuMemPoolCreate`, `cuMemAllocFromPoolAsync`, and related pool APIs.

Pros:

- Supported CUDA allocation API.
- Useful for reducing allocation overhead inside a process.
- Integrates with stream ordering.

Cons:

- Does not provide stable arbitrary VA remapping for POLARIS blocks.
- Does not expose GPU page faults or replay.
- Does not allow POLARIS to own a cross-process logical KV page table.

Best fit:

- Framework-local allocation optimization, not POLARIS's OS-level pager.

## Framework integration implications

### llama.cpp

The current `integrations/llama.cpp/README.md` is directionally right:
llama.cpp is the best first real integration target because it owns its CUDA
buffers and KV cache directly. The important requirement is that all GPU
kernels see stable contiguous addresses for K/V tensors.

Correct integration shape:

- Allocate KV tensors from a POLARIS-reserved CUDA VMM span.
- Keep ggml CUDA buffer typing unchanged where possible.
- Track KV block metadata in POLARIS.
- Before graph execution, explicitly ensure all blocks needed by attention are
  resident.
- After safe points, let POLARIS offload/unmap cold blocks under budget.

Do not rely on unmapped VMM faulting during graph execution.

### vLLM and SGLang

vLLM and SGLang have their own high-level KV allocators. POLARIS can integrate
at those allocator/block-manager boundaries, but it should not expect PyTorch
or CUDA to expose KV semantics.

Practical routes:

- Trace and replay their allocator behavior for policy comparison.
- Add optional adapters where their KV block manager asks POLARIS for block
  residency decisions.
- Use explicit prefetch/reload points before kernels.

Transparent raw-VMM faulting is not currently available to make these adapters
automatic.

## Required POLARIS course correction

The README phrase "kernel-directed page faults via CUDA VMM" is too strong if
it implies working transparent faults on raw CUDA VMM holes. The repo should
describe the current benchmarkable system as:

> an OS-style explicit KV pager using CUDA VMM, with a kernel-owned logical
> block table and in-process CUDA VMM executor; transparent replayable GPU
> faults for raw CUDA VMM are a research item requiring additional NVIDIA
> driver/firmware support.

The existing `docs/fault-driven-analysis.md` already reaches this conclusion.
This report reinforces it with the latest local 610.43.02 open-driver audit.

## Recommendation

1. Treat explicit CUDA VMM paging as the real implementation path.
2. Keep the UVM hook as a diagnostic/research branch only.
3. Do not spend more time trying to make raw `cuMemAddressReserve` holes
   transparently fault through current UVM; the evidence says they bypass the
   serviceable UVM path and become fatal RM/MMU faults.
4. Add process/context identity to POLARIS's driver-side concepts before any
   multi-process claim.
5. For llama.cpp, integrate at safe allocation and graph-execution boundaries:
   reserve stable VA, map/reload needed blocks before use, and offload only
   after streams/graphs are quiesced.
6. If asking NVIDIA for support, ask for an external replayable CUDA VMM pager
   API, not for generic access to GPU page tables.

The short answer is: public CUDA VMM is enough for manual allocation; a new
NVIDIA-supported external replayable VMM fault API is needed for transparent
fault-driven allocation; user-modifiable NVIDIA firmware is not available in
the open driver release, so firmware-level achievement requires NVIDIA.
