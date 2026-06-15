# libpolaris-shim

LD_PRELOAD shared library that makes unmodified CUDA workers (llama.cpp, vLLM,
SGLang, PyTorch) transparent participants in POLARIS's kernel-directed KV
paging path. v4 architecture — see `../docs/roadmap-v4.md`.

## What it does (eventually)

1. Intercepts `cuInit` / first CUDA driver-API call in the worker process.
2. Creates a fault-capable, externally-owned GPU VA-space via
   `NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING |
    IS_EXTERNALLY_OWNED` and registers it with UVM through
   `UvmRegisterGpuVaSpace`.
3. Hands the same user RM client handle plus user RM VA-space handle
   (`rm_client_token`, `va_space_token` in the v4 ABI) to `polaris.ko` via
   the `POLARIS_REGISTER_VASPACE` ioctl. The pair matters because RM object
   handles are scoped to a client.
4. Intercepts `cuMemAlloc` / `cudaMalloc` / `cudaMallocManaged` and the PyTorch / llama.cpp
   allocator backends to reserve VA inside the POLARIS VA-space — without
   mapping it — and registers the range with `polaris.ko` so the fault
   hook can resolve `addr → block_id`.
5. Intercepts `cuMemFree` / `cudaFree` to deregister the range.

## Current state

The shim resolves the real `cuInit` symbol, forwards to it, and can now create
the fault-capable RM/UVM VA-space itself in an explicit opt-in mode:

    POLARIS_SHIM_BOOTSTRAP_RM_UVM=1 \
    POLARIS_SHIM_GPU_ID=0 \
    POLARIS_SHIM_CUDA_ORDINAL=0 \
    POLARIS_SHIM_BLOCK_SIZE=0x200000 \
    LD_PRELOAD=$(realpath ./libpolaris-shim.so) ./your-cuda-app

In this mode, the shim opens `/dev/nvidiactl`, allocates an RM root client,
device, subdevice, and a `FERMI_VASPACE_A` with
`ENABLE_PAGE_FAULTING | IS_EXTERNALLY_OWNED`, initializes `/dev/nvidia-uvm`,
calls `UVM_REGISTER_GPU` and `UVM_REGISTER_GPU_VASPACE`, registers the same RM
client/VA-space handles with `polaris.ko`, and enables per-allocation
`UVM_CREATE_EXTERNAL_RANGE` / `UVM_FREE` automatically. The managed base and
length default to the RM-reported fault-capable VA-space window. To avoid
registering an unexpectedly huge range during bring-up, the default
bootstrapped managed window is capped at 1 TiB; set
`POLARIS_SHIM_MANAGED_LENGTH_CAP=<bytes>` to change that cap or `0` to use the
full RM-reported length. Set `POLARIS_SHIM_MANAGED_BLOCKS=<count>` to cap the
bootstrapped window by Polaris block count (`count * POLARIS_SHIM_BLOCK_SIZE`),
which is often easier to tune for KV experiments than a raw byte length. The
byte cap and block-count cap are both applied when present, and
`POLARIS_SHIM_MANAGED_BASE` and
`POLARIS_SHIM_MANAGED_LENGTH` still override the derived window explicitly.
To test incremental fault-window growth, set
`POLARIS_SHIM_MANAGED_INITIAL_BLOCKS=<count>` or
`POLARIS_SHIM_MANAGED_INITIAL_LENGTH=<bytes>` to register only an initial
prefix of that capacity with `polaris.ko`; later selected allocations grow the
same v4 VA-space registration in place before handing out addresses beyond the
current registered prefix. `POLARIS_SHIM_MANAGED_GROW_BLOCKS=<count>` controls
the minimum growth step when the next allocation needs more tokens. This does
not enable static RM; pages are still materialized through the daemon-backed RM
fault path.

The older harness mode is still supported. A harness can provide the exact UVM
token and managed window through environment variables:

    POLARIS_SHIM_GPU_ID=0 \
    POLARIS_SHIM_RM_CLIENT_TOKEN=0x5678 \
    POLARIS_SHIM_VASPACE_TOKEN=0x1234 \
    POLARIS_SHIM_MANAGED_BASE=0x1000000000 \
    POLARIS_SHIM_MANAGED_LENGTH=0x400000 \
    LD_PRELOAD=$(realpath ./libpolaris-shim.so) ./your-cuda-app

When all four variables are present, the shim submits
`POLARIS_REGISTER_VASPACE` on first `cuInit` and unregisters it at process
exit. `POLARIS_SHIM_RM_CLIENT_TOKEN` is optional for legacy M2 harnesses; if
omitted, the shim passes `rm_client_token=0` for non-fault control-plane
probes. The v4 UVM fault path needs the real RM client token because UVM
dispatch and RM memory handles are keyed by the full
`(gpu_id, rm_client_token, va_space_token)` tuple. Missing variables leave
`cuInit` as a pass-through.

An allocator slice is wired for shim-created and harness-created VA-spaces.
With `POLARIS_SHIM_MANAGE_ALLOCATIONS=1`, the shim interposes `cuMemAlloc`,
`cuMemAlloc_v2`, `cuMemAllocAsync`, `cuMemAllocAsync_v2`, `cudaMalloc`,
`cudaMallocManaged`, `cudaMallocAsync`, `cuMemFree`, `cuMemFree_v2`,
`cuMemFreeAsync`, `cuMemFreeAsync_v2`, `cudaFree`, and `cudaFreeAsync`.
It also forwards common CUDA runtime setup calls (`cudaSetDevice`,
`cudaSetDeviceFlags`, `cudaGetDeviceFlags`, `cudaGetDevice`,
`cudaGetDeviceCount`,
`cudaGetDeviceProperties`, `cudaDeviceGetAttribute`,
`cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`,
`cudaRuntimeGetVersion`, `cudaDriverGetVersion`, and
`cudaDeviceSynchronize`) so pre-allocation workload setup still goes through
the shim bootstrap path. `cudaSetDevice` / `cudaGetDevice` record the runtime
device selection; if `POLARIS_SHIM_CUDA_ORDINAL` is not set, RM/UVM bootstrap
uses that selected device.
Runtime and driver memory/error query calls (`cudaMemGetInfo`,
`cuMemGetInfo_v2`, `cudaGetErrorString`, `cudaGetErrorName`,
`cudaGetLastError`, `cudaPeekAtLastError`, `cuGetErrorString`, and
`cuGetErrorName`) are also forwarded so allocator probes and diagnostic paths
do not bypass or crash the shim.
Common runtime stream/event lifecycle calls (`cudaStreamCreate`,
`cudaStreamCreateWithFlags`, `cudaStreamSynchronize`, `cudaStreamDestroy`,
`cudaStreamWaitEvent`, `cudaStreamIsCapturing`, `cudaEventCreate`,
`cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaEventSynchronize`,
`cudaEventDestroy`, and `cudaEventElapsedTime`) are forwarded as well.
Runtime graph lifecycle/update calls (`cudaGraphInstantiate`,
`cudaGraphExecUpdate`, `cudaGraphDestroy`, and `cudaGraphExecDestroy`) are
forwarded for non-Polaris graph management, while graph capture/launch entry
points are guarded while Polaris allocations are live.
CUDA driver VMM pool primitives used by llama.cpp's CUDA backend
(`cuMemAddressReserve`, `cuMemAddressFree`, `cuMemCreate`, `cuMemRelease`,
`cuMemMap`, `cuMemUnmap`, `cuMemSetAccess`, and
`cuMemGetAllocationGranularity`) are forwarded as compatibility surfaces.
They are not used as the v4 production allocation path because raw CUDA VMM
reservations are not UVM fault-capable.
Runtime and driver kernel launch entry points (`cudaLaunchKernel`,
`cudaLaunchKernelExC`, `cuLaunchKernel`, and `cuLaunchKernelEx`) are forwarded
so ordinary CUDA kernel launches can run through the shim. Extended launch
configs that request CUDA Programmatic Dependent Launch through
`cudaLaunchAttributeProgrammaticStreamSerialization`,
`cudaLaunchAttributeProgrammaticEvent`,
`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`, or
`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT` are rejected with
`cudaErrorNotSupported` while shim-managed Polaris allocations are live.
Additional llama.cpp helper surfaces are forwarded as pass-throughs:
`cuDeviceGet`, `cuDeviceGetAttribute`, `cuDevicePrimaryCtxRetain`,
`cuCtxSetCurrent`, `cudaDeviceDisablePeerAccess`, `cudaDeviceGetPCIBusId`,
`cudaLaunchHostFunc`, `cudaLaunchHostFunc_v2`,
`cudaLaunchCooperativeKernel`, `cudaOccupancyMaxPotentialBlockSize`,
`cudaGraphGetNodes`, `cudaGraphNodeGetType`,
`cudaGraphKernelNodeGetParams`, and `cudaGraphKernelNodeSetParams`.
Pinned-host and kernel-query calls used by llama.cpp are also forwarded:
`cudaMallocHost`, `cudaHostAlloc`, `cudaHostGetDevicePointer`,
`cudaFreeHost`, `cudaHostRegister`, `cudaHostUnregister`,
`cudaFuncSetAttribute`, `cudaFuncGetAttributes`, and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
Each allocation reserves a deferred logical Polaris block, registers a
`block_id -> worker VA` mapping, and returns the Polaris VA to the worker.
Pages are still unmapped; the next GPU dereference must fault through UVM and
be serviced by polaris.ko.
For the production path, selected allocations are deferred logical blocks: the
shim does not allocate RM memory and does not register static backing. The first
GPU fault queues an `ALLOC` decision, `polarisd` publishes daemon-owned RM
backing through `POLARIS_COMPLETE_OPERATION`, and polaris.ko bridge-maps that
backing through UVM before returning `HANDLED`. Real CUDA kernels may fault
through CUDA's own UVM-registered VA-space rather than the shim-created RM/UVM
VA-space; for that case polaris.ko services the fault only when a single
unambiguous logical block mapping covers the faulting GPU/address.
`POLARISD_RM_BACKING=1` also wires daemon-backed `OFFLOAD` and `RELOAD`
through `POLARIS_RM_COPY`, which copies between daemon-owned RM vidmem and the
pinned CPU pool using UVM-owned staging and CE. RM-backed overwrite
`COW_BREAK` uses the same staged CPU-pool copy path for the current
`SESSION_BRANCH` + overwrite-reserve surface.
`POLARIS_SHIM_STATIC_RM_BACKEND=1` remains available only as a diagnostic
backend: it allocates shim-owned RM vidmem and registers both
`POLARIS_REGISTER_STATIC_BLOCK` and `POLARIS_REGISTER_BLOCK_BACKING`.
Set `POLARIS_SHIM_MIN_MANAGED_ALLOC=<bytes>` and/or
`POLARIS_SHIM_MAX_MANAGED_ALLOC=<bytes>` to restrict which allocation sizes
are routed through Polaris. Allocations outside that inclusive policy range
fall through to the real CUDA allocator even when
`POLARIS_SHIM_STRICT_MANAGED_ALLOC=1`; strict mode only changes failures for
allocations that the policy selected for Polaris management.
Explicit frees release the logical block. If a worker exits with outstanding
shim-managed allocations, the shim releases those blocks before destroying its
Polaris session and unregistering the VA-space.
Freed token spans are coalesced and reused by later allocations, so a workload
can repeatedly allocate/free within the configured managed window without
monotonically exhausting it.
The shim also answers `cuMemGetAddressRange` / `cuMemGetAddressRange_v2`,
`cuPointerGetAttribute` / `cuPointerGetAttributes`, and runtime
`cudaPointerGetAttributes` for managed Polaris pointers. Driver-API pointer
attribute queries support `MEMORY_TYPE`, `DEVICE_POINTER`,
`HOST_POINTER`, `IS_MANAGED`, `DEVICE_ORDINAL`, `RANGE_START_ADDR`,
`RANGE_SIZE`, and `MEMPOOL_HANDLE`; unsupported attributes fall through to
the real CUDA driver.
`libpolaris-shim/build/managed_alloc` can exercise either allocator path:
by default it calls `cuMemAlloc_v2` / `cuMemFree_v2`, and with
`POLARIS_SHIM_TEST_RUNTIME_ALLOC=1` it calls `cudaMalloc` / `cudaFree`.
Set `POLARIS_SHIM_TEST_MANAGED_ALLOC=1` to exercise
`cudaMallocManaged` / `cudaFree`, which is the llama.cpp path when
`GGML_CUDA_ENABLE_UNIFIED_MEMORY` is set. Polaris-serviced
`cudaMallocManaged` allocations are still external-range device VAs, so the
shim reports them through pointer attributes as device memory rather than CUDA
managed memory.
Set `POLARIS_SHIM_TEST_ADVISE=1` with the managed allocation smoke to validate
that `cudaMemAdvise` on a Polaris pointer is accepted as a no-op rather than
forwarded to CUDA's managed-memory subsystem.
Set `POLARIS_SHIM_TEST_ASYNC_ALLOC=1` to exercise the stream-ordered runtime
allocation surface (`cudaMallocAsync` / `cudaFreeAsync`) or
`POLARIS_SHIM_TEST_DRIVER_ASYNC_ALLOC=1` to exercise the stream-ordered
driver surface (`cuMemAllocAsync_v2` / `cuMemFreeAsync_v2`). The shim routes
successful async calls through the same Polaris allocation table and ignores
the stream because no pages are mapped until a later fault.
By default, if the fixed Polaris managed window cannot satisfy an allocation,
the shim falls through to the real CUDA allocator. Set
`POLARIS_SHIM_STRICT_MANAGED_ALLOC=1` to make managed-allocation failures
surface as CUDA allocation failures instead; this is useful for KV-only smoke
tests where silently mixing Polaris and non-Polaris pointers would hide
coverage gaps.
Set `POLARIS_SHIM_TEST_FILTER=1` with
`POLARIS_SHIM_MIN_MANAGED_ALLOC` above 1 MiB to validate that a small runtime
allocation falls through to CUDA while an in-policy allocation is still
managed by Polaris.
Set `POLARIS_SHIM_TEST_LARGE_WINDOW=1` to make the harness request three
managed blocks in one allocation. In bootstrapped mode, this validates that the
shim is using the derived RM/UVM managed window instead of the old 4 MiB smoke
window.
Set `POLARIS_SHIM_TEST_BLOCK_WINDOW=1` with
`POLARIS_SHIM_MANAGED_BLOCKS=2` and strict allocation mode to validate that the
bootstrapped managed window is capped to two allocator blocks: the first two
one-block allocations succeed and the third fails before falling through to
CUDA.
Set `POLARIS_SHIM_TEST_GROW_WINDOW=1` with
`POLARIS_SHIM_MANAGED_INITIAL_BLOCKS=1`, `POLARIS_SHIM_MANAGED_BLOCKS>=3`, and
strict allocation mode to validate that the shim grows the same registered
v4 fault window before admitting the second and third one-block allocations.
Set `POLARIS_SHIM_TEST_RUNTIME_SETUP=1` to validate runtime setup, memory/error
query, and stream/event pass-throughs before the managed allocation smoke.
Set `POLARIS_SHIM_REPORT_STATS=1` to print an exit-time allocator summary.
The report includes total intercepted allocation calls and bytes, selected
managed calls, size-policy pass-through calls, Polaris allocation successes
and failures, real CUDA fallback calls/results, managed/free counts, current
live bytes, peak live bytes, and per-allocation-API selected counters such as
`api_runtime_alloc_selected` and `api_runtime_managed_alloc_selected`. This is
intended for tuning M5 workload runs: with `POLARIS_SHIM_MIN_MANAGED_ALLOC` /
`POLARIS_SHIM_MAX_MANAGED_ALLOC`, the summary shows whether the intended
KV-sized allocations were routed through Polaris while smaller CUDA
runtime/control allocations stayed on the real CUDA allocator, and whether a
run used the default `cudaMalloc` path or the
`GGML_CUDA_ENABLE_UNIFIED_MEMORY` / `cudaMallocManaged` path.
The shim also interposes common runtime and driver memory operations
(`cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`, `cudaMemsetAsync`,
`cudaMemcpy2DAsync`, `cudaMemcpyPeerAsync`, `cudaMemcpy3DPeerAsync`,
`cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`, generic `cuMemcpy_v2`, their async
variants, and `cuMemsetD8` / `cuMemsetD16` / `cuMemsetD32` variants).
Operations that do not involve Polaris memory fall through to the real CUDA
library. Operations involving a Polaris pointer return
`cudaErrorNotSupported` instead of passing external VA to CUDA copy paths; local
hardware testing showed that a temporarily UVM-mapped static RM allocation is
still not safe to hand to ordinary CUDA copy APIs. For KV-only experiments,
`POLARIS_SHIM_ALLOW_ZERO_MEMSET=1` accepts base-address zero-fill requests
within a selected allocation as a no-op initialization declaration so the later
kernel dereference can still reach the replayable-fault path. Nonzero copy/fill
and daemon-backed production reload/copy/fill remain pending.
CUDA IPC export is explicitly guarded as unsupported for Polaris pointers:
`cuIpcGetMemHandle` and `cudaIpcGetMemHandle` return
`cudaErrorNotSupported` / the matching driver error for shim-managed
allocations, so callers cannot accidentally create an IPC handle for a VA
range whose residency is owned by Polaris.
CUDA Graph capture and launch are also guarded at the runtime and driver API
boundaries: `cudaStreamBeginCapture`, `cudaStreamEndCapture`,
`cudaGraphLaunch`, `cuStreamBeginCapture`, `cuStreamBeginCapture_v2`,
`cuStreamEndCapture`, and `cuStreamEndCapture_v2` return the stream-capture
unsupported error while the process has live shim-managed Polaris allocations.
If capture starts before any Polaris allocation exists, selected-size
`cudaMallocAsync` and `cuMemAllocAsync_v2` calls intentionally pass through to
CUDA instead of returning Polaris VA. This keeps graph capture or replay from
recording or launching work that may dereference unmapped Polaris VA.
The llama.cpp shim regression also sets `GGML_CUDA_PDL=0` by default so the
strict KV fault-path gate stays on ordinary kernel launches; if PDL launch
selection slips through while Polaris allocations are live, the shim rejects
the extended launch before CUDA can run work that may dereference unmapped
external VA.
Set `POLARIS_SHIM_TEST_GRAPH=1` in the `managed_alloc` harness to validate
the runtime and driver guards after Polaris allocations are live, and
`POLARIS_SHIM_TEST_GRAPH_CAPTURE_ALLOC=1` to validate capture-before-allocation
fallback.
Set `POLARIS_SHIM_TEST_VMM=1` to validate VMM symbol coverage and driver
pass-through with a safe invalid-argument probe that does not map or
dereference memory.
Set `POLARIS_SHIM_TEST_LAUNCH=1` to validate launch symbol coverage without
executing kernels.
Set `POLARIS_SHIM_TEST_PDL_GUARD=1` to validate that runtime and driver
extended launches with PDL attributes are rejected after a Polaris allocation
is live. The harness uses null kernel handles intentionally; the shim returns
before forwarding those guarded calls to CUDA.

For harness-created VA-spaces, the shim also has an opt-in UVM external-range
slice when the harness can pass the already-initialized UVM VA-space fd:

    POLARIS_SHIM_CREATE_EXTERNAL_RANGES=1 \
    POLARIS_SHIM_UVM_FD=7 \
    POLARIS_SHIM_MANAGE_ALLOCATIONS=1 \
    ...

`POLARIS_SHIM_UVM_FD` must refer to the same `/dev/nvidia-uvm` fd that was
used for `UVM_REGISTER_GPU_VASPACE`; creating ranges on a different fd creates
them in a different UVM VA-space, so the kernel bridge will not find them.
When enabled, every shim-managed allocation calls `UVM_CREATE_EXTERNAL_RANGE`
before registering the block mapping, and explicit frees call `UVM_FREE` to
destroy that external range.

The allocator slice requires a registered GPU and VA range in polaris.ko. For
diagnostic runs without a daemon, the shim can register a transient GPU before
registering the VA-space. `POLARIS_SHIM_BOOTSTRAP_RM_UVM=1` enables allocator
mode implicitly; harness mode should set `POLARIS_SHIM_MANAGE_ALLOCATIONS=1`:

    POLARIS_SHIM_MANAGE_ALLOCATIONS=1 \
    POLARIS_SHIM_TRANSIENT_GPU=1 \
    POLARIS_SHIM_GPU_ID=0 \
    POLARIS_SHIM_RM_CLIENT_TOKEN=0x5678 \
    POLARIS_SHIM_VASPACE_TOKEN=0x1234 \
    POLARIS_SHIM_MANAGED_BASE=0x1000000000 \
    POLARIS_SHIM_MANAGED_LENGTH=0x400000 \
    POLARIS_SHIM_BLOCK_SIZE=0x200000 \
    LD_PRELOAD=$(realpath ./libpolaris-shim.so) ./your-cuda-app

This is not yet the full production shim for llama.cpp: it creates the
fault-capable RM/UVM VA-space and per-allocation UVM external ranges, but it
still uses a bounded managed-window allocator model. The current M5 regression
now runs against a real `polarisd` with `POLARISD_RM_BACKING=1`, so llama.cpp
KV allocations are materialized through daemon-owned RM backing rather than
the shim's static RM diagnostic backend. Remaining production work is focused
on replacing this bounded window with workload-driven VA growth/reclamation,
broader stress coverage, and permission-based write-fault COW.

## Why this is not a Cargo crate

CUDA workers expect to `dlsym` C symbols out of the preloaded `.so`. A C
shared library is the path of least friction. The shim uses the small C ioctl
mirror in `include/polaris_abi.h`; the Rust `libpolaris` crate remains the
primary userspace API for Rust tools.

## Build

    make            # produces ./libpolaris-shim.so
    make tests      # builds ./build/smoke and ./build/managed_alloc
    make clean
