# M2 Static-Block Fault Diagnostic

This directory contains the staged M2 diagnostic for the v4 fault path:

```text
UVM synthetic fault
  -> uvm_polaris_dispatch_fault()
  -> polaris.ko handle_gpu_fault
  -> uvm_polaris_map_external_allocation()
  -> HANDLED
```

The M3 diagnostic extension adds:

```text
POLARIS_UNMAP_STATIC_BLOCK
  -> uvm_polaris_unmap_external_allocation()
  -> second synthetic fault remaps the same external range
```

The block-level M3 diagnostic adds the production-oriented teardown key:

```text
POLARIS_REGISTER_BLOCK_MAPPING(block_id, worker VA range)
  -> synthetic fault records the observed gpu_va_space_ptr
  -> POLARIS_UNMAP_BLOCK_MAPPINGS(block_id)
  -> second synthetic fault remaps the same external range
```

The logical-backed M3 diagnostic removes the static-block table from that
fault-map path:

```text
POLARIS_REGISTER_BLOCK_MAPPING(block_id, worker VA range)
POLARIS_REGISTER_BLOCK_BACKING(block_id, RM allocation tuple)
  -> synthetic fault maps the logical block through the UVM bridge
  -> POLARIS_UNMAP_BLOCK_MAPPINGS(block_id)
  -> second synthetic fault remaps through the logical mapping again
```

The completion-backed M3 diagnostic proves the same logical backing can be
published through the decision-completion path instead of a sideband backing
ioctl:

```text
POLARIS_BLOCK_RESERVE(block_id) without DEFER_FAULT
  -> POLARIS_GET_DECISION returns ALLOC for the block
  -> POLARIS_COMPLETE_OPERATION carries RM hClient/hMemory/length metadata
  -> POLARIS_REGISTER_BLOCK_MAPPING(block_id, worker VA range)
  -> synthetic fault maps the completed logical block through the UVM bridge
  -> POLARIS_SPILL_BLOCK rejects the RM-backed block with EOPNOTSUPP
  -> POLARIS_UNMAP_BLOCK_MAPPINGS(block_id)
  -> second synthetic fault remaps through the completed logical backing
```

The deferred completion diagnostic matches the shim's normal allocator shape:

```text
POLARIS_BLOCK_RESERVE(block_id) with DEFER_FAULT
POLARIS_REGISTER_BLOCK_MAPPING(block_id, worker VA range)
  -> synthetic fault has no RM backing yet
  -> polaris.ko queues ALLOC on the UVM hook slow path
  -> executor replies through POLARIS_COMPLETE_OPERATION with RM backing
  -> the same hook invocation bridge-maps the completed logical block
  -> unmap/refault still remaps through the completed backing
```

Current release/destroy cleanup separates production and diagnostic ownership:
daemon-owned live backing queues a `FREE` decision from `BLOCK_RELEASE` or
`SESSION_DESTROY`, while this harness releases its own RM objects by issuing
`BLOCK_RELEASE` with `POLARIS_RELEASE_FLAG_CALLER_OWNS_BACKING` before session
teardown.

The CUDA copy probe is a narrow diagnostic for the remaining RM-backed
spill/reload gap. It maps a harness-owned RM `NV01_MEMORY_LOCAL_USER` object
through public UVM external-allocation ioctls, establishes a normal CUDA primary
context for the same device, and tries `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2`
against the external VA. The probe runs in a child process so a libcuda crash is
reported as evidence instead of terminating the parent test runner. A failed or
crashing probe means ordinary daemon-side CUDA copy APIs should not be assumed
safe for UVM external VAs; implement the production copy path through a
kernel/UVM helper or another explicitly validated mechanism instead.

The diagnostic creates real RM objects, registers a fault-capable GPU
VA-space with UVM, creates a UVM external range, registers the same
`(rm_client_token, user_rm_va_space)` handle pair with polaris.ko, and
optionally asks UVM's builtin test ioctl to dispatch a synthetic fault through
the live hook.

## Prerequisites

- Patched `nvidia-uvm.ko` from the `polaris-v4` driver branch loaded.
- UVM builtin tests enabled so `UVM_TEST_POLARIS_DISPATCH_FAULT` is accepted.
- `polaris.ko` built against the patched UVM `Module.symvers` and loaded.
- CUDA headers and `libcuda.so` available.
- The `third_party/open-gpu-kernel-modules` submodule, or pass
  `NVIDIA_KO_DIR=/path/to/open-gpu-kernel-modules`.

Build Polaris and the diagnostic:

```sh
make kernel
make -C tests/m2
```

Build with explicit paths:

```sh
make kernel NVIDIA_KO_DIR=/path/to/open-gpu-kernel-modules
make -C tests/m2 NVIDIA_KO_DIR=/path/to/open-gpu-kernel-modules CUDA_HOME=/usr/local/cuda
```

## Setup-Only Probe

This validates RM client/device/VA-space/memory allocation, UVM GPU/VA-space
registration, UVM external-range creation, and polaris.ko static-block
registration. It pre-maps the external allocation through the stock UVM ioctl
and does not dispatch a Polaris-serviced fault:

```sh
sudo tests/m2/m2_static_block_setup
```

Optional positional arguments:

```text
tests/m2/m2_static_block_setup [cuda_ordinal] [polaris_gpu_id] [base]
```

## CUDA Copy Probe

This validates only CUDA-copy visibility of a UVM-mapped RM external
allocation. It does not register the range with polaris.ko and does not dispatch
a Polaris fault:

```sh
sudo tests/m2/m2_static_block_setup --cuda-copy-probe
```

Expected outcomes are diagnostic rather than pass/fail for production:

- `CUDA copy APIs can round-trip...`: daemon-side CUDA copy might be viable, but
  still needs integration into `polarisd` and real spill/reload tests.
- `current CUDA context cannot ...` or `child terminated by signal ...`:
  ordinary CUDA copy APIs are not safe/visible for this external VA shape, so
  the RM-backed spill/reload copy path should move into a UVM/kernel helper or
  another lower-level path.

Local result on 2026-06-14: the probe successfully created RM memory and mapped
it through `UVM_MAP_EXTERNAL_ALLOCATION`, then reached `cuMemcpyHtoD_v2` with a
current CUDA primary context and the child terminated with `SIGSEGV`. Treat that
as evidence against using ordinary daemon-side CUDA copy APIs for the production
RM-backed spill/reload path.

The sibling RM CPU-map probe checks whether daemon-owned vidmem can be copied by
mapping the RM object into the daemon CPU address space:

```sh
sudo tests/m2/m2_static_block_setup --rm-cpu-map-probe
```

Local result on 2026-06-14: `NV_ESC_RM_MAP_MEMORY` returned a CPU address for
the vidmem object, but the isolated child terminated with `SIGSEGV` as soon as
it touched that mapping. Treat that as evidence against a simple daemon-side
`memcpy` spill/reload path for these RM vidmem objects.

## RM Physical Address Probe

This is the next diagnostic for the same RM-backed spill/reload gap. It follows
the completion-backed resident path, fault-maps a harness-owned RM vidmem
allocation, then calls `POLARIS_PROBE_RM_PHYS`. The kernel asks UVM to duplicate
the RM allocation and query RM for GPU-visible physical-address geometry:

```sh
sudo tests/m2/m2_static_block_setup --rm-phys-probe
```

Expected success ends with:

```text
M3 Polaris RM phys probe passed.
```

The probe prints page size, physical-address count, first/last physical address,
and flags (`contiguous`, `sysmem`, `egm`, `fabricmem`). Passing this test means
the RM allocation shape can expose the physical-address metadata a later
UVM/kernel copy helper will need. It does not copy data, does not validate CE
programming, and does not make RM-backed `POLARIS_SPILL_BLOCK` production-ready;
the `EOPNOTSUPP` guards remain correct until a real RM-backed spill/reload
round-trip validates byte integrity.

## RM CE Copy Probe

This diagnostic builds on the physical-address probe. It follows the
completion-backed resident path, fault-maps a harness-owned RM vidmem
allocation, confirms `POLARIS_SPILL_BLOCK` still rejects RM-backed spill with
`EOPNOTSUPP`, then calls `POLARIS_PROBE_RM_COPY`. The kernel asks UVM to
duplicate/query the RM allocation, stage a deterministic CPU pattern in
UVM-owned sysmem DMA memory, CE-copy the pattern into the RM allocation's
GPU-visible physical address, CE-copy it back to another sysmem staging buffer,
and verify the bytes on CPU:

```sh
sudo tests/m2/m2_static_block_setup --rm-copy-probe
```

Expected success ends with:

```text
M3 Polaris completion-backed block refault test passed.
```

The probe prints the copied byte count, page size, physical-address geometry,
flags, and first mismatch. A successful run reports
`mismatch=0xffffffffffffffff`. Passing this test proves the narrow local
backing shape used by the harness (contiguous vidmem, one 2 MiB page on the
validated GPU) can move bytes through UVM's CE copy path. It is still a
diagnostic: production RM-backed `OFFLOAD`, `RELOAD`, and `COW_BREAK` remain
guarded until polarisd/runtime spill and reload are wired to this copy path.

## RM User-Buffer Copy Roundtrip

This validates the production-shaped copy ABI that `polarisd` needs for
RM-backed spill/reload. It follows the same completion-backed resident path as
the CE probe, then uses `POLARIS_RM_COPY` to copy a deterministic userspace
buffer into the RM allocation and copy it back into a second userspace buffer:

```sh
sudo tests/m2/m2_static_block_setup --rm-copy-roundtrip
```

Expected success ends with:

```text
M3 Polaris completion-backed block refault test passed.
```

The helper still supports only the narrow local shape proven by the probe:
contiguous vidmem reachable through UVM's CE path. Passing this mode proves the
kernel/UVM helper can move bytes between an RM-backed block and an ordinary
userspace CPU pointer, which is the primitive needed by daemon-backed
`OFFLOAD` and `RELOAD`. `COW_BREAK` remains guarded until an RM-to-RM or
staged copy path is integrated.

## Synthetic Fault Dispatch

This leaves the external range unmapped, probes UVM for the exact dispatch key,
registers that key with polaris.ko, then dispatches a synthetic write fault:

```sh
sudo tests/m2/m2_static_block_setup --dispatch-fault
```

Expected success ends with:

```text
M2 Polaris fault-dispatch test passed.
```

The test also prints the observed UVM dispatch key:

```text
UVM dispatch key: gpu_id=<id> client=0x<rm_client> token=0x<user_rm_va_space> result=1
```

`result=1` is `UVM_POLARIS_FAULT_HANDLED`.

## Unmap / Refault Diagnostic

This validates the first M3 spill primitive. It fault-maps the static block,
asks polaris.ko to unmap it through the new UVM bridge, then dispatches a
second synthetic fault that must remap the same external range:

```sh
sudo tests/m2/m2_static_block_setup --unmap-refault
```

Expected success ends with:

```text
M2 Polaris unmap/refault test passed.
```

## Block Unmap / Refault Diagnostic

This validates the next M3 teardown primitive. It creates a real Polaris
session and logical block, registers that block's worker mapping against the
same static external allocation, fault-maps it, unmaps by `block_id` through
`POLARIS_UNMAP_BLOCK_MAPPINGS`, verifies one observed mapping was unmapped,
then dispatches a second synthetic fault that must remap the range:

```sh
sudo tests/m2/m2_static_block_setup --block-unmap-refault
```

Expected success ends with:

```text
M3 Polaris block unmap/refault test passed.
```

This still does not perform device-to-host copy or RM allocation release. It
only proves the block-to-worker mapping registry and UVM unmap/refault
teardown path that production spill will build on.

## Logical-Backed Block Refault Diagnostic

This validates that the live UVM bridge no longer requires a static-block fast
entry. It creates the same real Polaris session and logical block as
`--block-unmap-refault`, skips `POLARIS_REGISTER_STATIC_BLOCK`, attaches the
diagnostic RM allocation to the logical block with
`POLARIS_REGISTER_BLOCK_BACKING`, fault-maps through the logical block mapping,
unmaps by `block_id`, then refaults through the logical path again:

```sh
sudo tests/m2/m2_static_block_setup --logical-backed-refault
```

Expected success ends with:

```text
M3 Polaris logical-backed block refault test passed.
```

This is still a bridge diagnostic. The RM object is created by the harness, not
by polarisd, and no host/device copy or daemon-backed spill/reload decision is
executed.

## Completion-Backed Block Refault Diagnostic

This validates the production-facing completion contract for RM-backed resident
blocks. Unlike `--logical-backed-refault`, it does not call
`POLARIS_REGISTER_BLOCK_BACKING`. Instead, the harness reserves a logical block
without `POLARIS_RESERVE_FLAG_DEFER_FAULT`, runs a small executor thread that
polls `POLARIS_GET_DECISION`, completes the queued `ALLOC` decision with the
diagnostic RM backing metadata in `POLARIS_COMPLETE_OPERATION`, registers the
worker mapping, fault-maps through the completed logical backing, verifies that
`POLARIS_SPILL_BLOCK` rejects this RM-backed resident block with `EOPNOTSUPP`
without tearing down the observed mapping, unmaps by `block_id`, then refaults:

```sh
sudo tests/m2/m2_static_block_setup --complete-backed-refault
```

Expected success ends with:

```text
M3 Polaris completion-backed block refault test passed.
```

This closes the kernel ABI gap for daemon/runtime executors to publish
UVM-bridge-mapable RM backing as part of normal decision completion. It still
uses a harness-created RM object. `polarisd` now has an opt-in
`POLARISD_RM_BACKING=1` backend that exercises the same completion ABI with
daemon-owned RM allocations and releases them on daemon `FREE` decisions, but
host/device copy for production RM-backed spill/reload is still pending. Until
that copy path is implemented, `POLARIS_SPILL_BLOCK` intentionally supports
only legacy CUDA VMM resident blocks with `gpu_phys_handle`; RM-backed
bridge-resident blocks are rejected before any PTE teardown.

## Deferred Completion Fault Diagnostic

This validates the shim-facing fault path for deferred logical blocks. Unlike
`--complete-backed-refault`, the block is reserved with
`POLARIS_RESERVE_FLAG_DEFER_FAULT`, so no `ALLOC` decision exists until the
synthetic UVM fault reaches polaris.ko. The fault hook detects the registered
logical mapping, queues the existing `ALLOC` decision path, waits for the
executor to complete it with RM backing metadata, and then bridge-maps the
same fault before returning `HANDLED`:

```sh
sudo tests/m2/m2_static_block_setup --deferred-complete-fault
```

Expected success ends with:

```text
M3 Polaris deferred completion fault test passed.
```

This is the closest M2/M3 diagnostic to the production shim allocation path:
the shim can reserve external-range VA without static backing, and the first
GPU fault can materialize bridge-mapable RM residency through the daemon
completion contract.

## Spill Ioctl Validation

This validates the public M3 spill ioctl surface for the static harness. It
creates a real Polaris session and logical block, registers its worker mapping,
then calls `POLARIS_SPILL_BLOCK` while the logical block is still unresident.
The expected result is `ENOENT`: the ioctl exists, decodes correctly, validates
the block, and refuses to queue an `OFFLOAD` decision without a resident
physical handle.

```sh
sudo tests/m2/m2_static_block_setup --spill-validation
```

Expected success ends with:

```text
M3 Polaris spill ioctl validation passed.
```

Positive spill execution still needs a daemon/runtime-created resident logical
block. The static RM harness maps a diagnostic RM allocation through UVM, but
that allocation is not stored as `gpu_phys_handle` in the logical block table,
so it cannot prove device-to-host copy or RM allocation release.

For non-CUDA control-plane coverage, `libpolaris/tests/kernel_spill_state.rs`
contains an ignored root-only test that uses a fake userspace executor to
complete `ALLOC`, `OFFLOAD`, and `RELOAD` decisions. It validates that
`POLARIS_SPILL_BLOCK` queues the expected `OFFLOAD` decision and that a later
overwrite reserve of the same offloaded block queues `RELOAD`, without
exercising NVIDIA UVM channel registration.

The same ignored test file also covers the first M4 COW control-plane slice:
branching a session increments parent block refcounts, overwrite reserve on
the child creates a private block, and the queued `COW_BREAK` decision carries
the writer's destination VA. The test also checks that the child session's
block table contains only the new private block after the split, so teardown
does not accidentally release the parent's remaining block.

It also covers the first multi-worker mapping cleanup slice without CUDA:
two v4 worker VA-spaces register mappings for the same logical block, explicit
`POLARIS_UNREGISTER_VASPACE` removes only the first worker's mapping, and
closing the second worker fd reaps the remaining mapping. The test checks
`v4_va_spaces` and `block_mappings` in `/sys/kernel/polaris/stats` after each
step.

Two additional ignored tests cover logical block lifetime cleanup without CUDA:
`SESSION_DESTROY` and `BLOCK_RELEASE` both remove stale block mappings for a
deferred, unmapped logical block. Those tests require a freshly loaded
`polaris.ko` built with the block-mapping cleanup fix.

Three FREE-lifetime tests cover live backing release without CUDA:
`block_release_queues_free_for_resident_block`,
`block_release_queues_free_for_rm_backed_block_without_phys_handle`, and
`session_destroy_queues_free_for_rm_backed_block_without_phys_handle` verify
that legacy resident backing, RM-backed resident backing with no legacy
`gpu_phys_handle`, and RM-backed session teardown all reach the daemon
`FREE` path.

For real RM allocation/free smoke coverage, `polarisd` has an ignored unit test
that opens `/dev/nvidiactl`, creates the daemon RM client/device/subdevice, and
allocates/frees one `NV01_MEMORY_LOCAL_USER` object:

```sh
sudo cargo test -p polarisd rm::tests::rm_backend_allocates_and_frees_device_memory -- --ignored --nocapture
```

`child_block_release_decrements_inherited_shared_block` covers the COW-shared
release case: after `SESSION_BRANCH`, the child releases an inherited block
through `BLOCK_RELEASE`, the parent's block remains, and its refcount drops
from 2 to 1.
`child_block_release_reaps_only_child_mapping_for_shared_block` extends that
coverage to block-to-worker mappings: parent and child mappings are registered
for the shared block, child release reaps only the child's mapping, and the
parent mapping remains until its VA-space is explicitly unregistered.

`worker_process_exit_reaps_vaspace_and_block_mapping` covers the first M6
worker-lifetime control-plane slice. It spawns a child process that opens
`/dev/polaris`, registers one v4 VA-space and block mapping, then the parent
kills the child and verifies fd-close cleanup drops both `v4_va_spaces` and
`block_mappings`. The same path records the registering pid and exposes a
`v4_worker_pids` aggregate in `/sys/kernel/polaris/stats`; it does not yet
exercise a real UVM `gpu_va_space_ptr` invalidation callback.

The M5 shim allocator slice is build-covered under `libpolaris-shim`:
`make -C libpolaris-shim all tests` builds the LD_PRELOAD library plus
`build/smoke` and `build/managed_alloc`. `managed_alloc` resolves the shim's
`cuMemAlloc_v2` / `cuMemFree_v2` interposers without linking CUDA and is meant
to be run with `POLARIS_SHIM_MANAGE_ALLOCATIONS=1` after a freshly loaded
`polaris.ko` is available. It validates the shim's allocator ioctl path over a
provided harness VA-space. With `POLARIS_SHIM_CREATE_EXTERNAL_RANGES=1` and
`POLARIS_SHIM_UVM_FD=<fd>`, the shim also creates the UVM external range on
the already-registered UVM VA-space fd and frees it with `UVM_FREE` during
explicit or exit cleanup.
With `POLARIS_SHIM_BOOTSTRAP_RM_UVM=1`, the shim creates the RM/UVM VA-space
itself and enables external-range creation automatically; this is the M5 path,
not an M2 bridge diagnostic.
Set `POLARIS_SHIM_TEST_LARGE_WINDOW=1` in bootstrapped mode to request three
managed blocks in one allocation. This validates that the shim derived a
managed window from the RM/UVM VA-space instead of using the old 4 MiB harness
window.
Set `POLARIS_SHIM_TEST_RUNTIME_SETUP=1` to validate common CUDA runtime setup
pass-throughs (`cudaSetDevice`, `cudaGetDevice`, version queries, device
count, `cudaSetDeviceFlags`, `cudaGetDeviceFlags`,
`cudaDeviceGetAttribute`, peer-access probes, memory info, error query
helpers, `cudaStreamWaitEvent`, `cudaStreamIsCapturing`, and stream/event
lifecycle) before allocation.
Set `POLARIS_SHIM_TEST_HOST_APIS=1` to validate pinned-host pass-throughs used
by llama.cpp (`cudaMallocHost`, `cudaHostAlloc`, `cudaHostGetDevicePointer`,
`cudaFreeHost`, `cudaHostRegister`, and `cudaHostUnregister`) before the
managed allocation smoke.
Set `POLARIS_SHIM_TEST_REUSE=1` to validate that explicit frees return token
spans to the shim allocator and that an allocation larger than the configured
managed window fails cleanly.
Set `POLARIS_SHIM_TEST_ATTRS=1` to validate shim-serviced driver and runtime
pointer-attribute metadata, including allocator-probe attributes such as
`IS_MANAGED`, `DEVICE_ORDINAL`, and `MEMPOOL_HANDLE`, plus
`cuMemGetAddressRange_v2` for a shim-managed pointer.
Set `POLARIS_SHIM_TEST_RUNTIME_ALLOC=1` to run the same managed-allocation
smoke through the runtime `cudaMalloc` / `cudaFree` interposers instead of
the driver `cuMemAlloc_v2` / `cuMemFree_v2` interposers.
Set `POLARIS_SHIM_TEST_MANAGED_ALLOC=1` to run it through
`cudaMallocManaged` / `cudaFree`. This covers the llama.cpp
`GGML_CUDA_ENABLE_UNIFIED_MEMORY` allocation path while still treating the
returned Polaris pointer as external-range device memory for metadata queries.
Set `POLARIS_SHIM_TEST_ADVISE=1` with `POLARIS_SHIM_TEST_MANAGED_ALLOC=1` to
validate that `cudaMemAdvise` against that Polaris pointer is accepted as a
no-op compatibility hint.
Set `POLARIS_SHIM_TEST_ASYNC_ALLOC=1` to run it through
`cudaMallocAsync` / `cudaFreeAsync`; this implies the runtime allocation path.
Set `POLARIS_SHIM_TEST_DRIVER_ASYNC_ALLOC=1` to run the same smoke through
the driver stream-ordered allocation surface
`cuMemAllocAsync_v2` / `cuMemFreeAsync_v2`.
Set `POLARIS_SHIM_STRICT_MANAGED_ALLOC=1` when validating allocator
exhaustion so oversize allocations fail in the shim instead of falling back to
ordinary CUDA memory.
Set `POLARIS_SHIM_MIN_MANAGED_ALLOC=<bytes>` and/or
`POLARIS_SHIM_MAX_MANAGED_ALLOC=<bytes>` to restrict which allocations are
managed by Polaris. Allocations outside the size policy intentionally fall
through to CUDA even in strict mode. `POLARIS_SHIM_TEST_FILTER=1` validates
this path for runtime `cudaMalloc` by checking that a 1 MiB allocation falls
through when the minimum managed size is larger, then allocating an in-policy
Polaris pointer.
Set `POLARIS_SHIM_TEST_MEMCPY=1` to validate that runtime
`cudaMemcpy` / `cudaMemcpyAsync` / `cudaMemcpy2DAsync` /
`cudaMemcpyPeerAsync` / `cudaMemcpy3DPeerAsync` and driver
`cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2` / generic `cuMemcpy_v2` calls involving
shim-managed Polaris pointers are caught by the shim and return
`cudaErrorNotSupported` instead of falling through to CUDA.
Set `POLARIS_SHIM_TEST_MEMSET=1` to validate the same guard behavior for
runtime `cudaMemset` / `cudaMemsetAsync` and driver `cuMemsetD8_v2` /
`cuMemsetD16_v2` / `cuMemsetD32_v2` plus their async variants.
Set `POLARIS_SHIM_TEST_IPC=1` to validate that driver and runtime IPC handle
export calls reject shim-managed Polaris pointers with `cudaErrorNotSupported`.
Set `POLARIS_SHIM_TEST_GRAPH=1` to validate that runtime
`cudaStreamBeginCapture`, `cudaStreamEndCapture`, and `cudaGraphLaunch`, plus
driver `cuStreamBeginCapture_v2`, reject graph capture/launch while
shim-managed Polaris allocations are live.
Set `POLARIS_SHIM_TEST_VMM=1` to validate that CUDA driver VMM symbols used by
llama.cpp resolve through the shim and reach the real driver through a safe
invalid-argument probe.
Set `POLARIS_SHIM_TEST_LAUNCH=1` to validate that runtime and driver kernel
launch symbols and related helper symbols resolve through the shim without
executing kernels.
Set `POLARIS_SHIM_TEST_LEAK=1` when running `managed_alloc` to validate the
shim's process-exit cleanup for outstanding managed allocations.

## Notes

- This is not the production shim path. It intentionally uses direct RM/UVM
  ioctls from the harness so M2 can validate the kernel bridge before
  the production workload path is complete.
- The harness registers its GPU entry as transient. After the fd closes,
  `/sys/kernel/polaris/stats` should show `gpus: 0`, `v4_va_spaces: 0`, and
  `static_blocks: 0`; after `--block-unmap-refault`,
  `--logical-backed-refault`, or `--spill-validation`, it should also show
  `block_mappings: 0`. Hook counters remain cumulative for the loaded module.
- UVM dispatch and Polaris registration use
  `(gpu_id, rm_client_token, va_space_token)`. RM object handles are only
  unique within an RM client, so the client token is required when concurrent
  diagnostics use recycled `hVaSpace` values.
- The default VA base is `0x1000000000` and the managed range is 4 MiB.
- The static block is 2 MiB.
