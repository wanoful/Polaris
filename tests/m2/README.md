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
  -> POLARIS_SPILL_BLOCK can queue OFFLOAD once RM copy support is enabled
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
programming, and does not by itself validate RM-backed `POLARIS_SPILL_BLOCK`;
use `--rm-spill-reload-roundtrip` for byte-integrity coverage across spill and
reload.

## RM CE Copy Probe

This diagnostic builds on the physical-address probe. It follows the
completion-backed resident path, fault-maps a harness-owned RM vidmem
allocation, then calls `POLARIS_PROBE_RM_COPY`. The kernel asks UVM to
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
diagnostic: production RM-backed `OFFLOAD`, `RELOAD`, and staged overwrite
`COW_BREAK` are covered by the later RM roundtrip and daemon-backed gates.

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
`OFFLOAD`, `RELOAD`, and staged overwrite `COW_BREAK`.

## RM Spill / Reload Roundtrip

This validates the production-shaped RM-backed spill/reload path without the
old RM-backed spill guard. It follows the completion-backed resident path,
writes a deterministic userspace pattern into RM backing with `POLARIS_RM_COPY`,
calls `POLARIS_SPILL_BLOCK`, completes the queued `OFFLOAD` by copying RM
backing to a CPU buffer, reloads into fresh RM backing through
`POLARIS_RM_COPY`, refaults the block, and verifies the bytes survived:

```sh
sudo tests/m2/m2_static_block_setup --rm-spill-reload-roundtrip
```

Expected success ends with:

```text
M3 Polaris RM spill/reload roundtrip passed.
```

The test still uses harness-owned RM allocations and a tiny executor loop; it
does not replace the long-running `polarisd` integration tests. It does prove
the same kernel/UVM copy primitive used by `polarisd` can preserve bytes across
RM-backed OFFLOAD and RELOAD.

## RM COW Roundtrip

This validates RM-backed COW byte preservation for the current
`SESSION_BRANCH` + overwrite-reserve COW surface. It writes a deterministic
pattern into a parent RM-backed block, branches the session, reserves an
overwrite block in the child, services the queued `COW_BREAK` by staging parent
RM backing through a CPU buffer with `POLARIS_RM_COPY`, copies that staged data
into fresh child RM backing, fault-maps the child, and verifies both parent and
child bytes:

```sh
sudo tests/m2/m2_static_block_setup --rm-cow-roundtrip
```

Expected success ends with:

```text
M4 Polaris RM COW roundtrip passed.
```

This is not permission-based write-fault COW for already mapped read-mostly
pages; it covers the overwrite-reserve COW control surface currently wired in
M4.

## Daemon-Backed RM COW Roundtrip

This validates the same RM-backed overwrite COW surface with a real
long-running `polarisd` and daemon-owned RM backing. Unlike
`--rm-cow-roundtrip`, this mode does not consume `GET_DECISION` or complete
operations inside the harness, does not register static backing, and does not
allocate harness-owned logical backing for the tested parent or child blocks.
Start `polarisd` with daemon-owned RM backing first:

```sh
POLARISD_RM_BACKING=1 target/debug/polarisd
sudo tests/m2/m2_static_block_setup --daemon-rm-cow-roundtrip
```

The mode reserves a deferred logical parent block, triggers the first synthetic
fault, waits for daemon-backed `ALLOC`, writes a deterministic parent pattern
with `POLARIS_RM_COPY`, branches the session, reserves an overwrite block in the
child, lets daemon `COW_BREAK` publish fresh child RM backing, fault-maps the
child, verifies parent and child bytes, and waits for daemon `FREE` cleanup to
drain.

Expected success ends with:

```text
M4 Polaris daemon-backed RM COW roundtrip passed.
```

This is the focused M4 gate for daemon-backed RM COW. It still covers the
overwrite-reserve control surface, not permission-based write-fault COW.

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
worker mapping, fault-maps through the completed logical backing, unmaps by
`block_id`, then refaults:

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
daemon-owned RM allocations, releases them on daemon `FREE` decisions, and
uses `POLARIS_RM_COPY` for RM-backed `OFFLOAD` and `RELOAD`.

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

Positive RM-backed spill execution is covered by
`--rm-spill-reload-roundtrip`,
`--daemon-rm-spill-reload-roundtrip`, and by the strict daemon-backed
llama.cpp gate. The `--spill-validation` mode remains a negative ioctl
validation for an unresident block.

## Daemon-Backed RM Spill / Reload Roundtrip

This validates the same byte-preserving spill/reload path with a real
long-running `polarisd` handling the decisions. Unlike
`--rm-spill-reload-roundtrip`, this mode does not consume `GET_DECISION` or
complete operations inside the harness, does not register static backing, and
does not allocate a harness-owned logical backing object. Start `polarisd` with
daemon-owned RM backing first:

```sh
POLARISD_RM_BACKING=1 target/debug/polarisd
sudo tests/m2/m2_static_block_setup --daemon-rm-spill-reload-roundtrip
```

The mode reserves a deferred logical block, triggers the first synthetic fault,
waits for daemon-backed `ALLOC`, writes a deterministic pattern into the
daemon-owned RM object with `POLARIS_RM_COPY`, calls `POLARIS_SPILL_BLOCK`,
waits for daemon `OFFLOAD`, reserves the same block again to force daemon
`RELOAD`, refaults it, and verifies the bytes survived.

Expected success ends with:

```text
M3 Polaris daemon-backed RM spill/reload roundtrip passed.
```

This is the focused M2/M3 gate for the production daemon-backed path; static RM
registration remains diagnostic-only.

## Daemon-Backed RM Spill / Reload Stress

This extends the focused daemon-backed spill/reload gate into a small repeated
cycle stress test. It uses the same real `polarisd` path as
`--daemon-rm-spill-reload-roundtrip`: no static RM registration, no
harness-owned logical backing, and no harness-side decision executor. Start
`polarisd` with daemon-owned RM backing first:

```sh
POLARISD_RM_BACKING=1 target/debug/polarisd
sudo tests/m2/m2_static_block_setup --daemon-rm-spill-reload-stress
```

The mode reserves one deferred logical block, lets the daemon materialize it,
then runs multiple `POLARIS_SPILL_BLOCK` / daemon `OFFLOAD` / daemon `RELOAD`
cycles. Each cycle writes a different deterministic pattern into daemon-owned
RM backing with `POLARIS_RM_COPY`, refaults after reload, copies the reloaded
bytes back, verifies byte integrity, and checks that the daemon decision queue
drains before the next cycle.

Expected success ends with:

```text
M6 Polaris daemon-backed RM spill/reload stress passed.
```

This is still a focused single-block stress gate. Fragmentation, dynamic KV
growth, and OOM pressure remain broader M6 work.

## Daemon-Backed RM Multi-Block Stress

This broadens the daemon-backed stress gate from one block to several deferred
logical blocks in one Polaris session. It still uses the production-shaped
daemon RM path: no static RM registration, no harness-owned logical backing, and
no harness-side decision executor. Start `polarisd` with daemon-owned RM backing
first:

```sh
POLARISD_RM_BACKING=1 target/debug/polarisd
sudo tests/m2/m2_static_block_setup --daemon-rm-multi-block-stress
```

The mode registers one UVM external range, reserves multiple deferred logical
blocks, fault-materializes each block through daemon `ALLOC`, writes unique
deterministic bytes into each daemon-owned RM backing, spills all blocks, waits
for daemon `OFFLOAD`, reloads the blocks in reverse order, refaults them, and
verifies byte integrity for every block after each cycle.

Expected success ends with:

```text
M6 Polaris daemon-backed RM multi-block stress passed.
```

This catches cross-block cleanup and queued decision ordering issues while
remaining a focused gate. Dynamic growth and fragmentation coverage lives in
the next daemon-backed gate; OOM pressure remains broader M6 work.

## Daemon-Backed RM Dynamic Fragmentation Stress

This adds a small dynamic-growth and fragmentation gate on top of the same real
daemon-backed RM path. It still avoids static RM registration, harness-owned
logical backing, and harness-side decision execution. Start `polarisd` with
daemon-owned RM backing first:

```sh
POLARISD_RM_BACKING=1 target/debug/polarisd
sudo tests/m2/m2_static_block_setup --daemon-rm-dynamic-fragmentation-stress
```

The mode registers one UVM external range, reserves five deferred logical
blocks in one Polaris session, fault-materializes each block through daemon
`ALLOC`, and writes deterministic bytes into every daemon-owned RM object with
`POLARIS_RM_COPY`. It then releases alternating blocks to create holes, waits
for daemon `FREE` cleanup and kernel block removal, regrows those token ranges
as fresh deferred logical blocks, writes new deterministic bytes into the
regrown backing, spills all live blocks, waits for daemon `OFFLOAD`, reloads
them, refaults, and verifies byte integrity for both survivor and regrown
blocks. The gate also checks that `pending_decs` drains and `static_blocks`
stays zero.

Expected success ends with:

```text
M6 Polaris daemon-backed RM dynamic fragmentation stress passed.
```

This covers the first dynamic-growth / fragmentation slice for the production
daemon-backed RM path. The host-pool and RM allocation OOM sides live in the
next daemon-backed gates. Near-capacity resident-set pressure lives in the
focused soak gate below.

## Daemon-Backed RM Near-Capacity Soak

This validates budget-pressure spill/reload behavior on the production-shaped
daemon RM path. It still avoids static RM registration, harness-owned logical
backing, and harness-side decision execution. Start `polarisd` with
daemon-owned RM backing and a two-block GPU budget:

```sh
POLARISD_RM_BACKING=1 POLARISD_GPU_BUDGET_BYTES=4194304 target/debug/polarisd
sudo tests/m2/m2_static_block_setup --daemon-rm-near-capacity-soak
```

The mode reserves four deferred 2 MiB logical blocks in one Polaris session,
fault-materializes each block through daemon `ALLOC`, and writes deterministic
bytes into daemon-owned RM backing with `POLARIS_RM_COPY`. With a 4 MiB budget,
the gate requires the kernel/daemon path to hold the resident set at two blocks:
`resident=2`, `gpu_used_mib=4`, and `offloaded=2`, while `static_blocks` stays
zero.

The reload phase intentionally exercises the UVM-hook async reload behavior. If
a block is `CpuOffloaded`, the first synthetic fault queues a real daemon
`RELOAD` and returns handled for replay; the harness waits for the block to
become `Resident`, then dispatches a second fault to map the daemon-published RM
backing and verifies byte integrity with `POLARIS_RM_COPY_TO_CPU`. Cleanup
releases all blocks through daemon `FREE` and checks the decision queue drains.
The same gate samples `/sys/kernel/polaris/stats` before and after the run and
requires bridge map telemetry to move on the live path:
`uvm_bridge_map_calls`, `uvm_bridge_map_ok`, `uvm_bridge_map_err`,
`uvm_bridge_map_last_ns`, and `uvm_bridge_map_avg_ns`.

Expected success ends with:

```text
M6 Polaris daemon-backed RM near-capacity soak passed.
```

Passing this gate proves the focused near-capacity resident-set cap and
reload/refault path on the live daemon-backed RM route without static RM or fake
test errors. It also covers the first M6 bridge latency telemetry assertion for
real UVM bridge map calls.

## Daemon-Backed RM Host-Pool OOM Pressure

This validates a deterministic host pinned-pool exhaustion path with the same
production-shaped daemon RM backend. It still avoids static RM registration,
harness-owned logical backing, and harness-side decision execution. Start
`polarisd` with daemon-owned RM backing and a one-block CPU pool:

```sh
POLARISD_RM_BACKING=1 POLARISD_CPU_POOL_BYTES=2097152 target/debug/polarisd
sudo tests/m2/m2_static_block_setup --daemon-rm-host-pool-oom-pressure
```

The mode reserves two deferred logical blocks, fault-materializes both through
real daemon `ALLOC`, writes deterministic bytes into each daemon-owned RM
object with `POLARIS_RM_COPY`, and spills the first block. That first daemon
`OFFLOAD` succeeds and fills the 2 MiB CPU pool. The second spill then reaches
real daemon `-ENOMEM`, the kernel retries the `OFFLOAD` decision, and the block
is marked `Evicted` after the bounded retry limit. Cleanup releases both the
CPU-offloaded block and the evicted block through daemon `FREE`, including the
evicted block's still-owned RM backing.

Expected success ends with:

```text
M6 Polaris daemon-backed RM host-pool OOM pressure passed.
```

Passing this gate proves the host-pool OOM path on the live daemon-backed RM
route without falling back to static RM or a fake test error.

## Daemon-Backed RM Allocation OOM Pressure

This validates the VRAM/RM allocation failure side of M6 on the same production
daemon-backed path. Start `polarisd` with daemon-owned RM backing and a kernel
GPU budget above the oversized logical block request:

```sh
POLARISD_RM_BACKING=1 POLARISD_RM_PRESSURE_OUTSIDE_FB_RANGE=1 POLARISD_GPU_BUDGET_BYTES=68719476736 target/debug/polarisd
sudo tests/m2/m2_static_block_setup --daemon-rm-alloc-oom-pressure
```

The mode registers a UVM external range matching one 32 GiB deferred logical
block. The daemon is started with a 64 GiB kernel GPU budget so the request
reaches `polarisd`, and with `POLARISD_RM_PRESSURE_OUTSIDE_FB_RANGE=1` so the
real RM `NV01_MEMORY_LOCAL_USER` request is constrained to a physical FB range
outside local memory using `NVOS32_ALLOC_FLAGS_USE_BEGIN_END`. On the local
16 GiB RTX 5070 Ti this makes RM return real `NV_ERR_NO_MEMORY` (`0x51`) before
any UVM bridge map is attempted. `polarisd` treats non-warning `NV_STATUS`
values as failures and completes the `ALLOC` with `-ENOMEM`; the kernel retries,
marks the block `Evicted` after the bounded retry limit, and reports UVM
`ERROR` for the fault instead of `HANDLED`.

Expected success ends with:

```text
M6 Polaris daemon-backed RM allocation OOM pressure passed.
```

Passing this gate proves the RM allocation OOM path on the live daemon-backed
RM route without static RM or `POLARIS_TEST_ERROR`. The pressure is a
deterministic out-of-FB physical range constraint on the daemon's real RM
allocation; the harness also checks that `uvm_last_map_ret` is unchanged so a
bridge mapping failure cannot satisfy the gate. Near-capacity fragmentation
pressure can still be added as a broader soak test.

For non-CUDA control-plane coverage, `libpolaris/tests/kernel_spill_state.rs`
contains ignored root-only tests that use fake userspace executors to complete
legacy `ALLOC`, `OFFLOAD`, and `RELOAD` decisions, plus a real-`polarisd`
RM-backed `ALLOC`/`FREE` lifetime test. They validate control-plane queueing
and cleanup without exercising NVIDIA UVM channel registration or the observed
`gpu_va_space_ptr` required by `POLARIS_RM_COPY`.

The same ignored test file also covers the first M4 COW control-plane slice:
branching a session increments parent block refcounts, overwrite reserve on
the child creates a private block, and the queued `COW_BREAK` decision carries
the writer's destination VA. The test also checks that the child session's
block table contains only the new private block after the split, so teardown
does not accidentally release the parent's remaining block. Byte-integrity
coverage lives in `--rm-cow-roundtrip`; the equivalent real-daemon gate is
`--daemon-rm-cow-roundtrip`.

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
