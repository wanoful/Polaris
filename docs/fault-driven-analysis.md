# POLARIS Fault-Driven Path Analysis

This note records the current evidence for the M4 fault-driven path.

## Summary

The POLARIS kernel-to-runtime decision path works, but the fully automatic
"CUDA kernel touches an unmapped POLARIS CUDA VMM VA and UVM replays it" path is
not currently usable on the tested NVIDIA 610.43.02 driver.

The current evidence points to a hard boundary:

- The POLARIS hook is inside NVIDIA UVM's replayable fault service path, after
  UVM has already fetched, parsed, coalesced, and translated a replayable fault
  packet to a UVM `va_space`.
- A raw CUDA VMM hole created with `cuMemAddressReserve` and left unmapped is
  not automatically handled as a serviceable UVM replayable managed-memory
  fault.
- A CUDA kernel write to that VA produces a fatal NVIDIA RM/MMU fault:
  `Xid 31`, `FAULT_PDE`, `ACCESS_TYPE_VIRT_WRITE`, followed by
  `cudaErrorIllegalAddress`.

Therefore, the benchmarkable POLARIS path should remain explicit
prefetch/offload/reload from the llama.cpp runtime. The true fault-driven path
is still a research item requiring a different VA substrate or a deeper driver
hook with replay/resume semantics.

## What Works

The following M4 decision path is verified:

1. `POLARIS_BLOCK_RESERVE` creates an unmapped logical POLARIS block.
2. The kernel resolver queues an `ALLOC` decision.
3. The in-process runtime worker receives the decision through
   `GET_DECISION`.
4. The runtime executes CUDA VMM operations in the owning process context.
5. `COMPLETE_OPERATION` wakes the kernel waiter.
6. CUDA can read/write the returned GPU VA after the block is mapped.

This validates the kernel decision protocol and the in-process CUDA VMM
executor. It does not prove that a real GPU replayable fault will reach that
resolver.

## Current Hook Location

The patched UVM module calls:

```c
uvm_polaris_filter_replayable_faults(parent_gpu, batch_context, &polaris_handled_faults)
```

from `uvm_parent_gpu_service_replayable_faults()` in
`kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`, after:

1. `fetch_fault_buffer_entries()` reads the replayable fault buffer.
2. `parse_replayable_entry()` parses fault packets.
3. `preprocess_fault_batch()` translates each fault's `instance_ptr` to a UVM
   `va_space` and sorts/coalesces the batch.

That means POLARIS sees only faults which have already reached the UVM
replayable fault buffer. It does not see RM PRI faults, fatal Graphics faults
that bypass UVM servicing, or faults that RM reports only through the
non-replayable/fatal path.

The loaded modules contain the expected symbols:

```text
uvm_polaris_filter_replayable_faults [nvidia_uvm]
polaris_uvm_handle_gpu_fault         [polaris]
```

## Why Raw CUDA VMM VA Does Not Work

The explicit POLARIS KV allocation path uses CUDA VMM:

```text
cuMemAddressReserve -> POLARIS VA
cuMemCreate         -> physical GPU allocation
cuMemMap            -> map backing into that VA
cuMemSetAccess      -> set access permissions
```

For M4 we intentionally skip the map step with
`POLARIS_RESERVE_FLAG_DEFER_FAULT` and launch a CUDA kernel that writes the
unmapped VA.

Observed result:

```text
polaris_touch_kernel synchronize failed: an illegal memory access was encountered
raw CUDA VMM VA was not replayed through POLARIS; check dmesg for NVIDIA Xid/MMU fault
deferred block id=5 va=0x320000000
EXIT:1
```

Kernel log:

```text
NVRM: Xid (...): 31, pid=..., name=polaris_uvm_fau, channel 0x00000002,
MMU Fault: ENGINE GRAPHICS GPC0 GPCCLIENT_T1_0 faulted @ 0x3_20000000.
Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_WRITE
```

POLARIS stats after the run:

```text
sessions:     0
blocks:       0
pending:      0
pending_decs: 0
```

No `ALLOC` decision is queued by the real GPU access. If the replayable UVM hook
had received the fault, `polaris_uvm_handle_gpu_fault()` would have matched the
registered POLARIS VA range and queued a decision for the reserved block. The
absence of pending decisions, combined with the RM Xid 31 fault, indicates that
the raw CUDA VMM hole is not being serviced through the UVM replayable path.

## NVIDIA UVM/RM Boundary

The NVIDIA source separates fault handling into several paths:

- Replayable Graphics faults are owned by UVM after
  `nvUvmInterfaceOwnPageFaultIntr(..., NV_TRUE)`. UVM reads the replayable
  fault buffer and issues replays.
- Non-replayable faults are owned/split with RM through a shadow buffer. UVM's
  own comments describe CE/PBDMA non-replayable faults as "fault and
  reschedule", not the Graphics replay path.
- PRI/fatal MMU faults and some fault-buffer-disabled cases are handled in RM
  code such as `kgmmuServicePriFaults_*` and may lead to channel RC/Xid rather
  than recoverable UVM replay.

The UVM replayable servicing code also expects faults to be associated with a
UVM `va_space` and a UVM-managed VA range, HMM block, or ATS-serviceable CPU
mapping. A bare CUDA VMM-reserved address range is mapped by CUDA driver page
tables, but it is not a UVM VA range with UVM residency metadata. When the PDE
is missing, RM reports it as a fatal MMU fault instead of giving UVM a
serviceable managed-memory page fault.

## Feasible Paths

### A. Explicit POLARIS Pager

This is the recommended course-project path.

llama.cpp uses POLARIS-managed CUDA VMM VA for KV cache. Before graph execution,
the runtime maps/reloads the blocks the graph will touch. After graph execution,
POLARIS offloads cold blocks according to a resident budget.

This path is already partially implemented and benchmarkable. It still has the
important OS-style properties:

- POLARIS owns the global logical block table.
- The kernel owns decisions, accounting, and future multi-process policy.
- The target process runtime performs CUDA VMM map/unmap in the correct CUDA
  context.
- eBPF and `/sys/kernel/polaris/stats` can trace policy behavior.

### B. UVM-Managed VA Substrate

Use `cudaMallocManaged`/HMM/pageable-memory access to force true UVM replayable
faults, then try to attach POLARIS policy to those faults.

Problem: CUDA VMM `cuMemMap` cannot directly back an arbitrary managed-memory
VA range. UVM owns the VA blocks, residency, migration, and page tables. Making
POLARIS replace that backing would require private UVM internals, not public
CUDA VMM APIs.

This path may prove that the hook works for genuine UVM faults, but it does not
directly produce a POLARIS-managed KV VA/backing allocator.

### C. Lower RM/PRI Fault Hook

Hook the RM path that emits the Xid 31 `FAULT_PDE` event.

Problem: reaching that path is already fatal for the channel in the observed
case. The open-source RM code does not expose a simple "map then replay this
Graphics work" continuation equivalent to the UVM replayable fault path. A hook
there may observe or log the fault, but it is unlikely to recover execution.

### D. Driver-Private UVM VA Range Integration

Teach UVM that a POLARIS CUDA VMM VA range is a serviceable managed range, or
create a UVM VA range whose backing is delegated to POLARIS.

This is closest to the original M4 design, but it requires invasive changes to
UVM VA range/block internals and the CUDA user-mode driver's assumptions. It is
well beyond a narrow hook.

## Current Recommendation

Keep M4 as:

```text
M4a: kernel decision path driven by synthetic resolver smoke       done
M4b: true raw CUDA VMM replayable-fault diagnostic                 done, negative
M4c: optional managed-memory/UVM fault hook proof                  future
M4d: production benchmark path via explicit prefetch/offload       active
```

Do not claim that POLARIS currently implements automatic replayable GPU page
fault paging for raw CUDA VMM VA. Claim instead that POLARIS implements an
OS-style explicit KV pager with the same kernel-to-runtime decision protocol
that a future replayable-fault hook would use.

## Reproduction Commands

Build diagnostic:

```bash
/usr/local/cuda/bin/nvcc polaris-runtime/tests/uvm_fault_smoke.cu \
  -Ipolaris-runtime/include -Ltarget/debug -lpolaris_runtime -lcuda \
  -o /tmp/polaris_uvm_fault_smoke
```

Run diagnostic:

```bash
LD_LIBRARY_PATH=/home/wano/workspace/Polaris/target/debug:$LD_LIBRARY_PATH \
  /tmp/polaris_uvm_fault_smoke
```

Inspect kernel log:

```bash
sudo dmesg | tail -n 80
```

Verify loaded hook symbols:

```bash
sudo grep -E 'uvm_polaris_filter_replayable_faults|polaris_uvm_handle_gpu_fault' /proc/kallsyms
```
