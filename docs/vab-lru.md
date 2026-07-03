# VAB-based access LRU (opt-in)

POLARIS can drive KV-cache LRU eviction from a real hardware **local-vidmem
access signal** using the GPU's Vidmem Access Bit Buffer (VAB, RM class
`MMU_VIDMEM_ACCESS_BIT_BUFFER`/`NVC763`). This removes the need for an explicit
userspace "touch" call and captures accesses to already-resident blocks that
never take a replayable fault.

The feature is **opt-in and OFF by default**. With it off, UVM behaves exactly
like stock (VAB is never allocated) and `last_touch_ns` is updated only on
map/reload as before.

## Why it exists

GPU access counters only fire for *remote* accesses (migration candidates), so
they never observe POLARIS's local-vidmem KV blocks. VAB tracks access bits for
*local* framebuffer regions, which is exactly the recency signal LRU needs.

## Enabling

Load the patched `nvidia-uvm` with the module params:

```sh
sudo insmod nvidia-uvm.ko uvm_polaris_vab_lru=1 uvm_polaris_vab_interval_ms=250
```

- `uvm_polaris_vab_lru` — `0` (default) off, `1` on. Gates VAB allocation and
  the poll. Must be set at module load.
- `uvm_polaris_vab_interval_ms` — poll/dump interval (default `250`).

Then run POLARIS with the LRU eviction policy
(`POLARIS_BENCH_EVICTION_POLICY=lru`). Verify via `/sys/kernel/polaris/stats`:
`uvm_vab_dumps` and `uvm_vab_touches` should be non-zero under load.

## How it works

1. On GPU registration, UVM allocates the VAB (only when `uvm_polaris_vab_lru=1`
   and the HAL reports `access_bits_supported`). Default granularity is 4 MB;
   bit *N* of the dump covers physical FB region `[N*region_bytes, ...)`.
2. A UVM workqueue dumps the access bitmap every `interval_ms` and forwards it to
   `polaris.ko` via `uvm_polaris_ops.handle_access_bits`.
3. `polarisd` records each block's physical FB offset (returned free by the
   NVOS32 vidmem allocation in `params.offset`) and passes it to the kernel via
   `PolarisCompleteOperationArg.phys_fb_addr`.
4. `polaris.ko` maps each resident block's physical address to its region bit; if
   set, it refreshes `last_touch_ns`. Blocks whose region is not set age toward
   eviction.

## Hardware support

Validated on **GB203 (RTX 5070 Ti)**. The Blackwell HAL keeps VAB disabled on
GB206/GB207 (known GSP issues) and integrated GPUs even when the param is set.
Other architectures are not enabled by this feature.

## Caveats / status

This is **experimental**. Benchmarking (Qwen2.5-14B, 16k pressure) shows that a
real recency signal makes LRU **diverge** from FIFO — but LRU performs *worse*
for KV cache: full attention reads every resident block each decode step, so
recency does not discriminate well and LRU thrashes (≈2.3× more offload/reload,
lower throughput). `phase_aware`/`fifo` remain the better defaults. The feature
is useful for research and for workloads with non-uniform KV reuse; it is not
recommended as the default eviction policy.

At 4 MB granularity one region covers two adjacent 2 MB blocks. Finer resolution
would require configuring VAB range checkers via `ENABLE_LOGGING` (not yet done).
