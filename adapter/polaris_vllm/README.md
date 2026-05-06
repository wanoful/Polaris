# POLARIS vLLM Integration Adapter (Phase 4e)

Drop-in replacement for vLLM v1's KV cache manager that delegates block
allocation to the POLARIS kernel module.

## Quick Start

### 1. Ensure POLARIS kernel module is loaded

```bash
sudo insmod /path/to/polaris.ko
ls /dev/polaris   # should exist
```

### 2. Install the adapter

```bash
cd /path/to/Polaris/adapter/vllm
pip install -e .          # or copy into your PYTHONPATH
```

### 3. Run vLLM with POLARIS backend

```bash
vllm serve meta-llama/Llama-2-7b-hf \
  --scheduler-cls polaris_vllm.scheduler.PolarisScheduler
```

That's it — no source patches, no fork of vLLM.

## What it does

| vLLM operation | POLARIS equivalent |
|----------------|-------------------|
| `allocate_slots()` | `POLARIS_BLOCK_GROW` (page-fault driven allocation) |
| `free()` | `POLARIS_SESSION_DESTROY` (kernel reclaims all blocks) |
| Prefix caching | Handled by vLLM (unchanged) |
| Sliding window | Handled by vLLM (unchanged) |
| Swap-out / swap-in | No-op — POLARIS kernel handles offload/reload transparently |

## Architecture

```
vLLM Scheduler
      │
      ▼
PolarisScheduler  ←  --scheduler-cls polaris_vllm.scheduler.PolarisScheduler
      │
      ├─ instantiate PolarisKVCacheManager
      └─ keep all other scheduler logic unchanged
      │
      ▼
PolarisKVCacheManager  ←  replaces KVCacheManager
      │
      ├─ allocate_slots()  →  super() + POLARIS_BLOCK_GROW
      ├─ free()            →  super() + POLARIS_SESSION_DESTROY
      └─ everything else   →  forwarded to parent KVCacheManager
      │
      ▼
  /dev/polaris  →  polaris.ko  →  polarisd  →  CUDA VMM
```

## Design Rationale

### Why not a git patch?

vLLM v1's `KVCacheManager` changes frequently (every release touches prefix
caching, block layout, or the coordinator).  Maintaining a patch per commit is
fragile and creates a heavy ongoing burden.

### Why not a vLLM fork?

A fork permanently diverges from upstream.  Security fixes, performance
improvements, and new model support in upstream vLLM would need to be
cherry-picked manually.

### Why `--scheduler-cls`?

vLLM v1 exposes `scheduler_cls` as a **first-class configuration option**
(see `vllm/config/scheduler.py`).  It is the intended extension point for
custom schedulers.  vLLM even logs a warning when a custom scheduler is used,
achnowledging that this is the supported mechanism.

By subclassing `Scheduler` and only overriding the `kv_cache_manager`
instantiation, we touch **zero** lines of vLLM core code.

## Scope and Limitations

This adapter is intentionally scoped to Phase-4e (smoke test / optional
integration).  It demonstrates that vLLM can run end-to-end with POLARIS
managing KV cache blocks.

Current limitations:
* **Prefix caching** — still handled by vLLM's native `BlockPool`.  A future
  revision could redirect prefix-cache hits to POLARIS COW sharing.
* **Multi-GPU** — `home_gpu` is hard-coded to `0`.  Multi-GPU routing is out
  of scope for the current POLARIS roadmap.
* **Block size assumptions** — assumes all KV cache groups share the same
  block size (true for standard models, but not enforced).
* **Error recovery** — if POLARIS_BLOCK_GROW fails, the exception propagates
  and vLLM's scheduler will preempt the request on the next step.

## Troubleshooting

### `/dev/polaris` not found

The adapter falls back to stock vLLM behaviour when the device is
unavailable.  Check that the kernel module is loaded:

```bash
lsmod | grep polaris
sudo dmesg | tail -n 20
```

### ImportError: cannot import name 'PolarisScheduler'

Ensure the adapter package is on your `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/Polaris/adapter/vllm:$PYTHONPATH"
```

### POLARIS_BLOCK_GROW fails with ENOMEM

GPU memory is exhausted.  POLARIS's phase-aware eviction policy should
offload blocks to CPU automatically.  If CPU pool is also full, the block
is evicted and vLLM will recompute.  Check stats:

```bash
cat /sys/kernel/polaris/stats
```

## Files

| File | Purpose |
|------|---------|
| `polaris_abi.py` | ctypes structs and ioctl wrappers for `/dev/polaris` |
| `polaris_kv_cache_manager.py` | Drop-in `KVCacheManager` replacement |
| `polaris_scheduler.py` | Minimal `Scheduler` subclass that swaps the manager |
| `__init__.py` | Package exports |
| `README.md` | This file |

## Success Criterion

vLLM serves at least one complete inference request end-to-end with
`PolarisScheduler` active and `nvidia-smi` showing memory managed by
POLARIS (via the `polarisd` CUDA VMM path).
