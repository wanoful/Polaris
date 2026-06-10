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
3. Hands the duped RM VA-space handle (the `va_space_token` in the v4 ABI)
   to `polaris.ko` via the `POLARIS_REGISTER_VASPACE` ioctl.
4. Intercepts `cuMemAlloc` / `cudaMalloc` and the PyTorch / llama.cpp
   allocator backends to reserve VA inside the POLARIS VA-space — without
   mapping it — and registers the range with `polaris.ko` so the fault
   hook can resolve `addr → block_id`.
5. Intercepts `cuMemFree` / `cudaFree` to deregister the range.

## Current state (M0 scaffolding)

This commit only stands up the build skeleton and a CUDA driver-API loader
that resolves the real `cuInit` symbol and passes the call through. It is
deliberately a no-op interceptor — proves the LD_PRELOAD shape end-to-end
before any UVM glue. Run with:

    LD_PRELOAD=$(realpath ./libpolaris-shim.so) ./your-cuda-app

and look for the single `[polaris-shim] cuInit intercepted` line on stderr.

## Why this is not a Cargo crate

CUDA workers expect to `dlsym` C symbols out of the preloaded `.so`. A C
shared library is the path of least friction. The shim links libpolaris
(Rust) once the M2 ioctl surface lands; until then it is freestanding.

## Build

    make            # produces ./libpolaris-shim.so
    make clean
