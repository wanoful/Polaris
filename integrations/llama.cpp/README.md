# llama.cpp POLARIS Integration Plan

This integration links `polaris-runtime` into llama.cpp and routes CUDA KV
cache storage through a process-local CUDA VMM VA range managed by POLARIS.

## Why llama.cpp First

llama.cpp owns its ggml CUDA buffers and KV cache directly. That makes it a
better first integration target than vLLM or SGLang, where most allocation
policy sits above CUDA in PyTorch-oriented managers. The first comparison should
therefore be:

- llama.cpp baseline vs. llama.cpp + POLARIS for live tokens/s and memory.
- vLLM/SGLang native managers vs. POLARIS trace replay for allocator behavior.

## Patch Points

The useful upstream files are:

- `src/llama-kv-cache.cpp`
  - KV tensors are created in the cache constructor.
  - Accessors such as `get_k`, `get_v`, `cpy_k`, and `cpy_v` should keep seeing
    stable contiguous GPU VAs.
- `ggml/include/ggml-backend.h`
  - Backend buffer API boundary used by KV tensor allocation.
- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - CUDA backend allocation path.
  - Existing CUDA VMM pool code is the closest local model for POLARIS-backed
    virtual allocation.

## Minimal First Version

1. Add a CMake option, for example `LLAMA_POLARIS=ON`.
2. Include `polaris_runtime.h` and link `libpolaris_runtime`.
3. Initialize `polaris_runtime_t` after the CUDA backend has selected the
   device/context.
4. Reserve a POLARIS VA span large enough for KV cache tensors and expose the
   returned `va_base` to the KV allocation path.
5. Register session/block metadata through existing POLARIS ioctls when
   llama.cpp reserves KV token ranges.
6. Start the runtime thread or call `polaris_runtime_poll_once()` at safe points.
7. Preserve ggml tensor pointer arithmetic: tensors should point into the
   reserved virtual span even when physical GPU memory is not mapped yet.

## Build Sketch

From the POLARIS repo:

```sh
cargo build -p polaris-runtime
```

From the llama.cpp repo:

```sh
cmake -B build-polaris \
  -DLLAMA_CUDA=ON \
  -DLLAMA_POLARIS=ON \
  -DPOLARIS_ROOT=/home/wano/workspace/Polaris
cmake --build build-polaris -j
```

The first patch should keep the integration optional and compile-time gated.
If `LLAMA_POLARIS` is off, llama.cpp must use its existing CUDA allocation path
unchanged.

## Correctness Notes

- `polarisd` must not call `cuMemMap` on behalf of llama.cpp. CUDA VMM mappings
  are context-local, so the executor runs inside llama.cpp.
- Multiple processes may reserve overlapping numeric GPU VA values. The kernel
  fault path ultimately needs process/context identity in addition to the fault
  address. Address-only lookup is enough for single-process bring-up, not for
  the final multi-process claim.
- Standard beam search mostly benefits from refcounted prefix sharing. COW break
  remains a correctness path for explicit overwrite/speculative cases, but it
  does not need a cross-process COW manager for the first llama.cpp prototype.
