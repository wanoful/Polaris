# POLARIS

Paged Operating Layer for Accelerated Routing and Inference Systems.

POLARIS v4 is a Linux kernel module, patched NVIDIA UVM path, LD_PRELOAD shim,
and userspace daemon for OS-level KV-cache paging. The production target is
not arbitrary CUDA-buffer virtualization. The target is narrower: keep model
weights and ordinary CUDA buffers on their normal CUDA path, while routing
LLM KV-cache storage through Polaris-managed, fault-capable GPU virtual
addresses.

## Current v4 Path

For llama.cpp, the supported path is:

```text
llama.cpp model weights / init buffers
  -> normal CUDA allocator and normal CUDA copies

llama.cpp KV-cache allocation
  -> libpolaris-shim selects ggml KV scope only
  -> shim returns one stable contiguous Polaris VA
  -> internally the range is split into Polaris-sized KV chunks
  -> first GPU kernel access to each chunk triggers a UVM replayable fault
  -> polaris.ko resolves the logical KV chunk
  -> polarisd allocates daemon-owned RM backing
  -> polaris.ko asks UVM to map the external allocation
  -> later pressure can spill/reload the same KV VA through POLARIS_RM_COPY
```

The shim defaults used by the llama regression enforce this contract with
`POLARIS_SHIM_REQUIRE_KV_SCOPE=1`. KV zero-fill initialization is allowed with
`POLARIS_SHIM_ALLOW_ZERO_MEMSET=1`; general host copies/fills involving
Polaris pointers remain guarded.

## Components

- **Patched NVIDIA UVM**: exports the Polaris fault hook, external-allocation
  map/unmap bridge, and RM-backed copy helpers. The active development tree is
  expected at `../open-gpu-kernel-modules` or under
  `third_party/open-gpu-kernel-modules`.
- **polaris.ko**: registers the UVM hook, owns logical KV block state,
  block-to-worker mappings, bridge-map telemetry, spill/reload decisions, and
  cleanup rules.
- **polarisd**: executes daemon-owned RM allocation/free plus RM-backed
  offload/reload/COW copies through `POLARIS_RM_COPY`.
- **libpolaris-shim.so**: bootstraps the fault-capable RM/UVM VA-space and
  selects llama.cpp KV-cache allocations without source changes.
- **tests/m2/**: root/GPU bring-up and hardening gates for the UVM bridge,
  daemon-backed RM spill/reload, COW, OOM pressure, and module unload stress.
- **integrations/llama.cpp/**: current llama.cpp operating contract and
  root/GPU regression instructions.

## Build Checks

```sh
make kernel NVIDIA_KO_DIR=../open-gpu-kernel-modules
cargo test --workspace --no-run
make -C libpolaris-shim all tests NVIDIA_KO_DIR=../open-gpu-kernel-modules
make -C tests/m2 NVIDIA_KO_DIR=../open-gpu-kernel-modules
```

Kernel and GPU tests require the matching patched NVIDIA module and root access.

## Root/GPU Gates

Run the strict llama.cpp KV fault-path regression:

```sh
sudo -E make llama-e2e \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules \
  LLAMA_CPP_DIR=../llama.cpp
```

Add the real llama.cpp small-budget KV pressure gate:

```sh
sudo env \
  POLARIS_LLAMA_STRICT_SHIM_FAULT_PASS=1 \
  POLARIS_LLAMA_RUN_DYNAMIC_WINDOW_PROBE=1 \
  POLARIS_LLAMA_RUN_PRESSURE_PROBE=1 \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules \
  LLAMA_CPP_DIR=../llama.cpp \
  bash tests/llama_cpp/run_llama_shim_e2e.sh
```

Run the daemon-backed RM soak:

```sh
sudo -E make m6-daemon-rm-soak \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules
```

Run module unload/reload stress:

```sh
sudo -E make m6-module-unload-stress \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules
```

## Status

The v4 codebase has the UVM hook, fault-capable VA-space registration,
external-range bridge mapping, daemon-owned RM backing, RM-backed
spill/reload, overwrite COW, llama.cpp KV-only shim selection, per-chunk KV
residency inside a contiguous llama.cpp allocation, an opt-in real llama.cpp
pressure gate, and focused M6 stress gates wired.

Still open:

- Longer llama.cpp server/long-context pressure coverage beyond the current
  opt-in `llama-bench` offload/reload gate.
- Workload-aware VA reclamation beyond the current bounded managed-window
  grow/shrink allocator.
- Permission-based write-fault COW for shared KV pages.
- PyTorch/vLLM allocator backend integration.
- Benchmark automation and vLLM/SGLang comparison results.

## Documentation

- [docs/roadmap-v4.md](docs/roadmap-v4.md): current v4 roadmap and milestone
  state.
- [integrations/llama.cpp/README.md](integrations/llama.cpp/README.md):
  llama.cpp KV-only integration details.
- [tests/m2/README.md](tests/m2/README.md): staged UVM/RM bridge diagnostics.
- [docs/fault-driven-analysis.md](docs/fault-driven-analysis.md): historical
  analysis showing why raw CUDA VMM holes were abandoned.
