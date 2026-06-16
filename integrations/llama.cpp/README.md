# llama.cpp POLARIS v4 Integration

The v4 llama.cpp path is an unmodified-worker, KV-cache-only path: build
llama.cpp normally, load `polaris.ko`, start `polarisd` with daemon-owned RM
backing, and run the binary with `libpolaris-shim.so` in `LD_PRELOAD`. The shim
creates a fault-capable RM/UVM VA-space, registers the same RM client and user
VA-space handles with `polaris.ko`, and intercepts CUDA allocation calls so
only selected KV-cache buffers are returned from the Polaris-managed VA window.
Model weights, model upload buffers, CUDA workspaces, and unrelated
allocations stay on llama.cpp's normal CUDA path.

This replaces the older source-patch plan that linked `polaris-runtime` into
llama.cpp and used raw CUDA VMM reservations. Raw CUDA VMM VA is not
fault-capable, so it remains diagnostic/pass-through surface only.

## Target Allocation Paths

The shim currently covers the llama.cpp CUDA allocation paths needed for M5:

- `cudaMalloc` / `cudaFree`
- `cudaMallocManaged` / `cudaFree`, for builds using
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY`
- `cudaMallocAsync` / `cudaFreeAsync`
- `cuMemAlloc_v2` / `cuMemFree_v2`
- `cuMemAllocAsync_v2` / `cuMemFreeAsync_v2`

Use the ggml KV-scope policy to avoid capturing copied model/init buffers, and
use the size policy only as an additional guardrail around expected KV sizes:

```sh
POLARIS_SHIM_REQUIRE_KV_SCOPE=1
POLARIS_SHIM_ALLOW_ZERO_MEMSET=1
POLARIS_SHIM_MIN_MANAGED_ALLOC=<bytes>
POLARIS_SHIM_MAX_MANAGED_ALLOC=<bytes>
POLARIS_SHIM_STRICT_MANAGED_ALLOC=1
POLARIS_SHIM_REPORT_STATS=1
```

Strict mode is important for experiments: if an in-policy allocation cannot be
served by the Polaris allocator, the allocation fails instead of silently
falling back to normal CUDA memory.

## CUDA API Compatibility

The shim forwards common setup/query APIs used by llama.cpp, including device
selection, runtime/driver version checks, memory-info probes, stream/event
lifecycle calls, pinned-host memory APIs, and kernel function query helpers.
It also forwards runtime and driver kernel-launch entry points, including
`cudaLaunchKernelExC` for llama.cpp's CUDA 11.8+ PDL path.
Driver device/context helpers (`cuDeviceGet`, `cuDeviceGetAttribute`,
`cuDevicePrimaryCtxRetain`, and `cuCtxSetCurrent`), PCI/peer helpers, host
launch callbacks, cooperative launch, occupancy-potential queries, and graph
node inspection/update helpers are pass-throughs as well.

The shim also forwards CUDA driver VMM pool primitives used by llama.cpp's CUDA
backend:

- `cuMemAddressReserve`
- `cuMemAddressFree`
- `cuMemCreate`
- `cuMemRelease`
- `cuMemMap`
- `cuMemUnmap`
- `cuMemSetAccess`
- `cuMemGetAllocationGranularity`

These VMM calls are pass-through compatibility surfaces. They do not create
Polaris-managed, fault-capable memory.

## Guarded Surfaces

CUDA copies/fills involving Polaris pointers currently return
`cudaErrorNotSupported`, except for the explicit KV zero-fill contract enabled
by `POLARIS_SHIM_ALLOW_ZERO_MEMSET=1`. This is intentional: POLARIS v4 is not
trying to page arbitrary CUDA buffers, and model-weight upload should never use
Polaris pointers. Daemon-backed KV spill/reload copies use `POLARIS_RM_COPY`
through UVM-owned staging, not ordinary CUDA host copy APIs.

CUDA IPC export for Polaris pointers is also rejected.

CUDA Graph capture and launch are guarded while Polaris allocations are live:
`cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphLaunch`,
`cuStreamBeginCapture`, and `cuStreamBeginCapture_v2` return the stream-capture
unsupported error. For llama.cpp M5 runs, disable CUDA Graph mode or keep
Polaris-managed KV ranges out of graph capture/replay.

## Safe Bring-Up Smoke

From the Polaris repo, build the shim and kernel:

```sh
make -C libpolaris-shim clean all tests
make kernel KDIR=/lib/modules/$(uname -r)/build
```

Run the non-dereferencing shim smoke before testing a real workload:

```sh
sudo rmmod polaris 2>/dev/null || true
sudo insmod kernel/polaris.ko
sudo env \
  POLARIS_SHIM_BOOTSTRAP_RM_UVM=1 \
  POLARIS_SHIM_TEST_RUNTIME_SETUP=1 \
  POLARIS_SHIM_TEST_HOST_APIS=1 \
  POLARIS_SHIM_TEST_MANAGED_ALLOC=1 \
  POLARIS_SHIM_TEST_ADVISE=1 \
  POLARIS_SHIM_TEST_LARGE_WINDOW=1 \
  POLARIS_SHIM_TEST_ATTRS=1 \
  POLARIS_SHIM_TEST_MEMCPY=1 \
  POLARIS_SHIM_TEST_GRAPH=1 \
  POLARIS_SHIM_TEST_VMM=1 \
  POLARIS_SHIM_TEST_LAUNCH=1 \
  POLARIS_SHIM_REPORT_STATS=1 \
  POLARIS_SHIM_STRICT_MANAGED_ALLOC=1 \
  POLARIS_SHIM_TRANSIENT_GPU=1 \
  POLARIS_SHIM_GPU_ID=0 \
  POLARIS_SHIM_BLOCK_SIZE=0x200000 \
  LD_PRELOAD=$PWD/libpolaris-shim/libpolaris-shim.so \
  libpolaris-shim/build/managed_alloc
cat /sys/kernel/polaris/stats
sudo rmmod polaris
```

This smoke does not launch kernels, does not map raw CUDA VMM memory, and does
not dereference Polaris GPU pointers. The launch checks verify symbol coverage
without invoking CUDA launch entry points.

## M5 Integration Regression

The repository includes a root/GPU integration test for the current llama.cpp
state. It separates the optional POLARIS backend check from the LD_PRELOAD shim
fault-path gate:

- if the local llama.cpp binary exposes `POLARIS0`, a real `llama-bench`
  workload runs on that backend and must complete;
- the LD_PRELOAD shim probes run unmodified CUDA `llama-bench` workloads for
  both the default `cudaMalloc` path and the
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` / `cudaMallocManaged` path. Each probe
  bootstraps RM/UVM, registers a Polaris VA-space, routes real llama allocation
  through Polaris, creates a UVM external range, starts `polarisd` with
  `POLARISD_RM_BACKING=1`, requires daemon-owned RM backing to be published
  through `POLARIS_COMPLETE_OPERATION`, requires the matching shim per-API counter
  (`api_runtime_alloc_selected` or
  `api_runtime_managed_alloc_selected`) to increase, and requires
  `/sys/kernel/polaris/stats` to show increased `uvm_hook_calls` and
  `uvm_handled`;
- when the allocator policy selects a copied model/init buffer, the shim probe
  is expected to stop cleanly at the guarded CUDA host-copy surface rather than
  letting libcuda touch Polaris external VA from the wrong context. The current
  passing no-source-change path selects KV allocations through the ggml KV
  scope hook, accepts their zero-fill initialization, and lets their first real
  kernel access fault through UVM.

Run it from this repository after loading the patched NVIDIA UVM module and
`polaris.ko`:

```sh
make -C libpolaris-shim all tests
sudo -E make llama-e2e \
  LLAMA_CPP_DIR=/home/wano/workspace/llama.cpp \
  LLAMA_CPP_MODEL=/home/wano/workspace/models/SmolLM2-135M-Instruct-F16.gguf
```

If `LLAMA_CPP_MODEL` is omitted, the script tries the local SmolLM2 model and
then llama.cpp's tiny stories model. The default binary is `llama-bench` from
`build-polaris/tools`, then the usual `build*/bin`/`tools` fallbacks. Override
with `LLAMA_CPP_BIN=/path/to/binary`.

`POLARIS_SHIM_STATIC_RM_BACKEND=1` remains available as a diagnostic backend,
but the llama regression no longer uses it by default. The production-shaped
gate leaves selected shim allocations as deferred logical blocks. A single
llama.cpp KV allocation is still returned as one contiguous GPU VA range, but
the shim reserves and registers one Polaris logical block per
`POLARIS_SHIM_BLOCK_SIZE` chunk inside that range. A first GPU access to each
chunk faults into polaris.ko, the kernel queues `ALLOC`, `polarisd` allocates
daemon-owned RM vidmem and returns the RM tuple through
`POLARIS_COMPLETE_OPERATION`, and the same fault is bridge-mapped through UVM
before returning `HANDLED`. Daemon-backed `OFFLOAD` and `RELOAD` use
`POLARIS_RM_COPY`; overwrite-reserve RM-backed `COW_BREAK` uses the same copy
primitive with CPU staging. Permission-based write-fault COW remains future M4
hardening.

Set `POLARIS_LLAMA_STRICT_SHIM_FAULT_PASS=1` to require that the shimmed
`llama-bench` commands complete and `/sys/kernel/polaris/stats` shows both
`uvm_hook_calls` and `uvm_handled` increasing for the default and
unified-memory allocator branches. The gate sets
`GGML_CUDA_DISABLE_GRAPHS=1` and `GGML_CUDA_PDL=0` by default because CUDA
Graph capture/replay and programmatic dependent launch are not part of the
current shim contract.

## Workload Run Shape

Build llama.cpp with CUDA enabled using its normal upstream options. Then run
with the shim preloaded:

```sh
sudo env \
  POLARIS_SHIM_BOOTSTRAP_RM_UVM=1 \
  POLARIS_SHIM_STRICT_MANAGED_ALLOC=1 \
  POLARIS_SHIM_REPORT_STATS=1 \
  POLARIS_SHIM_REQUIRE_KV_SCOPE=1 \
  POLARIS_SHIM_ALLOW_ZERO_MEMSET=1 \
  POLARIS_SHIM_GPU_ID=0 \
  POLARIS_SHIM_BLOCK_SIZE=0x200000 \
  POLARIS_SHIM_MIN_MANAGED_ALLOC=<kv-floor-bytes> \
  LD_PRELOAD=/home/wano/workspace/Polaris/libpolaris-shim/libpolaris-shim.so \
  /path/to/llama.cpp/build/bin/<llama-binary> <args>
```

The current regression validates llama.cpp's actual KV allocation and
kernel-deref path for both runtime `cudaMalloc` and runtime
`cudaMallocManaged` through daemon-owned RM backing. Set
`POLARIS_LLAMA_RUN_PRESSURE_PROBE=1` to add the small-budget pressure gate:
the script restarts `polarisd` with
`POLARIS_LLAMA_PRESSURE_BUDGET_BYTES` (default 4 MiB), mirrors that budget and
the CPU pool into the shim-registered transient GPU, runs a real CUDA
`llama-bench` workload, and requires daemon-backed `offloads`, `reloads`,
`uvm_bridge_map_calls`, and `uvm_bridge_map_ok` to increase without
`uvm_no_pte` or `uvm_errors`. Longer term, replace the bounded managed-window
allocator with VA management that is aware of llama.cpp KV lifecycle events
such as context growth, context shift, prompt cache reuse, and long-running
server churn.

Set `POLARIS_LLAMA_RUN_SUSTAINED_PRESSURE_PROBE=1` to add a longer version of
the same KV-only pressure gate. By default it runs the pressure probe with
`POLARIS_LLAMA_SUSTAINED_PROMPT_TOKENS=256`,
`POLARIS_LLAMA_SUSTAINED_GEN_TOKENS=32`, and
`POLARIS_LLAMA_SUSTAINED_REPETITIONS=2`; override those values to scale the
run up or down for local hardware. The gate still selects only ggml KV-scope
allocations and still requires daemon-backed offload/reload plus stable UVM
error counters.
