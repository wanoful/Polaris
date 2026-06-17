# AGENTS.md

This file gives coding agents the project-specific operating rules for this
repository. It applies to the whole tree unless a more specific `AGENTS.md`
exists in a subdirectory.

## Project Scope

POLARIS v4 is a Linux kernel module, patched NVIDIA UVM path, LD_PRELOAD shim,
and userspace daemon for KV-cache-only GPU memory paging.

Do not describe or implement POLARIS as a general transparent CUDA memory
pager. The supported production-shaped path is narrower:

- model weights, upload buffers, CUDA workspaces, and ordinary CUDA buffers
  stay on the normal CUDA path;
- llama.cpp KV-cache allocations are selected by the shim's KV scope and routed
  through POLARIS-managed, fault-capable GPU virtual addresses;
- pressure/offload uses daemon-owned RM backing and `POLARIS_RM_COPY`;
- vLLM/SGLang support is currently benchmark/trace comparison and future
  allocator-backend work, not live POLARIS execution.

## Local Repository Layout

Primary repo:

- `kernel/`: Rust-for-Linux `polaris.ko` module and policy code.
- `polarisd/`: daemon that executes RM allocation, offload, reload, and COW
  copy decisions.
- `polarisctl/`: userspace control CLI.
- `libpolaris/`: ioctl ABI and Rust userspace bindings.
- `libpolaris-shim/`: LD_PRELOAD CUDA shim for llama.cpp KV selection.
- `integrations/llama.cpp/`: llama.cpp integration notes and test contract.
- `tests/m2/`: root/GPU UVM/RM bridge and hardening tests.
- `tests/llama_cpp/`: llama.cpp shim e2e tests.
- `benchmarks/`: llama.cpp/POLARIS benchmark harness plus vLLM/SGLang trace
  comparison tooling.
- `docs/roadmap-v4.md`: current architecture and milestone roadmap.

Expected sibling repos on this machine:

- `../open-gpu-kernel-modules`: patched NVIDIA driver/UVM tree.
- `../llama.cpp`: local llama.cpp integration tree.
- `../vllm`: local vLLM baseline repo.
- `../sglang`: local SGLang baseline repo. Note the directory is lowercase;
  `../SGLang` is not expected to exist here.

Prefer these paths in commands unless the user explicitly overrides them.

## Build And Test Commands

Use focused checks first. Kernel/GPU gates require root and the patched NVIDIA
modules.

Common non-root checks:

```sh
cargo check -p polarisd
cargo test -p libpolaris --tests --no-run
make -C libpolaris-shim all tests NVIDIA_KO_DIR=../open-gpu-kernel-modules
bash -n benchmarks/scripts/run_llama_kv_bench.sh
python3 -m py_compile benchmarks/scripts/compare_kv_results.py
```

Kernel build:

```sh
make kernel NVIDIA_KO_DIR=../open-gpu-kernel-modules
```

Strict llama.cpp e2e gate:

```sh
sudo -E make llama-e2e \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules \
  LLAMA_CPP_DIR=../llama.cpp
```

Longer pressure gate:

```sh
sudo env \
  POLARIS_LLAMA_STRICT_SHIM_FAULT_PASS=1 \
  POLARIS_LLAMA_RUN_DYNAMIC_WINDOW_PROBE=1 \
  POLARIS_LLAMA_RUN_PRESSURE_PROBE=1 \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules \
  LLAMA_CPP_DIR=../llama.cpp \
  bash tests/llama_cpp/run_llama_shim_e2e.sh
```

M6 stress gates:

```sh
sudo -E make m6-daemon-rm-soak NVIDIA_KO_DIR=../open-gpu-kernel-modules
sudo -E make m6-module-unload-stress NVIDIA_KO_DIR=../open-gpu-kernel-modules
```

llama.cpp/POLARIS benchmark:

```sh
sudo env \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules \
  LLAMA_CPP_DIR=../llama.cpp \
  LLAMA_CPP_BIN=../llama.cpp/build/bin/llama-bench \
  POLARIS_BENCH_PROMPTS=128,512 \
  POLARIS_BENCH_GENS=32,128 \
  POLARIS_BENCH_REPETITIONS=3 \
  POLARIS_BENCH_MODES=native_cuda,polaris_no_pressure,polaris_pressure \
  POLARIS_BENCH_EVICTION_POLICY=phase_aware \
  POLARIS_LLAMA_KV_HINTS=1 \
  benchmarks/scripts/run_llama_kv_bench.sh
```

Detailed daemon profiling for pressure runs:

```sh
sudo env \
  NVIDIA_KO_DIR=../open-gpu-kernel-modules \
  LLAMA_CPP_DIR=../llama.cpp \
  LLAMA_CPP_BIN=../llama.cpp/build/bin/llama-bench \
  POLARIS_BENCH_PROMPTS=512 \
  POLARIS_BENCH_GENS=128 \
  POLARIS_BENCH_REPETITIONS=1 \
  POLARIS_BENCH_MODES=polaris_pressure \
  POLARIS_BENCH_EVICTION_POLICY=phase_aware \
  POLARIS_LLAMA_KV_HINTS=1 \
  POLARISD_PROFILE_DECISIONS=1 \
  benchmarks/scripts/run_llama_kv_bench.sh
```

`POLARISD_PROFILE_DECISIONS=1` records per-decision stage timing in daemon
logs and writes aggregated `polarisd_profile` data into `runs.jsonl`.

## Benchmark Interpretation

Be precise when comparing engines:

- `llama.cpp + POLARIS` benchmark results are end-to-end llama.cpp runs using
  the KV-only shim.
- vLLM/SGLang artifacts under `benchmarks/results/frameworks/` are native
  framework baselines unless explicitly stated otherwise.
- Current vLLM/SGLang comparison is not a live POLARIS backend A/B. Treat it as
  cross-engine throughput context or KV allocator trace comparison.
- `llama_avg_ts` averages llama-bench prompt and generation rows. It is not the
  same metric as vLLM/SGLang total tokens/sec.
- Decode/output tokens/sec is often the more relevant comparison for generation
  performance.

Known current pressure-profile behavior:

- In small-budget pressure runs, most daemon time is in RM-backed copy stages:
  `rm_offload_copy_to_cpu_ioctl` and `rm_reload_copy_from_cpu_ioctl`.
- eBPF/kprobe can confirm whether time is inside UVM helpers such as
  `uvm_polaris_copy_external_allocation`, but daemon profile should be the
  first layer because it preserves POLARIS decision semantics.

## Code Conventions

- Keep changes scoped to the requested behavior. Do not reformat unrelated
  files or run workspace-wide formatters that churn existing code.
- Use `rg` / `rg --files` for search.
- Use `apply_patch` for manual edits.
- Preserve the kernel/userspace ABI. If any ioctl struct or constant changes,
  update all mirrored ABI definitions together:
  - `kernel/polaris_abi.rs`
  - `kernel/polaris_types.rs`
  - `libpolaris/src/ioctl.rs`
  - `libpolaris-shim/include/polaris_abi.h`
- Keep the llama.cpp path KV-only. Do not relax guarded CUDA copy/fill behavior
  for arbitrary POLARIS pointers unless the task explicitly requires changing
  that contract.
- When touching pressure/offload code, track both correctness counters
  (`offloads`, `reloads`, `uvm_errors`, `uvm_no_pte`) and performance counters
  (`uvm_bridge_map_calls`, `uvm_bridge_map_avg_ns`, `polarisd_profile`).
- Rust kernel code may not follow ordinary userspace Rust formatting. Prefer
  focused checks over broad `cargo fmt` unless the user asks for formatting.

## Git And Workspace Safety

The worktree may already contain user or prior-agent changes. Do not revert
unrelated changes. Before editing, inspect the relevant files and work with the
current state.

Never run destructive git commands such as `git reset --hard` or
`git checkout --` unless explicitly requested.

Generated benchmark artifacts can be useful evidence. Do not delete them unless
the user asks for cleanup.
