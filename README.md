# POLARIS

Paged Operating Layer for Accelerated Routing and Inference Systems

POLARIS is a Linux kernel module + CUDA VMM daemon that provides OS-level paged KV Cache management for LLM inference. It implements kernel-directed page faults via CUDA VMM, CPU offload/reload under memory pressure, and reference-counted copy-on-write for beam search, and is benchmarked against vLLM and SGLang on real GPU hardware.

## Architecture

We can understand POLARIS as the middleware layer between the LLM inference workload and the GPU hardware, orchestrating CUDA VMM operations based on kernel-level decisions about which KV blocks to keep resident on the GPU, which to evict, and when to reload from CPU memory.

```
Inference layer (PyTorch / vLLM / synthetic workload)
      |
      |  ioctl  ("Require new KV block")
      ▼
┌──────────────┐
│  polaris.ko  │  ← kernel module: authoritative block table, page-fault
│  (kernel)    │    decisions, COW refcount, LRU state, victim selection
└──────┬───────┘
       |  ioctl  ("Execute: cuMemMap block 42 on GPU 0 at VA 0x7f...")
       ▼
┌──────────────┐
│   polarisd   │  ← userspace daemon: executes CUDA VMM operations —
│  (userspace) │    cuMemCreate, cuMemMap, cuMemUnmap, cuMemSetAccess,
│              │    cudaMemcpy (GPU↔CPU), cuMemRelease
└──────────────┘
       |
       ▼
┌──────────────┐
│  NVIDIA GPU  │  ← real hardware, managed through CUDA driver
└──────────────┘
```

## Key Components

- **polaris.ko**: Linux kernel module — maintains global block table, session state, eviction policies, and decision protocol.
- **polarisd**: Userspace daemon in Rust — executes CUDA VMM calls driven by kernel decisions.
- **polarisctl**: CLI tool for stats, session inspection, and debugging.
- **workloads/**: Synthetic KV stress, beam search, concurrent, and trace replay workloads.
- **benchmarks/**: vLLM/SGLang trace collection, replay, plotting, and automated suite.
- **adapter/** (deferred): Drop-in vLLM BlockSpaceManager replacement using POLARIS ioctls.
- **ebpf/** (deferred): XDP eBPF program for zero-copy request ingestion.

## Core Abstractions

- **KV Block**: Smallest scheduling and paging unit (~8 MB for LLaMA-2-7B, 16 tokens, FP16).
- **Session**: One LLM inference request; owns a linked list of KV blocks.
- **Page Table (Block Table)**: Kernel-authoritative mapping from `(session_id, token_range)` to `(gpu_phys_handle, gpu_vaddr, state)`.

## Quick Start

### Build

```bash
cd kernel && make && sudo insmod polaris.ko
cd ../polarisd && cargo build --release
sudo systemctl start polarisd
```

### CLI

```bash
polarisctl stats          # Show /sys/kernel/polaris/stats
polarisctl session list   # List active sessions
polarisctl debug blocks   # Dump block table
```

### Run Synthetic Workload

```bash
cd workloads && cargo run --bin synthetic_kv -- --help
cargo run --bin beam_search -- --help
```

## Benchmarking

- Collect traces: `benchmarks/scripts/collect_vllm_trace.py`, `collect_sglang_trace.py`
- Replay traces: `workloads/src/trace_replay.rs`
- Run full suite: `benchmarks/scripts/run_all.sh`
- Generate plots: `benchmarks/scripts/plot.py`

## Repository Structure

```
polaris/
  kernel/          # polaris.ko, Makefile, Kbuild
  polarisd/        # Rust daemon
  polarisctl/      # Rust CLI
  workloads/       # Synthetic and trace workloads
  benchmarks/      # Configs, scripts, results
  adapter/         # Optional vLLM adapter
  ebpf/            # Optional XDP eBPF
  docs/            # Design, API, evaluation docs
  report/          # Final report, figures
```
