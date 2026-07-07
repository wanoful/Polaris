mod async_offload;
mod cuda_vmm;
mod decision;
mod gpu;
mod lifecycle;
mod nvml;
mod offload;
mod rm;

use gpu::GpuState;
use libc::c_int;
use libpolaris::ioctl;
use libpolaris::types::*;
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("polarisd: starting POLARIS daemon");

    // Open /dev/polaris character device.
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/polaris")
        .map_err(|e| format!("Failed to open /dev/polaris: {e}"))?;

    let fd = file.as_raw_fd() as c_int;
    eprintln!("polarisd: /dev/polaris opened (fd={fd})");

    // Discover GPUs via NVML.
    let gpu_infos = nvml::discover_gpus()?;

    if gpu_infos.is_empty() {
        return Err("No GPUs discovered via NVML".into());
    }

    // For Phase 1b, use the first GPU.
    let info = &gpu_infos[0];
    eprintln!("polarisd: using GPU {} ({})", info.index, info.name);

    // Initialize CUDA.
    cuda_vmm::init(0)?;
    eprintln!("polarisd: CUDA driver initialized");

    // Get CUDA device handle.
    let dev = cuda_vmm::get_device(info.index as i32)?;
    let dev_name = cuda_vmm::device_get_name(dev)?;
    eprintln!("polarisd: CUDA device: {dev_name}");

    // Create CUDA context.
    let ctx = cuda_vmm::create_context(dev)?;
    eprintln!("polarisd: CUDA context created");

    // Push context for VMM setup.
    cuda_vmm::push_context(ctx)?;

    // Get allocation granularity.
    let granule = cuda_vmm::get_allocation_granularity(info.index as i32)?;
    eprintln!(
        "polarisd: allocation granularity = {granule} bytes ({} MiB)",
        granule / (1024 * 1024)
    );

    // Reserve GPU virtual address space (configurable via POLARIS_VA_RESERVE_GIB).
    let vas_size = cuda_vmm::va_reserve_size();
    let vas_base = cuda_vmm::reserve_va(vas_size)?;
    eprintln!(
        "polarisd: reserved GPU VA range {vas_base:#x}..{:#x} ({} GiB)",
        vas_base + vas_size,
        vas_size / (1024 * 1024 * 1024)
    );

    // Compute budget (use 75% of total GPU memory for KV cache by default).
    let budget_bytes = env_bytes("POLARISD_GPU_BUDGET_BYTES").unwrap_or(info.total_memory * 3 / 4);
    // CPU pool: 4 GiB by default for offload testing. Tests can shrink this to
    // make host-pool ENOMEM deterministic without exhausting machine memory.
    let requested_cpu_pool_bytes =
        env_bytes("POLARISD_CPU_POOL_BYTES").unwrap_or(4 * 1024 * 1024 * 1024u64);

    // Pre-allocate CPU pinned memory pool for block offloads.
    let (cpu_pool_bytes, cpu_pool_base) = match cuda_vmm::allocate_host(requested_cpu_pool_bytes) {
        Ok(ptr) => {
            eprintln!(
                "polarisd: allocated CPU pinned memory pool: {} MiB at {ptr:#x}",
                requested_cpu_pool_bytes / (1024 * 1024)
            );
            (requested_cpu_pool_bytes, ptr)
        }
        Err(e) => {
            eprintln!("polarisd: WARNING CPU pinned memory pool allocation failed: {e}");
            eprintln!("polarisd:   GPU↔CPU offload will not be available.");
            eprintln!("polarisd:   Check available host memory and try reducing the pool size.");
            (0u64, 0u64)
        }
    };

    cuda_vmm::pop_context()?;

    // Register GPU with the kernel module.
    let reg_arg = PolarisRegisterGpuArg {
        gpu_id: info.index,
        total_bytes: info.total_memory,
        budget_bytes,
        cpu_pool_bytes,
        numa_node: 0,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_REGISTER_GPU, &reg_arg)
        .map_err(|e| format!("REGISTER_GPU failed: errno {e}"))?;

    eprintln!(
        "polarisd: GPU {} registered (total={} MiB, budget={} MiB, cpu_pool={} MiB)",
        info.index,
        info.total_memory / (1024 * 1024),
        budget_bytes / (1024 * 1024),
        cpu_pool_bytes / (1024 * 1024),
    );

    // Register the global GPU VA pool with the kernel so that the UVM
    // fault hook can match fault addresses to POLARIS blocks.
    {
        let mut va_arg = PolarisRegisterVaRangeArg {
            gpu_id: info.index,
            base: vas_base,
            length: vas_size,
            block_size: POLARIS_DEFAULT_BYTES_PER_TOKEN * 16,
            ..Default::default()
        };
        ioctl::ioctl_read(fd, ioctl::POLARIS_REGISTER_VA_RANGE, &mut va_arg)
            .map_err(|e| format!("REGISTER_VA_RANGE failed: errno {e}"))?;
        eprintln!(
            "polarisd: VA range registered (range_id={}, base={:#x}, length={} GiB)",
            va_arg.range_id,
            va_arg.base,
            va_arg.length / (1024 * 1024 * 1024),
        );
    }

    // Build per-GPU state.
    let mut gpu_state = GpuState::new(
        info.index,
        info.index as i32,
        ctx,
        vas_base,
        vas_size,
        granule,
        info.total_memory,
        budget_bytes,
        cpu_pool_bytes,
        cpu_pool_base,
    );

    // Build CPU pool allocator from the pre-allocated pinned memory.
    let mut cpu_pool = offload::CpuPool::new(cpu_pool_base, cpu_pool_bytes);

    // Load test error mode from env.
    let test_err: i32 = std::env::var("POLARIS_TEST_ERROR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    if test_err != 0 {
        eprintln!("polarisd: [TEST] error injection mode enabled (POLARIS_TEST_ERROR={test_err})");
    }

    let rm_backing_enabled = env_enabled("POLARISD_RM_BACKING");
    let mut rm_backend = if rm_backing_enabled {
        eprintln!("polarisd: daemon-owned RM backing enabled (POLARISD_RM_BACKING=1)");
        Some(
            rm::RmBackend::new(info.index as i32)
                .map_err(|e| format!("RM backing initialization failed: {e}"))?,
        )
    } else {
        None
    };

    // Reconcile state with the kernel module.
    lifecycle::reconcile();

    // Notify systemd that the daemon is ready to serve requests.
    lifecycle::notify_ready();

    // Enter the decision loop.
    eprintln!("polarisd: entering decision loop");
    decision_loop(
        fd,
        &mut gpu_state,
        &mut cpu_pool,
        rm_backend.as_mut(),
        test_err,
    )?;

    // Notify systemd that the daemon is stopping cleanly.
    lifecycle::notify_stopping();

    Ok(())
}

fn decision_loop(
    fd: c_int,
    gpu: &mut GpuState,
    cpu_pool: &mut offload::CpuPool,
    mut rm_backend: Option<&mut rm::RmBackend>,
    test_err: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut decision_arg = PolarisGetDecisionArg::default();

    // Async offload pool: gated behind POLARISD_ASYNC_OFFLOAD. When disabled the
    // pool is None and every decision runs on the original synchronous path,
    // leaving default behavior byte-for-byte unchanged. When enabled, eligible
    // RM offloads have only their rm_copy ioctl run on worker threads; all
    // GpuState/CpuPool/RmBackend mutation still happens here in the main thread.
    let mut async_pool = if env_enabled("POLARISD_ASYNC_OFFLOAD") {
        let workers: usize = std::env::var("POLARISD_ASYNC_OFFLOAD_WORKERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(4);
        eprintln!("polarisd: async offload enabled ({workers} workers)");
        Some(async_offload::AsyncOffloadPool::new(fd, workers))
    } else {
        None
    };

    loop {
        // Finalize any offload copies workers have completed since last iteration
        // before doing anything else, so the CPU pool / RM backing are reclaimed
        // promptly for subsequent reloads.
        if let (Some(pool), Some(backend)) = (async_pool.as_mut(), rm_backend.as_deref_mut()) {
            for done in pool.drain_completed() {
                let exec = offload::finalize_rm_offload(
                    &done.job,
                    gpu,
                    cpu_pool,
                    backend,
                    done.copy_result,
                    done.bytes_copied,
                );
                complete_operation(fd, done.job.decision_id, done.job.generation, &exec);
            }
        }

        match ioctl::ioctl_read(fd, ioctl::POLARIS_GET_DECISION, &mut decision_arg) {
            Ok(()) => {
                let count = decision_arg.count as usize;
                if count == 0 {
                    continue;
                }

                eprintln!("polarisd: received {count} decision(s)");
                for i in 0..count {
                    let dec = decision_arg.decisions[i];

                    // Try to route an eligible RM offload to the async pool. On
                    // any ineligibility (not RM-backed, pool full, pool disabled)
                    // fall through to the synchronous executor below.
                    if dec.op == PolarisDecisionOp::Offload as u32 && test_err == 0 {
                        if let (Some(pool), Some(backend)) =
                            (async_pool.as_mut(), rm_backend.as_deref_mut())
                        {
                            // Apply back-pressure: if the pool is saturated, wait
                            // for and finalize at least one completion first.
                            if pool.is_full() {
                                for d in pool.wait_for_completion() {
                                    let exec = offload::finalize_rm_offload(
                                        &d.job, gpu, cpu_pool, backend, d.copy_result, d.bytes_copied,
                                    );
                                    complete_operation(fd, d.job.decision_id, d.job.generation, &exec);
                                }
                            }
                            match offload::prepare_rm_offload(&dec, gpu, cpu_pool, backend) {
                                Ok(job) => {
                                    pool.submit(job);
                                    // COMPLETE_OPERATION is emitted at finalize.
                                    continue;
                                }
                                Err(_) => { /* fall back to synchronous path */ }
                            }
                        }
                    }

                    let exec = decision::execute(
                        fd,
                        &dec,
                        gpu,
                        cpu_pool,
                        rm_backend.as_deref_mut(),
                        test_err,
                    );
                    complete_operation(fd, dec.decision_id, dec.generation, &exec);

                    if exec.result == 0 {
                        eprintln!(
                            "polarisd: decision {} completed (handle=0x{:x})",
                            dec.decision_id, exec.output_handle
                        );
                    } else {
                        eprintln!(
                            "polarisd: decision {} failed with result={}",
                            dec.decision_id, exec.result
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!("polarisd: GET_DECISION error: errno {e}");
                if e == libc::ENODEV as i32 {
                    eprintln!("polarisd: no daemon attached, exiting");
                    break;
                }
            }
        }
    }

    // Drain and finalize any in-flight offloads before returning so shared state
    // is consistent and nothing leaks.
    if let (Some(pool), Some(backend)) = (async_pool.take(), rm_backend.as_deref_mut()) {
        for done in pool.shutdown() {
            let exec = offload::finalize_rm_offload(
                &done.job,
                gpu,
                cpu_pool,
                backend,
                done.copy_result,
                done.bytes_copied,
            );
            complete_operation(fd, done.job.decision_id, done.job.generation, &exec);
        }
    }

    Ok(())
}

/// Emit COMPLETE_OPERATION for a finished decision (sync result or async offload
/// finalize). Kept as one helper so both paths report identically to the kernel.
fn complete_operation(
    fd: c_int,
    decision_id: u64,
    generation: u64,
    exec: &decision::ExecutionResult,
) {
    let complete = PolarisCompleteOperationArg {
        decision_id,
        generation,
        result: exec.result,
        rm_control_fd: exec.rm_control_fd,
        output_handle: exec.output_handle,
        output_cpu_addr: exec.output_cpu_addr,
        rm_h_client: exec.rm_h_client,
        rm_h_memory: exec.rm_h_memory,
        rm_backing_length: exec.rm_backing_length,
        phys_fb_addr: exec.phys_fb_addr,
        ..Default::default()
    };

    if let Err(e) = ioctl::ioctl_write(fd, ioctl::POLARIS_COMPLETE_OPERATION, &complete) {
        eprintln!(
            "polarisd: COMPLETE_OPERATION ioctl failed for decision {decision_id}: errno {e}"
        );
    }
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let v = v.trim();
            !v.is_empty()
                && v != "0"
                && !v.eq_ignore_ascii_case("false")
                && !v.eq_ignore_ascii_case("no")
        })
        .unwrap_or(false)
}

fn env_bytes(name: &str) -> Option<u64> {
    let value = std::env::var(name).ok()?;
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    match value.parse::<u64>() {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            eprintln!("polarisd: ignoring invalid {name}={value}: {e}");
            None
        }
    }
}
