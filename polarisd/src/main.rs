mod cuda_vmm;
mod decision;
mod gpu;
mod lifecycle;
mod nvml;
mod offload;

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
    eprintln!("polarisd: allocation granularity = {granule} bytes ({} MiB)", granule / (1024 * 1024));

    // Reserve GPU virtual address space (configurable via POLARIS_VA_RESERVE_GIB).
    let vas_size = cuda_vmm::va_reserve_size();
    let vas_base = cuda_vmm::reserve_va(vas_size)?;
    eprintln!(
        "polarisd: reserved GPU VA range {vas_base:#x}..{:#x} ({} GiB)",
        vas_base + vas_size,
        vas_size / (1024 * 1024 * 1024)
    );

    // Compute budget (use 75% of total GPU memory for KV cache).
    let budget_bytes = info.total_memory * 3 / 4;
    // CPU pool: 2x GPU memory for offload (Phase 2 feature, but reserve it now).
    let cpu_pool_bytes = info.total_memory * 2;

    // Pre-allocate CPU pinned memory pool for block offloads.
    let (cpu_pool_bytes, cpu_pool_base) = match cuda_vmm::allocate_host(cpu_pool_bytes) {
        Ok(ptr) => {
            let cpb = info.total_memory * 2;
            eprintln!(
                "polarisd: allocated CPU pinned memory pool: {} MiB at {ptr:#x}",
                cpb / (1024 * 1024)
            );
            (cpb, ptr)
        }
        Err(e) => {
            eprintln!(
                "polarisd: WARNING CPU pinned memory pool allocation failed: {e}"
            );
            eprintln!(
                "polarisd:   GPU↔CPU offload will not be available."
            );
            eprintln!(
                "polarisd:   Check available host memory and try reducing the pool size."
            );
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

    // Reconcile state with the kernel module.
    lifecycle::reconcile();

    // Notify systemd that the daemon is ready to serve requests.
    lifecycle::notify_ready();

    // Enter the decision loop.
    eprintln!("polarisd: entering decision loop");
    decision_loop(fd, &mut gpu_state, &mut cpu_pool, test_err)?;

    // Notify systemd that the daemon is stopping cleanly.
    lifecycle::notify_stopping();

    Ok(())
}

fn decision_loop(
    fd: c_int,
    gpu: &mut GpuState,
    cpu_pool: &mut offload::CpuPool,
    test_err: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut decision_arg = PolarisGetDecisionArg::default();

    loop {
        match ioctl::ioctl_read(fd, ioctl::POLARIS_GET_DECISION, &mut decision_arg) {
            Ok(()) => {
                let count = decision_arg.count as usize;
                if count == 0 {
                    continue;
                }

                eprintln!("polarisd: received {count} decision(s)");
                for i in 0..count {
                    let dec = &decision_arg.decisions[i];
                    let exec = decision::execute(fd, dec, gpu, cpu_pool, test_err);

                    let complete = PolarisCompleteOperationArg {
                        decision_id: dec.decision_id,
                        generation: dec.generation,
                        result: exec.result,
                        output_handle: exec.output_handle,
                        output_cpu_addr: exec.output_cpu_addr,
                        ..Default::default()
                    };

                    if let Err(e) = ioctl::ioctl_write(
                        fd,
                        ioctl::POLARIS_COMPLETE_OPERATION,
                        &complete,
                    ) {
                        eprintln!(
                            "polarisd: COMPLETE_OPERATION ioctl failed for decision {}: errno {e}",
                            dec.decision_id
                        );
                    }

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

    Ok(())
}
