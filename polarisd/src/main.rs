// POLARIS Daemon (polarisd)
//
// Responsibilities:
//   1. Open /dev/polaris
//   2. Discover GPUs via CUDA/NVML and register them with the kernel
//   3. Enter decision loop: poll POLARIS_GET_DECISION, execute, report COMPLETE_OPERATION
//   4. Handle daemon lifecycle (systemd integration, crash recovery)
//
// Phase 1b will add real CUDA VMM operations (cuMemCreate, cuMemMap, etc.).
// For now, this skeleton registers a dummy GPU and polls decisions.

use libc::c_int;
use libpolaris::ioctl;
use libpolaris::types::*;
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::thread;
use std::time::Duration;

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

    // Register a dummy GPU (Phase 1b will use real CUDA/NVML).
    register_dummy_gpu(fd)?;

    // Enter the decision loop.
    eprintln!("polarisd: entering decision loop");
    decision_loop(fd)?;

    Ok(())
}

/// Register a dummy GPU with the kernel.
/// Phase 1b: Replace with real CUDA discovery via nvml-rs or cudarc.
fn register_dummy_gpu(fd: c_int) -> Result<(), Box<dyn std::error::Error>> {
    let arg = PolarisRegisterGpuArg {
        gpu_id: 0,
        total_bytes: 8 * 1024 * 1024 * 1024, // 8 GiB
        budget_bytes: 6 * 1024 * 1024 * 1024, // 6 GiB budget
        cpu_pool_bytes: 4 * 1024 * 1024 * 1024, // 4 GiB CPU pool
        numa_node: 0,
        ..Default::default()
    };

    ioctl::ioctl_write(fd, ioctl::POLARIS_REGISTER_GPU, &arg)
        .map_err(|e| format!("REGISTER_GPU failed: errno {e}"))?;

    eprintln!("polarisd: GPU 0 registered (8 GiB total, 6 GiB budget, 4 GiB CPU pool)");
    Ok(())
}

/// Poll for decisions from the kernel and execute them.
/// Phase 1b: Will execute real CUDA VMM operations.
fn decision_loop(fd: c_int) -> Result<(), Box<dyn std::error::Error>> {
    let mut decision_arg = PolarisGetDecisionArg::default();

    loop {
        // Poll the kernel for pending decisions.
        match ioctl::ioctl_read(fd, ioctl::POLARIS_GET_DECISION, &mut decision_arg) {
            Ok(()) => {
                let count = decision_arg.count as usize;
                if count > 0 {
                    eprintln!("polarisd: received {count} decision(s)");
                    for i in 0..count {
                        let dec = &decision_arg.decisions[i];
                        execute_decision(fd, dec)?;
                    }
                }
            }
            Err(e) => {
                eprintln!("polarisd: GET_DECISION error: errno {e}");
                // If kernel module is unloaded or daemon is detached, exit.
                if e == libc::ENODEV as i32 {
                    eprintln!("polarisd: no daemon attached, exiting");
                    break;
                }
            }
        }

        // Sleep briefly to avoid busy-waiting.
        thread::sleep(Duration::from_millis(10));
    }

    Ok(())
}

/// Execute a single decision and report completion.
fn execute_decision(
    fd: c_int,
    dec: &PolarisDecision,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "polarisd: executing decision {} op={} block_id={} session_id={}",
        dec.decision_id, dec.op, dec.block_id, dec.session_id
    );

    // Set POLARIS_TEST_ERROR=1 to trigger error simulation for testing G4.
    let test_err: i32 = std::env::var("POLARIS_TEST_ERROR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let (result, output_handle) = if test_err != 0 {
        // Simulate various error codes based on block_id for G4 testing.
        let err = match dec.block_id % 5 {
            0 => -(libc::ENOMEM as i32),
            1 => -(libc::ENODEV as i32),
            2 => -(libc::EINVAL as i32),
            3 => -(libc::EFAULT as i32),
            _ => -(libc::EIO as i32),
        };
        eprintln!("polarisd: [TEST] simulating error {} for block {}", err, dec.block_id);
        (err, 0)
    } else {
        // Normal path: success with dummy phys handle.
        (0, dec.block_id.wrapping_mul(0x1000))
    };

    let complete = PolarisCompleteOperationArg {
        decision_id: dec.decision_id,
        result,
        output_handle,
        ..Default::default()
    };

    ioctl::ioctl_write(fd, ioctl::POLARIS_COMPLETE_OPERATION, &complete)
        .map_err(|e| format!("COMPLETE_OPERATION failed: errno {e}"))?;

    if result == 0 {
        eprintln!("polarisd: decision {} completed (handle=0x{output_handle:x})", dec.decision_id);
    } else {
        eprintln!("polarisd: decision {} failed with result={}", dec.decision_id, result);
    }
    Ok(())
}
