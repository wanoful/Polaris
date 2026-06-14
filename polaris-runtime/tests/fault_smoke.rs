use std::fs::OpenOptions;
use std::os::fd::AsRawFd;

use cudarc::driver::sys::{self, CUresult};
use libpolaris::ioctl;
use libpolaris::types::{
    PolarisBlockGetStateArg, PolarisBlockReserveArg, PolarisBlockState, PolarisPhase,
    PolarisSessionCreateArg, PolarisSessionDestroyArg, PolarisSpillBlockArg,
    POLARIS_RESERVE_FLAG_OVERWRITE,
};
use polaris_runtime::{Runtime, RuntimeConfig};

const MIB: u64 = 1024 * 1024;
const BLOCK_SIZE: u64 = 2 * MIB;

fn cuda_check(result: CUresult, what: &str) {
    assert_eq!(result, CUresult::CUDA_SUCCESS, "{what} failed: {result:?}");
}

fn push_primary_context(device_ordinal: i32) -> sys::CUcontext {
    unsafe {
        cuda_check(sys::cuInit(0), "cuInit");

        let mut dev = 0;
        cuda_check(sys::cuDeviceGet(&mut dev, device_ordinal), "cuDeviceGet");

        let mut ctx = std::ptr::null_mut();
        cuda_check(
            sys::cuDevicePrimaryCtxRetain(&mut ctx, dev),
            "cuDevicePrimaryCtxRetain",
        );
        cuda_check(sys::cuCtxPushCurrent_v2(ctx), "cuCtxPushCurrent");
        ctx
    }
}

fn pop_primary_context(device_ordinal: i32) {
    unsafe {
        let mut ctx = std::ptr::null_mut();
        let _ = sys::cuCtxPopCurrent_v2(&mut ctx);

        let mut dev = 0;
        if sys::cuDeviceGet(&mut dev, device_ordinal) == CUresult::CUDA_SUCCESS {
            let _ = sys::cuDevicePrimaryCtxRelease_v2(dev);
        }
    }
}

fn fill_pattern(buf: &mut [u8], seed: u8) {
    for (i, byte) in buf.iter_mut().enumerate() {
        *byte = seed
            .wrapping_add((i as u8).wrapping_mul(131))
            .wrapping_add((i >> 7) as u8);
    }
}

fn kernel_stat(name: &str) -> u64 {
    let stats = std::fs::read_to_string("/sys/kernel/polaris/stats")
        .expect("read /sys/kernel/polaris/stats");
    stats
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            if key.trim() != name {
                return None;
            }
            value.split_whitespace().next()?.parse::<u64>().ok()
        })
        .unwrap_or_else(|| panic!("stat {name} not present in /sys/kernel/polaris/stats"))
}

fn block_state(fd: i32, session_id: u64, block_id: u64) -> PolarisBlockGetStateArg {
    let mut state = PolarisBlockGetStateArg {
        session_id,
        block_id,
        ..Default::default()
    };
    ioctl::block_get_state(fd, &mut state).expect("POLARIS_BLOCK_GET_STATE");
    state
}

fn wait_for_block_state(
    fd: i32,
    session_id: u64,
    block_id: u64,
    expected: PolarisBlockState,
) -> PolarisBlockGetStateArg {
    let started = std::time::Instant::now();
    loop {
        let state = block_state(fd, session_id, block_id);
        if state.state == expected as u32 {
            return state;
        }
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "block {block_id} state={} did not become {:?}",
            state.state,
            expected
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn copy_to_device(va: u64, src: &[u8]) {
    unsafe {
        cuda_check(
            sys::cuMemcpyHtoD_v2(va, src.as_ptr() as *const std::ffi::c_void, src.len()),
            "cuMemcpyHtoD",
        );
    }
}

fn copy_from_device(va: u64, dst: &mut [u8]) {
    unsafe {
        cuda_check(
            sys::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut std::ffi::c_void, va, dst.len()),
            "cuMemcpyDtoH",
        );
    }
}

#[test]
#[ignore = "requires loaded polaris.ko, patched nvidia-uvm.ko, and an NVIDIA GPU"]
fn block_reserve_spill_and_reload_via_runtime_worker() {
    let device_ordinal = 0;
    let mut runtime = Runtime::create(RuntimeConfig {
        gpu_id: 0,
        device_ordinal,
        total_bytes: 16 * 1024 * MIB,
        budget_bytes: 512 * MIB,
        cpu_pool_bytes: 64 * MIB,
        va_reserve_bytes: 256 * MIB,
        block_size: BLOCK_SIZE,
    })
    .expect("create POLARIS runtime");

    runtime.start().expect("start POLARIS runtime worker");

    let dev = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/polaris")
        .expect("open /dev/polaris");
    let fd = dev.as_raw_fd();

    let mut session = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut session)
        .expect("POLARIS_SESSION_CREATE");

    let mut reserve = PolarisBlockReserveArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE");

    assert_ne!(reserve.block_id, 0);
    assert_ne!(reserve.gpu_vaddr, 0);

    let _ctx = push_primary_context(device_ordinal);
    let mut src = vec![0u8; BLOCK_SIZE as usize];
    let mut dst = vec![0u8; BLOCK_SIZE as usize];
    fill_pattern(&mut src, 0x41);

    copy_to_device(reserve.gpu_vaddr, &src);
    copy_from_device(reserve.gpu_vaddr, &mut dst);
    assert_eq!(src, dst);

    let offloads_before = kernel_stat("offloads");
    let reloads_before = kernel_stat("reloads");
    let cpu_used_before = kernel_stat("cpu_used_mib");

    let mut spill = PolarisSpillBlockArg {
        block_id: reserve.block_id,
        ..Default::default()
    };
    ioctl::spill_block(fd, &mut spill).expect("POLARIS_SPILL_BLOCK");
    assert_ne!(spill.decision_id, 0);
    assert_eq!(spill.unmapped_count, 0);
    wait_for_block_state(
        fd,
        session.session_id,
        reserve.block_id,
        PolarisBlockState::CpuOffloaded,
    );
    assert_eq!(kernel_stat("offloads"), offloads_before + 1);
    assert!(kernel_stat("cpu_used_mib") >= cpu_used_before + 1);

    let mut reload = PolarisBlockReserveArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        flags: POLARIS_RESERVE_FLAG_OVERWRITE,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reload)
        .expect("POLARIS_BLOCK_RESERVE reload");
    assert_eq!(reload.block_id, reserve.block_id);
    assert_eq!(reload.gpu_vaddr, reserve.gpu_vaddr);
    wait_for_block_state(
        fd,
        session.session_id,
        reserve.block_id,
        PolarisBlockState::Resident,
    );
    assert_eq!(kernel_stat("reloads"), reloads_before + 1);
    assert_eq!(kernel_stat("cpu_used_mib"), cpu_used_before);

    dst.fill(0);
    copy_from_device(reload.gpu_vaddr, &mut dst);
    assert_eq!(src, dst);
    pop_primary_context(device_ordinal);

    let destroy = PolarisSessionDestroyArg {
        session_id: session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &destroy)
        .expect("POLARIS_SESSION_DESTROY");
}
