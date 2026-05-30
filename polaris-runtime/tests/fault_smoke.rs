use std::fs::OpenOptions;
use std::os::fd::AsRawFd;

use cudarc::driver::sys::{self, CUresult};
use libpolaris::ioctl;
use libpolaris::types::{
    PolarisBlockReserveArg, PolarisPhase, PolarisSessionCreateArg, PolarisSessionDestroyArg,
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

#[test]
#[ignore = "requires loaded polaris.ko, patched nvidia-uvm.ko, and an NVIDIA GPU"]
fn block_reserve_maps_fault_decision_via_runtime_worker() {
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

    unsafe {
        cuda_check(
            sys::cuMemcpyHtoD_v2(
                reserve.gpu_vaddr,
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
            ),
            "cuMemcpyHtoD",
        );
        cuda_check(
            sys::cuMemcpyDtoH_v2(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                reserve.gpu_vaddr,
                dst.len(),
            ),
            "cuMemcpyDtoH",
        );
    }
    assert_eq!(src, dst);
    pop_primary_context(device_ordinal);

    let destroy = PolarisSessionDestroyArg {
        session_id: session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &destroy)
        .expect("POLARIS_SESSION_DESTROY");
}
