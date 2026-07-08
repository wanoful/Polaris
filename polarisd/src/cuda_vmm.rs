//! CUDA driver bootstrap for the daemon: primary-context setup, the daemon GPU
//! VA reservation, allocation granularity, and the pinned host-copy pool.
//!
//! The v4 production path allocates and pages KV backing through daemon-owned
//! RM objects (see `rm.rs`) and moves bytes with the kernel/UVM CE copy helper
//! (`POLARIS_RM_COPY`). The old CUDA-VMM residency executor (`cuMemCreate` /
//! `cuMemMap` / `cuMemcpy*`-based alloc, offload, reload, and COW) has been
//! removed; only the context/VA/host-pool bootstrap below remains.

use cudarc::driver::sys::{
    self, CUdeviceptr, CUmemAllocationGranularity_flags, CUmemAllocationHandleType,
    CUmemAllocationProp, CUmemAllocationType, CUmemLocation, CUmemLocationType,
    CUmemLocation_st__bindgen_ty_1, CUresult,
};

use crate::gpu::CudaContext;

/// Default GPU VA reservation: 64 GiB.
/// Override with POLARIS_VA_RESERVE_GIB env var.
pub const DEFAULT_VA_RESERVE_GIB: u64 = 64;
pub fn va_reserve_size() -> u64 {
    std::env::var("POLARIS_VA_RESERVE_GIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_VA_RESERVE_GIB)
        * (1 << 30)
}

pub fn init(flags: u32) -> Result<(), String> {
    let res = unsafe { sys::cuInit(flags) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuInit({flags}) failed: error code {res:?}"));
    }
    Ok(())
}

pub fn get_device(ordinal: i32) -> Result<sys::CUdevice, String> {
    let mut dev: sys::CUdevice = 0;
    let res = unsafe { sys::cuDeviceGet(&mut dev, ordinal) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuDeviceGet({ordinal}) failed: error code {res:?}"));
    }
    Ok(dev)
}

pub fn device_get_name(dev: sys::CUdevice) -> Result<String, String> {
    let mut buf = [0u8; 256];
    let res = unsafe { sys::cuDeviceGetName(buf.as_mut_ptr() as *mut i8, 256, dev) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuDeviceGetName failed: error code {res:?}"));
    }
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(String::from_utf8_lossy(&buf[..len]).to_string())
}

pub fn create_context(dev: sys::CUdevice) -> Result<CudaContext, String> {
    let mut ctx: CudaContext = std::ptr::null_mut();
    let res = unsafe { sys::cuDevicePrimaryCtxRetain(&mut ctx, dev) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuDevicePrimaryCtxRetain failed: error code {res:?}"));
    }
    Ok(ctx)
}

pub fn push_context(ctx: CudaContext) -> Result<(), String> {
    let res = unsafe { sys::cuCtxPushCurrent_v2(ctx) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuCtxPushCurrent failed: error code {res:?}"));
    }
    Ok(())
}

pub fn pop_context() -> Result<CudaContext, String> {
    let mut ctx: CudaContext = std::ptr::null_mut();
    let res = unsafe { sys::cuCtxPopCurrent_v2(&mut ctx) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuCtxPopCurrent failed: error code {res:?}"));
    }
    Ok(ctx)
}

fn make_device_location(dev_ordinal: i32) -> CUmemLocation {
    CUmemLocation {
        type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
        __bindgen_anon_1: CUmemLocation_st__bindgen_ty_1 { id: dev_ordinal },
    }
}

pub fn get_allocation_granularity(dev_ordinal: i32) -> Result<u64, String> {
    let mut granule: usize = 0;
    let prop = CUmemAllocationProp {
        type_: CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
        location: make_device_location(dev_ordinal),
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: unsafe { std::mem::zeroed() },
    };
    let res = unsafe {
        sys::cuMemGetAllocationGranularity(
            &mut granule,
            &prop,
            CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
    };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemGetAllocationGranularity failed: error code {res:?}"));
    }
    Ok(granule as u64)
}

pub fn reserve_va(len: u64) -> Result<u64, String> {
    let mut ptr: CUdeviceptr = 0;
    // addr=0: let the driver choose, avoids hint-rejection on constrained VA spaces.
    let res = unsafe {
        sys::cuMemAddressReserve(
            &mut ptr,
            len as usize,
            0, // use driver's preferred alignment
            0, // addr = 0: driver picks
            0,
        )
    };
    let gib = len / (1024 * 1024 * 1024);
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemAddressReserve({gib} GiB) failed: error code {res:?}"));
    }
    Ok(ptr)
}

pub fn allocate_host(size: u64) -> Result<u64, String> {
    let mut ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
    let res = unsafe { sys::cuMemAllocHost_v2(&mut ptr, size as usize) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemAllocHost({size}) failed: error code {res:?}"));
    }
    Ok(ptr as u64)
}
