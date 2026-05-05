use cudarc::driver::sys::{
    self, CUdeviceptr, CUmemAccessDesc, CUmemAccess_flags, CUmemAllocationGranularity_flags,
    CUmemAllocationHandleType, CUmemAllocationProp, CUmemAllocationType, CUmemGenericAllocationHandle,
    CUmemLocation, CUmemLocationType, CUresult, CUmemLocation_st__bindgen_ty_1,
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
/// Use the VMM recommended granularity as alignment.

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

pub fn create_physical(size: u64, dev_ordinal: i32) -> Result<u64, String> {
    let mut handle: CUmemGenericAllocationHandle = 0;
    let prop = CUmemAllocationProp {
        type_: CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
        location: make_device_location(dev_ordinal),
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: unsafe { std::mem::zeroed() },
    };
    let res = unsafe { sys::cuMemCreate(&mut handle, size as usize, &prop, 0) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemCreate({size}) failed: error code {res:?}"));
    }
    Ok(handle)
}

pub fn map_memory(vaddr: u64, phys_handle: u64, size: u64) -> Result<(), String> {
    let res = unsafe { sys::cuMemMap(vaddr, size as usize, 0, phys_handle, 0) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemMap({vaddr:#x}, {size}) failed: error code {res:?}"));
    }
    Ok(())
}

pub fn set_access(vaddr: u64, size: u64, dev_ordinal: i32) -> Result<(), String> {
    let desc = CUmemAccessDesc {
        location: make_device_location(dev_ordinal),
        flags: CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    let res = unsafe { sys::cuMemSetAccess(vaddr, size as usize, &desc, 1) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemSetAccess({vaddr:#x}, {size}) failed: error code {res:?}"));
    }
    Ok(())
}

pub fn unmap_memory(vaddr: u64, size: u64) -> Result<(), String> {
    let res = unsafe { sys::cuMemUnmap(vaddr, size as usize) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemUnmap({vaddr:#x}) failed: error code {res:?}"));
    }
    Ok(())
}

pub fn release_physical(phys_handle: u64) -> Result<(), String> {
    let res = unsafe { sys::cuMemRelease(phys_handle) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemRelease({phys_handle:#x}) failed: error code {res:?}"));
    }
    Ok(())
}

pub fn allocate_host(size: u64) -> Result<u64, String> {
    let mut ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
    let res = unsafe { sys::cuMemAllocHost_v2(&mut ptr, size as usize) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemAllocHost({size}) failed: error code {res:?}"));
    }
    Ok(ptr as u64)
}

pub fn free_host(ptr: u64) -> Result<(), String> {
    let res = unsafe { sys::cuMemFreeHost(ptr as *mut std::os::raw::c_void) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemFreeHost({ptr:#x}) failed: error code {res:?}"));
    }
    Ok(())
}

/// Copy data from GPU device memory to host (pinned) memory.
pub fn copy_device_to_host(dst_host: u64, src_device: u64, byte_count: usize) -> Result<(), String> {
    let res = unsafe {
        sys::cuMemcpyDtoH_v2(
            dst_host as *mut std::os::raw::c_void,
            src_device as sys::CUdeviceptr,
            byte_count,
        )
    };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemcpyDtoH({dst_host:#x}, {src_device:#x}, {byte_count}) failed: error code {res:?}"));
    }
    Ok(())
}

/// Copy data from host (pinned) memory to GPU device memory.
pub fn copy_host_to_device(dst_device: u64, src_host: u64, byte_count: usize) -> Result<(), String> {
    let res = unsafe {
        sys::cuMemcpyHtoD_v2(
            dst_device as sys::CUdeviceptr,
            src_host as *const std::os::raw::c_void,
            byte_count,
        )
    };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemcpyHtoD({dst_device:#x}, {src_host:#x}, {byte_count}) failed: error code {res:?}"));
    }
    Ok(())
}

/// Copy data from one GPU device memory location to another.
/// Used for COW break: copy old physical block contents to a new physical block.
pub fn copy_device_to_device(dst_device: u64, src_device: u64, byte_count: usize) -> Result<(), String> {
    let res = unsafe {
        sys::cuMemcpyDtoD_v2(
            dst_device as sys::CUdeviceptr,
            src_device as sys::CUdeviceptr,
            byte_count,
        )
    };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!(
            "cuMemcpyDtoD(dst={dst_device:#x}, src={src_device:#x}, {byte_count}) failed: error code {res:?}"
        ));
    }
    Ok(())
}
