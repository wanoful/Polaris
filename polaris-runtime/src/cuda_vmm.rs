use cudarc::driver::sys::{
    self, CUdeviceptr, CUmemAccessDesc, CUmemAccess_flags, CUmemAllocationGranularity_flags,
    CUmemAllocationHandleType, CUmemAllocationProp, CUmemAllocationType,
    CUmemGenericAllocationHandle, CUmemLocation, CUmemLocationType, CUmemLocation_st__bindgen_ty_1,
    CUresult,
};

pub type CudaContext = *mut sys::CUctx_st;

fn device_location(device_ordinal: i32) -> CUmemLocation {
    CUmemLocation {
        type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
        __bindgen_anon_1: CUmemLocation_st__bindgen_ty_1 { id: device_ordinal },
    }
}

pub fn init() -> Result<(), String> {
    let res = unsafe { sys::cuInit(0) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuInit failed: {res:?}"));
    }
    Ok(())
}

pub fn retain_primary_context(device_ordinal: i32) -> Result<CudaContext, String> {
    let mut dev: sys::CUdevice = 0;
    let res = unsafe { sys::cuDeviceGet(&mut dev, device_ordinal) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuDeviceGet({device_ordinal}) failed: {res:?}"));
    }

    let mut ctx: CudaContext = std::ptr::null_mut();
    let res = unsafe { sys::cuDevicePrimaryCtxRetain(&mut ctx, dev) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuDevicePrimaryCtxRetain failed: {res:?}"));
    }
    Ok(ctx)
}

pub fn push_context(ctx: CudaContext) -> Result<(), String> {
    let res = unsafe { sys::cuCtxPushCurrent_v2(ctx) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuCtxPushCurrent failed: {res:?}"));
    }
    Ok(())
}

pub fn pop_context() {
    let mut ctx: CudaContext = std::ptr::null_mut();
    let _ = unsafe { sys::cuCtxPopCurrent_v2(&mut ctx) };
}

pub fn release_primary_context(device_ordinal: i32) -> Result<(), String> {
    let mut dev: sys::CUdevice = 0;
    let res = unsafe { sys::cuDeviceGet(&mut dev, device_ordinal) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuDeviceGet({device_ordinal}) failed: {res:?}"));
    }

    let res = unsafe { sys::cuDevicePrimaryCtxRelease_v2(dev) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuDevicePrimaryCtxRelease failed: {res:?}"));
    }
    Ok(())
}

pub fn allocation_granularity(device_ordinal: i32) -> Result<u64, String> {
    let prop = CUmemAllocationProp {
        type_: CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
        location: device_location(device_ordinal),
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: unsafe { std::mem::zeroed() },
    };
    let mut granule = 0usize;
    let res = unsafe {
        sys::cuMemGetAllocationGranularity(
            &mut granule,
            &prop,
            CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
    };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemGetAllocationGranularity failed: {res:?}"));
    }
    Ok(granule as u64)
}

pub fn reserve_va(size: u64) -> Result<u64, String> {
    let mut ptr: CUdeviceptr = 0;
    let res = unsafe { sys::cuMemAddressReserve(&mut ptr, size as usize, 0, 0, 0) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemAddressReserve({size}) failed: {res:?}"));
    }
    Ok(ptr)
}

pub fn free_va(vaddr: u64, size: u64) -> Result<(), String> {
    if vaddr == 0 || size == 0 {
        return Ok(());
    }

    let res = unsafe { sys::cuMemAddressFree(vaddr, size as usize) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!(
            "cuMemAddressFree({vaddr:#x}, {size}) failed: {res:?}"
        ));
    }
    Ok(())
}

pub fn create_physical(size: u64, device_ordinal: i32) -> Result<u64, String> {
    let prop = CUmemAllocationProp {
        type_: CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
        location: device_location(device_ordinal),
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: unsafe { std::mem::zeroed() },
    };
    let mut handle: CUmemGenericAllocationHandle = 0;
    let res = unsafe { sys::cuMemCreate(&mut handle, size as usize, &prop, 0) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemCreate({size}) failed: {res:?}"));
    }
    Ok(handle)
}

pub fn map_memory(vaddr: u64, handle: u64, size: u64) -> Result<(), String> {
    let res = unsafe { sys::cuMemMap(vaddr, size as usize, 0, handle, 0) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemMap({vaddr:#x}, {size}) failed: {res:?}"));
    }
    Ok(())
}

pub fn set_access(vaddr: u64, size: u64, device_ordinal: i32) -> Result<(), String> {
    let desc = CUmemAccessDesc {
        location: device_location(device_ordinal),
        flags: CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    let res = unsafe { sys::cuMemSetAccess(vaddr, size as usize, &desc, 1) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!(
            "cuMemSetAccess({vaddr:#x}, {size}) failed: {res:?}"
        ));
    }
    Ok(())
}

pub fn unmap_memory(vaddr: u64, size: u64) -> Result<(), String> {
    let res = unsafe { sys::cuMemUnmap(vaddr, size as usize) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemUnmap({vaddr:#x}, {size}) failed: {res:?}"));
    }
    Ok(())
}

pub fn release_physical(handle: u64) -> Result<(), String> {
    let res = unsafe { sys::cuMemRelease(handle) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemRelease({handle:#x}) failed: {res:?}"));
    }
    Ok(())
}

pub fn alloc_host(size: u64) -> Result<u64, String> {
    if size == 0 {
        return Ok(0);
    }
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let res = unsafe { sys::cuMemAllocHost_v2(&mut ptr, size as usize) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemAllocHost({size}) failed: {res:?}"));
    }
    Ok(ptr as u64)
}

pub fn free_host(ptr: u64) -> Result<(), String> {
    if ptr == 0 {
        return Ok(());
    }

    let res = unsafe { sys::cuMemFreeHost(ptr as *mut std::ffi::c_void) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemFreeHost({ptr:#x}) failed: {res:?}"));
    }
    Ok(())
}

pub fn copy_dtoh(dst_cpu: u64, src_gpu: u64, size: u64) -> Result<(), String> {
    let res =
        unsafe { sys::cuMemcpyDtoH_v2(dst_cpu as *mut std::ffi::c_void, src_gpu, size as usize) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!(
            "cuMemcpyDtoH({src_gpu:#x}, {size}) failed: {res:?}"
        ));
    }
    Ok(())
}

pub fn copy_htod(dst_gpu: u64, src_cpu: u64, size: u64) -> Result<(), String> {
    let res =
        unsafe { sys::cuMemcpyHtoD_v2(dst_gpu, src_cpu as *const std::ffi::c_void, size as usize) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!(
            "cuMemcpyHtoD({dst_gpu:#x}, {size}) failed: {res:?}"
        ));
    }
    Ok(())
}

pub fn copy_dtod(dst_gpu: u64, src_gpu: u64, size: u64) -> Result<(), String> {
    let res = unsafe { sys::cuMemcpyDtoD_v2(dst_gpu, src_gpu, size as usize) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!(
            "cuMemcpyDtoD({src_gpu:#x}->{dst_gpu:#x}, {size}) failed: {res:?}"
        ));
    }
    Ok(())
}
