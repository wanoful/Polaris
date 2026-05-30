mod cuda_vmm;
mod runtime;

use runtime::{Runtime, RuntimeConfig};
use std::cell::RefCell;
use std::ffi::CString;
use std::ptr;

#[repr(C)]
pub struct PolarisRuntimeConfig {
    pub gpu_id: u32,
    pub device_ordinal: i32,
    pub total_bytes: u64,
    pub budget_bytes: u64,
    pub cpu_pool_bytes: u64,
    pub va_reserve_bytes: u64,
    pub block_size: u64,
    pub flags: u32,
}

#[repr(C)]
pub struct PolarisRuntimeInfo {
    pub va_base: u64,
    pub va_size: u64,
    pub granule: u64,
    pub cpu_pool_base: u64,
    pub cpu_pool_bytes: u64,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_error(msg: impl AsRef<str>) -> i32 {
    let sanitized = msg.as_ref().replace('\0', " ");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() =
            CString::new(sanitized).unwrap_or_else(|_| CString::new("unknown error").unwrap());
    });
    -(libc::EIO)
}

fn set_errno_error(errno: i32, msg: impl AsRef<str>) -> i32 {
    let sanitized = msg.as_ref().replace('\0', " ");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() =
            CString::new(sanitized).unwrap_or_else(|_| CString::new("unknown error").unwrap());
    });
    -errno.abs()
}

#[no_mangle]
pub extern "C" fn polaris_runtime_last_error() -> *const libc::c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// # Safety
///
/// `config` and `out_runtime` must be valid non-null pointers. `out_info` may
/// be null; when it is non-null it must point to writable storage for one
/// `PolarisRuntimeInfo`.
#[no_mangle]
pub unsafe extern "C" fn polaris_runtime_create(
    config: *const PolarisRuntimeConfig,
    out_runtime: *mut *mut Runtime,
    out_info: *mut PolarisRuntimeInfo,
) -> i32 {
    if config.is_null() || out_runtime.is_null() {
        return set_errno_error(libc::EINVAL, "null argument to polaris_runtime_create");
    }

    unsafe {
        *out_runtime = ptr::null_mut();
    }

    let cfg = unsafe { &*config };
    let cfg = RuntimeConfig {
        gpu_id: cfg.gpu_id,
        device_ordinal: cfg.device_ordinal,
        total_bytes: cfg.total_bytes,
        budget_bytes: cfg.budget_bytes,
        cpu_pool_bytes: cfg.cpu_pool_bytes,
        va_reserve_bytes: cfg.va_reserve_bytes,
        block_size: cfg.block_size,
    };

    match Runtime::create(cfg) {
        Ok(runtime) => {
            if !out_info.is_null() {
                let info = runtime.info();
                unsafe {
                    *out_info = PolarisRuntimeInfo {
                        va_base: info.va_base,
                        va_size: info.va_size,
                        granule: info.granule,
                        cpu_pool_base: info.cpu_pool_base,
                        cpu_pool_bytes: info.cpu_pool_bytes,
                    };
                }
            }
            unsafe {
                *out_runtime = Box::into_raw(Box::new(runtime));
            }
            0
        }
        Err(e) => set_error(e),
    }
}

/// # Safety
///
/// `runtime` must either be null or a pointer previously returned by
/// `polaris_runtime_create` and not yet destroyed.
#[no_mangle]
pub unsafe extern "C" fn polaris_runtime_start(runtime: *mut Runtime) -> i32 {
    if runtime.is_null() {
        return set_errno_error(libc::EINVAL, "null runtime in polaris_runtime_start");
    }

    match unsafe { &mut *runtime }.start() {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

/// # Safety
///
/// `runtime` must either be null or a pointer previously returned by
/// `polaris_runtime_create` and not yet destroyed.
#[no_mangle]
pub unsafe extern "C" fn polaris_runtime_poll_once(runtime: *mut Runtime, _timeout_ms: i32) -> i32 {
    if runtime.is_null() {
        return set_errno_error(libc::EINVAL, "null runtime in polaris_runtime_poll_once");
    }

    match unsafe { &mut *runtime }.poll_once() {
        Ok(count) => count as i32,
        Err(e) => set_error(e),
    }
}

/// # Safety
///
/// `runtime` must either be null or a pointer previously returned by
/// `polaris_runtime_create` and not yet destroyed.
#[no_mangle]
pub unsafe extern "C" fn polaris_runtime_stop(runtime: *mut Runtime) {
    if !runtime.is_null() {
        unsafe { &mut *runtime }.stop();
    }
}

/// # Safety
///
/// `runtime` must either be null or a pointer previously returned by
/// `polaris_runtime_create`. After this call returns, the pointer must not be
/// used again.
#[no_mangle]
pub unsafe extern "C" fn polaris_runtime_destroy(runtime: *mut Runtime) {
    if !runtime.is_null() {
        let mut boxed = unsafe { Box::from_raw(runtime) };
        boxed.stop();
    }
}
