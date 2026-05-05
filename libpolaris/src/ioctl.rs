//! IOCTL command codes and low-level wrappers for /dev/polaris.
//!
//! These command codes are computed using the standard Linux _IO/_IOW/_IOR/_IOWR
//! macros. They must match the kernel-side definitions exactly.

use crate::types::*;

/// Magic number 'P' = 0x50
const MAGIC: u8 = 0x50;

// ─── _IO/_IOW/_IOR/_IOWR helpers ────────────────────────────────────────────

const fn ioc(dir: u32, ty: u32, nr: u32, size: u32) -> u32 {
    (dir << 30) | (size << 16) | ((ty as u32) << 8) | nr
}

macro_rules! io {
    ($ty:expr, $nr:expr) => {
        ioc(0, $ty as u32, $nr, 0)
    };
}

macro_rules! iow {
    ($ty:expr, $nr:expr, $t:ty) => {
        ioc(1, $ty as u32, $nr, core::mem::size_of::<$t>() as u32)
    };
}

macro_rules! ior {
    ($ty:expr, $nr:expr, $t:ty) => {
        ioc(2, $ty as u32, $nr, core::mem::size_of::<$t>() as u32)
    };
}

macro_rules! iowr {
    ($ty:expr, $nr:expr, $t:ty) => {
        ioc(3, $ty as u32, $nr, core::mem::size_of::<$t>() as u32)
    };
}

pub(crate) use io;
pub(crate) use ior;
pub(crate) use iow;
pub(crate) use iowr;

// ─── IOCTL Command Codes ────────────────────────────────────────────────────

pub const POLARIS_REGISTER_GPU: u32 =
    iow!(MAGIC, 0x01, PolarisRegisterGpuArg);

pub const POLARIS_SESSION_CREATE: u32 =
    iowr!(MAGIC, 0x02, PolarisSessionCreateArg);

pub const POLARIS_SESSION_DESTROY: u32 =
    iow!(MAGIC, 0x03, PolarisSessionDestroyArg);

pub const POLARIS_SESSION_GET_STATS: u32 =
    iowr!(MAGIC, 0x04, PolarisSessionGetStatsArg);

pub const POLARIS_SESSION_BRANCH: u32 =
    iowr!(MAGIC, 0x05, PolarisSessionBranchArg);

pub const POLARIS_BLOCK_GROW: u32 =
    iowr!(MAGIC, 0x06, PolarisBlockGrowArg);

pub const POLARIS_BLOCK_FREE: u32 =
    iow!(MAGIC, 0x07, PolarisBlockFreeArg);

pub const POLARIS_BLOCK_TOUCH: u32 =
    iow!(MAGIC, 0x08, PolarisBlockTouchArg);

pub const POLARIS_BLOCK_GET_STATE: u32 =
    iowr!(MAGIC, 0x09, PolarisBlockGetStateArg);

pub const POLARIS_GET_DECISION: u32 =
    iowr!(MAGIC, 0x0A, PolarisGetDecisionArg);

pub const POLARIS_COMPLETE_OPERATION: u32 =
    iowr!(MAGIC, 0x0B, PolarisCompleteOperationArg);

pub const POLARIS_GET_GLOBAL_STATS: u32 =
    iowr!(MAGIC, 0x0C, PolarisGetGlobalStatsArg);

pub const POLARIS_LIST_SESSIONS: u32 =
    iowr!(MAGIC, 0x0D, PolarisListSessionsArg);

pub const POLARIS_SET_POLICY: u32 =
    iow!(MAGIC, 0x0E, PolarisSetPolicyArg);

// ─── Low-level ioctl wrappers ────────────────────────────────────────────────

/// Issue an ioctl to a file descriptor with a mutable argument.
///
/// # Safety
/// The caller must ensure `arg` matches the ioctl command's expected type.
pub unsafe fn ioctl_ptr(fd: i32, cmd: u32, arg: *mut ()) -> i32 {
    unsafe { libc::ioctl(fd, cmd as _, arg) }
}

/// Issue an ioctl that reads data from the kernel.
/// On syscall failure, returns the errno value as the error.
pub fn ioctl_read<T>(fd: i32, cmd: u32, arg: &mut T) -> Result<(), i32> {
    let ret = unsafe { libc::ioctl(fd, cmd as _, arg as *mut T as *mut ()) };
    if ret < 0 {
        let eno = unsafe { *libc::__errno_location() };
        // Guard: if errno wasn't set, fall back to a default.
        Err(if eno != 0 { eno } else { libc::EIO })
    } else {
        Ok(())
    }
}

/// Issue an ioctl that writes data to the kernel.
/// On syscall failure, returns the errno value as the error.
pub fn ioctl_write<T>(fd: i32, cmd: u32, arg: &T) -> Result<(), i32> {
    let ret = unsafe { libc::ioctl(fd, cmd as _, arg as *const T as *mut ()) };
    if ret < 0 {
        let eno = unsafe { *libc::__errno_location() };
        Err(if eno != 0 { eno } else { libc::EIO })
    } else {
        Ok(())
    }
}
