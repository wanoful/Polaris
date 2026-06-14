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

pub const POLARIS_REGISTER_VA_RANGE: u32 =
    iowr!(MAGIC, 0x02, PolarisRegisterVaRangeArg);

pub const POLARIS_SESSION_CREATE: u32 =
    iowr!(MAGIC, 0x03, PolarisSessionCreateArg);

pub const POLARIS_SESSION_DESTROY: u32 =
    iow!(MAGIC, 0x04, PolarisSessionDestroyArg);

pub const POLARIS_SESSION_GET_STATS: u32 =
    iowr!(MAGIC, 0x05, PolarisSessionGetStatsArg);

pub const POLARIS_SESSION_BRANCH: u32 =
    iowr!(MAGIC, 0x06, PolarisSessionBranchArg);

pub const POLARIS_BLOCK_RESERVE: u32 =
    iowr!(MAGIC, 0x07, PolarisBlockReserveArg);

pub const POLARIS_BLOCK_RELEASE: u32 =
    iow!(MAGIC, 0x08, PolarisBlockReleaseArg);

pub const POLARIS_BLOCK_TOUCH: u32 =
    iow!(MAGIC, 0x09, PolarisBlockTouchArg);

pub const POLARIS_BLOCK_GET_STATE: u32 =
    iowr!(MAGIC, 0x0A, PolarisBlockGetStateArg);

pub const POLARIS_GET_DECISION: u32 =
    iowr!(MAGIC, 0x0B, PolarisGetDecisionArg);

pub const POLARIS_COMPLETE_OPERATION: u32 =
    iowr!(MAGIC, 0x0C, PolarisCompleteOperationArg);

pub const POLARIS_GET_GLOBAL_STATS: u32 =
    iowr!(MAGIC, 0x0D, PolarisGetGlobalStatsArg);

pub const POLARIS_LIST_SESSIONS: u32 =
    iowr!(MAGIC, 0x0E, PolarisListSessionsArg);

pub const POLARIS_SET_POLICY: u32 =
    iow!(MAGIC, 0x0F, PolarisSetPolicyArg);

pub const POLARIS_REGISTER_VASPACE: u32 =
    iow!(MAGIC, 0x10, PolarisRegisterVaSpaceArg);

pub const POLARIS_UNREGISTER_VASPACE: u32 =
    iow!(MAGIC, 0x11, PolarisUnregisterVaSpaceArg);

pub const POLARIS_REGISTER_STATIC_BLOCK: u32 =
    iow!(MAGIC, 0x12, PolarisRegisterStaticBlockArg);

pub const POLARIS_UNMAP_STATIC_BLOCK: u32 =
    iow!(MAGIC, 0x13, PolarisUnmapStaticBlockArg);

pub const POLARIS_REGISTER_BLOCK_MAPPING: u32 =
    iow!(MAGIC, 0x14, PolarisRegisterBlockMappingArg);

pub const POLARIS_UNMAP_BLOCK_MAPPINGS: u32 =
    iowr!(MAGIC, 0x15, PolarisUnmapBlockMappingsArg);

pub const POLARIS_SPILL_BLOCK: u32 =
    iowr!(MAGIC, 0x16, PolarisSpillBlockArg);

pub const POLARIS_REGISTER_BLOCK_BACKING: u32 =
    iow!(MAGIC, 0x17, PolarisRegisterBlockBackingArg);

pub const POLARIS_PROBE_RM_PHYS: u32 =
    iowr!(MAGIC, 0x18, PolarisProbeRmPhysArg);

pub const POLARIS_PROBE_RM_COPY: u32 =
    iowr!(MAGIC, 0x19, PolarisProbeRmCopyArg);

pub const POLARIS_RM_COPY: u32 =
    iowr!(MAGIC, 0x1A, PolarisRmCopyArg);

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

/// Register a v4 fault-capable VA-space with polaris.ko.
pub fn register_vaspace(fd: i32, arg: &PolarisRegisterVaSpaceArg) -> Result<(), i32> {
    ioctl_write(fd, POLARIS_REGISTER_VASPACE, arg)
}

/// Query a logical block's current state.
pub fn block_get_state(fd: i32, arg: &mut PolarisBlockGetStateArg) -> Result<(), i32> {
    ioctl_read(fd, POLARIS_BLOCK_GET_STATE, arg)
}

/// Branch a session, sharing its current block table entries with COW semantics.
pub fn session_branch(fd: i32, arg: &mut PolarisSessionBranchArg) -> Result<(), i32> {
    ioctl_read(fd, POLARIS_SESSION_BRANCH, arg)
}

/// Unregister a v4 VA-space and any static M2 blocks attached to it.
pub fn unregister_vaspace(fd: i32, arg: &PolarisUnregisterVaSpaceArg) -> Result<(), i32> {
    ioctl_write(fd, POLARIS_UNREGISTER_VASPACE, arg)
}

/// Register an M2 static external allocation block for synthetic fault tests.
pub fn register_static_block(fd: i32, arg: &PolarisRegisterStaticBlockArg) -> Result<(), i32> {
    ioctl_write(fd, POLARIS_REGISTER_STATIC_BLOCK, arg)
}

/// Unmap an M2/M3 diagnostic static block after it has been fault-mapped once.
pub fn unmap_static_block(fd: i32, arg: &PolarisUnmapStaticBlockArg) -> Result<(), i32> {
    ioctl_write(fd, POLARIS_UNMAP_STATIC_BLOCK, arg)
}

/// Register a v4 logical block mapping for spill teardown.
pub fn register_block_mapping(fd: i32, arg: &PolarisRegisterBlockMappingArg) -> Result<(), i32> {
    ioctl_write(fd, POLARIS_REGISTER_BLOCK_MAPPING, arg)
}

/// Attach RM allocation backing to an existing v4 logical block.
pub fn register_block_backing(fd: i32, arg: &PolarisRegisterBlockBackingArg) -> Result<(), i32> {
    ioctl_write(fd, POLARIS_REGISTER_BLOCK_BACKING, arg)
}

/// Unmap all UVM-observed v4 mappings for a logical block.
pub fn unmap_block_mappings(fd: i32, arg: &mut PolarisUnmapBlockMappingsArg) -> Result<(), i32> {
    ioctl_read(fd, POLARIS_UNMAP_BLOCK_MAPPINGS, arg)
}

/// Unmap observed UVM mappings for a logical block and queue an OFFLOAD decision.
pub fn spill_block(fd: i32, arg: &mut PolarisSpillBlockArg) -> Result<(), i32> {
    ioctl_read(fd, POLARIS_SPILL_BLOCK, arg)
}

/// Probe whether UVM/RM can expose GPU physical addresses for an RM-backed logical block.
pub fn probe_rm_phys(fd: i32, arg: &mut PolarisProbeRmPhysArg) -> Result<(), i32> {
    ioctl_read(fd, POLARIS_PROBE_RM_PHYS, arg)
}

/// Probe whether UVM CE can copy bytes to and from an RM-backed logical block.
pub fn probe_rm_copy(fd: i32, arg: &mut PolarisProbeRmCopyArg) -> Result<(), i32> {
    ioctl_read(fd, POLARIS_PROBE_RM_COPY, arg)
}

/// Copy between an RM-backed logical block and a userspace CPU buffer.
pub fn rm_copy(fd: i32, arg: &mut PolarisRmCopyArg) -> Result<(), i32> {
    ioctl_read(fd, POLARIS_RM_COPY, arg)
}
