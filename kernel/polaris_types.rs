// SPDX-License-Identifier: GPL-2.0

//! Core data types for the POLARIS kernel module.
//!
//! The kernel↔userspace ABI types (ioctl argument structs, enums, and constants)
//! live in `polaris_abi.rs`.  This module re-exports them and adds kernel-only
//! extensions: `impl_flags!` wrappers, kernel-internal management structs, and
//! `FromBytes`/`AsBytes` trait impls.

#[path = "polaris_abi.rs"]
mod polaris_abi;
pub use polaris_abi::*;

use kernel::prelude::*;
use kernel::impl_flags;

// ─── IOCTL Command Codes ────────────────────────────────────────────────────

/// Magic number 'P' = 0x50
pub const POLARIS_IOCTL_MAGIC: u32 = 0x50;

pub const POLARIS_REGISTER_GPU: u32 =
    kernel::ioctl::_IOW::<PolarisRegisterGpuArg>(POLARIS_IOCTL_MAGIC, 0x01);

pub const POLARIS_REGISTER_VA_RANGE: u32 =
    kernel::ioctl::_IOWR::<PolarisRegisterVaRangeArg>(POLARIS_IOCTL_MAGIC, 0x02);

pub const POLARIS_SESSION_CREATE: u32 =
    kernel::ioctl::_IOWR::<PolarisSessionCreateArg>(POLARIS_IOCTL_MAGIC, 0x03);

pub const POLARIS_SESSION_DESTROY: u32 =
    kernel::ioctl::_IOW::<PolarisSessionDestroyArg>(POLARIS_IOCTL_MAGIC, 0x04);

pub const POLARIS_SESSION_GET_STATS: u32 =
    kernel::ioctl::_IOWR::<PolarisSessionGetStatsArg>(POLARIS_IOCTL_MAGIC, 0x05);

pub const POLARIS_SESSION_BRANCH: u32 =
    kernel::ioctl::_IOWR::<PolarisSessionBranchArg>(POLARIS_IOCTL_MAGIC, 0x06);

pub const POLARIS_BLOCK_RESERVE: u32 =
    kernel::ioctl::_IOWR::<PolarisBlockReserveArg>(POLARIS_IOCTL_MAGIC, 0x07);

pub const POLARIS_BLOCK_RELEASE: u32 =
    kernel::ioctl::_IOW::<PolarisBlockReleaseArg>(POLARIS_IOCTL_MAGIC, 0x08);

pub const POLARIS_BLOCK_TOUCH: u32 =
    kernel::ioctl::_IOW::<PolarisBlockTouchArg>(POLARIS_IOCTL_MAGIC, 0x09);

pub const POLARIS_BLOCK_GET_STATE: u32 =
    kernel::ioctl::_IOWR::<PolarisBlockGetStateArg>(POLARIS_IOCTL_MAGIC, 0x0A);

pub const POLARIS_GET_DECISION: u32 =
    kernel::ioctl::_IOWR::<PolarisGetDecisionArg>(POLARIS_IOCTL_MAGIC, 0x0B);

pub const POLARIS_COMPLETE_OPERATION: u32 =
    kernel::ioctl::_IOWR::<PolarisCompleteOperationArg>(POLARIS_IOCTL_MAGIC, 0x0C);

pub const POLARIS_GET_GLOBAL_STATS: u32 =
    kernel::ioctl::_IOWR::<PolarisGetGlobalStatsArg>(POLARIS_IOCTL_MAGIC, 0x0D);

pub const POLARIS_LIST_SESSIONS: u32 =
    kernel::ioctl::_IOWR::<PolarisListSessionsArg>(POLARIS_IOCTL_MAGIC, 0x0E);

pub const POLARIS_SET_POLICY: u32 =
    kernel::ioctl::_IOW::<PolarisSetPolicyArg>(POLARIS_IOCTL_MAGIC, 0x0F);

// ─── Block Flags (kernel-side type-safe wrappers) ───────────────────────────

impl_flags!(
    /// Bitmask of flags attached to a KV block.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct PolarisBlockFlags(u32);

    /// Individual block flag.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PolarisBlockFlag {
        /// refcount > 1, this block is COW-shared across sessions.
        Shared = 1 << 0,
    }
);

impl_flags!(
    /// Bitmask of flags passed to BLOCK_RESERVE.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct PolarisGrowFlags(u32);

    /// Individual BLOCK_RESERVE flag.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PolarisGrowFlag {
        /// Allow overlapping an existing shared block (triggers COW break).
        Overwrite = 1 << 0,
    }
);

// ─── Kernel-Internal Data Structures ────────────────────────────────────────

/// Represents a single KV Cache block — the GPU equivalent of a physical page.
#[derive(Clone, Debug)]
pub struct PolarisBlock {
    pub block_id: u64,
    pub session_id: u64,
    pub token_start: u32,
    pub token_count: u32,
    pub home_gpu: u32,
    pub gpu_vaddr: u64,
    pub gpu_phys_handle: u64,
    pub cpu_buf_addr: u64,
    pub size_bytes: u64,
    pub refcount: u64,
    pub state: PolarisBlockState,
    pub flags: PolarisBlockFlags,
    pub phase: PolarisPhase,
    pub last_touch_ns: u64,
    pub map_time_ns: u64,
    /// Opaque phys handle of the source block for in-flight COW_BREAK.
    /// Set when the decision is queued; used by requeue_decision on retry.
    pub cow_src_handle: u64,
    /// Number of consecutive COMPLETE_OPERATION failures for this block.
    /// Reset to 0 on success. After MAX_RETRIES (=3) → EVICTED.
    pub retry_count: u32,
    /// The decision_id currently in-flight for this block. Set when a
    /// decision is queued; cleared (set to 0) when the daemon reports
    /// completion. Used to match COMPLETE_OPERATION results to blocks.
    pub pending_decision_id: u64,
    /// Fault metadata for the in-flight driver notification path.
    pub pending_fault_id: u64,
    pub pending_generation: u64,
    pub fault_timeout_ms: u32,
    /// Stack-allocated completion pointer used by synchronous fault waits.
    /// Set before dropping the lock, consumed by COMPLETE_OPERATION.
    /// NULL when no waiter is pending.
    pub completion_ptr: *mut kernel::bindings::completion,
}

/// Maximum retries before a block is evicted (G4 contract).
pub const POLARIS_MAX_RETRIES: u32 = 3;

/// Maximum number of decisions that can be queued in pending_decisions.
/// When the queue reaches this limit, logical reserve and session teardown
/// return ENOMEM to the caller.
pub const POLARIS_MAX_PENDING_DECISIONS: usize = 1024;

// SAFETY: All PolarisBlock fields are accessed exclusively under the
// POLARIS_STATE mutex.  completion_ptr is set and cleared under the same
// lock and never accessed from interrupt context.
unsafe impl Send for PolarisBlock {}

/// A session represents one LLM inference request.
#[derive(Debug)]
pub struct PolarisSession {
    pub session_id: u64,
    pub home_gpu: u32,
    pub gpu_vas_base: u64,
    pub gpu_vas_size: u64,
    pub beam_width: u32,
    pub bytes_per_token: u64,
    pub parent_session_id: u64,
    pub priority: u32,
    pub block_ids: KVec<u64>, // ordered list of block IDs in token order
}

/// GPU registration info maintained by the kernel module.
#[derive(Clone, Debug)]
pub struct PolarisGpu {
    pub gpu_id: u32,
    pub gpu_uuid: [u8; 16],
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub budget_bytes: u64,
    pub pressure_score: u64,
    pub cpu_pool_total_bytes: u64,
    pub cpu_pool_used_bytes: u64,
    pub va_range_id: u64,
    pub va_range_base: u64,
    pub va_range_length: u64,
    pub va_block_size: u64,
    pub va_range_flags: u32,
    pub va_range_registered: bool,
    pub healthy: bool,
}

#[derive(Clone, Debug)]
pub struct PolarisFault {
    pub fault_id: u64,
    pub generation: u64,
    pub gpu_id: u32,
    pub fault_address: u64,
    pub block_id: u64,
    pub access_type: u32,
    pub state: u32,
    pub enqueue_ns: u64,
    pub deadline_ns: u64,
    pub resolved_ns: u64,
}

// ─── Trait impls for user/kernel boundary crossing ───────────────────────────
//
// `FromBytes`: Any bit pattern is valid. All fields are integers/integer arrays,
// so every possible byte sequence represents a valid value.
//
// `AsBytes`: No uninitialized bytes (no padding). These are #[repr(C)] structs
// with naturally-aligned integer fields and no padding.
//
// SAFETY: All fields in these structs are integer types (for which any bit
// pattern is valid). The structs are #[repr(C)] with no padding.

unsafe impl kernel::transmute::FromBytes for PolarisRegisterGpuArg {}
unsafe impl kernel::transmute::AsBytes for PolarisRegisterGpuArg {}

unsafe impl kernel::transmute::FromBytes for PolarisRegisterVaRangeArg {}
unsafe impl kernel::transmute::AsBytes for PolarisRegisterVaRangeArg {}

unsafe impl kernel::transmute::FromBytes for PolarisSessionCreateArg {}
unsafe impl kernel::transmute::AsBytes for PolarisSessionCreateArg {}

unsafe impl kernel::transmute::FromBytes for PolarisSessionDestroyArg {}
unsafe impl kernel::transmute::AsBytes for PolarisSessionDestroyArg {}

unsafe impl kernel::transmute::FromBytes for PolarisSessionGetStatsArg {}
unsafe impl kernel::transmute::AsBytes for PolarisSessionGetStatsArg {}

unsafe impl kernel::transmute::FromBytes for PolarisSessionBranchArg {}
unsafe impl kernel::transmute::AsBytes for PolarisSessionBranchArg {}

unsafe impl kernel::transmute::FromBytes for PolarisBlockReserveArg {}
unsafe impl kernel::transmute::AsBytes for PolarisBlockReserveArg {}

unsafe impl kernel::transmute::FromBytes for PolarisBlockReleaseArg {}
unsafe impl kernel::transmute::AsBytes for PolarisBlockReleaseArg {}

unsafe impl kernel::transmute::FromBytes for PolarisBlockTouchArg {}
unsafe impl kernel::transmute::AsBytes for PolarisBlockTouchArg {}

unsafe impl kernel::transmute::FromBytes for PolarisBlockGetStateArg {}
unsafe impl kernel::transmute::AsBytes for PolarisBlockGetStateArg {}

unsafe impl kernel::transmute::FromBytes for PolarisDecision {}
unsafe impl kernel::transmute::AsBytes for PolarisDecision {}

unsafe impl kernel::transmute::FromBytes for PolarisCompleteOperationArg {}
unsafe impl kernel::transmute::AsBytes for PolarisCompleteOperationArg {}

unsafe impl kernel::transmute::FromBytes for PolarisGetGlobalStatsArg {}
unsafe impl kernel::transmute::AsBytes for PolarisGetGlobalStatsArg {}

// PolarisGetDecisionArg contains an array of PolarisDecision.
// Since PolarisDecision: FromBytes, the array and struct are also FromBytes.
unsafe impl kernel::transmute::FromBytes for PolarisGetDecisionArg {}
unsafe impl kernel::transmute::AsBytes for PolarisGetDecisionArg {}

unsafe impl kernel::transmute::FromBytes for PolarisSetPolicyArg {}
unsafe impl kernel::transmute::AsBytes for PolarisSetPolicyArg {}

unsafe impl kernel::transmute::FromBytes for PolarisListSessionsArg {}
unsafe impl kernel::transmute::AsBytes for PolarisListSessionsArg {}
