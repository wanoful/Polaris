// SPDX-License-Identifier: GPL-2.0

//! Core data types for the POLARIS kernel module.
//!
//! This module defines:
//! - IOCTL command codes (used by both kernel and userspace)
//! - IOCTL argument structs (C-compatible, #[repr(C)])
//! - Kernel-internal data structures (block table, session table, GPU registry)
//! - Decision protocol types

use kernel::prelude::*;

// ─── IOCTL Command Codes ────────────────────────────────────────────────────

/// Magic number 'P' = 0x50
pub const POLARIS_IOCTL_MAGIC: u32 = 0x50;

pub const POLARIS_REGISTER_GPU: u32 =
    kernel::ioctl::_IOW::<PolarisRegisterGpuArg>(POLARIS_IOCTL_MAGIC, 0x01);

pub const POLARIS_SESSION_CREATE: u32 =
    kernel::ioctl::_IOWR::<PolarisSessionCreateArg>(POLARIS_IOCTL_MAGIC, 0x02);

pub const POLARIS_SESSION_DESTROY: u32 =
    kernel::ioctl::_IOW::<PolarisSessionDestroyArg>(POLARIS_IOCTL_MAGIC, 0x03);

pub const POLARIS_SESSION_GET_STATS: u32 =
    kernel::ioctl::_IOWR::<PolarisSessionGetStatsArg>(POLARIS_IOCTL_MAGIC, 0x04);

pub const POLARIS_SESSION_BRANCH: u32 =
    kernel::ioctl::_IOWR::<PolarisSessionBranchArg>(POLARIS_IOCTL_MAGIC, 0x05);

pub const POLARIS_BLOCK_GROW: u32 =
    kernel::ioctl::_IOWR::<PolarisBlockGrowArg>(POLARIS_IOCTL_MAGIC, 0x06);

pub const POLARIS_BLOCK_FREE: u32 =
    kernel::ioctl::_IOW::<PolarisBlockFreeArg>(POLARIS_IOCTL_MAGIC, 0x07);

pub const POLARIS_BLOCK_TOUCH: u32 =
    kernel::ioctl::_IOW::<PolarisBlockTouchArg>(POLARIS_IOCTL_MAGIC, 0x08);

pub const POLARIS_BLOCK_GET_STATE: u32 =
    kernel::ioctl::_IOWR::<PolarisBlockGetStateArg>(POLARIS_IOCTL_MAGIC, 0x09);

pub const POLARIS_GET_DECISION: u32 =
    kernel::ioctl::_IOWR::<PolarisGetDecisionArg>(POLARIS_IOCTL_MAGIC, 0x0A);

pub const POLARIS_COMPLETE_OPERATION: u32 =
    kernel::ioctl::_IOWR::<PolarisCompleteOperationArg>(POLARIS_IOCTL_MAGIC, 0x0B);

pub const POLARIS_GET_GLOBAL_STATS: u32 =
    kernel::ioctl::_IOWR::<PolarisGetGlobalStatsArg>(POLARIS_IOCTL_MAGIC, 0x0C);

// ─── Block States ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum PolarisBlockState {
    FreePending = 0,
    Resident = 1,
    AllocPending = 2,
    Unmapped = 3,
    CpuOffloaded = 4,
    OffloadPending = 5,
    ReloadPending = 6,
    CowPending = 7,
    Evicted = 8,
}

// ─── Block Flags ────────────────────────────────────────────────────────────

pub const POLARIS_BLOCK_FLAG_SHARED: u32 = 1 << 0;

// ─── Phase Constants ────────────────────────────────────────────────────────

pub const POLARIS_PHASE_PREFILL: u32 = 1;
pub const POLARIS_PHASE_DECODE: u32 = 2;

// ─── Decision Opcodes ───────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum PolarisDecisionOp {
    Alloc = 0,
    Free = 1,
    Map = 2,
    Unmap = 3,
    Offload = 4,
    Reload = 5,
    CowBreak = 6,
}

// ─── Constants ──────────────────────────────────────────────────────────────

pub const POLARIS_MAX_DECISIONS_PER_POLL: usize = 16;

pub const POLARIS_GROW_FLAG_OVERWRITE: u32 = 1 << 0;

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
    pub flags: u32,
    pub phase: u32,
    pub last_touch_ns: u64,
    pub map_time_ns: u64,
    /// Number of consecutive COMPLETE_OPERATION failures for this block.
    /// Reset to 0 on success. After MAX_RETRIES (=3) → EVICTED.
    pub retry_count: u32,
    /// The decision_id currently in-flight for this block. Set when a
    /// decision is queued; cleared (set to 0) when the daemon reports
    /// completion. Used to match COMPLETE_OPERATION results to blocks.
    pub pending_decision_id: u64,
}

/// Maximum retries before a block is evicted (G4 contract).
pub const POLARIS_MAX_RETRIES: u32 = 3;

/// A session represents one LLM inference request.
#[derive(Debug)]
pub struct PolarisSession {
    pub session_id: u64,
    pub home_gpu: u32,
    pub gpu_vas_base: u64,
    pub gpu_vas_size: u64,
    pub gpu_vas_cursor: u64,
    pub beam_width: u32,
    pub parent_session_id: u64,
    pub block_ids: KVec<u64>, // ordered list of block IDs in token order
}

/// GPU registration info maintained by the kernel module.
#[derive(Clone, Debug)]
pub struct PolarisGpu {
    pub gpu_id: u32,
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub budget_bytes: u64,
    pub pressure_score: u64,
    pub cpu_pool_total_bytes: u64,
    pub cpu_pool_used_bytes: u64,
    pub healthy: bool,
}

// ═════════════════════════════════════════════════════════════════════════════
// IOCTL Argument Structs (C-compatible, #[repr(C)] — cross the user/kernel boundary)
// ═════════════════════════════════════════════════════════════════════════════

/// Arg for POLARIS_REGISTER_GPU (daemon → kernel).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisRegisterGpuArg {
    pub gpu_id: u32,
    pub total_bytes: u64,
    pub budget_bytes: u64,
    pub cpu_pool_bytes: u64,
    pub numa_node: u32,
    pub _reserved: u32,
    pub _reserved2: [u64; 2],
}

/// Arg for POLARIS_SESSION_CREATE (workload → kernel).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisSessionCreateArg {
    pub session_id: u64,
    pub home_gpu: u32,
    pub beam_width: u32,
    pub gpu_vas_bytes: u64,
    pub _reserved: [u64; 4],
}

/// Arg for POLARIS_SESSION_DESTROY.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisSessionDestroyArg {
    pub session_id: u64,
    pub _reserved: [u64; 4],
}

/// Arg for POLARIS_SESSION_GET_STATS.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisSessionGetStatsArg {
    pub session_id: u64,
    pub home_gpu: u32,
    pub beam_width: u32,
    pub num_blocks: u32,
    pub _reserved: u32,
    pub total_bytes: u64,
    pub _reserved2: [u64; 2],
}

/// Arg for POLARIS_SESSION_BRANCH (COW fork).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisSessionBranchArg {
    pub parent_session_id: u64,
    pub child_session_id: u64,
    pub _reserved: [u64; 4],
}

/// Arg for POLARIS_BLOCK_GROW (page-fault entry).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisBlockGrowArg {
    pub session_id: u64,
    pub token_start: u32,
    pub token_count: u32,
    pub flags: u32,
    pub _reserved: u32,
    pub block_id: u64,
    pub ret_code: i32,
    pub _reserved2: u32,
    pub _reserved3: [u64; 2],
}

/// Arg for POLARIS_BLOCK_FREE.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisBlockFreeArg {
    pub session_id: u64,
    pub token_start: u32,
    pub token_count: u32,
    pub _reserved: [u64; 4],
}

/// Arg for POLARIS_BLOCK_TOUCH (updates LRU timestamp).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisBlockTouchArg {
    pub session_id: u64,
    pub token_start: u64,
    pub token_count: u64,
    pub _reserved: [u64; 4],
}

/// Arg for POLARIS_BLOCK_GET_STATE.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisBlockGetStateArg {
    pub session_id: u64,
    pub token_start: u32,
    pub token_count: u32,
    pub block_id: u64,
    pub state: u32,
    pub refcount: u64,
    pub gpu_vaddr: u64,
    pub _reserved: [u64; 2],
}

/// A single decision queued by the kernel for daemon execution.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisDecision {
    pub decision_id: u64,
    pub op: u32,
    pub gpu_id: u32,
    pub block_id: u64,
    pub session_id: u64,
    pub src_handle: u64,
    pub dst_vaddr: u64,
    pub size_bytes: u64,
    pub cpu_addr: u64,
    pub _reserved: [u64; 4],
}

/// Arg for POLARIS_GET_DECISION (daemon polls this).
#[repr(C)]
pub struct PolarisGetDecisionArg {
    pub count: u32,
    pub _reserved: u32,
    pub decisions: [PolarisDecision; POLARIS_MAX_DECISIONS_PER_POLL],
}

/// Arg for POLARIS_COMPLETE_OPERATION (daemon reports result).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisCompleteOperationArg {
    pub decision_id: u64,
    pub result: i32,
    pub _reserved: u32,
    pub output_handle: u64,
    pub output_cpu_addr: u64,
    pub _reserved2: [u64; 2],
}

/// Arg for POLARIS_GET_GLOBAL_STATS.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisGetGlobalStatsArg {
    pub total_gpus: u32,
    pub total_sessions: u32,
    pub total_blocks: u32,
    pub blocks_resident: u32,
    pub blocks_offloaded: u32,
    pub blocks_evicted: u32,
    pub shared_gpu_bytes: u64,
    pub private_gpu_bytes: u64,
    pub cow_break_count: u64,
    pub cow_copy_bytes: u64,
    pub total_gpu_bytes: u64,
    pub used_gpu_bytes: u64,
    pub cpu_pool_total: u64,
    pub cpu_pool_used: u64,
    pub _reserved: [u64; 4],
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

unsafe impl kernel::transmute::FromBytes for PolarisSessionCreateArg {}
unsafe impl kernel::transmute::AsBytes for PolarisSessionCreateArg {}

unsafe impl kernel::transmute::FromBytes for PolarisSessionDestroyArg {}
unsafe impl kernel::transmute::AsBytes for PolarisSessionDestroyArg {}

unsafe impl kernel::transmute::FromBytes for PolarisSessionGetStatsArg {}
unsafe impl kernel::transmute::AsBytes for PolarisSessionGetStatsArg {}

unsafe impl kernel::transmute::FromBytes for PolarisSessionBranchArg {}
unsafe impl kernel::transmute::AsBytes for PolarisSessionBranchArg {}

unsafe impl kernel::transmute::FromBytes for PolarisBlockGrowArg {}
unsafe impl kernel::transmute::AsBytes for PolarisBlockGrowArg {}

unsafe impl kernel::transmute::FromBytes for PolarisBlockFreeArg {}
unsafe impl kernel::transmute::AsBytes for PolarisBlockFreeArg {}

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
