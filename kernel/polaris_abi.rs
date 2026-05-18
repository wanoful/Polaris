// SPDX-License-Identifier: GPL-2.0
// AUTO-GENERATED SOURCE OF TRUTH for the POLARIS kernel↔userspace ABI.
//
// This file contains only pure Rust types — no kernel-specific dependencies
// (no KVec, no kernel::ioctl, no FromBytes).  It is:
//   - `mod`-ed by kernel/polaris_types.rs (re-exported as the kernel type set)
//   - copied by libpolaris/build.rs into the userspace type set
//
// Editing rules:
//   - NEVER add kernel-only dependencies here
//   - When changing an ioctl struct, re-check both sides compile

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

// ─── Phase ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum PolarisPhase {
    Prefill = 1,
    Decode = 2,
}

// ─── Eviction Policy (Phase 2b) ─────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum PolarisEvictionPolicy {
    /// Victim = block with the oldest map_time_ns.
    Fifo = 0,
    /// Victim = block with the oldest last_touch_ns.
    Lru = 1,
    /// Victim = highest scoring function value (prefill preferred, decode/recent protected).
    PhaseAware = 2,
}

// ─── Decision Opcodes ───────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum PolarisDecisionOp {
    Alloc = 0,
    Free = 1,
    MapExisting = 2,
    Unmap = 3,
    Offload = 4,
    Reload = 5,
    CowBreak = 6,
}

// ─── Flag Constants (raw u32 — the ioctl ABI uses plain integers) ───────────

pub const POLARIS_BLOCK_FLAG_SHARED: u32 = 1 << 0;
pub const POLARIS_RESERVE_FLAG_OVERWRITE: u32 = 1 << 0;
pub const POLARIS_RESERVE_FLAG_READ_MOSTLY: u32 = 1 << 1;
pub const POLARIS_RESERVE_FLAG_WRITE_NEW: u32 = 1 << 2;
pub const POLARIS_RESERVE_FLAG_FULL_OVERWRITE_NO_PRESERVE: u32 = 1 << 3;
pub const POLARIS_RELEASE_FLAG_STREAM_QUIESCED: u32 = 1 << 0;

pub const POLARIS_DEFAULT_FAULT_TIMEOUT_MS: u32 = 5000;

// ─── Constants ──────────────────────────────────────────────────────────────

pub const POLARIS_MAX_DECISIONS_PER_POLL: usize = 16;
pub const POLARIS_MAX_SESSIONS_PER_LIST: usize = 64;
pub const POLARIS_DEFAULT_BYTES_PER_TOKEN: u64 = 524_288;

// ═════════════════════════════════════════════════════════════════════════════
// IOCTL Argument Structs (C-compatible, #[repr(C)] — cross the user/kernel boundary)
// ═════════════════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisRegisterGpuArg {
    pub gpu_id: u32,
    pub total_bytes: u64,
    pub budget_bytes: u64,
    pub cpu_pool_bytes: u64,
    pub numa_node: u32,
    pub _reserved: u32,
    pub _reserved2: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisRegisterVaRangeArg {
    pub range_id: u64,
    pub gpu_id: u32,
    pub flags: u32,
    pub base: u64,
    pub length: u64,
    pub block_size: u64,
    pub _reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisSessionCreateArg {
    pub session_id: u64,
    pub home_gpu: u32,
    pub beam_width: u32,
    pub gpu_vas_bytes: u64,
    pub bytes_per_token: u64,
    pub priority: u32,
    pub _reserved: u32,
    pub _reserved2: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisSessionDestroyArg {
    pub session_id: u64,
    pub _reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisSessionGetStatsArg {
    pub session_id: u64,
    pub home_gpu: u32,
    pub beam_width: u32,
    pub num_blocks: u32,
    pub _reserved: u32,
    pub total_bytes: u64,
    pub _reserved2: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisSessionBranchArg {
    pub parent_session_id: u64,
    pub child_session_id: u64,
    pub _reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisBlockReserveArg {
    pub session_id: u64,
    pub token_start: u32,
    pub token_count: u32,
    pub phase: u32,
    pub flags: u32,
    pub block_id: u64,
    pub gpu_vaddr: u64,
    pub _reserved: [u64; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisBlockReleaseArg {
    pub session_id: u64,
    pub token_start: u32,
    pub token_count: u32,
    pub flags: u32,
    pub _reserved: u32,
    pub _reserved2: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisBlockTouchArg {
    pub session_id: u64,
    pub token_start: u64,
    pub token_count: u64,
    pub _reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
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

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisDecision {
    pub decision_id: u64,
    pub fault_id: u64,
    pub generation: u64,
    pub op: u32,
    pub gpu_id: u32,
    pub block_id: u64,
    pub session_id: u64,
    pub src_handle: u64,
    pub dst_handle: u64,
    pub src_vaddr: u64,
    pub dst_vaddr: u64,
    pub size_bytes: u64,
    pub cpu_addr: u64,
    pub access_flags: u32,
    pub timeout_ms: u32,
    pub _reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisGetDecisionArg {
    pub count: u32,
    pub _reserved: u32,
    pub decisions: [PolarisDecision; POLARIS_MAX_DECISIONS_PER_POLL],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisCompleteOperationArg {
    pub decision_id: u64,
    pub generation: u64,
    pub result: i32,
    pub _reserved: u32,
    pub output_handle: u64,
    pub output_cpu_addr: u64,
    pub _reserved2: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
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
    pub memory_saved_vs_naive: u64,
    pub total_gpu_bytes: u64,
    pub used_gpu_bytes: u64,
    pub cpu_pool_total: u64,
    pub cpu_pool_used: u64,
    pub eviction_policy: u32,
    pub _policy_pad: u32,
    pub offload_count: u64,
    pub reload_count: u64,
    pub total_evictions: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisSetPolicyArg {
    pub policy: u32,
    pub _reserved: u32,
    pub _reserved2: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisListSessionsArg {
    pub count: u32,
    pub _reserved: u32,
    pub session_ids: [u64; POLARIS_MAX_SESSIONS_PER_LIST],
}

impl Default for PolarisListSessionsArg {
    fn default() -> Self {
        Self {
            count: 0,
            _reserved: 0,
            session_ids: [0u64; POLARIS_MAX_SESSIONS_PER_LIST],
        }
    }
}
