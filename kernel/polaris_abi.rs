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
pub const POLARIS_RESERVE_FLAG_DEFER_FAULT: u32 = 1 << 4;
pub const POLARIS_RELEASE_FLAG_STREAM_QUIESCED: u32 = 1 << 0;
pub const POLARIS_RELEASE_FLAG_CALLER_OWNS_BACKING: u32 = 1 << 1;
pub const POLARIS_REGISTER_GPU_FLAG_TRANSIENT: u32 = 1 << 0;

pub const POLARIS_RM_PHYS_FLAG_CONTIGUOUS: u64 = 1 << 0;
pub const POLARIS_RM_PHYS_FLAG_SYSMEM: u64 = 1 << 1;
pub const POLARIS_RM_PHYS_FLAG_EGM: u64 = 1 << 2;
pub const POLARIS_RM_PHYS_FLAG_FABRICMEM: u64 = 1 << 3;
pub const POLARIS_DECISION_FLAG_SOURCE_BLOCK_ID_VALID: u64 = 1 << 0;
pub const POLARIS_RM_COPY_NO_MISMATCH: u64 = u64::MAX;
pub const POLARIS_RM_COPY_TO_CPU: u32 = 0;
pub const POLARIS_RM_COPY_FROM_CPU: u32 = 1;
pub const POLARIS_KV_HINT_FLAG_CLEAR: u32 = 1 << 0;

pub const POLARIS_DEFAULT_FAULT_TIMEOUT_MS: u32 = 5000;

// ─── Constants ──────────────────────────────────────────────────────────────

pub const POLARIS_MAX_DECISIONS_PER_POLL: usize = 16;
pub const POLARIS_MAX_SESSIONS_PER_LIST: usize = 64;
pub const POLARIS_DEFAULT_BYTES_PER_TOKEN: u64 = 524_288;
pub const POLARIS_DEFAULT_SESSION_VA_BYTES: u64 = 4 * 1024 * 1024 * 1024; // 4 GiB
pub const POLARIS_VA_ALIGNMENT: u64 = 2 * 1024 * 1024; // 2 MiB (GPU allocation granularity)

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

// ─── v4: fault-capable VA-space registration ────────────────────────────────
//
// The shim creates a fault-capable, externally-owned GPU VA-space
// (NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | IS_EXTERNALLY_OWNED),
// hands it to UVM via UvmRegisterGpuVaSpace, then announces it to polaris.ko
// with POLARIS_REGISTER_VASPACE. va_space_token is the user RM VA-space
// handle UVM also stores in gpu_va_space->user_rm_va_space — polaris.ko uses
// it as the per-(worker, gpu) key when the UVM fault hook dispatches into it.
//
// managed_base / managed_length carve the VA window inside the polaris-owned
// VA-space within which the fault hook is allowed to install PTEs. Anything
// outside falls back to NOT_MINE so stock UVM handles it (or faults fatally
// when it's truly unbacked).

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisRegisterVaSpaceArg {
    pub gpu_id: u32,
    pub _reserved0: u32,
    pub rm_client_token: u64,
    pub va_space_token: u64,
    pub managed_base: u64,
    pub managed_length: u64,
    pub _reserved1: [u64; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisUnregisterVaSpaceArg {
    pub gpu_id: u32,
    pub _reserved0: u32,
    pub rm_client_token: u64,
    pub va_space_token: u64,
    pub _reserved1: [u64; 1],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisRegisterStaticBlockArg {
    pub gpu_id: u32,
    pub rm_control_fd: i32,
    pub rm_client_token: u64,
    pub va_space_token: u64,
    pub base: u64,
    pub length: u64,
    pub offset: u64,
    pub h_client: u32,
    pub h_memory: u32,
    pub _reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisUnmapStaticBlockArg {
    pub gpu_id: u32,
    pub _reserved0: u32,
    pub rm_client_token: u64,
    pub va_space_token: u64,
    pub base: u64,
    pub length: u64,
    pub _reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisRegisterBlockMappingArg {
    pub block_id: u64,
    pub gpu_id: u32,
    pub _reserved0: u32,
    pub rm_client_token: u64,
    pub va_space_token: u64,
    pub base: u64,
    pub length: u64,
    pub _reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisRegisterBlockBackingArg {
    pub block_id: u64,
    pub gpu_id: u32,
    pub rm_control_fd: i32,
    pub h_client: u32,
    pub h_memory: u32,
    pub length: u64,
    pub offset: u64,
    pub _reserved: [u64; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisUnmapBlockMappingsArg {
    pub block_id: u64,
    pub flags: u32,
    pub unmapped_count: u32,
    pub _reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisSpillBlockArg {
    pub block_id: u64,
    pub flags: u32,
    pub unmapped_count: u32,
    pub decision_id: u64,
    pub _reserved: [u64; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisProbeRmPhysArg {
    pub block_id: u64,
    pub offset: u64,
    pub length: u64,
    pub page_size: u64,
    pub phys_addr_count: u64,
    pub first_phys_addr: u64,
    pub last_phys_addr: u64,
    pub flags: u64,
    pub _reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisProbeRmCopyArg {
    pub block_id: u64,
    pub offset: u64,
    pub length: u64,
    pub pattern_seed: u64,
    pub page_size: u64,
    pub phys_addr_count: u64,
    pub first_phys_addr: u64,
    pub last_phys_addr: u64,
    pub flags: u64,
    pub bytes_checked: u64,
    pub first_mismatch_offset: u64,
    pub expected_byte: u64,
    pub actual_byte: u64,
    pub _reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisRmCopyArg {
    pub block_id: u64,
    pub offset: u64,
    pub length: u64,
    pub user_cpu_addr: u64,
    pub direction: u32,
    pub _pad: u32,
    // Optional explicit RM tuple. When zero, polaris.ko copies against the
    // block's currently registered RM backing. A daemon may fill these fields
    // to copy into freshly allocated RELOAD backing before COMPLETE_OPERATION
    // makes that backing resident.
    pub rm_control_fd: i32,
    pub rm_h_client: u32,
    pub rm_h_memory: u32,
    pub _pad2: u32,
    pub page_size: u64,
    pub phys_addr_count: u64,
    pub first_phys_addr: u64,
    pub last_phys_addr: u64,
    pub flags: u64,
    pub bytes_copied: u64,
    pub _reserved: [u64; 2],
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
    // _reserved[0]: source block_id for COW_BREAK when
    // POLARIS_DECISION_FLAG_SOURCE_BLOCK_ID_VALID is set in _reserved[1].
    // This keeps the ioctl struct size stable while letting RM-backed COW use
    // block identity instead of a legacy CUDA physical handle.
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
    /// Optional RM control fd for UVM-bridge-mapable block backing. Leave all
    /// RM backing fields zero when the completed operation produced only a
    /// legacy CUDA VMM handle.
    pub rm_control_fd: i32,
    pub output_handle: u64,
    pub output_cpu_addr: u64,
    pub rm_h_client: u32,
    pub rm_h_memory: u32,
    pub rm_backing_length: u64,
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
#[derive(Clone, Copy, Default)]
pub struct PolarisKvActiveWindowArg {
    pub session_id: u64,
    pub phase: u32,
    pub flags: u32,
    pub read_start_token: u64,
    pub read_token_count: u64,
    pub write_start_token: u64,
    pub write_token_count: u64,
    pub epoch: u64,
    pub _reserved: [u64; 4],
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
