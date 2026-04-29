//! POLARIS ioctl argument types (userspace side).
//!
//! These must match the kernel-side definitions in kernel/polaris_types.rs exactly.
//! All structs are #[repr(C)] for ABI compatibility with the kernel.

/// Magic number 'P' = 0x50
pub const POLARIS_IOCTL_MAGIC: u8 = 0x50;

// ─── Block state enum ──────────────────────────────────────────────────────

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

// ─── Decision opcodes ──────────────────────────────────────────────────────

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

// ─── Constants ─────────────────────────────────────────────────────────────

pub const POLARIS_MAX_DECISIONS_PER_POLL: usize = 16;
pub const POLARIS_BLOCK_FLAG_SHARED: u32 = 1 << 0;
pub const POLARIS_GROW_FLAG_OVERWRITE: u32 = 1 << 0;
pub const POLARIS_PHASE_PREFILL: u32 = 1;
pub const POLARIS_PHASE_DECODE: u32 = 2;

// ─── IOCTL Argument Structs ────────────────────────────────────────────────

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
pub struct PolarisSessionCreateArg {
    pub session_id: u64,
    pub home_gpu: u32,
    pub beam_width: u32,
    pub gpu_vas_bytes: u64,
    pub _reserved: [u64; 4],
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

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisBlockFreeArg {
    pub session_id: u64,
    pub token_start: u32,
    pub token_count: u32,
    pub _reserved: [u64; 4],
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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolarisGetDecisionArg {
    pub count: u32,
    pub _reserved: u32,
    pub decisions: [PolarisDecision; POLARIS_MAX_DECISIONS_PER_POLL],
}

impl Default for PolarisGetDecisionArg {
    fn default() -> Self {
        Self {
            count: 0,
            _reserved: 0,
            decisions: [PolarisDecision::default(); POLARIS_MAX_DECISIONS_PER_POLL],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PolarisCompleteOperationArg {
    pub decision_id: u64,
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
    pub total_gpu_bytes: u64,
    pub used_gpu_bytes: u64,
    pub cpu_pool_total: u64,
    pub cpu_pool_used: u64,
    pub _reserved: [u64; 4],
}
