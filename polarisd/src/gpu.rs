use std::collections::HashMap;

pub type CudaContext = *mut cudarc::driver::sys::CUctx_st;

/// Pre-reserved GPU virtual address pool managed by a simple bump allocator.
pub struct GpuVaPool {
    pub base: u64,
    pub size: u64,
    pub cursor: u64,
}

/// Tracking for a single VA sub-allocation.
pub struct VaAlloc {
    pub vaddr: u64,
    pub size: u64,
}

/// Per-GPU state tracked by the daemon.
#[allow(dead_code)]
pub struct GpuState {
    pub gpu_id: u32,
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub budget_bytes: u64,
    pub cpu_pool_bytes: u64,
    pub cpu_pool_used_bytes: u64,
    pub device_ordinal: i32,
    pub context: CudaContext,
    pub vas: GpuVaPool,
    pub granule: u64,
    /// block_id -> physical handle
    pub phys_handles: HashMap<u64, u64>,
    /// block_id -> VA sub-allocation
    pub va_allocs: HashMap<u64, VaAlloc>,
}

impl GpuVaPool {
    pub fn new(base: u64, size: u64) -> Self {
        Self {
            base,
            size,
            cursor: 0,
        }
    }

    /// Allocate `size` bytes from the pool, aligned to `alignment`.
    /// Returns absolute GPU virtual address.
    pub fn allocate(&mut self, size: u64, alignment: u64) -> Option<u64> {
        let offset = (self.cursor + alignment - 1) & !(alignment - 1);
        if offset + size > self.size {
            return None;
        }
        let vaddr = self.base + offset;
        self.cursor = offset + size;
        Some(vaddr)
    }
}

impl GpuState {
    pub fn new(
        gpu_id: u32,
        device_ordinal: i32,
        context: CudaContext,
        vas_base: u64,
        vas_size: u64,
        granule: u64,
        total_bytes: u64,
        budget_bytes: u64,
        cpu_pool_bytes: u64,
    ) -> Self {
        Self {
            gpu_id,
            total_bytes,
            used_bytes: 0,
            budget_bytes,
            cpu_pool_bytes,
            cpu_pool_used_bytes: 0,
            device_ordinal,
            context,
            vas: GpuVaPool::new(vas_base, vas_size),
            granule,
            phys_handles: HashMap::new(),
            va_allocs: HashMap::new(),
        }
    }

    pub fn track_handle(&mut self, block_id: u64, phys_handle: u64) {
        self.phys_handles.insert(block_id, phys_handle);
    }

    pub fn track_va(&mut self, block_id: u64, vaddr: u64, size: u64) {
        self.va_allocs.insert(block_id, VaAlloc { vaddr, size });
    }

    pub fn get_handle(&self, block_id: u64) -> Option<u64> {
        self.phys_handles.get(&block_id).copied()
    }

    pub fn get_va_alloc(&self, block_id: u64) -> Option<&VaAlloc> {
        self.va_allocs.get(&block_id)
    }

    pub fn remove_block(&mut self, block_id: u64) {
        self.phys_handles.remove(&block_id);
        self.va_allocs.remove(&block_id);
    }
}
