use std::collections::HashMap;

pub type CudaContext = *mut cudarc::driver::sys::CUctx_st;

/// Pre-reserved GPU virtual address pool with a sorted free-list.
///
/// Gaps between allocations are minimal because CUDA VMM enforces
/// `granule`-byte alignment — every allocation is naturally a multiple
/// of the granularity, so the free-list stays compact.
pub struct GpuVaPool {
    pub base: u64,
    pub size: u64,
    /// Sorted free ranges (start, size).  Maintained without overlap.
    free_ranges: Vec<(u64, u64)>,
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
    pub cpu_pool_base: u64,
    pub device_ordinal: i32,
    pub context: CudaContext,
    pub vas: GpuVaPool,
    pub granule: u64,
    pub phys_handles: HashMap<u64, u64>,
    pub va_allocs: HashMap<u64, VaAlloc>,
}

impl GpuVaPool {
    pub fn new(base: u64, size: u64) -> Self {
        Self {
            base,
            size,
            free_ranges: vec![(base, size)],
        }
    }

    /// Allocate `size` bytes from the pool, aligned to `alignment`.
    /// Returns absolute GPU virtual address.  First-fit strategy.
    pub fn allocate(&mut self, size: u64, alignment: u64) -> Option<u64> {
        let align = alignment.max(1);
        for idx in 0..self.free_ranges.len() {
            let (range_start, range_size) = self.free_ranges[idx];
            // Align the allocation within the range.
            let aligned_start = (range_start + align - 1) & !(align - 1);
            if aligned_start < range_start {
                continue; // overflow / should not happen
            }
            let offset_in_range = aligned_start - range_start;
            if offset_in_range + size > range_size {
                continue; // doesn't fit in this range
            }
            // Carve out the allocated chunk.
            if offset_in_range == 0 {
                // Allocation at the start of the range.
                if size == range_size {
                    self.free_ranges.remove(idx);
                } else {
                    self.free_ranges[idx].0 += size;
                    self.free_ranges[idx].1 -= size;
                }
            } else if offset_in_range + size == range_size {
                // Allocation at the end of the range.
                self.free_ranges[idx].1 = offset_in_range;
            } else {
                // Allocation in the middle — split into two ranges.
                self.free_ranges[idx].1 = offset_in_range;
                self.free_ranges.insert(
                    idx + 1,
                    (aligned_start + size, range_size - offset_in_range - size),
                );
            }
            // Safety: every free range is a sub-range of [base, base+size),
            // so the allocated VA must be within bounds.
            debug_assert!(
                aligned_start >= self.base && aligned_start + size <= self.base + self.size,
                "VA allocation {:#x}+{:#x} exceeds reserved range [{:#x}, {:#x})",
                aligned_start, size, self.base, self.base + self.size
            );
            return Some(aligned_start);
        }
        None
    }

    /// Return a previously-allocated VA range back to the pool.
    pub fn free(&mut self, vaddr: u64, size: u64) {
        if size == 0 {
            return;
        }
        let end = vaddr + size;

        // Find insertion index (first range whose start >= vaddr).
        let mut ins = 0;
        while ins < self.free_ranges.len() && self.free_ranges[ins].0 < vaddr {
            ins += 1;
        }

        // Merge with left neighbour if adjacent.
        let mut merged = false;
        if ins > 0 {
            let left = &mut self.free_ranges[ins - 1];
            if left.0 + left.1 >= vaddr {
                // Overlap or adjacent — merge.
                assert!(left.0 + left.1 <= vaddr, "VA free-list corruption: left neighbour overlaps freed range");
                left.1 = end - left.0;
                merged = true;
            }
        }

        // Merge with right neighbour(s) if adjacent or overlapping.
        if ins < self.free_ranges.len() {
            if merged {
                // Left was extended — check if now adjacent to right.
                let right_end = self.free_ranges[ins].0 + self.free_ranges[ins].1;
                if self.free_ranges[ins].0 <= self.free_ranges[ins - 1].0 + self.free_ranges[ins - 1].1 {
                    // Left and right now overlap or adjacent — merge them.
                    self.free_ranges[ins - 1].1 = right_end - self.free_ranges[ins - 1].0;
                    self.free_ranges.remove(ins);
                }
            } else {
                let right = &mut self.free_ranges[ins];
                if end >= right.0 {
                    assert!(end <= right.0, "VA free-list corruption: right neighbour overlaps freed range");
                    // Adjacent or overlapping — extend right's start.
                    right.1 = right.0 + right.1 - vaddr;
                    right.0 = vaddr;
                    merged = true;
                }
            }
        }

        if !merged {
            self.free_ranges.insert(ins, (vaddr, size));
        }
    }

    /// Human-readable usage for diagnostics.
    #[allow(dead_code)]
    pub fn free_bytes(&self) -> u64 {
        self.free_ranges.iter().map(|(_, s)| s).sum()
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
        cpu_pool_base: u64,
    ) -> Self {
        Self {
            gpu_id,
            total_bytes,
            used_bytes: 0,
            budget_bytes,
            cpu_pool_bytes,
            cpu_pool_used_bytes: 0,
            cpu_pool_base,
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

    /// Find a block_id by its GPU VA.
    pub fn find_block_by_vaddr(&self, vaddr: u64) -> Option<u64> {
        self.va_allocs
            .iter()
            .find(|(_, va)| va.vaddr == vaddr)
            .map(|(&bid, _)| bid)
    }

    /// Find a block_id by its physical handle (reverse lookup).
    /// This is only a compatibility fallback when the kernel cannot provide
    /// the source GPU VA directly.
    pub fn find_block_by_phys(&self, phys_handle: u64) -> Option<u64> {
        self.phys_handles
            .iter()
            .find(|(_, &ph)| ph == phys_handle)
            .map(|(&bid, _)| bid)
    }

    pub fn contains_va(&self, vaddr: u64, size: u64) -> bool {
        size > 0 && vaddr >= self.vas.base && vaddr.saturating_add(size) <= self.vas.base + self.vas.size
    }

    /// Remove block tracking and return its VA to the pool.
    pub fn remove_block(&mut self, block_id: u64) {
        self.phys_handles.remove(&block_id);
        if let Some(va) = self.va_allocs.remove(&block_id) {
            self.vas.free(va.vaddr, va.size);
        }
    }

    pub fn clear_handle(&mut self, block_id: u64) {
        self.phys_handles.remove(&block_id);
    }
}
