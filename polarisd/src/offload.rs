use crate::cuda_vmm;
use crate::gpu;
use libpolaris::types::*;
use std::collections::HashMap;

/// Sub-allocation from the pre-allocated CPU pinned memory pool.
pub struct CpuPool {
    pub base: u64,
    pub total: u64,
    used: u64,
    free_ranges: Vec<(u64, u64)>,
    allocations: HashMap<u64, u64>, // block_id → offset within pool
}

impl CpuPool {
    /// Create a CPU pool allocator over the pre-allocated pinned range.
    /// `total` is the usable size (may be less than `allocated` due to alignment).
    pub fn new(base: u64, total: u64) -> Self {
        Self {
            base,
            total,
            used: 0,
            free_ranges: vec![(base, total)],
            allocations: HashMap::new(),
        }
    }

    /// Allocate `size` bytes (page-aligned). Returns the absolute CPU address,
    /// or None if the pool is exhausted.
    pub fn allocate(&mut self, size: u64) -> Option<u64> {
        // Align to 4 KiB (page) for DMA efficiency.
        let align = 4096u64;
        for idx in 0..self.free_ranges.len() {
            let (range_start, range_size) = self.free_ranges[idx];
            let aligned_start = (range_start + align - 1) & !(align - 1);
            if aligned_start < range_start {
                continue;
            }
            let offset_in_range = aligned_start - range_start;
            if offset_in_range + size > range_size {
                continue;
            }
            // Carve out the chunk (same logic as GpuVaPool).
            if offset_in_range == 0 {
                if size == range_size {
                    self.free_ranges.remove(idx);
                } else {
                    self.free_ranges[idx].0 += size;
                    self.free_ranges[idx].1 -= size;
                }
            } else if offset_in_range + size == range_size {
                self.free_ranges[idx].1 = offset_in_range;
            } else {
                self.free_ranges[idx].1 = offset_in_range;
                self.free_ranges
                    .insert(idx + 1, (aligned_start + size, range_size - offset_in_range - size));
            }
            self.used += size;
            return Some(aligned_start);
        }
        None
    }

    /// Free a previously-allocated range (absolute CPU address).
    pub fn free(&mut self, addr: u64, size: u64) {
        if size == 0 {
            return;
        }
        self.used = self.used.saturating_sub(size);
        let end = addr + size;

        // Find insertion point (sorted by start address).
        let mut ins = 0;
        while ins < self.free_ranges.len() && self.free_ranges[ins].0 < addr {
            ins += 1;
        }

        let mut merged = false;
        if ins > 0 {
            let left = &mut self.free_ranges[ins - 1];
            if left.0 + left.1 >= addr {
                debug_assert!(left.0 + left.1 <= addr, "CPU pool corruption: left neighbour overlaps freed range");
                left.1 = end - left.0;
                merged = true;
            }
        }

        if ins < self.free_ranges.len() {
            if merged {
                let right_end = self.free_ranges[ins].0 + self.free_ranges[ins].1;
                if self.free_ranges[ins].0 <= self.free_ranges[ins - 1].0 + self.free_ranges[ins - 1].1 {
                    self.free_ranges[ins - 1].1 = right_end - self.free_ranges[ins - 1].0;
                    self.free_ranges.remove(ins);
                }
            } else {
                let right = &mut self.free_ranges[ins];
                if end >= right.0 {
                    debug_assert!(end <= right.0, "CPU pool corruption: right neighbour overlaps freed range");
                    right.1 = right.0 + right.1 - addr;
                    right.0 = addr;
                    merged = true;
                }
            }
        }

        if !merged {
            self.free_ranges.insert(ins, (addr, size));
        }
    }

    pub fn used_bytes(&self) -> u64 {
        self.used
    }

    pub fn free_bytes(&self) -> u64 {
        self.free_ranges.iter().map(|(_, s)| s).sum()
    }

    pub fn track(&mut self, block_id: u64, addr: u64) {
        self.allocations.insert(block_id, addr);
    }

    pub fn untrack(&mut self, block_id: u64) -> Option<u64> {
        self.allocations.remove(&block_id)
    }

    pub fn get(&self, block_id: u64) -> Option<u64> {
        self.allocations.get(&block_id).copied()
    }
}

/// Execute the full OFFLOAD operation:
///   1. Copy data GPU VA → CPU buffer via cudaMemcpyDtoH
///   2. Unmap the GPU VA (cuMemUnmap)
///   3. Release the physical handle (cuMemRelease)
///   4. Keep the VA reservation for the eventual reload
///   5. Report the CPU buffer address
pub fn execute_offload(
    dec: &PolarisDecision,
    gpu: &mut gpu::GpuState,
    cpu_pool: &mut CpuPool,
) -> (i32, u64, u64) {
    let va_info = gpu.get_va_alloc(dec.block_id).map(|v| (v.vaddr, v.size));
    let src_vaddr = if dec.src_vaddr != 0 {
        dec.src_vaddr
    } else {
        va_info.map(|v| v.0).unwrap_or(0)
    };
    let size = snap_up(dec.size_bytes, gpu.granule).max(va_info.map(|v| v.1).unwrap_or(0));
    if src_vaddr == 0 || size == 0 {
        eprintln!(
            "polarisd: OFFLOAD block {} has no source GPU VA",
            dec.block_id
        );
        return (-(libc::EINVAL as i32), 0, 0);
    }
    if va_info.is_none() && dec.src_vaddr != 0 {
        if !gpu.contains_va(dec.src_vaddr, size) {
            eprintln!(
                "polarisd: OFFLOAD block {} source VA {:#x} is outside the reserved range",
                dec.block_id,
                dec.src_vaddr
            );
            return (-(libc::EINVAL as i32), 0, 0);
        }
    }

    let sz_usize = size as usize;

    // Allocate CPU buffer from the pinned pool.
    let cpu_addr = match cpu_pool.allocate(size) {
        Some(addr) => {
            eprintln!(
                "polarisd: OFFLOAD block {} -> cpu_buf={addr:#x} size={size}",
                dec.block_id
            );
            addr
        }
        None => {
            eprintln!(
                "polarisd: OFFLOAD block {} CPU pool exhausted (used={} total={})",
                dec.block_id,
                cpu_pool.used_bytes(),
                cpu_pool.total
            );
            return (-(libc::ENOMEM as i32), 0, 0);
        }
    };

    // Copy GPU → CPU (the VA is still mapped and accessible).
    if let Err(e) = cuda_vmm::copy_device_to_host(cpu_addr, src_vaddr, sz_usize) {
        eprintln!("polarisd: OFFLOAD cudaMemcpyDtoH failed for block {}: {e}", dec.block_id);
        cpu_pool.free(cpu_addr, size);
        return (-(libc::EFAULT as i32), 0, 0);
    }

    // Unmap the GPU VA — data now lives only in the CPU buffer.
    if let Err(e) = cuda_vmm::unmap_memory(src_vaddr, size) {
        eprintln!("polarisd: OFFLOAD unmap failed for block {}: {e}", dec.block_id);
        cpu_pool.free(cpu_addr, size);
        return (-(libc::EINVAL as i32), 0, 0);
    }

    // Release the physical GPU memory handle. This is what genuinely
    // frees GPU bytes — without it, cuMemCreate for new blocks would
    // fail with OUT_OF_MEMORY even though our accounting says we have room.
    if let Some(old_phys) = gpu.get_handle(dec.block_id) {
        if let Err(e) = cuda_vmm::release_physical(old_phys) {
            eprintln!("polarisd: OFFLOAD cuMemRelease failed for block {}: {e}", dec.block_id);
        }
    }

    // Track the CPU buffer allocation.
    cpu_pool.track(dec.block_id, cpu_addr);

    // Update GPU-side tracking: phys handle released, VA remains reserved.
    gpu.clear_handle(dec.block_id);
    gpu.used_bytes = gpu.used_bytes.saturating_sub(size);
    gpu.track_va(dec.block_id, src_vaddr, size);

    eprintln!(
        "polarisd: OFFLOAD block {} complete: GPU VA {src_vaddr:#x} → CPU buffer {cpu_addr:#x}",
        dec.block_id
    );

    (0, 0, cpu_addr)
}

/// Execute the full RELOAD operation:
///   1. Re-create physical memory
///   2. Re-map the existing GPU VA
///   3. Copy CPU → GPU VA
///   4. Report the new physical handle
///   5. Release the CPU buffer
///
/// Note: the old physical handle was already released during OFFLOAD.
pub fn execute_reload(
    dec: &PolarisDecision,
    gpu: &mut gpu::GpuState,
    cpu_pool: &mut CpuPool,
) -> (i32, u64, u64) {
    let size = snap_up(dec.size_bytes, gpu.granule);
    let vaddr = match if dec.dst_vaddr != 0 {
        Some(dec.dst_vaddr)
    } else {
        gpu.get_va_alloc(dec.block_id).map(|v| v.vaddr)
    } {
        Some(va) => va,
        None => {
            eprintln!("polarisd: RELOAD block {} has no destination GPU VA", dec.block_id);
            return (-(libc::EINVAL as i32), 0, 0);
        }
    };

    // The kernel passes the CPU buffer address in dec.cpu_addr.
    let cpu_addr = dec.cpu_addr;
    if cpu_addr == 0 {
        eprintln!(
            "polarisd: RELOAD block {} has no CPU buffer address",
            dec.block_id
        );
        return (-(libc::EINVAL as i32), 0, 0);
    }

    // Create new physical memory (old was released during offload).
    let new_phys = match cuda_vmm::create_physical(size, gpu.device_ordinal) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("polarisd: RELOAD cuMemCreate failed for block {}: {e}", dec.block_id);
            return (-(libc::ENOMEM as i32), 0, 0);
        }
    };

    // Map the new physical handle.
    if let Err(e) = cuda_vmm::map_memory(vaddr, new_phys, size) {
        eprintln!("polarisd: RELOAD cuMemMap failed for block {}: {e}", dec.block_id);
        if let Err(re) = cuda_vmm::release_physical(new_phys) {
            eprintln!("polarisd: RELOAD cuMemRelease cleanup failed for block {}: {re}", dec.block_id);
        }
        return (-(libc::EINVAL as i32), 0, 0);
    }

    // Set access for this GPU.
    if let Err(e) = cuda_vmm::set_access(vaddr, size, gpu.device_ordinal) {
        eprintln!("polarisd: RELOAD cuMemSetAccess failed for block {}: {e}", dec.block_id);
        let _ = cuda_vmm::unmap_memory(vaddr, size);
        if let Err(re) = cuda_vmm::release_physical(new_phys) {
            eprintln!("polarisd: RELOAD cuMemRelease cleanup failed for block {}: {re}", dec.block_id);
        }
        return (-(libc::EINVAL as i32), 0, 0);
    }

    // Copy CPU → GPU (the new mapping is now accessible).
    if let Err(e) = cuda_vmm::copy_host_to_device(vaddr, cpu_addr, size as usize) {
        eprintln!("polarisd: RELOAD cudaMemcpyHtoD failed for block {}: {e}", dec.block_id);
        let _ = cuda_vmm::unmap_memory(vaddr, size);
        if let Err(re) = cuda_vmm::release_physical(new_phys) {
            eprintln!("polarisd: RELOAD cuMemRelease cleanup failed for block {}: {re}", dec.block_id);
        }
        return (-(libc::EFAULT as i32), 0, 0);
    }

    // Release the CPU buffer — data now back on GPU.
    cpu_pool.untrack(dec.block_id);
    cpu_pool.free(cpu_addr, size);

    // Update daemon tracking.
    gpu.track_handle(dec.block_id, new_phys);
    gpu.track_va(dec.block_id, vaddr, size);
    gpu.used_bytes += size;

    eprintln!(
        "polarisd: RELOAD block {} complete: CPU→GPU copied, new phys={new_phys:#x} va={vaddr:#x}",
        dec.block_id
    );

    (0, new_phys, 0)
}

fn snap_up(val: u64, align: u64) -> u64 {
    if align == 0 {
        return val;
    }
    (val + align - 1) & !(align - 1)
}
