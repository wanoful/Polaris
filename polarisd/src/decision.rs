use crate::cuda_vmm;
use crate::gpu::GpuState;
use crate::offload::CpuPool;
use libc::c_int;
use libpolaris::types::*;
use std::time::Instant;

pub struct ExecutionResult {
    pub result: i32,
    pub output_handle: u64,
    pub output_cpu_addr: u64,
}

fn decision_name(op: u32) -> &'static str {
    match op {
        x if x == PolarisDecisionOp::Alloc as u32 => "ALLOC",
        x if x == PolarisDecisionOp::Free as u32 => "FREE",
        x if x == PolarisDecisionOp::MapExisting as u32 => "MAP_EXISTING",
        x if x == PolarisDecisionOp::Unmap as u32 => "UNMAP",
        x if x == PolarisDecisionOp::Offload as u32 => "OFFLOAD",
        x if x == PolarisDecisionOp::Reload as u32 => "RELOAD",
        x if x == PolarisDecisionOp::CowBreak as u32 => "COW_BREAK",
        _ => "UNKNOWN",
    }
}

/// Execute a single kernel decision against the real GPU.
pub fn execute(
    _fd: c_int,
    dec: &PolarisDecision,
    gpu: &mut GpuState,
    cpu_pool: &mut CpuPool,
    test_err: i32,
) -> ExecutionResult {
    if test_err != 0 {
        let err = match dec.block_id % 5 {
            0 => -(libc::ENOMEM as i32),
            1 => -(libc::ENODEV as i32),
            2 => -(libc::EINVAL as i32),
            3 => -(libc::EFAULT as i32),
            _ => -(libc::EIO as i32),
        };
        eprintln!(
            "polarisd: [TEST] simulating error {err} for block {}",
            dec.block_id
        );
        return ExecutionResult {
            result: err,
            output_handle: 0,
            output_cpu_addr: 0,
        };
    }

    eprintln!(
        "polarisd: executing decision {} op={} fault_id={} generation={} timeout_ms={} block_id={} session_id={} size={} src_vaddr={:#x} dst_vaddr={:#x}",
        dec.decision_id,
        decision_name(dec.op),
        dec.fault_id,
        dec.generation,
        dec.timeout_ms,
        dec.block_id,
        dec.session_id,
        dec.size_bytes,
        dec.src_vaddr,
        dec.dst_vaddr
    );

    if let Err(e) = cuda_vmm::push_context(gpu.context) {
        eprintln!("polarisd: push_context failed: {e}");
        return ExecutionResult {
            result: -(libc::ENODEV as i32),
            output_handle: 0,
            output_cpu_addr: 0,
        };
    }

    let started = Instant::now();
    let outcome = dispatch(dec, gpu, cpu_pool);
    let elapsed_ms = started.elapsed().as_millis() as u64;
    if dec.timeout_ms != 0 && elapsed_ms > dec.timeout_ms as u64 {
        eprintln!(
            "polarisd: decision {} exceeded timeout budget ({} ms > {} ms)",
            dec.decision_id, elapsed_ms, dec.timeout_ms
        );
    }

    let _ = cuda_vmm::pop_context();

    outcome
}

fn dispatch(dec: &PolarisDecision, gpu: &mut GpuState, cpu_pool: &mut CpuPool) -> ExecutionResult {
    let mut result: i32 = 0;
    let mut output_handle: u64 = 0;
    let output_cpu_addr: u64 = dec.cpu_addr;

    match dec.op {
        // ─── ALLOC: cuMemCreate + cuMemMap + cuMemSetAccess ──────────
        x if x == PolarisDecisionOp::Alloc as u32 => {
            let size = snap_up(dec.size_bytes, gpu.granule);
            let dst = match preferred_dst_vaddr(dec, gpu, size) {
                Some(dst) => dst,
                None => {
                    eprintln!("polarisd: VA pool exhausted for ALLOC block {}", dec.block_id);
                    return ExecutionResult {
                        result: -(libc::ENOMEM as i32),
                        output_handle: 0,
                        output_cpu_addr: 0,
                    };
                }
            };
            let vaddr = dst.vaddr;

            let phys = match cuda_vmm::create_physical(size, gpu.device_ordinal) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("polarisd: cuMemCreate failed for block {}: {e}", dec.block_id);
                    if dst.release_to_pool {
                        gpu.vas.free(vaddr, size);
                    }
                    return ExecutionResult {
                        result: -(libc::ENOMEM as i32),
                        output_handle: 0,
                        output_cpu_addr: 0,
                    };
                }
            };

            if let Err(e) = cuda_vmm::map_memory(vaddr, phys, size) {
                eprintln!("polarisd: cuMemMap failed for block {}: {e}", dec.block_id);
                if let Err(re) = cuda_vmm::release_physical(phys) {
                    eprintln!("polarisd: cuMemRelease cleanup failed for block {}: {re}", dec.block_id);
                }
                if dst.release_to_pool {
                    gpu.vas.free(vaddr, size);
                }
                return ExecutionResult {
                    result: -(libc::EINVAL as i32),
                    output_handle: 0,
                    output_cpu_addr: 0,
                };
            }

            if let Err(e) = cuda_vmm::set_access(vaddr, size, gpu.device_ordinal) {
                eprintln!("polarisd: cuMemSetAccess failed for block {}: {e}", dec.block_id);
                let _ = cuda_vmm::unmap_memory(vaddr, size);
                if let Err(re) = cuda_vmm::release_physical(phys) {
                    eprintln!("polarisd: cuMemRelease cleanup failed for block {}: {re}", dec.block_id);
                }
                if dst.release_to_pool {
                    gpu.vas.free(vaddr, size);
                }
                return ExecutionResult {
                    result: -(libc::EINVAL as i32),
                    output_handle: 0,
                    output_cpu_addr: 0,
                };
            }

            gpu.track_handle(dec.block_id, phys);
            gpu.track_va(dec.block_id, vaddr, size, dst.release_to_pool);
            gpu.used_bytes += size;
            output_handle = phys;

            eprintln!(
                "polarisd: ALLOC block {} -> phys={phys:#x} va={vaddr:#x} size={size}",
                dec.block_id
            );
        }

        // ─── FREE: cuMemUnmap + cuMemRelease + CPU pool cleanup ───────
        x if x == PolarisDecisionOp::Free as u32 => {
            let phys = if dec.src_handle != 0 {
                dec.src_handle
            } else {
                gpu.get_handle(dec.block_id).unwrap_or(0)
            };
            let vaddr = match dec.src_vaddr {
                0 => gpu.get_va_alloc(dec.block_id).map(|v| v.vaddr).unwrap_or(0),
                v => v,
            };

            if let Some(va) = gpu.get_va_alloc(dec.block_id) {
                let size = va.size;
                let _ = cuda_vmm::unmap_memory(vaddr, size);
                if phys != 0 {
                    if let Err(e) = cuda_vmm::release_physical(phys) {
                        eprintln!("polarisd: FREE cuMemRelease failed for block {}: {e}", dec.block_id);
                    }
                }
                gpu.used_bytes = gpu.used_bytes.saturating_sub(size);
                eprintln!(
                    "polarisd: FREE block {} -> phys={phys:#x} va={vaddr:#x}",
                    dec.block_id
                );
            } else if phys != 0 {
                if let Err(e) = cuda_vmm::release_physical(phys) {
                    eprintln!("polarisd: FREE cuMemRelease failed for block {}: {e}", dec.block_id);
                }
                eprintln!("polarisd: FREE block {} -> phys={phys:#x} (no VA track)", dec.block_id);
            }

            // Free the CPU buffer if this block was offloaded.
            if let Some(cpu_addr) = cpu_pool.untrack(dec.block_id) {
                let sz = dec.size_bytes.max(1);
                cpu_pool.free(cpu_addr, sz);
                eprintln!("polarisd: FREE block {} -> released CPU buffer {cpu_addr:#x}", dec.block_id);
            }

            gpu.remove_block(dec.block_id);
        }

        // ─── MAP_EXISTING: cuMemMap + cuMemSetAccess (COW sharing) ───
        x if x == PolarisDecisionOp::MapExisting as u32 => {
            let size = snap_up(dec.size_bytes, gpu.granule);
            if dec.src_handle == 0 || dec.dst_vaddr == 0 {
                result = -(libc::EINVAL as i32);
            } else if let Err(e) = cuda_vmm::map_memory(dec.dst_vaddr, dec.src_handle, size) {
                eprintln!("polarisd: MAP failed: {e}");
                result = -(libc::EINVAL as i32);
            } else if let Err(e) = cuda_vmm::set_access(dec.dst_vaddr, size, gpu.device_ordinal) {
                eprintln!("polarisd: MAP set_access failed: {e}");
                let _ = cuda_vmm::unmap_memory(dec.dst_vaddr, size);
                result = -(libc::EINVAL as i32);
            } else {
                gpu.track_va(dec.block_id, dec.dst_vaddr, size, false);
                eprintln!("polarisd: MAP_EXISTING block {} -> phys={:#x} va={:#x}", dec.block_id, dec.src_handle, dec.dst_vaddr);
            }
        }

        // ─── UNMAP: cuMemUnmap ─────────────────────────────────────
        x if x == PolarisDecisionOp::Unmap as u32 => {
            let vaddr = if dec.dst_vaddr != 0 {
                dec.dst_vaddr
            } else {
                gpu.get_va_alloc(dec.block_id)
                    .map(|v| v.vaddr)
                    .unwrap_or(0)
            };
            let size = dec
                .size_bytes
                .max(gpu.get_va_alloc(dec.block_id).map(|v| v.size).unwrap_or(0));
            if vaddr != 0 {
                if let Err(e) = cuda_vmm::unmap_memory(vaddr, size) {
                    eprintln!("polarisd: UNMAP failed: {e}");
                    result = -(libc::EINVAL as i32);
                } else {
                    eprintln!("polarisd: UNMAP block {} -> va={vaddr:#x}", dec.block_id);
                }
            }
        }

        // ─── OFFLOAD: copy GPU VA→CPU, unmap VA, report CPU buffer addr ──
        x if x == PolarisDecisionOp::Offload as u32 => {
            let (res, handle, cpu_addr) = crate::offload::execute_offload(dec, gpu, cpu_pool);
            result = res;
            output_handle = handle;
            return ExecutionResult {
                result,
                output_handle,
                output_cpu_addr: cpu_addr,
            };
        }

        // ─── RELOAD: alloc + map + copy CPU→GPU VA, report new phys handle ──
        x if x == PolarisDecisionOp::Reload as u32 => {
            let (res, handle, cpu_addr) = crate::offload::execute_reload(dec, gpu, cpu_pool);
            result = res;
            output_handle = handle;
            return ExecutionResult {
                result,
                output_handle,
                output_cpu_addr: cpu_addr,
            };
        }

        // ─── COW_BREAK: alloc new + copy old VA→new VA (GPU DtoD) ───
        x if x == PolarisDecisionOp::CowBreak as u32 => {
            let size = snap_up(dec.size_bytes, gpu.granule);
            let sz_usize = size as usize;

            let src_vaddr = match dec.src_vaddr {
                0 => {
                    let src_phys = dec.src_handle;
                    let src_block_id = match gpu.find_block_by_phys(src_phys) {
                        Some(bid) => bid,
                        None => {
                            eprintln!(
                                "polarisd: COW_BREAK block {} — no source VA or source handle match for {src_phys:#x}",
                                dec.block_id
                            );
                            return ExecutionResult {
                                result: -(libc::EINVAL as i32),
                                output_handle: 0,
                                output_cpu_addr: 0,
                            };
                        }
                    };
                    match gpu.get_va_alloc(src_block_id) {
                        Some(va) => va.vaddr,
                        None => {
                            eprintln!(
                                "polarisd: COW_BREAK block {} — source block {src_block_id} has no VA mapping",
                                dec.block_id
                            );
                            return ExecutionResult {
                                result: -(libc::EINVAL as i32),
                                output_handle: 0,
                                output_cpu_addr: 0,
                            };
                        }
                    }
                }
                v => v,
            };

            let dst = match preferred_dst_vaddr(dec, gpu, size) {
                Some(dst) => dst,
                None => {
                    eprintln!(
                        "polarisd: COW_BREAK block {} VA pool exhausted",
                        dec.block_id
                    );
                    return ExecutionResult {
                        result: -(libc::ENOMEM as i32),
                        output_handle: 0,
                        output_cpu_addr: 0,
                    };
                }
            };
            let dst_vaddr = dst.vaddr;

            if let Some(existing) = gpu.get_va_alloc(dec.block_id) {
                if existing.vaddr != dst_vaddr {
                    gpu.vas.free(existing.vaddr, existing.size);
                }
            }

            // Create new physical memory handle.
            let new_phys = match cuda_vmm::create_physical(size, gpu.device_ordinal) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("polarisd: COW_BREAK cuMemCreate failed for block {}: {e}", dec.block_id);
                    if dst.release_to_pool {
                        gpu.vas.free(dst_vaddr, size);
                    }
                    return ExecutionResult {
                        result: -(libc::ENOMEM as i32),
                        output_handle: 0,
                        output_cpu_addr: 0,
                    };
                }
            };

            // Map new physical into the destination VA.
            if let Err(e) = cuda_vmm::map_memory(dst_vaddr, new_phys, size) {
                eprintln!("polarisd: COW_BREAK cuMemMap failed for block {}: {e}", dec.block_id);
                if let Err(re) = cuda_vmm::release_physical(new_phys) {
                    eprintln!("polarisd: COW_BREAK cuMemRelease cleanup failed: {re}");
                }
                if dst.release_to_pool {
                    gpu.vas.free(dst_vaddr, size);
                }
                return ExecutionResult {
                    result: -(libc::EINVAL as i32),
                    output_handle: 0,
                    output_cpu_addr: 0,
                };
            }

            // Set access for the destination mapping.
            if let Err(e) = cuda_vmm::set_access(dst_vaddr, size, gpu.device_ordinal) {
                eprintln!("polarisd: COW_BREAK cuMemSetAccess failed for block {}: {e}", dec.block_id);
                let _ = cuda_vmm::unmap_memory(dst_vaddr, size);
                if let Err(re) = cuda_vmm::release_physical(new_phys) {
                    eprintln!("polarisd: COW_BREAK cuMemRelease cleanup failed: {re}");
                }
                if dst.release_to_pool {
                    gpu.vas.free(dst_vaddr, size);
                }
                return ExecutionResult {
                    result: -(libc::EINVAL as i32),
                    output_handle: 0,
                    output_cpu_addr: 0,
                };
            }

            // GPU-to-GPU copy: old (source) VA → new (destination) VA.
            if let Err(e) = cuda_vmm::copy_device_to_device(dst_vaddr, src_vaddr, sz_usize) {
                eprintln!("polarisd: COW_BREAK cuMemcpyDtoD failed for block {}: {e}", dec.block_id);
                let _ = cuda_vmm::unmap_memory(dst_vaddr, size);
                if let Err(re) = cuda_vmm::release_physical(new_phys) {
                    eprintln!("polarisd: COW_BREAK cuMemRelease cleanup failed: {re}");
                }
                if dst.release_to_pool {
                    gpu.vas.free(dst_vaddr, size);
                }
                return ExecutionResult {
                    result: -(libc::EFAULT as i32),
                    output_handle: 0,
                    output_cpu_addr: 0,
                };
            }

            // Success: track the new block.
            gpu.track_handle(dec.block_id, new_phys);
            gpu.track_va(dec.block_id, dst_vaddr, size, dst.release_to_pool);
            gpu.used_bytes += size;
            output_handle = new_phys;

            eprintln!(
                "polarisd: COW_BREAK block {} complete: src_va={src_vaddr:#x} → new phys={new_phys:#x} va={dst_vaddr:#x} size={size}",
                dec.block_id
            );
        }

        _ => {
            eprintln!(
                "polarisd: unknown decision op {} for block {}",
                dec.op, dec.block_id
            );
            result = -(libc::EINVAL as i32);
        }
    }

    ExecutionResult {
        result,
        output_handle,
        output_cpu_addr,
    }
}

fn snap_up(val: u64, align: u64) -> u64 {
    if align == 0 {
        return val;
    }
    (val + align - 1) & !(align - 1)
}

struct DstVa {
    vaddr: u64,
    release_to_pool: bool,
}

fn preferred_dst_vaddr(dec: &PolarisDecision, gpu: &mut GpuState, size: u64) -> Option<DstVa> {
    if dec.dst_vaddr != 0 {
        return Some(DstVa {
            vaddr: dec.dst_vaddr,
            release_to_pool: false,
        });
    }
    if let Some(existing) = gpu.get_va_alloc(dec.block_id) {
        return Some(DstVa {
            vaddr: existing.vaddr,
            release_to_pool: false,
        });
    }
    gpu.vas.allocate(size, gpu.granule).map(|vaddr| DstVa {
        vaddr,
        release_to_pool: true,
    })
}
