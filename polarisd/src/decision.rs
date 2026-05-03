use crate::cuda_vmm;
use crate::gpu::GpuState;
use crate::offload::CpuPool;
use libc::c_int;
use libpolaris::types::*;

pub struct ExecutionResult {
    pub result: i32,
    pub output_handle: u64,
    pub output_cpu_addr: u64,
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

    let op = match dec.op {
        x if x == PolarisDecisionOp::Alloc as u32 => "ALLOC",
        x if x == PolarisDecisionOp::Free as u32 => "FREE",
        x if x == PolarisDecisionOp::Map as u32 => "MAP",
        x if x == PolarisDecisionOp::Unmap as u32 => "UNMAP",
        x if x == PolarisDecisionOp::Offload as u32 => "OFFLOAD",
        x if x == PolarisDecisionOp::Reload as u32 => "RELOAD",
        x if x == PolarisDecisionOp::CowBreak as u32 => "COW_BREAK",
        _ => "UNKNOWN",
    };

    eprintln!(
        "polarisd: executing decision {} op={op} block_id={} session_id={} size={}",
        dec.decision_id, dec.block_id, dec.session_id, dec.size_bytes
    );

    if let Err(e) = cuda_vmm::push_context(gpu.context) {
        eprintln!("polarisd: push_context failed: {e}");
        return ExecutionResult {
            result: -(libc::ENODEV as i32),
            output_handle: 0,
            output_cpu_addr: 0,
        };
    }

    let outcome = dispatch(dec, gpu, cpu_pool);

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

            let vaddr = match gpu.vas.allocate(size, gpu.granule) {
                Some(va) => va,
                None => {
                    eprintln!("polarisd: VA pool exhausted for ALLOC block {}", dec.block_id);
                    return ExecutionResult {
                        result: -(libc::ENOMEM as i32),
                        output_handle: 0,
                        output_cpu_addr: 0,
                    };
                }
            };

            let phys = match cuda_vmm::create_physical(size, gpu.device_ordinal) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("polarisd: cuMemCreate failed for block {}: {e}", dec.block_id);
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
                return ExecutionResult {
                    result: -(libc::EINVAL as i32),
                    output_handle: 0,
                    output_cpu_addr: 0,
                };
            }

            gpu.track_handle(dec.block_id, phys);
            gpu.track_va(dec.block_id, vaddr, size);
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

            if let Some(va) = gpu.get_va_alloc(dec.block_id) {
                let vaddr = va.vaddr;
                let size = va.size;
                let _ = cuda_vmm::unmap_memory(vaddr, size);
                if let Err(e) = cuda_vmm::release_physical(phys) {
                    eprintln!("polarisd: FREE cuMemRelease failed for block {}: {e}", dec.block_id);
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

        // ─── MAP: cuMemMap + cuMemSetAccess (COW sharing) ──────────
        x if x == PolarisDecisionOp::Map as u32 => {
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
                gpu.track_va(dec.block_id, dec.dst_vaddr, size);
                eprintln!(
                    "polarisd: MAP block {} -> phys={:#x} va={:#x}",
                    dec.block_id, dec.src_handle, dec.dst_vaddr
                );
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

        // ─── OFFLOAD: copy GPU→CPU, unmap VA, report CPU buffer addr ──
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

        // ─── RELOAD: alloc + map + copy CPU→GPU, report new phys handle ──
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

        // ─── COW_BREAK (Phase 3 stub): alloc new + map ─────────────
        x if x == PolarisDecisionOp::CowBreak as u32 => {
            let size = snap_up(dec.size_bytes, gpu.granule);
            let vaddr = match gpu.vas.allocate(size, gpu.granule) {
                Some(va) => va,
                None => {
                    return ExecutionResult {
                        result: -(libc::ENOMEM as i32),
                        output_handle: 0,
                        output_cpu_addr: 0,
                    };
                }
            };
            match cuda_vmm::create_physical(size, gpu.device_ordinal) {
                Ok(h) => {
                    let _ = cuda_vmm::map_memory(vaddr, h, size);
                    let _ = cuda_vmm::set_access(vaddr, size, gpu.device_ordinal);
                    gpu.track_handle(dec.block_id, h);
                    gpu.track_va(dec.block_id, vaddr, size);
                    gpu.used_bytes += size;
                    output_handle = h;
                    eprintln!(
                        "polarisd: COW_BREAK block {} (stub: alloc+map, no copy)",
                        dec.block_id
                    );
                }
                Err(e) => {
                    eprintln!("polarisd: COW_BREAK failed: {e}");
                    result = -(libc::ENOMEM as i32);
                }
            }
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
