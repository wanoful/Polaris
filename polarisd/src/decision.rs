use crate::cuda_vmm;
use crate::gpu::GpuState;
use crate::offload::CpuPool;
use crate::rm;
use libc::c_int;
use libpolaris::types::*;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutionResult {
    pub result: i32,
    pub output_handle: u64,
    pub output_cpu_addr: u64,
    pub rm_control_fd: i32,
    pub rm_h_client: u32,
    pub rm_h_memory: u32,
    pub rm_backing_length: u64,
}

pub(crate) fn decision_name(op: u32) -> &'static str {
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

static PROFILE_DECISIONS: OnceLock<bool> = OnceLock::new();

pub(crate) fn profile_decisions_enabled() -> bool {
    *PROFILE_DECISIONS.get_or_init(|| env_enabled("POLARISD_PROFILE_DECISIONS"))
}

pub(crate) fn profile_decision_stage(
    dec: &PolarisDecision,
    stage: &str,
    elapsed: Duration,
    result: Option<i32>,
) {
    if !profile_decisions_enabled() {
        return;
    }

    if let Some(result) = result {
        eprintln!(
            "polarisd_profile decision={} op={} block_id={} session_id={} size={} stage={} elapsed_ns={} result={}",
            dec.decision_id,
            decision_name(dec.op),
            dec.block_id,
            dec.session_id,
            dec.size_bytes,
            stage,
            elapsed.as_nanos(),
            result
        );
    } else {
        eprintln!(
            "polarisd_profile decision={} op={} block_id={} session_id={} size={} stage={} elapsed_ns={}",
            dec.decision_id,
            decision_name(dec.op),
            dec.block_id,
            dec.session_id,
            dec.size_bytes,
            stage,
            elapsed.as_nanos()
        );
    }
}

pub(crate) fn profile_elapsed(
    dec: &PolarisDecision,
    stage: &str,
    started: Instant,
    result: Option<i32>,
) {
    profile_decision_stage(dec, stage, started.elapsed(), result);
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let v = v.trim();
            !v.is_empty()
                && v != "0"
                && !v.eq_ignore_ascii_case("false")
                && !v.eq_ignore_ascii_case("no")
        })
        .unwrap_or(false)
}

/// Execute a single kernel decision against the real GPU.
pub fn execute(
    fd: c_int,
    dec: &PolarisDecision,
    gpu: &mut GpuState,
    cpu_pool: &mut CpuPool,
    rm_backend: Option<&mut rm::RmBackend>,
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
            ..Default::default()
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

    let total_started = Instant::now();
    let push_started = Instant::now();
    if let Err(e) = cuda_vmm::push_context(gpu.context) {
        profile_elapsed(
            dec,
            "push_context",
            push_started,
            Some(-(libc::ENODEV as i32)),
        );
        profile_elapsed(
            dec,
            "decision_total",
            total_started,
            Some(-(libc::ENODEV as i32)),
        );
        eprintln!("polarisd: push_context failed: {e}");
        return ExecutionResult {
            result: -(libc::ENODEV as i32),
            ..Default::default()
        };
    }
    profile_elapsed(dec, "push_context", push_started, Some(0));

    let started = Instant::now();
    let outcome = dispatch(fd, dec, gpu, cpu_pool, rm_backend);
    let elapsed_ms = started.elapsed().as_millis() as u64;
    profile_elapsed(dec, "dispatch", started, Some(outcome.result));
    if dec.timeout_ms != 0 && elapsed_ms > dec.timeout_ms as u64 {
        eprintln!(
            "polarisd: decision {} exceeded timeout budget ({} ms > {} ms)",
            dec.decision_id, elapsed_ms, dec.timeout_ms
        );
    }

    let pop_started = Instant::now();
    let pop_result = cuda_vmm::pop_context();
    let pop_status = if pop_result.is_ok() {
        0
    } else {
        -(libc::EIO as i32)
    };
    profile_elapsed(dec, "pop_context", pop_started, Some(pop_status));
    let _ = pop_result;
    profile_elapsed(dec, "decision_total", total_started, Some(outcome.result));

    outcome
}

fn dispatch(
    fd: c_int,
    dec: &PolarisDecision,
    gpu: &mut GpuState,
    cpu_pool: &mut CpuPool,
    mut rm_backend: Option<&mut rm::RmBackend>,
) -> ExecutionResult {
    let mut result: i32 = 0;
    let mut output_handle: u64 = 0;
    let output_cpu_addr: u64 = dec.cpu_addr;

    match dec.op {
        // ─── ALLOC: CUDA VMM by default, RM backing when explicitly enabled ──
        x if x == PolarisDecisionOp::Alloc as u32 => {
            let size = snap_up(dec.size_bytes, gpu.granule);
            let dst = match preferred_dst_vaddr(dec, gpu, size) {
                Some(dst) => dst,
                None => {
                    eprintln!(
                        "polarisd: VA pool exhausted for ALLOC block {}",
                        dec.block_id
                    );
                    return ExecutionResult {
                        result: -(libc::ENOMEM as i32),
                        ..Default::default()
                    };
                }
            };
            let vaddr = dst.vaddr;

            if let Some(backend) = rm_backend.as_mut() {
                let alloc_started = Instant::now();
                let allocation = match backend.alloc_for_block(dec.block_id, size) {
                    Ok(allocation) => allocation,
                    Err(e) => {
                        profile_elapsed(
                            dec,
                            "rm_alloc_backing",
                            alloc_started,
                            Some(-(libc::ENOMEM as i32)),
                        );
                        eprintln!("polarisd: RM ALLOC failed for block {}: {e}", dec.block_id);
                        if dst.release_to_pool {
                            gpu.vas.free(vaddr, size);
                        }
                        return ExecutionResult {
                            result: -(libc::ENOMEM as i32),
                            ..Default::default()
                        };
                    }
                };
                profile_elapsed(dec, "rm_alloc_backing", alloc_started, Some(0));

                gpu.track_va(dec.block_id, vaddr, size, dst.release_to_pool);
                gpu.used_bytes += size;
                eprintln!(
                    "polarisd: ALLOC block {} -> RM hClient=0x{:x} hMemory=0x{:x} va={vaddr:#x} size={size}",
                    dec.block_id,
                    backend.h_client,
                    allocation.h_memory
                );
                return ExecutionResult {
                    result: 0,
                    rm_control_fd: backend.rm_control_fd(),
                    rm_h_client: backend.h_client,
                    rm_h_memory: allocation.h_memory,
                    rm_backing_length: allocation.size,
                    ..Default::default()
                };
            }

            let create_started = Instant::now();
            let phys = match cuda_vmm::create_physical(size, gpu.device_ordinal) {
                Ok(h) => h,
                Err(e) => {
                    profile_elapsed(
                        dec,
                        "alloc_create_physical",
                        create_started,
                        Some(-(libc::ENOMEM as i32)),
                    );
                    eprintln!(
                        "polarisd: cuMemCreate failed for block {}: {e}",
                        dec.block_id
                    );
                    if dst.release_to_pool {
                        gpu.vas.free(vaddr, size);
                    }
                    return ExecutionResult {
                        result: -(libc::ENOMEM as i32),
                        ..Default::default()
                    };
                }
            };
            profile_elapsed(dec, "alloc_create_physical", create_started, Some(0));

            let map_started = Instant::now();
            if let Err(e) = cuda_vmm::map_memory(vaddr, phys, size) {
                profile_elapsed(
                    dec,
                    "alloc_map_memory",
                    map_started,
                    Some(-(libc::EINVAL as i32)),
                );
                eprintln!("polarisd: cuMemMap failed for block {}: {e}", dec.block_id);
                if let Err(re) = cuda_vmm::release_physical(phys) {
                    eprintln!(
                        "polarisd: cuMemRelease cleanup failed for block {}: {re}",
                        dec.block_id
                    );
                }
                if dst.release_to_pool {
                    gpu.vas.free(vaddr, size);
                }
                return ExecutionResult {
                    result: -(libc::EINVAL as i32),
                    ..Default::default()
                };
            }
            profile_elapsed(dec, "alloc_map_memory", map_started, Some(0));

            let access_started = Instant::now();
            if let Err(e) = cuda_vmm::set_access(vaddr, size, gpu.device_ordinal) {
                profile_elapsed(
                    dec,
                    "alloc_set_access",
                    access_started,
                    Some(-(libc::EINVAL as i32)),
                );
                eprintln!(
                    "polarisd: cuMemSetAccess failed for block {}: {e}",
                    dec.block_id
                );
                let _ = cuda_vmm::unmap_memory(vaddr, size);
                if let Err(re) = cuda_vmm::release_physical(phys) {
                    eprintln!(
                        "polarisd: cuMemRelease cleanup failed for block {}: {re}",
                        dec.block_id
                    );
                }
                if dst.release_to_pool {
                    gpu.vas.free(vaddr, size);
                }
                return ExecutionResult {
                    result: -(libc::EINVAL as i32),
                    ..Default::default()
                };
            }
            profile_elapsed(dec, "alloc_set_access", access_started, Some(0));

            gpu.track_handle(dec.block_id, phys);
            gpu.track_va(dec.block_id, vaddr, size, dst.release_to_pool);
            gpu.used_bytes += size;
            output_handle = phys;

            eprintln!(
                "polarisd: ALLOC block {} -> phys={phys:#x} va={vaddr:#x} size={size}",
                dec.block_id
            );
        }

        // ─── FREE: release CUDA VMM or daemon-owned RM backing + CPU pool ──
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

            if let Some(backend) = rm_backend.as_mut() {
                if let Err(e) = backend.free_block(dec.block_id) {
                    eprintln!("polarisd: RM FREE failed for block {}: {e}", dec.block_id);
                    result = -(libc::EIO as i32);
                }
            }

            if phys == 0 && rm_backend.is_some() {
                if let Some(va) = gpu.get_va_alloc(dec.block_id) {
                    let size = va.size;
                    let vaddr = va.vaddr;
                    gpu.used_bytes = gpu.used_bytes.saturating_sub(size);
                    eprintln!(
                        "polarisd: FREE block {} -> RM backing va={:#x}",
                        dec.block_id, vaddr
                    );
                }
            } else if let Some(va) = gpu.get_va_alloc(dec.block_id) {
                let size = va.size;
                let _ = cuda_vmm::unmap_memory(vaddr, size);
                if phys != 0 {
                    if let Err(e) = cuda_vmm::release_physical(phys) {
                        eprintln!(
                            "polarisd: FREE cuMemRelease failed for block {}: {e}",
                            dec.block_id
                        );
                    }
                }
                gpu.used_bytes = gpu.used_bytes.saturating_sub(size);
                eprintln!(
                    "polarisd: FREE block {} -> phys={phys:#x} va={vaddr:#x}",
                    dec.block_id
                );
            } else if phys != 0 {
                if let Err(e) = cuda_vmm::release_physical(phys) {
                    eprintln!(
                        "polarisd: FREE cuMemRelease failed for block {}: {e}",
                        dec.block_id
                    );
                }
                eprintln!(
                    "polarisd: FREE block {} -> phys={phys:#x} (no VA track)",
                    dec.block_id
                );
            }

            // Free the CPU buffer if this block was offloaded.
            if let Some(cpu_addr) = cpu_pool.untrack(dec.block_id) {
                let sz = dec.size_bytes.max(1);
                cpu_pool.free(cpu_addr, sz);
                eprintln!(
                    "polarisd: FREE block {} -> released CPU buffer {cpu_addr:#x}",
                    dec.block_id
                );
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
                eprintln!(
                    "polarisd: MAP_EXISTING block {} -> phys={:#x} va={:#x}",
                    dec.block_id, dec.src_handle, dec.dst_vaddr
                );
            }
        }

        // ─── UNMAP: cuMemUnmap ─────────────────────────────────────
        x if x == PolarisDecisionOp::Unmap as u32 => {
            let vaddr = if dec.dst_vaddr != 0 {
                dec.dst_vaddr
            } else {
                gpu.get_va_alloc(dec.block_id).map(|v| v.vaddr).unwrap_or(0)
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
            if let Some(backend) = rm_backend.as_mut() {
                if backend.has_block(dec.block_id) && dec.src_handle == 0 {
                    return crate::offload::execute_rm_offload(fd, dec, gpu, cpu_pool, backend);
                }
            }
            let (res, handle, cpu_addr) = crate::offload::execute_offload(dec, gpu, cpu_pool);
            result = res;
            output_handle = handle;
            return ExecutionResult {
                result,
                output_handle,
                output_cpu_addr: cpu_addr,
                ..Default::default()
            };
        }

        // ─── RELOAD: alloc + map + copy CPU→GPU VA, report new phys handle ──
        x if x == PolarisDecisionOp::Reload as u32 => {
            if let Some(backend) = rm_backend.as_mut() {
                if dec.src_handle == 0 {
                    return crate::offload::execute_rm_reload(fd, dec, gpu, cpu_pool, backend);
                }
            }
            let (res, handle, cpu_addr) = crate::offload::execute_reload(dec, gpu, cpu_pool);
            result = res;
            output_handle = handle;
            return ExecutionResult {
                result,
                output_handle,
                output_cpu_addr: cpu_addr,
                ..Default::default()
            };
        }

        // ─── COW_BREAK: alloc new + copy old VA→new VA (GPU DtoD) ───
        x if x == PolarisDecisionOp::CowBreak as u32 => {
            let size = snap_up(dec.size_bytes, gpu.granule);
            let sz_usize = size as usize;

            if let Some(backend) = rm_backend.as_mut() {
                if dec._reserved[1] & POLARIS_DECISION_FLAG_SOURCE_BLOCK_ID_VALID != 0
                    || backend.has_block(dec.block_id)
                    || dec.src_handle == 0
                {
                    return crate::offload::execute_rm_cow_break(fd, dec, gpu, cpu_pool, backend);
                }
            }

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
                                ..Default::default()
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
                                ..Default::default()
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
                        ..Default::default()
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
                    eprintln!(
                        "polarisd: COW_BREAK cuMemCreate failed for block {}: {e}",
                        dec.block_id
                    );
                    if dst.release_to_pool {
                        gpu.vas.free(dst_vaddr, size);
                    }
                    return ExecutionResult {
                        result: -(libc::ENOMEM as i32),
                        ..Default::default()
                    };
                }
            };

            // Map new physical into the destination VA.
            if let Err(e) = cuda_vmm::map_memory(dst_vaddr, new_phys, size) {
                eprintln!(
                    "polarisd: COW_BREAK cuMemMap failed for block {}: {e}",
                    dec.block_id
                );
                if let Err(re) = cuda_vmm::release_physical(new_phys) {
                    eprintln!("polarisd: COW_BREAK cuMemRelease cleanup failed: {re}");
                }
                if dst.release_to_pool {
                    gpu.vas.free(dst_vaddr, size);
                }
                return ExecutionResult {
                    result: -(libc::EINVAL as i32),
                    ..Default::default()
                };
            }

            // Set access for the destination mapping.
            if let Err(e) = cuda_vmm::set_access(dst_vaddr, size, gpu.device_ordinal) {
                eprintln!(
                    "polarisd: COW_BREAK cuMemSetAccess failed for block {}: {e}",
                    dec.block_id
                );
                let _ = cuda_vmm::unmap_memory(dst_vaddr, size);
                if let Err(re) = cuda_vmm::release_physical(new_phys) {
                    eprintln!("polarisd: COW_BREAK cuMemRelease cleanup failed: {re}");
                }
                if dst.release_to_pool {
                    gpu.vas.free(dst_vaddr, size);
                }
                return ExecutionResult {
                    result: -(libc::EINVAL as i32),
                    ..Default::default()
                };
            }

            // GPU-to-GPU copy: old (source) VA → new (destination) VA.
            if let Err(e) = cuda_vmm::copy_device_to_device(dst_vaddr, src_vaddr, sz_usize) {
                eprintln!(
                    "polarisd: COW_BREAK cuMemcpyDtoD failed for block {}: {e}",
                    dec.block_id
                );
                let _ = cuda_vmm::unmap_memory(dst_vaddr, size);
                if let Err(re) = cuda_vmm::release_physical(new_phys) {
                    eprintln!("polarisd: COW_BREAK cuMemRelease cleanup failed: {re}");
                }
                if dst.release_to_pool {
                    gpu.vas.free(dst_vaddr, size);
                }
                return ExecutionResult {
                    result: -(libc::EFAULT as i32),
                    ..Default::default()
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
        ..Default::default()
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
