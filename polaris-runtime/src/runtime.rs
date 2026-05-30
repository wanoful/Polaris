use crate::cuda_vmm;
use libc::c_int;
use libpolaris::ioctl;
use libpolaris::types::*;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

const POLARIS_DEV: &str = "/dev/polaris";

#[derive(Clone, Copy)]
pub struct RuntimeConfig {
    pub gpu_id: u32,
    pub device_ordinal: i32,
    pub total_bytes: u64,
    pub budget_bytes: u64,
    pub cpu_pool_bytes: u64,
    pub va_reserve_bytes: u64,
    pub block_size: u64,
}

pub struct RuntimeInfo {
    pub va_base: u64,
    pub va_size: u64,
    pub granule: u64,
    pub cpu_pool_base: u64,
    pub cpu_pool_bytes: u64,
}

pub struct KvAllocationInfo {
    pub va: u64,
    pub size: u64,
    pub block_size: u64,
    pub block_count: u64,
}

pub struct Runtime {
    inner: Arc<Mutex<RuntimeInner>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

struct RuntimeInner {
    dev: File,
    gpu: GpuState,
    cpu_pool: CpuPool,
    explicit_kv: HashMap<u64, ExplicitKvAlloc>,
}

#[derive(Clone, Copy)]
struct VaAlloc {
    vaddr: u64,
    size: u64,
    from_pool: bool,
}

struct GpuState {
    gpu_id: u32,
    used_bytes: u64,
    device_ordinal: i32,
    context: cuda_vmm::CudaContext,
    vas: GpuVaPool,
    granule: u64,
    phys_handles: HashMap<u64, u64>,
    va_allocs: HashMap<u64, VaAlloc>,
}

// The CUDA driver context handle is process-global state managed by CUDA. The
// runtime serializes access through RuntimeInner's mutex before pushing it.
unsafe impl Send for GpuState {}

struct GpuVaPool {
    base: u64,
    size: u64,
    free_ranges: Vec<(u64, u64)>,
}

struct CpuPool {
    base: u64,
    total: u64,
    used: u64,
    free_ranges: Vec<(u64, u64)>,
    allocations: HashMap<u64, u64>,
}

struct ExecutionResult {
    result: i32,
    output_handle: u64,
    output_cpu_addr: u64,
}

struct DstVa {
    vaddr: u64,
    release_to_pool: bool,
}

struct ExplicitKvAlloc {
    va: u64,
    size: u64,
    block_size: u64,
    handles: Vec<u64>,
}

impl Runtime {
    pub fn create(cfg: RuntimeConfig) -> Result<Self, String> {
        let dev = OpenOptions::new()
            .read(true)
            .write(true)
            .open(POLARIS_DEV)
            .map_err(|e| format!("open {POLARIS_DEV} failed: {e}"))?;

        cuda_vmm::init()?;
        let context = cuda_vmm::retain_primary_context(cfg.device_ordinal)?;
        cuda_vmm::push_context(context)?;

        let create_result = Self::create_with_context(dev, cfg, context);
        cuda_vmm::pop_context();

        match create_result {
            Ok(runtime) => Ok(runtime),
            Err(e) => {
                let _ = cuda_vmm::release_primary_context(cfg.device_ordinal);
                Err(e)
            }
        }
    }

    fn create_with_context(
        dev: File,
        cfg: RuntimeConfig,
        context: cuda_vmm::CudaContext,
    ) -> Result<Self, String> {
        let fd = dev.as_raw_fd();
        let granule = cuda_vmm::allocation_granularity(cfg.device_ordinal)?;
        let va_size = snap_up(cfg.va_reserve_bytes.max(granule), granule);
        let va_base = cuda_vmm::reserve_va(va_size)?;

        let cpu_pool_bytes = snap_up(cfg.cpu_pool_bytes, 4096);
        let cpu_pool_base = if cpu_pool_bytes == 0 {
            0
        } else {
            cuda_vmm::alloc_host(cpu_pool_bytes)?
        };

        let register_result = register_runtime(fd, &cfg, va_base, va_size);
        if let Err(e) = register_result {
            let _ = cuda_vmm::free_host(cpu_pool_base);
            let _ = cuda_vmm::free_va(va_base, va_size);
            return Err(e);
        }

        let gpu = GpuState {
            gpu_id: cfg.gpu_id,
            used_bytes: 0,
            device_ordinal: cfg.device_ordinal,
            context,
            vas: GpuVaPool::new(va_base, va_size),
            granule,
            phys_handles: HashMap::new(),
            va_allocs: HashMap::new(),
        };

        let inner = RuntimeInner {
            dev,
            gpu,
            cpu_pool: CpuPool::new(cpu_pool_base, cpu_pool_bytes),
            explicit_kv: HashMap::new(),
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            stop: Arc::new(AtomicBool::new(false)),
            worker: None,
        })
    }

    pub fn info(&self) -> RuntimeInfo {
        let inner = self.inner.lock().expect("runtime mutex poisoned");
        RuntimeInfo {
            va_base: inner.gpu.vas.base,
            va_size: inner.gpu.vas.size,
            granule: inner.gpu.granule,
            cpu_pool_base: inner.cpu_pool.base,
            cpu_pool_bytes: inner.cpu_pool.total,
        }
    }

    pub fn start(&mut self) -> Result<(), String> {
        if self.worker.is_some() {
            return Ok(());
        }

        self.stop.store(false, Ordering::Release);
        let inner = Arc::clone(&self.inner);
        let stop = Arc::clone(&self.stop);
        self.worker = Some(
            thread::Builder::new()
                .name("polaris-runtime".to_string())
                .spawn(move || {
                    while !stop.load(Ordering::Acquire) {
                        let mut guard = match inner.lock() {
                            Ok(guard) => guard,
                            Err(_) => break,
                        };
                        if let Err(e) = guard.poll_once() {
                            eprintln!("polaris-runtime: poll_once failed: {e}");
                            if e.contains("ENODEV") {
                                break;
                            }
                        }
                    }
                })
                .map_err(|e| format!("spawn polaris runtime thread failed: {e}"))?,
        );
        Ok(())
    }

    pub fn poll_once(&mut self) -> Result<usize, String> {
        self.inner
            .lock()
            .map_err(|_| "runtime mutex poisoned".to_string())?
            .poll_once()
    }

    pub fn alloc_kv(&mut self, size: u64, alignment: u64) -> Result<KvAllocationInfo, String> {
        self.inner
            .lock()
            .map_err(|_| "runtime mutex poisoned".to_string())?
            .alloc_kv(size, alignment)
    }

    pub fn map_kv_all(&mut self, va: u64) -> Result<(), String> {
        self.inner
            .lock()
            .map_err(|_| "runtime mutex poisoned".to_string())?
            .map_kv_all(va)
    }

    pub fn unmap_kv(&mut self, va: u64) -> Result<(), String> {
        self.inner
            .lock()
            .map_err(|_| "runtime mutex poisoned".to_string())?
            .unmap_kv(va)
    }

    pub fn free_kv(&mut self, va: u64) -> Result<(), String> {
        self.inner
            .lock()
            .map_err(|_| "runtime mutex poisoned".to_string())?
            .free_kv(va)
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.stop();
        if let Ok(mut inner) = self.inner.lock() {
            inner.cleanup();
        }
    }
}

impl RuntimeInner {
    fn poll_once(&mut self) -> Result<usize, String> {
        let mut arg = PolarisGetDecisionArg::default();
        ioctl::ioctl_read(self.fd(), ioctl::POLARIS_GET_DECISION, &mut arg)
            .map_err(|e| format!("POLARIS_GET_DECISION failed: errno {e}"))?;

        let count = (arg.count as usize).min(POLARIS_MAX_DECISIONS_PER_POLL);
        for dec in arg.decisions.iter().take(count) {
            let exec = execute_decision(dec, &mut self.gpu, &mut self.cpu_pool);
            let complete = PolarisCompleteOperationArg {
                decision_id: dec.decision_id,
                generation: dec.generation,
                result: exec.result,
                output_handle: exec.output_handle,
                output_cpu_addr: exec.output_cpu_addr,
                ..Default::default()
            };

            ioctl::ioctl_write(self.fd(), ioctl::POLARIS_COMPLETE_OPERATION, &complete).map_err(
                |e| {
                    format!(
                        "POLARIS_COMPLETE_OPERATION decision {} failed: errno {e}",
                        dec.decision_id
                    )
                },
            )?;
        }

        Ok(count)
    }

    fn alloc_kv(&mut self, size: u64, alignment: u64) -> Result<KvAllocationInfo, String> {
        if size == 0 {
            return Err("KV allocation size must be non-zero".to_string());
        }

        let block_size = self.gpu.granule.max(alignment).max(1);
        let block_size = snap_up(block_size, self.gpu.granule);
        let padded_size = snap_up(size, block_size);
        let va = self
            .gpu
            .vas
            .allocate(padded_size, alignment.max(self.gpu.granule))
            .ok_or_else(|| {
                format!(
                    "POLARIS KV VA pool exhausted: requested {} bytes, alignment {}",
                    padded_size, alignment
                )
            })?;
        let block_count = padded_size / block_size;
        self.explicit_kv.insert(
            va,
            ExplicitKvAlloc {
                va,
                size: padded_size,
                block_size,
                handles: vec![0; block_count as usize],
            },
        );
        Ok(KvAllocationInfo {
            va,
            size: padded_size,
            block_size,
            block_count,
        })
    }

    fn map_kv_all(&mut self, va: u64) -> Result<(), String> {
        let alloc = self
            .explicit_kv
            .get_mut(&va)
            .ok_or_else(|| format!("unknown POLARIS KV allocation at {va:#x}"))?;

        cuda_vmm::push_context(self.gpu.context)?;
        let result = map_explicit_kv_all(alloc, &mut self.gpu);
        cuda_vmm::pop_context();
        result
    }

    fn unmap_kv(&mut self, va: u64) -> Result<(), String> {
        let alloc = self
            .explicit_kv
            .get_mut(&va)
            .ok_or_else(|| format!("unknown POLARIS KV allocation at {va:#x}"))?;

        cuda_vmm::push_context(self.gpu.context)?;
        let result = unmap_explicit_kv(alloc, &mut self.gpu);
        cuda_vmm::pop_context();
        result
    }

    fn free_kv(&mut self, va: u64) -> Result<(), String> {
        let mut alloc = self
            .explicit_kv
            .remove(&va)
            .ok_or_else(|| format!("unknown POLARIS KV allocation at {va:#x}"))?;

        cuda_vmm::push_context(self.gpu.context)?;
        let result = unmap_explicit_kv(&mut alloc, &mut self.gpu);
        cuda_vmm::pop_context();
        self.gpu.vas.free(alloc.va, alloc.size);
        result
    }

    fn cleanup(&mut self) {
        let _ = cuda_vmm::push_context(self.gpu.context);
        for (_, mut alloc) in self.explicit_kv.drain() {
            let _ = unmap_explicit_kv(&mut alloc, &mut self.gpu);
            self.gpu.vas.free(alloc.va, alloc.size);
        }
        for (_, va) in self.gpu.va_allocs.drain() {
            let _ = cuda_vmm::unmap_memory(va.vaddr, va.size);
        }
        for (_, handle) in self.gpu.phys_handles.drain() {
            let _ = cuda_vmm::release_physical(handle);
        }
        let _ = cuda_vmm::free_host(self.cpu_pool.base);
        let _ = cuda_vmm::free_va(self.gpu.vas.base, self.gpu.vas.size);
        cuda_vmm::pop_context();
        let _ = cuda_vmm::release_primary_context(self.gpu.device_ordinal);
    }

    fn fd(&self) -> RawFd {
        self.dev.as_raw_fd()
    }
}

fn register_runtime(
    fd: c_int,
    cfg: &RuntimeConfig,
    va_base: u64,
    va_size: u64,
) -> Result<(), String> {
    let gpu_arg = PolarisRegisterGpuArg {
        gpu_id: cfg.gpu_id,
        total_bytes: cfg.total_bytes,
        budget_bytes: cfg.budget_bytes,
        cpu_pool_bytes: cfg.cpu_pool_bytes,
        numa_node: 0,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_REGISTER_GPU, &gpu_arg)
        .map_err(|e| format!("POLARIS_REGISTER_GPU failed: errno {e}"))?;

    let mut va_arg = PolarisRegisterVaRangeArg {
        gpu_id: cfg.gpu_id,
        base: va_base,
        length: va_size,
        block_size: cfg.block_size,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_REGISTER_VA_RANGE, &mut va_arg)
        .map_err(|e| format!("POLARIS_REGISTER_VA_RANGE failed: errno {e}"))?;
    Ok(())
}

fn execute_decision(
    dec: &PolarisDecision,
    gpu: &mut GpuState,
    cpu_pool: &mut CpuPool,
) -> ExecutionResult {
    if dec.gpu_id != gpu.gpu_id {
        return exec_error(libc::ENODEV);
    }

    if let Err(e) = cuda_vmm::push_context(gpu.context) {
        eprintln!("polaris-runtime: push_context failed: {e}");
        return exec_error(libc::ENODEV);
    }

    let started = Instant::now();
    let result = dispatch_decision(dec, gpu, cpu_pool);
    let elapsed_ms = started.elapsed().as_millis() as u64;
    if dec.timeout_ms != 0 && elapsed_ms > dec.timeout_ms as u64 {
        eprintln!(
            "polaris-runtime: decision {} exceeded timeout ({} ms > {} ms)",
            dec.decision_id, elapsed_ms, dec.timeout_ms
        );
    }

    cuda_vmm::pop_context();
    result
}

fn dispatch_decision(
    dec: &PolarisDecision,
    gpu: &mut GpuState,
    cpu_pool: &mut CpuPool,
) -> ExecutionResult {
    match dec.op {
        x if x == PolarisDecisionOp::Alloc as u32 => alloc_block(dec, gpu),
        x if x == PolarisDecisionOp::Free as u32 => free_block(dec, gpu, cpu_pool),
        x if x == PolarisDecisionOp::MapExisting as u32 => map_existing(dec, gpu),
        x if x == PolarisDecisionOp::Unmap as u32 => unmap_block(dec, gpu),
        x if x == PolarisDecisionOp::Offload as u32 => offload_block(dec, gpu, cpu_pool),
        x if x == PolarisDecisionOp::Reload as u32 => reload_block(dec, gpu, cpu_pool),
        x if x == PolarisDecisionOp::CowBreak as u32 => cow_break(dec, gpu),
        _ => exec_error(libc::EINVAL),
    }
}

fn alloc_block(dec: &PolarisDecision, gpu: &mut GpuState) -> ExecutionResult {
    let size = snap_up(dec.size_bytes, gpu.granule);
    let Some(dst) = preferred_dst_vaddr(dec, gpu, size) else {
        return exec_error(libc::ENOMEM);
    };

    let phys = match cuda_vmm::create_physical(size, gpu.device_ordinal) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("polaris-runtime: ALLOC cuMemCreate failed: {e}");
            if dst.release_to_pool {
                gpu.vas.free(dst.vaddr, size);
            }
            return exec_error(libc::ENOMEM);
        }
    };

    if let Err(e) = map_and_access(dst.vaddr, phys, size, gpu.device_ordinal) {
        eprintln!("polaris-runtime: ALLOC map failed: {e}");
        let _ = cuda_vmm::release_physical(phys);
        if dst.release_to_pool {
            gpu.vas.free(dst.vaddr, size);
        }
        return exec_error(libc::EINVAL);
    }

    gpu.track_handle(dec.block_id, phys);
    gpu.track_va(dec.block_id, dst.vaddr, size, dst.release_to_pool);
    gpu.used_bytes += size;
    ExecutionResult {
        result: 0,
        output_handle: phys,
        output_cpu_addr: 0,
    }
}

fn free_block(
    dec: &PolarisDecision,
    gpu: &mut GpuState,
    cpu_pool: &mut CpuPool,
) -> ExecutionResult {
    let phys = if dec.src_handle != 0 {
        dec.src_handle
    } else {
        gpu.get_handle(dec.block_id).unwrap_or(0)
    };
    let va = gpu.get_va_alloc(dec.block_id).copied();
    let vaddr = dec.src_vaddr.max(va.map(|v| v.vaddr).unwrap_or(0));

    if let Some(va) = va {
        let _ = cuda_vmm::unmap_memory(vaddr, va.size);
        gpu.used_bytes = gpu.used_bytes.saturating_sub(va.size);
    }
    if phys != 0 {
        let _ = cuda_vmm::release_physical(phys);
    }
    if let Some(cpu_addr) = cpu_pool.untrack(dec.block_id) {
        cpu_pool.free(cpu_addr, snap_up(dec.size_bytes, 4096));
    }
    gpu.remove_block(dec.block_id);
    exec_ok()
}

fn map_existing(dec: &PolarisDecision, gpu: &mut GpuState) -> ExecutionResult {
    let size = snap_up(dec.size_bytes, gpu.granule);
    if dec.src_handle == 0 || dec.dst_vaddr == 0 {
        return exec_error(libc::EINVAL);
    }

    if let Err(e) = map_and_access(dec.dst_vaddr, dec.src_handle, size, gpu.device_ordinal) {
        eprintln!("polaris-runtime: MAP_EXISTING failed: {e}");
        return exec_error(libc::EINVAL);
    }
    gpu.track_va(dec.block_id, dec.dst_vaddr, size, false);
    exec_ok()
}

fn unmap_block(dec: &PolarisDecision, gpu: &mut GpuState) -> ExecutionResult {
    let va = gpu.get_va_alloc(dec.block_id).copied();
    let vaddr = if dec.dst_vaddr != 0 {
        dec.dst_vaddr
    } else {
        va.map(|v| v.vaddr).unwrap_or(0)
    };
    let size = snap_up(
        dec.size_bytes.max(va.map(|v| v.size).unwrap_or(0)),
        gpu.granule,
    );
    if vaddr != 0 && size != 0 {
        if let Err(e) = cuda_vmm::unmap_memory(vaddr, size) {
            eprintln!("polaris-runtime: UNMAP failed: {e}");
            return exec_error(libc::EINVAL);
        }
    }
    exec_ok()
}

fn offload_block(
    dec: &PolarisDecision,
    gpu: &mut GpuState,
    cpu_pool: &mut CpuPool,
) -> ExecutionResult {
    let va = gpu.get_va_alloc(dec.block_id).copied();
    let src_vaddr = if dec.src_vaddr != 0 {
        dec.src_vaddr
    } else {
        va.map(|v| v.vaddr).unwrap_or(0)
    };
    let size = snap_up(
        dec.size_bytes.max(va.map(|v| v.size).unwrap_or(0)),
        gpu.granule,
    );
    if src_vaddr == 0 || size == 0 {
        return exec_error(libc::EINVAL);
    }

    let Some(cpu_addr) = cpu_pool.allocate(size) else {
        return exec_error(libc::ENOMEM);
    };

    if let Err(e) = cuda_vmm::copy_dtoh(cpu_addr, src_vaddr, size) {
        eprintln!("polaris-runtime: OFFLOAD copy failed: {e}");
        cpu_pool.free(cpu_addr, size);
        return exec_error(libc::EFAULT);
    }
    if let Err(e) = cuda_vmm::unmap_memory(src_vaddr, size) {
        eprintln!("polaris-runtime: OFFLOAD unmap failed: {e}");
        cpu_pool.free(cpu_addr, size);
        return exec_error(libc::EINVAL);
    }
    if let Some(handle) = gpu.get_handle(dec.block_id) {
        let _ = cuda_vmm::release_physical(handle);
    }

    cpu_pool.track(dec.block_id, cpu_addr);
    gpu.clear_handle(dec.block_id);
    gpu.used_bytes = gpu.used_bytes.saturating_sub(size);
    gpu.track_va(dec.block_id, src_vaddr, size, false);
    ExecutionResult {
        result: 0,
        output_handle: 0,
        output_cpu_addr: cpu_addr,
    }
}

fn reload_block(
    dec: &PolarisDecision,
    gpu: &mut GpuState,
    cpu_pool: &mut CpuPool,
) -> ExecutionResult {
    let size = snap_up(dec.size_bytes, gpu.granule);
    let vaddr = if dec.dst_vaddr != 0 {
        dec.dst_vaddr
    } else {
        gpu.get_va_alloc(dec.block_id).map(|v| v.vaddr).unwrap_or(0)
    };
    let cpu_addr = if dec.cpu_addr != 0 {
        dec.cpu_addr
    } else {
        cpu_pool.get(dec.block_id).unwrap_or(0)
    };
    if vaddr == 0 || cpu_addr == 0 {
        return exec_error(libc::EINVAL);
    }

    let phys = match cuda_vmm::create_physical(size, gpu.device_ordinal) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("polaris-runtime: RELOAD cuMemCreate failed: {e}");
            return exec_error(libc::ENOMEM);
        }
    };
    if let Err(e) = map_and_access(vaddr, phys, size, gpu.device_ordinal) {
        eprintln!("polaris-runtime: RELOAD map failed: {e}");
        let _ = cuda_vmm::release_physical(phys);
        return exec_error(libc::EINVAL);
    }
    if let Err(e) = cuda_vmm::copy_htod(vaddr, cpu_addr, size) {
        eprintln!("polaris-runtime: RELOAD copy failed: {e}");
        let _ = cuda_vmm::unmap_memory(vaddr, size);
        let _ = cuda_vmm::release_physical(phys);
        return exec_error(libc::EFAULT);
    }

    let _ = cpu_pool.untrack(dec.block_id);
    cpu_pool.free(cpu_addr, size);
    gpu.track_handle(dec.block_id, phys);
    gpu.track_va(dec.block_id, vaddr, size, false);
    gpu.used_bytes += size;
    ExecutionResult {
        result: 0,
        output_handle: phys,
        output_cpu_addr: 0,
    }
}

fn cow_break(dec: &PolarisDecision, gpu: &mut GpuState) -> ExecutionResult {
    let size = snap_up(dec.size_bytes, gpu.granule);
    let src_vaddr = if dec.src_vaddr != 0 {
        dec.src_vaddr
    } else if let Some(src_block) = gpu.find_block_by_phys(dec.src_handle) {
        gpu.get_va_alloc(src_block).map(|v| v.vaddr).unwrap_or(0)
    } else {
        0
    };
    if src_vaddr == 0 {
        return exec_error(libc::EINVAL);
    }

    let Some(dst) = preferred_dst_vaddr(dec, gpu, size) else {
        return exec_error(libc::ENOMEM);
    };
    let phys = match cuda_vmm::create_physical(size, gpu.device_ordinal) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("polaris-runtime: COW_BREAK cuMemCreate failed: {e}");
            if dst.release_to_pool {
                gpu.vas.free(dst.vaddr, size);
            }
            return exec_error(libc::ENOMEM);
        }
    };
    if let Err(e) = map_and_access(dst.vaddr, phys, size, gpu.device_ordinal) {
        eprintln!("polaris-runtime: COW_BREAK map failed: {e}");
        let _ = cuda_vmm::release_physical(phys);
        if dst.release_to_pool {
            gpu.vas.free(dst.vaddr, size);
        }
        return exec_error(libc::EINVAL);
    }
    if let Err(e) = cuda_vmm::copy_dtod(dst.vaddr, src_vaddr, size) {
        eprintln!("polaris-runtime: COW_BREAK copy failed: {e}");
        let _ = cuda_vmm::unmap_memory(dst.vaddr, size);
        let _ = cuda_vmm::release_physical(phys);
        if dst.release_to_pool {
            gpu.vas.free(dst.vaddr, size);
        }
        return exec_error(libc::EFAULT);
    }

    gpu.track_handle(dec.block_id, phys);
    gpu.track_va(dec.block_id, dst.vaddr, size, dst.release_to_pool);
    gpu.used_bytes += size;
    ExecutionResult {
        result: 0,
        output_handle: phys,
        output_cpu_addr: 0,
    }
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

fn map_and_access(vaddr: u64, phys: u64, size: u64, device_ordinal: i32) -> Result<(), String> {
    cuda_vmm::map_memory(vaddr, phys, size)?;
    if let Err(e) = cuda_vmm::set_access(vaddr, size, device_ordinal) {
        let _ = cuda_vmm::unmap_memory(vaddr, size);
        return Err(e);
    }
    Ok(())
}

fn map_explicit_kv_all(alloc: &mut ExplicitKvAlloc, gpu: &mut GpuState) -> Result<(), String> {
    for idx in 0..alloc.handles.len() {
        if alloc.handles[idx] != 0 {
            continue;
        }

        let vaddr = alloc.va + (idx as u64 * alloc.block_size);
        let phys =
            cuda_vmm::create_physical(alloc.block_size, gpu.device_ordinal).map_err(|e| {
                format!(
                    "POLARIS KV cuMemCreate block {} at {vaddr:#x} failed: {e}",
                    idx
                )
            })?;

        if let Err(e) = map_and_access(vaddr, phys, alloc.block_size, gpu.device_ordinal) {
            let _ = cuda_vmm::release_physical(phys);
            return Err(format!(
                "POLARIS KV map block {} at {vaddr:#x} failed: {e}",
                idx
            ));
        }

        alloc.handles[idx] = phys;
        gpu.used_bytes += alloc.block_size;
    }
    Ok(())
}

fn unmap_explicit_kv(alloc: &mut ExplicitKvAlloc, gpu: &mut GpuState) -> Result<(), String> {
    let mut first_error = None;
    for (idx, handle) in alloc.handles.iter_mut().enumerate() {
        if *handle == 0 {
            continue;
        }

        let vaddr = alloc.va + (idx as u64 * alloc.block_size);
        if let Err(e) = cuda_vmm::unmap_memory(vaddr, alloc.block_size) {
            first_error.get_or_insert_with(|| {
                format!("POLARIS KV unmap block {} at {vaddr:#x} failed: {e}", idx)
            });
        }
        if let Err(e) = cuda_vmm::release_physical(*handle) {
            first_error.get_or_insert_with(|| {
                format!(
                    "POLARIS KV release block {} handle {:#x} failed: {e}",
                    idx, *handle
                )
            });
        }
        *handle = 0;
        gpu.used_bytes = gpu.used_bytes.saturating_sub(alloc.block_size);
    }

    match first_error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn exec_ok() -> ExecutionResult {
    ExecutionResult {
        result: 0,
        output_handle: 0,
        output_cpu_addr: 0,
    }
}

fn exec_error(errno: i32) -> ExecutionResult {
    ExecutionResult {
        result: -errno.abs(),
        output_handle: 0,
        output_cpu_addr: 0,
    }
}

impl GpuState {
    fn track_handle(&mut self, block_id: u64, phys_handle: u64) {
        self.phys_handles.insert(block_id, phys_handle);
    }

    fn track_va(&mut self, block_id: u64, vaddr: u64, size: u64, from_pool: bool) {
        self.va_allocs.insert(
            block_id,
            VaAlloc {
                vaddr,
                size,
                from_pool,
            },
        );
        if !from_pool {
            self.vas.remove(vaddr, size);
        }
    }

    fn get_handle(&self, block_id: u64) -> Option<u64> {
        self.phys_handles.get(&block_id).copied()
    }

    fn get_va_alloc(&self, block_id: u64) -> Option<&VaAlloc> {
        self.va_allocs.get(&block_id)
    }

    fn find_block_by_phys(&self, phys_handle: u64) -> Option<u64> {
        self.phys_handles
            .iter()
            .find(|(_, ph)| **ph == phys_handle)
            .map(|(bid, _)| *bid)
    }

    fn remove_block(&mut self, block_id: u64) {
        self.phys_handles.remove(&block_id);
        if let Some(va) = self.va_allocs.remove(&block_id) {
            if va.from_pool {
                self.vas.free(va.vaddr, va.size);
            }
        }
    }

    fn clear_handle(&mut self, block_id: u64) {
        self.phys_handles.remove(&block_id);
    }
}

impl GpuVaPool {
    fn new(base: u64, size: u64) -> Self {
        Self {
            base,
            size,
            free_ranges: vec![(base, size)],
        }
    }

    fn allocate(&mut self, size: u64, alignment: u64) -> Option<u64> {
        let align = alignment.max(1);
        for idx in 0..self.free_ranges.len() {
            let (start, range_size) = self.free_ranges[idx];
            let aligned = snap_up(start, align);
            let offset = aligned.checked_sub(start)?;
            if offset + size > range_size {
                continue;
            }
            if offset == 0 {
                if size == range_size {
                    self.free_ranges.remove(idx);
                } else {
                    self.free_ranges[idx] = (start + size, range_size - size);
                }
            } else if offset + size == range_size {
                self.free_ranges[idx] = (start, offset);
            } else {
                self.free_ranges[idx] = (start, offset);
                self.free_ranges
                    .insert(idx + 1, (aligned + size, range_size - offset - size));
            }
            return Some(aligned);
        }
        None
    }

    fn remove(&mut self, vaddr: u64, size: u64) {
        if size == 0 {
            return;
        }
        for idx in 0..self.free_ranges.len() {
            let (start, range_size) = self.free_ranges[idx];
            let end = start + range_size;
            if vaddr >= start && vaddr + size <= end {
                let before = vaddr - start;
                let after = end - (vaddr + size);
                if before == 0 && after == 0 {
                    self.free_ranges.remove(idx);
                } else if before == 0 {
                    self.free_ranges[idx] = (vaddr + size, after);
                } else if after == 0 {
                    self.free_ranges[idx] = (start, before);
                } else {
                    self.free_ranges[idx] = (start, before);
                    self.free_ranges.insert(idx + 1, (vaddr + size, after));
                }
                return;
            }
        }
    }

    fn free(&mut self, vaddr: u64, size: u64) {
        if size == 0 {
            return;
        }
        let mut insert_at = 0;
        while insert_at < self.free_ranges.len() && self.free_ranges[insert_at].0 < vaddr {
            insert_at += 1;
        }
        self.free_ranges.insert(insert_at, (vaddr, size));
        self.coalesce();
    }

    fn coalesce(&mut self) {
        if self.free_ranges.len() < 2 {
            return;
        }
        self.free_ranges.sort_by_key(|(start, _)| *start);
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(self.free_ranges.len());
        for (start, size) in self.free_ranges.drain(..) {
            if let Some(last) = merged.last_mut() {
                let last_end = last.0 + last.1;
                if start <= last_end {
                    last.1 = last.1.max(start + size - last.0);
                    continue;
                }
            }
            merged.push((start, size));
        }
        self.free_ranges = merged;
    }
}

impl CpuPool {
    fn new(base: u64, total: u64) -> Self {
        let free_ranges = if total == 0 {
            Vec::new()
        } else {
            vec![(base, total)]
        };
        Self {
            base,
            total,
            used: 0,
            free_ranges,
            allocations: HashMap::new(),
        }
    }

    fn allocate(&mut self, size: u64) -> Option<u64> {
        let size = snap_up(size, 4096);
        for idx in 0..self.free_ranges.len() {
            let (start, range_size) = self.free_ranges[idx];
            let aligned = snap_up(start, 4096);
            let offset = aligned.checked_sub(start)?;
            if offset + size > range_size {
                continue;
            }
            if offset == 0 {
                if size == range_size {
                    self.free_ranges.remove(idx);
                } else {
                    self.free_ranges[idx] = (start + size, range_size - size);
                }
            } else if offset + size == range_size {
                self.free_ranges[idx] = (start, offset);
            } else {
                self.free_ranges[idx] = (start, offset);
                self.free_ranges
                    .insert(idx + 1, (aligned + size, range_size - offset - size));
            }
            self.used += size;
            return Some(aligned);
        }
        None
    }

    fn free(&mut self, addr: u64, size: u64) {
        if addr == 0 || size == 0 {
            return;
        }
        let size = snap_up(size, 4096);
        self.used = self.used.saturating_sub(size);
        self.free_ranges.push((addr, size));
        self.free_ranges.sort_by_key(|(start, _)| *start);
    }

    fn track(&mut self, block_id: u64, addr: u64) {
        self.allocations.insert(block_id, addr);
    }

    fn untrack(&mut self, block_id: u64) -> Option<u64> {
        self.allocations.remove(&block_id)
    }

    fn get(&self, block_id: u64) -> Option<u64> {
        self.allocations.get(&block_id).copied()
    }
}

fn snap_up(val: u64, align: u64) -> u64 {
    if align == 0 {
        return val;
    }
    (val + align - 1) & !(align - 1)
}
