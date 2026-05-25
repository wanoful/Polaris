// SPDX-License-Identifier: GPL-2.0

// POLARIS: Paged Operating Layer for Accelerated Routing and Inference Systems
//
// This kernel module provides OS-level paged KV Cache management for LLM inference.
// It maintains the authoritative block table, manages sessions, and issues GPU memory
// management decisions to the userspace daemon (polarisd) via an ioctl-based
// decision protocol.
//
// All state is global: every open("/dev/polaris") shares the same block table,
// session table, GPU registry, and decision queue. This is the fundamental
// value of the kernel module — cross-process visibility.

mod polaris_types;
mod polaris_policy;

use core::pin::Pin;

use kernel::{
    bindings,
    c_str,
    device::Device,
    fs::File,
    ioctl::_IOC_SIZE,
    miscdevice::{MiscDevice, MiscDeviceOptions, MiscDeviceRegistration},
    prelude::*,
    sync::{
        aref::ARef,
        atomic::{Atomic, Relaxed},
    },
    uaccess::UserSlice,
};

use polaris_types::*;

// Return values for the C-callable NVIDIA UVM fault hook.
const POLARIS_UVM_FAULT_NOT_MINE: i32 = 0;
const POLARIS_UVM_FAULT_HANDLED: i32 = 1;
const POLARIS_UVM_FAULT_ERROR: i32 = -1;

#[derive(PartialEq)]
enum PolarisUvmFaultResult {
    NotMine,
    Handled,
    Error,
}

// ─── Global shared state ────────────────────────────────────────────────────

pub(crate) struct PolarisInner {
    next_block_id: u64,
    next_session_id: u64,
    next_decision_id: u64,
    next_fault_id: u64,
    fault_generation: u64,
    daemon_attached: u32,
    gpus: KVec<PolarisGpu>,
    blocks: KVec<PolarisBlock>,
    sessions: KVec<PolarisSession>,
    pending_decisions: KVec<PolarisDecision>,
    pending_faults: KVec<PolarisFault>,
    eviction_policy: PolarisEvictionPolicy,
    offload_count: u64,
    reload_count: u64,
    total_evictions: u64,
    cow_break_count: u64,
    cow_copy_bytes: u64,
}

// Global state protected by a kernel mutex.  Wrapped in Option because
// KVec cannot be const-constructed; the real state is installed at module
// init time and all handlers unwrap it.
kernel::sync::global_lock! {
    // SAFETY: Initialized in module init before any /dev/polaris open.
    unsafe(uninit) static POLARIS_STATE: Mutex<Option<PolarisInner>> = None;
}

// Set to 1 when the module begins its exit path.  Used by PolarisDevice's
// PinnedDrop to skip module_put during forced unload (rmmod -f) — the
// kernel has already zeroed the refcount, so calling module_put again
// would trigger BUG().
static MODULE_EXITING: Atomic<u32> = Atomic::new(0);

/// C-callable hook for patched NVIDIA UVM replayable GPU page faults.
#[no_mangle]
pub extern "C" fn polaris_uvm_handle_gpu_fault(
    gpu_id: u32,
    fault_address: u64,
    access_type: u32,
) -> i32 {
    match polaris_resolve_gpu_fault(gpu_id, fault_address, access_type) {
        Ok(PolarisUvmFaultResult::NotMine) => POLARIS_UVM_FAULT_NOT_MINE,
        Ok(PolarisUvmFaultResult::Handled) => POLARIS_UVM_FAULT_HANDLED,
        Ok(PolarisUvmFaultResult::Error) | Err(_) => POLARIS_UVM_FAULT_ERROR,
    }
}

fn polaris_resolve_gpu_fault(
    gpu_id: u32,
    fault_address: u64,
    access_type: u32,
) -> Result<PolarisUvmFaultResult> {
    let mut guard = POLARIS_STATE.lock();
    let inner = guard.as_mut().ok_or(ENODEV)?;

    let gpu = match inner.gpus.iter().find(|g| g.gpu_id == gpu_id) {
        Some(gpu) => gpu,
        None => return Ok(PolarisUvmFaultResult::NotMine),
    };
    if !gpu.va_range_registered {
        return Ok(PolarisUvmFaultResult::NotMine);
    }
    let va_range_end = gpu.va_range_base.saturating_add(gpu.va_range_length);
    if fault_address < gpu.va_range_base || fault_address >= va_range_end {
        return Ok(PolarisUvmFaultResult::NotMine);
    }

    // Search from the end so that newly-created pending blocks are
    // matched before older resident blocks that share the same VA
    // (COW break creates a CowPending block at the same VA as the
    // original shared block).
    let block_idx = match inner.blocks.iter()
        .enumerate()
        .rev()
        .find(|(_, b)| {
            b.home_gpu == gpu_id
                && fault_address >= b.gpu_vaddr
                && fault_address < b.gpu_vaddr.saturating_add(b.size_bytes)
        })
    {
        Some((idx, _)) => idx,
        None => return Ok(PolarisUvmFaultResult::Error),
    };
    let fault_id = inner.next_fault_id;
    inner.next_fault_id += 1;
    let generation = inner.fault_generation;
    let op = match inner.blocks[block_idx].state {
        PolarisBlockState::Unmapped => PolarisDecisionOp::Alloc,
        PolarisBlockState::CpuOffloaded => PolarisDecisionOp::Reload,
        PolarisBlockState::CowPending => PolarisDecisionOp::CowBreak,
        PolarisBlockState::Resident => PolarisDecisionOp::MapExisting,
        _ => PolarisDecisionOp::Alloc,
    };
    let decision_id = inner.next_decision_id;
    inner.next_decision_id += 1;
    let (
        block_id,
        session_id,
        src_handle,
        dst_vaddr,
        size_bytes,
        cpu_addr,
        timeout_ms,
    );
    {
        let block = &mut inner.blocks[block_idx];
        block.pending_fault_id = fault_id;
        block.pending_generation = generation;
        block.pending_decision_id = decision_id;
        block.state = match op {
            PolarisDecisionOp::Reload => PolarisBlockState::ReloadPending,
            PolarisDecisionOp::CowBreak => PolarisBlockState::CowPending,
            PolarisDecisionOp::MapExisting => PolarisBlockState::Resident,
            _ => PolarisBlockState::AllocPending,
        };
        block.fault_timeout_ms = POLARIS_DEFAULT_FAULT_TIMEOUT_MS;
        block_id = block.block_id;
        session_id = block.session_id;
        src_handle = if block.state == PolarisBlockState::CowPending {
            block.cow_src_handle
        } else {
            block.gpu_phys_handle
        };
        dst_vaddr = block.gpu_vaddr;
        size_bytes = block.size_bytes;
        cpu_addr = block.cpu_buf_addr;
        timeout_ms = block.fault_timeout_ms;
    }
    inner.pending_faults.push(
        PolarisFault {
            fault_id,
            generation,
            gpu_id,
            fault_address,
            block_id,
            access_type,
            state: 0,
            enqueue_ns: unsafe { bindings::ktime_get_mono_fast_ns() },
            deadline_ns: 0,
            resolved_ns: 0,
        },
        GFP_KERNEL,
    )?;
    inner.pending_decisions.push(
        PolarisDecision {
            decision_id,
            fault_id,
            generation,
            op: op as u32,
            gpu_id,
            block_id,
            session_id,
            src_handle,
            dst_handle: 0,
            src_vaddr: fault_address,
            dst_vaddr: if op == PolarisDecisionOp::CowBreak { 0 } else { dst_vaddr },
            size_bytes,
            cpu_addr,
            access_flags: access_type,
            timeout_ms,
            _reserved: [0u64; 4],
        },
        GFP_KERNEL,
    )?;
    let mut comp: bindings::completion = unsafe { core::mem::zeroed() };
    unsafe { bindings::init_completion(&raw mut comp); }
    inner.blocks[block_idx].completion_ptr = &raw mut comp;
    drop(guard);

    let wait_ret = unsafe {
        bindings::wait_for_completion_interruptible_timeout(
            &raw mut comp,
            bindings::__msecs_to_jiffies(POLARIS_DEFAULT_FAULT_TIMEOUT_MS),
        )
    };
    let mut guard = POLARIS_STATE.lock();
    let inner = guard.as_mut().ok_or(ENODEV)?;
    if let Some(block) = inner.blocks.iter_mut().find(|b| b.block_id == block_id) {
        block.completion_ptr = core::ptr::null_mut();
        if wait_ret == 0 && block.state != PolarisBlockState::Resident {
            block.state = PolarisBlockState::Evicted;
            return Err(ETIMEDOUT);
        }
    }
    Ok(PolarisUvmFaultResult::Handled)
}

// ─── sysfs buffer writer ────────────────────────────────────────────────────

/// Minimal `core::fmt::Write` impl over a fixed u8 buffer, for use in
/// sysfs show functions where we have a `char *buf` from the kernel.
struct BufWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> BufWriter<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
}

impl core::fmt::Write for BufWriter<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let room = self.buf.len().saturating_sub(1).saturating_sub(self.pos);
        let n = bytes.len().min(room);
        self.buf[self.pos..self.pos + n].copy_from_slice(&bytes[..n]);
        self.pos += n;
        Ok(())
    }
}

// ─── sysfs stats show function ──────────────────────────────────────────────

// Null-terminated "stats" as a byte array so the pointer can be used
// in a static. Using c_str!("stats").as_char_ptr() directly in a static
// may not work if as_char_ptr is not const fn.
const STATS_NAME_BYTES: &[u8] = b"stats\0";

// Newtype wrapper so we can safely mark the kobj_attribute as Sync.
// The attribute is write-once (at module init) then read-only forever;
// all its fields (function pointers, name pointer) are immutable.
// `unsafe impl Sync` is the standard kernel-Rust pattern for C structs
// that are only accessed under the kernel's concurrency guarantees.
struct PolarisStatsAttr(bindings::kobj_attribute);
unsafe impl Sync for PolarisStatsAttr {}

static POLARIS_STATS_ATTR: PolarisStatsAttr = PolarisStatsAttr(bindings::kobj_attribute {
    attr: bindings::attribute {
        name: STATS_NAME_BYTES.as_ptr() as *const kernel::ffi::c_char,
        mode: 0o444,
    },
    show: Some(polaris_stats_show),
    store: None,
});

unsafe extern "C" fn polaris_stats_show(
    _kobj: *mut bindings::kobject,
    _attr: *mut bindings::kobj_attribute,
    buf: *mut kernel::ffi::c_char,
) -> isize {
    let guard = POLARIS_STATE.lock();
    let inner = match guard.as_ref() {
        Some(i) => i,
        None => {
            // Module not fully initialized yet.
            return 0;
        }
    };

    // Count block states.
    let (mut resident, mut offloaded, mut evicted, mut pending) = (0u32, 0u32, 0u32, 0u32);
    let (mut shared, mut private) = (0u64, 0u64);
    let mut memory_saved: u64 = 0;
    for b in &inner.blocks {
        match b.state {
            PolarisBlockState::Resident => resident += 1,
            PolarisBlockState::CpuOffloaded => offloaded += 1,
            PolarisBlockState::Evicted => evicted += 1,
            _ => pending += 1,
        }
        if b.flags.contains(PolarisBlockFlag::Shared) {
            shared += b.size_bytes;
            memory_saved = memory_saved.saturating_add(
                b.refcount.saturating_sub(1).saturating_mul(b.size_bytes),
            );
        } else {
            private += b.size_bytes;
        }
    }

    let (mut total_gpu, mut used_gpu, mut cpu_total, mut cpu_used, mut unhealthy_gpus) =
        (0u64, 0u64, 0u64, 0u64, 0u32);
    for g in &inner.gpus {
        total_gpu += g.total_bytes;
        used_gpu += g.used_bytes;
        cpu_total += g.cpu_pool_total_bytes;
        cpu_used += g.cpu_pool_used_bytes;
        if !g.healthy {
            unhealthy_gpus += 1;
        }
    }

    let daemon = inner.daemon_attached;
    let sessions = inner.sessions.len();
    let blocks = inner.blocks.len();
    let gpus = inner.gpus.len();
    let decisions = inner.pending_decisions.len();
    let policy: u32 = inner.eviction_policy as u32;
    let offload_cnt = inner.offload_count;
    let reload_cnt = inner.reload_count;
    let evictions = inner.total_evictions;
    let cow_cnt = inner.cow_break_count;
    let cow_bytes = inner.cow_copy_bytes;
    let memory_saved_naive = memory_saved.saturating_sub(cow_bytes);
    drop(guard);

    // Write into the kernel-provided buffer (typically PAGE_SIZE = 4096).
    // SAFETY: buf points to a valid kernel buffer of at least PAGE_SIZE bytes.
    let buf_slice = unsafe { core::slice::from_raw_parts_mut(buf as *mut u8, 4096) };
    let len = {
        let mut w = BufWriter::new(buf_slice);
        let _ = core::fmt::write(
            &mut w,
            format_args!(
                "\
sessions:       {sessions}
blocks:         {blocks}
  resident:     {resident}
  offloaded:    {offloaded}
  evicted:      {evicted}
  pending:      {pending}
gpus:           {gpus}
  unhealthy:    {unhealthy}
daemon:         {daemon}
policy:         {policy} (0=fifo,1=lru,2=phase_aware)
offloads:       {offload_cnt}
reloads:        {reload_cnt}
evictions:      {evictions}
cow_breaks:     {cow_cnt}
cow_copy_mib:   {cow_copy_mib}
cow_saved_mib:  {saved_mib}
gpu_total_mib:  {gpu_total_mib}
gpu_used_mib:   {gpu_used_mib}
cpu_pool_mib:   {cpu_pool_mib}
cpu_used_mib:   {cpu_used_mib}
shared_mib:     {shared_mib}
private_mib:    {private_mib}
pending_decs:   {decisions}
",
                sessions = sessions,
                blocks = blocks,
                resident = resident,
                offloaded = offloaded,
                evicted = evicted,
                pending = pending,
                gpus = gpus,
                unhealthy = unhealthy_gpus,
                daemon = daemon,
                policy = policy,
                offload_cnt = offload_cnt,
                reload_cnt = reload_cnt,
                evictions = evictions,
                cow_cnt = cow_cnt,
                cow_copy_mib = cow_bytes / (1024 * 1024),
                saved_mib = memory_saved_naive / (1024 * 1024),
                gpu_total_mib = total_gpu / (1024 * 1024),
                gpu_used_mib = used_gpu / (1024 * 1024),
                cpu_pool_mib = cpu_total / (1024 * 1024),
                cpu_used_mib = cpu_used / (1024 * 1024),
                shared_mib = shared / (1024 * 1024),
                private_mib = private / (1024 * 1024),
                decisions = decisions,
            ),
        );
        w.pos
    };
    buf_slice[len] = 0; // null-terminate
    len as isize
}

// ─── Module declaration ─────────────────────────────────────────────────────

module! {
    type: PolarisModule,
    name: "polaris",
    authors: ["POLARIS Team"],
    description: "OS-level Paged KV Cache Management for LLM Inference",
    license: "GPL",
}

/// Create `/sys/kernel/polaris/stats` and return the kobject pointer.
fn init_polaris_sysfs() -> Result<*mut bindings::kobject> {
    // SAFETY: kernel_kobj is always valid. This is called only during module init.
    let polaris_kobj = unsafe {
        bindings::kobject_create_and_add(
            c_str!("polaris").as_char_ptr(),
            bindings::kernel_kobj,
        )
    };
    if polaris_kobj.is_null() {
        pr_err!("POLARIS: failed to create /sys/kernel/polaris kobject\n");
        return Err(ENOMEM);
    }

    // SAFETY: polaris_kobj is valid; POLARIS_STATS_ATTR is a static with
    // 'static lifetime (matches the kobject's lifetime).
    let ret = unsafe {
        bindings::sysfs_create_file_ns(
            polaris_kobj,
            &raw const POLARIS_STATS_ATTR.0.attr as *const bindings::attribute,
            core::ptr::null(),
        )
    };
    if ret != 0 {
        pr_err!("POLARIS: failed to create sysfs stats attribute (err {ret})\n");
        // SAFETY: polaris_kobj was just created above.
        unsafe { bindings::kobject_put(polaris_kobj) };
        return Err(ENOMEM);
    }

    pr_info!("POLARIS: /sys/kernel/polaris/stats created\n");
    Ok(polaris_kobj)
}

#[pin_data(PinnedDrop)]
struct PolarisModule {
    #[pin]
    _miscdev: MiscDeviceRegistration<PolarisDevice>,
    /// sysfs kobject created under /sys/kernel/polaris
    polaris_kobj: *mut bindings::kobject,
}

// The raw pointer `polaris_kobj` is only touched during module init/exit
// (single-threaded, before/after the module is live).  After init it is
// read-only until exit.
unsafe impl Send for PolarisModule {}
unsafe impl Sync for PolarisModule {}

impl kernel::InPlaceModule for PolarisModule {
    fn init(_module: &'static ThisModule) -> impl PinInit<Self, Error> {
        pr_info!("POLARIS: initializing kernel module\n");

        // Initialize the C mutex backing the global lock.
        // SAFETY: Called exactly once at module init, before any use.
        unsafe { POLARIS_STATE.init() };

        // Install the real shared state.
        {
            let mut guard = POLARIS_STATE.lock();
            *guard = Some(PolarisInner {
                next_block_id: 1,
                next_session_id: 1,
                next_decision_id: 1,
                next_fault_id: 1,
                fault_generation: 1,
                daemon_attached: 0,
                gpus: KVec::new(),
                blocks: KVec::new(),
                sessions: KVec::new(),
                pending_decisions: KVec::new(),
                pending_faults: KVec::new(),
                eviction_policy: PolarisEvictionPolicy::Fifo,
                offload_count: 0,
                reload_count: 0,
                total_evictions: 0,
                cow_break_count: 0,
                cow_copy_bytes: 0,
            });
        }

        let options = MiscDeviceOptions {
            name: c_str!("polaris"),
        };

        try_pin_init!(Self {
            _miscdev <- MiscDeviceRegistration::<PolarisDevice>::register(options),
            polaris_kobj: init_polaris_sysfs()?,
        })
    }
}

#[pinned_drop]
impl PinnedDrop for PolarisModule {
    fn drop(self: Pin<&mut Self>) {
        // Mark that we're in the module exit path.  PolarisDevice drops
        // that run as a side-effect of _miscdev being dropped during
        // forced unload (rmmod -f) will see this and skip module_put.
        MODULE_EXITING.store(1, Relaxed);

        if !self.polaris_kobj.is_null() {
            // SAFETY: The kobject was created in init and is valid.
            // Order matters: remove files → del kobject from sysfs tree
            // → put reference. kobject_del synchronously waits for all
            // in-flight readers, so after it returns no one can call
            // polaris_stats_show anymore.
            unsafe {
                bindings::sysfs_remove_file_ns(
                    self.polaris_kobj,
                    &raw const POLARIS_STATS_ATTR.0.attr as *const bindings::attribute,
                    core::ptr::null(),
                );
                bindings::kobject_del(self.polaris_kobj);
                bindings::kobject_put(self.polaris_kobj);
            }
        }
    }
}

// ─── Device (per open fd, but all share POLARIS_STATE) ──────────────────────

#[pin_data(PinnedDrop)]
struct PolarisDevice {
    dev: ARef<Device>,
    /// Whether this fd registered a GPU (belongs to the daemon).
    registered_gpu: Atomic<u32>,
}

#[vtable]
impl MiscDevice for PolarisDevice {
    type Ptr = Pin<KBox<Self>>;

    fn open(_file: &File, misc: &MiscDeviceRegistration<Self>) -> Result<Pin<KBox<Self>>> {
        let dev = ARef::from(misc.device());
        dev_info!(dev, "POLARIS: device opened\n");

        let ptr = KBox::try_pin_init(
            try_pin_init! {
                PolarisDevice {
                    dev: dev,
                    registered_gpu: Atomic::new(0),
                }
            },
            GFP_KERNEL,
        )?;

        // Keep the module loaded while this fd is open.  The kernel Rust
        // miscdevice vtable cannot yet set fops->owner = THIS_MODULE (the
        // const-eval limitation documented in gen_disk.rs), so we do it
        // by hand.
        // SAFETY: __this_module is valid for our lifetime; we are inside
        // the module's own open handler, so the module is definitely live.
        unsafe {
            bindings::__module_get(
                core::ptr::addr_of!(bindings::__this_module) as *mut bindings::module,
            );
        }

        Ok(ptr)
    }

    fn ioctl(me: Pin<&PolarisDevice>, _file: &File, cmd: u32, arg: usize) -> Result<isize> {
        let user_ptr = UserPtr::from_addr(arg);
        let size = _IOC_SIZE(cmd);

        // All handlers share the same global state via POLARIS_STATE.
        match cmd {
            POLARIS_REGISTER_GPU => me.handle_register_gpu(user_ptr, size),
            POLARIS_REGISTER_VA_RANGE => me.handle_register_va_range(user_ptr, size),
            POLARIS_SESSION_CREATE => me.handle_session_create(user_ptr, size),
            POLARIS_SESSION_DESTROY => me.handle_session_destroy(user_ptr, size),
            POLARIS_SESSION_GET_STATS => me.handle_session_get_stats(user_ptr, size),
            POLARIS_SESSION_BRANCH => me.handle_session_branch(user_ptr, size),
            POLARIS_BLOCK_RESERVE => me.handle_block_reserve(user_ptr, size),
            POLARIS_BLOCK_RELEASE => me.handle_block_release(user_ptr, size),
            POLARIS_BLOCK_TOUCH => me.handle_block_touch(user_ptr, size),
            POLARIS_BLOCK_GET_STATE => me.handle_block_get_state(user_ptr, size),
            POLARIS_GET_DECISION => me.handle_get_decision(user_ptr, size),
            POLARIS_COMPLETE_OPERATION => me.handle_complete_operation(user_ptr, size),
            POLARIS_GET_GLOBAL_STATS => me.handle_get_global_stats(user_ptr, size),
            POLARIS_LIST_SESSIONS => me.handle_list_sessions(user_ptr, size),
            POLARIS_SET_POLICY => me.handle_set_policy(user_ptr, size),
            _ => {
                dev_err!(me.dev, "POLARIS: unknown ioctl 0x{:x}\n", cmd);
                Err(ENOTTY)
            }
        }
    }
}

    #[pinned_drop]
impl PinnedDrop for PolarisDevice {
    fn drop(self: Pin<&mut Self>) {
        if self.registered_gpu.load(Relaxed) != 0 {
            dev_info!(self.dev, "POLARIS: daemon disconnected, evicting pending blocks\n");
            let mut guard = POLARIS_STATE.lock();
            if let Some(inner) = guard.as_mut() {
                inner.daemon_attached = inner.daemon_attached.saturating_sub(1);
                for block in inner.blocks.iter_mut() {
                    match block.state {
                        PolarisBlockState::AllocPending
                        | PolarisBlockState::OffloadPending
                        | PolarisBlockState::ReloadPending
                        | PolarisBlockState::CowPending
                        | PolarisBlockState::FreePending => {
                            block.state = PolarisBlockState::Evicted;
                            block.pending_decision_id = 0;
                            // Wake any bounded fault waiter.
                            let comp_ptr = block.completion_ptr;
                            block.completion_ptr = core::ptr::null_mut();
                            if !comp_ptr.is_null() {
                                unsafe { bindings::complete(comp_ptr); }
                            }
                        }
                        _ => {}
                    }
                }
                inner.pending_decisions.clear();
            }
        }
        dev_info!(self.dev, "POLARIS: device closed\n");

        // Release the module reference taken in open().  Skip during
        // force-unload (MODULE_EXITING is set before _miscdev is dropped
        // and triggers this path) — the kernel already zeroed the refcount
        // and calling module_put would BUG().
        if MODULE_EXITING.load(Relaxed) == 0 {
            // SAFETY: __this_module is valid; we took a reference in open().
            unsafe {
                bindings::module_put(
                    core::ptr::addr_of!(bindings::__this_module) as *mut bindings::module,
                );
            }
        }
    }
}

// ─── IOCTL handler implementations ──────────────────────────────────────────

impl PolarisDevice {
    fn handle_register_gpu(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisRegisterGpuArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        // Idempotent: if this GPU ID is already registered, update its
        // parameters instead of creating a duplicate entry.
        if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == arg.gpu_id) {
            gpu.total_bytes = arg.total_bytes;
            gpu.budget_bytes = arg.budget_bytes;
            gpu.cpu_pool_total_bytes = arg.cpu_pool_bytes;
            gpu.healthy = true;
            dev_info!(self.dev, "POLARIS: GPU {} re-registered\n", arg.gpu_id);
        } else {
            inner.gpus.push(
                PolarisGpu {
                    gpu_id: arg.gpu_id,
                    gpu_uuid: [0u8; 16],
                    total_bytes: arg.total_bytes,
                    used_bytes: 0,
                    budget_bytes: arg.budget_bytes,
                    pressure_score: 0,
                    cpu_pool_total_bytes: arg.cpu_pool_bytes,
                    cpu_pool_used_bytes: 0,
                    va_range_id: 0,
                    va_range_base: 0,
                    va_range_length: 0,
                    va_block_size: 0,
                    va_range_flags: 0,
                    va_range_registered: false,
                    healthy: true,
                },
                GFP_KERNEL,
            )?;
            dev_info!(self.dev, "POLARIS: GPU {} registered\n", arg.gpu_id);
        }

        self.registered_gpu.store(1, Relaxed);
        inner.daemon_attached += 1;
        Ok(0)
    }

    fn handle_register_va_range(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisRegisterVaRangeArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        if self.registered_gpu.load(Relaxed) == 0 {
            return Err(EPERM);
        }
        if arg.length == 0 || arg.block_size == 0 {
            return Err(EINVAL);
        }

        let gpu = inner.gpus.iter_mut().find(|g| g.gpu_id == arg.gpu_id).ok_or(ENOENT)?;
        if arg.range_id == 0 {
            arg.range_id = ((arg.gpu_id as u64) << 32) | 1;
        }
        gpu.va_range_id = arg.range_id;
        gpu.va_range_base = arg.base;
        gpu.va_range_length = arg.length;
        gpu.va_block_size = arg.block_size;
        gpu.va_range_flags = arg.flags;
        gpu.va_range_registered = true;

        drop(guard);
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_session_create(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisSessionCreateArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        let (gpu_vas_base, gpu_vas_size) =
            if let Some(gpu) = inner.gpus.iter().find(|g| g.gpu_id == arg.home_gpu) {
                if !gpu.healthy {
                dev_err!(self.dev, "POLARIS: GPU {} is unhealthy, rejecting session\n", arg.home_gpu);
                return Err(ENODEV);
            }
                if gpu.va_range_registered {
                    (gpu.va_range_base, gpu.va_range_length)
                } else {
                    (0, arg.gpu_vas_bytes)
                }
            } else {
                (0, arg.gpu_vas_bytes)
            };

        let session_id = inner.next_session_id;
        inner.next_session_id += 1;

        let bpt = if arg.bytes_per_token > 0 {
            arg.bytes_per_token
        } else {
            POLARIS_DEFAULT_BYTES_PER_TOKEN
        };

        let priority = if arg.priority > 0 { arg.priority } else { 5 };
        inner.sessions.push(
            PolarisSession {
                session_id,
                home_gpu: arg.home_gpu,
                gpu_vas_base,
                gpu_vas_size,
                beam_width: arg.beam_width,
                bytes_per_token: bpt,
                parent_session_id: 0,
                priority,
                block_ids: KVec::new(),
            },
            GFP_KERNEL,
        )?;

        arg.session_id = session_id;
        drop(guard); // release lock before writing back to userspace
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        dev_info!(self.dev, "POLARIS: session {} created\n", session_id);
        Ok(0)
    }

    fn handle_session_destroy(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisSessionDestroyArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;
        let sid = arg.session_id;

        if !inner.sessions.iter().any(|s| s.session_id == sid) {
            return Err(ENOENT);
        }

        let gpu_id = inner
            .sessions
            .iter()
            .find(|s| s.session_id == sid)
            .map(|s| s.home_gpu)
            .unwrap_or(0);

        // Collect info before any mutation (avoids double-borrow with
        // pending_decisions.push inside the loop).
        struct ToFree {
            idx: usize,
            block_id: u64,
            phys_handle: u64,
            size_bytes: u64,
            state: PolarisBlockState,
            refcount: u64,
            had_cpu_buf: bool,
        }
        let mut to_free: KVec<ToFree> = KVec::new();
        for idx in 0..inner.blocks.len() {
            let b = &inner.blocks[idx];
            if b.session_id == sid {
                to_free.push(
                    ToFree {
                        idx,
                        block_id: b.block_id,
                        phys_handle: b.gpu_phys_handle,
                        size_bytes: b.size_bytes,
                        state: b.state,
                        refcount: b.refcount,
                        had_cpu_buf: b.cpu_buf_addr != 0,
                    },
                    GFP_KERNEL,
                )?;
            }
        }

        // Second pass: for COW child sessions, the block table is scanned
        // by session_id which misses shared blocks (their session_id is the
        // parent's).  Walk the session's block_ids list to find these
        // COW-shared blocks and decrement their refcounts.
        {
            let session = inner.sessions.iter().find(|s| s.session_id == sid);
            if let Some(sess) = session {
                for &bid in &sess.block_ids {
                    // Skip blocks already collected in the first pass.
                    if to_free.iter().any(|tf| tf.block_id == bid) {
                        continue;
                    }
                    if let Some(idx) = inner.blocks.iter().position(|b| b.block_id == bid) {
                        let b = &inner.blocks[idx];
                        to_free.push(
                            ToFree {
                                idx,
                                block_id: b.block_id,
                                phys_handle: b.gpu_phys_handle,
                                size_bytes: b.size_bytes,
                                state: b.state,
                                refcount: b.refcount,
                                had_cpu_buf: b.cpu_buf_addr != 0,
                            },
                            GFP_KERNEL,
                        )?;
                    }
                }
            }
        }

        // Now mutate: queue FREE or clean up directly.
        // G6: if the pending decision queue is full, handle FREE decisions
        // directly (skip daemon FREE).  The blocks' phys handles will be
        // orphaned, but this prevents kernel OOM.  The daemon's CUDA context
        // cleanup on exit handles the orphaned handles.
        let queue_full = inner.pending_decisions.len() >= POLARIS_MAX_PENDING_DECISIONS;
        if queue_full {
            dev_warn!(
                self.dev,
                "POLARIS: pending decision queue full ({}), cleaning up session {} blocks directly\n",
                inner.pending_decisions.len(), sid
            );
        }

        for tf in &mut to_free {
            let block = &mut inner.blocks[tf.idx];
            let phys_handle = tf.phys_handle;
            let block_id = tf.block_id;
            let sz = tf.size_bytes;

            if tf.refcount > 1 {
                block.refcount -= 1;
                if block.refcount == 1 {
                    block.flags = block.flags & !PolarisBlockFlag::Shared;
                }
                dev_info!(
                    self.dev,
                    "POLARIS: session {} destroy: block {} refcount decremented to {}\n",
                    sid, block_id, block.refcount
                );
                continue;
            }

            if !queue_full && inner.daemon_attached > 0 && phys_handle != 0 {
                let dec_id = inner.next_decision_id;
                inner.next_decision_id += 1;

                block.state = PolarisBlockState::FreePending;
                block.pending_decision_id = dec_id;

                inner.pending_decisions.push(
                    PolarisDecision {
                        decision_id: dec_id,
                        fault_id: 0,
                        generation: 0,
                        op: PolarisDecisionOp::Free as u32,
                        gpu_id,
                        block_id,
                        session_id: sid,
                        src_handle: phys_handle,
                        dst_handle: 0,
                        src_vaddr: 0,
                        dst_vaddr: 0,
                        size_bytes: sz,
                        cpu_addr: 0,
                        access_flags: 0,
                        timeout_ms: 0,
                        _reserved: [0u64; 4],
                    },
                    GFP_KERNEL,
                )?;

                dev_info!(
                    self.dev,
                    "POLARIS: session {} destroy: block {} free queued (FREE {})\n",
                    sid, block_id, dec_id
                );
            } else {
                // No daemon or queue full or never mapped — remove directly.
                if tf.state == PolarisBlockState::Resident {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                        gpu.used_bytes = gpu.used_bytes.saturating_sub(sz);
                    }
                } else if tf.had_cpu_buf {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                        gpu.cpu_pool_used_bytes = gpu.cpu_pool_used_bytes.saturating_sub(sz);
                    }
                }
                dev_info!(
                    self.dev,
                    "POLARIS: session {} destroy: block {} freed directly\n",
                    sid, block_id
                );
            }
        }

        // Remove session entry. FreePending blocks stay in the table for
        // the daemon to complete; non-FreePending blocks for this session
        // are removed.
        inner.sessions.retain(|s| s.session_id != sid);
        // Only remove blocks whose refcount has dropped to 0.  COW-shared
        // blocks (refcount > 0 after decrement) must stay in the table for
        // child sessions that still reference them.
        inner.blocks.retain(|b| !(b.session_id == sid && b.state != PolarisBlockState::FreePending && b.refcount == 0));

        dev_info!(self.dev, "POLARIS: session {} destroyed\n", sid);
        Ok(0)
    }

    fn handle_session_get_stats(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisSessionGetStatsArg = reader.read()?;
        let guard = POLARIS_STATE.lock();
        let inner = guard.as_ref().ok_or(ENODEV)?;

        match inner.sessions.iter().find(|s| s.session_id == arg.session_id) {
            Some(session) => {
                arg.home_gpu = session.home_gpu;
                arg.beam_width = session.beam_width;
                arg.num_blocks = session.block_ids.len() as u32;
                arg.total_bytes = session.gpu_vas_size;
            }
            None => return Err(ENOENT),
        }
        drop(guard);

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_session_branch(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisSessionBranchArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        // Extract parent info (immutable borrow) before mutating.
        let parent = inner
            .sessions
            .iter()
            .find(|s| s.session_id == arg.parent_session_id)
            .ok_or(ENOENT)?;
        let parent_gpu = parent.home_gpu;
        let parent_vas_base = parent.gpu_vas_base;
        let parent_vas_size = parent.gpu_vas_size;
        let parent_beam = parent.beam_width;
        let parent_bpt = parent.bytes_per_token;
        let parent_priority = parent.priority;
        let parent_block_ids: KVec<u64> = {
            let mut ids = KVec::new();
            for &bid in &parent.block_ids {
                ids.push(bid, GFP_KERNEL)?;
            }
            ids
        };
        let parent_id = arg.parent_session_id;

        // G4: reject branch if the parent's GPU is unhealthy.
        if let Some(gpu) = inner.gpus.iter().find(|g| g.gpu_id == parent_gpu) {
            if !gpu.healthy {
                dev_err!(self.dev, "POLARIS: GPU {} unhealthy, rejecting SESSION_BRANCH\n", parent_gpu);
                return Err(ENODEV);
            }
        }
        let _ = parent;

        // COW: increment refcount on all parent blocks.
        for &bid in &parent_block_ids {
            if let Some(block) = inner.blocks.iter_mut().find(|b| b.block_id == bid) {
                block.refcount += 1;
                block.flags |= PolarisBlockFlag::Shared;
            }
        }

        let child_id = inner.next_session_id;
        inner.next_session_id += 1;

        inner.sessions.push(
            PolarisSession {
                session_id: child_id,
                home_gpu: parent_gpu,
                gpu_vas_base: parent_vas_base,
                gpu_vas_size: parent_vas_size,
                beam_width: parent_beam,
                bytes_per_token: parent_bpt,
                parent_session_id: parent_id,
                priority: parent_priority,
                block_ids: parent_block_ids,
            },
            GFP_KERNEL,
        )?;

        arg.child_session_id = child_id;
        drop(guard);
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        dev_info!(self.dev, "POLARIS: session {} branched from {}\n", child_id, parent_id);
        Ok(0)
    }

    fn handle_block_reserve(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisBlockReserveArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;
        if inner.daemon_attached == 0 {
            return Err(ENODEV);
        }
        if !inner.sessions.iter().any(|s| s.session_id == arg.session_id) {
            return Err(ENOENT);
        }
        // Extract all fields from the session reference before any
        // mutable borrow of inner.sessions below.
        let sess = inner.sessions.iter().find(|s| s.session_id == arg.session_id).ok_or(ENOENT)?;
        let gpu_vas_base = sess.gpu_vas_base;
        let bytes_per_token = sess.bytes_per_token;
        let home_gpu = sess.home_gpu;
        let gpu_vaddr = gpu_vas_base + ((arg.token_start as u64) * bytes_per_token);
        let size_bytes = (arg.token_count as u64).saturating_mul(bytes_per_token);
        let req_start = arg.token_start as u64;
        let req_end = req_start + arg.token_count as u64;
        let overwrite = arg.flags & POLARIS_RESERVE_FLAG_OVERWRITE != 0;
        let mut existing_block_idx: Option<usize> = None;

        // Collect all block IDs accessible to this session (directly
        // owned + COW-shared from parent).
        let mut session_bids: KVec<u64> = KVec::new();
        for &bid in &sess.block_ids {
            session_bids.push(bid, GFP_KERNEL)?;
        }

        // ── Phase-2 COW logic ──────────────────────────────────────────
        // Two-phase overlap detection:
        //   Pass 1: directly-owned blocks (session_id matches)
        //   Pass 2: COW-shared blocks (in session block_ids but
        //            different session_id)
        // This ordering ensures that a post-COW-break private block
        // (refcount==1) is found before the original shared block.

        // Pass 1: directly-owned blocks.
        for (i, b) in inner.blocks.iter().enumerate() {
            if b.session_id == arg.session_id {
                let b_start = b.token_start as u64;
                let b_end = b_start + b.token_count as u64;
                if req_start < b_end && req_end > b_start {
                    if !overwrite {
                        return Err(EINVAL);
                    }
                    existing_block_idx = Some(i);
                    break;
                }
            }
        }
        if existing_block_idx.is_none() {
            // Pass 2: COW-shared blocks (different session_id, but in this
            // session's block_ids list).
            for (i, b) in inner.blocks.iter().enumerate() {
                if session_bids.iter().any(|&bid| bid == b.block_id) {
                    let b_start = b.token_start as u64;
                    let b_end = b_start + b.token_count as u64;
                    if req_start < b_end && req_end > b_start {
                        if !overwrite {
                            return Err(EINVAL);
                        }
                        existing_block_idx = Some(i);
                        break;
                    }
                }
            }
        }

        if let Some(eb_idx) = existing_block_idx {
            let eb = &mut inner.blocks[eb_idx];
            if eb.refcount > 1 {
                // COW break: create a private copy.
                let block_id = inner.next_block_id;
                inner.next_block_id += 1;
                let cow_src = eb.gpu_phys_handle;
                eb.refcount -= 1;
                if eb.refcount == 1 {
                    eb.flags = eb.flags & !PolarisBlockFlag::Shared;
                }
                inner.blocks.push(
                    PolarisBlock {
                        block_id,
                        session_id: arg.session_id,
                        token_start: arg.token_start,
                        token_count: arg.token_count,
                        home_gpu,
                        gpu_vaddr,
                        gpu_phys_handle: 0,
                        cpu_buf_addr: 0,
                        size_bytes,
                        refcount: 1,
                        state: PolarisBlockState::CowPending,
                        flags: PolarisBlockFlags::empty(),
                        phase: if arg.phase == PolarisPhase::Decode as u32 {
                            PolarisPhase::Decode
                        } else {
                            PolarisPhase::Prefill
                        },
                        last_touch_ns: 0,
                        map_time_ns: 0,
                        cow_src_handle: cow_src,
                        retry_count: 0,
                        pending_decision_id: 0,
                        pending_fault_id: 0,
                        pending_generation: 0,
                        fault_timeout_ms: POLARIS_DEFAULT_FAULT_TIMEOUT_MS,
                        completion_ptr: core::ptr::null_mut(),
                    },
                    GFP_KERNEL,
                )?;
                if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == arg.session_id) {
                    session.block_ids.push(block_id, GFP_KERNEL)?;
                }
                arg.block_id = block_id;
                arg.gpu_vaddr = gpu_vaddr;
                drop(guard);
                let mut writer = UserSlice::new(user_ptr, size).writer();
                writer.write(&arg)?;

                let fault_result = polaris_resolve_gpu_fault(
                    home_gpu,
                    gpu_vaddr,
                    1, // access_type: write (COW break)
                )?;
                if fault_result != PolarisUvmFaultResult::Handled {
                    return Err(EIO);
                }
                return Ok(0);
            }
            // refcount == 1: in-place overwrite — return existing block.
            eb.token_start = arg.token_start;
            eb.token_count = arg.token_count;
            eb.size_bytes = size_bytes;
            eb.gpu_vaddr = gpu_vaddr;
            arg.block_id = eb.block_id;
            arg.gpu_vaddr = gpu_vaddr;
            drop(guard);
            let mut writer = UserSlice::new(user_ptr, size).writer();
            writer.write(&arg)?;
            return Ok(0);
        }

        // No overlap: create a fresh block (normal path).
        let block_id = inner.next_block_id;
        inner.next_block_id += 1;
        inner.blocks.push(
            PolarisBlock {
                block_id,
                session_id: arg.session_id,
                token_start: arg.token_start,
                token_count: arg.token_count,
                home_gpu,
                gpu_vaddr,
                gpu_phys_handle: 0,
                cpu_buf_addr: 0,
                size_bytes,
                refcount: 1,
                state: PolarisBlockState::Unmapped,
                flags: PolarisBlockFlags::empty(),
                phase: if arg.phase == PolarisPhase::Decode as u32 {
                    PolarisPhase::Decode
                } else {
                    PolarisPhase::Prefill
                },
                last_touch_ns: 0,
                map_time_ns: 0,
                cow_src_handle: 0,
                retry_count: 0,
                pending_decision_id: 0,
                pending_fault_id: 0,
                pending_generation: 0,
                fault_timeout_ms: POLARIS_DEFAULT_FAULT_TIMEOUT_MS,
                completion_ptr: core::ptr::null_mut(),
            },
            GFP_KERNEL,
        )?;
        if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == arg.session_id) {
            session.block_ids.push(block_id, GFP_KERNEL)?;
        }
        arg.block_id = block_id;
        arg.gpu_vaddr = gpu_vaddr;
        drop(guard);
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;

        // Simulate the GPU first-touch page fault that would normally be
        // delivered through the NVIDIA UVM replayable-fault hook.  This
        // exercises the full kernel→daemon→kernel decision protocol
        // (ALLOC + cuMemMap) without requiring a real CUDA kernel launch.
        // In production the UVM hook calls polaris_resolve_gpu_fault
        // directly; the synthetic path calls the same function so the
        // daemon-side execution is identical.
        let fault_result = polaris_resolve_gpu_fault(
            home_gpu,
            gpu_vaddr,
            0,           // access_type: 0 = read
        )?;
        if fault_result != PolarisUvmFaultResult::Handled {
            return Err(EIO);
        }

        Ok(0)
    }

    fn handle_block_release(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisBlockReleaseArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;
        let idx = inner.blocks.iter().position(|b| {
            b.session_id == arg.session_id
                && b.token_start == arg.token_start
                && b.token_count == arg.token_count
        }).ok_or(ENOENT)?;
        let block_id = inner.blocks[idx].block_id;
        if inner.blocks[idx].refcount > 1 {
            inner.blocks[idx].refcount -= 1;
            if inner.blocks[idx].refcount == 1 {
                inner.blocks[idx].flags = inner.blocks[idx].flags & !PolarisBlockFlag::Shared;
            }
        } else {
            let _ = inner.blocks.remove(idx);
        }
        if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == arg.session_id) {
            session.block_ids.retain(|bid| *bid != block_id);
        }
        Ok(0)
    }

    fn polaris_handle_gpu_fault(
        &self,
        gpu_id: u32,
        fault_address: u64,
        access_type: u32,
    ) -> Result<isize> {
        match polaris_resolve_gpu_fault(gpu_id, fault_address, access_type)? {
            PolarisUvmFaultResult::Handled => Ok(0),
            PolarisUvmFaultResult::NotMine => Err(ENOENT),
            PolarisUvmFaultResult::Error => Err(EIO),
        }
    }

    fn handle_block_touch(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisBlockTouchArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;
        let now = unsafe { bindings::ktime_get_mono_fast_ns() };
        let touch_end = arg.token_start + arg.token_count;

        // Collect block IDs to touch (covers both directly-owned and COW-shared blocks).
        let mut bids_to_touch: KVec<u64> = KVec::new();
        if let Some(sess) = inner.sessions.iter().find(|s| s.session_id == arg.session_id) {
            for &bid in &sess.block_ids {
                bids_to_touch.push(bid, GFP_KERNEL)?;
            }
        }
        // Also scan by session_id for directly-owned blocks not yet in block_ids list.
        for block in inner.blocks.iter() {
            if block.session_id == arg.session_id {
                let start = block.token_start as u64;
                let end = start + block.token_count as u64;
                if start < touch_end && end > arg.token_start {
                    if !bids_to_touch.iter().any(|&bid| bid == block.block_id) {
                        bids_to_touch.push(block.block_id, GFP_KERNEL)?;
                    }
                }
            }
        }

        for bid in &bids_to_touch {
            if let Some(block) = inner.blocks.iter_mut().find(|b| b.block_id == *bid) {
                let start = block.token_start as u64;
                let end = start + block.token_count as u64;
                if start < touch_end && end > arg.token_start {
                    block.last_touch_ns = now;
                }
            }
        }
        Ok(0)
    }

    fn handle_block_get_state(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisBlockGetStateArg = reader.read()?;
        let guard = POLARIS_STATE.lock();
        let inner = guard.as_ref().ok_or(ENODEV)?;

        // Look up by session_id and token_start.  For COW child sessions the
        // block's session_id is the parent's, so fall back to the session's
        // block_ids list.
        let maybe_block = {
            let direct = inner
                .blocks
                .iter()
                .find(|b| b.session_id == arg.session_id && b.token_start == arg.token_start);
            if direct.is_some() {
                direct
            } else {
                let session = inner.sessions.iter().find(|s| s.session_id == arg.session_id);
                session.and_then(|sess| {
                    sess.block_ids.iter().find_map(|&bid| {
                        inner.blocks.iter()
                            .find(|b| b.block_id == bid && b.token_start == arg.token_start)
                    })
                })
            }
        };

        match maybe_block {
            Some(block) => {
                arg.block_id = block.block_id;
                arg.state = block.state as u32;
                arg.refcount = block.refcount;
                arg.gpu_vaddr = block.gpu_vaddr;
            }
            None => return Err(ENOENT),
        }
        drop(guard);

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_get_decision(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisGetDecisionArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        // Gate: return empty list when no daemon is attached.
        if inner.daemon_attached == 0 {
            arg.count = 0;
            drop(guard);
            let mut writer = UserSlice::new(user_ptr, size).writer();
            writer.write(&arg)?;
            return Ok(0);
        }

        let count = core::cmp::min(inner.pending_decisions.len(), POLARIS_MAX_DECISIONS_PER_POLL);
        arg.count = count as u32;

        for i in 0..count {
            arg.decisions[i] = inner.pending_decisions[i];
        }
        if count < inner.pending_decisions.len() {
            let remaining = inner.pending_decisions.len() - count;
            for i in 0..remaining {
                inner.pending_decisions[i] = inner.pending_decisions[count + i];
            }
            inner.pending_decisions.truncate(remaining);
        } else {
            inner.pending_decisions.clear();
        }

        let now = unsafe { bindings::ktime_get_mono_fast_ns() };
        for i in 0..count {
            let block_id = arg.decisions[i].block_id;
            if let Some(block) = inner.blocks.iter_mut().find(|b| b.block_id == block_id) {
                block.last_touch_ns = now;
            }
        }

        drop(guard);
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_complete_operation(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisCompleteOperationArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        // Find the block tied to this decision by pending_decision_id.
        let block_idx = match inner
            .blocks
            .iter()
            .position(|b| b.pending_decision_id == arg.decision_id)
        {
            Some(idx) => idx,
            None => {
                dev_err!(
                    self.dev,
                    "POLARIS: COMPLETE_OPERATION for unknown decision_id {}\n",
                    arg.decision_id
                );
                return Ok(0);
            }
        };

        if inner.blocks[block_idx].pending_generation != 0
            && inner.blocks[block_idx].pending_generation != arg.generation
        {
            dev_warn!(
                self.dev,
                "POLARIS: stale COMPLETE_OPERATION decision={} generation={} active={}\n",
                arg.decision_id,
                arg.generation,
                inner.blocks[block_idx].pending_generation
            );
            return Ok(0);
        }

            // ── Success path ──
        if arg.result == 0 {
            let prev_state;
            let gpu_id;
            let sz;
            let comp_ptr: *mut bindings::completion;
            let was_cpu_offloaded: bool;
            {
                let block = &mut inner.blocks[block_idx];
                block.retry_count = 0;
                block.pending_decision_id = 0;
                block.pending_fault_id = 0;
                block.pending_generation = 0;
                prev_state = block.state;
                gpu_id = block.home_gpu;
                sz = block.size_bytes;
                was_cpu_offloaded = block.cpu_buf_addr != 0;
                match block.state {
                    PolarisBlockState::AllocPending => {
                        block.state = PolarisBlockState::Resident;
                        block.gpu_phys_handle = arg.output_handle;
                        block.map_time_ns = unsafe { bindings::ktime_get_mono_fast_ns() };
                    }
                    PolarisBlockState::OffloadPending => {
                        block.state = PolarisBlockState::CpuOffloaded;
                        block.cpu_buf_addr = arg.output_cpu_addr;
                        // Phys handle was released by the daemon during offload.
                        block.gpu_phys_handle = 0;
                    }
                    PolarisBlockState::ReloadPending | PolarisBlockState::CowPending => {
                        block.state = PolarisBlockState::Resident;
                        block.gpu_phys_handle = arg.output_handle;
                        let now = unsafe { bindings::ktime_get_mono_fast_ns() };
                        block.map_time_ns = now;
                        block.last_touch_ns = now;
                        block.cpu_buf_addr = 0;
                    }
                    PolarisBlockState::FreePending => {
                        block.state = PolarisBlockState::Evicted;
                    }
                    _ => {}
                }
                // Track per-policy statistics.
                match prev_state {
                    PolarisBlockState::OffloadPending => {
                        inner.offload_count = inner.offload_count.saturating_add(1);
                    }
                    PolarisBlockState::ReloadPending => {
                        inner.reload_count = inner.reload_count.saturating_add(1);
                    }
                    PolarisBlockState::CowPending => {
                        inner.cow_break_count = inner.cow_break_count.saturating_add(1);
                        inner.cow_copy_bytes = inner.cow_copy_bytes.saturating_add(sz);
                    }
                    _ => {}
                }
                // Capture the completion pointer before the mutable borrow ends.
                comp_ptr = block.completion_ptr;
                block.completion_ptr = core::ptr::null_mut();
            }
            // Signal the bounded fault waiter, if this decision came from the
            // replayable GPU fault path.
            if !comp_ptr.is_null() {
                unsafe { bindings::complete(comp_ptr); }
            }

            // Update GPU and CPU pool used_bytes based on the state transition.
            let mut should_remove = false;
            let mut sid_to_clean = 0u64;
            let mut bid_to_clean = 0u64;
            match prev_state {
                PolarisBlockState::AllocPending | PolarisBlockState::CowPending => {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                        gpu.used_bytes += sz;
                    }
                }
                PolarisBlockState::ReloadPending => {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                        gpu.used_bytes += sz;
                        gpu.cpu_pool_used_bytes = gpu.cpu_pool_used_bytes.saturating_sub(sz);
                    }
                }
                PolarisBlockState::OffloadPending => {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                        gpu.used_bytes = gpu.used_bytes.saturating_sub(sz);
                        gpu.cpu_pool_used_bytes += sz;
                    }
                }
                PolarisBlockState::FreePending => {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                        if was_cpu_offloaded {
                            gpu.cpu_pool_used_bytes = gpu.cpu_pool_used_bytes.saturating_sub(sz);
                        } else {
                            gpu.used_bytes = gpu.used_bytes.saturating_sub(sz);
                        }
                    }
                    sid_to_clean = inner.blocks[block_idx].session_id;
                    bid_to_clean = inner.blocks[block_idx].block_id;
                    should_remove = true;
                }
                _ => {}
            }

            if should_remove {
                let _ = inner.blocks.remove(block_idx);
                if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == sid_to_clean) {
                    session.block_ids.retain(|bid| *bid != bid_to_clean);
                }
            }
            dev_info!(
                self.dev,
                "POLARIS: operation {} completed (handle=0x{:x})\n",
                arg.decision_id,
                arg.output_handle
            );
            return Ok(0);
        }

        // ── Error handling contract (G4) ──
        // Standard Linux errno values (negative i32 from daemon).
        const E_NOMEM: i32 = -(bindings::ENOMEM as i32);
        const E_NODEV: i32 = -(bindings::ENODEV as i32);
        const E_INVAL: i32 = -(bindings::EINVAL as i32);
        const E_FAULT: i32 = -(bindings::EFAULT as i32);

        // Increment retry counter (borrow dropped before match body).
        inner.blocks[block_idx].retry_count += 1;
        let retries = inner.blocks[block_idx].retry_count;
        let block_id = inner.blocks[block_idx].block_id;
        let home_gpu = inner.blocks[block_idx].home_gpu;

        match arg.result {
            E_NOMEM => {
                dev_warn!(
                    self.dev,
                    "POLARIS: ENOMEM on decision {} (block {}, retry {}/{})\n",
                    arg.decision_id,
                    block_id,
                    retries,
                    POLARIS_MAX_RETRIES,
                );
                if retries < POLARIS_MAX_RETRIES {
                    self.requeue_decision(inner, block_idx);
                } else {
                    dev_err!(
                        self.dev,
                        "POLARIS: block {} evicted after {} ENOMEM failures\n",
                        block_id,
                        POLARIS_MAX_RETRIES,
                    );
                    inner.blocks[block_idx].state = PolarisBlockState::Evicted;
                    inner.blocks[block_idx].pending_decision_id = 0;
                    inner.blocks[block_idx].pending_fault_id = 0;
                    inner.blocks[block_idx].pending_generation = 0;
                }
            }
            E_NODEV => {
                dev_err!(
                    self.dev,
                    "POLARIS: ENODEV on decision {} — marking GPU {} unhealthy\n",
                    arg.decision_id,
                    home_gpu,
                );
                if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == home_gpu) {
                    gpu.healthy = false;
                }
                inner.blocks[block_idx].state = PolarisBlockState::Evicted;
                inner.blocks[block_idx].pending_decision_id = 0;
                inner.blocks[block_idx].pending_fault_id = 0;
                inner.blocks[block_idx].pending_generation = 0;
            }
            E_INVAL => {
                dev_err!(
                    self.dev,
                    "POLARIS: EINVAL on decision {} — marking block {} FAILED\n",
                    arg.decision_id,
                    block_id,
                );
                inner.blocks[block_idx].state = PolarisBlockState::Evicted;
                inner.blocks[block_idx].pending_decision_id = 0;
                inner.blocks[block_idx].pending_fault_id = 0;
                inner.blocks[block_idx].pending_generation = 0;
            }
            E_FAULT => {
                dev_warn!(
                    self.dev,
                    "POLARIS: EFAULT on decision {} (block {}, retry {})\n",
                    arg.decision_id,
                    block_id,
                    retries,
                );
                if retries < 2 {
                    // Policy: retry once.
                    self.requeue_decision(inner, block_idx);
                } else {
                    dev_err!(
                        self.dev,
                        "POLARIS: block {} evicted after EFAULT retries\n",
                        block_id,
                    );
                    inner.blocks[block_idx].state = PolarisBlockState::Evicted;
                    inner.blocks[block_idx].pending_decision_id = 0;
                    inner.blocks[block_idx].pending_fault_id = 0;
                    inner.blocks[block_idx].pending_generation = 0;
                }
            }
            other => {
                dev_err!(
                    self.dev,
                    "POLARIS: unknown failure {} on decision {} — evicting block {}\n",
                    other,
                    arg.decision_id,
                    block_id,
                );
                inner.blocks[block_idx].state = PolarisBlockState::Evicted;
                inner.blocks[block_idx].pending_decision_id = 0;
                inner.blocks[block_idx].pending_fault_id = 0;
                inner.blocks[block_idx].pending_generation = 0;
            }
        }

        // Signal any bounded fault waiter blocked on this completion if the block
        // reached a terminal state.  Retry paths (ENOMEM/EFAULT with retries
        // remaining) keep the AllocPending state and a new pending_decision_id —
        // the waiter should NOT be woken yet.
        if inner.blocks[block_idx].state == PolarisBlockState::Evicted {
            inner.total_evictions = inner.total_evictions.saturating_add(1);
            let comp_ptr = inner.blocks[block_idx].completion_ptr;
            inner.blocks[block_idx].completion_ptr = core::ptr::null_mut();
            if !comp_ptr.is_null() {
                unsafe { bindings::complete(comp_ptr); }
            }
        }

        Ok(0)
    }

    /// Re-queue a decision for a block (G4 retry path).
    /// Preserves the original operation type and relevant fields
    /// (phys handle for OFFLOAD/COW_BREAK, CPU addr for RELOAD).
    fn requeue_decision(&self, inner: &mut PolarisInner, block_idx: usize) {
        let block = &mut inner.blocks[block_idx];
        let dec_id = inner.next_decision_id;
        inner.next_decision_id += 1;
        block.pending_decision_id = dec_id;

        let op = match block.state {
            PolarisBlockState::AllocPending => PolarisDecisionOp::Alloc as u32,
            PolarisBlockState::OffloadPending => PolarisDecisionOp::Offload as u32,
            PolarisBlockState::ReloadPending => PolarisDecisionOp::Reload as u32,
            PolarisBlockState::CowPending => PolarisDecisionOp::CowBreak as u32,
            _ => PolarisDecisionOp::Alloc as u32,
        };

        let src_handle = match block.state {
            PolarisBlockState::OffloadPending => block.gpu_phys_handle,
            PolarisBlockState::CowPending => block.cow_src_handle,
            _ => 0,
        };

        let cpu_addr = match block.state {
            PolarisBlockState::ReloadPending => block.cpu_buf_addr,
            _ => 0,
        };

        let _ = inner.pending_decisions.push(
            PolarisDecision {
                decision_id: dec_id,
                fault_id: 0,
                generation: 0,
                op,
                gpu_id: block.home_gpu,
                block_id: block.block_id,
                session_id: block.session_id,
                src_handle,
                dst_handle: 0,
                src_vaddr: 0,
                dst_vaddr: 0,
                size_bytes: block.size_bytes,
                cpu_addr,
                access_flags: 0,
                timeout_ms: 0,
                _reserved: [0u64; 4],
            },
            GFP_KERNEL,
        );
    }

    fn handle_get_global_stats(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisGetGlobalStatsArg = reader.read()?;
        let guard = POLARIS_STATE.lock();
        let inner = guard.as_ref().ok_or(ENODEV)?;

        arg.total_gpus = inner.gpus.len() as u32;
        arg.total_sessions = inner.sessions.len() as u32;
        arg.total_blocks = inner.blocks.len() as u32;

        let mut resident: u32 = 0;
        let mut offloaded: u32 = 0;
        let mut evicted: u32 = 0;
        let mut shared: u64 = 0;
        let mut private: u64 = 0;
        let mut memory_saved: u64 = 0;
        let mut total_gpu: u64 = 0;
        let mut used_gpu: u64 = 0;
        let mut cpu_total: u64 = 0;
        let mut cpu_used: u64 = 0;

        for block in &inner.blocks {
            match block.state {
                PolarisBlockState::Resident => resident += 1,
                PolarisBlockState::CpuOffloaded => offloaded += 1,
                PolarisBlockState::Evicted => evicted += 1,
                _ => {}
            }
            if block.flags.contains(PolarisBlockFlag::Shared) {
                shared += block.size_bytes;
                // Naive allocation without COW: each session that references
                // this block would need its own private copy.  The block
                // already exists as one copy, so we saved (refcount-1) copies.
                memory_saved = memory_saved.saturating_add(
                    block.refcount.saturating_sub(1).saturating_mul(block.size_bytes),
                );
            } else {
                private += block.size_bytes;
            }
        }
        for gpu in &inner.gpus {
            total_gpu += gpu.total_bytes;
            used_gpu += gpu.used_bytes;
            cpu_total += gpu.cpu_pool_total_bytes;
            cpu_used += gpu.cpu_pool_used_bytes;
        }
        let policy = inner.eviction_policy as u32;
        let offload_cnt = inner.offload_count;
        let reload_cnt = inner.reload_count;
        let evictions = inner.total_evictions;
        let cow_cnt = inner.cow_break_count;
        let cow_bytes = inner.cow_copy_bytes;
        let memory_saved_naive = memory_saved.saturating_sub(cow_bytes);
        drop(guard);

        arg.blocks_resident = resident;
        arg.blocks_offloaded = offloaded;
        arg.blocks_evicted = evicted;
        arg.shared_gpu_bytes = shared;
        arg.private_gpu_bytes = private;
        arg.cow_break_count = cow_cnt;
        arg.cow_copy_bytes = cow_bytes;
        arg.memory_saved_vs_naive = memory_saved_naive;
        arg.total_gpu_bytes = total_gpu;
        arg.used_gpu_bytes = used_gpu;
        arg.cpu_pool_total = cpu_total;
        arg.cpu_pool_used = cpu_used;
        arg.eviction_policy = policy;
        arg.offload_count = offload_cnt;
        arg.reload_count = reload_cnt;
        arg.total_evictions = evictions;

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_set_policy(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisSetPolicyArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        match arg.policy {
            0 => inner.eviction_policy = PolarisEvictionPolicy::Fifo,
            1 => inner.eviction_policy = PolarisEvictionPolicy::Lru,
            2 => inner.eviction_policy = PolarisEvictionPolicy::PhaseAware,
            _ => {
                dev_err!(
                    self.dev,
                    "POLARIS: unknown eviction policy {} (valid: 0=fifo, 1=lru, 2=phase_aware)\n",
                    arg.policy
                );
                return Err(EINVAL);
            }
        }

        // Reset per-policy counters on policy switch.
        inner.offload_count = 0;
        inner.reload_count = 0;

        dev_info!(
            self.dev,
            "POLARIS: eviction policy set to {:?}\n",
            inner.eviction_policy
        );
        Ok(0)
    }

    fn handle_list_sessions(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisListSessionsArg = reader.read()?;
        let guard = POLARIS_STATE.lock();
        let inner = guard.as_ref().ok_or(ENODEV)?;

        let count = core::cmp::min(inner.sessions.len(), POLARIS_MAX_SESSIONS_PER_LIST);
        arg.count = count as u32;
        for (i, session) in inner.sessions.iter().take(count).enumerate() {
            arg.session_ids[i] = session.session_id;
        }
        drop(guard);

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }
}
