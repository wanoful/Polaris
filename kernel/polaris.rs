// SPDX-License-Identifier: GPL-2.0

//! POLARIS: Paged Operating Layer for Accelerated Routing and Inference Systems
//!
//! This kernel module provides OS-level paged KV Cache management for LLM inference.
//! It maintains the authoritative block table, manages sessions, and issues GPU memory
//! management decisions to the userspace daemon (polarisd) via an ioctl-based
//! decision protocol.
//!
//! All state is global: every open("/dev/polaris") shares the same block table,
//! session table, GPU registry, and decision queue. This is the fundamental
//! value of the kernel module — cross-process visibility.

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

// ─── Global shared state ────────────────────────────────────────────────────

pub(crate) struct PolarisInner {
    next_block_id: u64,
    next_session_id: u64,
    next_decision_id: u64,
    daemon_attached: u32,
    gpus: KVec<PolarisGpu>,
    blocks: KVec<PolarisBlock>,
    sessions: KVec<PolarisSession>,
    pending_decisions: KVec<PolarisDecision>,
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
                daemon_attached: 0,
                gpus: KVec::new(),
                blocks: KVec::new(),
                sessions: KVec::new(),
                pending_decisions: KVec::new(),
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
            POLARIS_SESSION_CREATE => me.handle_session_create(user_ptr, size),
            POLARIS_SESSION_DESTROY => me.handle_session_destroy(user_ptr, size),
            POLARIS_SESSION_GET_STATS => me.handle_session_get_stats(user_ptr, size),
            POLARIS_SESSION_BRANCH => me.handle_session_branch(user_ptr, size),
            POLARIS_BLOCK_GROW => me.handle_block_grow(user_ptr, size),
            POLARIS_BLOCK_FREE => me.handle_block_free(user_ptr, size),
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
                            // Wake any synchronous BLOCK_GROW waiter.
                            let comp_ptr = block.completion_ptr;
                            block.completion_ptr = core::ptr::null_mut();
                            if !comp_ptr.is_null() {
                                // SAFETY: comp_ptr was set by BLOCK_GROW; stack frame still alive.
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
                    total_bytes: arg.total_bytes,
                    used_bytes: 0,
                    budget_bytes: arg.budget_bytes,
                    pressure_score: 0,
                    cpu_pool_total_bytes: arg.cpu_pool_bytes,
                    cpu_pool_used_bytes: 0,
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

    fn handle_session_create(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisSessionCreateArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        // G4: reject session creation on an unhealthy GPU.
        if let Some(gpu) = inner.gpus.iter().find(|g| g.gpu_id == arg.home_gpu) {
            if !gpu.healthy {
                dev_err!(self.dev, "POLARIS: GPU {} is unhealthy, rejecting session\n", arg.home_gpu);
                return Err(ENODEV);
            }
        }

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
                gpu_vas_base: 0,
                gpu_vas_size: arg.gpu_vas_bytes,
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
                        op: PolarisDecisionOp::Free as u32,
                        gpu_id,
                        block_id,
                        session_id: sid,
                        src_handle: phys_handle,
                        dst_vaddr: 0,
                        size_bytes: sz,
                        cpu_addr: 0,
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
                gpu_vas_base: 0,
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

    fn handle_block_grow(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisBlockGrowArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();

        // Gate checks + extract settings (inner scoped to lock).
        let gpu_id;
        let bpt;
        let size_bytes;
        let mut offloaded_bids: KVec<u64>;
        {
            let inner = guard.as_mut().ok_or(ENODEV)?;
            if inner.daemon_attached == 0 {
                return Err(ENODEV);
            }
            if !inner.sessions.iter().any(|s| s.session_id == arg.session_id) {
                return Err(ENOENT);
            }
            // G4: check GPU health.
            {
                let sess = inner.sessions.iter().find(|s| s.session_id == arg.session_id);
                if let Some(sess) = sess {
                    if let Some(gpu) = inner.gpus.iter().find(|g| g.gpu_id == sess.home_gpu) {
                        if !gpu.healthy {
                            dev_err!(self.dev, "POLARIS: GPU {} unhealthy, rejecting BLOCK_GROW\n", sess.home_gpu);
                            return Err(ENODEV);
                        }
                    }
                    gpu_id = sess.home_gpu;
                    bpt = sess.bytes_per_token;
                } else {
                    gpu_id = 0;
                    bpt = POLARIS_DEFAULT_BYTES_PER_TOKEN;
                }
            }
            size_bytes = (arg.token_count as u64) * bpt;

            // ── Phase 2a: Collect CPU_OFFLOADED blocks for pre-decode residency ──
            // Iterate the session's block_ids to cover both directly-owned
            // and COW-shared blocks (whose session_id is the parent's).
            offloaded_bids = KVec::new();
            if let Some(sess) = inner.sessions.iter().find(|s| s.session_id == arg.session_id) {
                for &bid in &sess.block_ids {
                    if let Some(block) = inner.blocks.iter().find(|b| b.block_id == bid) {
                        if block.state == PolarisBlockState::CpuOffloaded {
                            offloaded_bids.push(block.block_id, GFP_KERNEL)?;
                        }
                    }
                }
            }
        } // inner dropped here, freeing the borrow on guard

        // ── Reload each offloaded block (lock/unlock per block) ─────────
        for &bid in &offloaded_bids {
            // Phase 2a: budget-check before reload. Reloading increases
            // GPU used_bytes — if the budget is tight, we must offload a
            // different block first to make room. If no victim is available,
            // the session cannot proceed (attention needs all blocks resident).
            let reload_sz;
            {
                let inner = guard.as_mut().ok_or(ENODEV)?;
                let block = match inner.blocks.iter_mut().find(|b| b.block_id == bid) {
                    Some(b) => b,
                    None => continue,
                };
                if block.state != PolarisBlockState::CpuOffloaded {
                    continue;
                }
                reload_sz = block.size_bytes;
            }

            // Make room for the reload if needed.
            loop {
                let has_budget;
                {
                    let inner = guard.as_mut().ok_or(ENODEV)?;
                    let pending = inner.blocks.iter()
                        .filter(|b| b.home_gpu == gpu_id)
                        .filter(|b| matches!(b.state,
                            PolarisBlockState::AllocPending
                            | PolarisBlockState::ReloadPending
                            | PolarisBlockState::CowPending))
                        .map(|b| b.size_bytes)
                        .sum::<u64>();

                    has_budget = match inner.gpus.iter().find(|g| g.gpu_id == gpu_id) {
                        Some(g) => g.used_bytes + pending + reload_sz <= g.budget_bytes,
                        None => false,
                    };
                }
                if has_budget {
                    break;
                }

                // Find a victim to offload to make room for this reload.
                let vid = {
                    let inner = guard.as_mut().ok_or(ENODEV)?;
                    polaris_policy::select_victim(inner, arg.session_id, gpu_id)
                };

                let (victim_id, victim_sz, victim_gpu) = match vid {
                    Some((_, id, sz, gpu)) => (id, sz, gpu),
                    None => {
                        dev_err!(
                            self.dev,
                            "POLARIS: cannot reload block {} — budget exceeded, no eligible resident block found\n",
                            bid,
                        );
                        return Err(ENOMEM);
                    }
                };

                // Queue OFFLOAD and wait.
                {
                    let inner = guard.as_mut().ok_or(ENODEV)?;
                    let block = match inner.blocks.iter_mut().find(|b| b.block_id == victim_id) {
                        Some(b) => b,
                        None => return Err(ENOENT),
                    };
                    let src_phys = block.gpu_phys_handle;
                    block.state = PolarisBlockState::OffloadPending;
                    let dec_id = inner.next_decision_id;
                    inner.next_decision_id += 1;
                    block.pending_decision_id = dec_id;

                    inner.pending_decisions.push(
                        PolarisDecision {
                            decision_id: dec_id,
                            op: PolarisDecisionOp::Offload as u32,
                            gpu_id: victim_gpu,
                            block_id: victim_id,
                            session_id: block.session_id,
                            src_handle: src_phys,
                            dst_vaddr: 0,
                            size_bytes: victim_sz,
                            cpu_addr: 0,
                            _reserved: [0u64; 4],
                        },
                        GFP_KERNEL,
                    )?;
                }

                let mut comp: bindings::completion = unsafe { core::mem::zeroed() };
                unsafe { bindings::init_completion(&raw mut comp); }

                {
                    let inner = guard.as_mut().ok_or(ENODEV)?;
                    if let Some(b) = inner.blocks.iter_mut().find(|b| b.block_id == victim_id) {
                        b.completion_ptr = &raw mut comp;
                    }
                }
                drop(guard);

                unsafe {
                    bindings::wait_for_completion_interruptible_timeout(
                        &raw mut comp,
                        bindings::__msecs_to_jiffies(5000),
                    );
                }

                guard = POLARIS_STATE.lock();
                {
                    let inner = guard.as_mut().ok_or(ENODEV)?;
                    if let Some(b) = inner.blocks.iter_mut().find(|b| b.block_id == victim_id) {
                        b.completion_ptr = core::ptr::null_mut();
                        if b.state != PolarisBlockState::CpuOffloaded {
                            dev_err!(
                                self.dev,
                                "POLARIS: OFFLOAD for block {} failed (state={:?}), reload aborted\n",
                                victim_id, b.state
                            );
                            return Err(ENOMEM);
                        }
                    }
                }

                dev_info!(
                    self.dev,
                    "POLARIS: offloaded block {} to make room for reload of {}\n",
                    victim_id, bid,
                );
            }

            // Budget is sufficient — now queue and wait for the RELOAD.
            {
                let inner = guard.as_mut().ok_or(ENODEV)?;
                let dec_id;
                let cpu_addr;
                let sz;
                {
                    let block = match inner.blocks.iter_mut().find(|b| b.block_id == bid) {
                        Some(b) => b,
                        None => continue,
                    };
                    if block.state != PolarisBlockState::CpuOffloaded {
                        continue;
                    }
                    block.state = PolarisBlockState::ReloadPending;
                    dec_id = inner.next_decision_id;
                    inner.next_decision_id += 1;
                    block.pending_decision_id = dec_id;
                    cpu_addr = block.cpu_buf_addr;
                    sz = block.size_bytes;

                    inner.pending_decisions.push(
                        PolarisDecision {
                            decision_id: dec_id,
                            op: PolarisDecisionOp::Reload as u32,
                            gpu_id,
                            block_id: bid,
                            session_id: arg.session_id,
                            src_handle: 0,
                            dst_vaddr: 0,
                            size_bytes: sz,
                            cpu_addr,
                            _reserved: [0u64; 4],
                        },
                        GFP_KERNEL,
                    )?;
                }
            } // inner dropped

            // Wait for the RELOAD to complete.
            let mut comp: bindings::completion = unsafe { core::mem::zeroed() };
            unsafe { bindings::init_completion(&raw mut comp); }

            {
                let inner = guard.as_mut().ok_or(ENODEV)?;
                if let Some(b) = inner.blocks.iter_mut().find(|b| b.block_id == bid) {
                    b.completion_ptr = &raw mut comp;
                }
            }
            drop(guard);

            let _wait_ret = unsafe {
                bindings::wait_for_completion_interruptible_timeout(
                    &raw mut comp,
                    bindings::__msecs_to_jiffies(5000),
                )
            };

            guard = POLARIS_STATE.lock();
            {
                let inner = guard.as_mut().ok_or(ENODEV)?;
                if let Some(b) = inner.blocks.iter_mut().find(|b| b.block_id == bid) {
                    b.completion_ptr = core::ptr::null_mut();
                        if b.state != PolarisBlockState::Resident {
                            dev_err!(
                                self.dev,
                                "POLARIS: pre-decode RELOAD for block {} failed (state={:?})\n",
                                bid, b.state
                            );
                            b.state = PolarisBlockState::Evicted;
                            b.pending_decision_id = 0;
                            return Err(ENOMEM);
                        }
                }
            }
        }

        // ── Phase 3: COW overlap detection ─────────────────────────────
        // We use a flag to route control flow: needs_cow_break.
        let mut needs_cow_break = false;
        let mut cb_old_block_id: u64 = 0;
        let mut cb_old_phys: u64 = 0;
        let mut cb_old_size: u64 = 0;
        let mut cb_old_token_start: u32 = 0;
        let mut cb_old_token_count: u32 = 0;
        let mut cb_old_phase: PolarisPhase = PolarisPhase::Prefill;
        {
            let inner = guard.as_mut().ok_or(ENODEV)?;
            let mut grow_flags = PolarisGrowFlags::empty();
            if arg.flags & POLARIS_GROW_FLAG_OVERWRITE != 0 {
                grow_flags |= PolarisGrowFlag::Overwrite;
            }
            let token_end = arg.token_start + arg.token_count;

            // Scan session's block_ids for an overlapping block.
            let mut overlap_idx: Option<usize> = None;
            if let Some(sess) = inner.sessions.iter().find(|s| s.session_id == arg.session_id) {
                for &bid in &sess.block_ids {
                    if let Some(idx) = inner.blocks.iter().position(|b| b.block_id == bid) {
                        let b = &inner.blocks[idx];
                        let b_end = b.token_start + b.token_count;
                        if arg.token_start < b_end && b.token_start < token_end {
                            overlap_idx = Some(idx);
                            break;
                        }
                    }
                }
            }

            match overlap_idx {
                None => { /* fall through to normal ALLOC path */ }
                Some(oi) => {
                    if !grow_flags.contains(PolarisGrowFlag::Overwrite) {
                        dev_err!(
                            self.dev,
                            "POLARIS: BLOCK_GROW overlap without OVERWRITE flag (session={}, tokens={}..{})\n",
                            arg.session_id, arg.token_start, token_end
                        );
                        return Err(EINVAL);
                    }

                    let old_refcount = inner.blocks[oi].refcount;
                    let old_block_id = inner.blocks[oi].block_id;

                    if old_refcount <= 1 {
                        // Private ─ in-place overwrite.
                        arg.block_id = old_block_id;
                        arg.ret_code = 0;
                        drop(guard);
                        let mut writer = UserSlice::new(user_ptr, size).writer();
                        writer.write(&arg)?;
                        dev_info!(
                            self.dev,
                            "POLARIS: in-place overwrite block {} for session {}\n",
                            old_block_id, arg.session_id
                        );
                        return Ok(0);
                    }

                    // Shared block (refcount > 1): COW break needed.
                    dev_info!(
                        self.dev,
                        "POLARIS: COW_BREAK trigger ─ session={} overwriting shared block {} (refcount={})\n",
                        arg.session_id, old_block_id, old_refcount
                    );

                    // Decrement old refcount; clear Shared if it drops to 1.
                    inner.blocks[oi].refcount -= 1;
                    if inner.blocks[oi].refcount == 1 {
                        inner.blocks[oi].flags = inner.blocks[oi].flags & !PolarisBlockFlag::Shared;
                    }

                    cb_old_block_id = old_block_id;
                    cb_old_phys = inner.blocks[oi].gpu_phys_handle;
                    cb_old_size = inner.blocks[oi].size_bytes;
                    cb_old_token_start = inner.blocks[oi].token_start;
                    cb_old_token_count = inner.blocks[oi].token_count;
                    cb_old_phase = inner.blocks[oi].phase;
                    needs_cow_break = true;
                }
            }
        } // guard (inner borrow) dropped

        // ── COW BREAK execution ─────────────────────────────────────────────────
        if needs_cow_break {
            let cow_size = cb_old_size;

            // Budget check + offload for the new private block.
            loop {
                let has_budget;
                {
                    let inner = guard.as_mut().ok_or(ENODEV)?;
                    let pending = inner.blocks.iter()
                        .filter(|b| b.home_gpu == gpu_id)
                        .filter(|b| matches!(b.state,
                            PolarisBlockState::AllocPending
                            | PolarisBlockState::ReloadPending
                            | PolarisBlockState::CowPending))
                        .map(|b| b.size_bytes)
                        .sum::<u64>();

                    has_budget = match inner.gpus.iter().find(|g| g.gpu_id == gpu_id) {
                        Some(g) => g.used_bytes + pending + cow_size <= g.budget_bytes,
                        None => false,
                    };
                }
                if has_budget {
                    break;
                }

                let victim = {
                    let inner = guard.as_mut().ok_or(ENODEV)?;
                    polaris_policy::select_victim(inner, arg.session_id, gpu_id)
                };

                let (victim_id, victim_sz, victim_gpu) = match victim {
                    Some((_, id, sz, gpu)) => (id, sz, gpu),
                    None => {
                        dev_err!(self.dev, "POLARIS: COW_BREAK budget exceeded, no victim available\n");
                        arg.ret_code = -(bindings::ENOMEM as i32);
                        return Err(ENOMEM);
                    }
                };

                {
                    let inner = guard.as_mut().ok_or(ENODEV)?;
                    let block = match inner.blocks.iter_mut().find(|b| b.block_id == victim_id) {
                        Some(b) => b,
                        None => {
                            arg.ret_code = -(bindings::ENOENT as i32);
                            return Err(ENOENT);
                        }
                    };
                    let src_phys = block.gpu_phys_handle;
                    block.state = PolarisBlockState::OffloadPending;
                    let dec_id = inner.next_decision_id;
                    inner.next_decision_id += 1;
                    block.pending_decision_id = dec_id;

                    inner.pending_decisions.push(
                        PolarisDecision {
                            decision_id: dec_id,
                            op: PolarisDecisionOp::Offload as u32,
                            gpu_id: victim_gpu,
                            block_id: victim_id,
                            session_id: block.session_id,
                            src_handle: src_phys,
                            dst_vaddr: 0,
                            size_bytes: victim_sz,
                            cpu_addr: 0,
                            _reserved: [0u64; 4],
                        },
                        GFP_KERNEL,
                    )?;
                }

                let mut comp: bindings::completion = unsafe { core::mem::zeroed() };
                unsafe { bindings::init_completion(&raw mut comp); }

                {
                    let inner = guard.as_mut().ok_or(ENODEV)?;
                    if let Some(b) = inner.blocks.iter_mut().find(|b| b.block_id == victim_id) {
                        b.completion_ptr = &raw mut comp;
                    }
                }
                drop(guard);

                unsafe {
                    bindings::wait_for_completion_interruptible_timeout(
                        &raw mut comp,
                        bindings::__msecs_to_jiffies(5000),
                    );
                }

                guard = POLARIS_STATE.lock();
                {
                    let inner = guard.as_mut().ok_or(ENODEV)?;
                    if let Some(b) = inner.blocks.iter_mut().find(|b| b.block_id == victim_id) {
                        b.completion_ptr = core::ptr::null_mut();
                        if b.state != PolarisBlockState::CpuOffloaded {
                            dev_err!(self.dev, "POLARIS: OFFLOAD for COW_BREAK victim {} failed, aborting\n", victim_id);
                            arg.ret_code = -(bindings::ENOMEM as i32);
                            return Err(ENOMEM);
                        }
                    }
                }
            }

            // Queue full check.
            {
                let inner = guard.as_mut().ok_or(ENODEV)?;
                if inner.pending_decisions.len() >= POLARIS_MAX_PENDING_DECISIONS {
                    dev_err!(self.dev, "POLARIS: pending decision queue full, rejecting COW_BREAK\n");
                    arg.ret_code = -(bindings::ENOMEM as i32);
                    return Err(ENOMEM);
                }
            }

            // Create new block and queue CowBreak decision.
            let new_block_id;
            let cow_dec_id;
            {
                let inner = guard.as_mut().ok_or(ENODEV)?;
                new_block_id = inner.next_block_id;
                inner.next_block_id += 1;
                cow_dec_id = inner.next_decision_id;
                inner.next_decision_id += 1;

                let new_block = PolarisBlock {
                    block_id: new_block_id,
                    session_id: arg.session_id,
                    token_start: cb_old_token_start,
                    token_count: cb_old_token_count,
                    home_gpu: gpu_id,
                    gpu_vaddr: 0,
                    gpu_phys_handle: 0,
                    cpu_buf_addr: 0,
                    size_bytes: cb_old_size,
                    refcount: 1,
                    state: PolarisBlockState::CowPending,
                    flags: PolarisBlockFlags::empty(),
                    phase: cb_old_phase,
                    last_touch_ns: 0,
                    map_time_ns: 0,
                    cow_src_handle: cb_old_phys,
                    retry_count: 0,
                    pending_decision_id: cow_dec_id,
                    completion_ptr: core::ptr::null_mut(),
                };

                inner.pending_decisions.push(
                    PolarisDecision {
                        decision_id: cow_dec_id,
                        op: PolarisDecisionOp::CowBreak as u32,
                        gpu_id,
                        block_id: new_block_id,
                        session_id: arg.session_id,
                        src_handle: cb_old_phys,
                        dst_vaddr: 0,
                        size_bytes: cb_old_size,
                        cpu_addr: 0,
                        _reserved: [0u64; 4],
                    },
                    GFP_KERNEL,
                )?;
                inner.blocks.push(new_block, GFP_KERNEL)?;

                // Replace old block_id with new one in session's block_ids.
                if let Some(sess) = inner.sessions.iter_mut().find(|s| s.session_id == arg.session_id) {
                    if let Some(pos) = sess.block_ids.iter().position(|&bid| bid == cb_old_block_id) {
                        sess.block_ids[pos] = new_block_id;
                    }
                }
            }

            arg.block_id = new_block_id;

            // Wait for COW_BREAK completion.
            let mut comp: bindings::completion = unsafe { core::mem::zeroed() };
            unsafe { bindings::init_completion(&raw mut comp); }

            {
                let inner = guard.as_mut().ok_or(ENODEV)?;
                if let Some(b) = inner.blocks.iter_mut().find(|b| b.block_id == new_block_id) {
                    b.completion_ptr = &raw mut comp;
                }
            }
            drop(guard);

            let wait_ret = unsafe {
                bindings::wait_for_completion_interruptible_timeout(
                    &raw mut comp,
                    bindings::__msecs_to_jiffies(5000),
                )
            };

            let mut g = POLARIS_STATE.lock();
            let outcome;
            {
                let inner = g.as_mut().ok_or(ENODEV)?;
                outcome = if let Some(b) = inner.blocks.iter_mut().find(|b| b.block_id == new_block_id) {
                    b.completion_ptr = core::ptr::null_mut();
                    match b.state {
                        PolarisBlockState::Resident => 0i32,
                        PolarisBlockState::Evicted => -(bindings::ENOMEM as i32),
                        _ => {
                            if wait_ret == 0 {
                                dev_err!(self.dev, "POLARIS: COW_BREAK block {} timed out\n", new_block_id);
                                b.state = PolarisBlockState::Evicted;
                                b.pending_decision_id = 0;
                                -(bindings::ETIMEDOUT as i32)
                            } else if wait_ret < 0 {
                                -(bindings::EINTR as i32)
                            } else {
                                dev_err!(self.dev, "POLARIS: COW_BREAK block {} unexpected state {:?}\n", new_block_id, b.state);
                                -(bindings::EIO as i32)
                            }
                        }
                    }
                } else {
                    -(bindings::EIO as i32)
                };
            }
            drop(g);

            arg.ret_code = outcome;
            let mut writer = UserSlice::new(user_ptr, size).writer();
            writer.write(&arg)?;
            if outcome == 0 {
                dev_info!(self.dev, "POLARIS: COW_BREAK complete ─ block {} private for session {}\n", new_block_id, arg.session_id);
            }
            return Ok(0);
        }

        // ── Budget check + offload loop (normal ALLOC path) ────────────

        loop {
            let has_budget;
            {
                let inner = guard.as_mut().ok_or(ENODEV)?;
                let pending = inner.blocks.iter()
                    .filter(|b| b.home_gpu == gpu_id)
                    .filter(|b| matches!(b.state,
                        PolarisBlockState::AllocPending
                        | PolarisBlockState::ReloadPending
                        | PolarisBlockState::CowPending))
                    .map(|b| b.size_bytes)
                    .sum::<u64>();

                has_budget = match inner.gpus.iter().find(|g| g.gpu_id == gpu_id) {
                    Some(g) => g.used_bytes + pending + size_bytes <= g.budget_bytes,
                    None => false,
                };
            }

            if has_budget {
                break;
            }

            // Select a victim to offload using the current eviction policy.
            let victim = {
                let inner = guard.as_mut().ok_or(ENODEV)?;
                polaris_policy::select_victim(inner, arg.session_id, gpu_id)
            };

            let (victim_id, victim_sz, victim_gpu) = match victim {
                Some((_, id, sz, gpu)) => (id, sz, gpu),
                None => {
                    dev_err!(self.dev, "POLARIS: GPU {} budget exceeded, no eligible resident block found (recheck budget or CPU pool)\n", gpu_id);
                    return Err(ENOMEM);
                }
            };

            // Queue OFFLOAD for the victim.
            {
                let inner = guard.as_mut().ok_or(ENODEV)?;
                let block = match inner.blocks.iter_mut().find(|b| b.block_id == victim_id) {
                    Some(b) => b,
                    None => return Err(ENOENT),
                };
                let src_phys = block.gpu_phys_handle;
                block.state = PolarisBlockState::OffloadPending;
                let dec_id = inner.next_decision_id;
                inner.next_decision_id += 1;
                block.pending_decision_id = dec_id;

                inner.pending_decisions.push(
                    PolarisDecision {
                        decision_id: dec_id,
                        op: PolarisDecisionOp::Offload as u32,
                        gpu_id: victim_gpu,
                        block_id: victim_id,
                        session_id: block.session_id,
                        src_handle: src_phys,
                        dst_vaddr: 0,
                        size_bytes: victim_sz,
                        cpu_addr: 0,
                        _reserved: [0u64; 4],
                    },
                    GFP_KERNEL,
                )?;
            }

            // Wait for OFFLOAD to complete.
            let mut comp: bindings::completion = unsafe { core::mem::zeroed() };
            unsafe { bindings::init_completion(&raw mut comp); }

            {
                let inner = guard.as_mut().ok_or(ENODEV)?;
                if let Some(b) = inner.blocks.iter_mut().find(|b| b.block_id == victim_id) {
                    b.completion_ptr = &raw mut comp;
                }
            }
            drop(guard);

            let _wait_ret = unsafe {
                bindings::wait_for_completion_interruptible_timeout(
                    &raw mut comp,
                    bindings::__msecs_to_jiffies(5000),
                )
            };

            guard = POLARIS_STATE.lock();
            {
                let inner = guard.as_mut().ok_or(ENODEV)?;
                if let Some(b) = inner.blocks.iter_mut().find(|b| b.block_id == victim_id) {
                    b.completion_ptr = core::ptr::null_mut();
                    if b.state != PolarisBlockState::CpuOffloaded {
                        dev_err!(
                            self.dev,
                            "POLARIS: OFFLOAD for block {} failed (state={:?})\n",
                            victim_id, b.state
                        );
                        return Err(ENOMEM);
                    }
                }
            }

            dev_info!(
                self.dev,
                "POLARIS: offloaded block {} to CPU, retrying budget check\n",
                victim_id,
            );
        }

        // ── Allocate new block ───────────────────────────────────────────
        // G6: reject BLOCK_GROW when the pending decision queue is full.
        // This prevents the kernel from OOM-ing when the daemon is stuck.
        {
            let inner = guard.as_mut().ok_or(ENODEV)?;
            if inner.pending_decisions.len() >= POLARIS_MAX_PENDING_DECISIONS {
                dev_err!(
                    self.dev,
                    "POLARIS: pending decision queue full ({}), rejecting BLOCK_GROW\n",
                    inner.pending_decisions.len()
                );
                return Err(ENOMEM);
            }
        }

        let block_id;
        let dec_id;
        {
            let inner = guard.as_mut().ok_or(ENODEV)?;
            block_id = inner.next_block_id;
            inner.next_block_id += 1;

            dec_id = inner.next_decision_id;
            inner.next_decision_id += 1;

            let phase = if arg.phase == PolarisPhase::Decode as u32 {
                PolarisPhase::Decode
            } else {
                PolarisPhase::Prefill
            };
            let block = PolarisBlock {
                block_id,
                session_id: arg.session_id,
                token_start: arg.token_start,
                token_count: arg.token_count,
                home_gpu: gpu_id,
                gpu_vaddr: 0,
                gpu_phys_handle: 0,
                cpu_buf_addr: 0,
                size_bytes,
                refcount: 1,
                state: PolarisBlockState::AllocPending,
                flags: PolarisBlockFlags::empty(),
                phase,
                last_touch_ns: 0,
                map_time_ns: 0,
                cow_src_handle: 0,
                retry_count: 0,
                pending_decision_id: dec_id,
                completion_ptr: core::ptr::null_mut(),
            };

            inner.pending_decisions.push(
                PolarisDecision {
                    decision_id: dec_id,
                    op: PolarisDecisionOp::Alloc as u32,
                    gpu_id,
                    block_id,
                    session_id: arg.session_id,
                    src_handle: 0,
                    dst_vaddr: 0,
                    size_bytes,
                    cpu_addr: 0,
                    _reserved: [0u64; 4],
                },
                GFP_KERNEL,
            )?;
            inner.blocks.push(block, GFP_KERNEL)?;

            if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == arg.session_id) {
                session.block_ids.push(block_id, GFP_KERNEL)?;
            }
        }

        arg.block_id = block_id;

        // Synchronous page-fault: wait for the daemon to complete this decision.
        let mut comp: bindings::completion = unsafe { core::mem::zeroed() };
        unsafe { bindings::init_completion(&raw mut comp); }

        {
            let inner = guard.as_mut().ok_or(ENODEV)?;
            if let Some(b) = inner.blocks.iter_mut().find(|b| b.block_id == block_id) {
                b.completion_ptr = &raw mut comp;
            }
        }
        drop(guard);

        let wait_ret = unsafe {
            bindings::wait_for_completion_interruptible_timeout(
                &raw mut comp,
                bindings::__msecs_to_jiffies(5000),
            )
        };

        // Re-acquire lock and read the result.
        let mut g = POLARIS_STATE.lock();
        let outcome;
        {
            let inner = g.as_mut().ok_or(ENODEV)?;
            outcome = if let Some(b) = inner.blocks.iter_mut().find(|b| b.block_id == block_id) {
                b.completion_ptr = core::ptr::null_mut();
                match b.state {
                    PolarisBlockState::Resident => 0i32,
                    PolarisBlockState::Evicted => -(bindings::ENOMEM as i32),
                    _ => {
                        if wait_ret == 0 {
                            dev_err!(self.dev, "POLARIS: block {} timed out waiting for daemon\n", block_id);
                            b.state = PolarisBlockState::Evicted;
                            b.pending_decision_id = 0;
                            -(bindings::ETIMEDOUT as i32)
                        } else if wait_ret < 0 {
                            dev_info!(self.dev, "POLARIS: block {} wait interrupted (ret={})\n", block_id, wait_ret);
                            -(bindings::EINTR as i32)
                        } else {
                            dev_err!(self.dev, "POLARIS: block {} in unexpected state {:?} after completion\n", block_id, b.state);
                            -(bindings::EIO as i32)
                        }
                    }
                }
            } else {
                dev_err!(self.dev, "POLARIS: block {} vanished during wait\n", block_id);
                -(bindings::EIO as i32)
            };
        }
        drop(g);

        arg.ret_code = outcome;
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        if outcome == 0 {
            dev_info!(self.dev, "POLARIS: block {} resident for session {}\n", block_id, arg.session_id);
        }
        Ok(0)
    }

    fn handle_block_free(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisBlockFreeArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        // Find the block by session_id and token_start.
        // Check both directly-owned blocks (session_id match) and COW-shared
        // blocks (looked up via the session's block_ids list).
        let block_idx = {
            // First: direct match on session_id.
            let direct = inner
                .blocks
                .iter()
                .position(|b| b.session_id == arg.session_id && b.token_start == arg.token_start);
            if direct.is_some() {
                direct
            } else {
                // COW child: search the session's block_ids for a matching token_start.
                let session = inner.sessions.iter().find(|s| s.session_id == arg.session_id);
                session.and_then(|sess| {
                    sess.block_ids.iter().find_map(|&bid| {
                        inner.blocks.iter()
                            .position(|b| b.block_id == bid && b.token_start == arg.token_start)
                    })
                })
            }
        }.ok_or(ENOENT)?;

        // Collect info before mutation.
        let block_id = inner.blocks[block_idx].block_id;
        let home_gpu = inner.blocks[block_idx].home_gpu;
        let session_id = inner.blocks[block_idx].session_id;
        let phys_handle = inner.blocks[block_idx].gpu_phys_handle;
        let size_bytes = inner.blocks[block_idx].size_bytes;
        let current_state = inner.blocks[block_idx].state;
        let had_cpu_buf = inner.blocks[block_idx].cpu_buf_addr != 0;

        // Decrement COW refcount.
        inner.blocks[block_idx].refcount -= 1;

        if inner.blocks[block_idx].refcount == 0 {
            // No more sessions reference this block — release GPU resources.
            // G6: if the pending decision queue is full, clean up directly.
            let queue_full = inner.pending_decisions.len() >= POLARIS_MAX_PENDING_DECISIONS;
            if !queue_full && inner.daemon_attached > 0 && phys_handle != 0
                && current_state != PolarisBlockState::FreePending
            {
                let dec_id = inner.next_decision_id;
                inner.next_decision_id += 1;

                inner.blocks[block_idx].state = PolarisBlockState::FreePending;
                inner.blocks[block_idx].pending_decision_id = dec_id;

                inner.pending_decisions.push(
                    PolarisDecision {
                        decision_id: dec_id,
                        op: PolarisDecisionOp::Free as u32,
                        gpu_id: home_gpu,
                        block_id,
                        session_id,
                        src_handle: phys_handle,
                        dst_vaddr: 0,
                        size_bytes,
                        cpu_addr: 0,
                        _reserved: [0u64; 4],
                    },
                    GFP_KERNEL,
                )?;

                dev_info!(self.dev, "POLARIS: block {} free queued (FREE decision {})\n", block_id, dec_id);
            } else {
                // No daemon, queue full, or never mapped: remove the block directly.
                if current_state == PolarisBlockState::Resident {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == home_gpu) {
                        gpu.used_bytes = gpu.used_bytes.saturating_sub(size_bytes);
                    }
                } else if had_cpu_buf {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == home_gpu) {
                        gpu.cpu_pool_used_bytes = gpu.cpu_pool_used_bytes.saturating_sub(size_bytes);
                    }
                }
                let _ = inner.blocks.remove(block_idx);
                dev_info!(self.dev, "POLARIS: block {} freed directly\n", block_id);
            }
        } else {
            // Still referenced by other sessions — clear Shared flag if now private.
            if inner.blocks[block_idx].refcount == 1 {
                inner.blocks[block_idx].flags = inner.blocks[block_idx].flags & !PolarisBlockFlag::Shared;
            }
            dev_info!(
                self.dev, "POLARIS: block {} refcount decremented to {}\n",
                block_id, inner.blocks[block_idx].refcount
            );
        }

        // Remove from session's block_ids list.
        if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == arg.session_id) {
            session.block_ids.retain(|bid| *bid != block_id);
        }

        Ok(0)
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
            // Signal any synchronous waiter (BLOCK_GROW) that the decision is done.
            // SAFETY: comp_ptr was written by BLOCK_GROW while holding the same lock.
            // The stack frame holding the completion is still alive (waiter sleeps in it).
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
            }
        }

        // Signal any BLOCK_GROW waiter blocked on this completion if the block
        // reached a terminal state.  Retry paths (ENOMEM/EFAULT with retries
        // remaining) keep the AllocPending state and a new pending_decision_id —
        // the waiter should NOT be woken yet.
        if inner.blocks[block_idx].state == PolarisBlockState::Evicted {
            inner.total_evictions = inner.total_evictions.saturating_add(1);
            let comp_ptr = inner.blocks[block_idx].completion_ptr;
            inner.blocks[block_idx].completion_ptr = core::ptr::null_mut();
            if !comp_ptr.is_null() {
                // SAFETY: comp_ptr was written by BLOCK_GROW while holding the
                // same lock.  The stack frame is still alive.
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
                op,
                gpu_id: block.home_gpu,
                block_id: block.block_id,
                session_id: block.session_id,
                src_handle,
                dst_vaddr: 0,
                size_bytes: block.size_bytes,
                cpu_addr,
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
