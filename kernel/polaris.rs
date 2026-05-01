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

struct PolarisInner {
    next_block_id: u64,
    next_session_id: u64,
    next_decision_id: u64,
    daemon_attached: u32,
    gpus: KVec<PolarisGpu>,
    blocks: KVec<PolarisBlock>,
    sessions: KVec<PolarisSession>,
    pending_decisions: KVec<PolarisDecision>,
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
    for b in &inner.blocks {
        match b.state {
            PolarisBlockState::Resident => resident += 1,
            PolarisBlockState::CpuOffloaded => offloaded += 1,
            PolarisBlockState::Evicted => evicted += 1,
            _ => pending += 1,
        }
        if b.flags.contains(PolarisBlockFlag::Shared) {
            shared += b.size_bytes;
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

        inner.sessions.push(
            PolarisSession {
                session_id,
                home_gpu: arg.home_gpu,
                gpu_vas_base: 0,
                gpu_vas_size: arg.gpu_vas_bytes,
                gpu_vas_cursor: 0,
                beam_width: arg.beam_width,
                bytes_per_token: bpt,
                parent_session_id: 0,
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

        // Check session exists before trying to remove.
        if !inner.sessions.iter().any(|s| s.session_id == sid) {
            return Err(ENOENT);
        }

        // Decrement used_bytes for Resident blocks belonging to this session.
        let gpu_id = inner.sessions.iter()
            .find(|s| s.session_id == sid)
            .map(|s| s.home_gpu)
            .unwrap_or(0);
        let mut freed: u64 = 0;
        for block in inner.blocks.iter() {
            if block.session_id == sid && block.state == PolarisBlockState::Resident {
                freed += block.size_bytes;
            }
        }

        inner.sessions.retain(|s| s.session_id != sid);
        inner.blocks.retain(|b| b.session_id != sid);

        if freed > 0 {
            if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                gpu.used_bytes = gpu.used_bytes.saturating_sub(freed);
            }
        }

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
        let parent_vas_cursor = parent.gpu_vas_cursor;
        let parent_beam = parent.beam_width;
        let parent_bpt = parent.bytes_per_token;
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
                gpu_vas_cursor: parent_vas_cursor,
                beam_width: parent_beam,
                bytes_per_token: parent_bpt,
                parent_session_id: parent_id,
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
        let inner = guard.as_mut().ok_or(ENODEV)?;

        // Gate: refuse allocation when no daemon is attached.
        if inner.daemon_attached == 0 {
            return Err(ENODEV);
        }

        if !inner.sessions.iter().any(|s| s.session_id == arg.session_id) {
            return Err(ENOENT);
        }

        let block_id = inner.next_block_id;
        inner.next_block_id += 1;

        // G4: check GPU health for the session's home GPU and extract settings.
        let (gpu_id, bpt) = {
            let sess = inner.sessions.iter().find(|s| s.session_id == arg.session_id);
            if let Some(sess) = sess {
                if let Some(gpu) = inner.gpus.iter().find(|g| g.gpu_id == sess.home_gpu) {
                    if !gpu.healthy {
                        dev_err!(self.dev, "POLARIS: GPU {} unhealthy, rejecting BLOCK_GROW\n", sess.home_gpu);
                        return Err(ENODEV);
                    }
                }
                (sess.home_gpu, sess.bytes_per_token)
            } else {
                (0, POLARIS_DEFAULT_BYTES_PER_TOKEN)
            }
        };

        let size_bytes = (arg.token_count as u64) * bpt;

        // G4: check GPU budget before allocating.
        if let Some(gpu) = inner.gpus.iter().find(|g| g.gpu_id == gpu_id) {
            if gpu.used_bytes + size_bytes > gpu.budget_bytes {
                dev_err!(self.dev, "POLARIS: GPU {} budget exceeded on BLOCK_GROW\n", gpu_id);
                return Err(ENOMEM);
            }
        }

        let dec_id = inner.next_decision_id;
        inner.next_decision_id += 1;

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
            phase: PolarisPhase::Prefill,
            last_touch_ns: 0,
            map_time_ns: 0,
            retry_count: 0,
            pending_decision_id: dec_id,
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

        arg.block_id = block_id;
        arg.ret_code = 0;
        drop(guard);
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        dev_info!(self.dev, "POLARIS: block {} allocated for session {}\n", block_id, arg.session_id);
        Ok(0)
    }

    fn handle_block_free(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisBlockFreeArg = reader.read()?;
        let mut guard = POLARIS_STATE.lock();
        let inner = guard.as_mut().ok_or(ENODEV)?;

        // Find the block by session_id and token_start.
        let block_idx = inner
            .blocks
            .iter()
            .position(|b| b.session_id == arg.session_id && b.token_start == arg.token_start)
            .ok_or(ENOENT)?;

        // Collect info before mutation.
        let block_id = inner.blocks[block_idx].block_id;
        let home_gpu = inner.blocks[block_idx].home_gpu;
        let session_id = inner.blocks[block_idx].session_id;
        let phys_handle = inner.blocks[block_idx].gpu_phys_handle;
        let size_bytes = inner.blocks[block_idx].size_bytes;
        let current_state = inner.blocks[block_idx].state;

        // Decrement COW refcount.
        inner.blocks[block_idx].refcount -= 1;

        if inner.blocks[block_idx].refcount == 0 {
            // No more sessions reference this block — release GPU resources.
            if inner.daemon_attached > 0 && phys_handle != 0 {
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
                // No daemon or never mapped: remove the block directly.
                if current_state == PolarisBlockState::Resident {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == home_gpu) {
                        gpu.used_bytes = gpu.used_bytes.saturating_sub(size_bytes);
                    }
                }
                let _ = inner.blocks.remove(block_idx);
                dev_info!(self.dev, "POLARIS: block {} freed directly\n", block_id);
            }
        } else {
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
        for block in inner.blocks.iter_mut() {
            if block.session_id == arg.session_id {
                let start = block.token_start as u64;
                let end = start + block.token_count as u64;
                if start >= arg.token_start && end <= arg.token_start + arg.token_count {
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

        match inner
            .blocks
            .iter()
            .find(|b| b.session_id == arg.session_id && b.token_start == arg.token_start)
        {
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

        for i in 0..count {
            let block_id = arg.decisions[i].block_id;
            if let Some(block) = inner.blocks.iter_mut().find(|b| b.block_id == block_id) {
                block.last_touch_ns = 0;
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
            {
                let block = &mut inner.blocks[block_idx];
                block.retry_count = 0;
                block.pending_decision_id = 0;
                prev_state = block.state;
                gpu_id = block.home_gpu;
                sz = block.size_bytes;
                match block.state {
                    PolarisBlockState::AllocPending => {
                        block.state = PolarisBlockState::Resident;
                        block.gpu_phys_handle = arg.output_handle;
                    }
                    PolarisBlockState::OffloadPending => {
                        block.state = PolarisBlockState::CpuOffloaded;
                        block.cpu_buf_addr = arg.output_cpu_addr;
                    }
                    PolarisBlockState::ReloadPending | PolarisBlockState::CowPending => {
                        block.state = PolarisBlockState::Resident;
                        block.gpu_phys_handle = arg.output_handle;
                    }
                    PolarisBlockState::FreePending => {
                        block.state = PolarisBlockState::Evicted;
                    }
                    _ => {}
                }
            }

            // Update GPU used_bytes based on the state transition.
            match prev_state {
                PolarisBlockState::AllocPending
                | PolarisBlockState::ReloadPending
                | PolarisBlockState::CowPending => {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                        gpu.used_bytes += sz;
                    }
                }
                PolarisBlockState::OffloadPending | PolarisBlockState::FreePending => {
                    if let Some(gpu) = inner.gpus.iter_mut().find(|g| g.gpu_id == gpu_id) {
                        gpu.used_bytes = gpu.used_bytes.saturating_sub(sz);
                    }
                }
                _ => {}
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

        Ok(0)
    }

    /// Re-queue an ALLOC decision for a block (G4 retry path).
    fn requeue_decision(&self, inner: &mut PolarisInner, block_idx: usize) {
        let block = &mut inner.blocks[block_idx];
        let dec_id = inner.next_decision_id;
        inner.next_decision_id += 1;
        block.pending_decision_id = dec_id;

        let _ = inner.pending_decisions.push(
            PolarisDecision {
                decision_id: dec_id,
                op: PolarisDecisionOp::Alloc as u32,
                gpu_id: block.home_gpu,
                block_id: block.block_id,
                session_id: block.session_id,
                src_handle: 0,
                dst_vaddr: 0,
                size_bytes: block.size_bytes,
                cpu_addr: 0,
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
        drop(guard);

        arg.blocks_resident = resident;
        arg.blocks_offloaded = offloaded;
        arg.blocks_evicted = evicted;
        arg.shared_gpu_bytes = shared;
        arg.private_gpu_bytes = private;
        arg.total_gpu_bytes = total_gpu;
        arg.used_gpu_bytes = used_gpu;
        arg.cpu_pool_total = cpu_total;
        arg.cpu_pool_used = cpu_used;

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }
}
