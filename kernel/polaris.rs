// SPDX-License-Identifier: GPL-2.0

//! POLARIS: Paged Operating Layer for Accelerated Routing and Inference Systems
//!
//! This kernel module provides OS-level paged KV Cache management for LLM inference.
//! It maintains the authoritative block table, manages sessions, and issues GPU memory
//! management decisions to the userspace daemon (polarisd) via an ioctl-based
//! decision protocol.
//!
//! Architecture:
//!   /dev/polaris  --  miscdevice, ioctl dispatch
//!     POLARIS_REGISTER_GPU       -- daemon reports GPU capacity
//!     POLARIS_SESSION_CREATE     -- workload creates a session
//!     POLARIS_BLOCK_GROW         -- page-fault: request new KV block
//!     POLARIS_GET_DECISION       -- daemon polls for work
//!     POLARIS_COMPLETE_OPERATION -- daemon reports completion
//!     ... (see polaris_types.rs for full list)

// Note: #![no_std] and #![feature(arbitrary_self_types)] are injected
// by the kernel build system. Do not redeclare them.

mod polaris_types;

use core::pin::Pin;

use kernel::{
    c_str,
    device::Device,
    fs::File,
    ioctl::_IOC_SIZE,
    miscdevice::{MiscDevice, MiscDeviceOptions, MiscDeviceRegistration},
    new_mutex,
    prelude::*,
    sync::{aref::ARef, Mutex},
    uaccess::UserSlice,
};

use polaris_types::*;

// ─── Module declaration ─────────────────────────────────────────────────────

module! {
    type: PolarisModule,
    name: "polaris",
    authors: ["POLARIS Team"],
    description: "OS-level Paged KV Cache Management for LLM Inference",
    license: "GPL",
}

#[pin_data]
struct PolarisModule {
    #[pin]
    _miscdev: MiscDeviceRegistration<PolarisDevice>,
}

impl kernel::InPlaceModule for PolarisModule {
    fn init(_module: &'static ThisModule) -> impl PinInit<Self, Error> {
        pr_info!("POLARIS: initializing kernel module\n");

        let options = MiscDeviceOptions {
            name: c_str!("polaris"),
        };

        try_pin_init!(Self {
            _miscdev <- MiscDeviceRegistration::<PolarisDevice>::register(options),
        })
    }
}

// ─── Per-device inner state ─────────────────────────────────────────────────

struct PolarisInner {
    next_block_id: u64,
    next_session_id: u64,
    next_decision_id: u64,
    gpus: KVec<PolarisGpu>,
    blocks: KVec<PolarisBlock>,
    sessions: KVec<PolarisSession>,
    pending_decisions: KVec<PolarisDecision>,
}

impl PolarisInner {
    fn new() -> Result<Self> {
        Ok(Self {
            next_block_id: 1,
            next_session_id: 1,
            next_decision_id: 1,
            gpus: KVec::new(),
            blocks: KVec::new(),
            sessions: KVec::new(),
            pending_decisions: KVec::new(),
        })
    }
}

// ─── Device implementation ──────────────────────────────────────────────────

#[pin_data(PinnedDrop)]
struct PolarisDevice {
    #[pin]
    inner: Mutex<PolarisInner>,
    dev: ARef<Device>,
}

#[vtable]
impl MiscDevice for PolarisDevice {
    type Ptr = Pin<KBox<Self>>;

    fn open(_file: &File, misc: &MiscDeviceRegistration<Self>) -> Result<Pin<KBox<Self>>> {
        let dev = ARef::from(misc.device());
        dev_info!(dev, "POLARIS: device opened\n");

        KBox::try_pin_init(
            try_pin_init! {
                PolarisDevice {
                    inner <- new_mutex!(PolarisInner::new()?),
                    dev: dev,
                }
            },
            GFP_KERNEL,
        )
    }

    fn ioctl(me: Pin<&PolarisDevice>, _file: &File, cmd: u32, arg: usize) -> Result<isize> {
        let user_ptr = UserPtr::from_addr(arg);
        let size = _IOC_SIZE(cmd);

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
        dev_info!(self.dev, "POLARIS: device closed\n");
    }
}

// ─── IOCTL handler methods ──────────────────────────────────────────────────

impl PolarisDevice {
    fn handle_register_gpu(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisRegisterGpuArg = reader.read()?;
        let mut inner = self.inner.lock();
        inner.gpus.push(
            PolarisGpu {
                gpu_id: arg.gpu_id,
                total_bytes: arg.total_bytes,
                used_bytes: 0,
                budget_bytes: arg.budget_bytes,
                pressure_score: 0,
                cpu_pool_total_bytes: arg.cpu_pool_bytes,
                cpu_pool_used_bytes: 0,
            },
            GFP_KERNEL,
        )?;
        dev_info!(self.dev, "POLARIS: GPU {} registered\n", arg.gpu_id);
        Ok(0)
    }

    fn handle_session_create(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisSessionCreateArg = reader.read()?;
        let mut inner = self.inner.lock();

        let session_id = inner.next_session_id;
        inner.next_session_id += 1;

        inner.sessions.push(
            PolarisSession {
                session_id,
                home_gpu: arg.home_gpu,
                gpu_vas_base: 0,
                gpu_vas_size: arg.gpu_vas_bytes,
                gpu_vas_cursor: 0,
                beam_width: arg.beam_width,
                parent_session_id: 0,
                block_ids: KVec::new(),
            },
            GFP_KERNEL,
        )?;

        arg.session_id = session_id;
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        dev_info!(self.dev, "POLARIS: session {} created\n", session_id);
        Ok(0)
    }

    fn handle_session_destroy(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisSessionDestroyArg = reader.read()?;
        let mut inner = self.inner.lock();
        let sid = arg.session_id;
        inner.sessions.retain(|s| s.session_id != sid);
        inner.blocks.retain(|b| b.session_id != sid);
        dev_info!(self.dev, "POLARIS: session {} destroyed\n", sid);
        Ok(0)
    }

    fn handle_session_get_stats(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisSessionGetStatsArg = reader.read()?;
        let inner = self.inner.lock();

        match inner.sessions.iter().find(|s| s.session_id == arg.session_id) {
            Some(session) => {
                arg.home_gpu = session.home_gpu;
                arg.beam_width = session.beam_width;
                arg.num_blocks = session.block_ids.len() as u32;
                arg.total_bytes = session.gpu_vas_size;
            }
            None => return Err(ENOENT),
        }

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_session_branch(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisSessionBranchArg = reader.read()?;
        let mut inner = self.inner.lock();

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
        let parent_block_ids: KVec<u64> = {
            let mut ids = KVec::new();
            for &bid in &parent.block_ids {
                ids.push(bid, GFP_KERNEL)?;
            }
            ids
        };
        drop(parent); // end immutable borrow of inner.sessions

        // COW: increment refcount on all parent blocks.
        for &bid in &parent_block_ids {
            if let Some(block) = inner.blocks.iter_mut().find(|b| b.block_id == bid) {
                block.refcount += 1;
                block.flags |= POLARIS_BLOCK_FLAG_SHARED;
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
                parent_session_id: arg.parent_session_id,
                block_ids: parent_block_ids,
            },
            GFP_KERNEL,
        )?;

        arg.child_session_id = child_id;
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        dev_info!(self.dev, "POLARIS: session {} branched from {}\n", child_id, arg.parent_session_id);
        Ok(0)
    }

    fn handle_block_grow(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisBlockGrowArg = reader.read()?;
        let mut inner = self.inner.lock();

        // Validate session exists.
        if !inner.sessions.iter().any(|s| s.session_id == arg.session_id) {
            return Err(ENOENT);
        }

        let block_id = inner.next_block_id;
        inner.next_block_id += 1;

        let block = PolarisBlock {
            block_id,
            session_id: arg.session_id,
            token_start: arg.token_start,
            token_count: arg.token_count,
            home_gpu: 0,
            gpu_vaddr: 0,
            gpu_phys_handle: 0,
            cpu_buf_addr: 0,
            size_bytes: 0,
            refcount: 1,
            state: PolarisBlockState::AllocPending,
            flags: 0,
            phase: POLARIS_PHASE_PREFILL,
            last_touch_ns: 0,
            map_time_ns: 0,
        };

        // Queue a decision for the daemon.
        let dec_id = inner.next_decision_id;
        inner.next_decision_id += 1;
        inner.pending_decisions.push(
            PolarisDecision {
                decision_id: dec_id,
                op: PolarisDecisionOp::Alloc as u32,
                gpu_id: 0,
                block_id,
                session_id: arg.session_id,
                src_handle: 0,
                dst_vaddr: 0,
                size_bytes: block.size_bytes,
                cpu_addr: 0,
                _reserved: [0u64; 4],
            },
            GFP_KERNEL,
        )?;
        inner.blocks.push(block, GFP_KERNEL)?;

        // Link block to session.
        if let Some(session) = inner.sessions.iter_mut().find(|s| s.session_id == arg.session_id) {
            session.block_ids.push(block_id, GFP_KERNEL)?;
        }

        arg.block_id = block_id;
        arg.ret_code = 0;
        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        dev_info!(self.dev, "POLARIS: block {} allocated for session {}\n", block_id, arg.session_id);
        Ok(0)
    }

    fn handle_block_free(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisBlockFreeArg = reader.read()?;
        let mut inner = self.inner.lock();
        inner.blocks.retain(|b| {
            !(b.session_id == arg.session_id && b.token_start == arg.token_start)
        });
        Ok(0)
    }

    fn handle_block_touch(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisBlockTouchArg = reader.read()?;
        let mut inner = self.inner.lock();
        // TODO: use ktime_get_ns()
        let now_ns: u64 = 0;
        for block in inner.blocks.iter_mut() {
            if block.session_id == arg.session_id {
                let start = block.token_start as u64;
                let end = start + block.token_count as u64;
                if start >= arg.token_start && end <= arg.token_start + arg.token_count {
                    block.last_touch_ns = now_ns;
                }
            }
        }
        Ok(0)
    }

    fn handle_block_get_state(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisBlockGetStateArg = reader.read()?;
        let inner = self.inner.lock();

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

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_get_decision(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisGetDecisionArg = reader.read()?;
        let mut inner = self.inner.lock();

        let count = core::cmp::min(inner.pending_decisions.len(), POLARIS_MAX_DECISIONS_PER_POLL);
        arg.count = count as u32;

        // Copy decisions into the output struct, then clear consumed ones.
        for i in 0..count {
            arg.decisions[i] = inner.pending_decisions[i];
        }
        // Remove consumed decisions.
        if count < inner.pending_decisions.len() {
            // Shift remaining decisions to front.
            let remaining = inner.pending_decisions.len() - count;
            for i in 0..remaining {
                inner.pending_decisions[i] = inner.pending_decisions[count + i];
            }
            inner.pending_decisions.truncate(remaining);
        } else {
            inner.pending_decisions.clear();
        }

        // Mark blocks that had decisions dispatched.
        for i in 0..count {
            let block_id = arg.decisions[i].block_id;
            if let Some(block) = inner.blocks.iter_mut().find(|b| b.block_id == block_id) {
                block.last_touch_ns = 0; // mark as dispatched
            }
        }

        let mut writer = UserSlice::new(user_ptr, size).writer();
        writer.write(&arg)?;
        Ok(0)
    }

    fn handle_complete_operation(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let arg: PolarisCompleteOperationArg = reader.read()?;
        let mut inner = self.inner.lock();

        if arg.result != 0 {
            dev_err!(
                self.dev,
                "POLARIS: operation {} failed with {}\n",
                arg.decision_id,
                arg.result
            );
            return Ok(0);
        }

        // Transition any AllocPending block to Resident with the returned handle.
        for block in inner.blocks.iter_mut() {
            if block.state == PolarisBlockState::AllocPending {
                block.state = PolarisBlockState::Resident;
                block.gpu_phys_handle = arg.output_handle;
                break;
            }
        }
        dev_info!(self.dev, "POLARIS: operation {} completed\n", arg.decision_id);
        Ok(0)
    }

    fn handle_get_global_stats(&self, user_ptr: UserPtr, size: usize) -> Result<isize> {
        let mut reader = UserSlice::new(user_ptr, size).reader();
        let mut arg: PolarisGetGlobalStatsArg = reader.read()?;
        let inner = self.inner.lock();

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
            if block.flags & POLARIS_BLOCK_FLAG_SHARED != 0 {
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
