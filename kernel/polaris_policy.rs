// SPDX-License-Identifier: GPL-2.0

//! Eviction policy implementations for Phase 2b.
//!
//! Three policies, selectable at runtime via the POLARIS_SET_POLICY ioctl:
//!
//!   FIFO        — Victim = block with the oldest map_time_ns
//!   LRU         — Victim = block with the oldest last_touch_ns
//!   PhaseAware  — Scoring function: prefill blocks preferred, shared/decode protected
//!
//! All policies use a two-pass search:
//!   Pass 1: only blocks from OTHER sessions (preferred)
//!   Pass 2: any session's blocks (fallback)
//!
//! Block ownership for COW child sessions is determined by the session's
//! block_ids list, not by block.session_id (which stays the original owner's ID).

use crate::polaris_types::*;
use crate::PolarisInner;
use kernel::bindings;

/// Check whether a block belongs to a session, covering both directly-owned
/// and COW-shared blocks (where block.session_id is the parent's ID).
fn block_owned_by_session(inner: &PolarisInner, block: &PolarisBlock, session_id: u64) -> bool {
    if block.session_id == session_id {
        return true;
    }
    inner.sessions.iter()
        .find(|s| s.session_id == session_id)
        .map(|s| s.block_ids.iter().any(|&bid| bid == block.block_id))
        .unwrap_or(false)
}

/// Look up the requesting session's home GPU. Returns 0 if not found.
fn session_home_gpu(inner: &PolarisInner, session_id: u64) -> u32 {
    inner.sessions.iter()
        .find(|s| s.session_id == session_id)
        .map(|s| s.home_gpu)
        .unwrap_or(0)
}

/// Select a victim block for offload based on the current eviction policy.
///
/// `target_gpu` restricts the search to blocks on a specific GPU (the
/// requesting session's home GPU).  Returns `(index_in_blocks, block_id,
/// size_bytes, gpu_id)` or `None` if no eligible candidate exists.
pub fn select_victim(
    inner: &PolarisInner,
    requesting_session_id: u64,
    target_gpu: u32,
    protected_phys_handle: u64,
    protected_block_id: u64,
) -> Option<(usize, u64, u64, u32)> {
    match inner.eviction_policy {
        PolarisEvictionPolicy::Fifo => {
            find_victim_fifo(
                inner,
                requesting_session_id,
                target_gpu,
                protected_phys_handle,
                protected_block_id,
            )
        }
        PolarisEvictionPolicy::Lru => {
            find_victim_lru(
                inner,
                requesting_session_id,
                target_gpu,
                protected_phys_handle,
                protected_block_id,
            )
        }
        PolarisEvictionPolicy::PhaseAware => find_victim_phase_aware(
            inner,
            requesting_session_id,
            target_gpu,
            protected_phys_handle,
            protected_block_id,
        ),
    }
}

// ─── Eligibility checks ──────────────────────────────────────────────────────

/// Returns true if the block is eligible for eviction under FIFO/LRU.
/// Hard-filters: Resident, refcount <= 1, not pending, fits in CPU pool,
/// matches target GPU.
fn is_eligible_fifo_lru(block: &PolarisBlock, inner: &PolarisInner, target_gpu: u32) -> bool {
    if block.state != PolarisBlockState::Resident {
        return false;
    }
    if block.refcount > 1 {
        return false;
    }
    if block.pending_decision_id != 0 {
        return false;
    }
    if block.home_gpu != target_gpu {
        return false;
    }
    if let Some(gpu) = inner.gpus.iter().find(|g| g.gpu_id == block.home_gpu) {
        if gpu.cpu_pool_used_bytes.saturating_add(block.size_bytes) > gpu.cpu_pool_total_bytes {
            return false;
        }
        true
    } else {
        false
    }
}

fn is_protected_source(
    block: &PolarisBlock,
    protected_phys_handle: u64,
    protected_block_id: u64,
) -> bool {
    (protected_block_id != 0 && block.block_id == protected_block_id)
        || (protected_phys_handle != 0 && block.gpu_phys_handle == protected_phys_handle)
}

/// Returns true if the block is eligible for eviction under Phase-Aware.
/// Softer filter: allows shared blocks (scoring protects them).
fn is_eligible_phase_aware(block: &PolarisBlock, inner: &PolarisInner, target_gpu: u32) -> bool {
    if block.state != PolarisBlockState::Resident {
        return false;
    }
    if block.pending_decision_id != 0 {
        return false;
    }
    if block.home_gpu != target_gpu {
        return false;
    }
    if let Some(gpu) = inner.gpus.iter().find(|g| g.gpu_id == block.home_gpu) {
        if gpu.cpu_pool_used_bytes.saturating_add(block.size_bytes) > gpu.cpu_pool_total_bytes {
            return false;
        }
        true
    } else {
        false
    }
}

// ─── FIFO: victim = oldest map_time_ns ──────────────────────────────────────

fn find_victim_fifo(
    inner: &PolarisInner,
    requesting_session_id: u64,
    target_gpu: u32,
    protected_phys_handle: u64,
    protected_block_id: u64,
) -> Option<(usize, u64, u64, u32)> {
    // Pass 1: other sessions only.
    let mut best: Option<(usize, u64, u64, u32)> = None;
    let mut best_time: u64 = u64::MAX;

    for (idx, block) in inner.blocks.iter().enumerate() {
        if block_owned_by_session(inner, block, requesting_session_id) {
            continue;
        }
        if is_protected_source(block, protected_phys_handle, protected_block_id) {
            continue;
        }
        if !is_eligible_fifo_lru(block, inner, target_gpu) {
            continue;
        }
        if block.map_time_ns < best_time {
            best_time = block.map_time_ns;
            best = Some((idx, block.block_id, block.size_bytes, block.home_gpu));
        }
    }

    if best.is_some() {
        return best;
    }

    // Pass 2: any session (including the requesting one).
    best_time = u64::MAX;
    for (idx, block) in inner.blocks.iter().enumerate() {
        if is_protected_source(block, protected_phys_handle, protected_block_id) {
            continue;
        }
        if !is_eligible_fifo_lru(block, inner, target_gpu) {
            continue;
        }
        if block.map_time_ns < best_time {
            best_time = block.map_time_ns;
            best = Some((idx, block.block_id, block.size_bytes, block.home_gpu));
        }
    }
    best
}

// ─── LRU: victim = oldest last_touch_ns ─────────────────────────────────────

fn find_victim_lru(
    inner: &PolarisInner,
    requesting_session_id: u64,
    target_gpu: u32,
    protected_phys_handle: u64,
    protected_block_id: u64,
) -> Option<(usize, u64, u64, u32)> {
    // Pass 1: other sessions only.
    let mut best: Option<(usize, u64, u64, u32)> = None;
    let mut best_time: u64 = u64::MAX;

    for (idx, block) in inner.blocks.iter().enumerate() {
        if block_owned_by_session(inner, block, requesting_session_id) {
            continue;
        }
        if is_protected_source(block, protected_phys_handle, protected_block_id) {
            continue;
        }
        if !is_eligible_fifo_lru(block, inner, target_gpu) {
            continue;
        }
        let touch_time = if block.last_touch_ns != 0 {
            block.last_touch_ns
        } else {
            block.map_time_ns
        };
        if touch_time < best_time {
            best_time = touch_time;
            best = Some((idx, block.block_id, block.size_bytes, block.home_gpu));
        }
    }

    if best.is_some() {
        return best;
    }

    // Pass 2: any session (including the requesting one).
    best_time = u64::MAX;
    for (idx, block) in inner.blocks.iter().enumerate() {
        if is_protected_source(block, protected_phys_handle, protected_block_id) {
            continue;
        }
        if !is_eligible_fifo_lru(block, inner, target_gpu) {
            continue;
        }
        let touch_time = if block.last_touch_ns != 0 {
            block.last_touch_ns
        } else {
            block.map_time_ns
        };
        if touch_time < best_time {
            best_time = touch_time;
            best = Some((idx, block.block_id, block.size_bytes, block.home_gpu));
        }
    }
    best
}

// ─── Phase-Aware: scoring function ──────────────────────────────────────────
//
// victim_score =
//     age_weight         × age_seconds
//   + prefill_weight     × is_prefill_block
//   + pressure_weight    × gpu_pressure_norm
//   + priority_weight    × inverse_session_priority
//   - sharing_weight     × refcount
//   - decode_weight      × recent_decode_access
//
// Higher score → more likely victim.

const AGE_WEIGHT: i64 = 1;
const PREFILL_WEIGHT: i64 = 100;
const PRESSURE_WEIGHT: i64 = 1;
const PRIORITY_WEIGHT: i64 = 50;
const SHARING_WEIGHT: i64 = 1000;
const DECODE_WEIGHT: i64 = 500;

/// Score the block. Higher = more evictable.
fn phase_aware_score(
    block: &PolarisBlock,
    now: u64,
    gpu_pressure: u64,
    session_priority: u32,
) -> i64 {
    // Age in seconds (saturating to avoid overflow on very old timestamps).
    let touch_time = if block.last_touch_ns != 0 {
        block.last_touch_ns
    } else {
        block.map_time_ns
    };
    let age_ns = now.saturating_sub(touch_time);
    let age_sec = (age_ns / 1_000_000_000u64) as i64;

    let is_prefill = if block.phase == PolarisPhase::Prefill { 1i64 } else { 0i64 };
    let refcount = block.refcount as i64;
    let recent_decode = if block.phase == PolarisPhase::Decode && age_ns < 1_000_000_000u64 { 1i64 } else { 0i64 };

    // Normalize gpu_pressure: clamp 0..1000 → 0..100
    let pressure_norm = (gpu_pressure.min(1000) / 10) as i64;

    // Inverse session priority: 1 (highest)..10 (lowest), inverse = 11 − priority.
    // Default priority = 5.
    let prio = if session_priority > 0 && session_priority <= 10 {
        session_priority
    } else {
        5u32
    };
    let inv_priority = (11u32.saturating_sub(prio)) as i64;

    AGE_WEIGHT * age_sec
        + PREFILL_WEIGHT * is_prefill
        + PRESSURE_WEIGHT * pressure_norm
        + PRIORITY_WEIGHT * inv_priority
        - SHARING_WEIGHT * refcount
        - DECODE_WEIGHT * recent_decode
}

fn find_victim_phase_aware(
    inner: &PolarisInner,
    requesting_session_id: u64,
    target_gpu: u32,
    protected_phys_handle: u64,
    protected_block_id: u64,
) -> Option<(usize, u64, u64, u32)> {
    let now = unsafe { bindings::ktime_get_mono_fast_ns() };

    // Look up GPU pressure once.
    let gpu_pressure = inner.gpus.iter()
        .find(|g| g.gpu_id == target_gpu)
        .map(|g| g.pressure_score)
        .unwrap_or(0);

    // Pass 1: other sessions only.
    let mut best: Option<(usize, u64, u64, u32, i64)> = None;
    let mut best_score: i64 = i64::MIN;

    for (idx, block) in inner.blocks.iter().enumerate() {
        if block_owned_by_session(inner, block, requesting_session_id) {
            continue;
        }
        if is_protected_source(block, protected_phys_handle, protected_block_id) {
            continue;
        }
        if !is_eligible_phase_aware(block, inner, target_gpu) {
            continue;
        }
        let session_priority = inner.sessions.iter()
            .find(|s| s.session_id == block.session_id)
            .map(|s| s.priority)
            .unwrap_or(5u32);
        let score = phase_aware_score(block, now, gpu_pressure, session_priority);
        if score > best_score {
            best_score = score;
            best = Some((idx, block.block_id, block.size_bytes, block.home_gpu, score));
        }
    }

    if best.is_some() {
        return best.map(|(a, b, c, d, _)| (a, b, c, d));
    }

    // Pass 2: any session (including the requesting one).
    best_score = i64::MIN;
    for (idx, block) in inner.blocks.iter().enumerate() {
        if is_protected_source(block, protected_phys_handle, protected_block_id) {
            continue;
        }
        if !is_eligible_phase_aware(block, inner, target_gpu) {
            continue;
        }
        let session_priority = inner.sessions.iter()
            .find(|s| s.session_id == block.session_id)
            .map(|s| s.priority)
            .unwrap_or(5u32);
        let score = phase_aware_score(block, now, gpu_pressure, session_priority);
        if score > best_score {
            best_score = score;
            best = Some((idx, block.block_id, block.size_bytes, block.home_gpu, score));
        }
    }

    best.map(|(a, b, c, d, _)| (a, b, c, d))
}
