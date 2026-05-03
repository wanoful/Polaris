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

use crate::polaris_types::*;
use crate::PolarisInner;
use kernel::bindings;

/// Select a victim block for offload based on the current eviction policy.
///
/// Returns `(index_in_blocks, block_id, size_bytes, gpu_id)` or `None`
/// if no eligible candidate exists.
pub fn select_victim(
    inner: &PolarisInner,
    requesting_session_id: u64,
) -> Option<(usize, u64, u64, u32)> {
    match inner.eviction_policy {
        PolarisEvictionPolicy::Fifo => find_victim_fifo(inner, requesting_session_id),
        PolarisEvictionPolicy::Lru => find_victim_lru(inner, requesting_session_id),
        PolarisEvictionPolicy::PhaseAware => find_victim_phase_aware(inner, requesting_session_id),
    }
}

// ─── Eligibility check shared by FIFO and LRU ───────────────────────────────

/// Returns true if the block is eligible for eviction under FIFO/LRU.
/// Hard-filters: Resident, refcount <= 1, not pending, fits in CPU pool.
fn is_eligible_fifo_lru(block: &PolarisBlock, inner: &PolarisInner) -> bool {
    if block.state != PolarisBlockState::Resident {
        return false;
    }
    if block.refcount > 1 {
        return false;
    }
    if block.pending_decision_id != 0 {
        return false;
    }
    // Must fit in CPU pool.
    if let Some(gpu) = inner.gpus.iter().find(|g| g.gpu_id == block.home_gpu) {
        if gpu.cpu_pool_used_bytes.saturating_add(block.size_bytes) > gpu.cpu_pool_total_bytes {
            return false;
        }
        true
    } else {
        false
    }
}

/// Returns true if the block is eligible for eviction under Phase-Aware.
/// Softer: allows shared blocks (scoring protects them).
fn is_eligible_phase_aware(block: &PolarisBlock, inner: &PolarisInner) -> bool {
    if block.state != PolarisBlockState::Resident {
        return false;
    }
    if block.pending_decision_id != 0 {
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
) -> Option<(usize, u64, u64, u32)> {
    // Pass 1: other sessions only.
    let mut best: Option<(usize, u64, u64, u32)> = None;
    let mut best_time: u64 = u64::MAX;

    for (idx, block) in inner.blocks.iter().enumerate() {
        if block.session_id == requesting_session_id {
            continue;
        }
        if !is_eligible_fifo_lru(block, inner) {
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

    // Pass 2: any session.
    best_time = u64::MAX;
    for (idx, block) in inner.blocks.iter().enumerate() {
        if !is_eligible_fifo_lru(block, inner) {
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
) -> Option<(usize, u64, u64, u32)> {
    // Pass 1: other sessions only.
    let mut best: Option<(usize, u64, u64, u32)> = None;
    let mut best_time: u64 = u64::MAX;

    for (idx, block) in inner.blocks.iter().enumerate() {
        if block.session_id == requesting_session_id {
            continue;
        }
        if !is_eligible_fifo_lru(block, inner) {
            continue;
        }
        if block.last_touch_ns < best_time {
            best_time = block.last_touch_ns;
            best = Some((idx, block.block_id, block.size_bytes, block.home_gpu));
        }
    }

    if best.is_some() {
        return best;
    }

    // Pass 2: any session.
    best_time = u64::MAX;
    for (idx, block) in inner.blocks.iter().enumerate() {
        if !is_eligible_fifo_lru(block, inner) {
            continue;
        }
        if block.last_touch_ns < best_time {
            best_time = block.last_touch_ns;
            best = Some((idx, block.block_id, block.size_bytes, block.home_gpu));
        }
    }
    best
}

// ─── Phase-Aware: scoring function ──────────────────────────────────────────
//
// victim_score =
//     age_weight       × age_seconds
//   + prefill_weight   × is_prefill_block
//   - sharing_weight   × refcount
//   - decode_weight    × recent_decode_access
//
// Higher score → more likely victim.

const AGE_WEIGHT: i64 = 1;
const PREFILL_WEIGHT: i64 = 100;
const SHARING_WEIGHT: i64 = 1000;
const DECODE_WEIGHT: i64 = 500;

/// Score the block. Higher = more evictable.
fn phase_aware_score(block: &PolarisBlock, now: u64) -> i64 {
    // Age in seconds (saturating to avoid overflow on very old timestamps).
    let age_ns = now.saturating_sub(block.last_touch_ns);
    let age_sec = (age_ns / 1_000_000_000u64) as i64;

    let is_prefill = if block.phase == PolarisPhase::Prefill {
        1i64
    } else {
        0i64
    };
    let refcount = block.refcount as i64;
    // "Recent decode access": decode block touched within the last 1 second.
    let recent_decode = if block.phase == PolarisPhase::Decode && age_ns < 1_000_000_000u64 {
        1i64
    } else {
        0i64
    };

    AGE_WEIGHT * age_sec
        + PREFILL_WEIGHT * is_prefill
        - SHARING_WEIGHT * refcount
        - DECODE_WEIGHT * recent_decode
}

fn find_victim_phase_aware(
    inner: &PolarisInner,
    requesting_session_id: u64,
) -> Option<(usize, u64, u64, u32)> {
    let now = unsafe { bindings::ktime_get_mono_fast_ns() };

    // Pass 1: other sessions only.
    let mut best: Option<(usize, u64, u64, u32, i64)> = None;
    let mut best_score: i64 = i64::MIN;

    for (idx, block) in inner.blocks.iter().enumerate() {
        if block.session_id == requesting_session_id {
            continue;
        }
        if !is_eligible_phase_aware(block, inner) {
            continue;
        }
        let score = phase_aware_score(block, now);
        if score > best_score {
            best_score = score;
            best = Some((idx, block.block_id, block.size_bytes, block.home_gpu, score));
        }
    }

    if best.is_some() {
        return best.map(|(a, b, c, d, _)| (a, b, c, d));
    }

    // Pass 2: any session.
    best_score = i64::MIN;
    for (idx, block) in inner.blocks.iter().enumerate() {
        if !is_eligible_phase_aware(block, inner) {
            continue;
        }
        let score = phase_aware_score(block, now);
        if score > best_score {
            best_score = score;
            best = Some((idx, block.block_id, block.size_bytes, block.home_gpu, score));
        }
    }

    best.map(|(a, b, c, d, _)| (a, b, c, d))
}
