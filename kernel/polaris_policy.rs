// SPDX-License-Identifier: GPL-2.0

//! Eviction policy implementations for Phase 2b.
//!
//! Three policies, selectable at runtime via the POLARIS_SET_POLICY ioctl:
//!
//!   FIFO        — Victim = block with the oldest map_time_ns
//!   LRU         — Victim = block with the oldest last_touch_ns
//!   PhaseAware  — Scoring function: active KV chunk protected, shared/decode protected
//!   AttentionStream — llama.cpp KV active-window reuse-distance policy
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
        PolarisEvictionPolicy::AttentionStream => find_victim_attention_stream(
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

fn phase_aware_should_use_fifo_order(inner: &PolarisInner, target_gpu: u32) -> bool {
    let mut saw_candidate = false;

    for block in inner.blocks.iter() {
        if !is_eligible_phase_aware(block, inner, target_gpu) {
            continue;
        }
        saw_candidate = true;

        if block.phase != PolarisPhase::Prefill || block.refcount > 1 {
            return false;
        }
    }

    saw_candidate
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
const MAX_AGE_SCORE_MS: u64 = 120_000;
const ACTIVE_AHEAD_PROTECT_CHUNKS: u32 = 4;
const ACTIVE_BEHIND_PROTECT_CHUNKS: u32 = 4;
const ACTIVE_WINDOW_PROTECT_SCORE: i64 = 500_000;
const HINTED_ACTIVE_READ_PROTECT_SCORE: i64 = 750_000;
const HINTED_ACTIVE_WRITE_PROTECT_SCORE: i64 = 1_000_000;
const HINTED_ACTIVE_NEAR_PROTECT_SCORE: i64 = 250_000;
const HINTED_ACTIVE_NEAR_TOKENS: u64 = 4;
const ATTENTION_STREAM_LOOKAHEAD_TOKENS: u64 = 8;
const ATTENTION_STREAM_SUFFIX_TOKENS: u64 = 16;
const ATTENTION_STREAM_OUTSIDE_WINDOW_SCORE: i64 = 2_000_000;
const ATTENTION_STREAM_PAST_SCORE: i64 = 1_000_000;
const ATTENTION_STREAM_OTHER_IDLE_SCORE: i64 = 250_000;
const ATTENTION_STREAM_FUTURE_SCALE: u64 = 4;
const ATTENTION_STREAM_SHARED_PENALTY: i64 = 750_000;
const ATTENTION_STREAM_DECODE_SUFFIX_PENALTY: i64 = 600_000;
const ATTENTION_STREAM_LOOKAHEAD_PENALTY: i64 = 1_000_000;

#[derive(Clone, Copy)]
struct ActiveChunkWindow {
    session_id: u64,
    start: u32,
    end: u32,
}

#[derive(Clone, Copy)]
struct ActiveKvWindow {
    session_id: u64,
    read_start: u64,
    read_end: u64,
    write_start: u64,
    write_end: u64,
}

fn active_chunk_window(
    inner: &PolarisInner,
    requesting_session_id: u64,
    protected_block_id: u64,
) -> Option<ActiveChunkWindow> {
    if protected_block_id == 0 {
        return None;
    }
    inner.blocks.iter()
        .find(|b| b.block_id == protected_block_id)
        .map(|b| {
            let session_id = if block_owned_by_session(inner, b, requesting_session_id) {
                requesting_session_id
            } else {
                b.session_id
            };
            ActiveChunkWindow {
                session_id,
                start: b.token_start,
                end: b.token_start.saturating_add(b.token_count),
            }
        })
}

fn active_kv_window(inner: &PolarisInner, session_id: u64) -> Option<ActiveKvWindow> {
    let session = inner.sessions.iter().find(|s| s.session_id == session_id)?;
    let read_end = session
        .active_read_start_token
        .saturating_add(session.active_read_token_count);
    let write_end = session
        .active_write_start_token
        .saturating_add(session.active_write_token_count);

    if session.active_read_token_count == 0 && session.active_write_token_count == 0 {
        return None;
    }

    Some(ActiveKvWindow {
        session_id,
        read_start: session.active_read_start_token,
        read_end,
        write_start: session.active_write_start_token,
        write_end,
    })
}

fn ranges_overlap(a_start: u64, a_end: u64, b_start: u64, b_end: u64) -> bool {
    a_start < b_end && b_start < a_end
}

fn active_window_score(
    inner: &PolarisInner,
    block: &PolarisBlock,
    active: Option<ActiveChunkWindow>,
) -> i64 {
    let Some(active) = active else {
        return 0;
    };
    if !block_owned_by_session(inner, block, active.session_id) {
        return 0;
    }

    let candidate_start = block.token_start;
    let candidate_end = block.token_start.saturating_add(block.token_count);

    if candidate_start >= active.end {
        let ahead = candidate_start.saturating_sub(active.end);
        if ahead < ACTIVE_AHEAD_PROTECT_CHUNKS {
            return -ACTIVE_WINDOW_PROTECT_SCORE;
        }
    } else if candidate_end <= active.start {
        let behind = active.start.saturating_sub(candidate_end);
        if behind < ACTIVE_BEHIND_PROTECT_CHUNKS {
            return -ACTIVE_WINDOW_PROTECT_SCORE;
        }
    } else {
        return -ACTIVE_WINDOW_PROTECT_SCORE;
    }

    0
}

fn hinted_active_window_score(
    inner: &PolarisInner,
    block: &PolarisBlock,
    active: Option<ActiveKvWindow>,
) -> i64 {
    let Some(active) = active else {
        return 0;
    };
    if !block_owned_by_session(inner, block, active.session_id) {
        return 0;
    }

    let block_start = block.token_start as u64;
    let block_end = block_start.saturating_add(block.token_count as u64);

    if active.write_start < active.write_end
        && ranges_overlap(block_start, block_end, active.write_start, active.write_end)
    {
        return -HINTED_ACTIVE_WRITE_PROTECT_SCORE;
    }

    if active.read_start < active.read_end
        && ranges_overlap(block_start, block_end, active.read_start, active.read_end)
    {
        return -HINTED_ACTIVE_READ_PROTECT_SCORE;
    }

    if active.read_start < active.read_end {
        let near_start = active.read_start.saturating_sub(HINTED_ACTIVE_NEAR_TOKENS);
        let near_end = active.read_end.saturating_add(HINTED_ACTIVE_NEAR_TOKENS);
        if ranges_overlap(block_start, block_end, near_start, near_end) {
            return -HINTED_ACTIVE_NEAR_PROTECT_SCORE;
        }
    }

    0
}

fn hinted_active_window_overlaps(
    inner: &PolarisInner,
    block: &PolarisBlock,
    active: Option<ActiveKvWindow>,
) -> bool {
    let Some(active) = active else {
        return false;
    };
    if !block_owned_by_session(inner, block, active.session_id) {
        return false;
    }

    let block_start = block.token_start as u64;
    let block_end = block_start.saturating_add(block.token_count as u64);

    (active.write_start < active.write_end
        && ranges_overlap(block_start, block_end, active.write_start, active.write_end))
        || (active.read_start < active.read_end
            && ranges_overlap(block_start, block_end, active.read_start, active.read_end))
}

fn hinted_active_write_overlaps(block: &PolarisBlock, active: ActiveKvWindow) -> bool {
    if active.write_start >= active.write_end {
        return false;
    }

    let block_start = block.token_start as u64;
    let block_end = block_start.saturating_add(block.token_count as u64);
    ranges_overlap(block_start, block_end, active.write_start, active.write_end)
}

fn hinted_active_read_overlaps(block: &PolarisBlock, active: ActiveKvWindow) -> bool {
    if active.read_start >= active.read_end {
        return false;
    }

    let block_start = block.token_start as u64;
    let block_end = block_start.saturating_add(block.token_count as u64);
    ranges_overlap(block_start, block_end, active.read_start, active.read_end)
}

fn phase_aware_order_time(block: &PolarisBlock) -> u64 {
    if block.phase == PolarisPhase::Decode && block.last_touch_ns != 0 {
        block.last_touch_ns
    } else {
        block.map_time_ns
    }
}

/// Score the block. Higher = more evictable.
fn phase_aware_score(
    inner: &PolarisInner,
    block: &PolarisBlock,
    now: u64,
    gpu_pressure: u64,
    session_priority: u32,
    active: Option<ActiveChunkWindow>,
    hinted_active: Option<ActiveKvWindow>,
) -> i64 {
    let order_time = phase_aware_order_time(block);
    let age_ns = now.saturating_sub(order_time);
    let age_ms = (age_ns / 1_000_000u64).min(MAX_AGE_SCORE_MS) as i64;

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

    AGE_WEIGHT * age_ms
        + PREFILL_WEIGHT * is_prefill
        + PRESSURE_WEIGHT * pressure_norm
        + PRIORITY_WEIGHT * inv_priority
        + active_window_score(inner, block, active)
        + hinted_active_window_score(inner, block, hinted_active)
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
    // llama.cpp's current shim exposes one Polaris "token" per KV chunk and
    // marks selected allocations as prefill. In that common single-session,
    // private-prefill shape, FIFO map order is the best available signal and
    // avoids treating chunk IDs as semantic LLM token positions.
    let hinted_active = active_kv_window(inner, requesting_session_id);

    if hinted_active.is_none() && phase_aware_should_use_fifo_order(inner, target_gpu) {
        return find_victim_fifo(
            inner,
            requesting_session_id,
            target_gpu,
            protected_phys_handle,
            protected_block_id,
        );
    }

    let now = unsafe { bindings::ktime_get_mono_fast_ns() };
    let active = active_chunk_window(inner, requesting_session_id, protected_block_id);

    // Look up GPU pressure once.
    let gpu_pressure = inner.gpus.iter()
        .find(|g| g.gpu_id == target_gpu)
        .map(|g| g.pressure_score)
        .unwrap_or(0);

    // Pass 1: other sessions only.
    let mut best: Option<(usize, u64, u64, u32, i64)> = None;
    let mut best_score: i64 = i64::MIN;
    let mut best_order_time: u64 = u64::MAX;

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
        if hinted_active_window_overlaps(inner, block, hinted_active) {
            continue;
        }
        let session_priority = inner.sessions.iter()
            .find(|s| s.session_id == block.session_id)
            .map(|s| s.priority)
            .unwrap_or(5u32);
        let score = phase_aware_score(
            inner,
            block,
            now,
            gpu_pressure,
            session_priority,
            active,
            hinted_active,
        );
        let order_time = phase_aware_order_time(block);
        if score > best_score || (score == best_score && order_time < best_order_time) {
            best_score = score;
            best_order_time = order_time;
            best = Some((idx, block.block_id, block.size_bytes, block.home_gpu, score));
        }
    }

    if best.is_some() {
        return best.map(|(a, b, c, d, _)| (a, b, c, d));
    }

    // Pass 2: any session (including the requesting one).
    best_score = i64::MIN;
    best_order_time = u64::MAX;
    for (idx, block) in inner.blocks.iter().enumerate() {
        if is_protected_source(block, protected_phys_handle, protected_block_id) {
            continue;
        }
        if !is_eligible_phase_aware(block, inner, target_gpu) {
            continue;
        }
        if hinted_active_window_overlaps(inner, block, hinted_active) {
            continue;
        }
        let session_priority = inner.sessions.iter()
            .find(|s| s.session_id == block.session_id)
            .map(|s| s.priority)
            .unwrap_or(5u32);
        let score = phase_aware_score(
            inner,
            block,
            now,
            gpu_pressure,
            session_priority,
            active,
            hinted_active,
        );
        let order_time = phase_aware_order_time(block);
        if score > best_score || (score == best_score && order_time < best_order_time) {
            best_score = score;
            best_order_time = order_time;
            best = Some((idx, block.block_id, block.size_bytes, block.home_gpu, score));
        }
    }

    best.map(|(a, b, c, d, _)| (a, b, c, d))
}

// ─── Attention-Stream: llama.cpp KV reuse-distance policy ───────────────────
//
// This policy consumes the existing llama.cpp KV active-window hints as a
// sequential attention schedule. It hard-protects the current write/read span
// and the near read lookahead, then prefers victims that are outside the active
// attention window or already behind the current read cursor. With today's ABI,
// read_start is the best available cursor; if no hint is present, fall back to
// PhaseAware so policy 3 remains safe on non-hinting workloads.

fn attention_stream_session_has_active_hint(inner: &PolarisInner, session_id: u64) -> bool {
    active_kv_window(inner, session_id).is_some()
}

fn attention_stream_order_time(block: &PolarisBlock) -> u64 {
    if block.last_touch_ns != 0 {
        block.last_touch_ns
    } else {
        block.map_time_ns
    }
}

fn attention_stream_score(
    inner: &PolarisInner,
    block: &PolarisBlock,
    active: ActiveKvWindow,
    session_priority: u32,
) -> i64 {
    let block_start = block.token_start as u64;
    let block_end = block_start.saturating_add(block.token_count as u64);
    let cursor = active.read_start;
    let read_end = active.read_end;
    let lookahead_end = cursor.saturating_add(ATTENTION_STREAM_LOOKAHEAD_TOKENS);
    let suffix_start = if active.write_start < active.write_end {
        active.write_start.saturating_sub(ATTENTION_STREAM_SUFFIX_TOKENS)
    } else {
        read_end.saturating_sub(ATTENTION_STREAM_SUFFIX_TOKENS)
    };

    let mut score = 0i64;

    if block_owned_by_session(inner, block, active.session_id) {
        if active.read_start < active.read_end
            && !ranges_overlap(block_start, block_end, active.read_start, active.read_end)
        {
            score += ATTENTION_STREAM_OUTSIDE_WINDOW_SCORE;
        } else if block_end <= cursor {
            let behind = cursor.saturating_sub(block_end).min(i64::MAX as u64) as i64;
            score += ATTENTION_STREAM_PAST_SCORE.saturating_add(behind);
        } else if block_start >= lookahead_end {
            let ahead = block_start.saturating_sub(lookahead_end) / ATTENTION_STREAM_FUTURE_SCALE;
            score += ahead.min(i64::MAX as u64) as i64;
        }

        if active.read_start < active.read_end
            && ranges_overlap(block_start, block_end, cursor, lookahead_end.min(active.read_end))
        {
            score -= ATTENTION_STREAM_LOOKAHEAD_PENALTY;
        }

        if ranges_overlap(block_start, block_end, suffix_start, active.write_end.max(read_end)) {
            score -= ATTENTION_STREAM_DECODE_SUFFIX_PENALTY;
        }
    } else {
        score += ATTENTION_STREAM_OTHER_IDLE_SCORE;
        if !attention_stream_session_has_active_hint(inner, block.session_id) {
            score += ATTENTION_STREAM_OTHER_IDLE_SCORE;
        }
    }

    if block.refcount > 1 {
        let refcount = block.refcount.min(8) as i64;
        score -= ATTENTION_STREAM_SHARED_PENALTY.saturating_mul(refcount);
    }

    let prio = if session_priority > 0 && session_priority <= 10 {
        session_priority
    } else {
        5u32
    };
    score += (11u32.saturating_sub(prio) as i64) * PRIORITY_WEIGHT;

    score
}

fn find_victim_attention_stream(
    inner: &PolarisInner,
    requesting_session_id: u64,
    target_gpu: u32,
    protected_phys_handle: u64,
    protected_block_id: u64,
) -> Option<(usize, u64, u64, u32)> {
    let Some(active) = active_kv_window(inner, requesting_session_id) else {
        return find_victim_phase_aware(
            inner,
            requesting_session_id,
            target_gpu,
            protected_phys_handle,
            protected_block_id,
        );
    };

    for pass in 0..2 {
        let mut best: Option<(usize, u64, u64, u32, i64)> = None;
        let mut best_score: i64 = i64::MIN;
        let mut best_order_time: u64 = u64::MAX;

        for (idx, block) in inner.blocks.iter().enumerate() {
            if pass == 0 && block_owned_by_session(inner, block, requesting_session_id) {
                continue;
            }
            if is_protected_source(block, protected_phys_handle, protected_block_id) {
                continue;
            }
            if !is_eligible_phase_aware(block, inner, target_gpu) {
                continue;
            }
            if block_owned_by_session(inner, block, active.session_id)
                && (hinted_active_write_overlaps(block, active)
                    || hinted_active_read_overlaps(block, active))
            {
                continue;
            }

            let session_priority = inner.sessions.iter()
                .find(|s| s.session_id == block.session_id)
                .map(|s| s.priority)
                .unwrap_or(5u32);
            let score = attention_stream_score(
                inner,
                block,
                active,
                session_priority,
            );
            let order_time = attention_stream_order_time(block);
            if score > best_score || (score == best_score && order_time < best_order_time) {
                best_score = score;
                best_order_time = order_time;
                best = Some((idx, block.block_id, block.size_bytes, block.home_gpu, score));
            }
        }

        if best.is_some() {
            return best.map(|(a, b, c, d, _)| (a, b, c, d));
        }
    }

    find_victim_phase_aware(
        inner,
        requesting_session_id,
        target_gpu,
        protected_phys_handle,
        protected_block_id,
    )
}
