// POLARIS Workload Generator — Phase 4a
//
// Synthetic inference workload runtime that exercises the POLARIS kernel
// module's KV cache management path.  Not a real model — only exercises the
// KV cache management path with real CUDA memory.
//
// Subcommands:
//   synthetic-kv     — single-session KV reserve/touch/release simulation
//   beam-search      — COW beam search workload with decode simulation
//   concurrent       — multi-session concurrent workload with decode simulation
//   trace-replay     — vLLM/SGLang trace file replay (Phase 4b)
//   cow-break        — comprehensive COW break test (Phase 3 validation)
//
// All workloads emit CSV traces when --csv <path> is given:
//   timestamp_ns,operation,session_id,token_start,token_count,block_id,gpu_id,latency_us

use clap::{Parser, Subcommand};
use libc::c_int;
use libpolaris::ioctl;
use libpolaris::types::*;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::os::fd::AsRawFd;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "polaris-workload", about = "POLARIS workload generator")]
struct Cli {
    #[arg(long, default_value = "/dev/polaris")]
    device: String,

    /// CSV trace output file path.  If set, all operations are logged.
    #[arg(long)]
    csv: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Single-session KV reserve/touch stress test with decode loop simulation
    ///
    /// Flat mode: --num-blocks N
    /// Decode loop mode: --prompt-tokens P --output-tokens O [--csv path]
    SyntheticKv {
        /// Number of blocks to allocate (flat mode, no decode sim)
        #[arg(long, default_value = "32")]
        num_blocks: u32,
        /// Tokens per block
        #[arg(long, default_value = "16")]
        tokens_per_block: u32,
        /// Prompt tokens: enables prefill + decode loop simulation
        #[arg(long)]
        prompt_tokens: Option<u32>,
        /// Output tokens: enables decode loop simulation
        #[arg(long)]
        output_tokens: Option<u32>,
    },
    /// COW beam search workload with decode loop simulation
    BeamSearch {
        #[arg(long, default_value = "8")]
        beam_width: u32,
        /// Prompt tokens for the parent session
        #[arg(long, default_value = "512")]
        prompt_tokens: u32,
        /// Decode steps (output tokens) per beam
        #[arg(long, default_value = "64")]
        decode_steps: u32,
        /// Tokens per block
        #[arg(long, default_value = "16")]
        tokens_per_block: u32,
        /// Test COW break: first child overwrites shared block 0
        #[arg(long)]
        cow_break: bool,
    },
    /// Multi-session concurrent workload with decode loop simulation
    Concurrent {
        #[arg(long, default_value = "10")]
        num_sessions: u32,
        /// Prompt tokens per session
        #[arg(long, default_value = "256")]
        prompt_tokens: u32,
        /// Output tokens per session
        #[arg(long, default_value = "64")]
        output_tokens: u32,
        /// Tokens per block
        #[arg(long, default_value = "16")]
        tokens_per_block: u32,
    },
    /// Comprehensive COW break test (Phase 3 validation)
    CowBreak,
}

// ─── CSV Trace Writer ────────────────────────────────────────────────────────

struct CsvWriter {
    wtr: BufWriter<std::fs::File>,
}

impl CsvWriter {
    fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)
            .map_err(|e| format!("failed to create trace file {path}: {e}"))?;
        let mut wtr = BufWriter::new(file);
        writeln!(
            wtr,
            "timestamp_ns,operation,session_id,token_start,token_count,block_id,gpu_id,latency_us"
        )?;
        Ok(Self { wtr })
    }

    fn record(
        &mut self,
        op: &str,
        session_id: u64,
        token_start: u32,
        token_count: u32,
        block_id: u64,
        gpu_id: u32,
        latency_us: u64,
    ) {
        let ts = Self::epoch_ns();
        let _ = writeln!(
            self.wtr,
            "{ts},{op},{session_id},{token_start},{token_count},{block_id},{gpu_id},{latency_us}"
        );
    }

    fn record_simple(&mut self, op: &str, session_id: u64) {
        let ts = Self::epoch_ns();
        let _ = writeln!(self.wtr, "{ts},{op},{session_id},0,0,0,0,0");
    }

    #[cfg(target_os = "linux")]
    fn epoch_ns() -> u64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
        (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
    }

    #[cfg(not(target_os = "linux"))]
    fn epoch_ns() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

// ─── IOCTL helpers ───────────────────────────────────────────────────────────

fn create_session(
    fd: c_int,
    csv: &mut Option<CsvWriter>,
    gpu: u32,
    beam_width: u32,
    vas_bytes: u64,
    priority: u32,
) -> Result<u64, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let mut arg = PolarisSessionCreateArg {
        home_gpu: gpu,
        beam_width,
        gpu_vas_bytes: vas_bytes,
        priority,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut arg)
        .map_err(|e| format!("SESSION_CREATE: errno {e}"))?;
    let lat_us = start.elapsed().as_micros() as u64;
    if let Some(ref mut c) = csv {
        c.record_simple("SESSION_CREATE", arg.session_id);
    }
    eprintln!(
        "  SESSION_CREATE id={} gpu={} ({} µs)",
        arg.session_id, gpu, lat_us
    );
    Ok(arg.session_id)
}

fn destroy_session(
    fd: c_int,
    csv: &mut Option<CsvWriter>,
    sid: u64,
) {
    let start = Instant::now();
    let arg = PolarisSessionDestroyArg {
        session_id: sid,
        ..Default::default()
    };
    let _ = ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &arg);
    let lat_us = start.elapsed().as_micros() as u64;
    if let Some(ref mut c) = csv {
        c.record_simple("SESSION_DESTROY", sid);
    }
    eprintln!("  SESSION_DESTROY id={sid} ({lat_us} µs)");
}

fn block_reserve(
    fd: c_int,
    csv: &mut Option<CsvWriter>,
    sid: u64,
    token_start: u32,
    token_count: u32,
    flags: u32,
    phase: PolarisPhase,
) -> Result<(i32, u64, u64), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let mut arg = PolarisBlockReserveArg {
        session_id: sid,
        token_start,
        token_count,
        flags,
        phase: phase as u32,
        ..Default::default()
    };
    let rc = match ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut arg) {
        Ok(()) => 0,
        Err(e) => -(e as i32),
    };
    let lat_us = start.elapsed().as_micros() as u64;
    let op = if flags & POLARIS_RESERVE_FLAG_OVERWRITE != 0 {
        "COW_BREAK"
    } else {
        "BLOCK_RESERVE"
    };
    if let Some(ref mut c) = csv {
        c.record(op, sid, token_start, token_count, arg.block_id, 0, lat_us);
    }
    if rc != 0 {
        eprintln!(
            "  {op} sid={sid} tokens={token_start}..{} rc={rc} ({lat_us} µs)",
            token_start + token_count,
        );
    }
    Ok((rc, arg.block_id, lat_us))
}

fn block_touch(
    fd: c_int,
    csv: &mut Option<CsvWriter>,
    sid: u64,
    token_start: u64,
    token_count: u64,
) -> Result<u64, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let arg = PolarisBlockTouchArg {
        session_id: sid,
        token_start,
        token_count,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_BLOCK_TOUCH, &arg)
        .map_err(|e| format!("BLOCK_TOUCH: errno {e}"))?;
    let lat_us = start.elapsed().as_micros() as u64;
    if let Some(ref mut c) = csv {
        c.record("BLOCK_TOUCH", sid, token_start as u32, token_count as u32, 0, 0, lat_us);
    }
    Ok(lat_us)
}

fn branch_session(
    fd: c_int,
    csv: &mut Option<CsvWriter>,
    parent_id: u64,
) -> Result<u64, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let mut arg = PolarisSessionBranchArg {
        parent_session_id: parent_id,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_BRANCH, &mut arg)
        .map_err(|e| format!("SESSION_BRANCH: errno {e}"))?;
    let lat_us = start.elapsed().as_micros() as u64;
    if let Some(ref mut c) = csv {
        c.record_simple("SESSION_BRANCH", arg.child_session_id);
    }
    eprintln!(
        "  SESSION_BRANCH parent={parent_id} → child={} ({lat_us} µs)",
        arg.child_session_id
    );
    Ok(arg.child_session_id)
}

fn get_global_stats(fd: c_int) -> PolarisGetGlobalStatsArg {
    let mut s = PolarisGetGlobalStatsArg::default();
    let _ = ioctl::ioctl_read(fd, ioctl::POLARIS_GET_GLOBAL_STATS, &mut s);
    s
}

// ═══════════════════════════════════════════════════════════════════════════════
// ─── Decode Loop Simulation ───────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

/// Simulate a full prefill + decode cycle for one session.
///
/// Returns (total_prefill_blocks, total_decode_blocks, total_latency_us).
///
/// Behavior:
///   - Prefill: reserve prompt_blocks in batch with phase=Prefill
///   - Decode: reserve the next output block, then touch the existing range
fn run_decode_loop(
    fd: c_int,
    csv: &mut Option<CsvWriter>,
    sid: u64,
    prompt_tokens: u32,
    output_tokens: u32,
    tokens_per_block: u32,
    overwrite_block: Option<(u32, u32)>, // (token_start, token_count) for COW break
) -> Result<(u32, u32, u64), Box<dyn std::error::Error>> {
    let mut total_lat_us: u64 = 0;
    let prompt_blocks = prompt_tokens.div_ceil(tokens_per_block);
    let output_blocks = output_tokens.div_ceil(tokens_per_block);
    let mut total_tokens: u64 = 0;

    // ── Prefill phase ────────────────────────────────────────────────────
    eprintln!("  Prefill: reserving {prompt_blocks} blocks for {prompt_tokens} tokens...");
    for i in 0..prompt_blocks {
        let start = i * tokens_per_block;
        let count = tokens_per_block.min(prompt_tokens - start);
        total_tokens += count as u64;
        let (rc, bid, lat_us) = block_reserve(
            fd, csv, sid, start, count, 0, PolarisPhase::Prefill,
        )?;
        total_lat_us += lat_us;
        if rc != 0 {
            return Err(format!(
                "prefill block {i} (tokens {start}..{}) failed with rc={rc}",
                start + count
            )
            .into());
        }
        if i % 16 == 15 || i == prompt_blocks - 1 {
            eprintln!(
                "    prefill block {}/{} done (id={bid}, {} µs)",
                i + 1,
                prompt_blocks,
                lat_us,
            );
        }
    }

    // ── Decode phase ─────────────────────────────────────────────────────
    eprintln!("  Decode: {output_blocks} blocks for {output_tokens} tokens...");
    for i in 0..output_blocks {
        let token_offset = prompt_tokens + i * tokens_per_block;
        let count = tokens_per_block.min(prompt_tokens + output_tokens - token_offset);
        total_tokens += count as u64;

        // Reserve the new decode block.
        let (rc, _bid, lat_us) = block_reserve(
            fd, csv, sid, token_offset, count, 0, PolarisPhase::Decode,
        )?;
        total_lat_us += lat_us;
        if rc != 0 {
            return Err(format!(
                "decode block {i} (tokens {token_offset}..{}) failed with rc={rc}",
                token_offset + count
            )
            .into());
        }

        // Touch the active range to model the access pattern seen by the
        // interrupt-driven fault path.
        let touch_lat = block_touch(fd, csv, sid, 0, total_tokens)?;
        total_lat_us += touch_lat;

        if i % 16 == 15 || i == output_blocks - 1 {
            eprintln!(
                "    decode step {}/{} (total tokens={total_tokens}, reserve={lat_us} µs, touch={touch_lat} µs)",
                i + 1,
                output_blocks,
            );
        }
    }

    // ── Optional COW-break overlay ───────────────────────────────────────
    if let Some((over_start, over_count)) = overwrite_block {
        eprintln!("  COW_BREAK overwrite: tokens {over_start}..{}", over_start + over_count);
        let (rc, _bid, lat_us) = block_reserve(
            fd, csv, sid, over_start, over_count,
            POLARIS_RESERVE_FLAG_OVERWRITE, PolarisPhase::Prefill,
        )?;
        total_lat_us += lat_us;
        if rc != 0 {
            eprintln!("  COW_BREAK failed with rc={rc} (expected if refcount==1)");
        }
    }

    Ok((prompt_blocks, output_blocks, total_lat_us))
}

// ═══════════════════════════════════════════════════════════════════════════════
// ─── Synthetic KV Workload ────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

fn run_synthetic_kv(
    fd: c_int,
    csv: &mut Option<CsvWriter>,
    num_blocks: u32,
    tokens_per_block: u32,
    prompt_tokens: Option<u32>,
    output_tokens: Option<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Synthetic KV Workload ===");

    let total_start = Instant::now();

    // Create session
    let vas_bytes: u64 = if let (Some(p), Some(o)) = (prompt_tokens, output_tokens) {
        // Decode loop mode: reserve enough VA for all tokens
        ((p + o) as u64) * POLARIS_DEFAULT_BYTES_PER_TOKEN * 2
    } else {
        (num_blocks as u64) * (tokens_per_block as u64) * POLARIS_DEFAULT_BYTES_PER_TOKEN
    };
    let sid = create_session(fd, csv, 0, 1, vas_bytes, 5)?;

    if let (Some(prompt), Some(output)) = (prompt_tokens, output_tokens) {
        // ── Decode loop mode ──────────────────────────────────────────────
        eprintln!(
            "Mode: decode loop  |  prompt={prompt} tokens  |  output={output} tokens  |  {tokens_per_block} tok/block"
        );
        let (pf, dec, total_lat) = run_decode_loop(
            fd, csv, sid, prompt, output, tokens_per_block, None,
        )?;
        let total_elapsed = total_start.elapsed();
        eprintln!(
            "Decode loop complete: {pf} prefill + {dec} decode blocks, {total_lat} µs ioctl total, {:.2?} wall clock",
            total_elapsed
        );
    } else {
        // ── Flat reserve mode ────────────────────────────────────────────
        eprintln!("Mode: flat reserve  |  {num_blocks} blocks  |  {tokens_per_block} tok/block");
        let start = Instant::now();
        let mut total_lat: u64 = 0;
        for i in 0..num_blocks {
            let (rc, _bid, lat_us) = block_reserve(
                fd, csv, sid,
                i * tokens_per_block,
                tokens_per_block,
                0,
                PolarisPhase::Prefill,
            )?;
            total_lat += lat_us;
            if rc != 0 {
                eprintln!("  Block {i} reserve failed (ret_code={rc})");
                destroy_session(fd, csv, sid);
                return Err(format!("BLOCK_RESERVE {i} failed with {rc}").into());
            }
        }
        let elapsed = start.elapsed();
        eprintln!("Reserved {num_blocks} blocks in {elapsed:?} ({total_lat} µs ioctl)");

        // Touch the full reserved range once.
        let _ = block_touch(
            fd, csv, sid, 0,
            (num_blocks * tokens_per_block) as u64,
        )?;
    }

    destroy_session(fd, csv, sid);

    let stats = get_global_stats(fd);
    eprintln!(
        "Global: gpu_used={} MiB  cpu_used={} MiB  sessions={}  blocks={}",
        stats.used_gpu_bytes / (1024 * 1024),
        stats.cpu_pool_used / (1024 * 1024),
        stats.total_sessions,
        stats.total_blocks,
    );

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// ─── Beam Search Workload ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

fn run_beam_search(
    fd: c_int,
    csv: &mut Option<CsvWriter>,
    beam_width: u32,
    prompt_tokens: u32,
    decode_steps: u32,
    tokens_per_block: u32,
    cow_break: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Beam Search Workload ===");
    eprintln!(
        "beam={beam_width}  prompt={prompt_tokens}tok  decode={decode_steps}tok  blk={tokens_per_block}tok  cow_break={cow_break}"
    );

    let total_start = Instant::now();

    // ── Parent session + prompt ──────────────────────────────────────────
    let parent_id = create_session(fd, csv, 0, 1, 1024 * 1024 * 1024, 5)?;
    eprintln!("Parent session: {parent_id}");

    eprintln!("Parent prompt: {prompt_tokens} tokens...");
    let (pf_blocks, _, pf_lat) = run_decode_loop(
        fd, csv, parent_id,
        prompt_tokens, 0, // prefill only, no decode for parent
        tokens_per_block, None,
    )?;
    eprintln!("Parent prompt done: {pf_blocks} blocks, {pf_lat} µs");

    // ── Branch into children ─────────────────────────────────────────────
    let mut children: Vec<u64> = Vec::new();
    for b in 1..beam_width {
        let child_id = branch_session(fd, csv, parent_id)?;
        children.push(child_id);
        if b % 4 == 0 || b + 1 == beam_width {
            eprintln!("  Branched {b}/{beam_width} children");
        }
    }
    eprintln!("COW branching complete: {} children", children.len());

    // Print sharing stats
    let stats = get_global_stats(fd);
    eprintln!(
        "After branch: shared={} MiB  private={} MiB  cow_breaks={}",
        stats.shared_gpu_bytes / (1024 * 1024),
        stats.private_gpu_bytes / (1024 * 1024),
        stats.cow_break_count,
    );

    // ── Decode for each child (round-robin step) ────────────────────────
    if decode_steps > 0 {
        let decode_blocks = decode_steps.div_ceil(tokens_per_block);
        eprintln!("Decode: {decode_steps} tokens ({decode_blocks} blocks) per child...");

        for step in 0..decode_blocks {
            for (ci, &child_id) in children.iter().enumerate() {
                let token_offset = prompt_tokens + step * tokens_per_block;
                let count =
                    tokens_per_block.min(prompt_tokens + decode_steps - token_offset);

                let (rc, _, _) = block_reserve(
                    fd, csv, child_id, token_offset, count, 0, PolarisPhase::Decode,
                )?;
                if rc != 0 {
                    eprintln!("  Child {ci} decode step {step} failed (rc={rc}) — stopping");
                }

                // Touch all existing blocks in this child
                let total_tokens = (prompt_tokens + step * tokens_per_block + count) as u64;
                let _ = block_touch(fd, csv, child_id, 0, total_tokens)?;
            }
            if step % 8 == 0 || step == decode_blocks - 1 {
                eprintln!(
                    "  Decode step {}/{} done ({} children)",
                    step + 1,
                    decode_blocks,
                    children.len(),
                );
            }
        }
    }

    // ── COW break test ──────────────────────────────────────────────────
    if cow_break && !children.is_empty() {
        let first_child = children[0];
        eprintln!(
            "COW_BREAK test: child {first_child} overwrites token 0..{tokens_per_block}"
        );
        let (rc, bid, lat_us) = block_reserve(
            fd, csv, first_child, 0, tokens_per_block,
            POLARIS_RESERVE_FLAG_OVERWRITE, PolarisPhase::Prefill,
        )?;
        if rc != 0 {
            eprintln!("  COW_BREAK failed (rc={rc})");
        } else {
            eprintln!("  COW_BREAK success: new block_id={bid} ({lat_us} µs)");
        }

        let stats2 = get_global_stats(fd);
        eprintln!(
            "After COW break: shared={} MiB  private={} MiB  cow_breaks={}  cow_copy={} KiB",
            stats2.shared_gpu_bytes / (1024 * 1024),
            stats2.private_gpu_bytes / (1024 * 1024),
            stats2.cow_break_count,
            stats2.cow_copy_bytes / 1024,
        );
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    for child_id in &children {
        destroy_session(fd, csv, *child_id);
    }
    destroy_session(fd, csv, parent_id);

    let elapsed = total_start.elapsed();
    eprintln!("Beam search complete: {:.2?}", elapsed);

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// ─── Concurrent Workload ──────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

fn run_concurrent(
    fd: c_int,
    csv: &mut Option<CsvWriter>,
    num_sessions: u32,
    prompt_tokens: u32,
    output_tokens: u32,
    tokens_per_block: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Concurrent Workload ===");
    eprintln!(
        "sessions={num_sessions}  prompt={prompt_tokens}tok  output={output_tokens}tok  blk={tokens_per_block}tok"
    );

    let total_start = Instant::now();
    let mut session_ids: Vec<u64> = Vec::new();

    // ── Create all sessions + prefill ────────────────────────────────────
    for s in 0..num_sessions {
        let sid = create_session(fd, csv, 0, 1, 512 * 1024 * 1024, 5)?;
        session_ids.push(sid);

        // Prefill for this session
        let prompt_blocks = prompt_tokens.div_ceil(tokens_per_block);
        for i in 0..prompt_blocks {
            let start = i * tokens_per_block;
            let count = tokens_per_block.min(prompt_tokens - start);
            let (rc, _, _) = block_reserve(
                fd, csv, sid, start, count, 0, PolarisPhase::Prefill,
            )?;
            if rc != 0 {
                eprintln!("  Session {s} block {i} failed (rc={rc})");
            }
        }

        if (s + 1) % 10 == 0 || s + 1 == num_sessions {
            eprintln!("  {}/{} sessions prefill done", s + 1, num_sessions);
        }
    }

    // ── Decode for all sessions (round-robin) ────────────────────────────
    let decode_blocks = output_tokens.div_ceil(tokens_per_block);
    if decode_blocks > 0 {
        eprintln!("Decode: {output_tokens} tokens ({decode_blocks} steps) per session...");
        for step in 0..decode_blocks {
            for &sid in &session_ids {
                let token_offset = prompt_tokens + step * tokens_per_block;
                let count =
                    tokens_per_block.min(prompt_tokens + output_tokens - token_offset);

                let (rc, _, _) = block_reserve(
                    fd, csv, sid, token_offset, count, 0, PolarisPhase::Decode,
                )?;
                if rc != 0 {
                    // Skip this session if reserve fails.
                    continue;
                }

                let total_tokens = (prompt_tokens + step * tokens_per_block + count) as u64;
                let _ = block_touch(fd, csv, sid, 0, total_tokens)?;
            }
            if step % 8 == 0 || step == decode_blocks - 1 {
                eprintln!(
                    "  Decode step {}/{} done",
                    step + 1,
                    decode_blocks,
                );
            }
        }
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    for sid in &session_ids {
        destroy_session(fd, csv, *sid);
    }

    let elapsed = total_start.elapsed();
    eprintln!("Concurrent complete: {:.2?} — {} sessions", elapsed, num_sessions);

    let stats = get_global_stats(fd);
    eprintln!(
        "Global: gpu_used={} MiB  sessions={}  blocks={}  offloads={}  reloads={}",
        stats.used_gpu_bytes / (1024 * 1024),
        stats.total_sessions,
        stats.total_blocks,
        stats.offload_count,
        stats.reload_count,
    );

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// ─── Comprehensive COW Break Test (Phase 3 validation) ────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

fn run_cow_break_test(fd: c_int) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Comprehensive COW Break Test ===");
    let mut passed = 0u32;
    let mut failed = 0u32;

    fn get_stats(fd: c_int) -> (u64, u64, u64, u64) {
        let mut s = PolarisGetGlobalStatsArg::default();
        let _ = ioctl::ioctl_read(fd, ioctl::POLARIS_GET_GLOBAL_STATS, &mut s);
        (
            s.shared_gpu_bytes / (1024 * 1024),
            s.private_gpu_bytes / (1024 * 1024),
            s.cow_break_count,
            s.cow_copy_bytes / (1024 * 1024),
        )
    }

    fn reserve(
        fd: c_int, sid: u64, start: u32, count: u32, flags: u32, phase: u32,
    ) -> (i32, u64) {
        let mut arg = PolarisBlockReserveArg {
            session_id: sid,
            token_start: start,
            token_count: count,
            flags,
            phase,
            ..Default::default()
        };
        match ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut arg) {
            Ok(()) => (0, arg.block_id),
            Err(eno) => (-(eno as i32), arg.block_id),
        }
    }

    fn branch(fd: c_int, parent: u64) -> u64 {
        let mut arg = PolarisSessionBranchArg {
            parent_session_id: parent,
            ..Default::default()
        };
        let _ = ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_BRANCH, &mut arg);
        arg.child_session_id
    }

    fn create_session(fd: c_int, priority: u32) -> u64 {
        let mut arg = PolarisSessionCreateArg {
            home_gpu: 0,
            beam_width: 1,
            gpu_vas_bytes: 1024 * 1024 * 1024,
            priority,
            ..Default::default()
        };
        let _ = ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut arg);
        arg.session_id
    }

    fn destroy_session(fd: c_int, sid: u64) {
        let arg = PolarisSessionDestroyArg {
            session_id: sid,
            ..Default::default()
        };
        let _ = ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &arg);
    }

    // ── Setup: parent + 8 prompt blocks ──────────────────────────────────
    let parent = create_session(fd, 5);
    eprintln!("Parent session: {parent}");

    let mut block_ids: Vec<u64> = Vec::new();
    for i in 0..8u32 {
        let (rc, bid) = reserve(fd, parent, i * 16, 16, 0, PolarisPhase::Prefill as u32);
        if rc != 0 {
            eprintln!("  FAIL: parent block {i} reserve failed (rc={rc})");
            destroy_session(fd, parent);
            return Err(format!("setup failed at block {i}").into());
        }
        block_ids.push(bid);
    }
    eprintln!("Parent prompt: 8 blocks reserved");

    // ── Branch: 3 children ───────────────────────────────────────────────
    let mut children: Vec<u64> = Vec::new();
    for _ in 0..3 {
        let cid = branch(fd, parent);
        children.push(cid);
    }
    let (sh, pr, _, _) = get_stats(fd);
    eprintln!("After branch: shared={sh} MiB  private={pr} MiB");

    // Test A: first child COW-breaks block 0
    eprintln!("\n--- Test A: first child COW-break block 0 ---");
    let c1 = children[0];
    let (rc, new_bid_a) = reserve(
        fd, c1, 0, 16, POLARIS_RESERVE_FLAG_OVERWRITE, PolarisPhase::Prefill as u32,
    );
    let (_sh_a, _pr_a, cb_a, cc_a) = get_stats(fd);
    if rc == 0 && cb_a == 1 && new_bid_a != block_ids[0] {
        eprintln!("  PASS: rc=0, new_block={new_bid_a}, cow_breaks={cb_a}, cow_copy={cc_a} MiB");
        passed += 1;
    } else {
        eprintln!(
            "  FAIL: rc={rc}, new_block={new_bid_a}, cow_breaks={cb_a}, cow_copy={cc_a} MiB"
        );
        failed += 1;
    }

    // Test B: second child COW-breaks same block 0
    eprintln!("\n--- Test B: second child COW-break same block 0 ---");
    let c2 = children[1];
    let (rc, new_bid_b) = reserve(
        fd, c2, 0, 16, POLARIS_RESERVE_FLAG_OVERWRITE, PolarisPhase::Prefill as u32,
    );
    let (_sh_b, _pr_b, cb_b, _cc_b) = get_stats(fd);
    if rc == 0 && cb_b == 2 && new_bid_b != block_ids[0] && new_bid_b != new_bid_a {
        eprintln!("  PASS: rc=0, new_block={new_bid_b}, cow_breaks={cb_b}");
        passed += 1;
    } else {
        eprintln!("  FAIL: rc={rc}, new_block={new_bid_b}, cow_breaks={cb_b}");
        failed += 1;
    }

    // Test C: first child in-place overwrite (block now private)
    eprintln!("\n--- Test C: first child in-place overwrite (refcount=1) ---");
    let (rc_c, bid_c) = reserve(
        fd, c1, 0, 16, POLARIS_RESERVE_FLAG_OVERWRITE, PolarisPhase::Prefill as u32,
    );
    let (_sh_c, _pr_c, cb_c, _) = get_stats(fd);
    if rc_c == 0 && bid_c == new_bid_a && cb_c == 2 {
        eprintln!("  PASS: in-place returned same block_id={bid_c}, cow_breaks still {cb_c}");
        passed += 1;
    } else {
        eprintln!("  FAIL: rc={rc_c}, block_id={bid_c} (expected {new_bid_a}), cow_breaks={cb_c}");
        failed += 1;
    }

    // Test D: overlap without OVERWRITE → EINVAL
    eprintln!("\n--- Test D: overlap without OVERWRITE flag ---");
    let (rc_d, bid_d) = reserve(fd, children[2], 0, 16, 0, PolarisPhase::Prefill as u32);
    if rc_d == -(libc::EINVAL as i32) {
        eprintln!("  PASS: rejected with EINVAL ({rc_d})");
        passed += 1;
    } else {
        eprintln!(
            "  FAIL: expected EINVAL ({}), got rc={rc_d} block_id={bid_d}",
            -(libc::EINVAL as i32)
        );
        failed += 1;
    }

    // Test E: third child COW-breaks block 1
    eprintln!("\n--- Test E: third child COW-break block 1 (still shared) ---");
    let c3 = children[2];
    let (rc_e, new_bid_e) = reserve(
        fd, c3, 16, 16, POLARIS_RESERVE_FLAG_OVERWRITE, PolarisPhase::Prefill as u32,
    );
    let (_sh_e, _pr_e, cb_e, _) = get_stats(fd);
    if rc_e == 0 && cb_e == 3 && new_bid_e != block_ids[1] {
        eprintln!("  PASS: rc=0, new_block={new_bid_e}, cow_breaks={cb_e}");
        passed += 1;
    } else {
        eprintln!("  FAIL: rc={rc_e}, new_block={new_bid_e}, cow_breaks={cb_e}");
        failed += 1;
    }

    // Test F: nested branch + COW break
    eprintln!("\n--- Test F: nested branch + COW break in grandchild ---");
    let grandchild = branch(fd, c1);
    eprintln!("  Grandchild session: {grandchild}");
    let (rc_f, new_bid_f) = reserve(
        fd, grandchild, 32, 16, POLARIS_RESERVE_FLAG_OVERWRITE, PolarisPhase::Prefill as u32,
    );
    let (sh_f, pr_f, cb_f, _) = get_stats(fd);
    if rc_f == 0 && cb_f == 4 && new_bid_f != block_ids[2] {
        eprintln!(
            "  PASS: rc=0, new_block={new_bid_f}, cow_breaks={cb_f}, shared={sh_f} MiB, private={pr_f} MiB"
        );
        passed += 1;
    } else {
        eprintln!("  FAIL: rc={rc_f}, new_block={new_bid_f}, cow_breaks={cb_f}");
        failed += 1;
    }

    // Test G: grandchild COW-break block 0 (shared with c1)
    eprintln!("\n--- Test G: grandchild COW-break block 0 (shared with c1) ---");
    let (rc_g, new_bid_g) = reserve(
        fd, grandchild, 0, 16, POLARIS_RESERVE_FLAG_OVERWRITE, PolarisPhase::Prefill as u32,
    );
    let (_sh_g, _pr_g, cb_g, _) = get_stats(fd);
    if rc_g == 0 && cb_g == 5 && new_bid_g != new_bid_a {
        eprintln!(
            "  PASS: rc=0, new_block={new_bid_g} (different from c1's {new_bid_a}), cow_breaks={cb_g}"
        );
        passed += 1;
    } else {
        eprintln!("  FAIL: rc={rc_g}, new_block={new_bid_g}, cow_breaks={cb_g}");
        failed += 1;
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    destroy_session(fd, grandchild);
    for cid in &children {
        destroy_session(fd, *cid);
    }
    destroy_session(fd, parent);

    // ── Final summary ────────────────────────────────────────────────────
    eprintln!("\n═══════════════════════════════");
    eprintln!("COW Break Test Results: {passed} passed, {failed} failed");
    if failed > 0 {
        eprintln!("SOME TESTS FAILED!");
        Err(format!("{failed} test(s) failed").into())
    } else {
        eprintln!("All tests passed!");
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ─── Main ─────────────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&cli.device)
        .map_err(|e| format!("Failed to open {}: {e}", cli.device))?;

    let fd = file.as_raw_fd() as c_int;

    // Create the CSV writer if --csv is specified
    let mut csv: Option<CsvWriter> = cli.csv.as_ref().map(|path| {
        CsvWriter::new(path).unwrap_or_else(|e| {
            eprintln!("WARNING: failed to open CSV file: {e}");
            panic!("csv error: {e}")
        })
    });

    match cli.command {
        Commands::SyntheticKv {
            num_blocks,
            tokens_per_block,
            prompt_tokens,
            output_tokens,
        } => run_synthetic_kv(fd, &mut csv, num_blocks, tokens_per_block, prompt_tokens, output_tokens)?,
        Commands::BeamSearch {
            beam_width,
            prompt_tokens,
            decode_steps,
            tokens_per_block,
            cow_break,
        } => run_beam_search(fd, &mut csv, beam_width, prompt_tokens, decode_steps, tokens_per_block, cow_break)?,
        Commands::Concurrent {
            num_sessions,
            prompt_tokens,
            output_tokens,
            tokens_per_block,
        } => run_concurrent(fd, &mut csv, num_sessions, prompt_tokens, output_tokens, tokens_per_block)?,
        Commands::CowBreak => run_cow_break_test(fd)?,
    }

    Ok(())
}
