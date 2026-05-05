// POLARIS Workload Generator
//
// Generates synthetic KV Cache workloads to exercise the POLARIS kernel module.
// Not a real model — only exercises the KV cache management path with real CUDA memory.
//
// Subcommands:
//   synthetic-kv     — single-session KV allocation stress
//   beam-search      — COW beam search workload
//   concurrent       — multi-session concurrent workload
//   trace-replay     — vLLM/SGLang trace file replay (Phase 4b)

use clap::{Parser, Subcommand};
use libc::c_int;
use libpolaris::ioctl;
use libpolaris::types::*;
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "polaris-workload", about = "POLARIS workload generator")]
struct Cli {
    /// Path to /dev/polaris (default: /dev/polaris)
    #[arg(long, default_value = "/dev/polaris")]
    device: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Single-session KV allocation stress test
    SyntheticKv {
        /// Number of blocks to allocate
        #[arg(long, default_value = "32")]
        num_blocks: u32,
        /// Tokens per block
        #[arg(long, default_value = "16")]
        tokens_per_block: u32,
    },
    /// COW beam search workload
    BeamSearch {
        /// Beam width
        #[arg(long, default_value = "8")]
        beam_width: u32,
        /// Number of decode steps
        #[arg(long, default_value = "64")]
        decode_steps: u32,
        /// Test COW break: first child overwrites shared block 0
        #[arg(long)]
        cow_break: bool,
    },
    /// Multi-session concurrent workload
    Concurrent {
        /// Number of concurrent sessions
        #[arg(long, default_value = "10")]
        num_sessions: u32,
        /// Blocks per session
        #[arg(long, default_value = "16")]
        blocks_per_session: u32,
    },
    /// Comprehensive COW break test (Phase 3 validation)
    CowBreak,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&cli.device)
        .map_err(|e| format!("Failed to open {}: {e}", cli.device))?;

    let fd = file.as_raw_fd() as c_int;

    match cli.command {
        Commands::SyntheticKv {
            num_blocks,
            tokens_per_block,
        } => run_synthetic_kv(fd, num_blocks, tokens_per_block)?,
        Commands::BeamSearch {
            beam_width,
            decode_steps,
            cow_break,
        } => run_beam_search(fd, beam_width, decode_steps, cow_break)?,
        Commands::Concurrent {
            num_sessions,
            blocks_per_session,
        } => run_concurrent(fd, num_sessions, blocks_per_session)?,
        Commands::CowBreak => run_cow_break_test(fd)?,
    }

    Ok(())
}

// ─── Synthetic KV workload ──────────────────────────────────────────────────

fn run_synthetic_kv(fd: c_int, num_blocks: u32, tokens_per_block: u32) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Synthetic KV Workload ===");
    eprintln!("Blocks: {num_blocks}, Tokens/block: {tokens_per_block}");

    // Create a session.
    let mut session_arg = PolarisSessionCreateArg {
        home_gpu: 0,
        beam_width: 1,
        gpu_vas_bytes: (num_blocks as u64) * (tokens_per_block as u64) * 512 * 1024, // rough estimate
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut session_arg)
        .map_err(|e| format!("SESSION_CREATE: errno {e}"))?;
    let sid = session_arg.session_id;
    eprintln!("Session created: id={sid}");

    let start = Instant::now();

    // Allocate blocks.
    for i in 0..num_blocks {
        let mut grow_arg = PolarisBlockGrowArg {
            session_id: sid,
            token_start: i * tokens_per_block,
            token_count: tokens_per_block,
            phase: PolarisPhase::Prefill as u32,
            ..Default::default()
        };
        ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_GROW, &mut grow_arg)
            .map_err(|e| format!("BLOCK_GROW: errno {e}"))?;
        if grow_arg.ret_code != 0 {
            eprintln!("  Block {} allocation failed (ret_code={})", i, grow_arg.ret_code);
            return Err(format!("BLOCK_GROW {} failed with {}", i, grow_arg.ret_code).into());
        }
        eprintln!("  Block {} allocated (id={})", i, grow_arg.block_id);
    }

    let elapsed = start.elapsed();
    eprintln!("Allocated {num_blocks} blocks in {elapsed:?}");

    // Touch all blocks (simulate decode).
    let touch_arg = PolarisBlockTouchArg {
        session_id: sid,
        token_start: 0,
        token_count: (num_blocks * tokens_per_block) as u64,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_BLOCK_TOUCH, &touch_arg)
        .map_err(|e| format!("BLOCK_TOUCH: errno {e}"))?;

    // Destroy session.
    let destroy_arg = PolarisSessionDestroyArg {
        session_id: sid,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &destroy_arg)
        .map_err(|e| format!("SESSION_DESTROY: errno {e}"))?;
    eprintln!("Session {sid} destroyed");

    Ok(())
}

// ─── Beam Search workload ───────────────────────────────────────────────────

fn run_beam_search(fd: c_int, beam_width: u32, decode_steps: u32, cow_break: bool) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Beam Search Workload ===");
    eprintln!("Beam width: {beam_width}, Decode steps: {decode_steps}, COW break test: {cow_break}");

    // Create parent session.
    let mut parent_arg = PolarisSessionCreateArg {
        home_gpu: 0,
        beam_width: 1,
        gpu_vas_bytes: 1024 * 1024 * 1024,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut parent_arg)
        .map_err(|e| format!("SESSION_CREATE: errno {e}"))?;
    let parent_id = parent_arg.session_id;
    eprintln!("Parent session: {parent_id}");

    // Allocate prompt blocks (16 tokens each, ~32 blocks = 512 tokens).
    for i in 0..32u32 {
        let mut grow = PolarisBlockGrowArg {
            session_id: parent_id,
            token_start: i * 16,
            token_count: 16,
            phase: PolarisPhase::Prefill as u32,
            ..Default::default()
        };
        ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_GROW, &mut grow)
            .map_err(|e| format!("BLOCK_GROW: errno {e}"))?;
        if grow.ret_code != 0 {
            return Err(format!("BLOCK_GROW block {} for parent failed with {}", i, grow.ret_code).into());
        }
    }
    eprintln!("Prompt blocks allocated for parent");

    // Branch into beam_width children.
    let mut children: Vec<u64> = Vec::new();
    for b in 1..beam_width {
        let mut branch_arg = PolarisSessionBranchArg {
            parent_session_id: parent_id,
            ..Default::default()
        };
        ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_BRANCH, &mut branch_arg)
            .map_err(|e| format!("SESSION_BRANCH: errno {e}"))?;
        children.push(branch_arg.child_session_id);
        eprintln!("  Branch {}: child session {}", b, branch_arg.child_session_id);
    }

    eprintln!("COW branching complete: {} children", children.len());

    // Check stats.
    let mut stats = PolarisGetGlobalStatsArg::default();
    ioctl::ioctl_read(fd, ioctl::POLARIS_GET_GLOBAL_STATS, &mut stats)
        .map_err(|e| format!("GET_GLOBAL_STATS: errno {e}"))?;
    eprintln!("Shared GPU bytes: {} MiB", stats.shared_gpu_bytes / (1024 * 1024));
    eprintln!("Private GPU bytes: {} MiB", stats.private_gpu_bytes / (1024 * 1024));
    eprintln!("COW break count: {}", stats.cow_break_count);
    eprintln!("COW copy bytes: {} KiB", stats.cow_copy_bytes / 1024);

    // COW break test: first child overwrites a shared prompt block.
    if cow_break && !children.is_empty() {
        let first_child = children[0];
        eprintln!("Testing COW_BREAK: child {first_child} overwrites token range 0..16");

        let mut cow_grow = PolarisBlockGrowArg {
            session_id: first_child,
            token_start: 0, // overlaps the first shared block
            token_count: 16,
            flags: POLARIS_GROW_FLAG_OVERWRITE,
            phase: PolarisPhase::Prefill as u32,
            ..Default::default()
        };
        ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_GROW, &mut cow_grow)
            .map_err(|e| format!("COW_BREAK BLOCK_GROW: errno {e}"))?;
        if cow_grow.ret_code != 0 {
            eprintln!("  COW_BREAK failed (ret_code={})", cow_grow.ret_code);
        } else {
            eprintln!("  COW_BREAK succeeded, new private block_id={}", cow_grow.block_id);
        }

        // Re-check stats after COW break.
        let mut stats2 = PolarisGetGlobalStatsArg::default();
        ioctl::ioctl_read(fd, ioctl::POLARIS_GET_GLOBAL_STATS, &mut stats2)
            .map_err(|e| format!("GET_GLOBAL_STATS: errno {e}"))?;
        eprintln!(
            "After COW break: shared={} MiB, private={} MiB, cow_breaks={}, cow_copy={} KiB",
            stats2.shared_gpu_bytes / (1024 * 1024),
            stats2.private_gpu_bytes / (1024 * 1024),
            stats2.cow_break_count,
            stats2.cow_copy_bytes / 1024,
        );
    }

    // Cleanup.
    for child_id in children {
        let destroy = PolarisSessionDestroyArg {
            session_id: child_id,
            ..Default::default()
        };
        ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &destroy).ok();
    }
    let destroy = PolarisSessionDestroyArg {
        session_id: parent_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &destroy).ok();

    Ok(())
}

// ─── Concurrent workload ────────────────────────────────────────────────────

fn run_concurrent(fd: c_int, num_sessions: u32, blocks_per_session: u32) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Concurrent Workload ===");
    eprintln!("Sessions: {num_sessions}, Blocks/session: {blocks_per_session}");

    let mut session_ids: Vec<u64> = Vec::new();

    // Create all sessions and allocate blocks.
    for s in 0..num_sessions {
        let mut session_arg = PolarisSessionCreateArg {
            home_gpu: 0,
            beam_width: 1,
            gpu_vas_bytes: 512 * 1024 * 1024,
            priority: 5,
            ..Default::default()
        };
        ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut session_arg)
            .map_err(|e| format!("SESSION_CREATE: errno {e}"))?;
        session_ids.push(session_arg.session_id);

        for i in 0..blocks_per_session {
            let mut grow = PolarisBlockGrowArg {
                session_id: session_arg.session_id,
                token_start: i * 16,
                token_count: 16,
                phase: PolarisPhase::Prefill as u32,
                ..Default::default()
            };
            ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_GROW, &mut grow).ok();
            if grow.ret_code != 0 {
                eprintln!("  Session {} block {} allocation failed (ret_code={})", s, i, grow.ret_code);
            }
        }

        if (s + 1) % 10 == 0 {
            eprintln!("  {}/{} sessions created", s + 1, num_sessions);
        }
    }

    eprintln!("All {num_sessions} sessions created and blocks allocated");

    // Cleanup.
    for sid in session_ids {
        let destroy = PolarisSessionDestroyArg {
            session_id: sid,
            ..Default::default()
        };
        ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &destroy).ok();
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// ─── Comprehensive COW Break Test (Phase 3 validation) ────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

fn run_cow_break_test(fd: c_int) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Comprehensive COW Break Test ===");
    let mut passed = 0u32;
    let mut failed = 0u32;

    /// Read global stats and return (shared_mib, private_mib, cow_breaks, cow_copy_mib).
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

    /// Shortcut: BLOCK_GROW with specified tokens and flags. Returns (ret_code, block_id).
    fn grow(fd: c_int, sid: u64, start: u32, count: u32, flags: u32, phase: u32) -> (i32, u64) {
        let mut arg = PolarisBlockGrowArg {
            session_id: sid,
            token_start: start,
            token_count: count,
            flags,
            phase,
            ..Default::default()
        };
        match ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_GROW, &mut arg) {
            Ok(()) => (arg.ret_code, arg.block_id),
            Err(eno) => (-(eno as i32), arg.block_id),
        }
    }

    /// Shortcut: SESSION_BRANCH. Returns child_session_id.
    fn branch(fd: c_int, parent: u64) -> u64 {
        let mut arg = PolarisSessionBranchArg {
            parent_session_id: parent,
            ..Default::default()
        };
        let _ = ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_BRANCH, &mut arg);
        arg.child_session_id
    }

    /// Shortcut: SESSION_CREATE. Returns session_id.
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

    /// Shortcut: SESSION_DESTROY.
    fn destroy_session(fd: c_int, sid: u64) {
        let arg = PolarisSessionDestroyArg {
            session_id: sid,
            ..Default::default()
        };
        let _ = ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &arg);
    }

    // ── Setup: parent + 8 prompt blocks (16 tokens each, 0..127) ─────────
    let parent = create_session(fd, 5);
    eprintln!("Parent session: {parent}");

    let mut block_ids: Vec<u64> = Vec::new();
    for i in 0..8u32 {
        let (rc, bid) = grow(fd, parent, i * 16, 16, 0, PolarisPhase::Prefill as u32);
        if rc != 0 {
            eprintln!("  FAIL: parent block {i} allocation failed (rc={rc})");
            destroy_session(fd, parent);
            return Err(format!("setup failed at block {i}").into());
        }
        block_ids.push(bid);
    }
    eprintln!("Parent prompt: 8 blocks allocated");

    // ── Branch: 3 children share parent's blocks ─────────────────────────
    let mut children: Vec<u64> = Vec::new();
    for _ in 0..3 {
        let cid = branch(fd, parent);
        children.push(cid);
    }
    let (sh, pr, _, _) = get_stats(fd);
    eprintln!("After branch: shared={sh} MiB  private={pr} MiB");

    // ── Test A: first child COW-breaks block 0 ───────────────────────────
    eprintln!("\n--- Test A: first child COW-break block 0 ---");
    let c1 = children[0];
    let (rc, new_bid_a) = grow(fd, c1, 0, 16, POLARIS_GROW_FLAG_OVERWRITE, PolarisPhase::Prefill as u32);
    let (_sh_a, _pr_a, cb_a, cc_a) = get_stats(fd);
    if rc == 0 && cb_a == 1 && new_bid_a != block_ids[0] {
        eprintln!("  PASS: rc=0, new_block={new_bid_a}, cow_breaks={cb_a}, cow_copy={cc_a} MiB");
        passed += 1;
    } else {
        eprintln!("  FAIL: rc={rc}, new_block={new_bid_a}, cow_breaks={cb_a}, cow_copy={cc_a} MiB");
        failed += 1;
    }

    // ── Test B: second child COW-breaks the SAME block 0 (still shared) ──
    eprintln!("\n--- Test B: second child COW-break same block 0 ---");
    let c2 = children[1];
    let (rc, new_bid_b) = grow(fd, c2, 0, 16, POLARIS_GROW_FLAG_OVERWRITE, PolarisPhase::Prefill as u32);
    let (_sh_b, _pr_b, cb_b, _cc_b) = get_stats(fd);
    if rc == 0 && cb_b == 2 && new_bid_b != block_ids[0] && new_bid_b != new_bid_a {
        eprintln!("  PASS: rc=0, new_block={new_bid_b}, cow_breaks={cb_b}");
        passed += 1;
    } else {
        eprintln!("  FAIL: rc={rc}, new_block={new_bid_b}, cow_breaks={cb_b}");
        failed += 1;
    }

    // ── Test C: first child in-place overwrite (block now private) ───────
    eprintln!("\n--- Test C: first child in-place overwrite (refcount=1) ---");
    let (rc_c, bid_c) = grow(fd, c1, 0, 16, POLARIS_GROW_FLAG_OVERWRITE, PolarisPhase::Prefill as u32);
    let (_sh_c, _pr_c, cb_c, _) = get_stats(fd);
    // In-place overwrite: should return the SAME block_id, no new COW break.
    if rc_c == 0 && bid_c == new_bid_a && cb_c == 2 {
        eprintln!("  PASS: in-place returned same block_id={bid_c}, cow_breaks still {cb_c}");
        passed += 1;
    } else {
        eprintln!("  FAIL: rc={rc_c}, block_id={bid_c} (expected {new_bid_a}), cow_breaks={cb_c}");
        failed += 1;
    }

    // ── Test D: overlap without OVERWRITE → should get EINVAL ────────────
    eprintln!("\n--- Test D: overlap without OVERWRITE flag ---");
    let (rc_d, bid_d) = grow(fd, children[2], 0, 16, 0, PolarisPhase::Prefill as u32);
    // POLARIS_BLOCK_GROW with overlap and no OVERWRITE → kernel returns EINVAL (22)
    if rc_d == -(libc::EINVAL as i32) {
        eprintln!("  PASS: rejected with EINVAL ({rc_d})");
        passed += 1;
    } else {
        eprintln!("  FAIL: expected EINVAL ({}), got rc={rc_d} block_id={bid_d}", -(libc::EINVAL as i32));
        failed += 1;
    }

    // ── Test E: third child COW-breaks a DIFFERENT shared block ──────────
    eprintln!("\n--- Test E: third child COW-break block 1 (still shared) ---");
    // Block 0 has refcount decremented by first two children's COW breaks.
    // Block 1 still has original refcount (parent + 3 children = 4).
    let c3 = children[2];
    let (rc_e, new_bid_e) = grow(fd, c3, 16, 16, POLARIS_GROW_FLAG_OVERWRITE, PolarisPhase::Prefill as u32);
    let (_sh_e, _pr_e, cb_e, _) = get_stats(fd);
    if rc_e == 0 && cb_e == 3 && new_bid_e != block_ids[1] {
        eprintln!("  PASS: rc=0, new_block={new_bid_e}, cow_breaks={cb_e}");
        passed += 1;
    } else {
        eprintln!("  FAIL: rc={rc_e}, new_block={new_bid_e}, cow_breaks={cb_e}");
        failed += 1;
    }

    // ── Test F: nested branch + COW break ────────────────────────────────
    eprintln!("\n--- Test F: nested branch + COW break in grandchild ---");
    // Branch from first child (which already did a COW break).
    // The grandchild shares c1's blocks (including the private block from Test A,
    // plus shared blocks from the parent).
    let grandchild = branch(fd, c1);
    eprintln!("  Grandchild session: {grandchild}");

    // Grandchild COW-breaks block 2 (should still be shared among parent + 3 children = 4 refs).
    let (rc_f, new_bid_f) = grow(fd, grandchild, 32, 16, POLARIS_GROW_FLAG_OVERWRITE, PolarisPhase::Prefill as u32);
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

    // ── Test G: grandchild in-place overwrite of block 0 ─────────────────
    // Block 0 was COW-broken by c1; grandchild inherited c1's private copy.
    // grandchild's refcount for that block should be 2 (c1 + grandchild).
    // So COW break should trigger again.
    eprintln!("\n--- Test G: grandchild COW-break block 0 (shared with c1) ---");
    let (rc_g, new_bid_g) = grow(fd, grandchild, 0, 16, POLARIS_GROW_FLAG_OVERWRITE, PolarisPhase::Prefill as u32);
    let (_sh_g, _pr_g, cb_g, _) = get_stats(fd);
    if rc_g == 0 && cb_g == 5 && new_bid_g != new_bid_a {
        eprintln!("  PASS: rc=0, new_block={new_bid_g} (different from c1's {new_bid_a}), cow_breaks={cb_g}");
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
