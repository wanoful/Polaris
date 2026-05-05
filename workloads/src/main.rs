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
        } => run_beam_search(fd, beam_width, decode_steps)?,
        Commands::Concurrent {
            num_sessions,
            blocks_per_session,
        } => run_concurrent(fd, num_sessions, blocks_per_session)?,
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

fn run_beam_search(fd: c_int, beam_width: u32, decode_steps: u32) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Beam Search Workload ===");
    eprintln!("Beam width: {beam_width}, Decode steps: {decode_steps}");

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
