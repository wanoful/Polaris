// POLARIS CLI Tool (polarisctl)
//
// Commands:
//   polarisctl stats      — read global stats from the kernel module
//   polarisctl session    — create/destroy/list sessions
//   polarisctl block      — inspect block table
//
// Phase 1a skeleton: stats command only.

use clap::{Parser, Subcommand};
use libc::c_int;
use libpolaris::ioctl;
use libpolaris::types::*;
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;

#[derive(Parser)]
#[command(name = "polarisctl", about = "POLARIS CLI tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show global POLARIS statistics
    Stats,
    /// Create a new session
    CreateSession {
        /// GPU ID
        #[arg(long, default_value = "0")]
        gpu: u32,
        /// Beam width
        #[arg(long, default_value = "1")]
        beam: u32,
        /// GPU virtual address space in bytes
        #[arg(long, default_value_t = 1024 * 1024 * 1024)]
        vas_bytes: u64,
        /// Bytes per token (0 = use kernel default 524288)
        #[arg(long, default_value = "0")]
        bytes_per_token: u64,
    },
    /// Destroy a session
    DestroySession {
        /// Session ID
        session_id: u64,
    },
    /// List all sessions
    ListSessions,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/polaris")
        .map_err(|e| format!("Failed to open /dev/polaris: {e}"))?;

    let fd = file.as_raw_fd() as c_int;

    match cli.command {
        Commands::Stats => cmd_stats(fd)?,
        Commands::CreateSession { gpu, beam, vas_bytes, bytes_per_token } => {
            cmd_create_session(fd, gpu, beam, vas_bytes, bytes_per_token)?
        }
        Commands::DestroySession { session_id } => cmd_destroy_session(fd, session_id)?,
        Commands::ListSessions => cmd_list_sessions(fd)?,
    }

    Ok(())
}

fn cmd_stats(fd: c_int) -> Result<(), Box<dyn std::error::Error>> {
    let mut arg = PolarisGetGlobalStatsArg::default();
    ioctl::ioctl_read(fd, ioctl::POLARIS_GET_GLOBAL_STATS, &mut arg)
        .map_err(|e| format!("GET_GLOBAL_STATS failed: errno {e}"))?;

    println!("POLARIS Global Statistics");
    println!("=========================");
    println!("GPUs:              {}", arg.total_gpus);
    println!("Sessions:          {}", arg.total_sessions);
    println!("Blocks:            {}", arg.total_blocks);
    println!("  Resident:        {}", arg.blocks_resident);
    println!("  Offloaded:       {}", arg.blocks_offloaded);
    println!("  Evicted:         {}", arg.blocks_evicted);
    println!("GPU memory:");
    println!("  Total:           {} MiB", arg.total_gpu_bytes / (1024 * 1024));
    println!("  Used:            {} MiB", arg.used_gpu_bytes / (1024 * 1024));
    println!("  Shared:          {} MiB", arg.shared_gpu_bytes / (1024 * 1024));
    println!("  Private:         {} MiB", arg.private_gpu_bytes / (1024 * 1024));
    println!("CPU pool:");
    println!("  Total:           {} MiB", arg.cpu_pool_total / (1024 * 1024));
    println!("  Used:            {} MiB", arg.cpu_pool_used / (1024 * 1024));
    println!("COW:");
    println!("  Break count:     {}", arg.cow_break_count);
    println!("  Copy bytes:      {}", arg.cow_copy_bytes);

    Ok(())
}

fn cmd_create_session(
    fd: c_int,
    gpu: u32,
    beam: u32,
    vas_bytes: u64,
    bytes_per_token: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut arg = PolarisSessionCreateArg {
        home_gpu: gpu,
        beam_width: beam,
        gpu_vas_bytes: vas_bytes,
        bytes_per_token,
        ..Default::default()
    };

    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut arg)
        .map_err(|e| format!("SESSION_CREATE failed: errno {e}"))?;

    println!("Session created: id={}", arg.session_id);
    Ok(())
}

fn cmd_destroy_session(fd: c_int, session_id: u64) -> Result<(), Box<dyn std::error::Error>> {
    let arg = PolarisSessionDestroyArg {
        session_id,
        ..Default::default()
    };

    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &arg)
        .map_err(|e| format!("SESSION_DESTROY failed: errno {e}"))?;

    println!("Session {session_id} destroyed");
    Ok(())
}

fn cmd_list_sessions(fd: c_int) -> Result<(), Box<dyn std::error::Error>> {
    let mut arg = PolarisListSessionsArg::default();
    ioctl::ioctl_read(fd, ioctl::POLARIS_LIST_SESSIONS, &mut arg)
        .map_err(|e| format!("LIST_SESSIONS failed: errno {e}"))?;

    let count = arg.count as usize;
    println!("Sessions: {count}");
    for i in 0..count {
        let sid = arg.session_ids[i];
        let mut info = PolarisSessionGetStatsArg {
            session_id: sid,
            ..Default::default()
        };
        match ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_GET_STATS, &mut info) {
            Ok(()) => {
                println!(
                    "  id={} gpu={} beam={} blocks={} vas_bytes={}",
                    sid, info.home_gpu, info.beam_width, info.num_blocks, info.total_bytes
                );
            }
            Err(e) => {
                println!("  id={} (error fetching details: errno {e})", sid);
            }
        }
    }
    Ok(())
}
