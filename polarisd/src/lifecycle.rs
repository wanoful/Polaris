// Daemon lifecycle and crash-recovery module.
//
// Handles:
//   1. State reconciliation when the daemon (re)starts — reads kernel stats
//      from /sys/kernel/polaris/stats to understand the aftermath of a
//      previous daemon crash or clean shutdown.
//   2. Health-check signals for systemd (sd_notify ready/reloading/stopping).
//
// The kernel module is the sole authority for all state.  The daemon's job
// during reconciliation is purely diagnostic: log how many blocks were
// evicted by the crash, warn if orphaned sessions remain, and confirm
// re-attachment succeeded.

use std::fs;
use std::collections::HashMap;

/// Parsed snapshot of /sys/kernel/polaris/stats.
#[derive(Debug, Default)]
pub struct KernelStats {
    pub sessions: u64,
    pub blocks: u64,
    pub blocks_resident: u64,
    pub blocks_offloaded: u64,
    pub blocks_evicted: u64,
    pub blocks_pending: u64,
    pub gpus: u64,
    pub gpus_unhealthy: u64,
    pub daemon_attached: u64,
    pub gpu_total_mib: u64,
    pub gpu_used_mib: u64,
    pub cpu_pool_mib: u64,
    pub cpu_used_mib: u64,
    pub shared_mib: u64,
    pub private_mib: u64,
    pub pending_decisions: u64,
}

/// Read and parse /sys/kernel/polaris/stats.
fn read_kernel_stats() -> Result<KernelStats, String> {
    let raw = fs::read_to_string("/sys/kernel/polaris/stats")
        .map_err(|e| format!("Cannot open /sys/kernel/polaris/stats: {e}"))?;

    let mut map = HashMap::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.splitn(2, ':').collect();
        if parts.len() != 2 {
            continue;
        }
        let key = parts[0].trim().to_string();
        let value = parts[1].trim();
        map.insert(key, value.to_string());
    }

    let get_u64 = |key: &str| -> u64 {
        map.get(key)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    };

    Ok(KernelStats {
        sessions: get_u64("sessions"),
        blocks: get_u64("blocks"),
        blocks_resident: get_u64("resident"),
        blocks_offloaded: get_u64("offloaded"),
        blocks_evicted: get_u64("evicted"),
        blocks_pending: get_u64("pending"),
        gpus: get_u64("gpus"),
        gpus_unhealthy: get_u64("unhealthy"),
        daemon_attached: get_u64("daemon"),
        gpu_total_mib: get_u64("gpu_total_mib"),
        gpu_used_mib: get_u64("gpu_used_mib"),
        cpu_pool_mib: get_u64("cpu_pool_mib"),
        cpu_used_mib: get_u64("cpu_used_mib"),
        shared_mib: get_u64("shared_mib"),
        private_mib: get_u64("private_mib"),
        pending_decisions: get_u64("pending_decs"),
    })
}

/// Reconcile daemon state after (re)attaching to the kernel module.
///
/// Called once after GPU registration.  Returns `true` if the state looks
/// clean (fresh start or successful restart), `false` if warnings were
/// logged that the operator should investigate.
pub fn reconcile() {
    match read_kernel_stats() {
        Ok(stats) => {
            eprintln!("polarisd: ── state reconciliation ──────────────────");

            if stats.daemon_attached == 0 {
                eprintln!(
                    "polarisd: WARNING kernel reports daemon_attached=0 — "
                );
                eprintln!(
                    "polarisd:   REGISTER_GPU may not have taken effect.  "
                );
                eprintln!(
                    "polarisd:   Check 'dmesg' for kernel errors."
                );
            } else {
                eprintln!(
                    "polarisd: daemon attached (kernel count: {})",
                    stats.daemon_attached
                );
            }

            if stats.sessions > 0 {
                eprintln!(
                    "polarisd: WARNING {} session(s) still present in the kernel",
                    stats.sessions
                );
                eprintln!(
                    "polarisd:   These may be orphaned from a previous daemon.  "
                );
                eprintln!(
                    "polarisd:   They will be cleaned up on DESTROY or "
                );
                eprintln!(
                    "polarisd:   when the workload process exits."
                );
            }

            if stats.pending_decisions > 0 {
                eprintln!(
                    "polarisd: WARNING {} pending decision(s) in kernel queue",
                    stats.pending_decisions
                );
                eprintln!(
                    "polarisd:   Previous daemon may not have drained its queue "
                );
                eprintln!(
                    "polarisd:   before exiting."
                );
            }

            if stats.blocks_evicted > 0 {
                eprintln!(
                    "polarisd: INFO {} block(s) were evicted (likely from a previous crash)",
                    stats.blocks_evicted
                );
            }

            if stats.blocks_offloaded > 0 {
                eprintln!(
                    "polarisd: WARNING {} block(s) marked as CPU_OFFLOADED",
                    stats.blocks_offloaded
                );
                eprintln!(
                    "polarisd:   These blocks need reloading before use.  "
                );
                eprintln!(
                    "polarisd:   The kernel will queue RELOAD decisions "
                );
                eprintln!(
                    "polarisd:   when workloads touch them."
                );
            }

            if stats.gpus_unhealthy > 0 {
                eprintln!(
                    "polarisd: CRITICAL {} GPU(s) marked unhealthy — "
                    ,
                    stats.gpus_unhealthy
                );
                eprintln!(
                    "polarisd:   No new sessions will be accepted on these GPUs.  "
                );
                eprintln!(
                    "polarisd:   Check hardware status and reload polaris.ko "
                );
                eprintln!(
                    "polarisd:   to reset the health flag."
                );
            }

            eprintln!(
                "polarisd: block summary: {} total, {} resident, {} offloaded, {} evicted",
                stats.blocks, stats.blocks_resident, stats.blocks_offloaded, stats.blocks_evicted,
            );
            eprintln!(
                "polarisd: GPU memory: {} MiB used / {} MiB total",
                stats.gpu_used_mib, stats.gpu_total_mib,
            );
            eprintln!("polarisd: ─────────────────────────────────────────");
        }
        Err(e) => {
            eprintln!(
                "polarisd: WARNING state reconciliation skipped: {e}",
            );
            eprintln!(
                "polarisd:   Is polaris.ko loaded and /sys/kernel/polaris/stats readable?"
            );
        }
    }
}

/// Notify systemd that the daemon has started and is ready.
/// Sends READY=1 over the $NOTIFY_SOCKET Unix datagram socket.
/// No-op when NOTIFY_SOCKET is unset (not running under systemd).
pub fn notify_ready() {
    if let Ok(socket_path) = std::env::var("NOTIFY_SOCKET") {
        let _ = send_notify_message(&socket_path, "READY=1");
    }
}

/// Notify systemd that the daemon is stopping.
pub fn notify_stopping() {
    if let Ok(socket_path) = std::env::var("NOTIFY_SOCKET") {
        let _ = send_notify_message(&socket_path, "STOPPING=1");
    }
}

fn send_notify_message(socket_path: &str, msg: &str) -> Result<(), String> {
    use std::os::unix::net::UnixDatagram;
    let sock = UnixDatagram::unbound()
        .map_err(|e| format!("UnixDatagram::unbound: {e}"))?;
    sock.send_to(msg.as_bytes(), socket_path)
        .map_err(|e| format!("send_to NOTIFY_SOCKET: {e}"))?;
    Ok(())
}
