use libpolaris::ioctl;
use libpolaris::types::{
    PolarisBlockGetStateArg, PolarisBlockReleaseArg, PolarisBlockReserveArg, PolarisBlockState,
    PolarisCompleteOperationArg, PolarisDecisionOp, PolarisGetDecisionArg, PolarisPhase,
    PolarisRegisterBlockMappingArg, PolarisRegisterGpuArg, PolarisRegisterVaRangeArg,
    PolarisRegisterVaSpaceArg, PolarisSessionBranchArg, PolarisSessionCreateArg,
    PolarisSessionDestroyArg, PolarisSetPolicyArg, PolarisSpillBlockArg,
    PolarisUnregisterVaSpaceArg, POLARIS_MAX_DECISIONS_PER_POLL,
    POLARIS_REGISTER_GPU_FLAG_TRANSIENT, POLARIS_RESERVE_FLAG_DEFER_FAULT,
    POLARIS_RESERVE_FLAG_OVERWRITE,
};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const MIB: u64 = 1024 * 1024;
const BLOCK_SIZE: u64 = 2 * MIB;

#[derive(Default)]
struct SeenOps {
    alloc: u32,
    free: u32,
    offload: u32,
    reload: u32,
    cow_break: u32,
    last_cow_src_handle: u64,
    last_cow_dst_vaddr: u64,
    last_free_src_handle: u64,
    last_free_cpu_addr: u64,
    last_free_block_id: u64,
}

#[derive(Clone, Copy, Default)]
struct FakeExecutorConfig {
    complete_alloc_with_rm_backing: bool,
}

fn open_polaris() -> std::fs::File {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/polaris")
        .expect("open /dev/polaris; run this ignored test as root")
}

fn sysfs_stats() -> HashMap<String, u64> {
    let stats = std::fs::read_to_string("/sys/kernel/polaris/stats")
        .expect("read /sys/kernel/polaris/stats");
    stats
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            let first = value.split_whitespace().next()?;
            Some((key.trim().to_string(), first.parse::<u64>().ok()?))
        })
        .collect()
}

fn assert_stat(name: &str, expected: u64) {
    let stats = sysfs_stats();
    assert_eq!(
        stats.get(name).copied(),
        Some(expected),
        "stat {name} did not match expected value; stats={stats:?}"
    );
}

struct PolarisdChild {
    child: Child,
}

impl Drop for PolarisdChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn wait_for_stat(name: &str, expected: u64, context: &str) {
    let started = Instant::now();
    loop {
        let stats = sysfs_stats();
        if stats.get(name).copied() == Some(expected) {
            return;
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "stat {name} did not become {expected} while waiting for {context}; stats={stats:?}"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn wait_for_stat_at_least(name: &str, expected_min: u64, context: &str) {
    let started = Instant::now();
    loop {
        let stats = sysfs_stats();
        if stats.get(name).copied().unwrap_or(0) >= expected_min {
            return;
        }
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "stat {name} did not reach at least {expected_min} while waiting for {context}; stats={stats:?}"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn polarisd_bin() -> PathBuf {
    if let Some(path) = std::env::var_os("POLARISD_BIN") {
        return PathBuf::from(path);
    }

    let current = std::env::current_exe().expect("current test executable path");
    let deps_dir = current
        .parent()
        .expect("test executable has parent directory");
    let target_debug = deps_dir
        .parent()
        .expect("test executable is under target/debug/deps");
    target_debug.join("polarisd")
}

fn start_polarisd_rm_backing() -> PolarisdChild {
    let child = Command::new(polarisd_bin())
        .env("POLARISD_RM_BACKING", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn polarisd with POLARISD_RM_BACKING=1");
    wait_for_stat("daemon", 1, "polarisd attach");
    wait_for_stat_at_least("gpus", 1, "polarisd GPU registration");
    PolarisdChild { child }
}

fn env_u64(name: &str) -> u64 {
    std::env::var(name)
        .unwrap_or_else(|_| panic!("{name} not set"))
        .parse()
        .unwrap_or_else(|_| panic!("{name} is not a u64"))
}

fn maybe_run_worker_exit_child() -> bool {
    if std::env::var("POLARIS_WORKER_EXIT_CHILD").ok().as_deref() != Some("1") {
        return false;
    }

    let dev = open_polaris();
    let fd = dev.as_raw_fd();
    let block_id = env_u64("POLARIS_CHILD_BLOCK_ID");
    let vas = PolarisRegisterVaSpaceArg {
        gpu_id: 0,
        rm_client_token: env_u64("POLARIS_CHILD_CLIENT"),
        va_space_token: env_u64("POLARIS_CHILD_TOKEN"),
        managed_base: env_u64("POLARIS_CHILD_BASE"),
        managed_length: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::register_vaspace(fd, &vas).expect("child POLARIS_REGISTER_VASPACE");
    let mapping = PolarisRegisterBlockMappingArg {
        block_id,
        gpu_id: vas.gpu_id,
        rm_client_token: vas.rm_client_token,
        va_space_token: vas.va_space_token,
        base: vas.managed_base,
        length: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::register_block_mapping(fd, &mapping).expect("child POLARIS_REGISTER_BLOCK_MAPPING");

    if let Ok(path) = std::env::var("POLARIS_CHILD_READY_PATH") {
        std::fs::write(path, b"ready\n").expect("write child ready marker");
    }

    thread::sleep(Duration::from_secs(30));
    true
}

fn get_block_state(fd: i32, session_id: u64, block_id: u64) -> PolarisBlockGetStateArg {
    let mut state = PolarisBlockGetStateArg {
        session_id,
        block_id,
        ..Default::default()
    };
    ioctl::block_get_state(fd, &mut state).expect("POLARIS_BLOCK_GET_STATE");
    state
}

fn wait_for_state(
    fd: i32,
    session_id: u64,
    block_id: u64,
    expected: PolarisBlockState,
) -> PolarisBlockGetStateArg {
    let started = Instant::now();
    loop {
        let state = get_block_state(fd, session_id, block_id);
        if state.state == expected as u32 {
            return state;
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "block {block_id} state={} did not become {:?}",
            state.state,
            expected
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn register_transient_gpu_and_range_with_budget(
    fd: i32,
    base: u64,
    length: u64,
    budget_bytes: u64,
) {
    let gpu = PolarisRegisterGpuArg {
        gpu_id: 0,
        total_bytes: 16 * 1024 * MIB,
        budget_bytes,
        cpu_pool_bytes: 64 * MIB,
        _reserved: POLARIS_REGISTER_GPU_FLAG_TRANSIENT,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_REGISTER_GPU, &gpu).expect("POLARIS_REGISTER_GPU");

    let mut range = PolarisRegisterVaRangeArg {
        gpu_id: 0,
        base,
        length,
        block_size: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_REGISTER_VA_RANGE, &mut range)
        .expect("POLARIS_REGISTER_VA_RANGE");
}

fn register_transient_gpu_and_range(fd: i32, base: u64, length: u64) {
    register_transient_gpu_and_range_with_budget(fd, base, length, 512 * MIB);
}

fn wait_for_seen<F>(seen: &Arc<Mutex<SeenOps>>, predicate: F, context: &str)
where
    F: Fn(&SeenOps) -> bool,
{
    let started = Instant::now();
    loop {
        {
            let seen_ops = seen.lock().expect("seen mutex poisoned");
            if predicate(&seen_ops) {
                return;
            }
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "timed out waiting for {context}"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn start_fake_executor_with_config(
    stop: Arc<AtomicBool>,
    seen: Arc<Mutex<SeenOps>>,
    config: FakeExecutorConfig,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let dev = open_polaris();
        let fd = dev.as_raw_fd();
        let mut decisions = PolarisGetDecisionArg::default();

        while !stop.load(Ordering::Acquire) {
            ioctl::ioctl_read(fd, ioctl::POLARIS_GET_DECISION, &mut decisions)
                .expect("POLARIS_GET_DECISION");
            let count = (decisions.count as usize).min(POLARIS_MAX_DECISIONS_PER_POLL);
            if count == 0 {
                thread::sleep(Duration::from_millis(5));
                continue;
            }

            for decision in decisions.decisions.iter().take(count) {
                let mut output_handle = 0;
                let mut output_cpu_addr = 0;
                let mut rm_control_fd = 0;
                let mut rm_h_client = 0;
                let mut rm_h_memory = 0;
                let mut rm_backing_length = 0;

                {
                    let mut seen = seen.lock().expect("seen mutex poisoned");
                    match decision.op {
                        x if x == PolarisDecisionOp::Alloc as u32 => {
                            seen.alloc += 1;
                            if config.complete_alloc_with_rm_backing {
                                rm_control_fd = fd;
                                rm_h_client = 0xabc0_0000u32.wrapping_add(decision.block_id as u32);
                                rm_h_memory = 0xdef0_0000u32.wrapping_add(decision.block_id as u32);
                                rm_backing_length = decision.size_bytes;
                            } else {
                                output_handle = 0x1000_0000 + decision.block_id;
                            }
                        }
                        x if x == PolarisDecisionOp::Free as u32 => {
                            seen.free += 1;
                            seen.last_free_src_handle = decision.src_handle;
                            seen.last_free_cpu_addr = decision.cpu_addr;
                            seen.last_free_block_id = decision.block_id;
                        }
                        x if x == PolarisDecisionOp::Offload as u32 => {
                            seen.offload += 1;
                            output_cpu_addr = 0x2000_0000 + decision.block_id * BLOCK_SIZE;
                        }
                        x if x == PolarisDecisionOp::Reload as u32 => {
                            seen.reload += 1;
                            output_handle = 0x3000_0000 + decision.block_id;
                        }
                        x if x == PolarisDecisionOp::CowBreak as u32 => {
                            seen.cow_break += 1;
                            seen.last_cow_src_handle = decision.src_handle;
                            seen.last_cow_dst_vaddr = decision.dst_vaddr;
                            output_handle = 0x4000_0000 + decision.block_id;
                        }
                        other => panic!("unexpected decision op {other}"),
                    }
                }

                let complete = PolarisCompleteOperationArg {
                    decision_id: decision.decision_id,
                    generation: decision.generation,
                    result: 0,
                    rm_control_fd,
                    output_handle,
                    output_cpu_addr,
                    rm_h_client,
                    rm_h_memory,
                    rm_backing_length,
                    ..Default::default()
                };
                ioctl::ioctl_write(fd, ioctl::POLARIS_COMPLETE_OPERATION, &complete)
                    .expect("POLARIS_COMPLETE_OPERATION");
            }
        }
    })
}

fn start_fake_executor(stop: Arc<AtomicBool>, seen: Arc<Mutex<SeenOps>>) -> thread::JoinHandle<()> {
    start_fake_executor_with_config(stop, seen, FakeExecutorConfig::default())
}

#[test]
#[ignore = "internal helper for worker_process_exit_reaps_vaspace_and_block_mapping"]
fn worker_exit_child_process_entrypoint() {
    let _ = maybe_run_worker_exit_child();
}

#[test]
#[ignore = "requires root and a loaded polaris.ko; exercises kernel spill/reload state machine without CUDA"]
fn spill_block_queues_offload_and_reload_decisions() {
    let dev = open_polaris();
    let fd = dev.as_raw_fd();

    let stop = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(Mutex::new(SeenOps::default()));
    let executor = start_fake_executor(Arc::clone(&stop), Arc::clone(&seen));

    register_transient_gpu_and_range(fd, 0x1000_0000_0000, 4 * BLOCK_SIZE);

    let mut session = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut session)
        .expect("POLARIS_SESSION_CREATE");

    let mut reserve = PolarisBlockReserveArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE alloc");
    assert_ne!(reserve.block_id, 0);
    assert_ne!(reserve.gpu_vaddr, 0);
    wait_for_state(
        fd,
        session.session_id,
        reserve.block_id,
        PolarisBlockState::Resident,
    );

    let mut spill = PolarisSpillBlockArg {
        block_id: reserve.block_id,
        ..Default::default()
    };
    ioctl::spill_block(fd, &mut spill).expect("POLARIS_SPILL_BLOCK");
    assert_ne!(spill.decision_id, 0);
    wait_for_state(
        fd,
        session.session_id,
        reserve.block_id,
        PolarisBlockState::CpuOffloaded,
    );

    let mut reload = PolarisBlockReserveArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        flags: POLARIS_RESERVE_FLAG_OVERWRITE,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reload)
        .expect("POLARIS_BLOCK_RESERVE reload");
    assert_eq!(reload.block_id, reserve.block_id);
    assert_eq!(reload.gpu_vaddr, reserve.gpu_vaddr);
    wait_for_state(
        fd,
        session.session_id,
        reserve.block_id,
        PolarisBlockState::Resident,
    );

    let seen_ops = seen.lock().expect("seen mutex poisoned");
    assert_eq!(seen_ops.alloc, 1);
    assert_eq!(seen_ops.offload, 1);
    assert_eq!(seen_ops.reload, 1);
    drop(seen_ops);

    let destroy = PolarisSessionDestroyArg {
        session_id: session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &destroy)
        .expect("POLARIS_SESSION_DESTROY");
    wait_for_seen(
        &seen,
        |ops| ops.free >= 1,
        "FREE decision after spill test destroy",
    );

    stop.store(true, Ordering::Release);
    executor.join().expect("fake executor join");
}

#[test]
#[ignore = "requires root and freshly loaded polaris.ko; exercises BLOCK_RELEASE FREE queueing without CUDA"]
fn block_release_queues_free_for_resident_block() {
    let dev = open_polaris();
    let fd = dev.as_raw_fd();

    let stop = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(Mutex::new(SeenOps::default()));
    let executor = start_fake_executor(Arc::clone(&stop), Arc::clone(&seen));

    register_transient_gpu_and_range(fd, 0x1200_0000_0000, 4 * BLOCK_SIZE);

    let mut session = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut session)
        .expect("POLARIS_SESSION_CREATE");

    let mut reserve = PolarisBlockReserveArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE alloc");
    wait_for_state(
        fd,
        session.session_id,
        reserve.block_id,
        PolarisBlockState::Resident,
    );

    let release = PolarisBlockReleaseArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_BLOCK_RELEASE, &release)
        .expect("POLARIS_BLOCK_RELEASE");

    wait_for_seen(
        &seen,
        |ops| ops.free == 1 && ops.last_free_src_handle == 0x1000_0000 + reserve.block_id,
        "FREE decision after resident BLOCK_RELEASE",
    );
    wait_for_stat("blocks", 0, "resident BLOCK_RELEASE FREE completion");

    let destroy = PolarisSessionDestroyArg {
        session_id: session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &destroy)
        .expect("POLARIS_SESSION_DESTROY");

    stop.store(true, Ordering::Release);
    executor.join().expect("fake executor join");
}

#[test]
#[ignore = "requires root and freshly loaded polaris.ko; exercises RM-backed BLOCK_RELEASE FREE queueing without CUDA"]
fn block_release_queues_free_for_rm_backed_block_without_phys_handle() {
    let dev = open_polaris();
    let fd = dev.as_raw_fd();

    let stop = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(Mutex::new(SeenOps::default()));
    let executor = start_fake_executor_with_config(
        Arc::clone(&stop),
        Arc::clone(&seen),
        FakeExecutorConfig {
            complete_alloc_with_rm_backing: true,
        },
    );

    register_transient_gpu_and_range(fd, 0x1300_0000_0000, 4 * BLOCK_SIZE);

    let mut session = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut session)
        .expect("POLARIS_SESSION_CREATE");

    let mut reserve = PolarisBlockReserveArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE RM-backed alloc");
    wait_for_state(
        fd,
        session.session_id,
        reserve.block_id,
        PolarisBlockState::Resident,
    );

    let release = PolarisBlockReleaseArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_BLOCK_RELEASE, &release)
        .expect("POLARIS_BLOCK_RELEASE RM-backed");

    wait_for_seen(
        &seen,
        |ops| ops.free == 1 && ops.last_free_block_id == reserve.block_id,
        "FREE decision after RM-backed BLOCK_RELEASE",
    );
    wait_for_stat("blocks", 0, "RM-backed BLOCK_RELEASE FREE completion");

    let destroy = PolarisSessionDestroyArg {
        session_id: session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &destroy)
        .expect("POLARIS_SESSION_DESTROY");

    stop.store(true, Ordering::Release);
    executor.join().expect("fake executor join");
}

#[test]
#[ignore = "requires root, freshly loaded polaris.ko, target/debug/polarisd, and a live NVIDIA RM stack"]
fn polarisd_rm_backing_alloc_and_free_decision_flow() {
    assert!(
        polarisd_bin().exists(),
        "build polarisd first: cargo build -p polarisd"
    );

    let _daemon = start_polarisd_rm_backing();

    let dev = open_polaris();
    let fd = dev.as_raw_fd();

    let mut session = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut session)
        .expect("POLARIS_SESSION_CREATE");

    let mut reserve = PolarisBlockReserveArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE through polarisd RM backing");
    assert_ne!(reserve.block_id, 0);
    assert_ne!(reserve.gpu_vaddr, 0);
    wait_for_state(
        fd,
        session.session_id,
        reserve.block_id,
        PolarisBlockState::Resident,
    );
    wait_for_stat("blocks", 1, "polarisd RM-backed ALLOC completion");

    let release = PolarisBlockReleaseArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_BLOCK_RELEASE, &release)
        .expect("POLARIS_BLOCK_RELEASE through polarisd RM backing");
    wait_for_stat("blocks", 0, "polarisd RM-backed FREE completion");
    wait_for_stat("pending_decs", 0, "polarisd RM-backed FREE queue drain");

    let destroy = PolarisSessionDestroyArg {
        session_id: session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &destroy)
        .expect("POLARIS_SESSION_DESTROY");
}

#[test]
#[ignore = "requires root, freshly loaded polaris.ko, target/debug/polarisd, and a live NVIDIA RM stack"]
fn clean_polarisd_shutdown_reaps_inserted_gpu() {
    assert!(
        polarisd_bin().exists(),
        "build polarisd first: cargo build -p polarisd"
    );

    {
        let _daemon = start_polarisd_rm_backing();
        wait_for_stat_at_least("gpus", 1, "polarisd GPU registration");
    }

    wait_for_stat("daemon", 0, "polarisd shutdown");
    wait_for_stat("gpus", 0, "polarisd GPU cleanup");
    assert_stat("gpu_total_mib", 0);
    assert_stat("gpu_budget_mib", 0);
    assert_stat("cpu_pool_mib", 0);
}

#[test]
#[ignore = "requires root, freshly loaded polaris.ko, target/debug/polarisd, and a live NVIDIA RM stack"]
fn transient_gpu_reregister_does_not_reap_daemon_gpu() {
    assert!(
        polarisd_bin().exists(),
        "build polarisd first: cargo build -p polarisd"
    );

    let _daemon = start_polarisd_rm_backing();
    let before = sysfs_stats();
    let daemon_gpus = before
        .get("gpus")
        .copied()
        .expect("gpus stat after polarisd start");
    assert!(daemon_gpus >= 1, "daemon did not register a GPU; stats={before:?}");
    let daemon_total_mib = before.get("gpu_total_mib").copied();
    let daemon_budget_mib = before.get("gpu_budget_mib").copied();
    let daemon_cpu_pool_mib = before.get("cpu_pool_mib").copied();

    {
        let shim = open_polaris();
        let shim_fd = shim.as_raw_fd();
        let gpu = PolarisRegisterGpuArg {
            gpu_id: 0,
            total_bytes: 16 * 1024 * MIB,
            budget_bytes: 512 * MIB,
            cpu_pool_bytes: 64 * MIB,
            _reserved: POLARIS_REGISTER_GPU_FLAG_TRANSIENT,
            ..Default::default()
        };
        ioctl::ioctl_write(shim_fd, ioctl::POLARIS_REGISTER_GPU, &gpu)
            .expect("transient POLARIS_REGISTER_GPU re-register");
        assert_stat("gpus", daemon_gpus);
        let during = sysfs_stats();
        assert_eq!(during.get("gpu_total_mib").copied(), daemon_total_mib);
        assert_eq!(during.get("gpu_budget_mib").copied(), daemon_budget_mib);
        assert_eq!(during.get("cpu_pool_mib").copied(), daemon_cpu_pool_mib);
    }

    wait_for_stat("daemon", 1, "daemon still attached after transient fd close");
    assert_stat("gpus", daemon_gpus);
    let after = sysfs_stats();
    assert_eq!(after.get("gpu_total_mib").copied(), daemon_total_mib);
    assert_eq!(after.get("gpu_budget_mib").copied(), daemon_budget_mib);
    assert_eq!(after.get("cpu_pool_mib").copied(), daemon_cpu_pool_mib);
}

#[test]
#[ignore = "requires root and freshly loaded polaris.ko; exercises RM-backed SESSION_DESTROY FREE queueing without CUDA"]
fn session_destroy_queues_free_for_rm_backed_block_without_phys_handle() {
    let dev = open_polaris();
    let fd = dev.as_raw_fd();

    let stop = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(Mutex::new(SeenOps::default()));
    let executor = start_fake_executor_with_config(
        Arc::clone(&stop),
        Arc::clone(&seen),
        FakeExecutorConfig {
            complete_alloc_with_rm_backing: true,
        },
    );

    register_transient_gpu_and_range(fd, 0x1400_0000_0000, 4 * BLOCK_SIZE);

    let mut session = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut session)
        .expect("POLARIS_SESSION_CREATE");

    let mut reserve = PolarisBlockReserveArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE RM-backed alloc");
    wait_for_state(
        fd,
        session.session_id,
        reserve.block_id,
        PolarisBlockState::Resident,
    );

    let destroy = PolarisSessionDestroyArg {
        session_id: session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &destroy)
        .expect("POLARIS_SESSION_DESTROY");

    wait_for_seen(
        &seen,
        |ops| ops.free == 1 && ops.last_free_block_id == reserve.block_id,
        "FREE decision after RM-backed SESSION_DESTROY",
    );
    wait_for_stat("blocks", 0, "RM-backed SESSION_DESTROY FREE completion");
    wait_for_stat("block_mappings", 0, "RM-backed SESSION_DESTROY mapping cleanup");

    stop.store(true, Ordering::Release);
    executor.join().expect("fake executor join");
}

#[test]
#[ignore = "requires root and a loaded polaris.ko; exercises fixed-policy budget scheduling without CUDA"]
fn reserve_under_budget_pressure_queues_offload_before_alloc() {
    let dev = open_polaris();
    let fd = dev.as_raw_fd();

    let stop = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(Mutex::new(SeenOps::default()));
    let executor = start_fake_executor(Arc::clone(&stop), Arc::clone(&seen));

    register_transient_gpu_and_range_with_budget(fd, 0x1800_0000_0000, 8 * BLOCK_SIZE, BLOCK_SIZE);
    let policy = PolarisSetPolicyArg {
        policy: 0,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SET_POLICY, &policy).expect("POLARIS_SET_POLICY FIFO");

    let mut first_session = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut first_session)
        .expect("POLARIS_SESSION_CREATE first");

    let mut first = PolarisBlockReserveArg {
        session_id: first_session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut first)
        .expect("POLARIS_BLOCK_RESERVE first");
    wait_for_state(
        fd,
        first_session.session_id,
        first.block_id,
        PolarisBlockState::Resident,
    );

    let mut second_session = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut second_session)
        .expect("POLARIS_SESSION_CREATE second");

    let mut second = PolarisBlockReserveArg {
        session_id: second_session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut second)
        .expect("POLARIS_BLOCK_RESERVE second");

    wait_for_state(
        fd,
        first_session.session_id,
        first.block_id,
        PolarisBlockState::CpuOffloaded,
    );
    wait_for_state(
        fd,
        second_session.session_id,
        second.block_id,
        PolarisBlockState::Resident,
    );

    let seen_ops = seen.lock().expect("seen mutex poisoned");
    assert_eq!(seen_ops.alloc, 2);
    assert_eq!(seen_ops.offload, 1);
    drop(seen_ops);

    let second_destroy = PolarisSessionDestroyArg {
        session_id: second_session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &second_destroy)
        .expect("POLARIS_SESSION_DESTROY second");
    let first_destroy = PolarisSessionDestroyArg {
        session_id: first_session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &first_destroy)
        .expect("POLARIS_SESSION_DESTROY first");
    wait_for_seen(
        &seen,
        |ops| ops.free >= 1,
        "FREE decision after budget-pressure scheduler test destroy",
    );

    stop.store(true, Ordering::Release);
    executor.join().expect("fake executor join");
}

#[test]
#[ignore = "requires root and a loaded polaris.ko; exercises kernel COW state machine without CUDA"]
fn branch_overwrite_queues_cow_break_for_child_session() {
    let dev = open_polaris();
    let fd = dev.as_raw_fd();

    let stop = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(Mutex::new(SeenOps::default()));
    let executor = start_fake_executor(Arc::clone(&stop), Arc::clone(&seen));

    register_transient_gpu_and_range(fd, 0x2000_0000_0000, 8 * BLOCK_SIZE);

    let mut parent = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut parent)
        .expect("POLARIS_SESSION_CREATE parent");

    let mut parent_reserve = PolarisBlockReserveArg {
        session_id: parent.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut parent_reserve)
        .expect("POLARIS_BLOCK_RESERVE parent");
    wait_for_state(
        fd,
        parent.session_id,
        parent_reserve.block_id,
        PolarisBlockState::Resident,
    );
    let parent_state = get_block_state(fd, parent.session_id, parent_reserve.block_id);
    assert_eq!(parent_state.refcount, 1);

    let mut branch = PolarisSessionBranchArg {
        parent_session_id: parent.session_id,
        ..Default::default()
    };
    ioctl::session_branch(fd, &mut branch).expect("POLARIS_SESSION_BRANCH");
    assert_ne!(branch.child_session_id, 0);

    let shared_parent_state = get_block_state(fd, parent.session_id, parent_reserve.block_id);
    assert_eq!(shared_parent_state.refcount, 2);
    let child_shared_state = get_block_state(fd, branch.child_session_id, parent_reserve.block_id);
    assert_eq!(child_shared_state.block_id, parent_reserve.block_id);
    assert_eq!(child_shared_state.refcount, 2);

    let mut child_overwrite = PolarisBlockReserveArg {
        session_id: branch.child_session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        flags: POLARIS_RESERVE_FLAG_OVERWRITE,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut child_overwrite)
        .expect("POLARIS_BLOCK_RESERVE child COW");
    assert_ne!(child_overwrite.block_id, 0);
    assert_ne!(child_overwrite.block_id, parent_reserve.block_id);
    assert_ne!(child_overwrite.gpu_vaddr, parent_reserve.gpu_vaddr);
    wait_for_state(
        fd,
        branch.child_session_id,
        child_overwrite.block_id,
        PolarisBlockState::Resident,
    );

    let parent_after = get_block_state(fd, parent.session_id, parent_reserve.block_id);
    assert_eq!(parent_after.refcount, 1);
    let child_after = get_block_state(fd, branch.child_session_id, child_overwrite.block_id);
    assert_eq!(child_after.block_id, child_overwrite.block_id);
    assert_eq!(child_after.refcount, 1);
    assert_eq!(child_after.gpu_vaddr, child_overwrite.gpu_vaddr);
    let mut child_stats = libpolaris::types::PolarisSessionGetStatsArg {
        session_id: branch.child_session_id,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_GET_STATS, &mut child_stats)
        .expect("POLARIS_SESSION_GET_STATS child");
    assert_eq!(child_stats.num_blocks, 1);

    let seen_ops = seen.lock().expect("seen mutex poisoned");
    assert_eq!(seen_ops.alloc, 1);
    assert_eq!(seen_ops.cow_break, 1);
    assert_eq!(
        seen_ops.last_cow_src_handle,
        0x1000_0000 + parent_reserve.block_id
    );
    assert_eq!(seen_ops.last_cow_dst_vaddr, child_overwrite.gpu_vaddr);
    drop(seen_ops);

    let child_destroy = PolarisSessionDestroyArg {
        session_id: branch.child_session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &child_destroy)
        .expect("POLARIS_SESSION_DESTROY child");
    let parent_destroy = PolarisSessionDestroyArg {
        session_id: parent.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &parent_destroy)
        .expect("POLARIS_SESSION_DESTROY parent");
    wait_for_seen(
        &seen,
        |ops| ops.free >= 2,
        "FREE decisions after COW test destroy",
    );

    stop.store(true, Ordering::Release);
    executor.join().expect("fake executor join");
}

#[test]
#[ignore = "requires root and a loaded polaris.ko; exercises multi-worker v4 mapping cleanup without CUDA"]
fn unregister_vaspace_reaps_only_that_workers_block_mappings() {
    let control = open_polaris();
    let control_fd = control.as_raw_fd();
    let worker_a = open_polaris();
    let worker_a_fd = worker_a.as_raw_fd();
    let worker_b = open_polaris();
    let worker_b_fd = worker_b.as_raw_fd();

    register_transient_gpu_and_range(control_fd, 0x3000_0000_0000, 8 * BLOCK_SIZE);

    let mut session = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(control_fd, ioctl::POLARIS_SESSION_CREATE, &mut session)
        .expect("POLARIS_SESSION_CREATE");

    let mut reserve = PolarisBlockReserveArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        flags: POLARIS_RESERVE_FLAG_DEFER_FAULT,
        ..Default::default()
    };
    ioctl::ioctl_read(control_fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE defer");
    assert_ne!(reserve.block_id, 0);
    assert_eq!(
        get_block_state(control_fd, session.session_id, reserve.block_id).state,
        PolarisBlockState::Unmapped as u32
    );

    let vas_a = PolarisRegisterVaSpaceArg {
        gpu_id: 0,
        rm_client_token: 0xabc0_0001,
        va_space_token: 0xdef0_0001,
        managed_base: 0x4100_0000_0000,
        managed_length: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::register_vaspace(worker_a_fd, &vas_a).expect("POLARIS_REGISTER_VASPACE A");
    let vas_b = PolarisRegisterVaSpaceArg {
        gpu_id: 0,
        rm_client_token: 0xabc0_0002,
        va_space_token: 0xdef0_0002,
        managed_base: 0x4200_0000_0000,
        managed_length: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::register_vaspace(worker_b_fd, &vas_b).expect("POLARIS_REGISTER_VASPACE B");

    for vas in [vas_a, vas_b] {
        let mapping = PolarisRegisterBlockMappingArg {
            block_id: reserve.block_id,
            gpu_id: vas.gpu_id,
            rm_client_token: vas.rm_client_token,
            va_space_token: vas.va_space_token,
            base: vas.managed_base,
            length: BLOCK_SIZE,
            ..Default::default()
        };
        ioctl::register_block_mapping(control_fd, &mapping)
            .expect("POLARIS_REGISTER_BLOCK_MAPPING");
    }

    assert_stat("v4_va_spaces", 2);
    assert_stat("v4_worker_pids", 2);
    assert_stat("block_mappings", 2);

    let unregister_a = PolarisUnregisterVaSpaceArg {
        gpu_id: vas_a.gpu_id,
        rm_client_token: vas_a.rm_client_token,
        va_space_token: vas_a.va_space_token,
        ..Default::default()
    };
    let err = ioctl::unregister_vaspace(worker_b_fd, &unregister_a)
        .expect_err("non-owner fd must not unregister another worker's VA-space");
    assert_eq!(err, libc::EPERM);
    assert_stat("v4_va_spaces", 2);
    assert_stat("v4_worker_pids", 2);
    assert_stat("block_mappings", 2);

    ioctl::unregister_vaspace(worker_a_fd, &unregister_a).expect("POLARIS_UNREGISTER_VASPACE A");
    assert_stat("v4_va_spaces", 1);
    assert_stat("v4_worker_pids", 1);
    assert_stat("block_mappings", 1);

    drop(worker_b);
    assert_stat("v4_va_spaces", 0);
    assert_stat("v4_worker_pids", 0);
    assert_stat("block_mappings", 0);

    let destroy = PolarisSessionDestroyArg {
        session_id: session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(control_fd, ioctl::POLARIS_SESSION_DESTROY, &destroy)
        .expect("POLARIS_SESSION_DESTROY");
}

#[test]
#[ignore = "requires root and freshly loaded polaris.ko; exercises v4 fast-table saturation cleanup"]
fn failed_vaspace_fast_slot_registration_does_not_leave_authoritative_entry() {
    let control = open_polaris();
    let control_fd = control.as_raw_fd();
    register_transient_gpu_and_range(control_fd, 0x3900_0000_0000, 8 * BLOCK_SIZE);

    let mut workers = Vec::new();
    for idx in 0..16u64 {
        let worker = open_polaris();
        let vas = PolarisRegisterVaSpaceArg {
            gpu_id: 0,
            rm_client_token: 0x3900_0000 + idx,
            va_space_token: 0x4900_0000 + idx,
            managed_base: 0x4a00_0000_0000 + idx * BLOCK_SIZE,
            managed_length: BLOCK_SIZE,
            ..Default::default()
        };
        ioctl::register_vaspace(worker.as_raw_fd(), &vas).expect("fill fast v4 VA-space slot");
        workers.push(worker);
    }
    assert_stat("v4_va_spaces", 16);
    assert_stat("v4_worker_pids", 16);

    let overflow_worker = open_polaris();
    let overflow = PolarisRegisterVaSpaceArg {
        gpu_id: 0,
        rm_client_token: 0x3900_1000,
        va_space_token: 0x4900_1000,
        managed_base: 0x4b00_0000_0000,
        managed_length: BLOCK_SIZE,
        ..Default::default()
    };
    let err = ioctl::register_vaspace(overflow_worker.as_raw_fd(), &overflow)
        .expect_err("17th v4 VA-space should fail when the fast hook table is full");
    assert_eq!(err, libc::ENOMEM);
    assert_stat("v4_va_spaces", 16);
    assert_stat("v4_worker_pids", 16);

    drop(overflow_worker);
    assert_stat("v4_va_spaces", 16);
    assert_stat("v4_worker_pids", 16);

    drop(workers);
    wait_for_stat("v4_va_spaces", 0, "filled v4 worker fd cleanup");
    wait_for_stat("v4_worker_pids", 0, "filled v4 worker pid cleanup");
}

#[test]
#[ignore = "requires root and freshly loaded polaris.ko; exercises worker process-exit v4 cleanup without CUDA"]
fn worker_process_exit_reaps_vaspace_and_block_mapping() {
    let control = open_polaris();
    let control_fd = control.as_raw_fd();

    register_transient_gpu_and_range(control_fd, 0x4a00_0000_0000, 4 * BLOCK_SIZE);

    let mut session = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(control_fd, ioctl::POLARIS_SESSION_CREATE, &mut session)
        .expect("POLARIS_SESSION_CREATE");

    let mut reserve = PolarisBlockReserveArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        flags: POLARIS_RESERVE_FLAG_DEFER_FAULT,
        ..Default::default()
    };
    ioctl::ioctl_read(control_fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE defer");
    assert_ne!(reserve.block_id, 0);

    let ready_path = format!(
        "/tmp/polaris-worker-exit-{}-{}.ready",
        std::process::id(),
        reserve.block_id
    );
    let _ = std::fs::remove_file(&ready_path);

    let mut child = Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg("worker_exit_child_process_entrypoint")
        .arg("--ignored")
        .arg("--nocapture")
        .env("POLARIS_WORKER_EXIT_CHILD", "1")
        .env("POLARIS_CHILD_BLOCK_ID", reserve.block_id.to_string())
        .env("POLARIS_CHILD_CLIENT", "2864705793")
        .env("POLARIS_CHILD_TOKEN", "3736076545")
        .env("POLARIS_CHILD_BASE", "81363860496384")
        .env("POLARIS_CHILD_READY_PATH", &ready_path)
        .spawn()
        .expect("spawn worker-exit child");

    let started = Instant::now();
    while !std::path::Path::new(&ready_path).exists() {
        if started.elapsed() >= Duration::from_secs(5) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("worker-exit child did not publish ready marker");
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert_stat("v4_va_spaces", 1);
    assert_stat("block_mappings", 1);

    child.kill().expect("kill worker-exit child");
    let status = child.wait().expect("wait worker-exit child");
    assert!(
        !status.success(),
        "worker-exit child should have been killed"
    );
    wait_for_stat("v4_va_spaces", 0, "child fd close");
    wait_for_stat("block_mappings", 0, "child fd close");
    let _ = std::fs::remove_file(&ready_path);

    let destroy = PolarisSessionDestroyArg {
        session_id: session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(control_fd, ioctl::POLARIS_SESSION_DESTROY, &destroy)
        .expect("POLARIS_SESSION_DESTROY");
}

#[test]
#[ignore = "requires root and freshly loaded polaris.ko; exercises logical block mapping reaping without CUDA"]
fn session_destroy_reaps_mappings_for_removed_deferred_block() {
    let control = open_polaris();
    let control_fd = control.as_raw_fd();
    let worker = open_polaris();
    let worker_fd = worker.as_raw_fd();

    register_transient_gpu_and_range(control_fd, 0x5000_0000_0000, 4 * BLOCK_SIZE);

    let mut session = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(control_fd, ioctl::POLARIS_SESSION_CREATE, &mut session)
        .expect("POLARIS_SESSION_CREATE");

    let mut reserve = PolarisBlockReserveArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        flags: POLARIS_RESERVE_FLAG_DEFER_FAULT,
        ..Default::default()
    };
    ioctl::ioctl_read(control_fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE defer");
    assert_ne!(reserve.block_id, 0);

    let vas = PolarisRegisterVaSpaceArg {
        gpu_id: 0,
        rm_client_token: 0xabc0_0101,
        va_space_token: 0xdef0_0101,
        managed_base: 0x5100_0000_0000,
        managed_length: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::register_vaspace(worker_fd, &vas).expect("POLARIS_REGISTER_VASPACE");
    let mapping = PolarisRegisterBlockMappingArg {
        block_id: reserve.block_id,
        gpu_id: vas.gpu_id,
        rm_client_token: vas.rm_client_token,
        va_space_token: vas.va_space_token,
        base: vas.managed_base,
        length: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::register_block_mapping(control_fd, &mapping).expect("POLARIS_REGISTER_BLOCK_MAPPING");
    assert_stat("block_mappings", 1);

    let destroy = PolarisSessionDestroyArg {
        session_id: session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(control_fd, ioctl::POLARIS_SESSION_DESTROY, &destroy)
        .expect("POLARIS_SESSION_DESTROY");
    assert_stat("blocks", 0);
    assert_stat("block_mappings", 0);
}

#[test]
#[ignore = "requires root and freshly loaded polaris.ko; exercises BLOCK_RELEASE mapping reaping without CUDA"]
fn block_release_reaps_mappings_for_removed_deferred_block() {
    let control = open_polaris();
    let control_fd = control.as_raw_fd();
    let worker = open_polaris();
    let worker_fd = worker.as_raw_fd();

    register_transient_gpu_and_range(control_fd, 0x6000_0000_0000, 4 * BLOCK_SIZE);

    let mut session = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(control_fd, ioctl::POLARIS_SESSION_CREATE, &mut session)
        .expect("POLARIS_SESSION_CREATE");

    let mut reserve = PolarisBlockReserveArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        flags: POLARIS_RESERVE_FLAG_DEFER_FAULT,
        ..Default::default()
    };
    ioctl::ioctl_read(control_fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE defer");
    assert_ne!(reserve.block_id, 0);

    let vas = PolarisRegisterVaSpaceArg {
        gpu_id: 0,
        rm_client_token: 0xabc0_0201,
        va_space_token: 0xdef0_0201,
        managed_base: 0x6100_0000_0000,
        managed_length: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::register_vaspace(worker_fd, &vas).expect("POLARIS_REGISTER_VASPACE");
    let mapping = PolarisRegisterBlockMappingArg {
        block_id: reserve.block_id,
        gpu_id: vas.gpu_id,
        rm_client_token: vas.rm_client_token,
        va_space_token: vas.va_space_token,
        base: vas.managed_base,
        length: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::register_block_mapping(control_fd, &mapping).expect("POLARIS_REGISTER_BLOCK_MAPPING");
    assert_stat("block_mappings", 1);

    let release = PolarisBlockReleaseArg {
        session_id: session.session_id,
        token_start: 0,
        token_count: 1,
        ..Default::default()
    };
    ioctl::ioctl_write(control_fd, ioctl::POLARIS_BLOCK_RELEASE, &release)
        .expect("POLARIS_BLOCK_RELEASE");
    assert_stat("blocks", 0);
    assert_stat("block_mappings", 0);

    let destroy = PolarisSessionDestroyArg {
        session_id: session.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(control_fd, ioctl::POLARIS_SESSION_DESTROY, &destroy)
        .expect("POLARIS_SESSION_DESTROY");
}

#[test]
#[ignore = "requires root and freshly loaded polaris.ko; exercises COW-shared BLOCK_RELEASE without CUDA"]
fn child_block_release_decrements_inherited_shared_block() {
    let control = open_polaris();
    let fd = control.as_raw_fd();

    register_transient_gpu_and_range(fd, 0x7000_0000_0000, 4 * BLOCK_SIZE);

    let mut parent = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut parent)
        .expect("POLARIS_SESSION_CREATE parent");

    let mut reserve = PolarisBlockReserveArg {
        session_id: parent.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        flags: POLARIS_RESERVE_FLAG_DEFER_FAULT,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE defer");

    let mut branch = PolarisSessionBranchArg {
        parent_session_id: parent.session_id,
        ..Default::default()
    };
    ioctl::session_branch(fd, &mut branch).expect("POLARIS_SESSION_BRANCH");
    assert_eq!(
        get_block_state(fd, parent.session_id, reserve.block_id).refcount,
        2
    );

    let release = PolarisBlockReleaseArg {
        session_id: branch.child_session_id,
        token_start: 0,
        token_count: 1,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_BLOCK_RELEASE, &release)
        .expect("POLARIS_BLOCK_RELEASE child shared");

    let parent_state = get_block_state(fd, parent.session_id, reserve.block_id);
    assert_eq!(parent_state.block_id, reserve.block_id);
    assert_eq!(parent_state.refcount, 1);
    assert_eq!(parent_state.state, PolarisBlockState::Unmapped as u32);

    let child_stats = {
        let mut stats = libpolaris::types::PolarisSessionGetStatsArg {
            session_id: branch.child_session_id,
            ..Default::default()
        };
        ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_GET_STATS, &mut stats)
            .expect("POLARIS_SESSION_GET_STATS child");
        stats
    };
    assert_eq!(child_stats.num_blocks, 0);
    assert_stat("blocks", 1);

    let child_destroy = PolarisSessionDestroyArg {
        session_id: branch.child_session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &child_destroy)
        .expect("POLARIS_SESSION_DESTROY child");
    let parent_destroy = PolarisSessionDestroyArg {
        session_id: parent.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &parent_destroy)
        .expect("POLARIS_SESSION_DESTROY parent");
}

#[test]
#[ignore = "requires root and freshly loaded polaris.ko; exercises COW-shared BLOCK_RELEASE mapping cleanup without CUDA"]
fn child_block_release_reaps_only_child_mapping_for_shared_block() {
    let control = open_polaris();
    let fd = control.as_raw_fd();
    let parent_worker = open_polaris();
    let parent_worker_fd = parent_worker.as_raw_fd();
    let child_worker = open_polaris();
    let child_worker_fd = child_worker.as_raw_fd();

    register_transient_gpu_and_range(fd, 0x8000_0000_0000, 8 * BLOCK_SIZE);

    let mut parent = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut parent)
        .expect("POLARIS_SESSION_CREATE parent");

    let mut reserve = PolarisBlockReserveArg {
        session_id: parent.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        flags: POLARIS_RESERVE_FLAG_DEFER_FAULT,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE defer");

    let mut branch = PolarisSessionBranchArg {
        parent_session_id: parent.session_id,
        ..Default::default()
    };
    ioctl::session_branch(fd, &mut branch).expect("POLARIS_SESSION_BRANCH");

    let parent_vas = PolarisRegisterVaSpaceArg {
        gpu_id: 0,
        rm_client_token: 0xabc0_0301,
        va_space_token: 0xdef0_0301,
        managed_base: reserve.gpu_vaddr,
        managed_length: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::register_vaspace(parent_worker_fd, &parent_vas)
        .expect("POLARIS_REGISTER_VASPACE parent");
    let child_base = reserve.gpu_vaddr + BLOCK_SIZE;
    let child_vas = PolarisRegisterVaSpaceArg {
        gpu_id: 0,
        rm_client_token: 0xabc0_0302,
        va_space_token: 0xdef0_0302,
        managed_base: child_base,
        managed_length: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::register_vaspace(child_worker_fd, &child_vas).expect("POLARIS_REGISTER_VASPACE child");

    for vas in [parent_vas, child_vas] {
        let mapping = PolarisRegisterBlockMappingArg {
            block_id: reserve.block_id,
            gpu_id: vas.gpu_id,
            rm_client_token: vas.rm_client_token,
            va_space_token: vas.va_space_token,
            base: vas.managed_base,
            length: BLOCK_SIZE,
            ..Default::default()
        };
        ioctl::register_block_mapping(fd, &mapping).expect("POLARIS_REGISTER_BLOCK_MAPPING");
    }
    assert_stat("block_mappings", 2);

    let release = PolarisBlockReleaseArg {
        session_id: branch.child_session_id,
        token_start: 0,
        token_count: 1,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_BLOCK_RELEASE, &release)
        .expect("POLARIS_BLOCK_RELEASE child shared");
    assert_eq!(
        get_block_state(fd, parent.session_id, reserve.block_id).refcount,
        1
    );
    assert_stat("block_mappings", 1);

    let unregister_parent = PolarisUnregisterVaSpaceArg {
        gpu_id: parent_vas.gpu_id,
        rm_client_token: parent_vas.rm_client_token,
        va_space_token: parent_vas.va_space_token,
        ..Default::default()
    };
    ioctl::unregister_vaspace(parent_worker_fd, &unregister_parent)
        .expect("POLARIS_UNREGISTER_VASPACE parent");
    assert_stat("block_mappings", 0);

    let child_destroy = PolarisSessionDestroyArg {
        session_id: branch.child_session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &child_destroy)
        .expect("POLARIS_SESSION_DESTROY child");
    let parent_destroy = PolarisSessionDestroyArg {
        session_id: parent.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &parent_destroy)
        .expect("POLARIS_SESSION_DESTROY parent");
}

#[test]
#[ignore = "requires root and freshly loaded polaris.ko; exercises COW-shared SESSION_DESTROY mapping cleanup without CUDA"]
fn child_session_destroy_reaps_only_child_mapping_for_shared_block() {
    let control = open_polaris();
    let fd = control.as_raw_fd();
    let parent_worker = open_polaris();
    let parent_worker_fd = parent_worker.as_raw_fd();
    let child_worker = open_polaris();
    let child_worker_fd = child_worker.as_raw_fd();

    register_transient_gpu_and_range(fd, 0x9000_0000_0000, 8 * BLOCK_SIZE);

    let mut parent = PolarisSessionCreateArg {
        home_gpu: 0,
        gpu_vas_bytes: BLOCK_SIZE,
        bytes_per_token: BLOCK_SIZE,
        priority: 5,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_SESSION_CREATE, &mut parent)
        .expect("POLARIS_SESSION_CREATE parent");

    let mut reserve = PolarisBlockReserveArg {
        session_id: parent.session_id,
        token_start: 0,
        token_count: 1,
        phase: PolarisPhase::Prefill as u32,
        flags: POLARIS_RESERVE_FLAG_DEFER_FAULT,
        ..Default::default()
    };
    ioctl::ioctl_read(fd, ioctl::POLARIS_BLOCK_RESERVE, &mut reserve)
        .expect("POLARIS_BLOCK_RESERVE defer");

    let mut branch = PolarisSessionBranchArg {
        parent_session_id: parent.session_id,
        ..Default::default()
    };
    ioctl::session_branch(fd, &mut branch).expect("POLARIS_SESSION_BRANCH");

    let parent_vas = PolarisRegisterVaSpaceArg {
        gpu_id: 0,
        rm_client_token: 0xabc0_0401,
        va_space_token: 0xdef0_0401,
        managed_base: reserve.gpu_vaddr,
        managed_length: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::register_vaspace(parent_worker_fd, &parent_vas)
        .expect("POLARIS_REGISTER_VASPACE parent");
    let child_base = reserve.gpu_vaddr + BLOCK_SIZE;
    let child_vas = PolarisRegisterVaSpaceArg {
        gpu_id: 0,
        rm_client_token: 0xabc0_0402,
        va_space_token: 0xdef0_0402,
        managed_base: child_base,
        managed_length: BLOCK_SIZE,
        ..Default::default()
    };
    ioctl::register_vaspace(child_worker_fd, &child_vas).expect("POLARIS_REGISTER_VASPACE child");

    for vas in [parent_vas, child_vas] {
        let mapping = PolarisRegisterBlockMappingArg {
            block_id: reserve.block_id,
            gpu_id: vas.gpu_id,
            rm_client_token: vas.rm_client_token,
            va_space_token: vas.va_space_token,
            base: vas.managed_base,
            length: BLOCK_SIZE,
            ..Default::default()
        };
        ioctl::register_block_mapping(fd, &mapping).expect("POLARIS_REGISTER_BLOCK_MAPPING");
    }
    assert_stat("block_mappings", 2);

    let child_destroy = PolarisSessionDestroyArg {
        session_id: branch.child_session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &child_destroy)
        .expect("POLARIS_SESSION_DESTROY child");
    assert_eq!(
        get_block_state(fd, parent.session_id, reserve.block_id).refcount,
        1
    );
    assert_stat("block_mappings", 1);

    let unregister_parent = PolarisUnregisterVaSpaceArg {
        gpu_id: parent_vas.gpu_id,
        rm_client_token: parent_vas.rm_client_token,
        va_space_token: parent_vas.va_space_token,
        ..Default::default()
    };
    ioctl::unregister_vaspace(parent_worker_fd, &unregister_parent)
        .expect("POLARIS_UNREGISTER_VASPACE parent");
    assert_stat("block_mappings", 0);

    let parent_destroy = PolarisSessionDestroyArg {
        session_id: parent.session_id,
        ..Default::default()
    };
    ioctl::ioctl_write(fd, ioctl::POLARIS_SESSION_DESTROY, &parent_destroy)
        .expect("POLARIS_SESSION_DESTROY parent");
}
