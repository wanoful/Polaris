//! Async offload worker pool.
//!
//! Under KV pressure the daemon's decision loop spends ~90% of its time in the
//! blocking `rm_copy` ioctl (~820 µs per 2 MiB block). Reloads are on the GPU
//! fault-critical path and must stay synchronous, but offloads (evictions) are
//! scheduled *after* a reload completes to free space for future faults — they
//! do not block a specific fault. This pool runs the offload `rm_copy` ioctl on
//! worker threads so it overlaps with GPU compute and subsequent reloads instead
//! of serializing in the main loop.
//!
//! Correctness model: workers touch **only** the `fd` (via the `rm_copy` ioctl,
//! which the kernel runs without holding its state lock, so concurrent calls are
//! safe). All `GpuState` / `CpuPool` / `RmBackend` mutation stays in the main
//! thread — the pre-copy slot allocation and the post-copy finalize both run
//! there. The worker payload is a `Copy` snapshot with no shared state.

use crate::offload::RmOffloadJob;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

/// Result reported by a worker back to the main thread.
pub struct OffloadDone {
    pub job: RmOffloadJob,
    pub copy_result: i32,
    pub bytes_copied: u64,
}

enum WorkerMsg {
    Job(RmOffloadJob),
    Shutdown,
}

pub struct AsyncOffloadPool {
    tx: Sender<WorkerMsg>,
    done_rx: Receiver<OffloadDone>,
    workers: Vec<JoinHandle<()>>,
    /// Offloads dispatched but not yet finalized. Bounds in-flight work so the
    /// CPU pool can't be drained by an unbounded backlog.
    outstanding: usize,
    max_outstanding: usize,
}

impl AsyncOffloadPool {
    /// Spawn `num_workers` worker threads. `fd` is the /dev/polaris descriptor;
    /// it is shared read-only by the workers, which only issue `rm_copy` on it.
    pub fn new(fd: i32, num_workers: usize) -> Self {
        let num_workers = num_workers.max(1);
        // A single shared job queue drained by all workers. std::mpsc is
        // single-consumer, so guard the receiver with a mutex for work-stealing.
        let (tx, rx) = std::sync::mpsc::channel::<WorkerMsg>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<OffloadDone>();
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));

        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let rx = rx.clone();
            let done_tx = done_tx.clone();
            workers.push(std::thread::spawn(move || loop {
                // Take one message under the lock, then release it before the
                // long-running copy so other workers can pull the next job.
                let msg = {
                    let guard = rx.lock().unwrap();
                    guard.recv()
                };
                let job = match msg {
                    Ok(WorkerMsg::Job(job)) => job,
                    Ok(WorkerMsg::Shutdown) | Err(_) => break,
                };
                let (copy_result, bytes_copied) = crate::offload::copy_rm_offload(fd, &job);
                if done_tx
                    .send(OffloadDone {
                        job,
                        copy_result,
                        bytes_copied,
                    })
                    .is_err()
                {
                    break;
                }
            }));
        }

        Self {
            tx,
            done_rx,
            workers,
            outstanding: 0,
            max_outstanding: num_workers * 2,
        }
    }

    /// True when the pool is at its in-flight cap and the caller should instead
    /// run the offload synchronously (or drain completions first).
    pub fn is_full(&self) -> bool {
        self.outstanding >= self.max_outstanding
    }

    pub fn outstanding(&self) -> usize {
        self.outstanding
    }

    /// Hand an offload copy to the worker pool. The job's CPU pool slot must
    /// already be allocated by the caller (main thread).
    pub fn submit(&mut self, job: RmOffloadJob) {
        // Send can only fail if all workers are gone, which only happens at
        // shutdown; drop the job in that case (finalize won't run, but the
        // process is exiting).
        if self.tx.send(WorkerMsg::Job(job)).is_ok() {
            self.outstanding += 1;
        }
    }

    /// Non-blocking drain of completed offloads. Each returned item still needs
    /// `finalize_rm_offload` + COMPLETE_OPERATION in the main thread.
    pub fn drain_completed(&mut self) -> Vec<OffloadDone> {
        let mut out = Vec::new();
        while let Ok(done) = self.done_rx.try_recv() {
            self.outstanding -= 1;
            out.push(done);
        }
        out
    }

    /// Block until at least one offload completes, returning all currently
    /// available completions. Used to apply back-pressure when the pool is full.
    pub fn wait_for_completion(&mut self) -> Vec<OffloadDone> {
        let mut out = Vec::new();
        if let Ok(done) = self.done_rx.recv() {
            self.outstanding -= 1;
            out.push(done);
        }
        out.extend(self.drain_completed());
        out
    }

    /// Drain any remaining completions and join all workers. Callers must
    /// finalize the returned completions before dropping shared state.
    pub fn shutdown(mut self) -> Vec<OffloadDone> {
        // Signal every worker to exit, then collect whatever is still pending.
        for _ in 0..self.workers.len() {
            let _ = self.tx.send(WorkerMsg::Shutdown);
        }
        let mut remaining = Vec::new();
        while self.outstanding > 0 {
            if let Ok(done) = self.done_rx.recv() {
                self.outstanding -= 1;
                remaining.push(done);
            } else {
                break;
            }
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
        remaining
    }
}
