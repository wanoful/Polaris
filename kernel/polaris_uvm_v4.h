/* SPDX-License-Identifier: GPL-2.0 */
#ifndef POLARIS_UVM_V4_H
#define POLARIS_UVM_V4_H

#include <linux/types.h>

/*
 * v4 ABI mirror of UVM's uvm_polaris.h (kernel-open/nvidia-uvm/uvm_polaris.h
 * on the polaris-v4 branch of open-gpu-kernel-modules). Kept in sync by hand:
 * the UVM header is the source of truth, this header is what polaris.ko
 * compiles against once the M2 rewrite migrates the fault path off the
 * legacy single-function export declared in polaris_uvm.h.
 *
 * polaris_uvm.h (v3) and this header coexist during M1/M2: v3 keeps the live
 * single-function hook polaris.ko already exports for diagnostics; v4 is what
 * the new ops-vector registration path will use once the fault handler is
 * extended to thread va_space_token through the block→worker map.
 */

enum polaris_uvm_v4_fault_result {
	POLARIS_UVM_V4_FAULT_NOT_MINE = 0,
	POLARIS_UVM_V4_FAULT_HANDLED  = 1,
	POLARIS_UVM_V4_FAULT_ERROR    = -1,
};

/*
 * Ops vector polaris.ko publishes to UVM via uvm_polaris_register_hook().
 *
 * handle_gpu_fault is invoked from UVM's replayable-fault bottom half once
 * per fault entry, after UVM has resolved the per-fault GPU VA-space and
 * before any managed-VA or ATS lookup runs.
 *
 * Parameters:
 *   gpu_id         - uvm_parent_id_value(parent_gpu->id) for the faulting GPU
 *   va_space_token - the duped RM GPU VA-space handle; stable per
 *                    (worker, gpu) and used as the key into polaris.ko's
 *                    worker map
 *   fault_address  - the GPU virtual address that faulted
 *   access_type    - one of uvm_fault_access_type_t (PREFETCH/READ/WRITE/
 *                    ATOMIC_WEAK/ATOMIC_STRONG)
 *
 * The callback runs under rcu_read_lock() inside UVM. It must not sleep on
 * userspace IPC. Cold faults and COW splits that need polarisd live behind
 * polaris.ko's own slow-path queue and reach this entry point only after the
 * kernel decision has been made.
 */
struct uvm_polaris_ops {
	int (*handle_gpu_fault)(u32 gpu_id,
				u64 va_space_token,
				u64 fault_address,
				u32 access_type);
};

/*
 * Exported by UVM (EXPORT_SYMBOL_GPL). polaris.ko owns the ops storage and
 * must keep it alive until uvm_polaris_unregister_hook() returns.
 *
 * Returns 0 on success, -EBUSY if a hook is already installed, -EINVAL if
 * ops or ops->handle_gpu_fault is NULL.
 */
int uvm_polaris_register_hook(const struct uvm_polaris_ops *ops);
void uvm_polaris_unregister_hook(const struct uvm_polaris_ops *ops);

#endif /* POLARIS_UVM_V4_H */
