/* SPDX-License-Identifier: GPL-2.0 */
#ifndef POLARIS_UVM_V4_H
#define POLARIS_UVM_V4_H

#include <linux/types.h>

/*
 * v4 ABI mirror of UVM's uvm_polaris.h (kernel-open/nvidia-uvm/uvm_polaris.h
 * on the polaris-v4 branch of open-gpu-kernel-modules). Kept in sync by hand:
 * the UVM header is the source of truth.
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
 *   gpu_id           - uvm_parent_id_value(parent_gpu->id) for the faulting GPU
 *   rm_client_token  - the user RM client handle passed to
 *                      UvmRegisterGpuVaSpace
 *   va_space_token   - the user RM GPU VA-space handle passed to
 *                      UvmRegisterGpuVaSpace; stable with rm_client_token
 *   gpu_va_space_ptr - opaque uvm_gpu_va_space_t pointer to pass back to
 *                      uvm_polaris_map_external_allocation()
 *   fault_address    - the GPU virtual address that faulted
 *   access_type      - one of uvm_fault_access_type_t (PREFETCH/READ/WRITE/
 *                      ATOMIC_WEAK/ATOMIC_STRONG)
 *
 * The callback runs under rcu_read_lock() inside UVM. It must not sleep on
 * userspace IPC. Cold faults and COW splits that need polarisd live behind
 * polaris.ko's own slow-path queue and reach this entry point only after the
 * kernel decision has been made.
 */
struct uvm_polaris_ops {
	struct module *owner;
	int (*handle_gpu_fault)(u32 gpu_id,
				u64 rm_client_token,
				u64 va_space_token,
				u64 gpu_va_space_ptr,
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

/*
 * Exported by UVM (EXPORT_SYMBOL_GPL). gpu_va_space_ptr must be an opaque
 * uvm_gpu_va_space_t pointer previously passed to handle_gpu_fault. The VA
 * range must already be registered with UVM as an external range.
 */
int uvm_polaris_map_external_allocation(u64 gpu_va_space_ptr,
					u64 base,
					u64 length,
					u64 offset,
					s32 rm_control_fd,
					u32 h_client,
					u32 h_memory);

/*
 * Exported by UVM (EXPORT_SYMBOL_GPL). gpu_va_space_ptr must be an opaque
 * uvm_gpu_va_space_t pointer previously passed to handle_gpu_fault. The VA
 * range must already be registered with UVM as an external range. This tears
 * down mappings for [base, base + length) on the corresponding GPU VA-space.
 */
int uvm_polaris_unmap_external_allocation(u64 gpu_va_space_ptr,
					  u64 base,
					  u64 length);

#endif /* POLARIS_UVM_V4_H */
