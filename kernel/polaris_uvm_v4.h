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
	POLARIS_UVM_V4_FAULT_RETRY    = 2,
	POLARIS_UVM_V4_FAULT_DEFERRED = 3,
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

/*
 * Exported by UVM (EXPORT_SYMBOL_GPL). Diagnostic helper for the RM-backed
 * copy path: duplicate an RM allocation and query GPU-visible physical
 * addresses through RM.
 */
int uvm_polaris_probe_external_allocation(u64 gpu_va_space_ptr,
					  u64 offset,
					  u64 length,
					  s32 rm_control_fd,
					  u32 h_client,
					  u32 h_memory,
					  u64 *page_size_out,
					  u64 *phys_addr_count_out,
					  u64 *first_phys_addr_out,
					  u64 *last_phys_addr_out,
					  u64 *contiguous_out,
					  u64 *sysmem_out,
					  u64 *egm_out,
					  u64 *fabricmem_out);

/*
 * Exported by UVM (EXPORT_SYMBOL_GPL). Diagnostic helper for proving a narrow
 * RM-backed byte-copy path: write a deterministic pattern into the RM external
 * allocation through CE, read it back through CE, and report the first mismatch.
 */
int uvm_polaris_probe_external_copy(u64 gpu_va_space_ptr,
				    u64 offset,
				    u64 length,
				    s32 rm_control_fd,
				    u32 h_client,
				    u32 h_memory,
				    u64 pattern_seed,
				    u64 *page_size_out,
				    u64 *phys_addr_count_out,
				    u64 *first_phys_addr_out,
				    u64 *last_phys_addr_out,
				    u64 *flags_out,
				    u64 *bytes_checked_out,
				    u64 *first_mismatch_offset_out,
				    u64 *expected_byte_out,
				    u64 *actual_byte_out);

#define UVM_POLARIS_RM_COPY_TO_CPU   0
#define UVM_POLARIS_RM_COPY_FROM_CPU 1

/*
 * Exported by UVM (EXPORT_SYMBOL_GPL). Production-shaped helper for copying
 * between a Polaris RM-backed allocation and a userspace CPU buffer through
 * UVM-owned DMA staging memory plus CE. Currently supports contiguous vidmem.
 */
int uvm_polaris_copy_external_allocation(u64 gpu_va_space_ptr,
					 u64 offset,
					 u64 length,
					 s32 rm_control_fd,
					 u32 h_client,
					 u32 h_memory,
					 u64 user_cpu_addr,
					 u32 direction,
					 u64 *page_size_out,
					 u64 *phys_addr_count_out,
					 u64 *first_phys_addr_out,
					 u64 *last_phys_addr_out,
					 u64 *flags_out,
					 u64 *bytes_copied_out);

#endif /* POLARIS_UVM_V4_H */
