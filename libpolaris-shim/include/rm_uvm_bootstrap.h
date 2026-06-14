/* SPDX-License-Identifier: GPL-2.0
 *
 * Opt-in RM/UVM bootstrap for the v4 shim.
 */

#ifndef POLARIS_SHIM_RM_UVM_BOOTSTRAP_H
#define POLARIS_SHIM_RM_UVM_BOOTSTRAP_H

#include <stdint.h>

struct polaris_shim_bootstrap {
    int rm_control_fd;
    int gpu_fd;
    int uvm_fd;
    int uvm_mm_fd;
    uint32_t gpu_id;
    uint32_t h_client;
    uint32_t h_device;
    uint32_t h_subdevice;
    uint32_t h_vaspace;
    uint32_t observed_gpu_id;
    uint64_t observed_rm_client_token;
    uint64_t observed_va_space_token;
    uint64_t vaspace_base;
    uint64_t vaspace_size;
};

struct polaris_shim_rm_allocation {
    uint32_t h_memory;
    uint64_t size;
};

int polaris_shim_bootstrap_rm_uvm(int cuda_ordinal,
                                  uint32_t gpu_id,
                                  struct polaris_shim_bootstrap *out);
void polaris_shim_bootstrap_cleanup(struct polaris_shim_bootstrap *state);

int polaris_shim_rm_alloc_device_memory(const struct polaris_shim_bootstrap *state,
                                        uint64_t size,
                                        struct polaris_shim_rm_allocation *out);
void polaris_shim_rm_free_device_memory(const struct polaris_shim_bootstrap *state,
                                        struct polaris_shim_rm_allocation *allocation);

#endif /* POLARIS_SHIM_RM_UVM_BOOTSTRAP_H */
