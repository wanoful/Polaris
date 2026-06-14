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
    uint64_t vaspace_base;
    uint64_t vaspace_size;
};

int polaris_shim_bootstrap_rm_uvm(int cuda_ordinal,
                                  uint32_t gpu_id,
                                  struct polaris_shim_bootstrap *out);
void polaris_shim_bootstrap_cleanup(struct polaris_shim_bootstrap *state);

#endif /* POLARIS_SHIM_RM_UVM_BOOTSTRAP_H */
