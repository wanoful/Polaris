/* SPDX-License-Identifier: GPL-2.0
 *
 * Minimal UVM external-range helper for shim-managed allocations.
 *
 * The fd passed to polaris_shim_uvm_adopt_fd() must be the initialized
 * /dev/nvidia-uvm VA-space fd that owns the RM VA-space registered with
 * UVM_REGISTER_GPU_VASPACE. Creating an external range on any other UVM fd
 * creates it in the wrong VA-space and the Polaris bridge cannot find it.
 */

#ifndef POLARIS_SHIM_UVM_EXTERNAL_H
#define POLARIS_SHIM_UVM_EXTERNAL_H

#include <stdint.h>

int polaris_shim_uvm_adopt_fd(int uvm_fd);
int polaris_shim_uvm_create_external_range(uint64_t base, uint64_t length);
int polaris_shim_uvm_free_external_range(uint64_t base);

#endif /* POLARIS_SHIM_UVM_EXTERNAL_H */
