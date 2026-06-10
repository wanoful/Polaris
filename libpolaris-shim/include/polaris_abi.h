/* SPDX-License-Identifier: GPL-2.0 */
/*
 * libpolaris-shim ABI mirror for the v4 VA-space registration ioctls.
 *
 * Kept in sync by hand with kernel/polaris_abi.rs and kernel/polaris_types.rs.
 * If those drift, update this header at the same time — both sides must agree
 * on struct layout and the IOC encoding.
 *
 * Reference layout (Linux generic _IOC):
 *   bits 30..31 = direction (1=write)
 *   bits 16..29 = size
 *   bits  8..15 = magic ('P' = 0x50)
 *   bits  0..7  = nr
 */

#ifndef POLARIS_SHIM_POLARIS_ABI_H
#define POLARIS_SHIM_POLARIS_ABI_H

#include <stdint.h>
#include <sys/ioctl.h>

#define POLARIS_DEVICE_PATH "/dev/polaris"
#define POLARIS_IOCTL_MAGIC 'P' /* 0x50 */

struct polaris_register_va_space_arg {
    uint32_t gpu_id;
    uint32_t _reserved0;
    uint64_t va_space_token;
    uint64_t managed_base;
    uint64_t managed_length;
    uint64_t _reserved1[4];
};

struct polaris_unregister_va_space_arg {
    uint32_t gpu_id;
    uint32_t _reserved0;
    uint64_t va_space_token;
    uint64_t _reserved1[2];
};

#define POLARIS_REGISTER_VASPACE   _IOW(POLARIS_IOCTL_MAGIC, 0x10, struct polaris_register_va_space_arg)
#define POLARIS_UNREGISTER_VASPACE _IOW(POLARIS_IOCTL_MAGIC, 0x11, struct polaris_unregister_va_space_arg)

#endif /* POLARIS_SHIM_POLARIS_ABI_H */
