/* SPDX-License-Identifier: GPL-2.0
 *
 * Tiny wrapper around /dev/polaris. The shim opens it once on first cuInit
 * (after libcuda is up) and reuses the fd for subsequent ioctls. Failures
 * to open are non-fatal: the worker keeps running without POLARIS managed
 * paging, just like running without the LD_PRELOAD. We print one diagnostic
 * line and disable further attempts.
 */

#ifndef POLARIS_SHIM_DEVICE_H
#define POLARIS_SHIM_DEVICE_H

#include <stdint.h>

/*
 * Returns the /dev/polaris fd, or -1 if the device is unavailable (driver
 * not loaded, permissions wrong, etc). The fd is cached process-wide;
 * callers must NOT close it.
 */
int polaris_shim_device_fd(void);

/*
 * Submit POLARIS_REGISTER_VASPACE. Returns 0 on success, -errno on failure.
 * Safe to call when /dev/polaris is unavailable: returns -ENODEV and the
 * caller is expected to keep the worker running unmanaged.
 */
int polaris_shim_register_vaspace(uint32_t gpu_id,
                                  uint64_t va_space_token,
                                  uint64_t managed_base,
                                  uint64_t managed_length);

#endif /* POLARIS_SHIM_DEVICE_H */
