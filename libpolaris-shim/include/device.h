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

int polaris_shim_register_gpu(uint32_t gpu_id,
                              uint64_t total_bytes,
                              uint64_t budget_bytes,
                              uint64_t cpu_pool_bytes,
                              uint32_t flags);

int polaris_shim_register_va_range(uint32_t gpu_id,
                                   uint64_t base,
                                   uint64_t length,
                                   uint64_t block_size,
                                   uint64_t *range_id_out);

int polaris_shim_session_create(uint32_t gpu_id,
                                uint64_t gpu_vas_bytes,
                                uint64_t bytes_per_token,
                                uint64_t *session_id_out);

int polaris_shim_session_destroy(uint64_t session_id);

int polaris_shim_block_reserve(uint64_t session_id,
                               uint32_t token_start,
                               uint32_t token_count,
                               uint32_t flags,
                               uint64_t *block_id_out,
                               uint64_t *gpu_vaddr_out);

int polaris_shim_block_release(uint64_t session_id,
                               uint32_t token_start,
                               uint32_t token_count);

int polaris_shim_block_release_with_flags(uint64_t session_id,
                                          uint32_t token_start,
                                          uint32_t token_count,
                                          uint32_t flags);

int polaris_shim_update_kv_active_window(uint64_t session_id,
                                         uint32_t phase,
                                         uint32_t flags,
                                         uint64_t read_start_token,
                                         uint64_t read_token_count,
                                         uint64_t write_start_token,
                                         uint64_t write_token_count,
                                         uint64_t epoch);

/*
 * Publish/clear a KV active-window hint for the shim-owned Polaris session.
 * These are exported by libpolaris-shim.so so framework adapters can report
 * precise prefill/decode read/write ranges without knowing the session id.
 */
int polaris_shim_set_kv_active_window(uint32_t phase,
                                      uint64_t read_start_token,
                                      uint64_t read_token_count,
                                      uint64_t write_start_token,
                                      uint64_t write_token_count);

int polaris_shim_set_kv_active_byte_ranges(uint32_t phase,
                                           const void *read0,
                                           uint64_t read0_bytes,
                                           const void *read1,
                                           uint64_t read1_bytes,
                                           const void *write0,
                                           uint64_t write0_bytes,
                                           const void *write1,
                                           uint64_t write1_bytes);

int polaris_shim_clear_kv_active_window(void);

/*
 * Submit POLARIS_REGISTER_VASPACE. Returns 0 on success, -errno on failure.
 * Safe to call when /dev/polaris is unavailable: returns -ENODEV and the
 * caller is expected to keep the worker running unmanaged.
 */
int polaris_shim_register_vaspace(uint32_t gpu_id,
                                  uint64_t rm_client_token,
                                  uint64_t va_space_token,
                                  uint64_t managed_base,
                                  uint64_t managed_length);

/*
 * Submit POLARIS_UNREGISTER_VASPACE for a VA-space previously registered by
 * this process. Returns 0 on success, -errno on failure.
 */
int polaris_shim_unregister_vaspace(uint32_t gpu_id,
                                    uint64_t rm_client_token,
                                    uint64_t va_space_token);

/*
 * Submit POLARIS_REGISTER_STATIC_BLOCK for the M2 microbenchmark path. The
 * RM allocation must cover [offset, offset + length), mapped at [base,
 * base + length) inside a VA-space registered by polaris_shim_register_vaspace.
 */
int polaris_shim_register_static_block(uint32_t gpu_id,
                                       uint64_t rm_client_token,
                                       uint64_t va_space_token,
                                       uint64_t base,
                                       uint64_t length,
                                       uint64_t offset,
                                       int32_t rm_control_fd,
                                       uint32_t h_client,
                                       uint32_t h_memory);

/*
 * Submit POLARIS_UNMAP_STATIC_BLOCK for the M3 spill/refault diagnostic. The
 * block must have been registered and fault-mapped at least once in this
 * process, so polaris.ko has observed the UVM GPU VA-space pointer.
 */
int polaris_shim_unmap_static_block(uint32_t gpu_id,
                                    uint64_t rm_client_token,
                                    uint64_t va_space_token,
                                    uint64_t base,
                                    uint64_t length);

int polaris_shim_register_block_mapping(uint64_t block_id,
                                        uint32_t gpu_id,
                                        uint64_t rm_client_token,
                                        uint64_t va_space_token,
                                        uint64_t base,
                                        uint64_t length);

int polaris_shim_register_block_backing(uint64_t block_id,
                                        uint32_t gpu_id,
                                        int32_t rm_control_fd,
                                        uint32_t h_client,
                                        uint32_t h_memory,
                                        uint64_t length,
                                        uint64_t offset);

int polaris_shim_unmap_block_mappings(uint64_t block_id,
                                      uint32_t *unmapped_count_out);

#endif /* POLARIS_SHIM_DEVICE_H */
