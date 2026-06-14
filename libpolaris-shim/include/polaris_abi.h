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
#define POLARIS_MAX_SESSIONS_PER_LIST 64

#define POLARIS_RESERVE_FLAG_OVERWRITE (1U << 0)
#define POLARIS_RESERVE_FLAG_DEFER_FAULT (1U << 4)
#define POLARIS_PHASE_PREFILL 1U

struct polaris_register_gpu_arg {
    uint32_t gpu_id;
    uint64_t total_bytes;
    uint64_t budget_bytes;
    uint64_t cpu_pool_bytes;
    uint32_t numa_node;
    uint32_t _reserved;
    uint64_t _reserved2[2];
};

struct polaris_register_va_range_arg {
    uint64_t range_id;
    uint32_t gpu_id;
    uint32_t flags;
    uint64_t base;
    uint64_t length;
    uint64_t block_size;
    uint64_t _reserved[4];
};

struct polaris_register_va_space_arg {
    uint32_t gpu_id;
    uint32_t _reserved0;
    uint64_t rm_client_token;
    uint64_t va_space_token;
    uint64_t managed_base;
    uint64_t managed_length;
    uint64_t _reserved1[3];
};

struct polaris_unregister_va_space_arg {
    uint32_t gpu_id;
    uint32_t _reserved0;
    uint64_t rm_client_token;
    uint64_t va_space_token;
    uint64_t _reserved1[1];
};

struct polaris_register_static_block_arg {
    uint32_t gpu_id;
    int32_t rm_control_fd;
    uint64_t rm_client_token;
    uint64_t va_space_token;
    uint64_t base;
    uint64_t length;
    uint64_t offset;
    uint32_t h_client;
    uint32_t h_memory;
    uint64_t _reserved[4];
};

struct polaris_unmap_static_block_arg {
    uint32_t gpu_id;
    uint32_t _reserved0;
    uint64_t rm_client_token;
    uint64_t va_space_token;
    uint64_t base;
    uint64_t length;
    uint64_t _reserved[4];
};

struct polaris_register_block_mapping_arg {
    uint64_t block_id;
    uint32_t gpu_id;
    uint32_t _reserved0;
    uint64_t rm_client_token;
    uint64_t va_space_token;
    uint64_t base;
    uint64_t length;
    uint64_t _reserved[4];
};

struct polaris_register_block_backing_arg {
    uint64_t block_id;
    uint32_t gpu_id;
    int32_t rm_control_fd;
    uint32_t h_client;
    uint32_t h_memory;
    uint64_t length;
    uint64_t offset;
    uint64_t _reserved[3];
};

struct polaris_unmap_block_mappings_arg {
    uint64_t block_id;
    uint32_t flags;
    uint32_t unmapped_count;
    uint64_t _reserved[4];
};

struct polaris_spill_block_arg {
    uint64_t block_id;
    uint32_t flags;
    uint32_t unmapped_count;
    uint64_t decision_id;
    uint64_t _reserved[3];
};

struct polaris_session_create_arg {
    uint64_t session_id;
    uint32_t home_gpu;
    uint32_t beam_width;
    uint64_t gpu_vas_bytes;
    uint64_t bytes_per_token;
    uint32_t priority;
    uint32_t _reserved;
    uint64_t _reserved2[2];
};

struct polaris_session_destroy_arg {
    uint64_t session_id;
    uint64_t _reserved[4];
};

struct polaris_block_reserve_arg {
    uint64_t session_id;
    uint32_t token_start;
    uint32_t token_count;
    uint32_t phase;
    uint32_t flags;
    uint64_t block_id;
    uint64_t gpu_vaddr;
    uint64_t _reserved[3];
};

struct polaris_block_release_arg {
    uint64_t session_id;
    uint32_t token_start;
    uint32_t token_count;
    uint32_t flags;
    uint32_t _reserved;
    uint64_t _reserved2[4];
};

#define POLARIS_REGISTER_GPU _IOW(POLARIS_IOCTL_MAGIC, 0x01, struct polaris_register_gpu_arg)
#define POLARIS_REGISTER_VA_RANGE _IOWR(POLARIS_IOCTL_MAGIC, 0x02, struct polaris_register_va_range_arg)
#define POLARIS_SESSION_CREATE _IOWR(POLARIS_IOCTL_MAGIC, 0x03, struct polaris_session_create_arg)
#define POLARIS_SESSION_DESTROY _IOW(POLARIS_IOCTL_MAGIC, 0x04, struct polaris_session_destroy_arg)
#define POLARIS_BLOCK_RESERVE _IOWR(POLARIS_IOCTL_MAGIC, 0x07, struct polaris_block_reserve_arg)
#define POLARIS_BLOCK_RELEASE _IOW(POLARIS_IOCTL_MAGIC, 0x08, struct polaris_block_release_arg)
#define POLARIS_REGISTER_VASPACE   _IOW(POLARIS_IOCTL_MAGIC, 0x10, struct polaris_register_va_space_arg)
#define POLARIS_UNREGISTER_VASPACE _IOW(POLARIS_IOCTL_MAGIC, 0x11, struct polaris_unregister_va_space_arg)
#define POLARIS_REGISTER_STATIC_BLOCK _IOW(POLARIS_IOCTL_MAGIC, 0x12, struct polaris_register_static_block_arg)
#define POLARIS_UNMAP_STATIC_BLOCK _IOW(POLARIS_IOCTL_MAGIC, 0x13, struct polaris_unmap_static_block_arg)
#define POLARIS_REGISTER_BLOCK_MAPPING _IOW(POLARIS_IOCTL_MAGIC, 0x14, struct polaris_register_block_mapping_arg)
#define POLARIS_UNMAP_BLOCK_MAPPINGS _IOWR(POLARIS_IOCTL_MAGIC, 0x15, struct polaris_unmap_block_mappings_arg)
#define POLARIS_SPILL_BLOCK _IOWR(POLARIS_IOCTL_MAGIC, 0x16, struct polaris_spill_block_arg)
#define POLARIS_REGISTER_BLOCK_BACKING _IOW(POLARIS_IOCTL_MAGIC, 0x17, struct polaris_register_block_backing_arg)

#define POLARIS_REGISTER_GPU_FLAG_TRANSIENT (1U << 0)
#define POLARIS_RELEASE_FLAG_CALLER_OWNS_BACKING (1U << 1)

#endif /* POLARIS_SHIM_POLARIS_ABI_H */
