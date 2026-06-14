// SPDX-License-Identifier: GPL-2.0
//
// /dev/polaris connection management for libpolaris-shim. The device is
// opened lazily on first use, the fd is cached for the life of the process,
// and failures are non-fatal: the worker keeps running without POLARIS
// managed paging — the LD_PRELOAD becomes equivalent to a no-op shim.

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "device.h"
#include "polaris_abi.h"

static pthread_once_t g_device_once = PTHREAD_ONCE_INIT;
static int g_device_fd = -1;

static void open_device(void)
{
    g_device_fd = open(POLARIS_DEVICE_PATH, O_RDWR | O_CLOEXEC);
    if (g_device_fd < 0) {
        // ENOENT is the normal "polaris.ko not loaded" case; everything
        // else is at least worth a one-line note.
        fprintf(stderr, "[polaris-shim] %s unavailable: %s\n",
                POLARIS_DEVICE_PATH, strerror(errno));
    }
}

int polaris_shim_device_fd(void)
{
    pthread_once(&g_device_once, open_device);
    return g_device_fd;
}

int polaris_shim_register_gpu(uint32_t gpu_id,
                              uint64_t total_bytes,
                              uint64_t budget_bytes,
                              uint64_t cpu_pool_bytes,
                              uint32_t flags)
{
    int fd = polaris_shim_device_fd();
    if (fd < 0)
        return -ENODEV;

    struct polaris_register_gpu_arg arg = {
        .gpu_id = gpu_id,
        .total_bytes = total_bytes,
        .budget_bytes = budget_bytes,
        .cpu_pool_bytes = cpu_pool_bytes,
        ._reserved = flags,
    };

    if (ioctl(fd, POLARIS_REGISTER_GPU, &arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] POLARIS_REGISTER_GPU gpu=%u failed: %s\n",
                gpu_id, strerror(e));
        return -e;
    }
    return 0;
}

int polaris_shim_register_va_range(uint32_t gpu_id,
                                   uint64_t base,
                                   uint64_t length,
                                   uint64_t block_size,
                                   uint64_t *range_id_out)
{
    int fd = polaris_shim_device_fd();
    if (fd < 0)
        return -ENODEV;

    struct polaris_register_va_range_arg arg = {
        .gpu_id = gpu_id,
        .base = base,
        .length = length,
        .block_size = block_size,
    };

    if (ioctl(fd, POLARIS_REGISTER_VA_RANGE, &arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] POLARIS_REGISTER_VA_RANGE gpu=%u base=0x%llx len=0x%llx block=0x%llx failed: %s\n",
                gpu_id,
                (unsigned long long)base,
                (unsigned long long)length,
                (unsigned long long)block_size,
                strerror(e));
        return -e;
    }

    if (range_id_out)
        *range_id_out = arg.range_id;
    return 0;
}

int polaris_shim_session_create(uint32_t gpu_id,
                                uint64_t gpu_vas_bytes,
                                uint64_t bytes_per_token,
                                uint64_t *session_id_out)
{
    int fd = polaris_shim_device_fd();
    if (fd < 0)
        return -ENODEV;

    struct polaris_session_create_arg arg = {
        .home_gpu = gpu_id,
        .gpu_vas_bytes = gpu_vas_bytes,
        .bytes_per_token = bytes_per_token,
        .priority = 5,
    };

    if (ioctl(fd, POLARIS_SESSION_CREATE, &arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] POLARIS_SESSION_CREATE gpu=%u vas=0x%llx bpt=0x%llx failed: %s\n",
                gpu_id,
                (unsigned long long)gpu_vas_bytes,
                (unsigned long long)bytes_per_token,
                strerror(e));
        return -e;
    }

    if (session_id_out)
        *session_id_out = arg.session_id;
    return 0;
}

int polaris_shim_session_destroy(uint64_t session_id)
{
    int fd = polaris_shim_device_fd();
    if (fd < 0)
        return -ENODEV;

    struct polaris_session_destroy_arg arg = {
        .session_id = session_id,
    };

    if (ioctl(fd, POLARIS_SESSION_DESTROY, &arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] POLARIS_SESSION_DESTROY session=%llu failed: %s\n",
                (unsigned long long)session_id, strerror(e));
        return -e;
    }
    return 0;
}

int polaris_shim_block_reserve(uint64_t session_id,
                               uint32_t token_start,
                               uint32_t token_count,
                               uint32_t flags,
                               uint64_t *block_id_out,
                               uint64_t *gpu_vaddr_out)
{
    int fd = polaris_shim_device_fd();
    if (fd < 0)
        return -ENODEV;

    struct polaris_block_reserve_arg arg = {
        .session_id = session_id,
        .token_start = token_start,
        .token_count = token_count,
        .phase = POLARIS_PHASE_PREFILL,
        .flags = flags,
    };

    if (ioctl(fd, POLARIS_BLOCK_RESERVE, &arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] POLARIS_BLOCK_RESERVE session=%llu token=%u count=%u flags=0x%x failed: %s\n",
                (unsigned long long)session_id,
                token_start,
                token_count,
                flags,
                strerror(e));
        return -e;
    }

    if (block_id_out)
        *block_id_out = arg.block_id;
    if (gpu_vaddr_out)
        *gpu_vaddr_out = arg.gpu_vaddr;
    return 0;
}

int polaris_shim_block_release(uint64_t session_id,
                               uint32_t token_start,
                               uint32_t token_count)
{
    int fd = polaris_shim_device_fd();
    if (fd < 0)
        return -ENODEV;

    struct polaris_block_release_arg arg = {
        .session_id = session_id,
        .token_start = token_start,
        .token_count = token_count,
    };

    if (ioctl(fd, POLARIS_BLOCK_RELEASE, &arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] POLARIS_BLOCK_RELEASE session=%llu token=%u count=%u failed: %s\n",
                (unsigned long long)session_id,
                token_start,
                token_count,
                strerror(e));
        return -e;
    }
    return 0;
}

int polaris_shim_register_vaspace(uint32_t gpu_id,
                                  uint64_t rm_client_token,
                                  uint64_t va_space_token,
                                  uint64_t managed_base,
                                  uint64_t managed_length)
{
    int fd = polaris_shim_device_fd();
    if (fd < 0)
        return -ENODEV;

    struct polaris_register_va_space_arg arg = {
        .gpu_id = gpu_id,
        .rm_client_token = rm_client_token,
        .va_space_token = va_space_token,
        .managed_base = managed_base,
        .managed_length = managed_length,
    };

    if (ioctl(fd, POLARIS_REGISTER_VASPACE, &arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] POLARIS_REGISTER_VASPACE gpu=%u client=0x%llx token=0x%llx failed: %s\n",
                gpu_id,
                (unsigned long long)rm_client_token,
                (unsigned long long)va_space_token,
                strerror(e));
        return -e;
    }
    return 0;
}

int polaris_shim_register_block_mapping(uint64_t block_id,
                                        uint32_t gpu_id,
                                        uint64_t rm_client_token,
                                        uint64_t va_space_token,
                                        uint64_t base,
                                        uint64_t length)
{
    int fd = polaris_shim_device_fd();
    if (fd < 0)
        return -ENODEV;

    struct polaris_register_block_mapping_arg arg = {
        .block_id = block_id,
        .gpu_id = gpu_id,
        .rm_client_token = rm_client_token,
        .va_space_token = va_space_token,
        .base = base,
        .length = length,
    };

    if (ioctl(fd, POLARIS_REGISTER_BLOCK_MAPPING, &arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] POLARIS_REGISTER_BLOCK_MAPPING block=%llu gpu=%u client=0x%llx token=0x%llx base=0x%llx len=0x%llx failed: %s\n",
                (unsigned long long)block_id,
                gpu_id,
                (unsigned long long)rm_client_token,
                (unsigned long long)va_space_token,
                (unsigned long long)base,
                (unsigned long long)length,
                strerror(e));
        return -e;
    }
    return 0;
}

int polaris_shim_unregister_vaspace(uint32_t gpu_id,
                                    uint64_t rm_client_token,
                                    uint64_t va_space_token)
{
    int fd = polaris_shim_device_fd();
    if (fd < 0)
        return -ENODEV;

    struct polaris_unregister_va_space_arg arg = {
        .gpu_id = gpu_id,
        .rm_client_token = rm_client_token,
        .va_space_token = va_space_token,
    };

    if (ioctl(fd, POLARIS_UNREGISTER_VASPACE, &arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] POLARIS_UNREGISTER_VASPACE gpu=%u client=0x%llx token=0x%llx failed: %s\n",
                gpu_id,
                (unsigned long long)rm_client_token,
                (unsigned long long)va_space_token,
                strerror(e));
        return -e;
    }
    return 0;
}

int polaris_shim_register_static_block(uint32_t gpu_id,
                                       uint64_t rm_client_token,
                                       uint64_t va_space_token,
                                       uint64_t base,
                                       uint64_t length,
                                       uint64_t offset,
                                       int32_t rm_control_fd,
                                       uint32_t h_client,
                                       uint32_t h_memory)
{
    int fd = polaris_shim_device_fd();
    if (fd < 0)
        return -ENODEV;

    struct polaris_register_static_block_arg arg = {
        .gpu_id = gpu_id,
        .rm_control_fd = rm_control_fd,
        .rm_client_token = rm_client_token,
        .va_space_token = va_space_token,
        .base = base,
        .length = length,
        .offset = offset,
        .h_client = h_client,
        .h_memory = h_memory,
    };

    if (ioctl(fd, POLARIS_REGISTER_STATIC_BLOCK, &arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] POLARIS_REGISTER_STATIC_BLOCK gpu=%u client=0x%llx token=0x%llx base=0x%llx len=0x%llx failed: %s\n",
                gpu_id,
                (unsigned long long)rm_client_token,
                (unsigned long long)va_space_token,
                (unsigned long long)base,
                (unsigned long long)length,
                strerror(e));
        return -e;
    }
    return 0;
}

int polaris_shim_unmap_static_block(uint32_t gpu_id,
                                    uint64_t rm_client_token,
                                    uint64_t va_space_token,
                                    uint64_t base,
                                    uint64_t length)
{
    int fd = polaris_shim_device_fd();
    if (fd < 0)
        return -ENODEV;

    struct polaris_unmap_static_block_arg arg = {
        .gpu_id = gpu_id,
        .rm_client_token = rm_client_token,
        .va_space_token = va_space_token,
        .base = base,
        .length = length,
    };

    if (ioctl(fd, POLARIS_UNMAP_STATIC_BLOCK, &arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] POLARIS_UNMAP_STATIC_BLOCK gpu=%u client=0x%llx token=0x%llx base=0x%llx len=0x%llx failed: %s\n",
                gpu_id,
                (unsigned long long)rm_client_token,
                (unsigned long long)va_space_token,
                (unsigned long long)base,
                (unsigned long long)length,
                strerror(e));
        return -e;
    }
    return 0;
}
