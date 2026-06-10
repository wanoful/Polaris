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

int polaris_shim_register_vaspace(uint32_t gpu_id,
                                  uint64_t va_space_token,
                                  uint64_t managed_base,
                                  uint64_t managed_length)
{
    int fd = polaris_shim_device_fd();
    if (fd < 0)
        return -ENODEV;

    struct polaris_register_va_space_arg arg = {
        .gpu_id = gpu_id,
        .va_space_token = va_space_token,
        .managed_base = managed_base,
        .managed_length = managed_length,
    };

    if (ioctl(fd, POLARIS_REGISTER_VASPACE, &arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] POLARIS_REGISTER_VASPACE gpu=%u token=0x%llx failed: %s\n",
                gpu_id, (unsigned long long)va_space_token, strerror(e));
        return -e;
    }
    return 0;
}
