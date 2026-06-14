// SPDX-License-Identifier: GPL-2.0
//
// UVM external-range ioctls used by the shim allocator. This intentionally
// owns only the public UVM range lifecycle; RM object allocation and
// UVM_REGISTER_GPU_VASPACE remain separate bootstrap work.

#define NVTYPES_USE_STDINT 1

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "uvm_external.h"

#include "uvm_linux_ioctl.h"

static int g_uvm_fd = -1;

static int nv_status_ok(NV_STATUS status)
{
    return NV_STATUS_LEVEL(status) <= NV_STATUS_LEVEL_WARN;
}

int polaris_shim_uvm_adopt_fd(int uvm_fd)
{
    int dup_fd;

    if (uvm_fd < 0)
        return -EINVAL;

    dup_fd = fcntl(uvm_fd, F_DUPFD_CLOEXEC, 3);
    if (dup_fd < 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] dup UVM fd %d failed: %s\n",
                uvm_fd,
                strerror(e));
        return -e;
    }

    if (g_uvm_fd >= 0)
        close(g_uvm_fd);
    g_uvm_fd = dup_fd;
    return 0;
}

int polaris_shim_uvm_create_external_range(uint64_t base, uint64_t length)
{
    UVM_CREATE_EXTERNAL_RANGE_PARAMS params = {0};

    if (g_uvm_fd < 0)
        return -ENODEV;
    if (base == 0 || length == 0)
        return -EINVAL;

    params.base = base;
    params.length = length;

    if (ioctl(g_uvm_fd, UVM_CREATE_EXTERNAL_RANGE, &params) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] UVM_CREATE_EXTERNAL_RANGE base=0x%" PRIx64
                " len=0x%" PRIx64 " failed: %s\n",
                base,
                length,
                strerror(e));
        return -e;
    }

    if (!nv_status_ok(params.rmStatus)) {
        fprintf(stderr,
                "[polaris-shim] UVM_CREATE_EXTERNAL_RANGE base=0x%" PRIx64
                " len=0x%" PRIx64 " rmStatus=0x%x\n",
                base,
                length,
                params.rmStatus);
        return -EIO;
    }

    return 0;
}

int polaris_shim_uvm_free_external_range(uint64_t base)
{
    UVM_FREE_PARAMS params = {0};

    if (g_uvm_fd < 0)
        return -ENODEV;
    if (base == 0)
        return -EINVAL;

    params.base = base;

    if (ioctl(g_uvm_fd, UVM_FREE, &params) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] UVM_FREE base=0x%" PRIx64 " failed: %s\n",
                base,
                strerror(e));
        return -e;
    }

    if (!nv_status_ok(params.rmStatus)) {
        fprintf(stderr,
                "[polaris-shim] UVM_FREE base=0x%" PRIx64
                " rmStatus=0x%x\n",
                base,
                params.rmStatus);
        return -EIO;
    }

    return 0;
}
