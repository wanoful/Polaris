// SPDX-License-Identifier: GPL-2.0
//
// In-shim RM/UVM VA-space bootstrap. This mirrors the proven M2 harness setup
// but deliberately stops before static RM memory allocation: the production
// allocator registers external ranges per shim allocation and lets polaris.ko
// provide block backing through the UVM bridge.

#define NVTYPES_USE_STDINT 1

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "cuda_loader.h"
#include "rm_uvm_bootstrap.h"

#include "nv-ioctl.h"
#include "nv_escape.h"
#include "nvos.h"
#include "class/cl0000.h"
#include "class/cl003e.h"
#include "class/cl0040.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "class/cl90f1.h"
#include "uvm_linux_ioctl.h"
#include "uvm_test_ioctl.h"

#ifndef UVM_FAULT_ACCESS_TYPE_WRITE
#define UVM_FAULT_ACCESS_TYPE_WRITE 2
#endif

static int nv_status_ok(NV_STATUS status)
{
    return NV_STATUS_LEVEL(status) <= NV_STATUS_LEVEL_WARN;
}

static int nv_ioctl_checked(int fd, unsigned int esc, void *arg, size_t size, const char *what)
{
    unsigned long cmd = _IOWR(NV_IOCTL_MAGIC, esc, char[size]);

    if (ioctl(fd, cmd, arg) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] %s ioctl failed: %s\n",
                what,
                strerror(e));
        return -e;
    }
    return 0;
}

static int rm_alloc(int fd,
                    NvHandle h_root,
                    NvHandle h_parent,
                    NvHandle *h_new,
                    NvU32 h_class,
                    void *params,
                    NvU32 params_size,
                    const char *what)
{
    NVOS21_PARAMETERS alloc = {
        .hRoot = h_root,
        .hObjectParent = h_parent,
        .hObjectNew = *h_new,
        .hClass = h_class,
        .pAllocParms = (NvP64)(uintptr_t)params,
        .paramsSize = params_size,
    };
    int ret;

    ret = nv_ioctl_checked(fd, NV_ESC_RM_ALLOC, &alloc, sizeof(alloc), what);
    if (ret != 0)
        return ret;
    if (!nv_status_ok(alloc.status)) {
        fprintf(stderr,
                "[polaris-shim] %s RM status=0x%x\n",
                what,
                alloc.status);
        return -EIO;
    }

    *h_new = alloc.hObjectNew;
    return 0;
}

static void rm_free_object(int fd, NvHandle h_client, NvHandle h_parent, NvHandle h_object)
{
    NVOS00_PARAMETERS free_arg = {0};

    if (fd < 0 || h_client == 0 || h_object == 0)
        return;

    free_arg.hRoot = h_client;
    free_arg.hObjectParent = h_parent;
    free_arg.hObjectOld = h_object;
    (void)nv_ioctl_checked(fd, NV_ESC_RM_FREE, &free_arg, sizeof(free_arg), "RM_FREE");
}

static int get_gpu_uuid(int cuda_ordinal, NvProcessorUuid *uuid)
{
    cuInit_fn cu_init;
    cuDeviceGet_fn cu_device_get;
    cuDeviceGetUuid_fn cu_device_get_uuid;
    CUdevice dev = 0;
    CUuuid cu_uuid = {{0}};
    CUresult res;

    *(void **)(&cu_init) = polaris_shim_resolve_cuda_symbol("cuInit");
    if (!cu_init)
        return -ENODEV;
    res = cu_init(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr,
                "[polaris-shim] cuInit for RM/UVM bootstrap failed: %d\n",
                res);
        return -EIO;
    }

    *(void **)(&cu_device_get) = polaris_shim_resolve_cuda_symbol("cuDeviceGet");
    if (!cu_device_get)
        return -ENODEV;

    *(void **)(&cu_device_get_uuid) = polaris_shim_resolve_cuda_symbol("cuDeviceGetUuid");
    if (!cu_device_get_uuid)
        *(void **)(&cu_device_get_uuid) = polaris_shim_resolve_cuda_symbol("cuDeviceGetUuid_v2");
    if (!cu_device_get_uuid)
        return -ENODEV;

    res = cu_device_get(&dev, cuda_ordinal);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr,
                "[polaris-shim] cuDeviceGet(%d) failed: %d\n",
                cuda_ordinal,
                res);
        return -EIO;
    }

    res = cu_device_get_uuid(&cu_uuid, dev);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr,
                "[polaris-shim] cuDeviceGetUuid failed: %d\n",
                res);
        return -EIO;
    }

    memcpy(uuid->uuid, cu_uuid.bytes, sizeof(uuid->uuid));
    return 0;
}

static int setup_rm(int cuda_ordinal,
                    uint32_t gpu_id,
                    struct polaris_shim_bootstrap *state)
{
    NV0000_ALLOC_PARAMETERS root_params = {0};
    NV0080_ALLOC_PARAMETERS device_params = {0};
    NV2080_ALLOC_PARAMETERS subdevice_params = {0};
    NV_VASPACE_ALLOCATION_PARAMETERS vaspace_params = {0};
    char gpu_path[64];
    int ret;

    state->rm_control_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (state->rm_control_fd < 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] open /dev/nvidiactl failed: %s\n",
                strerror(e));
        return -e;
    }

    snprintf(gpu_path, sizeof(gpu_path), "/dev/nvidia%d", cuda_ordinal);
    state->gpu_fd = open(gpu_path, O_RDWR | O_CLOEXEC);
    if (state->gpu_fd < 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] open %s failed: %s\n",
                gpu_path,
                strerror(e));
        return -e;
    }

    snprintf(root_params.processName, sizeof(root_params.processName), "polaris-shim");
    ret = rm_alloc(state->rm_control_fd,
                   NV01_NULL_OBJECT,
                   NV01_NULL_OBJECT,
                   &state->h_client,
                   NV01_ROOT_CLIENT,
                   &root_params,
                   sizeof(root_params),
                   "RM_ALLOC root client");
    if (ret != 0)
        return ret;
    if (root_params.hClient != 0)
        state->h_client = root_params.hClient;

    (void)gpu_id;
    device_params.deviceId = (NvU32)cuda_ordinal;
    device_params.hClientShare = state->h_client;
    device_params.vaMode = NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES;
    ret = rm_alloc(state->rm_control_fd,
                   state->h_client,
                   state->h_client,
                   &state->h_device,
                   NV01_DEVICE_0,
                   &device_params,
                   sizeof(device_params),
                   "RM_ALLOC device");
    if (ret != 0)
        return ret;

    subdevice_params.subDeviceId = 0;
    ret = rm_alloc(state->rm_control_fd,
                   state->h_client,
                   state->h_device,
                   &state->h_subdevice,
                   NV20_SUBDEVICE_0,
                   &subdevice_params,
                   sizeof(subdevice_params),
                   "RM_ALLOC subdevice");
    if (ret != 0)
        return ret;

    vaspace_params.index = NV_VASPACE_ALLOCATION_INDEX_GPU_NEW;
    vaspace_params.flags = NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING |
                           NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED;
    ret = rm_alloc(state->rm_control_fd,
                   state->h_client,
                   state->h_device,
                   &state->h_vaspace,
                   FERMI_VASPACE_A,
                   &vaspace_params,
                   sizeof(vaspace_params),
                   "RM_ALLOC fault-capable VA-space");
    if (ret != 0)
        return ret;

    state->vaspace_base = vaspace_params.vaBase;
    state->vaspace_size = vaspace_params.vaSize;
    return 0;
}

static int setup_uvm(NvProcessorUuid *gpu_uuid,
                     struct polaris_shim_bootstrap *state)
{
    UVM_INITIALIZE_PARAMS init = {0};
    UVM_MM_INITIALIZE_PARAMS mm_init = {0};
    UVM_REGISTER_GPU_PARAMS reg_gpu = {0};
    UVM_REGISTER_GPU_VASPACE_PARAMS reg_va = {0};
    UVM_TEST_POLARIS_DISPATCH_FAULT_PARAMS fault = {0};

    state->uvm_fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (state->uvm_fd < 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] open /dev/nvidia-uvm failed: %s\n",
                strerror(e));
        return -e;
    }

    if (ioctl(state->uvm_fd, UVM_INITIALIZE, &init) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] UVM_INITIALIZE failed: %s\n",
                strerror(e));
        return -e;
    }
    if (!nv_status_ok(init.rmStatus)) {
        fprintf(stderr,
                "[polaris-shim] UVM_INITIALIZE rmStatus=0x%x\n",
                init.rmStatus);
        return -EIO;
    }

    state->uvm_mm_fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (state->uvm_mm_fd >= 0) {
        mm_init.uvmFd = state->uvm_fd;
        if (ioctl(state->uvm_mm_fd, UVM_MM_INITIALIZE, &mm_init) != 0) {
            int e = errno;
            fprintf(stderr,
                    "[polaris-shim] UVM_MM_INITIALIZE failed: %s\n",
                    strerror(e));
            return -e;
        }
        if (!nv_status_ok(mm_init.rmStatus)) {
            fprintf(stderr,
                    "[polaris-shim] UVM_MM_INITIALIZE rmStatus=0x%x\n",
                    mm_init.rmStatus);
            return -EIO;
        }
    }

    reg_gpu.gpu_uuid = *gpu_uuid;
    reg_gpu.rmCtrlFd = state->rm_control_fd;
    reg_gpu.hClient = state->h_client;
    reg_gpu.hSmcPartRef = 0;
    if (ioctl(state->uvm_fd, UVM_REGISTER_GPU, &reg_gpu) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] UVM_REGISTER_GPU failed: %s\n",
                strerror(e));
        return -e;
    }
    if (!nv_status_ok(reg_gpu.rmStatus)) {
        fprintf(stderr,
                "[polaris-shim] UVM_REGISTER_GPU rmStatus=0x%x\n",
                reg_gpu.rmStatus);
        return -EIO;
    }
    *gpu_uuid = reg_gpu.gpu_uuid;

    reg_va.gpuUuid = *gpu_uuid;
    reg_va.rmCtrlFd = state->rm_control_fd;
    reg_va.hClient = state->h_client;
    reg_va.hVaSpace = state->h_vaspace;
    if (ioctl(state->uvm_fd, UVM_REGISTER_GPU_VASPACE, &reg_va) != 0) {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] UVM_REGISTER_GPU_VASPACE failed: %s\n",
                strerror(e));
        return -e;
    }
    if (!nv_status_ok(reg_va.rmStatus)) {
        fprintf(stderr,
                "[polaris-shim] UVM_REGISTER_GPU_VASPACE rmStatus=0x%x\n",
                reg_va.rmStatus);
        return -EIO;
    }

    fault.gpu_uuid = *gpu_uuid;
    fault.fault_address = state->vaspace_base;
    fault.access_type = UVM_FAULT_ACCESS_TYPE_WRITE;
    if (ioctl(state->uvm_fd, UVM_TEST_POLARIS_DISPATCH_FAULT, &fault) == 0 &&
        nv_status_ok(fault.rmStatus)) {
        state->observed_gpu_id = fault.observed_gpu_id;
        state->observed_rm_client_token = fault.observed_rm_client_token;
        state->observed_va_space_token = fault.observed_va_space_token;
        fprintf(stderr,
                "[polaris-shim] UVM dispatch key gpu=%u client=0x%" PRIx64
                " token=0x%" PRIx64 " dry_result=%d\n",
                state->observed_gpu_id,
                state->observed_rm_client_token,
                state->observed_va_space_token,
                fault.polaris_status);
    } else {
        int e = errno;
        fprintf(stderr,
                "[polaris-shim] UVM dispatch key probe unavailable "
                "(errno=%d rmStatus=0x%x); using bootstrap RM handles\n",
                e,
                fault.rmStatus);
    }

    return 0;
}

int polaris_shim_bootstrap_rm_uvm(int cuda_ordinal,
                                  uint32_t gpu_id,
                                  struct polaris_shim_bootstrap *out)
{
    NvProcessorUuid gpu_uuid = {0};
    struct polaris_shim_bootstrap state = {
        .rm_control_fd = -1,
        .gpu_fd = -1,
        .uvm_fd = -1,
        .uvm_mm_fd = -1,
        .gpu_id = gpu_id,
    };
    int ret;

    if (!out)
        return -EINVAL;

    ret = get_gpu_uuid(cuda_ordinal, &gpu_uuid);
    if (ret != 0)
        return ret;

    ret = setup_rm(cuda_ordinal, gpu_id, &state);
    if (ret != 0)
        goto err;

    ret = setup_uvm(&gpu_uuid, &state);
    if (ret != 0)
        goto err;

    *out = state;
    fprintf(stderr,
            "[polaris-shim] RM/UVM bootstrap ready gpu=%u client=0x%x vaspace=0x%x va_base=0x%" PRIx64
            " va_size=0x%" PRIx64 "\n",
            gpu_id,
            state.h_client,
            state.h_vaspace,
            state.vaspace_base,
            state.vaspace_size);
    return 0;

err:
    polaris_shim_bootstrap_cleanup(&state);
    return ret;
}

void polaris_shim_bootstrap_cleanup(struct polaris_shim_bootstrap *state)
{
    if (!state)
        return;

    if (state->uvm_mm_fd >= 0) {
        close(state->uvm_mm_fd);
        state->uvm_mm_fd = -1;
    }
    if (state->uvm_fd >= 0) {
        close(state->uvm_fd);
        state->uvm_fd = -1;
    }

    rm_free_object(state->rm_control_fd, state->h_client, state->h_device, state->h_vaspace);
    state->h_vaspace = 0;
    rm_free_object(state->rm_control_fd, state->h_client, state->h_device, state->h_subdevice);
    state->h_subdevice = 0;
    rm_free_object(state->rm_control_fd, state->h_client, state->h_client, state->h_device);
    state->h_device = 0;
    rm_free_object(state->rm_control_fd, state->h_client, state->h_client, state->h_client);
    state->h_client = 0;

    if (state->gpu_fd >= 0) {
        close(state->gpu_fd);
        state->gpu_fd = -1;
    }
    if (state->rm_control_fd >= 0) {
        close(state->rm_control_fd);
        state->rm_control_fd = -1;
    }
}

int polaris_shim_rm_alloc_device_memory(const struct polaris_shim_bootstrap *state,
                                        uint64_t size,
                                        struct polaris_shim_rm_allocation *out)
{
    NV_MEMORY_ALLOCATION_PARAMS memory_params = {0};
    NvHandle h_memory = 0;
    int ret;

    if (!state || !out || state->rm_control_fd < 0 || state->h_client == 0 ||
        state->h_device == 0 || size == 0) {
        return -EINVAL;
    }

    memory_params.size = size;
    memory_params.owner = state->h_client;
    memory_params.type = NVOS32_TYPE_IMAGE;
    memory_params.attr = (NVOS32_ATTR_LOCATION_VIDMEM << 25);

    ret = rm_alloc(state->rm_control_fd,
                   state->h_client,
                   state->h_device,
                   &h_memory,
                   NV01_MEMORY_LOCAL_USER,
                   &memory_params,
                   sizeof(memory_params),
                   "RM_ALLOC shim static memory");
    if (ret != 0)
        return ret;

    out->h_memory = h_memory;
    out->size = memory_params.size;
    fprintf(stderr,
            "[polaris-shim] RM device allocation hMemory=0x%x size=0x%" PRIx64
            " limit=0x%" PRIx64 " offset=0x%" PRIx64 "\n",
            h_memory,
            memory_params.size,
            memory_params.limit,
            memory_params.offset);
    return 0;
}

void polaris_shim_rm_free_device_memory(const struct polaris_shim_bootstrap *state,
                                        struct polaris_shim_rm_allocation *allocation)
{
    if (!state || !allocation || allocation->h_memory == 0)
        return;

    rm_free_object(state->rm_control_fd,
                   state->h_client,
                   state->h_device,
                   allocation->h_memory);
    allocation->h_memory = 0;
    allocation->size = 0;
}
