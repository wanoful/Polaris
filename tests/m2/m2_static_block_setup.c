// SPDX-License-Identifier: GPL-2.0
//
// Staged M2 diagnostic for the POLARIS v4 static-block fault path.
//
// By default this program validates the hard setup half of M2: real RM
// client/device/VA-space/memory handles, UVM GPU registration, external-range
// creation, and POLARIS static-block registration. With --dispatch-fault it
// leaves the external range unmapped and asks UVM's builtin test ioctl to
// dispatch a synthetic fault through the live Polaris hook, validating the
// hook-to-PTE-install bridge without a userspace RM GPFIFO channel. With
// --unmap-refault it additionally tears the mapping down through POLARIS's M3
// static unmap bridge and validates that a second synthetic fault maps it
// again. With --block-unmap-refault it registers a logical block mapping,
// unmaps that block through the production-oriented block teardown ioctl, and
// validates refault.

#define _GNU_SOURCE
#define NVTYPES_USE_STDINT 1

#include <cuda.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/ioctl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

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

#ifndef BIT
#define BIT(n) (1U << (n))
#endif

#define POLARIS_IOCTL_MAGIC 0x50
#define POLARIS_BLOCK_SIZE (2ULL * 1024ULL * 1024ULL)
#define POLARIS_MANAGED_SIZE (4ULL * 1024ULL * 1024ULL)

struct polaris_register_gpu_arg {
    uint32_t gpu_id;
    uint32_t _pad0;
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

struct polaris_register_vaspace_arg {
    uint32_t gpu_id;
    uint32_t _reserved0;
    uint64_t rm_client_token;
    uint64_t va_space_token;
    uint64_t managed_base;
    uint64_t managed_length;
    uint64_t _reserved1[3];
};

struct polaris_unregister_vaspace_arg {
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

enum {
    POLARIS_DECISION_OP_ALLOC = 0,
    POLARIS_DECISION_OP_FREE = 1,
    POLARIS_DECISION_OP_MAP_EXISTING = 2,
    POLARIS_DECISION_OP_UNMAP = 3,
    POLARIS_DECISION_OP_OFFLOAD = 4,
    POLARIS_DECISION_OP_RELOAD = 5,
    POLARIS_DECISION_OP_COW_BREAK = 6,
};

#define POLARIS_MAX_DECISIONS_PER_POLL 16

struct polaris_decision {
    uint64_t decision_id;
    uint64_t fault_id;
    uint64_t generation;
    uint32_t op;
    uint32_t gpu_id;
    uint64_t block_id;
    uint64_t session_id;
    uint64_t src_handle;
    uint64_t dst_handle;
    uint64_t src_vaddr;
    uint64_t dst_vaddr;
    uint64_t size_bytes;
    uint64_t cpu_addr;
    uint32_t access_flags;
    uint32_t timeout_ms;
    uint64_t _reserved[4];
};

struct polaris_get_decision_arg {
    uint32_t count;
    uint32_t _reserved;
    struct polaris_decision decisions[POLARIS_MAX_DECISIONS_PER_POLL];
};

struct polaris_complete_operation_arg {
    uint64_t decision_id;
    uint64_t generation;
    int32_t result;
    int32_t rm_control_fd;
    uint64_t output_handle;
    uint64_t output_cpu_addr;
    uint32_t rm_h_client;
    uint32_t rm_h_memory;
    uint64_t rm_backing_length;
};

#define POLARIS_REGISTER_GPU _IOW(POLARIS_IOCTL_MAGIC, 0x01, struct polaris_register_gpu_arg)
#define POLARIS_REGISTER_VA_RANGE _IOWR(POLARIS_IOCTL_MAGIC, 0x02, struct polaris_register_va_range_arg)
#define POLARIS_SESSION_CREATE _IOWR(POLARIS_IOCTL_MAGIC, 0x03, struct polaris_session_create_arg)
#define POLARIS_SESSION_DESTROY _IOW(POLARIS_IOCTL_MAGIC, 0x04, struct polaris_session_destroy_arg)
#define POLARIS_BLOCK_RESERVE _IOWR(POLARIS_IOCTL_MAGIC, 0x07, struct polaris_block_reserve_arg)
#define POLARIS_GET_DECISION _IOWR(POLARIS_IOCTL_MAGIC, 0x0b, struct polaris_get_decision_arg)
#define POLARIS_COMPLETE_OPERATION _IOWR(POLARIS_IOCTL_MAGIC, 0x0c, struct polaris_complete_operation_arg)
#define POLARIS_REGISTER_VASPACE _IOW(POLARIS_IOCTL_MAGIC, 0x10, struct polaris_register_vaspace_arg)
#define POLARIS_UNREGISTER_VASPACE _IOW(POLARIS_IOCTL_MAGIC, 0x11, struct polaris_unregister_vaspace_arg)
#define POLARIS_REGISTER_STATIC_BLOCK _IOW(POLARIS_IOCTL_MAGIC, 0x12, struct polaris_register_static_block_arg)
#define POLARIS_UNMAP_STATIC_BLOCK _IOW(POLARIS_IOCTL_MAGIC, 0x13, struct polaris_unmap_static_block_arg)
#define POLARIS_REGISTER_BLOCK_MAPPING _IOW(POLARIS_IOCTL_MAGIC, 0x14, struct polaris_register_block_mapping_arg)
#define POLARIS_UNMAP_BLOCK_MAPPINGS _IOWR(POLARIS_IOCTL_MAGIC, 0x15, struct polaris_unmap_block_mappings_arg)
#define POLARIS_SPILL_BLOCK _IOWR(POLARIS_IOCTL_MAGIC, 0x16, struct polaris_spill_block_arg)
#define POLARIS_REGISTER_BLOCK_BACKING _IOW(POLARIS_IOCTL_MAGIC, 0x17, struct polaris_register_block_backing_arg)
#define POLARIS_REGISTER_GPU_FLAG_TRANSIENT (1U << 0)
#define POLARIS_RESERVE_FLAG_DEFER_FAULT (1U << 4)

#define NV_IOCTL(cmd, type) _IOWR(NV_IOCTL_MAGIC, (cmd), type)
#define UVM_TEST_IOCTL_BASE(i) UVM_IOCTL_BASE(200 + (i))
#define UVM_TEST_POLARIS_DISPATCH_FAULT UVM_TEST_IOCTL_BASE(117)

enum {
    UVM_FAULT_ACCESS_TYPE_PREFETCH = 0,
    UVM_FAULT_ACCESS_TYPE_READ = 1,
    UVM_FAULT_ACCESS_TYPE_WRITE = 2,
    UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK = 3,
    UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG = 4,
};

struct uvm_test_polaris_dispatch_fault_params {
    NvProcessorUuid gpu_uuid;
    NvU64 fault_address;
    NvU32 access_type;
    NvS32 polaris_status;
    NvU32 observed_gpu_id;
    NvU64 observed_rm_client_token;
    NvU64 observed_va_space_token;
    NV_STATUS rmStatus;
};

struct m2_state {
    int ctl_fd;
    int gpu_fd;
    int uvm_fd;
    int uvm_mm_fd;
    int polaris_fd;
    NvHandle h_client;
    NvHandle h_device;
    NvHandle h_subdevice;
    NvHandle h_vaspace;
    NvHandle h_memory;
    NvProcessorUuid gpu_uuid;
    bool uvm_initialized;
    bool uvm_gpu_registered;
    bool uvm_vaspace_registered;
    bool polaris_vaspace_registered;
    uint64_t polaris_session_id;
};

struct completion_executor_args {
    struct m2_state *state;
    uint32_t gpu_id;
    uint64_t expected_base;
    uint64_t completed_block_id;
    int result;
};

static int nv_ioctl_checked(int fd, unsigned int esc, void *arg, size_t size, const char *what)
{
    unsigned long cmd = _IOWR(NV_IOCTL_MAGIC, esc, char[size]);
    if (ioctl(fd, cmd, arg) != 0) {
        fprintf(stderr, "%s ioctl failed: errno=%d (%s)\n", what, errno, strerror(errno));
        return -1;
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

    if (nv_ioctl_checked(fd, NV_ESC_RM_ALLOC, &alloc, sizeof(alloc), what) != 0)
        return -1;
    if (alloc.status != NV_OK) {
        fprintf(stderr, "%s RM status=0x%x\n", what, alloc.status);
        return -1;
    }

    *h_new = alloc.hObjectNew;
    printf("%s: handle=0x%08x\n", what, *h_new);
    return 0;
}

static void rm_free_object(int fd, NvHandle h_client, NvHandle h_parent, NvHandle h_object)
{
    if (h_client == 0 || h_object == 0)
        return;

    NVOS00_PARAMETERS free_arg = {
        .hRoot = h_client,
        .hObjectParent = h_parent,
        .hObjectOld = h_object,
    };
    (void)nv_ioctl_checked(fd, NV_ESC_RM_FREE, &free_arg, sizeof(free_arg), "RM_FREE");
}

static int uvm_ioctl_checked(int fd, unsigned long cmd, void *arg, const char *what)
{
    if (ioctl(fd, cmd, arg) != 0) {
        fprintf(stderr, "%s ioctl failed: errno=%d (%s)\n", what, errno, strerror(errno));
        return -1;
    }
    return 0;
}

static int polaris_ioctl_checked(int fd, unsigned long cmd, void *arg, const char *what)
{
    if (ioctl(fd, cmd, arg) != 0) {
        fprintf(stderr, "%s ioctl failed: errno=%d (%s)\n", what, errno, strerror(errno));
        return -1;
    }
    return 0;
}

static int get_cuda_uuid(int ordinal, NvProcessorUuid *uuid)
{
    CUdevice dev;
    CUuuid cu_uuid;
    CUresult res;

    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", res);
        return -1;
    }
    res = cuDeviceGet(&dev, ordinal);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGet(%d) failed: %d\n", ordinal, res);
        return -1;
    }
    res = cuDeviceGetUuid(&cu_uuid, dev);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGetUuid failed: %d\n", res);
        return -1;
    }

    memcpy(uuid->uuid, cu_uuid.bytes, sizeof(uuid->uuid));
    printf("CUDA GPU%d UUID bytes:", ordinal);
    for (size_t i = 0; i < sizeof(uuid->uuid); ++i)
        printf("%02x", uuid->uuid[i]);
    printf("\n");
    return 0;
}

static int setup_rm(struct m2_state *s, int device_ordinal)
{
    NV0000_ALLOC_PARAMETERS root_params = {0};
    NV0080_ALLOC_PARAMETERS device_params = {0};
    NV2080_ALLOC_PARAMETERS subdevice_params = {0};
    NV_VASPACE_ALLOCATION_PARAMETERS vaspace_params = {0};
    NV_MEMORY_ALLOCATION_PARAMS memory_params = {0};

    s->ctl_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (s->ctl_fd < 0) {
        fprintf(stderr, "open /dev/nvidiactl failed: errno=%d (%s)\n", errno, strerror(errno));
        return -1;
    }

    char gpu_path[64];
    snprintf(gpu_path, sizeof(gpu_path), "/dev/nvidia%d", device_ordinal);
    s->gpu_fd = open(gpu_path, O_RDWR | O_CLOEXEC);
    if (s->gpu_fd < 0) {
        fprintf(stderr, "open %s failed: errno=%d (%s)\n", gpu_path, errno, strerror(errno));
        return -1;
    }

    snprintf(root_params.processName, sizeof(root_params.processName), "polaris-m2");
    if (rm_alloc(s->ctl_fd,
                 NV01_NULL_OBJECT,
                 NV01_NULL_OBJECT,
                 &s->h_client,
                 NV01_ROOT_CLIENT,
                 &root_params,
                 sizeof(root_params),
                 "RM_ALLOC root client") != 0)
        return -1;
    if (root_params.hClient != 0)
        s->h_client = root_params.hClient;

    device_params.deviceId = (NvU32)device_ordinal;
    device_params.hClientShare = s->h_client;
    device_params.vaMode = NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES;
    if (rm_alloc(s->ctl_fd,
                 s->h_client,
                 s->h_client,
                 &s->h_device,
                 NV01_DEVICE_0,
                 &device_params,
                 sizeof(device_params),
                 "RM_ALLOC device") != 0)
        return -1;

    subdevice_params.subDeviceId = 0;
    if (rm_alloc(s->ctl_fd,
                 s->h_client,
                 s->h_device,
                 &s->h_subdevice,
                 NV20_SUBDEVICE_0,
                 &subdevice_params,
                 sizeof(subdevice_params),
                 "RM_ALLOC subdevice") != 0)
        return -1;

    const struct {
        const char *name;
        NvU32 flags;
    } vaspace_attempts[] = {
        {
            "fault-capable externally-owned VA-space",
            NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING |
                NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED,
        },
        {
            "externally-owned VA-space",
            NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED,
        },
        {
            "plain private VA-space",
            NV_VASPACE_ALLOCATION_FLAGS_NONE,
        },
    };
    int vaspace_ok = 0;
    for (size_t i = 0; i < sizeof(vaspace_attempts) / sizeof(vaspace_attempts[0]); ++i) {
        memset(&vaspace_params, 0, sizeof(vaspace_params));
        s->h_vaspace = 0;
        vaspace_params.index = NV_VASPACE_ALLOCATION_INDEX_GPU_NEW;
        vaspace_params.flags = vaspace_attempts[i].flags;
        vaspace_params.vaSize = 0;
        vaspace_params.vaBase = 0;
        vaspace_params.bigPageSize = 0;

        if (rm_alloc(s->ctl_fd,
                     s->h_client,
                     s->h_device,
                     &s->h_vaspace,
                     FERMI_VASPACE_A,
                     &vaspace_params,
                     sizeof(vaspace_params),
                     vaspace_attempts[i].name) == 0) {
            if (vaspace_attempts[i].flags !=
                (NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING |
                 NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)) {
                fprintf(stderr,
                        "Allocated %s, but M2 requires fault-capable externally-owned VA-space\n",
                        vaspace_attempts[i].name);
                return -1;
            }
            vaspace_ok = 1;
            break;
        }
    }
    if (!vaspace_ok)
        return -1;

    printf("RM VA-space base=0x%llx size=0x%llx\n",
           (unsigned long long)vaspace_params.vaBase,
           (unsigned long long)vaspace_params.vaSize);

    memory_params.size = POLARIS_BLOCK_SIZE;
    memory_params.owner = s->h_client;
    memory_params.type = NVOS32_TYPE_IMAGE;
    memory_params.attr = (NVOS32_ATTR_LOCATION_VIDMEM << 25);
    if (rm_alloc(s->ctl_fd,
                 s->h_client,
                 s->h_device,
                 &s->h_memory,
                 NV01_MEMORY_LOCAL_USER,
                 &memory_params,
                 sizeof(memory_params),
                 "RM_ALLOC static memory") != 0)
        return -1;

    printf("RM memory size=0x%llx limit=0x%llx offset=0x%llx\n",
           (unsigned long long)memory_params.size,
           (unsigned long long)memory_params.limit,
           (unsigned long long)memory_params.offset);
    return 0;
}

static int setup_uvm(struct m2_state *s, uint64_t base, uint64_t length, bool premap)
{
    UVM_INITIALIZE_PARAMS init = {0};
    UVM_MM_INITIALIZE_PARAMS mm_init = {0};
    UVM_REGISTER_GPU_PARAMS reg_gpu = {0};
    UVM_REGISTER_GPU_VASPACE_PARAMS reg_va = {0};
    UVM_CREATE_EXTERNAL_RANGE_PARAMS ext = {0};

    s->uvm_fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (s->uvm_fd < 0) {
        fprintf(stderr, "open /dev/nvidia-uvm failed: errno=%d (%s)\n", errno, strerror(errno));
        return -1;
    }

    if (uvm_ioctl_checked(s->uvm_fd, UVM_INITIALIZE, &init, "UVM_INITIALIZE") != 0)
        return -1;
    if (init.rmStatus != NV_OK) {
        fprintf(stderr, "UVM_INITIALIZE status=0x%x\n", init.rmStatus);
        return -1;
    }
    s->uvm_initialized = true;

    s->uvm_mm_fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (s->uvm_mm_fd < 0) {
        fprintf(stderr, "open /dev/nvidia-uvm for MM failed: errno=%d (%s)\n", errno, strerror(errno));
        return -1;
    }

    mm_init.uvmFd = s->uvm_fd;
    if (uvm_ioctl_checked(s->uvm_mm_fd, UVM_MM_INITIALIZE, &mm_init, "UVM_MM_INITIALIZE") != 0)
        return -1;
    if (mm_init.rmStatus != NV_OK && mm_init.rmStatus != NV_WARN_NOTHING_TO_DO) {
        fprintf(stderr, "UVM_MM_INITIALIZE status=0x%x\n", mm_init.rmStatus);
        return -1;
    }

    reg_gpu.gpu_uuid = s->gpu_uuid;
    reg_gpu.rmCtrlFd = s->ctl_fd;
    reg_gpu.hClient = s->h_client;
    reg_gpu.hSmcPartRef = 0;
    if (uvm_ioctl_checked(s->uvm_fd, UVM_REGISTER_GPU, &reg_gpu, "UVM_REGISTER_GPU") != 0)
        return -1;
    if (reg_gpu.rmStatus != NV_OK) {
        fprintf(stderr, "UVM_REGISTER_GPU status=0x%x\n", reg_gpu.rmStatus);
        return -1;
    }
    s->uvm_gpu_registered = true;
    s->gpu_uuid = reg_gpu.gpu_uuid;

    reg_va.gpuUuid = s->gpu_uuid;
    reg_va.rmCtrlFd = s->ctl_fd;
    reg_va.hClient = s->h_client;
    reg_va.hVaSpace = s->h_vaspace;
    if (uvm_ioctl_checked(s->uvm_fd, UVM_REGISTER_GPU_VASPACE, &reg_va, "UVM_REGISTER_GPU_VASPACE") != 0)
        return -1;
    if (reg_va.rmStatus != NV_OK) {
        fprintf(stderr, "UVM_REGISTER_GPU_VASPACE status=0x%x\n", reg_va.rmStatus);
        return -1;
    }
    s->uvm_vaspace_registered = true;

    ext.base = base;
    ext.length = length;
    if (uvm_ioctl_checked(s->uvm_fd, UVM_CREATE_EXTERNAL_RANGE, &ext, "UVM_CREATE_EXTERNAL_RANGE") != 0)
        return -1;
    if (ext.rmStatus != NV_OK) {
        fprintf(stderr, "UVM_CREATE_EXTERNAL_RANGE status=0x%x\n", ext.rmStatus);
        return -1;
    }

    if (premap) {
        UVM_MAP_EXTERNAL_ALLOCATION_PARAMS map = {0};

        map.base = base;
        map.length = POLARIS_BLOCK_SIZE;
        map.offset = 0;
        map.perGpuAttributes[0].gpuUuid = s->gpu_uuid;
        map.perGpuAttributes[0].gpuMappingType = UvmGpuMappingTypeReadWriteAtomic;
        map.perGpuAttributes[0].gpuCachingType = UvmGpuCachingTypeDefault;
        map.perGpuAttributes[0].gpuFormatType = UvmGpuFormatTypeDefault;
        map.perGpuAttributes[0].gpuElementBits = UvmGpuFormatElementBitsDefault;
        map.perGpuAttributes[0].gpuCompressionType = UvmGpuCompressionTypeDefault;
        map.gpuAttributesCount = 1;
        map.rmCtrlFd = s->ctl_fd;
        map.hClient = s->h_client;
        map.hMemory = s->h_memory;
        if (uvm_ioctl_checked(s->uvm_fd, UVM_MAP_EXTERNAL_ALLOCATION, &map, "UVM_MAP_EXTERNAL_ALLOCATION") != 0)
            return -1;
        if (map.rmStatus != NV_OK) {
            fprintf(stderr, "UVM_MAP_EXTERNAL_ALLOCATION status=0x%x\n", map.rmStatus);
            return -1;
        }
    }

    printf("UVM setup complete for base=0x%llx len=0x%llx premap=%s\n",
           (unsigned long long)base,
           (unsigned long long)length,
           premap ? "yes" : "no");
    return 0;
}

static int dispatch_test_fault(struct m2_state *s, uint64_t fault_address)
{
    struct uvm_test_polaris_dispatch_fault_params fault = {
        .gpu_uuid = s->gpu_uuid,
        .fault_address = fault_address,
        .access_type = UVM_FAULT_ACCESS_TYPE_WRITE,
    };

    if (uvm_ioctl_checked(s->uvm_fd,
                          UVM_TEST_POLARIS_DISPATCH_FAULT,
                          &fault,
                          "UVM_TEST_POLARIS_DISPATCH_FAULT") != 0)
        return -1;
    if (fault.rmStatus != NV_OK) {
        fprintf(stderr, "UVM_TEST_POLARIS_DISPATCH_FAULT status=0x%x\n", fault.rmStatus);
        return -1;
    }
    printf("UVM dispatch key: gpu_id=%u client=0x%llx token=0x%llx result=%d\n",
           fault.observed_gpu_id,
           (unsigned long long)fault.observed_rm_client_token,
           (unsigned long long)fault.observed_va_space_token,
           fault.polaris_status);
    if (fault.polaris_status != 1) {
        fprintf(stderr, "Polaris dispatch result=%d, expected HANDLED(1)\n", fault.polaris_status);
        return -1;
    }

    printf("Polaris synthetic fault dispatch handled address=0x%llx\n",
           (unsigned long long)fault_address);
    return 0;
}

static int unmap_static_block(struct m2_state *s,
                              uint32_t gpu_id,
                              uint64_t rm_client_token,
                              uint64_t va_space_token,
                              uint64_t base,
                              uint64_t length)
{
    struct polaris_unmap_static_block_arg unmap = {
        .gpu_id = gpu_id,
        .rm_client_token = rm_client_token,
        .va_space_token = va_space_token,
        .base = base,
        .length = length,
    };

    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_UNMAP_STATIC_BLOCK,
                              &unmap,
                              "POLARIS_UNMAP_STATIC_BLOCK") != 0)
        return -1;

    printf("POLARIS unmapped static block: gpu_id=%u client=0x%llx token=0x%llx base=0x%llx len=0x%llx\n",
           gpu_id,
           (unsigned long long)rm_client_token,
           (unsigned long long)va_space_token,
           (unsigned long long)base,
           (unsigned long long)length);
    return 0;
}

static void *complete_backing_executor(void *arg)
{
    struct completion_executor_args *args = arg;
    struct m2_state *s = args->state;

    args->result = -1;
    for (int attempt = 0; attempt < 500; ++attempt) {
        struct polaris_get_decision_arg get = {0};

        if (ioctl(s->polaris_fd, POLARIS_GET_DECISION, &get) != 0) {
            fprintf(stderr,
                    "POLARIS_GET_DECISION executor failed: errno=%d (%s)\n",
                    errno,
                    strerror(errno));
            return NULL;
        }

        for (uint32_t i = 0; i < get.count && i < POLARIS_MAX_DECISIONS_PER_POLL; ++i) {
            const struct polaris_decision *decision = &get.decisions[i];
            if (decision->op != POLARIS_DECISION_OP_ALLOC ||
                decision->gpu_id != args->gpu_id ||
                decision->dst_vaddr != args->expected_base ||
                decision->size_bytes != POLARIS_BLOCK_SIZE) {
                continue;
            }

            struct polaris_complete_operation_arg complete = {
                .decision_id = decision->decision_id,
                .generation = decision->generation,
                .result = 0,
                .rm_control_fd = s->ctl_fd,
                .output_handle = 0,
                .output_cpu_addr = 0,
                .rm_h_client = s->h_client,
                .rm_h_memory = s->h_memory,
                .rm_backing_length = POLARIS_BLOCK_SIZE,
            };

            if (ioctl(s->polaris_fd, POLARIS_COMPLETE_OPERATION, &complete) != 0) {
                fprintf(stderr,
                        "POLARIS_COMPLETE_OPERATION executor failed: errno=%d (%s)\n",
                        errno,
                        strerror(errno));
                return NULL;
            }

            args->completed_block_id = decision->block_id;
            args->result = 0;
            printf("POLARIS completed ALLOC with RM backing: block=%llu decision=%llu hClient=0x%x hMemory=0x%x len=0x%llx\n",
                   (unsigned long long)decision->block_id,
                   (unsigned long long)decision->decision_id,
                   s->h_client,
                   s->h_memory,
                   (unsigned long long)POLARIS_BLOCK_SIZE);
            return NULL;
        }

        usleep(1000);
    }

    fprintf(stderr,
            "timed out waiting for ALLOC decision at base=0x%llx\n",
            (unsigned long long)args->expected_base);
    return NULL;
}

static int register_logical_block_mapping(struct m2_state *s,
                                          uint32_t gpu_id,
                                          uint64_t rm_client_token,
                                          uint64_t va_space_token,
                                          uint64_t base,
                                          uint64_t length,
                                          bool defer_fault,
                                          uint64_t *block_id_out)
{
    struct polaris_register_va_range_arg range = {
        .gpu_id = gpu_id,
        .base = base,
        .length = POLARIS_MANAGED_SIZE,
        .block_size = POLARIS_BLOCK_SIZE,
    };
    struct polaris_session_create_arg session = {
        .home_gpu = gpu_id,
        .beam_width = 1,
        .gpu_vas_bytes = POLARIS_MANAGED_SIZE,
        .bytes_per_token = POLARIS_BLOCK_SIZE,
        .priority = 5,
    };
    struct polaris_block_reserve_arg reserve = {
        .token_start = 0,
        .token_count = 1,
        .phase = 2,
        .flags = defer_fault ? POLARIS_RESERVE_FLAG_DEFER_FAULT : 0,
    };

    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_REGISTER_VA_RANGE,
                              &range,
                              "POLARIS_REGISTER_VA_RANGE") != 0)
        return -1;
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_SESSION_CREATE,
                              &session,
                              "POLARIS_SESSION_CREATE") != 0)
        return -1;

    s->polaris_session_id = session.session_id;
    reserve.session_id = session.session_id;
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_BLOCK_RESERVE,
                              &reserve,
                              "POLARIS_BLOCK_RESERVE") != 0)
        return -1;
    if (reserve.block_id == 0) {
        fprintf(stderr, "POLARIS_BLOCK_RESERVE returned block_id=0\n");
        return -1;
    }
    if (reserve.gpu_vaddr != base) {
        fprintf(stderr,
                "POLARIS_BLOCK_RESERVE returned gpu_vaddr=0x%llx, expected base=0x%llx\n",
                (unsigned long long)reserve.gpu_vaddr,
                (unsigned long long)base);
        return -1;
    }

    struct polaris_register_block_mapping_arg mapping = {
        .block_id = reserve.block_id,
        .gpu_id = gpu_id,
        .rm_client_token = rm_client_token,
        .va_space_token = va_space_token,
        .base = base,
        .length = length,
    };

    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_REGISTER_BLOCK_MAPPING,
                              &mapping,
                              "POLARIS_REGISTER_BLOCK_MAPPING") != 0)
        return -1;

    *block_id_out = reserve.block_id;
    printf("POLARIS registered logical block mapping: block=%llu gpu=%u client=0x%llx token=0x%llx base=0x%llx len=0x%llx defer=%s\n",
           (unsigned long long)reserve.block_id,
           gpu_id,
           (unsigned long long)rm_client_token,
           (unsigned long long)va_space_token,
           (unsigned long long)base,
           (unsigned long long)length,
           defer_fault ? "yes" : "no");
    return 0;
}

static int register_logical_block_backing(struct m2_state *s,
                                          uint32_t gpu_id,
                                          uint64_t block_id)
{
    struct polaris_register_block_backing_arg backing = {
        .block_id = block_id,
        .gpu_id = gpu_id,
        .rm_control_fd = s->ctl_fd,
        .h_client = s->h_client,
        .h_memory = s->h_memory,
        .length = POLARIS_BLOCK_SIZE,
        .offset = 0,
    };

    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_REGISTER_BLOCK_BACKING,
                              &backing,
                              "POLARIS_REGISTER_BLOCK_BACKING") != 0)
        return -1;

    printf("POLARIS registered logical block backing: block=%llu gpu=%u hClient=0x%x hMemory=0x%x len=0x%llx\n",
           (unsigned long long)block_id,
           gpu_id,
           s->h_client,
           s->h_memory,
           (unsigned long long)POLARIS_BLOCK_SIZE);
    return 0;
}

static int unmap_block_mappings(struct m2_state *s,
                                uint64_t block_id,
                                uint32_t *unmapped_count)
{
    struct polaris_unmap_block_mappings_arg unmap = {
        .block_id = block_id,
    };

    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_UNMAP_BLOCK_MAPPINGS,
                              &unmap,
                              "POLARIS_UNMAP_BLOCK_MAPPINGS") != 0)
        return -1;

    *unmapped_count = unmap.unmapped_count;
    printf("POLARIS unmapped logical block mappings: block=%llu count=%u\n",
           (unsigned long long)block_id,
           unmap.unmapped_count);
    return 0;
}

static int expect_unresident_spill_rejected(struct m2_state *s, uint64_t block_id)
{
    struct polaris_spill_block_arg spill = {
        .block_id = block_id,
    };

    if (ioctl(s->polaris_fd, POLARIS_SPILL_BLOCK, &spill) == 0) {
        fprintf(stderr,
                "POLARIS_SPILL_BLOCK unexpectedly succeeded for unresident block=%llu decision=%llu\n",
                (unsigned long long)block_id,
                (unsigned long long)spill.decision_id);
        return -1;
    }
    if (errno != ENOENT) {
        fprintf(stderr,
                "POLARIS_SPILL_BLOCK errno=%d (%s), expected ENOENT for unresident block=%llu\n",
                errno,
                strerror(errno),
                (unsigned long long)block_id);
        return -1;
    }

    printf("POLARIS spill validation rejected unresident block=%llu with ENOENT\n",
           (unsigned long long)block_id);
    return 0;
}

static int setup_polaris(struct m2_state *s,
                         uint32_t gpu_id,
                         uint64_t rm_client_token,
                         uint64_t va_space_token,
                         uint64_t base,
                         uint64_t length,
                         bool register_static_block)
{
    struct polaris_register_gpu_arg gpu = {
        .gpu_id = gpu_id,
        .total_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL,
        .budget_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL,
        .cpu_pool_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL,
        ._reserved = POLARIS_REGISTER_GPU_FLAG_TRANSIENT,
    };
    struct polaris_register_vaspace_arg va = {
        .gpu_id = gpu_id,
        .rm_client_token = rm_client_token,
        .va_space_token = va_space_token,
        .managed_base = base,
        .managed_length = length,
    };
    struct polaris_register_static_block_arg block = {
        .gpu_id = gpu_id,
        .rm_control_fd = s->ctl_fd,
        .rm_client_token = rm_client_token,
        .va_space_token = va_space_token,
        .base = base,
        .length = POLARIS_BLOCK_SIZE,
        .offset = 0,
        .h_client = s->h_client,
        .h_memory = s->h_memory,
    };

    s->polaris_fd = open("/dev/polaris", O_RDWR | O_CLOEXEC);
    if (s->polaris_fd < 0) {
        fprintf(stderr, "open /dev/polaris failed: errno=%d (%s)\n", errno, strerror(errno));
        return -1;
    }

    if (polaris_ioctl_checked(s->polaris_fd, POLARIS_REGISTER_GPU, &gpu, "POLARIS_REGISTER_GPU") != 0)
        return -1;
    if (polaris_ioctl_checked(s->polaris_fd, POLARIS_REGISTER_VASPACE, &va, "POLARIS_REGISTER_VASPACE") != 0)
        return -1;
    s->polaris_vaspace_registered = true;
    if (register_static_block) {
        if (polaris_ioctl_checked(s->polaris_fd,
                                  POLARIS_REGISTER_STATIC_BLOCK,
                                  &block,
                                  "POLARIS_REGISTER_STATIC_BLOCK") != 0)
            return -1;
    }

    printf("POLARIS setup complete: client=0x%llx token=0x%llx hClient=0x%x hMemory=0x%x static=%s\n",
           (unsigned long long)rm_client_token,
           (unsigned long long)va_space_token,
           s->h_client,
           s->h_memory,
           register_static_block ? "yes" : "no");
    return 0;
}

static int probe_uvm_dispatch_key(struct m2_state *s,
                                  uint64_t fault_address,
                                  uint32_t *gpu_id,
                                  uint64_t *rm_client_token,
                                  uint64_t *va_space_token)
{
    struct uvm_test_polaris_dispatch_fault_params fault = {
        .gpu_uuid = s->gpu_uuid,
        .fault_address = fault_address,
        .access_type = UVM_FAULT_ACCESS_TYPE_WRITE,
    };

    if (uvm_ioctl_checked(s->uvm_fd,
                          UVM_TEST_POLARIS_DISPATCH_FAULT,
                          &fault,
                          "UVM_TEST_POLARIS_DISPATCH_FAULT probe") != 0)
        return -1;
    if (fault.rmStatus != NV_OK) {
        fprintf(stderr, "UVM_TEST_POLARIS_DISPATCH_FAULT probe status=0x%x\n", fault.rmStatus);
        return -1;
    }

    *gpu_id = fault.observed_gpu_id;
    *rm_client_token = fault.observed_rm_client_token;
    *va_space_token = fault.observed_va_space_token;
    printf("UVM dispatch key: gpu_id=%u client=0x%llx token=0x%llx dry_result=%d\n",
           *gpu_id,
           (unsigned long long)*rm_client_token,
           (unsigned long long)*va_space_token,
           fault.polaris_status);
    return 0;
}

static void cleanup(struct m2_state *s,
                    uint32_t gpu_id,
                    uint64_t rm_client_token,
                    uint64_t va_space_token)
{
    if (s->polaris_session_id != 0) {
        struct polaris_session_destroy_arg destroy = {
            .session_id = s->polaris_session_id,
        };
        (void)polaris_ioctl_checked(s->polaris_fd,
                                    POLARIS_SESSION_DESTROY,
                                    &destroy,
                                    "POLARIS_SESSION_DESTROY");
        s->polaris_session_id = 0;
    }

    if (s->polaris_vaspace_registered) {
        struct polaris_unregister_vaspace_arg unva = {
            .gpu_id = gpu_id,
            .rm_client_token = rm_client_token,
            .va_space_token = va_space_token,
        };
        (void)polaris_ioctl_checked(s->polaris_fd,
                                    POLARIS_UNREGISTER_VASPACE,
                                    &unva,
                                    "POLARIS_UNREGISTER_VASPACE");
    }

    if (s->polaris_fd >= 0)
        close(s->polaris_fd);
    if (s->uvm_mm_fd >= 0)
        close(s->uvm_mm_fd);
    if (s->uvm_fd >= 0)
        close(s->uvm_fd);

    rm_free_object(s->ctl_fd, s->h_client, s->h_device, s->h_memory);
    rm_free_object(s->ctl_fd, s->h_client, s->h_device, s->h_vaspace);
    rm_free_object(s->ctl_fd, s->h_client, s->h_device, s->h_subdevice);
    rm_free_object(s->ctl_fd, s->h_client, s->h_client, s->h_device);
    rm_free_object(s->ctl_fd, s->h_client, s->h_client, s->h_client);

    if (s->gpu_fd >= 0)
        close(s->gpu_fd);
    if (s->ctl_fd >= 0)
        close(s->ctl_fd);
}

int main(int argc, char **argv)
{
    int ordinal = 0;
    uint32_t polaris_gpu_id = 0;
    uint64_t base = 0x1000000000ULL;
    uint64_t polaris_rm_client_token = 0;
    uint64_t polaris_va_space_token = 0;
    bool dispatch_fault = false;
    bool unmap_refault = false;
    bool block_unmap_refault = false;
    bool logical_backed_refault = false;
    bool complete_backed_refault = false;
    bool spill_validation = false;
    uint64_t block_id = 0;
    struct m2_state s = {
        .ctl_fd = -1,
        .gpu_fd = -1,
        .uvm_fd = -1,
        .uvm_mm_fd = -1,
        .polaris_fd = -1,
    };
    int rc = 1;

    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dispatch-fault") == 0) {
            dispatch_fault = true;
            continue;
        }
        if (strcmp(argv[i], "--unmap-refault") == 0) {
            dispatch_fault = true;
            unmap_refault = true;
            continue;
        }
        if (strcmp(argv[i], "--block-unmap-refault") == 0) {
            dispatch_fault = true;
            block_unmap_refault = true;
            continue;
        }
        if (strcmp(argv[i], "--logical-backed-refault") == 0) {
            dispatch_fault = true;
            block_unmap_refault = true;
            logical_backed_refault = true;
            continue;
        }
        if (strcmp(argv[i], "--complete-backed-refault") == 0) {
            dispatch_fault = true;
            block_unmap_refault = true;
            complete_backed_refault = true;
            continue;
        }
        if (strcmp(argv[i], "--spill-validation") == 0) {
            spill_validation = true;
            continue;
        }

        switch (positional++) {
            case 0:
                ordinal = atoi(argv[i]);
                break;
            case 1:
                polaris_gpu_id = (uint32_t)strtoul(argv[i], NULL, 0);
                break;
            case 2:
                base = strtoull(argv[i], NULL, 0);
                break;
            default:
                fprintf(stderr,
                        "usage: %s [--dispatch-fault|--unmap-refault|--block-unmap-refault|--logical-backed-refault|--complete-backed-refault|--spill-validation] [cuda_ordinal] [polaris_gpu_id] [base]\n",
                        argv[0]);
                goto out;
        }
    }

    if (get_cuda_uuid(ordinal, &s.gpu_uuid) != 0)
        goto out;
    if (setup_rm(&s, ordinal) != 0)
        goto out;
    if (setup_uvm(&s, base, POLARIS_MANAGED_SIZE, !dispatch_fault) != 0)
        goto out;
    polaris_va_space_token = s.h_vaspace;
    if (probe_uvm_dispatch_key(&s,
                               base,
                               &polaris_gpu_id,
                               &polaris_rm_client_token,
                               &polaris_va_space_token) != 0)
        goto out;
    if (setup_polaris(&s,
                      polaris_gpu_id,
                      polaris_rm_client_token,
                      polaris_va_space_token,
                      base,
                      POLARIS_MANAGED_SIZE,
                      !logical_backed_refault && !complete_backed_refault) != 0)
        goto out;
    if (complete_backed_refault) {
        pthread_t executor;
        struct completion_executor_args exec_args = {
            .state = &s,
            .gpu_id = polaris_gpu_id,
            .expected_base = base,
        };
        int thread_ret = pthread_create(&executor,
                                        NULL,
                                        complete_backing_executor,
                                        &exec_args);
        if (thread_ret != 0) {
            fprintf(stderr,
                    "pthread_create completion executor failed: %s\n",
                    strerror(thread_ret));
            goto out;
        }

        if (register_logical_block_mapping(&s,
                                           polaris_gpu_id,
                                           polaris_rm_client_token,
                                           polaris_va_space_token,
                                           base,
                                           POLARIS_BLOCK_SIZE,
                                           false,
                                           &block_id) != 0) {
            (void)pthread_join(executor, NULL);
            goto out;
        }
        if (pthread_join(executor, NULL) != 0) {
            fprintf(stderr, "pthread_join completion executor failed\n");
            goto out;
        }
        if (exec_args.result != 0 || exec_args.completed_block_id != block_id) {
            fprintf(stderr,
                    "completion executor result=%d completed_block=%llu expected_block=%llu\n",
                    exec_args.result,
                    (unsigned long long)exec_args.completed_block_id,
                    (unsigned long long)block_id);
            goto out;
        }
    }
    if ((block_unmap_refault || spill_validation) && !complete_backed_refault) {
        if (register_logical_block_mapping(&s,
                                           polaris_gpu_id,
                                           polaris_rm_client_token,
                                           polaris_va_space_token,
                                           base,
                                           POLARIS_BLOCK_SIZE,
                                           true,
                                           &block_id) != 0)
            goto out;
    }
    if (logical_backed_refault) {
        if (register_logical_block_backing(&s, polaris_gpu_id, block_id) != 0)
            goto out;
    }
    if (spill_validation) {
        if (expect_unresident_spill_rejected(&s, block_id) != 0)
            goto out;
    }
    if (dispatch_fault) {
        if (dispatch_test_fault(&s, base) != 0)
            goto out;
        if (block_unmap_refault) {
            uint32_t unmapped_count = 0;
            if (unmap_block_mappings(&s, block_id, &unmapped_count) != 0)
                goto out;
            if (unmapped_count != 1) {
                fprintf(stderr,
                        "POLARIS_UNMAP_BLOCK_MAPPINGS unmapped_count=%u, expected 1\n",
                        unmapped_count);
                goto out;
            }
            if (dispatch_test_fault(&s, base) != 0)
                goto out;
        } else if (unmap_refault) {
            if (unmap_static_block(&s,
                                   polaris_gpu_id,
                                   polaris_rm_client_token,
                                   polaris_va_space_token,
                                   base,
                                   POLARIS_BLOCK_SIZE) != 0)
                goto out;
            if (dispatch_test_fault(&s, base) != 0)
                goto out;
        }
    }

    if (complete_backed_refault)
        puts("M3 Polaris completion-backed block refault test passed.");
    else if (logical_backed_refault)
        puts("M3 Polaris logical-backed block refault test passed.");
    else if (block_unmap_refault)
        puts("M3 Polaris block unmap/refault test passed.");
    else if (spill_validation)
        puts("M3 Polaris spill ioctl validation passed.");
    else if (unmap_refault)
        puts("M2 Polaris unmap/refault test passed.");
    else
        puts(dispatch_fault ? "M2 Polaris fault-dispatch test passed." : "M2 static-block setup passed.");
    if (!dispatch_fault)
        puts("Next step: add an RM GPFIFO channel bound to this VA-space and submit a write to the unmapped block.");
    rc = 0;

out:
    cleanup(&s, polaris_gpu_id, polaris_rm_client_token, polaris_va_space_token);
    return rc;
}
