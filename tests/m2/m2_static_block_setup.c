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
#include <sys/wait.h>
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

struct polaris_session_branch_arg {
    uint64_t parent_session_id;
    uint64_t child_session_id;
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

struct polaris_probe_rm_phys_arg {
    uint64_t block_id;
    uint64_t offset;
    uint64_t length;
    uint64_t page_size;
    uint64_t phys_addr_count;
    uint64_t first_phys_addr;
    uint64_t last_phys_addr;
    uint64_t flags;
    uint64_t _reserved[4];
};

struct polaris_probe_rm_copy_arg {
    uint64_t block_id;
    uint64_t offset;
    uint64_t length;
    uint64_t pattern_seed;
    uint64_t page_size;
    uint64_t phys_addr_count;
    uint64_t first_phys_addr;
    uint64_t last_phys_addr;
    uint64_t flags;
    uint64_t bytes_checked;
    uint64_t first_mismatch_offset;
    uint64_t expected_byte;
    uint64_t actual_byte;
    uint64_t _reserved[4];
};

struct polaris_rm_copy_arg {
    uint64_t block_id;
    uint64_t offset;
    uint64_t length;
    uint64_t user_cpu_addr;
    uint32_t direction;
    uint32_t _pad;
    int32_t rm_control_fd;
    uint32_t rm_h_client;
    uint32_t rm_h_memory;
    uint32_t _pad2;
    uint64_t page_size;
    uint64_t phys_addr_count;
    uint64_t first_phys_addr;
    uint64_t last_phys_addr;
    uint64_t flags;
    uint64_t bytes_copied;
    uint64_t _reserved[2];
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
#define POLARIS_SESSION_BRANCH _IOWR(POLARIS_IOCTL_MAGIC, 0x06, struct polaris_session_branch_arg)
#define POLARIS_BLOCK_RESERVE _IOWR(POLARIS_IOCTL_MAGIC, 0x07, struct polaris_block_reserve_arg)
#define POLARIS_BLOCK_RELEASE _IOW(POLARIS_IOCTL_MAGIC, 0x08, struct polaris_block_release_arg)
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
#define POLARIS_PROBE_RM_PHYS _IOWR(POLARIS_IOCTL_MAGIC, 0x18, struct polaris_probe_rm_phys_arg)
#define POLARIS_PROBE_RM_COPY _IOWR(POLARIS_IOCTL_MAGIC, 0x19, struct polaris_probe_rm_copy_arg)
#define POLARIS_RM_COPY _IOWR(POLARIS_IOCTL_MAGIC, 0x1a, struct polaris_rm_copy_arg)
#define POLARIS_REGISTER_GPU_FLAG_TRANSIENT (1U << 0)
#define POLARIS_RESERVE_FLAG_OVERWRITE (1U << 0)
#define POLARIS_RESERVE_FLAG_DEFER_FAULT (1U << 4)
#define POLARIS_RELEASE_FLAG_CALLER_OWNS_BACKING (1U << 1)

#define POLARIS_RM_PHYS_FLAG_CONTIGUOUS (1ULL << 0)
#define POLARIS_RM_PHYS_FLAG_SYSMEM (1ULL << 1)
#define POLARIS_RM_PHYS_FLAG_EGM (1ULL << 2)
#define POLARIS_RM_PHYS_FLAG_FABRICMEM (1ULL << 3)
#define POLARIS_DECISION_FLAG_SOURCE_BLOCK_ID_VALID (1ULL << 0)
#define POLARIS_RM_COPY_NO_MISMATCH UINT64_MAX
#define POLARIS_RM_COPY_TO_CPU 0U
#define POLARIS_RM_COPY_FROM_CPU 1U

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

struct nv_os33_parameters {
    NvHandle h_client;
    NvHandle h_device;
    NvHandle h_memory;
    NvU64 offset;
    NvU64 length;
    NvP64 p_linear_address;
    NvU32 status;
    NvU32 flags;
};

struct nv_os34_parameters {
    NvHandle h_client;
    NvHandle h_device;
    NvHandle h_memory;
    NvP64 p_linear_address;
    NvU32 status;
    NvU32 flags;
};

struct nv_os33_parameters_with_fd {
    struct nv_os33_parameters params;
    int fd;
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
    uint32_t polaris_block_token_start;
    uint32_t polaris_block_token_count;
    bool polaris_block_reserved;
};

struct completion_executor_args {
    struct m2_state *state;
    uint32_t gpu_id;
    uint64_t expected_base;
    uint64_t completed_block_id;
    int result;
};

struct rm_spill_reload_args {
    struct m2_state *state;
    uint32_t gpu_id;
    uint64_t block_id;
    uint64_t cpu_addr;
    NvHandle new_h_memory;
    int result;
};

struct rm_cow_args {
    struct m2_state *state;
    uint64_t parent_block_id;
    uint64_t child_block_id;
    uint64_t child_vaddr;
    NvHandle child_h_memory;
    int result;
};

static void cleanup(struct m2_state *s,
                    uint32_t gpu_id,
                    uint64_t rm_client_token,
                    uint64_t va_space_token);

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

static void print_cuda_result(const char *what, CUresult res)
{
    const char *name = NULL;
    const char *str = NULL;

    (void)cuGetErrorName(res, &name);
    (void)cuGetErrorString(res, &str);
    fprintf(stderr,
            "%s failed: %d%s%s%s%s%s\n",
            what,
            res,
            name ? " (" : "",
            name ? name : "",
            name ? ")" : "",
            str ? ": " : "",
            str ? str : "");
}

static int run_cuda_copy_probe(struct m2_state *s, int ordinal, uint64_t base, size_t length)
{
    CUdevice dev;
    CUcontext ctx = NULL;
    CUresult res;
    unsigned char *host_in = NULL;
    unsigned char *host_out = NULL;
    int ret = -1;

    host_in = malloc(length);
    host_out = malloc(length);
    if (!host_in || !host_out) {
        fprintf(stderr, "CUDA copy probe host allocation failed for %zu bytes\n", length);
        goto out;
    }

    for (size_t i = 0; i < length; ++i) {
        host_in[i] = (unsigned char)((i * 131U + 17U) & 0xffU);
        host_out[i] = 0;
    }

    res = cuDeviceGet(&dev, ordinal);
    if (res != CUDA_SUCCESS) {
        print_cuda_result("cuDeviceGet", res);
        goto out;
    }

    res = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (res != CUDA_SUCCESS) {
        print_cuda_result("cuDevicePrimaryCtxRetain", res);
        goto out;
    }

    res = cuCtxSetCurrent(ctx);
    if (res != CUDA_SUCCESS) {
        print_cuda_result("cuCtxSetCurrent", res);
        goto out_release;
    }

    printf("CUDA copy probe: copying %zu bytes through current CUDA primary context to external VA 0x%llx\n",
           length,
           (unsigned long long)base);

    res = cuMemcpyHtoD_v2((CUdeviceptr)base, host_in, length);
    if (res != CUDA_SUCCESS) {
        print_cuda_result("cuMemcpyHtoD_v2 external VA", res);
        puts("CUDA copy probe result: current CUDA context cannot write the UVM external VA.");
        ret = 0;
        goto out_release;
    }

    res = cuMemcpyDtoH_v2(host_out, (CUdeviceptr)base, length);
    if (res != CUDA_SUCCESS) {
        print_cuda_result("cuMemcpyDtoH_v2 external VA", res);
        puts("CUDA copy probe result: HtoD succeeded, but current CUDA context cannot read the UVM external VA.");
        ret = 0;
        goto out_release;
    }

    if (memcmp(host_in, host_out, length) != 0) {
        fprintf(stderr, "CUDA copy probe data mismatch after successful HtoD/DtoH round-trip\n");
        goto out_release;
    }

    puts("CUDA copy probe result: CUDA copy APIs can round-trip the UVM external VA in this process.");
    ret = 0;

out_release:
    (void)cuCtxSetCurrent(NULL);
    if (ctx)
        (void)cuDevicePrimaryCtxRelease(dev);
out:
    free(host_out);
    free(host_in);
    (void)s;
    return ret;
}

static int run_cuda_copy_probe_child(int ordinal, uint64_t base)
{
    struct m2_state s = {
        .ctl_fd = -1,
        .gpu_fd = -1,
        .uvm_fd = -1,
        .uvm_mm_fd = -1,
        .polaris_fd = -1,
    };
    int ret = 1;

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    if (get_cuda_uuid(ordinal, &s.gpu_uuid) != 0)
        goto out;
    if (setup_rm(&s, ordinal) != 0)
        goto out;
    if (setup_uvm(&s, base, POLARIS_MANAGED_SIZE, true) != 0)
        goto out;

    ret = run_cuda_copy_probe(&s, ordinal, base, POLARIS_BLOCK_SIZE) == 0 ? 0 : 1;

out:
    cleanup(&s, 0, 0, 0);
    return ret;
}

static int run_cuda_copy_probe_isolated(int ordinal, uint64_t base)
{
    pid_t pid = fork();
    int status = 0;

    if (pid < 0) {
        fprintf(stderr, "fork for CUDA copy probe failed: errno=%d (%s)\n", errno, strerror(errno));
        return -1;
    }

    if (pid == 0) {
        int ret = run_cuda_copy_probe_child(ordinal, base);
        fflush(stdout);
        fflush(stderr);
        _exit(ret == 0 ? 0 : 1);
    }

    if (waitpid(pid, &status, 0) < 0) {
        fprintf(stderr, "waitpid for CUDA copy probe failed: errno=%d (%s)\n", errno, strerror(errno));
        return -1;
    }

    if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        printf("CUDA copy probe result: child terminated by signal %d (%s) while running the isolated external-VA CUDA copy probe.\n",
               sig,
               strsignal(sig));
        return 0;
    }

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "CUDA copy probe child exited inconclusively: status=0x%x\n", status);
        return -1;
    }

    return 0;
}

static int rm_map_memory_cpu(struct m2_state *s, uint64_t length, void **mapped_out)
{
    struct nv_os33_parameters_with_fd map = {
        .params = {
            .h_client = s->h_client,
            .h_device = s->h_device,
            .h_memory = s->h_memory,
            .offset = 0,
            .length = length,
            .p_linear_address = 0,
            .flags = 0,
        },
        .fd = s->gpu_fd,
    };

    if (nv_ioctl_checked(s->ctl_fd,
                         NV_ESC_RM_MAP_MEMORY,
                         &map,
                         sizeof(map),
                         "RM_MAP_MEMORY CPU") != 0)
        return -1;
    if (map.params.status != NV_OK) {
        fprintf(stderr, "RM_MAP_MEMORY CPU status=0x%x\n", map.params.status);
        return -1;
    }
    if (map.params.p_linear_address == 0) {
        fprintf(stderr, "RM_MAP_MEMORY CPU returned null address\n");
        return -1;
    }

    *mapped_out = (void *)(uintptr_t)map.params.p_linear_address;
    printf("RM_MAP_MEMORY CPU mapped hMemory=0x%x len=0x%llx at %p\n",
           s->h_memory,
           (unsigned long long)length,
           *mapped_out);
    return 0;
}

static void rm_unmap_memory_cpu(struct m2_state *s, void *mapped)
{
    struct nv_os34_parameters unmap = {
        .h_client = s->h_client,
        .h_device = s->h_device,
        .h_memory = s->h_memory,
        .p_linear_address = (NvP64)(uintptr_t)mapped,
        .flags = 0,
    };

    if (!mapped)
        return;

    if (nv_ioctl_checked(s->ctl_fd,
                         NV_ESC_RM_UNMAP_MEMORY,
                         &unmap,
                         sizeof(unmap),
                         "RM_UNMAP_MEMORY CPU") != 0)
        return;
    if (unmap.status != NV_OK)
        fprintf(stderr, "RM_UNMAP_MEMORY CPU status=0x%x\n", unmap.status);
}

static int run_rm_cpu_map_probe_child(int ordinal)
{
    struct m2_state s = {
        .ctl_fd = -1,
        .gpu_fd = -1,
        .uvm_fd = -1,
        .uvm_mm_fd = -1,
        .polaris_fd = -1,
    };
    unsigned char *mapped = NULL;
    unsigned char *verify = NULL;
    const size_t probe_len = 4096;
    int ret = -1;

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    if (get_cuda_uuid(ordinal, &s.gpu_uuid) != 0)
        goto out;
    if (setup_rm(&s, ordinal) != 0)
        goto out;

    if (rm_map_memory_cpu(&s, POLARIS_BLOCK_SIZE, (void **)&mapped) != 0)
        goto out;

    verify = malloc(probe_len);
    if (!verify) {
        fprintf(stderr, "RM CPU map probe host verify allocation failed\n");
        goto out;
    }

    for (size_t i = 0; i < probe_len; ++i)
        mapped[i] = (unsigned char)((i * 29U + 3U) & 0xffU);
    memcpy(verify, mapped, probe_len);
    for (size_t i = 0; i < probe_len; ++i) {
        unsigned char expected = (unsigned char)((i * 29U + 3U) & 0xffU);
        if (verify[i] != expected) {
            fprintf(stderr,
                    "RM CPU map probe mismatch at byte %zu: got=0x%02x expected=0x%02x\n",
                    i,
                    verify[i],
                    expected);
            goto out;
        }
    }

    puts("RM CPU map probe result: RM vidmem can be CPU-mapped and byte-round-tripped through the daemon process.");
    ret = 0;

out:
    free(verify);
    if (mapped)
        rm_unmap_memory_cpu(&s, mapped);
    cleanup(&s, 0, 0, 0);
    return ret;
}

static int run_rm_cpu_map_probe_isolated(int ordinal)
{
    pid_t pid = fork();
    int status = 0;

    if (pid < 0) {
        fprintf(stderr, "fork for RM CPU map probe failed: errno=%d (%s)\n", errno, strerror(errno));
        return -1;
    }

    if (pid == 0) {
        int ret = run_rm_cpu_map_probe_child(ordinal);
        fflush(stdout);
        fflush(stderr);
        _exit(ret == 0 ? 0 : 1);
    }

    if (waitpid(pid, &status, 0) < 0) {
        fprintf(stderr, "waitpid for RM CPU map probe failed: errno=%d (%s)\n", errno, strerror(errno));
        return -1;
    }

    if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        printf("RM CPU map probe result: child terminated by signal %d (%s) while touching the RM CPU mapping.\n",
               sig,
               strsignal(sig));
        return 0;
    }

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "RM CPU map probe child exited inconclusively: status=0x%x\n", status);
        return -1;
    }

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
                                          uint64_t managed_length,
                                          uint64_t length,
                                          bool defer_fault,
                                          uint64_t *block_id_out)
{
    struct polaris_register_va_range_arg range = {
        .gpu_id = gpu_id,
        .base = base,
        .length = managed_length,
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
    s->polaris_block_token_start = reserve.token_start;
    s->polaris_block_token_count = reserve.token_count;
    s->polaris_block_reserved = true;
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

static int probe_rm_phys(struct m2_state *s, uint64_t block_id)
{
    struct polaris_probe_rm_phys_arg probe = {
        .block_id = block_id,
        .offset = 0,
        .length = POLARIS_BLOCK_SIZE,
    };

    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_PROBE_RM_PHYS,
                              &probe,
                              "POLARIS_PROBE_RM_PHYS") != 0)
        return -1;

    printf("POLARIS RM phys probe: block=%llu offset=0x%llx len=0x%llx page=0x%llx count=%llu first=0x%llx last=0x%llx flags=0x%llx [%s%s%s%s]\n",
           (unsigned long long)probe.block_id,
           (unsigned long long)probe.offset,
           (unsigned long long)probe.length,
           (unsigned long long)probe.page_size,
           (unsigned long long)probe.phys_addr_count,
           (unsigned long long)probe.first_phys_addr,
           (unsigned long long)probe.last_phys_addr,
           (unsigned long long)probe.flags,
           (probe.flags & POLARIS_RM_PHYS_FLAG_CONTIGUOUS) ? "contiguous" : "noncontiguous",
           (probe.flags & POLARIS_RM_PHYS_FLAG_SYSMEM) ? "|sysmem" : "|vidmem",
           (probe.flags & POLARIS_RM_PHYS_FLAG_EGM) ? "|egm" : "",
           (probe.flags & POLARIS_RM_PHYS_FLAG_FABRICMEM) ? "|fabricmem" : "");

    if (probe.length != POLARIS_BLOCK_SIZE) {
        fprintf(stderr,
                "POLARIS_PROBE_RM_PHYS returned len=0x%llx, expected 0x%llx\n",
                (unsigned long long)probe.length,
                (unsigned long long)POLARIS_BLOCK_SIZE);
        return -1;
    }
    if (probe.page_size == 0 || probe.phys_addr_count == 0) {
        fprintf(stderr,
                "POLARIS_PROBE_RM_PHYS returned empty geometry: page=0x%llx count=%llu\n",
                (unsigned long long)probe.page_size,
                (unsigned long long)probe.phys_addr_count);
        return -1;
    }
    if ((probe.offset % probe.page_size) != 0 || (probe.length % probe.page_size) != 0) {
        fprintf(stderr,
                "POLARIS_PROBE_RM_PHYS returned unaligned range: offset=0x%llx len=0x%llx page=0x%llx\n",
                (unsigned long long)probe.offset,
                (unsigned long long)probe.length,
                (unsigned long long)probe.page_size);
        return -1;
    }
    if (probe.phys_addr_count != probe.length / probe.page_size) {
        fprintf(stderr,
                "POLARIS_PROBE_RM_PHYS count=%llu, expected %llu for len/page geometry\n",
                (unsigned long long)probe.phys_addr_count,
                (unsigned long long)(probe.length / probe.page_size));
        return -1;
    }
    if (probe.flags & POLARIS_RM_PHYS_FLAG_SYSMEM) {
        fprintf(stderr,
                "POLARIS_PROBE_RM_PHYS warning: RM reported sysmem for the diagnostic allocation; this does not prove the vidmem copy path\n");
    }

    return 0;
}

static int probe_rm_copy(struct m2_state *s, uint64_t block_id)
{
    struct polaris_probe_rm_copy_arg probe = {
        .block_id = block_id,
        .offset = 0,
        .length = POLARIS_BLOCK_SIZE,
        .pattern_seed = 0x504f4c4152495343ULL,
        .first_mismatch_offset = POLARIS_RM_COPY_NO_MISMATCH,
    };

    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_PROBE_RM_COPY,
                              &probe,
                              "POLARIS_PROBE_RM_COPY") != 0)
        return -1;

    printf("POLARIS RM copy probe: block=%llu offset=0x%llx len=0x%llx bytes=0x%llx page=0x%llx count=%llu first=0x%llx last=0x%llx flags=0x%llx mismatch=0x%llx expected=0x%llx actual=0x%llx\n",
           (unsigned long long)probe.block_id,
           (unsigned long long)probe.offset,
           (unsigned long long)probe.length,
           (unsigned long long)probe.bytes_checked,
           (unsigned long long)probe.page_size,
           (unsigned long long)probe.phys_addr_count,
           (unsigned long long)probe.first_phys_addr,
           (unsigned long long)probe.last_phys_addr,
           (unsigned long long)probe.flags,
           (unsigned long long)probe.first_mismatch_offset,
           (unsigned long long)probe.expected_byte,
           (unsigned long long)probe.actual_byte);

    if (probe.length != POLARIS_BLOCK_SIZE || probe.bytes_checked != POLARIS_BLOCK_SIZE) {
        fprintf(stderr,
                "POLARIS_PROBE_RM_COPY returned len=0x%llx bytes=0x%llx, expected 0x%llx\n",
                (unsigned long long)probe.length,
                (unsigned long long)probe.bytes_checked,
                (unsigned long long)POLARIS_BLOCK_SIZE);
        return -1;
    }
    if (probe.page_size == 0 || probe.phys_addr_count == 0) {
        fprintf(stderr,
                "POLARIS_PROBE_RM_COPY returned empty geometry: page=0x%llx count=%llu\n",
                (unsigned long long)probe.page_size,
                (unsigned long long)probe.phys_addr_count);
        return -1;
    }
    if ((probe.flags & POLARIS_RM_PHYS_FLAG_CONTIGUOUS) == 0 ||
        (probe.flags & POLARIS_RM_PHYS_FLAG_SYSMEM) != 0) {
        fprintf(stderr,
                "POLARIS_PROBE_RM_COPY expected contiguous vidmem flags, got 0x%llx\n",
                (unsigned long long)probe.flags);
        return -1;
    }
    if (probe.first_mismatch_offset != POLARIS_RM_COPY_NO_MISMATCH) {
        fprintf(stderr,
                "POLARIS_PROBE_RM_COPY mismatch at 0x%llx: expected=0x%llx actual=0x%llx\n",
                (unsigned long long)probe.first_mismatch_offset,
                (unsigned long long)probe.expected_byte,
                (unsigned long long)probe.actual_byte);
        return -1;
    }

    return 0;
}

static uint8_t rm_copy_roundtrip_pattern(uint64_t seed, uint64_t offset)
{
    uint64_t x = seed + offset * 0x9e3779b97f4a7c15ULL;

    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 29;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 32;
    return (uint8_t)x;
}

static int rm_copy_roundtrip(struct m2_state *s, uint64_t block_id)
{
    uint8_t *src = NULL;
    uint8_t *dst = NULL;
    const uint64_t seed = 0x504f4c4152495355ULL;
    int rc = -1;

    if (posix_memalign((void **)&src, 4096, POLARIS_BLOCK_SIZE) != 0) {
        fprintf(stderr, "posix_memalign src failed\n");
        goto out;
    }
    if (posix_memalign((void **)&dst, 4096, POLARIS_BLOCK_SIZE) != 0) {
        fprintf(stderr, "posix_memalign dst failed\n");
        goto out;
    }

    for (uint64_t i = 0; i < POLARIS_BLOCK_SIZE; ++i) {
        src[i] = rm_copy_roundtrip_pattern(seed, i);
        dst[i] = 0;
    }

    struct polaris_rm_copy_arg to_rm = {
        .block_id = block_id,
        .offset = 0,
        .length = POLARIS_BLOCK_SIZE,
        .user_cpu_addr = (uint64_t)(uintptr_t)src,
        .direction = POLARIS_RM_COPY_FROM_CPU,
    };

    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_RM_COPY,
                              &to_rm,
                              "POLARIS_RM_COPY FROM_CPU") != 0)
        goto out;

    printf("POLARIS RM copy user->rm: block=%llu len=0x%llx bytes=0x%llx page=0x%llx count=%llu first=0x%llx last=0x%llx flags=0x%llx\n",
           (unsigned long long)to_rm.block_id,
           (unsigned long long)to_rm.length,
           (unsigned long long)to_rm.bytes_copied,
           (unsigned long long)to_rm.page_size,
           (unsigned long long)to_rm.phys_addr_count,
           (unsigned long long)to_rm.first_phys_addr,
           (unsigned long long)to_rm.last_phys_addr,
           (unsigned long long)to_rm.flags);

    if (to_rm.length != POLARIS_BLOCK_SIZE || to_rm.bytes_copied != POLARIS_BLOCK_SIZE) {
        fprintf(stderr,
                "POLARIS_RM_COPY FROM_CPU returned len=0x%llx bytes=0x%llx, expected 0x%llx\n",
                (unsigned long long)to_rm.length,
                (unsigned long long)to_rm.bytes_copied,
                (unsigned long long)POLARIS_BLOCK_SIZE);
        goto out;
    }
    if ((to_rm.flags & POLARIS_RM_PHYS_FLAG_CONTIGUOUS) == 0 ||
        (to_rm.flags & POLARIS_RM_PHYS_FLAG_SYSMEM) != 0) {
        fprintf(stderr,
                "POLARIS_RM_COPY FROM_CPU expected contiguous vidmem flags, got 0x%llx\n",
                (unsigned long long)to_rm.flags);
        goto out;
    }

    struct polaris_rm_copy_arg to_cpu = {
        .block_id = block_id,
        .offset = 0,
        .length = POLARIS_BLOCK_SIZE,
        .user_cpu_addr = (uint64_t)(uintptr_t)dst,
        .direction = POLARIS_RM_COPY_TO_CPU,
    };

    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_RM_COPY,
                              &to_cpu,
                              "POLARIS_RM_COPY TO_CPU") != 0)
        goto out;

    printf("POLARIS RM copy rm->user: block=%llu len=0x%llx bytes=0x%llx page=0x%llx count=%llu first=0x%llx last=0x%llx flags=0x%llx\n",
           (unsigned long long)to_cpu.block_id,
           (unsigned long long)to_cpu.length,
           (unsigned long long)to_cpu.bytes_copied,
           (unsigned long long)to_cpu.page_size,
           (unsigned long long)to_cpu.phys_addr_count,
           (unsigned long long)to_cpu.first_phys_addr,
           (unsigned long long)to_cpu.last_phys_addr,
           (unsigned long long)to_cpu.flags);

    if (to_cpu.length != POLARIS_BLOCK_SIZE || to_cpu.bytes_copied != POLARIS_BLOCK_SIZE) {
        fprintf(stderr,
                "POLARIS_RM_COPY TO_CPU returned len=0x%llx bytes=0x%llx, expected 0x%llx\n",
                (unsigned long long)to_cpu.length,
                (unsigned long long)to_cpu.bytes_copied,
                (unsigned long long)POLARIS_BLOCK_SIZE);
        goto out;
    }
    if (to_cpu.page_size != to_rm.page_size ||
        to_cpu.phys_addr_count != to_rm.phys_addr_count ||
        to_cpu.first_phys_addr != to_rm.first_phys_addr ||
        to_cpu.last_phys_addr != to_rm.last_phys_addr ||
        to_cpu.flags != to_rm.flags) {
        fprintf(stderr, "POLARIS_RM_COPY geometry changed between write/read\n");
        goto out;
    }

    for (uint64_t i = 0; i < POLARIS_BLOCK_SIZE; ++i) {
        if (dst[i] != src[i]) {
            fprintf(stderr,
                    "POLARIS_RM_COPY roundtrip mismatch at 0x%llx: expected=0x%x actual=0x%x\n",
                    (unsigned long long)i,
                    src[i],
                    dst[i]);
            goto out;
        }
    }

    rc = 0;

out:
    free(dst);
    free(src);
    return rc;
}

static int allocate_rm_memory(struct m2_state *s, const char *what, uint64_t size, NvHandle *h_memory_out)
{
    NV_MEMORY_ALLOCATION_PARAMS memory_params = {
        .size = size,
        .owner = s->h_client,
        .type = NVOS32_TYPE_IMAGE,
        .attr = (NVOS32_ATTR_LOCATION_VIDMEM << 25),
    };

    *h_memory_out = 0;
    if (rm_alloc(s->ctl_fd,
                 s->h_client,
                 s->h_device,
                 h_memory_out,
                 NV01_MEMORY_LOCAL_USER,
                 &memory_params,
                 sizeof(memory_params),
                 what) != 0)
        return -1;

    printf("%s size=0x%llx hMemory=0x%x\n",
           what,
           (unsigned long long)memory_params.size,
           *h_memory_out);
    return 0;
}

static int complete_decision(struct m2_state *s,
                             const struct polaris_decision *decision,
                             int32_t result,
                             uint64_t output_cpu_addr,
                             NvHandle h_memory)
{
    struct polaris_complete_operation_arg complete = {
        .decision_id = decision->decision_id,
        .generation = decision->generation,
        .result = result,
        .rm_control_fd = h_memory ? s->ctl_fd : 0,
        .output_handle = 0,
        .output_cpu_addr = output_cpu_addr,
        .rm_h_client = h_memory ? s->h_client : 0,
        .rm_h_memory = h_memory,
        .rm_backing_length = h_memory ? POLARIS_BLOCK_SIZE : 0,
    };

    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_COMPLETE_OPERATION,
                              &complete,
                              "POLARIS_COMPLETE_OPERATION") != 0)
        return -1;
    return 0;
}

static int wait_for_one_decision(struct m2_state *s,
                                 uint32_t expected_op,
                                 uint64_t expected_block_id,
                                 struct polaris_decision *decision_out)
{
    for (int attempt = 0; attempt < 5000; ++attempt) {
        struct polaris_get_decision_arg get = {0};

        if (polaris_ioctl_checked(s->polaris_fd,
                                  POLARIS_GET_DECISION,
                                  &get,
                                  "POLARIS_GET_DECISION") != 0)
            return -1;

        for (uint32_t i = 0; i < get.count && i < POLARIS_MAX_DECISIONS_PER_POLL; ++i) {
            const struct polaris_decision *decision = &get.decisions[i];
            if (decision->op == expected_op && decision->block_id == expected_block_id) {
                *decision_out = *decision;
                return 0;
            }

            fprintf(stderr,
                    "unexpected decision while waiting for op=%u block=%llu: op=%u block=%llu decision=%llu\n",
                    expected_op,
                    (unsigned long long)expected_block_id,
                    decision->op,
                    (unsigned long long)decision->block_id,
                    (unsigned long long)decision->decision_id);
            if (complete_decision(s, decision, -EINVAL, 0, 0) != 0)
                return -1;
        }

        usleep(1000);
    }

    fprintf(stderr,
            "timed out waiting for decision op=%u block=%llu\n",
            expected_op,
            (unsigned long long)expected_block_id);
    return -1;
}

static int execute_rm_offload_decision(struct m2_state *s,
                                       const struct polaris_decision *decision,
                                       uint64_t *cpu_addr_out)
{
    uint8_t *cpu = NULL;
    struct polaris_rm_copy_arg copy = {0};

    if (posix_memalign((void **)&cpu, 4096, POLARIS_BLOCK_SIZE) != 0) {
        fprintf(stderr, "posix_memalign offload CPU buffer failed\n");
        return -1;
    }

    copy.block_id = decision->block_id;
    copy.length = POLARIS_BLOCK_SIZE;
    copy.user_cpu_addr = (uint64_t)(uintptr_t)cpu;
    copy.direction = POLARIS_RM_COPY_TO_CPU;
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_RM_COPY,
                              &copy,
                              "POLARIS_RM_COPY OFFLOAD TO_CPU") != 0) {
        free(cpu);
        return -1;
    }
    if (copy.bytes_copied != POLARIS_BLOCK_SIZE) {
        fprintf(stderr,
                "OFFLOAD RM copy bytes=0x%llx expected=0x%llx\n",
                (unsigned long long)copy.bytes_copied,
                (unsigned long long)POLARIS_BLOCK_SIZE);
        free(cpu);
        return -1;
    }

    if (complete_decision(s, decision, 0, (uint64_t)(uintptr_t)cpu, 0) != 0) {
        free(cpu);
        return -1;
    }

    *cpu_addr_out = (uint64_t)(uintptr_t)cpu;
    printf("POLARIS RM OFFLOAD completed: block=%llu cpu=0x%llx bytes=0x%llx page=0x%llx flags=0x%llx\n",
           (unsigned long long)decision->block_id,
           (unsigned long long)*cpu_addr_out,
           (unsigned long long)copy.bytes_copied,
           (unsigned long long)copy.page_size,
           (unsigned long long)copy.flags);
    return 0;
}

static int execute_rm_reload_decision(struct m2_state *s,
                                      const struct polaris_decision *decision,
                                      NvHandle *new_h_memory_out)
{
    NvHandle new_h_memory = 0;
    struct polaris_rm_copy_arg copy = {0};

    if (decision->cpu_addr == 0) {
        fprintf(stderr, "RELOAD decision missing cpu_addr for block=%llu\n",
                (unsigned long long)decision->block_id);
        return -1;
    }

    if (allocate_rm_memory(s, "RM_ALLOC reload memory", POLARIS_BLOCK_SIZE, &new_h_memory) != 0)
        return -1;

    copy.block_id = decision->block_id;
    copy.length = POLARIS_BLOCK_SIZE;
    copy.user_cpu_addr = decision->cpu_addr;
    copy.direction = POLARIS_RM_COPY_FROM_CPU;
    copy.rm_control_fd = s->ctl_fd;
    copy.rm_h_client = s->h_client;
    copy.rm_h_memory = new_h_memory;
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_RM_COPY,
                              &copy,
                              "POLARIS_RM_COPY RELOAD FROM_CPU") != 0) {
        rm_free_object(s->ctl_fd, s->h_client, s->h_device, new_h_memory);
        return -1;
    }
    if (copy.bytes_copied != POLARIS_BLOCK_SIZE) {
        fprintf(stderr,
                "RELOAD RM copy bytes=0x%llx expected=0x%llx\n",
                (unsigned long long)copy.bytes_copied,
                (unsigned long long)POLARIS_BLOCK_SIZE);
        rm_free_object(s->ctl_fd, s->h_client, s->h_device, new_h_memory);
        return -1;
    }

    if (complete_decision(s, decision, 0, 0, new_h_memory) != 0) {
        rm_free_object(s->ctl_fd, s->h_client, s->h_device, new_h_memory);
        return -1;
    }

    free((void *)(uintptr_t)decision->cpu_addr);
    *new_h_memory_out = new_h_memory;
    printf("POLARIS RM RELOAD completed: block=%llu hMemory=0x%x bytes=0x%llx page=0x%llx flags=0x%llx\n",
           (unsigned long long)decision->block_id,
           new_h_memory,
           (unsigned long long)copy.bytes_copied,
           (unsigned long long)copy.page_size,
           (unsigned long long)copy.flags);
    return 0;
}

static void *rm_reload_executor(void *arg)
{
    struct rm_spill_reload_args *args = arg;
    struct polaris_decision reload = {0};

    args->result = -1;
    if (wait_for_one_decision(args->state,
                              POLARIS_DECISION_OP_RELOAD,
                              args->block_id,
                              &reload) != 0)
        return NULL;
    if (reload.cpu_addr != args->cpu_addr) {
        fprintf(stderr,
                "RELOAD cpu_addr=0x%llx expected=0x%llx\n",
                (unsigned long long)reload.cpu_addr,
                (unsigned long long)args->cpu_addr);
        return NULL;
    }
    if (execute_rm_reload_decision(args->state, &reload, &args->new_h_memory) != 0)
        return NULL;
    args->result = 0;
    return NULL;
}

static int execute_rm_cow_decision(struct m2_state *s,
                                   const struct polaris_decision *decision,
                                   uint64_t expected_parent_block_id,
                                   NvHandle *new_h_memory_out)
{
    uint8_t *scratch = NULL;
    NvHandle new_h_memory = 0;
    struct polaris_rm_copy_arg copy = {0};

    if (decision->_reserved[1] != POLARIS_DECISION_FLAG_SOURCE_BLOCK_ID_VALID ||
        decision->_reserved[0] != expected_parent_block_id) {
        fprintf(stderr,
                "COW decision source block metadata invalid: reserved0=%llu reserved1=0x%llx expected_block=%llu\n",
                (unsigned long long)decision->_reserved[0],
                (unsigned long long)decision->_reserved[1],
                (unsigned long long)expected_parent_block_id);
        return -1;
    }
    if (decision->dst_vaddr == 0) {
        fprintf(stderr, "COW decision missing destination VA\n");
        return -1;
    }

    if (posix_memalign((void **)&scratch, 4096, POLARIS_BLOCK_SIZE) != 0) {
        fprintf(stderr, "posix_memalign COW scratch failed\n");
        return -1;
    }

    copy.block_id = expected_parent_block_id;
    copy.length = POLARIS_BLOCK_SIZE;
    copy.user_cpu_addr = (uint64_t)(uintptr_t)scratch;
    copy.direction = POLARIS_RM_COPY_TO_CPU;
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_RM_COPY,
                              &copy,
                              "POLARIS_RM_COPY COW source TO_CPU") != 0)
        goto fail;
    if (copy.bytes_copied != POLARIS_BLOCK_SIZE) {
        fprintf(stderr,
                "COW source copy bytes=0x%llx expected=0x%llx\n",
                (unsigned long long)copy.bytes_copied,
                (unsigned long long)POLARIS_BLOCK_SIZE);
        goto fail;
    }

    if (allocate_rm_memory(s, "RM_ALLOC COW memory", POLARIS_BLOCK_SIZE, &new_h_memory) != 0)
        goto fail;

    memset(&copy, 0, sizeof(copy));
    copy.block_id = decision->block_id;
    copy.length = POLARIS_BLOCK_SIZE;
    copy.user_cpu_addr = (uint64_t)(uintptr_t)scratch;
    copy.direction = POLARIS_RM_COPY_FROM_CPU;
    copy.rm_control_fd = s->ctl_fd;
    copy.rm_h_client = s->h_client;
    copy.rm_h_memory = new_h_memory;
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_RM_COPY,
                              &copy,
                              "POLARIS_RM_COPY COW destination FROM_CPU") != 0)
        goto fail_free_rm;
    if (copy.bytes_copied != POLARIS_BLOCK_SIZE) {
        fprintf(stderr,
                "COW destination copy bytes=0x%llx expected=0x%llx\n",
                (unsigned long long)copy.bytes_copied,
                (unsigned long long)POLARIS_BLOCK_SIZE);
        goto fail_free_rm;
    }

    if (complete_decision(s, decision, 0, 0, new_h_memory) != 0)
        goto fail_free_rm;

    free(scratch);
    *new_h_memory_out = new_h_memory;
    printf("POLARIS RM COW completed: src_block=%llu child_block=%llu hMemory=0x%x bytes=0x%llx\n",
           (unsigned long long)expected_parent_block_id,
           (unsigned long long)decision->block_id,
           new_h_memory,
           (unsigned long long)copy.bytes_copied);
    return 0;

fail_free_rm:
    rm_free_object(s->ctl_fd, s->h_client, s->h_device, new_h_memory);
fail:
    free(scratch);
    return -1;
}

static void *rm_cow_executor(void *arg)
{
    struct rm_cow_args *args = arg;
    struct polaris_decision cow = {0};

    args->result = -1;
    for (int attempt = 0; attempt < 5000; ++attempt) {
        struct polaris_get_decision_arg get = {0};

        if (polaris_ioctl_checked(args->state->polaris_fd,
                                  POLARIS_GET_DECISION,
                                  &get,
                                  "POLARIS_GET_DECISION COW executor") != 0)
            return NULL;

        for (uint32_t i = 0; i < get.count && i < POLARIS_MAX_DECISIONS_PER_POLL; ++i) {
            if (get.decisions[i].op == POLARIS_DECISION_OP_COW_BREAK &&
                (args->child_block_id == 0 ||
                 get.decisions[i].block_id == args->child_block_id)) {
                cow = get.decisions[i];
                goto found;
            }

            fprintf(stderr,
                    "unexpected decision while waiting for COW block=%llu: op=%u block=%llu decision=%llu\n",
                    (unsigned long long)args->child_block_id,
                    get.decisions[i].op,
                    (unsigned long long)get.decisions[i].block_id,
                    (unsigned long long)get.decisions[i].decision_id);
            if (complete_decision(args->state, &get.decisions[i], -EINVAL, 0, 0) != 0)
                return NULL;
        }

        usleep(1000);
    }

    fprintf(stderr,
            "timed out waiting for COW decision block=%llu\n",
            (unsigned long long)args->child_block_id);
    return NULL;

found:
    if (args->child_vaddr != 0 && cow.dst_vaddr != args->child_vaddr) {
        fprintf(stderr,
                "COW dst_vaddr=0x%llx expected=0x%llx\n",
                (unsigned long long)cow.dst_vaddr,
                (unsigned long long)args->child_vaddr);
        return NULL;
    }
    args->child_block_id = cow.block_id;
    args->child_vaddr = cow.dst_vaddr;
    if (execute_rm_cow_decision(args->state,
                                &cow,
                                args->parent_block_id,
                                &args->child_h_memory) != 0)
        return NULL;
    args->result = 0;
    return NULL;
}

static int rm_spill_reload_roundtrip(struct m2_state *s, uint64_t block_id)
{
    uint8_t *src = NULL;
    uint8_t *dst = NULL;
    uint64_t cpu_addr = 0;
    NvHandle old_h_memory = s->h_memory;
    NvHandle new_h_memory = 0;
    const uint64_t seed = 0x5350494c4c524d31ULL;
    int rc = -1;

    if (posix_memalign((void **)&src, 4096, POLARIS_BLOCK_SIZE) != 0) {
        fprintf(stderr, "posix_memalign spill src failed\n");
        goto out;
    }
    if (posix_memalign((void **)&dst, 4096, POLARIS_BLOCK_SIZE) != 0) {
        fprintf(stderr, "posix_memalign spill dst failed\n");
        goto out;
    }
    for (uint64_t i = 0; i < POLARIS_BLOCK_SIZE; ++i) {
        src[i] = rm_copy_roundtrip_pattern(seed, i);
        dst[i] = 0;
    }

    struct polaris_rm_copy_arg write_initial = {
        .block_id = block_id,
        .length = POLARIS_BLOCK_SIZE,
        .user_cpu_addr = (uint64_t)(uintptr_t)src,
        .direction = POLARIS_RM_COPY_FROM_CPU,
    };
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_RM_COPY,
                              &write_initial,
                              "POLARIS_RM_COPY initial FROM_CPU") != 0)
        goto out;

    struct polaris_spill_block_arg spill = {
        .block_id = block_id,
    };
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_SPILL_BLOCK,
                              &spill,
                              "POLARIS_SPILL_BLOCK RM-backed") != 0)
        goto out;
    if (spill.decision_id == 0) {
        fprintf(stderr, "POLARIS_SPILL_BLOCK did not queue an OFFLOAD decision\n");
        goto out;
    }
    printf("POLARIS RM spill queued: block=%llu decision=%llu unmapped=%u\n",
           (unsigned long long)block_id,
           (unsigned long long)spill.decision_id,
           spill.unmapped_count);

    struct polaris_decision offload = {0};
    if (wait_for_one_decision(s,
                              POLARIS_DECISION_OP_OFFLOAD,
                              block_id,
                              &offload) != 0)
        goto out;
    if (execute_rm_offload_decision(s, &offload, &cpu_addr) != 0)
        goto out;

    rm_free_object(s->ctl_fd, s->h_client, s->h_device, old_h_memory);
    s->h_memory = 0;

    struct rm_spill_reload_args reload_args = {
        .state = s,
        .gpu_id = offload.gpu_id,
        .block_id = block_id,
        .cpu_addr = cpu_addr,
        .new_h_memory = 0,
        .result = -1,
    };
    pthread_t reload_thread;
    int thread_ret = pthread_create(&reload_thread, NULL, rm_reload_executor, &reload_args);
    if (thread_ret != 0) {
        fprintf(stderr, "pthread_create reload executor failed: %s\n", strerror(thread_ret));
        goto out;
    }

    struct polaris_block_reserve_arg reserve = {
        .session_id = s->polaris_session_id,
        .token_start = s->polaris_block_token_start,
        .token_count = s->polaris_block_token_count,
        .phase = 2,
        .flags = POLARIS_RESERVE_FLAG_OVERWRITE,
    };
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_BLOCK_RESERVE,
                              &reserve,
                              "POLARIS_BLOCK_RESERVE RM reload") != 0) {
        (void)pthread_join(reload_thread, NULL);
        goto out;
    }
    if (pthread_join(reload_thread, NULL) != 0) {
        fprintf(stderr, "pthread_join reload executor failed\n");
        goto out;
    }
    if (reload_args.result != 0 || reload_args.new_h_memory == 0) {
        fprintf(stderr, "reload executor result=%d hMemory=0x%x\n",
                reload_args.result,
                (unsigned int)reload_args.new_h_memory);
        goto out;
    }
    new_h_memory = reload_args.new_h_memory;
    s->h_memory = new_h_memory;

    if (dispatch_test_fault(s, reserve.gpu_vaddr) != 0)
        goto out;

    struct polaris_rm_copy_arg read_back = {
        .block_id = block_id,
        .length = POLARIS_BLOCK_SIZE,
        .user_cpu_addr = (uint64_t)(uintptr_t)dst,
        .direction = POLARIS_RM_COPY_TO_CPU,
    };
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_RM_COPY,
                              &read_back,
                              "POLARIS_RM_COPY final TO_CPU") != 0)
        goto out;
    if (read_back.bytes_copied != POLARIS_BLOCK_SIZE) {
        fprintf(stderr,
                "final RM copy bytes=0x%llx expected=0x%llx\n",
                (unsigned long long)read_back.bytes_copied,
                (unsigned long long)POLARIS_BLOCK_SIZE);
        goto out;
    }

    for (uint64_t i = 0; i < POLARIS_BLOCK_SIZE; ++i) {
        if (dst[i] != src[i]) {
            fprintf(stderr,
                    "RM spill/reload mismatch at 0x%llx: expected=0x%x actual=0x%x\n",
                    (unsigned long long)i,
                    src[i],
                    dst[i]);
            goto out;
        }
    }

    printf("POLARIS RM spill/reload roundtrip complete: block=%llu old_hMemory=0x%x new_hMemory=0x%x bytes=0x%llx\n",
           (unsigned long long)block_id,
           old_h_memory,
           new_h_memory,
           (unsigned long long)read_back.bytes_copied);
    rc = 0;

out:
    free(dst);
    free(src);
    return rc;
}

static int rm_cow_roundtrip(struct m2_state *s,
                            uint32_t gpu_id,
                            uint64_t rm_client_token,
                            uint64_t va_space_token,
                            uint64_t parent_block_id)
{
    uint8_t *src = NULL;
    uint8_t *parent_dst = NULL;
    uint8_t *child_dst = NULL;
    NvHandle parent_h_memory = s->h_memory;
    NvHandle child_h_memory = 0;
    uint64_t child_session_id = 0;
    uint64_t child_block_id = 0;
    uint64_t child_vaddr = 0;
    pthread_t cow_thread = 0;
    bool cow_thread_started = false;
    const uint64_t seed = 0x434f57524d434f50ULL;
    int rc = -1;

    if (posix_memalign((void **)&src, 4096, POLARIS_BLOCK_SIZE) != 0 ||
        posix_memalign((void **)&parent_dst, 4096, POLARIS_BLOCK_SIZE) != 0 ||
        posix_memalign((void **)&child_dst, 4096, POLARIS_BLOCK_SIZE) != 0) {
        fprintf(stderr, "posix_memalign COW buffers failed\n");
        goto out;
    }
    for (uint64_t i = 0; i < POLARIS_BLOCK_SIZE; ++i) {
        src[i] = rm_copy_roundtrip_pattern(seed, i);
        parent_dst[i] = 0;
        child_dst[i] = 0;
    }

    struct polaris_rm_copy_arg write_parent = {
        .block_id = parent_block_id,
        .length = POLARIS_BLOCK_SIZE,
        .user_cpu_addr = (uint64_t)(uintptr_t)src,
        .direction = POLARIS_RM_COPY_FROM_CPU,
    };
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_RM_COPY,
                              &write_parent,
                              "POLARIS_RM_COPY COW parent FROM_CPU") != 0)
        goto out;

    struct polaris_session_branch_arg branch = {
        .parent_session_id = s->polaris_session_id,
    };
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_SESSION_BRANCH,
                              &branch,
                              "POLARIS_SESSION_BRANCH RM COW") != 0)
        goto out;
    child_session_id = branch.child_session_id;
    if (child_session_id == 0) {
        fprintf(stderr, "POLARIS_SESSION_BRANCH returned child_session_id=0\n");
        goto out;
    }

    struct rm_cow_args cow_args = {
        .state = s,
        .parent_block_id = parent_block_id,
        .child_block_id = 0,
        .child_vaddr = 0,
        .child_h_memory = 0,
        .result = -1,
    };
    int thread_ret = pthread_create(&cow_thread, NULL, rm_cow_executor, &cow_args);
    if (thread_ret != 0) {
        fprintf(stderr, "pthread_create COW executor failed: %s\n", strerror(thread_ret));
        goto out;
    }
    cow_thread_started = true;

    struct polaris_block_reserve_arg reserve = {
        .session_id = child_session_id,
        .token_start = s->polaris_block_token_start,
        .token_count = s->polaris_block_token_count,
        .phase = 2,
        .flags = POLARIS_RESERVE_FLAG_OVERWRITE,
    };
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_BLOCK_RESERVE,
                              &reserve,
                              "POLARIS_BLOCK_RESERVE RM COW deferred") != 0)
        goto out;
    child_block_id = reserve.block_id;
    child_vaddr = reserve.gpu_vaddr;
    if (child_block_id == 0 || child_block_id == parent_block_id || child_vaddr == 0) {
        fprintf(stderr,
                "RM COW reserve returned invalid child block=%llu vaddr=0x%llx parent=%llu\n",
                (unsigned long long)child_block_id,
                (unsigned long long)child_vaddr,
                (unsigned long long)parent_block_id);
        goto out;
    }

    if (pthread_join(cow_thread, NULL) != 0) {
        cow_thread_started = false;
        fprintf(stderr, "pthread_join COW executor failed\n");
        goto out;
    }
    cow_thread_started = false;
    if (cow_args.result != 0 || cow_args.child_h_memory == 0) {
        fprintf(stderr,
                "COW executor result=%d hMemory=0x%x\n",
                cow_args.result,
                (unsigned int)cow_args.child_h_memory);
        goto out;
    }
    if (cow_args.child_block_id != child_block_id || cow_args.child_vaddr != child_vaddr) {
        fprintf(stderr,
                "COW executor completed block=%llu vaddr=0x%llx, expected block=%llu vaddr=0x%llx\n",
                (unsigned long long)cow_args.child_block_id,
                (unsigned long long)cow_args.child_vaddr,
                (unsigned long long)child_block_id,
                (unsigned long long)child_vaddr);
        goto out;
    }
    child_h_memory = cow_args.child_h_memory;

    struct polaris_register_block_mapping_arg child_mapping = {
        .block_id = child_block_id,
        .gpu_id = gpu_id,
        .rm_client_token = rm_client_token,
        .va_space_token = va_space_token,
        .base = child_vaddr,
        .length = POLARIS_BLOCK_SIZE,
    };
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_REGISTER_BLOCK_MAPPING,
                              &child_mapping,
                              "POLARIS_REGISTER_BLOCK_MAPPING RM COW child") != 0)
        goto out;

    if (dispatch_test_fault(s, child_vaddr) != 0) {
        goto out;
    }

    struct polaris_rm_copy_arg read_parent = {
        .block_id = parent_block_id,
        .length = POLARIS_BLOCK_SIZE,
        .user_cpu_addr = (uint64_t)(uintptr_t)parent_dst,
        .direction = POLARIS_RM_COPY_TO_CPU,
    };
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_RM_COPY,
                              &read_parent,
                              "POLARIS_RM_COPY COW parent TO_CPU") != 0)
        goto out;

    struct polaris_rm_copy_arg read_child = {
        .block_id = child_block_id,
        .length = POLARIS_BLOCK_SIZE,
        .user_cpu_addr = (uint64_t)(uintptr_t)child_dst,
        .direction = POLARIS_RM_COPY_TO_CPU,
    };
    if (polaris_ioctl_checked(s->polaris_fd,
                              POLARIS_RM_COPY,
                              &read_child,
                              "POLARIS_RM_COPY COW child TO_CPU") != 0)
        goto out;

    for (uint64_t i = 0; i < POLARIS_BLOCK_SIZE; ++i) {
        if (parent_dst[i] != src[i] || child_dst[i] != src[i]) {
            fprintf(stderr,
                    "RM COW mismatch at 0x%llx: expected=0x%x parent=0x%x child=0x%x\n",
                    (unsigned long long)i,
                    src[i],
                    parent_dst[i],
                    child_dst[i]);
            goto out;
        }
    }

    printf("POLARIS RM COW roundtrip complete: parent_block=%llu child_block=%llu parent_hMemory=0x%x child_hMemory=0x%x bytes=0x%llx\n",
           (unsigned long long)parent_block_id,
           (unsigned long long)child_block_id,
           parent_h_memory,
           child_h_memory,
           (unsigned long long)read_child.bytes_copied);
    rc = 0;

out:
    if (cow_thread_started) {
        (void)pthread_join(cow_thread, NULL);
    }
    if (child_session_id != 0) {
        struct polaris_session_destroy_arg destroy_child = {
            .session_id = child_session_id,
        };
        (void)polaris_ioctl_checked(s->polaris_fd,
                                    POLARIS_SESSION_DESTROY,
                                    &destroy_child,
                                    "POLARIS_SESSION_DESTROY RM COW child");
    }
    if (s->polaris_session_id != 0 && s->polaris_block_reserved) {
        struct polaris_block_release_arg release_parent = {
            .session_id = s->polaris_session_id,
            .token_start = s->polaris_block_token_start,
            .token_count = s->polaris_block_token_count,
            .flags = POLARIS_RELEASE_FLAG_CALLER_OWNS_BACKING,
        };
        (void)polaris_ioctl_checked(s->polaris_fd,
                                    POLARIS_BLOCK_RELEASE,
                                    &release_parent,
                                    "POLARIS_BLOCK_RELEASE RM COW parent");
        s->polaris_block_reserved = false;
    }
    rm_free_object(s->ctl_fd, s->h_client, s->h_device, child_h_memory);
    free(child_dst);
    free(parent_dst);
    free(src);
    return rc;
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
    if (s->polaris_session_id != 0 && s->polaris_block_reserved) {
        struct polaris_block_release_arg release = {
            .session_id = s->polaris_session_id,
            .token_start = s->polaris_block_token_start,
            .token_count = s->polaris_block_token_count,
            .flags = POLARIS_RELEASE_FLAG_CALLER_OWNS_BACKING,
        };
        (void)polaris_ioctl_checked(s->polaris_fd,
                                    POLARIS_BLOCK_RELEASE,
                                    &release,
                                    "POLARIS_BLOCK_RELEASE");
        s->polaris_block_reserved = false;
    }

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
    bool deferred_complete_fault = false;
    bool spill_validation = false;
    bool cuda_copy_probe = false;
    bool rm_cpu_map_probe = false;
    bool rm_phys_probe = false;
    bool rm_copy_probe = false;
    bool rm_copy_roundtrip_mode = false;
    bool rm_spill_reload_roundtrip_mode = false;
    bool rm_cow_roundtrip_mode = false;
    uint64_t block_id = 0;
    struct m2_state s = {
        .ctl_fd = -1,
        .gpu_fd = -1,
        .uvm_fd = -1,
        .uvm_mm_fd = -1,
        .polaris_fd = -1,
    };
    pthread_t completion_executor = 0;
    bool completion_executor_started = false;
    struct completion_executor_args exec_args = {0};
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
        if (strcmp(argv[i], "--rm-phys-probe") == 0) {
            dispatch_fault = true;
            block_unmap_refault = true;
            complete_backed_refault = true;
            rm_phys_probe = true;
            continue;
        }
        if (strcmp(argv[i], "--rm-copy-probe") == 0) {
            dispatch_fault = true;
            block_unmap_refault = true;
            complete_backed_refault = true;
            rm_copy_probe = true;
            continue;
        }
        if (strcmp(argv[i], "--rm-copy-roundtrip") == 0) {
            dispatch_fault = true;
            block_unmap_refault = true;
            complete_backed_refault = true;
            rm_copy_roundtrip_mode = true;
            continue;
        }
        if (strcmp(argv[i], "--rm-spill-reload-roundtrip") == 0) {
            dispatch_fault = true;
            complete_backed_refault = true;
            rm_spill_reload_roundtrip_mode = true;
            continue;
        }
        if (strcmp(argv[i], "--rm-cow-roundtrip") == 0) {
            dispatch_fault = true;
            complete_backed_refault = true;
            rm_cow_roundtrip_mode = true;
            continue;
        }
        if (strcmp(argv[i], "--complete-backed-refault") == 0) {
            dispatch_fault = true;
            block_unmap_refault = true;
            complete_backed_refault = true;
            continue;
        }
        if (strcmp(argv[i], "--deferred-complete-fault") == 0) {
            dispatch_fault = true;
            block_unmap_refault = true;
            deferred_complete_fault = true;
            continue;
        }
        if (strcmp(argv[i], "--spill-validation") == 0) {
            spill_validation = true;
            continue;
        }
        if (strcmp(argv[i], "--cuda-copy-probe") == 0) {
            cuda_copy_probe = true;
            continue;
        }
        if (strcmp(argv[i], "--rm-cpu-map-probe") == 0) {
            rm_cpu_map_probe = true;
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
                        "usage: %s [--dispatch-fault|--unmap-refault|--block-unmap-refault|--logical-backed-refault|--rm-phys-probe|--rm-copy-probe|--rm-copy-roundtrip|--rm-spill-reload-roundtrip|--rm-cow-roundtrip|--complete-backed-refault|--deferred-complete-fault|--spill-validation|--cuda-copy-probe|--rm-cpu-map-probe] [cuda_ordinal] [polaris_gpu_id] [base]\n",
                        argv[0]);
                goto out;
        }
    }

    if (rm_cpu_map_probe) {
        rc = run_rm_cpu_map_probe_isolated(ordinal) == 0 ? 0 : 1;
        goto out_no_cleanup;
    }

    if (cuda_copy_probe) {
        rc = run_cuda_copy_probe_isolated(ordinal, base) == 0 ? 0 : 1;
        goto out_no_cleanup;
    }

    if (get_cuda_uuid(ordinal, &s.gpu_uuid) != 0)
        goto out;
    if (setup_rm(&s, ordinal) != 0)
        goto out;
    uint64_t managed_length = rm_cow_roundtrip_mode
        ? (2 * POLARIS_MANAGED_SIZE)
        : POLARIS_MANAGED_SIZE;
    if (setup_uvm(&s, base, managed_length, !dispatch_fault) != 0)
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
                      managed_length,
                      !logical_backed_refault && !complete_backed_refault && !deferred_complete_fault) != 0)
        goto out;
    if (complete_backed_refault || deferred_complete_fault) {
        exec_args.state = &s;
        exec_args.gpu_id = polaris_gpu_id;
        exec_args.expected_base = base;
        int thread_ret = pthread_create(&completion_executor,
                                        NULL,
                                        complete_backing_executor,
                                        &exec_args);
        if (thread_ret != 0) {
            fprintf(stderr,
                    "pthread_create completion executor failed: %s\n",
                    strerror(thread_ret));
            goto out;
        }
        completion_executor_started = true;

        if (register_logical_block_mapping(&s,
                                           polaris_gpu_id,
                                                   polaris_rm_client_token,
                                                   polaris_va_space_token,
                                                   base,
                                                   managed_length,
                                                   POLARIS_BLOCK_SIZE,
                                                   deferred_complete_fault,
                                                   &block_id) != 0) {
            goto out;
        }
        if (complete_backed_refault) {
            if (pthread_join(completion_executor, NULL) != 0) {
                completion_executor_started = false;
                fprintf(stderr, "pthread_join completion executor failed\n");
                goto out;
            }
            completion_executor_started = false;
            if (exec_args.result != 0 || exec_args.completed_block_id != block_id) {
                fprintf(stderr,
                        "completion executor result=%d completed_block=%llu expected_block=%llu\n",
                        exec_args.result,
                        (unsigned long long)exec_args.completed_block_id,
                        (unsigned long long)block_id);
                goto out;
            }
        }
    }
    if ((block_unmap_refault || spill_validation) && !complete_backed_refault && !deferred_complete_fault) {
        if (register_logical_block_mapping(&s,
                                           polaris_gpu_id,
                                                   polaris_rm_client_token,
                                                   polaris_va_space_token,
                                                   base,
                                                   managed_length,
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
        if (deferred_complete_fault) {
            if (pthread_join(completion_executor, NULL) != 0) {
                completion_executor_started = false;
                fprintf(stderr, "pthread_join completion executor failed\n");
                goto out;
            }
            completion_executor_started = false;
            if (exec_args.result != 0 || exec_args.completed_block_id != block_id) {
                fprintf(stderr,
                        "completion executor result=%d completed_block=%llu expected_block=%llu\n",
                        exec_args.result,
                        (unsigned long long)exec_args.completed_block_id,
                        (unsigned long long)block_id);
                goto out;
            }
        }
        if ((complete_backed_refault || deferred_complete_fault) &&
            !rm_spill_reload_roundtrip_mode &&
            !rm_cow_roundtrip_mode) {
            if (dispatch_test_fault(&s, base) != 0)
                goto out;
        }
        if (rm_phys_probe) {
            if (probe_rm_phys(&s, block_id) != 0)
                goto out;
        }
        if (rm_copy_probe) {
            if (probe_rm_copy(&s, block_id) != 0)
                goto out;
        }
        if (rm_copy_roundtrip_mode) {
            if (rm_copy_roundtrip(&s, block_id) != 0)
                goto out;
        }
        if (rm_spill_reload_roundtrip_mode) {
            if (rm_spill_reload_roundtrip(&s, block_id) != 0)
                goto out;
        }
        if (rm_cow_roundtrip_mode) {
            if (rm_cow_roundtrip(&s,
                                 polaris_gpu_id,
                                 polaris_rm_client_token,
                                 polaris_va_space_token,
                                 block_id) != 0)
                goto out;
        }
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

    if (rm_phys_probe)
        puts("M3 Polaris RM phys probe passed.");
    else if (rm_cow_roundtrip_mode)
        puts("M4 Polaris RM COW roundtrip passed.");
    else if (rm_spill_reload_roundtrip_mode)
        puts("M3 Polaris RM spill/reload roundtrip passed.");
    else if (deferred_complete_fault)
        puts("M3 Polaris deferred completion fault test passed.");
    else if (complete_backed_refault)
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
    if (completion_executor_started)
        (void)pthread_join(completion_executor, NULL);
    cleanup(&s, polaris_gpu_id, polaris_rm_client_token, polaris_va_space_token);
out_no_cleanup:
    return rc;
}
