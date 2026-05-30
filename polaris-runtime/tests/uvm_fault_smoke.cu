#include "polaris_runtime.h"

// Diagnostic for M4 true-fault validation.
//
// This intentionally reserves a POLARIS block without running the synchronous
// fault resolver, then launches a CUDA kernel that writes the unmapped VA. A
// fully working replayable-fault path would map the block and let the kernel
// complete. On the current tested NVIDIA driver, this fails as a fatal raw CUDA
// VMM MMU fault (Xid 31 / FAULT_PDE), which is useful evidence that explicit
// prefetch/reload must remain the benchmarkable path for now.

#include <cuda_runtime.h>
#include <errno.h>
#include <linux/ioctl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

#define POLARIS_IOCTL_MAGIC 0x50
#define POLARIS_RESERVE_FLAG_DEFER_FAULT (1u << 4)

typedef struct polaris_session_create_arg {
    uint64_t session_id;
    uint32_t home_gpu;
    uint32_t beam_width;
    uint64_t gpu_vas_bytes;
    uint64_t bytes_per_token;
    uint32_t priority;
    uint32_t _reserved;
    uint64_t _reserved2[2];
} polaris_session_create_arg_t;

typedef struct polaris_session_destroy_arg {
    uint64_t session_id;
    uint64_t _reserved[4];
} polaris_session_destroy_arg_t;

typedef struct polaris_block_reserve_arg {
    uint64_t session_id;
    uint32_t token_start;
    uint32_t token_count;
    uint32_t phase;
    uint32_t flags;
    uint64_t block_id;
    uint64_t gpu_vaddr;
    uint64_t _reserved[3];
} polaris_block_reserve_arg_t;

#define POLARIS_SESSION_CREATE  _IOWR(POLARIS_IOCTL_MAGIC, 0x03, polaris_session_create_arg_t)
#define POLARIS_SESSION_DESTROY _IOW(POLARIS_IOCTL_MAGIC, 0x04, polaris_session_destroy_arg_t)
#define POLARIS_BLOCK_RESERVE   _IOWR(POLARIS_IOCTL_MAGIC, 0x07, polaris_block_reserve_arg_t)

__global__ void polaris_touch_kernel(uint8_t * ptr) {
    ptr[0] = 0x5a;
}

static int check_polaris(int rc, const char * what) {
    if (rc != 0) {
        fprintf(stderr, "%s failed: rc=%d err=%s\n", what, rc, polaris_runtime_last_error());
        return 1;
    }
    return 0;
}

static int check_ioctl(int rc, const char * what) {
    if (rc != 0) {
        fprintf(stderr, "%s failed: errno=%d (%s)\n", what, errno, strerror(errno));
        return 1;
    }
    return 0;
}

static int check_cuda(cudaError_t rc, const char * what) {
    if (rc != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(rc));
        return 1;
    }
    return 0;
}

int main(void) {
    const uint64_t mib = 1024ull * 1024ull;
    const uint64_t block_size = 2ull * mib;
    polaris_runtime_t * rt = NULL;
    polaris_runtime_info_t info = {0};
    int failed = 0;

    polaris_runtime_config_t cfg = {
        .gpu_id = 0,
        .device_ordinal = 0,
        .total_bytes = 16ull * 1024ull * mib,
        .budget_bytes = 512ull * mib,
        .cpu_pool_bytes = 64ull * mib,
        .va_reserve_bytes = 256ull * mib,
        .block_size = block_size,
        .flags = 0,
    };

    if (check_polaris(polaris_runtime_create(&cfg, &rt, &info), "polaris_runtime_create")) {
        return 1;
    }
    if (check_polaris(polaris_runtime_start(rt), "polaris_runtime_start")) {
        polaris_runtime_destroy(rt);
        return 1;
    }

    int fd = open("/dev/polaris", O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "open /dev/polaris failed: errno=%d (%s)\n", errno, strerror(errno));
        polaris_runtime_destroy(rt);
        return 1;
    }

    polaris_session_create_arg_t session = {
        .home_gpu = 0,
        .beam_width = 1,
        .gpu_vas_bytes = block_size,
        .bytes_per_token = block_size,
        .priority = 5,
    };
    failed |= check_ioctl(ioctl(fd, POLARIS_SESSION_CREATE, &session), "POLARIS_SESSION_CREATE");

    polaris_block_reserve_arg_t reserve = {
        .session_id = session.session_id,
        .token_start = 0,
        .token_count = 1,
        .phase = 0,
        .flags = POLARIS_RESERVE_FLAG_DEFER_FAULT,
    };
    if (!failed) {
        failed |= check_ioctl(ioctl(fd, POLARIS_BLOCK_RESERVE, &reserve), "POLARIS_BLOCK_RESERVE defer");
    }
    if (!failed) {
        printf("deferred block id=%llu va=0x%llx\n",
               (unsigned long long)reserve.block_id,
               (unsigned long long)reserve.gpu_vaddr);
    }

    if (!failed) {
        polaris_touch_kernel<<<1, 1>>>((uint8_t *)(uintptr_t)reserve.gpu_vaddr);
        failed |= check_cuda(cudaGetLastError(), "polaris_touch_kernel launch");
        cudaError_t sync_rc = cudaDeviceSynchronize();
        if (sync_rc == cudaErrorIllegalAddress) {
            fprintf(stderr,
                    "polaris_touch_kernel synchronize failed: %s\n"
                    "raw CUDA VMM VA was not replayed through POLARIS; check dmesg for NVIDIA Xid/MMU fault\n",
                    cudaGetErrorString(sync_rc));
            failed = 1;
        } else {
            failed |= check_cuda(sync_rc, "polaris_touch_kernel synchronize");
        }
    }

    if (!failed) {
        uint8_t value = 0;
        failed |= check_cuda(cudaMemcpy(&value,
                                        (const void *)(uintptr_t)reserve.gpu_vaddr,
                                        sizeof(value),
                                        cudaMemcpyDeviceToHost),
                             "cudaMemcpy DtoH");
        if (!failed && value != 0x5a) {
            fprintf(stderr, "unexpected value: got 0x%02x want 0x5a\n", value);
            failed = 1;
        }
    }

    if (session.session_id != 0) {
        polaris_session_destroy_arg_t destroy = {
            .session_id = session.session_id,
        };
        (void)ioctl(fd, POLARIS_SESSION_DESTROY, &destroy);
    }
    close(fd);
    usleep(100000);
    polaris_runtime_destroy(rt);

    if (failed) {
        return 1;
    }
    printf("POLARIS UVM replayable fault smoke passed\n");
    return 0;
}
