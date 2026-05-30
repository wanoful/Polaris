#include "polaris_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st * CUcontext;
typedef int CUresult;

enum { CUDA_SUCCESS = 0 };

extern CUresult cuInit(unsigned int flags);
extern CUresult cuDeviceGet(CUdevice * device, int ordinal);
extern CUresult cuDevicePrimaryCtxRetain(CUcontext * pctx, CUdevice dev);
extern CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
extern CUresult cuCtxPushCurrent_v2(CUcontext ctx);
extern CUresult cuCtxPopCurrent_v2(CUcontext * pctx);
extern CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void * srcHost, size_t ByteCount);
extern CUresult cuMemcpyDtoH_v2(void * dstHost, CUdeviceptr srcDevice, size_t ByteCount);

static int check_cuda(CUresult rc, const char * what) {
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "%s failed: %d\n", what, rc);
        return 1;
    }
    return 0;
}

static int check_polaris(int rc, const char * what) {
    if (rc != 0) {
        fprintf(stderr, "%s failed: rc=%d err=%s\n", what, rc, polaris_runtime_last_error());
        return 1;
    }
    return 0;
}

static void fill_pattern(uint8_t * buf, size_t size, uint8_t seed) {
    for (size_t i = 0; i < size; ++i) {
        buf[i] = (uint8_t)(seed + (i * 131u) + (i >> 7));
    }
}

static int copy_and_verify(uint64_t va, size_t size, uint8_t seed) {
    uint8_t * src = malloc(size);
    uint8_t * dst = malloc(size);
    if (!src || !dst) {
        fprintf(stderr, "malloc failed\n");
        free(src);
        free(dst);
        return 1;
    }

    fill_pattern(src, size, seed);
    memset(dst, 0, size);
    if (check_cuda(cuMemcpyHtoD_v2((CUdeviceptr)va, src, size), "cuMemcpyHtoD")) {
        free(src);
        free(dst);
        return 1;
    }
    if (check_cuda(cuMemcpyDtoH_v2(dst, (CUdeviceptr)va, size), "cuMemcpyDtoH")) {
        free(src);
        free(dst);
        return 1;
    }
    int mismatch = memcmp(src, dst, size);
    if (mismatch != 0) {
        fprintf(stderr, "verification failed for VA 0x%llx size %zu\n",
                (unsigned long long)va, size);
    }
    free(src);
    free(dst);
    return mismatch != 0;
}

int main(void) {
    polaris_runtime_t * rt = NULL;
    polaris_runtime_info_t info = {0};
    polaris_kv_allocation_t kv = {0};
    const uint64_t mib = 1024ull * 1024ull;

    polaris_runtime_config_t cfg = {
        .gpu_id = 0,
        .device_ordinal = 0,
        .total_bytes = 16ull * 1024ull * mib,
        .budget_bytes = 512ull * mib,
        .cpu_pool_bytes = 64ull * mib,
        .va_reserve_bytes = 256ull * mib,
        .block_size = 2ull * mib,
        .flags = 0,
    };

    if (check_polaris(polaris_runtime_create(&cfg, &rt, &info), "polaris_runtime_create")) {
        return 1;
    }
    printf("runtime va=0x%llx size=%llu MiB granule=%llu KiB\n",
           (unsigned long long)info.va_base,
           (unsigned long long)(info.va_size / mib),
           (unsigned long long)(info.granule / 1024ull));

    if (check_polaris(polaris_runtime_alloc_kv(rt, 4ull * mib, 2ull * mib, &kv),
                      "polaris_runtime_alloc_kv")) {
        polaris_runtime_destroy(rt);
        return 1;
    }
    printf("kv va=0x%llx size=%llu MiB block=%llu MiB blocks=%llu\n",
           (unsigned long long)kv.va,
           (unsigned long long)(kv.size / mib),
           (unsigned long long)(kv.block_size / mib),
           (unsigned long long)kv.block_count);

    CUdevice dev = 0;
    CUcontext ctx = NULL;
    if (check_cuda(cuInit(0), "cuInit") ||
        check_cuda(cuDeviceGet(&dev, 0), "cuDeviceGet") ||
        check_cuda(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain") ||
        check_cuda(cuCtxPushCurrent_v2(ctx), "cuCtxPushCurrent")) {
        polaris_runtime_destroy(rt);
        return 1;
    }

    int failed = 0;
    const uint64_t block0 = kv.va;
    const uint64_t block1 = kv.va + kv.block_size;

    failed |= check_polaris(polaris_runtime_map_kv_block(rt, kv.va, 0),
                            "polaris_runtime_map_kv_block block0");
    if (!failed) {
        failed |= copy_and_verify(block0, (size_t)kv.block_size, 0x31);
    }

    failed |= check_polaris(polaris_runtime_offload_kv_block(rt, kv.va, 0),
                            "polaris_runtime_offload_kv_block block0");
    failed |= check_polaris(polaris_runtime_map_kv_block(rt, kv.va, 1),
                            "polaris_runtime_map_kv_block block1");
    if (!failed) {
        failed |= copy_and_verify(block1, (size_t)kv.block_size, 0x52);
    }

    failed |= check_polaris(polaris_runtime_reload_kv_block(rt, kv.va, 0),
                            "polaris_runtime_reload_kv_block block0");
    if (!failed) {
        uint8_t * dst = malloc((size_t)kv.block_size);
        uint8_t * expected = malloc((size_t)kv.block_size);
        if (!dst || !expected) {
            fprintf(stderr, "malloc failed\n");
            free(dst);
            free(expected);
            failed = 1;
        } else {
            fill_pattern(expected, (size_t)kv.block_size, 0x31);
            memset(dst, 0, (size_t)kv.block_size);
            failed |= check_cuda(cuMemcpyDtoH_v2(dst, (CUdeviceptr)block0, (size_t)kv.block_size),
                                 "cuMemcpyDtoH reloaded block0");
            if (!failed && memcmp(expected, dst, (size_t)kv.block_size) != 0) {
                fprintf(stderr, "offload/reload verification failed for block0\n");
                failed = 1;
            }
            free(dst);
            free(expected);
        }
    }

    failed |= check_polaris(polaris_runtime_unmap_kv_block(rt, kv.va, 1),
                            "polaris_runtime_unmap_kv_block block1");
    failed |= check_polaris(polaris_runtime_map_kv_all(rt, kv.va),
                            "polaris_runtime_map_kv_all after block ops");
    if (!failed) {
        failed |= copy_and_verify(block1, (size_t)kv.block_size, 0x79);
    }

    failed |= check_polaris(polaris_runtime_free_kv(rt, kv.va), "polaris_runtime_free_kv");

    CUcontext popped = NULL;
    (void)cuCtxPopCurrent_v2(&popped);
    (void)cuDevicePrimaryCtxRelease(dev);
    polaris_runtime_destroy(rt);

    if (failed) {
        return 1;
    }
    printf("POLARIS explicit KV block allocation/offload smoke passed\n");
    return 0;
}
