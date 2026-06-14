// SPDX-License-Identifier: GPL-2.0
//
// Opt-in smoke for the shim's managed allocator interposer. This deliberately
// avoids linking against CUDA: with LD_PRELOAD, RTLD_DEFAULT should resolve the
// shim's allocator exports. Running it requires a loaded polaris.ko and the
// POLARIS_SHIM_* VA-space bootstrap variables.
//
// Build:   cc -O2 -ldl managed_alloc.c -o managed_alloc
/*
 * Run:
 *   env POLARIS_SHIM_MANAGE_ALLOCATIONS=1 POLARIS_SHIM_TRANSIENT_GPU=1 \
 *       POLARIS_SHIM_GPU_ID=0 POLARIS_SHIM_VASPACE_TOKEN=0xdef00001 \
 *       POLARIS_SHIM_MANAGED_BASE=0x410000000000 \
 *       POLARIS_SHIM_MANAGED_LENGTH=0x400000 \
 *       LD_PRELOAD=$PWD/../libpolaris-shim.so ./managed_alloc
 */

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef int CUdevice_attribute;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef int cudaError_t;
typedef int cudaStreamCaptureMode;
typedef void *CUstream;
typedef void *CUfunction;
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
typedef void *cudaGraph_t;
typedef void *cudaGraphExec_t;
typedef void *cudaGraphNode_t;
typedef void cudaGraphExecUpdateResultInfo;
typedef void cudaKernelNodeParams;
typedef int cudaGraphNodeType;
typedef void (*cudaHostFn_t)(void *userData);
typedef int cudaStreamCaptureStatus;
typedef void CUmemAllocationProp;
typedef void CUmemAccessDesc;
typedef void CUlaunchConfig;
typedef void cudaLaunchConfig_t;
typedef int CUmemAllocationGranularity_flags;
typedef struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} dim3;
struct cudaMemLocation {
    int type;
    int id;
};
typedef struct {
    char bytes[64];
} CUipcMemHandle;
typedef struct {
    char bytes[64];
} cudaIpcMemHandle_t;
struct cudaPitchedPtr {
    void *ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
};
struct cudaExtent {
    size_t width;
    size_t height;
    size_t depth;
};
struct cudaPos {
    size_t x;
    size_t y;
    size_t z;
};
struct cudaMemcpy3DPeerParms {
    void *srcArray;
    struct cudaPos srcPos;
    struct cudaPitchedPtr srcPtr;
    int srcDevice;
    void *dstArray;
    struct cudaPos dstPos;
    struct cudaPitchedPtr dstPtr;
    int dstDevice;
    struct cudaExtent extent;
};
typedef CUresult (*cuDeviceGet_fn)(CUdevice *device, int ordinal);
typedef CUresult (*cuDeviceGetAttribute_fn)(int *pi, CUdevice_attribute attrib, CUdevice dev);
typedef CUresult (*cuDevicePrimaryCtxRetain_fn)(CUcontext *pctx, CUdevice dev);
typedef CUresult (*cuCtxSetCurrent_fn)(CUcontext ctx);
typedef CUresult (*cuMemAlloc_v2_fn)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (*cuMemFree_v2_fn)(CUdeviceptr dptr);
typedef CUresult (*cuMemAllocAsync_v2_fn)(CUdeviceptr *dptr, size_t bytesize, void *stream);
typedef CUresult (*cuMemFreeAsync_v2_fn)(CUdeviceptr dptr, void *stream);
typedef CUresult (*cuMemGetInfo_v2_fn)(size_t *free_bytes, size_t *total_bytes);
typedef CUresult (*cuMemAddressReserve_fn)(CUdeviceptr *ptr,
                                           size_t size,
                                           size_t alignment,
                                           CUdeviceptr addr,
                                           unsigned long long flags);
typedef CUresult (*cuMemAddressFree_fn)(CUdeviceptr ptr, size_t size);
typedef CUresult (*cuMemCreate_fn)(CUmemGenericAllocationHandle *handle,
                                   size_t size,
                                   const CUmemAllocationProp *prop,
                                   unsigned long long flags);
typedef CUresult (*cuMemRelease_fn)(CUmemGenericAllocationHandle handle);
typedef CUresult (*cuMemMap_fn)(CUdeviceptr ptr,
                                size_t size,
                                size_t offset,
                                CUmemGenericAllocationHandle handle,
                                unsigned long long flags);
typedef CUresult (*cuMemUnmap_fn)(CUdeviceptr ptr, size_t size);
typedef CUresult (*cuMemSetAccess_fn)(CUdeviceptr ptr,
                                      size_t size,
                                      const CUmemAccessDesc *desc,
                                      size_t count);
typedef CUresult (*cuMemGetAllocationGranularity_fn)(
    size_t *granularity,
    const CUmemAllocationProp *prop,
    CUmemAllocationGranularity_flags option);
typedef CUresult (*cuGetErrorString_fn)(CUresult error, const char **pStr);
typedef CUresult (*cuGetErrorName_fn)(CUresult error, const char **pStr);
typedef CUresult (*cuStreamBeginCapture_v2_fn)(CUstream stream, cudaStreamCaptureMode mode);
typedef CUresult (*cuMemGetAddressRange_v2_fn)(CUdeviceptr *pbase,
                                               size_t *psize,
                                               CUdeviceptr dptr);
typedef CUresult (*cuMemcpyHtoD_v2_fn)(CUdeviceptr dstDevice,
                                       const void *srcHost,
                                       size_t ByteCount);
typedef CUresult (*cuMemcpyDtoH_v2_fn)(void *dstHost,
                                       CUdeviceptr srcDevice,
                                       size_t ByteCount);
typedef CUresult (*cuMemcpyHtoDAsync_v2_fn)(CUdeviceptr dstDevice,
                                            const void *srcHost,
                                            size_t ByteCount,
                                            void *stream);
typedef CUresult (*cuMemcpyDtoHAsync_v2_fn)(void *dstHost,
                                            CUdeviceptr srcDevice,
                                            size_t ByteCount,
                                            void *stream);
typedef CUresult (*cuMemcpy_v2_fn)(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
typedef CUresult (*cuMemcpyAsync_v2_fn)(CUdeviceptr dst,
                                        CUdeviceptr src,
                                        size_t ByteCount,
                                        void *stream);
typedef CUresult (*cuMemsetD8_v2_fn)(CUdeviceptr dstDevice, unsigned char uc, size_t N);
typedef CUresult (*cuMemsetD8Async_v2_fn)(CUdeviceptr dstDevice,
                                          unsigned char uc,
                                          size_t N,
                                          void *stream);
typedef CUresult (*cuMemsetD16_v2_fn)(CUdeviceptr dstDevice, unsigned short us, size_t N);
typedef CUresult (*cuMemsetD16Async_v2_fn)(CUdeviceptr dstDevice,
                                           unsigned short us,
                                           size_t N,
                                           void *stream);
typedef CUresult (*cuMemsetD32_v2_fn)(CUdeviceptr dstDevice, unsigned int ui, size_t N);
typedef CUresult (*cuMemsetD32Async_v2_fn)(CUdeviceptr dstDevice,
                                           unsigned int ui,
                                           size_t N,
                                           void *stream);
typedef CUresult (*cuIpcGetMemHandle_fn)(CUipcMemHandle *pHandle, CUdeviceptr dptr);
typedef CUresult (*cuPointerGetAttribute_fn)(void *data, int attribute, CUdeviceptr ptr);
typedef CUresult (*cuPointerGetAttributes_fn)(unsigned int numAttributes,
                                              int *attributes,
                                              void **data,
                                              CUdeviceptr ptr);
typedef CUresult (*cuLaunchKernel_fn)(CUfunction f,
                                      unsigned int gridDimX,
                                      unsigned int gridDimY,
                                      unsigned int gridDimZ,
                                      unsigned int blockDimX,
                                      unsigned int blockDimY,
                                      unsigned int blockDimZ,
                                      unsigned int sharedMemBytes,
                                      CUstream hStream,
                                      void **kernelParams,
                                      void **extra);
typedef CUresult (*cuLaunchKernelEx_fn)(const CUlaunchConfig *config,
                                        CUfunction f,
                                        void **kernelParams,
                                        void **extra);
typedef cudaError_t (*cudaMalloc_fn)(void **devPtr, size_t size);
typedef cudaError_t (*cudaMallocManaged_fn)(void **devPtr, size_t size, unsigned int flags);
typedef cudaError_t (*cudaFree_fn)(void *devPtr);
typedef cudaError_t (*cudaMallocAsync_fn)(void **devPtr, size_t size, void *stream);
typedef cudaError_t (*cudaFreeAsync_fn)(void *devPtr, void *stream);
typedef cudaError_t (*cudaGetDevice_fn)(int *device);
typedef cudaError_t (*cudaSetDevice_fn)(int device);
typedef cudaError_t (*cudaSetDeviceFlags_fn)(unsigned int flags);
typedef cudaError_t (*cudaGetDeviceFlags_fn)(unsigned int *flags);
typedef cudaError_t (*cudaGetDeviceCount_fn)(int *count);
typedef cudaError_t (*cudaDeviceGetPCIBusId_fn)(char *pciBusId, int len, int device);
typedef cudaError_t (*cudaRuntimeGetVersion_fn)(int *runtimeVersion);
typedef cudaError_t (*cudaDriverGetVersion_fn)(int *driverVersion);
typedef cudaError_t (*cudaMemGetInfo_fn)(size_t *free_bytes, size_t *total_bytes);
typedef const char *(*cudaGetErrorString_fn)(cudaError_t error);
typedef const char *(*cudaGetErrorName_fn)(cudaError_t error);
typedef cudaError_t (*cudaGetLastError_fn)(void);
typedef cudaError_t (*cudaPeekAtLastError_fn)(void);
typedef cudaError_t (*cudaStreamCreate_fn)(cudaStream_t *stream);
typedef cudaError_t (*cudaStreamCreateWithFlags_fn)(cudaStream_t *stream, unsigned int flags);
typedef cudaError_t (*cudaStreamDestroy_fn)(cudaStream_t stream);
typedef cudaError_t (*cudaStreamSynchronize_fn)(cudaStream_t stream);
typedef cudaError_t (*cudaStreamWaitEvent_fn)(cudaStream_t stream,
                                              cudaEvent_t event,
                                              unsigned int flags);
typedef cudaError_t (*cudaStreamIsCapturing_fn)(cudaStream_t stream,
                                                cudaStreamCaptureStatus *capture_status);
typedef cudaError_t (*cudaEventCreateWithFlags_fn)(cudaEvent_t *event, unsigned int flags);
typedef cudaError_t (*cudaEventRecord_fn)(cudaEvent_t event, cudaStream_t stream);
typedef cudaError_t (*cudaEventSynchronize_fn)(cudaEvent_t event);
typedef cudaError_t (*cudaEventDestroy_fn)(cudaEvent_t event);
typedef cudaError_t (*cudaStreamBeginCapture_fn)(void *stream, cudaStreamCaptureMode mode);
typedef cudaError_t (*cudaStreamEndCapture_fn)(void *stream, cudaGraph_t *graph);
typedef cudaError_t (*cudaGraphInstantiate_fn)(cudaGraphExec_t *graph_exec,
                                               cudaGraph_t graph,
                                               unsigned long long flags);
typedef cudaError_t (*cudaGraphLaunch_fn)(cudaGraphExec_t graph_exec, void *stream);
typedef cudaError_t (*cudaGraphExecUpdate_fn)(cudaGraphExec_t graph_exec,
                                              cudaGraph_t graph,
                                              cudaGraphExecUpdateResultInfo *result_info);
typedef cudaError_t (*cudaGraphGetNodes_fn)(cudaGraph_t graph,
                                            cudaGraphNode_t *nodes,
                                            size_t *numNodes);
typedef cudaError_t (*cudaGraphNodeGetType_fn)(cudaGraphNode_t node,
                                               cudaGraphNodeType *type);
typedef cudaError_t (*cudaGraphKernelNodeGetParams_fn)(
    cudaGraphNode_t node,
    cudaKernelNodeParams *node_params);
typedef cudaError_t (*cudaGraphKernelNodeSetParams_fn)(
    cudaGraphNode_t node,
    const cudaKernelNodeParams *node_params);
typedef cudaError_t (*cudaGraphDestroy_fn)(cudaGraph_t graph);
typedef cudaError_t (*cudaGraphExecDestroy_fn)(cudaGraphExec_t graph_exec);
typedef cudaError_t (*cudaMemcpy_fn)(void *dst, const void *src, size_t count, int kind);
typedef cudaError_t (*cudaMemcpyAsync_fn)(void *dst,
                                          const void *src,
                                          size_t count,
                                          int kind,
                                          void *stream);
typedef cudaError_t (*cudaMemcpy2DAsync_fn)(void *dst,
                                            size_t dpitch,
                                            const void *src,
                                            size_t spitch,
                                            size_t width,
                                            size_t height,
                                            int kind,
                                            void *stream);
typedef cudaError_t (*cudaMemcpyPeerAsync_fn)(void *dst,
                                              int dstDevice,
                                              const void *src,
                                              int srcDevice,
                                              size_t count,
                                              void *stream);
typedef cudaError_t (*cudaMemcpy3DPeerAsync_fn)(const struct cudaMemcpy3DPeerParms *p,
                                                void *stream);
typedef cudaError_t (*cudaMemset_fn)(void *devPtr, int value, size_t count);
typedef cudaError_t (*cudaMemsetAsync_fn)(void *devPtr,
                                          int value,
                                          size_t count,
                                          void *stream);
typedef cudaError_t (*cudaIpcGetMemHandle_fn)(cudaIpcMemHandle_t *handle, void *devPtr);
typedef cudaError_t (*cudaPointerGetAttributes_fn)(void *attributes, const void *ptr);
typedef cudaError_t (*cudaDeviceGetAttribute_fn)(int *value, int attr, int device);
typedef cudaError_t (*cudaMallocHost_fn)(void **ptr, size_t size);
typedef cudaError_t (*cudaHostAlloc_fn)(void **pHost, size_t size, unsigned int flags);
typedef cudaError_t (*cudaFreeHost_fn)(void *ptr);
typedef cudaError_t (*cudaHostGetDevicePointer_fn)(void **pDevice,
                                                   void *pHost,
                                                   unsigned int flags);
typedef cudaError_t (*cudaHostRegister_fn)(void *ptr, size_t size, unsigned int flags);
typedef cudaError_t (*cudaHostUnregister_fn)(void *ptr);
typedef cudaError_t (*cudaDeviceCanAccessPeer_fn)(int *canAccessPeer,
                                                  int device,
                                                  int peerDevice);
typedef cudaError_t (*cudaDeviceEnablePeerAccess_fn)(int peerDevice, unsigned int flags);
typedef cudaError_t (*cudaDeviceDisablePeerAccess_fn)(int peerDevice);
typedef cudaError_t (*cudaMemAdvise_fn)(const void *devPtr,
                                        size_t count,
                                        int advice,
                                        struct cudaMemLocation location);
typedef cudaError_t (*cudaFuncSetAttribute_fn)(const void *func, int attr, int value);
typedef cudaError_t (*cudaFuncGetAttributes_fn)(void *attr, const void *func);
typedef cudaError_t (*cudaOccupancyMaxActiveBlocksPerMultiprocessor_fn)(
    int *numBlocks,
    const void *func,
    int blockSize,
    size_t dynamicSMemSize);
typedef cudaError_t (*cudaLaunchKernel_fn)(const void *func,
                                           dim3 gridDim,
                                           dim3 blockDim,
                                           void **args,
                                           size_t sharedMem,
                                           void *stream);
typedef cudaError_t (*cudaLaunchKernelExC_fn)(const cudaLaunchConfig_t *config,
                                              const void *func,
                                              void **args);
typedef cudaError_t (*cudaLaunchCooperativeKernel_fn)(const void *func,
                                                      dim3 gridDim,
                                                      dim3 blockDim,
                                                      void **args,
                                                      size_t sharedMem,
                                                      void *stream);
typedef cudaError_t (*cudaLaunchHostFunc_fn)(void *stream,
                                             cudaHostFn_t fn,
                                             void *userData);
typedef cudaError_t (*cudaLaunchHostFunc_v2_fn)(void *stream,
                                                cudaHostFn_t fn,
                                                void *userData,
                                                unsigned int syncMode);
typedef cudaError_t (*cudaOccupancyMaxPotentialBlockSize_fn)(int *minGridSize,
                                                             int *blockSize,
                                                             const void *func,
                                                             size_t dynamicSMemSize,
                                                             int blockSizeLimit);

struct cudaPointerAttributes {
    int type;
    int device;
    void *devicePointer;
    void *hostPointer;
};

#define CU_POINTER_ATTRIBUTE_MEMORY_TYPE 2
#define CU_POINTER_ATTRIBUTE_DEVICE_POINTER 3
#define CU_POINTER_ATTRIBUTE_HOST_POINTER 4
#define CU_POINTER_ATTRIBUTE_IS_MANAGED 8
#define CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL 9
#define CU_POINTER_ATTRIBUTE_RANGE_START_ADDR 11
#define CU_POINTER_ATTRIBUTE_RANGE_SIZE 12
#define CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE 17
#define CU_MEMORYTYPE_DEVICE 2
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3
#define cudaErrorNotSupported 801
#define cudaErrorStreamCaptureUnsupported 900
#define cudaStreamNonBlocking 1
#define cudaEventDisableTiming 2
#define cudaMemAttachGlobal 1
#define cudaHostAllocPortable 0x01
#define cudaHostAllocMapped 0x02
#define cudaHostRegisterPortable 0x01
#define cudaHostRegisterReadOnly 0x08
#define cudaDeviceScheduleSpin 0x01
#define cudaStreamCaptureStatusNone 0
#define cudaMemLocationTypeDevice 1
#define cudaMemAdviseSetAccessedBy 5
#define cudaDevAttrCooperativeLaunch 95
#define cudaDevAttrHostRegisterReadOnlySupported 113

static void *must_resolve(const char *name)
{
    void *sym = dlsym(RTLD_DEFAULT, name);
    if (!sym)
        fprintf(stderr, "managed_alloc: dlsym(%s) failed: %s\n", name, dlerror());
    return sym;
}

static int env_enabled(const char *name)
{
    const char *value = getenv(name);

    return value && value[0] != '\0' && strcmp(value, "0") != 0;
}

static int parse_u64_env(const char *name, uint64_t *out)
{
    const char *value = getenv(name);
    char *end = NULL;
    unsigned long long parsed;

    if (!value || value[0] == '\0')
        return 0;

    parsed = strtoull(value, &end, 0);
    if (!end || *end != '\0')
        return -1;

    *out = (uint64_t)parsed;
    return 1;
}

static void host_callback_mark(void *userData)
{
    int *flag = (int *)userData;

    if (flag)
        *flag = 1;
}

static uint64_t align_up_u64(uint64_t value, uint64_t alignment)
{
    if (alignment == 0)
        return value;
    return (value + alignment - 1) & ~(alignment - 1);
}

static struct cudaPitchedPtr make_test_pitched_ptr(void *ptr,
                                                   size_t pitch,
                                                   size_t xsize,
                                                   size_t ysize)
{
    struct cudaPitchedPtr p;

    p.ptr = ptr;
    p.pitch = pitch;
    p.xsize = xsize;
    p.ysize = ysize;
    return p;
}

static struct cudaExtent make_test_extent(size_t width, size_t height, size_t depth)
{
    struct cudaExtent e;

    e.width = width;
    e.height = height;
    e.depth = depth;
    return e;
}

static int pointer_in_range(CUdeviceptr ptr, uint64_t base, uint64_t length)
{
    return ptr >= base && ptr - base < length;
}

static int alloc_test_pointer(int use_runtime,
                              int use_async,
                              int use_managed_alloc,
                              cuMemAllocAsync_v2_fn driver_alloc_async_fn,
                              cuMemAlloc_v2_fn driver_alloc_fn,
                              cudaMalloc_fn runtime_alloc_fn,
                              cudaMallocManaged_fn runtime_alloc_managed_fn,
                              cudaMallocAsync_fn runtime_alloc_async_fn,
                              CUdeviceptr *ptr,
                              size_t size)
{
    if (use_runtime) {
        void *runtime_ptr = NULL;
        cudaError_t cr = use_managed_alloc
            ? runtime_alloc_managed_fn(&runtime_ptr, size, cudaMemAttachGlobal)
            : (use_async ? runtime_alloc_async_fn(&runtime_ptr, size, NULL)
                         : runtime_alloc_fn(&runtime_ptr, size));

        *ptr = (CUdeviceptr)(uintptr_t)runtime_ptr;
        if (cr != 0 || runtime_ptr == NULL) {
            fprintf(stderr,
                    "managed_alloc: %s returned %d ptr=%p\n",
                    use_managed_alloc ? "cudaMallocManaged"
                                      : (use_async ? "cudaMallocAsync" : "cudaMalloc"),
                    cr, runtime_ptr);
            return 1;
        }
        return 0;
    }

    CUresult r = use_async
        ? driver_alloc_async_fn(ptr, size, NULL)
        : driver_alloc_fn(ptr, size);
    if (r != 0 || *ptr == 0) {
        fprintf(stderr,
                "managed_alloc: %s returned %d ptr=0x%llx\n",
                use_async ? "cuMemAllocAsync_v2" : "cuMemAlloc_v2",
                r,
                *ptr);
        return 1;
    }
    return 0;
}

static int free_test_pointer(int use_runtime,
                             int use_async,
                             cuMemFreeAsync_v2_fn driver_free_async_fn,
                             cuMemFree_v2_fn driver_free_fn,
                             cudaFree_fn runtime_free_fn,
                             cudaFreeAsync_fn runtime_free_async_fn,
                             CUdeviceptr ptr,
                             const char *label)
{
    if (use_runtime) {
        cudaError_t cr = use_async
            ? runtime_free_async_fn((void *)(uintptr_t)ptr, NULL)
            : runtime_free_fn((void *)(uintptr_t)ptr);

        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: %s %s returned %d\n",
                    label,
                    use_async ? "cudaFreeAsync" : "cudaFree",
                    cr);
            return 1;
        }
        return 0;
    }

    CUresult r = use_async
        ? driver_free_async_fn(ptr, NULL)
        : driver_free_fn(ptr);
    if (r != 0) {
        fprintf(stderr,
                "managed_alloc: %s %s returned %d\n",
                label,
                use_async ? "cuMemFreeAsync_v2" : "cuMemFree_v2",
                r);
        return 1;
    }
    return 0;
}

static int check_cu_result(const char *label, CUresult got, CUresult expected)
{
    if (got != expected) {
        fprintf(stderr,
                "managed_alloc: %s returned %d, expected %d\n",
                label,
                got,
                expected);
        return 1;
    }
    return 0;
}

static int check_cuda_result(const char *label, cudaError_t got, cudaError_t expected)
{
    if (got != expected) {
        fprintf(stderr,
                "managed_alloc: %s returned %d, expected %d\n",
                label,
                got,
                expected);
        return 1;
    }
    return 0;
}

int main(void)
{
    cuDeviceGet_fn driver_device_get_fn;
    cuDeviceGetAttribute_fn driver_device_get_attribute_fn;
    cuDevicePrimaryCtxRetain_fn driver_primary_ctx_retain_fn;
    cuCtxSetCurrent_fn driver_ctx_set_current_fn;
    cuMemAlloc_v2_fn alloc_fn;
    cuMemFree_v2_fn free_fn;
    cuMemAllocAsync_v2_fn alloc_async_fn;
    cuMemFreeAsync_v2_fn free_async_fn;
    cuMemGetInfo_v2_fn driver_mem_get_info_fn;
    cuMemAddressReserve_fn vmm_address_reserve_fn;
    cuMemAddressFree_fn vmm_address_free_fn;
    cuMemCreate_fn vmm_create_fn;
    cuMemRelease_fn vmm_release_fn;
    cuMemMap_fn vmm_map_fn;
    cuMemUnmap_fn vmm_unmap_fn;
    cuMemSetAccess_fn vmm_set_access_fn;
    cuMemGetAllocationGranularity_fn vmm_granularity_fn;
    cuGetErrorString_fn driver_get_error_string_fn;
    cuGetErrorName_fn driver_get_error_name_fn;
    cuStreamBeginCapture_v2_fn driver_stream_begin_capture_fn;
    cudaMalloc_fn runtime_alloc_fn;
    cudaMallocManaged_fn runtime_alloc_managed_fn;
    cudaFree_fn runtime_free_fn;
    cudaMallocAsync_fn runtime_alloc_async_fn;
    cudaFreeAsync_fn runtime_free_async_fn;
    cudaGetDevice_fn runtime_get_device_fn;
    cudaSetDevice_fn runtime_set_device_fn;
    cudaSetDeviceFlags_fn runtime_set_device_flags_fn;
    cudaGetDeviceFlags_fn runtime_get_device_flags_fn;
    cudaGetDeviceCount_fn runtime_get_device_count_fn;
    cudaDeviceGetPCIBusId_fn runtime_device_get_pci_bus_id_fn;
    cudaRuntimeGetVersion_fn runtime_get_version_fn;
    cudaDriverGetVersion_fn runtime_driver_get_version_fn;
    cudaMemGetInfo_fn runtime_mem_get_info_fn;
    cudaDeviceGetAttribute_fn runtime_device_get_attribute_fn;
    cudaGetErrorString_fn runtime_get_error_string_fn;
    cudaGetErrorName_fn runtime_get_error_name_fn;
    cudaGetLastError_fn runtime_get_last_error_fn;
    cudaPeekAtLastError_fn runtime_peek_at_last_error_fn;
    cudaStreamCreate_fn runtime_stream_create_fn;
    cudaStreamCreateWithFlags_fn runtime_stream_create_with_flags_fn;
    cudaStreamDestroy_fn runtime_stream_destroy_fn;
    cudaStreamSynchronize_fn runtime_stream_synchronize_fn;
    cudaStreamWaitEvent_fn runtime_stream_wait_event_fn;
    cudaStreamIsCapturing_fn runtime_stream_is_capturing_fn;
    cudaEventCreateWithFlags_fn runtime_event_create_with_flags_fn;
    cudaEventRecord_fn runtime_event_record_fn;
    cudaEventSynchronize_fn runtime_event_synchronize_fn;
    cudaEventDestroy_fn runtime_event_destroy_fn;
    cudaStreamBeginCapture_fn stream_begin_capture_fn;
    cudaStreamEndCapture_fn stream_end_capture_fn;
    cudaGraphInstantiate_fn graph_instantiate_fn;
    cudaGraphLaunch_fn graph_launch_fn;
    cudaGraphExecUpdate_fn graph_exec_update_fn;
    cudaGraphGetNodes_fn graph_get_nodes_fn;
    cudaGraphNodeGetType_fn graph_node_get_type_fn;
    cudaGraphKernelNodeGetParams_fn graph_kernel_node_get_params_fn;
    cudaGraphKernelNodeSetParams_fn graph_kernel_node_set_params_fn;
    cudaGraphDestroy_fn graph_destroy_fn;
    cudaGraphExecDestroy_fn graph_exec_destroy_fn;
    cuMemcpyHtoD_v2_fn driver_memcpy_htod_fn;
    cuMemcpyDtoH_v2_fn driver_memcpy_dtoh_fn;
    cuMemcpyHtoDAsync_v2_fn driver_memcpy_htod_async_fn;
    cuMemcpyDtoHAsync_v2_fn driver_memcpy_dtoh_async_fn;
    cuMemcpy_v2_fn driver_memcpy_fn;
    cuMemcpyAsync_v2_fn driver_memcpy_async_fn;
    cudaMemcpy_fn memcpy_fn;
    cudaMemcpyAsync_fn memcpy_async_fn;
    cudaMemcpy2DAsync_fn memcpy_2d_async_fn;
    cudaMemcpyPeerAsync_fn memcpy_peer_async_fn;
    cudaMemcpy3DPeerAsync_fn memcpy_3d_peer_async_fn;
    cuMemsetD8_v2_fn driver_memset_d8_fn;
    cuMemsetD8Async_v2_fn driver_memset_d8_async_fn;
    cuMemsetD16_v2_fn driver_memset_d16_fn;
    cuMemsetD16Async_v2_fn driver_memset_d16_async_fn;
    cuMemsetD32_v2_fn driver_memset_d32_fn;
    cuMemsetD32Async_v2_fn driver_memset_d32_async_fn;
    cudaMemset_fn memset_fn;
    cudaMemsetAsync_fn memset_async_fn;
    cuIpcGetMemHandle_fn driver_ipc_get_fn;
    cudaIpcGetMemHandle_fn runtime_ipc_get_fn;
    cuMemGetAddressRange_v2_fn range_fn;
    cuPointerGetAttribute_fn attr_fn;
    cuPointerGetAttributes_fn attrs_fn;
    cuLaunchKernel_fn driver_launch_kernel_fn;
    cuLaunchKernelEx_fn driver_launch_kernel_ex_fn;
    cudaPointerGetAttributes_fn runtime_attrs_fn;
    cudaMallocHost_fn malloc_host_fn;
    cudaHostAlloc_fn host_alloc_fn;
    cudaFreeHost_fn free_host_fn;
    cudaHostGetDevicePointer_fn host_get_device_pointer_fn;
    cudaHostRegister_fn host_register_fn;
    cudaHostUnregister_fn host_unregister_fn;
    cudaDeviceCanAccessPeer_fn device_can_access_peer_fn;
    cudaDeviceEnablePeerAccess_fn device_enable_peer_access_fn;
    cudaDeviceDisablePeerAccess_fn device_disable_peer_access_fn;
    cudaMemAdvise_fn mem_advise_fn;
    cudaFuncSetAttribute_fn func_set_attribute_fn;
    cudaFuncGetAttributes_fn func_get_attributes_fn;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor_fn occupancy_fn;
    cudaOccupancyMaxPotentialBlockSize_fn occupancy_potential_fn;
    cudaLaunchKernel_fn runtime_launch_kernel_fn;
    cudaLaunchKernelExC_fn runtime_launch_kernel_ex_fn;
    cudaLaunchCooperativeKernel_fn runtime_launch_cooperative_kernel_fn;
    cudaLaunchHostFunc_fn runtime_launch_host_func_fn;
    cudaLaunchHostFunc_v2_fn runtime_launch_host_func_v2_fn;
    CUdeviceptr ptr = 0;
    CUresult r;
    const char *mode = getenv("POLARIS_SHIM_TEST_LEAK");
    int use_runtime = env_enabled("POLARIS_SHIM_TEST_RUNTIME_ALLOC");
    int use_async = env_enabled("POLARIS_SHIM_TEST_ASYNC_ALLOC") ||
        env_enabled("POLARIS_SHIM_TEST_DRIVER_ASYNC_ALLOC");
    int use_managed_alloc = env_enabled("POLARIS_SHIM_TEST_MANAGED_ALLOC");
    int test_filter = env_enabled("POLARIS_SHIM_TEST_FILTER");
    int test_large_window = env_enabled("POLARIS_SHIM_TEST_LARGE_WINDOW");
    int test_runtime_setup = env_enabled("POLARIS_SHIM_TEST_RUNTIME_SETUP");
    int test_host_apis = env_enabled("POLARIS_SHIM_TEST_HOST_APIS");
    int allow_zero_memset = env_enabled("POLARIS_SHIM_ALLOW_ZERO_MEMSET");
    uint64_t block_size = 2ULL * 1024ULL * 1024ULL;
    size_t primary_alloc_size = 1024 * 1024;
    size_t expected_range_size = 0;

    *(void **)(&driver_device_get_fn) = must_resolve("cuDeviceGet");
    *(void **)(&driver_device_get_attribute_fn) = must_resolve("cuDeviceGetAttribute");
    *(void **)(&driver_primary_ctx_retain_fn) = must_resolve("cuDevicePrimaryCtxRetain");
    *(void **)(&driver_ctx_set_current_fn) = must_resolve("cuCtxSetCurrent");
    *(void **)(&alloc_fn) = must_resolve("cuMemAlloc_v2");
    *(void **)(&free_fn) = must_resolve("cuMemFree_v2");
    *(void **)(&alloc_async_fn) = must_resolve("cuMemAllocAsync_v2");
    *(void **)(&free_async_fn) = must_resolve("cuMemFreeAsync_v2");
    *(void **)(&driver_mem_get_info_fn) = must_resolve("cuMemGetInfo_v2");
    *(void **)(&vmm_address_reserve_fn) = must_resolve("cuMemAddressReserve");
    *(void **)(&vmm_address_free_fn) = must_resolve("cuMemAddressFree");
    *(void **)(&vmm_create_fn) = must_resolve("cuMemCreate");
    *(void **)(&vmm_release_fn) = must_resolve("cuMemRelease");
    *(void **)(&vmm_map_fn) = must_resolve("cuMemMap");
    *(void **)(&vmm_unmap_fn) = must_resolve("cuMemUnmap");
    *(void **)(&vmm_set_access_fn) = must_resolve("cuMemSetAccess");
    *(void **)(&vmm_granularity_fn) = must_resolve("cuMemGetAllocationGranularity");
    *(void **)(&driver_get_error_string_fn) = must_resolve("cuGetErrorString");
    *(void **)(&driver_get_error_name_fn) = must_resolve("cuGetErrorName");
    *(void **)(&driver_stream_begin_capture_fn) = must_resolve("cuStreamBeginCapture_v2");
    *(void **)(&runtime_alloc_fn) = must_resolve("cudaMalloc");
    *(void **)(&runtime_alloc_managed_fn) = must_resolve("cudaMallocManaged");
    *(void **)(&runtime_free_fn) = must_resolve("cudaFree");
    *(void **)(&runtime_alloc_async_fn) = must_resolve("cudaMallocAsync");
    *(void **)(&runtime_free_async_fn) = must_resolve("cudaFreeAsync");
    *(void **)(&runtime_get_device_fn) = must_resolve("cudaGetDevice");
    *(void **)(&runtime_set_device_fn) = must_resolve("cudaSetDevice");
    *(void **)(&runtime_set_device_flags_fn) = must_resolve("cudaSetDeviceFlags");
    *(void **)(&runtime_get_device_flags_fn) = must_resolve("cudaGetDeviceFlags");
    *(void **)(&runtime_get_device_count_fn) = must_resolve("cudaGetDeviceCount");
    *(void **)(&runtime_device_get_pci_bus_id_fn) = must_resolve("cudaDeviceGetPCIBusId");
    *(void **)(&runtime_get_version_fn) = must_resolve("cudaRuntimeGetVersion");
    *(void **)(&runtime_driver_get_version_fn) = must_resolve("cudaDriverGetVersion");
    *(void **)(&runtime_mem_get_info_fn) = must_resolve("cudaMemGetInfo");
    *(void **)(&runtime_device_get_attribute_fn) = must_resolve("cudaDeviceGetAttribute");
    *(void **)(&runtime_get_error_string_fn) = must_resolve("cudaGetErrorString");
    *(void **)(&runtime_get_error_name_fn) = must_resolve("cudaGetErrorName");
    *(void **)(&runtime_get_last_error_fn) = must_resolve("cudaGetLastError");
    *(void **)(&runtime_peek_at_last_error_fn) = must_resolve("cudaPeekAtLastError");
    *(void **)(&runtime_stream_create_fn) = must_resolve("cudaStreamCreate");
    *(void **)(&runtime_stream_create_with_flags_fn) = must_resolve("cudaStreamCreateWithFlags");
    *(void **)(&runtime_stream_destroy_fn) = must_resolve("cudaStreamDestroy");
    *(void **)(&runtime_stream_synchronize_fn) = must_resolve("cudaStreamSynchronize");
    *(void **)(&runtime_stream_wait_event_fn) = must_resolve("cudaStreamWaitEvent");
    *(void **)(&runtime_stream_is_capturing_fn) = must_resolve("cudaStreamIsCapturing");
    *(void **)(&runtime_event_create_with_flags_fn) = must_resolve("cudaEventCreateWithFlags");
    *(void **)(&runtime_event_record_fn) = must_resolve("cudaEventRecord");
    *(void **)(&runtime_event_synchronize_fn) = must_resolve("cudaEventSynchronize");
    *(void **)(&runtime_event_destroy_fn) = must_resolve("cudaEventDestroy");
    *(void **)(&stream_begin_capture_fn) = must_resolve("cudaStreamBeginCapture");
    *(void **)(&stream_end_capture_fn) = must_resolve("cudaStreamEndCapture");
    *(void **)(&graph_instantiate_fn) = must_resolve("cudaGraphInstantiate");
    *(void **)(&graph_launch_fn) = must_resolve("cudaGraphLaunch");
    *(void **)(&graph_exec_update_fn) = must_resolve("cudaGraphExecUpdate");
    *(void **)(&graph_get_nodes_fn) = must_resolve("cudaGraphGetNodes");
    *(void **)(&graph_node_get_type_fn) = must_resolve("cudaGraphNodeGetType");
    *(void **)(&graph_kernel_node_get_params_fn) = must_resolve("cudaGraphKernelNodeGetParams");
    *(void **)(&graph_kernel_node_set_params_fn) = must_resolve("cudaGraphKernelNodeSetParams");
    *(void **)(&graph_destroy_fn) = must_resolve("cudaGraphDestroy");
    *(void **)(&graph_exec_destroy_fn) = must_resolve("cudaGraphExecDestroy");
    *(void **)(&driver_memcpy_htod_fn) = must_resolve("cuMemcpyHtoD_v2");
    *(void **)(&driver_memcpy_dtoh_fn) = must_resolve("cuMemcpyDtoH_v2");
    *(void **)(&driver_memcpy_htod_async_fn) = must_resolve("cuMemcpyHtoDAsync_v2");
    *(void **)(&driver_memcpy_dtoh_async_fn) = must_resolve("cuMemcpyDtoHAsync_v2");
    *(void **)(&driver_memcpy_fn) = must_resolve("cuMemcpy_v2");
    *(void **)(&driver_memcpy_async_fn) = must_resolve("cuMemcpyAsync_v2");
    *(void **)(&memcpy_fn) = must_resolve("cudaMemcpy");
    *(void **)(&memcpy_async_fn) = must_resolve("cudaMemcpyAsync");
    *(void **)(&memcpy_2d_async_fn) = must_resolve("cudaMemcpy2DAsync");
    *(void **)(&memcpy_peer_async_fn) = must_resolve("cudaMemcpyPeerAsync");
    *(void **)(&memcpy_3d_peer_async_fn) = must_resolve("cudaMemcpy3DPeerAsync");
    *(void **)(&driver_memset_d8_fn) = must_resolve("cuMemsetD8_v2");
    *(void **)(&driver_memset_d8_async_fn) = must_resolve("cuMemsetD8Async_v2");
    *(void **)(&driver_memset_d16_fn) = must_resolve("cuMemsetD16_v2");
    *(void **)(&driver_memset_d16_async_fn) = must_resolve("cuMemsetD16Async_v2");
    *(void **)(&driver_memset_d32_fn) = must_resolve("cuMemsetD32_v2");
    *(void **)(&driver_memset_d32_async_fn) = must_resolve("cuMemsetD32Async_v2");
    *(void **)(&memset_fn) = must_resolve("cudaMemset");
    *(void **)(&memset_async_fn) = must_resolve("cudaMemsetAsync");
    *(void **)(&driver_ipc_get_fn) = must_resolve("cuIpcGetMemHandle");
    *(void **)(&runtime_ipc_get_fn) = must_resolve("cudaIpcGetMemHandle");
    *(void **)(&range_fn) = must_resolve("cuMemGetAddressRange_v2");
    *(void **)(&attr_fn) = must_resolve("cuPointerGetAttribute");
    *(void **)(&attrs_fn) = must_resolve("cuPointerGetAttributes");
    *(void **)(&driver_launch_kernel_fn) = must_resolve("cuLaunchKernel");
    *(void **)(&driver_launch_kernel_ex_fn) = must_resolve("cuLaunchKernelEx");
    *(void **)(&runtime_attrs_fn) = must_resolve("cudaPointerGetAttributes");
    *(void **)(&malloc_host_fn) = must_resolve("cudaMallocHost");
    *(void **)(&host_alloc_fn) = must_resolve("cudaHostAlloc");
    *(void **)(&free_host_fn) = must_resolve("cudaFreeHost");
    *(void **)(&host_get_device_pointer_fn) = must_resolve("cudaHostGetDevicePointer");
    *(void **)(&host_register_fn) = must_resolve("cudaHostRegister");
    *(void **)(&host_unregister_fn) = must_resolve("cudaHostUnregister");
    *(void **)(&device_can_access_peer_fn) = must_resolve("cudaDeviceCanAccessPeer");
    *(void **)(&device_enable_peer_access_fn) = must_resolve("cudaDeviceEnablePeerAccess");
    *(void **)(&device_disable_peer_access_fn) = must_resolve("cudaDeviceDisablePeerAccess");
    *(void **)(&mem_advise_fn) = must_resolve("cudaMemAdvise");
    *(void **)(&func_set_attribute_fn) = must_resolve("cudaFuncSetAttribute");
    *(void **)(&func_get_attributes_fn) = must_resolve("cudaFuncGetAttributes");
    *(void **)(&occupancy_fn) = must_resolve("cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    *(void **)(&occupancy_potential_fn) = must_resolve("cudaOccupancyMaxPotentialBlockSize");
    *(void **)(&runtime_launch_kernel_fn) = must_resolve("cudaLaunchKernel");
    *(void **)(&runtime_launch_kernel_ex_fn) = must_resolve("cudaLaunchKernelExC");
    *(void **)(&runtime_launch_cooperative_kernel_fn) =
        must_resolve("cudaLaunchCooperativeKernel");
    *(void **)(&runtime_launch_host_func_fn) = must_resolve("cudaLaunchHostFunc");
    *(void **)(&runtime_launch_host_func_v2_fn) = must_resolve("cudaLaunchHostFunc_v2");
    if (!alloc_fn || !free_fn || !runtime_alloc_fn || !runtime_free_fn)
        return 1;
    if (test_runtime_setup) {
        CUdevice driver_device = -1;
        int device = -1;
        int count = 0;
        int version = 0;
        unsigned int device_flags = 0;
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const char *error_text = NULL;
        char pci_bus_id[64];
        cudaStream_t stream = NULL;
        cudaStream_t stream2 = NULL;
        cudaEvent_t event = NULL;
        cudaStreamCaptureStatus capture_status = -1;
        cudaError_t cr;

        if (!driver_device_get_fn ||
            !driver_device_get_attribute_fn ||
            !driver_primary_ctx_retain_fn ||
            !driver_ctx_set_current_fn ||
            !runtime_get_device_fn ||
            !runtime_set_device_fn ||
            !runtime_set_device_flags_fn ||
            !runtime_get_device_flags_fn ||
            !runtime_get_device_count_fn ||
            !runtime_device_get_pci_bus_id_fn ||
            !runtime_get_version_fn ||
            !runtime_driver_get_version_fn ||
            !runtime_mem_get_info_fn ||
            !runtime_device_get_attribute_fn ||
            !runtime_get_error_string_fn ||
            !runtime_get_error_name_fn ||
            !runtime_get_last_error_fn ||
            !runtime_peek_at_last_error_fn ||
            !runtime_stream_create_fn ||
            !runtime_stream_create_with_flags_fn ||
            !runtime_stream_destroy_fn ||
            !runtime_stream_synchronize_fn ||
            !runtime_stream_wait_event_fn ||
            !runtime_stream_is_capturing_fn ||
            !runtime_event_create_with_flags_fn ||
            !runtime_event_record_fn ||
            !runtime_event_synchronize_fn ||
            !runtime_event_destroy_fn ||
            !driver_mem_get_info_fn ||
            !driver_get_error_string_fn ||
            !driver_get_error_name_fn)
            return 1;

        r = driver_device_get_fn(&driver_device, 0);
        if (r != 0 || driver_device < 0) {
            fprintf(stderr,
                    "managed_alloc: cuDeviceGet returned %d device=%d\n",
                    r,
                    driver_device);
            return 1;
        }

        r = driver_device_get_attribute_fn(&version, cudaDevAttrCooperativeLaunch, driver_device);
        if (r != 0 || (version != 0 && version != 1)) {
            fprintf(stderr,
                    "managed_alloc: cuDeviceGetAttribute returned %d value=%d\n",
                    r,
                    version);
            return 1;
        }

        cr = runtime_get_version_fn(&version);
        if (cr != 0 || version <= 0) {
            fprintf(stderr,
                    "managed_alloc: cudaRuntimeGetVersion returned %d version=%d\n",
                    cr, version);
            return 1;
        }

        cr = runtime_driver_get_version_fn(&version);
        if (cr != 0 || version <= 0) {
            fprintf(stderr,
                    "managed_alloc: cudaDriverGetVersion returned %d version=%d\n",
                    cr, version);
            return 1;
        }

        cr = runtime_set_device_flags_fn(cudaDeviceScheduleSpin);
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaSetDeviceFlags returned %d\n",
                    cr);
            return 1;
        }

        cr = runtime_set_device_fn(0);
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaSetDevice returned %d\n",
                    cr);
            return 1;
        }

        cr = runtime_get_device_flags_fn(&device_flags);
        if (cr != 0 || (device_flags & cudaDeviceScheduleSpin) == 0) {
            fprintf(stderr,
                    "managed_alloc: cudaGetDeviceFlags returned %d flags=0x%x\n",
                    cr, device_flags);
            return 1;
        }

        cr = runtime_get_device_fn(&device);
        if (cr != 0 || device != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaGetDevice returned %d device=%d\n",
                    cr, device);
            return 1;
        }

        cr = runtime_get_device_count_fn(&count);
        if (cr != 0 || count <= 0) {
            fprintf(stderr,
                    "managed_alloc: cudaGetDeviceCount returned %d count=%d\n",
                    cr, count);
            return 1;
        }

        memset(pci_bus_id, 0, sizeof(pci_bus_id));
        cr = runtime_device_get_pci_bus_id_fn(pci_bus_id, (int)sizeof(pci_bus_id), 0);
        if (cr != 0 || pci_bus_id[0] == '\0') {
            fprintf(stderr,
                    "managed_alloc: cudaDeviceGetPCIBusId returned %d id=%s\n",
                    cr,
                    pci_bus_id);
            return 1;
        }

        cr = runtime_mem_get_info_fn(&free_bytes, &total_bytes);
        if (cr != 0 || total_bytes == 0 || free_bytes > total_bytes) {
            fprintf(stderr,
                    "managed_alloc: cudaMemGetInfo returned %d free=%zu total=%zu\n",
                    cr, free_bytes, total_bytes);
            return 1;
        }

        r = driver_mem_get_info_fn(&free_bytes, &total_bytes);
        if (r != 0 || total_bytes == 0 || free_bytes > total_bytes) {
            fprintf(stderr,
                    "managed_alloc: cuMemGetInfo_v2 returned %d free=%zu total=%zu\n",
                    r, free_bytes, total_bytes);
            return 1;
        }

        cr = runtime_device_get_attribute_fn(&version, cudaDevAttrCooperativeLaunch, 0);
        if (cr != 0 || (version != 0 && version != 1)) {
            fprintf(stderr,
                    "managed_alloc: cudaDeviceGetAttribute returned %d value=%d\n",
                    cr, version);
            return 1;
        }

        error_text = runtime_get_error_string_fn(0);
        if (!error_text || error_text[0] == '\0') {
            fprintf(stderr, "managed_alloc: cudaGetErrorString returned empty text\n");
            return 1;
        }

        error_text = runtime_get_error_name_fn(0);
        if (!error_text || error_text[0] == '\0') {
            fprintf(stderr, "managed_alloc: cudaGetErrorName returned empty text\n");
            return 1;
        }

        r = driver_get_error_string_fn(0, &error_text);
        if (r != 0 || !error_text || error_text[0] == '\0') {
            fprintf(stderr,
                    "managed_alloc: cuGetErrorString returned %d text=%p\n",
                    r, (void *)error_text);
            return 1;
        }

        r = driver_get_error_name_fn(0, &error_text);
        if (r != 0 || !error_text || error_text[0] == '\0') {
            fprintf(stderr,
                    "managed_alloc: cuGetErrorName returned %d text=%p\n",
                    r, (void *)error_text);
            return 1;
        }

        cr = runtime_peek_at_last_error_fn();
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaPeekAtLastError returned %d\n",
                    cr);
            return 1;
        }

        cr = runtime_get_last_error_fn();
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaGetLastError returned %d\n",
                    cr);
            return 1;
        }

        cr = runtime_stream_create_fn(&stream);
        if (cr != 0 || stream == NULL) {
            fprintf(stderr,
                    "managed_alloc: cudaStreamCreate returned %d stream=%p\n",
                    cr, stream);
            return 1;
        }

        cr = runtime_stream_synchronize_fn(stream);
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaStreamSynchronize returned %d\n",
                    cr);
            return 1;
        }

        cr = runtime_event_create_with_flags_fn(&event, cudaEventDisableTiming);
        if (cr != 0 || event == NULL) {
            fprintf(stderr,
                    "managed_alloc: cudaEventCreateWithFlags returned %d event=%p\n",
                    cr, event);
            return 1;
        }

        cr = runtime_event_record_fn(event, stream);
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaEventRecord returned %d\n",
                    cr);
            return 1;
        }

        cr = runtime_stream_wait_event_fn(stream, event, 0);
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaStreamWaitEvent returned %d\n",
                    cr);
            return 1;
        }

        cr = runtime_stream_is_capturing_fn(stream, &capture_status);
        if (cr != 0 || capture_status != cudaStreamCaptureStatusNone) {
            fprintf(stderr,
                    "managed_alloc: cudaStreamIsCapturing returned %d status=%d\n",
                    cr, capture_status);
            return 1;
        }

        cr = runtime_event_synchronize_fn(event);
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaEventSynchronize returned %d\n",
                    cr);
            return 1;
        }

        cr = runtime_event_destroy_fn(event);
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaEventDestroy returned %d\n",
                    cr);
            return 1;
        }
        event = NULL;

        cr = runtime_stream_destroy_fn(stream);
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaStreamDestroy returned %d\n",
                    cr);
            return 1;
        }
        stream = NULL;

        cr = runtime_stream_create_with_flags_fn(&stream2, cudaStreamNonBlocking);
        if (cr != 0 || stream2 == NULL) {
            fprintf(stderr,
                    "managed_alloc: cudaStreamCreateWithFlags returned %d stream=%p\n",
                    cr, stream2);
            return 1;
        }

        cr = runtime_stream_destroy_fn(stream2);
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaStreamDestroy(nonblocking) returned %d\n",
                    cr);
            return 1;
        }
    }
    if (test_host_apis) {
        void *host_ptr = NULL;
        void *mapped_host = NULL;
        void *device_ptr = NULL;
        char stack_buf[4096];
        cudaStream_t host_stream = NULL;
        int host_callback_seen = 0;
        int readonly_supported = 0;
        int can_access_peer = 0;
        cudaError_t cr;

        if (!malloc_host_fn ||
            !host_alloc_fn ||
            !free_host_fn ||
            !host_get_device_pointer_fn ||
            !host_register_fn ||
            !host_unregister_fn ||
            !runtime_device_get_attribute_fn ||
            !device_can_access_peer_fn ||
            !device_enable_peer_access_fn ||
            !device_disable_peer_access_fn ||
            !runtime_stream_create_with_flags_fn ||
            !runtime_stream_synchronize_fn ||
            !runtime_stream_destroy_fn ||
            !runtime_launch_host_func_fn ||
            !runtime_launch_host_func_v2_fn ||
            !func_set_attribute_fn ||
            !func_get_attributes_fn ||
            !occupancy_fn ||
            !occupancy_potential_fn)
            return 1;

        cr = malloc_host_fn(&host_ptr, 4096);
        if (cr != 0 || host_ptr == NULL) {
            fprintf(stderr,
                    "managed_alloc: cudaMallocHost returned %d ptr=%p\n",
                    cr, host_ptr);
            return 1;
        }
        cr = free_host_fn(host_ptr);
        if (cr != 0) {
            fprintf(stderr, "managed_alloc: cudaFreeHost returned %d\n", cr);
            return 1;
        }
        host_ptr = NULL;

        cr = host_alloc_fn(&mapped_host, 4096, cudaHostAllocPortable | cudaHostAllocMapped);
        if (cr != 0 || mapped_host == NULL) {
            fprintf(stderr,
                    "managed_alloc: cudaHostAlloc returned %d ptr=%p\n",
                    cr, mapped_host);
            return 1;
        }
        cr = host_get_device_pointer_fn(&device_ptr, mapped_host, 0);
        if (cr != 0 || device_ptr == NULL) {
            fprintf(stderr,
                    "managed_alloc: cudaHostGetDevicePointer returned %d ptr=%p\n",
                    cr, device_ptr);
            return 1;
        }
        cr = free_host_fn(mapped_host);
        if (cr != 0) {
            fprintf(stderr, "managed_alloc: cudaFreeHost(mapped) returned %d\n", cr);
            return 1;
        }
        mapped_host = NULL;

        cr = runtime_device_get_attribute_fn(&readonly_supported,
                                             cudaDevAttrHostRegisterReadOnlySupported,
                                             0);
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: host-register attribute returned %d\n",
                    cr);
            return 1;
        }
        cr = host_register_fn(stack_buf,
                              sizeof(stack_buf),
                              cudaHostRegisterPortable |
                                  (readonly_supported ? cudaHostRegisterReadOnly : 0));
        if (cr != 0) {
            fprintf(stderr, "managed_alloc: cudaHostRegister returned %d\n", cr);
            return 1;
        }
        cr = host_unregister_fn(stack_buf);
        if (cr != 0) {
            fprintf(stderr, "managed_alloc: cudaHostUnregister returned %d\n", cr);
            return 1;
        }

        cr = device_can_access_peer_fn(&can_access_peer, 0, 0);
        if (cr != 0 || can_access_peer != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaDeviceCanAccessPeer(self) returned %d value=%d\n",
                    cr, can_access_peer);
            return 1;
        }

        cr = runtime_stream_create_with_flags_fn(&host_stream, cudaStreamNonBlocking);
        if (cr != 0 || host_stream == NULL) {
            fprintf(stderr,
                    "managed_alloc: cudaStreamCreateWithFlags(host callback) returned %d stream=%p\n",
                    cr,
                    host_stream);
            return 1;
        }
        cr = runtime_launch_host_func_fn(host_stream, host_callback_mark, &host_callback_seen);
        if (cr != 0) {
            fprintf(stderr, "managed_alloc: cudaLaunchHostFunc returned %d\n", cr);
            return 1;
        }
        cr = runtime_stream_synchronize_fn(host_stream);
        if (cr != 0 || host_callback_seen != 1) {
            fprintf(stderr,
                    "managed_alloc: cudaStreamSynchronize(host callback) returned %d seen=%d\n",
                    cr,
                    host_callback_seen);
            return 1;
        }
        cr = runtime_stream_destroy_fn(host_stream);
        if (cr != 0) {
            fprintf(stderr, "managed_alloc: cudaStreamDestroy(host callback) returned %d\n", cr);
            return 1;
        }
    }
    if (env_enabled("POLARIS_SHIM_TEST_ASYNC_ALLOC")) {
        use_runtime = 1;
        if (!runtime_alloc_async_fn || !runtime_free_async_fn)
            return 1;
    }
    if (env_enabled("POLARIS_SHIM_TEST_DRIVER_ASYNC_ALLOC")) {
        use_runtime = 0;
        if (!alloc_async_fn || !free_async_fn)
            return 1;
    }
    if (use_managed_alloc) {
        use_runtime = 1;
        use_async = 0;
        if (!runtime_alloc_managed_fn)
            return 1;
    }
    if (test_filter) {
        uint64_t managed_base = 0;
        uint64_t managed_length = 0;
        uint64_t min_managed = 0;
        void *fallback_ptr = NULL;
        CUdeviceptr fallback_device_ptr;
        cudaError_t cr;

        use_runtime = 1;
        if (parse_u64_env("POLARIS_SHIM_MANAGED_BASE", &managed_base) <= 0 ||
            parse_u64_env("POLARIS_SHIM_MANAGED_LENGTH", &managed_length) <= 0 ||
            parse_u64_env("POLARIS_SHIM_MIN_MANAGED_ALLOC", &min_managed) <= 0 ||
            min_managed <= 1024 * 1024 ||
            min_managed > SIZE_MAX) {
            fprintf(stderr,
                    "managed_alloc: filter test requires MANAGED_BASE, MANAGED_LENGTH, "
                    "and 1MiB < MIN_MANAGED_ALLOC <= SIZE_MAX\n");
            return 1;
        }
        primary_alloc_size = (size_t)min_managed;

        cr = runtime_alloc_fn(&fallback_ptr, 1024 * 1024);
        fallback_device_ptr = (CUdeviceptr)(uintptr_t)fallback_ptr;
        if (cr != 0 || fallback_ptr == NULL) {
            fprintf(stderr,
                    "managed_alloc: below-threshold cudaMalloc returned %d ptr=%p\n",
                    cr, fallback_ptr);
            return 1;
        }
        if (pointer_in_range(fallback_device_ptr, managed_base, managed_length)) {
            fprintf(stderr,
                    "managed_alloc: below-threshold allocation was managed ptr=0x%llx\n",
                    fallback_device_ptr);
            return 1;
        }
        cr = runtime_free_fn(fallback_ptr);
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: below-threshold cudaFree returned %d\n",
                    cr);
            return 1;
        }
    }
    if (test_large_window) {
        (void)parse_u64_env("POLARIS_SHIM_BLOCK_SIZE", &block_size);
        if (block_size == 0 || block_size > SIZE_MAX / 3) {
            fprintf(stderr,
                    "managed_alloc: large-window test requires usable POLARIS_SHIM_BLOCK_SIZE\n");
            return 1;
        }
        primary_alloc_size = (size_t)(block_size * 3);
    }
    if (!test_large_window)
        (void)parse_u64_env("POLARIS_SHIM_BLOCK_SIZE", &block_size);
    if (block_size == 0 || primary_alloc_size > SIZE_MAX - (size_t)block_size + 1) {
        fprintf(stderr,
                "managed_alloc: cannot compute expected rounded allocation size\n");
        return 1;
    }
    expected_range_size = (size_t)align_up_u64((uint64_t)primary_alloc_size, block_size);

    if (alloc_test_pointer(use_runtime,
                           use_async,
                           use_managed_alloc,
                           alloc_async_fn,
                           alloc_fn,
                           runtime_alloc_fn,
                           runtime_alloc_managed_fn,
                           runtime_alloc_async_fn,
                           &ptr,
                           primary_alloc_size))
        return 1;

    if (env_enabled("POLARIS_SHIM_TEST_ATTRS")) {
        unsigned int memory_type = 0;
        CUdeviceptr device_ptr = 0;
        CUdeviceptr range_start = 0;
        void *host_ptr = (void *)1;
        unsigned int is_managed = 1;
        int device_ordinal = -1;
        void *mempool_handle = (void *)1;
        CUdeviceptr address_range_start = 0;
        size_t range_size = 0;
        size_t address_range_size = 0;
        struct cudaPointerAttributes runtime_attrs;
        int attrs[] = {
            CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
            CU_POINTER_ATTRIBUTE_HOST_POINTER,
            CU_POINTER_ATTRIBUTE_IS_MANAGED,
            CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
            CU_POINTER_ATTRIBUTE_RANGE_SIZE,
            CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
        };
        void *datas[] = {
            &memory_type,
            &device_ptr,
            &host_ptr,
            &is_managed,
            &device_ordinal,
            &range_start,
            &range_size,
            &mempool_handle,
        };

        if (!range_fn || !attr_fn || !attrs_fn || !runtime_attrs_fn)
            return 1;

        r = attr_fn(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr);
        if (r != 0 || memory_type != CU_MEMORYTYPE_DEVICE) {
            fprintf(stderr,
                    "managed_alloc: memory type attr returned %d value=%u\n",
                    r, memory_type);
            return 1;
        }

        r = attr_fn(&device_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr + 4096);
        if (r != 0 || device_ptr != ptr + 4096) {
            fprintf(stderr,
                    "managed_alloc: device pointer attr returned %d value=0x%llx\n",
                    r, device_ptr);
            return 1;
        }

        r = attr_fn(&host_ptr, CU_POINTER_ATTRIBUTE_HOST_POINTER, ptr + 4096);
        if (r != 0 || host_ptr != NULL) {
            fprintf(stderr,
                    "managed_alloc: host pointer attr returned %d value=%p\n",
                    r, host_ptr);
            return 1;
        }

        r = attr_fn(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, ptr + 4096);
        if (r != 0 || is_managed != 0) {
            fprintf(stderr,
                    "managed_alloc: is-managed attr returned %d value=%u\n",
                    r, is_managed);
            return 1;
        }

        r = attr_fn(&device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr + 4096);
        if (r != 0 || device_ordinal != 0) {
            fprintf(stderr,
                    "managed_alloc: device ordinal attr returned %d value=%d\n",
                    r, device_ordinal);
            return 1;
        }

        r = attr_fn(&mempool_handle, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, ptr + 4096);
        if (r != 0 || mempool_handle != NULL) {
            fprintf(stderr,
                    "managed_alloc: mempool handle attr returned %d value=%p\n",
                    r, mempool_handle);
            return 1;
        }

        memory_type = 0;
        device_ptr = 0;
        host_ptr = (void *)1;
        is_managed = 1;
        device_ordinal = -1;
        range_start = 0;
        range_size = 0;
        mempool_handle = (void *)1;
        r = attrs_fn(8, attrs, datas, ptr + 4096);
        if (r != 0 ||
            memory_type != CU_MEMORYTYPE_DEVICE ||
            device_ptr != ptr + 4096 ||
            host_ptr != NULL ||
            is_managed != 0 ||
            device_ordinal != 0 ||
            range_start != ptr ||
            range_size != expected_range_size ||
            mempool_handle != NULL) {
            fprintf(stderr,
                    "managed_alloc: batched attrs returned %d type=%u dev=0x%llx host=%p managed=%u ordinal=%d start=0x%llx size=%zu mempool=%p\n",
                    r,
                    memory_type,
                    device_ptr,
                    host_ptr,
                    is_managed,
                    device_ordinal,
                    range_start,
                    range_size,
                    mempool_handle);
            return 1;
        }

        r = range_fn(&address_range_start, &address_range_size, ptr + 4096);
        if (r != 0 ||
            address_range_start != ptr ||
            address_range_size != expected_range_size) {
            fprintf(stderr,
                    "managed_alloc: address range returned %d start=0x%llx size=%zu\n",
                    r, address_range_start, address_range_size);
            return 1;
        }

        r = range_fn(NULL, &address_range_size, ptr + 4096);
        if (r != 0 || address_range_size != expected_range_size) {
            fprintf(stderr,
                    "managed_alloc: address range size-only returned %d size=%zu\n",
                    r, address_range_size);
            return 1;
        }

        r = range_fn(&address_range_start, NULL, ptr + 4096);
        if (r != 0 || address_range_start != ptr) {
            fprintf(stderr,
                    "managed_alloc: address range base-only returned %d start=0x%llx\n",
                    r, address_range_start);
            return 1;
        }

        memset(&runtime_attrs, 0, sizeof(runtime_attrs));
        r = runtime_attrs_fn(&runtime_attrs, (const void *)(uintptr_t)(ptr + 4096));
        if (r != 0 ||
            runtime_attrs.type != CU_MEMORYTYPE_DEVICE ||
            runtime_attrs.device != 0 ||
            runtime_attrs.devicePointer != (void *)(uintptr_t)(ptr + 4096) ||
            runtime_attrs.hostPointer != NULL) {
            fprintf(stderr,
                    "managed_alloc: runtime attrs returned %d type=%d device=%d dev_ptr=%p host_ptr=%p\n",
                    r,
                    runtime_attrs.type,
                    runtime_attrs.device,
                    runtime_attrs.devicePointer,
                    runtime_attrs.hostPointer);
            return 1;
        }
    }

    if (env_enabled("POLARIS_SHIM_TEST_ADVISE")) {
        struct cudaMemLocation location;
        cudaError_t cr;

        if (!mem_advise_fn)
            return 1;
        location.type = cudaMemLocationTypeDevice;
        location.id = 0;
        cr = mem_advise_fn((const void *)(uintptr_t)ptr,
                           primary_alloc_size,
                           cudaMemAdviseSetAccessedBy,
                           location);
        if (cr != 0) {
            fprintf(stderr,
                    "managed_alloc: cudaMemAdvise returned %d\n",
                    cr);
            return 1;
        }
    }

    if (env_enabled("POLARIS_SHIM_TEST_MEMCPY")) {
        char host_buf[128];
        cudaError_t cr;
        size_t i;

        if (!driver_memcpy_htod_fn ||
            !driver_memcpy_dtoh_fn ||
            !driver_memcpy_htod_async_fn ||
            !driver_memcpy_dtoh_async_fn ||
            !driver_memcpy_fn ||
            !driver_memcpy_async_fn ||
            !memcpy_fn ||
            !memcpy_async_fn ||
            !memcpy_2d_async_fn ||
            !memcpy_peer_async_fn ||
            !memcpy_3d_peer_async_fn)
            return 1;

        for (i = 0; i < sizeof(host_buf); i++)
            host_buf[i] = (char)(0x30 + (i % 67));

        r = driver_memcpy_htod_fn(ptr, host_buf, sizeof(host_buf));
        if (check_cu_result("cuMemcpyHtoD_v2", r, cudaErrorNotSupported))
            return 1;

        r = driver_memcpy_dtoh_fn(host_buf, ptr, sizeof(host_buf));
        if (check_cu_result("cuMemcpyDtoH_v2", r, cudaErrorNotSupported))
            return 1;

        for (i = 0; i < sizeof(host_buf); i++)
            host_buf[i] = (char)(0x51 + (i % 41));
        r = driver_memcpy_htod_async_fn(ptr, host_buf, sizeof(host_buf), NULL);
        if (check_cu_result("cuMemcpyHtoDAsync_v2", r, cudaErrorNotSupported))
            return 1;

        r = driver_memcpy_dtoh_async_fn(host_buf, ptr, sizeof(host_buf), NULL);
        if (check_cu_result("cuMemcpyDtoHAsync_v2", r, cudaErrorNotSupported))
            return 1;

        r = driver_memcpy_fn(ptr, (CUdeviceptr)(uintptr_t)host_buf, sizeof(host_buf));
        if (check_cu_result("cuMemcpy_v2 dst", r, cudaErrorNotSupported))
            return 1;

        r = driver_memcpy_fn((CUdeviceptr)(uintptr_t)host_buf, ptr, sizeof(host_buf));
        if (check_cu_result("cuMemcpy_v2 src", r, cudaErrorNotSupported))
            return 1;

        r = driver_memcpy_async_fn(ptr,
                                   (CUdeviceptr)(uintptr_t)host_buf,
                                   sizeof(host_buf),
                                   NULL);
        if (check_cu_result("cuMemcpyAsync_v2", r, cudaErrorNotSupported))
            return 1;

        for (i = 0; i < sizeof(host_buf); i++)
            host_buf[i] = (char)(0x12 + (i % 101));
        cr = memcpy_fn((void *)(uintptr_t)ptr,
                       host_buf,
                       sizeof(host_buf),
                       cudaMemcpyHostToDevice);
        if (check_cuda_result("cudaMemcpy H2D", cr, cudaErrorNotSupported))
            return 1;

        cr = memcpy_fn(host_buf,
                       (const void *)(uintptr_t)ptr,
                       sizeof(host_buf),
                       cudaMemcpyDeviceToHost);
        if (check_cuda_result("cudaMemcpy D2H", cr, cudaErrorNotSupported))
            return 1;

        for (i = 0; i < sizeof(host_buf); i++)
            host_buf[i] = (char)(0x22 + (i % 53));
        cr = memcpy_async_fn((void *)(uintptr_t)ptr,
                             host_buf,
                             sizeof(host_buf),
                             cudaMemcpyHostToDevice,
                             NULL);
        if (check_cuda_result("cudaMemcpyAsync H2D", cr, cudaErrorNotSupported))
            return 1;

        cr = memcpy_2d_async_fn((void *)(uintptr_t)ptr,
                                8,
                                host_buf,
                                8,
                                8,
                                2,
                                cudaMemcpyHostToDevice,
                                NULL);
        if (check_cuda_result("cudaMemcpy2DAsync", cr, cudaErrorNotSupported))
            return 1;

        cr = memcpy_peer_async_fn((void *)(uintptr_t)ptr,
                                  0,
                                  host_buf,
                                  0,
                                  sizeof(host_buf),
                                  NULL);
        if (check_cuda_result("cudaMemcpyPeerAsync", cr, cudaErrorNotSupported))
            return 1;

        struct cudaMemcpy3DPeerParms peer_params;
        memset(&peer_params, 0, sizeof(peer_params));
        peer_params.dstDevice = 0;
        peer_params.dstPtr = make_test_pitched_ptr((void *)(uintptr_t)ptr, 8, 8, 2);
        peer_params.srcDevice = 0;
        peer_params.srcPtr = make_test_pitched_ptr(host_buf, 8, 8, 2);
        peer_params.extent = make_test_extent(8, 2, 1);
        cr = memcpy_3d_peer_async_fn(&peer_params, NULL);
        if (check_cuda_result("cudaMemcpy3DPeerAsync", cr, cudaErrorNotSupported))
            return 1;
    }

    if (env_enabled("POLARIS_SHIM_TEST_MEMSET")) {
        cudaError_t cr;
        CUresult zero_expected = allow_zero_memset ? 0 : cudaErrorNotSupported;
        cudaError_t runtime_zero_expected = allow_zero_memset ? 0 : cudaErrorNotSupported;

        if (!driver_memset_d8_fn ||
            !driver_memset_d8_async_fn ||
            !driver_memset_d16_fn ||
            !driver_memset_d16_async_fn ||
            !driver_memset_d32_fn ||
            !driver_memset_d32_async_fn ||
            !memset_fn ||
            !memset_async_fn)
            return 1;

        r = driver_memset_d8_fn(ptr, 0x5a, 16);
        if (check_cu_result("cuMemsetD8_v2", r, cudaErrorNotSupported))
            return 1;

        r = driver_memset_d8_async_fn(ptr, 0x6b, 16, NULL);
        if (check_cu_result("cuMemsetD8Async_v2", r, cudaErrorNotSupported))
            return 1;

        r = driver_memset_d8_fn(ptr, 0, 16);
        if (check_cu_result("cuMemsetD8_v2 zero", r, zero_expected))
            return 1;

        r = driver_memset_d8_async_fn(ptr, 0, 16, NULL);
        if (check_cu_result("cuMemsetD8Async_v2 zero", r, zero_expected))
            return 1;

        r = driver_memset_d16_fn(ptr, 0, 8);
        if (check_cu_result("cuMemsetD16_v2", r, zero_expected))
            return 1;

        r = driver_memset_d16_async_fn(ptr, 0, 8, NULL);
        if (check_cu_result("cuMemsetD16Async_v2", r, zero_expected))
            return 1;

        r = driver_memset_d32_fn(ptr, 0, 4);
        if (check_cu_result("cuMemsetD32_v2", r, zero_expected))
            return 1;

        r = driver_memset_d32_async_fn(ptr, 0, 4, NULL);
        if (check_cu_result("cuMemsetD32Async_v2", r, zero_expected))
            return 1;

        cr = memset_fn((void *)(uintptr_t)ptr, 0x7c, 16);
        if (check_cuda_result("cudaMemset", cr, cudaErrorNotSupported))
            return 1;

        cr = memset_async_fn((void *)(uintptr_t)ptr, 0x2d, 16, NULL);
        if (check_cuda_result("cudaMemsetAsync", cr, cudaErrorNotSupported))
            return 1;

        cr = memset_fn((void *)(uintptr_t)ptr, 0, 16);
        if (check_cuda_result("cudaMemset zero", cr, runtime_zero_expected))
            return 1;

        cr = memset_async_fn((void *)(uintptr_t)ptr, 0, 16, NULL);
        if (check_cuda_result("cudaMemsetAsync zero", cr, runtime_zero_expected))
            return 1;
    }

    if (env_enabled("POLARIS_SHIM_TEST_IPC")) {
        CUipcMemHandle driver_handle;
        cudaIpcMemHandle_t runtime_handle;
        cudaError_t cr;

        if (!driver_ipc_get_fn || !runtime_ipc_get_fn)
            return 1;

        r = driver_ipc_get_fn(&driver_handle, ptr);
        if (r != cudaErrorNotSupported) {
            fprintf(stderr,
                    "managed_alloc: cuIpcGetMemHandle returned %d, expected %d\n",
                    r,
                    cudaErrorNotSupported);
            return 1;
        }

        cr = runtime_ipc_get_fn(&runtime_handle, (void *)(uintptr_t)ptr);
        if (cr != cudaErrorNotSupported) {
            fprintf(stderr,
                    "managed_alloc: cudaIpcGetMemHandle returned %d, expected %d\n",
                    cr,
                    cudaErrorNotSupported);
            return 1;
        }
    }

    if (env_enabled("POLARIS_SHIM_TEST_GRAPH")) {
        cudaError_t cr;
        CUresult driver_result;
        cudaGraph_t graph = (cudaGraph_t)(uintptr_t)0x1;

        if (!stream_begin_capture_fn ||
            !stream_end_capture_fn ||
            !graph_instantiate_fn ||
            !graph_launch_fn ||
            !graph_exec_update_fn ||
            !graph_get_nodes_fn ||
            !graph_node_get_type_fn ||
            !graph_kernel_node_get_params_fn ||
            !graph_kernel_node_set_params_fn ||
            !graph_destroy_fn ||
            !graph_exec_destroy_fn ||
            !driver_stream_begin_capture_fn)
            return 1;

        cr = stream_begin_capture_fn(NULL, 0);
        if (cr != cudaErrorStreamCaptureUnsupported) {
            fprintf(stderr,
                    "managed_alloc: cudaStreamBeginCapture returned %d, expected %d\n",
                    cr,
                    cudaErrorStreamCaptureUnsupported);
            return 1;
        }

        cr = stream_end_capture_fn(NULL, &graph);
        if (cr != cudaErrorStreamCaptureUnsupported || graph != NULL) {
            fprintf(stderr,
                    "managed_alloc: cudaStreamEndCapture returned %d graph=%p, "
                    "expected %d and NULL graph\n",
                    cr,
                    graph,
                    cudaErrorStreamCaptureUnsupported);
            return 1;
        }

        cr = graph_launch_fn(NULL, NULL);
        if (cr != cudaErrorStreamCaptureUnsupported) {
            fprintf(stderr,
                    "managed_alloc: cudaGraphLaunch returned %d, expected %d\n",
                    cr,
                    cudaErrorStreamCaptureUnsupported);
            return 1;
        }

        driver_result = driver_stream_begin_capture_fn(NULL, 0);
        if (driver_result != cudaErrorStreamCaptureUnsupported) {
            fprintf(stderr,
                    "managed_alloc: cuStreamBeginCapture_v2 returned %d, expected %d\n",
                    driver_result,
                    cudaErrorStreamCaptureUnsupported);
            return 1;
        }
    }

    if (env_enabled("POLARIS_SHIM_TEST_VMM")) {
        if (!vmm_address_reserve_fn ||
            !vmm_address_free_fn ||
            !vmm_create_fn ||
            !vmm_release_fn ||
            !vmm_map_fn ||
            !vmm_unmap_fn ||
            !vmm_set_access_fn ||
            !vmm_granularity_fn)
            return 1;

        r = vmm_address_reserve_fn(NULL, 0, 0, 0, 0);
        if (r != 1) {
            fprintf(stderr,
                    "managed_alloc: cuMemAddressReserve invalid probe returned %d, "
                    "expected 1\n",
                    r);
            return 1;
        }
    }

    if (env_enabled("POLARIS_SHIM_TEST_LAUNCH")) {
        if (!runtime_launch_kernel_fn ||
            !runtime_launch_kernel_ex_fn ||
            !runtime_launch_cooperative_kernel_fn ||
            !runtime_launch_host_func_fn ||
            !runtime_launch_host_func_v2_fn ||
            !driver_launch_kernel_fn ||
            !driver_launch_kernel_ex_fn)
            return 1;
    }

    if (mode && strcmp(mode, "0") != 0) {
        fprintf(stderr, "managed_alloc: intentionally leaking ptr=0x%llx\n", ptr);
        return 0;
    }

    if (env_enabled("POLARIS_SHIM_TEST_REUSE")) {
        CUdeviceptr ptr2 = 0;
        CUdeviceptr ptr3 = 0;

        if (free_test_pointer(use_runtime,
                              use_async,
                              free_async_fn,
                              free_fn,
                              runtime_free_fn,
                              runtime_free_async_fn,
                              ptr,
                              "first free"))
            return 1;

        if (alloc_test_pointer(use_runtime,
                               use_async,
                               use_managed_alloc,
                               alloc_async_fn,
                               alloc_fn,
                               runtime_alloc_fn,
                               runtime_alloc_managed_fn,
                               runtime_alloc_async_fn,
                               &ptr2,
                               1024 * 1024))
            return 1;
        if (ptr2 != ptr) {
            fprintf(stderr,
                    "managed_alloc: expected token-span reuse ptr=0x%llx got=0x%llx\n",
                    ptr, ptr2);
            return 1;
        }

        if (use_runtime) {
            void *runtime_ptr3 = NULL;
            cudaError_t cr = use_async
                ? runtime_alloc_async_fn(&runtime_ptr3, 3 * 1024 * 1024, NULL)
                : (use_managed_alloc
                       ? runtime_alloc_managed_fn(&runtime_ptr3,
                                                  3 * 1024 * 1024,
                                                  cudaMemAttachGlobal)
                       : runtime_alloc_fn(&runtime_ptr3, 3 * 1024 * 1024));

            ptr3 = (CUdeviceptr)(uintptr_t)runtime_ptr3;
            if (cr == 0 || runtime_ptr3 != NULL) {
                fprintf(stderr,
                        "managed_alloc: oversize %s unexpectedly returned %d ptr=%p\n",
                        use_managed_alloc ? "cudaMallocManaged"
                                          : (use_async ? "cudaMallocAsync" : "cudaMalloc"),
                        cr, runtime_ptr3);
                return 1;
            }
        } else {
            r = use_async
                ? alloc_async_fn(&ptr3, 3 * 1024 * 1024, NULL)
                : alloc_fn(&ptr3, 3 * 1024 * 1024);
            if (r == 0 || ptr3 != 0) {
                fprintf(stderr,
                        "managed_alloc: oversize %s unexpectedly returned %d ptr=0x%llx\n",
                        use_async ? "cuMemAllocAsync_v2" : "cuMemAlloc_v2",
                        r, ptr3);
                return 1;
            }
        }

        if (free_test_pointer(use_runtime,
                              use_async,
                              free_async_fn,
                              free_fn,
                              runtime_free_fn,
                              runtime_free_async_fn,
                              ptr2,
                              "second free"))
            return 1;

        fprintf(stderr,
                "managed_alloc: %s reuse/exhaustion passed ptr=0x%llx\n",
                use_async ? (use_runtime ? "runtime-async" : "driver-async")
                          : (use_runtime ? (use_managed_alloc ? "runtime-managed" : "runtime")
                                         : "driver"),
                ptr2);
        return 0;
    }

    if (free_test_pointer(use_runtime,
                          use_async,
                          free_async_fn,
                          free_fn,
                          runtime_free_fn,
                          runtime_free_async_fn,
                          ptr,
                          "final free"))
        return 1;

    fprintf(stderr,
            "managed_alloc: %s managed alloc/free passed ptr=0x%llx\n",
            use_async ? (use_runtime ? "runtime-async" : "driver-async")
                      : (use_runtime ? (use_managed_alloc ? "runtime-managed" : "runtime")
                                     : "driver"),
            ptr);
    return 0;
}
