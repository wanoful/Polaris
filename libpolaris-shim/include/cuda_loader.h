// SPDX-License-Identifier: GPL-2.0
//
// Internal interface to the dlopen()-based CUDA driver-API symbol loader.
// libcuda.so.1 is resolved lazily on first use so that the shim has no link-
// time dependency on the CUDA toolkit — workers that do not actually use
// CUDA pay nothing.

#ifndef POLARIS_SHIM_CUDA_LOADER_H
#define POLARIS_SHIM_CUDA_LOADER_H

#include <stddef.h>

// CUDA driver-API minimal type shims. Matching cuda.h would drag the toolkit
// in; we only need the bits we actually intercept.
typedef int CUresult;
typedef int CUdevice;
typedef int CUdevice_attribute;
typedef void *CUcontext;
typedef void *CUfunction;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef struct {
    char bytes[16];
} CUuuid;
typedef struct {
    char bytes[64];
} CUipcMemHandle;
#define CUDA_SUCCESS 0

typedef int cudaError_t;
typedef void *cudaStream_t;
typedef void *CUstream;
typedef void *cudaEvent_t;
typedef void *CUevent;
typedef void (*cudaHostFn_t)(void *userData);
typedef struct CUgraph_st *cudaGraph_t;
typedef struct CUgraphExec_st *cudaGraphExec_t;
typedef struct CUgraphNode_st *cudaGraphNode_t;
typedef int cudaGraphNodeType;
typedef int cudaStreamCaptureMode;
typedef int cudaStreamCaptureStatus;
typedef int CUmemAllocationGranularity_flags;
typedef struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} dim3;
typedef struct cudaGraphExecUpdateResultInfo_st cudaGraphExecUpdateResultInfo;
typedef struct cudaKernelNodeParams cudaKernelNodeParams;
typedef struct CUmemAllocationProp_st CUmemAllocationProp;
typedef struct CUmemAccessDesc_st CUmemAccessDesc;
typedef union {
    char pad[64];
    int programmaticStreamSerializationAllowed;
    struct {
        void *event;
        int flags;
        int triggerAtBlockStart;
    } programmaticEvent;
} CUlaunchAttributeValue;
typedef struct {
    int id;
    char pad[8 - sizeof(int)];
    CUlaunchAttributeValue value;
} CUlaunchAttribute;
typedef struct CUlaunchConfig_st {
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    CUlaunchAttribute *attrs;
    unsigned int numAttrs;
} CUlaunchConfig;
typedef union {
    char pad[64];
    int programmaticStreamSerializationAllowed;
    struct {
        void *event;
        int flags;
        int triggerAtBlockStart;
    } programmaticEvent;
} cudaLaunchAttributeValue;
typedef struct {
    int id;
    char pad[8 - sizeof(int)];
    cudaLaunchAttributeValue val;
} cudaLaunchAttribute;
typedef struct cudaLaunchConfig_st {
    dim3 gridDim;
    dim3 blockDim;
    size_t dynamicSmemBytes;
    cudaStream_t stream;
    cudaLaunchAttribute *attrs;
    unsigned int numAttrs;
} cudaLaunchConfig_t;
struct cudaMemLocation {
    int type;
    int id;
};
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

typedef CUresult (*cuInit_fn)(unsigned int Flags);
typedef CUresult (*cuDeviceGet_fn)(CUdevice *device, int ordinal);
typedef CUresult (*cuDeviceGetAttribute_fn)(int *pi, CUdevice_attribute attrib, CUdevice dev);
typedef CUresult (*cuDeviceGetUuid_fn)(CUuuid *uuid, CUdevice dev);
typedef CUresult (*cuDevicePrimaryCtxRetain_fn)(CUcontext *pctx, CUdevice dev);
typedef CUresult (*cuCtxSetCurrent_fn)(CUcontext ctx);
typedef CUresult (*cuMemAlloc_fn)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (*cuMemFree_fn)(CUdeviceptr dptr);
typedef CUresult (*cuMemAllocAsync_fn)(CUdeviceptr *dptr, size_t bytesize, cudaStream_t hStream);
typedef CUresult (*cuMemFreeAsync_fn)(CUdeviceptr dptr, cudaStream_t hStream);
typedef CUresult (*cuMemGetInfo_fn)(size_t *free, size_t *total);
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
typedef CUresult (*cuStreamBeginCapture_fn)(CUstream hStream, cudaStreamCaptureMode mode);
typedef CUresult (*cuStreamEndCapture_fn)(CUstream hStream, cudaGraph_t *phGraph);
typedef CUresult (*cuMemGetAddressRange_fn)(CUdeviceptr *pbase,
                                            size_t *psize,
                                            CUdeviceptr dptr);
typedef CUresult (*cuMemcpyHtoD_fn)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
typedef CUresult (*cuMemcpyDtoH_fn)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
typedef CUresult (*cuMemcpyHtoDAsync_fn)(CUdeviceptr dstDevice,
                                         const void *srcHost,
                                         size_t ByteCount,
                                         cudaStream_t hStream);
typedef CUresult (*cuMemcpyDtoHAsync_fn)(void *dstHost,
                                         CUdeviceptr srcDevice,
                                         size_t ByteCount,
                                         cudaStream_t hStream);
typedef CUresult (*cuMemcpy_fn)(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
typedef CUresult (*cuMemcpyAsync_fn)(CUdeviceptr dst,
                                     CUdeviceptr src,
                                     size_t ByteCount,
                                     cudaStream_t hStream);
typedef CUresult (*cuMemsetD8_fn)(CUdeviceptr dstDevice, unsigned char uc, size_t N);
typedef CUresult (*cuMemsetD8Async_fn)(CUdeviceptr dstDevice,
                                       unsigned char uc,
                                       size_t N,
                                       cudaStream_t hStream);
typedef CUresult (*cuMemsetD16_fn)(CUdeviceptr dstDevice, unsigned short us, size_t N);
typedef CUresult (*cuMemsetD16Async_fn)(CUdeviceptr dstDevice,
                                        unsigned short us,
                                        size_t N,
                                        cudaStream_t hStream);
typedef CUresult (*cuMemsetD32_fn)(CUdeviceptr dstDevice, unsigned int ui, size_t N);
typedef CUresult (*cuMemsetD32Async_fn)(CUdeviceptr dstDevice,
                                        unsigned int ui,
                                        size_t N,
                                        cudaStream_t hStream);
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
typedef cudaError_t (*cudaMallocAsync_fn)(void **devPtr, size_t size, cudaStream_t stream);
typedef cudaError_t (*cudaFreeAsync_fn)(void *devPtr, cudaStream_t stream);
typedef cudaError_t (*cudaGetDevice_fn)(int *device);
typedef cudaError_t (*cudaSetDevice_fn)(int device);
typedef cudaError_t (*cudaSetDeviceFlags_fn)(unsigned int flags);
typedef cudaError_t (*cudaGetDeviceFlags_fn)(unsigned int *flags);
typedef cudaError_t (*cudaGetDeviceCount_fn)(int *count);
typedef cudaError_t (*cudaGetDeviceProperties_fn)(void *prop, int device);
typedef cudaError_t (*cudaDeviceGetAttribute_fn)(int *value, int attr, int device);
typedef cudaError_t (*cudaDeviceCanAccessPeer_fn)(int *canAccessPeer,
                                                  int device,
                                                  int peerDevice);
typedef cudaError_t (*cudaDeviceEnablePeerAccess_fn)(int peerDevice, unsigned int flags);
typedef cudaError_t (*cudaDeviceDisablePeerAccess_fn)(int peerDevice);
typedef cudaError_t (*cudaDeviceGetPCIBusId_fn)(char *pciBusId, int len, int device);
typedef cudaError_t (*cudaRuntimeGetVersion_fn)(int *runtimeVersion);
typedef cudaError_t (*cudaDriverGetVersion_fn)(int *driverVersion);
typedef cudaError_t (*cudaDeviceSynchronize_fn)(void);
typedef cudaError_t (*cudaMemGetInfo_fn)(size_t *free, size_t *total);
typedef const char *(*cudaGetErrorString_fn)(cudaError_t error);
typedef const char *(*cudaGetErrorName_fn)(cudaError_t error);
typedef cudaError_t (*cudaGetLastError_fn)(void);
typedef cudaError_t (*cudaPeekAtLastError_fn)(void);
typedef cudaError_t (*cudaStreamCreate_fn)(cudaStream_t *pStream);
typedef cudaError_t (*cudaStreamCreateWithFlags_fn)(cudaStream_t *pStream, unsigned int flags);
typedef cudaError_t (*cudaStreamDestroy_fn)(cudaStream_t stream);
typedef cudaError_t (*cudaStreamSynchronize_fn)(cudaStream_t stream);
typedef cudaError_t (*cudaStreamWaitEvent_fn)(cudaStream_t stream,
                                              cudaEvent_t event,
                                              unsigned int flags);
typedef cudaError_t (*cudaStreamIsCapturing_fn)(cudaStream_t stream,
                                                cudaStreamCaptureStatus *pCaptureStatus);
typedef cudaError_t (*cudaEventCreate_fn)(cudaEvent_t *event);
typedef cudaError_t (*cudaEventCreateWithFlags_fn)(cudaEvent_t *event, unsigned int flags);
typedef cudaError_t (*cudaEventRecord_fn)(cudaEvent_t event, cudaStream_t stream);
typedef cudaError_t (*cudaEventSynchronize_fn)(cudaEvent_t event);
typedef cudaError_t (*cudaEventDestroy_fn)(cudaEvent_t event);
typedef cudaError_t (*cudaEventElapsedTime_fn)(float *ms, cudaEvent_t start, cudaEvent_t end);
typedef cudaError_t (*cudaStreamBeginCapture_fn)(cudaStream_t stream,
                                                 cudaStreamCaptureMode mode);
typedef cudaError_t (*cudaStreamEndCapture_fn)(cudaStream_t stream, cudaGraph_t *pGraph);
typedef cudaError_t (*cudaGraphInstantiate_fn)(cudaGraphExec_t *pGraphExec,
                                               cudaGraph_t graph,
                                               unsigned long long flags);
typedef cudaError_t (*cudaGraphLaunch_fn)(cudaGraphExec_t graphExec, cudaStream_t stream);
typedef cudaError_t (*cudaGraphExecUpdate_fn)(cudaGraphExec_t hGraphExec,
                                              cudaGraph_t hGraph,
                                              cudaGraphExecUpdateResultInfo *resultInfo);
typedef cudaError_t (*cudaGraphGetNodes_fn)(cudaGraph_t graph,
                                            cudaGraphNode_t *nodes,
                                            size_t *numNodes);
typedef cudaError_t (*cudaGraphNodeGetType_fn)(cudaGraphNode_t node,
                                               cudaGraphNodeType *pType);
typedef cudaError_t (*cudaGraphKernelNodeGetParams_fn)(
    cudaGraphNode_t node,
    cudaKernelNodeParams *pNodeParams);
typedef cudaError_t (*cudaGraphKernelNodeSetParams_fn)(
    cudaGraphNode_t node,
    const cudaKernelNodeParams *pNodeParams);
typedef cudaError_t (*cudaGraphDestroy_fn)(cudaGraph_t graph);
typedef cudaError_t (*cudaGraphExecDestroy_fn)(cudaGraphExec_t graphExec);
typedef cudaError_t (*cudaMemcpy_fn)(void *dst, const void *src, size_t count, int kind);
typedef cudaError_t (*cudaMemcpyAsync_fn)(void *dst,
                                          const void *src,
                                          size_t count,
                                          int kind,
                                          cudaStream_t stream);
typedef cudaError_t (*cudaMemcpy2DAsync_fn)(void *dst,
                                            size_t dpitch,
                                            const void *src,
                                            size_t spitch,
                                            size_t width,
                                            size_t height,
                                            int kind,
                                            cudaStream_t stream);
typedef cudaError_t (*cudaMemcpyPeerAsync_fn)(void *dst,
                                              int dstDevice,
                                              const void *src,
                                              int srcDevice,
                                              size_t count,
                                              cudaStream_t stream);
typedef cudaError_t (*cudaMemcpy3DPeerAsync_fn)(const struct cudaMemcpy3DPeerParms *p,
                                                cudaStream_t stream);
typedef cudaError_t (*cudaMemset_fn)(void *devPtr, int value, size_t count);
typedef cudaError_t (*cudaMemsetAsync_fn)(void *devPtr,
                                          int value,
                                          size_t count,
                                          cudaStream_t stream);
typedef cudaError_t (*cudaMemAdvise_fn)(const void *devPtr,
                                        size_t count,
                                        int advice,
                                        struct cudaMemLocation location);
typedef cudaError_t (*cudaIpcGetMemHandle_fn)(cudaIpcMemHandle_t *handle, void *devPtr);
typedef cudaError_t (*cudaPointerGetAttributes_fn)(void *attributes, const void *ptr);
typedef cudaError_t (*cudaMallocHost_fn)(void **ptr, size_t size);
typedef cudaError_t (*cudaHostAlloc_fn)(void **pHost, size_t size, unsigned int flags);
typedef cudaError_t (*cudaFreeHost_fn)(void *ptr);
typedef cudaError_t (*cudaHostGetDevicePointer_fn)(void **pDevice,
                                                   void *pHost,
                                                   unsigned int flags);
typedef cudaError_t (*cudaHostRegister_fn)(void *ptr, size_t size, unsigned int flags);
typedef cudaError_t (*cudaHostUnregister_fn)(void *ptr);
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
                                           cudaStream_t stream);
typedef cudaError_t (*cudaLaunchKernelExC_fn)(const cudaLaunchConfig_t *config,
                                              const void *func,
                                              void **args);
typedef cudaError_t (*cudaLaunchCooperativeKernel_fn)(const void *func,
                                                      dim3 gridDim,
                                                      dim3 blockDim,
                                                      void **args,
                                                      size_t sharedMem,
                                                      cudaStream_t stream);
typedef cudaError_t (*cudaLaunchHostFunc_fn)(cudaStream_t stream,
                                             cudaHostFn_t fn,
                                             void *userData);
typedef cudaError_t (*cudaLaunchHostFunc_v2_fn)(cudaStream_t stream,
                                                cudaHostFn_t fn,
                                                void *userData,
                                                unsigned int syncMode);
typedef cudaError_t (*cudaOccupancyMaxPotentialBlockSize_fn)(int *minGridSize,
                                                             int *blockSize,
                                                             const void *func,
                                                             size_t dynamicSMemSize,
                                                             int blockSizeLimit);

// Resolve a real CUDA driver-API symbol via dlsym. Returns NULL on failure
// (libcuda.so.1 not loadable, or symbol absent). Cached after first lookup.
void *polaris_shim_resolve_cuda_symbol(const char *name);

#endif // POLARIS_SHIM_CUDA_LOADER_H
