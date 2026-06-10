// SPDX-License-Identifier: GPL-2.0
//
// libpolaris-shim entry point. v4-M0 scaffolding: intercept cuInit, pass
// through to the real implementation, log once on stderr so the LD_PRELOAD
// path is observable end-to-end. UVM VA-space creation, polaris.ko ioctl
// registration, and allocator interception arrive in M2.

#include <pthread.h>
#include <stdio.h>

#include "cuda_loader.h"
#include "device.h"

// CUDA driver-API symbols are interposed via the standard LD_PRELOAD pattern:
// a same-named exported function in this .so wins over libcuda's during the
// dynamic linker's symbol resolution. We then dlsym() the real function out
// of libcuda.so.1 and forward to it after doing our work.
//
// `default` visibility is required for the interposer to be picked up; the
// Makefile sets -fvisibility=hidden globally, so this attribute is the
// explicit opt-in.
#define POLARIS_SHIM_INTERPOSER __attribute__((visibility("default")))

static pthread_once_t g_announce_once = PTHREAD_ONCE_INIT;

static void announce(void)
{
    // One-shot banner so users can tell the shim is actually loaded.
    fprintf(stderr, "[polaris-shim] active (v4-M0 scaffolding)\n");
}

POLARIS_SHIM_INTERPOSER
CUresult cuInit(unsigned int Flags)
{
    pthread_once(&g_announce_once, announce);

    // POSIX-blessed dlsym() return → fn-pointer dance to keep -Wpedantic
    // happy. Plain casts trip ISO-C's object/function-pointer rule.
    cuInit_fn real;
    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuInit");
    if (!real) {
        // libcuda was not loadable. Propagate a not-initialised result so
        // the worker fails loudly rather than crashing on a NULL call.
        fprintf(stderr, "[polaris-shim] cuInit: real symbol unavailable\n");
        return /* CUDA_ERROR_NOT_INITIALIZED */ 3;
    }

    // TODO(M2): create a fault-capable, externally-owned GPU VA-space here,
    // UvmRegisterGpuVaSpace it, and pass the real (gpu_id, va_space_token,
    // managed_base, managed_length) to polaris_shim_register_vaspace.
    //
    // M1 only exercises the ioctl plumbing with sentinel values so we can
    // confirm the /dev/polaris path is reachable end-to-end from the
    // interposer. Failures are logged but do not abort cuInit.
    fprintf(stderr, "[polaris-shim] cuInit intercepted (flags=0x%x)\n", Flags);
    (void)polaris_shim_register_vaspace(/*gpu_id=*/0,
                                        /*va_space_token=*/0xdeadbeefULL,
                                        /*managed_base=*/0,
                                        /*managed_length=*/1ULL << 30);

    return real(Flags);
}
