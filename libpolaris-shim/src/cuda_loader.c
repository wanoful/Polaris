// SPDX-License-Identifier: GPL-2.0
//
// CUDA driver-API symbol loader: a tiny dlopen() wrapper. The shim does not
// link against libcuda at build time, so it can be LD_PRELOAD-ed into worker
// processes that do not actually need CUDA without forcing the toolkit onto
// systems that lack it.

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>

#include "cuda_loader.h"

static pthread_once_t g_libcuda_once = PTHREAD_ONCE_INIT;
static void *g_libcuda_handle = NULL;

static void open_libcuda(void)
{
    // RTLD_NOW so we surface missing symbols at dlopen time rather than at
    // the first cuFoo() call. RTLD_GLOBAL because downstream NVIDIA libraries
    // (NVML, libnvidia-ml) expect to find driver symbols in the global scope.
    g_libcuda_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!g_libcuda_handle) {
        // Workers that don't link CUDA at all will still load the shim;
        // that's fine, but log so misconfigurations are visible.
        fprintf(stderr, "[polaris-shim] libcuda.so.1 not found: %s\n", dlerror());
    }
}

void *polaris_shim_resolve_cuda_symbol(const char *name)
{
    pthread_once(&g_libcuda_once, open_libcuda);
    if (!g_libcuda_handle)
        return NULL;

    void *sym = dlsym(g_libcuda_handle, name);
    if (!sym) {
        fprintf(stderr, "[polaris-shim] dlsym(%s) failed: %s\n", name, dlerror());
    }
    return sym;
}
