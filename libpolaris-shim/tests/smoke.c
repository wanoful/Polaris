// SPDX-License-Identifier: GPL-2.0
//
// Smoke test for libpolaris-shim: links against libcuda at run time (via
// dlopen, so the test still builds on hosts without the CUDA toolkit) and
// calls cuInit once. With the shim LD_PRELOAD-ed the call goes through the
// interposer; without it the call goes straight to libcuda. Both paths
// must succeed when an NVIDIA driver is installed; on a CPU-only host the
// test reports "skipped (no libcuda)".
//
// Build:   cc -O2 -ldl smoke.c -o smoke
// Run:     LD_PRELOAD=$PWD/../libpolaris-shim.so ./smoke

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*cuInit_fn)(unsigned int);

int main(void)
{
    // Pull libcuda.so.1 into the process so a global-scope dlsym can find
    // cuInit. We deliberately do NOT capture the dlopen handle for the
    // lookup: that would bypass the LD_PRELOAD interposer because dlsym on
    // a specific handle skips global scope. RTLD_DEFAULT walks the default
    // scope, where the shim's exported cuInit sits ahead of libcuda's.
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        fprintf(stderr, "smoke: skipped (no libcuda: %s)\n", dlerror());
        return 0;
    }

    cuInit_fn cuInit;
    *(void **)(&cuInit) = dlsym(RTLD_DEFAULT, "cuInit");
    if (!cuInit) {
        fprintf(stderr, "smoke: dlsym(cuInit) failed: %s\n", dlerror());
        return 1;
    }

    int r = cuInit(0);
    fprintf(stderr, "smoke: cuInit returned %d\n", r);
    // 0 (success) or 100 (no device) both indicate the call reached a real
    // implementation. The interesting bit is whether the shim's stderr line
    // appeared above this one.
    return (r == 0 || r == 100) ? 0 : 1;
}
