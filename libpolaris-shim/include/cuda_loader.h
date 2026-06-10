// SPDX-License-Identifier: GPL-2.0
//
// Internal interface to the dlopen()-based CUDA driver-API symbol loader.
// libcuda.so.1 is resolved lazily on first use so that the shim has no link-
// time dependency on the CUDA toolkit — workers that do not actually use
// CUDA pay nothing.

#ifndef POLARIS_SHIM_CUDA_LOADER_H
#define POLARIS_SHIM_CUDA_LOADER_H

// CUDA driver-API minimal type shims. Matching cuda.h would drag the toolkit
// in; we only need the bits we actually intercept.
typedef int CUresult;
#define CUDA_SUCCESS 0

typedef CUresult (*cuInit_fn)(unsigned int Flags);

// Resolve a real CUDA driver-API symbol via dlsym. Returns NULL on failure
// (libcuda.so.1 not loadable, or symbol absent). Cached after first lookup.
void *polaris_shim_resolve_cuda_symbol(const char *name);

#endif // POLARIS_SHIM_CUDA_LOADER_H
