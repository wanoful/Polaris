// SPDX-License-Identifier: GPL-2.0
//
// libpolaris-shim entry point. The current v4 path can bootstrap from a
// harness-created fault-capable VA-space and, when explicitly enabled,
// interpose CUDA allocator calls to hand back Polaris-managed VA slices.

#include <dlfcn.h>
#include <pthread.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_loader.h"
#include "device.h"
#include "polaris_abi.h"
#include "rm_uvm_bootstrap.h"
#include "uvm_external.h"

// CUDA driver-API symbols are interposed via the standard LD_PRELOAD pattern:
// a same-named exported function in this .so wins over libcuda's during the
// dynamic linker's symbol resolution. We then dlsym() the real function out
// of libcuda.so.1 and forward to it after doing our work.
//
// `default` visibility is required for the interposer to be picked up; the
// Makefile sets -fvisibility=hidden globally, so this attribute is the
// explicit opt-in.
#define POLARIS_SHIM_INTERPOSER __attribute__((visibility("default")))
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_UNKNOWN 999
#define CUDA_ERROR_NOT_SUPPORTED 801
#define CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED 900
#define CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED 704
#define CU_MEMORYTYPE_DEVICE 2
#define CUDA_MEMORY_TYPE_DEVICE 2
#define CUDA_MEM_LOCATION_TYPE_DEVICE 1
#define CU_POINTER_ATTRIBUTE_MEMORY_TYPE 2
#define CU_POINTER_ATTRIBUTE_DEVICE_POINTER 3
#define CU_POINTER_ATTRIBUTE_HOST_POINTER 4
#define CU_POINTER_ATTRIBUTE_IS_MANAGED 8
#define CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL 9
#define CU_POINTER_ATTRIBUTE_RANGE_START_ADDR 11
#define CU_POINTER_ATTRIBUTE_RANGE_SIZE 12
#define CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE 17
#define CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION 6
#define CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT 7
#define CUDA_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION 6
#define CUDA_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT 7

struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;
struct ggml_backend_buffer_type;

typedef struct ggml_backend_buffer *ggml_backend_buffer_t;
typedef struct ggml_backend_buffer_type *ggml_backend_buffer_type_t;
typedef struct ggml_tensor *(*ggml_get_first_tensor_fn)(const struct ggml_context *ctx);
typedef struct ggml_tensor *(*ggml_get_next_tensor_fn)(const struct ggml_context *ctx,
                                                       struct ggml_tensor *tensor);
typedef const char *(*ggml_get_name_fn)(const struct ggml_tensor *tensor);
typedef ggml_backend_buffer_t (*ggml_backend_alloc_ctx_tensors_from_buft_fn)(
    struct ggml_context *ctx,
    ggml_backend_buffer_type_t buft);

static pthread_once_t g_announce_once = PTHREAD_ONCE_INIT;
static pthread_once_t g_bootstrap_once = PTHREAD_ONCE_INIT;
static uint32_t g_registered_gpu_id;
static uint32_t g_registered_cuda_ordinal;
static uint64_t g_registered_rm_client_token;
static uint64_t g_registered_va_space_token;
static uint64_t g_managed_base;
static uint64_t g_managed_length;
static uint64_t g_registered_managed_length;
static uint64_t g_managed_grow_blocks;
static uint64_t g_block_size;
static uint64_t g_min_managed_alloc;
static uint64_t g_max_managed_alloc;
static uint64_t g_selected_alloc_skip;
static uint64_t g_scope_selected_seen;
static uint64_t g_session_id;
static uint64_t g_next_token;
static int g_registered_vaspace;
static int g_manage_allocations;
static int g_allocator_ready;
static int g_create_external_ranges;
static int g_bootstrap_rm_uvm;
static int g_static_rm_backend;
static int g_require_kv_scope;
static int g_allow_zero_memset;
static int g_trace_scope;
static int g_strict_managed_alloc;
static int g_report_stats;
static int g_runtime_selected_device = -1;
static __thread int g_allocation_scope_is_kv;
static struct polaris_shim_bootstrap g_bootstrap_state = {
    .rm_control_fd = -1,
    .uvm_fd = -1,
    .uvm_mm_fd = -1,
};
static void *g_cudart_handle;

struct polaris_shim_allocation {
    CUdeviceptr ptr;
    size_t requested_size;
    uint32_t token_start;
    uint32_t token_count;
    uint64_t block_id;
    uint64_t external_base;
    uint32_t chunk_count;
    struct polaris_shim_allocation_chunk *chunks;
    struct polaris_shim_allocation *next;
};

struct polaris_shim_allocation_chunk {
    uint64_t block_id;
    uint64_t gpu_vaddr;
    uint32_t static_h_memory;
    uint64_t static_size;
};

struct polaris_shim_free_span {
    uint32_t token_start;
    uint32_t token_count;
    struct polaris_shim_free_span *next;
};

struct polaris_shim_cuda_pointer_attributes {
    int type;
    int device;
    void *devicePointer;
    void *hostPointer;
};

static pthread_mutex_t g_alloc_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_window_lock = PTHREAD_MUTEX_INITIALIZER;
static struct polaris_shim_allocation *g_allocations;
static struct polaris_shim_free_span *g_free_spans;
static size_t g_live_managed_allocations;
static size_t g_active_graph_captures;

enum polaris_shim_alloc_api {
    POLARIS_SHIM_ALLOC_API_DRIVER,
    POLARIS_SHIM_ALLOC_API_DRIVER_ASYNC,
    POLARIS_SHIM_ALLOC_API_RUNTIME,
    POLARIS_SHIM_ALLOC_API_RUNTIME_MANAGED,
    POLARIS_SHIM_ALLOC_API_RUNTIME_ASYNC,
};

struct polaris_shim_api_alloc_stats {
    uint64_t calls;
    uint64_t bytes;
    uint64_t selected_calls;
    uint64_t selected_bytes;
};

struct polaris_shim_alloc_stats {
    uint64_t alloc_calls;
    uint64_t alloc_bytes;
    uint64_t selected_calls;
    uint64_t selected_bytes;
    uint64_t policy_passthrough_calls;
    uint64_t policy_passthrough_bytes;
    uint64_t managed_success_calls;
    uint64_t managed_success_requested_bytes;
    uint64_t managed_success_rounded_bytes;
    uint64_t managed_failure_calls;
    uint64_t managed_failure_bytes;
    uint64_t strict_failure_calls;
    uint64_t fallback_alloc_calls;
    uint64_t fallback_alloc_bytes;
    uint64_t fallback_alloc_success_calls;
    uint64_t fallback_alloc_failure_calls;
    uint64_t free_calls;
    uint64_t managed_free_calls;
    uint64_t managed_free_requested_bytes;
    uint64_t managed_free_rounded_bytes;
    uint64_t fallback_free_calls;
    uint64_t fallback_free_success_calls;
    uint64_t fallback_free_failure_calls;
    uint64_t managed_window_grow_calls;
    uint64_t managed_window_grow_failure_calls;
    uint64_t managed_window_grow_bytes;
    uint64_t managed_window_shrink_calls;
    uint64_t managed_window_shrink_failure_calls;
    uint64_t managed_window_shrink_bytes;
    uint64_t live_requested_bytes;
    uint64_t live_rounded_bytes;
    uint64_t peak_live_requested_bytes;
    uint64_t peak_live_rounded_bytes;
    struct polaris_shim_api_alloc_stats driver_alloc;
    struct polaris_shim_api_alloc_stats driver_async_alloc;
    struct polaris_shim_api_alloc_stats runtime_alloc;
    struct polaris_shim_api_alloc_stats runtime_managed_alloc;
    struct polaris_shim_api_alloc_stats runtime_async_alloc;
};

static struct polaris_shim_alloc_stats g_alloc_stats;

static void announce(void)
{
    // One-shot banner so users can tell the shim is actually loaded.
    fprintf(stderr, "[polaris-shim] active (v4 daemon-backed RM path)\n");
}

static int parse_u32_env(const char *name, uint32_t *out)
{
    const char *value = getenv(name);
    char *end = NULL;
    unsigned long parsed;

    if (!value || value[0] == '\0')
        return 0;

    errno = 0;
    parsed = strtoul(value, &end, 0);
    if (errno != 0 || !end || *end != '\0' || parsed > UINT32_MAX) {
        fprintf(stderr, "[polaris-shim] ignoring invalid %s=%s\n", name, value);
        return -EINVAL;
    }

    *out = (uint32_t)parsed;
    return 1;
}

static int parse_u64_env(const char *name, uint64_t *out)
{
    const char *value = getenv(name);
    char *end = NULL;
    unsigned long long parsed;

    if (!value || value[0] == '\0')
        return 0;

    errno = 0;
    parsed = strtoull(value, &end, 0);
    if (errno != 0 || !end || *end != '\0') {
        fprintf(stderr, "[polaris-shim] ignoring invalid %s=%s\n", name, value);
        return -EINVAL;
    }

    *out = (uint64_t)parsed;
    return 1;
}

static int env_enabled(const char *name)
{
    const char *value = getenv(name);

    if (!value || value[0] == '\0')
        return 0;
    return strcmp(value, "0") != 0 && strcmp(value, "false") != 0 &&
           strcmp(value, "FALSE") != 0 && strcmp(value, "no") != 0 &&
           strcmp(value, "NO") != 0;
}

static uint64_t align_up_u64(uint64_t value, uint64_t alignment)
{
    if (alignment == 0)
        return value;
    return (value + alignment - 1) & ~(alignment - 1);
}

static int should_manage_allocation_size(size_t size)
{
    uint64_t request = (uint64_t)size;

    if (size == 0)
        return 0;
    if (g_min_managed_alloc != 0 && request < g_min_managed_alloc)
        return 0;
    if (g_max_managed_alloc != 0 && request > g_max_managed_alloc)
        return 0;
    return 1;
}

static int should_manage_allocation(size_t size)
{
    int selected;
    size_t active_captures;

    if (!should_manage_allocation_size(size)) {
        if (g_trace_scope) {
            fprintf(stderr,
                    "[polaris-shim] allocation policy size=%zu selected=0 "
                    "reason=size scope_kv=%d require_kv_scope=%d\n",
                    size,
                    g_allocation_scope_is_kv,
                    g_require_kv_scope);
        }
        return 0;
    }

    pthread_mutex_lock(&g_alloc_lock);
    active_captures = g_active_graph_captures;
    pthread_mutex_unlock(&g_alloc_lock);
    if (active_captures != 0) {
        if (g_trace_scope) {
            fprintf(stderr,
                    "[polaris-shim] allocation policy size=%zu selected=0 "
                    "reason=graph_capture active_captures=%zu scope_kv=%d "
                    "require_kv_scope=%d\n",
                    size,
                    active_captures,
                    g_allocation_scope_is_kv,
                    g_require_kv_scope);
        }
        return 0;
    }

    if (g_require_kv_scope && !g_allocation_scope_is_kv) {
        if (g_trace_scope) {
            fprintf(stderr,
                    "[polaris-shim] allocation policy size=%zu selected=0 "
                    "reason=scope scope_kv=%d require_kv_scope=%d\n",
                    size,
                    g_allocation_scope_is_kv,
                    g_require_kv_scope);
        }
        return 0;
    }

    selected = 1;
    if (g_selected_alloc_skip != 0) {
        pthread_mutex_lock(&g_alloc_lock);
        if (g_scope_selected_seen < g_selected_alloc_skip) {
            g_scope_selected_seen++;
            selected = 0;
        }
        pthread_mutex_unlock(&g_alloc_lock);
    }

    if (g_trace_scope) {
        fprintf(stderr,
                "[polaris-shim] allocation policy size=%zu selected=%d "
                "reason=%s scope_kv=%d require_kv_scope=%d\n",
                size,
                selected,
                selected ? "match" : "skip",
                g_allocation_scope_is_kv,
                g_require_kv_scope);
    }

    return selected;
}

static void *resolve_next_or_default_symbol(const char *name)
{
    void *symbol = dlsym(RTLD_NEXT, name);

    if (!symbol)
        symbol = dlsym(RTLD_DEFAULT, name);
    return symbol;
}

static int string_starts_with(const char *value, const char *prefix)
{
    size_t prefix_len;

    if (!value || !prefix)
        return 0;

    prefix_len = strlen(prefix);
    return strncmp(value, prefix, prefix_len) == 0;
}

static int ggml_context_has_kv_cache_tensors(const struct ggml_context *ctx)
{
    ggml_get_first_tensor_fn get_first;
    ggml_get_next_tensor_fn get_next;
    ggml_get_name_fn get_name;
    struct ggml_tensor *tensor;

    if (!ctx)
        return 0;

    *(void **)(&get_first) = resolve_next_or_default_symbol("ggml_get_first_tensor");
    *(void **)(&get_next) = resolve_next_or_default_symbol("ggml_get_next_tensor");
    *(void **)(&get_name) = resolve_next_or_default_symbol("ggml_get_name");
    if (!get_first || !get_next || !get_name) {
        if (g_trace_scope) {
            fprintf(stderr,
                    "[polaris-shim] ggml KV scope detection unavailable "
                    "first=%d next=%d name=%d\n",
                    get_first != NULL,
                    get_next != NULL,
                    get_name != NULL);
        }
        return 0;
    }

    for (tensor = get_first(ctx); tensor; tensor = get_next(ctx, tensor)) {
        const char *name = get_name(tensor);
        if (string_starts_with(name, "cache_k_l") ||
            string_starts_with(name, "cache_v_l")) {
            if (g_trace_scope) {
                fprintf(stderr,
                        "[polaris-shim] ggml KV scope detected tensor=%s\n",
                        name);
            }
            return 1;
        }
    }

    return 0;
}

static uint64_t default_bootstrap_window_cap(void)
{
    return 1ULL << 40;
}

static void choose_bootstrap_managed_window(uint64_t *base,
                                            uint64_t *length,
                                            uint64_t block_size)
{
    uint64_t cap = default_bootstrap_window_cap();
    uint64_t block_count = 0;
    uint64_t block_cap = 0;
    int have_base;
    int have_length;
    int have_block_count;

    have_base = parse_u64_env("POLARIS_SHIM_MANAGED_BASE", base);
    have_length = parse_u64_env("POLARIS_SHIM_MANAGED_LENGTH", length);
    (void)parse_u64_env("POLARIS_SHIM_MANAGED_LENGTH_CAP", &cap);
    have_block_count = parse_u64_env("POLARIS_SHIM_MANAGED_BLOCKS", &block_count);

    if (have_base <= 0 && g_bootstrap_state.vaspace_base != 0)
        *base = g_bootstrap_state.vaspace_base;
    if (have_length <= 0 && g_bootstrap_state.vaspace_size != 0)
        *length = g_bootstrap_state.vaspace_size;
    if (have_length > 0)
        return;

    if (cap != 0 && *length > cap)
        *length = cap;

    if (have_block_count <= 0 || block_count == 0)
        return;
    if (block_size == 0) {
        fprintf(stderr,
                "[polaris-shim] ignoring POLARIS_SHIM_MANAGED_BLOCKS without "
                "a valid POLARIS_SHIM_BLOCK_SIZE\n");
        return;
    }
    if (block_count > UINT32_MAX) {
        fprintf(stderr,
                "[polaris-shim] clamping POLARIS_SHIM_MANAGED_BLOCKS=%" PRIu64
                " to %" PRIu32 "\n",
                block_count,
                UINT32_MAX);
        block_count = UINT32_MAX;
    }
    if (block_count > UINT64_MAX / block_size) {
        fprintf(stderr,
                "[polaris-shim] ignoring oversized POLARIS_SHIM_MANAGED_BLOCKS=%"
                PRIu64 " block_size=0x%" PRIx64 "\n",
                block_count,
                block_size);
        return;
    }
    block_cap = block_count * block_size;
    if (block_cap != 0 && *length > block_cap)
        *length = block_cap;
}

static uint64_t choose_initial_managed_window_length(uint64_t capacity,
                                                     uint64_t block_size)
{
    uint64_t initial_length = capacity;
    uint64_t initial_blocks = 0;
    uint64_t block_length = 0;
    int have_initial_length;
    int have_initial_blocks;

    have_initial_length =
        parse_u64_env("POLARIS_SHIM_MANAGED_INITIAL_LENGTH", &initial_length);
    have_initial_blocks =
        parse_u64_env("POLARIS_SHIM_MANAGED_INITIAL_BLOCKS", &initial_blocks);
    (void)parse_u64_env("POLARIS_SHIM_MANAGED_GROW_BLOCKS",
                        &g_managed_grow_blocks);

    if (capacity == 0 || block_size == 0)
        return capacity;

    if (have_initial_length <= 0)
        initial_length = capacity;
    else if (initial_length == 0)
        initial_length = block_size;

    if (have_initial_blocks > 0 && initial_blocks != 0) {
        if (initial_blocks > UINT64_MAX / block_size) {
            fprintf(stderr,
                    "[polaris-shim] ignoring oversized "
                    "POLARIS_SHIM_MANAGED_INITIAL_BLOCKS=%" PRIu64
                    " block_size=0x%" PRIx64 "\n",
                    initial_blocks,
                    block_size);
        } else {
            block_length = initial_blocks * block_size;
            if (have_initial_length > 0 && initial_length < block_length)
                block_length = initial_length;
            initial_length = block_length;
        }
    }

    initial_length = align_up_u64(initial_length, block_size);
    if (initial_length == 0)
        initial_length = block_size;
    if (initial_length > capacity)
        initial_length = capacity;
    return initial_length;
}

static struct polaris_shim_api_alloc_stats *
stats_api_bucket_locked(enum polaris_shim_alloc_api api)
{
    switch (api) {
    case POLARIS_SHIM_ALLOC_API_DRIVER:
        return &g_alloc_stats.driver_alloc;
    case POLARIS_SHIM_ALLOC_API_DRIVER_ASYNC:
        return &g_alloc_stats.driver_async_alloc;
    case POLARIS_SHIM_ALLOC_API_RUNTIME:
        return &g_alloc_stats.runtime_alloc;
    case POLARIS_SHIM_ALLOC_API_RUNTIME_MANAGED:
        return &g_alloc_stats.runtime_managed_alloc;
    case POLARIS_SHIM_ALLOC_API_RUNTIME_ASYNC:
        return &g_alloc_stats.runtime_async_alloc;
    }
    return &g_alloc_stats.driver_alloc;
}

static void stats_note_alloc_call(enum polaris_shim_alloc_api api,
                                  size_t size,
                                  int selected)
{
    struct polaris_shim_api_alloc_stats *api_stats;

    pthread_mutex_lock(&g_alloc_lock);
    g_alloc_stats.alloc_calls++;
    g_alloc_stats.alloc_bytes += (uint64_t)size;
    api_stats = stats_api_bucket_locked(api);
    api_stats->calls++;
    api_stats->bytes += (uint64_t)size;
    if (selected) {
        g_alloc_stats.selected_calls++;
        g_alloc_stats.selected_bytes += (uint64_t)size;
        api_stats->selected_calls++;
        api_stats->selected_bytes += (uint64_t)size;
    } else {
        g_alloc_stats.policy_passthrough_calls++;
        g_alloc_stats.policy_passthrough_bytes += (uint64_t)size;
    }
    pthread_mutex_unlock(&g_alloc_lock);
}

static void stats_note_managed_failure(size_t size)
{
    pthread_mutex_lock(&g_alloc_lock);
    g_alloc_stats.managed_failure_calls++;
    g_alloc_stats.managed_failure_bytes += (uint64_t)size;
    pthread_mutex_unlock(&g_alloc_lock);
}

static void stats_note_strict_failure(void)
{
    pthread_mutex_lock(&g_alloc_lock);
    g_alloc_stats.strict_failure_calls++;
    pthread_mutex_unlock(&g_alloc_lock);
}

static void stats_note_fallback_alloc(size_t size)
{
    pthread_mutex_lock(&g_alloc_lock);
    g_alloc_stats.fallback_alloc_calls++;
    g_alloc_stats.fallback_alloc_bytes += (uint64_t)size;
    pthread_mutex_unlock(&g_alloc_lock);
}

static void stats_note_fallback_alloc_result(int success)
{
    pthread_mutex_lock(&g_alloc_lock);
    if (success)
        g_alloc_stats.fallback_alloc_success_calls++;
    else
        g_alloc_stats.fallback_alloc_failure_calls++;
    pthread_mutex_unlock(&g_alloc_lock);
}

static void stats_note_free_call(void)
{
    pthread_mutex_lock(&g_alloc_lock);
    g_alloc_stats.free_calls++;
    pthread_mutex_unlock(&g_alloc_lock);
}

static void stats_note_fallback_free(void)
{
    pthread_mutex_lock(&g_alloc_lock);
    g_alloc_stats.fallback_free_calls++;
    pthread_mutex_unlock(&g_alloc_lock);
}

static void stats_note_fallback_free_result(int success)
{
    pthread_mutex_lock(&g_alloc_lock);
    if (success)
        g_alloc_stats.fallback_free_success_calls++;
    else
        g_alloc_stats.fallback_free_failure_calls++;
    pthread_mutex_unlock(&g_alloc_lock);
}

static void release_static_rm_backend(uint32_t h_memory, uint64_t size)
{
    struct polaris_shim_rm_allocation allocation = {
        .h_memory = h_memory,
        .size = size,
    };

    if (h_memory == 0)
        return;

    polaris_shim_rm_free_device_memory(&g_bootstrap_state, &allocation);
}

static int allocate_static_rm_backend(uint64_t gpu_vaddr,
                                      uint64_t rounded,
                                      uint64_t block_id,
                                      uint32_t *h_memory_out,
                                      uint64_t *rm_size_out)
{
    struct polaris_shim_rm_allocation allocation = {0};
    int ret;

    if (!h_memory_out || !rm_size_out)
        return -EINVAL;
    *h_memory_out = 0;
    *rm_size_out = 0;

    if (!g_static_rm_backend)
        return 0;

    if (!g_bootstrap_rm_uvm || g_bootstrap_state.rm_control_fd < 0 ||
        g_bootstrap_state.h_client == 0) {
        fprintf(stderr,
                "[polaris-shim] static RM backend requires "
                "POLARIS_SHIM_BOOTSTRAP_RM_UVM=1\n");
        return -EINVAL;
    }

    ret = polaris_shim_rm_alloc_device_memory(&g_bootstrap_state,
                                              rounded,
                                              &allocation);
    if (ret != 0)
        return ret;

    ret = polaris_shim_register_static_block(g_registered_gpu_id,
                                             g_registered_rm_client_token,
                                             g_registered_va_space_token,
                                             gpu_vaddr,
                                             rounded,
                                             0,
                                             g_bootstrap_state.rm_control_fd,
                                             g_bootstrap_state.h_client,
                                             allocation.h_memory);
    if (ret != 0) {
        polaris_shim_rm_free_device_memory(&g_bootstrap_state, &allocation);
        return ret;
    }

    if (block_id != 0) {
        ret = polaris_shim_register_block_backing(block_id,
                                                  g_registered_gpu_id,
                                                  g_bootstrap_state.rm_control_fd,
                                                  g_bootstrap_state.h_client,
                                                  allocation.h_memory,
                                                  rounded,
                                                  0);
        if (ret != 0) {
            polaris_shim_rm_free_device_memory(&g_bootstrap_state, &allocation);
            return ret;
        }
    }

    *h_memory_out = allocation.h_memory;
    *rm_size_out = allocation.size;
    fprintf(stderr,
            "[polaris-shim] static RM backend block va=0x%" PRIx64
            " len=0x%" PRIx64 " hMemory=0x%x\n",
            gpu_vaddr,
            rounded,
            allocation.h_memory);
    return 0;
}

static void free_allocation_chunks_storage(struct polaris_shim_allocation *alloc)
{
    if (!alloc)
        return;
    free(alloc->chunks);
    alloc->chunks = NULL;
    alloc->chunk_count = 0;
}

static int release_allocation_chunks(struct polaris_shim_allocation *alloc)
{
    int first_ret = 0;
    uint32_t i;

    if (!alloc)
        return -EINVAL;

    for (i = 0; i < alloc->chunk_count; ++i) {
        struct polaris_shim_allocation_chunk *chunk = &alloc->chunks[i];
        int ret = 0;

        if (chunk->block_id != 0) {
            ret = polaris_shim_unmap_block_mappings(chunk->block_id, NULL);
            if (ret == 0) {
                uint32_t release_flags = chunk->static_h_memory != 0
                                             ? POLARIS_RELEASE_FLAG_CALLER_OWNS_BACKING
                                             : 0;
                int release_ret = polaris_shim_block_release_with_flags(
                    g_session_id,
                    alloc->token_start + i,
                    1,
                    release_flags);
                if (release_ret != 0)
                    ret = release_ret;
            }
        }

        if (ret == 0) {
            release_static_rm_backend(chunk->static_h_memory, chunk->static_size);
            chunk->static_h_memory = 0;
            chunk->static_size = 0;
            chunk->block_id = 0;
            chunk->gpu_vaddr = 0;
        } else if (first_ret == 0) {
            first_ret = ret;
        }
    }

    return first_ret;
}

static void *resolve_next_symbol(const char *name)
{
    void *symbol = dlsym(RTLD_NEXT, name);
    return symbol;
}

static void *resolve_cudart_symbol(const char *name)
{
    void *symbol = resolve_next_symbol(name);
    if (symbol)
        return symbol;

    if (!g_cudart_handle) {
        g_cudart_handle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_LOCAL);
        if (!g_cudart_handle)
            g_cudart_handle = dlopen("libcudart.so.11.0", RTLD_LAZY | RTLD_LOCAL);
        if (!g_cudart_handle)
            g_cudart_handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_LOCAL);
    }
    if (!g_cudart_handle)
        return NULL;
    return dlsym(g_cudart_handle, name);
}

static void *resolve_cuda_symbol_with_alias(const char *name, const char *alias)
{
    void *symbol = resolve_next_symbol(name);

    if (!symbol && alias)
        symbol = resolve_next_symbol(alias);
    if (!symbol)
        symbol = polaris_shim_resolve_cuda_symbol(name);
    if (!symbol && alias)
        symbol = polaris_shim_resolve_cuda_symbol(alias);
    return symbol;
}

static void report_stats_at_exit(void)
{
    struct polaris_shim_alloc_stats stats;
    size_t live_allocations;

    if (!g_report_stats)
        return;

    pthread_mutex_lock(&g_alloc_lock);
    stats = g_alloc_stats;
    live_allocations = g_live_managed_allocations;
    pthread_mutex_unlock(&g_alloc_lock);

    fprintf(stderr,
            "[polaris-shim] stats alloc_calls=%" PRIu64
            " alloc_bytes=%" PRIu64
            " selected_calls=%" PRIu64
            " selected_bytes=%" PRIu64
            " policy_passthrough_calls=%" PRIu64
            " policy_passthrough_bytes=%" PRIu64 "\n",
            stats.alloc_calls,
            stats.alloc_bytes,
            stats.selected_calls,
            stats.selected_bytes,
            stats.policy_passthrough_calls,
            stats.policy_passthrough_bytes);
    fprintf(stderr,
            "[polaris-shim] stats managed_success_calls=%" PRIu64
            " managed_success_requested_bytes=%" PRIu64
            " managed_success_rounded_bytes=%" PRIu64
            " managed_failure_calls=%" PRIu64
            " managed_failure_bytes=%" PRIu64
            " strict_failure_calls=%" PRIu64 "\n",
            stats.managed_success_calls,
            stats.managed_success_requested_bytes,
            stats.managed_success_rounded_bytes,
            stats.managed_failure_calls,
            stats.managed_failure_bytes,
            stats.strict_failure_calls);
    fprintf(stderr,
            "[polaris-shim] stats api_driver_alloc_calls=%" PRIu64
            " api_driver_alloc_selected=%" PRIu64
            " api_driver_async_alloc_calls=%" PRIu64
            " api_driver_async_alloc_selected=%" PRIu64
            " api_runtime_alloc_calls=%" PRIu64
            " api_runtime_alloc_selected=%" PRIu64
            " api_runtime_managed_alloc_calls=%" PRIu64
            " api_runtime_managed_alloc_selected=%" PRIu64
            " api_runtime_async_alloc_calls=%" PRIu64
            " api_runtime_async_alloc_selected=%" PRIu64 "\n",
            stats.driver_alloc.calls,
            stats.driver_alloc.selected_calls,
            stats.driver_async_alloc.calls,
            stats.driver_async_alloc.selected_calls,
            stats.runtime_alloc.calls,
            stats.runtime_alloc.selected_calls,
            stats.runtime_managed_alloc.calls,
            stats.runtime_managed_alloc.selected_calls,
            stats.runtime_async_alloc.calls,
            stats.runtime_async_alloc.selected_calls);
    fprintf(stderr,
            "[polaris-shim] stats fallback_alloc_calls=%" PRIu64
            " fallback_alloc_bytes=%" PRIu64
            " fallback_alloc_success_calls=%" PRIu64
            " fallback_alloc_failure_calls=%" PRIu64
            " free_calls=%" PRIu64
            " managed_free_calls=%" PRIu64
            " fallback_free_calls=%" PRIu64
            " fallback_free_success_calls=%" PRIu64
            " fallback_free_failure_calls=%" PRIu64
            " managed_window_grow_calls=%" PRIu64
            " managed_window_grow_failure_calls=%" PRIu64
            " managed_window_grow_bytes=%" PRIu64
            " managed_window_shrink_calls=%" PRIu64
            " managed_window_shrink_failure_calls=%" PRIu64
            " managed_window_shrink_bytes=%" PRIu64 "\n",
            stats.fallback_alloc_calls,
            stats.fallback_alloc_bytes,
            stats.fallback_alloc_success_calls,
            stats.fallback_alloc_failure_calls,
            stats.free_calls,
            stats.managed_free_calls,
            stats.fallback_free_calls,
            stats.fallback_free_success_calls,
            stats.fallback_free_failure_calls,
            stats.managed_window_grow_calls,
            stats.managed_window_grow_failure_calls,
            stats.managed_window_grow_bytes,
            stats.managed_window_shrink_calls,
            stats.managed_window_shrink_failure_calls,
            stats.managed_window_shrink_bytes);
    fprintf(stderr,
            "[polaris-shim] stats live_allocations=%zu"
            " live_requested_bytes=%" PRIu64
            " live_rounded_bytes=%" PRIu64
            " peak_live_requested_bytes=%" PRIu64
            " peak_live_rounded_bytes=%" PRIu64 "\n",
            live_allocations,
            stats.live_requested_bytes,
            stats.live_rounded_bytes,
            stats.peak_live_requested_bytes,
            stats.peak_live_rounded_bytes);
}

static void unregister_vaspace_at_exit(void)
{
    struct polaris_shim_allocation *list;

    pthread_mutex_lock(&g_alloc_lock);
    list = g_allocations;
    g_allocations = NULL;
    g_live_managed_allocations = 0;
    pthread_mutex_unlock(&g_alloc_lock);

    while (list) {
        struct polaris_shim_allocation *next = list->next;
        int ret = release_allocation_chunks(list);
        if (ret == 0 && list->external_base != 0) {
            ret = polaris_shim_uvm_free_external_range(list->external_base);
        }
        fprintf(stderr,
                "[polaris-shim] exit cleanup ptr=0x%" PRIx64
                " blocks=%u ret=%d\n",
                (uint64_t)list->ptr, list->chunk_count, ret);
        free_allocation_chunks_storage(list);
        free(list);
        list = next;
    }

    pthread_mutex_lock(&g_alloc_lock);
    while (g_free_spans) {
        struct polaris_shim_free_span *next = g_free_spans->next;
        free(g_free_spans);
        g_free_spans = next;
    }
    pthread_mutex_unlock(&g_alloc_lock);

    if (g_session_id != 0) {
        (void)polaris_shim_session_destroy(g_session_id);
        g_session_id = 0;
    }

    if (!g_registered_vaspace)
        goto cleanup_bootstrap;

    (void)polaris_shim_unregister_vaspace(g_registered_gpu_id,
                                          g_registered_rm_client_token,
                                          g_registered_va_space_token);
    g_registered_vaspace = 0;

cleanup_bootstrap:
    if (g_bootstrap_rm_uvm) {
        polaris_shim_bootstrap_cleanup(&g_bootstrap_state);
        g_bootstrap_rm_uvm = 0;
    }
}

static void cleanup_bootstrap_after_failure(void)
{
    if (!g_bootstrap_rm_uvm)
        return;

    polaris_shim_bootstrap_cleanup(&g_bootstrap_state);
    g_bootstrap_rm_uvm = 0;
}

static int bootstrap_allocator_control_plane(uint32_t gpu_id,
                                             uint64_t base,
                                             uint64_t length)
{
    uint64_t block_size = 2ULL * 1024ULL * 1024ULL;
    uint64_t ignored = 0;
    int ret;

    (void)parse_u64_env("POLARIS_SHIM_BLOCK_SIZE", &block_size);
    if (block_size == 0) {
        fprintf(stderr, "[polaris-shim] invalid POLARIS_SHIM_BLOCK_SIZE=0\n");
        return -EINVAL;
    }

    ret = polaris_shim_register_va_range(gpu_id, base, length, block_size, &ignored);
    if (ret != 0)
        return ret;

    ret = polaris_shim_session_create(gpu_id, length, block_size, &g_session_id);
    if (ret != 0)
        return ret;

    g_block_size = block_size;
    g_next_token = 0;
    g_allocator_ready = 1;
    fprintf(stderr,
            "[polaris-shim] allocator ready session=%" PRIu64
            " block_size=0x%" PRIx64
            " min_alloc=0x%" PRIx64
            " max_alloc=0x%" PRIx64
            " require_kv_scope=%d"
            " selected_skip=%" PRIu64 "\n",
            g_session_id,
            g_block_size,
            g_min_managed_alloc,
            g_max_managed_alloc,
            g_require_kv_scope,
            g_selected_alloc_skip);
    return 0;
}

static void bootstrap_vaspace(void)
{
    uint32_t gpu_id = 0;
    uint32_t cuda_ordinal = 0;
    uint64_t rm_client_token = 0;
    uint64_t token = 0;
    uint64_t base = 0;
    uint64_t length = 0;
    uint64_t registered_length = 0;
    uint64_t block_size = 2ULL * 1024ULL * 1024ULL;
    uint64_t total_bytes = 0;
    uint64_t budget_bytes = 0;
    uint64_t cpu_pool_bytes = 0;
    int invalid_block_size = 0;
    int ret;

    g_bootstrap_rm_uvm = env_enabled("POLARIS_SHIM_BOOTSTRAP_RM_UVM");

    /*
     * Temporary M2 harness path: a microbenchmark that creates the external
     * fault-capable RM VA-space can pass the real UVM token/range here. The
     * production shim will replace this with direct RM allocation,
     * UvmRegisterGpuVaSpace, and external-range registration calls.
     */
    if (parse_u32_env("POLARIS_SHIM_GPU_ID", &gpu_id) <= 0)
        gpu_id = 0;
    if (parse_u64_env("POLARIS_SHIM_BLOCK_SIZE", &block_size) > 0 && block_size == 0)
        invalid_block_size = 1;
    if (g_bootstrap_rm_uvm && invalid_block_size) {
        fprintf(stderr, "[polaris-shim] invalid POLARIS_SHIM_BLOCK_SIZE=0\n");
        cleanup_bootstrap_after_failure();
        return;
    }

    if (g_bootstrap_rm_uvm) {
        if (parse_u32_env("POLARIS_SHIM_CUDA_ORDINAL", &cuda_ordinal) <= 0 &&
            g_runtime_selected_device >= 0) {
            cuda_ordinal = (uint32_t)g_runtime_selected_device;
        }
        ret = polaris_shim_bootstrap_rm_uvm((int)cuda_ordinal,
                                            gpu_id,
                                            &g_bootstrap_state);
        if (ret != 0) {
            fprintf(stderr,
                    "[polaris-shim] RM/UVM bootstrap failed: %d\n",
                    ret);
            return;
        }
        rm_client_token = g_bootstrap_state.h_client;
        token = g_bootstrap_state.h_vaspace;
        if (g_bootstrap_state.observed_va_space_token != 0) {
            gpu_id = g_bootstrap_state.observed_gpu_id;
            rm_client_token = g_bootstrap_state.observed_rm_client_token;
            token = g_bootstrap_state.observed_va_space_token;
        }
        choose_bootstrap_managed_window(&base, &length, block_size);
        g_create_external_ranges = 1;
        ret = polaris_shim_uvm_adopt_fd(g_bootstrap_state.uvm_fd);
        if (ret != 0) {
            fprintf(stderr,
                    "[polaris-shim] failed to adopt bootstrapped UVM fd: %d\n",
                    ret);
            polaris_shim_bootstrap_cleanup(&g_bootstrap_state);
            g_bootstrap_rm_uvm = 0;
            return;
        }
    } else {
        if (parse_u64_env("POLARIS_SHIM_VASPACE_TOKEN", &token) <= 0 ||
            parse_u64_env("POLARIS_SHIM_MANAGED_BASE", &base) <= 0 ||
            parse_u64_env("POLARIS_SHIM_MANAGED_LENGTH", &length) <= 0) {
            return;
        }
        (void)parse_u64_env("POLARIS_SHIM_RM_CLIENT_TOKEN", &rm_client_token);
    }

    if (token == 0 || length == 0) {
        fprintf(stderr,
                "[polaris-shim] incomplete VA-space bootstrap token=0x%" PRIx64
                " length=0x%" PRIx64 "\n",
                token, length);
        return;
    }

    g_manage_allocations = env_enabled("POLARIS_SHIM_MANAGE_ALLOCATIONS");
    if (g_bootstrap_rm_uvm)
        g_manage_allocations = 1;
    g_strict_managed_alloc = env_enabled("POLARIS_SHIM_STRICT_MANAGED_ALLOC");
    g_report_stats = env_enabled("POLARIS_SHIM_REPORT_STATS");
    g_static_rm_backend = env_enabled("POLARIS_SHIM_STATIC_RM_BACKEND");
    g_require_kv_scope = env_enabled("POLARIS_SHIM_REQUIRE_KV_SCOPE");
    g_allow_zero_memset = env_enabled("POLARIS_SHIM_ALLOW_ZERO_MEMSET");
    g_trace_scope = env_enabled("POLARIS_SHIM_TRACE_SCOPE");
    if (g_static_rm_backend) {
        if (!g_bootstrap_rm_uvm) {
            fprintf(stderr,
                    "[polaris-shim] POLARIS_SHIM_STATIC_RM_BACKEND requires "
                    "POLARIS_SHIM_BOOTSTRAP_RM_UVM=1; disabling static backend\n");
            g_static_rm_backend = 0;
        } else {
            g_create_external_ranges = 1;
        }
    }

    if (g_manage_allocations) {
        if (invalid_block_size) {
            fprintf(stderr, "[polaris-shim] invalid POLARIS_SHIM_BLOCK_SIZE=0\n");
            g_manage_allocations = 0;
            cleanup_bootstrap_after_failure();
            return;
        }
        (void)parse_u64_env("POLARIS_SHIM_MIN_MANAGED_ALLOC", &g_min_managed_alloc);
        (void)parse_u64_env("POLARIS_SHIM_MAX_MANAGED_ALLOC", &g_max_managed_alloc);
        (void)parse_u64_env("POLARIS_SHIM_SELECTED_ALLOC_SKIP", &g_selected_alloc_skip);
        length = align_up_u64(length, block_size);
        if (g_max_managed_alloc != 0 && g_max_managed_alloc < g_min_managed_alloc) {
            fprintf(stderr,
                    "[polaris-shim] managed allocation filter excludes all sizes "
                    "(min=0x%" PRIx64 " max=0x%" PRIx64 ")\n",
                    g_min_managed_alloc,
                    g_max_managed_alloc);
        }
        registered_length = choose_initial_managed_window_length(length, block_size);
        if (!g_bootstrap_rm_uvm)
            g_create_external_ranges = env_enabled("POLARIS_SHIM_CREATE_EXTERNAL_RANGES");
        if (g_create_external_ranges) {
            uint32_t uvm_fd = 0;
            if (g_bootstrap_rm_uvm) {
                ret = 1;
            } else {
                ret = parse_u32_env("POLARIS_SHIM_UVM_FD", &uvm_fd);
            }
            if ((!g_bootstrap_rm_uvm && ret <= 0) ||
                (!g_bootstrap_rm_uvm && polaris_shim_uvm_adopt_fd((int)uvm_fd) != 0)) {
                fprintf(stderr,
                        "[polaris-shim] external-range creation disabled; "
                        "set POLARIS_SHIM_UVM_FD to the registered UVM VA-space fd\n");
                g_create_external_ranges = 0;
            }
        }
    }
    if (registered_length == 0)
        registered_length = length;

    if (g_manage_allocations || env_enabled("POLARIS_SHIM_REGISTER_GPU")) {
        (void)parse_u64_env("POLARIS_SHIM_GPU_TOTAL_BYTES", &total_bytes);
        (void)parse_u64_env("POLARIS_SHIM_GPU_BUDGET_BYTES", &budget_bytes);
        (void)parse_u64_env("POLARIS_SHIM_CPU_POOL_BYTES", &cpu_pool_bytes);
        if (total_bytes == 0)
            total_bytes = align_up_u64(length, block_size);
        if (budget_bytes == 0)
            budget_bytes = total_bytes;
        ret = polaris_shim_register_gpu(
            gpu_id,
            total_bytes,
            budget_bytes,
            cpu_pool_bytes,
            env_enabled("POLARIS_SHIM_TRANSIENT_GPU") ? POLARIS_REGISTER_GPU_FLAG_TRANSIENT : 0);
        if (ret != 0) {
            fprintf(stderr,
                    "[polaris-shim] GPU bootstrap failed; VA-space registration skipped\n");
            g_manage_allocations = 0;
            cleanup_bootstrap_after_failure();
            return;
        }
    }

    if (polaris_shim_register_vaspace(gpu_id,
                                      rm_client_token,
                                      token,
                                      base,
                                      registered_length) == 0) {
        g_registered_gpu_id = gpu_id;
        g_registered_cuda_ordinal = cuda_ordinal;
        g_registered_rm_client_token = rm_client_token;
        g_registered_va_space_token = token;
        g_managed_base = base;
        g_managed_length = length;
        g_registered_managed_length = registered_length;
        g_registered_vaspace = 1;
        atexit(unregister_vaspace_at_exit);
        if (g_report_stats)
            atexit(report_stats_at_exit);
        fprintf(stderr,
                "[polaris-shim] registered VA-space gpu=%u client=0x%" PRIx64
                " token=0x%" PRIx64
                " base=0x%" PRIx64 " length=0x%" PRIx64
                " capacity=0x%" PRIx64 "\n",
                gpu_id,
                rm_client_token,
                token,
                base,
                registered_length,
                length);
    } else {
        cleanup_bootstrap_after_failure();
        return;
    }

    if (g_manage_allocations) {
        ret = bootstrap_allocator_control_plane(gpu_id, base, length);
        if (ret != 0) {
            fprintf(stderr,
                    "[polaris-shim] allocator disabled after bootstrap failure: %d\n",
                    ret);
            if (g_registered_vaspace) {
                (void)polaris_shim_unregister_vaspace(g_registered_gpu_id,
                                                      g_registered_rm_client_token,
                                                      g_registered_va_space_token);
                g_registered_vaspace = 0;
            }
            g_manage_allocations = 0;
            cleanup_bootstrap_after_failure();
        }
    }
}

static struct polaris_shim_allocation *find_allocation(CUdeviceptr ptr,
                                                       struct polaris_shim_allocation ***prev_next)
{
    struct polaris_shim_allocation **link = &g_allocations;
    while (*link) {
        if ((*link)->ptr == ptr) {
            if (prev_next)
                *prev_next = link;
            return *link;
        }
        link = &(*link)->next;
    }
    return NULL;
}

static int find_allocation_range(CUdeviceptr ptr,
                                 CUdeviceptr *base_out,
                                 uint64_t *length_out)
{
    struct polaris_shim_allocation *alloc;

    pthread_mutex_lock(&g_alloc_lock);
    alloc = g_allocations;
    while (alloc) {
        uint64_t length = (uint64_t)alloc->token_count * g_block_size;
        if (ptr >= alloc->ptr && ptr < alloc->ptr + length) {
            if (base_out)
                *base_out = alloc->ptr;
            if (length_out)
                *length_out = length;
            pthread_mutex_unlock(&g_alloc_lock);
            return 1;
        }
        alloc = alloc->next;
    }
    pthread_mutex_unlock(&g_alloc_lock);
    return 0;
}

static int get_managed_pointer_attribute(void *data, int attribute, CUdeviceptr ptr)
{
    CUdeviceptr base = 0;
    uint64_t length = 0;

    if (!data)
        return -EINVAL;

    if (!find_allocation_range(ptr, &base, &length))
        return -ENOENT;

    switch (attribute) {
        case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
            *(unsigned int *)data = CU_MEMORYTYPE_DEVICE;
            return 0;
        case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
            *(CUdeviceptr *)data = ptr;
            return 0;
        case CU_POINTER_ATTRIBUTE_HOST_POINTER:
            *(void **)data = NULL;
            return 0;
        case CU_POINTER_ATTRIBUTE_IS_MANAGED:
            *(unsigned int *)data = 0;
            return 0;
        case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
            *(int *)data = (int)g_registered_cuda_ordinal;
            return 0;
        case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
            *(CUdeviceptr *)data = base;
            return 0;
        case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
            *(size_t *)data = (size_t)length;
            return 0;
        case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
            *(void **)data = NULL;
            return 0;
        default:
            return -EOPNOTSUPP;
    }
}

static int get_managed_runtime_pointer_attributes(
    struct polaris_shim_cuda_pointer_attributes *attrs,
    const void *ptr)
{
    CUdeviceptr base = 0;
    uint64_t length = 0;
    CUdeviceptr raw = (CUdeviceptr)(uintptr_t)ptr;

    if (!attrs)
        return -EINVAL;
    if (!find_allocation_range(raw, &base, &length))
        return -ENOENT;

    (void)base;
    (void)length;
    memset(attrs, 0, sizeof(*attrs));
    attrs->type = CUDA_MEMORY_TYPE_DEVICE;
    attrs->device = (int)g_registered_cuda_ordinal;
    attrs->devicePointer = (void *)(uintptr_t)raw;
    attrs->hostPointer = NULL;
    return 0;
}

static int get_managed_address_range(CUdeviceptr *pbase,
                                     size_t *psize,
                                     CUdeviceptr ptr)
{
    CUdeviceptr base = 0;
    uint64_t length = 0;

    if (!find_allocation_range(ptr, &base, &length))
        return -ENOENT;

    if (pbase)
        *pbase = base;
    if (psize)
        *psize = (size_t)length;
    return 0;
}

static int is_managed_pointer_value(const void *ptr)
{
    if (!ptr)
        return 0;
    return find_allocation_range((CUdeviceptr)(uintptr_t)ptr, NULL, NULL);
}

static int take_token_span_locked(uint32_t token_count, uint32_t *token_start_out)
{
    struct polaris_shim_free_span **link = &g_free_spans;
    uint64_t registered_tokens;
    uint64_t max_tokens;

    while (*link) {
        struct polaris_shim_free_span *span = *link;
        if (span->token_count >= token_count) {
            *token_start_out = span->token_start;
            span->token_start += token_count;
            span->token_count -= token_count;
            if (span->token_count == 0) {
                *link = span->next;
                free(span);
            }
            return 0;
        }
        link = &span->next;
    }

    max_tokens = g_managed_length / g_block_size;
    registered_tokens = g_registered_managed_length / g_block_size;
    if (g_next_token > UINT32_MAX ||
        g_next_token + token_count > UINT32_MAX ||
        g_next_token + token_count > max_tokens)
        return -ENOMEM;
    if (g_next_token + token_count > registered_tokens)
        return -EAGAIN;

    *token_start_out = (uint32_t)g_next_token;
    g_next_token += token_count;
    return 0;
}

static void stats_note_window_grow(uint64_t old_length,
                                   uint64_t new_length,
                                   int success)
{
    pthread_mutex_lock(&g_alloc_lock);
    if (success) {
        g_alloc_stats.managed_window_grow_calls++;
        if (new_length > old_length)
            g_alloc_stats.managed_window_grow_bytes += new_length - old_length;
    } else {
        g_alloc_stats.managed_window_grow_failure_calls++;
    }
    pthread_mutex_unlock(&g_alloc_lock);
}

static int grow_registered_managed_window(uint64_t required_tokens)
{
    uint64_t required_length;
    uint64_t new_length;
    uint64_t old_length;
    uint64_t grow_length = 0;
    int ret;

    if (!g_registered_vaspace || g_block_size == 0)
        return -EINVAL;
    if (required_tokens > UINT64_MAX / g_block_size)
        return -ENOMEM;

    required_length = required_tokens * g_block_size;
    if (required_length > g_managed_length)
        return -ENOMEM;

    old_length = g_registered_managed_length;
    if (required_length <= old_length) {
        return 0;
    }

    new_length = required_length;
    if (g_managed_grow_blocks != 0) {
        if (g_managed_grow_blocks > UINT64_MAX / g_block_size) {
            fprintf(stderr,
                    "[polaris-shim] ignoring oversized "
                    "POLARIS_SHIM_MANAGED_GROW_BLOCKS=%" PRIu64
                    " block_size=0x%" PRIx64 "\n",
                    g_managed_grow_blocks,
                    g_block_size);
        } else {
            grow_length = g_managed_grow_blocks * g_block_size;
            if (old_length <= UINT64_MAX - grow_length &&
                old_length + grow_length > new_length) {
                new_length = old_length + grow_length;
            }
        }
    }
    if (new_length > g_managed_length)
        new_length = g_managed_length;

    ret = polaris_shim_register_vaspace(g_registered_gpu_id,
                                        g_registered_rm_client_token,
                                        g_registered_va_space_token,
                                        g_managed_base,
                                        new_length);
    if (ret == 0) {
        pthread_mutex_lock(&g_alloc_lock);
        g_registered_managed_length = new_length;
        g_alloc_stats.managed_window_grow_calls++;
        if (new_length > old_length)
            g_alloc_stats.managed_window_grow_bytes += new_length - old_length;
        pthread_mutex_unlock(&g_alloc_lock);
        fprintf(stderr,
                "[polaris-shim] grew registered VA-space window "
                "old=0x%" PRIx64 " new=0x%" PRIx64
                " capacity=0x%" PRIx64 "\n",
                old_length,
                new_length,
                g_managed_length);
    } else {
        stats_note_window_grow(old_length, old_length, 0);
        fprintf(stderr,
                "[polaris-shim] failed to grow registered VA-space window "
                "old=0x%" PRIx64 " required=0x%" PRIx64
                " capacity=0x%" PRIx64 " ret=%d\n",
                old_length,
                required_length,
                g_managed_length,
                ret);
    }

    return ret;
}

static int reserve_token_span_locked(uint32_t token_count, uint32_t *token_start_out)
{
    int ret;

    for (;;) {
        uint64_t required_tokens;

        pthread_mutex_lock(&g_alloc_lock);
        ret = take_token_span_locked(token_count, token_start_out);
        required_tokens = g_next_token + token_count;
        pthread_mutex_unlock(&g_alloc_lock);

        if (ret != -EAGAIN)
            return ret;

        ret = grow_registered_managed_window(required_tokens);
        if (ret != 0)
            return ret;
    }
}

static int reserve_token_span(uint32_t token_count, uint32_t *token_start_out)
{
    int ret;

    pthread_mutex_lock(&g_window_lock);
    ret = reserve_token_span_locked(token_count, token_start_out);
    pthread_mutex_unlock(&g_window_lock);
    return ret;
}

static void return_token_span_locked(uint32_t token_start, uint32_t token_count)
{
    struct polaris_shim_free_span **link = &g_free_spans;
    struct polaris_shim_free_span *span;
    uint64_t returned_end;

    if (token_count == 0)
        return;

    returned_end = (uint64_t)token_start + token_count;
    while (*link && (*link)->token_start < token_start)
        link = &(*link)->next;

    if (*link && returned_end == (*link)->token_start) {
        (*link)->token_start = token_start;
        (*link)->token_count += token_count;
        span = *link;
    } else {
        span = calloc(1, sizeof(*span));
        if (!span) {
            fprintf(stderr,
                    "[polaris-shim] leaked reusable token span start=%u count=%u after OOM\n",
                    token_start,
                    token_count);
            return;
        }
        span->token_start = token_start;
        span->token_count = token_count;
        span->next = *link;
        *link = span;
    }

    if (link != &g_free_spans) {
        struct polaris_shim_free_span *prev = g_free_spans;
        while (prev && prev->next != span)
            prev = prev->next;
        if (prev && (uint64_t)prev->token_start + prev->token_count == span->token_start) {
            prev->token_count += span->token_count;
            prev->next = span->next;
            free(span);
        }
    }
}

static uint64_t collapse_tail_free_spans_locked(void)
{
    for (;;) {
        struct polaris_shim_free_span **link = &g_free_spans;
        int collapsed = 0;

        while (*link) {
            struct polaris_shim_free_span *span = *link;
            uint64_t span_end = (uint64_t)span->token_start + span->token_count;

            if (span_end == g_next_token) {
                g_next_token = span->token_start;
                *link = span->next;
                free(span);
                collapsed = 1;
                break;
            }
            link = &span->next;
        }

        if (!collapsed)
            return g_next_token;
    }
}

static int shrink_registered_managed_window(uint64_t target_tokens)
{
    uint64_t target_length;
    uint64_t old_length;
    int ret;

    if (!g_registered_vaspace || g_block_size == 0)
        return -EINVAL;
    if (target_tokens > UINT64_MAX / g_block_size)
        return -ENOMEM;

    target_length = target_tokens * g_block_size;
    if (target_length == 0)
        target_length = g_block_size;
    if (target_length > g_managed_length)
        target_length = g_managed_length;

    old_length = g_registered_managed_length;
    if (target_length >= old_length)
        return 0;

    ret = polaris_shim_register_vaspace(g_registered_gpu_id,
                                        g_registered_rm_client_token,
                                        g_registered_va_space_token,
                                        g_managed_base,
                                        target_length);
    if (ret == 0) {
        pthread_mutex_lock(&g_alloc_lock);
        g_registered_managed_length = target_length;
        g_alloc_stats.managed_window_shrink_calls++;
        g_alloc_stats.managed_window_shrink_bytes += old_length - target_length;
        pthread_mutex_unlock(&g_alloc_lock);
        fprintf(stderr,
                "[polaris-shim] shrank registered VA-space window "
                "old=0x%" PRIx64 " new=0x%" PRIx64
                " capacity=0x%" PRIx64 "\n",
                old_length,
                target_length,
                g_managed_length);
    } else {
        pthread_mutex_lock(&g_alloc_lock);
        g_alloc_stats.managed_window_shrink_failure_calls++;
        pthread_mutex_unlock(&g_alloc_lock);
        fprintf(stderr,
                "[polaris-shim] failed to shrink registered VA-space window "
                "old=0x%" PRIx64 " target=0x%" PRIx64
                " capacity=0x%" PRIx64 " ret=%d\n",
                old_length,
                target_length,
                g_managed_length,
                ret);
    }

    return ret;
}

static void return_token_span_and_reclaim(uint32_t token_start, uint32_t token_count)
{
    uint64_t target_tokens;

    pthread_mutex_lock(&g_window_lock);
    pthread_mutex_lock(&g_alloc_lock);
    return_token_span_locked(token_start, token_count);
    target_tokens = collapse_tail_free_spans_locked();
    pthread_mutex_unlock(&g_alloc_lock);
    (void)shrink_registered_managed_window(target_tokens);
    pthread_mutex_unlock(&g_window_lock);
}

static int polaris_alloc_managed(size_t size, CUdeviceptr *out)
{
    struct polaris_shim_allocation *alloc = NULL;
    uint64_t gpu_vaddr = 0;
    uint64_t rounded;
    uint32_t token_start;
    uint32_t token_count;
    uint32_t i;
    int ret;

    if (!out || size == 0 || !g_allocator_ready)
        return -EINVAL;

    pthread_mutex_lock(&g_alloc_lock);

    rounded = align_up_u64((uint64_t)size, g_block_size);
    if (rounded == 0 || rounded / g_block_size > UINT32_MAX) {
        pthread_mutex_unlock(&g_alloc_lock);
        return -ENOMEM;
    }

    token_count = (uint32_t)(rounded / g_block_size);
    pthread_mutex_unlock(&g_alloc_lock);
    ret = reserve_token_span(token_count, &token_start);
    if (ret != 0)
        return ret;

    alloc = calloc(1, sizeof(*alloc));
    if (!alloc) {
        return_token_span_and_reclaim(token_start, token_count);
        return -ENOMEM;
    }
    alloc->chunks = calloc(token_count, sizeof(*alloc->chunks));
    if (!alloc->chunks) {
        free(alloc);
        return_token_span_and_reclaim(token_start, token_count);
        return -ENOMEM;
    }
    alloc->requested_size = size;
    alloc->token_start = token_start;
    alloc->token_count = token_count;
    alloc->chunk_count = token_count;

    for (i = 0; i < token_count; ++i) {
        uint64_t block_id = 0;
        uint64_t chunk_vaddr = 0;
        uint64_t expected_vaddr;

        ret = polaris_shim_block_reserve(g_session_id,
                                         token_start + i,
                                         1,
                                         POLARIS_RESERVE_FLAG_DEFER_FAULT,
                                         &block_id,
                                         &chunk_vaddr);
        if (ret != 0)
            goto fail_release_chunks;

        if (i == 0) {
            gpu_vaddr = chunk_vaddr;
        } else {
            expected_vaddr = gpu_vaddr + ((uint64_t)i * g_block_size);
            if (chunk_vaddr != expected_vaddr) {
                fprintf(stderr,
                        "[polaris-shim] non-contiguous managed chunk "
                        "index=%u got=0x%" PRIx64 " expected=0x%" PRIx64 "\n",
                        i,
                        chunk_vaddr,
                        expected_vaddr);
                ret = -EINVAL;
                alloc->chunks[i].block_id = block_id;
                alloc->chunks[i].gpu_vaddr = chunk_vaddr;
                goto fail_release_chunks;
            }
        }

        alloc->chunks[i].block_id = block_id;
        alloc->chunks[i].gpu_vaddr = chunk_vaddr;
        if (i == 0)
            alloc->block_id = block_id;
    }

    if (g_create_external_ranges) {
        ret = polaris_shim_uvm_create_external_range(gpu_vaddr, rounded);
        if (ret != 0) {
            goto fail_release_chunks;
        }
        alloc->external_base = gpu_vaddr;
    }

    for (i = 0; i < token_count; ++i) {
        struct polaris_shim_allocation_chunk *chunk = &alloc->chunks[i];

        ret = polaris_shim_register_block_mapping(chunk->block_id,
                                                  g_registered_gpu_id,
                                                  g_registered_rm_client_token,
                                                  g_registered_va_space_token,
                                                  chunk->gpu_vaddr,
                                                  g_block_size);
        if (ret != 0)
            goto fail_release_external_and_chunks;

        ret = allocate_static_rm_backend(chunk->gpu_vaddr,
                                         g_block_size,
                                         chunk->block_id,
                                         &chunk->static_h_memory,
                                         &chunk->static_size);
        if (ret != 0)
            goto fail_release_external_and_chunks;
    }

    alloc->ptr = gpu_vaddr;

    pthread_mutex_lock(&g_alloc_lock);
    alloc->next = g_allocations;
    g_allocations = alloc;
    g_live_managed_allocations++;
    g_alloc_stats.managed_success_calls++;
    g_alloc_stats.managed_success_requested_bytes += (uint64_t)size;
    g_alloc_stats.managed_success_rounded_bytes += rounded;
    g_alloc_stats.live_requested_bytes += (uint64_t)size;
    g_alloc_stats.live_rounded_bytes += rounded;
    if (g_alloc_stats.live_requested_bytes > g_alloc_stats.peak_live_requested_bytes)
        g_alloc_stats.peak_live_requested_bytes = g_alloc_stats.live_requested_bytes;
    if (g_alloc_stats.live_rounded_bytes > g_alloc_stats.peak_live_rounded_bytes)
        g_alloc_stats.peak_live_rounded_bytes = g_alloc_stats.live_rounded_bytes;
    pthread_mutex_unlock(&g_alloc_lock);

    *out = gpu_vaddr;
    fprintf(stderr,
            "[polaris-shim] managed allocation size=%zu rounded=0x%" PRIx64
            " ptr=0x%" PRIx64 " blocks=%u first_block=%" PRIu64 "\n",
            size, rounded, gpu_vaddr, token_count, alloc->block_id);
    return 0;

fail_release_external_and_chunks:
    if (alloc->external_base != 0) {
        (void)polaris_shim_uvm_free_external_range(alloc->external_base);
        alloc->external_base = 0;
    }
fail_release_chunks:
    (void)release_allocation_chunks(alloc);
    free_allocation_chunks_storage(alloc);
    free(alloc);
    return_token_span_and_reclaim(token_start, token_count);
    return ret;
}

static int polaris_free_managed(CUdeviceptr ptr)
{
    struct polaris_shim_allocation **link = NULL;
    struct polaris_shim_allocation *alloc;
    uint32_t token_start;
    uint32_t token_count;
    uint64_t external_base;
    uint32_t chunk_count;
    uint64_t first_block_id;
    size_t requested_size;
    uint64_t rounded_size;
    int ret;

    pthread_mutex_lock(&g_alloc_lock);
    alloc = find_allocation(ptr, &link);
    if (!alloc) {
        pthread_mutex_unlock(&g_alloc_lock);
        return -ENOENT;
    }
    token_start = alloc->token_start;
    token_count = alloc->token_count;
    external_base = alloc->external_base;
    chunk_count = alloc->chunk_count;
    first_block_id = alloc->block_id;
    requested_size = alloc->requested_size;
    rounded_size = (uint64_t)token_count * g_block_size;
    pthread_mutex_unlock(&g_alloc_lock);

    ret = release_allocation_chunks(alloc);
    if (ret == 0 && external_base != 0)
        ret = polaris_shim_uvm_free_external_range(external_base);
    if (ret == 0) {
        pthread_mutex_lock(&g_alloc_lock);
        alloc = find_allocation(ptr, &link);
        if (alloc)
            *link = alloc->next;
        if (g_live_managed_allocations != 0)
            g_live_managed_allocations--;
        g_alloc_stats.managed_free_calls++;
        g_alloc_stats.managed_free_requested_bytes += (uint64_t)requested_size;
        g_alloc_stats.managed_free_rounded_bytes += rounded_size;
        if (g_alloc_stats.live_requested_bytes >= requested_size)
            g_alloc_stats.live_requested_bytes -= (uint64_t)requested_size;
        else
            g_alloc_stats.live_requested_bytes = 0;
        if (g_alloc_stats.live_rounded_bytes >= rounded_size)
            g_alloc_stats.live_rounded_bytes -= rounded_size;
        else
            g_alloc_stats.live_rounded_bytes = 0;
        pthread_mutex_unlock(&g_alloc_lock);
        return_token_span_and_reclaim(token_start, token_count);
        if (alloc) {
            free_allocation_chunks_storage(alloc);
            free(alloc);
        }
    }
    fprintf(stderr,
            "[polaris-shim] managed free ptr=0x%" PRIx64
            " blocks=%u first_block=%" PRIu64 " ret=%d\n",
            (uint64_t)ptr, chunk_count, first_block_id, ret);
    return ret;
}

static int has_live_managed_allocations(void)
{
    int live;

    pthread_mutex_lock(&g_alloc_lock);
    live = g_live_managed_allocations != 0;
    pthread_mutex_unlock(&g_alloc_lock);
    return live;
}

static int cu_launch_config_uses_pdl(const CUlaunchConfig *config)
{
    unsigned int i;

    if (!config || !config->attrs || config->numAttrs == 0)
        return 0;

    for (i = 0; i < config->numAttrs; ++i) {
        const CUlaunchAttribute *attr = &config->attrs[i];

        switch (attr->id) {
        case CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION:
            if (attr->value.programmaticStreamSerializationAllowed != 0)
                return 1;
            break;
        case CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT:
            if (attr->value.programmaticEvent.event != NULL ||
                attr->value.programmaticEvent.flags != 0 ||
                attr->value.programmaticEvent.triggerAtBlockStart != 0)
                return 1;
            break;
        default:
            break;
        }
    }

    return 0;
}

static int cuda_launch_config_uses_pdl(const cudaLaunchConfig_t *config)
{
    unsigned int i;

    if (!config || !config->attrs || config->numAttrs == 0)
        return 0;

    for (i = 0; i < config->numAttrs; ++i) {
        const cudaLaunchAttribute *attr = &config->attrs[i];

        switch (attr->id) {
        case CUDA_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION:
            if (attr->val.programmaticStreamSerializationAllowed != 0)
                return 1;
            break;
        case CUDA_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT:
            if (attr->val.programmaticEvent.event != NULL ||
                attr->val.programmaticEvent.flags != 0 ||
                attr->val.programmaticEvent.triggerAtBlockStart != 0)
                return 1;
            break;
        default:
            break;
        }
    }

    return 0;
}

static int reject_pdl_launch_if_live(const char *name, int uses_pdl)
{
    if (!uses_pdl || !has_live_managed_allocations())
        return 0;

    fprintf(stderr,
            "[polaris-shim] %s with CUDA PDL attributes is unsupported while "
            "Polaris-managed allocations are live\n",
            name);
    return 1;
}

static void graph_capture_started(void)
{
    pthread_mutex_lock(&g_alloc_lock);
    g_active_graph_captures++;
    pthread_mutex_unlock(&g_alloc_lock);
}

static void graph_capture_finished(void)
{
    pthread_mutex_lock(&g_alloc_lock);
    if (g_active_graph_captures != 0)
        g_active_graph_captures--;
    pthread_mutex_unlock(&g_alloc_lock);
}

static CUresult cu_stream_begin_capture_impl(const char *real_name,
                                             CUstream hStream,
                                             cudaStreamCaptureMode mode)
{
    cuStreamBeginCapture_fn real;
    CUresult result;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (has_live_managed_allocations()) {
        fprintf(stderr,
                "[polaris-shim] %s is unsupported while "
                "Polaris-managed allocations are live\n",
                real_name);
        return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    }

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol(real_name);
    if (!real) {
        fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(hStream, mode);
    if (result == CUDA_SUCCESS)
        graph_capture_started();
    return result;
}

static CUresult cu_stream_end_capture_impl(const char *real_name,
                                           CUstream hStream,
                                           cudaGraph_t *phGraph)
{
    cuStreamEndCapture_fn real;
    const char *alias = strcmp(real_name, "cuStreamEndCapture_v2") == 0
                            ? "cuStreamEndCapture"
                            : NULL;
    CUresult result;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (has_live_managed_allocations()) {
        if (phGraph)
            *phGraph = NULL;
        fprintf(stderr,
                "[polaris-shim] %s is unsupported while "
                "Polaris-managed allocations are live\n",
                real_name);
        return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    }

    *(void **)(&real) = resolve_cuda_symbol_with_alias(real_name, alias);
    if (!real) {
        fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(hStream, phGraph);
    graph_capture_finished();
    return result;
}

POLARIS_SHIM_INTERPOSER
CUresult cuInit(unsigned int Flags)
{
    CUresult result;

    pthread_once(&g_announce_once, announce);

    // POSIX-blessed dlsym() return → fn-pointer dance to keep -Wpedantic
    // happy. Plain casts trip ISO-C's object/function-pointer rule.
    cuInit_fn real;
    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuInit");
    if (!real) {
        // libcuda was not loadable. Propagate a not-initialised result so
        // the worker fails loudly rather than crashing on a NULL call.
        fprintf(stderr, "[polaris-shim] cuInit: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    fprintf(stderr, "[polaris-shim] cuInit intercepted (flags=0x%x)\n", Flags);
    result = real(Flags);
    if (result == CUDA_SUCCESS)
        pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    return result;
}

POLARIS_SHIM_INTERPOSER
CUresult cuDeviceGet(CUdevice *device, int ordinal)
{
    cuDeviceGet_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuDeviceGet");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuDeviceGet: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(device, ordinal);
}

POLARIS_SHIM_INTERPOSER
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    cuDeviceGetAttribute_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuDeviceGetAttribute");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuDeviceGetAttribute: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(pi, attrib, dev);
}

POLARIS_SHIM_INTERPOSER
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
{
    cuDevicePrimaryCtxRetain_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuDevicePrimaryCtxRetain");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuDevicePrimaryCtxRetain: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(pctx, dev);
}

POLARIS_SHIM_INTERPOSER
CUresult cuCtxSetCurrent(CUcontext ctx)
{
    cuCtxSetCurrent_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuCtxSetCurrent");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuCtxSetCurrent: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(ctx);
}

static CUresult cu_mem_alloc_impl(const char *real_name,
                                  CUdeviceptr *dptr,
                                  size_t bytesize)
{
    cuMemAlloc_fn real;
    CUresult result;
    int selected = 0;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (g_manage_allocations) {
        selected = dptr && should_manage_allocation(bytesize);
        stats_note_alloc_call(POLARIS_SHIM_ALLOC_API_DRIVER, bytesize, selected);
    }

    if (selected) {
        CUdeviceptr managed = 0;
        int ret = polaris_alloc_managed(bytesize, &managed);

        if (ret == 0) {
            *dptr = managed;
            return CUDA_SUCCESS;
        }
        stats_note_managed_failure(bytesize);
        if (g_strict_managed_alloc) {
            stats_note_strict_failure();
            return ret == -ENOMEM ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
        }
    }

    if (g_manage_allocations)
        stats_note_fallback_alloc(bytesize);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol(real_name);
    if (!real) {
        fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
        if (g_manage_allocations)
            stats_note_fallback_alloc_result(0);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(dptr, bytesize);
    if (g_manage_allocations)
        stats_note_fallback_alloc_result(result == CUDA_SUCCESS);
    return result;
}

static CUresult cu_mem_alloc_async_impl(const char *real_name,
                                        CUdeviceptr *dptr,
                                        size_t bytesize,
                                        cudaStream_t hStream)
{
    cuMemAllocAsync_fn real;
    const char *alias = strcmp(real_name, "cuMemAllocAsync_v2") == 0
                            ? "cuMemAllocAsync"
                            : NULL;
    CUresult result;
    int selected = 0;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (g_manage_allocations) {
        selected = dptr && should_manage_allocation(bytesize);
        stats_note_alloc_call(POLARIS_SHIM_ALLOC_API_DRIVER_ASYNC, bytesize, selected);
    }

    if (selected) {
        CUdeviceptr managed = 0;
        int ret = polaris_alloc_managed(bytesize, &managed);

        if (ret == 0) {
            (void)hStream;
            *dptr = managed;
            return CUDA_SUCCESS;
        }
        stats_note_managed_failure(bytesize);
        if (g_strict_managed_alloc) {
            stats_note_strict_failure();
            return ret == -ENOMEM ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
        }
    }

    if (g_manage_allocations)
        stats_note_fallback_alloc(bytesize);

    *(void **)(&real) = resolve_cuda_symbol_with_alias(real_name, alias);
    if (!real) {
        fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
        if (g_manage_allocations)
            stats_note_fallback_alloc_result(0);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(dptr, bytesize, hStream);
    if (g_manage_allocations)
        stats_note_fallback_alloc_result(result == CUDA_SUCCESS);
    return result;
}

static CUresult cu_mem_free_impl(const char *real_name, CUdeviceptr dptr)
{
    cuMemFree_fn real;
    CUresult result;
    int ret;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (g_manage_allocations) {
        stats_note_free_call();
        ret = polaris_free_managed(dptr);
        if (ret == 0)
            return CUDA_SUCCESS;
        if (ret != -ENOENT)
            return CUDA_ERROR_UNKNOWN;
        stats_note_fallback_free();
    }

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol(real_name);
    if (!real) {
        fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
        if (g_manage_allocations)
            stats_note_fallback_free_result(0);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(dptr);
    if (g_manage_allocations)
        stats_note_fallback_free_result(result == CUDA_SUCCESS);
    return result;
}

static CUresult cu_mem_free_async_impl(const char *real_name,
                                       CUdeviceptr dptr,
                                       cudaStream_t hStream)
{
    cuMemFreeAsync_fn real;
    const char *alias = strcmp(real_name, "cuMemFreeAsync_v2") == 0
                            ? "cuMemFreeAsync"
                            : NULL;
    CUresult result;
    int ret;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (g_manage_allocations) {
        stats_note_free_call();
        ret = polaris_free_managed(dptr);
        if (ret == 0) {
            (void)hStream;
            return CUDA_SUCCESS;
        }
        if (ret != -ENOENT)
            return CUDA_ERROR_UNKNOWN;
        stats_note_fallback_free();
    }

    *(void **)(&real) = resolve_cuda_symbol_with_alias(real_name, alias);
    if (!real) {
        fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
        if (g_manage_allocations)
            stats_note_fallback_free_result(0);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(dptr, hStream);
    if (g_manage_allocations)
        stats_note_fallback_free_result(result == CUDA_SUCCESS);
    return result;
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
{
    return cu_mem_alloc_impl("cuMemAlloc", dptr, bytesize);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
    return cu_mem_alloc_impl("cuMemAlloc_v2", dptr, bytesize);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, cudaStream_t hStream)
{
    return cu_mem_alloc_async_impl("cuMemAllocAsync", dptr, bytesize, hStream);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemAllocAsync_v2(CUdeviceptr *dptr, size_t bytesize, cudaStream_t hStream)
{
    return cu_mem_alloc_async_impl("cuMemAllocAsync_v2", dptr, bytesize, hStream);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemFree(CUdeviceptr dptr)
{
    return cu_mem_free_impl("cuMemFree", dptr);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    return cu_mem_free_impl("cuMemFree_v2", dptr);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemFreeAsync(CUdeviceptr dptr, cudaStream_t hStream)
{
    return cu_mem_free_async_impl("cuMemFreeAsync", dptr, hStream);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemFreeAsync_v2(CUdeviceptr dptr, cudaStream_t hStream)
{
    return cu_mem_free_async_impl("cuMemFreeAsync_v2", dptr, hStream);
}

static CUresult cu_mem_get_info_impl(const char *real_name, size_t *free_bytes, size_t *total_bytes)
{
    cuMemGetInfo_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol(real_name);
    if (!real) {
        fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(free_bytes, total_bytes);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
    return cu_mem_get_info_impl("cuMemGetInfo", free_bytes, total_bytes);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemGetInfo_v2(size_t *free_bytes, size_t *total_bytes)
{
    return cu_mem_get_info_impl("cuMemGetInfo_v2", free_bytes, total_bytes);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemAddressReserve(CUdeviceptr *ptr,
                             size_t size,
                             size_t alignment,
                             CUdeviceptr addr,
                             unsigned long long flags)
{
    cuMemAddressReserve_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuMemAddressReserve");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuMemAddressReserve: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(ptr, size, alignment, addr, flags);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
{
    cuMemAddressFree_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuMemAddressFree");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuMemAddressFree: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(ptr, size);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle,
                     size_t size,
                     const CUmemAllocationProp *prop,
                     unsigned long long flags)
{
    cuMemCreate_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuMemCreate");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuMemCreate: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(handle, size, prop, flags);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
{
    cuMemRelease_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuMemRelease");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuMemRelease: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(handle);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemMap(CUdeviceptr ptr,
                  size_t size,
                  size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags)
{
    cuMemMap_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuMemMap");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuMemMap: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(ptr, size, offset, handle, flags);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    cuMemUnmap_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuMemUnmap");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuMemUnmap: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(ptr, size);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemSetAccess(CUdeviceptr ptr,
                        size_t size,
                        const CUmemAccessDesc *desc,
                        size_t count)
{
    cuMemSetAccess_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuMemSetAccess");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuMemSetAccess: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(ptr, size, desc, count);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemGetAllocationGranularity(size_t *granularity,
                                       const CUmemAllocationProp *prop,
                                       CUmemAllocationGranularity_flags option)
{
    cuMemGetAllocationGranularity_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuMemGetAllocationGranularity");
    if (!real) {
        fprintf(stderr,
                "[polaris-shim] cuMemGetAllocationGranularity: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(granularity, prop, option);
}

POLARIS_SHIM_INTERPOSER
CUresult cuGetErrorString(CUresult error, const char **pStr)
{
    cuGetErrorString_fn real;

    pthread_once(&g_announce_once, announce);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuGetErrorString");
    if (!real)
        return CUDA_ERROR_NOT_INITIALIZED;
    return real(error, pStr);
}

POLARIS_SHIM_INTERPOSER
CUresult cuGetErrorName(CUresult error, const char **pStr)
{
    cuGetErrorName_fn real;

    pthread_once(&g_announce_once, announce);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuGetErrorName");
    if (!real)
        return CUDA_ERROR_NOT_INITIALIZED;
    return real(error, pStr);
}

static CUresult cu_mem_get_address_range_impl(const char *real_name,
                                              CUdeviceptr *pbase,
                                              size_t *psize,
                                              CUdeviceptr dptr)
{
    cuMemGetAddressRange_fn real;
    int ret;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    ret = get_managed_address_range(pbase, psize, dptr);
    if (ret == 0)
        return CUDA_SUCCESS;
    if (ret != -ENOENT)
        return CUDA_ERROR_UNKNOWN;

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol(real_name);
    if (!real) {
        fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(pbase, psize, dptr);
}

POLARIS_SHIM_INTERPOSER
void polaris_shim_set_allocation_scope(const char *scope)
{
    if (scope && strcmp(scope, "kv") == 0)
        g_allocation_scope_is_kv = 1;
    else
        g_allocation_scope_is_kv = 0;
    if (g_trace_scope) {
        fprintf(stderr,
                "[polaris-shim] allocation scope=%s scope_kv=%d\n",
                scope ? scope : "none",
                g_allocation_scope_is_kv);
    }
}

POLARIS_SHIM_INTERPOSER
ggml_backend_buffer_t
ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context *ctx,
                                         ggml_backend_buffer_type_t buft)
{
    ggml_backend_alloc_ctx_tensors_from_buft_fn real;
    int previous_scope;
    int kv_scope;
    ggml_backend_buffer_t result;

    *(void **)(&real) = dlsym(RTLD_NEXT, "ggml_backend_alloc_ctx_tensors_from_buft");
    if (!real) {
        fprintf(stderr,
                "[polaris-shim] ggml_backend_alloc_ctx_tensors_from_buft: "
                "real symbol unavailable\n");
        return NULL;
    }

    previous_scope = g_allocation_scope_is_kv;
    kv_scope = ggml_context_has_kv_cache_tensors(ctx);
    if (kv_scope)
        g_allocation_scope_is_kv = 1;

    result = real(ctx, buft);

    g_allocation_scope_is_kv = previous_scope;
    return result;
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemGetAddressRange(CUdeviceptr *pbase,
                              size_t *psize,
                              CUdeviceptr dptr)
{
    return cu_mem_get_address_range_impl("cuMemGetAddressRange", pbase, psize, dptr);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase,
                                 size_t *psize,
                                 CUdeviceptr dptr)
{
    return cu_mem_get_address_range_impl("cuMemGetAddressRange_v2", pbase, psize, dptr);
}

POLARIS_SHIM_INTERPOSER
CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr)
{
    cuIpcGetMemHandle_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (find_allocation_range(dptr, NULL, NULL)) {
        (void)pHandle;
        fprintf(stderr,
                "[polaris-shim] cuIpcGetMemHandle is unsupported for Polaris pointer 0x%" PRIx64 "\n",
                (uint64_t)dptr);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuIpcGetMemHandle");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuIpcGetMemHandle: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(pHandle, dptr);
}

static CUresult cu_memcpy_htod_impl(const char *real_name,
                                    CUdeviceptr dstDevice,
                                    const void *srcHost,
                                    size_t ByteCount,
                                    cudaStream_t stream,
                                    int is_async)
{
    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (find_allocation_range(dstDevice, NULL, NULL)) {
        (void)srcHost;
        (void)ByteCount;
        (void)stream;
        (void)is_async;
        fprintf(stderr,
                "[polaris-shim] %s involving Polaris pointer is not wired yet "
                "(dst=0x%" PRIx64 " bytes=%zu)\n",
                real_name,
                (uint64_t)dstDevice,
                ByteCount);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (is_async) {
        cuMemcpyHtoDAsync_fn real_async;

        *(void **)(&real_async) = polaris_shim_resolve_cuda_symbol(real_name);
        if (!real_async) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real_async(dstDevice, srcHost, ByteCount, stream);
    } else {
        cuMemcpyHtoD_fn real;

        *(void **)(&real) = polaris_shim_resolve_cuda_symbol(real_name);
        if (!real) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real(dstDevice, srcHost, ByteCount);
    }
}

static CUresult cu_memcpy_dtoh_impl(const char *real_name,
                                    void *dstHost,
                                    CUdeviceptr srcDevice,
                                    size_t ByteCount,
                                    cudaStream_t stream,
                                    int is_async)
{
    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (find_allocation_range(srcDevice, NULL, NULL)) {
        (void)dstHost;
        (void)ByteCount;
        (void)stream;
        (void)is_async;
        fprintf(stderr,
                "[polaris-shim] %s involving Polaris pointer is not wired yet "
                "(src=0x%" PRIx64 " bytes=%zu)\n",
                real_name,
                (uint64_t)srcDevice,
                ByteCount);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (is_async) {
        cuMemcpyDtoHAsync_fn real_async;

        *(void **)(&real_async) = polaris_shim_resolve_cuda_symbol(real_name);
        if (!real_async) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real_async(dstHost, srcDevice, ByteCount, stream);
    } else {
        cuMemcpyDtoH_fn real;

        *(void **)(&real) = polaris_shim_resolve_cuda_symbol(real_name);
        if (!real) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real(dstHost, srcDevice, ByteCount);
    }
}

static CUresult cu_memcpy_impl(const char *real_name,
                               CUdeviceptr dst,
                               CUdeviceptr src,
                               size_t ByteCount,
                               cudaStream_t stream,
                               int is_async)
{
    int dst_is_polaris;
    int src_is_polaris;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    dst_is_polaris = find_allocation_range(dst, NULL, NULL);
    src_is_polaris = find_allocation_range(src, NULL, NULL);
    if (dst_is_polaris || src_is_polaris) {
        (void)stream;
        (void)is_async;
        fprintf(stderr,
                "[polaris-shim] %s involving Polaris pointer is not wired yet "
                "(dst=0x%" PRIx64 " src=0x%" PRIx64 " bytes=%zu)\n",
                real_name,
                (uint64_t)dst,
                (uint64_t)src,
                ByteCount);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (is_async) {
        cuMemcpyAsync_fn real_async;

        *(void **)(&real_async) = polaris_shim_resolve_cuda_symbol(real_name);
        if (!real_async) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real_async(dst, src, ByteCount, stream);
    } else {
        cuMemcpy_fn real;

        *(void **)(&real) = polaris_shim_resolve_cuda_symbol(real_name);
        if (!real) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real(dst, src, ByteCount);
    }
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
    return cu_memcpy_htod_impl("cuMemcpyHtoD", dstDevice, srcHost, ByteCount, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
    return cu_memcpy_htod_impl("cuMemcpyHtoD_v2", dstDevice, srcHost, ByteCount, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    return cu_memcpy_dtoh_impl("cuMemcpyDtoH", dstHost, srcDevice, ByteCount, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    return cu_memcpy_dtoh_impl("cuMemcpyDtoH_v2", dstHost, srcDevice, ByteCount, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice,
                           const void *srcHost,
                           size_t ByteCount,
                           cudaStream_t hStream)
{
    return cu_memcpy_htod_impl("cuMemcpyHtoDAsync", dstDevice, srcHost, ByteCount, hStream, 1);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                              const void *srcHost,
                              size_t ByteCount,
                              cudaStream_t hStream)
{
    return cu_memcpy_htod_impl("cuMemcpyHtoDAsync_v2", dstDevice, srcHost, ByteCount, hStream, 1);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemcpyDtoHAsync(void *dstHost,
                           CUdeviceptr srcDevice,
                           size_t ByteCount,
                           cudaStream_t hStream)
{
    return cu_memcpy_dtoh_impl("cuMemcpyDtoHAsync", dstHost, srcDevice, ByteCount, hStream, 1);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemcpyDtoHAsync_v2(void *dstHost,
                              CUdeviceptr srcDevice,
                              size_t ByteCount,
                              cudaStream_t hStream)
{
    return cu_memcpy_dtoh_impl("cuMemcpyDtoHAsync_v2", dstHost, srcDevice, ByteCount, hStream, 1);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    return cu_memcpy_impl("cuMemcpy", dst, src, ByteCount, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemcpy_v2(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    return cu_memcpy_impl("cuMemcpy_v2", dst, src, ByteCount, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemcpyAsync(CUdeviceptr dst,
                       CUdeviceptr src,
                       size_t ByteCount,
                       cudaStream_t hStream)
{
    return cu_memcpy_impl("cuMemcpyAsync", dst, src, ByteCount, hStream, 1);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemcpyAsync_v2(CUdeviceptr dst,
                          CUdeviceptr src,
                          size_t ByteCount,
                          cudaStream_t hStream)
{
    return cu_memcpy_impl("cuMemcpyAsync_v2", dst, src, ByteCount, hStream, 1);
}

static CUresult cu_memset_guard(const char *real_name,
                                CUdeviceptr dstDevice,
                                uint32_t value,
                                size_t N)
{
    CUdeviceptr base = 0;
    uint64_t length = 0;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (!find_allocation_range(dstDevice, &base, &length))
        return CUDA_SUCCESS;

    if (g_allow_zero_memset && value == 0 && dstDevice == base && N <= length) {
        fprintf(stderr,
                "[polaris-shim] %s zero-fill accepted for Polaris pointer "
                "0x%" PRIx64 " bytes=%zu\n",
                real_name,
                (uint64_t)dstDevice,
                N);
        return CUDA_SUCCESS;
    }

    fprintf(stderr,
            "[polaris-shim] %s involving Polaris pointer is not wired yet "
            "(dst=0x%" PRIx64 " value=%u bytes=%zu)\n",
            real_name,
            (uint64_t)dstDevice,
            value,
            N);
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult cu_memset_d8_impl(const char *real_name,
                                  CUdeviceptr dstDevice,
                                  unsigned char uc,
                                  size_t N,
                                  cudaStream_t stream,
                                  int is_async)
{
    CUresult guard = cu_memset_guard(real_name, dstDevice, (uint32_t)uc, N);

    if (guard != CUDA_SUCCESS)
        return guard;
    if (find_allocation_range(dstDevice, NULL, NULL))
        return CUDA_SUCCESS;

    if (is_async) {
        cuMemsetD8Async_fn real_async;

        *(void **)(&real_async) = polaris_shim_resolve_cuda_symbol(real_name);
        if (!real_async) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real_async(dstDevice, uc, N, stream);
    } else {
        cuMemsetD8_fn real;

        *(void **)(&real) = polaris_shim_resolve_cuda_symbol(real_name);
        if (!real) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real(dstDevice, uc, N);
    }
}

static CUresult cu_memset_d16_impl(const char *real_name,
                                   CUdeviceptr dstDevice,
                                   unsigned short us,
                                   size_t N,
                                   cudaStream_t stream,
                                   int is_async)
{
    CUresult guard = cu_memset_guard(real_name, dstDevice, (uint32_t)us, N * sizeof(us));

    if (guard != CUDA_SUCCESS)
        return guard;
    if (find_allocation_range(dstDevice, NULL, NULL))
        return CUDA_SUCCESS;

    if (is_async) {
        cuMemsetD16Async_fn real_async;

        *(void **)(&real_async) = polaris_shim_resolve_cuda_symbol(real_name);
        if (!real_async) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real_async(dstDevice, us, N, stream);
    } else {
        cuMemsetD16_fn real;

        *(void **)(&real) = polaris_shim_resolve_cuda_symbol(real_name);
        if (!real) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real(dstDevice, us, N);
    }
}

static CUresult cu_memset_d32_impl(const char *real_name,
                                   CUdeviceptr dstDevice,
                                   unsigned int ui,
                                   size_t N,
                                   cudaStream_t stream,
                                   int is_async)
{
    CUresult guard = cu_memset_guard(real_name, dstDevice, ui, N * sizeof(ui));

    if (guard != CUDA_SUCCESS)
        return guard;
    if (find_allocation_range(dstDevice, NULL, NULL))
        return CUDA_SUCCESS;

    if (is_async) {
        cuMemsetD32Async_fn real_async;

        *(void **)(&real_async) = polaris_shim_resolve_cuda_symbol(real_name);
        if (!real_async) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real_async(dstDevice, ui, N, stream);
    } else {
        cuMemsetD32_fn real;

        *(void **)(&real) = polaris_shim_resolve_cuda_symbol(real_name);
        if (!real) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real(dstDevice, ui, N);
    }
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    return cu_memset_d8_impl("cuMemsetD8", dstDevice, uc, N, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    return cu_memset_d8_impl("cuMemsetD8_v2", dstDevice, uc, N, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, cudaStream_t hStream)
{
    return cu_memset_d8_impl("cuMemsetD8Async", dstDevice, uc, N, hStream, 1);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemsetD8Async_v2(CUdeviceptr dstDevice,
                            unsigned char uc,
                            size_t N,
                            cudaStream_t hStream)
{
    return cu_memset_d8_impl("cuMemsetD8Async_v2", dstDevice, uc, N, hStream, 1);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    return cu_memset_d16_impl("cuMemsetD16", dstDevice, us, N, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    return cu_memset_d16_impl("cuMemsetD16_v2", dstDevice, us, N, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemsetD16Async(CUdeviceptr dstDevice,
                          unsigned short us,
                          size_t N,
                          cudaStream_t hStream)
{
    return cu_memset_d16_impl("cuMemsetD16Async", dstDevice, us, N, hStream, 1);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemsetD16Async_v2(CUdeviceptr dstDevice,
                             unsigned short us,
                             size_t N,
                             cudaStream_t hStream)
{
    return cu_memset_d16_impl("cuMemsetD16Async_v2", dstDevice, us, N, hStream, 1);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    return cu_memset_d32_impl("cuMemsetD32", dstDevice, ui, N, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    return cu_memset_d32_impl("cuMemsetD32_v2", dstDevice, ui, N, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemsetD32Async(CUdeviceptr dstDevice,
                          unsigned int ui,
                          size_t N,
                          cudaStream_t hStream)
{
    return cu_memset_d32_impl("cuMemsetD32Async", dstDevice, ui, N, hStream, 1);
}

POLARIS_SHIM_INTERPOSER
CUresult cuMemsetD32Async_v2(CUdeviceptr dstDevice,
                             unsigned int ui,
                             size_t N,
                             cudaStream_t hStream)
{
    return cu_memset_d32_impl("cuMemsetD32Async_v2", dstDevice, ui, N, hStream, 1);
}

POLARIS_SHIM_INTERPOSER
CUresult cuPointerGetAttribute(void *data, int attribute, CUdeviceptr ptr)
{
    cuPointerGetAttribute_fn real;
    int ret;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    ret = get_managed_pointer_attribute(data, attribute, ptr);
    if (ret == 0)
        return CUDA_SUCCESS;
    if (ret == -EINVAL)
        return CUDA_ERROR_INVALID_VALUE;
    if (ret != -ENOENT && ret != -EOPNOTSUPP)
        return CUDA_ERROR_UNKNOWN;

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuPointerGetAttribute");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuPointerGetAttribute: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(data, attribute, ptr);
}

POLARIS_SHIM_INTERPOSER
CUresult cuPointerGetAttributes(unsigned int numAttributes,
                                int *attributes,
                                void **data,
                                CUdeviceptr ptr)
{
    cuPointerGetAttributes_fn real;
    unsigned int i;
    int saw_managed = 0;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if ((!attributes || !data) && numAttributes != 0)
        return CUDA_ERROR_INVALID_VALUE;

    if (find_allocation_range(ptr, NULL, NULL)) {
        for (i = 0; i < numAttributes; ++i) {
            int ret = get_managed_pointer_attribute(data[i], attributes[i], ptr);
            if (ret == 0) {
                saw_managed = 1;
                continue;
            }
            if (ret == -EINVAL)
                return CUDA_ERROR_INVALID_VALUE;
            if (ret == -EOPNOTSUPP)
                goto fallback;
            return CUDA_ERROR_UNKNOWN;
        }
        if (saw_managed || numAttributes == 0)
            return CUDA_SUCCESS;
    }

fallback:
    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuPointerGetAttributes");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuPointerGetAttributes: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(numAttributes, attributes, data, ptr);
}

POLARIS_SHIM_INTERPOSER
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,
                        unsigned int gridDimY,
                        unsigned int gridDimZ,
                        unsigned int blockDimX,
                        unsigned int blockDimY,
                        unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void **kernelParams,
                        void **extra)
{
    cuLaunchKernel_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuLaunchKernel");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuLaunchKernel: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(f,
                gridDimX,
                gridDimY,
                gridDimZ,
                blockDimX,
                blockDimY,
                blockDimZ,
                sharedMemBytes,
                hStream,
                kernelParams,
                extra);
}

POLARIS_SHIM_INTERPOSER
CUresult cuLaunchKernelEx(const CUlaunchConfig *config,
                          CUfunction f,
                          void **kernelParams,
                          void **extra)
{
    cuLaunchKernelEx_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (reject_pdl_launch_if_live("cuLaunchKernelEx",
                                  cu_launch_config_uses_pdl(config)))
        return CUDA_ERROR_NOT_SUPPORTED;

    *(void **)(&real) = polaris_shim_resolve_cuda_symbol("cuLaunchKernelEx");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cuLaunchKernelEx: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(config, f, kernelParams, extra);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    cudaMalloc_fn real;
    cudaError_t result;
    int selected = 0;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (g_manage_allocations) {
        selected = devPtr && should_manage_allocation(size);
        stats_note_alloc_call(POLARIS_SHIM_ALLOC_API_RUNTIME, size, selected);
    }

    if (selected) {
        CUdeviceptr managed = 0;
        int ret = polaris_alloc_managed(size, &managed);

        if (ret == 0) {
            *devPtr = (void *)(uintptr_t)managed;
            return CUDA_SUCCESS;
        }
        stats_note_managed_failure(size);
        if (g_strict_managed_alloc) {
            stats_note_strict_failure();
            return ret == -ENOMEM ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
        }
    }

    if (g_manage_allocations)
        stats_note_fallback_alloc(size);

    *(void **)(&real) = resolve_cudart_symbol("cudaMalloc");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaMalloc: real symbol unavailable\n");
        if (g_manage_allocations)
            stats_note_fallback_alloc_result(0);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(devPtr, size);
    if (g_manage_allocations)
        stats_note_fallback_alloc_result(result == CUDA_SUCCESS);
    return result;
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
{
    cudaMallocManaged_fn real;
    cudaError_t result;
    int selected = 0;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (g_manage_allocations) {
        selected = devPtr && should_manage_allocation(size);
        stats_note_alloc_call(POLARIS_SHIM_ALLOC_API_RUNTIME_MANAGED, size, selected);
    }

    if (selected) {
        CUdeviceptr managed = 0;
        int ret = polaris_alloc_managed(size, &managed);

        if (ret == 0) {
            (void)flags;
            *devPtr = (void *)(uintptr_t)managed;
            return CUDA_SUCCESS;
        }
        stats_note_managed_failure(size);
        if (g_strict_managed_alloc) {
            stats_note_strict_failure();
            return ret == -ENOMEM ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
        }
    }

    if (g_manage_allocations)
        stats_note_fallback_alloc(size);

    *(void **)(&real) = resolve_cudart_symbol("cudaMallocManaged");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaMallocManaged: real symbol unavailable\n");
        if (g_manage_allocations)
            stats_note_fallback_alloc_result(0);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(devPtr, size, flags);
    if (g_manage_allocations)
        stats_note_fallback_alloc_result(result == CUDA_SUCCESS);
    return result;
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaFree(void *devPtr)
{
    cudaFree_fn real;
    CUdeviceptr ptr = (CUdeviceptr)(uintptr_t)devPtr;
    cudaError_t result;
    int ret;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (g_manage_allocations) {
        stats_note_free_call();
        ret = polaris_free_managed(ptr);
        if (ret == 0)
            return CUDA_SUCCESS;
        if (ret != -ENOENT)
            return CUDA_ERROR_UNKNOWN;
        stats_note_fallback_free();
    }

    *(void **)(&real) = resolve_cudart_symbol("cudaFree");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaFree: real symbol unavailable\n");
        if (g_manage_allocations)
            stats_note_fallback_free_result(0);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(devPtr);
    if (g_manage_allocations)
        stats_note_fallback_free_result(result == CUDA_SUCCESS);
    return result;
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream)
{
    cudaMallocAsync_fn real;
    cudaError_t result;
    int selected = 0;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (g_manage_allocations) {
        selected = devPtr && should_manage_allocation(size);
        stats_note_alloc_call(POLARIS_SHIM_ALLOC_API_RUNTIME_ASYNC, size, selected);
    }

    if (selected) {
        CUdeviceptr managed = 0;
        int ret = polaris_alloc_managed(size, &managed);

        if (ret == 0) {
            (void)stream;
            *devPtr = (void *)(uintptr_t)managed;
            return CUDA_SUCCESS;
        }
        stats_note_managed_failure(size);
        if (g_strict_managed_alloc) {
            stats_note_strict_failure();
            return ret == -ENOMEM ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
        }
    }

    if (g_manage_allocations)
        stats_note_fallback_alloc(size);

    *(void **)(&real) = resolve_cudart_symbol("cudaMallocAsync");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaMallocAsync: real symbol unavailable\n");
        if (g_manage_allocations)
            stats_note_fallback_alloc_result(0);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(devPtr, size, stream);
    if (g_manage_allocations)
        stats_note_fallback_alloc_result(result == CUDA_SUCCESS);
    return result;
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream)
{
    cudaFreeAsync_fn real;
    CUdeviceptr ptr = (CUdeviceptr)(uintptr_t)devPtr;
    cudaError_t result;
    int ret;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (g_manage_allocations) {
        stats_note_free_call();
        ret = polaris_free_managed(ptr);
        if (ret == 0) {
            (void)stream;
            return CUDA_SUCCESS;
        }
        if (ret != -ENOENT)
            return CUDA_ERROR_UNKNOWN;
        stats_note_fallback_free();
    }

    *(void **)(&real) = resolve_cudart_symbol("cudaFreeAsync");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaFreeAsync: real symbol unavailable\n");
        if (g_manage_allocations)
            stats_note_fallback_free_result(0);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(devPtr, stream);
    if (g_manage_allocations)
        stats_note_fallback_free_result(result == CUDA_SUCCESS);
    return result;
}

static void bootstrap_for_runtime_call(void)
{
    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGetDevice(int *device)
{
    cudaGetDevice_fn real;
    cudaError_t result;

    pthread_once(&g_announce_once, announce);

    *(void **)(&real) = resolve_cudart_symbol("cudaGetDevice");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGetDevice: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(device);
    if (result == CUDA_SUCCESS && device) {
        g_runtime_selected_device = *device;
        pthread_once(&g_bootstrap_once, bootstrap_vaspace);
    }
    return result;
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaSetDevice(int device)
{
    cudaSetDevice_fn real;
    cudaError_t result;

    pthread_once(&g_announce_once, announce);

    *(void **)(&real) = resolve_cudart_symbol("cudaSetDevice");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaSetDevice: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(device);
    if (result == CUDA_SUCCESS)
        g_runtime_selected_device = device;
    return result;
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
    cudaSetDeviceFlags_fn real;

    pthread_once(&g_announce_once, announce);

    *(void **)(&real) = resolve_cudart_symbol("cudaSetDeviceFlags");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaSetDeviceFlags: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(flags);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGetDeviceFlags(unsigned int *flags)
{
    cudaGetDeviceFlags_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaGetDeviceFlags");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGetDeviceFlags: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(flags);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGetDeviceCount(int *count)
{
    cudaGetDeviceCount_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaGetDeviceCount");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGetDeviceCount: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(count);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGetDeviceProperties(void *prop, int device)
{
    cudaGetDeviceProperties_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaGetDeviceProperties");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGetDeviceProperties: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(prop, device);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device)
{
    cudaDeviceGetAttribute_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaDeviceGetAttribute");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaDeviceGetAttribute: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(value, attr, device);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)
{
    cudaDeviceCanAccessPeer_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaDeviceCanAccessPeer");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaDeviceCanAccessPeer: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(canAccessPeer, device, peerDevice);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    cudaDeviceEnablePeerAccess_fn real;
    cudaError_t result;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaDeviceEnablePeerAccess");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaDeviceEnablePeerAccess: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(peerDevice, flags);
    if (result == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
        return CUDA_SUCCESS;
    return result;
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice)
{
    cudaDeviceDisablePeerAccess_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaDeviceDisablePeerAccess");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaDeviceDisablePeerAccess: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(peerDevice);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device)
{
    cudaDeviceGetPCIBusId_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaDeviceGetPCIBusId");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaDeviceGetPCIBusId: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(pciBusId, len, device);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
    cudaRuntimeGetVersion_fn real;

    pthread_once(&g_announce_once, announce);

    *(void **)(&real) = resolve_cudart_symbol("cudaRuntimeGetVersion");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaRuntimeGetVersion: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(runtimeVersion);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaDriverGetVersion(int *driverVersion)
{
    cudaDriverGetVersion_fn real;

    pthread_once(&g_announce_once, announce);

    *(void **)(&real) = resolve_cudart_symbol("cudaDriverGetVersion");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaDriverGetVersion: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(driverVersion);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaDeviceSynchronize(void)
{
    cudaDeviceSynchronize_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaDeviceSynchronize");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaDeviceSynchronize: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real();
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
    cudaMemGetInfo_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaMemGetInfo");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaMemGetInfo: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(free_bytes, total_bytes);
}

POLARIS_SHIM_INTERPOSER
const char *cudaGetErrorString(cudaError_t error)
{
    cudaGetErrorString_fn real;

    pthread_once(&g_announce_once, announce);

    *(void **)(&real) = resolve_cudart_symbol("cudaGetErrorString");
    if (!real)
        return "polaris-shim: cudaGetErrorString unavailable";
    return real(error);
}

POLARIS_SHIM_INTERPOSER
const char *cudaGetErrorName(cudaError_t error)
{
    cudaGetErrorName_fn real;

    pthread_once(&g_announce_once, announce);

    *(void **)(&real) = resolve_cudart_symbol("cudaGetErrorName");
    if (!real)
        return "CUDA_ERROR_UNKNOWN";
    return real(error);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGetLastError(void)
{
    cudaGetLastError_fn real;

    pthread_once(&g_announce_once, announce);

    *(void **)(&real) = resolve_cudart_symbol("cudaGetLastError");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGetLastError: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real();
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaPeekAtLastError(void)
{
    cudaPeekAtLastError_fn real;

    pthread_once(&g_announce_once, announce);

    *(void **)(&real) = resolve_cudart_symbol("cudaPeekAtLastError");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaPeekAtLastError: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real();
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
    cudaStreamCreate_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaStreamCreate");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaStreamCreate: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(pStream);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
    cudaStreamCreateWithFlags_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaStreamCreateWithFlags");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaStreamCreateWithFlags: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(pStream, flags);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    cudaStreamDestroy_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaStreamDestroy");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaStreamDestroy: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(stream);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    cudaStreamSynchronize_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaStreamSynchronize");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaStreamSynchronize: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(stream);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    cudaStreamWaitEvent_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaStreamWaitEvent");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaStreamWaitEvent: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(stream, event, flags);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus)
{
    cudaStreamIsCapturing_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaStreamIsCapturing");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaStreamIsCapturing: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(stream, pCaptureStatus);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    cudaEventCreate_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaEventCreate");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaEventCreate: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(event);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    cudaEventCreateWithFlags_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaEventCreateWithFlags");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaEventCreateWithFlags: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(event, flags);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    cudaEventRecord_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaEventRecord");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaEventRecord: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(event, stream);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    cudaEventSynchronize_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaEventSynchronize");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaEventSynchronize: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(event);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    cudaEventDestroy_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaEventDestroy");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaEventDestroy: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(event);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    cudaEventElapsedTime_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaEventElapsedTime");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaEventElapsedTime: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(ms, start, end);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    cudaMallocHost_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaMallocHost");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaMallocHost: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(ptr, size);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    cudaHostAlloc_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaHostAlloc");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaHostAlloc: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(pHost, size, flags);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaFreeHost(void *ptr)
{
    cudaFreeHost_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaFreeHost");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaFreeHost: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(ptr);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
    cudaHostGetDevicePointer_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaHostGetDevicePointer");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaHostGetDevicePointer: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(pDevice, pHost, flags);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
    cudaHostRegister_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaHostRegister");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaHostRegister: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(ptr, size, flags);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaHostUnregister(void *ptr)
{
    cudaHostUnregister_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaHostUnregister");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaHostUnregister: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(ptr);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaFuncSetAttribute(const void *func, int attr, int value)
{
    cudaFuncSetAttribute_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaFuncSetAttribute");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaFuncSetAttribute: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(func, attr, value);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaFuncGetAttributes(void *attr, const void *func)
{
    cudaFuncGetAttributes_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaFuncGetAttributes");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaFuncGetAttributes: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(attr, func);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                          const void *func,
                                                          int blockSize,
                                                          size_t dynamicSMemSize)
{
    cudaOccupancyMaxActiveBlocksPerMultiprocessor_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    if (!real) {
        fprintf(stderr,
                "[polaris-shim] cudaOccupancyMaxActiveBlocksPerMultiprocessor: "
                "real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(numBlocks, func, blockSize, dynamicSMemSize);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaOccupancyMaxPotentialBlockSize(int *minGridSize,
                                               int *blockSize,
                                               const void *func,
                                               size_t dynamicSMemSize,
                                               int blockSizeLimit)
{
    cudaOccupancyMaxPotentialBlockSize_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaOccupancyMaxPotentialBlockSize");
    if (!real) {
        fprintf(stderr,
                "[polaris-shim] cudaOccupancyMaxPotentialBlockSize: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(minGridSize, blockSize, func, dynamicSMemSize, blockSizeLimit);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaLaunchKernel(const void *func,
                             dim3 gridDim,
                             dim3 blockDim,
                             void **args,
                             size_t sharedMem,
                             cudaStream_t stream)
{
    cudaLaunchKernel_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaLaunchKernel");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaLaunchKernel: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(func, gridDim, blockDim, args, sharedMem, stream);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaLaunchCooperativeKernel(const void *func,
                                        dim3 gridDim,
                                        dim3 blockDim,
                                        void **args,
                                        size_t sharedMem,
                                        cudaStream_t stream)
{
    cudaLaunchCooperativeKernel_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaLaunchCooperativeKernel");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaLaunchCooperativeKernel: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(func, gridDim, blockDim, args, sharedMem, stream);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args)
{
    cudaLaunchKernelExC_fn real;

    bootstrap_for_runtime_call();

    if (reject_pdl_launch_if_live("cudaLaunchKernelExC",
                                  cuda_launch_config_uses_pdl(config)))
        return CUDA_ERROR_NOT_SUPPORTED;

    *(void **)(&real) = resolve_cudart_symbol("cudaLaunchKernelExC");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaLaunchKernelExC: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(config, func, args);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData)
{
    cudaLaunchHostFunc_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaLaunchHostFunc");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaLaunchHostFunc: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(stream, fn, userData);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaLaunchHostFunc_v2(cudaStream_t stream,
                                  cudaHostFn_t fn,
                                  void *userData,
                                  unsigned int syncMode)
{
    cudaLaunchHostFunc_v2_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaLaunchHostFunc_v2");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaLaunchHostFunc_v2: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(stream, fn, userData, syncMode);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)
{
    cudaStreamBeginCapture_fn real;
    cudaError_t result;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (has_live_managed_allocations()) {
        fprintf(stderr,
                "[polaris-shim] cudaStreamBeginCapture is unsupported while "
                "Polaris-managed allocations are live\n");
        return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    }

    *(void **)(&real) = resolve_cudart_symbol("cudaStreamBeginCapture");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaStreamBeginCapture: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(stream, mode);
    if (result == CUDA_SUCCESS)
        graph_capture_started();
    return result;
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph)
{
    cudaStreamEndCapture_fn real;
    cudaError_t result;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (has_live_managed_allocations()) {
        if (pGraph)
            *pGraph = NULL;
        fprintf(stderr,
                "[polaris-shim] cudaStreamEndCapture is unsupported while "
                "Polaris-managed allocations are live\n");
        return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    }

    *(void **)(&real) = resolve_cudart_symbol("cudaStreamEndCapture");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaStreamEndCapture: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    result = real(stream, pGraph);
    graph_capture_finished();
    return result;
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec,
                                 cudaGraph_t graph,
                                 unsigned long long flags)
{
    cudaGraphInstantiate_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaGraphInstantiate");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGraphInstantiate: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(pGraphExec, graph, flags);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    cudaGraphLaunch_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (has_live_managed_allocations()) {
        fprintf(stderr,
                "[polaris-shim] cudaGraphLaunch is unsupported while "
                "Polaris-managed allocations are live\n");
        return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    }

    *(void **)(&real) = resolve_cudart_symbol("cudaGraphLaunch");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGraphLaunch: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(graphExec, stream);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec,
                                cudaGraph_t hGraph,
                                cudaGraphExecUpdateResultInfo *resultInfo)
{
    cudaGraphExecUpdate_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaGraphExecUpdate");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGraphExecUpdate: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(hGraphExec, hGraph, resultInfo);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes)
{
    cudaGraphGetNodes_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaGraphGetNodes");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGraphGetNodes: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(graph, nodes, numNodes);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType *pType)
{
    cudaGraphNodeGetType_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaGraphNodeGetType");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGraphNodeGetType: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(node, pType);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node,
                                         cudaKernelNodeParams *pNodeParams)
{
    cudaGraphKernelNodeGetParams_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaGraphKernelNodeGetParams");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGraphKernelNodeGetParams: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(node, pNodeParams);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node,
                                         const cudaKernelNodeParams *pNodeParams)
{
    cudaGraphKernelNodeSetParams_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaGraphKernelNodeSetParams");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGraphKernelNodeSetParams: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(node, pNodeParams);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGraphDestroy(cudaGraph_t graph)
{
    cudaGraphDestroy_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaGraphDestroy");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGraphDestroy: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(graph);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec)
{
    cudaGraphExecDestroy_fn real;

    bootstrap_for_runtime_call();

    *(void **)(&real) = resolve_cudart_symbol("cudaGraphExecDestroy");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaGraphExecDestroy: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(graphExec);
}

POLARIS_SHIM_INTERPOSER
CUresult cuStreamBeginCapture(CUstream hStream, cudaStreamCaptureMode mode)
{
    return cu_stream_begin_capture_impl("cuStreamBeginCapture", hStream, mode);
}

POLARIS_SHIM_INTERPOSER
CUresult cuStreamBeginCapture_v2(CUstream hStream, cudaStreamCaptureMode mode)
{
    return cu_stream_begin_capture_impl("cuStreamBeginCapture_v2", hStream, mode);
}

POLARIS_SHIM_INTERPOSER
CUresult cuStreamEndCapture(CUstream hStream, cudaGraph_t *phGraph)
{
    return cu_stream_end_capture_impl("cuStreamEndCapture", hStream, phGraph);
}

POLARIS_SHIM_INTERPOSER
CUresult cuStreamEndCapture_v2(CUstream hStream, cudaGraph_t *phGraph)
{
    return cu_stream_end_capture_impl("cuStreamEndCapture_v2", hStream, phGraph);
}

static cudaError_t cuda_memcpy_impl(const char *real_name,
                                    void *dst,
                                    const void *src,
                                    size_t count,
                                    int kind,
                                    cudaStream_t stream,
                                    int is_async)
{
    int dst_is_polaris;
    int src_is_polaris;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    dst_is_polaris = is_managed_pointer_value(dst);
    src_is_polaris = is_managed_pointer_value(src);
    if (dst_is_polaris || src_is_polaris) {
        (void)kind;
        (void)stream;
        (void)is_async;
        fprintf(stderr,
                "[polaris-shim] %s involving Polaris pointer is not wired yet "
                "(dst=%p src=%p bytes=%zu)\n",
                real_name,
                dst,
                src,
                count);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (is_async) {
        cudaMemcpyAsync_fn real_async;

        *(void **)(&real_async) = resolve_cudart_symbol(real_name);
        if (!real_async) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real_async(dst, src, count, kind, stream);
    } else {
        cudaMemcpy_fn real;

        *(void **)(&real) = resolve_cudart_symbol(real_name);
        if (!real) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real(dst, src, count, kind);
    }
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind)
{
    return cuda_memcpy_impl("cudaMemcpy", dst, src, count, kind, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMemcpyAsync(void *dst,
                            const void *src,
                            size_t count,
                            int kind,
                            cudaStream_t stream)
{
    return cuda_memcpy_impl("cudaMemcpyAsync", dst, src, count, kind, stream, 1);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMemcpy2DAsync(void *dst,
                              size_t dpitch,
                              const void *src,
                              size_t spitch,
                              size_t width,
                              size_t height,
                              int kind,
                              cudaStream_t stream)
{
    cudaMemcpy2DAsync_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (is_managed_pointer_value(dst) || is_managed_pointer_value(src)) {
        (void)dpitch;
        (void)spitch;
        (void)kind;
        (void)stream;
        fprintf(stderr,
                "[polaris-shim] cudaMemcpy2DAsync involving Polaris pointer is not wired yet "
                "(dst=%p src=%p width=%zu height=%zu)\n",
                dst,
                src,
                width,
                height);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    *(void **)(&real) = resolve_cudart_symbol("cudaMemcpy2DAsync");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaMemcpy2DAsync: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(dst, dpitch, src, spitch, width, height, kind, stream);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMemcpyPeerAsync(void *dst,
                                int dstDevice,
                                const void *src,
                                int srcDevice,
                                size_t count,
                                cudaStream_t stream)
{
    cudaMemcpyPeerAsync_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (is_managed_pointer_value(dst) || is_managed_pointer_value(src)) {
        (void)dstDevice;
        (void)srcDevice;
        (void)stream;
        fprintf(stderr,
                "[polaris-shim] cudaMemcpyPeerAsync involving Polaris pointer is not wired yet "
                "(dst=%p src=%p bytes=%zu)\n",
                dst,
                src,
                count);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    *(void **)(&real) = resolve_cudart_symbol("cudaMemcpyPeerAsync");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaMemcpyPeerAsync: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(dst, dstDevice, src, srcDevice, count, stream);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream)
{
    cudaMemcpy3DPeerAsync_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (p && (is_managed_pointer_value(p->dstPtr.ptr) ||
              is_managed_pointer_value(p->srcPtr.ptr))) {
        (void)stream;
        fprintf(stderr,
                "[polaris-shim] cudaMemcpy3DPeerAsync involving Polaris pointer is not wired yet "
                "(dst=%p src=%p)\n",
                p->dstPtr.ptr,
                p->srcPtr.ptr);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    *(void **)(&real) = resolve_cudart_symbol("cudaMemcpy3DPeerAsync");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaMemcpy3DPeerAsync: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(p, stream);
}

static cudaError_t cuda_memset_impl(const char *real_name,
                                    void *devPtr,
                                    int value,
                                    size_t count,
                                    cudaStream_t stream,
                                    int is_async)
{
    CUdeviceptr base = 0;
    uint64_t length = 0;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (find_allocation_range((CUdeviceptr)(uintptr_t)devPtr, &base, &length)) {
        if (g_allow_zero_memset &&
            value == 0 &&
            (CUdeviceptr)(uintptr_t)devPtr == base &&
            count <= length) {
            fprintf(stderr,
                    "[polaris-shim] %s zero-fill accepted for Polaris pointer "
                    "%p bytes=%zu\n",
                    real_name,
                    devPtr,
                    count);
            return CUDA_SUCCESS;
        }
        (void)stream;
        (void)is_async;
        fprintf(stderr,
                "[polaris-shim] %s involving Polaris pointer is not wired yet "
                "(ptr=%p value=%d bytes=%zu)\n",
                real_name,
                devPtr,
                value,
                count);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (is_async) {
        cudaMemsetAsync_fn real_async;

        *(void **)(&real_async) = resolve_cudart_symbol(real_name);
        if (!real_async) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real_async(devPtr, value, count, stream);
    } else {
        cudaMemset_fn real;

        *(void **)(&real) = resolve_cudart_symbol(real_name);
        if (!real) {
            fprintf(stderr, "[polaris-shim] %s: real symbol unavailable\n", real_name);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        return real(devPtr, value, count);
    }
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
    return cuda_memset_impl("cudaMemset", devPtr, value, count, NULL, 0);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    return cuda_memset_impl("cudaMemsetAsync", devPtr, value, count, stream, 1);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaMemAdvise(const void *devPtr,
                          size_t count,
                          int advice,
                          struct cudaMemLocation location)
{
    cudaMemAdvise_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (is_managed_pointer_value(devPtr)) {
        fprintf(stderr,
                "[polaris-shim] cudaMemAdvise ignored for Polaris pointer "
                "%p bytes=%zu advice=%d location=(%d,%d)\n",
                devPtr,
                count,
                advice,
                location.type,
                location.id);
        return CUDA_SUCCESS;
    }

    *(void **)(&real) = resolve_cudart_symbol("cudaMemAdvise");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaMemAdvise: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(devPtr, count, advice, location);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr)
{
    cudaIpcGetMemHandle_fn real;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    if (is_managed_pointer_value(devPtr)) {
        (void)handle;
        fprintf(stderr,
                "[polaris-shim] cudaIpcGetMemHandle is unsupported for Polaris pointer %p\n",
                devPtr);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    *(void **)(&real) = resolve_cudart_symbol("cudaIpcGetMemHandle");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaIpcGetMemHandle: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(handle, devPtr);
}

POLARIS_SHIM_INTERPOSER
cudaError_t cudaPointerGetAttributes(void *attributes, const void *ptr)
{
    cudaPointerGetAttributes_fn real;
    int ret;

    pthread_once(&g_announce_once, announce);
    pthread_once(&g_bootstrap_once, bootstrap_vaspace);

    ret = get_managed_runtime_pointer_attributes(
        (struct polaris_shim_cuda_pointer_attributes *)attributes,
        ptr);
    if (ret == 0)
        return CUDA_SUCCESS;
    if (ret == -EINVAL)
        return CUDA_ERROR_INVALID_VALUE;
    if (ret != -ENOENT)
        return CUDA_ERROR_UNKNOWN;

    *(void **)(&real) = resolve_cudart_symbol("cudaPointerGetAttributes");
    if (!real) {
        fprintf(stderr, "[polaris-shim] cudaPointerGetAttributes: real symbol unavailable\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    return real(attributes, ptr);
}
