#ifndef POLARIS_RUNTIME_H
#define POLARIS_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct polaris_runtime polaris_runtime_t;

typedef struct polaris_runtime_config {
    uint32_t gpu_id;
    int32_t device_ordinal;
    uint64_t total_bytes;
    uint64_t budget_bytes;
    uint64_t cpu_pool_bytes;
    uint64_t va_reserve_bytes;
    uint64_t block_size;
    uint32_t flags;
} polaris_runtime_config_t;

typedef struct polaris_runtime_info {
    uint64_t va_base;
    uint64_t va_size;
    uint64_t granule;
    uint64_t cpu_pool_base;
    uint64_t cpu_pool_bytes;
} polaris_runtime_info_t;

int polaris_runtime_create(const polaris_runtime_config_t * config,
                           polaris_runtime_t ** out_runtime,
                           polaris_runtime_info_t * out_info);

int polaris_runtime_start(polaris_runtime_t * runtime);
int polaris_runtime_poll_once(polaris_runtime_t * runtime, int timeout_ms);
void polaris_runtime_stop(polaris_runtime_t * runtime);
void polaris_runtime_destroy(polaris_runtime_t * runtime);

const char * polaris_runtime_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* POLARIS_RUNTIME_H */
