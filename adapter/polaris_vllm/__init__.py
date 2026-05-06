# POLARIS vLLM Integration Adapter
# 
# Phase 4e: Optional vLLM Integration Adapter
# 
# This package provides a drop-in replacement for vLLM v1's KVCacheManager
# that delegates KV cache block management to the POLARIS kernel module.
#
# Usage:
#   vllm serve meta-llama/Llama-2-7b-hf \
#     --scheduler-cls polaris_vllm.scheduler.PolarisScheduler
#
# Architecture:
#   PolarisScheduler -> PolarisKVCacheManager -> POLARIS (/dev/polaris)
#
# The adapter maintains vLLM's prefix caching, sliding window, and beam search
# semantics while using POLARIS for actual GPU memory allocation, page-fault
# handling, CPU offload, and COW sharing.

__version__ = "0.1.0"

# Lazy imports so that the ABI layer can be used even when vLLM is not
# installed (e.g. in polarisctl or test harnesses).
try:
    from polaris_vllm.polaris_scheduler import PolarisScheduler
    from polaris_vllm.polaris_kv_cache_manager import PolarisKVCacheManager
    __all__ = ["PolarisScheduler", "PolarisKVCacheManager"]
except ImportError:
    # vLLM not available — only the ABI layer is usable.
    __all__ = []
