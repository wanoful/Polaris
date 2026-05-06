# SPDX-License-Identifier: Apache-2.0
"""
POLARIS Scheduler — vLLM v1 Integration Adapter.

This is a thin wrapper around vLLM's ``Scheduler`` that swaps in
``PolarisKVCacheManager`` in place of the stock ``KVCacheManager``.

Because vLLM v1 exposes ``scheduler_cls`` as a first-class configuration
option (``--scheduler-cls``), users can enable POLARIS without modifying
a single line of vLLM source code::

    vllm serve meta-llama/Llama-2-7b-hf \\
        --scheduler-cls polaris_vllm.scheduler.PolarisScheduler

What changes
────────────
* ``__init__`` — after the parent ``Scheduler`` finishes initialisation we
  replace ``self.kv_cache_manager`` with a ``PolarisKVCacheManager`` instance.
* Everything else — forwarded to the parent class unchanged.

This means the scheduler loop, preemption logic, chunked-prefill heuristics,
beam-search scheduling, etc. all remain exactly as vLLM implements them.
Only the *block allocation backend* is redirected to POLARIS.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.kv_cache_manager import KVCacheBlocks

from polaris_vllm.polaris_kv_cache_manager import PolarisKVCacheManager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.interface import SchedulerInterface
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.multimodal import MultiModalRegistry

logger = logging.getLogger(__name__)


class PolarisScheduler(Scheduler):
    """
    vLLM Scheduler with POLARIS-backed KV cache management.

    See module docstring for usage instructions.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = None,  # type: ignore[assignment]
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        # Let vLLM perform its normal Scheduler initialisation.  This creates
        # the stock KVCacheManager, connector, event publisher, etc.
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            hash_block_size=hash_block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        # ── Swap in POLARIS KV cache manager ───────────────────────────────
        # The parent __init__ has already created self.kv_cache_manager.
        # We replace it with our adapter which keeps the same public API but
        # forwards block allocation / free to /dev/polaris.
        try:
            polaris_manager = PolarisKVCacheManager(
                kv_cache_config=kv_cache_config,
                max_model_len=self.max_model_len,
                hash_block_size=hash_block_size or block_size,
                max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
                enable_caching=self.cache_config.enable_prefix_caching,
                use_eagle=getattr(self, "use_eagle", False),
                log_stats=log_stats,
                enable_kv_cache_events=getattr(self, "enable_kv_cache_events", False),
                dcp_world_size=getattr(self, "dcp_world_size", 1),
                pcp_world_size=getattr(self, "pcp_world_size", 1),
                metrics_collector=self.kv_metrics_collector,
            )
            # Gracefully shut down the stock manager (best effort)
            old_manager = self.kv_cache_manager
            self.kv_cache_manager = polaris_manager
            del old_manager

            # Re-bind connector to the new block pool if necessary
            if (
                self.connector is not None
                and hasattr(self.connector, "bind_gpu_block_pool")
            ):
                self.connector.bind_gpu_block_pool(
                    self.kv_cache_manager.block_pool
                )

            logger.info("PolarisScheduler initialised — KV cache managed by POLARIS")
        except Exception as exc:
            logger.error(
                "Failed to instantiate PolarisKVCacheManager, keeping stock manager: %s",
                exc,
            )
