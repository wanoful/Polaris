# SPDX-License-Identifier: Apache-2.0
"""
POLARIS KV Cache Manager — vLLM v1 Integration Adapter.

This class replaces vLLM's KVCacheManager, delegating actual GPU memory
management to the POLARIS kernel module while preserving vLLM's high-level
scheduling semantics (prefix caching, sliding window, beam search).

Architecture
────────────
vLLM Scheduler
      │
      ▼
PolarisKVCacheManager  ←  drop-in replacement for KVCacheManager
      │
      ├─ allocate_slots()  →  POLARIS_BLOCK_GROW ioctls
      ├─ free()            →  POLARIS_SESSION_DESTROY
      └─ (prefix caching, COW, etc. handled by POLARIS kernel)
      │
      ▼
  /dev/polaris  →  polaris.ko  →  polarisd  →  CUDA VMM

Usage
─────
Instantiated automatically by PolarisScheduler.  Do not construct directly.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_manager import KVCacheManager, KVCacheBlocks

from polaris_vllm.polaris_abi import (
    polaris_session_create,
    polaris_session_destroy,
    polaris_block_grow,
    polaris_block_touch,
    POLARIS_PHASE_PREFILL,
    POLARIS_PHASE_DECODE,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)

# Sentinel value for a block that has not yet been assigned a POLARIS block_id.
_UNMAPPED = -1


class PolarisKVCacheManager(KVCacheManager):
    """
    Drop-in replacement for vLLM v1 ``KVCacheManager``.

    The manager intercepts block allocation / free events and translates them
    into POLARIS ioctls.  vLLM's own ``BlockPool`` is kept as a lightweight
    bookkeeping layer (it only stores ``KVCacheBlock`` metadata objects); the
    actual GPU physical memory is allocated by POLARIS via ``cuMemMap``.

    For Phase-4e smoke-test scope we implement the happy path:
    * ``allocate_slots``  → ``POLARIS_BLOCK_GROW``
    * ``free``            → ``POLARIS_SESSION_DESTROY``
    * Prefix-caching and sliding-window logic is forwarded to the parent
      ``KVCacheManager`` unchanged.

    Attributes
    ----------
    _polaris_fd : int
        File descriptor for ``/dev/polaris`` (``-1`` if unavailable).
    _session_ids : dict[str, int]
        Mapping from vLLM ``request.request_id`` to POLARIS ``session_id``.
    _block_size : int
        Tokens per block (taken from the first KV cache group spec).
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        max_num_batched_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector=None,
    ) -> None:
        # Initialise the underlying vLLM manager first.  This creates the
        # BlockPool, coordinator, prefix-cache hashes, etc.  We keep it around
        # so that vLLM's scheduler, model runner and connectors continue to
        # work unmodified.
        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            hash_block_size=hash_block_size,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_caching=enable_caching,
            use_eagle=use_eagle,
            log_stats=log_stats,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            metrics_collector=metrics_collector,
        )

        # ── POLARIS state ──────────────────────────────────────────────────
        self._polaris_fd = _open_polaris_device()
        self._session_ids: dict[str, int] = {}
        self._block_size = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size

        # Track the number of blocks we have already reported to POLARIS for
        # each request so that we only emit BLOCK_GROW for *new* blocks.
        self._reported_blocks: dict[str, int] = {}

        if self._polaris_fd < 0:
            logger.warning(
                "/dev/polaris is not available — PolarisKVCacheManager will "
                "fall back to pure vLLM behaviour."
            )

    # ── Public API overrides ───────────────────────────────────────────────

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
    ) -> KVCacheBlocks | None:
        """
        Allocate slots for *request* and synchronise the allocation with POLARIS.

        The implementation is a three-step dance:
        1. Let vLLM do its normal allocation (prefix cache, sliding window,
           admission check, etc.).  This returns ``KVCacheBlocks`` or ``None``.
        2. If vLLM succeeded and ``/dev/polaris`` is open, ensure a POLARIS
           session exists for this request.
        3. Emit ``POLARIS_BLOCK_GROW`` for every *new* block that vLLM just
           allocated (delta between previously-reported count and current count).
        """
        # Step 1 — vLLM native allocation
        result = super().allocate_slots(
            request=request,
            num_new_tokens=num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
        )

        if result is None or self._polaris_fd < 0:
            return result

        rid = request.request_id

        # Step 2 — ensure POLARIS session exists
        if rid not in self._session_ids:
            session_id = self._create_polaris_session(request)
            self._session_ids[rid] = session_id
            self._reported_blocks[rid] = 0
            logger.debug("Created POLARIS session %d for request %s", session_id, rid)

        session_id = self._session_ids[rid]

        # Step 3 — compute delta and grow POLARIS blocks
        total_blocks_now = sum(len(g) for g in result.blocks)
        previously_reported = self._reported_blocks.get(rid, 0)
        delta = total_blocks_now - previously_reported

        if delta > 0:
            phase = (
                POLARIS_PHASE_PREFILL
                if request.num_computed_tokens < request.num_tokens
                else POLARIS_PHASE_DECODE
            )
            for i in range(delta):
                token_start = (previously_reported + i) * self._block_size
                try:
                    block_id = polaris_block_grow(
                        self._polaris_fd,
                        session_id,
                        token_start=token_start,
                        token_count=self._block_size,
                        phase=phase,
                    )
                    logger.debug(
                        "POLARIS_BLOCK_GROW session=%d token_start=%d "
                        "token_count=%d → block_id=%d",
                        session_id,
                        token_start,
                        self._block_size,
                        block_id,
                    )
                except OSError as exc:
                    # POLARIS allocation failed.  We intentionally do *not*
                    # roll back the vLLM allocation here — the scheduler will
                    # see the failure on the next step and preempt the request.
                    logger.error(
                        "POLARIS_BLOCK_GROW failed for request %s: %s", rid, exc
                    )
                    raise

            self._reported_blocks[rid] = total_blocks_now

        # Touch all existing blocks to keep LRU fresh
        if total_blocks_now > 0:
            try:
                polaris_block_touch(
                    self._polaris_fd,
                    session_id,
                    token_start=0,
                    token_count=total_blocks_now * self._block_size,
                )
            except OSError:
                logger.debug("POLARIS_BLOCK_TOUCH failed for request %s", rid)

        return result

    def free(self, request: Request) -> None:
        """Free the request's blocks in vLLM *and* destroy its POLARIS session."""
        rid = request.request_id

        # 1. Native vLLM free (returns blocks to BlockPool)
        super().free(request)

        # 2. Destroy POLARIS session (kernel will reclaim all blocks)
        if rid in self._session_ids and self._polaris_fd >= 0:
            session_id = self._session_ids.pop(rid)
            self._reported_blocks.pop(rid, None)
            try:
                polaris_session_destroy(self._polaris_fd, session_id)
                logger.debug("Destroyed POLARIS session %d for request %s", session_id, rid)
            except OSError as exc:
                logger.error(
                    "POLARIS_SESSION_DESTROY failed for session %d: %s", session_id, exc
                )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _create_polaris_session(self, request: Request) -> int:
        """Create a POLARIS session for *request*.

        The ``gpu_vas_bytes`` reservation is sized to the maximum sequence
        length the request could ever need.
        """
        max_tokens = min(request.num_tokens + request.max_tokens, self.max_model_len)
        # Reserve enough VA space for the full sequence
        gpu_vas_bytes = max_tokens * POLARIS_DEFAULT_BYTES_PER_TOKEN

        return polaris_session_create(
            self._polaris_fd,
            home_gpu=0,  # TODO: multi-GPU support
            beam_width=1,
            gpu_vas_bytes=gpu_vas_bytes,
            bytes_per_token=POLARIS_DEFAULT_BYTES_PER_TOKEN,
        )

    def __del__(self):
        """Best-effort cleanup of the POLARIS device fd."""
        if hasattr(self, "_polaris_fd") and self._polaris_fd >= 0:
            try:
                os.close(self._polaris_fd)
            except OSError:
                pass


# ─── Helpers ────────────────────────────────────────────────────────────────

POLARIS_DEFAULT_BYTES_PER_TOKEN = 524_288  # ~512 KiB for Llama-2-7B FP16


def _open_polaris_device() -> int:
    """Open ``/dev/polaris`` and return the fd, or ``-1`` on failure."""
    try:
        return os.open("/dev/polaris", os.O_RDWR)
    except (OSError, FileNotFoundError):
        return -1
