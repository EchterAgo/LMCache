# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the lazy-offload path on a hybrid (Mamba/attention)
KV-cache layout.

Covers the three blockers that ``lmcache.mp.lazy_offload`` exposed on the
Qwen3.8-Flash-Next layouts:

- MTP (group token spans ``[1600, 1600, 1600, 1600, 0, 1600]``): a zero-span
  spec-decode scratch group, and align-mode mamba block tables padded with
  vLLM's null placeholder block.
- DFlash2 (``[3200, 3200, 3200, 3200, 0, 3200, 64]``): the drafter's
  sliding-window group, whose real out-of-window blocks never carry a
  prefix-cache hash.

- ``2b5ec51d``: ``_drain`` divided the scheduled-token count by every group
  span, so the zero-span scratch group raised ``ZeroDivisionError`` on the
  first drain and killed the EngineCore.
- ``0bd3219c``: ``add_store_candidate`` snapshotted a hash for every covered
  block including the null placeholder (which never carries one), so the
  eviction-aware policy rejected every store op as ``rejected_unhashed`` and
  the LMCache prefix cache never populated on a hybrid model.
- (this change): the same rejection recurred with DFlash2 because a real
  sliding-window group's out-of-window blocks keep ``block_hash is None``
  forever (vLLM only hashes what can still serve a hit). The snapshot now
  skips unhashed blocks in groups declared sliding-window.

A real ``vllm.v1.core.block_pool.BlockPool`` is used so ``is_null``, the free
queue, and ``touch``/``free_blocks`` behave exactly as in production.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_manager import LazyOffloadManager
from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata
from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp
from lmcache.v1.multiprocess.token_codec import pack_token_ids

vllm_block_pool = pytest.importorskip(
    "vllm.v1.core.block_pool", reason="lazy offload needs the vLLM block pool"
)
BlockPool = vllm_block_pool.BlockPool


class _FakeCompletionTracker:
    """Single-worker completion tracker: every batch settles on one receipt."""

    def __init__(self) -> None:
        self._pending: dict[str, int] = {}

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        self._pending[request_id] = self._pending.get(request_id, 0) + count
        return True


def _make_pool(num_blocks: int = 8) -> BlockPool:
    return BlockPool(num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=16)


def _hash_block(pool: BlockPool, block_id: int, tag: bytes) -> None:
    """Give a real block a prefix-cache hash, as ``cache_full_blocks`` would."""
    pool.blocks[block_id].set_block_hash(tag, num_tokens=16)


def _store_meta(
    request_id: str,
    block_ids_per_group: list[list[int]],
    token_ids: list[int],
    start: int,
    end: int,
) -> LMCacheMPRequestMetadata:
    op = LoadStoreOp(
        token_bytes=pack_token_ids(token_ids),
        block_ids=[list(group) for group in block_ids_per_group],
        start=start,
        end=end,
    )
    return LMCacheMPRequestMetadata(request_id=request_id, direction="STORE", op=op)


def _scheduler_output(
    total_num_scheduled_tokens: int,
    new_block_ids: list[list[int]] | None = None,
) -> SimpleNamespace:
    """Minimal stand-in for the attributes ``_drain``/``_new_blocks`` read."""
    cached = SimpleNamespace(
        req_ids=[],
        new_block_ids=new_block_ids or [],
        resumed_req_ids=set(),
    )
    return SimpleNamespace(
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
    )


# ---------------------------------------------------------------------------
# Fix 2b5ec51d: zero-span scratch group must not crash the drain estimate
# ---------------------------------------------------------------------------


def test_drain_skips_zero_span_scratch_group():
    """A Qwen3.8-style layout with a 0-token scratch group must not raise.

    Before the fix, ``est_next_step_blocks`` divided by every group span and
    the scratch group's 0 raised ``ZeroDivisionError`` on the first drain.
    """
    manager = LazyOffloadManager(
        configs={},  # default EVICTION_AWARE policy
        group_tokens_per_block=[1600, 1600, 1600, 1600, 0, 1600],
        completion_tracker=_FakeCompletionTracker(),
    )
    manager.bind_block_pool(_make_pool())

    # 4800 tokens scheduled; must not raise.
    manager.on_scheduler_step(_scheduler_output(4800))


def test_drain_estimate_counts_only_positive_spans():
    """The next-step estimate sums ceil(tokens/span) over cacheable groups.

    With spans ``[1600, 1600, 0, 1600]`` and 4800 tokens the scratch group
    contributes 0, so the estimate is ``3 * ceil(4800/1600) = 9``.
    """
    manager = LazyOffloadManager(
        configs={},
        group_tokens_per_block=[1600, 1600, 0, 1600],
        completion_tracker=_FakeCompletionTracker(),
    )
    manager.bind_block_pool(_make_pool())
    manager.on_scheduler_step(_scheduler_output(4800))

    assert manager._policy._next_step_estimate == 9


# ---------------------------------------------------------------------------
# Fix 0bd3219c: null placeholder blocks excluded from the hash snapshot
# ---------------------------------------------------------------------------


def test_null_block_excluded_from_snapshot_and_admitted():
    """A store op covering the null block is admitted, not rejected.

    The align-mode block table fills retired positions with the null block
    (id 0, ``is_null``, no hash). The snapshot must skip it so the policy does
    not see a ``None`` and reject the whole op as unhashed.
    """
    pool = _make_pool()
    _hash_block(pool, 1, b"h1")
    _hash_block(pool, 2, b"h2")
    assert pool.blocks[0].is_null and pool.blocks[0].block_hash is None

    manager = LazyOffloadManager(
        configs={},
        group_tokens_per_block=[1600],
        completion_tracker=_FakeCompletionTracker(),
    )
    manager.bind_block_pool(pool)

    # Null block sits between two hashed blocks, as a retired interior state.
    meta = _store_meta("a", [[1, 0, 2]], token_ids=list(range(48)), start=0, end=48)
    manager.add_store_candidate(meta)

    counters = manager._policy._counters
    assert counters.rejected_unhashed == 0
    assert manager._policy.has_pending_request("a")

    # The buffered snapshot holds the real blocks only; the null is excluded.
    snapshot = manager._policy._pending["a"][0].block_hashes
    assert set(snapshot) == {1, 2}
    assert 0 not in snapshot


def test_real_unhashed_block_still_rejected():
    """A ``None`` hash on a *real* block is the recycled case: still rejected.

    Excluding nulls must not weaken the guard for a real block whose hash was
    cleared (eviction-undetectable), which still breaks the prefix chain.
    """
    pool = _make_pool()
    _hash_block(pool, 1, b"h1")
    # Block 3 is real (not null) but carries no hash.
    assert not pool.blocks[3].is_null and pool.blocks[3].block_hash is None

    manager = LazyOffloadManager(
        configs={},
        group_tokens_per_block=[1600],
        completion_tracker=_FakeCompletionTracker(),
    )
    manager.bind_block_pool(pool)

    meta = _store_meta("a", [[1, 3]], token_ids=list(range(32)), start=0, end=32)
    manager.add_store_candidate(meta)

    assert manager._policy._counters.rejected_unhashed == 1
    assert not manager._policy.has_pending_request("a")


def test_null_block_not_pinned_on_drain():
    """The drain pins only the hashed blocks, never the shared null block.

    ``gpu_block_ids`` is taken from the snapshot, so a null block excluded
    there is neither counted as unhashed nor ``touch``ed. The null block is
    globally shared and must not accrue per-request references.
    """
    pool = _make_pool()
    _hash_block(pool, 1, b"h1")
    _hash_block(pool, 2, b"h2")

    # FIFO drains a finished request whole, independent of free-queue pressure.
    manager = LazyOffloadManager(
        configs={
            "lmcache.mp.lazy_offload_policy": "FIFO",
            "lmcache.mp.lazy_offload_threshold": 1,
        },
        group_tokens_per_block=[1600],
        completion_tracker=_FakeCompletionTracker(),
    )
    manager.bind_block_pool(pool)

    meta = _store_meta("a", [[1, 0, 2]], token_ids=list(range(48)), start=0, end=48)
    manager.add_store_candidate(meta)
    manager.on_request_finished("a")
    actions = manager.on_scheduler_step(_scheduler_output(16))

    assert len(actions.stores_to_submit) == 1
    assert pool.blocks[1].ref_cnt == 1
    assert pool.blocks[2].ref_cnt == 1
    # The null block was covered by the op but must not be pinned.
    assert pool.blocks[0].ref_cnt == 0


# ---------------------------------------------------------------------------
# DFlash2: unhashed real blocks of a sliding-window group must not reject
# ---------------------------------------------------------------------------


def test_sliding_window_unhashed_block_admitted():
    """A store op covering a sliding-window group's unhashed blocks is admitted.

    vLLM's sliding-window manager only hashes blocks that can still serve a
    hit, so retired positions keep ``block_hash is None`` forever. With the
    group declared sliding-window, the snapshot skips them instead of
    rejecting the op and breaking the request's whole prefix chain.
    """
    pool = _make_pool(num_blocks=64)
    _hash_block(pool, 1, b"h1")
    # Group 1 (sliding window): blocks 4 and 5 are real, out-of-window, no hash.
    assert not pool.blocks[4].is_null and pool.blocks[4].block_hash is None

    manager = LazyOffloadManager(
        configs={},
        group_tokens_per_block=[1600, 64],
        completion_tracker=_FakeCompletionTracker(),
        group_sliding_window=[False, True],
    )
    manager.bind_block_pool(pool)

    meta = _store_meta(
        "a", [[1], [4, 5]], token_ids=list(range(1664)), start=0, end=1664
    )
    manager.add_store_candidate(meta)

    counters = manager._policy._counters
    assert counters.rejected_unhashed == 0
    assert manager._policy.has_pending_request("a")

    # The buffered snapshot holds the hashed block; the SW group's unhashed
    # blocks are excluded.
    snapshot = manager._policy._pending["a"][0].block_hashes
    assert set(snapshot) == {1}


def test_sliding_window_hashed_block_still_tracked():
    """A sliding-window group's in-window blocks keep their hash and snapshot.

    The skip is per block (only ``None`` hashes are dropped), not per group:
    a hashed block in a sliding-window group is still watched for eviction
    exactly like any other cached block.
    """
    pool = _make_pool(num_blocks=64)
    _hash_block(pool, 1, b"h1")
    _hash_block(pool, 4, b"h4")  # in-window block of the sliding group

    manager = LazyOffloadManager(
        configs={},
        group_tokens_per_block=[1600, 64],
        completion_tracker=_FakeCompletionTracker(),
        group_sliding_window=[False, True],
    )
    manager.bind_block_pool(pool)

    meta = _store_meta(
        "a", [[1], [4, 5]], token_ids=list(range(1664)), start=0, end=1664
    )
    manager.add_store_candidate(meta)

    snapshot = manager._policy._pending["a"][0].block_hashes
    # h4 hashed in-window block tracked; block 5 unhashed out-of-window skipped.
    assert set(snapshot) == {1, 4}


def test_unhashed_block_in_normal_group_still_rejected():
    """Without a sliding-window declaration, the reject rule is unchanged.

    A ``None`` hash on a real block of a group that is *not* sliding-window
    remains the recycled-block case the policy must reject: the group flag
    only relaxes the groups vLLM provably never hashes.
    """
    pool = _make_pool(num_blocks=64)
    _hash_block(pool, 1, b"h1")

    manager = LazyOffloadManager(
        configs={},
        group_tokens_per_block=[1600, 64],
        completion_tracker=_FakeCompletionTracker(),
        group_sliding_window=[False, False],
    )
    manager.bind_block_pool(pool)

    meta = _store_meta(
        "a", [[1], [4, 5]], token_ids=list(range(1664)), start=0, end=1664
    )
    manager.add_store_candidate(meta)

    assert manager._policy._counters.rejected_unhashed == 1
    assert not manager._policy.has_pending_request("a")
