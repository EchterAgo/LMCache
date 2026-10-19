# SPDX-License-Identifier: Apache-2.0
"""Tests for synchronous L1 evict-to-fit on reserve_write OOM.

Without evict-to-fit, a full L1 pool returns OUT_OF_MEMORY and the caller
(e.g. an L2 prefetch) drops the batch even though the LRU eviction loop
could free the memory instantly. With it, ``reserve_write`` reclaims LRU
objects under the manager lock and retries the allocation in the same call.
"""

# Standard
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

ALIGN = 4096
# 1 MiB objects; pool holds exactly 8.
OBJ_BYTES = 256 * ALIGN
CAPACITY = 8 * OBJ_BYTES
LAYOUT = MemoryLayoutDesc([torch.Size([OBJ_BYTES])], [torch.uint8])


def key(index: int) -> ObjectKey:
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "evict-to-fit", 0)


@pytest.fixture(autouse=True)
def fake_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)


@pytest.fixture
def manager() -> "L1Manager":
    mgr = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=CAPACITY, use_lazy=False, align_bytes=ALIGN, shm_name=""
            ),
        )
    )
    yield mgr
    mgr.close()


def _admit(mgr: L1Manager, index: int) -> None:
    """Reserve and admit one object so it becomes resident (evictable)."""
    k = key(index)
    error, obj = mgr.reserve_write([k], [False], LAYOUT, tag="store")[k]
    assert error is L1Error.SUCCESS and obj is not None
    assert mgr.finish_write([k], tag="store")[k] is L1Error.SUCCESS


def _fill(mgr: L1Manager, count: int) -> None:
    for i in range(count):
        _admit(mgr, i)


class TestEvictToFit:
    def test_full_pool_without_evictor_returns_oom(self, manager):
        """Baseline: no evictor registered -> OOM on a full pool."""
        _fill(manager, CAPACITY // OBJ_BYTES)
        error, obj = manager.reserve_write([key(99)], [False], LAYOUT, tag="p")[key(99)]
        assert error is L1Error.OUT_OF_MEMORY and obj is None

    def test_full_pool_with_evictor_succeeds(self, manager):
        """Evict-to-fit frees the LRU object and the write succeeds."""
        # Constructing the controller registers the sync evictor; the
        # background watermark thread is left unstarted here.
        L1EvictionController(manager, EvictionConfig(eviction_policy="LRU"))
        _fill(manager, CAPACITY // OBJ_BYTES)
        error, obj = manager.reserve_write([key(99)], [False], LAYOUT, tag="p")[
            key(99)
        ]
        assert error is L1Error.SUCCESS and obj is not None
        # The LRU object (key 0) was evicted to make room.
        assert manager.reserve_read([key(0)])[key(0)][0] is L1Error.KEY_NOT_EXIST
        # The newest objects survived.
        assert manager.reserve_read([key(7)])[key(7)][0] is L1Error.SUCCESS

    def test_evicts_only_what_is_needed(self, manager):
        """A 1-object request must not sweep the whole LRU list."""
        L1EvictionController(manager, EvictionConfig(eviction_policy="LRU"))
        _fill(manager, CAPACITY // OBJ_BYTES)
        error, _ = manager.reserve_write([key(99)], [False], LAYOUT, tag="p")[key(99)]
        assert error is L1Error.SUCCESS
        # Exactly one resident object was sacrificed: keys 1..7 remain.
        survivors = sum(
            1
            for i in range(1, 8)
            if manager.reserve_read([key(i)])[key(i)][0] is L1Error.SUCCESS
        )
        assert survivors == 7

    def test_all_locked_returns_oom(self, manager):
        """When nothing is evictable (all read-locked), OOM is still reported."""
        L1EvictionController(manager, EvictionConfig(eviction_policy="LRU"))
        _fill(manager, CAPACITY // OBJ_BYTES)
        # Read-lock every resident object: none may be evicted.
        manager.reserve_read([key(i) for i in range(8)])
        error, obj = manager.reserve_write([key(99)], [False], LAYOUT, tag="p")[
            key(99)
        ]
        assert error is L1Error.OUT_OF_MEMORY and obj is None

    def test_batch_larger_than_capacity_fails(self, manager):
        """A batch exceeding the whole pool cannot be satisfied by eviction."""
        L1EvictionController(manager, EvictionConfig(eviction_policy="LRU"))
        _fill(manager, CAPACITY // OBJ_BYTES)
        keys = [key(200 + i) for i in range(CAPACITY // OBJ_BYTES + 1)]
        results = manager.reserve_write(keys, [False] * len(keys), LAYOUT, tag="p")
        assert all(e is L1Error.OUT_OF_MEMORY for e, _ in results.values())

    def test_fits_without_eviction_below_watermark(self, manager):
        """When the batch fits, evict-to-fit is not invoked and nothing is
        evicted even though the controller is registered."""
        L1EvictionController(
            manager, EvictionConfig(eviction_policy="LRU", trigger_watermark=0.99)
        )
        _fill(manager, 4)  # 50% usage
        error, _ = manager.reserve_write([key(99)], [False], LAYOUT, tag="p")[key(99)]
        assert error is L1Error.SUCCESS  # fits without eviction
        assert manager.reserve_read([key(0)])[key(0)][0] is L1Error.SUCCESS

    def test_concurrent_reserves_do_not_double_evict(self, manager):
        """Serialized reserve_write: N concurrent 1-object writes on a full
        pool evict exactly N LRU objects, not N * sweep."""
        L1EvictionController(manager, EvictionConfig(eviction_policy="LRU"))
        _fill(manager, CAPACITY // OBJ_BYTES)
        n_threads = 4
        results: dict[int, L1Error] = {}
        lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def worker(idx: int) -> None:
            barrier.wait()
            error, _ = manager.reserve_write(
                [key(300 + idx)], [False], LAYOUT, tag=f"p{idx}"
            )[key(300 + idx)]
            with lock:
                results[idx] = error

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(e is L1Error.SUCCESS for e in results.values())
        # Exactly n_threads LRU victims (keys 0..n_threads-1) are gone.
        for i in range(n_threads):
            assert manager.reserve_read([key(i)])[key(i)][0] is L1Error.KEY_NOT_EXIST
        for i in range(n_threads, CAPACITY // OBJ_BYTES):
            assert manager.reserve_read([key(i)])[key(i)][0] is L1Error.SUCCESS
