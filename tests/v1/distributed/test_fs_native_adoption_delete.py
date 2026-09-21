# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for FS-native adoption + eviction delete accounting.

Startup adoption re-seeds byte accounting and the LRU eviction list from
object files left by a previous server lifetime. The demux thread's delete
completion path (``_OP_DELETE``) only fires ``_notify_keys_deleted`` — which
decrements the adapter's usage counters and drains the eviction policy's LRU
list — for keys registered in ``NativeConnectorL2Adapter._key_sizes``.

Before the fix, ``adopt_existing_keys`` called ``_notify_keys_stored``
directly without registering ``_key_sizes``, so evicting an adopted key
unlinked the file on disk but left the usage counter frozen and the LRU
un-drained. The eviction controller then re-selected the same already-deleted
keys every cycle (an ENOENT spin that never reaches the real files queued
behind them) and the backing disk filled until stores failed.
"""

# Standard
import importlib.util
import os

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction import L2EvictionPolicy
from lmcache.v1.distributed.eviction_policy import CreateEvictionPolicy
from lmcache.v1.distributed.l2_adapters.fs_key_codec import object_key_to_filename
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    _make_adapter_class,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
    _object_key_to_string,
)

# Reuse the pure-Python mock connector from the native-adapter test module
# (tests/ has no __init__.py, so load it by path).
_mock_path = os.path.join(
    os.path.dirname(__file__), "test_native_connector_l2_adapter.py"
)
_spec = importlib.util.spec_from_file_location("_nca_mock", _mock_path)
_nca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nca)
MockNativeConnector = _nca.MockNativeConnector


def _make_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="meta-llama/Llama-3",
        kv_rank=7,
        object_group_id=1,
    )


class _RecordingListener:
    """Minimal L2AdapterListener that records delete notifications."""

    def __init__(self):
        self.deleted: list[ObjectKey] = []

    def on_l2_keys_stored(self, keys, sizes):
        pass

    def on_l2_keys_accessed(self, keys):
        pass

    def on_l2_keys_deleted(self, keys):
        self.deleted.extend(keys)


def _make_fs_native_adapter(tmp_path, mock, **kwargs):
    adapter_cls = _make_adapter_class(NativeConnectorL2Adapter)
    return adapter_cls(
        mock,
        type_name="FSNativeL2Adapter",
        base_path=str(tmp_path),
        relative_tmp_dir="",
        adopt_existing=True,
        **kwargs,
    )


def _write_object_file(base_path, key: ObjectKey, payload: bytes) -> None:
    path = os.path.join(str(base_path), object_key_to_filename(key))
    with open(path, "wb") as f:
        f.write(payload)


def _write_sparse_object_file(base_path, key: ObjectKey, size: int) -> None:
    path = os.path.join(str(base_path), object_key_to_filename(key))
    with open(path, "wb") as f:
        f.truncate(size)


class TestAdoptionRegistersKeySizes:
    def test_adopted_keys_enter_key_sizes(self, tmp_path):
        keys = [_make_key(i) for i in range(3)]
        sizes = [100, 200, 300]
        for key, size in zip(keys, sizes):
            _write_object_file(tmp_path, key, b"x" * size)

        mock = MockNativeConnector()
        adapter = _make_fs_native_adapter(tmp_path, mock)
        try:
            assert adapter.adopt_existing_keys() == 3
            with adapter._lock:
                assert set(adapter._key_sizes) == set(keys)
                assert [adapter._key_sizes[k] for k in keys] == sizes
        finally:
            adapter.close()

    def test_usage_fraction_seeded_by_adoption(self, tmp_path):
        key = _make_key(1)
        # Sparse 1 GiB file against a 2 GiB capacity => fraction 0.5.
        # (Sizes must be powers-of-GiB so the float max_capacity_gb ->
        # int-bytes conversion in the adapter base is exact.)
        _write_sparse_object_file(tmp_path, key, 1024**3)

        mock = MockNativeConnector()
        adapter = _make_fs_native_adapter(tmp_path, mock, max_capacity_gb=2.0)
        try:
            adapter.adopt_existing_keys()
            usage = adapter.get_usage()
            assert usage.total_bytes_used == 1024**3
            assert usage.usage_fraction == pytest.approx(0.5)
        finally:
            adapter.close()


class TestAdoptedKeyDeleteAccounting:
    """The core regression: deleting an adopted key must decrement usage
    bytes and notify the eviction policy, exactly like deleting a key
    stored in-process."""

    def _adopt_and_delete(self, tmp_path):
        key = _make_key(42)
        _write_object_file(tmp_path, key, b"x" * 4096)

        mock = MockNativeConnector()
        # Simulate the file also being visible to the backend's key index
        # so the mock's delete reports success (mirrors the native FS
        # connector unlinking the real file).
        mock._store[_object_key_to_string(key)] = b"x" * 4096

        adapter = _make_fs_native_adapter(tmp_path, mock)
        listener = _RecordingListener()
        try:
            adapter.adopt_existing_keys()
            adapter.register_listener(listener)
            assert adapter.get_usage().total_bytes_used == 4096

            adapter.delete([key])

            assert adapter.get_usage().total_bytes_used == 0
            assert listener.deleted == [key]
            with adapter._lock:
                assert key not in adapter._key_sizes
        finally:
            adapter.close()

    def test_adopted_key_delete_decrements_and_notifies(self, tmp_path):
        self._adopt_and_delete(tmp_path)

    def test_lru_drains_after_adopted_key_delete(self, tmp_path):
        """End-to-end shape of the jam: adopted keys seeded oldest-first,
        delete the LRU-front key, and the policy must drop it so the next
        selection moves to the following keys instead of spinning."""

        keys = [_make_key(i) for i in range(4)]
        for key in keys:
            _write_object_file(tmp_path, key, b"x" * 100)
            # Stagger mtimes so adoption order is keys[0] .. keys[3].
            path = os.path.join(str(tmp_path), object_key_to_filename(key))
            os.utime(path, (100 + keys.index(key), 100 + keys.index(key)))

        mock = MockNativeConnector()
        for key in keys:
            mock._store[_object_key_to_string(key)] = b"x" * 100

        adapter = _make_fs_native_adapter(tmp_path, mock)
        try:
            config = EvictionConfig(
                eviction_policy="LRU", trigger_watermark=0.8, eviction_ratio=0.25
            )
            policy = CreateEvictionPolicy(config)
            # Mirror StorageManager: the eviction listener is registered
            # BEFORE adoption so adopted keys seed the LRU list.
            adapter.register_listener(L2EvictionPolicy(policy))

            adapter.adopt_existing_keys()

            # 25% of 4 keys = 1, taken from the LRU front. (Within one
            # adoption batch the policy inserts keys so the later ones are
            # evicted first, so assert on set membership rather than a
            # fixed index — the regression is about *draining*, not order.)
            assert policy.get_num_tracked_keys() == 4
            actions = policy.get_eviction_actions(0.25)
            assert len(actions) == 1
            victim = actions[0].keys[0]
            assert victim in keys

            adapter.delete([victim])

            # The deleted key must leave the policy's LRU list so the next
            # selection advances to a surviving key instead of re-picking
            # the already-deleted one (the ENOENT spin that jammed eviction).
            assert policy.get_num_tracked_keys() == 3
            actions = policy.get_eviction_actions(0.25)
            assert len(actions) == 1
            assert victim not in actions[0].keys
            assert actions[0].keys[0] in keys
        finally:
            adapter.close()
