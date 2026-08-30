# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for FS-native startup adoption (``scan_adoptable_files``).

Adoption re-seeds byte accounting and eviction from object files a previous
server lifetime left on disk. The scan must accept exactly the completed
object files (atomically renamed to their final ``.data`` names), skip temp
and foreign files, and report results in mtime order (oldest first) so the
caller seeds LRU age correctly.
"""

# Standard
import os

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_key_codec import (
    object_key_to_filename,
)
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    scan_adoptable_files,
)


def _make_key(chunk_hash: bytes, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name="meta-llama/Llama-3",
        kv_rank=42,
        object_group_id=1,
        cache_salt=cache_salt,
    )


def _write(path, payload: bytes, mtime: float) -> None:
    with open(path, "wb") as f:
        f.write(payload)
    os.utime(path, (mtime, mtime))


class TestScanAdoptableFiles:
    def test_adopts_only_complete_objects_in_mtime_order(self, tmp_path):
        old_key = _make_key(b"\x01" * 4)
        new_key = _make_key(b"\x02" * 4, cache_salt="alice")
        _write(tmp_path / object_key_to_filename(old_key), b"x" * 10, mtime=100.0)
        _write(tmp_path / object_key_to_filename(new_key), b"y" * 20, mtime=200.0)

        # Skipped: foreign file, in-progress temp file, temp sub-dir contents.
        _write(tmp_path / "notes.txt", b"junk", mtime=50.0)
        _write(tmp_path / "partial.data.tmp", b"junk", mtime=50.0)
        tmp_dir = tmp_path / "tmp"
        tmp_dir.mkdir()
        _write(tmp_dir / object_key_to_filename(old_key), b"junk", mtime=50.0)

        found = scan_adoptable_files(str(tmp_path), "tmp")

        assert [(key, size) for key, size, _mtime in found] == [
            (old_key, 10),
            (new_key, 20),
        ]
        mtimes = [mtime for _key, _size, mtime in found]
        assert mtimes == sorted(mtimes)

    def test_missing_directory_adopts_nothing(self, tmp_path):
        assert scan_adoptable_files(str(tmp_path / "absent"), "") == []

    def test_undecodable_data_file_is_skipped(self, tmp_path):
        _write(tmp_path / "just-one-field.data", b"junk", mtime=10.0)
        assert scan_adoptable_files(str(tmp_path), "") == []
