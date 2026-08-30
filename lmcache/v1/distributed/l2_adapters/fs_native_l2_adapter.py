# SPDX-License-Identifier: Apache-2.0
"""
Filesystem native L2 adapter config and factory.

Backed by the native C++ filesystem connector wrapped with
``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Optional
import os

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.fs_key_codec import (
    filename_to_object_key,
)

logger = init_logger(__name__)

# Enough reader threads for the default 1536 MiB budget to bind at 6 MiB
# objects (chunk_size 256), and for small objects to reach an array's
# ceiling, which takes a few hundred reads in flight.  Idle readers hold no
# file descriptor.
DEFAULT_NUM_READ_WORKERS = 256


def _validate_non_negative_int(name: str, raw: object) -> int:
    """Return ``raw`` if it is a non-negative ``int``.

    ``bool`` is rejected although it subclasses ``int``: ``true`` in a
    config would otherwise mean one.

    Raises:
        ValueError: If ``raw`` is a bool, not an int, or negative.
    """
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return raw


class FSNativeL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native C++
    filesystem connector.

    Fields:
    - base_path: directory for storing KV cache files.
    - num_workers: C++ worker threads for I/O (default 4).
    - relative_tmp_dir: relative sub-dir for temp files.
    - use_odirect: bypass page cache via O_DIRECT.
    - num_read_workers: threads dedicated to executing reads, and so the
      maximum reads in flight (default 256).  0 keeps the legacy path,
      where reads run on the worker threads and the depth against the
      device therefore equals num_workers.
    - read_max_bytes_in_flight: bytes the connector may keep outstanding
      against the device.  0 selects 1536 MiB when num_read_workers is
      positive; ignored otherwise.
      See docs/source/mp/l2_storage/fs_native.rst for how to size both.
    - read_ahead_size: trigger filesystem readahead by
      reading this many bytes first (optional).
    - max_capacity_gb: declared L2 capacity in GB, used for usage
      accounting. ``0`` (default) disables it. A value ``> 0`` does
      **not** bound disk usage on its own -- eviction runs only when
      the adapter spec also carries an ``eviction`` block, e.g.
      ``{"eviction": {"eviction_policy": "LRU"}}``. Without one,
      files accumulate past the declared capacity.
    - adopt_existing: on startup, scan ``base_path`` and seed byte
      accounting / eviction with files persisted by a previous run
      (default true).
    """

    def __init__(
        self,
        base_path: str,
        num_workers: int = 4,
        relative_tmp_dir: str = "",
        use_odirect: bool = False,
        read_ahead_size: Optional[int] = None,
        max_capacity_gb: float = 0,
        num_read_workers: int = DEFAULT_NUM_READ_WORKERS,
        read_max_bytes_in_flight: int = 0,
        adopt_existing: bool = True,
    ):
        """Build the config; see the class docstring for each field.

        Raises:
            ValueError: If ``num_read_workers`` or
                ``read_max_bytes_in_flight`` is not a non-negative integer.
        """
        self.base_path = base_path
        self.num_workers = num_workers
        self.relative_tmp_dir = relative_tmp_dir
        self.use_odirect = use_odirect
        self.read_ahead_size = read_ahead_size
        self.max_capacity_gb = max_capacity_gb
        self.num_read_workers = _validate_non_negative_int(
            "num_read_workers", num_read_workers
        )
        self.read_max_bytes_in_flight = _validate_non_negative_int(
            "read_max_bytes_in_flight", read_max_bytes_in_flight
        )
        if num_read_workers == 0 and read_max_bytes_in_flight > 0:
            logger.warning(
                "fs_native: read_max_bytes_in_flight=%d is ignored because "
                "num_read_workers=0 selects the legacy read path.",
                read_max_bytes_in_flight,
            )
        self.adopt_existing = adopt_existing

    @classmethod
    def from_dict(cls, d: dict) -> "FSNativeL2AdapterConfig":
        base_path = d.get("base_path")
        if not isinstance(base_path, str) or not base_path:
            raise ValueError("base_path must be a non-empty string")

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        relative_tmp_dir = d.get("relative_tmp_dir", "")
        if not isinstance(relative_tmp_dir, str):
            raise ValueError("relative_tmp_dir must be a string")

        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")

        read_ahead_size = d.get("read_ahead_size", None)
        if read_ahead_size is not None:
            if not isinstance(read_ahead_size, int) or read_ahead_size <= 0:
                raise ValueError("read_ahead_size must be a positive integer")

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")
        if max_capacity_gb > 0 and d.get("eviction") is None:
            logger.warning(
                "fs_native: max_capacity_gb=%s is declared for %s but no "
                "'eviction' block is configured, so it is used for usage "
                "accounting only and will NOT bound disk usage. Add an "
                "eviction block to enforce it, e.g. "
                '"eviction": {"eviction_policy": "LRU"}.',
                max_capacity_gb,
                base_path,
            )

        adopt_existing = d.get("adopt_existing", True)
        if not isinstance(adopt_existing, bool):
            raise ValueError("adopt_existing must be a boolean")

        return cls(
            base_path=base_path,
            num_workers=num_workers,
            relative_tmp_dir=str(relative_tmp_dir),
            use_odirect=use_odirect,
            read_ahead_size=read_ahead_size,
            max_capacity_gb=float(max_capacity_gb),
            num_read_workers=d.get("num_read_workers", DEFAULT_NUM_READ_WORKERS),
            read_max_bytes_in_flight=d.get("read_max_bytes_in_flight", 0),
            adopt_existing=adopt_existing,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "FS native L2 adapter config fields:\n"
            "- base_path (str): directory for KV "
            "cache files (required)\n"
            "- num_workers (int): C++ worker threads "
            "for I/O (default 4, >0)\n"
            "- relative_tmp_dir (str): relative "
            "sub-dir for temp files (default empty)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (default false)\n"
            "- num_read_workers (int): threads dedicated to "
            "reads, decoupling device queue depth from "
            "num_workers (default 256, 0 = legacy path)\n"
            "- read_max_bytes_in_flight (int): bytes kept "
            "outstanding against the device (default 0 = "
            "1536 MiB when num_read_workers > 0)\n"
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes "
            "first (optional)\n"
            "- max_capacity_gb (float): declared L2 capacity in GB "
            "for usage accounting (default 0 = disabled). Does not "
            "bound disk usage by itself; add an 'eviction' block "
            "to enforce it\n"
            "- adopt_existing (bool): on startup, seed "
            "accounting/eviction from files persisted by "
            "a previous run (default true)"
        )


def scan_adoptable_files(
    base_path: str,
    relative_tmp_dir: str,
) -> list[tuple["ObjectKey", int, float]]:
    """Scan ``base_path`` for adoptable object files from a previous run.

    Only completed objects are considered: the FS connectors write through a
    temp file and atomically rename to the final name, so any file that both
    carries the ``.data`` extension and decodes back to an ``ObjectKey`` is a
    fully written object. Temp files (under ``relative_tmp_dir`` or with a
    non-``.data`` extension) and foreign files are skipped.

    Args:
        base_path: The adapter's object directory.
        relative_tmp_dir: The adapter's temp sub-directory name (empty when
            temp files live next to the objects under a different extension).

    Returns:
        One ``(key, size_bytes, mtime)`` tuple per adoptable file, sorted by
        ``mtime`` ascending so callers can seed LRU order oldest-first.
    """
    found: list[tuple[ObjectKey, int, float]] = []
    try:
        entries = list(os.scandir(base_path))
    except FileNotFoundError:
        return []
    for entry in entries:
        if not entry.is_file(follow_symlinks=False):
            continue
        key = filename_to_object_key(entry.name)
        if key is None:
            continue
        stat = entry.stat(follow_symlinks=False)
        found.append((key, stat.st_size, stat.st_mtime))
    found.sort(key=lambda item: item[2])
    return found


_ADOPT_NOTIFY_BATCH = 4096


def _make_adapter_class(native_cls):
    """Build the FS-native adapter class on top of the lazily imported
    ``NativeConnectorL2Adapter`` (imported lazily to avoid the circular
    dependency documented in ``_create_fs_native_l2_adapter``)."""

    class _FSNativeConnectorL2Adapter(native_cls):
        """``NativeConnectorL2Adapter`` plus filesystem startup adoption."""

        def __init__(
            self,
            *args,
            base_path="",
            relative_tmp_dir="",
            adopt_existing=True,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self._base_path = base_path
            self._relative_tmp_dir = relative_tmp_dir
            self._adopt_existing = adopt_existing

        def adopt_existing_keys(self) -> int:
            if not self._adopt_existing:
                return 0
            adoptable = scan_adoptable_files(self._base_path, self._relative_tmp_dir)
            for start in range(0, len(adoptable), _ADOPT_NOTIFY_BATCH):
                batch = adoptable[start : start + _ADOPT_NOTIFY_BATCH]
                self._notify_keys_stored(
                    [key for key, _size, _mtime in batch],
                    [size for _key, size, _mtime in batch],
                )
            if adoptable:
                total = sum(size for _key, size, _mtime in adoptable)
                logger.info(
                    "FS native adapter adopted %d existing objects (%.2f GiB) from %s",
                    len(adoptable),
                    total / (1 << 30),
                    self._base_path,
                )
            return len(adoptable)

    return _FSNativeConnectorL2Adapter


def _create_fs_native_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    C++ filesystem connector."""
    try:
        # First Party
        from lmcache.lmcache_fs import (
            LMCacheFSClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "FS native L2 adapter requires the C++ FS "
            "extension. Build with: pip install -e ."
        ) from e

    # Lazy import to avoid circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (  # noqa: E501
        NativeConnectorL2Adapter,
    )

    assert isinstance(config, FSNativeL2AdapterConfig)
    native_client = LMCacheFSClient(
        config.base_path,
        config.num_workers,
        config.relative_tmp_dir,
        config.use_odirect,
        config.read_ahead_size or 0,
        config.num_read_workers,
        config.read_max_bytes_in_flight,
    )
    effective_read_budget = native_client.read_budget_bytes()
    logger.info(
        "Created FS native L2 adapter: %s (workers=%d, odirect=%s, "
        "read_ahead=%s, num_read_workers=%d, "
        "read_max_bytes_in_flight=%d)",
        config.base_path,
        config.num_workers,
        config.use_odirect,
        config.read_ahead_size,
        config.num_read_workers,
        effective_read_budget,
    )
    residency = MemoryResidency.REMOTE if config.shared else MemoryResidency.LOCAL
    memory_tags = MemoryTags(MemorySegment.STORAGE, residency)
    adapter_cls = _make_adapter_class(NativeConnectorL2Adapter)
    return adapter_cls(
        native_client,
        memory_tags,
        max_capacity_gb=config.max_capacity_gb,
        type_name="FSNativeL2Adapter",
        pad_buffers_to_alignment=config.use_odirect,
        extra_status={
            "base_path": config.base_path,
            "use_odirect": config.use_odirect,
            "num_workers": config.num_workers,
            "read_ahead_size": config.read_ahead_size,
            "num_read_workers": config.num_read_workers,
            "read_max_bytes_in_flight": effective_read_budget,
            "pad_buffers_to_alignment": config.use_odirect,
        },
        base_path=config.base_path,
        relative_tmp_dir=config.relative_tmp_dir,
        adopt_existing=config.adopt_existing,
    )


register_l2_adapter_type("fs_native", FSNativeL2AdapterConfig)
register_l2_adapter_factory("fs_native", _create_fs_native_l2_adapter)
