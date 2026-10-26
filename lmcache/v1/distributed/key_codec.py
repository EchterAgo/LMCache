# SPDX-License-Identifier: Apache-2.0
"""Reversible L2 key names shared by storage adapters."""

# Standard
from collections.abc import Callable
import urllib.parse

# First Party
from lmcache.v1.distributed.api import ObjectKey


def encode_storage_key(key: ObjectKey) -> str:
    """Return the canonical L2 wire name, including every matching field."""
    return format_storage_key(key, "{model}@{rank}@{group}@{hash}{suffix}")


def format_storage_key(
    key: ObjectKey,
    template: str,
    *,
    model_separator: str = "/",
    rank_format: str = "08x",
    model_suffix: str = "",
) -> str:
    """Format an adapter's existing key template without changing its names.

    Args:
        key: Object whose identity determines the name.
        template: Format string accepting ``model``, ``quoted_model``, ``rank``,
            ``group``, ``hash``, ``hash_prefix``, ``salt``, ``dtype`` and
            ``suffix``. ``suffix`` includes the salt when nonempty, and the
            dtype when nonempty (with the salt slot always emitted for a
            non-empty dtype, so the dtype stays in a fixed position). An
            empty trailing ``@{dtype}`` or ``@{salt}`` is omitted to preserve
            existing storage names.
        model_separator: Replacement for slashes in ``model``.
        rank_format: Python integer format for ``rank``.
        model_suffix: Adapter-specific suffix applied to ``model``.

    Returns:
        The formatted storage name.
    """
    if not key.kv_dtype and template.endswith("@{dtype}"):
        template = template[:-8]
    if not key.cache_salt and template.endswith("@{salt}"):
        template = template[:-7]
    if key.kv_dtype:
        suffix = f"@{key.cache_salt}@{key.kv_dtype}"
    elif key.cache_salt:
        suffix = f"@{key.cache_salt}"
    else:
        suffix = ""
    chunk_hex = key.chunk_hash.hex()
    result = template.format(
        model=key.model_name.replace("/", model_separator) + model_suffix,
        quoted_model=urllib.parse.quote(key.model_name, safe=""),
        rank=format(key.kv_rank, rank_format),
        group=f"{key.object_group_id:x}",
        hash=chunk_hex,
        hash_prefix=chunk_hex[:4],
        salt=key.cache_salt,
        dtype=key.kv_dtype,
        suffix=suffix,
    )
    return result


def encode_fs_filename(key: ObjectKey) -> str:
    """Return the filename shared by Python and native filesystem adapters."""
    filename = format_storage_key(
        key,
        "{model}@{rank}@{group}@{hash}{suffix}.data",
        model_separator="-SEP-",
        rank_format="#010x",
    )
    return filename


def decode_storage_key(
    encoded: str, *, model_decoder: Callable[[str], str] = str
) -> ObjectKey:
    """Recover an L2 key from its storage name.

    Args:
        encoded: ``model@rank@group@hash[@salt][@dtype]``. The 6-field shape
        carries ``salt@dtype`` (the salt slot is always present when a
        dtype is).
        model_decoder: Decoder for backend-specific model escaping.

    Returns:
        An identity-equivalent key.

    Raises:
        ValueError: The field count or hexadecimal fields are malformed.
    """
    parts = encoded.split("@")
    kv_dtype = ""
    if len(parts) == 4:
        parts.append("")
    elif len(parts) == 6:
        kv_dtype = parts.pop()
    if len(parts) != 5:
        raise ValueError("invalid ObjectKey encoding: expected 4, 5 or 6 fields")
    model, rank, group, chunk_hash, salt = parts
    return ObjectKey(
        chunk_hash=bytes.fromhex(chunk_hash),
        kv_rank=int(rank, 16),
        object_group_id=int(group, 16),
        model_name=model_decoder(model),
        cache_salt=salt,
        kv_dtype=kv_dtype,
    )


def decode_fs_filename(filename: str) -> ObjectKey:
    """Recover a Python/native FS key; reject invalid suffixes and fields."""
    if not filename.endswith(".data"):
        raise ValueError("invalid ObjectKey filename extension")
    return decode_storage_key(
        filename[:-5], model_decoder=lambda model: model.replace("-SEP-", "/")
    )
