# SPDX-License-Identifier: Apache-2.0
"""Flattening domain values into the plain data a capture may hold.

Object keys appear in several sections and are captured by the million,
so they go out as positional tuples: per-key field names cost more than
the fields. Sharing the encoding keeps the sections agreeing on it.
"""

# Standard
from typing import cast

# First Party
from lmcache.v1.distributed.api import ObjectKey

EncodedKey = tuple[bytes, str, int, int, str, str]
"""An :class:`ObjectKey` as ``(chunk_hash, model_name, kv_rank,
object_group_id, cache_salt, kv_dtype)``."""


def encode_key(key: ObjectKey) -> EncodedKey:
    """Flatten a key for a capture, fields in declaration order."""
    return (
        key.chunk_hash,
        key.model_name,
        key.kv_rank,
        key.object_group_id,
        key.cache_salt,
        key.kv_dtype,
    )


def decode_key(encoded: object) -> ObjectKey:
    """Rebuild a key from :func:`encode_key`.

    Args:
        encoded: An :func:`encode_key` value, as read back from an
            artifact -- a sequence, not necessarily a tuple. Pre-
            ``kv_dtype`` captures carry only the first five fields; the
            dtype then defaults to empty.

    Returns:
        The key, validated by ``ObjectKey`` itself.

    Raises:
        ValueError: If a field violates an ``ObjectKey`` invariant.
    """
    fields = tuple(encoded)  # type: ignore[arg-type]
    if len(fields) == 6:
        (
            chunk_hash,
            model_name,
            kv_rank,
            object_group_id,
            cache_salt,
            kv_dtype,
        ) = cast("EncodedKey", fields)
    elif len(fields) == 5:
        (
            chunk_hash,
            model_name,
            kv_rank,
            object_group_id,
            cache_salt,
        ) = fields
        kv_dtype = ""
    else:
        raise ValueError(f"Invalid encoded ObjectKey arity: {len(fields)}")
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name=model_name,
        kv_rank=kv_rank,
        object_group_id=object_group_id,
        cache_salt=cache_salt,
        kv_dtype=kv_dtype,
    )
