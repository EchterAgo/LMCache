# SPDX-License-Identifier: Apache-2.0
"""Reversible ObjectKey <-> filename codec shared by filesystem L2 adapters.

Both the pure-Python ``fs`` adapter and the C++-backed ``fs_native`` adapter
(see ``csrc/storage_backends/fs/connector.cpp``) lay objects out as one file
per key using this exact encoding, so a cache directory written by either is
readable by both — and by the startup-adoption scan that re-seeds byte
accounting from files left by a previous server lifetime.
"""

# Standard
from typing import Optional

# First Party
from lmcache.v1.distributed.api import ObjectKey

KEY_SEP = "@"
# ``@`` in both ``model_name`` and ``cache_salt`` is rejected by
# ObjectKey.__post_init__, so splitting on ``@`` is unambiguous.
# Kept in sync with native_connector_l2_adapter.py and
# csrc/storage_backends/fs/connector.cpp.
PATH_SLASH_REPLACEMENT = "-SEP-"
FILE_EXT = ".data"


def object_key_to_filename(key: ObjectKey) -> str:
    """Build a reversible, filesystem-safe filename.

    Unsalted::

        <safe_model>@0x<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>.data

    Salted (trailing ``cache_salt``)::

        <safe_model>@0x<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>.data

    ``kv_rank`` is written in ``0x`` prefixed hex so each byte
    of the bitmap ``(ws<<24)|(rank<<16)|(local_ws<<8)|local``
    is directly readable. ``object_group_id`` is written in plain hex.
    """
    safe_model = key.model_name.replace("/", PATH_SLASH_REPLACEMENT)
    base = (
        f"{safe_model}{KEY_SEP}{key.kv_rank:#010x}"
        f"{KEY_SEP}{key.object_group_id:x}{KEY_SEP}{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}{KEY_SEP}{key.cache_salt}{FILE_EXT}"
    return f"{base}{FILE_EXT}"


def filename_to_object_key(
    filename: str,
) -> Optional[ObjectKey]:
    """Reverse :func:`object_key_to_filename`.

    Accepts both the 4-field unsalted shape and the 5-field salted
    shape (trailing ``cache_salt``). Returns ``None`` for anything
    else. Since ``model_name`` is guaranteed not to contain ``@``,
    plain ``split`` suffices — no marker, no rsplit.
    """
    if not filename.endswith(FILE_EXT):
        return None
    stem = filename[: -len(FILE_EXT)]
    parts = stem.split(KEY_SEP)
    if len(parts) == 4:
        safe_model, kv_rank_str, object_group_str, chunk_hash_hex = parts
        cache_salt = ""
    elif len(parts) == 5:
        safe_model, kv_rank_str, object_group_str, chunk_hash_hex, cache_salt = parts
    else:
        return None

    model_name = safe_model.replace(PATH_SLASH_REPLACEMENT, "/")
    try:
        chunk_hash = bytes.fromhex(chunk_hash_hex)
        kv_rank = int(kv_rank_str, 16)
        object_group_id = int(object_group_str, 16)
        # ObjectKey.__post_init__ raises ValueError when the decoded
        # model_name / cache_salt violate the forbidden-char or length
        # invariants (e.g. a stray file from another tool on disk).
        # The contract here is to return None for anything unparsable,
        # so keep the constructor inside the try block.
        return ObjectKey(
            chunk_hash=chunk_hash,
            model_name=model_name,
            kv_rank=kv_rank,
            object_group_id=object_group_id,
            cache_salt=cache_salt,
        )
    except ValueError:
        return None
