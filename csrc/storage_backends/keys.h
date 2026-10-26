// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>
#include <stdexcept>

namespace lmcache {
namespace connector {

static constexpr char KEY_SEP = '@';
static constexpr const char* PATH_SLASH_REPLACEMENT = "-SEP-";
static constexpr const char* FILE_EXT = ".data";

std::string replace_all(const std::string& str, const std::string& from,
                        const std::string& to) {
  std::string result = str;
  size_t pos = 0;
  while ((pos = result.find(from, pos)) != std::string::npos) {
    result.replace(pos, from.size(), to);
    pos += to.size();
  }
  return result;
}

std::string key_to_filename(const std::string& key,
                            bool shard_directories = false) {
  // Input key format (from _object_key_to_string):
  // Unsalted:
  //     <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>
  // Salted:
  //     <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>
  // Salted + KV dtype (kv_dtype forces the salt slot, even when empty):
  //     <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>@<kv_dtype>
  //
  // Output filename (matching fs_key_codec.py.object_key_to_filename):
  // Unsalted::
  //     <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>.data
  // Salted (trailing ``cache_salt``)::
  //     <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>.data
  // With dtype::
  //     <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>@<kv_dtype>.data
  //
  // The unsalted 4-field shape is bit-identical to the pre-cache_salt
  // format, so existing cache directories remain valid.
  //
  // NOTE: model_name, cache_salt and kv_dtype are forbidden from
  // containing '@' (invariant enforced on the Python side), so
  // splitting on '@' is unambiguous — no marker, no rsplit.

  // Split on '@' — must yield 4 (unsalted), 5 (salted) or 6
  // (salted + kv_dtype) fields.
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= key.size(); ++pos) {
    if (pos == key.size() || key[pos] == KEY_SEP) {
      parts.emplace_back(key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  if (parts.size() != 4 && parts.size() != 5 && parts.size() != 6) {
    throw std::runtime_error(
        "FSConnector: malformed key (expected 4, 5 or 6 '@'-separated "
        "fields): " +
        key);
  }

  const std::string& model_name = parts[0];
  const std::string& kv_rank_hex = parts[1];
  const std::string& object_group_id_hex = parts[2];
  const std::string& chunk_hash = parts[3];
  const std::string cache_salt = parts.size() >= 5 ? parts[4] : std::string();
  const std::string kv_dtype = parts.size() == 6 ? parts[5] : std::string();

  // Replace '/' with '-SEP-' for filesystem safety
  std::string safe_model = replace_all(model_name, "/", PATH_SLASH_REPLACEMENT);

  // Emit filename. Salt and dtype are appended at the tail so the
  // pre-dtype shapes match what older builds wrote to disk. A non-empty
  // dtype always emits the (possibly empty) salt slot first, keeping the
  // dtype field positionally unambiguous.
  std::string result;
  result.reserve(safe_model.size() + kv_rank_hex.size() +
                 object_group_id_hex.size() + chunk_hash.size() +
                 cache_salt.size() + kv_dtype.size() + 32);
  result += safe_model;
  result += KEY_SEP;
  result += "0x";
  result += kv_rank_hex;
  result += KEY_SEP;
  result += object_group_id_hex;
  result += KEY_SEP;
  result += chunk_hash;
  if (!kv_dtype.empty()) {
    result += KEY_SEP;
    result += cache_salt;
    result += KEY_SEP;
    result += kv_dtype;
  } else if (!cache_salt.empty()) {
    result += KEY_SEP;
    result += cache_salt;
  }
  result += FILE_EXT;

  if (!shard_directories) return result;
  if (chunk_hash.size() < 4) {
    throw std::runtime_error(
        "shard_dirs requires a chunk hash of at least two bytes");
  }
  return chunk_hash.substr(0, 2) + "/" + chunk_hash.substr(2, 2) + "/" + result;
}

// Inverse of key_to_filename(): convert a flat (non-sharded) filename back to
// wire key.
//
// The flat (non-sharded) filename shape produced by key_to_filename():
//   <safe_model>@0x<kv_rank_hex>@<ogid_hex>@<chunk_hash_hex>[@<cache_salt>].data
// Returns the wire key:
//   <model>@<kv_rank_hex>@<ogid_hex>@<chunk_hash_hex>[@<cache_salt>]
//
// Returns an empty string for any file that is not a well-formed KV-key
// data file (e.g.: wrong extension, fewer than 4 '@'-separated fields).
std::string filename_to_key(const std::string& filename) {
  const std::string ext(FILE_EXT);
  if (filename.size() <= ext.size() ||
      filename.compare(filename.size() - ext.size(), ext.size(), ext) != 0) {
    return {};
  }

  // Strip the ".data" suffix
  const std::string body = filename.substr(0, filename.size() - ext.size());

  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= body.size(); ++pos) {
    if (pos == body.size() || body[pos] == KEY_SEP) {
      parts.emplace_back(body.substr(start, pos - start));
      start = pos + 1;
    }
  }
  // Need model, kv_rank, ogid and chunk_hash (4 fields minimum);
  // 5 adds cache_salt, 6 adds cache_salt + kv_dtype.
  if (parts.size() != 4 && parts.size() != 5 && parts.size() != 6) {
    return {};
  }

  // Restore '/' in model_name
  std::string model = parts[0];
  const std::string sep(PATH_SLASH_REPLACEMENT);
  size_t spos = 0;
  while ((spos = model.find(sep, spos)) != std::string::npos) {
    model.replace(spos, sep.size(), "/");
    spos += 1;
  }

  // Drop the "0x" prefix from kv_rank.
  std::string kv_rank = parts[1];
  if (kv_rank.size() >= 2 && kv_rank.substr(0, 2) == "0x") {
    kv_rank = kv_rank.substr(2);
  }

  // Reassemble the wire key.
  std::string result = model;
  result += KEY_SEP;
  result += kv_rank;
  for (size_t i = 2; i < parts.size(); ++i) {
    result += KEY_SEP;
    result += parts[i];
  }
  return result;
}

// Extract the ``chunk_hash`` field from key.
//
// Wire key format:
//   <model>@<kv_rank_hex>@<ogid_hex>@<chunk_hash_hex>[@<cache_salt>][@<kv_dtype>]
//
// The chunk_hash is always the 4th '@'-separated field (index 3).
std::string get_chunk_hash_from_key(const std::string& key) {
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= key.size(); ++pos) {
    if (pos == key.size() || key[pos] == KEY_SEP) {
      parts.emplace_back(key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  if (parts.size() != 4 && parts.size() != 5 && parts.size() != 6) {
    throw std::runtime_error(
        "Malformed key (expected 4, 5 or 6 '@'-separated fields): " + key);
  }
  if (parts[3].empty()) {
    throw std::runtime_error(
        "Malformed key (chunk hash field which behind the third '@' is "
        "empty): " +
        key);
  }
  return parts[3];
}
}  // namespace connector
}  // namespace lmcache
