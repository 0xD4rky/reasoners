#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class QwenBPETokenizer {
public:
  struct Config {
    bool add_prefix_space = false;
    bool trim_offsets = true;
  };

  explicit QwenBPETokenizer(const std::string& tokenizer_json_path) {
    load_from_hf_json(tokenizer_json_path);
    build_tables();
  }

  std::vector<int> encode(const std::string& text,
                          const std::unordered_set<std::string>& allowed_special = {}) const {
    std::vector<int> out;
    out.reserve(text.size() / 3);
    encode_into(text, allowed_special, out);
    return out;
  }

  std::vector<std::vector<int>> batch_encode(
      const std::vector<std::string>& texts,
      const std::unordered_set<std::string>& allowed_special = {}) const {
    std::vector<std::vector<int>> results(texts.size());
    for (size_t i = 0; i < texts.size(); i++) {
      results[i].reserve(texts[i].size() / 3);
      encode_into(texts[i], allowed_special, results[i]);
    }
    return results;
  }

  std::string decode(const std::vector<int>& ids) const {
    std::string byte_level;
    byte_level.reserve(ids.size() * 4);
    const int vec_sz = static_cast<int>(id_to_token_vec_.size());
    for (int id : ids) {
      if (id >= 0 && id < vec_sz)
        byte_level.append(id_to_token_vec_[id]);
    }
    return bytelevel_decode(byte_level);
  }

  inline int token_to_id(const std::string& t) const {
    auto it = token_to_id_.find(t);
    return (it == token_to_id_.end()) ? -1 : it->second;
  }

  const Config& config() const { return config_; }

private:
  Config config_;
  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::string> id_to_token_vec_;
  std::vector<std::pair<int,int>> merges_;

  int byte_initial_id_[256];
  uint8_t decode_byte_[512];
  bool decode_valid_[512];

  // Character class table — single lookup replaces multi-branch comparisons
  enum CClass : uint8_t { CC_OTHER=0, CC_LETTER=1, CC_DIGIT=2, CC_SPACE=3, CC_NL=4, CC_APOS=5, CC_HIGH=6 };
  uint8_t cclass_[256];

  struct MergeInfo { int rank; int merged_id; };

  // Open-addressing flat hash map — cache-line friendly merge lookups
  struct MergeMap {
    static constexpr uint64_t EMPTY = ~0ULL;
    struct Slot { uint64_t key; MergeInfo val; };
    std::vector<Slot> slots_;
    uint32_t mask_ = 0;

    static uint64_t mix(uint64_t h) {
      h ^= h >> 33;
      h *= 0xff51afd7ed558ccdULL;
      h ^= h >> 33;
      return h;
    }
    void build(size_t count) {
      uint32_t cap = 16;
      while (cap < count * 2) cap <<= 1;
      slots_.assign(cap, {EMPTY, {}});
      mask_ = cap - 1;
    }
    void insert(uint64_t key, MergeInfo val) {
      uint32_t idx = static_cast<uint32_t>(mix(key)) & mask_;
      while (slots_[idx].key != EMPTY) {
        if (slots_[idx].key == key) { slots_[idx].val = val; return; }
        idx = (idx + 1) & mask_;
      }
      slots_[idx] = {key, val};
    }
    inline const MergeInfo* find(uint64_t key) const {
      uint32_t idx = static_cast<uint32_t>(mix(key)) & mask_;
      while (true) {
        if (slots_[idx].key == key) return &slots_[idx].val;
        if (slots_[idx].key == EMPTY) return nullptr;
        idx = (idx + 1) & mask_;
      }
    }
  } merge_map_;

  std::unordered_map<char, std::vector<std::pair<std::string,int>>> special_by_prefix_;

  // Flat word cache — zero allocations on lookup (no std::string key construction)
  struct CacheSlot {
    uint64_t hash;
    uint16_t data_len;
    uint16_t token_count;
    int tokens[13];
  };
  static constexpr size_t CACHE_SLOTS = 1 << 14;
  static constexpr size_t CACHE_MASK = CACHE_SLOTS - 1;
  static constexpr int MAX_CACHED_TOKENS = 13;
  mutable std::vector<CacheSlot> slot_cache_;

  static inline uint64_t pack_pair(int a, int b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
           static_cast<uint32_t>(b);
  }

  static inline uint64_t chunk_hash(const char* data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
      h ^= static_cast<uint8_t>(data[i]);
      h *= 0x100000001b3ULL;
    }
    return h ^ (h >> 32);
  }

  // ---- Shared encode core ----

  void encode_into(const std::string& text,
                   const std::unordered_set<std::string>& allowed_special,
                   std::vector<int>& out) const {
    size_t pos = 0;
    const size_t len = text.size();
    const char* data = text.data();

    while (pos < len) {
      // Special token match (rare path)
      if (__builtin_expect(!special_by_prefix_.empty(), 0)) {
        auto it = special_by_prefix_.find(data[pos]);
        if (it != special_by_prefix_.end()) {
          size_t matched = 0;
          int matched_id = -1;
          for (const auto& [tok, id] : it->second) {
            if (!allowed_special.empty() && !allowed_special.count(tok)) continue;
            if (tok.size() <= len - pos &&
                memcmp(tok.data(), data + pos, tok.size()) == 0) {
              if (tok.size() > matched) { matched = tok.size(); matched_id = id; }
            }
          }
          if (matched) {
            out.push_back(matched_id);
            pos += matched;
            continue;
          }
        }
      }

      size_t chunk_end = find_next_chunk(data, len, pos);
      if (__builtin_expect(chunk_end > pos, 1)) {
        encode_chunk(data + pos, chunk_end - pos, out);
        pos = chunk_end;
      } else {
        pos++;
      }
    }
  }

  // ---- BPE encode per chunk: flat cache → BPE fallback ----

  inline void encode_chunk(const char* data, size_t len, std::vector<int>& out) const {
    if (__builtin_expect(!len, 0)) return;

    // Flat cache probe — no string allocation, just hash + array index
    uint64_t h = chunk_hash(data, len);
    size_t slot_idx = static_cast<size_t>(h) & CACHE_MASK;
    auto& slot = slot_cache_[slot_idx];

    if (__builtin_expect(slot.hash == h && slot.data_len == static_cast<uint16_t>(len), 1)) {
      out.insert(out.end(), slot.tokens, slot.tokens + slot.token_count);
      return;
    }

    // Cache miss — run BPE
    if (len == 1) {
      int id = byte_initial_id_[static_cast<uint8_t>(data[0])];
      if (id >= 0) {
        out.push_back(id);
        slot = {h, 1, 1, {id}};
      }
      return;
    }

    size_t before = out.size();
    bpe_encode(data, len, out);
    size_t produced = out.size() - before;

    // Store in flat cache if it fits
    if (produced <= MAX_CACHED_TOKENS && len <= 0xFFFF) {
      slot.hash = h;
      slot.data_len = static_cast<uint16_t>(len);
      slot.token_count = static_cast<uint16_t>(produced);
      memcpy(slot.tokens, &out[before], produced * sizeof(int));
    }
  }

  void bpe_encode(const char* data, size_t len, std::vector<int>& out) const {
    constexpr int STACK_MAX = 128;
    int stack_ids[STACK_MAX], stack_ranks[STACK_MAX];
    std::vector<int> heap_ids, heap_ranks;
    int *ids, *ranks;

    if (__builtin_expect(len <= static_cast<size_t>(STACK_MAX), 1)) {
      ids = stack_ids;
      ranks = stack_ranks;
    } else {
      heap_ids.resize(len);
      heap_ranks.resize(len);
      ids = heap_ids.data();
      ranks = heap_ranks.data();
    }

    int n = static_cast<int>(len);
    for (int i = 0; i < n; i++)
      ids[i] = byte_initial_id_[static_cast<uint8_t>(data[i])];

    for (int i = 0; i < n - 1; i++) {
      auto m = merge_map_.find(pack_pair(ids[i], ids[i + 1]));
      ranks[i] = m ? m->rank : INT_MAX;
    }
    ranks[n - 1] = INT_MAX;

    while (n > 1) {
      int min_rank = INT_MAX, min_idx = -1;
      for (int i = 0; i < n - 1; i++) {
        if (ranks[i] < min_rank) {
          min_rank = ranks[i];
          min_idx = i;
        }
      }
      if (__builtin_expect(min_idx < 0, 0)) break;

      auto m = merge_map_.find(pack_pair(ids[min_idx], ids[min_idx + 1]));
      ids[min_idx] = m->merged_id;

      int move_count = n - min_idx - 2;
      if (move_count > 0) {
        memmove(&ids[min_idx + 1], &ids[min_idx + 2], move_count * sizeof(int));
        memmove(&ranks[min_idx + 1], &ranks[min_idx + 2], move_count * sizeof(int));
      }
      n--;

      if (min_idx < n - 1) {
        auto r = merge_map_.find(pack_pair(ids[min_idx], ids[min_idx + 1]));
        ranks[min_idx] = r ? r->rank : INT_MAX;
      } else {
        ranks[min_idx] = INT_MAX;
      }
      if (min_idx > 0) {
        auto r = merge_map_.find(pack_pair(ids[min_idx - 1], ids[min_idx]));
        ranks[min_idx - 1] = r ? r->rank : INT_MAX;
      }
    }

    for (int i = 0; i < n; i++)
      if (ids[i] >= 0) out.push_back(ids[i]);
  }

  // ---- Chunk boundary detection using character class table ----

  size_t find_next_chunk(const char* text, size_t len, size_t start) const {
    if (start >= len) return start;

    size_t pos = start;
    const uint8_t cc_first = cclass_[static_cast<uint8_t>(text[pos])];

    // Contraction handling
    if (cc_first == CC_APOS && pos + 1 < len) {
      const char next = text[pos + 1];
      if (next == 's' || next == 'S' || next == 't' || next == 'T' ||
          next == 'm' || next == 'M' || next == 'd' || next == 'D')
        return pos + 2;
      if (pos + 2 < len) {
        if ((next == 'l' || next == 'L') && (text[pos+2] == 'l' || text[pos+2] == 'L')) return pos + 3;
        if ((next == 'r' || next == 'R') && (text[pos+2] == 'e' || text[pos+2] == 'E')) return pos + 3;
        if ((next == 'v' || next == 'V') && (text[pos+2] == 'e' || text[pos+2] == 'E')) return pos + 3;
      }
    }

    bool has_prefix = false;
    if (cc_first != CC_LETTER && cc_first != CC_HIGH && cc_first != CC_DIGIT && cc_first != CC_NL) {
      has_prefix = true;
      pos++;
      if (pos >= len) return pos;
    }

    uint8_t cc = cclass_[static_cast<uint8_t>(text[pos])];

    if (cc == CC_LETTER || cc == CC_HIGH) {
      while (pos < len) {
        uint8_t c = cclass_[static_cast<uint8_t>(text[pos])];
        if (c != CC_LETTER && c != CC_HIGH) break;
        pos++;
      }
      return pos;
    }

    if (cc == CC_DIGIT) {
      size_t count = 0;
      while (pos < len && cclass_[static_cast<uint8_t>(text[pos])] == CC_DIGIT && count < 3) { pos++; count++; }
      return pos;
    }

    if (!has_prefix) {
      if (cc_first == CC_NL) {
        while (pos < len) {
          uint8_t c = cclass_[static_cast<uint8_t>(text[pos])];
          if (c != CC_SPACE && c != CC_NL) break;
          pos++;
        }
        return pos;
      }
      if (cc_first == CC_SPACE) {
        while (pos < len && text[pos] == ' ') pos++;
        return pos;
      }
    }

    return pos > start ? pos : start + 1;
  }

  // ---- Decode: flat vector + flat array ----

  std::string bytelevel_decode(const std::string& s) const {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
      auto [cp, adv] = utf8_to_cp(s, i);
      i += adv;
      if (cp < 512 && decode_valid_[cp])
        out.push_back(static_cast<char>(decode_byte_[cp]));
    }
    return out;
  }

  // ---- Construction ----

  void load_from_hf_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open tokenizer.json at " + path);
    json j;
    f >> j;

    const auto& model = j.at("model");
    if (model.at("type").get<std::string>() != "BPE")
      throw std::runtime_error("Only BPE model is supported");

    const auto& vocab = model.at("vocab");
    token_to_id_.reserve(vocab.size());
    for (auto it = vocab.begin(); it != vocab.end(); ++it)
      token_to_id_.emplace(it.key(), it.value().get<int>());

    if (!model.contains("merges"))
      throw std::runtime_error("BPE merges missing");
    const auto& merges_json = model.at("merges");
    merges_.reserve(merges_json.size());
    for (const auto& m : merges_json) {
      const std::string s = m.get<std::string>();
      auto sp = s.find(' ');
      if (sp == std::string::npos) continue;
      merges_.emplace_back(token_to_id_.at(s.substr(0, sp)),
                           token_to_id_.at(s.substr(sp + 1)));
    }

    if (j.contains("pre_tokenizer")) {
      auto scan = [&](const json& node) {
        if (!node.is_object() || !node.contains("type")) return;
        if (node.at("type").get<std::string>() != "ByteLevel") return;
        if (node.contains("add_prefix_space"))
          config_.add_prefix_space = node.at("add_prefix_space").get<bool>();
        if (node.contains("trim_offsets"))
          config_.trim_offsets = node.at("trim_offsets").get<bool>();
      };
      if (j["pre_tokenizer"].is_array())
        for (const auto& n : j["pre_tokenizer"]) scan(n);
      else
        scan(j["pre_tokenizer"]);
    }

    if (j.contains("added_tokens")) {
      for (const auto& t : j["added_tokens"]) {
        if (!t.value("special", false)) continue;
        std::string content = t.at("content").get<std::string>();
        int id = t.at("id").get<int>();
        special_by_prefix_[content.empty() ? '\0' : content[0]].emplace_back(content, id);
      }
      for (auto& [_, vec] : special_by_prefix_)
        std::sort(vec.begin(), vec.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }
  }

  void build_tables() {
    // Character class table
    memset(cclass_, CC_OTHER, sizeof(cclass_));
    for (int i = 'A'; i <= 'Z'; i++) cclass_[i] = CC_LETTER;
    for (int i = 'a'; i <= 'z'; i++) cclass_[i] = CC_LETTER;
    for (int i = '0'; i <= '9'; i++) cclass_[i] = CC_DIGIT;
    for (int i = 0x80; i <= 0xFF; i++) cclass_[i] = CC_HIGH;
    cclass_[static_cast<uint8_t>(' ')] = CC_SPACE;
    cclass_[static_cast<uint8_t>('\t')] = CC_SPACE;
    cclass_[static_cast<uint8_t>('\n')] = CC_NL;
    cclass_[static_cast<uint8_t>('\r')] = CC_NL;
    cclass_[static_cast<uint8_t>('\'')] = CC_APOS;

    // Flat id→token vector
    int max_id = 0;
    for (auto& [tok, id] : token_to_id_)
      max_id = std::max(max_id, id);
    id_to_token_vec_.resize(max_id + 1);
    for (auto& [tok, id] : token_to_id_)
      id_to_token_vec_[id] = tok;

    // Byte-level tables
    uint32_t byte2unicode[256];
    std::unordered_set<int> visible;
    for (int i = 0; i < 256; i++)
      if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))
        visible.insert(i);

    memset(decode_valid_, 0, sizeof(decode_valid_));
    int n = 0;
    for (int b = 0; b < 256; b++) {
      uint32_t cp = visible.count(b) ? b : 256 + n++;
      byte2unicode[b] = cp;
      if (cp < 512) {
        decode_byte_[cp] = static_cast<uint8_t>(b);
        decode_valid_[cp] = true;
      }
    }

    for (int b = 0; b < 256; b++) {
      std::string utf8 = codepoint_to_utf8(byte2unicode[b]);
      auto it = token_to_id_.find(utf8);
      byte_initial_id_[b] = (it != token_to_id_.end()) ? it->second : -1;
    }

    // Merge map
    merge_map_.build(merges_.size());
    for (size_t i = 0; i < merges_.size(); i++) {
      auto [a, b] = merges_[i];
      if (a >= 0 && a <= max_id && b >= 0 && b <= max_id) {
        std::string merged = id_to_token_vec_[a] + id_to_token_vec_[b];
        auto it = token_to_id_.find(merged);
        if (it != token_to_id_.end())
          merge_map_.insert(pack_pair(a, b), {static_cast<int>(i), it->second});
      }
    }

    // Flat word cache (zero-initialized = all empty)
    slot_cache_.resize(CACHE_SLOTS);
    memset(slot_cache_.data(), 0, CACHE_SLOTS * sizeof(CacheSlot));
  }

  static inline std::string codepoint_to_utf8(uint32_t cp) {
    if (cp <= 0x7F) return std::string(1, static_cast<char>(cp));
    if (cp <= 0x7FF) return std::string{
      static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)),
      static_cast<char>(0x80 | (cp & 0x3F))
    };
    if (cp <= 0xFFFF) return std::string{
      static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)),
      static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
      static_cast<char>(0x80 | (cp & 0x3F))
    };
    return std::string{
      static_cast<char>(0xF0 | ((cp >> 18) & 0x07)),
      static_cast<char>(0x80 | ((cp >> 12) & 0x3F)),
      static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
      static_cast<char>(0x80 | (cp & 0x3F))
    };
  }

  static inline std::pair<uint32_t, size_t> utf8_to_cp(const std::string& s, size_t i) {
    const unsigned char c0 = s[i];
    if (c0 < 0x80) return {c0, 1};
    if ((c0 >> 5) == 0x6) return {((c0 & 0x1F) << 6) | (s[i+1] & 0x3F), 2};
    if ((c0 >> 4) == 0xE) return {((c0 & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F), 3};
    return {((c0 & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F), 4};
  }
};
