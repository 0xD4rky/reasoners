#pragma once
#include <algorithm>
#include <cassert>
#include <cctype>
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

struct PairHash {
  inline size_t operator()(const std::pair<int,int>& p) const noexcept {
    return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
  }
};

class QwenBPETokenizer {
public:
  struct Config {
    bool add_prefix_space = false;
    bool trim_offsets = true;
  };

  explicit QwenBPETokenizer(const std::string& tokenizer_json_path) {
    load_from_hf_json(tokenizer_json_path);
    build_byte_maps();
    build_bpe_rank();
  }

  std::vector<int> encode(const std::string& text,
                          const std::unordered_set<std::string>& allowed_special = {}) const {
    std::vector<int> out;
    out.reserve(text.size() / 3);
    
    size_t pos = 0;
    const size_t len = text.size();
    
    while (pos < len) {
      size_t matched = 0;
      int matched_id = -1;
      
      if (!special_by_prefix_.empty()) {
        auto it = special_by_prefix_.find(text[pos]);
        if (it != special_by_prefix_.end()) {
          for (const auto& [tok, id] : it->second) {
            if (!allowed_special.empty() && !allowed_special.count(tok)) continue;
            if (tok.size() <= len - pos && memcmp(tok.data(), text.data() + pos, tok.size()) == 0) {
              if (tok.size() > matched) { matched = tok.size(); matched_id = id; }
            }
          }
        }
      }
      
      if (matched) {
        out.push_back(matched_id);
        pos += matched;
        continue;
      }

      size_t chunk_end = find_next_chunk(text, pos);
      if (chunk_end > pos) {
        std::string encoded;
        encoded.reserve((chunk_end - pos) * 2);
        for (size_t i = pos; i < chunk_end; ++i) {
          encoded += codepoint_to_utf8(byte2unicode_[static_cast<uint8_t>(text[i])]);
        }
        
        encode_word_bpe_optimized(encoded, out);
        pos = chunk_end;
      } else {
        pos++;
      }
    }
    
    return out;
  }

  std::string decode(const std::vector<int>& ids) const {
    std::string byte_level_concat;
    byte_level_concat.reserve(ids.size() * 3);
    for (int id : ids) {
      auto it = id_to_token_.find(id);
      if (it != id_to_token_.end()) {
        byte_level_concat.append(it->second);
      }
    }
    return bytelevel_decode(byte_level_concat);
  }

  inline int token_to_id(const std::string& t) const {
    auto it = token_to_id_.find(t);
    return (it == token_to_id_.end()) ? -1 : it->second;
  }

  const Config& config() const { return config_; }

private:
  Config config_;
  std::unordered_map<std::string,int> token_to_id_;
  std::unordered_map<int,std::string> id_to_token_;
  std::vector<std::pair<int,int>> merges_;
  std::unordered_map<std::pair<int,int>, int, PairHash> bpe_rank_;
  std::vector<uint32_t> byte2unicode_;
  std::unordered_map<uint32_t,uint8_t> unicode2byte_;
  std::unordered_map<char, std::vector<std::pair<std::string,int>>> special_by_prefix_;

  inline static bool is_letter(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= 0x80);
  }

  inline static bool is_digit(unsigned char c) {
    return c >= '0' && c <= '9';
  }

  inline static bool is_whitespace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  }

  size_t find_next_chunk(const std::string& text, size_t start) const {
    if (start >= text.size()) return start;
    
    size_t pos = start;
    const unsigned char first = text[pos];
    
    if (pos + 1 < text.size() && first == '\'') {
      const char next = text[pos + 1];
      if (next == 's' || next == 'S' || next == 't' || next == 'T' || 
          next == 'm' || next == 'M' || next == 'd' || next == 'D') {
        return pos + 2;
      }
      if (pos + 2 < text.size()) {
        if ((next == 'l' || next == 'L') && (text[pos+2] == 'l' || text[pos+2] == 'L')) return pos + 3;
        if ((next == 'r' || next == 'R') && (text[pos+2] == 'e' || text[pos+2] == 'E')) return pos + 3;
        if ((next == 'v' || next == 'V') && (text[pos+2] == 'e' || text[pos+2] == 'E')) return pos + 3;
      }
    }
    
    bool has_prefix = false;
    if (!is_letter(first) && !is_digit(first) && first != '\r' && first != '\n') {
      has_prefix = true;
      pos++;
      if (pos >= text.size()) return pos;
    }
    
    if (is_letter(text[pos])) {
      while (pos < text.size() && is_letter(text[pos])) pos++;
      return pos;
    }
    
    if (is_digit(text[pos])) {
      size_t count = 0;
      while (pos < text.size() && is_digit(text[pos]) && count < 3) {
        pos++;
        count++;
      }
      return pos;
    }
    
    if (!has_prefix) {
      if (first == '\n' || first == '\r') {
        while (pos < text.size() && is_whitespace(text[pos])) pos++;
        return pos;
      }
      
      if (is_whitespace(first)) {
        while (pos < text.size() && text[pos] == ' ') pos++;
        return pos;
      }
    }
    
    return pos > start ? pos : start + 1;
  }

  void encode_word_bpe_optimized(const std::string& encoded_word, std::vector<int>& out) const {
    if (encoded_word.empty()) return;
    
    auto symbols = split_into_utf8_chars(encoded_word);
    if (symbols.empty()) return;
    
    if (symbols.size() == 1) {
      auto it = token_to_id_.find(symbols[0]);
      if (it != token_to_id_.end()) out.push_back(it->second);
      return;
    }

    std::vector<int> ids;
    ids.reserve(symbols.size());
    for (const auto& s : symbols) {
      auto it = token_to_id_.find(s);
      if (it == token_to_id_.end()) {
        ids.push_back(-1);
      } else {
        ids.push_back(it->second);
      }
    }

    while (ids.size() > 1) {
      int best_rank = std::numeric_limits<int>::max();
      int best_idx = -1;
      
      for (size_t i = 0; i + 1 < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i+1] < 0) continue;
        
        auto it = bpe_rank_.find({ids[i], ids[i+1]});
        if (it != bpe_rank_.end() && it->second < best_rank) {
          best_rank = it->second;
          best_idx = i;
        }
      }
      
      if (best_idx == -1) break;
      
      const std::string& tok1 = id_to_token_.at(ids[best_idx]);
      const std::string& tok2 = id_to_token_.at(ids[best_idx + 1]);
      
      std::string merged;
      merged.reserve(tok1.size() + tok2.size());
      merged = tok1 + tok2;
      
      auto it = token_to_id_.find(merged);
      if (it == token_to_id_.end()) break;
      
      ids[best_idx] = it->second;
      symbols[best_idx] = std::move(merged);
      ids.erase(ids.begin() + best_idx + 1);
      symbols.erase(symbols.begin() + best_idx + 1);
    }

    for (int id : ids) {
      if (id >= 0) out.push_back(id);
    }
  }

  void load_from_hf_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open tokenizer.json at " + path);
    json j;
    f >> j;

    const auto& model = j.at("model");
    const std::string type = model.at("type").get<std::string>();
    if (type != "BPE")
      throw std::runtime_error("Only BPE model is supported");

    const auto& vocab = model.at("vocab");
    token_to_id_.reserve(vocab.size());
    id_to_token_.reserve(vocab.size());
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
      token_to_id_.emplace(it.key(), it.value().get<int>());
      id_to_token_.emplace(it.value().get<int>(), it.key());
    }

    if (model.contains("merges")) {
      const auto& merges = model.at("merges");
      merges_.reserve(merges.size());
      for (const auto& m : merges) {
        const std::string s = m.get<std::string>();
        auto sp = s.find(' ');
        if (sp == std::string::npos) continue;
        merges_.emplace_back(token_to_id_.at(s.substr(0, sp)), token_to_id_.at(s.substr(sp+1)));
      }
    } else {
      throw std::runtime_error("BPE merges missing");
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
      if (j["pre_tokenizer"].is_array()) {
        for (const auto& n : j["pre_tokenizer"]) scan(n);
      } else {
        scan(j["pre_tokenizer"]);
      }
    }

    if (j.contains("added_tokens")) {
      for (const auto& t : j["added_tokens"]) {
        if (!t.value("special", false)) continue;
        std::string content = t.at("content").get<std::string>();
        int id = t.at("id").get<int>();
        special_by_prefix_[content.empty() ? '\0' : content[0]].emplace_back(content, id);
      }
      for (auto& [_, vec] : special_by_prefix_) {
        std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
          return a.first.size() > b.first.size();
        });
      }
    }
  }

  void build_bpe_rank() {
    bpe_rank_.reserve(merges_.size() * 1.3);
    for (size_t i = 0; i < merges_.size(); ++i) {
      bpe_rank_.emplace(merges_[i], static_cast<int>(i));
    }
  }

  void build_byte_maps() {
    byte2unicode_.resize(256);
    
    std::unordered_set<int> visible;
    for (int i = 0; i < 256; ++i) {
      if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))
        visible.insert(i);
    }
    
    int n = 0;
    for (int byte = 0; byte < 256; ++byte) {
      uint32_t cp = visible.count(byte) ? byte : 256 + n++;
      byte2unicode_[byte] = cp;
      unicode2byte_[cp] = static_cast<uint8_t>(byte);
    }
  }

  inline std::string bytelevel_decode(const std::string& s) const { // qwen impls diff
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
      auto [cp, adv] = utf8_to_cp(s, i);
      i += adv;
      auto it = unicode2byte_.find(cp);
      if (it != unicode2byte_.end()) {
        out.push_back(static_cast<char>(it->second));
      }
    }
    return out;
  }

  static inline std::string codepoint_to_utf8(uint32_t cp) {
    if (cp <= 0x7F) {
      return std::string(1, static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
      return std::string{
        static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)),
        static_cast<char>(0x80 | (cp & 0x3F))
      };
    } else if (cp <= 0xFFFF) {
      return std::string{
        static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)),
        static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
        static_cast<char>(0x80 | (cp & 0x3F))
      };
    } else {
      return std::string{
        static_cast<char>(0xF0 | ((cp >> 18) & 0x07)),
        static_cast<char>(0x80 | ((cp >> 12) & 0x3F)),
        static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
        static_cast<char>(0x80 | (cp & 0x3F))
      };
    }
  }

  static inline std::pair<uint32_t,size_t> utf8_to_cp(const std::string& s, size_t i) {
    const unsigned char c0 = s[i];
    if (c0 < 0x80) return {c0, 1};
    if ((c0 >> 5) == 0x6) return {((c0 & 0x1F) << 6) | (s[i+1] & 0x3F), 2};
    if ((c0 >> 4) == 0xE) return {((c0 & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F), 3};
    return {((c0 & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F), 4};
  }

  static std::vector<std::string> split_into_utf8_chars(const std::string& s) {
    std::vector<std::string> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
      auto [cp, adv] = utf8_to_cp(s, i);
      out.emplace_back(s.substr(i, adv));
      i += adv;
    }
    return out;
  }
};

