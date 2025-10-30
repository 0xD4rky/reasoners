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
  size_t operator()(const std::pair<int,int>& p) const noexcept {
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
    build_special_maps();
  }

  std::vector<int> encode(const std::string& text,
                          const std::unordered_set<std::string>& allowed_special = {}) const {
    std::vector<int> out;
    size_t i = 0;
    while (i < text.size()) {
      size_t matched = 0;
      int matched_id = -1;
      if (!special_by_prefix_.empty()) {
        auto it = special_by_prefix_.find(text[i]);
        if (it != special_by_prefix_.end()) {
          for (const auto& cand : it->second) {
            const std::string& tok = cand.first;
            int id = cand.second;
            if (!allowed_special.empty() && !allowed_special.count(tok)) continue;
            if (tok.size() <= text.size()-i && memcmp(tok.data(), text.data()+i, tok.size())==0) {
              if (tok.size() > matched) { matched = tok.size(); matched_id = id; }
            }
          }
        }
      }
      if (matched) {
        out.push_back(matched_id);
        i += matched;
        continue;
      }

      size_t j = next_word_span(text, i);
      if (j == i) {
        j = i + 1;
      }
      std::string_view chunk(text.data() + i, j - i);

      std::string encoded = bytelevel_encode(chunk, (i == 0 && config_.add_prefix_space));
      encode_word_bpe(encoded, out);

      i = j;
    }
    return out;
  }

  std::string decode(const std::vector<int>& ids) const {
    std::string byte_level_concat;
    byte_level_concat.reserve(ids.size() * 3);
    for (int id : ids) {
      auto it = id_to_token_.find(id);
      if (it == id_to_token_.end()) {
        continue;
      }
      byte_level_concat += it->second;
    }
    return bytelevel_decode(byte_level_concat);
  }

  int token_to_id(const std::string& t) const {
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

  void load_from_hf_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open tokenizer.json at " + path);
    json j; f >> j;

    const auto& model = j.at("model");
    const std::string type = model.at("type").get<std::string>();
    if (type != "BPE")
      throw std::runtime_error("Only BPE model is supported for Qwen 2.5");

    const auto& vocab = model.at("vocab");
    token_to_id_.reserve(vocab.size());
    id_to_token_.reserve(vocab.size());
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
      const std::string tok = it.key();
      int id = it.value().get<int>();
      token_to_id_.emplace(tok, id);
      id_to_token_.emplace(id, tok);
    }

    if (model.contains("merges")) {
      const auto& merges = model.at("merges");
      merges_.reserve(merges.size());
      for (const auto& m : merges) {
        const std::string s = m.get<std::string>();
        auto sp = s.find(' ');
        if (sp == std::string::npos) continue;
        int a = token_to_id_.at(s.substr(0, sp));
        int b = token_to_id_.at(s.substr(sp+1));
        merges_.push_back({a,b});
      }
    } else {
      throw std::runtime_error("BPE merges missing in tokenizer.json");
    }

    if (j.contains("pre_tokenizer")) {
      auto scan = [&](const json& node) {
        if (!node.is_object()) return;
        if (!node.contains("type")) return;
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
        special_by_prefix_[content.empty() ? '\0' : content[0]].push_back({content, id});
      }
      for (auto& kv : special_by_prefix_) {
        auto& vec = kv.second;
        std::sort(vec.begin(), vec.end(), [](auto& a, auto& b){
          return a.first.size() > b.first.size();
        });
      }
    }
  }

  void build_bpe_rank() {
    bpe_rank_.reserve(merges_.size()*1.2);
    for (size_t i = 0; i < merges_.size(); ++i) {
      bpe_rank_.emplace(merges_[i], static_cast<int>(i));
    }
  }

  void build_byte_maps() {
    byte2unicode_.resize(256);
    std::vector<int> bs;
    for (int i = 0; i < 256; ++i) {
      if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))
        bs.push_back(i);
    }
    std::vector<int> cs = bs;
    int n = 0;
    for (int i = 0; i < 256; ++i) {
      if (std::find(bs.begin(), bs.end(), i) == bs.end()) {
        cs.push_back(256 + n);
        n++;
      }
    }
    for (size_t i = 0; i < 256; ++i) {
      uint32_t cp = static_cast<uint32_t>(cs[i]);
      byte2unicode_[i] = cp;
      unicode2byte_[cp] = static_cast<uint8_t>(i);
    }
  }

  void encode_word_bpe(const std::string& encoded_word, std::vector<int>& out) const {
    std::vector<std::string> symbols = split_into_utf8_chars(encoded_word);

    std::vector<int> ids;
    ids.reserve(symbols.size());
    for (auto& s : symbols) {
      auto it = token_to_id_.find(s);
      if (it == token_to_id_.end()) {
        ids.push_back(-1);
      } else {
        ids.push_back(it->second);
      }
    }

    if (ids.size() <= 1) {
      if (!ids.empty()) out.push_back(ids[0] == -1 ? token_to_id_.at(symbols[0]) : ids[0]);
      return;
    }

    auto get_rank = [&](int a, int b) -> int {
      auto it = bpe_rank_.find({a,b});
      if (it == bpe_rank_.end()) return std::numeric_limits<int>::max();
      return it->second;
    };

    std::vector<std::string> syms = symbols;
    while (true) {
      if (ids.size() == 1) { out.push_back(ids[0]); break; }
      int best_rank = std::numeric_limits<int>::max();
      int best_i = -1;
      for (int i = 0; i+1 < (int)ids.size(); ++i) {
        if (ids[i] < 0 || ids[i+1] < 0) continue;
        int r = get_rank(ids[i], ids[i+1]);
        if (r < best_rank) { best_rank = r; best_i = i; }
      }
      if (best_i == -1) {
        for (int id : ids) {
          if (id >= 0) out.push_back(id);
          else {
            const std::string& g = syms[&id - &ids[0]];
            out.push_back(token_to_id_.at(g));
          }
        }
        break;
      }
      std::string merged = id_to_token_.at(ids[best_i]) + id_to_token_.at(ids[best_i+1]);
      auto it = token_to_id_.find(merged);
      assert(it != token_to_id_.end());
      int merged_id = it->second;

      ids[best_i] = merged_id;
      syms[best_i] = merged;
      ids.erase(ids.begin()+best_i+1);
      syms.erase(syms.begin()+best_i+1);
    }
  }

  std::string bytelevel_encode(std::string_view sv, bool add_leading_space) const {
    std::string out;
    out.reserve(sv.size()*2 + 1);
    if (add_leading_space && (sv.empty() || sv.front() != ' ')) {
      out += codepoint_to_utf8(byte2unicode_[static_cast<uint8_t>(' ')]);
    }
    for (unsigned char c : sv) {
      uint32_t cp = byte2unicode_[c];
      out += codepoint_to_utf8(cp);
    }
    return out;
  }

  std::string bytelevel_decode(const std::string& s) const {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
      uint32_t cp = 0; size_t adv = 0;
      std::tie(cp, adv) = utf8_to_cp(s, i);
      i += adv;
      auto it = unicode2byte_.find(cp);
      if (it != unicode2byte_.end())
        out.push_back(static_cast<char>(it->second));
      else {
        out += codepoint_to_utf8(cp);
      }
    }
    return out;
  }

  static std::string codepoint_to_utf8(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
      out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
      out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
      out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
  }

  static std::pair<uint32_t,size_t> utf8_to_cp(const std::string& s, size_t i) {
    const unsigned char c0 = s[i];
    if (c0 < 0x80) return {c0, 1};
    if ((c0 >> 5) == 0x6) {
      uint32_t cp = ((c0 & 0x1F) << 6) | (s[i+1] & 0x3F);
      return {cp, 2};
    }
    if ((c0 >> 4) == 0xE) {
      uint32_t cp = ((c0 & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F);
      return {cp, 3};
    }
    // 4-byte
    uint32_t cp = ((c0 & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F);
    return {cp, 4};
  }

  static std::vector<std::string> split_into_utf8_chars(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
      auto [cp, adv] = utf8_to_cp(s, i);
      out.emplace_back(s.substr(i, adv));
      i += adv;
    }
    return out;
  }

  static bool is_letter(uint32_t cp) {
    if (cp < 128) return std::isalpha(static_cast<unsigned char>(cp));
    return true;
  }
  
  static bool is_number(uint32_t cp) {
    if (cp < 128) return std::isdigit(static_cast<unsigned char>(cp));
    return false;
  }
  
  static bool is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
  }

  size_t next_word_span(const std::string& s, size_t i) const {
    auto starts_with = [&](size_t pos, const char* lit) {
      size_t n = std::strlen(lit);
      return pos + n <= s.size() && memcmp(s.data()+pos, lit, n) == 0;
    };

    {
      auto [cp, adv] = utf8_to_cp(s, i);
      if (is_space(cp)) {
        size_t j = i + adv;
        while (j < s.size()) {
          auto [cp2, adv2] = utf8_to_cp(s, j);
          if (!is_space(cp2)) break;
          j += adv2;
        }
        return j;
      }
    }

    size_t p = i;
    {
      auto [cp, adv] = utf8_to_cp(s, p);
      if (cp == ' ') { p += adv; }
    }
    if (p >= s.size()) return p;

    if (starts_with(p, "'s") || starts_with(p, "'t") || starts_with(p, "'re") ||
        starts_with(p, "'ve") || starts_with(p, "'m") || starts_with(p, "'ll") || starts_with(p, "'d")) {
      return p + (s[p+1]=='r' || s[p+1]=='v' ? 3 : (s[p+1]=='l' || s[p+1]=='d' ? 3 : 2));
    }

    {
      auto [cp, adv] = utf8_to_cp(s, p);
      if (is_letter(cp)) {
        size_t j = p + adv;
        while (j < s.size()) {
          auto [cp2, adv2] = utf8_to_cp(s, j);
          if (!is_letter(cp2)) break;
          j += adv2;
        }
        return j;
      }
    }
    
    {
      auto [cp, adv] = utf8_to_cp(s, p);
      if (is_number(cp)) {
        size_t j = p + adv;
        while (j < s.size()) {
          auto [cp2, adv2] = utf8_to_cp(s, j);
          if (!is_number(cp2)) break;
          j += adv2;
        }
        return j;
      }
    }
    
    {
      auto [cp, adv] = utf8_to_cp(s, p);
      if (!is_space(cp) && !is_letter(cp) && !is_number(cp)) {
        size_t j = p + adv;
        while (j < s.size()) {
          auto [cp2, adv2] = utf8_to_cp(s, j);
          if (is_space(cp2) || is_letter(cp2) || is_number(cp2)) break;
          j += adv2;
        }
        return j;
      }
    }
    
    auto [cp, adv] = utf8_to_cp(s, p);
    return p + adv;
  }

  void build_special_maps() {
  }
};
