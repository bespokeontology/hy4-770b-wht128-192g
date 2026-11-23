// Pure-C++17 ByteLevel BPE Tokenizer, Official Hy4-flash Chat Template Renderer & Tool-Call Parser
// Zero Python anywhere.

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "hy4_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace hy4 {

namespace {

struct ByteLevelTables {
    std::array<std::string, 256> byte_to_utf8;
    std::unordered_map<uint32_t, uint8_t> cp_to_byte;

    ByteLevelTables() {
        std::array<bool, 256> direct{};
        for (int b = 33; b <= 126; ++b) direct[b] = true;
        for (int b = 161; b <= 172; ++b) direct[b] = true;
        for (int b = 174; b <= 255; ++b) direct[b] = true;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const uint32_t cp = direct[b] ? uint32_t(b) : uint32_t(256 + n++);
            std::string u8;
            if (cp <= 0x7Fu) {
                u8.push_back(char(cp));
            } else {
                u8.push_back(char(0xC0u | (cp >> 6)));
                u8.push_back(char(0x80u | (cp & 0x3Fu)));
            }
            byte_to_utf8[b] = u8;
            cp_to_byte[cp] = uint8_t(b);
        }
    }
};

const ByteLevelTables& byte_tables() {
    static const ByteLevelTables t;
    return t;
}

uint32_t next_utf8_lossy(const std::string& s, size_t& i) {
    const size_t n = s.size();
    const uint8_t b0 = uint8_t(s[i]);
    if (b0 < 0x80u) { ++i; return b0; }
    if (b0 >= 0xC2u && b0 <= 0xDFu) {
        if (i + 1 < n && (uint8_t(s[i + 1]) & 0xC0u) == 0x80u) {
            uint32_t cp = ((uint32_t(b0 & 0x1Fu) << 6) | uint32_t(uint8_t(s[i + 1]) & 0x3Fu));
            i += 2;
            return cp;
        }
        ++i;
        return 0xFFFDu;
    }
    if (b0 >= 0xE0u && b0 <= 0xEFu) {
        const uint8_t min_b1 = (b0 == 0xE0u) ? 0xA0u : 0x80u;
        const uint8_t max_b1 = (b0 == 0xEDu) ? 0x9Fu : 0xBFu;
        if (i + 1 < n) {
            const uint8_t b1 = uint8_t(s[i + 1]);
            if (b1 >= min_b1 && b1 <= max_b1) {
                if (i + 2 < n && (uint8_t(s[i + 2]) & 0xC0u) == 0x80u) {
                    uint32_t cp = (uint32_t(b0 & 0x0Fu) << 12) |
                                  (uint32_t(b1 & 0x3Fu) << 6) |
                                  uint32_t(uint8_t(s[i + 2]) & 0x3Fu);
                    i += 3;
                    return cp;
                }
                i += 2;
                return 0xFFFDu;
            }
        }
        ++i;
        return 0xFFFDu;
    }
    if (b0 >= 0xF0u && b0 <= 0xF4u) {
        const uint8_t min_b1 = (b0 == 0xF0u) ? 0x90u : 0x80u;
        const uint8_t max_b1 = (b0 == 0xF4u) ? 0x8Fu : 0xBFu;
        if (i + 1 < n) {
            const uint8_t b1 = uint8_t(s[i + 1]);
            if (b1 >= min_b1 && b1 <= max_b1) {
                if (i + 2 < n && (uint8_t(s[i + 2]) & 0xC0u) == 0x80u) {
                    if (i + 3 < n && (uint8_t(s[i + 3]) & 0xC0u) == 0x80u) {
                        uint32_t cp = (uint32_t(b0 & 0x07u) << 18) |
                                      (uint32_t(b1 & 0x3Fu) << 12) |
                                      (uint32_t(uint8_t(s[i + 2]) & 0x3Fu) << 6) |
                                      uint32_t(uint8_t(s[i + 3]) & 0x3Fu);
                        i += 4;
                        return cp;
                    }
                    i += 3;
                    return 0xFFFDu;
                }
                i += 2;
                return 0xFFFDu;
            }
        }
        ++i;
        return 0xFFFDu;
    }
    ++i;
    return 0xFFFDu;
}

void append_cp_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7Fu) {
        out.push_back(char(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(char(0xC0u | (cp >> 6)));
        out.push_back(char(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(char(0xE0u | (cp >> 12)));
        out.push_back(char(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(char(0x80u | (cp & 0x3Fu)));
    } else {
        out.push_back(char(0xF0u | (cp >> 18)));
        out.push_back(char(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(char(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(char(0x80u | (cp & 0x3Fu)));
    }
}

std::string bytes_to_utf8_lossy(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    size_t i = 0;
    while (i < raw.size()) append_cp_utf8(out, next_utf8_lossy(raw, i));
    return out;
}

struct AddedTokenEntry {
    int32_t id = -1;
    std::string content;
    bool special = false;
};

std::string build_args_json(const std::vector<std::pair<std::string, std::string>>& args) {
    JsonValue obj = JsonValue::make_object();
    for (const auto& kv : args) {
        JsonValue parsed;
        if (!kv.second.empty() &&
            (kv.second.front() == '{' || kv.second.front() == '[' ||
             kv.second == "true" || kv.second == "false" || kv.second == "null" ||
             (kv.second.front() == '-' || (kv.second.front() >= '0' && kv.second.front() <= '9'))) &&
            JsonParser::try_parse(kv.second, parsed)) {
            obj.set(kv.first, std::move(parsed));
        } else {
            obj.set(kv.first, JsonValue::make_string(kv.second));
        }
    }
    return obj.dumps_compact();
}

}  // namespace

struct Hy4Tokenizer::Impl {
    std::vector<pcre2_code*> split_regexes;
    std::vector<AddedTokenEntry> added_sorted;
    std::vector<uint8_t> is_added_id;
    std::unordered_map<std::string, uint32_t> merge_ranks;

    ~Impl() {
        for (pcre2_code* re : split_regexes) {
            if (re) pcre2_code_free(re);
        }
    }

    static std::string make_pair_key(const std::string& a, const std::string& b) {
        std::string k;
        k.reserve(a.size() + b.size() + 1);
        k.append(a);
        k.push_back('\x1f');
        k.append(b);
        return k;
    }

    void apply_split_isolated(pcre2_code* re, const std::vector<std::string>& in_pieces,
                              std::vector<std::string>& out_pieces) const {
        out_pieces.clear();
        pcre2_match_data* md = pcre2_match_data_create_from_pattern(re, nullptr);
        for (const std::string& piece : in_pieces) {
            if (piece.empty()) continue;
            size_t offset = 0;
            const size_t len = piece.size();
            while (offset < len) {
                const int rc = pcre2_match(
                    re, reinterpret_cast<PCRE2_SPTR>(piece.data()), len, offset, 0, md, nullptr);
                if (rc < 0) {
                    out_pieces.push_back(piece.substr(offset));
                    break;
                }
                PCRE2_SIZE* ovec = pcre2_get_ovector_pointer(md);
                const size_t m0 = ovec[0];
                const size_t m1 = ovec[1];
                if (m0 > offset) out_pieces.push_back(piece.substr(offset, m0 - offset));
                if (m1 > m0) {
                    out_pieces.push_back(piece.substr(m0, m1 - m0));
                    offset = m1;
                } else {
                    size_t next = offset + 1;
                    while (next < len && (uint8_t(piece[next]) & 0xC0u) == 0x80u) ++next;
                    out_pieces.push_back(piece.substr(offset, next - offset));
                    offset = next;
                }
            }
        }
        pcre2_match_data_free(md);
    }

    void bpe_encode_piece(const std::string& raw_piece,
                          const std::unordered_map<std::string, int32_t>& vocab,
                          std::vector<int32_t>& out_ids) const {
        if (raw_piece.empty()) return;
        const auto& bt = byte_tables();
        std::vector<std::string> word;
        word.reserve(raw_piece.size());
        for (unsigned char b : raw_piece) word.push_back(bt.byte_to_utf8[b]);

        if (word.size() == 1) {
            auto it = vocab.find(word[0]);
            if (it != vocab.end()) out_ids.push_back(it->second);
            return;
        }

        while (word.size() > 1) {
            uint32_t best_rank = std::numeric_limits<uint32_t>::max();
            size_t best_idx = 0;
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                auto it = merge_ranks.find(make_pair_key(word[i], word[i + 1]));
                if (it != merge_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                }
            }
            if (best_rank == std::numeric_limits<uint32_t>::max()) break;
            const std::string first = word[best_idx];
            const std::string second = word[best_idx + 1];
            const std::string merged = first + second;
            std::vector<std::string> next_word;
            next_word.reserve(word.size() - 1);
            size_t i = 0;
            while (i < word.size()) {
                if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
                    next_word.push_back(merged);
                    i += 2;
                } else {
                    next_word.push_back(std::move(word[i]));
                    ++i;
                }
            }
            word = std::move(next_word);
        }

        for (const std::string& sym : word) {
            auto it = vocab.find(sym);
            if (it != vocab.end()) out_ids.push_back(it->second);
        }
    }
};

Hy4Tokenizer::Hy4Tokenizer() : impl_(std::make_unique<Impl>()) {}
Hy4Tokenizer::~Hy4Tokenizer() = default;

void Hy4Tokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream in(tokenizer_json_path, std::ios::binary);
    if (!in) throw std::runtime_error("Hy4Tokenizer: cannot open " + tokenizer_json_path);
    std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    JsonValue root = JsonParser::parse(raw);

    int32_t max_id = -1;
    const auto& model = root.get("model");
    const auto& vocab = model.get("vocab");
    if (vocab.is_object()) {
        for (const auto& kv : vocab.obj_val) {
            const int32_t id = int32_t(kv.second.as_int(-1));
            if (id > max_id) max_id = id;
        }
    }
    const auto& added = root.get("added_tokens");
    if (added.is_array()) {
        for (const auto& item : added.arr_val) {
            const int32_t id = int32_t(item.get("id").as_int(-1));
            if (id > max_id) max_id = id;
        }
    }
    if (max_id < 0) throw std::runtime_error("Hy4Tokenizer: empty vocab in " + tokenizer_json_path);

    const size_t vsz = size_t(max_id + 1);
    id_to_token_.assign(vsz, "");
    id_to_raw_bytes_.assign(vsz, "");
    id_is_special_.assign(vsz, 0);
    impl_->is_added_id.assign(vsz, 0);
    token_to_id_.clear();
    token_to_id_.reserve(vsz * 2);

    if (vocab.is_object()) {
        for (const auto& kv : vocab.obj_val) {
            const int32_t id = int32_t(kv.second.as_int(-1));
            if (id >= 0 && size_t(id) < vsz) {
                id_to_token_[size_t(id)] = kv.first;
                token_to_id_[kv.first] = id;
            }
        }
    }

    impl_->added_sorted.clear();
    if (added.is_array()) {
        for (const auto& item : added.arr_val) {
            const int32_t id = int32_t(item.get("id").as_int(-1));
            const std::string content = item.get("content").as_str();
            const bool special = item.get("special").as_bool(false);
            if (id >= 0 && size_t(id) < vsz) {
                id_to_token_[size_t(id)] = content;
                id_is_special_[size_t(id)] = special ? 1u : 0u;
                impl_->is_added_id[size_t(id)] = 1u;
                token_to_id_[content] = id;
                impl_->added_sorted.push_back({id, content, special});
            }
        }
    }
    std::sort(impl_->added_sorted.begin(), impl_->added_sorted.end(),
              [](const AddedTokenEntry& a, const AddedTokenEntry& b) {
                  return a.content.size() > b.content.size();
              });

    // Precompute exact ByteLevel raw bytes for every token ID
    const auto& bt = byte_tables();
    for (size_t id = 0; id < vsz; ++id) {
        const std::string& tok = id_to_token_[id];
        if (tok.empty()) continue;
        if (impl_->is_added_id[id]) {
            id_to_raw_bytes_[id] = tok;
            continue;
        }
        std::string rb;
        rb.reserve(tok.size());
        size_t i = 0;
        while (i < tok.size()) {
            const uint32_t cp = next_utf8_lossy(tok, i);
            auto it = bt.cp_to_byte.find(cp);
            if (it != bt.cp_to_byte.end()) rb.push_back(char(it->second));
            else append_cp_utf8(rb, cp);
        }
        id_to_raw_bytes_[id] = std::move(rb);
    }

    // Parse BPE merges (handles both ["a","b"] array format used by Hy4-flash and "a b" string format!)
    impl_->merge_ranks.clear();
    const auto& merges = model.get("merges");
    if (merges.is_array()) {
        impl_->merge_ranks.reserve(merges.arr_val.size() * 2);
        uint32_t rank = 0;
        for (const auto& m : merges.arr_val) {
            if (m.is_array() && m.arr_val.size() >= 2) {
                impl_->merge_ranks[Impl::make_pair_key(m.arr_val[0].as_str(), m.arr_val[1].as_str())] = rank++;
            } else if (m.is_string()) {
                const std::string& ms = m.as_str();
                const size_t sp = ms.find(' ');
                if (sp != std::string::npos) {
                    impl_->merge_ranks[Impl::make_pair_key(ms.substr(0, sp), ms.substr(sp + 1))] = rank++;
                }
            }
        }
    }

    // Compile PCRE2 Split regexes from pre_tokenizer
    for (pcre2_code* re : impl_->split_regexes) if (re) pcre2_code_free(re);
    impl_->split_regexes.clear();
    const auto& pt = root.get("pre_tokenizer");
    const auto& pts = pt.get("pretokenizers");
    if (pts.is_array()) {
        for (const auto& sub : pts.arr_val) {
            if (sub.get("type").as_str() == "Split") {
                const std::string pat = sub.get("pattern").get("Regex").as_str();
                if (!pat.empty()) {
                    int errcode = 0;
                    PCRE2_SIZE erroff = 0;
                    pcre2_code* re = pcre2_compile(
                        reinterpret_cast<PCRE2_SPTR>(pat.data()), pat.size(),
                        PCRE2_UTF | PCRE2_UCP, &errcode, &erroff, nullptr);
                    if (!re) throw std::runtime_error("Hy4Tokenizer: PCRE2 compile failed for pattern: " + pat);
                    (void)pcre2_jit_compile(re, PCRE2_JIT_COMPLETE);
                    impl_->split_regexes.push_back(re);
                }
            }
        }
    }
}

std::vector<int32_t> Hy4Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int32_t> out;
    if (text.empty()) return out;

    auto encode_chunk = [&](const std::string& chunk) {
        if (chunk.empty()) return;
        std::vector<std::string> pieces = {chunk};
        std::vector<std::string> next_pieces;
        for (pcre2_code* re : impl_->split_regexes) {
            impl_->apply_split_isolated(re, pieces, next_pieces);
            pieces.swap(next_pieces);
        }
        for (const std::string& p : pieces) {
            impl_->bpe_encode_piece(p, token_to_id_, out);
        }
    };

    if (!parse_special || impl_->added_sorted.empty()) {
        encode_chunk(text);
        return out;
    }

    size_t pos = 0;
    std::string plain;
    while (pos < text.size()) {
        bool matched = false;
        if (text[pos] == '<') {
            for (const auto& at : impl_->added_sorted) {
                if (!at.content.empty() && pos + at.content.size() <= text.size() &&
                    std::memcmp(text.data() + pos, at.content.data(), at.content.size()) == 0) {
                    if (!plain.empty()) { encode_chunk(plain); plain.clear(); }
                    out.push_back(at.id);
                    pos += at.content.size();
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            plain.push_back(text[pos++]);
        }
    }
    if (!plain.empty()) encode_chunk(plain);
    return out;
}

std::vector<int32_t> Hy4Tokenizer::encode_segments(const std::vector<PromptSegment>& segments) const {
    std::vector<std::vector<int32_t>> per_seg(segments.size());
    #pragma omp parallel for schedule(dynamic, 1) if (segments.size() > 4)
    for (size_t i = 0; i < segments.size(); ++i) {
        if (segments[i].kind == PromptSegment::Kind::TokenId) {
            per_seg[i].assign(size_t(std::max(0, segments[i].repeat)), segments[i].token_id);
        } else {
            per_seg[i] = encode(segments[i].text, false);
        }
    }
    size_t total = 0;
    for (const auto& v : per_seg) total += v.size();
    std::vector<int32_t> out;
    out.reserve(total);
    for (const auto& v : per_seg) out.insert(out.end(), v.begin(), v.end());
    return out;
}

std::string Hy4Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
    std::string raw;
    raw.reserve(ids.size() * 4);
    for (int32_t id : ids) {
        if (id < 0 || size_t(id) >= id_to_raw_bytes_.size()) continue;
        if (skip_special_tokens && id_is_special_[size_t(id)]) continue;
        raw.append(id_to_raw_bytes_[size_t(id)]);
    }
    return bytes_to_utf8_lossy(raw);
}

std::string Hy4Tokenizer::decode_single(int32_t id, bool skip_special_tokens) const {
    if (id < 0 || size_t(id) >= id_to_raw_bytes_.size()) return "";
    if (skip_special_tokens && id_is_special_[size_t(id)]) return "";
    return bytes_to_utf8_lossy(id_to_raw_bytes_[size_t(id)]);
}

std::string_view Hy4Tokenizer::raw_bytes_single(int32_t id, bool skip_special_tokens) const {
    if (id < 0 || size_t(id) >= id_to_raw_bytes_.size()) return {};
    if (skip_special_tokens && id_is_special_[size_t(id)]) return {};
    return id_to_raw_bytes_[size_t(id)];
}

const std::string& Hy4Tokenizer::id_to_token(int32_t id) const {
    static const std::string kEmpty;
    if (id < 0 || size_t(id) >= id_to_token_.size()) return kEmpty;
    return id_to_token_[size_t(id)];
}

int32_t Hy4Tokenizer::token_to_id(const std::string& tok) const {
    auto it = token_to_id_.find(tok);
    return (it == token_to_id_.end()) ? -1 : it->second;
}

// ============================================================================
// Official Hy4-flash chat_template.jinja renderer
// ============================================================================
static std::string tool_to_json_hy4(const JsonValue& tool_obj) {
    const JsonValue& t = (tool_obj.is_object() && tool_obj.has("function")) ? tool_obj.get("function") : tool_obj;
    JsonValue filtered = JsonValue::make_object();
    if (t.is_object()) {
        for (const auto& kv : t.obj_val) {
            if (kv.first != "defer_loading" && kv.first != "strict") {
                filtered.set(kv.first, kv.second);
            }
        }
    }
    return filtered.dumps_py();
}

static void append_content_segments(std::vector<PromptSegment>& segs, const JsonValue& content) {
    if (content.is_string()) {
        if (!content.as_str().empty()) segs.push_back(PromptSegment::txt(content.as_str()));
        return;
    }
    if (content.is_array()) {
        for (const auto& item : content.arr_val) {
            if (item.is_string()) {
                segs.push_back(PromptSegment::txt(item.as_str()));
            } else if (item.is_object()) {
                const std::string tp = item.get("type").as_str();
                if (tp == "text") {
                    segs.push_back(PromptSegment::txt(item.get("text").as_str()));
                } else if (tp == "image" || tp == "image_url" || item.has("image") || item.has("image_url")) {
                    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kImgStart));
                    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kImgPad));
                    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kImgEnd));
                }
            }
        }
    }
}

static void append_tools_prompt(std::vector<PromptSegment>& segs, const JsonValue& tools) {
    std::string header =
        "# Tools\n\nYou may call one or more functions to assist with the user query."
        "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
    bool first = true;
    for (const auto& tool : tools.arr_val) {
        const JsonValue& t = (tool.is_object() && tool.has("function")) ? tool.get("function") : tool;
        if (t.has("defer_loading") && t.get("defer_loading").as_bool(false)) continue;
        if (!first) header += "\n";
        first = false;
        header += tool_to_json_hy4(tool);
    }
    header += "\n</tools>\n\nFor function call returns, you should first print ";
    segs.push_back(PromptSegment::txt(header));
    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kToolCallsBegin));
    segs.push_back(PromptSegment::txt("\nFor each function call, you should return object like:\n"));
    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kToolCallBegin));
    segs.push_back(PromptSegment::txt("{function-name}"));
    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kArgKeyBegin));
    segs.push_back(PromptSegment::txt("{arg-key-1}"));
    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kArgKeyEnd));
    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kArgValBegin));
    segs.push_back(PromptSegment::txt("{arg-value-1}"));
    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kArgValEnd));
    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kArgKeyBegin));
    segs.push_back(PromptSegment::txt("{arg-key-2}"));
    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kArgKeyEnd));
    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kArgValBegin));
    segs.push_back(PromptSegment::txt("{arg-value-2}"));
    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kArgValEnd));
    segs.push_back(PromptSegment::txt("..."));
    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kToolCallEnd));
    segs.push_back(PromptSegment::txt("\nAt the end of function call returns, you should print "));
    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kToolCallsEnd));
}

std::vector<PromptSegment> render_hy4_chat_segments(
    const JsonValue& messages,
    const JsonValue* tools,
    const std::string& reasoning_effort_in,
    bool add_generation_prompt,
    bool preserved_thinking_override,
    bool has_preserved_override)
{
    std::vector<PromptSegment> segs;
    const std::string effort = (reasoning_effort_in == "no_think" || reasoning_effort_in == "none" || reasoning_effort_in == "low")
                               ? "no_think" : "high";
    const bool has_tools = (tools && tools->is_array() && !tools->arr_val.empty());
    const bool preserved_thinking = has_preserved_override ? preserved_thinking_override : has_tools;

    // Group consecutive tool messages per chat_template.jinja:merge_tool_responses
    struct NormMsg {
        std::string role;
        const JsonValue* single = nullptr;
        std::vector<const JsonValue*> tool_cluster;
    };
    std::vector<NormMsg> normed;
    if (messages.is_array()) {
        for (const auto& m : messages.arr_val) {
            const std::string role = m.get("role").as_str();
            if (role == "tool") {
                if (!normed.empty() && normed.back().role == "tool") {
                    normed.back().tool_cluster.push_back(&m);
                } else {
                    NormMsg nm; nm.role = "tool"; nm.tool_cluster.push_back(&m);
                    normed.push_back(std::move(nm));
                }
            } else {
                NormMsg nm; nm.role = role; nm.single = &m;
                normed.push_back(std::move(nm));
            }
        }
    }

    int last_user_idx = -1;
    const bool has_leading_system = (!normed.empty() && normed[0].role == "system");
    for (int i = 0; i < int(normed.size()); ++i) {
        if (normed[size_t(i)].role == "user") last_user_idx = i;
    }

    if (!has_leading_system) {
        segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyStart));
        segs.push_back(PromptSegment::txt("system"));
        segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyMiddle));
        if (has_tools) append_tools_prompt(segs, *tools);
        segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kReasoningMode));
        segs.push_back(PromptSegment::txt("reasoning_effort:" + effort));
        segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyEnd));
    }

    bool last_is_assistant = false;
    for (int i = 0; i < int(normed.size()); ++i) {
        const auto& nm = normed[size_t(i)];
        if (nm.role == "system") {
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyStart));
            segs.push_back(PromptSegment::txt("system"));
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyMiddle));
            if (i == 0) {
                if (has_tools) append_tools_prompt(segs, *tools);
                const std::string sc = nm.single->get("content").as_str();
                if (has_tools && !sc.empty()) segs.push_back(PromptSegment::txt("\n\n"));
                append_content_segments(segs, nm.single->get("content"));
                segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kReasoningMode));
                segs.push_back(PromptSegment::txt("reasoning_effort:" + effort));
            } else {
                append_content_segments(segs, nm.single->get("content"));
            }
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyEnd));
        } else if (nm.role == "user") {
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyStart));
            segs.push_back(PromptSegment::txt("user"));
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyMiddle));
            append_content_segments(segs, nm.single->get("content"));
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyEnd));
        } else if (nm.role == "assistant") {
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyStart));
            segs.push_back(PromptSegment::txt("assistant"));
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyMiddle));
            std::string reasoning = nm.single->get("reasoning").as_str();
            if (reasoning.empty()) reasoning = nm.single->get("reasoning_content").as_str();
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kThinkBegin));
            if (effort != "no_think" && (preserved_thinking || i > last_user_idx) && !reasoning.empty()) {
                segs.push_back(PromptSegment::txt(reasoning));
            }
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kThinkEnd));
            append_content_segments(segs, nm.single->get("content"));

            const auto& tcs = nm.single->get("tool_calls");
            if (tcs.is_array() && !tcs.arr_val.empty()) {
                segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kToolCallsBegin));
                for (const auto& tc : tcs.arr_val) {
                    const JsonValue& fn = tc.has("function") ? tc.get("function") : tc;
                    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kToolCallBegin));
                    segs.push_back(PromptSegment::txt(fn.get("name").as_str()));
                    JsonValue args_obj = fn.get("arguments");
                    if (args_obj.is_string()) {
                        JsonValue parsed;
                        if (JsonParser::try_parse(args_obj.as_str(), parsed) && parsed.is_object()) {
                            args_obj = std::move(parsed);
                        }
                    }
                    if (args_obj.is_object()) {
                        for (const auto& kv : args_obj.obj_val) {
                            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kArgKeyBegin));
                            segs.push_back(PromptSegment::txt(kv.first));
                            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kArgKeyEnd));
                            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kArgValBegin));
                            if (kv.second.is_string()) segs.push_back(PromptSegment::txt(kv.second.as_str()));
                            else segs.push_back(PromptSegment::txt(kv.second.dumps_py()));
                            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kArgValEnd));
                        }
                    }
                    segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kToolCallEnd));
                }
                segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kToolCallsEnd));
            }
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyEnd));
            if (i + 1 == int(normed.size())) last_is_assistant = true;
        } else if (nm.role == "tool") {
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyStart));
            segs.push_back(PromptSegment::txt("tool"));
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyMiddle));
            for (const JsonValue* tm : nm.tool_cluster) {
                segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kToolRespBegin));
                append_content_segments(segs, tm->get("content"));
                segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kToolRespEnd));
            }
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyEnd));
        }
    }

    if (add_generation_prompt && !last_is_assistant) {
        segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyStart));
        segs.push_back(PromptSegment::txt("assistant"));
        segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kHyMiddle));
        segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kThinkBegin));
        if (effort == "no_think") {
            segs.push_back(PromptSegment::tok(Hy4SpecialTokens::kThinkEnd));
        }
    }
    return segs;
}

// ============================================================================
// Offline Completion Parser & Incremental Token-ID-Gated Stream Parser
// ============================================================================
Hy4ParsedCompletion parse_hy4_completion(const std::string& raw_text, bool started_in_think, uint64_t call_id_seed) {
    Hy4ParsedCompletion out;
    std::string rest = raw_text;

    const std::string kTE = "</think:opensource>";
    const std::string kTB = "<think:opensource>";
    if (started_in_think) {
        const size_t pos = rest.find(kTE);
        if (pos != std::string::npos) {
            out.reasoning_content = rest.substr(0, pos);
            rest = rest.substr(pos + kTE.size());
        } else {
            out.reasoning_content = rest;
            return out;
        }
    } else {
        const size_t bpos = rest.find(kTB);
        const size_t epos = rest.find(kTE);
        if (bpos == 0 && epos != std::string::npos) {
            out.reasoning_content = rest.substr(kTB.size(), epos - kTB.size());
            rest = rest.substr(epos + kTE.size());
        }
    }

    const std::string kTCsB = "<tool_calls:opensource>";
    const std::string kTCsE = "</tool_calls:opensource>";
    const std::string kTCB  = "<tool_call:opensource>";
    const std::string kTCE  = "</tool_call:opensource>";
    const std::string kAKB  = "<arg_key:opensource>";
    const std::string kAKE  = "</arg_key:opensource>";
    const std::string kAVB  = "<arg_value:opensource>";
    const std::string kAVE  = "</arg_value:opensource>";

    const size_t tcs_pos = rest.find(kTCsB);
    if (tcs_pos == std::string::npos) {
        out.content = rest;
        return out;
    }
    out.content = rest.substr(0, tcs_pos);
    size_t scan = tcs_pos + kTCsB.size();
    const size_t end_block = rest.find(kTCsE, scan);
    const size_t limit = (end_block == std::string::npos) ? rest.size() : end_block;
    uint64_t seq = 0;

    while (scan < limit) {
        const size_t cb = rest.find(kTCB, scan);
        if (cb == std::string::npos || cb >= limit) break;
        const size_t ce = rest.find(kTCE, cb + kTCB.size());
        const size_t call_end = (ce == std::string::npos || ce > limit) ? limit : ce;
        const std::string body = rest.substr(cb + kTCB.size(), call_end - (cb + kTCB.size()));
        scan = (ce == std::string::npos) ? limit : (ce + kTCE.size());

        const size_t first_k = body.find(kAKB);
        std::string fn_name = (first_k == std::string::npos) ? body : body.substr(0, first_k);
        while (!fn_name.empty() && (fn_name.back() == ' ' || fn_name.back() == '\n' || fn_name.back() == '\r')) fn_name.pop_back();

        std::vector<std::pair<std::string, std::string>> args;
        size_t p = (first_k == std::string::npos) ? body.size() : first_k;
        while (p < body.size()) {
            const size_t kb = body.find(kAKB, p);
            if (kb == std::string::npos) break;
            const size_t ke = body.find(kAKE, kb + kAKB.size());
            if (ke == std::string::npos) break;
            const std::string key = body.substr(kb + kAKB.size(), ke - (kb + kAKB.size()));
            const size_t vb = body.find(kAVB, ke + kAKE.size());
            if (vb == std::string::npos) break;
            const size_t ve = body.find(kAVE, vb + kAVB.size());
            const std::string val = (ve == std::string::npos)
                ? body.substr(vb + kAVB.size())
                : body.substr(vb + kAVB.size(), ve - (vb + kAVB.size()));
            args.emplace_back(key, val);
            p = (ve == std::string::npos) ? body.size() : (ve + kAVE.size());
        }
        char idbuf[64];
        std::snprintf(idbuf, sizeof(idbuf), "call_hy4_%llu_%llu",
                      static_cast<unsigned long long>(call_id_seed),
                      static_cast<unsigned long long>(seq++));
        out.tool_calls.push_back({idbuf, fn_name, build_args_json(args)});
    }
    return out;
}

Hy4StreamParser::Hy4StreamParser(bool start_in_think, uint64_t call_id_seed)
    : state_(start_in_think ? State::Think : State::Content), call_id_seed_(call_id_seed) {}

std::string Hy4StreamParser::consume_complete_utf8(std::string_view bytes) {
    utf8_pending_.append(bytes.data(), bytes.size());
    size_t valid_end = 0;
    const size_t n = utf8_pending_.size();
    while (valid_end < n) {
        const uint8_t b0 = uint8_t(utf8_pending_[valid_end]);
        size_t need = 1;
        if (b0 < 0x80u) need = 1;
        else if ((b0 & 0xE0u) == 0xC0u) need = 2;
        else if ((b0 & 0xF0u) == 0xE0u) need = 3;
        else if ((b0 & 0xF8u) == 0xF0u) need = 4;
        else { ++valid_end; continue; }
        if (valid_end + need > n) break;
        valid_end += need;
    }
    std::string ready = utf8_pending_.substr(0, valid_end);
    utf8_pending_.erase(0, valid_end);
    return ready;
}

void Hy4StreamParser::feed_token(
    int32_t token_id, std::string_view raw_bytes,
    std::string& out_reasoning_delta, std::string& out_content_delta,
    std::vector<Hy4ToolStreamDelta>* out_tool_deltas)
{
    out_reasoning_delta.clear();
    out_content_delta.clear();

    if (token_id == Hy4SpecialTokens::kHyEnd || token_id == Hy4SpecialTokens::kHyEot) return;
    if (token_id == Hy4SpecialTokens::kThinkBegin) { state_ = State::Think; return; }
    if (token_id == Hy4SpecialTokens::kThinkEnd)   { state_ = State::Content; return; }
    if (token_id == Hy4SpecialTokens::kToolCallsBegin) { state_ = State::ToolCalls; return; }
    if (token_id == Hy4SpecialTokens::kToolCallsEnd)   { state_ = State::Content; return; }
    if (token_id == Hy4SpecialTokens::kToolCallBegin) {
        cur_tool_name_.clear(); cur_arg_key_.clear(); cur_arg_val_.clear(); cur_args_.clear();
        state_ = State::ToolCallName;
        return;
    }
    if (token_id == Hy4SpecialTokens::kArgKeyBegin) { cur_arg_key_.clear(); state_ = State::ToolArgKey; return; }
    if (token_id == Hy4SpecialTokens::kArgKeyEnd)   { state_ = State::ToolCalls; return; }
    if (token_id == Hy4SpecialTokens::kArgValBegin) { cur_arg_val_.clear(); state_ = State::ToolArgVal; return; }
    if (token_id == Hy4SpecialTokens::kArgValEnd) {
        cur_args_.emplace_back(cur_arg_key_, cur_arg_val_);
        cur_arg_key_.clear(); cur_arg_val_.clear();
        state_ = State::ToolCalls;
        return;
    }
    if (token_id == Hy4SpecialTokens::kToolCallEnd) {
        if (out_tool_deltas) {
            while (!cur_tool_name_.empty() && (cur_tool_name_.back() == ' ' || cur_tool_name_.back() == '\n')) cur_tool_name_.pop_back();
            char idbuf[64];
            std::snprintf(idbuf, sizeof(idbuf), "call_hy4_%llu_%zu",
                          static_cast<unsigned long long>(call_id_seed_), tool_count_);
            Hy4ToolStreamDelta d;
            d.index = tool_count_++;
            d.id = idbuf;
            d.name = cur_tool_name_;
            d.arguments_delta = build_args_json(cur_args_);
            d.is_start = true;
            d.is_end = true;
            out_tool_deltas->push_back(std::move(d));
        }
        state_ = State::ToolCalls;
        return;
    }

    const std::string s = consume_complete_utf8(raw_bytes);
    if (s.empty()) return;
    switch (state_) {
        case State::Think:        out_reasoning_delta = s; break;
        case State::Content:      out_content_delta = s; break;
        case State::ToolCallName: cur_tool_name_ += s; break;
        case State::ToolArgKey:   cur_arg_key_ += s; break;
        case State::ToolArgVal:   cur_arg_val_ += s; break;
        default: break;
    }
}

void Hy4StreamParser::flush(
    std::string& out_reasoning_delta, std::string& out_content_delta,
    std::vector<Hy4ToolStreamDelta>* /*out_tool_deltas*/)
{
    out_reasoning_delta.clear();
    out_content_delta.clear();
    if (!utf8_pending_.empty()) {
        std::string s = bytes_to_utf8_lossy(utf8_pending_);
        utf8_pending_.clear();
        if (state_ == State::Think) out_reasoning_delta = std::move(s);
        else if (state_ == State::Content) out_content_delta = std::move(s);
    }
}

}  // namespace hy4
