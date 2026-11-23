#pragma once
// Pure-C++17 ByteLevel BPE Tokenizer, Official Hy4-flash Chat Template Renderer & Tool-Call Streaming Parser
// Model: tencent/Hy4-flash (vocab=120,832, <｜hy_start:opensource｜>..120125)
// Zero Python anywhere. Links against system libpcre2-8.

#include "hy4_json.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hy4 {

struct Hy4SpecialTokens {
    static constexpr int32_t kHyStart         = 120000;  // <｜hy_start:opensource｜>
    static constexpr int32_t kHyMiddle        = 120001;  // <｜hy_middle:opensource｜>
    static constexpr int32_t kHyPad           = 120002;  // <｜hy_pad:opensource｜>
    static constexpr int32_t kHyEot           = 120008;  // <｜hy_EOT｜>
    static constexpr int32_t kDFly2Mask       = 120009;  // <｜hy_place▁holder▁no▁120009｜>
    static constexpr int32_t kHyEnd           = 120025;  // <｜hy_end:opensource｜>
    static constexpr int32_t kThinkBegin      = 120029;  // <think:opensource>
    static constexpr int32_t kThinkEnd        = 120030;  // </think:opensource>
    static constexpr int32_t kToolCallBegin   = 120033;  // <tool_call:opensource>
    static constexpr int32_t kToolCallEnd     = 120034;  // </tool_call:opensource>
    static constexpr int32_t kToolCallsBegin  = 120035;  // <tool_calls:opensource>
    static constexpr int32_t kToolCallsEnd    = 120036;  // </tool_calls:opensource>
    static constexpr int32_t kToolRespBegin   = 120037;  // <tool_response:opensource>
    static constexpr int32_t kToolRespEnd     = 120038;  // </tool_response:opensource>
    static constexpr int32_t kReasoningMode   = 120039;  // <｜reasoning_mode:opensource｜>
    static constexpr int32_t kArgKeyBegin     = 120040;  // <arg_key:opensource>
    static constexpr int32_t kArgKeyEnd       = 120041;  // </arg_key:opensource>
    static constexpr int32_t kArgValBegin     = 120042;  // <arg_value:opensource>
    static constexpr int32_t kArgValEnd       = 120043;  // </arg_value:opensource>
    static constexpr int32_t kImgStart        = 120118;  // <｜hy_image▁start:opensource｜>
    static constexpr int32_t kImgEnd          = 120119;  // <｜hy_image▁end:opensource｜>
    static constexpr int32_t kImgPad          = 120120;  // <｜hy_image▁pad:opensource｜>
    static constexpr int32_t kImgNewline      = 120121;  // <｜hy_image▁newline:opensource｜>
};

struct PromptSegment {
    enum class Kind : uint8_t { OrdinaryText = 0, TokenId = 1 };
    Kind kind = Kind::OrdinaryText;
    int32_t token_id = -1;
    int32_t repeat = 1;
    std::string text;

    static PromptSegment txt(std::string s) {
        PromptSegment seg;
        seg.kind = Kind::OrdinaryText;
        seg.text = std::move(s);
        return seg;
    }
    static PromptSegment tok(int32_t id, int32_t count = 1) {
        PromptSegment seg;
        seg.kind = Kind::TokenId;
        seg.token_id = id;
        seg.repeat = count;
        return seg;
    }
};

struct Hy4ParsedToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

struct Hy4ParsedCompletion {
    std::string reasoning_content;
    std::string content;
    std::vector<Hy4ParsedToolCall> tool_calls;
};

struct Hy4ToolStreamDelta {
    size_t index = 0;
    std::string id;
    std::string name;
    std::string arguments_delta;
    bool is_start = false;
    bool is_end = false;
};

class Hy4Tokenizer {
public:
    Hy4Tokenizer();
    ~Hy4Tokenizer();

    Hy4Tokenizer(const Hy4Tokenizer&) = delete;
    Hy4Tokenizer& operator=(const Hy4Tokenizer&) = delete;

    void load(const std::string& tokenizer_json_path);
    size_t vocab_size() const noexcept { return id_to_token_.size(); }

    std::vector<int32_t> encode(const std::string& text, bool parse_special = true) const;
    std::vector<int32_t> encode_segments(const std::vector<PromptSegment>& segments) const;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens = false) const;
    std::string decode_single(int32_t id, bool skip_special_tokens = false) const;
    std::string_view raw_bytes_single(int32_t id, bool skip_special_tokens = false) const;
    const std::string& id_to_token(int32_t id) const;
    int32_t token_to_id(const std::string& tok) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<std::string> id_to_token_;
    std::vector<std::string> id_to_raw_bytes_;
    std::vector<uint8_t> id_is_special_;
    std::unordered_map<std::string, int32_t> token_to_id_;
};

// Renders OpenAI/Anthropic chat messages + tools + reasoning_effort ('high' or 'no_think')
// faithfully matching tencent/Hy4-flash/chat_template.jinja using anti-injection TokenId segments.
std::vector<PromptSegment> render_hy4_chat_segments(
    const JsonValue& messages,
    const JsonValue* tools,
    const std::string& reasoning_effort,
    bool add_generation_prompt = true,
    bool preserved_thinking_override = false,
    bool has_preserved_override = false);

// Parses raw assistant completion text (after prompt's opening <think:opensource> if high effort)
Hy4ParsedCompletion parse_hy4_completion(
    const std::string& raw_text,
    bool started_in_think,
    uint64_t call_id_seed = 1);

// Token-ID-gated streaming SSE parser for Hy4-flash (<think:opensource>, </think:opensource>, <tool_calls:opensource>...)
class Hy4StreamParser {
public:
    explicit Hy4StreamParser(bool start_in_think, uint64_t call_id_seed = 1);

    void feed_token(
        int32_t token_id,
        std::string_view raw_bytes,
        std::string& out_reasoning_delta,
        std::string& out_content_delta,
        std::vector<Hy4ToolStreamDelta>* out_tool_deltas = nullptr);

    void flush(
        std::string& out_reasoning_delta,
        std::string& out_content_delta,
        std::vector<Hy4ToolStreamDelta>* out_tool_deltas = nullptr);

    bool in_think() const noexcept { return state_ == State::Think; }
    bool in_tool_calls() const noexcept { return state_ == State::ToolCalls || state_ == State::ToolCallName || state_ == State::ToolArgKey || state_ == State::ToolArgVal; }

private:
    enum class State : uint8_t {
        Think = 0,
        Content = 1,
        ToolCalls = 2,
        ToolCallName = 3,
        ToolArgKey = 4,
        ToolArgVal = 5
    };
    State state_;
    uint64_t call_id_seed_;
    size_t tool_count_ = 0;
    std::string utf8_pending_;
    std::string cur_tool_name_;
    std::string cur_arg_key_;
    std::string cur_arg_val_;
    std::vector<std::pair<std::string, std::string>> cur_args_;

    std::string consume_complete_utf8(std::string_view bytes);
};

}  // namespace hy4
