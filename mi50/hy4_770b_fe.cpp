// Pure-C++17 OpenAI & Anthropic Compatible HTTP/SSE Frontend for tencent/Hy4 (770B) on SB-M5-P2-192G
// Default port: 127.0.0.1:8050 (DeepSeek Harness :3081 provider: Hy4-770B-WHT128-192G)
// Endpoints:
//   GET  /health, /v1/health, /v1/models, /v1/metrics (live 4-tier routing telemetry)
//   POST /v1/chat/completions, /v1/messages, /v1/completions
//   POST /v1/clear_all_caches, /v1/kv/save (/v1/kv_save), /v1/kv/restore
//   POST /v1/shutdown (strict graceful teardown draining 5x MI50 hipDeviceSynchronize() + P150A SHM!)
// Zero Python anywhere.

#include "hy4_770b_nvme_kv_cache.hpp"
#include "hy4_770b_runtime.hpp"
#include "hy4_json.hpp"
#include "hy4_tokenizer.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace hy4_770b {
namespace {

using hy4::JsonParser;
using hy4::JsonValue;

bool send_all(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        sent += size_t(n);
    }
    return true;
}

bool send_str(int fd, const std::string& s) { return send_all(fd, s.data(), s.size()); }

void send_http_json(int fd, int status, const char* reason, const std::string& body) {
    std::string hdr =
        "HTTP/1.1 " + std::to_string(status) + " " + reason + "\r\n"
        "Content-Type: application/json\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n\r\n";
    (void)send_str(fd, hdr);
    (void)send_str(fd, body);
}

void send_sse_headers(int fd) {
    const std::string hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n\r\n";
    (void)send_str(fd, hdr);
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

bool read_http_request(int fd, HttpRequest& req) {
    std::string buf;
    buf.reserve(8192);
    char tmp[4096];
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf.append(tmp, size_t(n));
        header_end = buf.find("\r\n\r\n");
        if (buf.size() > (256u << 20)) return false;
    }
    const std::string head = buf.substr(0, header_end);
    size_t sp1 = head.find(' ');
    if (sp1 == std::string::npos) return false;
    size_t sp2 = head.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) return false;
    req.method = head.substr(0, sp1);
    req.path   = head.substr(sp1 + 1, sp2 - sp1 - 1);
    const size_t qm = req.path.find('?');
    if (qm != std::string::npos) req.path.resize(qm);

    size_t content_len = 0;
    std::istringstream iss(head);
    std::string line;
    std::getline(iss, line);
    while (std::getline(iss, line)) {
        std::string low = line;
        for (char& c : low) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        if (low.rfind("content-length:", 0) == 0) {
            content_len = size_t(std::strtoull(low.c_str() + 15, nullptr, 10));
        }
    }
    req.body = buf.substr(header_end + 4);
    while (req.body.size() < content_len) {
        ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        req.body.append(tmp, size_t(n));
    }
    if (req.body.size() > content_len) req.body.resize(content_len);
    return true;
}

class Hy4770bServer {
public:
    Hy4770bRuntime rt;
    std::mutex rt_mu;
    std::string model_id = "Hy4-770B-WHT128-192G";
    std::atomic<uint64_t> req_counter{1};
    std::atomic<int> active_clients{0};

    struct ClientGuard {
        std::atomic<int>& cnt;
        explicit ClientGuard(std::atomic<int>& c) : cnt(c) { cnt.fetch_add(1, std::memory_order_acq_rel); }
        ~ClientGuard() { cnt.fetch_sub(1, std::memory_order_acq_rel); }
    };

    JsonValue build_metrics_json() const {
        const auto& mr = rt.memory_report();
        const auto ts  = rt.cumulative_tier_stats();
        JsonValue j = JsonValue::make_object();
        j.set("status", should_stop() ? "shutting_down" : "ok");
        j.set("model", model_id);
        j.set("profile", profile_name(rt.config().profile));
        j.set("dfly2_active", mr.dfly2_active);
        j.set("p150_active", mr.p150_active);
        j.set("p150_l1_hot_experts", mr.p150_l1_hot_experts);
        j.set("p150_l1_mib_total", mr.p150_l1_mib_total);
        j.set("p150_gddr6_gib_per_card", mr.p150_used_gib[0]);
        j.set("ddr4_cold_gib", mr.ddr4_cold_gib);
        j.set("kv_vram_gib", mr.kv_vram_gib);
        j.set("kv_len", rt.current_kv_len());
        j.set("max_kv_len", rt.config().max_kv_len);
        JsonValue vram = JsonValue::make_array();
        for (int d = 0; d < kNumMi50Ranks; ++d) vram.push(mr.hip_used_gib[d]);
        j.set("mi50_vram_gib", std::move(vram));

        JsonValue tiers = JsonValue::make_object();
        tiers.set("total_routes", int64_t(ts.total_routes));
        tiers.set("tensix_l1_hits", int64_t(ts.tensix_l1_hits));
        tiers.set("mi50_hbm2_hits", int64_t(ts.mi50_hbm2_hits));
        tiers.set("p150_gddr6_hits", int64_t(ts.p150_gddr6_hits));
        tiers.set("epyc_ddr4_hits", int64_t(ts.epyc_ddr4_hits));
        tiers.set("tensix_l1_pct", ts.tensix_l1_pct());
        tiers.set("mi50_hbm2_pct", ts.mi50_hbm2_pct());
        tiers.set("p150_gddr6_pct", ts.p150_gddr6_pct());
        tiers.set("epyc_ddr4_pct", ts.epyc_ddr4_pct());
        tiers.set("mi50_ms_total", ts.mi50_ms_total);
        tiers.set("p150_ms_total", ts.p150_ms_total);
        tiers.set("ddr4_ms_total", ts.ddr4_ms_total);
        j.set("tier_routing", std::move(tiers));
        return j;
    }

    void handle_client(int fd) {
        ClientGuard cg(active_clients);
        int one = 1;
        (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        struct timeval tv{600, 0};
        (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        HttpRequest req;
        if (!read_http_request(fd, req)) { ::close(fd); return; }

        if (req.method == "OPTIONS") {
            send_http_json(fd, 200, "OK", "{}");
            ::close(fd);
            return;
        }

        if (req.path == "/v1/shutdown" && req.method == "POST") {
            request_graceful_stop();
            std::printf("[HY4-770B-FE] POST /v1/shutdown received — draining active requests & 5x MI50 + 2x P150A...\n");
            std::fflush(stdout);
            send_http_json(fd, 200, "OK", "{\"status\":\"shutting_down\"}");
            ::close(fd);
            return;
        }

        if (req.path == "/health" || req.path == "/v1/health" || req.path == "/v1/metrics") {
            auto j = build_metrics_json();
            send_http_json(fd, should_stop() ? 503 : 200, should_stop() ? "Service Unavailable" : "OK", j.dumps_compact());
            ::close(fd);
            return;
        }

        if (req.path == "/v1/models") {
            JsonValue m = JsonValue::make_object();
            m.set("id", model_id);
            m.set("object", "model");
            m.set("created", int64_t(std::time(nullptr)));
            m.set("owned_by", "bespoke-ontology-hy4-770b-cpp");
            JsonValue data = JsonValue::make_array();
            data.push(std::move(m));
            JsonValue resp = JsonValue::make_object();
            resp.set("object", "list");
            resp.set("data", std::move(data));
            send_http_json(fd, 200, "OK", resp.dumps_compact());
            ::close(fd);
            return;
        }

        if (req.path == "/v1/clear_all_caches") {
            if (should_stop()) { send_http_json(fd, 503, "Service Unavailable", "{\"error\":\"shutting_down\"}"); ::close(fd); return; }
            std::lock_guard<std::mutex> lk(rt_mu);
            rt.clear_kv_cache();
            send_http_json(fd, 200, "OK", "{\"status\":\"cleared\"}");
            ::close(fd);
            return;
        }

        if (req.path == "/v1/kv/save" || req.path == "/v1/kv_save") {
            if (should_stop()) { send_http_json(fd, 503, "Service Unavailable", "{\"error\":\"shutting_down\"}"); ::close(fd); return; }
            std::lock_guard<std::mutex> lk(rt_mu);
            const auto& toks = rt.cached_tokens();
            const int n_tok = std::min<int>(rt.current_kv_len(), int(toks.size()));
            if (n_tok <= 0) {
                send_http_json(fd, 400, "Bad Request", "{\"error\":\"no_active_kv\"}");
                ::close(fd);
                return;
            }
            const uint64_t phash = hy4_770b_prefix_hash64(toks.data(), size_t(n_tok));
            const auto t0 = std::chrono::steady_clock::now();
            const bool ok = rt.save_kv_to_nvme(phash);
            const double save_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ok) {
                const std::string idx_path = rt.config().kv_dir0 + "/hy4_770b_last_kv_index.bin";
                std::ofstream ofs(idx_path, std::ios::binary | std::ios::trunc);
                if (ofs) {
                    const uint64_t cnt = uint64_t(n_tok);
                    ofs.write(reinterpret_cast<const char*>(&phash), 8);
                    ofs.write(reinterpret_cast<const char*>(&cnt), 8);
                    ofs.write(reinterpret_cast<const char*>(toks.data()), std::streamsize(size_t(n_tok) * 4u));
                }
            }
            JsonValue resp = JsonValue::make_object();
            resp.set("status", ok ? "saved" : "failed");
            resp.set("kv_len", n_tok);
            resp.set("prefix_hash", std::to_string(phash));
            resp.set("save_ms", save_ms);
            std::printf("[HY4-770B-FE] KV_SAVE ok=%d kv_len=%d hash=%llu save_ms=%.1f\n",
                        ok ? 1 : 0, n_tok, (unsigned long long)phash, save_ms);
            std::fflush(stdout);
            send_http_json(fd, ok ? 200 : 500, ok ? "OK" : "Error", resp.dumps_compact());
            ::close(fd);
            return;
        }

        if (req.path == "/v1/kv/restore") {
            if (should_stop()) { send_http_json(fd, 503, "Service Unavailable", "{\"error\":\"shutting_down\"}"); ::close(fd); return; }
            std::lock_guard<std::mutex> lk(rt_mu);
            const std::string idx_path = rt.config().kv_dir0 + "/hy4_770b_last_kv_index.bin";
            std::ifstream ifs(idx_path, std::ios::binary);
            uint64_t phash = 0, cnt = 0;
            bool ok = false;
            Hy4770bNvmeKvCache::ReloadStats st{};
            if (ifs && ifs.read(reinterpret_cast<char*>(&phash), 8) && ifs.read(reinterpret_cast<char*>(&cnt), 8) && cnt > 0) {
                std::vector<int32_t> toks(static_cast<size_t>(cnt));
                if (ifs.read(reinterpret_cast<char*>(toks.data()), std::streamsize(cnt * 4ULL))) {
                    ok = rt.load_kv_from_nvme(phash, toks, true, &st);
                }
            }
            JsonValue resp = JsonValue::make_object();
            resp.set("status", ok ? "restored" : "not_found");
            resp.set("kv_len", int64_t(cnt));
            resp.set("file_bytes", int64_t(st.file_bytes));
            resp.set("disk_read_ms", st.disk_read_ms);
            resp.set("crc32c_ms", st.crc32c_ms);
            resp.set("h2d_ms", st.h2d_ms);
            resp.set("total_ms", st.total_ms);
            resp.set("eff_gibs", st.eff_gibs);
            resp.set("tok_per_sec", st.tok_per_sec);
            send_http_json(fd, ok ? 200 : 404, ok ? "OK" : "Not Found", resp.dumps_compact());
            ::close(fd);
            return;
        }

        if (req.path == "/v1/chat/completions" || req.path == "/v1/messages") {
            if (should_stop()) { send_http_json(fd, 503, "Service Unavailable", "{\"error\":\"shutting_down\"}"); ::close(fd); return; }
            handle_chat(fd, req.body, req.path == "/v1/messages");
            ::close(fd);
            return;
        }

        if (req.path == "/v1/completions") {
            if (should_stop()) { send_http_json(fd, 503, "Service Unavailable", "{\"error\":\"shutting_down\"}"); ::close(fd); return; }
            handle_completion(fd, req.body);
            ::close(fd);
            return;
        }

        send_http_json(fd, 404, "Not Found", "{\"error\":\"not_found\"}");
        ::close(fd);
    }

private:
    void try_auto_restore_kv(const std::vector<int32_t>& prompt_ids) {
        if (rt.current_kv_len() != 0 || !rt.config().enable_kv_disk_cache) return;
        const std::string idx_path = rt.config().kv_dir0 + "/hy4_770b_last_kv_index.bin";
        std::ifstream ifs(idx_path, std::ios::binary);
        uint64_t saved_hash = 0ULL, saved_cnt = 0ULL;
        if (!ifs || !ifs.read(reinterpret_cast<char*>(&saved_hash), 8) ||
            !ifs.read(reinterpret_cast<char*>(&saved_cnt), 8) ||
            saved_cnt == 0ULL || saved_cnt > uint64_t(prompt_ids.size())) return;
        std::vector<int32_t> saved_toks(static_cast<size_t>(saved_cnt));
        if (!ifs.read(reinterpret_cast<char*>(saved_toks.data()), std::streamsize(saved_cnt * 4ULL))) return;
        if (std::memcmp(saved_toks.data(), prompt_ids.data(), size_t(saved_cnt) * 4u) != 0) return;
        Hy4770bNvmeKvCache::ReloadStats st{};
        if (rt.load_kv_from_nvme(saved_hash, saved_toks, true, &st)) {
            std::printf("[HY4-770B-FE] COLD .kvblk RESTORE kv_len=%llu bytes=%llu read=%.1fms crc=%.1fms h2d=%.1fms total=%.1fms (%.2f GiB/s, %.0f tok/s E2E)\n",
                        (unsigned long long)saved_cnt, (unsigned long long)st.file_bytes,
                        st.disk_read_ms, st.crc32c_ms, st.h2d_ms, st.total_ms, st.eff_gibs, st.tok_per_sec);
            std::fflush(stdout);
        }
    }

    void handle_completion(int fd, const std::string& body) {
        JsonValue jreq;
        if (!JsonParser::try_parse(body, jreq) || !jreq.is_object()) {
            send_http_json(fd, 400, "Bad Request", "{\"error\":\"invalid_json\"}");
            return;
        }
        const std::string prompt = jreq.get("prompt").as_str();
        Hy4770bSamplingParams sp;
        sp.max_new_tokens = std::clamp(int(jreq.get("max_tokens").as_int(256)), 1, 16384);
        sp.temperature    = float(jreq.get("temperature").as_double(0.0));
        sp.top_p          = float(jreq.get("top_p").as_double(0.95));
        sp.top_k          = int(jreq.get("top_k").as_int(40));
        sp.greedy         = (sp.temperature <= 1e-4f);
        sp.use_dfly2      = rt.config().use_dfly2 && (rt.config().dfly2_num_drafts > 0);
        if (jreq.has("use_dfly2")) sp.use_dfly2 = jreq.get("use_dfly2").as_bool(sp.use_dfly2);
        if (jreq.has("dfly2_num_drafts")) sp.dfly2_num_drafts = int(jreq.get("dfly2_num_drafts").as_int(0));
        const uint64_t rid = req_counter.fetch_add(1, std::memory_order_relaxed);
        sp.seed = uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()) ^ (rid * 0x9E3779B97F4A7C15ULL);

        std::lock_guard<std::mutex> lk(rt_mu);
        auto prompt_ids = rt.tokenizer().encode(prompt);
        if (prompt_ids.empty()) prompt_ids.push_back(kBosTokenId);
        try_auto_restore_kv(prompt_ids);

        Hy4770bGenMetrics met{};
        auto gen_ids = rt.generate(prompt_ids, sp, &met, nullptr, nullptr);
        std::string text = rt.tokenizer().decode(gen_ids, true);

        JsonValue ch = JsonValue::make_object();
        ch.set("index", 0);
        ch.set("text", text);
        ch.set("finish_reason", met.finish_reason);
        JsonValue choices = JsonValue::make_array();
        choices.push(std::move(ch));
        JsonValue usage = JsonValue::make_object();
        usage.set("prompt_tokens", met.prompt_tokens);
        usage.set("completion_tokens", met.generated_tokens);
        usage.set("total_tokens", met.prompt_tokens + met.generated_tokens);
        JsonValue resp = JsonValue::make_object();
        resp.set("id", "cmpl-hy4-770b-" + std::to_string(rid));
        resp.set("object", "text_completion");
        resp.set("created", int64_t(std::time(nullptr)));
        resp.set("model", model_id);
        resp.set("choices", std::move(choices));
        resp.set("usage", std::move(usage));
        send_http_json(fd, 200, "OK", resp.dumps_compact());
    }

    void handle_chat(int fd, const std::string& body, bool is_anthropic) {
        JsonValue jreq;
        if (!JsonParser::try_parse(body, jreq) || !jreq.is_object()) {
            send_http_json(fd, 400, "Bad Request", "{\"error\":\"invalid_json\"}");
            return;
        }
        JsonValue messages = jreq.get("messages");
        if (is_anthropic && jreq.has("system")) {
            JsonValue wrapped = JsonValue::make_array();
            JsonValue sm = JsonValue::make_object();
            sm.set("role", "system");
            sm.set("content", jreq.get("system"));
            wrapped.push(std::move(sm));
            if (messages.is_array()) for (const auto& m : messages.arr_val) wrapped.push(m);
            messages = std::move(wrapped);
        }

        const JsonValue* tools = jreq.find("tools");
        std::string reasoning_effort = jreq.get("reasoning_effort").as_str("high");
        if (jreq.has("chat_template_kwargs") && jreq.get("chat_template_kwargs").has("reasoning_effort")) {
            reasoning_effort = jreq.get("chat_template_kwargs").get("reasoning_effort").as_str(reasoning_effort);
        }
        const bool stream = jreq.get("stream").as_bool(false);

        Hy4770bSamplingParams sp;
        sp.max_new_tokens = std::clamp(int(jreq.get("max_tokens").as_int(jreq.get("max_completion_tokens").as_int(2048))), 1, 16384);
        sp.temperature    = float(jreq.get("temperature").as_double(0.6));
        sp.top_p          = float(jreq.get("top_p").as_double(0.95));
        sp.top_k          = int(jreq.get("top_k").as_int(40));
        sp.greedy         = (sp.temperature <= 1e-4f);
        sp.use_dfly2      = rt.config().use_dfly2 && (rt.config().dfly2_num_drafts > 0);
        if (jreq.has("use_dfly2")) sp.use_dfly2 = jreq.get("use_dfly2").as_bool(sp.use_dfly2);
        if (jreq.has("dfly2_num_drafts")) sp.dfly2_num_drafts = int(jreq.get("dfly2_num_drafts").as_int(0));
        const uint64_t rid = req_counter.fetch_add(1, std::memory_order_relaxed);
        sp.seed = uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()) ^ (rid * 0x9E3779B97F4A7C15ULL);

        // Extract multimodal images
        std::vector<std::string> image_sources;
        if (messages.is_array()) {
            for (const auto& msg : messages.arr_val) {
                const auto& content = msg.get("content");
                if (!content.is_array()) continue;
                for (const auto& item : content.arr_val) {
                    if (!item.is_object()) continue;
                    const std::string t = item.get("type").as_str();
                    if (t == "image_url" && item.has("image_url")) {
                        const auto& iu = item.get("image_url");
                        image_sources.push_back(iu.is_object() ? iu.get("url").as_str() : iu.as_str());
                    } else if (t == "image") {
                        if (item.has("source") && item.get("source").is_object()) {
                            const auto& src = item.get("source");
                            image_sources.push_back("data:" + src.get("media_type").as_str("image/png") + ";base64," + src.get("data").as_str());
                        } else if (item.has("image")) image_sources.push_back(item.get("image").as_str());
                        else if (item.has("url"))     image_sources.push_back(item.get("url").as_str());
                    }
                }
            }
        }

        const auto segs = hy4::render_hy4_chat_segments(messages, tools, reasoning_effort, true);
        const bool started_in_think = (reasoning_effort != "no_think" && reasoning_effort != "none" && reasoning_effort != "low");
        const std::string req_id = "chatcmpl-hy4-770b-" + std::to_string(rid);
        const int64_t created = int64_t(std::time(nullptr));

        std::lock_guard<std::mutex> lk(rt_mu);
        std::vector<Hy4VitOutput> encoded_imgs;
        std::vector<Hy4770bVisualSpan> visual_spans;
        std::vector<int32_t> prompt_ids;
        if (!image_sources.empty() && rt.vit_ready()) {
            encoded_imgs.reserve(image_sources.size());
            for (const auto& src : image_sources) {
                Hy4ImageRgb rgb{};
                std::string ierr;
                if (!load_image_from_url_or_path(src, rgb, &ierr)) {
                    send_http_json(fd, 400, "Bad Request", "{\"error\":\"image decode failed: " + ierr + "\"}");
                    return;
                }
                encoded_imgs.push_back(rt.encode_image(rgb));
                const auto& vo = encoded_imgs.back();
                std::printf("[HY4-770B-VIT] Encoded %dx%d -> grid=%dx%d merged=%dx%d (%d visual toks) in %.1f ms\n",
                            rgb.width, rgb.height, vo.grid_h, vo.grid_w, vo.merged_h, vo.merged_w, vo.num_tokens, vo.encode_ms);
            }
            size_t img_idx = 0;
            for (const auto& seg : segs) {
                if (seg.kind == hy4::PromptSegment::Kind::TokenId && seg.token_id == kImageTokenId && img_idx < encoded_imgs.size()) {
                    const auto& vo = encoded_imgs[img_idx++];
                    visual_spans.push_back({int(prompt_ids.size()), vo.num_tokens, vo.embeddings_bf16.data()});
                    prompt_ids.insert(prompt_ids.end(), size_t(vo.num_tokens), kImageTokenId);
                } else if (seg.kind == hy4::PromptSegment::Kind::TokenId) {
                    prompt_ids.insert(prompt_ids.end(), size_t(seg.repeat), seg.token_id);
                } else {
                    const auto ids = rt.tokenizer().encode(seg.text);
                    prompt_ids.insert(prompt_ids.end(), ids.begin(), ids.end());
                }
            }
        } else {
            prompt_ids = rt.tokenizer().encode_segments(segs);
        }
        const std::vector<Hy4770bVisualSpan>* vspan_ptr = visual_spans.empty() ? nullptr : &visual_spans;
        try_auto_restore_kv(prompt_ids);

        Hy4770bGenMetrics met{};
        auto build_timings = [&]() {
            JsonValue timings = JsonValue::make_object();
            timings.set("prefill_ms", met.prefill_ms);
            timings.set("ttft_ms", met.ttft_ms);
            timings.set("decode_ms", met.decode_ms);
            timings.set("prefill_tok_s", met.prefill_tok_s);
            timings.set("decode_tok_s", met.decode_tok_s);
            timings.set("cached_prefix_tokens", met.cached_prefix_tokens);
            timings.set("dfly2_accept_rate", met.dfly2_accept_rate);
            timings.set("dfly2_mean_tokens_per_step", met.dfly2_mean_tokens_per_step);
            timings.set("dfly2_verify_step_ms_mean", met.dfly2_verify_step_ms_mean);
            timings.set("tensix_l1_pct", met.tier_delta.tensix_l1_pct());
            timings.set("mi50_hbm2_pct", met.tier_delta.mi50_hbm2_pct());
            timings.set("p150_gddr6_pct", met.tier_delta.p150_gddr6_pct());
            timings.set("epyc_ddr4_pct", met.tier_delta.epyc_ddr4_pct());
            return timings;
        };

        if (stream) {
            send_sse_headers(fd);
            hy4::Hy4StreamParser parser(started_in_think, rid);
            bool client_ok = true;
            std::vector<hy4::Hy4ToolStreamDelta> seen_tools;

            auto emit_chunk = [&](const std::string& rd, const std::string& cd,
                                  const std::vector<hy4::Hy4ToolStreamDelta>& tds, const char* finish) {
                if (!client_ok) return;
                JsonValue delta = JsonValue::make_object();
                if (!rd.empty()) delta.set("reasoning_content", rd);
                if (!cd.empty()) delta.set("content", cd);
                if (!tds.empty()) {
                    JsonValue tca = JsonValue::make_array();
                    for (const auto& td : tds) {
                        JsonValue fn = JsonValue::make_object();
                        fn.set("name", td.name);
                        fn.set("arguments", td.arguments_delta);
                        JsonValue tc = JsonValue::make_object();
                        tc.set("index", int64_t(td.index));
                        tc.set("id", td.id);
                        tc.set("type", "function");
                        tc.set("function", std::move(fn));
                        tca.push(std::move(tc));
                    }
                    delta.set("tool_calls", std::move(tca));
                }
                JsonValue ch = JsonValue::make_object();
                ch.set("index", 0);
                ch.set("delta", std::move(delta));
                if (finish) ch.set("finish_reason", finish);
                else ch.set("finish_reason", JsonValue::make_null());
                JsonValue choices = JsonValue::make_array();
                choices.push(std::move(ch));
                JsonValue ev = JsonValue::make_object();
                ev.set("id", req_id);
                ev.set("object", "chat.completion.chunk");
                ev.set("created", created);
                ev.set("model", model_id);
                ev.set("choices", std::move(choices));
                client_ok = send_str(fd, "data: " + ev.dumps_compact() + "\n\n");
            };

            (void)rt.generate(prompt_ids, sp, &met, [&](int32_t tid, int /*step*/) -> bool {
                std::string rd, cd;
                std::vector<hy4::Hy4ToolStreamDelta> tds;
                parser.feed_token(tid, rt.tokenizer().raw_bytes_single(tid, false), rd, cd, &tds);
                for (const auto& t : tds) seen_tools.push_back(t);
                if (!rd.empty() || !cd.empty() || !tds.empty()) emit_chunk(rd, cd, tds, nullptr);
                return client_ok && !should_stop();
            }, vspan_ptr);

            std::string rd, cd;
            std::vector<hy4::Hy4ToolStreamDelta> tds;
            parser.flush(rd, cd, &tds);
            if (!rd.empty() || !cd.empty()) emit_chunk(rd, cd, tds, nullptr);
            emit_chunk("", "", {}, !seen_tools.empty() ? "tool_calls" : met.finish_reason.c_str());
            if (client_ok) {
                JsonValue pdet = JsonValue::make_object();
                pdet.set("cached_tokens", met.cached_prefix_tokens);
                JsonValue usage = JsonValue::make_object();
                usage.set("prompt_tokens", met.prompt_tokens);
                usage.set("completion_tokens", met.generated_tokens);
                usage.set("total_tokens", met.prompt_tokens + met.generated_tokens);
                usage.set("prompt_tokens_details", std::move(pdet));
                JsonValue uev = JsonValue::make_object();
                uev.set("id", req_id);
                uev.set("object", "chat.completion.chunk");
                uev.set("created", created);
                uev.set("model", model_id);
                uev.set("choices", JsonValue::make_array());
                uev.set("usage", std::move(usage));
                uev.set("timings", build_timings());
                (void)send_str(fd, "data: " + uev.dumps_compact() + "\n\n");
                (void)send_str(fd, "data: [DONE]\n\n");
            }
        } else {
            const auto gen_ids = rt.generate(prompt_ids, sp, &met, nullptr, vspan_ptr);
            const std::string raw_out = rt.tokenizer().decode(gen_ids, false);
            const auto parsed = hy4::parse_hy4_completion(raw_out, started_in_think, rid);

            JsonValue msg = JsonValue::make_object();
            msg.set("role", "assistant");
            msg.set("content", parsed.content);
            if (!parsed.reasoning_content.empty()) msg.set("reasoning_content", parsed.reasoning_content);
            if (!parsed.tool_calls.empty()) {
                JsonValue tca = JsonValue::make_array();
                for (const auto& tc : parsed.tool_calls) {
                    JsonValue fn = JsonValue::make_object();
                    fn.set("name", tc.name);
                    fn.set("arguments", tc.arguments_json);
                    JsonValue item = JsonValue::make_object();
                    item.set("id", tc.id);
                    item.set("type", "function");
                    item.set("function", std::move(fn));
                    tca.push(std::move(item));
                }
                msg.set("tool_calls", std::move(tca));
            }
            JsonValue ch = JsonValue::make_object();
            ch.set("index", 0);
            ch.set("message", std::move(msg));
            ch.set("finish_reason", !parsed.tool_calls.empty() ? "tool_calls" : met.finish_reason);
            JsonValue choices = JsonValue::make_array();
            choices.push(std::move(ch));
            JsonValue usage = JsonValue::make_object();
            usage.set("prompt_tokens", met.prompt_tokens);
            usage.set("completion_tokens", met.generated_tokens);
            usage.set("total_tokens", met.prompt_tokens + met.generated_tokens);

            JsonValue resp = JsonValue::make_object();
            resp.set("id", req_id);
            resp.set("object", "chat.completion");
            resp.set("created", created);
            resp.set("model", model_id);
            resp.set("choices", std::move(choices));
            resp.set("usage", std::move(usage));
            resp.set("timings", build_timings());
            send_http_json(fd, 200, "OK", resp.dumps_compact());
        }

        std::printf("[HY4-770B-FE] req=%llu prompt=%d cached=%d gen=%d ttft=%.1fms decode=%.2f tok/s | dfly2=%.2f tok/step (%.1f ms) | "
                    "Tiers: L1=%.1f%% MI50=%.1f%% P150=%.1f%% DDR4=%.1f%%\n",
                    (unsigned long long)rid, met.prompt_tokens, met.cached_prefix_tokens, met.generated_tokens, met.ttft_ms, met.decode_tok_s,
                    met.dfly2_mean_tokens_per_step, met.dfly2_verify_step_ms_mean,
                    met.tier_delta.tensix_l1_pct(), met.tier_delta.mi50_hbm2_pct(),
                    met.tier_delta.p150_gddr6_pct(), met.tier_delta.epyc_ddr4_pct());
        std::fflush(stdout);
    }
};

}  // namespace
}  // namespace hy4_770b

int main(int argc, char** argv) {
    using namespace hy4_770b;
    install_graceful_signal_handlers();

    Hy4770bRuntimeConfig cfg;
    cfg.shutdown_p150_on_exit = false;  // Keep warm Blackhole P150A daemon alive across FE restarts unless --shutdown-p150!
    std::string host     = "127.0.0.1";
    int port             = 8050;
    std::string model_id = "Hy4-770B-WHT128-192G";
    std::string prof_str = "A5";
    bool link_check      = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* f) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing arg for ") + f);
            return argv[++i];
        };
        if (a == "--host")               host = need("--host");
        else if (a == "--port")          port = std::stoi(need("--port"));
        else if (a == "--model-id")      model_id = need("--model-id");
        else if (a == "--profile")       prof_str = need("--profile");
        else if (a == "--root")          cfg.root_dir = need("--root");
        else if (a == "--weight-root")   cfg.weight_root = need("--weight-root");
        else if (a == "--kv-dir0")       cfg.kv_dir0 = need("--kv-dir0");
        else if (a == "--kv-dir1")       cfg.kv_dir1 = need("--kv-dir1");
        else if (a == "--p150-shm")      cfg.p150_shm_name = need("--p150-shm");
        else if (a == "--max-kv" || a == "--max-ctx") cfg.max_kv_len = std::stoi(need("--max-kv"));
        else if (a == "--ddr4-threads")  cfg.ddr4_threads = std::stoi(need("--ddr4-threads"));
        else if (a == "--dfly2-drafts")  { cfg.dfly2_num_drafts = std::stoi(need("--dfly2-drafts")); if (cfg.dfly2_num_drafts <= 0) cfg.use_dfly2 = false; }
        else if (a == "--no-p150")       cfg.use_p150 = false;
        else if (a == "--no-dfly2")      { cfg.use_dfly2 = false; cfg.dfly2_num_drafts = 0; }
        else if (a == "--no-vit")        cfg.load_vit = false;
        else if (a == "--keep-p150")     cfg.shutdown_p150_on_exit = false;
        else if (a == "--shutdown-p150") cfg.shutdown_p150_on_exit = true;
        else if (a == "--link-check")    link_check = true;
    }

    if (link_check) {
        std::printf("[HY4-770B-FE] link-check OK (pure C++17 / 5x MI50 + 2x Blackhole P150A + 28C EPYC AVX2)\n");
        return 0;
    }

    if (prof_str == "A5" || prof_str == "a5" || prof_str == "FP8Q5" || prof_str == "fp8q5" || prof_str == "3") {
        cfg.profile = RuntimeProfile::PROFILE_A_FIDELITY_FP8_HLUT4_Q5;
    } else if (prof_str == "A4" || prof_str == "a4" || prof_str == "HLUT4" || prof_str == "hlut4" || prof_str == "2") {
        cfg.profile = RuntimeProfile::PROFILE_A_FIDELITY_HLUT4_DDR4;
    } else if (prof_str == "B" || prof_str == "b" || prof_str == "1") {
        cfg.profile = RuntimeProfile::PROFILE_B_MAXVRAM_HLUT3_HLUT2;
    } else if (prof_str == "A3" || prof_str == "a3" || prof_str == "0") {
        cfg.profile = RuntimeProfile::PROFILE_A_FIDELITY_HLUT3_DDR4;
    } else {
        cfg.profile = RuntimeProfile::PROFILE_A_FIDELITY_FP8_HLUT4_Q5;
    }

    try {
        Hy4770bServer srv;
        srv.model_id = model_id;
        srv.rt.init(cfg);

        int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (lfd < 0) throw std::runtime_error("socket() failed");
        int reuse = 1;
        (void)::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(uint16_t(port));
        ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
        if (::bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            ::close(lfd);
            throw std::runtime_error("bind() failed on " + host + ":" + std::to_string(port));
        }
        if (::listen(lfd, 32) < 0) {
            ::close(lfd);
            throw std::runtime_error("listen() failed");
        }

        std::printf("[HY4-770B-FE] Listening on http://%s:%d/ (model=%s, profile=%s, max_kv=%d)\n",
                    host.c_str(), port, model_id.c_str(), profile_name(cfg.profile), cfg.max_kv_len);
        std::fflush(stdout);

        while (!should_stop()) {
            struct timeval tv{0, 200000};
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(lfd, &rfds);
            if (::select(lfd + 1, &rfds, nullptr, nullptr, &tv) <= 0) continue;
            int cfd = ::accept(lfd, nullptr, nullptr);
            if (cfd < 0) continue;
            if (should_stop()) {
                send_http_json(cfd, 503, "Service Unavailable", "{\"error\":\"shutting_down\"}");
                ::close(cfd);
                continue;
            }
            std::thread([&srv, cfd]() { srv.handle_client(cfd); }).detach();
        }
        ::close(lfd);
        std::printf("[HY4-770B-FE] Graceful shutdown initiated — draining active requests before 5x MI50 hipDeviceSynchronize()...\n");
        std::fflush(stdout);
        for (int wait_it = 0; wait_it < 6000 && srv.active_clients.load(std::memory_order_acquire) > 0; ++wait_it) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        {
            std::lock_guard<std::mutex> lk(srv.rt_mu);
            srv.rt.shutdown();
        }
        std::printf("[HY4-770B-FE] Graceful shutdown complete (all 5 MI50s drained, P150A SHM closed, active_clients=%d)\n",
                    srv.active_clients.load(std::memory_order_acquire));
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[HY4-770B-FE] FATAL: %s\n", ex.what());
        drain_all_hip_devices(kNumMi50Ranks, "HY4_770B_FE_ERR");
        return 1;
    }
}
