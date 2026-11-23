// Pure C++17 Stage A Safetensors Header Harvester for tencent/Hy4-FP8 (245 shards)
// Fetches ONLY the tiny JSON headers via HTTP/2 206 Range requests (~30 MiB total)
// and emits a complete tensor-to-byte-range manifest for zero-bloat causal layer streaming.
// Zero Python. Zero GPU calls.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>

namespace {

struct TensorRange {
    std::string name;
    int shard_id = 0;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t file_offset = 0;   // absolute byte offset in safetensors file
    uint64_t nbytes = 0;
};

static bool run_curl_range(const std::string& url, uint64_t start, uint64_t end, std::vector<uint8_t>& out) {
    out.clear();
    char cmd[2048];
    std::snprintf(cmd, sizeof(cmd),
        "curl -fsSL --retry 5 --retry-delay 2 --max-time 60 -r %llu-%llu '%s'",
        (unsigned long long)start, (unsigned long long)end, url.c_str());
    FILE* fp = popen(cmd, "r");
    if (!fp) return false;
    uint8_t buf[65536];
    for (;;) {
        size_t n = std::fread(buf, 1, sizeof(buf), fp);
        if (n > 0) out.insert(out.end(), buf, buf + n);
        if (n < sizeof(buf)) break;
    }
    int rc = pclose(fp);
    return rc == 0 && !out.empty();
}

static size_t skip_ws(const std::string& s, size_t p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) ++p;
    return p;
}

static bool parse_json_string(const std::string& s, size_t& p, std::string& out) {
    p = skip_ws(s, p);
    if (p >= s.size() || s[p] != '"') return false;
    ++p;
    out.clear();
    while (p < s.size()) {
        char c = s[p++];
        if (c == '"') return true;
        if (c == '\\' && p < s.size()) {
            char e = s[p++];
            out.push_back(e);
        } else {
            out.push_back(c);
        }
    }
    return false;
}

static bool parse_json_int64(const std::string& s, size_t& p, int64_t& out) {
    p = skip_ws(s, p);
    if (p >= s.size()) return false;
    char* endp = nullptr;
    out = std::strtoll(s.c_str() + p, &endp, 10);
    if (endp == s.c_str() + p) return false;
    p = size_t(endp - s.c_str());
    return true;
}

static bool skip_json_value(const std::string& s, size_t& p);

static bool skip_json_object(const std::string& s, size_t& p) {
    p = skip_ws(s, p);
    if (p >= s.size() || s[p] != '{') return false;
    ++p;
    p = skip_ws(s, p);
    if (p < s.size() && s[p] == '}') { ++p; return true; }
    while (p < s.size()) {
        std::string k;
        if (!parse_json_string(s, p, k)) return false;
        p = skip_ws(s, p);
        if (p >= s.size() || s[p] != ':') return false;
        ++p;
        if (!skip_json_value(s, p)) return false;
        p = skip_ws(s, p);
        if (p < s.size() && s[p] == ',') { ++p; continue; }
        if (p < s.size() && s[p] == '}') { ++p; return true; }
        return false;
    }
    return false;
}

static bool skip_json_array(const std::string& s, size_t& p) {
    p = skip_ws(s, p);
    if (p >= s.size() || s[p] != '[') return false;
    ++p;
    p = skip_ws(s, p);
    if (p < s.size() && s[p] == ']') { ++p; return true; }
    while (p < s.size()) {
        if (!skip_json_value(s, p)) return false;
        p = skip_ws(s, p);
        if (p < s.size() && s[p] == ',') { ++p; continue; }
        if (p < s.size() && s[p] == ']') { ++p; return true; }
        return false;
    }
    return false;
}

static bool skip_json_value(const std::string& s, size_t& p) {
    p = skip_ws(s, p);
    if (p >= s.size()) return false;
    if (s[p] == '"') { std::string tmp; return parse_json_string(s, p, tmp); }
    if (s[p] == '{') return skip_json_object(s, p);
    if (s[p] == '[') return skip_json_array(s, p);
    while (p < s.size() && s[p] != ',' && s[p] != '}' && s[p] != ']' &&
           s[p] != ' ' && s[p] != '\n' && s[p] != '\r' && s[p] != '\t') ++p;
    return true;
}

static bool parse_safetensors_header(const std::string& hdr, int shard_id, uint64_t data_base, std::vector<TensorRange>& out) {
    size_t p = skip_ws(hdr, 0);
    if (p >= hdr.size() || hdr[p] != '{') return false;
    ++p;
    while (p < hdr.size()) {
        p = skip_ws(hdr, p);
        if (p < hdr.size() && hdr[p] == '}') return true;
        std::string key;
        if (!parse_json_string(hdr, p, key)) return false;
        p = skip_ws(hdr, p);
        if (p >= hdr.size() || hdr[p] != ':') return false;
        ++p;
        if (key == "__metadata__") {
            if (!skip_json_value(hdr, p)) return false;
        } else {
            p = skip_ws(hdr, p);
            if (p >= hdr.size() || hdr[p] != '{') return false;
            ++p;
            TensorRange tr;
            tr.name = key;
            tr.shard_id = shard_id;
            uint64_t off0 = 0, off1 = 0;
            while (p < hdr.size()) {
                p = skip_ws(hdr, p);
                if (p < hdr.size() && hdr[p] == '}') { ++p; break; }
                std::string fk;
                if (!parse_json_string(hdr, p, fk)) return false;
                p = skip_ws(hdr, p);
                if (p >= hdr.size() || hdr[p] != ':') return false;
                ++p;
                if (fk == "dtype") {
                    if (!parse_json_string(hdr, p, tr.dtype)) return false;
                } else if (fk == "shape") {
                    p = skip_ws(hdr, p);
                    if (p >= hdr.size() || hdr[p] != '[') return false;
                    ++p;
                    while (p < hdr.size()) {
                        p = skip_ws(hdr, p);
                        if (p < hdr.size() && hdr[p] == ']') { ++p; break; }
                        int64_t dim = 0;
                        if (!parse_json_int64(hdr, p, dim)) return false;
                        tr.shape.push_back(dim);
                        p = skip_ws(hdr, p);
                        if (p < hdr.size() && hdr[p] == ',') ++p;
                    }
                } else if (fk == "data_offsets") {
                    p = skip_ws(hdr, p);
                    if (p >= hdr.size() || hdr[p] != '[') return false;
                    ++p;
                    int64_t a = 0, b = 0;
                    if (!parse_json_int64(hdr, p, a)) return false;
                    p = skip_ws(hdr, p);
                    if (p < hdr.size() && hdr[p] == ',') ++p;
                    if (!parse_json_int64(hdr, p, b)) return false;
                    p = skip_ws(hdr, p);
                    if (p >= hdr.size() || hdr[p] != ']') return false;
                    ++p;
                    off0 = uint64_t(a);
                    off1 = uint64_t(b);
                } else {
                    if (!skip_json_value(hdr, p)) return false;
                }
                p = skip_ws(hdr, p);
                if (p < hdr.size() && hdr[p] == ',') ++p;
            }
            tr.file_offset = data_base + off0;
            tr.nbytes = off1 - off0;
            out.push_back(std::move(tr));
        }
        p = skip_ws(hdr, p);
        if (p < hdr.size() && hdr[p] == ',') ++p;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string repo = (argc > 1) ? argv[1] : "tencent/Hy4-FP8";
    const int num_shards = (argc > 2) ? std::atoi(argv[2]) : 245;
    const std::string out_tsv = (argc > 3) ? argv[3] : "<HY4_770B_WORKSPACE_ROOT>/meta/hy4_fp8_tensor_ranges.tsv";
    const int nthreads = (argc > 4) ? std::atoi(argv[4]) : 16;

    std::vector<std::vector<TensorRange>> per_shard(size_t(num_shards + 1));
    std::atomic<int> next_shard{1};
    std::atomic<int> done_shards{0};
    std::atomic<int> fail_shards{0};

    auto worker = [&]() {
        std::vector<uint8_t> buf;
        while (true) {
            int s = next_shard.fetch_add(1);
            if (s > num_shards) break;
            char fname[128];
            std::snprintf(fname, sizeof(fname), "model-%05d-of-%05d.safetensors", s, num_shards);
            std::string url = "https://huggingface.co/" + repo + "/resolve/main/" + fname;

            // Fetch first 256 KiB (almost all safetensors headers fit in <= 256 KiB; if larger, fetch exact remainder)
            const uint64_t kProbe = 262144ULL;
            if (!run_curl_range(url, 0, kProbe - 1, buf) || buf.size() < 8) {
                std::fprintf(stderr, "[ERROR] Shard %03d initial probe failed\n", s);
                fail_shards.fetch_add(1);
                continue;
            }
            uint64_t hdr_len = 0;
            std::memcpy(&hdr_len, buf.data(), 8);
            if (hdr_len > 64ULL * 1024ULL * 1024ULL) {
                std::fprintf(stderr, "[ERROR] Shard %03d bogus header_len=%llu\n", s, (unsigned long long)hdr_len);
                fail_shards.fetch_add(1);
                continue;
            }
            if (8ULL + hdr_len > buf.size()) {
                std::vector<uint8_t> rest;
                if (!run_curl_range(url, uint64_t(buf.size()), 8ULL + hdr_len - 1, rest)) {
                    std::fprintf(stderr, "[ERROR] Shard %03d rest header fetch failed\n", s);
                    fail_shards.fetch_add(1);
                    continue;
                }
                buf.insert(buf.end(), rest.begin(), rest.end());
            }
            std::string hdr(reinterpret_cast<const char*>(buf.data() + 8), size_t(hdr_len));
            if (!parse_safetensors_header(hdr, s, 8ULL + hdr_len, per_shard[size_t(s)])) {
                std::fprintf(stderr, "[ERROR] Shard %03d JSON parse failed\n", s);
                fail_shards.fetch_add(1);
                continue;
            }
            int d = done_shards.fetch_add(1) + 1;
            if (d % 25 == 0 || d == num_shards) {
                std::printf("[HEADERS] %d / %d shards parsed\n", d, num_shards);
                std::fflush(stdout);
            }
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    if (fail_shards.load() > 0) {
        std::fprintf(stderr, "[FATAL] %d shards failed\n", fail_shards.load());
        return 1;
    }

    std::ofstream ofs(out_tsv);
    if (!ofs) {
        std::fprintf(stderr, "[FATAL] Cannot open %s\n", out_tsv.c_str());
        return 1;
    }
    ofs << "name\tshard_id\tdtype\tshape\tfile_offset\tnbytes\n";

    uint64_t total_bytes = 0;
    uint64_t routed_bytes = 0, spine_bytes = 0, dfly_bytes = 0, vit_bytes = 0, embed_lm_bytes = 0;
    size_t total_tensors = 0;
    std::map<std::string, uint64_t> dtype_bytes;

    for (int s = 1; s <= num_shards; ++s) {
        for (const auto& tr : per_shard[size_t(s)]) {
            ++total_tensors;
            total_bytes += tr.nbytes;
            dtype_bytes[tr.dtype] += tr.nbytes;
            if (tr.name.find(".mlp.experts.") != std::string::npos) {
                routed_bytes += tr.nbytes;
            } else if (tr.name.find("dfly_layers") != std::string::npos) {
                dfly_bytes += tr.nbytes;
            } else if (tr.name.find("visual.") == 0 || tr.name.find("vit.") == 0) {
                vit_bytes += tr.nbytes;
            } else if (tr.name.find("embed_tokens") != std::string::npos || tr.name.find("lm_head") != std::string::npos) {
                embed_lm_bytes += tr.nbytes;
            } else {
                spine_bytes += tr.nbytes;
            }
            ofs << tr.name << '\t' << tr.shard_id << '\t' << tr.dtype << '\t';
            for (size_t i = 0; i < tr.shape.size(); ++i) {
                if (i) ofs << 'x';
                ofs << tr.shape[i];
            }
            ofs << '\t' << tr.file_offset << '\t' << tr.nbytes << '\n';
        }
    }
    ofs.close();

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    std::printf("=== %s HEADER AUDIT (%zu tensors across %d shards) ===\n", repo.c_str(), total_tensors, num_shards);
    std::printf("  Total bytes:      %.3f GiB (%llu B)\n", total_bytes / kGiB, (unsigned long long)total_bytes);
    std::printf("  Routed experts:   %.3f GiB\n", routed_bytes / kGiB);
    std::printf("  Backbone spine:   %.3f GiB\n", spine_bytes / kGiB);
    std::printf("  DFly2 drafter:    %.3f GiB\n", dfly_bytes / kGiB);
    std::printf("  Dynamic ViT:      %.3f GiB\n", vit_bytes / kGiB);
    std::printf("  Embed + lm_head:  %.3f GiB\n", embed_lm_bytes / kGiB);
    for (const auto& kv : dtype_bytes) {
        std::printf("  dtype %-8s:   %.3f GiB\n", kv.first.c_str(), kv.second / kGiB);
    }
    std::printf("Wrote manifest: %s\n", out_tsv.c_str());
    return 0;
}
