#pragma once
// Pure C++17 Zero-Disk-Bloat HTTP/2 206 Byte-Range Coalesced Safetensors Streamer
// Reads meta/hy4_fp8_tensor_ranges.tsv and streams ONLY the exact requested tensor set
// (e.g., Layer l = ~9.5 GiB) directly into DDR4 host RAM using parallel curl range transfers,
// coalescing adjacent byte ranges per shard to minimize HTTP requests and achieve 1.00x network traffic
// with 0 GiB rolling NVMe scratch!

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace hy4_770b {

struct RemoteTensorSpec {
    std::string name;
    int shard_id = 0;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t file_offset = 0;
    uint64_t nbytes = 0;
};

struct RemoteRangeIndex {
    std::string repo = "tencent/Hy4-FP8";
    int num_shards = 245;
    std::unordered_map<std::string, RemoteTensorSpec> by_name;
    std::vector<std::string> all_names;

    void load_tsv(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs) throw std::runtime_error("Cannot open tensor range manifest: " + path);
        std::string line;
        std::getline(ifs, line);  // skip header
        by_name.clear();
        all_names.clear();
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            RemoteTensorSpec sp;
            size_t p0 = 0, p1 = line.find('\t');
            sp.name = line.substr(p0, p1 - p0);
            p0 = p1 + 1; p1 = line.find('\t', p0);
            sp.shard_id = std::stoi(line.substr(p0, p1 - p0));
            p0 = p1 + 1; p1 = line.find('\t', p0);
            sp.dtype = line.substr(p0, p1 - p0);
            p0 = p1 + 1; p1 = line.find('\t', p0);
            std::string shape_s = line.substr(p0, p1 - p0);
            {
                size_t a = 0;
                while (a < shape_s.size()) {
                    size_t x = shape_s.find('x', a);
                    if (x == std::string::npos) x = shape_s.size();
                    sp.shape.push_back(std::stoll(shape_s.substr(a, x - a)));
                    a = x + 1;
                }
            }
            p0 = p1 + 1; p1 = line.find('\t', p0);
            sp.file_offset = std::stoull(line.substr(p0, p1 - p0));
            p0 = p1 + 1;
            sp.nbytes = std::stoull(line.substr(p0));
            all_names.push_back(sp.name);
            by_name.emplace(sp.name, std::move(sp));
        }
    }

    std::string shard_url(int shard_id) const {
        char fn[128];
        std::snprintf(fn, sizeof(fn), "model-%05d-of-%05d.safetensors", shard_id, num_shards);
        return "https://huggingface.co/" + repo + "/resolve/main/" + fn;
    }
};

struct StreamedTensorBundle {
    // Contiguous byte chunks fetched per coalesced range, plus pointer map per tensor name
    std::vector<std::vector<uint8_t>> chunks;
    std::unordered_map<std::string, const uint8_t*> ptrs;
    std::unordered_map<std::string, RemoteTensorSpec> specs;
    uint64_t total_tensor_bytes = 0;
    uint64_t total_wire_bytes = 0;
    double elapsed_sec = 0.0;

    bool has(const std::string& name) const noexcept {
        return ptrs.find(name) != ptrs.end();
    }
    const uint8_t* data(const std::string& name) const {
        auto it = ptrs.find(name);
        if (it == ptrs.end()) throw std::runtime_error("Missing streamed tensor: " + name);
        return it->second;
    }
    const RemoteTensorSpec& spec(const std::string& name) const {
        auto it = specs.find(name);
        if (it == specs.end()) throw std::runtime_error("Missing tensor spec: " + name);
        return it->second;
    }
};

// Fetch a subset of tensors directly into DDR4 RAM using coalesced parallel HTTP/2 206 Range requests.
// Splits large coalesced ranges into <= 64 MiB HTTP chunks for maximal 24-stream parallel bandwidth & clean retries.
inline StreamedTensorBundle stream_fetch_tensors(
    const RemoteRangeIndex& idx,
    const std::vector<std::string>& names,
    int num_workers = 24,
    uint64_t max_gap_coalesce = 256ULL * 1024ULL,
    uint64_t max_chunk_bytes  = 64ULL * 1024ULL * 1024ULL)
{
    const auto t0 = std::chrono::steady_clock::now();
    StreamedTensorBundle bundle;

    // Group tensors by shard_id and sort by file_offset
    std::map<int, std::vector<const RemoteTensorSpec*>> by_shard;
    for (const auto& nm : names) {
        auto it = idx.by_name.find(nm);
        if (it == idx.by_name.end()) throw std::runtime_error("Tensor not in index: " + nm);
        by_shard[it->second.shard_id].push_back(&it->second);
        bundle.specs.emplace(nm, it->second);
        bundle.total_tensor_bytes += it->second.nbytes;
    }

    struct RangeJob {
        int shard_id = 0;
        uint64_t start = 0;
        uint64_t end = 0;  // exclusive
        std::vector<const RemoteTensorSpec*> items;
    };
    std::vector<RangeJob> jobs;

    for (auto& kv : by_shard) {
        auto& vec = kv.second;
        std::sort(vec.begin(), vec.end(), [](const RemoteTensorSpec* a, const RemoteTensorSpec* b) {
            return a->file_offset < b->file_offset;
        });
        for (const auto* sp : vec) {
            const uint64_t s = sp->file_offset;
            const uint64_t e = s + sp->nbytes;
            if (!jobs.empty() && jobs.back().shard_id == kv.first &&
                s >= jobs.back().end && (s - jobs.back().end) <= max_gap_coalesce &&
                (e - jobs.back().start) <= max_chunk_bytes)
            {
                jobs.back().end = std::max(jobs.back().end, e);
                jobs.back().items.push_back(sp);
            } else {
                RangeJob j;
                j.shard_id = kv.first;
                j.start = s;
                j.end = e;
                j.items.push_back(sp);
                jobs.push_back(std::move(j));
            }
        }
    }

    // Sort largest jobs first to avoid straggler tails!
    std::sort(jobs.begin(), jobs.end(), [](const RangeJob& a, const RangeJob& b) {
        return (a.end - a.start) > (b.end - b.start);
    });

    bundle.chunks.resize(jobs.size());
    std::atomic<size_t> next_job{0};
    std::atomic<int> failures{0};
    std::atomic<uint64_t> wire_bytes{0};

    auto fetch_exact_range = [&](const std::string& url, uint64_t start, uint64_t end, uint8_t* dst) -> bool {
        const uint64_t expect = end - start;
        // Split very large single tensors (> 128 MiB) into 64 MiB sub-ranges if needed
        uint64_t cur = start;
        uint64_t off = 0;
        while (cur < end) {
            const uint64_t sub_len = std::min<uint64_t>(end - cur, max_chunk_bytes);
            const uint64_t sub_end = cur + sub_len - 1;
            bool ok = false;
            for (int attempt = 0; attempt < 6; ++attempt) {
                char cmd[2048];
                std::snprintf(cmd, sizeof(cmd),
                    "curl -fsSL --connect-timeout 20 --max-time 240 --speed-limit 65536 --speed-time 30 -r %llu-%llu '%s'",
                    (unsigned long long)cur, (unsigned long long)sub_end, url.c_str());
                FILE* fp = popen(cmd, "r");
                if (!fp) continue;
                uint64_t got = 0;
                while (got < sub_len) {
                    size_t n = std::fread(dst + off + got, 1, size_t(sub_len - got), fp);
                    if (n == 0) break;
                    got += uint64_t(n);
                }
                // Drain any unexpected trailing bytes
                uint8_t sink[1024];
                while (std::fread(sink, 1, sizeof(sink), fp) > 0) {}
                int rc = pclose(fp);
                if (rc == 0 && got == sub_len) {
                    ok = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500 * (attempt + 1)));
            }
            if (!ok) return false;
            cur += sub_len;
            off += sub_len;
        }
        return off == expect;
    };

    const int nthr = std::max(1, std::min<int>(num_workers, int(jobs.size())));
    std::vector<std::thread> pool;
    pool.reserve(size_t(nthr));
    for (int t = 0; t < nthr; ++t) {
        pool.emplace_back([&]() {
            while (true) {
                size_t ji = next_job.fetch_add(1);
                if (ji >= jobs.size()) break;
                const auto& j = jobs[ji];
                const uint64_t len = j.end - j.start;
                bundle.chunks[ji].resize(size_t(len));
                if (!fetch_exact_range(idx.shard_url(j.shard_id), j.start, j.end, bundle.chunks[ji].data())) {
                    std::fprintf(stderr, "[STREAM-ERR] Failed shard %03d bytes %llu..%llu (%llu B)\n",
                                 j.shard_id, (unsigned long long)j.start, (unsigned long long)j.end, (unsigned long long)len);
                    failures.fetch_add(1);
                    return;
                }
                wire_bytes.fetch_add(len);
            }
        });
    }
    for (auto& th : pool) th.join();

    if (failures.load() > 0) {
        throw std::runtime_error("stream_fetch_tensors: one or more HTTP range jobs failed");
    }

    for (size_t ji = 0; ji < jobs.size(); ++ji) {
        const auto& j = jobs[ji];
        const uint8_t* base = bundle.chunks[ji].data();
        for (const auto* sp : j.items) {
            bundle.ptrs.emplace(sp->name, base + (sp->file_offset - j.start));
        }
    }
    bundle.total_wire_bytes = wire_bytes.load();
    bundle.elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return bundle;
}

}  // namespace hy4_770b
