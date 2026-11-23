#pragma once
// Pure-C++17 Zero-Dependency safetensors mmap Parser, Multi-Shard Index Reader & Writer
// Adapted from Wave 10 reference (<HY4_770B_WORKSPACE_ROOT>/src/safetensors_mmap.hpp).
// Zero Python anywhere. Supports BF16, F16, F32, I32, I64, U8, U16, U32, F8_E4M3.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hy4 {

struct SafetensorEntry {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin_offset = 0;
    uint64_t end_offset = 0;

    uint64_t byte_size() const noexcept { return end_offset - begin_offset; }
    int64_t numel() const noexcept {
        if (shape.empty()) return 1;
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

class SafetensorsMmap {
public:
    SafetensorsMmap() = default;
    explicit SafetensorsMmap(const std::string& path) { open(path); }
    ~SafetensorsMmap() { close(); }

    SafetensorsMmap(const SafetensorsMmap&) = delete;
    SafetensorsMmap& operator=(const SafetensorsMmap&) = delete;

    SafetensorsMmap(SafetensorsMmap&& o) noexcept
        : path_(std::move(o.path_)),
          fd_(o.fd_),
          size_(o.size_),
          data_base_(o.data_base_),
          map_(o.map_),
          entries_(std::move(o.entries_)) {
        o.fd_ = -1;
        o.size_ = 0;
        o.data_base_ = 0;
        o.map_ = nullptr;
    }

    SafetensorsMmap& operator=(SafetensorsMmap&& o) noexcept {
        if (this != &o) {
            close();
            path_ = std::move(o.path_);
            fd_ = o.fd_;
            size_ = o.size_;
            data_base_ = o.data_base_;
            map_ = o.map_;
            entries_ = std::move(o.entries_);
            o.fd_ = -1;
            o.size_ = 0;
            o.data_base_ = 0;
            o.map_ = nullptr;
        }
        return *this;
    }

    void open(const std::string& path) {
        close();
        path_ = path;
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("SafetensorsMmap: failed to open " + path);
        struct stat st {};
        if (::fstat(fd_, &st) != 0 || st.st_size < 8) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("SafetensorsMmap: invalid file size for " + path);
        }
        size_ = static_cast<size_t>(st.st_size);
        map_ = static_cast<const uint8_t*>(::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0));
        if (map_ == MAP_FAILED) {
            map_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("SafetensorsMmap: mmap failed for " + path);
        }
        uint64_t hdr_len = 0;
        std::memcpy(&hdr_len, map_, 8);
        if (hdr_len == 0 || 8ULL + hdr_len > size_) {
            close();
            throw std::runtime_error("SafetensorsMmap: corrupt header length in " + path);
        }
        data_base_ = 8ULL + hdr_len;
        parse_header(reinterpret_cast<const char*>(map_ + 8), static_cast<size_t>(hdr_len));
        // Validate every tensor offset lies within file bounds
        for (const auto& kv : entries_) {
            const auto& e = kv.second;
            if (e.end_offset < e.begin_offset || data_base_ + e.end_offset > size_) {
                close();
                throw std::runtime_error("SafetensorsMmap: out-of-bounds tensor " + e.name + " in " + path);
            }
        }
    }

    void close() {
        if (map_) {
            ::munmap(const_cast<uint8_t*>(map_), size_);
            map_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        size_ = 0;
        data_base_ = 0;
        entries_.clear();
    }

    const std::string& path() const noexcept { return path_; }
    size_t file_size() const noexcept { return size_; }
    bool contains(const std::string& name) const { return entries_.find(name) != entries_.end(); }

    const SafetensorEntry& at(const std::string& name) const {
        auto it = entries_.find(name);
        if (it == entries_.end()) throw std::runtime_error("SafetensorsMmap: missing tensor " + name + " in " + path_);
        return it->second;
    }

    const uint8_t* tensor_bytes(const std::string& name) const {
        const auto& e = at(name);
        return map_ + data_base_ + e.begin_offset;
    }

    const std::map<std::string, SafetensorEntry>& entries() const noexcept { return entries_; }

    // Drop resident pages of this mapping without unmapping (streaming SPC sweeps).
    void release_pages() const noexcept {
        if (map_ && size_) ::madvise(const_cast<uint8_t*>(map_), size_, MADV_DONTNEED);
    }

private:
    std::string path_;
    int fd_ = -1;
    size_t size_ = 0;
    uint64_t data_base_ = 0;
    const uint8_t* map_ = nullptr;
    std::map<std::string, SafetensorEntry> entries_;

    static void skip_ws(const char* s, size_t n, size_t& p) {
        while (p < n && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
    }

    static std::string parse_string(const char* s, size_t n, size_t& p) {
        skip_ws(s, n, p);
        if (p >= n || s[p] != '"') throw std::runtime_error("SafetensorsMmap: expected '\"'");
        ++p;
        std::string out;
        while (p < n && s[p] != '"') {
            if (s[p] == '\\' && p + 1 < n) {
                ++p;
                switch (s[p]) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: out.push_back(s[p]); break;
                }
            } else {
                out.push_back(s[p]);
            }
            ++p;
        }
        if (p >= n || s[p] != '"') throw std::runtime_error("SafetensorsMmap: unterminated string");
        ++p;
        return out;
    }

    static uint64_t parse_u64(const char* s, size_t n, size_t& p) {
        skip_ws(s, n, p);
        uint64_t v = 0;
        bool any = false;
        while (p < n && std::isdigit(static_cast<unsigned char>(s[p]))) {
            v = v * 10ULL + uint64_t(s[p] - '0');
            ++p;
            any = true;
        }
        if (!any) throw std::runtime_error("SafetensorsMmap: expected integer");
        return v;
    }

    static void skip_value(const char* s, size_t n, size_t& p) {
        skip_ws(s, n, p);
        if (p >= n) return;
        if (s[p] == '"') {
            (void)parse_string(s, n, p);
        } else if (s[p] == '{') {
            ++p;
            int depth = 1;
            while (p < n && depth > 0) {
                if (s[p] == '"') {
                    (void)parse_string(s, n, p);
                    continue;
                }
                if (s[p] == '{') ++depth;
                else if (s[p] == '}') --depth;
                ++p;
            }
        } else if (s[p] == '[') {
            ++p;
            int depth = 1;
            while (p < n && depth > 0) {
                if (s[p] == '"') {
                    (void)parse_string(s, n, p);
                    continue;
                }
                if (s[p] == '[') ++depth;
                else if (s[p] == ']') --depth;
                ++p;
            }
        } else {
            while (p < n && s[p] != ',' && s[p] != '}' && s[p] != ']') ++p;
        }
    }

    void parse_header(const char* s, size_t n) {
        size_t p = 0;
        skip_ws(s, n, p);
        if (p >= n || s[p] != '{') throw std::runtime_error("SafetensorsMmap: header must start with '{'");
        ++p;
        while (true) {
            skip_ws(s, n, p);
            if (p >= n) break;
            if (s[p] == '}') { ++p; break; }
            std::string key = parse_string(s, n, p);
            skip_ws(s, n, p);
            if (p >= n || s[p] != ':') throw std::runtime_error("SafetensorsMmap: expected ':'");
            ++p;
            skip_ws(s, n, p);
            if (key == "__metadata__") {
                skip_value(s, n, p);
            } else {
                if (p >= n || s[p] != '{') throw std::runtime_error("SafetensorsMmap: tensor entry must be '{'");
                ++p;
                SafetensorEntry ent;
                ent.name = key;
                while (true) {
                    skip_ws(s, n, p);
                    if (p < n && s[p] == '}') { ++p; break; }
                    std::string fkey = parse_string(s, n, p);
                    skip_ws(s, n, p);
                    if (p >= n || s[p] != ':') throw std::runtime_error("SafetensorsMmap: expected ':' in tensor");
                    ++p;
                    skip_ws(s, n, p);
                    if (fkey == "dtype") {
                        ent.dtype = parse_string(s, n, p);
                    } else if (fkey == "shape") {
                        if (p >= n || s[p] != '[') throw std::runtime_error("SafetensorsMmap: shape must be '['");
                        ++p;
                        while (true) {
                            skip_ws(s, n, p);
                            if (p < n && s[p] == ']') { ++p; break; }
                            ent.shape.push_back(static_cast<int64_t>(parse_u64(s, n, p)));
                            skip_ws(s, n, p);
                            if (p < n && s[p] == ',') ++p;
                        }
                    } else if (fkey == "data_offsets") {
                        if (p >= n || s[p] != '[') throw std::runtime_error("SafetensorsMmap: offsets must be '['");
                        ++p;
                        ent.begin_offset = parse_u64(s, n, p);
                        skip_ws(s, n, p);
                        if (p >= n || s[p] != ',') throw std::runtime_error("SafetensorsMmap: expected ',' in offsets");
                        ++p;
                        ent.end_offset = parse_u64(s, n, p);
                        skip_ws(s, n, p);
                        if (p >= n || s[p] != ']') throw std::runtime_error("SafetensorsMmap: expected ']' in offsets");
                        ++p;
                    } else {
                        skip_value(s, n, p);
                    }
                    skip_ws(s, n, p);
                    if (p < n && s[p] == ',') ++p;
                }
                entries_.emplace(ent.name, std::move(ent));
            }
            skip_ws(s, n, p);
            if (p < n && s[p] == ',') ++p;
        }
    }
};

// Multi-shard HuggingFace safetensors directory reader using model.safetensors.index.json
class ShardedSafetensors {
public:
    ShardedSafetensors() = default;
    explicit ShardedSafetensors(const std::string& dir) { open_dir(dir); }

    void open_dir(const std::string& dir) {
        dir_ = dir;
        weight_map_.clear();
        shards_.clear();
        const std::string idx_path = dir + "/model.safetensors.index.json";
        std::ifstream f(idx_path, std::ios::binary);
        if (!f) throw std::runtime_error("ShardedSafetensors: cannot open " + idx_path);
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const size_t wm = content.find("\"weight_map\"");
        if (wm == std::string::npos) throw std::runtime_error("ShardedSafetensors: missing weight_map");
        size_t p = content.find('{', wm);
        if (p == std::string::npos) throw std::runtime_error("ShardedSafetensors: malformed weight_map");
        ++p;
        auto skip_ws = [&]() { while (p < content.size() && std::isspace(static_cast<unsigned char>(content[p]))) ++p; };
        auto parse_str = [&]() -> std::string {
            skip_ws();
            if (p >= content.size() || content[p] != '"') throw std::runtime_error("ShardedSafetensors: expected '\"'");
            ++p;
            std::string out;
            while (p < content.size() && content[p] != '"') {
                if (content[p] == '\\' && p + 1 < content.size()) { ++p; }
                out.push_back(content[p++]);
            }
            if (p >= content.size()) throw std::runtime_error("ShardedSafetensors: unterminated string");
            ++p;
            return out;
        };
        while (p < content.size()) {
            skip_ws();
            if (content[p] == '}') break;
            std::string tname = parse_str();
            skip_ws();
            if (p >= content.size() || content[p] != ':') throw std::runtime_error("ShardedSafetensors: expected ':'");
            ++p;
            std::string fname = parse_str();
            weight_map_.emplace(std::move(tname), std::move(fname));
            skip_ws();
            if (p < content.size() && content[p] == ',') ++p;
        }
    }

    bool contains(const std::string& name) const { return weight_map_.find(name) != weight_map_.end(); }
    const std::unordered_map<std::string, std::string>& weight_map() const noexcept { return weight_map_; }

    const std::string& shard_filename(const std::string& name) const {
        auto it = weight_map_.find(name);
        if (it == weight_map_.end()) throw std::runtime_error("ShardedSafetensors: unknown tensor " + name);
        return it->second;
    }

    const SafetensorsMmap& get_shard(const std::string& fname) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = shards_.find(fname);
        if (it != shards_.end()) return *it->second;
        auto st = std::make_unique<SafetensorsMmap>(dir_ + "/" + fname);
        const SafetensorsMmap& ref = *st;
        shards_.emplace(fname, std::move(st));
        return ref;
    }

    const SafetensorEntry& entry(const std::string& name) const {
        return get_shard(shard_filename(name)).at(name);
    }

    const uint8_t* tensor_bytes(const std::string& name) const {
        return get_shard(shard_filename(name)).tensor_bytes(name);
    }

    void clear_mmap_cache() {
        std::lock_guard<std::mutex> lk(mu_);
        shards_.clear();
    }

private:
    std::string dir_;
    std::unordered_map<std::string, std::string> weight_map_;
    mutable std::mutex mu_;
    mutable std::unordered_map<std::string, std::unique_ptr<SafetensorsMmap>> shards_;
};

// Streaming safetensors writer (zero extra RAM copy for large packed outputs)
struct SafetensorWriteSpec {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    const void* data = nullptr;
    uint64_t nbytes = 0;
};

inline void write_safetensors_file(
    const std::string& path,
    const std::vector<SafetensorWriteSpec>& tensors,
    const std::map<std::string, std::string>& metadata = {})
{
    std::string hdr = "{";
    bool first = true;
    if (!metadata.empty()) {
        hdr += "\"__metadata__\":{";
        bool mf = true;
        for (const auto& kv : metadata) {
            if (!mf) hdr += ",";
            mf = false;
            hdr += "\"" + kv.first + "\":\"" + kv.second + "\"";
        }
        hdr += "}";
        first = false;
    }
    uint64_t off = 0;
    for (const auto& t : tensors) {
        // 64-byte alignment padding between payloads is NOT standard unless offsets track it;
        // keep packed contiguous offsets matching HuggingFace safetensors exactly.
        if (!first) hdr += ",";
        first = false;
        hdr += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < t.shape.size(); ++i) {
            if (i) hdr += ",";
            hdr += std::to_string(t.shape[i]);
        }
        hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.nbytes) + "]}";
        off += t.nbytes;
    }
    hdr += "}";
    while ((hdr.size() & 7u) != 0u) hdr.push_back(' ');

    const std::string tmp = path + ".tmp";
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) throw std::runtime_error("write_safetensors_file: open failed for " + tmp);

    auto write_all = [&](const void* buf, uint64_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(buf);
        uint64_t rem = len;
        while (rem > 0) {
            const size_t chunk = (rem > (1ULL << 28)) ? size_t(1ULL << 28) : size_t(rem);
            ssize_t w = ::write(fd, p, chunk);
            if (w <= 0) {
                if (w < 0 && errno == EINTR) continue;
                ::close(fd);
                ::unlink(tmp.c_str());
                throw std::runtime_error("write_safetensors_file: write failed for " + tmp);
            }
            p += w;
            rem -= uint64_t(w);
        }
    };

    const uint64_t hlen = uint64_t(hdr.size());
    write_all(&hlen, 8);
    write_all(hdr.data(), hlen);
    for (const auto& t : tensors) {
        if (t.nbytes > 0) write_all(t.data, t.nbytes);
    }
    ::close(fd);
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        ::unlink(tmp.c_str());
        throw std::runtime_error("write_safetensors_file: rename failed for " + path);
    }
}

}  // namespace hy4
