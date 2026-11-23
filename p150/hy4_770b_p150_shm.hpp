#pragma once
// Lock-Free POSIX /dev/shm IPC Channel between 5x MI50 Host (hy4_770b_runtime) and 2x Blackhole P150A
// Supports M = 1..4096, top_k = 8, H = 6144, mixed-precision HLUT3 + HLUT2 + 330 MiB L1 SRAM pinning!
// Zero Python anywhere.

#include "../src/hy4_770b_config.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__HIP_DEVICE_COMPILE__)
#include <immintrin.h>
#endif

namespace hy4_770b {

inline constexpr uint32_t HY4_770B_P150_SHM_MAGIC   = 0x48373730u;  // 'H770'
inline constexpr uint32_t HY4_770B_P150_SHM_VERSION = 1u;
inline constexpr uint32_t HY4_770B_P150_MAX_ROWS    = 512u;
inline constexpr uint32_t HY4_770B_P150_FAST_ROWS   = 16u;

enum Hy4770bP150Op : uint32_t {
    HY4_770B_P150_OP_MOE_EXEC = 1u,
    HY4_770B_P150_OP_PING     = 2u,
    HY4_770B_P150_OP_SHUTDOWN = 99u,
};

struct alignas(64) Hy4770bP150ShmChannel {
    uint32_t magic;
    uint32_t version;
    int32_t  worker_pid;
    uint32_t profile_id;         // 0=ProfileA (HLUT4 or HLUT3), 1=ProfileB (HLUT3+HLUT2)
    uint32_t n_hlut3_per_layer;  // local slots [n_hlut4 .. n_hlut4+n_hlut3) in HLUT3
    uint32_t n_hlut2_per_layer;  // local slots [n_hlut4+n_hlut3 .. total) in HLUT2
    uint32_t ready;
    uint32_t fabric_active;
    uint32_t l1_hot_experts;
    uint32_t n_hlut4_per_layer;  // local slots [0 .. n_hlut4) in HLUT4 (100% HLUT4 Profile A!)
    uint64_t l1_pinned_bytes_total;
    uint64_t gddr6_bytes_per_card;

    alignas(64) std::atomic<uint32_t> req_seq;
    uint32_t op;
    uint32_t layer_idx;          // 1..77
    uint32_t rows;               // M in [1..HY4_770B_P150_MAX_ROWS]
    uint32_t top_k;              // 8

    alignas(64) std::atomic<uint32_t> resp_seq;
    int32_t  status;
    uint64_t last_device_ns;
    std::atomic<uint64_t> total_device_ns;
    std::atomic<uint64_t> stat_total_slots;
    std::atomic<uint64_t> stat_l1_hits;

    // Input: local P150 slot IDs in [0, n_hlut3 + n_hlut2) or -1 if routed elsewhere
    alignas(64) int32_t  row_slots[HY4_770B_P150_MAX_ROWS][kTopK];
    alignas(64) float    row_weights[HY4_770B_P150_MAX_ROWS][kTopK];
    // WHT128-rotated FP32 hidden activations [M, 6144] -> 12 MiB max
    alignas(64) float    x_rot_f32[HY4_770B_P150_MAX_ROWS * kHiddenSize];
    // Accumulated weighted expert output [M, 6144] FP32
    alignas(64) float    y_out_f32[HY4_770B_P150_MAX_ROWS * kHiddenSize];
};

inline void p150_cpu_relax() noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__HIP_DEVICE_COMPILE__)
    _mm_pause();
#endif
}

class Hy4770bP150ShmMapping {
public:
    Hy4770bP150ShmChannel* ptr = nullptr;
    int fd = -1;
    bool owner = false;
    std::string shm_name;

    Hy4770bP150ShmMapping() = default;
    ~Hy4770bP150ShmMapping() { close_mapping(); }
    Hy4770bP150ShmMapping(const Hy4770bP150ShmMapping&) = delete;
    Hy4770bP150ShmMapping& operator=(const Hy4770bP150ShmMapping&) = delete;

    void close_mapping() noexcept {
        if (ptr) { ::munmap(ptr, sizeof(Hy4770bP150ShmChannel)); ptr = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
        if (owner && !shm_name.empty()) { ::shm_unlink(shm_name.c_str()); owner = false; }
    }

    bool create_server(const std::string& name = "/hy4_770b_p150_ipc") {
        close_mapping();
        shm_name = name;
        ::shm_unlink(shm_name.c_str());
        fd = ::shm_open(shm_name.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (fd < 0) return false;
        if (::ftruncate(fd, sizeof(Hy4770bP150ShmChannel)) != 0) { close_mapping(); return false; }
        void* m = ::mmap(nullptr, sizeof(Hy4770bP150ShmChannel), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { close_mapping(); return false; }
        (void)::mlock(m, sizeof(Hy4770bP150ShmChannel));
        std::memset(m, 0, sizeof(Hy4770bP150ShmChannel));
        ptr = static_cast<Hy4770bP150ShmChannel*>(m);
        owner = true;
        ptr->worker_pid = int32_t(::getpid());
        ptr->version = HY4_770B_P150_SHM_VERSION;
        ptr->req_seq.store(0u, std::memory_order_relaxed);
        ptr->resp_seq.store(0u, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        ptr->magic = HY4_770B_P150_SHM_MAGIC;
        return true;
    }

    bool attach_client(const std::string& name = "/hy4_770b_p150_ipc") {
        close_mapping();
        shm_name = name;
        fd = ::shm_open(shm_name.c_str(), O_RDWR | O_CLOEXEC, 0600);
        if (fd < 0) return false;
        void* m = ::mmap(nullptr, sizeof(Hy4770bP150ShmChannel), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { close_mapping(); return false; }
        auto* ch = static_cast<Hy4770bP150ShmChannel*>(m);
        if (ch->magic != HY4_770B_P150_SHM_MAGIC || ch->version != HY4_770B_P150_SHM_VERSION || !ch->ready) {
            ::munmap(m, sizeof(Hy4770bP150ShmChannel));
            ::close(fd);
            fd = -1;
            return false;
        }
        volatile const uint8_t* pb = static_cast<volatile const uint8_t*>(m);
        uint8_t sink = 0;
        for (size_t off = 0; off < sizeof(Hy4770bP150ShmChannel); off += 4096u) sink ^= pb[off];
        (void)sink;
        ptr = ch;
        owner = false;
        return true;
    }
};

}  // namespace hy4_770b
