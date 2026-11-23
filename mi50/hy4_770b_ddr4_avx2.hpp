#pragma once
// Persistent Lock-Free 28-Core AVX2+FMA+F16C DDR4 Cold-Tail Engine for tencent/Hy4 (770B)
// Computes WHT128-HLUT4 (528 B tile, 4.125 bpw) routed experts directly out of hugepage-backed
// 8-channel EPYC DDR4-3200 ECC (~170 GB/s), completely overlapped with 5x MI50 + 2x Blackhole P150A!
// Zero Python. Zero heap allocation in the hot path.

#include "../src/hy4_770b_config.hpp"
#include "../src/hy4_770b_hlut_pack.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace hy4_770b {

struct Ddr4ColdRoute {
    int32_t row = 0;          // token index in [0, M)
    int32_t local_slot = 0;   // index in [0, n_cold_per_layer)
    float   weight = 0.0f;    // scaled router weight (2.827 * norm_prob)
};

class Hy4Ddr4ColdEngine {
public:
    Hy4Ddr4ColdEngine() = default;
    ~Hy4Ddr4ColdEngine();

    Hy4Ddr4ColdEngine(const Hy4Ddr4ColdEngine&) = delete;
    Hy4Ddr4ColdEngine& operator=(const Hy4Ddr4ColdEngine&) = delete;

    // Load cold experts for layers 1..77 into hugepage-backed DDR4 (ranks [cold_rank_begin, 256))
    // When use_q5 == true: maps q5_dir/q5_layer_XX.bin (Q5_LM128, 5.125 bpw, ranks [kPackQ5Ddr4Begin..256), NO WHT!).
    // When use_q5 == false: maps hlut4_layer_XX.bin (+ optional hlut4_hot_dir) (WHT128-HLUT4, 4.125 bpw).
    bool init(const std::string& weight_dir,
              int cold_rank_begin,
              int num_threads = 28,
              int core_offset = 0,
              const std::string& hlut4_hot_dir = "",
              bool use_q5 = false,
              bool use_hugepages = true);

    // Non-blocking submit for layer `layer_idx` (1..77).
    // x_rot_h_f32: [M, 6144] already WHT128-rotated with layer's signs_h!
    // routes / n_routes: cold routes for this layer (grouped by unique local_slot internally).
    // y_out_f32: [M, 6144] caller buffer (zeroed internally before accumulate).
    void submit_async(int layer_idx,
                      const float* x_rot_h_f32,
                      int M,
                      const Ddr4ColdRoute* routes,
                      int n_routes,
                      float* y_out_f32);

    // Zero-overhead direct-BF16 submit for Q5_LM128 mode (parallelizes BF16->int16 quantization across all DDR4 threads!)
    void submit_async_bf16(int layer_idx,
                           const uint16_t* x_raw_bf16,
                           int M,
                           const Ddr4ColdRoute* routes,
                           int n_routes,
                           float* y_out_f32);

    // Wait for submitted layer to complete; returns elapsed microseconds of cold compute.
    double wait_done();

    // Synchronous helper for unit tests & verification
    void compute_sync(int layer_idx,
                      const float* x_rot_h_f32,
                      int M,
                      const Ddr4ColdRoute* routes,
                      int n_routes,
                      float* y_out_f32) {
        submit_async(layer_idx, x_rot_h_f32, M, routes, n_routes, y_out_f32);
        (void)wait_done();
    }

    void shutdown();

    int cold_rank_begin() const noexcept { return cold_rank_begin_; }
    int num_cold_per_layer() const noexcept { return num_cold_per_layer_; }
    bool is_q5_mode() const noexcept { return use_q5_; }
    size_t resident_bytes() const noexcept { return mapped_bytes_; }
    uint64_t total_expert_evals() const noexcept { return total_evals_.load(std::memory_order_relaxed); }
    int last_unique_groups() const noexcept { return job_num_groups_; }

    // Single-expert AVX2 WHT128-HLUT4 & Q5_LM128 forward (exposed for unit tests & verification)
    static void avx2_hlut4_gemv_rows(
        const float* x_rot,
        const Hlut4Tile32x32* tiles,
        const float levels8[8],
        int n_tile_begin,
        int n_tile_end,
        int kt,
        float* y_out) noexcept;

    static void avx2_q5_lm128_gemv_rows(
        const float* x_in,
        const Q5Lm128Block* blocks,
        int row_begin,
        int row_end,
        int kb,
        float* y_out) noexcept;

private:
    void worker_loop(int tid);
    void execute_job_slice(int tid);

    bool use_q5_ = false;
    int cold_rank_begin_ = 125;
    int num_cold_per_layer_ = 131;
    int hot_split_cnt_ = 0;  // number of leading cold slots mapped from hlut4_hot_dir when cold_rank_begin < kPackHlut4Begin
    int num_threads_ = 0;

    // Zero-copy MAP_SHARED file mappings for layers 1..77 (zero double-buffering in DDR4!)
    std::array<void*,          kNumLayers> layer_map_{};
    std::array<size_t,         kNumLayers> layer_map_len_{};
    std::array<const uint8_t*, kNumLayers> layer_data_{};
    std::array<void*,          kNumLayers> hot_map_{};
    std::array<size_t,         kNumLayers> hot_map_len_{};
    std::array<const uint8_t*, kNumLayers> hot_data_{};
    size_t   mapped_bytes_ = 0;

    inline const uint8_t* expert_ptr(int layer_idx, int local_slot) const noexcept {
        if (use_q5_) {
            return layer_data_[size_t(layer_idx)] + size_t(local_slot) * size_t(kQ5ExpertBytes);
        }
        if (local_slot < hot_split_cnt_) {
            return hot_data_[size_t(layer_idx)] + size_t(local_slot) * size_t(kHlut4ExpertBytes);
        }
        return layer_data_[size_t(layer_idx)] + size_t(local_slot - hot_split_cnt_) * size_t(kHlut4ExpertBytes);
    }

    // Per-layer WHT128 signs for down_proj (K = 2048)
    std::array<std::vector<uint64_t>, kNumLayers> signs_i_;
    float levels8_[8]{};

    // Active job descriptor (cacheline aligned, lock-free epoch barrier)
    static constexpr int kMaxBatchM = 512;
    static constexpr int kMaxRoutes = 4096;
    struct UniqueColdGroup {
        int32_t local_slot;
        int32_t count;
        int32_t rows[kMaxBatchM];
        float   weights[kMaxBatchM];
    };

    int job_layer_ = 1;
    int job_M_ = 0;
    const float* job_x_rot_ = nullptr;
    const uint16_t* job_x_bf16_ = nullptr;
    float* job_y_out_ = nullptr;
    int job_num_groups_ = 0;
    UniqueColdGroup job_groups_[kNumRoutedExperts]{};
    // Intermediate z_rot buffer: [max_groups, max_M_per_group] flattened up to kMaxRoutes * 2048 floats (32 MiB)
    std::vector<float>   z_rot_buf_;
    // Pre-quantized int16 even/odd-permuted activation buffers for exact vpshufb+vpmaddwd Q5_LM128 path (cosine=0.99999999!)
    std::vector<int16_t> x_q16_buf_;   // [kMaxBatchM * kHiddenSize]
    std::vector<float>   x_sc_buf_;    // [kMaxBatchM * (kHiddenSize / 128)]
    std::vector<int16_t> z_q16_buf_;   // [kMaxRoutes * kMoeInterSize]
    std::vector<float>   z_sc_buf_;    // [kMaxRoutes * (kMoeInterSize / 128)]
    int job_group_z_off_[kNumRoutedExperts]{};

    std::atomic<uint64_t> job_epoch_{0};
    std::atomic<int>      phase0_done_{0};
    std::atomic<int>      phase1_done_{0};
    std::atomic<int>      phase1b_done_{0};
    std::atomic<int>      phase2_done_{0};
    std::atomic<bool>     stop_flag_{false};
    std::atomic<uint64_t> total_evals_{0};
    std::vector<std::thread> threads_;
    double last_us_ = 0.0;
    int64_t submit_t0_ns_ = 0;
    std::atomic<int64_t> phase2_end_ns_{0};
};

}  // namespace hy4_770b
