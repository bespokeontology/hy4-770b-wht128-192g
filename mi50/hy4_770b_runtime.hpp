#pragma once
// Pure-C++17 Unified 3-Tier Runtime for tencent/Hy4 (770B / 49B Active / 78 Layers / 3.3B DFly2 / 1.23B ViT)
// Hardware: SB-M5-P2-192G (5x MI50 128 GiB HBM2 + 2x Blackhole P150A 64 GiB GDDR6 + 330 MiB L1 + 384 GiB DDR4)
//   - Asymmetric 2:2:2:1:1 TP5 Spine + 3:3:3:1:1 Sequence-Sharded FP8-g128 KV Arena + 21-Full/57-Shared IndexCache
//   - Simultaneous 3-tier MoE dispatch: 5x MI50 EP + 2x P150A FABRIC_1D TP2 (+ L1 pin) + 28-core EPYC AVX2 DDR4
//   - Built-in 3.3B 5-layer DFly2 parallel speculative drafter (M=6 verify, common &= ~3 rollback, cycle breaker)
//   - Dual-NVMe O_DIRECT .kvblk save/restore + 40-layer on-demand Dynamic ViT
//   - Strict graceful shutdown() calling hipDeviceSynchronize() across all 5 MI50s + IPC P150 shutdown
// Zero Python anywhere.

#include "../src/hy4_770b_config.hpp"
#include "../src/hy4_770b_graceful.hpp"
#include "../src/hy4_770b_hlut_pack.hpp"
#include "../src/safetensors_mmap.hpp"
#include "../p150/hy4_770b_p150_shm.hpp"
#include "hy4_770b_core_kernels.hpp"
#include "hy4_770b_ddr4_avx2.hpp"
#include "hy4_770b_hlut_kernels.hpp"
#include "hy4_770b_nvme_kv_cache.hpp"
#include "hy4_770b_vit.hpp"
#include "hy4_cycle_breaker.hpp"
#include "hy4_tokenizer.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hy4_770b {

struct Hy4770bRuntimeConfig {
    std::string root_dir       = "<HY4_770B_WORKSPACE_ROOT>";
    std::string weight_root    = "<HY4_770B_WEIGHT_ROOT>";
    std::string kv_dir0        = "<NVME_STRIPE_0>";
    std::string kv_dir1        = "<NVME_STRIPE_1>";
    std::string p150_shm_name  = "/hy4_770b_p150_ipc";
    RuntimeProfile profile     = RuntimeProfile::PROFILE_A_FIDELITY_HLUT3_DDR4;

    int  max_kv_len            = 262144;   // 262144 or 524288 (up to 1048576)
    bool use_p150              = true;
    bool use_dfly2             = true;
    int  dfly2_num_drafts      = 5;        // 0=off (M=1), 1..5 -> verify block M = 2..6!
    bool use_ddr4_avx2         = true;
    bool use_hugepages         = true;
    int  ddr4_threads          = 26;   // 26 pure physical cores {2..23, 28..31}: zero collision with HIP (0..1) & P150A (24..27)!
    int  ddr4_core_start       = 0;
    bool enable_dpm_warm_guard = true;
    bool enable_kv_disk_cache  = true;
    bool load_vit              = true;
    bool shutdown_p150_on_exit = true;
    int  prefill_chunk_size    = 256;  // 1..512 (default 256 amortizes DDR4 cold-tail weight reads 4x!)
    bool diag_verify_p150_cpu  = false; // cross-check P150A output vs exact CPU HLUT3 reference per layer
};

struct Hy4770bSamplingParams {
    int   max_new_tokens       = 1024;
    float temperature          = 0.6f;
    float top_p                = 0.95f;
    int   top_k                = 40;
    float repetition_penalty   = 1.0f;
    bool  greedy               = false;
    bool  use_dfly2            = true;
    int   dfly2_num_drafts     = 0;        // 0 = inherit cfg_.dfly2_num_drafts; 1..5 = explicit M = 2..6
    uint64_t seed              = 20261122ULL;
    std::vector<int32_t> stop_token_ids = {kEosTokenId, 120020};
};

struct Hy4770bTierStats {
    uint64_t total_routes      = 0;
    uint64_t tensix_l1_hits    = 0;
    uint64_t mi50_hbm2_hits    = 0;
    uint64_t p150_gddr6_hits   = 0;
    uint64_t epyc_ddr4_hits    = 0;
    double   mi50_ms_total     = 0.0;
    double   p150_ms_total     = 0.0;
    double   ddr4_ms_total     = 0.0;

    double tensix_l1_pct() const noexcept  { return total_routes ? (100.0 * double(tensix_l1_hits)  / double(total_routes)) : 0.0; }
    double mi50_hbm2_pct() const noexcept  { return total_routes ? (100.0 * double(mi50_hbm2_hits)  / double(total_routes)) : 0.0; }
    double p150_gddr6_pct() const noexcept { return total_routes ? (100.0 * double(p150_gddr6_hits) / double(total_routes)) : 0.0; }
    double epyc_ddr4_pct() const noexcept  { return total_routes ? (100.0 * double(epyc_ddr4_hits)  / double(total_routes)) : 0.0; }
};

struct Hy4770bGenMetrics {
    int    prompt_tokens               = 0;
    int    generated_tokens            = 0;
    int    cached_prefix_tokens        = 0;
    double prefill_ms                  = 0.0;
    double decode_ms                   = 0.0;
    double ttft_ms                     = 0.0;
    double prefill_tok_s               = 0.0;
    double decode_tok_s                = 0.0;
    int    dfly2_steps                 = 0;
    int    dfly2_proposed              = 0;
    int    dfly2_accepted              = 0;
    double dfly2_accept_rate           = 0.0;
    double dfly2_mean_tokens_per_step  = 1.0;
    double dfly2_verify_step_ms_mean   = 0.0;
    double dfly2_draft_ms_mean         = 0.0;
    std::vector<double> step_ms;
    Hy4770bTierStats tier_delta{};
    std::string finish_reason          = "stop";
};

struct Hy4770bStepProfile {
    uint64_t forward_calls  = 0;
    uint64_t moe_layers     = 0;
    double   ihc_dense_ms   = 0.0;
    double   attn_ms        = 0.0;
    double   router_ms      = 0.0;
    double   moe_wall_ms    = 0.0;
    double   mi50_moe_ms    = 0.0;
    double   p150_moe_ms    = 0.0;
    double   ddr4_moe_ms    = 0.0;
    double   moe_reduce_ms  = 0.0;
    double   lm_head_ms     = 0.0;
    uint64_t ddr4_unique_exp = 0;
};

struct Hy4770bMemoryReport {
    double hip_used_gib[kNumMi50Ranks]  = {};
    double hip_total_gib[kNumMi50Ranks] = {};
    double p150_used_gib[2]             = {};
    double p150_l1_mib_total            = 0.0;
    double ddr4_cold_gib                = 0.0;
    double kv_vram_gib                  = 0.0;
    int    p150_l1_hot_experts          = 0;
    bool   p150_active                  = false;
    bool   dfly2_active                 = false;
    TierPlacementPlan tier_plan{};
};

struct Hy4770bVisualSpan {
    int start_pos = 0;
    int num_tokens = 0;
    const uint16_t* emb_bf16 = nullptr;  // [num_tokens * kHiddenSize(6144)] BF16
};

struct Hy4770bLayerDiag {
    int layer = 0;
    int M = 0;
    float h_rms = 0.0f;
    float h_max_abs = 0.0f;
    float x_rot_max_abs = 0.0f;
    float p150_max_abs = 0.0f;
    float ddr4_max_abs = 0.0f;
    double p150_cpu_cos = 1.0;
    double p150_cpu_max_err = 0.0;
    int p150_max_rows_per_exp = 0;
    int p150_routes = 0;
    int mi50_routes = 0;
    int ddr4_routes = 0;
    std::array<int32_t, kTopK> last_tok_topk{};
    std::vector<float> last_tok_h_stream0;  // [6144] FP32
};

using Hy4770bTokenCallback = std::function<bool(int32_t token_id, int step_idx)>;

class Hy4770bRuntime {
public:
    Hy4770bRuntime();
    ~Hy4770bRuntime();
    Hy4770bRuntime(const Hy4770bRuntime&) = delete;
    Hy4770bRuntime& operator=(const Hy4770bRuntime&) = delete;

    void init(const Hy4770bRuntimeConfig& cfg);
    void shutdown();  // Strictly drains all 5 MI50s via hipDeviceSynchronize() + sends P150 IPC shutdown!

    const Hy4770bRuntimeConfig& config() const noexcept { return cfg_; }
    const hy4::Hy4Tokenizer& tokenizer() const noexcept { return tokenizer_; }
    const Hy4770bMemoryReport& memory_report() const noexcept { return mem_report_; }
    Hy4770bTierStats cumulative_tier_stats() const noexcept;
    void reset_step_profile();
    Hy4770bStepProfile step_profile() const noexcept;

    // KV Cache management (3:3:3:1:1 sequence-sharded across 5x MI50 + Dual-NVMe .kvblk)
    void clear_kv_cache();
    int  current_kv_len() const noexcept;
    const std::vector<int32_t>& cached_tokens() const noexcept;
    bool save_kv_to_nvme(uint64_t prefix_hash);
    bool load_kv_from_nvme(uint64_t prefix_hash, const std::vector<int32_t>& prefix_tokens,
                           bool evict_page_cache_first = false,
                           Hy4770bNvmeKvCache::ReloadStats* out_stats = nullptr);
    // Tiled filler helper for GATE-4 262K / 512K / 1M context verification
    void tile_kv_for_long_context_probe(int target_kv_len);

    // Runtime diagnostics & prefill chunking controls
    void set_prefill_chunk_size(int chunk_sz);
    void set_diag_sink(std::vector<Hy4770bLayerDiag>* sink, bool verify_p150_cpu = false);

    // Core forward pass: runs tokens [pos_start .. pos_start + M) through all 78 layers,
    // updates 3:3:3:1:1 KV cache + 21-full IndexCache, and returns FP32 logits [120832]
    void forward_tokens(
        const std::vector<int32_t>& token_ids,
        int pos_start,
        std::vector<float>& out_last_logits_f32,
        std::vector<float>* out_all_logits_f32 = nullptr,
        std::vector<uint16_t>* out_last_hidden_bf16 = nullptr,
        const std::vector<Hy4770bVisualSpan>* visual_spans = nullptr);

    // Built-in 3.3B 5-layer DFly2 parallel speculative drafter (proposes 1..5 draft tokens in 1 step!)
    void dfly2_draft_block(
        int32_t anchor_token_id,
        int anchor_pos,
        const uint16_t* prev_target_hidden_bf16,
        std::array<int32_t, kDflyNumSpecTokens>& out_draft_ids,
        std::array<float, kDflyNumSpecTokens>& out_confidences,
        int num_drafts = kDflyNumSpecTokens);

    // Native 40-layer Dynamic ViT Vision Encoder
    bool vit_ready() const noexcept;
    Hy4VitOutput encode_image(const Hy4ImageRgb& img) const;

    // Full end-to-end generation with prefix caching, DFly2 M=6 verify, cycle breaker & SSE token streaming
    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt_ids,
        const Hy4770bSamplingParams& params,
        Hy4770bGenMetrics* out_metrics = nullptr,
        const Hy4770bTokenCallback& on_token = nullptr,
        const std::vector<Hy4770bVisualSpan>* visual_spans = nullptr);

    // Continuous-refill lockstep B<=32 batched greedy generation in disjoint 2K KV partitions (100% bit-exact vs B=1,
    // eliminates static-batch straggler waste & saturates B=4 weight-stationary DDR4 Q5_LM128 for 3-5x faster GATE-3!)
    using Hy4770bBatchDoneCallback = std::function<void(int prompt_idx, const std::vector<int32_t>& gen, double running_tok_s)>;
    std::vector<std::vector<int32_t>> generate_batch_greedy(
        const std::vector<std::vector<int32_t>>& prompts,
        int max_new_tokens,
        const std::vector<int32_t>& stop_token_ids = {kEosTokenId, 120020},
        double* out_decode_tok_s = nullptr,
        int max_concurrent = 32,
        const Hy4770bBatchDoneCallback& on_done = nullptr);

private:
    struct Impl;
    Hy4770bRuntimeConfig cfg_;
    hy4::Hy4Tokenizer tokenizer_;
    Hy4770bMemoryReport mem_report_{};
    std::unique_ptr<Impl> impl_;
};

}  // namespace hy4_770b
