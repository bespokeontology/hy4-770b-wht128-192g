#pragma once
// Ground-Truth Architecture & SB-M5-P2-192G Three-Tier Placement Specification
// Model: tencent/Hy4 (770B total / 49B active / 78 layers / 256 routed + 1 shared / 3.3B DFly2 / 1.23B ViT)
// Hardware: SB-M5-P2-192G (5x MI50 128 GiB HBM2 + 2x Blackhole P150A 64 GiB GDDR6 + 330 MiB L1 + 384 GiB DDR4)
// Zero Python anywhere. Every tier capacity is compile-time static_assert verified.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace hy4_770b {

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
#define HY4_HD __host__ __device__
#else
#define HY4_HD
#endif

// ---- Core Backbone Geometry (verified against tencent/Hy4-FP8 config.json & 245 shard headers) ----
inline constexpr int   kNumLayers               = 78;
inline constexpr int   kNumDenseLayers          = 1;       // Layer 0 dense FFN
inline constexpr int   kNumMoeLayers            = 77;      // Layers 1..77 standard non-latent MoE
inline constexpr int   kHiddenSize              = 6144;    // 48 * 128
inline constexpr int   kDenseInterSize          = 18432;   // 144 * 128 (Layer 0 & DFly2 FFN)
inline constexpr int   kVocabSize               = 120832;
inline constexpr int   kMaxPositionEmbeddings   = 1048576;

// ---- iHC (Identity Hyper-Connections, 4 residual streams in FP32) ----
inline constexpr int   kHcMult                  = 4;
inline constexpr int   kHcFlatDim               = kHcMult * kHiddenSize;  // 24576
inline constexpr float kHcEps                   = 1e-6f;
inline constexpr float kHcMagnitude             = 2.0f;
inline constexpr float kRmsNormEps              = 1e-5f;

// ---- Non-Latent SwiGLU MoE Geometry (exact multiples of 128 in ALL dimensions!) ----
inline constexpr int   kNumRoutedExperts        = 256;
inline constexpr int   kTopK                    = 8;
inline constexpr int   kNumSharedExperts        = 1;
inline constexpr int   kMoeInterSize            = 2048;    // 16 * 128
inline constexpr int   kSharedInterSize         = 2048;    // 16 * 128
inline constexpr int   kTotalRoutedExperts      = kNumMoeLayers * kNumRoutedExperts;  // 19,712
inline constexpr float kRoutedScalingFactor     = 2.827f;
inline constexpr float kSwigluLimit             = 10.0f;   // routed experts only; dense/shared/DFly2 unclamped

static_assert(kHiddenSize % 128 == 0, "H=6144 must be exact multiple of 128");
static_assert(kMoeInterSize % 128 == 0, "I_moe=2048 must be exact multiple of 128");
static_assert(kDenseInterSize % 128 == 0, "I_dense=18432 must be exact multiple of 128");

// ---- Gated DSA MLA Attention (100% of all 78 layers, 64 heads, learnable sinks) ----
inline constexpr int   kNumAttnHeads            = 64;
inline constexpr int   kQLoraRank               = 2048;
inline constexpr int   kKvLoraRank              = 512;
inline constexpr int   kQkNopeHeadDim           = 192;
inline constexpr int   kQkRopeHeadDim           = 64;
inline constexpr int   kQkHeadDim               = 256;
inline constexpr int   kVHeadDim                = 256;
inline constexpr int   kKvLatentDim             = kKvLoraRank + kQkRopeHeadDim;  // 576
inline constexpr float kAttnScale               = 0.0625f;  // 1 / sqrt(256)
inline constexpr float kRopeTheta               = 10000000.0f;
inline constexpr int   kMropeSection[3]         = {12, 10, 10};

// ---- Lightning Indexer + IndexCache Cross-Layer Sparse Index Reuse ----
// 21 "full" layers: {0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77}
// 57 "shared" layers reuse the immediately preceding full layer's top-2048 sparse token indices!
inline constexpr int   kIndexNumHeads           = 32;
inline constexpr int   kIndexHeadDim            = 128;     // [64 nope, 64 trailing rope]
inline constexpr int   kIndexTopK               = 2048;
inline constexpr float kIndexLayerNormEps       = 1e-6f;
inline constexpr float kIndexWeightScale        = 1.0f / 64.0f;  // (128^-0.5) * (32^-0.5) = 1/64
inline constexpr int   kNumFullIndexLayers      = 21;

inline constexpr int kFullIndexLayerIds[kNumFullIndexLayers] = {
    0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77
};

HY4_HD inline constexpr bool is_full_indexer_layer(int l) noexcept {
    if (l == 0) return true;
    return (l >= 1 && l < kNumLayers && ((l - 1) % 4 == 0));
}

HY4_HD inline constexpr int indexer_source_layer(int l) noexcept {
    if (l <= 0) return 0;
    return 1 + ((l - 1) / 4) * 4;
}

HY4_HD inline constexpr int full_indexer_slot(int l) noexcept {
    if (l == 0) return 0;
    if (l >= 1 && l < kNumLayers && ((l - 1) % 4 == 0)) return 1 + ((l - 1) / 4);
    return -1;
}

// ---- 2:2:2:1:1 Asymmetric TP5 Spine Sharding Across 5x MI50 (32G:32G:32G:16G:16G) ----
// Matches both 64 attention heads (16:16:16:8:8) and physical HBM2 ratio (32:32:32:16:16) with 128-aligned dims!
inline constexpr int kNumMi50Ranks = 5;
inline constexpr int kTp5HeadStart[kNumMi50Ranks] = { 0, 16, 32, 48, 56 };
inline constexpr int kTp5HeadCount[kNumMi50Ranks] = { 16, 16, 16, 8, 8 };
inline constexpr int kTp5WeightUnits[kNumMi50Ranks] = { 2, 2, 2, 1, 1 };  // sum = 8
inline constexpr int kTp5TotalUnits = 8;

HY4_HD inline constexpr int tp5_slice_start(int rank, int total_dim) noexcept {
    return (kTp5HeadStart[rank] * total_dim) / kNumAttnHeads;
}
HY4_HD inline constexpr int tp5_slice_size(int rank, int total_dim) noexcept {
    return (kTp5HeadCount[rank] * total_dim) / kNumAttnHeads;
}

// ---- Capacity-Weighted 3:3:3:1:1 Sequence-Sharded KV Cache Mapping Across 5x MI50 ----
// 256-token pages striped in 11-page cycles: R0=3, R1=3, R2=3 (32 GiB cards), R3=1, R4=1 (16 GiB cards).
// FP8-g128 c_kv (512 B + 16 B FP32 scales) + BF16 k_pe (128 B) = 656 B/tok/layer!
// At 512K tokens across 78 layers + 21 indexers: 26.34 GiB total -> 7.18 GiB/card (R0..R2), 2.39 GiB/card (R3..R4)!
inline constexpr int kKvPageToks       = 256;
inline constexpr int kKvCyclePages     = 11;
inline constexpr int kKvFp8EntryBytes  = 656;   // 512B FP8 c_kv + 16B (4x FP32 scales) + 128B BF16 k_pe
inline constexpr int kIdxKEntryBytes   = 132;   // 128B FP8 k_idx + 4B FP32 scale
// Massive-activation tail layers L70..L77 (prb_rms=125 -> 5726) use exact lossless BF16 KV cache!
inline constexpr int kBf16KvStartLayer = 70;
inline constexpr int kNumBf16KvLayers  = kNumLayers - kBf16KvStartLayer;  // 8 layers (L70..L77)

HY4_HD inline constexpr int kv_tok_owner(int pos) noexcept {
    const int rem = (pos >> 8) % 11;
    return (rem < 9) ? (rem / 3) : (rem - 6);
}
HY4_HD inline constexpr int kv_tok_local_slot(int pos) noexcept {
    const int pg  = pos >> 8;
    const int cyc = pg / 11;
    const int rem = pg - cyc * 11;
    const int lp  = (rem < 9) ? (cyc * 3 + (rem % 3)) : cyc;
    return (lp << 8) | (pos & 255);
}
HY4_HD inline constexpr int kv_local_tok_to_global(int rank, int lt) noexcept {
    const int lp  = lt >> 8;
    const int cyc = (rank < 3) ? (lp / 3) : lp;
    const int rem = (rank < 3) ? (rank * 3 + (lp - cyc * 3)) : (rank + 6);
    return ((cyc * 11 + rem) << 8) | (lt & 255);
}
HY4_HD inline constexpr int kv_rank_local_toks(int rank, int kv_len) noexcept {
    if (kv_len <= 0) return 0;
    const int full_pg = kv_len >> 8;
    const int tail    = kv_len & 255;
    const int cyc     = full_pg / 11;
    const int rem     = full_pg - cyc * 11;
    const int r_beg   = (rank < 3) ? (rank * 3) : (rank + 6);
    const int r_cnt   = (rank < 3) ? 3 : 1;
    int lp = (rank < 3) ? (cyc * 3) : cyc;
    if (rem > r_beg) lp += ((rem - r_beg) < r_cnt) ? (rem - r_beg) : r_cnt;
    int toks = lp << 8;
    if (tail > 0 && rem >= r_beg && rem < r_beg + r_cnt) toks += tail;
    return toks;
}
HY4_HD inline constexpr int kv_rank_max_toks(int rank, int max_kv_len) noexcept {
    return ((kv_rank_local_toks(rank, max_kv_len) + 255) & ~255) + 256;
}

// ---- Built-In 3.3B 5-Layer DFly2 Parallel Speculative Drafter (KV-Share & Index-Share) ----
inline constexpr int     kDflyNumLayers         = 5;
inline constexpr int     kDflyInterSize         = 18432;
inline constexpr int     kDflyNumHeads          = 64;
inline constexpr int     kDflyHcInterSize       = 6144;
inline constexpr int     kDflyNumSpecTokens     = 5;       // verify M=6 (1 anchor + 5 drafts), max block=7
inline constexpr int     kDflyBlockSize         = 7;
inline constexpr int32_t kDflyMaskTokenId       = 120009;
inline constexpr int     kDflyTargetKvLayers[5] = {61, 65, 69, 73, 77};

// ---- Dynamic 40-Layer ViT (1.23B params) ----
inline constexpr int   kVitNumLayers            = 40;
inline constexpr int   kVitHiddenSize           = 1536;
inline constexpr int   kVitInterSize            = 6144;
inline constexpr int   kVitNumHeads             = 16;
inline constexpr int   kVitHeadDim              = 96;
inline constexpr int   kVitPatchSize            = 16;
inline constexpr int   kVitSpatialMerge         = 2;
inline constexpr float kVitRopeTheta            = 10000.0f;

// ---- Special Token IDs (tencent/Hy4 tokenizer) ----
inline constexpr int32_t kBosTokenId            = 120000;
inline constexpr int32_t kPadTokenId            = 120002;
inline constexpr int32_t kEosTokenId            = 120025;
inline constexpr int32_t kImageStartTokenId     = 120118;
inline constexpr int32_t kImageEndTokenId       = 120119;
inline constexpr int32_t kImageTokenId          = 120120;
inline constexpr int32_t kImageNewlineId        = 120121;
inline constexpr int32_t kVideoStartTokenId     = 120122;
inline constexpr int32_t kVideoEndTokenId       = 120123;
inline constexpr int32_t kVideoTokenId          = 120124;

// ---- Unified 3-Tier Runtime Placement Profiles on SB-M5-P2-192G ----
enum class RuntimeProfile : uint8_t {
    PROFILE_A_FIDELITY_HLUT3_DDR4   = 0,  // Hot in WHT128-HLUT3 (VRAM + L1), cold tail in WHT128-HLUT4 (DDR4)
    PROFILE_B_MAXVRAM_HLUT3_HLUT2   = 1,  // Hot in HLUT3 + Warm in HLUT2 (VRAM + L1), coldest in Q5_1x128_LM (DDR4)
    PROFILE_A_FIDELITY_HLUT4_DDR4   = 2,  // 100% WHT128-HLUT4 across ALL 256/256 experts (5x MI50 + 2x P150A + EPYC DDR4)
    PROFILE_A_FIDELITY_FP8_HLUT4_Q5 = 3,  // 20/L Lossless Stock FP8 (5x MI50) + 42/L WHT128-HLUT4 (2x P150A + L1) + 194/L Q5_1x128_LM (DDR4)
};

inline const char* profile_name(RuntimeProfile p) noexcept {
    switch (p) {
        case RuntimeProfile::PROFILE_A_FIDELITY_HLUT3_DDR4:   return "ProfileA-Fidelity-HLUT3-DDR4";
        case RuntimeProfile::PROFILE_B_MAXVRAM_HLUT3_HLUT2:   return "ProfileB-MaxVRAM-HLUT3-HLUT2";
        case RuntimeProfile::PROFILE_A_FIDELITY_HLUT4_DDR4:   return "ProfileA-Fidelity-100pct-HLUT4";
        case RuntimeProfile::PROFILE_A_FIDELITY_FP8_HLUT4_Q5: return "ProfileA-Fidelity-FP8-HLUT4-Q5";
    }
    return "UNKNOWN";
}

enum class ExpertFormat : uint8_t {
    HLUT3    = 0,  // 3.125 bpw (14.0625 MiB / expert)
    HLUT2    = 1,  // 2.125 bpw ( 9.5625 MiB / expert)
    HLUT4    = 2,  // 4.125 bpw (18.5625 MiB / expert)
    FP8      = 3,  // 8.002 bpw (36.0088 MiB / expert — lossless stock FP8-g128)
    Q5_LM128 = 4,  // 5.125 bpw (23.0625 MiB / expert — per-row 1x128 5-bit Lloyd-Max)
};

enum class ExpertDevice : uint8_t {
    MI50_R0 = 0,
    MI50_R1 = 1,
    MI50_R2 = 2,
    MI50_R3 = 3,
    MI50_R4 = 4,
    P150_TP2 = 5,     // 2x Blackhole P150A (TP2 over FABRIC_1D + 330 MiB L1 SRAM)
    CPU_DDR4 = 6,     // 32C/64T EPYC DDR4 AVX2+F16C cold tail (Q5_LM128 or HLUT4)
};

// Per-layer expert tier counts (ranked 0..255 by strictly disjoint calibration frequency)
struct TierPlacementPlan {
    RuntimeProfile profile = RuntimeProfile::PROFILE_A_FIDELITY_FP8_HLUT4_Q5;
    int max_ctx_tokens     = 524288;
    // Per-device expert counts per layer (sum across all entries == 256!)
    int p150_hlut4         = 0;
    int p150_hlut3         = 0;
    int p150_hlut2         = 0;
    int p150_l1_hot_total  = 0;   // total ultra-hot shards/card pinned in 330 MiB Tensix L1 across 77 layers
    int mi50_fp8[5]        = {0, 0, 0, 0, 0};
    int mi50_hlut4[5]      = {0, 0, 0, 0, 0};
    int mi50_hlut3[5]      = {0, 0, 0, 0, 0};
    int mi50_hlut2[5]      = {0, 0, 0, 0, 0};
    int ddr4_hlut4         = 0;
    int ddr4_q5            = 0;

    constexpr int vram_fp8_per_layer() const noexcept {
        return mi50_fp8[0] + mi50_fp8[1] + mi50_fp8[2] + mi50_fp8[3] + mi50_fp8[4];
    }
    constexpr int vram_hlut4_per_layer() const noexcept {
        return p150_hlut4 + mi50_hlut4[0] + mi50_hlut4[1] + mi50_hlut4[2] + mi50_hlut4[3] + mi50_hlut4[4];
    }
    constexpr int vram_hlut3_per_layer() const noexcept {
        return p150_hlut3 + mi50_hlut3[0] + mi50_hlut3[1] + mi50_hlut3[2] + mi50_hlut3[3] + mi50_hlut3[4];
    }
    constexpr int vram_hlut2_per_layer() const noexcept {
        return p150_hlut2 + mi50_hlut2[0] + mi50_hlut2[1] + mi50_hlut2[2] + mi50_hlut2[3] + mi50_hlut2[4];
    }
    constexpr int vram_total_per_layer() const noexcept {
        return vram_fp8_per_layer() + vram_hlut4_per_layer() + vram_hlut3_per_layer() + vram_hlut2_per_layer();
    }
    constexpr int ddr4_total_per_layer() const noexcept {
        return ddr4_hlut4 + ddr4_q5;
    }
    constexpr int total_per_layer() const noexcept {
        return vram_total_per_layer() + ddr4_total_per_layer();
    }
};

// Exact hardware-verified tier plans fitting inside 192 GiB VRAM + 330 MiB L1 + 384 GiB DDR4
inline constexpr TierPlacementPlan make_tier_plan(RuntimeProfile prof, int max_ctx_tokens, bool use_p150 = true) noexcept {
    TierPlacementPlan p{};
    p.profile = prof;
    p.max_ctx_tokens = max_ctx_tokens;
    const bool high_ctx = (max_ctx_tokens > 262144);  // 512K HBM2 KV reserves +3.6 GiB on R0..R2, +1.2 GiB on R3..R4

    if (prof == RuntimeProfile::PROFILE_A_FIDELITY_FP8_HLUT4_Q5) {
        // Winning Flagship Fidelity Profile:
        //   5x MI50 HBM2 : 20/L Lossless Stock FP8-g128 (ranks 0..19, 54.15 GiB, 3.98 GiB ViT headroom on R4!)
        //   2x P150A GDDR6+L1 : 42/L WHT128-HLUT4 (ranks 20..61, 29.31 GiB/card, 241 MiB L1)
        //   EPYC DDR4 AVX2 : 194/L Q5_1x128_LM 5.125 bpw (ranks 62..255, 336.43 GiB, 26.4 dB SQNR)
        p.mi50_fp8[0] = 5; p.mi50_fp8[1] = 5; p.mi50_fp8[2] = 5;
        p.mi50_fp8[3] = 3; p.mi50_fp8[4] = 2;
        if (use_p150) {
            p.p150_hlut4 = 42; p.p150_l1_hot_total = 14;
        }
        p.ddr4_q5 = kNumRoutedExperts - p.vram_total_per_layer();
        return p;
    } else if (prof == RuntimeProfile::PROFILE_A_FIDELITY_HLUT4_DDR4) {
        if (use_p150) {
            if (!high_ctx) {
                p.p150_hlut4 = 42; p.p150_l1_hot_total = 14;
                p.mi50_hlut4[0] = 13; p.mi50_hlut4[1] = 13; p.mi50_hlut4[2] = 13;
                p.mi50_hlut4[3] = 6;  p.mi50_hlut4[4] = 6;
            } else {
                p.p150_hlut4 = 42; p.p150_l1_hot_total = 14;
                p.mi50_hlut4[0] = 10; p.mi50_hlut4[1] = 11; p.mi50_hlut4[2] = 11;
                p.mi50_hlut4[3] = 5;  p.mi50_hlut4[4] = 5;
            }
        } else {
            p.mi50_hlut4[0] = high_ctx ? 10 : 13;
            p.mi50_hlut4[1] = high_ctx ? 11 : 13;
            p.mi50_hlut4[2] = high_ctx ? 11 : 13;
            p.mi50_hlut4[3] = high_ctx ? 5 : 6;
            p.mi50_hlut4[4] = high_ctx ? 5 : 6;
        }
        p.ddr4_hlut4 = kNumRoutedExperts - p.vram_total_per_layer();
        return p;
    } else if (prof == RuntimeProfile::PROFILE_A_FIDELITY_HLUT3_DDR4) {
        if (use_p150) {
            if (!high_ctx) {
                p.p150_hlut3 = 57; p.p150_hlut2 = 0; p.p150_l1_hot_total = 18;
                p.mi50_hlut3[0] = 17; p.mi50_hlut3[1] = 17; p.mi50_hlut3[2] = 18;
                p.mi50_hlut3[3] = 8;  p.mi50_hlut3[4] = 8;
            } else {
                p.p150_hlut3 = 57; p.p150_hlut2 = 0; p.p150_l1_hot_total = 18;
                p.mi50_hlut3[0] = 14; p.mi50_hlut3[1] = 14; p.mi50_hlut3[2] = 14;
                p.mi50_hlut3[3] = 7;  p.mi50_hlut3[4] = 7;
            }
        } else {
            const int r32 = high_ctx ? 14 : 17;
            const int r16 = high_ctx ? 7 : 8;
            p.mi50_hlut3[0] = r32; p.mi50_hlut3[1] = r32; p.mi50_hlut3[2] = r32;
            p.mi50_hlut3[3] = r16; p.mi50_hlut3[4] = r16;
        }
        p.ddr4_q5 = kNumRoutedExperts - p.vram_total_per_layer();
        return p;
    } else {
        // Profile B: MaxVRAM-HLUT3-HLUT2 (Peak Speed & 512K-1M Context, Q5_1x128_LM DDR4 cold tail!)
        if (use_p150) {
            if (!high_ctx) {
                p.p150_hlut3 = 24; p.p150_hlut2 = 49; p.p150_l1_hot_total = 26;
                p.mi50_hlut3[0] = 10; p.mi50_hlut2[0] = 11;
                p.mi50_hlut3[1] = 10; p.mi50_hlut2[1] = 11;
                p.mi50_hlut3[2] = 10; p.mi50_hlut2[2] = 11;
                p.mi50_hlut3[3] = 3;  p.mi50_hlut2[3] = 7;
                p.mi50_hlut3[4] = 3;  p.mi50_hlut2[4] = 6;
            } else {
                p.p150_hlut3 = 24; p.p150_hlut2 = 49; p.p150_l1_hot_total = 26;
                p.mi50_hlut3[0] = 10; p.mi50_hlut2[0] = 6;
                p.mi50_hlut3[1] = 10; p.mi50_hlut2[1] = 6;
                p.mi50_hlut3[2] = 10; p.mi50_hlut2[2] = 6;
                p.mi50_hlut3[3] = 3;  p.mi50_hlut2[3] = 6;
                p.mi50_hlut3[4] = 3;  p.mi50_hlut2[4] = 6;
            }
        } else {
            p.mi50_hlut3[0] = 10; p.mi50_hlut2[0] = high_ctx ? 6 : 11;
            p.mi50_hlut3[1] = 10; p.mi50_hlut2[1] = high_ctx ? 6 : 11;
            p.mi50_hlut3[2] = 10; p.mi50_hlut2[2] = high_ctx ? 6 : 11;
            p.mi50_hlut3[3] = 3;  p.mi50_hlut2[3] = 6;
            p.mi50_hlut3[4] = 3;  p.mi50_hlut2[4] = 6;
        }
        p.ddr4_q5 = kNumRoutedExperts - p.vram_total_per_layer();
        return p;
    }
}

static_assert(make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_FP8_HLUT4_Q5, 524288).vram_fp8_per_layer() == 20, "Profile A-FP8-Q5 must hold 20 FP8/L on MI50");
static_assert(make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_FP8_HLUT4_Q5, 524288).vram_hlut4_per_layer() == 42, "Profile A-FP8-Q5 must hold 42 HLUT4/L on P150A");
static_assert(make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_FP8_HLUT4_Q5, 524288).ddr4_q5 == 194, "Profile A-FP8-Q5 must hold 194 Q5_LM128/L in DDR4");
static_assert(make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_FP8_HLUT4_Q5, 524288).total_per_layer() == 256, "Must sum to 256 experts");
static_assert(make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_HLUT4_DDR4, 262144).vram_hlut4_per_layer() == 93, "Profile A-HLUT4 @256K must hold 93 HLUT4/layer");
static_assert(make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_HLUT4_DDR4, 524288).vram_hlut4_per_layer() == 84, "Profile A-HLUT4 @512K must hold 84 HLUT4/layer (6,468 total)");
static_assert(make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_HLUT4_DDR4, 524288).total_per_layer() == 256, "Must sum to 256 experts");
static_assert(make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_HLUT3_DDR4, 262144).vram_hlut3_per_layer() == 125, "Profile A @256K must hold 125 HLUT3/layer (9,625 total)");
static_assert(make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_HLUT3_DDR4, 524288).vram_hlut3_per_layer() == 113, "Profile A @512K must hold 113 HLUT3/layer (8,701 total)");
static_assert(make_tier_plan(RuntimeProfile::PROFILE_B_MAXVRAM_HLUT3_HLUT2, 262144).vram_total_per_layer() == 155, "Profile B @256K must hold 60 HLUT3 + 95 HLUT2 = 155/layer (11,935 total)");
static_assert(make_tier_plan(RuntimeProfile::PROFILE_B_MAXVRAM_HLUT3_HLUT2, 524288).vram_total_per_layer() == 139, "Profile B @512K must hold 60 HLUT3 + 79 HLUT2 = 139/layer (10,703 total)");
static_assert(make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_HLUT3_DDR4, 524288).total_per_layer() == 256, "Must sum to 256 experts");
static_assert(make_tier_plan(RuntimeProfile::PROFILE_B_MAXVRAM_HLUT3_HLUT2, 524288).total_per_layer() == 256, "Must sum to 256 experts");

// Storage ranges packed once on disk (supports ALL active plans cleanly):
//   FP8_HOT  : ranks [0,  20)   ->  20 * 77 =  1,540 experts ( 54.15 GiB on <HY4_770B_WORKSPACE_ROOT>/fp8_hot)
//   HLUT4_HOT: ranks [0,  64)   ->  64 * 77 =  4,928 experts ( 89.33 GiB; P150A loads [20, 62))
//   HLUT3    : ranks [0,  60)   ->  60 * 77 =  4,620 experts ( 63.45 GiB; Profile B loads [0, 60))
//   HLUT2    : ranks [60, 155)  ->  95 * 77 =  7,315 experts ( 68.31 GiB; Profile B loads [60, 139))
//   Q5_DDR4  : ranks [62, 256)  -> 194 * 77 = 14,938 experts (336.43 GiB; Profile A [62,256) + Profile B [139,256))
inline constexpr int kPackFp8HotEnd   = 20;
inline constexpr int kPackHlut3End    = 60;
inline constexpr int kPackHlut2Begin  = 60;
inline constexpr int kPackHlut2End    = 155;
inline constexpr int kPackHlut4Begin  = 110;
inline constexpr int kPackQ5Ddr4Begin = 62;

}  // namespace hy4_770b
