#pragma once
// Unified Pure-C++17 WHT128-HLUT2 (2.125 bpw) + WHT128-HLUT3 (3.125 bpw) + WHT128-HLUT4 (4.125 bpw) + Block-FP8
// Quantization, FP64 TARQ RAREBAL-LDLQ Reverse-Cholesky Error Feedback, and Hardware Packers
// Target Geometry: tencent/Hy4 (770B: H = 6144, I_moe = 2048; MI50 EP full expert; P150A TP2 I_shard = 1024)
// Zero Python anywhere. All Gaussian Lloyd-Max levels solved analytically in FP64 to <= 1e-12.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hy4_770b_config.hpp"

namespace hy4_770b {

inline constexpr int kWhtBlock     = 128;
inline constexpr int kTileRows     = 32;
inline constexpr int kTileCols     = 32;
inline constexpr int kTileWeights  = 1024;
inline constexpr int kScaleGroups  = 8;     // 8 groups of 128 weights (4 rows x 32 cols)
inline constexpr int kGroupWeights = 128;
inline constexpr int kSuperRows    = 128;
inline constexpr int kSuperCols    = 128;
inline constexpr int kSuperTiles   = 16;    // 4 x 4 subtiles

// ---- Software/Hardware CRC32C (Castagnoli) ----
inline uint32_t crc32c_update(uint32_t crc, const void* data, size_t len) noexcept {
    static const std::array<uint32_t, 256> tbl = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0x82F63B78u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    uint32_t c = ~crc;
    for (size_t i = 0; i < len; ++i) c = tbl[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return ~c;
}

// ---- Exact IEEE BF16, FP16, and FP8 E4M3FN conversions ----
inline uint16_t f32_to_bf16_rne(float x) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t((u >> 16) | 0x0040u);
    u += 0x00007FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

inline float bf16_to_f32(uint16_t b) noexcept {
    uint32_t u = uint32_t(b) << 16;
    float x = 0.0f;
    std::memcpy(&x, &u, 4);
    return x;
}

inline uint16_t f32_to_fp16_rne(float x) noexcept {
    uint32_t f = 0;
    std::memcpy(&f, &x, 4);
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = int32_t((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = f & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x00800000u;
        const int shift = 14 - exp;
        const uint32_t half_bit = 1u << (shift - 1);
        const uint32_t mask = (1u << shift) - 1u;
        uint32_t h = mant >> shift;
        if ((mant & mask) > half_bit || ((mant & mask) == half_bit && (h & 1u))) ++h;
        return uint16_t(sign | h);
    } else if (exp >= 31) {
        if ((f & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t(sign | 0x7E00u);
        return uint16_t(sign | 0x7C00u);
    }
    const uint32_t round_bit = 0x00001000u;
    const uint32_t rem = mant & 0x00001FFFu;
    uint32_t h_mant = mant >> 13;
    uint32_t h_exp = uint32_t(exp) << 10;
    uint32_t res = h_exp | h_mant;
    if (rem > round_bit || (rem == round_bit && (h_mant & 1u))) ++res;
    return uint16_t(sign | res);
}

inline float fp16_to_f32(uint16_t h) noexcept {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t u = 0;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            int e = -14;
            uint32_t m = mant;
            while ((m & 0x0400u) == 0) { m <<= 1; --e; }
            m &= 0x03FFu;
            u = sign | (uint32_t(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | (uint32_t(int(exp) - 15 + 127) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline const std::array<float, 256>& fp8_e4m3fn_lut() noexcept {
    static const std::array<float, 256> lut = []() {
        std::array<float, 256> t{};
        for (int b = 0; b < 256; ++b) {
            const uint32_t s = uint32_t(b >> 7) & 1u;
            const uint32_t e = uint32_t(b >> 3) & 0xFu;
            const uint32_t m = uint32_t(b) & 0x7u;
            float val = 0.0f;
            if (e == 0u) val = float(m) * (1.0f / 512.0f);
            else if (e == 15u && m == 7u) val = 0.0f;
            else val = std::ldexp(1.0f + float(m) * 0.125f, int(e) - 7);
            t[size_t(b)] = s ? -val : val;
        }
        return t;
    }();
    return lut;
}

inline float fp8_e4m3fn_to_f32(uint8_t b) noexcept { return fp8_e4m3fn_lut()[b]; }

inline uint8_t f32_to_fp8_e4m3fn_rne(float x) noexcept {
    if (std::isnan(x)) return 0x7Fu;
    const uint8_t sign = std::signbit(x) ? 0x80u : 0x00u;
    float ax = std::fabs(x);
    if (ax >= 448.0f) return uint8_t(sign | 0x7Eu);
    if (ax == 0.0f) return sign;
    uint32_t bits = 0;
    std::memcpy(&bits, &ax, 4);
    int exp = int((bits >> 23) & 0xFFu) - 127;
    uint32_t mant = bits & 0x007FFFFFu;
    if (exp < -6) {
        int m = int(std::nearbyintf(ax * 512.0f));
        if (m <= 0) return sign;
        if (m >= 8) return uint8_t(sign | (1u << 3));
        return uint8_t(sign | uint8_t(m));
    }
    int e_fp8 = exp + 7;
    uint32_t m3 = mant >> 20;
    const uint32_t rem = mant & 0x000FFFFFu;
    const uint32_t half = 0x00080000u;
    if (rem > half || (rem == half && (m3 & 1u))) {
        ++m3;
        if (m3 == 8u) { m3 = 0u; ++e_fp8; }
    }
    if (e_fp8 > 15 || (e_fp8 == 15 && m3 >= 7u)) return uint8_t(sign | 0x7Eu);
    return uint8_t(sign | (uint8_t(e_fp8) << 3) | uint8_t(m3));
}

// Dequantize compressed-tensors [128,128] block-FP8 (weight [N, K] F8_E4M3 + weight_scale [N/128, K/128] F32)
inline void dequant_fp8_block128(
    const uint8_t* w_fp8, const float* w_scale, int N, int K, float* out_f32) noexcept
{
    const int kb = (K + 127) / 128;
    const auto& lut = fp8_e4m3fn_lut();
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < N; ++r) {
        const int bi = r >> 7;
        const uint8_t* src = w_fp8 + size_t(r) * size_t(K);
        float* dst = out_f32 + size_t(r) * size_t(K);
        for (int bj = 0; bj < kb; ++bj) {
            const float sc = w_scale[size_t(bi) * size_t(kb) + size_t(bj)];
            const int c0 = bj * 128;
            const int c1 = std::min(K, c0 + 128);
            for (int c = c0; c < c1; ++c) dst[c] = lut[src[c]] * sc;
        }
    }
}

// Quantize float32 [N, K] to [128,128] block-FP8 E4M3FN + F32 scale
struct Fp8BlockQuantMatrix {
    int N = 0, K = 0;
    std::vector<uint8_t> weight_fp8;
    std::vector<float>   weight_scale;  // [(N+127)/128, (K+127)/128]
};

inline Fp8BlockQuantMatrix quantize_fp8_block128(const float* W_rm, int N, int K) {
    Fp8BlockQuantMatrix out;
    out.N = N; out.K = K;
    const int nb = (N + 127) / 128;
    const int kb = (K + 127) / 128;
    out.weight_fp8.resize(size_t(N) * size_t(K));
    out.weight_scale.resize(size_t(nb) * size_t(kb));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nb; ++bi) {
        for (int bj = 0; bj < kb; ++bj) {
            const int r0 = bi * 128, r1 = std::min(N, r0 + 128);
            const int c0 = bj * 128, c1 = std::min(K, c0 + 128);
            float amax = 0.0f;
            for (int r = r0; r < r1; ++r) {
                const float* row = W_rm + size_t(r) * size_t(K);
                for (int c = c0; c < c1; ++c) amax = std::max(amax, std::fabs(row[c]));
            }
            const float sc = std::max(amax, 1e-12f) / 448.0f;
            const float inv = 1.0f / sc;
            out.weight_scale[size_t(bi) * size_t(kb) + size_t(bj)] = sc;
            for (int r = r0; r < r1; ++r) {
                const float* src = W_rm + size_t(r) * size_t(K);
                uint8_t* dst = out.weight_fp8.data() + size_t(r) * size_t(K);
                for (int c = c0; c < c1; ++c) dst[c] = f32_to_fp8_e4m3fn_rne(src[c] * inv);
            }
        }
    }
    return out;
}

// ---- Deterministic Per-Layer Rademacher Signs & Orthonormal WHT128 ----
inline uint64_t hy4_770b_wht_seed_h(int layer_idx) noexcept {
    return 0x4859343737304800ULL ^ (uint64_t(layer_idx + 1) * 0x9E3779B97F4A7C15ULL);
}
inline uint64_t hy4_770b_wht_seed_i(int layer_idx) noexcept {
    return 0x4859343737304900ULL ^ (uint64_t(layer_idx + 1) * 0xBF58476D1CE4E5B9ULL);
}

inline std::vector<uint64_t> make_wht_signs(int K, uint64_t seed) {
    if (K <= 0 || (K % kWhtBlock) != 0) throw std::invalid_argument("WHT128 requires K % 128 == 0");
    std::vector<uint64_t> mask(size_t(K / 64), 0ULL);
    uint64_t st = seed ^ 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < mask.size(); ++i) {
        st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
        mask[i] = st * 0x2545F4914F6CDD1DULL;
    }
    return mask;
}

inline std::vector<uint64_t> make_signs_h(int layer_idx) { return make_wht_signs(kHiddenSize, hy4_770b_wht_seed_h(layer_idx)); }
inline std::vector<uint64_t> make_signs_i(int layer_idx) { return make_wht_signs(kMoeInterSize, hy4_770b_wht_seed_i(layer_idx)); }

inline void wht128_block(float* v, const uint64_t* sign_bits_block2) noexcept {
    for (int i = 0; i < 64; ++i) {
        if ((sign_bits_block2[0] >> i) & 1ULL) v[i] = -v[i];
        if ((sign_bits_block2[1] >> i) & 1ULL) v[64 + i] = -v[64 + i];
    }
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += (len << 1)) {
            for (int j = 0; j < len; ++j) {
                const float a = v[i + j];
                const float b = v[i + len + j];
                v[i + j] = a + b;
                v[i + len + j] = a - b;
            }
        }
    }
    constexpr float kInvSqrt128 = 0.08838834764831844655f;
    for (int i = 0; i < 128; ++i) v[i] *= kInvSqrt128;
}

inline void wht128_rows(const float* in, float* out, int rows, int K, const std::vector<uint64_t>& signs) {
    const int nblk = K / 128;
    #pragma omp parallel for schedule(static) if(rows >= 16)
    for (int r = 0; r < rows; ++r) {
        const float* xr = in + size_t(r) * size_t(K);
        float* yr = out + size_t(r) * size_t(K);
        if (yr != xr) std::memcpy(yr, xr, size_t(K) * sizeof(float));
        for (int b = 0; b < nblk; ++b) wht128_block(yr + b * 128, &signs[size_t(b) * 2]);
    }
}

inline void build_wht128_block_matrix(int b, const std::vector<uint64_t>& signs, float out_128x128[128][128]) noexcept {
    constexpr float kInvSqrt128 = 0.08838834764831844655f;
    const uint64_t s0 = signs[size_t(b) * 2];
    const uint64_t s1 = signs[size_t(b) * 2 + 1];
    for (int i = 0; i < 128; ++i) {
        const float si = (((i < 64 ? (s0 >> i) : (s1 >> (i - 64))) & 1ULL) != 0ULL) ? -1.0f : 1.0f;
        for (int j = 0; j < 128; ++j) {
            const int pop = __builtin_popcount(unsigned(i & j));
            out_128x128[i][j] = si * ((pop & 1) ? -kInvSqrt128 : kInvSqrt128);
        }
    }
}

inline float hy4_swiglu_clamped(float g, float u) noexcept {
    const float gc = std::min(g, kSwigluLimit);
    const float uc = std::max(-kSwigluLimit, std::min(kSwigluLimit, u));
    return (gc / (1.0f + std::exp(-gc))) * uc;
}

// ---- Exact Analytic FP64 Gaussian Lloyd-Max Solver (HLUT2 n_pos=2, HLUT3 n_pos=4, HLUT4 n_pos=8) ----
inline constexpr int kMaxLloydPosLevels = 8;

inline double gaussian_pdf(double x) noexcept {
    static const double kInvSqrtTwoPi = 1.0 / std::sqrt(2.0 * std::acos(-1.0));
    return kInvSqrtTwoPi * std::exp(-0.5 * x * x);
}
inline double gaussian_cdf(double x) noexcept { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

struct HalfNormalMoments { double mass = 0.0, first = 0.0, second = 0.0; };
inline HalfNormalMoments half_normal_moments(double a, double b) noexcept {
    const bool inf_b = std::isinf(b) || b > 1e300;
    const double pb = inf_b ? 1.0 : gaussian_cdf(b);
    const double fb = inf_b ? 0.0 : gaussian_pdf(b);
    const double pa = gaussian_cdf(a);
    const double fa = gaussian_pdf(a);
    return {2.0 * (pb - pa), 2.0 * (fa - fb), 2.0 * (pb - pa + a * fa - (inf_b ? 0.0 : b * fb))};
}
inline double positive_normal_quantile(double p) noexcept {
    const double target = 0.5 * (1.0 + p);
    double lo = 0.0, hi = 1.0;
    while (gaussian_cdf(hi) < target) hi *= 2.0;
    for (int it = 0; it < 256; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (gaussian_cdf(mid) < target) lo = mid; else hi = mid;
        if (hi - lo <= 1e-15) break;
    }
    return 0.5 * (lo + hi);
}

struct LloydMaxSolution {
    int n_pos = 0;
    std::array<double, kMaxLloydPosLevels> levels{};
    std::array<double, kMaxLloydPosLevels - 1> thresholds{};
    double mse = 0.0, eq2 = 1.0, sqnr_db = 0.0;
    double convergence_residual = 0.0, fixed_point_residual = 0.0;
    int iterations = 0;
    bool converged = false;
};

inline LloydMaxSolution solve_gaussian_lloyd_max(int n_pos) {
    if (n_pos < 2 || n_pos > kMaxLloydPosLevels) throw std::invalid_argument("n_pos out of range");
    LloydMaxSolution out;
    out.n_pos = n_pos;
    double thresholds[kMaxLloydPosLevels - 1]{}, levels[kMaxLloydPosLevels]{};
    for (int i = 0; i < n_pos - 1; ++i) thresholds[i] = positive_normal_quantile(double(i + 1) / double(n_pos));
    auto compute_levels = [&](const double* th, double* lev) {
        double a = 0.0;
        for (int i = 0; i < n_pos; ++i) {
            const double b = (i < n_pos - 1) ? th[i] : HUGE_VAL;
            const auto m = half_normal_moments(a, b);
            lev[i] = m.first / m.mass;
            a = b;
        }
    };
    compute_levels(thresholds, levels);
    double max_delta = HUGE_VAL;
    int it = 0;
    for (; it < 2048; ++it) {
        max_delta = 0.0;
        for (int i = 0; i < n_pos - 1; ++i) thresholds[i] = 0.5 * (levels[i] + levels[i + 1]);
        double next[kMaxLloydPosLevels]{};
        compute_levels(thresholds, next);
        for (int i = 0; i < n_pos; ++i) { max_delta = std::max(max_delta, std::fabs(next[i] - levels[i])); levels[i] = next[i]; }
        if (max_delta <= 1e-12) break;
    }
    for (int i = 0; i < n_pos - 1; ++i) thresholds[i] = 0.5 * (levels[i] + levels[i + 1]);
    double eq2 = 0.0, a = 0.0, fixed_res = 0.0;
    for (int i = 0; i < n_pos; ++i) {
        const double b = (i < n_pos - 1) ? thresholds[i] : HUGE_VAL;
        const auto m = half_normal_moments(a, b);
        fixed_res = std::max(fixed_res, std::fabs(levels[i] - m.first / m.mass));
        eq2 += levels[i] * levels[i] * m.mass;
        a = b;
    }
    out.mse = 1.0 - eq2; out.eq2 = eq2; out.sqnr_db = 10.0 * std::log10(1.0 / out.mse);
    out.convergence_residual = max_delta; out.fixed_point_residual = fixed_res;
    out.iterations = it + 1; out.converged = (max_delta <= 1e-9);
    for (int i = 0; i < n_pos; ++i) out.levels[size_t(i)] = levels[i];
    for (int i = 0; i < n_pos - 1; ++i) out.thresholds[size_t(i)] = thresholds[i];
    return out;
}

inline const LloydMaxSolution& lloyd_max_hlut2() { static const auto s = solve_gaussian_lloyd_max(2); return s; }
inline const LloydMaxSolution& lloyd_max_hlut3() { static const auto s = solve_gaussian_lloyd_max(4); return s; }
inline const LloydMaxSolution& lloyd_max_hlut4() { static const auto s = solve_gaussian_lloyd_max(8); return s; }

// ============================================================================
// EXACT HARDWARE TILE & SUPERBLOCK STRUCTS (HLUT2 / HLUT3 / HLUT4)
// ============================================================================

// HLUT2: 2.125 bpw (272 B tile, 4,352 B = 17 x 256 B superblock)
// 2-bit code per weight: [bit1 = sign | bit0 = mag (0 or 1)], 16 weights/uint32_t, 2 words/row
inline constexpr int kHlut2TileBytes  = 272;
inline constexpr int kHlut2SuperBytes = 4352;
struct alignas(16) Hlut2Tile32x32 {
    uint32_t q_2b[64];       // 256 B: 2 words/row (16 cols/word)
    uint16_t scales_f16[8];  // 16 B:  FP16 RMS scale per 128-weight group (4 rows x 32 cols)
};
static_assert(sizeof(Hlut2Tile32x32) == kHlut2TileBytes, "HLUT2 tile must be 272 B = 2.125 bpw");
struct alignas(256) Hlut2Superblock128x128 { Hlut2Tile32x32 subtiles[kSuperTiles]; };
static_assert(sizeof(Hlut2Superblock128x128) == kHlut2SuperBytes, "HLUT2 superblock must be 4,352 B");

// HLUT3: 3.125 bpw (400 B tile, 6,400 B = 25 x 256 B superblock)
inline constexpr int kHlut3TileBytes  = 400;
inline constexpr int kHlut3SuperBytes = 6400;
struct alignas(16) Hlut3Tile32x32 {
    uint32_t q_lo2[64];      // 256 B: 2-bit mag (0..3), 16 weights/word
    uint32_t q_hi1[32];      // 128 B: 1-bit sign plane, 1 word/row
    uint16_t scales_f16[8];  // 16 B
};
static_assert(sizeof(Hlut3Tile32x32) == kHlut3TileBytes, "HLUT3 tile must be 400 B = 3.125 bpw");
struct alignas(256) Hlut3Superblock128x128 { Hlut3Tile32x32 subtiles[kSuperTiles]; };
static_assert(sizeof(Hlut3Superblock128x128) == kHlut3SuperBytes, "HLUT3 superblock must be 6,400 B");

// HLUT4: 4.125 bpw (528 B tile, 8,448 B = 33 x 256 B superblock)
inline constexpr int kHlut4TileBytes  = 528;
inline constexpr int kHlut4SuperBytes = 8448;
struct alignas(16) Hlut4Tile32x32 {
    uint32_t q_nib4[128];    // 512 B: 4-bit signed nibble [sign<<3 | mag(0..7)], 4 words/row
    uint16_t scales_f16[8];  // 16 B
};
static_assert(sizeof(Hlut4Tile32x32) == kHlut4TileBytes, "HLUT4 tile must be 528 B = 4.125 bpw");
struct alignas(256) Hlut4Superblock128x128 { Hlut4Tile32x32 subtiles[kSuperTiles]; };
static_assert(sizeof(Hlut4Superblock128x128) == kHlut4SuperBytes, "HLUT4 superblock must be 8,448 B");

// Exact Hy4-770B [6144, 2048] per-expert byte geometry (compile-time verified!)
inline constexpr size_t   kProjTiles            = (kHiddenSize / 32) * (kMoeInterSize / 32);    // 192 * 64 = 12,288
inline constexpr size_t   kProjSuperblocks      = (kHiddenSize / 128) * (kMoeInterSize / 128);  // 48 * 16  = 768
inline constexpr size_t   kTp2ProjSuperblocks   = (kHiddenSize / 128) * ((kMoeInterSize / 2) / 128); // 48 * 8 = 384
inline constexpr uint64_t kHlut2ProjBytes       = kProjTiles * uint64_t(kHlut2TileBytes);       // 3,342,336 B
inline constexpr uint64_t kHlut3ProjBytes       = kProjTiles * uint64_t(kHlut3TileBytes);       // 4,915,200 B
inline constexpr uint64_t kHlut4ProjBytes       = kProjTiles * uint64_t(kHlut4TileBytes);       // 6,488,064 B
inline constexpr uint64_t kHlut2ExpertBytes     = 3ULL * kHlut2ProjBytes;                       // 10,027,008 B (9.5625 MiB)
inline constexpr uint64_t kHlut3ExpertBytes     = 3ULL * kHlut3ProjBytes;                       // 14,745,600 B (14.0625 MiB)
inline constexpr uint64_t kHlut4ExpertBytes     = 3ULL * kHlut4ProjBytes;                       // 19,464,192 B (18.5625 MiB)
inline constexpr uint64_t kHlut2Tp2CardExpBytes = 3ULL * kTp2ProjSuperblocks * uint64_t(kHlut2SuperBytes); // 5,013,504 B
inline constexpr uint64_t kHlut3Tp2CardExpBytes = 3ULL * kTp2ProjSuperblocks * uint64_t(kHlut3SuperBytes); // 7,372,800 B

static_assert(kHlut2ExpertBytes == 10'027'008ULL, "HLUT2 expert = 9.5625 MiB");
static_assert(kHlut3ExpertBytes == 14'745'600ULL, "HLUT3 expert = 14.0625 MiB");
static_assert(kHlut4ExpertBytes == 19'464'192ULL, "HLUT4 expert = 18.5625 MiB");
static_assert(kHlut2Tp2CardExpBytes * 2ULL == kHlut2ExpertBytes, "TP2 HLUT2 sum matches full expert");
static_assert(kHlut3Tp2CardExpBytes * 2ULL == kHlut3ExpertBytes, "TP2 HLUT3 sum matches full expert");

// ---- Per-Row 1x128 5-Bit Symmetric Gaussian Lloyd-Max (Q5_LM128: 5.125 bpw, 26.4 dB SQNR, 0 WHT!) ----
inline constexpr float kLm5Levels16[16] = {
    0.06841367f, 0.20524099f, 0.34260064f, 0.48129931f,
    0.62213445f, 0.76643509f, 0.91554350f, 1.07103181f,
    1.23501885f, 1.41010308f, 1.60014248f, 1.81094837f,
    2.05182028f, 2.33973432f, 2.71157813f, 3.27883911f
};
inline constexpr float kLm5Thresh15[15] = {
    0.13682733f, 0.27392082f, 0.41194998f, 0.55171688f,
    0.69428477f, 0.84098929f, 0.99328766f, 1.15302533f,
    1.32256097f, 1.50512278f, 1.70554543f, 1.93138433f,
    2.19577730f, 2.52565623f, 2.99520862f
};

inline constexpr int kQ5BlockBytes = 82;
struct Q5Lm128Block {
    uint8_t  mag4[64];   // 64 B: 128 x 4-bit magnitude indices [0..15] (low nibble = even idx, high = odd)
    uint8_t  sign1[16];  // 16 B: 128 x 1-bit sign bits (bit b of sign1[j] = sign of elem 8*j + b)
    uint16_t scale_f16;  //  2 B: 2-step L2-optimal FP16 sigma
};
static_assert(sizeof(Q5Lm128Block) == kQ5BlockBytes, "Q5Lm128Block must be exactly 82 B (5.125 bpw)");

inline constexpr size_t   kQ5ProjBlocks  = (size_t(kHiddenSize) / 128u) * size_t(kMoeInterSize);  // 98,304
inline constexpr uint64_t kQ5ProjBytes   = uint64_t(kQ5ProjBlocks) * uint64_t(kQ5BlockBytes);     // 8,060,928 B
inline constexpr uint64_t kQ5ExpertBytes = 3ULL * kQ5ProjBytes;                                   // 24,182,784 B (23.0625 MiB)
static_assert(kQ5ExpertBytes == 24'182'784ULL, "Q5_LM128 expert = 23.0625 MiB");

// Exact Lossless Stock FP8-g128 Expert Layout on Disk & HBM2:
//   [gate_w (12 MiB), gate_s (3 KiB), up_w (12 MiB), up_s (3 KiB), down_w (12 MiB), down_s (3 KiB)]
inline constexpr uint64_t kFp8ProjWBytes  = uint64_t(kHiddenSize) * uint64_t(kMoeInterSize);       // 12,582,912 B
inline constexpr uint64_t kFp8ProjSBytes  = uint64_t(kHiddenSize / 128) * uint64_t(kMoeInterSize / 128) * 4ULL; // 3,072 B
inline constexpr uint64_t kFp8ProjBytes   = kFp8ProjWBytes + kFp8ProjSBytes;                       // 12,585,984 B
inline constexpr uint64_t kFp8ExpertBytes = 3ULL * kFp8ProjBytes;                                  // 37,757,952 B (36.0088 MiB)
static_assert(kFp8ExpertBytes == 37'757'952ULL, "Lossless stock FP8-g128 expert = 36.0088 MiB");

inline void q5_lm128_pack_block(const float* src128, Q5Lm128Block& dst) noexcept {
    double ss = 0.0;
    for (int i = 0; i < 128; ++i) ss += double(src128[i]) * double(src128[i]);
    float sig = float(std::sqrt(ss * (1.0 / 128.0) + 1e-18));
    uint8_t idx[128];
    auto refine = [&](float cur_sig) noexcept -> float {
        const float inv = 1.0f / std::max(cur_sig, 1e-12f);
        double num = 0.0, den = 0.0;
        for (int i = 0; i < 128; ++i) {
            const float av = std::fabs(src128[i]);
            const float z = av * inv;
            // Branchless binary search over 15 thresholds (4 comparisons)
            int k = (z >= kLm5Thresh15[7]) ? 8 : 0;
            if (z >= kLm5Thresh15[k + 3]) k += 4;
            if (z >= kLm5Thresh15[k + 1]) k += 2;
            if (k < 15 && z >= kLm5Thresh15[k]) k += 1;
            idx[i] = uint8_t(k);
            num += double(av) * double(kLm5Levels16[k]);
            den += double(kLm5Levels16[k]) * double(kLm5Levels16[k]);
        }
        return (den > 1e-18) ? float(num / den) : cur_sig;
    };
    float s1 = refine(fp16_to_f32(f32_to_fp16_rne(sig)));
    float s2 = refine(fp16_to_f32(f32_to_fp16_rne(s1)));
    const uint16_t s_f16 = f32_to_fp16_rne(s2);
    (void)refine(fp16_to_f32(s_f16));
    dst.scale_f16 = s_f16;
    for (int j = 0; j < 16; ++j) {
        const int base = j * 8;
        uint8_t sb = 0u;
        for (int b = 0; b < 8; ++b) {
            if (std::signbit(src128[base + b])) sb |= uint8_t(1u << b);
        }
        dst.sign1[j] = sb;
        dst.mag4[j * 4 + 0] = uint8_t((idx[base + 0] & 0xFu) | ((idx[base + 1] & 0xFu) << 4));
        dst.mag4[j * 4 + 1] = uint8_t((idx[base + 2] & 0xFu) | ((idx[base + 3] & 0xFu) << 4));
        dst.mag4[j * 4 + 2] = uint8_t((idx[base + 4] & 0xFu) | ((idx[base + 5] & 0xFu) << 4));
        dst.mag4[j * 4 + 3] = uint8_t((idx[base + 6] & 0xFu) | ((idx[base + 7] & 0xFu) << 4));
    }
}

inline void q5_lm128_unpack_block(const Q5Lm128Block& blk, float* out128) noexcept {
    const float sig = fp16_to_f32(blk.scale_f16);
    for (int j = 0; j < 16; ++j) {
        const uint8_t sb = blk.sign1[j];
        for (int p = 0; p < 4; ++p) {
            const uint8_t byte = blk.mag4[j * 4 + p];
            const int b0 = p * 2, b1 = p * 2 + 1;
            const float v0 = kLm5Levels16[byte & 0xFu] * sig;
            const float v1 = kLm5Levels16[(byte >> 4) & 0xFu] * sig;
            out128[j * 8 + b0] = ((sb >> b0) & 1u) ? -v0 : v0;
            out128[j * 8 + b1] = ((sb >> b1) & 1u) ? -v1 : v1;
        }
    }
}

// ---- Cell Accessors & Quantizers ----
inline uint32_t hlut2_quant_mag(double z, const LloydMaxSolution& sol) noexcept {
    return (z >= sol.thresholds[0]) ? 1u : 0u;
}
inline void hlut2_set_cell(Hlut2Tile32x32& tile, int r, int c, uint32_t mag, uint32_t sign) noexcept {
    const int idx = r * 32 + c;
    const int w = idx >> 4, sh = 2 * (idx & 15);
    const uint32_t code = ((sign & 1u) << 1) | (mag & 1u);
    tile.q_2b[w] = (tile.q_2b[w] & ~(3u << sh)) | (code << sh);
}
inline void hlut2_pack_tile(const float* w_rot_rm, int K, int n0, int k0, const LloydMaxSolution& sol, Hlut2Tile32x32& tile) noexcept {
    std::memset(&tile, 0, sizeof(tile));
    for (int g = 0; g < kScaleGroups; ++g) {
        double sq = 0.0;
        for (int r = 4 * g; r < 4 * g + 4; ++r) {
            const float* src = w_rot_rm + size_t(n0 + r) * size_t(K) + size_t(k0);
            for (int c = 0; c < 32; ++c) sq += double(src[c]) * double(src[c]);
        }
        tile.scales_f16[g] = f32_to_fp16_rne(float(std::sqrt(sq / 128.0 + 1e-30)));
    }
    for (int r = 0; r < 32; ++r) {
        const float* src = w_rot_rm + size_t(n0 + r) * size_t(K) + size_t(k0);
        const float sigma = fp16_to_f32(tile.scales_f16[r >> 2]);
        const double inv_s = (sigma > 0.0f) ? (1.0 / double(sigma)) : 0.0;
        for (int c = 0; c < 32; ++c) {
            hlut2_set_cell(tile, r, c, hlut2_quant_mag(double(std::fabs(src[c])) * inv_s, sol), std::signbit(src[c]) ? 1u : 0u);
        }
    }
}
inline void hlut2_unpack_tile(const Hlut2Tile32x32& tile, const LloydMaxSolution& sol, float* out_rm, int K, int n0, int k0) noexcept {
    for (int r = 0; r < 32; ++r) {
        float* dst = out_rm + size_t(n0 + r) * size_t(K) + size_t(k0);
        const float sigma = fp16_to_f32(tile.scales_f16[r >> 2]);
        for (int c = 0; c < 32; ++c) {
            const int idx = r * 32 + c;
            const uint32_t code = (tile.q_2b[idx >> 4] >> (2 * (idx & 15))) & 3u;
            const float v = float(sol.levels[code & 1u]) * sigma;
            dst[c] = (code & 2u) ? -v : v;
        }
    }
}

inline uint32_t hlut3_quant_mag(double z, const LloydMaxSolution& sol) noexcept {
    if (z >= sol.thresholds[2]) return 3u;
    if (z >= sol.thresholds[1]) return 2u;
    if (z >= sol.thresholds[0]) return 1u;
    return 0u;
}
inline void hlut3_set_cell(Hlut3Tile32x32& tile, int r, int c, uint32_t mag, uint32_t sign) noexcept {
    const int idx = r * 32 + c;
    const int mw = idx >> 4, ms = 2 * (idx & 15);
    tile.q_lo2[mw] = (tile.q_lo2[mw] & ~(3u << ms)) | ((mag & 3u) << ms);
    tile.q_hi1[r]  = (tile.q_hi1[r]  & ~(1u << c))  | ((sign & 1u) << c);
}
inline void hlut3_pack_tile(const float* w_rot_rm, int K, int n0, int k0, const LloydMaxSolution& sol, Hlut3Tile32x32& tile) noexcept {
    std::memset(&tile, 0, sizeof(tile));
    for (int g = 0; g < kScaleGroups; ++g) {
        double sq = 0.0;
        for (int r = 4 * g; r < 4 * g + 4; ++r) {
            const float* src = w_rot_rm + size_t(n0 + r) * size_t(K) + size_t(k0);
            for (int c = 0; c < 32; ++c) sq += double(src[c]) * double(src[c]);
        }
        tile.scales_f16[g] = f32_to_fp16_rne(float(std::sqrt(sq / 128.0 + 1e-30)));
    }
    for (int r = 0; r < 32; ++r) {
        const float* src = w_rot_rm + size_t(n0 + r) * size_t(K) + size_t(k0);
        const float sigma = fp16_to_f32(tile.scales_f16[r >> 2]);
        const double inv_s = (sigma > 0.0f) ? (1.0 / double(sigma)) : 0.0;
        for (int c = 0; c < 32; ++c) {
            hlut3_set_cell(tile, r, c, hlut3_quant_mag(double(std::fabs(src[c])) * inv_s, sol), std::signbit(src[c]) ? 1u : 0u);
        }
    }
}
inline void hlut3_unpack_tile(const Hlut3Tile32x32& tile, const LloydMaxSolution& sol, float* out_rm, int K, int n0, int k0) noexcept {
    for (int r = 0; r < 32; ++r) {
        float* dst = out_rm + size_t(n0 + r) * size_t(K) + size_t(k0);
        const float sigma = fp16_to_f32(tile.scales_f16[r >> 2]);
        const uint32_t sw = tile.q_hi1[r];
        for (int c = 0; c < 32; ++c) {
            const int idx = r * 32 + c;
            const uint32_t mag = (tile.q_lo2[idx >> 4] >> (2 * (idx & 15))) & 3u;
            const float v = float(sol.levels[mag]) * sigma;
            dst[c] = ((sw >> c) & 1u) ? -v : v;
        }
    }
}

inline uint32_t hlut4_quant_mag(double z, const LloydMaxSolution& sol) noexcept {
    if (z >= sol.thresholds[3]) {
        if (z >= sol.thresholds[5]) return (z >= sol.thresholds[6]) ? 7u : 6u;
        return (z >= sol.thresholds[4]) ? 5u : 4u;
    } else {
        if (z >= sol.thresholds[1]) return (z >= sol.thresholds[2]) ? 3u : 2u;
        return (z >= sol.thresholds[0]) ? 1u : 0u;
    }
}
inline void hlut4_set_cell(Hlut4Tile32x32& tile, int r, int c, uint32_t mag, uint32_t sign) noexcept {
    const int w = r * 4 + (c >> 3), sh = 4 * (c & 7);
    const uint32_t nib = ((sign & 1u) << 3) | (mag & 7u);
    tile.q_nib4[w] = (tile.q_nib4[w] & ~(0xFu << sh)) | (nib << sh);
}
inline void hlut4_pack_tile(const float* w_rot_rm, int K, int n0, int k0, const LloydMaxSolution& sol, Hlut4Tile32x32& tile) noexcept {
    std::memset(&tile, 0, sizeof(tile));
    for (int g = 0; g < kScaleGroups; ++g) {
        double sq = 0.0;
        for (int r = 4 * g; r < 4 * g + 4; ++r) {
            const float* src = w_rot_rm + size_t(n0 + r) * size_t(K) + size_t(k0);
            for (int c = 0; c < 32; ++c) sq += double(src[c]) * double(src[c]);
        }
        tile.scales_f16[g] = f32_to_fp16_rne(float(std::sqrt(sq / 128.0 + 1e-30)));
    }
    for (int r = 0; r < 32; ++r) {
        const float* src = w_rot_rm + size_t(n0 + r) * size_t(K) + size_t(k0);
        const float sigma = fp16_to_f32(tile.scales_f16[r >> 2]);
        const double inv_s = (sigma > 0.0f) ? (1.0 / double(sigma)) : 0.0;
        for (int c = 0; c < 32; ++c) {
            hlut4_set_cell(tile, r, c, hlut4_quant_mag(double(std::fabs(src[c])) * inv_s, sol), std::signbit(src[c]) ? 1u : 0u);
        }
    }
}
inline void hlut4_unpack_tile(const Hlut4Tile32x32& tile, const LloydMaxSolution& sol, float* out_rm, int K, int n0, int k0) noexcept {
    for (int r = 0; r < 32; ++r) {
        float* dst = out_rm + size_t(n0 + r) * size_t(K) + size_t(k0);
        const float sigma = fp16_to_f32(tile.scales_f16[r >> 2]);
        for (int c = 0; c < 32; ++c) {
            const uint32_t nib = (tile.q_nib4[r * 4 + (c >> 3)] >> (4 * (c & 7))) & 0xFu;
            const float v = float(sol.levels[nib & 7u]) * sigma;
            dst[c] = (nib & 8u) ? -v : v;
        }
    }
}

// ============================================================================
// FP64 TARQ RAREBAL-LDLQ REVERSE CHOLESKY (Fast Blocked FP64 + 4-Row AVX2 LDLQ)
// ============================================================================
struct RarebalStats {
    int m_cal = 0, m_common = 0, m_rare = 0;
    bool used_starvation_fallback = false;
    double tr_common = 0.0, tr_rare = 0.0, lambda_tarq = 1.0, d_bar = 0.0, lambda_reg = 0.0, cond_est = 1.0;
};

// Compute R_up [K, K] FP32 from symmetric FP64 Hessian H_in [K, K] (damped with ridge fraction damp_frac).
// Uses fast right-looking blocked Cholesky (block B=64) in FP64 with OpenMP + AVX2 FMA.
inline std::vector<float> build_rup_from_fp64_hessian(
    const double* H_sym, int K, double damp_frac, RarebalStats* out_stats = nullptr)
{
    double diag_mean = 0.0;
    for (int i = 0; i < K; ++i) diag_mean += H_sym[size_t(i) * size_t(K) + size_t(i)];
    diag_mean /= double(K);
    const double damp = damp_frac * std::max(1e-12, diag_mean);
    if (out_stats) { out_stats->d_bar = diag_mean; out_stats->lambda_reg = damp; }

    // Order-reversed lower triangle H_rev[ir, jr] = H[K-1-ir, K-1-jr] (ir >= jr)
    std::vector<double> L(size_t(K) * size_t(K), 0.0);
    #pragma omp parallel for schedule(static)
    for (int ir = 0; ir < K; ++ir) {
        const int is = K - 1 - ir;
        double* dst = L.data() + size_t(ir) * size_t(K);
        const double* src = H_sym + size_t(is) * size_t(K);
        for (int jr = 0; jr <= ir; ++jr) dst[jr] = src[K - 1 - jr];
        dst[ir] += damp;
    }

    // Blocked right-looking Cholesky (B = 64)
    constexpr int B = 64;
    for (int k0 = 0; k0 < K; k0 += B) {
        const int kb = std::min(B, K - k0);
        // 1. Panel diagonal block L[k0:k0+kb, k0:k0+kb]
        for (int j = 0; j < kb; ++j) {
            const int jj = k0 + j;
            double* lj = L.data() + size_t(jj) * size_t(K);
            double s = lj[jj];
            for (int m = k0; m < jj; ++m) s -= lj[m] * lj[m];
            if (s <= 1e-18) s = 1e-18;
            const double ljj = std::sqrt(s);
            lj[jj] = ljj;
            const double inv = 1.0 / ljj;
            for (int i = j + 1; i < kb; ++i) {
                const int ii = k0 + i;
                double* li = L.data() + size_t(ii) * size_t(K);
                double dot = li[jj];
                for (int m = k0; m < jj; ++m) dot -= li[m] * lj[m];
                li[jj] = dot * inv;
            }
        }
        // 2. Panel below diagonal: TRSM L[k0+kb:K, k0:k0+kb] * L11^-T
        const int rem_start = k0 + kb;
        if (rem_start >= K) break;
        #pragma omp parallel for schedule(static)
        for (int ii = rem_start; ii < K; ++ii) {
            double* li = L.data() + size_t(ii) * size_t(K);
            for (int j = 0; j < kb; ++j) {
                const int jj = k0 + j;
                const double* lj = L.data() + size_t(jj) * size_t(K);
                double dot = li[jj];
                for (int m = k0; m < jj; ++m) dot -= li[m] * lj[m];
                li[jj] = dot / lj[jj];
            }
        }
        // 3. Trailing matrix SYRK update: L[ii, jj] -= sum_{m=k0..k0+kb-1} L[ii, m] * L[jj, m] for rem_start <= jj <= ii < K
        #pragma omp parallel for schedule(dynamic, 16)
        for (int ii = rem_start; ii < K; ++ii) {
            double* li = L.data() + size_t(ii) * size_t(K);
            const double* lip = li + k0;
            for (int jj = rem_start; jj <= ii; ++jj) {
                const double* ljp = L.data() + size_t(jj) * size_t(K) + k0;
                double dot = 0.0;
                #pragma omp simd reduction(+:dot)
                for (int m = 0; m < kb; ++m) dot += lip[m] * ljp[m];
                li[jj] -= dot;
            }
        }
    }

    if (out_stats) {
        double lmin = HUGE_VAL, lmax = 0.0;
        for (int i = 0; i < K; ++i) {
            const double d = L[size_t(i) * size_t(K) + size_t(i)];
            if (d < lmin) lmin = d;
            if (d > lmax) lmax = d;
        }
        const double r = (lmin > 1e-18) ? (lmax / lmin) : 1e18;
        out_stats->cond_est = r * r;
    }

    std::vector<float> R_up(size_t(K) * size_t(K), 0.0f);
    #pragma omp parallel for schedule(static)
    for (int col = 0; col < K; ++col) {
        const int ir = K - 1 - col;
        float* r_row = R_up.data() + size_t(col) * size_t(K);
        r_row[col] = 1.0f;
        for (int k = col + 1; k < K; ++k) {
            const int jr = K - 1 - k;
            r_row[k] = float(L[size_t(ir) * size_t(K) + size_t(jr)] / L[size_t(jr) * size_t(K) + size_t(jr)]);
        }
    }
    return R_up;
}

// Build TARQ RAREBAL-weighted FP64 Hessian from already-WHT128-rotated calibration activations x_rot [M, K]
inline std::vector<double> build_rarebal_fp64_hessian(
    const float* x_rot, int M_cal, int K, bool enable_rarebal, RarebalStats* out_stats = nullptr)
{
    if (out_stats) *out_stats = RarebalStats{};
    std::vector<uint8_t> rare(size_t(M_cal), 0u);
    if (enable_rarebal && M_cal >= 8) {
        std::vector<double> mean_dir(size_t(K), 0.0);
        std::vector<float> inv_n(size_t(M_cal), 0.0f);
        for (int m = 0; m < M_cal; ++m) {
            const float* xm = x_rot + size_t(m) * size_t(K);
            double n2 = 0.0;
            for (int k = 0; k < K; ++k) n2 += double(xm[k]) * double(xm[k]);
            const double inv = (n2 > 1e-30) ? (1.0 / std::sqrt(n2)) : 0.0;
            inv_n[size_t(m)] = float(inv);
            for (int k = 0; k < K; ++k) mean_dir[size_t(k)] += double(xm[k]) * inv;
        }
        double mn2 = 0.0;
        for (int k = 0; k < K; ++k) mn2 += mean_dir[size_t(k)] * mean_dir[size_t(k)];
        const double inv_mn = (mn2 > 1e-30) ? (1.0 / std::sqrt(mn2)) : 0.0;
        for (int k = 0; k < K; ++k) mean_dir[size_t(k)] *= inv_mn;
        std::vector<std::pair<float, int>> align(static_cast<size_t>(M_cal));
        for (int m = 0; m < M_cal; ++m) {
            const float* xm = x_rot + size_t(m) * size_t(K);
            double dot = 0.0;
            for (int k = 0; k < K; ++k) dot += double(xm[k]) * mean_dir[size_t(k)];
            align[size_t(m)] = {float(std::fabs(dot) * double(inv_n[size_t(m)])), m};
        }
        const int rt = std::max(1, M_cal / 8);
        std::nth_element(align.begin(), align.begin() + rt, align.end());
        for (int i = 0; i < rt; ++i) rare[size_t(align[size_t(i)].second)] = 1u;
    }

    double tr_c = 0.0, tr_r = 0.0;
    int mc = 0, mr = 0;
    for (int m = 0; m < M_cal; ++m) {
        const float* xm = x_rot + size_t(m) * size_t(K);
        double sq = 0.0;
        for (int k = 0; k < K; ++k) sq += double(xm[k]) * double(xm[k]);
        if (enable_rarebal && rare[size_t(m)]) { tr_r += sq; ++mr; } else { tr_c += sq; ++mc; }
    }
    const double lam = (enable_rarebal && mc > 0 && mr > 0 && tr_c > 0.0 && tr_r > 0.0) ? (tr_c / std::max(1e-12, tr_r)) : 1.0;

    // Weighted panel: sqrt(w) * x_rot in FP64 -> H = Xw^T Xw
    const double wc = 1.0 / double(std::max(1, M_cal));
    const double wr = lam * wc;
    std::vector<double> H(size_t(K) * size_t(K), 0.0);

    // Blocked SYRK on CPU (fallback when GPU rocblas_dsyrk isn't used)
    constexpr int kMBlk = 128;
    #pragma omp parallel
    {
        std::vector<double> a0(static_cast<size_t>(K)), a1(static_cast<size_t>(K)),
                            a2(static_cast<size_t>(K)), a3(static_cast<size_t>(K));
        #pragma omp for schedule(dynamic, 4)
        for (int i0 = 0; i0 < K; i0 += 4) {
            const int rc = std::min(4, K - i0);
            for (int mb = 0; mb < M_cal; mb += kMBlk) {
                const int mend = std::min(M_cal, mb + kMBlk);
                std::fill(a0.begin() + i0, a0.end(), 0.0);
                if (rc > 1) std::fill(a1.begin() + i0 + 1, a1.end(), 0.0);
                if (rc > 2) std::fill(a2.begin() + i0 + 2, a2.end(), 0.0);
                if (rc > 3) std::fill(a3.begin() + i0 + 3, a3.end(), 0.0);
                for (int m = mb; m < mend; ++m) {
                    const float* xm = x_rot + size_t(m) * size_t(K);
                    const double wm = rare[size_t(m)] ? wr : wc;
                    const double v0 = wm * double(xm[i0]);
                    #pragma omp simd
                    for (int j = i0; j < K; ++j) a0[size_t(j)] += v0 * double(xm[j]);
                    if (rc > 1) {
                        const double v1 = wm * double(xm[i0 + 1]);
                        #pragma omp simd
                        for (int j = i0 + 1; j < K; ++j) a1[size_t(j)] += v1 * double(xm[j]);
                    }
                    if (rc > 2) {
                        const double v2 = wm * double(xm[i0 + 2]);
                        #pragma omp simd
                        for (int j = i0 + 2; j < K; ++j) a2[size_t(j)] += v2 * double(xm[j]);
                    }
                    if (rc > 3) {
                        const double v3 = wm * double(xm[i0 + 3]);
                        #pragma omp simd
                        for (int j = i0 + 3; j < K; ++j) a3[size_t(j)] += v3 * double(xm[j]);
                    }
                }
                for (int j = i0; j < K; ++j) H[size_t(i0) * size_t(K) + size_t(j)] += a0[size_t(j)];
                if (rc > 1) for (int j = i0 + 1; j < K; ++j) H[size_t(i0 + 1) * size_t(K) + size_t(j)] += a1[size_t(j)];
                if (rc > 2) for (int j = i0 + 2; j < K; ++j) H[size_t(i0 + 2) * size_t(K) + size_t(j)] += a2[size_t(j)];
                if (rc > 3) for (int j = i0 + 3; j < K; ++j) H[size_t(i0 + 3) * size_t(K) + size_t(j)] += a3[size_t(j)];
            }
        }
    }
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < i; ++j) H[size_t(i) * size_t(K) + size_t(j)] = H[size_t(j) * size_t(K) + size_t(i)];

    if (out_stats) {
        out_stats->m_cal = M_cal; out_stats->m_common = mc; out_stats->m_rare = mr;
        out_stats->tr_common = tr_c; out_stats->tr_rare = tr_r; out_stats->lambda_tarq = lam;
    }
    return H;
}

// ---- Quantized Matrix Containers & Fast 4-Row-Blocked AVX2+FMA LDLQ ----
template <typename TileT>
struct HlutQuantMatrix {
    int N = 0, K = 0, nt = 0, kt = 0;
    LloydMaxSolution lloyd{};
    std::vector<TileT> tiles;
    size_t raw_bytes() const noexcept { return tiles.size() * sizeof(TileT); }
    const TileT& tile_at(int n_tile, int k_tile) const noexcept { return tiles[size_t(n_tile) * size_t(kt) + size_t(k_tile)]; }
};

using Hlut2QuantMatrix = HlutQuantMatrix<Hlut2Tile32x32>;
using Hlut3QuantMatrix = HlutQuantMatrix<Hlut3Tile32x32>;
using Hlut4QuantMatrix = HlutQuantMatrix<Hlut4Tile32x32>;

// 4-row simultaneous AVX2+FMA causal LDLQ update (4x lower R_up memory traffic!)
inline void ldlq_axpy4_avx2(float* w0, float* w1, float* w2, float* w3,
                            float e0, float e1, float e2, float e3,
                            const float* rrow, int start_k, int K) noexcept
{
#if defined(__AVX2__)
    int k = start_k;
    while (k < K && (k & 7) != 0) {
        const float rv = rrow[k];
        w0[k] += e0 * rv; w1[k] += e1 * rv; w2[k] += e2 * rv; w3[k] += e3 * rv;
        ++k;
    }
    const __m256 ve0 = _mm256_set1_ps(e0);
    const __m256 ve1 = _mm256_set1_ps(e1);
    const __m256 ve2 = _mm256_set1_ps(e2);
    const __m256 ve3 = _mm256_set1_ps(e3);
    for (; k + 8 <= K; k += 8) {
        const __m256 rv = _mm256_loadu_ps(rrow + k);
        _mm256_storeu_ps(w0 + k, _mm256_fmadd_ps(ve0, rv, _mm256_loadu_ps(w0 + k)));
        _mm256_storeu_ps(w1 + k, _mm256_fmadd_ps(ve1, rv, _mm256_loadu_ps(w1 + k)));
        _mm256_storeu_ps(w2 + k, _mm256_fmadd_ps(ve2, rv, _mm256_loadu_ps(w2 + k)));
        _mm256_storeu_ps(w3 + k, _mm256_fmadd_ps(ve3, rv, _mm256_loadu_ps(w3 + k)));
    }
    for (; k < K; ++k) {
        const float rv = rrow[k];
        w0[k] += e0 * rv; w1[k] += e1 * rv; w2[k] += e2 * rv; w3[k] += e3 * rv;
    }
#else
    for (int k = start_k; k < K; ++k) {
        const float rv = rrow[k];
        w0[k] += e0 * rv; w1[k] += e1 * rv; w2[k] += e2 * rv; w3[k] += e3 * rv;
    }
#endif
}

template <typename TileT, typename QuantFn, typename SetFn>
inline HlutQuantMatrix<TileT> pack_hlut_from_wrot_ldlq(
    const float* w_rot, int N, int K,
    const LloydMaxSolution& sol,
    const std::vector<float>& R_up,
    QuantFn&& quant_mag_fn,
    SetFn&& set_cell_fn)
{
    HlutQuantMatrix<TileT> q;
    q.N = N; q.K = K; q.nt = N / 32; q.kt = K / 32; q.lloyd = sol;
    q.tiles.assign(size_t(q.nt) * size_t(q.kt), TileT{});
    const bool use_ldlq = (R_up.size() == size_t(K) * size_t(K));

    // Step 1: Compute per-(4 rows x 32 cols) RMS FP16 scales in parallel
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nt_i = 0; nt_i < q.nt; ++nt_i) {
        for (int kt_i = 0; kt_i < q.kt; ++kt_i) {
            TileT& tile = q.tiles[size_t(nt_i) * size_t(q.kt) + size_t(kt_i)];
            for (int g = 0; g < kScaleGroups; ++g) {
                double sq = 0.0;
                for (int r = 4 * g; r < 4 * g + 4; ++r) {
                    const float* src = w_rot + size_t(nt_i * 32 + r) * size_t(K) + size_t(kt_i * 32);
                    for (int c = 0; c < 32; ++c) sq += double(src[c]) * double(src[c]);
                }
                tile.scales_f16[g] = f32_to_fp16_rne(float(std::sqrt(sq / 128.0 + 1e-30)));
            }
        }
    }

    // Step 2: 4-row-blocked LDLQ (each 4-row group shares exact same scale group `sg` and `tile`!)
    #pragma omp parallel for schedule(static)
    for (int r4 = 0; r4 < N; r4 += 4) {
        const int nt_i = r4 >> 5;
        const int lr0  = r4 & 31;
        const int sg   = lr0 >> 2;
        const float* o0 = w_rot + size_t(r4 + 0) * size_t(K);
        const float* o1 = w_rot + size_t(r4 + 1) * size_t(K);
        const float* o2 = w_rot + size_t(r4 + 2) * size_t(K);
        const float* o3 = w_rot + size_t(r4 + 3) * size_t(K);

        if (!use_ldlq) {
            for (int col = 0; col < K; ++col) {
                TileT& tile = q.tiles[size_t(nt_i) * size_t(q.kt) + size_t(col >> 5)];
                const float sigma = fp16_to_f32(tile.scales_f16[sg]);
                const double inv_s = (sigma > 0.0f) ? (1.0 / double(sigma)) : 0.0;
                const int lc = col & 31;
                set_cell_fn(tile, lr0 + 0, lc, quant_mag_fn(double(std::fabs(o0[col])) * inv_s, sol), std::signbit(o0[col]) ? 1u : 0u);
                set_cell_fn(tile, lr0 + 1, lc, quant_mag_fn(double(std::fabs(o1[col])) * inv_s, sol), std::signbit(o1[col]) ? 1u : 0u);
                set_cell_fn(tile, lr0 + 2, lc, quant_mag_fn(double(std::fabs(o2[col])) * inv_s, sol), std::signbit(o2[col]) ? 1u : 0u);
                set_cell_fn(tile, lr0 + 3, lc, quant_mag_fn(double(std::fabs(o3[col])) * inv_s, sol), std::signbit(o3[col]) ? 1u : 0u);
            }
            continue;
        }

        std::vector<float> w0(o0, o0 + K), w1(o1, o1 + K), w2(o2, o2 + K), w3(o3, o3 + K);
        for (int col = 0; col < K; ++col) {
            TileT& tile = q.tiles[size_t(nt_i) * size_t(q.kt) + size_t(col >> 5)];
            const float sigma = fp16_to_f32(tile.scales_f16[sg]);
            const double inv_s = (sigma > 0.0f) ? (1.0 / double(sigma)) : 0.0;
            const int lc = col & 31;

            auto quant1 = [&](float wv, float orig, int local_r) -> float {
                const uint32_t mag = quant_mag_fn(double(std::fabs(wv)) * inv_s, sol);
                const uint32_t sgn = std::signbit(wv) ? 1u : 0u;
                set_cell_fn(tile, local_r, lc, mag, sgn);
                const float v = float(sol.levels[mag]) * sigma;
                return orig - (sgn ? -v : v);
            };
            const float e0 = quant1(w0[size_t(col)], o0[col], lr0 + 0);
            const float e1 = quant1(w1[size_t(col)], o1[col], lr0 + 1);
            const float e2 = quant1(w2[size_t(col)], o2[col], lr0 + 2);
            const float e3 = quant1(w3[size_t(col)], o3[col], lr0 + 3);

            if (col + 1 < K) {
                ldlq_axpy4_avx2(w0.data(), w1.data(), w2.data(), w3.data(),
                                e0, e1, e2, e3, R_up.data() + size_t(col) * size_t(K), col + 1, K);
            }
        }
    }
    return q;
}

inline Hlut2QuantMatrix pack_hlut2_wrot_ldlq(const float* w_rot, int N, int K, const std::vector<float>& R_up) {
    return pack_hlut_from_wrot_ldlq<Hlut2Tile32x32>(w_rot, N, K, lloyd_max_hlut2(), R_up, hlut2_quant_mag, hlut2_set_cell);
}
inline Hlut3QuantMatrix pack_hlut3_wrot_ldlq(const float* w_rot, int N, int K, const std::vector<float>& R_up) {
    return pack_hlut_from_wrot_ldlq<Hlut3Tile32x32>(w_rot, N, K, lloyd_max_hlut3(), R_up, hlut3_quant_mag, hlut3_set_cell);
}
inline Hlut4QuantMatrix pack_hlut4_wrot_ldlq(const float* w_rot, int N, int K, const std::vector<float>& R_up) {
    return pack_hlut_from_wrot_ldlq<Hlut4Tile32x32>(w_rot, N, K, lloyd_max_hlut4(), R_up, hlut4_quant_mag, hlut4_set_cell);
}

// Compute exact weight-Frobenius SQNR (dB) of packed HLUT{2,3,4} vs w_rot
template <typename TileT, typename UnpackFn>
inline double compute_hlut_sqnr_db(const HlutQuantMatrix<TileT>& q, const float* w_rot, UnpackFn&& unpack_fn) {
    double sig = 0.0, err = 0.0;
    #pragma omp parallel for reduction(+:sig,err) schedule(static)
    for (int nt_i = 0; nt_i < q.nt; ++nt_i) {
        float tmp[32 * 32];
        for (int kt_i = 0; kt_i < q.kt; ++kt_i) {
            unpack_fn(q.tile_at(nt_i, kt_i), q.lloyd, tmp, 32, 0, 0);
            for (int r = 0; r < 32; ++r) {
                const float* wr = w_rot + size_t(nt_i * 32 + r) * size_t(q.K) + size_t(kt_i * 32);
                for (int c = 0; c < 32; ++c) {
                    const double a = wr[c], d = a - double(tmp[r * 32 + c]);
                    sig += a * a; err += d * d;
                }
            }
        }
    }
    return 10.0 * std::log10((sig + 1e-30) / (err + 1e-30));
}

// ---- P150A TP2 128x128 Superblock Packer (HLUT2 4,352 B & HLUT3 6,400 B) ----
template <typename SuperT, typename TileT>
inline std::vector<SuperT> pack_for_p150_shard(const HlutQuantMatrix<TileT>& q, int n_col_offset, int N_shard) {
    const int nt4 = N_shard / 128, kt4 = q.K / 128, n_tile_off = n_col_offset / 32;
    std::vector<SuperT> out(size_t(nt4) * size_t(kt4));
    for (int nb4 = 0; nb4 < nt4; ++nb4) {
        for (int kb4 = 0; kb4 < kt4; ++kb4) {
            SuperT& sb = out[size_t(nb4) * size_t(kt4) + size_t(kb4)];
            for (int ns = 0; ns < 4; ++ns)
                for (int ks = 0; ks < 4; ++ks)
                    sb.subtiles[ns * 4 + ks] = q.tile_at(n_tile_off + nb4 * 4 + ns, kb4 * 4 + ks);
        }
    }
    return out;
}

}  // namespace hy4_770b
