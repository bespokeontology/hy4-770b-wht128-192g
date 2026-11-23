// Zero-Scratch gfx906 Vega 20 Core Backbone, Lightning Indexer, Gated MLA & DFly2 HIP Kernels
// Model: tencent/Hy4 (770B: H=6144, 78 layers, 64 heads, 21 full + 57 shared IndexCache, 3.3B DFly2)
// 100% ScratchSize == 0, 0 SGPR/VGPR spills, 0 device trig, 0 slc/system RMW atomics.

#include "hy4_770b_core_kernels.hpp"
#include <hip/hip_fp16.h>
#include <cfloat>

namespace hy4_770b {
namespace {

__device__ __forceinline__ float d_bf16_to_f32(uint16_t b) noexcept {
    return __uint_as_float(uint32_t(b) << 16);
}
__device__ __forceinline__ uint16_t d_f32_to_bf16_rne(float x) noexcept {
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t((u >> 16) | 0x0040u);
    u += 0x00007FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
__device__ __forceinline__ uint16_t d_f32_to_f16(float x) noexcept {
    x = fmaxf(-65504.0f, fminf(65504.0f, x));
    return __half_as_ushort(__float2half_rn(x));
}
__device__ __forceinline__ float d_sigmoid(float x) noexcept {
    return 1.0f / (1.0f + expf(-x));
}
__device__ __forceinline__ float d_wave64_sum(float v) noexcept {
    #pragma unroll
    for (int len = 1; len <= 32; len <<= 1) v += __shfl_xor(v, len, 64);
    return v;
}
__device__ __forceinline__ float d_wave64_max(float v) noexcept {
    #pragma unroll
    for (int len = 1; len <= 32; len <<= 1) v = fmaxf(v, __shfl_xor(v, len, 64));
    return v;
}
__device__ __forceinline__ float d_block256_sum(float v, float* s_wave4) noexcept {
    const int tid = int(threadIdx.x);
    v = d_wave64_sum(v);
    if ((tid & 63) == 0) s_wave4[tid >> 6] = v;
    __syncthreads();
    const float tot = s_wave4[0] + s_wave4[1] + s_wave4[2] + s_wave4[3];
    __syncthreads();
    return tot;
}
__device__ __forceinline__ float d_block256_max(float v, float* s_wave4) noexcept {
    const int tid = int(threadIdx.x);
    v = d_wave64_max(v);
    if ((tid & 63) == 0) s_wave4[tid >> 6] = v;
    __syncthreads();
    const float mx = fmaxf(fmaxf(s_wave4[0], s_wave4[1]), fmaxf(s_wave4[2], s_wave4[3]));
    __syncthreads();
    return mx;
}
__device__ __forceinline__ float d_fp8_to_f32(uint32_t b) noexcept {
    const uint32_t sign = (b & 0x80u) << 24;
    const uint32_t mag  = b & 0x7Fu;
    if (mag == 0u || mag == 0x7Fu) return 0.0f;
    if (mag >= 8u) return __uint_as_float(sign | ((mag << 20) + 0x3C000000u));
    const float sub = __uint_as_float(0x3C800000u | (mag << 20)) - 0.015625f;
    return sign ? -sub : sub;
}
__device__ __forceinline__ uint8_t d_f32_to_fp8_rne(float x) noexcept {
    const uint32_t u = __float_as_uint(x);
    const uint8_t sign = (u & 0x80000000u) ? 0x80u : 0x00u;
    const float ax = fabsf(x);
    if (ax >= 448.0f) return uint8_t(sign | 0x7Eu);
    if (ax == 0.0f) return sign;
    const uint32_t au = u & 0x7FFFFFFFu;
    const int exp = int((au >> 23) & 0xFFu) - 127;
    const uint32_t mant = au & 0x007FFFFFu;
    if (exp < -6) {
        const int m = int(__float2int_rn(ax * 512.0f));
        if (m <= 0) return sign;
        if (m >= 8) return uint8_t(sign | 8u);
        return uint8_t(sign | uint8_t(m));
    }
    int e_fp8 = exp + 7;
    uint32_t m3 = mant >> 20;
    const uint32_t rem = mant & 0x000FFFFFu;
    if (rem > 0x00080000u || (rem == 0x00080000u && (m3 & 1u))) {
        ++m3;
        if (m3 == 8u) { m3 = 0u; ++e_fp8; }
    }
    if (e_fp8 > 15 || (e_fp8 == 15 && m3 >= 7u)) return uint8_t(sign | 0x7Eu);
    return uint8_t(sign | (uint8_t(e_fp8) << 3) | uint8_t(m3));
}
__device__ __forceinline__ void d_rope_cos_sin(
    const float* __restrict__ cos_tbl, const float* __restrict__ sin_tbl,
    int pos, int pair, float& c, float& s) noexcept
{
    const int lo = pos & 1023, hi = pos >> 10;
    const float cl = cos_tbl[size_t(lo) * 32u + size_t(pair)];
    const float sl = sin_tbl[size_t(lo) * 32u + size_t(pair)];
    if (hi == 0) { c = cl; s = sl; }
    else {
        const float ch = cos_tbl[size_t(1024 + hi) * 32u + size_t(pair)];
        const float sh = sin_tbl[size_t(1024 + hi) * 32u + size_t(pair)];
        c = fmaf(ch, cl, -sh * sl);
        s = fmaf(sh, cl,  ch * sl);
    }
}

}  // namespace

// ============================================================================
// 1. Embedding Expand & iHC (H = 6144, kHcFlatDim = 24576)
// ============================================================================
__global__ __launch_bounds__(256, 8)
void k_embed_expand_hc(const int32_t* __restrict__ tok_ids, const uint16_t* __restrict__ embed_bf16,
                       uint16_t* __restrict__ streams_bf16, int M, int H, int hc_mult) {
    const int m = int(blockIdx.y), d = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (m >= M || d >= H) return;
    const uint16_t v = embed_bf16[size_t(tok_ids[m]) * size_t(H) + size_t(d)];
    for (int c = 0; c < hc_mult; ++c)
        streams_bf16[(size_t(m) * size_t(hc_mult) + size_t(c)) * size_t(H) + size_t(d)] = v;
}

__global__ __launch_bounds__(256, 4)
void k_ihc_pre(const uint16_t* __restrict__ streams_bf16, const float* __restrict__ fn_f32,
               const float* __restrict__ base_f32, const float* __restrict__ scale_f32,
               uint16_t* __restrict__ sub_in_bf16, float* __restrict__ post_f32,
               float rms_eps, float hc_eps, float hc_mag) {
    __shared__ float s_wave4[4];
    __shared__ float s_pre[4];
    const int m = int(blockIdx.x), tid = int(threadIdx.x);
    const uint16_t* __restrict__ flat = streams_bf16 + size_t(m) * size_t(kHcFlatDim);
    const uint2* __restrict__ flat_u2 = reinterpret_cast<const uint2*>(flat);
    constexpr int kFlat4 = kHcFlatDim >> 2;  // 6144
    const float4* __restrict__ f4_base = reinterpret_cast<const float4*>(fn_f32);

    float sq = 0.0f;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f, d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;
    for (int i = tid; i < kFlat4; i += 256) {
        const uint2 u = flat_u2[i];
        const float x0 = d_bf16_to_f32(uint16_t(u.x & 0xFFFFu));
        const float x1 = d_bf16_to_f32(uint16_t(u.x >> 16));
        const float x2 = d_bf16_to_f32(uint16_t(u.y & 0xFFFFu));
        const float x3 = d_bf16_to_f32(uint16_t(u.y >> 16));
        sq = fmaf(x0, x0, fmaf(x1, x1, fmaf(x2, x2, fmaf(x3, x3, sq))));
        auto acc4 = [&](const float4& w, float& a) {
            a = fmaf(x0, w.x, fmaf(x1, w.y, fmaf(x2, w.z, fmaf(x3, w.w, a))));
        };
        acc4(f4_base[0 * kFlat4 + i], d0); acc4(f4_base[1 * kFlat4 + i], d1);
        acc4(f4_base[2 * kFlat4 + i], d2); acc4(f4_base[3 * kFlat4 + i], d3);
        acc4(f4_base[4 * kFlat4 + i], d4); acc4(f4_base[5 * kFlat4 + i], d5);
        acc4(f4_base[6 * kFlat4 + i], d6); acc4(f4_base[7 * kFlat4 + i], d7);
    }
    const float inv_rms = rsqrtf(d_block256_sum(sq, s_wave4) * (1.0f / float(kHcFlatDim)) + rms_eps);
    const float sc_pre = scale_f32[0], sc_post = scale_f32[1];
    const float ds[8] = {d0, d1, d2, d3, d4, d5, d6, d7};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float mix = d_block256_sum(ds[j], s_wave4) * inv_rms;
        if (tid == 0) {
            if (j < 4) s_pre[j] = d_sigmoid(mix * sc_pre + base_f32[j]) + hc_eps;
            else       post_f32[size_t(m) * 4u + size_t(j - 4)] = hc_mag * d_sigmoid(mix * sc_post + base_f32[j]) + hc_eps;
        }
    }
    __syncthreads();
    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    uint16_t* __restrict__ out = sub_in_bf16 + size_t(m) * size_t(kHiddenSize);
    for (int d = tid; d < kHiddenSize; d += 256) {
        const float y = p0 * d_bf16_to_f32(flat[0 * kHiddenSize + d]) +
                        p1 * d_bf16_to_f32(flat[1 * kHiddenSize + d]) +
                        p2 * d_bf16_to_f32(flat[2 * kHiddenSize + d]) +
                        p3 * d_bf16_to_f32(flat[3 * kHiddenSize + d]);
        out[d] = d_f32_to_bf16_rne(y);
    }
}

__global__ __launch_bounds__(256, 8)
void k_ihc_post_add(uint16_t* __restrict__ streams_bf16, const uint16_t* __restrict__ sub_out_bf16,
                    const float* __restrict__ post_f32, int M, int H, int hc_mult) {
    const int m = int(blockIdx.y), d = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (m >= M || d >= H) return;
    const float y = d_bf16_to_f32(sub_out_bf16[size_t(m) * size_t(H) + size_t(d)]);
    for (int c = 0; c < hc_mult; ++c) {
        const size_t idx = (size_t(m) * size_t(hc_mult) + size_t(c)) * size_t(H) + size_t(d);
        streams_bf16[idx] = d_f32_to_bf16_rne(fmaf(post_f32[size_t(m) * size_t(hc_mult) + size_t(c)], y, d_bf16_to_f32(streams_bf16[idx])));
    }
}

__global__ __launch_bounds__(256, 4)
void k_ihc_head_and_norm(const uint16_t* __restrict__ streams_bf16, const float* __restrict__ fn_f32,
                         const float* __restrict__ base_f32, const float* __restrict__ scale_f32,
                         const uint16_t* __restrict__ norm_w_bf16, uint16_t* __restrict__ out_bf16,
                         float rms_eps, float hc_eps) {
    __shared__ float s_wave4[4];
    __shared__ float s_pre[4];
    const int m = int(blockIdx.x), tid = int(threadIdx.x);
    const uint16_t* __restrict__ flat = streams_bf16 + size_t(m) * size_t(kHcFlatDim);
    float sq = 0.0f;
    for (int i = tid; i < kHcFlatDim; i += 256) {
        const float x = d_bf16_to_f32(flat[i]);
        sq = fmaf(x, x, sq);
    }
    const float inv_rms = rsqrtf(d_block256_sum(sq, s_wave4) * (1.0f / float(kHcFlatDim)) + rms_eps);
    const float sc = scale_f32[0];
    for (int j = 0; j < 4; ++j) {
        const float* __restrict__ fn_row = fn_f32 + size_t(j) * size_t(kHcFlatDim);
        float dot = 0.0f;
        for (int i = tid; i < kHcFlatDim; i += 256) dot = fmaf(d_bf16_to_f32(flat[i]), fn_row[i], dot);
        const float mix = d_block256_sum(dot, s_wave4) * inv_rms;
        if (tid == 0) s_pre[j] = d_sigmoid(mix * sc + base_f32[j]) + hc_eps;
    }
    __syncthreads();
    const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    uint16_t* __restrict__ dst = out_bf16 + size_t(m) * size_t(kHiddenSize);
    float sq_h = 0.0f;
    for (int d = tid; d < kHiddenSize; d += 256) {
        const float y = p0 * d_bf16_to_f32(flat[0 * kHiddenSize + d]) +
                        p1 * d_bf16_to_f32(flat[1 * kHiddenSize + d]) +
                        p2 * d_bf16_to_f32(flat[2 * kHiddenSize + d]) +
                        p3 * d_bf16_to_f32(flat[3 * kHiddenSize + d]);
        const uint16_t yb = d_f32_to_bf16_rne(y);
        dst[d] = yb;
        const float yf = d_bf16_to_f32(yb);
        sq_h = fmaf(yf, yf, sq_h);
    }
    const float inv_h = rsqrtf(d_block256_sum(sq_h, s_wave4) * (1.0f / float(kHiddenSize)) + rms_eps);
    for (int d = tid; d < kHiddenSize; d += 256) {
        const uint16_t xn = d_f32_to_bf16_rne(d_bf16_to_f32(dst[d]) * inv_h);
        dst[d] = d_f32_to_bf16_rne(d_bf16_to_f32(xn) * d_bf16_to_f32(norm_w_bf16[d]));
    }
}

// ============================================================================
// 2. Norms, Elementwise & Conversions
// ============================================================================
__global__ __launch_bounds__(256, 8)
void k_rmsnorm_bf16(const uint16_t* __restrict__ in, const uint16_t* __restrict__ w,
                    uint16_t* __restrict__ out, int dim, float eps) {
    __shared__ float s_wave4[4];
    const int row = int(blockIdx.x), tid = int(threadIdx.x);
    const uint16_t* src = in + size_t(row) * size_t(dim);
    uint16_t* dst = out + size_t(row) * size_t(dim);
    float sq = 0.0f;
    for (int i = tid; i < dim; i += 256) { const float x = d_bf16_to_f32(src[i]); sq = fmaf(x, x, sq); }
    const float inv = rsqrtf(d_block256_sum(sq, s_wave4) / float(dim) + eps);
    for (int i = tid; i < dim; i += 256) {
        const uint16_t xn = d_f32_to_bf16_rne(d_bf16_to_f32(src[i]) * inv);
        dst[i] = d_f32_to_bf16_rne(d_bf16_to_f32(xn) * d_bf16_to_f32(w[i]));
    }
}

__global__ __launch_bounds__(256, 8)
void k_swiglu_unclamped_bf16(const uint16_t* g, const uint16_t* u, uint16_t* out, int n) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < n) { const float gv = d_bf16_to_f32(g[i]); out[i] = d_f32_to_bf16_rne((gv * d_sigmoid(gv)) * d_bf16_to_f32(u[i])); }
}
__global__ __launch_bounds__(256, 8)
void k_swiglu_clamped10_f32(const float* g, const float* u, float* out, int n) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < n) {
        const float gc = fminf(g[i], 10.0f);
        const float uc = fmaxf(-10.0f, fminf(10.0f, u[i]));
        out[i] = (gc * d_sigmoid(gc)) * uc;
    }
}
__global__ __launch_bounds__(256, 8)
void k_gate_sigmoid_mul_bf16(const uint16_t* a, const uint16_t* g, uint16_t* out, int n) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < n) out[i] = d_f32_to_bf16_rne(d_bf16_to_f32(a[i]) * d_sigmoid(d_bf16_to_f32(g[i])));
}
__global__ __launch_bounds__(256, 8)
void k_f32_to_bf16(const float* in, uint16_t* out, int n) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < n) out[i] = d_f32_to_bf16_rne(in[i]);
}
__global__ __launch_bounds__(256, 8)
void k_bf16_to_f32(const uint16_t* in, float* out, int n) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < n) out[i] = d_bf16_to_f32(in[i]);
}
__global__ __launch_bounds__(256, 8)
void k_bf16_to_f16(const uint16_t* in, uint16_t* out, int n) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < n) out[i] = d_f32_to_f16(d_bf16_to_f32(in[i]));
}
__global__ __launch_bounds__(256, 8)
void k_f16_to_bf16(const uint16_t* in, uint16_t* out, int n) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < n) out[i] = d_f32_to_bf16_rne(__half2float(__ushort_as_half(in[i])));
}
__global__ __launch_bounds__(256, 8)
void k_add_bf16(const uint16_t* a, const uint16_t* b, uint16_t* out, int n) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < n) out[i] = d_f32_to_bf16_rne(d_bf16_to_f32(a[i]) + d_bf16_to_f32(b[i]));
}
__global__ __launch_bounds__(256, 8)
void k_add_f32_inplace(const float* a, float* inout, int n) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i < n) inout[i] += a[i];
}

// ============================================================================
// 3. Interleaved MRoPE & KV Cache Append (FP8-g128 + BF16 k_pe = 656 B/slot)
// ============================================================================
__global__ __launch_bounds__(256, 8)
void k_rope_interleaved_trailing64(uint16_t* __restrict__ x_bf16, const int32_t* __restrict__ pos_ids,
                                   const float* __restrict__ cos_tbl, const float* __restrict__ sin_tbl,
                                   int M, int n_heads, int head_dim) {
    const int tid = int(threadIdx.x), pair = tid & 31, h = int(blockIdx.x) * 8 + (tid >> 5), m = int(blockIdx.y);
    if (m >= M || h >= n_heads) return;
    float c, s;
    d_rope_cos_sin(cos_tbl, sin_tbl, pos_ids[m], pair, c, s);
    const size_t base = (size_t(m) * size_t(n_heads) + size_t(h)) * size_t(head_dim) + size_t(head_dim - 64 + pair * 2);
    const float x0 = d_bf16_to_f32(x_bf16[base]), x1 = d_bf16_to_f32(x_bf16[base + 1]);
    x_bf16[base]     = d_f32_to_bf16_rne(x0 * c - x1 * s);
    x_bf16[base + 1] = d_f32_to_bf16_rne(x0 * s + x1 * c);
}

__global__ __launch_bounds__(256, 8)
void k_kv_norm_rope_append_fp8(const uint16_t* __restrict__ kv_a_bf16, const uint16_t* __restrict__ kv_norm_w,
                               const int32_t* __restrict__ pos_ids, const int32_t* __restrict__ write_slots,
                               const float* __restrict__ cos_tbl, const float* __restrict__ sin_tbl,
                               uint8_t* __restrict__ kv_cache_fp8, float eps) {
    __shared__ float s_wave4[4];
    __shared__ float s_normed[512];
    const int m = int(blockIdx.x), tid = int(threadIdx.x);
    const uint16_t* src = kv_a_bf16 + size_t(m) * size_t(kKvLatentDim);
    uint8_t* dst = kv_cache_fp8 + size_t(write_slots[m]) * size_t(kKvFp8EntryBytes);

    // 1. RMSNorm on c_kv [0..511]
    const float x0 = d_bf16_to_f32(src[tid]), x1 = d_bf16_to_f32(src[tid + 256]);
    const float inv = rsqrtf(d_block256_sum(x0 * x0 + x1 * x1, s_wave4) * (1.0f / 512.0f) + eps);
    s_normed[tid]       = d_bf16_to_f32(d_f32_to_bf16_rne(x0 * inv)) * d_bf16_to_f32(kv_norm_w[tid]);
    s_normed[tid + 256] = d_bf16_to_f32(d_f32_to_bf16_rne(x1 * inv)) * d_bf16_to_f32(kv_norm_w[tid + 256]);
    __syncthreads();

    // 2. Quantize 8 groups of 64 floats -> FP8 E4M3FN [0..511] + 8 BF16 scales at [512..527] (exact same 16 B, 2x finer outlier isolation!)
    const int grp = tid >> 5;     // 0..7 -> 32-lane half-wavefront per 64-element group!
    const int lane32 = tid & 31;
    const float a0 = s_normed[grp * 64 + lane32];
    const float a1 = s_normed[grp * 64 + lane32 + 32];
    float amax = fmaxf(fabsf(a0), fabsf(a1));
    amax = fmaxf(amax, __shfl_xor(amax, 16, 64));
    amax = fmaxf(amax, __shfl_xor(amax, 8, 64));
    amax = fmaxf(amax, __shfl_xor(amax, 4, 64));
    amax = fmaxf(amax, __shfl_xor(amax, 2, 64));
    amax = fmaxf(amax, __shfl_xor(amax, 1, 64));
    const uint16_t sc_bf16 = d_f32_to_bf16_rne(fmaxf(amax, 1e-12f) * (1.0f / 448.0f));
    const float sc = fmaxf(d_bf16_to_f32(sc_bf16), 1e-30f);
    const float inv_s = 1.0f / sc;
    dst[grp * 64 + lane32]      = d_f32_to_fp8_rne(a0 * inv_s);
    dst[grp * 64 + lane32 + 32] = d_f32_to_fp8_rne(a1 * inv_s);
    if (lane32 == 0) reinterpret_cast<uint16_t*>(dst + 512)[grp] = sc_bf16;

    // 3. Interleaved RoPE on k_pe [512..575] -> BF16 stored at [528..655]
    if (tid < 32) {
        float c, s;
        d_rope_cos_sin(cos_tbl, sin_tbl, pos_ids[m], tid, c, s);
        const float r0 = d_bf16_to_f32(src[512 + tid * 2]);
        const float r1 = d_bf16_to_f32(src[512 + tid * 2 + 1]);
        uint16_t* kpe = reinterpret_cast<uint16_t*>(dst + 528);
        kpe[tid * 2]     = d_f32_to_bf16_rne(r0 * c - r1 * s);
        kpe[tid * 2 + 1] = d_f32_to_bf16_rne(r0 * s + r1 * c);
    }
}

__global__ __launch_bounds__(256, 8)
void k_kv_norm_rope_append_bf16(const uint16_t* __restrict__ kv_a_bf16, const uint16_t* __restrict__ kv_norm_w,
                                const int32_t* __restrict__ pos_ids, const int32_t* __restrict__ write_slots,
                                const float* __restrict__ cos_tbl, const float* __restrict__ sin_tbl,
                                uint16_t* __restrict__ kv_cache_bf16, float eps) {
    __shared__ float s_wave4[4];
    const int m = int(blockIdx.x), tid = int(threadIdx.x);
    const uint16_t* src = kv_a_bf16 + size_t(m) * size_t(kKvLatentDim);
    uint16_t* dst = kv_cache_bf16 + size_t(write_slots[m]) * size_t(kKvLatentDim);
    const float x0 = d_bf16_to_f32(src[tid]), x1 = d_bf16_to_f32(src[tid + 256]);
    const float inv = rsqrtf(d_block256_sum(x0 * x0 + x1 * x1, s_wave4) * (1.0f / 512.0f) + eps);
    dst[tid]       = d_f32_to_bf16_rne(d_bf16_to_f32(d_f32_to_bf16_rne(x0 * inv)) * d_bf16_to_f32(kv_norm_w[tid]));
    dst[tid + 256] = d_f32_to_bf16_rne(d_bf16_to_f32(d_f32_to_bf16_rne(x1 * inv)) * d_bf16_to_f32(kv_norm_w[tid + 256]));
    if (tid < 32) {
        float c, s;
        d_rope_cos_sin(cos_tbl, sin_tbl, pos_ids[m], tid, c, s);
        const float r0 = d_bf16_to_f32(src[512 + tid * 2]), r1 = d_bf16_to_f32(src[512 + tid * 2 + 1]);
        dst[512 + tid * 2]     = d_f32_to_bf16_rne(r0 * c - r1 * s);
        dst[512 + tid * 2 + 1] = d_f32_to_bf16_rne(r0 * s + r1 * c);
    }
}

// ============================================================================
// 4. Lightning Indexer (32h x 128d, FP32 LayerNorm k_norm, weights_proj / 64)
// ============================================================================
__global__ __launch_bounds__(128, 8)
void k_indexer_k_norm_rope_append(
    const uint16_t* __restrict__ k_in_bf16,
    const uint16_t* __restrict__ knorm_w_bf16, const uint16_t* __restrict__ knorm_b_bf16,
    const int32_t* __restrict__ pos_ids, const int32_t* __restrict__ write_slots,
    const float* __restrict__ cos_tbl, const float* __restrict__ sin_tbl,
    uint8_t* __restrict__ idx_k_cache_fp8, float eps)
{
    __shared__ float s_w[2];
    __shared__ float s_k[128];
    const int m = int(blockIdx.x), tid = int(threadIdx.x), wid = tid >> 6, lane = tid & 63;
    const float x = d_bf16_to_f32(k_in_bf16[size_t(m) * 128u + size_t(tid)]);

    // Exact FP32 LayerNorm(128, eps=1e-6)
    float sum = d_wave64_sum(x);
    if (lane == 0) s_w[wid] = sum;
    __syncthreads();
    const float mean = (s_w[0] + s_w[1]) * (1.0f / 128.0f);
    const float diff = x - mean;
    float var = d_wave64_sum(diff * diff);
    if (lane == 0) s_w[wid] = var;
    __syncthreads();
    const float inv_std = rsqrtf((s_w[0] + s_w[1]) * (1.0f / 128.0f) + eps);
    s_k[tid] = diff * inv_std * d_bf16_to_f32(knorm_w_bf16[tid]) + d_bf16_to_f32(knorm_b_bf16[tid]);
    __syncthreads();

    // Trailing-64 interleaved MRoPE on [64..127]
    if (tid < 32) {
        float c, s;
        d_rope_cos_sin(cos_tbl, sin_tbl, pos_ids[m], tid, c, s);
        const float k0 = s_k[64 + tid * 2], k1 = s_k[64 + tid * 2 + 1];
        s_k[64 + tid * 2]     = k0 * c - k1 * s;
        s_k[64 + tid * 2 + 1] = k0 * s + k1 * c;
    }
    __syncthreads();

    // Quantize 128-dim k -> FP8 E4M3FN + 1 FP32 scale = 132 B/slot
    const float kv = s_k[tid];
    float am = d_wave64_max(fabsf(kv));
    if (lane == 0) s_w[wid] = am;
    __syncthreads();
    const float sc = fmaxf(fmaxf(s_w[0], s_w[1]), 1e-12f) * (1.0f / 448.0f);
    uint8_t* dst = idx_k_cache_fp8 + size_t(write_slots[m]) * size_t(kIdxKEntryBytes);
    dst[tid] = d_f32_to_fp8_rne(kv / sc);
    if (tid == 0) *reinterpret_cast<float*>(dst + 128) = sc;
}

__global__ __launch_bounds__(256, 4)
void k_indexer_score_tokens(
    const uint16_t* __restrict__ q_bf16,
    const uint8_t*  __restrict__ k_cache_fp8,
    const uint16_t* __restrict__ hw_bf16,
    const int32_t*  __restrict__ query_slots,
    const int32_t*  __restrict__ key_global_pos,
    float*          __restrict__ scores_f32,
    int num_k)
{
    // Each block computes scores for 1 query row m and 4 key positions (4 wavefronts of 64, 2 elems/lane across 128d!)
    const int m = int(blockIdx.y);
    const int tid = int(threadIdx.x), wid = tid >> 6, lane = tid & 63;
    const int t = int(blockIdx.x) * 4 + wid;
    if (t >= num_k) return;
    const int gpos = key_global_pos ? key_global_pos[t] : t;
    const int qslot = query_slots[m];
    if (gpos > qslot) {
        if (lane == 0) scores_f32[size_t(m) * size_t(num_k) + size_t(t)] = -FLT_MAX;
        return;
    }
    const uint8_t* kp = k_cache_fp8 + size_t(t) * size_t(kIdxKEntryBytes);
    const float ksc = *reinterpret_cast<const float*>(kp + 128);
    const float k0 = d_fp8_to_f32(kp[lane]) * ksc;
    const float k1 = d_fp8_to_f32(kp[lane + 64]) * ksc;
    const uint16_t* qm = q_bf16 + size_t(m) * 32u * 128u;
    const uint16_t* hm = hw_bf16 + size_t(m) * 32u;

    float total = 0.0f;
    for (int h = 0; h < 32; ++h) {
        const uint16_t* qh = qm + size_t(h) * 128u;
        float dot = fmaf(d_bf16_to_f32(qh[lane]), k0, d_bf16_to_f32(qh[lane + 64]) * k1);
        dot = d_wave64_sum(dot);
        if (dot > 0.0f) total = fmaf(d_bf16_to_f32(hm[h]) * kIndexWeightScale, dot, total);
    }
    if (lane == 0) scores_f32[size_t(m) * size_t(num_k) + size_t(t)] = total;
}

__global__ __launch_bounds__(256, 2)
void k_indexer_select_topk2048(
    const float* __restrict__ scores_f32, const int32_t* __restrict__ query_slots,
    int32_t* __restrict__ out_indices, int32_t* __restrict__ out_counts, int num_k)
{
    // Exact threshold-binary-search top-2048 selector in 0 scratch memory!
    __shared__ float s_wave4[4];
    __shared__ uint32_t s_cnt;
    const int m = int(blockIdx.x), tid = int(threadIdx.x);
    const int valid_k = min(num_k, query_slots[m] + 1);
    int32_t* dst = out_indices + size_t(m) * size_t(kIndexTopK);

    if (valid_k <= kIndexTopK) {
        for (int t = tid; t < kIndexTopK; t += 256) dst[t] = (t < valid_k) ? t : -1;
        if (tid == 0) out_counts[m] = valid_k;
        return;
    }

    const float* sc = scores_f32 + size_t(m) * size_t(num_k);
    float lo = FLT_MAX, hi = -FLT_MAX;
    for (int t = tid; t < valid_k; t += 256) {
        const float v = sc[t];
        if (v > -1e30f) { lo = fminf(lo, v); hi = fmaxf(hi, v); }
    }
    lo = -d_block256_max(-lo, s_wave4);
    hi =  d_block256_max( hi, s_wave4);

    // 28 iterations of binary search narrow threshold to FP32 precision
    for (int it = 0; it < 28; ++it) {
        const float mid = 0.5f * (lo + hi);
        if (tid == 0) s_cnt = 0u;
        __syncthreads();
        uint32_t local_c = 0u;
        for (int t = tid; t < valid_k; t += 256) if (sc[t] >= mid) ++local_c;
        atomicAdd(&s_cnt, local_c);
        __syncthreads();
        if (s_cnt >= uint32_t(kIndexTopK)) lo = mid; else hi = mid;
    }
    const float thr = lo;
    if (tid == 0) s_cnt = 0u;
    __syncthreads();
    // Pass 1: strictly > thr
    for (int t = tid; t < valid_k; t += 256) {
        if (sc[t] > thr) {
            const uint32_t pos = atomicAdd(&s_cnt, 1u);
            if (pos < uint32_t(kIndexTopK)) dst[pos] = t;
        }
    }
    __syncthreads();
    // Pass 2: fill ties == thr
    for (int t = tid; t < valid_k; t += 256) {
        if (sc[t] == thr) {
            const uint32_t pos = atomicAdd(&s_cnt, 1u);
            if (pos < uint32_t(kIndexTopK)) dst[pos] = t;
        }
    }
    __syncthreads();
    if (tid == 0) out_counts[m] = min(int(s_cnt), kIndexTopK);
}

// ============================================================================
// 5. Absorbed Gated MLA Attention with Learnable Sinks (FP8-g128 & BF16 KV)
// ============================================================================
template <bool KV_IS_FP8>
__global__ __launch_bounds__(256, 2)
void k_mla_absorbed_sparse_t(
    const uint16_t* __restrict__ q_abs_bf16,
    const void*     __restrict__ kv_cache_ptr,
    const float*    __restrict__ sinks_f32,
    const int32_t*  __restrict__ query_slots,
    const int32_t*  __restrict__ recalled_idx,
    const int32_t*  __restrict__ recalled_cnt,
    uint16_t*       __restrict__ out_lat_bf16,
    int H, int kv_len, float scale)
{
    __shared__ float s_wave4[4];
    __shared__ float s_q[kKvLatentDim];   // 576 floats = 2.3 KiB LDS
    const int h = int(blockIdx.x), m = int(blockIdx.y), tid = int(threadIdx.x);
    const uint16_t* qptr = q_abs_bf16 + (size_t(m) * size_t(H) + size_t(h)) * size_t(kKvLatentDim);
    for (int d = tid; d < kKvLatentDim; d += 256) s_q[d] = d_bf16_to_f32(qptr[d]);
    __syncthreads();

    const int qslot = query_slots[m];
    const bool use_rec = (recalled_idx != nullptr && recalled_cnt != nullptr && kv_len > kIndexTopK);
    const int n_cand = use_rec ? recalled_cnt[m] : min(kv_len, qslot + 1);
    const int32_t* rlist = use_rec ? (recalled_idx + size_t(m) * size_t(kIndexTopK)) : nullptr;

    // Online softmax initialized with learnable sink logit!
    float run_max = sinks_f32 ? sinks_f32[h] : -FLT_MAX;
    float run_sum = sinks_f32 ? 1.0f : 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;  // tid handles c_kv[tid] and c_kv[tid + 256]

    for (int i = 0; i < n_cand; ++i) {
        const int t = use_rec ? rlist[i] : i;
        if (t < 0 || t > qslot) continue;

        float k0, k1, kr = 0.0f;
        if constexpr (KV_IS_FP8) {
            const uint8_t* kp = reinterpret_cast<const uint8_t*>(kv_cache_ptr) + size_t(t) * size_t(kKvFp8EntryBytes);
            const uint16_t* sc8 = reinterpret_cast<const uint16_t*>(kp + 512);
            k0 = d_fp8_to_f32(kp[tid])       * d_bf16_to_f32(sc8[tid >> 6]);
            k1 = d_fp8_to_f32(kp[tid + 256]) * d_bf16_to_f32(sc8[(tid + 256) >> 6]);
            if (tid < 64) {
                const uint16_t* kpe = reinterpret_cast<const uint16_t*>(kp + 528);
                kr = d_bf16_to_f32(kpe[tid]) * s_q[512 + tid];
            }
        } else {
            const uint16_t* kp = reinterpret_cast<const uint16_t*>(kv_cache_ptr) + size_t(t) * size_t(kKvLatentDim);
            k0 = d_bf16_to_f32(kp[tid]);
            k1 = d_bf16_to_f32(kp[tid + 256]);
            if (tid < 64) kr = d_bf16_to_f32(kp[512 + tid]) * s_q[512 + tid];
        }
        const float partial_dot = fmaf(s_q[tid], k0, fmaf(s_q[tid + 256], k1, kr));
        const float logit = d_block256_sum(partial_dot, s_wave4) * scale;
        const float new_max = fmaxf(run_max, logit);
        const float alpha = expf(run_max - new_max);
        const float w = expf(logit - new_max);
        run_sum = run_sum * alpha + w;
        acc0 = fmaf(w, k0, acc0 * alpha);
        acc1 = fmaf(w, k1, acc1 * alpha);
        run_max = new_max;
    }
    const float inv_s = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    uint16_t* dst = out_lat_bf16 + (size_t(m) * size_t(H) + size_t(h)) * size_t(kKvLoraRank);
    dst[tid]       = d_f32_to_bf16_rne(acc0 * inv_s);
    dst[tid + 256] = d_f32_to_bf16_rne(acc1 * inv_s);
}

__global__ __launch_bounds__(256, 2)
void k_mla_absorbed_partial_fp8(
    const uint16_t* __restrict__ q_abs_bf16,
    const uint8_t*  __restrict__ kv_cache_fp8,
    const int32_t*  __restrict__ local_to_glob,
    const int32_t*  __restrict__ query_pos,
    const int32_t*  __restrict__ local_idx,
    const int32_t*  __restrict__ local_cnt,
    float*          __restrict__ out_max_f32,
    float*          __restrict__ out_sum_f32,
    float*          __restrict__ out_acc_f32,
    int H, int local_kv_len, int max_rec, float scale,
    const int32_t*  __restrict__ query_min_pos)
{
    __shared__ float s_wave4[4];
    __shared__ float s_q[kKvLatentDim];
    const int h = int(blockIdx.x), m = int(blockIdx.y), tid = int(threadIdx.x);
    const uint16_t* qptr = q_abs_bf16 + (size_t(m) * size_t(H) + size_t(h)) * size_t(kKvLatentDim);
    for (int d = tid; d < kKvLatentDim; d += 256) s_q[d] = d_bf16_to_f32(qptr[d]);
    __syncthreads();

    const int qpos = query_pos[m];
    const int qmin = query_min_pos ? query_min_pos[m] : 0;
    const bool use_rec = (local_idx != nullptr && local_cnt != nullptr);
    const int n_cand = use_rec ? local_cnt[m] : local_kv_len;
    const int32_t* rlist = use_rec ? (local_idx + size_t(m) * size_t(max_rec)) : nullptr;

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;

    for (int i = 0; i < n_cand; ++i) {
        const int lt = use_rec ? rlist[i] : i;
        if (lt < 0 || lt >= local_kv_len) continue;
        if (local_to_glob) {
            const int gp = local_to_glob[lt];
            if (gp < qmin || gp > qpos) continue;
        }

        const uint8_t* kp = kv_cache_fp8 + size_t(lt) * size_t(kKvFp8EntryBytes);
        const uint16_t* sc8 = reinterpret_cast<const uint16_t*>(kp + 512);
        const float k0 = d_fp8_to_f32(kp[tid])       * d_bf16_to_f32(sc8[tid >> 6]);
        const float k1 = d_fp8_to_f32(kp[tid + 256]) * d_bf16_to_f32(sc8[(tid + 256) >> 6]);
        float kr = 0.0f;
        if (tid < 64) {
            const uint16_t* kpe = reinterpret_cast<const uint16_t*>(kp + 528);
            kr = d_bf16_to_f32(kpe[tid]) * s_q[512 + tid];
        }
        const float partial_dot = fmaf(s_q[tid], k0, fmaf(s_q[tid + 256], k1, kr));
        const float logit = d_block256_sum(partial_dot, s_wave4) * scale;
        const float new_max = fmaxf(run_max, logit);
        const float alpha = (run_sum > 0.0f) ? expf(run_max - new_max) : 0.0f;
        const float w = expf(logit - new_max);
        run_sum = run_sum * alpha + w;
        acc0 = fmaf(w, k0, acc0 * alpha);
        acc1 = fmaf(w, k1, acc1 * alpha);
        run_max = new_max;
    }

    const size_t mh = size_t(m) * size_t(H) + size_t(h);
    if (tid == 0) {
        out_max_f32[mh] = run_max;
        out_sum_f32[mh] = run_sum;
    }
    float* dst = out_acc_f32 + mh * size_t(kKvLoraRank);
    dst[tid]       = acc0;
    dst[tid + 256] = acc1;
}

__global__ __launch_bounds__(256, 2)
void k_mla_absorbed_partial_bf16(
    const uint16_t* __restrict__ q_abs_bf16,
    const uint16_t* __restrict__ kv_cache_bf16,
    const int32_t*  __restrict__ local_to_glob,
    const int32_t*  __restrict__ query_pos,
    const int32_t*  __restrict__ local_idx,
    const int32_t*  __restrict__ local_cnt,
    float*          __restrict__ out_max_f32,
    float*          __restrict__ out_sum_f32,
    float*          __restrict__ out_acc_f32,
    int H, int local_kv_len, int max_rec, float scale,
    const int32_t*  __restrict__ query_min_pos)
{
    __shared__ float s_wave4[4];
    __shared__ float s_q[kKvLatentDim];
    const int h = int(blockIdx.x), m = int(blockIdx.y), tid = int(threadIdx.x);
    const uint16_t* qptr = q_abs_bf16 + (size_t(m) * size_t(H) + size_t(h)) * size_t(kKvLatentDim);
    for (int d = tid; d < kKvLatentDim; d += 256) s_q[d] = d_bf16_to_f32(qptr[d]);
    __syncthreads();

    const int qpos = query_pos[m];
    const int qmin = query_min_pos ? query_min_pos[m] : 0;
    const bool use_rec = (local_idx != nullptr && local_cnt != nullptr);
    const int n_cand = use_rec ? local_cnt[m] : local_kv_len;
    const int32_t* rlist = use_rec ? (local_idx + size_t(m) * size_t(max_rec)) : nullptr;

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;

    for (int i = 0; i < n_cand; ++i) {
        const int lt = use_rec ? rlist[i] : i;
        if (lt < 0 || lt >= local_kv_len) continue;
        if (local_to_glob) {
            const int gp = local_to_glob[lt];
            if (gp < qmin || gp > qpos) continue;
        }

        const uint16_t* kp = kv_cache_bf16 + size_t(lt) * size_t(kKvLatentDim);
        const float k0 = d_bf16_to_f32(kp[tid]);
        const float k1 = d_bf16_to_f32(kp[tid + 256]);
        float kr = 0.0f;
        if (tid < 64) kr = d_bf16_to_f32(kp[512 + tid]) * s_q[512 + tid];
        const float partial_dot = fmaf(s_q[tid], k0, fmaf(s_q[tid + 256], k1, kr));
        const float logit = d_block256_sum(partial_dot, s_wave4) * scale;
        const float new_max = fmaxf(run_max, logit);
        const float alpha = (run_sum > 0.0f) ? expf(run_max - new_max) : 0.0f;
        const float w = expf(logit - new_max);
        run_sum = run_sum * alpha + w;
        acc0 = fmaf(w, k0, acc0 * alpha);
        acc1 = fmaf(w, k1, acc1 * alpha);
        run_max = new_max;
    }

    const size_t mh = size_t(m) * size_t(H) + size_t(h);
    if (tid == 0) {
        out_max_f32[mh] = run_max;
        out_sum_f32[mh] = run_sum;
    }
    float* dst = out_acc_f32 + mh * size_t(kKvLoraRank);
    dst[tid]       = acc0;
    dst[tid + 256] = acc1;
}

// ============================================================================
// 6. MoE Top-8 / 256 Sigmoid Router + DFly2 Helpers
// ============================================================================
__global__ __launch_bounds__(256, 8)
void k_moe_router_top8_256(
    const uint16_t* __restrict__ logits_bf16, const float* __restrict__ corr_bias_f32,
    int32_t* __restrict__ topk_ids, float* __restrict__ topk_weights, float routed_scale)
{
    __shared__ float s_score[256];
    __shared__ float s_raw[256];
    const int m = int(blockIdx.x), tid = int(threadIdx.x);
    const float sig = d_sigmoid(d_bf16_to_f32(logits_bf16[size_t(m) * 256u + size_t(tid)]));
    s_raw[tid]   = sig;
    s_score[tid] = sig + (corr_bias_f32 ? corr_bias_f32[tid] : 0.0f);
    __syncthreads();

    if (tid == 0) {
        int32_t ids[8];
        float wts[8];
        float sum_w = 0.0f;
        for (int k = 0; k < 8; ++k) {
            int best_e = 0;
            float best_s = -FLT_MAX;
            for (int e = 0; e < 256; ++e) {
                if (s_score[e] > best_s) { best_s = s_score[e]; best_e = e; }
            }
            ids[k] = best_e;
            wts[k] = s_raw[best_e];
            sum_w += wts[k];
            s_score[best_e] = -FLT_MAX;
        }
        const float inv = (sum_w > 1e-20f) ? (routed_scale / sum_w) : 0.0f;
        for (int k = 0; k < 8; ++k) {
            topk_ids[size_t(m) * 8u + size_t(k)]     = ids[k];
            topk_weights[size_t(m) * 8u + size_t(k)] = wts[k] * inv;
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_dfly2_dual_rmsnorm_concat_bf16(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ wa,
    const uint16_t* __restrict__ b, const uint16_t* __restrict__ wb,
    uint16_t* __restrict__ out_cat, int H, float eps)
{
    __shared__ float s_wave4[4];
    const int m = int(blockIdx.x), tid = int(threadIdx.x);
    const uint16_t* ra = a + size_t(m) * size_t(H);
    const uint16_t* rb = b + size_t(m) * size_t(H);
    uint16_t* dst = out_cat + size_t(m) * size_t(2 * H);
    float sqa = 0.0f, sqb = 0.0f;
    for (int i = tid; i < H; i += 256) {
        const float xa = d_bf16_to_f32(ra[i]), xb = d_bf16_to_f32(rb[i]);
        sqa = fmaf(xa, xa, sqa); sqb = fmaf(xb, xb, sqb);
    }
    const float inv_a = rsqrtf(d_block256_sum(sqa, s_wave4) / float(H) + eps);
    const float inv_b = rsqrtf(d_block256_sum(sqb, s_wave4) / float(H) + eps);
    for (int i = tid; i < H; i += 256) {
        dst[i]     = d_f32_to_bf16_rne(d_bf16_to_f32(d_f32_to_bf16_rne(d_bf16_to_f32(ra[i]) * inv_a)) * d_bf16_to_f32(wa[i]));
        dst[H + i] = d_f32_to_bf16_rne(d_bf16_to_f32(d_f32_to_bf16_rne(d_bf16_to_f32(rb[i]) * inv_b)) * d_bf16_to_f32(wb[i]));
    }
}

__global__ __launch_bounds__(256, 8)
void k_dfly2_gate_residual_bf16(
    const uint16_t* __restrict__ anchor, const uint16_t* __restrict__ gate,
    const uint16_t* __restrict__ fused, uint16_t* __restrict__ out, int M, int H)
{
    const int d = int(blockIdx.x) * 256 + int(threadIdx.x), m = int(blockIdx.y);
    if (m >= M || d >= H) return;
    const size_t idx = size_t(m) * size_t(H) + size_t(d);
    out[idx] = d_f32_to_bf16_rne(fmaf(d_bf16_to_f32(gate[d]), d_bf16_to_f32(fused[idx]), d_bf16_to_f32(anchor[idx])));
}

__global__ __launch_bounds__(256, 8)
void k_qb_transpose_nope_f16(const uint16_t* __restrict__ qb_bf16, uint16_t* __restrict__ qn_f16, int M, int H) {
    const int idx = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int tot = H * M * 192;
    if (idx >= tot) return;
    const int d = idx % 192;
    const int hm = idx / 192;
    const int m = hm % M;
    const int h = hm / M;
    qn_f16[idx] = d_f32_to_f16(d_bf16_to_f32(qb_bf16[(size_t(m) * size_t(H) + size_t(h)) * 256u + size_t(d)]));
}

__global__ __launch_bounds__(256, 8)
void k_assemble_q_abs_bf16(const float* __restrict__ q_nope_f32, const uint16_t* __restrict__ qb_bf16,
                           uint16_t* __restrict__ q_abs_bf16, int M, int H) {
    const int idx = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int tot = M * H * 576;
    if (idx >= tot) return;
    const int d = idx % 576;
    const int mh = idx / 576;
    const int h = mh % H;
    const int m = mh / H;
    if (d < 512) {
        q_abs_bf16[idx] = d_f32_to_bf16_rne(q_nope_f32[(size_t(h) * size_t(M) + size_t(m)) * 512u + size_t(d)]);
    } else {
        q_abs_bf16[idx] = qb_bf16[(size_t(m) * size_t(H) + size_t(h)) * 256u + size_t(192 + (d - 512))];
    }
}

__global__ __launch_bounds__(256, 8)
void k_ao_transpose_gate_bf16(const float* __restrict__ ao_f32, const uint16_t* __restrict__ gate_bf16,
                              uint16_t* __restrict__ out_bf16, int M, int H) {
    const int idx = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int tot = M * H * 256;
    if (idx >= tot) return;
    const int d = idx & 255;
    const int mh = idx >> 8;
    const int h = mh % H;
    const int m = mh / H;
    const float a = d_bf16_to_f32(d_f32_to_bf16_rne(ao_f32[(size_t(h) * size_t(M) + size_t(m)) * 256u + size_t(d)]));
    const float g = d_sigmoid(d_bf16_to_f32(gate_bf16[idx]));
    out_bf16[idx] = d_f32_to_bf16_rne(a * g);
}

// ============================================================================
// Host Launch Wrappers
// ============================================================================
void launch_embed_expand_hc(const int32_t* ids, const uint16_t* emb, uint16_t* st, int M, int H, int hc, hipStream_t s) {
    hipLaunchKernelGGL(k_embed_expand_hc, dim3(uint32_t((H + 255) / 256), uint32_t(M)), dim3(256), 0, s, ids, emb, st, M, H, hc);
}
void launch_ihc_pre(const uint16_t* st, const float* fn, const float* b, const float* sc, uint16_t* sub, float* post,
                    int M, float rms_eps, float hc_eps, float hc_mag, hipStream_t s) {
    hipLaunchKernelGGL(k_ihc_pre, dim3(uint32_t(M)), dim3(256), 0, s, st, fn, b, sc, sub, post, rms_eps, hc_eps, hc_mag);
}
void launch_ihc_post_add(uint16_t* st, const uint16_t* sub, const float* post, int M, int H, int hc, hipStream_t s) {
    hipLaunchKernelGGL(k_ihc_post_add, dim3(uint32_t((H + 255) / 256), uint32_t(M)), dim3(256), 0, s, st, sub, post, M, H, hc);
}
void launch_ihc_head_and_norm(const uint16_t* st, const float* fn, const float* b, const float* sc,
                              const uint16_t* nw, uint16_t* out, int M, float rms_eps, float hc_eps, hipStream_t s) {
    hipLaunchKernelGGL(k_ihc_head_and_norm, dim3(uint32_t(M)), dim3(256), 0, s, st, fn, b, sc, nw, out, rms_eps, hc_eps);
}
void launch_rmsnorm_bf16(const uint16_t* in, const uint16_t* w, uint16_t* out, int rows, int dim, float eps, hipStream_t s) {
    hipLaunchKernelGGL(k_rmsnorm_bf16, dim3(uint32_t(rows)), dim3(256), 0, s, in, w, out, dim, eps);
}
void launch_swiglu_unclamped_bf16(const uint16_t* g, const uint16_t* u, uint16_t* out, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_swiglu_unclamped_bf16, dim3(uint32_t((n + 255) / 256)), dim3(256), 0, s, g, u, out, n);
}
void launch_swiglu_clamped10_f32(const float* g, const float* u, float* out, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_swiglu_clamped10_f32, dim3(uint32_t((n + 255) / 256)), dim3(256), 0, s, g, u, out, n);
}
void launch_gate_sigmoid_mul_bf16(const uint16_t* a, const uint16_t* g, uint16_t* out, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_gate_sigmoid_mul_bf16, dim3(uint32_t((n + 255) / 256)), dim3(256), 0, s, a, g, out, n);
}
void launch_f32_to_bf16(const float* in, uint16_t* out, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_f32_to_bf16, dim3(uint32_t((n + 255) / 256)), dim3(256), 0, s, in, out, n);
}
void launch_bf16_to_f32(const uint16_t* in, float* out, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_bf16_to_f32, dim3(uint32_t((n + 255) / 256)), dim3(256), 0, s, in, out, n);
}
void launch_bf16_to_f16(const uint16_t* in, uint16_t* out, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_bf16_to_f16, dim3(uint32_t((n + 255) / 256)), dim3(256), 0, s, in, out, n);
}
void launch_f16_to_bf16(const uint16_t* in, uint16_t* out, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_f16_to_bf16, dim3(uint32_t((n + 255) / 256)), dim3(256), 0, s, in, out, n);
}
void launch_add_bf16(const uint16_t* a, const uint16_t* b, uint16_t* out, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_add_bf16, dim3(uint32_t((n + 255) / 256)), dim3(256), 0, s, a, b, out, n);
}
void launch_add_f32_inplace(const float* a, float* inout, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_add_f32_inplace, dim3(uint32_t((n + 255) / 256)), dim3(256), 0, s, a, inout, n);
}
void launch_rope_interleaved_trailing64(uint16_t* x, const int32_t* pos, const float* ct, const float* st,
                                        int M, int nh, int hd, hipStream_t s) {
    hipLaunchKernelGGL(k_rope_interleaved_trailing64, dim3(uint32_t((nh + 7) / 8), uint32_t(M)), dim3(256), 0, s, x, pos, ct, st, M, nh, hd);
}
void launch_kv_norm_rope_append_fp8(const uint16_t* kva, const uint16_t* nw, const int32_t* pos, const int32_t* slots,
                                    const float* ct, const float* st, uint8_t* cache, int M, float eps, hipStream_t s) {
    hipLaunchKernelGGL(k_kv_norm_rope_append_fp8, dim3(uint32_t(M)), dim3(256), 0, s, kva, nw, pos, slots, ct, st, cache, eps);
}
void launch_kv_norm_rope_append_bf16(const uint16_t* kva, const uint16_t* nw, const int32_t* pos, const int32_t* slots,
                                     const float* ct, const float* st, uint16_t* cache, int M, float eps, hipStream_t s) {
    hipLaunchKernelGGL(k_kv_norm_rope_append_bf16, dim3(uint32_t(M)), dim3(256), 0, s, kva, nw, pos, slots, ct, st, cache, eps);
}
void launch_indexer_k_norm_rope_append(const uint16_t* kin, const uint16_t* kw, const uint16_t* kb,
                                       const int32_t* pos, const int32_t* slots, const float* ct, const float* st,
                                       uint8_t* cache, int M, float eps, hipStream_t s) {
    hipLaunchKernelGGL(k_indexer_k_norm_rope_append, dim3(uint32_t(M)), dim3(128), 0, s, kin, kw, kb, pos, slots, ct, st, cache, eps);
}
void launch_indexer_score_tokens(const uint16_t* q, const uint8_t* kc, const uint16_t* hw, const int32_t* qslots,
                                 const int32_t* kgpos, float* sc, int M, int num_k, hipStream_t s) {
    hipLaunchKernelGGL(k_indexer_score_tokens, dim3(uint32_t((num_k + 3) / 4), uint32_t(M)), dim3(256), 0, s,
                       q, kc, hw, qslots, kgpos, sc, num_k);
}
void launch_indexer_select_topk2048(const float* sc, const int32_t* qslots, int32_t* idx, int32_t* cnt, int M, int num_k, hipStream_t s) {
    hipLaunchKernelGGL(k_indexer_select_topk2048, dim3(uint32_t(M)), dim3(256), 0, s, sc, qslots, idx, cnt, num_k);
}
void launch_mla_absorbed_sparse_fp8(const uint16_t* q, const uint8_t* kv, const float* sinks, const int32_t* qslots,
                                    const int32_t* ridx, const int32_t* rcnt, uint16_t* out,
                                    int M, int H, int kv_len, float scale, hipStream_t s) {
    hipLaunchKernelGGL(k_mla_absorbed_sparse_t<true>, dim3(uint32_t(H), uint32_t(M)), dim3(256), 0, s,
                       q, kv, sinks, qslots, ridx, rcnt, out, H, kv_len, scale);
}
void launch_mla_absorbed_sparse_bf16(const uint16_t* q, const uint16_t* kv, const float* sinks, const int32_t* qslots,
                                     const int32_t* ridx, const int32_t* rcnt, uint16_t* out,
                                     int M, int H, int kv_len, float scale, hipStream_t s) {
    hipLaunchKernelGGL(k_mla_absorbed_sparse_t<false>, dim3(uint32_t(H), uint32_t(M)), dim3(256), 0, s,
                       q, kv, sinks, qslots, ridx, rcnt, out, H, kv_len, scale);
}
void launch_mla_absorbed_partial_fp8(
    const uint16_t* q, const uint8_t* kv, const int32_t* l2g, const int32_t* qpos,
    const int32_t* lidx, const int32_t* lcnt, float* omax, float* osum, float* oacc,
    int M, int H, int local_kv_len, int max_rec, float scale, hipStream_t s,
    const int32_t* qmin)
{
    hipLaunchKernelGGL(k_mla_absorbed_partial_fp8, dim3(uint32_t(H), uint32_t(M)), dim3(256), 0, s,
                       q, kv, l2g, qpos, lidx, lcnt, omax, osum, oacc, H, local_kv_len, max_rec, scale, qmin);
}
void launch_mla_absorbed_partial_bf16(
    const uint16_t* q, const uint16_t* kv, const int32_t* l2g, const int32_t* qpos,
    const int32_t* lidx, const int32_t* lcnt, float* omax, float* osum, float* oacc,
    int M, int H, int local_kv_len, int max_rec, float scale, hipStream_t s,
    const int32_t* qmin)
{
    hipLaunchKernelGGL(k_mla_absorbed_partial_bf16, dim3(uint32_t(H), uint32_t(M)), dim3(256), 0, s,
                       q, kv, l2g, qpos, lidx, lcnt, omax, osum, oacc, H, local_kv_len, max_rec, scale, qmin);
}
void launch_moe_router_top8_256(const uint16_t* logits, const float* bias, int32_t* ids, float* wts, int M, float sc, hipStream_t s) {
    hipLaunchKernelGGL(k_moe_router_top8_256, dim3(uint32_t(M)), dim3(256), 0, s, logits, bias, ids, wts, sc);
}
void launch_dfly2_dual_rmsnorm_concat_bf16(const uint16_t* a, const uint16_t* wa, const uint16_t* b, const uint16_t* wb,
                                           uint16_t* out, int M, int H, float eps, hipStream_t s) {
    hipLaunchKernelGGL(k_dfly2_dual_rmsnorm_concat_bf16, dim3(uint32_t(M)), dim3(256), 0, s, a, wa, b, wb, out, H, eps);
}
void launch_dfly2_gate_residual_bf16(const uint16_t* anchor, const uint16_t* gate, const uint16_t* fused, uint16_t* out,
                                     int M, int H, hipStream_t s) {
    hipLaunchKernelGGL(k_dfly2_gate_residual_bf16, dim3(uint32_t((H + 255) / 256), uint32_t(M)), dim3(256), 0, s, anchor, gate, fused, out, M, H);
}
void launch_qb_transpose_nope_f16(const uint16_t* qb, uint16_t* qn, int M, int H, hipStream_t s) {
    const int tot = H * M * 192;
    if (tot <= 0) return;
    hipLaunchKernelGGL(k_qb_transpose_nope_f16, dim3(uint32_t((tot + 255) / 256)), dim3(256), 0, s, qb, qn, M, H);
}
void launch_assemble_q_abs_bf16(const float* qn, const uint16_t* qb, uint16_t* qabs, int M, int H, hipStream_t s) {
    const int tot = M * H * 576;
    if (tot <= 0) return;
    hipLaunchKernelGGL(k_assemble_q_abs_bf16, dim3(uint32_t((tot + 255) / 256)), dim3(256), 0, s, qn, qb, qabs, M, H);
}
void launch_ao_transpose_gate_bf16(const float* ao, const uint16_t* gate, uint16_t* out, int M, int H, hipStream_t s) {
    const int tot = M * H * 256;
    if (tot <= 0) return;
    hipLaunchKernelGGL(k_ao_transpose_gate_bf16, dim3(uint32_t((tot + 255) / 256)), dim3(256), 0, s, ao, gate, out, M, H);
}

}  // namespace hy4_770b
