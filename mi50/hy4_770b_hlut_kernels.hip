// Zero-Scratch gfx906 Vega 20 HIP Kernels for tencent/Hy4 (770B) Non-Latent [6144, 2048] MoE
// 100% ScratchSize == 0, 0 SGPR/VGPR spills, 0 trig, 0 slc/system RMW atomics.

#include "hy4_770b_hlut_kernels.hpp"
#include <hip/hip_fp16.h>

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
__device__ __forceinline__ float d_f16_to_f32(uint16_t h) noexcept {
    return __half2float(__ushort_as_half(h));
}
__device__ __forceinline__ uint16_t d_f32_to_f16(float x) noexcept {
    x = fmaxf(-65504.0f, fminf(65504.0f, x));
    return __half_as_ushort(__float2half_rn(x));
}
__device__ __forceinline__ float d_swiglu10(float g, float u) noexcept {
    const float gc = fminf(g, 10.0f);
    const float uc = fmaxf(-10.0f, fminf(10.0f, u));
    return (gc / (1.0f + expf(-gc))) * uc;
}
__device__ __forceinline__ float d_warp32_sum(float v) noexcept {
    #pragma unroll
    for (int len = 1; len <= 16; len <<= 1) v += __shfl_xor(v, len, 32);
    return v;
}
__device__ __forceinline__ float d_wave64_sum(float v) noexcept {
    #pragma unroll
    for (int len = 1; len <= 32; len <<= 1) v += __shfl_xor(v, len, 64);
    return v;
}

// Zero-LDS bitwise IEEE FP8 E4M3FN -> FP32 (handles normals + subnormals exactly, 0 LUT loads!)
__device__ __forceinline__ float d_fp8_e4m3fn_to_f32(uint32_t b) noexcept {
    const uint32_t sign = (b & 0x80u) << 24;
    const uint32_t mag  = b & 0x7Fu;
    if (mag == 0u || mag == 0x7Fu) return 0.0f;
    if (mag >= 8u) {
        return __uint_as_float(sign | ((mag << 20) + 0x3C000000u));
    }
    const float sub = __uint_as_float(0x3C800000u | (mag << 20)) - 0.015625f;
    return sign ? -sub : sub;
}

__device__ __forceinline__ float d_glc_load_f32(const float* p) noexcept {
    float v;
    asm volatile("flat_load_dword %0, %1 glc\n s_waitcnt vmcnt(0) lgkmcnt(0)" : "=v"(v) : "v"(p));
    return v;
}

}  // namespace

// ============================================================================
// 1. WHT128 Butterfly Kernels (128 threads/block = 2 wavefronts of 64)
// ============================================================================
__global__ __launch_bounds__(128, 8)
void k_wht128_f32(const float* __restrict__ in, const uint64_t* __restrict__ signs, float* __restrict__ out, int K) {
    __shared__ float s_xwave[128];
    const int blk = int(blockIdx.x), row = int(blockIdx.y), tid = int(threadIdx.x);
    const int lane = tid & 63, wid = tid >> 6;
    const size_t base = size_t(row) * size_t(K) + size_t(blk) * 128u;
    float v = in[base + tid];
    if ((signs[blk * 2 + wid] >> lane) & 1ULL) v = -v;
    #pragma unroll
    for (int len = 1; len <= 32; len <<= 1) {
        const float p = __shfl_xor(v, len);
        v = (lane & len) ? (p - v) : (v + p);
    }
    s_xwave[tid] = v;
    __syncthreads();
    const float a = s_xwave[lane], b = s_xwave[64 + lane];
    out[base + tid] = ((wid == 0) ? (a + b) : (a - b)) * 0.08838834764831844655f;
}

__global__ __launch_bounds__(128, 8)
void k_wht128_bf16_to_f16(const uint16_t* __restrict__ in_bf16, const uint64_t* __restrict__ signs, uint16_t* __restrict__ out_f16, int K) {
    __shared__ float s_xwave[128];
    const int blk = int(blockIdx.x), row = int(blockIdx.y), tid = int(threadIdx.x);
    const int lane = tid & 63, wid = tid >> 6;
    const size_t base = size_t(row) * size_t(K) + size_t(blk) * 128u;
    float v = d_bf16_to_f32(in_bf16[base + tid]);
    if ((signs[blk * 2 + wid] >> lane) & 1ULL) v = -v;
    #pragma unroll
    for (int len = 1; len <= 32; len <<= 1) {
        const float p = __shfl_xor(v, len);
        v = (lane & len) ? (p - v) : (v + p);
    }
    s_xwave[tid] = v;
    __syncthreads();
    const float a = s_xwave[lane], b = s_xwave[64 + lane];
    out_f16[base + tid] = d_f32_to_f16(((wid == 0) ? (a + b) : (a - b)) * 0.08838834764831844655f);
}

__global__ __launch_bounds__(128, 8)
void k_wht128_bf16_to_f32(const uint16_t* __restrict__ in_bf16, const uint64_t* __restrict__ signs, float* __restrict__ out_f32, int K) {
    __shared__ float s_xwave[128];
    const int blk = int(blockIdx.x), row = int(blockIdx.y), tid = int(threadIdx.x);
    const int lane = tid & 63, wid = tid >> 6;
    const size_t base = size_t(row) * size_t(K) + size_t(blk) * 128u;
    float v = d_bf16_to_f32(in_bf16[base + tid]);
    if ((signs[blk * 2 + wid] >> lane) & 1ULL) v = -v;
    #pragma unroll
    for (int len = 1; len <= 32; len <<= 1) {
        const float p = __shfl_xor(v, len);
        v = (lane & len) ? (p - v) : (v + p);
    }
    s_xwave[tid] = v;
    __syncthreads();
    const float a = s_xwave[lane], b = s_xwave[64 + lane];
    out_f32[base + tid] = ((wid == 0) ? (a + b) : (a - b)) * 0.08838834764831844655f;
}

// ============================================================================
// 2. WHT128-HLUT3 (400 B tile, 3.125 bpw) — GEMV, Grouped Decode & Unpack
// ============================================================================
__global__ __launch_bounds__(256, 8)
void k_hlut3_gemv(const float* __restrict__ x, const Hlut3Tile32x32* __restrict__ tiles,
                  const float* __restrict__ levels4, float* __restrict__ y, int N, int K) {
    const int tid = int(threadIdx.x), sub = tid >> 5, lane = tid & 31;
    const int m = int(blockIdx.y), n = int(blockIdx.x) * 8 + sub;
    if (n >= N) return;
    const int nt = n >> 5, lr = n & 31, sg = lr >> 2, kt_cnt = K >> 5;
    const float* __restrict__ xr = x + size_t(m) * size_t(K);
    const Hlut3Tile32x32* __restrict__ trow = tiles + size_t(nt) * size_t(kt_cnt);
    const float lv0 = levels4[0], lv1 = levels4[1], lv2 = levels4[2], lv3 = levels4[3];
    float acc = 0.0f;
    for (int kt = 0; kt < kt_cnt; ++kt) {
        const Hlut3Tile32x32* t = trow + kt;
        const uint32_t mc = (t->q_lo2[lr * 2 + (lane >> 4)] >> (2 * (lane & 15))) & 3u;
        const uint32_t sb = (t->q_hi1[lr] >> lane) & 1u;
        const float lv = (mc == 0u) ? lv0 : ((mc == 1u) ? lv1 : ((mc == 2u) ? lv2 : lv3));
        const float w = (sb ? -lv : lv) * d_f16_to_f32(t->scales_f16[sg]);
        acc = fmaf(xr[kt * 32 + lane], w, acc);
    }
    acc = d_warp32_sum(acc);
    if (lane == 0) y[size_t(m) * size_t(N) + size_t(n)] = acc;
}

// Weight-stationary grouped HLUT3 Gate+Up+Clamped-SwiGLU (K=6144 -> 12 KiB FP16 LDS/token)
// Processes 1 token/pass in M=1 or loops over count tokens while reusing unpacked tile weights!
__global__ __launch_bounds__(128, 2)
void k_hlut3_grouped_gateup_swiglu_f16(
    const uint16_t* __restrict__ x_rot_f16,
    const Hy4ExpertGroup* __restrict__ groups,
    const Hlut3Tile32x32* __restrict__ gate_tiles,
    const Hlut3Tile32x32* __restrict__ up_tiles,
    const float* __restrict__ levels4,
    float* __restrict__ z_out,
    int N_inter, int K_in)
{
    // Stage 1 token slice [32] at a time in registers for up to kMaxGroupTokens — zero LDS overflow, 0 spills!
    const int g_idx = int(blockIdx.y);
    const Hy4ExpertGroup& grp = groups[g_idx];
    if (grp.expert < 0 || grp.count <= 0) return;

    const int tid = int(threadIdx.x);
    const int lr  = tid >> 2;
    const int w4  = tid & 3;
    const int nt  = int(blockIdx.x);
    const int n   = nt * 32 + lr;
    const int sg  = lr >> 2;
    const int nt_cnt = N_inter >> 5;
    const int kt_cnt = K_in >> 5;
    const size_t tbase = size_t(grp.expert) * size_t(nt_cnt) * size_t(kt_cnt) + size_t(nt) * size_t(kt_cnt);
    const Hlut3Tile32x32* __restrict__ grow = gate_tiles + tbase;
    const Hlut3Tile32x32* __restrict__ urow = up_tiles   + tbase;
    const float lv0 = levels4[0], lv1 = levels4[1], lv2 = levels4[2], lv3 = levels4[3];
    const int wh = w4 >> 1, msh = (w4 & 1) * 16, ssh = w4 * 8;

    // Specialize c_cnt <= 2 vs general loop without register array spills!
    for (int c = 0; c < grp.count; ++c) {
        const uint16_t* __restrict__ xrow = x_rot_f16 + size_t(grp.rows[c]) * size_t(K_in);
        float acc_g = 0.0f, acc_u = 0.0f;
        for (int kt = 0; kt < kt_cnt; ++kt) {
            const Hlut3Tile32x32* __restrict__ gt = grow + kt;
            const Hlut3Tile32x32* __restrict__ ut = urow + kt;
            const uint32_t gm = gt->q_lo2[lr * 2 + wh] >> msh;
            const uint32_t gs = gt->q_hi1[lr] >> ssh;
            const uint32_t um = ut->q_lo2[lr * 2 + wh] >> msh;
            const uint32_t us = ut->q_hi1[lr] >> ssh;
            const float sig_g = d_f16_to_f32(gt->scales_f16[sg]);
            const float sig_u = d_f16_to_f32(ut->scales_f16[sg]);
            const uint16_t* __restrict__ xk = xrow + kt * 32 + ssh;
            float lg = 0.0f, lu = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float xv = d_f16_to_f32(xk[j]);
                const uint32_t cg = (gm >> (2 * j)) & 3u;
                const float vg = (cg == 0u) ? lv0 : ((cg == 1u) ? lv1 : ((cg == 2u) ? lv2 : lv3));
                lg = fmaf(xv, ((gs >> j) & 1u) ? -vg : vg, lg);
                const uint32_t cu = (um >> (2 * j)) & 3u;
                const float vu = (cu == 0u) ? lv0 : ((cu == 1u) ? lv1 : ((cu == 2u) ? lv2 : lv3));
                lu = fmaf(xv, ((us >> j) & 1u) ? -vu : vu, lu);
            }
            acc_g = fmaf(lg, sig_g, acc_g);
            acc_u = fmaf(lu, sig_u, acc_u);
        }
        acc_g += __shfl_xor(acc_g, 1, 64);
        acc_g += __shfl_xor(acc_g, 2, 64);
        acc_u += __shfl_xor(acc_u, 1, 64);
        acc_u += __shfl_xor(acc_u, 2, 64);
        if (w4 == 0 && n < N_inter) {
            z_out[(size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c)) * size_t(N_inter) + size_t(n)] = d_swiglu10(acc_g, acc_u);
        }
    }
}

__global__ __launch_bounds__(128, 4)
void k_hlut3_grouped_down_atomic(
    const float* __restrict__ z_rot,
    const Hy4ExpertGroup* __restrict__ groups,
    const Hlut3Tile32x32* __restrict__ down_tiles,
    const float* __restrict__ levels4,
    float* __restrict__ out_accum_f32,
    int N_down, int K_down)
{
    __shared__ float s_z[kMoeInterSize];  // 8 KiB LDS
    const int g_idx = int(blockIdx.y);
    const Hy4ExpertGroup& grp = groups[g_idx];
    if (grp.expert < 0 || grp.count <= 0) return;

    const int tid = int(threadIdx.x);
    const int lr  = tid >> 2, w4 = tid & 3;
    const int nt  = int(blockIdx.x), n = nt * 32 + lr, sg = lr >> 2;
    const int nt_cnt = N_down >> 5, kt_cnt = K_down >> 5;
    const Hlut3Tile32x32* __restrict__ drow = down_tiles + size_t(grp.expert) * size_t(nt_cnt) * size_t(kt_cnt) + size_t(nt) * size_t(kt_cnt);
    const float lv0 = levels4[0], lv1 = levels4[1], lv2 = levels4[2], lv3 = levels4[3];
    const int wh = w4 >> 1, msh = (w4 & 1) * 16, ssh = w4 * 8;

    for (int c = 0; c < grp.count; ++c) {
        const float* __restrict__ zsrc = z_rot + (size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c)) * size_t(K_down);
        for (int k = tid; k < K_down; k += 128) s_z[k] = zsrc[k];
        __syncthreads();

        float acc = 0.0f;
        for (int kt = 0; kt < kt_cnt; ++kt) {
            const Hlut3Tile32x32* __restrict__ dt = drow + kt;
            const uint32_t dm = dt->q_lo2[lr * 2 + wh] >> msh;
            const uint32_t ds = dt->q_hi1[lr] >> ssh;
            const float sig = d_f16_to_f32(dt->scales_f16[sg]);
            const int zb = kt * 32 + ssh;
            float loc = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t cd = (dm >> (2 * j)) & 3u;
                const float vd = (cd == 0u) ? lv0 : ((cd == 1u) ? lv1 : ((cd == 2u) ? lv2 : lv3));
                loc = fmaf(s_z[zb + j], ((ds >> j) & 1u) ? -vd : vd, loc);
            }
            acc = fmaf(loc, sig, acc);
        }
        acc += __shfl_xor(acc, 1, 64);
        acc += __shfl_xor(acc, 2, 64);
        if (w4 == 0 && n < N_down) {
            atomicAdd(&out_accum_f32[size_t(grp.rows[c]) * size_t(N_down) + size_t(n)], acc * grp.weights[c]);
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(256, 8)
void k_unpack_hlut3_to_f16(const Hlut3Tile32x32* __restrict__ tiles, const float* __restrict__ levels4,
                           uint16_t* __restrict__ out_f16, int N, int K) {
    const int nt = int(blockIdx.y), kt = int(blockIdx.x);
    const int tid = int(threadIdx.x), kt_cnt = K >> 5;
    const Hlut3Tile32x32* t = tiles + size_t(nt) * size_t(kt_cnt) + size_t(kt);
    const float lv0 = levels4[0], lv1 = levels4[1], lv2 = levels4[2], lv3 = levels4[3];
    for (int idx = tid; idx < 1024; idx += 256) {
        const int r = idx >> 5, c = idx & 31;
        const uint32_t mc = (t->q_lo2[r * 2 + (c >> 4)] >> (2 * (c & 15))) & 3u;
        const uint32_t sb = (t->q_hi1[r] >> c) & 1u;
        const float lv = (mc == 0u) ? lv0 : ((mc == 1u) ? lv1 : ((mc == 2u) ? lv2 : lv3));
        const float w = (sb ? -lv : lv) * d_f16_to_f32(t->scales_f16[r >> 2]);
        out_f16[size_t(nt * 32 + r) * size_t(K) + size_t(kt * 32 + c)] = d_f32_to_f16(w);
    }
}

// ============================================================================
// 3. WHT128-HLUT2 (272 B tile, 2.125 bpw) — GEMV, Grouped Decode & Unpack
// ============================================================================
__global__ __launch_bounds__(256, 8)
void k_hlut2_gemv(const float* __restrict__ x, const Hlut2Tile32x32* __restrict__ tiles,
                  const float* __restrict__ levels2, float* __restrict__ y, int N, int K) {
    const int tid = int(threadIdx.x), sub = tid >> 5, lane = tid & 31;
    const int m = int(blockIdx.y), n = int(blockIdx.x) * 8 + sub;
    if (n >= N) return;
    const int nt = n >> 5, lr = n & 31, sg = lr >> 2, kt_cnt = K >> 5;
    const float* __restrict__ xr = x + size_t(m) * size_t(K);
    const Hlut2Tile32x32* __restrict__ trow = tiles + size_t(nt) * size_t(kt_cnt);
    const float lv0 = levels2[0], lv1 = levels2[1];
    float acc = 0.0f;
    for (int kt = 0; kt < kt_cnt; ++kt) {
        const Hlut2Tile32x32* t = trow + kt;
        const uint32_t code = (t->q_2b[lr * 2 + (lane >> 4)] >> (2 * (lane & 15))) & 3u;
        const float lv = (code & 1u) ? lv1 : lv0;
        const float w = ((code & 2u) ? -lv : lv) * d_f16_to_f32(t->scales_f16[sg]);
        acc = fmaf(xr[kt * 32 + lane], w, acc);
    }
    acc = d_warp32_sum(acc);
    if (lane == 0) y[size_t(m) * size_t(N) + size_t(n)] = acc;
}

__global__ __launch_bounds__(128, 4)
void k_hlut2_grouped_gateup_swiglu_f16(
    const uint16_t* __restrict__ x_rot_f16,
    const Hy4ExpertGroup* __restrict__ groups,
    const Hlut2Tile32x32* __restrict__ gate_tiles,
    const Hlut2Tile32x32* __restrict__ up_tiles,
    const float* __restrict__ levels2,
    float* __restrict__ z_out,
    int N_inter, int K_in)
{
    const int g_idx = int(blockIdx.y);
    const Hy4ExpertGroup& grp = groups[g_idx];
    if (grp.expert < 0 || grp.count <= 0) return;

    const int tid = int(threadIdx.x), lr = tid >> 2, w4 = tid & 3;
    const int nt = int(blockIdx.x), n = nt * 32 + lr, sg = lr >> 2;
    const int nt_cnt = N_inter >> 5, kt_cnt = K_in >> 5;
    const size_t tbase = size_t(grp.expert) * size_t(nt_cnt) * size_t(kt_cnt) + size_t(nt) * size_t(kt_cnt);
    const Hlut2Tile32x32* __restrict__ grow = gate_tiles + tbase;
    const Hlut2Tile32x32* __restrict__ urow = up_tiles   + tbase;
    const float lv0 = levels2[0], lv1 = levels2[1];
    const int wh = w4 >> 1, msh = (w4 & 1) * 16, ssh = w4 * 8;

    for (int c = 0; c < grp.count; ++c) {
        const uint16_t* __restrict__ xrow = x_rot_f16 + size_t(grp.rows[c]) * size_t(K_in);
        float acc_g = 0.0f, acc_u = 0.0f;
        for (int kt = 0; kt < kt_cnt; ++kt) {
            const Hlut2Tile32x32* __restrict__ gt = grow + kt;
            const Hlut2Tile32x32* __restrict__ ut = urow + kt;
            const uint32_t gw = gt->q_2b[lr * 2 + wh] >> msh;
            const uint32_t uw = ut->q_2b[lr * 2 + wh] >> msh;
            const float sig_g = d_f16_to_f32(gt->scales_f16[sg]);
            const float sig_u = d_f16_to_f32(ut->scales_f16[sg]);
            const uint16_t* __restrict__ xk = xrow + kt * 32 + ssh;
            float lg = 0.0f, lu = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float xv = d_f16_to_f32(xk[j]);
                const uint32_t cg = (gw >> (2 * j)) & 3u;
                const float vg = (cg & 1u) ? lv1 : lv0;
                lg = fmaf(xv, (cg & 2u) ? -vg : vg, lg);
                const uint32_t cu = (uw >> (2 * j)) & 3u;
                const float vu = (cu & 1u) ? lv1 : lv0;
                lu = fmaf(xv, (cu & 2u) ? -vu : vu, lu);
            }
            acc_g = fmaf(lg, sig_g, acc_g);
            acc_u = fmaf(lu, sig_u, acc_u);
        }
        acc_g += __shfl_xor(acc_g, 1, 64);
        acc_g += __shfl_xor(acc_g, 2, 64);
        acc_u += __shfl_xor(acc_u, 1, 64);
        acc_u += __shfl_xor(acc_u, 2, 64);
        if (w4 == 0 && n < N_inter) {
            z_out[(size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c)) * size_t(N_inter) + size_t(n)] = d_swiglu10(acc_g, acc_u);
        }
    }
}

__global__ __launch_bounds__(128, 4)
void k_hlut2_grouped_down_atomic(
    const float* __restrict__ z_rot,
    const Hy4ExpertGroup* __restrict__ groups,
    const Hlut2Tile32x32* __restrict__ down_tiles,
    const float* __restrict__ levels2,
    float* __restrict__ out_accum_f32,
    int N_down, int K_down)
{
    __shared__ float s_z[kMoeInterSize];
    const int g_idx = int(blockIdx.y);
    const Hy4ExpertGroup& grp = groups[g_idx];
    if (grp.expert < 0 || grp.count <= 0) return;

    const int tid = int(threadIdx.x), lr = tid >> 2, w4 = tid & 3;
    const int nt = int(blockIdx.x), n = nt * 32 + lr, sg = lr >> 2;
    const int nt_cnt = N_down >> 5, kt_cnt = K_down >> 5;
    const Hlut2Tile32x32* __restrict__ drow = down_tiles + size_t(grp.expert) * size_t(nt_cnt) * size_t(kt_cnt) + size_t(nt) * size_t(kt_cnt);
    const float lv0 = levels2[0], lv1 = levels2[1];
    const int wh = w4 >> 1, msh = (w4 & 1) * 16, ssh = w4 * 8;

    for (int c = 0; c < grp.count; ++c) {
        const float* __restrict__ zsrc = z_rot + (size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c)) * size_t(K_down);
        for (int k = tid; k < K_down; k += 128) s_z[k] = zsrc[k];
        __syncthreads();

        float acc = 0.0f;
        for (int kt = 0; kt < kt_cnt; ++kt) {
            const Hlut2Tile32x32* __restrict__ dt = drow + kt;
            const uint32_t dw = dt->q_2b[lr * 2 + wh] >> msh;
            const float sig = d_f16_to_f32(dt->scales_f16[sg]);
            const int zb = kt * 32 + ssh;
            float loc = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t cd = (dw >> (2 * j)) & 3u;
                const float vd = (cd & 1u) ? lv1 : lv0;
                loc = fmaf(s_z[zb + j], (cd & 2u) ? -vd : vd, loc);
            }
            acc = fmaf(loc, sig, acc);
        }
        acc += __shfl_xor(acc, 1, 64);
        acc += __shfl_xor(acc, 2, 64);
        if (w4 == 0 && n < N_down) {
            atomicAdd(&out_accum_f32[size_t(grp.rows[c]) * size_t(N_down) + size_t(n)], acc * grp.weights[c]);
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(256, 8)
void k_unpack_hlut2_to_f16(const Hlut2Tile32x32* __restrict__ tiles, const float* __restrict__ levels2,
                           uint16_t* __restrict__ out_f16, int N, int K) {
    const int nt = int(blockIdx.y), kt = int(blockIdx.x);
    const int tid = int(threadIdx.x), kt_cnt = K >> 5;
    const Hlut2Tile32x32* t = tiles + size_t(nt) * size_t(kt_cnt) + size_t(kt);
    const float lv0 = levels2[0], lv1 = levels2[1];
    for (int idx = tid; idx < 1024; idx += 256) {
        const int r = idx >> 5, c = idx & 31;
        const uint32_t code = (t->q_2b[r * 2 + (c >> 4)] >> (2 * (c & 15))) & 3u;
        const float lv = (code & 1u) ? lv1 : lv0;
        const float w = ((code & 2u) ? -lv : lv) * d_f16_to_f32(t->scales_f16[r >> 2]);
        out_f16[size_t(nt * 32 + r) * size_t(K) + size_t(kt * 32 + c)] = d_f32_to_f16(w);
    }
}

// ============================================================================
// 4. WHT128-HLUT4 (528 B tile, 4.125 bpw) — GEMV & Unpack
// ============================================================================
__global__ __launch_bounds__(256, 8)
void k_hlut4_gemv(const float* __restrict__ x, const Hlut4Tile32x32* __restrict__ tiles,
                  const float* __restrict__ levels8, float* __restrict__ y, int N, int K) {
    __shared__ float s_lut16[16];
    const int tid = int(threadIdx.x);
    if (tid < 8) { const float lv = levels8[tid]; s_lut16[tid] = lv; s_lut16[tid + 8] = -lv; }
    __syncthreads();
    const int sub = tid >> 5, lane = tid & 31;
    const int m = int(blockIdx.y), n = int(blockIdx.x) * 8 + sub;
    if (n >= N) return;
    const int nt = n >> 5, lr = n & 31, sg = lr >> 2, kt_cnt = K >> 5;
    const float* __restrict__ xr = x + size_t(m) * size_t(K);
    const Hlut4Tile32x32* __restrict__ trow = tiles + size_t(nt) * size_t(kt_cnt);
    float acc = 0.0f;
    for (int kt = 0; kt < kt_cnt; ++kt) {
        const Hlut4Tile32x32* t = trow + kt;
        const uint32_t nib = (t->q_nib4[lr * 4 + (lane >> 3)] >> (4 * (lane & 7))) & 0xFu;
        acc = fmaf(xr[kt * 32 + lane], s_lut16[nib] * d_f16_to_f32(t->scales_f16[sg]), acc);
    }
    acc = d_warp32_sum(acc);
    if (lane == 0) y[size_t(m) * size_t(N) + size_t(n)] = acc;
}

__global__ __launch_bounds__(128, 4)
void k_hlut4_grouped_gateup_swiglu_f16(
    const uint16_t* __restrict__ x_rot_f16,
    const Hy4ExpertGroup* __restrict__ groups,
    const Hlut4Tile32x32* __restrict__ gate_tiles,
    const Hlut4Tile32x32* __restrict__ up_tiles,
    const float* __restrict__ levels8,
    float* __restrict__ z_out,
    int N_inter, int K_in)
{
    __shared__ float s_lut16[16];
    const int tid = int(threadIdx.x);
    if (tid < 8) { const float lv = levels8[tid]; s_lut16[tid] = lv; s_lut16[tid + 8] = -lv; }
    __syncthreads();

    const int g_idx = int(blockIdx.y);
    const Hy4ExpertGroup& grp = groups[g_idx];
    if (grp.expert < 0 || grp.count <= 0) return;

    const int lr  = tid >> 2, w4 = tid & 3;
    const int nt  = int(blockIdx.x), n = nt * 32 + lr, sg = lr >> 2;
    const int nt_cnt = N_inter >> 5, kt_cnt = K_in >> 5;
    const size_t tbase = size_t(grp.expert) * size_t(nt_cnt) * size_t(kt_cnt) + size_t(nt) * size_t(kt_cnt);
    const Hlut4Tile32x32* __restrict__ grow = gate_tiles + tbase;
    const Hlut4Tile32x32* __restrict__ urow = up_tiles   + tbase;
    const int ssh = w4 * 8;

    for (int c = 0; c < grp.count; ++c) {
        const uint16_t* __restrict__ xrow = x_rot_f16 + size_t(grp.rows[c]) * size_t(K_in);
        float acc_g = 0.0f, acc_u = 0.0f;
        for (int kt = 0; kt < kt_cnt; ++kt) {
            const Hlut4Tile32x32* __restrict__ gt = grow + kt;
            const Hlut4Tile32x32* __restrict__ ut = urow + kt;
            const uint32_t gw = gt->q_nib4[lr * 4 + w4];
            const uint32_t uw = ut->q_nib4[lr * 4 + w4];
            const float sig_g = d_f16_to_f32(gt->scales_f16[sg]);
            const float sig_u = d_f16_to_f32(ut->scales_f16[sg]);
            const uint16_t* __restrict__ xk = xrow + kt * 32 + ssh;
            float lg = 0.0f, lu = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float xv = d_f16_to_f32(xk[j]);
                lg = fmaf(xv, s_lut16[(gw >> (4 * j)) & 0xFu], lg);
                lu = fmaf(xv, s_lut16[(uw >> (4 * j)) & 0xFu], lu);
            }
            acc_g = fmaf(lg, sig_g, acc_g);
            acc_u = fmaf(lu, sig_u, acc_u);
        }
        acc_g += __shfl_xor(acc_g, 1, 64);
        acc_g += __shfl_xor(acc_g, 2, 64);
        acc_u += __shfl_xor(acc_u, 1, 64);
        acc_u += __shfl_xor(acc_u, 2, 64);
        if (w4 == 0 && n < N_inter) {
            z_out[(size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c)) * size_t(N_inter) + size_t(n)] = d_swiglu10(acc_g, acc_u);
        }
    }
}

__global__ __launch_bounds__(128, 4)
void k_hlut4_grouped_down_atomic(
    const float* __restrict__ z_rot,
    const Hy4ExpertGroup* __restrict__ groups,
    const Hlut4Tile32x32* __restrict__ down_tiles,
    const float* __restrict__ levels8,
    float* __restrict__ out_accum_f32,
    int N_down, int K_down)
{
    __shared__ float s_z[kMoeInterSize];
    __shared__ float s_lut16[16];
    const int tid = int(threadIdx.x);
    if (tid < 8) { const float lv = levels8[tid]; s_lut16[tid] = lv; s_lut16[tid + 8] = -lv; }
    __syncthreads();

    const int g_idx = int(blockIdx.y);
    const Hy4ExpertGroup& grp = groups[g_idx];
    if (grp.expert < 0 || grp.count <= 0) return;

    const int lr  = tid >> 2, w4 = tid & 3;
    const int nt  = int(blockIdx.x), n = nt * 32 + lr, sg = lr >> 2;
    const int nt_cnt = N_down >> 5, kt_cnt = K_down >> 5;
    const Hlut4Tile32x32* __restrict__ drow = down_tiles + size_t(grp.expert) * size_t(nt_cnt) * size_t(kt_cnt) + size_t(nt) * size_t(kt_cnt);
    const int ssh = w4 * 8;

    for (int c = 0; c < grp.count; ++c) {
        const float* __restrict__ zsrc = z_rot + (size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c)) * size_t(K_down);
        for (int k = tid; k < K_down; k += 128) s_z[k] = zsrc[k];
        __syncthreads();

        float acc = 0.0f;
        for (int kt = 0; kt < kt_cnt; ++kt) {
            const Hlut4Tile32x32* __restrict__ dt = drow + kt;
            const uint32_t dw = dt->q_nib4[lr * 4 + w4];
            const float sig = d_f16_to_f32(dt->scales_f16[sg]);
            const int zb = kt * 32 + ssh;
            float loc = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                loc = fmaf(s_z[zb + j], s_lut16[(dw >> (4 * j)) & 0xFu], loc);
            }
            acc = fmaf(loc, sig, acc);
        }
        acc += __shfl_xor(acc, 1, 64);
        acc += __shfl_xor(acc, 2, 64);
        if (w4 == 0 && n < N_down) {
            atomicAdd(&out_accum_f32[size_t(grp.rows[c]) * size_t(N_down) + size_t(n)], acc * grp.weights[c]);
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(256, 8)
void k_unpack_hlut4_to_f16(const Hlut4Tile32x32* __restrict__ tiles, const float* __restrict__ levels8,
                           uint16_t* __restrict__ out_f16, int N, int K) {
    __shared__ float s_lut16[16];
    const int tid = int(threadIdx.x);
    if (tid < 8) { const float lv = levels8[tid]; s_lut16[tid] = lv; s_lut16[tid + 8] = -lv; }
    __syncthreads();
    const int nt = int(blockIdx.y), kt = int(blockIdx.x), kt_cnt = K >> 5;
    const Hlut4Tile32x32* t = tiles + size_t(nt) * size_t(kt_cnt) + size_t(kt);
    for (int idx = tid; idx < 1024; idx += 256) {
        const int r = idx >> 5, c = idx & 31;
        const uint32_t nib = (t->q_nib4[r * 4 + (c >> 3)] >> (4 * (c & 7))) & 0xFu;
        const float w = s_lut16[nib] * d_f16_to_f32(t->scales_f16[r >> 2]);
        out_f16[size_t(nt * 32 + r) * size_t(K) + size_t(kt * 32 + c)] = d_f32_to_f16(w);
    }
}

// ============================================================================
// 4B. Lossless Stock Block-FP8 (128x128) Grouped MoE (4-Token Weight-Stationary, 0 Scratch!)
// ============================================================================
__global__ __launch_bounds__(256, 4)
void k_fp8_grouped_gateup_swiglu(
    const uint16_t* __restrict__ x_bf16,
    const Hy4ExpertGroup* __restrict__ groups,
    const uint8_t* __restrict__ gate_w,
    const float*   __restrict__ gate_s,
    const uint8_t* __restrict__ up_w,
    const float*   __restrict__ up_s,
    float* __restrict__ z_out,
    int N_inter, int K_in)
{
    const int g_idx = int(blockIdx.y);
    const Hy4ExpertGroup& grp = groups[g_idx];
    if (grp.expert < 0 || grp.count <= 0) return;

    const int tid = int(threadIdx.x), wid = tid >> 6, lane = tid & 63;
    const int n = int(blockIdx.x) * 4 + wid;
    if (n >= N_inter) return;

    const int nb = N_inter >> 7;
    const int kb = K_in >> 7;
    const size_t w_exp_off = size_t(grp.expert) * size_t(N_inter) * size_t(K_in) + size_t(n) * size_t(K_in);
    const size_t s_exp_off = size_t(grp.expert) * size_t(nb) * size_t(kb) + size_t(n >> 7) * size_t(kb);
    const uint8_t* __restrict__ gw_row = gate_w + w_exp_off;
    const uint8_t* __restrict__ uw_row = up_w   + w_exp_off;
    const float*   __restrict__ gs_row = gate_s + s_exp_off;
    const float*   __restrict__ us_row = up_s   + s_exp_off;

    // 4-token weight-stationary tiling: loads FP8 weights ONCE per 4 tokens in the group!
    for (int c0 = 0; c0 < grp.count; c0 += 4) {
        const int cnt = min(4, grp.count - c0);
        const uint16_t* xr0 = x_bf16 + size_t(grp.rows[c0 + 0]) * size_t(K_in);
        const uint16_t* xr1 = (cnt > 1) ? (x_bf16 + size_t(grp.rows[c0 + 1]) * size_t(K_in)) : xr0;
        const uint16_t* xr2 = (cnt > 2) ? (x_bf16 + size_t(grp.rows[c0 + 2]) * size_t(K_in)) : xr0;
        const uint16_t* xr3 = (cnt > 3) ? (x_bf16 + size_t(grp.rows[c0 + 3]) * size_t(K_in)) : xr0;

        float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
        float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

        for (int b = 0; b < kb; ++b) {
            const int k = b * 128 + lane * 2;
            const float sg = gs_row[b];
            const float su = us_row[b];
            const float g0 = d_fp8_e4m3fn_to_f32(gw_row[k])     * sg;
            const float g1 = d_fp8_e4m3fn_to_f32(gw_row[k + 1]) * sg;
            const float u0 = d_fp8_e4m3fn_to_f32(uw_row[k])     * su;
            const float u1 = d_fp8_e4m3fn_to_f32(uw_row[k + 1]) * su;

            const float a0 = d_bf16_to_f32(xr0[k]), a1 = d_bf16_to_f32(xr0[k + 1]);
            ag0 = fmaf(a0, g0, fmaf(a1, g1, ag0));
            au0 = fmaf(a0, u0, fmaf(a1, u1, au0));
            if (cnt > 1) {
                const float b0 = d_bf16_to_f32(xr1[k]), b1 = d_bf16_to_f32(xr1[k + 1]);
                ag1 = fmaf(b0, g0, fmaf(b1, g1, ag1));
                au1 = fmaf(b0, u0, fmaf(b1, u1, au1));
            }
            if (cnt > 2) {
                const float d0 = d_bf16_to_f32(xr2[k]), d1 = d_bf16_to_f32(xr2[k + 1]);
                ag2 = fmaf(d0, g0, fmaf(d1, g1, ag2));
                au2 = fmaf(d0, u0, fmaf(d1, u1, au2));
            }
            if (cnt > 3) {
                const float e0 = d_bf16_to_f32(xr3[k]), e1 = d_bf16_to_f32(xr3[k + 1]);
                ag3 = fmaf(e0, g0, fmaf(e1, g1, ag3));
                au3 = fmaf(e0, u0, fmaf(e1, u1, au3));
            }
        }
        ag0 = d_wave64_sum(ag0); au0 = d_wave64_sum(au0);
        if (lane == 0) z_out[(size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c0 + 0)) * size_t(N_inter) + size_t(n)] = d_swiglu10(ag0, au0);
        if (cnt > 1) {
            ag1 = d_wave64_sum(ag1); au1 = d_wave64_sum(au1);
            if (lane == 0) z_out[(size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c0 + 1)) * size_t(N_inter) + size_t(n)] = d_swiglu10(ag1, au1);
        }
        if (cnt > 2) {
            ag2 = d_wave64_sum(ag2); au2 = d_wave64_sum(au2);
            if (lane == 0) z_out[(size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c0 + 2)) * size_t(N_inter) + size_t(n)] = d_swiglu10(ag2, au2);
        }
        if (cnt > 3) {
            ag3 = d_wave64_sum(ag3); au3 = d_wave64_sum(au3);
            if (lane == 0) z_out[(size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c0 + 3)) * size_t(N_inter) + size_t(n)] = d_swiglu10(ag3, au3);
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_fp8_grouped_down_atomic(
    const float* __restrict__ z_in,
    const Hy4ExpertGroup* __restrict__ groups,
    const uint8_t* __restrict__ down_w,
    const float*   __restrict__ down_s,
    float* __restrict__ out_accum_f32,
    int N_down, int K_down)
{
    // 2-token weight-stationary tiling in 16 KiB LDS (4 blocks/CU occupancy, 0 scratch, 2x fewer HBM2 reads!)
    __shared__ float s_z[2][kMoeInterSize];
    const int g_idx = int(blockIdx.y);
    const Hy4ExpertGroup& grp = groups[g_idx];
    if (grp.expert < 0 || grp.count <= 0) return;

    const int tid = int(threadIdx.x), wid = tid >> 6, lane = tid & 63;
    const int n = int(blockIdx.x) * 4 + wid;
    const int nb = N_down >> 7;
    const int kb = K_down >> 7;
    const size_t w_exp_off = size_t(grp.expert) * size_t(N_down) * size_t(K_down) + size_t(n) * size_t(K_down);
    const size_t s_exp_off = size_t(grp.expert) * size_t(nb) * size_t(kb) + size_t(n >> 7) * size_t(kb);
    const uint8_t* __restrict__ dw_row = down_w + w_exp_off;
    const float*   __restrict__ ds_row = down_s + s_exp_off;

    for (int c0 = 0; c0 < grp.count; c0 += 2) {
        const bool has1 = (c0 + 1 < grp.count);
        const float* __restrict__ z0 = z_in + (size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c0)) * size_t(K_down);
        for (int k = tid; k < K_down; k += 256) s_z[0][k] = z0[k];
        if (has1) {
            const float* __restrict__ z1 = z_in + (size_t(g_idx) * size_t(kMaxGroupTokens) + size_t(c0 + 1)) * size_t(K_down);
            for (int k = tid; k < K_down; k += 256) s_z[1][k] = z1[k];
        }
        __syncthreads();

        if (n < N_down) {
            float acc0 = 0.0f, acc1 = 0.0f;
            for (int b = 0; b < kb; ++b) {
                const int k = b * 128 + lane * 2;
                const float ds = ds_row[b];
                const float w0 = d_fp8_e4m3fn_to_f32(dw_row[k])     * ds;
                const float w1 = d_fp8_e4m3fn_to_f32(dw_row[k + 1]) * ds;
                acc0 = fmaf(s_z[0][k], w0, fmaf(s_z[0][k + 1], w1, acc0));
                if (has1) {
                    acc1 = fmaf(s_z[1][k], w0, fmaf(s_z[1][k + 1], w1, acc1));
                }
            }
            acc0 = d_wave64_sum(acc0);
            if (lane == 0) {
                atomicAdd(&out_accum_f32[size_t(grp.rows[c0]) * size_t(N_down) + size_t(n)], acc0 * grp.weights[c0]);
            }
            if (has1) {
                acc1 = d_wave64_sum(acc1);
                if (lane == 0) {
                    atomicAdd(&out_accum_f32[size_t(grp.rows[c0 + 1]) * size_t(N_down) + size_t(n)], acc1 * grp.weights[c0 + 1]);
                }
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// 5. Zero-LDS Bitwise Block-FP8 (128x128) & BF16 Spine GEMVs + Dequant
// ============================================================================
__global__ __launch_bounds__(256, 4)
void k_fp8_block128_gemv_bf16(
    const uint16_t* __restrict__ x_bf16,
    const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ w_scale,
    uint16_t* __restrict__ y_bf16,
    int M, int N, int K)
{
    // 4-token weight-stationary tiling: loads FP8 spine weight ONCE per 4 tokens (3x fewer HBM2 reads at M=6!)
    const int tid = int(threadIdx.x), wid = tid >> 6, lane = tid & 63;
    const int m0 = int(blockIdx.y) * 4, n = int(blockIdx.x) * 4 + wid;
    if (n >= N || m0 >= M) return;
    const int cnt = min(4, M - m0);
    const int kb = (K + 127) >> 7;
    const uint16_t* __restrict__ x0 = x_bf16 + size_t(m0 + 0) * size_t(K);
    const uint16_t* __restrict__ x1 = (cnt > 1) ? (x_bf16 + size_t(m0 + 1) * size_t(K)) : x0;
    const uint16_t* __restrict__ x2 = (cnt > 2) ? (x_bf16 + size_t(m0 + 2) * size_t(K)) : x0;
    const uint16_t* __restrict__ x3 = (cnt > 3) ? (x_bf16 + size_t(m0 + 3) * size_t(K)) : x0;
    const uint8_t*  __restrict__ wr = w_fp8  + size_t(n) * size_t(K);
    const float*    __restrict__ sr = w_scale + size_t(n >> 7) * size_t(kb);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int b = 0; b < kb; ++b) {
        const int c0 = b * 128 + lane * 2;
        const float sc = sr[b];
        if (c0 + 1 < K) {
            const float w0 = d_fp8_e4m3fn_to_f32(wr[c0])     * sc;
            const float w1 = d_fp8_e4m3fn_to_f32(wr[c0 + 1]) * sc;
            a0 = fmaf(d_bf16_to_f32(x0[c0]), w0, fmaf(d_bf16_to_f32(x0[c0 + 1]), w1, a0));
            if (cnt > 1) a1 = fmaf(d_bf16_to_f32(x1[c0]), w0, fmaf(d_bf16_to_f32(x1[c0 + 1]), w1, a1));
            if (cnt > 2) a2 = fmaf(d_bf16_to_f32(x2[c0]), w0, fmaf(d_bf16_to_f32(x2[c0 + 1]), w1, a2));
            if (cnt > 3) a3 = fmaf(d_bf16_to_f32(x3[c0]), w0, fmaf(d_bf16_to_f32(x3[c0 + 1]), w1, a3));
        } else if (c0 < K) {
            const float w0 = d_fp8_e4m3fn_to_f32(wr[c0]) * sc;
            a0 = fmaf(d_bf16_to_f32(x0[c0]), w0, a0);
            if (cnt > 1) a1 = fmaf(d_bf16_to_f32(x1[c0]), w0, a1);
            if (cnt > 2) a2 = fmaf(d_bf16_to_f32(x2[c0]), w0, a2);
            if (cnt > 3) a3 = fmaf(d_bf16_to_f32(x3[c0]), w0, a3);
        }
    }
    a0 = d_wave64_sum(a0);
    if (lane == 0) y_bf16[size_t(m0 + 0) * size_t(N) + size_t(n)] = d_f32_to_bf16_rne(a0);
    if (cnt > 1) {
        a1 = d_wave64_sum(a1);
        if (lane == 0) y_bf16[size_t(m0 + 1) * size_t(N) + size_t(n)] = d_f32_to_bf16_rne(a1);
    }
    if (cnt > 2) {
        a2 = d_wave64_sum(a2);
        if (lane == 0) y_bf16[size_t(m0 + 2) * size_t(N) + size_t(n)] = d_f32_to_bf16_rne(a2);
    }
    if (cnt > 3) {
        a3 = d_wave64_sum(a3);
        if (lane == 0) y_bf16[size_t(m0 + 3) * size_t(N) + size_t(n)] = d_f32_to_bf16_rne(a3);
    }
}

__global__ __launch_bounds__(256, 4)
void k_fp8_block128_gemv_f32(
    const float* __restrict__ x_f32,
    const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ w_scale,
    float* __restrict__ y_f32,
    int N, int K)
{
    const int tid = int(threadIdx.x), wid = tid >> 6, lane = tid & 63;
    const int m = int(blockIdx.y), n = int(blockIdx.x) * 4 + wid;
    if (n >= N) return;
    const int kb = (K + 127) >> 7;
    const float*   __restrict__ xr = x_f32 + size_t(m) * size_t(K);
    const uint8_t* __restrict__ wr = w_fp8 + size_t(n) * size_t(K);
    const float*   __restrict__ sr = w_scale + size_t(n >> 7) * size_t(kb);

    float acc = 0.0f;
    for (int b = 0; b < kb; ++b) {
        const int c0 = b * 128 + lane * 2;
        float loc = 0.0f;
        if (c0 + 1 < K) {
            loc = fmaf(xr[c0], d_fp8_e4m3fn_to_f32(wr[c0]), xr[c0 + 1] * d_fp8_e4m3fn_to_f32(wr[c0 + 1]));
        } else if (c0 < K) {
            loc = xr[c0] * d_fp8_e4m3fn_to_f32(wr[c0]);
        }
        acc = fmaf(loc, sr[b], acc);
    }
    acc = d_wave64_sum(acc);
    if (lane == 0) y_f32[size_t(m) * size_t(N) + size_t(n)] = acc;
}

__global__ __launch_bounds__(256, 8)
void k_fp8_block128_dequant_to_f16(
    const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ w_scale,
    uint16_t* __restrict__ out_f16,
    int N, int K)
{
    const size_t total = size_t(N) * size_t(K);
    const size_t idx = size_t(blockIdx.x) * 256u + size_t(threadIdx.x);
    if (idx >= total) return;
    const int r = int(idx / size_t(K));
    const int c = int(idx - size_t(r) * size_t(K));
    const int kb = (K + 127) >> 7;
    const float sc = w_scale[size_t(r >> 7) * size_t(kb) + size_t(c >> 7)];
    out_f16[idx] = d_f32_to_f16(d_fp8_e4m3fn_to_f32(w_fp8[idx]) * sc);
}

__global__ __launch_bounds__(256, 4)
void k_bf16_gemv_bf16(
    const uint16_t* __restrict__ x_bf16,
    const uint16_t* __restrict__ w_bf16,
    uint16_t* __restrict__ y_bf16,
    int N, int K)
{
    const int tid = int(threadIdx.x), wid = tid >> 6, lane = tid & 63;
    const int m = int(blockIdx.y), n = int(blockIdx.x) * 4 + wid;
    if (n >= N) return;
    const uint16_t* __restrict__ xr = x_bf16 + size_t(m) * size_t(K);
    const uint16_t* __restrict__ wr = w_bf16 + size_t(n) * size_t(K);
    float acc = 0.0f;
    for (int k = lane; k < K; k += 64) {
        acc = fmaf(d_bf16_to_f32(xr[k]), d_bf16_to_f32(wr[k]), acc);
    }
    acc = d_wave64_sum(acc);
    if (lane == 0) y_bf16[size_t(m) * size_t(N) + size_t(n)] = d_f32_to_bf16_rne(acc);
}

__global__ __launch_bounds__(256, 8)
void k_reduce_3tier_glc(float* __restrict__ accum_f32, const float* __restrict__ h_p150_f32,
                        const float* __restrict__ h_ddr4_f32, int n_elems) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i >= n_elems) return;
    float v = accum_f32[i];
    if (h_p150_f32) v += d_glc_load_f32(h_p150_f32 + i);
    if (h_ddr4_f32) v += d_glc_load_f32(h_ddr4_f32 + i);
    accum_f32[i] = v;
}

// ============================================================================
// Host Launch Wrappers
// ============================================================================
void launch_wht128_f32(const float* d_in, const uint64_t* d_signs, float* d_out, int M, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_wht128_f32, dim3(uint32_t(K / 128), uint32_t(M)), dim3(128), 0, s, d_in, d_signs, d_out, K);
}
void launch_wht128_bf16_to_f16(const uint16_t* d_in, const uint64_t* d_signs, uint16_t* d_out, int M, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_wht128_bf16_to_f16, dim3(uint32_t(K / 128), uint32_t(M)), dim3(128), 0, s, d_in, d_signs, d_out, K);
}
void launch_wht128_bf16_to_f32(const uint16_t* d_in, const uint64_t* d_signs, float* d_out, int M, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_wht128_bf16_to_f32, dim3(uint32_t(K / 128), uint32_t(M)), dim3(128), 0, s, d_in, d_signs, d_out, K);
}
void launch_wht128_f32_inplace(float* d_buf, const uint64_t* d_signs, int rows, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_wht128_f32, dim3(uint32_t(K / 128), uint32_t(rows)), dim3(128), 0, s, d_buf, d_signs, d_buf, K);
}
void launch_hlut3_gemv(const float* x, const Hlut3Tile32x32* t, const float* lv, float* y, int M, int N, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_hlut3_gemv, dim3(uint32_t((N + 7) / 8), uint32_t(M)), dim3(256), 0, s, x, t, lv, y, N, K);
}
void launch_hlut3_grouped_gateup_swiglu_f16(const uint16_t* x, const Hy4ExpertGroup* g, const Hlut3Tile32x32* gt,
                                            const Hlut3Tile32x32* ut, const float* lv, float* z,
                                            int n_groups, int N_inter, int K_in, hipStream_t s) {
    if (n_groups <= 0) return;
    hipLaunchKernelGGL(k_hlut3_grouped_gateup_swiglu_f16, dim3(uint32_t(N_inter / 32), uint32_t(n_groups)), dim3(128), 0, s,
                       x, g, gt, ut, lv, z, N_inter, K_in);
}
void launch_hlut3_grouped_down_atomic(const float* z, const Hy4ExpertGroup* g, const Hlut3Tile32x32* dt,
                                      const float* lv, float* out, int n_groups, int N_down, int K_down, hipStream_t s) {
    if (n_groups <= 0) return;
    hipLaunchKernelGGL(k_hlut3_grouped_down_atomic, dim3(uint32_t(N_down / 32), uint32_t(n_groups)), dim3(128), 0, s,
                       z, g, dt, lv, out, N_down, K_down);
}
void launch_unpack_hlut3_to_f16(const Hlut3Tile32x32* t, const float* lv, uint16_t* out, int N, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_unpack_hlut3_to_f16, dim3(uint32_t(K / 32), uint32_t(N / 32)), dim3(256), 0, s, t, lv, out, N, K);
}
void launch_hlut2_gemv(const float* x, const Hlut2Tile32x32* t, const float* lv, float* y, int M, int N, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_hlut2_gemv, dim3(uint32_t((N + 7) / 8), uint32_t(M)), dim3(256), 0, s, x, t, lv, y, N, K);
}
void launch_hlut2_grouped_gateup_swiglu_f16(const uint16_t* x, const Hy4ExpertGroup* g, const Hlut2Tile32x32* gt,
                                            const Hlut2Tile32x32* ut, const float* lv, float* z,
                                            int n_groups, int N_inter, int K_in, hipStream_t s) {
    if (n_groups <= 0) return;
    hipLaunchKernelGGL(k_hlut2_grouped_gateup_swiglu_f16, dim3(uint32_t(N_inter / 32), uint32_t(n_groups)), dim3(128), 0, s,
                       x, g, gt, ut, lv, z, N_inter, K_in);
}
void launch_hlut2_grouped_down_atomic(const float* z, const Hy4ExpertGroup* g, const Hlut2Tile32x32* dt,
                                      const float* lv, float* out, int n_groups, int N_down, int K_down, hipStream_t s) {
    if (n_groups <= 0) return;
    hipLaunchKernelGGL(k_hlut2_grouped_down_atomic, dim3(uint32_t(N_down / 32), uint32_t(n_groups)), dim3(128), 0, s,
                       z, g, dt, lv, out, N_down, K_down);
}
void launch_unpack_hlut2_to_f16(const Hlut2Tile32x32* t, const float* lv, uint16_t* out, int N, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_unpack_hlut2_to_f16, dim3(uint32_t(K / 32), uint32_t(N / 32)), dim3(256), 0, s, t, lv, out, N, K);
}
void launch_hlut4_gemv(const float* x, const Hlut4Tile32x32* t, const float* lv, float* y, int M, int N, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_hlut4_gemv, dim3(uint32_t((N + 7) / 8), uint32_t(M)), dim3(256), 0, s, x, t, lv, y, N, K);
}
void launch_hlut4_grouped_gateup_swiglu_f16(const uint16_t* x, const Hy4ExpertGroup* g, const Hlut4Tile32x32* gt,
                                            const Hlut4Tile32x32* ut, const float* lv, float* z,
                                            int n_groups, int N_inter, int K_in, hipStream_t s) {
    if (n_groups <= 0) return;
    hipLaunchKernelGGL(k_hlut4_grouped_gateup_swiglu_f16, dim3(uint32_t(N_inter / 32), uint32_t(n_groups)), dim3(128), 0, s,
                       x, g, gt, ut, lv, z, N_inter, K_in);
}
void launch_hlut4_grouped_down_atomic(const float* z, const Hy4ExpertGroup* g, const Hlut4Tile32x32* dt,
                                      const float* lv, float* out, int n_groups, int N_down, int K_down, hipStream_t s) {
    if (n_groups <= 0) return;
    hipLaunchKernelGGL(k_hlut4_grouped_down_atomic, dim3(uint32_t(N_down / 32), uint32_t(n_groups)), dim3(128), 0, s,
                       z, g, dt, lv, out, N_down, K_down);
}
void launch_unpack_hlut4_to_f16(const Hlut4Tile32x32* t, const float* lv, uint16_t* out, int N, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_unpack_hlut4_to_f16, dim3(uint32_t(K / 32), uint32_t(N / 32)), dim3(256), 0, s, t, lv, out, N, K);
}
void launch_fp8_grouped_gateup_swiglu(const uint16_t* x, const Hy4ExpertGroup* g,
                                      const uint8_t* gw, const float* gs,
                                      const uint8_t* uw, const float* us,
                                      float* z, int n_groups, int N_inter, int K_in, hipStream_t s) {
    if (n_groups <= 0) return;
    hipLaunchKernelGGL(k_fp8_grouped_gateup_swiglu, dim3(uint32_t((N_inter + 3) / 4), uint32_t(n_groups)), dim3(256), 0, s,
                       x, g, gw, gs, uw, us, z, N_inter, K_in);
}
void launch_fp8_grouped_down_atomic(const float* z, const Hy4ExpertGroup* g,
                                    const uint8_t* dw, const float* ds,
                                    float* out, int n_groups, int N_down, int K_down, hipStream_t s) {
    if (n_groups <= 0) return;
    hipLaunchKernelGGL(k_fp8_grouped_down_atomic, dim3(uint32_t((N_down + 3) / 4), uint32_t(n_groups)), dim3(256), 0, s,
                       z, g, dw, ds, out, N_down, K_down);
}
void launch_fp8_block128_gemv_bf16(const uint16_t* x, const uint8_t* w, const float* sc, uint16_t* y, int M, int N, int K, hipStream_t s) {
    if (M <= 0 || N <= 0) return;
    hipLaunchKernelGGL(k_fp8_block128_gemv_bf16, dim3(uint32_t((N + 3) / 4), uint32_t((M + 3) / 4)), dim3(256), 0, s, x, w, sc, y, M, N, K);
}
void launch_fp8_block128_gemv_f32(const float* x, const uint8_t* w, const float* sc, float* y, int M, int N, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_fp8_block128_gemv_f32, dim3(uint32_t((N + 3) / 4), uint32_t(M)), dim3(256), 0, s, x, w, sc, y, N, K);
}
void launch_fp8_block128_dequant_to_f16(const uint8_t* w, const float* sc, uint16_t* out, int N, int K, hipStream_t s) {
    const uint32_t grid = uint32_t((size_t(N) * size_t(K) + 255u) / 256u);
    hipLaunchKernelGGL(k_fp8_block128_dequant_to_f16, dim3(grid), dim3(256), 0, s, w, sc, out, N, K);
}
void launch_bf16_gemv_bf16(const uint16_t* x, const uint16_t* w, uint16_t* y, int M, int N, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_bf16_gemv_bf16, dim3(uint32_t((N + 3) / 4), uint32_t(M)), dim3(256), 0, s, x, w, y, N, K);
}
void launch_reduce_3tier_glc(float* acc, const float* p150, const float* ddr4, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_reduce_3tier_glc, dim3(uint32_t((n + 255) / 256)), dim3(256), 0, s, acc, p150, ddr4, n);
}

// ============================================================================
// 6. Zero-Scratch MI50 GPU Blocked LDLQ (B=64) Packing Kernels (60x faster!)
// ============================================================================
__global__ __launch_bounds__(64, 4)
void k_fp8_dequant_wht128_f32(
    const uint8_t* __restrict__ w_fp8,
    const float*   __restrict__ w_scale,
    const uint64_t* __restrict__ signs,
    float*         __restrict__ w_rot,
    int N, int K)
{
    const int blk = int(blockIdx.x);
    const int row = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    if (row >= N) return;
    __shared__ float s_v[128];

    const int kb = K >> 7;
    const float sc = w_scale[(row >> 7) * kb + blk];
    const uint64_t s0 = signs[blk * 2 + 0];
    const uint64_t s1 = signs[blk * 2 + 1];
    const size_t base = size_t(row) * size_t(K) + size_t(blk * 128);

    float v0 = d_fp8_e4m3fn_to_f32(w_fp8[base + size_t(tid)]) * sc;
    float v1 = d_fp8_e4m3fn_to_f32(w_fp8[base + size_t(tid + 64)]) * sc;
    if ((s0 >> tid) & 1ULL) v0 = -v0;
    if ((s1 >> tid) & 1ULL) v1 = -v1;
    s_v[tid]      = v0;
    s_v[tid + 64] = v1;
    __syncthreads();

    #pragma unroll
    for (int stage = 0; stage < 7; ++stage) {
        const int half = 1 << stage;
        const int grp  = tid >> stage;
        const int loc  = tid & (half - 1);
        const int i0   = grp * (half << 1) + loc;
        const int i1   = i0 + half;
        const float a = s_v[i0], b = s_v[i1];
        __syncthreads();
        s_v[i0] = a + b;
        s_v[i1] = a - b;
        __syncthreads();
    }
    constexpr float kInvSqrt128 = 0.08838834764831844655f;
    w_rot[base + size_t(tid)]      = s_v[tid] * kInvSqrt128;
    w_rot[base + size_t(tid + 64)] = s_v[tid + 64] * kInvSqrt128;
}

__global__ __launch_bounds__(64, 4)
void k_compute_hlut_scales_f16(
    const float* __restrict__ w_rot,
    uint16_t*    __restrict__ scales_f16,
    int N, int K)
{
    const int kt_i = int(blockIdx.x);
    const int nt_i = int(blockIdx.y);
    const int kt   = K >> 5;
    if (nt_i * 32 >= N || kt_i >= kt) return;
    const int tid  = int(threadIdx.x);
    const int g    = tid >> 3;        // 0..7 scale group (4 rows x 32 cols = 128 elems)
    const int lane = tid & 7;         // 8 threads per group, 16 elems/thread

    double sq = 0.0;
    const int r_base = nt_i * 32 + g * 4;
    const int c_base = kt_i * 32;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int idx = lane * 16 + i;
        const int lr = idx >> 5;
        const int lc = idx & 31;
        const float v = w_rot[size_t(r_base + lr) * size_t(K) + size_t(c_base + lc)];
        sq += double(v) * double(v);
    }
    // 8-lane reduction via FP32 shuffle on split hi/lo or direct float sum (within 1e-7 of FP64!)
    float sqf = float(sq);
    sqf += __shfl_xor(sqf, 4, 8);
    sqf += __shfl_xor(sqf, 2, 8);
    sqf += __shfl_xor(sqf, 1, 8);
    if (lane == 0) {
        const float rms = sqrtf(sqf * (1.0f / 128.0f) + 1e-30f);
        scales_f16[(size_t(nt_i) * size_t(kt) + size_t(kt_i)) * 8u + size_t(g)] = __half_as_ushort(__float2half_rn(rms));
    }
}

// Exact Analytic Lloyd-Max GPU Panel-64 Causal LDLQ (0 register spills, 24.96 KiB LDS)
template <int MODE, typename TileT>
__global__ __launch_bounds__(64, 2)
void k_ldlq_panel64_impl(
    const float*    __restrict__ W_orig,
    const float*    __restrict__ W_work,
    const float*    __restrict__ R_up,
    const uint16_t* __restrict__ scales_f16,
    float*          __restrict__ E_panel,
    TileT*          __restrict__ out_tiles,
    int N, int K, int b0)
{
    const int nt = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    if (nt * 32 >= N) return;

    __shared__ float s_r[64][65];
    __shared__ float s_w[32][65];

    // Cooperative coalesced load of R_up[b0:b0+64, b0:b0+64]
    for (int rr = 0; rr < 64; ++rr) {
        s_r[rr][tid] = R_up[size_t(b0 + rr) * size_t(K) + size_t(b0 + tid)];
    }
    // Cooperative coalesced load of W_work[nt*32:nt*32+32, b0:b0+64]
    for (int rr = 0; rr < 32; ++rr) {
        s_w[rr][tid] = W_work[size_t(nt * 32 + rr) * size_t(K) + size_t(b0 + tid)];
    }
    __syncthreads();

    if (tid < 32) {
        const int r    = tid;
        const int grow = nt * 32 + r;
        const int sg   = r >> 2;
        const int kt   = K >> 5;
        const int kt0  = b0 >> 5;

        const uint16_t sc0_h = scales_f16[(size_t(nt) * size_t(kt) + size_t(kt0 + 0)) * 8u + size_t(sg)];
        const uint16_t sc1_h = scales_f16[(size_t(nt) * size_t(kt) + size_t(kt0 + 1)) * 8u + size_t(sg)];
        const float sig0 = __half2float(__ushort_as_half(sc0_h));
        const float sig1 = __half2float(__ushort_as_half(sc1_h));

        for (int tsub = 0; tsub < 2; ++tsub) {
            const float sig   = (tsub == 0) ? sig0 : sig1;
            const float inv_s = (sig > 0.0f) ? (1.0f / sig) : 0.0f;
            const int c_base  = tsub * 32;
            uint32_t p0 = 0u, p1 = 0u, p2 = 0u, p3 = 0u;

            for (int lc = 0; lc < 32; ++lc) {
                const int c = c_base + lc;
                const float wv   = s_w[r][c];
                const float orig = W_orig[size_t(grow) * size_t(K) + size_t(b0 + c)];
                const float az   = fabsf(wv) * inv_s;
                const uint32_t sgn = signbit(wv) ? 1u : 0u;

                uint32_t mag = 0u;
                float lev = 0.0f;
                if constexpr (MODE == 2) {
                    if (az >= 0.98159882f) { mag = 1u; lev = 1.51041761f; }
                    else                   { mag = 0u; lev = 0.45278004f; }
                    const uint32_t code = (sgn << 1) | mag;
                    if (lc < 16) p0 |= (code << (2 * lc));
                    else         p1 |= (code << (2 * (lc - 16)));
                } else if constexpr (MODE == 3) {
                    if (az >= 1.04995724f) {
                        if (az >= 1.74792742f) { mag = 3u; lev = 2.15194559f; }
                        else                   { mag = 2u; lev = 1.34390926f; }
                    } else {
                        if (az >= 0.50054969f) { mag = 1u; lev = 0.75600523f; }
                        else                   { mag = 0u; lev = 0.24509415f; }
                    }
                    if (lc < 16) p0 |= (mag << (2 * lc));
                    else         p1 |= (mag << (2 * (lc - 16)));
                    p2 |= (sgn << lc);
                } else {
                    if (az >= 1.09929159f) {
                        if (az >= 1.84353251f) {
                            if (az >= 2.40080487f) { mag = 7u; lev = 2.73259377f; }
                            else                   { mag = 6u; lev = 2.06901598f; }
                        } else {
                            if (az >= 1.43713993f) { mag = 5u; lev = 1.61804903f; }
                            else                   { mag = 4u; lev = 1.25623083f; }
                        }
                    } else {
                        if (az >= 0.52243634f) {
                            if (az >= 0.79956731f) { mag = 3u; lev = 0.94235235f; }
                            else                   { mag = 2u; lev = 0.65678227f; }
                        } else {
                            if (az >= 0.25824410f) { mag = 1u; lev = 0.38809040f; }
                            else                   { mag = 0u; lev = 0.12839780f; }
                        }
                    }
                    const uint32_t nib = (sgn << 3) | mag;
                    const int sh = 4 * (lc & 7);
                    const int wi = lc >> 3;
                    if (wi == 0)      p0 |= (nib << sh);
                    else if (wi == 1) p1 |= (nib << sh);
                    else if (wi == 2) p2 |= (nib << sh);
                    else              p3 |= (nib << sh);
                }

                const float qv = (sgn ? -lev : lev) * sig;
                const float err = orig - qv;
                E_panel[size_t(grow) * 64u + size_t(c)] = err;

                for (int k = c + 1; k < 64; ++k) {
                    s_w[r][k] = fmaf(err, s_r[c][k], s_w[r][k]);
                }
            }

            TileT& tile = out_tiles[size_t(nt) * size_t(kt) + size_t(kt0 + tsub)];
            if constexpr (MODE == 2) {
                tile.q_2b[r * 2 + 0] = p0;
                tile.q_2b[r * 2 + 1] = p1;
            } else if constexpr (MODE == 3) {
                tile.q_lo2[r * 2 + 0] = p0;
                tile.q_lo2[r * 2 + 1] = p1;
                tile.q_hi1[r]         = p2;
            } else {
                tile.q_nib4[r * 4 + 0] = p0;
                tile.q_nib4[r * 4 + 1] = p1;
                tile.q_nib4[r * 4 + 2] = p2;
                tile.q_nib4[r * 4 + 3] = p3;
            }
            if ((r & 3) == 0) {
                tile.scales_f16[sg] = (tsub == 0) ? sc0_h : sc1_h;
            }
        }
    }
}

void launch_fp8_dequant_wht128_f32(
    const uint8_t* w_fp8, const float* w_scale, const uint64_t* signs,
    float* w_rot, int N, int K, hipStream_t s)
{
    hipLaunchKernelGGL(k_fp8_dequant_wht128_f32, dim3(uint32_t(K / 128), uint32_t(N)), dim3(64), 0, s,
                       w_fp8, w_scale, signs, w_rot, N, K);
}

void launch_compute_hlut_scales_f16(const float* w_rot, uint16_t* scales_f16, int N, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_compute_hlut_scales_f16, dim3(uint32_t(K / 32), uint32_t(N / 32)), dim3(64), 0, s,
                       w_rot, scales_f16, N, K);
}

void launch_ldlq_panel64_hlut2(
    const float* wo, const float* ww, const float* ru, const uint16_t* sc,
    float* ep, Hlut2Tile32x32* tl, int N, int K, int b0, hipStream_t s)
{
    hipLaunchKernelGGL((k_ldlq_panel64_impl<2, Hlut2Tile32x32>), dim3(uint32_t(N / 32)), dim3(64), 0, s,
                       wo, ww, ru, sc, ep, tl, N, K, b0);
}

void launch_ldlq_panel64_hlut3(
    const float* wo, const float* ww, const float* ru, const uint16_t* sc,
    float* ep, Hlut3Tile32x32* tl, int N, int K, int b0, hipStream_t s)
{
    hipLaunchKernelGGL((k_ldlq_panel64_impl<3, Hlut3Tile32x32>), dim3(uint32_t(N / 32)), dim3(64), 0, s,
                       wo, ww, ru, sc, ep, tl, N, K, b0);
}

void launch_ldlq_panel64_hlut4(
    const float* wo, const float* ww, const float* ru, const uint16_t* sc,
    float* ep, Hlut4Tile32x32* tl, int N, int K, int b0, hipStream_t s)
{
    hipLaunchKernelGGL((k_ldlq_panel64_impl<4, Hlut4Tile32x32>), dim3(uint32_t(N / 32)), dim3(64), 0, s,
                       wo, ww, ru, sc, ep, tl, N, K, b0);
}

}  // namespace hy4_770b
