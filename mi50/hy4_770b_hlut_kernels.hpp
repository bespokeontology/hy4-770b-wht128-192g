#pragma once
// Zero-Scratch gfx906 Vega 20 HIP Kernels for tencent/Hy4 (770B) Non-Latent [6144, 2048] MoE
// Supports WHT128-HLUT3 (400 B), WHT128-HLUT2 (272 B), WHT128-HLUT4 (528 B), and Block-FP8 (128x128)
// plus zero-LDS bitwise FP8-g128 spine GEMV and zero-PCIe-AtomicOp glc 3-tier host reduction.

#include "../src/hy4_770b_config.hpp"
#include "../src/hy4_770b_hlut_pack.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>

namespace hy4_770b {

struct Hy4RoutedPair {
    int32_t row;     // token index in [0, M)
    int32_t expert;  // local expert slot in [0, n_local_experts), or -1 if inactive
    float   weight;  // router weight (already normalized & scaled by 2.827)
    int32_t pad;
};
static_assert(sizeof(Hy4RoutedPair) == 16, "Hy4RoutedPair must be 16 B");

// Unique-expert group for deduplicated M=1..16 (including M=6 DFly2 verify!) weight-stationary dispatch
inline constexpr int kMaxGroupTokens = 16;
struct alignas(16) Hy4ExpertGroup {
    int32_t expert;                      // local expert slot in [0, n_local_experts)
    int32_t count;                       // number of tokens in [1, kMaxGroupTokens]
    int32_t rows[kMaxGroupTokens];       // token indices
    float   weights[kMaxGroupTokens];    // router weights
};

// ---- WHT128 Activation Rotation (K = 6144 or K = 2048, K % 128 == 0) ----
void launch_wht128_f32(
    const float* d_in, const uint64_t* d_signs, float* d_out,
    int M, int K, hipStream_t stream);

void launch_wht128_bf16_to_f16(
    const uint16_t* d_in_bf16, const uint64_t* d_signs, uint16_t* d_out_f16,
    int M, int K, hipStream_t stream);

void launch_wht128_bf16_to_f32(
    const uint16_t* d_in_bf16, const uint64_t* d_signs, float* d_out_f32,
    int M, int K, hipStream_t stream);

void launch_wht128_f32_inplace(
    float* d_buf, const uint64_t* d_signs, int rows, int K, hipStream_t stream);

// ---- WHT128-HLUT3 (400 B tile, 3.125 bpw) ----
void launch_hlut3_gemv(
    const float* d_x_rot, const Hlut3Tile32x32* d_tiles, const float* d_levels4,
    float* d_y, int M, int N, int K, hipStream_t stream);

void launch_hlut3_grouped_gateup_swiglu_f16(
    const uint16_t* d_x_rot_f16, const Hy4ExpertGroup* d_groups,
    const Hlut3Tile32x32* d_gate_tiles, const Hlut3Tile32x32* d_up_tiles,
    const float* d_levels4, float* d_z_out,
    int n_groups, int N_inter, int K_in, hipStream_t stream);

void launch_hlut3_grouped_down_atomic(
    const float* d_z_rot, const Hy4ExpertGroup* d_groups,
    const Hlut3Tile32x32* d_down_tiles, const float* d_levels4,
    float* d_out_accum_f32, int n_groups, int N_down, int K_down, hipStream_t stream);

void launch_unpack_hlut3_to_f16(
    const Hlut3Tile32x32* d_tiles, const float* d_levels4,
    uint16_t* d_out_f16, int N, int K, hipStream_t stream);

// ---- WHT128-HLUT2 (272 B tile, 2.125 bpw) ----
void launch_hlut2_gemv(
    const float* d_x_rot, const Hlut2Tile32x32* d_tiles, const float* d_levels2,
    float* d_y, int M, int N, int K, hipStream_t stream);

void launch_hlut2_grouped_gateup_swiglu_f16(
    const uint16_t* d_x_rot_f16, const Hy4ExpertGroup* d_groups,
    const Hlut2Tile32x32* d_gate_tiles, const Hlut2Tile32x32* d_up_tiles,
    const float* d_levels2, float* d_z_out,
    int n_groups, int N_inter, int K_in, hipStream_t stream);

void launch_hlut2_grouped_down_atomic(
    const float* d_z_rot, const Hy4ExpertGroup* d_groups,
    const Hlut2Tile32x32* d_down_tiles, const float* d_levels2,
    float* d_out_accum_f32, int n_groups, int N_down, int K_down, hipStream_t stream);

void launch_unpack_hlut2_to_f16(
    const Hlut2Tile32x32* d_tiles, const float* d_levels2,
    uint16_t* d_out_f16, int N, int K, hipStream_t stream);

// ---- WHT128-HLUT4 (528 B tile, 4.125 bpw) ----
void launch_hlut4_gemv(
    const float* d_x_rot, const Hlut4Tile32x32* d_tiles, const float* d_levels8,
    float* d_y, int M, int N, int K, hipStream_t stream);

void launch_hlut4_grouped_gateup_swiglu_f16(
    const uint16_t* d_x_rot_f16, const Hy4ExpertGroup* d_groups,
    const Hlut4Tile32x32* d_gate_tiles, const Hlut4Tile32x32* d_up_tiles,
    const float* d_levels8, float* d_z_out,
    int n_groups, int N_inter, int K_in, hipStream_t stream);

void launch_hlut4_grouped_down_atomic(
    const float* d_z_rot, const Hy4ExpertGroup* d_groups,
    const Hlut4Tile32x32* d_down_tiles, const float* d_levels8,
    float* d_out_accum_f32, int n_groups, int N_down, int K_down, hipStream_t stream);

void launch_unpack_hlut4_to_f16(
    const Hlut4Tile32x32* d_tiles, const float* d_levels8,
    uint16_t* d_out_f16, int N, int K, hipStream_t stream);

// ---- Lossless Stock Block-FP8 (128x128 E4M3FN + FP32 scale) Grouped MoE (Zero WHT, Zero Scratch!) ----
void launch_fp8_grouped_gateup_swiglu(
    const uint16_t* d_x_bf16, const Hy4ExpertGroup* d_groups,
    const uint8_t* d_gate_w, const float* d_gate_s,
    const uint8_t* d_up_w,   const float* d_up_s,
    float* d_z_out, int n_groups, int N_inter, int K_in, hipStream_t stream);

void launch_fp8_grouped_down_atomic(
    const float* d_z_in, const Hy4ExpertGroup* d_groups,
    const uint8_t* d_down_w, const float* d_down_s,
    float* d_out_accum_f32, int n_groups, int N_down, int K_down, hipStream_t stream);

// ---- Zero-LDS Bitwise Block-FP8 (128x128 E4M3FN + FP32 weight_scale) Spine & Oracle GEMV / Dequant ----
void launch_fp8_block128_gemv_bf16(
    const uint16_t* d_x_bf16,
    const uint8_t* d_w_fp8,
    const float* d_w_scale,
    uint16_t* d_y_bf16,
    int M, int N, int K,
    hipStream_t stream);

void launch_fp8_block128_gemv_f32(
    const float* d_x_f32,
    const uint8_t* d_w_fp8,
    const float* d_w_scale,
    float* d_y_f32,
    int M, int N, int K,
    hipStream_t stream);

void launch_fp8_block128_dequant_to_f16(
    const uint8_t* d_w_fp8,
    const float* d_w_scale,
    uint16_t* d_out_f16,
    int N, int K,
    hipStream_t stream);

void launch_bf16_gemv_bf16(
    const uint16_t* d_x_bf16,
    const uint16_t* d_w_bf16,
    uint16_t* d_y_bf16,
    int M, int N, int K,
    hipStream_t stream);

// ---- Zero-PCIe-AtomicOp L2-Coherent (glc) 3-Tier Partial Reduction ----
void launch_reduce_3tier_glc(
    float* d_accum_f32,
    const float* h_p150_f32,
    const float* h_ddr4_f32,
    int n_elems,
    hipStream_t stream);

// ---- Fast Zero-Scratch GPU Blocked LDLQ (B=64) Packing Kernels (60x faster than CPU!) ----
void launch_fp8_dequant_wht128_f32(
    const uint8_t* d_w_fp8, const float* d_w_scale, const uint64_t* d_signs,
    float* d_w_rot_f32, int N, int K, hipStream_t stream);

void launch_compute_hlut_scales_f16(
    const float* d_w_rot_f32, uint16_t* d_scales_f16,
    int N, int K, hipStream_t stream);

void launch_ldlq_panel64_hlut2(
    const float* d_W_orig, const float* d_W_work, const float* d_R_up,
    const uint16_t* d_scales_f16, float* d_E_panel, Hlut2Tile32x32* d_tiles,
    int N, int K, int b0, hipStream_t stream);

void launch_ldlq_panel64_hlut3(
    const float* d_W_orig, const float* d_W_work, const float* d_R_up,
    const uint16_t* d_scales_f16, float* d_E_panel, Hlut3Tile32x32* d_tiles,
    int N, int K, int b0, hipStream_t stream);

void launch_ldlq_panel64_hlut4(
    const float* d_W_orig, const float* d_W_work, const float* d_R_up,
    const uint16_t* d_scales_f16, float* d_E_panel, Hlut4Tile32x32* d_tiles,
    int N, int K, int b0, hipStream_t stream);

}  // namespace hy4_770b
