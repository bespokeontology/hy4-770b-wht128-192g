#pragma once
// Zero-Scratch gfx906 Vega 20 Core Backbone, Lightning Indexer, Gated MLA & DFly2 HIP Kernels
// Model: tencent/Hy4 (770B: H=6144, 78 layers, 64 heads, 21 full + 57 shared IndexCache, 3.3B DFly2)
// 100% ScratchSize == 0, 0 SGPR/VGPR spills, 0 device trig, 0 slc/system RMW atomics.

#include "../src/hy4_770b_config.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>

namespace hy4_770b {

// ---- 1. Embedding Expand & iHC (4-Stream FP32, H=6144, flat=24576) ----
void launch_embed_expand_hc(
    const int32_t* d_tok_ids, const uint16_t* d_embed_bf16,
    uint16_t* d_streams_bf16, int M, int H, int hc_mult, hipStream_t stream);

void launch_ihc_pre(
    const uint16_t* d_streams_bf16, const float* d_fn_f32,
    const float* d_base_f32, const float* d_scale_f32,
    uint16_t* d_sub_in_bf16, float* d_post_f32,
    int M, float rms_eps, float hc_eps, float hc_mag, hipStream_t stream);

void launch_ihc_post_add(
    uint16_t* d_streams_bf16, const uint16_t* d_sub_out_bf16,
    const float* d_post_f32, int M, int H, int hc_mult, hipStream_t stream);

void launch_ihc_head_and_norm(
    const uint16_t* d_streams_bf16, const float* d_fn_f32,
    const float* d_base_f32, const float* d_scale_f32,
    const uint16_t* d_norm_w_bf16, uint16_t* d_out_bf16,
    int M, float rms_eps, float hc_eps, hipStream_t stream);

// ---- 2. Norms, Elementwise, Conversions ----
void launch_rmsnorm_bf16(
    const uint16_t* d_in_bf16, const uint16_t* d_w_bf16,
    uint16_t* d_out_bf16, int rows, int dim, float eps, hipStream_t stream);

void launch_swiglu_unclamped_bf16(
    const uint16_t* d_gate, const uint16_t* d_up,
    uint16_t* d_out, int n, hipStream_t stream);

void launch_swiglu_clamped10_f32(
    const float* d_gate, const float* d_up,
    float* d_out, int n, hipStream_t stream);

void launch_gate_sigmoid_mul_bf16(
    const uint16_t* d_attn_out, const uint16_t* d_gate_states,
    uint16_t* d_out, int n, hipStream_t stream);

void launch_f32_to_bf16(const float* d_in, uint16_t* d_out, int n, hipStream_t stream);
void launch_bf16_to_f32(const uint16_t* d_in, float* d_out, int n, hipStream_t stream);
void launch_bf16_to_f16(const uint16_t* d_in, uint16_t* d_out, int n, hipStream_t stream);
void launch_f16_to_bf16(const uint16_t* d_in, uint16_t* d_out, int n, hipStream_t stream);
void launch_add_bf16(const uint16_t* d_a, const uint16_t* d_b, uint16_t* d_out, int n, hipStream_t stream);
void launch_add_f32_inplace(const float* d_a, float* d_inout, int n, hipStream_t stream);

// ---- 3. Interleaved MRoPE (0 device trig: factored lo[1024*32] + hi table) ----
void launch_rope_interleaved_trailing64(
    uint16_t* d_x_bf16, const int32_t* d_pos_ids,
    const float* d_cos_tbl, const float* d_sin_tbl,
    int M, int n_heads, int head_dim, hipStream_t stream);

// ---- 4. KV Cache Append (FP8-g128 c_kv + BF16 k_pe = 656 B/slot, OR lossless BF16 576 u16/slot) ----
void launch_kv_norm_rope_append_fp8(
    const uint16_t* d_kv_a_bf16, const uint16_t* d_kv_norm_w,
    const int32_t* d_pos_ids, const int32_t* d_write_slots,
    const float* d_cos_tbl, const float* d_sin_tbl,
    uint8_t* d_kv_cache_fp8, int M, float eps, hipStream_t stream);

void launch_kv_norm_rope_append_bf16(
    const uint16_t* d_kv_a_bf16, const uint16_t* d_kv_norm_w,
    const int32_t* d_pos_ids, const int32_t* d_write_slots,
    const float* d_cos_tbl, const float* d_sin_tbl,
    uint16_t* d_kv_cache_bf16, int M, float eps, hipStream_t stream);

// ---- 5. Lightning Indexer (32h x 128d, FP32 LayerNorm k_norm, weights_proj/64) + Top-2048 IndexCache ----
void launch_indexer_k_norm_rope_append(
    const uint16_t* d_k_in_bf16,
    const uint16_t* d_knorm_w_bf16, const uint16_t* d_knorm_b_bf16,
    const int32_t* d_pos_ids, const int32_t* d_write_slots,
    const float* d_cos_tbl, const float* d_sin_tbl,
    uint8_t* d_idx_k_cache_fp8, int M, float eps, hipStream_t stream);

void launch_indexer_score_tokens(
    const uint16_t* d_q_bf16,       // [M, 32, 128] (already trailing-64 RoPEd)
    const uint8_t*  d_k_cache_fp8,  // [num_k, 132] (128 B FP8 + 4 B F32 scale)
    const uint16_t* d_hw_bf16,      // [M, 32] (raw weights_proj output; scaled by 1/64 inside kernel)
    const int32_t*  d_query_slots,  // [M]
    const int32_t*  d_key_global_pos, // [num_k] or nullptr if slot == global pos
    float*          d_scores_f32,   // [M, num_k]
    int M, int num_k, hipStream_t stream);

void launch_indexer_select_topk2048(
    const float*   d_scores_f32,    // [M, num_k]
    const int32_t* d_query_slots,   // [M]
    int32_t*       d_out_indices,   // [M, 2048]
    int32_t*       d_out_counts,    // [M]
    int M, int num_k, hipStream_t stream);

// ---- 6. Absorbed Gated MLA Attention with Learnable Sinks (FP8-g128 or BF16 KV cache) ----
void launch_mla_absorbed_sparse_fp8(
    const uint16_t* d_q_abs_bf16,    // [M, H, 576]
    const uint8_t*  d_kv_cache_fp8,  // [cache_cap, 656]
    const float*    d_sinks_f32,     // [H]
    const int32_t*  d_query_slots,   // [M]
    const int32_t*  d_recalled_idx,  // [M, 2048] or nullptr for dense
    const int32_t*  d_recalled_cnt,  // [M]       or nullptr for dense
    uint16_t*       d_out_lat_bf16,  // [M, H, 512]
    int M, int H, int kv_len, float scale, hipStream_t stream);

void launch_mla_absorbed_sparse_bf16(
    const uint16_t* d_q_abs_bf16,    // [M, H, 576]
    const uint16_t* d_kv_cache_bf16, // [cache_cap, 576]
    const float*    d_sinks_f32,     // [H]
    const int32_t*  d_query_slots,   // [M]
    const int32_t*  d_recalled_idx,  // [M, 2048] or nullptr for dense
    const int32_t*  d_recalled_cnt,  // [M]       or nullptr for dense
    uint16_t*       d_out_lat_bf16,  // [M, H, 512]
    int M, int H, int kv_len, float scale, hipStream_t stream);

// Sequence-sharded FlashDecoding MLA partial pass across 5x MI50 (3:3:3:1:1 KV arena, 0 KV PCIe copies!)
void launch_mla_absorbed_partial_fp8(
    const uint16_t* d_q_abs_bf16,    // [M, H, 576]
    const uint8_t*  d_kv_cache_fp8,  // [local_cap, 656]
    const int32_t*  d_local_to_glob, // [local_cap] global token pos for causal mask
    const int32_t*  d_query_pos,     // [M] global query pos (max valid kv slot)
    const int32_t*  d_local_idx,     // [M, max_rec] local slot IDs on this rank, or nullptr for all local_len
    const int32_t*  d_local_cnt,     // [M] local selected count, or nullptr
    float*          d_out_max_f32,   // [M, H]
    float*          d_out_sum_f32,   // [M, H]
    float*          d_out_acc_f32,   // [M, H, 512] (unnormalized sum exp(logit - max) * c_kv)
    int M, int H, int local_kv_len, int max_rec, float scale, hipStream_t stream,
    const int32_t*  d_query_min_pos = nullptr);

void launch_mla_absorbed_partial_bf16(
    const uint16_t* d_q_abs_bf16,    // [M, H, 576]
    const uint16_t* d_kv_cache_bf16, // [local_cap, 576] lossless BF16 for massive-activation L70..L77
    const int32_t*  d_local_to_glob, // [local_cap]
    const int32_t*  d_query_pos,     // [M]
    const int32_t*  d_local_idx,     // [M, max_rec] or nullptr
    const int32_t*  d_local_cnt,     // [M] or nullptr
    float*          d_out_max_f32,   // [M, H]
    float*          d_out_sum_f32,   // [M, H]
    float*          d_out_acc_f32,   // [M, H, 512]
    int M, int H, int local_kv_len, int max_rec, float scale, hipStream_t stream,
    const int32_t*  d_query_min_pos = nullptr);

// ---- 7. MoE Top-8 / 256 Sigmoid Router (e_score_correction_bias + norm_topk_prob + 2.827 scale) ----
void launch_moe_router_top8_256(
    const uint16_t* d_logits_bf16,   // [M, 256]
    const float*    d_corr_bias_f32, // [256]
    int32_t*        d_topk_ids,      // [M, 8]
    float*          d_topk_weights,  // [M, 8]
    int M, float routed_scale, hipStream_t stream);

// ---- 8. DFly2 Fusion & HiddenCorrection Helpers ----
void launch_dfly2_dual_rmsnorm_concat_bf16(
    const uint16_t* d_a_bf16, const uint16_t* d_wa_bf16,
    const uint16_t* d_b_bf16, const uint16_t* d_wb_bf16,
    uint16_t* d_out_cat_bf16, int M, int H, float eps, hipStream_t stream);

void launch_dfly2_gate_residual_bf16(
    const uint16_t* d_anchor_bf16, const uint16_t* d_gate_bf16,
    const uint16_t* d_fused_bf16, uint16_t* d_out_bf16,
    int M, int H, hipStream_t stream);

// ---- 9. On-GPU MLA Transpose/Pack Kernels (eliminates 2 host syncs + 4 PCIe copies/layer!) ----
void launch_qb_transpose_nope_f16(
    const uint16_t* d_qb_bf16, uint16_t* d_qn_f16, int M, int H, hipStream_t stream);

void launch_assemble_q_abs_bf16(
    const float* d_q_nope_f32, const uint16_t* d_qb_bf16, uint16_t* d_q_abs_bf16, int M, int H, hipStream_t stream);

void launch_ao_transpose_gate_bf16(
    const float* d_ao_f32, const uint16_t* d_gate_bf16, uint16_t* d_out_bf16, int M, int H, hipStream_t stream);

}  // namespace hy4_770b
