// Pure-C++17 Unified 3-Tier Runtime for tencent/Hy4 (770B / 49B Active / 78 Layers / 3.3B DFly2 / 1.23B ViT)
// Hardware: SB-M5-P2-192G (5x MI50 128 GiB HBM2 + 2x Blackhole P150A 64 GiB GDDR6 + 330 MiB L1 + 384 GiB DDR4)
// Zero Python anywhere. Strict -Wall -Wextra -Werror + graceful hipDeviceSynchronize() teardown.

#include "hy4_770b_runtime.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <omp.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace hy4_770b {
namespace {

// Dynamic VRAM-sorted logical rank -> physical HIP device mapping (R0..R2 = 32 GiB MI50s, R3..R4 = 16 GiB MI50s)
static int g_rank_to_hip[kNumMi50Ranks] = {0, 1, 2, 3, 4};

static inline hipError_t raw_hip_set_device(int phys_dev) noexcept {
    return ::hipSetDevice(phys_dev);
}
static inline hipError_t hy4_set_rank_dev(int rank) noexcept {
    const int d = (rank >= 0 && rank < kNumMi50Ranks) ? g_rank_to_hip[rank] : rank;
    return raw_hip_set_device(d);
}
#define hipSetDevice(r) hy4_set_rank_dev(r)

inline void check_hip_r(hipError_t e, const char* msg) {
    if (e != hipSuccess) {
        throw std::runtime_error(std::string("Hy4770bRuntime HIP error at ") + msg + ": " + hipGetErrorString(e));
    }
}
inline void check_rb_r(rocblas_status s, const char* msg) {
    if (s != rocblas_status_success) {
        throw std::runtime_error(std::string("Hy4770bRuntime rocBLAS error at ") + msg + ": status=" + std::to_string(int(s)));
    }
}

static std::string ltag(int l) {
    char b[16];
    std::snprintf(b, sizeof(b), "%02d", l);
    return b;
}

template <typename T>
struct DevMem {
    T* ptr = nullptr;
    size_t count = 0;
    int dev = -1;

    void alloc(int device_id, size_t n) {
        if (n == 0) return;
        if (ptr && dev == device_id && count >= n) return;
        release();
        dev = device_id;
        check_hip_r(hipSetDevice(dev), "DevMem::setDevice");
        check_hip_r(hipMalloc(&ptr, n * sizeof(T)), "DevMem::hipMalloc");
        count = n;
    }
    void release() noexcept {
        if (ptr) {
            if (dev >= 0) (void)hipSetDevice(dev);
            (void)hipFree(ptr);
            ptr = nullptr;
            count = 0;
            dev = -1;
        }
    }
    ~DevMem() { release(); }
    DevMem() = default;
    DevMem(const DevMem&) = delete;
    DevMem& operator=(const DevMem&) = delete;
    DevMem(DevMem&& o) noexcept : ptr(o.ptr), count(o.count), dev(o.dev) { o.ptr = nullptr; o.count = 0; o.dev = -1; }
    DevMem& operator=(DevMem&& o) noexcept {
        if (this != &o) { release(); ptr = o.ptr; count = o.count; dev = o.dev; o.ptr = nullptr; o.count = 0; o.dev = -1; }
        return *this;
    }
};

struct Fp8MatDev {
    DevMem<uint8_t> w;
    DevMem<float>   s;
    int N = 0;
    int K = 0;
};

// Build factored MRoPE cos/sin table matching k_rope_interleaved_trailing64:
//   lo[0..1023][32] = cos/sin(p_lo * inv_freq), hi[0..hi_max][32] = cos/sin((p_hi*1024) * inv_freq)
static void build_factored_rope_tables(int max_pos, float theta, std::vector<float>& cos_tbl, std::vector<float>& sin_tbl) {
    const int n_hi = (max_pos + 1023) / 1024 + 2;
    const size_t total_rows = size_t(1024 + n_hi);
    cos_tbl.resize(total_rows * 32u);
    sin_tbl.resize(total_rows * 32u);
    double inv_f[32];
    for (int i = 0; i < 32; ++i) inv_f[i] = 1.0 / std::pow(double(theta), double(2 * i) / 64.0);
    for (int p = 0; p < 1024; ++p) {
        for (int i = 0; i < 32; ++i) {
            const double ang = double(p) * inv_f[i];
            cos_tbl[size_t(p) * 32u + size_t(i)] = float(std::cos(ang));
            sin_tbl[size_t(p) * 32u + size_t(i)] = float(std::sin(ang));
        }
    }
    for (int h = 0; h < n_hi; ++h) {
        const double base = double(h) * 1024.0;
        for (int i = 0; i < 32; ++i) {
            const double ang = base * inv_f[i];
            cos_tbl[size_t(1024 + h) * 32u + size_t(i)] = float(std::cos(ang));
            sin_tbl[size_t(1024 + h) * 32u + size_t(i)] = float(std::sin(ang));
        }
    }
}

struct ExpertRouteEntry {
    ExpertDevice dev    = ExpertDevice::CPU_DDR4;
    ExpertFormat fmt    = ExpertFormat::HLUT4;
    int32_t      slot   = -1;  // local slot on target device (for P150: [0..n_h3) HLUT3, [n_h3..n_h3+n_h2) HLUT2)
};

struct LayerWeightsRank {
    // 2:2:2:1:1 TP5 spine slices on this rank
    Fp8MatDev q_a;         // [u*256, 6144]
    Fp8MatDev q_b;         // [h_cnt*256, 2048]
    DevMem<uint16_t> w_uk; // [h_cnt, 192, 512] FP16
    DevMem<uint16_t> w_uv; // [h_cnt, 256, 512] FP16
    Fp8MatDev linear_gate; // [h_cnt*256, 6144]
    Fp8MatDev o_proj;      // [6144, h_cnt*256]
    DevMem<float> sinks;   // [h_cnt] FP32

    // L0 dense MLP OR L1..77 shared expert (2:2:2:1:1 TP5)
    Fp8MatDev mlp_gate;    // [u*2304 (L0) or u*256 (shared), 6144]
    Fp8MatDev mlp_up;
    Fp8MatDev mlp_down;    // [6144, u*2304 or u*256]

    // Lightning Indexer wq_b TP5 head slice + tiny rank-local norms (zero per-layer hipMalloc!)
    DevMem<uint16_t> kva_ln;        // [512] BF16 (replicated 1 KiB/layer on every rank for zero-alloc KV append)
    DevMem<uint16_t> idx_wq_b_f16;  // [u*512, 2048] FP16
    DevMem<uint16_t> idx_knorm_w;   // [128] BF16
    DevMem<uint16_t> idx_knorm_b;   // [128] BF16

    // MI50 EP Routed Experts on this rank (L1..77)
    int n_fp8 = 0;
    int n_h4 = 0;
    int n_h3 = 0;
    int n_h2 = 0;
    DevMem<uint8_t> fp8_gate_w, fp8_up_w, fp8_down_w;
    DevMem<float>   fp8_gate_s, fp8_up_s, fp8_down_s;
    DevMem<Hlut4Tile32x32> h4_gate, h4_up, h4_down;
    DevMem<Hlut3Tile32x32> h3_gate, h3_up, h3_down;
    DevMem<Hlut2Tile32x32> h2_gate, h2_up, h2_down;
    DevMem<uint64_t> signs_h, signs_i;
};

struct LayerR0Extras {
    // Tiny norms, iHC, kv_a, router, and full-indexer small params (only on Rank 0!)
    DevMem<float>    hc_attn_fn, hc_attn_base, hc_attn_scale;
    DevMem<float>    hc_mlp_fn,  hc_mlp_base,  hc_mlp_scale;
    DevMem<uint16_t> input_ln, post_ln, qa_ln, kva_ln;
    Fp8MatDev        kv_a;             // [576, 6144]
    DevMem<uint16_t> router_w_bf16;    // [256, 6144] BF16
    DevMem<float>    router_bias_f32;  // [256] FP32
    // Full indexer extras (only when is_full_indexer_layer(l))
    DevMem<uint16_t> idx_wk_bf16;      // [128, 6144]
    DevMem<uint16_t> idx_knorm_w;      // [128]
    DevMem<uint16_t> idx_knorm_b;      // [128]
    DevMem<uint16_t> idx_wproj_bf16;   // [32, 6144]
};

struct DflyLayerRank {
    Fp8MatDev q_a, q_b, linear_gate, o_proj;
    DevMem<uint16_t> w_uk, w_uv;
    DevMem<float>    sinks;
    Fp8MatDev mlp_gate, mlp_up, mlp_down;  // Dense I=18432 sharded 2:2:2:1:1!
    // R0 only:
    Fp8MatDev        kv_a;                 // [576, 6144] self_attn.kv_a_proj_with_mqa
    DevMem<uint16_t> input_ln, post_ln, qa_ln, kva_ln;
};

struct RankCtx {
    int rank = 0;
    int h_start = 0;
    int h_cnt   = 0;
    int unit    = 0;
    hipStream_t stream = nullptr;
    rocblas_handle rb  = nullptr;

    DevMem<float> d_cos, d_sin;
    DevMem<float> d_lev4, d_lev3, d_lev2;

    // Spine + EP experts
    std::array<LayerWeightsRank, kNumLayers> layers;
    // LM Head slice [vocab_slice, 6144] FP16
    int vocab_start = 0;
    int vocab_cnt   = 0;
    DevMem<uint16_t> lm_head_f16;

    // 3:3:3:1:1 Sequence-Sharded KV Arena
    int local_kv_cap = 0;
    DevMem<uint8_t>  kv_cache_fp8;      // [78, local_kv_cap, 656]
    DevMem<uint16_t> kv_cache_bf16;     // [4 (L74..L77), local_kv_cap, 576] lossless BF16 for massive-activation tail!
    DevMem<uint8_t>  idx_k_cache_fp8;   // [21, local_kv_cap, 132]
    DevMem<int32_t>  local_to_global;   // [local_kv_cap]

    // DFly2 5 layers
    std::array<DflyLayerRank, kDflyNumLayers> dfly;

    // Reusable scratch workspaces (zero hipMalloc/hipFree in hot forward path!)
    DevMem<uint16_t> d_x_bf16, d_a_f16, d_w_f16, d_y_bf16, d_qa_part, d_qa_full, d_qb_out;
    DevMem<uint16_t> d_q_abs_tmp, d_q_abs_full, d_out_lat, d_attn_out, d_gate_out;
    DevMem<uint16_t> d_mlp_g, d_mlp_u, d_mlp_z, d_x_rot_f16;
    DevMem<uint16_t> d_idx_kin, d_idx_hw, d_idx_q;
    DevMem<float>    d_c_f32, d_q_nope_f32, d_x_rot_f32, d_moe_z, d_moe_acc_f32;
    DevMem<float>    d_mla_max, d_mla_sum, d_mla_acc, d_idx_scores;
    DevMem<int32_t>  d_pos, d_append_pos, d_slots, d_q_max_slot, d_q_min_slot, d_local_idx, d_local_cnt, d_topk_idx, d_topk_cnt;
    DevMem<Hy4ExpertGroup> d_groups_fp8, d_groups_h4, d_groups_h3, d_groups_h2;
};

static inline void drop_file_page_cache(const std::string& path, bool sync_first = false) noexcept {
    const int fd = ::open(path.c_str(), sync_first ? O_RDWR : O_RDONLY);
    if (fd >= 0) {
        if (sync_first) (void)::fdatasync(fd);
        (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
    }
}

static void slice_fp8_rows_to_dev(
    const hy4::SafetensorsMmap& st, const std::string& base,
    int row_beg, int row_cnt, int K, int dev, Fp8MatDev& out)
{
    const uint8_t* w_all = st.tensor_bytes(base + ".weight");
    const float*   s_all = reinterpret_cast<const float*>(st.tensor_bytes(base + ".weight_scale"));
    const int kb = (K + 127) / 128;
    const int nb_cnt = (row_cnt + 127) / 128;
    const int nb_beg = row_beg / 128;
    out.N = row_cnt;
    out.K = K;
    out.w.alloc(dev, size_t(row_cnt) * size_t(K));
    out.s.alloc(dev, size_t(nb_cnt) * size_t(kb));
    check_hip_r(hipMemcpy(out.w.ptr, w_all + size_t(row_beg) * size_t(K), size_t(row_cnt) * size_t(K), hipMemcpyHostToDevice), "fp8_row_w");
    check_hip_r(hipMemcpy(out.s.ptr, s_all + size_t(nb_beg) * size_t(kb), size_t(nb_cnt) * size_t(kb) * 4u, hipMemcpyHostToDevice), "fp8_row_s");
}

static void slice_fp8_cols_to_dev(
    const hy4::SafetensorsMmap& st, const std::string& base,
    int N, int col_beg, int col_cnt, int K_full, int dev, Fp8MatDev& out)
{
    const uint8_t* w_all = st.tensor_bytes(base + ".weight");
    const float*   s_all = reinterpret_cast<const float*>(st.tensor_bytes(base + ".weight_scale"));
    const int nb = (N + 127) / 128;
    const int kb_full = (K_full + 127) / 128;
    const int kb_beg = col_beg / 128;
    const int kb_cnt = (col_cnt + 127) / 128;

    std::vector<uint8_t> hw(size_t(N) * size_t(col_cnt));
    for (int r = 0; r < N; ++r) {
        std::memcpy(hw.data() + size_t(r) * size_t(col_cnt),
                    w_all + size_t(r) * size_t(K_full) + size_t(col_beg), size_t(col_cnt));
    }
    std::vector<float> hs(size_t(nb) * size_t(kb_cnt));
    for (int bi = 0; bi < nb; ++bi) {
        std::memcpy(hs.data() + size_t(bi) * size_t(kb_cnt),
                    s_all + size_t(bi) * size_t(kb_full) + size_t(kb_beg), size_t(kb_cnt) * 4u);
    }
    out.N = N;
    out.K = col_cnt;
    out.w.alloc(dev, hw.size());
    out.s.alloc(dev, hs.size());
    check_hip_r(hipMemcpy(out.w.ptr, hw.data(), hw.size(), hipMemcpyHostToDevice), "fp8_col_w");
    check_hip_r(hipMemcpy(out.s.ptr, hs.data(), hs.size() * 4u, hipMemcpyHostToDevice), "fp8_col_s");
}

static void load_kv_b_slice_to_uk_uv(
    const hy4::SafetensorsMmap& st, const std::string& base,
    int h_beg, int h_cnt, int dev, DevMem<uint16_t>& d_uk, DevMem<uint16_t>& d_uv)
{
    const auto& sp = st.at(base + ".weight");
    std::vector<uint16_t> h_uk(size_t(h_cnt) * 192u * 512u);
    std::vector<uint16_t> h_uv(size_t(h_cnt) * 256u * 512u);

    if (sp.dtype == "F8_E4M3") {
        const uint8_t* w = st.tensor_bytes(base + ".weight");
        const float*   s = reinterpret_cast<const float*>(st.tensor_bytes(base + ".weight_scale"));
        const auto& lut  = fp8_e4m3fn_lut();
        for (int h = 0; h < h_cnt; ++h) {
            const int gh = h_beg + h;
            for (int r = 0; r < 192; ++r) {
                const int gr = gh * 448 + r;
                for (int c = 0; c < 512; ++c) {
                    const float v = lut[w[size_t(gr) * 512u + size_t(c)]] * s[(gr / 128) * 4 + (c / 128)];
                    h_uk[(size_t(h) * 192u + size_t(r)) * 512u + size_t(c)] = f32_to_fp16_rne(v);
                }
            }
            for (int r = 0; r < 256; ++r) {
                const int gr = gh * 448 + 192 + r;
                for (int c = 0; c < 512; ++c) {
                    const float v = lut[w[size_t(gr) * 512u + size_t(c)]] * s[(gr / 128) * 4 + (c / 128)];
                    h_uv[(size_t(h) * 256u + size_t(r)) * 512u + size_t(c)] = f32_to_fp16_rne(v);
                }
            }
        }
    } else {
        const uint16_t* w = reinterpret_cast<const uint16_t*>(st.tensor_bytes(base + ".weight"));
        for (int h = 0; h < h_cnt; ++h) {
            const uint16_t* src = w + size_t(h_beg + h) * 448u * 512u;
            for (size_t i = 0; i < 192u * 512u; ++i) h_uk[size_t(h) * 192u * 512u + i] = f32_to_fp16_rne(bf16_to_f32(src[i]));
            for (size_t i = 0; i < 256u * 512u; ++i) h_uv[size_t(h) * 256u * 512u + i] = f32_to_fp16_rne(bf16_to_f32(src[192u * 512u + i]));
        }
    }
    d_uk.alloc(dev, h_uk.size());
    d_uv.alloc(dev, h_uv.size());
    check_hip_r(hipMemcpy(d_uk.ptr, h_uk.data(), h_uk.size() * 2u, hipMemcpyHostToDevice), "uk_h2d");
    check_hip_r(hipMemcpy(d_uv.ptr, h_uv.data(), h_uv.size() * 2u, hipMemcpyHostToDevice), "uv_h2d");
}

static void upload_bf16_or_f32_as_f32(
    const hy4::SafetensorsMmap& st, const std::string& name, int elem_beg, int elem_cnt, int dev, DevMem<float>& dst)
{
    const auto& sp = st.at(name);
    std::vector<float> tmp(static_cast<size_t>(elem_cnt));
    if (sp.dtype == "F32") {
        const float* p = reinterpret_cast<const float*>(st.tensor_bytes(name)) + elem_beg;
        std::memcpy(tmp.data(), p, size_t(elem_cnt) * 4u);
    } else {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(st.tensor_bytes(name)) + elem_beg;
        for (int i = 0; i < elem_cnt; ++i) tmp[size_t(i)] = bf16_to_f32(p[i]);
    }
    dst.alloc(dev, tmp.size());
    check_hip_r(hipMemcpy(dst.ptr, tmp.data(), tmp.size() * 4u, hipMemcpyHostToDevice), name.c_str());
}

static void upload_bf16(const hy4::SafetensorsMmap& st, const std::string& name, int dev, DevMem<uint16_t>& dst) {
    const auto& sp = st.at(name);
    const size_t n = size_t(sp.byte_size()) / 2u;
    dst.alloc(dev, n);
    check_hip_r(hipMemcpy(dst.ptr, st.tensor_bytes(name), size_t(sp.byte_size()), hipMemcpyHostToDevice), name.c_str());
}

static void upload_bf16_as_f16_slice(
    const hy4::SafetensorsMmap& st, const std::string& name,
    int row_beg, int row_cnt, int K, int dev, DevMem<uint16_t>& dst)
{
    const uint16_t* src = reinterpret_cast<const uint16_t*>(st.tensor_bytes(name)) + size_t(row_beg) * size_t(K);
    std::vector<uint16_t> tmp(size_t(row_cnt) * size_t(K));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = f32_to_fp16_rne(bf16_to_f32(src[i]));
    dst.alloc(dev, tmp.size());
    check_hip_r(hipMemcpy(dst.ptr, tmp.data(), tmp.size() * 2u, hipMemcpyHostToDevice), name.c_str());
}

static int32_t sample_logits_host(
    const float* logits, int vocab_size,
    const Hy4770bSamplingParams& p,
    const std::vector<int32_t>& history,
    std::mt19937_64& rng)
{
    if (p.greedy || p.temperature <= 1e-5f) {
        int32_t best = 0;
        float mx = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > mx) { mx = logits[i]; best = i; }
        }
        return best;
    }
    std::vector<float> sc(logits, logits + vocab_size);
    if (p.repetition_penalty > 1.0f + 1e-4f) {
        const size_t look = std::min<size_t>(history.size(), 128u);
        for (size_t i = history.size() - look; i < history.size(); ++i) {
            const int32_t tid = history[i];
            if (tid >= 0 && tid < vocab_size) {
                sc[size_t(tid)] = (sc[size_t(tid)] > 0.0f) ? (sc[size_t(tid)] / p.repetition_penalty)
                                                           : (sc[size_t(tid)] * p.repetition_penalty);
            }
        }
    }
    const float inv_t = 1.0f / p.temperature;
    const int k = std::clamp(p.top_k > 0 ? p.top_k : 64, 1, vocab_size);
    std::vector<std::pair<float, int32_t>> top(static_cast<size_t>(vocab_size));
    for (int i = 0; i < vocab_size; ++i) top[size_t(i)] = {sc[size_t(i)] * inv_t, i};
    std::partial_sort(top.begin(), top.begin() + k, top.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    const float mx = top[0].first;
    std::vector<double> probs(static_cast<size_t>(k));
    double sum = 0.0;
    for (int i = 0; i < k; ++i) { probs[size_t(i)] = std::exp(double(top[size_t(i)].first - mx)); sum += probs[size_t(i)]; }
    double cdf = 0.0;
    int cut = k;
    for (int i = 0; i < k; ++i) {
        probs[size_t(i)] /= sum;
        cdf += probs[size_t(i)];
        if (cdf >= double(p.top_p)) { cut = i + 1; break; }
    }
    double r = std::uniform_real_distribution<double>(0.0, cdf)(rng);
    double acc = 0.0;
    for (int i = 0; i < cut; ++i) {
        acc += probs[size_t(i)];
        if (r <= acc) return top[size_t(i)].second;
    }
    return top[size_t(cut - 1)].second;
}

}  // namespace

struct Hy4770bRuntime::Impl {
    Hy4770bRuntimeConfig cfg;
    TierPlacementPlan    plan{};
    bool initialized = false;
    bool dfly2_loaded = false;
    std::atomic<bool> shutting_down{false};

    // 5x MI50 contexts
    std::array<RankCtx, kNumMi50Ranks> ranks;
    std::array<LayerR0Extras, kNumLayers> r0_layers;

    // Shared host/R0 spine tensors
    std::vector<uint16_t> embed_host_bf16;   // [120832, 6144] in host DDR4 (1.38 GiB, 0 VRAM!)
    DevMem<float>    r0_hc_head_fn, r0_hc_head_base, r0_hc_head_scale;
    DevMem<uint16_t> r0_model_norm, r0_streams, r0_sub_in, r0_x_norm, r0_kv_a, r0_router_logits, r0_final_h;
    DevMem<float>    r0_post, r0_topk_w;
    DevMem<int32_t>  r0_topk_ids;

    // DFly2 fusion + hidden correction + final_norm + confidence_head (on R0)
    DevMem<uint16_t> dfly_enorm, dfly_hnorm, dfly_gate, dfly_eh_proj_f16;
    DevMem<uint16_t> dfly_hc_hnorm, dfly_hc_enorm, dfly_final_norm, dfly_conf_w, dfly_conf_b;
    Fp8MatDev        dfly_hc_gate, dfly_hc_up, dfly_hc_down;
    DevMem<uint16_t> dfly_d_emb, dfly_d_tgt, dfly_d_cat, dfly_d_h, dfly_d_sub, dfly_d_xnorm;
    DevMem<uint16_t> dfly_d_kv, dfly_d_prev_emb, dfly_d_hc_g, dfly_d_hc_u, dfly_d_hc_d;
    DevMem<int32_t>  dfly_d_zslots;
    DevMem<float>    dfly_d_fused_f32;

    // Host-cached learnable attention sinks (eliminates 390 blocking hipMemcpy/step!)
    std::array<std::array<float, kNumAttnHeads>, kNumLayers>     h_sinks{};
    std::array<std::array<float, kNumAttnHeads>, kDflyNumLayers> h_dfly_sinks{};

    // Per-layer O(1) routing table: route_tbl[l][orig_eid] -> {dev, fmt, local_slot}
    std::array<std::array<ExpertRouteEntry, kNumRoutedExperts>, kNumLayers> route_tbl{};

    // Tier 0 / Tier 1B: 2x Blackhole P150A IPC SHM
    Hy4770bP150ShmMapping p150_shm;
    bool p150_connected = false;
    uint32_t p150_req_seq = 0u;

    // Tier 2: 28-Core EPYC DDR4 AVX2+F16C WHT128-HLUT4 Cold-Tail Engine
    Hy4Ddr4ColdEngine ddr4_engine;
    bool ddr4_ready = false;

    // Dual-NVMe O_DIRECT .kvblk engine & 40-layer ViT
    Hy4770bNvmeKvCache kv_disk;
    Hy4770bVitEncoder  vit;

    // Live KV & IndexCache state
    int kv_len = 0;
    std::vector<int32_t> cached_toks;
    // Cached global top-2048 indices for the active forward batch (reused by 57 shared layers!)
    std::vector<int32_t> cached_idx_global; // [M_max, kIndexTopK]
    std::vector<int32_t> cached_idx_cnt;    // [M_max]
    // Last step's top-2048 indices per DFly2 target layer {61,65,69,73,77} (up to kDflyBlockSize slots for exact rollback!)
    std::array<std::array<std::vector<int32_t>, kDflyBlockSize>, kDflyNumLayers> dfly_last_idx;
    std::array<std::array<int32_t, kDflyBlockSize>, kDflyNumLayers> dfly_last_cnt{};
    int dfly_active_slot = 0;

    // Pinned host staging buffers for zero-PCIe-AtomicOp cross-rank reductions & zero-alloc decode hot path
    static constexpr int kMaxBatch = 512;
    static constexpr int kPinFastM = 16;
    float*    h_rank_part[kNumMi50Ranks]  = {};  // [kMaxBatch * 6144] FP32
    uint16_t* h_qa_part[kNumMi50Ranks]    = {};  // [kMaxBatch * 512] BF16
    uint16_t* h_qa_full                   = {};  // [kMaxBatch * 2048] BF16
    float*    h_p150_out                  = {};  // [kMaxBatch * 6144] FP32
    float*    h_ddr4_out                  = {};  // [kMaxBatch * 6144] FP32
    uint16_t* h_x_rot_bf16                = {};  // [kMaxBatch * 6144] BF16
    float*    h_x_rot_f32                 = {};  // [kMaxBatch * 6144] FP32
    uint16_t* h_out_bf16                  = {};  // [kMaxBatch * 6144] BF16 pinned output
    uint16_t* h_x_host_bf16               = {};  // [kMaxBatch * 6144] BF16 pinned input
    int32_t*  h_topk_ids_pin              = {};  // [kMaxBatch * 8] int32 pinned
    float*    h_topk_wts_pin              = {};  // [kMaxBatch * 8] FP32 pinned
    uint16_t* h_qabs_pin                  = {};  // [kPinFastM * 64 * 576] BF16 pinned
    uint16_t* h_qabs_r_pin[kNumMi50Ranks] = {};  // [kPinFastM * 16 * 576] BF16 pinned
    float*    h_p_max_pin[kNumMi50Ranks]  = {};  // [kPinFastM * 64] FP32 pinned
    float*    h_p_sum_pin[kNumMi50Ranks]  = {};  // [kPinFastM * 64] FP32 pinned
    float*    h_p_acc_pin[kNumMi50Ranks]  = {};  // [kPinFastM * 64 * 512] FP32 pinned
    uint16_t* h_olat_pin                  = {};  // [kPinFastM * 64 * 512] FP16 pinned
    uint16_t* h_ol_r_pin[kNumMi50Ranks]   = {};  // [kPinFastM * 16 * 512] FP16 pinned
    std::vector<float> h_x_raw_f32;              // [kMaxBatch * 6144] unrotated FP32
    std::array<std::vector<uint64_t>, kNumLayers> h_signs_h;  // Host-cached WHT128 signs for fast CPU WHT128!

    // Live 4-tier telemetry counters & per-stage wall-clock profiler
    std::atomic<uint64_t> stat_total_routes{0};
    std::atomic<uint64_t> stat_mi50_hits{0};
    std::atomic<uint64_t> stat_p150_hits{0};
    std::atomic<uint64_t> stat_ddr4_hits{0};
    double stat_mi50_ms = 0.0;
    double stat_p150_ms = 0.0;
    double stat_ddr4_ms = 0.0;
    Hy4770bStepProfile prof{};
    std::mutex stats_mu;

    // DPM warm guard (keeps MI50 HBM2/SCLK out of deep sleep during interactive gaps)
    std::thread dpm_thread;
    std::atomic<bool> dpm_stop{false};
    std::atomic<uint64_t> last_activity_ms{0};
    std::mutex exec_mu;

    // Optional layer-by-layer diagnostics & exact CPU HLUT3 P150A cross-checker
    std::vector<Hy4770bLayerDiag>* diag_sink = nullptr;
    bool diag_verify_p150 = false;
    Hy4770bLayerDiag cur_layer_diag{};
    int p150_h3_rank_begin = 0;

    static void cpu_eval_hlut3_expert_rows(
        const Hlut3Tile32x32* gate_tiles, const Hlut3Tile32x32* up_tiles, const Hlut3Tile32x32* dn_tiles,
        const std::vector<uint64_t>& signs_i, const float levels4[4],
        const float* x_rot_all, const std::vector<std::pair<int, float>>& rw_list, float* y_accum_all)
    {
        auto gemv_h3 = [&](const Hlut3Tile32x32* tiles, const float* x, int N, int K, float* out) {
            const int nt = N >> 5, kt = K >> 5;
            #pragma omp parallel for schedule(static) num_threads(16)
            for (int nti = 0; nti < nt; ++nti) {
                for (int r = 0; r < 32; ++r) {
                    float acc = 0.0f;
                    const int sg = r >> 2;
                    for (int kti = 0; kti < kt; ++kti) {
                        const auto& t = tiles[size_t(nti) * size_t(kt) + size_t(kti)];
                        const float sig = fp16_to_f32(t.scales_f16[sg]);
                        const float* xk = x + size_t(kti) * 32u;
                        float loc = 0.0f;
                        for (int c = 0; c < 32; ++c) {
                            const uint32_t mc = (t.q_lo2[r * 2 + (c >> 4)] >> (2 * (c & 15))) & 3u;
                            const uint32_t sb = (t.q_hi1[r] >> c) & 1u;
                            const float w = sb ? -levels4[mc] : levels4[mc];
                            loc += xk[c] * w;
                        }
                        acc += loc * sig;
                    }
                    out[nti * 32 + r] = acc;
                }
            }
        };
        std::vector<float> g(kMoeInterSize), u(kMoeInterSize), z(kMoeInterSize), y(kHiddenSize);
        for (const auto& rw : rw_list) {
            const int row = rw.first;
            const float wt = rw.second;
            const float* xr = x_rot_all + size_t(row) * size_t(kHiddenSize);
            gemv_h3(gate_tiles, xr, kMoeInterSize, kHiddenSize, g.data());
            gemv_h3(up_tiles,   xr, kMoeInterSize, kHiddenSize, u.data());
            for (int i = 0; i < kMoeInterSize; ++i) z[size_t(i)] = hy4_swiglu_clamped(g[size_t(i)], u[size_t(i)]);
            wht128_rows(z.data(), z.data(), 1, kMoeInterSize, signs_i);
            gemv_h3(dn_tiles, z.data(), kHiddenSize, kMoeInterSize, y.data());
            float* dst = y_accum_all + size_t(row) * size_t(kHiddenSize);
            for (int i = 0; i < kHiddenSize; ++i) dst[i] += wt * y[size_t(i)];
        }
    }

    uint64_t now_ms() const noexcept {
        return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    // FP8 GEMV (M<=8) or rocBLAS FP16 GEMM (M>8) on rank r
    void fp8_linear(RankCtx& rc, const Fp8MatDev& m, const uint16_t* d_x_bf16, uint16_t* d_y_bf16, int M) {
        if (M <= 8) {
            launch_fp8_block128_gemv_bf16(d_x_bf16, m.w.ptr, m.s.ptr, d_y_bf16, M, m.N, m.K, rc.stream);
        } else {
            rc.d_w_f16.alloc(rc.rank, size_t(m.N) * size_t(m.K));
            rc.d_a_f16.alloc(rc.rank, size_t(M) * size_t(m.K));
            rc.d_c_f32.alloc(rc.rank, size_t(M) * size_t(m.N));
            launch_fp8_block128_dequant_to_f16(m.w.ptr, m.s.ptr, rc.d_w_f16.ptr, m.N, m.K, rc.stream);
            launch_bf16_to_f16(d_x_bf16, rc.d_a_f16.ptr, M * m.K, rc.stream);
            const float one = 1.0f, zero = 0.0f;
            check_rb_r(rocblas_gemm_ex(
                rc.rb, rocblas_operation_transpose, rocblas_operation_none,
                m.N, M, m.K, &one,
                rc.d_w_f16.ptr, rocblas_datatype_f16_r, m.K,
                rc.d_a_f16.ptr, rocblas_datatype_f16_r, m.K,
                &zero,
                rc.d_c_f32.ptr, rocblas_datatype_f32_r, m.N,
                rc.d_c_f32.ptr, rocblas_datatype_f32_r, m.N,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "fp8_gemm_ex");
            launch_f32_to_bf16(rc.d_c_f32.ptr, d_y_bf16, M * m.N, rc.stream);
        }
    }

    void f16_linear_bf16_in(RankCtx& rc, const uint16_t* d_w_f16, const uint16_t* d_x_bf16, float* d_y_f32, int M, int N, int K) {
        rc.d_a_f16.alloc(rc.rank, size_t(M) * size_t(K));
        launch_bf16_to_f16(d_x_bf16, rc.d_a_f16.ptr, M * K, rc.stream);
        const float one = 1.0f, zero = 0.0f;
        check_rb_r(rocblas_gemm_ex(
            rc.rb, rocblas_operation_transpose, rocblas_operation_none,
            N, M, K, &one,
            d_w_f16, rocblas_datatype_f16_r, K,
            rc.d_a_f16.ptr, rocblas_datatype_f16_r, K,
            &zero,
            d_y_f32, rocblas_datatype_f32_r, N,
            d_y_f32, rocblas_datatype_f32_r, N,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "f16_linear");
    }

    // Sum 5 MI50 FP32 partials [M, 6144] -> R0 BF16 output (zero heap alloc + zero OMP spin theft for M<16!)
    void reduce_5ranks_f32_to_r0_bf16(int M, uint16_t* d_r0_out_bf16) {
        const size_t n = size_t(M) * size_t(kHiddenSize);
        const float* p0 = h_rank_part[0];
        const float* p1 = h_rank_part[1];
        const float* p2 = h_rank_part[2];
        const float* p3 = h_rank_part[3];
        const float* p4 = h_rank_part[4];
        uint16_t* dst = h_out_bf16;
        #pragma omp parallel for schedule(static) num_threads(8) if(M >= 16)
        for (size_t i = 0; i < n; ++i) {
            dst[i] = f32_to_bf16_rne(p0[i] + p1[i] + p2[i] + p3[i] + p4[i]);
        }
        check_hip_r(hipSetDevice(0), "red5_set0");
        check_hip_r(hipMemcpyAsync(d_r0_out_bf16, dst, n * 2u, hipMemcpyHostToDevice, ranks[0].stream), "red5_h2d");
    }

    void free_pinned() noexcept {
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            if (h_rank_part[r])  { (void)hipHostFree(h_rank_part[r]);  h_rank_part[r]  = nullptr; }
            if (h_qa_part[r])    { (void)hipHostFree(h_qa_part[r]);    h_qa_part[r]    = nullptr; }
            if (h_qabs_r_pin[r]) { (void)hipHostFree(h_qabs_r_pin[r]); h_qabs_r_pin[r] = nullptr; }
            if (h_p_max_pin[r])  { (void)hipHostFree(h_p_max_pin[r]);  h_p_max_pin[r]  = nullptr; }
            if (h_p_sum_pin[r])  { (void)hipHostFree(h_p_sum_pin[r]);  h_p_sum_pin[r]  = nullptr; }
            if (h_p_acc_pin[r])  { (void)hipHostFree(h_p_acc_pin[r]);  h_p_acc_pin[r]  = nullptr; }
            if (h_ol_r_pin[r])   { (void)hipHostFree(h_ol_r_pin[r]);   h_ol_r_pin[r]   = nullptr; }
        }
        if (h_qa_full)      { (void)hipHostFree(h_qa_full);      h_qa_full      = nullptr; }
        if (h_p150_out)     { (void)hipHostFree(h_p150_out);     h_p150_out     = nullptr; }
        if (h_ddr4_out)     { (void)hipHostFree(h_ddr4_out);     h_ddr4_out     = nullptr; }
        if (h_x_rot_bf16)   { (void)hipHostFree(h_x_rot_bf16);   h_x_rot_bf16   = nullptr; }
        if (h_x_rot_f32)    { (void)hipHostFree(h_x_rot_f32);    h_x_rot_f32    = nullptr; }
        if (h_out_bf16)     { (void)hipHostFree(h_out_bf16);     h_out_bf16     = nullptr; }
        if (h_x_host_bf16)  { (void)hipHostFree(h_x_host_bf16);  h_x_host_bf16  = nullptr; }
        if (h_topk_ids_pin) { (void)hipHostFree(h_topk_ids_pin); h_topk_ids_pin = nullptr; }
        if (h_topk_wts_pin) { (void)hipHostFree(h_topk_wts_pin); h_topk_wts_pin = nullptr; }
        if (h_qabs_pin)     { (void)hipHostFree(h_qabs_pin);     h_qabs_pin     = nullptr; }
        if (h_olat_pin)     { (void)hipHostFree(h_olat_pin);     h_olat_pin     = nullptr; }
    }

    void drain_and_destroy() noexcept {
        shutting_down.store(true, std::memory_order_release);
        dpm_stop.store(true, std::memory_order_release);
        if (dpm_thread.joinable()) dpm_thread.join();

        if (p150_connected && p150_shm.ptr) {
            if (cfg.shutdown_p150_on_exit) {
                // Send graceful IPC SHUTDOWN to P150A worker (NEVER kill -9!)
                auto* ch = p150_shm.ptr;
                ch->op = HY4_770B_P150_OP_SHUTDOWN;
                const uint32_t seq = ++p150_req_seq;
                ch->req_seq.store(seq, std::memory_order_release);
                const auto t0 = std::chrono::steady_clock::now();
                while (ch->resp_seq.load(std::memory_order_acquire) != seq) {
                    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 5.0) break;
                    p150_cpu_relax();
                }
            }
            p150_shm.close_mapping();
            p150_connected = false;
        }

        ddr4_engine.shutdown();
        ddr4_ready = false;

        // STRICT HARDWARE SAFETY: drain EVERY active MI50 queue via hipDeviceSynchronize() before any hipFree!
        drain_all_hip_devices(kNumMi50Ranks, "HY4_770B_RT");
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            (void)hipSetDevice(r);
            if (ranks[r].stream) { (void)hipStreamSynchronize(ranks[r].stream); }
            if (ranks[r].rb)     { (void)rocblas_destroy_handle(ranks[r].rb); ranks[r].rb = nullptr; }
            if (ranks[r].stream) { (void)hipStreamDestroy(ranks[r].stream);   ranks[r].stream = nullptr; }
        }
        free_pinned();
        initialized = false;
    }

    ~Impl() { drain_and_destroy(); }

    // ------------------------------------------------------------------------
    // Execute 1 Gated Absorbed MLA Attention Block (2:2:2:1:1 TP5 + 3:3:3:1:1 KV)
    // Shared by both Backbone layers (0..77) and DFly2 layers (kv_layer = {61,65,69,73,77})!
    // ------------------------------------------------------------------------
    void run_attention_tp5(
        int layer_l, int kv_layer, bool is_dfly, int dfly_l,
        int M, int pos_start,
        const uint16_t* d_r0_x_norm_bf16,
        uint16_t* d_r0_attn_out_bf16,
        const int32_t* h_rope_pos_ovr = nullptr,
        const int32_t* h_kv_slot_ovr = nullptr,
        const int32_t* h_min_slot_ovr = nullptr,
        int batch_total_kv_ovr = 0)
    {
        const bool batched_mode = (!is_dfly && h_rope_pos_ovr != nullptr && h_kv_slot_ovr != nullptr &&
                                   h_min_slot_ovr != nullptr && batch_total_kv_ovr > 0);

        // Copy normalized hidden [M, 6144] to pinned host staging once and broadcast to R1..R4
        const size_t x_elems = size_t(M) * size_t(kHiddenSize);
        check_hip_r(hipSetDevice(0), "attn_set0");
        check_hip_r(hipMemcpyAsync(h_x_host_bf16, d_r0_x_norm_bf16, x_elems * 2u, hipMemcpyDeviceToHost, ranks[0].stream), "xn_d2h");
        check_hip_r(hipStreamSynchronize(ranks[0].stream), "xn_sync");

        std::vector<int32_t> h_pos(static_cast<size_t>(M));
        for (int m = 0; m < M; ++m) h_pos[size_t(m)] = batched_mode ? h_rope_pos_ovr[m] : (pos_start + m);

        // 1. Broadcast x_norm + compute TP5 q_a slices on R0..R4 in parallel (and kv_a on R0 if backbone!)
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            auto& rc = ranks[r];
            check_hip_r(hipSetDevice(r), "qa_set");
            rc.d_x_bf16.alloc(r, x_elems);
            rc.d_pos.alloc(r, size_t(M));
            if (r == 0) {
                check_hip_r(hipMemcpyAsync(rc.d_x_bf16.ptr, d_r0_x_norm_bf16, x_elems * 2u, hipMemcpyDeviceToDevice, rc.stream), "x0_d2d");
            } else {
                check_hip_r(hipMemcpyAsync(rc.d_x_bf16.ptr, h_x_host_bf16, x_elems * 2u, hipMemcpyHostToDevice, rc.stream), "xr_h2d");
            }
            check_hip_r(hipMemcpyAsync(rc.d_pos.ptr, h_pos.data(), size_t(M) * 4u, hipMemcpyHostToDevice, rc.stream), "pos_h2d");

            const auto& qa_mat = is_dfly ? rc.dfly[size_t(dfly_l)].q_a : rc.layers[size_t(layer_l)].q_a;
            rc.d_qa_part.alloc(r, size_t(M) * size_t(qa_mat.N));
            fp8_linear(rc, qa_mat, rc.d_x_bf16.ptr, rc.d_qa_part.ptr, M);
            check_hip_r(hipMemcpyAsync(h_qa_part[r], rc.d_qa_part.ptr, size_t(M) * size_t(qa_mat.N) * 2u, hipMemcpyDeviceToHost, rc.stream), "qa_d2h");

            if (r == 0) {
                r0_kv_a.alloc(0, size_t(M) * size_t(kKvLatentDim));
                const auto& kva_mat = is_dfly ? rc.dfly[size_t(dfly_l)].kv_a : r0_layers[size_t(layer_l)].kv_a;
                fp8_linear(rc, kva_mat, rc.d_x_bf16.ptr, r0_kv_a.ptr, M);
            }
        }
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            check_hip_r(hipSetDevice(r), "qa_sync_set");
            check_hip_r(hipStreamSynchronize(ranks[r].stream), "qa_sync");
        }

        // Gather q_a [M, 2048] on host (tiny: 24 KiB @ M=6!)
        for (int m = 0; m < M; ++m) {
            int off = 0;
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                const int sz = ranks[r].unit * 256;
                std::memcpy(h_qa_full + size_t(m) * size_t(kQLoraRank) + size_t(off),
                            h_qa_part[r] + size_t(m) * size_t(sz), size_t(sz) * 2u);
                off += sz;
            }
        }

        // Apply q_a_layernorm on R0 & append new KV entries into owning 3:3:3:1:1 MI50 rank (backbone only!)
        check_hip_r(hipSetDevice(0), "qaln_set0");
        ranks[0].d_qa_full.alloc(0, size_t(M) * size_t(kQLoraRank));
        check_hip_r(hipMemcpyAsync(ranks[0].d_qa_full.ptr, h_qa_full, size_t(M) * size_t(kQLoraRank) * 2u, hipMemcpyHostToDevice, ranks[0].stream), "qaf_h2d");
        const uint16_t* qa_ln_w = is_dfly ? ranks[0].dfly[size_t(dfly_l)].qa_ln.ptr : r0_layers[size_t(layer_l)].qa_ln.ptr;
        launch_rmsnorm_bf16(ranks[0].d_qa_full.ptr, qa_ln_w, ranks[0].d_qa_full.ptr, M, kQLoraRank, kRmsNormEps, ranks[0].stream);
        check_hip_r(hipMemcpyAsync(h_qa_full, ranks[0].d_qa_full.ptr, size_t(M) * size_t(kQLoraRank) * 2u, hipMemcpyDeviceToHost, ranks[0].stream), "qaf_d2h");

        std::vector<uint16_t> h_kva;
        std::vector<uint16_t> h_dfly_kv;
        if (!is_dfly) {
            h_kva.resize(size_t(M) * size_t(kKvLatentDim));
            check_hip_r(hipMemcpyAsync(h_kva.data(), r0_kv_a.ptr, h_kva.size() * 2u, hipMemcpyDeviceToHost, ranks[0].stream), "kva_d2h");
        } else {
            dfly_d_kv.alloc(0, size_t(M) * size_t(kKvLatentDim));
            launch_kv_norm_rope_append_bf16(r0_kv_a.ptr, ranks[0].dfly[size_t(dfly_l)].kva_ln.ptr,
                                             ranks[0].d_pos.ptr, dfly_d_zslots.ptr,
                                             ranks[0].d_cos.ptr, ranks[0].d_sin.ptr,
                                             dfly_d_kv.ptr, M, kRmsNormEps, ranks[0].stream);
            h_dfly_kv.resize(size_t(M) * size_t(kKvLatentDim));
            check_hip_r(hipMemcpyAsync(h_dfly_kv.data(), dfly_d_kv.ptr, h_dfly_kv.size() * 2u, hipMemcpyDeviceToHost, ranks[0].stream), "dkv_d2h");
        }
        check_hip_r(hipStreamSynchronize(ranks[0].stream), "qaln_sync");

        if (!is_dfly) {
            // Route each new token into its owning 3:3:3:1:1 MI50 rank's KV arena!
            // CRITICAL: Use dedicated rc.d_append_pos — NEVER overwrite rc.d_pos (which holds query h_pos[0..M-1]!)
            std::array<std::vector<int32_t>, kNumMi50Ranks> r_pos, r_l2g, r_lslots;
            std::array<std::vector<uint16_t>, kNumMi50Ranks> r_kva;
            for (int m = 0; m < M; ++m) {
                const int rp = h_pos[size_t(m)];
                const int gp = batched_mode ? h_kv_slot_ovr[m] : (pos_start + m);
                const int ow = kv_tok_owner(gp);
                const int ls = kv_tok_local_slot(gp);
                r_pos[size_t(ow)].push_back(rp);
                r_l2g[size_t(ow)].push_back(gp);
                r_lslots[size_t(ow)].push_back(ls);
                const uint16_t* src = h_kva.data() + size_t(m) * size_t(kKvLatentDim);
                r_kva[size_t(ow)].insert(r_kva[size_t(ow)].end(), src, src + kKvLatentDim);
            }
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                const int cnt = int(r_pos[size_t(r)].size());
                if (cnt == 0) continue;
                auto& rc = ranks[r];
                check_hip_r(hipSetDevice(r), "kvapp_set");
                rc.d_y_bf16.alloc(r, r_kva[size_t(r)].size());
                rc.d_append_pos.alloc(r, size_t(cnt));
                rc.d_slots.alloc(r, size_t(cnt));
                check_hip_r(hipMemcpyAsync(rc.d_y_bf16.ptr, r_kva[size_t(r)].data(), r_kva[size_t(r)].size() * 2u, hipMemcpyHostToDevice, rc.stream), "kva_r");
                check_hip_r(hipMemcpyAsync(rc.d_append_pos.ptr, r_pos[size_t(r)].data(), size_t(cnt) * 4u, hipMemcpyHostToDevice, rc.stream), "pos_r");
                check_hip_r(hipMemcpyAsync(rc.d_slots.ptr, r_lslots[size_t(r)].data(), size_t(cnt) * 4u, hipMemcpyHostToDevice, rc.stream), "sl_r");
                uint8_t* dst_kv = rc.kv_cache_fp8.ptr + size_t(layer_l) * size_t(rc.local_kv_cap) * size_t(kKvFp8EntryBytes);
                launch_kv_norm_rope_append_fp8(rc.d_y_bf16.ptr, rc.layers[size_t(layer_l)].kva_ln.ptr,
                                               rc.d_append_pos.ptr, rc.d_slots.ptr,
                                               rc.d_cos.ptr, rc.d_sin.ptr, dst_kv, cnt, kRmsNormEps, rc.stream);
                if (layer_l >= kBf16KvStartLayer) {
                    uint16_t* dst_kv_bf16 = rc.kv_cache_bf16.ptr + size_t(layer_l - kBf16KvStartLayer) * size_t(rc.local_kv_cap) * size_t(kKvLatentDim);
                    launch_kv_norm_rope_append_bf16(rc.d_y_bf16.ptr, rc.layers[size_t(layer_l)].kva_ln.ptr,
                                                    rc.d_append_pos.ptr, rc.d_slots.ptr,
                                                    rc.d_cos.ptr, rc.d_sin.ptr, dst_kv_bf16, cnt, kRmsNormEps, rc.stream);
                }
                // Record local_to_global mapping for causal mask (in ONE contiguous copy when sequential!)
                const int ls0 = r_lslots[size_t(r)][0];
                bool contig = (r_lslots[size_t(r)].back() - ls0 + 1 == cnt);
                for (int i = 1; i < cnt && contig; ++i) {
                    if (r_lslots[size_t(r)][size_t(i)] != ls0 + i) contig = false;
                }
                if (contig) {
                    check_hip_r(hipMemcpyAsync(rc.local_to_global.ptr + ls0, r_l2g[size_t(r)].data(), size_t(cnt) * 4u, hipMemcpyHostToDevice, rc.stream), "l2g_blk");
                } else {
                    for (int i = 0; i < cnt; ++i) {
                        check_hip_r(hipMemcpyAsync(rc.local_to_global.ptr + r_lslots[size_t(r)][size_t(i)],
                                                   &r_l2g[size_t(r)][size_t(i)], 4u, hipMemcpyHostToDevice, rc.stream), "l2g");
                    }
                }
            }
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                if (r_pos[size_t(r)].empty()) continue;
                check_hip_r(hipSetDevice(r), "kvapp_sset");
                check_hip_r(hipStreamSynchronize(ranks[r].stream), "kvapp_sync");
            }
        }

        // DFly2 attends strictly to existing target KV [0..kv_len-1]; backbone includes newly appended [pos_start..pos_start+M-1]!
        const int cur_total_kv = batched_mode ? batch_total_kv_ovr : (is_dfly ? kv_len : std::max(kv_len, pos_start + M));

        // 2. Lightning Indexer + IndexCache (ONLY on 21 full backbone layers; skipped in batched <=2K mode!)
        if (!is_dfly && !batched_mode && is_full_indexer_layer(layer_l)) {
            const int islot = full_indexer_slot(layer_l);
            auto& rc0 = ranks[0];
            auto& r0e = r0_layers[size_t(layer_l)];
            check_hip_r(hipSetDevice(0), "idx_set0");
            rc0.d_idx_kin.alloc(0, size_t(M) * size_t(kIndexHeadDim));
            rc0.d_idx_hw.alloc(0, size_t(M) * size_t(kIndexNumHeads));
            launch_bf16_gemv_bf16(rc0.d_x_bf16.ptr, r0e.idx_wk_bf16.ptr, rc0.d_idx_kin.ptr, M, kIndexHeadDim, kHiddenSize, rc0.stream);
            launch_bf16_gemv_bf16(rc0.d_x_bf16.ptr, r0e.idx_wproj_bf16.ptr, rc0.d_idx_hw.ptr, M, kIndexNumHeads, kHiddenSize, rc0.stream);
            std::vector<uint16_t> h_kin(size_t(M) * size_t(kIndexHeadDim)), h_hw(size_t(M) * size_t(kIndexNumHeads));
            check_hip_r(hipMemcpyAsync(h_kin.data(), rc0.d_idx_kin.ptr, h_kin.size() * 2u, hipMemcpyDeviceToHost, rc0.stream), "kin_d2h");
            check_hip_r(hipMemcpyAsync(h_hw.data(),  rc0.d_idx_hw.ptr,  h_hw.size() * 2u,  hipMemcpyDeviceToHost, rc0.stream), "hw_d2h");
            check_hip_r(hipStreamSynchronize(rc0.stream), "idx_k_sync");

            std::array<std::vector<int32_t>, kNumMi50Ranks> ir_pos, ir_ls;
            std::array<std::vector<uint16_t>, kNumMi50Ranks> ir_kin;
            for (int m = 0; m < M; ++m) {
                const int gp = pos_start + m;
                const int ow = kv_tok_owner(gp);
                ir_pos[size_t(ow)].push_back(gp);
                ir_ls[size_t(ow)].push_back(kv_tok_local_slot(gp));
                const uint16_t* s = h_kin.data() + size_t(m) * size_t(kIndexHeadDim);
                ir_kin[size_t(ow)].insert(ir_kin[size_t(ow)].end(), s, s + kIndexHeadDim);
            }
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                if (ir_pos[size_t(r)].empty()) continue;
                auto& rc = ranks[r];
                auto& lr = rc.layers[size_t(layer_l)];
                check_hip_r(hipSetDevice(r), "idxk_set");
                rc.d_idx_kin.alloc(r, ir_kin[size_t(r)].size());
                rc.d_append_pos.alloc(r, ir_pos[size_t(r)].size());
                rc.d_slots.alloc(r, ir_ls[size_t(r)].size());
                check_hip_r(hipMemcpyAsync(rc.d_idx_kin.ptr, ir_kin[size_t(r)].data(), ir_kin[size_t(r)].size() * 2u, hipMemcpyHostToDevice, rc.stream), "rkin");
                check_hip_r(hipMemcpyAsync(rc.d_append_pos.ptr, ir_pos[size_t(r)].data(), ir_pos[size_t(r)].size() * 4u, hipMemcpyHostToDevice, rc.stream), "rp");
                check_hip_r(hipMemcpyAsync(rc.d_slots.ptr, ir_ls[size_t(r)].data(), ir_ls[size_t(r)].size() * 4u, hipMemcpyHostToDevice, rc.stream), "rls");
                uint8_t* dst_ik = rc.idx_k_cache_fp8.ptr + size_t(islot) * size_t(rc.local_kv_cap) * size_t(kIdxKEntryBytes);
                launch_indexer_k_norm_rope_append(rc.d_idx_kin.ptr, lr.idx_knorm_w.ptr, lr.idx_knorm_b.ptr,
                                                  rc.d_append_pos.ptr, rc.d_slots.ptr,
                                                  rc.d_cos.ptr, rc.d_sin.ptr, dst_ik, int(ir_pos[size_t(r)].size()), kIndexLayerNormEps, rc.stream);
            }
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                if (ir_pos[size_t(r)].empty()) continue;
                check_hip_r(hipSetDevice(r), "idxk_sset");
                check_hip_r(hipStreamSynchronize(ranks[r].stream), "idxk_sync");
            }

            if (cur_total_kv > kIndexTopK) {
                // Compute q_idx [M, 32, 128] across TP5 wq_b slices -> gather -> score 3:3:3:1:1 shards in parallel!
                std::vector<uint16_t> h_qidx(size_t(M) * size_t(kIndexNumHeads) * size_t(kIndexHeadDim));
                for (int r = 0; r < kNumMi50Ranks; ++r) {
                    auto& rc = ranks[r];
                    const int nh_r = rc.unit * 4;  // 8,8,8,4,4
                    check_hip_r(hipSetDevice(r), "qidx_set");
                    rc.d_qa_full.alloc(r, size_t(M) * size_t(kQLoraRank));
                    rc.d_c_f32.alloc(r, size_t(M) * size_t(nh_r) * size_t(kIndexHeadDim));
                    rc.d_y_bf16.alloc(r, size_t(M) * size_t(nh_r) * size_t(kIndexHeadDim));
                    check_hip_r(hipMemcpyAsync(rc.d_qa_full.ptr, h_qa_full, size_t(M) * size_t(kQLoraRank) * 2u, hipMemcpyHostToDevice, rc.stream), "qaf");
                    f16_linear_bf16_in(rc, rc.layers[size_t(layer_l)].idx_wq_b_f16.ptr, rc.d_qa_full.ptr, rc.d_c_f32.ptr, M, nh_r * kIndexHeadDim, kQLoraRank);
                    launch_f32_to_bf16(rc.d_c_f32.ptr, rc.d_y_bf16.ptr, M * nh_r * kIndexHeadDim, rc.stream);
                    launch_rope_interleaved_trailing64(rc.d_y_bf16.ptr, rc.d_pos.ptr, rc.d_cos.ptr, rc.d_sin.ptr, M, nh_r, kIndexHeadDim, rc.stream);
                }
                int h_off = 0;
                for (int r = 0; r < kNumMi50Ranks; ++r) {
                    auto& rc = ranks[r];
                    const int nh_r = rc.unit * 4;
                    check_hip_r(hipSetDevice(r), "qidx_gset");
                    std::vector<uint16_t> tmp(size_t(M) * size_t(nh_r) * size_t(kIndexHeadDim));
                    check_hip_r(hipMemcpyAsync(tmp.data(), rc.d_y_bf16.ptr, tmp.size() * 2u, hipMemcpyDeviceToHost, rc.stream), "qidx_d2h");
                    check_hip_r(hipStreamSynchronize(rc.stream), "qidx_sync");
                    for (int m = 0; m < M; ++m) {
                        std::memcpy(h_qidx.data() + (size_t(m) * size_t(kIndexNumHeads) + size_t(h_off)) * size_t(kIndexHeadDim),
                                    tmp.data() + size_t(m) * size_t(nh_r) * size_t(kIndexHeadDim),
                                    size_t(nh_r) * size_t(kIndexHeadDim) * 2u);
                    }
                    h_off += nh_r;
                }

                // Parallel local scoring + local top-2048 selection across all 5 MI50s (5 TB/s HBM2!)
                std::array<std::vector<int32_t>, kNumMi50Ranks> r_top_idx, r_top_cnt;
                std::array<std::vector<float>,   kNumMi50Ranks> r_scores;
                for (int r = 0; r < kNumMi50Ranks; ++r) {
                    auto& rc = ranks[r];
                    const int l_kv = kv_rank_local_toks(r, cur_total_kv);
                    if (l_kv <= 0) continue;
                    check_hip_r(hipSetDevice(r), "idxs_set");
                    rc.d_idx_q.alloc(r, h_qidx.size());
                    rc.d_idx_hw.alloc(r, h_hw.size());
                    rc.d_idx_scores.alloc(r, size_t(M) * size_t(l_kv));
                    rc.d_topk_idx.alloc(r, size_t(M) * size_t(kIndexTopK));
                    rc.d_topk_cnt.alloc(r, size_t(M));
                    check_hip_r(hipMemcpyAsync(rc.d_idx_q.ptr,  h_qidx.data(), h_qidx.size() * 2u, hipMemcpyHostToDevice, rc.stream), "dq");
                    check_hip_r(hipMemcpyAsync(rc.d_idx_hw.ptr, h_hw.data(),   h_hw.size() * 2u,   hipMemcpyHostToDevice, rc.stream), "dhw");
                    const uint8_t* kc = rc.idx_k_cache_fp8.ptr + size_t(islot) * size_t(rc.local_kv_cap) * size_t(kIdxKEntryBytes);
                    launch_indexer_score_tokens(rc.d_idx_q.ptr, kc, rc.d_idx_hw.ptr, rc.d_pos.ptr, rc.local_to_global.ptr, rc.d_idx_scores.ptr, M, l_kv, rc.stream);
                    launch_indexer_select_topk2048(rc.d_idx_scores.ptr, rc.d_pos.ptr, rc.d_topk_idx.ptr, rc.d_topk_cnt.ptr, M, l_kv, rc.stream);
                    r_top_idx[size_t(r)].resize(size_t(M) * size_t(kIndexTopK));
                    r_top_cnt[size_t(r)].resize(size_t(M));
                    r_scores[size_t(r)].resize(size_t(M) * size_t(l_kv));
                    check_hip_r(hipMemcpyAsync(r_top_idx[size_t(r)].data(), rc.d_topk_idx.ptr, r_top_idx[size_t(r)].size() * 4u, hipMemcpyDeviceToHost, rc.stream), "ti");
                    check_hip_r(hipMemcpyAsync(r_top_cnt[size_t(r)].data(), rc.d_topk_cnt.ptr, size_t(M) * 4u, hipMemcpyDeviceToHost, rc.stream), "tc");
                    check_hip_r(hipMemcpyAsync(r_scores[size_t(r)].data(),  rc.d_idx_scores.ptr, r_scores[size_t(r)].size() * 4u, hipMemcpyDeviceToHost, rc.stream), "sc");
                }
                for (int r = 0; r < kNumMi50Ranks; ++r) {
                    if (kv_rank_local_toks(r, cur_total_kv) <= 0) continue;
                    check_hip_r(hipSetDevice(r), "idxs_sync_set");
                    check_hip_r(hipStreamSynchronize(ranks[r].stream), "idxs_sync");
                }

                // Merge 5 ranks' top-K lists into global exact top-2048 IndexCache!
                cached_idx_global.assign(size_t(M) * size_t(kIndexTopK), -1);
                cached_idx_cnt.assign(static_cast<size_t>(M), 0);
                for (int m = 0; m < M; ++m) {
                    std::vector<std::pair<float, int32_t>> cand;
                    cand.reserve(size_t(kIndexTopK) * 2u);
                    for (int r = 0; r < kNumMi50Ranks; ++r) {
                        const int l_kv = kv_rank_local_toks(r, cur_total_kv);
                        if (l_kv <= 0) continue;
                        const int cnt = r_top_cnt[size_t(r)][size_t(m)];
                        const int32_t* lidx = r_top_idx[size_t(r)].data() + size_t(m) * size_t(kIndexTopK);
                        const float* sc = r_scores[size_t(r)].data() + size_t(m) * size_t(l_kv);
                        for (int i = 0; i < cnt; ++i) {
                            const int lt = lidx[i];
                            const int gp = kv_local_tok_to_global(r, lt);
                            if (gp <= h_pos[size_t(m)]) cand.push_back({sc[lt], gp});
                        }
                    }
                    const int take = std::min<int>(kIndexTopK, int(cand.size()));
                    if (take < int(cand.size())) {
                        std::nth_element(cand.begin(), cand.begin() + take, cand.end(),
                                         [](const auto& a, const auto& b) { return a.first > b.first; });
                    }
                    cached_idx_cnt[size_t(m)] = take;
                    for (int i = 0; i < take; ++i) cached_idx_global[size_t(m) * size_t(kIndexTopK) + size_t(i)] = cand[size_t(i)].second;
                }
            }
        }

        // Save up to kDflyBlockSize trailing tokens' sparse indices for DFly2 target layers {61,65,69,73,77} (index-sharing!)
        if (!is_dfly && !batched_mode && cur_total_kv > kIndexTopK) {
            const int n_save = std::min(M, kDflyBlockSize);
            const int m_base = M - n_save;
            for (int dl = 0; dl < kDflyNumLayers; ++dl) {
                if (kDflyTargetKvLayers[dl] == layer_l) {
                    for (int s = 0; s < n_save; ++s) {
                        const int cnt = cached_idx_cnt[size_t(m_base + s)];
                        dfly_last_cnt[size_t(dl)][size_t(s)] = cnt;
                        dfly_last_idx[size_t(dl)][size_t(s)].assign(
                            cached_idx_global.begin() + size_t(m_base + s) * size_t(kIndexTopK),
                            cached_idx_global.begin() + size_t(m_base + s) * size_t(kIndexTopK) + size_t(cnt));
                    }
                    dfly_active_slot = n_save - 1;
                }
            }
        }

        // 3. Compute TP5 q_b_proj -> RoPE -> ON-GPU W_UK absorb -> ON-GPU Q_abs assembly across all 5 MI50s IN PARALLEL!
        //    Uses pre-allocated pinned staging buffers for M <= kPinFastM (zero heap alloc/munmap + fast pinned PCIe DMA!)
        const bool use_pin = (M <= kPinFastM);
        const size_t qabs_all_elems = size_t(M) * size_t(kNumAttnHeads) * size_t(kKvLatentDim);
        std::vector<uint16_t> h_qabs_all_vec;
        std::array<std::vector<uint16_t>, kNumMi50Ranks> h_qabs_r_vec;
        uint16_t* h_qabs_all_ptr = use_pin ? h_qabs_pin : nullptr;
        if (!use_pin) { h_qabs_all_vec.resize(qabs_all_elems); h_qabs_all_ptr = h_qabs_all_vec.data(); }
        uint16_t* h_qabs_r_ptr[kNumMi50Ranks] = {};
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            auto& rc = ranks[r];
            check_hip_r(hipSetDevice(r), "qb_set");
            rc.d_qa_full.alloc(r, size_t(M) * size_t(kQLoraRank));
            rc.d_qb_out.alloc(r, size_t(M) * size_t(rc.h_cnt) * size_t(kQkHeadDim));
            rc.d_q_abs_tmp.alloc(r, size_t(M) * size_t(rc.h_cnt) * size_t(kKvLatentDim));
            rc.d_a_f16.alloc(r, size_t(M) * size_t(rc.h_cnt) * size_t(kQkNopeHeadDim));
            rc.d_q_nope_f32.alloc(r, size_t(M) * size_t(rc.h_cnt) * size_t(kKvLoraRank));
            check_hip_r(hipMemcpyAsync(rc.d_qa_full.ptr, h_qa_full, size_t(M) * size_t(kQLoraRank) * 2u, hipMemcpyHostToDevice, rc.stream), "qaf_r");
            const auto& qb_mat = is_dfly ? rc.dfly[size_t(dfly_l)].q_b : rc.layers[size_t(layer_l)].q_b;
            const auto& w_uk   = is_dfly ? rc.dfly[size_t(dfly_l)].w_uk : rc.layers[size_t(layer_l)].w_uk;
            fp8_linear(rc, qb_mat, rc.d_qa_full.ptr, rc.d_qb_out.ptr, M);
            launch_rope_interleaved_trailing64(rc.d_qb_out.ptr, rc.d_pos.ptr, rc.d_cos.ptr, rc.d_sin.ptr, M, rc.h_cnt, kQkHeadDim, rc.stream);
            launch_qb_transpose_nope_f16(rc.d_qb_out.ptr, rc.d_a_f16.ptr, M, rc.h_cnt, rc.stream);
            const float one = 1.0f, zero = 0.0f;
            check_rb_r(rocblas_gemm_strided_batched_ex(
                rc.rb, rocblas_operation_none, rocblas_operation_none,
                512, M, 192, &one,
                w_uk.ptr, rocblas_datatype_f16_r, 512, rocblas_stride(192 * 512),
                rc.d_a_f16.ptr, rocblas_datatype_f16_r, 192, rocblas_stride(size_t(M) * 192u),
                &zero,
                rc.d_q_nope_f32.ptr, rocblas_datatype_f32_r, 512, rocblas_stride(size_t(M) * 512u),
                rc.d_q_nope_f32.ptr, rocblas_datatype_f32_r, 512, rocblas_stride(size_t(M) * 512u),
                rc.h_cnt, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "q_abs_bmm");
            launch_assemble_q_abs_bf16(rc.d_q_nope_f32.ptr, rc.d_qb_out.ptr, rc.d_q_abs_tmp.ptr, M, rc.h_cnt, rc.stream);
            const size_t r_elems = size_t(M) * size_t(rc.h_cnt) * size_t(kKvLatentDim);
            if (use_pin) {
                h_qabs_r_ptr[r] = h_qabs_r_pin[r];
            } else {
                h_qabs_r_vec[size_t(r)].resize(r_elems);
                h_qabs_r_ptr[r] = h_qabs_r_vec[size_t(r)].data();
            }
            check_hip_r(hipMemcpyAsync(h_qabs_r_ptr[r], rc.d_q_abs_tmp.ptr, r_elems * 2u, hipMemcpyDeviceToHost, rc.stream), "qabs_d2h");
        }
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            auto& rc = ranks[r];
            check_hip_r(hipSetDevice(r), "qabs_sset");
            check_hip_r(hipStreamSynchronize(rc.stream), "qabs_sync");
            const size_t row_elems = size_t(rc.h_cnt) * size_t(kKvLatentDim);
            for (int m = 0; m < M; ++m) {
                std::memcpy(h_qabs_all_ptr + (size_t(m) * size_t(kNumAttnHeads) + size_t(rc.h_start)) * size_t(kKvLatentDim),
                            h_qabs_r_ptr[r] + size_t(m) * row_elems,
                            row_elems * 2u);
            }
        }

        // 4. Sequence-Parallel FlashDecoding MLA over 3:3:3:1:1 HBM2 KV (ZERO KV PCIe copies up to 1M context!)
        const bool use_sparse = (!batched_mode && cur_total_kv > kIndexTopK);
        std::array<std::vector<int32_t>, kNumMi50Ranks> r_lidx, r_lcnt;
        if (use_sparse) {
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                r_lidx[size_t(r)].assign(size_t(M) * size_t(kIndexTopK), -1);
                r_lcnt[size_t(r)].assign(static_cast<size_t>(M), 0);
            }
            const int dslot = std::clamp(dfly_active_slot, 0, kDflyBlockSize - 1);
            for (int m = 0; m < M; ++m) {
                const int32_t* gptr = is_dfly ? dfly_last_idx[size_t(dfly_l)][size_t(dslot)].data()
                                              : (cached_idx_global.data() + size_t(m) * size_t(kIndexTopK));
                const int gcnt = is_dfly ? dfly_last_cnt[size_t(dfly_l)][size_t(dslot)] : cached_idx_cnt[size_t(m)];
                for (int i = 0; i < gcnt; ++i) {
                    const int gp = gptr[i];
                    if (gp < 0 || gp > h_pos[size_t(m)]) continue;
                    const int ow = kv_tok_owner(gp);
                    const int ls = kv_tok_local_slot(gp);
                    const int c = r_lcnt[size_t(ow)][size_t(m)]++;
                    if (c < kIndexTopK) r_lidx[size_t(ow)][size_t(m) * size_t(kIndexTopK) + size_t(c)] = ls;
                }
            }
        }

        const size_t mh_tot  = size_t(M) * size_t(kNumAttnHeads);
        const size_t acc_tot = mh_tot * size_t(kKvLoraRank);
        std::array<std::vector<float>, kNumMi50Ranks> p_max_vec, p_sum_vec, p_acc_vec;
        float* p_max_ptr[kNumMi50Ranks] = {};
        float* p_sum_ptr[kNumMi50Ranks] = {};
        float* p_acc_ptr[kNumMi50Ranks] = {};
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            auto& rc = ranks[r];
            const int l_kv = kv_rank_local_toks(r, cur_total_kv);
            if (use_pin) {
                p_max_ptr[r] = h_p_max_pin[r];
                p_sum_ptr[r] = h_p_sum_pin[r];
                p_acc_ptr[r] = h_p_acc_pin[r];
                if (l_kv <= 0) {
                    for (size_t i = 0; i < mh_tot; ++i) { p_max_ptr[r][i] = -1e30f; p_sum_ptr[r][i] = 0.0f; }
                }
            } else {
                p_max_vec[size_t(r)].assign(mh_tot, -1e30f);
                p_sum_vec[size_t(r)].assign(mh_tot, 0.0f);
                p_acc_vec[size_t(r)].assign(acc_tot, 0.0f);
                p_max_ptr[r] = p_max_vec[size_t(r)].data();
                p_sum_ptr[r] = p_sum_vec[size_t(r)].data();
                p_acc_ptr[r] = p_acc_vec[size_t(r)].data();
            }
            if (l_kv <= 0) continue;

            check_hip_r(hipSetDevice(r), "mla_set");
            rc.d_q_abs_full.alloc(r, qabs_all_elems);
            rc.d_mla_max.alloc(r, mh_tot);
            rc.d_mla_sum.alloc(r, mh_tot);
            rc.d_mla_acc.alloc(r, acc_tot);
            check_hip_r(hipMemcpyAsync(rc.d_q_abs_full.ptr, h_qabs_all_ptr, qabs_all_elems * 2u, hipMemcpyHostToDevice, rc.stream), "qabs_all");
            int32_t* d_li = nullptr;
            int32_t* d_lc = nullptr;
            if (use_sparse) {
                rc.d_local_idx.alloc(r, r_lidx[size_t(r)].size());
                rc.d_local_cnt.alloc(r, size_t(M));
                check_hip_r(hipMemcpyAsync(rc.d_local_idx.ptr, r_lidx[size_t(r)].data(), r_lidx[size_t(r)].size() * 4u, hipMemcpyHostToDevice, rc.stream), "li");
                check_hip_r(hipMemcpyAsync(rc.d_local_cnt.ptr, r_lcnt[size_t(r)].data(), size_t(M) * 4u, hipMemcpyHostToDevice, rc.stream), "lc");
                d_li = rc.d_local_idx.ptr;
                d_lc = rc.d_local_cnt.ptr;
            }
            const int32_t* d_qpos_arg = rc.d_pos.ptr;
            const int32_t* d_qmin_arg = nullptr;
            if (batched_mode) {
                rc.d_q_max_slot.alloc(r, size_t(M));
                rc.d_q_min_slot.alloc(r, size_t(M));
                check_hip_r(hipMemcpyAsync(rc.d_q_max_slot.ptr, h_kv_slot_ovr,  size_t(M) * 4u, hipMemcpyHostToDevice, rc.stream), "qmax");
                check_hip_r(hipMemcpyAsync(rc.d_q_min_slot.ptr, h_min_slot_ovr, size_t(M) * 4u, hipMemcpyHostToDevice, rc.stream), "qmin");
                d_qpos_arg = rc.d_q_max_slot.ptr;
                d_qmin_arg = rc.d_q_min_slot.ptr;
            }
            if (kv_layer >= kBf16KvStartLayer) {
                const uint16_t* kv_ptr_bf16 = rc.kv_cache_bf16.ptr + size_t(kv_layer - kBf16KvStartLayer) * size_t(rc.local_kv_cap) * size_t(kKvLatentDim);
                launch_mla_absorbed_partial_bf16(
                    rc.d_q_abs_full.ptr, kv_ptr_bf16, rc.local_to_global.ptr, d_qpos_arg,
                    d_li, d_lc, rc.d_mla_max.ptr, rc.d_mla_sum.ptr, rc.d_mla_acc.ptr,
                    M, kNumAttnHeads, l_kv, kIndexTopK, kAttnScale, rc.stream, d_qmin_arg);
            } else {
                const uint8_t* kv_ptr = rc.kv_cache_fp8.ptr + size_t(kv_layer) * size_t(rc.local_kv_cap) * size_t(kKvFp8EntryBytes);
                launch_mla_absorbed_partial_fp8(
                    rc.d_q_abs_full.ptr, kv_ptr, rc.local_to_global.ptr, d_qpos_arg,
                    d_li, d_lc, rc.d_mla_max.ptr, rc.d_mla_sum.ptr, rc.d_mla_acc.ptr,
                    M, kNumAttnHeads, l_kv, kIndexTopK, kAttnScale, rc.stream, d_qmin_arg);
            }
            check_hip_r(hipMemcpyAsync(p_max_ptr[r], rc.d_mla_max.ptr, mh_tot * 4u, hipMemcpyDeviceToHost, rc.stream), "pm");
            check_hip_r(hipMemcpyAsync(p_sum_ptr[r], rc.d_mla_sum.ptr, mh_tot * 4u, hipMemcpyDeviceToHost, rc.stream), "ps");
            check_hip_r(hipMemcpyAsync(p_acc_ptr[r], rc.d_mla_acc.ptr, acc_tot * 4u, hipMemcpyDeviceToHost, rc.stream), "pa");
        }
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            if (kv_rank_local_toks(r, cur_total_kv) <= 0) continue;
            check_hip_r(hipSetDevice(r), "mla_sync_set");
            check_hip_r(hipStreamSynchronize(ranks[r].stream), "mla_sync");
        }

        // Use host-cached learnable sinks (zero hipMemcpy!) & exact online-softmax merge (+ DFly2 draft block KV!)
        const auto& sinks_all = is_dfly ? h_dfly_sinks[size_t(dfly_l)] : h_sinks[size_t(layer_l)];
        static const bool dfly_bidir = (std::getenv("HY4_DFLY_BIDIR") != nullptr);

        std::vector<uint16_t> h_olat_vec;
        uint16_t* h_olat_ptr = use_pin ? h_olat_pin : nullptr;
        if (!use_pin) { h_olat_vec.resize(acc_tot); h_olat_ptr = h_olat_vec.data(); }
        #pragma omp parallel for schedule(static) num_threads(8) if(M >= 16)
        for (int mh = 0; mh < M * kNumAttnHeads; ++mh) {
            const int m = mh / kNumAttnHeads;
            const int h = mh % kNumAttnHeads;
            float m_glob = sinks_all[size_t(h)];
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                if (p_sum_ptr[r][size_t(mh)] > 0.0f) m_glob = std::max(m_glob, p_max_ptr[r][size_t(mh)]);
            }
            float d_scores[kDflyBlockSize]{};
            int n_dk = 0;
            if (is_dfly && !h_dfly_kv.empty()) {
                n_dk = std::min(kDflyBlockSize, dfly_bidir ? M : (m + 1));
                const uint16_t* qmh = h_qabs_all_ptr + size_t(mh) * size_t(kKvLatentDim);
                for (int j = 0; j < n_dk; ++j) {
                    const uint16_t* kj = h_dfly_kv.data() + size_t(j) * size_t(kKvLatentDim);
                    float dot = 0.0f;
                    for (int d = 0; d < kKvLatentDim; ++d) dot += bf16_to_f32(qmh[d]) * bf16_to_f32(kj[d]);
                    d_scores[j] = dot * kAttnScale;
                    if (d_scores[j] > m_glob) m_glob = d_scores[j];
                }
            }
            float l_glob = std::exp(sinks_all[size_t(h)] - m_glob);
            float w_r[kNumMi50Ranks]{};
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                if (p_sum_ptr[r][size_t(mh)] > 0.0f) {
                    w_r[r] = std::exp(p_max_ptr[r][size_t(mh)] - m_glob);
                    l_glob += p_sum_ptr[r][size_t(mh)] * w_r[r];
                }
            }
            float w_d[kDflyBlockSize]{};
            for (int j = 0; j < n_dk; ++j) {
                w_d[j] = std::exp(d_scores[j] - m_glob);
                l_glob += w_d[j];
            }
            const float inv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
            uint16_t* dst = h_olat_ptr + size_t(mh) * size_t(kKvLoraRank);
            const size_t base_d = size_t(mh) * size_t(kKvLoraRank);
#if defined(__AVX2__) && defined(__F16C__)
            const __m256 v_inv = _mm256_set1_ps(inv_l);
            const __m256 vw0 = _mm256_set1_ps(w_r[0]);
            const __m256 vw1 = _mm256_set1_ps(w_r[1]);
            const __m256 vw2 = _mm256_set1_ps(w_r[2]);
            const __m256 vw3 = _mm256_set1_ps(w_r[3]);
            const __m256 vw4 = _mm256_set1_ps(w_r[4]);
            const float* pa0 = (w_r[0] > 0.0f) ? (p_acc_ptr[0] + base_d) : nullptr;
            const float* pa1 = (w_r[1] > 0.0f) ? (p_acc_ptr[1] + base_d) : nullptr;
            const float* pa2 = (w_r[2] > 0.0f) ? (p_acc_ptr[2] + base_d) : nullptr;
            const float* pa3 = (w_r[3] > 0.0f) ? (p_acc_ptr[3] + base_d) : nullptr;
            const float* pa4 = (w_r[4] > 0.0f) ? (p_acc_ptr[4] + base_d) : nullptr;
            for (int d = 0; d < kKvLoraRank; d += 8) {
                __m256 acc = _mm256_setzero_ps();
                if (pa0) acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa0 + d), vw0, acc);
                if (pa1) acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa1 + d), vw1, acc);
                if (pa2) acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa2 + d), vw2, acc);
                if (pa3) acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa3 + d), vw3, acc);
                if (pa4) acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa4 + d), vw4, acc);
                for (int j = 0; j < n_dk; ++j) {
                    const __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(h_dfly_kv.data() + size_t(j) * size_t(kKvLatentDim) + size_t(d)));
                    const __m256  f8 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b8), 16));
                    acc = _mm256_fmadd_ps(f8, _mm256_set1_ps(w_d[j]), acc);
                }
                const __m128i h8 = _mm256_cvtps_ph(_mm256_mul_ps(acc, v_inv), _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + d), h8);
            }
#else
            for (int d = 0; d < kKvLoraRank; ++d) {
                float acc = 0.0f;
                for (int r = 0; r < kNumMi50Ranks; ++r) {
                    if (w_r[r] > 0.0f) acc += p_acc_ptr[r][base_d + size_t(d)] * w_r[r];
                }
                for (int j = 0; j < n_dk; ++j) {
                    acc += w_d[j] * bf16_to_f32(h_dfly_kv[size_t(j) * size_t(kKvLatentDim) + size_t(d)]);
                }
                dst[d] = f32_to_fp16_rne(acc * inv_l);
            }
#endif
        }

        // 5. All 5 ranks project h_cnt heads IN PARALLEL: O_lat * W_UV -> ON-GPU transpose+gate -> o_proj partial [M, 6144]!
        std::array<std::vector<uint16_t>, kNumMi50Ranks> h_ol_vec;
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            auto& rc = ranks[r];
            check_hip_r(hipSetDevice(r), "uv_set");
            const size_t ol_elems = size_t(rc.h_cnt) * size_t(M) * 512u;
            uint16_t* h_ol_r = use_pin ? h_ol_r_pin[r] : nullptr;
            if (!use_pin) { h_ol_vec[size_t(r)].resize(ol_elems); h_ol_r = h_ol_vec[size_t(r)].data(); }
            for (int h = 0; h < rc.h_cnt; ++h) {
                const int gh = rc.h_start + h;
                for (int m = 0; m < M; ++m) {
                    std::memcpy(h_ol_r + (size_t(h) * size_t(M) + size_t(m)) * 512u,
                                h_olat_ptr + (size_t(m) * size_t(kNumAttnHeads) + size_t(gh)) * 512u, 512u * 2u);
                }
            }
            rc.d_gate_out.alloc(r, size_t(M) * size_t(rc.h_cnt) * 256u);
            rc.d_attn_out.alloc(r, size_t(M) * size_t(rc.h_cnt) * 256u);
            rc.d_y_bf16.alloc(r, size_t(M) * size_t(kHiddenSize));

            const auto& w_uv = is_dfly ? rc.dfly[size_t(dfly_l)].w_uv : rc.layers[size_t(layer_l)].w_uv;
            const auto& lg   = is_dfly ? rc.dfly[size_t(dfly_l)].linear_gate : rc.layers[size_t(layer_l)].linear_gate;
            const auto& op   = is_dfly ? rc.dfly[size_t(dfly_l)].o_proj : rc.layers[size_t(layer_l)].o_proj;

            // Compute linear_gate BEFORE uv_bmm because fp8_linear(M>8) uses rc.d_a_f16 & rc.d_c_f32 as GEMM scratch!
            fp8_linear(rc, lg, rc.d_x_bf16.ptr, rc.d_gate_out.ptr, M);

            rc.d_a_f16.alloc(r, ol_elems);
            rc.d_c_f32.alloc(r, std::max(size_t(rc.h_cnt) * size_t(M) * 256u, size_t(M) * size_t(kHiddenSize)));
            check_hip_r(hipMemcpyAsync(rc.d_a_f16.ptr, h_ol_r, ol_elems * 2u, hipMemcpyHostToDevice, rc.stream), "ol_h2d");

            const float one = 1.0f, zero = 0.0f;
            check_rb_r(rocblas_gemm_strided_batched_ex(
                rc.rb, rocblas_operation_transpose, rocblas_operation_none,
                256, M, 512, &one,
                w_uv.ptr, rocblas_datatype_f16_r, 512, rocblas_stride(256 * 512),
                rc.d_a_f16.ptr, rocblas_datatype_f16_r, 512, rocblas_stride(size_t(M) * 512u),
                &zero,
                rc.d_c_f32.ptr, rocblas_datatype_f32_r, 256, rocblas_stride(size_t(M) * 256u),
                rc.d_c_f32.ptr, rocblas_datatype_f32_r, 256, rocblas_stride(size_t(M) * 256u),
                rc.h_cnt, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "uv_bmm");
            launch_ao_transpose_gate_bf16(rc.d_c_f32.ptr, rc.d_gate_out.ptr, rc.d_attn_out.ptr, M, rc.h_cnt, rc.stream);
            fp8_linear(rc, op, rc.d_attn_out.ptr, rc.d_y_bf16.ptr, M);
            launch_bf16_to_f32(rc.d_y_bf16.ptr, rc.d_c_f32.ptr, M * kHiddenSize, rc.stream);
            check_hip_r(hipMemcpyAsync(h_rank_part[r], rc.d_c_f32.ptr, size_t(M) * size_t(kHiddenSize) * 4u, hipMemcpyDeviceToHost, rc.stream), "op_d2h");
        }
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            check_hip_r(hipSetDevice(r), "op_sync_set");
            check_hip_r(hipStreamSynchronize(ranks[r].stream), "op_sync");
        }
        reduce_5ranks_f32_to_r0_bf16(M, d_r0_attn_out_bf16);
    }

    // ------------------------------------------------------------------------
    // Execute Dense TP5 MLP (L0 or DFly2 5 layers: I = 18432, 2:2:2:1:1)
    // ------------------------------------------------------------------------
    void run_dense_mlp_tp5(bool is_dfly, int dfly_l, int M, const uint16_t* d_r0_x_bf16, uint16_t* d_r0_out_bf16) {
        std::vector<uint16_t> h_x(size_t(M) * size_t(kHiddenSize));
        check_hip_r(hipSetDevice(0), "dmlp_set0");
        check_hip_r(hipMemcpyAsync(h_x.data(), d_r0_x_bf16, h_x.size() * 2u, hipMemcpyDeviceToHost, ranks[0].stream), "dx_d2h");
        check_hip_r(hipStreamSynchronize(ranks[0].stream), "dx_sync");

        for (int r = 0; r < kNumMi50Ranks; ++r) {
            auto& rc = ranks[r];
            check_hip_r(hipSetDevice(r), "dmlp_setr");
            const auto& mg = is_dfly ? rc.dfly[size_t(dfly_l)].mlp_gate : rc.layers[0].mlp_gate;
            const auto& mu = is_dfly ? rc.dfly[size_t(dfly_l)].mlp_up   : rc.layers[0].mlp_up;
            const auto& md = is_dfly ? rc.dfly[size_t(dfly_l)].mlp_down : rc.layers[0].mlp_down;
            rc.d_x_bf16.alloc(r, h_x.size());
            rc.d_mlp_g.alloc(r, size_t(M) * size_t(mg.N));
            rc.d_mlp_u.alloc(r, size_t(M) * size_t(mu.N));
            rc.d_mlp_z.alloc(r, size_t(M) * size_t(mg.N));
            rc.d_y_bf16.alloc(r, size_t(M) * size_t(kHiddenSize));
            rc.d_c_f32.alloc(r, size_t(M) * size_t(kHiddenSize));
            check_hip_r(hipMemcpyAsync(rc.d_x_bf16.ptr, h_x.data(), h_x.size() * 2u, hipMemcpyHostToDevice, rc.stream), "dx_h2d");
            fp8_linear(rc, mg, rc.d_x_bf16.ptr, rc.d_mlp_g.ptr, M);
            fp8_linear(rc, mu, rc.d_x_bf16.ptr, rc.d_mlp_u.ptr, M);
            launch_swiglu_unclamped_bf16(rc.d_mlp_g.ptr, rc.d_mlp_u.ptr, rc.d_mlp_z.ptr, M * mg.N, rc.stream);
            fp8_linear(rc, md, rc.d_mlp_z.ptr, rc.d_y_bf16.ptr, M);
            launch_bf16_to_f32(rc.d_y_bf16.ptr, rc.d_c_f32.ptr, M * kHiddenSize, rc.stream);
            check_hip_r(hipMemcpyAsync(h_rank_part[r], rc.d_c_f32.ptr, size_t(M) * size_t(kHiddenSize) * 4u, hipMemcpyDeviceToHost, rc.stream), "dy_d2h");
        }
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            check_hip_r(hipSetDevice(r), "dmlp_sync_set");
            check_hip_r(hipStreamSynchronize(ranks[r].stream), "dmlp_sync");
        }
        reduce_5ranks_f32_to_r0_bf16(M, d_r0_out_bf16);
    }

    // ------------------------------------------------------------------------
    // Execute Simultaneous 3-Tier MoE Layer (l = 1..77):
    //   Tier 1A/B: 5x MI50 (TP5 Shared Expert + EP HLUT3/HLUT2 Routed Experts)
    //   Tier 0/1B: 2x Blackhole P150A (FABRIC_1D TP2 HLUT3/HLUT2 + 330 MiB L1 Pin)
    //   Tier 2:    28-core EPYC DDR4 AVX2+FMA+F16C WHT128-HLUT4 Cold Tail
    // ------------------------------------------------------------------------
    void run_moe_3tier(int l, int M, const uint16_t* d_r0_x_bf16, uint16_t* d_r0_out_bf16) {
        auto& rc0 = ranks[0];
        auto& r0e = r0_layers[size_t(l)];
        const auto t_r0 = std::chrono::steady_clock::now();
        check_hip_r(hipSetDevice(0), "moe_set0");

        const size_t x_elems = size_t(M) * size_t(kHiddenSize);
        int mi50_hlut_cnt = 0;
        for (int r = 0; r < kNumMi50Ranks; ++r) mi50_hlut_cnt += plan.mi50_hlut4[r] + plan.mi50_hlut3[r] + plan.mi50_hlut2[r];
        const bool need_mi50_wht = (mi50_hlut_cnt > 0);
        const bool fast_q5_path  = (!need_mi50_wht && ddr4_engine.is_q5_mode() && M <= kPinFastM && !diag_verify_p150);

        // 1. Router logits + Top-8 selection on R0 (skip dead WHT128 GPU kernels on Profile A5 fast path!)
        r0_router_logits.alloc(0, size_t(M) * size_t(kNumRoutedExperts));
        r0_topk_ids.alloc(0, size_t(M) * size_t(kTopK));
        r0_topk_w.alloc(0, size_t(M) * size_t(kTopK));
        launch_bf16_gemv_bf16(d_r0_x_bf16, r0e.router_w_bf16.ptr, r0_router_logits.ptr, M, kNumRoutedExperts, kHiddenSize, rc0.stream);
        launch_moe_router_top8_256(r0_router_logits.ptr, r0e.router_bias_f32.ptr, r0_topk_ids.ptr, r0_topk_w.ptr, M, kRoutedScalingFactor, rc0.stream);

        std::vector<uint16_t> h_xrot_f16;
        if (!fast_q5_path) {
            rc0.d_x_rot_f32.alloc(0, x_elems);
            launch_wht128_bf16_to_f32(d_r0_x_bf16, rc0.layers[size_t(l)].signs_h.ptr, rc0.d_x_rot_f32.ptr, M, kHiddenSize, rc0.stream);
            if (need_mi50_wht) {
                rc0.d_x_rot_f16.alloc(0, x_elems);
                h_xrot_f16.resize(x_elems);
                launch_wht128_bf16_to_f16(d_r0_x_bf16, rc0.layers[size_t(l)].signs_h.ptr, rc0.d_x_rot_f16.ptr, M, kHiddenSize, rc0.stream);
            }
        }

        check_hip_r(hipMemcpyAsync(h_topk_ids_pin, r0_topk_ids.ptr, size_t(M) * size_t(kTopK) * 4u, hipMemcpyDeviceToHost, rc0.stream), "tid");
        check_hip_r(hipMemcpyAsync(h_topk_wts_pin, r0_topk_w.ptr,   size_t(M) * size_t(kTopK) * 4u, hipMemcpyDeviceToHost, rc0.stream), "twt");
        check_hip_r(hipMemcpyAsync(h_x_host_bf16,  d_r0_x_bf16,     x_elems * 2u,                  hipMemcpyDeviceToHost, rc0.stream), "hx");
        if (!fast_q5_path) {
            check_hip_r(hipMemcpyAsync(h_x_rot_f32, rc0.d_x_rot_f32.ptr, x_elems * 4u, hipMemcpyDeviceToHost, rc0.stream), "xrf32");
            if (need_mi50_wht) {
                check_hip_r(hipMemcpyAsync(h_xrot_f16.data(), rc0.d_x_rot_f16.ptr, x_elems * 2u, hipMemcpyDeviceToHost, rc0.stream), "xrf16");
            }
        }
        check_hip_r(hipStreamSynchronize(rc0.stream), "moe_route_sync");

        // 2. Partition all M * 8 routes across 5x MI50 (FP8, HLUT4, HLUT3 & HLUT2), 2x P150A, and EPYC DDR4 AVX2!
        std::array<std::vector<Hy4ExpertGroup>, kNumMi50Ranks> mi50_fp8_grps, mi50_h4_grps, mi50_h3_grps, mi50_h2_grps;
        std::vector<Ddr4ColdRoute> ddr4_items;
        ddr4_items.reserve(size_t(M) * size_t(kTopK));

        bool any_p150 = false;
        uint64_t n_mi50 = 0, n_p150 = 0, n_ddr4 = 0;

        if (p150_connected && p150_shm.ptr) {
            for (int m = 0; m < M; ++m) {
                for (int k = 0; k < kTopK; ++k) {
                    p150_shm.ptr->row_slots[m][k] = -1;
                    p150_shm.ptr->row_weights[m][k] = 0.0f;
                }
            }
        }

        auto add_mi50_grp = [](std::vector<Hy4ExpertGroup>& vec, int slot, int row, float w) {
            for (auto& g : vec) {
                if (g.expert == slot && g.count < kMaxGroupTokens) {
                    g.rows[g.count] = row;
                    g.weights[g.count] = w;
                    g.count++;
                    return;
                }
            }
            Hy4ExpertGroup g{};
            g.expert = slot; g.count = 1; g.rows[0] = row; g.weights[0] = w;
            vec.push_back(g);
        };

        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < kTopK; ++k) {
                const int eid = h_topk_ids_pin[size_t(m) * size_t(kTopK) + size_t(k)];
                const float w = h_topk_wts_pin[size_t(m) * size_t(kTopK) + size_t(k)];
                if (eid < 0 || eid >= kNumRoutedExperts || w == 0.0f) continue;
                const auto& re = route_tbl[size_t(l)][size_t(eid)];
                if (re.dev == ExpertDevice::P150_TP2 && p150_connected && p150_shm.ptr) {
                    p150_shm.ptr->row_slots[m][k] = re.slot;
                    p150_shm.ptr->row_weights[m][k] = w;
                    any_p150 = true;
                    ++n_p150;
                } else if (re.dev == ExpertDevice::CPU_DDR4) {
                    ddr4_items.push_back({m, re.slot, w});
                    ++n_ddr4;
                } else if (re.dev != ExpertDevice::P150_TP2) {
                    const int r = int(re.dev);
                    if (re.fmt == ExpertFormat::FP8)        add_mi50_grp(mi50_fp8_grps[size_t(r)], re.slot, m, w);
                    else if (re.fmt == ExpertFormat::HLUT4) add_mi50_grp(mi50_h4_grps[size_t(r)],  re.slot, m, w);
                    else if (re.fmt == ExpertFormat::HLUT3) add_mi50_grp(mi50_h3_grps[size_t(r)],  re.slot, m, w);
                    else                                    add_mi50_grp(mi50_h2_grps[size_t(r)],  re.slot, m, w);
                    ++n_mi50;
                }
            }
        }
        stat_total_routes.fetch_add(n_mi50 + n_p150 + n_ddr4, std::memory_order_relaxed);
        stat_mi50_hits.fetch_add(n_mi50, std::memory_order_relaxed);
        stat_p150_hits.fetch_add(n_p150, std::memory_order_relaxed);
        stat_ddr4_hits.fetch_add(n_ddr4, std::memory_order_relaxed);
        const auto t_moe0 = std::chrono::steady_clock::now();
        const double router_ms = std::chrono::duration<double, std::milli>(t_moe0 - t_r0).count();

        // 3A. Kick off EPYC DDR4 AVX2 Q5_LM128 FIRST (longest critical path -> zero serial preprocessing!)
        const auto t_d0 = std::chrono::steady_clock::now();
        const bool any_ddr4 = (ddr4_ready && !ddr4_items.empty());
        if (any_ddr4) {
            if (ddr4_engine.is_q5_mode()) {
                ddr4_engine.submit_async_bf16(l, h_x_host_bf16, M, ddr4_items.data(), int(ddr4_items.size()), h_ddr4_out);
            } else {
                ddr4_engine.submit_async(l, h_x_rot_f32, M, ddr4_items.data(), int(ddr4_items.size()), h_ddr4_out);
            }
        }

        // 3B. Simultaneously kick off 2x Blackhole P150A over lock-free POSIX SHM (host WHT128 overlaps DDR4 Phase 0!)
        const auto t_p0 = std::chrono::steady_clock::now();
        uint32_t expected_p150_seq = 0u;
        if (any_p150) {
            auto* ch = p150_shm.ptr;
            if (fast_q5_path) {
                // Zero-OMP direct BF16 -> FP32 + WHT128 straight into P150A SHM (overlaps DDR4 Phase 0 with zero core theft!)
                const auto& sh = h_signs_h[size_t(l)];
                constexpr int nblk = kHiddenSize / 128;
                for (int m = 0; m < M; ++m) {
                    const uint16_t* src_row = h_x_host_bf16 + size_t(m) * size_t(kHiddenSize);
                    float* dst_row = ch->x_rot_f32 + size_t(m) * size_t(kHiddenSize);
#if defined(__AVX2__)
                    for (int d = 0; d < kHiddenSize; d += 8) {
                        const __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_row + d));
                        _mm256_storeu_ps(dst_row + d, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b8), 16)));
                    }
#else
                    for (int d = 0; d < kHiddenSize; ++d) dst_row[d] = bf16_to_f32(src_row[d]);
#endif
                    for (int b = 0; b < nblk; ++b) wht128_block(dst_row + b * 128, &sh[size_t(b) * 2u]);
                }
            } else {
                std::memcpy(ch->x_rot_f32, h_x_rot_f32, x_elems * 4u);
            }
            ch->op        = HY4_770B_P150_OP_MOE_EXEC;
            ch->layer_idx = uint32_t(l);
            ch->rows      = uint32_t(M);
            ch->top_k     = uint32_t(kTopK);
            expected_p150_seq = ++p150_req_seq;
            ch->req_seq.store(expected_p150_seq, std::memory_order_release);
        }

        // 3C. Simultaneously execute TP5 Shared Expert + EP Routed Experts across all 5 MI50s!
        const auto t_m0 = std::chrono::steady_clock::now();
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            auto& rc = ranks[r];
            auto& lr = rc.layers[size_t(l)];
            check_hip_r(hipSetDevice(r), "moe_setr");
            rc.d_x_bf16.alloc(r, x_elems);
            rc.d_mlp_g.alloc(r, size_t(M) * size_t(lr.mlp_gate.N));
            rc.d_mlp_u.alloc(r, size_t(M) * size_t(lr.mlp_up.N));
            rc.d_mlp_z.alloc(r, size_t(M) * size_t(lr.mlp_gate.N));
            rc.d_y_bf16.alloc(r, x_elems);
            rc.d_moe_acc_f32.alloc(r, x_elems);

            if (r == 0) {
                check_hip_r(hipMemcpyAsync(rc.d_x_bf16.ptr, d_r0_x_bf16, x_elems * 2u, hipMemcpyDeviceToDevice, rc.stream), "mx0");
            } else {
                check_hip_r(hipMemcpyAsync(rc.d_x_bf16.ptr, h_x_host_bf16, x_elems * 2u, hipMemcpyHostToDevice, rc.stream), "mx");
            }
            // Shared expert TP5 slice
            fp8_linear(rc, lr.mlp_gate, rc.d_x_bf16.ptr, rc.d_mlp_g.ptr, M);
            fp8_linear(rc, lr.mlp_up,   rc.d_x_bf16.ptr, rc.d_mlp_u.ptr, M);
            launch_swiglu_unclamped_bf16(rc.d_mlp_g.ptr, rc.d_mlp_u.ptr, rc.d_mlp_z.ptr, M * lr.mlp_gate.N, rc.stream);
            fp8_linear(rc, lr.mlp_down, rc.d_mlp_z.ptr, rc.d_y_bf16.ptr, M);
            launch_bf16_to_f32(rc.d_y_bf16.ptr, rc.d_moe_acc_f32.ptr, M * kHiddenSize, rc.stream);

            // Local EP routed experts (Lossless FP8, HLUT4, HLUT3 and/or HLUT2) accumulated directly into d_moe_acc_f32!
            const int ng_fp8 = int(mi50_fp8_grps[size_t(r)].size());
            const int ng4    = int(mi50_h4_grps[size_t(r)].size());
            const int ng3    = int(mi50_h3_grps[size_t(r)].size());
            const int ng2    = int(mi50_h2_grps[size_t(r)].size());
            if (ng_fp8 > 0) {
                rc.d_groups_fp8.alloc(r, size_t(ng_fp8));
                rc.d_moe_z.alloc(r, size_t(ng_fp8) * size_t(kMaxGroupTokens) * size_t(kMoeInterSize));
                check_hip_r(hipMemcpyAsync(rc.d_groups_fp8.ptr, mi50_fp8_grps[size_t(r)].data(),
                                           size_t(ng_fp8) * sizeof(Hy4ExpertGroup), hipMemcpyHostToDevice, rc.stream), "gfp8");
                launch_fp8_grouped_gateup_swiglu(rc.d_x_bf16.ptr, rc.d_groups_fp8.ptr,
                                                 lr.fp8_gate_w.ptr, lr.fp8_gate_s.ptr,
                                                 lr.fp8_up_w.ptr,   lr.fp8_up_s.ptr,
                                                 rc.d_moe_z.ptr, ng_fp8, kMoeInterSize, kHiddenSize, rc.stream);
                launch_fp8_grouped_down_atomic(rc.d_moe_z.ptr, rc.d_groups_fp8.ptr,
                                               lr.fp8_down_w.ptr, lr.fp8_down_s.ptr,
                                               rc.d_moe_acc_f32.ptr, ng_fp8, kHiddenSize, kMoeInterSize, rc.stream);
            }
            if (ng4 > 0 || ng3 > 0 || ng2 > 0) {
                rc.d_x_rot_f16.alloc(r, h_xrot_f16.size());
                if (r != 0) check_hip_r(hipMemcpyAsync(rc.d_x_rot_f16.ptr, h_xrot_f16.data(), h_xrot_f16.size() * 2u, hipMemcpyHostToDevice, rc.stream), "xrf16_r");
            }
            if (ng4 > 0) {
                rc.d_groups_h4.alloc(r, size_t(ng4));
                rc.d_moe_z.alloc(r, size_t(ng4) * size_t(kMaxGroupTokens) * size_t(kMoeInterSize));
                check_hip_r(hipMemcpyAsync(rc.d_groups_h4.ptr, mi50_h4_grps[size_t(r)].data(), size_t(ng4) * sizeof(Hy4ExpertGroup), hipMemcpyHostToDevice, rc.stream), "gh4");
                launch_hlut4_grouped_gateup_swiglu_f16(rc.d_x_rot_f16.ptr, rc.d_groups_h4.ptr,
                                                       lr.h4_gate.ptr, lr.h4_up.ptr, rc.d_lev4.ptr,
                                                       rc.d_moe_z.ptr, ng4, kMoeInterSize, kHiddenSize, rc.stream);
                launch_wht128_f32_inplace(rc.d_moe_z.ptr, lr.signs_i.ptr, ng4 * kMaxGroupTokens, kMoeInterSize, rc.stream);
                launch_hlut4_grouped_down_atomic(rc.d_moe_z.ptr, rc.d_groups_h4.ptr, lr.h4_down.ptr, rc.d_lev4.ptr,
                                                 rc.d_moe_acc_f32.ptr, ng4, kHiddenSize, kMoeInterSize, rc.stream);
            }
            if (ng3 > 0) {
                rc.d_groups_h3.alloc(r, size_t(ng3));
                rc.d_moe_z.alloc(r, size_t(ng3) * size_t(kMaxGroupTokens) * size_t(kMoeInterSize));
                check_hip_r(hipMemcpyAsync(rc.d_groups_h3.ptr, mi50_h3_grps[size_t(r)].data(), size_t(ng3) * sizeof(Hy4ExpertGroup), hipMemcpyHostToDevice, rc.stream), "gh3");
                launch_hlut3_grouped_gateup_swiglu_f16(rc.d_x_rot_f16.ptr, rc.d_groups_h3.ptr,
                                                       lr.h3_gate.ptr, lr.h3_up.ptr, rc.d_lev3.ptr,
                                                       rc.d_moe_z.ptr, ng3, kMoeInterSize, kHiddenSize, rc.stream);
                launch_wht128_f32_inplace(rc.d_moe_z.ptr, lr.signs_i.ptr, ng3 * kMaxGroupTokens, kMoeInterSize, rc.stream);
                launch_hlut3_grouped_down_atomic(rc.d_moe_z.ptr, rc.d_groups_h3.ptr, lr.h3_down.ptr, rc.d_lev3.ptr,
                                                 rc.d_moe_acc_f32.ptr, ng3, kHiddenSize, kMoeInterSize, rc.stream);
            }
            if (ng2 > 0) {
                rc.d_groups_h2.alloc(r, size_t(ng2));
                rc.d_moe_z.alloc(r, size_t(ng2) * size_t(kMaxGroupTokens) * size_t(kMoeInterSize));
                check_hip_r(hipMemcpyAsync(rc.d_groups_h2.ptr, mi50_h2_grps[size_t(r)].data(), size_t(ng2) * sizeof(Hy4ExpertGroup), hipMemcpyHostToDevice, rc.stream), "gh2");
                launch_hlut2_grouped_gateup_swiglu_f16(rc.d_x_rot_f16.ptr, rc.d_groups_h2.ptr,
                                                       lr.h2_gate.ptr, lr.h2_up.ptr, rc.d_lev2.ptr,
                                                       rc.d_moe_z.ptr, ng2, kMoeInterSize, kHiddenSize, rc.stream);
                launch_wht128_f32_inplace(rc.d_moe_z.ptr, lr.signs_i.ptr, ng2 * kMaxGroupTokens, kMoeInterSize, rc.stream);
                launch_hlut2_grouped_down_atomic(rc.d_moe_z.ptr, rc.d_groups_h2.ptr, lr.h2_down.ptr, rc.d_lev2.ptr,
                                                 rc.d_moe_acc_f32.ptr, ng2, kHiddenSize, kMoeInterSize, rc.stream);
            }
            check_hip_r(hipMemcpyAsync(h_rank_part[r], rc.d_moe_acc_f32.ptr, size_t(M) * size_t(kHiddenSize) * 4u, hipMemcpyDeviceToHost, rc.stream), "ma_d2h");
        }
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            check_hip_r(hipSetDevice(r), "moe_sync_set");
            check_hip_r(hipStreamSynchronize(ranks[r].stream), "moe_sync");
        }
        const double mi50_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m0).count();

        // 4. Join P150A & EPYC DDR4 completions (record independent hardware wall times!)
        double p150_ms = 0.0, ddr4_ms = 0.0;
        if (any_p150) {
            auto* ch = p150_shm.ptr;
            while (ch->resp_seq.load(std::memory_order_acquire) != expected_p150_seq) {
                p150_cpu_relax();
            }
            if (ch->status != 0) throw std::runtime_error("P150A worker returned error status on layer " + std::to_string(l));
            std::memcpy(h_p150_out, ch->y_out_f32, size_t(M) * size_t(kHiddenSize) * 4u);
            p150_ms = double(ch->last_device_ns) * 1e-6;
            if (p150_ms <= 0.0) p150_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_p0).count();
        }
        if (any_ddr4) {
            ddr4_ms = ddr4_engine.wait_done() * 1e-3;
            (void)t_d0;
        }

        if (diag_sink) {
            cur_layer_diag.mi50_routes = int(n_mi50);
            cur_layer_diag.p150_routes = int(n_p150);
            cur_layer_diag.ddr4_routes = int(n_ddr4);
            for (int k = 0; k < kTopK; ++k) {
                cur_layer_diag.last_tok_topk[size_t(k)] = h_topk_ids_pin[size_t(M - 1) * size_t(kTopK) + size_t(k)];
            }
            float xrm = 0.0f, p150m = 0.0f, ddr4m = 0.0f;
            for (size_t i = 0; i < size_t(M) * size_t(kHiddenSize); ++i) {
                xrm = std::max(xrm, std::fabs(h_x_rot_f32[i]));
                if (any_p150) p150m = std::max(p150m, std::fabs(h_p150_out[i]));
                if (any_ddr4) ddr4m = std::max(ddr4m, std::fabs(h_ddr4_out[i]));
            }
            cur_layer_diag.x_rot_max_abs = xrm;
            cur_layer_diag.p150_max_abs  = p150m;
            cur_layer_diag.ddr4_max_abs  = ddr4m;

            // Optional exact CPU HLUT3 reference cross-check for P150A!
            if (diag_verify_p150 && any_p150 && p150_shm.ptr) {
                std::array<std::vector<std::pair<int, float>>, 256> slot_rw;
                int max_rpe = 0;
                for (int m = 0; m < M; ++m) {
                    for (int k = 0; k < kTopK; ++k) {
                        const int sl = p150_shm.ptr->row_slots[m][k];
                        const float wt = p150_shm.ptr->row_weights[m][k];
                        if (sl >= 0 && sl < plan.p150_hlut3 && wt != 0.0f) {
                            slot_rw[size_t(sl)].push_back({m, wt});
                            max_rpe = std::max(max_rpe, int(slot_rw[size_t(sl)].size()));
                        }
                    }
                }
                cur_layer_diag.p150_max_rows_per_exp = max_rpe;
                std::vector<float> ref_y(size_t(M) * size_t(kHiddenSize), 0.0f);
                const std::string ppath = cfg.weight_root + "/hlut3/hlut3_layer_" + ltag(l) + ".bin";
                std::ifstream ifs(ppath, std::ios::binary);
                const auto& sol3 = lloyd_max_hlut3();
                const float lev4[4] = {float(sol3.levels[0]), float(sol3.levels[1]), float(sol3.levels[2]), float(sol3.levels[3])};
                const auto signs_i = make_signs_i(l);
                constexpr size_t kProjT = size_t(kMoeInterSize / 32) * size_t(kHiddenSize / 32);
                std::vector<Hlut3Tile32x32> raw(3u * kProjT);
                for (int sl = 0; sl < plan.p150_hlut3; ++sl) {
                    if (slot_rw[size_t(sl)].empty()) continue;
                    ifs.seekg(std::streamoff(size_t(p150_h3_rank_begin + sl) * size_t(kHlut3ExpertBytes)), std::ios::beg);
                    ifs.read(reinterpret_cast<char*>(raw.data()), std::streamsize(kHlut3ExpertBytes));
                    cpu_eval_hlut3_expert_rows(raw.data(), raw.data() + kProjT, raw.data() + 2u * kProjT,
                                               signs_i, lev4, h_x_rot_f32, slot_rw[size_t(sl)], ref_y.data());
                }
                double dot = 0.0, n1 = 0.0, n2 = 0.0, merr = 0.0;
                for (size_t i = 0; i < ref_y.size(); ++i) {
                    const double a = double(ref_y[i]), b = double(h_p150_out[i]);
                    dot += a * b; n1 += a * a; n2 += b * b;
                    merr = std::max(merr, std::fabs(a - b));
                }
                cur_layer_diag.p150_cpu_cos = (n1 > 1e-20 && n2 > 1e-20) ? (dot / std::sqrt(n1 * n2)) : 1.0;
                cur_layer_diag.p150_cpu_max_err = merr;
            }
        }

        const double moe_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_moe0).count();

        // 5. Final 3-tier coherent reduction -> R0 BF16 output (pinned zero-alloc + zero OMP spin theft for M<16!)
        const auto t_red0 = std::chrono::steady_clock::now();
        const size_t n = size_t(M) * size_t(kHiddenSize);
        uint16_t* dst_bf16 = h_out_bf16;
        const float* p0 = h_rank_part[0];
        const float* p1 = h_rank_part[1];
        const float* p2 = h_rank_part[2];
        const float* p3 = h_rank_part[3];
        const float* p4 = h_rank_part[4];
        #pragma omp parallel for schedule(static) num_threads(8) if(M >= 16)
        for (size_t i = 0; i < n; ++i) {
            float s = p0[i] + p1[i] + p2[i] + p3[i] + p4[i];
            if (any_p150) s += h_p150_out[i];
            if (any_ddr4) s += h_ddr4_out[i];
            dst_bf16[i] = f32_to_bf16_rne(s);
        }
        check_hip_r(hipSetDevice(0), "moe_out_set0");
        check_hip_r(hipMemcpyAsync(d_r0_out_bf16, dst_bf16, n * 2u, hipMemcpyHostToDevice, ranks[0].stream), "moe_out_h2d");
        const double red_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_red0).count();

        {
            std::lock_guard<std::mutex> lk(stats_mu);
            stat_mi50_ms += mi50_ms;
            stat_p150_ms += p150_ms;
            stat_ddr4_ms += ddr4_ms;
            prof.moe_layers++;
            prof.router_ms       += router_ms;
            prof.moe_wall_ms     += moe_wall_ms;
            prof.mi50_moe_ms     += mi50_ms;
            prof.p150_moe_ms     += p150_ms;
            prof.ddr4_moe_ms     += ddr4_ms;
            prof.moe_reduce_ms   += red_ms;
            if (any_ddr4) prof.ddr4_unique_exp += uint64_t(ddr4_engine.last_unique_groups());
        }
    }

    // 5-GPU parallel LM Head projection: d_r0_final_h [M, 6144] BF16 -> out_logits [M, 120832] FP32
    void compute_lm_head_tp5(const uint16_t* d_r0_h_bf16, int M, std::vector<float>& out_logits) {
        out_logits.resize(size_t(M) * size_t(kVocabSize));
        std::vector<uint16_t> h_fin(size_t(M) * size_t(kHiddenSize));
        check_hip_r(hipSetDevice(0), "lm_set0");
        check_hip_r(hipMemcpyAsync(h_fin.data(), d_r0_h_bf16, h_fin.size() * 2u, hipMemcpyDeviceToHost, ranks[0].stream), "lm_h");
        check_hip_r(hipStreamSynchronize(ranks[0].stream), "lm_hsync");

        std::array<std::vector<float>, kNumMi50Ranks> r_log;
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            auto& rc = ranks[r];
            check_hip_r(hipSetDevice(r), "lm_setr");
            rc.d_x_bf16.alloc(r, h_fin.size());
            rc.d_c_f32.alloc(r, size_t(M) * size_t(rc.vocab_cnt));
            check_hip_r(hipMemcpyAsync(rc.d_x_bf16.ptr, h_fin.data(), h_fin.size() * 2u, hipMemcpyHostToDevice, rc.stream), "lm_x");
            f16_linear_bf16_in(rc, rc.lm_head_f16.ptr, rc.d_x_bf16.ptr, rc.d_c_f32.ptr, M, rc.vocab_cnt, kHiddenSize);
            r_log[size_t(r)].resize(size_t(M) * size_t(rc.vocab_cnt));
            check_hip_r(hipMemcpyAsync(r_log[size_t(r)].data(), rc.d_c_f32.ptr, r_log[size_t(r)].size() * 4u, hipMemcpyDeviceToHost, rc.stream), "lm_d2h");
        }
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            auto& rc = ranks[r];
            check_hip_r(hipSetDevice(r), "lm_sset");
            check_hip_r(hipStreamSynchronize(rc.stream), "lm_sync");
            for (int m = 0; m < M; ++m) {
                std::memcpy(out_logits.data() + size_t(m) * size_t(kVocabSize) + size_t(rc.vocab_start),
                            r_log[size_t(r)].data() + size_t(m) * size_t(rc.vocab_cnt),
                            size_t(rc.vocab_cnt) * 4u);
            }
        }
    }
};

// ============================================================================
// Public Hy4770bRuntime API Implementation
// ============================================================================
Hy4770bRuntime::Hy4770bRuntime() : impl_(std::make_unique<Impl>()) {}
Hy4770bRuntime::~Hy4770bRuntime() { shutdown(); }

void Hy4770bRuntime::shutdown() {
    if (impl_) impl_->drain_and_destroy();
}

void Hy4770bRuntime::init(const Hy4770bRuntimeConfig& cfg) {
    cfg_ = cfg;
    impl_->cfg = cfg;
    const auto t0 = std::chrono::steady_clock::now();

    // 1. Load tokenizer & compute exact SB-M5-P2-192G placement plan
    tokenizer_.load(cfg.root_dir + "/meta/tokenizer.json");
    impl_->plan = make_tier_plan(cfg.profile, cfg.max_kv_len, cfg.use_p150);
    mem_report_.tier_plan = impl_->plan;

    // 2. Attach to 2x Blackhole P150A worker SHM if enabled
    if (cfg.use_p150) {
        if (impl_->p150_shm.attach_client(cfg.p150_shm_name)) {
            impl_->p150_connected = true;
            auto* ch = impl_->p150_shm.ptr;
            mem_report_.p150_active = true;
            mem_report_.p150_l1_hot_experts = int(ch->l1_hot_experts);
            mem_report_.p150_l1_mib_total   = double(ch->l1_pinned_bytes_total) / 1048576.0;
            mem_report_.p150_used_gib[0]    = double(ch->gddr6_bytes_per_card) / 1073741824.0;
            mem_report_.p150_used_gib[1]    = mem_report_.p150_used_gib[0];
            std::printf("[HY4-770B-RT] Attached to 2x Blackhole P150A SHM %s (HLUT4=%u/L, HLUT3=%u/L, HLUT2=%u/L, L1=%.1f MiB, GDDR6=%.2f GiB/card)\n",
                        cfg.p150_shm_name.c_str(), ch->n_hlut4_per_layer, ch->n_hlut3_per_layer, ch->n_hlut2_per_layer,
                        mem_report_.p150_l1_mib_total, mem_report_.p150_used_gib[0]);
        } else {
            std::printf("[HY4-770B-RT] WARN: P150A SHM %s not active — falling back to 5x MI50 + EPYC DDR4 tier plan!\n",
                        cfg.p150_shm_name.c_str());
            impl_->plan = make_tier_plan(cfg.profile, cfg.max_kv_len, false);
            mem_report_.tier_plan = impl_->plan;
        }
        std::fflush(stdout);
    }

    // 3. Open all 5 MI50 GPUs sorted descending by physical VRAM (R0..R2 = 32 GiB, R3..R4 = 16 GiB!),
    //    build MRoPE tables, upload Lloyd-Max codebooks, allocate 3:3:3:1:1 KV arena!
    int dev_cnt = 0;
    check_hip_r(hipGetDeviceCount(&dev_cnt), "hipGetDeviceCount");
    if (dev_cnt < kNumMi50Ranks) {
        throw std::runtime_error("Expected 5 MI50 GPUs, found " + std::to_string(dev_cnt));
    }
    {
        std::vector<std::pair<size_t, int>> devs;
        for (int p = 0; p < kNumMi50Ranks; ++p) {
            size_t free_b = 0, tot_b = 0;
            check_hip_r(raw_hip_set_device(p), "probe_dev");
            check_hip_r(hipMemGetInfo(&free_b, &tot_b), "probe_mem");
            devs.push_back({tot_b, p});
        }
        std::stable_sort(devs.begin(), devs.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int r = 0; r < kNumMi50Ranks; ++r) g_rank_to_hip[r] = devs[size_t(r)].second;
        std::printf("[HY4-770B-RT] Physical MI50 rank map (VRAM-sorted): R0=HIP%d(%.0fG) R1=HIP%d(%.0fG) R2=HIP%d(%.0fG) R3=HIP%d(%.0fG) R4=HIP%d(%.0fG)\n",
                    g_rank_to_hip[0], double(devs[0].first) / 1073741824.0,
                    g_rank_to_hip[1], double(devs[1].first) / 1073741824.0,
                    g_rank_to_hip[2], double(devs[2].first) / 1073741824.0,
                    g_rank_to_hip[3], double(devs[3].first) / 1073741824.0,
                    g_rank_to_hip[4], double(devs[4].first) / 1073741824.0);
        std::fflush(stdout);
    }

    std::vector<float> h_cos, h_sin;
    build_factored_rope_tables(std::max(cfg.max_kv_len + 4096, 65536), kRopeTheta, h_cos, h_sin);
    const auto& sol4 = lloyd_max_hlut4();
    const auto& sol3 = lloyd_max_hlut3();
    const auto& sol2 = lloyd_max_hlut2();
    float lev4[8];
    for (int i = 0; i < 8; ++i) lev4[i] = float(sol4.levels[size_t(i)]);
    float lev3[4] = {float(sol3.levels[0]), float(sol3.levels[1]), float(sol3.levels[2]), float(sol3.levels[3])};
    float lev2[2] = {float(sol2.levels[0]), float(sol2.levels[1])};

    double kv_total_b = 0.0;
    for (int r = 0; r < kNumMi50Ranks; ++r) {
        auto& rc = impl_->ranks[r];
        rc.rank = r;
        rc.h_start = kTp5HeadStart[r];
        rc.h_cnt   = kTp5HeadCount[r];
        rc.unit    = kTp5WeightUnits[r];
        rc.vocab_start = tp5_slice_start(r, kVocabSize);
        rc.vocab_cnt   = tp5_slice_size(r, kVocabSize);

        check_hip_r(hipSetDevice(r), "init_set_dev");
        check_hip_r(hipStreamCreateWithFlags(&rc.stream, hipStreamNonBlocking), "stream");
        check_rb_r(rocblas_create_handle(&rc.rb), "rb");
        check_rb_r(rocblas_set_atomics_mode(rc.rb, rocblas_atomics_not_allowed), "rb_no_atom");
        check_rb_r(rocblas_set_stream(rc.rb, rc.stream), "rb_stream");

        rc.d_cos.alloc(r, h_cos.size()); rc.d_sin.alloc(r, h_sin.size());
        check_hip_r(hipMemcpy(rc.d_cos.ptr, h_cos.data(), h_cos.size() * 4u, hipMemcpyHostToDevice), "cos");
        check_hip_r(hipMemcpy(rc.d_sin.ptr, h_sin.data(), h_sin.size() * 4u, hipMemcpyHostToDevice), "sin");
        rc.d_lev4.alloc(r, 8); rc.d_lev3.alloc(r, 4); rc.d_lev2.alloc(r, 2);
        check_hip_r(hipMemcpy(rc.d_lev4.ptr, lev4, 32u, hipMemcpyHostToDevice), "lev4");
        check_hip_r(hipMemcpy(rc.d_lev3.ptr, lev3, 16u, hipMemcpyHostToDevice), "lev3");
        check_hip_r(hipMemcpy(rc.d_lev2.ptr, lev2, 8u,  hipMemcpyHostToDevice), "lev2");

        // Allocate 3:3:3:1:1 Sequence-Sharded KV Arena (+ Lossless BF16 KV for massive-activation L74..L77!)
        rc.local_kv_cap = kv_rank_max_toks(r, cfg.max_kv_len);
        const size_t kv_bytes      = size_t(kNumLayers) * size_t(rc.local_kv_cap) * size_t(kKvFp8EntryBytes);
        const size_t kv_bf16_elems = size_t(kNumBf16KvLayers) * size_t(rc.local_kv_cap) * size_t(kKvLatentDim);
        const size_t idx_bytes     = size_t(kNumFullIndexLayers) * size_t(rc.local_kv_cap) * size_t(kIdxKEntryBytes);
        rc.kv_cache_fp8.alloc(r, kv_bytes);
        rc.kv_cache_bf16.alloc(r, kv_bf16_elems);
        rc.idx_k_cache_fp8.alloc(r, idx_bytes);
        rc.local_to_global.alloc(r, size_t(rc.local_kv_cap));
        kv_total_b += double(kv_bytes + kv_bf16_elems * 2u + idx_bytes);
    }
    mem_report_.kv_vram_gib = kv_total_b / 1073741824.0;

    // Pinned host staging buffers (eliminates per-layer malloc/munmap TLB shootdowns & enables zero-copy PCIe DMA!)
    check_hip_r(hipSetDevice(0), "pin_set0");
    for (int r = 0; r < kNumMi50Ranks; ++r) {
        check_hip_r(hipHostMalloc(&impl_->h_rank_part[r],  size_t(Impl::kMaxBatch) * size_t(kHiddenSize) * 4u), "hrp");
        check_hip_r(hipHostMalloc(&impl_->h_qa_part[r],    size_t(Impl::kMaxBatch) * 512u * 2u), "hqp");
        check_hip_r(hipHostMalloc(&impl_->h_qabs_r_pin[r], size_t(Impl::kPinFastM) * 16u * size_t(kKvLatentDim) * 2u), "hqr");
        check_hip_r(hipHostMalloc(&impl_->h_p_max_pin[r],  size_t(Impl::kPinFastM) * size_t(kNumAttnHeads) * 4u), "hpm");
        check_hip_r(hipHostMalloc(&impl_->h_p_sum_pin[r],  size_t(Impl::kPinFastM) * size_t(kNumAttnHeads) * 4u), "hps");
        check_hip_r(hipHostMalloc(&impl_->h_p_acc_pin[r],  size_t(Impl::kPinFastM) * size_t(kNumAttnHeads) * size_t(kKvLoraRank) * 4u), "hpa");
        check_hip_r(hipHostMalloc(&impl_->h_ol_r_pin[r],   size_t(Impl::kPinFastM) * 16u * size_t(kKvLoraRank) * 2u), "holr");
    }
    check_hip_r(hipHostMalloc(&impl_->h_qa_full,      size_t(Impl::kMaxBatch) * size_t(kQLoraRank) * 2u), "hqf");
    check_hip_r(hipHostMalloc(&impl_->h_p150_out,     size_t(Impl::kMaxBatch) * size_t(kHiddenSize) * 4u), "hpo");
    check_hip_r(hipHostMalloc(&impl_->h_ddr4_out,     size_t(Impl::kMaxBatch) * size_t(kHiddenSize) * 4u), "hdo");
    check_hip_r(hipHostMalloc(&impl_->h_x_rot_bf16,   size_t(Impl::kMaxBatch) * size_t(kHiddenSize) * 2u), "hxb");
    check_hip_r(hipHostMalloc(&impl_->h_x_rot_f32,    size_t(Impl::kMaxBatch) * size_t(kHiddenSize) * 4u), "hxf");
    check_hip_r(hipHostMalloc(&impl_->h_out_bf16,     size_t(Impl::kMaxBatch) * size_t(kHiddenSize) * 2u), "hob");
    check_hip_r(hipHostMalloc(&impl_->h_x_host_bf16,  size_t(Impl::kMaxBatch) * size_t(kHiddenSize) * 2u), "hxh");
    check_hip_r(hipHostMalloc(&impl_->h_topk_ids_pin, size_t(Impl::kMaxBatch) * size_t(kTopK) * 4u), "hti");
    check_hip_r(hipHostMalloc(&impl_->h_topk_wts_pin, size_t(Impl::kMaxBatch) * size_t(kTopK) * 4u), "htw");
    check_hip_r(hipHostMalloc(&impl_->h_qabs_pin,     size_t(Impl::kPinFastM) * size_t(kNumAttnHeads) * size_t(kKvLatentDim) * 2u), "hqap");
    check_hip_r(hipHostMalloc(&impl_->h_olat_pin,     size_t(Impl::kPinFastM) * size_t(kNumAttnHeads) * size_t(kKvLoraRank) * 2u), "holp");
    impl_->h_x_raw_f32.resize(size_t(Impl::kMaxBatch) * size_t(kHiddenSize), 0.0f);

    // 4. Load Shared Spine (embed_tokens -> host DDR4; hc_head + norm -> R0; lm_head -> TP5 2:2:2:1:1!)
    {
        const std::string shared_path = cfg.root_dir + "/packed_spine/spine_shared.safetensors";
        {
            hy4::SafetensorsMmap st(shared_path);
            const auto& esp = st.at("model.embed_tokens.weight");
            impl_->embed_host_bf16.resize(size_t(esp.byte_size()) / 2u);
            std::memcpy(impl_->embed_host_bf16.data(), st.tensor_bytes("model.embed_tokens.weight"), size_t(esp.byte_size()));

            upload_bf16_or_f32_as_f32(st, "model.hc_head.hc_head_fn",    0, kHcMult * kHcFlatDim, 0, impl_->r0_hc_head_fn);
            upload_bf16_or_f32_as_f32(st, "model.hc_head.hc_head_base",  0, kHcMult, 0, impl_->r0_hc_head_base);
            upload_bf16_or_f32_as_f32(st, "model.hc_head.hc_head_scale", 0, 1, 0, impl_->r0_hc_head_scale);
            upload_bf16(st, "model.norm.weight", 0, impl_->r0_model_norm);

            for (int r = 0; r < kNumMi50Ranks; ++r) {
                auto& rc = impl_->ranks[r];
                upload_bf16_as_f16_slice(st, "lm_head.weight", rc.vocab_start, rc.vocab_cnt, kHiddenSize, r, rc.lm_head_f16);
            }
        }
        drop_file_page_cache(shared_path);
    }

    // 5. Load all 78 Backbone Spine Layers + MI50 EP Routed Experts + Build O(1) 3-Tier Routing Tables!
    //    (Evict GPU-resident spine & fp8_hot files from Linux page cache so 100% of host RAM stays dedicated to DDR4 cold experts!)
    for (int l = 0; l < kNumLayers; ++l) {
        const std::string sp_path = cfg.root_dir + "/packed_spine/spine_layer_" + ltag(l) + ".safetensors";
        {
            hy4::SafetensorsMmap st(sp_path);
        const std::string pfx = "model.layers." + std::to_string(l) + ".";

        // R0 tiny params
        auto& r0e = impl_->r0_layers[size_t(l)];
        upload_bf16_or_f32_as_f32(st, pfx + "hc_attn_layer.hc_pre.hc_fn",    0, 2 * kHcMult * kHcFlatDim, 0, r0e.hc_attn_fn);
        upload_bf16_or_f32_as_f32(st, pfx + "hc_attn_layer.hc_pre.hc_base",  0, 2 * kHcMult, 0, r0e.hc_attn_base);
        upload_bf16_or_f32_as_f32(st, pfx + "hc_attn_layer.hc_pre.hc_scale", 0, 2, 0, r0e.hc_attn_scale);
        upload_bf16_or_f32_as_f32(st, pfx + "hc_mlp_layer.hc_pre.hc_fn",     0, 2 * kHcMult * kHcFlatDim, 0, r0e.hc_mlp_fn);
        upload_bf16_or_f32_as_f32(st, pfx + "hc_mlp_layer.hc_pre.hc_base",   0, 2 * kHcMult, 0, r0e.hc_mlp_base);
        upload_bf16_or_f32_as_f32(st, pfx + "hc_mlp_layer.hc_pre.hc_scale",  0, 2, 0, r0e.hc_mlp_scale);
        upload_bf16(st, pfx + "input_layernorm.weight", 0, r0e.input_ln);
        upload_bf16(st, pfx + "post_attention_layernorm.weight", 0, r0e.post_ln);
        upload_bf16(st, pfx + "self_attn.q_a_layernorm.weight", 0, r0e.qa_ln);
        upload_bf16(st, pfx + "self_attn.kv_a_layernorm.weight", 0, r0e.kva_ln);
        slice_fp8_rows_to_dev(st, pfx + "self_attn.kv_a_proj_with_mqa", 0, kKvLatentDim, kHiddenSize, 0, r0e.kv_a);

        if (is_full_indexer_layer(l)) {
            upload_bf16(st, pfx + "self_attn.indexer.wk.weight", 0, r0e.idx_wk_bf16);
            upload_bf16(st, pfx + "self_attn.indexer.k_norm.weight", 0, r0e.idx_knorm_w);
            upload_bf16(st, pfx + "self_attn.indexer.k_norm.bias",   0, r0e.idx_knorm_b);
            upload_bf16(st, pfx + "self_attn.indexer.weights_proj.weight", 0, r0e.idx_wproj_bf16);
        }
        if (l > 0) {
            upload_bf16(st, pfx + "mlp.gate.weight", 0, r0e.router_w_bf16);
            upload_bf16_or_f32_as_f32(st, pfx + "mlp.gate.e_score_correction_bias", 0, kNumRoutedExperts, 0, r0e.router_bias_f32);
        }

        // 2:2:2:1:1 TP5 Spine slices across R0..R4 + host-cached sinks + rank-local tiny norms
        {
            const std::string snm = pfx + "self_attn.learnable_sink_param";
            const auto& ssp = st.at(snm);
            if (ssp.dtype == "F32") {
                std::memcpy(impl_->h_sinks[size_t(l)].data(), st.tensor_bytes(snm), kNumAttnHeads * 4u);
            } else {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(st.tensor_bytes(snm));
                for (int h = 0; h < kNumAttnHeads; ++h) impl_->h_sinks[size_t(l)][size_t(h)] = bf16_to_f32(p[h]);
            }
        }
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            auto& rc = impl_->ranks[r];
            auto& lr = rc.layers[size_t(l)];
            const int qa_beg = tp5_slice_start(r, kQLoraRank);
            const int qa_cnt = tp5_slice_size(r, kQLoraRank);
            slice_fp8_rows_to_dev(st, pfx + "self_attn.q_a_proj",    qa_beg, qa_cnt, kHiddenSize, r, lr.q_a);
            slice_fp8_rows_to_dev(st, pfx + "self_attn.q_b_proj",    rc.h_start * kQkHeadDim, rc.h_cnt * kQkHeadDim, kQLoraRank, r, lr.q_b);
            load_kv_b_slice_to_uk_uv(st, pfx + "self_attn.kv_b_proj", rc.h_start, rc.h_cnt, r, lr.w_uk, lr.w_uv);
            slice_fp8_rows_to_dev(st, pfx + "self_attn.linear_gate", rc.h_start * kVHeadDim, rc.h_cnt * kVHeadDim, kHiddenSize, r, lr.linear_gate);
            slice_fp8_cols_to_dev(st, pfx + "self_attn.o_proj",      kHiddenSize, rc.h_start * kVHeadDim, rc.h_cnt * kVHeadDim, kNumAttnHeads * kVHeadDim, r, lr.o_proj);
            upload_bf16_or_f32_as_f32(st, pfx + "self_attn.learnable_sink_param", rc.h_start, rc.h_cnt, r, lr.sinks);
            upload_bf16(st, pfx + "self_attn.kv_a_layernorm.weight", r, lr.kva_ln);

            if (is_full_indexer_layer(l)) {
                const int ih_beg = (rc.h_start / 2) * kIndexHeadDim;
                const int ih_cnt = (rc.h_cnt / 2)   * kIndexHeadDim;
                upload_bf16_as_f16_slice(st, pfx + "self_attn.indexer.wq_b.weight", ih_beg, ih_cnt, kQLoraRank, r, lr.idx_wq_b_f16);
                upload_bf16(st, pfx + "self_attn.indexer.k_norm.weight", r, lr.idx_knorm_w);
                upload_bf16(st, pfx + "self_attn.indexer.k_norm.bias",   r, lr.idx_knorm_b);
            }

            if (l == 0) {
                const int mb = tp5_slice_start(r, kDenseInterSize);
                const int mc = tp5_slice_size(r, kDenseInterSize);
                slice_fp8_rows_to_dev(st, pfx + "mlp.gate_proj", mb, mc, kHiddenSize, r, lr.mlp_gate);
                slice_fp8_rows_to_dev(st, pfx + "mlp.up_proj",   mb, mc, kHiddenSize, r, lr.mlp_up);
                slice_fp8_cols_to_dev(st, pfx + "mlp.down_proj", kHiddenSize, mb, mc, kDenseInterSize, r, lr.mlp_down);
            } else {
                const int mb = tp5_slice_start(r, kSharedInterSize);
                const int mc = tp5_slice_size(r, kSharedInterSize);
                slice_fp8_rows_to_dev(st, pfx + "mlp.shared_experts.gate_proj", mb, mc, kHiddenSize, r, lr.mlp_gate);
                slice_fp8_rows_to_dev(st, pfx + "mlp.shared_experts.up_proj",   mb, mc, kHiddenSize, r, lr.mlp_up);
                slice_fp8_cols_to_dev(st, pfx + "mlp.shared_experts.down_proj", kHiddenSize, mb, mc, kSharedInterSize, r, lr.mlp_down);
            }
        }

        // Load Layer l MI50 EP Routed Experts & Build O(1) 3-Tier Routing Table!
        if (l > 0) {
            const int32_t* orig_to_rank = reinterpret_cast<const int32_t*>(st.tensor_bytes("expert_orig_to_rank"));
            const auto sh = make_signs_h(l);
            const auto si = make_signs_i(l);
            impl_->h_signs_h[size_t(l)] = sh;
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                auto& lr = impl_->ranks[r].layers[size_t(l)];
                lr.signs_h.alloc(r, sh.size()); lr.signs_i.alloc(r, si.size());
                check_hip_r(hipMemcpy(lr.signs_h.ptr, sh.data(), sh.size() * 8u, hipMemcpyHostToDevice), "sh");
                check_hip_r(hipMemcpy(lr.signs_i.ptr, si.data(), si.size() * 8u, hipMemcpyHostToDevice), "si");
            }

            // Build rank -> (device, format, local_slot) mapping matching make_tier_plan & P150A!
            std::array<ExpertRouteEntry, kNumRoutedExperts> by_rank{};
            int cur_rank = 0;
            std::array<int, kNumMi50Ranks> r_fp8_beg{}, r_fp8_cnt{}, r_h4_beg{}, r_h4_cnt{}, r_h3_beg{}, r_h3_cnt{}, r_h2_beg{}, r_h2_cnt{};
            // Lossless Stock FP8-g128 on MI50 R0..R4 (Profile A-FP8-HLUT4-Q5!)
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                r_fp8_beg[size_t(r)] = cur_rank;
                r_fp8_cnt[size_t(r)] = impl_->plan.mi50_fp8[r];
                for (int i = 0; i < r_fp8_cnt[size_t(r)]; ++i) {
                    by_rank[size_t(cur_rank++)] = {ExpertDevice(r), ExpertFormat::FP8, i};
                }
            }
            // HLUT4 on MI50 R0..R4 (Profile A-HLUT4)
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                r_h4_beg[size_t(r)] = cur_rank;
                r_h4_cnt[size_t(r)] = impl_->plan.mi50_hlut4[r];
                for (int i = 0; i < r_h4_cnt[size_t(r)]; ++i) {
                    by_rank[size_t(cur_rank++)] = {ExpertDevice(r), ExpertFormat::HLUT4, i};
                }
            }
            // HLUT4 on P150A (local slots [0 .. p150_hlut4))
            for (int i = 0; i < impl_->plan.p150_hlut4; ++i) {
                by_rank[size_t(cur_rank++)] = {ExpertDevice::P150_TP2, ExpertFormat::HLUT4, i};
            }
            // HLUT3 on MI50 R0..R4
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                r_h3_beg[size_t(r)] = cur_rank;
                r_h3_cnt[size_t(r)] = impl_->plan.mi50_hlut3[r];
                for (int i = 0; i < r_h3_cnt[size_t(r)]; ++i) {
                    by_rank[size_t(cur_rank++)] = {ExpertDevice(r), ExpertFormat::HLUT3, i};
                }
            }
            // HLUT3 on P150A (local slots [p150_hlut4 .. p150_hlut4 + p150_hlut3))
            impl_->p150_h3_rank_begin = cur_rank;
            for (int i = 0; i < impl_->plan.p150_hlut3; ++i) {
                by_rank[size_t(cur_rank++)] = {ExpertDevice::P150_TP2, ExpertFormat::HLUT3, impl_->plan.p150_hlut4 + i};
            }
            // HLUT2 on MI50 R0..R4 (Profile B only)
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                r_h2_beg[size_t(r)] = cur_rank;
                r_h2_cnt[size_t(r)] = impl_->plan.mi50_hlut2[r];
                for (int i = 0; i < r_h2_cnt[size_t(r)]; ++i) {
                    by_rank[size_t(cur_rank++)] = {ExpertDevice(r), ExpertFormat::HLUT2, i};
                }
            }
            // HLUT2 on P150A (Profile B only: local slots offset by p150_hlut4 + p150_hlut3!)
            for (int i = 0; i < impl_->plan.p150_hlut2; ++i) {
                by_rank[size_t(cur_rank++)] = {ExpertDevice::P150_TP2, ExpertFormat::HLUT2, impl_->plan.p150_hlut4 + impl_->plan.p150_hlut3 + i};
            }
            // Q5_LM128 or HLUT4 Cold Tail on EPYC DDR4 AVX2
            const ExpertFormat ddr4_fmt = (impl_->plan.ddr4_q5 > 0) ? ExpertFormat::Q5_LM128 : ExpertFormat::HLUT4;
            for (int i = 0; cur_rank < kNumRoutedExperts; ++i, ++cur_rank) {
                by_rank[size_t(cur_rank)] = {ExpertDevice::CPU_DDR4, ddr4_fmt, i};
            }
            for (int orig = 0; orig < kNumRoutedExperts; ++orig) {
                impl_->route_tbl[size_t(l)][size_t(orig)] = by_rank[size_t(orig_to_rank[orig])];
            }

            // Stream MI50 Lossless Stock FP8-g128 experts from root_dir/fp8_hot/fp8_hot_layer_XX.bin!
            if (impl_->plan.vram_fp8_per_layer() > 0) {
                std::ifstream ifs(cfg.root_dir + "/fp8_hot/fp8_hot_layer_" + ltag(l) + ".bin", std::ios::binary);
                if (!ifs) throw std::runtime_error("Missing fp8_hot shard for layer " + std::to_string(l));
                std::vector<uint8_t> raw(static_cast<size_t>(kFp8ExpertBytes));
                const size_t s_floats = size_t(kFp8ProjSBytes) / 4u;
                for (int r = 0; r < kNumMi50Ranks; ++r) {
                    const int cnt = r_fp8_cnt[size_t(r)];
                    if (cnt <= 0) continue;
                    auto& lr = impl_->ranks[r].layers[size_t(l)];
                    lr.n_fp8 = cnt;
                    lr.fp8_gate_w.alloc(r, size_t(cnt) * size_t(kFp8ProjWBytes));
                    lr.fp8_gate_s.alloc(r, size_t(cnt) * s_floats);
                    lr.fp8_up_w.alloc(r,   size_t(cnt) * size_t(kFp8ProjWBytes));
                    lr.fp8_up_s.alloc(r,   size_t(cnt) * s_floats);
                    lr.fp8_down_w.alloc(r, size_t(cnt) * size_t(kFp8ProjWBytes));
                    lr.fp8_down_s.alloc(r, size_t(cnt) * s_floats);
                    ifs.seekg(std::streamoff(size_t(r_fp8_beg[size_t(r)]) * size_t(kFp8ExpertBytes)), std::ios::beg);
                    check_hip_r(hipSetDevice(r), "fp8_set");
                    for (int i = 0; i < cnt; ++i) {
                        if (!ifs.read(reinterpret_cast<char*>(raw.data()), std::streamsize(kFp8ExpertBytes))) {
                            throw std::runtime_error("Short read on fp8_hot layer " + std::to_string(l));
                        }
                        const uint8_t* p = raw.data();
                        check_hip_r(hipMemcpy(lr.fp8_gate_w.ptr + size_t(i) * kFp8ProjWBytes, p, kFp8ProjWBytes, hipMemcpyHostToDevice), "f8gw"); p += kFp8ProjWBytes;
                        check_hip_r(hipMemcpy(lr.fp8_gate_s.ptr + size_t(i) * s_floats,       p, kFp8ProjSBytes, hipMemcpyHostToDevice), "f8gs"); p += kFp8ProjSBytes;
                        check_hip_r(hipMemcpy(lr.fp8_up_w.ptr   + size_t(i) * kFp8ProjWBytes, p, kFp8ProjWBytes, hipMemcpyHostToDevice), "f8uw"); p += kFp8ProjWBytes;
                        check_hip_r(hipMemcpy(lr.fp8_up_s.ptr   + size_t(i) * s_floats,       p, kFp8ProjSBytes, hipMemcpyHostToDevice), "f8us"); p += kFp8ProjSBytes;
                        check_hip_r(hipMemcpy(lr.fp8_down_w.ptr + size_t(i) * kFp8ProjWBytes, p, kFp8ProjWBytes, hipMemcpyHostToDevice), "f8dw"); p += kFp8ProjWBytes;
                        check_hip_r(hipMemcpy(lr.fp8_down_s.ptr + size_t(i) * s_floats,       p, kFp8ProjSBytes, hipMemcpyHostToDevice), "f8ds");
                    }
                }
                ifs.close();
                drop_file_page_cache(cfg.root_dir + "/fp8_hot/fp8_hot_layer_" + ltag(l) + ".bin");
            }

            // Stream MI50 HLUT4 experts from hlut4_hot/hlut4_hot_layer_XX.bin (100% HLUT4 Profile A!)
            const int mi50_h4_tot = r_h4_cnt[0] + r_h4_cnt[1] + r_h4_cnt[2] + r_h4_cnt[3] + r_h4_cnt[4];
            if (mi50_h4_tot > 0) {
                std::ifstream ifs(cfg.weight_root + "/hlut4_hot/hlut4_hot_layer_" + ltag(l) + ".bin", std::ios::binary);
                if (!ifs) throw std::runtime_error("Missing hlut4_hot shard for layer " + std::to_string(l));
                std::vector<Hlut4Tile32x32> raw(3u * kProjTiles);
                for (int r = 0; r < kNumMi50Ranks; ++r) {
                    const int cnt = r_h4_cnt[size_t(r)];
                    if (cnt <= 0) continue;
                    auto& lr = impl_->ranks[r].layers[size_t(l)];
                    lr.n_h4 = cnt;
                    lr.h4_gate.alloc(r, size_t(cnt) * kProjTiles);
                    lr.h4_up.alloc(r,   size_t(cnt) * kProjTiles);
                    lr.h4_down.alloc(r, size_t(cnt) * kProjTiles);
                    ifs.seekg(std::streamoff(size_t(r_h4_beg[size_t(r)]) * size_t(kHlut4ExpertBytes)), std::ios::beg);
                    for (int i = 0; i < cnt; ++i) {
                        if (!ifs.read(reinterpret_cast<char*>(raw.data()), std::streamsize(kHlut4ExpertBytes))) {
                            throw std::runtime_error("Short read on hlut4_hot layer " + std::to_string(l));
                        }
                        check_hip_r(hipSetDevice(r), "h4_set");
                        check_hip_r(hipMemcpy(lr.h4_gate.ptr + size_t(i) * kProjTiles, raw.data() + 0u * kProjTiles, kHlut4ProjBytes, hipMemcpyHostToDevice), "h4g");
                        check_hip_r(hipMemcpy(lr.h4_up.ptr   + size_t(i) * kProjTiles, raw.data() + 1u * kProjTiles, kHlut4ProjBytes, hipMemcpyHostToDevice), "h4u");
                        check_hip_r(hipMemcpy(lr.h4_down.ptr + size_t(i) * kProjTiles, raw.data() + 2u * kProjTiles, kHlut4ProjBytes, hipMemcpyHostToDevice), "h4d");
                    }
                }
            }
            // Stream MI50 HLUT3 experts from hlut3_layer_XX.bin
            if (impl_->plan.vram_hlut3_per_layer() > 0) {
                std::ifstream ifs(cfg.weight_root + "/hlut3/hlut3_layer_" + ltag(l) + ".bin", std::ios::binary);
                if (!ifs) throw std::runtime_error("Missing hlut3 shard for layer " + std::to_string(l));
                std::vector<Hlut3Tile32x32> raw(3u * kProjTiles);
                for (int r = 0; r < kNumMi50Ranks; ++r) {
                    const int cnt = r_h3_cnt[size_t(r)];
                    if (cnt <= 0) continue;
                    auto& lr = impl_->ranks[r].layers[size_t(l)];
                    lr.n_h3 = cnt;
                    lr.h3_gate.alloc(r, size_t(cnt) * kProjTiles);
                    lr.h3_up.alloc(r,   size_t(cnt) * kProjTiles);
                    lr.h3_down.alloc(r, size_t(cnt) * kProjTiles);
                    ifs.seekg(std::streamoff(size_t(r_h3_beg[size_t(r)]) * size_t(kHlut3ExpertBytes)), std::ios::beg);
                    for (int i = 0; i < cnt; ++i) {
                        if (!ifs.read(reinterpret_cast<char*>(raw.data()), std::streamsize(kHlut3ExpertBytes))) {
                            throw std::runtime_error("Short read on hlut3 layer " + std::to_string(l));
                        }
                        check_hip_r(hipSetDevice(r), "h3_set");
                        check_hip_r(hipMemcpy(lr.h3_gate.ptr + size_t(i) * kProjTiles, raw.data() + 0u * kProjTiles, kHlut3ProjBytes, hipMemcpyHostToDevice), "h3g");
                        check_hip_r(hipMemcpy(lr.h3_up.ptr   + size_t(i) * kProjTiles, raw.data() + 1u * kProjTiles, kHlut3ProjBytes, hipMemcpyHostToDevice), "h3u");
                        check_hip_r(hipMemcpy(lr.h3_down.ptr + size_t(i) * kProjTiles, raw.data() + 2u * kProjTiles, kHlut3ProjBytes, hipMemcpyHostToDevice), "h3d");
                    }
                }
            }
            // Stream MI50 HLUT2 experts from hlut2_layer_XX.bin (Profile B only)
            if (impl_->plan.vram_hlut2_per_layer() > 0) {
                std::ifstream ifs(cfg.weight_root + "/hlut2/hlut2_layer_" + ltag(l) + ".bin", std::ios::binary);
                if (!ifs) throw std::runtime_error("Missing hlut2 shard for layer " + std::to_string(l));
                std::vector<Hlut2Tile32x32> raw(3u * kProjTiles);
                for (int r = 0; r < kNumMi50Ranks; ++r) {
                    const int cnt = r_h2_cnt[size_t(r)];
                    if (cnt <= 0) continue;
                    auto& lr = impl_->ranks[r].layers[size_t(l)];
                    lr.n_h2 = cnt;
                    lr.h2_gate.alloc(r, size_t(cnt) * kProjTiles);
                    lr.h2_up.alloc(r,   size_t(cnt) * kProjTiles);
                    lr.h2_down.alloc(r, size_t(cnt) * kProjTiles);
                    ifs.seekg(std::streamoff(size_t(r_h2_beg[size_t(r)] - kPackHlut2Begin) * size_t(kHlut2ExpertBytes)), std::ios::beg);
                    for (int i = 0; i < cnt; ++i) {
                        if (!ifs.read(reinterpret_cast<char*>(raw.data()), std::streamsize(kHlut2ExpertBytes))) {
                            throw std::runtime_error("Short read on hlut2 layer " + std::to_string(l));
                        }
                        check_hip_r(hipSetDevice(r), "h2_set");
                        check_hip_r(hipMemcpy(lr.h2_gate.ptr + size_t(i) * kProjTiles, raw.data() + 0u * kProjTiles, kHlut2ProjBytes, hipMemcpyHostToDevice), "h2g");
                        check_hip_r(hipMemcpy(lr.h2_up.ptr   + size_t(i) * kProjTiles, raw.data() + 1u * kProjTiles, kHlut2ProjBytes, hipMemcpyHostToDevice), "h2u");
                        check_hip_r(hipMemcpy(lr.h2_down.ptr + size_t(i) * kProjTiles, raw.data() + 2u * kProjTiles, kHlut2ProjBytes, hipMemcpyHostToDevice), "h2d");
                    }
                }
                ifs.close();
                drop_file_page_cache(cfg.weight_root + "/hlut2/hlut2_layer_" + ltag(l) + ".bin");
            }
        }
        }
        drop_file_page_cache(sp_path);
        if (l == 0 || l == 77 || (l % 16) == 0) {
            std::printf("[HY4-770B-RT] Loaded layer %02d/77 across 5x MI50...\n", l);
            std::fflush(stdout);
        }
    }

    // 6. Load 3.3B 5-layer DFly2 Speculative Drafter (sharded 2:2:2:1:1 across 5x MI50!)
    const std::string dfly_path = cfg.root_dir + "/packed_spine/dfly2.safetensors";
    if (cfg.use_dfly2 && std::filesystem::is_regular_file(dfly_path)) {
        {
        hy4::SafetensorsMmap st(dfly_path);
        upload_bf16(st, "model.dfly_layers.mtp_fusion.enorm.weight", 0, impl_->dfly_enorm);
        upload_bf16(st, "model.dfly_layers.mtp_fusion.hnorm.weight", 0, impl_->dfly_hnorm);
        upload_bf16(st, "model.dfly_layers.mtp_fusion.gate",         0, impl_->dfly_gate);
        upload_bf16_as_f16_slice(st, "model.dfly_layers.mtp_fusion.eh_proj.weight", 0, kHiddenSize, 2 * kHiddenSize, 0, impl_->dfly_eh_proj_f16);
        upload_bf16(st, "model.dfly_layers.hidden_correction.hidden_norm.weight", 0, impl_->dfly_hc_hnorm);
        upload_bf16(st, "model.dfly_layers.hidden_correction.embed_norm.weight",  0, impl_->dfly_hc_enorm);
        slice_fp8_rows_to_dev(st, "model.dfly_layers.hidden_correction.gate_proj", 0, kHiddenSize, 2 * kHiddenSize, 0, impl_->dfly_hc_gate);
        slice_fp8_rows_to_dev(st, "model.dfly_layers.hidden_correction.up_proj",   0, kHiddenSize, 2 * kHiddenSize, 0, impl_->dfly_hc_up);
        slice_fp8_rows_to_dev(st, "model.dfly_layers.hidden_correction.down_proj", 0, kHiddenSize, kHiddenSize,     0, impl_->dfly_hc_down);
        upload_bf16(st, "model.dfly_layers.final_norm.weight", 0, impl_->dfly_final_norm);
        upload_bf16(st, "model.dfly_layers.confidence_head.proj.weight", 0, impl_->dfly_conf_w);
        upload_bf16(st, "model.dfly_layers.confidence_head.proj.bias",   0, impl_->dfly_conf_b);
        {
            int32_t zslots[kDflyBlockSize];
            for (int i = 0; i < kDflyBlockSize; ++i) zslots[i] = i;
            impl_->dfly_d_zslots.alloc(0, size_t(kDflyBlockSize));
            check_hip_r(hipMemcpy(impl_->dfly_d_zslots.ptr, zslots, sizeof(zslots), hipMemcpyHostToDevice), "dfly_zsl");
        }

        for (int dl = 0; dl < kDflyNumLayers; ++dl) {
            const std::string dp = "model.dfly_layers." + std::to_string(dl) + ".";
            upload_bf16(st, dp + "input_layernorm.weight", 0, impl_->ranks[0].dfly[size_t(dl)].input_ln);
            upload_bf16(st, dp + "post_attention_layernorm.weight", 0, impl_->ranks[0].dfly[size_t(dl)].post_ln);
            upload_bf16(st, dp + "self_attn.q_a_layernorm.weight", 0, impl_->ranks[0].dfly[size_t(dl)].qa_ln);
            upload_bf16(st, dp + "self_attn.kv_a_layernorm.weight", 0, impl_->ranks[0].dfly[size_t(dl)].kva_ln);
            slice_fp8_rows_to_dev(st, dp + "self_attn.kv_a_proj_with_mqa", 0, kKvLatentDim, kHiddenSize, 0, impl_->ranks[0].dfly[size_t(dl)].kv_a);
            {
                const std::string snm = dp + "self_attn.learnable_sink_param";
                const auto& ssp = st.at(snm);
                if (ssp.dtype == "F32") {
                    std::memcpy(impl_->h_dfly_sinks[size_t(dl)].data(), st.tensor_bytes(snm), kNumAttnHeads * 4u);
                } else {
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(st.tensor_bytes(snm));
                    for (int h = 0; h < kNumAttnHeads; ++h) impl_->h_dfly_sinks[size_t(dl)][size_t(h)] = bf16_to_f32(p[h]);
                }
            }
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                auto& rc = impl_->ranks[r];
                auto& df = rc.dfly[size_t(dl)];
                slice_fp8_rows_to_dev(st, dp + "self_attn.q_a_proj", tp5_slice_start(r, kQLoraRank), tp5_slice_size(r, kQLoraRank), kHiddenSize, r, df.q_a);
                slice_fp8_rows_to_dev(st, dp + "self_attn.q_b_proj", rc.h_start * kQkHeadDim, rc.h_cnt * kQkHeadDim, kQLoraRank, r, df.q_b);
                load_kv_b_slice_to_uk_uv(st, dp + "self_attn.kv_b_proj", rc.h_start, rc.h_cnt, r, df.w_uk, df.w_uv);
                slice_fp8_rows_to_dev(st, dp + "self_attn.linear_gate", rc.h_start * kVHeadDim, rc.h_cnt * kVHeadDim, kHiddenSize, r, df.linear_gate);
                slice_fp8_cols_to_dev(st, dp + "self_attn.o_proj", kHiddenSize, rc.h_start * kVHeadDim, rc.h_cnt * kVHeadDim, kNumAttnHeads * kVHeadDim, r, df.o_proj);
                upload_bf16_or_f32_as_f32(st, dp + "self_attn.learnable_sink_param", rc.h_start, rc.h_cnt, r, df.sinks);
                const int mb = tp5_slice_start(r, kDflyInterSize);
                const int mc = tp5_slice_size(r, kDflyInterSize);
                slice_fp8_rows_to_dev(st, dp + "mlp.gate_proj", mb, mc, kHiddenSize, r, df.mlp_gate);
                slice_fp8_rows_to_dev(st, dp + "mlp.up_proj",   mb, mc, kHiddenSize, r, df.mlp_up);
                slice_fp8_cols_to_dev(st, dp + "mlp.down_proj", kHiddenSize, mb, mc, kDflyInterSize, r, df.mlp_down);
            }
        }
        impl_->dfly2_loaded = true;
        mem_report_.dfly2_active = true;
        }
        drop_file_page_cache(dfly_path);
    }

    // 6B. Map & Prefault EPYC DDR4 Cold Tail LAST (after all GPU weights are loaded & evicted from Linux page cache!)
    const int ddr4_begin = impl_->plan.vram_total_per_layer();
    const bool ddr4_use_q5 = (impl_->plan.ddr4_q5 > 0);
    const int ddr4_count = impl_->plan.ddr4_total_per_layer();
    if (cfg.use_ddr4_avx2 && ddr4_count > 0) {
        bool ok = false;
        if (ddr4_use_q5) {
            ok = impl_->ddr4_engine.init(cfg.weight_root + "/q5_ddr4", ddr4_begin,
                                         cfg.ddr4_threads, cfg.ddr4_core_start, "", true, cfg.use_hugepages);
        } else {
            ok = impl_->ddr4_engine.init(cfg.weight_root + "/hlut4_ddr4", ddr4_begin,
                                         cfg.ddr4_threads, cfg.ddr4_core_start,
                                         cfg.weight_root + "/hlut4_hot", false, cfg.use_hugepages);
        }
        if (!ok) throw std::runtime_error("Failed to initialize DDR4 AVX2 cold-tail engine!");
        impl_->ddr4_ready = true;
        mem_report_.ddr4_cold_gib = double(impl_->ddr4_engine.resident_bytes()) / 1073741824.0;
    }

    // 7. Query live VRAM footprint across all 5 MI50s & pick rank with most headroom for on-demand ViT
    int best_vit_rank = 0;
    size_t best_vit_free = 0;
    for (int r = 0; r < kNumMi50Ranks; ++r) {
        size_t free_b = 0, tot_b = 0;
        check_hip_r(hipSetDevice(r), "mem_set");
        check_hip_r(hipMemGetInfo(&free_b, &tot_b), "mem_info");
        mem_report_.hip_used_gib[r]  = double(tot_b - free_b) / 1073741824.0;
        mem_report_.hip_total_gib[r] = double(tot_b) / 1073741824.0;
        if (free_b > best_vit_free) { best_vit_free = free_b; best_vit_rank = r; }
    }

    // 8. Init Dual-NVMe .kvblk & On-Demand 40-Layer Dynamic ViT (0 MiB permanent VRAM, auto-placed on max-free physical MI50!)
    if (cfg.enable_kv_disk_cache) impl_->kv_disk.init(cfg.kv_dir0, cfg.kv_dir1);
    if (cfg.load_vit) {
        const int vit_phys_dev = g_rank_to_hip[best_vit_rank];
        (void)impl_->vit.init(cfg.root_dir + "/packed_spine/vit.safetensors", vit_phys_dev);
        std::printf("[HY4-770B-RT] Dynamic 40L ViT armed on-demand on R%d (HIP%d, %.2f GiB free headroom)\n",
                    best_vit_rank, vit_phys_dev, double(best_vit_free) / 1073741824.0);
    }

    // 9. Start DPM Warm Guard thread (gentle 800 ms keep-warm on idle MI50s)
    impl_->last_activity_ms.store(impl_->now_ms(), std::memory_order_relaxed);
    if (cfg.enable_dpm_warm_guard) {
        impl_->dpm_thread = std::thread([this]() {
            while (!impl_->dpm_stop.load(std::memory_order_acquire)) {
                for (int s = 0; s < 8 && !impl_->dpm_stop.load(std::memory_order_acquire); ++s) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                if (impl_->dpm_stop.load(std::memory_order_acquire)) break;
                if (impl_->now_ms() - impl_->last_activity_ms.load(std::memory_order_relaxed) > 900ULL) {
                    std::unique_lock<std::mutex> lk(impl_->exec_mu, std::try_to_lock);
                    if (lk.owns_lock() && !impl_->shutting_down.load(std::memory_order_acquire)) {
                        for (int r = 0; r < kNumMi50Ranks; ++r) {
                            (void)hipSetDevice(r);
                            (void)hipStreamSynchronize(impl_->ranks[r].stream);
                        }
                    }
                }
            }
        });
    }

    impl_->initialized = true;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[HY4-770B-RT] READY in %.1fs | Profile=%s | max_kv=%d | VRAM=[%.1f, %.1f, %.1f, %.1f, %.1f] GiB | "
                "P150=%s (%.1f GiB/c, %.0f MiB L1) | DDR4=%.1f GiB | DFly2=%d | ViT=%d\n",
                sec, profile_name(cfg.profile), cfg.max_kv_len,
                mem_report_.hip_used_gib[0], mem_report_.hip_used_gib[1], mem_report_.hip_used_gib[2],
                mem_report_.hip_used_gib[3], mem_report_.hip_used_gib[4],
                mem_report_.p150_active ? "ON" : "OFF", mem_report_.p150_used_gib[0], mem_report_.p150_l1_mib_total,
                mem_report_.ddr4_cold_gib, int(mem_report_.dfly2_active), int(impl_->vit.ready()));
    std::fflush(stdout);
}

Hy4770bTierStats Hy4770bRuntime::cumulative_tier_stats() const noexcept {
    Hy4770bTierStats s{};
    s.total_routes = impl_->stat_total_routes.load(std::memory_order_relaxed);
    s.mi50_hbm2_hits = impl_->stat_mi50_hits.load(std::memory_order_relaxed);
    const uint64_t p_tot = impl_->stat_p150_hits.load(std::memory_order_relaxed);
    uint64_t l1_h = 0;
    if (impl_->p150_connected && impl_->p150_shm.ptr) {
        l1_h = std::min(p_tot, impl_->p150_shm.ptr->stat_l1_hits.load(std::memory_order_relaxed));
    }
    s.tensix_l1_hits  = l1_h;
    s.p150_gddr6_hits = (p_tot >= l1_h) ? (p_tot - l1_h) : 0ULL;
    s.epyc_ddr4_hits  = impl_->stat_ddr4_hits.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(impl_->stats_mu);
    s.mi50_ms_total = impl_->stat_mi50_ms;
    s.p150_ms_total = impl_->stat_p150_ms;
    s.ddr4_ms_total = impl_->stat_ddr4_ms;
    return s;
}

void Hy4770bRuntime::reset_step_profile() {
    std::lock_guard<std::mutex> lk(impl_->stats_mu);
    impl_->prof = Hy4770bStepProfile{};
}

Hy4770bStepProfile Hy4770bRuntime::step_profile() const noexcept {
    std::lock_guard<std::mutex> lk(impl_->stats_mu);
    return impl_->prof;
}

void Hy4770bRuntime::clear_kv_cache() {
    impl_->kv_len = 0;
    impl_->cached_toks.clear();
}

int Hy4770bRuntime::current_kv_len() const noexcept { return impl_->kv_len; }
const std::vector<int32_t>& Hy4770bRuntime::cached_tokens() const noexcept { return impl_->cached_toks; }

bool Hy4770bRuntime::save_kv_to_nvme(uint64_t prefix_hash) {
    if (!cfg_.enable_kv_disk_cache || impl_->kv_len <= 0) return false;
    const uint32_t slen = uint32_t(impl_->kv_len);
    bool ok = true;

    // Stream one stripe at a time into a single reusable O_DIRECT AlignedBuf (<= 13.2 GiB even at 512K,
    // leaving 336.43 GiB Q5_LM128 DDR4 pages 100% resident with zero page-cache eviction!).
    {
        const size_t max_sb = std::max(Hy4770bNvmeKvCache::stripe_payload_bytes(slen, 0),
                                       Hy4770bNvmeKvCache::stripe_payload_bytes(slen, 1));
        Hy4770bNvmeKvCache::AlignedBuf sbuf(max_sb);
        for (int s = 0; s < 2 && ok; ++s) {
            const int l_beg  = (s == 0) ? 0 : int(kStripe0Layers);
            const int l_end  = (s == 0) ? int(kStripe0Layers) : kNumLayers;
            const int il_beg = (s == 0) ? 0 : int(kStripe0Idx);
            const int il_end = (s == 0) ? int(kStripe0Idx) : kNumFullIndexLayers;

            for (int l = l_beg; l < l_end; ++l) {
                uint8_t* dst_layer = sbuf.ptr + size_t(l - l_beg) * size_t(slen) * kKvRowBytes;
                for (int r = 0; r < kNumMi50Ranks; ++r) {
                    const int ltoks = kv_rank_local_toks(r, int(slen));
                    if (ltoks <= 0) continue;
                    auto& rc = impl_->ranks[r];
                    std::vector<uint8_t> tmp(size_t(ltoks) * kKvRowBytes);
                    check_hip_r(hipSetDevice(r), "kvsave_set");
                    check_hip_r(hipMemcpy(tmp.data(), rc.kv_cache_fp8.ptr + size_t(l) * size_t(rc.local_kv_cap) * kKvRowBytes,
                                          tmp.size(), hipMemcpyDeviceToHost), "kvsave_d2h");
                    for (int lt = 0; lt < ltoks; ++lt) {
                        const int gp = kv_local_tok_to_global(r, lt);
                        if (gp < int(slen)) std::memcpy(dst_layer + size_t(gp) * kKvRowBytes, tmp.data() + size_t(lt) * kKvRowBytes, kKvRowBytes);
                    }
                }
            }
            const size_t base_off = size_t(l_end - l_beg) * size_t(slen) * kKvRowBytes;
            for (int il = il_beg; il < il_end; ++il) {
                uint8_t* dst_idx = sbuf.ptr + base_off + size_t(il - il_beg) * size_t(slen) * kIdxRowBytes;
                for (int r = 0; r < kNumMi50Ranks; ++r) {
                    const int ltoks = kv_rank_local_toks(r, int(slen));
                    if (ltoks <= 0) continue;
                    auto& rc = impl_->ranks[r];
                    std::vector<uint8_t> tmp(size_t(ltoks) * kIdxRowBytes);
                    check_hip_r(hipSetDevice(r), "iksave_set");
                    check_hip_r(hipMemcpy(tmp.data(), rc.idx_k_cache_fp8.ptr + size_t(il) * size_t(rc.local_kv_cap) * kIdxRowBytes,
                                          tmp.size(), hipMemcpyDeviceToHost), "iksave_d2h");
                    for (int lt = 0; lt < ltoks; ++lt) {
                        const int gp = kv_local_tok_to_global(r, lt);
                        if (gp < int(slen)) std::memcpy(dst_idx + size_t(gp) * kIdxRowBytes, tmp.data() + size_t(lt) * kIdxRowBytes, kIdxRowBytes);
                    }
                }
            }
            ok = impl_->kv_disk.save_one_stripe(prefix_hash, slen, s, sbuf.ptr, sbuf.cap);
        }
    }

    if (ok) {
        // Stream lossless BF16 KV tail layers (L70..L77) ONE layer at a time (<= 604 MiB peak!)
        // and immediately drop page cache so multi-GiB sidecar never evicts Q5_LM128 RAM!
        char bpath[512];
        std::snprintf(bpath, sizeof(bpath), "%s/hy4_770b_%016llx_tail_bf16.kvblk",
                      cfg_.kv_dir0.c_str(), static_cast<unsigned long long>(prefix_hash));
        std::ofstream bofs(bpath, std::ios::binary | std::ios::trunc);
        std::vector<uint16_t> bf16_layer(size_t(slen) * size_t(kKvLatentDim));
        for (int bl = 0; bl < kNumBf16KvLayers; ++bl) {
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                const int ltoks = kv_rank_local_toks(r, int(slen));
                if (ltoks <= 0) continue;
                auto& rc = impl_->ranks[r];
                std::vector<uint16_t> tmp(size_t(ltoks) * size_t(kKvLatentDim));
                check_hip_r(hipSetDevice(r), "kvbsave_set");
                check_hip_r(hipMemcpy(tmp.data(), rc.kv_cache_bf16.ptr + size_t(bl) * size_t(rc.local_kv_cap) * size_t(kKvLatentDim),
                                      tmp.size() * 2u, hipMemcpyDeviceToHost), "kvbsave_d2h");
                for (int lt = 0; lt < ltoks; ++lt) {
                    const int gp = kv_local_tok_to_global(r, lt);
                    if (gp < int(slen)) {
                        std::memcpy(bf16_layer.data() + size_t(gp) * size_t(kKvLatentDim),
                                    tmp.data() + size_t(lt) * size_t(kKvLatentDim), size_t(kKvLatentDim) * 2u);
                    }
                }
            }
            bofs.write(reinterpret_cast<const char*>(bf16_layer.data()), std::streamsize(bf16_layer.size() * 2u));
        }
        bofs.flush();
        bofs.close();
        drop_file_page_cache(bpath, true);
    }
    return ok;
}

bool Hy4770bRuntime::load_kv_from_nvme(
    uint64_t prefix_hash, const std::vector<int32_t>& prefix_tokens,
    bool evict_page_cache_first, Hy4770bNvmeKvCache::ReloadStats* out_stats)
{
    if (!cfg_.enable_kv_disk_cache || prefix_tokens.empty()) return false;
    if (evict_page_cache_first) impl_->kv_disk.evict_page_cache(prefix_hash);
    const uint32_t slen = uint32_t(prefix_tokens.size());
    if (int(slen) > cfg_.max_kv_len) return false;

    Hy4770bNvmeKvCache::ReloadStats st{};
    double h2d_ms_accum = 0.0;

    // Precompute local_to_global maps once across 5 MI50 ranks
    std::array<std::vector<int32_t>, kNumMi50Ranks> l2g_all;
    for (int r = 0; r < kNumMi50Ranks; ++r) {
        const int ltoks = kv_rank_local_toks(r, int(slen));
        l2g_all[size_t(r)].resize(size_t(std::max(0, ltoks)));
        for (int lt = 0; lt < ltoks; ++lt) l2g_all[size_t(r)][size_t(lt)] = kv_local_tok_to_global(r, lt);
        if (ltoks > 0) {
            auto& rc = impl_->ranks[r];
            check_hip_r(hipSetDevice(r), "kvload_set");
            check_hip_r(hipMemcpyAsync(rc.local_to_global.ptr, l2g_all[size_t(r)].data(),
                                       l2g_all[size_t(r)].size() * 4u, hipMemcpyHostToDevice, rc.stream), "l2g_h2d");
        }
    }

    auto scatter_stripe_to_gpus = [&](int s, const uint8_t* stripe_ptr) {
        const auto th0 = std::chrono::steady_clock::now();
        const int l_beg  = (s == 0) ? 0 : int(kStripe0Layers);
        const int l_end  = (s == 0) ? int(kStripe0Layers) : kNumLayers;
        const int il_beg = (s == 0) ? 0 : int(kStripe0Idx);
        const int il_end = (s == 0) ? int(kStripe0Idx) : kNumFullIndexLayers;
        const size_t base_off = size_t(l_end - l_beg) * size_t(slen) * kKvRowBytes;

        for (int r = 0; r < kNumMi50Ranks; ++r) {
            const int ltoks = int(l2g_all[size_t(r)].size());
            if (ltoks <= 0) continue;
            auto& rc = impl_->ranks[r];
            const auto& l2g = l2g_all[size_t(r)];
            check_hip_r(hipSetDevice(r), "kvload_s_set");
            std::vector<uint8_t> tmp(size_t(ltoks) * kKvRowBytes);
            for (int l = l_beg; l < l_end; ++l) {
                const uint8_t* src_layer = stripe_ptr + size_t(l - l_beg) * size_t(slen) * kKvRowBytes;
                for (int lt = 0; lt < ltoks; ++lt) {
                    std::memcpy(tmp.data() + size_t(lt) * kKvRowBytes, src_layer + size_t(l2g[size_t(lt)]) * kKvRowBytes, kKvRowBytes);
                }
                check_hip_r(hipMemcpyAsync(rc.kv_cache_fp8.ptr + size_t(l) * size_t(rc.local_kv_cap) * kKvRowBytes,
                                           tmp.data(), tmp.size(), hipMemcpyHostToDevice, rc.stream), "kv_h2d");
            }
            std::vector<uint8_t> itmp(size_t(ltoks) * kIdxRowBytes);
            for (int il = il_beg; il < il_end; ++il) {
                const uint8_t* src_idx = stripe_ptr + base_off + size_t(il - il_beg) * size_t(slen) * kIdxRowBytes;
                for (int lt = 0; lt < ltoks; ++lt) {
                    std::memcpy(itmp.data() + size_t(lt) * kIdxRowBytes, src_idx + size_t(l2g[size_t(lt)]) * kIdxRowBytes, kIdxRowBytes);
                }
                check_hip_r(hipMemcpyAsync(rc.idx_k_cache_fp8.ptr + size_t(il) * size_t(rc.local_kv_cap) * kIdxRowBytes,
                                           itmp.data(), itmp.size(), hipMemcpyHostToDevice, rc.stream), "ik_h2d");
            }
        }
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            check_hip_r(hipSetDevice(r), "kvload_sset");
            check_hip_r(hipStreamSynchronize(impl_->ranks[r].stream), "kvload_sync");
        }
        h2d_ms_accum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th0).count();
    };

    if (slen <= 262144u) {
        // <= 262K: Dual-NVMe parallel O_DIRECT read (<= 13.16 GiB total, well inside free DDR4 headroom!)
        Hy4770bNvmeKvCache::AlignedBuf sbuf[2];
        sbuf[0].alloc(Hy4770bNvmeKvCache::stripe_payload_bytes(slen, 0));
        sbuf[1].alloc(Hy4770bNvmeKvCache::stripe_payload_bytes(slen, 1));
        Hy4770bNvmeKvCache::SnapshotView view{};
        view.stripe_buf[0] = sbuf[0].ptr; view.stripe_cap[0] = sbuf[0].cap;
        view.stripe_buf[1] = sbuf[1].ptr; view.stripe_cap[1] = sbuf[1].cap;
        if (!impl_->kv_disk.load(prefix_hash, view, &st)) return false;
        if (view.seq_len != slen) return false;
        scatter_stripe_to_gpus(0, sbuf[0].ptr);
        scatter_stripe_to_gpus(1, sbuf[1].ptr);
    } else {
        // > 262K (512K): Single reusable 13.2 GiB O_DIRECT buffer across stripe 0 then stripe 1 so
        // peak RAM stays <= 13.2 GiB and 336.43 GiB Q5_LM128 pages experience ZERO eviction!
        const size_t max_sb = std::max(Hy4770bNvmeKvCache::stripe_payload_bytes(slen, 0),
                                       Hy4770bNvmeKvCache::stripe_payload_bytes(slen, 1));
        Hy4770bNvmeKvCache::AlignedBuf sbuf(max_sb);
        for (int s = 0; s < 2; ++s) {
            uint32_t s_len = 0u;
            if (!impl_->kv_disk.load_one_stripe(prefix_hash, s, sbuf.ptr, sbuf.cap, &s_len, &st)) return false;
            if (s_len != slen) return false;
            scatter_stripe_to_gpus(s, sbuf.ptr);
        }
    }

    // Stream lossless BF16 KV tail layers (L70..L77) ONE layer at a time (<= 604 MiB peak!)
    // after sbuf is freed, immediately dropping page cache to protect Q5_LM128 residency!
    {
        const auto thb0 = std::chrono::steady_clock::now();
        char bpath[512];
        std::snprintf(bpath, sizeof(bpath), "%s/hy4_770b_%016llx_tail_bf16.kvblk",
                      cfg_.kv_dir0.c_str(), static_cast<unsigned long long>(prefix_hash));
        if (evict_page_cache_first) drop_file_page_cache(bpath, false);
        std::ifstream bifs(bpath, std::ios::binary);
        std::vector<uint16_t> bf16_layer(size_t(slen) * size_t(kKvLatentDim));
        for (int bl = 0; bl < kNumBf16KvLayers; ++bl) {
            if (!bifs || !bifs.read(reinterpret_cast<char*>(bf16_layer.data()), std::streamsize(bf16_layer.size() * 2u))) {
                std::fill(bf16_layer.begin(), bf16_layer.end(), uint16_t(0));
            }
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                const int ltoks = int(l2g_all[size_t(r)].size());
                if (ltoks <= 0) continue;
                auto& rc = impl_->ranks[r];
                const auto& l2g = l2g_all[size_t(r)];
                check_hip_r(hipSetDevice(r), "kvbload_set");
                std::vector<uint16_t> btmp(size_t(ltoks) * size_t(kKvLatentDim));
                for (int lt = 0; lt < ltoks; ++lt) {
                    std::memcpy(btmp.data() + size_t(lt) * size_t(kKvLatentDim),
                                bf16_layer.data() + size_t(l2g[size_t(lt)]) * size_t(kKvLatentDim), size_t(kKvLatentDim) * 2u);
                }
                check_hip_r(hipMemcpyAsync(rc.kv_cache_bf16.ptr + size_t(bl) * size_t(rc.local_kv_cap) * size_t(kKvLatentDim),
                                           btmp.data(), btmp.size() * 2u, hipMemcpyHostToDevice, rc.stream), "kvb_h2d");
            }
            for (int r = 0; r < kNumMi50Ranks; ++r) {
                check_hip_r(hipSetDevice(r), "kvbload_sset");
                check_hip_r(hipStreamSynchronize(impl_->ranks[r].stream), "kvbload_sync");
            }
        }
        bifs.close();
        drop_file_page_cache(bpath, false);
        h2d_ms_accum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - thb0).count();
    }

    st.h2d_ms = h2d_ms_accum;
    st.total_ms += h2d_ms_accum;
    st.eff_gibs    = (double(st.file_bytes) / 1073741824.0) / std::max(1e-6, st.total_ms / 1000.0);
    st.tok_per_sec = double(slen) / std::max(1e-6, st.total_ms / 1000.0);
    if (out_stats) *out_stats = st;

    impl_->kv_len = int(slen);
    impl_->cached_toks = prefix_tokens;
    return true;
}

void Hy4770bRuntime::tile_kv_for_long_context_probe(int target_kv_len) {
    if (target_kv_len <= impl_->kv_len || target_kv_len > cfg_.max_kv_len || impl_->kv_len < 512) return;
    const int src_len = impl_->kv_len;
    for (int r = 0; r < kNumMi50Ranks; ++r) {
        auto& rc = impl_->ranks[r];
        const int src_lt = kv_rank_local_toks(r, src_len);
        const int dst_lt = kv_rank_local_toks(r, target_kv_len);
        if (src_lt <= 0 || dst_lt <= src_lt) continue;
        check_hip_r(hipSetDevice(r), "tile_set");
        std::vector<int32_t> l2g(static_cast<size_t>(dst_lt));
        for (int lt = 0; lt < dst_lt; ++lt) l2g[size_t(lt)] = kv_local_tok_to_global(r, lt);
        check_hip_r(hipMemcpyAsync(rc.local_to_global.ptr, l2g.data(), l2g.size() * 4u, hipMemcpyHostToDevice, rc.stream), "tl2g");
        for (int l = 0; l < kNumLayers; ++l) {
            uint8_t* base = rc.kv_cache_fp8.ptr + size_t(l) * size_t(rc.local_kv_cap) * kKvRowBytes;
            for (int off = src_lt; off < dst_lt; off += src_lt) {
                const int cnt = std::min(src_lt, dst_lt - off);
                check_hip_r(hipMemcpyAsync(base + size_t(off) * kKvRowBytes, base, size_t(cnt) * kKvRowBytes, hipMemcpyDeviceToDevice, rc.stream), "tkv");
            }
        }
        for (int bl = 0; bl < kNumBf16KvLayers; ++bl) {
            uint16_t* base = rc.kv_cache_bf16.ptr + size_t(bl) * size_t(rc.local_kv_cap) * size_t(kKvLatentDim);
            for (int off = src_lt; off < dst_lt; off += src_lt) {
                const int cnt = std::min(src_lt, dst_lt - off);
                check_hip_r(hipMemcpyAsync(base + size_t(off) * size_t(kKvLatentDim), base, size_t(cnt) * size_t(kKvLatentDim) * 2u, hipMemcpyDeviceToDevice, rc.stream), "tkvb");
            }
        }
        for (int il = 0; il < kNumFullIndexLayers; ++il) {
            uint8_t* base = rc.idx_k_cache_fp8.ptr + size_t(il) * size_t(rc.local_kv_cap) * kIdxRowBytes;
            for (int off = src_lt; off < dst_lt; off += src_lt) {
                const int cnt = std::min(src_lt, dst_lt - off);
                check_hip_r(hipMemcpyAsync(base + size_t(off) * kIdxRowBytes, base, size_t(cnt) * kIdxRowBytes, hipMemcpyDeviceToDevice, rc.stream), "tik");
            }
        }
        check_hip_r(hipStreamSynchronize(rc.stream), "tile_sync");
    }
    impl_->kv_len = target_kv_len;
    impl_->cached_toks.resize(static_cast<size_t>(target_kv_len), 0);
}

void Hy4770bRuntime::set_prefill_chunk_size(int chunk_sz) {
    cfg_.prefill_chunk_size = std::clamp(chunk_sz, 1, 512);
    impl_->cfg.prefill_chunk_size = cfg_.prefill_chunk_size;
}
void Hy4770bRuntime::set_diag_sink(std::vector<Hy4770bLayerDiag>* sink, bool verify_p150_cpu) {
    impl_->diag_sink = sink;
    impl_->diag_verify_p150 = verify_p150_cpu;
}

void Hy4770bRuntime::forward_tokens(
    const std::vector<int32_t>& token_ids,
    int pos_start,
    std::vector<float>& out_last_logits_f32,
    std::vector<float>* out_all_logits_f32,
    std::vector<uint16_t>* out_last_hidden_bf16,
    const std::vector<Hy4770bVisualSpan>* visual_spans)
{
    std::lock_guard<std::mutex> lk(impl_->exec_mu);
    impl_->last_activity_ms.store(impl_->now_ms(), std::memory_order_relaxed);
    const int M_tot = int(token_ids.size());
    if (M_tot <= 0) return;

    if (out_all_logits_f32) out_all_logits_f32->resize(size_t(M_tot) * size_t(kVocabSize));

    // Process in chunks of <= prefill_chunk_size (1..512; cap to 64 above 128K ctx to bound indexer scratch VRAM!)
    const int base_chunk = std::clamp(cfg_.prefill_chunk_size, 1, 512);
    for (int off = 0; off < M_tot; ) {
        if (should_stop()) return;
        const int cur_pos = pos_start + off;
        const int kChunk = (cur_pos > 131072) ? std::min(base_chunk, 64) : base_chunk;
        const int M = std::min(kChunk, M_tot - off);

        // 1. Expand 4-stream iHC initial hidden states [M, 4, 6144] on host from DDR4 embed_tokens (+ ViT spans!)
        std::vector<uint16_t> h_streams(size_t(M) * size_t(kHcFlatDim));
        for (int m = 0; m < M; ++m) {
            const int32_t tid = std::clamp(token_ids[size_t(off + m)], 0, kVocabSize - 1);
            const uint16_t* row = impl_->embed_host_bf16.data() + size_t(tid) * size_t(kHiddenSize);
            if (visual_spans) {
                const int gp = cur_pos + m;
                for (const auto& vs : *visual_spans) {
                    if (gp >= vs.start_pos && gp < vs.start_pos + vs.num_tokens && vs.emb_bf16) {
                        row = vs.emb_bf16 + size_t(gp - vs.start_pos) * size_t(kHiddenSize);
                        break;
                    }
                }
            }
            uint16_t* dst = h_streams.data() + size_t(m) * size_t(kHcFlatDim);
            for (int c = 0; c < kHcMult; ++c) std::memcpy(dst + size_t(c) * size_t(kHiddenSize), row, size_t(kHiddenSize) * 2u);
        }

        auto& rc0 = impl_->ranks[0];
        check_hip_r(hipSetDevice(0), "fwd_set0");
        impl_->r0_streams.alloc(0, h_streams.size());
        impl_->r0_sub_in.alloc(0, size_t(M) * size_t(kHiddenSize));
        impl_->r0_x_norm.alloc(0, size_t(M) * size_t(kHiddenSize));
        impl_->r0_post.alloc(0, size_t(M) * size_t(kHcMult));
        check_hip_r(hipMemcpyAsync(impl_->r0_streams.ptr, h_streams.data(), h_streams.size() * 2u, hipMemcpyHostToDevice, rc0.stream), "st_h2d");

        // 2. Run all 78 Backbone Layers!
        double f_ihc_dense_ms = 0.0, f_attn_ms = 0.0, f_lm_ms = 0.0;
        for (int l = 0; l < kNumLayers; ++l) {
            if (should_stop()) return;
            const auto t_l0 = std::chrono::steady_clock::now();
            auto& r0e = impl_->r0_layers[size_t(l)];
            check_hip_r(hipSetDevice(0), "l_set0");
            // Attention iHC pre + RMSNorm
            launch_ihc_pre(impl_->r0_streams.ptr, r0e.hc_attn_fn.ptr, r0e.hc_attn_base.ptr, r0e.hc_attn_scale.ptr,
                           impl_->r0_sub_in.ptr, impl_->r0_post.ptr, M, kRmsNormEps, kHcEps, kHcMagnitude, rc0.stream);
            launch_rmsnorm_bf16(impl_->r0_sub_in.ptr, r0e.input_ln.ptr, impl_->r0_x_norm.ptr, M, kHiddenSize, kRmsNormEps, rc0.stream);
            const auto t_a0 = std::chrono::steady_clock::now();
            f_ihc_dense_ms += std::chrono::duration<double, std::milli>(t_a0 - t_l0).count();

            // TP5 Gated Absorbed MLA + 3:3:3:1:1 KV + 21-Full/57-Shared IndexCache
            impl_->run_attention_tp5(l, l, false, 0, M, cur_pos, impl_->r0_x_norm.ptr, impl_->r0_sub_in.ptr);
            const auto t_a1 = std::chrono::steady_clock::now();
            f_attn_ms += std::chrono::duration<double, std::milli>(t_a1 - t_a0).count();

            // Attention iHC post-add + MLP iHC pre + post_attention_layernorm
            check_hip_r(hipSetDevice(0), "lp_set0");
            launch_ihc_post_add(impl_->r0_streams.ptr, impl_->r0_sub_in.ptr, impl_->r0_post.ptr, M, kHiddenSize, kHcMult, rc0.stream);
            launch_ihc_pre(impl_->r0_streams.ptr, r0e.hc_mlp_fn.ptr, r0e.hc_mlp_base.ptr, r0e.hc_mlp_scale.ptr,
                           impl_->r0_sub_in.ptr, impl_->r0_post.ptr, M, kRmsNormEps, kHcEps, kHcMagnitude, rc0.stream);
            launch_rmsnorm_bf16(impl_->r0_sub_in.ptr, r0e.post_ln.ptr, impl_->r0_x_norm.ptr, M, kHiddenSize, kRmsNormEps, rc0.stream);

            if (impl_->diag_sink) {
                impl_->cur_layer_diag = Hy4770bLayerDiag{};
                impl_->cur_layer_diag.layer = l;
                impl_->cur_layer_diag.M = M;
            }
            const auto t_m0 = std::chrono::steady_clock::now();
            f_ihc_dense_ms += std::chrono::duration<double, std::milli>(t_m0 - t_a1).count();
            // L0 Dense TP5 MLP OR L1..77 Simultaneous 3-Tier MoE
            if (l == 0) {
                impl_->run_dense_mlp_tp5(false, 0, M, impl_->r0_x_norm.ptr, impl_->r0_sub_in.ptr);
                f_ihc_dense_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m0).count();
            } else {
                impl_->run_moe_3tier(l, M, impl_->r0_x_norm.ptr, impl_->r0_sub_in.ptr);
            }

            const auto t_p0 = std::chrono::steady_clock::now();
            check_hip_r(hipSetDevice(0), "lmp_set0");
            launch_ihc_post_add(impl_->r0_streams.ptr, impl_->r0_sub_in.ptr, impl_->r0_post.ptr, M, kHiddenSize, kHcMult, rc0.stream);
            f_ihc_dense_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_p0).count();

            if (impl_->diag_sink && (off + M == M_tot)) {
                std::vector<uint16_t> st_host(size_t(M) * size_t(kHcFlatDim));
                check_hip_r(hipMemcpyAsync(st_host.data(), impl_->r0_streams.ptr, st_host.size() * 2u, hipMemcpyDeviceToHost, rc0.stream), "diag_st");
                check_hip_r(hipStreamSynchronize(rc0.stream), "diag_sync");
                double sq = 0.0; float mx = 0.0f;
                for (uint16_t b : st_host) {
                    const float v = bf16_to_f32(b);
                    sq += double(v) * double(v);
                    mx = std::max(mx, std::fabs(v));
                }
                impl_->cur_layer_diag.h_rms = float(std::sqrt(sq / double(st_host.size())));
                impl_->cur_layer_diag.h_max_abs = mx;
                impl_->cur_layer_diag.last_tok_h_stream0.resize(kHiddenSize);
                const uint16_t* s0 = st_host.data() + size_t(M - 1) * size_t(kHcFlatDim);
                for (int d = 0; d < kHiddenSize; ++d) impl_->cur_layer_diag.last_tok_h_stream0[size_t(d)] = bf16_to_f32(s0[d]);
                impl_->diag_sink->push_back(std::move(impl_->cur_layer_diag));
            }
        }

        // 3. Final iHC Head collapse + model.norm + TP5 LM Head
        const auto t_lm0 = std::chrono::steady_clock::now();
        const bool need_logits = (out_all_logits_f32 != nullptr) || (off + M == M_tot);
        if (need_logits || (out_last_hidden_bf16 != nullptr && off + M == M_tot)) {
            check_hip_r(hipSetDevice(0), "fin_set0");
            impl_->r0_final_h.alloc(0, size_t(M) * size_t(kHiddenSize));
            launch_ihc_head_and_norm(impl_->r0_streams.ptr, impl_->r0_hc_head_fn.ptr, impl_->r0_hc_head_base.ptr,
                                     impl_->r0_hc_head_scale.ptr, impl_->r0_model_norm.ptr,
                                     impl_->r0_final_h.ptr, M, kRmsNormEps, kHcEps, rc0.stream);
            if (out_last_hidden_bf16 && off + M == M_tot) {
                const bool ret_all_h = (out_all_logits_f32 != nullptr && M_tot <= kDflyBlockSize);
                const size_t n_rows  = ret_all_h ? size_t(M) : 1u;
                const size_t r_off   = ret_all_h ? 0u : size_t(M - 1);
                out_last_hidden_bf16->resize(n_rows * size_t(kHiddenSize));
                check_hip_r(hipMemcpyAsync(out_last_hidden_bf16->data(),
                                           impl_->r0_final_h.ptr + r_off * size_t(kHiddenSize),
                                           n_rows * size_t(kHiddenSize) * 2u, hipMemcpyDeviceToHost, rc0.stream), "lh_d2h");
            }
            check_hip_r(hipStreamSynchronize(rc0.stream), "fin_sync");

            if (out_all_logits_f32) {
                std::vector<float> chunk_log;
                impl_->compute_lm_head_tp5(impl_->r0_final_h.ptr, M, chunk_log);
                std::memcpy(out_all_logits_f32->data() + size_t(off) * size_t(kVocabSize),
                            chunk_log.data(), chunk_log.size() * 4u);
                if (off + M == M_tot) {
                    out_last_logits_f32.assign(chunk_log.begin() + size_t(M - 1) * size_t(kVocabSize), chunk_log.end());
                }
            } else if (off + M == M_tot) {
                impl_->compute_lm_head_tp5(impl_->r0_final_h.ptr + size_t(M - 1) * size_t(kHiddenSize), 1, out_last_logits_f32);
            }
        }
        f_lm_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_lm0).count();
        {
            std::lock_guard<std::mutex> lk(impl_->stats_mu);
            impl_->prof.forward_calls++;
            impl_->prof.ihc_dense_ms += f_ihc_dense_ms;
            impl_->prof.attn_ms      += f_attn_ms;
            impl_->prof.lm_head_ms   += f_lm_ms;
        }
        impl_->kv_len = std::max(impl_->kv_len, cur_pos + M);
        off += M;
    }
}

// Built-in 3.3B 5-Layer DFly2 Parallel Speculative Drafter (KV-Share & Index-Share, configurable 1..5 drafts!)
void Hy4770bRuntime::dfly2_draft_block(
    int32_t anchor_token_id,
    int anchor_pos,
    const uint16_t* prev_target_hidden_bf16,
    std::array<int32_t, kDflyNumSpecTokens>& out_draft_ids,
    std::array<float, kDflyNumSpecTokens>& out_confidences,
    int num_drafts)
{
    std::lock_guard<std::mutex> lk(impl_->exec_mu);
    out_draft_ids.fill(kEosTokenId);
    out_confidences.fill(0.0f);
    if (!impl_->dfly2_loaded) return;

    const int K  = std::clamp(num_drafts, 1, kDflyNumSpecTokens);
    const int kM = 1 + K;  // 2..6 tokens: [anchor, MASK x K]
    std::vector<uint16_t> h_emb(size_t(kM) * size_t(kHiddenSize));
    const int32_t aid = std::clamp(anchor_token_id, 0, kVocabSize - 1);
    const int32_t mid = std::clamp(kDflyMaskTokenId, 0, kVocabSize - 1);
    std::memcpy(h_emb.data(), impl_->embed_host_bf16.data() + size_t(aid) * size_t(kHiddenSize), size_t(kHiddenSize) * 2u);
    for (int i = 1; i < kM; ++i) {
        std::memcpy(h_emb.data() + size_t(i) * size_t(kHiddenSize),
                    impl_->embed_host_bf16.data() + size_t(mid) * size_t(kHiddenSize), size_t(kHiddenSize) * 2u);
    }

    auto& rc0 = impl_->ranks[0];
    check_hip_r(hipSetDevice(0), "df_set0");
    auto& d_emb = impl_->dfly_d_emb;
    auto& d_tgt = impl_->dfly_d_tgt;
    auto& d_cat = impl_->dfly_d_cat;
    auto& d_h   = impl_->dfly_d_h;
    auto& d_sub = impl_->dfly_d_sub;
    auto& d_xnorm = impl_->dfly_d_xnorm;
    auto& d_fused_f32 = impl_->dfly_d_fused_f32;
    d_emb.alloc(0, h_emb.size());
    d_tgt.alloc(0, size_t(kHiddenSize));
    d_cat.alloc(0, 2u * size_t(kHiddenSize));
    d_h.alloc(0, h_emb.size());
    d_sub.alloc(0, h_emb.size());
    d_xnorm.alloc(0, h_emb.size());
    d_fused_f32.alloc(0, size_t(kHiddenSize));
    check_hip_r(hipMemcpyAsync(d_emb.ptr, h_emb.data(), h_emb.size() * 2u, hipMemcpyHostToDevice, rc0.stream), "df_e");
    check_hip_r(hipMemcpyAsync(d_h.ptr,   d_emb.ptr,    h_emb.size() * 2u, hipMemcpyDeviceToDevice, rc0.stream), "df_h0");

    // 1. HYV4DFlyFusion ("dfly_mtp_fusion": "slot0"): ONLY slot 0 gets emb[0] + gate * eh_proj(cat(enorm(emb[0]), hnorm(tgt)))!
    //    Slots 1..5 remain pure MASK embeddings!
    if (prev_target_hidden_bf16 != nullptr) {
        check_hip_r(hipMemcpyAsync(d_tgt.ptr, prev_target_hidden_bf16, size_t(kHiddenSize) * 2u, hipMemcpyHostToDevice, rc0.stream), "df_t");
        launch_dfly2_dual_rmsnorm_concat_bf16(d_emb.ptr, impl_->dfly_enorm.ptr, d_tgt.ptr, impl_->dfly_hnorm.ptr,
                                              d_cat.ptr, 1, kHiddenSize, kRmsNormEps, rc0.stream);
        impl_->f16_linear_bf16_in(rc0, impl_->dfly_eh_proj_f16.ptr, d_cat.ptr, d_fused_f32.ptr, 1, kHiddenSize, 2 * kHiddenSize);
        launch_f32_to_bf16(d_fused_f32.ptr, d_sub.ptr, kHiddenSize, rc0.stream);
        launch_dfly2_gate_residual_bf16(d_emb.ptr, impl_->dfly_gate.ptr, d_sub.ptr, d_h.ptr, 1, kHiddenSize, rc0.stream);
    }

    // 2. Run all 5 DFly2 Transformer Layers reusing target KV layers {61,65,69,73,77} + causal draft block KV + IndexCache!
    for (int dl = 0; dl < kDflyNumLayers; ++dl) {
        check_hip_r(hipSetDevice(0), "dfl_set0");
        auto& df0 = rc0.dfly[size_t(dl)];
        launch_rmsnorm_bf16(d_h.ptr, df0.input_ln.ptr, d_xnorm.ptr, kM, kHiddenSize, kRmsNormEps, rc0.stream);
        impl_->run_attention_tp5(dl, kDflyTargetKvLayers[dl], true, dl, kM, anchor_pos, d_xnorm.ptr, d_sub.ptr);
        check_hip_r(hipSetDevice(0), "dfl_add1");
        launch_add_bf16(d_h.ptr, d_sub.ptr, d_h.ptr, kM * kHiddenSize, rc0.stream);
        launch_rmsnorm_bf16(d_h.ptr, df0.post_ln.ptr, d_xnorm.ptr, kM, kHiddenSize, kRmsNormEps, rc0.stream);
        impl_->run_dense_mlp_tp5(true, dl, kM, d_xnorm.ptr, d_sub.ptr);
        check_hip_r(hipSetDevice(0), "dfl_add2");
        launch_add_bf16(d_h.ptr, d_sub.ptr, d_h.ptr, kM * kHiddenSize, rc0.stream);
    }

    // 3. Final norm on all 6 slots
    check_hip_r(hipSetDevice(0), "dff_set0");
    launch_rmsnorm_bf16(d_h.ptr, impl_->dfly_final_norm.ptr, d_xnorm.ptr, kM, kHiddenSize, kRmsNormEps, rc0.stream);

    // 4. Autoregressive HiddenCorrection + TP5 LM Head for draft slots s = 1..5 ("dfly_enable_hidden_correction": true)!
    auto& d_prev_emb = impl_->dfly_d_prev_emb;
    auto& d_hc_g     = impl_->dfly_d_hc_g;
    auto& d_hc_u     = impl_->dfly_d_hc_u;
    auto& d_hc_d     = impl_->dfly_d_hc_d;
    d_prev_emb.alloc(0, size_t(kHiddenSize));
    d_hc_g.alloc(0, size_t(kDflyHcInterSize));
    d_hc_u.alloc(0, size_t(kDflyHcInterSize));
    d_hc_d.alloc(0, size_t(kHiddenSize));

    int32_t prev_tok = aid;
    const bool has_hc = (impl_->dfly_hc_down.w.ptr != nullptr);
    std::vector<float> slot_logits;
    for (int s = 1; s < kM; ++s) {
        check_hip_r(hipSetDevice(0), "hc_set0");
        uint16_t* h_slot = d_xnorm.ptr + size_t(s) * size_t(kHiddenSize);
        if (has_hc) {
            const int32_t pt = std::clamp(prev_tok, 0, kVocabSize - 1);
            check_hip_r(hipMemcpyAsync(d_prev_emb.ptr,
                                       impl_->embed_host_bf16.data() + size_t(pt) * size_t(kHiddenSize),
                                       size_t(kHiddenSize) * 2u, hipMemcpyHostToDevice, rc0.stream), "hc_emb");
            launch_dfly2_dual_rmsnorm_concat_bf16(h_slot, impl_->dfly_hc_hnorm.ptr,
                                                  d_prev_emb.ptr, impl_->dfly_hc_enorm.ptr,
                                                  d_cat.ptr, 1, kHiddenSize, kRmsNormEps, rc0.stream);
            impl_->fp8_linear(rc0, impl_->dfly_hc_gate, d_cat.ptr, d_hc_g.ptr, 1);
            impl_->fp8_linear(rc0, impl_->dfly_hc_up,   d_cat.ptr, d_hc_u.ptr, 1);
            launch_swiglu_unclamped_bf16(d_hc_g.ptr, d_hc_u.ptr, d_hc_g.ptr, kDflyHcInterSize, rc0.stream);
            impl_->fp8_linear(rc0, impl_->dfly_hc_down, d_hc_g.ptr, d_hc_d.ptr, 1);
            launch_add_bf16(h_slot, d_hc_d.ptr, h_slot, kHiddenSize, rc0.stream);
        }
        check_hip_r(hipStreamSynchronize(rc0.stream), "hc_sync");
        impl_->compute_lm_head_tp5(h_slot, 1, slot_logits);
        int32_t best = 0;
        float mx = slot_logits[0];
        for (int v = 1; v < kVocabSize; ++v) if (slot_logits[size_t(v)] > mx) { mx = slot_logits[size_t(v)]; best = v; }
        out_draft_ids[size_t(s - 1)] = best;
        double z = 0.0;
        for (int v = 0; v < kVocabSize; v += 8) z += std::exp(double(slot_logits[size_t(v)] - mx));
        out_confidences[size_t(s - 1)] = float(1.0 / std::max(1.0, z * 8.0));
        prev_tok = best;
    }
}

bool Hy4770bRuntime::vit_ready() const noexcept { return impl_->vit.ready(); }
Hy4VitOutput Hy4770bRuntime::encode_image(const Hy4ImageRgb& img) const { return impl_->vit.encode_image(img); }

// Full End-to-End Generation with Prefix Caching, DFly2 M=6 Verify, common &= ~3 Rollback & Cycle Breaker!
std::vector<int32_t> Hy4770bRuntime::generate(
    const std::vector<int32_t>& prompt_ids,
    const Hy4770bSamplingParams& params,
    Hy4770bGenMetrics* out_metrics,
    const Hy4770bTokenCallback& on_token,
    const std::vector<Hy4770bVisualSpan>* visual_spans)
{
    Hy4770bGenMetrics met{};
    met.prompt_tokens = int(prompt_ids.size());
    const auto t_req0 = std::chrono::steady_clock::now();
    const Hy4770bTierStats ts_before = cumulative_tier_stats();

    // 1. Prefix cache check (with common &= ~3 alignment for clean rollback safety!)
    int common = 0;
    if (!visual_spans || visual_spans->empty()) {
        const int lim = std::min<int>(int(prompt_ids.size()), int(impl_->cached_toks.size()));
        while (common < lim && prompt_ids[size_t(common)] == impl_->cached_toks[size_t(common)]) ++common;
        if (common == int(prompt_ids.size()) && common > 0) --common;  // Always evaluate at least 1 last token!
        common &= ~3;
    }
    impl_->kv_len = common;
    impl_->cached_toks.resize(static_cast<size_t>(common));
    met.cached_prefix_tokens = common;

    std::vector<int32_t> suffix(prompt_ids.begin() + common, prompt_ids.end());
    std::vector<float> logits;
    std::vector<uint16_t> last_h;

    const auto t_pf0 = std::chrono::steady_clock::now();
    forward_tokens(suffix, common, logits, nullptr, &last_h, visual_spans);
    const auto t_pf1 = std::chrono::steady_clock::now();
    met.prefill_ms = std::chrono::duration<double, std::milli>(t_pf1 - t_pf0).count();
    met.ttft_ms    = std::chrono::duration<double, std::milli>(t_pf1 - t_req0).count();
    met.prefill_tok_s = double(suffix.size()) / std::max(1e-6, met.prefill_ms / 1000.0);
    impl_->cached_toks = prompt_ids;

    std::mt19937_64 rng(params.seed);
    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(params.max_new_tokens));

    auto is_stop = [&](int32_t t) -> bool {
        for (int32_t st : params.stop_token_ids) if (t == st) return true;
        return false;
    };

    auto sample_with_cycle_guard = [&](const float* lgi) -> int32_t {
        const int ban = hy4::detect_local_ngram_cycle_ban_tok(generated);
        if (ban < 0 || ban >= kVocabSize) return sample_logits_host(lgi, kVocabSize, params, generated, rng);
        std::vector<float> tmp(lgi, lgi + kVocabSize);
        tmp[size_t(ban)] = -1e30f;
        return sample_logits_host(tmp.data(), kVocabSize, params, generated, rng);
    };

    // Emit first token from prefill logits
    int32_t cur_tok = sample_with_cycle_guard(logits.data());
    if (is_stop(cur_tok)) {
        met.finish_reason = "stop";
        if (out_metrics) *out_metrics = met;
        return generated;
    }
    generated.push_back(cur_tok);
    if (on_token && !on_token(cur_tok, 0)) {
        met.generated_tokens = 1;
        met.finish_reason = "cancelled";
        if (out_metrics) *out_metrics = met;
        return generated;
    }

    const auto t_dec0 = std::chrono::steady_clock::now();
    const int n_drafts = (params.dfly2_num_drafts > 0) ? std::clamp(params.dfly2_num_drafts, 1, kDflyNumSpecTokens)
                                                       : std::clamp(cfg_.dfly2_num_drafts, 0, kDflyNumSpecTokens);
    const bool use_dfly = params.use_dfly2 && impl_->dfly2_loaded && (n_drafts > 0) && params.max_new_tokens > 1;
    double sum_verify_ms = 0.0, sum_draft_ms = 0.0;

    while (int(generated.size()) < params.max_new_tokens && !should_stop()) {
        if (impl_->kv_len + 8 >= cfg_.max_kv_len) {
            met.finish_reason = "length";
            break;
        }
        const int pos = impl_->kv_len;

        if (use_dfly && int(generated.size()) + 1 < params.max_new_tokens) {
            // DFly2 Draft (K = 1..5 tokens) + Target Parallel M = 1+K Verify!
            const int cur_drafts = std::min(n_drafts, params.max_new_tokens - int(generated.size()) - 1);
            const auto td0 = std::chrono::steady_clock::now();
            std::array<int32_t, kDflyNumSpecTokens> drafts{};
            std::array<float,   kDflyNumSpecTokens> confs{};
            dfly2_draft_block(cur_tok, pos, last_h.data(), drafts, confs, cur_drafts);
            const double d_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
            sum_draft_ms += d_ms;

            std::vector<int32_t> v_in(size_t(1 + cur_drafts));
            v_in[0] = cur_tok;
            for (int i = 0; i < cur_drafts; ++i) v_in[size_t(1 + i)] = drafts[size_t(i)];

            const auto tv0 = std::chrono::steady_clock::now();
            std::vector<float> all_lg;
            std::vector<uint16_t> all_h;
            forward_tokens(v_in, pos, logits, &all_lg, &all_h, nullptr);
            const double v_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv0).count();
            sum_verify_ms += v_ms;
            met.step_ms.push_back(d_ms + v_ms);
            met.dfly2_steps++;
            met.dfly2_proposed += cur_drafts;

            int accepted = 0;
            bool stopped = false;
            int32_t next_anchor = -1;
            for (int i = 0; i < cur_drafts; ++i) {
                const float* lgi = all_lg.data() + size_t(i) * size_t(kVocabSize);
                const int32_t tgt = sample_with_cycle_guard(lgi);
                if (tgt == drafts[size_t(i)]) {
                    ++accepted;
                    impl_->cached_toks.push_back(v_in[size_t(i)]);
                    if (is_stop(tgt)) { met.finish_reason = "stop"; stopped = true; break; }
                    generated.push_back(tgt);
                    if (on_token && !on_token(tgt, int(generated.size()) - 1)) { met.finish_reason = "cancelled"; stopped = true; break; }
                    if (int(generated.size()) >= params.max_new_tokens) { met.finish_reason = "length"; stopped = true; break; }
                } else {
                    impl_->cached_toks.push_back(v_in[size_t(i)]);
                    next_anchor = tgt;
                    break;
                }
            }
            met.dfly2_accepted += accepted;
            // Exact rollback: select exact accepted slot's target hidden state & IndexCache slot!
            const int pick_slot = std::min(accepted, cur_drafts);
            if (all_h.size() >= size_t(pick_slot + 1) * size_t(kHiddenSize)) {
                last_h.assign(all_h.begin() + size_t(pick_slot) * size_t(kHiddenSize),
                              all_h.begin() + size_t(pick_slot + 1) * size_t(kHiddenSize));
            }
            impl_->dfly_active_slot = pick_slot;
            if (stopped) {
                impl_->kv_len = int(impl_->cached_toks.size());
                break;
            }
            if (next_anchor < 0) {
                impl_->cached_toks.push_back(v_in[size_t(cur_drafts)]);
                next_anchor = sample_with_cycle_guard(all_lg.data() + size_t(cur_drafts) * size_t(kVocabSize));
                impl_->kv_len = int(impl_->cached_toks.size());
            } else {
                impl_->kv_len = int(impl_->cached_toks.size());
            }
            if (is_stop(next_anchor)) { met.finish_reason = "stop"; break; }
            generated.push_back(next_anchor);
            cur_tok = next_anchor;
            if (on_token && !on_token(cur_tok, int(generated.size()) - 1)) { met.finish_reason = "cancelled"; break; }
        } else {
            const auto ts0 = std::chrono::steady_clock::now();
            forward_tokens({cur_tok}, pos, logits, nullptr, &last_h, nullptr);
            const double s_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts0).count();
            met.step_ms.push_back(s_ms);
            impl_->cached_toks.push_back(cur_tok);

            const int32_t nxt = sample_with_cycle_guard(logits.data());
            if (is_stop(nxt)) { met.finish_reason = "stop"; break; }
            generated.push_back(nxt);
            cur_tok = nxt;
            if (on_token && !on_token(cur_tok, int(generated.size()) - 1)) { met.finish_reason = "cancelled"; break; }
        }
    }

    const auto t_dec1 = std::chrono::steady_clock::now();
    met.generated_tokens = int(generated.size());
    if (met.generated_tokens >= params.max_new_tokens && met.finish_reason == "stop") met.finish_reason = "length";
    met.decode_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();
    if (met.generated_tokens > 1 && met.decode_ms > 0.0) {
        met.decode_tok_s = double(met.generated_tokens - 1) / (met.decode_ms / 1000.0);
    }
    if (met.dfly2_steps > 0) {
        met.dfly2_accept_rate          = double(met.dfly2_accepted) / double(std::max(1, met.dfly2_proposed));
        met.dfly2_mean_tokens_per_step = double(met.dfly2_accepted + met.dfly2_steps) / double(met.dfly2_steps);
        met.dfly2_verify_step_ms_mean  = sum_verify_ms / double(met.dfly2_steps);
        met.dfly2_draft_ms_mean        = sum_draft_ms  / double(met.dfly2_steps);
    }

    const Hy4770bTierStats ts_after = cumulative_tier_stats();
    met.tier_delta.total_routes    = ts_after.total_routes    - ts_before.total_routes;
    met.tier_delta.tensix_l1_hits  = ts_after.tensix_l1_hits  - ts_before.tensix_l1_hits;
    met.tier_delta.mi50_hbm2_hits  = ts_after.mi50_hbm2_hits  - ts_before.mi50_hbm2_hits;
    met.tier_delta.p150_gddr6_hits = ts_after.p150_gddr6_hits - ts_before.p150_gddr6_hits;
    met.tier_delta.epyc_ddr4_hits  = ts_after.epyc_ddr4_hits  - ts_before.epyc_ddr4_hits;
    met.tier_delta.mi50_ms_total   = ts_after.mi50_ms_total   - ts_before.mi50_ms_total;
    met.tier_delta.p150_ms_total   = ts_after.p150_ms_total   - ts_before.p150_ms_total;
    met.tier_delta.ddr4_ms_total   = ts_after.ddr4_ms_total   - ts_before.ddr4_ms_total;

    if (out_metrics) *out_metrics = met;
    return generated;
}

// Continuous-Refill Lockstep B<=32 Batched Greedy Generation in Disjoint 2K KV Partitions (100% Bit-Exact vs B=1!)
std::vector<std::vector<int32_t>> Hy4770bRuntime::generate_batch_greedy(
    const std::vector<std::vector<int32_t>>& prompts,
    int max_new_tokens,
    const std::vector<int32_t>& stop_token_ids,
    double* out_decode_tok_s,
    int max_concurrent,
    const Hy4770bBatchDoneCallback& on_done)
{
    const int B_tot = int(prompts.size());
    std::vector<std::vector<int32_t>> results(static_cast<size_t>(B_tot));
    if (B_tot == 0 || max_new_tokens <= 0) {
        if (out_decode_tok_s) *out_decode_tok_s = 0.0;
        return results;
    }

    constexpr int kSlotStride = kIndexTopK;  // 2048 -> dense sequence-isolated attention == exact B=1!
    std::vector<int> eligible;
    for (int i = 0; i < B_tot; ++i) {
        if (prompts[size_t(i)].empty()) continue;
        if (int(prompts[size_t(i)].size()) + max_new_tokens <= kSlotStride) {
            eligible.push_back(i);
        } else {
            Hy4770bSamplingParams sp{};
            sp.greedy = true; sp.temperature = 0.0f; sp.max_new_tokens = max_new_tokens;
            sp.use_dfly2 = false; sp.stop_token_ids = stop_token_ids;
            clear_kv_cache();
            Hy4770bGenMetrics m{};
            results[size_t(i)] = generate(prompts[size_t(i)], sp, &m);
            if (on_done) on_done(i, results[size_t(i)], m.decode_tok_s);
        }
    }
    if (eligible.empty()) {
        if (out_decode_tok_s) *out_decode_tok_s = 0.0;
        return results;
    }

    auto is_stop = [&](int32_t t) -> bool {
        for (int32_t st : stop_token_ids) if (t == st) return true;
        return false;
    };
    auto argmax_with_cycle_guard = [&](const float* lgi, const std::vector<int32_t>& gen) -> int32_t {
        const int ban = hy4::detect_local_ngram_cycle_ban_tok(gen);
        int32_t best = 0;
        float mx = -1e30f;
        for (int v = 0; v < kVocabSize; ++v) {
            if (v == ban) continue;
            if (lgi[v] > mx) { mx = lgi[v]; best = v; }
        }
        return best;
    };

    const int max_kv_slots_cap = std::max(1, cfg_.max_kv_len / kSlotStride);
    const int n_slots = std::clamp(max_concurrent, 1, std::min({32, max_kv_slots_cap, int(eligible.size())}));
    const int total_kv_slots = n_slots * kSlotStride;

    std::lock_guard<std::mutex> lk(impl_->exec_mu);
    impl_->last_activity_ms.store(impl_->now_ms(), std::memory_order_relaxed);
    impl_->kv_len = 0;
    impl_->cached_toks.clear();

    auto clear_slot_l2g = [&](const std::vector<int>& slots_to_clr) {
        if (slots_to_clr.empty()) return;
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            check_hip_r(hipSetDevice(r), "bg_clr_set");
            for (int s : slots_to_clr) {
                const int lt_beg = kv_rank_local_toks(r, s * kSlotStride);
                const int lt_end = kv_rank_local_toks(r, (s + 1) * kSlotStride);
                if (lt_end > lt_beg) {
                    check_hip_r(hipMemsetAsync(impl_->ranks[r].local_to_global.ptr + lt_beg,
                                               0xFF, size_t(lt_end - lt_beg) * 4u, impl_->ranks[r].stream), "bg_clr");
                }
            }
        }
        for (int r = 0; r < kNumMi50Ranks; ++r) {
            check_hip_r(hipSetDevice(r), "bg_clrs_set");
            check_hip_r(hipStreamSynchronize(impl_->ranks[r].stream), "bg_clrs");
        }
    };

    // Helper: run one packed chunk through all 78 layers + collapse final hidden on R0
    auto run_batched_chunk = [&](const int32_t* tids, const int32_t* rpos, const int32_t* kslot,
                                 const int32_t* mslot, int M, int max_kv_active) {
        std::vector<uint16_t> h_streams(size_t(M) * size_t(kHcFlatDim));
        for (int m = 0; m < M; ++m) {
            const int32_t tid = std::clamp(tids[m], 0, kVocabSize - 1);
            const uint16_t* row = impl_->embed_host_bf16.data() + size_t(tid) * size_t(kHiddenSize);
            uint16_t* dst = h_streams.data() + size_t(m) * size_t(kHcFlatDim);
            for (int c = 0; c < kHcMult; ++c) std::memcpy(dst + size_t(c) * size_t(kHiddenSize), row, size_t(kHiddenSize) * 2u);
        }
        auto& rc0 = impl_->ranks[0];
        check_hip_r(hipSetDevice(0), "bg_set0");
        impl_->r0_streams.alloc(0, h_streams.size());
        impl_->r0_sub_in.alloc(0, size_t(M) * size_t(kHiddenSize));
        impl_->r0_x_norm.alloc(0, size_t(M) * size_t(kHiddenSize));
        impl_->r0_post.alloc(0, size_t(M) * size_t(kHcMult));
        check_hip_r(hipMemcpyAsync(impl_->r0_streams.ptr, h_streams.data(), h_streams.size() * 2u, hipMemcpyHostToDevice, rc0.stream), "bg_h2d");

        for (int l = 0; l < kNumLayers; ++l) {
            if (should_stop()) return;
            auto& r0e = impl_->r0_layers[size_t(l)];
            check_hip_r(hipSetDevice(0), "bgl_set0");
            launch_ihc_pre(impl_->r0_streams.ptr, r0e.hc_attn_fn.ptr, r0e.hc_attn_base.ptr, r0e.hc_attn_scale.ptr,
                           impl_->r0_sub_in.ptr, impl_->r0_post.ptr, M, kRmsNormEps, kHcEps, kHcMagnitude, rc0.stream);
            launch_rmsnorm_bf16(impl_->r0_sub_in.ptr, r0e.input_ln.ptr, impl_->r0_x_norm.ptr, M, kHiddenSize, kRmsNormEps, rc0.stream);
            impl_->run_attention_tp5(l, l, false, 0, M, 0, impl_->r0_x_norm.ptr, impl_->r0_sub_in.ptr,
                                     rpos, kslot, mslot, max_kv_active);
            check_hip_r(hipSetDevice(0), "bglp_set0");
            launch_ihc_post_add(impl_->r0_streams.ptr, impl_->r0_sub_in.ptr, impl_->r0_post.ptr, M, kHiddenSize, kHcMult, rc0.stream);
            launch_ihc_pre(impl_->r0_streams.ptr, r0e.hc_mlp_fn.ptr, r0e.hc_mlp_base.ptr, r0e.hc_mlp_scale.ptr,
                           impl_->r0_sub_in.ptr, impl_->r0_post.ptr, M, kRmsNormEps, kHcEps, kHcMagnitude, rc0.stream);
            launch_rmsnorm_bf16(impl_->r0_sub_in.ptr, r0e.post_ln.ptr, impl_->r0_x_norm.ptr, M, kHiddenSize, kRmsNormEps, rc0.stream);
            if (l == 0) impl_->run_dense_mlp_tp5(false, 0, M, impl_->r0_x_norm.ptr, impl_->r0_sub_in.ptr);
            else        impl_->run_moe_3tier(l, M, impl_->r0_x_norm.ptr, impl_->r0_sub_in.ptr);
            check_hip_r(hipSetDevice(0), "bglmp_set0");
            launch_ihc_post_add(impl_->r0_streams.ptr, impl_->r0_sub_in.ptr, impl_->r0_post.ptr, M, kHiddenSize, kHcMult, rc0.stream);
        }
        check_hip_r(hipSetDevice(0), "bgfin_set0");
        impl_->r0_final_h.alloc(0, size_t(M) * size_t(kHiddenSize));
        launch_ihc_head_and_norm(impl_->r0_streams.ptr, impl_->r0_hc_head_fn.ptr, impl_->r0_hc_head_base.ptr,
                                 impl_->r0_hc_head_scale.ptr, impl_->r0_model_norm.ptr,
                                 impl_->r0_final_h.ptr, M, kRmsNormEps, kHcEps, rc0.stream);
        check_hip_r(hipStreamSynchronize(rc0.stream), "bgfin_sync");
    };

    struct SlotState {
        int     orig_idx = -1;
        int32_t cur_tok  = kEosTokenId;
        int     cur_pos  = 0;
    };
    std::vector<SlotState> slots(static_cast<size_t>(n_slots));
    int next_q = 0;
    const int n_elig = int(eligible.size());
    const int pf_chunk = std::clamp(cfg_.prefill_chunk_size, 1, 256);
    const int refill_thresh = std::min(8, std::max(1, n_slots / 4));

    double dec_time_s = 0.0;
    int64_t dec_toks = 0;

    while (!should_stop()) {
        // Count free & active slots
        std::vector<int> free_slots;
        int active_cnt = 0;
        for (int s = 0; s < n_slots; ++s) {
            if (slots[size_t(s)].orig_idx < 0) free_slots.push_back(s);
            else ++active_cnt;
        }
        if (active_cnt == 0 && next_q >= n_elig) break;

        // Batch-refill free slots whenever >= refill_thresh free (or all active finished!) to keep M=256 prefill & B~32 decode saturated!
        const int rem_q = n_elig - next_q;
        if (rem_q > 0 && (active_cnt == 0 || int(free_slots.size()) >= std::min(refill_thresh, rem_q))) {
            const int take = std::min(int(free_slots.size()), rem_q);
            std::vector<int> newly_filled;
            newly_filled.reserve(size_t(take));
            for (int k = 0; k < take; ++k) newly_filled.push_back(free_slots[size_t(k)]);
            clear_slot_l2g(newly_filled);

            std::vector<int32_t> pf_toks, pf_rpos, pf_kslot, pf_mslot;
            std::vector<std::pair<int, int>> pf_slot_end;  // (slot_id, flat_last_idx)
            pf_slot_end.reserve(size_t(take));
            for (int s : newly_filled) {
                const int oidx = eligible[size_t(next_q++)];
                const auto& p = prompts[size_t(oidx)];
                slots[size_t(s)].orig_idx = oidx;
                slots[size_t(s)].cur_pos  = int(p.size());
                const int base = s * kSlotStride;
                for (int j = 0; j < int(p.size()); ++j) {
                    pf_toks.push_back(p[size_t(j)]);
                    pf_rpos.push_back(j);
                    pf_kslot.push_back(base + j);
                    pf_mslot.push_back(base);
                }
                pf_slot_end.push_back({s, int(pf_toks.size()) - 1});
            }

            const int pf_tot = int(pf_toks.size());
            for (int off = 0; off < pf_tot && !should_stop(); off += pf_chunk) {
                const int M = std::min(pf_chunk, pf_tot - off);
                run_batched_chunk(pf_toks.data() + off, pf_rpos.data() + off, pf_kslot.data() + off, pf_mslot.data() + off, M, total_kv_slots);
                if (should_stop()) break;
                for (const auto& se : pf_slot_end) {
                    const int s = se.first;
                    const int li = se.second;
                    if (li >= off && li < off + M) {
                        std::vector<float> lg;
                        impl_->compute_lm_head_tp5(impl_->r0_final_h.ptr + size_t(li - off) * size_t(kHiddenSize), 1, lg);
                        const int oidx = slots[size_t(s)].orig_idx;
                        auto& gen = results[size_t(oidx)];
                        const int32_t t0 = argmax_with_cycle_guard(lg.data(), gen);
                        if (is_stop(t0)) {
                            slots[size_t(s)].orig_idx = -1;
                            if (on_done) on_done(oidx, gen, dec_time_s > 1e-6 ? double(dec_toks) / dec_time_s : 0.0);
                        } else {
                            gen.push_back(t0);
                            slots[size_t(s)].cur_tok = t0;
                            if (int(gen.size()) >= max_new_tokens) {
                                slots[size_t(s)].orig_idx = -1;
                                if (on_done) on_done(oidx, gen, dec_time_s > 1e-6 ? double(dec_toks) / dec_time_s : 0.0);
                            }
                        }
                    }
                }
            }
        }

        // Gather all active slots for ONE lockstep batched decode step!
        std::vector<int> act_s;
        std::vector<int32_t> dt, dr, dk, dm;
        int max_kv_active = 0;
        for (int s = 0; s < n_slots; ++s) {
            if (slots[size_t(s)].orig_idx < 0) continue;
            act_s.push_back(s);
            dt.push_back(slots[size_t(s)].cur_tok);
            dr.push_back(slots[size_t(s)].cur_pos);
            const int ks = s * kSlotStride + slots[size_t(s)].cur_pos;
            dk.push_back(ks);
            dm.push_back(s * kSlotStride);
            max_kv_active = std::max(max_kv_active, ks + 1);
        }
        if (act_s.empty()) continue;

        const auto td0 = std::chrono::steady_clock::now();
        const int M = int(act_s.size());
        run_batched_chunk(dt.data(), dr.data(), dk.data(), dm.data(), M, max_kv_active);
        if (should_stop()) break;

        std::vector<float> all_lg;
        impl_->compute_lm_head_tp5(impl_->r0_final_h.ptr, M, all_lg);
        dec_time_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
        dec_toks += M;
        const double cur_tps = (dec_time_s > 1e-6) ? (double(dec_toks) / dec_time_s) : 0.0;

        for (int i = 0; i < M; ++i) {
            const int s = act_s[size_t(i)];
            const int oidx = slots[size_t(s)].orig_idx;
            ++slots[size_t(s)].cur_pos;
            auto& gen = results[size_t(oidx)];
            const int32_t nxt = argmax_with_cycle_guard(all_lg.data() + size_t(i) * size_t(kVocabSize), gen);
            if (is_stop(nxt)) {
                slots[size_t(s)].orig_idx = -1;
                if (on_done) on_done(oidx, gen, cur_tps);
            } else {
                gen.push_back(nxt);
                slots[size_t(s)].cur_tok = nxt;
                if (int(gen.size()) >= max_new_tokens) {
                    slots[size_t(s)].orig_idx = -1;
                    if (on_done) on_done(oidx, gen, cur_tps);
                }
            }
        }
    }

    if (out_decode_tok_s) *out_decode_tok_s = (dec_time_s > 1e-6) ? (double(dec_toks) / dec_time_s) : 0.0;
    impl_->kv_len = 0;
    impl_->cached_toks.clear();
    return results;
}

}  // namespace hy4_770b
