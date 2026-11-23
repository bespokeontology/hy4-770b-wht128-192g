// Pure-C++17 GATE-1 Unit Verification for tencent/Hy4 (770B) gfx906 HIP + EPYC AVX2 Kernels
// Zero Python anywhere. Strict -Wall -Wextra -Werror + graceful SIGTERM/SIGINT hipDeviceSynchronize() guard.

#include "../src/hy4_770b_config.hpp"
#include "../src/hy4_770b_graceful.hpp"
#include "../src/hy4_770b_hlut_pack.hpp"
#include "../mi50/hy4_770b_core_kernels.hpp"
#include "../mi50/hy4_770b_hlut_kernels.hpp"
#include "../mi50/hy4_770b_ddr4_avx2.hpp"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using namespace hy4_770b;

static void chk(hipError_t e, const char* m) {
    if (e != hipSuccess) throw std::runtime_error(std::string(m) + ": " + hipGetErrorString(e));
}

static double cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += double(a[i]) * double(b[i]);
        na  += double(a[i]) * double(a[i]);
        nb  += double(b[i]) * double(b[i]);
    }
    return dot / (std::sqrt(na * nb) + 1e-30);
}

int main(int argc, char** argv) {
    install_graceful_signal_handlers();
    int target_gpu = 0;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--gpu") target_gpu = std::atoi(argv[++i]);
    }
    ScopedHipDrainGuard guard{5, "HY4_770B_VERIFY"};

    const auto& l2 = lloyd_max_hlut2();
    const auto& l3 = lloyd_max_hlut3();
    const auto& l4 = lloyd_max_hlut4();
    std::printf("[VERIFY] Lloyd-Max FP64: HLUT2=%.2f dB (res=%.2e) | HLUT3=%.2f dB (res=%.2e) | HLUT4=%.2f dB (res=%.2e)\n",
                l2.sqnr_db, l2.convergence_residual, l3.sqnr_db, l3.convergence_residual, l4.sqnr_db, l4.convergence_residual);
    if (!l2.converged || !l3.converged || !l4.converged) return 1;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    const int N = 256, K = 512;
    std::vector<float> W(static_cast<size_t>(N) * static_cast<size_t>(K));
    std::vector<float> X(static_cast<size_t>(K));
    for (float& v : W) v = nd(rng);
    for (float& v : X) v = nd(rng);

    const auto signs = make_wht_signs(K, 0x12345678ULL);
    std::vector<float> W_rot(W.size()), X_rot(X.size());
    wht128_rows(W.data(), W_rot.data(), N, K, signs);
    wht128_rows(X.data(), X_rot.data(), 1, K, signs);

    auto q2 = pack_hlut2_wrot_ldlq(W_rot.data(), N, K, {});
    auto q3 = pack_hlut3_wrot_ldlq(W_rot.data(), N, K, {});
    auto q4 = pack_hlut4_wrot_ldlq(W_rot.data(), N, K, {});

    auto ref_gemv = [&](const std::vector<float>& W_u) {
        std::vector<float> y(static_cast<size_t>(N), 0.0f);
        for (int r = 0; r < N; ++r) {
            double s = 0.0;
            for (int c = 0; c < K; ++c) s += double(W_u[size_t(r) * size_t(K) + size_t(c)]) * double(X_rot[size_t(c)]);
            y[size_t(r)] = float(s);
        }
        return y;
    };
    std::vector<float> W2u(W.size()), W3u(W.size()), W4u(W.size());
    for (int nt = 0; nt < N / 32; ++nt)
        for (int kt = 0; kt < K / 32; ++kt) {
            hlut2_unpack_tile(q2.tile_at(nt, kt), l2, W2u.data(), K, nt * 32, kt * 32);
            hlut3_unpack_tile(q3.tile_at(nt, kt), l3, W3u.data(), K, nt * 32, kt * 32);
            hlut4_unpack_tile(q4.tile_at(nt, kt), l4, W4u.data(), K, nt * 32, kt * 32);
        }
    auto y2_ref = ref_gemv(W2u), y3_ref = ref_gemv(W3u), y4_ref = ref_gemv(W4u);

    float lv8[8];
    for (int i = 0; i < 8; ++i) lv8[i] = float(l4.levels[size_t(i)]);
    std::vector<float> y4_avx2(static_cast<size_t>(N), 0.0f);
    Hy4Ddr4ColdEngine::avx2_hlut4_gemv_rows(X_rot.data(), q4.tiles.data(), lv8, 0, N / 32, K / 32, y4_avx2.data());
    const double cos_avx2 = cosine_sim(y4_ref, y4_avx2);
    std::printf("[VERIFY] AVX2 DDR4 HLUT4 GEMV vs scalar ref: cosine = %.8f\n", cos_avx2);
    if (cos_avx2 < 0.999999) return 2;

    int dev_cnt = 0;
    if (hipGetDeviceCount(&dev_cnt) != hipSuccess || dev_cnt <= 0) {
        std::printf("[VERIFY] Stage A CPU-only PASS (no HIP devices visible)\n");
        return 0;
    }
    chk(hipSetDevice(target_gpu), "setDevice");
    hipStream_t s = nullptr;
    chk(hipStreamCreate(&s), "streamCreate");

    float *d_x = nullptr, *d_y = nullptr, *d_lv2 = nullptr, *d_lv4 = nullptr, *d_lv8 = nullptr;
    Hlut2Tile32x32* d_t2 = nullptr;
    Hlut3Tile32x32* d_t3 = nullptr;
    Hlut4Tile32x32* d_t4 = nullptr;
    chk(hipMalloc(&d_x, size_t(K) * 4u), "mx");
    chk(hipMalloc(&d_y, size_t(N) * 4u), "my");
    chk(hipMalloc(&d_lv2, 2u * 4u), "ml2"); chk(hipMalloc(&d_lv4, 4u * 4u), "ml4"); chk(hipMalloc(&d_lv8, 8u * 4u), "ml8");
    chk(hipMalloc(&d_t2, q2.raw_bytes()), "mt2"); chk(hipMalloc(&d_t3, q3.raw_bytes()), "mt3"); chk(hipMalloc(&d_t4, q4.raw_bytes()), "mt4");

    float lv2[2] = {float(l2.levels[0]), float(l2.levels[1])};
    float lv4[4] = {float(l3.levels[0]), float(l3.levels[1]), float(l3.levels[2]), float(l3.levels[3])};
    chk(hipMemcpy(d_x, X_rot.data(), size_t(K) * 4u, hipMemcpyHostToDevice), "cx");
    chk(hipMemcpy(d_lv2, lv2, sizeof(lv2), hipMemcpyHostToDevice), "cl2");
    chk(hipMemcpy(d_lv4, lv4, sizeof(lv4), hipMemcpyHostToDevice), "cl4");
    chk(hipMemcpy(d_lv8, lv8, sizeof(lv8), hipMemcpyHostToDevice), "cl8");
    chk(hipMemcpy(d_t2, q2.tiles.data(), q2.raw_bytes(), hipMemcpyHostToDevice), "ct2");
    chk(hipMemcpy(d_t3, q3.tiles.data(), q3.raw_bytes(), hipMemcpyHostToDevice), "ct3");
    chk(hipMemcpy(d_t4, q4.tiles.data(), q4.raw_bytes(), hipMemcpyHostToDevice), "ct4");

    std::vector<float> y2_gpu(static_cast<size_t>(N)), y3_gpu(static_cast<size_t>(N)), y4_gpu(static_cast<size_t>(N));
    launch_hlut2_gemv(d_x, d_t2, d_lv2, d_y, 1, N, K, s);
    chk(hipMemcpyAsync(y2_gpu.data(), d_y, size_t(N) * 4u, hipMemcpyDeviceToHost, s), "cy2");
    chk(hipStreamSynchronize(s), "s2");

    launch_hlut3_gemv(d_x, d_t3, d_lv4, d_y, 1, N, K, s);
    chk(hipMemcpyAsync(y3_gpu.data(), d_y, size_t(N) * 4u, hipMemcpyDeviceToHost, s), "cy3");
    chk(hipStreamSynchronize(s), "s3");

    launch_hlut4_gemv(d_x, d_t4, d_lv8, d_y, 1, N, K, s);
    chk(hipMemcpyAsync(y4_gpu.data(), d_y, size_t(N) * 4u, hipMemcpyDeviceToHost, s), "cy4");
    chk(hipStreamSynchronize(s), "s4");

    const double c2 = cosine_sim(y2_ref, y2_gpu);
    const double c3 = cosine_sim(y3_ref, y3_gpu);
    const double c4 = cosine_sim(y4_ref, y4_gpu);
    std::printf("[VERIFY] GPU MI50 gfx906 GEMVs vs exact ref: HLUT2=%.8f HLUT3=%.8f HLUT4=%.8f (dev=%d/%d)\n",
                c2, c3, c4, target_gpu, dev_cnt);

    // Also verify grouped MI50 WHT128-HLUT4 Gate+Up+SwiGLU + WHT128 + Down atomic on exact [6144, 2048] geometry!
    double c4_grp = 0.0;
    {
        const int M = 6;
        const int H = kHiddenSize, I = kMoeInterSize;
        std::vector<float> Xm(size_t(M) * size_t(H));
        std::vector<uint16_t> Xm_f16(Xm.size());
        for (size_t i = 0; i < Xm.size(); ++i) {
            Xm[i] = 0.25f * nd(rng);
            Xm_f16[i] = f32_to_fp16_rne(Xm[i]);
            Xm[i] = fp16_to_f32(Xm_f16[i]);
        }
        std::vector<float> Wg(size_t(I) * size_t(H)), Wu(size_t(I) * size_t(H)), Wd(size_t(H) * size_t(I));
        for (float& v : Wg) v = 0.03f * nd(rng);
        for (float& v : Wu) v = 0.03f * nd(rng);
        for (float& v : Wd) v = 0.03f * nd(rng);
        auto qg = pack_hlut4_wrot_ldlq(Wg.data(), I, H, {});
        auto qu = pack_hlut4_wrot_ldlq(Wu.data(), I, H, {});
        auto qd = pack_hlut4_wrot_ldlq(Wd.data(), H, I, {});
        const auto signs_i = make_signs_i(1);

        // Exact FP64 CPU reference on unpacked tiles
        std::vector<float> ug(Wg.size()), uu(Wu.size()), ud(Wd.size());
        for (int nt = 0; nt < qg.nt; ++nt) for (int kt = 0; kt < qg.kt; ++kt) {
            hlut4_unpack_tile(qg.tile_at(nt, kt), l4, ug.data(), H, nt * 32, kt * 32);
            hlut4_unpack_tile(qu.tile_at(nt, kt), l4, uu.data(), H, nt * 32, kt * 32);
        }
        for (int nt = 0; nt < qd.nt; ++nt) for (int kt = 0; kt < qd.kt; ++kt) {
            hlut4_unpack_tile(qd.tile_at(nt, kt), l4, ud.data(), I, nt * 32, kt * 32);
        }
        Hy4ExpertGroup grp{};
        grp.expert = 0; grp.count = M;
        for (int m = 0; m < M; ++m) { grp.rows[m] = m; grp.weights[m] = 0.35f + 0.05f * float(m); }
        std::vector<float> y_ref(size_t(M) * size_t(H), 0.0f);
        for (int m = 0; m < M; ++m) {
            std::vector<float> z(static_cast<size_t>(I), 0.0f);
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < I; ++r) {
                double ag = 0.0, au = 0.0;
                for (int k = 0; k < H; ++k) {
                    const double xv = double(Xm[size_t(m) * size_t(H) + size_t(k)]);
                    ag += double(ug[size_t(r) * size_t(H) + size_t(k)]) * xv;
                    au += double(uu[size_t(r) * size_t(H) + size_t(k)]) * xv;
                }
                z[size_t(r)] = hy4_swiglu_clamped(float(ag), float(au));
            }
            wht128_rows(z.data(), z.data(), 1, I, signs_i);
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < H; ++r) {
                double ad = 0.0;
                for (int k = 0; k < I; ++k) ad += double(ud[size_t(r) * size_t(I) + size_t(k)]) * double(z[size_t(k)]);
                y_ref[size_t(m) * size_t(H) + size_t(r)] = float(ad) * grp.weights[m];
            }
        }

        uint16_t* d_xm = nullptr; Hy4ExpertGroup* d_gr = nullptr; uint64_t* d_si = nullptr;
        Hlut4Tile32x32 *d_g4 = nullptr, *d_u4 = nullptr, *d_d4 = nullptr;
        float *d_z = nullptr, *d_acc = nullptr;
        chk(hipMalloc(&d_xm, Xm_f16.size() * 2u), "gxm");
        chk(hipMalloc(&d_gr, sizeof(Hy4ExpertGroup)), "ggr");
        chk(hipMalloc(&d_si, signs_i.size() * 8u), "gsi");
        chk(hipMalloc(&d_g4, qg.raw_bytes()), "gg4");
        chk(hipMalloc(&d_u4, qu.raw_bytes()), "gu4");
        chk(hipMalloc(&d_d4, qd.raw_bytes()), "gd4");
        chk(hipMalloc(&d_z, size_t(kMaxGroupTokens) * size_t(I) * 4u), "gz");
        chk(hipMalloc(&d_acc, size_t(M) * size_t(H) * 4u), "gacc");
        chk(hipMemset(d_acc, 0, size_t(M) * size_t(H) * 4u), "zacc");
        chk(hipMemcpy(d_xm, Xm_f16.data(), Xm_f16.size() * 2u, hipMemcpyHostToDevice), "cxm");
        chk(hipMemcpy(d_gr, &grp, sizeof(grp), hipMemcpyHostToDevice), "cgr");
        chk(hipMemcpy(d_si, signs_i.data(), signs_i.size() * 8u, hipMemcpyHostToDevice), "csi");
        chk(hipMemcpy(d_g4, qg.tiles.data(), qg.raw_bytes(), hipMemcpyHostToDevice), "cg4");
        chk(hipMemcpy(d_u4, qu.tiles.data(), qu.raw_bytes(), hipMemcpyHostToDevice), "cu4");
        chk(hipMemcpy(d_d4, qd.tiles.data(), qd.raw_bytes(), hipMemcpyHostToDevice), "cd4");

        launch_hlut4_grouped_gateup_swiglu_f16(d_xm, d_gr, d_g4, d_u4, d_lv8, d_z, 1, I, H, s);
        launch_wht128_f32_inplace(d_z, d_si, kMaxGroupTokens, I, s);
        launch_hlut4_grouped_down_atomic(d_z, d_gr, d_d4, d_lv8, d_acc, 1, H, I, s);
        std::vector<float> y_gpu(size_t(M) * size_t(H));
        chk(hipMemcpyAsync(y_gpu.data(), d_acc, y_gpu.size() * 4u, hipMemcpyDeviceToHost, s), "cya");
        chk(hipStreamSynchronize(s), "sg4");
        c4_grp = cosine_sim(y_ref, y_gpu);
        std::printf("[VERIFY] GPU MI50 gfx906 Grouped HLUT4 [6144,2048] M=%d vs FP64 ref: cosine = %.8f\n", M, c4_grp);
        (void)hipFree(d_xm); (void)hipFree(d_gr); (void)hipFree(d_si);
        (void)hipFree(d_g4); (void)hipFree(d_u4); (void)hipFree(d_d4); (void)hipFree(d_z); (void)hipFree(d_acc);
    }

    // 7. Verify Lossless Stock FP8-g128 Grouped MoE [6144,2048] M=6 (4-token weight-stationary) vs exact FP64 ref!
    double c_fp8_grp = 0.0;
    {
        const int M = 6, H = kHiddenSize, I = kMoeInterSize;
        std::vector<float> Wg_f(size_t(I) * H), Wu_f(size_t(I) * H), Wd_f(size_t(H) * I);
        for (auto& v : Wg_f) v = nd(rng) * 0.02f;
        for (auto& v : Wu_f) v = nd(rng) * 0.02f;
        for (auto& v : Wd_f) v = nd(rng) * 0.02f;
        auto qg = quantize_fp8_block128(Wg_f.data(), I, H);
        auto qu = quantize_fp8_block128(Wu_f.data(), I, H);
        auto qd = quantize_fp8_block128(Wd_f.data(), H, I);
        std::vector<float> ug(size_t(I) * H), uu(size_t(I) * H), ud(size_t(H) * I);
        dequant_fp8_block128(qg.weight_fp8.data(), qg.weight_scale.data(), I, H, ug.data());
        dequant_fp8_block128(qu.weight_fp8.data(), qu.weight_scale.data(), I, H, uu.data());
        dequant_fp8_block128(qd.weight_fp8.data(), qd.weight_scale.data(), H, I, ud.data());

        std::vector<uint16_t> X_bf16(size_t(M) * H);
        std::vector<float> X_f(size_t(M) * H);
        for (size_t i = 0; i < X_f.size(); ++i) {
            X_bf16[i] = f32_to_bf16_rne(nd(rng) * 0.5f);
            X_f[i] = bf16_to_f32(X_bf16[i]);
        }
        Hy4ExpertGroup grp{};
        grp.expert = 0; grp.count = M;
        for (int m = 0; m < M; ++m) { grp.rows[m] = m; grp.weights[m] = 0.4f + 0.07f * float(m); }
        std::vector<float> y_ref(size_t(M) * H, 0.0f);
        for (int m = 0; m < M; ++m) {
            std::vector<float> z(static_cast<size_t>(I), 0.0f);
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < I; ++r) {
                double ag = 0.0, au = 0.0;
                for (int k = 0; k < H; ++k) {
                    const double xv = double(X_f[size_t(m) * H + k]);
                    ag += double(ug[size_t(r) * H + k]) * xv;
                    au += double(uu[size_t(r) * H + k]) * xv;
                }
                z[size_t(r)] = hy4_swiglu_clamped(float(ag), float(au));
            }
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < H; ++r) {
                double ad = 0.0;
                for (int k = 0; k < I; ++k) ad += double(ud[size_t(r) * I + k]) * double(z[size_t(k)]);
                y_ref[size_t(m) * H + r] = float(ad) * grp.weights[m];
            }
        }
        uint16_t* d_xb = nullptr; Hy4ExpertGroup* d_gr = nullptr;
        uint8_t *d_gw = nullptr, *d_uw = nullptr, *d_dw = nullptr;
        float *d_gs = nullptr, *d_us = nullptr, *d_ds = nullptr, *d_z = nullptr, *d_acc = nullptr;
        chk(hipMalloc(&d_xb, X_bf16.size() * 2u), "f8x"); chk(hipMalloc(&d_gr, sizeof(grp)), "f8g");
        chk(hipMalloc(&d_gw, qg.weight_fp8.size()), "f8gw"); chk(hipMalloc(&d_gs, qg.weight_scale.size() * 4u), "f8gs");
        chk(hipMalloc(&d_uw, qu.weight_fp8.size()), "f8uw"); chk(hipMalloc(&d_us, qu.weight_scale.size() * 4u), "f8us");
        chk(hipMalloc(&d_dw, qd.weight_fp8.size()), "f8dw"); chk(hipMalloc(&d_ds, qd.weight_scale.size() * 4u), "f8ds");
        chk(hipMalloc(&d_z, size_t(kMaxGroupTokens) * I * 4u), "f8z");
        chk(hipMalloc(&d_acc, size_t(M) * H * 4u), "f8a");
        chk(hipMemset(d_acc, 0, size_t(M) * H * 4u), "f8az");
        chk(hipMemcpy(d_xb, X_bf16.data(), X_bf16.size() * 2u, hipMemcpyHostToDevice), "cf8x");
        chk(hipMemcpy(d_gr, &grp, sizeof(grp), hipMemcpyHostToDevice), "cf8g");
        chk(hipMemcpy(d_gw, qg.weight_fp8.data(), qg.weight_fp8.size(), hipMemcpyHostToDevice), "cf8gw");
        chk(hipMemcpy(d_gs, qg.weight_scale.data(), qg.weight_scale.size() * 4u, hipMemcpyHostToDevice), "cf8gs");
        chk(hipMemcpy(d_uw, qu.weight_fp8.data(), qu.weight_fp8.size(), hipMemcpyHostToDevice), "cf8uw");
        chk(hipMemcpy(d_us, qu.weight_scale.data(), qu.weight_scale.size() * 4u, hipMemcpyHostToDevice), "cf8us");
        chk(hipMemcpy(d_dw, qd.weight_fp8.data(), qd.weight_fp8.size(), hipMemcpyHostToDevice), "cf8dw");
        chk(hipMemcpy(d_ds, qd.weight_scale.data(), qd.weight_scale.size() * 4u, hipMemcpyHostToDevice), "cf8ds");
        launch_fp8_grouped_gateup_swiglu(d_xb, d_gr, d_gw, d_gs, d_uw, d_us, d_z, 1, I, H, s);
        launch_fp8_grouped_down_atomic(d_z, d_gr, d_dw, d_ds, d_acc, 1, H, I, s);
        std::vector<float> y_gpu(size_t(M) * H);
        chk(hipMemcpyAsync(y_gpu.data(), d_acc, y_gpu.size() * 4u, hipMemcpyDeviceToHost, s), "cf8y");
        chk(hipStreamSynchronize(s), "sf8");
        c_fp8_grp = cosine_sim(y_ref, y_gpu);
        std::printf("[VERIFY] GPU MI50 gfx906 Grouped FP8-g128 [6144,2048] M=%d vs FP64 ref: cosine = %.8f\n", M, c_fp8_grp);
        (void)hipFree(d_xb); (void)hipFree(d_gr); (void)hipFree(d_gw); (void)hipFree(d_gs);
        (void)hipFree(d_uw); (void)hipFree(d_us); (void)hipFree(d_dw); (void)hipFree(d_ds);
        (void)hipFree(d_z);  (void)hipFree(d_acc);
    }

    // 8. Verify AVX2 Q5_LM128 GEMV vs exact scalar unpack reference!
    double c_q5_avx = 0.0;
    {
        const int Nr = 256, Kr = 1024, kb = Kr / 128;
        std::vector<float> Wq(size_t(Nr) * Kr), Xq(Kr), y_ref(Nr), y_avx(Nr);
        for (auto& v : Wq) v = nd(rng) * 0.05f;
        for (auto& v : Xq) v = nd(rng);
        std::vector<Q5Lm128Block> blks(size_t(Nr) * kb);
        for (int r = 0; r < Nr; ++r) {
            double acc = 0.0;
            for (int b = 0; b < kb; ++b) {
                q5_lm128_pack_block(Wq.data() + size_t(r) * Kr + size_t(b) * 128u, blks[size_t(r) * kb + b]);
                float rec[128];
                q5_lm128_unpack_block(blks[size_t(r) * kb + b], rec);
                for (int c = 0; c < 128; ++c) acc += double(rec[c]) * double(Xq[size_t(b) * 128u + c]);
            }
            y_ref[size_t(r)] = float(acc);
        }
        Hy4Ddr4ColdEngine::avx2_q5_lm128_gemv_rows(Xq.data(), blks.data(), 0, Nr, kb, y_avx.data());
        c_q5_avx = cosine_sim(y_ref, y_avx);
        std::printf("[VERIFY] CPU EPYC AVX2 Q5_LM128 [256,1024] vs scalar unpack ref: cosine = %.8f\n", c_q5_avx);
    }

    // 9. Verify GPU Lossless BF16 MLA Partial Kernel (L74..L77) vs exact FP64 ref!
    double c_mla_bf16 = 0.0;
    {
        const int M = 4, H = 8, L = 32;
        std::vector<uint16_t> q_bf16(size_t(M) * H * kKvLatentDim), kv_bf16(size_t(L) * kKvLatentDim);
        for (auto& v : q_bf16)  v = f32_to_bf16_rne(nd(rng) * 0.3f);
        for (auto& v : kv_bf16) v = f32_to_bf16_rne(nd(rng) * 500.0f);  // massive-activation regime!
        std::vector<int32_t> l2g(L), qpos(M);
        for (int t = 0; t < L; ++t) l2g[t] = t;
        for (int m = 0; m < M; ++m) qpos[m] = L - M + m;

        std::vector<float> ref_acc(size_t(M) * H * kKvLoraRank, 0.0f);
        for (int m = 0; m < M; ++m) {
            for (int h = 0; h < H; ++h) {
                const uint16_t* qp = q_bf16.data() + (size_t(m) * H + h) * kKvLatentDim;
                std::vector<double> logits(L, -1e30);
                double mx = -1e30;
                for (int t = 0; t <= qpos[m]; ++t) {
                    const uint16_t* kp = kv_bf16.data() + size_t(t) * kKvLatentDim;
                    double dot = 0.0;
                    for (int d = 0; d < kKvLatentDim; ++d) dot += double(bf16_to_f32(qp[d])) * double(bf16_to_f32(kp[d]));
                    logits[t] = dot * double(kAttnScale);
                    mx = std::max(mx, logits[t]);
                }
                double sm = 0.0;
                for (int t = 0; t <= qpos[m]; ++t) {
                    const double w = std::exp(logits[t] - mx);
                    sm += w;
                    const uint16_t* kp = kv_bf16.data() + size_t(t) * kKvLatentDim;
                    for (int d = 0; d < kKvLoraRank; ++d) {
                        ref_acc[(size_t(m) * H + h) * kKvLoraRank + d] += float(w * double(bf16_to_f32(kp[d])));
                    }
                }
                for (int d = 0; d < kKvLoraRank; ++d) ref_acc[(size_t(m) * H + h) * kKvLoraRank + d] /= float(sm);
            }
        }
        uint16_t *dq = nullptr, *dkv = nullptr;
        int32_t *dl2g = nullptr, *dqpos = nullptr;
        float *dom = nullptr, *dos = nullptr, *doa = nullptr;
        chk(hipMalloc(&dq, q_bf16.size() * 2u), "mq"); chk(hipMalloc(&dkv, kv_bf16.size() * 2u), "mkv");
        chk(hipMalloc(&dl2g, L * 4u), "ml2g"); chk(hipMalloc(&dqpos, M * 4u), "mqp");
        chk(hipMalloc(&dom, size_t(M) * H * 4u), "mom"); chk(hipMalloc(&dos, size_t(M) * H * 4u), "mos");
        chk(hipMalloc(&doa, size_t(M) * H * kKvLoraRank * 4u), "moa");
        chk(hipMemcpy(dq, q_bf16.data(), q_bf16.size() * 2u, hipMemcpyHostToDevice), "cmq");
        chk(hipMemcpy(dkv, kv_bf16.data(), kv_bf16.size() * 2u, hipMemcpyHostToDevice), "cmkv");
        chk(hipMemcpy(dl2g, l2g.data(), L * 4u, hipMemcpyHostToDevice), "cml2g");
        chk(hipMemcpy(dqpos, qpos.data(), M * 4u, hipMemcpyHostToDevice), "cmqp");
        launch_mla_absorbed_partial_bf16(dq, dkv, dl2g, dqpos, nullptr, nullptr, dom, dos, doa, M, H, L, 0, kAttnScale, s);
        std::vector<float> gm(size_t(M) * H), gs(size_t(M) * H), ga(size_t(M) * H * kKvLoraRank);
        chk(hipMemcpyAsync(gm.data(), dom, gm.size() * 4u, hipMemcpyDeviceToHost, s), "cgm");
        chk(hipMemcpyAsync(gs.data(), dos, gs.size() * 4u, hipMemcpyDeviceToHost, s), "cgs");
        chk(hipMemcpyAsync(ga.data(), doa, ga.size() * 4u, hipMemcpyDeviceToHost, s), "cga");
        chk(hipStreamSynchronize(s), "smla");
        for (int mh = 0; mh < M * H; ++mh)
            for (int d = 0; d < kKvLoraRank; ++d) ga[size_t(mh) * kKvLoraRank + d] /= gs[size_t(mh)];
        c_mla_bf16 = cosine_sim(ref_acc, ga);
        std::printf("[VERIFY] GPU MI50 gfx906 Lossless BF16 MLA Partial (RMS=500 massive-act) vs FP64 ref: cosine = %.8f\n", c_mla_bf16);
        (void)hipFree(dq); (void)hipFree(dkv); (void)hipFree(dl2g); (void)hipFree(dqpos);
        (void)hipFree(dom); (void)hipFree(dos); (void)hipFree(doa);
    }

    chk(hipFree(d_x), "fx"); chk(hipFree(d_y), "fy");
    chk(hipFree(d_lv2), "fl2"); chk(hipFree(d_lv4), "fl4"); chk(hipFree(d_lv8), "fl8");
    chk(hipFree(d_t2), "ft2"); chk(hipFree(d_t3), "ft3"); chk(hipFree(d_t4), "ft4");
    chk(hipStreamDestroy(s), "ds");

    if (c2 < 0.999999 || c3 < 0.999999 || c4 < 0.999999 || c4_grp < 0.999999 ||
        c_fp8_grp < 0.999999 || c_q5_avx < 0.999999 || c_mla_bf16 < 0.999999) return 3;
    std::printf("[VERIFY] GATE-1 GPU+AVX2 KERNEL SELF-TEST 100%% PASS!\n");
    return 0;
}
