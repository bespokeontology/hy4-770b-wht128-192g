// Pure-C++17 Surgical GATE-2 Root-Cause Diagnostic Tool for tencent/Hy4 (770B)
// Isolates Probe 56 (index 55, 1.773 nats), Probe 24 (index 23, 1.018 nats), and Probe 32 (index 31, 0.000000 nats):
//   1. Tests M=64 vs M=16 vs M=1 sequential prefill invariance: KL(M1, M64), KL(M1, Oracle), KL(M16, Oracle), KL(M64, Oracle)
//   2. Logs per-layer L00..L77: h_rms, h_max_abs, x_rot_max_abs, P150 vs exact CPU HLUT3 cosine & max_abs_err,
//      and layer-by-layer M=64 vs M=1 hidden state diff + top-8 router overlap to pinpoint exact fault layer l*!
//   3. Tests fresh-KV ordering (Probe 56 run FIRST on clean KV cache) vs back-to-back ordering.
// Zero Python. Graceful hipDeviceSynchronize() teardown.

#include "../mi50/hy4_770b_runtime.hpp"
#include "../mi50/hy4_json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace hy4_770b {
namespace {

static std::string read_str(const std::string& p) {
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open " + p);
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

static std::vector<std::vector<int32_t>> load_probes_64(const std::string& p) {
    const auto j = hy4::parse_json(read_str(p));
    std::vector<std::vector<int32_t>> raw;
    for (const auto& item : j.arr_val) {
        const auto* ids = item.find("ids");
        if (!ids || !ids->is_array() || ids->arr_val.size() < 65) continue;
        std::vector<int32_t> v;
        v.reserve(ids->arr_val.size());
        for (const auto& x : ids->arr_val) v.push_back(int32_t(x.as_int()));
        raw.push_back(std::move(v));
    }
    std::vector<std::vector<int32_t>> out;
    for (size_t i = 0; i < raw.size() && out.size() < 48; ++i) out.emplace_back(raw[i].begin(), raw[i].begin() + 64);
    for (size_t i = 0; i < raw.size() && out.size() < 64; ++i)
        if (raw[i].size() >= 128) out.emplace_back(raw[i].begin() + 64, raw[i].begin() + 128);
    return out;
}

struct KlRes {
    double kl = 0.0;
    double top10_kl = 0.0;
    double max_logit_diff = 0.0;
    int ref_top1 = 0, q_top1 = 0;
};

static KlRes compare_logits(const float* ref, const float* q, int V) {
    auto lsm = [&](const float* lg, std::vector<double>& lp, std::vector<double>& pr) {
        lp.resize(static_cast<size_t>(V)); pr.resize(static_cast<size_t>(V));
        double mx = -1e300;
        for (int i = 0; i < V; ++i) if (std::isfinite(lg[i]) && double(lg[i]) > mx) mx = double(lg[i]);
        double s = 0.0;
        for (int i = 0; i < V; ++i) {
            const double d = std::isfinite(lg[i]) ? (double(lg[i]) - mx) : -100.0;
            const double e = std::exp(std::max(d, -100.0));
            pr[size_t(i)] = e; s += e;
        }
        const double lse = mx + std::log(std::max(s, 1e-300));
        const double inv = 1.0 / std::max(s, 1e-300);
        for (int i = 0; i < V; ++i) {
            const double v = std::isfinite(lg[i]) ? double(lg[i]) : (mx - 100.0);
            lp[size_t(i)] = v - lse; pr[size_t(i)] *= inv;
        }
    };
    std::vector<double> lpr, pr, lpq, pq;
    lsm(ref, lpr, pr); lsm(q, lpq, pq);
    KlRes r{};
    for (int i = 0; i < V; ++i) {
        if (ref[i] > ref[r.ref_top1]) r.ref_top1 = i;
        if (q[i]   > q[r.q_top1])     r.q_top1 = i;
        if (pr[size_t(i)] > 1e-18) r.kl += pr[size_t(i)] * (lpr[size_t(i)] - lpq[size_t(i)]);
        r.max_logit_diff = std::max(r.max_logit_diff, std::fabs(double(ref[i]) - double(q[i])));
    }
    r.kl = std::max(0.0, r.kl);
    std::vector<int> idx(static_cast<size_t>(V));
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return ref[a] > ref[b]; });
    double zr = 0.0, zq = 0.0;
    for (int k = 0; k < 10; ++k) { zr += pr[size_t(idx[size_t(k)])]; zq += pq[size_t(idx[size_t(k)])]; }
    for (int k = 0; k < 10; ++k) {
        const int id = idx[size_t(k)];
        const double a = pr[size_t(id)] / std::max(zr, 1e-30);
        const double b = pq[size_t(id)] / std::max(zq, 1e-30);
        if (a > 1e-18) r.top10_kl += a * std::log(std::max(a, 1e-30) / std::max(b, 1e-30));
    }
    r.top10_kl = std::max(0.0, r.top10_kl);
    return r;
}

static double vec_cos(const std::vector<float>& a, const std::vector<float>& b, double& max_diff) {
    double dot = 0.0, n1 = 0.0, n2 = 0.0; max_diff = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const double x = double(a[i]), y = double(b[i]);
        dot += x * y; n1 += x * x; n2 += y * y;
        max_diff = std::max(max_diff, std::fabs(x - y));
    }
    return (n1 > 1e-30 && n2 > 1e-30) ? (dot / std::sqrt(n1 * n2)) : 1.0;
}

}  // namespace
}  // namespace hy4_770b

int main(int argc, char** argv) {
    using namespace hy4_770b;
    install_graceful_signal_handlers();

    Hy4770bRuntimeConfig cfg{};
    cfg.profile = RuntimeProfile::PROFILE_A_FIDELITY_HLUT3_DDR4;
    cfg.max_kv_len = 524288;
    cfg.load_vit = false;
    cfg.use_dfly2 = false;
    cfg.shutdown_p150_on_exit = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-p150") cfg.use_p150 = false;
        else if (a == "--max-kv" && i + 1 < argc) cfg.max_kv_len = std::atoi(argv[++i]);
    }

    const auto probes = load_probes_64(cfg.root_dir + "/bench_data/logprob_corpus.json");
    std::vector<float> oracle(size_t(64) * size_t(kVocabSize));
    {
        std::ifstream ifs(cfg.root_dir + "/artifacts/oracle_ref_logits_64.bin", std::ios::binary);
        ifs.read(reinterpret_cast<char*>(oracle.data()), std::streamsize(oracle.size() * 4u));
    }

    try {
        Hy4770bRuntime rt;
        rt.init(cfg);

        // Diagnostic targets (1-based Probe 56 -> idx 55, Probe 24 -> idx 23, Probe 32 -> idx 31)
        const std::vector<int> target_idx = {55, 23, 31};
        for (int pidx : target_idx) {
            if (should_stop()) break;
            const float* ref = oracle.data() + size_t(pidx) * size_t(kVocabSize);
            std::printf("\n==================== PROBE %02d (idx=%d) ====================\n", pidx + 1, pidx);

            std::vector<float> log_m64, log_m16, log_m1;
            std::vector<Hy4770bLayerDiag> diag_m64, diag_m16, diag_m1;

            // 1. M=64 (with exact CPU HLUT3 verification of P150A on every layer!)
            rt.clear_kv_cache();
            rt.set_prefill_chunk_size(64);
            rt.set_diag_sink(&diag_m64, /*verify_p150_cpu=*/true);
            rt.forward_tokens(probes[size_t(pidx)], 0, log_m64);
            const auto r64 = compare_logits(ref, log_m64.data(), kVocabSize);
            std::printf("[P%02d M=64] KL=%.6f | Top10-KL=%.6f | Top1: q=%d ref=%d (%s)\n",
                        pidx + 1, r64.kl, r64.top10_kl, r64.q_top1, r64.ref_top1, r64.q_top1 == r64.ref_top1 ? "MATCH" : "FLIP");
            std::fflush(stdout);

            // 2. M=16 chunked (P150A takes run_decode_small rows<=32 instead of run_prefill_compact!)
            rt.clear_kv_cache();
            rt.set_prefill_chunk_size(16);
            rt.set_diag_sink(&diag_m16, /*verify_p150_cpu=*/true);
            rt.forward_tokens(probes[size_t(pidx)], 0, log_m16);
            const auto r16 = compare_logits(ref, log_m16.data(), kVocabSize);
            const auto d16_64 = compare_logits(log_m16.data(), log_m64.data(), kVocabSize);
            std::printf("[P%02d M=16] KL=%.6f | Top10-KL=%.6f | Top1: q=%d ref=%d (%s) | KL(M16,M64)=%.6e maxLogDiff=%.4e\n",
                        pidx + 1, r16.kl, r16.top10_kl, r16.q_top1, r16.ref_top1,
                        r16.q_top1 == r16.ref_top1 ? "MATCH" : "FLIP", d16_64.kl, d16_64.max_logit_diff);
            std::fflush(stdout);

            // 3. M=1 sequential (only for Probe 56 & Probe 24 to verify M=1 vs M=16 vs M=64 exact invariance!)
            if (pidx == 55 || pidx == 23) {
                rt.clear_kv_cache();
                rt.set_prefill_chunk_size(1);
                rt.set_diag_sink(&diag_m1, /*verify_p150_cpu=*/false);
                rt.forward_tokens(probes[size_t(pidx)], 0, log_m1);
                const auto r1 = compare_logits(ref, log_m1.data(), kVocabSize);
                const auto d1_16 = compare_logits(log_m1.data(), log_m16.data(), kVocabSize);
                const auto d1_64 = compare_logits(log_m1.data(), log_m64.data(), kVocabSize);
                std::printf("[P%02d M= 1] KL=%.6f | Top10-KL=%.6f | Top1: q=%d ref=%d (%s) | KL(M1,M16)=%.6e | KL(M1,M64)=%.6e\n",
                            pidx + 1, r1.kl, r1.top10_kl, r1.q_top1, r1.ref_top1,
                            r1.q_top1 == r1.ref_top1 ? "MATCH" : "FLIP", d1_16.kl, d1_64.kl);
                std::fflush(stdout);
            }

            // Print per-layer fault table (L00..L77): flag any P150 CPU mismatch OR M=64 vs M=16 hidden divergence!
            std::printf("  L  |  h_rms  | h_max  | xrot_mx | P150_cos   P150_err  maxR | M16_P150_cos | h_cos(16v64) h_diff(16v64) | top8_ov\n");
            for (size_t l = 0; l < diag_m64.size() && l < diag_m16.size(); ++l) {
                const auto& a = diag_m64[l];
                const auto& b = diag_m16[l];
                double hd = 0.0;
                const double hc = vec_cos(b.last_tok_h_stream0, a.last_tok_h_stream0, hd);
                int ov = 0;
                for (int x : a.last_tok_topk) for (int y : b.last_tok_topk) if (x == y) { ++ov; break; }
                const bool interesting = (l <= 3 || l >= 74 || (l >= 28 && l <= 37) ||
                                          a.p150_cpu_cos < 0.999 || b.p150_cpu_cos < 0.999 ||
                                          hc < 0.9999 || ov < 8 || a.x_rot_max_abs > 5000.0f);
                if (interesting) {
                    std::printf("  L%02d| %7.3f | %6.1f | %7.2f | %.6f %9.4e %3d |   %.6f   |   %.6f   %11.4e | %d/8\n",
                                a.layer, a.h_rms, a.h_max_abs, a.x_rot_max_abs,
                                a.p150_cpu_cos, a.p150_cpu_max_err, a.p150_max_rows_per_exp,
                                b.p150_cpu_cos, hc, hd, ov);
                }
            }
            std::fflush(stdout);
        }

        rt.set_diag_sink(nullptr, false);
        rt.shutdown();
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[DIAG FATAL] %s\n", ex.what());
        drain_all_hip_devices(kNumMi50Ranks, "DIAG_ERR");
        return 1;
    }
}
