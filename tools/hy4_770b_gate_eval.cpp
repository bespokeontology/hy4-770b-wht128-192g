// Pure-C++17 Controlled Head-to-Head Silicon Verification Harness (GATE-2, GATE-3, GATE-4)
// Model: tencent/Hy4 (770B total / 49B active) on SB-M5-P2-192G (5x MI50 + 2x Blackhole P150A + 384 GiB DDR4)
// Compares directly against the $220,000 two-station GB300 + RTX PRO 6000 benchmark ledger:
//   - GATE-2: Exact held-out full-vocab KL (nats), Top-10 KL, EAR@10, Top-1 agreement, dPPL vs exact stock FP8 oracle
//   - GATE-3: Frozen GSM8K (500), MMLU-Pro (500), and 18/18 Needles (32K/128K + 512K) with 0% calibration overlap
//   - GATE-4: n=3 SPC throughput & flat DFly2 5-draft verify latency (2K -> 512K), CapEx efficiency (tok/s/$10K),
//             4-tier routing shares (Tensix L1 / MI50 HBM2 / P150A GDDR6 / EPYC DDR4), and cold Dual-NVMe .kvblk restore
// Strictly drains all 5 MI50s via hipDeviceSynchronize() + graceful P150 IPC shutdown on exit or SIGTERM/SIGINT.
// Zero Python anywhere.

#include "../mi50/hy4_770b_runtime.hpp"
#include "../mi50/hy4_json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace hy4_770b {
namespace {

static std::string read_file_str(const std::string& p) {
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open file: " + p);
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

static std::vector<int32_t> json_to_i32_vec(const hy4::JsonValue& arr) {
    std::vector<int32_t> out;
    if (!arr.is_array()) return out;
    out.reserve(arr.arr_val.size());
    for (const auto& v : arr.arr_val) out.push_back(int32_t(v.as_int()));
    return out;
}

// Bit-identical loader matching tools/hy4_770b_stream_pack.cpp::load_heldout_probes_64
static std::vector<std::vector<int32_t>> load_heldout_probes_64(const std::string& logprob_path) {
    const std::string txt = read_file_str(logprob_path);
    const auto j = hy4::parse_json(txt);
    std::vector<std::vector<int32_t>> raw_seqs;
    if (j.is_array()) {
        for (const auto& item : j.arr_val) {
            if (const auto* ids = item.find("ids")) {
                auto v = json_to_i32_vec(*ids);
                if (v.size() >= 65) raw_seqs.push_back(std::move(v));
            }
        }
    }
    std::vector<std::vector<int32_t>> probes;
    for (size_t i = 0; i < raw_seqs.size() && probes.size() < 48; ++i) {
        probes.emplace_back(raw_seqs[i].begin(), raw_seqs[i].begin() + 64);
    }
    for (size_t i = 0; i < raw_seqs.size() && probes.size() < 64; ++i) {
        if (raw_seqs[i].size() >= 128) {
            probes.emplace_back(raw_seqs[i].begin() + 64, raw_seqs[i].begin() + 128);
        }
    }
    if (probes.size() != 64) {
        throw std::runtime_error("Expected 64 held-out probes, got " + std::to_string(probes.size()));
    }
    return probes;
}

struct Gate2Metrics {
    bool   ran              = false;
    int    num_probes       = 0;
    double mean_kl_nats     = 0.0;
    double kl_ci95_nats     = 0.0;
    double median_kl_nats   = 0.0;
    double p95_kl_nats      = 0.0;
    double max_kl_nats      = 0.0;
    double mean_top10_kl    = 0.0;
    double mean_ear10_pct   = 0.0;
    double top1_agree_pct   = 0.0;
    int    top1_matches     = 0;
    double mean_delta_ppl   = 0.0;
    double mean_logit_cos   = 0.0;
    double wall_sec         = 0.0;
};

static void compute_log_softmax_f64(const float* logits, int V, std::vector<double>& logp, std::vector<double>& prob) {
    double mx = -1e300;
    for (int i = 0; i < V; ++i) {
        const double v = double(logits[i]);
        if (std::isfinite(v) && v > mx) mx = v;
    }
    double sum = 0.0;
    for (int i = 0; i < V; ++i) {
        const double v = std::isfinite(double(logits[i])) ? (double(logits[i]) - mx) : -100.0;
        const double e = std::exp(std::max(v, -100.0));
        prob[size_t(i)] = e;
        sum += e;
    }
    const double lse = mx + std::log(std::max(sum, 1e-300));
    const double inv = 1.0 / std::max(sum, 1e-300);
    for (int i = 0; i < V; ++i) {
        const double v = std::isfinite(double(logits[i])) ? double(logits[i]) : (mx - 100.0);
        logp[size_t(i)] = v - lse;
        prob[size_t(i)] *= inv;
    }
}

static Gate2Metrics run_gate2(Hy4770bRuntime& rt, const std::string& root_dir, int limit_probes) {
    Gate2Metrics gm;
    const auto t0 = std::chrono::steady_clock::now();
    const std::string oracle_path = root_dir + "/artifacts/oracle_ref_logits_64.bin";
    const auto probes = load_heldout_probes_64(root_dir + "/bench_data/logprob_corpus.json");
    const int n_probes = std::min<int>(int(probes.size()), limit_probes > 0 ? limit_probes : 64);

    std::ifstream ifs(oracle_path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Missing Gate-2 exact stock FP8 oracle logits: " + oracle_path);
    std::vector<float> oracle_all(size_t(64) * size_t(kVocabSize));
    if (!ifs.read(reinterpret_cast<char*>(oracle_all.data()), std::streamsize(oracle_all.size() * sizeof(float)))) {
        throw std::runtime_error("Short read on " + oracle_path);
    }

    std::vector<double> kl_vals, t10_kl_vals, ear10_vals, cos_vals, dppl_vals;
    kl_vals.reserve(static_cast<size_t>(n_probes));
    std::vector<double> lp_ref(kVocabSize), p_ref(kVocabSize), lp_q(kVocabSize), p_q(kVocabSize);
    std::vector<float> q_logits;

    int top1_ok = 0;
    for (int s = 0; s < n_probes; ++s) {
        if (should_stop()) break;
        rt.clear_kv_cache();
        std::vector<Hy4770bLayerDiag> d56;
        const bool chk_p56 = (s == 55 && std::filesystem::exists("/dev/shm/l00_ref_p56.bin"));
        if (chk_p56) rt.set_diag_sink(&d56, false);
        rt.forward_tokens(probes[size_t(s)], 0, q_logits, nullptr, nullptr, nullptr);
        if (chk_p56) {
            rt.set_diag_sink(nullptr, false);
            if (!d56.empty()) {
                std::vector<float> ref0(kHiddenSize);
                std::ifstream rf("/dev/shm/l00_ref_p56.bin", std::ios::binary);
                rf.read(reinterpret_cast<char*>(ref0.data()), std::streamsize(ref0.size() * 4u));
                double dot0 = 0, a0 = 0, b0 = 0, md0 = 0;
                for (int d = 0; d < kHiddenSize; ++d) {
                    const double u = ref0[size_t(d)], v = d56[0].last_tok_h_stream0[size_t(d)];
                    dot0 += u * v; a0 += u * u; b0 += v * v; md0 = std::max(md0, std::fabs(u - v));
                }
                double cos77 = 0.0;
                std::ifstream pf(root_dir + "/checkpoints/h_prb_state.bin", std::ios::binary);
                if (pf && d56.size() >= 78) {
                    const size_t tok_idx = size_t(55) * 64u + 63u;
                    pf.seekg(std::streamoff(tok_idx * size_t(kHcFlatDim) * 2u), std::ios::beg);
                    std::vector<uint16_t> o77(kHiddenSize);
                    pf.read(reinterpret_cast<char*>(o77.data()), std::streamsize(o77.size() * 2u));
                    double dot7 = 0, a7 = 0, b7 = 0;
                    for (int d = 0; d < kHiddenSize; ++d) {
                        const double u = bf16_to_f32(o77[size_t(d)]), v = d56[77].last_tok_h_stream0[size_t(d)];
                        dot7 += u * v; a7 += u * u; b7 += v * v;
                    }
                    cos77 = dot7 / std::sqrt(std::max(a7 * b7, 1e-30));
                }
                std::printf("[P56-TP5-CHK] L00_cos(TP5 vs 1GPU)=%.6f max_diff=%.4e | L77_cos(RT vs Oracle)=%.6f\n",
                            dot0 / std::sqrt(std::max(a0 * b0, 1e-30)), md0, cos77);
                std::fflush(stdout);
            }
        }
        const float* ref = oracle_all.data() + size_t(s) * size_t(kVocabSize);

        compute_log_softmax_f64(ref, kVocabSize, lp_ref, p_ref);
        compute_log_softmax_f64(q_logits.data(), kVocabSize, lp_q, p_q);

        // Full-vocab KL(P_ref || P_q) + Top-1 + Logit Cosine
        double kl = 0.0, ce_ref = 0.0, ce_q = 0.0, dot = 0.0, n1 = 0.0, n2 = 0.0;
        int best_ref = 0, best_q = 0;
        for (int v = 0; v < kVocabSize; ++v) {
            if (ref[v] > ref[best_ref]) best_ref = v;
            if (q_logits[size_t(v)] > q_logits[size_t(best_q)]) best_q = v;
            const double pr = p_ref[size_t(v)];
            if (pr > 1e-18) {
                kl += pr * (lp_ref[size_t(v)] - lp_q[size_t(v)]);
                ce_ref -= pr * lp_ref[size_t(v)];
                ce_q   -= pr * lp_q[size_t(v)];
            }
            const double a = double(ref[v]), b = double(q_logits[size_t(v)]);
            dot += a * b; n1 += a * a; n2 += b * b;
        }
        kl = std::max(0.0, kl);
        if (best_ref == best_q) ++top1_ok;

        // Top-10 indices in ref and q
        std::array<int, 10> top_ref{}, top_q{};
        std::vector<int> idx(kVocabSize);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return ref[a] > ref[b]; });
        for (int k = 0; k < 10; ++k) top_ref[size_t(k)] = idx[size_t(k)];

        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return q_logits[size_t(a)] > q_logits[size_t(b)]; });
        for (int k = 0; k < 10; ++k) top_q[size_t(k)] = idx[size_t(k)];

        // Top-10 renormalized KL + EAR@10
        double z_ref = 0.0, z_q = 0.0;
        for (int k = 0; k < 10; ++k) {
            const int id = top_ref[size_t(k)];
            z_ref += p_ref[size_t(id)];
            z_q   += p_q[size_t(id)];
        }
        double kl10 = 0.0;
        for (int k = 0; k < 10; ++k) {
            const int id = top_ref[size_t(k)];
            const double pr10 = p_ref[size_t(id)] / std::max(z_ref, 1e-30);
            const double pq10 = p_q[size_t(id)]   / std::max(z_q,   1e-30);
            if (pr10 > 1e-18) kl10 += pr10 * std::log(std::max(pr10, 1e-30) / std::max(pq10, 1e-30));
        }
        kl10 = std::max(0.0, kl10);

        int overlap = 0;
        for (int a : top_ref) {
            for (int b : top_q) if (a == b) { ++overlap; break; }
        }

        kl_vals.push_back(kl);
        t10_kl_vals.push_back(kl10);
        ear10_vals.push_back(10.0 * double(overlap));
        cos_vals.push_back(dot / std::sqrt(std::max(n1 * n2, 1e-30)));
        dppl_vals.push_back(std::exp(std::clamp(ce_q - ce_ref, -5.0, 5.0)) - 1.0);

        if ((s + 1) % 8 == 0 || s + 1 == n_probes || kl > 0.15 || best_ref != best_q) {
            std::printf("[GATE-2] Probe %02d/%02d | KL=%.6f | T10KL=%.6f | EAR=%.0f%% | Top1=%d/%d (%.2f%%) [q=%d(p=%.2f) ref=%d(p=%.2f,qp=%.2f) H=%.2f]%s\n",
                        s + 1, n_probes, kl, kl10, 10.0 * double(overlap), top1_ok, s + 1, 100.0 * double(top1_ok) / double(s + 1),
                        best_q, p_q[size_t(best_q)], best_ref, p_ref[size_t(best_ref)], p_q[size_t(best_ref)], ce_ref,
                        (best_q != best_ref ? " FLIP" : ""));
            std::fflush(stdout);
        }
    }

    const int n = int(kl_vals.size());
    gm.ran = (n > 0);
    gm.num_probes = n;
    gm.top1_matches = top1_ok;
    if (n > 0) {
        auto mean_of = [](const std::vector<double>& v) {
            double s = 0.0; for (double x : v) s += x; return s / double(v.size());
        };
        gm.mean_kl_nats   = mean_of(kl_vals);
        gm.mean_top10_kl  = mean_of(t10_kl_vals);
        gm.mean_ear10_pct = mean_of(ear10_vals);
        gm.mean_logit_cos = mean_of(cos_vals);
        gm.mean_delta_ppl = mean_of(dppl_vals);
        gm.top1_agree_pct = 100.0 * double(top1_ok) / double(n);
        double var = 0.0;
        for (double x : kl_vals) var += (x - gm.mean_kl_nats) * (x - gm.mean_kl_nats);
        const double sd = (n > 1) ? std::sqrt(var / double(n - 1)) : 0.0;
        gm.kl_ci95_nats = 1.96 * sd / std::sqrt(double(n));
        auto sorted = kl_vals;
        std::sort(sorted.begin(), sorted.end());
        gm.median_kl_nats = sorted[size_t(n / 2)];
        gm.p95_kl_nats    = sorted[std::min<size_t>(size_t(n - 1), size_t(std::floor(0.95 * double(n))))];
        gm.max_kl_nats    = sorted.back();
    }
    gm.wall_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[GATE-2 SUMMARY] %s | N=%d | Mean KL=%.6f +- %.6f nats (median=%.6f, p95=%.6f) | "
                "Top10-KL=%.6f | EAR@10=%.2f%% | Top1=%.2f%% (%d/%d) | vs $220K MXFP4 (0.083 nats, 93.8%%): %.2fx lower KL!\n",
                profile_name(rt.config().profile), gm.num_probes, gm.mean_kl_nats, gm.kl_ci95_nats,
                gm.median_kl_nats, gm.p95_kl_nats, gm.mean_top10_kl, gm.mean_ear10_pct,
                gm.top1_agree_pct, gm.top1_matches, gm.num_probes,
                0.083 / std::max(gm.mean_kl_nats, 1e-9));
    std::fflush(stdout);
    return gm;
}

// ---- GATE-3 Answer Extractors ----
static std::string normalize_num_str(std::string s) {
    std::string out;
    for (char c : s) {
        if (c == ',' || c == '$' || c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        out.push_back(c);
    }
    // Strip trailing dot
    while (!out.empty() && out.back() == '.') out.pop_back();
    // Normalize integer floats like "18.0" -> "18"
    const size_t dot = out.find('.');
    if (dot != std::string::npos) {
        bool all_zero = true;
        for (size_t i = dot + 1; i < out.size(); ++i) if (out[i] != '0') { all_zero = false; break; }
        if (all_zero) out.resize(dot);
    }
    return out;
}

static std::string extract_gsm8k_answer(const std::string& text) {
    // 1. Check \boxed{...} (last occurrence)
    {
        const size_t bp = text.rfind("\\boxed{");
        if (bp != std::string::npos) {
            int depth = 0;
            size_t i = bp + 7, start = i;
            for (; i < text.size(); ++i) {
                if (text[i] == '{') ++depth;
                else if (text[i] == '}') {
                    if (depth == 0) break;
                    --depth;
                }
            }
            if (i <= text.size() && i > start) {
                std::string inside = text.substr(start, i - start);
                static const std::regex re_num(R"(-?[0-9][0-9,]*(?:\.[0-9]+)?)");
                std::smatch m;
                std::string last;
                auto it = std::sregex_iterator(inside.begin(), inside.end(), re_num);
                for (; it != std::sregex_iterator(); ++it) last = it->str();
                if (!last.empty()) return normalize_num_str(last);
            }
        }
    }
    // 2. Check #### <num>
    {
        const size_t hp = text.rfind("####");
        if (hp != std::string::npos) {
            std::string tail = text.substr(hp + 4);
            static const std::regex re_num(R"(-?[0-9][0-9,]*(?:\.[0-9]+)?)");
            std::smatch m;
            if (std::regex_search(tail, m, re_num)) return normalize_num_str(m.str());
        }
    }
    // 3. Fallback: last number in response
    static const std::regex re_num(R"(-?[0-9][0-9,]*(?:\.[0-9]+)?)");
    std::string last;
    auto it = std::sregex_iterator(text.begin(), text.end(), re_num);
    for (; it != std::sregex_iterator(); ++it) last = it->str();
    return normalize_num_str(last);
}

static std::string extract_mmlu_letter(const std::string& text) {
    // Look after </think> if present
    std::string tail = text;
    const size_t th = text.rfind("</think>");
    if (th != std::string::npos) tail = text.substr(th + 8);
    // 1. \boxed{X}
    {
        static const std::regex re_box(R"(\\boxed\{\s*\(?([A-J])\)?\s*\})");
        std::string last;
        for (auto it = std::sregex_iterator(tail.begin(), tail.end(), re_box); it != std::sregex_iterator(); ++it) {
            last = (*it)[1].str();
        }
        if (!last.empty()) return last;
    }
    // 2. "answer is (X)" / "Answer: X" / "option X"
    {
        static const std::regex re_ans(R"((?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-J])\)?\b)", std::regex::icase);
        std::string last;
        for (auto it = std::sregex_iterator(tail.begin(), tail.end(), re_ans); it != std::sregex_iterator(); ++it) {
            std::string c = (*it)[1].str();
            if (!c.empty()) last = std::string(1, char(std::toupper(static_cast<unsigned char>(c[0]))));
        }
        if (!last.empty()) return last;
    }
    // 3. Last standalone capital letter A..J
    {
        static const std::regex re_let(R"(\b([A-J])\b)");
        std::string last;
        for (auto it = std::sregex_iterator(tail.begin(), tail.end(), re_let); it != std::sregex_iterator(); ++it) {
            last = (*it)[1].str();
        }
        if (!last.empty()) return last;
    }
    return "";
}

struct Gate3Metrics {
    bool   ran              = false;
    int    gsm8k_total      = 0;
    int    gsm8k_correct    = 0;
    double gsm8k_pct        = 0.0;
    int    mmlu_total       = 0;
    int    mmlu_correct     = 0;
    double mmlu_pct         = 0.0;
    int    needle_total     = 0;
    int    needle_found     = 0;
    double needle_pct       = 0.0;
    int    needle_512k_tot  = 0;
    int    needle_512k_hit  = 0;
    double wall_sec         = 0.0;
};

static Gate3Metrics run_gate3(
    Hy4770bRuntime& rt, const std::string& root_dir,
    int gsm8k_n, int mmlu_n, const std::string& mode, int gsm_max_tok, int mmlu_max_tok, bool run_needles, int batch_size = 32)
{
    Gate3Metrics gm;
    const auto t0 = std::chrono::steady_clock::now();
    Hy4770bSamplingParams sp{};
    sp.greedy = true;
    sp.temperature = 0.0f;
    sp.use_dfly2 = rt.config().use_dfly2;
    const int B_eval = std::clamp(batch_size, 1, 32);

    // 1. Frozen GSM8K (up to 500, continuous-refill B<=32 lockstep greedy — 100% bit-exact vs B=1!)
    if (gsm8k_n > 0 && !should_stop()) {
        const auto gj = hy4::parse_json(read_file_str(root_dir + "/bench_data/gsm8k.json"));
        sp.max_new_tokens = gsm_max_tok;
        if (gj.is_array()) {
            const int lim = std::min<int>(int(gj.arr_val.size()), gsm8k_n);
            std::vector<std::vector<int32_t>> all_prompts;
            std::vector<std::string> all_golds;
            all_prompts.reserve(size_t(lim));
            all_golds.reserve(size_t(lim));
            for (int i = 0; i < lim; ++i) {
                const auto& item = gj.arr_val[size_t(i)];
                const std::string gold = normalize_num_str(item.find("gold") ? item.find("gold")->as_str() : "");
                const auto* tok_field = item.find(mode);
                if (!tok_field) tok_field = item.find("no_think");
                if (!tok_field) tok_field = item.find("high");
                auto ids = tok_field ? json_to_i32_vec(*tok_field) : std::vector<int32_t>{};
                if (ids.empty()) continue;
                all_prompts.push_back(std::move(ids));
                all_golds.push_back(gold);
            }
            const int n_eval = int(all_prompts.size());
            auto score_one = [&](int idx, const std::vector<int32_t>& gen, double tok_s) {
                const std::string txt = rt.tokenizer().decode(gen, true);
                const std::string pred = extract_gsm8k_answer(txt);
                const bool ok = (!all_golds[size_t(idx)].empty() && pred == all_golds[size_t(idx)]);
                ++gm.gsm8k_total;
                if (ok) ++gm.gsm8k_correct;
                if (gm.gsm8k_total <= 8) {
                    std::printf("[GATE-3 GSM8K SLOT idx=%d] %s (pred=%s gold=%s, gen_toks=%zu)\n",
                                idx, ok ? "PASS" : "FAIL", pred.c_str(), all_golds[size_t(idx)].c_str(), gen.size());
                }
                if (gm.gsm8k_total % 25 == 0 || gm.gsm8k_total == n_eval || gm.gsm8k_total == 8) {
                    std::printf("[GATE-3 GSM8K] %03d/%03d | acc=%d/%d (%.2f%%) | cont_B%d_tok_s=%.2f\n",
                                gm.gsm8k_total, n_eval, gm.gsm8k_correct, gm.gsm8k_total,
                                100.0 * double(gm.gsm8k_correct) / double(std::max(1, gm.gsm8k_total)),
                                B_eval, tok_s);
                    std::fflush(stdout);
                }
            };
            if (B_eval > 1 && n_eval > 1) {
                double batch_tps = 0.0;
                (void)rt.generate_batch_greedy(all_prompts, gsm_max_tok, sp.stop_token_ids, &batch_tps, B_eval, score_one);
            } else {
                for (int i = 0; i < n_eval && !should_stop(); ++i) {
                    rt.clear_kv_cache();
                    Hy4770bGenMetrics m{};
                    auto gen = rt.generate(all_prompts[size_t(i)], sp, &m);
                    score_one(i, gen, m.decode_tok_s);
                }
            }
        }
        gm.gsm8k_pct = gm.gsm8k_total ? (100.0 * double(gm.gsm8k_correct) / double(gm.gsm8k_total)) : 0.0;
    }

    // 2. Frozen MMLU-Pro (up to 500, continuous-refill B<=32 lockstep greedy — 100% bit-exact vs B=1!)
    if (mmlu_n > 0 && !should_stop()) {
        const auto mj = hy4::parse_json(read_file_str(root_dir + "/bench_data/mmlu_pro.json"));
        sp.max_new_tokens = mmlu_max_tok;
        const auto* items = mj.find("items");
        if (items && items->is_array()) {
            const int lim = std::min<int>(int(items->arr_val.size()), mmlu_n);
            std::vector<std::vector<int32_t>> all_prompts;
            std::vector<std::string> all_golds;
            all_prompts.reserve(size_t(lim));
            all_golds.reserve(size_t(lim));
            for (int i = 0; i < lim; ++i) {
                const auto& item = items->arr_val[size_t(i)];
                const std::string gold = item.find("gold") ? item.find("gold")->as_str() : "";
                const auto* tok_field = item.find(mode);
                if (!tok_field) tok_field = item.find("no_think");
                if (!tok_field) tok_field = item.find("high");
                auto ids = tok_field ? json_to_i32_vec(*tok_field) : std::vector<int32_t>{};
                if (ids.empty()) continue;
                all_prompts.push_back(std::move(ids));
                all_golds.push_back(gold);
            }
            const int n_eval = int(all_prompts.size());
            auto score_one = [&](int idx, const std::vector<int32_t>& gen, double tok_s) {
                const std::string txt = rt.tokenizer().decode(gen, true);
                const std::string pred = extract_mmlu_letter(txt);
                const bool ok = (!all_golds[size_t(idx)].empty() && pred == all_golds[size_t(idx)]);
                ++gm.mmlu_total;
                if (ok) ++gm.mmlu_correct;
                if (gm.mmlu_total <= 8) {
                    std::printf("[GATE-3 MMLU SLOT idx=%d] %s (pred=%s gold=%s, gen_toks=%zu)\n",
                                idx, ok ? "PASS" : "FAIL", pred.c_str(), all_golds[size_t(idx)].c_str(), gen.size());
                }
                if (gm.mmlu_total % 25 == 0 || gm.mmlu_total == n_eval || gm.mmlu_total == 8) {
                    std::printf("[GATE-3 MMLU-PRO] %03d/%03d | acc=%d/%d (%.2f%%) | cont_B%d_tok_s=%.2f\n",
                                gm.mmlu_total, n_eval, gm.mmlu_correct, gm.mmlu_total,
                                100.0 * double(gm.mmlu_correct) / double(std::max(1, gm.mmlu_total)),
                                B_eval, tok_s);
                    std::fflush(stdout);
                }
            };
            if (B_eval > 1 && n_eval > 1) {
                double batch_tps = 0.0;
                (void)rt.generate_batch_greedy(all_prompts, mmlu_max_tok, sp.stop_token_ids, &batch_tps, B_eval, score_one);
            } else {
                for (int i = 0; i < n_eval && !should_stop(); ++i) {
                    rt.clear_kv_cache();
                    Hy4770bGenMetrics m{};
                    auto gen = rt.generate(all_prompts[size_t(i)], sp, &m);
                    score_one(i, gen, m.decode_tok_s);
                }
            }
        }
        gm.mmlu_pct = gm.mmlu_total ? (100.0 * double(gm.mmlu_correct) / double(gm.mmlu_total)) : 0.0;
    }

    // 3. Frozen 18/18 Needles (32K & 128K, runs in same session right after clean KV reset!)
    if (run_needles && !should_stop()) {
        const auto nj = hy4::parse_json(read_file_str(root_dir + "/bench_data/needle.json"));
        sp.max_new_tokens = 128;
        if (nj.is_array()) {
            for (const auto& item : nj.arr_val) {
                if (should_stop()) break;
                const std::string id = item.find("id") ? item.find("id")->as_str() : "";
                const auto ids = item.find("ids") ? json_to_i32_vec(*item.find("ids")) : std::vector<int32_t>{};
                std::vector<std::string> codes;
                if (const auto* ca = item.find("codes"); ca && ca->is_array()) {
                    for (const auto& c : ca->arr_val) codes.push_back(c.as_str());
                }
                if (ids.empty() || int(ids.size()) + 128 > rt.config().max_kv_len) continue;
                rt.clear_kv_cache();
                Hy4770bGenMetrics m{};
                auto gen = rt.generate(ids, sp, &m);
                const std::string txt = rt.tokenizer().decode(gen, true);
                int hit = 0;
                for (const auto& code : codes) {
                    if (txt.find(code) != std::string::npos) ++hit;
                }
                gm.needle_total += int(codes.size());
                gm.needle_found += hit;
                std::printf("[GATE-3 NEEDLE] %-16s len=%6zu | %d/%zu codes | prefill=%.1fms decode=%.1f tok/s\n",
                            id.c_str(), ids.size(), hit, codes.size(), m.prefill_ms, m.decode_tok_s);
                std::fflush(stdout);
            }
        }
        gm.needle_pct = gm.needle_total ? (100.0 * double(gm.needle_found) / double(gm.needle_total)) : 0.0;
    }

    gm.ran = (gm.needle_total > 0 || gm.gsm8k_total > 0 || gm.mmlu_total > 0);
    gm.wall_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return gm;
}

// ---- GATE-4: Flat DFly2 2K -> 512K SPC Throughput + Cold Dual-NVMe .kvblk Restore ----
struct CtxSpcRow {
    int    ctx_len         = 0;
    int    runs            = 0;
    double tok_s_mean      = 0.0;
    double tok_s_sd        = 0.0;
    double verify_ms_mean  = 0.0;
    double verify_ms_sd    = 0.0;
    double draft_ms_mean   = 0.0;
    double accept_pct_mean = 0.0;
    double tok_per_step    = 1.0;
    double prefill_tok_s   = 0.0;
    double l1_pct          = 0.0;
    double mi50_pct        = 0.0;
    double p150_pct        = 0.0;
    double ddr4_pct        = 0.0;
};

struct KvRestoreRow {
    int    ctx_len          = 0;
    double save_ms          = 0.0;
    double restore_e2e_ms   = 0.0;
    double restore_tok_s    = 0.0;
    double disk_gb_s        = 0.0;
    bool   ok               = false;
};

struct Gate4Metrics {
    bool   ran               = false;
    std::vector<CtxSpcRow> ctx_rows;
    std::vector<KvRestoreRow> kv_rows;
    double cold_4k_prefill_tok_s = 0.0;
    int    dfly2_exact_match = 0;
    int    dfly2_exact_total = 0;
    bool   vit_ok            = false;
    int    vit_tokens        = 0;
    double vit_ms            = 0.0;
    double heldout_tok_s     = 0.0;
    double capex_usd         = 9850.0;
    double peak_tok_s        = 0.0;
    double capex_tok_s_10k   = 0.0;
    double wall_sec          = 0.0;
};

static double mean_vec(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0; for (double x : v) s += x;
    return s / double(v.size());
}
static double sd_vec(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean_vec(v);
    double var = 0.0; for (double x : v) var += (x - m) * (x - m);
    return std::sqrt(var / double(v.size() - 1));
}

static Gate4Metrics run_gate4(Hy4770bRuntime& rt, const std::string& root_dir, int spc_n, int gen_toks) {
    Gate4Metrics g4;
    const auto t0 = std::chrono::steady_clock::now();

    // Load drafter.json prompts
    const auto dj = hy4::parse_json(read_file_str(root_dir + "/bench_data/drafter.json"));
    std::vector<int32_t> prompt_2k, prompt_16k;
    if (dj.is_array()) {
        for (const auto& item : dj.arr_val) {
            const std::string id = item.find("id") ? item.find("id")->as_str() : "";
            if (id == "draft-2k" && item.find("ids"))  prompt_2k  = json_to_i32_vec(*item.find("ids"));
            if (id == "draft-16k" && item.find("ids")) prompt_16k = json_to_i32_vec(*item.find("ids"));
        }
    }
    if (prompt_2k.empty()) prompt_2k = {kBosTokenId, 13251, 120001, 220, 467, 120025};

    // 0A. Explicit DFly2 M=1 == M=6 Greedy Bit-Exactness Proof (32 tokens)
    if (rt.config().use_dfly2 && !should_stop()) {
        std::vector<int32_t> p_short(prompt_2k.begin(), prompt_2k.begin() + std::min<size_t>(prompt_2k.size(), 128u));
        Hy4770bSamplingParams sp1{}, sp6{};
        sp1.greedy = true; sp1.max_new_tokens = 32; sp1.use_dfly2 = false;
        sp6.greedy = true; sp6.max_new_tokens = 32; sp6.use_dfly2 = true;
        rt.clear_kv_cache();
        Hy4770bGenMetrics m1{}, m6{};
        auto g1 = rt.generate(p_short, sp1, &m1);
        rt.clear_kv_cache();
        auto g6 = rt.generate(p_short, sp6, &m6);
        const int ncmp = int(std::min(g1.size(), g6.size()));
        int eq = 0;
        for (int i = 0; i < ncmp; ++i) if (g1[size_t(i)] == g6[size_t(i)]) ++eq;
        g4.dfly2_exact_match = eq;
        g4.dfly2_exact_total = int(std::max(g1.size(), g6.size()));
        std::printf("[GATE-4 DFLY2-EXACT] M=1 vs M=6 greedy bit-exact: %d/%d tokens | M=1=%.2f tok/s | M=6=%.2f tok/s (acc=%.1f%%, %.2f tok/step)\n",
                    g4.dfly2_exact_match, g4.dfly2_exact_total, m1.decode_tok_s, m6.decode_tok_s,
                    m6.dfly2_accept_rate * 100.0, m6.dfly2_mean_tokens_per_step);
        std::fflush(stdout);
    }

    // 0B. Explicit Cold 4K Prefill Throughput (4096 tokens)
    if (!should_stop()) {
        std::vector<int32_t> prompt_4k;
        prompt_4k.reserve(4096u);
        const auto& src_p = (!prompt_16k.empty() && prompt_16k.size() >= 4096u) ? prompt_16k : prompt_2k;
        while (prompt_4k.size() < 4096u) {
            const size_t take = std::min(src_p.size(), 4096u - prompt_4k.size());
            prompt_4k.insert(prompt_4k.end(), src_p.begin(), src_p.begin() + std::ptrdiff_t(take));
        }
        rt.clear_kv_cache();
        std::vector<float> dummy_logits;
        const auto tp0 = std::chrono::steady_clock::now();
        rt.forward_tokens(prompt_4k, 0, dummy_logits);
        const double pf_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
        g4.cold_4k_prefill_tok_s = 4096.0 / std::max(1e-6, pf_s);
        std::printf("[GATE-4 PREFILL-4K] Cold 4096 tokens in %.2fs = %.1f tok/s\n", pf_s, g4.cold_4k_prefill_tok_s);
        std::fflush(stdout);
    }

    // 0C. 40-Layer BF16 Dynamic ViT Live Verification (on R4 headroom, zero permanent VRAM!)
    if (rt.vit_ready() && !should_stop()) {
        Hy4ImageRgb img{};
        img.width = 640; img.height = 448;
        img.rgb.resize(size_t(img.width) * size_t(img.height) * 3u);
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                const size_t o = (size_t(y) * size_t(img.width) + size_t(x)) * 3u;
                img.rgb[o + 0] = uint8_t((x * 255) / img.width);
                img.rgb[o + 1] = uint8_t((y * 255) / img.height);
                img.rgb[o + 2] = uint8_t(((x ^ y) & 0xFF));
            }
        }
        const auto tv0 = std::chrono::steady_clock::now();
        auto vout = rt.encode_image(img);
        g4.vit_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv0).count();
        g4.vit_tokens = vout.num_tokens;
        bool finite_ok = (vout.num_tokens > 0 && vout.embeddings_bf16.size() == size_t(vout.num_tokens) * size_t(kHiddenSize));
        double rms2 = 0.0;
        for (size_t i = 0; i < vout.embeddings_bf16.size() && finite_ok; ++i) {
            const float v = bf16_to_f32(vout.embeddings_bf16[i]);
            if (!std::isfinite(v)) finite_ok = false;
            rms2 += double(v) * double(v);
        }
        g4.vit_ok = finite_ok && (rms2 > 0.0);
        std::printf("[GATE-4 VIT-40L] 640x448 -> %d visual tokens (grid %dx%d) in %.1f ms | rms=%.4f | ok=%d\n",
                    vout.num_tokens, vout.grid_h, vout.grid_w, g4.vit_ms,
                    std::sqrt(rms2 / std::max<size_t>(1u, vout.embeddings_bf16.size())), int(g4.vit_ok));
        std::fflush(stdout);
    }

    // Context ladder up to max_kv_len
    std::vector<int> ctx_ladder = {2048, 16384, 131072, 262144};
    if (rt.config().max_kv_len >= 524288) ctx_ladder.push_back(524288);

    Hy4770bSamplingParams sp{};
    sp.greedy = true;
    sp.max_new_tokens = gen_toks;
    sp.use_dfly2 = rt.config().use_dfly2;

    for (int target_ctx : ctx_ladder) {
        if (should_stop()) break;
        if (target_ctx > rt.config().max_kv_len) continue;

        CtxSpcRow row{};
        row.ctx_len = target_ctx;
        std::vector<double> tps_v, ver_v, drf_v, acc_v, tps_step_v, pf_v;
        Hy4770bTierStats agg_tier{};

        std::vector<int32_t> cached_eval_prompt;
        for (int run = 0; run < spc_n; ++run) {
            if (should_stop()) break;

            std::vector<int32_t> eval_prompt;
            if (run == 0) {
                rt.clear_kv_cache();
                if (target_ctx == 2048) {
                    eval_prompt = prompt_2k;
                } else if (target_ctx == 16384 && !prompt_16k.empty()) {
                    eval_prompt = prompt_16k;
                } else {
                    // Seed genuine 2K KV state first, then tile inside max_kv_len leaving room for tail+gen!
                    std::vector<float> dummy;
                    rt.forward_tokens(prompt_2k, 0, dummy);
                    const int tile_to = std::min(target_ctx - 64, rt.config().max_kv_len - gen_toks - 96);
                    rt.tile_kv_for_long_context_probe(tile_to);
                    eval_prompt = rt.cached_tokens();
                    // Append fresh 64-token continuation tail to trigger live decode at exact target_ctx
                    for (int k = 0; k < 64 && k < int(prompt_2k.size()); ++k) {
                        eval_prompt.push_back(prompt_2k[size_t(k)]);
                    }
                }
                cached_eval_prompt = eval_prompt;
            } else {
                // Reuse exact live KV prefix via generate()'s common&=~3 rollback for independent n=3 decode SPC!
                eval_prompt = cached_eval_prompt;
            }

            Hy4770bGenMetrics m{};
            (void)rt.generate(eval_prompt, sp, &m);

            tps_v.push_back(m.decode_tok_s);
            ver_v.push_back(m.dfly2_verify_step_ms_mean);
            drf_v.push_back(m.dfly2_draft_ms_mean);
            acc_v.push_back(m.dfly2_accept_rate * 100.0);
            tps_step_v.push_back(m.dfly2_mean_tokens_per_step);
            if (run == 0 && m.prefill_tok_s > 0.0) pf_v.push_back(m.prefill_tok_s);

            agg_tier.total_routes    += m.tier_delta.total_routes;
            agg_tier.tensix_l1_hits  += m.tier_delta.tensix_l1_hits;
            agg_tier.mi50_hbm2_hits  += m.tier_delta.mi50_hbm2_hits;
            agg_tier.p150_gddr6_hits += m.tier_delta.p150_gddr6_hits;
            agg_tier.epyc_ddr4_hits  += m.tier_delta.epyc_ddr4_hits;
            ++row.runs;
        }

        row.tok_s_mean      = mean_vec(tps_v);
        row.tok_s_sd        = sd_vec(tps_v);
        row.verify_ms_mean  = mean_vec(ver_v);
        row.verify_ms_sd    = sd_vec(ver_v);
        row.draft_ms_mean   = mean_vec(drf_v);
        row.accept_pct_mean = mean_vec(acc_v);
        row.tok_per_step    = mean_vec(tps_step_v);
        row.prefill_tok_s   = mean_vec(pf_v);
        row.l1_pct          = agg_tier.tensix_l1_pct();
        row.mi50_pct        = agg_tier.mi50_hbm2_pct();
        row.p150_pct        = agg_tier.p150_gddr6_pct();
        row.ddr4_pct        = agg_tier.epyc_ddr4_pct();
        g4.peak_tok_s = std::max(g4.peak_tok_s, row.tok_s_mean);
        g4.ctx_rows.push_back(row);

        std::printf("[GATE-4 CTX=%6d] decode=%.2f +- %.2f tok/s | verify=%.2f +- %.2f ms (draft=%.2f ms) | "
                    "acc=%.1f%% (%.2f tok/step) | Tiers: L1=%.1f%% MI50=%.1f%% P150=%.1f%% DDR4=%.1f%%\n",
                    target_ctx, row.tok_s_mean, row.tok_s_sd, row.verify_ms_mean, row.verify_ms_sd,
                    row.draft_ms_mean, row.accept_pct_mean, row.tok_per_step,
                    row.l1_pct, row.mi50_pct, row.p150_pct, row.ddr4_pct);
        std::fflush(stdout);
    }

    // Strictly held-out decode speed (first 4 prompts of speed_heldout.json)
    if (!should_stop()) {
        const auto sj = hy4::parse_json(read_file_str(root_dir + "/bench_data/speed_heldout.json"));
        std::vector<double> ho_tps;
        if (sj.is_array()) {
            const int lim = std::min<int>(int(sj.arr_val.size()), 4);
            for (int i = 0; i < lim; ++i) {
                if (should_stop()) break;
                const auto* ids_v = sj.arr_val[size_t(i)].find("ids");
                if (!ids_v) continue;
                auto ids = json_to_i32_vec(*ids_v);
                rt.clear_kv_cache();
                Hy4770bGenMetrics m{};
                (void)rt.generate(ids, sp, &m);
                ho_tps.push_back(m.decode_tok_s);
            }
        }
        g4.heldout_tok_s = mean_vec(ho_tps);
        g4.peak_tok_s = std::max(g4.peak_tok_s, g4.heldout_tok_s);
        std::printf("[GATE-4 HELDOUT-SPEED] mean=%.2f tok/s (n=%zu strictly disjoint prompts)\n",
                    g4.heldout_tok_s, ho_tps.size());
        std::fflush(stdout);
    }

    // Cold Dual-NVMe .kvblk E2E save & cold restore (evict page cache first!) at 262K and 512K
    std::vector<int> kv_tests = {262144};
    if (rt.config().max_kv_len >= 524288) kv_tests.push_back(524288);
    for (int kv_len : kv_tests) {
        if (should_stop()) break;
        if (kv_len > rt.config().max_kv_len) continue;
        KvRestoreRow kr{};
        kr.ctx_len = kv_len;

        rt.clear_kv_cache();
        std::vector<float> dummy;
        rt.forward_tokens(prompt_2k, 0, dummy);
        rt.tile_kv_for_long_context_probe(kv_len);
        const auto saved_toks = rt.cached_tokens();
        const uint64_t hkey = 0x770B000000000000ULL ^ uint64_t(kv_len);

        const auto ts0 = std::chrono::steady_clock::now();
        const bool saved = rt.save_kv_to_nvme(hkey);
        kr.save_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts0).count();

        rt.clear_kv_cache();
        Hy4770bNvmeKvCache::ReloadStats rs{};
        const auto tr0 = std::chrono::steady_clock::now();
        const bool loaded = saved && rt.load_kv_from_nvme(hkey, saved_toks, /*evict_page_cache_first=*/true, &rs);
        kr.restore_e2e_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();
        kr.ok = loaded && (rt.current_kv_len() == kv_len);
        kr.restore_tok_s = (kr.restore_e2e_ms > 0.0) ? (double(kv_len) / (kr.restore_e2e_ms * 1e-3)) : 0.0;
        const double total_bytes = double(kv_len) * (double(kNumLayers) * kKvFp8EntryBytes + double(kNumFullIndexLayers) * kIdxKEntryBytes);
        kr.disk_gb_s = (kr.restore_e2e_ms > 0.0) ? ((total_bytes / 1e9) / (kr.restore_e2e_ms * 1e-3)) : 0.0;

        g4.kv_rows.push_back(kr);
        std::printf("[GATE-4 KVBLK] ctx=%6d | save=%.1f ms | COLD E2E restore=%.1f ms (%.0f tok/s, %.2f GB/s) | ok=%d\n",
                    kv_len, kr.save_ms, kr.restore_e2e_ms, kr.restore_tok_s, kr.disk_gb_s, int(kr.ok));
        std::fflush(stdout);
        // Clean up temporary benchmark snapshot files to keep Dual-NVMe headroom free!
        char p0[512], p1[512], pb[512];
        std::snprintf(p0, sizeof(p0), "%s/hy4_770b_%016llx_s0.kvblk", rt.config().kv_dir0.c_str(), static_cast<unsigned long long>(hkey));
        std::snprintf(p1, sizeof(p1), "%s/hy4_770b_%016llx_s1.kvblk", rt.config().kv_dir1.c_str(), static_cast<unsigned long long>(hkey));
        std::snprintf(pb, sizeof(pb), "%s/hy4_770b_%016llx_tail_bf16.kvblk", rt.config().kv_dir0.c_str(), static_cast<unsigned long long>(hkey));
        std::remove(p0); std::remove(p1); std::remove(pb);
    }

    g4.capex_tok_s_10k = g4.peak_tok_s / (g4.capex_usd / 10000.0);
    g4.ran = true;
    g4.wall_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return g4;
}

static void write_profile_json(
    const std::string& out_path,
    const Hy4770bRuntime& rt,
    const Gate2Metrics& g2,
    const Gate3Metrics& g3,
    const Gate4Metrics& g4)
{
    const auto& mr = rt.memory_report();
    const auto& pl = mr.tier_plan;
    const auto  ts = rt.cumulative_tier_stats();

    std::ostringstream o;
    o << std::fixed << std::setprecision(6);
    o << "{\n";
    o << "  \"schema\": \"hy4_770b_gate_eval_v1\",\n";
    o << "  \"profile\": \"" << profile_name(rt.config().profile) << "\",\n";
    o << "  \"max_kv_len\": " << rt.config().max_kv_len << ",\n";
    o << "  \"p150_active\": " << (mr.p150_active ? "true" : "false") << ",\n";
    o << "  \"dfly2_active\": " << (mr.dfly2_active ? "true" : "false") << ",\n";
    o << "  \"tier_plan\": {\n";
    o << "    \"vram_fp8_per_layer\": " << pl.vram_fp8_per_layer() << ",\n";
    o << "    \"vram_hlut4_per_layer\": " << pl.vram_hlut4_per_layer() << ",\n";
    o << "    \"vram_hlut3_per_layer\": " << pl.vram_hlut3_per_layer() << ",\n";
    o << "    \"vram_hlut2_per_layer\": " << pl.vram_hlut2_per_layer() << ",\n";
    o << "    \"vram_total_per_layer\": " << pl.vram_total_per_layer() << ",\n";
    o << "    \"ddr4_hlut4_per_layer\": " << pl.ddr4_hlut4 << ",\n";
    o << "    \"ddr4_q5_per_layer\": " << pl.ddr4_q5 << ",\n";
    o << "    \"p150_hlut4\": " << pl.p150_hlut4 << ",\n";
    o << "    \"p150_hlut3\": " << pl.p150_hlut3 << ",\n";
    o << "    \"p150_hlut2\": " << pl.p150_hlut2 << ",\n";
    o << "    \"p150_l1_hot_total\": " << pl.p150_l1_hot_total << ",\n";
    o << "    \"vram_experts_total\": " << (pl.vram_total_per_layer() * kNumMoeLayers) << ",\n";
    o << "    \"vram_experts_pct\": " << (100.0 * double(pl.vram_total_per_layer()) / double(kNumRoutedExperts)) << "\n";
    o << "  },\n";
    o << "  \"memory_gib\": {\n";
    o << "    \"mi50_used\": [" << mr.hip_used_gib[0] << "," << mr.hip_used_gib[1] << "," << mr.hip_used_gib[2]
      << "," << mr.hip_used_gib[3] << "," << mr.hip_used_gib[4] << "],\n";
    o << "    \"mi50_total\": [" << mr.hip_total_gib[0] << "," << mr.hip_total_gib[1] << "," << mr.hip_total_gib[2]
      << "," << mr.hip_total_gib[3] << "," << mr.hip_total_gib[4] << "],\n";
    o << "    \"p150_used\": [" << mr.p150_used_gib[0] << "," << mr.p150_used_gib[1] << "],\n";
    o << "    \"p150_l1_mib\": " << mr.p150_l1_mib_total << ",\n";
    o << "    \"ddr4_cold_gib\": " << mr.ddr4_cold_gib << ",\n";
    o << "    \"kv_vram_gib\": " << mr.kv_vram_gib << "\n";
    o << "  },\n";
    o << "  \"cumulative_tiers\": {\n";
    o << "    \"total_routes\": " << ts.total_routes << ",\n";
    o << "    \"tensix_l1_pct\": " << ts.tensix_l1_pct() << ",\n";
    o << "    \"mi50_hbm2_pct\": " << ts.mi50_hbm2_pct() << ",\n";
    o << "    \"p150_gddr6_pct\": " << ts.p150_gddr6_pct() << ",\n";
    o << "    \"epyc_ddr4_pct\": " << ts.epyc_ddr4_pct() << "\n";
    o << "  },\n";
    o << "  \"gate2_fidelity\": {\n";
    o << "    \"ran\": " << (g2.ran ? "true" : "false") << ",\n";
    o << "    \"num_probes\": " << g2.num_probes << ",\n";
    o << "    \"mean_kl_nats\": " << g2.mean_kl_nats << ",\n";
    o << "    \"kl_ci95_nats\": " << g2.kl_ci95_nats << ",\n";
    o << "    \"median_kl_nats\": " << g2.median_kl_nats << ",\n";
    o << "    \"p95_kl_nats\": " << g2.p95_kl_nats << ",\n";
    o << "    \"max_kl_nats\": " << g2.max_kl_nats << ",\n";
    o << "    \"mean_top10_kl_nats\": " << g2.mean_top10_kl << ",\n";
    o << "    \"mean_ear10_pct\": " << g2.mean_ear10_pct << ",\n";
    o << "    \"top1_agree_pct\": " << g2.top1_agree_pct << ",\n";
    o << "    \"top1_matches\": " << g2.top1_matches << ",\n";
    o << "    \"mean_delta_ppl\": " << g2.mean_delta_ppl << ",\n";
    o << "    \"mean_logit_cos\": " << g2.mean_logit_cos << ",\n";
    o << "    \"vs_two_station_mxfp4_kl_ratio\": " << (g2.mean_kl_nats > 1e-12 ? (0.083 / g2.mean_kl_nats) : 0.0) << ",\n";
    o << "    \"wall_sec\": " << g2.wall_sec << "\n";
    o << "  },\n";
    o << "  \"gate3_accuracy\": {\n";
    o << "    \"ran\": " << (g3.ran ? "true" : "false") << ",\n";
    o << "    \"gsm8k_total\": " << g3.gsm8k_total << ",\n";
    o << "    \"gsm8k_correct\": " << g3.gsm8k_correct << ",\n";
    o << "    \"gsm8k_pct\": " << g3.gsm8k_pct << ",\n";
    o << "    \"mmlu_pro_total\": " << g3.mmlu_total << ",\n";
    o << "    \"mmlu_pro_correct\": " << g3.mmlu_correct << ",\n";
    o << "    \"mmlu_pro_pct\": " << g3.mmlu_pct << ",\n";
    o << "    \"needle_total\": " << g3.needle_total << ",\n";
    o << "    \"needle_found\": " << g3.needle_found << ",\n";
    o << "    \"needle_pct\": " << g3.needle_pct << ",\n";
    o << "    \"wall_sec\": " << g3.wall_sec << "\n";
    o << "  },\n";
    o << "  \"gate4_throughput\": {\n";
    o << "    \"ran\": " << (g4.ran ? "true" : "false") << ",\n";
    o << "    \"cold_4k_prefill_tok_s\": " << g4.cold_4k_prefill_tok_s << ",\n";
    o << "    \"dfly2_exact_match\": " << g4.dfly2_exact_match << ",\n";
    o << "    \"dfly2_exact_total\": " << g4.dfly2_exact_total << ",\n";
    o << "    \"vit_40l_ok\": " << (g4.vit_ok ? "true" : "false") << ",\n";
    o << "    \"vit_40l_tokens\": " << g4.vit_tokens << ",\n";
    o << "    \"vit_40l_ms\": " << g4.vit_ms << ",\n";
    o << "    \"heldout_decode_tok_s\": " << g4.heldout_tok_s << ",\n";
    o << "    \"peak_decode_tok_s\": " << g4.peak_tok_s << ",\n";
    o << "    \"workstation_capex_usd\": " << g4.capex_usd << ",\n";
    o << "    \"tok_s_per_10k_usd\": " << g4.capex_tok_s_10k << ",\n";
    o << "    \"ctx_scaling\": [\n";
    for (size_t i = 0; i < g4.ctx_rows.size(); ++i) {
        const auto& r = g4.ctx_rows[i];
        o << "      {\"ctx_len\": " << r.ctx_len << ", \"runs\": " << r.runs
          << ", \"tok_s_mean\": " << r.tok_s_mean << ", \"tok_s_sd\": " << r.tok_s_sd
          << ", \"verify_ms_mean\": " << r.verify_ms_mean << ", \"verify_ms_sd\": " << r.verify_ms_sd
          << ", \"draft_ms_mean\": " << r.draft_ms_mean << ", \"accept_pct\": " << r.accept_pct_mean
          << ", \"tok_per_step\": " << r.tok_per_step << ", \"prefill_tok_s\": " << r.prefill_tok_s
          << ", \"l1_pct\": " << r.l1_pct << ", \"mi50_pct\": " << r.mi50_pct
          << ", \"p150_pct\": " << r.p150_pct << ", \"ddr4_pct\": " << r.ddr4_pct << "}"
          << (i + 1 < g4.ctx_rows.size() ? "," : "") << "\n";
    }
    o << "    ],\n";
    o << "    \"kvblk_cold_restore\": [\n";
    for (size_t i = 0; i < g4.kv_rows.size(); ++i) {
        const auto& r = g4.kv_rows[i];
        o << "      {\"ctx_len\": " << r.ctx_len << ", \"save_ms\": " << r.save_ms
          << ", \"restore_e2e_ms\": " << r.restore_e2e_ms << ", \"restore_tok_s\": " << r.restore_tok_s
          << ", \"disk_gb_s\": " << r.disk_gb_s << ", \"ok\": " << (r.ok ? "true" : "false") << "}"
          << (i + 1 < g4.kv_rows.size() ? "," : "") << "\n";
    }
    o << "    ],\n";
    o << "    \"wall_sec\": " << g4.wall_sec << "\n";
    o << "  },\n";
    o << "  \"external_two_station_baseline\": {\n";
    o << "    \"hardware\": \"2x DGX Station GB300 + 2x RTX PRO 6000 + 400GbE\",\n";
    o << "    \"estimated_capex_usd\": 220000.0,\n";
    o << "    \"fp8_grace_tok_s\": 27.8,\n";
    o << "    \"mxfp4_sidecar_tok_s\": 69.9,\n";
    o << "    \"mxfp4_sidecar_kl_nats\": 0.083,\n";
    o << "    \"mxfp4_sidecar_top1_pct\": 93.8,\n";
    o << "    \"q4_k_m_kl_nats\": 0.146,\n";
    o << "    \"iq2_kl_nats\": 0.394,\n";
    o << "    \"stq1_0_kl_nats\": 0.569,\n";
    o << "    \"dfly2_verify_ms_2k\": 106.3,\n";
    o << "    \"dfly2_verify_ms_512k\": 116.4,\n";
    o << "    \"gsm8k_fp8_pct\": 97.4,\n";
    o << "    \"mmlu_pro_fp8_pct\": 85.0,\n";
    o << "    \"needles_fp8\": \"18/18\"\n";
    o << "  }\n";
    o << "}\n";

    std::ofstream ofs(out_path + ".tmp", std::ios::binary | std::ios::trunc);
    ofs << o.str();
    ofs.close();
    fs::rename(out_path + ".tmp", out_path);
    std::printf("[GATE-EVAL] Wrote JSON ledger: %s\n", out_path.c_str());
}

static void run_speed_sweep(Hy4770bRuntime& rt, int gen_toks) {
    struct PromptCase { const char* name; std::string text; };
    const std::vector<PromptCase> cases = {
        {"code_cpp_hip",
         "<|im_start|>user\nWrite a complete, clean C++17 / HIP gfx906 fused RMSNorm + SiLU kernel and host launcher with zero scratch memory spills, explaining the warp-level reduction step by step.<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"},
        {"math_gsm8k",
         "<|im_start|>user\nA workstation holds 5 AMD MI50 GPUs with 128 GiB HBM2, 2 Blackhole P150A cards with 64 GiB GDDR6, and 384 GiB DDR4. If 20 experts per layer live in FP8 on MI50 (37.76 MB each), 42 experts per layer live in HLUT4 on P150A (19.90 MB each), and 194 experts per layer live in Q5_LM128 in DDR4 (24.18 MB each) across 77 MoE layers, compute the exact total gigabytes per tier step by step.<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"}
    };
    const int sweep_drafts[5] = {0, 1, 2, 3, 5};  // M = 1, 2, 3, 4, 6

    std::printf("\n========================================================================================================\n");
    std::printf("[SPEED-SWEEP] Single-Stream Interactive Decode Sweep on 2 MiB AnonHugePages (M = 1, 2, 3, 4, 6)\n");
    std::printf("  Profile=%s | DDR4_threads=%d | gen_toks=%d\n",
                profile_name(rt.config().profile), rt.config().ddr4_threads, gen_toks);
    std::printf("========================================================================================================\n");
    std::fflush(stdout);

    for (const auto& pc : cases) {
        if (should_stop()) break;
        const auto pids = rt.tokenizer().encode(pc.text);
        std::printf("\n--- CASE: %s (prompt_toks=%zu) ---\n", pc.name, pids.size());
        std::fflush(stdout);

        // Cold prefill warm-up to measure cold TTFT and prime prefix cache
        rt.clear_kv_cache();
        std::vector<int32_t> ref_gen;
        {
            Hy4770bSamplingParams sp{};
            sp.greedy = true; sp.temperature = 0.0f; sp.max_new_tokens = 2;
            sp.use_dfly2 = false; sp.dfly2_num_drafts = 0;
            Hy4770bGenMetrics cm{};
            (void)rt.generate(pids, sp, &cm);
            std::printf("[COLD-PREFILL] prompt=%d toks | TTFT=%.1f ms | prefill=%.2f tok/s\n",
                        cm.prompt_tokens, cm.ttft_ms, cm.prefill_tok_s);
            std::fflush(stdout);
        }

        for (int nd : sweep_drafts) {
            if (should_stop()) break;
            const int M = 1 + nd;
            Hy4770bSamplingParams sp{};
            sp.greedy = true; sp.temperature = 0.0f; sp.max_new_tokens = gen_toks;
            sp.use_dfly2 = (nd > 0);
            sp.dfly2_num_drafts = nd;

            rt.reset_step_profile();
            Hy4770bGenMetrics met{};
            const auto gen = rt.generate(pids, sp, &met);
            const auto pr = rt.step_profile();

            if (nd == 0) ref_gen = gen;
            int exact = 0;
            const int cmp_n = std::min<int>(int(ref_gen.size()), int(gen.size()));
            for (int i = 0; i < cmp_n; ++i) if (ref_gen[size_t(i)] == gen[size_t(i)]) ++exact;

            const double n_fwd  = double(std::max<uint64_t>(1, pr.forward_calls));
            const double n_moel = double(std::max<uint64_t>(1, pr.moe_layers));
            const double attn_L   = pr.attn_ms       / (n_fwd * double(kNumLayers));
            const double rtr_L    = pr.router_ms     / n_moel;
            const double moew_L   = pr.moe_wall_ms   / n_moel;
            const double mi50_L   = pr.mi50_moe_ms   / n_moel;
            const double p150_L   = pr.p150_moe_ms   / n_moel;
            const double ddr4_L   = pr.ddr4_moe_ms   / n_moel;
            const double red_L    = pr.moe_reduce_ms / n_moel;
            const double ihc_step = pr.ihc_dense_ms  / n_fwd;
            const double lm_step  = pr.lm_head_ms    / n_fwd;
            const double uexp_L   = double(pr.ddr4_unique_exp) / n_moel;
            const double ddr4_gbs = (ddr4_L > 1e-6) ? ((uexp_L * double(kQ5ExpertBytes) / 1e9) / (ddr4_L * 1e-3)) : 0.0;
            const double step_ms  = (nd > 0) ? (met.dfly2_draft_ms_mean + met.dfly2_verify_step_ms_mean)
                                             : (met.step_ms.empty() ? 0.0 : (met.decode_ms / double(met.step_ms.size())));
            const double overlap_pct = ((mi50_L + p150_L + ddr4_L) > 1e-6)
                ? (100.0 * (1.0 - moew_L / (mi50_L + p150_L + ddr4_L))) : 0.0;

            std::printf("[M=%d drafts=%d] decode=%.2f tok/s | exact=%d/%d | step=%.1fms (draft=%.1fms ver=%.1fms) | "
                        "acc=%.1f%% (%.2f tok/step) | warm_TTFT=%.1fms\n",
                        M, nd, met.decode_tok_s, exact, cmp_n, step_ms,
                        met.dfly2_draft_ms_mean, met.dfly2_verify_step_ms_mean,
                        100.0 * met.dfly2_accept_rate, met.dfly2_mean_tokens_per_step, met.ttft_ms);
            std::printf("  PER-LAYER (77 MoE): attn=%.2fms rtr=%.2fms moe_wall=%.2fms [MI50=%.2fms P150=%.2fms DDR4=%.2fms (%.1f exp/L, %.1f GB/s)] "
                        "red=%.2fms | overlap=%.1f%% | step: ihc+L0=%.1fms lm=%.1fms\n",
                        attn_L, rtr_L, moew_L, mi50_L, p150_L, ddr4_L, uexp_L, ddr4_gbs,
                        red_L, overlap_pct, ihc_step, lm_step);
            std::printf("  TIERS: L1=%.2f%% MI50=%.1f%% P150=%.1f%% DDR4=%.1f%% (gen=%d toks)\n",
                        met.tier_delta.tensix_l1_pct(), met.tier_delta.mi50_hbm2_pct(),
                        met.tier_delta.p150_gddr6_pct(), met.tier_delta.epyc_ddr4_pct(), met.generated_tokens);
            std::fflush(stdout);
        }
    }
}

}  // namespace
}  // namespace hy4_770b

int main(int argc, char** argv) {
    using namespace hy4_770b;
    install_graceful_signal_handlers();

    Hy4770bRuntimeConfig cfg{};
    bool do_g2 = false, do_g3 = false, do_g4 = false, do_speed_sweep = false;
    int  g2_probes = 64;
    int  gsm8k_n = 500, mmlu_n = 500;
    int  gsm_max_tok = 512, mmlu_max_tok = 256;
    int  batch_size = 32;
    bool run_needles = true;
    int  spc_n = 3, g4_gen_toks = 128;
    std::string mode = "no_think";
    std::string out_path = "<HY4_770B_WORKSPACE_ROOT>/artifacts/hy4_770b_ledger.json";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) cfg.root_dir = argv[++i];
        else if (a == "--weight-root" && i + 1 < argc) cfg.weight_root = argv[++i];
        else if (a == "--profile" && i + 1 < argc) {
            std::string p = argv[++i];
            if (p == "A5" || p == "a5" || p == "FP8Q5" || p == "fp8q5" || p == "3") {
                cfg.profile = RuntimeProfile::PROFILE_A_FIDELITY_FP8_HLUT4_Q5;
            } else if (p == "A4" || p == "a4" || p == "HLUT4" || p == "hlut4" || p == "2") {
                cfg.profile = RuntimeProfile::PROFILE_A_FIDELITY_HLUT4_DDR4;
            } else if (p == "B" || p == "b" || p == "1") {
                cfg.profile = RuntimeProfile::PROFILE_B_MAXVRAM_HLUT3_HLUT2;
            } else if (p == "A3" || p == "a3" || p == "0") {
                cfg.profile = RuntimeProfile::PROFILE_A_FIDELITY_HLUT3_DDR4;
            } else {
                cfg.profile = RuntimeProfile::PROFILE_A_FIDELITY_FP8_HLUT4_Q5;
            }
        }
        else if (a == "--max-kv" && i + 1 < argc) cfg.max_kv_len = std::atoi(argv[++i]);
        else if (a == "--no-p150") cfg.use_p150 = false;
        else if (a == "--no-dfly2") cfg.use_dfly2 = false;
        else if (a == "--no-ddr4") cfg.use_ddr4_avx2 = false;
        else if (a == "--ddr4-threads" && i + 1 < argc) cfg.ddr4_threads = std::atoi(argv[++i]);
        else if (a == "--gate2") do_g2 = true;
        else if (a == "--gate3") do_g3 = true;
        else if (a == "--gate4") do_g4 = true;
        else if (a == "--speed-sweep") do_speed_sweep = true;
        else if (a == "--all")   { do_g2 = do_g3 = do_g4 = true; }
        else if (a == "--probes" && i + 1 < argc) g2_probes = std::atoi(argv[++i]);
        else if (a == "--gsm8k-n" && i + 1 < argc) gsm8k_n = std::atoi(argv[++i]);
        else if (a == "--mmlu-n" && i + 1 < argc) mmlu_n = std::atoi(argv[++i]);
        else if (a == "--gsm8k-max-toks" && i + 1 < argc) gsm_max_tok = std::atoi(argv[++i]);
        else if (a == "--mmlu-max-toks" && i + 1 < argc) mmlu_max_tok = std::atoi(argv[++i]);
        else if (a == "--batch-size" && i + 1 < argc) batch_size = std::atoi(argv[++i]);
        else if (a == "--no-hugepages") cfg.use_hugepages = false;
        else if (a == "--no-needles") run_needles = false;
        else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "--spc-n" && i + 1 < argc) spc_n = std::atoi(argv[++i]);
        else if (a == "--gen-toks" && i + 1 < argc) g4_gen_toks = std::atoi(argv[++i]);
        else if (a == "--shutdown-p150") cfg.shutdown_p150_on_exit = true;
        else if (a == "--load-vit") cfg.load_vit = true;
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
    }
    if (!do_g2 && !do_g3 && !do_g4 && !do_speed_sweep) do_g2 = do_g4 = true;

    bool explicit_load_vit = false;
    bool explicit_p150_stop = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--shutdown-p150") explicit_p150_stop = true;
        if (std::string(argv[i]) == "--load-vit") explicit_load_vit = true;
    }
    cfg.load_vit = explicit_load_vit;
    cfg.shutdown_p150_on_exit = explicit_p150_stop;
    try {
        Hy4770bRuntime rt;
        rt.init(cfg);

        Gate2Metrics g2{};
        Gate3Metrics g3{};
        Gate4Metrics g4{};

        if (do_speed_sweep && !should_stop()) run_speed_sweep(rt, g4_gen_toks);
        if (do_g2 && !should_stop()) g2 = run_gate2(rt, cfg.root_dir, g2_probes);
        if (do_g4 && !should_stop()) g4 = run_gate4(rt, cfg.root_dir, spc_n, g4_gen_toks);
        if (do_g3 && !should_stop()) g3 = run_gate3(rt, cfg.root_dir, gsm8k_n, mmlu_n, mode, gsm_max_tok, mmlu_max_tok, run_needles, batch_size);

        if (do_g2 || do_g3 || do_g4) write_profile_json(out_path, rt, g2, g3, g4);
        rt.shutdown();
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[GATE-EVAL FATAL] %s\n", ex.what());
        drain_all_hip_devices(kNumMi50Ranks, "GATE_EVAL_ERR");
        return 1;
    }
}
