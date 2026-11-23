// Pure-C++17 Streaming Packer for Hy4-770B Winning Fidelity Profile (ProfileA-Fidelity-FP8-HLUT4-Q5)
//   - Ranks [0..19]   (20/L =  54.15 GiB): Bit-Exact Lossless Stock FP8-g128 -> <HY4_770B_WORKSPACE_ROOT>/fp8_hot/
//   - Ranks [62..255] (194/L = 336.43 GiB): Per-Row 1x128 5-Bit Lloyd-Max (Q5_LM128, 5.125 bpw, ~26.4 dB) -> <HY4_770B_WEIGHT_ROOT>/q5_ddr4/
// Double-buffered async HTTP/2 range streaming + 32-core OpenMP quantization + atomic per-layer resume.
// Zero Python anywhere.

#include "../src/hy4_770b_config.hpp"
#include "../src/hy4_770b_graceful.hpp"
#include "../src/hy4_770b_hlut_pack.hpp"
#include "../src/hy4_770b_range_stream.hpp"
#include "../src/safetensors_mmap.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace hy4_770b;

namespace {

std::string ltag(int l) {
    char b[16];
    std::snprintf(b, sizeof(b), "%02d", l);
    return b;
}

bool file_exact_size(const std::string& p, uint64_t expected) {
    std::error_code ec;
    return fs::exists(p, ec) && fs::file_size(p, ec) == expected;
}

std::vector<std::string> build_layer_tensor_list(
    int l, const std::array<int32_t, kNumRoutedExperts>& rank_to_orig,
    bool need_fp8, bool need_q5)
{
    std::vector<std::string> names;
    names.reserve(214u * 6u);
    const std::string pfx = "model.layers." + std::to_string(l) + ".mlp.experts.";
    auto add_exp = [&](int orig) {
        const std::string eb = pfx + std::to_string(orig) + ".";
        names.push_back(eb + "gate_proj.weight");
        names.push_back(eb + "gate_proj.weight_scale");
        names.push_back(eb + "up_proj.weight");
        names.push_back(eb + "up_proj.weight_scale");
        names.push_back(eb + "down_proj.weight");
        names.push_back(eb + "down_proj.weight_scale");
    };
    if (need_fp8) {
        for (int r = 0; r < kPackFp8HotEnd; ++r) add_exp(rank_to_orig[size_t(r)]);
    }
    if (need_q5) {
        for (int r = kPackQ5Ddr4Begin; r < kNumRoutedExperts; ++r) add_exp(rank_to_orig[size_t(r)]);
    }
    return names;
}

// Quantize one [N, K] block-FP8 matrix directly to [N * (K / 128)] Q5Lm128Block rows, returning (sig_pow, err_pow)
void quantize_fp8_matrix_to_q5(
    const uint8_t* w8, const float* sc, int N, int K,
    Q5Lm128Block* out_blocks, double& out_sig, double& out_err) noexcept
{
    const int kb = K / 128;
    const auto& lut = fp8_e4m3fn_lut();
    double sig_p = 0.0, err_p = 0.0;

    for (int r = 0; r < N; ++r) {
        const int bi = r >> 7;
        const uint8_t* src_row = w8 + size_t(r) * size_t(K);
        Q5Lm128Block* dst_row = out_blocks + size_t(r) * size_t(kb);
        float tmp128[128];
        float rec128[128];
        for (int bj = 0; bj < kb; ++bj) {
            const float s = sc[size_t(bi) * size_t(kb) + size_t(bj)];
            const uint8_t* sp = src_row + size_t(bj) * 128u;
            for (int c = 0; c < 128; ++c) tmp128[c] = lut[sp[c]] * s;
            q5_lm128_pack_block(tmp128, dst_row[bj]);
            if (r == 0 || (r & 31) == 0) {
                q5_lm128_unpack_block(dst_row[bj], rec128);
                for (int c = 0; c < 128; ++c) {
                    sig_p += double(tmp128[c]) * double(tmp128[c]);
                    const double d = double(tmp128[c]) - double(rec128[c]);
                    err_p += d * d;
                }
            }
        }
    }
    out_sig += sig_p;
    out_err += err_p;
}

}  // namespace

int main(int argc, char** argv) {
    std::string root = "<HY4_770B_WORKSPACE_ROOT>";
    std::string wroot = "<HY4_770B_WEIGHT_ROOT>";
    int start_layer = 1, end_layer = kNumLayers - 1;
    int dl_threads = 24;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--weight-root" && i + 1 < argc) wroot = argv[++i];
        else if (a == "--start-layer" && i + 1 < argc) start_layer = std::stoi(argv[++i]);
        else if (a == "--end-layer" && i + 1 < argc) end_layer = std::stoi(argv[++i]);
        else if (a == "--dl-threads" && i + 1 < argc) dl_threads = std::stoi(argv[++i]);
    }

    install_graceful_signal_handlers();

    const std::string fp8_dir = root + "/fp8_hot";
    const std::string q5_dir  = wroot + "/q5_ddr4";
    fs::create_directories(fp8_dir);
    fs::create_directories(q5_dir);

    RemoteRangeIndex idx;
    idx.load_tsv(root + "/meta/hy4_fp8_tensor_ranges.tsv");

    const uint64_t expected_fp8_bytes = uint64_t(kPackFp8HotEnd) * kFp8ExpertBytes;
    const int q5_cnt = kNumRoutedExperts - kPackQ5Ddr4Begin;  // 194
    const uint64_t expected_q5_bytes  = uint64_t(q5_cnt) * kQ5ExpertBytes;

    // Preload rank_to_orig for all 77 MoE layers
    std::array<std::array<int32_t, kNumRoutedExperts>, kNumLayers> rank_to_orig{};
    for (int l = start_layer; l <= end_layer; ++l) {
        hy4::SafetensorsMmap st(root + "/packed_spine/spine_layer_" + ltag(l) + ".safetensors");
        const int32_t* p = reinterpret_cast<const int32_t*>(st.tensor_bytes("expert_rank_to_orig"));
        std::memcpy(rank_to_orig[size_t(l)].data(), p, size_t(kNumRoutedExperts) * sizeof(int32_t));
    }

    // Build list of layers still needing work
    struct LayerWork {
        int l = 0;
        bool need_fp8 = false;
        bool need_q5 = false;
        std::string fp8_path;
        std::string q5_path;
    };
    std::vector<LayerWork> todo;
    for (int l = start_layer; l <= end_layer; ++l) {
        LayerWork w;
        w.l = l;
        w.fp8_path = fp8_dir + "/fp8_hot_layer_" + ltag(l) + ".bin";
        w.q5_path  = q5_dir  + "/q5_layer_"      + ltag(l) + ".bin";
        w.need_fp8 = !file_exact_size(w.fp8_path, expected_fp8_bytes);
        w.need_q5  = !file_exact_size(w.q5_path,  expected_q5_bytes);
        if (w.need_fp8 || w.need_q5) todo.push_back(std::move(w));
    }

    std::printf("[PACK-FP8-Q5] Starting: %zu/%d layers pending | FP8_HOT=[0..%d) (%.2f GiB) | Q5_DDR4=[%d..256) (%.2f GiB) | dl_threads=%d\n",
                todo.size(), end_layer - start_layer + 1,
                kPackFp8HotEnd, double(expected_fp8_bytes) * 77.0 / 1073741824.0,
                kPackQ5Ddr4Begin, double(expected_q5_bytes) * 77.0 / 1073741824.0,
                dl_threads);
    std::fflush(stdout);

    if (todo.empty()) {
        std::printf("[PACK-FP8-Q5] All layers already complete!\n");
        return 0;
    }

    // Double-buffered async download pipeline: download next layer while quantizing & writing current layer!
    struct FetchResult {
        StreamedTensorBundle bundle;
        double dl_sec = 0.0;
        uint64_t bytes = 0;
    };
    auto launch_fetch = [&](const LayerWork& w) {
        return std::async(std::launch::async, [&, w]() -> FetchResult {
            FetchResult fr;
            auto names = build_layer_tensor_list(w.l, rank_to_orig[size_t(w.l)], w.need_fp8, w.need_q5);
            const auto t0 = std::chrono::steady_clock::now();
            fr.bundle = stream_fetch_tensors(idx, names, dl_threads);
            fr.dl_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            for (const auto& nm : names) fr.bytes += fr.bundle.spec(nm).nbytes;
            return fr;
        });
    };

    std::future<FetchResult> cur_fut = launch_fetch(todo[0]);
    const auto t_all0 = std::chrono::steady_clock::now();

    std::vector<uint8_t> fp8_buf(expected_fp8_bytes);
    std::vector<uint8_t> q5_buf(expected_q5_bytes);

    for (size_t wi = 0; wi < todo.size(); ++wi) {
        if (should_stop()) {
            std::printf("[PACK-FP8-Q5] Graceful stop requested before L%02d!\n", todo[wi].l);
            break;
        }
        const auto& w = todo[wi];
        FetchResult cur = cur_fut.get();

        // Immediately kick off next layer's HTTP download while we quantize + write current layer!
        std::future<FetchResult> next_fut;
        if (wi + 1 < todo.size() && !should_stop()) {
            next_fut = launch_fetch(todo[wi + 1]);
        }

        const auto tp0 = std::chrono::steady_clock::now();
        const std::string pfx = "model.layers." + std::to_string(w.l) + ".mlp.experts.";

        // 1. Pack lossless stock FP8-g128 for ranks [0..20)
        if (w.need_fp8) {
            #pragma omp parallel for schedule(static) num_threads(16)
            for (int r = 0; r < kPackFp8HotEnd; ++r) {
                const int orig = rank_to_orig[size_t(w.l)][size_t(r)];
                const std::string eb = pfx + std::to_string(orig) + ".";
                uint8_t* dst = fp8_buf.data() + size_t(r) * size_t(kFp8ExpertBytes);
                auto copy_proj = [&](const char* pn) {
                    const uint8_t* wd = cur.bundle.data(eb + pn + ".weight");
                    const uint8_t* sd = cur.bundle.data(eb + pn + ".weight_scale");
                    std::memcpy(dst, wd, size_t(kFp8ProjWBytes)); dst += kFp8ProjWBytes;
                    std::memcpy(dst, sd, size_t(kFp8ProjSBytes)); dst += kFp8ProjSBytes;
                };
                copy_proj("gate_proj");
                copy_proj("up_proj");
                copy_proj("down_proj");
            }
            const std::string tmp = w.fp8_path + ".tmp";
            std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
            ofs.write(reinterpret_cast<const char*>(fp8_buf.data()), std::streamsize(expected_fp8_bytes));
            ofs.close();
            if (!ofs) throw std::runtime_error("Write failed: " + tmp);
            fs::rename(tmp, w.fp8_path);
        }

        // 2. Quantize & pack Q5_LM128 for ranks [62..256)
        double sig_sum = 0.0, err_sum = 0.0;
        if (w.need_q5) {
            #pragma omp parallel for schedule(dynamic, 1) num_threads(32) reduction(+:sig_sum,err_sum)
            for (int qi = 0; qi < q5_cnt; ++qi) {
                const int r = kPackQ5Ddr4Begin + qi;
                const int orig = rank_to_orig[size_t(w.l)][size_t(r)];
                const std::string eb = pfx + std::to_string(orig) + ".";
                uint8_t* base = q5_buf.data() + size_t(qi) * size_t(kQ5ExpertBytes);
                Q5Lm128Block* g_blk = reinterpret_cast<Q5Lm128Block*>(base + 0u * kQ5ProjBytes);
                Q5Lm128Block* u_blk = reinterpret_cast<Q5Lm128Block*>(base + 1u * kQ5ProjBytes);
                Q5Lm128Block* d_blk = reinterpret_cast<Q5Lm128Block*>(base + 2u * kQ5ProjBytes);

                double ls = 0.0, le = 0.0;
                quantize_fp8_matrix_to_q5(
                    cur.bundle.data(eb + "gate_proj.weight"),
                    reinterpret_cast<const float*>(cur.bundle.data(eb + "gate_proj.weight_scale")),
                    kMoeInterSize, kHiddenSize, g_blk, ls, le);
                quantize_fp8_matrix_to_q5(
                    cur.bundle.data(eb + "up_proj.weight"),
                    reinterpret_cast<const float*>(cur.bundle.data(eb + "up_proj.weight_scale")),
                    kMoeInterSize, kHiddenSize, u_blk, ls, le);
                quantize_fp8_matrix_to_q5(
                    cur.bundle.data(eb + "down_proj.weight"),
                    reinterpret_cast<const float*>(cur.bundle.data(eb + "down_proj.weight_scale")),
                    kHiddenSize, kMoeInterSize, d_blk, ls, le);
                sig_sum += ls;
                err_sum += le;
            }
            const std::string tmp = w.q5_path + ".tmp";
            std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
            ofs.write(reinterpret_cast<const char*>(q5_buf.data()), std::streamsize(expected_q5_bytes));
            ofs.close();
            if (!ofs) throw std::runtime_error("Write failed: " + tmp);
            fs::rename(tmp, w.q5_path);
        }

        const double pack_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
        const double elapsed_m = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all0).count() / 60.0;
        const double eta_m = (wi + 1 < todo.size()) ? (elapsed_m * double(todo.size() - wi - 1) / double(wi + 1)) : 0.0;
        const double sqnr_db = (err_sum > 0.0) ? (10.0 * std::log10((sig_sum + 1e-30) / (err_sum + 1e-30))) : 0.0;
        std::printf("[FP8-Q5 L%02d/77] (%zu/%zu) dl=%.1fs (%.1f MiB/s) pack+write=%.1fs | Q5_SQNR=%.2fdB | elapsed=%.1fm ETA=%.1fm\n",
                    w.l, wi + 1, todo.size(), cur.dl_sec,
                    (double(cur.bytes) / 1048576.0) / std::max(1e-3, cur.dl_sec),
                    pack_sec, sqnr_db, elapsed_m, eta_m);
        std::fflush(stdout);

        cur_fut = std::move(next_fut);
    }

    std::printf("[PACK-FP8-Q5] ALL DONE!\n");
    return 0;
}
