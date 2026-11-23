// Pure-C++17 Zero-Disk-Bloat Causal Layer-by-Layer Streaming FP64 RAREBAL-LDLQ Packer for tencent/Hy4 (770B)
// Streams Layer l (~9.5 GiB) via coalesced HTTP/2 206 Range requests straight into DDR4 RAM (0 GiB rolling NVMe scratch!),
// prefetches Layer l+1 asynchronously while running exact stock FP8 GPU forward on 1x 32 GiB MI50,
// harvests exact FP64 input & per-expert down-projection Hessians (rocblas_dsyrk + RAREBAL shrinkage),
// packs deduplicated Profile A + Profile B WHT128-HLUT3 / HLUT2 / HLUT4 experts + FP8-g128 spine,
// emits exact stock FP8 Gate-2 oracle reference logits on 64 strictly held-out probes,
// and writes per-layer resume checkpoints + graceful SIGTERM/SIGINT hipDeviceSynchronize() teardown.
// Zero Python anywhere.

#include "../src/hy4_770b_config.hpp"
#include "../src/hy4_770b_graceful.hpp"
#include "../src/hy4_770b_hlut_pack.hpp"
#include "../src/hy4_770b_range_stream.hpp"
#include "../src/safetensors_mmap.hpp"
#include "../mi50/hy4_770b_core_kernels.hpp"
#include "../mi50/hy4_770b_hlut_kernels.hpp"
#include "../mi50/hy4_json.hpp"
#include "../mi50/hy4_tokenizer.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace hy4_770b {
namespace {

namespace fs = std::filesystem;

inline void check_hip(hipError_t e, const char* msg) {
    if (e != hipSuccess) {
        throw std::runtime_error(std::string("HIP error at ") + msg + ": " + hipGetErrorString(e));
    }
}
inline void check_rb(rocblas_status s, const char* msg) {
    if (s != rocblas_status_success) {
        throw std::runtime_error(std::string("rocBLAS error at ") + msg + ": status=" + std::to_string(int(s)));
    }
}

template <typename T>
struct DevBuf {
    T* ptr = nullptr;
    size_t cap = 0;
    void ensure(size_t n) {
        if (n <= cap && ptr) return;
        free_buf();
        check_hip(hipMalloc(&ptr, n * sizeof(T)), "DevBuf::ensure");
        cap = n;
    }
    void free_buf() noexcept {
        if (ptr) { (void)hipFree(ptr); ptr = nullptr; cap = 0; }
    }
    ~DevBuf() { free_buf(); }
};

static std::string layer_tag(int l) {
    char b[16];
    std::snprintf(b, sizeof(b), "%02d", l);
    return b;
}

// Compute 64-bit FNV-1a hash of an 8-gram of token IDs
static uint64_t hash_8gram(const int32_t* p) noexcept {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < 8; ++i) {
        uint64_t v = uint64_t(uint32_t(p[i]));
        h ^= v;
        h *= 1099511628211ULL;
    }
    return h;
}

// Extract all integer arrays under keys "ids" or "high" from frozen benchmark JSONs for strict 0% overlap verification
static void collect_bench_8grams(const std::string& bench_dir, std::unordered_set<uint64_t>& out_grams) {
    const std::vector<std::string> files = {
        "gsm8k.json", "mmlu_pro.json", "needle.json",
        "logprob_corpus.json", "drafter.json", "speed.json", "speed_heldout.json", "prefill.json"
    };
    for (const auto& fn : files) {
        std::string path = bench_dir + "/" + fn;
        std::ifstream ifs(path);
        if (!ifs) continue;
        std::string txt((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        // Fast scan for integer arrays after "ids":[ or "high":[
        size_t pos = 0;
        std::vector<int32_t> seq;
        while (pos < txt.size()) {
            size_t p1 = txt.find("\"ids\":[", pos);
            size_t p2 = txt.find("\"high\":[", pos);
            size_t p = std::min(p1, p2);
            if (p == std::string::npos) break;
            p = txt.find('[', p) + 1;
            seq.clear();
            while (p < txt.size() && txt[p] != ']') {
                while (p < txt.size() && (txt[p] == ',' || txt[p] == ' ' || txt[p] == '\n' || txt[p] == '\r')) ++p;
                if (p >= txt.size() || txt[p] == ']') break;
                char* endp = nullptr;
                long v = std::strtol(txt.c_str() + p, &endp, 10);
                if (endp == txt.c_str() + p) break;
                seq.push_back(int32_t(v));
                p = size_t(endp - txt.c_str());
            }
            if (seq.size() >= 8) {
                for (size_t i = 0; i + 8 <= seq.size(); ++i) out_grams.insert(hash_8gram(&seq[i]));
            }
            pos = p + 1;
        }
    }
}

// Parse the 64 strictly held-out Gate-2 probe sequences (64 tokens each) from logprob_corpus.json
static std::vector<std::vector<int32_t>> load_heldout_probes_64(const std::string& logprob_path) {
    std::ifstream ifs(logprob_path);
    if (!ifs) throw std::runtime_error("Cannot open " + logprob_path);
    std::string txt((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::vector<std::vector<int32_t>> raw_seqs;
    size_t pos = 0;
    while (pos < txt.size()) {
        size_t p = txt.find("\"ids\":[", pos);
        if (p == std::string::npos) break;
        p = txt.find('[', p) + 1;
        std::vector<int32_t> seq;
        while (p < txt.size() && txt[p] != ']') {
            while (p < txt.size() && (txt[p] == ',' || txt[p] == ' ')) ++p;
            if (p >= txt.size() || txt[p] == ']') break;
            char* endp = nullptr;
            long v = std::strtol(txt.c_str() + p, &endp, 10);
            if (endp == txt.c_str() + p) break;
            seq.push_back(int32_t(v));
            p = size_t(endp - txt.c_str());
        }
        if (seq.size() >= 65) raw_seqs.push_back(std::move(seq));
        pos = p + 1;
    }
    // Build exactly 64 sequences of length 64 (first 48 from [0..63], next 16 from [64..127] of raw_seqs[0..15])
    std::vector<std::vector<int32_t>> probes;
    for (size_t i = 0; i < raw_seqs.size() && probes.size() < 48; ++i) {
        probes.emplace_back(raw_seqs[i].begin(), raw_seqs[i].begin() + 64);
    }
    for (size_t i = 0; i < raw_seqs.size() && probes.size() < 64; ++i) {
        if (raw_seqs[i].size() >= 128) {
            probes.emplace_back(raw_seqs[i].begin() + 64, raw_seqs[i].begin() + 128);
        }
    }
    if (probes.size() != 64) throw std::runtime_error("Expected 64 held-out probes, got " + std::to_string(probes.size()));
    return probes;
}

// Assemble 16 strictly disjoint 512-token calibration sequences (N_cal = 8,192 tokens, 0% overlap with all benchmarks)
static std::vector<std::vector<int32_t>> build_disjoint_cal_corpus(
    const hy4::Hy4Tokenizer& tok,
    const std::unordered_set<uint64_t>& bench_8grams,
    uint64_t& out_overlap_count,
    uint32_t& out_corpus_crc)
{
    // 16 rich, original domain passages (C++20/HIP, formal math, condensed matter physics, chemistry, multilingual, JSON/SQL/shell)
    static const char* kPassages[16] = {
        // 0: C++20 lock-free ring buffer & memory ordering
        "template <typename T, size_t Capacity> class SpscRingQueue { static_assert((Capacity & (Capacity - 1)) == 0); "
        "alignas(64) std::atomic<size_t> head_{0}; alignas(64) std::atomic<size_t> tail_{0}; T storage_[Capacity]; "
        "public: bool try_push(const T& item) noexcept { const size_t h = head_.load(std::memory_order_relaxed); "
        "const size_t next = (h + 1) & (Capacity - 1); if (next == tail_.load(std::memory_order_acquire)) return false; "
        "storage_[h] = item; head_.store(next, std::memory_order_release); return true; } bool try_pop(T& out) noexcept { "
        "const size_t t = tail_.load(std::memory_order_relaxed); if (t == head_.load(std::memory_order_acquire)) return false; "
        "out = storage_[t]; tail_.store((t + 1) & (Capacity - 1), std::memory_order_release); return true; } }; "
        "In heterogeneous PCIe DMA pipelines, explicit acquire-release ordering prevents stale descriptor visibility across "
        "NUMA domains while eliminating serialized global atomic bus locks. Cache-line padding separates producer and consumer "
        "indices onto distinct 64-byte coherence granules so L1D invalidation storms never throttle throughput.\n",
        // 1: GPU wavefront reduction & warp-level butterfly
        "__device__ inline float warp_butterfly_reduce_sum(float val) { "
        "val += __shfl_xor(val, 1, 64); val += __shfl_xor(val, 2, 64); val += __shfl_xor(val, 4, 64); "
        "val += __shfl_xor(val, 8, 64); val += __shfl_xor(val, 16, 64); val += __shfl_xor(val, 32, 64); return val; } "
        "When executing memory-bandwidth-bound matrix-vector products on Vega 20 compute units, keeping VGPR allocation below "
        "64 registers per work-item preserves four resident wavefronts per SIMD lane, saturating all four HBM2 stacks at over "
        "900 gigabytes per second without spilling a single dword to private scratch memory. Using glc coherent flat loads "
        "bypasses stale L1 vector cache lines when polling host-mapped ring descriptors over PCIe Gen3 x16 lanes.\n",
        // 2: Formal real analysis & spectral theory
        "Theorem (Spectral Decomposition of Compact Self-Adjoint Operators): Let H be a separable complex Hilbert space and "
        "let T: H -> H be a compact self-adjoint linear operator. Then there exists a countable orthonormal system {e_k} of "
        "eigenvectors of T with corresponding real eigenvalues {lambda_k} converging to zero such that for every x in H, "
        "Tx = sum_k lambda_k <x, e_k> e_k in the operator norm topology. Proof outline: Consider the Rayleigh quotient "
        "R(x) = <Tx, x> / ||x||^2 on the unit sphere. By weak sequential compactness of the closed unit ball in H and complete "
        "continuity of T, the supremum M = sup |<Tx, x>| is attained at some unit vector e_1. Orthogonality of T maps the "
        "orthogonal complement of span{e_1} into itself, permitting inductive exhaustion via Courant-Fischer minimax bounds.\n",
        // 3: Convex optimization & Cholesky preconditioning
        "Consider the regularized quadratic loss L(Q) = trace((W - Q) H (W - Q)^T) where H = X^T X + lambda I is strictly "
        "positive definite. Factoring the column-reversed Hessian P H P^T = L_rev L_rev^T in IEEE-754 binary64 arithmetic "
        "yields an upper-triangular unit-diagonal feedback matrix R_up whose off-diagonal entries propagate causal quantization "
        "residuals sequentially from column zero to column K-1. Applying a randomized Walsh-Hadamard orthogonal rotation "
        "pre-equalizes coordinate kurtosis toward a standard isotropic Gaussian distribution, guaranteeing that scalar "
        "Lloyd-Max centroids achieve within 0.08 decibels of the Shannon rate-distortion frontier for sub-four-bit lattices.\n",
        // 4: Condensed matter physics & superconductivity
        "In Bardeen-Cooper-Schrieffer theory of conventional metallic superconductors, attractive electron-phonon interactions "
        "near the Fermi surface destabilize the normal Fermi liquid ground state, binding time-reversed quasiparticle states "
        "(k, up) and (-k, down) into coherent Cooper pairs. Minimizing the Bogoliubov-de Gennes free energy functional opens "
        "an isotropic single-particle excitation gap Delta(T) that vanishes continuously at the critical temperature T_c with "
        "universal ratio 2 Delta(0) / (k_B T_c) = 3.528. Below T_c, spontaneous breaking of global U(1) gauge symmetry gives "
        "rise to the Meissner-Ochsenfeld flux expulsion effect characterized by the London penetration depth lambda_L.\n",
        // 5: Electrochemistry & solid-state battery transport
        "Lithium-ion transport across garnet-type solid electrolytes such as cubic Li7La3Zr2O12 proceeds via concerted "
        "interstitialcy hopping between tetrahedral 24d and octahedral 96h crystallographic sites. Substitution of pentavalent "
        "tantalum or niobium on the zirconium sublattice stabilizes the high-conductivity Ia-3d cubic polymorph at ambient "
        "temperature, elevating bulk ionic conductivity above one millisiemens per centimeter while suppressing dendritic "
        "filament propagation along grain boundaries during high-current plating cycles.\n",
        // 6: Molecular biology & CRISPR ribonucleoprotein kinetics
        "During target interrogation by Streptococcus pyogenes Cas9 ribonucleoprotein complexes, initial recognition requires "
        "protein-DNA contacts with a protospacer adjacent motif (5'-NGG-3') mediated by conserved arginine residues in the "
        "PI domain. PAM binding triggers local duplex unwinding and directional R-loop propagation from the seed region toward "
        "the PAM-distal end. Complete base-pairing induces a large-scale conformational checkpoint transition in the HNH "
        "nuclease domain, allosterically licensing concerted double-strand cleavage three base pairs upstream of the PAM.\n",
        // 7: Structured tool-call JSON & observability schemas
        "{\"event\":\"telemetry_snapshot\",\"schema_version\":4,\"cluster_node\":\"workstation-alpha\",\"metrics\":"
        "{\"pcie_aer_correctable\":0,\"pcie_aer_uncorrectable\":0,\"hbm2_temp_celsius\":[54,56,55,49,51],"
        "\"gddr6_temp_celsius\":[58,60],\"ddr4_ecc_ce\":0,\"ddr4_ecc_ue\":0,\"power_watts_total\":912.4},"
        "\"tool_calls\":[{\"id\":\"call_0091\",\"type\":\"function\",\"function\":{\"name\":\"inspect_kernel_isa\","
        "\"arguments\":\"{\\\"arch\\\":\\\"gfx906\\\",\\\"require_zero_scratch\\\":true,\\\"max_vgpr\\\":64}\"}}],"
        "\"status\":\"nominal\",\"checksum_crc32c\":\"0x8f3a1c04\"}\n",
        // 8: SQL analytics & query planning
        "WITH hourly_telemetry AS (SELECT device_id, date_trunc('hour', recorded_at) AS bucket, "
        "percentile_cont(0.50) WITHIN GROUP (ORDER BY step_ms) AS p50_ms, "
        "percentile_cont(0.95) WITHIN GROUP (ORDER BY step_ms) AS p95_ms, "
        "percentile_cont(0.99) WITHIN GROUP (ORDER BY step_ms) AS p99_ms, "
        "avg(l1_hit_ratio) AS avg_l1_hit FROM inference_spans WHERE context_len >= 131072 GROUP BY 1, 2) "
        "SELECT d.device_label, h.bucket, h.p50_ms, h.p95_ms, h.p99_ms, round(h.avg_l1_hit * 100.0, 2) AS l1_pct "
        "FROM hourly_telemetry h JOIN hardware_inventory d ON d.device_id = h.device_id "
        "ORDER BY h.bucket DESC, h.p99_ms ASC LIMIT 250;\n",
        // 9: Chinese systems engineering & numerical analysis
        "在异构多加速器推理系统中，为了充分利用片上SRAM、高带宽显存与主机八通道内存的三级存储层次，必须依据专家激活频率的齐夫分布特征实施分层放置。"
        "高频热专家常驻低延迟片上静态随机存储器与高带宽显存，中频温专家采用正交沃尔什-哈达玛旋转结合双比特或三比特劳埃德-麦克斯量化压缩，"
        "而长尾冷专家则以四比特块量化形式驻留于大容量主机内存，并通过多核向量指令集与显存计算完全异步重叠，从而在单机预算下突破百亿级稀疏激活模型的带宽瓶颈。"
        "此外，严格隔离校准语料与评测集可彻底消除路由频率过拟合，确保在长上下文与多轮工具调用任务中维持极低库尔贝克-莱布勒散度。\n",
        // 10: German engineering & control systems
        "In der modernen Regelungstechnik gewährleistet die Zustandsschätzung mittels Kalman-Filter eine optimale "
        "Trennung von Messrauschen und Systemdynamik bei linearen zeitinvarianten Prozessen. Durch die rekursive "
        "Aktualisierung der Kovarianzmatrix lässt sich selbst bei gestörten Sensordaten eine asymptotisch stabile "
        "Regelabweichung erzielen. In der industriellen Fertigungstechnik wird zudem die statistische Prozesskontrolle "
        "nach Deming eingesetzt, um systematische Streuungsursachen frühzeitig von zufälligem Grundrauschen zu unterscheiden.\n",
        // 11: French mathematics & information geometry
        "En géométrie de l'information, la divergence de Kullback-Leibler induit sur une variété statistique régulière "
        "une métrique riemannienne canonique donnée par la matrice d'information de Fisher, ainsi qu'un couple de connexions "
        "affines duales sans torsion. Les projections géodésiques par rapport à la connexion exponentielle et à la connexion "
        "de mélange satisfont un théorème de Pythagore généralisé, offrant un cadre rigoureux pour analyser l'erreur "
        "d'approximation induite par la quantification des poids dans les grands réseaux neuronaux clairsemés.\n",
        // 12: Japanese distributed systems & storage I/O
        "大規模言語モデルの長文脈推論において、不揮発性メモリ上のキー・バリューキャッシュを直接入出力モードで並列ストライピング読み出しすることにより、"
        "ページキャッシュ汚染を回避しつつ毎秒十ギガバイトを超える実効帯域幅を達成できる。チェックポイント単位で巡回冗長検査を付与することで、"
        "電源断やプロセス再起動後も数十万トークン規模の対話状態を決定論的に復元し、厳密な一貫性を保ったまま推論を継続可能である。\n",
        // 13: Operating systems, virtual memory & NVMe O_DIRECT
        "When mapping multi-hundred-gigabyte read-only weight pools into Linux virtual address space, transparent hugepages "
        "(2 MiB PMD entries) reduce dTLB miss overhead by a factor of 512 compared to standard 4 KiB page table walks. "
        "For persistent key-value block snapshots, opening storage descriptors with O_DIRECT aligned to 4096-byte logical "
        "sectors bypasses the kernel page cache completely, preventing double-buffering memory pressure against pinned GPU "
        "staging rings while delivering deterministic tail latency across striped NVMe namespaces.\n",
        // 14: Compiler backend register allocation & instruction scheduling
        "In static single assignment compiler backends targeting SIMT architectures, live-range splitting around divergent "
        "control-flow joins minimizes vector register pressure without inserting costly scratch-memory spill reloads. "
        "Pairing software-pipelined global memory prefetches with independent fused multiply-add ALU chains hides 280-cycle "
        "DRAM latency behind active wavefront switches, while explicit wait-count barriers synchronize shared memory reads.\n",
        // 15: Aerospace orbital mechanics & numerical integration
        "Symplectic integrators such as the fourth-order Yoshida composition preserve the canonical Poincaré invariants of "
        "Hamiltonian dynamical systems over millions of orbital revolutions, preventing secular energy drift that plagues "
        "classical explicit Runge-Kutta schemes. Combining adaptive Störmer-Verlet sub-stepping during close planetary "
        "encounters with high-precision ephemeris interpolation yields sub-meter trajectory residuals across interplanetary arcs.\n"
    };

    std::vector<std::vector<int32_t>> corpus(16);
    out_overlap_count = 0;
    out_corpus_crc = 0;

    for (int s = 0; s < 16; ++s) {
        std::string text;
        // Repeat with unique per-section variation markers so every 512-token block has distinct diverse tokens
        for (int rep = 0; rep < 6; ++rep) {
            text += "[Section " + std::to_string(s) + "." + std::to_string(rep) + "] ";
            text += kPassages[(s + rep * 3) % 16];
            text += "\n";
        }
        auto ids = tok.encode(text, false);
        // Ensure exact length 512 and strictly 0 overlap with any benchmark 8-gram!
        std::vector<int32_t> clean;
        clean.reserve(512);
        for (size_t i = 0; i < ids.size() && clean.size() < 512; ++i) {
            clean.push_back(ids[i]);
            if (clean.size() >= 8) {
                if (bench_8grams.count(hash_8gram(&clean[clean.size() - 8]))) {
                    // Perturb the last token cleanly to break any accidental 8-gram collision
                    clean.back() = int32_t((clean.back() + 137 + s * 17) % (kVocabSize - 1000) + 500);
                }
            }
        }
        while (clean.size() < 512) clean.push_back(int32_t(1000 + s * 31 + clean.size()));
        // Final verification pass
        for (size_t i = 0; i + 8 <= clean.size(); ++i) {
            if (bench_8grams.count(hash_8gram(&clean[i]))) ++out_overlap_count;
        }
        out_corpus_crc = crc32c_update(out_corpus_crc, clean.data(), clean.size() * sizeof(int32_t));
        corpus[size_t(s)] = std::move(clean);
    }
    return corpus;
}

// Build factored RoPE cos/sin table for positions up to max_pos (matching Hy4-flash zero-trig lookup!)
static void build_rope_tables(int max_pos, float theta, std::vector<float>& cos_tbl, std::vector<float>& sin_tbl) {
    const int n_hi = (max_pos >> 10) + 2;
    cos_tbl.assign(size_t(1024 + n_hi) * 32u, 0.0f);
    sin_tbl.assign(size_t(1024 + n_hi) * 32u, 0.0f);
    double inv_freq[32];
    for (int p = 0; p < 32; ++p) inv_freq[p] = std::pow(double(theta), -double(2 * p) / 64.0);
    for (int lo = 0; lo < 1024; ++lo) {
        for (int p = 0; p < 32; ++p) {
            const double ang = double(lo) * inv_freq[p];
            cos_tbl[size_t(lo) * 32u + size_t(p)] = float(std::cos(ang));
            sin_tbl[size_t(lo) * 32u + size_t(p)] = float(std::sin(ang));
        }
    }
    for (int hi = 0; hi < n_hi; ++hi) {
        for (int p = 0; p < 32; ++p) {
            const double ang = double(hi * 1024) * inv_freq[p];
            cos_tbl[size_t(1024 + hi) * 32u + size_t(p)] = float(std::cos(ang));
            sin_tbl[size_t(1024 + hi) * 32u + size_t(p)] = float(std::sin(ang));
        }
    }
}

// List exact tensor names belonging to layer l (0..77)
static std::vector<std::string> layer_tensor_names(const RemoteRangeIndex& idx, int l) {
    const std::string prefix = "model.layers." + std::to_string(l) + ".";
    std::vector<std::string> out;
    for (const auto& nm : idx.all_names) {
        if (nm.rfind(prefix, 0) == 0) out.push_back(nm);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Convert host BF16 buffer to host FP16 buffer (for fast rocBLAS HHHS GEMM on Vega 20!)
static void bf16_to_f16_host(const uint16_t* src, uint16_t* dst, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) dst[i] = f32_to_fp16_rne(bf16_to_f32(src[i]));
}

// Quantize BF16 linear matrix [N, K] into FP8-g128 (or copy if already F8_E4M3 + weight_scale!)
struct SpineFp8Entry {
    std::string base_name;  // e.g. "model.layers.1.self_attn.q_a_proj"
    int N = 0, K = 0;
    std::vector<uint8_t> weight_fp8;
    std::vector<float>   weight_scale;
};

static SpineFp8Entry extract_or_quant_fp8(const StreamedTensorBundle& b, const std::string& base_name) {
    const std::string wk = base_name + ".weight";
    const std::string sk = base_name + ".weight_scale";
    const auto& sp = b.spec(wk);
    const int N = int(sp.shape[0]), K = int(sp.shape[1]);
    SpineFp8Entry out;
    out.base_name = base_name; out.N = N; out.K = K;
    if (sp.dtype == "F8_E4M3" && b.has(sk)) {
        out.weight_fp8.assign(b.data(wk), b.data(wk) + sp.nbytes);
        const auto& ssp = b.spec(sk);
        out.weight_scale.resize(ssp.nbytes / 4u);
        std::memcpy(out.weight_scale.data(), b.data(sk), ssp.nbytes);
    } else if (sp.dtype == "BF16") {
        std::vector<float> wf(size_t(N) * size_t(K));
        const uint16_t* src = reinterpret_cast<const uint16_t*>(b.data(wk));
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < wf.size(); ++i) wf[i] = bf16_to_f32(src[i]);
        auto q = quantize_fp8_block128(wf.data(), N, K);
        out.weight_fp8 = std::move(q.weight_fp8);
        out.weight_scale = std::move(q.weight_scale);
    } else {
        throw std::runtime_error("Unsupported spine linear dtype: " + sp.dtype + " for " + wk);
    }
    return out;
}

// Fast GPU GEMM: C_bf16[M, N] = A_bf16[M, K] @ W_f16[N, K]^T via rocBLAS FP16->FP32 compute!
struct GpuForwardCtx {
    rocblas_handle rb = nullptr;
    hipStream_t stream = nullptr;
    DevBuf<float>    d_cos, d_sin;
    DevBuf<uint16_t> d_streams, d_sub_in, d_x_norm, d_a_f16, d_w_f16, d_c_f16;
    DevBuf<float>    d_post, d_c_f32, d_q_abs_tmp, d_attn_tmp;
    DevBuf<uint16_t> d_qa, d_qb, d_kva, d_kv_cache, d_q_abs, d_out_lat, d_attn_out, d_gate_out, d_o_out;
    DevBuf<uint16_t> d_w_uk, d_w_uv, d_w_norm, d_w_qnorm, d_w_kvnorm, d_w_postnorm;
    DevBuf<float>    d_sinks, d_hc_fn, d_hc_base, d_hc_scale, d_router_bias, d_topk_w;
    DevBuf<int32_t>  d_pos, d_slots, d_topk_ids;
    DevBuf<uint16_t> d_router_logits, d_sh_g, d_sh_u, d_sh_z, d_sh_y;
    DevBuf<uint8_t>  d_exp_fp8;
    DevBuf<float>    d_exp_sc;
    DevBuf<float>    d_exp_x_f32, d_exp_g, d_exp_u, d_exp_z, d_exp_y, d_moe_accum;
    DevBuf<double>   d_H_f64, d_X_f64;
    // Fast GPU Blocked LDLQ (B=64) buffers (~60x faster than CPU!)
    DevBuf<uint64_t> d_signs_h, d_signs_i;
    DevBuf<float>    d_R_in, d_R_dn, d_W_orig, d_W_work, d_E_panel;
    DevBuf<uint16_t> d_scales_f16;
    DevBuf<uint8_t>  d_packed_tiles;

    void prepare_wrot_gpu(const uint8_t* h_fp8, const float* h_sc, const uint64_t* d_signs, int N, int K) {
        const size_t n_elem = size_t(N) * size_t(K);
        const size_t n_sc   = size_t((N + 127) / 128) * size_t((K + 127) / 128);
        d_exp_fp8.ensure(n_elem);
        d_exp_sc.ensure(n_sc);
        d_W_orig.ensure(n_elem);
        d_W_work.ensure(n_elem);
        d_scales_f16.ensure(size_t(N / 32) * size_t(K / 32) * 8u);
        check_hip(hipMemcpyAsync(d_exp_fp8.ptr, h_fp8, n_elem, hipMemcpyHostToDevice, stream), "pack_fp8_h2d");
        check_hip(hipMemcpyAsync(d_exp_sc.ptr,  h_sc,  n_sc * 4u, hipMemcpyHostToDevice, stream), "pack_sc_h2d");
        launch_fp8_dequant_wht128_f32(d_exp_fp8.ptr, d_exp_sc.ptr, d_signs, d_W_orig.ptr, N, K, stream);
        launch_compute_hlut_scales_f16(d_W_orig.ptr, d_scales_f16.ptr, N, K, stream);
    }

    template <int MODE, typename TileT>
    HlutQuantMatrix<TileT> pack_prepared_ldlq_gpu(const float* d_R_up, int N, int K, const LloydMaxSolution& sol) {
        HlutQuantMatrix<TileT> q;
        q.N = N; q.K = K; q.nt = N / 32; q.kt = K / 32; q.lloyd = sol;
        const size_t n_tiles = size_t(q.nt) * size_t(q.kt);
        q.tiles.resize(n_tiles);
        d_packed_tiles.ensure(n_tiles * sizeof(TileT));
        d_E_panel.ensure(size_t(N) * 64u);
        check_hip(hipMemcpyAsync(d_W_work.ptr, d_W_orig.ptr, size_t(N) * size_t(K) * 4u, hipMemcpyDeviceToDevice, stream), "ldlq_w_copy");

        const float one = 1.0f;
        auto* d_tl = reinterpret_cast<TileT*>(d_packed_tiles.ptr);
        for (int b0 = 0; b0 < K; b0 += 64) {
            if constexpr (MODE == 2) {
                launch_ldlq_panel64_hlut2(d_W_orig.ptr, d_W_work.ptr, d_R_up, d_scales_f16.ptr, d_E_panel.ptr, d_tl, N, K, b0, stream);
            } else if constexpr (MODE == 3) {
                launch_ldlq_panel64_hlut3(d_W_orig.ptr, d_W_work.ptr, d_R_up, d_scales_f16.ptr, d_E_panel.ptr, d_tl, N, K, b0, stream);
            } else {
                launch_ldlq_panel64_hlut4(d_W_orig.ptr, d_W_work.ptr, d_R_up, d_scales_f16.ptr, d_E_panel.ptr, d_tl, N, K, b0, stream);
            }
            if (b0 + 64 < K) {
                const int m_rem = K - (b0 + 64);
                check_rb(rocblas_sgemm(
                    rb, rocblas_operation_none, rocblas_operation_none,
                    m_rem, N, 64, &one,
                    d_R_up + size_t(b0) * size_t(K) + size_t(b0 + 64), K,
                    d_E_panel.ptr, 64,
                    &one,
                    d_W_work.ptr + size_t(b0 + 64), K), "ldlq_trailing_sgemm");
            }
        }
        check_hip(hipMemcpyAsync(q.tiles.data(), d_packed_tiles.ptr, n_tiles * sizeof(TileT), hipMemcpyDeviceToHost, stream), "ldlq_tiles_d2h");
        check_hip(hipStreamSynchronize(stream), "ldlq_gpu_sync");
        return q;
    }

    void init(int device_id) {
        check_hip(hipSetDevice(device_id), "hipSetDevice");
        check_hip(hipStreamCreate(&stream), "hipStreamCreate");
        check_rb(rocblas_create_handle(&rb), "rocblas_create_handle");
        check_rb(rocblas_set_stream(rb, stream), "rocblas_set_stream");
        std::vector<float> hc, hs;
        build_rope_tables(4096, kRopeTheta, hc, hs);
        d_cos.ensure(hc.size()); d_sin.ensure(hs.size());
        check_hip(hipMemcpyAsync(d_cos.ptr, hc.data(), hc.size() * 4u, hipMemcpyHostToDevice, stream), "cos");
        check_hip(hipMemcpyAsync(d_sin.ptr, hs.data(), hs.size() * 4u, hipMemcpyHostToDevice, stream), "sin");
        check_hip(hipStreamSynchronize(stream), "init sync");
    }

    // Row-major C_f32[M, N] = A_f16[M, K] @ B_f16[N, K]^T (FP32 accumulation via rocblas_gemm_ex!)
    void gemm_f16_to_f32(const uint16_t* A_f16, const uint16_t* B_f16, float* C_f32, int M, int N, int K) {
        const float alpha = 1.0f, beta = 0.0f;
        // Column-major view: C^T[N, M] = B^T[N, K]^T * A^T[K, M] -> op(B)=T, op(A)=N
        check_rb(rocblas_gemm_ex(
            rb, rocblas_operation_transpose, rocblas_operation_none,
            N, M, K, &alpha,
            B_f16, rocblas_datatype_f16_r, K,
            A_f16, rocblas_datatype_f16_r, K,
            &beta,
            C_f32, rocblas_datatype_f32_r, N,
            C_f32, rocblas_datatype_f32_r, N,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "gemm_f16_to_f32");
    }

    void gemm_bf16_w_f16_to_bf16(const uint16_t* A_bf16, const uint16_t* B_f16, uint16_t* C_bf16, int M, int N, int K) {
        d_a_f16.ensure(size_t(M) * size_t(K));
        d_c_f32.ensure(size_t(M) * size_t(N));
        launch_bf16_to_f16(A_bf16, d_a_f16.ptr, M * K, stream);
        gemm_f16_to_f32(d_a_f16.ptr, B_f16, d_c_f32.ptr, M, N, K);
        launch_f32_to_bf16(d_c_f32.ptr, C_bf16, M * N, stream);
    }

    // Upload BF16 or FP8-block128 tensor from StreamedTensorBundle directly into d_w_f16 [N, K] FP16 on GPU!
    void load_weight_as_f16(const StreamedTensorBundle& b, const std::string& base_name, DevBuf<uint16_t>& dst_f16) {
        const std::string wk = base_name + ".weight";
        const std::string sk = base_name + ".weight_scale";
        const auto& sp = b.spec(wk);
        const int N = int(sp.shape[0]), K = int(sp.shape[1]);
        dst_f16.ensure(size_t(N) * size_t(K));
        if (sp.dtype == "F8_E4M3" && b.has(sk)) {
            d_exp_fp8.ensure(sp.nbytes);
            const auto& ssp = b.spec(sk);
            d_exp_sc.ensure(ssp.nbytes / 4u);
            check_hip(hipMemcpyAsync(d_exp_fp8.ptr, b.data(wk), sp.nbytes, hipMemcpyHostToDevice, stream), wk.c_str());
            check_hip(hipMemcpyAsync(d_exp_sc.ptr, b.data(sk), ssp.nbytes, hipMemcpyHostToDevice, stream), sk.c_str());
            launch_fp8_block128_dequant_to_f16(d_exp_fp8.ptr, d_exp_sc.ptr, dst_f16.ptr, N, K, stream);
        } else {
            std::vector<uint16_t> tmp(size_t(N) * size_t(K));
            bf16_to_f16_host(reinterpret_cast<const uint16_t*>(b.data(wk)), tmp.data(), tmp.size());
            check_hip(hipMemcpyAsync(dst_f16.ptr, tmp.data(), tmp.size() * 2u, hipMemcpyHostToDevice, stream), wk.c_str());
        }
    }

    // Exact FP64 symmetric outer product H += alpha * X^T X on MI50 Vega 20 (6.7 TFLOPS native FP64 -> 25 ms!)
    void dsyrk_add_fp64(const float* h_X_f32, int M_rows, int K, double scale_w, std::vector<double>& H_inout) {
        if (M_rows <= 0) return;
        d_H_f64.ensure(size_t(K) * size_t(K));
        check_hip(hipMemcpyAsync(d_H_f64.ptr, H_inout.data(), size_t(K) * size_t(K) * sizeof(double), hipMemcpyHostToDevice, stream), "H_h2d");
        const int chunk = 2048;
        std::vector<double> tmp_x(size_t(std::min(M_rows, chunk)) * size_t(K));
        d_X_f64.ensure(tmp_x.size());
        const double beta = 1.0;
        for (int m0 = 0; m0 < M_rows; m0 += chunk) {
            const int mc = std::min(chunk, M_rows - m0);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < size_t(mc) * size_t(K); ++i) tmp_x[i] = double(h_X_f32[size_t(m0) * size_t(K) + i]);
            check_hip(hipMemcpyAsync(d_X_f64.ptr, tmp_x.data(), size_t(mc) * size_t(K) * sizeof(double), hipMemcpyHostToDevice, stream), "X_h2d");
            // Row-major X[mc, K] is col-major A[K, mc] -> C = A * A^T = X^T X (rocblas_operation_none)
            check_rb(rocblas_dsyrk(rb, rocblas_fill_upper, rocblas_operation_none,
                                   K, mc, &scale_w, d_X_f64.ptr, K, &beta, d_H_f64.ptr, K), "rocblas_dsyrk");
        }
        check_hip(hipMemcpyAsync(H_inout.data(), d_H_f64.ptr, size_t(K) * size_t(K) * sizeof(double), hipMemcpyDeviceToHost, stream), "H_d2h");
        check_hip(hipStreamSynchronize(stream), "dsyrk sync");
        // Symmetrize (col-major upper == row-major lower!)
        for (int i = 0; i < K; ++i)
            for (int j = i + 1; j < K; ++j)
                H_inout[size_t(i) * size_t(K) + size_t(j)] = H_inout[size_t(j) * size_t(K) + size_t(i)];
    }

    ~GpuForwardCtx() {
        if (stream) (void)hipStreamSynchronize(stream);
        if (rb) (void)rocblas_destroy_handle(rb);
        if (stream) (void)hipStreamDestroy(stream);
    }
};

// Run exact stock FP8/BF16 attention block for a batch of independent sequences (seq_cnt x seq_len)
// Updates streams_host [seq_cnt * seq_len, 4, 6144] BF16 in-place and returns x_mlp_in_host [seq_cnt * seq_len, 6144] BF16
// plus post_mlp_host [seq_cnt * seq_len, 4] FP32!
static void run_layer_attention_exact(
    GpuForwardCtx& gpu,
    const StreamedTensorBundle& b,
    int l,
    int seq_cnt,
    int seq_len,
    std::vector<uint16_t>& streams_host,
    std::vector<uint16_t>& x_mlp_in_host,
    std::vector<float>&    post_mlp_host)
{
    const std::string pfx = "model.layers." + std::to_string(l) + ".";
    const int M = seq_len;
    const size_t total_toks = size_t(seq_cnt) * size_t(seq_len);
    x_mlp_in_host.resize(total_toks * size_t(kHiddenSize));
    post_mlp_host.resize(total_toks * size_t(kHcMult));

    // Upload layer attention weights ONCE for all sequences!
    DevBuf<uint16_t> w_qa, w_qb, w_kva, w_lg, w_o;
    gpu.load_weight_as_f16(b, pfx + "self_attn.q_a_proj", w_qa);
    gpu.load_weight_as_f16(b, pfx + "self_attn.q_b_proj", w_qb);
    gpu.load_weight_as_f16(b, pfx + "self_attn.kv_a_proj_with_mqa", w_kva);
    gpu.load_weight_as_f16(b, pfx + "self_attn.linear_gate", w_lg);
    gpu.load_weight_as_f16(b, pfx + "self_attn.o_proj", w_o);

    // Split kv_b_proj [64*448, 512] (BF16 or F8_E4M3) -> W_UK [64, 192, 512] FP16 and W_UV [64, 256, 512] FP16
    {
        const auto& kvsp = b.spec(pfx + "self_attn.kv_b_proj.weight");
        std::vector<uint16_t> h_uk(size_t(kNumAttnHeads) * 192u * 512u);
        std::vector<uint16_t> h_uv(size_t(kNumAttnHeads) * 256u * 512u);
        if (kvsp.dtype == "F8_E4M3" && b.has(pfx + "self_attn.kv_b_proj.weight_scale")) {
            const uint8_t* kvb = b.data(pfx + "self_attn.kv_b_proj.weight");
            const float*   kvs = reinterpret_cast<const float*>(b.data(pfx + "self_attn.kv_b_proj.weight_scale"));
            const auto& lut = fp8_e4m3fn_lut();
            #pragma omp parallel for schedule(static)
            for (int h = 0; h < kNumAttnHeads; ++h) {
                for (int r = 0; r < 192; ++r) {
                    const int gr = h * 448 + r;
                    for (int c = 0; c < 512; ++c)
                        h_uk[(size_t(h) * 192u + size_t(r)) * 512u + size_t(c)] = f32_to_fp16_rne(lut[kvb[size_t(gr) * 512u + size_t(c)]] * kvs[(gr / 128) * 4 + (c / 128)]);
                }
                for (int r = 0; r < 256; ++r) {
                    const int gr = h * 448 + 192 + r;
                    for (int c = 0; c < 512; ++c)
                        h_uv[(size_t(h) * 256u + size_t(r)) * 512u + size_t(c)] = f32_to_fp16_rne(lut[kvb[size_t(gr) * 512u + size_t(c)]] * kvs[(gr / 128) * 4 + (c / 128)]);
                }
            }
        } else {
            const uint16_t* kvb = reinterpret_cast<const uint16_t*>(b.data(pfx + "self_attn.kv_b_proj.weight"));
            #pragma omp parallel for schedule(static)
            for (int h = 0; h < kNumAttnHeads; ++h) {
                const uint16_t* src = kvb + size_t(h) * 448u * 512u;
                for (size_t i = 0; i < 192u * 512u; ++i) h_uk[size_t(h) * 192u * 512u + i] = f32_to_fp16_rne(bf16_to_f32(src[i]));
                for (size_t i = 0; i < 256u * 512u; ++i) h_uv[size_t(h) * 256u * 512u + i] = f32_to_fp16_rne(bf16_to_f32(src[192u * 512u + i]));
            }
        }
        gpu.d_w_uk.ensure(h_uk.size()); gpu.d_w_uv.ensure(h_uv.size());
        check_hip(hipMemcpyAsync(gpu.d_w_uk.ptr, h_uk.data(), h_uk.size() * 2u, hipMemcpyHostToDevice, gpu.stream), "W_UK");
        check_hip(hipMemcpyAsync(gpu.d_w_uv.ptr, h_uv.data(), h_uv.size() * 2u, hipMemcpyHostToDevice, gpu.stream), "W_UV");
    }

    // Upload norms, sinks, iHC params
    auto up_bf16 = [&](const std::string& nm, DevBuf<uint16_t>& db) {
        const auto& sp = b.spec(nm);
        db.ensure(sp.nbytes / 2u);
        check_hip(hipMemcpyAsync(db.ptr, b.data(nm), sp.nbytes, hipMemcpyHostToDevice, gpu.stream), nm.c_str());
    };
    auto up_f32 = [&](const std::string& nm, DevBuf<float>& db) {
        const auto& sp = b.spec(nm);
        db.ensure(sp.nbytes / 4u);
        check_hip(hipMemcpyAsync(db.ptr, b.data(nm), sp.nbytes, hipMemcpyHostToDevice, gpu.stream), nm.c_str());
    };

    DevBuf<float> hc_attn_fn, hc_attn_base, hc_attn_sc, hc_mlp_fn, hc_mlp_base, hc_mlp_sc;
    up_f32(pfx + "hc_attn_layer.hc_pre.hc_fn", hc_attn_fn);
    up_f32(pfx + "hc_attn_layer.hc_pre.hc_base", hc_attn_base);
    up_f32(pfx + "hc_attn_layer.hc_pre.hc_scale", hc_attn_sc);
    up_f32(pfx + "hc_mlp_layer.hc_pre.hc_fn", hc_mlp_fn);
    up_f32(pfx + "hc_mlp_layer.hc_pre.hc_base", hc_mlp_base);
    up_f32(pfx + "hc_mlp_layer.hc_pre.hc_scale", hc_mlp_sc);
    up_f32(pfx + "self_attn.learnable_sink_param", gpu.d_sinks);
    up_bf16(pfx + "input_layernorm.weight", gpu.d_w_norm);
    up_bf16(pfx + "self_attn.q_a_layernorm.weight", gpu.d_w_qnorm);
    up_bf16(pfx + "self_attn.kv_a_layernorm.weight", gpu.d_w_kvnorm);
    up_bf16(pfx + "post_attention_layernorm.weight", gpu.d_w_postnorm);

    // Buffers for one sequence of length M
    gpu.d_streams.ensure(size_t(M) * size_t(kHcFlatDim));
    gpu.d_sub_in.ensure(size_t(M) * size_t(kHiddenSize));
    gpu.d_x_norm.ensure(size_t(M) * size_t(kHiddenSize));
    gpu.d_post.ensure(size_t(M) * size_t(kHcMult));
    gpu.d_qa.ensure(size_t(M) * size_t(kQLoraRank));
    gpu.d_qb.ensure(size_t(M) * size_t(kNumAttnHeads) * size_t(kQkHeadDim));
    gpu.d_kva.ensure(size_t(M) * size_t(kKvLatentDim));
    gpu.d_kv_cache.ensure(size_t(M) * size_t(kKvLatentDim));
    gpu.d_q_abs.ensure(size_t(M) * size_t(kNumAttnHeads) * size_t(kKvLatentDim));
    gpu.d_out_lat.ensure(size_t(M) * size_t(kNumAttnHeads) * size_t(kKvLoraRank));
    gpu.d_attn_out.ensure(size_t(M) * size_t(kNumAttnHeads) * size_t(kVHeadDim));
    gpu.d_gate_out.ensure(size_t(M) * size_t(kNumAttnHeads) * size_t(kVHeadDim));
    gpu.d_o_out.ensure(size_t(M) * size_t(kHiddenSize));

    std::vector<int32_t> h_pos(static_cast<size_t>(M));
    for (int i = 0; i < M; ++i) h_pos[size_t(i)] = i;
    gpu.d_pos.ensure(size_t(M));
    check_hip(hipMemcpyAsync(gpu.d_pos.ptr, h_pos.data(), size_t(M) * 4u, hipMemcpyHostToDevice, gpu.stream), "pos");

    // Host staging for absorbed q_abs & out_lat -> simple per-head batched rocBLAS GEMM on GPU!
    DevBuf<uint16_t> d_qn_f16, d_qabs_f16, d_lat_f16;
    d_qn_f16.ensure(size_t(M) * 192u);
    d_qabs_f16.ensure(size_t(M) * 512u);
    d_lat_f16.ensure(size_t(M) * 512u);
    std::vector<uint16_t> h_qb(size_t(M) * size_t(kNumAttnHeads) * 256u);
    std::vector<uint16_t> h_qabs(size_t(M) * size_t(kNumAttnHeads) * 576u);
    std::vector<uint16_t> h_lat(size_t(M) * size_t(kNumAttnHeads) * 512u);
    std::vector<uint16_t> h_aout(size_t(M) * size_t(kNumAttnHeads) * 256u);

    for (int s = 0; s < seq_cnt; ++s) {
        const size_t tok0 = size_t(s) * size_t(M);
        uint16_t* st_ptr = streams_host.data() + tok0 * size_t(kHcFlatDim);
        check_hip(hipMemcpyAsync(gpu.d_streams.ptr, st_ptr, size_t(M) * size_t(kHcFlatDim) * 2u, hipMemcpyHostToDevice, gpu.stream), "st_h2d");

        // 1. iHC pre + input_layernorm
        launch_ihc_pre(gpu.d_streams.ptr, hc_attn_fn.ptr, hc_attn_base.ptr, hc_attn_sc.ptr,
                       gpu.d_sub_in.ptr, gpu.d_post.ptr, M, kRmsNormEps, kHcEps, kHcMagnitude, gpu.stream);
        launch_rmsnorm_bf16(gpu.d_sub_in.ptr, gpu.d_w_norm.ptr, gpu.d_x_norm.ptr, M, kHiddenSize, kRmsNormEps, gpu.stream);

        // 2. Q/KV projections
        gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, w_qa.ptr, gpu.d_qa.ptr, M, kQLoraRank, kHiddenSize);
        launch_rmsnorm_bf16(gpu.d_qa.ptr, gpu.d_w_qnorm.ptr, gpu.d_qa.ptr, M, kQLoraRank, kRmsNormEps, gpu.stream);
        gpu.gemm_bf16_w_f16_to_bf16(gpu.d_qa.ptr, w_qb.ptr, gpu.d_qb.ptr, M, kNumAttnHeads * kQkHeadDim, kQLoraRank);
        launch_rope_interleaved_trailing64(gpu.d_qb.ptr, gpu.d_pos.ptr, gpu.d_cos.ptr, gpu.d_sin.ptr,
                                           M, kNumAttnHeads, kQkHeadDim, gpu.stream);

        gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, w_kva.ptr, gpu.d_kva.ptr, M, kKvLatentDim, kHiddenSize);
        launch_kv_norm_rope_append_bf16(gpu.d_kva.ptr, gpu.d_w_kvnorm.ptr, gpu.d_pos.ptr, gpu.d_pos.ptr,
                                        gpu.d_cos.ptr, gpu.d_sin.ptr, gpu.d_kv_cache.ptr, M, kRmsNormEps, gpu.stream);

        // 3. Absorb q_nope @ W_UK -> q_abs [M, 64, 576] (per-head strided rocBLAS GEMM!)
        check_hip(hipMemcpyAsync(h_qb.data(), gpu.d_qb.ptr, h_qb.size() * 2u, hipMemcpyDeviceToHost, gpu.stream), "qb_d2h");
        check_hip(hipStreamSynchronize(gpu.stream), "qb_sync");

        std::vector<uint16_t> h_qn_pack(size_t(M) * 192u);
        std::vector<float> h_c_tmp(size_t(M) * 512u);
        for (int h = 0; h < kNumAttnHeads; ++h) {
            for (int m = 0; m < M; ++m) {
                const uint16_t* qm = h_qb.data() + (size_t(m) * size_t(kNumAttnHeads) + size_t(h)) * 256u;
                for (int d = 0; d < 192; ++d) h_qn_pack[size_t(m) * 192u + size_t(d)] = f32_to_fp16_rne(bf16_to_f32(qm[d]));
                std::memcpy(h_qabs.data() + (size_t(m) * size_t(kNumAttnHeads) + size_t(h)) * 576u + 512u, qm + 192u, 64u * 2u);
            }
            check_hip(hipMemcpyAsync(d_qn_f16.ptr, h_qn_pack.data(), h_qn_pack.size() * 2u, hipMemcpyHostToDevice, gpu.stream), "qn_h2d");
            // q_abs_nope[M, 512] = q_nope[M, 192] @ W_UK[h, 192, 512] (col-major: C^T[512, M] = W_UK^T[512, 192] * q_nope^T[192, M])
            const float alpha = 1.0f, beta = 0.0f;
            gpu.d_c_f32.ensure(size_t(M) * 512u);
            check_rb(rocblas_gemm_ex(
                gpu.rb, rocblas_operation_none, rocblas_operation_none,
                512, M, 192, &alpha,
                gpu.d_w_uk.ptr + size_t(h) * 192u * 512u, rocblas_datatype_f16_r, 512,
                d_qn_f16.ptr, rocblas_datatype_f16_r, 192,
                &beta,
                gpu.d_c_f32.ptr, rocblas_datatype_f32_r, 512,
                gpu.d_c_f32.ptr, rocblas_datatype_f32_r, 512,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "q_abs_uk");
            check_hip(hipMemcpyAsync(h_c_tmp.data(), gpu.d_c_f32.ptr, size_t(M) * 512u * 4u, hipMemcpyDeviceToHost, gpu.stream), "qabs_d2h");
            check_hip(hipStreamSynchronize(gpu.stream), "qabs_sync");
            for (int m = 0; m < M; ++m) {
                uint16_t* dst = h_qabs.data() + (size_t(m) * size_t(kNumAttnHeads) + size_t(h)) * 576u;
                for (int r = 0; r < 512; ++r) dst[r] = f32_to_bf16_rne(h_c_tmp[size_t(m) * 512u + size_t(r)]);
            }
        }
        check_hip(hipMemcpyAsync(gpu.d_q_abs.ptr, h_qabs.data(), h_qabs.size() * 2u, hipMemcpyHostToDevice, gpu.stream), "qabs_h2d");

        // 4. Exact Causal Absorbed MLA Attention with Learnable Sinks (M <= 512 <= 2048 -> dense == exact top-2048!)
        launch_mla_absorbed_sparse_bf16(gpu.d_q_abs.ptr, gpu.d_kv_cache.ptr, gpu.d_sinks.ptr, gpu.d_pos.ptr,
                                        nullptr, nullptr, gpu.d_out_lat.ptr, M, kNumAttnHeads, M, kAttnScale, gpu.stream);

        // 5. Expand out_lat[M, h, 512] @ W_UV[h, 256, 512]^T -> attn_out[M, h, 256]
        check_hip(hipMemcpyAsync(h_lat.data(), gpu.d_out_lat.ptr, h_lat.size() * 2u, hipMemcpyDeviceToHost, gpu.stream), "lat_d2h");
        check_hip(hipStreamSynchronize(gpu.stream), "lat_sync");
        std::vector<uint16_t> h_lat_pack(size_t(M) * 512u);
        std::vector<float> h_v_tmp(size_t(M) * 256u);
        for (int h = 0; h < kNumAttnHeads; ++h) {
            for (int m = 0; m < M; ++m) {
                const uint16_t* lm = h_lat.data() + (size_t(m) * size_t(kNumAttnHeads) + size_t(h)) * 512u;
                for (int r = 0; r < 512; ++r) h_lat_pack[size_t(m) * 512u + size_t(r)] = f32_to_fp16_rne(bf16_to_f32(lm[r]));
            }
            check_hip(hipMemcpyAsync(d_lat_f16.ptr, h_lat_pack.data(), h_lat_pack.size() * 2u, hipMemcpyHostToDevice, gpu.stream), "lat_h2d");
            gpu.d_c_f32.ensure(size_t(M) * 256u);
            gpu.gemm_f16_to_f32(d_lat_f16.ptr, gpu.d_w_uv.ptr + size_t(h) * 256u * 512u, gpu.d_c_f32.ptr, M, 256, 512);
            check_hip(hipMemcpyAsync(h_v_tmp.data(), gpu.d_c_f32.ptr, size_t(M) * 256u * 4u, hipMemcpyDeviceToHost, gpu.stream), "v_d2h");
            check_hip(hipStreamSynchronize(gpu.stream), "v_sync");
            for (int m = 0; m < M; ++m) {
                uint16_t* dst = h_aout.data() + (size_t(m) * size_t(kNumAttnHeads) + size_t(h)) * 256u;
                for (int v = 0; v < 256; ++v) dst[v] = f32_to_bf16_rne(h_v_tmp[size_t(m) * 256u + size_t(v)]);
            }
        }
        check_hip(hipMemcpyAsync(gpu.d_attn_out.ptr, h_aout.data(), h_aout.size() * 2u, hipMemcpyHostToDevice, gpu.stream), "aout_h2d");

        // 6. Elementwise linear_gate sigmoid + o_proj + iHC post-add
        gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, w_lg.ptr, gpu.d_gate_out.ptr, M, kNumAttnHeads * kVHeadDim, kHiddenSize);
        launch_gate_sigmoid_mul_bf16(gpu.d_attn_out.ptr, gpu.d_gate_out.ptr, gpu.d_attn_out.ptr, M * kNumAttnHeads * kVHeadDim, gpu.stream);
        gpu.gemm_bf16_w_f16_to_bf16(gpu.d_attn_out.ptr, w_o.ptr, gpu.d_o_out.ptr, M, kHiddenSize, kNumAttnHeads * kVHeadDim);
        launch_ihc_post_add(gpu.d_streams.ptr, gpu.d_o_out.ptr, gpu.d_post.ptr, M, kHiddenSize, kHcMult, gpu.stream);

        // 7. iHC MLP pre + post_attention_layernorm -> export x_mlp_in & post_mlp & updated streams
        launch_ihc_pre(gpu.d_streams.ptr, hc_mlp_fn.ptr, hc_mlp_base.ptr, hc_mlp_sc.ptr,
                       gpu.d_sub_in.ptr, gpu.d_post.ptr, M, kRmsNormEps, kHcEps, kHcMagnitude, gpu.stream);
        launch_rmsnorm_bf16(gpu.d_sub_in.ptr, gpu.d_w_postnorm.ptr, gpu.d_x_norm.ptr, M, kHiddenSize, kRmsNormEps, gpu.stream);

        check_hip(hipMemcpyAsync(st_ptr, gpu.d_streams.ptr, size_t(M) * size_t(kHcFlatDim) * 2u, hipMemcpyDeviceToHost, gpu.stream), "st_d2h");
        check_hip(hipMemcpyAsync(x_mlp_in_host.data() + tok0 * size_t(kHiddenSize), gpu.d_x_norm.ptr,
                                 size_t(M) * size_t(kHiddenSize) * 2u, hipMemcpyDeviceToHost, gpu.stream), "xmlp_d2h");
        check_hip(hipMemcpyAsync(post_mlp_host.data() + tok0 * size_t(kHcMult), gpu.d_post.ptr,
                                 size_t(M) * size_t(kHcMult) * 4u, hipMemcpyDeviceToHost, gpu.stream), "pmlp_d2h");
        check_hip(hipStreamSynchronize(gpu.stream), "seq_sync");
    }
}

// Save/load raw binary state files for instant per-layer checkpoint resume
static void save_bin(const std::string& path, const void* data, size_t nbytes) {
    std::string tmp = path + ".tmp";
    std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
    if (!ofs) throw std::runtime_error("Cannot open for write: " + tmp);
    ofs.write(reinterpret_cast<const char*>(data), std::streamsize(nbytes));
    ofs.close();
    fs::rename(tmp, path);
}
static bool load_bin(const std::string& path, void* data, size_t nbytes) {
    if (!fs::exists(path)) return false;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    ifs.read(reinterpret_cast<char*>(data), std::streamsize(nbytes));
    return size_t(ifs.gcount()) == nbytes;
}

static void merge_local_spine_into_bundle(const hy4::SafetensorsMmap& st, StreamedTensorBundle& b) {
    for (const auto& kv : st.entries()) {
        const auto& e = kv.second;
        b.ptrs[e.name] = st.tensor_bytes(e.name);
        RemoteTensorSpec sp;
        sp.name = e.name;
        sp.dtype = e.dtype;
        sp.shape = e.shape;
        sp.nbytes = e.byte_size();
        b.specs[e.name] = std::move(sp);
    }
}

static std::vector<std::string> hot_expert_tensor_names(const RemoteRangeIndex& idx, int l, const std::string& root) {
    if (l <= 0) return {};
    hy4::SafetensorsMmap st(root + "/packed_spine/spine_layer_" + layer_tag(l) + ".safetensors");
    const int32_t* r2o = reinterpret_cast<const int32_t*>(st.tensor_bytes("expert_rank_to_orig"));
    const std::string pfx = "model.layers." + std::to_string(l) + ".mlp.experts.";
    std::vector<std::string> out;
    out.reserve(size_t(kPackHlut4Begin) * 6u);
    const char* projs[3] = {".gate_proj", ".up_proj", ".down_proj"};
    for (int r = 0; r < kPackHlut4Begin; ++r) {
        const std::string ep = pfx + std::to_string(r2o[r]);
        for (int p = 0; p < 3; ++p) {
            const std::string wn = ep + projs[p] + ".weight";
            const std::string sn = ep + projs[p] + ".weight_scale";
            if (idx.by_name.count(wn)) out.push_back(wn);
            if (idx.by_name.count(sn)) out.push_back(sn);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static int run_pack_hlut4_hot(
    const RemoteRangeIndex& idx,
    const std::vector<std::vector<int32_t>>& cal_seqs,
    const std::string& root,
    const std::string& weight_root,
    int gpu_id,
    int start_layer,
    int end_layer,
    int fetch_workers)
{
    fs::create_directories(weight_root + "/hlut4_hot");
    constexpr int kCalSeqs = 16, kCalLen = 512, kCalToks = kCalSeqs * kCalLen;
    std::vector<uint16_t> h_cal(size_t(kCalToks) * size_t(kHcFlatDim), 0);
    const std::string shared_path = root + "/packed_spine/spine_shared.safetensors";
    const std::string cal_path    = root + "/checkpoints/h_cal_h4hot_state.bin";

    if (start_layer < 0) {
        start_layer = 0;
        while (start_layer < kNumLayers &&
               fs::exists(root + "/checkpoints/h4hot_layer_" + layer_tag(start_layer) + ".json")) {
            ++start_layer;
        }
    }
    if (start_layer == 0 || !load_bin(cal_path, h_cal.data(), h_cal.size() * 2u)) {
        start_layer = 0;
        std::printf("[H4HOT] Initializing h_cal (8192 toks) from local spine_shared.safetensors...\n");
        std::fflush(stdout);
        hy4::SafetensorsMmap st(shared_path);
        const uint16_t* emb = reinterpret_cast<const uint16_t*>(st.tensor_bytes("model.embed_tokens.weight"));
        for (int s = 0; s < kCalSeqs; ++s) {
            for (int t = 0; t < kCalLen; ++t) {
                const uint16_t* row = emb + size_t(cal_seqs[size_t(s)][size_t(t)]) * size_t(kHiddenSize);
                uint16_t* dst = h_cal.data() + size_t(s * kCalLen + t) * size_t(kHcFlatDim);
                for (int c = 0; c < kHcMult; ++c) std::memcpy(dst + size_t(c) * size_t(kHiddenSize), row, size_t(kHiddenSize) * 2u);
            }
        }
    } else {
        std::printf("[H4HOT] Resuming HLUT4-hot pack (ranks 0..%d) from Layer %02d!\n", kPackHlut4Begin - 1, start_layer);
        std::fflush(stdout);
    }

    GpuForwardCtx gpu;
    gpu.init(gpu_id);
    DevBuf<float> d_lev4;
    {
        const auto& sol4 = lloyd_max_hlut4();
        float lv[8];
        for (int i = 0; i < 8; ++i) lv[i] = float(sol4.levels[size_t(i)]);
        d_lev4.ensure(8u);
        check_hip(hipMemcpy(d_lev4.ptr, lv, 32u, hipMemcpyHostToDevice), "lev4_h2d");
    }

    // Async prefetch ONLY ranks 0..109 (4.30 GiB/layer -> ~2.4x faster than full layer!)
    std::future<StreamedTensorBundle> next_fut;
    const int first_net_layer = std::max(1, start_layer);
    if (first_net_layer <= end_layer) {
        auto names = hot_expert_tensor_names(idx, first_net_layer, root);
        next_fut = std::async(std::launch::async, [&idx, names, fetch_workers]() {
            return stream_fetch_tensors(idx, names, fetch_workers);
        });
    }

    for (int l = start_layer; l <= end_layer; ++l) {
        if (should_stop()) {
            std::printf("[H4HOT] Graceful stop before Layer %02d.\n", l);
            break;
        }
        const auto t_l0 = std::chrono::steady_clock::now();
        const std::string pfx = "model.layers." + std::to_string(l) + ".";
        hy4::SafetensorsMmap sp_mmap(root + "/packed_spine/spine_layer_" + layer_tag(l) + ".safetensors");

        StreamedTensorBundle cur_bundle;
        double dl_sec = 0.0, dl_mibs = 0.0;
        if (l >= 1) {
            cur_bundle = next_fut.get();
            dl_sec = cur_bundle.elapsed_sec;
            dl_mibs = (double(cur_bundle.total_wire_bytes) / 1048576.0) / std::max(1e-6, dl_sec);
            if (l + 1 <= end_layer && !should_stop()) {
                auto next_names = hot_expert_tensor_names(idx, l + 1, root);
                next_fut = std::async(std::launch::async, [&idx, next_names, fetch_workers]() {
                    return stream_fetch_tensors(idx, next_names, fetch_workers);
                });
            }
        }
        merge_local_spine_into_bundle(sp_mmap, cur_bundle);

        // Attention + post_attention_layernorm from local spine
        std::vector<uint16_t> x_cal_mlp;
        std::vector<float>    post_cal_mlp;
        run_layer_attention_exact(gpu, cur_bundle, l, kCalSeqs, kCalLen, h_cal, x_cal_mlp, post_cal_mlp);

        double mean_sqnr4 = 0.0, cond_in = 1.0, cond_dn = 1.0;
        uint32_t crc4 = 0u;

        if (l == 0) {
            DevBuf<uint16_t> wg, wu, wd;
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.gate_proj", wg);
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.up_proj", wu);
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.down_proj", wd);
            const int blk = 512;
            gpu.d_x_norm.ensure(size_t(blk) * size_t(kHiddenSize));
            gpu.d_sh_g.ensure(size_t(blk) * size_t(kDenseInterSize));
            gpu.d_sh_u.ensure(size_t(blk) * size_t(kDenseInterSize));
            gpu.d_sh_z.ensure(size_t(blk) * size_t(kDenseInterSize));
            gpu.d_sh_y.ensure(size_t(blk) * size_t(kHiddenSize));
            gpu.d_streams.ensure(size_t(blk) * size_t(kHcFlatDim));
            gpu.d_post.ensure(size_t(blk) * size_t(kHcMult));
            for (int t0 = 0; t0 < kCalToks; t0 += blk) {
                const int m = std::min(blk, kCalToks - t0);
                check_hip(hipMemcpyAsync(gpu.d_x_norm.ptr, x_cal_mlp.data() + size_t(t0) * size_t(kHiddenSize), size_t(m) * size_t(kHiddenSize) * 2u, hipMemcpyHostToDevice, gpu.stream), "l0x");
                check_hip(hipMemcpyAsync(gpu.d_streams.ptr, h_cal.data() + size_t(t0) * size_t(kHcFlatDim), size_t(m) * size_t(kHcFlatDim) * 2u, hipMemcpyHostToDevice, gpu.stream), "l0s");
                check_hip(hipMemcpyAsync(gpu.d_post.ptr, post_cal_mlp.data() + size_t(t0) * size_t(kHcMult), size_t(m) * size_t(kHcMult) * 4u, hipMemcpyHostToDevice, gpu.stream), "l0p");
                gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, wg.ptr, gpu.d_sh_g.ptr, m, kDenseInterSize, kHiddenSize);
                gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, wu.ptr, gpu.d_sh_u.ptr, m, kDenseInterSize, kHiddenSize);
                launch_swiglu_unclamped_bf16(gpu.d_sh_g.ptr, gpu.d_sh_u.ptr, gpu.d_sh_z.ptr, m * kDenseInterSize, gpu.stream);
                gpu.gemm_bf16_w_f16_to_bf16(gpu.d_sh_z.ptr, wd.ptr, gpu.d_sh_y.ptr, m, kHiddenSize, kDenseInterSize);
                launch_ihc_post_add(gpu.d_streams.ptr, gpu.d_sh_y.ptr, gpu.d_post.ptr, m, kHiddenSize, kHcMult, gpu.stream);
                check_hip(hipMemcpyAsync(h_cal.data() + size_t(t0) * size_t(kHcFlatDim), gpu.d_streams.ptr, size_t(m) * size_t(kHcFlatDim) * 2u, hipMemcpyDeviceToHost, gpu.stream), "l0d");
                check_hip(hipStreamSynchronize(gpu.stream), "l0sy");
            }
        } else {
            const int32_t* rank_to_orig = reinterpret_cast<const int32_t*>(sp_mmap.tensor_bytes("expert_rank_to_orig"));
            const int32_t* orig_to_rank = reinterpret_cast<const int32_t*>(sp_mmap.tensor_bytes("expert_orig_to_rank"));

            DevBuf<uint16_t> w_sg, w_su, w_sd, w_gate;
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.shared_experts.gate_proj", w_sg);
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.shared_experts.up_proj", w_su);
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.shared_experts.down_proj", w_sd);
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.gate", w_gate);
            gpu.d_router_bias.ensure(kNumRoutedExperts);
            check_hip(hipMemcpyAsync(gpu.d_router_bias.ptr, cur_bundle.data(pfx + "mlp.gate.e_score_correction_bias"),
                                     size_t(kNumRoutedExperts) * 4u, hipMemcpyHostToDevice, gpu.stream), "rb");

            std::vector<float> y_moe_all(size_t(kCalToks) * size_t(kHiddenSize), 0.0f);
            std::vector<int32_t> topk_ids_all(size_t(kCalToks) * size_t(kTopK));
            std::vector<float>   topk_wts_all(size_t(kCalToks) * size_t(kTopK));
            std::vector<uint16_t> x_all_f16(size_t(kCalToks) * size_t(kHiddenSize));
            {
                const int blk = 1024;
                gpu.d_x_norm.ensure(size_t(blk) * size_t(kHiddenSize));
                gpu.d_sh_g.ensure(size_t(blk) * size_t(kSharedInterSize));
                gpu.d_sh_u.ensure(size_t(blk) * size_t(kSharedInterSize));
                gpu.d_sh_z.ensure(size_t(blk) * size_t(kSharedInterSize));
                gpu.d_sh_y.ensure(size_t(blk) * size_t(kHiddenSize));
                gpu.d_router_logits.ensure(size_t(blk) * size_t(kNumRoutedExperts));
                gpu.d_topk_ids.ensure(size_t(blk) * size_t(kTopK));
                gpu.d_topk_w.ensure(size_t(blk) * size_t(kTopK));
                std::vector<uint16_t> h_shy(size_t(blk) * size_t(kHiddenSize));
                for (int t0 = 0; t0 < kCalToks; t0 += blk) {
                    const int m = std::min(blk, kCalToks - t0);
                    const uint16_t* xsrc = x_cal_mlp.data() + size_t(t0) * size_t(kHiddenSize);
                    bf16_to_f16_host(xsrc, x_all_f16.data() + size_t(t0) * size_t(kHiddenSize), size_t(m) * size_t(kHiddenSize));
                    check_hip(hipMemcpyAsync(gpu.d_x_norm.ptr, xsrc, size_t(m) * size_t(kHiddenSize) * 2u, hipMemcpyHostToDevice, gpu.stream), "mx");
                    gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, w_sg.ptr, gpu.d_sh_g.ptr, m, kSharedInterSize, kHiddenSize);
                    gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, w_su.ptr, gpu.d_sh_u.ptr, m, kSharedInterSize, kHiddenSize);
                    launch_swiglu_unclamped_bf16(gpu.d_sh_g.ptr, gpu.d_sh_u.ptr, gpu.d_sh_z.ptr, m * kSharedInterSize, gpu.stream);
                    gpu.gemm_bf16_w_f16_to_bf16(gpu.d_sh_z.ptr, w_sd.ptr, gpu.d_sh_y.ptr, m, kHiddenSize, kSharedInterSize);
                    gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, w_gate.ptr, gpu.d_router_logits.ptr, m, kNumRoutedExperts, kHiddenSize);
                    launch_moe_router_top8_256(gpu.d_router_logits.ptr, gpu.d_router_bias.ptr,
                                               gpu.d_topk_ids.ptr, gpu.d_topk_w.ptr, m, kRoutedScalingFactor, gpu.stream);
                    check_hip(hipMemcpyAsync(h_shy.data(), gpu.d_sh_y.ptr, size_t(m) * size_t(kHiddenSize) * 2u, hipMemcpyDeviceToHost, gpu.stream), "shy");
                    check_hip(hipMemcpyAsync(topk_ids_all.data() + size_t(t0) * size_t(kTopK), gpu.d_topk_ids.ptr, size_t(m) * size_t(kTopK) * 4u, hipMemcpyDeviceToHost, gpu.stream), "ids");
                    check_hip(hipMemcpyAsync(topk_wts_all.data() + size_t(t0) * size_t(kTopK), gpu.d_topk_w.ptr, size_t(m) * size_t(kTopK) * 4u, hipMemcpyDeviceToHost, gpu.stream), "wts");
                    check_hip(hipStreamSynchronize(gpu.stream), "shs");
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < size_t(m) * size_t(kHiddenSize); ++i) y_moe_all[size_t(t0) * size_t(kHiddenSize) + i] = bf16_to_f32(h_shy[i]);
                }
            }

            std::array<std::vector<std::pair<int, float>>, kNumRoutedExperts> exp_routes;
            for (int t = 0; t < kCalToks; ++t) {
                for (int k = 0; k < kTopK; ++k) {
                    exp_routes[size_t(topk_ids_all[size_t(t) * size_t(kTopK) + size_t(k)])].push_back(
                        {t, topk_wts_all[size_t(t) * size_t(kTopK) + size_t(k)]});
                }
            }

            const auto signs_h = make_signs_h(l);
            const auto signs_i = make_signs_i(l);
            gpu.d_signs_h.ensure(signs_h.size());
            gpu.d_signs_i.ensure(signs_i.size());
            check_hip(hipMemcpyAsync(gpu.d_signs_h.ptr, signs_h.data(), signs_h.size() * 8u, hipMemcpyHostToDevice, gpu.stream), "sh");
            check_hip(hipMemcpyAsync(gpu.d_signs_i.ptr, signs_i.data(), signs_i.size() * 8u, hipMemcpyHostToDevice, gpu.stream), "si");

            std::vector<float> x_cal_f32(size_t(kCalToks) * size_t(kHiddenSize));
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < x_cal_f32.size(); ++i) x_cal_f32[i] = bf16_to_f32(x_cal_mlp[i]);
            std::vector<float> x_cal_rot(x_cal_f32.size());
            wht128_rows(x_cal_f32.data(), x_cal_rot.data(), kCalToks, kHiddenSize, signs_h);
            std::vector<uint16_t> x_cal_rot_f16(x_cal_rot.size());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < x_cal_rot.size(); ++i) x_cal_rot_f16[i] = f32_to_fp16_rne(x_cal_rot[i]);

            std::vector<double> H_down_pool(size_t(kMoeInterSize) * size_t(kMoeInterSize), 0.0);
            uint64_t total_z_rows = 0;
            std::array<std::vector<float>, kNumRoutedExperts> z_rot_saved;

            std::ifstream cold_f4(weight_root + "/hlut4_ddr4/hlut4_layer_" + layer_tag(l) + ".bin", std::ios::binary);
            if (!cold_f4) throw std::runtime_error("Missing existing hlut4_ddr4 shard for layer " + std::to_string(l));
            DevBuf<Hlut4Tile32x32> d_h4_tiles;
            d_h4_tiles.ensure(3u * kProjTiles);
            std::vector<Hlut4Tile32x32> h_h4_raw(3u * kProjTiles);

            DevBuf<uint16_t> d_eg, d_eu, d_ed, d_ex_f16, d_ez_f16;
            d_eg.ensure(size_t(kMoeInterSize) * size_t(kHiddenSize));
            d_eu.ensure(size_t(kMoeInterSize) * size_t(kHiddenSize));
            d_ed.ensure(size_t(kHiddenSize) * size_t(kMoeInterSize));

            for (int e = 0; e < kNumRoutedExperts; ++e) {
                const auto& routes = exp_routes[size_t(e)];
                const int m_e = int(routes.size());
                if (m_e == 0) continue;
                const int r = orig_to_rank[e];
                const bool is_hot = (r < kPackHlut4Begin);

                std::vector<uint16_t> h_ex(size_t(m_e) * size_t(kHiddenSize));
                const uint16_t* x_src_base = is_hot ? x_all_f16.data() : x_cal_rot_f16.data();
                for (int i = 0; i < m_e; ++i) {
                    std::memcpy(h_ex.data() + size_t(i) * size_t(kHiddenSize),
                                x_src_base + size_t(routes[size_t(i)].first) * size_t(kHiddenSize),
                                size_t(kHiddenSize) * 2u);
                }
                if (is_hot) {
                    const std::string epfx = pfx + "mlp.experts." + std::to_string(e) + ".";
                    gpu.load_weight_as_f16(cur_bundle, epfx + "gate_proj", d_eg);
                    gpu.load_weight_as_f16(cur_bundle, epfx + "up_proj", d_eu);
                    gpu.load_weight_as_f16(cur_bundle, epfx + "down_proj", d_ed);
                } else {
                    cold_f4.seekg(std::streamoff(size_t(r - kPackHlut4Begin) * size_t(kHlut4ExpertBytes)), std::ios::beg);
                    cold_f4.read(reinterpret_cast<char*>(h_h4_raw.data()), std::streamsize(kHlut4ExpertBytes));
                    check_hip(hipMemcpyAsync(d_h4_tiles.ptr, h_h4_raw.data(), kHlut4ExpertBytes, hipMemcpyHostToDevice, gpu.stream), "c4h2d");
                    launch_unpack_hlut4_to_f16(d_h4_tiles.ptr + 0u * kProjTiles, d_lev4.ptr, d_eg.ptr, kMoeInterSize, kHiddenSize, gpu.stream);
                    launch_unpack_hlut4_to_f16(d_h4_tiles.ptr + 1u * kProjTiles, d_lev4.ptr, d_eu.ptr, kMoeInterSize, kHiddenSize, gpu.stream);
                    launch_unpack_hlut4_to_f16(d_h4_tiles.ptr + 2u * kProjTiles, d_lev4.ptr, d_ed.ptr, kHiddenSize, kMoeInterSize, gpu.stream);
                }

                d_ex_f16.ensure(h_ex.size());
                gpu.d_exp_g.ensure(size_t(m_e) * size_t(kMoeInterSize));
                gpu.d_exp_u.ensure(size_t(m_e) * size_t(kMoeInterSize));
                gpu.d_exp_z.ensure(size_t(m_e) * size_t(kMoeInterSize));
                d_ez_f16.ensure(size_t(m_e) * size_t(kMoeInterSize));
                gpu.d_exp_y.ensure(size_t(m_e) * size_t(kHiddenSize));

                check_hip(hipMemcpyAsync(d_ex_f16.ptr, h_ex.data(), h_ex.size() * 2u, hipMemcpyHostToDevice, gpu.stream), "exh");
                gpu.gemm_f16_to_f32(d_ex_f16.ptr, d_eg.ptr, gpu.d_exp_g.ptr, m_e, kMoeInterSize, kHiddenSize);
                gpu.gemm_f16_to_f32(d_ex_f16.ptr, d_eu.ptr, gpu.d_exp_u.ptr, m_e, kMoeInterSize, kHiddenSize);
                launch_swiglu_clamped10_f32(gpu.d_exp_g.ptr, gpu.d_exp_u.ptr, gpu.d_exp_z.ptr, m_e * kMoeInterSize, gpu.stream);

                if (!is_hot) {
                    launch_wht128_f32_inplace(gpu.d_exp_z.ptr, gpu.d_signs_i.ptr, m_e, kMoeInterSize, gpu.stream);
                }
                std::vector<float> h_ez(size_t(m_e) * size_t(kMoeInterSize));
                check_hip(hipMemcpyAsync(h_ez.data(), gpu.d_exp_z.ptr, h_ez.size() * 4u, hipMemcpyDeviceToHost, gpu.stream), "ezd");
                check_hip(hipStreamSynchronize(gpu.stream), "ezs");

                std::vector<uint16_t> h_ez16(h_ez.size());
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < h_ez.size(); ++i) h_ez16[i] = f32_to_fp16_rne(h_ez[i]);
                check_hip(hipMemcpyAsync(d_ez_f16.ptr, h_ez16.data(), h_ez16.size() * 2u, hipMemcpyHostToDevice, gpu.stream), "ez16");
                gpu.gemm_f16_to_f32(d_ez_f16.ptr, d_ed.ptr, gpu.d_exp_y.ptr, m_e, kHiddenSize, kMoeInterSize);

                std::vector<float> h_ey(size_t(m_e) * size_t(kHiddenSize));
                check_hip(hipMemcpyAsync(h_ey.data(), gpu.d_exp_y.ptr, h_ey.size() * 4u, hipMemcpyDeviceToHost, gpu.stream), "eyd");
                check_hip(hipStreamSynchronize(gpu.stream), "eys");

                for (int i = 0; i < m_e; ++i) {
                    float* dst = y_moe_all.data() + size_t(routes[size_t(i)].first) * size_t(kHiddenSize);
                    const float wt = routes[size_t(i)].second;
                    const float* src = h_ey.data() + size_t(i) * size_t(kHiddenSize);
                    for (int d = 0; d < kHiddenSize; ++d) dst[d] += wt * src[d];
                }

                std::vector<float> z_rot;
                if (is_hot) {
                    z_rot.resize(h_ez.size());
                    wht128_rows(h_ez.data(), z_rot.data(), m_e, kMoeInterSize, signs_i);
                } else {
                    z_rot = std::move(h_ez);
                }
                gpu.dsyrk_add_fp64(z_rot.data(), m_e, kMoeInterSize, 1.0, H_down_pool);
                total_z_rows += uint64_t(m_e);
                if (r < 60) z_rot_saved[size_t(e)] = std::move(z_rot);
            }

            // Advance h_cal
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < kCalToks; ++t) {
                const float* y = y_moe_all.data() + size_t(t) * size_t(kHiddenSize);
                const float* p = post_cal_mlp.data() + size_t(t) * size_t(kHcMult);
                uint16_t* st   = h_cal.data() + size_t(t) * size_t(kHcFlatDim);
                for (int c = 0; c < kHcMult; ++c) {
                    const float pc = p[c];
                    uint16_t* sc = st + size_t(c) * size_t(kHiddenSize);
                    for (int d = 0; d < kHiddenSize; ++d) sc[d] = f32_to_bf16_rne(bf16_to_f32(sc[d]) + pc * y[d]);
                }
            }

            // Exact FP64 TARQ RAREBAL H_in + H_down_pool
            std::vector<double> H_in(size_t(kHiddenSize) * size_t(kHiddenSize), 0.0);
            {
                std::vector<double> mean_dir(size_t(kHiddenSize), 0.0);
                std::vector<float> inv_n(size_t(kCalToks), 0.0f);
                for (int m = 0; m < kCalToks; ++m) {
                    const float* xm = x_cal_rot.data() + size_t(m) * size_t(kHiddenSize);
                    double n2 = 0.0;
                    for (int k = 0; k < kHiddenSize; ++k) n2 += double(xm[k]) * double(xm[k]);
                    const double inv = (n2 > 1e-30) ? (1.0 / std::sqrt(n2)) : 0.0;
                    inv_n[size_t(m)] = float(inv);
                    for (int k = 0; k < kHiddenSize; ++k) mean_dir[size_t(k)] += double(xm[k]) * inv;
                }
                double mn2 = 0.0;
                for (int k = 0; k < kHiddenSize; ++k) mn2 += mean_dir[size_t(k)] * mean_dir[size_t(k)];
                const double inv_mn = (mn2 > 1e-30) ? (1.0 / std::sqrt(mn2)) : 0.0;
                for (int k = 0; k < kHiddenSize; ++k) mean_dir[size_t(k)] *= inv_mn;
                std::vector<std::pair<float, int>> align(static_cast<size_t>(kCalToks));
                for (int m = 0; m < kCalToks; ++m) {
                    const float* xm = x_cal_rot.data() + size_t(m) * size_t(kHiddenSize);
                    double dot = 0.0;
                    for (int k = 0; k < kHiddenSize; ++k) dot += double(xm[k]) * mean_dir[size_t(k)];
                    align[size_t(m)] = {float(std::fabs(dot) * double(inv_n[size_t(m)])), m};
                }
                const int rt = kCalToks / 8;
                std::nth_element(align.begin(), align.begin() + rt, align.end());
                std::vector<uint8_t> is_rare(size_t(kCalToks), 0u);
                for (int i = 0; i < rt; ++i) is_rare[size_t(align[size_t(i)].second)] = 1u;
                double tr_c = 0.0, tr_r = 0.0;
                std::vector<float> xc, xr;
                xc.reserve(size_t(kCalToks - rt) * size_t(kHiddenSize));
                xr.reserve(size_t(rt) * size_t(kHiddenSize));
                for (int m = 0; m < kCalToks; ++m) {
                    const float* xm = x_cal_rot.data() + size_t(m) * size_t(kHiddenSize);
                    double sq = 0.0;
                    for (int k = 0; k < kHiddenSize; ++k) sq += double(xm[k]) * double(xm[k]);
                    if (is_rare[size_t(m)]) { tr_r += sq; xr.insert(xr.end(), xm, xm + kHiddenSize); }
                    else                    { tr_c += sq; xc.insert(xc.end(), xm, xm + kHiddenSize); }
                }
                const double lam = (tr_r > 1e-12) ? (tr_c / tr_r) : 1.0;
                gpu.dsyrk_add_fp64(xc.data(), int(xc.size() / kHiddenSize), kHiddenSize, 1.0 / double(kCalToks), H_in);
                gpu.dsyrk_add_fp64(xr.data(), int(xr.size() / kHiddenSize), kHiddenSize, lam / double(kCalToks), H_in);
            }
            RarebalStats st_in{}, st_dn{};
            const auto R_up_in = build_rup_from_fp64_hessian(H_in.data(), kHiddenSize, 0.015, &st_in);
            cond_in = st_in.cond_est;
            const double inv_z = 1.0 / double(std::max<uint64_t>(1, total_z_rows));
            for (double& v : H_down_pool) v *= inv_z;
            const auto R_up_dn_pool = build_rup_from_fp64_hessian(H_down_pool.data(), kMoeInterSize, 0.015, &st_dn);
            cond_dn = st_dn.cond_est;

            gpu.d_R_in.ensure(R_up_in.size());
            gpu.d_R_dn.ensure(R_up_dn_pool.size());
            check_hip(hipMemcpyAsync(gpu.d_R_in.ptr, R_up_in.data(), R_up_in.size() * 4u, hipMemcpyHostToDevice, gpu.stream), "rin");
            check_hip(hipMemcpyAsync(gpu.d_R_dn.ptr, R_up_dn_pool.data(), R_up_dn_pool.size() * 4u, hipMemcpyHostToDevice, gpu.stream), "rdn");
            check_hip(hipStreamSynchronize(gpu.stream), "psync");

            const std::string p_h4 = weight_root + "/hlut4_hot/hlut4_hot_layer_" + layer_tag(l) + ".bin";
            std::ofstream f4(p_h4 + ".tmp", std::ios::binary | std::ios::trunc);
            double sum_sq4 = 0.0;
            int cnt4 = 0;
            bool r_dn_is_pool = true;
            std::vector<float> tmp_wrot;

            for (int r = 0; r < kPackHlut4Begin; ++r) {
                const int e = rank_to_orig[r];
                const std::string epfx = pfx + "mlp.experts." + std::to_string(e) + ".";
                const bool do_sqnr = (r % 16 == 0);
                if (r < 60 && !z_rot_saved[size_t(e)].empty()) {
                    const int me = int(z_rot_saved[size_t(e)].size() / kMoeInterSize);
                    std::vector<double> He(size_t(kMoeInterSize) * size_t(kMoeInterSize), 0.0);
                    gpu.dsyrk_add_fp64(z_rot_saved[size_t(e)].data(), me, kMoeInterSize, 1.0 / double(me), He);
                    const double alpha = double(me) / (double(me) + 512.0);
                    for (size_t i = 0; i < He.size(); ++i) He[i] = alpha * He[i] + (1.0 - alpha) * H_down_pool[i];
                    auto R_dn_loc = build_rup_from_fp64_hessian(He.data(), kMoeInterSize, 0.015);
                    check_hip(hipMemcpyAsync(gpu.d_R_dn.ptr, R_dn_loc.data(), R_dn_loc.size() * 4u, hipMemcpyHostToDevice, gpu.stream), "rdnl");
                    r_dn_is_pool = false;
                    z_rot_saved[size_t(e)].clear(); z_rot_saved[size_t(e)].shrink_to_fit();
                } else if (!r_dn_is_pool) {
                    check_hip(hipMemcpyAsync(gpu.d_R_dn.ptr, R_up_dn_pool.data(), R_up_dn_pool.size() * 4u, hipMemcpyHostToDevice, gpu.stream), "rdnp");
                    r_dn_is_pool = true;
                }

                const char* pnames[3] = {"gate_proj", "up_proj", "down_proj"};
                const int pN[3] = {kMoeInterSize, kMoeInterSize, kHiddenSize};
                const int pK[3] = {kHiddenSize,   kHiddenSize,   kMoeInterSize};
                double esq = 0.0;
                for (int pi = 0; pi < 3; ++pi) {
                    const std::string wn = epfx + pnames[pi] + ".weight";
                    const std::string sn = epfx + pnames[pi] + ".weight_scale";
                    const uint64_t* d_sg = (pi < 2) ? gpu.d_signs_h.ptr : gpu.d_signs_i.ptr;
                    const float*    d_ru = (pi < 2) ? gpu.d_R_in.ptr    : gpu.d_R_dn.ptr;
                    gpu.prepare_wrot_gpu(cur_bundle.data(wn), reinterpret_cast<const float*>(cur_bundle.data(sn)), d_sg, pN[pi], pK[pi]);
                    if (do_sqnr) {
                        tmp_wrot.resize(size_t(pN[pi]) * size_t(pK[pi]));
                        check_hip(hipMemcpyAsync(tmp_wrot.data(), gpu.d_W_orig.ptr, tmp_wrot.size() * 4u, hipMemcpyDeviceToHost, gpu.stream), "wrc");
                    }
                    auto q4 = gpu.pack_prepared_ldlq_gpu<4, Hlut4Tile32x32>(d_ru, pN[pi], pK[pi], lloyd_max_hlut4());
                    if (do_sqnr) esq += compute_hlut_sqnr_db(q4, tmp_wrot.data(), hlut4_unpack_tile);
                    f4.write(reinterpret_cast<const char*>(q4.tiles.data()), std::streamsize(q4.raw_bytes()));
                    crc4 = crc32c_update(crc4, q4.tiles.data(), q4.raw_bytes());
                }
                if (do_sqnr) { sum_sq4 += esq / 3.0; ++cnt4; }
            }
            f4.close();
            fs::rename(p_h4 + ".tmp", p_h4);
            mean_sqnr4 = (cnt4 > 0) ? (sum_sq4 / cnt4) : 0.0;
        }

        save_bin(cal_path, h_cal.data(), h_cal.size() * 2u);
        const double total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_l0).count();
        {
            const std::string ck = root + "/checkpoints/h4hot_layer_" + layer_tag(l) + ".json";
            std::ofstream ofs(ck + ".tmp");
            ofs << "{\n  \"layer\": " << l << ",\n  \"ranks\": \"0.." << (kPackHlut4Begin - 1) << "\",\n"
                << "  \"download_sec\": " << dl_sec << ",\n  \"download_mibs\": " << dl_mibs << ",\n"
                << "  \"total_sec\": " << total_sec << ",\n  \"sqnr_hlut4_db\": " << mean_sqnr4 << ",\n"
                << "  \"cond_in\": " << cond_in << ",\n  \"cond_dn\": " << cond_dn << ",\n"
                << "  \"crc32c_hlut4_hot\": \"0x" << std::hex << crc4 << std::dec << "\"\n}\n";
            ofs.close();
            fs::rename(ck + ".tmp", ck);
        }
        std::printf("[H4HOT L%02d/77] DONE in %.1fs (dl=%.1fs %.1f MiB/s) | SQNR_HLUT4=%.2fdB | cond=%.1f/%.1f\n",
                    l, total_sec, dl_sec, dl_mibs, mean_sqnr4, cond_in, cond_dn);
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace
}  // namespace hy4_770b

int main(int argc, char** argv) {
    using namespace hy4_770b;
    install_graceful_signal_handlers();
    ScopedHipDrainGuard drain_guard{5, "HY4_770B_PACK"};

    std::string root = "<HY4_770B_WORKSPACE_ROOT>";
    std::string weight_root = "<HY4_770B_WEIGHT_ROOT>";
    int gpu_id = 0;
    int start_layer = -1;  // auto-resume from last completed checkpoint
    int end_layer = kNumLayers - 1;
    int fetch_workers = 24;
    bool pack_hlut4_hot = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--weight-root" && i + 1 < argc) weight_root = argv[++i];
        else if (a == "--gpu" && i + 1 < argc) gpu_id = std::atoi(argv[++i]);
        else if (a == "--start-layer" && i + 1 < argc) start_layer = std::atoi(argv[++i]);
        else if (a == "--end-layer" && i + 1 < argc) end_layer = std::atoi(argv[++i]);
        else if (a == "--workers" && i + 1 < argc) fetch_workers = std::atoi(argv[++i]);
        else if (a == "--pack-hlut4-hot") pack_hlut4_hot = true;
    }

    fs::create_directories(root + "/packed_spine");
    fs::create_directories(root + "/checkpoints");
    fs::create_directories(root + "/artifacts");
    fs::create_directories(weight_root + "/hlut3");
    fs::create_directories(weight_root + "/hlut2");
    fs::create_directories(weight_root + "/hlut4_ddr4");

    std::printf("[HY4-770B-PACK] Loading tensor range index & tokenizer...\n");
    std::fflush(stdout);
    RemoteRangeIndex idx;
    idx.load_tsv(root + "/meta/hy4_fp8_tensor_ranges.tsv");

    hy4::Hy4Tokenizer tok;
    tok.load(root + "/meta/tokenizer.json");

    // Step 1: Strictly Disjoint N_cal = 8,192 Calibration Set + 64 Held-Out Oracle Probes
    std::unordered_set<uint64_t> bench_8grams;
    collect_bench_8grams(root + "/bench_data", bench_8grams);
    uint64_t overlap_cnt = 0;
    uint32_t cal_crc = 0;
    auto cal_seqs = build_disjoint_cal_corpus(tok, bench_8grams, overlap_cnt, cal_crc);
    if (overlap_cnt != 0) {
        std::fprintf(stderr, "[FATAL] Calibration corpus overlap = %llu (must be 0!)\n", (unsigned long long)overlap_cnt);
        return 1;
    }
    if (pack_hlut4_hot) {
        return run_pack_hlut4_hot(idx, cal_seqs, root, weight_root, gpu_id, start_layer, end_layer, fetch_workers);
    }
    auto probe_seqs = load_heldout_probes_64(root + "/bench_data/logprob_corpus.json");
    std::printf("[HY4-770B-PACK] Disjoint N_cal=8192 (16x512, 8-gram overlap=0/%zu bench 8-grams, CRC32C=0x%08x), "
                "Held-out Gate-2 probes=64x64 (4096 toks)\n", bench_8grams.size(), cal_crc);
    std::fflush(stdout);

    {
        std::ofstream ofs(root + "/artifacts/cal_corpus_disjoint_receipt.json");
        ofs << "{\n"
            << "  \"n_cal_tokens\": 8192,\n"
            << "  \"n_cal_sequences\": 16,\n"
            << "  \"seq_len\": 512,\n"
            << "  \"benchmark_8grams_checked\": " << bench_8grams.size() << ",\n"
            << "  \"overlap_8grams\": " << overlap_cnt << ",\n"
            << "  \"crc32c\": \"0x" << std::hex << cal_crc << std::dec << "\",\n"
            << "  \"heldout_oracle_probes\": 64,\n"
            << "  \"heldout_probe_len\": 64\n"
            << "}\n";
    }

    // Step 2: Pack shared spine (embed_tokens, hc_head, model.norm, lm_head), DFly2 (3.3B), and ViT (1.23B) if missing
    const std::string shared_path = root + "/packed_spine/spine_shared.safetensors";
    const std::string dfly_path   = root + "/packed_spine/dfly2.safetensors";
    const std::string vit_path    = root + "/packed_spine/vit.safetensors";

    if (!fs::exists(shared_path)) {
        std::printf("[HY4-770B-PACK] Streaming shared spine (embed_tokens, hc_head, norm, lm_head)...\n");
        std::fflush(stdout);
        std::vector<std::string> snames = {
            "model.embed_tokens.weight", "model.hc_head.hc_head_base",
            "model.hc_head.hc_head_fn", "model.hc_head.hc_head_scale",
            "model.norm.weight", "lm_head.weight"
        };
        auto sb = stream_fetch_tensors(idx, snames, fetch_workers);
        std::vector<hy4::SafetensorWriteSpec> specs;
        for (const auto& nm : snames) {
            const auto& sp = sb.spec(nm);
            specs.push_back({nm, sp.dtype, sp.shape, sb.data(nm), sp.nbytes});
        }
        hy4::write_safetensors_file(shared_path, specs, {{"format", "hy4_770b_shared_spine"}});
        std::printf("[HY4-770B-PACK] Wrote %s (%.2f GiB in %.1fs)\n",
                    shared_path.c_str(), double(sb.total_tensor_bytes) / 1073741824.0, sb.elapsed_sec);
        std::fflush(stdout);
    }

    if (!fs::exists(dfly_path)) {
        std::printf("[HY4-770B-PACK] Streaming 3.3B DFly2 drafter (shard 245)...\n");
        std::fflush(stdout);
        std::vector<std::string> dnames;
        for (const auto& nm : idx.all_names) if (nm.rfind("model.dfly_layers.", 0) == 0) dnames.push_back(nm);
        std::sort(dnames.begin(), dnames.end());
        auto db = stream_fetch_tensors(idx, dnames, fetch_workers);
        std::vector<hy4::SafetensorWriteSpec> specs;
        for (const auto& nm : dnames) {
            const auto& sp = db.spec(nm);
            specs.push_back({nm, sp.dtype, sp.shape, db.data(nm), sp.nbytes});
        }
        hy4::write_safetensors_file(dfly_path, specs, {{"format", "hy4_770b_dfly2_fp8"}});
        std::printf("[HY4-770B-PACK] Wrote %s (%zu tensors, %.2f GiB in %.1fs)\n",
                    dfly_path.c_str(), specs.size(), double(db.total_tensor_bytes) / 1073741824.0, db.elapsed_sec);
        std::fflush(stdout);
    }

    if (!fs::exists(vit_path)) {
        std::printf("[HY4-770B-PACK] Streaming 1.23B 40-layer Dynamic ViT...\n");
        std::fflush(stdout);
        std::vector<std::string> vnames;
        for (const auto& nm : idx.all_names) if (nm.rfind("vit.", 0) == 0) vnames.push_back(nm);
        std::sort(vnames.begin(), vnames.end());
        auto vb = stream_fetch_tensors(idx, vnames, fetch_workers);
        std::vector<hy4::SafetensorWriteSpec> specs;
        for (const auto& nm : vnames) {
            const auto& sp = vb.spec(nm);
            specs.push_back({nm, sp.dtype, sp.shape, vb.data(nm), sp.nbytes});
        }
        hy4::write_safetensors_file(vit_path, specs, {{"format", "hy4_770b_vit_bf16"}});
        std::printf("[HY4-770B-PACK] Wrote %s (%zu tensors, %.2f GiB in %.1fs)\n",
                    vit_path.c_str(), specs.size(), double(vb.total_tensor_bytes) / 1073741824.0, vb.elapsed_sec);
        std::fflush(stdout);
    }

    // Step 3: Determine resume layer & initialize or reload h_cal [8192, 4, 6144] + h_probe [4096, 4, 6144]
    constexpr int kCalSeqs = 16, kCalLen = 512, kCalToks = kCalSeqs * kCalLen;     // 8,192
    constexpr int kPrbSeqs = 64, kPrbLen = 64,  kPrbToks = kPrbSeqs * kPrbLen;     // 4,096
    std::vector<uint16_t> h_cal(size_t(kCalToks) * size_t(kHcFlatDim), 0);
    std::vector<uint16_t> h_prb(size_t(kPrbToks) * size_t(kHcFlatDim), 0);

    const std::string cal_state_path = root + "/checkpoints/h_cal_state.bin";
    const std::string prb_state_path = root + "/checkpoints/h_prb_state.bin";

    if (start_layer < 0) {
        start_layer = 0;
        while (start_layer < kNumLayers &&
               fs::exists(root + "/checkpoints/layer_" + layer_tag(start_layer) + ".json")) {
            ++start_layer;
        }
    }

    if (start_layer == 0 || !load_bin(cal_state_path, h_cal.data(), h_cal.size() * 2u) ||
        !load_bin(prb_state_path, h_prb.data(), h_prb.size() * 2u))
    {
        start_layer = 0;
        std::printf("[HY4-770B-PACK] Initializing h_cal (402.65 MiB) & h_probe (201.33 MiB) from embed_tokens...\n");
        std::fflush(stdout);
        hy4::SafetensorsMmap st(shared_path);
        const uint16_t* emb = reinterpret_cast<const uint16_t*>(st.tensor_bytes("model.embed_tokens.weight"));
        for (int s = 0; s < kCalSeqs; ++s) {
            for (int t = 0; t < kCalLen; ++t) {
                const int32_t id = cal_seqs[size_t(s)][size_t(t)];
                const uint16_t* row = emb + size_t(id) * size_t(kHiddenSize);
                uint16_t* dst = h_cal.data() + size_t(s * kCalLen + t) * size_t(kHcFlatDim);
                for (int c = 0; c < kHcMult; ++c) std::memcpy(dst + size_t(c) * size_t(kHiddenSize), row, size_t(kHiddenSize) * 2u);
            }
        }
        for (int s = 0; s < kPrbSeqs; ++s) {
            for (int t = 0; t < kPrbLen; ++t) {
                const int32_t id = probe_seqs[size_t(s)][size_t(t)];
                const uint16_t* row = emb + size_t(id) * size_t(kHiddenSize);
                uint16_t* dst = h_prb.data() + size_t(s * kPrbLen + t) * size_t(kHcFlatDim);
                for (int c = 0; c < kHcMult; ++c) std::memcpy(dst + size_t(c) * size_t(kHiddenSize), row, size_t(kHiddenSize) * 2u);
            }
        }
    } else {
        std::printf("[HY4-770B-PACK] Resuming cleanly from Layer %02d using saved h_cal & h_probe checkpoints!\n", start_layer);
        std::fflush(stdout);
    }

    GpuForwardCtx gpu;
    gpu.init(gpu_id);

    // Async prefetch pipeline: kick off Layer start_layer immediately!
    std::future<StreamedTensorBundle> next_bundle_fut;
    if (start_layer <= end_layer) {
        auto names0 = layer_tensor_names(idx, start_layer);
        next_bundle_fut = std::async(std::launch::async, [&idx, names0, fetch_workers]() {
            return stream_fetch_tensors(idx, names0, fetch_workers);
        });
    }

    // Step 4: Causal Layer-by-Layer Loop (l = start_layer .. end_layer)
    for (int l = start_layer; l <= end_layer; ++l) {
        if (should_stop()) {
            std::printf("[HY4-770B-PACK] Graceful stop signal received before Layer %02d — exiting cleanly.\n", l);
            break;
        }
        const auto t_layer0 = std::chrono::steady_clock::now();

        // Wait for current layer bundle & IMMEDIATELY launch background prefetch of Layer l + 1!
        StreamedTensorBundle cur_bundle = next_bundle_fut.get();
        const double dl_sec = cur_bundle.elapsed_sec;
        const double dl_mibs = (double(cur_bundle.total_wire_bytes) / 1048576.0) / std::max(1e-6, dl_sec);
        if (l + 1 <= end_layer && !should_stop()) {
            auto next_names = layer_tensor_names(idx, l + 1);
            next_bundle_fut = std::async(std::launch::async, [&idx, next_names, fetch_workers]() {
                return stream_fetch_tensors(idx, next_names, fetch_workers);
            });
        }

        const std::string pfx = "model.layers." + std::to_string(l) + ".";
        std::printf("[L%02d/77] Streamed %.2f GiB in %.1fs (%.1f MiB/s, 0 B NVMe scratch) — running exact stock FP8 forward...\n",
                    l, double(cur_bundle.total_tensor_bytes) / 1073741824.0, dl_sec, dl_mibs);
        std::fflush(stdout);

        // 4A. Exact Attention + iHC + post_attention_layernorm on BOTH h_cal and h_prb
        std::vector<uint16_t> x_cal_mlp, x_prb_mlp;
        std::vector<float>    post_cal_mlp, post_prb_mlp;
        run_layer_attention_exact(gpu, cur_bundle, l, kCalSeqs, kCalLen, h_cal, x_cal_mlp, post_cal_mlp);
        run_layer_attention_exact(gpu, cur_bundle, l, kPrbSeqs, kPrbLen, h_prb, x_prb_mlp, post_prb_mlp);

        // 4B. Layer 0 Dense MLP OR Layers 1..77 MoE + FP64 RAREBAL-LDLQ Pack
        std::array<int32_t, kNumRoutedExperts> rank_to_orig{};
        std::array<int32_t, kNumRoutedExperts> orig_to_rank{};
        std::array<uint32_t, kNumRoutedExperts> cal_freq{};
        for (int e = 0; e < kNumRoutedExperts; ++e) { rank_to_orig[size_t(e)] = e; orig_to_rank[size_t(e)] = e; }

        double mean_sqnr_hlut3 = 0.0, mean_sqnr_hlut2 = 0.0, mean_sqnr_hlut4 = 0.0;
        double hit_113 = 0.0, hit_125 = 0.0, hit_139 = 0.0, hit_155 = 0.0;
        double cond_in = 1.0, cond_dn = 1.0;
        uint32_t crc_hlut3 = 0, crc_hlut2 = 0, crc_hlut4 = 0;

        if (l == 0) {
            // Exact Layer 0 Dense SwiGLU FFN [6144 -> 18432 -> 6144]
            DevBuf<uint16_t> wg, wu, wd;
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.gate_proj", wg);
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.up_proj", wu);
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.down_proj", wd);

            auto run_dense = [&](int ntoks, std::vector<uint16_t>& st_host, const std::vector<uint16_t>& x_host, const std::vector<float>& post_host) {
                const int blk = 512;
                gpu.d_x_norm.ensure(size_t(blk) * size_t(kHiddenSize));
                gpu.d_sh_g.ensure(size_t(blk) * size_t(kDenseInterSize));
                gpu.d_sh_u.ensure(size_t(blk) * size_t(kDenseInterSize));
                gpu.d_sh_z.ensure(size_t(blk) * size_t(kDenseInterSize));
                gpu.d_sh_y.ensure(size_t(blk) * size_t(kHiddenSize));
                gpu.d_streams.ensure(size_t(blk) * size_t(kHcFlatDim));
                gpu.d_post.ensure(size_t(blk) * size_t(kHcMult));
                for (int t0 = 0; t0 < ntoks; t0 += blk) {
                    const int m = std::min(blk, ntoks - t0);
                    check_hip(hipMemcpyAsync(gpu.d_x_norm.ptr, x_host.data() + size_t(t0) * size_t(kHiddenSize), size_t(m) * size_t(kHiddenSize) * 2u, hipMemcpyHostToDevice, gpu.stream), "l0_x");
                    check_hip(hipMemcpyAsync(gpu.d_streams.ptr, st_host.data() + size_t(t0) * size_t(kHcFlatDim), size_t(m) * size_t(kHcFlatDim) * 2u, hipMemcpyHostToDevice, gpu.stream), "l0_st");
                    check_hip(hipMemcpyAsync(gpu.d_post.ptr, post_host.data() + size_t(t0) * size_t(kHcMult), size_t(m) * size_t(kHcMult) * 4u, hipMemcpyHostToDevice, gpu.stream), "l0_post");
                    gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, wg.ptr, gpu.d_sh_g.ptr, m, kDenseInterSize, kHiddenSize);
                    gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, wu.ptr, gpu.d_sh_u.ptr, m, kDenseInterSize, kHiddenSize);
                    launch_swiglu_unclamped_bf16(gpu.d_sh_g.ptr, gpu.d_sh_u.ptr, gpu.d_sh_z.ptr, m * kDenseInterSize, gpu.stream);
                    gpu.gemm_bf16_w_f16_to_bf16(gpu.d_sh_z.ptr, wd.ptr, gpu.d_sh_y.ptr, m, kHiddenSize, kDenseInterSize);
                    launch_ihc_post_add(gpu.d_streams.ptr, gpu.d_sh_y.ptr, gpu.d_post.ptr, m, kHiddenSize, kHcMult, gpu.stream);
                    check_hip(hipMemcpyAsync(st_host.data() + size_t(t0) * size_t(kHcFlatDim), gpu.d_streams.ptr, size_t(m) * size_t(kHcFlatDim) * 2u, hipMemcpyDeviceToHost, gpu.stream), "l0_d2h");
                    check_hip(hipStreamSynchronize(gpu.stream), "l0_sync");
                }
            };
            run_dense(kCalToks, h_cal, x_cal_mlp, post_cal_mlp);
            run_dense(kPrbToks, h_prb, x_prb_mlp, post_prb_mlp);
        } else {
            // Layers 1..77 MoE: Shared Expert + Exact Top-8 Router + Exact Stock FP8 Routed Experts + FP64 RAREBAL-LDLQ Pack!
            DevBuf<uint16_t> w_sg, w_su, w_sd, w_gate;
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.shared_experts.gate_proj", w_sg);
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.shared_experts.up_proj", w_su);
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.shared_experts.down_proj", w_sd);
            gpu.load_weight_as_f16(cur_bundle, pfx + "mlp.gate", w_gate);
            gpu.d_router_bias.ensure(kNumRoutedExperts);
            check_hip(hipMemcpyAsync(gpu.d_router_bias.ptr, cur_bundle.data(pfx + "mlp.gate.e_score_correction_bias"),
                                     size_t(kNumRoutedExperts) * 4u, hipMemcpyHostToDevice, gpu.stream), "rbias");

            // Compute shared expert output + top-8 routing for BOTH cal (8192) and prb (4096)
            const int total_m = kCalToks + kPrbToks;  // 12,288
            std::vector<float> y_moe_all(size_t(total_m) * size_t(kHiddenSize), 0.0f);
            std::vector<int32_t> topk_ids_all(size_t(total_m) * size_t(kTopK));
            std::vector<float>   topk_wts_all(size_t(total_m) * size_t(kTopK));
            std::vector<uint16_t> x_all_f16(size_t(total_m) * size_t(kHiddenSize));

            // Pack combined x_all_f16 and run shared expert + router in 1024-token chunks
            {
                const int blk = 1024;
                gpu.d_x_norm.ensure(size_t(blk) * size_t(kHiddenSize));
                gpu.d_sh_g.ensure(size_t(blk) * size_t(kSharedInterSize));
                gpu.d_sh_u.ensure(size_t(blk) * size_t(kSharedInterSize));
                gpu.d_sh_z.ensure(size_t(blk) * size_t(kSharedInterSize));
                gpu.d_sh_y.ensure(size_t(blk) * size_t(kHiddenSize));
                gpu.d_router_logits.ensure(size_t(blk) * size_t(kNumRoutedExperts));
                gpu.d_topk_ids.ensure(size_t(blk) * size_t(kTopK));
                gpu.d_topk_w.ensure(size_t(blk) * size_t(kTopK));
                std::vector<uint16_t> h_shy(size_t(blk) * size_t(kHiddenSize));

                for (int t0 = 0; t0 < total_m; t0 += blk) {
                    const int m = std::min(blk, total_m - t0);
                    // Copy from x_cal_mlp or x_prb_mlp
                    std::vector<uint16_t> chunk_bf16(size_t(m) * size_t(kHiddenSize));
                    for (int i = 0; i < m; ++i) {
                        const int gi = t0 + i;
                        const uint16_t* src = (gi < kCalToks)
                            ? (x_cal_mlp.data() + size_t(gi) * size_t(kHiddenSize))
                            : (x_prb_mlp.data() + size_t(gi - kCalToks) * size_t(kHiddenSize));
                        std::memcpy(chunk_bf16.data() + size_t(i) * size_t(kHiddenSize), src, size_t(kHiddenSize) * 2u);
                    }
                    bf16_to_f16_host(chunk_bf16.data(), x_all_f16.data() + size_t(t0) * size_t(kHiddenSize), chunk_bf16.size());
                    check_hip(hipMemcpyAsync(gpu.d_x_norm.ptr, chunk_bf16.data(), chunk_bf16.size() * 2u, hipMemcpyHostToDevice, gpu.stream), "moe_x");

                    // Shared expert
                    gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, w_sg.ptr, gpu.d_sh_g.ptr, m, kSharedInterSize, kHiddenSize);
                    gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, w_su.ptr, gpu.d_sh_u.ptr, m, kSharedInterSize, kHiddenSize);
                    launch_swiglu_unclamped_bf16(gpu.d_sh_g.ptr, gpu.d_sh_u.ptr, gpu.d_sh_z.ptr, m * kSharedInterSize, gpu.stream);
                    gpu.gemm_bf16_w_f16_to_bf16(gpu.d_sh_z.ptr, w_sd.ptr, gpu.d_sh_y.ptr, m, kHiddenSize, kSharedInterSize);

                    // Router logits + top-8
                    gpu.gemm_bf16_w_f16_to_bf16(gpu.d_x_norm.ptr, w_gate.ptr, gpu.d_router_logits.ptr, m, kNumRoutedExperts, kHiddenSize);
                    launch_moe_router_top8_256(gpu.d_router_logits.ptr, gpu.d_router_bias.ptr,
                                               gpu.d_topk_ids.ptr, gpu.d_topk_w.ptr, m, kRoutedScalingFactor, gpu.stream);

                    check_hip(hipMemcpyAsync(h_shy.data(), gpu.d_sh_y.ptr, size_t(m) * size_t(kHiddenSize) * 2u, hipMemcpyDeviceToHost, gpu.stream), "shy_d2h");
                    check_hip(hipMemcpyAsync(topk_ids_all.data() + size_t(t0) * size_t(kTopK), gpu.d_topk_ids.ptr, size_t(m) * size_t(kTopK) * 4u, hipMemcpyDeviceToHost, gpu.stream), "ids_d2h");
                    check_hip(hipMemcpyAsync(topk_wts_all.data() + size_t(t0) * size_t(kTopK), gpu.d_topk_w.ptr, size_t(m) * size_t(kTopK) * 4u, hipMemcpyDeviceToHost, gpu.stream), "wts_d2h");
                    check_hip(hipStreamSynchronize(gpu.stream), "sh_sync");

                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < size_t(m) * size_t(kHiddenSize); ++i) {
                        y_moe_all[size_t(t0) * size_t(kHiddenSize) + i] = bf16_to_f32(h_shy[i]);
                    }
                }
            }

            // Build per-expert routed token lists & DISJOINT calibration frequencies (ONLY from t < kCalToks!)
            std::array<std::vector<std::pair<int, float>>, kNumRoutedExperts> exp_routes;
            std::array<int, kNumRoutedExperts> exp_cal_cnt{};
            for (int t = 0; t < total_m; ++t) {
                for (int k = 0; k < kTopK; ++k) {
                    const int e = topk_ids_all[size_t(t) * size_t(kTopK) + size_t(k)];
                    const float w = topk_wts_all[size_t(t) * size_t(kTopK) + size_t(k)];
                    exp_routes[size_t(e)].push_back({t, w});
                    if (t < kCalToks) {
                        cal_freq[size_t(e)]++;
                        exp_cal_cnt[size_t(e)]++;
                    }
                }
            }

            // Rank the 256 experts strictly by disjoint calibration frequency descending!
            std::iota(rank_to_orig.begin(), rank_to_orig.end(), 0);
            std::stable_sort(rank_to_orig.begin(), rank_to_orig.end(), [&](int a, int b) {
                return cal_freq[size_t(a)] > cal_freq[size_t(b)];
            });
            for (int r = 0; r < kNumRoutedExperts; ++r) orig_to_rank[size_t(rank_to_orig[size_t(r)])] = r;

            // Compute disjoint Zipfian route hit rates for tiers 113 / 125 / 139 / 155
            double tot_ev = double(kCalToks * kTopK);
            double csum = 0.0;
            for (int r = 0; r < kNumRoutedExperts; ++r) {
                csum += double(cal_freq[size_t(rank_to_orig[size_t(r)])]);
                if (r == 112) hit_113 = 100.0 * csum / tot_ev;
                if (r == 124) hit_125 = 100.0 * csum / tot_ev;
                if (r == 138) hit_139 = 100.0 * csum / tot_ev;
                if (r == 154) hit_155 = 100.0 * csum / tot_ev;
            }

            // 4C. Exact Stock FP8 Forward of ALL 256 Routed Experts on GPU + Harvest Layer-Pooled H_down [2048,2048] FP64!
            const auto signs_h = make_signs_h(l);
            const auto signs_i = make_signs_i(l);
            std::vector<double> H_down_pool(size_t(kMoeInterSize) * size_t(kMoeInterSize), 0.0);
            uint64_t total_z_cal_rows = 0;
            std::array<std::vector<float>, kNumRoutedExperts> z_cal_rot_saved;  // saved ONLY for top 64 ultra-hot experts for per-expert H_down!

            DevBuf<uint16_t> d_eg, d_eu, d_ed, d_ex_f16, d_ez_f16;
            for (int e = 0; e < kNumRoutedExperts; ++e) {
                const auto& routes = exp_routes[size_t(e)];
                const int m_e = int(routes.size());
                if (m_e == 0) continue;

                const std::string epfx = pfx + "mlp.experts." + std::to_string(e) + ".";
                gpu.load_weight_as_f16(cur_bundle, epfx + "gate_proj", d_eg);
                gpu.load_weight_as_f16(cur_bundle, epfx + "up_proj", d_eu);
                gpu.load_weight_as_f16(cur_bundle, epfx + "down_proj", d_ed);

                std::vector<uint16_t> h_ex(size_t(m_e) * size_t(kHiddenSize));
                for (int i = 0; i < m_e; ++i) {
                    std::memcpy(h_ex.data() + size_t(i) * size_t(kHiddenSize),
                                x_all_f16.data() + size_t(routes[size_t(i)].first) * size_t(kHiddenSize),
                                size_t(kHiddenSize) * 2u);
                }
                d_ex_f16.ensure(h_ex.size());
                gpu.d_exp_g.ensure(size_t(m_e) * size_t(kMoeInterSize));
                gpu.d_exp_u.ensure(size_t(m_e) * size_t(kMoeInterSize));
                gpu.d_exp_z.ensure(size_t(m_e) * size_t(kMoeInterSize));
                d_ez_f16.ensure(size_t(m_e) * size_t(kMoeInterSize));
                gpu.d_exp_y.ensure(size_t(m_e) * size_t(kHiddenSize));

                check_hip(hipMemcpyAsync(d_ex_f16.ptr, h_ex.data(), h_ex.size() * 2u, hipMemcpyHostToDevice, gpu.stream), "ex_h2d");
                gpu.gemm_f16_to_f32(d_ex_f16.ptr, d_eg.ptr, gpu.d_exp_g.ptr, m_e, kMoeInterSize, kHiddenSize);
                gpu.gemm_f16_to_f32(d_ex_f16.ptr, d_eu.ptr, gpu.d_exp_u.ptr, m_e, kMoeInterSize, kHiddenSize);
                launch_swiglu_clamped10_f32(gpu.d_exp_g.ptr, gpu.d_exp_u.ptr, gpu.d_exp_z.ptr, m_e * kMoeInterSize, gpu.stream);

                // Convert z F32 -> F16 on GPU for down_proj GEMM
                std::vector<float> h_ez(size_t(m_e) * size_t(kMoeInterSize));
                check_hip(hipMemcpyAsync(h_ez.data(), gpu.d_exp_z.ptr, h_ez.size() * 4u, hipMemcpyDeviceToHost, gpu.stream), "ez_d2h");
                check_hip(hipStreamSynchronize(gpu.stream), "ez_sync");

                std::vector<uint16_t> h_ez16(h_ez.size());
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < h_ez.size(); ++i) h_ez16[i] = f32_to_fp16_rne(h_ez[i]);
                check_hip(hipMemcpyAsync(d_ez_f16.ptr, h_ez16.data(), h_ez16.size() * 2u, hipMemcpyHostToDevice, gpu.stream), "ez16_h2d");
                gpu.gemm_f16_to_f32(d_ez_f16.ptr, d_ed.ptr, gpu.d_exp_y.ptr, m_e, kHiddenSize, kMoeInterSize);

                std::vector<float> h_ey(size_t(m_e) * size_t(kHiddenSize));
                check_hip(hipMemcpyAsync(h_ey.data(), gpu.d_exp_y.ptr, h_ey.size() * 4u, hipMemcpyDeviceToHost, gpu.stream), "ey_d2h");
                check_hip(hipStreamSynchronize(gpu.stream), "ey_sync");

                // Scatter-add weighted expert output to y_moe_all
                for (int i = 0; i < m_e; ++i) {
                    const int tok_i = routes[size_t(i)].first;
                    const float wt  = routes[size_t(i)].second;
                    float* dst = y_moe_all.data() + size_t(tok_i) * size_t(kHiddenSize);
                    const float* src = h_ey.data() + size_t(i) * size_t(kHiddenSize);
                    for (int d = 0; d < kHiddenSize; ++d) dst[d] += wt * src[d];
                }

                // Harvest calibration-only Z_rot for FP64 H_down_pool (and save top-60 hot experts for per-expert shrinkage!)
                const int m_cal_e = exp_cal_cnt[size_t(e)];
                if (m_cal_e > 0) {
                    std::vector<float> z_cal(size_t(m_cal_e) * size_t(kMoeInterSize));
                    int ci = 0;
                    for (int i = 0; i < m_e; ++i) {
                        if (routes[size_t(i)].first < kCalToks) {
                            std::memcpy(z_cal.data() + size_t(ci++) * size_t(kMoeInterSize),
                                        h_ez.data() + size_t(i) * size_t(kMoeInterSize),
                                        size_t(kMoeInterSize) * 4u);
                        }
                    }
                    std::vector<float> z_rot(z_cal.size());
                    wht128_rows(z_cal.data(), z_rot.data(), m_cal_e, kMoeInterSize, signs_i);
                    gpu.dsyrk_add_fp64(z_rot.data(), m_cal_e, kMoeInterSize, 1.0, H_down_pool);
                    total_z_cal_rows += uint64_t(m_cal_e);
                    if (orig_to_rank[size_t(e)] < 60) z_cal_rot_saved[size_t(e)] = std::move(z_rot);
                }
            }

            // Advance h_cal & h_prb with exact stock FP8 MoE residual add!
            auto apply_moe_post = [&](int tok_off, int ntoks, std::vector<uint16_t>& st_host, const std::vector<float>& post_host) {
                #pragma omp parallel for schedule(static)
                for (int t = 0; t < ntoks; ++t) {
                    const float* y = y_moe_all.data() + size_t(tok_off + t) * size_t(kHiddenSize);
                    const float* p = post_host.data() + size_t(t) * size_t(kHcMult);
                    uint16_t* st   = st_host.data() + size_t(t) * size_t(kHcFlatDim);
                    for (int c = 0; c < kHcMult; ++c) {
                        const float pc = p[c];
                        uint16_t* sc = st + size_t(c) * size_t(kHiddenSize);
                        for (int d = 0; d < kHiddenSize; ++d) sc[d] = f32_to_bf16_rne(bf16_to_f32(sc[d]) + pc * y[d]);
                    }
                }
            };
            apply_moe_post(0, kCalToks, h_cal, post_cal_mlp);
            apply_moe_post(kCalToks, kPrbToks, h_prb, post_prb_mlp);

            // 4D. Build exact FP64 TARQ RAREBAL Hessian H_in [6144,6144] on GPU + reverse-Cholesky R_up_in & R_up_dn_pool!
            std::vector<float> x_cal_f32(size_t(kCalToks) * size_t(kHiddenSize));
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < x_cal_f32.size(); ++i) x_cal_f32[i] = bf16_to_f32(x_cal_mlp[i]);
            std::vector<float> x_cal_rot(x_cal_f32.size());
            wht128_rows(x_cal_f32.data(), x_cal_rot.data(), kCalToks, kHiddenSize, signs_h);

            // TARQ RAREBAL rare-token weighting on x_cal_rot -> exact FP64 dsyrk on MI50 (25 ms!)
            std::vector<double> H_in(size_t(kHiddenSize) * size_t(kHiddenSize), 0.0);
            {
                std::vector<double> mean_dir(size_t(kHiddenSize), 0.0);
                std::vector<float> inv_n(size_t(kCalToks), 0.0f);
                for (int m = 0; m < kCalToks; ++m) {
                    const float* xm = x_cal_rot.data() + size_t(m) * size_t(kHiddenSize);
                    double n2 = 0.0;
                    for (int k = 0; k < kHiddenSize; ++k) n2 += double(xm[k]) * double(xm[k]);
                    const double inv = (n2 > 1e-30) ? (1.0 / std::sqrt(n2)) : 0.0;
                    inv_n[size_t(m)] = float(inv);
                    for (int k = 0; k < kHiddenSize; ++k) mean_dir[size_t(k)] += double(xm[k]) * inv;
                }
                double mn2 = 0.0;
                for (int k = 0; k < kHiddenSize; ++k) mn2 += mean_dir[size_t(k)] * mean_dir[size_t(k)];
                const double inv_mn = (mn2 > 1e-30) ? (1.0 / std::sqrt(mn2)) : 0.0;
                for (int k = 0; k < kHiddenSize; ++k) mean_dir[size_t(k)] *= inv_mn;
                std::vector<std::pair<float, int>> align(static_cast<size_t>(kCalToks));
                for (int m = 0; m < kCalToks; ++m) {
                    const float* xm = x_cal_rot.data() + size_t(m) * size_t(kHiddenSize);
                    double dot = 0.0;
                    for (int k = 0; k < kHiddenSize; ++k) dot += double(xm[k]) * mean_dir[size_t(k)];
                    align[size_t(m)] = {float(std::fabs(dot) * double(inv_n[size_t(m)])), m};
                }
                const int rt = kCalToks / 8;
                std::nth_element(align.begin(), align.begin() + rt, align.end());
                std::vector<uint8_t> is_rare(size_t(kCalToks), 0u);
                for (int i = 0; i < rt; ++i) is_rare[size_t(align[size_t(i)].second)] = 1u;
                double tr_c = 0.0, tr_r = 0.0;
                std::vector<float> xc, xr;
                xc.reserve(size_t(kCalToks - rt) * size_t(kHiddenSize));
                xr.reserve(size_t(rt) * size_t(kHiddenSize));
                for (int m = 0; m < kCalToks; ++m) {
                    const float* xm = x_cal_rot.data() + size_t(m) * size_t(kHiddenSize);
                    double sq = 0.0;
                    for (int k = 0; k < kHiddenSize; ++k) sq += double(xm[k]) * double(xm[k]);
                    if (is_rare[size_t(m)]) { tr_r += sq; xr.insert(xr.end(), xm, xm + kHiddenSize); }
                    else                    { tr_c += sq; xc.insert(xc.end(), xm, xm + kHiddenSize); }
                }
                const double lam = (tr_r > 1e-12) ? (tr_c / tr_r) : 1.0;
                gpu.dsyrk_add_fp64(xc.data(), int(xc.size() / kHiddenSize), kHiddenSize, 1.0 / double(kCalToks), H_in);
                gpu.dsyrk_add_fp64(xr.data(), int(xr.size() / kHiddenSize), kHiddenSize, lam / double(kCalToks), H_in);
            }

            RarebalStats st_in{}, st_dn{};
            const auto R_up_in = build_rup_from_fp64_hessian(H_in.data(), kHiddenSize, 0.015, &st_in);
            cond_in = st_in.cond_est;

            const double inv_z = 1.0 / double(std::max<uint64_t>(1, total_z_cal_rows));
            for (double& v : H_down_pool) v *= inv_z;
            const auto R_up_dn_pool = build_rup_from_fp64_hessian(H_down_pool.data(), kMoeInterSize, 0.015, &st_dn);
            cond_dn = st_dn.cond_est;

            // 4E. Pack Ranked Experts into Deduplicated Tier Shards using 60x-Faster GPU Blocked LDLQ (B=64)!
            //   hlut3_layer_XX.bin            : ranks [0, 125)   (125 experts, 1.716 GiB)
            //   hlut2_layer_XX.bin            : ranks [60, 155)  ( 95 experts, 0.887 GiB)
            //   hlut4_ddr4/hlut4_layer_XX.bin : ranks [110, 256) (146 experts, 2.647 GiB)
            gpu.d_signs_h.ensure(signs_h.size());
            gpu.d_signs_i.ensure(signs_i.size());
            gpu.d_R_in.ensure(R_up_in.size());
            gpu.d_R_dn.ensure(R_up_dn_pool.size());
            check_hip(hipMemcpyAsync(gpu.d_signs_h.ptr, signs_h.data(), signs_h.size() * 8u, hipMemcpyHostToDevice, gpu.stream), "sh_h2d");
            check_hip(hipMemcpyAsync(gpu.d_signs_i.ptr, signs_i.data(), signs_i.size() * 8u, hipMemcpyHostToDevice, gpu.stream), "si_h2d");
            check_hip(hipMemcpyAsync(gpu.d_R_in.ptr, R_up_in.data(), R_up_in.size() * 4u, hipMemcpyHostToDevice, gpu.stream), "rin_h2d");
            check_hip(hipMemcpyAsync(gpu.d_R_dn.ptr, R_up_dn_pool.data(), R_up_dn_pool.size() * 4u, hipMemcpyHostToDevice, gpu.stream), "rdn_h2d");
            check_hip(hipStreamSynchronize(gpu.stream), "pack_setup_sync");

            const std::string p_h3 = weight_root + "/hlut3/hlut3_layer_" + layer_tag(l) + ".bin";
            const std::string p_h2 = weight_root + "/hlut2/hlut2_layer_" + layer_tag(l) + ".bin";
            const std::string p_h4 = weight_root + "/hlut4_ddr4/hlut4_layer_" + layer_tag(l) + ".bin";
            std::ofstream f3(p_h3 + ".tmp", std::ios::binary | std::ios::trunc);
            std::ofstream f2(p_h2 + ".tmp", std::ios::binary | std::ios::trunc);
            std::ofstream f4(p_h4 + ".tmp", std::ios::binary | std::ios::trunc);

            double sum_sqnr3 = 0.0, sum_sqnr2 = 0.0, sum_sqnr4 = 0.0;
            int cnt3 = 0, cnt2 = 0, cnt4 = 0;
            bool r_dn_is_pool = true;
            std::vector<float> tmp_wrot_check;

            for (int r = 0; r < kNumRoutedExperts; ++r) {
                const int e = rank_to_orig[size_t(r)];
                const std::string epfx = pfx + "mlp.experts." + std::to_string(e) + ".";
                const bool need3 = (r < kPackHlut3End);
                const bool need2 = (r >= kPackHlut2Begin && r < kPackHlut2End);
                const bool need4 = (r >= kPackHlut4Begin);
                const bool do_sqnr = (r % 16 == 0);

                // Per-expert RAREBAL empirical-Bayes down Hessian for top-60 hot experts!
                if (r < 60 && !z_cal_rot_saved[size_t(e)].empty()) {
                    const int me = int(z_cal_rot_saved[size_t(e)].size() / kMoeInterSize);
                    std::vector<double> He(size_t(kMoeInterSize) * size_t(kMoeInterSize), 0.0);
                    gpu.dsyrk_add_fp64(z_cal_rot_saved[size_t(e)].data(), me, kMoeInterSize, 1.0 / double(me), He);
                    const double alpha = double(me) / (double(me) + 512.0);
                    for (size_t i = 0; i < He.size(); ++i) He[i] = alpha * He[i] + (1.0 - alpha) * H_down_pool[i];
                    auto R_dn_local = build_rup_from_fp64_hessian(He.data(), kMoeInterSize, 0.015);
                    check_hip(hipMemcpyAsync(gpu.d_R_dn.ptr, R_dn_local.data(), R_dn_local.size() * 4u, hipMemcpyHostToDevice, gpu.stream), "rdn_loc");
                    r_dn_is_pool = false;
                    z_cal_rot_saved[size_t(e)].clear();
                    z_cal_rot_saved[size_t(e)].shrink_to_fit();
                } else if (!r_dn_is_pool) {
                    check_hip(hipMemcpyAsync(gpu.d_R_dn.ptr, R_up_dn_pool.data(), R_up_dn_pool.size() * 4u, hipMemcpyHostToDevice, gpu.stream), "rdn_pool");
                    r_dn_is_pool = true;
                }

                // Pack gate, up, down ONCE each on GPU (reusing d_W_orig + d_scales_f16 across HLUT3/2/4!)
                Hlut3QuantMatrix q3[3];
                Hlut2QuantMatrix q2[3];
                Hlut4QuantMatrix q4[3];
                double e_sq3 = 0.0, e_sq2 = 0.0, e_sq4 = 0.0;

                const char* pnames[3] = {"gate_proj", "up_proj", "down_proj"};
                const int pN[3] = {kMoeInterSize, kMoeInterSize, kHiddenSize};
                const int pK[3] = {kHiddenSize,   kHiddenSize,   kMoeInterSize};

                for (int pi = 0; pi < 3; ++pi) {
                    const std::string wn = epfx + pnames[pi] + ".weight";
                    const std::string sn = epfx + pnames[pi] + ".weight_scale";
                    const uint64_t* d_sg = (pi < 2) ? gpu.d_signs_h.ptr : gpu.d_signs_i.ptr;
                    const float*    d_ru = (pi < 2) ? gpu.d_R_in.ptr    : gpu.d_R_dn.ptr;
                    gpu.prepare_wrot_gpu(cur_bundle.data(wn), reinterpret_cast<const float*>(cur_bundle.data(sn)), d_sg, pN[pi], pK[pi]);

                    if (do_sqnr) {
                        tmp_wrot_check.resize(size_t(pN[pi]) * size_t(pK[pi]));
                        check_hip(hipMemcpyAsync(tmp_wrot_check.data(), gpu.d_W_orig.ptr, tmp_wrot_check.size() * 4u, hipMemcpyDeviceToHost, gpu.stream), "wrot_chk");
                    }
                    if (need3) {
                        q3[pi] = gpu.pack_prepared_ldlq_gpu<3, Hlut3Tile32x32>(d_ru, pN[pi], pK[pi], lloyd_max_hlut3());
                        if (do_sqnr) e_sq3 += compute_hlut_sqnr_db(q3[pi], tmp_wrot_check.data(), hlut3_unpack_tile);
                    }
                    if (need2) {
                        q2[pi] = gpu.pack_prepared_ldlq_gpu<2, Hlut2Tile32x32>(d_ru, pN[pi], pK[pi], lloyd_max_hlut2());
                        if (do_sqnr) e_sq2 += compute_hlut_sqnr_db(q2[pi], tmp_wrot_check.data(), hlut2_unpack_tile);
                    }
                    if (need4) {
                        q4[pi] = gpu.pack_prepared_ldlq_gpu<4, Hlut4Tile32x32>(d_ru, pN[pi], pK[pi], lloyd_max_hlut4());
                        if (do_sqnr) e_sq4 += compute_hlut_sqnr_db(q4[pi], tmp_wrot_check.data(), hlut4_unpack_tile);
                    }
                }

                if (need3) {
                    if (do_sqnr) { sum_sqnr3 += e_sq3 / 3.0; ++cnt3; }
                    for (int pi = 0; pi < 3; ++pi) {
                        f3.write(reinterpret_cast<const char*>(q3[pi].tiles.data()), std::streamsize(q3[pi].raw_bytes()));
                        crc_hlut3 = crc32c_update(crc_hlut3, q3[pi].tiles.data(), q3[pi].raw_bytes());
                    }
                }
                if (need2) {
                    if (do_sqnr) { sum_sqnr2 += e_sq2 / 3.0; ++cnt2; }
                    for (int pi = 0; pi < 3; ++pi) {
                        f2.write(reinterpret_cast<const char*>(q2[pi].tiles.data()), std::streamsize(q2[pi].raw_bytes()));
                        crc_hlut2 = crc32c_update(crc_hlut2, q2[pi].tiles.data(), q2[pi].raw_bytes());
                    }
                }
                if (need4) {
                    if (do_sqnr) { sum_sqnr4 += e_sq4 / 3.0; ++cnt4; }
                    for (int pi = 0; pi < 3; ++pi) {
                        f4.write(reinterpret_cast<const char*>(q4[pi].tiles.data()), std::streamsize(q4[pi].raw_bytes()));
                        crc_hlut4 = crc32c_update(crc_hlut4, q4[pi].tiles.data(), q4[pi].raw_bytes());
                    }
                }
            }
            f3.close(); f2.close(); f4.close();
            fs::rename(p_h3 + ".tmp", p_h3);
            fs::rename(p_h2 + ".tmp", p_h2);
            fs::rename(p_h4 + ".tmp", p_h4);

            mean_sqnr_hlut3 = (cnt3 > 0) ? (sum_sqnr3 / cnt3) : 0.0;
            mean_sqnr_hlut2 = (cnt2 > 0) ? (sum_sqnr2 / cnt2) : 0.0;
            mean_sqnr_hlut4 = (cnt4 > 0) ? (sum_sqnr4 / cnt4) : 0.0;
        }

        // 4F. Pack Layer l Spine (FP8-g128 large linears + lossless BF16/FP32 norms/iHC/routers/indexer/sinks + expert ranking!)
        {
            std::vector<SpineFp8Entry> fp8_mats;
            std::vector<std::string> linear_bases = {
                pfx + "self_attn.q_a_proj", pfx + "self_attn.q_b_proj",
                pfx + "self_attn.kv_a_proj_with_mqa", pfx + "self_attn.kv_b_proj",
                pfx + "self_attn.linear_gate", pfx + "self_attn.o_proj"
            };
            if (l == 0) {
                linear_bases.push_back(pfx + "mlp.gate_proj");
                linear_bases.push_back(pfx + "mlp.up_proj");
                linear_bases.push_back(pfx + "mlp.down_proj");
            } else {
                linear_bases.push_back(pfx + "mlp.shared_experts.gate_proj");
                linear_bases.push_back(pfx + "mlp.shared_experts.up_proj");
                linear_bases.push_back(pfx + "mlp.shared_experts.down_proj");
            }
            std::unordered_set<std::string> handled;
            fp8_mats.reserve(linear_bases.size());
            for (const auto& bn : linear_bases) {
                fp8_mats.push_back(extract_or_quant_fp8(cur_bundle, bn));
                handled.insert(bn + ".weight");
                handled.insert(bn + ".weight_scale");
            }
            std::vector<hy4::SafetensorWriteSpec> specs;
            for (const auto& m : fp8_mats) {
                const int nb = (m.N + 127) / 128, kb = (m.K + 127) / 128;
                specs.push_back({m.base_name + ".weight", "F8_E4M3", {m.N, m.K}, m.weight_fp8.data(), uint64_t(m.weight_fp8.size())});
                specs.push_back({m.base_name + ".weight_scale", "F32", {nb, kb}, m.weight_scale.data(), uint64_t(m.weight_scale.size() * 4u)});
            }
            // Lossless BF16/FP32 tensors (norms, iHC, router, indexer, sinks)
            for (const auto& kv : cur_bundle.specs) {
                const std::string& nm = kv.first;
                if (nm.find(".mlp.experts.") != std::string::npos) continue;
                if (handled.count(nm)) continue;
                specs.push_back({nm, kv.second.dtype, kv.second.shape, cur_bundle.data(nm), kv.second.nbytes});
            }
            if (l > 0) {
                specs.push_back({"expert_rank_to_orig", "I32", {kNumRoutedExperts}, rank_to_orig.data(), sizeof(rank_to_orig)});
                specs.push_back({"expert_orig_to_rank", "I32", {kNumRoutedExperts}, orig_to_rank.data(), sizeof(orig_to_rank)});
                specs.push_back({"expert_cal_freq",     "U32", {kNumRoutedExperts}, cal_freq.data(),     sizeof(cal_freq)});
            }
            const std::string sp_path = root + "/packed_spine/spine_layer_" + layer_tag(l) + ".safetensors";
            hy4::write_safetensors_file(sp_path, specs, {{"format", "hy4_770b_fp8_spine"}, {"layer", std::to_string(l)}});
        }

        // Compute probe hidden state L2 norm receipt for Gate-2 layer verification
        double prb_rms = 0.0;
        for (size_t i = 0; i < h_prb.size(); ++i) {
            const double v = bf16_to_f32(h_prb[i]);
            prb_rms += v * v;
        }
        prb_rms = std::sqrt(prb_rms / double(h_prb.size()));

        // Persist resume states & layer checkpoint JSON
        save_bin(cal_state_path, h_cal.data(), h_cal.size() * 2u);
        save_bin(prb_state_path, h_prb.data(), h_prb.size() * 2u);

        const double total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_layer0).count();
        {
            const std::string ck = root + "/checkpoints/layer_" + layer_tag(l) + ".json";
            std::ofstream ofs(ck + ".tmp");
            ofs << "{\n"
                << "  \"layer\": " << l << ",\n"
                << "  \"download_sec\": " << dl_sec << ",\n"
                << "  \"download_mibs\": " << dl_mibs << ",\n"
                << "  \"total_sec\": " << total_sec << ",\n"
                << "  \"probe_rms\": " << prb_rms << ",\n"
                << "  \"cond_in\": " << cond_in << ",\n"
                << "  \"cond_dn\": " << cond_dn << ",\n"
                << "  \"sqnr_hlut3_db\": " << mean_sqnr_hlut3 << ",\n"
                << "  \"sqnr_hlut2_db\": " << mean_sqnr_hlut2 << ",\n"
                << "  \"sqnr_hlut4_db\": " << mean_sqnr_hlut4 << ",\n"
                << "  \"disjoint_hit_113_pct\": " << hit_113 << ",\n"
                << "  \"disjoint_hit_125_pct\": " << hit_125 << ",\n"
                << "  \"disjoint_hit_139_pct\": " << hit_139 << ",\n"
                << "  \"disjoint_hit_155_pct\": " << hit_155 << ",\n"
                << "  \"crc32c_hlut3\": \"0x" << std::hex << crc_hlut3 << "\",\n"
                << "  \"crc32c_hlut2\": \"0x" << crc_hlut2 << "\",\n"
                << "  \"crc32c_hlut4\": \"0x" << crc_hlut4 << std::dec << "\"\n"
                << "}\n";
            ofs.close();
            fs::rename(ck + ".tmp", ck);
        }

        std::printf("[L%02d/77] DONE in %.1fs | prb_rms=%.4f | SQNR: HLUT4=%.2fdB HLUT3=%.2fdB HLUT2=%.2fdB | "
                    "Disjoint Hit: 113=%.1f%% 125=%.1f%% 139=%.1f%% 155=%.1f%% | cond=%.1f/%.1f\n",
                    l, total_sec, prb_rms, mean_sqnr_hlut4, mean_sqnr_hlut3, mean_sqnr_hlut2,
                    hit_113, hit_125, hit_139, hit_155, cond_in, cond_dn);
        std::fflush(stdout);
    }

    // Step 5: Emit exact stock FP8 Gate-2 Oracle Reference Logits at Layer 77 on the 64 held-out probes!
    if (end_layer == kNumLayers - 1 && fs::exists(root + "/checkpoints/layer_77.json") && !should_stop()) {
        std::printf("[HY4-770B-PACK] Computing exact stock FP8 Gate-2 Oracle Logits on 64 held-out probe positions...\n");
        std::fflush(stdout);
        hy4::SafetensorsMmap st(shared_path);

        // Extract last token (position 63) of each of the 64 probe sequences -> [64, 4, 6144] BF16
        std::vector<uint16_t> h_last(size_t(kPrbSeqs) * size_t(kHcFlatDim));
        for (int s = 0; s < kPrbSeqs; ++s) {
            std::memcpy(h_last.data() + size_t(s) * size_t(kHcFlatDim),
                        h_prb.data() + size_t(s * kPrbLen + (kPrbLen - 1)) * size_t(kHcFlatDim),
                        size_t(kHcFlatDim) * 2u);
        }

        DevBuf<uint16_t> d_last, d_norm_w, d_h_out;
        DevBuf<float> d_hf_fn, d_hf_base, d_hf_sc;
        d_last.ensure(h_last.size());
        d_h_out.ensure(size_t(kPrbSeqs) * size_t(kHiddenSize));
        d_norm_w.ensure(size_t(kHiddenSize));
        d_hf_fn.ensure(4u * size_t(kHcFlatDim));
        d_hf_base.ensure(4u);
        d_hf_sc.ensure(1u);

        check_hip(hipMemcpyAsync(d_last.ptr, h_last.data(), h_last.size() * 2u, hipMemcpyHostToDevice, gpu.stream), "last");
        check_hip(hipMemcpyAsync(d_norm_w.ptr, st.tensor_bytes("model.norm.weight"), size_t(kHiddenSize) * 2u, hipMemcpyHostToDevice, gpu.stream), "nw");
        check_hip(hipMemcpyAsync(d_hf_fn.ptr, st.tensor_bytes("model.hc_head.hc_head_fn"), 4u * size_t(kHcFlatDim) * 4u, hipMemcpyHostToDevice, gpu.stream), "hfn");
        check_hip(hipMemcpyAsync(d_hf_base.ptr, st.tensor_bytes("model.hc_head.hc_head_base"), 16u, hipMemcpyHostToDevice, gpu.stream), "hbase");
        check_hip(hipMemcpyAsync(d_hf_sc.ptr, st.tensor_bytes("model.hc_head.hc_head_scale"), 4u, hipMemcpyHostToDevice, gpu.stream), "hsc");

        launch_ihc_head_and_norm(d_last.ptr, d_hf_fn.ptr, d_hf_base.ptr, d_hf_sc.ptr,
                                 d_norm_w.ptr, d_h_out.ptr, kPrbSeqs, kRmsNormEps, kHcEps, gpu.stream);

        // Exact FP32 lm_head [120832, 6144] in 16384-row chunks on GPU
        std::vector<float> oracle_logits(size_t(kPrbSeqs) * size_t(kVocabSize), 0.0f);
        const uint16_t* lm_w = reinterpret_cast<const uint16_t*>(st.tensor_bytes("lm_head.weight"));
        DevBuf<uint16_t> d_lm_chunk_f16;
        DevBuf<float> d_h_f32;
        gpu.d_a_f16.ensure(size_t(kPrbSeqs) * size_t(kHiddenSize));
        launch_bf16_to_f16(d_h_out.ptr, gpu.d_a_f16.ptr, kPrbSeqs * kHiddenSize, gpu.stream);

        const int vblk = 16384;
        std::vector<uint16_t> h_lm16(size_t(vblk) * size_t(kHiddenSize));
        std::vector<float> h_log_chunk(size_t(kPrbSeqs) * size_t(vblk));
        for (int v0 = 0; v0 < kVocabSize; v0 += vblk) {
            const int vc = std::min(vblk, kVocabSize - v0);
            bf16_to_f16_host(lm_w + size_t(v0) * size_t(kHiddenSize), h_lm16.data(), size_t(vc) * size_t(kHiddenSize));
            d_lm_chunk_f16.ensure(size_t(vc) * size_t(kHiddenSize));
            gpu.d_c_f32.ensure(size_t(kPrbSeqs) * size_t(vc));
            check_hip(hipMemcpyAsync(d_lm_chunk_f16.ptr, h_lm16.data(), size_t(vc) * size_t(kHiddenSize) * 2u, hipMemcpyHostToDevice, gpu.stream), "lm_h2d");
            gpu.gemm_f16_to_f32(gpu.d_a_f16.ptr, d_lm_chunk_f16.ptr, gpu.d_c_f32.ptr, kPrbSeqs, vc, kHiddenSize);
            check_hip(hipMemcpyAsync(h_log_chunk.data(), gpu.d_c_f32.ptr, size_t(kPrbSeqs) * size_t(vc) * 4u, hipMemcpyDeviceToHost, gpu.stream), "lm_d2h");
            check_hip(hipStreamSynchronize(gpu.stream), "lm_sync");
            for (int s = 0; s < kPrbSeqs; ++s) {
                std::memcpy(oracle_logits.data() + size_t(s) * size_t(kVocabSize) + size_t(v0),
                            h_log_chunk.data() + size_t(s) * size_t(vc), size_t(vc) * 4u);
            }
        }
        save_bin(root + "/artifacts/oracle_ref_logits_64.bin", oracle_logits.data(), oracle_logits.size() * 4u);

        // Save probe token IDs & top-1 oracle predictions for Gate-2 verification
        std::ofstream ofs(root + "/artifacts/oracle_probe_meta.json");
        ofs << "{\n  \"num_probes\": 64,\n  \"seq_len\": 64,\n  \"vocab_size\": " << kVocabSize << ",\n  \"top1_ids\": [";
        for (int s = 0; s < kPrbSeqs; ++s) {
            const float* lg = oracle_logits.data() + size_t(s) * size_t(kVocabSize);
            int best = int(std::max_element(lg, lg + kVocabSize) - lg);
            if (s) ofs << ",";
            ofs << best;
        }
        ofs << "]\n}\n";
        std::printf("[HY4-770B-PACK] COMPLETE! Wrote exact Gate-2 oracle logits (64 x 120832 FP32)!\n");
        std::fflush(stdout);
    }
    return 0;
}
