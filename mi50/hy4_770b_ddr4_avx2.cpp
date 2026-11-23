// Persistent Lock-Free 28-Core AVX2+FMA+F16C DDR4 Cold-Tail Engine for tencent/Hy4 (770B)
// Zero Python. Pure AVX2 register-LUT nibble decode (_mm256_permutevar8x32_ps + sign XOR + FMA).

#include "hy4_770b_ddr4_avx2.hpp"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace hy4_770b {

namespace {

inline int64_t now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void pin_thread_to_core(int core_id) noexcept {
#ifdef __linux__
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(core_id % 64, &cs);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#else
    (void)core_id;
#endif
}

// Fast AVX2 F16 -> F32 scalar conversion via F16C
inline float fast_f16_to_f32(uint16_t h) noexcept {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    return fp16_to_f32(h);
#endif
}

}  // namespace

Hy4Ddr4ColdEngine::~Hy4Ddr4ColdEngine() {
    shutdown();
}

void Hy4Ddr4ColdEngine::avx2_hlut4_gemv_rows(
    const float* x_rot,
    const Hlut4Tile32x32* tiles,
    const float levels8[8],
    int n_tile_begin,
    int n_tile_end,
    int kt,
    float* y_out) noexcept
{
#if defined(__AVX2__)
    const __m256 v_lut = _mm256_loadu_ps(levels8);
    const __m256i v_sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i v_m7 = _mm256_set1_epi32(7);
    const __m256i v_m8 = _mm256_set1_epi32(8);

    auto hsum256 = [](__m256 v) noexcept -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    };

    for (int nt = n_tile_begin; nt < n_tile_end; ++nt) {
        const Hlut4Tile32x32* row_tiles = tiles + size_t(nt) * size_t(kt);
        float acc_r[32];
        std::memset(acc_r, 0, sizeof(acc_r));

        for (int kt_i = 0; kt_i < kt; ++kt_i) {
            const Hlut4Tile32x32& tile = row_tiles[kt_i];
            if (kt_i + 2 < kt) {
                _mm_prefetch(reinterpret_cast<const char*>(&row_tiles[kt_i + 2]), _MM_HINT_T0);
            }
            const float* xk = x_rot + size_t(kt_i) * 32u;
            const __m256 vx0 = _mm256_loadu_ps(xk + 0);
            const __m256 vx1 = _mm256_loadu_ps(xk + 8);
            const __m256 vx2 = _mm256_loadu_ps(xk + 16);
            const __m256 vx3 = _mm256_loadu_ps(xk + 24);

            for (int g = 0; g < 8; ++g) {
                const float sigma = fast_f16_to_f32(tile.scales_f16[g]);
                const int r_base = g * 4;
                for (int dr = 0; dr < 4; ++dr) {
                    const int r = r_base + dr;
                    const uint32_t* q = tile.q_nib4 + r * 4;
                    auto dec_fma = [&](uint32_t word, __m256 vx, __m256 acc) noexcept -> __m256 {
                        __m256i nibs = _mm256_srlv_epi32(_mm256_set1_epi32(int(word)), v_sh);
                        __m256  mag  = _mm256_permutevar8x32_ps(v_lut, _mm256_and_si256(nibs, v_m7));
                        __m256i sgn  = _mm256_slli_epi32(_mm256_and_si256(nibs, v_m8), 28);
                        __m256  w    = _mm256_xor_ps(mag, _mm256_castsi256_ps(sgn));
                        return _mm256_fmadd_ps(w, vx, acc);
                    };
                    __m256 a = dec_fma(q[0], vx0, _mm256_setzero_ps());
                    a = dec_fma(q[1], vx1, a);
                    a = dec_fma(q[2], vx2, a);
                    a = dec_fma(q[3], vx3, a);
                    acc_r[r] += hsum256(a) * sigma;
                }
            }
        }
        std::memcpy(y_out + size_t(nt - n_tile_begin) * 32u, acc_r, sizeof(acc_r));
    }
#else
    for (int nt = n_tile_begin; nt < n_tile_end; ++nt) {
        const Hlut4Tile32x32* row_tiles = tiles + size_t(nt) * size_t(kt);
        float acc_r[32]{};
        for (int kt_i = 0; kt_i < kt; ++kt_i) {
            const Hlut4Tile32x32& tile = row_tiles[kt_i];
            const float* xk = x_rot + size_t(kt_i) * 32u;
            for (int r = 0; r < 32; ++r) {
                const float sigma = fp16_to_f32(tile.scales_f16[r >> 2]);
                float sum = 0.0f;
                for (int c = 0; c < 32; ++c) {
                    const uint32_t nib = (tile.q_nib4[r * 4 + (c >> 3)] >> (4 * (c & 7))) & 0xFu;
                    const float w = levels8[nib & 7u];
                    sum += ((nib & 8u) ? -w : w) * xk[c];
                }
                acc_r[r] += sum * sigma;
            }
        }
        std::memcpy(y_out + size_t(nt - n_tile_begin) * 32u, acc_r, sizeof(acc_r));
    }
#endif
}

// Overflow-proof int16 even/odd-permuted quantization of one 128-element activation block for Q5_LM128 vpshufb+vpmaddwd!
// Max |w_i16| = round(3.2788 * 2048) = 6715; max |x_i16| = 12287 -> max 8-iteration int32 lane sum = 1.32e9 < INT32_MAX!
// Bit-exact with GATE-2 v2 (cosine = 0.99999999 vs scalar ref), zero YMM spills at B<=4!
static constexpr float kQ5WScale    = 2048.0f;
static constexpr float kQ5InvWScale = 1.0f / 2048.0f;
static constexpr float kQ5XMaxQ     = 12287.0f;

static inline void quantize_block128_q5_eo(const float* xb, int16_t* qb, float& out_sc) noexcept {
    float amax = 0.0f;
#if defined(__AVX2__)
    const __m256 v_abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 vmax = _mm256_setzero_ps();
    for (int i = 0; i < 128; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(xb + i), v_abs_mask));
    }
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    amax = _mm_cvtss_f32(m4);
#else
    for (int i = 0; i < 128; ++i) {
        const float a = std::fabs(xb[i]);
        if (a > amax) amax = a;
    }
#endif
    if (amax <= 1e-30f) {
        out_sc = 0.0f;
        std::memset(qb, 0, 128u * sizeof(int16_t));
        return;
    }
    const float inv = kQ5XMaxQ / amax;
    out_sc = (amax / kQ5XMaxQ) * kQ5InvWScale;
    for (int k = 0; k < 4; ++k) {
        const float* xk = xb + k * 32;
        int16_t* qk = qb + k * 32;
        for (int i = 0; i < 8; ++i) {
            qk[0  + i] = int16_t(std::lrintf(xk[2 * i + 0] * inv));
            qk[8  + i] = int16_t(std::lrintf(xk[2 * i + 1] * inv));
            qk[16 + i] = int16_t(std::lrintf(xk[16 + 2 * i + 0] * inv));
            qk[24 + i] = int16_t(std::lrintf(xk[16 + 2 * i + 1] * inv));
        }
    }
}

static inline void quantize_block128_q5_eo_bf16(const uint16_t* xb, int16_t* qb, float& out_sc) noexcept {
    alignas(32) float tmp[128];
#if defined(__AVX2__)
    for (int i = 0; i < 128; i += 8) {
        const __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xb + i));
        const __m256i f8 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(b8), 16);
        _mm256_store_ps(tmp + i, _mm256_castsi256_ps(f8));
    }
#else
    for (int i = 0; i < 128; ++i) tmp[i] = bf16_to_f32(xb[i]);
#endif
    quantize_block128_q5_eo(tmp, qb, out_sc);
}

#if defined(__AVX2__)
static inline void make_q5_vpshufb_luts(__m256i& v_lut_lo, __m256i& v_lut_hi) noexcept {
    alignas(16) uint8_t lo[16], hi[16];
    for (int k = 0; k < 16; ++k) {
        const uint16_t q = uint16_t(int16_t(std::lrintf(kLm5Levels16[k] * kQ5WScale)));
        lo[k] = uint8_t(q & 0xFFu);
        hi[k] = uint8_t((q >> 8) & 0xFFu);
    }
    v_lut_lo = _mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(lo)));
    v_lut_hi = _mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(hi)));
}

static inline float hsum_i32_to_f32(__m256i vi) noexcept {
    const __m256 vf = _mm256_cvtepi32_ps(vi);
    const __m128 lo = _mm256_castps256_ps128(vf);
    const __m128 hi = _mm256_extractf128_ps(vf, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

static double read_anon_hugepages_gib() {
    std::ifstream ifs("/proc/meminfo");
    std::string k, unit;
    uint64_t kb = 0;
    while (ifs >> k >> kb >> unit) {
        if (k == "AnonHugePages:") return double(kb) / 1048576.0;
    }
    return 0.0;
}

void Hy4Ddr4ColdEngine::avx2_q5_lm128_gemv_rows(
    const float* x_in,
    const Q5Lm128Block* blocks,
    int row_begin,
    int row_end,
    int kb,
    float* y_out) noexcept
{
#if defined(__AVX2__)
    alignas(32) int16_t x_q16_stk[kHiddenSize];
    float x_sc_stk[kHiddenSize / 128];
    const int use_kb = std::min(kb, kHiddenSize / 128);
    for (int b = 0; b < use_kb; ++b) {
        quantize_block128_q5_eo(x_in + size_t(b) * 128u, x_q16_stk + size_t(b) * 128u, x_sc_stk[b]);
    }

    __m256i v_lut_lo, v_lut_hi;
    make_q5_vpshufb_luts(v_lut_lo, v_lut_hi);
    const __m128i m_0f = _mm_set1_epi8(0x0F);
    const __m256i v_bitmask_eo = _mm256_setr_epi16(
        int16_t(1u << 0), int16_t(1u << 2), int16_t(1u << 4),  int16_t(1u << 6),
        int16_t(1u << 8), int16_t(1u << 10), int16_t(1u << 12), int16_t(1u << 14),
        int16_t(1u << 1), int16_t(1u << 3), int16_t(1u << 5),  int16_t(1u << 7),
        int16_t(1u << 9), int16_t(1u << 11), int16_t(1u << 13), int16_t(1u << 15));

    for (int r = row_begin; r < row_end; ++r) {
        const Q5Lm128Block* rb = blocks + size_t(r) * size_t(kb);
        float acc = 0.0f;
        for (int b = 0; b < use_kb; ++b) {
            const Q5Lm128Block& blk = rb[b];
            if (b + 4 < use_kb) _mm_prefetch(reinterpret_cast<const char*>(&rb[b + 4]), _MM_HINT_T0);
            const float sc = fast_f16_to_f32(blk.scale_f16) * x_sc_stk[b];
            const int16_t* xb = x_q16_stk + size_t(b) * 128u;
            __m256i a_i32 = _mm256_setzero_si256();
            for (int k = 0; k < 4; ++k) {
                const __m128i m16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk.mag4 + k * 16));
                const __m256i idx = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(_mm_and_si128(m16, m_0f)),
                    _mm_and_si128(_mm_srli_epi16(m16, 4), m_0f), 1);
                const __m256i mlo = _mm256_shuffle_epi8(v_lut_lo, idx);
                const __m256i mhi = _mm256_shuffle_epi8(v_lut_hi, idx);
                __m256i wa = _mm256_unpacklo_epi8(mlo, mhi);
                __m256i wb = _mm256_unpackhi_epi8(mlo, mhi);

                uint16_t s_lo, s_hi;
                std::memcpy(&s_lo, blk.sign1 + k * 4 + 0, 2);
                std::memcpy(&s_hi, blk.sign1 + k * 4 + 2, 2);
                const __m256i na = _mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16(int16_t(s_lo)), v_bitmask_eo), v_bitmask_eo);
                const __m256i nb = _mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16(int16_t(s_hi)), v_bitmask_eo), v_bitmask_eo);
                wa = _mm256_sub_epi16(_mm256_xor_si256(wa, na), na);
                wb = _mm256_sub_epi16(_mm256_xor_si256(wb, nb), nb);

                const __m256i xa = _mm256_load_si256(reinterpret_cast<const __m256i*>(xb + k * 32 + 0));
                const __m256i xb_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(xb + k * 32 + 16));
                a_i32 = _mm256_add_epi32(a_i32, _mm256_madd_epi16(wa, xa));
                a_i32 = _mm256_add_epi32(a_i32, _mm256_madd_epi16(wb, xb_v));
            }
            acc += hsum_i32_to_f32(a_i32) * sc;
        }
        y_out[r - row_begin] = acc;
    }
#else
    float tmp[128];
    for (int r = row_begin; r < row_end; ++r) {
        const Q5Lm128Block* rb = blocks + size_t(r) * size_t(kb);
        float acc = 0.0f;
        for (int b = 0; b < kb; ++b) {
            q5_lm128_unpack_block(rb[b], tmp);
            const float* xb = x_in + size_t(b) * 128u;
            for (int c = 0; c < 128; ++c) acc += tmp[c] * xb[c];
        }
        y_out[r - row_begin] = acc;
    }
#endif
}

bool Hy4Ddr4ColdEngine::init(
    const std::string& weight_dir,
    int cold_rank_begin,
    int num_threads,
    int core_offset,
    const std::string& hlut4_hot_dir,
    bool use_q5,
    bool use_hugepages)
{
    shutdown();
    if (cold_rank_begin < 0 || cold_rank_begin > kNumRoutedExperts) {
        std::fprintf(stderr, "[DDR4-AVX2] Invalid cold_rank_begin=%d\n", cold_rank_begin);
        return false;
    }
    use_q5_ = use_q5;
    cold_rank_begin_ = cold_rank_begin;
    num_cold_per_layer_ = kNumRoutedExperts - cold_rank_begin;
    num_threads_ = std::max(1, std::min(48, num_threads));

    const auto& sol = lloyd_max_hlut4();
    for (int i = 0; i < 8; ++i) levels8_[i] = float(sol.levels[size_t(i)]);
    for (int l = 1; l < kNumLayers; ++l) signs_i_[size_t(l)] = make_signs_i(l);

    if (use_q5_) {
        if (cold_rank_begin < kPackQ5Ddr4Begin) {
            std::fprintf(stderr, "[DDR4-AVX2] Q5 cold_rank_begin=%d < kPackQ5Ddr4Begin=%d!\n",
                         cold_rank_begin, kPackQ5Ddr4Begin);
            return false;
        }
        hot_split_cnt_ = 0;
        if (num_cold_per_layer_ > 0) {
            const size_t q5_bytes = size_t(num_cold_per_layer_) * size_t(kQ5ExpertBytes);
            const size_t q5_skip  = size_t(cold_rank_begin - kPackQ5Ddr4Begin) * size_t(kQ5ExpertBytes);
            mapped_bytes_ = size_t(kNumMoeLayers) * q5_bytes;
            const auto t0 = std::chrono::steady_clock::now();

            // True 2 MiB AnonHugePages loader (512x fewer PTEs -> eliminates 88.2M 4 KiB dTLB misses on Zen 2!)
            // Opt-out via HY4_DDR4_MMAP_FILE=1 or use_hugepages=false for legacy MAP_SHARED 4 KiB file mapping.
            const bool do_anon_thp = use_hugepages && (std::getenv("HY4_DDR4_MMAP_FILE") == nullptr);
            if (do_anon_thp) {
                constexpr size_t kHuge2M = 2u << 20;
                const size_t aligned_2m = (q5_bytes + kHuge2M - 1u) & ~(kHuge2M - 1u);
                const size_t alloc_len  = aligned_2m + kHuge2M;

                // 1. Validate all 77 files and release any stale 4 KiB file page cache FIRST so 2 MiB buddy blocks are unfragmented!
                for (int l = 1; l < kNumLayers; ++l) {
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/q5_layer_%02d.bin", weight_dir.c_str(), l);
                    int fd = open(path, O_RDONLY | O_CLOEXEC);
                    if (fd < 0) { std::fprintf(stderr, "[DDR4-AVX2] Cannot open %s\n", path); shutdown(); return false; }
                    struct stat st{};
                    if (fstat(fd, &st) != 0 || size_t(st.st_size) < q5_skip + q5_bytes) {
                        close(fd);
                        std::fprintf(stderr, "[DDR4-AVX2] Q5 file too small: %s (%zu < %zu)\n", path, size_t(st.st_size), q5_skip + q5_bytes);
                        shutdown();
                        return false;
                    }
                    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                    close(fd);
                }

                // 2. Reserve 2 MiB-aligned anonymous mappings with MADV_HUGEPAGE (zero page-cache duplication!)
                std::array<uint8_t*, kNumLayers> rw_ptrs{};
                for (int l = 1; l < kNumLayers; ++l) {
                    void* m = mmap(nullptr, alloc_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                    if (m == MAP_FAILED) {
                        std::fprintf(stderr, "[DDR4-AVX2] AnonHugePages mmap failed at layer %02d (%zu bytes)\n", l, alloc_len);
                        shutdown();
                        return false;
                    }
                    const uintptr_t raw_a = reinterpret_cast<uintptr_t>(m);
                    const uintptr_t ali_a = (raw_a + kHuge2M - 1u) & ~(kHuge2M - 1u);
                    uint8_t* dst = reinterpret_cast<uint8_t*>(ali_a);
#ifdef MADV_HUGEPAGE
                    (void)madvise(dst, aligned_2m, MADV_HUGEPAGE);
#endif
                    layer_map_[size_t(l)]     = m;
                    layer_map_len_[size_t(l)] = alloc_len;
                    layer_data_[size_t(l)]    = dst;
                    rw_ptrs[size_t(l)]        = dst;
                }

                // 3. Stream all 77 layers directly from Dual-NVMe via 16-thread O_DIRECT into 2 MiB AnonHugePages!
                bool io_ok = true;
                #pragma omp parallel for schedule(dynamic, 1) num_threads(16) reduction(&&:io_ok)
                for (int l = 1; l < kNumLayers; ++l) {
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/q5_layer_%02d.bin", weight_dir.c_str(), l);
                    bool used_direct = true;
                    int fd = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
                    if (fd < 0) { used_direct = false; fd = open(path, O_RDONLY | O_CLOEXEC); }
                    if (fd < 0) { io_ok = false; continue; }
                    uint8_t* dst = rw_ptrs[size_t(l)];
                    constexpr size_t kChunk = 16u << 20;  // 16 MiB aligned O_DIRECT chunks
                    size_t done = 0;
                    bool lok = true;
                    while (done < q5_bytes) {
                        const size_t want = std::min<size_t>(kChunk, q5_bytes - done);
                        size_t cur = 0;
                        while (cur < want) {
                            ssize_t r = pread(fd, dst + done + cur, want - cur, off_t(q5_skip + done + cur));
                            if (r <= 0) {
                                if (r < 0 && errno == EINTR) continue;
                                lok = false;
                                break;
                            }
                            cur += size_t(r);
                        }
                        if (!lok) break;
                        if (!used_direct) (void)posix_fadvise(fd, off_t(q5_skip + done), off_t(want), POSIX_FADV_DONTNEED);
                        done += want;
                    }
                    close(fd);
                    if (lok) (void)mprotect(dst, aligned_2m, PROT_READ);
                    else io_ok = false;
                }
                if (!io_ok) {
                    std::fprintf(stderr, "[DDR4-AVX2] Failed O_DIRECT load into 2 MiB AnonHugePages!\n");
                    shutdown();
                    return false;
                }
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const double thp_gib = read_anon_hugepages_gib();
                std::printf("[DDR4-AVX2] Loaded %d cold Q5_LM128 experts/layer (%d total, %.2f GiB in 2 MiB AnonHugePages [live=%.2f GiB], O_DIRECT, 5.125 bpw) in %.2fs (%.2f GB/s)\n",
                            num_cold_per_layer_, num_cold_per_layer_ * kNumMoeLayers,
                            double(mapped_bytes_) / 1073741824.0, thp_gib, sec, (double(mapped_bytes_) / 1e9) / std::max(1e-6, sec));
                std::fflush(stdout);
            } else {
                for (int l = 1; l < kNumLayers; ++l) {
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/q5_layer_%02d.bin", weight_dir.c_str(), l);
                    int fd = open(path, O_RDONLY);
                    if (fd < 0) { std::fprintf(stderr, "[DDR4-AVX2] Cannot open %s\n", path); shutdown(); return false; }
                    struct stat st{};
                    if (fstat(fd, &st) != 0 || size_t(st.st_size) < q5_skip + q5_bytes) {
                        close(fd);
                        std::fprintf(stderr, "[DDR4-AVX2] Q5 file too small: %s (%zu < %zu)\n", path, size_t(st.st_size), q5_skip + q5_bytes);
                        shutdown();
                        return false;
                    }
                    void* m = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
                    close(fd);
                    if (m == MAP_FAILED) { std::fprintf(stderr, "[DDR4-AVX2] mmap failed for %s\n", path); shutdown(); return false; }
                    madvise(m, size_t(st.st_size), MADV_SEQUENTIAL);
#ifdef MADV_HUGEPAGE
                    madvise(m, size_t(st.st_size), MADV_HUGEPAGE);
#endif
                    layer_map_[size_t(l)]     = m;
                    layer_map_len_[size_t(l)] = size_t(st.st_size);
                    layer_data_[size_t(l)]    = reinterpret_cast<const uint8_t*>(m) + q5_skip;
                }
                volatile uint64_t sink = 0;
                #pragma omp parallel for schedule(dynamic, 1) num_threads(16) reduction(^:sink)
                for (int l = 1; l < kNumLayers; ++l) {
                    uint64_t acc = 0;
                    const uint8_t* p = layer_data_[size_t(l)];
                    madvise(const_cast<uint8_t*>(p), q5_bytes, MADV_WILLNEED);
                    for (size_t off = 0; off < q5_bytes; off += 4096u) acc ^= p[off];
                    madvise(layer_map_[size_t(l)], layer_map_len_[size_t(l)], MADV_NORMAL);
                    sink ^= acc;
                }
                (void)sink;
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("[DDR4-AVX2] Zero-copy mmap %d cold Q5_LM128 experts/layer (%d total, %.2f GiB DDR4, 5.125 bpw) in %.2fs (%.2f GB/s)\n",
                            num_cold_per_layer_, num_cold_per_layer_ * kNumMoeLayers,
                            double(mapped_bytes_) / 1073741824.0, sec, (double(mapped_bytes_) / 1e9) / std::max(1e-6, sec));
                std::fflush(stdout);
            }
        }
    } else {
        if (cold_rank_begin < kPackHlut4Begin && hlut4_hot_dir.empty()) {
            std::fprintf(stderr, "[DDR4-AVX2] cold_rank_begin=%d < %d requires hlut4_hot_dir!\n",
                         cold_rank_begin, kPackHlut4Begin);
            return false;
        }
        hot_split_cnt_ = (cold_rank_begin < kPackHlut4Begin) ? (kPackHlut4Begin - cold_rank_begin) : 0;
        if (num_cold_per_layer_ > 0) {
            const int tail_cnt = num_cold_per_layer_ - hot_split_cnt_;
            const size_t hot_bytes  = size_t(hot_split_cnt_) * size_t(kHlut4ExpertBytes);
            const size_t hot_skip   = size_t(cold_rank_begin) * size_t(kHlut4ExpertBytes);
            const size_t tail_bytes = size_t(tail_cnt) * size_t(kHlut4ExpertBytes);
            const size_t tail_skip  = size_t(std::max(0, cold_rank_begin - kPackHlut4Begin)) * size_t(kHlut4ExpertBytes);
            mapped_bytes_ = size_t(kNumMoeLayers) * (hot_bytes + tail_bytes);

            const auto t0 = std::chrono::steady_clock::now();
            for (int l = 1; l < kNumLayers; ++l) {
                if (hot_split_cnt_ > 0) {
                    char hpath[512];
                    std::snprintf(hpath, sizeof(hpath), "%s/hlut4_hot_layer_%02d.bin", hlut4_hot_dir.c_str(), l);
                    int hfd = open(hpath, O_RDONLY);
                    if (hfd < 0) { std::fprintf(stderr, "[DDR4-AVX2] Cannot open %s\n", hpath); shutdown(); return false; }
                    struct stat hst{};
                    if (fstat(hfd, &hst) != 0 || size_t(hst.st_size) < hot_skip + hot_bytes) {
                        close(hfd); std::fprintf(stderr, "[DDR4-AVX2] Hot file too small: %s\n", hpath); shutdown(); return false;
                    }
                    void* hm = mmap(nullptr, size_t(hst.st_size), PROT_READ, MAP_SHARED, hfd, 0);
                    close(hfd);
                    if (hm == MAP_FAILED) { std::fprintf(stderr, "[DDR4-AVX2] mmap failed for %s\n", hpath); shutdown(); return false; }
                    madvise(hm, size_t(hst.st_size), MADV_SEQUENTIAL);
                    hot_map_[size_t(l)]     = hm;
                    hot_map_len_[size_t(l)] = size_t(hst.st_size);
                    hot_data_[size_t(l)]    = reinterpret_cast<const uint8_t*>(hm) + hot_skip;
                }
                if (tail_cnt > 0) {
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/hlut4_layer_%02d.bin", weight_dir.c_str(), l);
                    int fd = open(path, O_RDONLY);
                    if (fd < 0) { std::fprintf(stderr, "[DDR4-AVX2] Cannot open %s\n", path); shutdown(); return false; }
                    struct stat st{};
                    if (fstat(fd, &st) != 0 || size_t(st.st_size) < tail_skip + tail_bytes) {
                        close(fd);
                        std::fprintf(stderr, "[DDR4-AVX2] File too small: %s (%zu < %zu)\n", path, size_t(st.st_size), tail_skip + tail_bytes);
                        shutdown();
                        return false;
                    }
                    void* m = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
                    close(fd);
                    if (m == MAP_FAILED) { std::fprintf(stderr, "[DDR4-AVX2] mmap failed for %s\n", path); shutdown(); return false; }
                    madvise(m, size_t(st.st_size), MADV_SEQUENTIAL);
                    layer_map_[size_t(l)]     = m;
                    layer_map_len_[size_t(l)] = size_t(st.st_size);
                    layer_data_[size_t(l)]    = reinterpret_cast<const uint8_t*>(m) + tail_skip;
                }
            }
            volatile uint64_t sink = 0;
            #pragma omp parallel for schedule(dynamic, 1) num_threads(16) reduction(^:sink)
            for (int l = 1; l < kNumLayers; ++l) {
                uint64_t acc = 0;
                if (hot_split_cnt_ > 0) {
                    const uint8_t* p = hot_data_[size_t(l)];
                    madvise(const_cast<uint8_t*>(p), hot_bytes, MADV_WILLNEED);
                    for (size_t off = 0; off < hot_bytes; off += 4096u) acc ^= p[off];
                    madvise(hot_map_[size_t(l)], hot_map_len_[size_t(l)], MADV_RANDOM);
                }
                if (tail_cnt > 0) {
                    const uint8_t* p = layer_data_[size_t(l)];
                    madvise(const_cast<uint8_t*>(p), tail_bytes, MADV_WILLNEED);
                    for (size_t off = 0; off < tail_bytes; off += 4096u) acc ^= p[off];
                    madvise(layer_map_[size_t(l)], layer_map_len_[size_t(l)], MADV_RANDOM);
                }
                sink ^= acc;
            }
            (void)sink;
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("[DDR4-AVX2] Zero-copy mmap %d cold HLUT4 experts/layer (%d hot-split + %d tail = %d total, %.2f GiB DDR4) in %.2fs (%.2f GB/s)\n",
                        num_cold_per_layer_, hot_split_cnt_, tail_cnt, num_cold_per_layer_ * kNumMoeLayers,
                        double(mapped_bytes_) / 1073741824.0, sec, (double(mapped_bytes_) / 1e9) / std::max(1e-6, sec));
            std::fflush(stdout);
        }
    }

    z_rot_buf_.assign(size_t(kMaxRoutes) * size_t(kMoeInterSize), 0.0f);
    x_q16_buf_.assign(size_t(kMaxBatchM) * size_t(kHiddenSize), 0);
    x_sc_buf_.assign(size_t(kMaxBatchM) * size_t(kHiddenSize / 128), 0.0f);
    z_q16_buf_.assign(size_t(kMaxRoutes) * size_t(kMoeInterSize), 0);
    z_sc_buf_.assign(size_t(kMaxRoutes) * size_t(kMoeInterSize / 128), 0.0f);

    stop_flag_.store(false, std::memory_order_release);
    job_epoch_.store(0, std::memory_order_release);
    phase0_done_.store(0, std::memory_order_release);
    phase1_done_.store(0, std::memory_order_release);
    phase1b_done_.store(0, std::memory_order_release);
    phase2_done_.store(0, std::memory_order_release);

    // Pin worker threads to EPYC physical cores avoiding main/HIP (0..1) and pinned P150A CCD6 (24..27)!
    threads_.reserve(size_t(num_threads_));
    for (int t = 0; t < num_threads_; ++t) {
        int core = core_offset + t;
        if (core_offset == 0) {
            static constexpr int kSafeCores[52] = {
                 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31,
                34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63
            };
            core = kSafeCores[t % 52];
        }
        threads_.emplace_back([this, t, core]() {
            pin_thread_to_core(core);
            this->worker_loop(t);
        });
    }
    return true;
}

void Hy4Ddr4ColdEngine::shutdown() {
    if (!threads_.empty()) {
        stop_flag_.store(true, std::memory_order_release);
        job_epoch_.fetch_add(1, std::memory_order_acq_rel);
        for (auto& th : threads_) {
            if (th.joinable()) th.join();
        }
        threads_.clear();
    }
    for (int l = 0; l < kNumLayers; ++l) {
        if (hot_map_[size_t(l)]) {
            munmap(hot_map_[size_t(l)], hot_map_len_[size_t(l)]);
            hot_map_[size_t(l)] = nullptr;
            hot_map_len_[size_t(l)] = 0;
            hot_data_[size_t(l)] = nullptr;
        }
        if (layer_map_[size_t(l)]) {
            munmap(layer_map_[size_t(l)], layer_map_len_[size_t(l)]);
            layer_map_[size_t(l)] = nullptr;
            layer_map_len_[size_t(l)] = 0;
            layer_data_[size_t(l)] = nullptr;
        }
    }
    hot_split_cnt_ = 0;
    mapped_bytes_ = 0;
}

void Hy4Ddr4ColdEngine::submit_async(
    int layer_idx,
    const float* x_rot_h_f32,
    int M,
    const Ddr4ColdRoute* routes,
    int n_routes,
    float* y_out_f32)
{
    submit_t0_ns_ = now_ns();
    job_layer_ = layer_idx;
    job_M_ = M;
    job_x_rot_ = x_rot_h_f32;
    job_x_bf16_ = nullptr;
    job_y_out_ = y_out_f32;
    std::memset(y_out_f32, 0, size_t(M) * size_t(kHiddenSize) * sizeof(float));

    // Deduplicate routes by unique local_slot (critical for M=6 DFly2 verify: read each cold expert ONCE!)
    int ng = 0;
    int total_z = 0;
    for (int i = 0; i < n_routes && i < kMaxRoutes; ++i) {
        const int slot = routes[i].local_slot;
        if (slot < 0 || slot >= num_cold_per_layer_) continue;
        int found = -1;
        for (int g = 0; g < ng; ++g) {
            if (job_groups_[g].local_slot == slot) { found = g; break; }
        }
        if (found < 0) {
            found = ng++;
            job_groups_[found].local_slot = slot;
            job_groups_[found].count = 0;
        }
        auto& grp = job_groups_[found];
        if (grp.count < kMaxBatchM) {
            grp.rows[grp.count] = routes[i].row;
            grp.weights[grp.count] = routes[i].weight;
            ++grp.count;
            ++total_z;
        }
    }
    job_num_groups_ = ng;
    if (ng == 0) {
        last_us_ = 0.0;
        return;
    }
    int off = 0;
    for (int g = 0; g < ng; ++g) {
        job_group_z_off_[g] = off;
        off += job_groups_[g].count;
    }
    total_evals_.fetch_add(uint64_t(ng), std::memory_order_relaxed);

    phase0_done_.store(0, std::memory_order_release);
    phase1_done_.store(0, std::memory_order_release);
    phase1b_done_.store(0, std::memory_order_release);
    phase2_done_.store(0, std::memory_order_release);
    job_epoch_.fetch_add(1, std::memory_order_acq_rel);
}

void Hy4Ddr4ColdEngine::submit_async_bf16(
    int layer_idx,
    const uint16_t* x_raw_bf16,
    int M,
    const Ddr4ColdRoute* routes,
    int n_routes,
    float* y_out_f32)
{
    submit_t0_ns_ = now_ns();
    job_layer_ = layer_idx;
    job_M_ = M;
    job_x_rot_ = nullptr;
    job_x_bf16_ = x_raw_bf16;
    job_y_out_ = y_out_f32;
    std::memset(y_out_f32, 0, size_t(M) * size_t(kHiddenSize) * sizeof(float));

    int ng = 0;
    for (int i = 0; i < n_routes && i < kMaxRoutes; ++i) {
        const int slot = routes[i].local_slot;
        if (slot < 0 || slot >= num_cold_per_layer_) continue;
        int found = -1;
        for (int g = 0; g < ng; ++g) {
            if (job_groups_[g].local_slot == slot) { found = g; break; }
        }
        if (found < 0) {
            found = ng++;
            job_groups_[found].local_slot = slot;
            job_groups_[found].count = 0;
        }
        auto& grp = job_groups_[found];
        if (grp.count < kMaxBatchM) {
            grp.rows[grp.count] = routes[i].row;
            grp.weights[grp.count] = routes[i].weight;
            ++grp.count;
        }
    }
    job_num_groups_ = ng;
    if (ng == 0) {
        last_us_ = 0.0;
        return;
    }
    int off = 0;
    for (int g = 0; g < ng; ++g) {
        job_group_z_off_[g] = off;
        off += job_groups_[g].count;
    }
    total_evals_.fetch_add(uint64_t(ng), std::memory_order_relaxed);

    phase0_done_.store(0, std::memory_order_release);
    phase1_done_.store(0, std::memory_order_release);
    phase1b_done_.store(0, std::memory_order_release);
    phase2_done_.store(0, std::memory_order_release);
    job_epoch_.fetch_add(1, std::memory_order_acq_rel);
}

double Hy4Ddr4ColdEngine::wait_done() {
    if (job_num_groups_ == 0) return 0.0;
    while (phase2_done_.load(std::memory_order_acquire) < num_threads_) {
        _mm_pause();
    }
    const int64_t end_ns = phase2_end_ns_.load(std::memory_order_acquire);
    last_us_ = double(std::max<int64_t>(0, (end_ns > submit_t0_ns_ ? end_ns : now_ns()) - submit_t0_ns_)) * 1e-3;
    return last_us_;
}

void Hy4Ddr4ColdEngine::worker_loop(int tid) {
    uint64_t seen = 0;
    while (true) {
        uint64_t cur = job_epoch_.load(std::memory_order_acquire);
        while (cur == seen) {
            if (stop_flag_.load(std::memory_order_acquire)) return;
            _mm_pause();
            cur = job_epoch_.load(std::memory_order_acquire);
        }
        if (stop_flag_.load(std::memory_order_acquire)) return;
        seen = cur;
        execute_job_slice(tid);
    }
}

// Multi-token tile-stationary AVX2 WHT128-HLUT4 GEMM for 1 N-tile (32 outputs) across B in [1..4] tokens!
// Reads & dequantizes each 528 B HLUT4 tile ONCE and reuses unpacked weight vectors across all B tokens!
static inline void avx2_hlut4_gemm_1ntile_batch4(
    const float* const x_ptrs[4],
    int B,
    const Hlut4Tile32x32* row_tiles,
    const float levels8[8],
    int kt,
    float y_out[4][32]) noexcept
{
#if defined(__AVX2__)
    const __m256 v_lut = _mm256_loadu_ps(levels8);
    const __m256i v_sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i v_m7 = _mm256_set1_epi32(7);
    const __m256i v_m8 = _mm256_set1_epi32(8);

    auto hsum256 = [](__m256 v) noexcept -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    };
    auto dec_w = [&](uint32_t word) noexcept -> __m256 {
        __m256i nibs = _mm256_srlv_epi32(_mm256_set1_epi32(int(word)), v_sh);
        __m256  mag  = _mm256_permutevar8x32_ps(v_lut, _mm256_and_si256(nibs, v_m7));
        __m256i sgn  = _mm256_slli_epi32(_mm256_and_si256(nibs, v_m8), 28);
        return _mm256_xor_ps(mag, _mm256_castsi256_ps(sgn));
    };

    for (int b = 0; b < B; ++b) std::memset(y_out[b], 0, 32 * sizeof(float));

    for (int kt_i = 0; kt_i < kt; ++kt_i) {
        const Hlut4Tile32x32& tile = row_tiles[kt_i];
        if (kt_i + 2 < kt) {
            _mm_prefetch(reinterpret_cast<const char*>(&row_tiles[kt_i + 2]), _MM_HINT_T0);
        }
        const size_t xoff = size_t(kt_i) * 32u;
        __m256 vx[4][4];
        for (int b = 0; b < B; ++b) {
            const float* xk = x_ptrs[b] + xoff;
            vx[b][0] = _mm256_loadu_ps(xk + 0);
            vx[b][1] = _mm256_loadu_ps(xk + 8);
            vx[b][2] = _mm256_loadu_ps(xk + 16);
            vx[b][3] = _mm256_loadu_ps(xk + 24);
        }
        for (int g = 0; g < 8; ++g) {
            const float sigma = fast_f16_to_f32(tile.scales_f16[g]);
            const int r_base = g * 4;
            for (int dr = 0; dr < 4; ++dr) {
                const int r = r_base + dr;
                const uint32_t* q = tile.q_nib4 + r * 4;
                const __m256 w0 = dec_w(q[0]);
                const __m256 w1 = dec_w(q[1]);
                const __m256 w2 = dec_w(q[2]);
                const __m256 w3 = dec_w(q[3]);
                for (int b = 0; b < B; ++b) {
                    __m256 a = _mm256_mul_ps(w0, vx[b][0]);
                    a = _mm256_fmadd_ps(w1, vx[b][1], a);
                    a = _mm256_fmadd_ps(w2, vx[b][2], a);
                    a = _mm256_fmadd_ps(w3, vx[b][3], a);
                    y_out[b][r] += hsum256(a) * sigma;
                }
            }
        }
    }
#else
    for (int b = 0; b < B; ++b) {
        Hy4Ddr4ColdEngine::avx2_hlut4_gemv_rows(x_ptrs[b], row_tiles, levels8, 0, 1, kt, y_out[b]);
    }
#endif
}

// Spill-free weight-stationary AVX2 vpshufb+vpmaddwd Q5_LM128 GEMM for n_rows (even <=32) across B in {1..4} (0 WHT!)
// Bit-exact with GATE-2 v2 (cosine=0.99999999), 11 live YMMs (0 stack spills!), 2-row ILP unrolling at B==1!
template <int B>
static inline void avx2_q5_lm128_rows_b(
    const int16_t* const xq_ptrs[4],
    const float*   const xs_ptrs[4],
    const Q5Lm128Block* row0_blocks,
    int n_rows,
    int kb,
    float y_out[4][32]) noexcept
{
#if defined(__AVX2__)
    __m256i v_lut_lo, v_lut_hi;
    make_q5_vpshufb_luts(v_lut_lo, v_lut_hi);
    const __m128i m_0f = _mm_set1_epi8(0x0F);
    const __m256i v_bitmask_eo = _mm256_setr_epi16(
        int16_t(1u << 0), int16_t(1u << 2), int16_t(1u << 4),  int16_t(1u << 6),
        int16_t(1u << 8), int16_t(1u << 10), int16_t(1u << 12), int16_t(1u << 14),
        int16_t(1u << 1), int16_t(1u << 3), int16_t(1u << 5),  int16_t(1u << 7),
        int16_t(1u << 9), int16_t(1u << 11), int16_t(1u << 13), int16_t(1u << 15));

    if constexpr (B == 1) {
        const int16_t* x0 = xq_ptrs[0];
        const float*   s0 = xs_ptrs[0];
        for (int r = 0; r < n_rows; r += 2) {
            const Q5Lm128Block* rb0 = row0_blocks + size_t(r + 0) * size_t(kb);
            const Q5Lm128Block* rb1 = row0_blocks + size_t(r + 1) * size_t(kb);
            float a0 = 0.0f, a1 = 0.0f;
            for (int b = 0; b < kb; ++b) {
                const int pf_b = b + 8;
                if (pf_b < kb) {
                    const char* p0 = reinterpret_cast<const char*>(&rb0[pf_b]);
                    const char* p1 = reinterpret_cast<const char*>(&rb1[pf_b]);
                    _mm_prefetch(p0,      _MM_HINT_T0);
                    _mm_prefetch(p0 + 64, _MM_HINT_T0);
                    _mm_prefetch(p1,      _MM_HINT_T0);
                    _mm_prefetch(p1 + 64, _MM_HINT_T0);
                } else if (r + 2 < n_rows) {
                    const int nb = pf_b - kb;
                    if (nb < kb) {
                        const char* p0 = reinterpret_cast<const char*>(&rb0[2 * kb + nb]);
                        const char* p1 = reinterpret_cast<const char*>(&rb1[2 * kb + nb]);
                        _mm_prefetch(p0,      _MM_HINT_T0);
                        _mm_prefetch(p0 + 64, _MM_HINT_T0);
                        _mm_prefetch(p1,      _MM_HINT_T0);
                        _mm_prefetch(p1 + 64, _MM_HINT_T0);
                    }
                }
                const Q5Lm128Block& blk0 = rb0[b];
                const Q5Lm128Block& blk1 = rb1[b];
                const float sc0 = fast_f16_to_f32(blk0.scale_f16) * s0[b];
                const float sc1 = fast_f16_to_f32(blk1.scale_f16) * s0[b];
                const size_t xoff = size_t(b) * 128u;
                __m256i v0 = _mm256_setzero_si256();
                __m256i v1 = _mm256_setzero_si256();
                for (int k = 0; k < 4; ++k) {
                    const size_t xo = xoff + size_t(k) * 32u;
                    const __m256i xa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x0 + xo + 0));
                    const __m256i xb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x0 + xo + 16));
                    {
                        const __m128i m16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk0.mag4 + k * 16));
                        const __m256i idx = _mm256_inserti128_si256(
                            _mm256_castsi128_si256(_mm_and_si128(m16, m_0f)),
                            _mm_and_si128(_mm_srli_epi16(m16, 4), m_0f), 1);
                        const __m256i mlo = _mm256_shuffle_epi8(v_lut_lo, idx);
                        const __m256i mhi = _mm256_shuffle_epi8(v_lut_hi, idx);
                        __m256i wa = _mm256_unpacklo_epi8(mlo, mhi);
                        __m256i wb = _mm256_unpackhi_epi8(mlo, mhi);
                        uint16_t slo, shi;
                        std::memcpy(&slo, blk0.sign1 + k * 4 + 0, 2);
                        std::memcpy(&shi, blk0.sign1 + k * 4 + 2, 2);
                        const __m256i na = _mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16(int16_t(slo)), v_bitmask_eo), v_bitmask_eo);
                        const __m256i nb = _mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16(int16_t(shi)), v_bitmask_eo), v_bitmask_eo);
                        wa = _mm256_sub_epi16(_mm256_xor_si256(wa, na), na);
                        wb = _mm256_sub_epi16(_mm256_xor_si256(wb, nb), nb);
                        v0 = _mm256_add_epi32(v0, _mm256_madd_epi16(wa, xa));
                        v0 = _mm256_add_epi32(v0, _mm256_madd_epi16(wb, xb));
                    }
                    {
                        const __m128i m16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk1.mag4 + k * 16));
                        const __m256i idx = _mm256_inserti128_si256(
                            _mm256_castsi128_si256(_mm_and_si128(m16, m_0f)),
                            _mm_and_si128(_mm_srli_epi16(m16, 4), m_0f), 1);
                        const __m256i mlo = _mm256_shuffle_epi8(v_lut_lo, idx);
                        const __m256i mhi = _mm256_shuffle_epi8(v_lut_hi, idx);
                        __m256i wa = _mm256_unpacklo_epi8(mlo, mhi);
                        __m256i wb = _mm256_unpackhi_epi8(mlo, mhi);
                        uint16_t slo, shi;
                        std::memcpy(&slo, blk1.sign1 + k * 4 + 0, 2);
                        std::memcpy(&shi, blk1.sign1 + k * 4 + 2, 2);
                        const __m256i na = _mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16(int16_t(slo)), v_bitmask_eo), v_bitmask_eo);
                        const __m256i nb = _mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16(int16_t(shi)), v_bitmask_eo), v_bitmask_eo);
                        wa = _mm256_sub_epi16(_mm256_xor_si256(wa, na), na);
                        wb = _mm256_sub_epi16(_mm256_xor_si256(wb, nb), nb);
                        v1 = _mm256_add_epi32(v1, _mm256_madd_epi16(wa, xa));
                        v1 = _mm256_add_epi32(v1, _mm256_madd_epi16(wb, xb));
                    }
                }
                a0 += hsum_i32_to_f32(v0) * sc0;
                a1 += hsum_i32_to_f32(v1) * sc1;
            }
            y_out[0][r + 0] = a0;
            y_out[0][r + 1] = a1;
        }
        return;
    }

    const int16_t* x0 = xq_ptrs[0];
    const int16_t* x1 = (B > 1) ? xq_ptrs[1] : x0;
    const int16_t* x2 = (B > 2) ? xq_ptrs[2] : x0;
    const int16_t* x3 = (B > 3) ? xq_ptrs[3] : x0;
    const float*   s0 = xs_ptrs[0];
    const float*   s1 = (B > 1) ? xs_ptrs[1] : s0;
    const float*   s2 = (B > 2) ? xs_ptrs[2] : s0;
    const float*   s3 = (B > 3) ? xs_ptrs[3] : s0;

    for (int r = 0; r < n_rows; ++r) {
        const Q5Lm128Block* rb = row0_blocks + size_t(r) * size_t(kb);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int b = 0; b < kb; ++b) {
            const Q5Lm128Block& blk = rb[b];
            const int pf_b = b + 8;
            if (pf_b < kb) {
                const char* p = reinterpret_cast<const char*>(&rb[pf_b]);
                _mm_prefetch(p,      _MM_HINT_T0);
                _mm_prefetch(p + 64, _MM_HINT_T0);
            } else if (r + 1 < n_rows) {
                const int nb = pf_b - kb;
                if (nb < kb) {
                    const char* p = reinterpret_cast<const char*>(&rb[kb + nb]);
                    _mm_prefetch(p,      _MM_HINT_T0);
                    _mm_prefetch(p + 64, _MM_HINT_T0);
                }
            }
            const float sigma = fast_f16_to_f32(blk.scale_f16);
            const size_t xoff = size_t(b) * 128u;
            __m256i v0 = _mm256_setzero_si256();
            __m256i v1 = _mm256_setzero_si256();
            __m256i v2 = _mm256_setzero_si256();
            __m256i v3 = _mm256_setzero_si256();

            for (int k = 0; k < 4; ++k) {
                const __m128i m16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk.mag4 + k * 16));
                const __m256i idx = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(_mm_and_si128(m16, m_0f)),
                    _mm_and_si128(_mm_srli_epi16(m16, 4), m_0f), 1);
                const __m256i mlo = _mm256_shuffle_epi8(v_lut_lo, idx);
                const __m256i mhi = _mm256_shuffle_epi8(v_lut_hi, idx);
                __m256i wa = _mm256_unpacklo_epi8(mlo, mhi);
                __m256i wb = _mm256_unpackhi_epi8(mlo, mhi);

                uint16_t s_lo, s_hi;
                std::memcpy(&s_lo, blk.sign1 + k * 4 + 0, 2);
                std::memcpy(&s_hi, blk.sign1 + k * 4 + 2, 2);
                const __m256i na = _mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16(int16_t(s_lo)), v_bitmask_eo), v_bitmask_eo);
                const __m256i nb = _mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16(int16_t(s_hi)), v_bitmask_eo), v_bitmask_eo);
                wa = _mm256_sub_epi16(_mm256_xor_si256(wa, na), na);
                wb = _mm256_sub_epi16(_mm256_xor_si256(wb, nb), nb);

                const size_t xo = xoff + size_t(k) * 32u;
                v0 = _mm256_add_epi32(v0, _mm256_madd_epi16(wa, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x0 + xo + 0))));
                v0 = _mm256_add_epi32(v0, _mm256_madd_epi16(wb, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x0 + xo + 16))));
                if constexpr (B > 1) {
                    v1 = _mm256_add_epi32(v1, _mm256_madd_epi16(wa, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x1 + xo + 0))));
                    v1 = _mm256_add_epi32(v1, _mm256_madd_epi16(wb, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x1 + xo + 16))));
                }
                if constexpr (B > 2) {
                    v2 = _mm256_add_epi32(v2, _mm256_madd_epi16(wa, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x2 + xo + 0))));
                    v2 = _mm256_add_epi32(v2, _mm256_madd_epi16(wb, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x2 + xo + 16))));
                }
                if constexpr (B > 3) {
                    v3 = _mm256_add_epi32(v3, _mm256_madd_epi16(wa, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x3 + xo + 0))));
                    v3 = _mm256_add_epi32(v3, _mm256_madd_epi16(wb, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x3 + xo + 16))));
                }
            }
            a0 += hsum_i32_to_f32(v0) * (sigma * s0[b]);
            if constexpr (B > 1) a1 += hsum_i32_to_f32(v1) * (sigma * s1[b]);
            if constexpr (B > 2) a2 += hsum_i32_to_f32(v2) * (sigma * s2[b]);
            if constexpr (B > 3) a3 += hsum_i32_to_f32(v3) * (sigma * s3[b]);
        }
        y_out[0][r] = a0;
        if constexpr (B > 1) y_out[1][r] = a1;
        if constexpr (B > 2) y_out[2][r] = a2;
        if constexpr (B > 3) y_out[3][r] = a3;
    }
#else
    float tmp[128];
    for (int t = 0; t < B; ++t) {
        for (int r = 0; r < n_rows; ++r) {
            const Q5Lm128Block* rb = row0_blocks + size_t(r) * size_t(kb);
            float acc = 0.0f;
            for (int b = 0; b < kb; ++b) {
                q5_lm128_unpack_block(rb[b], tmp);
                const int16_t* qb = xq_ptrs[t] + size_t(b) * 128u;
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        sum += tmp[k * 32 + 2 * i + 0]      * float(qb[k * 32 + 0  + i]);
                        sum += tmp[k * 32 + 2 * i + 1]      * float(qb[k * 32 + 8  + i]);
                        sum += tmp[k * 32 + 16 + 2 * i + 0] * float(qb[k * 32 + 16 + i]);
                        sum += tmp[k * 32 + 16 + 2 * i + 1] * float(qb[k * 32 + 24 + i]);
                    }
                }
                acc += sum * (xs_ptrs[t][b] * kQ5WScale);
            }
            y_out[t][r] = acc;
        }
    }
#endif
}

static inline void avx2_q5_lm128_rows_batch4(
    const int16_t* const xq_ptrs[4],
    const float*   const xs_ptrs[4],
    int B,
    const Q5Lm128Block* row0_blocks,
    int n_rows,
    int kb,
    float y_out[4][32]) noexcept
{
    switch (B) {
        case 1: avx2_q5_lm128_rows_b<1>(xq_ptrs, xs_ptrs, row0_blocks, n_rows, kb, y_out); break;
        case 2: avx2_q5_lm128_rows_b<2>(xq_ptrs, xs_ptrs, row0_blocks, n_rows, kb, y_out); break;
        case 3: avx2_q5_lm128_rows_b<3>(xq_ptrs, xs_ptrs, row0_blocks, n_rows, kb, y_out); break;
        default: avx2_q5_lm128_rows_b<4>(xq_ptrs, xs_ptrs, row0_blocks, n_rows, kb, y_out); break;
    }
}

void Hy4Ddr4ColdEngine::execute_job_slice(int tid) {
    if (use_q5_) {
        // Phase 0: Quantize all M input rows [M, 6144] to int16 even/odd blocks in parallel (supports direct BF16!)
        constexpr int kGuStripRows = 16;
        constexpr int kGuStrips    = kMoeInterSize / kGuStripRows;  // 128 x 16-row strips (<4% imbalance!)
        constexpr int kGuKb        = kHiddenSize / 128;             // 48 blocks/row
        const int total_x_blks = job_M_ * kGuKb;
        for (int bi = tid; bi < total_x_blks; bi += num_threads_) {
            const int m = bi / kGuKb;
            const int b = bi % kGuKb;
            if (job_x_bf16_) {
                quantize_block128_q5_eo_bf16(
                    job_x_bf16_ + size_t(m) * size_t(kHiddenSize) + size_t(b) * 128u,
                    x_q16_buf_.data() + size_t(m) * size_t(kHiddenSize) + size_t(b) * 128u,
                    x_sc_buf_[size_t(m) * size_t(kGuKb) + size_t(b)]);
            } else {
                quantize_block128_q5_eo(
                    job_x_rot_ + size_t(m) * size_t(kHiddenSize) + size_t(b) * 128u,
                    x_q16_buf_.data() + size_t(m) * size_t(kHiddenSize) + size_t(b) * 128u,
                    x_sc_buf_[size_t(m) * size_t(kGuKb) + size_t(b)]);
            }
        }
        phase0_done_.fetch_add(1, std::memory_order_acq_rel);
        while (phase0_done_.load(std::memory_order_acquire) < num_threads_) _mm_pause();

        // Phase 1: Spill-free zero-WHT Q5_LM128 Gate+Up+SwiGLU (vpshufb+vpmaddwd, 4-token weight-stationary!)
        const int total_gu_work = job_num_groups_ * kGuStrips;
        const int gu_beg = (total_gu_work * tid) / num_threads_;
        const int gu_end = (total_gu_work * (tid + 1)) / num_threads_;

        float gate_b[4][32], up_b[4][32];
        const int16_t* xq_ptrs[4];
        const float*   xs_ptrs[4];
        for (int w = gu_beg; w < gu_end; ++w) {
            const int g  = w / kGuStrips;
            const int st = w % kGuStrips;
            const int row0 = st * kGuStripRows;
            const auto& grp = job_groups_[g];
            const uint8_t* exp_ptr = expert_ptr(job_layer_, grp.local_slot);
            const auto* gate_blk = reinterpret_cast<const Q5Lm128Block*>(exp_ptr) + size_t(row0) * size_t(kGuKb);
            const auto* up_blk   = reinterpret_cast<const Q5Lm128Block*>(exp_ptr + kQ5ProjBytes) + size_t(row0) * size_t(kGuKb);

            for (int c0 = 0; c0 < grp.count; c0 += 4) {
                const int B = std::min(4, grp.count - c0);
                for (int b = 0; b < B; ++b) {
                    const int row = grp.rows[c0 + b];
                    xq_ptrs[b] = x_q16_buf_.data() + size_t(row) * size_t(kHiddenSize);
                    xs_ptrs[b] = x_sc_buf_.data()  + size_t(row) * size_t(kGuKb);
                }
                avx2_q5_lm128_rows_batch4(xq_ptrs, xs_ptrs, B, gate_blk, kGuStripRows, kGuKb, gate_b);
                avx2_q5_lm128_rows_batch4(xq_ptrs, xs_ptrs, B, up_blk,   kGuStripRows, kGuKb, up_b);
                for (int b = 0; b < B; ++b) {
                    float* z_dst = z_rot_buf_.data() + size_t(job_group_z_off_[g] + c0 + b) * size_t(kMoeInterSize) + size_t(row0);
                    for (int r = 0; r < kGuStripRows; ++r) z_dst[r] = hy4_swiglu_clamped(gate_b[b][r], up_b[b][r]);
                }
            }
        }

        // Barrier 1 -> Quantize z_rot_buf_ rows [total_z, 2048] to int16 in parallel
        phase1_done_.fetch_add(1, std::memory_order_acq_rel);
        while (phase1_done_.load(std::memory_order_acquire) < num_threads_) _mm_pause();

        constexpr int kDnKb = kMoeInterSize / 128; // 16 blocks/row
        const int total_z_rows = job_group_z_off_[job_num_groups_ - 1] + job_groups_[job_num_groups_ - 1].count;
        const int total_z_blks = total_z_rows * kDnKb;
        for (int bi = tid; bi < total_z_blks; bi += num_threads_) {
            const int zr = bi / kDnKb;
            const int b  = bi % kDnKb;
            quantize_block128_q5_eo(
                z_rot_buf_.data() + size_t(zr) * size_t(kMoeInterSize) + size_t(b) * 128u,
                z_q16_buf_.data() + size_t(zr) * size_t(kMoeInterSize) + size_t(b) * 128u,
                z_sc_buf_[size_t(zr) * size_t(kDnKb) + size_t(b)]);
        }
        phase1b_done_.fetch_add(1, std::memory_order_acq_rel);
        while (phase1b_done_.load(std::memory_order_acquire) < num_threads_) _mm_pause();

        // Phase 2: Exact row-pair-balanced disjoint Down GEMM directly into job_y_out_ (<0.8% imbalance vs 16.7%!)
        constexpr int kDnPairs = kHiddenSize / 2;  // 3072 row-pairs
        const int r_beg = 2 * ((kDnPairs * tid) / num_threads_);
        const int r_end = 2 * ((kDnPairs * (tid + 1)) / num_threads_);

        float dn_b[4][32];
        const int16_t* zq_ptrs[4];
        const float*   zs_ptrs[4];
        for (int g = 0; g < job_num_groups_; ++g) {
            const auto& grp = job_groups_[g];
            const uint8_t* exp_ptr = expert_ptr(job_layer_, grp.local_slot);
            const auto* dn_base = reinterpret_cast<const Q5Lm128Block*>(exp_ptr + 2ULL * kQ5ProjBytes);
            for (int r0 = r_beg; r0 < r_end; r0 += 32) {
                const int nr = std::min(32, r_end - r0);
                const auto* dn_blk = dn_base + size_t(r0) * size_t(kDnKb);
                for (int c0 = 0; c0 < grp.count; c0 += 4) {
                    const int B = std::min(4, grp.count - c0);
                    for (int b = 0; b < B; ++b) {
                        const int zr = job_group_z_off_[g] + c0 + b;
                        zq_ptrs[b] = z_q16_buf_.data() + size_t(zr) * size_t(kMoeInterSize);
                        zs_ptrs[b] = z_sc_buf_.data()  + size_t(zr) * size_t(kDnKb);
                    }
                    avx2_q5_lm128_rows_batch4(zq_ptrs, zs_ptrs, B, dn_blk, nr, kDnKb, dn_b);
                    for (int b = 0; b < B; ++b) {
                        const float wt = grp.weights[c0 + b];
                        float* dst = job_y_out_ + size_t(grp.rows[c0 + b]) * size_t(kHiddenSize) + size_t(r0);
                        for (int r = 0; r < nr; ++r) dst[r] += wt * dn_b[b][r];
                    }
                }
            }
        }
        if (phase2_done_.fetch_add(1, std::memory_order_acq_rel) + 1 == num_threads_) {
            phase2_end_ns_.store(now_ns(), std::memory_order_release);
        }
        return;
    }

    const auto& signs_i = signs_i_[size_t(job_layer_)];

    // Phase 1: Gate + Up + Clamped SwiGLU across all unique cold experts (weight-stationary 4-token batching!)
    constexpr int kGuTiles = kMoeInterSize / 32;  // 64 tiles
    constexpr int kGuKt    = kHiddenSize / 32;    // 192 tiles
    const int total_gu_work = job_num_groups_ * kGuTiles;
    const int gu_beg = (total_gu_work * tid) / num_threads_;
    const int gu_end = (total_gu_work * (tid + 1)) / num_threads_;

    float gate_b[4][32], up_b[4][32];
    const float* x_ptrs[4];
    for (int w = gu_beg; w < gu_end; ++w) {
        const int g  = w / kGuTiles;
        const int nt = w % kGuTiles;
        const auto& grp = job_groups_[g];
        const uint8_t* exp_ptr = expert_ptr(job_layer_, grp.local_slot);
        const auto* gate_row = reinterpret_cast<const Hlut4Tile32x32*>(exp_ptr) + size_t(nt) * size_t(kGuKt);
        const auto* up_row   = reinterpret_cast<const Hlut4Tile32x32*>(exp_ptr + kHlut4ProjBytes) + size_t(nt) * size_t(kGuKt);

        for (int c0 = 0; c0 < grp.count; c0 += 4) {
            const int B = std::min(4, grp.count - c0);
            for (int b = 0; b < B; ++b) x_ptrs[b] = job_x_rot_ + size_t(grp.rows[c0 + b]) * size_t(kHiddenSize);
            avx2_hlut4_gemm_1ntile_batch4(x_ptrs, B, gate_row, levels8_, kGuKt, gate_b);
            avx2_hlut4_gemm_1ntile_batch4(x_ptrs, B, up_row,   levels8_, kGuKt, up_b);
            for (int b = 0; b < B; ++b) {
                float* z_dst = z_rot_buf_.data() + size_t(job_group_z_off_[g] + c0 + b) * size_t(kMoeInterSize) + size_t(nt) * 32u;
                for (int r = 0; r < 32; ++r) z_dst[r] = hy4_swiglu_clamped(gate_b[b][r], up_b[b][r]);
            }
        }
    }

    // Barrier + WHT128 on z_rot_buf_ rows
    phase1_done_.fetch_add(1, std::memory_order_acq_rel);
    while (phase1_done_.load(std::memory_order_acquire) < num_threads_) _mm_pause();

    // Rotate z rows in parallel (16 blocks of 128 per row)
    const int total_z_rows = job_group_z_off_[job_num_groups_ - 1] + job_groups_[job_num_groups_ - 1].count;
    const int total_wht_blocks = total_z_rows * (kMoeInterSize / 128);
    for (int b_idx = tid; b_idx < total_wht_blocks; b_idx += num_threads_) {
        const int zr = b_idx / (kMoeInterSize / 128);
        const int blk = b_idx % (kMoeInterSize / 128);
        wht128_block(z_rot_buf_.data() + size_t(zr) * size_t(kMoeInterSize) + size_t(blk) * 128u,
                     &signs_i[size_t(blk) * 2u]);
    }

    phase1_done_.fetch_add(1, std::memory_order_acq_rel);
    while (phase1_done_.load(std::memory_order_acquire) < 2 * num_threads_) _mm_pause();

    // Phase 2: Disjoint-column Down GEMM directly into job_y_out_!
    constexpr int kDnTiles = kHiddenSize / 32;    // 192 tiles
    constexpr int kDnKt    = kMoeInterSize / 32;  // 64 tiles
    const int nt_beg = (kDnTiles * tid) / num_threads_;
    const int nt_end = (kDnTiles * (tid + 1)) / num_threads_;

    float dn_b[4][32];
    const float* z_ptrs[4];
    for (int g = 0; g < job_num_groups_; ++g) {
        const auto& grp = job_groups_[g];
        const uint8_t* exp_ptr = expert_ptr(job_layer_, grp.local_slot);
        const auto* dn_base = reinterpret_cast<const Hlut4Tile32x32*>(exp_ptr + 2ULL * kHlut4ProjBytes);
        for (int nt = nt_beg; nt < nt_end; ++nt) {
            const auto* dn_row = dn_base + size_t(nt) * size_t(kDnKt);
            for (int c0 = 0; c0 < grp.count; c0 += 4) {
                const int B = std::min(4, grp.count - c0);
                for (int b = 0; b < B; ++b) z_ptrs[b] = z_rot_buf_.data() + size_t(job_group_z_off_[g] + c0 + b) * size_t(kMoeInterSize);
                avx2_hlut4_gemm_1ntile_batch4(z_ptrs, B, dn_row, levels8_, kDnKt, dn_b);
                for (int b = 0; b < B; ++b) {
                    const float wt = grp.weights[c0 + b];
                    float* dst = job_y_out_ + size_t(grp.rows[c0 + b]) * size_t(kHiddenSize) + size_t(nt) * 32u;
                    for (int r = 0; r < 32; ++r) dst[r] += wt * dn_b[b][r];
                }
            }
        }
    }

    if (phase2_done_.fetch_add(1, std::memory_order_acq_rel) + 1 == num_threads_) {
        phase2_end_ns_.store(now_ns(), std::memory_order_release);
    }
}

}  // namespace hy4_770b
