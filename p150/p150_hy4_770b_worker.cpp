// Pure C++20 2x Tenstorrent Blackhole P150A On-Device Tensix Worker for tencent/Hy4 (770B)
// ZERO host CPU AVX2 expert math, ZERO Python anywhere!
// Hardware path:
//   - 1x2 Blackhole P150A MeshDevice over 3.2 Tbps FABRIC_1D (4x 800G Ethernet links)
//   - Non-latent [6144, 2048] MoE TP2 shard: I_shard = 1024/card, 384 128x128 superblocks/proj
//   - 96 active Tensix cores/card (192 cores across 2x P150A), num_sb = 4 superblocks/core
//   - On-chip TRISC SFPU dequantization directly into Float16_b CB 1 + Tensix FPU matmul_block:
//       * QUANT_MODE == 2 : WHT128-HLUT3 (3.125 bpw, 6,400 B/superblock, 7.031 MiB/expert/card)
//       * QUANT_MODE == 3 : WHT128-HLUT2 (2.125 bpw, 4,352 B/superblock, 4.781 MiB/expert/card)
//   - Supports BOTH Profile A (pure HLUT3: 57/layer = 30.14 GiB/card) and
//     Profile B (mixed 24 HLUT3 + 49 HLUT2 = 73/layer = 30.30 GiB/card)
//   - O(1) l1_slot_lut dynamic hot-expert pinning into 330 MiB on-die Tensix L1 SRAM
//   - Strict graceful SIGTERM/SIGINT & IPC SHUTDOWN teardown (drains mesh command queue cleanly)

#include "hy4_770b_p150_shm.hpp"
#include "hy4_770b_p150_spec.hpp"
#include "../src/hy4_770b_hlut_pack.hpp"

#include <ttnn/device.hpp>
#include <ttnn/cluster.hpp>
#include <ttnn/config.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <ttnn/operations/generic/generic_op.hpp>
#include <ttnn/operations/ccl/all_reduce/all_reduce.hpp>
#include <ttnn/operations/ccl/broadcast/broadcast.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <immintrin.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <csignal>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace hy4_770b {
namespace {

using namespace tt::tt_metal;

volatile sig_atomic_t g_stop = 0;
void on_sig(int) { g_stop = 1; }

struct TraceStorageContextGuard {
    TraceStorageContextGuard()  { tt::tt_metal::push_allocation_context("trace_storage"); }
    ~TraceStorageContextGuard() { tt::tt_metal::pop_allocation_context(); }
    TraceStorageContextGuard(const TraceStorageContextGuard&) = delete;
    TraceStorageContextGuard& operator=(const TraceStorageContextGuard&) = delete;
};

static constexpr uint32_t kBatchSlots        = 16u;
static constexpr uint32_t kMaxCompactSlots   = 640u;
static constexpr uint32_t kSlotIdsBytes      = 1344u;  // [0..1279]=640 uint16_t slot_ids, [1280..1343]=64 B ldesc
static constexpr uint32_t kKSplitsUp         = 12u;    // 48 / 4 = 12 k-splits, 8*12 = 96 cores
static constexpr uint32_t kKSplitsDown       = 2u;     // 8  / 4 = 2  k-splits, 48*2 = 96 cores
static constexpr uint32_t kNumSbPerCore      = 4u;     // 4 superblocks/core (kt_core = 16 tiles = 512)
static constexpr uint32_t kSbPerProj         = 384u;   // 8 x 48 = 384 superblocks/proj/card
static constexpr uint32_t kL1SmallSizeBytes  = 0u;
static constexpr uint32_t kPrefillMaxRows    = 512u;   // Exact 6 pages/row at H=6144 (0 padding!)
static constexpr uint32_t kHiddenTiles       = uint32_t(kHiddenSize / 32);       // 192
static constexpr uint32_t kInterShardTiles   = uint32_t((kMoeInterSize / 2) / 32); // 32

static constexpr uint32_t kQModeHlut4        = 1u;
static constexpr uint32_t kQModeHlut3        = 2u;
static constexpr uint32_t kQModeHlut2        = 3u;
static constexpr uint32_t kPageBytesHlut4    = 8448u;
static constexpr uint32_t kPageBytesHlut3    = 6400u;
static constexpr uint32_t kPageBytesHlut2    = 4352u;

void check_no_existing_p150_owner() {
    namespace fs = std::filesystem;
    const std::string self_pid = std::to_string(::getpid());
    for (const auto& proc : fs::directory_iterator("/proc")) {
        const auto pid = proc.path().filename().string();
        if (pid.empty() || pid.find_first_not_of("0123456789") != std::string::npos || pid == self_pid) continue;
        std::error_code ec;
        fs::directory_iterator fd_it(proc.path() / "fd", fs::directory_options::skip_permission_denied, ec);
        if (ec) continue;
        for (const auto& entry : fd_it) {
            const auto link = fs::read_symlink(entry.path(), ec);
            if (ec) { ec.clear(); continue; }
            if (link.string().rfind("/dev/tenstorrent/", 0) == 0) {
                throw std::runtime_error("Blackhole P150A already owned by PID " + pid);
            }
        }
    }
}

struct LockedBuffer {
    void* ptr = nullptr;
    size_t size = 0;

    explicit LockedBuffer(size_t nbytes) : size((nbytes + 4095ULL) & ~4095ULL) {
        if (::posix_memalign(&ptr, 4096, size) != 0) throw std::bad_alloc();
        (void)::mlock(ptr, size);
        std::memset(ptr, 0, size);
    }
    ~LockedBuffer() {
        if (ptr) {
            (void)::munlock(ptr, size);
            ::free(ptr);
        }
    }
    LockedBuffer(const LockedBuffer&) = delete;
    LockedBuffer& operator=(const LockedBuffer&) = delete;
};

CoreRangeSet make_core_set(const std::vector<CoreCoord>& coords) {
    std::set<CoreRange> ranges;
    for (const auto& c : coords) ranges.emplace(c, c);
    return CoreRangeSet(ranges);
}

CBDescriptor make_cb(uint32_t id, uint32_t pages, uint32_t page_bytes, tt::DataFormat fmt, const CoreRangeSet& cores) {
    CBDescriptor cb;
    cb.total_size = pages * page_bytes;
    cb.core_ranges = cores;
    cb.format_descriptors.push_back(CBFormatDescriptor{static_cast<uint8_t>(id), fmt, page_bytes, {}, {}});
    return cb;
}

std::vector<uint32_t> make_accessor_args(std::initializer_list<const ttnn::Tensor*> tensors, std::vector<uint32_t> prefix = {}) {
    std::vector<uint32_t> out = std::move(prefix);
    for (const auto* t : tensors) {
        const auto a = TensorAccessorArgs(t->buffer()).get_compile_time_args();
        out.insert(out.end(), a.begin(), a.end());
    }
    return out;
}

KernelDescriptor make_kernel(
    const std::string& path,
    const CoreRangeSet& cores,
    KernelDescriptor::ConfigDescriptor config,
    std::vector<uint32_t> ct_args = {})
{
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("Missing P150 kernel file: " + path);
    }
    KernelDescriptor kd;
    kd.kernel_source = path;
    kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kd.core_ranges = cores;
    kd.config = std::move(config);
    kd.compile_time_args = std::move(ct_args);
    return kd;
}

ttnn::Tensor run_generic_preserving_topology(
    const std::vector<ttnn::Tensor>& tensors,
    const ProgramDescriptor& program)
{
    if (tensors.empty()) throw std::invalid_argument("run_generic_preserving_topology: empty tensor list");
    const auto topo = tensors.back().tensor_topology();
    auto res = ttnn::generic_op(tensors, program);
    if (res.buffer() != tensors.back().buffer()) {
        throw std::runtime_error("ttnn::generic_op replaced caller-owned output buffer");
    }
    res.update_tensor_topology(topo);
    return res;
}

ttnn::Tensor alloc_l1(distributed::MeshDevice& mesh, const Shape& shape, DataType dtype, Layout layout, bool sharded_axis0) {
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), ttnn::L1_MEMORY_CONFIG)),
        &mesh,
        sharded_axis0
            ? TensorTopology::create_sharded_tensor_topology(mesh.shape(), 0)
            : TensorTopology::create_fully_replicated_tensor_topology(mesh.shape()));
}

ttnn::Tensor alloc_dram_sharded0(distributed::MeshDevice& mesh, const Shape& shape, DataType dtype, Layout layout) {
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), ttnn::DRAM_MEMORY_CONFIG)),
        &mesh,
        TensorTopology::create_sharded_tensor_topology(mesh.shape(), 0));
}

ttnn::Tensor alloc_dram_replicated(distributed::MeshDevice& mesh, const Shape& shape, DataType dtype, Layout layout) {
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), ttnn::DRAM_MEMORY_CONFIG)),
        &mesh,
        TensorTopology::create_fully_replicated_tensor_topology(mesh.shape()));
}

void write_sharded_pair_blocking(distributed::MeshDevice& mesh, ttnn::Tensor& dst, const void* tp0_data, const void* tp1_data) {
    auto borrowed = std::shared_ptr<distributed::MeshBuffer>(
        const_cast<distributed::MeshBuffer*>(&dst.mesh_buffer()), [](auto*) {});
    mesh.mesh_command_queue().enqueue_write_shards(
        borrowed,
        {distributed::ShardDataTransfer{distributed::MeshCoordinate{0, 0}}.host_data(const_cast<void*>(tp0_data)),
         distributed::ShardDataTransfer{distributed::MeshCoordinate{0, 1}}.host_data(const_cast<void*>(tp1_data))},
        true);
}

inline constexpr uint32_t tensix_tile_offset(int r, int c) noexcept {
    const int face = ((r >= 16) ? 2 : 0) + ((c >= 16) ? 1 : 0);
    return uint32_t(face * 256 + (r & 15) * 16 + (c & 15));
}

const std::array<std::array<uint16_t, 32>, 32>& tile_offset_lut() {
    static const auto lut = []() {
        std::array<std::array<uint16_t, 32>, 32> t{};
        for (int r = 0; r < 32; ++r) {
            for (int c = 0; c < 32; ++c) {
                t[size_t(r)][size_t(c)] = static_cast<uint16_t>(tensix_tile_offset(r, c));
            }
        }
        return t;
    }();
    return lut;
}

inline void bf16_row_add_to_f32_avx2(const uint16_t* src_bf16, float* dst_f32) noexcept {
    for (int j = 0; j < kHiddenSize; j += 8) {
        const __m128i b16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_bf16 + j));
        const __m256  val = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b16), 16));
        _mm256_storeu_ps(dst_f32 + j, _mm256_add_ps(_mm256_loadu_ps(dst_f32 + j), val));
    }
}

inline void f32_row_to_bf16_avx2(const float* src_f32, uint16_t* dst_bf16) noexcept {
    const __m256i v_7fff = _mm256_set1_epi32(0x00007FFFu);
    const __m256i v_one  = _mm256_set1_epi32(1);
    for (int j = 0; j < kHiddenSize; j += 16) {
        const __m256i u0 = _mm256_castps_si256(_mm256_loadu_ps(src_f32 + j));
        const __m256i u1 = _mm256_castps_si256(_mm256_loadu_ps(src_f32 + j + 8));
        const __m256i r0 = _mm256_srli_epi32(_mm256_add_epi32(u0, _mm256_add_epi32(v_7fff, _mm256_and_si256(_mm256_srli_epi32(u0, 16), v_one))), 16);
        const __m256i r1 = _mm256_srli_epi32(_mm256_add_epi32(u1, _mm256_add_epi32(v_7fff, _mm256_and_si256(_mm256_srli_epi32(u1, 16), v_one))), 16);
        const __m256i pk = _mm256_packus_epi32(r0, r1);
        const __m256i ord = _mm256_permute4x64_epi64(pk, 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_bf16 + j), ord);
    }
}

void build_wht128_block_matrix_local(int blk, const uint64_t* signs_tp, float q_mat[128][128]) {
    constexpr float kInvSqrt128 = 0.08838834764831844655f;
    const uint64_t s0 = signs_tp[size_t(blk) * 2u + 0u];
    const uint64_t s1 = signs_tp[size_t(blk) * 2u + 1u];
    for (int r = 0; r < 128; ++r) {
        const bool neg = (r < 64) ? (((s0 >> r) & 1ULL) != 0ULL) : (((s1 >> (r - 64)) & 1ULL) != 0ULL);
        const float sr = neg ? -kInvSqrt128 : kInvSqrt128;
        for (int c = 0; c < 128; ++c) {
            const int pop = __builtin_popcount(unsigned(r & c));
            q_mat[r][c] = (pop & 1) ? -sr : sr;
        }
    }
}

uint32_t choose_decode_slot_bucket(uint32_t active) {
    static constexpr uint32_t kBuckets[] = {2u, 4u, 6u, 8u, 12u, 16u, 24u, 32u, 48u, 64u};
    for (uint32_t b : kBuckets) {
        if (active <= b) return b;
    }
    return std::min<uint32_t>(((active + kBatchSlots - 1u) / kBatchSlots) * kBatchSlots, kMaxCompactSlots);
}

uint32_t choose_compact_slot_bucket(uint32_t active) {
    const uint32_t step = (active <= 160u) ? 16u : 32u;
    const uint32_t rounded = ((active + step - 1u) / step) * step;
    return std::min<uint32_t>(std::max<uint32_t>(16u, rounded), kMaxCompactSlots);
}

// ============================================================================
// Host TP2 Superblock Packers (Hy4 770B: [2048, 6144] gate/up & [6144, 2048] down)
// Transposes W[N, K] -> W^T[K, N] in Exact Tensix 4-Face Order (384 superblocks/card!)
// ============================================================================
void pack_hlut4_expert_tp2(
    const Hlut4Tile32x32* tiles,  // [12,288] tiles
    bool is_down,
    uint8_t* tp0_dst,
    uint8_t* tp1_dst)
{
    constexpr uint32_t kPageB = kPageBytesHlut4;
    const auto& t_lut = tile_offset_lut();
    const int kt_full = is_down ? 64 : 192;
    const int nt4 = is_down ? 48 : 8;
    const int kt4 = is_down ? 8  : 48;

    for (int tp = 0; tp < 2; ++tp) {
        uint8_t* dst = (tp == 0) ? tp0_dst : tp1_dst;
        for (int nb4 = 0; nb4 < nt4; ++nb4) {
            for (int kb4 = 0; kb4 < kt4; ++kb4) {
                uint8_t* pg = dst + size_t(nb4 * kt4 + kb4) * kPageB;
                uint32_t* w_nib0 = reinterpret_cast<uint32_t*>(pg + 0u);
                uint32_t* w_nib1 = reinterpret_cast<uint32_t*>(pg + 4096u);
                uint16_t* sc_dst = reinterpret_cast<uint16_t*>(pg + 8192u);
                const int nt_base = (is_down ? nb4 : (tp * 8 + nb4)) * 4;
                const int kt_base = (is_down ? (tp * 8 + kb4) : kb4) * 4;

                const Hlut4Tile32x32* tp16[4][4];
                for (int ks = 0; ks < 4; ++ks) {
                    for (int ns = 0; ns < 4; ++ns) {
                        const auto* t = &tiles[size_t(nt_base + ns) * size_t(kt_full) + size_t(kt_base + ks)];
                        tp16[ks][ns] = t;
                        std::memcpy(sc_dst + (ks * 4 + ns) * 8, t->scales_f16, 16u);
                    }
                }
                for (int rk = 0; rk < 32; ++rk) {
                    const auto& roff = t_lut[size_t(rk)];
                    const int wi_k = rk >> 3;
                    const int sh_k = 4 * (rk & 7);
                    for (int cn = 0; cn < 32; ++cn) {
                        const int w_idx = cn * 4 + wi_k;
                        uint32_t nib0 = 0u, nib1 = 0u;
                        for (int ks = 0; ks < 2; ++ks) {
                            for (int ns = 0; ns < 4; ++ns) {
                                const uint32_t nb = (tp16[ks][ns]->q_nib4[w_idx] >> sh_k) & 0xFu;
                                nib0 |= (nb << (4u * uint32_t(ks * 4 + ns)));
                            }
                        }
                        for (int ks = 0; ks < 2; ++ks) {
                            for (int ns = 0; ns < 4; ++ns) {
                                const uint32_t nb = (tp16[ks + 2][ns]->q_nib4[w_idx] >> sh_k) & 0xFu;
                                nib1 |= (nb << (4u * uint32_t(ks * 4 + ns)));
                            }
                        }
                        const uint32_t tidx = roff[size_t(cn)];
                        w_nib0[tidx] = nib0;
                        w_nib1[tidx] = nib1;
                    }
                }
            }
        }
    }
}

void pack_hlut3_expert_tp2(
    const Hlut3Tile32x32* tiles,  // [12,288] tiles
    bool is_down,
    uint8_t* tp0_dst,
    uint8_t* tp1_dst)
{
    constexpr uint32_t kPageB = kPageBytesHlut3;
    const auto& t_lut = tile_offset_lut();
    const int kt_full = is_down ? 64 : 192;
    const int nt4 = is_down ? 48 : 8;
    const int kt4 = is_down ? 8  : 48;

    for (int tp = 0; tp < 2; ++tp) {
        uint8_t* dst = (tp == 0) ? tp0_dst : tp1_dst;
        for (int nb4 = 0; nb4 < nt4; ++nb4) {
            for (int kb4 = 0; kb4 < kt4; ++kb4) {
                uint8_t* pg = dst + size_t(nb4 * kt4 + kb4) * kPageB;
                uint32_t* w_lo2  = reinterpret_cast<uint32_t*>(pg + 0u);
                uint16_t* w_hi1  = reinterpret_cast<uint16_t*>(pg + 4096u);
                uint16_t* sc_dst = reinterpret_cast<uint16_t*>(pg + 6144u);
                const int nt_base = (is_down ? nb4 : (tp * 8 + nb4)) * 4;
                const int kt_base = (is_down ? (tp * 8 + kb4) : kb4) * 4;

                const Hlut3Tile32x32* tp16[4][4];
                for (int ks = 0; ks < 4; ++ks) {
                    for (int ns = 0; ns < 4; ++ns) {
                        const auto* t = &tiles[size_t(nt_base + ns) * size_t(kt_full) + size_t(kt_base + ks)];
                        tp16[ks][ns] = t;
                        std::memcpy(sc_dst + (ks * 4 + ns) * 8, t->scales_f16, 16u);
                    }
                }
                for (int rk = 0; rk < 32; ++rk) {
                    const auto& roff = t_lut[size_t(rk)];
                    for (int cn = 0; cn < 32; ++cn) {
                        const int idx = cn * 32 + rk;
                        const int wi  = idx >> 4;
                        const int sh2 = 2 * (idx & 15);
                        uint32_t lo_word = 0u;
                        uint16_t hi_word = 0u;
                        for (int ks = 0; ks < 4; ++ks) {
                            for (int ns = 0; ns < 4; ++ns) {
                                const uint32_t t_id = uint32_t(ks * 4 + ns);
                                const uint32_t mag  = (tp16[ks][ns]->q_lo2[wi] >> sh2) & 3u;
                                const uint32_t sgn  = (tp16[ks][ns]->q_hi1[cn] >> rk) & 1u;
                                lo_word |= (mag << (2u * t_id));
                                hi_word |= static_cast<uint16_t>(sgn << t_id);
                            }
                        }
                        const uint32_t tidx = roff[size_t(cn)];
                        w_lo2[tidx] = lo_word;
                        w_hi1[tidx] = hi_word;
                    }
                }
            }
        }
    }
}

void pack_hlut2_expert_tp2(
    const Hlut2Tile32x32* tiles,  // [12,288] tiles
    bool is_down,
    uint8_t* tp0_dst,
    uint8_t* tp1_dst)
{
    constexpr uint32_t kPageB = kPageBytesHlut2;
    const auto& t_lut = tile_offset_lut();
    const int kt_full = is_down ? 64 : 192;
    const int nt4 = is_down ? 48 : 8;
    const int kt4 = is_down ? 8  : 48;

    for (int tp = 0; tp < 2; ++tp) {
        uint8_t* dst = (tp == 0) ? tp0_dst : tp1_dst;
        for (int nb4 = 0; nb4 < nt4; ++nb4) {
            for (int kb4 = 0; kb4 < kt4; ++kb4) {
                uint8_t* pg = dst + size_t(nb4 * kt4 + kb4) * kPageB;
                uint32_t* w_q2   = reinterpret_cast<uint32_t*>(pg + 0u);
                uint16_t* sc_dst = reinterpret_cast<uint16_t*>(pg + 4096u);
                const int nt_base = (is_down ? nb4 : (tp * 8 + nb4)) * 4;
                const int kt_base = (is_down ? (tp * 8 + kb4) : kb4) * 4;

                const Hlut2Tile32x32* tp16[4][4];
                for (int ks = 0; ks < 4; ++ks) {
                    for (int ns = 0; ns < 4; ++ns) {
                        const auto* t = &tiles[size_t(nt_base + ns) * size_t(kt_full) + size_t(kt_base + ks)];
                        tp16[ks][ns] = t;
                        std::memcpy(sc_dst + (ks * 4 + ns) * 8, t->scales_f16, 16u);
                    }
                }
                for (int rk = 0; rk < 32; ++rk) {
                    const auto& roff = t_lut[size_t(rk)];
                    for (int cn = 0; cn < 32; ++cn) {
                        const int idx = cn * 32 + rk;
                        const int wi  = idx >> 4;
                        const int sh2 = 2 * (idx & 15);
                        uint32_t q_word = 0u;
                        for (int ks = 0; ks < 4; ++ks) {
                            for (int ns = 0; ns < 4; ++ns) {
                                const uint32_t t_id = uint32_t(ks * 4 + ns);
                                const uint32_t code = (tp16[ks][ns]->q_2b[wi] >> sh2) & 3u;
                                q_word |= (code << (2u * t_id));
                            }
                        }
                        w_q2[roff[size_t(cn)]] = q_word;
                    }
                }
            }
        }
    }
}

struct QuantBank {
    uint32_t n_experts = 0;
    ttnn::Tensor gate_pages;  // DRAM sharded [2, n_exp*384, page_u32] UINT32 ROW_MAJOR
    ttnn::Tensor up_pages;
    ttnn::Tensor down_pages;
    std::array<uint16_t, 256> hot_slot_for_eid = []() {
        std::array<uint16_t, 256> a{};
        a.fill(0xFFFFu);
        return a;
    }();
};

struct ResidentLayerBank {
    uint32_t layer = 0;
    QuantBank h4;
    QuantBank h3;
    QuantBank h2;
    ttnn::Tensor wht_q_i;     // DRAM sharded [2, 8, 128, 128] BFLOAT16 TILE (128 tiles/card)
};

struct CoreWorkspace770b {
    static constexpr uint32_t kDecodeFastWeightSlots = 64u;
    uint32_t m_tiles_ws = kBatchSlots;

    ttnn::Tensor x_tilized;             // Replicated [1, 16, 32, 6144] BFLOAT16 TILE (6 MiB)
    ttnn::Tensor x_decode_src;          // Card #0 staging [1, 1, 32, 6144] BFLOAT16 TILE (384 KiB)
    ttnn::Tensor slot_ids_src;          // Card #0 staging [1, 1, 1, 672] UINT16 ROW_MAJOR (1344 B)
    ttnn::Tensor weights_decode64_src;  // Card #0 staging [1, 64, 32, 32] FLOAT32 TILE (256 KiB)
    ttnn::Tensor weights_tiled_src;     // Card #0 staging [1, 640, 32, 32] FLOAT32 TILE (2.5 MiB)
    ttnn::Tensor x_decode;
    ttnn::Tensor slot_ids;
    ttnn::Tensor weights_decode64;
    ttnn::Tensor weights_tiled;
    ttnn::Tensor gate_partials;         // DRAM sharded [2, 12, 16*32, 1024] FLOAT32 TILE (24 MiB)
    ttnn::Tensor up_partials;           // DRAM sharded [2, 12, 16*32, 1024] FLOAT32 TILE (24 MiB)
    ttnn::Tensor z_rot_bf16;            // DRAM sharded [2, 1, 16*32, 1024] BFLOAT16 TILE (1 MiB)
    ttnn::Tensor down_partials;         // Aliases gate_partials (2 * 16 * 192 * 4096 = 24 MiB EXACT!)
    ttnn::Tensor combined_f32;          // DRAM sharded [2, 1, 32, 6144] FLOAT32 TILE (768 KiB)
    std::array<ttnn::Tensor, 3> finish_decode;

    explicit CoreWorkspace770b(distributed::MeshDevice& mesh)
        : x_tilized(alloc_dram_replicated(mesh, Shape({1, kBatchSlots, 32u, uint32_t(kHiddenSize)}), DataType::BFLOAT16, Layout::TILE)),
          x_decode_src(alloc_dram_replicated(mesh, Shape({1, 1u, 32u, uint32_t(kHiddenSize)}), DataType::BFLOAT16, Layout::TILE)),
          slot_ids_src(alloc_dram_replicated(mesh, Shape({1, 1, 1, kSlotIdsBytes / 2u}), DataType::UINT16, Layout::ROW_MAJOR)),
          weights_decode64_src(alloc_dram_replicated(mesh, Shape({1, kDecodeFastWeightSlots, 32u, 32u}), DataType::FLOAT32, Layout::TILE)),
          weights_tiled_src(alloc_dram_replicated(mesh, Shape({1, kMaxCompactSlots, 32u, 32u}), DataType::FLOAT32, Layout::TILE)),
          gate_partials(alloc_dram_sharded0(mesh, Shape({1, kKSplitsUp, kBatchSlots * 32u, kP150Hy4InterShard}), DataType::FLOAT32, Layout::TILE)),
          up_partials(alloc_dram_sharded0(mesh, Shape({1, kKSplitsUp, kBatchSlots * 32u, kP150Hy4InterShard}), DataType::FLOAT32, Layout::TILE)),
          z_rot_bf16(alloc_dram_sharded0(mesh, Shape({1, 1, kBatchSlots * 32u, kP150Hy4InterShard}), DataType::BFLOAT16, Layout::TILE)),
          down_partials(gate_partials),
          combined_f32(alloc_dram_sharded0(mesh, Shape({1, 1, 32u, uint32_t(kHiddenSize)}), DataType::FLOAT32, Layout::TILE))
    {}
};

struct CompactPrefillArena770b {
    uint32_t max_rows = kPrefillMaxRows;
    ttnn::Tensor y_accum_f32;  // Sharded [2, 1, max_rows, 6144] FLOAT32 TILE (6 exact 4096 B pages/row!)
    std::array<ttnn::Tensor, 2> finish_tensors;

    CompactPrefillArena770b(distributed::MeshDevice& mesh, uint32_t mrows)
        : max_rows(mrows),
          y_accum_f32(alloc_dram_sharded0(mesh, Shape({1, 1, mrows, uint32_t(kHiddenSize)}), DataType::FLOAT32, Layout::TILE))
    {}
};

struct WorkerConfig {
    std::string weight_root = "<HY4_770B_WEIGHT_ROOT>";
    std::string kernels_dir = "<HY4_770B_WORKSPACE_ROOT>/p150/kernels";
    std::string shm_name    = "/hy4_770b_p150_ipc";
    uint32_t profile_id     = 0;    // 0=ProfileA, 1=ProfileB
    int hlut4_begin         = 0;    // rank in [0..110) from hlut4_hot/
    int hlut4_end           = 0;
    int hlut3_begin         = 0;    // rank in [0..125)
    int hlut3_end           = 0;
    int hlut2_begin         = 0;    // rank in [60..155)
    int hlut2_end           = 0;
    bool enable_trace       = true;
    bool x16_relay          = true;
    bool self_test          = false;
    int max_layers_load     = kNumMoeLayers;  // 77
};

class P150Hy4770bEngine {
public:
    explicit P150Hy4770bEngine(const WorkerConfig& cfg)
        : cfg_(cfg),
          n_h4_(uint32_t(std::max(0, cfg.hlut4_end - cfg.hlut4_begin))),
          n_h3_(uint32_t(std::max(0, cfg.hlut3_end - cfg.hlut3_begin))),
          n_h2_(uint32_t(std::max(0, cfg.hlut2_end - cfg.hlut2_begin))),
          hot_qmode_(n_h4_ > 0u ? kQModeHlut4 : (n_h3_ > 0u ? kQModeHlut3 : kQModeHlut2))
    {
        check_no_existing_p150_owner();
        ttnn::CONFIG.set<"throw_exception_on_fallback">(true);
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
        const auto dispatch = ttnn::device::create_cluster_aware_dispatch_config();
        // Keep trace region compact (256 MiB) so 30.3 GiB GDDR6 weight footprint always fits cleanly!
        trace_region_bytes_ = cfg_.enable_trace ? (256u << 20) : 0u;
        mesh_ = ttnn::distributed::open_mesh_device(
            ttnn::MeshShape{1, 2}, kL1SmallSizeBytes, trace_region_bytes_, 1, dispatch, std::nullopt, {0, 1});

        grid_ = mesh_->compute_with_storage_grid_size();
        total_cores_ = uint32_t(grid_.x * grid_.y);
        if (total_cores_ < 96u) {
            throw std::runtime_error("Blackhole P150A grid too small for 96-core Hy4-770B: " + std::to_string(total_cores_));
        }

        ws_            = std::make_unique<CoreWorkspace770b>(*mesh_);
        x_rm_src_      = alloc_dram_replicated(*mesh_, Shape({1, 1, kPrefillMaxRows, uint32_t(kHiddenSize)}), DataType::BFLOAT16, Layout::TILE);
        slot_meta_src_ = alloc_dram_replicated(*mesh_, Shape({1, 1, kMaxCompactSlots, 32u}), DataType::UINT32, Layout::ROW_MAJOR);
        l1_lut_h4_     = alloc_dram_replicated(*mesh_, Shape({1, uint32_t(kNumLayers), 256u}), DataType::UINT16, Layout::ROW_MAJOR);
        l1_lut_h3_     = alloc_dram_replicated(*mesh_, Shape({1, uint32_t(kNumLayers), 256u}), DataType::UINT16, Layout::ROW_MAJOR);
        l1_lut_h2_     = alloc_dram_replicated(*mesh_, Shape({1, uint32_t(kNumLayers), 256u}), DataType::UINT16, Layout::ROW_MAJOR);
        dummy_hot_h4_  = alloc_dram_sharded0(*mesh_, Shape({1, 1u, kPageBytesHlut4 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);
        dummy_hot_h3_  = alloc_dram_sharded0(*mesh_, Shape({1, 1u, kPageBytesHlut3 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);
        dummy_hot_h2_  = alloc_dram_sharded0(*mesh_, Shape({1, 1u, kPageBytesHlut2 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);
        arena_         = std::make_unique<CompactPrefillArena770b>(*mesh_, kPrefillMaxRows);

        // Pin ultra-hot experts into 330 MiB on-die Tensix L1 SRAM while guaranteeing 256 KiB/core CB guard!
        const uint32_t hot_page_b = (hot_qmode_ == kQModeHlut4) ? kPageBytesHlut4
                                  : ((hot_qmode_ == kQModeHlut3) ? kPageBytesHlut3 : kPageBytesHlut2);
        const uint32_t max_try    = (hot_qmode_ == kQModeHlut4) ? 14u
                                  : ((hot_qmode_ == kQModeHlut3) ? kP150MaxHotL1Hlut3 : kP150MaxHotL1Hlut2);
        const uint32_t guard_pages = (total_cores_ * kP150Fabric1DCbGuardPerCore) / 4096u;
        for (uint32_t k_try = max_try; k_try >= 2u; --k_try) {
            try {
                auto hg = alloc_l1(*mesh_, Shape({1, k_try * kSbPerProj, hot_page_b / 4u}), DataType::UINT32, Layout::ROW_MAJOR, true);
                auto hu = alloc_l1(*mesh_, Shape({1, k_try * kSbPerProj, hot_page_b / 4u}), DataType::UINT32, Layout::ROW_MAJOR, true);
                auto hd = alloc_l1(*mesh_, Shape({1, k_try * kSbPerProj, hot_page_b / 4u}), DataType::UINT32, Layout::ROW_MAJOR, true);
                auto cb_guard = alloc_l1(*mesh_, Shape({1, guard_pages, 1024u}), DataType::UINT32, Layout::ROW_MAJOR, true);
                (void)cb_guard;
                hot_gate_l1_ = std::move(hg);
                hot_up_l1_   = std::move(hu);
                hot_down_l1_ = std::move(hd);
                num_hot_shards_ = k_try;
                pinned_hot_l1_bytes_ = hot_gate_l1_->buffer()->size() + hot_up_l1_->buffer()->size() + hot_down_l1_->buffer()->size();
                break;
            } catch (...) {
                hot_gate_l1_.reset();
                hot_up_l1_.reset();
                hot_down_l1_.reset();
            }
        }

        // Initialize O(1) L1 hot-expert LUTs to 0xFFFF
        std::memset(host_slot_lut_h4_.ptr, 0xFF, host_slot_lut_h4_.size);
        std::memset(host_slot_lut_h3_.ptr, 0xFF, host_slot_lut_h3_.size);
        std::memset(host_slot_lut_h2_.ptr, 0xFF, host_slot_lut_h2_.size);
        const distributed::MeshCoordinateRange all_devs(mesh_->shape());
        auto& cq = mesh_->mesh_command_queue();
        cq.enqueue_write_shard_to_sub_grid(l1_lut_h4_->mesh_buffer(), host_slot_lut_h4_.ptr, all_devs, false, BufferRegion(0, l1_lut_h4_->buffer()->size()));
        cq.enqueue_write_shard_to_sub_grid(l1_lut_h3_->mesh_buffer(), host_slot_lut_h3_.ptr, all_devs, false, BufferRegion(0, l1_lut_h3_->buffer()->size()));
        cq.enqueue_write_shard_to_sub_grid(l1_lut_h2_->mesh_buffer(), host_slot_lut_h2_.ptr, all_devs, false, BufferRegion(0, l1_lut_h2_->buffer()->size()));

        run_bcast_decode64_untraced();
        run_bcast_prefill_untraced();
        run_reduce_decode_untraced();
        run_reduce_prefill_untraced();
        cq.finish();

        const char* hot_nm = (hot_qmode_ == kQModeHlut4) ? "HLUT4" : ((hot_qmode_ == kQModeHlut3) ? "HLUT3" : "HLUT2");
        std::printf("[P150-770B] Opened 1x2 Blackhole P150A Mesh (grid=%ux%u=%u cores, FABRIC_1D 4x800G, "
                    "HLUT4=[%d..%d) (%u/L), HLUT3=[%d..%d) (%u/L), HLUT2=[%d..%d) (%u/L), hot_L1=%u %s shards/card [%.2f/330 MiB], trace=%d)\n",
                    unsigned(grid_.x), unsigned(grid_.y), total_cores_,
                    cfg_.hlut4_begin, cfg_.hlut4_end, n_h4_,
                    cfg_.hlut3_begin, cfg_.hlut3_end, n_h3_,
                    cfg_.hlut2_begin, cfg_.hlut2_end, n_h2_,
                    num_hot_shards_, hot_nm,
                    double(pinned_hot_l1_bytes_ * 2ULL) / (1024.0 * 1024.0), int(cfg_.enable_trace));
        std::fflush(stdout);
    }

    ~P150Hy4770bEngine() {
        if (mesh_) {
            mesh_->mesh_command_queue().finish();
            for (auto& [k, tr] : op_traces_) { if (tr.id) mesh_->release_mesh_trace(*tr.id); }
            for (auto& tr : reduce_traces_)  { if (tr.id) mesh_->release_mesh_trace(*tr.id); }
            for (auto& tr : bcast_traces_)   { if (tr.id) mesh_->release_mesh_trace(*tr.id); }
            hot_down_l1_.reset();
            hot_up_l1_.reset();
            hot_gate_l1_.reset();
            arena_.reset();
            dummy_hot_h2_.reset();
            dummy_hot_h3_.reset();
            dummy_hot_h4_.reset();
            l1_lut_h2_.reset();
            l1_lut_h3_.reset();
            l1_lut_h4_.reset();
            slot_meta_dst_.reset();
            slot_meta_src_.reset();
            x_rm_dst_.reset();
            x_rm_src_.reset();
            ws_.reset();
            for (auto& l : layers_) l.reset();
            mesh_->mesh_command_queue().finish();
            ttnn::distributed::close_mesh_device(mesh_);
        }
    }

    uint32_t n_hlut4() const noexcept { return n_h4_; }
    uint32_t n_hlut3() const noexcept { return n_h3_; }
    uint32_t n_hlut2() const noexcept { return n_h2_; }
    uint32_t num_hot_shards() const noexcept { return num_hot_shards_; }
    uint64_t pinned_l1_bytes_total() const noexcept { return pinned_hot_l1_bytes_ * 2ULL; }
    uint64_t gddr6_bytes_per_card() const noexcept { return gddr6_bytes_per_card_; }
    uint64_t stat_total_slots() const noexcept { return stat_total_slots_; }
    uint64_t stat_l1_hits() const noexcept { return stat_l1_hits_; }

    void load_layers(int first_layer, int last_layer) {
        const auto t0 = std::chrono::steady_clock::now();
        const uint32_t hot_page_b = (hot_qmode_ == kQModeHlut4) ? kPageBytesHlut4
                                  : ((hot_qmode_ == kQModeHlut3) ? kPageBytesHlut3 : kPageBytesHlut2);
        const size_t hot_exp_b = size_t(kSbPerProj) * hot_page_b;
        const size_t arena_b   = size_t(num_hot_shards_) * hot_exp_b;
        std::vector<uint8_t> hg0(arena_b, 0u), hg1(arena_b, 0u), hu0(arena_b, 0u), hu1(arena_b, 0u), hd0(arena_b, 0u), hd1(arena_b, 0u);
        std::vector<std::pair<uint16_t, uint16_t>> hot_list;
        hot_list.reserve(num_hot_shards_);

        const int total_layers = last_layer - first_layer + 1;
        uint32_t loaded = 0u;
        for (int l = first_layer; l <= last_layer && !g_stop; ++l) {
            load_one_layer(l, hg0, hg1, hu0, hu1, hd0, hd1, hot_list, total_layers);
            ++loaded;
            if (loaded == 1u || loaded == uint32_t(total_layers) || (loaded % 10u) == 0u) {
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("[P150-770B] Uploaded %u/%d layers (%.2f GiB/card GDDR6) in %.1fs\n",
                            loaded, total_layers, double(gddr6_bytes_per_card_) / 1073741824.0, sec);
                std::fflush(stdout);
            }
        }

        if (num_hot_shards_ > 0u && !hot_list.empty() && hot_gate_l1_.has_value()) {
            write_sharded_pair_blocking(*mesh_, *hot_gate_l1_, hg0.data(), hg1.data());
            write_sharded_pair_blocking(*mesh_, *hot_up_l1_,   hu0.data(), hu1.data());
            write_sharded_pair_blocking(*mesh_, *hot_down_l1_, hd0.data(), hd1.data());

            auto* lut_u16 = static_cast<uint16_t*>(
                (hot_qmode_ == kQModeHlut4) ? host_slot_lut_h4_.ptr
                : ((hot_qmode_ == kQModeHlut3) ? host_slot_lut_h3_.ptr : host_slot_lut_h2_.ptr));
            std::memset(lut_u16, 0xFF, uint32_t(kNumLayers) * 256u * sizeof(uint16_t));
            for (size_t k = 0; k < hot_list.size(); ++k) {
                lut_u16[size_t(hot_list[k].first) * 256u + size_t(hot_list[k].second)] = uint16_t(k);
            }
            const distributed::MeshCoordinateRange all_devs(mesh_->shape());
            auto& lut_t = (hot_qmode_ == kQModeHlut4) ? *l1_lut_h4_
                        : ((hot_qmode_ == kQModeHlut3) ? *l1_lut_h3_ : *l1_lut_h2_);
            mesh_->mesh_command_queue().enqueue_write_shard_to_sub_grid(
                lut_t.mesh_buffer(), lut_u16, all_devs, true, BufferRegion(0, lut_t.buffer()->size()));
            const double pinned_mib = double(hot_list.size() * hot_exp_b * 6ULL) / (1024.0 * 1024.0);
            const char* hot_nm = (hot_qmode_ == kQModeHlut4) ? "HLUT4" : ((hot_qmode_ == kQModeHlut3) ? "HLUT3" : "HLUT2");
            std::printf("[P150-770B] Pinned %zu/%u ultra-hot %s shards in 330 MiB Tensix L1 SRAM (%.2f MiB total)\n",
                        hot_list.size(), num_hot_shards_, hot_nm, pinned_mib);
            std::fflush(stdout);
        }

        if (loaded > 0u && layers_[size_t(first_layer)].has_value()) {
            prewarm_and_capture_traces(*layers_[size_t(first_layer)]);
        }
    }

    // 100% On-Device Blackhole Tensix Forward Pass (x_rot_f32 is ALREADY WHT128-rotated!)
    // Local slot IDs: [0 .. n_h3_) -> HLUT3 bank, [n_h3_ .. n_h3_+n_h2_) -> HLUT2 bank!
    void forward_layer(
        int layer_idx,
        uint32_t rows,
        uint32_t top_k,
        const int32_t row_slots[][kTopK],
        const float row_weights[][kTopK],
        const float* x_rot_f32,
        float* y_out_f32)
    {
        if (layer_idx < 1 || layer_idx >= kNumLayers || !layers_[size_t(layer_idx)].has_value()) {
            throw std::runtime_error("P150Hy4770bEngine: layer not loaded: " + std::to_string(layer_idx));
        }
        const auto& bank = *layers_[size_t(layer_idx)];
        std::memset(y_out_f32, 0, size_t(rows) * kHiddenSize * sizeof(float));
        if (rows == 0u) return;

        // Convert WHT128-rotated FP32 inputs -> BF16 once for both banks
        x_bf16_scratch_.resize(size_t(rows) * kHiddenSize);
        for (uint32_t r = 0u; r < rows; ++r) {
            f32_row_to_bf16_avx2(x_rot_f32 + size_t(r) * kHiddenSize, x_bf16_scratch_.data() + size_t(r) * kHiddenSize);
        }

        auto dispatch_subbank = [&](uint32_t qmode, const QuantBank& qb, int32_t slot_offset) {
            if (qb.n_experts == 0u) return;
            std::vector<std::array<int32_t, kTopK>> sub_slots(rows);
            std::vector<std::array<float, kTopK>>   sub_wts(rows);
            bool any = false;
            for (uint32_t r = 0u; r < rows; ++r) {
                sub_slots[r].fill(-1);
                sub_wts[r].fill(0.0f);
                for (uint32_t s = 0u; s < top_k; ++s) {
                    const int32_t raw = row_slots[r][s];
                    const float w = row_weights[r][s];
                    if (raw >= slot_offset && raw < slot_offset + int32_t(qb.n_experts) && w != 0.0f) {
                        sub_slots[r][s] = raw - slot_offset;
                        sub_wts[r][s] = w;
                        any = true;
                    }
                }
            }
            if (!any) return;
            if (rows <= 32u) {
                run_decode_small(bank, qb, qmode, rows, top_k,
                                 reinterpret_cast<const int32_t(*)[kTopK]>(sub_slots.data()),
                                 reinterpret_cast<const float(*)[kTopK]>(sub_wts.data()),
                                 x_bf16_scratch_.data(), y_out_f32);
            } else {
                run_prefill_compact(bank, qb, qmode, rows, top_k,
                                    reinterpret_cast<const int32_t(*)[kTopK]>(sub_slots.data()),
                                    reinterpret_cast<const float(*)[kTopK]>(sub_wts.data()),
                                    x_bf16_scratch_.data(), y_out_f32);
            }
        };

        if (n_h4_ > 0u) dispatch_subbank(kQModeHlut4, bank.h4, 0);
        if (n_h3_ > 0u) dispatch_subbank(kQModeHlut3, bank.h3, int32_t(n_h4_));
        if (n_h2_ > 0u) dispatch_subbank(kQModeHlut2, bank.h2, int32_t(n_h4_ + n_h3_));
    }

private:
    CoreCoord core_at(uint32_t idx) const noexcept { return CoreCoord{idx % grid_.x, idx / grid_.x}; }

    static std::string layer_tag(int l) {
        char b[16];
        std::snprintf(b, sizeof(b), "%02d", l);
        return b;
    }

    void load_one_layer(
        int l,
        std::vector<uint8_t>& hg0, std::vector<uint8_t>& hg1,
        std::vector<uint8_t>& hu0, std::vector<uint8_t>& hu1,
        std::vector<uint8_t>& hd0, std::vector<uint8_t>& hd1,
        std::vector<std::pair<uint16_t, uint16_t>>& hot_list,
        int total_layers)
    {
        ResidentLayerBank bank;
        bank.layer = uint32_t(l);

        // 0. Load HLUT4 experts [hlut4_begin .. hlut4_end) from hlut4_hot/hlut4_hot_layer_XX.bin (file starts at rank 0!)
        if (n_h4_ > 0u) {
            const std::string path = cfg_.weight_root + "/hlut4_hot/hlut4_hot_layer_" + layer_tag(l) + ".bin";
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs) throw std::runtime_error("Missing P150 HLUT4-hot file: " + path);
            const size_t exp_sb_b = size_t(kSbPerProj) * kPageBytesHlut4;
            std::vector<uint8_t> g0(size_t(n_h4_) * exp_sb_b), g1(size_t(n_h4_) * exp_sb_b);
            std::vector<uint8_t> u0(size_t(n_h4_) * exp_sb_b), u1(size_t(n_h4_) * exp_sb_b);
            std::vector<uint8_t> d0(size_t(n_h4_) * exp_sb_b), d1(size_t(n_h4_) * exp_sb_b);

            const size_t seek_off = size_t(cfg_.hlut4_begin) * size_t(kHlut4ExpertBytes);
            ifs.seekg(std::streamoff(seek_off), std::ios::beg);
            std::vector<Hlut4Tile32x32> raw_exp(3u * kProjTiles);
            for (uint32_t e = 0u; e < n_h4_; ++e) {
                if (!ifs.read(reinterpret_cast<char*>(raw_exp.data()), std::streamsize(kHlut4ExpertBytes))) {
                    throw std::runtime_error("Short read on " + path);
                }
                pack_hlut4_expert_tp2(raw_exp.data() + 0u * kProjTiles, false, g0.data() + size_t(e) * exp_sb_b, g1.data() + size_t(e) * exp_sb_b);
                pack_hlut4_expert_tp2(raw_exp.data() + 1u * kProjTiles, false, u0.data() + size_t(e) * exp_sb_b, u1.data() + size_t(e) * exp_sb_b);
                pack_hlut4_expert_tp2(raw_exp.data() + 2u * kProjTiles, true,  d0.data() + size_t(e) * exp_sb_b, d1.data() + size_t(e) * exp_sb_b);
            }
            ifs.close();
            if (int fd = ::open(path.c_str(), O_RDONLY); fd >= 0) { (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
            bank.h4.n_experts  = n_h4_;
            bank.h4.gate_pages = alloc_dram_sharded0(*mesh_, Shape({1, n_h4_ * kSbPerProj, kPageBytesHlut4 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);
            bank.h4.up_pages   = alloc_dram_sharded0(*mesh_, Shape({1, n_h4_ * kSbPerProj, kPageBytesHlut4 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);
            bank.h4.down_pages = alloc_dram_sharded0(*mesh_, Shape({1, n_h4_ * kSbPerProj, kPageBytesHlut4 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);
            write_sharded_pair_blocking(*mesh_, bank.h4.gate_pages, g0.data(), g1.data());
            write_sharded_pair_blocking(*mesh_, bank.h4.up_pages,   u0.data(), u1.data());
            write_sharded_pair_blocking(*mesh_, bank.h4.down_pages, d0.data(), d1.data());
            gddr6_bytes_per_card_ += 3ULL * size_t(n_h4_) * exp_sb_b;

            if (hot_qmode_ == kQModeHlut4 && num_hot_shards_ > 0u && hot_list.size() < num_hot_shards_) {
                const uint32_t stride = std::max<uint32_t>(1u, uint32_t(total_layers) / std::max<uint32_t>(1u, num_hot_shards_));
                if (total_layers == 1 || ((l - 1) % int(stride) == 0)) {
                    const uint32_t take = (total_layers == 1) ? std::min<uint32_t>(n_h4_, num_hot_shards_) : 1u;
                    for (uint32_t e = 0u; e < take && hot_list.size() < num_hot_shards_; ++e) {
                        const size_t slot = hot_list.size();
                        std::memcpy(hg0.data() + slot * exp_sb_b, g0.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hg1.data() + slot * exp_sb_b, g1.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hu0.data() + slot * exp_sb_b, u0.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hu1.data() + slot * exp_sb_b, u1.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hd0.data() + slot * exp_sb_b, d0.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hd1.data() + slot * exp_sb_b, d1.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        bank.h4.hot_slot_for_eid[e] = uint16_t(slot);
                        hot_list.emplace_back(uint16_t(l), uint16_t(e));
                    }
                }
            }
        }

        // 1. Load HLUT3 experts [hlut3_begin .. hlut3_end) from hlut3_layer_XX.bin (file starts at rank 0)
        if (n_h3_ > 0u) {
            const std::string path = cfg_.weight_root + "/hlut3/hlut3_layer_" + layer_tag(l) + ".bin";
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs) throw std::runtime_error("Missing P150 HLUT3 file: " + path);
            const size_t exp_sb_b = size_t(kSbPerProj) * kPageBytesHlut3;
            std::vector<uint8_t> g0(size_t(n_h3_) * exp_sb_b), g1(size_t(n_h3_) * exp_sb_b);
            std::vector<uint8_t> u0(size_t(n_h3_) * exp_sb_b), u1(size_t(n_h3_) * exp_sb_b);
            std::vector<uint8_t> d0(size_t(n_h3_) * exp_sb_b), d1(size_t(n_h3_) * exp_sb_b);

            const size_t seek_off = size_t(cfg_.hlut3_begin) * size_t(kHlut3ExpertBytes);
            ifs.seekg(std::streamoff(seek_off), std::ios::beg);
            std::vector<Hlut3Tile32x32> raw_exp(3u * kProjTiles);
            for (uint32_t e = 0u; e < n_h3_; ++e) {
                if (!ifs.read(reinterpret_cast<char*>(raw_exp.data()), std::streamsize(kHlut3ExpertBytes))) {
                    throw std::runtime_error("Short read on " + path);
                }
                pack_hlut3_expert_tp2(raw_exp.data() + 0u * kProjTiles, false, g0.data() + size_t(e) * exp_sb_b, g1.data() + size_t(e) * exp_sb_b);
                pack_hlut3_expert_tp2(raw_exp.data() + 1u * kProjTiles, false, u0.data() + size_t(e) * exp_sb_b, u1.data() + size_t(e) * exp_sb_b);
                pack_hlut3_expert_tp2(raw_exp.data() + 2u * kProjTiles, true,  d0.data() + size_t(e) * exp_sb_b, d1.data() + size_t(e) * exp_sb_b);
            }
            ifs.close();
            if (int fd = ::open(path.c_str(), O_RDONLY); fd >= 0) { (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
            bank.h3.n_experts  = n_h3_;
            bank.h3.gate_pages = alloc_dram_sharded0(*mesh_, Shape({1, n_h3_ * kSbPerProj, kPageBytesHlut3 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);
            bank.h3.up_pages   = alloc_dram_sharded0(*mesh_, Shape({1, n_h3_ * kSbPerProj, kPageBytesHlut3 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);
            bank.h3.down_pages = alloc_dram_sharded0(*mesh_, Shape({1, n_h3_ * kSbPerProj, kPageBytesHlut3 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);
            write_sharded_pair_blocking(*mesh_, bank.h3.gate_pages, g0.data(), g1.data());
            write_sharded_pair_blocking(*mesh_, bank.h3.up_pages,   u0.data(), u1.data());
            write_sharded_pair_blocking(*mesh_, bank.h3.down_pages, d0.data(), d1.data());
            gddr6_bytes_per_card_ += 3ULL * size_t(n_h3_) * exp_sb_b;

            if (hot_qmode_ == kQModeHlut3 && num_hot_shards_ > 0u && hot_list.size() < num_hot_shards_) {
                // Pin earliest/hottest experts in early & evenly spaced layers (including L1 for self-test!)
                const uint32_t stride = std::max<uint32_t>(1u, uint32_t(total_layers) / std::max<uint32_t>(1u, num_hot_shards_));
                if (total_layers == 1 || ((l - 1) % int(stride) == 0)) {
                    const uint32_t take = (total_layers == 1) ? std::min<uint32_t>(n_h3_, num_hot_shards_) : 1u;
                    for (uint32_t e = 0u; e < take && hot_list.size() < num_hot_shards_; ++e) {
                        const size_t slot = hot_list.size();
                        std::memcpy(hg0.data() + slot * exp_sb_b, g0.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hg1.data() + slot * exp_sb_b, g1.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hu0.data() + slot * exp_sb_b, u0.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hu1.data() + slot * exp_sb_b, u1.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hd0.data() + slot * exp_sb_b, d0.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hd1.data() + slot * exp_sb_b, d1.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        bank.h3.hot_slot_for_eid[e] = uint16_t(slot);
                        hot_list.emplace_back(uint16_t(l), uint16_t(e));
                    }
                }
            }
        }

        // 2. Load HLUT2 experts [hlut2_begin .. hlut2_end) from hlut2_layer_XX.bin (file starts at rank 60)
        if (n_h2_ > 0u) {
            const std::string path = cfg_.weight_root + "/hlut2/hlut2_layer_" + layer_tag(l) + ".bin";
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs) throw std::runtime_error("Missing P150 HLUT2 file: " + path);
            const size_t exp_sb_b = size_t(kSbPerProj) * kPageBytesHlut2;
            std::vector<uint8_t> g0(size_t(n_h2_) * exp_sb_b), g1(size_t(n_h2_) * exp_sb_b);
            std::vector<uint8_t> u0(size_t(n_h2_) * exp_sb_b), u1(size_t(n_h2_) * exp_sb_b);
            std::vector<uint8_t> d0(size_t(n_h2_) * exp_sb_b), d1(size_t(n_h2_) * exp_sb_b);

            const size_t seek_off = size_t(cfg_.hlut2_begin - kPackHlut2Begin) * size_t(kHlut2ExpertBytes);
            ifs.seekg(std::streamoff(seek_off), std::ios::beg);
            std::vector<Hlut2Tile32x32> raw_exp(3u * kProjTiles);
            for (uint32_t e = 0u; e < n_h2_; ++e) {
                if (!ifs.read(reinterpret_cast<char*>(raw_exp.data()), std::streamsize(kHlut2ExpertBytes))) {
                    throw std::runtime_error("Short read on " + path);
                }
                pack_hlut2_expert_tp2(raw_exp.data() + 0u * kProjTiles, false, g0.data() + size_t(e) * exp_sb_b, g1.data() + size_t(e) * exp_sb_b);
                pack_hlut2_expert_tp2(raw_exp.data() + 1u * kProjTiles, false, u0.data() + size_t(e) * exp_sb_b, u1.data() + size_t(e) * exp_sb_b);
                pack_hlut2_expert_tp2(raw_exp.data() + 2u * kProjTiles, true,  d0.data() + size_t(e) * exp_sb_b, d1.data() + size_t(e) * exp_sb_b);
            }
            ifs.close();
            if (int fd = ::open(path.c_str(), O_RDONLY); fd >= 0) { (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
            bank.h2.n_experts  = n_h2_;
            bank.h2.gate_pages = alloc_dram_sharded0(*mesh_, Shape({1, n_h2_ * kSbPerProj, kPageBytesHlut2 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);
            bank.h2.up_pages   = alloc_dram_sharded0(*mesh_, Shape({1, n_h2_ * kSbPerProj, kPageBytesHlut2 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);
            bank.h2.down_pages = alloc_dram_sharded0(*mesh_, Shape({1, n_h2_ * kSbPerProj, kPageBytesHlut2 / 4u}), DataType::UINT32, Layout::ROW_MAJOR);
            write_sharded_pair_blocking(*mesh_, bank.h2.gate_pages, g0.data(), g1.data());
            write_sharded_pair_blocking(*mesh_, bank.h2.up_pages,   u0.data(), u1.data());
            write_sharded_pair_blocking(*mesh_, bank.h2.down_pages, d0.data(), d1.data());
            gddr6_bytes_per_card_ += 3ULL * size_t(n_h2_) * exp_sb_b;

            if (hot_qmode_ == kQModeHlut2 && num_hot_shards_ > 0u && hot_list.size() < num_hot_shards_) {
                const uint32_t stride = std::max<uint32_t>(1u, uint32_t(total_layers) / std::max<uint32_t>(1u, num_hot_shards_));
                if (total_layers == 1 || ((l - 1) % int(stride) == 0)) {
                    const uint32_t take = (total_layers == 1) ? std::min<uint32_t>(n_h2_, num_hot_shards_) : 1u;
                    for (uint32_t e = 0u; e < take && hot_list.size() < num_hot_shards_; ++e) {
                        const size_t slot = hot_list.size();
                        std::memcpy(hg0.data() + slot * exp_sb_b, g0.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hg1.data() + slot * exp_sb_b, g1.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hu0.data() + slot * exp_sb_b, u0.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hu1.data() + slot * exp_sb_b, u1.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hd0.data() + slot * exp_sb_b, d0.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        std::memcpy(hd1.data() + slot * exp_sb_b, d1.data() + size_t(e) * exp_sb_b, exp_sb_b);
                        bank.h2.hot_slot_for_eid[e] = uint16_t(slot);
                        hot_list.emplace_back(uint16_t(l), uint16_t(e));
                    }
                }
            }
        }

        // 3. Build deterministic per-card tilized WHT128 Q_i [1, 8, 128, 128] BFLOAT16 TILE (I_shard = 1024 = 8 blocks)
        const auto signs_i = make_signs_i(l);
        std::array<std::vector<uint16_t>, 2> wht_tiles;
        for (int tp = 0; tp < 2; ++tp) {
            wht_tiles[size_t(tp)].assign(8u * 16u * 1024u, 0u);
            const uint64_t* si_tp = signs_i.data() + size_t(tp) * 16u;  // 1024 / 64 = 16 uint64s per TP2 card
            float q_mat[128][128];
            for (int nb4 = 0; nb4 < 8; ++nb4) {
                build_wht128_block_matrix_local(nb4, si_tp, q_mat);
                for (int kt = 0; kt < 4; ++kt) {
                    for (int nt = 0; nt < 4; ++nt) {
                        uint16_t* dst_t = wht_tiles[size_t(tp)].data() + size_t(nb4 * 16 + kt * 4 + nt) * 1024u;
                        for (int r = 0; r < 32; ++r) {
                            for (int c = 0; c < 32; ++c) {
                                dst_t[tensix_tile_offset(r, c)] = f32_to_bf16_rne(q_mat[kt * 32 + r][nt * 32 + c]);
                            }
                        }
                    }
                }
            }
        }
        bank.wht_q_i = alloc_dram_sharded0(*mesh_, Shape({1, 8, 128, 128}), DataType::BFLOAT16, Layout::TILE);
        write_sharded_pair_blocking(*mesh_, bank.wht_q_i, wht_tiles[0].data(), wht_tiles[1].data());

        layers_[size_t(l)].emplace(std::move(bank));
    }

    void write_layer_desc(const ResidentLayerBank& bank, const QuantBank& qb, uint32_t qmode) {
        auto* ldesc = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(host_slot_ids_.ptr) + (kSlotIdsBytes - 64u));
        const bool use_l1 = (qmode == hot_qmode_) && hot_gate_l1_.has_value();
        const ttnn::Tensor& dummy = (qmode == kQModeHlut4) ? *dummy_hot_h4_
                                  : ((qmode == kQModeHlut3) ? *dummy_hot_h3_ : *dummy_hot_h2_);
        const ttnn::Tensor& lut_t = (qmode == kQModeHlut4) ? *l1_lut_h4_
                                  : ((qmode == kQModeHlut3) ? *l1_lut_h3_ : *l1_lut_h2_);
        ldesc[0] = qb.gate_pages.buffer()->address();
        ldesc[1] = qb.up_pages.buffer()->address();
        ldesc[2] = qb.down_pages.buffer()->address();
        ldesc[3] = 0u;
        ldesc[4] = bank.wht_q_i.buffer()->address();
        ldesc[5] = use_l1 ? hot_gate_l1_->buffer()->address() : dummy.buffer()->address();
        ldesc[6] = use_l1 ? hot_up_l1_->buffer()->address()   : dummy.buffer()->address();
        ldesc[7] = use_l1 ? hot_down_l1_->buffer()->address() : dummy.buffer()->address();
        ldesc[8] = lut_t.buffer()->address();
        ldesc[9] = bank.layer;
        for (uint32_t k = 10u; k < 16u; ++k) ldesc[k] = 0u;
    }

    void run_bcast_decode64_untraced() {
        auto k0 = ws_->x_decode; auto k1 = ws_->slot_ids; auto k2 = ws_->weights_decode64;
        (void)k0; (void)k1; (void)k2;
        ws_->x_decode = ttnn::broadcast(ws_->x_decode_src, distributed::MeshCoordinate{0, 0}, std::nullopt,
                                        ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
        ws_->slot_ids = ttnn::broadcast(ws_->slot_ids_src, distributed::MeshCoordinate{0, 0}, 1,
                                        ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
        ws_->weights_decode64 = ttnn::broadcast(ws_->weights_decode64_src, distributed::MeshCoordinate{0, 0}, std::nullopt,
                                                ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
    }

    void run_bcast_prefill_untraced() {
        auto k0 = ws_->weights_tiled; auto k1 = slot_meta_dst_; auto k2 = x_rm_dst_;
        (void)k0; (void)k1; (void)k2;
        ws_->weights_tiled = ttnn::broadcast(ws_->weights_tiled_src, distributed::MeshCoordinate{0, 0}, std::nullopt,
                                             ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
        slot_meta_dst_ = ttnn::broadcast(*slot_meta_src_, distributed::MeshCoordinate{0, 0}, std::nullopt,
                                         ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
        x_rm_dst_ = ttnn::broadcast(*x_rm_src_, distributed::MeshCoordinate{0, 0}, std::nullopt,
                                    ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
    }

    void run_reduce_decode_untraced() {
        auto k = ws_->finish_decode; (void)k;
        ws_->finish_decode[0] = ttnn::typecast(ws_->combined_f32, DataType::BFLOAT16, ttnn::DRAM_MEMORY_CONFIG);
        ws_->finish_decode[1] = ttnn::all_reduce(
            ws_->finish_decode[0], 1, std::nullopt, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, tt::tt_fabric::Topology::Linear);
        ws_->finish_decode[2] = ttnn::untilize(ws_->finish_decode[1], ttnn::DRAM_MEMORY_CONFIG);
    }

    void run_reduce_prefill_untraced() {
        auto k = arena_->finish_tensors; (void)k;
        arena_->finish_tensors[0] = ttnn::typecast(arena_->y_accum_f32, DataType::BFLOAT16, ttnn::DRAM_MEMORY_CONFIG);
        arena_->finish_tensors[1] = ttnn::all_reduce(
            arena_->finish_tensors[0], 1, std::nullopt, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, tt::tt_fabric::Topology::Linear);
    }

    void enqueue_projection(
        const QuantBank& qb,
        uint32_t qmode,
        const ttnn::Tensor& act_l1,
        const ttnn::Tensor& pages_dram,
        ttnn::Tensor& partials_l1,
        uint32_t slot_start,
        uint32_t batch_cnt,
        uint32_t act_slot_stride,
        uint32_t proj_idx)
    {
        const bool is_down     = (proj_idx == 2u);
        const bool is_fused_gu = (proj_idx == 3u);
        const uint32_t num_proj = is_fused_gu ? 2u : 1u;
        const uint32_t keep_act = (act_slot_stride == 0u) ? 1u : 0u;
        const uint32_t nt4      = is_down ? 48u : 8u;
        const uint32_t kt4      = is_down ? 8u  : 48u;
        const uint32_t k_splits = is_down ? kKSplitsDown : kKSplitsUp;  // 2 or 12
        const uint32_t nt_total = nt4 * 4u;                             // 192 or 32
        const uint32_t kt_total = kt4 * 4u;                             // 32 or 192
        const uint32_t kt_core  = kNumSbPerCore * 4u;                   // 16

        const bool use_l1 = (qmode == hot_qmode_) && hot_gate_l1_.has_value();
        const ttnn::Tensor& dummy = (qmode == kQModeHlut4) ? *dummy_hot_h4_
                                  : ((qmode == kQModeHlut3) ? *dummy_hot_h3_ : *dummy_hot_h2_);
        const ttnn::Tensor& lut_t = (qmode == kQModeHlut4) ? *l1_lut_h4_
                                  : ((qmode == kQModeHlut3) ? *l1_lut_h3_ : *l1_lut_h2_);
        const ttnn::Tensor& hot_t = use_l1 ? (is_down ? *hot_down_l1_ : *hot_gate_l1_) : dummy;
        const ttnn::Tensor& partials0 = is_fused_gu ? ws_->gate_partials : partials_l1;

        std::vector<CoreCoord> all_coords;
        all_coords.reserve(96u);
        uint32_t ci = 0u;
        for (uint32_t nb4 = 0u; nb4 < nt4; ++nb4) {
            for (uint32_t ks = 0u; ks < k_splits; ++ks) {
                all_coords.push_back(core_at(ci++));
            }
        }
        const CoreRangeSet all_cores = make_core_set(all_coords);

        auto reader = make_kernel(
            cfg_.kernels_dir + "/hy4_p150_reader_a.cpp",
            all_cores,
            ReaderConfigDescriptor{},
            make_accessor_args({&act_l1}));
        auto writer = make_kernel(
            cfg_.kernels_dir + "/hy4_p150_writer_b.cpp",
            all_cores,
            WriterConfigDescriptor{},
            make_accessor_args({&pages_dram, &ws_->slot_ids, &partials0, &hot_t, &lut_t}, {qmode}));

        ComputeConfigDescriptor cfg;
        cfg.math_fidelity = MathFidelity::HiFi4;
        cfg.math_approx_mode = false;
        cfg.fp32_dest_acc_en = true;
        cfg.dst_full_sync_en = true;
        cfg.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        cfg.unpack_to_dest_mode[4] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[5] = UnpackToDestMode::UnpackToDestFp32;
        auto compute = make_kernel(cfg_.kernels_dir + "/hy4_p150_compute.cpp", all_cores, cfg, {qmode});

        ci = 0u;
        for (uint32_t nb4 = 0u; nb4 < nt4; ++nb4) {
            for (uint32_t ks = 0u; ks < k_splits; ++ks) {
                const uint32_t kb4_start = ks * kNumSbPerCore;
                const CoreCoord& c = all_coords[ci++];
                reader.runtime_args.emplace_back(
                    c, std::vector<uint32_t>{act_l1.buffer()->address(), kt_total, kb4_start * 4u, kt_core,
                                             batch_cnt, act_slot_stride, slot_start, num_proj});
                writer.runtime_args.emplace_back(
                    c, std::vector<uint32_t>{ws_->slot_ids.buffer()->address(), partials0.buffer()->address(),
                                             slot_start, proj_idx, nb4, kt4, kb4_start, kNumSbPerCore, ks,
                                             nt_total, ws_->m_tiles_ws, kSlotIdsBytes, batch_cnt,
                                             partials_l1.buffer()->address()});
                compute.runtime_args.emplace_back(
                    c, std::vector<uint32_t>{kNumSbPerCore, batch_cnt, num_proj, keep_act, slot_start});
            }
        }

        const uint32_t cb0_pages = (keep_act != 0u || is_down) ? kt_core : (2u * kt_core);  // 16 or 32
        const uint32_t cb4_pages = (qmode == kQModeHlut2) ? 1u : 2u;

        ProgramDescriptor prog;
        prog.kernels = {std::move(reader), std::move(writer), std::move(compute)};
        prog.cbs = {
            make_cb(0,  cb0_pages,           2048u, tt::DataFormat::Float16_b, all_cores),
            make_cb(1,  kNumSbPerCore * 16u, 2048u, tt::DataFormat::Float16_b, all_cores),
            make_cb(4,  cb4_pages,           4096u, tt::DataFormat::UInt32,    all_cores),
            make_cb(5,  2u,                  4096u, tt::DataFormat::UInt32,    all_cores),
            make_cb(6,  1u,                  4608u, tt::DataFormat::UInt32,    all_cores),
            make_cb(16, 8u,                  4096u, tt::DataFormat::Float32,   all_cores),
        };

        if (is_fused_gu) {
            const ttnn::Tensor& hot_up = use_l1 ? *hot_up_l1_ : dummy;
            run_generic_preserving_topology(
                {act_l1, ws_->slot_ids, pages_dram, qb.up_pages, hot_t, hot_up, lut_t, ws_->gate_partials, partials_l1}, prog);
        } else {
            run_generic_preserving_topology(
                {act_l1, ws_->slot_ids, pages_dram, hot_t, lut_t, partials_l1}, prog);
        }
    }

    void enqueue_swiglu_wht128(const ResidentLayerBank& bank, uint32_t batch_cnt) {
        const uint32_t max_groups = total_cores_ / 8u;
        const uint32_t m_per_grp  = (batch_cnt + max_groups - 1u) / max_groups;
        const uint32_t ngroups    = (batch_cnt + m_per_grp - 1u) / m_per_grp;
        const uint32_t ncores     = 8u * ngroups;
        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0u; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto reader = make_kernel(
            cfg_.kernels_dir + "/hy4_p150_swiglu_wht128_reader.cpp",
            cores,
            ReaderConfigDescriptor{},
            make_accessor_args({&ws_->gate_partials, &ws_->up_partials, &bank.wht_q_i, &ws_->slot_ids}));
        auto writer = make_kernel(
            cfg_.kernels_dir + "/hy4_p150_swiglu_wht128_writer.cpp",
            cores,
            WriterConfigDescriptor{},
            make_accessor_args({&ws_->z_rot_bf16}));

        ComputeConfigDescriptor cfg;
        cfg.math_fidelity = MathFidelity::HiFi4;
        cfg.math_approx_mode = false;
        cfg.fp32_dest_acc_en = true;
        cfg.dst_full_sync_en = false;
        cfg.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        cfg.unpack_to_dest_mode[0] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[1] = UnpackToDestMode::UnpackToDestFp32;
        auto compute = make_kernel(cfg_.kernels_dir + "/hy4_p150_swiglu_wht128_compute.cpp", cores, cfg);

        for (uint32_t g = 0u; g < ngroups; ++g) {
            const uint32_t m_start = g * m_per_grp;
            const uint32_t m_cnt   = std::min<uint32_t>(m_per_grp, batch_cnt - m_start);
            for (uint32_t nb4 = 0u; nb4 < 8u; ++nb4) {
                const CoreCoord& c = coords[g * 8u + nb4];
                reader.runtime_args.emplace_back(
                    c, std::vector<uint32_t>{ws_->gate_partials.buffer()->address(), ws_->up_partials.buffer()->address(),
                                             ws_->slot_ids.buffer()->address(), nb4, kKSplitsUp, ws_->m_tiles_ws,
                                             m_start, m_cnt, kSlotIdsBytes, 1u});
                writer.runtime_args.emplace_back(
                    c, std::vector<uint32_t>{ws_->z_rot_bf16.buffer()->address(), nb4, m_start, m_cnt});
                compute.runtime_args.emplace_back(
                    c, std::vector<uint32_t>{kKSplitsUp, m_cnt, 1u});
            }
        }

        ProgramDescriptor prog;
        prog.kernels = {std::move(reader), std::move(writer), std::move(compute)};
        prog.cbs = {
            make_cb(0,  2u,  4096u, tt::DataFormat::Float32,   cores),
            make_cb(1,  2u,  4096u, tt::DataFormat::Float32,   cores),
            make_cb(2,  16u, 2048u, tt::DataFormat::Float16_b, cores),
            make_cb(24, 4u,  2048u, tt::DataFormat::Float16_b, cores),
            make_cb(16, 8u,  2048u, tt::DataFormat::Float16_b, cores),
        };
        run_generic_preserving_topology(
            {ws_->gate_partials, ws_->up_partials, bank.wht_q_i, ws_->slot_ids, ws_->z_rot_bf16}, prog);
    }

    void enqueue_ksplit_combine_decode(uint32_t slot_start, uint32_t batch_cnt, bool use_weights64) {
        constexpr uint32_t ncores = 96u;  // 192 tiles / 96 cores = 2 tiles/core
        const uint32_t acc_prev = (slot_start > 0u) ? 1u : 0u;
        const ttnn::Tensor& w_tensor = use_weights64 ? ws_->weights_decode64 : ws_->weights_tiled;

        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0u; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto reader = make_kernel(
            cfg_.kernels_dir + "/hy4_p150_ksplit_combine_reader.cpp",
            cores,
            ReaderConfigDescriptor{},
            make_accessor_args({&ws_->down_partials, &w_tensor, &ws_->combined_f32}));
        auto writer = make_kernel(
            cfg_.kernels_dir + "/hy4_p150_ksplit_combine_writer.cpp",
            cores,
            WriterConfigDescriptor{},
            make_accessor_args({&ws_->combined_f32}));

        ComputeConfigDescriptor cfg;
        cfg.math_fidelity = MathFidelity::HiFi4;
        cfg.math_approx_mode = false;
        cfg.fp32_dest_acc_en = true;
        cfg.dst_full_sync_en = false;
        cfg.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        cfg.unpack_to_dest_mode[0] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[1] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[2] = UnpackToDestMode::UnpackToDestFp32;
        auto compute = make_kernel(cfg_.kernels_dir + "/hy4_p150_ksplit_combine_compute.cpp", cores, cfg);

        for (uint32_t i = 0u; i < ncores; ++i) {
            reader.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{ws_->down_partials.buffer()->address(), w_tensor.buffer()->address(),
                                                 ws_->combined_f32.buffer()->address(), slot_start, kKSplitsDown,
                                                 ws_->m_tiles_ws, 2u * i, 2u, acc_prev, batch_cnt});
            writer.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{ws_->combined_f32.buffer()->address(), 2u * i, 2u});
            compute.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{kKSplitsDown, 2u, acc_prev, batch_cnt});
        }

        ProgramDescriptor prog;
        prog.kernels = {std::move(reader), std::move(writer), std::move(compute)};
        prog.cbs = {
            make_cb(0,  2u * kKSplitsDown, 4096u, tt::DataFormat::Float32, cores),
            make_cb(1,  2u,                4096u, tt::DataFormat::Float32, cores),
            make_cb(2,  2u,                4096u, tt::DataFormat::Float32, cores),
            make_cb(16, 2u,                4096u, tt::DataFormat::Float32, cores),
        };
        run_generic_preserving_topology({ws_->down_partials, w_tensor, ws_->combined_f32}, prog);
    }

    void enqueue_zero_pages(ttnn::Tensor& y_accum, uint32_t total_pages) {
        const uint32_t ncores = total_cores_;
        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0u; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto writer = make_kernel(
            cfg_.kernels_dir + "/hy4_p150_zero_pages_writer.cpp",
            cores,
            WriterConfigDescriptor{},
            make_accessor_args({&y_accum}));
        uint32_t cur = 0u;
        for (uint32_t i = 0u; i < ncores; ++i) {
            const uint32_t cnt = (total_pages / ncores) + (i < (total_pages % ncores) ? 1u : 0u);
            writer.runtime_args.emplace_back(coords[i], std::vector<uint32_t>{y_accum.buffer()->address(), cur, cnt});
            cur += cnt;
        }
        ProgramDescriptor prog;
        prog.kernels = {std::move(writer)};
        prog.cbs = {make_cb(0, 1u, 4096u, tt::DataFormat::UInt32, cores)};
        run_generic_preserving_topology({*dummy_hot_h3_, y_accum}, prog);
    }

    void enqueue_compact_gather_tilize(ttnn::Tensor& x_rm_buf, uint32_t slot_start, uint32_t batch_cnt) {
        constexpr uint32_t ncores = 96u;  // 192 tiles / 2 tiles per core
        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0u; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto writer = make_kernel(
            cfg_.kernels_dir + "/hy4_p150_compact_gather_tilize.cpp",
            cores,
            WriterConfigDescriptor{},
            make_accessor_args({&x_rm_buf, &(*slot_meta_dst_), &ws_->x_tilized}));
        for (uint32_t i = 0u; i < ncores; ++i) {
            writer.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{x_rm_buf.buffer()->address(), slot_meta_dst_->buffer()->address(),
                                                 ws_->x_tilized.buffer()->address(), slot_start, i, batch_cnt});
        }
        ProgramDescriptor prog;
        prog.kernels = {std::move(writer)};
        prog.cbs = {make_cb(0, 1u, 10240u, tt::DataFormat::UInt32, cores)};
        run_generic_preserving_topology({x_rm_buf, *slot_meta_dst_, ws_->x_tilized}, prog);
    }

    void enqueue_compact_scatter_combine(ttnn::Tensor& y_accum, uint32_t slot_start, uint32_t batch_cnt) {
        constexpr uint32_t ncores = 96u;  // 192 tiles / 2 tiles per core
        std::vector<CoreCoord> coords;
        coords.reserve(ncores);
        for (uint32_t i = 0u; i < ncores; ++i) coords.push_back(core_at(i));
        const CoreRangeSet cores = make_core_set(coords);

        auto reader = make_kernel(
            cfg_.kernels_dir + "/hy4_p150_compact_scatter_reader.cpp",
            cores,
            ReaderConfigDescriptor{},
            make_accessor_args({&ws_->down_partials, &ws_->weights_tiled, &y_accum, &(*slot_meta_dst_)}));
        auto writer = make_kernel(
            cfg_.kernels_dir + "/hy4_p150_compact_scatter_writer.cpp",
            cores,
            WriterConfigDescriptor{},
            make_accessor_args({&y_accum, &(*slot_meta_dst_)}));

        ComputeConfigDescriptor cfg;
        cfg.math_fidelity = MathFidelity::HiFi4;
        cfg.math_approx_mode = false;
        cfg.fp32_dest_acc_en = true;
        cfg.dst_full_sync_en = false;
        cfg.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        cfg.unpack_to_dest_mode[0] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[1] = UnpackToDestMode::UnpackToDestFp32;
        cfg.unpack_to_dest_mode[2] = UnpackToDestMode::UnpackToDestFp32;
        auto compute = make_kernel(cfg_.kernels_dir + "/hy4_p150_ksplit_combine_compute.cpp", cores, cfg);

        for (uint32_t i = 0u; i < ncores; ++i) {
            reader.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{ws_->down_partials.buffer()->address(), ws_->weights_tiled.buffer()->address(),
                                                 y_accum.buffer()->address(), slot_meta_dst_->buffer()->address(),
                                                 slot_start, kKSplitsDown, i, ws_->m_tiles_ws, batch_cnt});
            writer.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{y_accum.buffer()->address(), slot_meta_dst_->buffer()->address(),
                                                 slot_start, i, batch_cnt});
            compute.runtime_args.emplace_back(
                coords[i], std::vector<uint32_t>{kKSplitsDown, 2u * batch_cnt, 1u, 1u});
        }

        ProgramDescriptor prog;
        prog.kernels = {std::move(reader), std::move(writer), std::move(compute)};
        prog.cbs = {
            make_cb(0,  2u * kKSplitsDown, 4096u, tt::DataFormat::Float32, cores),
            make_cb(1,  2u,                4096u, tt::DataFormat::Float32, cores),
            make_cb(2,  2u,                4096u, tt::DataFormat::Float32, cores),
            make_cb(5,  1u,                8448u, tt::DataFormat::UInt32,  cores),
            make_cb(6,  1u,                8448u, tt::DataFormat::UInt32,  cores),
            make_cb(7,  kBatchSlots,       64u,   tt::DataFormat::UInt32,  cores),
            make_cb(16, 2u,                4096u, tt::DataFormat::Float32, cores),
        };
        run_generic_preserving_topology({ws_->down_partials, ws_->weights_tiled, *slot_meta_dst_, y_accum}, prog);
    }

    void prewarm_and_capture_traces(const ResidentLayerBank& bank) {
        if (programs_prewarmed_) return;
        programs_prewarmed_ = true;

        auto* slot_u16 = static_cast<uint16_t*>(host_slot_ids_.ptr);
        for (uint32_t i = 0u; i < kMaxCompactSlots; ++i) slot_u16[i] = (i == 0u) ? 0u : 0xFFFFu;
        std::memset(host_slot_meta_.ptr, 0, kMaxCompactSlots * 128u);
        std::memset(host_weights_.ptr, 0, kBatchSlots * 4096u);
        std::memset(host_x_tilized_.ptr, 0, kHiddenTiles * 2048u);

        auto& cq = mesh_->mesh_command_queue();
        const distributed::MeshCoordinateRange all_devs(mesh_->shape());
        cq.enqueue_write_shard_to_sub_grid(slot_meta_dst_->mesh_buffer(), host_slot_meta_.ptr, all_devs, false, BufferRegion(0, kBatchSlots * 128u));
        cq.enqueue_write_shard_to_sub_grid(ws_->weights_decode64.mesh_buffer(), host_weights_.ptr, all_devs, false, BufferRegion(0, kBatchSlots * 4096u));
        cq.enqueue_write_shard_to_sub_grid(ws_->weights_tiled.mesh_buffer(), host_weights_.ptr, all_devs, false, BufferRegion(0, kBatchSlots * 4096u));
        cq.enqueue_write_shard_to_sub_grid(ws_->x_decode.mesh_buffer(), host_x_tilized_.ptr, all_devs, false, BufferRegion(0, kHiddenTiles * 2048u));
        cq.enqueue_write_shard_to_sub_grid(ws_->x_tilized.mesh_buffer(), host_x_tilized_.ptr, all_devs, false, BufferRegion(0, kHiddenTiles * 2048u));

        auto warm_bank = [&](const QuantBank& qb, uint32_t qmode) {
            if (qb.n_experts == 0u) return;
            write_layer_desc(bank, qb, qmode);
            cq.enqueue_write_shard_to_sub_grid(ws_->slot_ids.mesh_buffer(), host_slot_ids_.ptr, all_devs, false, BufferRegion(0, kSlotIdsBytes));
            for (uint32_t bcnt : {2u, 4u, 8u, kBatchSlots}) {
                enqueue_projection(qb, qmode, ws_->x_decode, qb.gate_pages, ws_->up_partials, 0u, bcnt, 0u, 3u);
                enqueue_swiglu_wht128(bank, bcnt);
                enqueue_projection(qb, qmode, ws_->z_rot_bf16, qb.down_pages, ws_->down_partials, 0u, bcnt, kInterShardTiles, 2u);
                enqueue_ksplit_combine_decode(0u, bcnt, true);
            }
            enqueue_projection(qb, qmode, ws_->x_tilized, qb.gate_pages, ws_->up_partials, 0u, kBatchSlots, kHiddenTiles, 3u);
        };
        warm_bank(bank.h4, kQModeHlut4);
        warm_bank(bank.h3, kQModeHlut3);
        warm_bank(bank.h2, kQModeHlut2);
        enqueue_zero_pages(arena_->y_accum_f32, kPrefillMaxRows * 6u);
        enqueue_compact_gather_tilize(*x_rm_dst_, 0u, kBatchSlots);
        enqueue_compact_scatter_combine(arena_->y_accum_f32, 0u, kBatchSlots);
        cq.finish();

        if (!cfg_.enable_trace) return;
        TraceStorageContextGuard trace_alloc_guard;
        {
            ws_->x_decode = ttnn::Tensor();
            ws_->slot_ids = ttnn::Tensor();
            ws_->weights_decode64 = ttnn::Tensor();
            cq.finish();
            const auto tid = mesh_->begin_mesh_trace(cq);
            ws_->x_decode = ttnn::broadcast(ws_->x_decode_src, distributed::MeshCoordinate{0, 0}, std::nullopt,
                                            ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
            ws_->slot_ids = ttnn::broadcast(ws_->slot_ids_src, distributed::MeshCoordinate{0, 0}, 1,
                                            ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
            ws_->weights_decode64 = ttnn::broadcast(ws_->weights_decode64_src, distributed::MeshCoordinate{0, 0}, std::nullopt,
                                                    ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
            mesh_->end_mesh_trace(cq, tid);
            bcast_traces_[0].id = tid;
            bcast_traces_[0].warmed = true;
        }
        {
            ws_->weights_tiled = ttnn::Tensor();
            slot_meta_dst_ = std::nullopt;
            x_rm_dst_ = std::nullopt;
            cq.finish();
            const auto tid = mesh_->begin_mesh_trace(cq);
            ws_->weights_tiled = ttnn::broadcast(ws_->weights_tiled_src, distributed::MeshCoordinate{0, 0}, std::nullopt,
                                                 ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
            slot_meta_dst_ = ttnn::broadcast(*slot_meta_src_, distributed::MeshCoordinate{0, 0}, std::nullopt,
                                             ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
            x_rm_dst_ = ttnn::broadcast(*x_rm_src_, distributed::MeshCoordinate{0, 0}, std::nullopt,
                                        ttnn::DRAM_MEMORY_CONFIG, tt::tt_fabric::Topology::Linear, 1);
            mesh_->end_mesh_trace(cq, tid);
            bcast_traces_[1].id = tid;
            bcast_traces_[1].warmed = true;
        }
        {
            for (auto& t : ws_->finish_decode) t = ttnn::Tensor();
            cq.finish();
            const auto tid = mesh_->begin_mesh_trace(cq);
            ws_->finish_decode[0] = ttnn::typecast(ws_->combined_f32, DataType::BFLOAT16, ttnn::DRAM_MEMORY_CONFIG);
            ws_->finish_decode[1] = ttnn::all_reduce(
                ws_->finish_decode[0], 1, std::nullopt, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, tt::tt_fabric::Topology::Linear);
            ws_->finish_decode[2] = ttnn::untilize(ws_->finish_decode[1], ttnn::DRAM_MEMORY_CONFIG);
            mesh_->end_mesh_trace(cq, tid);
            reduce_traces_[0].id = tid;
            reduce_traces_[0].warmed = true;
        }
        {
            for (auto& t : arena_->finish_tensors) t = ttnn::Tensor();
            cq.finish();
            const auto tid = mesh_->begin_mesh_trace(cq);
            arena_->finish_tensors[0] = ttnn::typecast(arena_->y_accum_f32, DataType::BFLOAT16, ttnn::DRAM_MEMORY_CONFIG);
            arena_->finish_tensors[1] = ttnn::all_reduce(
                arena_->finish_tensors[0], 1, std::nullopt, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, tt::tt_fabric::Topology::Linear);
            mesh_->end_mesh_trace(cq, tid);
            reduce_traces_[1].id = tid;
            reduce_traces_[1].warmed = true;
        }
        cq.finish();
    }

    template <typename ExecFn, typename RedFn>
    void run_traced(uint32_t qmode, uint32_t path_id, uint32_t scheduled_slots, uint64_t est_bytes, ExecFn&& exec_fn, RedFn&& red_fn) {
        TraceStorageContextGuard guard;
        auto& cq = mesh_->mesh_command_queue();
        const uint32_t trace_cat = qmode * 2u + path_id;
        TraceSlot* op_tr = nullptr;
        if (cfg_.enable_trace) {
            auto key = std::make_pair(trace_cat, scheduled_slots);
            auto it = op_traces_.find(key);
            if (it != op_traces_.end()) op_tr = &it->second;
            else if (op_traces_.size() < 64u) op_tr = &op_traces_[key];
        }
        if (op_tr && op_tr->warmed && !op_tr->skipped) {
            if (!op_tr->id) {
                if (uint64_t(mesh_->get_trace_buffers_size()) + est_bytes + (8ULL << 20) <= uint64_t(trace_region_bytes_)) {
                    cq.finish();
                    const auto tid = mesh_->begin_mesh_trace(cq);
                    exec_fn();
                    mesh_->end_mesh_trace(cq, tid);
                    op_tr->id = tid;
                } else {
                    op_tr->skipped = true;
                }
            }
            if (op_tr->id) mesh_->replay_mesh_trace(cq, *op_tr->id, false);
            else exec_fn();
        } else {
            exec_fn();
            if (op_tr) op_tr->warmed = true;
        }
        TraceSlot& red_tr = reduce_traces_[path_id];
        if (cfg_.enable_trace && red_tr.id.has_value()) mesh_->replay_mesh_trace(cq, *red_tr.id, false);
        else red_fn();
    }

    void run_decode_small(
        const ResidentLayerBank& bank,
        const QuantBank& qb,
        uint32_t qmode,
        uint32_t rows,
        uint32_t top_k,
        const int32_t row_slots[][kTopK],
        const float row_weights[][kTopK],
        const uint16_t* x_rot_bf16,
        float* y_out_f32)
    {
        struct SmallPair { uint16_t row; float weight; };
        std::array<std::array<SmallPair, 32>, 256> exp_pairs{};
        std::array<uint8_t, 256> exp_cnt{};
        std::vector<uint16_t> active_experts;
        active_experts.reserve(48);

        for (uint32_t r = 0u; r < rows; ++r) {
            for (uint32_t s = 0u; s < top_k; ++s) {
                const int32_t eid = row_slots[r][s];
                const float w = row_weights[r][s];
                if (eid >= 0 && uint32_t(eid) < qb.n_experts && w != 0.0f) {
                    uint8_t& c = exp_cnt[uint32_t(eid)];
                    if (c == 0u) active_experts.push_back(uint16_t(eid));
                    if (c < 32u) exp_pairs[uint32_t(eid)][c++] = SmallPair{uint16_t(r), w};
                }
            }
        }
        if (active_experts.empty()) return;

        const uint32_t n_act = uint32_t(active_experts.size());
        stat_total_slots_ += n_act;
        for (uint16_t eid : active_experts) {
            if (qb.hot_slot_for_eid[eid] != 0xFFFFu) ++stat_l1_hits_;
        }

        auto* x_tiles = static_cast<uint16_t*>(host_x_tilized_.ptr);
        const auto& t_lut = tile_offset_lut();
        for (uint32_t r = rows; r < dirty_mt0_; ++r) {
            const auto& row_off = t_lut[r];
            for (uint32_t kt = 0u; kt < kHiddenTiles; ++kt) {
                uint16_t* dst_t = x_tiles + kt * 1024u;
                for (int c = 0; c < 32; ++c) dst_t[row_off[size_t(c)]] = 0u;
            }
        }
        dirty_mt0_ = rows;

        for (uint32_t r = 0u; r < rows; ++r) {
            const uint16_t* src_row = x_rot_bf16 + size_t(r) * kHiddenSize;
            const auto& row_off = t_lut[r];
            const uint32_t off0  = row_off[0];
            const uint32_t off16 = row_off[16];
            for (uint32_t kt = 0u; kt < kHiddenTiles; ++kt) {
                uint16_t* dst_t = x_tiles + kt * 1024u;
                const uint16_t* sb = src_row + kt * 32u;
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_t + off0),
                                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sb)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_t + off16),
                                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sb + 16)));
            }
        }

        const uint32_t scheduled_slots = choose_decode_slot_bucket(n_act);
        auto* slot_u16 = static_cast<uint16_t*>(host_slot_ids_.ptr);
        for (uint32_t i = 0u; i < kMaxCompactSlots; ++i) slot_u16[i] = 0xFFFFu;
        for (uint32_t i = 0u; i < n_act; ++i) slot_u16[i] = active_experts[i];
        write_layer_desc(bank, qb, qmode);

        const size_t weights_bytes = size_t(scheduled_slots) * 4096ULL;
        auto* w_f32 = static_cast<float*>(host_weights_.ptr);
        std::memset(w_f32, 0, weights_bytes);
        for (uint32_t i = 0u; i < n_act; ++i) {
            const uint16_t eid = active_experts[i];
            const uint8_t cnt = exp_cnt[eid];
            float* dst_t = w_f32 + size_t(i) * 1024u;
            for (uint8_t k = 0u; k < cnt; ++k) {
                const auto& row_off = t_lut[exp_pairs[eid][k].row];
                const float w = exp_pairs[eid][k].weight;
                for (int c = 0; c < 32; ++c) dst_t[row_off[size_t(c)]] += w;
            }
        }

        const bool use_weights64 = (scheduled_slots <= CoreWorkspace770b::kDecodeFastWeightSlots);
        auto& cq = mesh_->mesh_command_queue();
        if (cfg_.x16_relay && rows > 6u) {
            TraceStorageContextGuard guard;
            const distributed::MeshCoordinateRange card0(distributed::MeshCoordinate{0, 0}, distributed::MeshCoordinate{0, 0});
            cq.enqueue_write_shard_to_sub_grid(ws_->x_decode_src.mesh_buffer(), host_x_tilized_.ptr, card0, false, BufferRegion(0, kHiddenTiles * 2048u));
            cq.enqueue_write_shard_to_sub_grid(ws_->slot_ids_src.mesh_buffer(), host_slot_ids_.ptr, card0, false, BufferRegion(0, kSlotIdsBytes));
            if (use_weights64) {
                cq.enqueue_write_shard_to_sub_grid(ws_->weights_decode64_src.mesh_buffer(), host_weights_.ptr, card0, false, BufferRegion(0, weights_bytes));
            } else {
                cq.enqueue_write_shard_to_sub_grid(ws_->weights_tiled_src.mesh_buffer(), host_weights_.ptr, card0, false, BufferRegion(0, weights_bytes));
            }
            if (bcast_traces_[0].id.has_value()) mesh_->replay_mesh_trace(cq, *bcast_traces_[0].id, false);
            else run_bcast_decode64_untraced();
            if (!use_weights64) {
                if (bcast_traces_[1].id.has_value()) mesh_->replay_mesh_trace(cq, *bcast_traces_[1].id, false);
                else run_bcast_prefill_untraced();
            }
        } else {
            const distributed::MeshCoordinateRange all_devs(mesh_->shape());
            cq.enqueue_write_shard_to_sub_grid(ws_->x_decode.mesh_buffer(), host_x_tilized_.ptr, all_devs, false, BufferRegion(0, kHiddenTiles * 2048u));
            cq.enqueue_write_shard_to_sub_grid(ws_->slot_ids.mesh_buffer(), host_slot_ids_.ptr, all_devs, false, BufferRegion(0, kSlotIdsBytes));
            auto& wt = use_weights64 ? ws_->weights_decode64 : ws_->weights_tiled;
            cq.enqueue_write_shard_to_sub_grid(wt.mesh_buffer(), host_weights_.ptr, all_devs, false, BufferRegion(0, weights_bytes));
        }

        run_traced(
            qmode, 0u, scheduled_slots,
            uint64_t((scheduled_slots + kBatchSlots - 1u) / kBatchSlots) * 131072ULL + (2ULL << 20),
            [&]() {
                for (uint32_t s = 0u; s < scheduled_slots; s += kBatchSlots) {
                    const uint32_t bcnt = std::min<uint32_t>(kBatchSlots, scheduled_slots - s);
                    enqueue_projection(qb, qmode, ws_->x_decode, qb.gate_pages, ws_->up_partials, s, bcnt, 0u, 3u);
                    enqueue_swiglu_wht128(bank, bcnt);
                    enqueue_projection(qb, qmode, ws_->z_rot_bf16, qb.down_pages, ws_->down_partials, s, bcnt, kInterShardTiles, 2u);
                    enqueue_ksplit_combine_decode(s, bcnt, use_weights64);
                }
            },
            [&]() { run_reduce_decode_untraced(); });

        const size_t out_bytes = size_t(rows) * kHiddenSize * sizeof(uint16_t);
        const auto& out_rm = ws_->finish_decode[2];
        auto borrowed_out = std::shared_ptr<distributed::MeshBuffer>(
            const_cast<distributed::MeshBuffer*>(&out_rm.mesh_buffer()), [](auto*) {});
        cq.enqueue_read_shards(
            {distributed::ShardDataTransfer{distributed::MeshCoordinate{0, 0}}
                 .host_data(host_out_rm_.ptr)
                 .region(BufferRegion(0, out_bytes))},
            borrowed_out, true);

        const auto* out_bf16 = static_cast<const uint16_t*>(host_out_rm_.ptr);
        for (uint32_t r = 0u; r < rows; ++r) {
            bf16_row_add_to_f32_avx2(out_bf16 + size_t(r) * kHiddenSize, y_out_f32 + size_t(r) * kHiddenSize);
        }
    }

    void run_prefill_compact(
        const ResidentLayerBank& bank,
        const QuantBank& qb,
        uint32_t qmode,
        uint32_t rows,
        uint32_t top_k,
        const int32_t row_slots[][kTopK],
        const float row_weights[][kTopK],
        const uint16_t* x_rot_bf16,
        float* y_out_f32)
    {
        for (uint32_t r0 = 0u; r0 < rows; r0 += kPrefillMaxRows) {
            const uint32_t chunk = std::min<uint32_t>(kPrefillMaxRows, rows - r0);
            run_prefill_chunk(bank, qb, qmode, chunk, top_k, row_slots + r0, row_weights + r0,
                              x_rot_bf16 + size_t(r0) * kHiddenSize, y_out_f32 + size_t(r0) * kHiddenSize);
        }
    }

    void run_prefill_chunk(
        const ResidentLayerBank& bank,
        const QuantBank& qb,
        uint32_t qmode,
        uint32_t rows,
        uint32_t top_k,
        const int32_t row_slots[][kTopK],
        const float row_weights[][kTopK],
        const uint16_t* x_rot_bf16,
        float* y_out_f32)
    {
        struct RoutePair { uint16_t row; float weight; };
        std::array<std::vector<RoutePair>, 256> buckets;
        uint32_t total_pairs = 0u;
        for (uint32_t r = 0u; r < rows; ++r) {
            for (uint32_t s = 0u; s < top_k; ++s) {
                const int32_t eid = row_slots[r][s];
                const float w = row_weights[r][s];
                if (eid >= 0 && uint32_t(eid) < qb.n_experts && w != 0.0f) {
                    buckets[uint32_t(eid)].push_back(RoutePair{uint16_t(r), w});
                    ++total_pairs;
                }
            }
        }
        if (total_pairs == 0u) return;
        for (uint32_t eid = 0u; eid < qb.n_experts; ++eid) {
            if (!buckets[eid].empty()) {
                ++stat_total_slots_;
                if (qb.hot_slot_for_eid[eid] != 0xFFFFu) ++stat_l1_hits_;
            }
        }

        const size_t x_rm_bytes = size_t(rows) * kHiddenSize * sizeof(uint16_t);
        std::memcpy(host_x_rm_.ptr, x_rot_bf16, x_rm_bytes);

        struct ExpertRun { uint16_t eid; uint16_t next_off; uint16_t rem_tiles; };
        std::vector<ExpertRun> runs;
        uint32_t left_tiles = 0u;
        for (uint32_t eid = 0u; eid < qb.n_experts; ++eid) {
            const size_t n = buckets[eid].size();
            if (n == 0u) continue;
            const uint16_t nt = uint16_t((n + 31u) / 32u);
            runs.push_back(ExpertRun{uint16_t(eid), 0u, nt});
            left_tiles += nt;
        }
        std::sort(runs.begin(), runs.end(), [](const ExpertRun& a, const ExpertRun& b) {
            return (a.rem_tiles != b.rem_tiles) ? (a.rem_tiles > b.rem_tiles) : (a.eid < b.eid);
        });

        struct PlacedTile { uint16_t eid; uint16_t off; uint16_t cnt; };
        std::vector<PlacedTile> placed;
        placed.reserve(left_tiles);
        while (left_tiles > 0u) {
            uint32_t batch_rem = kBatchSlots;
            while (batch_rem > 0u && left_tiles > 0u) {
                int32_t best = -1;
                for (size_t idx = 0; idx < runs.size(); ++idx) {
                    if (runs[idx].rem_tiles > 0u && runs[idx].rem_tiles <= batch_rem) { best = int32_t(idx); break; }
                }
                if (best < 0) {
                    uint16_t mx = 0u;
                    for (size_t idx = 0; idx < runs.size(); ++idx) {
                        if (runs[idx].rem_tiles > mx) { mx = runs[idx].rem_tiles; best = int32_t(idx); }
                    }
                }
                if (best < 0) break;
                auto& er = runs[size_t(best)];
                const uint32_t take = std::min<uint32_t>(batch_rem, er.rem_tiles);
                const size_t tot_n = buckets[er.eid].size();
                for (uint32_t t = 0u; t < take; ++t) {
                    const uint16_t cnt = uint16_t(std::min<size_t>(32u, tot_n - er.next_off));
                    placed.push_back(PlacedTile{er.eid, er.next_off, cnt});
                    er.next_off = uint16_t(er.next_off + cnt);
                }
                er.rem_tiles = uint16_t(er.rem_tiles - take);
                batch_rem -= take;
                left_tiles -= take;
            }
        }

        const auto& t_lut = tile_offset_lut();
        auto& cq = mesh_->mesh_command_queue();
        bool x_uploaded = false;

        for (size_t pass_off = 0; pass_off < placed.size(); pass_off += kMaxCompactSlots) {
            const uint32_t active_slots = uint32_t(std::min<size_t>(kMaxCompactSlots, placed.size() - pass_off));
            const uint32_t scheduled_slots = choose_compact_slot_bucket(active_slots);

            auto* slot_u16   = static_cast<uint16_t*>(host_slot_ids_.ptr);
            auto* meta_bytes = static_cast<uint8_t*>(host_slot_meta_.ptr);
            auto* w_f32      = static_cast<float*>(host_weights_.ptr);
            for (uint32_t i = 0u; i < kMaxCompactSlots; ++i) slot_u16[i] = 0xFFFFu;
            write_layer_desc(bank, qb, qmode);

            std::array<uint64_t, 8> inflight_mask{}, tile_mask{};
            const uint32_t mask_words = (rows + 63u) >> 6;
            for (uint32_t s = 0u; s < active_slots; ++s) {
                if ((s % kBatchSlots) == 0u) {
                    for (uint32_t w = 0u; w < mask_words; ++w) inflight_mask[w] = 0ULL;
                }
                const auto& pt = placed[pass_off + s];
                const auto& vec = buckets[pt.eid];
                for (uint32_t w = 0u; w < mask_words; ++w) tile_mask[w] = 0ULL;
                uint64_t overlap = 0ULL;
                for (uint32_t k = 0u; k < pt.cnt; ++k) {
                    const uint16_t r = vec[pt.off + k].row;
                    const uint32_t wi = r >> 6;
                    const uint64_t bit = 1ULL << (r & 63u);
                    tile_mask[wi] |= bit;
                    overlap |= (inflight_mask[wi] & bit);
                }
                const uint32_t wait_barrier = ((s % kBatchSlots) > 0u && overlap != 0ULL) ? 1u : 0u;
                if (wait_barrier != 0u) {
                    for (uint32_t w = 0u; w < mask_words; ++w) inflight_mask[w] = tile_mask[w];
                } else {
                    for (uint32_t w = 0u; w < mask_words; ++w) inflight_mask[w] |= tile_mask[w];
                }

                slot_u16[s] = pt.eid;
                uint8_t* mptr = meta_bytes + size_t(s) * 128u;
                std::memset(mptr, 0, 128u);
                reinterpret_cast<uint32_t*>(mptr)[0] = pt.cnt;
                reinterpret_cast<uint32_t*>(mptr)[1] = wait_barrier;
                uint16_t* rptr = reinterpret_cast<uint16_t*>(mptr + 8u);

                float* wtile = w_f32 + size_t(s) * 1024u;
                std::memset(wtile, 0, 4096u);
                for (uint32_t k = 0u; k < pt.cnt; ++k) {
                    rptr[k] = vec[pt.off + k].row;
                    const float wk = vec[pt.off + k].weight;
                    const auto& row_off = t_lut[k];
                    for (int c = 0; c < 32; ++c) wtile[row_off[size_t(c)]] = wk;
                }
            }
            if (scheduled_slots > active_slots) {
                std::memset(meta_bytes + size_t(active_slots) * 128u, 0, size_t(scheduled_slots - active_slots) * 128u);
                std::memset(w_f32 + size_t(active_slots) * 1024u, 0, size_t(scheduled_slots - active_slots) * 4096u);
            }

            if (cfg_.x16_relay) {
                TraceStorageContextGuard guard;
                const distributed::MeshCoordinateRange card0(distributed::MeshCoordinate{0, 0}, distributed::MeshCoordinate{0, 0});
                if (!x_uploaded) {
                    cq.enqueue_write_shard_to_sub_grid(x_rm_src_->mesh_buffer(), host_x_rm_.ptr, card0, false, BufferRegion(0, x_rm_bytes));
                    x_uploaded = true;
                }
                cq.enqueue_write_shard_to_sub_grid(ws_->slot_ids_src.mesh_buffer(), host_slot_ids_.ptr, card0, false, BufferRegion(0, kSlotIdsBytes));
                cq.enqueue_write_shard_to_sub_grid(slot_meta_src_->mesh_buffer(), host_slot_meta_.ptr, card0, false, BufferRegion(0, size_t(scheduled_slots) * 128u));
                cq.enqueue_write_shard_to_sub_grid(ws_->weights_tiled_src.mesh_buffer(), host_weights_.ptr, card0, false, BufferRegion(0, size_t(scheduled_slots) * 4096u));
                if (bcast_traces_[0].id.has_value()) mesh_->replay_mesh_trace(cq, *bcast_traces_[0].id, false);
                else run_bcast_decode64_untraced();
                if (bcast_traces_[1].id.has_value()) mesh_->replay_mesh_trace(cq, *bcast_traces_[1].id, false);
                else run_bcast_prefill_untraced();
                dirty_mt0_ = 32u;
            } else {
                const distributed::MeshCoordinateRange all_devs(mesh_->shape());
                if (!x_uploaded) {
                    cq.enqueue_write_shard_to_sub_grid(x_rm_dst_->mesh_buffer(), host_x_rm_.ptr, all_devs, false, BufferRegion(0, x_rm_bytes));
                    x_uploaded = true;
                }
                cq.enqueue_write_shard_to_sub_grid(ws_->slot_ids.mesh_buffer(), host_slot_ids_.ptr, all_devs, false, BufferRegion(0, kSlotIdsBytes));
                cq.enqueue_write_shard_to_sub_grid(slot_meta_dst_->mesh_buffer(), host_slot_meta_.ptr, all_devs, false, BufferRegion(0, size_t(scheduled_slots) * 128u));
                cq.enqueue_write_shard_to_sub_grid(ws_->weights_tiled.mesh_buffer(), host_weights_.ptr, all_devs, false, BufferRegion(0, size_t(scheduled_slots) * 4096u));
            }

            run_traced(
                qmode, 1u, scheduled_slots,
                uint64_t((scheduled_slots + kBatchSlots - 1u) / kBatchSlots) * 147456ULL + (4ULL << 20),
                [&]() {
                    enqueue_zero_pages(arena_->y_accum_f32, kPrefillMaxRows * 6u);
                    for (uint32_t s = 0u; s < scheduled_slots; s += kBatchSlots) {
                        const uint32_t bcnt = std::min<uint32_t>(kBatchSlots, scheduled_slots - s);
                        enqueue_compact_gather_tilize(*x_rm_dst_, s, bcnt);
                        enqueue_projection(qb, qmode, ws_->x_tilized, qb.gate_pages, ws_->up_partials, s, bcnt, kHiddenTiles, 3u);
                        enqueue_swiglu_wht128(bank, bcnt);
                        enqueue_projection(qb, qmode, ws_->z_rot_bf16, qb.down_pages, ws_->down_partials, s, bcnt, kInterShardTiles, 2u);
                        enqueue_compact_scatter_combine(arena_->y_accum_f32, s, bcnt);
                    }
                },
                [&]() { run_reduce_prefill_untraced(); });

            const auto& out_rm = arena_->finish_tensors[1];
            auto borrowed_out = std::shared_ptr<distributed::MeshBuffer>(
                const_cast<distributed::MeshBuffer*>(&out_rm.mesh_buffer()), [](auto*) {});
            cq.enqueue_read_shards(
                {distributed::ShardDataTransfer{distributed::MeshCoordinate{0, 0}}
                     .host_data(host_out_rm_.ptr)
                     .region(BufferRegion(0, x_rm_bytes))},
                borrowed_out, true);

            const auto* out_bf16 = static_cast<const uint16_t*>(host_out_rm_.ptr);
            for (uint32_t r = 0u; r < rows; ++r) {
                bf16_row_add_to_f32_avx2(out_bf16 + size_t(r) * kHiddenSize, y_out_f32 + size_t(r) * kHiddenSize);
            }
        }
    }

    WorkerConfig cfg_;
    uint32_t n_h4_ = 0u;
    uint32_t n_h3_ = 0u;
    uint32_t n_h2_ = 0u;
    uint32_t hot_qmode_ = kQModeHlut3;

    std::shared_ptr<distributed::MeshDevice> mesh_;
    CoreCoord grid_{0, 0};
    uint32_t total_cores_ = 0u;
    uint32_t trace_region_bytes_ = 0u;

    std::unique_ptr<CoreWorkspace770b> ws_;
    std::optional<ttnn::Tensor> x_rm_src_, x_rm_dst_;
    std::optional<ttnn::Tensor> slot_meta_src_, slot_meta_dst_;
    std::optional<ttnn::Tensor> l1_lut_h4_, l1_lut_h3_, l1_lut_h2_, dummy_hot_h4_, dummy_hot_h3_, dummy_hot_h2_;
    std::unique_ptr<CompactPrefillArena770b> arena_;

    uint32_t num_hot_shards_ = 0u;
    size_t pinned_hot_l1_bytes_ = 0u;
    uint64_t gddr6_bytes_per_card_ = 0u;
    std::optional<ttnn::Tensor> hot_gate_l1_, hot_up_l1_, hot_down_l1_;

    LockedBuffer host_x_tilized_{kHiddenTiles * 2048u};
    LockedBuffer host_x_rm_{kPrefillMaxRows * kHiddenSize * sizeof(uint16_t)};
    LockedBuffer host_slot_ids_{kSlotIdsBytes};
    LockedBuffer host_slot_lut_h4_{uint32_t(kNumLayers) * 256u * sizeof(uint16_t)};
    LockedBuffer host_slot_lut_h3_{uint32_t(kNumLayers) * 256u * sizeof(uint16_t)};
    LockedBuffer host_slot_lut_h2_{uint32_t(kNumLayers) * 256u * sizeof(uint16_t)};
    LockedBuffer host_slot_meta_{kMaxCompactSlots * 128u};
    LockedBuffer host_weights_{kMaxCompactSlots * 4096u};
    LockedBuffer host_out_rm_{kPrefillMaxRows * kHiddenSize * sizeof(uint16_t)};
    std::vector<uint16_t> x_bf16_scratch_;

    std::array<std::optional<ResidentLayerBank>, kNumLayers> layers_;

    struct TraceSlot {
        std::optional<distributed::MeshTraceId> id;
        bool warmed = false;
        bool skipped = false;
    };
    std::map<std::pair<uint32_t, uint32_t>, TraceSlot> op_traces_;
    std::array<TraceSlot, 2> reduce_traces_;
    std::array<TraceSlot, 2> bcast_traces_;
    uint32_t dirty_mt0_ = 0u;
    bool programs_prewarmed_ = false;
    uint64_t stat_total_slots_ = 0u;
    uint64_t stat_l1_hits_ = 0u;
};

// ============================================================================
// Exact FP64 Reference Verifier for --self-test (Layer 1 HLUT3 + HLUT2!)
// ============================================================================
void ref_expert_layer1_fp64(
    const WorkerConfig& cfg,
    uint32_t rows,
    const int32_t row_slots[][kTopK],
    const float row_weights[][kTopK],
    const float* x_rot_f32,
    std::vector<double>& y_ref)
{
    y_ref.assign(size_t(rows) * kHiddenSize, 0.0);
    const auto& sol4 = lloyd_max_hlut4();
    const auto& sol3 = lloyd_max_hlut3();
    const auto& sol2 = lloyd_max_hlut2();
    const auto signs_i = make_signs_i(1);
    const int n_h4 = std::max(0, cfg.hlut4_end - cfg.hlut4_begin);
    const int n_h3 = std::max(0, cfg.hlut3_end - cfg.hlut3_begin);

    auto load_dequant = [&](int local_slot, std::vector<double>& wg, std::vector<double>& wu, std::vector<double>& wd) {
        wg.assign(size_t(kMoeInterSize) * kHiddenSize, 0.0);
        wu.assign(size_t(kMoeInterSize) * kHiddenSize, 0.0);
        wd.assign(size_t(kHiddenSize) * kMoeInterSize, 0.0);
        if (local_slot < n_h4) {
            std::ifstream ifs(cfg.weight_root + "/hlut4_hot/hlut4_hot_layer_01.bin", std::ios::binary);
            ifs.seekg(std::streamoff(size_t(cfg.hlut4_begin + local_slot) * kHlut4ExpertBytes));
            std::vector<Hlut4Tile32x32> raw(3u * kProjTiles);
            ifs.read(reinterpret_cast<char*>(raw.data()), std::streamsize(kHlut4ExpertBytes));
            auto dec = [&](const Hlut4Tile32x32* t, int N, int K, std::vector<double>& W) {
                std::vector<float> tmp(size_t(N) * size_t(K));
                const int nt = N / 32, kt = K / 32;
                for (int ni = 0; ni < nt; ++ni)
                    for (int ki = 0; ki < kt; ++ki)
                        hlut4_unpack_tile(t[size_t(ni) * size_t(kt) + size_t(ki)], sol4, tmp.data(), K, ni * 32, ki * 32);
                for (size_t i = 0; i < tmp.size(); ++i) W[i] = double(tmp[i]);
            };
            dec(raw.data() + 0u * kProjTiles, kMoeInterSize, kHiddenSize, wg);
            dec(raw.data() + 1u * kProjTiles, kMoeInterSize, kHiddenSize, wu);
            dec(raw.data() + 2u * kProjTiles, kHiddenSize, kMoeInterSize, wd);
        } else if (local_slot < n_h4 + n_h3) {
            const int h3_idx = local_slot - n_h4;
            std::ifstream ifs(cfg.weight_root + "/hlut3/hlut3_layer_01.bin", std::ios::binary);
            ifs.seekg(std::streamoff(size_t(cfg.hlut3_begin + h3_idx) * kHlut3ExpertBytes));
            std::vector<Hlut3Tile32x32> raw(3u * kProjTiles);
            ifs.read(reinterpret_cast<char*>(raw.data()), std::streamsize(kHlut3ExpertBytes));
            auto dec = [&](const Hlut3Tile32x32* t, int N, int K, std::vector<double>& W) {
                std::vector<float> tmp(size_t(N) * size_t(K));
                const int nt = N / 32, kt = K / 32;
                for (int ni = 0; ni < nt; ++ni)
                    for (int ki = 0; ki < kt; ++ki)
                        hlut3_unpack_tile(t[size_t(ni) * size_t(kt) + size_t(ki)], sol3, tmp.data(), K, ni * 32, ki * 32);
                for (size_t i = 0; i < tmp.size(); ++i) W[i] = double(tmp[i]);
            };
            dec(raw.data() + 0u * kProjTiles, kMoeInterSize, kHiddenSize, wg);
            dec(raw.data() + 1u * kProjTiles, kMoeInterSize, kHiddenSize, wu);
            dec(raw.data() + 2u * kProjTiles, kHiddenSize, kMoeInterSize, wd);
        } else {
            const int h2_idx = local_slot - n_h4 - n_h3;
            std::ifstream ifs(cfg.weight_root + "/hlut2/hlut2_layer_01.bin", std::ios::binary);
            ifs.seekg(std::streamoff(size_t(cfg.hlut2_begin + h2_idx - kPackHlut2Begin) * kHlut2ExpertBytes));
            std::vector<Hlut2Tile32x32> raw(3u * kProjTiles);
            ifs.read(reinterpret_cast<char*>(raw.data()), std::streamsize(kHlut2ExpertBytes));
            auto dec = [&](const Hlut2Tile32x32* t, int N, int K, std::vector<double>& W) {
                std::vector<float> tmp(size_t(N) * size_t(K));
                const int nt = N / 32, kt = K / 32;
                for (int ni = 0; ni < nt; ++ni)
                    for (int ki = 0; ki < kt; ++ki)
                        hlut2_unpack_tile(t[size_t(ni) * size_t(kt) + size_t(ki)], sol2, tmp.data(), K, ni * 32, ki * 32);
                for (size_t i = 0; i < tmp.size(); ++i) W[i] = double(tmp[i]);
            };
            dec(raw.data() + 0u * kProjTiles, kMoeInterSize, kHiddenSize, wg);
            dec(raw.data() + 1u * kProjTiles, kMoeInterSize, kHiddenSize, wu);
            dec(raw.data() + 2u * kProjTiles, kHiddenSize, kMoeInterSize, wd);
        }
    };

    std::map<int, std::array<std::vector<double>, 3>> cache;
    for (uint32_t r = 0u; r < rows; ++r) {
        const float* xr = x_rot_f32 + size_t(r) * kHiddenSize;
        for (uint32_t s = 0u; s < kTopK; ++s) {
            const int e = row_slots[r][s];
            const double wt = double(row_weights[r][s]);
            if (e < 0 || wt == 0.0) continue;
            if (!cache.count(e)) load_dequant(e, cache[e][0], cache[e][1], cache[e][2]);
            const auto& wg = cache[e][0];
            const auto& wu = cache[e][1];
            const auto& wd = cache[e][2];
            std::vector<float> z(kMoeInterSize);
            for (int i = 0; i < kMoeInterSize; ++i) {
                double g = 0.0, u = 0.0;
                for (int k = 0; k < kHiddenSize; ++k) {
                    g += wg[size_t(i) * kHiddenSize + size_t(k)] * double(xr[k]);
                    u += wu[size_t(i) * kHiddenSize + size_t(k)] * double(xr[k]);
                }
                const double gc = std::min(g, double(kSwigluLimit));
                const double uc = std::clamp(u, -double(kSwigluLimit), double(kSwigluLimit));
                z[size_t(i)] = float((gc / (1.0 + std::exp(-gc))) * uc);
            }
            for (int b = 0; b < kMoeInterSize / 128; ++b) wht128_block(z.data() + b * 128, signs_i.data() + size_t(b) * 2u);
            for (int d = 0; d < kHiddenSize; ++d) {
                double acc = 0.0;
                for (int i = 0; i < kMoeInterSize; ++i) acc += wd[size_t(d) * kMoeInterSize + size_t(i)] * double(z[size_t(i)]);
                y_ref[size_t(r) * kHiddenSize + size_t(d)] += wt * acc;
            }
        }
    }
}

int run_self_test(P150Hy4770bEngine& eng, const WorkerConfig& cfg) {
    eng.load_layers(1, 1);
    std::mt19937 rng(770);
    std::normal_distribution<float> nd(0.0f, 0.25f);
    const int n_tot = int(eng.n_hlut4() + eng.n_hlut3() + eng.n_hlut2());

    bool all_ok = true;
    for (uint32_t M : {1u, 6u, 64u}) {
        std::vector<float> x_rot(size_t(M) * kHiddenSize);
        for (float& v : x_rot) v = nd(rng);
        std::vector<std::array<int32_t, kTopK>> slots(M);
        std::vector<std::array<float, kTopK>> wts(M);
        for (uint32_t r = 0u; r < M; ++r) {
            slots[r].fill(-1); wts[r].fill(0.0f);
            slots[r][0] = 0;                       wts[r][0] = 0.45f;  // L1 hot-pinned slot 0!
            slots[r][1] = std::min(n_tot - 1, 1);  wts[r][1] = 0.30f;
            if (eng.n_hlut2() > 0u) {
                slots[r][2] = int32_t(eng.n_hlut4() + eng.n_hlut3()); wts[r][2] = 0.25f;
            }
        }
        std::vector<double> y_ref;
        ref_expert_layer1_fp64(cfg, M,
                               reinterpret_cast<const int32_t(*)[kTopK]>(slots.data()),
                               reinterpret_cast<const float(*)[kTopK]>(wts.data()),
                               x_rot.data(), y_ref);
        std::vector<float> y_dev(size_t(M) * kHiddenSize, 0.0f);
        // Warmup + traced run
        eng.forward_layer(1, M, kTopK, reinterpret_cast<const int32_t(*)[kTopK]>(slots.data()),
                          reinterpret_cast<const float(*)[kTopK]>(wts.data()), x_rot.data(), y_dev.data());
        const auto t0 = std::chrono::steady_clock::now();
        eng.forward_layer(1, M, kTopK, reinterpret_cast<const int32_t(*)[kTopK]>(slots.data()),
                          reinterpret_cast<const float(*)[kTopK]>(wts.data()), x_rot.data(), y_dev.data());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        double dot = 0.0, n1 = 0.0, n2 = 0.0;
        for (size_t i = 0; i < y_ref.size(); ++i) {
            dot += y_ref[i] * double(y_dev[i]);
            n1  += y_ref[i] * y_ref[i];
            n2  += double(y_dev[i]) * double(y_dev[i]);
        }
        const double cos_sim = dot / (std::sqrt(n1 * n2) + 1e-30);
        const bool ok = (cos_sim >= 0.999);
        if (!ok) all_ok = false;
        std::printf("[P150-770B-SELFTEST] M=%2u | cos=%.8f | latency=%.3f ms | L1_hits=%llu/%llu | %s\n",
                    M, cos_sim, ms, (unsigned long long)eng.stat_l1_hits(), (unsigned long long)eng.stat_total_slots(),
                    ok ? "PASS" : "FAIL");
        std::fflush(stdout);
    }
    return (all_ok && eng.stat_l1_hits() > 0ULL) ? 0 : 1;
}

}  // namespace
}  // namespace hy4_770b

int main(int argc, char** argv) {
    using namespace hy4_770b;
    std::signal(SIGINT,  on_sig);
    std::signal(SIGTERM, on_sig);

    WorkerConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* f) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing arg for ") + f);
            return argv[++i];
        };
        if (a == "--weight-root")       cfg.weight_root = need("--weight-root");
        else if (a == "--kernels-dir")  cfg.kernels_dir = need("--kernels-dir");
        else if (a == "--shm")          cfg.shm_name = need("--shm");
        else if (a == "--profile")      cfg.profile_id = uint32_t(std::stoi(need("--profile")));
        else if (a == "--hlut4-begin")  cfg.hlut4_begin = std::stoi(need("--hlut4-begin"));
        else if (a == "--hlut4-end")    cfg.hlut4_end   = std::stoi(need("--hlut4-end"));
        else if (a == "--hlut3-begin")  cfg.hlut3_begin = std::stoi(need("--hlut3-begin"));
        else if (a == "--hlut3-end")    cfg.hlut3_end   = std::stoi(need("--hlut3-end"));
        else if (a == "--hlut2-begin")  cfg.hlut2_begin = std::stoi(need("--hlut2-begin"));
        else if (a == "--hlut2-end")    cfg.hlut2_end   = std::stoi(need("--hlut2-end"));
        else if (a == "--no-trace")     cfg.enable_trace = false;
        else if (a == "--no-x16-relay") cfg.x16_relay = false;
        else if (a == "--self-test")    cfg.self_test = true;
        else if (a == "--max-layers")   cfg.max_layers_load = std::stoi(need("--max-layers"));
    }

    try {
        P150Hy4770bEngine eng(cfg);
        if (cfg.self_test) {
            return run_self_test(eng, cfg);
        }

        eng.load_layers(1, std::min(kNumMoeLayers, cfg.max_layers_load));
        if (g_stop) return 0;

        Hy4770bP150ShmMapping shm;
        if (!shm.create_server(cfg.shm_name)) {
            std::fprintf(stderr, "[P150-770B] Failed to create POSIX SHM %s\n", cfg.shm_name.c_str());
            return 1;
        }
        auto* ch = shm.ptr;
        ch->profile_id            = cfg.profile_id;
        ch->n_hlut4_per_layer     = eng.n_hlut4();
        ch->n_hlut3_per_layer     = eng.n_hlut3();
        ch->n_hlut2_per_layer     = eng.n_hlut2();
        ch->fabric_active         = 1u;
        ch->l1_hot_experts        = eng.num_hot_shards();
        ch->l1_pinned_bytes_total = eng.pinned_l1_bytes_total();
        ch->gddr6_bytes_per_card  = eng.gddr6_bytes_per_card();
        std::atomic_thread_fence(std::memory_order_release);
        ch->ready = 1u;

        std::printf("[P150-770B] READY on SHM %s (PID=%d, HLUT4=%u/L, HLUT3=%u/L, HLUT2=%u/L, L1=%u shards [%.1f MiB], GDDR6=%.2f GiB/card)\n",
                    cfg.shm_name.c_str(), int(::getpid()), eng.n_hlut4(), eng.n_hlut3(), eng.n_hlut2(),
                    eng.num_hot_shards(), double(eng.pinned_l1_bytes_total()) / 1048576.0,
                    double(eng.gddr6_bytes_per_card()) / 1073741824.0);
        std::fflush(stdout);

        uint32_t last_seq = 0u;
        uint32_t idle_spins = 0u;
        while (!g_stop) {
            const uint32_t seq = ch->req_seq.load(std::memory_order_acquire);
            if (seq == last_seq) {
                p150_cpu_relax();
                if (++idle_spins > 50000u) {
                    struct timespec ts{0, 200000};  // 200 us sleep when idle
                    ::nanosleep(&ts, nullptr);
                    idle_spins = 0u;
                }
                continue;
            }
            idle_spins = 0u;
            last_seq = seq;

            const uint32_t op = ch->op;
            if (op == HY4_770B_P150_OP_SHUTDOWN) {
                ch->status = 0;
                ch->resp_seq.store(seq, std::memory_order_release);
                std::printf("[P150-770B] Received graceful IPC SHUTDOWN — draining Blackhole mesh...\n");
                std::fflush(stdout);
                break;
            } else if (op == HY4_770B_P150_OP_PING) {
                ch->status = 0;
                ch->resp_seq.store(seq, std::memory_order_release);
                continue;
            } else if (op == HY4_770B_P150_OP_MOE_EXEC) {
                const auto t0 = std::chrono::steady_clock::now();
                try {
                    eng.forward_layer(int(ch->layer_idx), ch->rows, ch->top_k,
                                      ch->row_slots, ch->row_weights,
                                      ch->x_rot_f32, ch->y_out_f32);
                    const uint64_t ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - t0).count());
                    ch->last_device_ns = ns;
                    ch->total_device_ns.fetch_add(ns, std::memory_order_relaxed);
                    ch->stat_total_slots.store(eng.stat_total_slots(), std::memory_order_relaxed);
                    ch->stat_l1_hits.store(eng.stat_l1_hits(), std::memory_order_relaxed);
                    ch->status = 0;
                } catch (const std::exception& ex) {
                    std::fprintf(stderr, "[P150-770B] EXEC error layer %u: %s\n", ch->layer_idx, ex.what());
                    ch->status = -1;
                }
                ch->resp_seq.store(seq, std::memory_order_release);
            }
        }
        std::printf("[P150-770B] Clean shutdown complete.\n");
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[P150-770B] FATAL: %s\n", ex.what());
        return 1;
    }
}
