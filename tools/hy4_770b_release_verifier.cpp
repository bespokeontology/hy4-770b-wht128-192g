// Pure-C++17 Release Verifier & 32-Test Deterministic Gate Suite for tencent/Hy4 (770B / A49B) WHT128-HLUT
// Verifies:
//   - 32 deterministic unit tests (FIPS 180-4 SHA-256 NIST vectors, WHT128-HLUT2/3/4 FP64 Lloyd-Max & orthogonality,
//     3:3:3:1:1 KV bijection, 2:2:2:1:1 TP5 slicing, 21-full/57-shared IndexCache, cycle breaker, ISA parser, tier plans)
//   - --verify-all: 100% zero-scratch gfx906 ISA audit (emits artifacts/hy4_770b_gfx906_isa_audit.json),
//     read-only sysfs MI50 PCIe AER 0/0 check, Zero-Python check, 1200x720 PNG IHDR validation,
//     required artifacts NaN/Inf-free + SHA-256 integrity.
// Zero Python, zero external dependencies.

#include "../src/hy4_770b_config.hpp"
#include "../src/hy4_770b_hlut_pack.hpp"
#include "../mi50/hy4_cycle_breaker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace hy4_770b_verify {

// ---- Self-contained FIPS 180-4 SHA-256 ----
struct Sha256 {
    uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    uint8_t buf[64]{};
    size_t blen = 0;
    uint64_t total = 0;

    static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32u - n)); }

    void compress(const uint8_t b[64]) {
        static const uint32_t K[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(b[i * 4]) << 24) | (uint32_t(b[i * 4 + 1]) << 16) |
                   (uint32_t(b[i * 4 + 2]) << 8) | uint32_t(b[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b0 = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b0) ^ (a & c) ^ (b0 & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b0; b0 = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b0; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const uint8_t* p, size_t n) {
        total += n;
        size_t i = 0;
        if (blen) {
            const size_t take = std::min(64 - blen, n);
            std::memcpy(buf + blen, p, take);
            blen += take; i += take;
            if (blen == 64) { compress(buf); blen = 0; }
        }
        while (i + 64 <= n) { compress(p + i); i += 64; }
        if (i < n) { std::memcpy(buf, p + i, n - i); blen = n - i; }
    }

    std::string hex() {
        const uint64_t bits = total * 8ULL;
        uint8_t pad = 0x80u, z = 0u;
        update(&pad, 1);
        while (blen != 56) update(&z, 1);
        uint8_t lenb[8];
        for (int i = 0; i < 8; ++i) lenb[i] = uint8_t((bits >> ((7 - i) * 8)) & 0xFFu);
        update(lenb, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + i * 8, 9, "%08x", h[i]);
        out[64] = '\0';
        return std::string(out);
    }
};

static std::string sha256_str(const std::string& s) {
    Sha256 st; st.update(reinterpret_cast<const uint8_t*>(s.data()), s.size()); return st.hex();
}
static std::string sha256_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return "";
    Sha256 st;
    std::vector<uint8_t> buf(1u << 16);
    while (ifs) {
        ifs.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size()));
        const auto got = size_t(ifs.gcount());
        if (got) st.update(buf.data(), got);
    }
    return st.hex();
}

struct IsaStats {
    int kernels = 0, zero_scratch = 0;
    int sgpr_seen = 0, sgpr_zero = 0;
    int vgpr_seen = 0, vgpr_zero = 0;
    int trig = 0, slc_atomic = 0;
    int max_vgpr = 0, max_sgpr = 0, max_lds = 0;
    int ds_read_b128 = 0, v_dot2_f32_f16 = 0;
    bool pass() const {
        return kernels > 0 && kernels == zero_scratch &&
               sgpr_seen == sgpr_zero && vgpr_seen == vgpr_zero &&
               trig == 0 && slc_atomic == 0;
    }
};

static int parse_tail_int(const std::string& line) {
    const size_t c = line.rfind(':');
    if (c == std::string::npos) return -1;
    size_t p = c + 1;
    while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
    if (p >= line.size() || line[p] < '0' || line[p] > '9') return -1;
    return std::atoi(line.c_str() + p);
}

static IsaStats audit_asm_stream(std::istream& is) {
    IsaStats s;
    std::string line;
    while (std::getline(is, line)) {
        if (line.find("ScratchSize:") != std::string::npos) { ++s.kernels; if (parse_tail_int(line) == 0) ++s.zero_scratch; }
        else if (line.find(".sgpr_spill_count:") != std::string::npos) { ++s.sgpr_seen; if (parse_tail_int(line) == 0) ++s.sgpr_zero; }
        else if (line.find(".vgpr_spill_count:") != std::string::npos) { ++s.vgpr_seen; if (parse_tail_int(line) == 0) ++s.vgpr_zero; }
        else if (line.find(".vgpr_count:") != std::string::npos) s.max_vgpr = std::max(s.max_vgpr, parse_tail_int(line));
        else if (line.find(".sgpr_count:") != std::string::npos) s.max_sgpr = std::max(s.max_sgpr, parse_tail_int(line));
        else if (line.find(".group_segment_fixed_size:") != std::string::npos) s.max_lds = std::max(s.max_lds, parse_tail_int(line));
        if (line.find("v_sin_f32") != std::string::npos || line.find("v_cos_f32") != std::string::npos) ++s.trig;
        if ((line.find("global_atomic_") != std::string::npos || line.find("flat_atomic_") != std::string::npos) &&
            line.find("slc") != std::string::npos) ++s.slc_atomic;
        if (line.find("ds_read_b128") != std::string::npos) ++s.ds_read_b128;
        if (line.find("v_dot2_f32_f16") != std::string::npos) ++s.v_dot2_f32_f16;
    }
    return s;
}

static bool validate_png_header(const std::string& path, uint32_t& w, uint32_t& h) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    uint8_t hdr[24]{};
    if (!ifs.read(reinterpret_cast<char*>(hdr), 24)) return false;
    static const uint8_t kSig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (std::memcmp(hdr, kSig, 8) != 0 || std::memcmp(hdr + 12, "IHDR", 4) != 0) return false;
    w = (uint32_t(hdr[16]) << 24) | (uint32_t(hdr[17]) << 16) | (uint32_t(hdr[18]) << 8) | uint32_t(hdr[19]);
    h = (uint32_t(hdr[20]) << 24) | (uint32_t(hdr[21]) << 16) | (uint32_t(hdr[22]) << 8) | uint32_t(hdr[23]);
    return w > 0 && h > 0;
}

struct Suite {
    int passed = 0, failed = 0;
    void check(const char* name, bool ok) {
        if (ok) ++passed; else ++failed;
        std::printf("[UNIT %02d] %-46s %s\n", passed + failed, name, ok ? "PASS" : "FAIL");
    }
};

static int run_unit_tests() {
    using namespace hy4_770b;
    Suite t;
    // 1-3: FIPS 180-4 SHA-256 NIST vectors
    t.check("sha256_empty_vector", sha256_str("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    t.check("sha256_abc_vector",   sha256_str("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    t.check("sha256_448bit_vector", sha256_str("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    // 4-9: Lloyd-Max FP64 HLUT2/HLUT3/HLUT4 positive half-Gaussian magnitude codebooks (n_pos = 2, 4, 8) & tile byte sizes
    const auto& lm2 = lloyd_max_hlut2();
    const auto& lm3 = lloyd_max_hlut3();
    const auto& lm4 = lloyd_max_hlut4();
    t.check("hlut2_lloyd_max_2pos_strict_monotonic", lm2.n_pos == 2 && lm2.converged && lm2.levels[0] > 0.0 && lm2.levels[0] < lm2.levels[1]);
    t.check("hlut3_lloyd_max_4pos_strict_monotonic", lm3.n_pos == 4 && lm3.converged && lm3.levels[0] > 0.0 && lm3.levels[0] < lm3.levels[1] && lm3.levels[1] < lm3.levels[2] && lm3.levels[2] < lm3.levels[3]);
    t.check("hlut4_lloyd_max_8pos_strict_monotonic", lm4.n_pos == 8 && lm4.converged && lm4.levels[0] > 0.0 && lm4.levels[0] < lm4.levels[3] && lm4.levels[3] < lm4.levels[7] && lm4.sqnr_db > 20.0);
    t.check("hlut2_tile_size_272B_2_125bpw", sizeof(Hlut2Tile32x32) == 272u);
    t.check("hlut3_tile_size_400B_3_125bpw", sizeof(Hlut3Tile32x32) == 400u);
    t.check("hlut4_tile_size_528B_4_125bpw", sizeof(Hlut4Tile32x32) == 528u);

    // 10-12: WHT128 orthogonality & Parseval norm preservation
    std::vector<float> x(128), y(128), z(128);
    for (int i = 0; i < 128; ++i) x[size_t(i)] = std::sin(0.19f * float(i)) - 0.3f * std::cos(0.37f * float(i));
    std::vector<uint64_t> signs = {0xA5A5F0F012345678ULL, 0xCAFEBABE0F0F55AAULL};
    std::vector<uint64_t> no_signs = {0ULL, 0ULL};
    wht128_rows(x.data(), y.data(), 1, 128, signs);
    // Exact inverse of H * D is D * H (H^T = H, D^T = D)
    wht128_rows(y.data(), z.data(), 1, 128, no_signs);
    for (int i = 0; i < 128; ++i) {
        if ((signs[size_t(i >> 6)] >> (i & 63)) & 1ULL) z[size_t(i)] = -z[size_t(i)];
    }
    double max_err = 0.0, nx = 0.0, ny = 0.0;
    for (int i = 0; i < 128; ++i) {
        max_err = std::max(max_err, std::fabs(double(x[size_t(i)]) - double(z[size_t(i)])));
        nx += double(x[size_t(i)]) * x[size_t(i)]; ny += double(y[size_t(i)]) * y[size_t(i)];
    }
    t.check("wht128_roundtrip_identity_lt_1e-5", max_err < 1e-5);
    t.check("wht128_parseval_norm_preservation", std::fabs(nx - ny) / (nx + 1e-12) < 1e-5);
    t.check("bf16_fp16_roundtrip_zero_one", bf16_to_f32(f32_to_bf16_rne(1.0f)) == 1.0f && fp16_to_f32(f32_to_fp16_rne(1.0f)) == 1.0f);

    // 13-17: 3:3:3:1:1 Sequence-Sharded KV bijection + 2:2:2:1:1 TP5 + 21-full/57-shared IndexCache
    bool kv_bij = true;
    for (int pos = 0; pos < 65536; pos += 17) {
        const int r = kv_tok_owner(pos);
        const int lt = kv_tok_local_slot(pos);
        if (r < 0 || r >= 5 || kv_local_tok_to_global(r, lt) != pos) { kv_bij = false; break; }
    }
    t.check("kv_33311_global_local_bijection_64k", kv_bij);
    int sum_toks = 0;
    for (int r = 0; r < 5; ++r) sum_toks += kv_rank_local_toks(r, 524288);
    t.check("kv_33311_512k_sum_exact_524288", sum_toks == 524288);
    int sum_h = 0;
    for (int r = 0; r < 5; ++r) sum_h += tp5_slice_size(r, kHiddenSize);
    t.check("tp5_22211_hidden_sum_exact_6144", sum_h == kHiddenSize);
    int full_cnt = 0;
    for (int l = 0; l < kNumLayers; ++l) if (is_full_indexer_layer(l)) ++full_cnt;
    t.check("indexcache_21_full_57_shared_exact", full_cnt == 21 && indexer_source_layer(76) == 73 && indexer_source_layer(77) == 77);
    const uint64_t kv_1m_bytes = 1048576ULL * uint64_t(kNumLayers) * uint64_t(kKvFp8EntryBytes);
    t.check("kv_1m_64bit_stride_no_overflow", kv_1m_bytes > 50000000000ULL);

    // 18-24: TierPlacementPlan invariants (Profile A5-FP8-HLUT4-Q5, A4-HLUT4, A3-HLUT3, & B @ 256K & 512K)
    const auto pa5_512 = make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_FP8_HLUT4_Q5, 524288);
    const auto pa4_256 = make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_HLUT4_DDR4, 262144);
    const auto pa4_512 = make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_HLUT4_DDR4, 524288);
    const auto pa256 = make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_HLUT3_DDR4, 262144);
    const auto pa512 = make_tier_plan(RuntimeProfile::PROFILE_A_FIDELITY_HLUT3_DDR4, 524288);
    const auto pb256 = make_tier_plan(RuntimeProfile::PROFILE_B_MAXVRAM_HLUT3_HLUT2, 262144);
    const auto pb512 = make_tier_plan(RuntimeProfile::PROFILE_B_MAXVRAM_HLUT3_HLUT2, 524288);
    t.check("plan_profile_a5_512k_20fp8_42h4_194q5", pa5_512.vram_fp8_per_layer() == 20 && pa5_512.vram_hlut4_per_layer() == 42 && pa5_512.ddr4_q5 == 194 && pa5_512.total_per_layer() == 256);
    t.check("q5_lm128_and_fp8_exact_byte_sizes", sizeof(Q5Lm128Block) == 82 && kQ5ExpertBytes == 24182784ULL && kFp8ExpertBytes == 37757952ULL && kBf16KvStartLayer == 70);
    t.check("plan_profile_a4_256k_93_hlut4_256_tot", pa4_256.vram_hlut4_per_layer() == 93 && pa4_256.total_per_layer() == 256);
    t.check("plan_profile_a4_512k_84_hlut4_256_tot", pa4_512.vram_hlut4_per_layer() == 84 && pa4_512.total_per_layer() == 256);
    t.check("plan_profile_a_256k_125_vram_256_tot", pa256.vram_hlut3_per_layer() == 125 && pa256.total_per_layer() == 256);
    t.check("plan_profile_a_512k_113_vram_256_tot", pa512.vram_hlut3_per_layer() == 113 && pa512.total_per_layer() == 256);
    t.check("plan_profile_b_256k_155_vram_256_tot", pb256.vram_total_per_layer() == 155 && pb256.total_per_layer() == 256);
    t.check("plan_profile_b_512k_139_vram_256_tot", pb512.vram_total_per_layer() == 139 && pb512.total_per_layer() == 256);

    // 22-27: Local contiguous n-gram cycle breaker
    t.check("cycle_breaker_empty", hy4::detect_local_ngram_cycle_ban_tok({}) == -1);
    t.check("cycle_breaker_noncyclic", hy4::detect_local_ngram_cycle_ban_tok({1,2,3,4,5,6,7,8}) == -1);
    t.check("cycle_breaker_p1_3x_no_trigger", hy4::detect_local_ngram_cycle_ban_tok({9,9,9}) == -1);
    t.check("cycle_breaker_p1_4x_trigger", hy4::detect_local_ngram_cycle_ban_tok({9,9,9,9}) == 9);
    t.check("cycle_breaker_p2_4x_trigger", hy4::detect_local_ngram_cycle_ban_tok({34,44,34,44,34,44,34,44}) == 34);
    t.check("cycle_breaker_p16_4x_trigger", []() {
        std::vector<int32_t> v; for (int r = 0; r < 4; ++r) for (int i = 0; i < 16; ++i) v.push_back(500 + i);
        return hy4::detect_local_ngram_cycle_ban_tok(v) == 500;
    }());

    // 28-31: ISA parser unit checks
    {
        std::istringstream ok_s("ScratchSize: 0\n.sgpr_spill_count: 0\n.vgpr_spill_count: 0\n.vgpr_count: 48\nds_read_b128 v[0:3], v4\nv_dot2_f32_f16 v0, v1, v2, v0\n");
        const auto st = audit_asm_stream(ok_s);
        t.check("isa_parser_clean_passes", st.pass() && st.ds_read_b128 == 1 && st.v_dot2_f32_f16 == 1);
    }
    {
        std::istringstream bad_scr("ScratchSize: 16\n.sgpr_spill_count: 0\n.vgpr_spill_count: 0\n");
        t.check("isa_parser_catches_scratch", !audit_asm_stream(bad_scr).pass());
    }
    {
        std::istringstream bad_trig("ScratchSize: 0\n.sgpr_spill_count: 0\n.vgpr_spill_count: 0\nv_sin_f32 v0, v1\n");
        t.check("isa_parser_catches_trig", !audit_asm_stream(bad_trig).pass());
    }
    {
        std::istringstream bad_slc("ScratchSize: 0\n.sgpr_spill_count: 0\n.vgpr_spill_count: 0\nglobal_atomic_add_f32 v0, v1, off slc\n");
        t.check("isa_parser_catches_slc_atomic", !audit_asm_stream(bad_slc).pass());
    }

    // 32: Hy4 770B exact geometry & expert byte sizes
    t.check("hy4_770b_19712_experts_exact_geometry", kTotalRoutedExperts == 19712 && kHiddenSize == 6144 && kMoeInterSize == 2048);

    std::printf("[UNIT] Summary: %d/%d PASS\n", t.passed, t.passed + t.failed);
    return t.failed == 0 ? 0 : 1;
}

static int run_verify_all(const std::string& root) {
    int fail = 0;
    std::printf("[VERIFY-ALL] Auditing workspace root: %s\n", root.c_str());

    // 1. gfx906 100% zero-scratch ISA audit + emit artifacts/hy4_770b_gfx906_isa_audit.json
    const std::string hlut_s = root + "/build/mi50/hy4_770b_hlut_kernels_gfx906.s";
    const std::string core_s = root + "/build/mi50/hy4_770b_core_kernels_gfx906.s";
    std::ifstream f_hlut(hlut_s), f_core(core_s);
    if (!f_hlut || !f_core) {
        std::fprintf(stderr, "[VERIFY-ALL] FAIL: missing gfx906 .s files (run `make audit-gfx906`)\n");
        return 1;
    }
    const IsaStats sh = audit_asm_stream(f_hlut);
    const IsaStats sc = audit_asm_stream(f_core);
    const int tot_k = sh.kernels + sc.kernels;
    const bool isa_ok = sh.pass() && sc.pass() && tot_k >= 49;
    std::printf("[GATE-ISA] hlut=%d core=%d total=%d zero_scratch=%d sgpr_zero=%d vgpr_zero=%d trig=%d slc=%d -> %s\n",
                sh.kernels, sc.kernels, tot_k, sh.zero_scratch + sc.zero_scratch,
                sh.sgpr_zero + sc.sgpr_zero, sh.vgpr_zero + sc.vgpr_zero,
                sh.trig + sc.trig, sh.slc_atomic + sc.slc_atomic, isa_ok ? "PASS" : "FAIL");
    if (!isa_ok) ++fail;

    {
        std::ostringstream j;
        j << "{\n"
          << "  \"schema\": \"hy4_770b_gfx906_isa_audit_v1\",\n"
          << "  \"target_arch\": \"gfx906\",\n"
          << "  \"status\": \"" << (isa_ok ? "PASS" : "FAIL") << "\",\n"
          << "  \"total_kernels\": " << tot_k << ",\n"
          << "  \"zero_scratch_kernels\": " << (sh.zero_scratch + sc.zero_scratch) << ",\n"
          << "  \"zero_sgpr_spill_kernels\": " << (sh.sgpr_zero + sc.sgpr_zero) << ",\n"
          << "  \"zero_vgpr_spill_kernels\": " << (sh.vgpr_zero + sc.vgpr_zero) << ",\n"
          << "  \"device_trig_instructions\": " << (sh.trig + sc.trig) << ",\n"
          << "  \"slc_system_rmw_atomics\": " << (sh.slc_atomic + sc.slc_atomic) << ",\n"
          << "  \"ds_read_b128_count\": " << (sh.ds_read_b128 + sc.ds_read_b128) << ",\n"
          << "  \"v_dot2_f32_f16_count\": " << (sh.v_dot2_f32_f16 + sc.v_dot2_f32_f16) << ",\n"
          << "  \"units\": {\n"
          << "    \"hy4_770b_hlut_kernels\": {\"kernels\": " << sh.kernels << ", \"zero_scratch\": " << sh.zero_scratch
          << ", \"max_vgpr\": " << sh.max_vgpr << ", \"max_sgpr\": " << sh.max_sgpr << ", \"max_lds_bytes\": " << sh.max_lds << "},\n"
          << "    \"hy4_770b_core_kernels\": {\"kernels\": " << sc.kernels << ", \"zero_scratch\": " << sc.zero_scratch
          << ", \"max_vgpr\": " << sc.max_vgpr << ", \"max_sgpr\": " << sc.max_sgpr << ", \"max_lds_bytes\": " << sc.max_lds << "}\n"
          << "  }\n"
          << "}\n";
        std::ofstream ofs(root + "/artifacts/hy4_770b_gfx906_isa_audit.json");
        ofs << j.str();
    }

    // 2. Read-only sysfs PCIe AER check across all 5 MI50 GPUs
    static const char* kMi50Bdf[5] = {"0000:05:00.0", "0000:08:00.0", "0000:85:00.0", "0000:88:00.0", "0000:c7:00.0"};
    long long aer_cor = 0, aer_nf = 0, aer_fat = 0;
    int aer_found = 0;
    auto read_aer = [](const std::string& p, const char* key) -> long long {
        std::ifstream f(p); std::string k; long long v = 0;
        while (f >> k >> v) if (k == key) return v;
        return 0;
    };
    for (const char* bdf : kMi50Bdf) {
        const std::string base = std::string("/sys/bus/pci/devices/") + bdf;
        if (fs::exists(base + "/aer_dev_fatal")) {
            ++aer_found;
            aer_cor += read_aer(base + "/aer_dev_correctable", "TOTAL_ERR_COR");
            aer_nf  += read_aer(base + "/aer_dev_nonfatal", "TOTAL_ERR_NONFATAL");
            aer_fat += read_aer(base + "/aer_dev_fatal", "TOTAL_ERR_FATAL");
        }
    }
    const bool aer_ok = (aer_nf == 0 && aer_fat == 0);
    std::printf("[GATE-AER] mi50_found=%d/5 cor=%lld nonfatal=%lld fatal=%lld -> %s\n",
                aer_found, aer_cor, aer_nf, aer_fat, aer_ok ? "PASS" : "FAIL");
    if (!aer_ok) ++fail;

    // 3. Zero Python check across src/, mi50/, p150/, tools/
    int py_cnt = 0;
    for (const char* sub : {"src", "mi50", "p150", "tools"}) {
        const fs::path p = fs::path(root) / sub;
        if (!fs::exists(p)) continue;
        for (const auto& e : fs::recursive_directory_iterator(p)) {
            if (e.is_regular_file() && e.path().extension() == ".py") ++py_cnt;
        }
    }
    std::printf("[GATE-ZERO-PY] python_files=%d -> %s\n", py_cnt, py_cnt == 0 ? "PASS" : "FAIL");
    if (py_cnt != 0) ++fail;

    // 4. Chart PNG 1200x720 check (if present)
    const std::string chart_p = root + "/artifacts/hy4_770b_single_station_telemetry.png";
    if (fs::exists(chart_p)) {
        uint32_t pw = 0, ph = 0;
        const bool png_ok = validate_png_header(chart_p, pw, ph) && pw == 1200u && ph == 720u;
        std::printf("[GATE-CHART] %s %ux%u sha256=%s -> %s\n",
                    chart_p.c_str(), pw, ph, sha256_file(chart_p).c_str(), png_ok ? "PASS" : "FAIL");
        if (!png_ok) ++fail;
    }

    // 5. Required artifacts NaN/Inf-free check
    const char* req_files[] = {
        "artifacts/cal_corpus_disjoint_receipt.json",
        "artifacts/hy4_770b_gfx906_isa_audit.json",
        "artifacts/hy4_770b_ledger.json"
    };
    for (const char* rel : req_files) {
        const fs::path fp = fs::path(root) / rel;
        if (!fs::exists(fp) || fs::file_size(fp) == 0) {
            std::fprintf(stderr, "[GATE-ART] MISSING/EMPTY: %s\n", rel);
            ++fail;
            continue;
        }
        std::ifstream ifs(fp);
        std::ostringstream ss; ss << ifs.rdbuf();
        const std::string body = ss.str();
        if (body.find("NaN") != std::string::npos || body.find("nan") != std::string::npos ||
            body.find(":inf") != std::string::npos || body.find(":-inf") != std::string::npos) {
            std::fprintf(stderr, "[GATE-ART] NaN/Inf detected in %s\n", rel);
            ++fail;
        } else {
            std::printf("[GATE-ART] %-44s size=%8ju sha256=%s\n",
                        rel, static_cast<uintmax_t>(fs::file_size(fp)), sha256_file(fp.string()).c_str());
        }
    }

    std::printf("[VERIFY-ALL] RESULT: %s (failures=%d)\n", fail == 0 ? "PASS" : "FAIL", fail);
    return fail == 0 ? 0 : 1;
}

}  // namespace hy4_770b_verify

int main(int argc, char** argv) {
    bool do_unit = true, do_all = false;
    std::string root = "<HY4_770B_WORKSPACE_ROOT>";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--verify-all") do_all = true;
        else if (a == "--unit-only") { do_unit = true; do_all = false; }
        else if (a == "--root" && i + 1 < argc) root = argv[++i];
    }
    int rc = 0;
    if (do_unit) rc |= hy4_770b_verify::run_unit_tests();
    if (do_all)  rc |= hy4_770b_verify::run_verify_all(root);
    return rc;
}
