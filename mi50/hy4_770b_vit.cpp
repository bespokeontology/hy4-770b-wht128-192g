// Pure-C++17 / rocBLAS Native 27-Layer SigLIP2 Vision Encoder for tencent/Hy4-flash
// Zero Python anywhere. Strict -Wall -Wextra -Werror clean.

#include "hy4_770b_vit.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <zlib.h>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace hy4_770b {
namespace {

inline constexpr int kVitDim         = kVitHiddenSize;
inline constexpr int kVitHeads       = kVitNumHeads;
inline constexpr int kVitInter       = kVitInterSize;
inline constexpr int kVitPatch       = kVitPatchSize;
inline constexpr int kVitMerge       = kVitSpatialMerge;
inline constexpr int kVitPosGrid     = 128;  // 16385 = 1 + 128*128
inline constexpr int kVitPercMid     = 6144;
inline constexpr float kVitLnEps     = 1e-6f;
inline constexpr float kVitRmsEps    = 1e-5f;
inline constexpr double kVitRope2dTheta = 10000.0;

inline float vit_bf16_to_f32(uint16_t b) noexcept {
    uint32_t u = uint32_t(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t vit_f32_to_bf16(float f) noexcept {
    uint32_t u = 0u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u) {
        return uint16_t((u >> 16) | 0x0040u);
    }
    u += 0x7FFFu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}

inline uint16_t vit_f32_to_f16(float f) noexcept {
    uint32_t x = 0u;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t abs_x = x & 0x7FFFFFFFu;
    if (abs_x >= 0x47800000u) {
        if (abs_x > 0x7F800000u) return uint16_t(sign | 0x7E00u);
        return uint16_t(sign | 0x7BFFu);  // clamp to finite FP16 max
    }
    if (abs_x < 0x38800000u) {
        if (abs_x < 0x33000000u) return uint16_t(sign);
        const uint32_t exp = abs_x >> 23;
        const uint32_t mant = (abs_x & 0x007FFFFFu) | 0x00800000u;
        const uint32_t shift = 126u - exp;
        return uint16_t(sign | ((mant + (1u << (shift - 1u))) >> shift));
    }
    const uint32_t round_x = abs_x + 0x00001000u;
    return uint16_t(sign | (((round_x >> 23) - 112u) << 10) | ((round_x >> 13) & 0x03FFu));
}

inline uint32_t read_be32(const uint8_t* p) noexcept {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

inline uint8_t paeth_predictor(int a, int b, int c) noexcept {
    const int p  = a + b - c;
    const int pa = std::abs(p - a);
    const int pb = std::abs(p - b);
    const int pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return uint8_t(a);
    if (pb <= pc) return uint8_t(b);
    return uint8_t(c);
}

bool decode_png(const uint8_t* data, size_t len, Hy4ImageRgb& out, std::string* err) {
    static constexpr uint8_t kSig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (len < 33 || std::memcmp(data, kSig, 8) != 0) return false;

    int w = 0, h = 0, bit_depth = 0, color_type = 0, interlace = 0;
    std::vector<uint8_t> idat;
    std::vector<uint8_t> plte;

    size_t pos = 8;
    while (pos + 12 <= len) {
        const uint32_t clen = read_be32(data + pos);
        const uint8_t* ctype = data + pos + 4;
        const uint8_t* cdata = data + pos + 8;
        if (pos + 12 + size_t(clen) > len) break;
        if (std::memcmp(ctype, "IHDR", 4) == 0 && clen >= 13) {
            w = int(read_be32(cdata + 0));
            h = int(read_be32(cdata + 4));
            bit_depth  = cdata[8];
            color_type = cdata[9];
            interlace  = cdata[12];
        } else if (std::memcmp(ctype, "PLTE", 4) == 0) {
            plte.assign(cdata, cdata + clen);
        } else if (std::memcmp(ctype, "IDAT", 4) == 0) {
            idat.insert(idat.end(), cdata, cdata + clen);
        } else if (std::memcmp(ctype, "IEND", 4) == 0) {
            break;
        }
        pos += 12 + size_t(clen);
    }

    if (w <= 0 || h <= 0 || w > 16384 || h > 16384 || bit_depth != 8 || interlace != 0 || idat.empty()) {
        if (err) *err = "Unsupported PNG parameters (require 8-bit non-interlaced PNG)";
        return false;
    }

    int bpp = 0;
    if (color_type == 0) bpp = 1;       // Gray
    else if (color_type == 2) bpp = 3;  // RGB
    else if (color_type == 3) bpp = 1;  // Palette
    else if (color_type == 4) bpp = 2;  // Gray+Alpha
    else if (color_type == 6) bpp = 4;  // RGBA
    else {
        if (err) *err = "Unsupported PNG color_type=" + std::to_string(color_type);
        return false;
    }

    const size_t stride = size_t(w) * size_t(bpp);
    const size_t raw_sz = size_t(h) * (stride + 1u);
    std::vector<uint8_t> raw(raw_sz);
    uLongf dest_len = uLongf(raw_sz);
    const int zrc = ::uncompress(raw.data(), &dest_len, idat.data(), uLong(idat.size()));
    if (zrc != Z_OK || dest_len < raw_sz) {
        if (err) *err = "PNG zlib uncompress failed (rc=" + std::to_string(zrc) + ")";
        return false;
    }

    std::vector<uint8_t> recon(size_t(h) * stride, 0);
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = raw.data() + size_t(y) * (stride + 1u);
        const uint8_t ftype = src[0];
        const uint8_t* in_row = src + 1;
        uint8_t* dst = recon.data() + size_t(y) * stride;
        const uint8_t* prev = (y > 0) ? (recon.data() + size_t(y - 1) * stride) : nullptr;
        for (size_t x = 0; x < stride; ++x) {
            const int a = (x >= size_t(bpp)) ? int(dst[x - size_t(bpp)]) : 0;
            const int b = prev ? int(prev[x]) : 0;
            const int c = (prev && x >= size_t(bpp)) ? int(prev[x - size_t(bpp)]) : 0;
            int v = int(in_row[x]);
            switch (ftype) {
                case 0: break;
                case 1: v = (v + a) & 0xFF; break;
                case 2: v = (v + b) & 0xFF; break;
                case 3: v = (v + ((a + b) >> 1)) & 0xFF; break;
                case 4: v = (v + int(paeth_predictor(a, b, c))) & 0xFF; break;
                default: break;
            }
            dst[x] = uint8_t(v);
        }
    }

    out.width = w;
    out.height = h;
    out.rgb.resize(size_t(w) * size_t(h) * 3u);
    for (int y = 0; y < h; ++y) {
        const uint8_t* srow = recon.data() + size_t(y) * stride;
        uint8_t* drow = out.rgb.data() + size_t(y) * size_t(w) * 3u;
        for (int x = 0; x < w; ++x) {
            if (color_type == 2) {
                drow[x * 3 + 0] = srow[x * 3 + 0];
                drow[x * 3 + 1] = srow[x * 3 + 1];
                drow[x * 3 + 2] = srow[x * 3 + 2];
            } else if (color_type == 6) {
                const int a = srow[x * 4 + 3];
                const int inv = 255 - a;
                drow[x * 3 + 0] = uint8_t((int(srow[x * 4 + 0]) * a + 255 * inv) / 255);
                drow[x * 3 + 1] = uint8_t((int(srow[x * 4 + 1]) * a + 255 * inv) / 255);
                drow[x * 3 + 2] = uint8_t((int(srow[x * 4 + 2]) * a + 255 * inv) / 255);
            } else if (color_type == 0) {
                drow[x * 3 + 0] = drow[x * 3 + 1] = drow[x * 3 + 2] = srow[x];
            } else if (color_type == 4) {
                const int g = srow[x * 2 + 0], a = srow[x * 2 + 1];
                const uint8_t v = uint8_t((g * a + 255 * (255 - a)) / 255);
                drow[x * 3 + 0] = drow[x * 3 + 1] = drow[x * 3 + 2] = v;
            } else if (color_type == 3) {
                const size_t pi = size_t(srow[x]) * 3u;
                if (pi + 2 < plte.size()) {
                    drow[x * 3 + 0] = plte[pi + 0];
                    drow[x * 3 + 1] = plte[pi + 1];
                    drow[x * 3 + 2] = plte[pi + 2];
                } else {
                    drow[x * 3 + 0] = drow[x * 3 + 1] = drow[x * 3 + 2] = 0;
                }
            }
        }
    }
    return true;
}

bool decode_ppm(const uint8_t* data, size_t len, Hy4ImageRgb& out) {
    if (len < 11 || data[0] != 'P' || data[1] != '6') return false;
    size_t p = 2;
    auto skip_ws = [&]() {
        while (p < len) {
            if (data[p] == '#') {
                while (p < len && data[p] != '\n') ++p;
            } else if (data[p] <= ' ') {
                ++p;
            } else {
                break;
            }
        }
    };
    auto read_int = [&]() -> int {
        skip_ws();
        int v = 0;
        while (p < len && data[p] >= '0' && data[p] <= '9') {
            v = v * 10 + int(data[p++] - '0');
        }
        return v;
    };
    const int w = read_int();
    const int h = read_int();
    const int mv = read_int();
    if (w <= 0 || h <= 0 || mv != 255 || p >= len) return false;
    ++p;  // single whitespace byte
    const size_t need = size_t(w) * size_t(h) * 3u;
    if (p + need > len) return false;
    out.width = w;
    out.height = h;
    out.rgb.assign(data + p, data + p + need);
    return true;
}

std::vector<uint8_t> decode_base64(const std::string& in) {
    static int8_t tbl[256];
    static bool init = false;
    if (!init) {
        std::memset(tbl, -1, sizeof(tbl));
        for (int i = 0; i < 26; ++i) tbl['A' + i] = int8_t(i);
        for (int i = 0; i < 26; ++i) tbl['a' + i] = int8_t(26 + i);
        for (int i = 0; i < 10; ++i) tbl['0' + i] = int8_t(52 + i);
        tbl[int('+')] = 62;
        tbl[int('/')] = 63;
        init = true;
    }
    std::vector<uint8_t> out;
    out.reserve(in.size() * 3u / 4u);
    uint32_t val = 0u;
    int valb = -8;
    for (unsigned char c : in) {
        if (c == '=') break;
        const int8_t d = tbl[c];
        if (d < 0) continue;
        val = (val << 6) | uint32_t(d);
        valb += 6;
        if (valb >= 0) {
            out.push_back(uint8_t((val >> valb) & 0xFFu));
            valb -= 8;
        }
    }
    return out;
}

void compute_smart_resize(int h, int w, int& out_h, int& out_w) {
    constexpr int kFactor = kVitPatch * kVitMerge;  // 32
    constexpr double kMinPix = 4096.0;
    constexpr double kMaxPix = 1048576.0;  // cap at 1024x1024 (~1024 visual tokens) for fast responsive GUI + crisp chart OCR!
    int hb = std::max(kFactor, int(std::round(double(h) / kFactor)) * kFactor);
    int wb = std::max(kFactor, int(std::round(double(w) / kFactor)) * kFactor);
    if (double(hb) * double(wb) > kMaxPix) {
        const double beta = std::sqrt((double(h) * double(w)) / kMaxPix);
        hb = std::max(kFactor, int(std::floor(double(h) / beta / kFactor)) * kFactor);
        wb = std::max(kFactor, int(std::floor(double(w) / beta / kFactor)) * kFactor);
    } else if (double(hb) * double(wb) < kMinPix) {
        const double beta = std::sqrt(kMinPix / std::max(1.0, double(h) * double(w)));
        hb = std::max(kFactor, int(std::ceil(double(h) * beta / kFactor)) * kFactor);
        wb = std::max(kFactor, int(std::ceil(double(w) * beta / kFactor)) * kFactor);
    }
    out_h = std::min(2048, hb);
    out_w = std::min(2048, wb);
}

inline float gelu_exact(float x) noexcept {
    return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f));
}

inline void check_hip_v(hipError_t e, const char* msg) {
    if (e != hipSuccess) throw std::runtime_error(std::string("Hy4Vit: ") + msg + ": " + hipGetErrorString(e));
}

inline void check_rb_v(rocblas_status s, const char* msg) {
    if (s != rocblas_status_success) {
        throw std::runtime_error(std::string("Hy4Vit: ") + msg + ": rocblas=" + std::to_string(int(s)));
    }
}

// Compute C[M, N] = A[M, K] * B[N, K]^T + bias[N] on MI50 rocBLAS (FP16 inputs, FP32 accumulate -> 2x H2D + 26.8 TFLOPS!)
struct VitGpuScratch {
    int dev = 0;
    hipStream_t stream = nullptr;
    rocblas_handle rb = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    uint16_t* d_a16 = nullptr;
    uint16_t* d_b16 = nullptr;
    size_t cap_a = 0, cap_b = 0, cap_c = 0;
    size_t cap_a16 = 0, cap_b16 = 0;
    std::vector<uint16_t> h_a16_tmp;
    std::vector<uint16_t> h_b16_tmp;

    explicit VitGpuScratch(int hip_dev) : dev(hip_dev) {
        check_hip_v(hipSetDevice(dev), "set dev");
        check_hip_v(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
        rocblas_initialize();
        check_rb_v(rocblas_create_handle(&rb), "rb create");
        check_rb_v(rocblas_set_atomics_mode(rb, rocblas_atomics_not_allowed), "rb no atomics");
        check_rb_v(rocblas_set_stream(rb, stream), "rb stream");
    }
    ~VitGpuScratch() {
        (void)hipSetDevice(dev);
        if (stream) (void)hipStreamSynchronize(stream);
        if (d_a) (void)hipFree(d_a);
        if (d_b) (void)hipFree(d_b);
        if (d_c) (void)hipFree(d_c);
        if (d_a16) (void)hipFree(d_a16);
        if (d_b16) (void)hipFree(d_b16);
        if (rb) (void)rocblas_destroy_handle(rb);
        if (stream) (void)hipStreamDestroy(stream);
    }

    template <typename T>
    void ensure_t(T*& ptr, size_t& cap, size_t need) {
        if (need <= cap) return;
        if (ptr) { (void)hipFree(ptr); ptr = nullptr; cap = 0; }
        check_hip_v(hipMalloc(reinterpret_cast<void**>(&ptr), need * sizeof(T)), "alloc");
        cap = need;
    }

    void ensure(float*& ptr, size_t& cap, size_t need) { ensure_t(ptr, cap, need); }

    void linear_bf16_w(
        const float* h_a, int M, int K,
        const uint16_t* w_bf16, const uint16_t* b_bf16, int N,
        std::vector<float>& /*h_w_tmp*/, std::vector<float>& h_out)
    {
        const size_t an = size_t(M) * size_t(K);
        const size_t wn = size_t(N) * size_t(K);
        if (h_a16_tmp.size() < an) h_a16_tmp.resize(an);
        if (h_b16_tmp.size() < wn) h_b16_tmp.resize(wn);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < an; ++i) h_a16_tmp[i] = vit_f32_to_f16(h_a[i]);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < wn; ++i) h_b16_tmp[i] = vit_f32_to_f16(vit_bf16_to_f32(w_bf16[i]));

        ensure_t(d_a16, cap_a16, an);
        ensure_t(d_b16, cap_b16, wn);
        ensure(d_c, cap_c, size_t(M) * size_t(N));

        check_hip_v(hipMemcpyAsync(d_a16, h_a16_tmp.data(), an * 2u, hipMemcpyHostToDevice, stream), "a16 h2d");
        check_hip_v(hipMemcpyAsync(d_b16, h_b16_tmp.data(), wn * 2u, hipMemcpyHostToDevice, stream), "b16 h2d");
        const float one = 1.0f, zero = 0.0f;
        check_rb_v(rocblas_gemm_ex(
            rb, rocblas_operation_transpose, rocblas_operation_none,
            N, M, K, &one,
            d_b16, rocblas_datatype_f16_r, K,
            d_a16, rocblas_datatype_f16_r, K,
            &zero,
            d_c, rocblas_datatype_f32_r, N,
            d_c, rocblas_datatype_f32_r, N,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0), "vit_gemm_f16");
        h_out.resize(size_t(M) * size_t(N));
        check_hip_v(hipMemcpyAsync(h_out.data(), d_c, size_t(M) * size_t(N) * 4u, hipMemcpyDeviceToHost, stream), "c d2h");
        check_hip_v(hipStreamSynchronize(stream), "sync");

        if (b_bf16) {
            std::vector<float> hb(static_cast<size_t>(N));
            for (int j = 0; j < N; ++j) hb[size_t(j)] = vit_bf16_to_f32(b_bf16[j]);
            #pragma omp parallel for schedule(static)
            for (int m = 0; m < M; ++m) {
                float* row = h_out.data() + size_t(m) * size_t(N);
                for (int j = 0; j < N; ++j) row[j] += hb[size_t(j)];
            }
        }
    }
};

void layer_norm_f32(
    const float* x, const uint16_t* w_bf16, const uint16_t* b_bf16,
    int N, int D, float eps, std::vector<float>& out)
{
    out.resize(size_t(N) * size_t(D));
    std::vector<float> hw(static_cast<size_t>(D));
    std::vector<float> hb(static_cast<size_t>(D));
    for (int d = 0; d < D; ++d) {
        hw[size_t(d)] = vit_bf16_to_f32(w_bf16[d]);
        hb[size_t(d)] = vit_bf16_to_f32(b_bf16[d]);
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const float* xi = x + size_t(i) * size_t(D);
        float* yi = out.data() + size_t(i) * size_t(D);
        double sum = 0.0;
        for (int d = 0; d < D; ++d) sum += double(xi[d]);
        const float mean = float(sum / double(D));
        double var = 0.0;
        for (int d = 0; d < D; ++d) {
            const float diff = xi[d] - mean;
            var += double(diff) * double(diff);
        }
        const float inv = 1.0f / std::sqrt(float(var / double(D)) + eps);
        for (int d = 0; d < D; ++d) {
            yi[d] = (xi[d] - mean) * inv * hw[size_t(d)] + hb[size_t(d)];
        }
    }
}

void rms_norm_f32(
    const float* x, const uint16_t* w_bf16,
    int N, int D, float eps, std::vector<float>& out)
{
    out.resize(size_t(N) * size_t(D));
    std::vector<float> hw(static_cast<size_t>(D));
    for (int d = 0; d < D; ++d) hw[size_t(d)] = vit_bf16_to_f32(w_bf16[d]);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const float* xi = x + size_t(i) * size_t(D);
        float* yi = out.data() + size_t(i) * size_t(D);
        double sq = 0.0;
        for (int d = 0; d < D; ++d) sq += double(xi[d]) * double(xi[d]);
        const float inv = 1.0f / std::sqrt(float(sq / double(D)) + eps);
        for (int d = 0; d < D; ++d) yi[d] = xi[d] * inv * hw[size_t(d)];
    }
}

void apply_rope_2d(
    float* q, float* k, int grid_h, int grid_w,
    const std::vector<float>& cos_y, const std::vector<float>& sin_y,
    const std::vector<float>& cos_x, const std::vector<float>& sin_x)
{
    constexpr int kHalf = kVitHeadDim / 2;   // 36 (y: 0..35, x: 36..71)
    constexpr int kQtr  = kHalf / 2;         // 18 pairs each
    const int N = grid_h * grid_w;
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < N; ++idx) {
        const int py = idx / grid_w;
        const int px = idx - py * grid_w;
        const float* cy = cos_y.data() + size_t(py) * kQtr;
        const float* sy = sin_y.data() + size_t(py) * kQtr;
        const float* cx = cos_x.data() + size_t(px) * kQtr;
        const float* sx = sin_x.data() + size_t(px) * kQtr;
        for (int h = 0; h < kVitHeads; ++h) {
            auto rot = [&](float* v) {
                for (int i = 0; i < kQtr; ++i) {
                    const float y0 = v[i], y1 = v[i + kQtr];
                    v[i]        = y0 * cy[i] - y1 * sy[i];
                    v[i + kQtr] = y0 * sy[i] + y1 * cy[i];
                    const float x0 = v[kHalf + i], x1 = v[kHalf + i + kQtr];
                    v[kHalf + i]        = x0 * cx[i] - x1 * sx[i];
                    v[kHalf + i + kQtr] = x0 * sx[i] + x1 * cx[i];
                }
            };
            rot(q + (size_t(idx) * kVitHeads + size_t(h)) * kVitHeadDim);
            rot(k + (size_t(idx) * kVitHeads + size_t(h)) * kVitHeadDim);
        }
    }
}

}  // namespace

bool decode_image_bytes(const uint8_t* data, size_t len, Hy4ImageRgb& out, std::string* err) {
    if (!data || len < 8) {
        if (err) *err = "Empty image bytes";
        return false;
    }
    if (data[0] == 137 && data[1] == 80 && data[2] == 78 && data[3] == 71) {
        return decode_png(data, len, out, err);
    }
    if (data[0] == 'P' && data[1] == '6') {
        return decode_ppm(data, len, out);
    }
    if (err) *err = "Unsupported image format (expected PNG or PPM P6)";
    return false;
}

bool load_image_from_url_or_path(const std::string& url_or_path, Hy4ImageRgb& out, std::string* err) {
    if (url_or_path.rfind("data:", 0) == 0) {
        const size_t comma = url_or_path.find(',');
        if (comma == std::string::npos) {
            if (err) *err = "Malformed data URI";
            return false;
        }
        const auto raw = decode_base64(url_or_path.substr(comma + 1));
        return decode_image_bytes(raw.data(), raw.size(), out, err);
    }
    std::string path = url_or_path;
    if (path.rfind("file://", 0) == 0) path = path.substr(7);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (err) *err = "Cannot open image file: " + path;
        return false;
    }
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return decode_image_bytes(raw.data(), raw.size(), out, err);
}

struct Hy4770bVitEncoder::Impl {
    std::unique_ptr<hy4::SafetensorsMmap> st;
    int hip_dev = 0;
};

Hy4770bVitEncoder::Hy4770bVitEncoder() : impl_(std::make_unique<Impl>()) {}
Hy4770bVitEncoder::~Hy4770bVitEncoder() = default;

bool Hy4770bVitEncoder::init(const std::string& vit_safetensors_path, int hip_dev) {
    if (!std::filesystem::exists(vit_safetensors_path)) return false;
    impl_->st = std::make_unique<hy4::SafetensorsMmap>(vit_safetensors_path);
    impl_->hip_dev = hip_dev;
    return true;
}

bool Hy4770bVitEncoder::ready() const noexcept {
    return impl_ && impl_->st != nullptr;
}

Hy4VitOutput Hy4770bVitEncoder::encode_image(const Hy4ImageRgb& img) const {
    if (!ready()) throw std::runtime_error("Hy4770bVitEncoder: not initialized");
    if (img.width <= 0 || img.height <= 0 || img.rgb.size() < size_t(img.width) * size_t(img.height) * 3u) {
        throw std::runtime_error("Hy4770bVitEncoder: invalid RGB image");
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto& st = *impl_->st;

    // 1. Dynamic aspect-preserving bilinear resize to exact multiples of 32 + normalize (pixel/255 - 0.5)/0.5
    int rh = 0, rw = 0;
    compute_smart_resize(img.height, img.width, rh, rw);
    const int gh = rh / kVitPatch;
    const int gw = rw / kVitPatch;
    const int N  = gh * gw;

    std::vector<float> norm_rgb(size_t(rh) * size_t(rw) * 3u);
    const float sy_scale = float(img.height) / float(rh);
    const float sx_scale = float(img.width)  / float(rw);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < rh; ++y) {
        const float fy = std::clamp((float(y) + 0.5f) * sy_scale - 0.5f, 0.0f, float(img.height - 1));
        const int y0 = int(std::floor(fy));
        const int y1 = std::min(img.height - 1, y0 + 1);
        const float wy = fy - float(y0);
        for (int x = 0; x < rw; ++x) {
            const float fx = std::clamp((float(x) + 0.5f) * sx_scale - 0.5f, 0.0f, float(img.width - 1));
            const int x0 = int(std::floor(fx));
            const int x1 = std::min(img.width - 1, x0 + 1);
            const float wx = fx - float(x0);
            for (int c = 0; c < 3; ++c) {
                const float p00 = float(img.rgb[(size_t(y0) * size_t(img.width) + size_t(x0)) * 3u + size_t(c)]);
                const float p01 = float(img.rgb[(size_t(y0) * size_t(img.width) + size_t(x1)) * 3u + size_t(c)]);
                const float p10 = float(img.rgb[(size_t(y1) * size_t(img.width) + size_t(x0)) * 3u + size_t(c)]);
                const float p11 = float(img.rgb[(size_t(y1) * size_t(img.width) + size_t(x1)) * 3u + size_t(c)]);
                const float v = (p00 * (1.0f - wx) + p01 * wx) * (1.0f - wy) +
                                (p10 * (1.0f - wx) + p11 * wx) * wy;
                norm_rgb[(size_t(y) * size_t(rw) + size_t(x)) * 3u + size_t(c)] = v * (2.0f / 255.0f) - 1.0f;
            }
        }
    }

    // 2. Patch Im2col [N, 3*16*16=768] in CHW order matching patch_embedding.weight [1152, 3, 16, 16]
    std::vector<float> patches(size_t(N) * 768u);
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < N; ++idx) {
        const int py = idx / gw;
        const int px = idx - py * gw;
        float* dst = patches.data() + size_t(idx) * 768u;
        for (int c = 0; c < 3; ++c) {
            for (int dy = 0; dy < kVitPatch; ++dy) {
                const int iy = py * kVitPatch + dy;
                for (int dx = 0; dx < kVitPatch; ++dx) {
                    const int ix = px * kVitPatch + dx;
                    dst[(c * kVitPatch + dy) * kVitPatch + dx] =
                        norm_rgb[(size_t(iy) * size_t(rw) + size_t(ix)) * 3u + size_t(c)];
                }
            }
        }
    }

    VitGpuScratch gpu(impl_->hip_dev);
    std::vector<float> h_w_tmp, x, x_ln, q, k, v, attn_out, proj_out, mlp_h;

    gpu.linear_bf16_w(
        patches.data(), N, 768,
        reinterpret_cast<const uint16_t*>(st.tensor_bytes("vit.embeddings.patch_embedding.weight")),
        reinterpret_cast<const uint16_t*>(st.tensor_bytes("vit.embeddings.patch_embedding.bias")),
        kVitDim, h_w_tmp, x);

    // 3. Add bilinearly interpolated 128x128 position_embedding (rows [1..16384], row 0 is CLS!)
    {
        const uint16_t* pos_grid = reinterpret_cast<const uint16_t*>(
            st.tensor_bytes("vit.embeddings.position_embedding.weight")) + size_t(kVitDim);  // skip row 0!
        const float py_sc = float(kVitPosGrid) / float(gh);
        const float px_sc = float(kVitPosGrid) / float(gw);
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < N; ++idx) {
            const int py = idx / gw;
            const int px = idx - py * gw;
            const float fy = std::clamp((float(py) + 0.5f) * py_sc - 0.5f, 0.0f, float(kVitPosGrid - 1));
            const float fx = std::clamp((float(px) + 0.5f) * px_sc - 0.5f, 0.0f, float(kVitPosGrid - 1));
            const int y0 = int(std::floor(fy)), y1 = std::min(kVitPosGrid - 1, y0 + 1);
            const int x0 = int(std::floor(fx)), x1 = std::min(kVitPosGrid - 1, x0 + 1);
            const float wy = fy - float(y0), wx = fx - float(x0);
            const float w00 = (1.0f - wy) * (1.0f - wx), w01 = (1.0f - wy) * wx;
            const float w10 = wy * (1.0f - wx),          w11 = wy * wx;
            const uint16_t* r00 = pos_grid + (size_t(y0) * kVitPosGrid + size_t(x0)) * kVitDim;
            const uint16_t* r01 = pos_grid + (size_t(y0) * kVitPosGrid + size_t(x1)) * kVitDim;
            const uint16_t* r10 = pos_grid + (size_t(y1) * kVitPosGrid + size_t(x0)) * kVitDim;
            const uint16_t* r11 = pos_grid + (size_t(y1) * kVitPosGrid + size_t(x1)) * kVitDim;
            float* dst = x.data() + size_t(idx) * kVitDim;
            for (int d = 0; d < kVitDim; ++d) {
                dst[d] += w00 * vit_bf16_to_f32(r00[d]) + w01 * vit_bf16_to_f32(r01[d]) +
                          w10 * vit_bf16_to_f32(r10[d]) + w11 * vit_bf16_to_f32(r11[d]);
            }
        }
    }

    // 4. Precompute 2D RoPE cos/sin tables (theta=10000.0, 18 freq pairs for Y and 18 for X -> 72 head_dim)
    constexpr int kQtr = kVitHeadDim / 4;  // 18
    std::vector<float> cos_y(size_t(gh) * kQtr), sin_y(size_t(gh) * kQtr);
    std::vector<float> cos_x(size_t(gw) * kQtr), sin_x(size_t(gw) * kQtr);
    double inv_f[kQtr];
    for (int i = 0; i < kQtr; ++i) inv_f[i] = 1.0 / std::pow(kVitRope2dTheta, double(2 * i) / double(kVitHeadDim / 2));
    for (int py = 0; py < gh; ++py) {
        for (int i = 0; i < kQtr; ++i) {
            const double ang = double(py) * inv_f[i];
            cos_y[size_t(py) * kQtr + size_t(i)] = float(std::cos(ang));
            sin_y[size_t(py) * kQtr + size_t(i)] = float(std::sin(ang));
        }
    }
    for (int px = 0; px < gw; ++px) {
        for (int i = 0; i < kQtr; ++i) {
            const double ang = double(px) * inv_f[i];
            cos_x[size_t(px) * kQtr + size_t(i)] = float(std::cos(ang));
            sin_x[size_t(px) * kQtr + size_t(i)] = float(std::sin(ang));
        }
    }

    // 5. Run all 27 SigLIP2 ViT Transformer Blocks (streamed to MI50 rocBLAS FP32!)
    std::vector<float> q_h(size_t(kVitHeads) * size_t(N) * kVitHeadDim);
    std::vector<float> k_h(size_t(kVitHeads) * size_t(N) * kVitHeadDim);
    std::vector<float> v_h(size_t(kVitHeads) * size_t(N) * kVitHeadDim);
    std::vector<float> scores(size_t(N) * size_t(N));
    std::vector<float> head_out(size_t(N) * kVitHeadDim);
    const float attn_scale = 1.0f / std::sqrt(float(kVitHeadDim));
    const float one = 1.0f, zero = 0.0f;

    for (int l = 0; l < kVitNumLayers; ++l) {
        const std::string p = "vit.layers." + std::to_string(l) + ".";
        layer_norm_f32(
            x.data(),
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "input_layernorm.weight")),
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "input_layernorm.bias")),
            N, kVitDim, kVitLnEps, x_ln);

        gpu.linear_bf16_w(x_ln.data(), N, kVitDim,
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "self_attn.q_proj.weight")),
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "self_attn.q_proj.bias")),
            kVitDim, h_w_tmp, q);
        gpu.linear_bf16_w(x_ln.data(), N, kVitDim,
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "self_attn.k_proj.weight")),
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "self_attn.k_proj.bias")),
            kVitDim, h_w_tmp, k);
        gpu.linear_bf16_w(x_ln.data(), N, kVitDim,
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "self_attn.v_proj.weight")),
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "self_attn.v_proj.bias")),
            kVitDim, h_w_tmp, v);

        apply_rope_2d(q.data(), k.data(), gh, gw, cos_y, sin_y, cos_x, sin_x);

        // Transpose [N, H, Dh] -> [H, N, Dh]
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < kVitHeads; ++h) {
            float* qd = q_h.data() + size_t(h) * size_t(N) * kVitHeadDim;
            float* kd = k_h.data() + size_t(h) * size_t(N) * kVitHeadDim;
            float* vd = v_h.data() + size_t(h) * size_t(N) * kVitHeadDim;
            for (int i = 0; i < N; ++i) {
                const size_t s = (size_t(i) * kVitHeads + size_t(h)) * kVitHeadDim;
                std::memcpy(qd + size_t(i) * kVitHeadDim, q.data() + s, kVitHeadDim * 4u);
                std::memcpy(kd + size_t(i) * kVitHeadDim, k.data() + s, kVitHeadDim * 4u);
                std::memcpy(vd + size_t(i) * kVitHeadDim, v.data() + s, kVitHeadDim * 4u);
            }
        }

        // Exact bidirectional Multi-Head SDPA on MI50 rocBLAS (batched across heads, bounded <= 64 MiB transient VRAM!)
        attn_out.resize(size_t(N) * kVitDim);
        const int h_blk = std::max(1, std::min(kVitHeads, int((16u * 1024u * 1024u) / std::max<size_t>(1u, size_t(N) * size_t(N)))));
        scores.resize(size_t(h_blk) * size_t(N) * size_t(N));
        head_out.resize(size_t(h_blk) * size_t(N) * kVitHeadDim);
        gpu.ensure(gpu.d_a, gpu.cap_a, size_t(h_blk) * size_t(N) * kVitHeadDim);
        gpu.ensure(gpu.d_b, gpu.cap_b, size_t(h_blk) * size_t(N) * kVitHeadDim);
        gpu.ensure(gpu.d_c, gpu.cap_c, size_t(h_blk) * size_t(N) * size_t(N));
        const rocblas_stride st_qkv = rocblas_stride(size_t(N) * kVitHeadDim);
        const rocblas_stride st_sc  = rocblas_stride(size_t(N) * size_t(N));
        for (int h0 = 0; h0 < kVitHeads; h0 += h_blk) {
            const int hc = std::min(h_blk, kVitHeads - h0);
            const float* qd = q_h.data() + size_t(h0) * size_t(N) * kVitHeadDim;
            const float* kd = k_h.data() + size_t(h0) * size_t(N) * kVitHeadDim;
            const float* vd = v_h.data() + size_t(h0) * size_t(N) * kVitHeadDim;
            check_hip_v(hipMemcpyAsync(gpu.d_a, qd, size_t(hc) * size_t(N) * kVitHeadDim * 4u, hipMemcpyHostToDevice, gpu.stream), "sdpa q");
            check_hip_v(hipMemcpyAsync(gpu.d_b, kd, size_t(hc) * size_t(N) * kVitHeadDim * 4u, hipMemcpyHostToDevice, gpu.stream), "sdpa k");
            check_rb_v(rocblas_sgemm_strided_batched(
                gpu.rb, rocblas_operation_transpose, rocblas_operation_none,
                N, N, kVitHeadDim, &attn_scale,
                gpu.d_b, kVitHeadDim, st_qkv,
                gpu.d_a, kVitHeadDim, st_qkv,
                &zero,
                gpu.d_c, N, st_sc, hc), "sdpa qk batched");
            check_hip_v(hipMemcpyAsync(scores.data(), gpu.d_c, size_t(hc) * size_t(N) * size_t(N) * 4u, hipMemcpyDeviceToHost, gpu.stream), "sdpa s d2h");
            check_hip_v(hipStreamSynchronize(gpu.stream), "sdpa s sync");

            #pragma omp parallel for schedule(static)
            for (int hi = 0; hi < hc * N; ++hi) {
                float* row = scores.data() + size_t(hi) * size_t(N);
                float mx = row[0];
                for (int j = 1; j < N; ++j) if (row[j] > mx) mx = row[j];
                double sm = 0.0;
                for (int j = 0; j < N; ++j) {
                    const float e = std::exp(row[j] - mx);
                    row[j] = e;
                    sm += double(e);
                }
                const float inv = float(1.0 / sm);
                for (int j = 0; j < N; ++j) row[j] *= inv;
            }

            check_hip_v(hipMemcpyAsync(gpu.d_c, scores.data(), size_t(hc) * size_t(N) * size_t(N) * 4u, hipMemcpyHostToDevice, gpu.stream), "sdpa p h2d");
            check_hip_v(hipMemcpyAsync(gpu.d_b, vd, size_t(hc) * size_t(N) * kVitHeadDim * 4u, hipMemcpyHostToDevice, gpu.stream), "sdpa v h2d");
            check_rb_v(rocblas_sgemm_strided_batched(
                gpu.rb, rocblas_operation_none, rocblas_operation_none,
                kVitHeadDim, N, N, &one,
                gpu.d_b, kVitHeadDim, st_qkv,
                gpu.d_c, N, st_sc,
                &zero,
                gpu.d_a, kVitHeadDim, st_qkv, hc), "sdpa pv batched");
            check_hip_v(hipMemcpyAsync(head_out.data(), gpu.d_a, size_t(hc) * size_t(N) * kVitHeadDim * 4u, hipMemcpyDeviceToHost, gpu.stream), "sdpa o d2h");
            check_hip_v(hipStreamSynchronize(gpu.stream), "sdpa o sync");

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                for (int hb = 0; hb < hc; ++hb) {
                    std::memcpy(attn_out.data() + (size_t(i) * kVitHeads + size_t(h0 + hb)) * kVitHeadDim,
                                head_out.data() + (size_t(hb) * size_t(N) + size_t(i)) * kVitHeadDim, kVitHeadDim * 4u);
                }
            }
        }

        gpu.linear_bf16_w(attn_out.data(), N, kVitDim,
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "self_attn.o_proj.weight")),
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "self_attn.o_proj.bias")),
            kVitDim, h_w_tmp, proj_out);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < x.size(); ++i) x[i] += proj_out[i];

        layer_norm_f32(
            x.data(),
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "post_attention_layernorm.weight")),
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "post_attention_layernorm.bias")),
            N, kVitDim, kVitLnEps, x_ln);
        gpu.linear_bf16_w(x_ln.data(), N, kVitDim,
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "mlp.dense_h_to_4h.weight")),
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "mlp.dense_h_to_4h.bias")),
            kVitInter, h_w_tmp, mlp_h);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < mlp_h.size(); ++i) mlp_h[i] = gelu_exact(mlp_h[i]);
        gpu.linear_bf16_w(mlp_h.data(), N, kVitInter,
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "mlp.dense_4h_to_h.weight")),
            reinterpret_cast<const uint16_t*>(st.tensor_bytes(p + "mlp.dense_4h_to_h.bias")),
            kVitDim, h_w_tmp, proj_out);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < x.size(); ++i) x[i] += proj_out[i];
    }

    // 6. Perceive Spatial 2x2 Merger (RMSNorm -> Conv2d(2x2, s=2) -> GELU -> Conv2d(1x1) -> append image_newline/row!)
    rms_norm_f32(
        x.data(),
        reinterpret_cast<const uint16_t*>(st.tensor_bytes("vit.perceive.norm.weight")),
        N, kVitDim, kVitRmsEps, x_ln);

    const int mh = gh / kVitMerge;
    const int mw = gw / kVitMerge;
    const int M_m = mh * mw;
    // Gather 2x2 patch blocks in [C=1152, kh=2, kw=2] order matching Conv2d weight [4608, 1152, 2, 2]
    std::vector<float> perc_in(size_t(M_m) * kVitPercMid);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M_m; ++m) {
        const int my = m / mw;
        const int mx = m - my * mw;
        const float* p00 = x_ln.data() + (size_t(my * 2 + 0) * size_t(gw) + size_t(mx * 2 + 0)) * kVitDim;
        const float* p01 = x_ln.data() + (size_t(my * 2 + 0) * size_t(gw) + size_t(mx * 2 + 1)) * kVitDim;
        const float* p10 = x_ln.data() + (size_t(my * 2 + 1) * size_t(gw) + size_t(mx * 2 + 0)) * kVitDim;
        const float* p11 = x_ln.data() + (size_t(my * 2 + 1) * size_t(gw) + size_t(mx * 2 + 1)) * kVitDim;
        float* dst = perc_in.data() + size_t(m) * kVitPercMid;
        for (int c = 0; c < kVitDim; ++c) {
            dst[c * 4 + 0] = p00[c];
            dst[c * 4 + 1] = p01[c];
            dst[c * 4 + 2] = p10[c];
            dst[c * 4 + 3] = p11[c];
        }
    }

    std::vector<float> perc_mid, perc_out;
    gpu.linear_bf16_w(perc_in.data(), M_m, kVitPercMid,
        reinterpret_cast<const uint16_t*>(st.tensor_bytes("vit.perceive.proj.0.weight")),
        reinterpret_cast<const uint16_t*>(st.tensor_bytes("vit.perceive.proj.0.bias")),
        kVitPercMid, h_w_tmp, perc_mid);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < perc_mid.size(); ++i) perc_mid[i] = gelu_exact(perc_mid[i]);
    gpu.linear_bf16_w(perc_mid.data(), M_m, kVitPercMid,
        reinterpret_cast<const uint16_t*>(st.tensor_bytes("vit.perceive.proj.2.weight")),
        reinterpret_cast<const uint16_t*>(st.tensor_bytes("vit.perceive.proj.2.bias")),
        kHiddenSize, h_w_tmp, perc_out);

    const uint16_t* img_nl = reinterpret_cast<const uint16_t*>(st.tensor_bytes("vit.perceive.image_newline"));
    Hy4VitOutput out{};
    out.grid_h = gh;
    out.grid_w = gw;
    out.merged_h = mh;
    out.merged_w = mw;
    out.num_tokens = mh * (mw + 1);
    out.embeddings_bf16.resize(size_t(out.num_tokens) * kHiddenSize);
    for (int my = 0; my < mh; ++my) {
        const size_t row_dst = size_t(my) * size_t(mw + 1);
        for (int mx = 0; mx < mw; ++mx) {
            const float* src = perc_out.data() + (size_t(my) * size_t(mw) + size_t(mx)) * kHiddenSize;
            uint16_t* dst = out.embeddings_bf16.data() + (row_dst + size_t(mx)) * kHiddenSize;
            for (int d = 0; d < kHiddenSize; ++d) dst[d] = vit_f32_to_bf16(src[d]);
        }
        std::memcpy(out.embeddings_bf16.data() + (row_dst + size_t(mw)) * kHiddenSize,
                    img_nl, size_t(kHiddenSize) * 2u);
    }
    out.encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace hy4_770b
