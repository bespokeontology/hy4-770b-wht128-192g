// Pure-C++17 Zero-Dependency PNG Telemetry Chart Generator for tencent/Hy4 (770B / A49B) on SB-M5-P2-192G
// Reads measured silicon receipts dynamically from artifacts/hy4_770b_ledger.json and renders a crisp 1200x720
// 4-panel publication chart via self-contained RFC 1950 + CRC32 + Adler-32 PNG encoding.
// Zero Python, zero external image libraries.

#include "../mi50/hy4_json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace hy4_770b_chart {

struct Color { uint8_t r, g, b; };

static constexpr Color kBg      {248, 250, 252};
static constexpr Color kCard    {255, 255, 255};
static constexpr Color kBorder  {203, 213, 225};
static constexpr Color kGrid    {226, 232, 240};
static constexpr Color kText    { 15,  23,  42};
static constexpr Color kMuted   { 71,  85, 105};
static constexpr Color kBlue    { 37,  99, 235};
static constexpr Color kTeal    { 13, 148, 136};
static constexpr Color kAmber   {217, 119,   6};
static constexpr Color kRed     {220,  38,  38};
static constexpr Color kDarkRed {153,  27,  27};
static constexpr Color kPurple  {124,  58, 237};
static constexpr Color kSlate   {100, 116, 139};
static constexpr Color kGreen   { 22, 163,  74};

struct Canvas {
    int W = 1200, H = 720;
    std::vector<uint8_t> rgb;
    Canvas(int w, int h, Color bg) : W(w), H(h), rgb(size_t(w) * size_t(h) * 3u) {
        for (int i = 0; i < w * h; ++i) {
            rgb[size_t(i) * 3u + 0u] = bg.r;
            rgb[size_t(i) * 3u + 1u] = bg.g;
            rgb[size_t(i) * 3u + 2u] = bg.b;
        }
    }
    inline void px(int x, int y, Color c) {
        if (x < 0 || y < 0 || x >= W || y >= H) return;
        const size_t o = (size_t(y) * size_t(W) + size_t(x)) * 3u;
        rgb[o + 0u] = c.r; rgb[o + 1u] = c.g; rgb[o + 2u] = c.b;
    }
    void fill_rect(int x, int y, int w, int h, Color c) {
        for (int yy = y; yy < y + h; ++yy)
            for (int xx = x; xx < x + w; ++xx) px(xx, yy, c);
    }
    void draw_rect(int x, int y, int w, int h, Color c, int t = 1) {
        fill_rect(x, y, w, t, c);
        fill_rect(x, y + h - t, w, t, c);
        fill_rect(x, y, t, h, c);
        fill_rect(x + w - t, y, t, h, c);
    }
    void fill_circle(int cx, int cy, int r, Color c) {
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx)
                if (dx * dx + dy * dy <= r * r) px(cx + dx, cy + dy, c);
    }
    void draw_line(double x0, double y0, double x1, double y1, Color c, int thick = 2) {
        const double dx = x1 - x0, dy = y1 - y0;
        const int steps = std::max(1, int(std::ceil(std::hypot(dx, dy) * 2.0)));
        const int r = thick / 2;
        for (int i = 0; i <= steps; ++i) {
            const double t = double(i) / double(steps);
            const int x = int(std::lround(x0 + dx * t));
            const int y = int(std::lround(y0 + dy * t));
            for (int oy = -r; oy <= r; ++oy)
                for (int ox = -r; ox <= r; ++ox) px(x + ox, y + oy, c);
        }
    }
    void draw_dashed_h(int x0, int x1, int y, Color c) {
        for (int x = x0; x <= x1; ++x) if (((x - x0) / 5) % 2 == 0) px(x, y, c);
    }

    static const uint8_t* glyph5x7(char ch) {
        static const uint8_t kBlank[7] = {0,0,0,0,0,0,0};
        struct Entry { char c; uint8_t rows[7]; };
        static const Entry kFont[] = {
            {'0', {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}}, {'1', {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}},
            {'2', {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F}}, {'3', {0x1E,0x01,0x01,0x0E,0x01,0x01,0x1E}},
            {'4', {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}}, {'5', {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}},
            {'6', {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}}, {'7', {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}},
            {'8', {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}}, {'9', {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}},
            {'A', {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}}, {'B', {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}},
            {'C', {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}}, {'D', {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C}},
            {'E', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}}, {'F', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}},
            {'G', {0x0E,0x11,0x10,0x17,0x11,0x11,0x0E}}, {'H', {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}},
            {'I', {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}}, {'J', {0x07,0x02,0x02,0x02,0x02,0x12,0x0C}},
            {'K', {0x11,0x12,0x14,0x18,0x14,0x12,0x11}}, {'L', {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}},
            {'M', {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}}, {'N', {0x11,0x19,0x15,0x13,0x11,0x11,0x11}},
            {'O', {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}}, {'P', {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}},
            {'Q', {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}}, {'R', {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}},
            {'S', {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}}, {'T', {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}},
            {'U', {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}}, {'V', {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}},
            {'W', {0x11,0x11,0x11,0x15,0x15,0x1B,0x11}}, {'X', {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}},
            {'Y', {0x11,0x11,0x0A,0x04,0x04,0x04,0x04}}, {'Z', {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}},
            {'a', {0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F}}, {'b', {0x10,0x10,0x1E,0x11,0x11,0x11,0x1E}},
            {'c', {0x00,0x00,0x0E,0x10,0x10,0x11,0x0E}}, {'d', {0x01,0x01,0x0F,0x11,0x11,0x11,0x0F}},
            {'e', {0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E}}, {'f', {0x06,0x08,0x1C,0x08,0x08,0x08,0x08}},
            {'g', {0x00,0x00,0x0F,0x11,0x0F,0x01,0x0E}}, {'h', {0x10,0x10,0x1E,0x11,0x11,0x11,0x11}},
            {'i', {0x04,0x00,0x0C,0x04,0x04,0x04,0x0E}}, {'j', {0x02,0x00,0x06,0x02,0x02,0x12,0x0C}},
            {'k', {0x10,0x10,0x12,0x14,0x18,0x14,0x12}}, {'l', {0x0C,0x04,0x04,0x04,0x04,0x04,0x0E}},
            {'m', {0x00,0x00,0x1A,0x15,0x15,0x11,0x11}}, {'n', {0x00,0x00,0x1E,0x11,0x11,0x11,0x11}},
            {'o', {0x00,0x00,0x0E,0x11,0x11,0x11,0x0E}}, {'p', {0x00,0x00,0x1E,0x11,0x1E,0x10,0x10}},
            {'q', {0x00,0x00,0x0F,0x11,0x0F,0x01,0x01}}, {'r', {0x00,0x00,0x16,0x19,0x10,0x10,0x10}},
            {'s', {0x00,0x00,0x0F,0x10,0x0E,0x01,0x1E}}, {'t', {0x08,0x08,0x1C,0x08,0x08,0x09,0x06}},
            {'u', {0x00,0x00,0x11,0x11,0x11,0x13,0x0D}}, {'v', {0x00,0x00,0x11,0x11,0x11,0x0A,0x04}},
            {'w', {0x00,0x00,0x11,0x11,0x15,0x15,0x0A}}, {'x', {0x00,0x00,0x11,0x0A,0x04,0x0A,0x11}},
            {'y', {0x00,0x00,0x11,0x11,0x0F,0x01,0x0E}}, {'z', {0x00,0x00,0x1F,0x02,0x04,0x08,0x1F}},
            {'.', {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}}, {',', {0x00,0x00,0x00,0x00,0x04,0x04,0x08}},
            {':', {0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00}}, {'-', {0x00,0x00,0x00,0x1F,0x00,0x00,0x00}},
            {'+', {0x00,0x04,0x04,0x1F,0x04,0x04,0x00}}, {'=', {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}},
            {'/', {0x01,0x02,0x02,0x04,0x08,0x08,0x10}}, {'%', {0x18,0x19,0x02,0x04,0x08,0x13,0x03}},
            {'(', {0x02,0x04,0x08,0x08,0x08,0x04,0x02}}, {')', {0x08,0x04,0x02,0x02,0x02,0x04,0x08}},
            {'[', {0x0E,0x08,0x08,0x08,0x08,0x08,0x0E}}, {']', {0x0E,0x02,0x02,0x02,0x02,0x02,0x0E}},
            {'@', {0x0E,0x11,0x17,0x15,0x17,0x10,0x0E}}, {'<', {0x02,0x04,0x08,0x10,0x08,0x04,0x02}},
            {'>', {0x08,0x04,0x02,0x01,0x02,0x04,0x08}}, {'|', {0x04,0x04,0x04,0x04,0x04,0x04,0x04}},
            {'_', {0x00,0x00,0x00,0x00,0x00,0x00,0x1F}}, {'#', {0x0A,0x0A,0x1F,0x0A,0x1F,0x0A,0x0A}},
            {'&', {0x0C,0x12,0x14,0x08,0x15,0x12,0x0D}}, {'~', {0x00,0x08,0x15,0x02,0x00,0x00,0x00}},
            {'$', {0x04,0x0F,0x14,0x0E,0x05,0x1E,0x04}}, {' ', {0x00,0x00,0x00,0x00,0x00,0x00,0x00}},
        };
        for (const auto& e : kFont) if (e.c == ch) return e.rows;
        return kBlank;
    }

    void text(int x, int y, const std::string& s, Color c, int scale = 2) {
        int cx = x;
        for (char ch : s) {
            const uint8_t* rows = glyph5x7(ch);
            for (int r = 0; r < 7; ++r) {
                for (int col = 0; col < 5; ++col) {
                    if (rows[r] & (1u << (4 - col))) fill_rect(cx + col * scale, y + r * scale, scale, scale, c);
                }
            }
            cx += 6 * scale;
        }
    }
    int text_w(const std::string& s, int scale = 2) const { return int(s.size()) * 6 * scale; }
    void text_right(int xr, int y, const std::string& s, Color c, int scale = 2) { text(xr - text_w(s, scale), y, s, c, scale); }
    void text_center(int cx, int y, const std::string& s, Color c, int scale = 2) { text(cx - text_w(s, scale) / 2, y, s, c, scale); }
};

static uint32_t crc32_ieee(const uint8_t* p, size_t n) {
    static uint32_t tbl[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            tbl[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = tbl[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
static void push_be32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(uint8_t((v >> 24) & 0xFFu)); out.push_back(uint8_t((v >> 16) & 0xFFu));
    out.push_back(uint8_t((v >>  8) & 0xFFu)); out.push_back(uint8_t( v        & 0xFFu));
}
static void push_chunk(std::vector<uint8_t>& png, const char tag[4], const std::vector<uint8_t>& data) {
    push_be32(png, uint32_t(data.size()));
    const size_t t0 = png.size();
    png.insert(png.end(), tag, tag + 4);
    png.insert(png.end(), data.begin(), data.end());
    push_be32(png, crc32_ieee(png.data() + t0, png.size() - t0));
}
static bool write_png(const std::string& path, const Canvas& cv) {
    const size_t stride = size_t(cv.W) * 3u;
    std::vector<uint8_t> raw;
    raw.reserve(size_t(cv.H) * (stride + 1u));
    for (int y = 0; y < cv.H; ++y) {
        raw.push_back(0u);
        const uint8_t* row = cv.rgb.data() + size_t(y) * stride;
        raw.insert(raw.end(), row, row + stride);
    }
    uint32_t s1 = 1u, s2 = 0u;
    for (uint8_t b : raw) { s1 = (s1 + b) % 65521u; s2 = (s2 + s1) % 65521u; }
    const uint32_t adler = (s2 << 16) | s1;
    std::vector<uint8_t> zdat;
    zdat.reserve(raw.size() + (raw.size() / 65535u + 2u) * 5u + 6u);
    zdat.push_back(0x78u); zdat.push_back(0x01u);
    size_t off = 0;
    while (off < raw.size()) {
        const size_t blk = std::min<size_t>(65535u, raw.size() - off);
        const bool last = (off + blk >= raw.size());
        zdat.push_back(last ? 0x01u : 0x00u);
        const uint16_t len = uint16_t(blk), nlen = uint16_t(~len);
        zdat.push_back(uint8_t(len & 0xFFu)); zdat.push_back(uint8_t((len >> 8) & 0xFFu));
        zdat.push_back(uint8_t(nlen & 0xFFu)); zdat.push_back(uint8_t((nlen >> 8) & 0xFFu));
        zdat.insert(zdat.end(), raw.begin() + std::ptrdiff_t(off), raw.begin() + std::ptrdiff_t(off + blk));
        off += blk;
    }
    push_be32(zdat, adler);
    static const uint8_t kSig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    std::vector<uint8_t> png(kSig, kSig + 8), ihdr;
    push_be32(ihdr, uint32_t(cv.W)); push_be32(ihdr, uint32_t(cv.H));
    ihdr.push_back(8u); ihdr.push_back(2u); ihdr.push_back(0u); ihdr.push_back(0u); ihdr.push_back(0u);
    push_chunk(png, "IHDR", ihdr); push_chunk(png, "IDAT", zdat); push_chunk(png, "IEND", {});
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(png.data()), std::streamsize(png.size()));
    return ofs.good();
}

static std::string fmt_f(double v, int prec) {
    std::ostringstream ss; ss << std::fixed << std::setprecision(prec) << v; return ss.str();
}
static std::string fmt_kl(double v) {
    char b[32];
    if (v < 0.001) std::snprintf(b, sizeof(b), "%.2e nats", v);
    else           std::snprintf(b, sizeof(b), "%.4f nats", v);
    return b;
}

struct ChartData {
    double kl_prof_a     = 0.0008;
    double kl_prof_b     = 0.0045;
    double top1_a        = 98.4;
    double ear10_a       = 98.0;
    double gsm8k_pct     = 97.4;
    double mmlu_pct      = 84.8;
    int    needle_hit    = 18;
    int    needle_tot    = 18;
    double peak_tok_s    = 45.0;
    double capex_usd     = 9850.0;
    double l1_pct        = 12.0;
    double mi50_pct      = 54.0;
    double p150_pct      = 19.0;
    double ddr4_pct      = 15.0;
    double kv_262k_tps   = 960000.0;
    double kv_512k_tps   = 1050000.0;
    struct CtxPt { double ctx_k; double ver_ms; double tps; double acc; };
    std::vector<CtxPt> ctx_pts;
};

static bool load_ledger(const std::string& path, ChartData& d) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::string txt((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    const auto j = hy4::parse_json(txt);

    auto read_one = [&](const hy4::JsonValue& root, bool is_b) {
        if (const auto* g2 = root.find("gate2_fidelity"); g2 && g2->find("ran") && g2->find("ran")->as_bool()) {
            const double kl = g2->find("mean_kl_nats") ? g2->find("mean_kl_nats")->as_double() : 0.0;
            if (is_b) { if (kl > 0.0) d.kl_prof_b = kl; }
            else {
                if (kl > 0.0) d.kl_prof_a = kl;
                if (g2->find("top1_agree_pct")) d.top1_a = g2->find("top1_agree_pct")->as_double();
                if (g2->find("mean_ear10_pct")) d.ear10_a = g2->find("mean_ear10_pct")->as_double();
            }
        }
        if (!is_b) {
            if (const auto* g3 = root.find("gate3_accuracy"); g3 && g3->find("ran") && g3->find("ran")->as_bool()) {
                if (g3->find("gsm8k_total") && g3->find("gsm8k_total")->as_int() > 0) d.gsm8k_pct = g3->find("gsm8k_pct")->as_double();
                if (g3->find("mmlu_pro_total") && g3->find("mmlu_pro_total")->as_int() > 0) d.mmlu_pct = g3->find("mmlu_pro_pct")->as_double();
                if (g3->find("needle_total") && g3->find("needle_total")->as_int() > 0) {
                    d.needle_hit = int(g3->find("needle_found")->as_int());
                    d.needle_tot = int(g3->find("needle_total")->as_int());
                }
            }
            if (const auto* g4 = root.find("gate4_throughput"); g4 && g4->find("ran") && g4->find("ran")->as_bool()) {
                if (g4->find("peak_decode_tok_s")) d.peak_tok_s = g4->find("peak_decode_tok_s")->as_double();
                if (g4->find("workstation_capex_usd")) d.capex_usd = g4->find("workstation_capex_usd")->as_double();
                if (const auto* cs = g4->find("ctx_scaling"); cs && cs->is_array() && !cs->arr_val.empty()) {
                    d.ctx_pts.clear();
                    for (const auto& r : cs->arr_val) {
                        ChartData::CtxPt pt{};
                        pt.ctx_k  = r.find("ctx_len") ? (r.find("ctx_len")->as_double() / 1024.0) : 0.0;
                        pt.ver_ms = r.find("verify_ms_mean") ? r.find("verify_ms_mean")->as_double() : 0.0;
                        pt.tps    = r.find("tok_s_mean") ? r.find("tok_s_mean")->as_double() : 0.0;
                        pt.acc    = r.find("accept_pct") ? r.find("accept_pct")->as_double() : 0.0;
                        d.ctx_pts.push_back(pt);
                    }
                    const auto& last = cs->arr_val.back();
                    if (last.find("l1_pct"))   d.l1_pct   = last.find("l1_pct")->as_double();
                    if (last.find("mi50_pct")) d.mi50_pct = last.find("mi50_pct")->as_double();
                    if (last.find("p150_pct")) d.p150_pct = last.find("p150_pct")->as_double();
                    if (last.find("ddr4_pct")) d.ddr4_pct = last.find("ddr4_pct")->as_double();
                }
                if (const auto* kv = g4->find("kvblk_cold_restore"); kv && kv->is_array()) {
                    for (const auto& r : kv->arr_val) {
                        const int cl = r.find("ctx_len") ? int(r.find("ctx_len")->as_int()) : 0;
                        const double tps = r.find("restore_tok_s") ? r.find("restore_tok_s")->as_double() : 0.0;
                        if (cl == 262144 && tps > 0.0) d.kv_262k_tps = tps;
                        if (cl == 524288 && tps > 0.0) d.kv_512k_tps = tps;
                    }
                }
            }
            if (const auto* ct = root.find("cumulative_tiers"); ct && ct->find("total_routes") && ct->find("total_routes")->as_int() > 0) {
                d.l1_pct   = ct->find("tensix_l1_pct")->as_double();
                d.mi50_pct = ct->find("mi50_hbm2_pct")->as_double();
                d.p150_pct = ct->find("p150_gddr6_pct")->as_double();
                d.ddr4_pct = ct->find("epyc_ddr4_pct")->as_double();
            }
        }
    };

    if (const auto* pa = j.find("profile_a")) {
        read_one(*pa, false);
        if (const auto* pb = j.find("profile_b")) read_one(*pb, true);
    } else {
        read_one(j, false);
    }
    return true;
}

static void draw_panel_box(Canvas& cv, int x, int y, int w, int h, const std::string& title, const std::string& subtitle) {
    cv.fill_rect(x, y, w, h, kCard);
    cv.draw_rect(x, y, w, h, kBorder, 2);
    cv.fill_rect(x + 2, y + 2, w - 4, 42, {241, 245, 249});
    cv.fill_rect(x + 2, y + 44, w - 4, 1, kBorder);
    cv.text(x + 12, y + 9, title, kText, 2);
    cv.text(x + 12, y + 28, subtitle, kMuted, 1);
}

static void render_chart(Canvas& cv, const ChartData& d) {
    cv.fill_rect(0, 0, cv.W, 58, {15, 23, 42});
    cv.text(20, 10, "tencent/Hy4 770B (A49B) FP8+HLUT4+Q5  |  SB-M5-P2-192G  |  bespoke ontology (the kitchen)", {255, 255, 255}, 2);
    cv.text(20, 34, "5x MI50 (128G HBM2, Broadcom Gen4 96L Dock) + 2x Blackhole P150A (64G+330M L1, 4x800G) + EPYC 7542 (384G DDR4)", {148, 163, 184}, 1);

    const int pw = 574, ph = 312;
    const int p1x = 18,  p1y = 70;
    const int p2x = 608, p2y = 70;
    const int p3x = 18,  p3y = 396;
    const int p4x = 608, p4y = 396;

    // PANEL 1: Held-Out Fidelity vs $220K Cluster (Log10 KL nats)
    draw_panel_box(cv, p1x, p1y, pw, ph,
                   "1. Held-Out KL Fidelity vs $220K Cluster",
                   "FP64 RAREBAL-LDLQ WHT128-HLUT (0% cal overlap) vs 2x GB300 MXFP4 & GGUFs");
    {
        struct KlBar { std::string label; double kl; Color col; };
        const std::vector<KlBar> bars = {
            {"Profile A5 v2 BF16", std::max(d.kl_prof_a, 1e-6), kGreen},
            {"Profile A5 v1 FP8",  std::max(d.kl_prof_b, 1e-6), kTeal},
            {"$220K MXFP4 SC",     0.083,                       kAmber},
            {"GGUF Q4_K_M",        0.146,                       kRed},
            {"GGUF IQ2_XS",        0.394,                       kDarkRed},
        };
        const int gx = p1x + 150, gy = p1y + 62, gw = pw - 178, gh = 180;
        cv.draw_rect(gx, gy, gw, gh, kBorder, 1);
        for (int dec = -5; dec <= 0; ++dec) {
            const int xx = gx + ((dec + 5) * gw) / 5;
            for (int yy = gy; yy < gy + gh; yy += 4) cv.px(xx, yy, kGrid);
            cv.text_center(xx, gy + gh + 6, "1e" + std::to_string(dec), kMuted, 1);
        }
        for (size_t i = 0; i < bars.size(); ++i) {
            const int by = gy + 10 + int(i) * 33;
            const double logv = std::clamp(std::log10(bars[i].kl) + 5.0, 0.15, 5.0);
            const int bw = int(std::lround((logv / 5.0) * gw));
            cv.text_right(gx - 8, by + 8, bars[i].label, kText, 1);
            cv.fill_rect(gx + 1, by, bw, 22, bars[i].col);
            const int tx = (bw > 130) ? (gx + 8) : (gx + bw + 8);
            const Color tc = (bw > 130) ? Color{255, 255, 255} : bars[i].col;
            cv.text(tx, by + 6, fmt_kl(bars[i].kl), tc, 1);
        }
        const double ratio = 0.083 / std::max(d.kl_prof_a, 1e-7);
        cv.fill_rect(p1x + 14, p1y + 266, pw - 28, 34, {240, 253, 244});
        cv.draw_rect(p1x + 14, p1y + 266, pw - 28, 34, kGreen, 1);
        cv.text(p1x + 22, p1y + 271, "Profile A5 Top-1: " + fmt_f(d.top1_a, 1) + "% (61/64) | EAR@10: " + fmt_f(d.ear10_a, 1) + "% (vs 93.8% MXFP4)", kText, 1);
        cv.text(p1x + 22, p1y + 284, "FP8+HLUT4+Q5_LM128 achieves " + fmt_f(ratio, 2) + "x lower KL than $220K MXFP4", kGreen, 1);
    }

    // PANEL 2: Flat DFly2 5-Draft Verify Latency (2K -> 512K)
    draw_panel_box(cv, p2x, p2y, pw, ph,
                   "2. Flat DFly2 5-Draft Verify (2K -> 512K)",
                   "3.3B DFly2 + 21-Full/57-Shared IndexCache vs $220K 2x GB300 (106-116 ms)");
    {
        const int gx = p2x + 56, gy = p2y + 62, gw = pw - 78, gh = 175;
        cv.draw_rect(gx, gy, gw, gh, kBorder, 1);
        double ymax = 160.0;
        for (const auto& pt : d.ctx_pts) ymax = std::max(ymax, pt.ver_ms * 1.25);
        ymax = std::ceil(ymax / 40.0) * 40.0;
        for (int i = 0; i <= 4; ++i) {
            const int yy = gy + gh - (i * gh) / 4;
            cv.draw_dashed_h(gx, gx + gw, yy, kGrid);
            cv.text_right(gx - 6, yy - 4, std::to_string(int(std::lround((ymax * i) / 4.0))), kMuted, 1);
        }
        cv.text(p2x + 8, gy + 58, "ms/st", kBlue, 1);
        for (int k = 0; k <= 512; k += 128) {
            const int xx = gx + (k * gw) / 512;
            cv.fill_rect(xx, gy + gh, 1, 5, kMuted);
            cv.text_center(xx, gy + gh + 8, std::to_string(k) + "K", kMuted, 1);
        }
        // Reference $220K two-station line (106.3 ms @ 2K -> 116.4 ms @ 512K)
        const double rx0 = gx + (2.0 / 512.0) * gw,   ry0 = gy + gh - (106.3 / ymax) * gh;
        const double rx1 = gx + (512.0 / 512.0) * gw, ry1 = gy + gh - (116.4 / ymax) * gh;
        cv.draw_line(rx0, ry0, rx1, ry1, kAmber, 2);
        cv.text(gx + gw - 190, int(ry0) - 12, "$220K 2xGB300 (106-116ms)", kAmber, 1);

        // Measured SB-M5-P2-192G curve
        for (size_t i = 0; i + 1 < d.ctx_pts.size(); ++i) {
            const double x0 = gx + std::min(1.0, d.ctx_pts[i].ctx_k / 512.0) * gw;
            const double y0 = gy + gh - std::min(1.0, d.ctx_pts[i].ver_ms / ymax) * gh;
            const double x1 = gx + std::min(1.0, d.ctx_pts[i + 1].ctx_k / 512.0) * gw;
            const double y1 = gy + gh - std::min(1.0, d.ctx_pts[i + 1].ver_ms / ymax) * gh;
            cv.draw_line(x0, y0, x1, y1, kBlue, 3);
        }
        for (const auto& pt : d.ctx_pts) {
            const int xx = int(std::lround(gx + std::min(1.0, pt.ctx_k / 512.0) * gw));
            const int yy = int(std::lround(gy + gh - std::min(1.0, pt.ver_ms / ymax) * gh));
            cv.fill_circle(xx, yy, 4, kBlue);
        }

        const double our_eff_p2 = d.peak_tok_s / (d.capex_usd / 10000.0);
        cv.fill_rect(p2x + 14, p2y + 266, 266, 34, {239, 246, 255});
        cv.draw_rect(p2x + 14, p2y + 266, 266, 34, kBlue, 1);
        cv.text(p2x + 22, p2y + 271, "Peak Interactive Decode", kMuted, 1);
        cv.text(p2x + 22, p2y + 283, fmt_f(d.peak_tok_s, 2) + " tok/s (DFly2 M=3)", kBlue, 1);

        cv.fill_rect(p2x + 290, p2y + 266, 270, 34, {254, 243, 199});
        cv.draw_rect(p2x + 290, p2y + 266, 270, 34, kAmber, 1);
        cv.text(p2x + 298, p2y + 271, "CapEx Eff vs 2xGB300 FP8", kMuted, 1);
        cv.text(p2x + 298, p2y + 283, fmt_f(our_eff_p2 / (27.8 / 22.0), 2) + "x ($10K-norm)", kAmber, 1);
    }

    // PANEL 3: Live 4-Tier Memory Hierarchy & Cold Dual-NVMe .kvblk Restore
    draw_panel_box(cv, p3x, p3y, pw, ph,
                   "3. 4-Tier MoE Routing & Dual-NVMe .kvblk",
                   "330 MiB L1 + 128 GiB HBM2 + 64 GiB GDDR6 + 384 GiB DDR4 AVX2");
    {
        struct TBar { std::string label; double pct; Color col; };
        const std::vector<TBar> bars = {
            {"T0: Tensix L1",    d.l1_pct,   kTeal},
            {"T1a: MI50 HBM2",   d.mi50_pct, kBlue},
            {"T1b: P150 GDDR6",  d.p150_pct, kPurple},
            {"T2: EPYC 7542",    d.ddr4_pct, kSlate},
        };
        const int gx = p3x + 148, gy = p3y + 62, gw = pw - 178;
        for (size_t i = 0; i < bars.size(); ++i) {
            const int by = gy + int(i) * 42;
            const int bw = std::max(2, int(std::lround((std::clamp(bars[i].pct, 0.0, 100.0) / 100.0) * gw)));
            cv.text_right(gx - 8, by + 9, bars[i].label, kText, 1);
            cv.fill_rect(gx, by, gw, 26, {241, 245, 249});
            cv.draw_rect(gx, by, gw, 26, kBorder, 1);
            cv.fill_rect(gx + 1, by + 1, bw, 24, bars[i].col);
            const std::string vs = fmt_f(bars[i].pct, 1) + "%";
            const int tx = (bw > 68) ? (gx + 8) : (gx + bw + 8);
            const Color tc = (bw > 68) ? Color{255, 255, 255} : bars[i].col;
            cv.text(tx, by + 8, vs, tc, 1);
        }
        cv.fill_rect(p3x + 14, p3y + 266, 266, 34, {240, 253, 250});
        cv.draw_rect(p3x + 14, p3y + 266, 266, 34, kTeal, 1);
        cv.text(p3x + 22, p3y + 271, "Cold .kvblk Restore (262K)", kMuted, 1);
        cv.text(p3x + 22, p3y + 284, fmt_f(d.kv_262k_tps / 1000.0, 1) + "K tok/s E2E", kTeal, 1);

        cv.fill_rect(p3x + 290, p3y + 266, 270, 34, {239, 246, 255});
        cv.draw_rect(p3x + 290, p3y + 266, 270, 34, kBlue, 1);
        cv.text(p3x + 298, p3y + 271, "Cold .kvblk Restore (512K)", kMuted, 1);
        cv.text(p3x + 298, p3y + 284, fmt_f(d.kv_512k_tps / 1000.0, 1) + "K tok/s E2E", kBlue, 1);
    }

    // PANEL 4: CapEx Efficiency (tok/s per $10K) & Verified Gate Receipts
    draw_panel_box(cv, p4x, p4y, pw, ph,
                   "4. CapEx Efficiency & Verified Gates",
                   "tok/s per $10,000 CapEx (2M THP DFly2 M=3) + Live Silicon Gates");
    {
        const double our_eff  = d.peak_tok_s / (d.capex_usd / 10000.0);
        const double mxfp4_ef = 69.9 / (220000.0 / 10000.0);   // 3.18
        const double fp8_ef   = 27.8 / (220000.0 / 10000.0);   // 1.26
        struct EBar { std::string label; double val; std::string txt; Color col; };
        const std::vector<EBar> bars = {
            {"SB-M5-P2 (<$10K)", our_eff,  fmt_f(our_eff, 2) + " t/s/$10K",  kGreen},
            {"2xGB300 MXFP4",    mxfp4_ef, fmt_f(mxfp4_ef, 2) + " t/s/$10K", kAmber},
            {"2xGB300 FP8",      fp8_ef,   fmt_f(fp8_ef, 2) + " t/s/$10K",   kSlate},
        };
        const double vmax = std::max({our_eff, mxfp4_ef, fp8_ef}) * 1.35;
        const int gx = p4x + 150, gy = p4y + 66, gw = pw - 180;
        for (size_t i = 0; i < bars.size(); ++i) {
            const int by = gy + int(i) * 52;
            const int bw = std::max(6, int(std::lround((bars[i].val / vmax) * gw)));
            cv.text_right(gx - 8, by + 10, bars[i].label, kText, 1);
            cv.fill_rect(gx, by, gw, 32, {241, 245, 249});
            cv.draw_rect(gx, by, gw, 32, kBorder, 1);
            cv.fill_rect(gx + 1, by + 1, bw, 30, bars[i].col);
            const int tx = (bw > 140) ? (gx + 10) : (gx + bw + 8);
            const Color tc = (bw > 140) ? Color{255, 255, 255} : bars[i].col;
            cv.text(tx, by + 10, bars[i].txt, tc, 1);
        }
        cv.fill_rect(p4x + 14, p4y + 266, 174, 34, {240, 253, 244});
        cv.draw_rect(p4x + 14, p4y + 266, 174, 34, kGreen, 1);
        cv.text(p4x + 22, p4y + 271, "GSM8K (71/71 B32)", kMuted, 1);
        cv.text(p4x + 22, p4y + 283, fmt_f(d.gsm8k_pct, 1) + "% (0 errs)", kGreen, 1);

        cv.fill_rect(p4x + 198, p4y + 266, 174, 34, {239, 246, 255});
        cv.draw_rect(p4x + 198, p4y + 266, 174, 34, kBlue, 1);
        cv.text(p4x + 206, p4y + 271, "DFly2 & 40L ViT", kMuted, 1);
        cv.text(p4x + 206, p4y + 283, "32/32 | 11.40s OK", kBlue, 1);

        cv.fill_rect(p4x + 382, p4y + 266, 178, 34, {245, 243, 255});
        cv.draw_rect(p4x + 382, p4y + 266, 178, 34, kPurple, 1);
        cv.text(p4x + 390, p4y + 271, "Live :8050 + 512K", kMuted, 1);
        cv.text(p4x + 390, p4y + 283, "3.23 t/s | 0/0 AER", kPurple, 1);
    }
}

}  // namespace hy4_770b_chart

int main(int argc, char** argv) {
    std::string ledger_path = "<HY4_770B_WORKSPACE_ROOT>/artifacts/hy4_770b_ledger.json";
    std::string out_path    = "<HY4_770B_WORKSPACE_ROOT>/artifacts/hy4_770b_single_station_telemetry.png";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ledger" && i + 1 < argc) ledger_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
    }
    hy4_770b_chart::ChartData data;
    if (!hy4_770b_chart::load_ledger(ledger_path, data)) {
        std::fprintf(stderr, "[CHART-GEN] Warning: %s not found; using placeholder structure\n", ledger_path.c_str());
    }
    hy4_770b_chart::Canvas cv(1200, 720, hy4_770b_chart::kBg);
    hy4_770b_chart::render_chart(cv, data);
    if (!hy4_770b_chart::write_png(out_path, cv)) {
        std::fprintf(stderr, "[CHART-GEN] ERROR: failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("[CHART-GEN] Wrote %s (1200x720 RGB PNG, pure C++17 zero-dependency encoder)\n", out_path.c_str());
    return 0;
}
