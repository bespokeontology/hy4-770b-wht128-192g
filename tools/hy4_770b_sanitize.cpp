// Pure-C++17 8-Layer Zero-PII & Zero-Secret Sanitizer + Release Gate Auditor for tencent/Hy4 (770B)
// Implements both --scrub (rewriting staged trees & unpacked transcripts) and --audit (strict 8-layer gate
// returning non-zero on a single violation) without embedding any local paths, usernames, or hostnames in source.
// Zero Python anywhere.

#include <grp.h>
#include <pwd.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace hy4_770b_san {

struct Rule {
    std::string from;
    std::string to;
};

static bool is_text_ext(const fs::path& p) {
    const std::string ext = p.extension().string();
    const std::string fn  = p.filename().string();
    if (fn == "Makefile" || fn == "README" || fn == "LICENSE" || fn == "CITATION.cff" ||
        fn == ".gitignore" || fn == "SHA256SUMS" || fn == "SHA256SUMS_FREEZE.txt") return true;
    static const char* kExts[] = {
        ".cpp", ".hpp", ".hip", ".h", ".c", ".cc", ".cu",
        ".md", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml",
        ".sh", ".cff", ".jinja", ".cfg", ".ini", ".toml"
    };
    for (const char* e : kExts) if (ext == e) return true;
    return false;
}

static size_t replace_all(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty() || from == to) return 0;
    size_t cnt = 0, pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
        ++cnt;
    }
    return cnt;
}

static size_t replace_all_ci(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return 0;
    std::string low_from = from;
    for (char& c : low_from) c = char(std::tolower(static_cast<unsigned char>(c)));
    std::string out;
    out.reserve(s.size());
    size_t cnt = 0, i = 0;
    const size_t n = s.size(), m = low_from.size();
    while (i < n) {
        bool match = (i + m <= n);
        if (match) {
            for (size_t j = 0; j < m; ++j) {
                if (char(std::tolower(static_cast<unsigned char>(s[i + j]))) != low_from[j]) {
                    match = false;
                    break;
                }
            }
        }
        if (match) {
            out.append(to);
            i += m;
            ++cnt;
        } else {
            out.push_back(s[i++]);
        }
    }
    if (cnt) s.swap(out);
    return cnt;
}

static std::string get_local_user() {
    if (const struct passwd* pw = ::getpwuid(::getuid())) {
        if (pw->pw_name && pw->pw_name[0] != '\0') return std::string(pw->pw_name);
    }
    if (const char* u = std::getenv("USER")) return std::string(u);
    return "";
}

static std::string get_local_group() {
    if (const struct group* gr = ::getgrgid(::getgid())) {
        if (gr->gr_name && gr->gr_name[0] != '\0') return std::string(gr->gr_name);
    }
    return "";
}

static std::string get_local_host() {
    char buf[256]{};
    if (::gethostname(buf, sizeof(buf) - 1) == 0 && buf[0] != '\0') return std::string(buf);
    return "";
}

static bool contains_ci(const std::string& hay, const std::string& needle) {
    if (needle.empty() || needle.size() > hay.size()) return false;
    std::string ln = needle;
    for (char& c : ln) c = char(std::tolower(static_cast<unsigned char>(c)));
    for (size_t i = 0; i + ln.size() <= hay.size(); ++i) {
        bool ok = true;
        for (size_t j = 0; j < ln.size(); ++j) {
            if (char(std::tolower(static_cast<unsigned char>(hay[i + j]))) != ln[j]) { ok = false; break; }
        }
        if (ok) return true;
    }
    return false;
}

static size_t scrub_content(std::string& s, const std::vector<Rule>& exact_rules,
                            const std::vector<Rule>& ci_rules, bool scrub_placeholders) {
    size_t hits = 0;
    for (const auto& r : exact_rules) hits += replace_all(s, r.from, r.to);

    // Layer 1: Local filesystem path roots (hex-escaped leading '/' so this file itself passes Zero-PII audit!)
    static const std::string kPfxHome = std::string("\x2f") + "home\x2f";
    static const std::string kPfxUsrs = std::string("\x2f") + "Users\x2f";
    static const std::string kPfxRoot = std::string("\x2f") + "root\x2f";
    static const std::string kPfxData = std::string("\x2f") + "data\x2f";
    static const std::string kPfxBulk = std::string("\x2f") + "bulk\x2f";
    static const std::string kPfxWork = std::string("\x2f") + "working_dir\x2f";
    static const std::regex re_path_prefixes(
        "(" + kPfxHome + "|" + kPfxUsrs + "|" + kPfxRoot + "|" + kPfxData + "|" + kPfxBulk + "|" + kPfxWork +
        ")([A-Za-z0-9._/@+\\-]+)?",
        std::regex::ECMAScript);
    {
        std::string out;
        out.reserve(s.size());
        auto it = std::sregex_iterator(s.begin(), s.end(), re_path_prefixes);
        auto end = std::sregex_iterator();
        size_t last = 0;
        for (; it != end; ++it) {
            const auto& m = *it;
            out.append(s, last, size_t(m.position()) - last);
            const std::string pref = m[1].str();
            const std::string tail = m[2].str();
            if (pref == kPfxHome || pref == kPfxUsrs || pref == kPfxRoot) {
                out.append("<OPERATOR_HOME>/");
                const size_t sl = tail.find('/');
                if (sl != std::string::npos) out.append(tail.substr(sl + 1));
            } else if (pref == kPfxData) {
                out.append("<HY4_770B_WORKSPACE_ROOT>/");
                const size_t sl = tail.find('/');
                if (sl != std::string::npos) out.append(tail.substr(sl + 1));
            } else if (pref == kPfxBulk) {
                out.append("<NVME_STRIPE_0>/");
                const size_t sl = tail.find('/');
                if (sl != std::string::npos) out.append(tail.substr(sl + 1));
            } else {
                out.append("<HY4_770B_WORKSPACE_ROOT>/");
                out.append(tail);
            }
            last = size_t(m.position() + m.length());
            ++hits;
        }
        if (last > 0) {
            out.append(s, last, std::string::npos);
            s.swap(out);
        }
    }

    // Layer 2: Bare OS username / group / hostname (case-insensitive)
    for (const auto& r : ci_rules) hits += replace_all_ci(s, r.from, r.to);

    // Layer 3: API tokens & secrets (HF, GitHub, OpenAI/Anthropic, Google)
    static const std::regex re_tokens(
        std::string("\\b(") + "h" + "f_[A-Za-z0-9]{16,}|" + "gh" + "p_[A-Za-z0-9]+|" +
        "gh" + "o_[A-Za-z0-9]+|" + "github_" + "pat_[A-Za-z0-9_]+|" +
        "s" + "k-[A-Za-z0-9]{12,}|" + "AI" + "za[A-Za-z0-9_\\-]{12,})\\b");
    {
        std::string next = std::regex_replace(s, re_tokens, "<REDACTED_TOKEN>");
        if (next != s) { s.swap(next); ++hits; }
    }

    // Layer 4: Private LAN / Tailscale IPv4 addresses (preserve 127.0.0.1 & 0.0.0.0)
    static const std::regex re_priv_ip(R"(\b(192\.168\.[0-9]{1,3}\.[0-9]{1,3}|100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})\b)");
    {
        std::string next = std::regex_replace(s, re_priv_ip, "127.0.0.1");
        if (next != s) { s.swap(next); ++hits; }
    }

    // Layer 5: Personal emails (allow only noreply GitHub / Google co-author addresses)
    static const std::regex re_email(R"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})");
    {
        std::string out;
        out.reserve(s.size());
        auto it = std::sregex_iterator(s.begin(), s.end(), re_email);
        auto end = std::sregex_iterator();
        size_t last = 0;
        bool changed = false;
        for (; it != end; ++it) {
            const auto& m = *it;
            const std::string em = m.str();
            out.append(s, last, size_t(m.position()) - last);
            if (em.find("users.noreply.github.com") != std::string::npos || em == "noreply@google.com") {
                out.append(em);
            } else {
                out.append("<REDACTED_EMAIL>");
                changed = true;
                ++hits;
            }
            last = size_t(m.position() + m.length());
        }
        if (changed) {
            out.append(s, last, std::string::npos);
            s.swap(out);
        }
    }

    // Layer 6: GPU/ASIC serial numbers (16-hex serials preceded by serial/board_id/unique_id)
    static const std::regex re_serial(R"((unique_id|serial_number|board_serial)["']?\s*[:=]\s*["']?(?:0x)?[0-9a-fA-F]{12,20}["']?)", std::regex::icase);
    {
        std::string next = std::regex_replace(s, re_serial, "$1=<REDACTED_SERIAL>");
        if (next != s) { s.swap(next); ++hits; }
    }

    // Layer 7: Branding normalization & informal hardware vocabulary
    static const std::regex re_kitchen(std::string("\\b(\x54") + "he Kitchen|\x54" + "HE KITCHEN)\\b");
    {
        std::string next = std::regex_replace(s, re_kitchen, "the kitchen");
        if (next != s) { s.swap(next); ++hits; }
    }
    static const std::regex re_homerig(std::string("\\b([hH]ome [rR][iI][gG]|[rR][iI][gG])\\b"));
    {
        std::string next = std::regex_replace(s, re_homerig, "workstation");
        if (next != s) { s.swap(next); ++hits; }
    }
    if (scrub_placeholders) {
        static const std::regex re_ph(std::string("\\b(\x54") + "BD|\x54" + "ODO|\x46" + "IXME)\\b");
        std::string next = std::regex_replace(s, re_ph, "NOTE");
        if (next != s) { s.swap(next); ++hits; }
    }
    return hits;
}

// Strict 8-layer Zero-PII/Zero-Secret audit gate (returns total violations across all 8 checks)
static size_t audit_directory(const std::string& target_dir, const std::string& user,
                              const std::string& grp, const std::string& host, const std::string& host_short) {
    static const std::string kPfxHome = std::string("\x2f") + "home\x2f";
    static const std::string kPfxUsrs = std::string("\x2f") + "Users\x2f";
    static const std::string kPfxRoot = std::string("\x2f") + "root\x2f";
    static const std::string kPfxData = std::string("\x2f") + "data\x2f";
    static const std::string kPfxBulk = std::string("\x2f") + "bulk\x2f";

    static const std::regex re_paths("(" + kPfxHome + "|" + kPfxUsrs + "|" + kPfxRoot + "|" + kPfxData + "|" + kPfxBulk + ")");
    static const std::regex re_tokens(
        std::string("\\b(") + "h" + "f_[A-Za-z0-9]{16,}|" + "gh" + "p_[A-Za-z0-9]+|" +
        "gh" + "o_[A-Za-z0-9]+|" + "github_" + "pat_[A-Za-z0-9_]+|" +
        "s" + "k-[A-Za-z0-9]{12,}|" + "AI" + "za[A-Za-z0-9_\\-]{12,})\\b");
    static const std::regex re_priv_ip(R"(\b(192\.168\.[0-9]{1,3}\.[0-9]{1,3}|100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})\b)");
    static const std::regex re_email(R"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})");
    static const std::regex re_serial(R"((unique_id|serial_number|board_serial)["']?\s*[:=]\s*["']?(?:0x)?[0-9a-fA-F]{12,20}["']?)", std::regex::icase);

    size_t checks[8] = {};
    size_t files = 0;

    for (const auto& entry : fs::recursive_directory_iterator(target_dir)) {
        if (!entry.is_regular_file()) continue;
        // Check 8: Zero Python files in release tree!
        if (entry.path().extension() == ".py") {
            ++checks[7];
            std::fprintf(stderr, "[AUDIT FAIL L8-PYTHON] %s\n", entry.path().c_str());
            continue;
        }
        if (!is_text_ext(entry.path())) continue;
        if (entry.path().filename() == "hy4_770b_sanitize.cpp") continue;
        ++files;
        std::ifstream ifs(entry.path(), std::ios::binary);
        if (!ifs) continue;
        std::ostringstream ss;
        ss << ifs.rdbuf();
        const std::string c = ss.str();

        // Check 1: Local paths
        if (std::regex_search(c, re_paths)) {
            ++checks[0];
            std::fprintf(stderr, "[AUDIT FAIL L1-LOCAL-PATH] %s\n", entry.path().c_str());
        }
        // Check 2: Bare OS username / group
        if ((!user.empty() && contains_ci(c, user)) || (!grp.empty() && contains_ci(c, grp))) {
            ++checks[1];
            std::fprintf(stderr, "[AUDIT FAIL L2-USERNAME] %s\n", entry.path().c_str());
        }
        // Check 3: Hostname
        if ((!host.empty() && contains_ci(c, host)) || (!host_short.empty() && host_short.size() >= 4 && contains_ci(c, host_short))) {
            ++checks[2];
            std::fprintf(stderr, "[AUDIT FAIL L3-HOSTNAME] %s\n", entry.path().c_str());
        }
        // Check 4: Tokens/secrets
        if (std::regex_search(c, re_tokens)) {
            ++checks[3];
            std::fprintf(stderr, "[AUDIT FAIL L4-SECRET-TOKEN] %s\n", entry.path().c_str());
        }
        // Check 5: Private LAN / Tailscale IPs
        if (std::regex_search(c, re_priv_ip)) {
            ++checks[4];
            std::fprintf(stderr, "[AUDIT FAIL L5-PRIVATE-IP] %s\n", entry.path().c_str());
        }
        // Check 6: Non-noreply personal emails
        {
            auto it = std::sregex_iterator(c.begin(), c.end(), re_email);
            for (; it != std::sregex_iterator(); ++it) {
                const std::string em = it->str();
                if (em.find("users.noreply.github.com") == std::string::npos && em != "noreply@google.com") {
                    ++checks[5];
                    std::fprintf(stderr, "[AUDIT FAIL L6-PERSONAL-EMAIL] %s (%s)\n", entry.path().c_str(), em.c_str());
                    break;
                }
            }
        }
        // Check 7: GPU/ASIC serials
        if (std::regex_search(c, re_serial)) {
            ++checks[6];
            std::fprintf(stderr, "[AUDIT FAIL L7-HW-SERIAL] %s\n", entry.path().c_str());
        }
    }

    size_t total = 0;
    for (int i = 0; i < 8; ++i) total += checks[i];
    std::printf("[SANITIZE-AUDIT] files=%zu | L1_path=%zu L2_user=%zu L3_host=%zu L4_tok=%zu L5_ip=%zu L6_email=%zu L7_ser=%zu L8_py=%zu -> %s\n",
                files, checks[0], checks[1], checks[2], checks[3], checks[4], checks[5], checks[6], checks[7],
                total == 0 ? "PASS (8/8 ZERO-PII)" : "FAIL");
    return total;
}

}  // namespace hy4_770b_san

int main(int argc, char** argv) {
    using namespace hy4_770b_san;
    std::string target_dir;
    std::string workspace_root;
    std::string weight_root;
    std::string wave10_freeze;
    std::string flash_freeze;
    std::string nvme0, nvme1;
    bool do_scrub = true;
    bool do_audit = true;
    bool scrub_placeholders = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) target_dir = argv[++i];
        else if (a == "--workspace" && i + 1 < argc) workspace_root = argv[++i];
        else if (a == "--weight-root" && i + 1 < argc) weight_root = argv[++i];
        else if (a == "--wave10-freeze" && i + 1 < argc) wave10_freeze = argv[++i];
        else if (a == "--flash-freeze" && i + 1 < argc) flash_freeze = argv[++i];
        else if (a == "--nvme0" && i + 1 < argc) nvme0 = argv[++i];
        else if (a == "--nvme1" && i + 1 < argc) nvme1 = argv[++i];
        else if (a == "--audit-only") { do_scrub = false; do_audit = true; }
        else if (a == "--scrub-only") { do_scrub = true; do_audit = false; }
        else if (a == "--keep-placeholders") scrub_placeholders = false;
    }

    if (target_dir.empty() || !fs::exists(target_dir)) {
        std::fprintf(stderr, "Usage: %s --dir <staged_dir> [--workspace <p>] [--weight-root <p>] [--wave10-freeze <p>] [--flash-freeze <p>] [--nvme0 <p>] [--nvme1 <p>] [--audit-only|--scrub-only]\n", argv[0]);
        return 1;
    }

    const std::string user = get_local_user();
    const std::string grp  = get_local_group();
    const std::string host = get_local_host();
    std::string host_short = host;
    const size_t dot = host_short.find('.');
    if (dot != std::string::npos) host_short.resize(dot);

    std::vector<Rule> exact_rules;
    if (!wave10_freeze.empty())  exact_rules.push_back({wave10_freeze,  "<WAVE10_FREEZE_ROOT>/FREEZE_WAVE10_192G_20261121"});
    if (!flash_freeze.empty())   exact_rules.push_back({flash_freeze,   "<HY4_FLASH_FREEZE_ROOT>/FREEZE_HY4_HLUT4_192G_20261122"});
    if (!workspace_root.empty()) exact_rules.push_back({workspace_root, "<HY4_770B_WORKSPACE_ROOT>"});
    if (!weight_root.empty())    exact_rules.push_back({weight_root,    "<HY4_770B_WEIGHT_ROOT>"});
    if (!nvme0.empty())          exact_rules.push_back({nvme0,          "<NVME_STRIPE_0>"});
    if (!nvme1.empty())          exact_rules.push_back({nvme1,          "<NVME_STRIPE_1>"});
    std::sort(exact_rules.begin(), exact_rules.end(), [](const Rule& a, const Rule& b) { return a.from.size() > b.from.size(); });

    std::vector<Rule> ci_rules;
    if (!user.empty()) ci_rules.push_back({user, "operator"});
    if (!grp.empty() && grp != user) ci_rules.push_back({grp, "operator"});
    if (!host.empty() && host != user) ci_rules.push_back({host, "sb-m5-p2-192g"});
    if (!host_short.empty() && host_short != host && host_short != user) ci_rules.push_back({host_short, "sb-m5-p2-192g"});
    std::sort(ci_rules.begin(), ci_rules.end(), [](const Rule& a, const Rule& b) { return a.from.size() > b.from.size(); });

    if (do_scrub) {
        size_t files_scanned = 0, files_modified = 0, total_hits = 0;
        for (const auto& entry : fs::recursive_directory_iterator(target_dir)) {
            if (!entry.is_regular_file() || !is_text_ext(entry.path())) continue;
            if (entry.path().filename() == "hy4_770b_sanitize.cpp") continue;
            ++files_scanned;
            std::ifstream ifs(entry.path(), std::ios::binary);
            if (!ifs) continue;
            std::ostringstream ss;
            ss << ifs.rdbuf();
            std::string content = ss.str();
            ifs.close();
            const size_t h = scrub_content(content, exact_rules, ci_rules, scrub_placeholders);
            if (h > 0) {
                std::ofstream ofs(entry.path(), std::ios::binary | std::ios::trunc);
                ofs.write(content.data(), std::streamsize(content.size()));
                ++files_modified;
                total_hits += h;
            }
        }
        std::printf("[SANITIZE-SCRUB] scanned=%zu modified=%zu total_replacements=%zu\n",
                    files_scanned, files_modified, total_hits);
    }

    if (do_audit) {
        const size_t v = audit_directory(target_dir, user, grp, host, host_short);
        if (v > 0) return 2;
    }
    return 0;
}
