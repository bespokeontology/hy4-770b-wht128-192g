#pragma once
// Pure-C++17 local contiguous n-gram cycle breaker for Hy4Runtime::generate.
// Inspects ONLY the trailing 64 generated tokens. If an identical contiguous
// span of length p in [1,16] repeats >=4 times consecutively at the tail,
// returns the single next repeating token gen[n-p] to mask (-FLT_MAX).
// Otherwise returns -1: all normal logits remain 100% bit-identical to
// greedy / DFly2 (ZERO global repetition_penalty).
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hy4 {

inline int detect_local_ngram_cycle_ban_tok(const std::vector<int32_t>& gen) noexcept {
    const int n = static_cast<int>(gen.size());
    if (n < 4) return -1;
    const int tail = (n < 64) ? n : 64;
    const int base = n - tail;
    for (int p = 1; p <= 16 && 4 * p <= tail; ++p) {
        const int b0 = base + tail - 4 * p;
        bool same = true;
        for (int r = 1; r < 4 && same; ++r) {
            for (int i = 0; i < p; ++i) {
                if (gen[static_cast<size_t>(b0 + i)] != gen[static_cast<size_t>(b0 + r * p + i)]) { same = false; break; }
            }
        }
        if (same) return gen[static_cast<size_t>(n - p)];
    }
    return -1;
}

}  // namespace hy4
