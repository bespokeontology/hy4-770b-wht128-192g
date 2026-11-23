#pragma once
// Compile-Checked 2x Tenstorrent Blackhole P150A (FABRIC_1D TP2 + 330 MiB Tensix L1 SRAM) Contract
// Model: tencent/Hy4 (770B non-latent MoE: H = 6144, I_moe = 2048, TP2 shard = 1024/card)
// Zero Python anywhere. Every GDDR6 and Tensix L1 bank limit is compile-time static_assert verified.

#include "../src/hy4_770b_config.hpp"
#include "../src/hy4_770b_hlut_pack.hpp"

#include <cstddef>
#include <cstdint>

namespace hy4_770b {

inline constexpr uint32_t kP150CoresPerCard           = 110u;       // Blackhole P150A active Tensix grid under FABRIC_1D
inline constexpr uint32_t kP150Fabric1DBankBytes      = 1'459'200u; // Live verified main-L1 bank ceiling per core
inline constexpr uint32_t kP150Fabric1DCbGuardPerCore = 262'144u;   // 256 KiB/core Circular Buffer guard
inline constexpr uint64_t kP150Gddr6CapPerCard        = 31ULL * 1024ULL * 1024ULL * 1024ULL;  // 31 GiB safe cap / 32 GiB GDDR6

// Hy4-770B Non-Latent MoE TP2 shard geometry per card:
//   Gate/Up shard per card: [N_shard=1024, K=6144] -> 8 x 48 = 384 128x128 superblocks
//   Down shard per card:    [N=6144, K_shard=1024] -> 48 x 8 = 384 128x128 superblocks
inline constexpr uint32_t kP150Hy4HiddenK             = uint32_t(kHiddenSize);       // 6144
inline constexpr uint32_t kP150Hy4InterShard          = uint32_t(kMoeInterSize / 2); // 1024
inline constexpr uint32_t kP150Hy4SuperPerProj        = (kP150Hy4InterShard / 128u) * (kP150Hy4HiddenK / 128u); // 384
static_assert(kP150Hy4SuperPerProj == 384u, "Hy4-770B TP2 P150A projection must be exact 384 superblocks (8x48)");

// Exact per-expert-shard bytes per P150A card:
//   HLUT3: 3 * 384 * 6400 B = 7,372,800 B (7.03125 MiB/card)
//   HLUT2: 3 * 384 * 4352 B = 5,013,504 B (4.78125 MiB/card)
inline constexpr uint64_t kP150Hlut3CardBytes = kHlut3Tp2CardExpBytes;
inline constexpr uint64_t kP150Hlut2CardBytes = kHlut2Tp2CardExpBytes;

// Verify BOTH Profile A (57 HLUT3/layer) and Profile B (24 HLUT3 + 49 HLUT2/layer) fit < 31 GiB GDDR6/card!
static_assert(uint64_t(kNumMoeLayers) * 57ULL * kP150Hlut3CardBytes <= kP150Gddr6CapPerCard,
              "Profile A: 57 HLUT3 experts/layer (30.14 GiB/card) must fit 31 GiB P150A GDDR6");
static_assert(uint64_t(kNumMoeLayers) * (24ULL * kP150Hlut3CardBytes + 49ULL * kP150Hlut2CardBytes) <= kP150Gddr6CapPerCard,
              "Profile B: 24 HLUT3 + 49 HLUT2 experts/layer (30.30 GiB/card) must fit 31 GiB P150A GDDR6");

// Exact Tensix L1 SRAM Hot-Expert Pinning Capacity per card (across 110 cores = 165 MiB/card, 330 MiB total!)
inline constexpr size_t p150_l1_per_bank_bytes(uint32_t hot_hlut3, uint32_t hot_hlut2) noexcept {
    const auto bank_pages = [](size_t pages, size_t page_b) constexpr -> size_t {
        return ((pages + kP150CoresPerCard - 1u) / kP150CoresPerCard) * page_b;
    };
    // Workspace: x_in BF16 [32, 6144] + gate/up/down partials + CB guard 256 KiB/core
    size_t b = 131'072u + kP150Fabric1DCbGuardPerCore;
    if (hot_hlut3 > 0u) b += 3u * bank_pages(size_t(hot_hlut3) * size_t(kP150Hy4SuperPerProj), size_t(kHlut3SuperBytes));
    if (hot_hlut2 > 0u) b += 3u * bank_pages(size_t(hot_hlut2) * size_t(kP150Hy4SuperPerProj), size_t(kHlut2SuperBytes));
    return b;
}

// Section 3 Table Tier 0 targets: 18 ultra-hot HLUT3 shards/card (Profile A) or 26 HLUT2 shards/card (Profile B)!
inline constexpr uint32_t kP150MaxHotL1Hlut3 = 15u;  // 15 * 7.031 MiB = 105.5 MiB/card (211 MiB total) + CB guard <= 1,459,200 B/core
inline constexpr uint32_t kP150MaxHotL1Hlut2 = 22u;  // 22 * 4.781 MiB = 105.2 MiB/card (210 MiB total) + CB guard <= 1,459,200 B/core
static_assert(p150_l1_per_bank_bytes(kP150MaxHotL1Hlut3, 0u) <= kP150Fabric1DBankBytes,
              "15 hot HLUT3 shards/card must fit 1,459,200 B FABRIC_1D L1 bank with 256 KiB CB guard");
static_assert(p150_l1_per_bank_bytes(0u, kP150MaxHotL1Hlut2) <= kP150Fabric1DBankBytes,
              "22 hot HLUT2 shards/card must fit 1,459,200 B FABRIC_1D L1 bank with 256 KiB CB guard");

}  // namespace hy4_770b
