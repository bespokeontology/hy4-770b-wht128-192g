// Blackhole P150A Hy4-flash Multi-Mode On-Chip SFPU Dequant + Tensix FPU Matmul Compute Kernel
// 100% pure on-device Tensix silicon across ALL 3 controlled A/B modes (ZERO host CPU AVX2 expert math!):
//   QUANT_MODE == 0 : Stock 128x128 Block-FP8 E4M3FN (exact normal + subnormal bit-arithmetic in SFPU!)
//   QUANT_MODE == 1 : WHT128-HLUT4 4.125 bpw (exact 16-level Lloyd-Max in SFPU!)
//   QUANT_MODE == 2 : WHT128-HLUT3 3.125 bpw (exact 8-level Lloyd-Max + sign bitplane in SFPU!)
// Unpacks 16 Float16_b (BF16) weight tiles (32 KiB L1, zero BFP8 rounding loss!) into CB 1, then executes
// weight-stationary wrap-safe rt_dim=2 FPU matmul_block into FP32 CB 16!

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {

// ---- MODE 0: Exact IEEE-754 FP8 E4M3FN (normal + subnormal) SFPU decoder ----
template <int NT_SUB>
sfpi_inline sfpi::vFloat decode_fp8_vec(const sfpi::vUInt& w, const sfpi::vFloat& scale_inv) {
    constexpr uint32_t sh = static_cast<uint32_t>(NT_SUB * 8);
    const sfpi::vUInt b = (sh == 0u) ? (w & 0xFFu) : ((w >> sh) & 0xFFu);
    const sfpi::vUInt mag = b & 0x7Fu;
    sfpi::vFloat v = 0.0f;
    v_if (mag >= 8u) {
        const sfpi::vUInt bits = (mag << 20) + 0x3C000000u;
        v = sfpi::as<sfpi::vFloat>(bits);
    } v_else {
        const sfpi::vUInt sbits = 0x3C800000u | (mag << 20);
        v = sfpi::as<sfpi::vFloat>(sbits) - 0.015625f;
    } v_endif;
    v = v * scale_inv;
    if constexpr (sh == 24u) {
        return sfpi::copysgn(v, sfpi::as<sfpi::vSMag>(w));
    } else {
        return sfpi::copysgn(v, sfpi::as<sfpi::vSMag>(w << (24u - sh)));
    }
}

template <int COL_FACE, int ODD>
__attribute__((noinline)) void unpack_fp8_4nt() {
    constexpr int face_top = COL_FACE * 8 + ODD;
    constexpr int face_bot = 16 + COL_FACE * 8 + ODD;
    const sfpi::vFloat sc = sfpi::dst_reg[224 + ODD];

    #pragma GCC unroll 4
    for (int r_grp = 0; r_grp < 4; ++r_grp) {
        const int row_off = face_top + 2 * r_grp;
        const sfpi::vInt w_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vUInt w  = sfpi::as<sfpi::vUInt>(w_i);
        sfpi::dst_reg[0 * 32 + row_off] = decode_fp8_vec<0>(w, sc);
        sfpi::dst_reg[1 * 32 + row_off] = decode_fp8_vec<1>(w, sc);
        sfpi::dst_reg[2 * 32 + row_off] = decode_fp8_vec<2>(w, sc);
        sfpi::dst_reg[3 * 32 + row_off] = decode_fp8_vec<3>(w, sc);
    }
    #pragma GCC unroll 4
    for (int r_grp = 0; r_grp < 4; ++r_grp) {
        const int row_off = face_bot + 2 * r_grp;
        const sfpi::vInt w_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vUInt w  = sfpi::as<sfpi::vUInt>(w_i);
        sfpi::dst_reg[0 * 32 + row_off] = decode_fp8_vec<0>(w, sc);
        sfpi::dst_reg[1 * 32 + row_off] = decode_fp8_vec<1>(w, sc);
        sfpi::dst_reg[2 * 32 + row_off] = decode_fp8_vec<2>(w, sc);
        sfpi::dst_reg[3 * 32 + row_off] = decode_fp8_vec<3>(w, sc);
    }
}

// ---- MODE 1: Exact 16-level Lloyd-Max N(0,1) WHT128-HLUT4 SFPU decoder (3 LREGs peak!) ----
template <int KT_LOCAL, int NT_SUB>
sfpi_inline sfpi::vFloat decode_hlut4_vec(const sfpi::vUInt& w, const sfpi::vFloat& sig) {
    constexpr uint32_t sh = static_cast<uint32_t>(4 * (KT_LOCAL * 4 + NT_SUB));
    constexpr uint32_t m0 = 1u << sh;
    constexpr uint32_t m1 = 2u << sh;
    constexpr uint32_t m2 = 4u << sh;

    sfpi::vFloat v = 0.12839780f;
    v_if ((w & m0) != 0u) { v = 0.38809040f; } v_endif;
    v_if ((w & m1) != 0u) {
        v = 0.65678227f;
        v_if ((w & m0) != 0u) { v = 0.94235235f; } v_endif;
    } v_endif;
    v_if ((w & m2) != 0u) {
        v = 1.25623083f;
        v_if ((w & m0) != 0u) { v = 1.61804903f; } v_endif;
        v_if ((w & m1) != 0u) {
            v = 2.06901598f;
            v_if ((w & m0) != 0u) { v = 2.73259377f; } v_endif;
        } v_endif;
    } v_endif;
    v = v * sig;
    if constexpr (sh == 28u) {
        return sfpi::copysgn(v, sfpi::as<sfpi::vSMag>(w));
    } else {
        return sfpi::copysgn(v, sfpi::as<sfpi::vSMag>(w << (28u - sh)));
    }
}

template <int KT_LOCAL, int NT_SUB, int COL_FACE, int ODD>
__attribute__((noinline)) void unpack_hlut4_1nt() {
    constexpr int sc_base  = (KT_LOCAL == 0) ? 224 : 240;
    constexpr int face_top = COL_FACE * 8 + ODD;
    constexpr int face_bot = 16 + COL_FACE * 8 + ODD;
    const sfpi::vFloat sig = sfpi::dst_reg[sc_base + COL_FACE * 8 + NT_SUB * 2 + ODD];

    #pragma GCC unroll 4
    for (int r_grp = 0; r_grp < 4; ++r_grp) {
        const int row_off = face_top + 2 * r_grp;
        const sfpi::vInt w_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vUInt w  = sfpi::as<sfpi::vUInt>(w_i);
        sfpi::dst_reg[NT_SUB * 32 + row_off] = decode_hlut4_vec<KT_LOCAL, NT_SUB>(w, sig);
    }
    #pragma GCC unroll 4
    for (int r_grp = 0; r_grp < 4; ++r_grp) {
        const int row_off = face_bot + 2 * r_grp;
        const sfpi::vInt w_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vUInt w  = sfpi::as<sfpi::vUInt>(w_i);
        sfpi::dst_reg[NT_SUB * 32 + row_off] = decode_hlut4_vec<KT_LOCAL, NT_SUB>(w, sig);
    }
}

// ---- MODE 2: Exact 8-level Lloyd-Max N(0,1) WHT128-HLUT3 SFPU decoder (4 LREGs peak!) ----
template <int KT_SUB, int NT_SUB>
sfpi_inline sfpi::vFloat decode_hlut3_vec(const sfpi::vUInt& w_lo2, const sfpi::vUInt& w_hi1, const sfpi::vFloat& sig) {
    constexpr uint32_t t_idx = static_cast<uint32_t>(KT_SUB * 4 + NT_SUB);
    constexpr uint32_t sh2   = 2u * t_idx;
    constexpr uint32_t m0    = 1u << sh2;
    constexpr uint32_t m1    = 2u << sh2;

    sfpi::vFloat v = 0.24509415f;
    v_if ((w_lo2 & m0) != 0u) { v = 0.75600523f; } v_endif;
    v_if ((w_lo2 & m1) != 0u) {
        v = 1.34390926f;
        v_if ((w_lo2 & m0) != 0u) { v = 2.15194559f; } v_endif;
    } v_endif;
    v = v * sig;
    return sfpi::copysgn(v, sfpi::as<sfpi::vSMag>(w_hi1 << (31u - t_idx)));
}

template <int KT_SUB, int NT_SUB, int COL_FACE, int ODD>
__attribute__((noinline)) void unpack_hlut3_1nt() {
    constexpr int kt_local = KT_SUB & 1;
    constexpr int sc_base  = (kt_local == 0) ? 224 : 240;
    constexpr int face_top = COL_FACE * 8 + ODD;
    constexpr int face_bot = 16 + COL_FACE * 8 + ODD;
    const sfpi::vFloat sig = sfpi::dst_reg[sc_base + COL_FACE * 8 + NT_SUB * 2 + ODD];

    #pragma GCC unroll 4
    for (int r_grp = 0; r_grp < 4; ++r_grp) {
        const int row_off = face_top + 2 * r_grp;
        const sfpi::vInt w0_i = sfpi::dst_reg[160 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vInt w1_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vUInt w0  = sfpi::as<sfpi::vUInt>(w0_i);
        const sfpi::vUInt w1  = sfpi::as<sfpi::vUInt>(w1_i);
        sfpi::dst_reg[NT_SUB * 32 + row_off] = decode_hlut3_vec<KT_SUB, NT_SUB>(w0, w1, sig);
    }
    #pragma GCC unroll 4
    for (int r_grp = 0; r_grp < 4; ++r_grp) {
        const int row_off = face_bot + 2 * r_grp;
        const sfpi::vInt w0_i = sfpi::dst_reg[160 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vInt w1_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vUInt w0  = sfpi::as<sfpi::vUInt>(w0_i);
        const sfpi::vUInt w1  = sfpi::as<sfpi::vUInt>(w1_i);
        sfpi::dst_reg[NT_SUB * 32 + row_off] = decode_hlut3_vec<KT_SUB, NT_SUB>(w0, w1, sig);
    }
}

// ---- MODE 3: Exact 4-level Lloyd-Max N(0,1) WHT128-HLUT2 SFPU decoder (3 LREGs peak!) ----
template <int KT_SUB, int NT_SUB>
sfpi_inline sfpi::vFloat decode_hlut2_vec(const sfpi::vUInt& w, const sfpi::vFloat& sig) {
    constexpr uint32_t t_idx = static_cast<uint32_t>(KT_SUB * 4 + NT_SUB);
    constexpr uint32_t sh    = 2u * t_idx;
    constexpr uint32_t m0    = 1u << sh;

    sfpi::vFloat v = 0.45278004f;
    v_if ((w & m0) != 0u) { v = 1.51041761f; } v_endif;
    v = v * sig;
    if constexpr (sh == 30u) {
        return sfpi::copysgn(v, sfpi::as<sfpi::vSMag>(w));
    } else {
        return sfpi::copysgn(v, sfpi::as<sfpi::vSMag>(w << (30u - sh)));
    }
}

template <int KT_SUB, int NT_SUB, int COL_FACE, int ODD>
__attribute__((noinline)) void unpack_hlut2_1nt() {
    constexpr int kt_local = KT_SUB & 1;
    constexpr int sc_base  = (kt_local == 0) ? 224 : 240;
    constexpr int face_top = COL_FACE * 8 + ODD;
    constexpr int face_bot = 16 + COL_FACE * 8 + ODD;
    const sfpi::vFloat sig = sfpi::dst_reg[sc_base + COL_FACE * 8 + NT_SUB * 2 + ODD];

    #pragma GCC unroll 4
    for (int r_grp = 0; r_grp < 4; ++r_grp) {
        const int row_off = face_top + 2 * r_grp;
        const sfpi::vInt w_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vUInt w  = sfpi::as<sfpi::vUInt>(w_i);
        sfpi::dst_reg[NT_SUB * 32 + row_off] = decode_hlut2_vec<KT_SUB, NT_SUB>(w, sig);
    }
    #pragma GCC unroll 4
    for (int r_grp = 0; r_grp < 4; ++r_grp) {
        const int row_off = face_bot + 2 * r_grp;
        const sfpi::vInt w_i = sfpi::dst_reg[192 + row_off].mode<sfpi::DataLayout::I32>();
        const sfpi::vUInt w  = sfpi::as<sfpi::vUInt>(w_i);
        sfpi::dst_reg[NT_SUB * 32 + row_off] = decode_hlut2_vec<KT_SUB, NT_SUB>(w, sig);
    }
}

}  // namespace ckernel::sfpu
#endif

#define CALL_FP8_4NT() \
    MATH((::ckernel::sfpu::unpack_fp8_4nt<0, 0>())); \
    MATH((::ckernel::sfpu::unpack_fp8_4nt<0, 1>())); \
    MATH((::ckernel::sfpu::unpack_fp8_4nt<1, 0>())); \
    MATH((::ckernel::sfpu::unpack_fp8_4nt<1, 1>()))

#define CALL_HLUT4_1NT(KL, NS) \
    MATH((::ckernel::sfpu::unpack_hlut4_1nt<KL, NS, 0, 0>())); \
    MATH((::ckernel::sfpu::unpack_hlut4_1nt<KL, NS, 0, 1>())); \
    MATH((::ckernel::sfpu::unpack_hlut4_1nt<KL, NS, 1, 0>())); \
    MATH((::ckernel::sfpu::unpack_hlut4_1nt<KL, NS, 1, 1>()))

#define CALL_HLUT4_4NT(KL) \
    CALL_HLUT4_1NT(KL, 0); CALL_HLUT4_1NT(KL, 1); CALL_HLUT4_1NT(KL, 2); CALL_HLUT4_1NT(KL, 3)

#define CALL_HLUT3_1NT(KS, NS) \
    MATH((::ckernel::sfpu::unpack_hlut3_1nt<KS, NS, 0, 0>())); \
    MATH((::ckernel::sfpu::unpack_hlut3_1nt<KS, NS, 0, 1>())); \
    MATH((::ckernel::sfpu::unpack_hlut3_1nt<KS, NS, 1, 0>())); \
    MATH((::ckernel::sfpu::unpack_hlut3_1nt<KS, NS, 1, 1>()))

#define CALL_HLUT3_4NT(KS) \
    CALL_HLUT3_1NT(KS, 0); CALL_HLUT3_1NT(KS, 1); CALL_HLUT3_1NT(KS, 2); CALL_HLUT3_1NT(KS, 3)

#define CALL_HLUT2_1NT(KS, NS) \
    MATH((::ckernel::sfpu::unpack_hlut2_1nt<KS, NS, 0, 0>())); \
    MATH((::ckernel::sfpu::unpack_hlut2_1nt<KS, NS, 0, 1>())); \
    MATH((::ckernel::sfpu::unpack_hlut2_1nt<KS, NS, 1, 0>())); \
    MATH((::ckernel::sfpu::unpack_hlut2_1nt<KS, NS, 1, 1>()))

#define CALL_HLUT2_4NT(KS) \
    CALL_HLUT2_1NT(KS, 0); CALL_HLUT2_1NT(KS, 1); CALL_HLUT2_1NT(KS, 2); CALL_HLUT2_1NT(KS, 3)

inline void pack_4nt_to_cb1(uint32_t base_tile) {
    tile_regs_wait();
    pack_tile<true>(0, 1, base_tile + 0u);
    pack_tile<true>(1, 1, base_tile + 1u);
    pack_tile<true>(2, 1, base_tile + 2u);
    pack_tile<true>(3, 1, base_tile + 3u);
    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t kQuantMode = get_compile_time_arg_val(0);  // 0=FP8, 1=HLUT4, 2=HLUT3

    const uint32_t num_sb        = get_arg_val<uint32_t>(0);  // 1
    const uint32_t batch_cnt     = get_arg_val<uint32_t>(1);
    const uint32_t num_projs     = get_arg_val<uint32_t>(2);  // 2 for fused gate+up, 1 for down
    const uint32_t keep_act      = get_arg_val<uint32_t>(3);  // 1 when act_slot_stride == 0
    const uint32_t slot_start    = get_arg_val<uint32_t>(4);
    const uint32_t kt_core       = num_sb * 4u;               // 4
    const uint32_t total_w_tiles = num_sb * 16u;              // 16

    cb_wait_front(6, 1);
    const uint32_t scratch_addr = get_tile_address(6, 0);
    const volatile uint16_t* slot_u16 = reinterpret_cast<const volatile uint16_t*>(scratch_addr);
    uint16_t local_eids[24];
    bool any_valid = false;
    for (uint32_t b = 0u; b < batch_cnt; ++b) {
        const uint16_t e = slot_u16[slot_start + b];
        local_eids[b] = e;
        if (e < 0xFFFEu) any_valid = true;
    }
    if (!any_valid) {
        return;
    }

    compute_kernel_hw_startup(4, 4, 1);
    reconfig_data_format_srca(4);
    copy_init(4);
    MATH((SFPU_BINARY_INIT(unused)));
    bool need_sfpu_reconfig = false;

    if (keep_act != 0u) {
        cb_wait_front(0, kt_core);
    }

    auto unpack_weights_to_cb1 = [&]() __attribute__((always_inline)) {
        if (need_sfpu_reconfig) {
            reconfig_full_operand(4, 4);
            pack_reconfig_data_format(1);
            reconfig_data_format_srca(4);
            copy_init(4);
            MATH((SFPU_BINARY_INIT(unused)));
        }
        cb_reserve_back(1, total_w_tiles);

        if constexpr (kQuantMode == 0u) {
            for (uint32_t sb = 0u; sb < num_sb; ++sb) {
                const uint32_t sb_base = sb * 16u;
                cb_wait_front(4, 4);
                cb_wait_front(5, 1);
                for (uint32_t kt = 0u; kt < 4u; ++kt) {
                    tile_regs_acquire();
                    copy_tile(4, kt, 6);
                    copy_tile(5, 0, 7);
                    MATH((_llk_math_eltwise_sfpu_start_(0)));
                    CALL_FP8_4NT();
                    MATH((_llk_math_eltwise_sfpu_done_()));
                    tile_regs_commit();
                    pack_4nt_to_cb1(sb_base + kt * 4u);
                }
                cb_pop_front(4, 4);
                cb_pop_front(5, 1);
            }
        } else if constexpr (kQuantMode == 1u) {
            for (uint32_t sb = 0u; sb < num_sb; ++sb) {
                const uint32_t sb_base = sb * 16u;
                cb_wait_front(4, 2);
                cb_wait_front(5, 2);
                for (uint32_t half = 0u; half < 2u; ++half) {
                    tile_regs_acquire();
                    copy_tile(4, half, 6);
                    copy_tile(5, half, 7);
                    MATH((_llk_math_eltwise_sfpu_start_(0)));
                    CALL_HLUT4_4NT(0);
                    MATH((_llk_math_eltwise_sfpu_done_()));
                    tile_regs_commit();
                    pack_4nt_to_cb1(sb_base + (half * 2u + 0u) * 4u);

                    tile_regs_acquire();
                    copy_tile(4, half, 6);
                    copy_tile(5, half, 7);
                    MATH((_llk_math_eltwise_sfpu_start_(0)));
                    CALL_HLUT4_4NT(1);
                    MATH((_llk_math_eltwise_sfpu_done_()));
                    tile_regs_commit();
                    pack_4nt_to_cb1(sb_base + (half * 2u + 1u) * 4u);
                }
                cb_pop_front(4, 2);
                cb_pop_front(5, 2);
            }
        } else if constexpr (kQuantMode == 2u) {
            for (uint32_t sb = 0u; sb < num_sb; ++sb) {
                const uint32_t sb_base = sb * 16u;
                cb_wait_front(4, 2);
                cb_wait_front(5, 2);
                // kt = 0
                tile_regs_acquire();
                copy_tile(4, 0, 5); copy_tile(4, 1, 6); copy_tile(5, 0, 7);
                MATH((_llk_math_eltwise_sfpu_start_(0))); CALL_HLUT3_4NT(0); MATH((_llk_math_eltwise_sfpu_done_()));
                tile_regs_commit(); pack_4nt_to_cb1(sb_base + 0u);
                // kt = 1
                tile_regs_acquire();
                copy_tile(4, 0, 5); copy_tile(4, 1, 6); copy_tile(5, 0, 7);
                MATH((_llk_math_eltwise_sfpu_start_(0))); CALL_HLUT3_4NT(1); MATH((_llk_math_eltwise_sfpu_done_()));
                tile_regs_commit(); pack_4nt_to_cb1(sb_base + 4u);
                // kt = 2
                tile_regs_acquire();
                copy_tile(4, 0, 5); copy_tile(4, 1, 6); copy_tile(5, 1, 7);
                MATH((_llk_math_eltwise_sfpu_start_(0))); CALL_HLUT3_4NT(2); MATH((_llk_math_eltwise_sfpu_done_()));
                tile_regs_commit(); pack_4nt_to_cb1(sb_base + 8u);
                // kt = 3
                tile_regs_acquire();
                copy_tile(4, 0, 5); copy_tile(4, 1, 6); copy_tile(5, 1, 7);
                MATH((_llk_math_eltwise_sfpu_start_(0))); CALL_HLUT3_4NT(3); MATH((_llk_math_eltwise_sfpu_done_()));
                tile_regs_commit(); pack_4nt_to_cb1(sb_base + 12u);

                cb_pop_front(4, 2);
                cb_pop_front(5, 2);
            }
        } else {
            // MODE 3: WHT128-HLUT2 (1 code tile in CB 4 + 2 scale tiles in CB 5)
            for (uint32_t sb = 0u; sb < num_sb; ++sb) {
                const uint32_t sb_base = sb * 16u;
                cb_wait_front(4, 1);
                cb_wait_front(5, 2);
                // kt = 0
                tile_regs_acquire();
                copy_tile(4, 0, 6); copy_tile(5, 0, 7);
                MATH((_llk_math_eltwise_sfpu_start_(0))); CALL_HLUT2_4NT(0); MATH((_llk_math_eltwise_sfpu_done_()));
                tile_regs_commit(); pack_4nt_to_cb1(sb_base + 0u);
                // kt = 1
                tile_regs_acquire();
                copy_tile(4, 0, 6); copy_tile(5, 0, 7);
                MATH((_llk_math_eltwise_sfpu_start_(0))); CALL_HLUT2_4NT(1); MATH((_llk_math_eltwise_sfpu_done_()));
                tile_regs_commit(); pack_4nt_to_cb1(sb_base + 4u);
                // kt = 2
                tile_regs_acquire();
                copy_tile(4, 0, 6); copy_tile(5, 1, 7);
                MATH((_llk_math_eltwise_sfpu_start_(0))); CALL_HLUT2_4NT(2); MATH((_llk_math_eltwise_sfpu_done_()));
                tile_regs_commit(); pack_4nt_to_cb1(sb_base + 8u);
                // kt = 3
                tile_regs_acquire();
                copy_tile(4, 0, 6); copy_tile(5, 1, 7);
                MATH((_llk_math_eltwise_sfpu_start_(0))); CALL_HLUT2_4NT(3); MATH((_llk_math_eltwise_sfpu_done_()));
                tile_regs_commit(); pack_4nt_to_cb1(sb_base + 12u);

                cb_pop_front(4, 1);
                cb_pop_front(5, 2);
            }
        }

        cb_push_back(1, total_w_tiles);
        cb_wait_front(1, total_w_tiles);
    };

    auto matmul_one_strip = [&]() __attribute__((always_inline)) {
        tile_regs_acquire();
        for (uint32_t inner_dim_idx = 0u; inner_dim_idx < kt_core; ++inner_dim_idx) {
            matmul_block(0, 1, inner_dim_idx, inner_dim_idx * 4u, 0, 0, 4, 1, kt_core);
        }
        tile_regs_commit();

        cb_reserve_back(16, 4);
        tile_regs_wait();
        pack_block(0, 16, 4);
        tile_regs_release();
        cb_push_back(16, 4);
    };

    auto matmul_two_strips = [&]() __attribute__((always_inline)) {
        tile_regs_acquire();
        for (uint32_t inner_dim_idx = 0u; inner_dim_idx < kt_core; ++inner_dim_idx) {
            matmul_block(0, 1, inner_dim_idx, inner_dim_idx * 4u, 0, 0, 4, 2, kt_core);
        }
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(16, 4);
        pack_block(0, 16, 4);
        cb_push_back(16, 4);
        cb_reserve_back(16, 4);
        pack_block(4, 16, 4);
        cb_push_back(16, 4);
        tile_regs_release();
    };

    const bool can_rt2 = (keep_act == 0u && num_projs == 2u);
    uint32_t cb0_half = 0u;
    uint32_t b0 = 0u;
    while (b0 < batch_cnt) {
        const uint32_t expert_slot = local_eids[b0];
        if (expert_slot >= 0xFFFEu) {
            ++b0;
            continue;
        }
        uint32_t run_len = 1u;
        while (b0 + run_len < batch_cnt && local_eids[b0 + run_len] == expert_slot) {
            ++run_len;
        }

        const bool hold_2 = (can_rt2 && run_len == 2u && cb0_half == 0u);
        if (keep_act == 0u && run_len == 1u) {
            cb_wait_front(0, kt_core);
        } else if (hold_2) {
            cb_wait_front(0, 2u * kt_core);
        }

        for (uint32_t p = 0u; p < num_projs; ++p) {
            unpack_weights_to_cb1();

            reconfig_full_operand<SrcOrder::Reverse>(0, 1);
            pack_reconfig_data_format(1, 16);
            need_sfpu_reconfig = true;

            if (keep_act != 0u || run_len == 1u) {
                matmul_block_init(0, 1, 0, 4, 1, kt_core);
                for (uint32_t r = 0u; r < run_len; ++r) {
                    matmul_one_strip();
                }
            } else if (hold_2) {
                matmul_block_init(0, 1, 0, 4, 2, kt_core);
                matmul_two_strips();
            } else if (can_rt2) {
                uint32_t cur_rt = 0u;
                uint32_t r = 0u;
                while (r < run_len) {
                    if (cb0_half == 0u && r + 2u <= run_len) {
                        if (cur_rt != 2u) {
                            matmul_block_init(0, 1, 0, 4, 2, kt_core);
                            cur_rt = 2u;
                        }
                        cb_wait_front(0, 2u * kt_core);
                        matmul_two_strips();
                        cb_pop_front(0, 2u * kt_core);
                        r += 2u;
                    } else {
                        if (cur_rt != 1u) {
                            matmul_block_init(0, 1, 0, 4, 1, kt_core);
                            cur_rt = 1u;
                        }
                        cb_wait_front(0, kt_core);
                        matmul_one_strip();
                        cb_pop_front(0, kt_core);
                        cb0_half ^= 1u;
                        r += 1u;
                    }
                }
            } else {
                matmul_block_init(0, 1, 0, 4, 1, kt_core);
                for (uint32_t r = 0u; r < run_len; ++r) {
                    cb_wait_front(0, kt_core);
                    matmul_one_strip();
                    cb_pop_front(0, kt_core);
                }
            }

            cb_pop_front(1, total_w_tiles);
        }

        if (keep_act == 0u && run_len == 1u) {
            cb_pop_front(0, kt_core);
            if (can_rt2) cb0_half ^= 1u;
        } else if (hold_2) {
            cb_pop_front(0, 2u * kt_core);
        }

        b0 += run_len;
    }

    if (keep_act != 0u) {
        cb_pop_front(0, kt_core);
    }
}
