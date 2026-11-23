// Blackhole P150A Hy4-flash Fused K-Split Reduction + Clamped SwiGLU(10.0) + Optional L1 WHT128 Compute Kernel
// Uses built-in Blackhole hardware SFPU clamped_silu_glu_tile (100% exact to Hy4-flash clamped SwiGLU!).
// When apply_wht==1 (HLUT4/HLUT3): applies 4x4 L1 WHT128 Q_i matmul into CB 16.
// When apply_wht==0 (FP8): packs directly into CB 16 with zero wasted WHT cycles!

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/clamped_silu_glu.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t k_splits  = get_arg_val<uint32_t>(0);  // 12
    const uint32_t m_cnt     = get_arg_val<uint32_t>(1);
    const uint32_t apply_wht = get_arg_val<uint32_t>(2);  // 0=FP8, 1=HLUT4/HLUT3

    const uint32_t out_cb_phase_a = (apply_wht != 0u) ? 24u : 16u;
    compute_kernel_hw_startup(0, 1, out_cb_phase_a);
    if (apply_wht != 0u) {
        cb_wait_front(2, 16);
    }

    for (uint32_t dm = 0u; dm < m_cnt; ++dm) {
        if (apply_wht != 0u) {
            reconfig_full_operand(0, 1);
            pack_reconfig_data_format(24);
        }
        cb_reserve_back(out_cb_phase_a, 4);

        for (uint32_t nt_sub = 0u; nt_sub < 4u; ++nt_sub) {
            tile_regs_acquire();
            reconfig_data_format_srca(0);

            cb_wait_front(0, 1);
            cb_wait_front(1, 1);
            copy_init(0);
            copy_tile(0, 0, 0);
            copy_tile(1, 0, 1);
            cb_pop_front(0, 1);
            cb_pop_front(1, 1);

            add_binary_tile_init();
            for (uint32_t ks = 1u; ks < k_splits; ++ks) {
                cb_wait_front(0, 1);
                cb_wait_front(1, 1);
                copy_tile(0, 0, 2);
                copy_tile(1, 0, 3);
                cb_pop_front(0, 1);
                cb_pop_front(1, 1);
                add_binary_tile(0, 2, 0);
                add_binary_tile(1, 3, 1);
            }

            clamped_silu_glu_tile_init();
            clamped_silu_glu_tile(0, 1, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, out_cb_phase_a);
            tile_regs_release();
        }
        cb_push_back(out_cb_phase_a, 4);

        if (apply_wht != 0u) {
            cb_wait_front(24, 4);
            reconfig_full_operand<SrcOrder::Reverse>(24, 2);
            pack_reconfig_data_format(24, 16);
            matmul_block_init(24, 2, 0, 4, 1, 4);

            tile_regs_acquire();
            for (uint32_t k = 0u; k < 4u; ++k) {
                matmul_block(24, 2, k, k * 4u, 0, 0, 4, 1, 4);
            }
            tile_regs_commit();

            cb_reserve_back(16, 4);
            tile_regs_wait();
            pack_block(0, 16, 4);
            tile_regs_release();
            cb_push_back(16, 4);
            cb_pop_front(24, 4);
        }
    }

    if (apply_wht != 0u) {
        cb_pop_front(2, 16);
    }
}
