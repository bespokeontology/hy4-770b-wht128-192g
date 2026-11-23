// Blackhole P150A Hy4-flash Expert-Compact Row Gather + Tilizer (RISCV_0 / NOC_1 across 24 cores)
// Hy4-flash H_latent = 1536 -> 48 tiles = 24 cores x 2 tiles/core (64 cols = 128 B BF16/row).
// x_rm is allocated as [1, 1, max_rows, 2048] BFLOAT16 TILE -> 2 exact 2048 B pages per row!
// page_grp = core_i >> 4 (0 for cols 0..1023, 1 for cols 1024..1535), byte_off = (core_i & 15) * 128.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_rm_addr      = get_arg_val<uint32_t>(0);
    const uint32_t slot_meta_addr = get_arg_val<uint32_t>(1);
    const uint32_t x_tilized_addr = get_arg_val<uint32_t>(2);
    const uint32_t slot_start     = get_arg_val<uint32_t>(3);
    const uint32_t core_i         = get_arg_val<uint32_t>(4);  // 0..23
    const uint32_t batch_cnt      = get_arg_val<uint32_t>(5);  // 1..16

    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto m_args = TensorAccessorArgs<x_args.next_compile_time_args_offset(), x_args.next_common_runtime_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<m_args.next_compile_time_args_offset(), m_args.next_common_runtime_args_offset()>();

    const auto x_rm_acc = TensorAccessor(x_args, x_rm_addr, 2048);
    const auto meta_acc = TensorAccessor(m_args, slot_meta_addr, 128);
    const auto out_acc  = TensorAccessor(o_args, x_tilized_addr, 2048);

    cb_reserve_back(0, 1);
    const uint32_t base_l1   = get_write_ptr(0);
    const uint32_t gather_l1 = base_l1 + 128u;
    const uint32_t tile0_l1  = base_l1 + 4224u;
    const uint32_t tile1_l1  = base_l1 + 6272u;

    const uint32_t page_grp = core_i >> 4;
    const uint32_t byte_off = (core_i & 15u) * 128u;

    for (uint32_t b = 0u; b < batch_cnt; ++b) {
        const uint32_t slot_idx = slot_start + b;
        noc_async_read(meta_acc.get_noc_addr(slot_idx, 0), base_l1, 128u);
        noc_async_read_barrier();

        const volatile uint32_t* meta_u32 = reinterpret_cast<const volatile uint32_t*>(base_l1);
        const uint32_t cnt = meta_u32[0];
        if (cnt == 0u) continue;
        const volatile uint16_t* row_idx = reinterpret_cast<const volatile uint16_t*>(base_l1 + 8u);

        for (uint32_t r = 0u; r < cnt; ++r) {
            const uint32_t src_row = row_idx[r];
            noc_async_read(x_rm_acc.get_noc_addr(src_row * 6u + page_grp, byte_off), gather_l1 + r * 128u, 128u);
        }

        uint64_t* dst0 = reinterpret_cast<uint64_t*>(tile0_l1);
        uint64_t* dst1 = reinterpret_cast<uint64_t*>(tile1_l1);
        const uint32_t top_lim = (cnt < 16u) ? 16u : 32u;
        for (uint32_t r = cnt; r < top_lim; ++r) {
            uint64_t* z = reinterpret_cast<uint64_t*>(gather_l1 + r * 128u);
            for (uint32_t w = 0u; w < 16u; ++w) z[w] = 0ULL;
        }
        if (cnt > 0u) noc_async_read_barrier();

        const uint64_t* src_u64 = reinterpret_cast<const uint64_t*>(gather_l1);
        for (uint32_t fr = 0u; fr < 16u; ++fr) {
            const uint64_t* s_top = src_u64 + fr * 16u;
            const uint32_t d_off  = fr * 4u;
            #pragma GCC unroll 4
            for (uint32_t c4 = 0u; c4 < 4u; ++c4) {
                dst0[0u   + d_off + c4] = s_top[0u  + c4];
                dst0[64u  + d_off + c4] = s_top[4u  + c4];
                dst1[0u   + d_off + c4] = s_top[8u  + c4];
                dst1[64u  + d_off + c4] = s_top[12u + c4];
            }
        }
        if (cnt <= 16u) {
            for (uint32_t w = 128u; w < 256u; ++w) {
                dst0[w] = 0ULL;
                dst1[w] = 0ULL;
            }
        } else {
            for (uint32_t fr = 0u; fr < 16u; ++fr) {
                const uint64_t* s_bot = src_u64 + (16u + fr) * 16u;
                const uint32_t d_off  = fr * 4u;
                #pragma GCC unroll 4
                for (uint32_t c4 = 0u; c4 < 4u; ++c4) {
                    dst0[128u + d_off + c4] = s_bot[0u  + c4];
                    dst0[192u + d_off + c4] = s_bot[4u  + c4];
                    dst1[128u + d_off + c4] = s_bot[8u  + c4];
                    dst1[192u + d_off + c4] = s_bot[12u + c4];
                }
            }
        }

        const uint32_t out_base = b * 192u + 2u * core_i;
        noc_async_write_tile(out_base + 0u, out_acc, tile0_l1);
        noc_async_write_tile(out_base + 1u, out_acc, tile1_l1);
        noc_async_writes_flushed();
    }
    noc_async_write_barrier();
}
