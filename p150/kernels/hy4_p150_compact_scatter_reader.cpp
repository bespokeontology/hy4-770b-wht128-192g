// Blackhole P150A Hy4-flash Expert-Compact K-Split + Scatter-Accumulate Reader (RISCV_1 / NOC_0 across 24 cores)
// y_accum_f32 is [1, 1, max_rows, 2048] FLOAT32 TILE -> 2 exact 4096 B pages per row (nt_total = 48).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t partials_addr  = get_arg_val<uint32_t>(0);
    const uint32_t weights_addr   = get_arg_val<uint32_t>(1);
    const uint32_t y_accum_addr   = get_arg_val<uint32_t>(2);
    const uint32_t slot_meta_addr = get_arg_val<uint32_t>(3);
    const uint32_t slot_start     = get_arg_val<uint32_t>(4);
    const uint32_t k_splits       = get_arg_val<uint32_t>(5);  // 6
    const uint32_t core_i         = get_arg_val<uint32_t>(6);  // 0..23
    const uint32_t m_tiles_ws     = get_arg_val<uint32_t>(7);  // 16
    const uint32_t batch_cnt      = get_arg_val<uint32_t>(8);  // 1..16

    constexpr auto p_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<p_args.next_compile_time_args_offset(), p_args.next_common_runtime_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<w_args.next_compile_time_args_offset(), w_args.next_common_runtime_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<y_args.next_compile_time_args_offset(), y_args.next_common_runtime_args_offset()>();

    const auto part_acc = TensorAccessor(p_args, partials_addr, 4096);
    const auto wgt_acc  = TensorAccessor(w_args, weights_addr, 4096);
    const auto y_acc    = TensorAccessor(y_args, y_accum_addr, 4096);
    const auto meta_acc = TensorAccessor(m_args, slot_meta_addr, 128);

    cb_reserve_back(5, 1);
    const uint32_t meta_l1   = get_write_ptr(5);
    const uint32_t gather_l1 = meta_l1 + 256u;
    const uint32_t page_grp  = core_i >> 4;
    const uint32_t byte_off  = (core_i & 15u) * 256u;
    constexpr uint32_t nt_total = 192u;
    const uint32_t p_tiles_per_slot = 2u * k_splits;
    uint32_t unacked = 0u;

    for (uint32_t b = 0u; b < batch_cnt; ++b) {
        const uint32_t slot_idx = slot_start + b;
        if (b > 0u) ++unacked;

        noc_async_read(meta_acc.get_noc_addr(slot_idx, 0), meta_l1, 128u);

        cb_reserve_back(0, p_tiles_per_slot);
        const uint32_t p_wr_base = get_write_ptr(0);
        for (uint32_t t = 0u; t < 2u; ++t) {
            const uint32_t nt = 2u * core_i + t;
            for (uint32_t ks = 0u; ks < k_splits; ++ks) {
                const uint32_t idx = t * k_splits + ks;
                noc_async_read_tile((ks * m_tiles_ws + b) * nt_total + nt, part_acc, p_wr_base + idx * 4096u);
            }
        }

        cb_reserve_back(1, 2);
        const uint32_t w_wr_base = get_write_ptr(1);
        noc_async_read_tile(slot_idx, wgt_acc, w_wr_base);
        noc_async_read_tile(slot_idx, wgt_acc, w_wr_base + 4096u);
        noc_async_read_barrier();

        cb_push_back(0, p_tiles_per_slot);
        cb_push_back(1, 2);

        const volatile uint32_t* meta_u32 = reinterpret_cast<const volatile uint32_t*>(meta_l1);
        const uint32_t cnt = meta_u32[0];
        const uint32_t wait_barrier = meta_u32[1];
        const volatile uint16_t* row_idx = reinterpret_cast<const volatile uint16_t*>(meta_l1 + 8u);

        if (wait_barrier != 0u && unacked > 0u) {
            cb_wait_front(7, unacked);
            cb_pop_front(7, unacked);
            unacked = 0u;
        }

        for (uint32_t r = 0u; r < cnt; ++r) {
            const uint32_t src_row = row_idx[r];
            noc_async_read(y_acc.get_noc_addr(src_row * 6u + page_grp, byte_off), gather_l1 + r * 256u, 256u);
        }
        const uint32_t top_lim = (cnt < 16u) ? 16u : 32u;
        for (uint32_t r = cnt; r < top_lim; ++r) {
            uint64_t* z = reinterpret_cast<uint64_t*>(gather_l1 + r * 256u);
            for (uint32_t w = 0u; w < 32u; ++w) z[w] = 0ULL;
        }
        if (cnt > 0u) noc_async_read_barrier();

        const uint64_t* src_u64 = reinterpret_cast<const uint64_t*>(gather_l1);
        for (uint32_t t = 0u; t < 2u; ++t) {
            cb_reserve_back(2, 1);
            uint64_t* dst_tile = reinterpret_cast<uint64_t*>(get_write_ptr(2));
            const uint32_t col_base = t * 16u;
            for (uint32_t fr = 0u; fr < 16u; ++fr) {
                const uint64_t* s_top = src_u64 + fr * 32u + col_base;
                const uint32_t d_off  = fr * 8u;
                #pragma GCC unroll 8
                for (uint32_t c = 0u; c < 8u; ++c) {
                    dst_tile[0u   + d_off + c] = s_top[0u + c];
                    dst_tile[128u + d_off + c] = s_top[8u + c];
                }
            }
            if (cnt <= 16u) {
                for (uint32_t w = 256u; w < 512u; ++w) dst_tile[w] = 0ULL;
            } else {
                for (uint32_t fr = 0u; fr < 16u; ++fr) {
                    const uint64_t* s_bot = src_u64 + (16u + fr) * 32u + col_base;
                    const uint32_t d_off  = fr * 8u;
                    #pragma GCC unroll 8
                    for (uint32_t c = 0u; c < 8u; ++c) {
                        dst_tile[256u + d_off + c] = s_bot[0u + c];
                        dst_tile[384u + d_off + c] = s_bot[8u + c];
                    }
                }
            }
            cb_push_back(2, 1);
        }
    }
    if (unacked > 0u) {
        cb_wait_front(7, unacked);
        cb_pop_front(7, unacked);
    }
}
