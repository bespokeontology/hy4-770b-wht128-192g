// Blackhole P150A Hy4-flash SwiGLU(10.0) (+ WHT128) Writer (RISCV_0 / NOC_1)
// Writes [m_cnt, 4] Float16_b (BF16) tiles from CB 16 into L1 z_rot_bf16 [m_tiles_total, 24].

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t z_rot_addr = get_arg_val<uint32_t>(0);
    const uint32_t nb4        = get_arg_val<uint32_t>(1);  // 0..5
    const uint32_t m_start    = get_arg_val<uint32_t>(2);
    const uint32_t m_cnt      = get_arg_val<uint32_t>(3);

    constexpr auto z_args = TensorAccessorArgs<0>();
    const auto z_acc = TensorAccessor(z_args, z_rot_addr, 2048);

    constexpr uint32_t nt_total = 32u;
    for (uint32_t dm = 0u; dm < m_cnt; ++dm) {
        const uint32_t m = m_start + dm;
        cb_wait_front(16, 4);
        const uint32_t src_l1 = get_read_ptr(16);
        const uint32_t dst_base = m * nt_total + nb4 * 4u;
        for (uint32_t n = 0u; n < 4u; ++n) {
            noc_async_write_tile(dst_base + n, z_acc, src_l1 + n * 2048u);
        }
        noc_async_writes_flushed();
        cb_pop_front(16, 4);
    }
    if (m_cnt > 0u) {
        noc_async_write_barrier();
    }
}
