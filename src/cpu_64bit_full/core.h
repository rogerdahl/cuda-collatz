extern "C" {
bool collatz_calc(
    u64* n_base, u64* delay_high,

    u32* sieve_buf, u64 sieve_size,

    u32* tail_buf, u64 tail_size,

    u64 step_bits, u64 d_high_bit, u32* step_table_c_d, u32* step_table_exp3,

    u64* total_loops_0, u64* total_loops_1);
}
