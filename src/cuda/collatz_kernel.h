#pragma once

#include "int_types.h"
#include <vector_types.h>

const u32 result_record_buf_size(100);
// const u32 delay_record_const_size(300);

// struct delay_record {
//	delay_record() {}
//	delay_record(u64 n, u32 delay) : n_(n), delay_(delay) { }
//	u64 n_;
//	u16 delay_;
//};

struct result_record
{
  result_record()
  {
  }
  result_record(u32 n_idx, u32 delay) : n_sieve_(n_idx), delay_(delay)
  {
  }
  u32 n_sieve_;
  u32 delay_;
};

extern "C" void init_collatz(
    u32** _host_sieve, u32* _sieve_size, // sieve
    u32* _c, u32* _d, u32* _exp3, u32 _step_size, // step
    // u32* _delay_record_buf, u32 _delay_record_buf_size, // delay records
    u32* _host_tail, u32 _tail_size // tail
    );

extern "C" void run_collatz(
    u64 n_base, u32 delay_high, result_record* device_result_buf,
    cudaStream_t stream);

extern "C" void clean_collatz();
