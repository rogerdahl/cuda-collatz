#include "collatz_kernel.h"
#include "cutil.h"
#include <stdio.h>

// The collatz delay calculator kernel.
__global__ void collatz_thread(u32 n_base, u32* device_result_buf)
{
  u32 idx(blockDim.x * blockIdx.x + threadIdx.x);
  u32 n(n_base + idx);

  u32 delay(0);
  while (n > 1) {
    if (n & 1) {
      n = (n << 1) + n + 1;
    }
    else {
      n >>= 1;
    }
    ++delay;
  }

  device_result_buf[idx] = delay;
}

void run_collatz(u32 n_base, u32* device_result_buf)
{
  // Start 32768 / 256 blocks, each with 256 threads.
  dim3 Dg(32768 / 256, 1);
  dim3 Db(256, 1);
  collatz_thread<<<Dg, Db>>>(n_base, device_result_buf);
}
