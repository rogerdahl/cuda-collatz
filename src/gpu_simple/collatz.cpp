#include "stdafx.h"

#include "collatz_kernel.h"

const int cuda_device(0);

using namespace std;
using namespace boost;

int main(int ac, char** av)
{
  // Set CUDA device.
  CUDA_SAFE_CALL(cudaSetDevice(cuda_device));

  // Create device result and host result buffers.
  u32* device_result_buf;
  CUDA_SAFE_CALL(
      cudaMalloc((void**)&device_result_buf, sizeof(u32) * threads_per_kernel));

  u32* host_result_buf;
  CUDA_SAFE_CALL(
      cudaMallocHost(
          (void**)&host_result_buf, sizeof(u32) * threads_per_kernel));

  // Start timer.
  timer t;

  // Check all N.
  u32 delay_high(0);
  for (u32 n_base(1); n_base < (1 << 26); n_base += threads_per_kernel) {
    // Run the calculation.
    run_collatz(n_base, device_result_buf);

    // Copy results from GPU to CPU memory.
    CUDA_SAFE_CALL(
        cudaMemcpy(
            host_result_buf, device_result_buf,
            sizeof(u32) * threads_per_kernel, cudaMemcpyDeviceToHost));

    for (u32 result_idx(0); result_idx < threads_per_kernel; ++result_idx) {
      u32 delay(host_result_buf[result_idx]);
      if (delay > delay_high) {
        cout << delay << " ";
        delay_high = delay;
      }
    }
  }

  // Show time.
  cout << endl;
  cout << t.elapsed() << endl;

  // Successful exit.
  return 0;
}
