#include <cstdlib>
#include <stdio.h>

#include "collatz_kernel.h"
#include "cuda_util.h"
#include "int_types.h"

// -ext=all --export-dir=$(PlatformName)\$(ConfigurationName)\collatz.devcode

// With smem=72 and reg=14, the Occupancy Calculator recommends 192 threads.
// However, 256 threads is slightly faster, which the Occupancy Calculator help
// indicates will happen if the threads are processor bound. How this kernel can
// be processor bound is a big mystery to me.
const u32 threads_per_block(512);

// Textures get special treatment by the compiler and are made available to the
// kernel automatically.
texture<uint4, 1, cudaReadModeElementType> tex_step;

// Linear memory
u32* device_sieve_linear[3];
u32* device_step_linear(0);
u32* device_tail_linear(0);

// Hold copies of these so that they don't have to be passed to run().
u32 sieve_size[3];
u32 step_bits(0);
u32 tail_size(0);

// Hold the current location on where store results in the result buf. This
// variable exists both on host and device.
__device__ u32 result_idx;

// This array is in shared memory. The array gets special treatment by the
// compiler. Memory for the array is dynamically allocated at runtime by the
// size specified in the 3rd argument in the <<<>>> syntax kernel call.
// extern __shared__ float shared[];

// Constant memory.
//__constant__ u32 delay_record_const[delay_record_const_size * 3];

// Increase the grid size by 1 if the block width or height does not divide
// evenly
// by the thread block dimensions
int div_up(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

// This is a place holder for the collatz kernel, required by CUDA. The actual
// kernel is implemented in PTX and is loaded, compiled and called instead of
// this one at runtime.
__global__ void collatz_kernel(
    u32 n_base_1, u32 n_base_0, // base
    u32 delay_high, // high delay
    u32* sieve_buf, // sieve
    u32 step_bits, // step
    u32* tail_buf, u32 tail_size, // tail
    result_record* result_buf) // result
{
  // uint4 q = tex1Dfetch(tex_step, 14);
  // 128 bit CUDA/PTX Collatz Delay Record Calculator (C)2008 Roger Dahl

  // This includes an implementation of the Early Break optimization. It outputs
  // two counts to debugging, one for how many times a check for Early Break is
  // done and one for how many times the Early Break is performed. However, even
  // though the Early Break was taken for all numbers in the region I tested on,
  // it halved the speed of the calculator instead of increasing it, so I have
  // disabled it. I don't know for sure that the algorithm works right and is
  // coded optimally. If it is, it could be that it either doesn't work well
  // with the multiple steps per iteration optimization or that it causes too
  // much divergence in the warp.

  //"  .version 1.2\n"
  //"  .target sm_13\n"

  //"	.tex .u64 tex_step;\n"
  //"	.global .u32 result_idx;\n"
  //"	.const .align 4 .b8 delay_record_const[3600];\n"

  //"	.entry _Z14collatz_kerneljjjPjjS_jP13result_record\n"
  //"	{\n"
  //"	.param .u32 _n_base_1;\n"
  //"	.param .u32 _n_base_0;\n"
  //"	.param .u32 _delay_high;\n"
  //"	.param .u64 _sieve_buf;\n"
  //"	.param .u32 _step_bits;\n"
  //"	.param .u64 _tail_buf;\n"
  //"	.param .u32 _tail_size;\n"
  //"	.param .u64 _result_buf;\n"

  // Temp registers.
  // asm("	.tex .u64 tex_step;");
  asm("	.reg .pred p;");
  asm("	.reg .u32 junk;");
  asm("	.reg .u16 u16_<2>;");
  asm("	.reg .u32 u32_<4>;");

  // Find the index of the N that this thread will check.
  // const int n_idx(blockDim.x * blockIdx.x + threadIdx.x);
  asm("	.reg .u32 n_idx;");
  asm("	mov.u16 u16_0, %ctaid.x;");
  asm("	mov.u16 u16_1, %ntid.x;");
  asm("	mul.wide.u16 n_idx, u16_1, u16_0;");
  asm("	cvt.u32.u16 u32_0, %tid.x;");
  asm("	add.u32 n_idx, u32_0, n_idx;");

  // Set up the starting N.
  // u128 n(n_base(128) + tex1Dfetch(tex_sieve, n_idx))(32);
  asm("	.reg .u32 n_<4>;");
  asm("	mov.u32 n_0, %0;" ::"r"(n_base_0)); // line 43, 65
  asm("	mov.u32 n_1, %0;" ::"r"(n_base_1));
  asm("	mov.u32 n_2, 0;");
  asm("	mov.u32 n_3, 0;");

  // Early Break optimization
  // asm("	.reg .u32 sieve_buf;");
  // asm("	ld.param.u32 sieve_buf, [_sieve_buf];");
  // asm("	shl.b32 u32_0, n_idx, 2;");
  // asm("	add.u32 sieve_buf, sieve_buf, u32_0;");
  // asm("	.reg .u32 sieve_val;");
  // asm("	ld.global.u32 sieve_val, [sieve_buf];");
  //
  // asm("	add.cc.u32 n_0, n_0, sieve_val;");
  // asm("	addc.cc.u32 n_1, n_1, 0;");
  // asm("	addc.cc.u32 n_2, n_2, 0;");
  // asm("	addc.u32 n_3, n_3, 0;");

  asm("	.reg .u64 sieve_buf;");
  asm("	.reg .u64 tmp_u64;");
  asm(" cvt.u64.u32 tmp_u64, n_idx;");
  asm("	shl.b64 sieve_buf, tmp_u64, 2;");
  asm("	add.u64 sieve_buf, sieve_buf, %0;" ::"l"(sieve_buf));
  asm("	.reg .u32 sieve_val;");
  asm("	ld.global.u32 sieve_val, [sieve_buf];");

  asm("	add.cc.u32 n_0, n_0, sieve_val;");
  asm("	addc.cc.u32 n_1, n_1, 0;");
  asm("	addc.cc.u32 n_2, n_2, 0;");
  asm("	addc.u32 n_3, n_3, 0;");

  // u128 n_tmp(n)
  asm("	.reg .u32 n_tmp_<4>;");
  asm("	mov.u32 n_tmp_0, n_0;");
  asm("	mov.u32 n_tmp_1, n_1;");
  asm("	mov.u32 n_tmp_2, n_2;");
  asm("	mov.u32 n_tmp_3, n_3;");

  // u32 delay(0);
  asm("	.reg .u32 delay;");
  asm("	mov.u32 delay, 0;");

  // Prepare some constants.
  asm("	.reg .u32 step_bits;");
  asm("	.reg .u32 step_bits_compl;");
  // asm("	ld.param.u32 step_bits, [_step_bits];");
  asm("	mov.u32 step_bits, %0;" ::"r"(step_bits));
  asm("	sub.u32 step_bits_compl, 32, step_bits;");

  asm("	.reg .u32 step_size;");
  asm("	shl.b32 step_size, 1, step_bits;");

  asm("	.reg .u32 step_mask;");
  asm("	sub.u32 step_mask, step_size, 1;");

  asm("	.reg .u32 tail_size;");
  asm("	mov.u32 tail_size, %0;" ::"r"(tail_size));

  asm("	.reg .u64 result_buf;");
  asm("	mov.u64 result_buf, %0;" ::"l"(result_buf));

  // We optimize the loop so that it checks only at the end. This would
  // invalidate results if we started calculating at n_tmp < step_size but we
  // always start calculating abobe that.

  // while (n_tmp >= step_size) {\n
  asm("	$main_loop:");
  // u32 b(n_tmp & step_mask);
  asm("		.reg .u32 b;");
  asm("		and.b32 b, step_mask, n_tmp_0;");

  // u128 a(n_tmp >> step_bits);
  asm("		.reg .u32 a_<4>;");

  asm("		shl.b32 u32_0, n_tmp_1, step_bits_compl;");
  asm("		shl.b32 u32_1, n_tmp_2, step_bits_compl;");
  asm("		shl.b32 u32_2, n_tmp_3, step_bits_compl;");

  asm("		shr.b32 n_tmp_0, n_tmp_0, step_bits;");
  asm("		shr.b32 n_tmp_1, n_tmp_1, step_bits;");
  asm("		shr.b32 n_tmp_2, n_tmp_2, step_bits;");
  asm("		shr.b32 a_3, n_tmp_3, step_bits;");

  asm("		or.b32 a_0, n_tmp_0, u32_0;");
  asm("		or.b32 a_1, n_tmp_1, u32_1;");
  asm("		or.b32 a_2, n_tmp_2, u32_2;");

  // Get exp3, c and d values from the tables.
  asm("		.reg .u32 exp3;");
  asm("		.reg .u32 c;");
  asm("		.reg .u32 d;");
  // asm("		tex.3d.v4.u32.s32 {exp3, c, d, junk}, [tex_step, {b, b, b, b}];");
  asm("		tex.1d.v4.u32.s32 {exp3, c, d, junk}, [tex_step, {b}];");

  // n_tmp = a * exp3;
  asm("		mul.lo.u32 n_tmp_0, a_0, exp3;");
  asm("		mul.hi.u32 u32_0, a_0, exp3;");
  asm("		mad.lo.u32 n_tmp_1, a_1, exp3, u32_0;");
  asm("		mul.hi.u32 u32_0, a_1, exp3;");
  asm("		mad.lo.u32 n_tmp_2, a_2, exp3, u32_0;");
  asm("		mul.hi.u32 u32_0, a_2, exp3;");
  asm("		mad.lo.u32 n_tmp_3, a_3, exp3, u32_0;");

  // n_tmp += d;
  asm("		add.cc.u32 n_tmp_0, n_tmp_0, d;");
  asm("		addc.cc.u32	n_tmp_1, n_tmp_1, 0;");
  asm("		addc.cc.u32	n_tmp_2, n_tmp_2, 0;");
  asm("		addc.u32 n_tmp_3, n_tmp_3, 0;");

  // delay += step_bits + c;
  asm("		add.u32 u32_0, c, step_bits;");
  asm("		add.u32 delay, delay, u32_0;");

  // Keep iterating as long as N is higher than 64 bit.
  asm("		setp.ne.u32 p, n_tmp_2, 0;");
  asm("		@p bra $main_loop;");
  asm("		setp.ne.u32 p, n_tmp_3, 0;");
  asm("		@p bra $main_loop;");

  // while (n_tmp >= tail_size)
  asm("		setp.ne.u32 p, n_tmp_1, 0;");
  asm("		@p bra $main_loop;");
  asm("		setp.ge.u32 p, n_tmp_0, tail_size;");
  asm("		@p bra $main_loop;");

  asm("		bra.uni $end_loop;");
  asm("	$end_loop:");

  // delay += tex1D(tex_tail, n_tmp);
  asm("	.reg .u64 tail_buf;");
  asm(" cvt.u64.u32 tmp_u64, n_tmp_0;");
  asm("	shl.b64 tail_buf, tmp_u64, 2;");
  asm("	add.u64 tail_buf, tail_buf, %0;" ::"l"(tail_buf));
  asm("	ld.global.u32 u32_0, [tail_buf];");
  asm("	add.u32 delay, delay, u32_0;");

  // if (delay > delay_high) {
  // asm("	ld.param.u32 u32_0, [_delay_high];");
  asm("	setp.le.u32 p, delay, %0;" ::"r"(delay_high));
  asm("	@p exit;");

  // Each Collatz Record is 8 bytes long. Calculate offset.
  // Update global pointer.
  asm("	.reg .u32 collatz_record_offset;");
  asm("	atom.global.inc.u32 collatz_record_offset, [result_idx], 100;");
  asm("	shl.b32 collatz_record_offset, collatz_record_offset, 3;");
  asm(" cvt.u64.u32 tmp_u64, collatz_record_offset;");
  asm("	add.u64 tmp_u64, tmp_u64, result_buf;");
  asm("	st.global.u32 [tmp_u64 + 0], sieve_val;");
  asm("	st.global.u32 [tmp_u64 + 4], delay;");
}

// Prepare data for run().
void init_collatz(
    u32** _host_sieve, u32* _sieve_size, u32* _c, u32* _d, u32* _exp3,
    u32 _step_bits,
    // u32* _delay_record_buf, u32 _delay_record_buf_size,
    u32* _host_tail, u32 _tail_size)
{
  // Store various.
  sieve_size[0] = _sieve_size[0];
  sieve_size[1] = _sieve_size[1];
  sieve_size[2] = _sieve_size[2];
  step_bits = _step_bits;
  tail_size = _tail_size;

  // Channel descriptions.
  cudaChannelFormatDesc ch_uint4 =
      cudaCreateChannelDesc<uint4>(); // uint3 is not supported.
  cudaChannelFormatDesc ch_u32 = cudaCreateChannelDesc<u32>();

  u32 head_bytes(sizeof(u32) * threads_per_block);

  // Create linear memory arrays containing the sieve indexes. We want to
  // preserve the entire texture cache for the c, d and exp3 table lookups that
  // are in the inner loop. So we use uncached global memory for the sieve
  // indexes.
  for (int i(0); i < 3; ++i) {
    u32 sieve_bytes(sizeof(u32) * _sieve_size[i]);
    checkCudaErrors(
        cudaMalloc((void**)&device_sieve_linear[i], sieve_bytes + head_bytes));
    checkCudaErrors(
        cudaMemset((void*)device_sieve_linear[i], 0, sieve_bytes + head_bytes));
    checkCudaErrors(
        cudaMemcpy(
            device_sieve_linear[i], _host_sieve[i], sieve_bytes,
            cudaMemcpyHostToDevice));
  }

  // We stuff the c, d and exp3 tables into a single texture so that the values
  // can be looked up with a single tex lookup. We use linear memory. We use a
  // four integer vector. The fourth integer is unused padding.
  u32 step_size(1LL << _step_bits);
  checkCudaErrors(
      cudaMalloc(
          (void**)&device_step_linear, sizeof(uint4) * step_size + head_bytes));
  uint4* t = new uint4[step_size];
  for (u32 i(0); i < step_size; ++i) {
    t[i] = make_uint4(_exp3[i], _c[i], _d[i], 0);
  }
  checkCudaErrors(
      cudaMemcpy(
          device_step_linear, t, sizeof(uint4) * step_size,
          cudaMemcpyHostToDevice));

  //// Set texture parameters
  // tex_step.addressMode[0] = cudaAddressModeClamp;
  // tex_step.addressMode[1] = cudaAddressModeClamp;
  // tex_step.filterMode = cudaFilterModePoint;
  // tex_step.normalized = false;

  // An "invalid texture reference" error here probably means that there's
  // a problem with the external ptx file (compute_xx file).
  checkCudaErrors(cudaBindTexture(0, tex_step, device_step_linear, ch_uint4));

  delete[] t;

  //// Create const memory tables containing the known delay records for use by
  //// the Early Break optimization.
  // checkCudaErrors(cudaMemcpyToSymbol("delay_record_const",
  //	(void*)_delay_record_buf, sizeof(u32) * _delay_record_buf_size));

  // Create an uncached global, linear memory array containing the tail.
  checkCudaErrors(
      cudaMalloc((void**)&device_tail_linear, sizeof(u32) * _tail_size));
  checkCudaErrors(
      cudaMemcpy(
          device_tail_linear, _host_tail, sizeof(u32) * _tail_size,
          cudaMemcpyHostToDevice));
}

// Async. Start GPU collatz calculator threads to process one buf of N.
void run_collatz(
    u64 n_base, u32 delay_high, result_record* device_result_buf,
    cudaStream_t stream)
{
  // set up the global result index counter
  // result_idx = 0;
  // checkCudaErrors(cudaMemcpyToSymbol(result_idx, (void*)&result_idx,
  // sizeof(result_idx)));
  u32* result_idx_ptr;
  checkCudaErrors(cudaGetSymbolAddress((void**)&result_idx_ptr, result_idx));
  checkCudaErrors(cudaMemset(result_idx_ptr, 0, sizeof(result_idx)));

  // Clear out the device result record.
  checkCudaErrors(
      cudaMemset(
          (void*)device_result_buf, 0xffffffff,
          sizeof(result_record) * result_record_buf_size));

  // DEBUG: Clear out the debug area.
  // checkCudaErrors(cudaMemset((void*)(device_result_buf + 50), 0,
  // sizeof(result_record) * 10));

  // Pick the sieve that filters out 3k+2 values for this n_base.
  u32 sieve_mod(static_cast<u32>(n_base % 3));

  // Determine how many blocks we'll need to run.
  u32 blocks(div_up(sieve_size[sieve_mod], threads_per_block));

  // Dg specifies the dimension and size of the grid, such that Dg.x * Dg.y
  // equals the number of blocks being launched; Dg.z is unused.
  dim3 Dg(blocks, 1);
  //	dim3 Dg(1, 1);

  // Db specifies the dimension and size of each block, such that Db.x * Db.y *
  // Db.z equals the number of threads per block.
  dim3 Db(threads_per_block, 1);
  //	dim3 Db(1, 1);

  collatz_kernel<<<Dg, Db, 0, stream>>>(
      static_cast<u32>(n_base >> 32), static_cast<u32>(n_base), // base
      delay_high, // high delay
      device_sieve_linear[sieve_mod], // sieve
      step_bits, // step
      device_tail_linear, tail_size, device_result_buf);

  cudaError_t err(cudaGetLastError());
  if (err != cudaSuccess) {
    printf("Error: Kernel launch failed. %s\n", cudaGetErrorString(err));
    exit(1);
  }
}

// Cleanup.
void clean_collatz()
{
  checkCudaErrors(cudaFree(device_sieve_linear[0]));
  checkCudaErrors(cudaFree(device_sieve_linear[1]));
  checkCudaErrors(cudaFree(device_sieve_linear[2]));
  checkCudaErrors(cudaFree(device_step_linear));
  checkCudaErrors(cudaFree(device_tail_linear));
}
