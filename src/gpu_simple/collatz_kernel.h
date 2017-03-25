#pragma once

#include "int_types.h"
#include <vector_types.h>

const u32 threads_per_kernel(32768);

extern "C" void run_collatz(u32 n_base, u32* device_result_buf);
