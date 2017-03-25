#include <cuda_runtime.h>

void __checkCudaErrors(cudaError err, const char* file, const int line);
void __getLastCudaError(
    const char* errorMessage, const char* file, const int line);

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// TODO!!!! THIS SYNCS. NEED SEPARATE IMPLEMENTATION.
#define checkCudaErrorsNoSync(err) __checkCudaErrors(err, __FILE__, __LINE__)

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
