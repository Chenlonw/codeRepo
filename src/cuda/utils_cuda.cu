#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

__device__ int getGlobalIdx_1D_1D()
	/*< device get GlobalIdx with 1D grid 1D block >*/
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}

