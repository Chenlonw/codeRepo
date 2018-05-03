#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

double cpuSecond()
	/*< return cpu time>*/
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)(tp.tv_sec)+(double)tp.tv_usec*1.e-6);
}

__device__ int getGlobalIdx_1D_1D()
	/*< device get GlobalIdx with 1D grid 1D block >*/
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_1D_2D()
	/*< device get GlobalIdx with 1D grid 2D block >*/
{
	return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_1D_3D()
	/*< device get GlobalIdx with 1D grid 3D block >*/
{
	return blockIdx.x * blockDim.x * blockDim.y * blockDim.z \
	+ threadIdx.z * blockDim.x * blockDim.y \
	+ threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_2D_1D()
	/*< device get GlobalIdx with 2D grid 1D block >*/
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	return blockId * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_2D_2D()
	/*< device get GlobalIdx with 2D grid 2D block >*/
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	return blockId * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_2D_3D()
	/*< device get GlobalIdx with 2D grid 3D block >*/
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x * blockDim.y * blockDim.z \
	+ threadIdx.z * blockDim.x * blockDim.y \
	+ threadIdx.y * blockDim.x + threadIdx.x;
	return threadId;
}

__device__ int getGlobalIdx_3D_1D()
	/*< device get GlobalIdx with 3D grid 1D block >*/
{
	int blockId = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}

__device__ int getGlobalIdx_3D_2D()
	/*< device get GlobalIdx with 3D grid 2D block >*/
{
	int blockId = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
	return blockId * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_3D_3D()
	/*< device get GlobalIdx with 3D grid 3D block >*/
{
	int blockId = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x * blockDim.y * blockDim.z \
	+ threadIdx.z * blockDim.x * blockDim.y \
	+ threadIdx.y * blockDim.x + threadIdx.x ;
	return threadId;
}
