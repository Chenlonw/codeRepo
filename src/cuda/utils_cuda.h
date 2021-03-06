/* This file is automatically generated. DO NOT EDIT! */

#ifndef _utils_cuda_h
#define _utils_cuda_h


double cpuSecond();
/*< return cpu time>*/


__device__ int getGlobalIdx_1D_1D();
/*< device get GlobalIdx with 1D grid 1D block >*/


__device__ int getGlobalIdx_1D_2D();
/*< device get GlobalIdx with 1D grid 2D block >*/


__device__ int getGlobalIdx_1D_3D();
/*< device get GlobalIdx with 1D grid 3D block >*/


__device__ int getGlobalIdx_2D_1D();
/*< device get GlobalIdx with 2D grid 1D block >*/


__device__ int getGlobalIdx_2D_2D();
/*< device get GlobalIdx with 2D grid 2D block >*/


__device__ int getGlobalIdx_2D_3D();
/*< device get GlobalIdx with 2D grid 3D block >*/


__device__ int getGlobalIdx_3D_1D();
/*< device get GlobalIdx with 3D grid 1D block >*/


__device__ int getGlobalIdx_3D_2D();
/*< device get GlobalIdx with 3D grid 2D block >*/


__device__ int getGlobalIdx_3D_3D();
/*< device get GlobalIdx with 3D grid 3D block >*/

#endif
