#include <cuda.h>
#include "cuda_utils.h"
#include <cstdio>

typedef int *ptr_int;

#define THREADS 4 
#define BLOCKS 2
#define SHM_SIZE (THREADS*sizeof(int))
#define NUM_VALS (THREADS*BLOCKS)

// __global__ void reduce (ptr_int gd);
void sort (ptr_int data, int len);
__global__ void localMerge (ptr_int data, int superBlock);
__global__ void globalMerge (ptr_int data, int blockSize, int superBlock);
__global__ void localSort (ptr_int data);
__device__ __forceinline__ void sorter (int& a, int& b, int desc);
__device__ __forceinline__ bool blockHalves(int id, int blockSize);