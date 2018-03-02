#include <cuda.h>
#include "cuda_utils.h"
#include <cstdio>

typedef int *ptr_int;

#define THREADS 4//1024 
#define BLOCKS 2//512
#define SHM_SIZE (THREADS*sizeof(int))
#define NUM_VALS (THREADS*BLOCKS)

// __global__ void reduce (ptr_int gd);
__global__ void sort (ptr_int data);
__device__ void sorter (int& a, int& b);
__device__ __forceinline__ bool blockHalves(int id, int blockSize);