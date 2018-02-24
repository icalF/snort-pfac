#include <cuda.h>
#include <cstdio>

typedef int *ptr_int;

#define THREADS 1024 
#define BLOCKS 512
#define SHM_SIZE (BLOCKS*sizeof(int))
#define NUM_VALS (2*THREADS*BLOCKS)

__global__ void reduce (ptr_int gd);