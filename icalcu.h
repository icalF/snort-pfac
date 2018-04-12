#include <cuda.h>
#include <cstdio>

typedef int *ptr_int;

#define THREADS 4
#define BLOCKS 2
#define SHM_SIZE (THREADS*sizeof(int))
#define NUM_VALS (2*THREADS*BLOCKS)

template <int BLOCKSIZE, int EXTRA_SIZE_PER_TB >
__global__ void PFAC_kernel_count ( size_t size, int *d_match_result, int *d_num_matched );