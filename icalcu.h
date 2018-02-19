#include <cuda.h>
#include <cstdio>

#define THREADS 1024 
#define BLOCKS 32768 
#define NUM_VALS (THREADS*BLOCKS)