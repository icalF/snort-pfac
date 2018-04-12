#include "icalcu.h"
#include <cuda.h>

int main(int argc, char **argv) 
{
    ptr_int h_array, d_array, d_res;
    int n = 11;

    int err;

    // scanf("%d",&n);
    
    h_array = (ptr_int) malloc(n * sizeof(int));
    err = cudaMalloc(&d_array, n * sizeof(int));    
    if (err) {
        puts("ALLOC ARR");
        return 0;
    }

    err = cudaMalloc(&d_res,   n * sizeof(int));
    if (err) {
        cudaFree(d_array);
        puts("ALLOC RES");
        return 0;
    }

    for(int i = 0; i < n; i++) 
    {
        h_array[i] = rand() % 5;        // rand() % 2;
        // printf("%d ", h_array[i]);
    }
    putchar('\n');

    err = cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err) {
        puts("COPY");
        cudaFree(d_array);
        return 0;
    }

    err = cudaMemset(d_res, 0, n * sizeof(int));
    if (err) {
        puts("COPY");
        cudaFree(d_array);
        return 0;
    }

    PFAC_kernel_count < THREADS, 0 > <<< BLOCKS, THREADS, SHM_SIZE >>> ( n, d_array, d_res );

    err = cudaMemcpy(h_array, d_res, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err) {
        puts("CPBK");
        printf("%d\n", err);
        cudaFree(d_array);
        return 0;
    }

    for(int i = 0; i < BLOCKS; i++) 
    printf("#%d : %d\n",i, h_array[i]);

    cudaFree(d_array);
    cudaFree(d_res);
    free(h_array);    
}

template <int BLOCKSIZE, int EXTRA_SIZE_PER_TB>
__global__ void PFAC_kernel_count ( size_t size, int *d_match_result, int *d_num_matched )
{
    unsigned tid = threadIdx.x;
    unsigned id = tid + ( 2 * blockDim.x ) * blockIdx.x;        // halve the blocks

    __shared__ int sdata[BLOCKSIZE + EXTRA_SIZE_PER_TB];

    if ( id >= size )
    {
        sdata[tid] = 0;
    }
    else 
    {
        int d1 = d_match_result[id] > 0;
        int d2 = ( id + blockDim.x ) >= size ? 0 : ( d_match_result[id + blockDim.x] > 0 );

        // read global data to shared memory
        // first level reduction
        sdata[tid] = d1 + d2;
    }
    __syncthreads();

    if ( id >= size ) 
    {
        return;
    }

    // reduce in block
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if ( tid < s ) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if ( tid == 0 ) // block root
    {
        d_num_matched[blockIdx.x] = sdata[0];
    }
}