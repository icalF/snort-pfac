#include "icalcu.h"
#include <cuda.h>

int main(int argc, char **argv) 
{
    ptr_int h_array, d_array;
    int n = NUM_VALS;
    int err;

    // scanf("%d",&n);
    
    h_array = (ptr_int) malloc(n * sizeof(int));
    HANDLE_ERROR( cudaMalloc(&d_array, n * sizeof(int)) );

    // for(int i = 0; i < n; i++) 
    // {
    //     scanf("%d ", &h_array[i]);        // rand() % 2;
    // }

    // HANDLE_ERROR( cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice) );

    // HANDLE_ERROR( cudaEventCreate(&start) );
    // HANDLE_ERROR( cudaEventCreate(&stop) );
    // HANDLE_ERROR( cudaEventRecord(start, 0) );

    // reduce<<<BLOCKS, THREADS, SHM_SIZE>>>(d_array);
    // HANDLE_ERROR( cudaDeviceSynchronize() );

    // HANDLE_ERROR( cudaEventRecord(stop, 0) );
    // HANDLE_ERROR( cudaEventSynchronize(stop) );
    // HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );

    // reduce<<<1, THREADS, SHM_SIZE>>>(d_array);
    sort<<<BLOCKS, THREADS, SHM_SIZE>>>(d_array);
    // HANDLE_ERROR( cudaDeviceSynchronize() );

    // HANDLE_ERROR( cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost) );

    // for(int i = 0; i < n; i++) 
    // {
    //     printf("%d ", h_array[i]);
    // }
    // puts(h_array[0] ? "TRUE" : "False");

    cudaFree(d_array);
    free(h_array);    
}