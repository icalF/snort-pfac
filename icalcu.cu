#include "icalcu.h"
#include <cuda.h>

int main(int argc, char **argv) 
{
    ptr_int h_array, d_array;
    
    int n = NUM_VALS;
    int err;

    // scanf("%d",&n);
    
    h_array = (ptr_int) malloc(n * sizeof(int));

    for(int i = 0; i < n; i++) 
    {
        h_array[i] = rand() % 452986;
    }

    HANDLE_ERROR( cudaMalloc(&d_array, n * sizeof(int)) );
    HANDLE_ERROR( cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice) );

    sort(d_array, n);

    // reduce<<<BLOCKS, THREADS, SHM_SIZE>>>(d_array);
    // HANDLE_ERROR( cudaDeviceSynchronize() );

    // reduce<<<1, THREADS, SHM_SIZE>>>(d_array);
    
    // HANDLE_ERROR( cudaDeviceSynchronize() );

    HANDLE_ERROR( cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost) );

    // for(int i = 0; i < n; i++) 
    // {
    //     printf("%d ", h_array[i]);
    // }
    // puts(h_array[0] ? "TRUE" : "False");
    
    cudaFree(d_array);
    free(h_array);    
}

void sort (ptr_int data, int len)
{   
    cudaEvent_t start, stop;
    float time;
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    localSort<<<BLOCKS, THREADS, SHM_SIZE>>>(data);
    HANDLE_ERROR( cudaDeviceSynchronize() );

    for (int block = THREADS << 1; block <= len; block <<= 1)
    {
        /* Aligning */
        globalAlign<<<BLOCKS, THREADS>>>(data, block);
        HANDLE_ERROR( cudaDeviceSynchronize() );

        /* Global merging */
        for (int innerBlock = block >> 1; innerBlock > THREADS; innerBlock >>= 1)
        {
            globalMerge<<<BLOCKS, THREADS>>>(data, innerBlock);          
            HANDLE_ERROR( cudaDeviceSynchronize() );
        }

        /* Local merging */
        for (int innerBlock = THREADS; innerBlock > 1; innerBlock >>= 1)
        {
            localMerge<<<BLOCKS, THREADS, SHM_SIZE>>>(data, innerBlock);
            HANDLE_ERROR( cudaDeviceSynchronize() );
        }
    }
    
    HANDLE_ERROR( cudaEventRecord(stop, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
    
    printf("%lf ms\n", time);
}