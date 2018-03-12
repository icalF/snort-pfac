#include "icalcu.h"
#include <cuda.h>

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

int main(int argc, char **argv) 
{
    ptr_int h_array, d_array;
    
    int n = NUM_VALS;
    int err;

    // cudaDeviceProp props;
    // cudaGetDeviceProperties(&props, 0);
    // printDevProp(props);

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

    for(int i = 0; i < n; i++) 
    {
        printf("%d ", h_array[i]);
    }
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
        /* Global merging */
        // for (int innerBlock = block; innerBlock > THREADS; innerBlock >>= 1)
        // {
        //     globalMerge<<<BLOCKS, THREADS>>>(data, innerBlock, block);          
        //     HANDLE_ERROR( cudaDeviceSynchronize() );
        // }

        // /* Local merging */        
        // localMerge<<<BLOCKS, THREADS, SHM_SIZE>>>(data, block);
        // HANDLE_ERROR( cudaDeviceSynchronize() );
    }
    
    HANDLE_ERROR( cudaEventRecord(stop, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
    
    printf("%lf ms\n", time);
}