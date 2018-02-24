#include "icalcu.h"
#include <cuda.h>

int main(int argc, char **argv) 
{
    ptr_int h_array, d_array;
    int n = NUM_VALS;

    int err;

    // scanf("%d",&n);
    
    h_array = (ptr_int) malloc(n * sizeof(int));
    err = cudaMalloc(&d_array, n * sizeof(int));

    if (err) {
        puts("BEGIN");
        return 0;
    }

    for(int i = 0; i < n; i++) 
    {
        h_array[i] = 0;        // rand() % 2;
    }
    h_array[rand() % NUM_VALS] = 1;

    err = cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);

    if (err) {
        puts("COPY");
        cudaFree(d_array);
        return 0;
    }

    reduce<<<BLOCKS, THREADS, SHM_SIZE>>>(d_array);
    err = cudaDeviceSynchronize();

    if (err) {
        puts("SYNC");
        printf("%d\n", err);
        cudaFree(d_array);
        return 0;
    }

    reduce<<<1, THREADS, SHM_SIZE>>>(d_array);

    err = cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);

    if (err) {
        puts("CPBK");
        printf("%d\n", err);
        cudaFree(d_array);
        return 0;
    }

    puts(h_array[0] ? "TRUE" : "False");

    cudaFree(d_array);
    free(h_array);    
}