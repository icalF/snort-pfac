#include "icalcu.h"
#include <cuda.h>

typedef int *ptr_int;

extern __global__ void sort(int* a, int start, int end, bool up);
extern __global__ void merge(int* a, int start, int end, bool up);

int main(int argc, char **argv) 
{
    ptr_int h_array, d_array;
    int n;

    scanf("%d",&n);
    
    h_array = (ptr_int) malloc(n * sizeof(int));
    cudaMalloc(&d_array, n * sizeof(int));

    for(int i = 0; i < n; i++) 
    {
        h_array[i] = rand() % 200 + 1;
        printf("%d ", h_array[i]);
    }
    puts("");

    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);

    sort<<<BLOCKS, THREADS>>>(d_array, 0, n, true);

    cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++) 
    {
        printf("%d ", h_array[i]);
    }
    puts("");

    cudaFree(d_array);
    free(h_array);    
}