#include "icalcu.h"
#include <cuda.h>

int main () 
{
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();
}