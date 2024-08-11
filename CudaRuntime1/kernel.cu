#include <stdio.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#define SIZE 1024

__global__ void VectorAddKernel(int* a, int* b, int* c, int n)
{
    int i = threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

extern "C" __declspec(dllexport) int* ExtVectorAdd(int* a, int* b)
{
    int* c;
    cudaMallocManaged(&c, SIZE * sizeof(int));

    // Launch the kernel
    VectorAddKernel <<<1, SIZE>>>(a, b, c, SIZE);

    // Synchronize to ensure the kernel has finished
    cudaDeviceSynchronize();

    // Return the result array
    return c;
}

extern "C" __declspec(dllexport) void FreeMemory(int* ptr)
{
    cudaFree(ptr);
}
