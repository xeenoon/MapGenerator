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

extern "C" __declspec(dllexport) int* ExtVectorAdd(int* a, int* b, int n)
{
    int *d_a, *d_b, *d_c;
    int* h_c = new int[n]; // Allocate host memory for result

    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));

    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    VectorAddKernel<<<1, n>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return h_c; // Return host memory pointer
}

extern "C" __declspec(dllexport) void FreeMemory(int* ptr)
{
    delete[] ptr; // Free host memory
}
