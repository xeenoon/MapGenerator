﻿#include <stdio.h>
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

// CUDA kernel for normal mapping
__global__ void ApplyNormalMapKernel(
    unsigned char* original, unsigned char* normal, 
    unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (x + y * width) * 4;
        
        // Pointers to the color values
        unsigned char* originalcolor = original + idx;
        unsigned char* normalcolor = normal + idx;
        
        // Convert normal color to a vector
        float nx = (normalcolor[2] / 255.0f) * 2.0f - 1.0f;
        float ny = (normalcolor[1] / 255.0f) * 2.0f - 1.0f;
        float nz = (normalcolor[0] / 255.0f) * 2.0f - 1.0f;

        // Simple light direction (from top-left)
        float lx = 0.5f;
        float ly = -0.5f;
        float lz = 1.0f;

        // Normalize the light direction
        float length = sqrtf(lx * lx + ly * ly + lz * lz);
        lx /= length;
        ly /= length;
        lz /= length;

        // Compute the dot product of the normal and light direction
        float dot = nx * lx + ny * ly + nz * lz;
        dot = fmaxf(0.0f, dot); // Clamp to [0, 1]

        // Apply the dot product to the original color to get the shaded color
        output[idx + 2] = (unsigned char)(originalcolor[2] * dot);
        output[idx + 1] = (unsigned char)(originalcolor[1] * dot);
        output[idx + 0] = (unsigned char)(originalcolor[0] * dot);
        output[idx + 3] = originalcolor[3]; // Preserve alpha channel
    }
}

extern "C" __declspec(dllexport) void ApplyNormalMap(
    unsigned char* original, unsigned char* normal, 
    unsigned char* output, int width, int height)
{
    unsigned char *d_original, *d_normal, *d_output;
    size_t imageSize = width * height * 4 * sizeof(unsigned char);

    cudaMalloc(&d_original, imageSize);
    cudaMalloc(&d_normal, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_original, original, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, normal, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    ApplyNormalMapKernel<<<gridSize, blockSize>>>(d_original, d_normal, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_original);
    cudaFree(d_normal);
    cudaFree(d_output);
}