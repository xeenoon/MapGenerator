#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>

// Define constants for block size
#define BLOCK_SIZE 16

// Kernel to generate white noise
__global__ void GenerateWhiteNoiseKernel(float *noise, int width, int height, unsigned long long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        curandState state;
        curand_init(seed, x + y * width, 0, &state);
        noise[y * width + x] = curand_uniform(&state);
    }
}

// Kernel to interpolate values
__device__ float Interpolate(float x0, float x1, float alpha)
{
    return x0 * (1 - alpha) + alpha * x1;
}

// Kernel to generate smooth noise
__global__ void GenerateSmoothNoiseKernel(float *baseNoise, float *smoothNoise, int width, int height, int octave)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int samplePeriod = 1 << octave;
        float sampleFrequency = 1.0f / samplePeriod;

        int sample_i0 = (x / samplePeriod) * samplePeriod;
        int sample_i1 = (sample_i0 + samplePeriod) % width;
        float horizontal_blend = (x - sample_i0) * sampleFrequency;

        int sample_j0 = (y / samplePeriod) * samplePeriod;
        int sample_j1 = (sample_j0 + samplePeriod) % height;
        float vertical_blend = (y - sample_j0) * sampleFrequency;

        float top = Interpolate(baseNoise[sample_j0 * width + sample_i0], baseNoise[sample_j0 * width + sample_i1], horizontal_blend);
        float bottom = Interpolate(baseNoise[sample_j1 * width + sample_i0], baseNoise[sample_j1 * width + sample_i1], horizontal_blend);

        smoothNoise[y * width + x] = Interpolate(top, bottom, vertical_blend);
    }
}

// Function to generate white noise
float *GenerateWhiteNoise(int width, int height)
{
    float *d_noise, *h_noise;
    size_t size = width * height * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_noise, size);

    // Launch kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    GenerateWhiteNoiseKernel<<<gridSize, blockSize>>>(d_noise, width, height, time(NULL));

    // Allocate host memory
    h_noise = (float *)malloc(size);
    cudaMemcpy(h_noise, d_noise, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_noise);

    return h_noise;
}

// Function to generate smooth noise
float *GenerateSmoothNoise(float *baseNoise, int width, int height, int octave)
{
    float *d_baseNoise, *d_smoothNoise, *h_smoothNoise;
    size_t size = width * height * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_baseNoise, size);
    cudaMalloc(&d_smoothNoise, size);

    // Copy base noise to device
    cudaMemcpy(d_baseNoise, baseNoise, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    GenerateSmoothNoiseKernel<<<gridSize, blockSize>>>(d_baseNoise, d_smoothNoise, width, height, octave);

    // Copy result to host
    h_smoothNoise = (float *)malloc(size);
    cudaMemcpy(h_smoothNoise, d_smoothNoise, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_baseNoise);
    cudaFree(d_smoothNoise);

    return h_smoothNoise;
}

// Function to generate Perlin noise
extern "C" __declspec(dllexport) float *GeneratePerlinNoise(int width, int height, int octaveCount)
{
    float *baseNoise = GenerateWhiteNoise(width, height);

    float *perlinNoise = (float *)calloc(width * height, sizeof(float));
    float *smoothNoise;
    float amplitude = 1.0f;
    float totalAmplitude = 0.0f;

    for (int octave = octaveCount - 1; octave >= 0; octave--)
    {
        smoothNoise = GenerateSmoothNoise(baseNoise, width, height, octave);
        amplitude *= 0.7f;
        totalAmplitude += amplitude;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                perlinNoise[y * width + x] += smoothNoise[y * width + x] * amplitude;
            }
        }
        free(smoothNoise);
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            perlinNoise[y * width + x] /= totalAmplitude;
        }
    }

    free(baseNoise);
    return perlinNoise;
}

// Function to generate Perlin noise
extern "C" __declspec(dllexport) float *GeneratePerlinNoiseHost(int width, int height, int octaveCount)
{
    return GeneratePerlinNoise(width, height, octaveCount);
}
