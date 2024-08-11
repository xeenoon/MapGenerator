#include <stdio.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#define BYTES_PER_PIXEL 4

// For drawing rocks
__global__ void CudaDrawKernel(
    int rockWidth, int rockHeight, int topleftX, int topleftY,
    int bakedrectangleLeft, int bakedrectangleTop, int bakedrectangleWidth, int bakedrectangleHeight,
    int rockcentreX, int rockcentreY, int shadowdst, int shinedst,
    unsigned char *resultbmp_scan0, unsigned char *rockbmp_scan0,
    unsigned char *bakeddistances_data, unsigned char *bakedbounds_data,
    int *filterarry, int resultwidth, int resultheight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x + topleftX;
    int y = blockIdx.y * blockDim.y + threadIdx.y + topleftY;

    if (x >= resultwidth || y >= resultheight || x < 0 || y < 0)
        return;
    int resultIndex = x * BYTES_PER_PIXEL + y * resultwidth * BYTES_PER_PIXEL;

    bool inpolygon = false;
    double distance = -1.0;

    int adjustedX = x - bakedrectangleLeft;
    int adjustedY = y - bakedrectangleTop;
    int checkidx = adjustedX * BYTES_PER_PIXEL + adjustedY * bakedrectangleWidth * BYTES_PER_PIXEL;

    if (adjustedX >= 0 && adjustedY >= 0 && adjustedX < bakedrectangleWidth && adjustedY < bakedrectangleHeight)
    {
        if (bakedbounds_data[checkidx + 2] == 255)
        {
            inpolygon = true;
        }
        distance = bakeddistances_data[checkidx];
    }

    if (inpolygon)
    {

        int rockIndex = (x % rockWidth) * BYTES_PER_PIXEL + (y % rockHeight) * rockWidth * BYTES_PER_PIXEL;
        memcpy(resultbmp_scan0 + resultIndex, rockbmp_scan0 + rockIndex, BYTES_PER_PIXEL);

        int xdist = rockcentreX - x;
        int ydist = rockcentreY - y;
        double centredst = sqrt((double)(xdist * xdist + ydist * ydist));

        if (centredst <= shinedst)
        {
            double shadowFactor = 1 + ((1.0 / 4.0) * (1 - (centredst / shinedst)));
            for (int i = 0; i < 3; ++i)
            {
                resultbmp_scan0[resultIndex + i] = min(resultbmp_scan0[resultIndex + i] * shadowFactor, 255.0);
            }
        }

        for (int i = 0; i < 3; ++i)
        {
            resultbmp_scan0[resultIndex + i] = min(255, filterarry[i] + resultbmp_scan0[resultIndex + i]);
        }
        resultbmp_scan0[resultIndex + 3] = 255;
    }

    if (distance <= shadowdst && distance != -1 && (distance == 0 ? inpolygon : true))
    {
        double shadowFactor = distance / (shadowdst * 3) + (1 - (1.0 / 3.0));
        for (int i = 0; i < 3; ++i)
        {
            resultbmp_scan0[resultIndex + i] = (unsigned char)(resultbmp_scan0[resultIndex + i] * shadowFactor);
        }
        resultbmp_scan0[resultIndex + 3] = 255;
    }
}

extern "C" __declspec(dllexport) uint8_t *CudaDraw(
    int rockcentreX, int rockcentreY, int topleftX, int topleftY, int bottomrightX, int bottomrightY,
    unsigned char *resultbmp_scan0, int bakedrectangleLeft, int bakedrectangleTop, int bakedrectangleWidth, int bakedrectangleHeight,
    unsigned char *bakeddistances_data, unsigned char *bakedbounds_data,
    int *filterarry, int resultwidth, int resultheight, unsigned char *rockbmp_scan0, int rockWidth, int rockHeight)
{
    const int shadowdst = 20;
    const int shinedst = 40;

    size_t imageSize = resultwidth * resultheight * BYTES_PER_PIXEL;

    // Allocate memory on the GPU for all variables
    unsigned char *d_resultbmp_scan0, *d_bakeddistances_data, *d_bakedbounds_data, *d_rockbmp_scan0;
    int *d_filterarry;

    cudaMalloc(&d_resultbmp_scan0, imageSize);
    cudaMalloc(&d_bakeddistances_data, bakedrectangleWidth * bakedrectangleHeight * BYTES_PER_PIXEL);
    cudaMalloc(&d_bakedbounds_data, bakedrectangleWidth * bakedrectangleHeight * BYTES_PER_PIXEL);
    cudaMalloc(&d_filterarry, 3 * sizeof(int)); // Assuming filterarry has 3 elements
    cudaMalloc(&d_rockbmp_scan0, rockWidth * rockHeight * BYTES_PER_PIXEL);

    // Copy data from host to GPU
    cudaMemcpy(d_resultbmp_scan0, resultbmp_scan0, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bakeddistances_data, bakeddistances_data, bakedrectangleWidth * bakedrectangleHeight * BYTES_PER_PIXEL, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bakedbounds_data, bakedbounds_data, bakedrectangleWidth * bakedrectangleHeight * BYTES_PER_PIXEL, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filterarry, filterarry, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rockbmp_scan0, rockbmp_scan0, rockWidth * rockHeight * BYTES_PER_PIXEL, cudaMemcpyHostToDevice);

    // Allocate memory for the output on the device
    unsigned char *d_output;
    cudaMalloc(&d_output, imageSize);

    // Copy the initial result data to the device
    cudaMemcpy(d_output, d_resultbmp_scan0, imageSize, cudaMemcpyDeviceToDevice);

    dim3 blockSize(16, 16);
    int xrange = bottomrightX - topleftX;
    int yrange = bottomrightY - topleftY;

    dim3 gridSize((xrange + blockSize.x - 1) / blockSize.x, (yrange + blockSize.y - 1) / blockSize.y);

    CudaDrawKernel<<<gridSize, blockSize>>>(
        rockWidth, rockHeight, topleftX, topleftY,
        bakedrectangleLeft, bakedrectangleTop, bakedrectangleWidth, bakedrectangleHeight,
        rockcentreX, rockcentreY, shadowdst, shinedst,
        d_output, d_rockbmp_scan0, d_bakeddistances_data, d_bakedbounds_data,
        d_filterarry, resultwidth, resultheight);
    cudaDeviceSynchronize();

    // Allocate memory on the host for the result
    uint8_t *h_output = new uint8_t[imageSize];

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_output);
    cudaFree(d_resultbmp_scan0);
    cudaFree(d_bakeddistances_data);
    cudaFree(d_bakedbounds_data);
    cudaFree(d_filterarry);
    cudaFree(d_rockbmp_scan0);

    return h_output; // Return the pointer to the host memory
}

// CUDA kernel for normal mapping
__global__ void ApplyNormalMapKernel(
    unsigned char *original, unsigned char *normal,
    unsigned char *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (x + y * width) * 4;

        // Pointers to the color values
        unsigned char *originalcolor = original + idx;
        unsigned char *normalcolor = normal + idx;

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
    unsigned char *original, unsigned char *normal,
    unsigned char *output, int width, int height)
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