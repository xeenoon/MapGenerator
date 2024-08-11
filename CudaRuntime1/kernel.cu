#include <stdio.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <fstream>
#include <stdarg.h>

#define BYTES_PER_PIXEL 4

std::ofstream logFile("C:\\Users\\ccw10\\Downloads\\segfaultlog.txt", std::ios::app);

void logMessage(const char *message, ...)
{
    if (logFile.is_open())
    {
        // Get current time
        std::time_t now = std::time(nullptr);
        std::tm *localTime = std::localtime(&now);

        // Write time to the file
        // logFile << std::asctime(localTime) << ": ";

        // Handle variable arguments
        va_list args;
        va_start(args, message);

        // Print formatted message to a temporary buffer
        char buffer[1024];
        vsnprintf(buffer, sizeof(buffer), message, args);

        // Write formatted message to the file
        logFile << buffer << std::endl;

        va_end(args);
    }
}

// For drawing rocks
__global__ void CudaDrawKernel(
    int rockWidth, int rockHeight, int topleftX, int topleftY,
    int bakedrectangleLeft, int bakedrectangleTop, int bakedrectangleWidth, int bakedrectangleHeight,
    int rockcentreX, int rockcentreY, int shadowdst, int shinedst,
    unsigned char *resultbmp_scan0, unsigned char *rockbmp_scan0,
    unsigned char *bakeddistances_data, unsigned char *bakedbounds_data,
    int *filterarry, int resultwidth, int resultheight)
{
    return; // This function literally does nothing🫠
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
    int *rockcentreXs, int *rockcentreYs, int *topleftXs, int *topleftYs, int *bottomrightXs, int *bottomrightYs,
    int *bakedrectangleLefts, int *bakedrectangleTops, int *bakedrectangleWidths, int *bakedrectangleHeights,
    unsigned char **bakeddistances_dataScan0s, unsigned char **bakedbounds_dataScan0s,
    int **filters, unsigned char *resultbmp_scan0, int resultwidth, int resultheight,
    unsigned char *rockbmp_scan0, int rockWidth, int rockHeight, int numItems)
{
    logMessage("Called from dll...");
    const int shadowdst = 20;
    const int shinedst = 40;

    size_t imageSize = resultwidth * resultheight * BYTES_PER_PIXEL;
    size_t rockImageSize = rockWidth * rockHeight * BYTES_PER_PIXEL;

    // Allocate memory on the GPU for all variables
    unsigned char *d_resultbmp_scan0;
    int *d_rockcentreXs, *d_rockcentreYs, *d_topleftXs, *d_topleftYs, *d_bottomrightXs, *d_bottomrightYs;
    int *d_bakedrectangleLefts, *d_bakedrectangleTops, *d_bakedrectangleWidths, *d_bakedrectangleHeights;
    unsigned char **d_bakeddistances_dataScan0s, **d_bakedbounds_dataScan0s;
    int **d_filters;
    unsigned char *d_rockbmp_scan0;

    cudaMalloc(&d_resultbmp_scan0, imageSize);
    cudaMalloc(&d_rockcentreXs, numItems * sizeof(int));
    cudaMalloc(&d_rockcentreYs, numItems * sizeof(int));
    cudaMalloc(&d_topleftXs, numItems * sizeof(int));
    cudaMalloc(&d_topleftYs, numItems * sizeof(int));
    cudaMalloc(&d_bottomrightXs, numItems * sizeof(int));
    cudaMalloc(&d_bottomrightYs, numItems * sizeof(int));
    cudaMalloc(&d_bakedrectangleLefts, numItems * sizeof(int));
    cudaMalloc(&d_bakedrectangleTops, numItems * sizeof(int));
    cudaMalloc(&d_bakedrectangleWidths, numItems * sizeof(int));
    cudaMalloc(&d_bakedrectangleHeights, numItems * sizeof(int));
    cudaMalloc(&d_bakeddistances_dataScan0s, numItems * sizeof(unsigned char *));
    cudaMalloc(&d_bakedbounds_dataScan0s, numItems * sizeof(unsigned char *));
    cudaMalloc(&d_filters, numItems * sizeof(int *));
    cudaMalloc(&d_rockbmp_scan0, rockImageSize);
    logMessage("Allocated arrays");

    // Copy the result bitmap to the device
    cudaError_t err;

    err = cudaMemcpy(d_resultbmp_scan0, resultbmp_scan0, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        logMessage("Crashed on memorycopy");
    }
    logMessage("Copied result bitmap");

    // Copy data arrays to the device
    cudaMemcpy(d_rockcentreXs, rockcentreXs, numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rockcentreYs, rockcentreYs, numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_topleftXs, topleftXs, numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_topleftYs, topleftYs, numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bottomrightXs, bottomrightXs, numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bottomrightYs, bottomrightYs, numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bakedrectangleLefts, bakedrectangleLefts, numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bakedrectangleTops, bakedrectangleTops, numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bakedrectangleWidths, bakedrectangleWidths, numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bakedrectangleHeights, bakedrectangleHeights, numItems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rockbmp_scan0, rockbmp_scan0, rockImageSize, cudaMemcpyHostToDevice);
    logMessage("Copied arrays");
    // Copy data pointers to the device and allocate memory for each array
    unsigned char **d_bakeddistances_dataScan0s_copy = new unsigned char *[numItems];
    unsigned char **d_bakedbounds_dataScan0s_copy = new unsigned char *[numItems];
    int **d_filters_copy = new int *[numItems];

    // Allocate memory on the device for each data pointer array
    for (int i = 0; i < numItems; ++i)
    {
        cudaMalloc(&d_bakeddistances_dataScan0s_copy[i], resultwidth * resultheight * BYTES_PER_PIXEL);
        cudaMemcpy(d_bakeddistances_dataScan0s_copy[i], bakeddistances_dataScan0s[i], resultwidth * resultheight * BYTES_PER_PIXEL, cudaMemcpyHostToDevice);

        cudaMalloc(&d_bakedbounds_dataScan0s_copy[i], resultwidth * resultheight * BYTES_PER_PIXEL);
        cudaMemcpy(d_bakedbounds_dataScan0s_copy[i], bakedbounds_dataScan0s[i], resultwidth * resultheight * BYTES_PER_PIXEL, cudaMemcpyHostToDevice);

        cudaMalloc(&d_filters_copy[i], sizeof(int));
        cudaMemcpy(d_filters_copy[i], filters[i], sizeof(int), cudaMemcpyHostToDevice);
    }
    logMessage("Copied 2d arrays");

    cudaMemcpy(d_bakeddistances_dataScan0s, d_bakeddistances_dataScan0s_copy, numItems * sizeof(unsigned char *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bakedbounds_dataScan0s, d_bakedbounds_dataScan0s_copy, numItems * sizeof(unsigned char *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filters, d_filters_copy, numItems * sizeof(int *), cudaMemcpyHostToDevice);

    // Allocate memory for the output on the device
    unsigned char *d_output;
    cudaMalloc(&d_output, imageSize);

    dim3 blockSize(16, 16);

    // Launch the kernel for each item
    for (int i = 0; i < numItems; ++i)
    {
        int xrange = bottomrightXs[i] - topleftXs[i];
        int yrange = bottomrightYs[i] - topleftYs[i];
        logMessage("Xrange: %d, Yrange %d", xrange, yrange);
        dim3 gridSize((xrange + blockSize.x - 1) / blockSize.x, (yrange + blockSize.y - 1) / blockSize.y);

        CudaDrawKernel<<<gridSize, blockSize>>>(
            rockWidth, rockHeight, d_topleftXs[i], d_topleftYs[i],
            d_bakedrectangleLefts[i], d_bakedrectangleTops[i], d_bakedrectangleWidths[i], d_bakedrectangleHeights[i],
            d_rockcentreXs[i], d_rockcentreYs[i], shadowdst, shinedst,
            d_output, d_rockbmp_scan0, d_bakeddistances_dataScan0s[i], d_bakedbounds_dataScan0s[i],
            d_filters[i], resultwidth, resultheight);
    }
    cudaDeviceSynchronize();

    // Allocate memory on the host for the result
    uint8_t *h_output = new uint8_t[imageSize];

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_output);
    cudaFree(d_resultbmp_scan0);
    cudaFree(d_rockcentreXs);
    cudaFree(d_rockcentreYs);
    cudaFree(d_topleftXs);
    cudaFree(d_topleftYs);
    cudaFree(d_bottomrightXs);
    cudaFree(d_bottomrightYs);
    cudaFree(d_bakedrectangleLefts);
    cudaFree(d_bakedrectangleTops);
    cudaFree(d_bakedrectangleWidths);
    cudaFree(d_bakedrectangleHeights);
    cudaFree(d_rockbmp_scan0);

    for (int i = 0; i < numItems; ++i)
    {
        cudaFree(d_bakeddistances_dataScan0s_copy[i]);
        cudaFree(d_bakedbounds_dataScan0s_copy[i]);
        cudaFree(d_filters_copy[i]);
    }
    delete[] d_bakeddistances_dataScan0s_copy;
    delete[] d_bakedbounds_dataScan0s_copy;
    delete[] d_filters_copy;

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