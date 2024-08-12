#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// Define the size of the output bitmap
__device__ bool isPointInTriangle(int x, int y, int x0, int y0, int x1, int y1, int x2, int y2)
{
    // Calculate vectors
    int v0x = x2 - x0;
    int v0y = y2 - y0;
    int v1x = x1 - x0;
    int v1y = y1 - y0;
    int v2x = x - x0;
    int v2y = y - y0;

    // Calculate dot products
    int dot00 = v0x * v0x + v0y * v0y;
    int dot01 = v0x * v1x + v0y * v1y;
    int dot02 = v0x * v2x + v0y * v2y;
    int dot11 = v1x * v1x + v1y * v1y;
    int dot12 = v1x * v2x + v1y * v2y;

    // Calculate barycentric coordinates
    int denom = dot00 * dot11 - dot01 * dot01;
    if (denom == 0)
    {
        return false; // Triangle is degenerate
    }
    float invDenom = 1.0f / denom;
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    // Check if point is in triangle
    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

__device__ void rasterizeTriangle(int x0, int y0, int x1, int y1, int x2, int y2, unsigned char *color, uint8_t *bitmap, int width, int height)
{
    int minX = min(x0, min(x1, x2));
    int minY = min(y0, min(y1, y2));
    int maxX = max(x0, max(x1, x2));
    int maxY = max(y0, max(y1, y2));

    // Clipping to the bitmap bounds
    minX = max(minX, 0);
    minY = max(minY, 0);
    maxX = min(maxX, width - 1);
    maxY = min(maxY, height - 1);

    for (int y = minY; y <= maxY; y++)
    {
        for (int x = minX; x <= maxX; x++)
        {
            int idx = (y * width + x) * 4;

            if (isPointInTriangle(x, y, x0, y0, x1, y1, x2, y2))
            {
                bitmap[idx] = color[0];     // B
                bitmap[idx + 1] = color[1]; // G
                bitmap[idx + 2] = color[2]; // R
                bitmap[idx + 3] = color[3]; // A
            }
        }
    }
}

__global__ void drawTrianglesKernel(int *x, int *y, int *indices, int numTriangles, unsigned char *color, uint8_t *bitmap, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numTriangles)
    {
        int idx0 = indices[i * 3];
        int idx1 = indices[i * 3 + 1];
        int idx2 = indices[i * 3 + 2];

        int x0 = x[idx0];
        int y0 = y[idx0];
        int x1 = x[idx1];
        int y1 = y[idx1];
        int x2 = x[idx2];
        int y2 = y[idx2];

        rasterizeTriangle(x0, y0, x1, y1, x2, y2, color, bitmap, width, height);
    }
}

extern "C" __declspec(dllexport) unsigned char *drawPolygonWithTriangles(int *x, int *y, int *indices, int numTriangles, uint8_t *color, uint8_t *bitmap, int width, int height)
{
    uint8_t *d_bitmap;
    uint8_t *d_color;
    int *d_x;
    int *d_y;
    int *d_indices;
    size_t bitmapsize = width * height * 4;
    int vertexcount = numTriangles + 2;

    // Allocate memory on the device
    cudaMalloc((void **)&d_bitmap, bitmapsize);
    cudaMalloc((void **)&d_color, 4);
    cudaMalloc((void **)&d_x, vertexcount * sizeof(int));
    cudaMalloc((void **)&d_y, vertexcount * sizeof(int));
    cudaMalloc((void **)&d_indices, numTriangles * 3 * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_color, color, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, vertexcount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, vertexcount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, numTriangles * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bitmap, bitmap, bitmapsize, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 blockDim(256);
    dim3 gridDim((numTriangles + blockDim.x - 1) / blockDim.x);

    // Launch the kernel
    drawTrianglesKernel<<<gridDim, blockDim>>>(d_x, d_y, d_indices, numTriangles, d_color, d_bitmap, width, height);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    uint8_t *result_bitmap = (uint8_t *)malloc(bitmapsize);
    cudaMemcpy(result_bitmap, d_bitmap, bitmapsize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_bitmap);
    cudaFree(d_color);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_indices);

    // Return the resulting bitmap
    return result_bitmap;
}
