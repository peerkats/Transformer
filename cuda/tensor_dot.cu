#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Define the size of the tile to be processed by each thread block
#define TILE_WIDTH 32

__global__ void tensorDotProduct2D(float* A, float* B, float* C, int M, int K, int N) {
    // Shared memory tiles for A and B
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    // Thread's row and column within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column for the start of the output tile
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Accumulator for the output element
    float sum = 0.0f;

    // Loop over the tiles of A and B required to compute the C tile
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Collaboratively load a tile of A and B into shared memory
        // Each thread loads one element of each tile

        // Load tile_A
        int a_row = blockIdx.y * TILE_WIDTH + ty;
        int a_col = t * TILE_WIDTH + tx;
        if (a_row < M && a_col < K) {
            tile_A[ty][tx] = A[a_row * K + a_col];
        } else {
            tile_A[ty][tx] = 0.0f;
        }

        // Load tile_B
        int b_row = t * TILE_WIDTH + ty;
        int b_col = blockIdx.x * TILE_WIDTH + tx;
        if (b_row < K && b_col < N) {
            tile_B[ty][tx] = B[b_row * N + b_col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }

        // Synchronize to ensure all data is loaded into shared memory
        __syncthreads();

        // Multiply the tiles from shared memory and accumulate the result
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += tile_A[ty][i] * tile_B[i][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the final result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}



extern "C" void launch_tensor_dot(float* A, float* B, float* C, int M, int K, int N) {
    // Use TILE_WIDTH for threads per block
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    // Calculate grid dimensions based on output size and tile width
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    tensorDotProduct2D<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}