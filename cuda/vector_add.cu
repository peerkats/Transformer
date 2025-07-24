#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
// vector_add.cu


__global__ void vectorAddKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

__global__ void tensorDotkernel(float* Dims, float* data, float* output, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int rows = (int)Dims[0];
    int cols = (int)Dims[1];

    if (row >= rows) return;

    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        int idx = row * cols + col;
        sum += data[idx];
    }
    output[row] = sum;
}

extern "C" void launch_tensor_dot(float* A, float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}


extern "C" void launch_vector_add(float* A, float* B, float* C, int N) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}