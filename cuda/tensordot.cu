#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void launch_cublas_dot(float* A, float* B, float* C, int M, int K, int N) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS assumes column-major ordering by default.
    // We simulate row-major by swapping A and B and transposing the operation.
    // That means we compute: Cᵗ = Bᵗ × Aᵗ
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose on B and A
        N, M, K,                   // Dimensions of the output matrix Cᵗ
        &alpha,
        B, N,                      // B: K×N
        A, K,                      // A: M×K
        &beta,
        C, N                       // C: M×N
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS SGEMM failed with code %d\n", status);
    }

    cublasDestroy(handle);
}