#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                              \
do {                                                                   \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        printf("CUDA Error: %s\n", cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

#define CHECK_CUBLAS(call)                                             \
do {                                                                   \
    cublasStatus_t stat = (call);                                      \
    if (stat != CUBLAS_STATUS_SUCCESS) {                               \
        printf("cuBLAS Error: %d\n", stat);                            \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

int main()
{


     int count = 0;

    cudaError_t err = cudaGetDeviceCount(&count);

    printf("cudaGetDeviceCount returned: %d\n", err);
    printf("Device count: %d\n", count);

    if(err != cudaSuccess)
    {
        printf("Error string: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);

    printf("cudaGetDeviceProperties returned: %d\n", err);

    if(err != cudaSuccess)
    {
        printf("Error string: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("GPU: %s\n", prop.name);
    // cudaDeviceProp prop;
    // CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    // printf("GPU: %s\n", prop.name);

    const int N = 8192;      // Use 4096 if VRAM is limited
    const int ITER = 100;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    size_t bytes = (size_t)N * N * sizeof(float);

    printf("Matrix size: %d x %d\n", N, N);
    printf("Allocating %.2f GB per matrix\n",
           bytes / (1024.0 * 1024.0 * 1024.0));

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);

    if (!h_A || !h_B)
    {
        printf("Host allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < (size_t)N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;

    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    printf("Warmup...\n");

    for (int i = 0; i < 10; i++)
    {
        CHECK_CUBLAS(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                N,
                N,
                &alpha,
                d_A,
                N,
                d_B,
                N,
                &beta,
                d_C,
                N
            )
        );
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("Starting benchmark...\n");

    CHECK_CUDA(cudaEventRecord(start, 0));

    for (int i = 0; i < ITER; i++)
    {
        CHECK_CUBLAS(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                N,
                N,
                &alpha,
                d_A,
                N,
                d_B,
                N,
                &beta,
                d_C,
                N
            )
        );
    }

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("Raw time: %.3f ms\n", ms);

    double seconds = ms / 1000.0;

    double flops =
        2.0 * (double)N * (double)N * (double)N * ITER;

    double tflops = flops / seconds / 1e12;

    printf("Elapsed time: %.3f sec\n", seconds);
    printf("Performance: %.2f TFLOPS\n", tflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    free(h_A);
    free(h_B);

    return 0;
}