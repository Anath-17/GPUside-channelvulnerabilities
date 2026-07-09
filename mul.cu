#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000000
#define ITERATIONS 20000
#define EXPERIMENT_TIME_SECONDS 6000

__global__ void mulKernel(
    float* A,
    float* B,
    float* C)
{
    int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (idx < N)
    {
        float x = A[idx];
        float y = B[idx];

        #pragma unroll 1
        for (int i = 0;
             i < ITERATIONS;
             i++)
        {
            x = x * y;

            // stop compiler optimization
            asm volatile("");
        }

        C[idx] = x;
    }
}

int main()
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t size =
        N * sizeof(float);

    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0001f;
        h_B[i] = 1.00001f;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(
        d_A,
        h_A,
        size,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_B,
        h_B,
        size,
        cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    std::cout
        << "Running MUL experiment for "
        << EXPERIMENT_TIME_SECONDS
        << " seconds..."
        << std::endl;

    auto start =
        std::chrono::
        high_resolution_clock::now();

    while (true)
    {
        mulKernel<<<
            blocksPerGrid,
            threadsPerBlock>>>(
            d_A,
            d_B,
            d_C);

        auto now =
            std::chrono::
            high_resolution_clock::now();

        double elapsed =
            std::chrono::duration<double>(
                now - start).count();

        if (elapsed >=
            EXPERIMENT_TIME_SECONDS)
        {
            break;
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_C,
        d_C,
        size,
        cudaMemcpyDeviceToHost);

    std::cout
        << "Finished MUL"
        << std::endl;

    std::cout
        << "Sample Output: "
        << h_C[0]
        << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}