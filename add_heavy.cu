#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 100000000   // 100 million elements
#define THREADS 256

__global__ void addKernel(float *a, float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float x = a[idx];
        float y = b[idx];

        // extremely heavy ADD workload
        #pragma unroll 1
        for (int i = 0; i < 100000; i++)
        {
            x = x + y;
            y = y + 0.000001f;

            x = x + y;
            y = y + 0.000001f;

            x = x + y;
            y = y + 0.000001f;

            x = x + y;
            y = y + 0.000001f;
        }

        c[idx] = x;
    }
}

int main()
{
    int size = N * sizeof(float);

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS - 1) / THREADS;

    std::cout << "Running HEAVY ADD workload...\n";

    auto start = std::chrono::high_resolution_clock::now();

    // run for 20 minutes
    while (true)
    {
        addKernel<<<blocks, THREADS>>>(d_a, d_b, d_c, N);

        auto now = std::chrono::high_resolution_clock::now();

        double elapsed =
            std::chrono::duration<double>(now - start).count();

        if (elapsed >= 1200) // 20 min
            break;
    }

    cudaDeviceSynchronize();

    std::cout << "Finished workload.\n";

    cudaMemcpy(h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sample output: " << h_c[0] << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}