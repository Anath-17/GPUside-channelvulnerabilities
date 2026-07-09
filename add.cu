#include <iostream>
#include <cuda_runtime.h>

#define N 1000000
#define BLOCK_SIZE 256
#define REPEAT 100000000

__global__ void addKernel(float *a, float *b, float *c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        float x = a[idx];
        float y = b[idx];

        #pragma unroll 1
        for (long long i = 0; i < REPEAT; i++)
        {
            x = x + y;
            y = y + 0.000001f;

            x = x * 1.0000001f;
            y = y * 0.9999999f;
        }

        c[idx] = x;
    }
}

int main()
{
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    size_t size = N * sizeof(float);

    a = new float[N];
    b = new float[N];
    c = new float[N];

    for (int i = 0; i < N; i++)
    {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::cout << "Running heavy ADD workload..." << std::endl;

    while (true)
    {
        addKernel<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c);
        cudaDeviceSynchronize();
    }

    return 0;
}