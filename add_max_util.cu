#include <iostream>
#include <cuda_runtime.h>

#define THREADS 1024
#define BLOCKS 32768

__global__ void heavyAddKernel(float *out)
{
    unsigned long long idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    float x = idx * 0.0001f;
    float y = 2.0f;

    // Infinite heavy ADD loop
    while (true)
    {
        #pragma unroll 1
        for (int i = 0; i < 1000000; i++)
        {
            x = x + y;
            y = y + 0.000001f;

            x = x + y;
            y = y + 0.000001f;

            x = x + y;
            y = y + 0.000001f;

            x = x + y;
            y = y + 0.000001f;

            x = x + y;
            y = y + 0.000001f;

            x = x + y;
            y = y + 0.000001f;

            x = x + y;
            y = y + 0.000001f;

            x = x + y;
            y = y + 0.000001f;
        }

        out[idx % 1024] = x;
    }
}

int main()
{
    float *d_out;

    cudaMalloc(&d_out, 1024 * sizeof(float));

    std::cout << "Starting MAX GPU ADD workload...\n";
    std::cout << "Press Ctrl+C to stop.\n";

    heavyAddKernel<<<BLOCKS, THREADS>>>(d_out);

    cudaDeviceSynchronize();

    cudaFree(d_out);

    return 0;
}