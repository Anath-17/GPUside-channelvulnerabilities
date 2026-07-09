#include <iostream>
#include <cuda_runtime.h>

__global__ void stressKernel()
{
    float x = threadIdx.x + blockIdx.x;

    while (true)
    {
        // Heavy floating point workload
        for (int i = 0; i < 1000000; i++)
        {
            x = x * 1.000001f + 0.000001f;
            x = x * 1.000001f + 0.000001f;
            x = x * 1.000001f + 0.000001f;
            x = x * 1.000001f + 0.000001f;
        }
    }
}

int main()
{
    std::cout << "Starting GPU stress...\n";
    std::cout << "Press Ctrl+C to stop.\n";

    // Huge launch to occupy GPU
    stressKernel<<<65535, 1024>>>();

    cudaDeviceSynchronize();

    return 0;
}