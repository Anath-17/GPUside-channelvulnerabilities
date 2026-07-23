#include <stdio.h>
#include <cuda_runtime.h>

__global__ void zero_kernel(float *A, float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N)
    {
        float a = A[idx];
        float b = B[idx];

        #pragma unroll 32
        for(int i=0;i<100000;i++)
        {
            a = a + b;
            b = b + a;
        }

        C[idx] = a + b;
    }
}

int main()
{
    const int N = 1<<26;

    size_t size = N*sizeof(float);

    float *h_A = new float[N];
    float *h_B = new float[N];

    for(int i=0;i<N;i++)
    {
        h_A[i] = 0.0f;
        h_B[i] = 0.0f;
    }

    float *d_A,*d_B,*d_C;

    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    int threads=1024;
    int blocks=(N+threads-1)/threads;

    while(true)
    {
        zero_kernel<<<blocks,threads>>>(d_A,d_B,d_C,N);
    }
}