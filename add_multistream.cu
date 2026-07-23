#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add_stress(float *A, float *B, float *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float a = A[idx];
    float b = B[idx];

    #pragma unroll 32
    for(long long i = 0; i < 5000000LL; i++)
    {
        a = a + b;
        b = b + a;

        a = a + 1.0f;
        b = b + 2.0f;
    }

    C[idx] = a + b;
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);

    int SMs = prop.multiProcessorCount;

    printf("GPU: %s\n",prop.name);
    printf("SMs: %d\n",SMs);

    const int N = 1<<20;

    size_t size = N*sizeof(float);

    float *h_A = new float[N];
    float *h_B = new float[N];

    for(int i=0;i<N;i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A,*d_B,*d_C;

    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    int threads = 256;

    int blocks = SMs * 64;

    cudaStream_t streams[8];

    for(int i=0;i<8;i++)
        cudaStreamCreate(&streams[i]);

    printf("Running ADD workload...\n");

    while(true)
    {
        for(int i=0;i<8;i++)
        {
            add_stress<<<blocks,
                        threads,
                        0,
                        streams[i]>>>
                        (d_A,d_B,d_C);
        }
    }

    return 0;
}