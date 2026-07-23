// // #include <stdio.h>
// // #include <cuda_runtime.h>

// // __global__ void mul_stress(float *A, float *B, float *C, int N)
// // {
// //     int idx = blockIdx.x * blockDim.x + threadIdx.x;

// //     if (idx < N)
// //     {
// //         float a = A[idx];
// //         float b = B[idx];

// //         #pragma unroll 32
// //         for (int i = 0; i < 5000000; i++)
// //         {
// //             a = a * 1.000001f;
// //             b = b * 0.999999f;

// //             a = a * b;
// //             b = b * a;

// //             a = a * 1.0000001f;
// //             b = b * 0.9999999f;
// //         }

// //         C[idx] = a + b;
// //     }
// // }

// // int main()
// // {
// //     const int N = 1 << 27;

// //     size_t size = N * sizeof(float);

// //     float *h_A = new float[N];
// //     float *h_B = new float[N];

// //     for (int i = 0; i < N; i++)
// //     {
// //         h_A[i] = 1.0001f;
// //         h_B[i] = 1.0002f;
// //     }

// //     float *d_A, *d_B, *d_C;

// //     cudaMalloc(&d_A, size);
// //     cudaMalloc(&d_B, size);
// //     cudaMalloc(&d_C, size);

// //     cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
// //     cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

// //     int threads = 1024;
// //     int blocks = (N + threads - 1) / threads;

// //     printf("Running MUL stress...\n");

// //     while (true)
// //     {
// //         mul_stress<<<blocks, threads>>>(d_A, d_B, d_C, N);
// //         cudaDeviceSynchronize();
// //     }

// //     return 0;
// // }

// #include <stdio.h>
// #include <cuda_runtime.h>

// __global__ void matmul(float *A, float *B, float *C, int N)
// {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < N && col < N)
//     {
//         float sum = 0.0f;

//         for (int k = 0; k < N; k++)
//         {
//             sum += A[row * N + k] * B[k * N + col];
//         }

//         C[row * N + col] = sum;
//     }
// }

// int main()
// {
//     const int N = 2048;

//     size_t bytes = (size_t)N * N * sizeof(float);

//     printf("Matrix size: %d x %d\n", N, N);

//     float *h_A = new float[N * N];
//     float *h_B = new float[N * N];

//     for (long long i = 0; i < (long long)N * N; i++)
//     {
//         h_A[i] = 1.0f;
//         h_B[i] = 1.0f;
//     }

//     float *d_A, *d_B, *d_C;

//     cudaMalloc(&d_A, bytes);
//     cudaMalloc(&d_B, bytes);
//     cudaMalloc(&d_C, bytes);

//     cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

//     dim3 threads(16, 16);
//     dim3 blocks(
//         (N + threads.x - 1) / threads.x,
//         (N + threads.y - 1) / threads.y
//     );

//     printf("Starting GPU matrix multiplication...\n");

//     while (true)
//     {
//         matmul<<<blocks, threads>>>(d_A, d_B, d_C, N);

//         cudaError_t err = cudaGetLastError();

//         if (err != cudaSuccess)
//         {
//             printf("Kernel Error: %s\n",
//                    cudaGetErrorString(err));
//             break;
//         }

//         cudaDeviceSynchronize();
//     }

//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     delete[] h_A;
//     delete[] h_B;

//     return 0;
// }
#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int count = 0;

    cudaError_t err = cudaGetDeviceCount(&count);

    printf("err = %d\n", (int)err);
    printf("count = %d\n", count);

    const char* msg = cudaGetErrorString(err);

    if(msg)
        printf("msg = %s\n", msg);
    else
        printf("msg = NULL\n");

    return 0;
}