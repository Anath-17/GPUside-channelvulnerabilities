#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int count = 0;

    cudaError_t err = cudaGetDeviceCount(&count);

    printf("err code   = %d\n", (int)err);
    printf("err name   = %s\n", cudaGetErrorName(err));
    printf("err string = %s\n", cudaGetErrorString(err));
    printf("count      = %d\n", count);

    return 0;
}