#include <stdio.h>
#include <cuda.h>

// Kernel that executes on the CUDA device
 __global__ void square_array(float *a, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    a[idx] = a[idx] * a[idx];
}

void print_device_properties(void)
{
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %zu\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %zu\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %zu\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %zu\n", devProp.totalConstMem);
    printf("Texture alignment:             %zu\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
}


// main routine that executes on the host
int main(void)
{
    print_device_properties();

    const int N = 10;
    float a_h[N];

    float *a_d;
    cudaMalloc((void **) &a_d, sizeof(a_h));

    for (int i=0; i < N; i++)
        a_h[i] = i;

    cudaMemcpy(a_d, a_h, sizeof(a_h), cudaMemcpyHostToDevice);

    int block_size = 4;
    int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
    square_array <<< n_blocks, block_size >>> (a_d, N);

    cudaMemcpy(a_h, a_d, sizeof(a_h), cudaMemcpyDeviceToHost);

    for (int i=0; i<N; i++) {
        printf("%d %f\n", i, a_h[i]);
        if (abs(a_h[i] - i*i) > 0.001)
            return -1;
    }

    cudaFree(a_d);
}
