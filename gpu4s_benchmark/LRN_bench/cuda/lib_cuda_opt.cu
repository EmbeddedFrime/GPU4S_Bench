#include "../benchmark_library.h"
#include "math.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

__global__ void
lrn_kernel(const bench_t *A, bench_t *B, const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i  < (size * size)){
        
        #ifdef INT
        B[i] = A[i]/powf((K+ALPHA*powf(A[i],2)),BETA);
        #elif FLOAT
        B[i] = A[i]/powf((K+ALPHA*powf(A[i],2)),BETA);
        #else
        B[i] = A[i]/powf((K+ALPHA*powf(A[i],2)),BETA);
        
        #endif
    }
}

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(ceil(float((n*n))/(dimBlock.x)));

    #ifdef PROFILING_CLOCK
        kernelCLK.start();
    #endif

    cudaEventRecord(*deviceObj->start);
    lrn_kernel<<<dimGrid, dimBlock>>>(deviceObj->d_A, deviceObj->d_B, n);
    cudaEventRecord(*deviceObj->stop);

    #ifdef PROFILING_CLOCK
        cudaDeviceSynchronize(); 
        kernelCLK.end();
    #endif
}

