#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

void execute_kernel(GraficCommon* device_object, int64_t size){

GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    cufftHandle plan;

    #ifdef PROFILING_CLOCK
        kernelCLK.start();
    #endif

    cudaEventRecord(*deviceObj->start);
    
    #ifdef FLOAT
    cufftPlan1d(&plan, size, CUFFT_C2C, 1);
    cufftExecC2C(plan, (cufftComplex *)deviceObj->d_B, (cufftComplex *)deviceObj->d_Br, CUFFT_FORWARD);
    #else 
    cufftPlan1d(&plan, size, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, (cufftDoubleComplex *)deviceObj->d_B, (cufftDoubleComplex *)deviceObj->d_Br, CUFFT_FORWARD);
    #endif
    
    cudaEventRecord(*deviceObj->stop);

    #ifdef PROFILING_CLOCK
        cudaDeviceSynchronize(); 
        kernelCLK.end();
    #endif

    cufftDestroy(plan);
    
}

