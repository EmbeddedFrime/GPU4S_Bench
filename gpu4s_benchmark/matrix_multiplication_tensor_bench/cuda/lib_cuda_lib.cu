#include <cublas_v2.h>
#include "../benchmark_library.h"

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // cublas settings
    int lda=m,ldb=m,ldc=m;
    const bench_t alf = 1;
    const bench_t bet = 0;
    const bench_t *alpha = &alf;
    const bench_t *beta = &bet;
    cublasHandle_t handle;
    cublasCreate(&handle);

    #ifdef PROFILING_CLOCK
        kernelCLK.start();
    #endif

    cudaEventRecord(*deviceObj->start);
    //cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    #ifdef INT
    printf("CUBLAS NOT SUPPORT INT OPERATIOS\n");
    #elif FLOAT
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, deviceObj->d_B, lda, deviceObj->d_A, ldb, beta, deviceObj->d_C, ldc);
    #else 
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, deviceObj->d_B, lda, deviceObj->d_A, ldb, beta, deviceObj->d_C, ldc);
    #endif
    
    cudaEventRecord(*deviceObj->stop);

    #ifdef PROFILING_CLOCK
        cudaDeviceSynchronize(); 
        kernelCLK.end();
    #endif

    // destroy cublas
    cublasDestroy(handle);
}
