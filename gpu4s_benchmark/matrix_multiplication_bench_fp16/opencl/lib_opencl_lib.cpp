// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <clblast.h>


void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    const bench_t alpha = 1.0f;
    const bench_t beta = 1.0f;
    const unsigned int a_ld = n;
    const unsigned int b_ld = n;
    const unsigned int c_ld = n;
    #ifdef INT
    printf("CLBLAST NOT SUPPORT INT OPERATIOS\n");
    #else
        #ifdef PROFILING_CLOCK
            // Ensure queue is idle before measuring
            deviceObj->queue->finish();
            kernelCLK.start();
        #endif

        auto status = clblast::Gemm(clblast::Layout::kRowMajor,clblast::Transpose::kNo, clblast::Transpose::kNo, n, n, n, alpha, (*deviceObj->d_A)() , 0, a_ld, (*deviceObj->d_B)(), 0, b_ld, beta, (*deviceObj->d_C)(), 0, c_ld,&(*deviceObj->queue)(), &(*deviceObj->evt)());
        
        // Wait for completion before stopping the clock
        deviceObj->queue->finish();
        #ifdef PROFILING_CLOCK
            kernelCLK.end();
        #endif
    #endif
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    #ifdef PROFILING_CLOCK
        d2hCLK.start();
    #endif

    deviceObj->queue->enqueueReadBuffer(*deviceObj->d_C,CL_TRUE,0,sizeof(bench_t)*size,h_C, NULL, deviceObj->evt_copyC);

    
    #ifdef PROFILING_CLOCK
        deviceObj->queue->finish();
        d2hCLK.end();
    #endif
}

