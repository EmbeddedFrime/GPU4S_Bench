// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <cstring>
#include "kernel.cl"


void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    const unsigned int x_local= BLOCK_SIZE;
    const unsigned int y_local= BLOCK_SIZE;
    cl::NDRange local(x_local, y_local);
    cl::NDRange global(n, w);

    cl::Program::Sources sources;
    deviceObj->evt = new cl::Event;
    // load kernel from file
    kernel_code = type_kernel + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    cl::Program program(*deviceObj->context,sources);
    if(program.build({deviceObj->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(deviceObj->default_device)<<"\n";
        exit(1);
    }
    cl::Kernel kernel_add=cl::Kernel(program,"kernel_matrix_multiplication");
    #ifdef FLOAT16
        kernel_add.setArg(0,*deviceObj->d_half_A);
        kernel_add.setArg(1,*deviceObj->d_half_B);
        kernel_add.setArg(2,*deviceObj->d_half_C);
    #else
        kernel_add.setArg(0,*deviceObj->d_A);
        kernel_add.setArg(1,*deviceObj->d_B);
        kernel_add.setArg(2,*deviceObj->d_C);
    #endif
    kernel_add.setArg(3,n);
    kernel_add.setArg(4,m);
    kernel_add.setArg(5,w);

    #ifdef PROFILING_CLOCK
        deviceObj->queue->finish();
        kernelCLK.start();
    #endif

    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt);

    deviceObj->queue->finish();
    #ifdef PROFILING_CLOCK
        kernelCLK.end();
    #endif
}
