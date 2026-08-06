// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <cstring>
#include "kernel_opt.cl"

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w, unsigned int stride, unsigned int lateral_stride){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    const unsigned int x_local= BLOCK_SIZE;
    cl::NDRange local, global;
    
    if(lateral_stride < BLOCK_SIZE)
    {
        local = cl::NDRange (1);
        global = cl::NDRange (lateral_stride * lateral_stride);
    }
    else
    {
        local = cl::NDRange(x_local);
        global = cl::NDRange(lateral_stride * lateral_stride);
    }
   
    cl::Program::Sources sources;
    deviceObj->evt = new cl::Event;
    
    // load kernel from file
    kernel_code = type_kernel_common + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    cl::Program program(*deviceObj->context,sources);
    if(program.build({deviceObj->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(deviceObj->default_device)<<"\n";
        exit(1);
    }

     #ifdef PROFILING_CLOCK
        deviceObj->queue->finish();
        kernelCLK.start();
    #endif
    
    cl::Kernel kernel_add=cl::Kernel(program,"kernel_max");
    kernel_add.setArg(0,*deviceObj->d_A);
    kernel_add.setArg(1,*deviceObj->d_B);
    kernel_add.setArg(2,n);
    kernel_add.setArg(3,stride);
    kernel_add.setArg(4,lateral_stride);


    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt);
    deviceObj->queue->finish();

    #ifdef PROFILING_CLOCK
        kernelCLK.end();
    #endif
}