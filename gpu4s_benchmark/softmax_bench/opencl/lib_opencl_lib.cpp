// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <cstring>
#include "GEN_kernel_opt.hcl"
#include "GEN_atomic_functions.hcl"

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    const unsigned int x_local= BLOCK_SIZE;
    const unsigned int y_local= BLOCK_SIZE;
    cl::NDRange local, global;
    if(n < BLOCK_SIZE)
    {
        local = cl::NDRange (1, 1);
        global = cl::NDRange (n, w);
    }
    else
    {
        local = cl::NDRange(x_local, y_local);
        global = cl::NDRange(n, w);
    }

    cl::Program::Sources sources;
    // FIX: Removed duplicate "new cl::Event" memory leak

    // --- FIX: load the full kernel from file + #define BLOCK_SIZE ---
    std::string preamble = "#define BLOCK_SIZE " + std::to_string(BLOCK_SIZE) + "\n";
    std::string full_kernel = preamble + type_kernel_common + atomic_code + kernel_code;
    sources.push_back(full_kernel);

    cl::Program program(*deviceObj->context,sources);
    if(program.build({deviceObj->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(deviceObj->default_device)<<"\n";
        exit(1);
    }

     #ifdef PROFILING_CLOCK
        deviceObj->queue->finish();
        kernelCLK.start();
    #endif

    cl::Kernel softmax_kernel=cl::Kernel(program,"kernel_softmax");
    softmax_kernel.setArg(0,*deviceObj->d_A);
    softmax_kernel.setArg(1,*deviceObj->d_B);
    softmax_kernel.setArg(2,*deviceObj->sum_d_B);
    softmax_kernel.setArg(3,n);

    deviceObj->queue->enqueueNDRangeKernel(softmax_kernel,cl::NullRange,global,local, NULL, deviceObj->evt);


    cl::Kernel softmax_end_kernel=cl::Kernel(program,"kernel_softmax_end");
    softmax_end_kernel.setArg(0,*deviceObj->d_B);
    softmax_end_kernel.setArg(1,*deviceObj->sum_d_B);
    softmax_end_kernel.setArg(2,n);

    deviceObj->queue->enqueueNDRangeKernel(softmax_end_kernel,cl::NullRange,global,local, NULL, deviceObj->evt_complemet);
    deviceObj->queue->finish();

    #ifdef PROFILING_CLOCK
        kernelCLK.end();
    #endif
}

