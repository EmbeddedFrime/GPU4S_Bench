// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <cstring>
#include "kernel_opt.cl"

#ifdef ANDROID
    // kernel time execution
    Clock kernelCLK;
    // host <-> device 
    Clock h2dCLK;
    Clock d2hCLK;
#endif

//#define BLOCK_SIZE 256
void init(GraficCommon* device_object, char* device_name){
    init(device_object, 0,0, device_name);
}

void init(GraficCommon* device_object, int platform ,int device, char* device_name){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[platform];
    //std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
   //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[device];
    //std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
    strcpy(device_name,default_device.getInfo<CL_DEVICE_NAME>().c_str() );
    
    // context
    deviceObj->context = new cl::Context(default_device);
    deviceObj->queue = new cl::CommandQueue(*deviceObj->context,default_device,CL_QUEUE_PROFILING_ENABLE);
    deviceObj->default_device = default_device;
    
    // events
    deviceObj->evt = new cl::Event; 
    deviceObj->evt_copyA = new cl::Event;
    deviceObj->evt_copyB = new cl::Event;
}

bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    cl_int err;
    
    deviceObj->d_A = new cl::Buffer(*deviceObj->context, CL_MEM_READ_ONLY, sizeof(bench_t)*size_a_matrix, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    // --- FIX: Switched to CL_MEM_READ_WRITE because enqueueNDRangeKernel writes ---
    deviceObj->d_B = new cl::Buffer(*deviceObj->context, CL_MEM_WRITE_ONLY, sizeof(bench_t)*size_b_matrix, nullptr, &err);
    if (err != CL_SUCCESS) return false;
   
    return true;
}


void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, unsigned int size_a){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    #ifdef ANDROID
        h2dCLK.start();
    #endif
	
    // Enqueue writing host memory h_A to device buffer d_A
    cl_int err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->d_A, CL_TRUE, 0, sizeof(bench_t)*size_a, h_A, NULL, deviceObj->evt_copyA);

    if (err != CL_SUCCESS) 
    {
        fprintf(stderr, "Failed to copy vector A from host to device (OpenCL error code %d)!\n", err);
        return;
    }

    #ifdef ANDROID
        deviceObj->queue->finish();
        h2dCLK.end();
    #endif
}


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

     #ifdef ANDROID
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

    #ifdef ANDROID
        kernelCLK.end();
    #endif
}

void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    #ifdef ANDROID
        d2hCLK.start();
    #endif

    deviceObj->queue->enqueueReadBuffer(*deviceObj->d_B,CL_TRUE,0,sizeof(bench_t)*size,h_C, NULL, deviceObj->evt_copyB);

    #ifdef ANDROID
        deviceObj->queue->finish();
        d2hCLK.end();
    #endif
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->evt_copyB->wait();
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    
    elapsed_h_d = deviceObj->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Host->Device: %.10f \n", elapsed / 1000000.0);

    elapsed = deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time kernel: %.10f \n", elapsed / 1000000.0);

    elapsed_d_h = deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Device->Host: %.10f \n", );

    #ifdef ANDROID
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on Android) ---
        elapsed_h_d  = h2dCLK.getElapsedNS();
        elapsed      = kernelCLK.getElapsedNS();
        elapsed_d_h  = d2hCLK.getElapsedNS();
    #endif

    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", elapsed_h_d / 1000000.0,elapsed / 1000000.0,elapsed_d_h / 1000000.0,current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0,elapsed / 1000000.0,elapsed_d_h / 1000000.0);
    }else{
         printf("Elapsed time Host->Device: %.10f milliseconds\n", (elapsed_h_d / 1000000.0));
         printf("Elapsed time kernel: %.10f milliseconds\n", elapsed / 1000000.0);
         printf("Elapsed time Device->Host: %.10f milliseconds\n", elapsed_d_h / 1000000.0);
    }
    return elapsed / 1000000.0; // TODO Change
}

void clean(GraficCommon* device_object){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // pointers clean
    delete deviceObj->context;
    delete deviceObj->queue;
    // pointer to memory
    delete deviceObj->d_A;
    delete deviceObj->d_B;
    delete deviceObj->evt;
    delete deviceObj->evt_copyA;
    delete deviceObj->evt_copyB;
}