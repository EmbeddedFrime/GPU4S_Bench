// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <chrono>
#include <clFFT.h>


//#define BLOCK_SIZE 1024
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
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
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
    deviceObj->evt_copyB = new cl::Event;
    deviceObj->evt_copyBr = new cl::Event;
    

}

bool device_memory_init(GraficCommon* device_object,  int64_t size_a_array, int64_t size_b_array){
   GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
   deviceObj->d_A = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size_a_array);
   deviceObj->d_B = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,sizeof(bench_t)*size_b_array);
   // inicialice Arrays
   return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A,int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // copy memory host -> device
    //TODO Errors check
    deviceObj->queue->enqueueWriteBuffer(*deviceObj->d_A,CL_TRUE,0,sizeof(bench_t)*size, h_A, NULL, deviceObj->evt_copyB);
}
void aux_execute_kernel(GraficCommon* device_object, int64_t size, cl::Buffer *d_A, cl::Buffer *d_B, clfftPlanHandle *planHandle){
    
GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    
    /* Execute the plan. */
    clfftEnqueueTransform(*planHandle, CLFFT_FORWARD, 1, &(*deviceObj->queue)(), 0, NULL, &(*deviceObj->evt)(), &(*d_A)(), &(*d_B)(), NULL);

    //deviceObj->queue->finish();
    
    

}
void execute_kernel(GraficCommon* device_object, int64_t window, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    struct timespec start, end;
    cl::Buffer *d_A = deviceObj->d_A;
    cl::Buffer *d_B = deviceObj->d_B;
    cl::Program::Sources sources;
    deviceObj->evt = new cl::Event;
    // load kernel from file
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /* Setup clFFT. */
    clfftSetupData fftSetup;
    clfftInitSetupData(&fftSetup);
    clfftSetup(&fftSetup);
    clfftPlanHandle planHandle;
    size_t clLengths[1] = {(long unsigned int)window/2};
    clfftCreateDefaultPlan(&planHandle, (*deviceObj->context)(), CLFFT_1D, clLengths);

    /* Set plan parameters. */
    #ifdef FLOAT
    clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
    #else
    clfftSetPlanPrecision(planHandle, CLFFT_DOUBLE);
    #endif
    clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);

    /* Bake the plan. */
    clfftBakePlan(planHandle, 1, &(*deviceObj->queue)(), NULL, NULL);
    for (unsigned int i = 0; i < (size * 2  - window + 1); i+=1){
        aux_execute_kernel(device_object, window, d_A, d_B, &planHandle);
        d_B += window;
        ++d_A;
    }
    deviceObj->queue->finish();
    clfftDestroyPlan( &planHandle );
    clfftTeardown( );
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    deviceObj->elapsed_time =  (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_B, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->queue->enqueueReadBuffer(*deviceObj->d_B,CL_TRUE,0,sizeof(bench_t)*size,h_B, NULL, deviceObj->evt_copyBr);
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->evt_copyBr->wait();
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    elapsed_h_d = deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Host->Device: %.10f \n", elapsed / 1000000.0);
    elapsed = deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time kernel: %.10f \n", elapsed / 1000000.0);
    elapsed_d_h = deviceObj->evt_copyBr->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyBr->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Device->Host: %.10f \n", );


    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", lapsed_h_d / 1000000.0,deviceObj->elapsed_time ,elapsed_d_h / 1000000.0, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0,deviceObj->elapsed_time,elapsed_d_h / 1000000.0);
    }else{
         printf("Elapsed time Host->Device: %.10f milliseconds\n", (elapsed_h_d / 1000000.0));
         printf("Elapsed time kernel: %.10f milliseconds\n",  deviceObj->elapsed_time);
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
    delete deviceObj->evt_copyB;
    delete deviceObj->evt_copyBr;
}
