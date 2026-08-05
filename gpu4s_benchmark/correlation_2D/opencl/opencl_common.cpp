/** * ====================================================================
 * @file        opencl_common.cpp (./correlation_2D)
 * @brief       Common OpenCL platform initialization, device setup, 
 *              profiling timer evaluation, and generic cleanup routines.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#include "../benchmark_library.h"
#include <cmath>


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
   //get default device of the default platformB
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
    deviceObj->evt_mean = new cl::Event; 
    deviceObj->evt_copyA = new cl::Event;
    deviceObj->evt_copyB = new cl::Event;
    deviceObj->evt_copyAB = new cl::Event;
    deviceObj->evt_copyAA = new cl::Event;
    deviceObj->evt_copyBB = new cl::Event;
    
}

bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    cl_int err;

    deviceObj->d_A = new cl::Buffer(*deviceObj->context, CL_MEM_READ_ONLY, sizeof(bench_t) * size_a_matrix, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    deviceObj->d_B = new cl::Buffer(*deviceObj->context, CL_MEM_READ_ONLY, sizeof(bench_t) * size_b_matrix, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    deviceObj->d_R = new cl::Buffer(*deviceObj->context, CL_MEM_READ_WRITE, sizeof(result_bench_t), nullptr, &err);
    if (err != CL_SUCCESS) return false;

    deviceObj->mean_A = new cl::Buffer(*deviceObj->context, CL_MEM_READ_WRITE, sizeof(result_bench_t), nullptr, &err);
    if (err != CL_SUCCESS) return false;

    deviceObj->mean_B = new cl::Buffer(*deviceObj->context, CL_MEM_READ_WRITE, sizeof(result_bench_t), nullptr, &err);
    if (err != CL_SUCCESS) return false;

    deviceObj->acumulate_value_a_b = new cl::Buffer(*deviceObj->context, CL_MEM_READ_WRITE, sizeof(result_bench_t), nullptr, &err);
    if (err != CL_SUCCESS) return false;

    deviceObj->acumulate_value_a_a = new cl::Buffer(*deviceObj->context, CL_MEM_READ_WRITE, sizeof(result_bench_t), nullptr, &err);
    if (err != CL_SUCCESS) return false;

    deviceObj->acumulate_value_b_b = new cl::Buffer(*deviceObj->context, CL_MEM_READ_WRITE, sizeof(result_bench_t), nullptr, &err);
    if (err != CL_SUCCESS) return false;

    // inicialice Arrays
    return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b){
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// copy memory host -> device

    #ifdef PROFILING_CLOCK
        h2dCLK.start();
    #endif

    cl_int err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->d_A,CL_TRUE,0,sizeof(bench_t)*size_a, h_A, NULL, deviceObj->evt_copyA);
    if (err != CL_SUCCESS) 
    {
        fprintf(stderr, "Failed to copy vector A from host to device (OpenCL error code %d)!\n", err);
        return;
    }

    err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->d_B,CL_TRUE,0,sizeof(bench_t)*size_b, h_B, NULL, deviceObj->evt_copyB);
    if (err != CL_SUCCESS) 
    {
        fprintf(stderr, "Failed to copy vector B from host to device (OpenCL error code %d)!\n", err);
        return;
    }

    #ifdef PROFILING_CLOCK
        deviceObj->queue->finish();
        h2dCLK.end();
    #endif
}

void copy_memory_to_host(GraficCommon* device_object, result_bench_t* h_R){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    #ifdef PROFILING_CLOCK
        d2hCLK.start();
    #endif

    result_bench_t acumulate_value_a_a;
    result_bench_t acumulate_value_a_b;
    result_bench_t acumulate_value_b_b;
    deviceObj->queue->enqueueReadBuffer(*deviceObj->acumulate_value_a_a,CL_TRUE,0,sizeof(result_bench_t),&acumulate_value_a_a, NULL, deviceObj->evt_copyAA);
    deviceObj->queue->enqueueReadBuffer(*deviceObj->acumulate_value_a_b,CL_TRUE,0,sizeof(result_bench_t),&acumulate_value_a_b, NULL, deviceObj->evt_copyAB);
    deviceObj->queue->enqueueReadBuffer(*deviceObj->acumulate_value_b_b,CL_TRUE,0,sizeof(result_bench_t),&acumulate_value_b_b, NULL, deviceObj->evt_copyBB);
    deviceObj->evt_copyBB->wait();
    *h_R = (result_bench_t)(acumulate_value_a_b / (result_bench_t)(sqrt(acumulate_value_a_a * acumulate_value_b_b)));

    #ifdef PROFILING_CLOCK
        deviceObj->queue->finish();
        d2hCLK.end();
    #endif
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format,bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->evt_copyBB->wait();
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    elapsed_h_d = deviceObj->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    
    elapsed = deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evt_mean->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_mean->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    
    elapsed_d_h = deviceObj->evt_copyAA->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyAA->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_d_h += deviceObj->evt_copyAB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyAB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_d_h += deviceObj->evt_copyBB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyBB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    

    #ifdef PROFILING_CLOCK
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on PROFILING_CLOCK) ---
        elapsed_h_d  = h2dCLK.getElapsedNS();
        elapsed      = kernelCLK.getElapsedNS();
        elapsed_d_h  = d2hCLK.getElapsedNS();
        const char* profilingMode = "CLOCK";
    #else
        const char* profilingMode = "GPU";
    #endif

    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", elapsed_h_d / 1000000.0,elapsed / 1000000.0,elapsed_d_h / 1000000.0, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0,elapsed / 1000000.0,elapsed_d_h / 1000000.0);
    }else{
         printf("profiling mode: %s\n", profilingMode);
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
    delete deviceObj->d_R;
    delete deviceObj->acumulate_value_a_a;
    delete deviceObj->acumulate_value_a_b;
    delete deviceObj->acumulate_value_b_b;
    delete deviceObj->evt;
    delete deviceObj->evt_copyA;
    delete deviceObj->evt_copyB;
    delete deviceObj->evt_copyBB;
    delete deviceObj->evt_copyAB;
    delete deviceObj->evt_copyAA;
}
