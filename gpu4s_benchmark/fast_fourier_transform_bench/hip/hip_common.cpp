/** * ====================================================================
 * @file        hip_common.cpp (./fast_fourier_transform_bench)
 * @brief       Common HIP platform initialization, device setup, 
 *              profiling timer evaluation, and generic cleanup routines.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#include "../benchmark_library.h"
#include "hip/hip_runtime.h"


void init(GraficCommon* device_object, char* device_name){
    init(device_object, 0,0, device_name);
}

void init(GraficCommon* device_object, int platform ,int device, char* device_name){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    (void)hipSetDevice(device);
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, device);
    //printf("Using device: %s\n", prop.name);
    strcpy(device_name,prop.name);
    //event create 
    deviceObj->start = new hipEvent_t;
    deviceObj->stop = new hipEvent_t;
    deviceObj->start_memory_copy_device = new hipEvent_t;
    deviceObj->stop_memory_copy_device = new hipEvent_t;
    deviceObj->start_memory_copy_host = new hipEvent_t;
    deviceObj->stop_memory_copy_host= new hipEvent_t;
    
    (void)hipEventCreate(deviceObj->start);
    (void)hipEventCreate(deviceObj->stop);
    (void)hipEventCreate(deviceObj->start_memory_copy_device);
    (void)hipEventCreate(deviceObj->stop_memory_copy_device);
    (void)hipEventCreate(deviceObj->start_memory_copy_host);
    (void)hipEventCreate(deviceObj->stop_memory_copy_host);
}

bool device_memory_init(GraficCommon* device_object,  int64_t size_b_matrix){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    hipError_t err = hipSuccess;
    // Allocate the device input vector B
    err = hipMalloc((void **)&deviceObj->d_B, size_b_matrix * sizeof(bench_t));

    if (err != hipSuccess)
    {
        return false;
    }
    // Allocate the device reverse vector Br
    err = hipMalloc((void **)&deviceObj->d_Br, size_b_matrix * sizeof(bench_t));

    if (err != hipSuccess)
    {
        return false;
    }

    return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_B,int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    hipError_t err = hipSuccess;

    #ifdef PROFILING_CLOCK
        h2dCLK.start();
    #endif

    (void)hipEventRecord(*deviceObj->start_memory_copy_device);
    err = hipMemcpy(deviceObj->d_B, h_B, sizeof(bench_t) * size, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    (void)hipEventRecord(*deviceObj->stop_memory_copy_device);

    #ifdef PROFILING_CLOCK
        h2dCLK.end();
    #endif
    
}

void copy_memory_to_host(GraficCommon* device_object, bench_t* h_B, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    #ifdef PROFILING_CLOCK
        d2hCLK.start();
    #endif

    (void)hipEventRecord(*deviceObj->start_memory_copy_host);
    hipMemcpy(h_B, deviceObj->d_Br, size * sizeof(bench_t), hipMemcpyDeviceToHost);
    (void)hipEventRecord(*deviceObj->stop_memory_copy_host);

    #ifdef PROFILING_CLOCK
        d2hCLK.end();
    #endif
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    (void)hipEventSynchronize(*deviceObj->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    (void)hipEventElapsedTime(&milliseconds_h_d, *deviceObj->start_memory_copy_device, *deviceObj->stop_memory_copy_device);
    // kernel time
    (void)hipEventElapsedTime(&milliseconds, *deviceObj->start, *deviceObj->stop);
    //  memory transfer time device-host
    (void)hipEventElapsedTime(&milliseconds_d_h, *deviceObj->start_memory_copy_host, *deviceObj->stop_memory_copy_host);

    #ifdef PROFILING_CLOCK
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on PROFILING_CLOCK) ---
        milliseconds_h_d  = h2dCLK.getElapsedMS();
        milliseconds      = kernelCLK.getElapsedMS();
        milliseconds_d_h  = d2hCLK.getElapsedMS();
        const char* profilingMode = "CLOCK";
    #else
        const char* profilingMode = "GPU";
    #endif
    
    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", milliseconds_h_d,milliseconds,milliseconds_d_h, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", milliseconds_h_d,milliseconds,milliseconds_d_h);
    }else{
         printf("profiling mode: %s\n", profilingMode);
         printf("Elapsed time Host->Device: %.10f milliseconds\n", milliseconds_h_d);
         printf("Elapsed time kernel: %.10f milliseconds\n", milliseconds);
         printf("Elapsed time Device->Host: %.10f milliseconds\n", milliseconds_d_h);
    }
    return milliseconds;
}

void clean(GraficCommon* device_object){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    hipError_t err = hipSuccess;

    err = hipFree(deviceObj->d_B);

    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", hipGetErrorString(err));
        return;
    }
     err = hipFree(deviceObj->d_Br);

    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector Br (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    // delete events
    delete deviceObj->start;
    delete deviceObj->stop;
    delete deviceObj->start_memory_copy_device;
    delete deviceObj->stop_memory_copy_device;
    delete deviceObj->start_memory_copy_host;
    delete deviceObj->stop_memory_copy_host;
}
