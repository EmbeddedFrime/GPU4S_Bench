/** * ====================================================================
 * @file        opencl_common.cpp (./matrix_multiplication_bench)
 * @brief       Common OpenCL platform initialization, device setup, 
 *              profiling timer evaluation, and generic cleanup routines.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#include "../benchmark_library.h"
#include "../cpu_functions/cpu_functions.h"
#include "opencl_common.h"
#include <cstring>



void init(GraficCommon* device_object, char* device_name){
	init(device_object, 0,0, device_name);
}
void init(GraficCommon* device_object, int platform ,int device, char* device_name){
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // --- Fix Initialize the struct to prevent garbage values in C++ members ---
    memset(device_object, 0, sizeof(GraficObject));
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
    deviceObj->evt_copyC = new cl::Event;
    
}

bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix){
   GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
   cl_int err;
   deviceObj->d_A = new cl::Buffer(*deviceObj->context, CL_MEM_READ_ONLY, sizeof(bench_t)*size_a_matrix, nullptr, &err);
   if (err != CL_SUCCESS) return false;

   deviceObj->d_B = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size_b_matrix, nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   deviceObj->d_C = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,sizeof(bench_t)*size_c_matrix, nullptr, &err);
   if (err != CL_SUCCESS) return false;

   // inicialice Arrays
   return true;
}



void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int size_b){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    #ifdef PROFILING_CLOCK
        h2dCLK.start();
    #endif

    // copy memory host -> device
    
    cl_int err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->d_A,CL_TRUE,0,sizeof(bench_t)*size_a, h_A, NULL, deviceObj->evt_copyA);
    if (err != CL_SUCCESS) 
    {
        fprintf(stderr, "Failed to copy vector A from host to device (OpenCL error code %d)!\n", err);
        return;
    }

    // Enqueue writing host memory h_B to device buffer d_B
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

#ifdef UNIFIED_MEMORY
void device_unified_memory_init_copy(GraficCommon* device_object, bench_t* &A, bench_t* &B, bench_t* &C, unsigned int buff_size, char input_file_A[100], char input_file_B[100]){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    unsigned int squared_buff_size = buff_size * buff_size;
    unsigned int mem_size = squared_buff_size * sizeof(bench_t);

    h2dCLK.start();
    //--- Aquire the pointer of buffer from graphic card ---
    A = (bench_t*)deviceObj->queue->enqueueMapBuffer(
        *deviceObj->d_A, CL_TRUE, CL_MAP_WRITE, 0, mem_size);
    B = (bench_t*)deviceObj->queue->enqueueMapBuffer(
        *deviceObj->d_B, CL_TRUE, CL_MAP_WRITE, 0, mem_size);
    C = (bench_t*)deviceObj->queue->enqueueMapBuffer(
        *deviceObj->d_C, CL_TRUE, CL_MAP_WRITE, 0, mem_size);
    h2dCLK.end();
    
    h2dTotal += h2dCLK.getElapsedNS();

    if (strlen(input_file_A) == 0)
    {

        // --- Initialize the buffer ---
        for (int i = 0; i < buff_size; i++)
            for (int j = 0; j < buff_size; j++)
            A[i*buff_size+j] = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);

        for (int i = 0; i < buff_size; i++)
            for (int j = 0; j < buff_size; j++)
                B[i*buff_size+j] = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);

        for (int i = 0; i < buff_size; i++)
            for (int j = 0; j < buff_size; j++)
                C[i*buff_size+j] = 0;

    } else 
    {
        get_double_hexadecimal_values(input_file_A, A,mem_size);
		get_double_hexadecimal_values(input_file_B, B,mem_size);
    }
    

    h2dCLK.start();
    // --- Unmap the buffers for GPU kernel ---
    deviceObj->queue->enqueueUnmapMemObject(*deviceObj->d_A, A, NULL, deviceObj->evt_copyA);
    deviceObj->queue->enqueueUnmapMemObject(*deviceObj->d_B, B, NULL, deviceObj->evt_copyB);
    deviceObj->queue->enqueueUnmapMemObject(*deviceObj->d_C, C, NULL, deviceObj->evt_copyC);


    deviceObj->queue->finish();
    h2dCLK.end();

    h2dTotal += h2dCLK.getElapsedNS();
}
#endif




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

#ifdef UNIFIED_MEMORY
void copy_memory_unified_to_host(GraficCommon* device_object, bench_t* &d_C, unsigned int buff_size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    d2hCLK.start();
    // Map the output buffer to d_C pointer   
    d_C = (bench_t*)deviceObj->queue->enqueueMapBuffer(*deviceObj->d_C, CL_TRUE, CL_MAP_READ, 0, buff_size, NULL, deviceObj->evt_copyC);
    
    
    deviceObj->queue->finish();
    d2hCLK.end();
}
#endif

float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->evt_copyC->wait();
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    elapsed_h_d = deviceObj->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Host->Device: %.10f \n", elapsed / 1000000.0);
    elapsed = deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time kernel: %.10f \n", elapsed / 1000000.0);
    elapsed_d_h = deviceObj->evt_copyC->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyC->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Device->Host: %.10f \n", );

    #ifdef PROFILING_CLOCK
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on PROFILING_CLOCK) ---
        elapsed_h_d  = h2dCLK.getElapsedNS();
        elapsed      = kernelCLK.getElapsedNS();
        elapsed_d_h  = d2hCLK.getElapsedNS();
        // --- select the profiling message ---
        const char* profilingMode = "CLOCK";
    #else
        const char* profilingMode = "GPU";
    #endif

    #ifdef UNIFIED_MEMORY
        elapsed_h_d = h2dTotal;
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
    delete deviceObj->d_C;
    delete deviceObj->evt;
    delete deviceObj->evt_copyA;
    delete deviceObj->evt_copyB;
    delete deviceObj->evt_copyC;
}
