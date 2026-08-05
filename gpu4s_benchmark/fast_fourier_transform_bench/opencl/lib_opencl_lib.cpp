// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include "Clock.h"
#include "vkFFT.h"

// kernel time execution
Clock kernelCLK;

void execute_kernel(GraficCommon* device_object, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    // --- convert c++ pointer to raw c ---
    cl_context          raw_context = (*deviceObj->context)();
    cl_device_id        raw_device  = deviceObj->default_device();
    cl_command_queue    raw_queue   = (*deviceObj->queue)();
    cl_mem              raw_input   = (*deviceObj->d_B)();
    cl_mem              raw_output  = (*deviceObj->d_Br)();

     // --- VkFFT configuration ---
    VkFFTConfiguration config = {};
    config.FFTdim           = 1;            // number of dim of FFT
    config.size[0]          = size;         // size of D1
    config.device           = &raw_device;  // select the device 
    config.context          = &raw_context; // memory space
    config.buffer           = &raw_output;  // output buff
    config.inputBuffer      = &raw_input;   // input buff  
    config.isInputFormatted = 1;            // different buffer for input/output
    #ifdef DOUBLE
    config.doublePrecision  = 1;
    #endif


    // --- init ---
    kernelCLK.start(); // Start clock
    VkFFTApplication app = {};
    initializeVkFFT(&app, config); // compile the FFT kernel for your GPU

    // --- launch ---
    VkFFTLaunchParams launchParams = {};
    launchParams.commandQueue  = &raw_queue; //select the queue

    VkFFTAppend(&app, -1, &launchParams); // -1 = forward FFT

    deviceObj->queue->finish();
    kernelCLK.end(); // End clock


    // --- cleanup ---
    deleteVkFFT(&app);
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->evt_copyBr->wait();
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    elapsed_h_d = deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Host->Device: %.10f \n", elapsed / 1000000.0);

    elapsed = kernelCLK.getElapsedNS();
    //printf("Elapsed time kernel: %.10f \n", elapsed / 1000000.0);

    elapsed_d_h = deviceObj->evt_copyBr->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyBr->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Device->Host: %.10f \n", );

     #ifdef PROFILING_CLOCK
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on PROFILING_CLOCK) ---
        elapsed_h_d  = h2dCLK.getElapsedNS();
        elapsed_d_h  = d2hCLK.getElapsedNS();
        const char* profilingMode = "CLOCK";
    #else
        const char* profilingMode = "GPU";
    #endif

    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n",elapsed_h_d / 1000000.0, elapsed / 1000000.0, elapsed_d_h / 1000000.0, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0, elapsed / 1000000.0,elapsed_d_h / 1000000.0);
    }else{
         printf("profiling mode: %s\n", profilingMode);
         printf("Elapsed time Host->Device: %.10f milliseconds\n", (elapsed_h_d / 1000000.0));
         printf("Elapsed time kernel: %.10f milliseconds\n", elapsed / 1000000.0 );
         printf("Elapsed time Device->Host: %.10f milliseconds\n", elapsed_d_h / 1000000.0);
    }
    return elapsed / 1000000.0; // TODO Change
}

