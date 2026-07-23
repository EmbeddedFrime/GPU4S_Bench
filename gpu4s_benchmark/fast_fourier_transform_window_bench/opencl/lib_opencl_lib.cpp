// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include "Clock.h"
#include "vkFFT.h"

// kernel time execution
Clock kernelCLK;
// host <-> device 
Clock h2dCLK;
Clock d2hCLK;


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
    cl_int err;


    deviceObj->d_A = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size_a_array, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    deviceObj->d_B = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,sizeof(bench_t)*size_b_array, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    // inicialice Arrays
    return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A,int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // copy memory host -> device

    h2dCLK.start();

    cl_int err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->d_A,CL_TRUE,0,sizeof(bench_t)*size, h_A, NULL, deviceObj->evt_copyB);
    if (err != CL_SUCCESS) 
    {
        fprintf(stderr, "Failed to copy data vector A from host to device (OpenCL error code %d)!\n", err);
        return;
    }
    
    deviceObj->queue->finish();
    h2dCLK.end();
}

void execute_kernel(GraficCommon* device_object, int64_t window, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    // --- convert c++ pointer to raw c ---
    cl_context          raw_context = (*deviceObj->context)();
    cl_device_id        raw_device  = deviceObj->default_device();
    cl_command_queue    raw_queue   = (*deviceObj->queue)();
    cl_mem              raw_input   = (*deviceObj->d_A)();
    cl_mem              raw_output  = (*deviceObj->d_B)();

    // --- VkFFT configuration ---
    VkFFTConfiguration config = {};
    config.FFTdim           = 1;            // number of dim of FFT
    config.size[0]          = window/2;         // size of D1
    config.device           = &raw_device;  // select the device 
    config.context          = &raw_context; // memory space
    config.buffer           = &raw_output;  // output buff
    config.inputBuffer      = &raw_input;   // input buff  
    config.isInputFormatted = 1;            // different buffer for input/output
    config.specifyOffsetsAtLaunch = 1;      // enables per-launch offsets
    #ifdef DOUBLE
    config.doublePrecision  = 1;
    #endif


    // --- init ---
    kernelCLK.start(); // Start clock
    VkFFTApplication app = {};
    initializeVkFFT(&app, config);

    // --- launch ---
    VkFFTLaunchParams launchParams = {};
    launchParams.commandQueue  = &raw_queue; //select the queue

    uint64_t input_offset  = 0;
    uint64_t output_offset = 0;

    for (unsigned int i = 0; i < (size * 2  - window + 1); i+=2){
        // update byte offsets into the same buffer for each window
        launchParams.inputBufferOffset  = input_offset;
        launchParams.bufferOffset       = output_offset; 

        VkFFTAppend(&app, -1, &launchParams);

        input_offset  += sizeof(bench_t) * 2;          // advance by 1 real element
        output_offset += sizeof(bench_t) * window *2; // advance by one window of output
    }

    deviceObj->queue->finish();
    kernelCLK.end(); // End clock


    // --- cleanup ---
    deleteVkFFT(&app);
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_B, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    d2hCLK.start();
    deviceObj->queue->enqueueReadBuffer(*deviceObj->d_B,CL_TRUE,0,sizeof(bench_t)*size,h_B, NULL, deviceObj->evt_copyBr);
    d2hCLK.end();
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->evt_copyBr->wait();
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    // elapsed_h_d = deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    // //printf("Elapsed time Host->Device: %.10f \n", elapsed / 1000000.0);
    // elapsed = deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    // //printf("Elapsed time kernel: %.10f \n", elapsed / 1000000.0);
    // elapsed_d_h = deviceObj->evt_copyBr->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyBr->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    // //printf("Elapsed time Device->Host: %.10f \n", );

    // --- FIX: unify all the clock ---
    elapsed_h_d  = h2dCLK.getElapsedNS();
    elapsed      = kernelCLK.getElapsedNS();
    elapsed_d_h  = d2hCLK.getElapsedNS();


    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", elapsed_h_d / 1000000.0,deviceObj->elapsed_time ,elapsed_d_h / 1000000.0, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0,deviceObj->elapsed_time,elapsed_d_h / 1000000.0);
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
    delete deviceObj->evt_copyB;
    delete deviceObj->evt_copyBr;
}
