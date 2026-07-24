// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include "vkFFT.h"


// kernel time execution
Clock kernelCLK;
#ifdef PROFILING_CLOCK
    // host <-> device 
    Clock h2dCLK;
    Clock d2hCLK;
#endif


//#define BLOCK_SIZE 32
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
    deviceObj->evt_copyB = new cl::Event;
    deviceObj->evt_copyBr = new cl::Event;
    

}

bool device_memory_init(GraficCommon* device_object, int64_t size){
   GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
   cl_int err;

   deviceObj->d_A = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,sizeof(bench_t)* size * size * 2, nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   deviceObj->d_B = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,sizeof(bench_t)* size * size * 2, nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // inicialice Arrays
   return true;
}

void copy_memory_to_device(GraficCommon *device_object, COMPLEX **h_B,int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // --- init ---
    bench_t *h_signal = (bench_t *)malloc(sizeof(bench_t) * size * size * 2);
    for (int i=0; i<size; ++i)
        {
            for (int j=0; j<size; ++j)
            {
                    h_signal[2*(j+i*size)] = h_B[i][j].x ;
                    h_signal[2*(j+i*size)+1] = h_B[i][j].y;
            }
        }

    #ifdef PROFILING_CLOCK
        h2dCLK.start();
    #endif

    // copy memory host -> device
    deviceObj->queue->enqueueWriteBuffer(*deviceObj->d_A,CL_TRUE,0,sizeof(bench_t)*size*size*2, h_signal, NULL, deviceObj->evt_copyB);

    #ifdef PROFILING_CLOCK
        deviceObj->queue->finish();
        h2dCLK.end();
    #endif
    free(h_signal);
}

void execute_kernel(GraficCommon* device_object, int64_t size) {
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    // --- convert c++ pointer to raw c ---
    cl_context      raw_context = (*deviceObj->context)();
    cl_device_id    raw_device  = deviceObj->default_device();
    cl_command_queue raw_queue  = (*deviceObj->queue)();
    cl_mem raw_input            = (*deviceObj->d_A)();
    cl_mem raw_output           = (*deviceObj->d_B)();

    // --- VkFFT configuration ---
    VkFFTConfiguration config = {};
    config.FFTdim           = 2;            // number of dim of FFT
    config.size[0]          = size;         // size of D1
    config.size[1]          = size;         // size of D2
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

void copy_memory_to_host(GraficCommon* device_object, COMPLEX **h_B, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    bench_t *h_signal = (bench_t *)malloc(sizeof(bench_t) * size * size * 2);

    #ifdef PROFILING_CLOCK
        d2hCLK.start();
    #endif
    
    deviceObj->queue->enqueueReadBuffer(*deviceObj->d_B,CL_TRUE,0,sizeof(bench_t)*size*size * 2,h_signal, NULL, deviceObj->evt_copyBr);
    
    #ifdef PROFILING_CLOCK
        deviceObj->queue->finish();
        d2hCLK.end();
    #endif
    
    for (int i=0; i<size; ++i)
        {
            for (int j=0; j<size; ++j)
            {
                    h_B[i][j].x = h_signal[2*(j+i*size)];
                    h_B[i][j].y = h_signal[2*(j+i*size)+1];
            }
        }
    free(h_signal);
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->evt_copyBr->wait();
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    elapsed_h_d = deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Host->Device: %.10f \n", elapsed / 1000000.0);
    elapsed      = kernelCLK.getElapsedNS();
    elapsed_d_h = deviceObj->evt_copyBr->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyBr->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Device->Host: %.10f \n", );


    #ifdef PROFILING_CLOCK
        elapsed_h_d  = h2dCLK.getElapsedNS();
        elapsed_d_h  = d2hCLK.getElapsedNS();
        const char* profilingMode = "CLOCK";
    #else
        const char* profilingMode = "GPU";
    #endif

    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", elapsed_h_d / 1000000.0, elapsed / 1000000.0,elapsed_d_h / 1000000.0, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0, elapsed / 1000000.0,elapsed_d_h / 1000000.0);
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
    //delete deviceObj->context;
    //delete deviceObj->queue;
    // pointer to memory
    delete deviceObj->d_A;
    delete deviceObj->d_B;
    delete deviceObj->evt;
    delete deviceObj->evt_copyB;
    delete deviceObj->evt_copyBr;
}