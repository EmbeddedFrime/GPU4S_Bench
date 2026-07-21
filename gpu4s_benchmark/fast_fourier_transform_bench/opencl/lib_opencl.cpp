// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include "GEN_kernel.hcl"
#include <chrono>


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

bool device_memory_init(GraficCommon* device_object, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    cl_int err;

   deviceObj->d_B = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size, nullptr, &err);
   if (err != CL_SUCCESS) return false;

   deviceObj->d_Br = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,sizeof(bench_t)*size, nullptr, &err);
   if (err != CL_SUCCESS) return false;
   // inicialice Arrays
   return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_B,int64_t size){
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// copy memory host -> device

    #ifdef ANDROID
        h2dCLK.start();
    #endif

    cl_int err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->d_B,CL_TRUE,0,sizeof(bench_t)*size, h_B, NULL, deviceObj->evt_copyB);
    if (err != CL_SUCCESS) 
    {
        fprintf(stderr, "Failed to copy data vector B from host to device (OpenCL error code %d)!\n", err);
        return;
    }
    
    #ifdef ANDROID
        deviceObj->queue->finish();
        h2dCLK.end();
    #endif
}

void execute_kernel(GraficCommon* device_object, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    struct timespec start, end;
    const unsigned int x_local= BLOCK_SIZE;
    unsigned int mode = (unsigned int)log2(size);
    cl::NDRange local_reverse, global_reverse, local, global;
    if (size > BLOCK_SIZE)
    {
        local_reverse =  cl::NDRange (x_local);
        global_reverse = cl::NDRange (size);
    }
    else
    {
        local_reverse = cl::NullRange;
        global_reverse = cl::NDRange(size);
    }
   

    //cl::NDRange local(x_local, y_local);
    //cl::NDRange global(n, w);

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
    // reverse bit operation 
    cl::Kernel kernel_add=cl::Kernel(program,"binary_reverse_kernel");
    kernel_add.setArg(0,*deviceObj->d_B);
    kernel_add.setArg(1,*deviceObj->d_Br);
    kernel_add.setArg(2,size);
    kernel_add.setArg(3,mode);

    // clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    #ifdef ANDROID
        deviceObj->queue->finish(); // Clear queue to ensure accurate start
        kernelCLK.start();
    #endif
    deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global_reverse,local_reverse, NULL, deviceObj->evt);

    deviceObj->queue->finish();
    // FFT calculation
    bench_t wtemp, wr, wpr, wpi, wi, theta;
    unsigned int theads = size/2;
    unsigned int loop = 1;
    cl::Kernel kernel_fft=cl::Kernel(program,"fft_kernel");
    

    while(loop < size){
        // caluclate values 
        theta = -(M_PI/loop); // check
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        // calculate block size and thead size
        if (theads % BLOCK_SIZE != 0){
            // inferior part
            global = cl::NDRange (theads);
            local = cl::NullRange;
        }
        else{
            // top part
            local =  cl::NDRange (x_local);
            global = cl::NDRange (theads);
        }
        // launch kernel loop times
        for(unsigned int i = 0; i < loop; ++i){
            //kernel launch 
            kernel_fft.setArg(0,*deviceObj->d_Br);
            kernel_fft.setArg(1,loop);
            kernel_fft.setArg(2,i);
            kernel_fft.setArg(3,wr);
            kernel_fft.setArg(4,wi);
            deviceObj->queue->enqueueNDRangeKernel(kernel_fft,cl::NullRange,global,local, NULL, deviceObj->evt);
            // update WR, WI
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
            
        }
        // update loop values
        loop = loop * 2;
        theads = theads / 2;
       
    }

    
    deviceObj->queue->finish();
    // clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    #ifdef ANDROID
        kernelCLK.end();
    #endif

    deviceObj->elapsed_time =  (end.tv_sec - start.tv_sec) * 1000.0f + (end.tv_nsec - start.tv_nsec) / 1000000.0f;

}

void copy_memory_to_host(GraficCommon* device_object, bench_t* h_B, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    #ifdef ANDROID
        d2hCLK.start();
    #endif

    deviceObj->queue->enqueueReadBuffer(*deviceObj->d_Br,CL_TRUE,0,sizeof(bench_t)*size,h_B, NULL, deviceObj->evt_copyBr);
     
    #ifdef ANDROID
        deviceObj->queue->finish();
        d2hCLK.end();
    #endif
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

    #ifdef ANDROID
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on Android) ---
        elapsed_h_d  = h2dCLK.getElapsed();
        elapsed      = kernelCLK.getElapsed();
        elapsed_d_h  = d2hCLK.getElapsed();
    #endif


    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n",elapsed_h_d / 1000000.0, deviceObj->elapsed_time , elapsed_d_h / 1000000.0, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0,deviceObj->elapsed_time,elapsed_d_h / 1000000.0);
    }else{
         printf("Elapsed time Host->Device: %.10f milliseconds\n", (elapsed_h_d / 1000000.0));
         printf("Elapsed time kernel: %.10f milliseconds\n", elapsed / 1000000.0 );
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
    delete deviceObj->d_B;
    delete deviceObj->d_Br;
    delete deviceObj->evt;
    delete deviceObj->evt_copyB;
    delete deviceObj->evt_copyBr;
}
