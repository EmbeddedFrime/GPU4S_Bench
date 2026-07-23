// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <cstring>
#include "GEN_kernel.hcl"
#include "GEN_atomic_functions.hcl"


// kernel time execution
Clock kernelCLK;
#ifdef ANDROID
    // host <-> device 
    Clock h2dCLK;
    Clock d2hCLK;
#endif


//#define BLOCK_SIZE 16
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
    deviceObj->evt_copyIN = new cl::Event;
    deviceObj->evt_copyK1 = new cl::Event;
    deviceObj->evt_copyK2 = new cl::Event;
    deviceObj->evt_copyW1 = new cl::Event;
    deviceObj->evt_copyW2 = new cl::Event;
    deviceObj->evt_copyOut = new cl::Event;

    deviceObj->evt1_1 = new cl::Event;
    deviceObj->evt1_2 = new cl::Event; 
    deviceObj->evt1_3 = new cl::Event; 
    deviceObj->evt1_4 = new cl::Event; 
    deviceObj->evt2_1 = new cl::Event;  
    deviceObj->evt2_2 = new cl::Event; 
    deviceObj->evt2_3 = new cl::Event;
    deviceObj->evt2_1 = new cl::Event;  
    deviceObj->evt2_2 = new cl::Event; 
    deviceObj->evt2_3 = new cl::Event; 
    deviceObj->evt2_4 = new cl::Event;  
    deviceObj->evtd_1 = new cl::Event; 
    deviceObj->evtd_1_a = new cl::Event;
    deviceObj->evtd_2 = new cl::Event; 
    deviceObj->evtd_2_a = new cl::Event;  
    deviceObj->evt_softmax = new cl::Event;
    deviceObj->evt_softmax_fin = new cl::Event;   

    
}

bool device_memory_init(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images){
   GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
   cl_int err;
   unsigned int size_pooling_1 = input_data / stride_1;
   unsigned int size_pooling_2 = size_pooling_1 / stride_2;
   unsigned int weights_layer_1 = size_pooling_2 * size_pooling_2 * neurons_dense_1;
   unsigned int weights_layer_2 = neurons_dense_1 * neurons_dense_2; 

   // input
   deviceObj->input_data = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,number_of_images * input_data * input_data * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // convolution 1
   deviceObj->kernel_1 = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,kernel_1 * kernel_1 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;

   deviceObj->conv_1_output = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,input_data * input_data * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // pooling 1
   deviceObj->pooling_1_output = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,size_pooling_1 * size_pooling_1 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
  
   // convolution 1
   deviceObj->kernel_2 = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,kernel_2 * kernel_2 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;

   deviceObj->conv_2_output = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,size_pooling_1 * size_pooling_1 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // pooling 2 
   deviceObj->pooling_2_output = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,size_pooling_2 * size_pooling_2 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // dense 1
   deviceObj->dense_layer_1_weights = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,weights_layer_1 * sizeof(bench_t));
   if (err != CL_SUCCESS) return false;

   deviceObj->dense_layer_1_output = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,neurons_dense_1 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // dense 2
   deviceObj->dense_layer_2_weights = new cl::Buffer(*deviceObj->context,CL_MEM_READ_ONLY ,weights_layer_2 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;

   deviceObj->dense_layer_2_output = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,neurons_dense_2 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   
   // out
   deviceObj->sum_ouput = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;

   deviceObj->output_data = new cl::Buffer(*deviceObj->context,CL_MEM_READ_WRITE ,number_of_images * neurons_dense_2 * sizeof(bench_t), nullptr, &err);
   if (err != CL_SUCCESS) return false;
   return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size, unsigned int number_of_images){
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// copy memory host -> device

    #ifdef ANDROID
        h2dCLK.start();
    #endif


    // input data
    cl_int err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->input_data,CL_TRUE,0,sizeof(bench_t)* input * input * number_of_images, input_data, NULL, deviceObj->evt_copyIN);
    if (err != CL_SUCCESS) 
    {
        fprintf(stderr, "Failed to copy input_data from host to device (OpenCL error code %d)!\n", err);
        return;
    }
    
    // kernels
    err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->kernel_1,CL_TRUE,0,sizeof(bench_t)* kernel_size_1 * kernel_size_1, kernel_1_data, NULL, deviceObj->evt_copyK1);
    if (err != CL_SUCCESS) 
    {
        fprintf(stderr, "Failed to copy kernel_1 from host to device (OpenCL error code %d)!\n", err);
        return;
    }

    err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->kernel_2,CL_TRUE,0,sizeof(bench_t)* kernel_size_2 * kernel_size_2, kernel_2_data, NULL, deviceObj->evt_copyK2);
    if (err != CL_SUCCESS) 
    {
        fprintf(stderr, "Failed to copy kernel_2 from host to device (OpenCL error code %d)!\n", err);
        return;
    }

    // dense layer
    err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->dense_layer_1_weights,CL_TRUE,0,sizeof(bench_t)* weights_1_size, weights_1, NULL, deviceObj->evt_copyW1);
    if (err != CL_SUCCESS) 
    {
        fprintf(stderr, "Failed to copy dense_layer_1_weights from host to device (OpenCL error code %d)!\n", err);
        return;
    }

    err = deviceObj->queue->enqueueWriteBuffer(*deviceObj->dense_layer_2_weights,CL_TRUE,0,sizeof(bench_t)* weights_2_size, weights_2, NULL, deviceObj->evt_copyW2);
    if (err != CL_SUCCESS) 
    {
        fprintf(stderr, "Failed to copy dense_layer_2 from host to device (OpenCL error code %d)!\n", err);
        return;
    }

    #ifdef ANDROID
        deviceObj->queue->finish();
        h2dCLK.end();
    #endif
}


void execute_kernel(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    unsigned int x_local= BLOCK_SIZE;
    unsigned int y_local= BLOCK_SIZE;
    cl::NDRange local;
    cl::NDRange global;

    cl::Program::Sources sources;
    // load kernel from file
    kernel_code = type_kernel_common + atomic_code + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    // build
    cl::Program program(*deviceObj->context,sources);
    if(program.build({deviceObj->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(deviceObj->default_device)<<"\n";
        exit(1);
    }

    // timing  
    deviceObj->queue->finish(); // Clear queue to ensure accurate start
    kernelCLK.start();

    cl::Buffer* aux_output_data = deviceObj->output_data;
    cl::Buffer* aux_input_data = deviceObj->input_data;
    for (unsigned int position = 0; position < number_of_images; ++position)
    {
        aux_input_data = deviceObj->input_data;
        aux_output_data = deviceObj->output_data;
        // 1-1 step convolution
        if (input_data <= BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange(input_data, input_data);
        }
        else
        {
            local = cl::NDRange(x_local, y_local);
            global = cl::NDRange(input_data, input_data);
        }
        cl::Kernel  kernel_conv=cl::Kernel(program,"kernel_matrix_convolution");
        kernel_conv.setArg(0,*aux_input_data);
        kernel_conv.setArg(1,*deviceObj->conv_1_output);
        kernel_conv.setArg(2,*deviceObj->kernel_1);
        kernel_conv.setArg(3,input_data);
        kernel_conv.setArg(4,input_data);
        kernel_conv.setArg(5,input_data);
        kernel_conv.setArg(6,kernel_1);
        kernel_conv.setArg(7, position * input_data * input_data);

        deviceObj->queue->enqueueNDRangeKernel(kernel_conv,cl::NullRange,global,local, NULL, deviceObj->evt1_1);

        // 1-2 step activation
        cl::Kernel  kernel_add=cl::Kernel(program,"kernel_relu");
        kernel_add.setArg(0,*deviceObj->conv_1_output);
        kernel_add.setArg(1,*deviceObj->conv_1_output);
        kernel_add.setArg(2,input_data);
        deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt1_2);

        // 1-3 step pooling
        unsigned int size_lateral_1 = input_data / stride_1;
        if(size_lateral_1 <= BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange (size_lateral_1, size_lateral_1);
        }
        else
        {
            local = cl::NDRange(x_local, y_local);
            global = cl::NDRange(size_lateral_1, size_lateral_1);
        }
        kernel_add=cl::Kernel(program,"kernel_max");
        kernel_add.setArg(0,*deviceObj->conv_1_output);
        kernel_add.setArg(1,*deviceObj->pooling_1_output);
        kernel_add.setArg(2,input_data);
        kernel_add.setArg(3,stride_1);
        kernel_add.setArg(4,size_lateral_1);
        deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt1_3);
        
        // 1-4 step normalitation
        kernel_add=cl::Kernel(program,"kernel_lrn");
        kernel_add.setArg(0,*deviceObj->pooling_1_output);
        kernel_add.setArg(1,*deviceObj->pooling_1_output);
        kernel_add.setArg(2,size_lateral_1);
        kernel_add.setArg(3,K);
        kernel_add.setArg(4,ALPHA);
        kernel_add.setArg(5,BETA);

       deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt1_4);

        // 2-1 step convolution
        kernel_conv=cl::Kernel(program,"kernel_matrix_convolution");
        kernel_conv.setArg(0,*deviceObj->pooling_1_output);
        kernel_conv.setArg(1,*deviceObj->conv_2_output);
        kernel_conv.setArg(2,*deviceObj->kernel_2);
        kernel_conv.setArg(3,size_lateral_1);
        kernel_conv.setArg(4,size_lateral_1);
        kernel_conv.setArg(5,size_lateral_1);
        kernel_conv.setArg(6,kernel_2);
        kernel_conv.setArg(7, 0);

        deviceObj->queue->enqueueNDRangeKernel(kernel_conv,cl::NullRange,global,local, NULL, deviceObj->evt2_1);

        // 2-2 step activation
        kernel_add=cl::Kernel(program,"kernel_relu");
        kernel_add.setArg(0,*deviceObj->conv_2_output);
        kernel_add.setArg(1,*deviceObj->conv_2_output);
        kernel_add.setArg(2,size_lateral_1);
        deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt2_2);
        
        // 2-3 normalization
        kernel_add=cl::Kernel(program,"kernel_lrn");
        kernel_add.setArg(0,*deviceObj->conv_2_output);
        kernel_add.setArg(1,*deviceObj->conv_2_output);
        kernel_add.setArg(2,size_lateral_1);
        kernel_add.setArg(3,K);
        kernel_add.setArg(4,ALPHA);
        kernel_add.setArg(5,BETA);

        deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt2_3);

        // 2-4 step pooling
        unsigned int size_lateral_2 = size_lateral_1 / stride_2;
        if(size_lateral_2 <= BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange (size_lateral_2, size_lateral_2);
        }
        else
        {
            local = cl::NDRange(x_local, y_local);
            global = cl::NDRange(size_lateral_2, size_lateral_2);
        }
        kernel_add=cl::Kernel(program,"kernel_max");
        kernel_add.setArg(0,*deviceObj->conv_2_output);
        kernel_add.setArg(1,*deviceObj->pooling_2_output);
        kernel_add.setArg(2,size_lateral_1);
        kernel_add.setArg(3,stride_2);
        kernel_add.setArg(4,size_lateral_2);
        deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evt2_4);
        // dense layer 1
        if(neurons_dense_1 <= BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange (neurons_dense_1, 1);
        }
        else
        {
            local = cl::NDRange(x_local, 1);
            global = cl::NDRange(neurons_dense_1, 1);
        }
        kernel_add=cl::Kernel(program,"kernel_matrix_multiplication");
        kernel_add.setArg(0,*deviceObj->dense_layer_1_weights);
        kernel_add.setArg(1,*deviceObj->pooling_2_output);
        kernel_add.setArg(2,*deviceObj->dense_layer_1_output);
        kernel_add.setArg(3,neurons_dense_1);
        kernel_add.setArg(4,1);
        kernel_add.setArg(5,size_lateral_2*size_lateral_2);

        deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evtd_1);

        //activation layer dense 1
        if(neurons_dense_1 <= BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange (neurons_dense_1/2, neurons_dense_1/2);
        }
        else
        {
            local = cl::NDRange(x_local, y_local);
            global = cl::NDRange(neurons_dense_1/2, neurons_dense_1/2);
        }
        kernel_add=cl::Kernel(program,"kernel_relu");
        kernel_add.setArg(0,*deviceObj->dense_layer_1_output);
        kernel_add.setArg(1,*deviceObj->dense_layer_1_output);
        kernel_add.setArg(2,neurons_dense_1/2);
        deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evtd_1_a);

        // dense layer 2
        if(neurons_dense_2 <= BLOCK_SIZE)
        {
            local = cl::NDRange(1, 1);
            global = cl::NDRange (neurons_dense_2, 1);
        }
        else
        {
            local = cl::NullRange;
            global = cl::NDRange(neurons_dense_2, 1);
        }
        kernel_add=cl::Kernel(program,"kernel_matrix_multiplication");
        kernel_add.setArg(0,*deviceObj->dense_layer_2_weights);
        kernel_add.setArg(1,*deviceObj->dense_layer_1_output);
        kernel_add.setArg(2,*deviceObj->dense_layer_2_output);
        kernel_add.setArg(3,neurons_dense_2);
        kernel_add.setArg(4,1);
        kernel_add.setArg(5,neurons_dense_1);

        deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evtd_2);

        //activation layer dense 2
        if(neurons_dense_2 < BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange (neurons_dense_2/2, neurons_dense_2/2);
        }
        else
        {
            local = cl::NDRange(x_local, y_local);
            global = cl::NDRange(neurons_dense_2/2, neurons_dense_2/2);
        }
        kernel_add=cl::Kernel(program,"kernel_relu");
        kernel_add.setArg(0,*deviceObj->dense_layer_2_output);
        kernel_add.setArg(1,*deviceObj->dense_layer_2_output);
        kernel_add.setArg(2,neurons_dense_2/2);
        deviceObj->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, deviceObj->evtd_2_a);

        //soft max
        if(neurons_dense_2 < BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange (1, neurons_dense_2);
        }
        else
        {
            local = cl::NDRange(1, x_local);
            global = cl::NDRange(1, neurons_dense_2);
        }
        cl::Kernel softmax_kernel=cl::Kernel(program,"kernel_softmax");
        softmax_kernel.setArg(0,*deviceObj->dense_layer_2_output);
        softmax_kernel.setArg(1,*aux_output_data);
        softmax_kernel.setArg(2,*deviceObj->sum_ouput);
        softmax_kernel.setArg(3,neurons_dense_2);
        softmax_kernel.setArg(4, position * output_data);

        deviceObj->queue->enqueueNDRangeKernel(softmax_kernel,cl::NullRange,global,local, NULL, deviceObj->evt_softmax);

        cl::Kernel softmax_end_kernel=cl::Kernel(program,"kernel_softmax_end");
        softmax_end_kernel.setArg(0,*aux_output_data);
        softmax_end_kernel.setArg(1,*deviceObj->sum_ouput);
        softmax_end_kernel.setArg(2,neurons_dense_2);
        softmax_end_kernel.setArg(3, position * output_data);

        deviceObj->queue->enqueueNDRangeKernel(softmax_end_kernel,cl::NullRange,global,local, NULL, deviceObj->evt_softmax_fin);
        deviceObj->queue->enqueueWriteBuffer(*deviceObj->sum_ouput,CL_TRUE,0,sizeof(bench_t), 0, NULL,NULL);
        
    }
    // end 
    deviceObj->queue->finish();
    kernelCLK.end();
    deviceObj->elapsed_time =  kernelCLK.getElapsedMS();

}

void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size, unsigned int number_of_images){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    #ifdef ANDROID
        d2hCLK.start();
    #endif

    deviceObj->queue->enqueueReadBuffer(*deviceObj->output_data,CL_TRUE,0,sizeof(bench_t)*size*number_of_images,h_C, NULL, deviceObj->evt_copyOut);
    //deviceObj->queue->enqueueReadBuffer(*deviceObj->conv_2_output,CL_TRUE,0,sizeof(bench_t)*16*16,h_C, NULL, deviceObj->evt_copyOut);
    
    #ifdef ANDROID
        deviceObj->queue->finish();
        d2hCLK.end();
    #endif
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format,bool csv_format_timestamp, long int current_time)
{
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->evt_copyOut->wait();

    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;

    // copy memory H -> D
    elapsed_h_d = deviceObj->evt_copyIN->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyIN->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += deviceObj->evt_copyK1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyK1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += deviceObj->evt_copyK2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyK2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += deviceObj->evt_copyW1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyW1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += deviceObj->evt_copyW2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyW2->getProfilingInfo<CL_PROFILING_COMMAND_START>();

    // kernel time

    elapsed = deviceObj->evt1_1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt1_1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evt1_2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt1_2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evt1_3->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt1_3->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evt1_4->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt1_4->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evt2_1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt2_1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evt2_2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt2_2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evt2_3->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt2_3->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evt2_4->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt2_4->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evtd_1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evtd_1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evtd_1_a->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evtd_1_a->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evtd_2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evtd_2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evtd_2_a->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evtd_2_a->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evt_softmax->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_softmax->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += deviceObj->evt_softmax_fin->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_softmax_fin->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    
    // copy memory D -> H
    elapsed_d_h = deviceObj->evt_copyOut->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyOut->getProfilingInfo<CL_PROFILING_COMMAND_START>();

    #ifdef ANDROID
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on Android) ---
        elapsed_h_d  = h2dCLK.getElapsedNS();
        elapsed      = kernelCLK.getElapsedNS();
        elapsed_d_h  = d2hCLK.getElapsedNS();
    #endif

    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", elapsed_h_d / 1000000.0,deviceObj->elapsed_time,elapsed_d_h / 1000000.0, current_time);
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

    delete deviceObj->evt_copyIN;
    delete deviceObj->evt_copyK1;
    delete deviceObj->evt_copyK2;
    delete deviceObj->evt_copyW1;
    delete deviceObj->evt_copyW2;
    delete deviceObj->evt_copyOut;
    delete deviceObj->evt1_1;
    delete deviceObj->evt1_2;
    delete deviceObj->evt1_3;
    delete deviceObj->evt1_4;
    delete deviceObj->evt2_1;
    delete deviceObj->evt2_2;
    delete deviceObj->evt2_3;
    delete deviceObj->evt2_4;
    delete deviceObj->evtd_1;
    delete deviceObj->evtd_1_a;
    delete deviceObj->evtd_2;
    delete deviceObj->evtd_2_a;
    delete deviceObj->evt_softmax;
    delete deviceObj->evt_softmax_fin;

    delete deviceObj->input_data;
    delete deviceObj->kernel_1;
    delete deviceObj->conv_1_output;
    delete deviceObj->pooling_1_output;
    delete deviceObj->kernel_2;
    delete deviceObj->conv_2_output;
    delete deviceObj->pooling_2_output;
    delete deviceObj->dense_layer_1_weights;
    delete deviceObj->dense_layer_1_output;
    delete deviceObj->dense_layer_2_weights;
    delete deviceObj->dense_layer_2_output;
    delete deviceObj->output_data;
    delete deviceObj->sum_ouput;
}
