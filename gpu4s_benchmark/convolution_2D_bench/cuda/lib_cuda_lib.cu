#include <cudnn.h>
#include "../benchmark_library.h"


#ifdef PROFILING_CLOCK
    // kernel time execution
    Clock kernelCLK;
    // host <-> device 
    Clock h2dCLK;
    Clock d2hCLK;
#endif
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#ifdef INT
  #define CUDNNTYPE CUDNN_DATA_INT32
#elif FLOAT
  #define CUDNNTYPE CUDNN_DATA_FLOAT
#elif DOUBLE
  #define CUDNNTYPE CUDNN_DATA_DOUBLE
#endif

void init(GraficCommon* device_object, char* device_name){
    init(device_object, 0,0, device_name);
}

void init(GraficCommon* device_object, int platform ,int device, char* device_name){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    //printf("Using device: %s\n", prop.name);
    strcpy(device_name,prop.name);
    //event create 
    deviceObj->start = new cudaEvent_t;
    deviceObj->stop = new cudaEvent_t;
    deviceObj->start_memory_copy_device = new cudaEvent_t;
    deviceObj->stop_memory_copy_device = new cudaEvent_t;
    deviceObj->start_memory_copy_host = new cudaEvent_t;
    deviceObj->stop_memory_copy_host= new cudaEvent_t;
    
    cudaEventCreate(deviceObj->start);
    cudaEventCreate(deviceObj->stop);
    cudaEventCreate(deviceObj->start_memory_copy_device);
    cudaEventCreate(deviceObj->stop_memory_copy_device);
    cudaEventCreate(deviceObj->start_memory_copy_host);
    cudaEventCreate(deviceObj->stop_memory_copy_host);
}


bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix){

GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

   // Allocate the device input vector A
	cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&deviceObj->d_A, size_a_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    // Allocate the device input vector B
    err = cudaMalloc((void **)&deviceObj->d_B, size_b_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    // Allocate the device output vector C
    err = cudaMalloc((void **)&deviceObj->kernel, size_c_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, bench_t* kernel, unsigned int size_a, unsigned int size_b){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    #ifdef PROFILING_CLOCK
        h2dCLK.start();
    #endif
    cudaEventRecord(*deviceObj->start_memory_copy_device);
	cudaError_t err = cudaMemcpy(deviceObj->d_A, h_A, sizeof(bench_t) * size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(deviceObj->kernel, kernel, sizeof(bench_t) * size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    cudaEventRecord(*deviceObj->stop_memory_copy_device);   
    #ifdef PROFILING_CLOCK
        h2dCLK.end();
    #endif
}
void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    if (kernel_size % 2 == 0){
        printf ("-k args must be an odd number\n");
        exit(1);
    }

    // cublas settings
    const bench_t alf = 1;
    const bench_t bet = 0;
    cudnnHandle_t cudnn;

    #ifdef PROFILING_CLOCK
        kernelCLK.start();
    #endif
    cudaEventRecord(*deviceObj->start);
    checkCUDNN(cudnnCreate(&cudnn));
    // create input tensor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/m,
                                      /*image_width=*/m));
    // create output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/m,
                                      /*image_width=*/m));
    // create kernel tensor
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/CUDNNTYPE,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*out_channels=*/1,
                                      /*in_channels=*/1,
                                      /*kernel_height=*/kernel_size,
                                      /*kernel_width=*/kernel_size));
    // create kernel descriptor
    // --- FIX: Change the size of pad depending of kernel_size --- 
    int pad = kernel_size / 2;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/pad,
                                           /*pad_width=*/pad,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CROSS_CORRELATION,
                                           /*computeType=*/CUDNNTYPE));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing convolution
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    int returned_algo_count;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        /*requestedAlgoCount=*/1,
                                        &returned_algo_count,
                                        &algo_perf));
    cudnnConvolutionFwdAlgo_t convolution_algorithm = algo_perf.algo;
    // get memory needed for the convolution
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   convolution_algorithm,
                                                   &workspace_bytes));
    // alocate memory for workspace
    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);
    // perform the convolution
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                   &alf,
                                   input_descriptor,
                                   deviceObj->d_A,
                                   kernel_descriptor,
                                   deviceObj->kernel,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &bet,
                                   output_descriptor,
                                   deviceObj->d_B));
    
   
    
    cudaEventRecord(*deviceObj->stop);
    #ifdef PROFILING_CLOCK
        cudaDeviceSynchronize(); 
        kernelCLK.end();
    #endif
    // destroy cuDNN
    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
}

void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    #ifdef PROFILING_CLOCK
        d2hCLK.start();
    #endif
    cudaEventRecord(*deviceObj->start_memory_copy_host);
    cudaMemcpy(h_C, deviceObj->d_B, size * sizeof(bench_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(*deviceObj->stop_memory_copy_host);
    #ifdef PROFILING_CLOCK
        d2hCLK.end();
    #endif
    }

    float get_elapsed_time(GraficCommon* device_object, bool csv_format,bool csv_format_timestamp, long int current_time){
        GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
        cudaEventSynchronize(*deviceObj->stop_memory_copy_host);
        float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
        // memory transfer time host-device
        cudaEventElapsedTime(&milliseconds_h_d, *deviceObj->start_memory_copy_device, *deviceObj->stop_memory_copy_device);
        // kernel time
        cudaEventElapsedTime(&milliseconds, *deviceObj->start, *deviceObj->stop);
        //  memory transfer time device-host
        cudaEventElapsedTime(&milliseconds_d_h, *deviceObj->start_memory_copy_host, *deviceObj->stop_memory_copy_host);

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
	cudaError_t err = cudaSuccess;
	err = cudaFree(deviceObj->d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(deviceObj->kernel);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
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