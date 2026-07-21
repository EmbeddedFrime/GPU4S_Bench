#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

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


bool device_memory_init(GraficCommon* device_object, int64_t size_b_matrix){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    cudaError_t err = cudaSuccess;
    // Allocate the device input vector A
    err = cudaMalloc((void **)&deviceObj->d_A, (size_b_matrix * size_b_matrix) * sizeof(bench_cuda_complex));

    if (err != cudaSuccess)
    {
        return false;
    }
    err = cudaMalloc((void **)&deviceObj->d_B, (size_b_matrix * size_b_matrix) * sizeof(bench_cuda_complex));

    if (err != cudaSuccess)
    {
        return false;
    }
    return true;
}

void copy_memory_to_device(GraficCommon* device_object, COMPLEX **h_B,int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    cudaError_t err = cudaSuccess;
    bench_cuda_complex *h_signal = (bench_cuda_complex *)malloc(sizeof(bench_cuda_complex) * (size * size));
    for (unsigned int i = 0; i < (size); ++i){
         for (unsigned int j = 0; j < (size); ++j){
            h_signal[i * size + j].x = h_B[i][j].x;
            h_signal[i * size + j].y = h_B[i][j].y;
        }
    }

    cudaEventRecord(*deviceObj->start_memory_copy_device);
    err = cudaMemcpy(deviceObj->d_A, h_signal, sizeof(bench_cuda_complex) * (size * size), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    cudaEventRecord(*deviceObj->stop_memory_copy_device);
    
}
void execute_kernel(GraficCommon* device_object, int64_t size){

GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    cufftHandle plan;

    cudaEventRecord(*deviceObj->start);
    
    #ifdef FLOAT
    cufftPlan2d(&plan, size, size, CUFFT_C2C);
    cufftExecC2C(plan, (cufftComplex *)deviceObj->d_A, (cufftComplex *)deviceObj->d_B, CUFFT_FORWARD);
    #else 
    cufftPlan2d(&plan, size, size, CUFFT_Z2Z);
    cufftExecZ2Z(plan, (cufftDoubleComplex *)deviceObj->d_A, (cufftDoubleComplex *)deviceObj->d_B, CUFFT_FORWARD);
    #endif
    
    cudaEventRecord(*deviceObj->stop);
    cufftDestroy(plan);
    
}

void copy_memory_to_host(GraficCommon* device_object, COMPLEX **h_B, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    bench_cuda_complex *h_signal = (bench_cuda_complex *)malloc(sizeof(bench_cuda_complex) * (size*size));
    cudaEventRecord(*deviceObj->start_memory_copy_host);
    cudaMemcpy(h_signal, deviceObj->d_B, (size*size) * sizeof(bench_cuda_complex), cudaMemcpyDeviceToHost);
    cudaEventRecord(*deviceObj->stop_memory_copy_host);
    for (unsigned int i = 0; i < (size); ++i){
         for (unsigned int j = 0; j < (size); ++j){
            h_B[i][j].x = h_signal[i * size + j].x;
            h_B[i][j].y = h_signal[i * size + j].y;
        }
    }
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    cudaEventSynchronize(*deviceObj->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    cudaEventElapsedTime(&milliseconds_h_d, *deviceObj->start_memory_copy_device, *deviceObj->stop_memory_copy_device);
    // kernel time
    cudaEventElapsedTime(&milliseconds, *deviceObj->start, *deviceObj->stop);
    //  memory transfer time device-host
    cudaEventElapsedTime(&milliseconds_d_h, *deviceObj->start_memory_copy_host, *deviceObj->stop_memory_copy_host);
    
    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", milliseconds_h_d,milliseconds,milliseconds_d_h, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", milliseconds_h_d,milliseconds,milliseconds_d_h);
    }else{
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
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(deviceObj->d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
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