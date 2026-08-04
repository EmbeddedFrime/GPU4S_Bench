#include "hip/hip_runtime.h"
#include "../benchmark_library.h"

#ifdef PROFILING_CLOCK
    // kernel time execution
    Clock kernelCLK;
    // host <-> device 
    Clock h2dCLK;
    Clock d2hCLK;
#endif

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
//#define BLOCK_SIZE 32

__global__ void
mean_matrices(const bench_t *A,const bench_t *B,result_bench_t *mean_A ,result_bench_t *mean_B ,const int n)
{   
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int tid_x = threadIdx.x;
    unsigned int tid_y = threadIdx.y;
  
    __shared__ bench_t shared_data_A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ bench_t shared_data_B[BLOCK_SIZE * BLOCK_SIZE];

    if (i < size && j < size){

        shared_data_A[tid_x*blockDim.y + tid_y] = A[i*size + j];
        shared_data_B[tid_x*blockDim.y + tid_y] = B[i*size + j];
    } else {
        shared_data_A[tid_x*blockDim.y + tid_y] = 0;
        shared_data_B[tid_x*blockDim.y + tid_y] = 0;
    }

    // sinc theads
    __syncthreads();

    // --- Reduce Y-axis ---
    for(unsigned int s_y = blockDim.y/2; s_y > 0; s_y >>= 1){
        if (tid_y < s_y)
        {
            shared_data_A[tid_x * blockDim.y + tid_y] += shared_data_A[tid_x * blockDim.y + tid_y + s_y];
            shared_data_B[tid_x * blockDim.y + tid_y] += shared_data_B[tid_x * blockDim.y + tid_y + s_y];
        }
        __syncthreads();
    }

    // --- Reduce X-axis ---
    for(unsigned int s_x = blockDim.x/2; s_x > 0; s_x >>= 1 ){
        if(tid_x < s_x && tid_y == 0)
        {
            shared_data_A[tid_x * blockDim.y] += shared_data_A[(tid_x + s_x) * blockDim.y];
            shared_data_B[tid_x * blockDim.y] += shared_data_B[(tid_x + s_x) * blockDim.y];
        }
        __syncthreads();
        
    }

    if( tid_x == 0 && tid_y == 0)
    { 
        atomicAdd(mean_A, shared_data_A[0]);
        atomicAdd(mean_B, shared_data_B[0]);
    }
}

__global__ void
correlation_2D(const bench_t *A,const bench_t *B, result_bench_t *R, result_bench_t *mean_A ,result_bench_t *mean_B, result_bench_t *acumulate_value_a_b, result_bench_t *acumulate_value_a_a, result_bench_t *acumulate_value_b_b,const int n){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int tid_x = threadIdx.x;
    unsigned int tid_y = threadIdx.y;

    result_bench_t mean_a_matrix =  *mean_A / (n * n);
    result_bench_t mean_b_matrix =  *mean_B / (n * n);
    
    __shared__ bench_t shared_data_A_B[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ bench_t shared_data_A_A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ bench_t shared_data_B_B[BLOCK_SIZE * BLOCK_SIZE];

    if (i < size && j < size){
        result_bench_t result_mean_a = 0;
        result_bench_t result_mean_b = 0;
        result_mean_a = A[i*size+j] - mean_a_matrix;
        result_mean_b = B[i*size+j] - mean_b_matrix;
        shared_data_A_B[tid_x*blockDim.y + tid_y] = result_mean_a * result_mean_b;
        shared_data_A_A[tid_x*blockDim.y + tid_y] = result_mean_a * result_mean_a;
        shared_data_B_B[tid_x*blockDim.y + tid_y] = result_mean_b * result_mean_b;

        // first get the final value  in A (A - mean(a)) and in B (B - mean(b))
        __syncthreads();
        
        for(unsigned int s_y = blockDim.y/2; s_y > 0; s_y >>= 1)
        {
            if (tid_y < s_y)
            {
                shared_data_A_B[tid_x * blockDim.y + tid_y] += shared_data_A_B[tid_x * blockDim.y + tid_y + s_y];
                shared_data_A_A[tid_x * blockDim.y + tid_y] += shared_data_A_A[tid_x * blockDim.y + tid_y + s_y];
                shared_data_B_B[tid_x * blockDim.y + tid_y] += shared_data_B_B[tid_x * blockDim.y + tid_y + s_y];
            }
            __syncthreads();
        }
        for(unsigned int s_x = blockDim.x/2; s_x > 0; s_x >>= 1 )
        {
            if(tid_x < s_x)
            {
                shared_data_A_B[tid_x * blockDim.y] += shared_data_A_B[(tid_x + s_x) * blockDim.y];
                shared_data_A_A[tid_x * blockDim.y] += shared_data_A_A[(tid_x + s_x) * blockDim.y];
                shared_data_B_B[tid_x * blockDim.y] += shared_data_B_B[(tid_x + s_x) * blockDim.y];
            }
            __syncthreads();
            
        }

        if( tid_x == 0 && tid_y == 0)
        { 
            atomicAdd(acumulate_value_a_b, shared_data_A_B[0]);
            atomicAdd(acumulate_value_a_a, shared_data_A_A[0]);
            atomicAdd(acumulate_value_b_b, shared_data_B_B[0]);
        }
 
    }

}

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

bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix){
   GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
   // Allocate the device input vector A
	hipError_t err = hipSuccess;
    err = hipMalloc((void **)&deviceObj->d_A, size_a_matrix * sizeof(bench_t));

    if (err != hipSuccess)
    {
        return false;
    }

    // Allocate the device input vector B
    err = hipMalloc((void **)&deviceObj->d_B, size_b_matrix * sizeof(bench_t));

    if (err != hipSuccess)
    {
        return false;
    }

    // Allocate the device output R value
    err = hipMalloc((void **)&deviceObj->d_R, sizeof(result_bench_t));

    if (err != hipSuccess)
    {
        return false;
    }

    // Allocate the auxiliar values for matrix A and B

    err =  hipMalloc((void **)&deviceObj->mean_A, sizeof(result_bench_t)); 
    if (err != hipSuccess)
    {
        return false;
    }

    err =  hipMalloc((void **)&deviceObj->mean_B, sizeof(result_bench_t)); 
    if (err != hipSuccess)
    {
        return false;
    }

    err =  hipMalloc((void **)&deviceObj->acumulate_value_a_b, sizeof(result_bench_t)); 
    if (err != hipSuccess)
    {
        return false;
    }

    err =  hipMalloc((void **)&deviceObj->acumulate_value_a_a, sizeof(result_bench_t)); 
    if (err != hipSuccess)
    {
        return false;
    }
    err =  hipMalloc((void **)&deviceObj->acumulate_value_b_b, sizeof(result_bench_t)); 
    if (err != hipSuccess)
    {
        return false;
    }

    return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    #ifdef PROFILING_CLOCK
        h2dCLK.start();
    #endif

    (void)hipEventRecord(*deviceObj->start_memory_copy_device);
	hipError_t err = hipMemcpy(deviceObj->d_A, h_A, sizeof(bench_t) * size_a, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipMemcpy(deviceObj->d_B, h_B, sizeof(bench_t) * size_b, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    
    (void)hipEventRecord(*deviceObj->stop_memory_copy_device);

    #ifdef PROFILING_CLOCK
        h2dCLK.end();
    #endif
    
}
void execute_kernel(GraficCommon* device_object, unsigned int n){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n)/dimBlock.x),ceil(float(n)/dimBlock.y));

    #ifdef PROFILING_CLOCK
        kernelCLK.start();
    #endif

    (void)hipEventRecord(*deviceObj->start);
    hipLaunchKernelGGL(mean_matrices, dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->d_A, deviceObj->d_B, deviceObj->mean_A, deviceObj->mean_B , n);
    hipLaunchKernelGGL(correlation_2D, dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->d_A, deviceObj->d_B, deviceObj->d_R, deviceObj->mean_A, deviceObj->mean_B,deviceObj->acumulate_value_a_b, deviceObj->acumulate_value_a_a, deviceObj->acumulate_value_b_b, n);

    (void)hipEventRecord(*deviceObj->stop);

    #ifdef PROFILING_CLOCK
        hipDeviceSynchronize(); 
        kernelCLK.end();
    #endif
}

void copy_memory_to_host(GraficCommon* device_object, result_bench_t* h_R){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    #ifdef PROFILING_CLOCK
        d2hCLK.start();
    #endif

    (void)hipEventRecord(*deviceObj->start_memory_copy_host);
    result_bench_t acumulate_value_a_a;
    result_bench_t acumulate_value_a_b;
    result_bench_t acumulate_value_b_b;
    hipMemcpy(&acumulate_value_a_a, deviceObj->acumulate_value_a_a, sizeof(result_bench_t), hipMemcpyDeviceToHost);
    hipMemcpy(&acumulate_value_a_b, deviceObj->acumulate_value_a_b, sizeof(result_bench_t), hipMemcpyDeviceToHost);
    hipMemcpy(&acumulate_value_b_b, deviceObj->acumulate_value_b_b, sizeof(result_bench_t), hipMemcpyDeviceToHost);
    *h_R = (result_bench_t)(acumulate_value_a_b / (result_bench_t)(sqrt(acumulate_value_a_a * acumulate_value_b_b)));
    //hipMemcpy(h_R, deviceObj->d_R, sizeof(result_bench_t), hipMemcpyDeviceToHost);
    (void)hipEventRecord(*deviceObj->stop_memory_copy_host);

    #ifdef PROFILING_CLOCK
        d2hCLK.end();
    #endif
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format,bool csv_format_timestamp, long int current_time){
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
    err = hipFree(deviceObj->d_A);

    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->d_B);

    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->d_R);

    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device R (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    // delete auxiliars
    err = hipFree(deviceObj->mean_A);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device  mean_A (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    err = hipFree( deviceObj->mean_B);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device mean_B (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipFree(deviceObj->acumulate_value_a_b);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device acumulate_value_a_b (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    err = hipFree(deviceObj->acumulate_value_a_a);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device acumulate_value_a_a (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    err = hipFree(deviceObj->acumulate_value_b_b);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device acumulate_value_b_b (error code %s)!\n", hipGetErrorString(err));
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
