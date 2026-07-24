#include "../benchmark_library.h"
#include "math.h"

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
//#define BLOCK_SIZE 1024
#define BLOCK_SIZE_PLANE (BLOCK_SIZE * BLOCK_SIZE)
__global__ void
max_pooling_kernel(const bench_t *A, bench_t *B, const int size, const unsigned int stride,  const unsigned int lateral_stride)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    //unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
   
    if (i  < lateral_stride*lateral_stride){
        
        bench_t max_value = A[(((i%lateral_stride) * stride )+ ((i/lateral_stride)*size * stride)) ];
        for(unsigned int x = 0; x < stride; ++x)
        {
            for(unsigned int y = 0; y < stride; ++y)
            {
                //unsigned int position_array = ((((i%lateral_stride) * stride )+ ((i/lateral_stride)*size * stride)) + x)  + ( y * size);
                //printf("max %f,value %f, pos x %d, pos y %d i position %d, final position %d\n", max_value,  A[position_array], x ,y, i, position_array);
                max_value = 
                // --- FIX: use the correct max function depending one the type ---
                #ifdef INT
                    max_value = max(max_value, A[((((i%lateral_stride) * stride )+ ((i/lateral_stride)*size * stride)) + x)  + ( y * size)]);
                #elif FLOAT
                    max_value = fmaxf(max_value, A[((((i%lateral_stride) * stride )+ ((i/lateral_stride)*size * stride)) + x)  + ( y * size)]);
                #elif DOUBLE
                     max_value = fmax(max_value, A[((((i%lateral_stride) * stride )+ ((i/lateral_stride)*size * stride)) + x)  + ( y * size)]);
                #endif     
                
            }
        }
        //printf("i position %d value %f \n", i, max_value);
        B[i] = max_value;
    }
}

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

bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix){

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
    return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, unsigned int size_a){
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

    cudaEventRecord(*deviceObj->stop_memory_copy_device);   

    #ifdef PROFILING_CLOCK
        h2dCLK.end();
    #endif
}
void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int stride, unsigned int lateral_stride){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock, dimGrid;
    if(lateral_stride < BLOCK_SIZE_PLANE)
    {
        dimBlock = dim3(lateral_stride*lateral_stride);
        dimGrid = dim3(1);
    }
    else
    {
        dimBlock = dim3(BLOCK_SIZE_PLANE);
        dimGrid = dim3(ceil((lateral_stride*lateral_stride)/dimBlock.x));
    }

    
    

    #ifdef PROFILING_CLOCK
        kernelCLK.start();
    #endif

    cudaEventRecord(*deviceObj->start);
    max_pooling_kernel<<<dimGrid, dimBlock>>>(deviceObj->d_A, deviceObj->d_B, n, stride, lateral_stride);
    cudaEventRecord(*deviceObj->stop);

    #ifdef PROFILING_CLOCK
        cudaDeviceSynchronize(); 
        kernelCLK.end();
    #endif
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
        printf("%.10f;%.10f;%.10f;%ld;\n", milliseconds_h_d,milliseconds,milliseconds_d_h,current_time);
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

    // delete events
    delete deviceObj->start;
    delete deviceObj->stop;
    delete deviceObj->start_memory_copy_device;
    delete deviceObj->stop_memory_copy_device;
    delete deviceObj->start_memory_copy_host;
    delete deviceObj->stop_memory_copy_host;
}
