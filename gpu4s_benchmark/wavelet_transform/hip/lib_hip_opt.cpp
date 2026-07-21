#include "hip/hip_runtime.h"
#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
//#define BLOCK_SIZE 32


#ifdef INT
#define NUMBERSUBDIVISIONS  4
__global__ void
wavelet_transform_low(const bench_t *A, bench_t *B, const int n, unsigned int iter, unsigned int grid_size){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + ( grid_size *blockDim.x * iter); 

    if (i < size){
        bench_t sum_value_low = 0;
        if(i == 0){
            sum_value_low = A[0] - (int)(- (B[size]/2.0) + (1.0/2.0));
        }
        else
        {
            sum_value_low = A[2*i] - (int)( - (( B[i + size -1] +  B[i + size])/ 4.0) + (1.0/2.0) );
        }
        
        B[i] = sum_value_low;
    }
}
__global__ void
wavelet_transform(const bench_t *A, bench_t *B, const int n, unsigned int iter, unsigned int grid_size){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + ( grid_size *blockDim.x * iter);

    if (i < size){
        bench_t sum_value_high = 0;
        // specific cases
        if(i == 0){
            sum_value_high = A[1] - (int)( ((9.0/16.0) * (A[0] + A[2])) - ((1.0/16.0) * (A[2] + A[4])) + (1.0/2.0));
        }
        else if(i == size -2){
            sum_value_high = A[2*size - 3] - (int)( ((9.0/16.0) * (A[2*size -4] + A[2*size -2])) - ((1.0/16.0) * (A[2*size - 6] + A[2*size - 2])) + (1.0/2.0));
        }
        else if(i == size - 1){
            sum_value_high = A[2*size - 1] - (int)( ((9.0/8.0) * (A[2*size -2])) -  ((1.0/8.0) * (A[2*size - 4])) + (1.0/2.0));
        }
        else{
            // generic case
            sum_value_high = A[2*i+1] - (int)( ((9.0/16.0) * (A[2*i] + A[2*i+2])) - ((1.0/16.0) * (A[2*i - 2] + A[2*i + 4])) + (1.0/2.0));
        }
        
        //store
        B[i+size] = sum_value_high;

        //__syncthreads();
        // low_part
        //for (unsigned int i = 0; i < size; ++i){
        
        //}
    }

}
/*__global__ void
wavelet_transform(const bench_t *A, bench_t *B, const int n){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        bench_t sum_value_high = 0;
        // specific cases
        if(i == 0){
            sum_value_high = A[1] - (int)( ((9.0/16.0) * (A[0] + A[2])) - ((1.0/16.0) * (A[2] + A[4])) + (1.0/2.0));
        }
        else if(i == size -2){
            sum_value_high = A[2*size - 3] - (int)( ((9.0/16.0) * (A[2*size -4] + A[2*size -2])) - ((1.0/16.0) * (A[2*size - 6] + A[2*size - 2])) + (1.0/2.0));
        }
        else if(i == size - 1){
            sum_value_high = A[2*size - 1] - (int)( ((9.0/8.0) * (A[2*size -2])) -  ((1.0/8.0) * (A[2*size - 4])) + (1.0/2.0));
        }
        else{
            // generic case
            sum_value_high = A[2*i+1] - (int)( ((9.0/16.0) * (A[2*i] + A[2*i+2])) - ((1.0/16.0) * (A[2*i - 2] + A[2*i + 4])) + (1.0/2.0));
        }
        
        //store
        B[i+size] = sum_value_high;

        __syncthreads();
        // low_part
        for (unsigned int i = 0; i < size; ++i){
            bench_t sum_value_low = 0;
            if(i == 0){
                sum_value_low = A[0] - (int)(- (B[size]/2.0) + (1.0/2.0));
            }
            else
            {
                sum_value_low = A[2*i] - (int)( - (( B[i + size -1] +  B[i + size])/ 4.0) + (1.0/2.0) );
            }
            
            B[i] = sum_value_low;
        }
    }

}*/
#else
__global__ void
wavelet_transform_high(const bench_t *A, bench_t *B, const int n, const bench_t *lowpass_filter,const bench_t *highpass_filter){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int full_size = size * 2;
	//int hi_start = -(LOWPASSFILTERSIZE / 2);
	//int hi_end = LOWPASSFILTERSIZE / 2;
	//int gi_start = -(HIGHPASSFILTERSIZE / 2 );
    const int gi_end = HIGHPASSFILTERSIZE / 2;
    
    if (i < size){
		bench_t sum_value_high = 0;
        // second process the Highpass filter
        //#pragma unroll
		//for (int gi = gi_start; gi < gi_end + 1; ++gi){
            //int x_position = (2 * i) + gi + 1;
            int x_position = (2 * i) + 1;
            x_position = x_position < 0 ? x_position * -1 : x_position > full_size - 1 ? full_size - 1 - (x_position - (full_size -1 )) : x_position;

            sum_value_high = (highpass_filter[-3 + gi_end] * A[ (x_position - 3) < 0 ? (x_position - 3) * -1 : (x_position - 3) > full_size - 1 ? full_size - 1 - ((x_position- 3) - (full_size -1 )) : (x_position- 3)]) + (highpass_filter[-2 + gi_end] * A[ (x_position - 2) < 0 ? (x_position - 2) * -1 : (x_position - 2) > full_size - 1 ? full_size - 1 - ((x_position- 2) - (full_size -1 )) : (x_position- 2)]) + (highpass_filter[-1 + gi_end] * A[ (x_position - 1) < 0 ? (x_position - 1) * -1 : (x_position - 1) > full_size - 1 ? full_size - 1 - ((x_position- 1) - (full_size -1 )) : (x_position- 1)]) + (highpass_filter[gi_end] * A[ (x_position) < 0 ? (x_position) * -1 : (x_position) > full_size - 1 ? full_size - 1 - ((x_position) - (full_size -1 )) : (x_position)]) + (highpass_filter[1 + gi_end] * A[ (x_position  + 1) < 0 ? (x_position + 1) * -1 : (x_position + 1) > full_size - 1 ? full_size - 1 - ((x_position + 1) - (full_size -1 )) : (x_position + 1)]) + (highpass_filter[2 + gi_end] * A[ (x_position + 2) < 0 ? (x_position + 2) * -1 : (x_position + 2) > full_size - 1 ? full_size - 1 - ((x_position + 2) - (full_size -1 )) : (x_position + 2)]) + (highpass_filter[3 + gi_end] * A[ (x_position + 3) < 0 ? (x_position + 3) * -1 : (x_position + 3) > full_size - 1 ? full_size - 1 - ((x_position + 3) - (full_size -1 )) : (x_position + 3)]);


			//sum_value_high += highpass_filter[gi + gi_end] * A[x_position];
		//}
		// store the value
		B[i+size] = sum_value_high;

    }
}
__global__ void
wavelet_transform(const bench_t *A, bench_t *B, const int n, const bench_t *lowpass_filter,const bench_t *highpass_filter){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int full_size = size * 2;
	//int hi_start = -(LOWPASSFILTERSIZE / 2);
	const int hi_end = LOWPASSFILTERSIZE / 2;
	//int gi_start = -(HIGHPASSFILTERSIZE / 2 );
    //int gi_end = HIGHPASSFILTERSIZE / 2;
    
    if (i < size){
        bench_t sum_value_low = 0;
        int x_position = (2 * i);
        sum_value_low = (lowpass_filter[-4 + hi_end] * A[(x_position -4) < 0 ? (x_position-4) * -1 : (x_position-4) > full_size - 1 ? full_size - 1 - ((x_position-4) - (full_size -1 )) : (x_position-4)]) + (lowpass_filter[-3 + hi_end] * A[(x_position -3) < 0 ? (x_position-3) * -1 : (x_position-3) > full_size - 1 ? full_size - 1 - ((x_position-3) - (full_size -1 )) : (x_position-3)]) + (lowpass_filter[-2 + hi_end] * A[(x_position -2) < 0 ? (x_position-2) * -1 : (x_position-2) > full_size - 1 ? full_size - 1 - ((x_position-2) - (full_size -1 )) : (x_position-2)]) + (lowpass_filter[-1 + hi_end] * A[(x_position -1) < 0 ? (x_position-1) * -1 : (x_position-1) > full_size - 1 ? full_size - 1 - ((x_position-1) - (full_size -1 )) : (x_position-1)]) + (lowpass_filter[hi_end] * A[(x_position) < 0 ? (x_position) * -1 : (x_position) > full_size - 1 ? full_size - 1 - ((x_position) - (full_size -1 )) : (x_position)]) + (lowpass_filter[1 + hi_end] * A[(x_position + 1) < 0 ? (x_position + 1) * -1 : (x_position + 1) > full_size - 1 ? full_size - 1 - ((x_position + 1) - (full_size -1 )) : (x_position+1)]) + (lowpass_filter[+2 + hi_end] * A[(x_position +2) < 0 ? (x_position+2) * -1 : (x_position+2) > full_size - 1 ? full_size - 1 - ((x_position+2) - (full_size -1 )) : (x_position+2)]) + (lowpass_filter[3 + hi_end] * A[(x_position +3) < 0 ? (x_position + 3) * -1 : (x_position + 3) > full_size - 1 ? full_size - 1 - ((x_position+3) - (full_size -1 )) : (x_position+3)]) + (lowpass_filter[4 + hi_end] * A[(x_position +4) < 0 ? (x_position+4) * -1 : (x_position+4) > full_size - 1 ? full_size - 1 - ((x_position+4) - (full_size -1 )) : (x_position+4)]);
		B[i] = sum_value_low;

    }
}
#endif

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
    #ifdef INT
    // if int don't add the allocation
    #else
    // Allocate the device low_filter
    err = hipMalloc((void **)&deviceObj->low_filter, LOWPASSFILTERSIZE * sizeof(bench_t));

    if (err != hipSuccess)
    {
        return false;
    }

    // Allocate the device high_filter
    err = hipMalloc((void **)&deviceObj->high_filter, HIGHPASSFILTERSIZE * sizeof(bench_t));

    if (err != hipSuccess)
    {
        return false;
    }
    #endif

    return true;
}

void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, unsigned int size_a){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    (void)hipEventRecord(*deviceObj->start_memory_copy_device);
	hipError_t err = hipMemcpy(deviceObj->d_A, h_A, sizeof(bench_t) * size_a, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    #ifdef INT
    // if int don't add the copy of the filters
    #else
    err = hipMemcpy(deviceObj->low_filter, lowpass_filter, sizeof(bench_t) * LOWPASSFILTERSIZE, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector lowpass filter from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }

    err = hipMemcpy(deviceObj->high_filter, highpass_filter, sizeof(bench_t) * HIGHPASSFILTERSIZE, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector highpass filter from host to device (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    #endif

    (void)hipEventRecord(*deviceObj->stop_memory_copy_device);
    
}
void execute_kernel(GraficCommon* device_object, unsigned int n){
    
GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    
    (void)hipEventRecord(*deviceObj->start);
    #ifdef INT

    dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n/NUMBERSUBDIVISIONS)/dimBlock.x));
    
    hipStream_t cuda_streams[NUMBERSUBDIVISIONS];
    for (unsigned int streams = 0; streams < NUMBERSUBDIVISIONS; ++streams)
    {
        hipStreamCreate(&cuda_streams[streams]);
    }
    
    for (unsigned int iter = 0; iter < NUMBERSUBDIVISIONS; ++iter)
    {   
        hipLaunchKernelGGL((wavelet_transform), dim3(dimGrid), dim3(dimBlock), 0, cuda_streams[iter], deviceObj->d_A, deviceObj->d_B, n, iter,dimGrid.x);
        hipLaunchKernelGGL((wavelet_transform_low), dim3(dimGrid), dim3(dimBlock), 0, cuda_streams[iter], deviceObj->d_A, deviceObj->d_B, n, iter,dimGrid.x);
    }
    

    #else
    hipStream_t cuda_streams[2];
    dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n)/dimBlock.x));
    for (unsigned int streams = 0; streams < 2; ++streams)
    {
        (void)hipStreamCreate(&cuda_streams[streams]);
    }
    hipLaunchKernelGGL((wavelet_transform), dim3(dimGrid), dim3(dimBlock), 0, cuda_streams[0], deviceObj->d_A, deviceObj->d_B, n, deviceObj->low_filter, deviceObj->high_filter);
    hipLaunchKernelGGL((wavelet_transform_high), dim3(dimGrid), dim3(dimBlock), 0, cuda_streams[1], deviceObj->d_A, deviceObj->d_B, n, deviceObj->low_filter, deviceObj->high_filter);
    #endif
    (void)hipEventRecord(*deviceObj->stop);
}

void copy_memory_to_host(GraficCommon* device_object, bench_t* h_B, int size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    (void)hipEventRecord(*deviceObj->start_memory_copy_host);
    hipMemcpy(h_B, deviceObj->d_B, size * sizeof(bench_t), hipMemcpyDeviceToHost);
    (void)hipEventRecord(*deviceObj->stop_memory_copy_host);
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    (void)hipEventSynchronize(*deviceObj->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    (void)hipEventElapsedTime(&milliseconds_h_d, *deviceObj->start_memory_copy_device, *deviceObj->stop_memory_copy_device);
    // kernel time
    (void)hipEventElapsedTime(&milliseconds, *deviceObj->start, *deviceObj->stop);
    //  memory transfer time device-host
    (void)hipEventElapsedTime(&milliseconds_d_h, *deviceObj->start_memory_copy_host, *deviceObj->stop_memory_copy_host);
    
    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", milliseconds_h_d,milliseconds,milliseconds_d_h,current_time);
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
    err = hipFree(deviceObj->low_filter);
    err = hipFree(deviceObj->high_filter);


    // delete events
    delete deviceObj->start;
    delete deviceObj->stop;
    delete deviceObj->start_memory_copy_device;
    delete deviceObj->stop_memory_copy_device;
    delete deviceObj->start_memory_copy_host;
    delete deviceObj->stop_memory_copy_host;
}
