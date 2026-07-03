/** * ====================================================================
 * @file        benchmark_library.h (./fast_fourier_transform_2D_bench)
 * @brief       Specific memory structures and function overloads 
 *              for the Fast Fourier Transform 2D benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#include "shared_variables.h"
#include "benchmark_common.h"

#ifdef OPENCL
// OpenCL lib
#include <CL/opencl.hpp>
//#include <CL/cl.hpp>
#elif CUDA
// CUDA lib
#include <cuda_runtime.h>
#include <cufft.h>
#ifdef FLOAT
typedef cufftComplex bench_cuda_complex;
#elif DOUBLE 
typedef cufftDoubleComplex bench_cuda_complex;
#endif
#endif

#ifndef BENCHMARK_H
#define BENCHMARK_H

struct GraficObject{
	#ifdef CUDA 
	// CUDA PART
	#ifdef LIB
	bench_cuda_complex* d_A;
	bench_cuda_complex* d_B;
	#else
	bench_t* d_A;
	bench_t* d_B;
	#endif
	cudaEvent_t *start_memory_copy_device;
	cudaEvent_t *stop_memory_copy_device;
	cudaEvent_t *start_memory_copy_host;
	cudaEvent_t *stop_memory_copy_host;
	cudaEvent_t *start;
	cudaEvent_t *stop;
   	#elif OPENCL
   	// OpenCL PART
	cl::Context *context;
	cl::CommandQueue *queue;
	cl::Device default_device;
	cl::Event *evt_copyB;
	cl::Event *evt_copyBr;
	cl::Event *evt;
	cl::Buffer *d_A;
	cl::Buffer *d_B;
	
	#else
	bench_t* d_A;
	bench_t* d_B;
	
	#endif
	float elapsed_time;
};
bool device_memory_init(GraficObject *device_object, int64_t size_b_matrix);
void copy_memory_to_device(GraficObject *device_object, COMPLEX **h_B,int64_t size);
void execute_kernel(GraficObject *device_object, int64_t n);
void copy_memory_to_host(GraficObject *device_object, COMPLEX **h_B, int64_t size);
#endif
