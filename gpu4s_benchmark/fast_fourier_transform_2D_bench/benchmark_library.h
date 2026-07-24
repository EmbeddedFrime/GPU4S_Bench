/** * ====================================================================
 * @file        benchmark_library.h (./fast_fourier_transform_2D_bench)
 * @brief       Specific memory structures and function overloads 
 *              for the Fast Fourier Transform 2D benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
 // Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- CUDA Data types  ---
#ifdef CUDA
	// CUDA lib
	#include <cufft.h>
	#ifdef FLOAT
		typedef cufftComplex bench_cuda_complex;
	#elif DOUBLE 
		typedef cufftDoubleComplex bench_cuda_complex;
	#endif
#endif

// --- 2D specific struct  ---
struct COMPLEX{
	bench_t x;
	bench_t y;
};

struct GraficObject : public GraficCommon {
	#ifdef CUDA 
		// CUDA PART
		#ifdef LIB
			bench_cuda_complex* d_A;
			bench_cuda_complex* d_B;
		#else
			bench_t* d_A;
			bench_t* d_B;
		#endif
   	#elif OPENCL
		// OpenCL PART
		cl::Event *evt_copyB;
		cl::Event *evt_copyBr;
		cl::Event *evt;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
	
	#else
		//CPU PART
		COMPLEX** d_A;
		COMPLEX** d_B;
	#endif
};

// --- Specefic overload of benchmarking function ---
bool device_memory_init(GraficCommon* device_object, int64_t size_b_matrix);
void copy_memory_to_device(GraficCommon* device_object, COMPLEX **h_B,int64_t size);
void execute_kernel(GraficCommon* device_object, int64_t n);
void copy_memory_to_host(GraficCommon* device_object, COMPLEX **h_B, int64_t size);
