/** * ====================================================================
 * @file        benchmark_library.h (./convolution_2D_bench)
 * @brief       Specific memory structures and function overloads 
 *              for the Convolution 2D benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- Nothing ---


struct GraficObject : public GraficCommon {
	#ifdef CUDA
		// CUDA PART
		bench_t* d_A;
		bench_t* d_B;
		bench_t* kernel;
   	#elif OPENCL
		// OpenCL PART
		cl::Event *evt_copyA;
		cl::Event *evt_copyB;
		cl::Event *evt_copyC;
		cl::Event *evt;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
		cl::Buffer *kernel;
	#elif HIP
		// HIP PART
		bench_t* d_A;
		bench_t* d_B;
		bench_t* kernel;
		hipEvent_t *start_memory_copy_device;
		hipEvent_t *stop_memory_copy_device;
		hipEvent_t *start_memory_copy_host;
		hipEvent_t *stop_memory_copy_host;
		hipEvent_t *start;
		hipEvent_t *stop;
	#elif OPENMP
		// OpenMP part
		bench_t* d_A;
		bench_t* d_B;
		bench_t* kernel;
	#else
		// CPU PART
		bench_t* d_A;
		bench_t* d_B;
		bench_t* kernel;
	#endif
};

// --- Specefic overload of benchmarking function ---
void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w, unsigned int kernel_size);
