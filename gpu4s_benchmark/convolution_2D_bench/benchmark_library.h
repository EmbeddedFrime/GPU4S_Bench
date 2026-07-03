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


struct GraficObject{
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
bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix);
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int kernel_size);
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w, unsigned int kernel_size);
void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size);
