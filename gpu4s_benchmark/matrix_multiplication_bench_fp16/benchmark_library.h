/** * ====================================================================
 * @file        benchmark_library.h (./matrix_multiplication_bench_fp16)
 * @brief       Specific memory structures and function overloads 
 *              for the Matrix Multiplication FP16 benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#ifndef BENCHMARK_H
#define BENCHMARK_H

// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- Core Data Types ---
#ifdef INT
	typedef int bench_t_gpu;
#elif FLOAT16
	typedef float bench_t; //not in benchmark_common.h
#elif FLOAT
	typedef float bench_t_gpu;
#elif DOUBLE
	typedef double bench_t_gpu;
#endif

// --- OpenCL Runtime Kernel Code  ---
#ifdef OPENCL
	#ifdef FLOAT16
		// OpenCL float16 lib
		static const std::string type_kernel = 
			"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
			"typedef half bench_t_gpu;\n";
    #endif
#endif

// --- CUDA Runtime lib Code  ---
#ifdef CUDA
	#ifdef FLOAT16
		// CUDA float16 lib
		#include <cuda_fp16.h>
		typedef half bench_t_gpu;
    #endif
#endif

struct GraficObject{
	#ifdef CUDA
		// CUDA PART
		cl::Event *evt_copyA;
		cl::Event *evt_copyB;
		cl::Event *evt_copyC;
		cl::Event *evt;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
		cl::Buffer *d_C;
	#elif OPENCL
		// OpenCL PART
		bench_t* d_A;
		bench_t* d_B;
		bench_t_gpu* d_half_A;
		bench_t_gpu* d_half_B;
		bench_t_gpu* d_half_C;
		bench_t* d_C;
	#else
		//CPU PART
	#endif

};

// --- Specefic overload of benchmarking function ---
bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix);
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int size_b);
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w);
void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size);

#endif
