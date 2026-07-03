/** * ====================================================================
 * @file        benchmark_library.h (./max_pooling_bench)
 * @brief       Specific memory structures and function overloads 
 *              for the Max Pooling benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- Nothing for now --

struct GraficObject{
	#ifdef CUDA
		bench_t* d_A;
		bench_t* d_B;

   	#elif OPENCL
		// OpenCL PART
		cl::Event *evt_copyA;
		cl::Event *evt_copyB;
		cl::Event *evt;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
	#elif HIP
		// Hip part 
		bench_t* d_A;
		bench_t* d_B;
	#elif OPENMP
		// OpenMP part
		bench_t* d_A;
		bench_t* d_B;
	#else
		// CPU PART
		bench_t* d_A;
		bench_t* d_B;
	#endif
};

// --- Specefic overload of benchmarking function ---
bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix);
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a);
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w, unsigned int stride, unsigned int size_lateral);
void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size);
