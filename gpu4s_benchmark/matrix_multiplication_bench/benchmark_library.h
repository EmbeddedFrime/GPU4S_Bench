/** * ====================================================================
 * @file        benchmark_library.h (./matrix_multiplication_bench)
 * @brief       Specific memory structures and function overloads 
 *              for the Matrix Multiplication benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- Nothing for now ---
#define NUMBER_BASE 1


struct GraficObject : public GraficCommon {
	#ifdef CUDA
		// CUDA PART
		bench_t* d_A;
		bench_t* d_B;
		bench_t* d_C;
   	#elif OPENCL
   	// OpenCL PART
		cl::Event *evt_copyA;
		cl::Event *evt_copyB;
		cl::Event *evt_copyC;
		cl::Event *evt;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
		cl::Buffer *d_C;
	#elif HIP
		// Hip part 
		bench_t* d_A;
		bench_t* d_B;
		bench_t* d_C;
	#elif OPENMP
		// OpenMP part 
		bench_t* d_A;
		bench_t* d_B;
		bench_t* d_C;
	#else
		// CPU part
		bench_t* d_A;
		bench_t* d_B;
		bench_t* d_C;
	#endif
};

// --- Specefic overload of benchmarking function ---
	// #define UNIFIED_MEMORY
#ifdef UNIFIED_MEMORY
	void device_unified_memory_init_copy(GraficCommon* device_object, bench_t* &A, bench_t* &B, bench_t* &C, unsigned int buff_size, char input_file_A[100],char input_file_B[100]);
	void copy_memory_unified_to_host(GraficCommon* device_object, bench_t* &d_C, unsigned int buff_size);
#endif