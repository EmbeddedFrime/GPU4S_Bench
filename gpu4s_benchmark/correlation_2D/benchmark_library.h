/** * ====================================================================
 * @file        benchmark_library.h (./correlation_2D)
 * @brief       Specific memory structures and function overloads 
 *              for the Correlation 2D benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- Core Data Types ---
#ifdef INT
	typedef float result_bench_t;
	static const char type_kernel[] = "typedef int bench_t;\ntypedef float result_bench_t;\n";
#elif FLOAT
	typedef float result_bench_t;
	static const char type_kernel[] = "typedef float bench_t;\ntypedef float result_bench_t;\n";
#elif DOUBLE
	typedef double result_bench_t;
	static const char type_kernel[] = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\ntypedef double bench_t;\ntypedef double result_bench_t;\n";
#endif

struct GraficObject{
	#ifdef CUDA
		// CUDA PART
		bench_t* d_A;
		bench_t* d_B;
		result_bench_t* d_R;
		result_bench_t* mean_A; // axuliar values for the mean of matrix A
		result_bench_t* mean_B; // axuliar values for the mean of matrix B
		result_bench_t* acumulate_value_a_b; // auxiliar values for the acumulation
		result_bench_t* acumulate_value_a_a; // auxiliar values for the acumulation
		result_bench_t* acumulate_value_b_b; // auxiliar values for the acumulation
   	#elif OPENCL
		// OpenCL PART
		cl::Event *evt_copyA;
		cl::Event *evt_copyB;
		cl::Event *evt_copyAB;
		cl::Event *evt_copyAA;
		cl::Event *evt_copyBB;
		cl::Event *evt;
		cl::Event *evt_mean;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
		cl::Buffer *d_R;
		cl::Buffer *mean_A; // axuliar values for the mean of matrix A
		cl::Buffer *mean_B; // axuliar values for the mean of matrix B
		cl::Buffer *acumulate_value_a_b; // auxiliar values for the acumulation
		cl::Buffer *acumulate_value_a_a; // auxiliar values for the acumulation
		cl::Buffer *acumulate_value_b_b; // auxiliar values for the acumulation
	#elif HIP
		// Hip part --
		bench_t* d_A;
		bench_t* d_B;
		result_bench_t* d_R;
		result_bench_t* mean_A; // axuliar values for the mean of matrix A
		result_bench_t* mean_B; // axuliar values for the mean of matrix B
		result_bench_t* acumulate_value_a_b; // auxiliar values for the acumulation
		result_bench_t* acumulate_value_a_a; // auxiliar values for the acumulation
		result_bench_t* acumulate_value_b_b; // auxiliar values for the acumulation
	#elif OPENMP
		// OpenMP part
		bench_t* d_A;
		bench_t* d_B;
		result_bench_t d_R;
		result_bench_t mean_A; // axuliar values for the mean of matrix A
		result_bench_t mean_B; // axuliar values for the mean of matrix B
		result_bench_t acumulate_value_a_b; // auxiliar values for the acumulation
		result_bench_t acumulate_value_a_a; // auxiliar values for the acumulation
		result_bench_t acumulate_value_b_b; // auxiliar values for the acumulation
	#else
		// CPU part
		bench_t* d_A;
		bench_t* d_B;
		result_bench_t d_R;
		result_bench_t mean_A; // axuliar values for the mean of matrix A
		result_bench_t mean_B; // axuliar values for the mean of matrix B
		result_bench_t acumulate_value_a_b; // auxiliar values for the acumulation
		result_bench_t acumulate_value_a_a; // auxiliar values for the acumulation
		result_bench_t acumulate_value_b_b; // auxiliar values for the acumulation
	#endif
};

// --- Specefic overload of benchmarking function ---
bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix);
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b);
void execute_kernel(GraficObject *device_object, unsigned int n);
void copy_memory_to_host(GraficObject *device_object, result_bench_t* h_R);
