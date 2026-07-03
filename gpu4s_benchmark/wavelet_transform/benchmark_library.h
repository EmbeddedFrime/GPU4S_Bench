/** * ====================================================================
 * @file        benchmark_library.h (./wavelet_transform)
 * @brief       Specific memory structures and function overloads 
 *              for the Wavelet Transform benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"


#define HIGHPASSFILTERSIZE 7
#define LOWPASSFILTERSIZE 9

// ======= Benchmark local variable =======
// --- typedef and compute --
#ifdef INT
	static const std::string type_kernel = "typedef int bench_t;\n#define HIGHPASSFILTERSIZE 7\n#define LOWPASSFILTERSIZE 9\n";
	static const bench_t lowpass_filter[LOWPASSFILTERSIZE] = {1,1,1,1,1,1,1,1,1};
	static const bench_t highpass_filter[HIGHPASSFILTERSIZE] = {1,1,1,1,1,1,1};
#elif FLOAT
	static const std::string type_kernel = "typedef float bench_t;\n#define HIGHPASSFILTERSIZE 7\n#define LOWPASSFILTERSIZE 9\n";
	static const bench_t lowpass_filter[LOWPASSFILTERSIZE] = {0.037828455507,-0.023849465020,-0.110624404418,0.377402855613, 0.852698679009,0.377402855613, -0.110624404418,-0.023849465020, 0.037828455507};
	static const bench_t highpass_filter[HIGHPASSFILTERSIZE] = {-0.064538882629, 0.040689417609, 0.418092273222,-0.788485616406,0.418092273222,0.040689417609,-0.064538882629};
#elif DOUBLE
	static const std::string type_kernel = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\ntypedef double bench_t;\n#define HIGHPASSFILTERSIZE 7\n#define LOWPASSFILTERSIZE 9\n";
	static const bench_t lowpass_filter[LOWPASSFILTERSIZE] = {0.037828455507,-0.023849465020,-0.110624404418,0.377402855613, 0.852698679009,0.377402855613, -0.110624404418,-0.023849465020, 0.037828455507};
	static const bench_t highpass_filter[HIGHPASSFILTERSIZE] = {-0.064538882629, 0.040689417609, 0.418092273222,-0.788485616406,0.418092273222,0.040689417609,-0.064538882629};
#endif


struct GraficObject{
	#ifdef CUDA
		// CUDA PART
		bench_t* d_A;
		bench_t* d_B;
		bench_t* low_filter;
		bench_t* high_filter;

   	#elif OPENCL
		// OpenCL PART
		cl::Event *evt_copyA;
		cl::Event *evt_copyB;
		cl::Event *evt_copyC;
		cl::Event *evt;
		cl::Event *evt_int;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
		cl::Buffer *low_filter;
		cl::Buffer *high_filter;
	#elif HIP
		// Hip part 
		bench_t* d_A;
		bench_t* d_B;
		bench_t* low_filter;
		bench_t* high_filter;
	#elif OPENMP
		// OpenMP part
		bench_t* d_A;
		bench_t* d_B;
		bench_t* low_filter;
		bench_t* high_filter;
	#else
		// CPU part
		bench_t* d_A;
		bench_t* d_B;
		bench_t* low_filter;
		bench_t* high_filter;
	#endif
};

// --- Specefic overload of benchmarking function ---
bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix);
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a);
void execute_kernel(GraficObject *device_object, unsigned int n);
void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size);
