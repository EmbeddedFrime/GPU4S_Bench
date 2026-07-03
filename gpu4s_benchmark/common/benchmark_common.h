/** * ====================================================================
 * @file        benchmark_common.h
 * @brief       Shared data structures, macros, and universal helpers 
 *              for hardware acceleration benchmarks.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once

// --- Global lib ---
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

// --- Specefic framework lib ---
#ifdef CUDA
    // CUDA lib
    #include <cuda_runtime.h>
#elif OPENCL
    // OpenCL lib
    #include <CL/opencl.hpp>
    // #include <CL/cl.hpp> // OLD framework
#elif HIP
	// HIP part
    #include <hip/hip_runtime.h>
#elif OPENMP
    // OpenMP lib
    #include <omp.h>
#else
    //CPU part
#endif


// ======= Commmon variable =======
// --- Core Data Types ---
#ifdef INT
	#define __ptype "%d"
    typedef int bench_t;
#elif FLOAT
	#define __ptype "%f"
    typedef float bench_t;
#elif DOUBLE
	#define __ptype "%f"
    typedef double bench_t;
#endif

// --- OpenCL Runtime Kernel Code  ---
#ifdef OPENCL
    #ifdef INT
        static const std::string type_kernel = "typedef int bench_t;\n";
    #elif FLOAT
        static const std::string type_kernel = "typedef float bench_t;\n";
    #elif DOUBLE
        static const std::string type_kernel = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\ntypedef double bench_t;\n";
    #endif
#endif



// --- Commmon struct ---
struct GraficObject{
	#ifdef CUDA 
        // --- CUDA Variable ---
        cudaEvent_t *start_memory_copy_device;
        cudaEvent_t *stop_memory_copy_device;
        cudaEvent_t *start_memory_copy_host;
        cudaEvent_t *stop_memory_copy_host;
        cudaEvent_t *start;
        cudaEvent_t *stop;
   	#elif OPENCL
        // --- OpenCL variable ---
        cl::Context *context;
        cl::CommandQueue *queue;
        cl::Device default_device;
	#elif HIP
		// --- Hip variable ---
		hipEvent_t *start_memory_copy_device;
		hipEvent_t *stop_memory_copy_device;
		hipEvent_t *start_memory_copy_host;
		hipEvent_t *stop_memory_copy_host;
		hipEvent_t *start;
		hipEvent_t *stop;
	#elif OPENMP
		// --- OpenMP part----
	#else
		// --- CPU variable ---
	#endif
	float elapsed_time;
};

#ifdef ANDROID
	#include <chrono>

	//Create an class for a shorter call
    // chrono timestamps for kernel timing (CLBlast event profiling unreliable on Android)
	class Clock
	{
	private:
		std::chrono::high_resolution_clock::time_point _timePointA, _timePointB;
	public:
		
		void start(){
			_timePointA = std::chrono::high_resolution_clock::now();
		}

		void end(){
			_timePointB = std::chrono::high_resolution_clock::now();
		}

		float getElapsed(){
			return std::chrono::duration<float, std::milli>(_timePointB - _timePointA).count() * 1000000.0f;
		}
	};
#endif

// --- Standard initialization use by every benchmarks ---
void init(GraficObject *device_object, char* device_name);
void init(GraficObject *device_object, int platform, int device, char* device_name);


// --- Initialization of the memory ---
// Overload for Correlation2D, LRN, max_pooling, memory_bandwidth, relu, softmax, wavelet_transform 
bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix);
// Overload for Convolution2D, FIR, matrix_mult:(naïve/FP16/tensor) 
bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix);

// --- Copy RAM memory to GPU memory ---
// Overload for LRN, max_pooling, memory_bandwidth, relu, softmax, wavelet_transform 
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a);
// Correlation 2D 
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b);

//caution to be merge
// Overload for matrix_mult:(naïve/FP16/tensor) 
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int size_b);
// Overload for Convolution2D, FIR 
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int kernel_size);


// --- Launch the benchmarks ---
//caution to be merge
// Overload for Correlation2D, wavelet_transform 2
void execute_kernel(GraficObject *device_object, unsigned int n);
// Memory bandwidth 1
void execute_kernel(GraficObject *device_object,unsigned int size_a);

// Overload for LRN, matrix_mult:(naïve/FP16/tensor), relu, softmax 6
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w);

// --- Copy back to CPU RAM memory ---
// Overload for Cifar_10, Convolution2D, FIR, LRN, matrix_mult:(naïve/FP16/tensor), max_pooling, memory_bandwidth, relu, softmax, wavelet_transform 
void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size);

// --- return the duration of the benchmark ---
// Overload for matrix_mult:(FP16/tensor), memory_bandwidth 
float get_elapsed_time(GraficObject *device_object, bool csv_format);

// Standard prototype  of get_elapsed_time used by most of the benchmarks :
// Overload for Cifar_10(naïve/mutiple), Convolution2D, Correlation2D, fft:(Naïve/2D/window), FIR, LRN, matrix_mult:(naïve), max_pooling, relu, softmax, wavelet_transform 
float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int timestamp);

// Standard clean prototype used by every the benchmarks
void clean(GraficObject *device_object);
