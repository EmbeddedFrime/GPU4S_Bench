/** * ====================================================================
 * @file        benchmark_common.h
 * @brief       Shared data structures, macros, and universal helpers 
 *              for hardware acceleration benchmarks.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once

#ifdef OPENCL
// OpenCL lib
#include <CL/opencl.hpp>
//#include <CL/cl.hpp>
#elif CUDA
// CUDA lib
#include <cuda_runtime.h>

#endif


struct GraficObject{
	#ifdef CUDA 
	#ifdef LIB
	bench_cuda_complex* d_B;
	#else
	bench_t* d_B;
	bench_t* d_Br;
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
	cl::Buffer *d_B;
	cl::Buffer *d_Br;
	#elif OPENMP
	// OpenMP part
	bench_t* d_B;
	bench_t* d_Br;
	#elif HIP
	// Hip part --
	bench_t* d_B;
	bench_t* d_Br;
	hipEvent_t *start_memory_copy_device;
	hipEvent_t *stop_memory_copy_device;
	hipEvent_t *start_memory_copy_host;
	hipEvent_t *stop_memory_copy_host;
	hipEvent_t *start;
	hipEvent_t *stop;
	#else
	// CPU part
	bench_t* d_B;
	bench_t* d_Br;
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