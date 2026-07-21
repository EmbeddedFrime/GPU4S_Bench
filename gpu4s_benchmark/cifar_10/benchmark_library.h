/** * ====================================================================
 * @file        benchmark_library.h (./cifar_10)
 * @brief       Specific memory structures and function overloads 
 *              for the Cifar 10 benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- Compute ---
const bench_t K = 2;
const bench_t ALPHA = 10e-4;
const bench_t BETA = 0.75;


struct GraficObject : public GraficCommon {
	#ifdef CUDA
		// CUDA PART
		bench_t* input_data;
		bench_t* kernel_1;
		bench_t* conv_1_output;
		bench_t* pooling_1_output;
		bench_t* kernel_2;
		bench_t* conv_2_output;
		bench_t* pooling_2_output;
		bench_t* dense_layer_1_weights;
		bench_t* dense_layer_1_output;
		bench_t* dense_layer_2_weights;
		bench_t* dense_layer_2_output;
		bench_t* output_data;
		bench_t* sum_ouput;
   	#elif OPENCL
		// OpenCL PART
		cl::Event *evt_copyIN;
		cl::Event *evt_copyK1;
		cl::Event *evt_copyK2;
		cl::Event *evt_copyW1;
		cl::Event *evt_copyW2;
		cl::Event *evt_copyOut;
		cl::Event *evt1_1;
		cl::Event *evt1_2;
		cl::Event *evt1_3;
		cl::Event *evt1_4;
		cl::Event *evt2_1;
		cl::Event *evt2_2;
		cl::Event *evt2_3;
		cl::Event *evt2_4;
		cl::Event *evtd_1;
		cl::Event *evtd_1_a;
		cl::Event *evtd_2;
		cl::Event *evtd_2_a;
		cl::Event *evt_softmax;
		cl::Event *evt_softmax_fin;

		cl::Buffer *input_data;
		cl::Buffer *kernel_1;
		cl::Buffer *conv_1_output;
		cl::Buffer *pooling_1_output;
		cl::Buffer *kernel_2;
		cl::Buffer *conv_2_output;
		cl::Buffer *pooling_2_output;
		cl::Buffer *dense_layer_1_weights;
		cl::Buffer *dense_layer_1_output;
		cl::Buffer *dense_layer_2_weights;
		cl::Buffer *dense_layer_2_output;
		cl::Buffer *output_data;
		cl::Buffer *sum_ouput;
	#elif HIP
		//HIP part
		bench_t* input_data;
		bench_t* kernel_1;
		bench_t* conv_1_output;
		bench_t* pooling_1_output;
		bench_t* kernel_2;
		bench_t* conv_2_output;
		bench_t* pooling_2_output;
		bench_t* dense_layer_1_weights;
		bench_t* dense_layer_1_output;
		bench_t* dense_layer_2_weights;
		bench_t* dense_layer_2_output;
		bench_t* output_data;
		bench_t* sum_ouput;
	#elif OPENMP
		// OpenMP part
		bench_t* input_data;
		bench_t* kernel_1;
		bench_t* conv_1_output;
		bench_t* pooling_1_output;
		bench_t* kernel_2;
		bench_t* conv_2_output;
		bench_t* pooling_2_output;
		bench_t* dense_layer_1_weights;
		bench_t* dense_layer_1_output;
		bench_t* dense_layer_2_weights;
		bench_t* dense_layer_2_output;
		bench_t* output_data;
	#else
		// CPU part
		bench_t* input_data;
		bench_t* kernel_1;
		bench_t* conv_1_output;
		bench_t* pooling_1_output;
		bench_t* kernel_2;
		bench_t* conv_2_output;
		bench_t* pooling_2_output;
		bench_t* dense_layer_1_weights;
		bench_t* dense_layer_1_output;
		bench_t* dense_layer_2_weights;
		bench_t* dense_layer_2_output;
		bench_t* output_data;
	#endif
};

// --- Specefic overload of benchmarking function ---
bool device_memory_init(GraficCommon*device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2);
void copy_memory_to_device(GraficCommon*device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size);
void execute_kernel(GraficCommon*device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2);
