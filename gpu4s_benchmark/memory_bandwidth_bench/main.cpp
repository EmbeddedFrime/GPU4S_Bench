#include <time.h>
#include "benchmark_library.h"
#include "cpu_functions/cpu_functions.h"
#include <sys/time.h>

#define NUMBER_BASE 1
// OUTPUT C is N x W matrix
// Print hexadecimal values of result 

#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1

#define GPU_FILE "gpu_file.out"
#define CPU_FILE "cpu_file.out"

int arguments_handler(int argc, char ** argv, BenchmarkParameters* arguments_parameters);

int main(int argc, char *argv[])
{
	// random init
	srand (21121993);
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Arguments  
	///////////////////////////////////////////////////////////////////////////////////////////////
	BenchmarkParameters *arguments_parameters = (BenchmarkParameters *)malloc(sizeof(BenchmarkParameters));


	int resolution = arguments_handler(argc,argv,arguments_parameters);
	if (resolution == ERROR_ARGUMENTS)
	{
		exit(-1);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// VARIABLES 
	///////////////////////////////////////////////////////////////////////////////////////////////
	// linearizable versions of matrix
	unsigned int size_matrix =arguments_parameters->size * arguments_parameters->size;
	// A input matrix
	unsigned int size_A = arguments_parameters->size * arguments_parameters->size;
    unsigned int mem_size_A = sizeof(bench_t) * size_A;
	bench_t* A = (bench_t*) malloc(mem_size_A);
	// B output matrix
	unsigned int size_B = arguments_parameters->size * arguments_parameters->size;
    unsigned int mem_size_B = sizeof(bench_t) * size_B;
	bench_t* h_B = (bench_t*) malloc(mem_size_B);
	bench_t* d_B = (bench_t*) malloc(mem_size_B);
	// comparation result
	bool result = false;
	///////////////////////////////////////////////////////////////////////////////////////////////
	// DATA INIT
	///////////////////////////////////////////////////////////////////////////////////////////////
	if (strlen(arguments_parameters->input_file_A) == 0)
	{
	// inicialice A matrix 
		for (int i=0; i<size_A; i++){
	    		#ifdef INT
	        	A[i] = rand() % (NUMBER_BASE * 100);

	        	#else
	        	A[i] = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
	        	#endif
		}
		// iniciate B matrix
		for (int i=0; i<size_B; i++){
			h_B[i] = 0;
			d_B[i] = 0;
		}
		

	}
	else
	{	
		// load data
		get_double_hexadecimal_values(arguments_parameters->input_file_A, A,size_A);

		// iniciate B matrix
		for (int i=0; i<size_B; i++){
			h_B[i] = 0;
			d_B[i] = 0;
		}
	}
	


	///////////////////////////////////////////////////////////////////////////////////////////////
	// CODE BENCKMARK
	///////////////////////////////////////////////////////////////////////////////////////////////

	// base object init
	GraficCommon*mem_bench = (GraficCommon*)malloc(sizeof(GraficObject));
	// init devices
	char device[100] = "";
	init(mem_bench, 0,arguments_parameters->gpu, device);
	if (!arguments_parameters->csv_format_timestamp && !arguments_parameters->csv_format && !arguments_parameters->mute_messages ){
		printf("Using device: %s\n", device);
	}
	
	// Update profiling clock mode
	mem_bench->profiling_clock = arguments_parameters->profiling_clock;

	/ If android and opencl force profiling clock
	#ifdef PROFILING_CLOCK 
		mem_bench->profiling_clock = true;
	#endif


	// init memory
	device_memory_init(mem_bench, size_A , size_B );
	// copy memory to device
	copy_memory_to_device(mem_bench, A, size_A);
	// execute kernel
	execute_kernel(mem_bench, size_A);
	// copy memory to host
	copy_memory_to_host(mem_bench, h_B, size_B);

	// get time
	if (arguments_parameters->print_timing || arguments_parameters->csv_format || arguments_parameters->csv_format_timestamp)
	{
		get_elapsed_time(mem_bench, arguments_parameters->csv_format);
	}
	if (arguments_parameters->print_output)
	{
		#ifdef INT
		for (int i=0; i<size_B; i++){
	    	printf("%d ", d_B[i]); 	
		}
		printf("\n");
		#else
		for (int i=0; i<size_B; i++){
	    	printf("%f ", d_B[i]);
	        	
		}
		printf("\n");
		#endif
	}
	


	if (arguments_parameters->verification)
	{
		Clock cpuKernelCLK;
		cpuKernelCLK.start();
		memcpy(d_B, A, mem_size_A); 
		cpuKernelCLK.end();
		if (arguments_parameters->print_timing)
		{
			printf("CPU Time %.0f milliseconds\n", cpuKernelCLK.getElapsedMS());
		}
		if (arguments_parameters->print_output)
		{
		#ifdef INT
			for (int i=0; i<size_B; i++){
		    	printf("%d ", h_B[i]);  	
			}
	    	printf("\n");
		#else
			for (int i=0; i<size_B; i++){
		    	printf("%f ", h_B[i]);
			}
			printf("\n");
		#endif
		} 
	    result = compare_vectors(A, h_B, size_B);
	    if (result){
	    	printf("OK\n");
	    }
	    if (arguments_parameters->export_results){
	    	print_double_hexadecimal_values(GPU_FILE, h_B, size_B);
	    	print_double_hexadecimal_values(CPU_FILE, A, size_B);
	    }

	}
	if (arguments_parameters->export_results_gpu)
	{
		print_double_hexadecimal_values(GPU_FILE, h_B, size_B);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CLEAN MEMORY
	///////////////////////////////////////////////////////////////////////////////////////////////
	// clean device memory
	clean(mem_bench);
	free(arguments_parameters);
	// free object memory 
	free(mem_bench);
	free(A);
	free(h_B);
	free(d_B);
return 0;
}


// Arguments part
void print_usage(const char * appName)
{
	printf("Usage: %s -s Size [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
	printf(" -s Size : set size of x and y of matrices A and B with Size \n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -d: selects GPU\n");
	printf(" -f: mutes all print\n");
	printf(" -h: print help information\n");
	printf(" -p: clock profilling \n");
}

void init_arguments(BenchmarkParameters* arguments_parameters){
	arguments_parameters->size = 0;
	arguments_parameters->gpu = 0;
	arguments_parameters->verification = false;
	arguments_parameters->export_results = false;
	arguments_parameters->export_results_gpu = false;
	arguments_parameters->print_output = false;
	arguments_parameters->print_timing = false;
	arguments_parameters->csv_format = false;
	arguments_parameters->mute_messages = false;
	arguments_parameters->csv_format_timestamp = false;
	arguments_parameters->profiling_clock = false;
	// --- Properly clear character arrays ---
	arguments_parameters->input_file_A[0] = '\0';
	arguments_parameters->input_file_B[0] = '\0';
	arguments_parameters->output_file[0] = '\0';
}


int arguments_handler(int argc, char ** argv, BenchmarkParameters* arguments_parameters){
	init_arguments(arguments_parameters);
	if (argc == 1){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	} 
	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
			// comon part
			case 'v' : arguments_parameters->verification = true;break;
			case 'e' : arguments_parameters->verification = true; arguments_parameters->export_results= true;break;
			case 'o' : arguments_parameters->print_output = true;break;
			case 't' : arguments_parameters->print_timing = true;break;
			case 'c' : arguments_parameters->csv_format   = true;break;
			case 'C' : arguments_parameters->csv_format_timestamp = true;break;
			case 'g' : arguments_parameters->export_results_gpu   = true;break;
			case 'd' : args +=1; arguments_parameters->gpu = atoi(argv[args]);break;
			case 'f' : arguments_parameters->mute_messages = true;break;
					   args +=1;
					   strcpy(arguments_parameters->output_file,argv[args]);
					   break;
			case 'i' : args +=1;
					strcpy(arguments_parameters->input_file_A,argv[args]);
					args +=1;
					strcpy(arguments_parameters->input_file_B,argv[args]); //TODO FIX with final version of input files
					break;
			case 's' : args +=1; arguments_parameters->size  = atoi(argv[args]);break;
			case 'p' : arguments_parameters->profiling_clock = true;break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	if ( arguments_parameters->size <= 0){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if (arguments_parameters->mute_messages){
		arguments_parameters->csv_format = false;
	}
	return OK_ARGUMENTS;
}
