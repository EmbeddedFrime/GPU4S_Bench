#ifdef __cplusplus
extern "C" {
#endif

// Core OpenCL functions
void clGetPlatformIDs() {}
void clGetDeviceIDs() {}
void clGetDeviceInfo() {}
void clCreateContext() {}
void clCreateCommandQueue() {}
void clCreateCommandQueueWithProperties() {}
void clCreateBuffer() {}
void clCreateProgramWithSource() {}
void clBuildProgram() {}
void clGetProgramBuildInfo() {}
void clCreateKernel() {}
void clSetKernelArg() {}
void clEnqueueNDRangeKernel() {}
void clEnqueueReadBuffer() {}
void clEnqueueWriteBuffer() {}
void clFinish() {}
void clFlush() {}
void clReleaseMemObject() {}
void clReleaseKernel() {}
void clReleaseProgram() {}
void clReleaseCommandQueue() {}
void clReleaseContext() {}

// Device and event lifecycle functions requested by your benchmark
void clRetainDevice() {}
void clReleaseDevice() {}
void clReleaseEvent() {}
void clWaitForEvents() {}
void clGetEventProfilingInfo() {}

#ifdef __cplusplus
}
#endif