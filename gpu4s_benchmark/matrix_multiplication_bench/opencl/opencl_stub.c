#ifdef __cplusplus
extern "C" {
#endif

// ── Platform ────────────────────────
void clGetPlatformIDs() {}
void clGetPlatformInfo() {}

// ── Device ──────────────────────────
void clGetDeviceIDs() {}
void clGetDeviceInfo() {}
void clRetainDevice() {}
void clReleaseDevice() {}

// ── Context ─────────────────────────
void clCreateContextFromType() {}
void clRetainContext() {}
void clReleaseContext() {}
void clGetContextInfo() {}
void clCreateContext() {}

// ── Command Queue ───────────────────
void clCreateCommandQueue() {}
void clCreateCommandQueueWithProperties() {}
void clRetainCommandQueue() {}
void clReleaseCommandQueue() {}
void clGetCommandQueueInfo() {}
void clFlush() {}
void clFinish() {}

// ── Memory ──────────────────────────
void clCreateBuffer() {}
void clCreateSubBuffer() {}
void clRetainMemObject() {}
void clReleaseMemObject() {}
void clGetMemObjectInfo() {}
void clEnqueueReadBuffer() {}
void clEnqueueReadBufferRect() {}
void clEnqueueWriteBuffer() {}
void clEnqueueWriteBufferRect() {}
void clEnqueueCopyBuffer() {}
void clEnqueueCopyBufferRect() {}
void clEnqueueFillBuffer() {}
void clEnqueueMapBuffer() {}
void clEnqueueUnmapMemObject() {}
void clEnqueueMigrateMemObjects() {}

// ── Program ─────────────────────────
void clCreateProgramWithSource() {}
void clCreateProgramWithBinary() {}
void clCreateProgramWithBuiltInKernels() {}
void clRetainProgram() {}
void clReleaseProgram() {}
void clBuildProgram() {}
void clCompileProgram() {}
void clLinkProgram() {}
void clGetProgramInfo() {}
void clGetProgramBuildInfo() {}

// ── Kernel ──────────────────────────
void clCreateKernel() {}
void clCreateKernelsInProgram() {}
void clRetainKernel() {}
void clReleaseKernel() {}
void clSetKernelArg() {}
void clGetKernelInfo() {}
void clGetKernelWorkGroupInfo() {}
void clGetKernelArgInfo() {}
void clEnqueueNDRangeKernel() {}
void clEnqueueTask() {}
void clEnqueueNativeKernel() {}

// ── Events ──────────────────────────
void clCreateUserEvent() {}
void clRetainEvent() {}
void clReleaseEvent() {}
void clSetUserEventStatus() {}
void clWaitForEvents() {}
void clGetEventInfo() {}
void clGetEventProfilingInfo() {}
void clEnqueueMarkerWithWaitList() {}
void clEnqueueBarrierWithWaitList() {}
void clEnqueueMarker() {}
void clEnqueueBarrier() {}
void clEnqueueWaitForEvents() {}

// ── Images  ─────────────────────────
void clCreateImage() {}
void clCreateImage2D() {}
void clCreateImage3D() {}
void clRetainSampler() {}
void clReleaseSampler() {}
void clGetSamplerInfo() {}
void clGetImageInfo() {}
void clEnqueueReadImage() {}
void clEnqueueWriteImage() {}
void clEnqueueCopyImage() {}
void clEnqueueCopyImageToBuffer() {}
void clEnqueueCopyBufferToImage() {}
void clEnqueueMapImage() {}

// ── Misc ────────────────────────────
void clGetSupportedImageFormats() {}
void clGetExtensionFunctionAddress() {}
void clGetExtensionFunctionAddressForPlatform() {}
void clSetEventCallback() {}
void clSetMemObjectDestructorCallback() {}
void clUnloadCompiler() {}
void clUnloadPlatformCompiler() {}

#ifdef __cplusplus
}
#endif