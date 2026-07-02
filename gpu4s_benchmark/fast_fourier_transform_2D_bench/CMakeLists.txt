# =======================================================================
# File:         CMakeLists.txt (./fast_fourier_transform_2D_bench)
# Description:  Build targets for 2D Fast Fourier Transform benchmark
# Target:       OpenCL, CUDA
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================

cmake_minimum_required(VERSION 3.24)
project(FFT_2D CXX) 

# ====== Include cmake module ======
list(APPEND CMAKE_MODULE_PATH 
    "${CMAKE_SOURCE_DIR}/../common/cmake/"
    "${CMAKE_SOURCE_DIR}/../common/cmake/module"
)
# Global:
include(setup)
include(compileBlueprint)

# ====== Compilation of the targets ======

# --- Android Targets---
if(ANDROID)

    if(ANDROID_OPENCL_LIB_INC)
        # --- OpenCL-lib ---
        compile_target(${PROJECT_NAME}_opencl_lib
            BENCH_DIR          ${CMAKE_SOURCE_DIR}
            SOURCES_FILES      opencl/lib_opencl_lib.cpp
            COMPILE_DEFS       OPENCL   
                               CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION} 

            INCLUDES           ${ANDROID_INC}
            LIBRARIES          ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so
                               

            SHORTCUTS_NAMES    opencl-lib OpenCL-lib
        )
    endif()


endif(ANDROID)

# --- Computeur targets ---
if(NOT ANDROID)
    # --- OpenCL Targets ---
    if(OpenCL_FOUND)
        # --- OpenCL-lib ---
        compile_target(${PROJECT_NAME}_opencl_lib
            BENCH_DIR          ${CMAKE_SOURCE_DIR}
            SOURCES_FILES      opencl/lib_opencl_lib.cpp
            COMPILE_DEFS       OPENCL
                               CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}

            LIBRARIES          OpenCL::OpenCL 
            INCLUDES           ${ANDROID_INC}
            SHORTCUTS_NAMES    opencl-lib OpenCL-lib
        )
    endif()

    # --- CUDA Targets---
    if(CUDAToolkit_FOUND)
        # --- CUDA-lib ---
        compile_target(${PROJECT_NAME}_cuda_lib
            BENCH_DIR          ${CMAKE_SOURCE_DIR}
            SOURCES_FILES      cuda/lib_cuda_lib.cu
            COMPILE_DEFS       CUDA
            SET_CUDA           1
            LIBRARIES          CUDA::cudart CUDA::cufft fftw3
            SHORTCUTS_NAMES    cuda-lib CUDA-lib
        )
    endif()

endif(NOT ANDROID)


# ====== Shorcuts ======

# --- all-opencl (Android) ---
if(ANDROID AND ANDROID_OPENCL_LIB_INC )
    add_custom_target(all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl_lib
    )
endif()

# --- all-opencl---
if(NOT ANDROID AND OpenCL_FOUND )
    add_custom_target(all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl_lib
    )
endif()


# --- all-cuda---
if(NOT ANDROID AND CUDAToolkit_FOUND)
    add_custom_target(all-cuda DEPENDS 
        ${PROJECT_NAME}_cuda_lib
    )
endif()