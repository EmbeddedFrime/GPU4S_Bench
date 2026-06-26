# ====== CMake Configuration ======

# --- Global Variable ---
set(OPENCL_VERSION  300       CACHE STRING "API OpenCL Version : 200, 210, 220, 300, 310")
set(ENDIANFLAGS    ""         CACHE STRING "ENDIANFLAGS : , BIGENDIAN")
set(OPT_FLAG       "-O3"      CACHE STRING "Compiler optimization level : -O2, -O3, -Ofast")
set(CUDA_ARCH      "native"   CACHE STRING "API CUDA version: native, sm_72-86")
set(BLA_VENDOR     "OpenBLAS" CACHE STRING "BLAS lib : ATLAS, OpenBLAS")


if(NOT BLOCKSIZE)
    string(ASCII 27 Esc)
    message(STATUS "${Esc}[1;34mNote: BLOCKSIZE is empty. Fallback to default: 16${Esc}[0m")
    set(BLOCKSIZE       16        CACHE STRING "Block size for tiled kernels")
endif()

if(NOT DATATYPE)
    message(STATUS "${Esc}[1;34mNote: DATATYPE is empty. Fallback to default: FLOAT${Esc}[0m")
    set(DATATYPE       "FLOAT"    CACHE STRING "Data type: FLOAT, DOUBLE, INT")
endif()


# --- Android Variable ---
if(ANDROID)
    set(ANDROID_INC ${CMAKE_SOURCE_DIR}/../common/android/include/)
    set(ANDROID_LIB ${CMAKE_SOURCE_DIR}/../common/android/libs/) 
endif(ANDROID)


# --- Define Output directory ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

# --- Global Settings ---
set(CMAKE_CXX_STANDARD 17)    # force clang 17 or g++ 17 compiler
set(CMAKE_BUILD_TYPE Release) # add production-level optimization
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # set -fPIC flag
add_compile_options(${OPT_FLAG})



# ====== Package/language Handler ======

if(NOT ANDROID)
    # Searching for package installation
    find_package(OpenCL QUIET) 
    find_package(CLBlast QUIET) 
    find_package(OpenMP QUIET)

    # --- OpenBLAS & ATLAS ---
    find_package(BLAS QUIET) 
    if(BLAS_FOUND)
        find_path(BLAS_INCLUDE_DIRS 
            NAMES cblas.h cblas_atlas.h
            PATH_SUFFIXES openblas blas atlas
        )
    endif()

    # --- CUDA ---
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND AND CUDAToolkit_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
        enable_language(CUDA)
    endif()

    # --- HIP ---
    find_package(hip QUIET)
    if(hip_FOUND)
        enable_language(HIP)
    endif()

endif()

# ====== User warning ======
if(NOT ANDROID)

    if(NOT OpenCL_FOUND)
        message(WARNING "OpenCL installation was not found on this system. Skipping opencl targets.")
    endif()

    if(NOT CLBlast_FOUND)
        message(WARNING "CLBlast installation was not found on this system. Skipping OpenCL-lib target.")
    endif()

    if(NOT OpenMP_CXX_FOUND)
        message(WARNING "OpenMP installation was not found on this system. Skipping OpenCL-lib target.")
    endif()
    # --- ATLAS OR OPENBLAS
    if(NOT BLAS_INCLUDE_DIRS)
        message(WARNING "${BLA_VENDOR} installation was not found on this system. Skipping OpenMP-lib target.")
    endif()

    if(NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA installation was not found on this system. Skipping CUDA targets.")
    endif()

    if(NOT hip_FOUND)
        message(WARNING "HIP installation was not found on this system. Skipping HIP targets.")
    endif()

else()
    include(FetchContent)

    # --- Download clblast Headers ---
    if(NOT EXISTS "${ANDROID_INC}/clblast.h")
        message(STATUS "Downloading 1.7.0 clblast.h header...")
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/CNugteren/CLBlast/1.7.0/include/clblast.h"
            "${ANDROID_INC}/clblast.h"
            SHOW_PROGRESS
        )
    endif()

    # --- Download OpenCL Headers ---
    if(NOT EXISTS "${ANDROID_INC}/CL/opencl.hpp")
        message(STATUS "Downloading v2026.05.29 OpenCL C++ headers...")
        
        # Download Khronos OpenCL C headers
        FetchContent_Declare(opencl_headers
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
            GIT_TAG        v2026.05.29
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(opencl_headers)
        
        # cp the CL directory into ${ANDROID_INC}
        file(COPY "${opencl_headers_SOURCE_DIR}/CL" DESTINATION "${ANDROID_INC}")

        # Download the modern C++ wrapper (opencl.hpp)
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/KhronosGroup/OpenCL-CLHPP/v2026.05.29/include/CL/opencl.hpp"
            "${ANDROID_INC}/CL/opencl.hpp"
            SHOW_PROGRESS
        )
    endif()

    # --- Check de depandancy files ---
    if(EXISTS ${ANDROID_LIB}${ANDROID_ABI}/libOpenCL.so)
            set(ANDROID_OPENCL_LIB_INC TRUE)
    else()
        set(ANDROID_OPENCL_LIB_INC FALSE)
        message(WARNING "Opencl android libs file was not found. Skipping OpenCL targets. See README.md")
    endif()

    if(EXISTS ${ANDROID_LIB}${ANDROID_ABI}/libclblast.a)
        set(ANDROID_CLBLAST_LIB_INC TRUE)
    else()
        set(ANDROID_CLBLAST_LIB_INC FALSE)
        message(WARNING "clblast android libs file was not found. Skipping OpenCL-lib target. See README.md")
    endif()

    if(EXISTS ${ANDROID_LIB}${ANDROID_ABI}/libopenblas.a
    AND EXISTS ${ANDROID_INC}${ANDROID_ABI}/openblas_config.h
    AND BLA_VENDOR STREQUAL "OpenBLAS")
        set(ANDROID_OPENBLAS_LIB_INC TRUE)
    else()
        set(ANDROID_OPENBLAS_LIB_INC FALSE)
        message(WARNING "Openblast android libs/headers files was not found. Skipping OpenMP-lib target. See README.md")
    endif()    
endif()