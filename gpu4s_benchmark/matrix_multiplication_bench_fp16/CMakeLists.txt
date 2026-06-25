cmake_minimum_required(VERSION 3.24)
project(matrix_mult_fp16 CXX) 


# ====== CMake Configuration ======

# --- Global Variable ---
set(OPENCL_VERSION  300       CACHE STRING "API OpenCL Version : 200, 210, 220, 300, 310")
set(ENDIANFLAGS    ""         CACHE STRING "ENDIANFLAGS : , BIGENDIAN")
set(OPT_FLAG       "-O3"      CACHE STRING "Compiler optimization level : -O2, -O3, -Ofast")
set(CUDA_ARCH      "native"   CACHE STRING "API CUDA version: native, sm_72-86")
set(BLA_VENDOR     "OpenBLAS" CACHE STRING "BLAS lib : ATLAS, OpenBLAS")


string(ASCII 27 Esc)
if(NOT BLOCKSIZE)
    message(STATUS "${Esc}[1;34mNote: BLOCKSIZE is empty. Fallback to default: 16${Esc}[0m")
    set(BLOCKSIZE       16        CACHE STRING "Block size for tiled kernels")
endif()

if(NOT DATATYPE)
    message(STATUS "${Esc}[1;34mNote: DATATYPE is empty. Fallback to default: FLOAT${Esc}[0m")
    set(DATATYPE       "FLOAT"    CACHE STRING "Data type: FLOAT, DOUBLE, INT, DOUBLE")
endif()

if(NOT DATATYPEGPU)
    message(STATUS "${Esc}[1;34mNote: DATATYPEGPU is empty. Fallback to default: FLOAT16${Esc}[0m")
    set(DATATYPEGPU       "FLOAT16"    CACHE STRING "Data type: FLOAT, FLOAT16, INT, DOUBLE")
endif()


# --- Android Variable ---
if(ANDROID)
    set(ANDROID_INC ${CMAKE_SOURCE_DIR}/android/include/)
    set(ANDROID_LIB ${CMAKE_SOURCE_DIR}/android/libs/) 
endif(ANDROID)


# --- Define Output directory ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

# --- Global Settings ---
set(CMAKE_CXX_STANDARD 17)    # force clang 17 or g++ 17 compiler
set(CMAKE_BUILD_TYPE Release) # add production-level optimization
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # set -fPIC flag
add_compile_options(${OPT_FLAG})  # -O3 optimisation flag



# ====== Package/language Handler ======

if(NOT ANDROID)
    # Searching for package installation
    find_package(OpenCL QUIET) 
    find_package(CLBlast QUIET) 
    find_package(OpenMP QUIET)

    # --- CUDA ---
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND AND CUDAToolkit_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
        enable_language(CUDA)
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

    if(NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA installation was not found on this system. Skipping CUDA targets.")
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

endif()

# ====== Compilation of Targets ======

# --- Blueprint function ---
#  Compile all the targets
# --------------------------
function(compile_target TARGET_NAME)

# Parse arguments
set(singleArgs  SET_CUDA) 
set(multipleArgs SOURCES_FILES SET_HIP_FILES COMPILE_DEFS SHORTCUTS_NAMES LIBRARIES INCLUDES COMPILE_OPTIONS) 

cmake_parse_arguments(ARG "" "${singleArgs}" "${multipleArgs}" ${ARGN})

    # --- Add all source file ---
    add_executable(${TARGET_NAME}
        main.cpp
        ${ARG_SOURCES_FILES}
        cpu/lib_cpu.cpp
    )

    # --- Set specefic files as HIP API files ---
    if(ARG_SET_HIP_FILES)
        foreach(HIP_FILE IN LISTS ARG_SET_HIP_FILES)
            set_source_files_properties(${HIP_FILE} PROPERTIES LANGUAGE HIP)
        endforeach()
    endif()

    # --- Add compile define option ---
    target_compile_definitions(${TARGET_NAME} PRIVATE
        ${DATATYPE}
        BLOCK_SIZE=${BLOCKSIZE}
        ${ARG_COMPILE_DEFS}
        ${ENDIANFLAGS}    
    )

    # --- Set specific compiler flags ---
    if(ARG_COMPILE_OPTIONS)
        target_compile_options(${TARGET_NAME} PRIVATE ${ARG_COMPILE_OPTIONS})
    endif()

    # --- Set target CUDA ARCHITECTURES ---
    if(ARG_SET_CUDA)
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
    endif()
    
    # --- Includes header directories ---
    if(ARG_INCLUDES)
        target_include_directories(${TARGET_NAME} PRIVATE ${ARG_INCLUDES})
    endif()
    
    # --- Link required libraries ---
    if(ARG_LIBRARIES)
        target_link_libraries(${TARGET_NAME} PRIVATE ${ARG_LIBRARIES}) 
    endif()

    # --- Create custom shortcut targets ---
    foreach(SHORTCUT IN LISTS ARG_SHORTCUTS_NAMES)
        add_custom_target(${SHORTCUT} DEPENDS ${TARGET_NAME})
    endforeach()

endfunction()

# --- Android Targets---
if(ANDROID)

    if(ANDROID_OPENCL_LIB_INC)
    # --- OpenCL ---
    compile_target(${PROJECT_NAME}_opencl
        SOURCES_FILES      opencl/lib_opencl.cpp
        COMPILE_DEFS       OPENCL
                           CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
                             
        INCLUDES           ${ANDROID_INC}
        LIBRARIES          ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so
        SHORTCUTS_NAMES    cl OpenCL
    )
    # --- OpenCL-opt ---
    compile_target(${PROJECT_NAME}_opencl_opt
        SOURCES_FILES      opencl/lib_opencl_opt.cpp
        COMPILE_DEFS       OPENCL   
                           CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION} 

        INCLUDES           ${ANDROID_INC}
        LIBRARIES          ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so
        SHORTCUTS_NAMES    opencl-opt OpenCL-opt
    )
    endif()

    if(ANDROID_OPENCL_LIB_INC AND ANDROID_CLBLAST_LIB_INC)
    # --- OpenCL-lib ---
    compile_target(${PROJECT_NAME}_opencl_lib
        SOURCES_FILES      opencl/lib_opencl_lib.cpp
        COMPILE_DEFS       OPENCL   
                           CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION} 

        INCLUDES           ${ANDROID_INC}
        LIBRARIES          ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so
                           ${ANDROID_LIB}/${ANDROID_ABI}/libclblast.a

        SHORTCUTS_NAMES    opencl-lib OpenCL-lib
    )
    endif()

endif(ANDROID)

# --- Computeur targets ---
if(NOT ANDROID)
    # --- OpenCL Targets ---
    if(OpenCL_FOUND)
        # --- OpenCL ---
        compile_target(${PROJECT_NAME}_opencl
            SOURCES_FILES      opencl/lib_opencl.cpp
            COMPILE_DEFS       OPENCL
                               CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}

            LIBRARIES          OpenCL::OpenCL
            SHORTCUTS_NAMES    cl OpenCL
        )

        # --- OpenCL-opt ---
        compile_target(${PROJECT_NAME}_opencl_opt
            SOURCES_FILES      opencl/lib_opencl_opt.cpp
            COMPILE_DEFS       OPENCL
                               CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}

            LIBRARIES          OpenCL::OpenCL
            SHORTCUTS_NAMES    opencl-opt OpenCL-opt
        )

        # --- OpenCL-lib ---
        if(CLBlast_FOUND)
            compile_target(${PROJECT_NAME}_opencl_lib
                SOURCES_FILES      opencl/lib_opencl_lib.cpp
                COMPILE_DEFS       OPENCL
                                   CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}

                LIBRARIES          OpenCL::OpenCL clblast
                SHORTCUTS_NAMES    opencl-lib OpenCL-lib
            )
        endif()        
    endif()


    # --- CUDA Targets---
    if(CUDAToolkit_FOUND)
        # --- CUDA ---
        compile_target(${PROJECT_NAME}_cuda
            SOURCES_FILES      cuda/lib_cuda.cu
            COMPILE_DEFS       CUDA 
            SET_CUDA           1
            LIBRARIES          CUDA::cudart 
            SHORTCUTS_NAMES    cuda CUDA
        )

        # --- CUDA-opt ---
        compile_target(${PROJECT_NAME}_cuda_opt
            SOURCES_FILES      cuda/lib_cuda_opt.cu
            COMPILE_DEFS       CUDA ${DATATYPEGPU}
            SET_CUDA           1
            LIBRARIES          CUDA::cuda_driver CUDA::cudart 
            SHORTCUTS_NAMES    cuda-opt CUDA-opt
        )

        # --- CUDA-lib ---
        compile_target(${PROJECT_NAME}_cuda_lib
            SOURCES_FILES      cuda/lib_cuda_lib.cu
            COMPILE_DEFS       CUDA ${DATATYPEGPU}
            SET_CUDA           1
            LIBRARIES          CUDA::cudart CUDA::cublas
            SHORTCUTS_NAMES    cuda-lib CUDA-lib
        )
    endif()

endif(NOT ANDROID)


# ====== Shorcuts ======

# --- all-opencl (Android) ---
if(ANDROID AND ANDROID_OPENCL_LIB_INC AND ANDROID_CLBLAST_LIB_INC )
    add_custom_target(all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl
        ${PROJECT_NAME}_opencl_opt 
        ${PROJECT_NAME}_opencl_lib
    )
endif()

# --- all-opencl---
if(NOT ANDROID AND OpenCL_FOUND AND CLBlast_FOUND )
    add_custom_target(all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl
        ${PROJECT_NAME}_opencl_opt 
        ${PROJECT_NAME}_opencl_lib
    )
endif()


# --- all-cuda---
if(NOT ANDROID AND CUDAToolkit_FOUND)
    add_custom_target(all-cuda DEPENDS 
        ${PROJECT_NAME}_cuda 
        ${PROJECT_NAME}_cuda_opt 
        ${PROJECT_NAME}_cuda_lib
    )
endif()



