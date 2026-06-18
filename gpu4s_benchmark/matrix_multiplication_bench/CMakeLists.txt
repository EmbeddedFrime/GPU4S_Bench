cmake_minimum_required(VERSION 3.27)
project(matrix_mult CXX) 



# ====== CMake Configuration ======

# --- Global Variable ---
set(OPENCL_VERSION  300       CACHE STRING "API OpenCL Version : 200, 210, 220, 300, 310")
set(BLOCKSIZE       16        CACHE STRING "Block size for tiled kernels")
set(OPT_FLAG       "-O3"      CACHE STRING "Compiler optimization level : -O2, -O3, -Ofast")
set(DATATYPE       "FLOAT"    CACHE STRING "Data type: FLOAT, DOUBLE, INT")
set(CUDA_ARCH      "native"   CACHE STRING "API CUDA version: native, sm_72-86")
set(BLA_VENDOR     "OpenBLAS" CACHE STRING "BLAS lib : ATLAS, OpenBLAS")

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





# ====== Compilation of Targets ======


# --- CPU target ---
add_executable(${PROJECT_NAME}_cpu
    main.cpp
    cpu/lib_cpu.cpp
    cpu_functions/cpu_functions.cpp
)
target_compile_definitions(${PROJECT_NAME}_cpu PRIVATE
    ${DATATYPE}
    BLOCK_SIZE=${BLOCKSIZE}
)

add_custom_target(cpu DEPENDS ${PROJECT_NAME}_cpu)
add_custom_target(CPU DEPENDS ${PROJECT_NAME}_cpu)



# --- openCL target ---
if(ANDROID OR OpenCL_FOUND)
    # --- mutiple device compilation ---
    add_executable(${PROJECT_NAME}_opencl
            main.cpp
            opencl/lib_opencl.cpp
            cpu_functions/cpu_functions.cpp
    )
    target_compile_definitions(${PROJECT_NAME}_opencl PRIVATE
            ${DATATYPE}
            BLOCK_SIZE=${BLOCKSIZE}
            CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
            OPENCL  # define OPENCL
    )


    if(ANDROID)
        # --- specefic android opencl include and libs ---
        target_include_directories(${PROJECT_NAME}_opencl PRIVATE ${ANDROID_INC})
        target_link_libraries(${PROJECT_NAME}_opencl PRIVATE ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so)

    elseif(OpenCL_FOUND)
        # specefic computer opencl libs
        target_link_libraries(${PROJECT_NAME}_opencl PRIVATE OpenCL::OpenCL)
    endif()

    #  Shortcuts 
    add_custom_target(cl DEPENDS ${PROJECT_NAME}_opencl)
    add_custom_target(OpenCL DEPENDS ${PROJECT_NAME}_opencl)



else ()
    # Fallback warning message 
    message(WARNING "OpenCL installation was not found on this system. Skipping opencl target.")
endif()


# --- openCL-opt target ---
if(ANDROID OR OpenCL_FOUND)
    # --- mutiple device compilation ---
    add_executable(${PROJECT_NAME}_opencl_opt
            main.cpp
            opencl/lib_opencl_opt.cpp
            cpu_functions/cpu_functions.cpp
    )
    target_compile_definitions(${PROJECT_NAME}_opencl_opt PRIVATE
            ${DATATYPE}
            BLOCK_SIZE=${BLOCKSIZE}
            CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
            OPENCL  # define OPENCL
    )

    target_compile_options(${PROJECT_NAME}_opencl_opt PRIVATE OPT_FLAG)

    if(ANDROID)
        # --- specefic android opencl include and libs ---
        target_include_directories(${PROJECT_NAME}_opencl_opt PRIVATE ${ANDROID_INC})
        target_link_libraries(${PROJECT_NAME}_opencl_opt PRIVATE ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so)

    elseif(OpenCL_FOUND)
        # specefic computer opencl libs
        target_link_libraries(${PROJECT_NAME}_opencl_opt PRIVATE OpenCL::OpenCL)
    endif()

    # --- Shortcuts ---
    add_custom_target(opencl-opt DEPENDS ${PROJECT_NAME}_opencl_opt)
    add_custom_target(OpenCL-opt DEPENDS ${PROJECT_NAME}_opencl_opt)

else ()
    # Fallback warning message 
    message(WARNING "OpenCL installation was not found on this system. Skipping OpenCL-opt target.")
endif()


# --- openCL-lib target ---
if(ANDROID OR ( OpenCL_FOUND AND CLBlast_FOUND ))
    # --- mutiple device compilation ---
    add_executable(${PROJECT_NAME}_opencl_lib
            main.cpp
            opencl/lib_opencl_lib.cpp
            cpu_functions/cpu_functions.cpp
    )
    target_compile_definitions(${PROJECT_NAME}_opencl_lib PRIVATE
            ${DATATYPE}
            BLOCK_SIZE=${BLOCKSIZE}
            CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
            OPENCL  # define OPENCL
    )

    if(ANDROID)
        # --- specefic android opencl & clblast include and libs ---
        target_include_directories(${PROJECT_NAME}_opencl_lib PRIVATE ${ANDROID_INC})
        target_link_libraries(${PROJECT_NAME}_opencl_lib PRIVATE
            ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so
            ${ANDROID_LIB}/${ANDROID_ABI}/libclblast.a
        )

    elseif(OpenCL_FOUND AND CLBlast_FOUND)
        # specefic computer opencl libs
        target_link_libraries(${PROJECT_NAME}_opencl_lib PRIVATE
            clblast
            OpenCL::OpenCL
        )
    endif()

    # --- Shortcuts ---
    add_custom_target(opencl-lib DEPENDS ${PROJECT_NAME}_opencl_lib)
    add_custom_target(OpenCL-lib DEPENDS ${PROJECT_NAME}_opencl_lib)

else ()
    # Fallback warning message 
    if(NOT OpenCL_FOUND AND NOT CLBlast_FOUND)
        set(MISSING_LIB "CLBlast and OpenCL")
    elseif(NOT CLBlast_FOUND)
        set(MISSING_LIB "CLBlast")
    elseif(NOT OpenCL_FOUND)
        set(MISSING_LIB "OpenCL")
    endif()
        message(WARNING "${MISSING_LIB} installation was not found on this system. Skipping OpenCL-lib target.")
endif()



# --- OpenMP target ---
if(ANDROID OR OpenMP_CXX_FOUND)
    # --- multiple device compilation ---
    add_executable(${PROJECT_NAME}_openmp
        main.cpp
        openmp/lib_omp.cpp
        cpu_functions/cpu_functions.cpp
    )
    target_compile_definitions(${PROJECT_NAME}_openmp PRIVATE
        ${DATATYPE}
        BLOCK_SIZE=${BLOCKSIZE}
        OPENMP
    )

    if(ANDROID)
        # --- specific android openmp flags ---
        target_compile_options(${PROJECT_NAME}_openmp PRIVATE -fopenmp)
        target_link_libraries(${PROJECT_NAME}_openmp PRIVATE 
            -static-openmp  -fopenmp  #openmp flags
            -lm     #add math lib
        )

    elseif(OpenMP_CXX_FOUND)
        # --- specific computer openmp libs ---
        target_link_libraries(${PROJECT_NAME}_openmp PRIVATE 
            OpenMP::OpenMP_CXX #auto openmp flags
            m   # equivalent to -lm 
        )
    endif()

    # --- Shortcuts ---
    add_custom_target(OpenMP DEPENDS ${PROJECT_NAME}_openmp)
    add_custom_target(openmp DEPENDS ${PROJECT_NAME}_openmp)

else()
    # Fallback warning message
    message(WARNING "OpenMP installation was not found on this system. Skipping OpenMP target.")
endif()


# --- OpenMP-opt target ---
if(ANDROID OR OpenMP_CXX_FOUND )
    # --- multiple device compilation ---
    add_executable(${PROJECT_NAME}_openmp_opt
        main.cpp
        openmp/lib_omp_opt.cpp
        cpu_functions/cpu_functions.cpp
    )
    target_compile_definitions(${PROJECT_NAME}_openmp_opt PRIVATE
        ${DATATYPE}
        BLOCK_SIZE=${BLOCKSIZE}
        OPENMP
    )

    if(ANDROID)
        # --- specific android openmp flags ---
        target_compile_options(${PROJECT_NAME}_openmp_opt PRIVATE -fopenmp)
        target_link_libraries(${PROJECT_NAME}_openmp_opt PRIVATE 
            -static-openmp  -fopenmp  #openmp flags
            -lm     #add math lib
        )

    elseif(OpenMP_CXX_FOUND)
        # --- specific computer openmp libs ---
        target_link_libraries(${PROJECT_NAME}_openmp_opt PRIVATE 
            OpenMP::OpenMP_CXX #auto openmp flags
            m   # equivalent to -lm 
        )
    endif()

    # --- Shortcuts ---
    add_custom_target(OpenMP-opt DEPENDS ${PROJECT_NAME}_openmp_opt)
    add_custom_target(openmp-opt DEPENDS ${PROJECT_NAME}_openmp_opt)

else()
    # Fallback warning message
    message(WARNING "OpenMP installation was not found on this system. Skipping OpenMP-opt target.")
endif()



# --- OpenMP-lib target ---
if(ANDROID OR ( OpenMP_CXX_FOUND AND (BLAS_FOUND OR OpenBLAS_FOUND) ))
    # --- multiple device compilation ---
    add_executable(${PROJECT_NAME}_openmp_lib
        main.cpp
        openmp/lib_omp_lib.cpp
        cpu_functions/cpu_functions.cpp
    )
    target_compile_definitions(${PROJECT_NAME}_openmp_lib PRIVATE
        ${DATATYPE}
        BLOCK_SIZE=${BLOCKSIZE}
        OPENMP
        "$<UPPER_CASE:${BLA_VENDOR}>"
    )

    if(ANDROID)
        # --- specific android openmp flags ---
        
        target_include_directories(${PROJECT_NAME}_openmp_lib PRIVATE 
            ${ANDROID_INC}
            ${ANDROID_INC}${ANDROID_ABI}
        )
        
        target_link_libraries(${PROJECT_NAME}_openmp_lib PRIVATE 
            ${ANDROID_LIB}${ANDROID_ABI}/libopenblas.a #static openblas
            -static-openmp  -fopenmp  #openmp flags
            -lm     #add math lib
        )

    elseif(OpenMP_CXX_FOUND AND BLAS_INCLUDE_DIRS)

        target_include_directories(${PROJECT_NAME}_openmp_lib PRIVATE ${BLAS_INCLUDE_DIRS})

        target_link_libraries(${PROJECT_NAME}_openmp_lib PRIVATE
            BLAS::BLAS  #handle ATLAS and OpenBLAS
            OpenMP::OpenMP_CXX
            m
        )

    endif()

    # --- Shortcuts ---
    add_custom_target(OpenMP-lib DEPENDS ${PROJECT_NAME}_openmp_lib)
    add_custom_target(openmp-lib DEPENDS ${PROJECT_NAME}_openmp_lib)

else()
    # Fallback warning message 
    if(NOT OpenMP_CXX_FOUND AND ( NOT BLAS_FOUND AND NOT OpenBLAS_FOUND ))
        set(MISSING_LIB "OpenBLAS and OpenMP")
    elseif(NOT BLAS_FOUND AND NOT OpenBLAS_FOUND)
        set(MISSING_LIB "OpenBLAS")
    elseif(NOT OpenMP_CXX_FOUND)
        set(MISSING_LIB "OpenMP")
    endif()
        message(WARNING "${MISSING_LIB} installation was not found on this system. Skipping OpenMP-lib target.")
endif()



# --- CUDA target ---
if(NOT ANDROID AND CUDAToolkit_FOUND)
    # --- multiple device compilation ---
    add_executable(${PROJECT_NAME}_cuda
        main.cpp
        cuda/lib_cuda.cu
        cpu_functions/cpu_functions.cpp
    )
    target_compile_definitions(${PROJECT_NAME}_cuda PRIVATE
        ${DATATYPE}
        BLOCK_SIZE=${BLOCKSIZE}
        CUDA
    )

    set_target_properties(${PROJECT_NAME}_cuda PROPERTIES
        CUDA_ARCHITECTURES ${CUDA_ARCH}   # equivalent to -arch= in Makefile
    )

    # --- specific cuda libs ---
    target_link_libraries(${PROJECT_NAME}_cuda PRIVATE 
        CUDA::cudart 
    )

    # --- Shortcuts ---
    add_custom_target(cuda DEPENDS ${PROJECT_NAME}_cuda)
    add_custom_target(CUDA DEPENDS ${PROJECT_NAME}_cuda)

elseif(NOT ANDROID)
    # Fallback warning message
    message(WARNING "CUDA installation was not found on this system. Skipping CUDA target.")
endif()


# --- CUDA-opt target ---
if(NOT ANDROID AND CUDAToolkit_FOUND)
    # --- multiple device compilation ---
    add_executable(${PROJECT_NAME}_cuda_opt
        main.cpp
        cuda/lib_cuda_opt.cu
        cpu_functions/cpu_functions.cpp
    )
    target_compile_definitions(${PROJECT_NAME}_cuda_opt PRIVATE
        ${DATATYPE}
        BLOCK_SIZE=${BLOCKSIZE}
        CUDA
    )
    set_target_properties(${PROJECT_NAME}_cuda_opt PROPERTIES
        CUDA_ARCHITECTURES ${CUDA_ARCH}   
    )
    # --- specific cuda libs ---
    target_link_libraries(${PROJECT_NAME}_cuda_opt PRIVATE 
        CUDA::cudart 
    )

    # --- Shortcuts ---
    add_custom_target(cuda-opt DEPENDS ${PROJECT_NAME}_cuda_opt)
    add_custom_target(CUDA-opt DEPENDS ${PROJECT_NAME}_cuda_opt)

elseif(NOT ANDROID)
    # Fallback warning message
    message(WARNING "CUDA installation was not found on this system. Skipping CUDA-opt target.")
endif()


# --- CUDA-lib target ---
if(NOT ANDROID AND CUDAToolkit_FOUND)
    # --- multiple device compilation ---
    add_executable(${PROJECT_NAME}_cuda_lib
        main.cpp
        cuda/lib_cuda_lib.cu
        cpu_functions/cpu_functions.cpp
    )
    target_compile_definitions(${PROJECT_NAME}_cuda_lib PRIVATE
        ${DATATYPE}
        BLOCK_SIZE=${BLOCKSIZE}
        CUDA
    )
    set_target_properties(${PROJECT_NAME}_cuda_lib PROPERTIES
        CUDA_ARCHITECTURES ${CUDA_ARCH}   
    )
    # --- specific computer openmp libs ---
    target_link_libraries(${PROJECT_NAME}_cuda_lib PRIVATE 
        CUDA::cudart 
        CUDA::cublas
    )

    # --- Shortcuts ---
    add_custom_target(cuda-lib DEPENDS ${PROJECT_NAME}_cuda_lib)
    add_custom_target(CUDA-lib DEPENDS ${PROJECT_NAME}_cuda_lib)

elseif(NOT ANDROID)
    # Fallback warning message
    message(WARNING "CUDA installation was not found on this system. Skipping CUDA-lib target.")
endif()



# --- HIP target ---
if(NOT ANDROID AND hip_FOUND)
    # --- mixed device compilation ---
    add_executable(${PROJECT_NAME}_hip
        main.cpp                         
        hip/lib_hip.cpp                  
        cpu_functions/cpu_functions.cpp   
    )

    # Tell CMake lib_hip file contains HIP API code
    set_source_files_properties(hip/lib_hip.cpp PROPERTIES LANGUAGE HIP)

    target_compile_definitions(${PROJECT_NAME}_hip PRIVATE
        ${DATATYPE}
        BLOCK_SIZE=${BLOCKSIZE}
        HIP
    )


    target_link_libraries(${PROJECT_NAME}_hip PRIVATE 
        hip::host
    )

    # --- Shortcuts ---
    add_custom_target(hip- DEPENDS ${PROJECT_NAME}_hip)
    add_custom_target(HIP DEPENDS ${PROJECT_NAME}_hip)

elseif(NOT ANDROID)
    message(WARNING "HIP installation was not found on this system. Skipping HIP target.")
endif()


# --- HIP-opt target ---
if(NOT ANDROID AND hip_FOUND)
    # --- mixed device compilation ---
    add_executable(${PROJECT_NAME}_hip_opt
        main.cpp                        
        hip/lib_hip_opt.cpp                  
        cpu_functions/cpu_functions.cpp  
    )

    # Tell CMake lib_hip file contains HIP API code
    set_source_files_properties(hip/lib_hip_opt.cpp PROPERTIES LANGUAGE HIP)

    target_compile_definitions(${PROJECT_NAME}_hip_opt PRIVATE
        ${DATATYPE}
        BLOCK_SIZE=${BLOCKSIZE}
        HIP
    )

    target_link_libraries(${PROJECT_NAME}_hip_opt PRIVATE 
        hip::host
    )

    # --- Shortcuts ---
    add_custom_target(hip-opt DEPENDS ${PROJECT_NAME}_hip_opt)
    add_custom_target(HIP-opt DEPENDS ${PROJECT_NAME}_hip_opt)

elseif(NOT ANDROID)
    message(WARNING "HIP installation was not found on this system. Skipping HIP-opt target.")
endif()


