cmake_minimum_required(VERSION 3.24)
project(matrix_mult CXX) 


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
        cpu_functions/cpu_functions.cpp
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

# --- CPU target ---
compile_target(${PROJECT_NAME}_cpu
    SOURCES_FILES           cpu/lib_cpu.cpp 
    SHORTCUTS_NAMES         cpu CPU
)

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

    # --- OpenMP---
    compile_target(${PROJECT_NAME}_openmp
        SOURCES_FILES      openmp/lib_omp.cpp
        COMPILE_DEFS       OPENMP 
        COMPILE_OPTIONS    -fopenmp  
        LIBRARIES          -static-openmp   #openmp flags
                           -fopenmp  
                           -lm              # add math lib

        SHORTCUTS_NAMES    openmp OpenMP
    )

    # --- OpenMP-opt ---
    compile_target(${PROJECT_NAME}_openmp_opt
        SOURCES_FILES      openmp/lib_omp_opt.cpp
        COMPILE_DEFS       OPENMP 
        COMPILE_OPTIONS    -fopenmp  
        LIBRARIES          -static-openmp   #openmp flags
                           -fopenmp  
                           -lm              # add math lib

        SHORTCUTS_NAMES    openmp-opt OpenMP-opt
    )

    if(ANDROID_OPENBLAS_LIB_INC)
        # --- OpenMP-lib ---
        compile_target(${PROJECT_NAME}_openmp_lib
            SOURCES_FILES      openmp/lib_omp_lib.cpp
            COMPILE_DEFS       OPENMP 
                            "$<UPPER_CASE:${BLA_VENDOR}>"

            INCLUDES           ${ANDROID_INC}
                            ${ANDROID_INC}${ANDROID_ABI}

            LIBRARIES          ${ANDROID_LIB}${ANDROID_ABI}/libopenblas.a #static openblas         
                            -static-openmp   #openmp flags
                            -fopenmp
                            -lm              # add math lib
            SHORTCUTS_NAMES    openmp-lib OpenMP-lib
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

    # --- OpenMP Targets ---
    if(OpenMP_CXX_FOUND)
        # --- OpenMP ---
        compile_target(${PROJECT_NAME}_openmp
            SOURCES_FILES      openmp/lib_omp.cpp
            COMPILE_DEFS       OPENMP

            LIBRARIES          OpenMP::OpenMP_CXX # openmp flags
                               m   # equivalent to -lm

            SHORTCUTS_NAMES    openmp OpenMP 
        )

        # --- OpenMP-opt ---
        compile_target(${PROJECT_NAME}_openmp_opt
            SOURCES_FILES      openmp/lib_omp_opt.cpp
            COMPILE_DEFS       OPENMP

            LIBRARIES          OpenMP::OpenMP_CXX #  openmp flags
                               m   # equivalent to -lm 

            SHORTCUTS_NAMES    openmp-opt OpenMP-opt 
        )

        if(BLAS_INCLUDE_DIRS)
            compile_target(${PROJECT_NAME}_openmp_lib
                SOURCES_FILES      openmp/lib_omp_lib.cpp
                COMPILE_DEFS       OPENMP
                                   "$<UPPER_CASE:${BLA_VENDOR}>"

                INCLUDES           ${BLAS_INCLUDE_DIRS} 
                LIBRARIES          BLAS::BLAS  # handle ATLAS and OpenBLAS  
                                   OpenMP::OpenMP_CXX #auto openmp flags
                                   m   # equivalent to -lm 

                SHORTCUTS_NAMES    openmp-lib OpenMP-lib 
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
            COMPILE_DEFS       CUDA
            SET_CUDA           1
            LIBRARIES          CUDA::cudart 
            SHORTCUTS_NAMES    cuda-opt CUDA-opt
        )

        # --- CUDA-lib ---
        compile_target(${PROJECT_NAME}_cuda_lib
            SOURCES_FILES      cuda/lib_cuda_lib.cu
            COMPILE_DEFS       CUDA
            SET_CUDA           1
            LIBRARIES          CUDA::cudart CUDA::cublas
            SHORTCUTS_NAMES    cuda-lib CUDA-lib
        )
    endif()

    # --- HIP targets---
    if(hip_FOUND)
        # --- HIP ---
        compile_target(${PROJECT_NAME}_hip
            SOURCES_FILES       hip/lib_hip.cpp 
            SET_HIP_FILES       hip/lib_hip.cpp 
            COMPILE_DEFS        HIP
            LIBRARIES           hip::host
            SHORTCUTS_NAMES     hip- HIP
        )

        # --- HIP-opt ---
        compile_target(${PROJECT_NAME}_hip_opt
            SOURCES_FILES       hip/lib_hip_opt.cpp 
            SET_HIP_FILES       hip/lib_hip_opt.cpp 
            COMPILE_DEFS        HIP
            LIBRARIES           hip::host
            SHORTCUTS_NAMES     hip-opt HIP-opt
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


# --- all-openmp (Android) ---
if(ANDROID AND ANDROID_OPENBLAS_LIB_INC  )
    add_custom_target(all-openmp DEPENDS 
        ${PROJECT_NAME}_openmp 
        ${PROJECT_NAME}_openmp_opt 
        ${PROJECT_NAME}_openmp_lib
    )
endif()


# --- all-openmp---
if(NOT ANDROID AND OpenMP_CXX_FOUND AND BLAS_INCLUDE_DIRS )
    add_custom_target(all-openmp DEPENDS 
        ${PROJECT_NAME}_openmp 
        ${PROJECT_NAME}_openmp_opt 
        ${PROJECT_NAME}_openmp_lib
    )
endif()


# --- all-hip---
if(NOT ANDROID AND hip_FOUND)
    add_custom_target(all-hip DEPENDS 
        ${PROJECT_NAME}_hip 
        ${PROJECT_NAME}_hip_opt 
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




# # --- CPU target ---
# add_executable(${PROJECT_NAME}_cpu
#     main.cpp
#     cpu/lib_cpu.cpp
#     cpu_functions/cpu_functions.cpp
# )
# target_compile_definitions(${PROJECT_NAME}_cpu PRIVATE
#     ${DATATYPE}
#     BLOCK_SIZE=${BLOCKSIZE}
# )


# # --- openCL target ---
# if(ANDROID OR OpenCL_FOUND)
#     # --- mutiple device compilation ---
#     add_executable(${PROJECT_NAME}_opencl
#             main.cpp
#             opencl/lib_opencl.cpp
#             cpu_functions/cpu_functions.cpp
#     )
#     target_compile_definitions(${PROJECT_NAME}_opencl PRIVATE
#             ${DATATYPE}
#             BLOCK_SIZE=${BLOCKSIZE}
#             CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
#             OPENCL  # define OPENCL
#     )


#     if(ANDROID)
#         # --- specefic android opencl include and libs ---
#         target_include_directories(${PROJECT_NAME}_opencl PRIVATE ${ANDROID_INC})
#         target_link_libraries(${PROJECT_NAME}_opencl PRIVATE ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so)

#     elseif(OpenCL_FOUND)
#         # specefic computer opencl libs
#         target_link_libraries(${PROJECT_NAME}_opencl PRIVATE OpenCL::OpenCL)
#     endif()

#     #  Shortcuts 
#     add_custom_target(cl DEPENDS ${PROJECT_NAME}_opencl)
#     add_custom_target(OpenCL DEPENDS ${PROJECT_NAME}_opencl)



# else ()
#     # Fallback warning message 
#     message(WARNING "OpenCL installation was not found on this system. Skipping opencl target.")
# endif()


# # --- openCL-opt target ---
# if(ANDROID OR OpenCL_FOUND)
#     # --- mutiple device compilation ---
#     add_executable(${PROJECT_NAME}_opencl_opt
#             main.cpp
#             opencl/lib_opencl_opt.cpp
#             cpu_functions/cpu_functions.cpp
#     )
#     target_compile_definitions(${PROJECT_NAME}_opencl_opt PRIVATE
#             ${DATATYPE}
#             BLOCK_SIZE=${BLOCKSIZE}
#             CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
#             OPENCL  # define OPENCL
#     )

#     if(ANDROID)
#         # --- specefic android opencl include and libs ---
#         target_include_directories(${PROJECT_NAME}_opencl_opt PRIVATE ${ANDROID_INC})
#         target_link_libraries(${PROJECT_NAME}_opencl_opt PRIVATE ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so)

#     elseif(OpenCL_FOUND)
#         # specefic computer opencl libs
#         target_link_libraries(${PROJECT_NAME}_opencl_opt PRIVATE OpenCL::OpenCL)
#     endif()

#     # --- Shortcuts ---
#     add_custom_target(opencl-opt DEPENDS ${PROJECT_NAME}_opencl_opt)
#     add_custom_target(OpenCL-opt DEPENDS ${PROJECT_NAME}_opencl_opt)

# else ()
#     # Fallback warning message 
#     message(WARNING "OpenCL installation was not found on this system. Skipping OpenCL-opt target.")
# endif()


# # --- openCL-lib target ---
# if(ANDROID OR ( OpenCL_FOUND AND CLBlast_FOUND ))
#     # --- mutiple device compilation ---
#     add_executable(${PROJECT_NAME}_opencl_lib
#             main.cpp
#             opencl/lib_opencl_lib.cpp
#             cpu_functions/cpu_functions.cpp
#     )
#     target_compile_definitions(${PROJECT_NAME}_opencl_lib PRIVATE
#             ${DATATYPE}
#             BLOCK_SIZE=${BLOCKSIZE}
#             CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
#             OPENCL  # define OPENCL
#     )

#     if(ANDROID)
#         # --- specefic android opencl & clblast include and libs ---
#         target_include_directories(${PROJECT_NAME}_opencl_lib PRIVATE ${ANDROID_INC})
#         target_link_libraries(${PROJECT_NAME}_opencl_lib PRIVATE
#             ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so
#             ${ANDROID_LIB}/${ANDROID_ABI}/libclblast.a
#         )

#     elseif(OpenCL_FOUND AND CLBlast_FOUND)
#         # specefic computer opencl libs
#         target_link_libraries(${PROJECT_NAME}_opencl_lib PRIVATE
#             clblast
#             OpenCL::OpenCL
#         )
#     endif()

#     # --- Shortcuts ---
#     add_custom_target(opencl-lib DEPENDS ${PROJECT_NAME}_opencl_lib)
#     add_custom_target(OpenCL-lib DEPENDS ${PROJECT_NAME}_opencl_lib)

# else ()
#     # Fallback warning message 
#     if(NOT OpenCL_FOUND AND NOT CLBlast_FOUND)
#         set(MISSING_LIB "CLBlast and OpenCL")
#     elseif(NOT CLBlast_FOUND)
#         set(MISSING_LIB "CLBlast")
#     elseif(NOT OpenCL_FOUND)
#         set(MISSING_LIB "OpenCL")
#     endif()
#         message(WARNING "${MISSING_LIB} installation was not found on this system. Skipping OpenCL-lib target.")
# endif()



# # --- OpenMP target ---
# if(ANDROID OR OpenMP_CXX_FOUND)
#     # --- multiple device compilation ---
#     add_executable(${PROJECT_NAME}_openmp
#         main.cpp
#         openmp/lib_omp.cpp
#         cpu_functions/cpu_functions.cpp
#     )
#     target_compile_definitions(${PROJECT_NAME}_openmp PRIVATE
#         ${DATATYPE}
#         BLOCK_SIZE=${BLOCKSIZE}
#         OPENMP
#     )

#     if(ANDROID)
#         # --- specific android openmp flags ---
#         target_compile_options(${PROJECT_NAME}_openmp PRIVATE -fopenmp)
#         target_link_libraries(${PROJECT_NAME}_openmp PRIVATE 
#             -static-openmp  -fopenmp  #openmp flags
#             -lm     #add math lib
#         )

#     elseif(OpenMP_CXX_FOUND)
#         # --- specific computer openmp libs ---
#         target_link_libraries(${PROJECT_NAME}_openmp PRIVATE 
#             OpenMP::OpenMP_CXX #auto openmp flags
#             m   # equivalent to -lm 
#         )
#     endif()

#     # --- Shortcuts ---
#     add_custom_target(OpenMP DEPENDS ${PROJECT_NAME}_openmp)
#     add_custom_target(openmp DEPENDS ${PROJECT_NAME}_openmp)

# else()
#     # Fallback warning message
#     message(WARNING "OpenMP installation was not found on this system. Skipping OpenMP target.")
# endif()


# # --- OpenMP-opt target ---
# if(ANDROID OR OpenMP_CXX_FOUND )
#     # --- multiple device compilation ---
#     add_executable(${PROJECT_NAME}_openmp_opt
#         main.cpp
#         openmp/lib_omp_opt.cpp
#         cpu_functions/cpu_functions.cpp
#     )
#     target_compile_definitions(${PROJECT_NAME}_openmp_opt PRIVATE
#         ${DATATYPE}
#         BLOCK_SIZE=${BLOCKSIZE}
#         OPENMP
#     )

#     if(ANDROID)
#         # --- specific android openmp flags ---
#         target_compile_options(${PROJECT_NAME}_openmp_opt PRIVATE -fopenmp)
#         target_link_libraries(${PROJECT_NAME}_openmp_opt PRIVATE 
#             -static-openmp  -fopenmp  #openmp flags
#             -lm     #add math lib
#         )

#     elseif(OpenMP_CXX_FOUND)
#         # --- specific computer openmp libs ---
#         target_link_libraries(${PROJECT_NAME}_openmp_opt PRIVATE 
#             OpenMP::OpenMP_CXX #auto openmp flags
#             m   # equivalent to -lm 
#         )
#     endif()

#     # --- Shortcuts ---
#     add_custom_target(OpenMP-opt DEPENDS ${PROJECT_NAME}_openmp_opt)
#     add_custom_target(openmp-opt DEPENDS ${PROJECT_NAME}_openmp_opt)

# else()
#     # Fallback warning message
#     message(WARNING "OpenMP installation was not found on this system. Skipping OpenMP-opt target.")
# endif()



# # --- OpenMP-lib target ---
# if(ANDROID OR ( OpenMP_CXX_FOUND AND (BLAS_FOUND OR OpenBLAS_FOUND) ))
#     # --- multiple device compilation ---
#     add_executable(${PROJECT_NAME}_openmp_lib
#         main.cpp
#         openmp/lib_omp_lib.cpp
#         cpu_functions/cpu_functions.cpp
#     )
#     target_compile_definitions(${PROJECT_NAME}_openmp_lib PRIVATE
#         ${DATATYPE}
#         BLOCK_SIZE=${BLOCKSIZE}
#         OPENMP
#         "$<UPPER_CASE:${BLA_VENDOR}>"
#     )

#     if(ANDROID)
#         # --- specific android openmp flags ---
        
#         target_include_directories(${PROJECT_NAME}_openmp_lib PRIVATE 
#             ${ANDROID_INC}
#             ${ANDROID_INC}${ANDROID_ABI}
#         )
        
#         target_link_libraries(${PROJECT_NAME}_openmp_lib PRIVATE 
#             ${ANDROID_LIB}${ANDROID_ABI}/libopenblas.a #static openblas
#             -static-openmp  -fopenmp  #openmp flags
#             -lm     #add math lib
#         )

#     elseif(OpenMP_CXX_FOUND AND BLAS_INCLUDE_DIRS)

#         target_include_directories(${PROJECT_NAME}_openmp_lib PRIVATE ${BLAS_INCLUDE_DIRS})

#         target_link_libraries(${PROJECT_NAME}_openmp_lib PRIVATE
#             BLAS::BLAS  #handle ATLAS and OpenBLAS
#             OpenMP::OpenMP_CXX
#             m
#         )

#     endif()

#     # --- Shortcuts ---
#     add_custom_target(OpenMP-lib DEPENDS ${PROJECT_NAME}_openmp_lib)
#     add_custom_target(openmp-lib DEPENDS ${PROJECT_NAME}_openmp_lib)

# else()
#     # Fallback warning message 
#     if(NOT OpenMP_CXX_FOUND AND ( NOT BLAS_FOUND AND NOT OpenBLAS_FOUND ))
#         set(MISSING_LIB "OpenBLAS and OpenMP")
#     elseif(NOT BLAS_FOUND AND NOT OpenBLAS_FOUND)
#         set(MISSING_LIB "OpenBLAS")
#     elseif(NOT OpenMP_CXX_FOUND)
#         set(MISSING_LIB "OpenMP")
#     endif()
#         message(WARNING "${MISSING_LIB} installation was not found on this system. Skipping OpenMP-lib target.")
# endif()



# # --- CUDA target ---
# if(NOT ANDROID AND CUDAToolkit_FOUND)
#     # --- multiple device compilation ---
#     add_executable(${PROJECT_NAME}_cuda
#         main.cpp
#         cuda/lib_cuda.cu
#         cpu_functions/cpu_functions.cpp
#     )
#     target_compile_definitions(${PROJECT_NAME}_cuda PRIVATE
#         ${DATATYPE}
#         BLOCK_SIZE=${BLOCKSIZE}
#         CUDA
#     )

#     set_target_properties(${PROJECT_NAME}_cuda PROPERTIES
#         CUDA_ARCHITECTURES ${CUDA_ARCH}   # equivalent to -arch= in Makefile
#     )

#     # --- specific cuda libs ---
#     target_link_libraries(${PROJECT_NAME}_cuda PRIVATE 
#         CUDA::cudart 
#     )

#     # --- Shortcuts ---
#     add_custom_target(cuda DEPENDS ${PROJECT_NAME}_cuda)
#     add_custom_target(CUDA DEPENDS ${PROJECT_NAME}_cuda)

# elseif(NOT ANDROID)
#     # Fallback warning message
#     message(WARNING "CUDA installation was not found on this system. Skipping CUDA target.")
# endif()


# # --- CUDA-opt target ---
# if(NOT ANDROID AND CUDAToolkit_FOUND)
#     # --- multiple device compilation ---
#     add_executable(${PROJECT_NAME}_cuda_opt
#         main.cpp
#         cuda/lib_cuda_opt.cu
#         cpu_functions/cpu_functions.cpp
#     )
#     target_compile_definitions(${PROJECT_NAME}_cuda_opt PRIVATE
#         ${DATATYPE}
#         BLOCK_SIZE=${BLOCKSIZE}
#         CUDA
#     )
#     set_target_properties(${PROJECT_NAME}_cuda_opt PROPERTIES
#         CUDA_ARCHITECTURES ${CUDA_ARCH}   
#     )
#     # --- specific cuda libs ---
#     target_link_libraries(${PROJECT_NAME}_cuda_opt PRIVATE 
#         CUDA::cudart 
#     )

#     # --- Shortcuts ---
#     add_custom_target(cuda-opt DEPENDS ${PROJECT_NAME}_cuda_opt)
#     add_custom_target(CUDA-opt DEPENDS ${PROJECT_NAME}_cuda_opt)

# elseif(NOT ANDROID)
#     # Fallback warning message
#     message(WARNING "CUDA installation was not found on this system. Skipping CUDA-opt target.")
# endif()


# # --- CUDA-lib target ---
# if(NOT ANDROID AND CUDAToolkit_FOUND)
#     # --- multiple device compilation ---
#     add_executable(${PROJECT_NAME}_cuda_lib
#         main.cpp
#         cuda/lib_cuda_lib.cu
#         cpu_functions/cpu_functions.cpp
#     )
#     target_compile_definitions(${PROJECT_NAME}_cuda_lib PRIVATE
#         ${DATATYPE}
#         BLOCK_SIZE=${BLOCKSIZE}
#         CUDA
#     )
#     set_target_properties(${PROJECT_NAME}_cuda_lib PROPERTIES
#         CUDA_ARCHITECTURES ${CUDA_ARCH}   
#     )
#     # --- specific computer openmp libs ---
#     target_link_libraries(${PROJECT_NAME}_cuda_lib PRIVATE 
#         CUDA::cudart 
#         CUDA::cublas
#     )

#     # --- Shortcuts ---
#     add_custom_target(cuda-lib DEPENDS ${PROJECT_NAME}_cuda_lib)
#     add_custom_target(CUDA-lib DEPENDS ${PROJECT_NAME}_cuda_lib)

# elseif(NOT ANDROID)
#     # Fallback warning message
#     message(WARNING "CUDA installation was not found on this system. Skipping CUDA-lib target.")
# endif()



# # --- HIP target ---
# if(NOT ANDROID AND hip_FOUND)
#     # --- mixed device compilation ---
#     add_executable(${PROJECT_NAME}_hip
#         main.cpp                         
#         hip/lib_hip.cpp                  
#         cpu_functions/cpu_functions.cpp   
#     )

#     # Tell CMake lib_hip file contains HIP API code
#     set_source_files_properties(hip/lib_hip.cpp PROPERTIES LANGUAGE HIP)

#     target_compile_definitions(${PROJECT_NAME}_hip PRIVATE
#         ${DATATYPE}
#         BLOCK_SIZE=${BLOCKSIZE}
#         HIP
#     )


#     target_link_libraries(${PROJECT_NAME}_hip PRIVATE 
#         hip::host
#     )

#     # --- Shortcuts ---
#     add_custom_target(hip- DEPENDS ${PROJECT_NAME}_hip)
#     add_custom_target(HIP DEPENDS ${PROJECT_NAME}_hip)

# elseif(NOT ANDROID)
#     message(WARNING "HIP installation was not found on this system. Skipping HIP target.")
# endif()


# # --- HIP-opt target ---
# if(NOT ANDROID AND hip_FOUND)
#     # --- mixed device compilation ---
#     add_executable(${PROJECT_NAME}_hip_opt
#         main.cpp                        
#         hip/lib_hip_opt.cpp                  
#         cpu_functions/cpu_functions.cpp  
#     )

#     # Tell CMake lib_hip file contains HIP API code
#     set_source_files_properties(hip/lib_hip_opt.cpp PROPERTIES LANGUAGE HIP)

#     target_compile_definitions(${PROJECT_NAME}_hip_opt PRIVATE
#         ${DATATYPE}
#         BLOCK_SIZE=${BLOCKSIZE}
#         HIP
#     )

#     target_link_libraries(${PROJECT_NAME}_hip_opt PRIVATE 
#         hip::host
#     )

#     # --- Shortcuts ---
#     add_custom_target(hip-opt DEPENDS ${PROJECT_NAME}_hip_opt)
#     add_custom_target(HIP-opt DEPENDS ${PROJECT_NAME}_hip_opt)

# elseif(NOT ANDROID)
#     message(WARNING "HIP installation was not found on this system. Skipping HIP-opt target.")
# endif()