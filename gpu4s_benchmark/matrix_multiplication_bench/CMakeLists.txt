cmake_minimum_required(VERSION 3.16)
project(matrix_multiplication_bench CXX) 

# ====== CMake Configuration ======

# --- Global compilation settings ---
set(CMAKE_CXX_STANDARD 17)    # force clang 17 or g++ 17 compiler
set(CMAKE_BUILD_TYPE Release) # add production-level optimization

# --- Cache Variable ---
set(DATATYPE "FLOAT" CACHE STRING "Data type: FLOAT, DOUBLE, INT")
set(BLOCKSIZE 4      CACHE STRING "Block size for tiled kernels")

# --- Define Output directory ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

# ====== Shortcut Handler ======
add_custom_target(CPU DEPENDS matrix_mult_cpu)
add_custom_target(cpu DEPENDS matrix_mult_cpu)


# ====== Compilation of Targets ======

# --- CPU target ---
add_executable(matrix_mult_cpu
    main.cpp
    cpu/lib_cpu.cpp
    cpu_functions/cpu_functions.cpp
)
target_compile_definitions(matrix_mult_cpu PRIVATE
    ${DATATYPE}
    BLOCK_SIZE=${BLOCKSIZE}
)
target_compile_options(matrix_mult_cpu PRIVATE -O3)



