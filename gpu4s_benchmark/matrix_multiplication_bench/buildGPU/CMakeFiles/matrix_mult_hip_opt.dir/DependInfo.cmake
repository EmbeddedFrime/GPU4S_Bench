
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "HIP"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_HIP
  "/home/noah/Documents/BSC/GPU4S_Bench/gpu4s_benchmark/matrix_multiplication_bench/hip/lib_hip_opt.cpp" "/home/noah/Documents/BSC/GPU4S_Bench/gpu4s_benchmark/matrix_multiplication_bench/buildGPU/CMakeFiles/matrix_mult_hip_opt.dir/hip/lib_hip_opt.cpp.o"
  )
set(CMAKE_HIP_COMPILER_ID "Clang")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_HIP
  "BLOCK_SIZE=16"
  "FLOAT"
  "GPU"
  "HIP"
  "USE_PROF_API=1"
  "__HIP_PLATFORM_AMD__=1"
  "__HIP_ROCclr__=1"
  "little"
  )

# The include file search paths:
set(CMAKE_HIP_TARGET_INCLUDE_PATH
  "/home/noah/Documents/BSC/GPU4S_Bench/gpu4s_benchmark/common/cmake/.."
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  "/home/noah/Documents/BSC/GPU4S_Bench/gpu4s_benchmark/matrix_multiplication_bench/cpu_functions/cpu_functions.cpp" "CMakeFiles/matrix_mult_hip_opt.dir/cpu_functions/cpu_functions.cpp.o" "gcc" "CMakeFiles/matrix_mult_hip_opt.dir/cpu_functions/cpu_functions.cpp.o.d"
  "/home/noah/Documents/BSC/GPU4S_Bench/gpu4s_benchmark/matrix_multiplication_bench/main.cpp" "CMakeFiles/matrix_mult_hip_opt.dir/main.cpp.o" "gcc" "CMakeFiles/matrix_mult_hip_opt.dir/main.cpp.o.d"
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
