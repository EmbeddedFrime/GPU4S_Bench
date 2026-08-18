# Result of my research on embeded CPU and GPU performance

## Introduction
To compare the performance of embedded CPUs and GPUs, I conducted a benchmark test using matrix multiplication as a common computational task. The benchmark was performed differents platforms: 

 - x86_64 CPU: AMD Ryzen 7 7840HS
 - Laptop GPU: NVIDIA RTX 4060 Laptop
 - ARM64  CPU: Qualcomm Snapdragon 810 & Qualcomm Snapdragon 8 Gen 2
 - Phone  GPU: Adreno 430 & Adreno 740


### 2. Execution Guide

All benchmarks accept size parameters via the `-s` flag and timing via `-t`.

#### PC / Laptop Build (Fedora Host)

```bash
# Native CPU Execution
./bin/matrix_multiplication_cpu_float_256 -s 1024 -t

# Native OpenMP Execution
./bin/matrix_multiplication_omp_float -s 1024 -t

# Native OpenCL Execution (NVIDIA RTX 4060)
./bin/matrix_multiplication_opencl_float_256 -s 1024 -t
```

#### Pushing and Running on Android Devices via ADB

```bash
# Push binaries to the device
adb push ./build-android/bin/* /data/local/tmp/

# Grant execution permissions
adb shell chmod 755 /data/local/tmp/*
```

Execute the benchmarks:

```bash
# Android CPU Execution
adb shell /data/local/tmp/matrix_mult_opencl -s 1024 -t

# Android OpenMP Execution
adb shell /data/local/tmp/matrix_android_openmp -s 1024 -t
adb shell /data/local/tmp/matrix_android_openmp_opt -s 1024 -t
adb shell /data/local/tmp/matrix_android_openmp_lib -s 1024 -t

# Android OpenCL Execution (QUALCOMM Adreno)
adb shell /data/local/tmp/matrix_android_opencl -s 1024 -t
adb shell /data/local/tmp/matrix_android_opencl_opt -s 1024 -t
adb shell /data/local/tmp/matrix_android_opencl_lib -s 1024 -t
```

</details>



```bash
# 1. Clean the old binaries
make clean

# 2. Compile the library with debugging symbols (-g)
g++ -DFLOAT -DOPENCL -g -c ./opencl/lib_opencl_lib.cpp -o ./opencl/lib_opencl_lib.o -I/usr/local/cuda/include/ -lclblast

# 3. Compile the main harness with debugging symbols (-g) AND turn off optimizations (-O0)
g++ -DFLOAT -DOPENCL -g -O0 main.cpp ./opencl/lib_opencl_lib.o ./cpu_functions/cpu_functions.cpp -o ./bin/matrix_multiplication_opencl_lib_float -I/usr/local/cuda/include/ -lOpenCL -lclblast

# launch gdb
gdb --args ./bin/matrix_multiplication_opencl_lib_float -s 1024 -t


(gdb) r
(gdb) bt
```




### 1. CMake Build for Android
```
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/27.3.13750724/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-21
  ```

### 2. Compile your target
```
cmake --build build-android --target android_matrix_mult_cpu
```


### 1. CMake Build for Android
```
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/27.3.13750724/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-21

cmake --build build-android 

adb push ./build-android/bin/* /data/local/tmp/
adb shell chmod 755 /data/local/tmp/*


./data/local/tmp/


cmake --build build-android --target cpu
cmake --build build-android --target openmp openmp-opt cl opencl-opt
```


### 2. Push it to your phone and run it
```
adb push ./build-android/bin/* /data/local/tmp/
adb shell chmod 755 /data/local/tmp/*
adb shell /data/local/tmp/matrix_mult_opencl -s 1024 -t -v
```

i=1; while [ "$i" -le 100 ]; do ./data/local/tmp/matrix_mult_opencl_opt -s 512 -c >> /data/local/tmp/results_512.csv; i=$((i + 1)); done; echo "Done!"

### 1. CMake Build for computer
```
cmake -B build

cmake --build build --target cpu
```

### 2. execute it on your computer
```
./build/bin/matrix_mult_cpu -s 1024 -t
```


speak about all the parameter you can change in the benchmark and how to change them, like the size of the matrix, the block size, the data type, the number of threads, etc.



for b in ./build/bin/*; do if [ -x "$b" ]; then name="${b##*/}"; echo -e "\n=== Running $name ==="; "$b" -s 2048  -t ; fi; done


adb shell 'for b in /data/local/tmp/convolution*; do [ -x "$b" ] || continue; name="${b##*/}"; echo "\n=== Running $name ==="; "$b" -s 2048 -t -v; done'

cmake -B build-float-256 -DDATATYPE=FLOAT -DBLOCKSIZE=256
cmake -B build-double-128 -DDATATYPE=DOUBLE -DBLOCKSIZE=128




