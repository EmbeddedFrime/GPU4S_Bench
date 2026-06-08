# Result of my research on embeded CPU and GPU performance


## Introduction
To compare the performance of embedded CPUs and GPUs, I conducted a benchmark test using matrix multiplication as a common computational task. The benchmark was performed on two different platforms: 
 - Laptop : AMD Ryzen 7 7840HS & NVIDIA GeForce RTX 4060 
 - DragonBoard 810 : Qualcomm Snapdragon 810 & Adreno 430 



## Pre-requisites
### 1. Android NDK r27d (27.3.13750724)
 
First, you need to download the [Android NDK](https://github.com/android/ndk/wiki
) (I use r27d 27.3.13750724) installed in your environment to cross-compile for Android targets. Ensure that the `NDK_ROOT` environment variable points to the directory of your NDK installation in the make file.
```makefile
# Set the  NDK_PATH to your Android NDK installation path
NDK_VERSION ?= 27.3.13750724
NDK_PATH = $(ANDROID_HOME)/ndk/$(NDK_VERSION)
```

In addition you need to select the compilator for your phone device, for my example I use aarch64-linux-android that is for android 5 - 64 bits device :
```makefile
# Set the target architecture for Android cross-compilation
CLANG = $(LLVM_PREBUILT)/bin/aarch64-linux-android21-clang++ 
```

### 3. OpenCL header .h/.hpp
Furthermore, you must download the OpenCL header .h  and  .hpp (I used v2026.05.29):  

```bash
#pull the latest release tag from the OpenCL-Headers repository
LATEST_TAG_H=$(curl -s https://api.github.com/repos/KhronosGroup/OpenCL-Headers/releases/latest | grep -oP '"tag_name": "\K[^"]+')

# Download OpenCL repository for header files
git clone -b "$LATEST_TAG_H" --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git ./opencl/tmp

# Rm & Move the CL directory with openCL.h to the opencl_headers directory
rm -rf ./opencl/opencl_headers/CL & mkdir -p ./opencl/opencl_headers/CL && mv ./opencl/tmp/CL/* ./opencl/opencl_headers/CL

# Download the opencl.hpp header file
LATEST_TAG_HPP=$(curl -s https://api.github.com/repos/KhronosGroup/OpenCL-CLHPP/releases/latest | grep -oP '"tag_name": "\K[^"]+')

curl -o ./opencl/opencl_headers/CL/opencl.hpp https://raw.githubusercontent.com/KhronosGroup/OpenCL-CLHPP/${LATEST_TAG_HPP}/include/CL/opencl.hpp

#For old version of openCL :
#curl -o ./opencl/opencl_headers/CL/opencl.hpp https://raw.githubusercontent.com/#KhronosGroup/OpenCL-CLHPP/${LATEST_TAG_HPP}/include/CL/cl2.hpp

# Remove the temporary directory
rm -rf ./opencl/tmp

```

### 3. install ADB (Bonus)

Also I recommad need to install ADB (Android Debug Bridge) to be able to push the compiled binaries to the phone and execute them. You can install it using the following command :
```bash
fedora : sudo dnf install android-tools
Debian : sudo apt install android-tools-adb
```


### 4. libOpenCL.so (Bonus)
To link the OpenCL benchmark correctly, you need the `libOpenCL.so` library from your target device. Typically, you can extract this  driver directly from your phone using ADB:

```bash
# For 64-bit architectures (Recommended)
adb pull /vendor/lib64/libOpenCL.so ./opencl/

# For 32-bit architectures
adb pull /vendor/lib/libOpenCL.so ./opencl/
```

### Alternative: Using the Compilation Stub

Depending on your phone's age or vendor-specific toolchain, the pulled .so library might contain corrupted or non-standard ELF symbol tables that cause the modern PC host linker (ld.lld) to fail during compilation.

To bypass this driver limitation, a local compilation stub (opencl_stub.c) is provided. This file contains empty function prototypes matching the OpenCL API signatures. The Makefile automatically compiles this stub into a clean, structurally compliant libOpenCL.so to satisfy the linker on your PC. At runtime on the phone, the operating system will seamlessly swap this stub out for the actual physical GPU driver.

## Instructions: 
Always clean the build artifacts before compiling a new architecture target configuration. Pass `DATATYPE` and `BLOCKSIZE` as environment overrides.

```bash
# Clean project workspace
make clean
```

### 1. Compile the code using the provided Makefile on both platforms

#### PC / Laptop Build (Fedora Host)

To compile the native Linux binaries utilizing your local toolchain:

```bash
# Compile individual backends
make cpu DATATYPE=FLOAT BLOCKSIZE=16
make openmp DATATYPE=FLOAT BLOCKSIZE=16
make opencl DATATYPE=FLOAT BLOCKSIZE=16
```

#### Android Cross-Compilation Build

To cross-compile binaries optimized for AArch64 Android architectures using the Android NDK (r27d):

```bash
# Compile all Android targets simultaneously (CPU, OpenMP, and OpenCL)
make all-android DATATYPE=FLOAT BLOCKSIZE=16
```


### 2. Execution Guide

All benchmarks accept size parameters via the -s flag and timing flags via -t.

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

Before executing, transfer the cross-compiled binaries from your ./bin/ directory over to the high-privilege temporary directory on the target Android device:

```bash
# Push binaries to the device
adb push ./bin/matrix_multiplication_android_cpu_float_256 /data/local/tmp/matrix_android_cpu
adb push ./bin/matrix_multiplication_android_omp_float /data/local/tmp/matrix_android_openmp
adb push ./bin/matrix_multiplication_android_omp_opt_float /data/local/tmp/matrix_android_openmp_opt
adb push ./bin/matrix_multiplication_android_opencl_float_256 /data/local/tmp/matrix_android_opencl
adb push ./bin/matrix_multiplication_android_opencl_opt_float_256 /data/local/tmp/matrix_android_opencl_opt

# Grant execution permissions
adb shell chmod 755 /data/local/tmp/matrix_android_*
```

Execute the benchmarks directly on the phone shell:
```bash
# Android CPU Execution
adb shell /data/local/tmp/matrix_android_cpu -s 1024 -t

# Android OpenMP Execution
adb shell /data/local/tmp/matrix_android_openmp -s 1024 -t
adb shell /data/local/tmp/matrix_android_openmp_opt -s 1024 -t

# Android OpenCL Execution (QUALCOMM Adreno)
adb shell /data/local/tmp/matrix_android_opencl -s 1024 -t
adb shell /data/local/tmp/matrix_android_opencl_opt -s 1024 -t
```

### 3. Performance Metric Sheet

Below are the benchmarking metrics recorded for a matrix size of 1024 × 1024 (-s 1024) with a block size configuration of 16 (BLOCKSIZE=16).

| Environment / Platform | Compute Backend | Target Device | Host -> Device Copy | Kernel Execution Time | Device -> Host Copy | Total Elapsed Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fedora Laptop** | Single-Thread CPU | Generic Device | 0.0000 ms | 6958.0000 ms | 0.0000 ms | 6958.0000 ms |
| **Fedora Laptop** | Multi-Thread OpenMP | Generic Device | 0.0000 ms | 604.5398 ms | 0.0000 ms | 604.5398 ms |
| **Fedora Laptop** | Heterogeneous OpenCL | NVIDIA RTX 4060 Laptop | 0.6914 ms | 11.5353 ms | 0.3446 ms | 12.5713 ms |
| | | | | | | |
| **Android Mobile** | Single-Thread CPU | Generic Device | 0.0000 ms | 25455.0000 ms | 0.0000 ms | 25455.0000 ms |
| **Android Mobile** | Multi-Thread OpenMP | Generic Device | 0.0000 ms | 11063.5859 ms | 0.0000 ms | 11063.5859 ms |
| **Android Mobile** | Heterogeneous OpenCL | QUALCOMM Adreno GPU | 0.3150 ms | 1353.3130 ms | 14.3440 ms | 1367.9720 ms |


atlas and openblast 

try 32 bits and 64 bits

The objectives is to translate all the benchmarks to the same code and then compare the performance of the different languages and libraries on the same hardware.




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


How to install cblas : 
https://www.openmathlib.org/OpenBLAS/docs/install/#__tabbed_1_3



how to install  CLBBlast : 
https://github.com/CNugteren/CLBlast/blob/master/doc/installation.md