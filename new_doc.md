# GPU4S Benchmark Suite documentation
## Introduction
GPU4S is a benchmarking suite that rely OBPMark-Kernel to test the perfomance adn reliability of GPUs and  multi-thread processor for space applications. The benchmarking suites have been developed in order to be compatible with heterogeneous platforms : 
- Computer (x86_64, x86_32)
- Android (ARM64, ARM32)
- Jetson (ARM64) (currently in development)

The benchmark also support a couple of different programming models, Libs and frameworks to be able to compare the perfomance of the same benchmark in different implementation: 
- Standard C/C++
- CUDA
- HIP
- OpenCL
- OpenMP


## Benchmark List and Basic Description

For most of the benchmark suite there is a naïve, optimized and library version. The benchmarks with their implementations are listed below.

| Benchmark | Naïve | Optimized | Library |
|---|:---:|:---:|:---:|
| Cifar 10 | ✅ | ✅ | ✅ (CUDA only) |
| Cifar 10 Multiple | ✅ | ✅ | ✅ (CUDA only) |
| Convolution 2D | ✅ | ✅ | ✅ (CUDA only) |
| Correlation 2D | ✅ | ✅ | ❌ |
| Fast Fourier Transform 2D | ❌ | ❌ | ✅ |
| Fast Fourier Transform | ✅ | ✅ | ✅ |
| Fast Fourier Transform Window | ✅ | ✅ | ✅ |
| Finite Impulse Response Filter | ✅ | ❌ | ❌ |
| Local Response Normalization (LRN) | ✅ | ✅ | ✅ (CUDA only) |
| Matrix Multiplication | ✅ | ✅ | ✅ |
| Max Pooling | ✅ | ✅ | ✅ (CUDA only) |
| Memory Bandwidth | ✅ | ❌ | ❌ |
| ReLU | ✅ | ✅ | ✅ (CUDA only) |
| Softmax | ✅ | ✅ | ✅ (CUDA only) |
| Wavelet Transform | ✅ | ✅ | ❌ |


## GLobal Prerequisites

### 1. GNU Compiler Collection (GCC) / CLang
In order to compile the file of the project you need to have installed [**GCC**](https://gcc.gnu.org/releases.html) ≥ 7. or [**Clang**](https://releases.llvm.org/download.html) ≥ 6 in order to support  C++17. 
```bash
# Install on Fedora
sudo dnf install gcc-c++ clang

# Install on Ubuntu
sudo apt update
sudo apt install build-essential clang

# Check version
g++ --version
clang --version
```


### 2. CMake
In order to build the project you need to install [**CMake**](https://cmake.org/download) ≥ 3.24 
```bash
  # Install on Fedora
  sudo dnf install cmake
  # Install on Ubuntu
  sudo apt install cmake
  # Check version
  cmake --version
```

### 3. Git

To fetch external dependencies and CMake `FetchContent` modules, you need [**Git**](https://git-scm.com/install/).

```bash
# Install on Fedora
sudo dnf install git

# Install on Ubuntu
sudo apt install git

# Check version
git --version
```

## Specefic Framework Prerequisites

### 1. NVIDIA CUDA (nvcc)

Required for compiling [**CUDA**](https://developer.nvidia.com/cuda-downloads) accelerated benchmarks (`.cu` targets).

>Tested with version: **V13.2.86**

```bash
# Install on Fedora (via RPM Fusion / NVIDIA repo)
sudo dnf install xorg-x11-drv-nvidia-cuda cuda-toolkit

# Install on Ubuntu
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Check version
nvcc --version
```
> **Additionnal lib NVIDIA cuDNN**<br>
> NVIDIA CUDA Deep Neural Network library (cuDNN) is requiried by some of the benchmark. <br>
> Tested with **V9.23.2**, can be downloaded from [NVIDIA website](https://developer.nvidia.com/cudnn-9-23-2-download-archive).

<details>
<summary><b style="font-size: 1.25em;">Installation Details</b></summary>

When downloading the local package installer from the NVIDIA website, select your operating system distribution:
* **Fedora:** Select **Linux** $\rightarrow$ **x86_64** $\rightarrow$ **RHEL** (`.rpm` package) $\rightarrow$  **FULL**.
* **Ubuntu:** Select **Linux** $\rightarrow$ **x86_64** $\rightarrow$ **Ubuntu** or **Debian** (`.deb` package) $\rightarrow$  **FULL**.


### Fedora Installation

```bash
# Download the RPM reposipackagetory for cuDNN 9.23.2
wget https://developer.download.nvidia.com/compute/cudnn/9.23.2/local_installers/cudnn-local-repo-rhel10-9.23.2-1.0-1.x86_64.rpm
# Install the package to register it with the package manager
sudo rpm -i cudnn-local-repo-rhel10-9.23.2-1.0-1.x86_64.rpm
# Clean the DNF package manager 
sudo dnf clean all

# Install the cuDNN 9 library for CUDA 13
sudo dnf -y install cudnn9-cuda-13
```

### Ubuntu Installation

```bash
# Download the local DEB package for cuDNN 9.23.2
wget https://developer.download.nvidia.com/compute/cudnn/9.23.2/local_installers/cudnn-local-repo-ubuntu2404-9.23.2_1.0-1_amd64.deb
# Install the local package to register it with APT
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.23.2_1.0-1_amd64.deb
# Copy the GPG key to authenticate packages
sudo cp /var/cudnn-local-repo-ubuntu2404-9.23.2/cudnn-*-keyring.gpg /usr/share/keyrings/
# Refresh the APT package
sudo apt-get update

# Install the cuDNN 9 library for CUDA 13
sudo apt-get -y install cudnn9-cuda-13
```
</details>


### 2. AMD ROCm (hipcc)

Required for compiling [AMD HIP](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=all&w=graphics&os=ubuntu&ubuntu-ver=26.04) benchmarks (.cpp HIP targets).

>Tested with version: **6.4.43484-9999**

```Bash
# Install on Fedora
sudo dnf install rocm-hip-devel rocm-runtime

# Install on Ubuntu
sudo apt update
sudo apt install hipcc rocm-dev

# Check version
hipcc --version
```

### 3. OpenCL

Required for compiling [**OpenCL**](https://www.khronos.org/opencl/) accelerated benchmarks (`.cpp` OpenCL targets).

> Tested with version: **OpenCL 3.0**

```bash
# Install on Fedora
sudo dnf install opencl-headers ocl-icd-devel clinfo

# Install on Ubuntu
sudo apt update
sudo apt install opencl-headers ocl-icd-opencl-dev clinfo

# Check version and available devices
clinfo
```

> **Additional lib CLBlast**<br>
> The tuned OpenCL BLAS library (CLBlast) is required by some benchmarks. <br>
> Tested with **v1.6.3 and v1.7.0**, can be downloaded from the [CLBlast GitHub Releases](https://github.com/CNugteren/CLBlast/releases).

<details>
<summary><b style="font-size: 1.25em;">Installation Details</b></summary>

### Fedora Installation

```bash
# Install CLBlast development libraries directly via DNF
sudo dnf -y install clblast-devel
```

### Ubuntu Installation

```bash
# Install CLBlast development libraries directly via APT
sudo apt-get -y install libclblast-dev
```
</details>


### 4. OpenMP

Required for multi-threaded CPU parallel execution (`-fopenmp`). 

> Tested with: **OpenMP 4.5 (201511)**

```bash
# Install runtime & development libraries on Fedora
sudo dnf install libgomp

# Install runtime & development libraries on Ubuntu
sudo apt install libomp-dev

# Check supported OpenMP version via compiler macro (_OPENMP)
echo | g++ -fopenmp -dM -E - | grep _OPENMP
```
>**Additional lib OpenBLAS**<br>
> An optimized Basic Linear Algebra Subprograms (BLAS) library required by some benchmarks.<br>
> Tested with **v0.3.x**, can be downloaded from the [OpenBLAS GitHub Releases](https://github.com/OpenMathLib/OpenBLAS/releases).

<details>
<summary><b style="font-size: 1.25em;">Installation Details</b></summary>

### Fedora Installation

```bash
# Install OpenBLAS development libraries directly via DNF
sudo dnf -y install openblas-devel
```

### Ubuntu Installation

```bash
# Install OpenBLAS development libraries directly via APT
sudo apt-get -y install libopenblas-dev
```
</details>




### 5.  **Additional lib FFTW3**<br>
> A C subroutine library for computing the Discrete Fourier Transform (DFT) required by some benchmarks.<br>
> Tested with v3.3.10, can be downloaded from the [FFTW Official Website](http://www.fftw.org/download.html).

<details>
<summary><b style="font-size: 1.25em;">Installation Details</b></summary>

### Fedora Installation

```bash
# Install FFTW3 development libraries directly via DNF
sudo dnf -y install fftw-devel
```

### Ubuntu Installation

```bash
# Install FFTW3 development libraries directly via APT
sudo apt-get -y install libfftw3-dev
```
</details>


## ANDROID specefic prerequisites

### 1. Android NDK r27d (27.3.13750724)

First, you will need to download the [Android NDK](https://github.com/android/ndk/wiki) in your environment to cross-compile for Android targets.

You can use other versions of the Android NDK, but the project was built and successfully tested using **NDK r27d (27.3.13750724)**."

**Recommanded installation instructions:**
1. Download the [Android Command Line Tools](https://developer.android.com/studio#command-line-tools-only)
2. Install them and set `ANDROID_HOME` and `sdkmanager`to your bash path:
```bash
   # Put default Android Studio path 
   # or wherever you installed it
   echo 'export ANDROID_HOME=$HOME/Android/Sdk' >> ~/.bashrc  
   echo 'export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin' >> ~/.bashrc
   source ~/.bashrc #reload bash configuration
```
3. Download your preferred version of the NDK using the `sdkmanager`:
```bash
sdkmanager --install "ndk;27.3.13750724"
```
### 2. Android Debug Bridge (ADB)

In addtion, if you want to push the binaries to your phone and execute them, you should install **ADB (Android Debug Bridge)**. 

```bash
# Ubuntu 
  sudo apt install android-tools-adb
# Fedora
  sudo dnf install android-tools
```

### 3. Additional Android Libs: libOpenCL.so, libclblast.a, libopenblas.a, libfftw3.a, libfftw3_omp.a

Some benchmarks on Android will required several pre-compiled static libraries, headers, and an OpenCL stub.

In release versions, these dependencies are already pre-compiled and included under `./gpu4s_benchmark/common/android/` this version targets the following specifications:

- Target ABIs: arm64-v8a, armeabi-v7a
-  Minimum Android API: 21
- Built With: Android NDK r27d

If you need to target a newer architecture (like ARMv9) or running the main repo, You can use my custom toolchains to download, compile the stub, libraries and headers:
- **[clblast-android-toolchain](https://github.com/EmbeddedFrime/clblast-android-toolchain)** (`libclblast.a` and `libOpenCL.so` stub)
- **[openblas-android-toolchain](https://github.com/EmbeddedFrime/openblas-android-toolchain)** (`cblash.h,` `openblas_config.h`, and `libopenblas.a`)
- **[fftw-android-toolchain](https://github.com/EmbeddedFrime/fftw-android-toolchain)** (`libfftw3.a` and `libfftw3_omp.a`)

After having recompiled the libraries, you need to replace the corresponding files in the common/android/ directory with the new ones:

<pre>
common/android/
├── include/
│   ├── arm64-v8a/
│   │   ├── cblas.h <span style="color:red">*</span>
│   │   └── openblas_config.h <span style="color:red">*</span>
│   └── armeabi-v7a/
│       ├── cblas.h <span style="color:red">*</span>
│       └── openblas_config.h <span style="color:red">*</span>
└── libs/
    ├── arm64-v8a/
    │   ├── libOpenCL.so <span style="color:blue">*</span>
    │   ├── libclblast.a <span style="color:blue">*</span>
    │   ├── libfftw3_omp.a <span style="color:green">*</span>
    │   ├── libfftw3.a <span style="color:green">*</span>
    │   └── libopenblas.a <span style="color:red">*</span>
    └── armeabi-v7a/
        ├── libOpenCL.so <span style="color:blue">*</span>
        ├── libopenblas.a <span style="color:blue">*</span>
        ├── libfftw3_omp.a <span style="color:green">*</span>
        ├── libfftw3.a <span style="color:green">*</span>
        └── libopenblas.so <span style="color:red">*</span>

OpenBLAS toolchain<span style="color:red">*</span>  CLBlast toolchain<span style="color:blue">*</span>  fftw3 toolchain<span style="color:green">*</span>
</pre>


## Build Instructions

GPU4S Bench can be built with 2 different methods:

- **globally** (all 17 benchmarks at once, from the gpu4s root)
- **standalone** (single benchmark, from its own subfolder). 

Both follow the same  steps: configure and build. The executables are saved in `build/bin/` (or `build-android/bin/`).

### 1. Generate build files

#### Host (x86_64 / Linux)
```bash
cmake -B build
```

#### Android
```bash
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/27.3.13750724/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-21
```

#### You can choose any build directory name, in order to handle your different configurations easily:

```bash
cmake -B build-float-256 -DDATATYPE=FLOAT -DBLOCKSIZE=256
cmake -B build-double-128 -DDATATYPE=DOUBLE -DBLOCKSIZE=128
#et cetera ...
```

### 2. Build every targets

```bash
cmake --build build -j$(nproc)            # host
cmake --build build-android -j$(nproc)    # android
```

>**Notes**: The option `-j$(nproc)` forces your CPU to use all of his core to speed up the process.

### 3. Build a specific target (optional)

```bash
# Standalone (e.g.: from inside matrix_multiplication_bench/ directory)
cmake --build build --target cpu
cmake --build build --target all-opencl
cmake --build build --target cuda-opt

# Global (from gpu4s_benchmark/ root)
cmake --build build --target matrix_mult-cpu
cmake --build build --target matrix_mult-all-opencl
cmake --build build --target matrix_mult-cuda-opt
```
>**Notes**: The option `--target` can be used to select a specefic target.


<details>
<summary><b style="font-size: 1.25em;">📋 List of Target Shortcuts</b></summary>

#### lists of all possible Frameworks Shortcuts :
- **CPU**: `cpu`, `CPU`
- **HIP**: `hip-`, `HIP`, `hip-opt`, `HIP-opt`, `all-hip`
- **CUDA**: `cuda`, `CUDA`, `cuda-opt`, `CUDA-opt`, `cuda-lib`, `CUDA-lib`, `all-cuda`
- **OpenCL**: `cl`, `OpenCL`, `opencl-opt`, `OpenCL-opt`, `opencl-lib`, `OpenCL-lib`, `all-opencl`
- **OpenMP**: `openmp`, `OpenMP`, `openmp-opt`, `OpenMP-opt`, `openmp-lib`, `OpenMP-lib`, `all-openmp`


</details>

### 4. Compile options

You can customize compilation options during the configuration step by adding `-D<PARAMETER>=<VALUE>` to `cmake -B ...`:

#### Option 1 - Change the Data Type

```bash
# e.g. Use double precision instead of float
cmake -B build -DDATATYPE=DOUBLE
```

#### Option 2 - Change the Block/Tile Size

```bash
# e.g. Use a 32x32 GPU kernel tile size
cmake -B build -DBLOCKSIZE=32
```

#### Option 3 - Change the Host Optimization Level

```bash
# e.g. Compile with aggressive optimizations
cmake -B build -DOPT_FLAG=-Ofast
```

#### Option 4 - Change the CUDA Architecture

```bash
# e.g. Target NVIDIA Ampere (RTX 30-series)
cmake -B build -DCUDA_ARCH=sm_86
```

#### Option 5 - Change the OpenCL Target Version

```bash
# e.g. Target OpenCL 2.0 instead of 3.0
cmake -B build -DOPENCL_VERSION=200
```

#### Option 6 - Change the BLAS Vendor

```bash
# e.g. Use ATLAS instead of OpenBLAS for OpenMP-lib
cmake -B build -DBLA_VENDOR=ATLAS
```

#### Option 7 - Change the Number of Streams (cifar_10_multiple only)

```bash
# e.g. Run with 8 concurrent streams
cmake -B build -DNSTREAMS=8
```



<details>
<summary><b style="font-size: 1.25em;">Advanced example + table </b></summary>

#### Full Customization Example (Multiple Parameters)

You can chain multiple `-D` flags together to fully customize data type, tuning, and target hardware in a single command.

For example, to build a double-precision version with a 32x32 block size targeting an Ampere GPU with -0fast optimizations:

```bash
cmake -B build \
  -DDATATYPE=DOUBLE \
  -DBLOCKSIZE=32 \
  -DCUDA_ARCH=sm_86 \
  -DOPT_FLAG=-Ofast
```

| Parameter | Default | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `DATATYPE` | `FLOAT` | `FLOAT`, `DOUBLE`, `INT` | Data type used in computations |
| `BLOCKSIZE` | `16` | `4`, `8`, `16`, `32`, etc. | 2D tile/block size for GPU kernels |
| `OPT_FLAG` | `-O3` | `-O2`, `-O3`, `-Ofast` | Host compiler optimization level |
| `CUDA_ARCH` | `native` | `native`, `sm_70`, `sm_75`, `sm_80`, `sm_86`, etc. | NVIDIA GPU compute architecture |
| `OPENCL_VERSION` | `300` | `200`, `210`, `220`, `300`, `310` | OpenCL target API version |
| `BLA_VENDOR` | `OpenBLAS` | `OpenBLAS`, `ATLAS`, `Generic` | BLAS library vendor for OpenMP-lib |
| `NSTREAMS` | `4` | Integer $\ge 1$ | Number of concurrent streams (`cifar_10_multiple`) |
</details>

#### Cleaning Up the Workspace

In order to reset your build environment, you can remove all generated build directories and start fresh:

```bash
rm -rf build*
```


### 4. Deprecated make build Instructions

> **Deprecated Build Instructions**<br>
> The Makefile-based build system is **deprecated** and is retained only for legacy compatibility.
> **Please use the CMake build system instead**, as it provides a cleaner, more maintainable, and cross-platform workflow.

<details>
<summary><b style="font-size: 1.25em;">Legacy Makefile Build (Deprecated)</b></summary>

### ⚠️ Legacy Makefile Workflow

### 1. Compile the code using the provided Makefile

#### PC / Laptop Build (Fedora Host)

```bash
# You must provide the data type and block size
# Compile individual backend
make cpu DATATYPE=FLOAT BLOCKSIZE=16
# Compile all opencl frameworks backends 
make all-opencl DATATYPE=FLOAT BLOCKSIZE=16
# Compile all backends simultaneously 
make all-bin DATATYPE=FLOAT BLOCKSIZE=16
```

#### Android Cross-Compilation Build
> **⚠️ Only work for matrix multiplication benchmark**


```bash
# Compile individual backend
make android_cpu DATATYPE=FLOAT BLOCKSIZE=16
# Compile all backends simultaneously 
make all-android DATATYPE=FLOAT BLOCKSIZE=16
```
</details>



## Execution Guide

Once compiled, the executables are located in the `build/bin/` (for host) or `build-android/bin/` (for Android) directories. 


### 1. Running on Host (Linux / PC)

To execute a benchmark on your local machine, simply run the generated binary from the terminal. 

```bash
# General syntax
./build/bin/<target_name> [arguments]

# Example: Running the standard CPU matrix multiplication
./build/bin/matrix_mult_cpu -s 1024

# Example: Running the optimized CUDA version
./build/bin/matrix_mult_cuda_opt -s 2048

```

### 2. Running on Host (Linux / PC)

In mobile it's the same command but first you need to push the file to your phone and after that you can execute it trough adb shell. 
```bash
# Push all binaries to the device
adb push ./build-android/bin/* /data/local/tmp/
# Grant execution permissions to the binaries
adb shell chmod 755 /data/local/tmp/*
# Execute the benchmarks on the Android device
adb shell /data/local/tmp/matrix_mult_opencl -s 1024 -t -v
```



### 3. Runtime Parameters

All benchmarks in the GPU4S suite share a common command-line interface. You must provide the problem size (`-s`), while all other parameters are optional and allow you to configure memory management, validation, and output formatting.

| Flag | Name | Description |
| :--- | :---: | :--- |
| `-s <size>` | **Size** | Sets the X and Y dimensions for the benchmark (e.g., matrix size $N \times N$). |
| `-v` | **Verify** | Executes a CPU baseline calculation and compares it with the GPU output to verify correctness. |
| `-t` | **Timing** | Prints the elapsed execution time to the console. |
| `-c` / `-C` | **CSV format** | Prints timing results in CSV format. Use `-C` to include a timestamp. |
| `-u` | **Unified Memory** | Enables Unified Memory mapping (Zero-Copy). **Highly recommended for Android and Jetson** platforms to prevent unnecessary memory transfers. |
| `-p` | **Profiling** | Enables internal clock profiling for more granular hardware timing. |
| `-d <id>` | **Device ID** | Selects the specific GPU device ID to use (default is usually 0). |
| `-o` | **Print Output** | Prints the resulting matrix or array directly to the terminal. |
| `-f` | **Mute** | Mutes all standard print messages (useful for batch testing scripts). |
| `-i <fileA> <fileB>`| **Input Files** | Loads custom hexadecimal input files instead of using randomized data generation. |
| `-g` / `-e` | **Export** | `-g` exports the GPU result to `gpu_file.out`. `-e` exports both GPU and CPU results in hex format (automatically enables `-v`). |
| `-k <size>` | **Kernel Size** | *(Specific to `convolution_2D`, `FIR_filter`)* Defines the size of the convolution filter / mask ($K \times K$) or FIR tap length. |
| `-l <stride>`| **Stride Size** | *(Specific to `max_pooling`)* Defines the horizontal and vertical step size of the pooling sliding window. |

## To do list:
- [X] fix issue of correctness between cpu and gpu in some benchmark
- [X] refactor to extract the common of frameworks
- [X] fix of the clock to be executed in runtime + kernerCLK->deviceOBJ
- [X] Check for cl error dunring memory copy to host and clean
- [X] add UMA implementation for Android
- [X] documented this fucntion I have added in the code
- [X] Big cmake to compile everything 
- [ ] be compatible with jetson board + add UMA for jetson board

- [X] Tuto to install every deps(HIP,CUDA,OpenMP,OpenBLAS,CLBlast,FFTW3,OpenCL,CUDNN...)
- [ ] Tuto how to use the benchmark(param etc) + compile with cmake
- [ ] fix matrix_mult_FP16/Tensor/Memory_bandwith -> add timestamp csv format
- [ ] add map for verification of the result (ANDROID UMA)
- [ ] refactor main, cuda, hip, opencl, OpenMP, -> create common 




**Bonus**:
- [ ] create a test with vulkan for android to have best perfomance (with softmax ?)
