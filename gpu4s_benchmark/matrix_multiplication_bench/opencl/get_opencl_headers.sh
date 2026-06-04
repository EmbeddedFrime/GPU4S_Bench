#!/bin/bash
# Exit immediately if any command fails
set -e
#pull the latest release tag from the OpenCL-Headers repository
LATEST_TAG_H=$(curl -s https://api.github.com/repos/KhronosGroup/OpenCL-Headers/releases/latest | grep -oP '"tag_name": "\K[^"]+')

# Download OpenCL repository for header files
git clone -b "$LATEST_TAG_H" --depth 1 -c advice.detachedHead=false https://github.com/KhronosGroup/OpenCL-Headers.git ./opencl/tmp

# Rm & Move the CL directory with openCL.h to the opencl_headers directory
rm -rf ./opencl/opencl_headers/CL && mkdir -p ./opencl/opencl_headers/CL && mv ./opencl/tmp/CL/* ./opencl/opencl_headers/CL

#pull the latest release tag from the OpenCL-HPP repository
LATEST_TAG_HPP=$(curl -s https://api.github.com/repos/KhronosGroup/OpenCL-CLHPP/releases/latest | grep -oP '"tag_name": "\K[^"]+')

# Download the opencl.hpp header file
curl -o ./opencl/opencl_headers/CL/opencl.hpp https://raw.githubusercontent.com/KhronosGroup/OpenCL-CLHPP/${LATEST_TAG_HPP}/include/CL/opencl.hpp

#For old version of openCL :
#curl -o ./opencl/opencl_headers/CL/opencl.hpp https://raw.githubusercontent.com/#KhronosGroup/OpenCL-CLHPP/${LATEST_TAG_HPP}/include/CL/cl2.hpp

# Remove the temporary directory
rm -rf ./opencl/tmp
