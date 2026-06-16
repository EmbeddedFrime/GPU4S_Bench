cmake_minimum_required(VERSION 3.16)
project(matrix_mult CXX) 


# ====== CMake Configuration ======

# --- Global Settings ---
set(CMAKE_CXX_STANDARD 17)    # force clang 17 or g++ 17 compiler
set(CMAKE_BUILD_TYPE Release) # add production-level optimization

# --- Global Variable ---
set(OPENCL_VERSION  300   CACHE STRING "API OpenCL Version : 200, 210, 220, 300, 310")
set(BLOCKSIZE       4      CACHE STRING "Block size for tiled kernels")
set(DATATYPE       "FLOAT" CACHE STRING "Data type: FLOAT, DOUBLE, INT")
set(COMPUTER true)

# --- Android Variable ---
if(ANDROID)
    set(ANDROID_INC ${CMAKE_SOURCE_DIR}/android/include/)
    set(ANDROID_LIB ${CMAKE_SOURCE_DIR}/android/libs/)
    set(COMPUTER false)
endif(ANDROID)

# --- Define Output directory ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)




# ====== Package Handler ======
if(COMPUTER)
    # Searching for package installation
    find_package(OpenCL QUIET) 
    find_package(CLBlast QUIET) 
endif(COMPUTER)




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
target_compile_options(${PROJECT_NAME}_cpu PRIVATE -O3)

add_custom_target(CPU DEPENDS ${PROJECT_NAME}_cpu)
add_custom_target(cpu DEPENDS ${PROJECT_NAME}_cpu)


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

    target_compile_options(${PROJECT_NAME}_opencl PRIVATE -O3)

    if(ANDROID)
        # --- specefic android opencl include and libs ---
        target_include_directories(${PROJECT_NAME}_opencl PRIVATE ${ANDROID_INC})
        target_link_libraries(${PROJECT_NAME}_opencl PRIVATE ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so)

    elseif(OpenCL_FOUND)
        # specefic computer opencl libs
        target_link_libraries(${PROJECT_NAME}_opencl PRIVATE OpenCL::OpenCL)
    endif()

    # --- Shortcuts ---
    add_custom_target(opencl DEPENDS ${PROJECT_NAME}_opencl)
    add_custom_target(Opencl DEPENDS ${PROJECT_NAME}_opencl)
    add_custom_target(openCL DEPENDS ${PROJECT_NAME}_opencl)
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

    target_compile_options(${PROJECT_NAME}_opencl_opt PRIVATE -O3)

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
    add_custom_target(Opencl-opt DEPENDS ${PROJECT_NAME}_opencl_opt)
    add_custom_target(openCL-opt DEPENDS ${PROJECT_NAME}_opencl_opt)
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

    target_compile_options(${PROJECT_NAME}_opencl_lib PRIVATE -O3)

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
    add_custom_target(Opencl-lib DEPENDS ${PROJECT_NAME}_opencl_lib)
    add_custom_target(openCL-lib DEPENDS ${PROJECT_NAME}_opencl_lib)
    add_custom_target(OpenCL-lib DEPENDS ${PROJECT_NAME}_opencl_lib)

else ()
    # Fallback warning message 
    if(OpenCL_FOUND)
        set(MISING_LIB "CLBlast")
    elseif(CLBlast_FOUND)
        set(MISING_LIB "OpenCL")
    endif()
        message(WARNING "${MISING_LIB} installation was not found on this system. Skipping OpenCL-lib target.")
endif()


    







