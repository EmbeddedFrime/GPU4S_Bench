# ====== Blueprint function ======
#     Compile all the targets
# ===============================
function(compile_target TARGET_NAME)

# Parse arguments
set(singleArgs   BENCH_DIR SET_CUDA) 
set(multipleArgs SOURCES_FILES SET_HIP_FILES COMPILE_DEFS SHORTCUTS_NAMES LIBRARIES INCLUDES COMPILE_OPTIONS) 

cmake_parse_arguments(ARG "" "${singleArgs}" "${multipleArgs}" ${ARGN})

    # --- add bench dir to src dir ---
    foreach(SRC IN LISTS ARG_SOURCES_FILES)
        list(APPEND ABSOLUTE_SOURCES "${ARG_BENCH_DIR}/${SRC}")
    endforeach()

    # --- Add all source file ---
    add_executable(${TARGET_NAME}
        ${ARG_BENCH_DIR}/main.cpp
        ${ABSOLUTE_SOURCES}
        ${ARG_BENCH_DIR}/cpu_functions/cpu_functions.cpp
    )

    # --- Set specefic files as HIP API files ---
    if(ARG_SET_HIP_FILES)
        foreach(HIP_FILE IN LISTS ARG_SET_HIP_FILES)
            set_source_files_properties(${ARG_BENCH_DIR}/${HIP_FILE} PROPERTIES LANGUAGE HIP)
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