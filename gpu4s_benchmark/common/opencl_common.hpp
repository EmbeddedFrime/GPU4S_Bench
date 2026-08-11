/** * ====================================================================
 * @file        lib_opencl_common.h
 * @brief       Shared data structures, macros, and universal helpers 
 *              for hardware acceleration benchmarks.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
#include "benchmark_common.h"

// ============ global opencl function ============
// nothing for now



#ifdef UMA_COMPATIBILITY
// ============ UMA struct ============

/**
 * @brief Encapsulates a mapping association between a host pointer address,
 *       an OpenCL device buffer, and an event handle.
 * 
 */
struct BufferMapCL {
    bench_t**   hostBuffer;   /**< Pointer to the host-side memory pointer address */
    cl::Buffer* deviceBuffer; /**< Pointer to the OpenCL device buffer object */
    cl::Event*  deviceEvent;  /**< Pointer to the OpenCL device event object used for profiling timing */
};

// ============ UMA function ============

/**
 * @brief Maps device memory buffers into the host's virtual address space,
 *        granting the CPU direct write access to the shared memory.
 * 
 * @tparam MapCL 
 * @param device_object 
 * @param memSize 
 * @param mapCL 
 */
template <typename... MapCL>
inline void map_unified_memory(GraficCommon* device_object, unsigned memSize, MapCL... mapCL) {
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    Clock mapCLK;
    mapCLK.start();

    // --- C++17 Fold Expression Unrolled at compile-time  ---
    // For each MapCL map host buffer to devcie buffer
    ((*(mapCL.hostBuffer) = static_cast<bench_t*>(
        deviceObj->queue->enqueueMapBuffer(
            *(mapCL.deviceBuffer), CL_TRUE, CL_MAP_WRITE, 0, memSize, nullptr, mapCL.deviceEvent)
    )), ...);

    // equivalent but less optimized
    // for (const auto& item : { mapCL... }) {
    //     *(item.host_buffer) = static_cast<bench_t*>(
    //         deviceObj->queue->enqueueMapBuffer(*(item.device_clBuffer), CL_TRUE, CL_MAP_WRITE, 0, memSize)
    //     );
    // }

    mapCLK.end();
    deviceObj->h2d_elapsed_time = mapCLK.getElapsedNS();
}

/**
 * @brief Give CPU control of the shared memory back to the GPU, 
 *        ensuring the device has exclusive access to the buffers 
 * 
 * @tparam MapCL 
 * @param device_object 
 * @param mapCL 
 */
template <typename... MapCL>
inline void unmap_unified_memory(GraficCommon* device_object, MapCL... mapCL) {
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    Clock unmapCLK;
    unmapCLK.start();

    // --- C++17 Fold Expression Unrolled at compile-time  ---
    // For each MapCL unmap host buffer to devcie buffer
    ((deviceObj->queue->enqueueUnmapMemObject(
        *(mapCL.deviceBuffer), mapCL.hostBuffer, NULL, mapCL.deviceEvent)
    ), ...);

    unmapCLK.end();
    deviceObj->d2h_elapsed_time = unmapCLK.getElapsedNS();
}

/**
 * @brief Synchronizes the output memory back to the host,
 *        give CPU direct access to the GPU's results.
 * 
 * @tparam device_object 
 * @param d_C 
 * @param buff_size 
 */
template <typename... MapCL>
inline void map_unified_memory_to_host(GraficCommon* device_object, unsigned memSize, MapCL... mapCL) {
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    Clock mapCLK;
    mapCLK.start();

    // --- C++17 Fold Expression Unrolled at compile-time  ---
    // For each MapCL map host buffer to devcie buffer
    ((*(mapCL.hostBuffer) = static_cast<bench_t*>(
        deviceObj->queue->enqueueMapBuffer(
            *(mapCL.deviceBuffer), CL_TRUE, CL_MAP_WRITE, 0, memSize, nullptr, mapCL.deviceEvent)
    )), ...);

    mapCLK.end();
    deviceObj->d2h_elapsed_time += mapCLK.getElapsedNS();
}
#endif


