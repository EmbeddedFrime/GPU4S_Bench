/** * ====================================================================
 * @file        cuda_common.h (./matrix_multiplication_tensor_bench)
 * @brief       Shared declarations, data structures, and timing utilities
 *              for CUDA benchmark backends.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once

#ifdef PROFILING_CLOCK
    // kernel time execution
    Clock kernelCLK;
    // host <-> device 
    Clock h2dCLK;
    Clock d2hCLK;
#endif

