#ifndef GMX_EWALD_CUH
#define GMX_EWALD_CUH
#include <stdio.h>

#include "gromacs/gpu_utils/gputraits.cuh"

extern "C" cudaError_t generate_random_number();
extern "C" cudaError_t Memcy(float* c, const float* a, const float* b);
extern "C" cudaError_t cal_force();
extern "C" cudaError_t cal_rho();

#endif
