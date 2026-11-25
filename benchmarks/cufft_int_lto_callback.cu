/*
 * Project: RapiDHT
 * File: benchmarks/cufft_int_lto_callback.cu
 * Brief: Load callbacks for cuFFT's LTO interface: uint8 / uint16 to float.
 *
 * Not compiled into any binary. benchmarks/CMakeLists.txt turns this file into
 * LTO-IR (nvcc -dlto -dc -fatbin) and then into a byte array (bin2c) that
 * cufft_baseline.cu hands to cufftXtSetJITCallback; cuFFT links the callback
 * into its own kernels when the plan is made, by the name it is given. That
 * name is the C++ one: cuFFT declares the callback in code it compiles itself as
 * C++, so an extern "C" callback is an undefined symbol to it and the plan
 * fails with CUFFT_NVJITLINK_FAILURE (tried, 2026-11-25).
 *
 * callerInfo points to one float in device memory: the offset subtracted from
 * every voxel. dataIn is the integer volume itself, passed to cufftExecR2C in
 * place of a float one; offset counts voxels.
 */

#include <cufftXt.h>

#include <cstdint>

__device__ cufftReal rapidht_load_u8_lto(
    void* dataIn, unsigned long long offset, void* callerInfo, void* /*sharedPointer*/)
{
    return static_cast<float>(static_cast<const std::uint8_t*>(dataIn)[offset])
         - *static_cast<const float*>(callerInfo);
}

__device__ cufftReal rapidht_load_u16_lto(
    void* dataIn, unsigned long long offset, void* callerInfo, void* /*sharedPointer*/)
{
    return static_cast<float>(static_cast<const std::uint16_t*>(dataIn)[offset])
         - *static_cast<const float*>(callerInfo);
}
