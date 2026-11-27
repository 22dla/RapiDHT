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
 *
 * R-5 step 4.3: store callbacks for the inverse (C2R) into integers. callerInfo
 * points to one float, the scale 1/N; each value is scaled, rounded to nearest
 * and clamped to the type -- rint(fma(x, 1/N, 0)), the conversion of this
 * library's integer inverse with no offset -- and written to dataOut, the
 * integer volume passed to cufftExecC2R in place of a float one.
 *
 * R-5 step 5.1: the same with an offset, for the Gaussian filter of task T2,
 * which subtracts the offset on reading and adds it back on writing.
 * callerInfo points to two floats, { 1/N, offset }: rint(fma(x, 1/N, offset)),
 * clamped. The load callback of that plan gets a pointer to the second.
 */

#include <cufftXt.h>

#include <cstdint>

__device__ cufftReal rapidht_load_u8_lto(
    void* dataIn, unsigned long long offset, void* callerInfo, void* /*sharedPointer*/)
{
    return static_cast<float>(static_cast<const std::uint8_t*>(dataIn)[offset])
         - *static_cast<const float*>(callerInfo);
}

template <typename TOut>
__device__ __forceinline__ void StoreInteger(void* dataOut, unsigned long long offset, cufftReal element, float scale)
{
    const float r = rintf(fmaf(element, scale, 0.0f));
    const float top = sizeof(TOut) == 1 ? 255.0f : 65535.0f;
    static_cast<TOut*>(dataOut)[offset] = static_cast<TOut>(r >= 0.0f ? (r <= top ? r : top) : 0.0f);
}

__device__ void rapidht_store_u8_lto(
    void* dataOut, unsigned long long offset, cufftReal element, void* callerInfo, void* /*sharedPointer*/)
{
    StoreInteger<std::uint8_t>(dataOut, offset, element, *static_cast<const float*>(callerInfo));
}

__device__ void rapidht_store_u16_lto(
    void* dataOut, unsigned long long offset, cufftReal element, void* callerInfo, void* /*sharedPointer*/)
{
    StoreInteger<std::uint16_t>(dataOut, offset, element, *static_cast<const float*>(callerInfo));
}

__device__ cufftReal rapidht_load_u16_lto(
    void* dataIn, unsigned long long offset, void* callerInfo, void* /*sharedPointer*/)
{
    return static_cast<float>(static_cast<const std::uint16_t*>(dataIn)[offset])
         - *static_cast<const float*>(callerInfo);
}

__device__ void rapidht_store_u8_offset_lto(
    void* dataOut, unsigned long long offset, cufftReal element, void* callerInfo, void* /*sharedPointer*/)
{
    const float* p = static_cast<const float*>(callerInfo);
    const float r = rintf(fmaf(element, p[0], p[1]));
    static_cast<std::uint8_t*>(dataOut)[offset] = static_cast<std::uint8_t>(r >= 0.0f ? (r <= 255.0f ? r : 255.0f) : 0.0f);
}

__device__ void rapidht_store_u16_offset_lto(
    void* dataOut, unsigned long long offset, cufftReal element, void* callerInfo, void* /*sharedPointer*/)
{
    const float* p = static_cast<const float*>(callerInfo);
    const float r = rintf(fmaf(element, p[0], p[1]));
    static_cast<std::uint16_t*>(dataOut)[offset]
        = static_cast<std::uint16_t>(r >= 0.0f ? (r <= 65535.0f ? r : 65535.0f) : 0.0f);
}
