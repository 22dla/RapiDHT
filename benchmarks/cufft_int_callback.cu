/*
 * Project: RapiDHT
 * File: benchmarks/cufft_int_callback.cu
 * Brief: Variant (b) of the integer cuFFT baseline: the legacy load callback.
 *
 * Legacy callbacks are device functions whose addresses cuFFT calls from its
 * own kernels, so the callback has to be device-linked with cuFFT itself: they
 * work only with the static cuFFT (libcufft_static) and separate compilation.
 * That cuFFT must not meet the shared one the other baselines use, so this file
 * is built alone into a shared library, rapidht_cufft_callback, which links the
 * static cuFFT and exports nothing of it (benchmarks/CMakeLists.txt) -- only
 * MakeCufftIntCallback. The rows measured through it are cuFFTIntCb/.
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <cstdint>
#include <memory>

#include "cufft_baseline.h"
#include "cufft_common.cuh"

namespace rapidht_bench {
namespace {

// callerInfo points to one float in device memory: the offset.
__device__ cufftReal LoadU8(void* dataIn, size_t offset, void* callerInfo, void* /*sharedPointer*/)
{
    return static_cast<float>(static_cast<const std::uint8_t*>(dataIn)[offset])
         - *static_cast<const float*>(callerInfo);
}

__device__ cufftReal LoadU16(void* dataIn, size_t offset, void* callerInfo, void* /*sharedPointer*/)
{
    return static_cast<float>(static_cast<const std::uint16_t*>(dataIn)[offset])
         - *static_cast<const float*>(callerInfo);
}

// cufftXtSetCallback wants the device address of the function, which the host
// can only read from a device variable.
__device__ cufftCallbackLoadR loadU8Pointer = LoadU8;
__device__ cufftCallbackLoadR loadU16Pointer = LoadU16;

class CufftIntCallback final : public CufftIntBase {
public:
    CufftIntCallback(int width, int height, int depth, IntInput input, double offset):
        CufftIntBase(width, height, depth, input, offset)
    {
        MakePlan();
        const float hostOffset = static_cast<float>(offset);
        Check(cudaMalloc(&_callerInfo, sizeof(float)), "cudaMalloc(callerInfo)");
        Check(cudaMemcpy(_callerInfo, &hostOffset, sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(callerInfo)");
        cufftCallbackLoadR load = nullptr;
        if (input == IntInput::U8) {
            Check(cudaMemcpyFromSymbol(&load, loadU8Pointer, sizeof(load)), "cudaMemcpyFromSymbol");
        } else {
            Check(cudaMemcpyFromSymbol(&load, loadU16Pointer, sizeof(load)), "cudaMemcpyFromSymbol");
        }
        void* routine = reinterpret_cast<void*>(load);
        void* info = _callerInfo;
        Check(cufftXtSetCallback(_plan, &routine, CUFFT_CB_LD_REAL, &info), "cufftXtSetCallback");
    }

    ~CufftIntCallback() override
    {
        if (_callerInfo != nullptr) {
            cudaFree(_callerInfo);
        }
    }

    void Forward(const void* deviceIn, float* deviceOut) override { TransformAndConvert(deviceIn, deviceOut); }

private:
    float* _callerInfo = nullptr;
};

} // namespace

__attribute__((visibility("default"))) std::unique_ptr<CufftIntHartley> MakeCufftIntCallback(
    int width, int height, int depth, IntInput input, double offset)
{
    return std::make_unique<CufftIntCallback>(width, height, depth, input, offset);
}

} // namespace rapidht_bench
