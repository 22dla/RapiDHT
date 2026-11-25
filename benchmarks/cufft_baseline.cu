/*
 * Project: RapiDHT
 * File: benchmarks/cufft_baseline.cu
 * Brief: The 2D and 3D Hartley transform via cuFFT, for comparison; from float
 *        input, and (R-5) from uint8 / uint16 input.
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cufft_baseline.h"
#include "cufft_common.cuh"
// The LTO-IR of the load callbacks (cufft_int_lto_callback.cu), as a byte
// array: built by nvcc -dlto -fatbin and bin2c, see benchmarks/CMakeLists.txt.
#include "cufft_int_lto_fatbin.h"

namespace rapidht_bench {
namespace {

/*
 * Integer to float, minus the offset: the first pass of variant (a). Four
 * voxels a thread, read as one ushort4 / uchar4 and written as one float4, so
 * that the pass runs at the bandwidth of the card -- it is the part of the
 * baseline a load callback is meant to save, and it should not be slower than
 * it has to be. W is a power of two of at least 4, so the count divides by 4.
 */
template <typename TIn, typename TIn4>
__global__ void IntToFloat(const TIn4* __restrict__ in, float4* __restrict__ out, size_t quads, float offset)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= quads) {
        return;
    }
    const TIn4 v = in[i];
    out[i] = make_float4(static_cast<float>(v.x) - offset, static_cast<float>(v.y) - offset,
        static_cast<float>(v.z) - offset, static_cast<float>(v.w) - offset);
}

__global__ void AddToFirstKernel(float* data, double value)
{
    data[0] = static_cast<float>(static_cast<double>(data[0]) + value);
}

} // namespace

void ConvertIntToFloat(const void* deviceIn, IntInput input, float* deviceOut, size_t count, double offset)
{
    const size_t quads = count / 4;
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((quads + block - 1) / block);
    const float off = static_cast<float>(offset);
    if (input == IntInput::U8) {
        IntToFloat<std::uint8_t>
            <<<grid, block>>>(static_cast<const uchar4*>(deviceIn), reinterpret_cast<float4*>(deviceOut), quads, off);
    } else {
        IntToFloat<std::uint16_t>
            <<<grid, block>>>(static_cast<const ushort4*>(deviceIn), reinterpret_cast<float4*>(deviceOut), quads, off);
    }
    Check(cudaGetLastError(), "IntToFloat launch");
}

namespace {

/*
 * Sum of every step-th line, exact: a fixed grid of blocks strides over the
 * sampled lines, each thread adds the quads it reads into a 64-bit register,
 * the warp and then the block reduce, one atomicAdd a block. Memory-bound for
 * step 1 (one read of the volume); for a sparse sample it is the launch.
 */
template <typename TIn4>
__global__ void SumLines(
    const TIn4* __restrict__ in, size_t quadsPerLine, size_t sampled, size_t step, unsigned long long* sum)
{
    unsigned long long local = 0;
    const size_t threads = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t total = sampled * quadsPerLine;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += threads) {
        const size_t line = (i / quadsPerLine) * step;
        const TIn4 v = in[line * quadsPerLine + i % quadsPerLine];
        local += static_cast<unsigned long long>(v.x) + v.y + v.z + v.w;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        local += __shfl_down_sync(0xffffffffu, local, offset);
    }
    __shared__ unsigned long long warps[32];
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    if (lane == 0) {
        warps[warp] = local;
    }
    __syncthreads();
    if (warp == 0) {
        local = lane < static_cast<int>(blockDim.x) / 32 ? warps[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            local += __shfl_down_sync(0xffffffffu, local, offset);
        }
        if (lane == 0) {
            atomicAdd(sum, local);
        }
    }
}

__global__ void RoundedMean(const unsigned long long* sum, double count, float* offset)
{
    *offset = static_cast<float>(rint(static_cast<double>(*sum) / count));
}

} // namespace

void IntegerMeanOffset(const void* deviceIn, IntInput input, size_t width, size_t lines, size_t step,
    unsigned long long* deviceSum, float* deviceOffset)
{
    const size_t quadsPerLine = width / 4;
    const size_t sampled = (lines + step - 1) / step;
    int device = 0;
    int sms = 0;
    Check(cudaGetDevice(&device), "cudaGetDevice");
    Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "cudaDeviceGetAttribute");
    const unsigned block = 256;
    // Enough blocks to fill the card (8 a SM), never more than there are quads.
    const size_t wanted = (sampled * quadsPerLine + block - 1) / block;
    const unsigned grid = static_cast<unsigned>(std::min<size_t>(wanted, static_cast<size_t>(sms) * 8));
    Check(cudaMemsetAsync(deviceSum, 0, sizeof(unsigned long long)), "cudaMemsetAsync(sum)");
    if (input == IntInput::U8) {
        SumLines<<<grid, block>>>(static_cast<const uchar4*>(deviceIn), quadsPerLine, sampled, step, deviceSum);
    } else {
        SumLines<<<grid, block>>>(static_cast<const ushort4*>(deviceIn), quadsPerLine, sampled, step, deviceSum);
    }
    Check(cudaGetLastError(), "SumLines launch");
    RoundedMean<<<1, 1>>>(deviceSum, static_cast<double>(sampled * width), deviceOffset);
    Check(cudaGetLastError(), "RoundedMean launch");
}

void AddToFirst(float* deviceData, double value)
{
    AddToFirstKernel<<<1, 1>>>(deviceData, value);
    Check(cudaGetLastError(), "AddToFirst launch");
}

namespace {

/// (a) Conversion kernel into the caller's float output, R2C from there.
class CufftIntConvert final : public CufftIntBase {
public:
    CufftIntConvert(int width, int height, int depth, IntInput input, double offset):
        CufftIntBase(width, height, depth, input, offset)
    {
        MakePlan();
    }

    void Forward(const void* deviceIn, float* deviceOut) override
    {
        ConvertIntToFloat(deviceIn, _input, deviceOut, Count(), _offset);
        // The output buffer holds the float input until R2C has read it, then
        // the Hartley pass overwrites it with the result.
        TransformAndConvert(deviceOut, deviceOut);
    }
};

/// (c) LTO load callback: cuFFT links it into its own kernels when the plan is
/// made, so it has to be attached before cufftMakePlan3d.
class CufftIntLto final : public CufftIntBase {
public:
    CufftIntLto(int width, int height, int depth, IntInput input, double offset):
        CufftIntBase(width, height, depth, input, offset)
    {
        const float hostOffset = static_cast<float>(offset);
        Check(cudaMalloc(&_callerInfo, sizeof(float)), "cudaMalloc(callerInfo)");
        Check(cudaMemcpy(_callerInfo, &hostOffset, sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(callerInfo)");
        void* info = _callerInfo;
        const char* symbol = input == IntInput::U8 ? "rapidht_load_u8_lto" : "rapidht_load_u16_lto";
        Check(cufftXtSetJITCallback(_plan, symbol, rapidht_lto_fatbin, sizeof(rapidht_lto_fatbin), CUFFT_CB_LD_REAL,
                  &info),
            "cufftXtSetJITCallback");
        MakePlan();
    }

    ~CufftIntLto() override
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

struct CufftHartley::Impl {
    int width = 0;
    int height = 0;
    int depth = 0;
    cufftHandle plan = 0;
    cufftComplex* spectrum = nullptr;
    size_t spectrumBytes = 0;
    size_t workspaceBytes = 0;

    ~Impl()
    {
        if (plan != 0) {
            cufftDestroy(plan);
        }
        if (spectrum != nullptr) {
            cudaFree(spectrum);
        }
    }
};

CufftHartley::CufftHartley(int width, int height, int depth):
    _impl(std::make_unique<Impl>())
{
    // depth == 0 is a 2D image, as in the library; the conversion kernel then
    // runs on one layer (D = 1 mirrors z = 0 onto itself).
    _impl->width = width;
    _impl->height = height;
    _impl->depth = depth > 0 ? depth : 1;

    // cuFFT takes the slowest dimension first and makes the last one
    // contiguous, which matches this library's idx = x + W*(y + H*z).
    if (depth > 0) {
        Check(cufftPlan3d(&_impl->plan, depth, height, width, CUFFT_R2C), "cufftPlan3d");
        Check(cufftGetSize3d(_impl->plan, depth, height, width, CUFFT_R2C, &_impl->workspaceBytes),
            "cufftGetSize3d");
    } else {
        Check(cufftPlan2d(&_impl->plan, height, width, CUFFT_R2C), "cufftPlan2d");
        Check(cufftGetSize2d(_impl->plan, height, width, CUFFT_R2C, &_impl->workspaceBytes), "cufftGetSize2d");
    }

    const size_t half = static_cast<size_t>(width) / 2 + 1;
    _impl->spectrumBytes = half * height * _impl->depth * sizeof(cufftComplex);
    Check(cudaMalloc(&_impl->spectrum, _impl->spectrumBytes), "cudaMalloc(spectrum)");
}

CufftHartley::~CufftHartley() = default;

void CufftHartley::Forward(float* deviceInOut)
{
    Check(cufftExecR2C(_impl->plan, deviceInOut, _impl->spectrum), "cufftExecR2C");

    LaunchSpectrumToHartley<false>(_impl->spectrum, deviceInOut, _impl->width, _impl->height, _impl->depth, 0.0);
    Check(cudaDeviceSynchronize(), "SpectrumToHartley");
}

void CufftHartley::ForwardSpectrumOnly(const float* deviceIn)
{
    // cufftExecR2C takes a non-const pointer but does not write an
    // out-of-place input.
    Check(cufftExecR2C(_impl->plan, const_cast<float*>(deviceIn), _impl->spectrum), "cufftExecR2C");
    Check(cudaDeviceSynchronize(), "cufftExecR2C");
}

size_t CufftHartley::ExtraDeviceBytes() const
{
    return _impl->spectrumBytes + _impl->workspaceBytes;
}

size_t CufftHartley::SpectrumBytes() const
{
    return _impl->spectrumBytes;
}

size_t CufftHartley::WorkspaceBytes() const
{
    return _impl->workspaceBytes;
}

std::unique_ptr<CufftIntHartley> MakeCufftIntHartley(
    int width, int height, int depth, IntInput input, IntLoad load, double offset)
{
    switch (load) {
        case IntLoad::Convert:
            return std::make_unique<CufftIntConvert>(width, height, depth, input, offset);
        case IntLoad::LtoCallback:
            return std::make_unique<CufftIntLto>(width, height, depth, input, offset);
        case IntLoad::LegacyCallback:
            // Not here: it needs the static cuFFT, which only targets linking
            // rapidht_cufft_callback have (MakeCufftIntCallback).
            break;
    }
    throw std::runtime_error("MakeCufftIntHartley: Convert or LtoCallback only");
}

} // namespace rapidht_bench
