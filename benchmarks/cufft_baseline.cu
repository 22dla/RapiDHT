/*
 * Project: RapiDHT
 * File: benchmarks/cufft_baseline.cu
 * Brief: The 3D Hartley transform via cuFFT, for comparison.
 */

#include <cuda_runtime.h>
#include <cufft.h>

#include <stdexcept>
#include <string>

#include "cufft_baseline.h"

namespace rapidht_bench {
namespace {

void Check(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void Check(cufftResult status, const char* what)
{
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuFFT error " + std::to_string(status));
    }
}

/*
 * Turns the half spectrum cuFFT produces into the full real Hartley transform.
 *
 * For real input, H(k) = Re(X(k)) - Im(X(k)), and cuFFT stores only
 * k1 <= W/2, leaving the rest to Hermitian symmetry X(-k) = conj X(k). A thread
 * owns one stored coefficient X(k1, y, z) and writes both of its outputs,
 *
 *     H(k1, y, z)     = Re X - Im X
 *     H(W-k1, -y, -z) = Re X + Im X        (0 < k1 < W/2),
 *
 * since X(W-k1, -y, -z) = conj X(k1, y, z). Columns k1 = 0 and k1 = W/2 are
 * their own mirrors in x and are stored in full, so they write one output.
 * so the spectrum is read once: two volumes of traffic. Looping over the
 * outputs instead reads every coefficient twice, three volumes, and was 10 %
 * slower in total at 128^3-512^3 (R-4 step 0.3, benchmarks/results/
 * 2026-11-13_*_r4_cufft_{rows,pairs}.json). The baseline is to be beaten at its
 * strongest, so this is the one kept.
 */
__global__ void SpectrumToHartley(const cufftComplex* __restrict__ spectrum, float* __restrict__ out,
    int W, int H, int D)
{
    const int half = W / 2 + 1;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= half || y >= H || z >= D) {
        return;
    }
    const cufftComplex c = spectrum[static_cast<size_t>(half) * (y + static_cast<size_t>(H) * z) + x];
    out[static_cast<size_t>(W) * (y + static_cast<size_t>(H) * z) + x] = c.x - c.y;
    if (x > 0 && 2 * x < W) {
        const int ym = (y == 0) ? 0 : H - y;
        const int zm = (z == 0) ? 0 : D - z;
        out[static_cast<size_t>(W) * (ym + static_cast<size_t>(H) * zm) + (W - x)] = c.x + c.y;
    }
}

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
    _impl->width = width;
    _impl->height = height;
    _impl->depth = depth;

    // cuFFT takes the slowest dimension first and makes the last one
    // contiguous, which matches this library's idx = x + W*(y + H*z).
    Check(cufftPlan3d(&_impl->plan, depth, height, width, CUFFT_R2C), "cufftPlan3d");
    Check(cufftGetSize3d(_impl->plan, depth, height, width, CUFFT_R2C, &_impl->workspaceBytes),
        "cufftGetSize3d");

    const size_t half = static_cast<size_t>(width) / 2 + 1;
    _impl->spectrumBytes = half * height * depth * sizeof(cufftComplex);
    Check(cudaMalloc(&_impl->spectrum, _impl->spectrumBytes), "cudaMalloc(spectrum)");
}

CufftHartley::~CufftHartley() = default;

void CufftHartley::Forward(float* deviceInOut)
{
    Check(cufftExecR2C(_impl->plan, deviceInOut, _impl->spectrum), "cufftExecR2C");

    const int half = _impl->width / 2 + 1;
    dim3 block(32, 4, 2);
    dim3 grid((half + block.x - 1) / block.x, (_impl->height + block.y - 1) / block.y,
        (_impl->depth + block.z - 1) / block.z);
    SpectrumToHartley<<<grid, block>>>(_impl->spectrum, deviceInOut, _impl->width, _impl->height, _impl->depth);
    Check(cudaGetLastError(), "SpectrumToHartley launch");
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

} // namespace rapidht_bench
