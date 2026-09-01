/*
 * Project: RapiDHT
 * File: benchmarks/cufft_baseline.cu
 * Brief: The 3D Hartley transform via cuFFT, for comparison.
 */

#include "cufft_baseline.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>

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
 * For real input, H(k) = Re(X(k)) - Im(X(k)). cuFFT only stores k1 <= W/2 and
 * leaves the rest to Hermitian symmetry, X(k) = conj(X(-k)), so the mirrored
 * half needs Re + Im of the conjugate partner instead.
 */
__global__ void SpectrumToHartley(const cufftComplex* __restrict__ spectrum,
    float* __restrict__ out, int W, int H, int D)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= W || y >= H || z >= D) {
        return;
    }

    const int half = W / 2 + 1;
    float value;

    if (x < half) {
        const cufftComplex c = spectrum[static_cast<size_t>(half) * (y + static_cast<size_t>(H) * z) + x];
        value = c.x - c.y;
    } else {
        const int xm = W - x;
        const int ym = (y == 0) ? 0 : H - y;
        const int zm = (z == 0) ? 0 : D - z;
        const cufftComplex c
            = spectrum[static_cast<size_t>(half) * (ym + static_cast<size_t>(H) * zm) + xm];
        value = c.x + c.y; // conjugate flips the sign of the imaginary part
    }

    out[static_cast<size_t>(z) * H * W + static_cast<size_t>(y) * W + x] = value;
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

CufftHartley::CufftHartley(int width, int height, int depth): _impl(std::make_unique<Impl>())
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

    dim3 block(16, 8, 4);
    dim3 grid((_impl->width + block.x - 1) / block.x,
        (_impl->height + block.y - 1) / block.y,
        (_impl->depth + block.z - 1) / block.z);

    SpectrumToHartley<<<grid, block>>>(_impl->spectrum, deviceInOut, _impl->width, _impl->height,
        _impl->depth);
    Check(cudaGetLastError(), "SpectrumToHartley launch");
    Check(cudaDeviceSynchronize(), "SpectrumToHartley");
}

size_t CufftHartley::ExtraDeviceBytes() const
{
    return _impl->spectrumBytes + _impl->workspaceBytes;
}

} // namespace rapidht_bench
