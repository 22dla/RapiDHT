/*
 * Project: RapiDHT
 * File: benchmarks/cufft_common.cuh
 * Brief: What the cuFFT baselines share: error checks, the Hartley conversion
 *        pass, and the plan of the integer-input variants (R-5 step 0.2).
 *
 * Included by cufft_baseline.cu and by cufft_int_callback.cu. The two end up in
 * different binaries -- the second is a shared library with its own, static,
 * cuFFT -- so everything here has internal linkage.
 */

#ifndef RAPIDHT_BENCH_CUFFT_COMMON_CUH
#define RAPIDHT_BENCH_CUFFT_COMMON_CUH

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "cufft_baseline.h"

namespace rapidht_bench {
namespace {

inline void Check(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

inline void Check(cufftResult status, const char* what)
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
 *
 * kAddDc (R-5): the integer variants read x - offset; the zero coefficient gets
 * offset * N back here, added in double by the one thread that owns it.
 */
template <bool kAddDc>
__global__ void SpectrumToHartley(const cufftComplex* __restrict__ spectrum, float* __restrict__ out,
    int W, int H, int D, double dc)
{
    const int half = W / 2 + 1;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= half || y >= H || z >= D) {
        return;
    }
    const cufftComplex c = spectrum[static_cast<size_t>(half) * (y + static_cast<size_t>(H) * z) + x];
    if constexpr (kAddDc) {
        if ((x | y | z) == 0) {
            out[0] = static_cast<float>(static_cast<double>(c.x - c.y) + dc);
            return; // column 0 is its own mirror: nothing else to write
        }
    }
    out[static_cast<size_t>(W) * (y + static_cast<size_t>(H) * z) + x] = c.x - c.y;
    if (x > 0 && 2 * x < W) {
        const int ym = (y == 0) ? 0 : H - y;
        const int zm = (z == 0) ? 0 : D - z;
        out[static_cast<size_t>(W) * (ym + static_cast<size_t>(H) * zm) + (W - x)] = c.x + c.y;
    }
}

template <bool kAddDc>
void LaunchSpectrumToHartley(const cufftComplex* spectrum, float* out, int W, int H, int D, double dc)
{
    const int half = W / 2 + 1;
    dim3 block(32, 4, 2);
    dim3 grid((half + block.x - 1) / block.x, (H + block.y - 1) / block.y, (D + block.z - 1) / block.z);
    SpectrumToHartley<kAddDc><<<grid, block>>>(spectrum, out, W, H, D, dc);
    Check(cudaGetLastError(), "SpectrumToHartley launch");
}

/// Bytes per voxel of the integer input.
inline size_t IntInputBytes(IntInput input)
{
    return input == IntInput::U8 ? sizeof(std::uint8_t) : sizeof(std::uint16_t);
}

/*
 * The part every integer variant has in common: an R2C plan of a W x H x D
 * volume, the half spectrum, and the Hartley pass that writes the caller's
 * float output and puts offset * N back into the zero coefficient. A variant
 * says how the plan is made (callbacks are attached before or after) and what,
 * if anything, runs before R2C.
 */
class CufftIntBase : public CufftIntHartley {
public:
    CufftIntBase(int width, int height, int depth, IntInput input, double offset):
        _width(width), _height(height), _depth(depth), _input(input), _offset(offset)
    {
        if (depth <= 0 || height <= 0) {
            throw std::runtime_error("integer cuFFT baselines are 3D only");
        }
        Check(cufftCreate(&_plan), "cufftCreate");
        const size_t half = static_cast<size_t>(width) / 2 + 1;
        _spectrumBytes = half * height * depth * sizeof(cufftComplex);
        Check(cudaMalloc(&_spectrum, _spectrumBytes), "cudaMalloc(spectrum)");
    }

    ~CufftIntBase() override
    {
        if (_plan != 0) {
            cufftDestroy(_plan);
        }
        if (_spectrum != nullptr) {
            cudaFree(_spectrum);
        }
    }

    CufftIntBase(const CufftIntBase&) = delete;
    CufftIntBase& operator=(const CufftIntBase&) = delete;

    size_t ExtraDeviceBytes() const override { return _spectrumBytes + _workspaceBytes; }

protected:
    /// cufftMakePlan3d on the handle made in the constructor; callbacks that
    /// must be known to the planner (LTO) are set before this.
    void MakePlan()
    {
        // cuFFT takes the slowest dimension first and makes the last one
        // contiguous, which matches this library's idx = x + W*(y + H*z).
        Check(cufftMakePlan3d(_plan, _depth, _height, _width, CUFFT_R2C, &_workspaceBytes), "cufftMakePlan3d");
    }

    /// R2C from `realIn` (floats, or the integers themselves when a load
    /// callback reads them), then the Hartley pass into `out`.
    void TransformAndConvert(const void* realIn, float* out)
    {
        // cufftExecR2C takes a non-const pointer but does not write an
        // out-of-place input.
        Check(cufftExecR2C(_plan, static_cast<cufftReal*>(const_cast<void*>(realIn)), _spectrum), "cufftExecR2C");
        const double n = static_cast<double>(_width) * _height * _depth;
        LaunchSpectrumToHartley<true>(_spectrum, out, _width, _height, _depth, _offset * n);
        Check(cudaDeviceSynchronize(), "SpectrumToHartley");
    }

    size_t Count() const { return static_cast<size_t>(_width) * _height * _depth; }

    int _width, _height, _depth;
    IntInput _input;
    double _offset;
    cufftHandle _plan = 0;
    cufftComplex* _spectrum = nullptr;
    size_t _spectrumBytes = 0;
    size_t _workspaceBytes = 0;
};

} // namespace
} // namespace rapidht_bench

#endif // RAPIDHT_BENCH_CUFFT_COMMON_CUH
