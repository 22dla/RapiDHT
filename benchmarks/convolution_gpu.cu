/*
 * Project: RapiDHT
 * File: benchmarks/convolution_gpu.cu
 * Brief: Circular convolution on the device, through Hartley and through cuFFT.
 */

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "convolution_gpu.h"

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
 * H(f * g)(k) = F(k) E(k) + F(-k) O(k), with E and O the even and odd parts of
 * G = H(g). The same identity holds for the true multidimensional transform
 * (cas of the dot product) with -k negated on every axis. The thread of the
 * lower index of a pair writes both: Z(-k) = F(-k) E(k) - F(k) O(k), since E is
 * even and O odd. A self-mirrored point has O = 0.
 */
__global__ void HartleyPairProduct(float* __restrict__ data, const float* __restrict__ kernel, int W, int H, int D)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) {
        return;
    }
    const int xm = (x == 0) ? 0 : W - x;
    const int ym = (y == 0) ? 0 : H - y;
    const int zm = (z == 0) ? 0 : D - z;
    const size_t i = x + static_cast<size_t>(W) * (y + static_cast<size_t>(H) * z);
    const size_t m = xm + static_cast<size_t>(W) * (ym + static_cast<size_t>(H) * zm);
    if (i > m) {
        return;
    }
    const float fk = data[i];
    const float fm = data[m];
    const float gk = kernel[i];
    const float gm = kernel[m];
    const float even = 0.5f * (gk + gm);
    const float odd = 0.5f * (gk - gm);
    data[i] = fk * even + fm * odd;
    if (m != i) {
        data[m] = fm * even - fk * odd;
    }
}

__global__ void RealProduct(float* __restrict__ data, const float* __restrict__ kernel, size_t count)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < count) {
        data[i] *= kernel[i];
    }
}

__global__ void ComplexProduct(cufftComplex* __restrict__ spectrum, const cufftComplex* __restrict__ kernel,
    size_t count)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < count) {
        const cufftComplex a = spectrum[i];
        const cufftComplex b = kernel[i];
        spectrum[i] = make_cuComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
}

/// A complex spectrum times a real one: the kernel of an even g.
__global__ void ComplexByReal(cufftComplex* __restrict__ spectrum, const float* __restrict__ kernel, size_t count)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < count) {
        const float b = kernel[i];
        spectrum[i].x *= b;
        spectrum[i].y *= b;
    }
}

/// Keeps the real part, scaled: the half spectrum of an even kernel, as floats.
__global__ void RealPartScaled(const cufftComplex* __restrict__ in, float* __restrict__ out, float scale,
    size_t count)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < count) {
        out[i] = in[i].x * scale;
    }
}

__global__ void ScaleComplex(cufftComplex* __restrict__ data, float scale, size_t count)
{
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < count) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

unsigned Blocks(size_t count, unsigned block) { return static_cast<unsigned>((count + block - 1) / block); }

} // namespace

void HartleyConvolveSpectrum(float* deviceData, const float* deviceKernelSpectrum, int width, int height,
    int depth)
{
    const int D = depth > 0 ? depth : 1;
    const dim3 block(32, 4, 2);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (D + block.z - 1) / block.z);
    HartleyPairProduct<<<grid, block>>>(deviceData, deviceKernelSpectrum, width, height, D);
    Check(cudaGetLastError(), "HartleyPairProduct launch");
    Check(cudaDeviceSynchronize(), "HartleyPairProduct");
}

void MultiplyPointwise(float* deviceData, const float* deviceKernelSpectrum, size_t count)
{
    RealProduct<<<Blocks(count, 256), 256>>>(deviceData, deviceKernelSpectrum, count);
    Check(cudaGetLastError(), "RealProduct launch");
    Check(cudaDeviceSynchronize(), "RealProduct");
}

struct CufftConvolution::Impl {
    bool even = false;
    size_t halfCount = 0; ///< complex coefficients in the half spectrum
    size_t workBytes = 0;
    cufftHandle forward = 0;
    cufftHandle inverse = 0;
    void* work = nullptr;
    cufftComplex* spectrum = nullptr; ///< the data's half spectrum
    void* kernel = nullptr; ///< cufftComplex[halfCount], or float[halfCount] if even

    ~Impl()
    {
        if (forward != 0) {
            cufftDestroy(forward);
        }
        if (inverse != 0) {
            cufftDestroy(inverse);
        }
        cudaFree(work);
        cudaFree(spectrum);
        cudaFree(kernel);
    }
};

CufftConvolution::CufftConvolution(int width, int height, int depth, const float* hostKernel, bool evenKernel,
    float* deviceScratch):
    _impl(std::make_unique<Impl>())
{
    Impl& s = *_impl;
    s.even = evenKernel;
    const size_t D = depth > 0 ? static_cast<size_t>(depth) : 1;
    const size_t count = static_cast<size_t>(width) * height * D;
    s.halfCount = (static_cast<size_t>(width) / 2 + 1) * height * D;

    // Plans without their own work areas; one area of the larger size serves
    // both, since the two never run at the same time.
    Check(cufftCreate(&s.forward), "cufftCreate");
    Check(cufftCreate(&s.inverse), "cufftCreate");
    Check(cufftSetAutoAllocation(s.forward, 0), "cufftSetAutoAllocation");
    Check(cufftSetAutoAllocation(s.inverse, 0), "cufftSetAutoAllocation");
    size_t forwardBytes = 0, inverseBytes = 0;
    if (depth > 0) {
        Check(cufftMakePlan3d(s.forward, depth, height, width, CUFFT_R2C, &forwardBytes), "cufftMakePlan3d");
        Check(cufftMakePlan3d(s.inverse, depth, height, width, CUFFT_C2R, &inverseBytes), "cufftMakePlan3d");
    } else {
        Check(cufftMakePlan2d(s.forward, height, width, CUFFT_R2C, &forwardBytes), "cufftMakePlan2d");
        Check(cufftMakePlan2d(s.inverse, height, width, CUFFT_C2R, &inverseBytes), "cufftMakePlan2d");
    }
    s.workBytes = std::max<size_t>(std::max(forwardBytes, inverseBytes), 1);
    Check(cudaMalloc(&s.work, s.workBytes), "cudaMalloc(work)");
    Check(cufftSetWorkArea(s.forward, s.work), "cufftSetWorkArea");
    Check(cufftSetWorkArea(s.inverse, s.work), "cufftSetWorkArea");
    Check(cudaMalloc(&s.spectrum, s.halfCount * sizeof(cufftComplex)), "cudaMalloc(spectrum)");

    // The kernel's half spectrum, with the 1/N of the round trip folded in.
    Check(cudaMemcpy(deviceScratch, hostKernel, count * sizeof(float), cudaMemcpyHostToDevice), "upload kernel");
    Check(cufftExecR2C(s.forward, deviceScratch, s.spectrum), "cufftExecR2C(kernel)");
    const float scale = 1.0f / static_cast<float>(count);
    if (evenKernel) {
        Check(cudaMalloc(&s.kernel, s.halfCount * sizeof(float)), "cudaMalloc(kernel)");
        RealPartScaled<<<Blocks(s.halfCount, 256), 256>>>(s.spectrum, static_cast<float*>(s.kernel), scale,
            s.halfCount);
    } else {
        Check(cudaMalloc(&s.kernel, s.halfCount * sizeof(cufftComplex)), "cudaMalloc(kernel)");
        ScaleComplex<<<Blocks(s.halfCount, 256), 256>>>(s.spectrum, scale, s.halfCount);
        Check(cudaMemcpy(s.kernel, s.spectrum, s.halfCount * sizeof(cufftComplex), cudaMemcpyDeviceToDevice),
            "copy kernel spectrum");
    }
    Check(cudaGetLastError(), "kernel spectrum");
    Check(cudaDeviceSynchronize(), "kernel spectrum");
}

CufftConvolution::~CufftConvolution() = default;

size_t CufftConvolution::DeviceBytes() const
{
    const Impl& s = *_impl;
    return s.workBytes + s.halfCount * sizeof(cufftComplex)
         + s.halfCount * (s.even ? sizeof(float) : sizeof(cufftComplex));
}

void CufftConvolution::Apply(float* deviceInOut)
{
    Impl& s = *_impl;
    Check(cufftExecR2C(s.forward, deviceInOut, s.spectrum), "cufftExecR2C");
    if (s.even) {
        ComplexByReal<<<Blocks(s.halfCount, 256), 256>>>(s.spectrum, static_cast<const float*>(s.kernel),
            s.halfCount);
    } else {
        ComplexProduct<<<Blocks(s.halfCount, 256), 256>>>(s.spectrum,
            static_cast<const cufftComplex*>(s.kernel), s.halfCount);
    }
    Check(cudaGetLastError(), "spectrum product launch");
    Check(cufftExecC2R(s.inverse, s.spectrum, deviceInOut), "cufftExecC2R");
    Check(cudaDeviceSynchronize(), "CufftConvolution::Apply");
}

} // namespace rapidht_bench
