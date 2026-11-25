/*
 * Project: RapiDHT
 * File: benchmarks/convolution_gpu.h
 * Brief: Circular convolution on the device, through Hartley and through cuFFT.
 *
 * R-4 step 5.3 of docs/R4_PLAN.md. A transform is rarely the end product; the
 * usual reason to take one is a convolution: forward, multiply by the kernel's
 * spectrum, back. That is the comparison this file serves.
 *
 *   Hartley: the transform runs in place, the product is real, and the kernel's
 *   spectrum is one volume of floats. Nothing else lives beside the data.
 *   cuFFT:   R2C into a half spectrum, a complex product, C2R back: the half
 *   spectrum (about one volume), the work area and the kernel's half spectrum.
 *
 * The Hartley side needs only the multiply step from here; the transforms are
 * the library's own (Modes::GPUFactored, on a DeviceVolume). Benchmark
 * scaffolding, not part of the library, float only, like the cuFFT baseline.
 */

#ifndef RAPIDHT_BENCH_CONVOLUTION_GPU_H
#define RAPIDHT_BENCH_CONVOLUTION_GPU_H

#include <cstddef>
#include <memory>

namespace rapidht_bench {

/**
 * The multiply step of the Hartley convolution theorem, in place, for a kernel
 * of any symmetry and the true multidimensional transform:
 *
 *     Z(k) = F(k) E(k) + F(-k) O(k),   E, O = (S(k) +- S(-k)) / 2,
 *
 * where F is the data's spectrum and S the kernel's, -k negated per axis. A
 * thread owns the pair (k, -k), so the update is safe in place. depth == 0 is
 * a 2D image. Synchronous.
 */
void HartleyConvolveSpectrum(float* deviceData, const float* deviceKernelSpectrum, int width, int height,
    int depth);

/// data <- data * kernelSpectrum, pointwise: the multiply step for a kernel
/// even along every axis, whose separable spectrum is the whole story.
/// Synchronous.
void MultiplyPointwise(float* deviceData, const float* deviceKernelSpectrum, size_t count);

/**
 * cuFFT's convolution: R2C, product with the kernel's half spectrum, C2R.
 * The two plans share one work area, as a careful caller would arrange.
 */
class CufftConvolution {
public:
    /// @param hostKernel W*H*D floats, wrapped layout (centre at index 0).
    /// @param evenKernel The kernel is even along every axis: its spectrum is
    ///        real, and only the real half spectrum is kept and multiplied.
    /// @param deviceScratch W*H*D floats on the device, overwritten: the
    ///        caller's volume serves to stage the kernel, so building costs no
    ///        memory beyond what Apply needs.
    CufftConvolution(int width, int height, int depth, const float* hostKernel, bool evenKernel,
        float* deviceScratch);
    ~CufftConvolution();

    CufftConvolution(const CufftConvolution&) = delete;
    CufftConvolution& operator=(const CufftConvolution&) = delete;

    /// deviceInOut <- deviceInOut * kernel (circular). Synchronous.
    void Apply(float* deviceInOut);

    /// Device memory this holds beside the caller's volume: the data's half
    /// spectrum, the kernel's, and the shared work area.
    size_t DeviceBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace rapidht_bench

#endif // RAPIDHT_BENCH_CONVOLUTION_GPU_H
