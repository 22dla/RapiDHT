/*
 * Project: RapiDHT
 * File: benchmarks/cufft_baseline.h
 * Brief: The 3D Hartley transform computed through cuFFT, as a baseline.
 *
 * This is the competitor the matrix formulation has to answer for. cuFFT
 * computes the Fourier transform, but for real input the Hartley transform
 * follows from it by a single linear pass:
 *
 *     X(k) = sum x(n) * (cos - i*sin)   =>   Re(X) - Im(X) = sum x(n) * cas
 *
 * so "cuFFT plus an O(N) conversion" is a legitimate way to compute exactly
 * what this library computes, and the obvious thing a reviewer will ask about.
 *
 * Benchmark scaffolding, not part of the library: it lives here so that the
 * public headers stay free of CUDA, and it is float-only and 3D-only because
 * that is the case being compared.
 */

#ifndef RAPIDHT_BENCH_CUFFT_BASELINE_H
#define RAPIDHT_BENCH_CUFFT_BASELINE_H

#include <cstddef>
#include <memory>

namespace rapidht_bench {

class CufftHartley {
public:
    /// Plans an R2C/C2R pair for a W x H x D volume, W contiguous.
    CufftHartley(int width, int height, int depth);
    ~CufftHartley();

    CufftHartley(const CufftHartley&) = delete;
    CufftHartley& operator=(const CufftHartley&) = delete;

    /// Transforms a device-resident real volume in place, leaving the true
    /// multidimensional Hartley transform in it.
    void Forward(float* deviceInOut);

    /// Device memory this baseline needs beyond the caller's volume: the
    /// complex spectrum plus whatever cuFFT asks for as workspace.
    size_t ExtraDeviceBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace rapidht_bench

#endif // RAPIDHT_BENCH_CUFFT_BASELINE_H
