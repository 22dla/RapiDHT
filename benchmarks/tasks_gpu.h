/*
 * Project: RapiDHT
 * File: benchmarks/tasks_gpu.h
 * Brief: R-5 stage 5 -- the end-to-end tasks on an integer tomogram.
 *
 * docs/R5_PLAN.md, step 5.1. A transform is a means; what is measured here is
 * what it is taken for, from a uint8 / uint16 volume as a tomograph writes it:
 *
 *   T1  the spectrum: integers -> the Hartley spectrum in float on the device
 *       (the library's integer entry point, or cuFFT with an LTO load callback);
 *   T2  a Gaussian filter, integers -> integers: forward, product with the
 *       kernel's spectrum, inverse (MultiplySeparable, CufftGaussianFilter),
 *       and for reference the same filter in space (SpatialGaussian);
 *   T3  the radial power spectrum: integers -> P(|k|) in 128 bins on the host
 *       (RadialPowerHartley, CufftPowerProfile).
 *
 * The library's transforms are its own; this file holds what the tasks add
 * around them, and cuFFT's side of each task done the strongest way the
 * earlier steps found (LTO callbacks for reading and writing integers, steps
 * 0.2 and 4.3). Benchmark scaffolding, float only, like the cuFFT baseline.
 */

#ifndef RAPIDHT_BENCH_TASKS_GPU_H
#define RAPIDHT_BENCH_TASKS_GPU_H

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "cufft_baseline.h"

#if defined(__CUDACC__)
#define RAPIDHT_TASK_HD __host__ __device__
#else
#define RAPIDHT_TASK_HD
#endif

namespace rapidht_bench {

/*
 * T2's kernel: a Gaussian of `sigma` voxels cut at radius r = 3 sigma (6 sigma
 * + 1 taps), normalised to sum 1, the same along every axis. Taps c[0..r] in
 * double, c[d] the weight at distance d. The filter is circular, like the
 * spectral one, so all implementations compute the same thing.
 */
std::vector<double> GaussianTaps(double sigma);

/// The DHT of the kernel along an axis of length n: G(k) = sum_d c(|d|) cos(2 pi k d / n),
/// real and even, since the kernel is. Its 3D separable spectrum is
/// Gx(kx) Gy(ky) Gz(kz).
std::vector<float> GaussianAxisSpectrum(const std::vector<double>& taps, size_t n);

/// T2 in Hartley: data(kx, ky, kz) *= sx[kx] * sy[ky] * sz[kz], the product of a
/// separable spectrum with the separable transform of the data. The kernel's
/// spectrum is three short tables, not a volume. Synchronous.
void MultiplySeparable(float* deviceData, const float* deviceSx, const float* deviceSy, const float* deviceSz,
    int width, int height, int depth);

/*
 * T2 through cuFFT: R2C reading the integers (LTO load callback, offset
 * subtracted), the half spectrum times sx[kx] sy[ky] sz[kz], C2R writing the
 * integers (LTO store callback: rint(x / N + offset), clamped). The offset is
 * the library's default policy -- the rounded mean of every 64th line, on the
 * device -- so both sides subtract the same thing at the same price. The two
 * plans share one work area.
 */
class CufftGaussianFilter {
public:
    /// sx, sy, sz: host tables of GaussianAxisSpectrum (W, H, D entries).
    CufftGaussianFilter(int width, int height, int depth, IntInput type, const std::vector<float>& sx,
        const std::vector<float>& sy, const std::vector<float>& sz);
    ~CufftGaussianFilter();

    CufftGaussianFilter(const CufftGaussianFilter&) = delete;
    CufftGaussianFilter& operator=(const CufftGaussianFilter&) = delete;

    /// deviceOut <- filter(deviceIn), both W*H*D integers of the type given. Synchronous.
    void Apply(const void* deviceIn, void* deviceOut);

    /// What this holds beside the caller's input and output.
    size_t ExtraDeviceBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/*
 * T2 in space, for reference: three passes along x, y, z, each a circular
 * convolution with the 2r + 1 taps -- x reads the integers into a float
 * volume, y runs in place on it, z writes the rounded, clamped integers. Only
 * sigma = 1 and 2 (r = 3 and 6), where a spatial filter is the natural choice.
 */
class SpatialGaussian {
public:
    SpatialGaussian(int width, int height, int depth, IntInput type, const std::vector<double>& taps);
    ~SpatialGaussian();

    SpatialGaussian(const SpatialGaussian&) = delete;
    SpatialGaussian& operator=(const SpatialGaussian&) = delete;

    /// deviceOut <- filter(deviceIn). Synchronous.
    void Apply(const void* deviceIn, void* deviceOut);

    /// The float volume between the passes.
    size_t ExtraDeviceBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/*
 * T3's bins. A frequency index k along an axis of length n is the signed
 * frequency k or k - n (|k| <= n / 2), scaled by L / n, L the longest axis, so
 * that a non-cubic volume bins by physical frequency. With r^2 the sum of the
 * three squares (an integer), the bin is floor(128 r / (sqrt(3) L / 2)), the
 * last bin closed: decided in integers, so the host and the device agree.
 */
constexpr int kRadialBins = 128;

RAPIDHT_TASK_HD inline long long ScaledFrequency(long long index, long long n, long long longest)
{
    return (2 * index <= n ? index : index - n) * (longest / n);
}

RAPIDHT_TASK_HD inline int RadialBin(long long r2, long long longest)
{
    // b <= 128 r / (sqrt(3) L / 2)  <=>  3 L^2 b^2 <= 4 * 128^2 r^2
    const long long num = r2 * 4 * kRadialBins * kRadialBins;
    const long long den = 3 * longest * longest;
    int b = static_cast<int>(sqrtf(static_cast<float>(num) / static_cast<float>(den)));
    while (static_cast<long long>(b + 1) * (b + 1) * den <= num) {
        ++b;
    }
    while (b > 0 && static_cast<long long>(b) * b * den > num) {
        --b;
    }
    return b < kRadialBins ? b : kRadialBins - 1;
}

/*
 * T3 in Hartley: the sum of H(k)^2 over each bin, into hostBins[128]. That is
 * the sum of |F(k)|^2 = (H(k)^2 + H(-k)^2) / 2, since a bin holds k and -k
 * alike. A bin is also closed under negating any one axis, and over such an
 * orbit the squares of the separable transform sum to those of the true one
 * (the cross terms of cos and sin cancel), so `deviceData` may hold either.
 * Synchronous; deviceBins: 128 doubles of scratch on the device.
 */
void RadialPowerHartley(const float* deviceData, int width, int height, int depth, double* deviceBins,
    double* hostBins);

/*
 * T3 through cuFFT: R2C reading the integers (LTO load callback, the default
 * offset subtracted on the device, offset * N returned to F(0) in the binning)
 * and the sum of |F(k)|^2 over each bin from the half spectrum, twice for the
 * columns 0 < kx < W/2 that stand for their mirrors.
 */
class CufftPowerProfile {
public:
    CufftPowerProfile(int width, int height, int depth, IntInput type);
    ~CufftPowerProfile();

    CufftPowerProfile(const CufftPowerProfile&) = delete;
    CufftPowerProfile& operator=(const CufftPowerProfile&) = delete;

    /// hostBins[128] <- the bin sums of the integers at deviceIn. Synchronous.
    void Apply(const void* deviceIn, double* hostBins);

    size_t ExtraDeviceBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace rapidht_bench

#endif // RAPIDHT_BENCH_TASKS_GPU_H
