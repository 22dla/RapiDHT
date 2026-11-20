/*
 * Project: RapiDHT
 * File: benchmarks/accuracy_gpu3d.cpp
 * Brief: Single-precision error of the GPU paths and of cuFFT, against double.
 *
 *   ./accuracy_gpu3d 256 512
 *
 * R-4 step 0.4 of docs/R4_PLAN.md: a faster transform that is less accurate
 * than cuFFT has not beaten it, so the error is measured beside the time. For
 * each cube edge, two zero-mean pseudo-random volumes (SplitMix64, then the
 * tests' LCG; see Noise) are transformed by
 *   - the CPU backend in double: the reference;
 *   - the factored GPU path in float (Modes::GPUFactored, resident);
 *   - the matrix GPU path in float (Modes::GPU, resident), for context;
 *   - cuFFT R2C in float plus its conversion to Hartley (the benchmark baseline).
 * Errors are given relative to the RMS of the reference spectrum: for noise
 * every coefficient is of the same order, so this is the scale at which a
 * wrong coefficient would show. Both the largest error and the RMS error are
 * printed; the plan compares the largest one.
 *
 * R-4a (finding where the largest error comes from) adds, per edge:
 *   - quantiles of the pointwise error, to tell a lone outlier from a tail;
 *   - the worst points, with the size of the reference there and cuFFT's
 *     error at the same point, to tell an error of the method from a hard
 *     coefficient;
 *   - per axis, the largest error over each residue k1 = index mod N1 of the
 *     factored split N = N1*N2: a twiddle or a C1 column that loses accuracy
 *     shows as one residue standing out;
 *   - the separable transform (no Bracewell pass) against the CPU separable
 *     transform in double, to tell the axis passes from the fused last pass.
 *
 * Needs, at 512: 2.5 GiB of host memory and 1.1 GiB on the device (cuFFT
 * holds its spectrum and workspace beside the volume); the R-4a breakdown
 * needs about 6 GiB of host memory at 512. The double references take a few
 * seconds each on the CPU.
 */

#include <rapidht/device_volume.h>
#include <rapidht/transform.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "cufft_baseline.h"

namespace {

using RapiDHT::DeviceVolume;
using RapiDHT::HartleyTransform;
using RapiDHT::Modes;

/// Which pseudo-random input to transform.
enum class Noise {
    /// The tests' 32-bit LCG. Sampled at power-of-two strides its bits fall on
    /// a lattice, so the volume is not white: rounding errors of sums along an
    /// axis come out correlated and pile up on a few output coefficients. Kept
    /// because it is a structured input, the kind a factored transform may
    /// handle worse than cuFFT, and because the earlier results used it.
    Lcg,
    /// SplitMix64: no structure at any stride, so it measures the transform
    /// rather than the generator. P3 of docs/R4_PLAN.md is judged on this one.
    SplitMix,
};

const char* NoiseName(Noise noise)
{
    return noise == Noise::Lcg ? "LCG noise (the tests' generator)" : "SplitMix64 noise";
}

/// Zero-mean noise in [-1, 1) on a 2^-23 grid, so every value is exact in float.
std::vector<double> MakeNoise(size_t count, Noise noise)
{
    std::vector<double> data(count);
    if (noise == Noise::Lcg) {
        uint32_t seed = 2463534242u;
        for (double& value : data) {
            seed = seed * 1664525u + 1013904223u;
            value = static_cast<double>(seed >> 8) / 8388608.0 - 1.0;
        }
        return data;
    }
    uint64_t state = 2463534242u;
    for (double& value : data) {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        value = static_cast<double>(z >> 40) / 8388608.0 - 1.0;
    }
    return data;
}

struct Error {
    double largest = 0.0; ///< max |actual - reference| / RMS(reference)
    double rms = 0.0; ///< RMS(actual - reference) / RMS(reference)
};

Error Compare(const std::vector<float>& actual, const std::vector<double>& reference)
{
    double sumRef = 0.0;
    double sumErr = 0.0;
    double largest = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double error = static_cast<double>(actual[i]) - reference[i];
        sumRef += reference[i] * reference[i];
        sumErr += error * error;
        const double magnitude = std::fabs(error);
        if (!(magnitude <= largest)) { // NaN propagates as the largest error
            largest = magnitude;
        }
    }
    const double rmsRef = std::sqrt(sumRef / static_cast<double>(reference.size()));
    return { largest / rmsRef, std::sqrt(sumErr / static_cast<double>(reference.size())) / rmsRef };
}

std::vector<float> ViaLibrary(const std::vector<float>& input, size_t n, Modes mode, bool separable = false)
{
    HartleyTransform<float> transform(n, n, n, mode);
    DeviceVolume<float> volume(input.size());
    volume.Upload(input.data());
    if (separable) {
        transform.ForwardSeparable(volume);
    } else {
        transform.ForwardTransform(volume);
    }
    std::vector<float> output(input.size());
    volume.Download(output.data());
    return output;
}

std::vector<float> ViaCufft(const std::vector<float>& input, size_t n)
{
    float* device = nullptr;
    if (cudaMalloc(&device, input.size() * sizeof(float)) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
    }
    std::vector<float> output(input.size());
    try {
        rapidht_bench::CufftHartley cufft(static_cast<int>(n), static_cast<int>(n), static_cast<int>(n));
        cudaMemcpy(device, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
        cufft.Forward(device);
        cudaMemcpy(output.data(), device, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    } catch (...) {
        cudaFree(device);
        throw;
    }
    cudaFree(device);
    return output;
}

void Report(const char* label, const Error& e, const Error& baseline)
{
    std::printf("  %-28s largest %.3e   rms %.3e   largest / cuFFT's %.2f\n", label, e.largest, e.rms,
        e.largest / baseline.largest);
}

double RmsOf(const std::vector<double>& reference)
{
    double sum = 0.0;
    for (const double value : reference) {
        sum += value * value;
    }
    return std::sqrt(sum / static_cast<double>(reference.size()));
}

/// |actual - reference| / rms, pointwise.
std::vector<float> PointErrors(const std::vector<float>& actual, const std::vector<double>& reference, double rms)
{
    std::vector<float> errors(reference.size());
    for (size_t i = 0; i < reference.size(); ++i) {
        errors[i] = static_cast<float>(std::fabs(static_cast<double>(actual[i]) - reference[i]) / rms);
    }
    return errors;
}

/// The median, the 99.9th and 99.999th percentiles and the maximum.
std::array<double, 4> Quantiles(std::vector<float> errors)
{
    std::array<double, 4> out {};
    const double levels[3] = { 0.5, 0.999, 0.99999 };
    auto from = errors.begin();
    for (size_t q = 0; q < 3; ++q) {
        const auto at = errors.begin() + static_cast<std::ptrdiff_t>(levels[q] * static_cast<double>(errors.size() - 1));
        std::nth_element(from, at, errors.end());
        out[q] = *at;
        from = at;
    }
    out[3] = *std::max_element(from, errors.end());
    return out;
}

void ReportQuantiles(const char* label, const std::vector<float>& errors)
{
    const std::array<double, 4> q = Quantiles(errors);
    std::printf("  %-28s median %.2e   p99.9 %.2e   p99.999 %.2e   max %.2e\n", label, q[0], q[1], q[2], q[3]);
}

/// The largest pointwise errors, largest first.
std::vector<size_t> WorstPoints(const std::vector<float>& errors, size_t count)
{
    std::vector<size_t> worst;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (worst.size() == count && !(errors[i] > errors[worst.back()])) {
            continue;
        }
        auto at = std::find_if(worst.begin(), worst.end(), [&](size_t w) { return errors[i] > errors[w]; });
        worst.insert(at, i);
        if (worst.size() > count) {
            worst.pop_back();
        }
    }
    return worst;
}

/// N1 of the factored split, as FactoredN1Of: 2^ceil(log2(n) / 2).
size_t SplitN1(size_t n)
{
    int bits = 0;
    while ((size_t { 1 } << bits) < n) {
        ++bits;
    }
    return size_t { 1 } << ((bits + 1) / 2);
}

/// Per axis, the largest error over each residue index mod N1; prints the three
/// worst residues and the median over residues.
void ReportResidues(const std::vector<float>& errors, size_t n)
{
    const size_t n1 = SplitN1(n);
    std::vector<double> worst(3 * n1, 0.0);
    for (size_t k = 0; k < n; ++k) {
        for (size_t j = 0; j < n; ++j) {
            const float* row = errors.data() + n * (j + n * k);
            for (size_t i = 0; i < n; ++i) {
                const double e = row[i];
                double& x = worst[0 * n1 + i % n1];
                double& y = worst[1 * n1 + j % n1];
                double& z = worst[2 * n1 + k % n1];
                x = std::max(x, e);
                y = std::max(y, e);
                z = std::max(z, e);
            }
        }
    }
    const char* names[3] = { "x", "y", "z" };
    for (size_t a = 0; a < 3; ++a) {
        std::vector<std::pair<double, size_t>> classes;
        for (size_t c = 0; c < n1; ++c) {
            classes.emplace_back(worst[a * n1 + c], c);
        }
        std::sort(classes.begin(), classes.end(), [](const auto& l, const auto& r) { return l.first > r.first; });
        std::printf("    %s: worst k1 (of N1 = %zu)  %2zu: %.2e   %2zu: %.2e   %2zu: %.2e   median over k1 %.2e\n",
            names[a], n1, classes[0].second, classes[0].first, classes[1].second, classes[1].first,
            classes[2].second, classes[2].first, classes[n1 / 2].first);
    }
}

void ReportBreakdown(size_t n, const std::vector<double>& reference, const std::vector<float>& cufft,
    const std::vector<float>& factored, const std::vector<double>& separableReference,
    const std::vector<float>& separable)
{
    const double rms = RmsOf(reference);
    const std::vector<float> cufftErrors = PointErrors(cufft, reference, rms);
    const std::vector<float> factoredErrors = PointErrors(factored, reference, rms);

    std::printf(" R-4a breakdown, %zu^3 (same units)\n", n);
    ReportQuantiles("cuFFT R2C + conversion", cufftErrors);
    ReportQuantiles("RapiDHT GPUFactored", factoredErrors);
    {
        const double separableRms = RmsOf(separableReference);
        ReportQuantiles("GPUFactored separable", PointErrors(separable, separableReference, separableRms));
    }

    std::printf("  worst points of GPUFactored (x, y, z): error, |reference|, cuFFT's error there\n");
    for (const size_t at : WorstPoints(factoredErrors, 8)) {
        const size_t i = at % n;
        const size_t j = (at / n) % n;
        const size_t k = at / (n * n);
        std::printf("    (%3zu, %3zu, %3zu)  %.2e   %.2f   %.2e\n", i, j, k, factoredErrors[at],
            std::fabs(reference[at]) / rms, cufftErrors[at]);
    }
    std::printf("  worst points of cuFFT (x, y, z): error, |reference|, GPUFactored's error there\n");
    for (const size_t at : WorstPoints(cufftErrors, 4)) {
        std::printf("    (%3zu, %3zu, %3zu)  %.2e   %.2f   %.2e\n", at % n, (at / n) % n, at / (n * n),
            cufftErrors[at], std::fabs(reference[at]) / rms, factoredErrors[at]);
    }
    std::printf("  GPUFactored by residue:\n");
    ReportResidues(factoredErrors, n);
    std::printf("  cuFFT by residue, for comparison:\n");
    ReportResidues(cufftErrors, n);
}

int Run(size_t n, Noise noise)
{
    const size_t count = n * n * n;
    std::printf("%zu^3, %s in [-1, 1), errors relative to the RMS of the double spectrum\n", n, NoiseName(noise));
    const std::vector<double> input = MakeNoise(count, noise);

    std::vector<double> reference = input;
    HartleyTransform<double>(n, n, n, Modes::CPU).ForwardTransform(reference.data());

    const std::vector<float> single(input.begin(), input.end());
    const std::vector<float> cufftOut = ViaCufft(single, n);
    const std::vector<float> factoredOut = ViaLibrary(single, n, Modes::GPUFactored);
    const Error cufft = Compare(cufftOut, reference);
    const Error factored = Compare(factoredOut, reference);
    const Error matrix = Compare(ViaLibrary(single, n, Modes::GPU), reference);

    Report("cuFFT R2C + conversion", cufft, cufft);
    Report("RapiDHT GPUFactored", factored, cufft);
    Report("RapiDHT GPU (matrices)", matrix, cufft);

    std::vector<double> separableReference = input;
    HartleyTransform<double>(n, n, n, Modes::CPU).ForwardSeparable(separableReference.data());
    const std::vector<float> separable = ViaLibrary(single, n, Modes::GPUFactored, true);
    ReportBreakdown(n, reference, cufftOut, factoredOut, separableReference, separable);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s EDGE [EDGE ...]   (powers of two, e.g. 256 512)\n", argv[0]);
        return 2;
    }
    try {
        for (int i = 1; i < argc; ++i) {
            const size_t n = static_cast<size_t>(std::strtoul(argv[i], nullptr, 10));
            Run(n, Noise::SplitMix);
            Run(n, Noise::Lcg);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
