/*
 * Project: RapiDHT
 * File: benchmarks/accuracy_gpu3d.cpp
 * Brief: Single-precision error of the GPU paths and of cuFFT, against double.
 *
 *   ./accuracy_gpu3d 256 512
 *
 * R-4 step 0.4 of docs/R4_PLAN.md: a faster transform that is less accurate
 * than cuFFT has not beaten it, so the error is measured beside the time. For
 * each cube edge, one zero-mean pseudo-random volume (the same LCG as the
 * tests' MakeNoise) is transformed by
 *   - the CPU backend in double: the reference;
 *   - the factored GPU path in float (Modes::GPUFactored, resident);
 *   - the matrix GPU path in float (Modes::GPU, resident), for context;
 *   - cuFFT R2C in float plus its conversion to Hartley (the benchmark baseline).
 * Errors are given relative to the RMS of the reference spectrum: for noise
 * every coefficient is of the same order, so this is the scale at which a
 * wrong coefficient would show. Both the largest error and the RMS error are
 * printed; the plan compares the largest one.
 *
 * Needs, at 512: 2.5 GiB of host memory and 1.1 GiB on the device (cuFFT
 * holds its spectrum and workspace beside the volume). The double reference
 * takes a few seconds on the CPU.
 */

#include <rapidht/device_volume.h>
#include <rapidht/transform.h>

#include <cuda_runtime.h>

#include <algorithm>
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

std::vector<double> MakeNoise(size_t count)
{
    std::vector<double> data(count);
    uint32_t seed = 2463534242u;
    for (double& value : data) {
        seed = seed * 1664525u + 1013904223u;
        value = static_cast<double>(seed >> 8) / 8388608.0 - 1.0;
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

std::vector<float> ViaLibrary(const std::vector<float>& input, size_t n, Modes mode)
{
    HartleyTransform<float> transform(n, n, n, mode);
    DeviceVolume<float> volume(input.size());
    volume.Upload(input.data());
    transform.ForwardTransform(volume);
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

int Run(size_t n)
{
    const size_t count = n * n * n;
    std::printf("%zu^3, zero-mean noise in [-1, 1), errors relative to the RMS of the double spectrum\n", n);
    const std::vector<double> input = MakeNoise(count);

    std::vector<double> reference = input;
    HartleyTransform<double>(n, n, n, Modes::CPU).ForwardTransform(reference.data());

    const std::vector<float> single(input.begin(), input.end());
    const Error cufft = Compare(ViaCufft(single, n), reference);
    const Error factored = Compare(ViaLibrary(single, n, Modes::GPUFactored), reference);
    const Error matrix = Compare(ViaLibrary(single, n, Modes::GPU), reference);

    Report("cuFFT R2C + conversion", cufft, cufft);
    Report("RapiDHT GPUFactored", factored, cufft);
    Report("RapiDHT GPU (matrices)", matrix, cufft);
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
            Run(static_cast<size_t>(std::strtoul(argv[i], nullptr, 10)));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
