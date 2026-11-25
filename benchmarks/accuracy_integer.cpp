/*
 * Project: RapiDHT
 * File: benchmarks/accuracy_integer.cpp
 * Brief: How accurate is a float transform of an integer tomogram, and against what?
 *
 *   ./accuracy_integer 128 256 512
 *
 * R-5 of docs/BACKLOG.md, first question: would integer arithmetic beat the
 * error of the float transform on uint8 / uint16 volumes? Integer arithmetic
 * cannot compute the transform exactly -- cas(2 pi k n / N) is irrational, so
 * it would be fixed point -- and it can only help if the float error matters.
 * This program puts the float error next to two floors that no arithmetic
 * inside the transform can go below:
 *
 *   - quantisation of the input: a volume rounded to integers carries noise of
 *     RMS 1/sqrt(12) LSB per voxel, which reaches every coefficient as
 *     sqrt(N) / sqrt(12). In the units used here (error / sqrt(N), "LSB") that
 *     is 0.289 whatever the size.
 *   - the float output: the exact spectrum rounded to float. A transform that
 *     returns float cannot be more accurate than this, however it computes.
 *
 * Inputs (white integer noise, the case with the most information per voxel):
 *   uint8 full   uniform 0..255
 *   uint16 full  uniform 0..65535
 *   uint16 CT    30000 + uniform -2000..2000: a narrow band far from zero, the
 *                way a reconstructed tomogram sits in its range
 * Each is transformed raw, and with an integer offset subtracted first: half
 * the range (no extra pass needed), and the rounded mean (one reduction). The
 * subtraction is exact, and it changes only the zero-frequency coefficient, by
 * offset * N, which is added back in double before comparing. Errors are taken
 * over the non-zero frequencies; the zero frequency is reported on its own.
 *
 * The reference is the CPU backend in double: for these inputs every partial
 * sum is far below 2^53, so it is exact to well below the floors above.
 */

#include <rapidht/device_volume.h>
#include <rapidht/transform.h>

#include <cuda_runtime.h>

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

struct Input {
    const char* name;
    int32_t base; ///< values are base + uniform integer in [0, span)
    int32_t span;
    int32_t halfRange; ///< the offset a format would suggest without looking at the data
};

/// SplitMix64 integers in [base, base + span): white, exact in float (< 2^24).
std::vector<double> MakeIntegers(size_t count, const Input& in)
{
    std::vector<double> data(count);
    uint64_t state = 2463534242u;
    for (double& value : data) {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        value = static_cast<double>(in.base + static_cast<int32_t>(z % static_cast<uint64_t>(in.span)));
    }
    return data;
}

std::vector<float> OnDevice(const std::vector<float>& input, size_t n, bool cufft)
{
    std::vector<float> output(input.size());
    if (cufft) {
        float* device = nullptr;
        if (cudaMalloc(&device, input.size() * sizeof(float)) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
        try {
            rapidht_bench::CufftHartley plan(static_cast<int>(n), static_cast<int>(n), static_cast<int>(n));
            cudaMemcpy(device, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
            plan.Forward(device);
            cudaMemcpy(output.data(), device, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
        } catch (...) {
            cudaFree(device);
            throw;
        }
        cudaFree(device);
        return output;
    }
    HartleyTransform<float> transform(n, n, n, Modes::GPUFactored);
    DeviceVolume<float> volume(input.size());
    volume.Upload(input.data());
    transform.ForwardTransform(volume);
    volume.Download(output.data());
    return output;
}

struct Error {
    double rms = 0.0; ///< over k != 0, in LSB: RMS(error) / sqrt(N)
    double largest = 0.0; ///< over k != 0, in LSB
    double dc = 0.0; ///< |error| at k = 0, in LSB
};

/// `actual` is the transform of (input - offset); `reference` that of the input.
Error Measure(const std::vector<float>& actual, double offset, const std::vector<double>& reference)
{
    const double n = static_cast<double>(reference.size());
    const double unit = std::sqrt(n);
    Error e;
    e.dc = std::fabs(static_cast<double>(actual[0]) + offset * n - reference[0]) / unit;
    double sum = 0.0;
    for (size_t i = 1; i < reference.size(); ++i) {
        const double error = std::fabs(static_cast<double>(actual[i]) - reference[i]);
        sum += error * error;
        if (!(error <= e.largest)) {
            e.largest = error;
        }
    }
    e.rms = std::sqrt(sum / (n - 1.0)) / unit;
    e.largest /= unit;
    return e;
}

/// The float output floor: the exact spectrum rounded to float.
Error OutputFloor(const std::vector<double>& reference)
{
    std::vector<float> rounded(reference.begin(), reference.end());
    return Measure(rounded, 0.0, reference);
}

void Row(const char* label, const Error& e)
{
    std::printf("  %-34s rms %9.3e   largest %9.3e   zero frequency %9.3e\n", label, e.rms, e.largest, e.dc);
}

void Run(size_t n, const Input& in)
{
    const size_t count = n * n * n;
    const std::vector<double> input = MakeIntegers(count, in);
    double mean = 0.0;
    for (double v : input) {
        mean += v;
    }
    mean /= static_cast<double>(count);

    std::vector<double> reference = input;
    HartleyTransform<double>(n, n, n, Modes::CPU).ForwardTransform(reference.data());

    std::printf("%zu^3, %s (mean %.1f); errors / sqrt(N), in LSB of the input\n", n, in.name, mean);
    std::printf("  %-34s rms %9.3e   (input rounding: 1/sqrt(12))\n", "floor: quantisation of the input",
        1.0 / std::sqrt(12.0));
    Row("floor: exact spectrum stored as float", OutputFloor(reference));

    const double offsets[] = { 0.0, static_cast<double>(in.halfRange), std::round(mean) };
    const char* offsetNames[] = { "raw", "minus half range", "minus rounded mean" };
    for (int o = 0; o < 3; ++o) {
        if (o == 2 && offsets[2] == offsets[1]) {
            continue;
        }
        std::vector<float> single(count);
        for (size_t i = 0; i < count; ++i) {
            single[i] = static_cast<float>(input[i] - offsets[o]); // exact: integers below 2^24
        }
        for (const bool cufft : { true, false }) {
            const std::string label = std::string(cufft ? "cuFFT + conversion, " : "GPUFactored, ") + offsetNames[o];
            Row(label.c_str(), Measure(OnDevice(single, n, cufft), offsets[o], reference));
        }
    }
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s EDGE [EDGE ...]   (powers of two, e.g. 128 256 512)\n", argv[0]);
        return 2;
    }
    const Input inputs[] = {
        { "uint8 full range, uniform 0..255", 0, 256, 128 },
        { "uint16 full range, uniform 0..65535", 0, 65536, 32768 },
        { "uint16 CT-like band, 30000 +- 2000", 28000, 4001, 32768 },
    };
    try {
        for (int i = 1; i < argc; ++i) {
            const size_t n = static_cast<size_t>(std::strtoul(argv[i], nullptr, 10));
            for (const Input& in : inputs) {
                Run(n, in);
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
