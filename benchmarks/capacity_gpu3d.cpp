/*
 * Project: RapiDHT
 * File: benchmarks/capacity_gpu3d.cpp
 * Brief: Does the cuFFT baseline fit a W^3 float volume on this card?
 *
 *   ./capacity_gpu3d 1024
 *
 * R-4 step 0.5 of docs/R4_PLAN.md. The factored path transforms in place with
 * a few KiB of tables, so its largest volume is whatever fits on the device;
 * profile_gpu3d measures it. The cuFFT baseline needs the half spectrum and a
 * workspace beside the volume. This program tries exactly what the benchmark
 * does -- volume, plan, spectrum, one forward call -- and prints either the
 * time and the memory used or the error cuFFT or CUDA gave, word for word.
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

#include "cufft_baseline.h"

namespace {

double FreeMiB()
{
    size_t freeBytes = 0, totalBytes = 0;
    cudaDeviceSynchronize();
    cudaMemGetInfo(&freeBytes, &totalBytes);
    return static_cast<double>(freeBytes) / (1024.0 * 1024.0);
}

int Run(size_t n, int iterations)
{
    const size_t count = n * n * n;
    const double volumeMiB = static_cast<double>(count * sizeof(float)) / (1024.0 * 1024.0);
    const double halfMiB = static_cast<double>((n / 2 + 1) * n * n * 8) / (1024.0 * 1024.0);
    std::printf("cuFFT baseline, %zu^3 float: volume %.0f MiB, half spectrum %.0f MiB; %.0f MiB free before\n", n,
        volumeMiB, halfMiB, FreeMiB());

    float* device = nullptr;
    const cudaError_t status = cudaMalloc(&device, count * sizeof(float));
    if (status != cudaSuccess) {
        std::printf("  volume: cudaMalloc failed: %s\n", cudaGetErrorString(status));
        return 1;
    }
    cudaMemset(device, 0, count * sizeof(float));
    int result = 0;
    try {
        rapidht_bench::CufftHartley cufft(static_cast<int>(n), static_cast<int>(n), static_cast<int>(n));
        std::printf("  plan built: spectrum %.0f MiB + workspace %.0f MiB; %.0f MiB free\n",
            static_cast<double>(cufft.SpectrumBytes()) / (1024.0 * 1024.0),
            static_cast<double>(cufft.WorkspaceBytes()) / (1024.0 * 1024.0), FreeMiB());
        cufft.Forward(device); // warm-up
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            cufft.Forward(device);
        }
        const auto finish = std::chrono::steady_clock::now();
        std::printf("  fits: %.2f ms per transform (R2C + conversion)\n",
            std::chrono::duration<double, std::milli>(finish - start).count() / iterations);
    } catch (const std::exception& error) {
        std::printf("  does not fit: %s\n", error.what());
        result = 1;
    }
    cudaFree(device);
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s EDGE [ITERATIONS]\n", argv[0]);
        return 2;
    }
    const size_t n = std::strtoul(argv[1], nullptr, 10);
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 3;
    return Run(n, iterations > 0 ? iterations : 1);
}
