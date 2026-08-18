/*
 * Project: RapiDHT
 * File: benchmarks/profile_gpu3d.cpp
 * Brief: Minimal driver for profiling the 3D GPU path.
 *
 * Deliberately does one thing: construct a transform once, then run the forward
 * transform in a loop. Nothing else runs, so a profiler attributes every kernel
 * to the transform rather than to harness machinery.
 *
 *   nsys profile --stats=true ./profile_gpu3d 512 20 f32
 *
 * The point of the exercise is the CUDA Kernel Summary that nsys prints. The
 * matrix formulation is only worth defending if the GEMMs dominate it; time
 * spent in transposes, permutations and the Bracewell pass is overhead that the
 * FFT-based alternatives do not pay.
 */

#include "rapidht.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

template <typename T>
int Run(size_t width, int iterations, const char* label)
{
    const size_t count = width * width * width;

    std::vector<T> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(std::sin(0.7 * static_cast<double>(i)) + 1.0);
    }

    std::printf("%s  %zu^3  (%zu elements, %.1f MiB per buffer)  x%d iterations\n",
        label, width, count, static_cast<double>(count * sizeof(T)) / (1024.0 * 1024.0),
        iterations);

    try {
        RapiDHT::HartleyTransform<T> transform(width, width, width, RapiDHT::Modes::GPU);

        // Untimed warm-up: the first call pays for context creation and for
        // cuBLAS picking its kernels.
        transform.ForwardTransform(data.data());

        const auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            transform.ForwardTransform(data.data());
        }
        const auto finish = std::chrono::high_resolution_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(finish - start).count();
        // 3 axes, W^2 lines each, 2*W^2 flops per line.
        const double gflop = 6.0 * std::pow(static_cast<double>(width), 4.0) / 1e9;
        std::printf("  %.2f ms per transform, %.1f GFLOP of matrix arithmetic, %.2f TFLOP/s\n",
            ms / iterations, gflop, gflop / (ms / iterations) / 1000.0);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "  failed: %s\n", error.what());
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (!RapiDHT::kCudaEnabled) {
        std::fprintf(stderr,
            "Built without CUDA support. Reconfigure with -DRAPIDHT_WITH_CUDA=ON.\n");
        return 1;
    }

    const size_t width = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 256;
    const int iterations = (argc > 2) ? std::atoi(argv[2]) : 10;
    const bool useDouble = (argc > 3) && std::strcmp(argv[3], "f64") == 0;

    if (width == 0 || (width & (width - 1)) != 0) {
        std::fprintf(stderr, "width must be a power of two, got %zu\n", width);
        return 1;
    }

    return useDouble ? Run<double>(width, iterations, "f64")
                     : Run<float>(width, iterations, "f32");
}
