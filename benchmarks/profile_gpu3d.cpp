/*
 * Project: RapiDHT
 * File: benchmarks/profile_gpu3d.cpp
 * Brief: Minimal driver for profiling the 3D GPU path.
 *
 * Deliberately does one thing: upload a volume once, then run the forward
 * transform on it in a loop. Nothing else runs, so a profiler attributes every
 * kernel to the transform rather than to harness machinery.
 *
 *   nsys profile --stats=true ./profile_gpu3d 512 20 f32
 *
 * The point of the exercise is the CUDA Kernel Summary that nsys prints. The
 * matrix formulation is only worth defending if the GEMMs dominate it; time
 * spent in transposes, permutations and the Bracewell pass is overhead that the
 * FFT-based alternatives do not pay.
 *
 * The loop runs on a DeviceVolume rather than a host pointer. It did not used to
 * -- this file predates the resident API and was never updated -- so every
 * iteration carried two 512 MiB copies across the bus. At 512^3 that reported
 * 175.7 ms against the 58.2 ms the transform actually takes, and a rate three
 * times below the truth: exactly the measurement error the resident API was
 * added to remove.
 */

#include <rapidht/transform.h>

#ifdef RAPIDHT_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

/// Device memory in use, MiB, as the driver reports it. Differences between two
/// readings give what an object holds, which is the figure the benchmarks
/// document quotes as "extra device memory".
double UsedMiB()
{
#ifdef RAPIDHT_WITH_CUDA
    size_t freeBytes = 0, totalBytes = 0;
    cudaDeviceSynchronize();
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
        return -1.0;
    }
    return static_cast<double>(totalBytes - freeBytes) / (1024.0 * 1024.0);
#else
    return -1.0;
#endif
}

template <typename T>
int Run(size_t width, int iterations, const char* label, RapiDHT::Modes mode)
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
        // Touch the device first so that context creation is not counted as
        // memory the transform holds.
        RapiDHT::DeviceVolume<T> volume(count);
        volume.Upload(data.data());

        const double base = UsedMiB();
        RapiDHT::HartleyTransform<T> transform(width, width, width, mode);
        const double afterConstruct = UsedMiB();

        // Untimed warm-up: the first call pays for context creation and for
        // cuBLAS picking its kernels.
        transform.ForwardTransform(volume);

        const auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            transform.ForwardTransform(volume);
        }
        const auto finish = std::chrono::high_resolution_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(finish - start).count();
        const double msPer = ms / iterations;
        // 3 axes, W^2 lines each, 2*W^2 flops per line.
        // Dense matrices: 2W flops per element per axis. Factored (R-4): 2(N1 + N2),
        // N1 = 2^ceil(log2(W)/2), the twiddle not counted.
        double perAxis = 2.0 * static_cast<double>(width);
        if (mode == RapiDHT::Modes::GPUFactored) {
            size_t n1 = 1;
            while (n1 * n1 < width) {
                n1 *= 2;
            }
            perAxis = 2.0 * static_cast<double>(n1 + width / n1);
        }
        const double gflop = 3.0 * perAxis * std::pow(static_cast<double>(width), 3.0) / 1e9;
        // GFLOP per millisecond is already TFLOP per second; the earlier
        // division by 1000 here printed 0.00 for every run.
        std::printf("  %.2f ms per transform, %.1f GFLOP of multiply-add arithmetic, %.2f TFLOP/s\n",
            msPer, gflop, gflop / msPer);

        // Device memory the transform holds beyond the volume itself: after
        // construction, after resident use (cuBLAS workspace included), and
        // after one host-pointer call, which allocates the staging buffer.
        const double afterResident = UsedMiB();
        transform.ForwardTransform(data.data());
        const double afterHost = UsedMiB();
        std::printf("  device memory held by the transform: %.0f MiB built, %.0f MiB after resident use, "
                    "%.0f MiB after a host-pointer call\n",
            afterConstruct - base, afterResident - base, afterHost - base);
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
    // Compiled in is not the same as present, and a DeviceVolume on a machine
    // with no card throws rather than profiling anything.
    if (!RapiDHT::IsGpuAvailable()) {
        std::fprintf(stderr, "No usable CUDA device is present.\n");
        return 1;
    }

    const size_t width = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 256;
    const int iterations = (argc > 2) ? std::atoi(argv[2]) : 10;
    const bool useDouble = (argc > 3) && std::strcmp(argv[3], "f64") == 0;
    // A fourth argument "factored" runs the R-4 path instead of the matrices.
    const bool factored = (argc > 4) && std::strcmp(argv[4], "factored") == 0;
    const RapiDHT::Modes mode = factored ? RapiDHT::Modes::GPUFactored : RapiDHT::Modes::GPU;

    if (width == 0 || (width & (width - 1)) != 0) {
        std::fprintf(stderr, "width must be a power of two, got %zu\n", width);
        return 1;
    }

    const char* label = useDouble ? (factored ? "f64 factored" : "f64") : (factored ? "f32 factored" : "f32");
    return useDouble ? Run<double>(width, iterations, label, mode) : Run<float>(width, iterations, label, mode);
}
