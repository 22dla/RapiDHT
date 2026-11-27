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
 *   RAPIDHT_TRACE_KERNELS=1 ./profile_gpu3d 512 15 u16 factored   # R-5
 *   RAPIDHT_TRACE_KERNELS=1 ./profile_gpu3d 1024x1024x512 15 f32 factored   # R-6: W x H x D
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
#include "internal/device_access.h"

#include <cuda_runtime.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdint>
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

/// Device memory still free, MiB.
double FreeMiB()
{
#ifdef RAPIDHT_WITH_CUDA
    size_t freeBytes = 0, totalBytes = 0;
    cudaDeviceSynchronize();
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
        return -1.0;
    }
    return static_cast<double>(freeBytes) / (1024.0 * 1024.0);
#else
    return -1.0;
#endif
}

/// The volume profiled: a cube N^3, or W x H x D (R-6 step 0.1).
struct Extent {
    size_t width = 0, height = 0, depth = 0;
    size_t Count() const { return width * height * depth; }
    std::string Name() const
    {
        return (width == height && height == depth)
                 ? std::to_string(width) + "^3"
                 : std::to_string(width) + "x" + std::to_string(height) + "x" + std::to_string(depth);
    }
};

/// Multiply-adds per element of one axis: dense 2n, factored 2(N1 + N2),
/// N1 = 2^ceil(log2(n)/2), the twiddle not counted.
double PerAxisFlops(size_t n, bool factored)
{
    if (!factored) {
        return 2.0 * static_cast<double>(n);
    }
    size_t n1 = 1;
    while (n1 * n1 < n) {
        n1 *= 2;
    }
    return 2.0 * static_cast<double>(n1 + n / n1);
}

template <typename T>
int Run(const Extent& e, int iterations, const char* label, RapiDHT::Modes mode)
{
    const size_t count = e.Count();

    std::vector<T> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(std::sin(0.7 * static_cast<double>(i)) + 1.0);
    }

    std::printf("%s  %s  (%zu elements, %.1f MiB per buffer)  x%d iterations\n",
        label, e.Name().c_str(), count, static_cast<double>(count * sizeof(T)) / (1024.0 * 1024.0),
        iterations);

    try {
        // Touch the device first so that context creation is not counted as
        // memory the transform holds.
        RapiDHT::DeviceVolume<T> volume(count);
        volume.Upload(data.data());

        const double base = UsedMiB();
        RapiDHT::HartleyTransform<T> transform(e.width, e.height, e.depth, mode);
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
        const bool f = mode == RapiDHT::Modes::GPUFactored;
        const double gflop = (PerAxisFlops(e.width, f) + PerAxisFlops(e.height, f) + PerAxisFlops(e.depth, f))
                           * static_cast<double>(count) / 1e9;
        // GFLOP per millisecond is already TFLOP per second; the earlier
        // division by 1000 here printed 0.00 for every run.
        std::printf("  %.2f ms per transform, %.1f GFLOP of multiply-add arithmetic, %.2f TFLOP/s\n",
            msPer, gflop, gflop / msPer);

        // Device memory the transform holds beyond the volume itself: after
        // construction, after resident use (cuBLAS workspace included), and
        // after one host-pointer call, which allocates the staging buffer.
        const double afterResident = UsedMiB();
        // The host-pointer call stages through a second device buffer of the
        // volume's size. At 1024^3 f32 that is another 4 GiB, which an 8 GB
        // card does not have beside the resident volume, and the exception
        // used to hide the two figures above. Skip it when it cannot fit.
        const double volumeMiB = static_cast<double>(count * sizeof(T)) / (1024.0 * 1024.0);
        if (FreeMiB() < volumeMiB + 64.0) {
            std::printf("  device memory held by the transform: %.0f MiB built, %.0f MiB after resident use; "
                        "host-pointer call skipped (needs another %.0f MiB, %.0f MiB free)\n",
                afterConstruct - base, afterResident - base, volumeMiB, FreeMiB());
        } else {
            transform.ForwardTransform(data.data());
            const double afterHost = UsedMiB();
            std::printf("  device memory held by the transform: %.0f MiB built, %.0f MiB after resident use, "
                        "%.0f MiB after a host-pointer call\n",
                afterConstruct - base, afterResident - base, afterHost - base);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "  failed: %s\n", error.what());
        return 1;
    }
    return 0;
}

#ifdef RAPIDHT_WITH_CUDA
/*
 * R-5 step 2.4: the factored transform of a device-resident integer volume
 * (internal::DeviceAccess::ForwardFromIntegers -- the first pass reads the
 * integers, offset half the range), so that RAPIDHT_TRACE_KERNELS times the
 * integer path per kernel the way it does the float one.
 */
template <typename TIn>
int RunIntegers(const Extent& e, int iterations, const char* label)
{
    const size_t count = e.Count();
    std::vector<TIn> host(count);
    std::uint64_t state = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < count; ++i) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        host[i] = static_cast<TIn>(state >> 40);
    }
    const double offset = static_cast<double>(1u << (8 * sizeof(TIn) - 1));

    std::printf("%s factored  %s  (%zu elements, %.1f MiB of input)  x%d iterations\n", label, e.Name().c_str(), count,
        static_cast<double>(count * sizeof(TIn)) / (1024.0 * 1024.0), iterations);

    TIn* in = nullptr;
    if (cudaMalloc(&in, count * sizeof(TIn)) != cudaSuccess) {
        std::fprintf(stderr, "  failed: the integer input does not fit\n");
        return 1;
    }
    int status = 0;
    try {
        cudaMemcpy(in, host.data(), count * sizeof(TIn), cudaMemcpyHostToDevice);
        RapiDHT::DeviceVolume<float> out(count);
        RapiDHT::HartleyTransform<float> transform(e.width, e.height, e.depth, RapiDHT::Modes::GPUFactored);
        RapiDHT::internal::DeviceAccess::ForwardFromIntegers(transform, in, out, offset); // warm-up

        const auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            RapiDHT::internal::DeviceAccess::ForwardFromIntegers(transform, in, out, offset);
        }
        const auto finish = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(finish - start).count();
        std::printf("  %.2f ms per transform\n", ms / iterations);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "  failed: %s\n", error.what());
        status = 1;
    }
    cudaFree(in);
    return status;
}
#endif

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

    // N for a cube N^3, or WxHxD (R-6 step 0.1: 1024x1024x512).
    Extent e;
    {
        const char* arg = (argc > 1) ? argv[1] : "256";
        char* end = nullptr;
        e.width = std::strtoul(arg, &end, 10);
        e.height = e.depth = e.width;
        if (*end == 'x') {
            e.height = std::strtoul(end + 1, &end, 10);
            e.depth = (*end == 'x') ? std::strtoul(end + 1, &end, 10) : 0;
        }
    }
    const int iterations = (argc > 2) ? std::atoi(argv[2]) : 10;
    const bool useDouble = (argc > 3) && std::strcmp(argv[3], "f64") == 0;
    // A fourth argument "factored" runs the R-4 path instead of the matrices.
    const bool factored = (argc > 4) && std::strcmp(argv[4], "factored") == 0;
    const RapiDHT::Modes mode = factored ? RapiDHT::Modes::GPUFactored : RapiDHT::Modes::GPU;

    for (const size_t n : { e.width, e.height, e.depth }) {
        if (n == 0 || (n & (n - 1)) != 0) {
            std::fprintf(stderr, "extents must be powers of two, got %s\n", e.Name().c_str());
            return 1;
        }
    }

    // R-5: "u16" or "u8" in place of the precision runs the integer path.
    const bool u16 = (argc > 3) && std::strcmp(argv[3], "u16") == 0;
    const bool u8 = (argc > 3) && std::strcmp(argv[3], "u8") == 0;
    if (u16 || u8) {
        if (!factored) {
            std::fprintf(stderr, "integer input runs only on the factored path: add \"factored\"\n");
            return 1;
        }
#ifdef RAPIDHT_WITH_CUDA
        return u16 ? RunIntegers<std::uint16_t>(e, iterations, "u16")
                   : RunIntegers<std::uint8_t>(e, iterations, "u8");
#endif
    }

    const char* label = useDouble ? (factored ? "f64 factored" : "f64") : (factored ? "f32 factored" : "f32");
    return useDouble ? Run<double>(e, iterations, label, mode) : Run<float>(e, iterations, label, mode);
}
