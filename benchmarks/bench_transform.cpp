/*
 * Project: RapiDHT
 * File: benchmarks/bench_transform.cpp
 * Brief: Throughput measurements for the Hartley transform backends.
 *
 * What this is built to answer
 * ----------------------------
 * The library targets 3D volumes, where a transform along one axis is really a
 * large batch of short 1D transforms. That is the case the GPU backend was
 * written for: the dense cas matrix is only W x W, small enough to stay
 * resident, and it is reused once per line, which turns the work into a dense
 * batched GEMM rather than the memory-bound matrix-vector product that a single
 * long 1D transform degenerates into. So the volumes below are the point, and
 * the 1D and 2D entries are context.
 *
 * Precision matters as much as size here. Consumer NVIDIA cards run FP64 at a
 * small fraction of their FP32 rate, so a GeForce can have *less* double
 * precision throughput than a modern desktop CPU. Both types are therefore
 * measured; on such a card the float numbers are the ones that reflect what the
 * hardware can do.
 *
 * Method notes
 * ------------
 * - The forward transform is unnormalised, so applying it repeatedly to the
 *   same buffer multiplies the magnitude by N each time. The input is restored
 *   before every iteration, with the timer paused.
 *
 * - Timings for the GPU backend include the host-to-device and device-to-host
 *   copies, because that is what the current API does on every call. The PCIe
 *   entries measure that round trip alone, so it can be subtracted to see the
 *   device-side cost.
 *
 * - Results are reported as items per second, so different extents compare
 *   directly.
 *
 * - The FFTW baseline uses FFTW_DHT. In 1D that is exactly what RapiDHT
 *   computes. In 2D and 3D FFTW applies a separable transform along each axis,
 *   whereas RapiDHT produces the true multidimensional Hartley transform and
 *   pays for an extra Bracewell pass, so it is doing strictly more work there.
 */

#include "rapidht.h"

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef RAPIDHT_BENCH_WITH_FFTW
#include <fftw3.h>
#endif

#ifdef RAPIDHT_WITH_CUDA
#include <cuda_runtime.h>
#endif

using RapiDHT::HartleyTransform;
using RapiDHT::kCudaEnabled;
using RapiDHT::Modes;

namespace {

template <typename T>
std::vector<T> MakeSignal(size_t count)
{
    std::vector<T> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(std::sin(0.7 * static_cast<double>(i)) + 1.0);
    }
    return data;
}

size_t ElementCount(size_t width, size_t height, size_t depth)
{
    return width * (height == 0 ? 1 : height) * (depth == 0 ? 1 : depth);
}

template <typename T>
void BM_Forward(benchmark::State& state, size_t width, size_t height, size_t depth, Modes mode)
{
    const size_t count = ElementCount(width, height, depth);
    const std::vector<T> pristine = MakeSignal<T>(count);
    std::vector<T> data = pristine;

    std::unique_ptr<HartleyTransform<T>> transform;
    try {
        transform = std::make_unique<HartleyTransform<T>>(width, height, depth, mode);
    } catch (const std::exception& error) {
        state.SkipWithError(error.what());
        return;
    }

    for (auto _ : state) {
        state.PauseTiming();
        std::memcpy(data.data(), pristine.data(), count * sizeof(T));
        state.ResumeTiming();

        transform->ForwardTransform(data.data());
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * count * sizeof(T)));
    state.counters["N"] = static_cast<double>(count);
}

#ifdef RAPIDHT_WITH_CUDA
/// Host to device and back, nothing else. Subtract from the GPU figure for the
/// same extent to separate the transform from the transfer.
template <typename T>
void BM_DeviceRoundTrip(benchmark::State& state, size_t count)
{
    T* device = nullptr;
    if (cudaMalloc(&device, count * sizeof(T)) != cudaSuccess) {
        state.SkipWithError("cudaMalloc failed");
        return;
    }
    std::vector<T> host(count, T { 1 });

    for (auto _ : state) {
        cudaMemcpy(device, host.data(), count * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost);
        benchmark::ClobberMemory();
    }

    cudaFree(device);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * count * sizeof(T) * 2));
    state.counters["N"] = static_cast<double>(count);
}
#endif

#ifdef RAPIDHT_BENCH_WITH_FFTW
void BM_Fftw(benchmark::State& state, size_t width, size_t height, size_t depth)
{
    const size_t count = ElementCount(width, height, depth);
    const std::vector<double> pristine = MakeSignal<double>(count);

    double* in = fftw_alloc_real(count);
    double* out = fftw_alloc_real(count);
    if (in == nullptr || out == nullptr) {
        state.SkipWithError("fftw_alloc_real failed");
        return;
    }
    std::memcpy(in, pristine.data(), count * sizeof(double));

    // FFTW_MEASURE is deliberate: planning is excluded from the timing, which
    // is the fair comparison for repeated transforms of the same extent.
    fftw_plan plan = nullptr;
    if (depth > 0) {
        plan = fftw_plan_r2r_3d(static_cast<int>(depth), static_cast<int>(height),
            static_cast<int>(width), in, out, FFTW_DHT, FFTW_DHT, FFTW_DHT, FFTW_MEASURE);
    } else if (height > 0) {
        plan = fftw_plan_r2r_2d(static_cast<int>(height), static_cast<int>(width), in, out,
            FFTW_DHT, FFTW_DHT, FFTW_MEASURE);
    } else {
        plan = fftw_plan_r2r_1d(static_cast<int>(width), in, out, FFTW_DHT, FFTW_MEASURE);
    }

    if (plan == nullptr) {
        fftw_free(in);
        fftw_free(out);
        state.SkipWithError("fftw_plan_r2r failed");
        return;
    }

    for (auto _ : state) {
        state.PauseTiming();
        std::memcpy(in, pristine.data(), count * sizeof(double));
        state.ResumeTiming();

        fftw_execute(plan);
        benchmark::DoNotOptimize(out);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * count * sizeof(double)));
    state.counters["N"] = static_cast<double>(count);

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}
#endif

struct Extent {
    size_t width;
    size_t height;
    size_t depth;
};

std::string Describe(const Extent& e)
{
    std::string name = std::to_string(e.width);
    if (e.height > 0) {
        name += "x" + std::to_string(e.height);
    }
    if (e.depth > 0) {
        name += "x" + std::to_string(e.depth);
    }
    return name;
}

/*
 * Device memory the GPU backend needs for its transform matrices: an axis of
 * length L costs L^2 elements. For volumes this is nothing -- 512^3 needs 6 MiB
 * of matrices against 2 GiB of data -- but a single long 1D transform makes the
 * matrix the whole cost, which is why the long 1D cases are absent below.
 */
size_t GpuMatrixBytes(const Extent& e, size_t elementSize)
{
    size_t elements = e.width * e.width;
    if (e.height > 0) {
        elements += e.height * e.height;
    }
    if (e.depth > 0) {
        elements += e.depth * e.depth;
    }
    return elements * elementSize;
}

constexpr size_t kGpuMatrixBudget = size_t { 1 } << 30; // 1 GiB

// The volumes the library is actually for. 512^3 is the ceiling on an 8 GiB
// card: two buffers of 2 GiB in double, half that in float.
const Extent kExtents3D[] = { { 128, 128, 128 }, { 256, 256, 256 }, { 512, 512, 512 } };

// Context only. Long 1D transforms are deliberately excluded: they are not what
// the library is for, and the dense matrix makes them meaningless for the GPU.
const Extent kExtents1D[] = { { 1024, 0, 0 }, { 16384, 0, 0 } };
const Extent kExtents2D[] = { { 128, 128, 0 }, { 512, 512, 0 }, { 1024, 1024, 0 } };

template <typename T>
void RegisterFor(const Extent* extents, size_t n, const char* rank, const char* type)
{
    for (size_t i = 0; i < n; ++i) {
        const Extent e = extents[i];
        const std::string suffix = std::string(type) + "/" + rank + "/" + Describe(e);

        benchmark::RegisterBenchmark(("CPU/" + suffix).c_str(),
            [e](benchmark::State& s) { BM_Forward<T>(s, e.width, e.height, e.depth, Modes::CPU); })
            ->Unit(benchmark::kMicrosecond);

        if (kCudaEnabled && GpuMatrixBytes(e, sizeof(T)) <= kGpuMatrixBudget) {
            benchmark::RegisterBenchmark(("GPU/" + suffix).c_str(),
                [e](benchmark::State& s) { BM_Forward<T>(s, e.width, e.height, e.depth, Modes::GPU); })
                ->Unit(benchmark::kMicrosecond);

#ifdef RAPIDHT_WITH_CUDA
            const size_t count = ElementCount(e.width, e.height, e.depth);
            benchmark::RegisterBenchmark(("PCIe/" + suffix).c_str(),
                [count](benchmark::State& s) { BM_DeviceRoundTrip<T>(s, count); })
                ->Unit(benchmark::kMicrosecond);
#endif
        }
    }
}

void RegisterDoubleOnly(const Extent* extents, size_t n, const char* rank)
{
    for (size_t i = 0; i < n; ++i) {
        const Extent e = extents[i];
        const std::string suffix = std::string("f64/") + rank + "/" + Describe(e);

        // RFFT is only implemented for the 1D case; in 2D/3D it silently falls
        // through to the CPU path, so measuring it there would be misleading.
        if (e.height == 0 && e.depth == 0) {
            benchmark::RegisterBenchmark(("RFFT/" + suffix).c_str(),
                [e](benchmark::State& s) { BM_Forward<double>(s, e.width, e.height, e.depth, Modes::RFFT); })
                ->Unit(benchmark::kMicrosecond);
        }

#ifdef RAPIDHT_BENCH_WITH_FFTW
        benchmark::RegisterBenchmark(("FFTW/" + suffix).c_str(),
            [e](benchmark::State& s) { BM_Fftw(s, e.width, e.height, e.depth); })
            ->Unit(benchmark::kMicrosecond);
#endif
    }
}

template <size_t N>
constexpr size_t CountOf(const Extent (&)[N])
{
    return N;
}

} // namespace

int main(int argc, char** argv)
{
    benchmark::Initialize(&argc, argv);

    // Volumes first: they are what the measurements are for.
    RegisterFor<double>(kExtents3D, CountOf(kExtents3D), "3D", "f64");
    RegisterFor<float>(kExtents3D, CountOf(kExtents3D), "3D", "f32");
    RegisterDoubleOnly(kExtents3D, CountOf(kExtents3D), "3D");

    RegisterFor<double>(kExtents2D, CountOf(kExtents2D), "2D", "f64");
    RegisterFor<float>(kExtents2D, CountOf(kExtents2D), "2D", "f32");
    RegisterDoubleOnly(kExtents2D, CountOf(kExtents2D), "2D");

    RegisterFor<double>(kExtents1D, CountOf(kExtents1D), "1D", "f64");
    RegisterDoubleOnly(kExtents1D, CountOf(kExtents1D), "1D");

    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
