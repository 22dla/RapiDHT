/*
 * Project: RapiDHT
 * File: benchmarks/bench_transform.cpp
 * Brief: Throughput measurements for the Hartley transform backends.
 *
 * Method notes, because they affect how the numbers should be read:
 *
 * - The forward transform is unnormalised, so applying it repeatedly to the
 *   same buffer multiplies the magnitude by N each time and reaches infinity
 *   within a few iterations. The input is therefore restored before every
 *   iteration, with the timer paused. Sizes are kept large enough that the
 *   pause/resume overhead stays negligible against the transform itself.
 *
 * - Results are reported as items per second, so different extents can be
 *   compared directly. Wall time alone cannot be, since the work grows with N.
 *
 * - The FFTW baseline uses FFTW_DHT. In one dimension that computes exactly
 *   what RapiDHT computes. In two and three dimensions FFTW applies a
 *   *separable* transform along each axis, whereas RapiDHT produces the true
 *   multidimensional Hartley transform and pays for an extra Bracewell pass to
 *   do so. The multidimensional comparison is therefore not like for like:
 *   RapiDHT is doing strictly more work, and a gap in FFTW's favour is
 *   expected rather than a defect.
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

using RapiDHT::HartleyTransform;
using RapiDHT::kCudaEnabled;
using RapiDHT::Modes;

namespace {

std::vector<double> MakeSignal(size_t count)
{
    std::vector<double> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = std::sin(0.7 * static_cast<double>(i)) + 1.0;
    }
    return data;
}

size_t ElementCount(size_t width, size_t height, size_t depth)
{
    return width * (height == 0 ? 1 : height) * (depth == 0 ? 1 : depth);
}

void BM_Forward(benchmark::State& state, size_t width, size_t height, size_t depth, Modes mode)
{
    const size_t count = ElementCount(width, height, depth);
    const std::vector<double> pristine = MakeSignal(count);
    std::vector<double> data = pristine;

    std::unique_ptr<HartleyTransform<double>> transform;
    try {
        transform = std::make_unique<HartleyTransform<double>>(width, height, depth, mode);
    } catch (const std::exception& error) {
        state.SkipWithError(error.what());
        return;
    }

    for (auto _ : state) {
        state.PauseTiming();
        std::memcpy(data.data(), pristine.data(), count * sizeof(double));
        state.ResumeTiming();

        transform->ForwardTransform(data.data());
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * count * sizeof(double)));
    state.counters["N"] = static_cast<double>(count);
}

#ifdef RAPIDHT_BENCH_WITH_FFTW
void BM_Fftw(benchmark::State& state, size_t width, size_t height, size_t depth)
{
    const size_t count = ElementCount(width, height, depth);
    const std::vector<double> pristine = MakeSignal(count);

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
        const fftw_r2r_kind kinds[3] = { FFTW_DHT, FFTW_DHT, FFTW_DHT };
        plan = fftw_plan_r2r_3d(static_cast<int>(depth), static_cast<int>(height),
            static_cast<int>(width), in, out, kinds[0], kinds[1], kinds[2], FFTW_MEASURE);
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

/*
 * Device memory the GPU backend needs just to hold its transform matrices.
 *
 * That backend multiplies by a dense cas matrix per axis, so an axis of length
 * L costs L^2 elements. In 2D and 3D the axes are short and this is nothing,
 * but in 1D the single axis is the whole signal: n = 262144 would ask for
 * 549 GB. Attempting it produced a stream of allocation failures and then a
 * segfault inside the harness, so such cases are not registered at all.
 */
size_t GpuMatrixBytes(const Extent& e)
{
    size_t elements = e.width * e.width;
    if (e.height > 0) {
        elements += e.height * e.height;
    }
    if (e.depth > 0) {
        elements += e.depth * e.depth;
    }
    return elements * sizeof(double);
}

// Deliberately conservative: large enough to keep the cases that show the
// trend, small enough to fit a modest card.
constexpr size_t kGpuMatrixBudget = size_t { 1 } << 30; // 1 GiB

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

// Powers of two only: FDHT1D rejects anything else.
const Extent kExtents1D[] = { { 1024, 0, 0 }, { 16384, 0, 0 }, { 262144, 0, 0 }, { 1048576, 0, 0 } };
const Extent kExtents2D[] = { { 128, 128, 0 }, { 512, 512, 0 }, { 1024, 1024, 0 } };
const Extent kExtents3D[] = { { 32, 32, 32 }, { 64, 64, 64 }, { 128, 128, 128 } };

void RegisterFor(const Extent* extents, size_t n, const char* rank)
{
    for (size_t i = 0; i < n; ++i) {
        const Extent e = extents[i];
        const std::string suffix = std::string(rank) + "/" + Describe(e);

        benchmark::RegisterBenchmark(("CPU/" + suffix).c_str(),
            [e](benchmark::State& s) { BM_Forward(s, e.width, e.height, e.depth, Modes::CPU); })
            ->Unit(benchmark::kMicrosecond);

        if (kCudaEnabled && GpuMatrixBytes(e) <= kGpuMatrixBudget) {
            benchmark::RegisterBenchmark(("GPU/" + suffix).c_str(),
                [e](benchmark::State& s) { BM_Forward(s, e.width, e.height, e.depth, Modes::GPU); })
                ->Unit(benchmark::kMicrosecond);
        }

        // RFFT is only implemented for the 1D case; in 2D/3D it silently falls
        // through to the CPU path, so measuring it there would be misleading.
        if (e.height == 0 && e.depth == 0) {
            benchmark::RegisterBenchmark(("RFFT/" + suffix).c_str(),
                [e](benchmark::State& s) { BM_Forward(s, e.width, e.height, e.depth, Modes::RFFT); })
                ->Unit(benchmark::kMicrosecond);
        }

#ifdef RAPIDHT_BENCH_WITH_FFTW
        benchmark::RegisterBenchmark(("FFTW/" + suffix).c_str(),
            [e](benchmark::State& s) { BM_Fftw(s, e.width, e.height, e.depth); })
            ->Unit(benchmark::kMicrosecond);
#endif
    }
}

} // namespace

int main(int argc, char** argv)
{
    benchmark::Initialize(&argc, argv);

    RegisterFor(kExtents1D, sizeof(kExtents1D) / sizeof(kExtents1D[0]), "1D");
    RegisterFor(kExtents2D, sizeof(kExtents2D) / sizeof(kExtents2D[0]), "2D");
    RegisterFor(kExtents3D, sizeof(kExtents3D) / sizeof(kExtents3D[0]), "3D");

    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
