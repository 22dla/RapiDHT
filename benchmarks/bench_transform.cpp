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

#include <rapidht/minkowski.h>
#include <rapidht/transform.h>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef RAPIDHT_BENCH_WITH_FFTW
#include <fftw3.h>
#endif

#ifdef RAPIDHT_WITH_CUDA
#include <cuda_runtime.h>

#include "internal/device_access.h"
#include "convolution_gpu.h"
#include "cufft_baseline.h"
#include "tasks_gpu.h"
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

#ifdef RAPIDHT_WITH_CUDA
/// Device memory in use, MiB, as the driver reports it (all processes).
double DeviceUsedMiB()
{
    size_t freeBytes = 0, totalBytes = 0;
    cudaDeviceSynchronize();
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
        return 0.0;
    }
    return static_cast<double>(totalBytes - freeBytes) / (1024.0 * 1024.0);
}
#endif

template <typename T>
void BM_Forward(
    benchmark::State& state, size_t width, size_t height, size_t depth, Modes mode, bool separable = false)
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

        if (separable) {
            transform->ForwardSeparable(data.data());
        } else {
            transform->ForwardTransform(data.data());
        }
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * count * sizeof(T)));
    state.counters["N"] = static_cast<double>(count);
}

#ifdef RAPIDHT_WITH_CUDA
/*
 * The transform on data that is already on the device: upload once outside the
 * loop, then transform in place.
 *
 * This is what the GPU figure would be if the caller kept the volume on the
 * card between operations, which is the realistic pattern for any pipeline and
 * the one the MPI decomposition assumes. The difference against the plain GPU
 * entry is the price of the current copy-per-call interface.
 */
template <typename T>
void BM_Resident(benchmark::State& state, size_t width, size_t height, size_t depth, bool separable = false,
    Modes mode = Modes::GPU)
{
    const size_t count = ElementCount(width, height, depth);
    const std::vector<T> pristine = MakeSignal<T>(count);

    // Device memory the transform holds beyond the volume: what the driver
    // reports in use after the transform is built and has run once, minus
    // what it reported with the volume alone (R-4 step 5.2). The first call
    // is untimed; it also takes any lazy allocation out of the measurement.
    std::unique_ptr<HartleyTransform<T>> transform;
    std::unique_ptr<RapiDHT::DeviceVolume<T>> volume;
    double usedBefore = 0.0;
    try {
        volume = std::make_unique<RapiDHT::DeviceVolume<T>>(count);
        volume->Upload(pristine.data());
        usedBefore = DeviceUsedMiB();
        transform = std::make_unique<HartleyTransform<T>>(width, height, depth, mode);
        if (separable) {
            transform->ForwardSeparable(*volume);
        } else {
            transform->ForwardTransform(*volume);
        }
    } catch (const std::exception& error) {
        state.SkipWithError(error.what());
        return;
    }
    state.counters["extraMiB"] = DeviceUsedMiB() - usedBefore;

    // The input is not restored between iterations: values grow by N each time,
    // but nothing here depends on their magnitude, and restoring would put a
    // transfer back into the measurement, which is exactly what this case is
    // built to exclude.
    for (auto _ : state) {
        if (separable) {
            transform->ForwardSeparable(*volume);
        } else {
            transform->ForwardTransform(*volume);
        }
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * count * sizeof(T)));
    state.counters["N"] = static_cast<double>(count);
}

/*
 * cuFFT computing the same thing, on data that is likewise already resident.
 *
 * Before timing anything the result is checked against this library's own GPU
 * path, which the test suite validates against a directly evaluated reference.
 * A baseline that computes something else is worse than no baseline, so a
 * mismatch skips the case rather than reporting a number.
 */
void BM_Cufft(benchmark::State& state, size_t width, size_t height, size_t depth)
{
    const size_t count = ElementCount(width, height, depth);
    const std::vector<float> pristine = MakeSignal<float>(count);

    float* device = nullptr;
    if (cudaMalloc(&device, count * sizeof(float)) != cudaSuccess) {
        state.SkipWithError("cudaMalloc failed");
        return;
    }

    try {
        rapidht_bench::CufftHartley cufft(static_cast<int>(width), static_cast<int>(height),
            static_cast<int>(depth));

        // Agreement check against our own GPU path.
        std::vector<float> viaCufft(count);
        cudaMemcpy(device, pristine.data(), count * sizeof(float), cudaMemcpyHostToDevice);
        cufft.Forward(device);
        cudaMemcpy(viaCufft.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> viaMatrix = pristine;
        HartleyTransform<float> ours(width, height, depth, Modes::GPU);
        ours.ForwardTransform(viaMatrix.data());

        double worst = 0.0;
        double scale = 0.0;
        for (size_t i = 0; i < count; ++i) {
            worst = std::max(worst, std::fabs(double(viaCufft[i]) - double(viaMatrix[i])));
            scale = std::max(scale, std::fabs(double(viaMatrix[i])));
        }
        if (worst > 1e-3 * std::max(scale, 1.0)) {
            cudaFree(device);
            state.SkipWithError("cuFFT baseline disagrees with the matrix path");
            return;
        }
        state.counters["extraMiB"]
            = static_cast<double>(cufft.ExtraDeviceBytes()) / (1024.0 * 1024.0);
        state.counters["spectrumMiB"] = static_cast<double>(cufft.SpectrumBytes()) / (1024.0 * 1024.0);
        state.counters["workspaceMiB"] = static_cast<double>(cufft.WorkspaceBytes()) / (1024.0 * 1024.0);

        cudaMemcpy(device, pristine.data(), count * sizeof(float), cudaMemcpyHostToDevice);
        for (auto _ : state) {
            cufft.Forward(device);
        }
    } catch (const std::exception& error) {
        cudaFree(device);
        state.SkipWithError(error.what());
        return;
    }

    cudaFree(device);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * count * sizeof(float)));
    state.counters["N"] = static_cast<double>(count);
}

/*
 * cuFFT's R2C transform alone, without the conversion to Hartley: the part of
 * the baseline no conversion pass can remove. The separable transform of this
 * library (FactoredResidentSeparable, no Bracewell pass) is the like-for-like
 * opponent of this figure, the true transform the opponent of cuFFT/.
 */
void BM_CufftR2C(benchmark::State& state, size_t width, size_t height, size_t depth)
{
    const size_t count = ElementCount(width, height, depth);
    const std::vector<float> pristine = MakeSignal<float>(count);

    float* device = nullptr;
    if (cudaMalloc(&device, count * sizeof(float)) != cudaSuccess) {
        state.SkipWithError("cudaMalloc failed");
        return;
    }
    try {
        rapidht_bench::CufftHartley cufft(static_cast<int>(width), static_cast<int>(height),
            static_cast<int>(depth));
        cudaMemcpy(device, pristine.data(), count * sizeof(float), cudaMemcpyHostToDevice);
        state.counters["extraMiB"]
            = static_cast<double>(cufft.ExtraDeviceBytes()) / (1024.0 * 1024.0);
        for (auto _ : state) {
            cufft.ForwardSpectrumOnly(device);
        }
    } catch (const std::exception& error) {
        cudaFree(device);
        state.SkipWithError(error.what());
        return;
    }
    cudaFree(device);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * count * sizeof(float)));
    state.counters["N"] = static_cast<double>(count);
}

/*
 * R-5 step 0.6: one way over the bus, host to device, for the input types of
 * the integer track. A tomogram arrives as uint8 / uint16; today the library
 * accepts only float, so the user converts on the host and sends twice (uint16)
 * or four times (uint8) the bytes. pageable is ordinary memory (what a user's
 * std::vector is: the driver stages it), pinned is cudaMallocHost.
 */
template <typename T>
void BM_HostToDevice(benchmark::State& state, size_t count, bool pinned)
{
    T* device = nullptr;
    if (cudaMalloc(&device, count * sizeof(T)) != cudaSuccess) {
        cudaGetLastError();
        state.SkipWithError("cudaMalloc failed");
        return;
    }
    T* host = nullptr;
    std::vector<T> pageable;
    if (pinned) {
        if (cudaMallocHost(&host, count * sizeof(T)) != cudaSuccess) {
            cudaGetLastError();
            cudaFree(device);
            state.SkipWithError("cudaMallocHost failed");
            return;
        }
        std::memset(host, 1, count * sizeof(T)); // touch every page before timing
    } else {
        pageable.assign(count, T { 1 });
        host = pageable.data();
    }

    for (auto _ : state) {
        cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice);
    }

    if (pinned) {
        cudaFreeHost(host);
    }
    cudaFree(device);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * count * sizeof(T)));
    state.counters["N"] = static_cast<double>(count);
    state.counters["MiB"] = static_cast<double>(count * sizeof(T)) / (1024.0 * 1024.0);
}

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

#ifdef RAPIDHT_WITH_CUDA
/*
 * R-5 step 0.2: cuFFT from an integer volume, the baseline of the integer
 * track (docs/R5_PLAN.md). Three ways to feed the integers to cuFFT, one row
 * family each:
 *
 *   cuFFTInt/     conversion kernel into the float output, then R2C
 *   cuFFTIntCb/   legacy load callback (static cuFFT, rapidht_cufft_callback)
 *   cuFFTIntLto/  LTO load callback (cufftXtSetJITCallback)
 *
 * Input: white integer noise over the full range of the type, resident on the
 * device; output: the float Hartley transform of that input, in a buffer of
 * its own (the input is not written). The offset subtracted on reading is half
 * the range -- it costs no reduction, and which offset is used does not change
 * the time.
 *
 * Before any timing each variant is checked against the CPU backend in double,
 * which is exact on integer input (every partial sum is far below 2^53). The
 * error is reported like accuracy_integer does, in LSB of the input
 * (|error| / sqrt(N)), over the non-zero frequencies: errLSB. A case whose
 * largest error passes 0.5 LSB is skipped instead of timed: rounding alone
 * gives 0.04 at 512^3 (2026-11-25_evgeny_r5_aa1_accuracy.txt), a wrong
 * transform gives the size of the signal. The zero frequency, about offset * N,
 * is checked on its own against float rounding.
 *
 * Memory: peakMiB is what the device holds once the case has run once --
 * input, output, the half spectrum and cuFFT's workspace -- by cudaMemGetInfo;
 * extraMiB is that minus the input and the output.
 */
template <typename TIn>
std::vector<TIn> MakeIntegerNoise(size_t count)
{
    std::vector<TIn> data(count);
    uint64_t state = 0x9E3779B97F4A7C15ull; // SplitMix64: reproducible, fast
    for (auto& v : data) {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        v = static_cast<TIn>((z ^ (z >> 31)) >> (64 - 8 * sizeof(TIn)));
    }
    return data;
}

/// The CPU transform in double of the integer noise for one type and extent.
/// Kept for the last pair asked for: the variants of one row run one after
/// another, and 512^3 takes seconds and a GiB.
template <typename TIn>
const std::vector<double>& IntegerReference(const std::vector<TIn>& input, const Extent& e)
{
    static std::string key;
    static std::vector<double> reference;
    const std::string wanted = std::to_string(sizeof(TIn)) + "/" + Describe(e);
    if (key != wanted) {
        key.clear();
        reference.assign(input.begin(), input.end());
        HartleyTransform<double> cpu(e.width, e.height, e.depth, Modes::CPU);
        cpu.ForwardTransform(reference.data());
        key = wanted;
    }
    return reference;
}

/// Largest error over the non-zero frequencies in LSB of the input, or a
/// negative number when the zero frequency is off by more than float rounding.
double IntegerErrorLsb(const std::vector<float>& got, const std::vector<double>& reference)
{
    const double dcError = std::fabs(double(got[0]) - reference[0]);
    if (dcError > 1e-6 * std::fabs(reference[0]) + 1.0) {
        return -1.0;
    }
    double worst = 0.0;
    for (size_t i = 1; i < got.size(); ++i) {
        worst = std::max(worst, std::fabs(double(got[i]) - reference[i]));
    }
    return worst / std::sqrt(static_cast<double>(got.size()));
}

/*
 * R-5 step 1.2: what the mean offset costs. The exact sum of the integers on
 * every step-th line along x -- step 1 is the rounded mean of the whole volume,
 * 64 and 512 the sampled policies of step 0.5 -- reduced on the device and
 * turned into the float offset a first pass would read, without a trip to the
 * host. Checked once against the sum on the host (integers: exact), then timed
 * as a whole: zeroing the sum, the reduction, the rounding kernel.
 */
template <typename TIn>
void BM_IntOffset(benchmark::State& state, Extent e, size_t step)
{
    const size_t count = ElementCount(e.width, e.height, e.depth);
    const size_t lines = count / e.width;
    const rapidht_bench::IntInput input
        = sizeof(TIn) == 1 ? rapidht_bench::IntInput::U8 : rapidht_bench::IntInput::U16;
    const std::vector<TIn> host = MakeIntegerNoise<TIn>(count);
    TIn* in = nullptr;
    unsigned long long* sum = nullptr;
    float* offset = nullptr;
    if (cudaMalloc(&in, count * sizeof(TIn)) != cudaSuccess || cudaMalloc(&sum, sizeof(*sum)) != cudaSuccess
        || cudaMalloc(&offset, sizeof(*offset)) != cudaSuccess) {
        cudaFree(in);
        cudaFree(sum);
        cudaGetLastError();
        state.SkipWithError("cudaMalloc failed");
        return;
    }
    try {
        cudaMemcpy(in, host.data(), count * sizeof(TIn), cudaMemcpyHostToDevice);
        unsigned long long expected = 0;
        size_t sampled = 0;
        for (size_t line = 0; line < lines; line += step, ++sampled) {
            for (size_t x = 0; x < e.width; ++x) {
                expected += host[line * e.width + x];
            }
        }
        const float expectedOffset
            = static_cast<float>(std::rint(static_cast<double>(expected) / static_cast<double>(sampled * e.width)));
        rapidht_bench::IntegerMeanOffset(in, input, e.width, lines, step, sum, offset);
        unsigned long long got = 0;
        float gotOffset = 0.0f;
        cudaMemcpy(&got, sum, sizeof(got), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gotOffset, offset, sizeof(gotOffset), cudaMemcpyDeviceToHost);
        if (got != expected || gotOffset != expectedOffset) {
            throw std::runtime_error("the device sum or offset disagrees with the host");
        }
        for (auto _ : state) {
            rapidht_bench::IntegerMeanOffset(in, input, e.width, lines, step, sum, offset);
            cudaDeviceSynchronize();
        }
        state.counters["readMiB"]
            = static_cast<double>(sampled * e.width * sizeof(TIn)) / (1024.0 * 1024.0);
    } catch (const std::exception& error) {
        cudaFree(in);
        cudaFree(sum);
        cudaFree(offset);
        state.SkipWithError(error.what());
        return;
    }
    cudaFree(in);
    cudaFree(sum);
    cudaFree(offset);
    state.counters["N"] = static_cast<double>(count);
}

template <typename TIn>
void BM_CufftInt(benchmark::State& state, Extent e, rapidht_bench::IntLoad load)
{
    using rapidht_bench::IntLoad;
    // Checked once per row: Google Benchmark calls the function again for
    // every repetition, and the check reads the whole volume on the CPU.
    static std::map<std::string, double> checked; // row -> largest error, LSB
    const std::string row = state.name();

    const size_t count = ElementCount(e.width, e.height, e.depth);
    const rapidht_bench::IntInput input = sizeof(TIn) == 1 ? rapidht_bench::IntInput::U8 : rapidht_bench::IntInput::U16;
    const double offset = std::ldexp(1.0, 8 * static_cast<int>(sizeof(TIn)) - 1); // half the range
    const std::vector<TIn> host = MakeIntegerNoise<TIn>(count);

    const double usedStart = DeviceUsedMiB();
    TIn* in = nullptr;
    float* out = nullptr;
    if (cudaMalloc(&in, count * sizeof(TIn)) != cudaSuccess || cudaMalloc(&out, count * sizeof(float)) != cudaSuccess) {
        cudaFree(in);
        cudaGetLastError(); // clear the allocation error
        state.SkipWithError("cudaMalloc failed: input and output do not fit");
        return;
    }
    try {
        std::unique_ptr<rapidht_bench::CufftIntHartley> cufft;
        if (load == IntLoad::LegacyCallback) {
#ifdef RAPIDHT_BENCH_WITH_CUFFT_CALLBACK
            cufft = rapidht_bench::MakeCufftIntCallback(
                static_cast<int>(e.width), static_cast<int>(e.height), static_cast<int>(e.depth), input, offset);
#else
            throw std::runtime_error("built without rapidht_cufft_callback");
#endif
        } else {
            cufft = rapidht_bench::MakeCufftIntHartley(static_cast<int>(e.width), static_cast<int>(e.height),
                static_cast<int>(e.depth), input, load, offset);
        }
        cudaMemcpy(in, host.data(), count * sizeof(TIn), cudaMemcpyHostToDevice);
        cufft->Forward(in, out); // untimed: the check, and any lazy allocation
        const double peak = DeviceUsedMiB() - usedStart;
        state.counters["peakMiB"] = peak;
        state.counters["extraMiB"]
            = peak - static_cast<double>(count * (sizeof(TIn) + sizeof(float))) / (1024.0 * 1024.0);

        if (checked.find(row) == checked.end()) {
            std::vector<float> got(count);
            cudaMemcpy(got.data(), out, count * sizeof(float), cudaMemcpyDeviceToHost);
            const double error = IntegerErrorLsb(got, IntegerReference(host, e));
            if (error < 0.0 || error > 0.5) {
                throw std::runtime_error(error < 0.0 ? "zero frequency disagrees with the CPU in double"
                                                     : "disagrees with the CPU in double: largest error "
                                                           + std::to_string(error) + " LSB");
            }
            checked[row] = error;
        }
        state.counters["errLSB"] = checked[row];

        for (auto _ : state) {
            cufft->Forward(in, out);
        }
    } catch (const std::exception& error) {
        cudaFree(in);
        cudaFree(out);
        cudaGetLastError();
        state.SkipWithError(error.what());
        return;
    }
    cudaFree(in);
    cudaFree(out);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * count * (sizeof(TIn) + sizeof(float))));
    state.counters["N"] = static_cast<double>(count);
}
#endif

#ifdef RAPIDHT_WITH_CUDA
/*
 * R-4 step 5.3: circular convolution, forward -- multiply -- back, on data
 * resident on the device. This is how a transform is used in practice, and the
 * place where Hartley's in-place, real arithmetic should tell: cuFFT needs a
 * half spectrum beside the volume, Hartley does not.
 *
 *   ConvHartley/      any kernel: true transform, pair product, true transform
 *   ConvHartleyEven/  kernel even along every axis (the hno-porous windows):
 *                     separable transform, pointwise product, separable again
 *   ConvCufft/        R2C, complex product, C2R
 *   ConvCufftEven/    the same with the real spectrum of an even kernel
 *
 * The kernel is a Gaussian, sigma = n/32 per axis; for ConvHartley/ and
 * ConvCufft/ its centre is moved off the origin, so that it is not even and the
 * odd part of the product is exercised. Each case is checked once before it is
 * timed: at four points against the definition of circular convolution summed
 * in double, and in full against cuFFT where cuFFT fits. A case that fails
 * either check is skipped with the error, not reported.
 */

} // namespace

/*
 * The public API has no way to touch a DeviceVolume between two transforms:
 * its device pointer is private, shared only with HartleyTransform. That is a
 * gap for convolution on the device (backlog), but R-4 does not change the
 * API, so the benchmark gets the pointer through the friendship: it is granted
 * to every specialisation of HartleyTransform, and this one, for a tag type of
 * the benchmark's own, does nothing else.
 */
namespace rapidht_bench {
struct PeekTag { };
} // namespace rapidht_bench

template <>
class RapiDHT::HartleyTransform<rapidht_bench::PeekTag> {
public:
    static float* Data(const RapiDHT::DeviceVolume<float>& volume)
    {
        return static_cast<float*>(volume.DeviceData());
    }
};

namespace {

using DevicePeek = HartleyTransform<rapidht_bench::PeekTag>;

/*
 * R-5 step 0.3: an integer volume through this library as it is today, with
 * no kernel that reads integers. The conversion kernel of cuFFTInt/ (the same
 * code, rapidht_bench::ConvertIntToFloat) writes x - offset into a float
 * DeviceVolume, the factored path transforms it in place, and a kernel of one
 * thread adds offset * N to the zero coefficient -- the cuFFT variants do that
 * inside their Hartley pass for free, this path has no pass of its own to put
 * it in. Same input, offset, check and counters as BM_CufftInt; the opponents
 * are the cuFFTInt, cuFFTIntCb and cuFFTIntLto rows of the same type and size.
 */
template <typename TIn>
void BM_IntConvertFactored(benchmark::State& state, Extent e, bool fromIntegers = false,
    const RapiDHT::internal::IntegerOffset* policy = nullptr)
{
    static std::map<std::string, double> checked; // row -> largest error, LSB
    const std::string row = state.name();

    const size_t count = ElementCount(e.width, e.height, e.depth);
    const rapidht_bench::IntInput input
        = sizeof(TIn) == 1 ? rapidht_bench::IntInput::U8 : rapidht_bench::IntInput::U16;
    const double offset = std::ldexp(1.0, 8 * static_cast<int>(sizeof(TIn)) - 1); // half the range
    const double dc = offset * static_cast<double>(count);
    const std::vector<TIn> host = MakeIntegerNoise<TIn>(count);

    const double usedStart = DeviceUsedMiB();
    TIn* in = nullptr;
    if (cudaMalloc(&in, count * sizeof(TIn)) != cudaSuccess) {
        cudaGetLastError();
        state.SkipWithError("cudaMalloc failed: the input does not fit");
        return;
    }
    try {
        RapiDHT::DeviceVolume<float> volume(count);
        HartleyTransform<float> transform(e.width, e.height, e.depth, Modes::GPUFactored);
        float* out = DevicePeek::Data(volume);
        auto forward = [&] {
            if (fromIntegers) {
                // R-5 step 2.3, IntFactored/: the first pass reads the
                // integers itself and returns the offset to H(0); synchronous.
                // R-5 step 3.2: with a policy, the library resolves the offset
                // (the means reduced on the device) instead of half the range.
                if (policy != nullptr) {
                    RapiDHT::internal::DeviceAccess::ForwardFromIntegers(transform, in, volume, *policy);
                } else {
                    RapiDHT::internal::DeviceAccess::ForwardFromIntegers(transform, in, volume, offset);
                }
                return;
            }
            rapidht_bench::ConvertIntToFloat(in, input, out, count, offset);
            transform.ForwardTransform(volume);
            rapidht_bench::AddToFirst(out, dc);
            cudaDeviceSynchronize();
        };
        cudaMemcpy(in, host.data(), count * sizeof(TIn), cudaMemcpyHostToDevice);
        forward(); // untimed: the check, and any lazy allocation
        const double peak = DeviceUsedMiB() - usedStart;
        state.counters["peakMiB"] = peak;
        state.counters["extraMiB"]
            = peak - static_cast<double>(count * (sizeof(TIn) + sizeof(float))) / (1024.0 * 1024.0);

        if (checked.find(row) == checked.end()) {
            std::vector<float> got(count);
            volume.Download(got.data());
            const double error = IntegerErrorLsb(got, IntegerReference(host, e));
            // With no offset the large mean costs 8-24 LSB on uint16 by
            // itself (docs/R5_PLAN.md, section 3, item 2): the check there
            // only catches a wrong result, not that error.
            const bool none = policy != nullptr && policy->kind == RapiDHT::internal::IntegerOffset::Kind::None;
            if (error < 0.0 || error > (none ? 32.0 : 0.5)) {
                throw std::runtime_error(error < 0.0 ? "zero frequency disagrees with the CPU in double"
                                                     : "disagrees with the CPU in double: largest error "
                                                           + std::to_string(error) + " LSB");
            }
            checked[row] = error;
        }
        state.counters["errLSB"] = checked[row];

        for (auto _ : state) {
            forward();
        }
    } catch (const std::exception& error) {
        cudaFree(in);
        cudaGetLastError();
        state.SkipWithError(error.what());
        return;
    }
    cudaFree(in);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * count * (sizeof(TIn) + sizeof(float))));
    state.counters["N"] = static_cast<double>(count);
}

/// Gaussian with sum 1 in the wrapped layout; separable, so the volume is a
/// product of three short tables.
std::vector<float> MakeConvKernel(const Extent& e, bool even)
{
    const size_t W = e.width, H = std::max<size_t>(e.height, 1), D = std::max<size_t>(e.depth, 1);
    auto axis = [](size_t n, long centre) {
        std::vector<double> t(n, 1.0);
        if (n == 1) {
            return t;
        }
        const double sigma = std::max(1.5, static_cast<double>(n) / 32.0);
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            long d = static_cast<long>(i) - centre;
            d = ((d % static_cast<long>(n)) + static_cast<long>(n)) % static_cast<long>(n);
            d = std::min(d, static_cast<long>(n) - d);
            t[i] = std::exp(-0.5 * static_cast<double>(d * d) / (sigma * sigma));
            sum += t[i];
        }
        for (auto& v : t) {
            v /= sum;
        }
        return t;
    };
    const std::vector<double> gx = axis(W, even ? 0 : 5), gy = axis(H, even ? 0 : -3), gz = axis(D, even ? 0 : 2);
    std::vector<float> g(W * H * D);
    for (size_t k = 0; k < D; ++k) {
        for (size_t j = 0; j < H; ++j) {
            for (size_t i = 0; i < W; ++i) {
                g[i + W * (j + H * k)] = static_cast<float>(gx[i] * gy[j] * gz[k]);
            }
        }
    }
    return g;
}

/// (f * g)(p), circular, summed in double: the definition, for spot checks.
double DirectConvolutionAt(const std::vector<float>& f, const std::vector<float>& g, const Extent& e, size_t p)
{
    const size_t W = e.width, H = std::max<size_t>(e.height, 1), D = std::max<size_t>(e.depth, 1);
    const size_t px = p % W, py = (p / W) % H, pz = p / (W * H);
    double sum = 0.0;
    for (size_t k = 0; k < D; ++k) {
        const size_t gk = (pz + D - k) % D;
        for (size_t j = 0; j < H; ++j) {
            const size_t gj = (py + H - j) % H;
            const float* frow = &f[W * (j + H * k)];
            const float* grow = &g[W * (gj + H * gk)];
            for (size_t i = 0; i < W; ++i) {
                sum += static_cast<double>(frow[i]) * grow[(px + W - i) % W];
            }
        }
    }
    return sum;
}

/// Empty when `result` (the convolution of `f` with `g`) agrees with the
/// definition at four points, else what went wrong.
std::string SpotCheckConvolution(const std::vector<float>& result, const std::vector<float>& f,
    const std::vector<float>& g, const Extent& e)
{
    const size_t count = result.size();
    for (size_t p : { size_t { 0 }, count / 3 + 7, (2 * count) / 3 + 11, count - 1 }) {
        const double reference = DirectConvolutionAt(f, g, e, p);
        if (std::fabs(result[p] - reference) > 1e-4 * std::max(std::fabs(reference), 1.0)) {
            return "convolution disagrees with its definition at index " + std::to_string(p) + ": "
                 + std::to_string(result[p]) + " against " + std::to_string(reference);
        }
    }
    return {};
}

/// Each case is checked in its first repetition only: the check at 1024^3
/// sums 4 * 2^30 products on the host.
bool FirstRun(const std::string& name)
{
    static std::set<std::string> seen;
    return seen.insert(name).second;
}

void BM_ConvHartley(benchmark::State& state, Extent e, bool even)
{
    const size_t count = ElementCount(e.width, e.height, e.depth);
    const int W = static_cast<int>(e.width), H = static_cast<int>(e.height), D = static_cast<int>(e.depth);
    const std::vector<float> pristine = MakeSignal<float>(count);
    const std::vector<float> g = MakeConvKernel(e, even);

    std::unique_ptr<HartleyTransform<float>> transform;
    std::unique_ptr<RapiDHT::DeviceVolume<float>> volume;
    std::unique_ptr<RapiDHT::DeviceVolume<float>> spectrum;
    auto transformOnce = [&](RapiDHT::DeviceVolume<float>& v) {
        if (even) {
            transform->ForwardSeparable(v);
        } else {
            transform->ForwardTransform(v);
        }
    };
    auto apply = [&] {
        transformOnce(*volume);
        float* data = DevicePeek::Data(*volume);
        const float* kernel = DevicePeek::Data(*spectrum);
        if (even) {
            rapidht_bench::MultiplyPointwise(data, kernel, count);
        } else {
            rapidht_bench::HartleyConvolveSpectrum(data, kernel, W, H, D);
        }
        transformOnce(*volume);
    };

    double usedBefore = 0.0;
    try {
        volume = std::make_unique<RapiDHT::DeviceVolume<float>>(count);
        volume->Upload(pristine.data());
        usedBefore = DeviceUsedMiB();
        transform = std::make_unique<HartleyTransform<float>>(e.width, e.height, e.depth, Modes::GPUFactored);
        // The kernel's spectrum, with the 1/N of the round trip folded in.
        spectrum = std::make_unique<RapiDHT::DeviceVolume<float>>(count);
        spectrum->Upload(g.data());
        transformOnce(*spectrum);
        std::vector<float> host(count);
        spectrum->Download(host.data());
        for (auto& v : host) {
            v /= static_cast<float>(count);
        }
        spectrum->Upload(host.data());
        apply(); // untimed: first-call allocations
    } catch (const std::exception& error) {
        state.SkipWithError(error.what());
        return;
    }
    // extraMiB is what the driver sees; the library allocates from a CUDA
    // memory pool that grows in chunks, so small buffers may not show there.
    // allocMiB is what the convolution itself holds: the kernel's spectrum.
    state.counters["extraMiB"] = DeviceUsedMiB() - usedBefore;
    state.counters["allocMiB"] = static_cast<double>(count * sizeof(float)) / (1024.0 * 1024.0);

    if (FirstRun(state.name())) {
        std::vector<float> ours(count);
        try {
            volume->Upload(pristine.data());
            apply();
            volume->Download(ours.data());
        } catch (const std::exception& error) {
            state.SkipWithError(error.what());
            return;
        }
        if (const std::string error = SpotCheckConvolution(ours, pristine, g, e); !error.empty()) {
            state.SkipWithError(error);
            return;
        }
        try {
            std::vector<float> theirs = pristine;
            float* device = nullptr;
            if (cudaMalloc(&device, count * sizeof(float)) != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed");
            }
            try {
                rapidht_bench::CufftConvolution cufft(W, H, D, g.data(), even, device);
                cudaMemcpy(device, pristine.data(), count * sizeof(float), cudaMemcpyHostToDevice);
                cufft.Apply(device);
                cudaMemcpy(theirs.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost);
            } catch (...) {
                cudaFree(device);
                throw;
            }
            cudaFree(device);
            double worst = 0.0, scale = 0.0;
            for (size_t i = 0; i < count; ++i) {
                worst = std::max(worst, std::fabs(double(ours[i]) - double(theirs[i])));
                scale = std::max(scale, std::fabs(double(theirs[i])));
            }
            if (worst > 1e-4 * std::max(scale, 1.0)) {
                state.SkipWithError("Hartley convolution disagrees with cuFFT's");
                return;
            }
        } catch (const std::exception&) {
            // cuFFT does not fit beside this case: the definition check above
            // stands alone. Clear the failed allocation's error, or the next
            // launch check of the library reports it as its own.
            cudaGetLastError();
        }
    }

    for (auto _ : state) {
        apply();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.counters["N"] = static_cast<double>(count);
}

void BM_ConvCufft(benchmark::State& state, Extent e, bool even)
{
    const size_t count = ElementCount(e.width, e.height, e.depth);
    const std::vector<float> pristine = MakeSignal<float>(count);
    const std::vector<float> g = MakeConvKernel(e, even);

    float* device = nullptr;
    if (cudaMalloc(&device, count * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        state.SkipWithError("cudaMalloc failed");
        return;
    }
    try {
        const double usedBefore = DeviceUsedMiB();
        rapidht_bench::CufftConvolution cufft(static_cast<int>(e.width), static_cast<int>(e.height),
            static_cast<int>(e.depth), g.data(), even, device);
        cudaMemcpy(device, pristine.data(), count * sizeof(float), cudaMemcpyHostToDevice);
        cufft.Apply(device); // untimed
        state.counters["extraMiB"] = DeviceUsedMiB() - usedBefore;
        state.counters["allocMiB"] = static_cast<double>(cufft.DeviceBytes()) / (1024.0 * 1024.0);

        if (FirstRun(state.name())) {
            std::vector<float> result(count);
            cudaMemcpy(device, pristine.data(), count * sizeof(float), cudaMemcpyHostToDevice);
            cufft.Apply(device);
            cudaMemcpy(result.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost);
            if (const std::string error = SpotCheckConvolution(result, pristine, g, e); !error.empty()) {
                throw std::runtime_error(error);
            }
        }
        for (auto _ : state) {
            cufft.Apply(device);
        }
    } catch (const std::exception& error) {
        cudaFree(device);
        cudaGetLastError(); // an out-of-memory error must not reach the next case
        state.SkipWithError(error.what());
        return;
    }
    cudaFree(device);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.counters["N"] = static_cast<double>(count);
}
#endif

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

// R-4 step 5.2: the full table of the factored path against cuFFT. These sizes
// get only FactoredResident[Separable]/ and the cuFFT pair, not the CPU and
// matrix entries: 1024^3 is 4 GiB in float, past what the matrix path and the
// cuFFT baseline fit on an 8 GiB card (the cuFFT entry is kept and reports
// the error), and 64^3 / 2048^2 add nothing the CPU table needs.
const Extent kExtentsR4Only3D[] = { { 64, 64, 64 }, { 1024, 1024, 1024 } };
const Extent kExtentsR4Only2D[] = { { 2048, 2048, 0 } };
// R-5 step 0.2: the integer baselines also at the two large sizes, to find
// where cuFFT stops fitting (1024^3 uint16 needs 2 GiB in, 4 out, 4 spectrum).
const Extent kExtentsR5Large3D[] = { { 1024, 1024, 512 }, { 1024, 1024, 1024 } };
// R-4 step 5.3: the largest convolution, 2 GiB in float.
const Extent kExtentsConvLargest[] = { { 1024, 1024, 512 } };

/*
 * R-4: the factored device path. Registered for every extent up to its 2048
 * limit, matrices or not -- it holds none. Factored/ copies across the bus like
 * GPU/, FactoredResident/ does not, like Resident/; compare those two pairs and
 * FactoredResident/ against cuFFT/.
 */
template <typename T>
void RegisterFactored(const Extent* extents, size_t n, const char* rank, const char* type)
{
#ifdef RAPIDHT_WITH_CUDA
    for (size_t i = 0; i < n; ++i) {
        const Extent e = extents[i];
        if (e.width > 2048 || e.height > 2048 || e.depth > 2048) {
            continue;
        }
        const std::string suffix = std::string(type) + "/" + rank + "/" + Describe(e);
        benchmark::RegisterBenchmark(("Factored/" + suffix).c_str(),
            [e](benchmark::State& s) { BM_Forward<T>(s, e.width, e.height, e.depth, Modes::GPUFactored); })
            ->Unit(benchmark::kMicrosecond);
        benchmark::RegisterBenchmark(("FactoredResident/" + suffix).c_str(),
            [e](benchmark::State& s) {
                BM_Resident<T>(s, e.width, e.height, e.depth, false, Modes::GPUFactored);
            })
            ->Unit(benchmark::kMicrosecond);
        if (e.height > 0) {
            benchmark::RegisterBenchmark(("FactoredResidentSeparable/" + suffix).c_str(),
                [e](benchmark::State& s) {
                    BM_Resident<T>(s, e.width, e.height, e.depth, true, Modes::GPUFactored);
                })
                ->Unit(benchmark::kMicrosecond);
        }
    }
#else
    (void)extents;
    (void)n;
    (void)rank;
    (void)type;
#endif
}

template <typename T>
void RegisterFor(const Extent* extents, size_t n, const char* rank, const char* type)
{
    for (size_t i = 0; i < n; ++i) {
        const Extent e = extents[i];
        const std::string suffix = std::string(type) + "/" + rank + "/" + Describe(e);

        benchmark::RegisterBenchmark(("CPU/" + suffix).c_str(),
            [e](benchmark::State& s) { BM_Forward<T>(s, e.width, e.height, e.depth, Modes::CPU); })
            ->Unit(benchmark::kMicrosecond);

        // The separable transform (no Bracewell pass), which is what an
        // even-kernel convolution runs: the difference against CPU/ and
        // Resident/ is what skipping the correction saves (B-102). 1D has no
        // correction to skip.
        const bool multiDimensional = (e.height > 0);
        if (multiDimensional) {
            benchmark::RegisterBenchmark(("CPUSeparable/" + suffix).c_str(),
                [e](benchmark::State& s) { BM_Forward<T>(s, e.width, e.height, e.depth, Modes::CPU, true); })
                ->Unit(benchmark::kMicrosecond);
        }

        if (kCudaEnabled && GpuMatrixBytes(e, sizeof(T)) <= kGpuMatrixBudget) {
            benchmark::RegisterBenchmark(("GPU/" + suffix).c_str(),
                [e](benchmark::State& s) { BM_Forward<T>(s, e.width, e.height, e.depth, Modes::GPU); })
                ->Unit(benchmark::kMicrosecond);

#ifdef RAPIDHT_WITH_CUDA
            benchmark::RegisterBenchmark(("Resident/" + suffix).c_str(),
                [e](benchmark::State& s) { BM_Resident<T>(s, e.width, e.height, e.depth); })
                ->Unit(benchmark::kMicrosecond);

            if (multiDimensional) {
                benchmark::RegisterBenchmark(("ResidentSeparable/" + suffix).c_str(),
                    [e](benchmark::State& s) { BM_Resident<T>(s, e.width, e.height, e.depth, true); })
                    ->Unit(benchmark::kMicrosecond);
            }

            const size_t count = ElementCount(e.width, e.height, e.depth);
            benchmark::RegisterBenchmark(("PCIe/" + suffix).c_str(),
                [count](benchmark::State& s) { BM_DeviceRoundTrip<T>(s, count); })
                ->Unit(benchmark::kMicrosecond);
#endif
        }
    }
}

/// R-5: integer input, 3D -- the cuFFT baselines (step 0.2) and this library
/// with a conversion kernel in front (step 0.3). Per extent and type all four
/// run one after another, so that the CPU reference of one type and size is
/// computed once (IntegerReference).
void RegisterIntegerInput(const Extent* extents, size_t n, const char* rank = "3D")
{
#ifdef RAPIDHT_WITH_CUDA
    if (!RapiDHT::IsGpuAvailable()) {
        return;
    }
    using rapidht_bench::IntLoad;
    const std::pair<const char*, IntLoad> variants[] = {
        { "cuFFTInt", IntLoad::Convert },
        { "cuFFTIntCb", IntLoad::LegacyCallback },
        { "cuFFTIntLto", IntLoad::LtoCallback },
    };
    for (size_t i = 0; i < n; ++i) {
        const Extent e = extents[i];
        for (const auto& [family, load] : variants) {
            benchmark::RegisterBenchmark((std::string(family) + "/u8/" + rank + "/" + Describe(e)).c_str(),
                [e, load = load](benchmark::State& s) { BM_CufftInt<uint8_t>(s, e, load); })
                ->Unit(benchmark::kMicrosecond);
        }
        benchmark::RegisterBenchmark((std::string("IntConvertFactored/u8/") + rank + "/" + Describe(e)).c_str(),
            [e](benchmark::State& s) { BM_IntConvertFactored<uint8_t>(s, e); })
            ->Unit(benchmark::kMicrosecond);
        benchmark::RegisterBenchmark((std::string("IntFactored/u8/") + rank + "/" + Describe(e)).c_str(),
            [e](benchmark::State& s) { BM_IntConvertFactored<uint8_t>(s, e, true); })
            ->Unit(benchmark::kMicrosecond);
        for (const auto& [family, load] : variants) {
            benchmark::RegisterBenchmark((std::string(family) + "/u16/" + rank + "/" + Describe(e)).c_str(),
                [e, load = load](benchmark::State& s) { BM_CufftInt<uint16_t>(s, e, load); })
                ->Unit(benchmark::kMicrosecond);
        }
        benchmark::RegisterBenchmark((std::string("IntConvertFactored/u16/") + rank + "/" + Describe(e)).c_str(),
            [e](benchmark::State& s) { BM_IntConvertFactored<uint16_t>(s, e); })
            ->Unit(benchmark::kMicrosecond);
        benchmark::RegisterBenchmark((std::string("IntFactored/u16/") + rank + "/" + Describe(e)).c_str(),
            [e](benchmark::State& s) { BM_IntConvertFactored<uint16_t>(s, e, true); })
            ->Unit(benchmark::kMicrosecond);
        // R-5 step 3.2: the same with each offset policy of the library
        // (IntFactored itself subtracts half the range, a known value).
        using RapiDHT::internal::IntegerOffset;
        static const std::pair<const char*, IntegerOffset> policies[] = {
            { "IntFactoredNone", IntegerOffset::NoOffset() },
            { "IntFactoredEvery512", IntegerOffset::MeanOfEvery(512) },
            { "IntFactoredEvery64", IntegerOffset::MeanOfEvery(64) },
            { "IntFactoredMean", IntegerOffset::MeanOfAll() },
        };
        for (const auto& [family, policy] : policies) {
            const IntegerOffset* p = &policy;
            benchmark::RegisterBenchmark((std::string(family) + "/u8/" + rank + "/" + Describe(e)).c_str(),
                [e, p](benchmark::State& s) { BM_IntConvertFactored<uint8_t>(s, e, true, p); })
                ->Unit(benchmark::kMicrosecond);
            benchmark::RegisterBenchmark((std::string(family) + "/u16/" + rank + "/" + Describe(e)).c_str(),
                [e, p](benchmark::State& s) { BM_IntConvertFactored<uint16_t>(s, e, true, p); })
                ->Unit(benchmark::kMicrosecond);
        }
    }
#else
    (void)extents;
    (void)n;
    (void)rank;
#endif
}

#ifdef RAPIDHT_WITH_CUDA
/*
 * R-5 step 4.3: the inverse into integers -- from a spectrum on the device to
 * uint8 / uint16 voxels, the task of stage 4:
 *   IntInverse         this library's integer inverse (the mean off H(0), the
 *                      conversion folded into the last pass);
 *   IntInverseConvert  this library's float inverse, then a conversion kernel;
 *   cuFFTIntInv        cuFFT C2R from its half spectrum, then the same kernel;
 *   cuFFTIntInvLto     cuFFT C2R with an LTO store callback writing integers.
 * Every inverse overwrites its input, so the spectrum is restored from a copy
 * before each iteration with the timer paused. Before timing, each variant
 * must give back the integer input voxel for voxel (white noise; the forward
 * transform with this library's default offset policy, or cuFFT's own R2C).
 */
enum class IntInverse { Ours,
    OursConvert,
    Cufft,
    CufftLto };

template <typename TOut>
void BM_IntInverse(benchmark::State& state, Extent e, IntInverse variant)
{
    const size_t count = ElementCount(e.width, e.height, e.depth);
    const rapidht_bench::IntInput type = sizeof(TOut) == 1 ? rapidht_bench::IntInput::U8 : rapidht_bench::IntInput::U16;
    const std::vector<TOut> host = MakeIntegerNoise<TOut>(count);
    const double usedStart = DeviceUsedMiB();
    TOut* ints = nullptr;
    if (cudaMalloc(&ints, count * sizeof(TOut)) != cudaSuccess) {
        cudaGetLastError();
        state.SkipWithError("cudaMalloc failed: the integers do not fit");
        return;
    }
    try {
        cudaMemcpy(ints, host.data(), count * sizeof(TOut), cudaMemcpyHostToDevice);
        std::function<void()> restore;
        std::function<void()> inverse;
        std::unique_ptr<rapidht_bench::CufftIntInverse> cufft;
        std::unique_ptr<HartleyTransform<float>> transform;
        std::unique_ptr<RapiDHT::DeviceVolume<float>> work;
        std::unique_ptr<RapiDHT::DeviceVolume<float>> pristine;
        if (variant == IntInverse::Cufft || variant == IntInverse::CufftLto) {
            cufft = rapidht_bench::MakeCufftIntInverse(static_cast<int>(e.width), static_cast<int>(e.height),
                static_cast<int>(e.depth), type, variant == IntInverse::CufftLto);
            cufft->SetSpectrumFrom(ints);
            restore = [&] { cufft->Restore(); };
            inverse = [&] { cufft->Inverse(ints); };
        } else {
            transform = std::make_unique<HartleyTransform<float>>(e.width, e.height, e.depth, Modes::GPUFactored);
            work = std::make_unique<RapiDHT::DeviceVolume<float>>(count);
            pristine = std::make_unique<RapiDHT::DeviceVolume<float>>(count);
            RapiDHT::internal::DeviceAccess::ForwardFromIntegers(
                *transform, ints, *pristine, RapiDHT::internal::IntegerOffset::Default());
            float* to = DevicePeek::Data(*work);
            const float* from = DevicePeek::Data(*pristine);
            restore = [&, to, from] {
                cudaMemcpy(to, from, count * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaDeviceSynchronize();
            };
            if (variant == IntInverse::Ours) {
                inverse = [&] {
                    RapiDHT::internal::DeviceAccess::InverseToIntegers(
                        *transform, *work, ints, RapiDHT::internal::IntegerOffset::MeanOfAll());
                };
            } else {
                inverse = [&, to] {
                    transform->InverseTransform(*work); // forward, then 1/N
                    rapidht_bench::ConvertFloatToInt(to, type, ints, count, 1.0f);
                    cudaDeviceSynchronize();
                };
            }
        }
        // The check: the input back, every voxel.
        cudaMemset(ints, 0, count * sizeof(TOut));
        restore();
        inverse();
        const double peak = DeviceUsedMiB() - usedStart;
        std::vector<TOut> back(count);
        cudaMemcpy(back.data(), ints, count * sizeof(TOut), cudaMemcpyDeviceToHost);
        size_t differ = 0;
        for (size_t i = 0; i < count; ++i) {
            differ += back[i] != host[i] ? 1 : 0;
        }
        if (differ != 0) {
            throw std::runtime_error(std::to_string(differ) + " voxels not restored");
        }
        for (auto _ : state) {
            state.PauseTiming();
            restore();
            state.ResumeTiming();
            inverse();
        }
        state.counters["peakMiB"] = peak;
    } catch (const std::exception& error) {
        cudaFree(ints);
        state.SkipWithError(error.what());
        return;
    }
    cudaFree(ints);
    state.counters["N"] = static_cast<double>(count);
}

void RegisterIntegerInverse(const Extent* extents, size_t n)
{
    if (!RapiDHT::IsGpuAvailable()) {
        return;
    }
    const std::pair<const char*, IntInverse> variants[] = {
        { "IntInverse", IntInverse::Ours },
        { "IntInverseConvert", IntInverse::OursConvert },
        { "cuFFTIntInv", IntInverse::Cufft },
        { "cuFFTIntInvLto", IntInverse::CufftLto },
    };
    for (size_t i = 0; i < n; ++i) {
        const Extent e = extents[i];
        for (const auto& [family, variant] : variants) {
            benchmark::RegisterBenchmark((std::string(family) + "/u8/3D/" + Describe(e)).c_str(),
                [e, variant = variant](benchmark::State& s) { BM_IntInverse<uint8_t>(s, e, variant); })
                ->Unit(benchmark::kMicrosecond);
            benchmark::RegisterBenchmark((std::string(family) + "/u16/3D/" + Describe(e)).c_str(),
                [e, variant = variant](benchmark::State& s) { BM_IntInverse<uint16_t>(s, e, variant); })
                ->Unit(benchmark::kMicrosecond);
        }
    }
}
#endif

#ifdef RAPIDHT_WITH_CUDA
/*
 * R-5 step 5.1: the end-to-end tasks on an integer tomogram (docs/R5_PLAN.md,
 * stage 5; the device side is benchmarks/tasks_gpu.h). Rows are named
 *
 *     Task<N>/<implementation>/<type>[/s<sigma>]/<from>/3D/<extent>
 *
 * and <from> says where the task starts and ends:
 *   device    the input already on the device; T1 leaves the spectrum there,
 *             T2 the filtered integers, T3 brings its 128 bins to the host;
 *   pageable  the input copied from ordinary host memory inside the timing,
 *             and T2's output copied back into it;
 *   pinned    the same from and to page-locked memory (cudaMallocHost).
 *
 *   T1 "spectrum"   IntFactored    the library from the integers (default offset policy)
 *                   FloatFactored  the library as it is today: a float volume
 *                                  from the host (converted there, untimed)
 *                   cuFFTLto       R2C with the LTO load callback + Hartley pass
 *                                  (offset half the range: free, step 0.2)
 *   T2 "filter"     HartleySeparable  integers -> separable forward -> product with
 *                                  the separable Gaussian -> separable inverse -> integers
 *                   Hartley        the same with the true transform: for a kernel
 *                                  even along every axis its spectrum is the same
 *                                  table and the product the same (the odd part is
 *                                  0), and the last pass of the true inverse writes
 *                                  the integers itself (step 4.2), where the
 *                                  separable one needs a conversion pass
 *                   cuFFTLto       LTO load, R2C, product, C2R, LTO store
 *                   Spatial        three passes in space (sigma 1 and 2 only)
 *   T3 "power"      Hartley        true forward, H^2 into 128 radial bins
 *                   HartleySeparable  separable forward, S^2 into the same bins
 *                                  (equal sums: a bin is closed under sign flips)
 *                   cuFFTLto       LTO load, R2C, |F|^2 of the half spectrum
 *
 * The input is white integer noise over the full range, as in the other R-5
 * rows. Each row is checked once against the CPU in double before it is timed:
 * T1 by the largest error in LSB of the input (errLSB, as IntFactored/), T2 by
 * the share of voxels equal to the rounded filter of the definition (matchPct;
 * maxDiff the largest difference, in LSB; floorPct the same share for the exact
 * filter rounded to float first -- what any float output can reach), T3 by the largest relative error of
 * a bin (relErr). A result that is plainly wrong -- T1 over 0.5 LSB (32 for the
 * float path, which has no offset), T2 under 99 % equal, T3 over 1e-3 -- is
 * skipped with the error instead of timed. peakMiB is what the device holds
 * once the row has run, by cudaMemGetInfo, as in the other R-5 rows.
 */
enum class TaskFrom { Device,
    Pageable,
    Pinned };

const char* TaskFromName(TaskFrom from)
{
    return from == TaskFrom::Device ? "device" : (from == TaskFrom::Pageable ? "pageable" : "pinned");
}

/// count elements of device memory; throws when they do not fit.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count)
    {
        if (cudaMalloc(&_data, count * sizeof(T)) != cudaSuccess) {
            cudaGetLastError();
            _data = nullptr;
            throw std::runtime_error("cudaMalloc failed: " + std::to_string(count * sizeof(T) >> 20) + " MiB");
        }
    }
    ~DeviceBuffer() { cudaFree(_data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* Data() const { return _data; }

private:
    T* _data = nullptr;
};

/// Host memory of a task: an ordinary vector, or page-locked.
template <typename T>
class HostBuffer {
public:
    HostBuffer(size_t count, bool pinned):
        _pinned(pinned)
    {
        if (pinned) {
            if (cudaMallocHost(&_data, count * sizeof(T)) != cudaSuccess) {
                cudaGetLastError();
                _data = nullptr;
                throw std::runtime_error("cudaMallocHost failed");
            }
            std::memset(_data, 0, count * sizeof(T)); // touch every page before timing
        } else {
            _vector.assign(count, T {});
            _data = _vector.data();
        }
    }
    ~HostBuffer()
    {
        if (_pinned && _data != nullptr) {
            cudaFreeHost(_data);
        }
    }
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    T* Data() const { return _data; }

private:
    bool _pinned;
    std::vector<T> _vector;
    T* _data = nullptr;
};

template <typename TIn>
rapidht_bench::IntInput IntTypeOf()
{
    return sizeof(TIn) == 1 ? rapidht_bench::IntInput::U8 : rapidht_bench::IntInput::U16;
}

/// Runs a task row: once untimed (lazy allocations; the peak), then the loop.
/// `check` runs once per row name and returns the error to skip with, if any.
void RunTask(benchmark::State& state, size_t count, double usedStart, const std::function<void()>& task,
    const std::function<std::string()>& check)
{
    static std::set<std::string> checked;
    task();
    state.counters["peakMiB"] = DeviceUsedMiB() - usedStart;
    if (checked.find(state.name()) == checked.end()) {
        if (const std::string error = check(); !error.empty()) {
            throw std::runtime_error(error);
        }
        checked.insert(state.name());
    }
    for (auto _ : state) {
        task();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * count));
    state.counters["N"] = static_cast<double>(count);
}

/// The last row's accuracy counters, kept for the repetitions that skip the check.
std::map<std::string, std::map<std::string, double>>& TaskAccuracy()
{
    static std::map<std::string, std::map<std::string, double>> accuracy;
    return accuracy;
}

void ReportTaskAccuracy(benchmark::State& state)
{
    for (const auto& [name, value] : TaskAccuracy()[state.name()]) {
        state.counters[name] = value;
    }
}

enum class Task1Impl { IntFactored,
    FloatFactored,
    CufftLto };

template <typename TIn>
void BM_Task1(benchmark::State& state, Extent e, Task1Impl impl, TaskFrom from)
{
    const size_t count = ElementCount(e.width, e.height, e.depth);
    const std::vector<TIn> host = MakeIntegerNoise<TIn>(count);
    const double usedStart = DeviceUsedMiB();
    try {
        RapiDHT::DeviceVolume<float> volume(count); // the spectrum; FloatFactored's input too
        float* out = DevicePeek::Data(volume);
        std::unique_ptr<DeviceBuffer<TIn>> in;
        std::unique_ptr<HostBuffer<TIn>> stagedInt;
        std::unique_ptr<HostBuffer<float>> stagedFloat;
        std::unique_ptr<HartleyTransform<float>> transform;
        std::unique_ptr<rapidht_bench::CufftIntHartley> cufft;
        std::function<void()> upload = [] { };
        if (impl == Task1Impl::FloatFactored) {
            stagedFloat = std::make_unique<HostBuffer<float>>(count, from == TaskFrom::Pinned);
            std::copy(host.begin(), host.end(), stagedFloat->Data());
            upload = [&] { cudaMemcpy(out, stagedFloat->Data(), count * sizeof(float), cudaMemcpyHostToDevice); };
        } else {
            in = std::make_unique<DeviceBuffer<TIn>>(count);
            cudaMemcpy(in->Data(), host.data(), count * sizeof(TIn), cudaMemcpyHostToDevice);
            if (from != TaskFrom::Device) {
                stagedInt = std::make_unique<HostBuffer<TIn>>(count, from == TaskFrom::Pinned);
                std::copy(host.begin(), host.end(), stagedInt->Data());
                upload = [&] {
                    cudaMemcpy(in->Data(), stagedInt->Data(), count * sizeof(TIn), cudaMemcpyHostToDevice);
                };
            }
        }
        std::function<void()> compute;
        if (impl == Task1Impl::CufftLto) {
            cufft = rapidht_bench::MakeCufftIntHartley(static_cast<int>(e.width), static_cast<int>(e.height),
                static_cast<int>(e.depth), IntTypeOf<TIn>(), rapidht_bench::IntLoad::LtoCallback,
                std::ldexp(1.0, 8 * static_cast<int>(sizeof(TIn)) - 1));
            compute = [&] { cufft->Forward(in->Data(), out); };
        } else {
            transform = std::make_unique<HartleyTransform<float>>(e.width, e.height, e.depth, Modes::GPUFactored);
            if (impl == Task1Impl::IntFactored) {
                compute = [&] {
                    RapiDHT::internal::DeviceAccess::ForwardFromIntegers(
                        *transform, in->Data(), volume, RapiDHT::internal::IntegerOffset::Default());
                };
            } else {
                compute = [&] { transform->ForwardTransform(volume); };
            }
        }
        RunTask(
            state, count, usedStart,
            [&] {
                upload();
                compute();
                cudaDeviceSynchronize();
            },
            [&]() -> std::string {
                std::vector<float> got(count);
                volume.Download(got.data());
                const double error = IntegerErrorLsb(got, IntegerReference(host, e));
                TaskAccuracy()[state.name()]["errLSB"] = error;
                if (error < 0.0) {
                    return "zero frequency disagrees with the CPU in double";
                }
                if (error > (impl == Task1Impl::FloatFactored ? 32.0 : 0.5)) {
                    return "disagrees with the CPU in double: " + std::to_string(error) + " LSB";
                }
                return {};
            });
        ReportTaskAccuracy(state);
    } catch (const std::exception& error) {
        cudaGetLastError();
        state.SkipWithError(error.what());
    }
}

/*
 * The filter of T2 by its definition, in double: the three circular passes of
 * the 6 sigma + 1 taps, then rounded and clamped. Kept for the last (type,
 * extent, sigma): the rows of one sigma run one after another.
 */
template <typename TIn>
struct FilterTruth {
    std::vector<TIn> rounded; ///< the filter in double, rounded and clamped
    double floorShare = 0.0; ///< share of voxels where rounding through float first gives the same
};

template <typename TIn>
const FilterTruth<TIn>& FilterReference(const std::vector<TIn>& input, const Extent& e, double sigma)
{
    static std::string key;
    static FilterTruth<TIn> reference;
    const std::string wanted = std::to_string(sizeof(TIn)) + "/" + Describe(e) + "/" + std::to_string(sigma);
    if (key == wanted) {
        return reference;
    }
    key.clear();
    const std::vector<double> taps = rapidht_bench::GaussianTaps(sigma);
    const long long r = static_cast<long long>(taps.size()) - 1;
    const long long W = static_cast<long long>(e.width), H = static_cast<long long>(e.height),
                    D = static_cast<long long>(e.depth);
    std::vector<double> a(input.begin(), input.end());
    // One pass along an axis of length n, `step` apart, for `outer` groups of
    // `rows` contiguous runs of W (a group: a line along x, or a slab).
    auto pass = [&](long long outer, long long groupStride, long long step, long long n, long long rows) {
#pragma omp parallel for
        for (long long o = 0; o < outer; ++o) {
            std::vector<double> slab(static_cast<size_t>(n * rows));
            const long long base = o * groupStride;
            for (long long i = 0; i < n; ++i) {
                for (long long x = 0; x < rows; ++x) {
                    slab[static_cast<size_t>(i * rows + x)] = a[static_cast<size_t>(base + i * step + x)];
                }
            }
            for (long long i = 0; i < n; ++i) {
                for (long long x = 0; x < rows; ++x) {
                    double s = taps[0] * slab[static_cast<size_t>(i * rows + x)];
                    for (long long d = 1; d <= r; ++d) {
                        s += taps[static_cast<size_t>(d)]
                           * (slab[static_cast<size_t>(((i - d + n) % n) * rows + x)]
                               + slab[static_cast<size_t>(((i + d) % n) * rows + x)]);
                    }
                    a[static_cast<size_t>(base + i * step + x)] = s;
                }
            }
        }
    };
    pass(H * D, W, 1, W, 1); // x: each line, elements 1 apart
    pass(D, W * H, W, H, W); // y: each z slab, rows of x
    pass(H, W, W * H, D, W); // z: each y, rows of x
    const double top = rapidht_bench::IntInput::U8 == IntTypeOf<TIn>() ? 255.0 : 65535.0;
    // The floor of a float output: the exact value rounded to float, then to
    // an integer. Near 32768 a float step is 1/256 LSB, so a voxel within that
    // of a half rounds the other way -- about 0.2 % of uniform values.
    auto toInteger = [top](double v) { return static_cast<TIn>(std::min(std::max(std::rint(v), 0.0), top)); };
    reference.rounded.resize(a.size());
    size_t same = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        reference.rounded[i] = toInteger(a[i]);
        same += toInteger(static_cast<float>(a[i])) == reference.rounded[i] ? 1 : 0;
    }
    reference.floorShare = static_cast<double>(same) / static_cast<double>(a.size());
    key = wanted;
    return reference;
}

enum class Task2Impl { Hartley,
    HartleySeparable,
    CufftLto,
    Spatial };

template <typename TIn>
void BM_Task2(benchmark::State& state, Extent e, Task2Impl impl, double sigma, TaskFrom from)
{
    const size_t count = ElementCount(e.width, e.height, e.depth);
    const int W = static_cast<int>(e.width), H = static_cast<int>(e.height), D = static_cast<int>(e.depth);
    const std::vector<TIn> host = MakeIntegerNoise<TIn>(count);
    const std::vector<double> taps = rapidht_bench::GaussianTaps(sigma);
    const std::vector<float> sx = rapidht_bench::GaussianAxisSpectrum(taps, e.width);
    const std::vector<float> sy = rapidht_bench::GaussianAxisSpectrum(taps, e.height);
    const std::vector<float> sz = rapidht_bench::GaussianAxisSpectrum(taps, e.depth);
    const double usedStart = DeviceUsedMiB();
    try {
        DeviceBuffer<TIn> in(count);
        DeviceBuffer<TIn> out(count);
        cudaMemcpy(in.Data(), host.data(), count * sizeof(TIn), cudaMemcpyHostToDevice);
        std::unique_ptr<HostBuffer<TIn>> stagedIn, stagedOut;
        if (from != TaskFrom::Device) {
            stagedIn = std::make_unique<HostBuffer<TIn>>(count, from == TaskFrom::Pinned);
            stagedOut = std::make_unique<HostBuffer<TIn>>(count, from == TaskFrom::Pinned);
            std::copy(host.begin(), host.end(), stagedIn->Data());
        }
        std::unique_ptr<HartleyTransform<float>> transform;
        std::unique_ptr<RapiDHT::DeviceVolume<float>> volume;
        std::unique_ptr<DeviceBuffer<float>> tx, ty, tz;
        std::unique_ptr<rapidht_bench::CufftGaussianFilter> cufft;
        std::unique_ptr<rapidht_bench::SpatialGaussian> spatial;
        std::function<void()> compute;
        if (impl == Task2Impl::Hartley || impl == Task2Impl::HartleySeparable) {
            const bool separable = impl == Task2Impl::HartleySeparable;
            transform = std::make_unique<HartleyTransform<float>>(e.width, e.height, e.depth, Modes::GPUFactored);
            volume = std::make_unique<RapiDHT::DeviceVolume<float>>(count);
            tx = std::make_unique<DeviceBuffer<float>>(sx.size());
            ty = std::make_unique<DeviceBuffer<float>>(sy.size());
            tz = std::make_unique<DeviceBuffer<float>>(sz.size());
            cudaMemcpy(tx->Data(), sx.data(), sx.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(ty->Data(), sy.data(), sy.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(tz->Data(), sz.data(), sz.size() * sizeof(float), cudaMemcpyHostToDevice);
            compute = [&, separable] {
                using RapiDHT::internal::DeviceAccess;
                using RapiDHT::internal::IntegerOffset;
                DeviceAccess::ForwardFromIntegers(
                    *transform, in.Data(), *volume, IntegerOffset::Default(), separable);
                rapidht_bench::MultiplySeparable(
                    DevicePeek::Data(*volume), tx->Data(), ty->Data(), tz->Data(), W, H, D);
                // The kernel sums to 1, so the rounded mean of the output is
                // read off its H(0), as the integer inverse of step 4.3 does.
                DeviceAccess::InverseToIntegers(
                    *transform, *volume, out.Data(), IntegerOffset::MeanOfAll(), nullptr, separable);
            };
        } else if (impl == Task2Impl::CufftLto) {
            cufft = std::make_unique<rapidht_bench::CufftGaussianFilter>(W, H, D, IntTypeOf<TIn>(), sx, sy, sz);
            compute = [&] { cufft->Apply(in.Data(), out.Data()); };
        } else {
            spatial = std::make_unique<rapidht_bench::SpatialGaussian>(W, H, D, IntTypeOf<TIn>(), taps);
            compute = [&] { spatial->Apply(in.Data(), out.Data()); };
        }
        RunTask(
            state, count, usedStart,
            [&] {
                if (from != TaskFrom::Device) {
                    cudaMemcpy(in.Data(), stagedIn->Data(), count * sizeof(TIn), cudaMemcpyHostToDevice);
                }
                compute();
                if (from != TaskFrom::Device) {
                    cudaMemcpy(stagedOut->Data(), out.Data(), count * sizeof(TIn), cudaMemcpyDeviceToHost);
                }
                cudaDeviceSynchronize();
            },
            [&]() -> std::string {
                std::vector<TIn> got(count);
                cudaMemcpy(got.data(), out.Data(), count * sizeof(TIn), cudaMemcpyDeviceToHost);
                const FilterTruth<TIn>& truth = FilterReference(host, e, sigma);
                const std::vector<TIn>& reference = truth.rounded;
                size_t equal = 0;
                int worst = 0;
                for (size_t i = 0; i < count; ++i) {
                    const int diff = std::abs(static_cast<int>(got[i]) - static_cast<int>(reference[i]));
                    equal += diff == 0 ? 1 : 0;
                    worst = std::max(worst, diff);
                }
                const double share = static_cast<double>(equal) / static_cast<double>(count);
                TaskAccuracy()[state.name()]["matchPct"] = 100.0 * share;
                TaskAccuracy()[state.name()]["maxDiff"] = worst;
                TaskAccuracy()[state.name()]["floorPct"] = 100.0 * truth.floorShare;
                if (share < 0.99) {
                    return "only " + std::to_string(100.0 * share) + " % of the voxels agree with the definition";
                }
                return {};
            });
        ReportTaskAccuracy(state);
    } catch (const std::exception& error) {
        cudaGetLastError();
        state.SkipWithError(error.what());
    }
}

/// T3 by the definition: the bin sums of H^2 of the CPU transform in double,
/// and the number of frequencies in each bin. Kept for the last (type, extent).
struct PowerProfile {
    std::vector<double> mean; ///< sum / count; 0 for an empty bin
    std::vector<double> count;
};

template <typename TIn>
const PowerProfile& PowerReference(const std::vector<TIn>& input, const Extent& e)
{
    static std::string key;
    static PowerProfile profile;
    const std::string wanted = std::to_string(sizeof(TIn)) + "/" + Describe(e);
    if (key == wanted) {
        return profile;
    }
    key.clear();
    const std::vector<double>& spectrum = IntegerReference(input, e);
    const long long W = static_cast<long long>(e.width), H = static_cast<long long>(e.height),
                    D = static_cast<long long>(e.depth);
    const long long longest = std::max({ W, H, D });
    constexpr int kBins = rapidht_bench::kRadialBins;
    std::vector<double> sum(kBins, 0.0), number(kBins, 0.0);
#pragma omp parallel
    {
        std::vector<double> s(kBins, 0.0), c(kBins, 0.0);
#pragma omp for
        for (long long z = 0; z < D; ++z) {
            const long long kz = rapidht_bench::ScaledFrequency(z, D, longest);
            for (long long y = 0; y < H; ++y) {
                const long long ky = rapidht_bench::ScaledFrequency(y, H, longest);
                for (long long x = 0; x < W; ++x) {
                    const long long kx = rapidht_bench::ScaledFrequency(x, W, longest);
                    const int b = rapidht_bench::RadialBin(kx * kx + ky * ky + kz * kz, longest);
                    const double h = spectrum[static_cast<size_t>(x + W * (y + H * z))];
                    s[static_cast<size_t>(b)] += h * h;
                    c[static_cast<size_t>(b)] += 1.0;
                }
            }
        }
#pragma omp critical
        for (int b = 0; b < kBins; ++b) {
            sum[static_cast<size_t>(b)] += s[static_cast<size_t>(b)];
            number[static_cast<size_t>(b)] += c[static_cast<size_t>(b)];
        }
    }
    profile.count = number;
    profile.mean.assign(kBins, 0.0);
    for (int b = 0; b < kBins; ++b) {
        if (number[static_cast<size_t>(b)] > 0.0) {
            profile.mean[static_cast<size_t>(b)] = sum[static_cast<size_t>(b)] / number[static_cast<size_t>(b)];
        }
    }
    key = wanted;
    return profile;
}

enum class Task3Impl { Hartley,
    HartleySeparable,
    CufftLto };

template <typename TIn>
void BM_Task3(benchmark::State& state, Extent e, Task3Impl impl, TaskFrom from)
{
    const size_t count = ElementCount(e.width, e.height, e.depth);
    const int W = static_cast<int>(e.width), H = static_cast<int>(e.height), D = static_cast<int>(e.depth);
    constexpr int kBins = rapidht_bench::kRadialBins;
    const std::vector<TIn> host = MakeIntegerNoise<TIn>(count);
    // The counts are a property of the extent, not of the data: a plan.
    const std::vector<double> counts = PowerReference(host, e).count;
    const double usedStart = DeviceUsedMiB();
    try {
        DeviceBuffer<TIn> in(count);
        cudaMemcpy(in.Data(), host.data(), count * sizeof(TIn), cudaMemcpyHostToDevice);
        std::unique_ptr<HostBuffer<TIn>> staged;
        if (from != TaskFrom::Device) {
            staged = std::make_unique<HostBuffer<TIn>>(count, from == TaskFrom::Pinned);
            std::copy(host.begin(), host.end(), staged->Data());
        }
        std::vector<double> bins(kBins, 0.0), profile(kBins, 0.0);
        std::unique_ptr<HartleyTransform<float>> transform;
        std::unique_ptr<RapiDHT::DeviceVolume<float>> volume;
        std::unique_ptr<DeviceBuffer<double>> deviceBins;
        std::unique_ptr<rapidht_bench::CufftPowerProfile> cufft;
        std::function<void()> compute;
        if (impl == Task3Impl::CufftLto) {
            cufft = std::make_unique<rapidht_bench::CufftPowerProfile>(W, H, D, IntTypeOf<TIn>());
            compute = [&] { cufft->Apply(in.Data(), bins.data()); };
        } else {
            transform = std::make_unique<HartleyTransform<float>>(e.width, e.height, e.depth, Modes::GPUFactored);
            volume = std::make_unique<RapiDHT::DeviceVolume<float>>(count);
            deviceBins = std::make_unique<DeviceBuffer<double>>(kBins);
            const bool separable = impl == Task3Impl::HartleySeparable;
            compute = [&, separable] {
                RapiDHT::internal::DeviceAccess::ForwardFromIntegers(
                    *transform, in.Data(), *volume, RapiDHT::internal::IntegerOffset::Default(), separable);
                rapidht_bench::RadialPowerHartley(DevicePeek::Data(*volume), W, H, D, deviceBins->Data(), bins.data());
            };
        }
        RunTask(
            state, count, usedStart,
            [&] {
                if (from != TaskFrom::Device) {
                    cudaMemcpy(in.Data(), staged->Data(), count * sizeof(TIn), cudaMemcpyHostToDevice);
                }
                compute(); // ends with the bins on the host
                for (int b = 0; b < kBins; ++b) {
                    const double c = counts[static_cast<size_t>(b)];
                    profile[static_cast<size_t>(b)] = c > 0.0 ? bins[static_cast<size_t>(b)] / c : 0.0;
                }
            },
            [&]() -> std::string {
                const PowerProfile& reference = PowerReference(host, e);
                double worst = 0.0;
                for (int b = 0; b < kBins; ++b) {
                    const double want = reference.mean[static_cast<size_t>(b)];
                    if (reference.count[static_cast<size_t>(b)] > 0.0) {
                        worst = std::max(worst, std::fabs(profile[static_cast<size_t>(b)] - want) / want);
                    }
                }
                TaskAccuracy()[state.name()]["relErr"] = worst;
                if (!(worst <= 1e-3)) {
                    return "the profile disagrees with the CPU in double: relative error " + std::to_string(worst);
                }
                return {};
            });
        ReportTaskAccuracy(state);
    } catch (const std::exception& error) {
        cudaGetLastError();
        state.SkipWithError(error.what());
    }
}

/*
 * The rows of stage 5. For each extent and type: T1, then T3 (both check
 * against the same CPU spectrum, kept between rows), then T2 by sigma (its
 * reference is the filter, kept per sigma).
 */
template <typename TIn>
void RegisterTasksFor(const Extent& e, const char* type)
{
    const TaskFrom all[] = { TaskFrom::Device, TaskFrom::Pageable, TaskFrom::Pinned };
    const std::string tail = std::string("/3D/") + Describe(e);
    auto name = [&](const char* family, const std::string& middle, TaskFrom from) {
        return std::string(family) + "/" + type + middle + "/" + TaskFromName(from) + tail;
    };
    const std::pair<const char*, Task1Impl> t1[] = { { "Task1/IntFactored", Task1Impl::IntFactored },
        { "Task1/FloatFactored", Task1Impl::FloatFactored }, { "Task1/cuFFTLto", Task1Impl::CufftLto } };
    for (const auto& [family, impl] : t1) {
        for (const TaskFrom from : all) {
            if (impl == Task1Impl::FloatFactored && from == TaskFrom::Device) {
                continue; // on the device it is FactoredResident/, not a task from integers
            }
            benchmark::RegisterBenchmark(name(family, "", from).c_str(),
                [e, impl = impl, from](benchmark::State& s) { BM_Task1<TIn>(s, e, impl, from); })
                ->Unit(benchmark::kMillisecond);
        }
    }
    const std::pair<const char*, Task3Impl> t3[] = { { "Task3/Hartley", Task3Impl::Hartley },
        { "Task3/HartleySeparable", Task3Impl::HartleySeparable }, { "Task3/cuFFTLto", Task3Impl::CufftLto } };
    for (const auto& [family, impl] : t3) {
        for (const TaskFrom from : all) {
            benchmark::RegisterBenchmark(name(family, "", from).c_str(),
                [e, impl = impl, from](benchmark::State& s) { BM_Task3<TIn>(s, e, impl, from); })
                ->Unit(benchmark::kMillisecond);
        }
    }
    const std::pair<const char*, Task2Impl> t2[] = { { "Task2/Hartley", Task2Impl::Hartley },
        { "Task2/HartleySeparable", Task2Impl::HartleySeparable }, { "Task2/cuFFTLto", Task2Impl::CufftLto },
        { "Task2/Spatial", Task2Impl::Spatial } };
    for (const int sigma : { 1, 2, 4 }) {
        for (const auto& [family, impl] : t2) {
            if (impl == Task2Impl::Spatial && sigma > 2) {
                continue;
            }
            for (const TaskFrom from : all) {
                benchmark::RegisterBenchmark(name(family, "/s" + std::to_string(sigma), from).c_str(),
                    [e, impl = impl, sigma, from](benchmark::State& s) {
                        BM_Task2<TIn>(s, e, impl, static_cast<double>(sigma), from);
                    })
                    ->Unit(benchmark::kMillisecond);
            }
        }
    }
}

// R-5 step 5.1: 256^3, 512^3 and the largest slab of a Korneev column that is
// a power of two, 1024 x 1024 x 512; uint16, and uint8 at 512^3.
const Extent kExtentsTasks[] = { { 256, 256, 256 }, { 512, 512, 512 }, { 1024, 1024, 512 } };

void RegisterTasks()
{
    if (!RapiDHT::IsGpuAvailable()) {
        return;
    }
    for (const Extent& e : kExtentsTasks) {
        RegisterTasksFor<uint16_t>(e, "u16");
        if (e.width == 512) {
            RegisterTasksFor<uint8_t>(e, "u8");
        }
    }
}
#endif

/// R-5 step 1.2: the mean offset, exact and sampled.
void RegisterIntOffset(const Extent* extents, size_t n)
{
#ifdef RAPIDHT_WITH_CUDA
    if (!RapiDHT::IsGpuAvailable()) {
        return;
    }
    const std::pair<const char*, size_t> policies[] = { { "mean", 1 }, { "every64", 64 }, { "every512", 512 } };
    for (size_t i = 0; i < n; ++i) {
        const Extent e = extents[i];
        for (const auto& [name, step] : policies) {
            const std::string tail = std::string("/") + name + "/3D/" + Describe(e);
            benchmark::RegisterBenchmark(("IntOffset/u8" + tail).c_str(),
                [e, step = step](benchmark::State& s) { BM_IntOffset<uint8_t>(s, e, step); })
                ->Unit(benchmark::kMicrosecond);
            benchmark::RegisterBenchmark(("IntOffset/u16" + tail).c_str(),
                [e, step = step](benchmark::State& s) { BM_IntOffset<uint16_t>(s, e, step); })
                ->Unit(benchmark::kMicrosecond);
        }
    }
#else
    (void)extents;
    (void)n;
#endif
}

/// R-5 step 0.6: host to device, the three input types, ordinary and pinned
/// host memory.
void RegisterHostToDevice(const Extent* extents, size_t n)
{
#ifdef RAPIDHT_WITH_CUDA
    if (!RapiDHT::IsGpuAvailable()) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        const Extent e = extents[i];
        const size_t count = ElementCount(e.width, e.height, e.depth);
        for (const bool pinned : { false, true }) {
            const std::string tail = std::string(pinned ? "/pinned/3D/" : "/pageable/3D/") + Describe(e);
            benchmark::RegisterBenchmark(("H2D/u8" + tail).c_str(),
                [count, pinned](benchmark::State& s) { BM_HostToDevice<uint8_t>(s, count, pinned); })
                ->Unit(benchmark::kMillisecond);
            benchmark::RegisterBenchmark(("H2D/u16" + tail).c_str(),
                [count, pinned](benchmark::State& s) { BM_HostToDevice<uint16_t>(s, count, pinned); })
                ->Unit(benchmark::kMillisecond);
            benchmark::RegisterBenchmark(("H2D/f32" + tail).c_str(),
                [count, pinned](benchmark::State& s) { BM_HostToDevice<float>(s, count, pinned); })
                ->Unit(benchmark::kMillisecond);
        }
    }
#else
    (void)extents;
    (void)n;
#endif
}

/// cuFFT is registered for volumes in single precision only: that is the
/// comparison being made, and keeping it narrow keeps the run short.
void RegisterCufft(const Extent* extents, size_t n, const char* rank)
{
#ifdef RAPIDHT_WITH_CUDA
    if (!RapiDHT::IsGpuAvailable()) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        const Extent e = extents[i];
        benchmark::RegisterBenchmark(
            ("cuFFT/f32/" + std::string(rank) + "/" + Describe(e)).c_str(),
            [e](benchmark::State& s) { BM_Cufft(s, e.width, e.height, e.depth); })
            ->Unit(benchmark::kMicrosecond);
        benchmark::RegisterBenchmark(
            ("cuFFT_R2C/f32/" + std::string(rank) + "/" + Describe(e)).c_str(),
            [e](benchmark::State& s) { BM_CufftR2C(s, e.width, e.height, e.depth); })
            ->Unit(benchmark::kMicrosecond);
    }
#else
    (void)extents;
    (void)n;
    (void)rank;
#endif
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

/*
 * Local Minkowski functionals as hno-porous uses them: 4 fields on 3 ball
 * windows (radii N/32, N/16, N/8) of a random 40% solid. This is the B-203
 * target -- 512^3, 4 fields, 3 scales under a second -- measured on the CPU
 * path before deciding whether a GPU one is needed. Memory: 12 output fields,
 * 4 densities and 3 window spectra, about 10 GiB in float at 512^3.
 */
void BM_MinkowskiFields(benchmark::State& state, size_t n)
{
    std::vector<uint8_t> solid(n * n * n);
    uint32_t seed = 12345;
    for (auto& v : solid) {
        seed = seed * 1664525u + 1013904223u; // fixed LCG: reproducible, no <random> state
        v = static_cast<uint8_t>((seed >> 24) < 102); // ~40%
    }
    const double r = static_cast<double>(n) / 16.0;

    std::unique_ptr<RapiDHT::MinkowskiFields<float>> fields;
    std::vector<float> out;
    try {
        fields = std::make_unique<RapiDHT::MinkowskiFields<float>>(
            n, n, n, RapiDHT::WindowShape::Ball, std::vector<double> { r / 2, r, 2 * r });
        out.resize(fields->FieldCount() * fields->Size());
    } catch (const std::exception& error) {
        state.SkipWithError(error.what());
        return;
    }

    for (auto _ : state) {
        fields->Compute(solid.data(), out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n * n * n));
}

void RegisterConvolution(const Extent* extents, size_t n, const char* rank)
{
#ifdef RAPIDHT_WITH_CUDA
    if (!RapiDHT::IsGpuAvailable()) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        const Extent e = extents[i];
        const std::string suffix = std::string("f32/") + rank + "/" + Describe(e);
        for (const bool even : { false, true }) {
            const std::string tail = even ? "Even/" : "/";
            benchmark::RegisterBenchmark(("ConvHartley" + tail + suffix).c_str(),
                [e, even](benchmark::State& s) { BM_ConvHartley(s, e, even); })
                ->Unit(benchmark::kMicrosecond);
            benchmark::RegisterBenchmark(("ConvCufft" + tail + suffix).c_str(),
                [e, even](benchmark::State& s) { BM_ConvCufft(s, e, even); })
                ->Unit(benchmark::kMicrosecond);
        }
    }
#else
    (void)extents;
    (void)n;
    (void)rank;
#endif
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
    RegisterCufft(kExtents3D, CountOf(kExtents3D), "3D");
    RegisterFactored<double>(kExtents3D, CountOf(kExtents3D), "3D", "f64");
    RegisterFactored<float>(kExtents3D, CountOf(kExtents3D), "3D", "f32");
    RegisterCufft(kExtentsR4Only3D, CountOf(kExtentsR4Only3D), "3D");
    RegisterFactored<float>(kExtentsR4Only3D, CountOf(kExtentsR4Only3D), "3D", "f32");
    RegisterIntegerInput(kExtents3D, CountOf(kExtents3D));
    RegisterIntegerInput(kExtentsR5Large3D, CountOf(kExtentsR5Large3D));
#ifdef RAPIDHT_WITH_CUDA
    RegisterIntegerInverse(kExtents3D, CountOf(kExtents3D)); // R-5 step 4.3
    RegisterIntegerInverse(kExtentsR5Large3D, CountOf(kExtentsR5Large3D));
    RegisterTasks(); // R-5 step 5.1
#endif
    RegisterIntOffset(kExtents3D + 2, 1); // 512^3
    RegisterIntOffset(kExtentsR5Large3D + 1, 1); // 1024^3
    RegisterHostToDevice(kExtents3D + 2, 1); // 512^3
    RegisterHostToDevice(kExtentsR5Large3D + 1, 1); // 1024^3
    RegisterConvolution(kExtentsR4Only3D, 1, "3D"); // 64^3
    RegisterConvolution(kExtents3D, CountOf(kExtents3D), "3D");
    // 1024^3 does not fit either way (the volume and the kernel's spectrum
    // alone are 8 GiB); half of it shows the difference in memory: about
    // 4 GiB for Hartley, about 8 for cuFFT.
    RegisterConvolution(kExtentsConvLargest, CountOf(kExtentsConvLargest), "3D");
    RegisterDoubleOnly(kExtents3D, CountOf(kExtents3D), "3D");
    for (const Extent& e : kExtents3D) {
        benchmark::RegisterBenchmark(("MinkowskiFields/f32/3D/" + Describe(e)).c_str(),
            [e](benchmark::State& s) { BM_MinkowskiFields(s, e.width); })
            ->Unit(benchmark::kMillisecond);
    }

    RegisterFor<double>(kExtents2D, CountOf(kExtents2D), "2D", "f64");
    RegisterFor<float>(kExtents2D, CountOf(kExtents2D), "2D", "f32");
    RegisterFactored<float>(kExtents2D, CountOf(kExtents2D), "2D", "f32");
    RegisterCufft(kExtents2D, CountOf(kExtents2D), "2D");
    RegisterFactored<float>(kExtentsR4Only2D, CountOf(kExtentsR4Only2D), "2D", "f32");
    RegisterCufft(kExtentsR4Only2D, CountOf(kExtentsR4Only2D), "2D");
    RegisterConvolution(kExtents2D + 1, CountOf(kExtents2D) - 1, "2D"); // from 512^2
    RegisterConvolution(kExtentsR4Only2D, CountOf(kExtentsR4Only2D), "2D");
    RegisterDoubleOnly(kExtents2D, CountOf(kExtents2D), "2D");
    // R-5 step 2.6: integer input in 2D, 512^2-2048^2.
    RegisterIntegerInput(kExtents2D + 1, CountOf(kExtents2D) - 1, "2D");
    RegisterIntegerInput(kExtentsR4Only2D, CountOf(kExtentsR4Only2D), "2D");

    RegisterFor<double>(kExtents1D, CountOf(kExtents1D), "1D", "f64");
    RegisterDoubleOnly(kExtents1D, CountOf(kExtents1D), "1D");

    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
