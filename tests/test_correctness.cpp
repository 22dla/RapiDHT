/*
 * Project: RapiDHT
 * File: tests/test_correctness.cpp
 * Brief: Checks that the transform computes the discrete Hartley transform,
 *        rather than merely being invertible.
 *
 * The pre-existing smoke tests only apply Forward followed by Inverse and
 * compare against the input. That passes for *any* invertible operation and
 * therefore cannot detect a wrong transform. The tests below pin the actual
 * mathematics down in two independent ways: against a direct O(N^2) evaluation
 * of the definition, and against analytically known signal/spectrum pairs.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "test_support.h"

using namespace RapiDHT;
using namespace rapidht_test;

namespace {

struct Extent {
    size_t width;
    size_t height;
    size_t depth;
};

// Sizes must be powers of two: FDHT1D rejects anything else. These stay small
// because the reference implementation is O(N^2).
const Extent kReferenceExtents[] = {
    { 8, 0, 0 },
    { 64, 0, 0 },
    { 4, 4, 0 },
    { 8, 4, 0 },
    { 16, 8, 0 },
    { 4, 4, 4 },
    { 8, 4, 2 },
    // Longer than one block of lines (32) along the axis the CPU passes group
    // lines by: width for the X and Z passes, height for the Y pass.
    { 64, 8, 4 },
    { 8, 64, 2 },
};

// Larger sizes, used only for properties that do not need the O(N^2) reference.
const Extent kPropertyExtents[] = {
    { 1024, 0, 0 },
    { 64, 32, 0 },
    { 16, 16, 16 },
};

std::string Describe(const Extent& e)
{
    return std::to_string(e.width) + "x" + std::to_string(e.height) + "x" + std::to_string(e.depth);
}

} // namespace

// ---------------------------------------------------------------------------
// Against a direct evaluation of the definition
// ---------------------------------------------------------------------------

TEST(Correctness, ForwardMatchesReference_CPU)
{
    for (const Extent& e : kReferenceExtents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(dims.Total());

        const auto actual = Forward(input, e.width, e.height, e.depth, Modes::CPU);
        const auto expected = ReferenceDht(input, dims);

        ExpectClose(actual, expected);
    }
}

TEST(Correctness, ForwardMatchesReference_GPU)
{
    SKIP_IF_NO_CUDA();
    for (const Extent& e : kReferenceExtents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(dims.Total());

        const auto actual = Forward(input, e.width, e.height, e.depth, Modes::GPU);
        const auto expected = ReferenceDht(input, dims);

        // Looser tolerance: the GPU path accumulates through cuBLAS GEMMs.
        ExpectClose(actual, expected, 1e-9);
    }
}

// R-4: the factored device path, host pointer and resident volume, in double
// and in float, against the definition. The 3D extents include axes longer
// than one block of lines and axes of length 2, where N1/2 is the only
// non-zero column.
TEST(Correctness, FactoredMatchesReference_GPU)
{
    SKIP_IF_NO_GPU();
    for (const Extent& e : kReferenceExtents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(dims.Total());
        const auto expected = ReferenceDht(input, dims);

        ExpectClose(Forward(input, e.width, e.height, e.depth, Modes::GPUFactored), expected, 1e-10);

        HartleyTransform<double> ht(e.width, e.height, e.depth, Modes::GPUFactored);
        DeviceVolume<double> volume(dims.Total());
        volume.Upload(input.data());
        ht.ForwardTransform(volume);
        std::vector<double> resident(dims.Total());
        volume.Download(resident.data());
        ExpectClose(resident, expected, 1e-10);

        std::vector<float> single(input.begin(), input.end());
        HartleyTransform<float> htf(e.width, e.height, e.depth, Modes::GPUFactored);
        htf.ForwardTransform(single.data());
        ExpectClose(std::vector<double>(single.begin(), single.end()), expected, 2e-5);
    }
}

// Larger extents, against the CPU backend, which the tests above pin to the
// definition: 512 is the 32 x 16 split the benchmarks run, 2048 the limit.
TEST(Correctness, FactoredAgreesWithCpuOnLargeExtents_GPU)
{
    SKIP_IF_NO_GPU();
    const Extent extents[] = { { 512, 0, 0 }, { 2048, 0, 0 }, { 256, 128, 0 }, { 64, 32, 128 }, { 128, 128, 128 } };
    for (const Extent& e : extents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(dims.Total());
        ExpectClose(Forward(input, e.width, e.height, e.depth, Modes::GPUFactored),
            Forward(input, e.width, e.height, e.depth, Modes::CPU), 1e-10);
    }
}

// The configuration the benchmarks run: float, resident, axes long enough
// that the register-tiled kernels use every thread of a block and each block
// of the pipelined kernel walks several batches of lines. The tests above
// reach those kernels in float only on extents of one batch to a block, and on
// long axes only in double, so the code path whose timings are published was
// never compared with anything on the device.
//
// A zero-mean pseudo-random volume against the CPU backend in double. Every
// coefficient of its spectrum is of the same order, so the tolerance is tied
// to their RMS: with MakeSignal the DC term is N times larger than the rest
// and a tolerance scaled to it would hide a wrong coefficient. Float rounding
// alone grows with the volume: 1.1e-5 of the RMS at 256^3 and 6.3e-5 at 512^3
// (the same kernels built for the host, float against double). 5e-4 is eight
// times that and still two thousand times below one wrong coefficient.
namespace {

std::vector<double> MakeNoise(size_t count)
{
    std::vector<double> data(count);
    uint32_t seed = 2463534242u;
    for (double& value : data) {
        seed = seed * 1664525u + 1013904223u; // fixed LCG: reproducible
        value = static_cast<double>(seed >> 8) / 8388608.0 - 1.0;
    }
    return data;
}

void ExpectFactoredFloatMatchesCpu(const Extent& e)
{
    const Dims dims = Dims::Of(e.width, e.height, e.depth);
    const auto input = MakeNoise(dims.Total());
    const auto expected = Forward(input, e.width, e.height, e.depth, Modes::CPU);

    const std::vector<float> single(input.begin(), input.end());
    HartleyTransform<float> transform(e.width, e.height, e.depth, Modes::GPUFactored);
    DeviceVolume<float> volume(dims.Total());
    volume.Upload(single.data());
    transform.ForwardTransform(volume);
    std::vector<float> actual(dims.Total());
    volume.Download(actual.data());

    double sumOfSquares = 0.0;
    for (const double value : expected) {
        sumOfSquares += value * value;
    }
    const double rms = std::sqrt(sumOfSquares / static_cast<double>(expected.size()));

    double worst = 0.0;
    size_t worstIndex = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double error = std::fabs(static_cast<double>(actual[i]) - expected[i]);
        if (!(error <= worst)) { // also catches NaN
            worst = error;
            worstIndex = i;
        }
    }
    EXPECT_LE(worst, 5e-4 * rms) << "largest error at index " << worstIndex << ", spectrum RMS " << rms;
}

} // namespace

// One long axis in each position, so that the pass along it runs in each of
// the three geometries (lines contiguous, lines 4 apart in a row, lines a
// plane apart) with several batches to a block; 1024 and 2048 are the other
// splits (32 x 32 and 64 x 32, the latter with 4 lines to a block).
TEST(Correctness, FactoredFloatAgreesWithCpuOnLongAxes_GPU)
{
    SKIP_IF_NO_GPU();
    const Extent extents[] = {
        { 512, 64, 32 },
        { 64, 512, 32 },
        { 32, 64, 512 },
        { 1024, 32, 16 },
        { 16, 32, 1024 },
        { 2048, 16, 16 },
        { 256, 128, 0 },
        { 128, 128, 128 },
    };
    for (const Extent& e : extents) {
        SCOPED_TRACE("extent " + Describe(e));
        ExpectFactoredFloatMatchesCpu(e);
    }
}

// Exactly what docs/benchmarks.md quotes against cuFFT. Kept apart from the
// test above because it needs 512 MiB on the device and about 4 GiB of host
// memory; filter it out with --gtest_filter=-*BenchmarkExtents* where that is
// too much.
TEST(Correctness, FactoredFloatAgreesWithCpuOnBenchmarkExtents_GPU)
{
    SKIP_IF_NO_GPU();
    for (const Extent& e : { Extent { 256, 256, 256 }, Extent { 512, 512, 512 } }) {
        SCOPED_TRACE("extent " + Describe(e));
        ExpectFactoredFloatMatchesCpu(e);
    }
}

TEST(Correctness, FactoredRejectsTooLongOrNonPowerOfTwoExtents_GPU)
{
    SKIP_IF_NO_GPU();
    EXPECT_THROW(HartleyTransform<float>(4096, 0, 0, Modes::GPUFactored), std::invalid_argument);
    EXPECT_THROW(HartleyTransform<float>(64, 12, 0, Modes::GPUFactored), std::invalid_argument);
}

// Sizes above 65536 exercise a bit-reversal path that the smaller cases do
// not: an inline 32-bit reversal used to overflow into indices far outside the
// array there, which segfaulted rather than returning a wrong answer.
TEST(Correctness, RFFT_AgreesWithCpuAboveTheBitReversalThreshold)
{
    for (size_t width : { size_t { 1 } << 16, size_t { 1 } << 17, size_t { 1 } << 18 }) {
        SCOPED_TRACE("width " + std::to_string(width));
        const auto input = MakeSignal(width);

        const auto viaRfft = Forward(input, width, 0, 0, Modes::RFFT);
        const auto viaCpu = Forward(input, width, 0, 0, Modes::CPU);

        ExpectClose(viaRfft, viaCpu, 1e-10);
    }
}

// RFFT is only wired up for the 1D case; 2D/3D fall through to the CPU path.
TEST(Correctness, ForwardMatchesReference_RFFT_1D)
{
    for (size_t width : { size_t { 8 }, size_t { 64 }, size_t { 1024 } }) {
        SCOPED_TRACE("width " + std::to_string(width));
        const auto input = MakeSignal(width);

        const auto actual = Forward(input, width, 0, 0, Modes::RFFT);
        const auto expected = ReferenceDht(input, Dims::Of(width, 0, 0));

        ExpectClose(actual, expected, 1e-11);
    }
}

// ---------------------------------------------------------------------------
// Analytically known pairs
// ---------------------------------------------------------------------------

// A unit impulse at the origin has a flat spectrum: cas(0) == 1 everywhere.
TEST(Correctness, ImpulseTransformsToConstant)
{
    for (const Extent& e : kPropertyExtents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);

        std::vector<double> impulse(dims.Total(), 0.0);
        impulse[0] = 1.0;

        const auto actual = Forward(impulse, e.width, e.height, e.depth, Modes::CPU);
        ExpectClose(actual, std::vector<double>(dims.Total(), 1.0));
    }
}

// The dual statement: a constant signal collapses onto the zero-th bin, whose
// value is N times the constant (the forward transform is unnormalised).
TEST(Correctness, ConstantTransformsToImpulse)
{
    const double kValue = 3.0;
    for (const Extent& e : kPropertyExtents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);

        const auto actual = Forward(std::vector<double>(dims.Total(), kValue), e.width, e.height,
            e.depth, Modes::CPU);

        std::vector<double> expected(dims.Total(), 0.0);
        expected[0] = kValue * static_cast<double>(dims.Total());

        ExpectClose(actual, expected);
    }
}

// B-004: pins where the normalisation sits, which the round trip cannot do --
// Forward then Inverse gives the input back for 1/N in the inverse (ours), 1/N
// in the forward (the dissertation, section 2.1) and 1/sqrt(N) in both alike.
// An impulse tells them apart: forward gives 1 (not 1/N or 1/sqrt(N)) in every
// bin, inverse gives 1/N (not 1 or 1/sqrt(N)). ForwardSeparable follows the
// forward convention, which EvenKernelConvolution relies on.
namespace {

void ExpectNormalisationConvention(Modes mode)
{
    for (const Extent& e : kPropertyExtents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);
        const double n = static_cast<double>(dims.Total());

        std::vector<double> impulse(dims.Total(), 0.0);
        impulse[0] = 1.0;

        HartleyTransform<double> ht(e.width, e.height, e.depth, mode);

        auto forward = impulse;
        ht.ForwardTransform(forward.data());
        ExpectClose(forward, std::vector<double>(dims.Total(), 1.0));

        auto inverse = impulse;
        ht.InverseTransform(inverse.data());
        ExpectClose(inverse, std::vector<double>(dims.Total(), 1.0 / n));

        auto separable = impulse;
        ht.ForwardSeparable(separable.data());
        ExpectClose(separable, std::vector<double>(dims.Total(), 1.0));
    }
}

} // namespace

TEST(Correctness, NormalisationIsOneOverNInInverseOnly_CPU)
{
    ExpectNormalisationConvention(Modes::CPU);
}

TEST(Correctness, NormalisationIsOneOverNInInverseOnly_GPU)
{
    SKIP_IF_NO_GPU();
    ExpectNormalisationConvention(Modes::GPU);
    ExpectNormalisationConvention(Modes::GPUFactored);
}

// The DHT is its own inverse up to the factor N, so applying it twice must
// scale the input by N. This is a much stronger statement than Forward followed
// by Inverse, because Inverse is implemented as Forward plus a 1/N scaling and
// would agree with any involution.
TEST(Correctness, ForwardAppliedTwiceScalesByN)
{
    for (const Extent& e : kPropertyExtents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(dims.Total());

        auto twice = Forward(input, e.width, e.height, e.depth, Modes::CPU);
        twice = Forward(twice, e.width, e.height, e.depth, Modes::CPU);

        std::vector<double> expected(dims.Total());
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = static_cast<double>(dims.Total()) * input[i];
        }

        ExpectClose(twice, expected, 1e-11);
    }
}

// Parseval's theorem for the unnormalised DHT: sum(H^2) == N * sum(x^2).
TEST(Correctness, PreservesEnergy)
{
    for (const Extent& e : kPropertyExtents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(dims.Total());
        const auto spectrum = Forward(input, e.width, e.height, e.depth, Modes::CPU);

        const double inputEnergy = std::inner_product(input.begin(), input.end(), input.begin(), 0.0);
        const double spectrumEnergy
            = std::inner_product(spectrum.begin(), spectrum.end(), spectrum.begin(), 0.0);

        EXPECT_NEAR(spectrumEnergy, static_cast<double>(dims.Total()) * inputEnergy,
            1e-9 * spectrumEnergy);
    }
}

// The transform is linear: H(a*x + b*y) == a*H(x) + b*H(y).
TEST(Correctness, IsLinear)
{
    const double a = 2.5;
    const double b = -1.25;

    for (const Extent& e : kPropertyExtents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);

        const auto x = MakeSignal(dims.Total());
        std::vector<double> y(dims.Total());
        for (size_t i = 0; i < y.size(); ++i) {
            y[i] = std::cos(0.31 * static_cast<double>(i)) - 0.5;
        }

        std::vector<double> combined(dims.Total());
        for (size_t i = 0; i < combined.size(); ++i) {
            combined[i] = a * x[i] + b * y[i];
        }

        const auto lhs = Forward(combined, e.width, e.height, e.depth, Modes::CPU);
        const auto hx = Forward(x, e.width, e.height, e.depth, Modes::CPU);
        const auto hy = Forward(y, e.width, e.height, e.depth, Modes::CPU);

        std::vector<double> rhs(dims.Total());
        for (size_t i = 0; i < rhs.size(); ++i) {
            rhs[i] = a * hx[i] + b * hy[i];
        }

        ExpectClose(lhs, rhs, 1e-11);
    }
}

// ---------------------------------------------------------------------------
// Device-resident path
// ---------------------------------------------------------------------------

// Upload once, transform in place, download once: the result must be identical
// to the host path, which copies on every call.
TEST(Correctness, DeviceResidentMatchesReference_GPU)
{
    SKIP_IF_NO_CUDA();
    for (const Extent& e : kReferenceExtents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(dims.Total());

        HartleyTransform<double> ht(e.width, e.height, e.depth, Modes::GPU);
        DeviceVolume<double> volume(dims.Total());

        volume.Upload(input.data());
        ht.ForwardTransform(volume);

        std::vector<double> actual(dims.Total());
        volume.Download(actual.data());

        ExpectClose(actual, ReferenceDht(input, dims), 1e-9);
    }
}

// B-003: the host-path staging buffer is allocated on first use, so the same
// object must give the right answer whichever path runs first -- resident then
// host, and host then resident. A buffer left unallocated would make the host
// path transfer nothing (Upload truncates to the buffer's size) and hand back
// the input unchanged, which the reference comparison catches.
TEST(Correctness, HostAndResidentPathsShareOneTransform_GPU)
{
    SKIP_IF_NO_CUDA();
    for (const Extent& e : kReferenceExtents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(dims.Total());
        const auto expected = ReferenceDht(input, dims);

        for (const bool residentFirst : { true, false }) {
            SCOPED_TRACE(residentFirst ? "resident first" : "host first");
            HartleyTransform<double> ht(e.width, e.height, e.depth, Modes::GPU);
            DeviceVolume<double> volume(dims.Total());
            std::vector<double> fromResident(dims.Total());
            auto fromHost = input;

            if (residentFirst) {
                volume.Upload(input.data());
                ht.ForwardTransform(volume);
                volume.Download(fromResident.data());
                ht.ForwardTransform(fromHost.data());
            } else {
                ht.ForwardTransform(fromHost.data());
                volume.Upload(input.data());
                ht.ForwardTransform(volume);
                volume.Download(fromResident.data());
            }

            ExpectClose(fromResident, expected, 1e-9);
            ExpectClose(fromHost, expected, 1e-9);
        }
    }
}

// The whole point of the type is that the data survives between calls. Two
// forward transforms without touching the host must scale the input by N, the
// same as two host-path calls would.
TEST(Correctness, DeviceResidentSurvivesBetweenCalls_GPU)
{
    SKIP_IF_NO_CUDA();
    const size_t width = 16, height = 8, depth = 4;
    const Dims dims = Dims::Of(width, height, depth);
    const auto input = MakeSignal(dims.Total());

    HartleyTransform<double> ht(width, height, depth, Modes::GPU);
    DeviceVolume<double> volume(dims.Total());

    volume.Upload(input.data());
    ht.ForwardTransform(volume);
    ht.ForwardTransform(volume);

    std::vector<double> actual(dims.Total());
    volume.Download(actual.data());

    std::vector<double> expected(dims.Total());
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<double>(dims.Total()) * input[i];
    }
    ExpectClose(actual, expected, 1e-9);
}

TEST(Correctness, DeviceResidentInverseUndoesForward_GPU)
{
    SKIP_IF_NO_CUDA();
    const size_t width = 16, height = 8, depth = 4;
    const Dims dims = Dims::Of(width, height, depth);
    const auto input = MakeSignal(dims.Total());

    HartleyTransform<double> ht(width, height, depth, Modes::GPU);
    DeviceVolume<double> volume(dims.Total());

    volume.Upload(input.data());
    ht.ForwardTransform(volume);
    ht.InverseTransform(volume);

    std::vector<double> actual(dims.Total());
    volume.Download(actual.data());

    ExpectClose(actual, input, 1e-9);
}

TEST(Correctness, DeviceResidentRejectsWrongSize_GPU)
{
    SKIP_IF_NO_CUDA();
    HartleyTransform<double> ht(16, 8, 4, Modes::GPU);
    DeviceVolume<double> wrong(16 * 8 * 2);
    EXPECT_THROW(ht.ForwardTransform(wrong), std::invalid_argument);
}

// Needs a device despite testing a rejection path: constructing the volume
// allocates on the card before the transform ever sees it.
TEST(Correctness, DeviceResidentRejectsCpuMode)
{
    SKIP_IF_NO_GPU();
    HartleyTransform<double> ht(16, 8, 4, Modes::CPU);
    DeviceVolume<double> volume(16 * 8 * 4);
    EXPECT_THROW(ht.ForwardTransform(volume), std::invalid_argument);
}

TEST(Correctness, DeviceVolumeIsRejectedWithoutCuda)
{
    if (kCudaEnabled) {
        GTEST_SKIP() << "Built with CUDA, so DeviceVolume is available";
    }
    EXPECT_THROW(DeviceVolume<double>(64), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Integer voxels (R-5 step 6.1)
// ---------------------------------------------------------------------------

namespace {

// Unequal axes in 1D, 2D and 3D, with lines longer than a CPU block (32).
const Extent kIntegerExtents[] = { { 64, 0, 0 }, { 32, 8, 0 }, { 8, 4, 16 }, { 64, 4, 2 } };

const IntegerOffset kIntegerPolicies[] = { IntegerOffset::Default(), IntegerOffset::NoOffset(),
    IntegerOffset::Of(77.0), IntegerOffset::MeanOfAll(), IntegerOffset::MeanOfEvery(3) };

/// Integers over the full range of U, from a fixed formula, with a large
/// mean on top for uint16 (the tomogram case the offset is for).
template <typename U>
std::vector<U> IntegerSignal(size_t count)
{
    std::vector<U> v(count);
    for (size_t i = 0; i < count; ++i) {
        const uint32_t h = static_cast<uint32_t>((i + 1) * 2654435761u);
        v[i] = sizeof(U) == 1 ? static_cast<U>(h >> 24) : static_cast<U>(30000 + (h >> 22));
    }
    return v;
}

template <typename V>
std::vector<double> AsDouble(const std::vector<V>& v)
{
    return std::vector<double>(v.begin(), v.end());
}

/// The integer overloads of one mode against the definition (true
/// transform) and against the double separable path of the CPU (separable),
/// then forward and back: every voxel as it was.
template <typename T, typename U>
void ExpectIntegerHostOverloads(Modes mode, double tolerance)
{
    for (const Extent& e : kIntegerExtents) {
        SCOPED_TRACE("extent " + Describe(e) + (sizeof(U) == 1 ? ", u8" : ", u16"));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);
        const auto input = IntegerSignal<U>(dims.Total());
        const auto truth = ReferenceDht(AsDouble(input), dims);
        std::vector<double> separableTruth = AsDouble(input);
        HartleyTransform<double>(e.width, e.height, e.depth, Modes::CPU).ForwardSeparable(separableTruth.data());

        HartleyTransform<T> ht(e.width, e.height, e.depth, mode);
        std::vector<T> spectrum(dims.Total());
        for (const IntegerOffset& policy : kIntegerPolicies) {
            SCOPED_TRACE("policy " + std::to_string(static_cast<int>(policy.kind)) + ", step "
                         + std::to_string(policy.step) + ", value " + std::to_string(policy.value));
            ht.ForwardTransform(input.data(), spectrum.data(), policy);
            ExpectClose(AsDouble(spectrum), truth, tolerance);
            ht.ForwardSeparable(input.data(), spectrum.data(), policy);
            ExpectClose(AsDouble(spectrum), separableTruth, tolerance);

            ht.ForwardTransform(input.data(), spectrum.data(), policy);
            const std::vector<T> kept = spectrum;
            std::vector<U> back(dims.Total());
            size_t saturated = 99;
            ht.InverseTransform(spectrum.data(), back.data(), policy, &saturated);
            EXPECT_EQ(spectrum, kept) << "the host inverse only reads its input";
            EXPECT_EQ(saturated, 0u);
            const bool nearMean
                = policy.kind == IntegerOffset::Kind::Mean || policy.kind == IntegerOffset::Kind::SampledMean;
            if (std::is_same<T, float>::value && !nearMean) {
                // With an offset far from the mean (none, or 77 under a mean
                // near 30500) float misses uint16 voxels by one on Modes::GPU:
                // what the mean policies are for.
                for (size_t i = 0; i < back.size(); ++i) {
                    ASSERT_LE(std::abs(int(back[i]) - int(input[i])), 1) << "at " << i;
                }
            } else {
                ASSERT_EQ(back, input);
            }
        }

        // Three times the spectrum: the voxels above the range clamp to its top.
        std::vector<double> scaled = truth;
        for (double& v : scaled) {
            v *= 3.0;
        }
        const std::vector<T> tripled(scaled.begin(), scaled.end());
        size_t above = 0;
        std::vector<U> expected(dims.Total());
        for (size_t i = 0; i < expected.size(); ++i) {
            const double v = 3.0 * static_cast<double>(input[i]);
            above += v > (sizeof(U) == 1 ? 255.0 : 65535.0) ? 1 : 0;
            expected[i] = static_cast<U>(std::min(v, sizeof(U) == 1 ? 255.0 : 65535.0));
        }
        std::vector<U> back(dims.Total());
        size_t saturated = 0;
        ht.InverseTransform(tripled.data(), back.data(), IntegerOffset::MeanOfAll(), &saturated);
        EXPECT_GT(above, 0u);
        EXPECT_EQ(saturated, above);
        ASSERT_EQ(back, expected);
    }
}

} // namespace

TEST(Correctness, IntegerOverloadsMatchReference_CPU)
{
    ExpectIntegerHostOverloads<double, uint8_t>(Modes::CPU, 1e-12);
    ExpectIntegerHostOverloads<double, uint16_t>(Modes::CPU, 1e-12);
    ExpectIntegerHostOverloads<float, uint8_t>(Modes::CPU, 1e-5);
    ExpectIntegerHostOverloads<float, uint16_t>(Modes::CPU, 1e-5);
}

TEST(Correctness, IntegerOverloadsMatchReference_GPU)
{
    SKIP_IF_NO_GPU();
    for (const Modes mode : { Modes::GPUFactored, Modes::GPU }) {
        SCOPED_TRACE(mode == Modes::GPU ? "Modes::GPU" : "Modes::GPUFactored");
        ExpectIntegerHostOverloads<double, uint8_t>(mode, 1e-10);
        ExpectIntegerHostOverloads<double, uint16_t>(mode, 1e-10);
        ExpectIntegerHostOverloads<float, uint8_t>(mode, 1e-5);
        ExpectIntegerHostOverloads<float, uint16_t>(mode, 1e-5);
    }
}

// The DeviceVolume overloads run the passes the host-pointer ones run on
// Modes::GPUFactored, so the two agree bit for bit; and forward and back on
// the device restores every voxel. 128^3 takes the fused last pass.
TEST(Correctness, IntegerDeviceVolumesMatchHostOverloads_GPU)
{
    SKIP_IF_NO_GPU();
    const Extent extents[] = { { 128, 128, 128 }, { 8, 4, 16 }, { 256, 32, 0 }, { 512, 0, 0 } };
    for (const Extent& e : extents) {
        SCOPED_TRACE("extent " + Describe(e));
        const Dims dims = Dims::Of(e.width, e.height, e.depth);
        const auto input = IntegerSignal<uint16_t>(dims.Total());
        HartleyTransform<float> ht(e.width, e.height, e.depth, Modes::GPUFactored);
        DeviceVolume<uint16_t> voxels(dims.Total());
        voxels.Upload(input.data());
        DeviceVolume<float> spectrum(dims.Total());

        for (const bool separable : { false, true }) {
            SCOPED_TRACE(separable ? "separable" : "true transform");
            std::vector<float> fromHost(dims.Total());
            std::vector<float> fromDevice(dims.Total());
            if (separable) {
                ht.ForwardSeparable(input.data(), fromHost.data());
                ht.ForwardSeparable(voxels, spectrum);
            } else {
                ht.ForwardTransform(input.data(), fromHost.data());
                ht.ForwardTransform(voxels, spectrum);
            }
            spectrum.Download(fromDevice.data());
            ASSERT_EQ(fromDevice, fromHost);
        }

        ht.ForwardTransform(voxels, spectrum);
        std::vector<float> host(dims.Total());
        spectrum.Download(host.data());
        std::vector<uint16_t> fromHost(dims.Total());
        size_t hostSaturated = 99;
        ht.InverseTransform(host.data(), fromHost.data(), IntegerOffset::MeanOfAll(), &hostSaturated);
        DeviceVolume<uint16_t> restored(dims.Total());
        size_t saturated = 99;
        ht.InverseTransform(spectrum, restored, IntegerOffset::MeanOfAll(), &saturated);
        std::vector<uint16_t> back(dims.Total());
        restored.Download(back.data());
        EXPECT_EQ(saturated, 0u);
        EXPECT_EQ(hostSaturated, 0u);
        ASSERT_EQ(back, input);
        ASSERT_EQ(fromHost, input);
    }
}

TEST(Correctness, IntegerDeviceVolumesRejectOtherModesAndSizes_GPU)
{
    SKIP_IF_NO_GPU();
    DeviceVolume<uint16_t> voxels(16 * 8 * 4);
    DeviceVolume<float> spectrum(16 * 8 * 4);
    HartleyTransform<float> dense(16, 8, 4, Modes::GPU);
    EXPECT_THROW(dense.ForwardTransform(voxels, spectrum), std::invalid_argument);
    EXPECT_THROW(dense.InverseTransform(spectrum, voxels), std::invalid_argument);
    HartleyTransform<float> cpu(16, 8, 4, Modes::CPU);
    EXPECT_THROW(cpu.ForwardSeparable(voxels, spectrum), std::invalid_argument);
    HartleyTransform<float> other(16, 8, 2, Modes::GPUFactored);
    EXPECT_THROW(other.ForwardTransform(voxels, spectrum), std::invalid_argument);
    DeviceVolume<uint8_t> shorter(16 * 8 * 2);
    HartleyTransform<float> factored(16, 8, 4, Modes::GPUFactored);
    EXPECT_THROW(factored.InverseTransform(spectrum, shorter), std::invalid_argument);
}

TEST(Correctness, IntegerOverloadsRejectNullData)
{
    HartleyTransform<float> ht(16, 8, 4, Modes::CPU);
    std::vector<float> spectrum(16 * 8 * 4);
    std::vector<uint16_t> voxels(16 * 8 * 4);
    EXPECT_THROW(ht.ForwardTransform(static_cast<const uint16_t*>(nullptr), spectrum.data()), std::invalid_argument);
    EXPECT_THROW(ht.ForwardSeparable(voxels.data(), static_cast<float*>(nullptr)), std::invalid_argument);
    EXPECT_THROW(ht.InverseTransform(spectrum.data(), static_cast<uint16_t*>(nullptr)), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Argument validation
// ---------------------------------------------------------------------------

TEST(Correctness, RejectsNonPowerOfTwoLength)
{
    HartleyTransform<double> ht(12, 0, 0, Modes::CPU);
    std::vector<double> data(12, 1.0);
    EXPECT_THROW(ht.ForwardTransform(data.data()), std::invalid_argument);
}

// B-007: in 2D and 3D the 1D passes run inside an OpenMP parallel region, and
// an exception escaping one aborts the process through std::terminate. The
// extents must therefore be checked before any pass starts, which also leaves
// the caller's data untouched. Each axis is tried in turn, because the passes
// run in a fixed order and only the first one reached would have thrown.
TEST(Correctness, RejectsNonPowerOfTwoExtentIn2DAnd3DWithoutTouchingData)
{
    struct Case {
        size_t width, height, depth;
    };
    const Case cases[] = {
        // 2D: each axis
        { 12, 8, 0 },
        { 8, 12, 0 },
        // 3D: each axis
        { 12, 8, 8 },
        { 8, 12, 8 },
        { 8, 8, 12 },
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(std::to_string(c.width) + "x" + std::to_string(c.height) + "x" + std::to_string(c.depth));
        const size_t total = c.width * c.height * (c.depth == 0 ? 1 : c.depth);
        const auto input = MakeSignal(total);

        HartleyTransform<double> ht(c.width, c.height, c.depth, Modes::CPU);

        auto data = input;
        EXPECT_THROW(ht.ForwardTransform(data.data()), std::invalid_argument);
        EXPECT_EQ(data, input);

        data = input;
        EXPECT_THROW(ht.InverseTransform(data.data()), std::invalid_argument);
        EXPECT_EQ(data, input);

        data = input;
        EXPECT_THROW(ht.ForwardSeparable(data.data()), std::invalid_argument);
        EXPECT_EQ(data, input);
    }
}

// Powers of two must be accepted right up to the boundary cases, where the
// previous ceil(log2)/floor(log2) test was at the mercy of rounding.
TEST(Correctness, AcceptsEveryPowerOfTwo)
{
    for (size_t width = 1; width <= (size_t { 1 } << 20); width <<= 1) {
        SCOPED_TRACE("width " + std::to_string(width));
        std::vector<double> data(width, 1.0);
        HartleyTransform<double> ht(width, 0, 0, Modes::CPU);
        EXPECT_NO_THROW(ht.ForwardTransform(data.data()));
    }
}

// The twiddle table is built once in the constructor and reused by every call,
// so a stale or partially built table would only show up on the second and
// later transforms. Guards against exactly that.
TEST(Correctness, RepeatedCallsAgreeWithReference)
{
    const size_t width = 256;
    const auto input = MakeSignal(width);
    const auto expected = ReferenceDht(input, Dims::Of(width, 0, 0));

    HartleyTransform<double> ht(width, 0, 0, Modes::CPU);
    for (int call = 0; call < 5; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        std::vector<double> data = input;
        ht.ForwardTransform(data.data());
        ExpectClose(data, expected);
    }
}

TEST(Correctness, RejectsNullData)
{
    HartleyTransform<double> ht(8, 0, 0, Modes::CPU);
    EXPECT_THROW(ht.ForwardTransform(nullptr), std::invalid_argument);
}

// RFFT only has a 1D implementation. It used to accept 2D and 3D and quietly
// run the CPU path instead, so the caller got a different backend than asked
// for with no indication.
TEST(Correctness, RejectsRfftBeyondOneDimension)
{
    EXPECT_NO_THROW(HartleyTransform<double>(64, 0, 0, Modes::RFFT));
    EXPECT_THROW(HartleyTransform<double>(64, 64, 0, Modes::RFFT), std::invalid_argument);
    EXPECT_THROW(HartleyTransform<double>(64, 64, 64, Modes::RFFT), std::invalid_argument);
}

TEST(Correctness, RejectsZeroWidth)
{
    EXPECT_THROW(HartleyTransform<double>(0, 0, 0, Modes::CPU), std::invalid_argument);
}

TEST(Correctness, GpuModeIsRejectedWhenNotCompiledIn)
{
    if (kCudaEnabled) {
        GTEST_SKIP() << "Built with CUDA support, so Modes::GPU is available";
    }
    EXPECT_THROW(HartleyTransform<double>(8, 0, 0, Modes::GPU), std::runtime_error);
}
