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

#include "test_support.h"

#include <gtest/gtest.h>
#include <numeric>
#include <string>
#include <vector>

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

        const double input_energy = std::inner_product(input.begin(), input.end(), input.begin(), 0.0);
        const double spectrum_energy
            = std::inner_product(spectrum.begin(), spectrum.end(), spectrum.begin(), 0.0);

        EXPECT_NEAR(spectrum_energy, static_cast<double>(dims.Total()) * input_energy,
            1e-9 * spectrum_energy);
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
// Argument validation
// ---------------------------------------------------------------------------

TEST(Correctness, RejectsNonPowerOfTwoLength)
{
    HartleyTransform<double> ht(12, 0, 0, Modes::CPU);
    std::vector<double> data(12, 1.0);
    EXPECT_THROW(ht.ForwardTransform(data.data()), std::invalid_argument);
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
