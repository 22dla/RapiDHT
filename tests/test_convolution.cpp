/*
 * Project: RapiDHT
 * File: tests/test_convolution.cpp
 * Brief: The separable transform and convolution with even kernels, checked
 *        against direct evaluation of their definitions.
 */

#include <rapidht/convolution.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
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

// Small, because the references are O(N^2). Non-square on purpose: a swapped
// axis cannot hide behind equal extents.
const Extent kExtents[] = {
    { 16, 0, 0 },
    { 8, 4, 0 },
    { 4, 16, 0 },
    { 4, 4, 4 },
    { 8, 4, 2 },
    { 2, 8, 16 },
};

std::string Describe(const Extent& e)
{
    return std::to_string(e.width) + "x" + std::to_string(e.height) + "x" + std::to_string(e.depth);
}

size_t Mirror(size_t i, size_t n) { return i == 0 ? 0 : n - i; }

/// Separable DHT by definition: a direct 1D DHT along each axis in turn.
std::vector<double> ReferenceSeparableDht(std::vector<double> data, const Dims& d)
{
    const double kTwoPi = 2.0 * std::acos(-1.0);
    const size_t extents[3] = { d.width, d.height, d.depth };
    for (int axis = 0; axis < 3; ++axis) {
        const size_t n = extents[axis];
        if (n == 1) {
            continue;
        }
        std::vector<double> out(data.size(), 0.0);
        for (size_t k = 0; k < d.depth; ++k) {
            for (size_t j = 0; j < d.height; ++j) {
                for (size_t i = 0; i < d.width; ++i) {
                    const size_t idx[3] = { i, j, k };
                    double sum = 0.0;
                    for (size_t m = 0; m < n; ++m) {
                        size_t src[3] = { i, j, k };
                        src[axis] = m;
                        const double phase = kTwoPi * static_cast<double>(idx[axis] * m) / static_cast<double>(n);
                        sum += data[d.Index(src[0], src[1], src[2])] * Cas(phase);
                    }
                    out[d.Index(i, j, k)] = sum;
                }
            }
        }
        data.swap(out);
    }
    return data;
}

/// Direct circular convolution: (f*g)(x) = sum_y f(y) g(x - y).
std::vector<double> ReferenceCircularConvolution(const std::vector<double>& f, const std::vector<double>& g,
    const Dims& d)
{
    std::vector<double> out(f.size(), 0.0);
    for (size_t k = 0; k < d.depth; ++k) {
        for (size_t j = 0; j < d.height; ++j) {
            for (size_t i = 0; i < d.width; ++i) {
                double sum = 0.0;
                for (size_t c = 0; c < d.depth; ++c) {
                    for (size_t b = 0; b < d.height; ++b) {
                        for (size_t a = 0; a < d.width; ++a) {
                            const size_t gi = (i + d.width - a) % d.width;
                            const size_t gj = (j + d.height - b) % d.height;
                            const size_t gk = (k + d.depth - c) % d.depth;
                            sum += f[d.Index(a, b, c)] * g[d.Index(gi, gj, gk)];
                        }
                    }
                }
                out[d.Index(i, j, k)] = sum;
            }
        }
    }
    return out;
}

/// An anisotropic Gaussian in the wrapped layout: even along every axis.
std::vector<double> MakeEvenKernel(const Dims& d)
{
    std::vector<double> g(d.Total());
    for (size_t k = 0; k < d.depth; ++k) {
        for (size_t j = 0; j < d.height; ++j) {
            for (size_t i = 0; i < d.width; ++i) {
                const double x = static_cast<double>(std::min(i, d.width - i));
                const double y = static_cast<double>(std::min(j, d.height - j));
                const double z = static_cast<double>(std::min(k, d.depth - k));
                g[d.Index(i, j, k)] = std::exp(-(x * x / 3.0 + y * y / 2.0 + z * z / 5.0));
            }
        }
    }
    return g;
}

/// A kernel that is even but not separable and not smooth: the average of a
/// fixed signal over all eight axis flips. Guards against a test that only
/// passes because the Gaussian happens to factorise.
std::vector<double> MakeRoughEvenKernel(const Dims& d)
{
    const std::vector<double> s = MakeSignal(d.Total());
    std::vector<double> g(d.Total(), 0.0);
    for (size_t k = 0; k < d.depth; ++k) {
        for (size_t j = 0; j < d.height; ++j) {
            for (size_t i = 0; i < d.width; ++i) {
                double sum = 0.0;
                for (int flip = 0; flip < 8; ++flip) {
                    const size_t a = (flip & 1) ? Mirror(i, d.width) : i;
                    const size_t b = (flip & 2) ? Mirror(j, d.height) : j;
                    const size_t c = (flip & 4) ? Mirror(k, d.depth) : k;
                    sum += s[d.Index(a, b, c)];
                }
                g[d.Index(i, j, k)] = sum / 8.0;
            }
        }
    }
    return g;
}

} // namespace

// ---------------------------------------------------------------------------
// The separable transform
// ---------------------------------------------------------------------------

TEST(Separable, MatchesDefinition)
{
    for (const auto& e : kExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(d.Total());

        std::vector<double> actual = input;
        HartleyTransform<double> ht(e.width, e.height, e.depth, Modes::CPU);
        ht.ForwardSeparable(actual.data());

        ExpectClose(actual, ReferenceSeparableDht(input, d), 1e-12);
    }
}

TEST(Separable, AppliedTwiceScalesByN)
{
    for (const auto& e : kExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(d.Total());

        std::vector<double> data = input;
        HartleyTransform<double> ht(e.width, e.height, e.depth, Modes::CPU);
        ht.ForwardSeparable(data.data());
        ht.ForwardSeparable(data.data());

        std::vector<double> expected = input;
        for (auto& v : expected) {
            v *= static_cast<double>(d.Total());
        }
        ExpectClose(data, expected, 1e-12);
    }
}

TEST(Separable, CoincidesWithTrueTransformIn1D)
{
    const auto input = MakeSignal(64);
    std::vector<double> separable = input;
    HartleyTransform<double> ht(64, 0, 0, Modes::CPU);
    ht.ForwardSeparable(separable.data());
    ExpectClose(separable, ReferenceDht(input, Dims::Of(64, 0, 0)), 1e-12);
}

TEST(Separable, DiffersFromTrueTransformIn3D)
{
    // A guard on the guard: if these coincided, the tests above could not tell
    // a missing Bracewell pass from a present one.
    const Dims d = Dims::Of(4, 4, 4);
    const auto input = MakeSignal(d.Total());
    std::vector<double> separable = input;
    HartleyTransform<double> ht(4, 4, 4, Modes::CPU);
    ht.ForwardSeparable(separable.data());
    const auto truth = ReferenceDht(input, d);

    double worst = 0.0;
    for (size_t n = 0; n < truth.size(); ++n) {
        worst = std::max(worst, std::fabs(truth[n] - separable[n]));
    }
    EXPECT_GT(worst, 1e-6);
}

// ---------------------------------------------------------------------------
// Convolution with even kernels
// ---------------------------------------------------------------------------

TEST(EvenConvolution, MatchesDirectConvolutionGaussian)
{
    for (const auto& e : kExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto f = MakeSignal(d.Total());
        const auto g = MakeEvenKernel(d);

        std::vector<double> actual = f;
        EvenKernelConvolution<double> conv(e.width, e.height, e.depth);
        conv.SetKernel(g.data());
        conv.Apply(actual.data());

        ExpectClose(actual, ReferenceCircularConvolution(f, g, d), 1e-12);
    }
}

TEST(EvenConvolution, MatchesDirectConvolutionRoughKernel)
{
    for (const auto& e : kExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto f = MakeSignal(d.Total());
        const auto g = MakeRoughEvenKernel(d);

        std::vector<double> actual = f;
        EvenKernelConvolution<double> conv(e.width, e.height, e.depth);
        conv.SetKernel(g.data());
        conv.Apply(actual.data());

        ExpectClose(actual, ReferenceCircularConvolution(f, g, d), 1e-12);
    }
}

TEST(EvenConvolution, FloatIsAccurate)
{
    const Extent e { 8, 4, 2 };
    const Dims d = Dims::Of(e.width, e.height, e.depth);
    const auto f = MakeSignal(d.Total());
    const auto g = MakeEvenKernel(d);
    const auto expected = ReferenceCircularConvolution(f, g, d);

    std::vector<float> data(f.begin(), f.end());
    std::vector<float> kernel(g.begin(), g.end());
    EvenKernelConvolution<float> conv(e.width, e.height, e.depth);
    conv.SetKernel(kernel.data());
    conv.Apply(data.data());

    ExpectClose(std::vector<double>(data.begin(), data.end()), expected, 1e-5);
}

TEST(EvenConvolution, KernelReusedAcrossCalls)
{
    const Extent e { 4, 4, 4 };
    const Dims d = Dims::Of(e.width, e.height, e.depth);
    const auto g = MakeEvenKernel(d);
    EvenKernelConvolution<double> conv(e.width, e.height, e.depth);
    conv.SetKernel(g.data());

    for (int round = 0; round < 3; ++round) {
        std::vector<double> f = MakeSignal(d.Total());
        for (auto& v : f) {
            v += round;
        }
        const auto expected = ReferenceCircularConvolution(f, g, d);
        conv.Apply(f.data());
        ExpectClose(f, expected, 1e-12);
    }
}

TEST(EvenConvolution, DeltaKernelIsIdentity)
{
    const Dims d = Dims::Of(16, 8, 4);
    std::vector<double> delta(d.Total(), 0.0);
    delta[0] = 1.0;
    const auto f = MakeSignal(d.Total());

    std::vector<double> actual = f;
    EvenKernelConvolution<double> conv(16, 8, 4);
    conv.SetKernel(delta.data());
    conv.Apply(actual.data());
    ExpectClose(actual, f, 1e-12);
}

TEST(EvenConvolution, RejectsKernelThatIsNotEven)
{
    const Dims d = Dims::Of(8, 4, 2);
    std::vector<double> shifted(d.Total(), 0.0);
    shifted[d.Index(1, 0, 0)] = 1.0; // a delta off the origin: odd along X
    EXPECT_FALSE(IsEvenPerAxis(shifted.data(), 8, 4, 2));

    EvenKernelConvolution<double> conv(8, 4, 2);
    EXPECT_THROW(conv.SetKernel(shifted.data()), std::invalid_argument);
}

TEST(EvenConvolution, ApplyWithoutKernelThrows)
{
    std::vector<double> data(16, 1.0);
    EvenKernelConvolution<double> conv(16, 0, 0);
    EXPECT_THROW(conv.Apply(data.data()), std::logic_error);
}

TEST(EvenConvolution, RejectsGpuModeForNow)
{
    EXPECT_THROW(EvenKernelConvolution<double>(8, 8, 8, Modes::GPU), std::exception);
}
