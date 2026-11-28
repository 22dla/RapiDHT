/*
 * Project: RapiDHT
 * File: tests/test_filters.cpp
 * Brief: Kernels in physical units (B-104): Gaussian, ball and Butterworth on
 *        anisotropic voxels, and the wrapped layout, against their formulas.
 */

#include <rapidht/convolution.h>
#include <rapidht/filters.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "test_support.h"

using namespace RapiDHT;
using namespace rapidht_test;

namespace {

const double kPi = std::acos(-1.0);

/// The physical offset of element `i` of a compact extent `n` (centre n/2).
double Offset(size_t i, size_t n, double d) { return (static_cast<double>(i) - static_cast<double>(n / 2)) * d; }

size_t AtLeastOne(size_t n) { return n == 0 ? size_t { 1 } : n; }

template <typename T>
double Sum(const std::vector<T>& v)
{
    double s = 0.0;
    for (T x : v) {
        s += static_cast<double>(x);
    }
    return s;
}

/// sum g(x) x_a^2 over the kernel, x_a the physical offset along axis a.
template <typename T>
double SecondMoment(const CompactKernel<T>& k, const Spacing& s, int axis)
{
    const size_t W = k.width, H = AtLeastOne(k.height), D = AtLeastOne(k.depth);
    double m = 0.0;
    for (size_t c = 0; c < D; ++c) {
        for (size_t b = 0; b < H; ++b) {
            for (size_t a = 0; a < W; ++a) {
                const double x[3] = { Offset(a, W, s.x), Offset(b, H, s.y), Offset(c, D, s.z) };
                m += static_cast<double>(k.values[a + W * (b + H * c)]) * x[axis] * x[axis];
            }
        }
    }
    return m;
}

} // namespace

TEST(GaussianKernel, ReachesTruncateSigmasAlongEachAxisInPhysicalUnits)
{
    const Spacing s { 1.0, 0.5, 2.5 };
    const auto k = GaussianKernel<double>(3, 2.0, s); // 4 sigma = 8 units
    EXPECT_EQ(k.width, 17u); // 8 / 1
    EXPECT_EQ(k.height, 33u); // 8 / 0.5
    EXPECT_EQ(k.depth, 9u); // ceil(8 / 2.5) = 4
    ASSERT_EQ(k.values.size(), 17u * 33u * 9u);
    EXPECT_NEAR(Sum(k.values), 1.0, 1e-14);

    // Every value is the formula, up to the one normalising constant.
    const double centre = k.values[8 + 17 * (16 + 33 * 4)];
    for (size_t c = 0; c < 9; ++c) {
        for (size_t b = 0; b < 33; ++b) {
            for (size_t a = 0; a < 17; ++a) {
                const double x = Offset(a, 17, s.x), y = Offset(b, 33, s.y), z = Offset(c, 9, s.z);
                const double expected = centre * std::exp(-(x * x + y * y + z * z) / 8.0);
                ASSERT_NEAR(k.values[a + 17 * (b + 33 * c)], expected, 1e-15) << a << " " << b << " " << c;
            }
        }
    }

    const auto flat = GaussianKernel<float>(2, 1.5, { 0.5, 1.0, 99.0 }, 3.0); // z ignored in 2D
    EXPECT_EQ(flat.width, 19u);
    EXPECT_EQ(flat.height, 11u);
    EXPECT_EQ(flat.depth, 0u);
    EXPECT_NEAR(Sum(flat.values), 1.0, 1e-6);
    const auto line = GaussianKernel<double>(1, 1.0);
    EXPECT_EQ(line.width, 9u);
    EXPECT_EQ(line.height, 0u);
    EXPECT_EQ(line.values.size(), 9u);
}

TEST(GaussianKernel, VarianceIsSigmaSquaredAlongEveryAxis)
{
    // A sampled Gaussian keeps its variance to exp(-2 pi^2 (sigma/d)^2): nil
    // for sigma/d >= 1.5. Truncating at 8 sigma removes the tail's share.
    const Spacing s { 1.0, 0.5, 2.0 };
    const double sigma = 3.0;
    const auto k = GaussianKernel<double>(3, sigma, s, 8.0);
    for (int axis = 0; axis < 3; ++axis) {
        SCOPED_TRACE(axis);
        EXPECT_NEAR(SecondMoment(k, s, axis), sigma * sigma, 1e-9 * sigma * sigma);
    }
}

TEST(BallKernel, HoldsTheVoxelsWhoseCentreIsInside)
{
    const Spacing s { 1.0, 1.5, 2.0 };
    const double radius = 4.0;
    const auto k = BallKernel<double>(3, radius, s, false);
    EXPECT_EQ(k.width, 9u); // floor(4 / 1) = 4
    EXPECT_EQ(k.height, 5u); // floor(4 / 1.5) = 2
    EXPECT_EQ(k.depth, 5u); // floor(4 / 2) = 2, and (0, 0, 4) is on the sphere: inside
    for (size_t c = 0; c < k.depth; ++c) {
        for (size_t b = 0; b < k.height; ++b) {
            for (size_t a = 0; a < k.width; ++a) {
                const double x = Offset(a, k.width, s.x), y = Offset(b, k.height, s.y), z = Offset(c, k.depth, s.z);
                const double expected = x * x + y * y + z * z <= radius * radius + 1e-9 ? 1.0 : 0.0;
                ASSERT_EQ(k.values[a + k.width * (b + k.height * c)], expected) << a << " " << b << " " << c;
            }
        }
    }
    EXPECT_EQ(k.values[4 + 9 * (2 + 5 * 4)], 1.0); // (0, 0, 4): exactly on the sphere

    // The physical volume: the count times the voxel volume is the ball's.
    const auto big = BallKernel<double>(3, 12.0, { 1.0, 1.0, 2.0 }, false);
    EXPECT_NEAR(Sum(big.values) * 2.0 / (4.0 / 3.0 * kPi * 12.0 * 12.0 * 12.0), 1.0, 0.02);

    const auto averaging = BallKernel<float>(2, 2.0);
    EXPECT_EQ(averaging.width, 5u);
    EXPECT_EQ(averaging.depth, 0u);
    EXPECT_NEAR(Sum(averaging.values), 1.0, 1e-6);
    const auto single = BallKernel<double>(3, 0.5, { 1.0, 1.0, 1.0 });
    EXPECT_EQ(single.values.size(), 1u);
    EXPECT_EQ(single.values[0], 1.0);
}

TEST(ButterworthKernel, HasTheTransferFunctionAsItsSpectrum)
{
    struct Case {
        size_t w, h, d;
        Spacing s;
        double cutoff;
        int order;
    };
    const Case cases[] = {
        { 16, 8, 4, { 1.0, 0.5, 2.0 }, 0.2, 2 },
        { 8, 16, 0, { 0.7, 1.0, 1.0 }, 0.3, 1 },
        { 32, 0, 0, { 2.0, 1.0, 1.0 }, 0.05, 4 },
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(std::to_string(c.w) + "x" + std::to_string(c.h) + "x" + std::to_string(c.d));
        const Dims d = Dims::Of(c.w, c.h, c.d);
        const auto g = ButterworthKernel<double>(c.w, c.h, c.d, c.cutoff, c.order, c.s);
        ASSERT_EQ(g.size(), d.Total());
        EXPECT_TRUE(IsEvenPerAxis(g.data(), c.w, c.h, c.d, 1e-12));
        EXPECT_NEAR(Sum(g), 1.0, 1e-12); // B(0) = 1

        // The definition of the spectrum (true DHT, O(N^2)) against B itself.
        const auto spectrum = ReferenceDht(g, d);
        auto f = [](size_t i, size_t n, double sp) {
            const double m = i < (n + 1) / 2 ? static_cast<double>(i) : static_cast<double>(i) - static_cast<double>(n);
            return n == 1 ? 0.0 : m / (static_cast<double>(n) * sp);
        };
        for (size_t k = 0; k < d.depth; ++k) {
            for (size_t j = 0; j < d.height; ++j) {
                for (size_t i = 0; i < d.width; ++i) {
                    const double fx = f(i, d.width, c.s.x), fy = f(j, d.height, c.s.y), fz = f(k, d.depth, c.s.z);
                    const double r2 = (fx * fx + fy * fy + fz * fz) / (c.cutoff * c.cutoff);
                    const double expected = 1.0 / (1.0 + std::pow(r2, c.order));
                    ASSERT_NEAR(spectrum[d.Index(i, j, k)], expected, 1e-12) << i << " " << j << " " << k;
                }
            }
        }

        const auto single = ButterworthKernel<float>(c.w, c.h, c.d, c.cutoff, c.order, c.s);
        ExpectClose(std::vector<double>(single.begin(), single.end()), g, 1e-6);
    }

    // Half power at the cutoff: bin 4 of 64 at spacing 0.5 is 4 / 32 = 0.125 cycles per unit.
    const auto g = ButterworthKernel<double>(64, 0, 0, 0.125, 3, { 0.5, 1.0, 1.0 });
    EXPECT_NEAR(ReferenceDht(g, Dims::Of(64, 0, 0))[4], 0.5, 1e-12);
}

TEST(WrapKernel, PlacesTheCentreAtZeroAndSumsOverlappingTaps)
{
    // 7 taps into a period of 4: offsets -3..3 land on 1, 2, 3, 0, 1, 2, 3.
    CompactKernel<double> line { { 1, 2, 4, 8, 16, 32, 64 }, 7, 0, 0 };
    const auto w = WrapKernel(line, 4, 0, 0);
    ASSERT_EQ(w.size(), 4u);
    EXPECT_EQ(w[0], 8.0);
    EXPECT_EQ(w[1], 1.0 + 16.0);
    EXPECT_EQ(w[2], 2.0 + 32.0);
    EXPECT_EQ(w[3], 4.0 + 64.0);

    // A 3D anisotropic Gaussian: symmetric, so wrapped it is even, and each tap
    // sits at its offset modulo the extent.
    const auto g = GaussianKernel<double>(3, 1.0, { 1.0, 2.0, 0.5 }, 2.0); // 5 x 3 x 9
    const auto wrapped = WrapKernel(g, 16, 8, 32);
    EXPECT_TRUE(IsEvenPerAxis(wrapped.data(), 16, 8, 32, 0.0));
    EXPECT_NEAR(Sum(wrapped), 1.0, 1e-14);
    const Dims d = Dims::Of(16, 8, 32);
    EXPECT_EQ(wrapped[d.Index(0, 0, 0)], g.values[2 + 5 * (1 + 3 * 4)]);
    EXPECT_EQ(wrapped[d.Index(14, 1, 28)], g.values[0 + 5 * (2 + 3 * 0)]); // offsets (-2, 1, -4)

    EXPECT_THROW(WrapKernel(g, 16, 8, 0), std::invalid_argument); // 3D kernel, 2D volume
    CompactKernel<double> even { { 1, 1 }, 2, 0, 0 };
    EXPECT_THROW(WrapKernel(even, 4, 0, 0), std::invalid_argument);
}

TEST(Filters, AnisotropicGaussianSmoothsLikeTheLinearConvolutionDefinition)
{
    // The two halves together: a Gaussian on a grid with thick slices, through
    // LinearConvolution, against the sum written out with physical offsets.
    const Spacing s { 1.0, 1.0, 2.0 };
    const auto g = GaussianKernel<double>(3, 1.5, s, 3.0); // 11 x 11 x 7: ceil(4.5 / 2) = 3 along z
    const size_t W = 9, H = 7, D = 6;
    const auto f = MakeSignal(W * H * D);
    std::vector<double> out(f.size());
    LinearConvolution<double> conv(W, H, D, g.width, g.height, g.depth, Edges::Normalized);
    conv.SetKernel(g.values.data());
    conv.Apply(f.data(), out.data());

    const long long rx = 5, ry = 5, rz = 3;
    ASSERT_EQ(g.depth, 7u);
    for (long long z = 0; z < static_cast<long long>(D); ++z) {
        for (long long y = 0; y < static_cast<long long>(H); ++y) {
            for (long long x = 0; x < static_cast<long long>(W); ++x) {
                double num = 0.0, den = 0.0;
                for (long long tz = -rz; tz <= rz; ++tz) {
                    for (long long ty = -ry; ty <= ry; ++ty) {
                        for (long long tx = -rx; tx <= rx; ++tx) {
                            const long long sx = x - tx, sy = y - ty, sz = z - tz;
                            if (sx < 0 || sy < 0 || sz < 0 || sx >= static_cast<long long>(W)
                                || sy >= static_cast<long long>(H) || sz >= static_cast<long long>(D)) {
                                continue;
                            }
                            const double px = static_cast<double>(tx) * s.x, py = static_cast<double>(ty) * s.y,
                                         pz = static_cast<double>(tz) * s.z;
                            const double w = std::exp(-(px * px + py * py + pz * pz) / (2.0 * 1.5 * 1.5));
                            num += w * f[static_cast<size_t>(sx + static_cast<long long>(W) * (sy + static_cast<long long>(H) * sz))];
                            den += w;
                        }
                    }
                }
                const size_t n = static_cast<size_t>(x + static_cast<long long>(W) * (y + static_cast<long long>(H) * z));
                ASSERT_NEAR(out[n], num / den, 1e-12) << x << " " << y << " " << z;
            }
        }
    }
}

TEST(Filters, RejectBadArguments)
{
    EXPECT_THROW(GaussianKernel<double>(0, 1.0), std::invalid_argument);
    EXPECT_THROW(GaussianKernel<double>(4, 1.0), std::invalid_argument);
    EXPECT_THROW(GaussianKernel<double>(3, 0.0), std::invalid_argument);
    EXPECT_THROW(GaussianKernel<double>(3, 1.0, {}, -1.0), std::invalid_argument);
    EXPECT_THROW(GaussianKernel<double>(3, 1.0, { 1.0, 0.0, 1.0 }), std::invalid_argument);
    EXPECT_NO_THROW(GaussianKernel<double>(1, 1.0, { 1.0, 0.0, -1.0 })); // y, z unused in 1D
    EXPECT_THROW(BallKernel<double>(3, -1.0), std::invalid_argument);
    EXPECT_THROW(ButterworthKernel<double>(12, 0, 0, 0.1, 1), std::invalid_argument);
    EXPECT_THROW(ButterworthKernel<double>(16, 0, 4, 0.1, 1), std::invalid_argument);
    EXPECT_THROW(ButterworthKernel<double>(16, 0, 0, 0.0, 1), std::invalid_argument);
    EXPECT_THROW(ButterworthKernel<double>(16, 0, 0, 0.1, 0), std::invalid_argument);
}
