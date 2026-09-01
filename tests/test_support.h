/*
 * Project: RapiDHT
 * File: tests/test_support.h
 * Brief: Shared test helpers: a naive reference Hartley transform derived
 *        straight from the definition, plus small assertion utilities.
 */

#ifndef RAPIDHT_TEST_SUPPORT_H
#define RAPIDHT_TEST_SUPPORT_H

#include "rapidht.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

/*
 * Skips a test that needs the GPU, for either of the two separate reasons it
 * might be unavailable: the backend was not compiled in, or it was but there is
 * no card. The second case is the normal state of a CI runner, which carries
 * the toolkit so the code compiles and no device to run it on.
 *
 * Every test that touches the device must use this. Excluding them by a "GPU"
 * substring in the test name is not enough -- that convention silently failed
 * for DeviceResidentRejectsCpuMode, which needs a device despite testing a
 * rejection path.
 */
#define SKIP_IF_NO_GPU()                                                          \
    do {                                                                          \
        if (!RapiDHT::kCudaEnabled) {                                             \
            GTEST_SKIP() << "Built without CUDA support (RAPIDHT_WITH_CUDA=OFF)"; \
        }                                                                         \
        if (!RapiDHT::IsGpuAvailable()) {                                         \
            GTEST_SKIP() << "No usable CUDA device on this machine";              \
        }                                                                         \
    } while (0)

/// Retained spelling, now meaning the same thing.
#define SKIP_IF_NO_CUDA() SKIP_IF_NO_GPU()

namespace rapidht_test {

/// Hartley kernel: cas(t) = cos(t) + sin(t).
inline double Cas(double t) { return std::cos(t) + std::sin(t); }

/// Extent of a test volume, following HartleyTransform's own convention:
/// `height == 0` means 1D and `depth == 0` means 2D. Elements are stored
/// row-major as idx = i + W * (j + H * k).
struct Dims {
    size_t width = 1;
    size_t height = 1;
    size_t depth = 1;

    static Dims Of(size_t width, size_t height, size_t depth)
    {
        return Dims { width, height == 0 ? size_t { 1 } : height,
            depth == 0 ? size_t { 1 } : depth };
    }

    size_t Total() const { return width * height * depth; }
    size_t Index(size_t i, size_t j, size_t k) const { return i + width * (j + height * k); }
};

/// Direct O(N^2) evaluation of the *true* multidimensional discrete Hartley
/// transform:
///
///   H(a,b,c) = sum_ijk x(i,j,k) * cas(2*pi * (i*a/W + j*b/H + k*c/D))
///
/// This is deliberately not the separable product of independent 1D
/// transforms. The two definitions coincide in 1D but differ in 2D and 3D;
/// RapiDHT implements the true multidimensional transform (separable passes
/// followed by the Bracewell correction), so this is the right reference.
inline std::vector<double> ReferenceDht(const std::vector<double>& in, const Dims& dims)
{
    const double kTwoPi = 2.0 * std::acos(-1.0);
    std::vector<double> out(in.size(), 0.0);

    for (size_t a = 0; a < dims.width; ++a) {
        for (size_t b = 0; b < dims.height; ++b) {
            for (size_t c = 0; c < dims.depth; ++c) {
                double sum = 0.0;
                for (size_t i = 0; i < dims.width; ++i) {
                    for (size_t j = 0; j < dims.height; ++j) {
                        for (size_t k = 0; k < dims.depth; ++k) {
                            const double phase = kTwoPi
                                               * (static_cast<double>(i * a) / static_cast<double>(dims.width)
                                                   + static_cast<double>(j * b) / static_cast<double>(dims.height)
                                                   + static_cast<double>(k * c) / static_cast<double>(dims.depth));
                            sum += in[dims.Index(i, j, k)] * Cas(phase);
                        }
                    }
                }
                out[dims.Index(a, b, c)] = sum;
            }
        }
    }
    return out;
}

/// Deterministic, reproducible test signal. A fixed formula is used instead of
/// a random generator so that a failure can always be replayed.
inline std::vector<double> MakeSignal(size_t count)
{
    std::vector<double> data(count);
    for (size_t i = 0; i < count; ++i) {
        const double t = static_cast<double>(i);
        data[i] = std::sin(0.7 * t) + 0.3 * std::cos(2.1 * t) + 1.0;
    }
    return data;
}

inline double MaxAbs(const std::vector<double>& v)
{
    double m = 0.0;
    for (double x : v) {
        m = std::max(m, std::fabs(x));
    }
    return m;
}

/// Compares two vectors with a tolerance scaled to the magnitude of `expected`,
/// which matters because an unnormalised DHT grows like N.
inline void ExpectClose(const std::vector<double>& actual, const std::vector<double>& expected,
    double relative_tolerance = 1e-12)
{
    ASSERT_EQ(actual.size(), expected.size());
    const double tol = relative_tolerance * std::max(MaxAbs(expected), 1.0);
    for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_NEAR(actual[i], expected[i], tol) << "Mismatch at index " << i;
    }
}

/// Runs the forward transform for the given extent and mode.
inline std::vector<double> Forward(const std::vector<double>& in, size_t width, size_t height,
    size_t depth, RapiDHT::Modes mode)
{
    std::vector<double> data = in;
    RapiDHT::HartleyTransform<double> ht(width, height, depth, mode);
    ht.ForwardTransform(data.data());
    return data;
}

} // namespace rapidht_test

#endif // RAPIDHT_TEST_SUPPORT_H
