/*
 * Project: RapiDHT
 * File: tests/test_factored.cpp
 * Brief: R-4 -- the factored DHT steps against the O(N^2) definition.
 *
 * The CUDA kernel of R-4 runs these steps on a shared-memory tile; the host
 * reference here runs them on the same layout. Pinning the reference to the
 * definition first means a wrong device result can only be the kernel's own
 * fault, not the algebra's.
 */

#include "internal/factored.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "test_support.h"

using namespace RapiDHT;
using namespace RapiDHT::internal;
using namespace rapidht_test;

namespace {

/// Runs the factored steps on `lanes` copies-with-variation of a signal and
/// returns each line's result, so that a lane mix-up shows as a mismatch.
template <typename T>
std::vector<std::vector<double>> FactoredLines(size_t n, size_t lanes)
{
    const auto tables = MakeFactoredTables<T>(n);
    std::vector<T> tile(n * lanes), work(n * lanes);
    for (size_t l = 0; l < lanes; ++l) {
        const auto signal = MakeSignal(n + l); // a different signal per lane
        for (size_t i = 0; i < n; ++i) {
            tile[i * lanes + l] = static_cast<T>(signal[i + l]);
        }
    }
    FactoredDhtLanes(tile.data(), work.data(), lanes, tables);

    std::vector<std::vector<double>> out(lanes, std::vector<double>(n));
    for (size_t l = 0; l < lanes; ++l) {
        for (size_t i = 0; i < n; ++i) {
            out[l][i] = static_cast<double>(tile[i * lanes + l]);
        }
    }
    return out;
}

template <typename T>
void ExpectLinesMatchDefinition(double tolerance)
{
    for (size_t n = 1; n <= 1024; n <<= 1) {
        for (const size_t lanes : { size_t { 1 }, size_t { 3 } }) {
            SCOPED_TRACE("n " + std::to_string(n) + ", lanes " + std::to_string(lanes));
            const auto lines = FactoredLines<T>(n, lanes);
            for (size_t l = 0; l < lanes; ++l) {
                const auto signal = MakeSignal(n + l);
                const std::vector<double> input(signal.begin() + static_cast<long>(l), signal.end());
                ExpectClose(lines[l], ReferenceDht(input, Dims::Of(n, 0, 0)), tolerance);
            }
        }
    }
}

} // namespace

TEST(Factored, SplitsIntoBalancedPowersOfTwo)
{
    EXPECT_EQ(SplitLength(1).n1, 1u);
    EXPECT_EQ(SplitLength(2).n1, 2u);
    EXPECT_EQ(SplitLength(8).n1, 4u);
    EXPECT_EQ(SplitLength(512).n1, 32u);
    EXPECT_EQ(SplitLength(512).n2, 16u);
    EXPECT_EQ(SplitLength(1024).n1, 32u);
    EXPECT_THROW(SplitLength(12), std::invalid_argument);
    EXPECT_THROW(SplitLength(0), std::invalid_argument);
}

TEST(Factored, LinesMatchDefinition_Double)
{
    ExpectLinesMatchDefinition<double>(1e-11);
}

TEST(Factored, LinesMatchDefinition_Float)
{
    ExpectLinesMatchDefinition<float>(2e-5);
}

// The whole 3D path the kernel will take: the factored steps along each axis
// (the separable transform), then the Bracewell correction, against the true
// 3D definition. Non-cubic, with every axis split differently.
TEST(Factored, TrueThreeDimensionalTransformMatchesDefinition)
{
    const size_t W = 16, H = 8, D = 32;
    const Dims dims = Dims::Of(W, H, D);
    const auto input = MakeSignal(dims.Total());
    std::vector<double> v = input;

    const size_t extents[3] = { W, H, D };
    const size_t strides[3] = { 1, W, W * H };
    for (int axis = 0; axis < 3; ++axis) {
        const size_t n = extents[axis];
        const auto tables = MakeFactoredTables<double>(n);
        std::vector<double> tile(n), work(n);
        for (size_t base = 0; base < dims.Total(); ++base) {
            // A line starts wherever this axis's coordinate is zero.
            if ((base / strides[axis]) % n != 0) {
                continue;
            }
            for (size_t i = 0; i < n; ++i) {
                tile[i] = v[base + i * strides[axis]];
            }
            FactoredDhtLanes(tile.data(), work.data(), 1, tables);
            for (size_t i = 0; i < n; ++i) {
                v[base + i * strides[axis]] = tile[i];
            }
        }
    }

    auto mirror = [](size_t i, size_t n) { return i == 0 ? 0 : n - i; };
    std::vector<double> truth(dims.Total());
    for (size_t k = 0; k < D; ++k) {
        for (size_t j = 0; j < H; ++j) {
            for (size_t i = 0; i < W; ++i) {
                const size_t im = mirror(i, W), jm = mirror(j, H), km = mirror(k, D);
                truth[dims.Index(i, j, k)] = 0.5
                                           * (v[dims.Index(im, j, k)] + v[dims.Index(i, jm, k)]
                                               + v[dims.Index(i, j, km)] - v[dims.Index(im, jm, km)]);
            }
        }
    }
    ExpectClose(truth, ReferenceDht(input, dims), 1e-10);
}
