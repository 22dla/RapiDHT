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

#include <rapidht/device_volume.h>
#include <rapidht/transform.h>
#ifdef RAPIDHT_WITH_CUDA
#include <cuda_runtime.h>

#include "internal/device_access.h"
#endif

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
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

/// Integers of type TIn over its full range, different per lane, from a
/// fixed formula (Knuth's multiplicative hash): no two lanes alike.
template <typename TIn>
std::vector<TIn> IntegerLines(size_t n, size_t lanes)
{
    std::vector<TIn> in(n * lanes);
    for (size_t i = 0; i < n; ++i) {
        for (size_t l = 0; l < lanes; ++l) {
            const uint32_t h = static_cast<uint32_t>((i * 7 + l * 131 + 1) * 2654435761u);
            in[i * lanes + l] = static_cast<TIn>(h >> (32 - 8 * sizeof(TIn)));
        }
    }
    return in;
}

/// Offsets of R-5: none, half the range, and an arbitrary one.
template <typename TIn>
std::vector<double> IntegerOffsets()
{
    return sizeof(TIn) == 1 ? std::vector<double> { 0.0, 128.0, 77.0 } : std::vector<double> { 0.0, 32768.0, 12345.0 };
}

/// R-5 step 2.1: FactoredDhtLanesFrom against the definition of the DHT of
/// (x - offset), line by line.
template <typename TIn, typename T>
void ExpectIntegerLinesMatchDefinition(double tolerance)
{
    constexpr size_t lanes = 3;
    for (size_t n = 1; n <= 1024; n <<= 1) {
        const auto tables = MakeFactoredTables<T>(n);
        const auto in = IntegerLines<TIn>(n, lanes);
        for (const double offset : IntegerOffsets<TIn>()) {
            SCOPED_TRACE("n " + std::to_string(n) + ", offset " + std::to_string(offset));
            std::vector<T> tile(n * lanes), work(n * lanes);
            FactoredDhtLanesFrom(in.data(), static_cast<T>(offset), tile.data(), work.data(), lanes, tables);
            for (size_t l = 0; l < lanes; ++l) {
                std::vector<double> line(n), got(n);
                for (size_t i = 0; i < n; ++i) {
                    line[i] = static_cast<double>(in[i * lanes + l]) - offset;
                    got[i] = static_cast<double>(tile[i * lanes + l]);
                }
                ExpectClose(got, ReferenceDht(line, Dims::Of(n, 0, 0)), tolerance);
            }
        }
    }
}

/// Rule 9 of docs/R5_PLAN.md: reading the integers in step 1 is the float path
/// on a pre-converted input, bit for bit.
template <typename TIn, typename T>
void ExpectIntegerLinesBitIdenticalToConverted()
{
    constexpr size_t lanes = 5;
    for (size_t n = 1; n <= 1024; n <<= 1) {
        const auto tables = MakeFactoredTables<T>(n);
        const auto in = IntegerLines<TIn>(n, lanes);
        for (const double offset : IntegerOffsets<TIn>()) {
            SCOPED_TRACE("n " + std::to_string(n) + ", offset " + std::to_string(offset));
            std::vector<T> fromIntegers(n * lanes), converted(n * lanes), work(n * lanes);
            FactoredDhtLanesFrom(in.data(), static_cast<T>(offset), fromIntegers.data(), work.data(), lanes, tables);
            for (size_t i = 0; i < n * lanes; ++i) {
                converted[i] = static_cast<T>(in[i]) - static_cast<T>(offset);
            }
            FactoredDhtLanes(converted.data(), work.data(), lanes, tables);
            ASSERT_EQ(0, std::memcmp(fromIntegers.data(), converted.data(), n * lanes * sizeof(T)));
        }
    }
}

} // namespace

TEST(Factored, IntegerLinesMatchDefinition_Double)
{
    ExpectIntegerLinesMatchDefinition<uint8_t, double>(1e-11);
    ExpectIntegerLinesMatchDefinition<uint16_t, double>(1e-11);
}

TEST(Factored, IntegerLinesMatchDefinition_Float)
{
    ExpectIntegerLinesMatchDefinition<uint8_t, float>(2e-5);
    ExpectIntegerLinesMatchDefinition<uint16_t, float>(2e-5);
}

TEST(Factored, IntegerLinesAreBitIdenticalToConvertedInput)
{
    ExpectIntegerLinesBitIdenticalToConverted<uint8_t, float>();
    ExpectIntegerLinesBitIdenticalToConverted<uint16_t, float>();
    ExpectIntegerLinesBitIdenticalToConverted<uint8_t, double>();
    ExpectIntegerLinesBitIdenticalToConverted<uint16_t, double>();
}

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

namespace {

/// The factored DHT along one axis of v, line by line, in T.
template <typename T>
void FactoredAlongAxis(std::vector<T>& v, const Dims& dims, int axis)
{
    const size_t extents[3] = { dims.width, dims.height, dims.depth };
    const size_t strides[3] = { 1, dims.width, dims.width * dims.height };
    const size_t n = extents[axis];
    if (n <= 1) {
        return;
    }
    const auto tables = MakeFactoredTables<T>(n);
    std::vector<T> tile(n), work(n);
    for (size_t base = 0; base < dims.Total(); ++base) {
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

/// The path the fused last pass will take on the GPU: z, then y, then x by
/// groups of four mirror lines with the Bracewell combination folded in.
/// Every (y, z) belongs to exactly one group, (y, z) with y <= H/2, z <= D/2.
template <typename T>
std::vector<double> FusedTransform(const std::vector<double>& input, const Dims& dims)
{
    std::vector<T> v(input.begin(), input.end());
    FactoredAlongAxis(v, dims, 2);
    FactoredAlongAxis(v, dims, 1);

    const size_t W = dims.width, H = dims.height, D = dims.depth;
    const auto tables = MakeFactoredTables<T>(W);
    std::vector<T> tile(4 * W), work(4 * W);
    for (size_t z = 0; z <= D / 2; ++z) {
        for (size_t y = 0; y <= H / 2; ++y) {
            const size_t my = (H - y) % H, mz = (D - z) % D;
            const size_t rows[4] = { W * (y + H * z), W * (my + H * z), W * (y + H * mz), W * (my + H * mz) };
            for (size_t i = 0; i < W; ++i) {
                for (size_t l = 0; l < 4; ++l) {
                    tile[i * 4 + l] = v[rows[l] + i];
                }
            }
            FactoredDhtBracewellGroup(tile.data(), work.data(), tables);
            for (size_t i = 0; i < W; ++i) {
                for (size_t l = 0; l < 4; ++l) {
                    v[rows[l] + i] = tile[i * 4 + l];
                }
            }
        }
    }
    return std::vector<double>(v.begin(), v.end());
}

template <typename T>
void ExpectFusedMatchesDefinition(double tolerance)
{
    // Non-cubic, every axis split differently; then the shapes where a mirror
    // group degenerates: H = 2 (every y is its own mirror), D = 2, the 2D case
    // (D = 1, groups of two lines) and W = 2 (shortest line).
    const Dims shapes[] = {
        Dims::Of(16, 8, 32),
        Dims::Of(32, 4, 8),
        Dims::Of(8, 2, 16),
        Dims::Of(16, 16, 2),
        Dims::Of(32, 16, 0),
        Dims::Of(64, 2, 0),
        Dims::Of(2, 8, 4),
    };
    for (const Dims& dims : shapes) {
        SCOPED_TRACE(std::to_string(dims.width) + "x" + std::to_string(dims.height) + "x"
                     + std::to_string(dims.depth));
        const auto input = MakeSignal(dims.Total());
        ExpectClose(FusedTransform<T>(input, dims), ReferenceDht(input, dims), tolerance);
    }
}

} // namespace

TEST(Factored, FusedBracewellGroupMatchesDefinition_Double)
{
    ExpectFusedMatchesDefinition<double>(1e-11);
}

TEST(Factored, FusedBracewellGroupMatchesDefinition_Float)
{
    ExpectFusedMatchesDefinition<float>(2e-5);
}

#ifdef RAPIDHT_WITH_CUDA
namespace {

struct IntegerShape {
    size_t width, height, depth;
};

/*
 * R-5 step 2.3 on the device: the transform of an integer volume
 * (DeviceAccess::ForwardFromIntegers -- the first present pass reads the
 * integers, offset * N back into H(0)) against the float path on the same
 * values converted beforehand, with the same offset returned to H(0) the same
 * way. Same passes, same kernel variants, same arithmetic: bit for bit.
 */
template <typename TIn>
void ExpectIntegerInputMatchesConvertedFloat(const IntegerShape& s, double offset, bool separable)
{
    const size_t count = s.width * (s.height == 0 ? 1 : s.height) * (s.depth == 0 ? 1 : s.depth);
    std::vector<TIn> host(count);
    for (size_t i = 0; i < count; ++i) {
        host[i] = static_cast<TIn>(static_cast<uint32_t>((i + 1) * 2654435761u) >> (32 - 8 * sizeof(TIn)));
    }

    HartleyTransform<float> transform(s.width, s.height, s.depth, Modes::GPUFactored);
    TIn* in = nullptr;
    ASSERT_EQ(cudaMalloc(&in, count * sizeof(TIn)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(in, host.data(), count * sizeof(TIn), cudaMemcpyHostToDevice), cudaSuccess);
    DeviceVolume<float> out(count);
    internal::DeviceAccess::ForwardFromIntegers(transform, in, out, offset, separable);
    std::vector<float> got(count);
    out.Download(got.data());
    // The input is read, never written.
    std::vector<TIn> after(count);
    ASSERT_EQ(cudaMemcpy(after.data(), in, count * sizeof(TIn), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(in);
    ASSERT_EQ(after, host);

    std::vector<float> expected(count);
    for (size_t i = 0; i < count; ++i) {
        expected[i] = static_cast<float>(host[i]) - static_cast<float>(offset);
    }
    DeviceVolume<float> viaFloat(count);
    viaFloat.Upload(expected.data());
    if (separable) {
        transform.ForwardSeparable(viaFloat);
    } else {
        transform.ForwardTransform(viaFloat);
    }
    viaFloat.Download(expected.data());
    if (offset != 0.0) {
        expected[0] = static_cast<float>(static_cast<double>(expected[0]) + offset * static_cast<double>(count));
    }
    ASSERT_EQ(got, expected);
}

} // namespace

TEST(Factored, IntegerInputMatchesConvertedFloat_GPU)
{
    SKIP_IF_NO_GPU();
    // Cubes on each kernel variant of z (128, 256 and 512 take different ones),
    // unequal axes, 2D and 1D (the first present pass is y, then x), a 2D
    // transform of height 1 (its first pass is the fused last one, so the
    // integers are converted first) and axes too short for the pipelined
    // kernel (converted first as well).
    const IntegerShape shapes[] = { { 128, 128, 128 }, { 32, 64, 256 }, { 64, 32, 512 }, { 512, 256, 0 },
        { 1024, 0, 0 }, { 64, 1, 0 }, { 8, 4, 16 } };
    for (const IntegerShape& s : shapes) {
        for (const bool separable : { false, true }) {
            if (separable && s.height == 0) {
                continue; // 1D has no separable variant
            }
            SCOPED_TRACE(std::to_string(s.width) + "x" + std::to_string(s.height) + "x" + std::to_string(s.depth)
                         + (separable ? ", separable" : ""));
            for (const double offset : { 0.0, 128.0, 77.0 }) {
                ExpectIntegerInputMatchesConvertedFloat<uint8_t>(s, offset, separable);
            }
            for (const double offset : { 0.0, 32768.0, 12345.0 }) {
                ExpectIntegerInputMatchesConvertedFloat<uint16_t>(s, offset, separable);
            }
        }
    }
}

TEST(Factored, IntegerInputAtBenchmarkSizes_GPU)
{
    SKIP_IF_NO_GPU();
    // The sizes the benchmark times: the first pass along z at 512^3 takes the
    // four-tile in-place kernel, which the shapes above do not reach.
    ExpectIntegerInputMatchesConvertedFloat<uint16_t>({ 256, 256, 256 }, 32768.0, false);
    ExpectIntegerInputMatchesConvertedFloat<uint16_t>({ 512, 512, 512 }, 25914.0, false);
    ExpectIntegerInputMatchesConvertedFloat<uint8_t>({ 512, 512, 512 }, 128.0, true);
}

TEST(Factored, IntegerInput2DAnd1D_GPU)
{
    SKIP_IF_NO_GPU();
    // R-5 step 2.6: in 2D the first present pass is y, in 1D the only one (x,
    // which the pipelined kernel cannot take -- one line -- so the integers are
    // converted first). 512^2-2048^2, unequal sides, 1D up to 2048 (the
    // longest axis GPUFactored takes).
    const IntegerShape shapes[] = { { 512, 512, 0 }, { 1024, 1024, 0 }, { 2048, 2048, 0 }, { 2048, 512, 0 },
        { 512, 2048, 0 }, { 512, 0, 0 }, { 1024, 0, 0 }, { 2048, 0, 0 } };
    for (const IntegerShape& s : shapes) {
        for (const bool separable : { false, true }) {
            if (separable && s.height == 0) {
                continue;
            }
            SCOPED_TRACE(std::to_string(s.width) + "x" + std::to_string(s.height) + (separable ? ", separable" : ""));
            ExpectIntegerInputMatchesConvertedFloat<uint8_t>(s, 128.0, separable);
            ExpectIntegerInputMatchesConvertedFloat<uint16_t>(s, 32768.0, separable);
            ExpectIntegerInputMatchesConvertedFloat<uint16_t>(s, 0.0, separable);
        }
    }
}
#endif
