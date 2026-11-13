/*
 * Project: RapiDHT
 * File: tests/test_factored_kernels.cpp
 * Brief: R-4 -- the CUDA kernels' own code, run on the host, against the
 *        O(N^2) definition.
 *
 * src/internal/factored_kernels.cuh is compiled here by the host compiler,
 * with the CUDA keywords defined away and every kernel launched as one thread
 * per block, blocks in turn. One thread makes __syncthreads() a no-op and
 * every block-stride loop cover its whole range, so the device code -- index
 * arithmetic, tiling, vector loads, the in-place Bracewell orbits -- is
 * checked on any machine. What this cannot see is anything that only shows
 * with many threads: races, bank conflicts, launch configuration. The _GPU
 * tests in test_correctness.cpp cover those on a device.
 *
 * Both kernels are run wherever their preconditions hold, so the generic one
 * is covered on lengths and widths the launcher would hand to the fast one.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

#include "internal/factored.h"
#include "test_support.h"

// ---- the CUDA keywords and built-ins, for one thread per block ----
namespace {
struct HostDim3 {
    unsigned x = 0, y = 0, z = 0;
};
HostDim3 threadIdx;
HostDim3 blockIdx;
HostDim3 blockDim;
HostDim3 gridDim;
} // namespace

struct float4 {
    float x, y, z, w;
};
struct double2 {
    double x, y;
};
inline float4 make_float4(float x, float y, float z, float w)
{
    return { x, y, z, w };
}
inline double2 make_double2(double x, double y)
{
    return { x, y };
}

#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__
#define __shared__
#define __align__(n) // the buffers below carry the alignment
#define __syncthreads() ((void)0)

#include "internal/factored_kernels.cuh"

namespace RapiDHT {
// The kernels declare these extern; one buffer each is enough for one block.
alignas(16) unsigned char sharedRaw[1 << 20];
alignas(16) unsigned char sharedFast[1 << 20];
alignas(16) unsigned char sharedPipe[1 << 20];
} // namespace RapiDHT

using namespace RapiDHT;
using namespace rapidht_test;

namespace {

enum class Kernel { Generic,
    Fast,
    Pipelined };

// Registers a thread holds for the next batch: with one thread to a block it
// has to hold the whole tile, 512 values at most in the shapes below.
constexpr int kHostMaxQuads = 128;

// Fewer blocks than batches, so that every block runs the pipeline over
// several batches, including a ragged last round.
constexpr size_t kHostGrid = 3;

template <typename T>
struct AxisTables {
    std::vector<T> packed;
    FactoredAxisTables<T> view;
};

template <typename T>
AxisTables<T> MakeAxisTables(size_t n)
{
    AxisTables<T> a;
    const auto t = internal::MakeFactoredTables<T>(n);
    for (const auto* part : { &t.c1, &t.c2, &t.cHalf, &t.twC, &t.twD }) {
        a.packed.insert(a.packed.end(), part->begin(), part->end());
    }
    const size_t n1 = t.split.n1, n2 = t.split.n2, pairs = (n1 / 2 == 0 ? 1 : n1 / 2);
    a.view.c1 = a.packed.data();
    a.view.c2 = a.view.c1 + n1 * n1;
    a.view.cHalf = a.view.c2 + n2 * n2;
    a.view.twC = a.view.cHalf + n2 * n2;
    a.view.twD = a.view.twC + pairs * n2;
    a.view.n = static_cast<int>(n);
    a.view.n1 = static_cast<int>(n1);
    a.view.n2 = static_cast<int>(n2);
    return a;
}

/// One axis, every block in turn. Lanes: as wide as allowed, at most `maxLanes`.
template <typename T>
void RunAxis(T* data, const FactoredAxisTables<T>& t, size_t axisStride, size_t laneStride, size_t laneLength,
    size_t otherStride, size_t otherLength, Kernel kernel, size_t maxLanes)
{
    if (t.n <= 1) {
        return;
    }
    size_t lanes = maxLanes;
    while (lanes > laneLength) {
        lanes /= 2;
    }
    const size_t blocksPerOther = laneLength / lanes;
    blockDim = { 1, 1, 1 };
    threadIdx = { 0, 0, 0 };
    if (kernel == Kernel::Pipelined) {
        const size_t batches = blocksPerOther * otherLength;
        const size_t grid = batches < kHostGrid ? batches : kHostGrid;
        gridDim = { static_cast<unsigned>(grid), 1, 1 };
        // The device path moves quads as vectors exactly when this holds; the
        // host buffers are 16-byte aligned like cudaMalloc's.
        const bool vec = laneStride == 1 && axisStride % 4 == 0 && otherStride % 4 == 0;
        for (size_t b = 0; b < grid; ++b) {
            blockIdx = { static_cast<unsigned>(b), 0, 0 };
            FactoredAxisKernelPipelined<T, kHostMaxQuads>(data, t, axisStride, laneStride, static_cast<int>(lanes),
                static_cast<int>(blocksPerOther), otherStride, static_cast<int>(batches),
                static_cast<int>(FactoredWorkStride<T>(static_cast<size_t>(t.n2), lanes)), vec);
        }
        return;
    }
    for (size_t b = 0; b < blocksPerOther * otherLength; ++b) {
        blockIdx = { static_cast<unsigned>(b), 0, 0 };
        if (kernel == Kernel::Fast) {
            FactoredAxisKernelFast<T>(data, t, axisStride, laneStride, static_cast<int>(lanes),
                static_cast<int>(blocksPerOther), otherStride,
                static_cast<int>(FactoredWorkStride<T>(static_cast<size_t>(t.n2), lanes)));
        } else {
            FactoredAxisKernel<T>(data, t, axisStride, laneStride, static_cast<int>(lanes),
                static_cast<int>(blocksPerOther), otherStride);
        }
    }
}

/// The whole of FactoredOnDevice: three axes, then the in-place Bracewell.
template <typename T>
std::vector<double> RunTransform(const std::vector<double>& input, size_t W, size_t H, size_t D, Kernel kernel)
{
    std::vector<T> v(input.begin(), input.end());
    if (reinterpret_cast<uintptr_t>(v.data()) % 16 != 0) {
        ADD_FAILURE() << "the vector paths need a 16-byte aligned buffer";
    }
    const auto tx = MakeAxisTables<T>(W);
    const auto ty = MakeAxisTables<T>(H);
    const auto tz = MakeAxisTables<T>(D);
    const size_t maxLanes = (kernel == Kernel::Generic) ? 16 : (sizeof(T) == 4 ? 8 : 4);
    RunAxis(v.data(), tx.view, 1, W, H, W * H, D, kernel, maxLanes);
    RunAxis(v.data(), ty.view, W, 1, W, W * H, D, kernel, maxLanes);
    RunAxis(v.data(), tz.view, W * H, 1, W, W, H, kernel, maxLanes);

    blockDim = { 1, 1, 1 };
    threadIdx = { 0, 0, 0 };
    if (D > 1) {
        for (size_t z = 0; z <= D / 2; ++z) {
            for (size_t x = 0; x <= H / 2; ++x) {
                for (size_t y = 0; y <= W / 2; ++y) {
                    blockIdx = { static_cast<unsigned>(y), static_cast<unsigned>(x), static_cast<unsigned>(z) };
                    BracewellInPlace3DKernel<T>(v.data(), static_cast<int>(W), static_cast<int>(H), static_cast<int>(D));
                }
            }
        }
    } else if (H > 1) {
        for (size_t x = 0; x <= H / 2; ++x) {
            for (size_t y = 0; y <= W / 2; ++y) {
                blockIdx = { static_cast<unsigned>(y), static_cast<unsigned>(x), 0 };
                BracewellInPlace2DKernel<T>(v.data(), static_cast<int>(W), static_cast<int>(H));
            }
        }
    }
    return std::vector<double>(v.begin(), v.end());
}

struct Shape {
    size_t width, height, depth;
};

std::string Describe(const Shape& s)
{
    return std::to_string(s.width) + "x" + std::to_string(s.height) + "x" + std::to_string(s.depth);
}

template <typename T>
void ExpectKernelMatchesDefinition(Kernel kernel, const Shape* shapes, size_t count, double tolerance)
{
    for (size_t i = 0; i < count; ++i) {
        const Shape& s = shapes[i];
        SCOPED_TRACE(Describe(s));
        // depth 1 here is a 2D transform, height 1 too a 1D one
        const Dims dims = Dims::Of(s.width, s.height == 1 && s.depth == 1 ? 0 : s.height, s.depth == 1 ? 0 : s.depth);
        const auto input = MakeSignal(dims.Total());
        ExpectClose(RunTransform<T>(input, s.width, s.height, s.depth, kernel), ReferenceDht(input, dims), tolerance);
    }
}

// Every rank, axes of length 2 and 4 (N1/2 the only non-zero column), axes
// longer than one block of lines, unequal axes.
const Shape kGenericShapes[] = {
    { 2, 1, 1 },
    { 8, 1, 1 },
    { 512, 1, 1 },
    { 8, 4, 1 },
    { 4, 64, 1 },
    { 4, 4, 4 },
    { 8, 4, 2 },
    { 64, 8, 4 },
    { 2, 8, 16 },
};

// The fast kernel needs every axis >= 16 (N1, N2 >= 4) and at least 4 lines
// to a block along each lane axis; 32 gives several blocks along a lane axis
// in float (8 lines to a block). Small, because the reference is O(N^2).
const Shape kFastShapes[] = {
    { 16, 16, 1 },
    { 64, 16, 1 },
    { 16, 16, 16 },
    { 32, 16, 16 },
    { 16, 32, 16 },
};

} // namespace

TEST(FactoredKernels, GenericMatchesDefinition_Double)
{
    ExpectKernelMatchesDefinition<double>(Kernel::Generic, kGenericShapes, std::size(kGenericShapes), 1e-11);
}

TEST(FactoredKernels, GenericMatchesDefinition_Float)
{
    ExpectKernelMatchesDefinition<float>(Kernel::Generic, kGenericShapes, std::size(kGenericShapes), 2e-5);
}

TEST(FactoredKernels, FastMatchesDefinition_Double)
{
    ExpectKernelMatchesDefinition<double>(Kernel::Fast, kFastShapes, std::size(kFastShapes), 1e-11);
}

TEST(FactoredKernels, FastMatchesDefinition_Float)
{
    ExpectKernelMatchesDefinition<float>(Kernel::Fast, kFastShapes, std::size(kFastShapes), 2e-5);
}

TEST(FactoredKernels, PipelinedMatchesDefinition_Double)
{
    ExpectKernelMatchesDefinition<double>(Kernel::Pipelined, kFastShapes, std::size(kFastShapes), 1e-11);
}

TEST(FactoredKernels, PipelinedMatchesDefinition_Float)
{
    ExpectKernelMatchesDefinition<float>(Kernel::Pipelined, kFastShapes, std::size(kFastShapes), 2e-5);
}
