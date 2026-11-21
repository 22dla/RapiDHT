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

#include <algorithm>
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
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __shared__
#define __align__(n) // the buffers below carry the alignment
#define __syncthreads() ((void)0)
#define __launch_bounds__(...)

// Device code calls the CUDA built-in min(int, int).
using std::min;

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
    Pipelined, // through a work buffer (v2.3)
    InPlace, // the same steps in place (R-4 step 4B)
    PipelinedFold, // and each with the twiddle inside step 1 (step 4A)
    InPlaceFold,
    InPlaceQuarters }; // in place with kQuarterQuads: the next batch fetched in quarters (step 4.Z)

// Registers a thread holds for the next batch: with one thread to a block it
// has to hold the whole batch -- up to 32 lines of n = 32 in the wide-batch
// tests, 1 024 values. Too few and the rest of the batch is never loaded; the
// device launcher checks the same bound against kFactoredMaxQuads.
constexpr int kHostMaxQuads = 256;

// Likewise the 4 x 4 output tiles a thread holds across the in-place steps
// (R-4 step 4B): n * lanes / 16 of them, 64 for 32 lines of n = 32.
constexpr int kHostItems = 64;

// Fewer blocks than batches, so that every block runs the pipeline over
// several batches, including a ragged last round.
constexpr size_t kHostGrid = 3;

// Prefetch slots of Kernel::InPlaceQuarters. From 16 slots the kernel fetches
// the next batch in four parts at hooks inside the steps; with one thread a
// block, quad q sits in slot q, so 32 slots and batches of 32 quads (4 lines
// of n = 32) put quads in every quarter. More slots would leave the upper
// quarters empty and the split untested.
constexpr int kQuarterQuads = 32;

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
    size_t otherStride, size_t otherLength, Kernel kernel, size_t maxLanes, int ablate = kAblateNone)
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
    if (kernel != Kernel::Generic && kernel != Kernel::Fast) {
        const size_t batches = blocksPerOther * otherLength;
        const size_t grid = batches < kHostGrid ? batches : kHostGrid;
        gridDim = { static_cast<unsigned>(grid), 1, 1 };
        // The device path moves quads as vectors exactly when this holds; the
        // host buffers are 16-byte aligned like cudaMalloc's.
        const bool vec = laneStride == 1 && axisStride % 4 == 0 && otherStride % 4 == 0;
        for (size_t b = 0; b < grid; ++b) {
            blockIdx = { static_cast<unsigned>(b), 0, 0 };
            const int ws = static_cast<int>(FactoredWorkStride<T>(static_cast<size_t>(t.n2), lanes));
            const int lanesI = static_cast<int>(lanes);
            const int perOther = static_cast<int>(blocksPerOther);
            const int batchesI = static_cast<int>(batches);
            if (kernel == Kernel::InPlaceQuarters) {
                FactoredAxisKernelInPlace<T, kQuarterQuads, kHostItems, 1, true>(
                    data, t, axisStride, laneStride, lanesI, perOther, otherStride, batchesI, ws, vec, ablate);
            } else if (kernel == Kernel::InPlace) {
                FactoredAxisKernelInPlace<T, kHostMaxQuads, kHostItems, 1, false>(
                    data, t, axisStride, laneStride, lanesI, perOther, otherStride, batchesI, ws, vec, ablate);
            } else if (kernel == Kernel::InPlaceFold) {
                FactoredAxisKernelInPlace<T, kHostMaxQuads, kHostItems, 1, true>(
                    data, t, axisStride, laneStride, lanesI, perOther, otherStride, batchesI, ws, vec, ablate);
            } else if (kernel == Kernel::PipelinedFold) {
                FactoredAxisKernelPipelined<T, kHostMaxQuads, true>(
                    data, t, axisStride, laneStride, lanesI, perOther, otherStride, batchesI, ws, vec, ablate);
            } else {
                FactoredAxisKernelPipelined<T, kHostMaxQuads, false>(
                    data, t, axisStride, laneStride, lanesI, perOther, otherStride, batchesI, ws, vec, ablate);
            }
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
std::vector<double> RunTransform(
    const std::vector<double>& input, size_t W, size_t H, size_t D, Kernel kernel, size_t lanesOverride = 0)
{
    std::vector<T> v(input.begin(), input.end());
    if (reinterpret_cast<uintptr_t>(v.data()) % 16 != 0) {
        ADD_FAILURE() << "the vector paths need a 16-byte aligned buffer";
    }
    const auto tx = MakeAxisTables<T>(W);
    const auto ty = MakeAxisTables<T>(H);
    const auto tz = MakeAxisTables<T>(D);
    const size_t maxLanes = (kernel == Kernel::Generic) ? 16 : (sizeof(T) == 4 ? 8 : 4);
    const size_t lanes = lanesOverride != 0 ? lanesOverride : maxLanes;
    RunAxis(v.data(), tx.view, 1, W, H, W * H, D, kernel, lanes);
    RunAxis(v.data(), ty.view, W, 1, W, W * H, D, kernel, lanes);
    RunAxis(v.data(), tz.view, W * H, 1, W, W, H, kernel, lanes);

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

TEST(FactoredKernels, InPlaceMatchesDefinition_Double)
{
    ExpectKernelMatchesDefinition<double>(Kernel::InPlace, kFastShapes, std::size(kFastShapes), 1e-11);
}

TEST(FactoredKernels, InPlaceMatchesDefinition_Float)
{
    ExpectKernelMatchesDefinition<float>(Kernel::InPlace, kFastShapes, std::size(kFastShapes), 2e-5);
}

TEST(FactoredKernels, FoldedTwiddleMatchesDefinition_Double)
{
    ExpectKernelMatchesDefinition<double>(Kernel::PipelinedFold, kFastShapes, std::size(kFastShapes), 1e-11);
    ExpectKernelMatchesDefinition<double>(Kernel::InPlaceFold, kFastShapes, std::size(kFastShapes), 1e-11);
}

TEST(FactoredKernels, FoldedTwiddleMatchesDefinition_Float)
{
    ExpectKernelMatchesDefinition<float>(Kernel::PipelinedFold, kFastShapes, std::size(kFastShapes), 2e-5);
    ExpectKernelMatchesDefinition<float>(Kernel::InPlaceFold, kFastShapes, std::size(kFastShapes), 2e-5);
}

// The "io only" diagnostic mode of the pipelined kernel (RAPIDHT_R4_ABLATE=io,
// R-4 step 1.1) must move every value through the same loads and stores as the
// real pass and put it back where it came from: then the time it measures is
// the time of the data movement of a real pass, and nothing is skipped. So
// along each axis the pass is the identity.
template <typename T>
void ExpectIoModeIsIdentity(Kernel kernel)
{
    const Shape shapes[] = { { 16, 16, 16 }, { 64, 16, 8 }, { 16, 64, 8 }, { 32, 16, 64 } };
    for (const Shape& s : shapes) {
        SCOPED_TRACE(Describe(s));
        const size_t W = s.width, H = s.height, D = s.depth;
        std::vector<T> v(W * H * D);
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<T>(static_cast<double>((i * 2654435761u) % 1000003) * 1e-3);
        }
        const std::vector<T> original = v;
        const auto tx = MakeAxisTables<T>(W);
        const auto ty = MakeAxisTables<T>(H);
        const auto tz = MakeAxisTables<T>(D);
        const size_t lanes = sizeof(T) == 4 ? 8 : 4;
        RunAxis(v.data(), tx.view, 1, W, H, W * H, D, kernel, lanes, kAblateIo);
        RunAxis(v.data(), ty.view, W, 1, W, W * H, D, kernel, lanes, kAblateIo);
        RunAxis(v.data(), tz.view, W * H, 1, W, W, H, kernel, lanes, kAblateIo);
        size_t moved = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            moved += (v[i] != original[i]) ? 1 : 0;
        }
        EXPECT_EQ(moved, 0u);
    }
}

TEST(FactoredKernels, PipelinedIoModeIsIdentity_Float)
{
    ExpectIoModeIsIdentity<float>(Kernel::Pipelined);
    ExpectIoModeIsIdentity<float>(Kernel::InPlace);
}

TEST(FactoredKernels, PipelinedIoModeIsIdentity_Double)
{
    ExpectIoModeIsIdentity<double>(Kernel::Pipelined);
    ExpectIoModeIsIdentity<double>(Kernel::InPlace);
}

// Wide batches (R-4 step 2.2): on short axes the launcher gives the pipelined
// kernel 16 or 32 lines to a batch. RunAxis halves the width where a lane axis
// is shorter, so each shape reaches 32 lines on one pass and 16 on another;
// the second one also has 32 lines of n = 32 (1 024 values) in one batch.
const Shape kWideShapes[] = {
    { 16, 32, 16 },
    { 32, 32, 16 },
};

template <typename T>
void ExpectWideBatchesMatchDefinition(double tolerance)
{
    for (const Shape& s : kWideShapes) {
        const Dims dims = Dims::Of(s.width, s.height, s.depth);
        const auto input = MakeSignal(dims.Total());
        const auto expected = ReferenceDht(input, dims); // O(N^2): once per shape
        for (const size_t lanes : { size_t { 16 }, size_t { 32 } }) {
            for (const Kernel kernel :
                { Kernel::Pipelined, Kernel::InPlace, Kernel::PipelinedFold, Kernel::InPlaceFold }) {
                SCOPED_TRACE(Describe(s) + ", " + std::to_string(lanes) + " lines, variant "
                             + std::to_string(static_cast<int>(kernel)));
                ExpectClose(RunTransform<T>(input, s.width, s.height, s.depth, kernel, lanes), expected, tolerance);
            }
        }
    }
}

// The fetch in quarters (step 4.Z): 4 lines a batch, axes of 16 and 32, so a
// batch of 16 or 32 quads fills two or all four quarters of the 32 slots, and
// kHostGrid blocks each take several batches -- the next one is fetched in parts
// while the current one is computed.
template <typename T>
void ExpectQuartersMatchDefinition(double tolerance)
{
    const Shape shapes[] = { { 16, 16, 16 }, { 32, 16, 16 }, { 16, 32, 16 } };
    for (const Shape& s : shapes) {
        SCOPED_TRACE(Describe(s));
        const Dims dims = Dims::Of(s.width, s.height, s.depth);
        const auto input = MakeSignal(dims.Total());
        ExpectClose(RunTransform<T>(input, s.width, s.height, s.depth, Kernel::InPlaceQuarters, 4),
            ReferenceDht(input, dims), tolerance);
    }
}

TEST(FactoredKernels, FetchInQuartersMatchesDefinition_Float)
{
    ExpectQuartersMatchDefinition<float>(2e-5);
}

TEST(FactoredKernels, FetchInQuartersMatchesDefinition_Double)
{
    ExpectQuartersMatchDefinition<double>(1e-11);
}

TEST(FactoredKernels, PipelinedWideBatchesMatchDefinition_Float)
{
    ExpectWideBatchesMatchDefinition<float>(2e-5);
}

TEST(FactoredKernels, PipelinedWideBatchesMatchDefinition_Double)
{
    ExpectWideBatchesMatchDefinition<double>(1e-11);
}

// The fused last pass (R-4 step 3.3): z and y as before, then x with the
// Bracewell combination folded in, groups of four mirror lines to a quad.
// Shapes where groups degenerate (H = 2, D = 2, 2D) and where the group count
// is not a multiple of the batch, so that the last batch is padded; every
// batch width the launcher can pick.
template <typename T>
std::vector<double> RunFusedTransform(
    const std::vector<double>& input, size_t W, size_t H, size_t D, size_t lanes, bool inPlace, bool fold)
{
    std::vector<T> v(input.begin(), input.end());
    const auto tx = MakeAxisTables<T>(W);
    const auto ty = MakeAxisTables<T>(H);
    const auto tz = MakeAxisTables<T>(D);
    RunAxis(v.data(), tz.view, W * H, 1, W, W, H, Kernel::Generic, 16);
    RunAxis(v.data(), ty.view, W, 1, W, W * H, D, Kernel::Generic, 16);

    const size_t groupsY = H / 2 + 1; // 1 when H = 1
    const size_t groupsZ = D / 2 + 1; // 1 in 2D
    const size_t groups = groupsY * groupsZ;
    const size_t perBatch = lanes / 4;
    const size_t batches = (groups + perBatch - 1) / perBatch;
    const size_t grid = batches < kHostGrid ? batches : kHostGrid;
    blockDim = { 1, 1, 1 };
    threadIdx = { 0, 0, 0 };
    gridDim = { static_cast<unsigned>(grid), 1, 1 };
    for (size_t b = 0; b < grid; ++b) {
        blockIdx = { static_cast<unsigned>(b), 0, 0 };
        const int ws = static_cast<int>(FactoredWorkStride<T>(static_cast<size_t>(tx.view.n2), lanes));
        const int Hi = static_cast<int>(H);
        const int Di = static_cast<int>(D);
        const int gy = static_cast<int>(groupsY);
        const int gs = static_cast<int>(groups);
        const int li = static_cast<int>(lanes);
        const int bi = static_cast<int>(batches);
        if (inPlace && fold) {
            FactoredLastAxisBracewellKernelInPlace<T, kHostMaxQuads, kHostItems, 1, true>(
                v.data(), tx.view, Hi, Di, gy, gs, li, bi, ws);
        } else if (inPlace) {
            FactoredLastAxisBracewellKernelInPlace<T, kHostMaxQuads, kHostItems, 1, false>(
                v.data(), tx.view, Hi, Di, gy, gs, li, bi, ws);
        } else if (fold) {
            FactoredLastAxisBracewellKernel<T, kHostMaxQuads, true>(v.data(), tx.view, Hi, Di, gy, gs, li, bi, ws);
        } else {
            FactoredLastAxisBracewellKernel<T, kHostMaxQuads, false>(v.data(), tx.view, Hi, Di, gy, gs, li, bi, ws);
        }
    }
    return std::vector<double>(v.begin(), v.end());
}

template <typename T>
void ExpectFusedLastPassMatchesDefinition(double tolerance)
{
    const Shape shapes[] = { { 16, 8, 32 }, { 32, 4, 8 }, { 16, 2, 16 }, { 16, 16, 2 }, { 32, 16, 1 }, { 16, 2, 1 } };
    for (const Shape& s : shapes) {
        const Dims dims = Dims::Of(s.width, s.height, s.depth == 1 ? 0 : s.depth);
        const auto input = MakeSignal(dims.Total());
        const auto expected = ReferenceDht(input, dims);
        for (const size_t lanes : { size_t { 4 }, size_t { 8 }, size_t { 16 }, size_t { 32 } }) {
            for (const bool inPlace : { false, true }) {
                for (const bool fold : { false, true }) {
                    SCOPED_TRACE(Describe(s) + ", " + std::to_string(lanes) + " lines" + (inPlace ? ", in place" : "")
                                 + (fold ? ", twiddle in step 1" : ""));
                    ExpectClose(RunFusedTransform<T>(input, s.width, s.height, s.depth, lanes, inPlace, fold), expected,
                        tolerance);
                }
            }
        }
    }
}

TEST(FactoredKernels, FusedLastPassMatchesDefinition_Float)
{
    ExpectFusedLastPassMatchesDefinition<float>(2e-5);
}

TEST(FactoredKernels, FusedLastPassMatchesDefinition_Double)
{
    ExpectFusedLastPassMatchesDefinition<double>(1e-11);
}

// The order of step 1's frequencies (R-4 step 4A) for every N1 the kernels
// take, not only the N1 = 4 and 8 of the shapes above: a permutation of
// 0..N1-1 whose slots 4g, 4g+1 and 4g+2, 4g+3 are mirror pairs (k, N1 - k)
// with k < N1/2 -- except slots 0 and 1, which must be 0 and N1/2.
TEST(FactoredKernels, Step1OrderPairsMirrorFrequencies)
{
    for (int n1 = 4; n1 <= 64; n1 *= 2) {
        SCOPED_TRACE("N1 " + std::to_string(n1));
        std::vector<int> seen(static_cast<size_t>(n1), 0);
        for (int p = 0; p < n1; ++p) {
            const int k = FactoredStep1Order(p, n1);
            ASSERT_GE(k, 0);
            ASSERT_LT(k, n1);
            ++seen[static_cast<size_t>(k)];
        }
        for (int k = 0; k < n1; ++k) {
            EXPECT_EQ(seen[static_cast<size_t>(k)], 1) << "frequency " << k;
        }
        EXPECT_EQ(FactoredStep1Order(0, n1), 0);
        EXPECT_EQ(FactoredStep1Order(1, n1), n1 / 2);
        for (int p = 2; p < n1; p += 2) {
            const int lo = FactoredStep1Order(p, n1);
            EXPECT_LT(lo, n1 / 2) << "slot " << p;
            EXPECT_EQ(FactoredStep1Order(p + 1, n1), n1 - lo) << "slot " << p;
        }
    }
}

// R-4 step 4D: kernels compiled for one (length, batch width) against the same
// kernels with runtime sizes, for every pair the launcher specializes and the
// variant (work buffer or in place, twiddle folded or not) it uses it with.
// The runtime-sized kernels are checked against the definition above; here the
// lengths are the real ones (128..1024), too long for the O(N^2) reference.
// One host thread holds a whole batch: up to 512 x 32 or 1024 x 16 values, so
// the register budgets of these instantiations are larger than kHostMaxQuads /
// kHostItems.
constexpr int kBigQuads = 4096;

double MaxAbsDifference(const std::vector<double>& a, const std::vector<double>& b)
{
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}
constexpr int kBigItems = 1024;

// kN, kL: the specialization (0, 0 = runtime sizes); n, lanes: the real ones.
template <typename T, int kItems, bool kFold, int kN, int kL>
std::vector<double> RunAxisX(const std::vector<double>& input, size_t n, size_t lanes, size_t D)
{
    std::vector<T> v(input.begin(), input.end());
    const auto t = MakeAxisTables<T>(n);
    const size_t W = n, H = lanes;
    const int L = static_cast<int>(lanes);
    const int ws = static_cast<int>(FactoredWorkStride<T>(static_cast<size_t>(t.view.n2), H));
    blockDim = { 1, 1, 1 };
    threadIdx = { 0, 0, 0 };
    gridDim = { static_cast<unsigned>(D), 1, 1 };
    for (size_t b = 0; b < D; ++b) {
        blockIdx = { static_cast<unsigned>(b), 0, 0 };
        // A batch is the H lines of one z-plane; the lane axis is y.
        if constexpr (kItems == 0) {
            FactoredAxisKernelPipelined<T, kBigQuads, kFold, kN, kL>(v.data(), t.view, 1, W, L, 1, W * H,
                static_cast<int>(D), ws, false, kAblateNone);
        } else {
            FactoredAxisKernelInPlace<T, kBigQuads, kBigItems, 1, kFold, kN, kL>(v.data(), t.view, 1, W, L, 1,
                W * H, static_cast<int>(D), ws, false, kAblateNone);
        }
    }
    return std::vector<double>(v.begin(), v.end());
}

template <typename T, int kItems, bool kFold, int kN, int kL>
void ExpectAxisSpecializationMatches(double tolerance)
{
    SCOPED_TRACE("n " + std::to_string(kN) + ", " + std::to_string(kL) + " lines, "
                 + (kItems == 0 ? "work buffer" : "in place") + (kFold ? ", twiddle in step 1" : ""));
    const size_t D = 3;
    const auto input = MakeSignal(static_cast<size_t>(kN) * kL * D);
    const size_t n = kN, lanes = kL;
    const auto specialized = RunAxisX<T, kItems, kFold, kN, kL>(input, n, lanes, D);
    ExpectClose(specialized, RunAxisX<T, kItems, kFold, 0, 0>(input, n, lanes, D), tolerance);
    EXPECT_GT(MaxAbsDifference(specialized, input), 1.0) << "the kernel left the data as it was";
}

template <typename T, bool kInPlace, bool kFold, int kN, int kL>
std::vector<double> RunFusedX(const std::vector<double>& input, size_t n, int lanes, size_t H, size_t D)
{
    std::vector<T> v(input.begin(), input.end());
    const auto t = MakeAxisTables<T>(n);
    const int groupsY = static_cast<int>(H / 2 + 1);
    const int groups = groupsY * static_cast<int>(D / 2 + 1);
    const int perBatch = lanes / 4;
    const int batches = (groups + perBatch - 1) / perBatch;
    const int ws = static_cast<int>(FactoredWorkStride<T>(static_cast<size_t>(t.view.n2), static_cast<size_t>(lanes)));
    blockDim = { 1, 1, 1 };
    threadIdx = { 0, 0, 0 };
    gridDim = { static_cast<unsigned>(batches), 1, 1 };
    for (int b = 0; b < batches; ++b) {
        blockIdx = { static_cast<unsigned>(b), 0, 0 };
        if constexpr (kInPlace) {
            FactoredLastAxisBracewellKernelInPlace<T, kBigQuads, kBigItems, 1, kFold, kN, kL>(v.data(), t.view,
                static_cast<int>(H), static_cast<int>(D), groupsY, groups, lanes, batches, ws);
        } else {
            FactoredLastAxisBracewellKernel<T, kBigQuads, kFold, kN, kL>(v.data(), t.view, static_cast<int>(H),
                static_cast<int>(D), groupsY, groups, lanes, batches, ws);
        }
    }
    return std::vector<double>(v.begin(), v.end());
}

template <typename T, bool kInPlace, bool kFold, int kN, int kL>
void ExpectFusedSpecializationMatches(double tolerance)
{
    SCOPED_TRACE("fused, n " + std::to_string(kN) + ", " + std::to_string(kL) + " lines, "
                 + (kInPlace ? "in place" : "work buffer") + (kFold ? ", twiddle in step 1" : ""));
    const size_t H = 4, D = 4; // 9 groups of mirror lines: more than one batch at 8 and 16 lines
    const auto input = MakeSignal(static_cast<size_t>(kN) * H * D);
    const auto specialized = RunFusedX<T, kInPlace, kFold, kN, kL>(input, kN, kL, H, D);
    ExpectClose(specialized, RunFusedX<T, kInPlace, kFold, 0, 0>(input, kN, kL, H, D), tolerance);
    EXPECT_GT(MaxAbsDifference(specialized, input), 1.0) << "the kernel left the data as it was";
}

template <typename T>
void ExpectSpecializationsMatch(double tolerance)
{
    // The pairs and variants FactoredSpecializes (kernels.cu) picks.
    ExpectAxisSpecializationMatches<T, 0, false, 128, 16>(tolerance);
    ExpectAxisSpecializationMatches<T, 2, true, 512, 16>(tolerance);
    ExpectAxisSpecializationMatches<T, 4, true, 512, 32>(tolerance);
    ExpectFusedSpecializationMatches<T, false, true, 128, 16>(tolerance);
    ExpectFusedSpecializationMatches<T, false, true, 256, 16>(tolerance);
    ExpectFusedSpecializationMatches<T, true, false, 512, 8>(tolerance);
}

TEST(FactoredKernels, SpecializedSizesMatchRuntimeSizes_Float)
{
    ExpectSpecializationsMatch<float>(1e-6);
}

TEST(FactoredKernels, SpecializedSizesMatchRuntimeSizes_Double)
{
    ExpectSpecializationsMatch<double>(1e-13);
}

TEST(FactoredKernels, N1OfFollowsSplitLength)
{
    for (size_t n = 4; n <= 4096; n *= 2) {
        EXPECT_EQ(static_cast<size_t>(FactoredN1Of(static_cast<int>(n))), internal::SplitLength(n).n1) << "n " << n;
    }
}
