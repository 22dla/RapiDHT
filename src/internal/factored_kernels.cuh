/*
 * Project: RapiDHT
 * File: src/internal/factored_kernels.cuh
 * Brief: R-4 device code -- the factored axis kernels and the in-place
 *        Bracewell kernels.
 *
 * Kept apart from kernels.cu so that it can be compiled twice: by nvcc, as
 * part of kernels.cu, and by the host compiler in tests/test_factored_kernels.cpp,
 * which runs every kernel with one thread per block in turn. With one thread,
 * __syncthreads() is a no-op and the block-stride loops cover everything, so
 * the very code that runs on the device is checked against the definition on
 * any machine, GPU or not.
 *
 * Include it inside nothing: it opens namespace RapiDHT itself.
 */

#ifndef RAPIDHT_INTERNAL_FACTORED_KERNELS_CUH
#define RAPIDHT_INTERNAL_FACTORED_KERNELS_CUH

#include "internal/kernels.h"

#include <cstddef>

// `RAPIDHT_UNROLL` for nvcc; nothing for the host compiler of the tests, which
// would warn about an unknown pragma.
#if defined(__CUDACC__)
#define RAPIDHT_UNROLL _Pragma("unroll")
#else
#define RAPIDHT_UNROLL
#endif

namespace RapiDHT {

/*
 * R-4: the factored DHT of a block of lines, entirely in shared memory.
 *
 * Mirrors FactoredDhtLanes in factored.cpp step for step -- the tests pin that
 * one to the definition -- with the loops over (row, lane) spread across the
 * threads of the block. Element idx of lane l lives at [idx * lanes + l].
 *
 * The block reads all of its lines before writing any, and no two blocks share
 * a line, so the transform is in place in global memory.
 *
 * Two kernels share the load, twiddle and store phases below. The generic one
 * computes one output per thread per pass: two loads (tile and table) per
 * multiply-add, which on Ampere, where loads issue at a quarter of the FMA
 * rate, held the first version to 0.78 TFLOP/s. The fast one gives every
 * thread a 4 x 4 tile of outputs -- four frequencies of four lines -- and
 * reads both operands as 4-vectors, two loads per sixteen multiply-adds.
 */

/// Loads the block's lines into the tile. Along the fastest axis a line is
/// contiguous, so consecutive threads walk along it; along the others they
/// walk across neighbouring lines, which are the contiguous direction there.
///
/// Each thread first issues kLoadBatch global loads into registers and only
/// then stores them to shared memory. A plain block-stride loop alternates
/// load and store, so every iteration waits out a full global-memory latency
/// before the next load is even issued; with two blocks to an SM nothing hides
/// that, and the first register-tiled version spent most of each axis there
/// (all three axes took the same ~9.5 ms at 512^3 f32, 3.5x the bandwidth
/// bound).
template <typename T>
__device__ void FactoredLoad(const T* base, T* tile, int n, int lanes, size_t axisStride, size_t laneStride)
{
    constexpr int kLoadBatch = 8;
    const int total = n * lanes;
    const int step = static_cast<int>(blockDim.x);
    for (int e0 = static_cast<int>(threadIdx.x); e0 < total; e0 += kLoadBatch * step) {
        T v[kLoadBatch] = {};
        RAPIDHT_UNROLL
        for (int b = 0; b < kLoadBatch; ++b) {
            const int e = e0 + b * step;
            if (e < total) {
                if (axisStride == 1) {
                    v[b] = base[e % n + static_cast<size_t>(e / n) * laneStride];
                } else {
                    v[b] = base[static_cast<size_t>(e / lanes) * axisStride + static_cast<size_t>(e % lanes) * laneStride];
                }
            }
        }
        RAPIDHT_UNROLL
        for (int b = 0; b < kLoadBatch; ++b) {
            const int e = e0 + b * step;
            if (e < total) {
                if (axisStride == 1) {
                    tile[(e % n) * lanes + e / n] = v[b];
                } else {
                    tile[e] = v[b];
                }
            }
        }
    }
}

/// Twiddles the pairs (k1, N1 - k1), 0 < k1 < N1/2, in place in work, whose
/// column k1 starts at k1 * workStride.
template <typename T>
__device__ void FactoredTwiddle(T* work, const FactoredAxisTables<T>& t, int lanes, int workStride)
{
    const int n1 = t.n1;
    const int n2 = t.n2;
    const int pairCount = (n1 / 2 > 1) ? (n1 / 2 - 1) : 0;
    for (int e = threadIdx.x; e < pairCount * n2 * lanes; e += blockDim.x) {
        const int lane = e % lanes;
        const int r = e / lanes;
        const int k1 = 1 + r / n2;
        const int j2 = r % n2;
        const int m1 = n1 - k1;
        const T c = t.twC[k1 * n2 + j2];
        const T d = t.twD[k1 * n2 + j2];
        T& p = work[k1 * workStride + j2 * lanes + lane];
        T& q = work[m1 * workStride + j2 * lanes + lane];
        const T yp = p;
        const T ym = q;
        p = d * yp + c * ym;
        q = c * yp - d * ym;
    }
}

/// Recombines the column transforms in tile into output order k = k1 + N1*k2
/// and stores them over the block's lines.
template <typename T>
__device__ void FactoredStore(
    const T* tile, T* base, int n, int n1, int n2, int lanes, size_t axisStride, size_t laneStride)
{
    auto value = [&](int k, int lane) -> T {
        const int k1 = k % n1;
        const int k2 = k / n1;
        if (k1 == 0 || 2 * k1 == n1) {
            return tile[(k1 * n2 + k2) * lanes + lane];
        }
        if (2 * k1 < n1) {
            const int mk2 = (n2 - k2) % n2;
            return tile[(k1 * n2 + k2) * lanes + lane] + tile[((n1 - k1) * n2 + mk2) * lanes + lane];
        }
        // k1 > N1/2: the mirror half of the pair j = N1 - k1, with k2' = N2-1-k2.
        const int j = n1 - k1;
        const int kb = n2 - 1 - k2;
        const int mkb = (n2 - kb) % n2;
        return tile[(j * n2 + mkb) * lanes + lane] - tile[(k1 * n2 + kb) * lanes + lane];
    };
    const int total = n * lanes;
    if (axisStride == 1) {
        for (int e = threadIdx.x; e < total; e += blockDim.x) {
            const int lane = e / n;
            const int k = e % n;
            base[k + static_cast<size_t>(lane) * laneStride] = value(k, lane);
        }
    } else {
        for (int e = threadIdx.x; e < total; e += blockDim.x) {
            const int k = e / lanes;
            const int lane = e % lanes;
            base[static_cast<size_t>(k) * axisStride + static_cast<size_t>(lane) * laneStride] = value(k, lane);
        }
    }
}

template <typename T>
__global__ void FactoredAxisKernel(T* data, FactoredAxisTables<T> t, size_t axisStride, size_t laneStride,
    int lanes, int blocksPerOther, size_t otherStride)
{
    extern __shared__ unsigned char sharedRaw[];
    T* tile = reinterpret_cast<T*>(sharedRaw);
    const int n = t.n;
    const int n1 = t.n1;
    const int n2 = t.n2;
    const int total = n * lanes;
    T* work = tile + total;

    const int other = blockIdx.x / blocksPerOther;
    const int firstLane = (blockIdx.x % blocksPerOther) * lanes;
    T* base = data + static_cast<size_t>(other) * otherStride + static_cast<size_t>(firstLane) * laneStride;

    FactoredLoad(base, tile, n, lanes, axisStride, laneStride);
    __syncthreads();

    // Step 1: Y[k1][j2] = sum_j1 C1[k1][j1] x[N2*j1 + j2].
    for (int e = threadIdx.x; e < total; e += blockDim.x) {
        const int lane = e % lanes;
        const int row = e / lanes;
        const int k1 = row / n2;
        const int j2 = row % n2;
        const T* c = t.c1 + k1 * n1;
        T sum = 0;
        for (int j1 = 0; j1 < n1; ++j1) {
            sum += c[j1] * tile[(n2 * j1 + j2) * lanes + lane];
        }
        work[e] = sum;
    }
    __syncthreads();

    FactoredTwiddle(work, t, lanes, n2 * lanes);
    __syncthreads();

    // Step 2: a DHT of length N2 down every column; half-shifted for N1/2.
    for (int e = threadIdx.x; e < total; e += blockDim.x) {
        const int lane = e % lanes;
        const int row = e / lanes;
        const int k1 = row / n2;
        const int k2 = row % n2;
        const T* m = ((n1 >= 2 && k1 == n1 / 2) ? t.cHalf : t.c2) + k2 * n2;
        T sum = 0;
        for (int j2 = 0; j2 < n2; ++j2) {
            sum += m[j2] * work[(k1 * n2 + j2) * lanes + lane];
        }
        tile[e] = sum;
    }
    __syncthreads();

    FactoredStore(tile, base, n, n1, n2, lanes, axisStride, laneStride);
}

/// Four consecutive values as one or two vector loads; p is 16-byte aligned.
__device__ __forceinline__ void Load4(const float* p, float (&out)[4])
{
    const float4 q = *reinterpret_cast<const float4*>(p);
    out[0] = q.x;
    out[1] = q.y;
    out[2] = q.z;
    out[3] = q.w;
}

__device__ __forceinline__ void Load4(const double* p, double (&out)[4])
{
    const double2 a = *reinterpret_cast<const double2*>(p);
    const double2 b = *reinterpret_cast<const double2*>(p + 2);
    out[0] = a.x;
    out[1] = a.y;
    out[2] = b.x;
    out[3] = b.y;
}

__device__ __forceinline__ void Store4(float* p, const float (&in)[4])
{
    *reinterpret_cast<float4*>(p) = make_float4(in[0], in[1], in[2], in[3]);
}

__device__ __forceinline__ void Store4(double* p, const double (&in)[4])
{
    *reinterpret_cast<double2*>(p) = make_double2(in[0], in[1]);
    *reinterpret_cast<double2*>(p + 2) = make_double2(in[2], in[3]);
}

/*
 * The register-tiled kernel. Requires n1, n2 and lanes to be multiples of 4.
 * Shared memory: tile | work | C1 | C2 | C_half transposed, every part a
 * multiple of 16 bytes so that each starts 16-byte aligned. Column k1 of work
 * starts at k1 * workStride, which the launcher pads past n2 * lanes: in step
 * 2 a warp reads several columns at once, and with an unpadded stride that is
 * a multiple of 32 words they all fall into the same banks. C1 and C2 are
 * symmetric, so C[k][j] read as [j][k] puts four frequencies side by side;
 * C_half is not, hence the transposed copy.
 */
template <typename T>
__global__ void FactoredAxisKernelFast(T* data, FactoredAxisTables<T> t, size_t axisStride, size_t laneStride,
    int lanes, int blocksPerOther, size_t otherStride, int workStride)
{
    extern __shared__ __align__(16) unsigned char sharedFast[];
    const int n = t.n;
    const int n1 = t.n1;
    const int n2 = t.n2;
    const int total = n * lanes;
    T* tile = reinterpret_cast<T*>(sharedFast);
    T* work = tile + total;
    T* c1s = work + n1 * workStride;
    T* c2s = c1s + n1 * n1;
    T* chT = c2s + n2 * n2;

    for (int i = threadIdx.x; i < n1 * n1; i += blockDim.x) {
        c1s[i] = t.c1[i];
    }
    for (int i = threadIdx.x; i < n2 * n2; i += blockDim.x) {
        c2s[i] = t.c2[i];
        chT[(i % n2) * n2 + i / n2] = t.cHalf[i];
    }

    const int other = blockIdx.x / blocksPerOther;
    const int firstLane = (blockIdx.x % blocksPerOther) * lanes;
    T* base = data + static_cast<size_t>(other) * otherStride + static_cast<size_t>(firstLane) * laneStride;

    FactoredLoad(base, tile, n, lanes, axisStride, laneStride);
    __syncthreads();

    const int laneGroups = lanes / 4;

    // Step 1, 4 frequencies k1 x 4 lines per thread.
    const int items1 = (n1 / 4) * n2 * laneGroups;
    for (int it = threadIdx.x; it < items1; it += blockDim.x) {
        const int lg = it % laneGroups;
        const int r = it / laneGroups;
        const int j2 = r % n2;
        const int kg = r / n2;
        T acc[4][4] = {};
        for (int j1 = 0; j1 < n1; ++j1) {
            T x[4];
            T c[4];
            Load4(tile + (n2 * j1 + j2) * lanes + lg * 4, x);
            Load4(c1s + j1 * n1 + kg * 4, c);
            RAPIDHT_UNROLL
            for (int a = 0; a < 4; ++a) {
                RAPIDHT_UNROLL
                for (int b = 0; b < 4; ++b) {
                    acc[a][b] += c[a] * x[b];
                }
            }
        }
        RAPIDHT_UNROLL
        for (int a = 0; a < 4; ++a) {
            Store4(work + (kg * 4 + a) * workStride + j2 * lanes + lg * 4, acc[a]);
        }
    }
    __syncthreads();

    FactoredTwiddle(work, t, lanes, workStride);
    __syncthreads();

    // Step 2, 4 frequencies k2 x 4 lines per thread.
    const int k2Groups = n2 / 4;
    const int items2 = n1 * k2Groups * laneGroups;
    for (int it = threadIdx.x; it < items2; it += blockDim.x) {
        const int lg = it % laneGroups;
        const int r = it / laneGroups;
        const int kg = r % k2Groups;
        const int k1 = r / k2Groups;
        const T* m = (k1 == n1 / 2) ? chT : c2s; // read as [j2][k2]
        T acc[4][4] = {};
        for (int j2 = 0; j2 < n2; ++j2) {
            T w[4];
            T c[4];
            Load4(work + k1 * workStride + j2 * lanes + lg * 4, w);
            Load4(m + j2 * n2 + kg * 4, c);
            RAPIDHT_UNROLL
            for (int a = 0; a < 4; ++a) {
                RAPIDHT_UNROLL
                for (int b = 0; b < 4; ++b) {
                    acc[a][b] += c[a] * w[b];
                }
            }
        }
        RAPIDHT_UNROLL
        for (int a = 0; a < 4; ++a) {
            Store4(tile + (k1 * n2 + kg * 4 + a) * lanes + lg * 4, acc[a]);
        }
    }
    __syncthreads();

    FactoredStore(tile, base, n, n1, n2, lanes, axisStride, laneStride);
}

/*
 * v2.3: the register-tiled kernel as a pipeline over many blocks of lines.
 *
 * In v2.2 every block did load -> compute -> store once and exited, and with
 * two blocks to an SM the load of one block could hide behind the arithmetic
 * of the other only by chance: all three axes stayed at ~7 ms against a
 * bandwidth bound of ~2.4 ms at 512^3 f32, with the bus and the FMA units both
 * mostly idle. Here the grid is only as large as fits on the device at once,
 * each block walks the batches blockIdx.x, blockIdx.x + gridDim.x, ..., and the
 * global loads of the next batch are issued into registers right after the
 * current one reaches shared memory -- they are in flight during steps 1 and
 * 2 and are waited for only at the top of the next iteration. No extra shared
 * memory, so occupancy stays where it was; the cost is 4 * kMaxQuads registers
 * a thread. No other block ever touches a batch, so prefetching it before
 * this block has written the previous one keeps the transform in place.
 *
 * Every shared-memory access is also laid out against bank conflicts, the
 * fast axis included (the x pass of v2.2 was 1.7 ms slower than y and z):
 *  - a thread moves a quad -- 4 neighbouring lines at one position -- and
 *    consecutive threads take consecutive quads, so the tile is written and
 *    read as one contiguous run of 16-byte vectors on every axis;
 *  - step 2 runs k1 fastest across threads, so the padded work columns spread
 *    over the banks, and writes its result in output order k = k1 + N1*k2,
 *    which makes the final store read the tile contiguously (the v2.2 layout,
 *    row k1*N2 + k2, put consecutive outputs 512 bytes apart: 4-way conflicts).
 * Along the lane axis a quad is contiguous in global memory when laneStride
 * is 1 (axes y and z), and then it moves as one vector.
 */

/// Four values p[0], p[s], p[2s], p[3s]; one vector access when vec (then s is
/// 1 and p is 16-byte aligned).
template <typename T>
__device__ __forceinline__ void LoadQuad(const T* p, size_t s, bool vec, T (&out)[4])
{
    if (vec) {
        Load4(p, out);
    } else {
        RAPIDHT_UNROLL
        for (int j = 0; j < 4; ++j) {
            out[j] = p[static_cast<size_t>(j) * s];
        }
    }
}

template <typename T>
__device__ __forceinline__ void StoreQuad(T* p, size_t s, bool vec, const T (&in)[4])
{
    if (vec) {
        Store4(p, in);
    } else {
        RAPIDHT_UNROLL
        for (int j = 0; j < 4; ++j) {
            p[static_cast<size_t>(j) * s] = in[j];
        }
    }
}

/// Issues this thread's global loads of a batch: quad q = threadIdx.x +
/// i * blockDim.x is position q / (lanes/4), lines 4 * (q % (lanes/4)) + 0..3.
template <typename T, int kMaxQuads>
__device__ __forceinline__ void PipelineLoad(
    const T* base, T (&reg)[kMaxQuads][4], int quads, int quadsPerRow, size_t axisStride, size_t laneStride, bool vec)
{
    RAPIDHT_UNROLL
    for (int i = 0; i < kMaxQuads; ++i) {
        const int q = static_cast<int>(threadIdx.x) + i * static_cast<int>(blockDim.x);
        if (q < quads) {
            const size_t row = static_cast<size_t>(q / quadsPerRow);
            const size_t line = static_cast<size_t>(q % quadsPerRow) * 4;
            LoadQuad(base + row * axisStride + line * laneStride, laneStride, vec, reg[i]);
        }
    }
}

/// Puts the loaded quads into the tile; quad q lands at tile + 4q.
template <typename T, int kMaxQuads>
__device__ __forceinline__ void PipelineFill(T* tile, const T (&reg)[kMaxQuads][4], int quads)
{
    RAPIDHT_UNROLL
    for (int i = 0; i < kMaxQuads; ++i) {
        const int q = static_cast<int>(threadIdx.x) + i * static_cast<int>(blockDim.x);
        if (q < quads) {
            Store4(tile + 4 * q, reg[i]);
        }
    }
}

/*
 * Diagnostic modes of the pipelined kernel (R-4 step 1.1, docs/R4_PLAN.md),
 * chosen by RAPIDHT_R4_ABLATE. Both give a WRONG result and exist only to time
 * the two halves of the kernel apart:
 *  - kAblateIo: every global load and store as usual, steps 1-2 and the
 *    recombination skipped; each line is written back unchanged, so the pass
 *    is the identity (the host test checks exactly that);
 *  - kAblateCompute: the first batch is fetched once and computed over and
 *    over without new loads; only that first result is stored, so that the
 *    compiler cannot drop the arithmetic.
 */
constexpr int kAblateNone = 0;
constexpr int kAblateIo = 1;
constexpr int kAblateCompute = 2;

/*
 * Requires n1, n2 and lanes to be multiples of 4 and blockDim.x * kMaxQuads
 * >= n * lanes / 4. Shared memory exactly as FactoredAxisKernelFast.
 */
template <typename T, int kMaxQuads>
__global__ void FactoredAxisKernelPipelined(T* data, FactoredAxisTables<T> t, size_t axisStride, size_t laneStride,
    int lanes, int blocksPerOther, size_t otherStride, int batches, int workStride, bool vec, int ablate)
{
    extern __shared__ __align__(16) unsigned char sharedPipe[];
    const int n = t.n;
    const int n1 = t.n1;
    const int n2 = t.n2;
    T* tile = reinterpret_cast<T*>(sharedPipe);
    T* work = tile + n * lanes;
    T* c1s = work + n1 * workStride;
    T* c2s = c1s + n1 * n1;
    T* chT = c2s + n2 * n2;

    for (int i = threadIdx.x; i < n1 * n1; i += blockDim.x) {
        c1s[i] = t.c1[i];
    }
    for (int i = threadIdx.x; i < n2 * n2; i += blockDim.x) {
        c2s[i] = t.c2[i];
        chT[(i % n2) * n2 + i / n2] = t.cHalf[i];
    }

    const int quadsPerRow = lanes / 4;
    const int quads = n * quadsPerRow;
    const int laneGroups = lanes / 4;
    auto batchBase = [&](int batch) -> T* {
        const int other = batch / blocksPerOther;
        const int firstLane = (batch % blocksPerOther) * lanes;
        return data + static_cast<size_t>(other) * otherStride + static_cast<size_t>(firstLane) * laneStride;
    };

    T reg[kMaxQuads][4];
    int batch = static_cast<int>(blockIdx.x);
    if (batch < batches) {
        PipelineLoad(batchBase(batch), reg, quads, quadsPerRow, axisStride, laneStride, vec);
    }
    for (; batch < batches; batch += static_cast<int>(gridDim.x)) {
        T* base = batchBase(batch);
        __syncthreads(); // the previous batch's store has finished reading the tile
        PipelineFill(tile, reg, quads);
        __syncthreads();

        // The next batch is fetched while this one is computed.
        const int next = batch + static_cast<int>(gridDim.x);
        if (next < batches && ablate != kAblateCompute) {
            PipelineLoad(batchBase(next), reg, quads, quadsPerRow, axisStride, laneStride, vec);
        }

        if (ablate != kAblateIo) { // uniform across the block, so the barriers inside are safe
            // Step 1, 4 frequencies k1 x 4 lines per thread, lines fastest.
            const int items1 = (n1 / 4) * n2 * laneGroups;
            for (int it = threadIdx.x; it < items1; it += blockDim.x) {
                const int lg = it % laneGroups;
                const int r = it / laneGroups;
                const int j2 = r % n2;
                const int kg = r / n2;
                T acc[4][4] = {};
                for (int j1 = 0; j1 < n1; ++j1) {
                    T x[4];
                    T c[4];
                    Load4(tile + (n2 * j1 + j2) * lanes + lg * 4, x);
                    Load4(c1s + j1 * n1 + kg * 4, c);
                    RAPIDHT_UNROLL
                    for (int a = 0; a < 4; ++a) {
                        RAPIDHT_UNROLL
                        for (int b = 0; b < 4; ++b) {
                            acc[a][b] += c[a] * x[b];
                        }
                    }
                }
                RAPIDHT_UNROLL
                for (int a = 0; a < 4; ++a) {
                    Store4(work + (kg * 4 + a) * workStride + j2 * lanes + lg * 4, acc[a]);
                }
            }
            __syncthreads();

            FactoredTwiddle(work, t, lanes, workStride);
            __syncthreads();

            // Step 2, 4 frequencies k2 x 4 lines per thread, k1 fastest after the
            // lines; the result goes to row k1 + N1*k2 of the tile.
            const int k2Groups = n2 / 4;
            const int items2 = n1 * k2Groups * laneGroups;
            for (int it = threadIdx.x; it < items2; it += blockDim.x) {
                const int lg = it % laneGroups;
                const int r = it / laneGroups;
                const int k1 = r % n1;
                const int kg = r / n1;
                const T* m = (k1 == n1 / 2) ? chT : c2s; // read as [j2][k2]
                T acc[4][4] = {};
                for (int j2 = 0; j2 < n2; ++j2) {
                    T w[4];
                    T c[4];
                    Load4(work + k1 * workStride + j2 * lanes + lg * 4, w);
                    Load4(m + j2 * n2 + kg * 4, c);
                    RAPIDHT_UNROLL
                    for (int a = 0; a < 4; ++a) {
                        RAPIDHT_UNROLL
                        for (int b = 0; b < 4; ++b) {
                            acc[a][b] += c[a] * w[b];
                        }
                    }
                }
                RAPIDHT_UNROLL
                for (int a = 0; a < 4; ++a) {
                    Store4(tile + (k1 + n1 * (kg * 4 + a)) * lanes + lg * 4, acc[a]);
                }
            }
            __syncthreads();
        }

        // Recombination and store, quad by quad. Same formulas as
        // FactoredStore, with (k1, k2) at row k1 + N1*k2.
        for (int q = threadIdx.x; q < quads; q += blockDim.x) {
            const int k = q / quadsPerRow;
            const int line = (q % quadsPerRow) * 4;
            const int k1 = k % n1;
            const int k2 = k / n1;
            T out[4];
            if (ablate == kAblateIo || k1 == 0 || 2 * k1 == n1) {
                Load4(tile + k * lanes + line, out);
            } else {
                T a[4];
                T b[4];
                if (2 * k1 < n1) {
                    const int mk2 = (n2 - k2) % n2;
                    Load4(tile + k * lanes + line, a);
                    Load4(tile + ((n1 - k1) + n1 * mk2) * lanes + line, b);
                    RAPIDHT_UNROLL
                    for (int j = 0; j < 4; ++j) {
                        out[j] = a[j] + b[j];
                    }
                } else {
                    // k1 > N1/2: the mirror half of the pair j = N1 - k1, k2' = N2-1-k2.
                    const int jm = n1 - k1;
                    const int kb = n2 - 1 - k2;
                    const int mkb = (n2 - kb) % n2;
                    Load4(tile + (jm + n1 * mkb) * lanes + line, a);
                    Load4(tile + (k1 + n1 * kb) * lanes + line, b);
                    RAPIDHT_UNROLL
                    for (int j = 0; j < 4; ++j) {
                        out[j] = a[j] - b[j];
                    }
                }
            }
            if (ablate != kAblateCompute || batch == static_cast<int>(blockIdx.x)) {
                StoreQuad(base + static_cast<size_t>(k) * axisStride + static_cast<size_t>(line) * laneStride,
                    laneStride, vec, out);
            }
        }
    }
}

/// Mirror of an index along an axis of length n: (n - i) mod n.
__device__ __forceinline__ int MirrorIndex(int i, int n)
{
    return i == 0 ? 0 : n - i;
}

// One thread per orbit representative (y in [0, W/2], x in [0, H/2]): loads
// the four mirror images, then stores the four outputs. Same formulas as
// BracewellTransform2DCPU.
template <typename T>
__global__ void BracewellInPlace2DKernel(T* image, int W, int H)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    if (y > W / 2 || x > H / 2) {
        return;
    }
    const int ym = MirrorIndex(y, W);
    const int xm = MirrorIndex(x, H);
    T* r0 = image + static_cast<size_t>(W) * x;
    T* r1 = image + static_cast<size_t>(W) * xm;
    const T v00 = r0[y], v10 = r0[ym], v01 = r1[y], v11 = r1[ym];
    const T half = static_cast<T>(0.5);
    r0[y] = (v00 + v01 + v10 - v11) * half;
    r0[ym] = (v10 + v11 + v00 - v01) * half;
    r1[y] = (v01 + v00 + v11 - v10) * half;
    r1[ym] = (v11 + v10 + v01 - v00) * half;
}

// The 3D orbit of up to 8 points, as BracewellTransform3DCPU.
template <typename T>
__global__ void BracewellInPlace3DKernel(T* volume, int W, int H, int D)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (y > W / 2 || x > H / 2 || z > D / 2) {
        return;
    }
    const int ym = MirrorIndex(y, W);
    const int xm = MirrorIndex(x, H);
    const int zm = MirrorIndex(z, D);
    T* r00 = volume + static_cast<size_t>(W) * (x + static_cast<size_t>(H) * z);
    T* r10 = volume + static_cast<size_t>(W) * (xm + static_cast<size_t>(H) * z);
    T* r01 = volume + static_cast<size_t>(W) * (x + static_cast<size_t>(H) * zm);
    T* r11 = volume + static_cast<size_t>(W) * (xm + static_cast<size_t>(H) * zm);

    const T v000 = r00[y], v100 = r00[ym];
    const T v010 = r10[y], v110 = r10[ym];
    const T v001 = r01[y], v101 = r01[ym];
    const T v011 = r11[y], v111 = r11[ym];
    const T half = static_cast<T>(0.5);
    r00[y] = (v100 + v010 + v001 - v111) * half;
    r00[ym] = (v000 + v110 + v101 - v011) * half;
    r10[y] = (v110 + v000 + v011 - v101) * half;
    r10[ym] = (v010 + v100 + v111 - v001) * half;
    r01[y] = (v101 + v011 + v000 - v110) * half;
    r01[ym] = (v001 + v111 + v100 - v010) * half;
    r11[y] = (v111 + v001 + v010 - v100) * half;
    r11[ym] = (v011 + v101 + v110 - v000) * half;
}

} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_FACTORED_KERNELS_CUH
