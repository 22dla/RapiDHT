/*
 * Project: RapiDHT
 * File: src/kernels.cu
 * Brief: The CUDA kernels for the matrix operations and the Hartley matrices.
 * Author: Volkov Evgeny Aleksandrovich, volkov22dla@yandex.ru
 */

#include "internal/device_array.h"
#include "internal/kernels.h"

// Outside namespace RapiDHT, which the header opens itself.
#include "internal/factored_kernels.cuh"

#include "device_launch_parameters.h"

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

// ------------------------------ Kernels ------------------------------

namespace RapiDHT {

template <typename T>
__global__ void MatrixMultiplicationSharedKernel(const T* __restrict__ A, const T* __restrict__ B,
    T* __restrict__ C, int M, int K, int N)
{
    const int BLOCK_SIZE = 16;
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    T sum = 0.0;

    // Tile by tile over A and B.
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Stage both tiles in shared memory.
        if (row < M && t * BLOCK_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && t * BLOCK_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // Multiply the staged tiles.
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the accumulated result out.
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

template <typename T>
__global__ void VectorMatrixMultiplicationKernel(const T* A, const T* x, T* y, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        T sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

template <typename T>
__global__ void MatrixTransposeKernel(const T* A, T* B, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows) {
        B[col * rows + row] = A[row * cols + col];
    }
}

/// Multiplies every element by a constant, for the 1/N of an inverse transform
/// applied to data that stays on the device.
template <typename T>
__global__ void ScaleKernel(T* data, size_t count, T factor)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        data[i] *= factor;
    }
}

/*
 * Bracewell correction, turning the separable result of the per-axis passes
 * into the true multidimensional Hartley transform.
 *
 * Out of place on purpose: every output reads four (2D) or four (3D) input
 * points, including mirrored ones, so writing into the source buffer races
 * with neighbouring threads that still need the original values.
 *
 * The formulas match BracewellTransform2DCPU and BracewellTransform3DCPU
 * exactly; the tests compare both backends against the same reference.
 */
template <typename T>
__global__ void BracewellTransform2DKernel(const T* __restrict__ in, T* __restrict__ out, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) {
        return;
    }

    const int xm = (x == 0) ? 0 : W - x;
    const int ym = (y == 0) ? 0 : H - y;

    const T a = in[y * W + x];
    const T b = in[y * W + xm]; // mirrored in X
    const T c = in[ym * W + x]; // mirrored in Y
    const T d = in[ym * W + xm]; // mirrored in both

    out[y * W + x] = static_cast<T>(0.5) * (a + b + c - d);
}

template <typename T>
__global__ void BracewellTransform3DKernel(const T* __restrict__ in, T* __restrict__ out, int W, int H, int D)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= W || y >= H || z >= D) {
        return;
    }

    const int xm = (x == 0) ? 0 : W - x;
    const int ym = (y == 0) ? 0 : H - y;
    const int zm = (z == 0) ? 0 : D - z;

    const size_t plane = static_cast<size_t>(H) * W;
    const T a = in[zm * plane + y * W + x]; // mirrored in Z
    const T b = in[z * plane + ym * W + x]; // mirrored in Y
    const T c = in[z * plane + y * W + xm]; // mirrored in X
    const T d = in[zm * plane + ym * W + xm]; // mirrored in all three

    out[z * plane + y * W + x] = static_cast<T>(0.5) * (a + b + c - d);
}

__global__ void InitializeHartleyMatrixKernel(double* kernel, size_t height)
{
    size_t k = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < height && j < height) {
        const double kPi = 3.14159265358979323846;
        kernel[k * height + j] = cos(2.0 * kPi * k * j / height) + sin(2.0 * kPi * k * j / height);
    }
}

__global__ void InitializeHartleyMatrixKernel(float* kernel, size_t height)
{
    size_t k = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < height && j < height) {
        const float kPi = 3.14159265358979323846f;
        kernel[k * height + j] = cosf(2.0f * kPi * k * j / height) + sinf(2.0f * kPi * k * j / height);
    }
}

// ------------------------------ Host Wrappers ------------------------------

template <typename T>
void MatrixMultiplication(const T* A, const T* B, T* C, int M, int K, int N)
{
    const int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    MatrixMultiplicationSharedKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
    RAPIDHT_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
void VectorMatrixMultiplication(const T* A, const T* x, T* y, int N)
{
    int threadsPerBlock = (N > 512) ? 512 : N;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    VectorMatrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, x, y, N);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
    RAPIDHT_CUDA_CHECK(cudaDeviceSynchronize());
}

// rows and cols describe the destination, that is the shape of B.
template <typename T>
void MatrixTranspose(const T* A, T* B, int rows, int cols)
{
    int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    MatrixTransposeKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, rows, cols);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
    RAPIDHT_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
void ScaleOnDevice(T* deviceData, size_t count, T factor)
{
    const int threads = 256;
    const size_t blocks = (count + threads - 1) / threads;

    ScaleKernel<<<static_cast<unsigned int>(blocks), threads>>>(deviceData, count, factor);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
    RAPIDHT_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
void BracewellTransform2D(const T* deviceIn, T* deviceOut, int W, int H)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
        (H + blockDim.y - 1) / blockDim.y);

    BracewellTransform2DKernel<<<gridDim, blockDim>>>(deviceIn, deviceOut, W, H);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
    RAPIDHT_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
void BracewellTransform3D(const T* deviceIn, T* deviceOut, int W, int H, int D)
{
    // A full warp along x, so every one of the four reads is a 128-byte row.
    // The 8 x 8 x 8 block this replaces gave each warp four 32-byte pieces
    // and took 6.5 ms at 512^3 in f32, against 4.9 ms for this shape.
    dim3 blockDim(32, 8, 1);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
        (H + blockDim.y - 1) / blockDim.y,
        (D + blockDim.z - 1) / blockDim.z);

    BracewellTransform3DKernel<<<gridDim, blockDim>>>(deviceIn, deviceOut, W, H, D);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
    RAPIDHT_CUDA_CHECK(cudaDeviceSynchronize());
}

void InitializeHartleyMatrix(double* deviceMatrix, size_t height)
{
    // An absent axis (height 0, in 1D and 2D) has no matrix. Launching anyway
    // meant a 0 x 0 grid, an invalid configuration that nothing checked: the
    // error stayed pending and surfaced at the next cudaGetLastError, in
    // whichever unrelated kernel happened to ask (found through R-4).
    if (height == 0) {
        return;
    }
    dim3 block(16, 16);
    dim3 grid((height + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    InitializeHartleyMatrixKernel<<<grid, block>>>(deviceMatrix, height);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
    RAPIDHT_CUDA_CHECK(cudaDeviceSynchronize());
}

void InitializeHartleyMatrix(float* deviceMatrix, size_t height)
{
    // An absent axis (height 0, in 1D and 2D) has no matrix. Launching anyway
    // meant a 0 x 0 grid, an invalid configuration that nothing checked: the
    // error stayed pending and surfaced at the next cudaGetLastError, in
    // whichever unrelated kernel happened to ask (found through R-4).
    if (height == 0) {
        return;
    }
    dim3 block(16, 16);
    dim3 grid((height + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    InitializeHartleyMatrixKernel<<<grid, block>>>(deviceMatrix, height);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
    RAPIDHT_CUDA_CHECK(cudaDeviceSynchronize());
}

namespace {

// v2.3 pipelined kernel: threads per block and the most quads (4 values) a
// thread holds in registers for the next batch -- 8 covers every length up to
// 2048 at the widths the launcher picks.
constexpr int kFactoredThreads = 256;
constexpr int kFactoredMaxQuads = 8;

/*
 * Batch width of the pipelined kernel by axis and length, from the sweep of
 * R-4 step 2.4 (benchmarks/results/2026-11-15_*_r4_s24_L{4,8,16,32}_trace.txt,
 * 512^3 / 256^3 / 128^3 f32, sm_86). The best width differs by axis, not only
 * by length: along z (lines a plane apart) 16 lines win at every size, along y
 * (lines a row apart) 8 win from n = 256 and 16 lose 27 % there, along x
 * (contiguous lines, batches spread over rows) 16 win up to n = 256 and lose
 * 37 % at 512. 4 lines are always the slowest -- 3x at 512^3 on y and z.
 */
size_t FactoredPreferredLanes(size_t n, size_t axisStride, size_t laneLength)
{
    const bool alongX = axisStride == 1;
    const bool alongY = !alongX && axisStride == laneLength;
    if (alongX) {
        // The separable pass along x takes 16 lines at every length: at 512
        // that is the two-tile kernel and 3 % on the separable transform
        // (step 4.X, 2026-11-20_*_r4_x512_*). The fused last pass keeps 8 at
        // 512 -- with 16 it took 4.22 ms against 3.62 (FactoredLastAxisBracewell).
        return 16;
    }
    // z at exactly 512 takes 32 lines: a row of a batch is then 128 contiguous
    // bytes instead of 64, and moving the line sets alone drops from 3.58 to
    // 2.61 ms -- the speed of a plain copy (benchmarks/stride_probe,
    // 2026-11-20_*_r4_stride_probe; two 64-byte rows in one page did not help,
    // so it is the contiguity, not address translation). The four-tile kernel
    // this needs overlaps its fetch with the steps worse, so the pass gains
    // less: 3.67 -> 3.43 ms with the fetch in halves (_r4_z32s), 3.24 in
    // quarters (_r4_s4z). y keeps 16: with 32 lines it lost as a pass both in
    // halves (2.84 -> 3.43) and in quarters (2.84 -> 3.11 at a locked clock;
    // the full transform at boost moved -1.4 %, within noise; _r4_y32q).
    // Not tried at 1024.
    if (n == 512 && !alongY) {
        return 32;
    }
    if (alongY) {
        // 16 again from n = 512 (step 4.Y, 2026-11-20_*_r4_y512_*): with the
        // two-tile in-place kernel of step 4B and the specialization of 4D, y
        // at 512^3 takes 2.83 ms against 4.51 with 8 lines -- the 2.4 sweep
        // that chose 8 ran the v2.3 kernel, through the work buffer.
        return (n <= 128 || n >= 512) ? 16 : 8;
    }
    return 16;
}

/*
 * What FactoredAxis asks the driver, asked once (R-4 step 2.3). The shared
 * memory limit, the SM count, the occupancy of a kernel at a given shared size
 * and the opt-in to more than 48 KiB do not change while the device stays the
 * same, but they were queried on every call of every axis -- four or five
 * driver calls in front of a kernel that runs 50 us at 128^3. One cache per
 * host thread, so no lock; keyed by device, so switching devices stays right.
 */
struct FactoredDeviceInfo {
    int device = -1;
    int optIn = 48 * 1024;
    int sms = 1;
    size_t fastSharedOptIn = 48 * 1024;
    /// Per pipelined kernel -- keyed by the kernel itself: since step 4D a
    /// variant has several instantiations, one per specialized size, and each
    /// has its own registers and so its own occupancy -- the largest dynamic
    /// shared size its attribute is set to, and shared bytes -> blocks per SM.
    struct KernelInfo {
        const void* kernel = nullptr;
        size_t sharedOptIn = 48 * 1024;
        std::vector<std::pair<size_t, int>> blocksPerSm;
    };
    std::vector<KernelInfo> kernels;
};

template <typename T>
FactoredDeviceInfo& FactoredInfo(int device)
{
    thread_local FactoredDeviceInfo info;
    if (info.device != device) {
        info = FactoredDeviceInfo {};
        info.device = device;
        RAPIDHT_CUDA_CHECK(cudaDeviceGetAttribute(&info.optIn, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        RAPIDHT_CUDA_CHECK(cudaDeviceGetAttribute(&info.sms, cudaDevAttrMultiProcessorCount, device));
    }
    return info;
}

/// Most 4 x 4 output tiles a thread holds in the in-place steps (kItems).
constexpr size_t kFactoredMaxItems = 4;

/// Prefetch registers (quads of values a thread) of a kernel variant: four
/// tiles a thread go with 32 lines at 512, n * lanes / 4 / 256 = 16 quads.
constexpr int FactoredMaxQuadsFor(int items)
{
    return items == 4 ? 16 : kFactoredMaxQuads;
}

/// Shared memory of the pipelined kernels. Through a work buffer (v2.3):
/// tile, work, C1, C2 and the transposed half-shift matrix. In place (4B):
/// the tile in the padded layout and the tables.
template <typename T>
size_t PipelinedSharedBytes(size_t n, size_t n1, size_t n2, size_t lanes, bool inPlace)
{
    return ((inPlace ? 0 : n * lanes) + n1 * FactoredWorkStride<T>(n2, lanes) + n1 * n1 + 2 * n2 * n2) * sizeof(T);
}

/// Whether a batch of `lanes` lines of length n fits the pipelined kernels
/// through a work buffer: the shared memory and the prefetch registers.
template <typename T>
bool PipelinedFits(size_t n, size_t n1, size_t n2, size_t lanes, int optIn)
{
    return lanes >= 4 && PipelinedSharedBytes<T>(n, n1, n2, lanes, false) <= static_cast<size_t>(optIn)
        && (n * lanes / 4 + kFactoredThreads - 1) / kFactoredThreads <= kFactoredMaxQuads;
}

/// Blocks of `kernel` resident on one SM at this shared size, asked once.
template <typename Kernel>
int FactoredBlocksPerSm(FactoredDeviceInfo& info, Kernel kernel, size_t sharedBytes)
{
    const void* key = reinterpret_cast<const void*>(kernel);
    FactoredDeviceInfo::KernelInfo* entry = nullptr;
    for (auto& known : info.kernels) {
        if (known.kernel == key) {
            entry = &known;
        }
    }
    if (entry == nullptr) {
        info.kernels.push_back({ key, 48 * 1024, {} });
        entry = &info.kernels.back();
    }
    if (sharedBytes > entry->sharedOptIn) {
        RAPIDHT_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
        entry->sharedOptIn = sharedBytes;
    }
    for (const auto& known : entry->blocksPerSm) {
        if (known.first == sharedBytes) {
            return known.second;
        }
    }
    int perSm = 0;
    RAPIDHT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kernel, kFactoredThreads, sharedBytes));
    perSm = perSm > 0 ? perSm : 1;
    entry->blocksPerSm.emplace_back(sharedBytes, perSm);
    return perSm;
}

/*
 * Which variant of the pipelined kernels runs a pass (R-4 step 4B): the steps
 * through a work buffer (items = 0, v2.3) or in place with `items` tiles a
 * thread, built for `minBlocks` blocks to an SM. No variant wins everywhere
 * (benchmarks/results/2026-11-15_*_r4_s4b*, f32, sm_86), so the choice is a
 * table, like the batch widths of step 2.4:
 *  - below n = 512 the work buffer, except z at 256: in place there frees the
 *    shared memory for a second block (0.47 -> 0.41 ms);
 *  - from n = 512 in place everywhere. z takes 16 lines, two tiles a thread
 *    (116 registers, 41 instead of 74 KiB, two blocks to an SM: 5.2 -> 4.3 ms);
 *    x, the fused pass, two blocks capped at 128 registers (3.96 -> 3.79);
 *    y with 8 lines one block with 140 registers (4.50 -> 3.96) -- capped to
 *    two blocks it takes 6.71 ms, with no spill (STACK 0), cause not
 *    measured. Since step 4.Y y takes 16 lines from n = 512 and so the
 *    two-tile kernel, like z (FactoredPreferredLanes).
 *  - at exactly 512, z takes 32 lines and four tiles a thread (one block to
 *    an SM, 16 prefetch quads, 212 registers): 128 contiguous bytes a row
 *    move at the speed of a copy (benchmarks/stride_probe).
 */
struct FactoredPath {
    int items;
    int minBlocks;
};

/*
 * Step 4A, the twiddle inside step 1 (FactoredStep1Order), by variant: two
 * A-B runs (benchmarks/results/2026-11-15_*_r4_s4a*, f32, sm_86) found it
 * wins on the fused pass through the work buffer (80 instead of 90 registers,
 * three blocks to an SM: x -17..-21 % at 128^3 and 256^3) and on the axis
 * kernel with two tiles a thread (z at 512^3, -7 %), but costs the axis kernel
 * through the work buffer 21 % (y at 256^3; ncu, flushing caches, showed the
 * opposite, cause not found) and the one-tile kernels in place 3 %.
 */
constexpr bool FactoredFoldsTwiddle(bool fused, int items)
{
    return fused ? items == 0 : items >= 2;
}

FactoredPath FactoredPathFor(size_t n, size_t lanes, int axis) // axis 0 = x, 1 = y, 2 = z
{
    const size_t items = (n * lanes / 16 + kFactoredThreads - 1) / kFactoredThreads;
    // Four tiles a thread only where they were measured: y and z at 512 with
    // 32 lines (FactoredPreferredLanes gives them to z). Elsewhere at most
    // two, as before.
    const size_t maxItems = (n == 512 && axis != 0) ? kFactoredMaxItems : 2;
    const bool inPlace = items <= maxItems && (n >= 512 || (axis == 2 && n == 256));
    if (!inPlace) {
        return { 0, 0 };
    }
    if (items == 4) {
        return { 4, 1 };
    }
    if (items == 2) {
        return { 2, 1 };
    }
    return axis == 1 ? FactoredPath { 1, 1 } : FactoredPath { 1, 2 };
}

/// PipelinedFits for the axis kernel, by the variant FactoredPathFor gives it:
/// in place it needs no work buffer, and four tiles a thread prefetch 16 quads.
template <typename T>
bool PipelinedAxisFits(size_t n, size_t n1, size_t n2, size_t lanes, int optIn, int axis)
{
    const FactoredPath path = FactoredPathFor(n, lanes, axis);
    if (path.items == 0) {
        return PipelinedFits<T>(n, n1, n2, lanes, optIn);
    }
    return lanes >= 4 && PipelinedSharedBytes<T>(n, n1, n2, lanes, true) <= static_cast<size_t>(optIn)
        && (n * lanes / 4 + kFactoredThreads - 1) / kFactoredThreads
               <= static_cast<size_t>(FactoredMaxQuadsFor(path.items));
}

/// R-5: the kernel a variant launches -- in place for float input, from
/// integers (the ...From kernels of step 2.2) otherwise.
template <typename T, typename TIn, int kItems, int kMinBlocks, bool kFold, int kN, int kLanes>
constexpr auto PipelinedAxisKernel()
{
    constexpr int kQuads = FactoredMaxQuadsFor(kItems);
    if constexpr (std::is_same<TIn, T>::value) {
        if constexpr (kItems == 0) {
            return FactoredAxisKernelPipelined<T, kFactoredMaxQuads, kFold, kN, kLanes>;
        } else {
            return FactoredAxisKernelInPlace<T, kQuads, kItems, kMinBlocks, kFold, kN, kLanes>;
        }
    } else {
        if constexpr (kItems == 0) {
            return FactoredAxisKernelPipelinedFrom<T, TIn, kFactoredMaxQuads, kFold, kN, kLanes>;
        } else {
            return FactoredAxisKernelInPlaceFrom<T, TIn, kQuads, kItems, kMinBlocks, kFold, kN, kLanes>;
        }
    }
}

template <typename T, typename TIn, int kItems, int kMinBlocks, int kN, int kLanes>
void LaunchPipelinedAxisAs(FactoredDeviceInfo& info, size_t batches, size_t sharedBytes, const TIn* in, T* data,
    const T* offset, const FactoredAxisTables<T>& tables, size_t axisStride, size_t laneStride, size_t lanes,
    size_t blocksPerOther, size_t otherStride, int workStride, bool vec)
{
    constexpr bool kFold = FactoredFoldsTwiddle(false, kItems);
    constexpr int kQuads = FactoredMaxQuadsFor(kItems);
    int perSm = FactoredBlocksPerSm(
        info, PipelinedAxisKernel<T, TIn, kItems, kMinBlocks, kFold, kN, kLanes>(), sharedBytes);
    const size_t resident = static_cast<size_t>(perSm) * static_cast<size_t>(info.sms);
    const unsigned int grid = static_cast<unsigned int>(batches < resident ? batches : resident);
    const int lanesI = static_cast<int>(lanes);
    const int perOther = static_cast<int>(blocksPerOther);
    const int batchesI = static_cast<int>(batches);
    if constexpr (!std::is_same<TIn, T>::value) {
        PipelinedAxisKernel<T, TIn, kItems, kMinBlocks, kFold, kN, kLanes>()<<<grid, kFactoredThreads, sharedBytes>>>(
            in, data, offset, tables, axisStride, laneStride, lanesI, perOther, otherStride, batchesI, workStride, vec);
    } else if constexpr (kItems == 0) {
        (void)in;
        (void)offset;
        FactoredAxisKernelPipelined<T, kFactoredMaxQuads, kFold, kN, kLanes><<<grid, kFactoredThreads, sharedBytes>>>(
            data, tables, axisStride, laneStride, lanesI, perOther, otherStride, batchesI, workStride, vec);
    } else {
        (void)in;
        (void)offset;
        FactoredAxisKernelInPlace<T, kQuads, kItems, kMinBlocks, kFold, kN, kLanes>
            <<<grid, kFactoredThreads, sharedBytes>>>(data, tables, axisStride, laneStride, lanesI, perOther,
                otherStride, batchesI, workStride, vec);
    }
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
}

template <typename T, int kItems, int kMinBlocks, int kN, int kLanes>
void LaunchFusedLastAs(FactoredDeviceInfo& info, size_t batches, size_t sharedBytes, T* data,
    const FactoredAxisTables<T>& tables, size_t H, size_t D, size_t groupsY, size_t groups, size_t lanes,
    int workStride)
{
    constexpr bool kFold = FactoredFoldsTwiddle(true, kItems);
    int perSm = 0;
    if constexpr (kItems == 0) {
        perSm = FactoredBlocksPerSm(
            info, FactoredLastAxisBracewellKernel<T, kFactoredMaxQuads, kFold, kN, kLanes>, sharedBytes);
    } else {
        perSm = FactoredBlocksPerSm(info,
            FactoredLastAxisBracewellKernelInPlace<T, kFactoredMaxQuads, kItems, kMinBlocks, kFold, kN, kLanes>,
            sharedBytes);
    }
    const size_t resident = static_cast<size_t>(perSm) * static_cast<size_t>(info.sms);
    const unsigned int grid = static_cast<unsigned int>(batches < resident ? batches : resident);
    const int Hi = static_cast<int>(H);
    const int Di = static_cast<int>(D);
    const int gy = static_cast<int>(groupsY);
    const int gs = static_cast<int>(groups);
    const int lanesI = static_cast<int>(lanes);
    const int batchesI = static_cast<int>(batches);
    if constexpr (kItems == 0) {
        FactoredLastAxisBracewellKernel<T, kFactoredMaxQuads, kFold, kN, kLanes>
            <<<grid, kFactoredThreads, sharedBytes>>>(data, tables, Hi, Di, gy, gs, lanesI, batchesI, workStride);
    } else {
        FactoredLastAxisBracewellKernelInPlace<T, kFactoredMaxQuads, kItems, kMinBlocks, kFold, kN, kLanes>
            <<<grid, kFactoredThreads, sharedBytes>>>(data, tables, Hi, Di, gy, gs, lanesI, batchesI, workStride);
    }
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
}

/*
 * R-4 step 4D: kernels compiled for one (length, batch width) -- the sizes
 * become constants (FactoredSpecialize in factored_kernels.cuh). Only where
 * that won: specialized everywhere it made 256^3 and 512^3 13-15 % slower
 * (benchmarks/results/2026-11-20_*_r4_s4d*, f32, sm_86), and per kernel it
 * both won and lost. The register counts explain the worst loss: the in-place
 * y pass at 512^3 went from 150 to 96 registers, so two blocks fit an SM
 * instead of one -- and that pass is 1.5 times slower with two blocks, as in
 * step 4B. So, from the per-kernel traces at a locked clock:
 *  - fused x pass: 128 x 16 and 256 x 16 through the work buffer (-9..-11 %),
 *    512 x 8 in place (-16 %);
 *  - axis kernel: 128 x 16 through the work buffer (z, y -6 %), 512 x 16 in
 *    place with two tiles a thread (z, -9 %).
 * Since step 4.Y the 512 x 16 two-tile kernel also runs y at 512 (2.83 ms
 * against 4.51 with 8 lines, 2026-11-20_*_r4_y512_*).
 * Not specialized: z at 256 in place (+8 %), y at
 * 256 (the minimum -15 % but the median +2 %, and 256^3 lost 13 % in the
 * benchmark), the separable x pass (not traced), 1024 (not measured), double.
 */
constexpr bool FactoredSpecializes(bool fused, int items, int minBlocks, int n, int lanes)
{
    if (fused) {
        return (items == 0 && lanes == 16 && (n == 128 || n == 256))
            || (items == 1 && minBlocks == 2 && n == 512 && lanes == 8);
    }
    return (items == 0 && n == 128 && lanes == 16) || (items == 2 && n == 512 && lanes == 16)
        || (items == 4 && n == 512 && lanes == 32);
}

/// Calls `launch` with std::integral_constant<int, n> and <int, lanes> of the
/// specialized kernel when FactoredSpecializes says so, with <0>, <0> -- the
/// runtime-sized kernel -- otherwise.
template <typename T, bool kFused, int kItems, int kMinBlocks, typename Launch>
void FactoredDispatchSize(const FactoredAxisTables<T>& tables, size_t lanes, Launch&& launch)
{
    const int n = tables.n;
    const int l = static_cast<int>(lanes);
    auto as = [&](auto kN, auto kL) -> bool {
        constexpr int kNv = decltype(kN)::value;
        constexpr int kLv = decltype(kL)::value;
        if constexpr (FactoredSpecializes(kFused, kItems, kMinBlocks, kNv, kLv)) {
            if (n == kNv && l == kLv && tables.n1 == FactoredN1Of(kNv)) {
                launch(kN, kL);
                return true;
            }
        }
        return false;
    };
    if constexpr (std::is_same_v<T, float>) {
        using std::integral_constant;
        if (as(integral_constant<int, 128> {}, integral_constant<int, 16> {})
            || as(integral_constant<int, 256> {}, integral_constant<int, 16> {})
            || as(integral_constant<int, 512> {}, integral_constant<int, 8> {})
            || as(integral_constant<int, 512> {}, integral_constant<int, 16> {})
            || as(integral_constant<int, 512> {}, integral_constant<int, 32> {})) {
            return;
        }
    }
    launch(std::integral_constant<int, 0> {}, std::integral_constant<int, 0> {});
}

template <typename T, typename TIn, int kItems, int kMinBlocks>
void LaunchPipelinedAxis(FactoredDeviceInfo& info, size_t batches, size_t sharedBytes, const TIn* in, T* data,
    const T* offset, const FactoredAxisTables<T>& tables, size_t axisStride, size_t laneStride, size_t lanes,
    size_t blocksPerOther, size_t otherStride, int workStride, bool vec)
{
    FactoredDispatchSize<T, false, kItems, kMinBlocks>(tables, lanes, [&](auto kN, auto kL) {
        LaunchPipelinedAxisAs<T, TIn, kItems, kMinBlocks, decltype(kN)::value, decltype(kL)::value>(info, batches,
            sharedBytes, in, data, offset, tables, axisStride, laneStride, lanes, blocksPerOther, otherStride,
            workStride, vec);
    });
}

/// R-5: integers to T minus an offset, element by element, between two
/// buffers of the same layout -- the first pass from integers where the
/// pipelined kernel does not apply (short axes, double): convert, then the
/// usual pass in place.
template <typename T, typename TIn>
__global__ void ConvertFromIntegersKernel(const TIn* in, T* out, const T* offsetOnDevice, size_t count)
{
    const T offset = *offsetOnDevice;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        out[i] = static_cast<T>(in[i]) - offset;
    }
}

/// data[0] += *offset * n in double: the offset returned to the zero
/// coefficient (R-5; the same arithmetic as adding offset * N computed on the
/// host before step 3.1).
template <typename T>
__global__ void AddOffsetToFirstKernel(T* data, const T* offset, double n)
{
    data[0] = static_cast<T>(static_cast<double>(data[0]) + static_cast<double>(*offset) * n);
}

template <typename T>
__global__ void SetOffsetKernel(T* offset, double value)
{
    *offset = static_cast<T>(value);
}

/*
 * R-5 step 3.1 (the reduction of step 1.2, benchmarks/cufft_baseline.cu): the
 * exact sum of every step-th line along x -- lines 0, step, 2 step, ... --
 * into *sum. A fixed grid strides over the sampled values, each thread adds
 * into a 64-bit register, the warp and then the block reduce, one atomicAdd a
 * block. Quads where the width allows (kQuad), single values otherwise.
 */
template <typename TIn, bool kQuad>
__global__ void SumLinesKernel(const TIn* in, size_t width, size_t sampled, size_t step, unsigned long long* sum)
{
    unsigned long long local = 0;
    const size_t threads = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t perLine = kQuad ? width / 4 : width;
    const size_t total = sampled * perLine;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += threads) {
        const size_t line = (i / perLine) * step;
        const size_t at = i % perLine;
        if constexpr (kQuad) {
            using Quad = std::conditional_t<sizeof(TIn) == 1, uchar4, ushort4>;
            const Quad v = reinterpret_cast<const Quad*>(in + line * width)[at];
            local += static_cast<unsigned long long>(v.x) + v.y + v.z + v.w;
        } else {
            local += in[line * width + at];
        }
    }
    for (int shift = 16; shift > 0; shift /= 2) {
        local += __shfl_down_sync(0xffffffffu, local, shift);
    }
    __shared__ unsigned long long warps[32];
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    if (lane == 0) {
        warps[warp] = local;
    }
    __syncthreads();
    if (warp == 0) {
        local = lane < static_cast<int>(blockDim.x) / 32 ? warps[lane] : 0;
        for (int shift = 16; shift > 0; shift /= 2) {
            local += __shfl_down_sync(0xffffffffu, local, shift);
        }
        if (lane == 0) {
            atomicAdd(sum, local);
        }
    }
}

/// *offset = the sum over `count` values, rounded to the nearest integer.
template <typename T>
__global__ void RoundedMeanKernel(const unsigned long long* sum, double count, T* offset)
{
    *offset = static_cast<T>(rint(static_cast<double>(*sum) / count));
}

template <typename T, int kItems, int kMinBlocks>
void LaunchFusedLast(FactoredDeviceInfo& info, size_t batches, size_t sharedBytes, T* data,
    const FactoredAxisTables<T>& tables, size_t H, size_t D, size_t groupsY, size_t groups, size_t lanes,
    int workStride)
{
    FactoredDispatchSize<T, true, kItems, kMinBlocks>(tables, lanes, [&](auto kN, auto kL) {
        LaunchFusedLastAs<T, kItems, kMinBlocks, decltype(kN)::value, decltype(kL)::value>(
            info, batches, sharedBytes, data, tables, H, D, groupsY, groups, lanes, workStride);
    });
}

} // namespace

namespace {

/*
 * The pass along one axis, from `in` into `data`. For TIn = T, `in` is `data`
 * and this is the pass as it was before R-5 (in place). For integer TIn the
 * pipelined kernels read the integers themselves (step 2.2); where they do not
 * apply, the integers are converted into `data` first and the pass runs in
 * place as usual.
 */
template <typename T, typename TIn>
void FactoredAxisImpl(const TIn* in, T* data, const T* offset, const FactoredAxisTables<T>& tables, size_t axisStride,
    size_t laneStride, size_t laneLength, size_t otherStride, size_t otherLength)
{
    if constexpr (!std::is_same<TIn, T>::value) {
        // Converted in place of a pass that cannot read integers; also the
        // whole of a pass of length 1, whose DHT is the value itself.
        auto convertFirst = [&] {
            ConvertFromIntegers(in, data, offset, static_cast<size_t>(tables.n) * laneLength * otherLength);
        };
        if (tables.n <= 1) {
            convertFirst();
            return;
        }
        const size_t n = static_cast<size_t>(tables.n);
        const size_t n1 = static_cast<size_t>(tables.n1);
        const size_t n2 = static_cast<size_t>(tables.n2);
        int device = 0;
        RAPIDHT_CUDA_CHECK(cudaGetDevice(&device));
        FactoredDeviceInfo& info = FactoredInfo<T>(device);
        const int optIn = info.optIn;
        const int axis = axisStride == 1 ? 0 : (axisStride == laneLength ? 1 : 2);
        size_t lanes = 0;
        if (sizeof(T) == 4 && n1 >= 4 && n2 >= 4) {
            // The widths of the float pass (R-5 step 2.4): z from u16 / u8
            // with 8, 16 and 32 lines at 128^3-512^3 found nothing faster
            // than them anywhere -- 8 loses 6-18 %, 32 below 512 up to 9 %,
            // 16 at 512 3 % on u8 (benchmarks/results/2026-11-26_*_r5_s24_*).
            lanes = FactoredPreferredLanes(n, axisStride, laneLength);
            while (lanes > 4 && !(lanes <= laneLength && PipelinedAxisFits<T>(n, n1, n2, lanes, optIn, axis))) {
                lanes /= 2;
            }
        }
        // The same conditions as FactoredAxis for its pipelined branch: the
        // register bound is taken at the starting width of 8 lines, the
        // preferred width is then checked by PipelinedAxisFits (its in-place
        // variants hold up to 16 quads, more than kFactoredMaxQuads).
        const size_t startLanes = laneLength < 8 ? laneLength : 8;
        if (lanes < 4 || lanes > laneLength || !PipelinedAxisFits<T>(n, n1, n2, lanes, optIn, axis)
            || (n * startLanes / 4 + kFactoredThreads - 1) / kFactoredThreads > kFactoredMaxQuads) {
            convertFirst();
            FactoredAxis(data, tables, axisStride, laneStride, laneLength, otherStride, otherLength);
            return;
        }
        const FactoredPath path = FactoredPathFor(n, lanes, axis);
        const size_t sharedBytes = PipelinedSharedBytes<T>(n, n1, n2, lanes, path.items != 0);
        const size_t blocksPerOther = laneLength / lanes;
        const size_t batches = blocksPerOther * otherLength;
        // Quads as vectors: the same offsets keep a float4 store 16-byte and
        // a ushort4 / uchar4 load 8- / 4-byte aligned.
        const bool vec = laneStride == 1 && axisStride % 4 == 0 && otherStride % 4 == 0
                      && reinterpret_cast<uintptr_t>(data) % 16 == 0
                      && reinterpret_cast<uintptr_t>(in) % (4 * sizeof(TIn)) == 0;
        const int ws = static_cast<int>(FactoredWorkStride<T>(n2, lanes));
        // R-5 step 2.5: z at 512 from uint16 reads batches of 64 lines (128
        // contiguous bytes a row) and computes them as two tiles of 32: the
        // pass 4.08 -> 3.26 ms, the transform 10.47 -> 9.69 ms (A-B
        // benchmarks/results/2026-11-26_*_r5_s25_{a,b}). Not uint8: 64 lines
        // of it are 64 bytes, and z went 3.91 -> 4.24 ms (unlocked clock).
        if constexpr (std::is_same<T, float>::value) {
            if (sizeof(TIn) == 2 && n == 512 && n1 == 32 && axis == 2 && lanes == 32 && path.items == 4 && vec
                && laneLength % 64 == 0) {
                constexpr bool kFold = FactoredFoldsTwiddle(false, 4);
                auto kernel = FactoredAxisKernelWideFrom<T, TIn, 512, 32, kFold>;
                const size_t wideBlocksPerOther = laneLength / 64;
                const size_t wideBatches = wideBlocksPerOther * otherLength;
                const int perSm = FactoredBlocksPerSm(info, kernel, sharedBytes);
                const size_t resident = static_cast<size_t>(perSm) * static_cast<size_t>(info.sms);
                const unsigned int grid
                    = static_cast<unsigned int>(wideBatches < resident ? wideBatches : resident);
                kernel<<<grid, kFactoredThreads, sharedBytes>>>(in, data, offset, tables, axisStride,
                    static_cast<int>(wideBlocksPerOther), otherStride, static_cast<int>(wideBatches));
                RAPIDHT_CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
        if (path.items == 0) {
            LaunchPipelinedAxis<T, TIn, 0, 0>(info, batches, sharedBytes, in, data, offset, tables, axisStride,
                laneStride, lanes, blocksPerOther, otherStride, ws, vec);
        } else if (path.items == 4) {
            LaunchPipelinedAxis<T, TIn, 4, 1>(info, batches, sharedBytes, in, data, offset, tables, axisStride,
                laneStride, lanes, blocksPerOther, otherStride, ws, vec);
        } else if (path.items == 2) {
            LaunchPipelinedAxis<T, TIn, 2, 1>(info, batches, sharedBytes, in, data, offset, tables, axisStride,
                laneStride, lanes, blocksPerOther, otherStride, ws, vec);
        } else if (path.minBlocks == 1) {
            LaunchPipelinedAxis<T, TIn, 1, 1>(info, batches, sharedBytes, in, data, offset, tables, axisStride,
                laneStride, lanes, blocksPerOther, otherStride, ws, vec);
        } else {
            LaunchPipelinedAxis<T, TIn, 1, 2>(info, batches, sharedBytes, in, data, offset, tables, axisStride,
                laneStride, lanes, blocksPerOther, otherStride, ws, vec);
        }
    } else {
        (void)in;
        (void)offset;
        FactoredAxis(data, tables, axisStride, laneStride, laneLength, otherStride, otherLength);
    }
}

} // namespace

template <typename T, typename TIn>
void FactoredAxisFrom(const TIn* in, T* out, const T* offset, const FactoredAxisTables<T>& tables, size_t axisStride,
    size_t laneStride, size_t laneLength, size_t otherStride, size_t otherLength)
{
    FactoredAxisImpl(in, out, offset, tables, axisStride, laneStride, laneLength, otherStride, otherLength);
}

template <typename T, typename TIn>
void ConvertFromIntegers(const TIn* in, T* out, const T* offset, size_t count)
{
    ConvertFromIntegersKernel<T, TIn><<<1024, 256>>>(in, out, offset, count);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void AddOffsetToFirst(T* data, const T* offset, double n)
{
    AddOffsetToFirstKernel<T><<<1, 1>>>(data, offset, n);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void SetIntegerOffset(T* offset, double value)
{
    SetOffsetKernel<T><<<1, 1>>>(offset, value);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename TIn>
void IntegerMeanOffset(const TIn* in, size_t width, size_t lines, size_t step, unsigned long long* sum, T* offset)
{
    const size_t sampled = (lines + step - 1) / step;
    const bool quad = width % 4 == 0 && reinterpret_cast<uintptr_t>(in) % (4 * sizeof(TIn)) == 0;
    const size_t perLine = quad ? width / 4 : width;
    int device = 0;
    int sms = 0;
    RAPIDHT_CUDA_CHECK(cudaGetDevice(&device));
    RAPIDHT_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const unsigned block = 256;
    // Enough blocks to fill the card (8 a SM), never more than there are reads.
    const size_t wanted = (sampled * perLine + block - 1) / block;
    const unsigned grid = static_cast<unsigned>(std::min<size_t>(wanted, static_cast<size_t>(sms) * 8));
    RAPIDHT_CUDA_CHECK(cudaMemsetAsync(sum, 0, sizeof(unsigned long long)));
    if (quad) {
        SumLinesKernel<TIn, true><<<grid, block>>>(in, width, sampled, step, sum);
    } else {
        SumLinesKernel<TIn, false><<<grid, block>>>(in, width, sampled, step, sum);
    }
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
    RoundedMeanKernel<T><<<1, 1>>>(sum, static_cast<double>(sampled * width), offset);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void FactoredAxis(T* data, const FactoredAxisTables<T>& tables, size_t axisStride, size_t laneStride,
    size_t laneLength, size_t otherStride, size_t otherLength)
{
    if (tables.n <= 1) {
        return; // the DHT of a single value is that value
    }
    const size_t n = static_cast<size_t>(tables.n);
    const size_t n1 = static_cast<size_t>(tables.n1);
    const size_t n2 = static_cast<size_t>(tables.n2);

    int device = 0;
    RAPIDHT_CUDA_CHECK(cudaGetDevice(&device));
    FactoredDeviceInfo& info = FactoredInfo<T>(device);
    const int optIn = info.optIn;

    // The register-tiled kernel wherever its multiples of 4 hold: 8 lines per
    // block in float, 4 in double, for two blocks per SM at n = 512.
    if (n1 >= 4 && n2 >= 4) {
        size_t lanes = (sizeof(T) == 4 ? 8 : 4);
        while (lanes > laneLength) {
            lanes /= 2;
        }
        auto bytesFor = [&](size_t l) {
            return (n * l + n1 * FactoredWorkStride<T>(n2, l) + n1 * n1 + 2 * n2 * n2) * sizeof(T);
        };
        if (lanes > 4 && bytesFor(lanes) > static_cast<size_t>(optIn)) {
            lanes = 4;
        }
        const size_t quadsPerThread = (n * lanes / 4 + kFactoredThreads - 1) / kFactoredThreads;
        // Double stays on the v2 kernel: the pipelined one holds the next
        // batch in registers, 168 of them a thread in double, so one block
        // fits on an SM and it is 8.5-42 % slower than v2 at 128^3-512^3
        // (R-4 step 2.0, benchmarks/results/2026-11-15_*_r4_s20_f64_*.json).
        if (sizeof(T) == 4 && lanes >= 4 && bytesFor(lanes) <= static_cast<size_t>(optIn)
            && quadsPerThread <= kFactoredMaxQuads) {
            // The width the sweep found best for this axis and length, halved
            // until it fits the shared memory, the lane axis and the registers
            // (R-4 steps 2.2, 2.4); then the variant of the kernel (step 4B).
            const int axis = axisStride == 1 ? 0 : (axisStride == laneLength ? 1 : 2);
            size_t preferred = FactoredPreferredLanes(n, axisStride, laneLength);
            while (preferred > 4
                   && !(preferred <= laneLength && PipelinedAxisFits<T>(n, n1, n2, preferred, optIn, axis))) {
                preferred /= 2;
            }
            if (preferred <= laneLength && PipelinedAxisFits<T>(n, n1, n2, preferred, optIn, axis)) {
                lanes = preferred;
            }
            if (PipelinedAxisFits<T>(n, n1, n2, lanes, optIn, axis)) {
                const FactoredPath path = FactoredPathFor(n, lanes, axis);
                const size_t sharedBytes = PipelinedSharedBytes<T>(n, n1, n2, lanes, path.items != 0);
                const size_t blocksPerOther = laneLength / lanes;
                const size_t batches = blocksPerOther * otherLength;
                // A quad of lines is one vector access when the lines are
                // neighbours in memory and every offset keeps 16-byte alignment.
                const bool vec = laneStride == 1 && axisStride % 4 == 0 && otherStride % 4 == 0
                              && reinterpret_cast<uintptr_t>(data) % 16 == 0;
                const int ws = static_cast<int>(FactoredWorkStride<T>(n2, lanes));
                if (path.items == 0) {
                    LaunchPipelinedAxis<T, T, 0, 0>(info, batches, sharedBytes, data, data, nullptr, tables, axisStride,
                        laneStride, lanes, blocksPerOther, otherStride, ws, vec);
                } else if (path.items == 4) {
                    LaunchPipelinedAxis<T, T, 4, 1>(info, batches, sharedBytes, data, data, nullptr, tables, axisStride,
                        laneStride, lanes, blocksPerOther, otherStride, ws, vec);
                } else if (path.items == 2) {
                    LaunchPipelinedAxis<T, T, 2, 1>(info, batches, sharedBytes, data, data, nullptr, tables, axisStride,
                        laneStride, lanes, blocksPerOther, otherStride, ws, vec);
                } else if (path.minBlocks == 1) {
                    LaunchPipelinedAxis<T, T, 1, 1>(info, batches, sharedBytes, data, data, nullptr, tables, axisStride,
                        laneStride, lanes, blocksPerOther, otherStride, ws, vec);
                } else {
                    LaunchPipelinedAxis<T, T, 1, 2>(info, batches, sharedBytes, data, data, nullptr, tables, axisStride,
                        laneStride, lanes, blocksPerOther, otherStride, ws, vec);
                }
                return;
            }
        }
        if (lanes >= 4 && bytesFor(lanes) <= static_cast<size_t>(optIn)) {
            const size_t sharedBytes = bytesFor(lanes);
            if (sharedBytes > info.fastSharedOptIn) {
                RAPIDHT_CUDA_CHECK(cudaFuncSetAttribute(FactoredAxisKernelFast<T>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
                info.fastSharedOptIn = sharedBytes;
            }
            const size_t blocksPerOther = laneLength / lanes;
            FactoredAxisKernelFast<T><<<static_cast<unsigned int>(blocksPerOther * otherLength), 256, sharedBytes>>>(
                data, tables, axisStride, laneStride, static_cast<int>(lanes), static_cast<int>(blocksPerOther),
                otherStride, static_cast<int>(FactoredWorkStride<T>(n2, lanes)));
            RAPIDHT_CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    // Generic kernel: short lines (n < 16) or fewer than 4 lines to a block.
    // As many neighbouring lines as fit twice into 48 KiB, at most 16, a power
    // of two so that it divides laneLength.
    const size_t lineBytes = 2 * n * sizeof(T);
    size_t lanes = 16;
    while (lanes > 1 && (lanes * lineBytes > 48 * 1024 || lanes > laneLength)) {
        lanes /= 2;
    }
    const size_t blocksPerOther = laneLength / lanes;
    const size_t blocks = blocksPerOther * otherLength;
    const size_t sharedBytes = lanes * lineBytes;

    FactoredAxisKernel<T><<<static_cast<unsigned int>(blocks), 256, sharedBytes>>>(data, tables, axisStride,
        laneStride, static_cast<int>(lanes), static_cast<int>(blocksPerOther), otherStride);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
bool FactoredLastAxisBracewell(T* data, const FactoredAxisTables<T>& tables, size_t H, size_t D)
{
    const size_t n = static_cast<size_t>(tables.n);
    const size_t n1 = static_cast<size_t>(tables.n1);
    const size_t n2 = static_cast<size_t>(tables.n2);
    // Float only, like the pipelined kernel it is built on (double stays on
    // v2, R-4 step 2.0); n1, n2 >= 4 for the 4 x 4 register tiles.
    if (sizeof(T) != 4 || n1 < 4 || n2 < 4) {
        return false;
    }
    int device = 0;
    RAPIDHT_CUDA_CHECK(cudaGetDevice(&device));
    FactoredDeviceInfo& info = FactoredInfo<T>(device);

    const size_t groupsY = H / 2 + 1; // orbits (y, -y); 1 when H = 1
    const size_t groupsZ = D / 2 + 1; // orbits (z, -z); 1 in 2D
    const size_t groups = groupsY * groupsZ;
    // The width the x pass would take (step 2.4), no wider than the groups
    // need, halved until it fits; then the variant of the kernel (step 4B).
    size_t lanes = n >= 512 ? 8 : FactoredPreferredLanes(n, 1, H); // 16 lose 17 % at 512 (step 4.X)
    while (lanes > 4 && (lanes / 4 > groups || !PipelinedFits<T>(n, n1, n2, lanes, info.optIn))) {
        lanes /= 2;
    }
    if (!PipelinedFits<T>(n, n1, n2, lanes, info.optIn)) {
        return false;
    }
    const FactoredPath path = FactoredPathFor(n, lanes, 0);
    const size_t perBatch = lanes / 4;
    const size_t batches = (groups + perBatch - 1) / perBatch;
    const size_t sharedBytes = PipelinedSharedBytes<T>(n, n1, n2, lanes, path.items != 0);
    const int ws = static_cast<int>(FactoredWorkStride<T>(n2, lanes));
    if (path.items == 0) {
        LaunchFusedLast<T, 0, 0>(info, batches, sharedBytes, data, tables, H, D, groupsY, groups, lanes, ws);
    } else if (path.items == 2) {
        LaunchFusedLast<T, 2, 1>(info, batches, sharedBytes, data, tables, H, D, groupsY, groups, lanes, ws);
    } else if (path.minBlocks == 1) {
        LaunchFusedLast<T, 1, 1>(info, batches, sharedBytes, data, tables, H, D, groupsY, groups, lanes, ws);
    } else {
        LaunchFusedLast<T, 1, 2>(info, batches, sharedBytes, data, tables, H, D, groupsY, groups, lanes, ws);
    }
    return true;
}

template <typename T>
void BracewellInPlace2D(T* data, int W, int H)
{
    const dim3 block(32, 8);
    const dim3 grid((W / 2 + 1 + block.x - 1) / block.x, (H / 2 + 1 + block.y - 1) / block.y);
    BracewellInPlace2DKernel<T><<<grid, block>>>(data, W, H);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void BracewellInPlace3D(T* data, int W, int H, int D)
{
    const dim3 block(32, 8);
    const dim3 grid((W / 2 + 1 + block.x - 1) / block.x, (H / 2 + 1 + block.y - 1) / block.y, D / 2 + 1);
    BracewellInPlace3DKernel<T><<<grid, block>>>(data, W, H, D);
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
}

template void FactoredAxis<float>(float*, const FactoredAxisTables<float>&, size_t, size_t, size_t, size_t, size_t);
template void FactoredAxis<double>(
    double*, const FactoredAxisTables<double>&, size_t, size_t, size_t, size_t, size_t);
template void FactoredAxisFrom<float, std::uint8_t>(const std::uint8_t*, float*, const float*,
    const FactoredAxisTables<float>&, size_t, size_t, size_t, size_t, size_t);
template void FactoredAxisFrom<float, std::uint16_t>(const std::uint16_t*, float*, const float*,
    const FactoredAxisTables<float>&, size_t, size_t, size_t, size_t, size_t);
template void FactoredAxisFrom<double, std::uint8_t>(const std::uint8_t*, double*, const double*,
    const FactoredAxisTables<double>&, size_t, size_t, size_t, size_t, size_t);
template void FactoredAxisFrom<double, std::uint16_t>(const std::uint16_t*, double*, const double*,
    const FactoredAxisTables<double>&, size_t, size_t, size_t, size_t, size_t);
template void ConvertFromIntegers<float, std::uint8_t>(const std::uint8_t*, float*, const float*, size_t);
template void IntegerMeanOffset<float, std::uint8_t>(const std::uint8_t*, size_t, size_t, size_t, unsigned long long*, float*);
template void ConvertFromIntegers<float, std::uint16_t>(const std::uint16_t*, float*, const float*, size_t);
template void IntegerMeanOffset<float, std::uint16_t>(const std::uint16_t*, size_t, size_t, size_t, unsigned long long*, float*);
template void ConvertFromIntegers<double, std::uint8_t>(const std::uint8_t*, double*, const double*, size_t);
template void IntegerMeanOffset<double, std::uint8_t>(const std::uint8_t*, size_t, size_t, size_t, unsigned long long*, double*);
template void ConvertFromIntegers<double, std::uint16_t>(const std::uint16_t*, double*, const double*, size_t);
template void IntegerMeanOffset<double, std::uint16_t>(const std::uint16_t*, size_t, size_t, size_t, unsigned long long*, double*);
template void AddOffsetToFirst<float>(float*, const float*, double);
template void AddOffsetToFirst<double>(double*, const double*, double);
template void SetIntegerOffset<float>(float*, double);
template void SetIntegerOffset<double>(double*, double);
template bool FactoredLastAxisBracewell<float>(float*, const FactoredAxisTables<float>&, size_t, size_t);
template bool FactoredLastAxisBracewell<double>(double*, const FactoredAxisTables<double>&, size_t, size_t);
template void BracewellInPlace2D<float>(float*, int, int);
template void BracewellInPlace2D<double>(double*, int, int);
template void BracewellInPlace3D<float>(float*, int, int, int);
template void BracewellInPlace3D<double>(double*, int, int, int);

// Matrix operations
template void MatrixMultiplication<float>(const float* A, const float* B, float* C, int M, int K, int N);
template void MatrixMultiplication<double>(const double* A, const double* B, double* C, int M, int K, int N);

// Transposition
template void MatrixTranspose<float>(const float* A, float* B, int rows, int cols);
template void MatrixTranspose<double>(const double* A, double* B, int rows, int cols);

template void ScaleOnDevice<float>(float* deviceData, size_t count, float factor);
template void ScaleOnDevice<double>(double* deviceData, size_t count, double factor);

// Vector times matrix
template void VectorMatrixMultiplication<float>(const float* A, const float* x, float* y, int N);
template void VectorMatrixMultiplication<double>(const double* A, const double* x, double* y, int N);

// Bracewell correction
template void BracewellTransform2D<float>(const float* deviceIn, float* deviceOut, int W, int H);
template void BracewellTransform2D<double>(const double* deviceIn, double* deviceOut, int W, int H);

template void BracewellTransform3D<float>(const float* deviceIn, float* deviceOut, int W, int H, int D);
template void BracewellTransform3D<double>(const double* deviceIn, double* deviceOut, int W, int H, int D);

} // namespace RapiDHT
