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
#include <cstdlib>
#include <cstring>
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

/// RAPIDHT_FACTORED_KERNEL=v2 keeps the one-batch-per-block kernel of v2.2,
/// so that both can be timed from the same build.
bool FactoredUseV2Kernel()
{
    static const bool v2 = [] {
        const char* s = std::getenv("RAPIDHT_FACTORED_KERNEL");
        return s != nullptr && std::strcmp(s, "v2") == 0;
    }();
    return v2;
}

/// RAPIDHT_R4_ABLATE=io|compute: the diagnostic modes of the pipelined kernel
/// (R-4 step 1.1). Wrong results by design; never set outside a measurement.
int FactoredAblateMode()
{
    static const int mode = [] {
        const char* s = std::getenv("RAPIDHT_R4_ABLATE");
        if (s == nullptr) {
            return kAblateNone;
        }
        if (std::strcmp(s, "io") == 0) {
            return kAblateIo;
        }
        if (std::strcmp(s, "compute") == 0) {
            return kAblateCompute;
        }
        return kAblateNone;
    }();
    return mode;
}

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
        return n <= 256 ? 16 : 8;
    }
    if (alongY) {
        return n <= 128 ? 16 : 8;
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
    size_t pipelinedSharedOptIn = 48 * 1024; ///< largest size the attribute is set to
    std::vector<std::pair<size_t, int>> pipelinedBlocksPerSm; ///< shared bytes -> blocks per SM
    size_t fastSharedOptIn = 48 * 1024;
    size_t fusedSharedOptIn = 48 * 1024; ///< the same two for the fused last pass
    std::vector<std::pair<size_t, int>> fusedBlocksPerSm;
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

} // namespace

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
            && quadsPerThread <= kFactoredMaxQuads && !FactoredUseV2Kernel()) {
            // The width the sweep found best for this axis and length, halved
            // until it fits the shared memory, the lane axis and the registers
            // the next batch is fetched into (R-4 steps 2.2 and 2.4).
            auto fits = [&](size_t l) {
                return l >= 4 && l <= laneLength && bytesFor(l) <= static_cast<size_t>(optIn)
                    && (n * l / 4 + kFactoredThreads - 1) / kFactoredThreads <= kFactoredMaxQuads;
            };
            size_t preferred = FactoredPreferredLanes(n, axisStride, laneLength);
            while (preferred > 4 && !fits(preferred)) {
                preferred /= 2;
            }
            if (fits(preferred)) {
                lanes = preferred;
            }
            // v2.3: a pipelined grid of as many blocks as fit at once.
            const size_t sharedBytes = bytesFor(lanes);
            const auto kernel = FactoredAxisKernelPipelined<T, kFactoredMaxQuads>;
            if (sharedBytes > info.pipelinedSharedOptIn) {
                RAPIDHT_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
                info.pipelinedSharedOptIn = sharedBytes;
            }
            int perSm = 0;
            for (const auto& known : info.pipelinedBlocksPerSm) {
                if (known.first == sharedBytes) {
                    perSm = known.second;
                }
            }
            if (perSm == 0) {
                RAPIDHT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &perSm, kernel, kFactoredThreads, sharedBytes));
                perSm = perSm > 0 ? perSm : 1;
                info.pipelinedBlocksPerSm.emplace_back(sharedBytes, perSm);
            }
            const int sms = info.sms;
            const size_t blocksPerOther = laneLength / lanes;
            const size_t batches = blocksPerOther * otherLength;
            const size_t resident = static_cast<size_t>(perSm) * static_cast<size_t>(sms);
            const size_t grid = batches < resident ? batches : resident;
            // A quad of lines is one vector access when the lines are
            // neighbours in memory and every offset keeps 16-byte alignment.
            const bool vec = laneStride == 1 && axisStride % 4 == 0 && otherStride % 4 == 0
                          && reinterpret_cast<uintptr_t>(data) % 16 == 0;
            FactoredAxisKernelPipelined<T, kFactoredMaxQuads>
                <<<static_cast<unsigned int>(grid), kFactoredThreads, sharedBytes>>>(data, tables, axisStride,
                    laneStride, static_cast<int>(lanes), static_cast<int>(blocksPerOther), otherStride,
                    static_cast<int>(batches), static_cast<int>(FactoredWorkStride<T>(n2, lanes)), vec,
                    FactoredAblateMode());
            RAPIDHT_CUDA_CHECK(cudaGetLastError());
            return;
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
    if (sizeof(T) != 4 || n1 < 4 || n2 < 4 || FactoredUseV2Kernel()) {
        return false;
    }
    int device = 0;
    RAPIDHT_CUDA_CHECK(cudaGetDevice(&device));
    FactoredDeviceInfo& info = FactoredInfo<T>(device);
    const size_t optIn = static_cast<size_t>(info.optIn);

    const size_t groupsY = H / 2 + 1; // orbits (y, -y); 1 when H = 1
    const size_t groupsZ = D / 2 + 1; // orbits (z, -z); 1 in 2D
    const size_t groups = groupsY * groupsZ;
    auto bytesFor = [&](size_t l) {
        return (n * l + n1 * FactoredWorkStride<T>(n2, l) + n1 * n1 + 2 * n2 * n2) * sizeof(T);
    };
    auto fits = [&](size_t l) {
        return bytesFor(l) <= optIn && (n * l / 4 + kFactoredThreads - 1) / kFactoredThreads <= kFactoredMaxQuads;
    };
    // The width the x pass would take (step 2.4), no wider than the groups
    // need, then halved until it fits.
    size_t lanes = FactoredPreferredLanes(n, 1, H);
    while (lanes > 4 && (lanes / 4 > groups || !fits(lanes))) {
        lanes /= 2;
    }
    if (!fits(lanes)) {
        return false;
    }
    const size_t perBatch = lanes / 4;
    const size_t batches = (groups + perBatch - 1) / perBatch;
    const size_t sharedBytes = bytesFor(lanes);
    const auto kernel = FactoredLastAxisBracewellKernel<T, kFactoredMaxQuads>;
    if (sharedBytes > info.fusedSharedOptIn) {
        RAPIDHT_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
        info.fusedSharedOptIn = sharedBytes;
    }
    int perSm = 0;
    for (const auto& known : info.fusedBlocksPerSm) {
        if (known.first == sharedBytes) {
            perSm = known.second;
        }
    }
    if (perSm == 0) {
        RAPIDHT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kernel, kFactoredThreads, sharedBytes));
        perSm = perSm > 0 ? perSm : 1;
        info.fusedBlocksPerSm.emplace_back(sharedBytes, perSm);
    }
    const size_t resident = static_cast<size_t>(perSm) * static_cast<size_t>(info.sms);
    const size_t grid = batches < resident ? batches : resident;
    FactoredLastAxisBracewellKernel<T, kFactoredMaxQuads><<<static_cast<unsigned int>(grid), kFactoredThreads,
        sharedBytes>>>(data, tables, static_cast<int>(H), static_cast<int>(D), static_cast<int>(groupsY),
        static_cast<int>(groups), static_cast<int>(lanes), static_cast<int>(batches),
        static_cast<int>(FactoredWorkStride<T>(n2, lanes)));
    RAPIDHT_CUDA_CHECK(cudaGetLastError());
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
