/*
 * Project: RapiDHT
 * File: src/kernels.cu
 * Brief: CUDA-ядра и хост-обёртки для матричных операций и преобразования Хартли.
 * Author: Волков Евгений Александрович, volkov22dla@yandex.ru
 */

#include "device_launch_parameters.h"
#include "internal/device_array.h"
#include "internal/kernels.h"

// #ifndef TILE_DIM
// #define TILE_DIM 32
// #endif
// #ifndef BLOCK_ROWS
// #define BLOCK_ROWS 8
// #endif

// ------------------------------ Kernels ------------------------------

namespace RapiDHT {

template <typename T>
__global__ void transpose_YZ_kernel(const T* __restrict__ in, T* __restrict__ out, int W, int H, int D)
{
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int bz = blockIdx.z * blockDim.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int x = bx + tx;
    int y = by + ty;
    int z = bz + tz;

    if (x >= W || y >= H || z >= D)
        return;

    // исходный индекс (row-major, x fastest)
    size_t in_idx = (size_t)z * (W * (size_t)H) + (size_t)y * W + x;
    // целевой индекс после swap Y<->Z: out dims = W x D x H
    // координаты в out: (x_out, y_out, z_out) = (x, z, y)
    size_t out_idx = (size_t)y * (W * (size_t)D) + (size_t)z * W + x;

    out[out_idx] = in[in_idx];
}

template <typename T>
__global__ void MatrixMultiplicationKernelShared(const T* __restrict__ A, const T* __restrict__ B,
    T* __restrict__ C, int M, int K, int N)
{
    const int BLOCK_SIZE = 16;
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    T sum = 0.0;

    // Цикл по "плиткам" (tiles) матриц A и B
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Загружаем кусок A и B в shared memory
        if (row < M && t * BLOCK_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && t * BLOCK_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        // Умножаем плитки
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Записываем результат
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

template <typename T>
__global__ void MatrixVectorMultKernel(const T* A, const T* x, T* y, int N)
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

/*
 * The same transpose applied to every slice of a volume, in one launch.
 *
 * The 3D path used to call cublas<t>geam once per slice, in a host-side loop.
 * Profiling a 512^3 transform showed 1024 such launches against 3 GEMMs: the
 * kernels themselves cost 5.8 ms of the 54.8 ms spent on the device, but the
 * device sat idle for most of the 313 ms of wall time waiting between them.
 * One launch with the slice on blockIdx.z removes that entirely.
 */
template <typename T>
__global__ void MatrixTransposeBatchedKernel(const T* __restrict__ in, T* __restrict__ out,
    int rows, int cols, int batch)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int slice = blockIdx.z;

    if (col >= cols || row >= rows || slice >= batch) {
        return;
    }

    const size_t offset = static_cast<size_t>(slice) * rows * cols;
    out[offset + static_cast<size_t>(col) * rows + row]
        = in[offset + static_cast<size_t>(row) * cols + col];
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
__global__ void BracewellTransform2D_Kernel(const T* __restrict__ in, T* __restrict__ out, int W, int H)
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
__global__ void BracewellTransform3D_Kernel(const T* __restrict__ in, T* __restrict__ out, int W, int H, int D)
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
        const double m_pi = 3.14159265358979323846;
        kernel[k * height + j] = cos(2.0 * m_pi * k * j / height) + sin(2.0 * m_pi * k * j / height);
    }
}

__global__ void InitializeHartleyMatrixKernel(float* kernel, size_t height)
{
    size_t k = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < height && j < height) {
        const float m_pi = 3.14159265358979323846f;
        kernel[k * height + j] = cosf(2.0f * m_pi * k * j / height) + sinf(2.0f * m_pi * k * j / height);
    }
}

// ------------------------------ Host Wrappers ------------------------------

template <typename T>
void transpose_YZ_cuda(const T* d_in, T* d_out, int W, int H, int D)
{
    dim3 block(8, 8, 8);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, (D + block.z - 1) / block.z);

    transpose_YZ_kernel<T><<<grid, block>>>(d_in, d_out, W, H, D);
    cudaDeviceSynchronize();
}

template <typename T>
void MatrixMultiplication(const T* A, const T* B, T* C, int M, int K, int N)
{
    const int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // MatrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, M, K, N);
    MatrixMultiplicationKernelShared<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);

    cudaDeviceSynchronize();
}

template <typename T>
void VectorMatrixMultiplication(const T* A, const T* x, T* y, int N)
{
    int threadsPerBlock = (N > 512) ? 512 : N;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    MatrixVectorMultKernel<<<blocksPerGrid, threadsPerBlock>>>(A, x, y, N);
    cudaDeviceSynchronize();
}

// rows, cols - целевые (размеры матрицы B)
template <typename T>
void MatrixTranspose(const T* A, T* B, int rows, int cols)
{
    int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    MatrixTransposeKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, rows, cols);
    cudaDeviceSynchronize();
}

template <typename T>
void ScaleOnDevice(T* d_data, size_t count, T factor)
{
    const int threads = 256;
    const size_t blocks = (count + threads - 1) / threads;

    ScaleKernel<<<static_cast<unsigned int>(blocks), threads>>>(d_data, count, factor);
    cudaDeviceSynchronize();
}

template <typename T>
void MatrixTransposeBatched(const T* d_in, T* d_out, int rows, int cols, int batch)
{
    const int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch);

    MatrixTransposeBatchedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, rows, cols, batch);
    cudaDeviceSynchronize();
}

template <typename T>
void BracewellTransform2D(const T* d_in, T* d_out, int W, int H)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
        (H + blockDim.y - 1) / blockDim.y);

    BracewellTransform2D_Kernel<<<gridDim, blockDim>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();
}

template <typename T>
void BracewellTransform3D(const T* d_in, T* d_out, int W, int H, int D)
{
    dim3 blockDim(8, 8, 8); // можно подбирать под вашу карту
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
        (H + blockDim.y - 1) / blockDim.y,
        (D + blockDim.z - 1) / blockDim.z);

    BracewellTransform3D_Kernel<<<gridDim, blockDim>>>(d_in, d_out, W, H, D);
    cudaDeviceSynchronize();
}

void InitializeHartleyMatrix(double* dKernel, size_t height)
{
    dim3 block(16, 16);
    dim3 grid((height + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    InitializeHartleyMatrixKernel<<<grid, block>>>(dKernel, height);
    cudaDeviceSynchronize();
}

void InitializeHartleyMatrix(float* dKernel, size_t height)
{
    dim3 block(16, 16);
    dim3 grid((height + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    InitializeHartleyMatrixKernel<<<grid, block>>>(dKernel, height);
    cudaDeviceSynchronize();
}

template void transpose_YZ_cuda<float>(const float* d_in, float* d_out, int W, int H, int D);
template void transpose_YZ_cuda<double>(const double* d_in, double* d_out, int W, int H, int D);

// Общая матричная операция
template void MatrixMultiplication<float>(const float* A, const float* B, float* C, int M, int K, int N);
template void MatrixMultiplication<double>(const double* A, const double* B, double* C, int M, int K, int N);

// Транспонирование
template void MatrixTranspose<float>(const float* A, float* B, int rows, int cols);
template void MatrixTranspose<double>(const double* A, double* B, int rows, int cols);

template void MatrixTransposeBatched<float>(const float* d_in, float* d_out, int rows, int cols, int batch);
template void MatrixTransposeBatched<double>(const double* d_in, double* d_out, int rows, int cols, int batch);

template void ScaleOnDevice<float>(float* d_data, size_t count, float factor);
template void ScaleOnDevice<double>(double* d_data, size_t count, double factor);

// Умножение вектор-матрица
template void VectorMatrixMultiplication<float>(const float* A, const float* x, float* y, int N);
template void VectorMatrixMultiplication<double>(const double* A, const double* x, double* y, int N);

// 3D преобразование Брэсвелла
template void BracewellTransform2D<float>(const float* d_in, float* d_out, int W, int H);
template void BracewellTransform2D<double>(const double* d_in, double* d_out, int W, int H);

template void BracewellTransform3D<float>(const float* d_in, float* d_out, int W, int H, int D);
template void BracewellTransform3D<double>(const double* d_in, double* d_out, int W, int H, int D);

} // namespace RapiDHT
