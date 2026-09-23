/*
 * Project: RapiDHT
 * File: src/kernels.cu
 * Brief: The CUDA kernels for the matrix operations and the Hartley matrices.
 * Author: Volkov Evgeny Aleksandrovich, volkov22dla@yandex.ru
 */

#include "internal/kernels.h"

#include "device_launch_parameters.h"

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

    cudaDeviceSynchronize();
}

template <typename T>
void VectorMatrixMultiplication(const T* A, const T* x, T* y, int N)
{
    int threadsPerBlock = (N > 512) ? 512 : N;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    VectorMatrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, x, y, N);
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
}

template <typename T>
void ScaleOnDevice(T* deviceData, size_t count, T factor)
{
    const int threads = 256;
    const size_t blocks = (count + threads - 1) / threads;

    ScaleKernel<<<static_cast<unsigned int>(blocks), threads>>>(deviceData, count, factor);
    cudaDeviceSynchronize();
}

template <typename T>
void BracewellTransform2D(const T* deviceIn, T* deviceOut, int W, int H)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
        (H + blockDim.y - 1) / blockDim.y);

    BracewellTransform2DKernel<<<gridDim, blockDim>>>(deviceIn, deviceOut, W, H);
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
}

void InitializeHartleyMatrix(double* deviceMatrix, size_t height)
{
    dim3 block(16, 16);
    dim3 grid((height + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    InitializeHartleyMatrixKernel<<<grid, block>>>(deviceMatrix, height);
    cudaDeviceSynchronize();
}

void InitializeHartleyMatrix(float* deviceMatrix, size_t height)
{
    dim3 block(16, 16);
    dim3 grid((height + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    InitializeHartleyMatrixKernel<<<grid, block>>>(deviceMatrix, height);
    cudaDeviceSynchronize();
}

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
