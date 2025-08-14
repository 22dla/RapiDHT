#include "kernel.h"
#include "device_launch_parameters.h"
#include "dev_array.h"

// ------------------------------ Kernels ------------------------------

__global__ void matrixMultiplicationKernel(double* A, double* B, double* C, int N)
{
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    if (ROW < N && COL < N) {
        double tmpSum = 0.0;
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
        C[ROW * N + COL] = tmpSum;
    }
}

__global__ void matrixVectorMultKernel(double* A, double* x, double* y, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

__global__ void matrixTransposeKernel(double* A, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N && i < j) {
        double tmp = A[i * N + j];
        A[i * N + j] = A[j * N + i];
        A[j * N + i] = tmp;
    }
}

// ------------------------------ Host Wrappers ------------------------------

void matrixMultiplication(double* A, double* B, double* C, const int N)
{
    int BLOCK_SIZE = 16;  // оптимальный размер блока
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMultiplicationKernel <<< blocksPerGrid, threadsPerBlock >>> (A, B, C, N);
    cudaDeviceSynchronize();
}

void vectorMatrixMultiplication(double* A, double* x, double* y, const int N)
{
    int threadsPerBlock = (N > 512) ? 512 : N;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    matrixVectorMultKernel <<< blocksPerGrid, threadsPerBlock >>> (A, x, y, N);
    cudaDeviceSynchronize();
}

void matrixTranspose(double* A, const int N)
{
    int BLOCK_SIZE = 16;  // оптимальный размер блока
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixTransposeKernel <<< blocksPerGrid, threadsPerBlock >>> (A, N);
    cudaDeviceSynchronize();
}
