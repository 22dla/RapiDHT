#include "kernel.h"
#include "device_launch_parameters.h"
#include "dev_array.h"

// ------------------------------ Kernels ------------------------------

__global__ void matrixMultiplicationKernel(const double* A, const double* B, double* C, int M, int K, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y; // индекс строки C
	int col = blockIdx.x * blockDim.x + threadIdx.x; // индекс столбца C

	if (row < M && col < N) {
		double sum = 0.0;
		for (int t = 0; t < K; ++t) {
			sum += A[row * K + t] * B[t * N + col];
		}
		C[row * N + col] = sum;
	}
}

__global__ void matrixVectorMultKernel(const double* A, const double* x, double* y, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		double sum = 0.0;
		for (int j = 0; j < N; j++) {
			sum += A[i * N + j] * x[j];
		}
		y[i] = sum;
	}
}

__global__ void matrixTransposeKernel(double* A, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N && i < j) {
		double tmp = A[i * N + j];
		A[i * N + j] = A[j * N + i];
		A[j * N + i] = tmp;
	}
}

__global__ void matrixTransposeKernel(const double* A, double* B, int rows, int cols) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		B[col * rows + row] = A[row * cols + col];
	}
}

// ------------------------------ Host Wrappers ------------------------------

void matrixMultiplication(const double* A, const double* B, double* C, int M, int K, int N) {
	const int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid(
		(N + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(M + BLOCK_SIZE - 1) / BLOCK_SIZE);

	matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, M, K, N);

	cudaDeviceSynchronize();
}

void matrixMultiplication(const double* A, const double* B, double* C, int N) {
	matrixMultiplication(A, B, C, N, N, N);
}

void vectorMatrixMultiplication(const double* A, const double* x, double* y, int N) {
	int threadsPerBlock = (N > 512) ? 512 : N;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	matrixVectorMultKernel << < blocksPerGrid, threadsPerBlock >> > (A, x, y, N);
	cudaDeviceSynchronize();
}

// rows, cols - целевые (размеры матрицы B)
void matrixTranspose(const double* A, double* B, int rows, int cols) {
	int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

	matrixTransposeKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, rows, cols);
	cudaDeviceSynchronize();
}

void matrixTranspose(double* A, int N) {
	int BLOCK_SIZE = 16;  // оптимальный размер блока
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	matrixTransposeKernel << < blocksPerGrid, threadsPerBlock >> > (A, N);
	cudaDeviceSynchronize();
}
