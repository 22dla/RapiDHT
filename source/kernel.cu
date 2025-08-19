/*
 * Project: RapiDHT
 * File: kernel.cu
 * Brief: CUDA-ядра и хост-обёртки для матричных операций и преобразования Хартли.
 * Author: Волков Евгений Александрович, volkov22dla@yandex.ru
 */

#include "kernel.h"
#include "device_launch_parameters.h"
#include "dev_array.h"

// ------------------------------ Kernels ------------------------------

namespace RapiDHT {

__global__ void MatrixMultiplicationKernel(const double* A, const double* B, double* C, int M, int K, int N) {
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

__global__ void MatrixVectorMultKernel(const double* A, const double* x, double* y, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		double sum = 0.0;
		for (int j = 0; j < N; j++) {
			sum += A[i * N + j] * x[j];
		}
		y[i] = sum;
	}
}

__global__ void MatrixTransposeKernel(double* A, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N && i < j) {
		double tmp = A[i * N + j];
		A[i * N + j] = A[j * N + i];
		A[j * N + i] = tmp;
	}
}

__global__ void MatrixTransposeKernel(const double* A, double* B, int rows, int cols) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		B[col * rows + row] = A[row * cols + col];
	}
}

__global__ void BracewellKernel(double* A, int N) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= N || v >= N) return;

	int iu = u == 0 ? 0 : N - u;
	int iv = v == 0 ? 0 : N - v;

	double Auv = A[u * N + v];
	double Buv = A[u * N + iv];
	double Cuv = A[iu * N + v];
	double Duv = A[iu * N + iv];

	A[u * N + v] = 0.5 * (Auv + Buv + Cuv - Duv);
}

__global__ void MatrixMultiplication3D_Z_Kernel(
	const double* d_input,
	const double* d_transformZ,
	double* d_output,
	int W, int H, int D) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= W || y >= H) return;

	for (int z_out = 0; z_out < D; ++z_out) {
		double sum = 0.0;
		for (int z_in = 0; z_in < D; ++z_in) {
			sum += d_input[z_in * H * W + y * W + x] * d_transformZ[z_out * D + z_in];
		}
		d_output[z_out * H * W + y * W + x] = sum;
	}
}

__global__ void BracewellTransform3D_Kernel(double* A, int W, int H, int D) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= W || y >= H || z >= D) return;

	int xm = x == 0 ? 0 : W - x;
	int ym = y == 0 ? 0 : H - y;
	int zm = z == 0 ? 0 : D - z;

	double Axyz = A[z * H * W + y * W + x];
	double Bxyz = A[z * H * W + y * W + xm];
	double Cxyz = A[z * H * W + ym * W + x];
	double Dxyz = A[z * H * W + ym * W + xm];
	double Exyz = A[zm * H * W + y * W + x];
	double Fxyz = A[zm * H * W + y * W + xm];
	double Gxyz = A[zm * H * W + ym * W + x];
	double Hxyz = A[zm * H * W + ym * W + xm];

	A[z * H * W + y * W + x] = 0.5 * (Axyz + Bxyz + Cxyz - Dxyz + Exyz + Fxyz + Gxyz - Hxyz);
}

// ------------------------------ Host Wrappers ------------------------------

void MatrixMultiplication(const double* A, const double* B, double* C, int M, int K, int N) {
	const int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid(
		(N + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(M + BLOCK_SIZE - 1) / BLOCK_SIZE);

	MatrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, M, K, N);

	cudaDeviceSynchronize();
}

void MatrixMultiplication(const double* A, const double* B, double* C, int N) {
	MatrixMultiplication(A, B, C, N, N, N);
}

void VectorMatrixMultiplication(const double* A, const double* x, double* y, int N) {
	int threadsPerBlock = (N > 512) ? 512 : N;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	MatrixVectorMultKernel << < blocksPerGrid, threadsPerBlock >> > (A, x, y, N);
	cudaDeviceSynchronize();
}

// rows, cols - целевые (размеры матрицы B)
void MatrixTranspose(const double* A, double* B, int rows, int cols) {
	int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

	MatrixTransposeKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, rows, cols);
	cudaDeviceSynchronize();
}

void MatrixTranspose(double* A, int N) {
	int BLOCK_SIZE = 16;  // оптимальный размер блока
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	MatrixTransposeKernel << < blocksPerGrid, threadsPerBlock >> > (A, N);
	cudaDeviceSynchronize();
}

void BracewellTransform2D(double* A, int N) {
	dim3 blockDim(16, 16);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
		(N + blockDim.y - 1) / blockDim.y);

	BracewellKernel << <gridDim, blockDim >> > (A, N);
	cudaDeviceSynchronize();
}

void MatrixMultiplication3D_Z(const double* d_input, const double* d_transformZ, double* d_output, int W, int H, int D) {
	dim3 blockDim(16, 16);
	dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
		(H + blockDim.y - 1) / blockDim.y);

	MatrixMultiplication3D_Z_Kernel << <gridDim, blockDim >> > (d_input, d_transformZ, d_output, W, H, D);
	cudaDeviceSynchronize();
}

void BracewellTransform3D(double* d_A, int W, int H, int D) {
	dim3 blockDim(8, 8, 8);  // можно подбирать под вашу карту
	dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
		(H + blockDim.y - 1) / blockDim.y,
		(D + blockDim.z - 1) / blockDim.z);

	BracewellTransform3D_Kernel << <gridDim, blockDim >> > (d_A, W, H, D);
	cudaDeviceSynchronize();
}

} // namespace RapiDHT
