/*
 * Project: RapiDHT
 * File: kernel.cu
 * Brief: CUDA-ядра и хост-обёртки для матричных операций и преобразования Хартли.
 * Author: Волков Евгений Александрович, volkov22dla@yandex.ru
 */

#include "kernel.h"
#include "device_launch_parameters.h"
#include "dev_array.h"


//#ifndef TILE_DIM
//#define TILE_DIM 32
//#endif
//#ifndef BLOCK_ROWS
//#define BLOCK_ROWS 8
//#endif

// ------------------------------ Kernels ------------------------------

namespace RapiDHT {

template <typename T>
__global__ void transpose_YZ_kernel(const T* __restrict__ in, T* __restrict__ out, int W, int H, int D) {
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
__global__ void permute_ZXY_simple_kernel(const T* __restrict__ in, T* __restrict__ out, int W, int H, int D) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= W || y >= H || z >= D)
		return;

	// исходный индекс (row-major, x fastest)
	size_t in_idx = (size_t)z * (W * (size_t)H) + (size_t)y * W + x;
	size_t out_idx = (size_t)y * ((size_t)D * W) + (size_t)x * D + z;

	out[out_idx] = in[in_idx];
}

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

template <typename T>
__global__ void MatrixMultiplicationKernelShared(const T* __restrict__ A, const T* __restrict__ B,
												 T* __restrict__ C, int M, int K, int N) {
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
__global__ void MatrixVectorMultKernel(const T* A, const T* x, T* y, int N) {
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
__global__ void MatrixTransposeKernel(T* A, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N && i < j) {
		T tmp = A[i * N + j];
		A[i * N + j] = A[j * N + i];
		A[j * N + i] = tmp;
	}
}

template <typename T>
__global__ void MatrixTransposeKernel(const T* A, T* B, int rows, int cols) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows) {
		B[col * rows + row] = A[row * cols + col];
	}
}

template <typename T>
__global__ void BracewellKernel(T* A, int N) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= N || v >= N) return;

	int iu = u == 0 ? 0 : N - u;
	int iv = v == 0 ? 0 : N - v;

	T Auv = A[u * N + v];
	T Buv = A[u * N + iv];
	T Cuv = A[iu * N + v];
	T Duv = A[iu * N + iv];

	A[u * N + v] = (T)0.5 * (Auv + Buv + Cuv - Duv);
}

template <typename T>
__global__ void MatrixMultiplication3D_Z_Kernel(
	const T* d_input,
	const T* d_transformZ,
	T* d_output,
	int W, int H, int D) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= W || y >= H) return;

	for (int z_out = 0; z_out < D; ++z_out) {
		T sum = 0.0;
		for (int z_in = 0; z_in < D; ++z_in) {
			sum += d_input[z_in * H * W + y * W + x] * d_transformZ[z_out * D + z_in];
		}
		d_output[z_out * H * W + y * W + x] = sum;
	}
}

template <typename T>
__global__ void BracewellTransform3D_Kernel(T* A, int W, int H, int D) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= W || y >= H || z >= D) {
		return;
	}

	int xm = x == 0 ? 0 : W - x;
	int ym = y == 0 ? 0 : H - y;
	int zm = z == 0 ? 0 : D - z;

	T Axyz = A[z * H * W + y * W + x];
	T Bxyz = A[z * H * W + y * W + xm];
	T Cxyz = A[z * H * W + ym * W + x];
	T Dxyz = A[z * H * W + ym * W + xm];
	T Exyz = A[zm * H * W + y * W + x];
	T Fxyz = A[zm * H * W + y * W + xm];
	T Gxyz = A[zm * H * W + ym * W + x];
	T Hxyz = A[zm * H * W + ym * W + xm];

	A[z * H * W + y * W + x] = 0.5 * (Axyz + Bxyz + Cxyz - Dxyz + Exyz + Fxyz + Gxyz - Hxyz);
}

__global__ void InitializeHartleyMatrixKernel(double* kernel, size_t height) {
	size_t k = blockIdx.y * blockDim.y + threadIdx.y;
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;

	if (k < height && j < height) {
		const double m_pi = 3.14159265358979323846;
		kernel[k * height + j] = cos(2.0 * m_pi * k * j / height) + sin(2.0 * m_pi * k * j / height);
	}
}

__global__ void InitializeHartleyMatrixKernel(float* kernel, size_t height) {
	size_t k = blockIdx.y * blockDim.y + threadIdx.y;
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;

	if (k < height && j < height) {
		const float m_pi = 3.14159265358979323846f;
		kernel[k * height + j] = cosf(2.0f * m_pi * k * j / height) + sinf(2.0f * m_pi * k * j / height);
	}
}

// ------------------------------ Host Wrappers ------------------------------

template <typename T> void transpose_YZ_cuda(const T* d_in, T* d_out, int W, int H, int D) {
	dim3 block(8, 8, 8);
	dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, (D + block.z - 1) / block.z);

	transpose_YZ_kernel<T><<<grid, block>>>(d_in, d_out, W, H, D);
	cudaDeviceSynchronize();
}

template <typename T> void permute_ZXY_simple(const T* d_in, T* d_out, int W, int H, int D) {
	dim3 block(8, 8, 8);
	dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, (D + block.z - 1) / block.z);

	permute_ZXY_simple_kernel<T><<<grid, block>>>(d_in, d_out, W, H, D);
	cudaDeviceSynchronize();
}

template <typename T>
void MatrixMultiplication(const T* A, const T* B, T* C, int M, int K, int N) {
	const int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid(
		(N + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(M + BLOCK_SIZE - 1) / BLOCK_SIZE);

	//MatrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, M, K, N);
	MatrixMultiplicationKernelShared<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);

	cudaDeviceSynchronize();
}

template <typename T>
void MatrixMultiplication(const T* A, const T* B, T* C, int N) {
	MatrixMultiplication(A, B, C, N, N, N);
}

template <typename T>
void VectorMatrixMultiplication(const T* A, const T* x, T* y, int N) {
	int threadsPerBlock = (N > 512) ? 512 : N;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	MatrixVectorMultKernel << < blocksPerGrid, threadsPerBlock >> > (A, x, y, N);
	cudaDeviceSynchronize();
}

// rows, cols - целевые (размеры матрицы B)
template <typename T>
void MatrixTranspose(const T* A, T* B, int rows, int cols) {
	int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

	MatrixTransposeKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, rows, cols);
	cudaDeviceSynchronize();
}

template <typename T>
void MatrixTranspose(T* A, int N) {
	int BLOCK_SIZE = 16;  // оптимальный размер блока
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	MatrixTransposeKernel << < blocksPerGrid, threadsPerBlock >> > (A, N);
	cudaDeviceSynchronize();
}

template <typename T>
void BracewellTransform2D(T* A, int N) {
	dim3 blockDim(16, 16);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
		(N + blockDim.y - 1) / blockDim.y);

	BracewellKernel << <gridDim, blockDim >> > (A, N);
	cudaDeviceSynchronize();
}

template <typename T>
void MatrixMultiplication3D_Z(const T* d_input, const T* d_transformZ, T* d_output, int W, int H, int D) {
	dim3 blockDim(16, 16);
	dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
		(H + blockDim.y - 1) / blockDim.y);

	MatrixMultiplication3D_Z_Kernel << <gridDim, blockDim >> > (d_input, d_transformZ, d_output, W, H, D);
	cudaDeviceSynchronize();
}

template <typename T>
void BracewellTransform3D(T* d_A, int W, int H, int D) {
	dim3 blockDim(8, 8, 8);  // можно подбирать под вашу карту
	dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
		(H + blockDim.y - 1) / blockDim.y,
		(D + blockDim.z - 1) / blockDim.z);

	BracewellTransform3D_Kernel << <gridDim, blockDim >> > (d_A, W, H, D);
	cudaDeviceSynchronize();
}

void InitializeHartleyMatrix(double* dKernel, size_t height) {
	dim3 block(16, 16);
	dim3 grid((height + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	InitializeHartleyMatrixKernel<<<grid, block>>>(dKernel, height);
	cudaDeviceSynchronize();
}

void InitializeHartleyMatrix(float* dKernel, size_t height) {
	dim3 block(16, 16);
	dim3 grid((height + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	InitializeHartleyMatrixKernel<<<grid, block>>>(dKernel, height);
	cudaDeviceSynchronize();
}

template void transpose_YZ_cuda<float>(const float* d_in, float* d_out, int W, int H, int D);
template void transpose_YZ_cuda<double>(const double* d_in, double* d_out, int W, int H, int D);

template void permute_ZXY_simple<float>(const float* d_in, float* d_out, int W, int H, int D);
template void permute_ZXY_simple<double>(const double* d_in, double* d_out, int W, int H, int D);

// 3D матричные умножения
template void MatrixMultiplication3D_Z<float>(const float* d_input, const float* d_transformZ, float* d_output, int W,
											  int H, int D);
template void MatrixMultiplication3D_Z<double>(const double* d_input, const double* d_transformZ, double* d_output,
											   int W, int H, int D);

// Общая матричная операция
template void MatrixMultiplication<float>(const float* A, const float* B, float* C, int M, int K, int N);
template void MatrixMultiplication<double>(const double* A, const double* B, double* C, int M, int K, int N);

// Транспонирование
template void MatrixTranspose<float>(const float* A, float* B, int rows, int cols);
template void MatrixTranspose<double>(const double* A, double* B, int rows, int cols);

// Умножение вектор-матрица
template void VectorMatrixMultiplication<float>(const float* A, const float* x, float* y, int N);
template void VectorMatrixMultiplication<double>(const double* A, const double* x, double* y, int N);

// 3D преобразование Брэсвелла
template void BracewellTransform3D<float>(float* d_A, int W, int H, int D);
template void BracewellTransform3D<double>(double* d_A, int W, int H, int D);

} // namespace RapiDHT
