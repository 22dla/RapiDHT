/*
 * Project: RapiDHT
 * File: rapidht.cpp
 * Brief: Реализация ND-преобразований Хартли (CPU/OpenMP и GPU/CUDA), транспонирования и Bracewell.
 * Author: Волков Евгений Александрович, volkov22dla@yandex.ru
 */

#include <complex>
#include <mpi.h>
#include <cublas_v2.h>
#include <omp.h>

#include "kernel.h"
#include "rapidht.h"
#include "utilities.h"

namespace RapiDHT {

template <typename T>
HartleyTransform<T>::HartleyTransform(size_t width, size_t height, size_t depth, Modes mode) : _mode(mode) {
	PROFILE_FUNCTION();

	if (width == 0) {
		throw std::invalid_argument("Width must be positive.");
	}
	if (height == 0 && depth > 0) {
		throw std::invalid_argument("If height is zero, depth must also be zero.");
	}

	_dims = { width, height, depth };

	// Preparation to 1D transforms
	if (_mode == Modes::CPU || _mode == Modes::RFFT) {
		for (size_t i = 0; i < _bitReversedIndices.size(); ++i) {
			_bitReversedIndices[i].resize(_dims[i]);
			BitReverse(_bitReversedIndices[i]);
		}
	}
	if (_mode == Modes::GPU) {
		_dTransformMatrices[static_cast<size_t>(Direction::Y)].resize(Width() * Width());
		_dTransformMatrices[static_cast<size_t>(Direction::X)].resize(Height() * Height());
		_dTransformMatrices[static_cast<size_t>(Direction::Z)].resize(Depth() * Depth());

		InitializeHartleyMatrix(_dTransformMatrices[static_cast<size_t>(Direction::X)].getData(), Height());
		InitializeHartleyMatrix(_dTransformMatrices[static_cast<size_t>(Direction::Y)].getData(), Width());
		InitializeHartleyMatrix(_dTransformMatrices[static_cast<size_t>(Direction::Z)].getData(), Depth());
	}
}

template <typename T>
void HartleyTransform<T>::ForwardTransform(T* data) {
	PROFILE_FUNCTION();

	bool is1D = (Height() == 0 && Depth() == 0);
	bool is2D = (Height() > 0 && Depth() == 0);
	bool is3D = (Depth() > 0);

	// Проверяем, инициализирован ли MPI
	int mpiInitialized = 0;
	MPI_Initialized(&mpiInitialized);

	int rank = 0, size = 1;
	if (mpiInitialized) {
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
	}

	if (is1D || is2D) {
		// Для 1D и 2D нет смысла в MPI
		switch (_mode) {
		case Modes::CPU:
			if (is1D) {
				FDHT1D(data);
			} else {
				FDHT2D(data);
			}
			break;
		case Modes::GPU:
			if (is1D) {
				DHT1DCuda(data);
			} else {
				DHT2DCuda(data);
			}
			break;
		case Modes::RFFT:
			if (is1D) {
				RealFFT1D(data);
			} else {
				FDHT2D(data);
			}

			break;
		}
		if (mpiInitialized) MPI_Barrier(MPI_COMM_WORLD);
		return;
	}

	// 3D case: делим по Z между процессами, если MPI включен
	size_t depthPerProc = Depth() / size;
	size_t remainder = Depth() % size;
	size_t offset = rank * depthPerProc + std::min(static_cast<size_t>(rank), remainder);
	depthPerProc += (rank < remainder) ? 1 : 0;

	T* localData = data + offset * Width() * Height();

	switch (_mode) {
	case Modes::CPU:
		FDHT3D(localData);
		break;
	case Modes::GPU:
		DHT3DCuda(localData);
		break;
	case Modes::RFFT:
		FDHT3D(localData);
		break;
	}

	// Сбор данных только если MPI активен
	if (mpiInitialized) {
		std::vector<int> sendcounts(size);
		std::vector<int> displs(size);
		int offs = 0;
		for (int i = 0; i < size; ++i) {
			sendcounts[i] = static_cast<int>((Depth() / size + (i < remainder ? 1 : 0)) * Width() * Height());
			displs[i] = offs;
			offs += sendcounts[i];
		}
		MPI_Allgatherv(localData, sendcounts[rank], MPI_DOUBLE,
			data, sendcounts.data(), displs.data(), MPI_DOUBLE,
			MPI_COMM_WORLD);
	}
}

template <typename T>
void HartleyTransform<T>::InverseTransform(T* data) {
	PROFILE_FUNCTION();

	bool is1D = (Height() == 0 && Depth() == 0);
	bool is2D = (Height() > 0 && Depth() == 0);
	bool is3D = (Depth() > 0);

	// Проверяем, инициализирован ли MPI
	int mpiInitialized = 0;
	MPI_Initialized(&mpiInitialized);

	int rank = 0, size = 1;
	if (mpiInitialized) {
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
	}

	// Сначала выполняем прямое преобразование
	ForwardTransform(data);

	// Общий размер данных
	size_t totalSize = Width();
	if (Height() > 0) {
		totalSize *= Height();
	}
	if (Depth() > 0) {
		totalSize *= Depth();
	}

	auto denominator = 1.0 / static_cast<double>(totalSize);

	if (is1D || is2D) {
		// Масштабируем полностью только на rank=0
		for (size_t i = 0; i < totalSize; ++i) {
			data[i] *= denominator;
		}
		if (mpiInitialized) MPI_Barrier(MPI_COMM_WORLD);
		return;
	}

	// 3D case: делим по Z между процессами
	size_t depthPerProc = Depth() / size;
	size_t remainder = Depth() % size;
	size_t offset = rank * depthPerProc + std::min(static_cast<size_t>(rank), remainder);
	depthPerProc += (rank < remainder) ? 1 : 0;

	size_t localSize = depthPerProc * Width() * Height();
	T* localData = data + offset * Width() * Height();

	// Масштабируем локальный блок
	for (size_t i = 0; i < localSize; ++i) {
		localData[i] *= denominator;
	}

	// Синхронизация процессов
	if (mpiInitialized) {
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

template <typename T>
void HartleyTransform<T>::BitReverse(std::vector<size_t>& indices) {
	PROFILE_FUNCTION();

	if (indices.empty()) {
		return;
	}

	const size_t n = indices.size();
	const int kLog2n = static_cast<int>(std::log2(n));

	indices[0] = 0;
	for (size_t j = 1; j < n; ++j) {
		size_t reversed = 0;
		size_t temp = j;
		for (int i = 0; i < kLog2n; ++i) {
			if (temp & 1) {
				reversed |= 1 << (kLog2n - 1 - i);
			}
			temp >>= 1;
		}
		indices[j] = reversed;
	}
}

template <typename T>
void HartleyTransform<T>::InitializeKernelHost(std::vector<T>& kernel, size_t height) {
	PROFILE_FUNCTION();

	kernel.resize(height * height);
	const auto m_pi = std::acos(-1);

	// Initialize the matrice on the host
	for (size_t k = 0; k < height; ++k) {
		for (size_t j = 0; j < height; ++j) {
			kernel[k * height + j] = std::cos(2 * m_pi * k * j / height) + std::sin(2 * m_pi * k * j / height);
		}
	}
}

// test function
template <typename T>
std::vector<T> HartleyTransform<T>::DHT1D(const std::vector<T>& a, const std::vector<T>& kernel) {
	PROFILE_FUNCTION();

	std::vector<T> result(a.size());
	for (size_t i = 0; i < a.size(); i++) {
		for (size_t j = 0; j < a.size(); j++) {
			result[i] += (kernel[i * a.size() + j] * a[j]);
		}
	}
	return result;
}

template <typename T>
void HartleyTransform<T>::Transpose(std::vector<std::vector<T>>& matrix) {
	PROFILE_FUNCTION();

#pragma omp parallel for
	for (int i = 0; i < matrix.size(); ++i) {
	#pragma omp parallel for
		for (int j = i + 1; j < matrix[0].size(); ++j) {
			std::swap(matrix[i][j], matrix[j][i]);
		}
	}
}

template <typename T>
void HartleyTransform<T>::TransposeSimple(T* matrix, size_t width, size_t height) {
	PROFILE_FUNCTION();

	if (matrix == nullptr) {
		throw std::invalid_argument("The pointer to matrix is null.");
	}

	if (width == height) {
	#pragma omp parallel for
		// Square matrix
		for (int i = 0; i < width; ++i) {
		#pragma omp parallel for
			for (int j = i + 1; j < height; ++j) {
				std::swap(matrix[i * height + j], matrix[j * height + i]);
			}
		}
	} else {
		// Non-square matrix
		std::vector<T> transposed(width * height);
	#pragma omp parallel for
		for (int i = 0; i < width; ++i) {
		#pragma omp parallel for
			for (int j = 0; j < height; ++j) {
				transposed[j * width + i] = matrix[i * height + j];
			}
		}
		//std::memcpy(matrix, transposed.data(), sizeof(double) * width * height);
		//require to check
		std::copy(transposed.data(), transposed.data() + (width * height), matrix);
	}
}

template <typename T>
void HartleyTransform<T>::Series1D(T* data, Direction direction) {
	PROFILE_FUNCTION();

	if (data == nullptr) {
		throw std::invalid_argument("The pointer to image is null.");
	}

	size_t M1 = 0, M2 = 0;
	switch (direction) {
	case Direction::X:
		M1 = Height();
		M2 = (Depth() == 0 ? 1 : Depth());
		break;
	case Direction::Y:
		M1 = Width();
		M2 = (Depth() == 0 ? 1 : Depth());
		break;
	case Direction::Z:
		M1 = Width();
		M2 = Height();
		break;
	default:
		throw std::invalid_argument("Invalid direction");
	}

	if (_mode == Modes::CPU) {
	#pragma omp parallel for
		for (int i = 0; i < M1; ++i) {
			for (size_t j = 0; j < M2; ++j) {
				auto index = AxisIndex(0, i, j, direction);
				FDHT1D(data + index, direction);
			}
		}
		return;
	}
	//if (_mode == Modes::RFFT) {
	//#pragma omp parallel for
	//	for (int i = 0; i < Width(); ++i) {
	//		RealFFT1D(image_ptr + i * Height(), direction);
	//	}
	//	return;
	//}
}

template <typename T>
void HartleyTransform<T>::BracewellTransform2DCPU(T* image_ptr) {
	PROFILE_FUNCTION();

	std::vector<T> H(Width() * Height(), 0.0);
#pragma omp parallel for
	for (int i = 0; i < Width(); ++i) {
		for (int j = 0; j < Height(); ++j) {
			const T A = image_ptr[i * Height() + j];
			const T B = (i > 0 && j > 0) ? image_ptr[i * Height() + (Height() - j)] : A;
			const T C = (i > 0 && j > 0) ? image_ptr[(Width() - i) * Height() + j] : A;
			const T D = (i > 0 && j > 0) ? image_ptr[(Width() - i) * Height() + (Height() - j)] : A;
			H[i * Height() + j] = (A + B + C - D) / static_cast<T>(2);
		}
	}

	std::copy(H.begin(), H.end(), image_ptr);
}

template <typename T>
void HartleyTransform<T>::BracewellTransform3DCPU(T* volumePtr) {
	PROFILE_FUNCTION();
	int W = Width();
	int H = Height();
	int D = Depth();

	std::vector<T> result(W * H * D, T(0));

	#pragma omp parallel for collapse(3)
	for (int x = 0; x < W; ++x) {
		const int xm = (x > 0) ? (W - x) : 0;
		for (int y = 0; y < H; ++y) {
			const int ym = (y > 0) ? (H - y) : 0;
			for (int z = 0; z < D; ++z) {
				const int zm = (z > 0) ? (D - z) : 0;

				const T A = volumePtr[LinearIndex(xm, y, z)];	// flip X
				const T B = volumePtr[LinearIndex(x, ym, z)];	// flip Y
				const T C = volumePtr[LinearIndex(x, y, zm)];	// flip Z
				const T D_ = volumePtr[LinearIndex(xm, ym, zm)];// flip all

				result[LinearIndex(x, y, z)] = (A + B + C - D_) / static_cast<T>(2);
			}
		}
	}

	std::copy(result.begin(), result.end(), volumePtr);
}

template <typename T>
void HartleyTransform<T>::FDHT1D(T* data, Direction direction) {
	if (data == nullptr) {
		throw std::invalid_argument("The pointer to vector is null.");
	}

	const size_t n = Length(direction);

	// Indices for bit reversal operation
	// and length of vector depending of direction
	if (n < 0) {
		std::cout << "Error: length must be non-negative." << std::endl;
		throw std::invalid_argument("Error: length must be non-negative.");
	}
	// Check that length is power of 2
	if (std::ceil(std::log2(n)) != std::floor(std::log2(n))) {
		std::cout << "Error: length must be a power of two." << std::endl;
		throw std::invalid_argument("Error: length must be a power of two.");
	}

    // временный буфер
	std::vector<T> vec(n);

	// собрать данные в буфер
	for (size_t idx = 0; idx < n; ++idx) {
		vec[idx] = data[AxisIndex(idx, 0, 0, direction)];
	}

	for (size_t i = 1; i < n; ++i) {
		auto j = BitReversedIndex(direction, i);
		if (j > i) {
			std::swap(vec[i], vec[j]);
		}
	}

	// FHT for 1rd axis
	const auto kLog2n = static_cast<size_t>(std::log2(n));
	const T kPi = static_cast<T>(std::acos(-1));

	// Main cicle
	for (size_t s = 1; s <= kLog2n; ++s) {
		const auto m = size_t(1) << s;
		const auto m2 = m / 2;
		const auto m4 = m / 4;

		for (size_t r = 0; r <= n - m; r = r + m) {
			for (size_t j = 1; j < m4; ++j) {
				int k = m2 - j;
				const auto u = vec[r + m2 + j];
				const auto v = vec[r + m2 + k];
				const auto cosVal = std::cos(j * kPi / m2);
				const auto sinVal = std::sin(j * kPi / m2);
				vec[r + m2 + j] = u * cosVal + v * sinVal;
				vec[r + m2 + k] = u * sinVal - v * cosVal;
			}
			for (size_t j = 0; j < m2; ++j) {
				const auto u = vec[r + j];
				const auto v = vec[r + j + m2];
				vec[r + j] = u + v;
				vec[r + j + m2] = u - v;
			}
		}
	}

	// записать обратно
	for (size_t idx = 0; idx < n; ++idx) {
		data[AxisIndex(idx, 0, 0, direction)] = vec[idx];
	}
}

template <typename T>
void HartleyTransform<T>::FDHT2D(T* image_ptr) {
	PROFILE_FUNCTION();

	if (image_ptr == nullptr) {
		std::cout << "The pointer to image is null." << std::endl;
		throw std::invalid_argument("The pointer to image is null.");
	}

	Series1D(image_ptr, Direction::X);
	Series1D(image_ptr, Direction::Y);

	BracewellTransform2DCPU(image_ptr);
}

template <typename T>
void HartleyTransform<T>::FDHT3D(T* volume_ptr) {
	PROFILE_FUNCTION();

	if (volume_ptr == nullptr) {
		std::cout << "The pointer to volume is null." << std::endl;
		throw std::invalid_argument("The pointer to volume is null.");
	}

	// 1D transforms along X, Y, Z dimensions
	Series1D(volume_ptr, Direction::X);
	Series1D(volume_ptr, Direction::Y);
	Series1D(volume_ptr, Direction::Z);

	// Bracewell 3D
	BracewellTransform3DCPU(volume_ptr);
}

template <typename T>
void HartleyTransform<T>::RealFFT1D(T* vec, Direction direction) {
	PROFILE_FUNCTION();

	if (vec == nullptr) {
		std::cout << "The pointer to vector is null." << std::endl;
		throw std::invalid_argument("The pointer to vector is null.");
	}

	if (Length(direction) < 0) {
		std::cout << "Error: length must be non-negative." << std::endl;
		throw std::invalid_argument("Error: length must be non-negative.");
	}
	// Check that length is power of 2
	if (std::ceil(std::log2(Length(direction))) != std::floor(std::log2(Length(direction)))) {
		std::cout << "Error: length must be a power of two." << std::endl;
		throw std::invalid_argument("Error: length must be a power of two.");
	}

	// RealFFT
	std::vector<std::complex<T>> x(Length(direction));
	for (size_t i = 0; i < Length(direction); i++) {
		x[i] = std::complex<T>(vec[i], 0);
	}
	size_t k = Length(direction);
	size_t n;
	const T thetaT = static_cast<T>(std::acos(-1)) / Length(direction);
	std::complex<T> phiT = std::complex<T>(std::cos(thetaT), -std::sin(thetaT)), TTT;
	while (k > 1) {
		n = k;
		k >>= 1;
		phiT = phiT * phiT;
		TTT = 1.0L;
		for (size_t l = 0; l < k; l++) {
			for (size_t a = l; a < Length(direction); a += n) {
				size_t b = a + k;
				std::complex<T> t = x[a] - x[b];
				x[a] += x[b];
				x[b] = t * TTT;
			}
			TTT *= phiT;
		}
	}
	// Decimate
	size_t m = (size_t)log2(Length(direction));
	for (size_t a = 0; a < Length(direction); a++) {
		size_t b = a;
		// Reverse bits
		b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
		b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
		b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
		b = ((b >> 16) | (b << 16)) >> (32 - m);
		if (b > a) {
			std::complex<T> t = x[a];
			x[a] = x[b];
			x[b] = t;
		}
	}

	for (int i = 0; i < Length(direction); i++) {
		vec[i] = x[i].real();
	}
}

template <typename T>
void HartleyTransform<T>::DHT1DCuda(T* h_x) {
	PROFILE_FUNCTION();

	// Allocate memory on the device
	dev_array<T> d_x(Width());			// input vector
	dev_array<T> d_y(Width());			// output vector

	//write_matrix_to_csv(h_A.data(), length, length, "matrix.csv");

	// transfer CPU -> GPU
	d_x.set(h_x, Width());

	VectorMatrixMultiplication(_dTransformMatrices[static_cast<size_t>(Direction::Y)].getData(), d_x.getData(),
							   d_y.getData(), Width());
	// transfer GPU -> CPU
	d_y.get(h_x, Width());
}

template <typename T>
void HartleyTransform<T>::DHT2DCuda(T* h_X) {
	PROFILE_FUNCTION();

	// TODO: Перевести умножения матриц/векторов на cuBLAS для повышения производительности
	// (cublasDgemm/cublasDgemv). Текущая реализация использует собственные CUDA-ядра.

	// Allocate memory on the device
	dev_array<T> d_X(Width() * Height()); // one slice
	dev_array<T> d_Y(Width() * Height()); // one slice

    // Events
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	float ms = 0.0f;

	// ---------------- CPU -> GPU ----------------
	CUDA_CHECK(cudaEventRecord(start));
	d_X.set(&h_X[0], Width() * Height());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	std::cout << "Memcpy H2D:\t\t\t" << ElapsedMsGPU(start, stop) << " ms\n";

	// ---------------- MatrixMultiplication X ----------------
	CUDA_CHECK(cudaEventRecord(start));
	MatrixMultiplication(d_X.getData(), _dTransformMatrices[static_cast<size_t>(Direction::X)].getData(), d_Y.getData(),
						 Height(), Width(), Width());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	std::cout << "MatrixMultiplication (X):\t" << ElapsedMsGPU(start, stop) << " ms\n";

    // ---------------- Transpose ----------------
	CUDA_CHECK(cudaEventRecord(start));
	MatrixTranspose(d_Y.getData(), d_X.getData(), Height(), Width());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	std::cout << "Transpose (after X):\t\t" << ElapsedMsGPU(start, stop) << " ms\n";

	// ---------------- MatrixMultiplication Y ----------------
	CUDA_CHECK(cudaEventRecord(start));
	MatrixMultiplication(d_X.getData(), _dTransformMatrices[static_cast<size_t>(Direction::Y)].getData(), d_Y.getData(),
						 Width(), Height(), Height());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	std::cout << "MatrixMultiplication (Y):\t" << ElapsedMsGPU(start, stop) << " ms\n";

    // ---------------- Transpose ----------------
	CUDA_CHECK(cudaEventRecord(start));
	MatrixTranspose(d_Y.getData(), d_X.getData(), Width(), Height());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	std::cout << "Transpose (after Y):\t\t" << ElapsedMsGPU(start, stop) << " ms\n";

	// Bracewell
	//BracewellTransform2D(d_X.getData(), Width());

    // ---------------- GPU -> CPU ----------------
	CUDA_CHECK(cudaEventRecord(start));
	d_X.get(&h_X[0], Width() * Height());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	std::cout << "Memcpy D2H:\t\t\t" << ElapsedMsGPU(start, stop) << " ms\n";

	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	std::cout << std::endl << std::endl << std::endl;
}

namespace {
	template <typename T> struct CublasGemmStridedBatched;

	template <> struct CublasGemmStridedBatched<float> {
		static cublasStatus_t call(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
								   int n, int k, const float* alpha, const float* A, int lda, long long int strideA,
								   const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc,
								   long long int strideC, int batchCount) {
			return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB,
											 beta, C, ldc, strideC, batchCount);
		}
	};

	template <> struct CublasGemmStridedBatched<double> {
		static cublasStatus_t call(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
								   int n, int k, const double* alpha, const double* A, int lda, long long int strideA,
								   const double* B, int ldb, long long int strideB, const double* beta, double* C,
								   int ldc, long long int strideC, int batchCount) {
			return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB,
											 beta, C, ldc, strideC, batchCount);
		}
	};

	template <typename T> struct CublasGeam;

	template <> struct CublasGeam<float> {
		static cublasStatus_t call(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
								   int n, const float* alpha, const float* A, int lda, const float* beta,
								   const float* B, int ldb, float* C, int ldc) {
			return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
		}
	};

	template <> struct CublasGeam<double> {
		static cublasStatus_t call(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
								   int n, const double* alpha, const double* A, int lda, const double* beta,
								   const double* B, int ldb, double* C, int ldc) {
			return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
		}
	};
}

template <typename T> void HartleyTransform<T>::DHT3DCuda(T* h_X) {
	PROFILE_FUNCTION();

	auto W = Width();
	auto H = Height();
	auto D = Depth();

	size_t totalSize = W * H * D;

	// Allocate device memory
	dev_array<T> d_X(totalSize);
	dev_array<T> d_Y(totalSize);

	// copy CPU -> GPU
	d_X.set(h_X, totalSize);

	cublasHandle_t handle;
	cublasCreate(&handle);
	const T alpha = 1.0;
	const T beta = 0.0;

	std::vector<T> original_data(W * H * D);
	d_X.get(original_data.data(), W * H * D);

	// -------------------------------
	// Приводим к column major
	// -------------------------------
	for (size_t z = 0; z < D; ++z) {
		CublasGeam<T>::call(handle, CUBLAS_OP_T, CUBLAS_OP_N, H, W, &alpha, d_X.getData() + z*W*H, W, &beta,
							nullptr, H, d_Y.getData() + z * W * H, H);
	}

	// -------------------------------
	// 1D Hartley along Y (batched GEMM)
	// -------------------------------
	{
		int m = H;
		int n = W;
		int k = W;
		int lda = H;
		int ldb = W;
		int ldc = H;

		long long int strideA = H * W;
		long long int strideB = 0;
		long long int strideC = H * W;

		int batchCount = D;

		CublasGemmStridedBatched<T>::call(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_Y.getData(), lda,
										  strideA, _dTransformMatrices[(size_t)Direction::Y].getData(), ldb, strideB,
										  &beta, d_X.getData(), ldc, strideC, batchCount);
	}

	// Транспонируем
	for (size_t z = 0; z < D; ++z) {
		CublasGeam<T>::call(handle, CUBLAS_OP_T, CUBLAS_OP_N, W, H, &alpha, d_X.getData() + z * W * H, H, &beta,
							nullptr, W, d_Y.getData() + z * W * H, W);
	}

	// -------------------------------
	// 1D Hartley along X (batched GEMM)
	// -------------------------------
	{
		int m = W;
		int n = H;
		int k = H;
		int lda = W;
		int ldb = H;
		int ldc = W;

		long long int strideA = H * W;
		long long int strideB = 0;
		long long int strideC = H * W;

		int batchCount = D;

		CublasGemmStridedBatched<T>::call(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_Y.getData(), lda,
										  strideA, _dTransformMatrices[(size_t)Direction::X].getData(), ldb, strideB,
										  &beta, d_X.getData(), ldc, strideC, batchCount);
	}

	// Транспонируем
	for (size_t z = 0; z < D; ++z) {
		CublasGeam<T>::call(handle, CUBLAS_OP_T, CUBLAS_OP_N, W, H, &alpha, d_X.getData() + z * W * H, H, &beta,
							nullptr, W, d_Y.getData() + z * W * H, W);
	}

	{
		int m = W;
		int n = D;
		int k = D;
		int lda = W;
		int ldb = D;
		int ldc = W;

		long long int strideA = D * W;
		long long int strideB = 0;
		long long int strideC = D * W;

		int batchCount = H;

		CublasGemmStridedBatched<T>::call(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_Y.getData(), lda,
										  strideA, _dTransformMatrices[(size_t)Direction::Z].getData(), ldb, strideB,
										  &beta, d_X.getData(), ldc, strideC, batchCount);
	}

	// Транспонируем
	{
		permute_ZXY_simple(d_X.getData(), d_Y.getData(), W, H, D);
	}

	// -------------------------------
	// Bracewell 3D (осталось так, как у тебя)
	// -------------------------------
	//BracewellTransform3D(d_Y.getData(), W, H, D);

	// copy GPU -> CPU
	d_Y.get(h_X, totalSize);

	cublasDestroy(handle);
	cudaDeviceSynchronize();
}

template class HartleyTransform<float>;
template class HartleyTransform<double>;

} // namespace RapiDHT
