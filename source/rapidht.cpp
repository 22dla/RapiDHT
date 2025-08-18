#include <omp.h>
#include <mpi.h>
#include <rapidht.h>
#include <utilities.h>
#include <complex>
#include <kernel.h>
#include <cublas_v2.h>

namespace RapiDHT {

HartleyTransform::HartleyTransform(size_t width, size_t height = 0, size_t depth = 0, Modes mode = Modes::CPU) :_mode(mode) {
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
		// Initialize transform matrices on the host
		InitializeKernelHost(_hTransformMatrices[static_cast<size_t>(Direction::X)], Width());
		InitializeKernelHost(_hTransformMatrices[static_cast<size_t>(Direction::Y)], Height());
		InitializeKernelHost(_hTransformMatrices[static_cast<size_t>(Direction::Z)], Depth());

		// transfer CPU -> GPU
		_dTransformMatrices[static_cast<size_t>(Direction::X)].resize(Width() * Width());
		_dTransformMatrices[static_cast<size_t>(Direction::Y)].resize(Height() * Height());
		_dTransformMatrices[static_cast<size_t>(Direction::Z)].resize(Depth() * Depth());

		_dTransformMatrices[static_cast<size_t>(Direction::X)].set(_hTransformMatrices[static_cast<size_t>(Direction::X)].data(), Width() * Width());
		_dTransformMatrices[static_cast<size_t>(Direction::Y)].set(_hTransformMatrices[static_cast<size_t>(Direction::Y)].data(), Height() * Height());
		_dTransformMatrices[static_cast<size_t>(Direction::Z)].set(_hTransformMatrices[static_cast<size_t>(Direction::Z)].data(), Depth() * Depth());
	}
}

void HartleyTransform::ForwardTransform(double* data) {
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

	double* localData = data + offset * Width() * Height();

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

void HartleyTransform::InverseTransform(double* data) {
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
	double* localData = data + offset * Width() * Height();

	// Масштабируем локальный блок
	for (size_t i = 0; i < localSize; ++i) {
		localData[i] *= denominator;
	}

	// Синхронизация процессов
	if (mpiInitialized) {
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

void HartleyTransform::BitReverse(std::vector<size_t>& indices) {
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

void HartleyTransform::InitializeKernelHost(std::vector<double>& kernel, size_t height) {
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
std::vector<double> HartleyTransform::DHT1D(const std::vector<double>& a, const std::vector<double>& kernel) {
	PROFILE_FUNCTION();

	std::vector<double> result(a.size());
	for (size_t i = 0; i < a.size(); i++) {
		for (size_t j = 0; j < a.size(); j++) {
			result[i] += (kernel[i * a.size() + j] * a[j]);
		}
	}

	// RVO works
	return result;
}

template <typename T>
void HartleyTransform::Transpose(std::vector<std::vector<T>>& matrix) {
	PROFILE_FUNCTION();

	const size_t width = matrix.size();
	const size_t height = matrix[0].size();

#pragma omp parallel for
	for (int i = 0; i < width; ++i) {
	#pragma omp parallel for
		for (int j = i + 1; j < height; ++j) {
			std::swap(matrix[i][j], matrix[j][i]);
		}
	}
}

void HartleyTransform::TransposeSimple(double* matrix, size_t width, size_t height) {
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
		std::vector<double> transposed(width * height);
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

void HartleyTransform::Series1D(double* image_ptr, Direction direction) {
	PROFILE_FUNCTION();

	if (image_ptr == nullptr) {
		throw std::invalid_argument("The pointer to image is null.");
	}

	if (_mode == Modes::CPU) {
	#pragma omp parallel for
		for (int i = 0; i < Width(); ++i) {
			FDHT1D(image_ptr + i * Height(), direction);
		}
		return;
	}
	if (_mode == Modes::RFFT) {
	#pragma omp parallel for
		for (int i = 0; i < Width(); ++i) {
			RealFFT1D(image_ptr + i * Height(), direction);
		}
		return;
	}
}

void HartleyTransform::BracewellTransform2DCPU(double* image_ptr) {
	PROFILE_FUNCTION();

	std::vector<double> H(Width() * Height(), 0.0);
#pragma omp parallel for
	for (int i = 0; i < Width(); ++i) {
		for (int j = 0; j < Height(); ++j) {
			const double A = image_ptr[i * Height() + j];
			const double B = (i > 0 && j > 0) ? image_ptr[i * Height() + (Height() - j)] : A;
			const double C = (i > 0 && j > 0) ? image_ptr[(Width() - i) * Height() + j] : A;
			const double D = (i > 0 && j > 0) ? image_ptr[(Width() - i) * Height() + (Height() - j)] : A;
			H[i * Height() + j] = (A + B + C - D) / 2.0;
		}
	}

	std::copy(H.begin(), H.end(), image_ptr);
}

void HartleyTransform::BracewellTransform3DCPU(double* volumePtr, int W, int H, int D) {
	PROFILE_FUNCTION();

	std::vector<double> result(W * H * D, 0.0);

#pragma omp parallel for collapse(3)
	for (int x = 0; x < W; ++x) {
		for (int y = 0; y < H; ++y) {
			for (int z = 0; z < D; ++z) {
				const int xm = (x > 0) ? W - x : x;
				const int ym = (y > 0) ? H - y : y;
				const int zm = (z > 0) ? D - z : z;

				const double A = volumePtr[z * H * W + y * W + x];
				const double B = volumePtr[z * H * W + y * W + xm];
				const double C = volumePtr[z * H * W + ym * W + x];
				const double D_ = volumePtr[z * H * W + ym * W + xm];
				const double E = volumePtr[zm * H * W + y * W + x];
				const double F = volumePtr[zm * H * W + y * W + xm];
				const double G = volumePtr[zm * H * W + ym * W + x];
				const double H_ = volumePtr[zm * H * W + ym * W + xm];

				result[z * H * W + y * W + x] = 0.5 * (A + B + C - D_ + E + F + G - H_);
			}
		}
	}

	std::copy(result.begin(), result.end(), volumePtr);
}

void HartleyTransform::FDHT1D(double* vec, Direction direction) {
	if (vec == nullptr) {
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

	for (size_t i = 1; i < n; ++i) {
		auto j = BitReversedIndex(direction, i);
		if (j > i) {
			std::swap(vec[i], vec[j]);
		}
	}

	// FHT for 1rd axis
	const auto kLog2n = static_cast<size_t>(std::log2(n));
	const auto kPi = std::acos(-1);

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
				const auto c = std::cos(j * kPi / m2);
				const auto s = std::sin(j * kPi / m2);
				vec[r + m2 + j] = u * c + v * s;
				vec[r + m2 + k] = u * s - v * c;
			}
			for (size_t j = 0; j < m2; ++j) {
				const auto u = vec[r + j];
				const auto v = vec[r + j + m2];
				vec[r + j] = u + v;
				vec[r + j + m2] = u - v;
			}
		}
	}
}

void HartleyTransform::FDHT2D(double* image_ptr) {
	PROFILE_FUNCTION();

	if (image_ptr == nullptr) {
		std::cout << "The pointer to image is null." << std::endl;
		throw std::invalid_argument("The pointer to image is null.");
	}
	if (Width() < 0 || Height() < 0) {
		std::cout << "Error: rows, and cols must be non-negative." << std::endl;
		throw std::invalid_argument("Error: rows, and cols must be non-negative.");
	}

	Series1D(image_ptr, Direction::X);
	TransposeSimple(image_ptr, Width(), Height());

	Series1D(image_ptr, Direction::Y);
	TransposeSimple(image_ptr, Height(), Width());

	BracewellTransform2DCPU(image_ptr);
}

void HartleyTransform::FDHT3D(double* volume_ptr) {
	PROFILE_FUNCTION();

	if (volume_ptr == nullptr) {
		std::cout << "The pointer to volume is null." << std::endl;
		throw std::invalid_argument("The pointer to volume is null.");
	}
	if (Width() <= 0 || Height() <= 0 || Depth() <= 0) {
		std::cout << "Error: dimensions must be positive." << std::endl;
		throw std::invalid_argument("Error: dimensions must be positive.");
	}

	const size_t W = Width();
	const size_t H = Height();
	const size_t D = Depth();

	// 1D transforms along X dimension
	for (size_t z = 0; z < D; ++z) {
		for (size_t y = 0; y < H; ++y) {
			Series1D(volume_ptr + z * H * W + y * W, Direction::X);
		}
	}

	// Transpose XY slices
	for (size_t z = 0; z < D; ++z) {
		TransposeSimple(volume_ptr + z * H * W, W, H);
	}

	// 1D transforms along Y dimension
	for (size_t z = 0; z < D; ++z) {
		for (size_t x = 0; x < W; ++x) {
			double* col = new double[H];
			for (size_t y = 0; y < H; ++y) {
				col[y] = volume_ptr[z * H * W + y * W + x];
			}

			Series1D(col, Direction::Y);

			for (size_t y = 0; y < H; ++y) {
				volume_ptr[z * H * W + y * W + x] = col[y];
			}
			delete[] col;
		}
	}

	// Transpose XY slices back
	for (size_t z = 0; z < D; ++z) {
		TransposeSimple(volume_ptr + z * H * W, H, W);
	}

	// 1D transforms along Z dimension
	for (size_t y = 0; y < H; ++y) {
		for (size_t x = 0; x < W; ++x) {
			double* line = new double[D];
			for (size_t z = 0; z < D; ++z) {
				line[z] = volume_ptr[z * H * W + y * W + x];
			}

			Series1D(line, Direction::Z);
			for (size_t z = 0; z < D; ++z) {
				volume_ptr[z * H * W + y * W + x] = line[z];
			}

			delete[] line;
		}
	}

	// Bracewell 3D
	BracewellTransform3DCPU(volume_ptr, W, H, D);
}

void HartleyTransform::RealFFT1D(double* vec, Direction direction) {
	PROFILE_FUNCTION();

	if (vec == nullptr) {
		std::cout << "The pointer to vector is null." << std::endl;
		throw std::invalid_argument("The pointer to vector is null.");
	}

	// Indices for bit reversal operation
	// and length of vector depending of direction
	//auto [bit_reversed_indices, length] = choose_reverced_indices(direction);

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
	std::vector<std::complex<double>> x(Length(direction));
	for (int i = 0; i < Length(direction); i++) {
		x[i] = std::complex<double>(vec[i], 0);
	}
	size_t k = Length(direction);
	size_t n;
	const double thetaT = std::acos(-1) / Length(direction);
	std::complex<double> phiT = std::complex<double>(std::cos(thetaT), -std::sin(thetaT)), T;
	while (k > 1) {
		n = k;
		k >>= 1;
		phiT = phiT * phiT;
		T = 1.0L;
		for (size_t l = 0; l < k; l++) {
			for (size_t a = l; a < Length(direction); a += n) {
				size_t b = a + k;
				std::complex<double> t = x[a] - x[b];
				x[a] += x[b];
				x[b] = t * T;
			}
			T *= phiT;
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
			std::complex<double> t = x[a];
			x[a] = x[b];
			x[b] = t;
		}
	}

	for (int i = 0; i < Length(direction); i++) {
		vec[i] = x[i].real();
	}
}

void HartleyTransform::DHT1DCuda(double* h_x) {
	PROFILE_FUNCTION();

	// Allocate memory on the device
	dev_array<double> d_x(Width());			// input vector
	dev_array<double> d_y(Width());			// output vector

	//write_matrix_to_csv(h_A.data(), length, length, "matrix.csv");

	// transfer CPU -> GPU
	d_x.set(h_x, Width());
	VectorMatrixMultiplication(_dTransformMatrices[static_cast<size_t>(Direction::X)].getData(),
		d_x.getData(), d_y.getData(), Width());
	// transfer GPU -> CPU
	d_y.get(h_x, Width());
}

void HartleyTransform::DHT2DCuda(double* h_X) {
	PROFILE_FUNCTION();

	// Allocate memory on the device
	dev_array<double> d_X(Width() * Height()); // one slice
	dev_array<double> d_Y(Width() * Height()); // one slice

	// transfer CPU -> GPU
	d_X.set(&h_X[0], Width() * Height());
	MatrixMultiplication(d_X.getData(), _dTransformMatrices[static_cast<size_t>(Direction::X)].getData(),
		d_Y.getData(), Height(), Width(), Width());

	MatrixTranspose(d_Y.getData(), d_X.getData(), Height(), Width());

	MatrixMultiplication(d_X.getData(),
		_dTransformMatrices[static_cast<size_t>(Direction::Y)].getData(),
		d_Y.getData(), Width(), Height(), Height());

	MatrixTranspose(d_Y.getData(), d_X.getData(), Width(), Height());

	// Bracewell
	BracewellTransform2D(d_X.getData(), Width());

	// transfer GPU -> CPU
	d_X.get(&h_X[0], Width() * Height());
	cudaDeviceSynchronize();
}

void HartleyTransform::DHT3DCuda(double* h_X) {
	PROFILE_FUNCTION();

	const auto W = Width();
	const auto H = Height();
	const auto D = Depth();

	// Allocate memory on the device
	dev_array<double> d_X(W * H * D);
	dev_array<double> d_Y(W * H * D);

	// transfer CPU -> GPU
	d_X.set(h_X, W * H * D);

	// 1D Hartley along X
	for (size_t z = 0; z < D; ++z) {
		for (size_t y = 0; y < H; ++y) {
			MatrixMultiplication(
				d_X.getData() + (z * H + y) * W,
				_dTransformMatrices[static_cast<size_t>(Direction::X)].getData(),
				d_Y.getData() + (z * H + y) * W,
				W, W, W
			);
		}
	}

	// transpose XY slices
	for (size_t z = 0; z < D; ++z) {
		MatrixTranspose(
			d_Y.getData() + z * H * W,
			d_X.getData() + z * H * W,
			W, H
		);
	}

	// 1D Hartley along Y
	for (size_t z = 0; z < D; ++z) {
		MatrixMultiplication(
			d_X.getData() + z * H * W,
			_dTransformMatrices[static_cast<size_t>(Direction::Y)].getData(),
			d_Y.getData() + z * H * W,
			H, H, W
		);
	}

	// transpose XY slices back
	for (int z = 0; z < D; ++z) {
		MatrixTranspose(
			d_Y.getData() + z * H * W,
			d_X.getData() + z * H * W,
			H, W
		);
	}

	// 1D Hartley along Z
	MatrixMultiplication3D_Z(d_X.getData(), _dTransformMatrices[static_cast<size_t>(Direction::Z)].getData(), d_Y.getData(), W, H, D);

	// Bracewell 3D
	BracewellTransform3D(d_Y.getData(), W, H, D);

	// transfer GPU -> CPU
	d_Y.get(h_X, W * H * D);

	cudaDeviceSynchronize();
}

} // namespace RapiDHT