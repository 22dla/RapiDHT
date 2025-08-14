#include <omp.h>
#include <rapidht.h>
#include <utilities.h>
#include <complex>
#include <kernel.h>

namespace RapiDHT {

HartleyTransform::HartleyTransform(size_t width, size_t height = 0, size_t depth = 0, Modes mode = Modes::CPU) :_mode(mode) {
	if (width <= 0 || height < 0 || depth < 0) {
		throw std::invalid_argument(
			"Error (initialization): at least Rows must be positive. "
			"height and depth can be zero (by default) but not negative.");
	}
	if (height == 0 && depth > 0) {
		throw std::invalid_argument(
			"Error (initialization): if height is zero, depth must be zero.");
	}

	_dims[0] = static_cast<size_t>(width);
	_dims[1] = static_cast<size_t>(height);
	_dims[2] = static_cast<size_t>(depth);

	// Preparation to 1D transforms
	if (_mode == Modes::CPU || _mode == Modes::RFFT) {
		for (size_t i = 0; i < _bit_reversed_indices.size(); ++i) {
			_bit_reversed_indices[i].resize(_dims[i]);
			bit_reverse(&_bit_reversed_indices[i]);
		}
	}
	if (_mode == Modes::GPU) {
		// Initialize Vandermonde matrice on the host
		initialize_kernel_host(&_h_transform_matrix_x, width);
		//initializeKernelHost(h_A, width);
		//initializeKernelHost(h_A, width);


		// transfer CPU -> GPU
		_d_transform_matrix_x.resize(_dims[0] * _dims[1]);
		_d_transform_matrix_x.set(&_h_transform_matrix_x[0], _dims[0] * _dims[1]);
	}
}

void HartleyTransform::ForwardTransform(double* data) {
	bool is1D = (height() == 0 && depth() == 0);
	bool is2D = (height() > 0 && depth() == 0);
	// bool is3D = (depth() > 0); // пока не используется

	switch (_mode) {
	case Modes::CPU:
		if (is1D) {
			FDHT1D(data);
		} else if (is2D) {
			FDHT2D(data);
		}
		break;

	case Modes::GPU:
		if (is1D) {
			DHT1DCuda(data, _h_transform_matrix_x.data(), width());
		} else if (is2D) {
			DHT2DCuda(data);
		}
		break;

	case Modes::RFFT:
		if (is1D) {
			RealFFT1D(data);
		} else if (is2D) {
			FDHT2D(data);
		}
		break;

	default:
		break;
	}
}

void HartleyTransform::InverseTransform(double* data) {
	ForwardTransform(data);

	size_t totalSize = width();
	if (height() > 0) {
		totalSize *= height();
	}
	if (depth() > 0) {
		totalSize *= depth();
	}

	auto denominator = 1.0 / static_cast<double>(totalSize);

	for (size_t i = 0; i < totalSize; ++i) {
		data[i] *= denominator;
	}
}

void HartleyTransform::bit_reverse(std::vector<size_t>* indices_ptr) {
	std::vector<size_t>& indices = *indices_ptr;
	if (indices.size() == 0) {
		return;
	}
	const int kLog2n = static_cast<int>(log2f(static_cast<float>(indices.size())));

	// array to store binary number
	std::vector<bool> binary_num(indices.size());

	indices[0] = 0;
	for (int j = 1; j < indices.size(); ++j) {
		// counter for binary array
		size_t count = 0;
		int base = j;
		while (base > 0) {
			// storing remainder in binary array
			binary_num[count] = static_cast<bool>(base % 2);
			base /= 2;
			++count;
		}
		for (size_t i = count; i < kLog2n; ++i) {
			binary_num[i] = false;
		}

		int dec_value = 0;
		base = 1;
		for (int i = kLog2n - 1; i >= 0; --i) {
			if (binary_num[i]) {
				dec_value += base;
			}
			base *= 2;
		}
		indices[j] = dec_value;
	}
}

void HartleyTransform::initialize_kernel_host(std::vector<double>* kernel, size_t height) {
	if (kernel == nullptr) {
		throw std::invalid_argument("Error: kernell==nullptr (initialize_kernel_host)");
	}
	auto& ker = *kernel;
	ker.resize(height * height);
	const double m_pi = std::acos(-1);

	// Initialize the matrice on the host
	for (size_t k = 0; k < height; ++k) {
		for (size_t j = 0; j < height; ++j) {
			ker[k * height + j] = std::cos(2 * m_pi * k * j / height) + std::sin(2 * m_pi * k * j / height);
		}
	}
}

// test function
std::vector<double> HartleyTransform::DHT1D(
	const std::vector<double>& a, const std::vector<double>& kernel) {
	std::vector<double> result(a.size());

	for (size_t i = 0; i < a.size(); i++)
		for (size_t j = 0; j < a.size(); j++)
			result[i] += (kernel[i * a.size() + j] * a[j]);

	// RVO works
	return result;
}

template <typename T>
void HartleyTransform::transpose(std::vector<std::vector<T>>* matrix_ptr) {
	std::vector<std::vector<T>>& matrix = *matrix_ptr;

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

void HartleyTransform::transpose_simple(double* matrix, size_t width, size_t height) {
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

void HartleyTransform::series1d(double* image_ptr, Direction direction) {
	PROFILE_FUNCTION();

	if (image_ptr == nullptr) {
		throw std::invalid_argument("The pointer to image is null.");
	}

	if (_mode == Modes::CPU) {
	#pragma omp parallel for
		for (int i = 0; i < width(); ++i) {
			this->FDHT1D(image_ptr + i * height(), direction);
		}
	}
	if (_mode == Modes::RFFT) {
	#pragma omp parallel for
		for (int i = 0; i < width(); ++i) {
			RealFFT1D(image_ptr + i * height(), direction);
		}
	}
}

void HartleyTransform::FDHT1D(double* vec, const Direction& direction) {
	if (vec == nullptr) {
		throw std::invalid_argument("The pointer to vector is null.");
	}

	// Indices for bit reversal operation
	// and length of vector depending of direction
	if (length(direction) < 0) {
		std::cout << "Error: length must be non-negative." << std::endl;
		throw std::invalid_argument("Error: length must be non-negative.");
	}
	// Check that length is power of 2
	if (std::ceil(std::log2(length(direction))) != std::floor(std::log2(length(direction)))) {
		std::cout << "Error: length must be a power of two." << std::endl;
		throw std::invalid_argument("Error: length must be a power of two.");
	}

	for (int i = 1; i < length(direction); ++i) {
		size_t j = bit_reversed_index(direction, i);
		if (j > i) {
			std::swap(vec[i], vec[j]);
		}
	}

	// FHT for 1rd axis
	const int kLog2n = static_cast<int>(log2f(static_cast<float>(length(direction))));
	const double kPi = std::acos(-1);

	// Main cicle
	for (int s = 1; s <= kLog2n; ++s) {
		int m = static_cast<int>(powf(2, s));
		int m2 = m / 2;
		int m4 = m / 4;

		for (size_t r = 0; r <= length(direction) - m; r = r + m) {
			for (size_t j = 1; j < m4; ++j) {
				int k = m2 - j;
				double u = vec[r + m2 + j];
				double v = vec[r + m2 + k];
				double c = std::cos(static_cast<double>(j) * kPi / m2);
				double s = std::sin(static_cast<double>(j) * kPi / m2);
				vec[r + m2 + j] = u * c + v * s;
				vec[r + m2 + k] = u * s - v * c;
			}
			for (size_t j = 0; j < m2; ++j) {
				double u = vec[r + j];
				double v = vec[r + j + m2];
				vec[r + j] = u + v;
				vec[r + j + m2] = u - v;
			}
		}
	}
}

void HartleyTransform::BracewellTransform2DCPU(double* image_ptr) {
	//PROFILE_FUNCTION();
	std::vector<double> H(width() * height(), 0.0);
#pragma omp parallel for
	for (int i = 0; i < width(); ++i) {
		for (int j = 0; j < height(); ++j) {
			double A = image_ptr[i * height() + j];
			double B = (i > 0 && j > 0) ? image_ptr[i * height() + (height() - j)] : A;
			double C = (i > 0 && j > 0) ? image_ptr[(width() - i) * height() + j] : A;
			double D = (i > 0 && j > 0) ? image_ptr[(width() - i) * height() + (height() - j)] : A;
			H[i * height() + j] = (A + B + C - D) / 2.0;
		}
	}

	//image = std::move(H);
	std::copy(H.begin(), H.end(), image_ptr);
}

void HartleyTransform::FDHT2D(double* image_ptr) {
	if (image_ptr == nullptr) {
		std::cout << "The pointer to image is null." << std::endl;
		throw std::invalid_argument("The pointer to image is null.");
	}
	if (width() < 0 || height() < 0) {
		std::cout << "Error: rows, and cols must be non-negative." << std::endl;
		throw std::invalid_argument("Error: rows, and cols must be non-negative.");
	}

	// write_matrix_to_csv(image_ptr, width, height, "matrix1.txt");

	// 1D transforms along X dimension
	series1d(image_ptr, Direction::X);

	transpose_simple(image_ptr, width(), height());

	// 1D transforms along Y dimension
	series1d(image_ptr, Direction::Y);

	transpose_simple(image_ptr, height(), width());

	BracewellTransform2DCPU(image_ptr);

	// write_matrix_to_csv(image_ptr, width, height, "matrix2.txt");
}

// test functions
void HartleyTransform::RealFFT1D(double* vec, Direction direction) {
	if (vec == nullptr) {
		std::cout << "The pointer to vector is null." << std::endl;
		throw std::invalid_argument("The pointer to vector is null.");
	}

	// Indices for bit reversal operation
	// and length of vector depending of direction
	//auto [bit_reversed_indices, length] = choose_reverced_indices(direction);

	if (length(direction) < 0) {
		std::cout << "Error: length must be non-negative." << std::endl;
		throw std::invalid_argument("Error: length must be non-negative.");
	}
	// Check that length is power of 2
	if (std::ceil(std::log2(length(direction))) != std::floor(std::log2(length(direction)))) {
		std::cout << "Error: length must be a power of two." << std::endl;
		throw std::invalid_argument("Error: length must be a power of two.");
	}

	// RealFFT
	std::vector<std::complex<double>> x(length(direction));
	for (int i = 0; i < length(direction); i++) {
		x[i] = std::complex<double>(vec[i], 0);
	}
	unsigned int k = length(direction);
	unsigned int n;
	double thetaT = 3.14159265358979323846264338328L / length(direction);
	std::complex<double> phiT = std::complex<double>(cos(thetaT), -sin(thetaT)), T;
	while (k > 1) {
		n = k;
		k >>= 1;
		phiT = phiT * phiT;
		T = 1.0L;
		for (unsigned int l = 0; l < k; l++) {
			for (unsigned int a = l; a < length(direction); a += n) {
				unsigned int b = a + k;
				std::complex<double> t = x[a] - x[b];
				x[a] += x[b];
				x[b] = t * T;
			}
			T *= phiT;
		}
	}
	// Decimate
	unsigned int m = (unsigned int)log2(length(direction));
	for (unsigned int a = 0; a < length(direction); a++) {
		unsigned int b = a;
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

	for (int i = 0; i < length(direction); i++) {
		vec[i] = x[i].real();
	}
}

void HartleyTransform::DHT1DCuda(double* h_x, double* h_A, size_t length) {
	// Allocate memory on the device
	dev_array<double> d_A(length * length);	// matrix for one line
	dev_array<double> d_x(length);			// input vector
	dev_array<double> d_y(length);			// output vector

	//write_matrix_to_csv(h_A.data(), length, length, "matrix.csv");

	// transfer CPU -> GPU
	d_A.set(&h_A[0], length * length);
	// transfer CPU -> GPU
	d_x.set(h_x, length * length);
	vectorMatrixMultiplication(d_A.getData(), d_x.getData(), d_y.getData(), length);
	// transfer GPU -> CPU
	d_y.get(h_x, length);
	cudaDeviceSynchronize();
}

void HartleyTransform::DHT2DCuda(double* h_X) {
	// Allocate memory on the device
	dev_array<double> d_X(width() * height()); // one slice
	dev_array<double> d_Y(width() * height()); // one slice

	// transfer CPU -> GPU
	d_X.set(&h_X[0], width() * height());
	matrixMultiplication(_d_transform_matrix_x.getData(), d_X.getData(), d_Y.getData(), height());
	matrixTranspose(d_Y.getData(), height());
	matrixMultiplication(_d_transform_matrix_x.getData(), d_Y.getData(), d_X.getData(), height());
	matrixTranspose(d_X.getData(), height());

	// transfer GPU -> CPU
	d_X.get(&h_X[0], width() * height());
	cudaDeviceSynchronize();
}

} // namespace RapiDHT