#include <omp.h>
#include <rapidht.h>
#include <utilities.h>
#include <complex>
#include <kernel.h>

namespace RapiDHT {

void HartleyTransform::ForwardTransform(double* data)
{

	switch (mode_) {
	case RapiDHT::CPU:
		if (cols_ == 0 && depth_ == 0) {
			FDHT1D(data);
		}
		else if (depth_ == 0) {
			FDHT2D(data);
		}
		break;
	case RapiDHT::GPU:
		if (cols_ == 0 && depth_ == 0) {
			DHT1DCuda(data, h_Vandermonde_Matrix_x_.data(), rows_);
		}
		else if (depth_ == 0) {
			DHT2DCuda(data);
		}
		break;
	case RapiDHT::RFFT:
		if (cols_ == 0 && depth_ == 0) {
			RealFFT1D(data);
		}
		else if (depth_ == 0) {
			FDHT2D(data);
		}
		break;
	default:
		break;
	}
}

void HartleyTransform::InverseTransform(double* data)
{
	this->ForwardTransform(data);

	double denominator = 0;
	if (cols_ == 0 && depth_ == 0) {	// 1D
		denominator = 1.0f / rows_;
	}
	else if (depth_ == 0) {			// 2D
		denominator = 1.0f / (rows_ * cols_);
	}
	else {							// 3D
		denominator = 1.0f / (rows_ * cols_ * depth_);
	}

	size_t size = (depth_ > 0) ? rows_ * cols_ * depth_ : ((cols_ > 0) ? rows_ * cols_ : rows_);
	for (int i = 0; i < size; ++i) {
		data[i] *= denominator;
	}
}

void HartleyTransform::bit_reverse(std::vector<size_t>* indices_ptr)
{
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

void HartleyTransform::initialize_kernel_host(std::vector<double>* kernel, const int cols)
{
	if (kernel == nullptr) {
		throw std::invalid_argument("Error: kernell==nullptr (initialize_kernel_host)");
	}
	auto& ker = *kernel;
	ker.resize(cols * cols);
	const double m_pi = std::acos(-1);

	// Initialize the matrice on the host
	for (size_t k = 0; k < cols; ++k) {
		for (size_t j = 0; j < cols; ++j) {
			ker[k * cols + j] = std::cos(2 * m_pi * k * j / cols) + std::sin(2 * m_pi * k * j / cols);
		}
	}
}

// test function
std::vector<double> HartleyTransform::DHT1D(
	const std::vector<double>& a, const std::vector<double>& kernel)
{
	std::vector<double> result(a.size());

	for (size_t i = 0; i < a.size(); i++)
		for (size_t j = 0; j < a.size(); j++)
			result[i] += (kernel[i * a.size() + j] * a[j]);

	// RVO works
	return result;
}

template <typename T>
void HartleyTransform::transpose(std::vector<std::vector<T>>* matrix_ptr)
{
	std::vector<std::vector<T>>& matrix = *matrix_ptr;

	const size_t rows = matrix.size();
	const size_t cols = matrix[0].size();

#pragma omp parallel for
	for (int i = 0; i < rows; ++i) {
	#pragma omp parallel for
		for (int j = i + 1; j < cols; ++j) {
			std::swap(matrix[i][j], matrix[j][i]);
		}
	}
}

void HartleyTransform::transpose_simple(double* matrix, const int rows, const int cols)
{
	if (matrix == nullptr) {
		throw std::invalid_argument("The pointer to matrix is null.");
	}

	if (rows == cols) {
	#pragma omp parallel for
		// Square matrix
		for (int i = 0; i < rows; ++i) {
		#pragma omp parallel for
			for (int j = i + 1; j < cols; ++j) {
				std::swap(matrix[i * cols + j], matrix[j * cols + i]);
			}
		}
	}
	else {
		// Non-square matrix
		std::vector<double> transposed(rows * cols);
	#pragma omp parallel for
		for (int i = 0; i < rows; ++i) {
		#pragma omp parallel for
			for (int j = 0; j < cols; ++j) {
				transposed[j * rows + i] = matrix[i * cols + j];
			}
		}
		//std::memcpy(matrix, transposed.data(), sizeof(double) * rows * cols);
		//require to check
		std::copy(transposed.data(), transposed.data() + (rows * cols), matrix);
	}
}

void HartleyTransform::series1d(double* image_ptr, const Directions direction)
{
	PROFILE_FUNCTION();

	if (image_ptr == nullptr) {
		throw std::invalid_argument("The pointer to image is null.");
	}

	if (mode_ == Modes::CPU) {
	#pragma omp parallel for
		for (int i = 0; i < rows_; ++i) {
			this->FDHT1D(image_ptr + i * cols_, direction);
		}
	}
	if (mode_ == Modes::RFFT) {
	#pragma omp parallel for
		for (int i = 0; i < rows_; ++i) {
			RealFFT1D(image_ptr + i * cols_, direction);
		}
	}
}

void HartleyTransform::FDHT1D(double* vec, const Directions direction)
{
	if (vec == nullptr) {
		throw std::invalid_argument("The pointer to vector is null.");
	}

	// Indices for bit reversal operation
	// and length of vector depending of direction
	int length = 0;
	auto bit_reversed_indices = this->choose_reverced_indices(&length, direction);

	if (length < 0) {
		std::cout << "Error: length must be non-negative." << std::endl;
		throw std::invalid_argument("Error: length must be non-negative.");
	}
	// Check that length is power of 2
	if (std::ceil(std::log2(length)) != std::floor(std::log2(length))) {
		std::cout << "Error: length must be a power of two." << std::endl;
		throw std::invalid_argument("Error: length must be a power of two.");
	}

	for (int i = 1; i < length; ++i) {
		size_t j = bit_reversed_indices[i];
		if (j > i) {
			std::swap(vec[i], vec[j]);
		}
	}

	// FHT for 1rd axis
	const int kLog2n = static_cast<int>(log2f(static_cast<float>(length)));
	const double kPi = std::acos(-1);

	// Main cicle
	for (int s = 1; s <= kLog2n; ++s) {
		int m = static_cast<int>(powf(2, s));
		int m2 = m / 2;
		int m4 = m / 4;

		for (size_t r = 0; r <= length - m; r = r + m) {
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

void HartleyTransform::BracewellTransform2DCPU(double* image_ptr)
{
	//PROFILE_FUNCTION();
	std::vector<double> H(rows_ * cols_, 0.0);
#pragma omp parallel for
	for (int i = 0; i < rows_; ++i) {
		for (int j = 0; j < cols_; ++j) {
			double A = image_ptr[i * cols_ + j];
			double B = (i > 0 && j > 0) ? image_ptr[i * cols_ + (cols_ - j)] : A;
			double C = (i > 0 && j > 0) ? image_ptr[(rows_ - i) * cols_ + j] : A;
			double D = (i > 0 && j > 0) ? image_ptr[(rows_ - i) * cols_ + (cols_ - j)] : A;
			H[i * cols_ + j] = (A + B + C - D) / 2.0;
		}
	}

	//image = std::move(H);
	std::copy(H.begin(), H.end(), image_ptr);
}

void HartleyTransform::FDHT2D(double* image_ptr)
{
	if (image_ptr == nullptr) {
		std::cout << "The pointer to image is null." << std::endl;
		throw std::invalid_argument("The pointer to image is null.");
	}
	if (rows_ < 0 || cols_ < 0) {
		std::cout << "Error: rows, and cols must be non-negative." << std::endl;
		throw std::invalid_argument("Error: rows, and cols must be non-negative.");
	}

	// write_matrix_to_csv(image_ptr, rows, cols, "matrix1.txt");

	// 1D transforms along X dimension
	this->series1d(image_ptr, X);

	transpose_simple(image_ptr, rows_, cols_);

	// 1D transforms along Y dimension
	this->series1d(image_ptr, Y);

	transpose_simple(image_ptr, cols_, rows_);

	BracewellTransform2DCPU(image_ptr);

	// write_matrix_to_csv(image_ptr, rows, cols, "matrix2.txt");
}

size_t* HartleyTransform::choose_reverced_indices(int* length, const Directions direction)
{

	size_t* bit_reversed_indices;
	switch (direction) {
	case X:
		*length = rows_;
		bit_reversed_indices = bit_reversed_indices_x_.data();
		break;
	case Y:
		*length = cols_;
		bit_reversed_indices = bit_reversed_indices_y_.data();
		break;
	case Z:
		*length = depth_;
		bit_reversed_indices = bit_reversed_indices_z_.data();
		break;
	default:
		break;
	}
	return bit_reversed_indices;
}

// test functions
void HartleyTransform::RealFFT1D(double* vec, const Directions direction)
{
	if (vec == nullptr) {
		std::cout << "The pointer to vector is null." << std::endl;
		throw std::invalid_argument("The pointer to vector is null.");
	}

	// Indices for bit reversal operation
	// and length of vector depending of direction
	int length = 0;
	auto bit_reversed_indices = this->choose_reverced_indices(&length, direction);

	if (length < 0) {
		std::cout << "Error: length must be non-negative." << std::endl;
		throw std::invalid_argument("Error: length must be non-negative.");
	}
	// Check that length is power of 2
	if (std::ceil(std::log2(length)) != std::floor(std::log2(length))) {
		std::cout << "Error: length must be a power of two." << std::endl;
		throw std::invalid_argument("Error: length must be a power of two.");
	}

	// RealFFT
	std::vector<std::complex<double>> x(length);
	for (int i = 0; i < length; i++) {
		x[i] = std::complex<double>(vec[i], 0);
	}
	unsigned int k = length;
	unsigned int n;
	double thetaT = 3.14159265358979323846264338328L / length;
	std::complex<double> phiT = std::complex<double>(cos(thetaT), -sin(thetaT)), T;
	while (k > 1) {
		n = k;
		k >>= 1;
		phiT = phiT * phiT;
		T = 1.0L;
		for (unsigned int l = 0; l < k; l++) {
			for (unsigned int a = l; a < length; a += n) {
				unsigned int b = a + k;
				std::complex<double> t = x[a] - x[b];
				x[a] += x[b];
				x[b] = t * T;
			}
			T *= phiT;
		}
	}
	// Decimate
	unsigned int m = (unsigned int)log2(length);
	for (unsigned int a = 0; a < length; a++) {
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

	for (int i = 0; i < length; i++) {
		vec[i] = x[i].real();
	}
}

void HartleyTransform::DHT1DCuda(double* h_x, double* h_A, const int length)
{
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

void HartleyTransform::DHT2DCuda(double* h_X)
{
	// Allocate memory on the device
	dev_array<double> d_X(rows_ * cols_); // one slice
	dev_array<double> d_Y(rows_ * cols_); // one slice

	// transfer CPU -> GPU
	d_X.set(&h_X[0], rows_ * cols_);
	matrixMultiplication(d_Vandermonde_Matrix_x_.getData(), d_X.getData(), d_Y.getData(), cols_);
	matrixTranspose(d_Y.getData(), cols_);
	matrixMultiplication(d_Vandermonde_Matrix_x_.getData(), d_Y.getData(), d_X.getData(), cols_);
	matrixTranspose(d_X.getData(), cols_);

	// transfer GPU -> CPU
	d_X.get(&h_X[0], rows_ * cols_);
	cudaDeviceSynchronize();
}

}