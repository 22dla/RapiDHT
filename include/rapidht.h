#ifndef FHT_H
#define FHT_H

#include <vector>
#include <array>
#include <dev_array.h>

namespace RapiDHT {
enum class Direction : size_t { X = 0, Y = 1, Z = 2, Count };
enum class Modes { CPU, GPU, RFFT };

class HartleyTransform {
public:
	HartleyTransform(size_t width, size_t height = 0, size_t depth = 0, Modes mode = Modes::CPU) :_mode(mode) {
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
	void ForwardTransform(double* data);
	void InverseTransform(double* data);

	constexpr size_t width() const noexcept { return _dims[static_cast<size_t>(Direction::X)]; }
	constexpr size_t height() const noexcept { return _dims[static_cast<size_t>(Direction::Y)]; }
	constexpr size_t depth() const noexcept { return _dims[static_cast<size_t>(Direction::Z)]; }
	constexpr size_t length(Direction direction) const noexcept { return _dims[static_cast<size_t>(direction)]; }
	inline size_t bit_reversed_index(Direction direction, size_t index) const noexcept {
		return _bit_reversed_indices[static_cast<size_t>(direction)][index];
	}


private:
	/* ------------------------- ND Transforms ------------------------- */
	/**
	 * FDHT1D(double* vector) returns the Hartley transform
	 * of an 1D array using a fast Hartley transform algorithm.
	 */
	void FDHT1D(double* vector, const Direction& direction = Direction::X);

	/**
	 * FHT2D(double* image_ptr) returns the Hartley transform
	 * of an 2D array using a fast Hartley transform algorithm. The 2D transform
	 * is equivalent to computing the 1D transform along each dimension of image.
	 */
	void FDHT2D(double* image);

	/**
	* DHT1DCuda(double* h_x, double* h_A, const int length) returns the Hartley
	* transform of an 1D array using a matrix x vector multiplication.
	*/
	void DHT1DCuda(double* h_x, double* h_A, size_t length);

	/**
	* DHT2DCuda(double* image) returns the Hartley
	* transform of an 1D array using a matrix x matrix multiplication.
	*/
	void DHT2DCuda(double* image);

	/**
	 * RealFFT1D(double* vector) returns the Fourier transform
	 * of an 1D array using a real Fourier transform algorithm.
	 */
	void RealFFT1D(double* vector, Direction direction = Direction::X);

	void series1d(double* image, Direction direction);

	static void bit_reverse(std::vector<size_t>* indices);
	static void initialize_kernel_host(std::vector<double>* kernel, size_t height);
	static std::vector<double> DHT1D(const std::vector<double>& a, const std::vector<double>& kernel);
	template <typename T>
	static void transpose(std::vector<std::vector<T>>* image);
	static void transpose_simple(double* image, size_t width, size_t height);
	void BracewellTransform2DCPU(double* image_ptr);

	std::array<size_t, static_cast<size_t>(Direction::Count)> _dims{};
	std::array<std::vector<size_t>, static_cast<size_t>(Direction::Count)> _bit_reversed_indices;

	Modes _mode = Modes::CPU;

	std::vector<double> _h_transform_matrix_x;
	dev_array<double> _d_transform_matrix_x;
};
}

#endif // !FHT_H