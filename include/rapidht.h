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
	HartleyTransform() = delete;

	HartleyTransform(size_t width, size_t height, size_t depth, Modes mode);
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