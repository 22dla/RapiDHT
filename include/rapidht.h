#ifndef FHT_H
#define FHT_H

//#define _USE_MATH_DEFINES

#include <vector>
#include <dev_array.h>

namespace RapiDHT {
enum Directions
{
	DIRECTION_X,
	DIRECTION_Y,
	DIRECTION_Z
};
enum Modes
{
	CPU,
	GPU,
	RFFT
};

class HartleyTransform
{
public:
	HartleyTransform(int rows, int cols = 0, int depth = 0, Modes mode = Modes::CPU)
		: rows_(rows), cols_(cols), depth_(depth), mode_(mode)
	{
		if (rows_ <= 0 || cols_ < 0 || depth_ < 0) {
			throw std::invalid_argument("Error (initialization): at least Rows must be positive. \
					Cols and Depth can be zero (by default) but not negative.");
		}
		else if (cols_ == 0 && depth_ > 0) {
			throw std::invalid_argument("Error (initialization): if cols is zero, depth must be zero.");
		}

		// Preparation to 1D transforms
		if (mode_ == Modes::CPU || mode_ == Modes::RFFT) {
			bit_reversed_indices_x_.resize(rows_);
			bit_reversed_indices_y_.resize(cols_);
			bit_reversed_indices_z_.resize(depth_);
			bit_reverse(&bit_reversed_indices_x_);
			bit_reverse(&bit_reversed_indices_y_);
			bit_reverse(&bit_reversed_indices_z_);
		}
		if (mode_ == Modes::GPU) {
			// Initialize Vandermonde matrice on the host
			initialize_kernel_host(&h_Vandermonde_Matrix_x_, rows);
			//initializeKernelHost(h_A, rows);
			//initializeKernelHost(h_A, rows);


			// transfer CPU -> GPU
			d_Vandermonde_Matrix_x_.resize(rows_ * cols_);
			d_Vandermonde_Matrix_x_.set(&h_Vandermonde_Matrix_x_[0], rows_ * cols_);
		}
	}
	void ForwardTransform(double* data);
	void InverseTransform(double* data);

private:
	/* ------------------------- ND Transforms ------------------------- */
	/**
	 * FDHT1D(double* vector) returns the Hartley transform
	 * of an 1D array using a fast Hartley transform algorithm.
	 */
	void FDHT1D(double* vector, const Directions direction = Directions::DIRECTION_X);

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
	void DHT1DCuda(double* h_x, double* h_A, const int length);

	/**
	* DHT2DCuda(double* image) returns the Hartley
	* transform of an 1D array using a matrix x matrix multiplication.
	*/
	void DHT2DCuda(double* image);

	/**
	 * RealFFT1D(double* vector) returns the Fourier transform
	 * of an 1D array using a real Fourier transform algorithm.
	 */
	void RealFFT1D(double* vector, const Directions direction = Directions::DIRECTION_X);

	void series1d(double* image, const Directions direction);

	static void bit_reverse(std::vector<size_t>* indices);
	static void initialize_kernel_host(std::vector<double>* kernel, const int cols);
	static std::vector<double> DHT1D(const std::vector<double>& a, const std::vector<double>& kernel);
	template <typename T>
	static void transpose(std::vector<std::vector<T>>* image);
	static void transpose_simple(double* image, int rows, int cols);
	void BracewellTransform2DCPU(double* image_ptr);

	size_t* choose_reverced_indices(int* length, const Directions direction);

	int rows_ = 0;
	int cols_ = 0;
	int depth_ = 0;
	Modes mode_ = Modes::CPU;
	std::vector<size_t> bit_reversed_indices_x_;
	std::vector<size_t> bit_reversed_indices_y_;
	std::vector<size_t> bit_reversed_indices_z_;
	std::vector<double> h_Vandermonde_Matrix_x_;
	dev_array<double> d_Vandermonde_Matrix_x_;
};
}

#endif // !FHT_H