/*
 * Project: RapiDHT
 * File: kernel.h
 * Brief: Заголовок CUDA-обёрток и ядер для линейной алгебры и преобразования Хартли.
 * Author: Волков Евгений Александрович, volkov22dla@yandex.ru
 *
 * Internal header: not installed and not part of the public API.
 *
 * Every declaration below is backed by an explicit instantiation for float and
 * double at the bottom of kernel.cu. Three declarations used to sit here that
 * were not: they were templates whose parameter appeared nowhere in the
 * signature, e.g.
 *
 *     template <typename T>
 *     void MatrixMultiplication(const double* A, const double* B, double* C, int N);
 *
 * T could not be deduced, so such a function could only be called by naming T
 * explicitly, which nothing did -- and no matching definition existed anyway.
 */

#ifndef KERNEL_H
#define KERNEL_H

#include <cstddef>

namespace RapiDHT {

/// C[M x N] = A[M x K] * B[K x N].
template <typename T>
void MatrixMultiplication(const T* A, const T* B, T* C, int M, int K, int N);

/// B = transpose(A), for an A of rows x cols.
template <typename T>
void MatrixTranspose(const T* A, T* B, int rows, int cols);

/// The same transpose applied to each of `batch` consecutive rows x cols
/// slices, in a single launch.
template <typename T>
void MatrixTransposeBatched(const T* d_in, T* d_out, int rows, int cols, int batch);

/// Scales a device buffer in place, for the 1/N of an inverse transform that
/// never leaves the device.
template <typename T>
void ScaleOnDevice(T* d_data, size_t count, T factor);

/// y = A * x, for a square A of order N.
template <typename T>
void VectorMatrixMultiplication(const T* A, const T* x, T* y, int N);

/// Swaps the Y and Z axes of a W x H x D volume.
template <typename T>
void transpose_YZ_cuda(const T* d_in, T* d_out, int W, int H, int D);

/// Bracewell correction turning the separable per-axis result into the true
/// multidimensional Hartley transform. Out of place: each output reads
/// mirrored inputs, so sharing one buffer would race.
template <typename T>
void BracewellTransform2D(const T* d_in, T* d_out, int W, int H);

template <typename T>
void BracewellTransform3D(const T* d_in, T* d_out, int W, int H, int D);

/// Fills a square device matrix with cas(2*pi*k*j/height).
void InitializeHartleyMatrix(double* dKernel, size_t height);
void InitializeHartleyMatrix(float* dKernel, size_t height);

} // namespace RapiDHT

#endif // KERNEL_H
