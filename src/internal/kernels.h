/*
 * Project: RapiDHT
 * File: src/internal/kernels.h
 * Brief: Host-side declarations of the CUDA kernels and their wrappers.
 * Author: Volkov Evgeny Aleksandrovich, volkov22dla@yandex.ru
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

#ifndef RAPIDHT_INTERNAL_KERNELS_H
#define RAPIDHT_INTERNAL_KERNELS_H

#include <cstddef>

namespace RapiDHT {

/// C[M x N] = A[M x K] * B[K x N].
template <typename T>
void MatrixMultiplication(const T* A, const T* B, T* C, int M, int K, int N);

/// B = transpose(A), for an A of rows x cols.
template <typename T>
void MatrixTranspose(const T* A, T* B, int rows, int cols);

/// Scales a device buffer in place, for the 1/N of an inverse transform that
/// never leaves the device.
template <typename T>
void ScaleOnDevice(T* deviceData, size_t count, T factor);

/// y = A * x, for a square A of order N.
template <typename T>
void VectorMatrixMultiplication(const T* A, const T* x, T* y, int N);

/// Bracewell correction turning the separable per-axis result into the true
/// multidimensional Hartley transform. Out of place: each output reads
/// mirrored inputs, so sharing one buffer would race.
template <typename T>
void BracewellTransform2D(const T* deviceIn, T* deviceOut, int W, int H);

template <typename T>
void BracewellTransform3D(const T* deviceIn, T* deviceOut, int W, int H, int D);

/*
 * R-4: the factored DHT along one axis (see internal/factored.h for the
 * algebra). Device pointers to the tables of one axis, laid out back to back
 * as c1 | c2 | cHalf | twC | twD.
 */
template <typename T>
struct FactoredAxisTables {
    const T* c1 = nullptr;
    const T* c2 = nullptr;
    const T* cHalf = nullptr;
    const T* twC = nullptr;
    const T* twD = nullptr;
    int n = 1;
    int n1 = 1;
    int n2 = 1;
};

/// Column stride of the fast factored kernel's work buffer: n2 * lanes, plus
/// `lanes` 4-byte words when that is a multiple of 32 words, so that the
/// columns a warp reads together in step 2 land in different banks.
template <typename T>
constexpr size_t FactoredWorkStride(size_t n2, size_t lanes)
{
    return n2 * lanes + ((n2 * lanes * sizeof(T)) % 128 == 0 ? lanes * 4 / sizeof(T) : 0);
}

/// Longest axis the factored kernel takes: the whole line sits in shared
/// memory, twice (tile and work), so 2 * n * sizeof(T) must fit in 48 KiB.
constexpr size_t kFactoredMaxLength = 2048;

/**
 * Transforms every line along one axis in place. A line is
 * data[base + i * axisStride]; lines are grouped in blocks of neighbours along
 * the lane axis (laneStride, laneLength), and otherLength blocks of those are
 * otherStride apart -- the same geometry as the CPU backend's Series1D.
 */
template <typename T>
void FactoredAxis(T* data, const FactoredAxisTables<T>& tables, size_t axisStride, size_t laneStride,
    size_t laneLength, size_t otherStride, size_t otherLength);

/// The Bracewell correction in place, one thread per orbit of mirror images
/// (the CPU backend's scheme), so no second buffer is needed.
template <typename T>
void BracewellInPlace2D(T* data, int W, int H);

template <typename T>
void BracewellInPlace3D(T* data, int W, int H, int D);

/// Fills a square device matrix with cas(2*pi*k*j/height).
void InitializeHartleyMatrix(double* deviceMatrix, size_t height);
void InitializeHartleyMatrix(float* deviceMatrix, size_t height);

} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_KERNELS_H
