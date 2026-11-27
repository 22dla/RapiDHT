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

/**
 * R-5: FactoredAxis reading integers (TIn = std::uint8_t or std::uint16_t) from
 * `in` and writing T to `out`, `offset` subtracted from every value as it is
 * read -- the first pass of a transform of an integer volume, not in place.
 * `in` is not written. Where the pipelined kernel applies it reads the
 * integers itself; elsewhere they are converted into `out` and the usual pass
 * runs there.
 */
template <typename T, typename TIn>
void FactoredAxisFrom(const TIn* in, T* out, const T* offset, const FactoredAxisTables<T>& tables, size_t axisStride,
    size_t laneStride, size_t laneLength, size_t otherStride, size_t otherLength);

/// R-5: out[i] = T(in[i]) - *offset for i < count, elementwise.
template <typename T, typename TIn>
void ConvertFromIntegers(const TIn* in, T* out, const T* offset, size_t count);

/// R-5: data[0] += *offset * n, computed in double -- the offset subtracted
/// on reading, returned to the zero-frequency coefficient (n = N).
template <typename T>
void AddOffsetToFirst(T* data, const T* offset, double n);

/// R-5 step 3.1: *offset = value (an integer), by a kernel of one thread, so
/// that every offset policy leaves the value where the first pass reads it.
template <typename T>
void SetIntegerOffset(T* offset, double value);

/// R-5 step 3.1: the rounded mean of every step-th line along x (lines 0,
/// step, 2 step, ... of `lines` lines of `width` values; step 1 is the whole
/// volume) into *offset, exactly: the sum in uint64 into *sum first. All on
/// the device, asynchronous.
template <typename T, typename TIn>
void IntegerMeanOffset(const TIn* in, size_t width, size_t lines, size_t step, unsigned long long* sum, T* offset);

/// R-5 step 4.2: FactoredLastAxisBracewell ending in integers -- the last pass
/// of the inverse of an integer volume. Each value goes through
/// ToIntegerSaturated(value, scale, *offset) into `out` (TOut = std::uint8_t or
/// std::uint16_t, same layout, not in place); clamped voxels are added to
/// *saturated, each once. Returns false, launching nothing, where the fused
/// kernel does not apply.
template <typename T, typename TOut>
bool FactoredLastAxisBracewellTo(T* data, TOut* out, T scale, const T* offset, unsigned long long* saturated,
    const FactoredAxisTables<T>& tables, size_t H, size_t D);

/// R-5 step 4.2: out[i] = ToIntegerSaturated(in[i], scale, *offset) for i <
/// count, clamped voxels added to *saturated.
template <typename T, typename TOut>
void ConvertToIntegers(const T* in, TOut* out, size_t count, T scale, const T* offset, unsigned long long* saturated);

/// R-5 step 4.2: data[0] -= *offset * n, in double.
template <typename T>
void SubtractOffsetFromFirst(T* data, const T* offset, double n);

/// R-5 step 4.2: *offset = rint(data[0] / n) -- the rounded mean of the volume
/// whose spectrum `data` is.
template <typename T>
void MeanFromFirst(const T* data, double n, T* offset);

/**
 * The last pass along x (the fastest axis) of a true 2D or 3D transform with
 * the Bracewell correction folded in (R-4 step 3.3): call it after the z and y
 * passes, in place of FactoredAxis along x and BracewellInPlace*. Width is
 * tables.n; D is 1 in 2D. Returns false, launching nothing, where the fused
 * kernel does not apply (double, width < 16, shared memory) -- the caller then
 * takes the two separate steps.
 */
template <typename T>
bool FactoredLastAxisBracewell(T* data, const FactoredAxisTables<T>& tables, size_t H, size_t D);

/**
 * R-6 step 1.1: the x pass of a circular convolution, fused -- transform,
 * product with `spectrum` (H(g) / N, same layout as data), transform -- to be
 * called between the z, y passes of the separable transform and the y, z
 * passes again (FactoredConvolveXKernel in factored_kernels.cuh). `pair`: any
 * kernel (Bracewell and the pair product inside the pass); otherwise the
 * kernel is even along every axis and the product is pointwise. Returns false,
 * launching nothing, where it does not apply (double, width < 16, shared
 * memory); the caller then takes ConvolutionProduct between two whole
 * transforms.
 */
template <typename T>
bool FactoredConvolveX(T* data, const T* spectrum, const FactoredAxisTables<T>& tables, size_t H, size_t D, bool pair);

/// R-6: the product of the convolution theorem over a whole spectrum, in
/// place -- the pair product Z(k) = F(k) E(k) + F(-k) O(k) of the true
/// transform (`pair`), or the pointwise product of an even kernel's.
template <typename T>
void ConvolutionProduct(T* data, const T* spectrum, size_t W, size_t H, size_t D, bool pair);

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
