/*
 * Project: RapiDHT
 * File: include/rapidht/convolution.h
 * Brief: Circular convolution with kernels that are even along every axis,
 *        through the separable Hartley transform.
 * Author: Volkov Evgeny Aleksandrovich, volkov22dla@yandex.ru
 */

#ifndef RAPIDHT_CONVOLUTION_H
#define RAPIDHT_CONVOLUTION_H

#include <rapidht/transform.h>

#include <cstddef>
#include <vector>

namespace RapiDHT {

/**
 * @brief Checks that a kernel is even along every axis.
 *
 * Even means g(i,j,k) = g(-i,j,k) = g(i,-j,k) = g(i,j,-k), with indices taken
 * modulo the extent, i.e. -i is (W - i) % W. The kernel is stored in the
 * wrapped layout: its centre sits at index (0,0,0), not in the middle of the
 * array. Gaussians, spheres and centred boxes placed this way are even.
 *
 * Extents follow HartleyTransform: height == 0 means 1D, depth == 0 means 2D.
 *
 * @param relativeTolerance Allowed mismatch, relative to max |g|.
 */
template <typename T>
bool IsEvenPerAxis(const T* kernel, size_t width, size_t height, size_t depth,
    double relativeTolerance = 1e-6);

/**
 * @brief Circular convolution with a kernel even along every axis.
 *
 * For such kernels the separable Hartley transform S turns convolution into a
 * real pointwise product, S(f * g) = S(f) . S(g), so one application costs two
 * separable transforms and a multiply, with no Bracewell pass and no complex
 * arithmetic. The kernel spectrum is computed once, in SetKernel, and reused,
 * which is the case that matters: one window applied to many fields.
 *
 * The convolution is circular. For a linear one, pad the data (and place the
 * kernel in the padded extent) so that wrap-around cannot reach the region of
 * interest.
 *
 * Usage:
 * @code
 *   EvenKernelConvolution<float> conv(256, 256, 256, Modes::CPU);
 *   conv.SetKernel(gaussian.data());   // wrapped layout, centre at index 0
 *   conv.Apply(volume.data());         // volume <- volume * gaussian
 * @endcode
 *
 * @tparam T float or double.
 */
template <typename T>
class EvenKernelConvolution {
public:
    /**
     * @param width, height, depth Extent, as for HartleyTransform.
     * @param mode Backend. Only Modes::CPU is supported for now.
     * @throws std::invalid_argument for any other mode.
     */
    EvenKernelConvolution(size_t width, size_t height, size_t depth, Modes mode = Modes::CPU);

    /**
     * @brief Sets the kernel and precomputes its spectrum.
     * @param kernel Width*Height*Depth elements, wrapped layout.
     * @param relativeTolerance Passed to IsEvenPerAxis.
     * @throws std::invalid_argument if the kernel is not even along every axis:
     *         for such a kernel the result would silently be wrong.
     */
    void SetKernel(const T* kernel, double relativeTolerance = 1e-6);

    /// @brief data <- data * kernel (circular), in place. Requires SetKernel.
    void Apply(T* data);

    size_t Size() const noexcept { return _size; }

private:
    size_t _width;
    size_t _height;
    size_t _depth;
    size_t _size;
    HartleyTransform<T> _transform;
    std::vector<T> _spectrum; ///< S(g) / N, so Apply needs no extra scaling.
};

} // namespace RapiDHT

#endif // RAPIDHT_CONVOLUTION_H
