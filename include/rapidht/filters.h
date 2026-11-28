/*
 * Project: RapiDHT
 * File: include/rapidht/filters.h
 * Brief: Convolution kernels in physical units on an anisotropic voxel grid:
 *        compact Gaussian and ball kernels for LinearConvolution, the
 *        Butterworth low-pass for the circular convolutions, and the wrapped
 *        layout of a compact kernel.
 * Author: Volkov Evgeny Aleksandrovich, volkov22dla@yandex.ru
 */

#ifndef RAPIDHT_FILTERS_H
#define RAPIDHT_FILTERS_H

#include <cstddef>
#include <vector>

namespace RapiDHT {

/**
 * @brief The voxel's edge lengths along x, y and z, in any one unit of length.
 *
 * A tomogram's slices are often thicker than its pixels (dz > dx): a kernel
 * given in voxels then smooths more along z than across, and every length the
 * filters below take -- sigma, radius, cutoff -- is in the unit of these.
 */
struct Spacing {
    double x = 1.0;
    double y = 1.0;
    double z = 1.0;
};

/**
 * @brief A compact kernel: odd extents, centre in the middle.
 *
 * The layout LinearConvolution takes: element (width/2, height/2, depth/2) is
 * the centre, stored as i + width*(j + height*k). An absent axis has extent 0
 * (as for HartleyTransform: height == 0 is 1D, depth == 0 is 2D).
 */
template <typename T>
struct CompactKernel {
    std::vector<T> values;
    size_t width = 1;
    size_t height = 0;
    size_t depth = 0;
};

/**
 * @brief The sampled Gaussian exp(-|x|^2 / (2 sigma^2)), x the physical offset of the voxel centre.
 *
 * Along axis a it reaches ceil(truncate * sigma / spacing_a) voxels from the
 * centre (scipy.ndimage.gaussian_filter cuts at 4 sigma too), so on a grid
 * with dz = 2 dx the kernel is half as deep as it is wide. Separable; normalised
 * to unit sum, so it keeps the mean of what it smooths.
 *
 * @param dimensions 1, 2 or 3.
 * @param sigma Standard deviation in the unit of `spacing`; positive.
 * @param truncate Half-width in sigmas; positive.
 * @throws std::invalid_argument on a bad dimension, sigma, truncate or spacing.
 */
template <typename T>
CompactKernel<T> GaussianKernel(int dimensions, double sigma, Spacing spacing = {}, double truncate = 4.0);

/**
 * @brief The ball |x| <= radius: 1 at every voxel whose centre lies inside, 0 elsewhere.
 *
 * Along axis a it reaches floor(radius / spacing_a) voxels from the centre. A
 * radius below every spacing gives the single centre voxel.
 *
 * @param normalise Divide by the voxel count, for a moving average; otherwise
 *        the kernel is the 0/1 indicator, for a moving sum.
 * @throws std::invalid_argument on a bad dimension, a negative radius or a bad spacing.
 */
template <typename T>
CompactKernel<T> BallKernel(int dimensions, double radius, Spacing spacing = {}, bool normalise = true);

/**
 * @brief The Butterworth low-pass, as a kernel in the wrapped layout for the circular convolutions.
 *
 * Its transfer function is
 *
 *     B(f) = 1 / (1 + (|f| / cutoff)^(2 order)),
 *
 * f the frequency in cycles per unit length: f_x = m / (width * spacing.x) for
 * the signed index m in [-width/2, width/2), and so on. B is real and even
 * along every axis, so the kernel is too, its separable and true Hartley
 * spectra both equal B, and EvenKernelConvolution applies it exactly. B(0) = 1
 * (the kernel sums to 1) and B = 1/2 at |f| = cutoff. Unlike the Gaussian it
 * has no compact form: it is built over the whole extent of the volume.
 *
 * @param width, height, depth The extent of the volume, as for HartleyTransform;
 *        powers of two.
 * @param cutoff Half-power frequency in cycles per unit length; positive.
 *        The Nyquist frequency along axis a is 1 / (2 spacing_a).
 * @param order Steepness; 1 or more.
 * @throws std::invalid_argument on a bad extent, cutoff, order or spacing.
 */
template <typename T>
std::vector<T> ButterworthKernel(
    size_t width, size_t height, size_t depth, double cutoff, int order, Spacing spacing = {});

/**
 * @brief A compact kernel in the wrapped layout of a volume: centre at index 0.
 *
 * Tap t (t from -r to r along each axis) goes to index t mod N, the layout
 * EvenKernelConvolution, EvenKernelBank and CircularConvolution take. Taps
 * that land on the same index -- a kernel wider than the volume -- are summed,
 * which is what the circular convolution with the compact kernel would do.
 *
 * @param width, height, depth The volume's extent; the kernel must have the
 *        same dimensionality (an absent axis has extent 0 in both).
 * @throws std::invalid_argument if the dimensionality or the kernel's size does not match.
 */
template <typename T>
std::vector<T> WrapKernel(const CompactKernel<T>& kernel, size_t width, size_t height, size_t depth);

} // namespace RapiDHT

#endif // RAPIDHT_FILTERS_H
