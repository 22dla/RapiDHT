/*
 * Project: RapiDHT
 * File: include/rapidht/convolution.h
 * Brief: Circular convolution through the Hartley transform: kernels even
 *        along every axis (separable transform, plain product) and any kernel
 *        (true transform, pair product), on the CPU and on the device; linear
 *        convolution of any extent by padding, with zero or normalised edges.
 * Author: Volkov Evgeny Aleksandrovich, volkov22dla@yandex.ru
 */

#ifndef RAPIDHT_CONVOLUTION_H
#define RAPIDHT_CONVOLUTION_H

#include <rapidht/device_volume.h>
#include <rapidht/transform.h>

#include <cstddef>
#include <memory>
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
 * On Modes::GPUFactored the kernel's spectrum lives on the device, and one
 * application is five passes over the volume: z and y of the transform, one
 * fused pass along x (transform, product, transform), then y and z again
 * (R-6). Apply(DeviceVolume&) works on a volume that is already there; the
 * host-pointer Apply uploads, convolves and downloads, through a device copy
 * of the volume allocated on first use.
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
     * @param mode Backend: Modes::CPU or Modes::GPUFactored.
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

    /**
     * @brief The same on a volume on the device, in place.
     * @throws std::invalid_argument Unless the mode is Modes::GPUFactored and
     *         the volume's size matches.
     */
    void Apply(DeviceVolume<T>& volume);

    size_t Size() const noexcept { return _size; }

private:
    size_t _width;
    size_t _height;
    size_t _depth;
    size_t _size;
    Modes _mode;
    HartleyTransform<T> _transform;
    std::vector<T> _spectrum; ///< S(g) / N, so Apply needs no extra scaling (CPU).
    std::unique_ptr<DeviceVolume<T>> _deviceSpectrum; ///< the same on the device (GPUFactored)
    std::unique_ptr<DeviceVolume<T>> _staging; ///< the host-pointer Apply's device copy
};

/**
 * @brief Several even kernels applied to the same fields: a bank of scales.
 *
 * A field is transformed once and multiplied by every kernel's spectrum, so K
 * outputs cost 1 + K separable transforms instead of the 2K that K separate
 * EvenKernelConvolution calls would take. On Modes::GPUFactored the field
 * goes through the z and y passes once, and each kernel costs a copy of that
 * and three passes (the fused x pass, y and z).
 *
 * Usage:
 * @code
 *   EvenKernelBank<float> bank(64, 64, 64);
 *   for (auto& w : windows) bank.AddKernel(w.data());
 *   bank.Apply(field.data(), out.data());   // out: bank.KernelCount() fields
 * @endcode
 *
 * @tparam T float or double.
 */
template <typename T>
class EvenKernelBank {
public:
    /// @copydoc EvenKernelConvolution::EvenKernelConvolution
    EvenKernelBank(size_t width, size_t height, size_t depth, Modes mode = Modes::CPU);

    /**
     * @brief Adds a kernel (wrapped layout, centre at index 0) and precomputes its spectrum.
     * @return Its position in the output of Apply.
     * @throws std::invalid_argument if the kernel is not even along every axis.
     */
    size_t AddKernel(const T* kernel, double relativeTolerance = 1e-6);

    /**
     * @brief out[k] <- field * kernel k (circular), for every kernel in the order added.
     * @param field Size() elements; left unchanged.
     * @param out KernelCount() * Size() elements, one field after another.
     * @throws std::logic_error if no kernel was added.
     */
    void Apply(const T* field, T* out);

    size_t KernelCount() const noexcept { return _mode == Modes::CPU ? _spectra.size() : _deviceSpectra.size(); }
    size_t Size() const noexcept { return _size; }

private:
    size_t _width;
    size_t _height;
    size_t _depth;
    size_t _size;
    Modes _mode;
    HartleyTransform<T> _transform;
    std::vector<std::vector<T>> _spectra; ///< S(g_k) / N each (CPU).
    std::vector<T> _scratch; ///< S(field) (CPU).
    std::vector<std::unique_ptr<DeviceVolume<T>>> _deviceSpectra; ///< S(g_k) / N each (GPUFactored)
    std::unique_ptr<DeviceVolume<T>> _field; ///< the field after the z and y passes (GPUFactored)
    std::unique_ptr<DeviceVolume<T>> _work; ///< one output at a time (GPUFactored)
};

/**
 * @brief Circular convolution with any kernel, through the true transform.
 *
 * The Hartley convolution theorem for a kernel without symmetry:
 *
 *     H(f * g)(k) = F(k) E(k) + F(-k) O(k),   E, O = (G(k) +- G(-k)) / 2,
 *
 * F = H(f), G = H(g), -k negated along every axis. So one application is the
 * true transform, this product of mirror pairs, and the true transform again
 * (with the 1/N folded into the kernel's spectrum, computed once in SetKernel).
 * For a kernel even along every axis EvenKernelConvolution gives the same
 * result for a little less work.
 *
 * On Modes::GPUFactored the whole product sits inside one fused pass along x,
 * with the Bracewell combination on both sides of it: five passes over the
 * volume (R-6), as for an even kernel.
 *
 * Usage:
 * @code
 *   CircularConvolution<float> conv(512, 512, 512, Modes::GPUFactored);
 *   conv.SetKernel(psf.data());        // wrapped layout, centre at index 0
 *   conv.Apply(deviceVolume);          // volume <- volume * psf, on the device
 * @endcode
 *
 * @tparam T float or double.
 */
template <typename T>
class CircularConvolution {
public:
    /// @copydoc EvenKernelConvolution::EvenKernelConvolution
    CircularConvolution(size_t width, size_t height, size_t depth, Modes mode = Modes::CPU);

    /// @brief Sets the kernel (Size() elements, wrapped layout) and precomputes its spectrum.
    void SetKernel(const T* kernel);

    /// @brief data <- data * kernel (circular), in place. Requires SetKernel.
    void Apply(T* data);

    /// @copydoc EvenKernelConvolution::Apply(DeviceVolume<T>&)
    void Apply(DeviceVolume<T>& volume);

    size_t Size() const noexcept { return _size; }

private:
    size_t _width;
    size_t _height;
    size_t _depth;
    size_t _size;
    Modes _mode;
    HartleyTransform<T> _transform;
    std::vector<T> _spectrum; ///< H(g) / N (CPU).
    std::unique_ptr<DeviceVolume<T>> _deviceSpectrum; ///< the same on the device (GPUFactored)
    std::unique_ptr<DeviceVolume<T>> _staging; ///< the host-pointer Apply's device copy
};

/// @brief What LinearConvolution assumes outside the volume.
enum class Edges {
    /// Zeros: the plain linear convolution, so a smoothing kernel darkens the
    /// border, where part of its weight falls outside.
    Zero,
    /// Normalised convolution (Knutsson & Westin): divide by the kernel's
    /// weight that falls inside the volume (or on the certainty, see Apply),
    ///
    ///     out = ((c f) * g) / (c * g),
    ///
    /// so a constant stays constant up to the border. Meant for non-negative
    /// smoothing kernels; where the denominator is too small to divide by,
    /// the output is 0.
    Normalized,
};

/**
 * @brief Linear (non-circular) convolution of a volume of any size with a
 *        small kernel, through zero padding and a circular convolution.
 *
 *     out(x) = sum_t f(x - t) g(t),   t = -r..r per axis,
 *
 * the output having the volume's extent and origin ("same" size, as
 * scipy.ndimage.convolve with mode="constant"). Each present axis of extent N
 * and kernel radius r is padded to the power of two P >= max(N + r, 2r + 1):
 * then the wrap-around of the circular convolution only ever reads padding,
 * so the result is exact. The volume itself need not be a power of two.
 *
 * The kernel is given compactly, with its centre in the middle: an odd extent
 * 2r + 1 along every present axis, element (r_x, r_y, r_z) the centre. A
 * kernel symmetric about its centre along every axis goes through
 * EvenKernelConvolution, any other through CircularConvolution. One Apply is
 * two transforms of the padded extent; Edges::Normalized with a certainty adds
 * two more (the denominator), without one the denominator is computed once,
 * in SetKernel.
 *
 * On Modes::GPUFactored the padded circular convolution runs on the device
 * (upload, five passes, download); padding and cropping stay on the host.
 *
 * Usage:
 * @code
 *   LinearConvolution<float> smooth(300, 300, 200, 9, 9, 9, Edges::Normalized);
 *   smooth.SetKernel(gauss9.data());        // 9*9*9, centre at (4, 4, 4)
 *   smooth.Apply(volume.data(), out.data());
 * @endcode
 *
 * @tparam T float or double.
 */
template <typename T>
class LinearConvolution {
public:
    /**
     * @param width, height, depth The volume's extent, any size; as for
     *        HartleyTransform, height == 0 means 1D and depth == 0 means 2D.
     * @param kernelWidth, kernelHeight, kernelDepth The kernel's extent: odd
     *        along every present axis, 0 or 1 along an absent one.
     * @param edges Zeros outside, or normalised convolution.
     * @param mode Modes::CPU or Modes::GPUFactored.
     * @throws std::invalid_argument On a zero width, a missing height with a
     *         depth, an even or missing kernel extent, or another mode.
     */
    LinearConvolution(size_t width, size_t height, size_t depth, size_t kernelWidth, size_t kernelHeight,
        size_t kernelDepth, Edges edges = Edges::Zero, Modes mode = Modes::CPU);

    /// @brief Sets the kernel (KernelSize() elements, centre in the middle).
    void SetKernel(const T* kernel);

    /**
     * @brief out <- data * kernel (linear), Size() elements each.
     * @param out May be the same pointer as data.
     */
    void Apply(const T* data, T* out);

    /**
     * @brief Normalised convolution with a certainty: out <- ((c f) * g) / (c * g).
     *
     * The certainty c (Size() elements, usually 0 or 1, any non-negative
     * weight) marks which voxels to trust -- say, 0 outside the cylinder of a
     * core sample -- so the untrusted ones do not leak into their neighbours.
     * Where c * g is too small (the kernel covers no trusted voxel) the
     * output is 0.
     * @throws std::logic_error Unless the edges are Edges::Normalized.
     */
    void Apply(const T* data, const T* certainty, T* out);

    size_t Size() const noexcept { return _size; }
    size_t KernelSize() const noexcept { return _kernelSize; }
    /// @brief The padded extent of the circular convolution (0 for an absent axis).
    size_t PaddedWidth() const noexcept { return _padded[0]; }
    size_t PaddedHeight() const noexcept { return _padded[1]; }
    size_t PaddedDepth() const noexcept { return _padded[2]; }

private:
    /// The volume into the zero-padded buffer, scaled voxel by voxel by `weight` if given.
    void Pad(const T* data, const T* weight);
    /// The padded buffer's volume part out to `out`.
    void Crop(T* out) const;
    /// The circular convolution of the padded buffer, in place.
    void ConvolvePadded();
    /// out /= denominator, 0 where the denominator is below the floor.
    void Divide(T* out, const T* denominator) const;

    size_t _extent[3]; ///< W, H, D of the volume (1 for an absent axis)
    size_t _radius[3]; ///< kernel radius per axis
    size_t _padded[3]; ///< padded extent as passed to the transform (0 for an absent axis)
    size_t _size;
    size_t _kernelSize;
    size_t _paddedSize;
    Edges _edges;
    Modes _mode;
    std::unique_ptr<EvenKernelConvolution<T>> _even; ///< for a kernel symmetric along every axis
    std::unique_ptr<CircularConvolution<T>> _general; ///< for any other
    std::vector<T> _buffer; ///< the padded volume
    std::vector<T> _weight; ///< c * g with c = 1 on the volume (Edges::Normalized)
    std::vector<T> _scratch; ///< the numerator, while the buffer holds c * g (certainty)
    T _floor = T(0); ///< denominators at or below it give 0
};

} // namespace RapiDHT

#endif // RAPIDHT_CONVOLUTION_H
