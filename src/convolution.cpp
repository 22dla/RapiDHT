/*
 * Project: RapiDHT
 * File: src/convolution.cpp
 * Brief: EvenKernelConvolution, EvenKernelBank, CircularConvolution and
 *        IsEvenPerAxis.
 */

#include <rapidht/convolution.h>
#include <rapidht/utilities.h>

#include "internal/device_access.h"
#include "internal/support.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace RapiDHT {

namespace {

size_t Mirror(size_t i, size_t n) { return i == 0 ? 0 : n - i; }

size_t Extent(size_t n) { return n == 0 ? size_t { 1 } : n; }

/// The 2D/3D transforms would reject such extents only inside an OpenMP region,
/// which terminates the process instead of throwing; check up front.
void CheckPowersOfTwo(size_t width, size_t height, size_t depth, const char* caller)
{
    for (size_t n : { width, height, depth }) {
        if (n != 0 && !internal::IsPowerOfTwo(n)) {
            throw std::invalid_argument(std::string(caller) + ": every extent must be a power of two, got "
                                        + std::to_string(n) + ".");
        }
    }
}

/// The convolutions run on the CPU or on the device through the factored
/// transform; the matrix and FFT modes have no device convolution.
void CheckConvolutionMode(Modes mode, const char* caller)
{
    if (mode != Modes::CPU && mode != Modes::GPUFactored) {
        throw std::invalid_argument(std::string(caller) + ": only Modes::CPU and Modes::GPUFactored are supported.");
    }
}

/// g / N on the host: the transform of it is the spectrum with the 1/N of the
/// round trip folded in, so no scaling pass is needed on the device.
template <typename T>
std::vector<T> ScaledKernel(const T* kernel, size_t size)
{
    const T scale = static_cast<T>(1.0 / static_cast<double>(size));
    std::vector<T> g(kernel, kernel + size);
    for (auto& v : g) {
        v *= scale;
    }
    return g;
}

/// A device copy of g / N, transformed in place on the device (true or
/// separable) -- the kernel's spectrum where the device convolution reads it.
template <typename T>
std::unique_ptr<DeviceVolume<T>> DeviceSpectrum(HartleyTransform<T>& transform, const T* kernel, size_t size, bool separable)
{
    auto spectrum = std::make_unique<DeviceVolume<T>>(size);
    spectrum->Upload(ScaledKernel(kernel, size).data());
    if (separable) {
        transform.ForwardSeparable(*spectrum);
    } else {
        transform.ForwardTransform(*spectrum);
    }
    return spectrum;
}

/// The host-pointer Apply on the device: upload, convolve, download, through a
/// device copy allocated on first use.
template <typename T, typename Convolve>
void ApplyThroughDevice(std::unique_ptr<DeviceVolume<T>>& staging, size_t size, T* data, Convolve&& convolve)
{
    if (!staging) {
        staging = std::make_unique<DeviceVolume<T>>(size);
    }
    staging->Upload(data);
    convolve(*staging);
    staging->Download(data);
}

} // namespace

template <typename T>
bool IsEvenPerAxis(const T* kernel, size_t width, size_t height, size_t depth, double relativeTolerance)
{
    if (kernel == nullptr) {
        throw std::invalid_argument("IsEvenPerAxis: the pointer to kernel is null.");
    }
    const size_t W = width;
    const size_t H = Extent(height);
    const size_t D = Extent(depth);
    const size_t total = W * H * D;

    double peak = 0.0;
    for (size_t n = 0; n < total; ++n) {
        peak = std::max(peak, std::fabs(static_cast<double>(kernel[n])));
    }
    const double tol = relativeTolerance * std::max(peak, 1e-300);

    auto at = [&](size_t i, size_t j, size_t k) { return static_cast<double>(kernel[i + W * (j + H * k)]); };

    // Evenness along each axis separately implies evenness under any
    // combination of flips, so three comparisons per point are enough.
    for (size_t k = 0; k < D; ++k) {
        for (size_t j = 0; j < H; ++j) {
            for (size_t i = 0; i < W; ++i) {
                const double v = at(i, j, k);
                if (std::fabs(v - at(Mirror(i, W), j, k)) > tol
                    || std::fabs(v - at(i, Mirror(j, H), k)) > tol
                    || std::fabs(v - at(i, j, Mirror(k, D))) > tol) {
                    return false;
                }
            }
        }
    }
    return true;
}

template <typename T>
EvenKernelConvolution<T>::EvenKernelConvolution(size_t width, size_t height, size_t depth, Modes mode):
    _width(width),
    _height(height),
    _depth(depth),
    _size(width * Extent(height) * Extent(depth)),
    _mode(mode),
    _transform(width, height, depth, mode)
{
    CheckConvolutionMode(mode, "EvenKernelConvolution");
    CheckPowersOfTwo(width, height, depth, "EvenKernelConvolution");
}

template <typename T>
void EvenKernelConvolution<T>::SetKernel(const T* kernel, double relativeTolerance)
{
    PROFILE_FUNCTION();

    if (!IsEvenPerAxis(kernel, _width, _height, _depth, relativeTolerance)) {
        throw std::invalid_argument(
            "EvenKernelConvolution: the kernel is not even along every axis "
            "(g(i) != g(-i) mod the extent). Store it with its centre at index 0, "
            "or use a general convolution.");
    }

    // Applying S twice multiplies by N; fold the 1/N in here once.
    if (_mode == Modes::GPUFactored) {
        _deviceSpectrum = DeviceSpectrum(_transform, kernel, _size, true);
        return;
    }
    _spectrum = ScaledKernel(kernel, _size);
    _transform.ForwardSeparable(_spectrum.data());
}

template <typename T>
void EvenKernelConvolution<T>::Apply(T* data)
{
    PROFILE_FUNCTION();

    if (data == nullptr) {
        throw std::invalid_argument("EvenKernelConvolution::Apply: the pointer to data is null.");
    }
    if (_spectrum.empty() && !_deviceSpectrum) {
        throw std::logic_error("EvenKernelConvolution::Apply: call SetKernel first.");
    }
    if (_mode == Modes::GPUFactored) {
        ApplyThroughDevice(_staging, _size, data, [&](DeviceVolume<T>& v) { Apply(v); });
        return;
    }

    _transform.ForwardSeparable(data);

    const long long count = static_cast<long long>(_size);
    const T* spectrum = _spectrum.data();
#pragma omp parallel for
    for (long long n = 0; n < count; ++n) {
        data[n] *= spectrum[n];
    }

    _transform.ForwardSeparable(data);
}

template <typename T>
void EvenKernelConvolution<T>::Apply(DeviceVolume<T>& volume)
{
    PROFILE_FUNCTION();

    if (_mode != Modes::GPUFactored) {
        throw std::invalid_argument("EvenKernelConvolution::Apply(DeviceVolume&) needs Modes::GPUFactored.");
    }
    if (!_deviceSpectrum) {
        throw std::logic_error("EvenKernelConvolution::Apply: call SetKernel first.");
    }
    internal::DeviceAccess::Convolve(_transform, volume, *_deviceSpectrum, false);
}

template <typename T>
EvenKernelBank<T>::EvenKernelBank(size_t width, size_t height, size_t depth, Modes mode):
    _width(width),
    _height(height),
    _depth(depth),
    _size(width * Extent(height) * Extent(depth)),
    _mode(mode),
    _transform(width, height, depth, mode)
{
    CheckConvolutionMode(mode, "EvenKernelBank");
    CheckPowersOfTwo(width, height, depth, "EvenKernelBank");
}

template <typename T>
size_t EvenKernelBank<T>::AddKernel(const T* kernel, double relativeTolerance)
{
    PROFILE_FUNCTION();

    if (!IsEvenPerAxis(kernel, _width, _height, _depth, relativeTolerance)) {
        throw std::invalid_argument(
            "EvenKernelBank: the kernel is not even along every axis "
            "(g(i) != g(-i) mod the extent). Store it with its centre at index 0.");
    }
    if (_mode == Modes::GPUFactored) {
        _deviceSpectra.push_back(DeviceSpectrum(_transform, kernel, _size, true));
        return _deviceSpectra.size() - 1;
    }
    std::vector<T> spectrum = ScaledKernel(kernel, _size);
    _transform.ForwardSeparable(spectrum.data());
    _spectra.push_back(std::move(spectrum));
    return _spectra.size() - 1;
}

template <typename T>
void EvenKernelBank<T>::Apply(const T* field, T* out)
{
    PROFILE_FUNCTION();

    if (field == nullptr || out == nullptr) {
        throw std::invalid_argument("EvenKernelBank::Apply: null pointer.");
    }
    if (KernelCount() == 0) {
        throw std::logic_error("EvenKernelBank::Apply: add a kernel first.");
    }
    if (_mode == Modes::GPUFactored) {
        if (!_field) {
            _field = std::make_unique<DeviceVolume<T>>(_size);
            _work = std::make_unique<DeviceVolume<T>>(_size);
        }
        _field->Upload(field);
        internal::DeviceAccess::ConvolvePrepare(_transform, *_field);
        for (size_t k = 0; k < _deviceSpectra.size(); ++k) {
            internal::DeviceAccess::Convolve(_transform, *_work, *_deviceSpectra[k], false, _field.get());
            _work->Download(out + k * _size);
        }
        return;
    }

    _scratch.assign(field, field + _size);
    _transform.ForwardSeparable(_scratch.data());

    const long long count = static_cast<long long>(_size);
    const T* f = _scratch.data();
    for (size_t k = 0; k < _spectra.size(); ++k) {
        T* o = out + k * _size;
        const T* s = _spectra[k].data();
#pragma omp parallel for
        for (long long n = 0; n < count; ++n) {
            o[n] = f[n] * s[n];
        }
        _transform.ForwardSeparable(o);
    }
}

template <typename T>
CircularConvolution<T>::CircularConvolution(size_t width, size_t height, size_t depth, Modes mode):
    _width(width),
    _height(height),
    _depth(depth),
    _size(width * Extent(height) * Extent(depth)),
    _mode(mode),
    _transform(width, height, depth, mode)
{
    CheckConvolutionMode(mode, "CircularConvolution");
    CheckPowersOfTwo(width, height, depth, "CircularConvolution");
}

template <typename T>
void CircularConvolution<T>::SetKernel(const T* kernel)
{
    PROFILE_FUNCTION();

    if (kernel == nullptr) {
        throw std::invalid_argument("CircularConvolution::SetKernel: the pointer to kernel is null.");
    }
    if (_mode == Modes::GPUFactored) {
        _deviceSpectrum = DeviceSpectrum(_transform, kernel, _size, false);
        return;
    }
    _spectrum = ScaledKernel(kernel, _size);
    _transform.ForwardTransform(_spectrum.data());
}

template <typename T>
void CircularConvolution<T>::Apply(T* data)
{
    PROFILE_FUNCTION();

    if (data == nullptr) {
        throw std::invalid_argument("CircularConvolution::Apply: the pointer to data is null.");
    }
    if (_spectrum.empty() && !_deviceSpectrum) {
        throw std::logic_error("CircularConvolution::Apply: call SetKernel first.");
    }
    if (_mode == Modes::GPUFactored) {
        ApplyThroughDevice(_staging, _size, data, [&](DeviceVolume<T>& v) { Apply(v); });
        return;
    }

    _transform.ForwardTransform(data);

    // Z(k) = F(k) E(k) + F(-k) O(k); the lower index of each pair (k, -k)
    // writes both, Z(-k) = F(-k) E(k) - F(k) O(k).
    const size_t W = _width;
    const size_t H = Extent(_height);
    const size_t D = Extent(_depth);
    const T* g = _spectrum.data();
    const long long depthCount = static_cast<long long>(D);
#pragma omp parallel for
    for (long long kk = 0; kk < depthCount; ++kk) {
        const size_t k = static_cast<size_t>(kk);
        for (size_t j = 0; j < H; ++j) {
            for (size_t i = 0; i < W; ++i) {
                const size_t a = i + W * (j + H * k);
                const size_t b = Mirror(i, W) + W * (Mirror(j, H) + H * Mirror(k, D));
                if (a > b) {
                    continue;
                }
                const T fa = data[a];
                const T fb = data[b];
                const T even = static_cast<T>(0.5) * (g[a] + g[b]);
                const T odd = static_cast<T>(0.5) * (g[a] - g[b]);
                data[a] = fa * even + fb * odd;
                if (b != a) {
                    data[b] = fb * even - fa * odd;
                }
            }
        }
    }

    _transform.ForwardTransform(data);
}

template <typename T>
void CircularConvolution<T>::Apply(DeviceVolume<T>& volume)
{
    PROFILE_FUNCTION();

    if (_mode != Modes::GPUFactored) {
        throw std::invalid_argument("CircularConvolution::Apply(DeviceVolume&) needs Modes::GPUFactored.");
    }
    if (!_deviceSpectrum) {
        throw std::logic_error("CircularConvolution::Apply: call SetKernel first.");
    }
    internal::DeviceAccess::Convolve(_transform, volume, *_deviceSpectrum, true);
}

template bool IsEvenPerAxis<float>(const float*, size_t, size_t, size_t, double);
template bool IsEvenPerAxis<double>(const double*, size_t, size_t, size_t, double);

template class EvenKernelConvolution<float>;
template class EvenKernelConvolution<double>;
template class EvenKernelBank<float>;
template class EvenKernelBank<double>;
template class CircularConvolution<float>;
template class CircularConvolution<double>;

} // namespace RapiDHT
