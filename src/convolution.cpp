/*
 * Project: RapiDHT
 * File: src/convolution.cpp
 * Brief: EvenKernelConvolution and IsEvenPerAxis.
 */

#include <rapidht/convolution.h>
#include <rapidht/utilities.h>

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
    _transform(width, height, depth, mode)
{
    if (mode != Modes::CPU) {
        throw std::invalid_argument(
            "EvenKernelConvolution: only Modes::CPU is supported for now.");
    }
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

    _spectrum.assign(kernel, kernel + _size);
    _transform.ForwardSeparable(_spectrum.data());

    // Applying S twice multiplies by N; fold the 1/N in here once.
    const T scale = static_cast<T>(1.0 / static_cast<double>(_size));
    for (auto& v : _spectrum) {
        v *= scale;
    }
}

template <typename T>
void EvenKernelConvolution<T>::Apply(T* data)
{
    PROFILE_FUNCTION();

    if (data == nullptr) {
        throw std::invalid_argument("EvenKernelConvolution::Apply: the pointer to data is null.");
    }
    if (_spectrum.empty()) {
        throw std::logic_error("EvenKernelConvolution::Apply: call SetKernel first.");
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
EvenKernelBank<T>::EvenKernelBank(size_t width, size_t height, size_t depth, Modes mode):
    _width(width),
    _height(height),
    _depth(depth),
    _size(width * Extent(height) * Extent(depth)),
    _transform(width, height, depth, mode)
{
    if (mode != Modes::CPU) {
        throw std::invalid_argument("EvenKernelBank: only Modes::CPU is supported for now.");
    }
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
    std::vector<T> spectrum(kernel, kernel + _size);
    _transform.ForwardSeparable(spectrum.data());
    const T scale = static_cast<T>(1.0 / static_cast<double>(_size));
    for (auto& v : spectrum) {
        v *= scale;
    }
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
    if (_spectra.empty()) {
        throw std::logic_error("EvenKernelBank::Apply: add a kernel first.");
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

template bool IsEvenPerAxis<float>(const float*, size_t, size_t, size_t, double);
template bool IsEvenPerAxis<double>(const double*, size_t, size_t, size_t, double);

template class EvenKernelConvolution<float>;
template class EvenKernelConvolution<double>;
template class EvenKernelBank<float>;
template class EvenKernelBank<double>;

} // namespace RapiDHT
