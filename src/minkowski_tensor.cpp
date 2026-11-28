/*
 * Project: RapiDHT
 * File: src/minkowski_tensor.cpp
 * Brief: SurfaceNormalTensor: the integral of n (x) n dS from the gradient of
 *        the smoothed body (B-205).
 */

#include <rapidht/minkowski.h>
#include <rapidht/utilities.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace RapiDHT {

namespace {

/*
 * The derivative of a Gaussian along `axis`, in the wrapped layout of the volume:
 * -x_a / sigma^2 G(x), physical offsets, truncated at 4 sigma, then scaled so that
 * -sum x_a D(x) = 1 -- the convolution then gives a linear ramp's slope exactly, and
 * the gradient of a step integrates to exactly its height.
 */
template <typename T>
std::vector<T> GaussianDerivative(size_t width, size_t height, size_t depth, double sigma, Spacing spacing, int axis)
{
    const CompactKernel<double> g = GaussianKernel<double>(3, sigma, spacing);
    const double step[3] = { spacing.x, spacing.y, spacing.z };
    const size_t extent[3] = { g.width, g.height, g.depth };
    CompactKernel<double> d = g;
    double moment = 0.0;
    for (size_t c = 0; c < g.depth; ++c) {
        for (size_t b = 0; b < g.height; ++b) {
            for (size_t a = 0; a < g.width; ++a) {
                const size_t index[3] = { a, b, c };
                const double x
                    = (static_cast<double>(index[axis]) - static_cast<double>(extent[axis] / 2)) * step[axis];
                const size_t n = a + g.width * (b + g.height * c);
                d.values[n] = -x / (sigma * sigma) * g.values[n];
                moment -= x * d.values[n];
            }
        }
    }
    CompactKernel<T> scaled { std::vector<T>(d.values.size()), d.width, d.height, d.depth };
    for (size_t n = 0; n < d.values.size(); ++n) {
        scaled.values[n] = static_cast<T>(d.values[n] / moment);
    }
    return WrapKernel(scaled, width, height, depth);
}

} // namespace

template <typename T>
SurfaceNormalTensor<T>::SurfaceNormalTensor(
    size_t width, size_t height, size_t depth, double sigma, Spacing spacing, Modes mode):
    _size(width * height * depth),
    _voxelVolume(spacing.x * spacing.y * spacing.z),
    _dx(width, height, depth, mode),
    _dy(width, height, depth, mode),
    _dz(width, height, depth, mode)
{
    // CircularConvolution has checked the mode and the powers of two.
    if (width < 2 || height < 2 || depth < 2) {
        throw std::invalid_argument("SurfaceNormalTensor: the volume must be 3D, every extent at least 2.");
    }
    // GaussianKernel checks sigma and the spacing.
    _dx.SetKernel(GaussianDerivative<T>(width, height, depth, sigma, spacing, 0).data());
    _dy.SetKernel(GaussianDerivative<T>(width, height, depth, sigma, spacing, 1).data());
    _dz.SetKernel(GaussianDerivative<T>(width, height, depth, sigma, spacing, 2).data());
}

template <typename T>
void SurfaceNormalTensor<T>::Densities(const uint8_t* volume, T* out)
{
    PROFILE_FUNCTION();

    if (volume == nullptr || out == nullptr) {
        throw std::invalid_argument("SurfaceNormalTensor: null pointer.");
    }
    const size_t N = _size;
    _gradient.resize(3 * N);
    T* gx = _gradient.data();
    T* gy = gx + N;
    T* gz = gy + N;
    for (size_t n = 0; n < N; ++n) {
        gx[n] = volume[n] != 0 ? T(1) : T(0);
    }
    std::copy(gx, gx + N, gy);
    std::copy(gx, gx + N, gz);
    _dx.Apply(gx);
    _dy.Apply(gy);
    _dz.Apply(gz);

    const T dv = static_cast<T>(_voxelVolume);
    const long long count = static_cast<long long>(N);
#pragma omp parallel for
    for (long long nn = 0; nn < count; ++nn) {
        const size_t n = static_cast<size_t>(nn);
        const T a = gx[n];
        const T b = gy[n];
        const T c = gz[n];
        const T norm = std::sqrt(a * a + b * b + c * c);
        const T scale = norm > T(0) ? dv / norm : T(0);
        out[n] = a * a * scale;
        out[N + n] = b * b * scale;
        out[2 * N + n] = c * c * scale;
        out[3 * N + n] = a * b * scale;
        out[4 * N + n] = a * c * scale;
        out[5 * N + n] = b * c * scale;
    }
}

template <typename T>
SurfaceTensor SurfaceNormalTensor<T>::Compute(const uint8_t* volume)
{
    PROFILE_FUNCTION();

    _densities.resize(6 * _size);
    Densities(volume, _densities.data());
    double sums[6] = { 0, 0, 0, 0, 0, 0 };
    for (size_t p = 0; p < 6; ++p) {
        const T* plane = _densities.data() + p * _size;
        double s = 0.0;
        for (size_t n = 0; n < _size; ++n) {
            s += static_cast<double>(plane[n]);
        }
        sums[p] = s;
    }
    SurfaceTensor t;
    t.xx = sums[0];
    t.yy = sums[1];
    t.zz = sums[2];
    t.xy = sums[3];
    t.xz = sums[4];
    t.yz = sums[5];
    return t;
}

template class SurfaceNormalTensor<float>;
template class SurfaceNormalTensor<double>;

} // namespace RapiDHT
