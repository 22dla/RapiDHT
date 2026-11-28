/*
 * Project: RapiDHT
 * File: src/filters.cpp
 * Brief: GaussianKernel, BallKernel, ButterworthKernel and WrapKernel.
 */

#include <rapidht/filters.h>
#include <rapidht/transform.h>

#include "internal/support.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace RapiDHT {

namespace {

void CheckDimensions(int dimensions, const char* caller)
{
    if (dimensions < 1 || dimensions > 3) {
        throw std::invalid_argument(std::string(caller) + ": the dimension must be 1, 2 or 3.");
    }
}

/// Spacing along the first `dimensions` axes; the others are not used.
void CheckSpacing(const Spacing& spacing, int dimensions, const char* caller)
{
    const double s[3] = { spacing.x, spacing.y, spacing.z };
    for (int a = 0; a < dimensions; ++a) {
        if (!(s[a] > 0.0) || !std::isfinite(s[a])) {
            throw std::invalid_argument(std::string(caller) + ": every voxel spacing must be positive and finite.");
        }
    }
}

/// An empty compact kernel of radius r[a] along the first `dimensions` axes.
template <typename T>
CompactKernel<T> Allocate(int dimensions, const size_t r[3])
{
    CompactKernel<T> k;
    k.width = 2 * r[0] + 1;
    k.height = dimensions >= 2 ? 2 * r[1] + 1 : 0;
    k.depth = dimensions >= 3 ? 2 * r[2] + 1 : 0;
    const size_t H = k.height == 0 ? 1 : k.height;
    const size_t D = k.depth == 0 ? 1 : k.depth;
    k.values.assign(k.width * H * D, T(0));
    return k;
}

/// Fills k with f(x, y, z) at the physical offsets of the voxel centres, then
/// scales it to unit sum if asked.
template <typename T, typename F>
void Sample(CompactKernel<T>& k, const Spacing& spacing, bool normalise, F&& f)
{
    const size_t W = k.width;
    const size_t H = k.height == 0 ? 1 : k.height;
    const size_t D = k.depth == 0 ? 1 : k.depth;
    auto offset = [](size_t i, size_t n, double d) {
        return (static_cast<double>(i) - static_cast<double>(n / 2)) * d;
    };
    std::vector<double> values(k.values.size());
    double sum = 0.0;
    for (size_t c = 0; c < D; ++c) {
        for (size_t b = 0; b < H; ++b) {
            for (size_t a = 0; a < W; ++a) {
                const double v = f(offset(a, W, spacing.x), offset(b, H, spacing.y), offset(c, D, spacing.z));
                values[a + W * (b + H * c)] = v;
                sum += v;
            }
        }
    }
    const double scale = normalise ? 1.0 / sum : 1.0;
    for (size_t n = 0; n < values.size(); ++n) {
        k.values[n] = static_cast<T>(values[n] * scale);
    }
}

size_t AtLeastOne(size_t n) { return n == 0 ? size_t { 1 } : n; }

/// The signed frequency index of DFT bin i of n: i, or i - n past the middle.
double SignedIndex(size_t i, size_t n)
{
    return i < (n + 1) / 2 ? static_cast<double>(i) : static_cast<double>(i) - static_cast<double>(n);
}

} // namespace

template <typename T>
CompactKernel<T> GaussianKernel(int dimensions, double sigma, Spacing spacing, double truncate)
{
    CheckDimensions(dimensions, "GaussianKernel");
    CheckSpacing(spacing, dimensions, "GaussianKernel");
    if (!(sigma > 0.0) || !std::isfinite(sigma)) {
        throw std::invalid_argument("GaussianKernel: sigma must be positive and finite.");
    }
    if (!(truncate > 0.0) || !std::isfinite(truncate)) {
        throw std::invalid_argument("GaussianKernel: truncate must be positive and finite.");
    }
    const double s[3] = { spacing.x, spacing.y, spacing.z };
    size_t r[3] = { 0, 0, 0 };
    for (int a = 0; a < dimensions; ++a) {
        r[a] = static_cast<size_t>(std::ceil(truncate * sigma / s[a]));
    }
    CompactKernel<T> k = Allocate<T>(dimensions, r);
    const double inv = 1.0 / (2.0 * sigma * sigma);
    Sample(k, spacing, true, [&](double x, double y, double z) { return std::exp(-(x * x + y * y + z * z) * inv); });
    return k;
}

template <typename T>
CompactKernel<T> BallKernel(int dimensions, double radius, Spacing spacing, bool normalise)
{
    CheckDimensions(dimensions, "BallKernel");
    CheckSpacing(spacing, dimensions, "BallKernel");
    if (!(radius >= 0.0) || !std::isfinite(radius)) {
        throw std::invalid_argument("BallKernel: the radius must be non-negative and finite.");
    }
    const double s[3] = { spacing.x, spacing.y, spacing.z };
    size_t r[3] = { 0, 0, 0 };
    for (int a = 0; a < dimensions; ++a) {
        r[a] = static_cast<size_t>(std::floor(radius / s[a]));
    }
    CompactKernel<T> k = Allocate<T>(dimensions, r);
    // A relative slack, so a voxel exactly on the sphere (radius 2, offset
    // (2, 0, 0)) is inside however the squares round.
    const double limit = radius * radius * (1.0 + 1e-12);
    Sample(k, spacing, normalise,
        [&](double x, double y, double z) { return x * x + y * y + z * z <= limit ? 1.0 : 0.0; });
    return k;
}

template <typename T>
std::vector<T> ButterworthKernel(size_t width, size_t height, size_t depth, double cutoff, int order, Spacing spacing)
{
    if (width == 0 || (height == 0 && depth != 0)) {
        throw std::invalid_argument("ButterworthKernel: the width must not be zero, and a depth needs a height.");
    }
    for (size_t n : { width, height, depth }) {
        if (n != 0 && !internal::IsPowerOfTwo(n)) {
            throw std::invalid_argument("ButterworthKernel: every extent must be a power of two, got "
                                        + std::to_string(n) + ".");
        }
    }
    const int dimensions = depth != 0 ? 3 : height != 0 ? 2
                                                        : 1;
    CheckSpacing(spacing, dimensions, "ButterworthKernel");
    if (!(cutoff > 0.0) || !std::isfinite(cutoff)) {
        throw std::invalid_argument("ButterworthKernel: the cutoff must be positive and finite.");
    }
    if (order < 1) {
        throw std::invalid_argument("ButterworthKernel: the order must be 1 or more.");
    }

    const size_t W = width;
    const size_t H = AtLeastOne(height);
    const size_t D = AtLeastOne(depth);
    const size_t N = W * H * D;

    // B on the frequency grid. It is even along every axis (|f| is), and the
    // Nyquist bin has the same |f| from either side.
    std::vector<double> response(N);
    for (size_t k = 0; k < D; ++k) {
        const double fz = SignedIndex(k, D) / (static_cast<double>(D) * spacing.z);
        for (size_t j = 0; j < H; ++j) {
            const double fy = SignedIndex(j, H) / (static_cast<double>(H) * spacing.y);
            for (size_t i = 0; i < W; ++i) {
                const double fx = SignedIndex(i, W) / (static_cast<double>(W) * spacing.x);
                const double ratio2 = (fx * fx + (H > 1 ? fy * fy : 0.0) + (D > 1 ? fz * fz : 0.0)) / (cutoff * cutoff);
                response[i + W * (j + H * k)] = 1.0 / (1.0 + std::pow(ratio2, order));
            }
        }
    }

    // The kernel whose spectrum is B: for an even function the separable
    // transform is its own inverse up to 1/N.
    HartleyTransform<double> transform(width, height, depth, Modes::CPU);
    transform.ForwardSeparable(response.data());
    std::vector<T> kernel(N);
    const double scale = 1.0 / static_cast<double>(N);
    for (size_t n = 0; n < N; ++n) {
        kernel[n] = static_cast<T>(response[n] * scale);
    }
    return kernel;
}

template <typename T>
std::vector<T> WrapKernel(const CompactKernel<T>& kernel, size_t width, size_t height, size_t depth)
{
    if (width == 0 || (height == 0) != (kernel.height == 0) || (depth == 0) != (kernel.depth == 0)) {
        throw std::invalid_argument("WrapKernel: the kernel and the volume must have the same dimensionality.");
    }
    const size_t KW = kernel.width;
    const size_t KH = AtLeastOne(kernel.height);
    const size_t KD = AtLeastOne(kernel.depth);
    if (KW % 2 == 0 || KH % 2 == 0 || KD % 2 == 0 || kernel.values.size() != KW * KH * KD) {
        throw std::invalid_argument("WrapKernel: the kernel needs odd extents and width*height*depth values.");
    }
    const size_t W = width;
    const size_t H = AtLeastOne(height);
    const size_t D = AtLeastOne(depth);
    // Tap a sits at offset a - r; adding a multiple of N keeps the modulo
    // non-negative however wide the kernel is.
    auto wrap = [](size_t a, size_t k, size_t n) { return (a + n * (k / 2 / n + 1) - k / 2) % n; };
    std::vector<T> wrapped(W * H * D, T(0));
    for (size_t c = 0; c < KD; ++c) {
        const size_t k = wrap(c, KD, D);
        for (size_t b = 0; b < KH; ++b) {
            const size_t j = wrap(b, KH, H);
            for (size_t a = 0; a < KW; ++a) {
                wrapped[wrap(a, KW, W) + W * (j + H * k)] += kernel.values[a + KW * (b + KH * c)];
            }
        }
    }
    return wrapped;
}

template CompactKernel<float> GaussianKernel<float>(int, double, Spacing, double);
template CompactKernel<double> GaussianKernel<double>(int, double, Spacing, double);
template CompactKernel<float> BallKernel<float>(int, double, Spacing, bool);
template CompactKernel<double> BallKernel<double>(int, double, Spacing, bool);
template std::vector<float> ButterworthKernel<float>(size_t, size_t, size_t, double, int, Spacing);
template std::vector<double> ButterworthKernel<double>(size_t, size_t, size_t, double, int, Spacing);
template std::vector<float> WrapKernel<float>(const CompactKernel<float>&, size_t, size_t, size_t);
template std::vector<double> WrapKernel<double>(const CompactKernel<double>&, size_t, size_t, size_t);

} // namespace RapiDHT
