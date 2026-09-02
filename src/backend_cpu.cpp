/*
 * Project: RapiDHT
 * File: src/backend_cpu.cpp
 * Brief: The OpenMP backend: butterflies, twiddles and the Bracewell passes.
 */

#include <rapidht/transform.h>
#include <rapidht/utilities.h>

#include "internal/support.h"

#include <cmath>
#include <complex>
#include <omp.h>
#include <string>

namespace RapiDHT {

using internal::IsPowerOfTwo;

template <typename T>
void HartleyTransform<T>::BitReverse(std::vector<size_t>& indices)
{
    PROFILE_FUNCTION();

    if (indices.empty()) {
        return;
    }

    const size_t n = indices.size();
    const int kLog2n = static_cast<int>(std::log2(n));

    indices[0] = 0;
    for (size_t j = 1; j < n; ++j) {
        size_t reversed = 0;
        size_t temp = j;
        for (int i = 0; i < kLog2n; ++i) {
            if (temp & 1) {
                // size_t{1}, not 1: shifting an int is undefined past 31 bits.
                reversed |= size_t { 1 } << (kLog2n - 1 - i);
            }
            temp >>= 1;
        }
        indices[j] = reversed;
    }
}

template <typename T>
void HartleyTransform<T>::BuildTwiddleTable(Direction direction)
{
    PROFILE_FUNCTION();

    const size_t n = Length(direction);
    auto& table = _twiddles[static_cast<size_t>(direction)];
    table.clear();

    // Sizes that FDHT1D would reject need no table; it throws on its own.
    if (n < 4 || !IsPowerOfTwo(n)) {
        return;
    }

    const T kPi = static_cast<T>(std::acos(-1));
    table.resize(n);

    // Mirrors the loop structure of FDHT1D: stage s uses m2 = 2^(s-1) and
    // indices j in [1, m2/2), with angle j*pi/m2.
    for (size_t s = 2; (size_t(1) << s) <= n; ++s) {
        const size_t m2 = size_t(1) << (s - 1);
        const size_t m4 = m2 / 2;
        for (size_t j = 1; j < m4; ++j) {
            const T angle = static_cast<T>(j) * kPi / static_cast<T>(m2);
            table[m2 + j] = Twiddle { std::cos(angle), std::sin(angle) };
        }
    }
}

template <typename T>
void HartleyTransform<T>::Series1D(T* data, Direction direction)
{
    PROFILE_FUNCTION();

    if (data == nullptr) {
        throw std::invalid_argument("The pointer to image is null.");
    }

    size_t M1 = 0, M2 = 0;
    switch (direction) {
        case Direction::Y:
            M1 = Height();
            M2 = (Depth() == 0 ? 1 : Depth());
            break;
        case Direction::X:
            M1 = Width();
            M2 = (Depth() == 0 ? 1 : Depth());
            break;
        case Direction::Z:
            M1 = Width();
            M2 = Height();
            break;
        default:
            throw std::invalid_argument("Invalid direction");
    }

    if (_mode == Modes::CPU) {
#pragma omp parallel for
        for (int i = 0; i < M1; ++i) {
            for (size_t j = 0; j < M2; ++j) {
                auto index = AxisIndex(0, i, j, direction);
                FDHT1D(data + index, direction);
            }
        }
        return;
    }
    // if (_mode == Modes::RFFT) {
    // #pragma omp parallel for
    //	for (int i = 0; i < Width(); ++i) {
    //		RealFFT1D(image_ptr + i * Height(), direction);
    //	}
    //	return;
    // }
}

template <typename T>
void HartleyTransform<T>::BracewellTransform2DCPU(T* image_ptr)
{
    PROFILE_FUNCTION();

    int W = Width();
    int H = Height();

    std::vector<T> result(W * H, T(0));

    // collapse(N) требует идеально вложенных циклов: объявления переносим внутрь
#pragma omp parallel for collapse(2)
    for (int y = 0; y < W; ++y) {
        for (int x = 0; x < H; ++x) {
            const int ym = (y > 0) ? (W - y) : 0;
            const int xm = (x > 0) ? (H - x) : 0;

            const T A = image_ptr[LinearIndex(y, x, 0)];
            const T B = image_ptr[LinearIndex(y, xm, 0)]; // flip X
            const T C = image_ptr[LinearIndex(ym, x, 0)]; // flip Y
            const T D = image_ptr[LinearIndex(ym, xm, 0)]; // flip both

            result[LinearIndex(y, x, 0)] = (A + B + C - D) / static_cast<T>(2);
        }
    }

    std::copy(result.begin(), result.end(), image_ptr);
}

template <typename T>
void HartleyTransform<T>::BracewellTransform3DCPU(T* volumePtr)
{
    PROFILE_FUNCTION();
    int W = Width();
    int H = Height();
    int D = Depth();

    std::vector<T> result(W * H * D, T(0));

    // collapse(N) требует идеально вложенных циклов: объявления переносим внутрь
#pragma omp parallel for collapse(3)
    for (int y = 0; y < W; ++y) {
        for (int x = 0; x < H; ++x) {
            for (int z = 0; z < D; ++z) {
                const int ym = (y > 0) ? (W - y) : 0;
                const int xm = (x > 0) ? (H - x) : 0;
                const int zm = (z > 0) ? (D - z) : 0;

                const T A = volumePtr[LinearIndex(ym, x, z)]; // flip X
                const T B = volumePtr[LinearIndex(y, xm, z)]; // flip Y
                const T C = volumePtr[LinearIndex(y, x, zm)]; // flip Z
                const T D_ = volumePtr[LinearIndex(ym, xm, zm)]; // flip all

                result[LinearIndex(y, x, z)] = (A + B + C - D_) / static_cast<T>(2);
            }
        }
    }

    std::copy(result.begin(), result.end(), volumePtr);
}

template <typename T>
void HartleyTransform<T>::FDHT1D(T* data, Direction direction)
{
    if (data == nullptr) {
        throw std::invalid_argument("The pointer to vector is null.");
    }

    const size_t n = Length(direction);

    // No lower-bound check: n is size_t, so "n < 0" was always false.
    if (!IsPowerOfTwo(n)) {
        throw std::invalid_argument("FDHT1D: length must be a power of two, got "
                                    + std::to_string(n) + ".");
    }

    // временный буфер
    std::vector<T> vec(n);

    // собрать данные в буфер
    for (size_t idx = 0; idx < n; ++idx) {
        vec[idx] = data[AxisIndex(idx, 0, 0, direction)];
    }

    for (size_t i = 1; i < n; ++i) {
        auto j = BitReversedIndex(direction, i);
        if (j > i) {
            std::swap(vec[i], vec[j]);
        }
    }

    // FHT for 1rd axis
    const auto kLog2n = static_cast<size_t>(std::log2(n));
    const Twiddle* twiddles = _twiddles[static_cast<size_t>(direction)].data();

    // Main cicle
    for (size_t s = 1; s <= kLog2n; ++s) {
        const auto m = size_t(1) << s;
        const auto m2 = m / 2;
        const auto m4 = m / 4;

        // Hoisted out of the loop over r: the factors depend on the stage and
        // on j, never on the block, so the inner loops below only ever read
        // from the table built in the constructor.
        const Twiddle* stage = twiddles + m2;

        for (size_t r = 0; r <= n - m; r = r + m) {
            for (size_t j = 1; j < m4; ++j) {
                const size_t k = m2 - j;
                const auto u = vec[r + m2 + j];
                const auto v = vec[r + m2 + k];
                const T cosVal = stage[j].cosine;
                const T sinVal = stage[j].sine;
                vec[r + m2 + j] = u * cosVal + v * sinVal;
                vec[r + m2 + k] = u * sinVal - v * cosVal;
            }
            for (size_t j = 0; j < m2; ++j) {
                const auto u = vec[r + j];
                const auto v = vec[r + j + m2];
                vec[r + j] = u + v;
                vec[r + j + m2] = u - v;
            }
        }
    }

    // записать обратно
    for (size_t idx = 0; idx < n; ++idx) {
        data[AxisIndex(idx, 0, 0, direction)] = vec[idx];
    }
}

template <typename T>
void HartleyTransform<T>::FDHT2D(T* image_ptr)
{
    PROFILE_FUNCTION();

    if (image_ptr == nullptr) {
        throw std::invalid_argument("FDHT2D: the pointer to image is null.");
    }

    Series1D(image_ptr, Direction::X);
    Series1D(image_ptr, Direction::Y);

    BracewellTransform2DCPU(image_ptr);
}

template <typename T>
void HartleyTransform<T>::FDHT3D(T* volume_ptr)
{
    PROFILE_FUNCTION();

    if (volume_ptr == nullptr) {
        throw std::invalid_argument("FDHT3D: the pointer to volume is null.");
    }

    // 1D transforms along X, Y, Z dimensions
    Series1D(volume_ptr, Direction::Y);
    Series1D(volume_ptr, Direction::X);
    Series1D(volume_ptr, Direction::Z);
    // Bracewell 3D
    BracewellTransform3DCPU(volume_ptr);
}

template <typename T>
void HartleyTransform<T>::RealFFT1D(T* vec, Direction direction)
{
    PROFILE_FUNCTION();

    if (vec == nullptr) {
        throw std::invalid_argument("RealFFT1D: the pointer to vector is null.");
    }

    // No lower-bound check: Length() returns size_t, so "< 0" was always false.
    if (!IsPowerOfTwo(Length(direction))) {
        throw std::invalid_argument("RealFFT1D: length must be a power of two, got "
                                    + std::to_string(Length(direction)) + ".");
    }

    // RealFFT
    std::vector<std::complex<T>> x(Length(direction));
    for (size_t i = 0; i < Length(direction); i++) {
        x[i] = std::complex<T>(vec[i], 0);
    }
    size_t k = Length(direction);
    size_t n;
    const T thetaT = static_cast<T>(std::acos(-1)) / Length(direction);
    std::complex<T> phiT = std::complex<T>(std::cos(thetaT), -std::sin(thetaT)), TTT;
    while (k > 1) {
        n = k;
        k >>= 1;
        phiT = phiT * phiT;
        TTT = 1.0L;
        for (size_t l = 0; l < k; l++) {
            for (size_t a = l; a < Length(direction); a += n) {
                size_t b = a + k;
                std::complex<T> t = x[a] - x[b];
                x[a] += x[b];
                x[b] = t * TTT;
            }
            TTT *= phiT;
        }
    }
    // Decimate, reusing the table the constructor already builds.
    //
    // This used to reverse the bits inline with the classic 32-bit sequence of
    // masked shifts, but on a size_t the final "b << 16" keeps the bits that
    // the 32-bit version discards, and the following shift does not remove
    // them. Above n = 65536 the result exceeded the array: at n = 262144 it
    // reached index 12885164031, which segfaulted. The precomputed table is
    // both correct and already paid for.
    for (size_t a = 1; a < Length(direction); ++a) {
        const size_t b = BitReversedIndex(direction, a);
        if (b > a) {
            std::swap(x[a], x[b]);
        }
    }

    // The Hartley transform is Re(X) - Im(X), not Re(X): cas(t) = cos(t) + sin(t)
    // while the Fourier kernel is cos(t) - i*sin(t).
    for (size_t i = 0; i < Length(direction); i++) {
        vec[i] = x[i].real() - x[i].imag();
    }
}

// Explicit instantiation is per translation unit: it only reaches members
// whose definition is visible here, so each backend file instantiates its
// own.
template void HartleyTransform<float>::BitReverse(std::vector<size_t>&);
template void HartleyTransform<double>::BitReverse(std::vector<size_t>&);
template void HartleyTransform<float>::BuildTwiddleTable(Direction);
template void HartleyTransform<double>::BuildTwiddleTable(Direction);
template void HartleyTransform<float>::Series1D(float*, Direction);
template void HartleyTransform<double>::Series1D(double*, Direction);
template void HartleyTransform<float>::BracewellTransform2DCPU(float*);
template void HartleyTransform<double>::BracewellTransform2DCPU(double*);
template void HartleyTransform<float>::BracewellTransform3DCPU(float*);
template void HartleyTransform<double>::BracewellTransform3DCPU(double*);
template void HartleyTransform<float>::FDHT1D(float*, Direction);
template void HartleyTransform<double>::FDHT1D(double*, Direction);
template void HartleyTransform<float>::FDHT2D(float*);
template void HartleyTransform<double>::FDHT2D(double*);
template void HartleyTransform<float>::FDHT3D(float*);
template void HartleyTransform<double>::FDHT3D(double*);
template void HartleyTransform<float>::RealFFT1D(float*, Direction);
template void HartleyTransform<double>::RealFFT1D(double*, Direction);

} // namespace RapiDHT
