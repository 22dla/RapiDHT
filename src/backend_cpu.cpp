/*
 * Project: RapiDHT
 * File: src/backend_cpu.cpp
 * Brief: The OpenMP backend: butterflies, twiddles and the Bracewell passes.
 */

#include <rapidht/transform.h>
#include <rapidht/utilities.h>

#include "internal/support.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
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
    const int log2N = static_cast<int>(std::log2(n));

    indices[0] = 0;
    for (size_t j = 1; j < n; ++j) {
        size_t reversed = 0;
        size_t temp = j;
        for (int i = 0; i < log2N; ++i) {
            if (temp & 1) {
                // size_t{1}, not 1: shifting an int is undefined past 31 bits.
                reversed |= size_t { 1 } << (log2N - 1 - i);
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
void HartleyTransform<T>::RequirePowerOfTwoExtents(const char* caller) const
{
    static const char* const kAxisNames[] = { "width", "height", "depth" };
    for (size_t axis = 0; axis < _dims.size(); ++axis) {
        // Zero means "this axis is absent" (1D or 2D), not a length.
        if (_dims[axis] != 0 && !IsPowerOfTwo(_dims[axis])) {
            throw std::invalid_argument(std::string(caller) + ": " + kAxisNames[axis]
                                        + " must be a power of two for Modes::CPU, got "
                                        + std::to_string(_dims[axis]) + ".");
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
    // Last line of defence for any future caller: an exception thrown inside
    // the parallel region below would terminate the process.
    RequirePowerOfTwoExtents("Series1D");

    // No RFFT branch on purpose: the constructor rejects Modes::RFFT for
    // anything but 1D, and Series1D is only ever reached from 2D and 3D.
    if (_mode != Modes::CPU) {
        return;
    }

    const size_t W = Width();
    const size_t H = Height();
    const size_t D = (Depth() == 0 ? 1 : Depth());

    // A line runs along `direction` with stride axisStride. Lines are taken in
    // blocks of `lanes` neighbours along a second axis ("lane" axis), chosen
    // as the fastest axis whenever the line itself is not along it, so that
    // gathering one row of the block reads contiguous memory. The remaining
    // axis ("other") just enumerates blocks.
    size_t axisStride = 0, laneStride = 0, laneLength = 0, otherStride = 0, otherLength = 0;
    switch (direction) {
        case Direction::Y: // along the fastest axis; neighbouring lines are rows
            axisStride = 1;
            laneStride = W;
            laneLength = H;
            otherStride = W * H;
            otherLength = D;
            break;
        case Direction::X:
            axisStride = W;
            laneStride = 1;
            laneLength = W;
            otherStride = W * H;
            otherLength = D;
            break;
        case Direction::Z:
            axisStride = W * H;
            laneStride = 1;
            laneLength = W;
            otherStride = W;
            otherLength = H;
            break;
        default:
            throw std::invalid_argument("Invalid direction");
    }

    // 32 lanes measured best against 8, 16 and 64; with n = 512 a float block is 64 KiB --
    // it stays in L1/L2 for the whole log2(n) stages. Extents are powers of
    // two, so `lanes` divides laneLength exactly.
    constexpr size_t kMaxLanes = 32;
    const size_t n = Length(direction);
    const size_t lanes = (laneLength < kMaxLanes ? laneLength : kMaxLanes);
    const size_t blocksPerOther = laneLength / lanes;

    // Signed for OpenMP 2.0 (MSVC).
    const int blockCount = static_cast<int>(blocksPerOther * otherLength);
#pragma omp parallel
    {
        // One buffer per thread for the whole call, not one per line.
        std::vector<T> buffer(n * lanes);
#pragma omp for
        for (int p = 0; p < blockCount; ++p) {
            const size_t other = static_cast<size_t>(p) / blocksPerOther;
            const size_t firstLane = (static_cast<size_t>(p) % blocksPerOther) * lanes;
            T* base = data + other * otherStride + firstLane * laneStride;

            for (size_t idx = 0; idx < n; ++idx) {
                const T* src = base + idx * axisStride;
                T* dst = buffer.data() + idx * lanes;
                for (size_t b = 0; b < lanes; ++b) {
                    dst[b] = src[b * laneStride];
                }
            }

            FDHTLanes(buffer.data(), lanes, direction);

            for (size_t idx = 0; idx < n; ++idx) {
                const T* src = buffer.data() + idx * lanes;
                T* dst = base + idx * axisStride;
                for (size_t b = 0; b < lanes; ++b) {
                    dst[b * laneStride] = src[b];
                }
            }
        }
    }
}

template <typename T>
void HartleyTransform<T>::FDHTLanes(T* buffer, size_t lanes, Direction direction) const
{
    const size_t n = Length(direction);

    for (size_t i = 1; i < n; ++i) {
        const size_t j = BitReversedIndex(direction, i);
        if (j > i) {
            std::swap_ranges(buffer + i * lanes, buffer + (i + 1) * lanes, buffer + j * lanes);
        }
    }

    // The same stages as FDHT1D, with every scalar operation widened into a
    // loop across the lanes. The pairs of rows touched in one step are always
    // distinct, so the restrict qualifiers hold and let the loops vectorise.
    const Twiddle* twiddles = _twiddles[static_cast<size_t>(direction)].data();
    for (size_t m = 2; m <= n; m <<= 1) {
        const size_t m2 = m / 2;
        const size_t m4 = m / 4;
        // Only stages with m >= 8 read twiddles; for n < 4 the table is empty
        // and offsetting its (possibly null) data pointer would be undefined.
        const Twiddle* stage = (m4 > 1 ? twiddles + m2 : nullptr);

        for (size_t r = 0; r + m <= n; r += m) {
            for (size_t j = 1; j < m4; ++j) {
                const T c = stage[j].cosine;
                const T s = stage[j].sine;
                T* __restrict u = buffer + (r + m2 + j) * lanes;
                T* __restrict v = buffer + (r + m - j) * lanes;
                for (size_t b = 0; b < lanes; ++b) {
                    const T uu = u[b];
                    const T vv = v[b];
                    u[b] = uu * c + vv * s;
                    v[b] = uu * s - vv * c;
                }
            }
            for (size_t j = 0; j < m2; ++j) {
                T* __restrict a = buffer + (r + j) * lanes;
                T* __restrict b2 = buffer + (r + j + m2) * lanes;
                for (size_t b = 0; b < lanes; ++b) {
                    const T x = a[b];
                    const T y = b2[b];
                    a[b] = x + y;
                    b2[b] = x - y;
                }
            }
        }
    }
}

/*
 * The Bracewell correction turns the separable spectrum S into the true one.
 * Each output reads S only at the point itself and at its mirror images
 * (index -> (N - index) mod N along some axes), so the points split into
 * disjoint orbits under the axis flips -- up to 4 points in 2D, 8 in 3D -- and
 * every orbit can be transformed in place: load all its values, then store
 * all its outputs. That replaces a full-size result buffer and a final copy.
 *
 * The loops run over one representative per orbit, index in [0, N/2] on every
 * axis, with the fastest axis innermost so that the point and its mirror are
 * both read as contiguous runs (one forwards, one backwards). Where an index is
 * its own mirror (0 or N/2) the orbit is smaller and some loads and stores
 * coincide; the formulas then give identical values for the coinciding stores,
 * so nothing special is needed. Orbits are disjoint, so the outer loop
 * parallelises without races.
 */
namespace {

/// Mirror of an index along an axis of length n: (n - i) mod n.
inline size_t Mirror(size_t i, size_t n) noexcept
{
    return i == 0 ? 0 : n - i;
}

} // namespace

template <typename T>
void HartleyTransform<T>::BracewellTransform2DCPU(T* image)
{
    PROFILE_FUNCTION();

    const size_t W = Width();
    const size_t H = Height();
    const T half = static_cast<T>(0.5);

    // Signed index for OpenMP 2.0 (MSVC). Representatives x in [0, H/2].
    const int rowPairs = static_cast<int>(H / 2 + 1);
#pragma omp parallel for
    for (int xi = 0; xi < rowPairs; ++xi) {
        const size_t x = static_cast<size_t>(xi);
        T* r0 = image + W * x; // row x
        T* r1 = image + W * Mirror(x, H); // row -x
        for (size_t y = 0; y <= W / 2; ++y) {
            const size_t ym = Mirror(y, W);
            // v<flip y><flip x>
            const T v00 = r0[y], v10 = r0[ym], v01 = r1[y], v11 = r1[ym];
            r0[y] = (v00 + v01 + v10 - v11) * half;
            r0[ym] = (v10 + v11 + v00 - v01) * half;
            r1[y] = (v01 + v00 + v11 - v10) * half;
            r1[ym] = (v11 + v10 + v01 - v00) * half;
        }
    }
}

template <typename T>
void HartleyTransform<T>::BracewellTransform3DCPU(T* volumePtr)
{
    PROFILE_FUNCTION();

    const size_t W = Width();
    const size_t H = Height();
    const size_t D = Depth();
    const T half = static_cast<T>(0.5);

    // One flat signed loop over the (z, x) representatives instead of
    // collapse(2): OpenMP 2.0 (MSVC) has no collapse, and z alone gives too
    // few iterations to balance on small depths.
    const size_t xReps = H / 2 + 1;
    const int pairCount = static_cast<int>((D / 2 + 1) * xReps);
#pragma omp parallel for
    for (int p = 0; p < pairCount; ++p) {
        const size_t z = static_cast<size_t>(p) / xReps;
        const size_t x = static_cast<size_t>(p) % xReps;
        const size_t zm = Mirror(z, D);
        const size_t xm = Mirror(x, H);

        // Rows along the fastest axis; suffix = <flip y><flip x><flip z>.
        T* r00 = volumePtr + W * (x + H * z);
        T* r10 = volumePtr + W * (xm + H * z);
        T* r01 = volumePtr + W * (x + H * zm);
        T* r11 = volumePtr + W * (xm + H * zm);

        for (size_t y = 0; y <= W / 2; ++y) {
            const size_t ym = Mirror(y, W);
            const T v000 = r00[y], v100 = r00[ym];
            const T v010 = r10[y], v110 = r10[ym];
            const T v001 = r01[y], v101 = r01[ym];
            const T v011 = r11[y], v111 = r11[ym];

            // out(s) = (v(s^100) + v(s^010) + v(s^001) - v(s^111)) / 2, with
            // s^100 flipping y only, and so on: the definition used before,
            // A + B + C - D over the single-axis and all-axes mirrors.
            r00[y] = (v100 + v010 + v001 - v111) * half;
            r00[ym] = (v000 + v110 + v101 - v011) * half;
            r10[y] = (v110 + v000 + v011 - v101) * half;
            r10[ym] = (v010 + v100 + v111 - v001) * half;
            r01[y] = (v101 + v011 + v000 - v110) * half;
            r01[ym] = (v001 + v111 + v100 - v010) * half;
            r11[y] = (v111 + v001 + v010 - v100) * half;
            r11[ym] = (v011 + v101 + v110 - v000) * half;
        }
    }
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

    // Gather buffer: the line along this axis is strided in memory, and the
    // 1D transform wants it contiguous.
    std::vector<T> vec(n);

    // Gather
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
    const auto log2N = static_cast<size_t>(std::log2(n));
    const Twiddle* twiddles = _twiddles[static_cast<size_t>(direction)].data();

    // Main cicle
    for (size_t s = 1; s <= log2N; ++s) {
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

    // Scatter back
    for (size_t idx = 0; idx < n; ++idx) {
        data[AxisIndex(idx, 0, 0, direction)] = vec[idx];
    }
}

template <typename T>
void HartleyTransform<T>::FDHT2D(T* image)
{
    PROFILE_FUNCTION();

    if (image == nullptr) {
        throw std::invalid_argument("FDHT2D: the pointer to image is null.");
    }
    RequirePowerOfTwoExtents("ForwardTransform");

    Series1D(image, Direction::X);
    Series1D(image, Direction::Y);

    BracewellTransform2DCPU(image);
}

template <typename T>
void HartleyTransform<T>::FDHT3D(T* volume)
{
    PROFILE_FUNCTION();

    if (volume == nullptr) {
        throw std::invalid_argument("FDHT3D: the pointer to volume is null.");
    }
    RequirePowerOfTwoExtents("ForwardTransform");

    // 1D transforms along X, Y, Z dimensions
    Series1D(volume, Direction::Y);
    Series1D(volume, Direction::X);
    Series1D(volume, Direction::Z);
    // Bracewell 3D
    BracewellTransform3DCPU(volume);
}

template <typename T>
void HartleyTransform<T>::ForwardSeparable(T* data)
{
    PROFILE_FUNCTION();

    if (data == nullptr) {
        throw std::invalid_argument("ForwardSeparable: the pointer to data is null.");
    }

    const bool is1D = (Height() == 0 && Depth() == 0);
    const bool is2D = (Height() > 0 && Depth() == 0);

    if (_mode == Modes::GPUFactored) {
        FactoredOnHost(data, true);
        return;
    }
    if (_mode == Modes::GPU) {
        // The device paths take a flag rather than having separate copies:
        // the separable transform is the true one minus its last pass.
        if (is1D) {
            DHT1DCuda(data);
        } else if (is2D) {
            DHT2DCuda(data, true);
        } else {
            DHT3DCuda(data, true);
        }
        return;
    }

    if (is1D) {
        // In 1D the separable and the true transform are the same function.
        if (_mode == Modes::RFFT) {
            RealFFT1D(data);
        } else {
            FDHT1D(data);
        }
        return;
    }

    // The constructor rejects Modes::RFFT beyond 1D, so this is Modes::CPU.
    RequirePowerOfTwoExtents("ForwardSeparable");
    Series1D(data, Direction::Y);
    Series1D(data, Direction::X);
    if (!is2D) {
        Series1D(data, Direction::Z);
    }
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
template void HartleyTransform<float>::ForwardSeparable(float*);
template void HartleyTransform<double>::ForwardSeparable(double*);
template void HartleyTransform<float>::RealFFT1D(float*, Direction);
template void HartleyTransform<double>::RealFFT1D(double*, Direction);

} // namespace RapiDHT
