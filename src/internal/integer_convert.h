/*
 * Project: RapiDHT
 * File: src/internal/integer_convert.h
 * Brief: R-5 step 4.1 -- a value of the inverse transform to an integer voxel.
 *
 * The inverse of an integer volume ends with one conversion a voxel:
 *
 *   v = fma(x, scale, offset)   x the unscaled inverse, scale 1/N, offset the
 *                               integer taken out of H(0) before the passes
 *   r = rint(v)                 to nearest, ties to even
 *   r clamped to [0, max of TOut], each clamped voxel counted
 *
 * One function for the host reference and the device kernel, so the two agree
 * bit for bit by construction: the multiply-add is an explicit fma because
 * nvcc contracts a * b + c into one on its own and a host compiler need not.
 * NaN counts as saturated and gives 0.
 */

#ifndef RAPIDHT_INTERNAL_INTEGER_CONVERT_H
#define RAPIDHT_INTERNAL_INTEGER_CONVERT_H

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define RAPIDHT_HOST_DEVICE __host__ __device__
#else
#define RAPIDHT_HOST_DEVICE
#endif

namespace RapiDHT {
namespace internal {

/// The largest value of an unsigned integer type of one or two bytes.
template <typename TOut>
RAPIDHT_HOST_DEVICE constexpr double IntegerMaxOf()
{
    return sizeof(TOut) == 1 ? 255.0 : 65535.0;
}

/// fma(x, scale, offset), rounded to nearest and saturated to TOut; adds 1 to
/// `saturated` when the value had to be clamped (or was NaN).
template <typename TOut, typename T, typename Count>
RAPIDHT_HOST_DEVICE inline TOut ToIntegerSaturated(T x, T scale, T offset, Count& saturated)
{
    using std::fma;
    using std::rint;
    const T r = rint(fma(x, scale, offset));
    const T top = static_cast<T>(IntegerMaxOf<TOut>());
    if (!(r >= T(0))) { // also NaN
        ++saturated;
        return TOut(0);
    }
    if (r > top) {
        ++saturated;
        return static_cast<TOut>(IntegerMaxOf<TOut>());
    }
    return static_cast<TOut>(r);
}

} // namespace internal
} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_INTEGER_CONVERT_H
