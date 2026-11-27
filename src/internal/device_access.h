/*
 * Project: RapiDHT
 * File: src/internal/device_access.h
 * Brief: R-5 -- the internal entry points on raw device pointers.
 *
 * The transform of an integer volume (docs/R5_PLAN.md, stage 2) became public
 * at step 6.1 as overloads on DeviceVolume<std::uint8_t / std::uint16_t> and
 * on host pointers. These entry points take the integers at a raw device
 * pointer instead, and the inverse can be separable; the benchmarks and the
 * tests of stages 2-5 use them. HartleyTransform befriends this struct, and
 * the struct lives in src/internal, which no installed header includes.
 */

#ifndef RAPIDHT_INTERNAL_DEVICE_ACCESS_H
#define RAPIDHT_INTERNAL_DEVICE_ACCESS_H

#include <rapidht/device_volume.h>
#include <rapidht/transform.h>

#include "internal/integer_offset.h"

namespace RapiDHT {
namespace internal {

struct DeviceAccess {
    /// The factored transform of a device-resident integer volume into `out`
    /// (HartleyTransform::FactoredFromIntegers). TIn: std::uint8_t or
    /// std::uint16_t; `offset` the policy that gives the integer subtracted on
    /// reading and returned to the zero coefficient (integer_offset.h).
    template <typename T, typename TIn>
    static void ForwardFromIntegers(HartleyTransform<T>& transform, const TIn* deviceIn, DeviceVolume<T>& out,
        const IntegerOffset& offset, bool separable = false)
    {
        transform.FactoredFromIntegers(deviceIn, out, offset, separable);
    }

    /// R-5 step 4.2: the inverse of `spectrum` into integers at `deviceOut`
    /// (HartleyTransform::FactoredToIntegers); `spectrum` is used as the work
    /// buffer. The mean policies take the rounded mean off H(0).
    template <typename T, typename TOut>
    static void InverseToIntegers(HartleyTransform<T>& transform, DeviceVolume<T>& spectrum, TOut* deviceOut,
        const IntegerOffset& offset, size_t* saturated = nullptr, bool separable = false)
    {
        transform.FactoredToIntegers(spectrum, deviceOut, offset, saturated, separable);
    }

    /// The same with a known integer `offset` (IntegerOffset::Of).
    template <typename T, typename TIn>
    static void ForwardFromIntegers(
        HartleyTransform<T>& transform, const TIn* deviceIn, DeviceVolume<T>& out, double offset, bool separable = false)
    {
        transform.FactoredFromIntegers(deviceIn, out, IntegerOffset::Of(offset), separable);
    }
};

} // namespace internal
} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_DEVICE_ACCESS_H
