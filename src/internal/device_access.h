/*
 * Project: RapiDHT
 * File: src/internal/device_access.h
 * Brief: R-5 -- the internal entry points that are not public API yet.
 *
 * The transform of an integer volume (docs/R5_PLAN.md, stage 2) runs inside
 * HartleyTransform, but whether and how it becomes public is decided only at
 * step 6.1, from the measurements of stages 2-5. Until then benchmarks and
 * tests reach it here: HartleyTransform befriends this struct, and the
 * struct lives in src/internal, which no installed header includes.
 */

#ifndef RAPIDHT_INTERNAL_DEVICE_ACCESS_H
#define RAPIDHT_INTERNAL_DEVICE_ACCESS_H

#include <rapidht/device_volume.h>
#include <rapidht/transform.h>

namespace RapiDHT {
namespace internal {

struct DeviceAccess {
    /// The factored transform of a device-resident integer volume into `out`
    /// (HartleyTransform::FactoredFromIntegers). TIn: std::uint8_t or
    /// std::uint16_t; `offset` an integer, subtracted on reading and returned
    /// to the zero coefficient.
    template <typename T, typename TIn>
    static void ForwardFromIntegers(
        HartleyTransform<T>& transform, const TIn* deviceIn, DeviceVolume<T>& out, double offset, bool separable = false)
    {
        transform.FactoredFromIntegers(deviceIn, out, static_cast<T>(offset), separable);
    }
};

} // namespace internal
} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_DEVICE_ACCESS_H
