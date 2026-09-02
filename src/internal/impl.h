/*
 * Project: RapiDHT
 * File: src/internal/impl.h
 * Brief: Definitions of the opaque Impl types the public headers only declare.
 *
 * Both HartleyTransform and DeviceVolume hide their device-side state behind a
 * pointer so that no installed header names a CUDA type. The definitions have
 * to be shared between the translation units that touch that state --
 * transform.cpp, backend_cuda.cpp and device_volume.cpp -- which is what this
 * header is for. It is internal and not installed.
 */

#ifndef RAPIDHT_INTERNAL_IMPL_H
#define RAPIDHT_INTERNAL_IMPL_H

#include <rapidht/device_volume.h>
#include <rapidht/transform.h>

#ifdef RAPIDHT_WITH_CUDA
#include "internal/device_array.h"
#endif

#include <array>
#include <cstddef>

namespace RapiDHT {

template <typename T>
struct DeviceVolume<T>::Impl {
#ifdef RAPIDHT_WITH_CUDA
    internal::DeviceArray<T> storage;
    size_t count = 0;
#endif
};

template <typename T>
struct HartleyTransform<T>::Impl {
#ifdef RAPIDHT_WITH_CUDA
    std::array<internal::DeviceArray<T>, static_cast<size_t>(Direction::Count)> transformMatrices;

    /*
     * Working buffers, allocated once with the object rather than on every
     * call. They used to be locals inside each DHT*Cuda method, so a 512^3
     * transform allocated and released two 512 MiB regions, and created and
     * destroyed two CUDA streams, every single time it ran.
     *
     * Holding them costs twice the volume in device memory for the lifetime of
     * the object, which is the usual bargain for a transform plan.
     */
    internal::DeviceArray<T> scratchA;
    internal::DeviceArray<T> scratchB;
#endif
};

} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_IMPL_H
