/*
 * Project: RapiDHT
 * File: include/rapidht/device_volume.h
 * Brief: A volume that lives in device memory across several transforms.
 */

#ifndef RAPIDHT_DEVICE_VOLUME_H
#define RAPIDHT_DEVICE_VOLUME_H

#include <cstddef>
#include <memory>

namespace RapiDHT {

template <typename T>
class HartleyTransform;

/**
 * @brief A volume held in device memory, so that repeated transforms do not
 *        copy it across the bus every time.
 *
 * The host-pointer overloads of ForwardTransform and InverseTransform upload
 * the data, transform it and download it again on every call. For a 512^3
 * volume that round trip costs roughly three times as much as the transform
 * itself, which makes it the dominant cost of any pipeline that applies more
 * than one operation. Uploading once and transforming in place removes it.
 *
 * The type is deliberately opaque: it names no CUDA type, so consuming this
 * header still requires no CUDA toolkit. It owns its allocation and is
 * move-only, since device memory cannot be duplicated implicitly.
 *
 * Constructing one in a build without CUDA throws std::runtime_error.
 */
template <typename T>
class DeviceVolume {
public:
    /**
     * @brief Allocates room for `count` elements on the device.
     * @param count Number of elements, which must match the transform's extent.
     */
    explicit DeviceVolume(size_t count);

    ~DeviceVolume();
    DeviceVolume(const DeviceVolume&) = delete;
    DeviceVolume& operator=(const DeviceVolume&) = delete;
    DeviceVolume(DeviceVolume&&) noexcept;
    DeviceVolume& operator=(DeviceVolume&&) noexcept;

    /**
     * @brief Copies `count` elements from host memory onto the device.
     * @param host Source buffer, at least Size() elements long.
     */
    void Upload(const T* host);

    /**
     * @brief Copies the volume back from the device into host memory.
     * @param host Destination buffer, at least Size() elements long.
     */
    void Download(T* host) const;

    /// Number of elements the volume holds.
    size_t Size() const noexcept;

private:
    template <typename>
    friend class HartleyTransform;

    /// Raw device pointer, for the transform to work on. Typed as void* so the
    /// header stays free of CUDA, and only ever handed to the implementation.
    void* DeviceData() const noexcept;

    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace RapiDHT

#endif // RAPIDHT_DEVICE_VOLUME_H
