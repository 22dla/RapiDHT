/*
 * Project: RapiDHT
 * File: src/device_volume.cpp
 * Brief: DeviceVolume, and the runtime device-presence check.
 */

#include <rapidht/config_flags.h>
#include <rapidht/device_volume.h>

#include "internal/impl.h"

#include <stdexcept>

namespace RapiDHT {

bool IsGpuAvailable() noexcept
{
#ifdef RAPIDHT_WITH_CUDA
    int count = 0;
    // Reports the failure through its return value rather than a sticky error,
    // so an absent or too-old driver just yields false.
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

/* ------------------------------ DeviceVolume ------------------------------ */

template <typename T>
DeviceVolume<T>::DeviceVolume(size_t count)
{
#ifdef RAPIDHT_WITH_CUDA
    if (count == 0) {
        throw std::invalid_argument("DeviceVolume: size must be positive.");
    }
    _impl = std::make_unique<Impl>();
    _impl->storage.Resize(count);
    _impl->count = count;
#else
    (void)count;
    throw std::runtime_error(
        "RapiDHT was built without CUDA support (RAPIDHT_WITH_CUDA=OFF): "
        "DeviceVolume is unavailable. Reconfigure with -DRAPIDHT_WITH_CUDA=ON.");
#endif
}

template <typename T>
DeviceVolume<T>::~DeviceVolume() = default;

template <typename T>
DeviceVolume<T>::DeviceVolume(DeviceVolume&&) noexcept = default;

template <typename T>
DeviceVolume<T>& DeviceVolume<T>::operator=(DeviceVolume&&) noexcept = default;

template <typename T>
void DeviceVolume<T>::Upload(const T* host)
{
#ifdef RAPIDHT_WITH_CUDA
    if (host == nullptr) {
        throw std::invalid_argument("DeviceVolume::Upload: host pointer is null.");
    }
    _impl->storage.Upload(host, _impl->count);
#else
    (void)host;
#endif
}

template <typename T>
void DeviceVolume<T>::Download(T* host) const
{
#ifdef RAPIDHT_WITH_CUDA
    if (host == nullptr) {
        throw std::invalid_argument("DeviceVolume::Download: host pointer is null.");
    }
    _impl->storage.Download(host, _impl->count);
#else
    (void)host;
#endif
}

template <typename T>
size_t DeviceVolume<T>::Size() const noexcept
{
#ifdef RAPIDHT_WITH_CUDA
    return _impl ? _impl->count : 0;
#else
    return 0;
#endif
}

template <typename T>
void* DeviceVolume<T>::DeviceData() const noexcept
{
#ifdef RAPIDHT_WITH_CUDA
    return _impl ? static_cast<void*>(const_cast<T*>(_impl->storage.Data())) : nullptr;
#else
    return nullptr;
#endif
}

template class DeviceVolume<float>;
template class DeviceVolume<double>;

} // namespace RapiDHT
