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

#include <cublas_v2.h>
#endif

#include <array>
#include <cstddef>

namespace RapiDHT {

#ifdef RAPIDHT_WITH_CUDA
namespace internal {

/// Reports a failed cuBLAS call by throwing, as CudaCheck does for the runtime.
inline void CublasCheck(cublasStatus_t status, const char* expression, const char* file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error ") + std::to_string(static_cast<int>(status))
                                 + " while evaluating '" + expression + "' at " + file + ":"
                                 + std::to_string(line));
    }
}

} // namespace internal
} // namespace RapiDHT

#define RAPIDHT_CUBLAS_CHECK(status) ::RapiDHT::internal::CublasCheck((status), #status, __FILE__, __LINE__)

namespace RapiDHT {
namespace internal {

/// Owns a cuBLAS handle. Move-only for the same reason as DeviceArray.
class CublasHandle {
public:
    CublasHandle()
    {
        RAPIDHT_CUBLAS_CHECK(cublasCreate(&_handle));
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    ~CublasHandle()
    {
        cublasDestroy(_handle);
    }

    cublasHandle_t Get() const
    {
        return _handle;
    }

private:
    cublasHandle_t _handle = nullptr;
};

} // namespace internal
#endif

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

    // Created once with the object: the 3D path used to create and destroy a
    // handle on every call.
    internal::CublasHandle cublas;
#endif
};

} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_IMPL_H
