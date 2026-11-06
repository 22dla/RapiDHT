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
#include "internal/factored.h"

#include <cublas_v2.h>
#endif

#include <array>
#include <cstddef>
#include <memory>

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
     * Working buffers, kept with the object rather than allocated on every
     * call. They used to be locals inside each DHT*Cuda method, so a 512^3
     * transform allocated and released two 512 MiB regions, and created and
     * destroyed two CUDA streams, every single time it ran.
     *
     * scratchB is the multiplies' output and every path needs it, so it is
     * allocated with the object. staging only receives host data on the
     * host-pointer path; a DeviceVolume plays its part on the resident path.
     * It is therefore allocated on first use (B-003): a transform used only
     * on resident volumes costs one volume of device memory, not two, which
     * at 512^3 in float is 512 MiB and the difference between fitting a
     * volume on an 8 GiB card or not.
     */
    internal::DeviceArray<T> staging;
    internal::DeviceArray<T> scratchB;

    /// The host-path staging buffer, allocated the first time it is asked for.
    /// DeviceArray::Upload truncates to its size, so an unallocated buffer
    /// would silently transfer nothing -- hence a getter rather than the field.
    internal::DeviceArray<T>& Staging(size_t count)
    {
        if (staging.Size() < count) {
            staging.Resize(count);
        }
        return staging;
    }

    // Created once with the object: the 3D path used to create and destroy a
    // handle on every call. Modes::GPU only -- the handle alone holds about
    // 41 MiB of device memory, which the factored mode does not need.
    std::unique_ptr<internal::CublasHandle> cublas;

    // Modes::GPUFactored (R-4): the tables of each present axis, back to back
    // as c1 | c2 | cHalf | twC | twD, and their split.
    std::array<internal::DeviceArray<T>, static_cast<size_t>(Direction::Count)> factoredTables;
    std::array<internal::FactoredSplit, static_cast<size_t>(Direction::Count)> factoredSplit;
#endif
};

} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_IMPL_H
