/*
 * Project: RapiDHT
 * File: src/internal/device_array.h
 * Brief: An RAII owner for a one-dimensional buffer in CUDA device memory.
 *
 * Internal header: not installed and not part of the public API.
 */

#ifndef RAPIDHT_INTERNAL_DEVICE_ARRAY_H
#define RAPIDHT_INTERNAL_DEVICE_ARRAY_H

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace RapiDHT {
namespace internal {

/*
 * Reports a failed CUDA call by throwing. This used to call exit(code):
 * a library has no business terminating its host process, which denied the
 * caller any chance to recover, run its own cleanup, or even report what
 * happened.
 */
inline void CudaCheck(cudaError_t code, const char* expression, const char* file, int line)
{
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(code)
                                 + " (" + cudaGetErrorName(code) + ") while evaluating '" + expression + "' at "
                                 + file + ":" + std::to_string(line));
    }
}

} // namespace internal
} // namespace RapiDHT

/*
 * Prefixed, because a macro obeys no namespace: the old name CUDA_CHECK is
 * common enough that a consumer including this alongside another CUDA library
 * could well have collided with it.
 */
#define RAPIDHT_CUDA_CHECK(err) ::RapiDHT::internal::CudaCheck((err), #err, __FILE__, __LINE__)

namespace RapiDHT {
namespace internal {

/**
 * @brief Owns a device allocation and the stream its transfers run on.
 *
 * Move-only rather than copyable: device memory cannot be duplicated by an
 * implicit copy, and a copy that shared the pointer would double-free.
 */
template <class T>
class DeviceArray {
public:
    DeviceArray()
    {
        RAPIDHT_CUDA_CHECK(cudaStreamCreate(&_stream));
    }

    explicit DeviceArray(size_t size)
    {
        RAPIDHT_CUDA_CHECK(cudaStreamCreate(&_stream));
        Allocate(size);
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    ~DeviceArray()
    {
        // Destructors must not propagate exceptions: Free() throws, and a throw
        // during unwinding turns into std::terminate. Nothing useful can be
        // done about a failed release here anyway.
        if (_begin != nullptr) {
            cudaFreeAsync(_begin, _stream);
            cudaStreamSynchronize(_stream);
            _begin = _end = nullptr;
        }
        cudaStreamDestroy(_stream);
    }

    /// Releases the current allocation and takes a new one of `size` elements.
    void Resize(size_t size)
    {
        Free();
        Allocate(size);
    }

    /// Number of elements the array holds.
    size_t Size() const
    {
        return static_cast<size_t>(_end - _begin);
    }

    const T* Data() const
    {
        return _begin;
    }

    T* Data()
    {
        return _begin;
    }

    /// Host to device. Copies min(size, Size()) elements and blocks until done.
    void Upload(const T* source, size_t size)
    {
        const size_t count = std::min(size, Size());
        RAPIDHT_CUDA_CHECK(
            cudaMemcpyAsync(_begin, source, count * sizeof(T), cudaMemcpyHostToDevice, _stream));
        RAPIDHT_CUDA_CHECK(cudaStreamSynchronize(_stream));
    }

    /// Device to host, with the same truncating rule as Upload.
    void Download(T* destination, size_t size) const
    {
        const size_t count = std::min(size, Size());
        RAPIDHT_CUDA_CHECK(
            cudaMemcpyAsync(destination, _begin, count * sizeof(T), cudaMemcpyDeviceToHost, _stream));
        RAPIDHT_CUDA_CHECK(cudaStreamSynchronize(_stream));
    }

private:
    void Allocate(size_t size)
    {
        const cudaError_t result = cudaMallocAsync((void**)&_begin, size * sizeof(T), _stream);
        if (result != cudaSuccess) {
            _begin = _end = nullptr;
        }
        RAPIDHT_CUDA_CHECK(result);
        _end = _begin + size;
    }

    void Free()
    {
        if (_begin != nullptr) {
            RAPIDHT_CUDA_CHECK(cudaFreeAsync(_begin, _stream));
            _begin = _end = nullptr;
            RAPIDHT_CUDA_CHECK(cudaStreamSynchronize(_stream));
        }
    }

    T* _begin = nullptr;
    T* _end = nullptr;
    cudaStream_t _stream = nullptr;
};

} // namespace internal
} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_DEVICE_ARRAY_H
