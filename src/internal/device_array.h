/*
 * Project: RapiDHT
 * File: src/internal/device_array.h
 * Brief: RAII-обёртка над памятью GPU (CUDA) для одномерных массивов.
 * Author: Волков Евгений Александрович, volkov22dla@yandex.ru
 */

#ifndef RAPIDHT_INTERNAL_DEVICE_ARRAY_H
#define RAPIDHT_INTERNAL_DEVICE_ARRAY_H

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/*
 * Reports a failed CUDA call by throwing. This used to call exit(code):
 * a library has no business terminating its host process, which denied the
 * caller any chance to recover, run its own cleanup, or even report what
 * happened.
 */
inline void CudaCheckImpl(cudaError_t code, const char* expression, const char* file, int line)
{
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(code)
                                 + " (" + cudaGetErrorName(code) + ") while evaluating '" + expression + "' at "
                                 + file + ":" + std::to_string(line));
    }
}

#define CUDA_CHECK(err) ::CudaCheckImpl((err), #err, __FILE__, __LINE__)

template <class T>
class dev_array {
public:
    explicit dev_array(): start_(nullptr), end_(nullptr), stream_(0)
    {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    explicit dev_array(size_t size): start_(nullptr), end_(nullptr), stream_(0)
    {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        allocate(size);
    }

    // Owns device memory, so copying would double-free.
    dev_array(const dev_array&) = delete;
    dev_array& operator=(const dev_array&) = delete;

    ~dev_array()
    {
        // Destructors must not propagate exceptions: free() used to throw,
        // which turns any failure during unwinding into std::terminate.
        // Nothing useful can be done about a failed release here anyway.
        if (start_ != nullptr) {
            cudaFreeAsync(start_, stream_);
            cudaStreamSynchronize(stream_);
            start_ = end_ = nullptr;
        }
        cudaStreamDestroy(stream_);
    }

    // resize с использованием cudaMallocAsync
    void resize(size_t size)
    {
        free();
        allocate(size);
    }

    size_t getSize() const
    {
        return end_ - start_;
    }

    const T* getData() const
    {
        return start_;
    }

    T* getData()
    {
        return start_;
    }

    void set(const T* src, size_t size)
    {
        const size_t count = std::min(size, getSize());
        CUDA_CHECK(cudaMemcpyAsync(start_, src, count * sizeof(T), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void get(T* dest, size_t size) const
    {
        const size_t count = std::min(size, getSize());
        CUDA_CHECK(cudaMemcpyAsync(dest, start_, count * sizeof(T), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    void allocate(size_t size)
    {
        const cudaError_t result = cudaMallocAsync((void**)&start_, size * sizeof(T), stream_);
        if (result != cudaSuccess) {
            start_ = end_ = nullptr;
        }
        CUDA_CHECK(result);
        end_ = start_ + size;
    }

    void free()
    {
        if (start_ != nullptr) {
            CUDA_CHECK(cudaFreeAsync(start_, stream_));
            start_ = end_ = nullptr;
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }

    T* start_;
    T* end_;
    cudaStream_t stream_;
};

#endif // RAPIDHT_INTERNAL_DEVICE_ARRAY_H
