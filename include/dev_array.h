/*
 * Project: RapiDHT
 * File: dev_array.h
 * Brief: RAII-обёртка над памятью GPU (CUDA) для одномерных массивов.
 * Author: Волков Евгений Александрович, volkov22dla@yandex.ru
 */

#ifndef DEV_ARRAY_H
#define DEV_ARRAY_H

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(err)                                                                                                \
	{                                                                                                                  \
		gpuAssert((err), __FILE__, __LINE__);                                                                          \
	}
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << std::endl;
		if (abort) {
			exit(code);
		}
	}
}

inline float ElapsedMsGPU(cudaEvent_t start, cudaEvent_t stop) {
	float ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	return ms;
}

template <class T>
class dev_array {
  public:
	explicit dev_array() : start_(nullptr), end_(nullptr), stream_(0) {
		cudaStreamCreate(&stream_);
	}

	explicit dev_array(size_t size) : start_(nullptr), end_(nullptr), stream_(0) {
		cudaStreamCreate(&stream_);
		allocate(size);
	}

	~dev_array() {
		free();
		cudaStreamDestroy(stream_);
	}

	// resize с использованием cudaMallocAsync
	void resize(size_t size) {
		free();
		allocate(size);
	}

	size_t getSize() const {
		return end_ - start_;
	}

	const T* getData() const {
		return start_;
	}

	T* getData() {
		return start_;
	}

	void set(const T* src, size_t size) {
		size_t min = std::min(size, getSize());
		cudaError_t result = cudaMemcpyAsync(start_, src, min * sizeof(T), cudaMemcpyHostToDevice, stream_);
		if (result != cudaSuccess) {
			throw std::runtime_error("failed to copy to device memory");
		}
		// дождаться завершения копирования
		cudaStreamSynchronize(stream_);
	}

	void get(T* dest, size_t size) const {
		size_t min = std::min(size, getSize());

		cudaError_t result = cudaMemcpyAsync(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost, stream_);
		if (result != cudaSuccess) {
			std::cerr << "cudaMemcpyAsync failed: " << cudaGetErrorName(result) << " - " << cudaGetErrorString(result)
					  << std::endl;
			std::abort();
		}

		result = cudaStreamSynchronize(stream_);
		if (result != cudaSuccess) {
			std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorName(result) << " - "
					  << cudaGetErrorString(result) << std::endl;
			std::abort();
		}
	}

  private:
	void allocate(size_t size) {
		cudaError_t result = cudaMallocAsync((void**)&start_, size * sizeof(T), stream_);
		if (result != cudaSuccess) {
			start_ = end_ = nullptr;
			std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorName(result) << " - "
					  << cudaGetErrorString(result) << std::endl;
			throw std::runtime_error("failed to allocate device memory (cudaMallocAsync)");
		}
		end_ = start_ + size;
	}

	void free() {
		if (start_ != nullptr) {
			cudaError_t result = cudaFreeAsync(start_, stream_);
			if (result != cudaSuccess) {
				throw std::runtime_error("failed to free device memory (cudaFreeAsync)");
			}
			start_ = end_ = nullptr;
			cudaStreamSynchronize(stream_);
		}
	}

	T* start_;
	T* end_;
	cudaStream_t stream_;
};

#endif // DEV_ARRAY_H
