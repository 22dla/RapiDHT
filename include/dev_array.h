#ifndef _DEV_ARRAY_H_
#define _DEV_ARRAY_H_

#include <stdexcept>
#include <algorithm>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

template <class T>
class dev_array {
	// public functions
public:
	explicit dev_array()
		: start_(0),
		end_(0) {}

	// constructor
	explicit dev_array(size_t size) {
		allocate(size);
	}
	// destructor
	~dev_array() {
		free();
	}

	// resize the vector
	void resize(size_t size) {
		free();
		allocate(size);
	}

	// get the size of the array
	size_t getSize() const {
		return end_ - start_;
	}

	// get data
	const T* getData() const {
		return start_;
	}

	T* getData() {
		return start_;
	}

	// set
	void set(const T* src, size_t size) {
#ifdef USE_CUDA
		size_t min = std::min(size, getSize());
		cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			throw std::runtime_error("failed to copy to device memory");
		}
#else
		(void)src; (void)size;
		throw std::runtime_error("CUDA disabled: cannot copy to device memory");
#endif
	}
	// get
	void get(T* dest, size_t size) {
#ifdef USE_CUDA
		size_t min = std::min(size, getSize());
		cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			throw std::runtime_error("failed to copy to host memory");
		}
#else
		(void)dest; (void)size;
		throw std::runtime_error("CUDA disabled: cannot copy from device memory");
#endif
	}


private:
	// allocate memory on the device
	void allocate(size_t size) {
#ifdef USE_CUDA
		size_t free_bytes, total_bytes;
		cudaError_t result = cudaMemGetInfo(&free_bytes, &total_bytes);
		if (result != cudaSuccess) {
			start_ = end_ = 0;
			throw std::runtime_error("failed to get GPU memory info");
		}
		if (size * sizeof(T) > free_bytes) {
			start_ = end_ = 0;
			throw std::runtime_error("not enough free GPU memory");
		}
		result = cudaMalloc((void**)&start_, size * sizeof(T));
		if (result != cudaSuccess) {
			start_ = end_ = 0;
			throw std::runtime_error("failed to allocate device memory");
		}
		end_ = start_ + size;
#else
		(void)size;
		throw std::runtime_error("CUDA disabled: cannot allocate device memory");
#endif
	}


	// free memory on the device
	void free() {
#ifdef USE_CUDA
		if (start_ != 0) {
			cudaFree(start_);
			start_ = end_ = 0;
		}
#else
		start_ = end_ = 0;
#endif
	}

	T* start_;
	T* end_;
};

#endif