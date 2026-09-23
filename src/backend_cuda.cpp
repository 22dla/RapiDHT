/*
 * Project: RapiDHT
 * File: src/backend_cuda.cpp
 * Brief: The CUDA backend: dense matrices through cuBLAS, plus the kernels.
 */

#include <rapidht/transform.h>
#include <rapidht/utilities.h>

#include "internal/device_array.h"
#include "internal/impl.h"
#include "internal/kernels.h"
#include "internal/support.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace RapiDHT {

template <typename T>
void HartleyTransform<T>::DHT1DCuda(T* hostData)
{
    PROFILE_FUNCTION();

    // Buffers live in Impl and are allocated once with the object, not on
    // every call.
    _impl->scratchA.Upload(hostData, Width());
    DHT1DOnDevice(_impl->scratchA.Data(), _impl->scratchB.Data());
    _impl->scratchA.Download(hostData, Width());
}

template <typename T>
void HartleyTransform<T>::DHT1DOnDevice(T* deviceInOut, T* deviceScratch)
{
    PROFILE_FUNCTION();

    VectorMatrixMultiplication(_impl->transformMatrices[static_cast<size_t>(Direction::Y)].Data(),
        deviceInOut, deviceScratch, Width());

    // The multiply cannot write in place, so the answer lands in the scratch
    // buffer and has to come back to satisfy this method's contract.
    RAPIDHT_CUDA_CHECK(cudaMemcpy(deviceInOut, deviceScratch, Width() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
void HartleyTransform<T>::DHT2DCuda(T* hostData)
{
    PROFILE_FUNCTION();

    const size_t sliceSize = Width() * Height();

    _impl->scratchA.Upload(hostData, sliceSize);
    DHT2DOnDevice(_impl->scratchA.Data(), _impl->scratchB.Data());
    _impl->scratchA.Download(hostData, sliceSize);
}

template <typename T>
void HartleyTransform<T>::DHT2DOnDevice(T* deviceInOut, T* deviceScratch)
{
    PROFILE_FUNCTION();

    const size_t sliceSize = Width() * Height();

    // The slice is Height() rows of Width() elements, so the first pass runs
    // along the fast axis and must use the Width()-sized matrix -- that is
    // Direction::Y, per InitializeHartleyMatrix in the constructor.
    //
    // These two were the other way round, which made the inner dimension of
    // the multiply disagree with the size of the matrix: on any non-square
    // extent the kernel read past the end of the transform matrix, which is
    // why 8x4 and 16x8 produced stable garbage while 4x4 happened to work.
    MatrixMultiplication(deviceInOut, _impl->transformMatrices[static_cast<size_t>(Direction::Y)].Data(),
        deviceScratch, Height(), Width(), Width());
    MatrixTranspose(deviceScratch, deviceInOut, Height(), Width());

    MatrixMultiplication(deviceInOut, _impl->transformMatrices[static_cast<size_t>(Direction::X)].Data(),
        deviceScratch, Width(), Height(), Height());
    MatrixTranspose(deviceScratch, deviceInOut, Width(), Height());

    // Without this the GPU produced the separable transform while the CPU
    // produced the true multidimensional one: the two backends computed
    // different functions for every extent except 1D.
    BracewellTransform2D(deviceInOut, deviceScratch, static_cast<int>(Width()), static_cast<int>(Height()));

    // The correction reads mirrored points and so cannot write in place; bring
    // the answer back to satisfy this method's contract.
    RAPIDHT_CUDA_CHECK(cudaMemcpy(deviceInOut, deviceScratch, sliceSize * sizeof(T), cudaMemcpyDeviceToDevice));
}

namespace {
template <typename T>
struct CublasGemmStridedBatched;

template <>
struct CublasGemmStridedBatched<float> {
    static cublasStatus_t call(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
        int n, int k, const float* alpha, const float* A, int lda, long long int strideA,
        const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc,
        long long int strideC, int batchCount)
    {
        return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB,
            beta, C, ldc, strideC, batchCount);
    }
};

template <>
struct CublasGemmStridedBatched<double> {
    static cublasStatus_t call(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
        int n, int k, const double* alpha, const double* A, int lda, long long int strideA,
        const double* B, int ldb, long long int strideB, const double* beta, double* C,
        int ldc, long long int strideC, int batchCount)
    {
        return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB,
            beta, C, ldc, strideC, batchCount);
    }
};

} // namespace

template <typename T>
void HartleyTransform<T>::DHT3DCuda(T* hostData)
{
    PROFILE_FUNCTION();

    const size_t totalSize = Width() * Height() * Depth();

    _impl->scratchA.Upload(hostData, totalSize);
    DHT3DOnDevice(_impl->scratchA.Data(), _impl->scratchB.Data());
    _impl->scratchA.Download(hostData, totalSize);
}

template <typename T>
void HartleyTransform<T>::DHT3DOnDevice(T* deviceInOut, T* deviceScratch)
{
    PROFILE_FUNCTION();

    const int W = static_cast<int>(Width());
    const int H = static_cast<int>(Height());
    const int D = static_cast<int>(Depth());
    const long long plane = static_cast<long long>(W) * H;

    cublasHandle_t handle = _impl->cublas.Get();
    const T alpha = 1.0;
    const T beta = 0.0;

    /*
     * No transposes. The volume is stored x fastest, idx = x + W*(y + H*z),
     * which cuBLAS already reads as column-major, and each cas matrix is
     * symmetric, so every axis is one multiply on the data as it lies:
     *
     *   X:  C_W * [W x HD]              one GEMM
     *   Y:  [W x H] * C_H, per z-slice  D batches, stride W*H
     *   Z:  [WH x D] * C_D              one GEMM
     *
     * This used to transpose the volume around each multiply -- four
     * transposes and two Y/Z swaps, each a full pass over memory. At 512^3
     * they took about 10 of the 58 ms, which went on data movement alone.
     *
     * Direction::Y holds the Width()-sized matrix and Direction::X the
     * Height()-sized one, per the constructor.
     */

    // Along X: deviceInOut -> deviceScratch.
    RAPIDHT_CUBLAS_CHECK(CublasGemmStridedBatched<T>::call(handle, CUBLAS_OP_N, CUBLAS_OP_N, W, H * D, W,
        &alpha, _impl->transformMatrices[(size_t)Direction::Y].Data(), W, 0, deviceInOut, W, 0, &beta,
        deviceScratch, W, 0, 1));

    // Along Y: deviceScratch -> deviceInOut, one W x H slice per batch.
    RAPIDHT_CUBLAS_CHECK(CublasGemmStridedBatched<T>::call(handle, CUBLAS_OP_N, CUBLAS_OP_N, W, H, H, &alpha,
        deviceScratch, W, plane, _impl->transformMatrices[(size_t)Direction::X].Data(), H, 0, &beta,
        deviceInOut, W, plane, D));

    // Along Z: deviceInOut -> deviceScratch.
    RAPIDHT_CUBLAS_CHECK(CublasGemmStridedBatched<T>::call(handle, CUBLAS_OP_N, CUBLAS_OP_N, W * H, D, D,
        &alpha, deviceInOut, W * H, 0, _impl->transformMatrices[(size_t)Direction::Z].Data(), D, 0, &beta,
        deviceScratch, W * H, 0, 1));

    // The correction reads mirrored points and cannot share its input and
    // output buffer, which conveniently lands the result in deviceInOut, as
    // this method promises. It synchronises the device on return.
    BracewellTransform3D(deviceScratch, deviceInOut, W, H, D);
}

// Explicit instantiation is per translation unit: it only reaches members
// whose definition is visible here, so each backend file instantiates its
// own.
template void HartleyTransform<double>::DHT1DCuda(double*);
template void HartleyTransform<double>::DHT1DOnDevice(double*, double*);
template void HartleyTransform<double>::DHT2DCuda(double*);
template void HartleyTransform<double>::DHT2DOnDevice(double*, double*);
template void HartleyTransform<double>::DHT3DCuda(double*);
template void HartleyTransform<double>::DHT3DOnDevice(double*, double*);
template void HartleyTransform<float>::DHT1DCuda(float*);
template void HartleyTransform<float>::DHT1DOnDevice(float*, float*);
template void HartleyTransform<float>::DHT2DCuda(float*);
template void HartleyTransform<float>::DHT2DOnDevice(float*, float*);
template void HartleyTransform<float>::DHT3DCuda(float*);
template void HartleyTransform<float>::DHT3DOnDevice(float*, float*);

} // namespace RapiDHT
