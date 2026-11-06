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

#include <cstdio>
#include <cstdlib>

namespace RapiDHT {

template <typename T>
void HartleyTransform<T>::DHT1DCuda(T* hostData)
{
    PROFILE_FUNCTION();

    // Buffers live in Impl and are allocated once, not on every call; the
    // staging one on the first host-path call.
    auto& staging = _impl->Staging(Width());
    staging.Upload(hostData, Width());
    DHT1DOnDevice(staging.Data(), _impl->scratchB.Data());
    staging.Download(hostData, Width());
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
void HartleyTransform<T>::DHT2DCuda(T* hostData, bool separable)
{
    PROFILE_FUNCTION();

    const size_t sliceSize = Width() * Height();

    auto& staging = _impl->Staging(sliceSize);
    staging.Upload(hostData, sliceSize);
    DHT2DOnDevice(staging.Data(), _impl->scratchB.Data(), separable);
    staging.Download(hostData, sliceSize);
}

template <typename T>
void HartleyTransform<T>::DHT2DOnDevice(T* deviceInOut, T* deviceScratch, bool separable)
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

    // The separable transform ends here, already in deviceInOut. Synchronise
    // so that both variants return with the device idle, as the Bracewell
    // path below does.
    if (separable) {
        RAPIDHT_CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

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
void HartleyTransform<T>::DHT3DCuda(T* hostData, bool separable)
{
    PROFILE_FUNCTION();

    const size_t totalSize = Width() * Height() * Depth();

    auto& staging = _impl->Staging(totalSize);
    staging.Upload(hostData, totalSize);
    DHT3DOnDevice(staging.Data(), _impl->scratchB.Data(), separable);
    staging.Download(hostData, totalSize);
}

template <typename T>
void HartleyTransform<T>::DHT3DOnDevice(T* deviceInOut, T* deviceScratch, bool separable)
{
    PROFILE_FUNCTION();

    const int W = static_cast<int>(Width());
    const int H = static_cast<int>(Height());
    const int D = static_cast<int>(Depth());
    const long long plane = static_cast<long long>(W) * H;

    cublasHandle_t handle = _impl->cublas->Get();
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

    // Three out-of-place multiplies leave the separable spectrum in the
    // scratch buffer, so without the correction it has to be copied back --
    // one plain pass over the volume in place of the Bracewell kernel's
    // four-point gather.
    if (separable) {
        RAPIDHT_CUDA_CHECK(cudaMemcpy(deviceInOut, deviceScratch,
            static_cast<size_t>(plane) * static_cast<size_t>(D) * sizeof(T), cudaMemcpyDeviceToDevice));
        RAPIDHT_CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    // The correction reads mirrored points and cannot share its input and
    // output buffer, which conveniently lands the result in deviceInOut, as
    // this method promises. It synchronises the device on return.
    BracewellTransform3D(deviceScratch, deviceInOut, W, H, D);
}

namespace {

/*
 * RAPIDHT_TRACE_KERNELS=1 makes the factored path time each of its kernels
 * with CUDA events and print one line per call to stderr. It exists because
 * the profiler is not always usable (nsys from the Ubuntu archive ships
 * without its report importer); the events cost nothing when it is unset.
 */
class KernelTrace {
public:
    KernelTrace()
    {
        const char* flag = std::getenv("RAPIDHT_TRACE_KERNELS");
        _enabled = (flag != nullptr && flag[0] != '\0' && flag[0] != '0');
    }
    ~KernelTrace()
    {
        // Only the events that were created: destroying a null handle fails
        // with cudaErrorInvalidResourceHandle, and an ignored failure still
        // becomes the pending error that the next cudaGetLastError reports --
        // which is how this first showed up, as a "failed" kernel launch.
        for (int i = 0; i < _count; ++i) {
            cudaEventDestroy(_events[i]);
        }
    }
    KernelTrace(const KernelTrace&) = delete;
    KernelTrace& operator=(const KernelTrace&) = delete;

    /// Marks the end of the previous phase and the start of the next, `name`.
    void Mark(const char* name)
    {
        if (!_enabled || _count >= kMax) {
            return;
        }
        RAPIDHT_CUDA_CHECK(cudaEventCreate(&_events[_count]));
        RAPIDHT_CUDA_CHECK(cudaEventRecord(_events[_count]));
        _names[_count] = name;
        ++_count;
    }

    /// Call after the last phase has been synchronised.
    void Print() const
    {
        if (!_enabled || _count < 2) {
            return;
        }
        std::fprintf(stderr, "[RapiDHT trace]");
        float total = 0.0f;
        for (int i = 0; i + 1 < _count; ++i) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, _events[i], _events[i + 1]);
            total += ms;
            std::fprintf(stderr, " %s %.3f ms,", _names[i], static_cast<double>(ms));
        }
        std::fprintf(stderr, " total %.3f ms\n", static_cast<double>(total));
    }

private:
    static constexpr int kMax = 8;
    bool _enabled = false;
    int _count = 0;
    cudaEvent_t _events[kMax] = {};
    const char* _names[kMax] = {};
};

} // namespace

template <typename T>
void HartleyTransform<T>::FactoredOnDevice(T* deviceData, bool separable)
{
    PROFILE_FUNCTION();
    KernelTrace trace;

    const size_t W = Width();
    const size_t H = (Height() == 0 ? 1 : Height());
    const size_t D = (Depth() == 0 ? 1 : Depth());

    // The geometry of Series1D: per axis, the stride along a line, the lane
    // axis that neighbouring lines run along, and the remaining axis.
    struct Geometry {
        Direction axis;
        size_t axisStride, laneStride, laneLength, otherStride, otherLength;
    };
    const Geometry passes[] = {
        { Direction::Y, 1, W, H, W * H, D }, // along the fastest axis
        { Direction::X, W, 1, W, W * H, D },
        { Direction::Z, W * H, 1, W, W, H },
    };
    // Named by memory layout, not by the historical Direction names: "x" is
    // the fastest axis (Width), "z" the slowest.
    const char* const passNames[] = { "axis-x(width)", "axis-y(height)", "axis-z(depth)" };
    int passIndex = 0;

    for (const Geometry& g : passes) {
        const char* passName = passNames[passIndex++];
        const size_t a = static_cast<size_t>(g.axis);
        const internal::FactoredSplit& split = _impl->factoredSplit[a];
        if (split.n <= 1) {
            continue; // absent axis (1D, 2D) or length 1
        }
        trace.Mark(passName);
        const size_t n1 = split.n1;
        const size_t n2 = split.n2;
        const size_t pairs = (n1 / 2 == 0 ? 1 : n1 / 2);
        const T* t = _impl->factoredTables[a].Data();

        FactoredAxisTables<T> tables;
        tables.c1 = t;
        tables.c2 = tables.c1 + n1 * n1;
        tables.cHalf = tables.c2 + n2 * n2;
        tables.twC = tables.cHalf + n2 * n2;
        tables.twD = tables.twC + pairs * n2;
        tables.n = static_cast<int>(split.n);
        tables.n1 = static_cast<int>(n1);
        tables.n2 = static_cast<int>(n2);

        FactoredAxis(deviceData, tables, g.axisStride, g.laneStride, g.laneLength, g.otherStride, g.otherLength);
    }

    if (!separable) {
        trace.Mark("bracewell");
        if (Depth() > 0) {
            BracewellInPlace3D(deviceData, static_cast<int>(W), static_cast<int>(H), static_cast<int>(D));
        } else if (Height() > 0) {
            BracewellInPlace2D(deviceData, static_cast<int>(W), static_cast<int>(H));
        }
    }
    trace.Mark("end");
    // Every other device path returns with the device idle; so does this one,
    // and a kernel fault surfaces here rather than at some later call.
    RAPIDHT_CUDA_CHECK(cudaDeviceSynchronize());
    trace.Print();
}

template <typename T>
void HartleyTransform<T>::FactoredOnHost(T* hostData, bool separable)
{
    const size_t total = Width() * (Height() == 0 ? 1 : Height()) * (Depth() == 0 ? 1 : Depth());
    auto& staging = _impl->Staging(total);
    staging.Upload(hostData, total);
    FactoredOnDevice(staging.Data(), separable);
    staging.Download(hostData, total);
}

// Explicit instantiation is per translation unit: it only reaches members
// whose definition is visible here, so each backend file instantiates its
// own.
template void HartleyTransform<double>::DHT1DCuda(double*);
template void HartleyTransform<double>::DHT1DOnDevice(double*, double*);
template void HartleyTransform<double>::DHT2DCuda(double*, bool);
template void HartleyTransform<double>::DHT2DOnDevice(double*, double*, bool);
template void HartleyTransform<double>::DHT3DCuda(double*, bool);
template void HartleyTransform<double>::DHT3DOnDevice(double*, double*, bool);
template void HartleyTransform<double>::FactoredOnDevice(double*, bool);
template void HartleyTransform<double>::FactoredOnHost(double*, bool);
template void HartleyTransform<float>::DHT1DCuda(float*);
template void HartleyTransform<float>::DHT1DOnDevice(float*, float*);
template void HartleyTransform<float>::DHT2DCuda(float*, bool);
template void HartleyTransform<float>::DHT2DOnDevice(float*, float*, bool);
template void HartleyTransform<float>::DHT3DCuda(float*, bool);
template void HartleyTransform<float>::DHT3DOnDevice(float*, float*, bool);
template void HartleyTransform<float>::FactoredOnDevice(float*, bool);
template void HartleyTransform<float>::FactoredOnHost(float*, bool);

} // namespace RapiDHT
