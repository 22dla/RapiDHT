/*
 * Project: RapiDHT
 * File: src/transform.cpp
 * Brief: HartleyTransform: construction, dispatch and the public entry points.
 */

#include <rapidht/transform.h>
#include <rapidht/utilities.h>

#include "internal/impl.h"
#include "internal/support.h"

#ifdef RAPIDHT_WITH_CUDA
#include "internal/kernels.h"
#endif

#include <cmath>
#include <string>

namespace RapiDHT {

using internal::IsPowerOfTwo;
using internal::MpiBarrier;
using internal::MpiContext;
using internal::QueryMpi;
using internal::ThrowGpuUnavailable;

/* ---------------------------- HartleyTransform ---------------------------- */

template <typename T>
HartleyTransform<T>::~HartleyTransform() = default;

template <typename T>
HartleyTransform<T>::HartleyTransform(HartleyTransform&&) noexcept = default;

template <typename T>
HartleyTransform<T>& HartleyTransform<T>::operator=(HartleyTransform&&) noexcept = default;

template <typename T>
HartleyTransform<T>::HartleyTransform(size_t width, size_t height, size_t depth, Modes mode):
    _mode(mode)
{
    PROFILE_FUNCTION();

    if (width == 0) {
        throw std::invalid_argument("Width must be positive.");
    }
    if (height == 0 && depth > 0) {
        throw std::invalid_argument("If height is zero, depth must also be zero.");
    }
    // RealFFT1D is the only thing this mode has. Asking for it in 2D or 3D used
    // to run FDHT2D/FDHT3D instead, silently handing back a different backend
    // than the caller selected.
    if (mode == Modes::RFFT && height > 0) {
        throw std::invalid_argument(
            "Modes::RFFT is implemented for 1D only. Use Modes::CPU for 2D and 3D.");
    }

    _dims = { width, height, depth };

    // Preparation to 1D transforms
    if (_mode == Modes::CPU || _mode == Modes::RFFT) {
        for (size_t i = 0; i < _bitReversedIndices.size(); ++i) {
            _bitReversedIndices[i].resize(_dims[i]);
            BitReverse(_bitReversedIndices[i]);
            BuildTwiddleTable(static_cast<Direction>(i));
        }
    }
    if (_mode == Modes::GPU) {
#ifdef RAPIDHT_WITH_CUDA
        // Allocated only for the GPU backend: a CPU-only transform must not
        // touch the CUDA runtime at all, not even to create a stream.
        _impl = std::make_unique<Impl>();
        auto& matrices = _impl->transformMatrices;

        matrices[static_cast<size_t>(Direction::Y)].Resize(Width() * Width());
        matrices[static_cast<size_t>(Direction::X)].Resize(Height() * Height());
        matrices[static_cast<size_t>(Direction::Z)].Resize(Depth() * Depth());

        InitializeHartleyMatrix(matrices[static_cast<size_t>(Direction::X)].Data(), Height());
        InitializeHartleyMatrix(matrices[static_cast<size_t>(Direction::Y)].Data(), Width());
        InitializeHartleyMatrix(matrices[static_cast<size_t>(Direction::Z)].Data(), Depth());

        // Sized for the whole volume, which is the largest any of the 1D, 2D
        // or 3D paths asks for.
        const size_t totalElements = Width() * (Height() == 0 ? size_t { 1 } : Height())
                                   * (Depth() == 0 ? size_t { 1 } : Depth());
        _impl->scratchA.Resize(totalElements);
        _impl->scratchB.Resize(totalElements);
#else
        ThrowGpuUnavailable();
#endif
    }
}

template <typename T>
void HartleyTransform<T>::ForwardTransform(T* data)
{
    PROFILE_FUNCTION();

    bool is1D = (Height() == 0 && Depth() == 0);
    bool is2D = (Height() > 0 && Depth() == 0);
    bool is3D = (Depth() > 0);

    // Rank/size, если MPI собран и инициализирован хост-программой
    const MpiContext mpi = QueryMpi();
    const int rank = mpi.rank;
    const int size = mpi.size;

    if (is1D || is2D) {
        // Для 1D и 2D нет смысла в MPI
        switch (_mode) {
            case Modes::CPU:
                if (is1D) {
                    FDHT1D(data);
                } else {
                    FDHT2D(data);
                }
                break;
            case Modes::GPU:
                if (is1D) {
                    DHT1DCuda(data);
                } else {
                    DHT2DCuda(data);
                }
                break;
            case Modes::RFFT:
                // The constructor rejects RFFT for anything but 1D.
                RealFFT1D(data);
                break;
        }
        MpiBarrier(mpi);
        return;
    }

    // 3D case: делим по Z между процессами, если MPI включен
    size_t depthPerProc = Depth() / size;
    size_t remainder = Depth() % size;
    size_t offset = rank * depthPerProc + std::min(static_cast<size_t>(rank), remainder);
    depthPerProc += (rank < remainder) ? 1 : 0;

    T* localData = data + offset * Width() * Height();

    switch (_mode) {
        case Modes::CPU:
        case Modes::RFFT: // rejected for 3D by the constructor; kept for the warning
            FDHT3D(localData);
            break;
        case Modes::GPU:
            DHT3DCuda(localData);
            break;
    }

#ifdef RAPIDHT_WITH_MPI
    // Сбор данных только если MPI активен
    if (mpi.active) {
        std::vector<int> sendcounts(size);
        std::vector<int> displs(size);
        int offs = 0;
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = static_cast<int>((Depth() / size + (i < remainder ? 1 : 0)) * Width() * Height());
            displs[i] = offs;
            offs += sendcounts[i];
        }
        const MPI_Datatype elementType = MpiDatatype<T>::value();
        MPI_Allgatherv(localData, sendcounts[rank], elementType,
            data, sendcounts.data(), displs.data(), elementType,
            MPI_COMM_WORLD);
    }
#endif
}

template <typename T>
void HartleyTransform<T>::InverseTransform(T* data)
{
    PROFILE_FUNCTION();

    bool is1D = (Height() == 0 && Depth() == 0);
    bool is2D = (Height() > 0 && Depth() == 0);
    bool is3D = (Depth() > 0);

    const MpiContext mpi = QueryMpi();
    const int rank = mpi.rank;
    const int size = mpi.size;

    // Сначала выполняем прямое преобразование
    ForwardTransform(data);

    // Общий размер данных
    size_t totalSize = Width();
    if (Height() > 0) {
        totalSize *= Height();
    }
    if (Depth() > 0) {
        totalSize *= Depth();
    }

    auto denominator = 1.0 / static_cast<double>(totalSize);

    if (is1D || is2D) {
        // Масштабируем полностью только на rank=0
        for (size_t i = 0; i < totalSize; ++i) {
            data[i] *= denominator;
        }
        MpiBarrier(mpi);
        return;
    }

    // 3D case: делим по Z между процессами
    size_t depthPerProc = Depth() / size;
    size_t remainder = Depth() % size;
    size_t offset = rank * depthPerProc + std::min(static_cast<size_t>(rank), remainder);
    depthPerProc += (rank < remainder) ? 1 : 0;

    size_t localSize = depthPerProc * Width() * Height();
    T* localData = data + offset * Width() * Height();

    // Масштабируем локальный блок
    for (size_t i = 0; i < localSize; ++i) {
        localData[i] *= denominator;
    }

    // Синхронизация процессов
    MpiBarrier(mpi);
}

#ifndef RAPIDHT_WITH_CUDA

/*
 * Defined even without CUDA so that the class has no member that is declared
 * but never defined, which keeps explicit instantiation well behaved across
 * compilers. Unreachable in practice: the constructor already rejects
 * Modes::GPU in this configuration.
 */
template <typename T>
void HartleyTransform<T>::DHT1DCuda(T*)
{
    ThrowGpuUnavailable();
}

template <typename T>
void HartleyTransform<T>::DHT2DCuda(T*)
{
    ThrowGpuUnavailable();
}

template <typename T>
void HartleyTransform<T>::DHT3DCuda(T*)
{
    ThrowGpuUnavailable();
}

template <typename T>
void HartleyTransform<T>::DHT1DOnDevice(T*, T*)
{
    ThrowGpuUnavailable();
}

template <typename T>
void HartleyTransform<T>::DHT2DOnDevice(T*, T*)
{
    ThrowGpuUnavailable();
}

template <typename T>
void HartleyTransform<T>::DHT3DOnDevice(T*, T*)
{
    ThrowGpuUnavailable();
}

#endif

template <typename T>
void HartleyTransform<T>::TransformOnDevice(T* deviceInOut, T* deviceScratch)
{
    if (_mode != Modes::GPU) {
        throw std::invalid_argument(
            "Device-resident transforms require Modes::GPU; this object was built for another mode.");
    }

    const bool is1D = (Height() == 0 && Depth() == 0);
    const bool is2D = (Height() > 0 && Depth() == 0);

    if (is1D) {
        DHT1DOnDevice(deviceInOut, deviceScratch);
    } else if (is2D) {
        DHT2DOnDevice(deviceInOut, deviceScratch);
    } else {
        DHT3DOnDevice(deviceInOut, deviceScratch);
    }
}

template <typename T>
void HartleyTransform<T>::ForwardTransform(DeviceVolume<T>& volume)
{
    PROFILE_FUNCTION();

    // This has to come first. _impl exists only for Modes::GPU, and the call
    // below reads _impl->scratchB while evaluating its own arguments -- before
    // any check inside the callee can run. Validating the mode there instead
    // meant dereferencing a null _impl on the way to the diagnostic.
    if (_mode != Modes::GPU) {
        throw std::invalid_argument(
            "Device-resident transforms require Modes::GPU; this object was built for another mode.");
    }

    const size_t expected = Width() * (Height() == 0 ? size_t { 1 } : Height())
                          * (Depth() == 0 ? size_t { 1 } : Depth());
    if (volume.Size() != expected) {
        throw std::invalid_argument("DeviceVolume holds " + std::to_string(volume.Size())
                                    + " elements but this transform expects " + std::to_string(expected) + ".");
    }

    if (volume.DeviceData() == nullptr) {
        throw std::invalid_argument("DeviceVolume holds no allocation.");
    }

#ifdef RAPIDHT_WITH_CUDA
    // scratchB is the working buffer; the volume itself plays the part that
    // scratchA plays on the host path.
    TransformOnDevice(static_cast<T*>(volume.DeviceData()), _impl->scratchB.Data());
#else
    ThrowGpuUnavailable();
#endif
}

template <typename T>
void HartleyTransform<T>::InverseTransform(DeviceVolume<T>& volume)
{
    PROFILE_FUNCTION();

    ForwardTransform(volume);

#ifdef RAPIDHT_WITH_CUDA
    // The inverse is the forward transform scaled by 1/N. Reuse the existing
    // scaling by borrowing the transform matrix path would be wrong here, so
    // scale on the device directly.
    const size_t count = volume.Size();
    ScaleOnDevice(static_cast<T*>(volume.DeviceData()), count,
        static_cast<T>(1.0 / static_cast<double>(count)));
#else
    ThrowGpuUnavailable();
#endif
}

template class DeviceVolume<float>;
template class DeviceVolume<double>;

template class HartleyTransform<float>;
template class HartleyTransform<double>;

} // namespace RapiDHT
