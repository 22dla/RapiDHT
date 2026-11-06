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
#include "internal/factored.h"
#include "internal/kernels.h"
#endif

#include <cmath>
#include <string>

namespace RapiDHT {

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
        _impl->cublas = std::make_unique<internal::CublasHandle>();
        auto& matrices = _impl->transformMatrices;

        matrices[static_cast<size_t>(Direction::Y)].Resize(Width() * Width());
        matrices[static_cast<size_t>(Direction::X)].Resize(Height() * Height());
        matrices[static_cast<size_t>(Direction::Z)].Resize(Depth() * Depth());

        InitializeHartleyMatrix(matrices[static_cast<size_t>(Direction::X)].Data(), Height());
        InitializeHartleyMatrix(matrices[static_cast<size_t>(Direction::Y)].Data(), Width());
        InitializeHartleyMatrix(matrices[static_cast<size_t>(Direction::Z)].Data(), Depth());

        // Sized for the whole volume, which is the largest any of the 1D, 2D
        // or 3D paths asks for. The host-path staging buffer is not allocated
        // here but on first use; see Impl.
        const size_t totalElements = Width() * (Height() == 0 ? size_t { 1 } : Height())
                                   * (Depth() == 0 ? size_t { 1 } : Depth());
        _impl->scratchB.Resize(totalElements);
#else
        ThrowGpuUnavailable();
#endif
    }
    if (_mode == Modes::GPUFactored) {
#ifdef RAPIDHT_WITH_CUDA
        // Only the tables: the transform runs in place in shared memory, so
        // there is no scratch buffer, no matrix and no cuBLAS handle.
        _impl = std::make_unique<Impl>();
        for (size_t a = 0; a < _dims.size(); ++a) {
            const size_t n = (_dims[a] == 0 ? 1 : _dims[a]);
            if (n > kFactoredMaxLength) {
                throw std::invalid_argument("Modes::GPUFactored handles extents up to "
                                            + std::to_string(kFactoredMaxLength) + ", got "
                                            + std::to_string(n) + ".");
            }
            const auto tables = internal::MakeFactoredTables<T>(n); // throws unless a power of two
            std::vector<T> packed;
            for (const auto* part : { &tables.c1, &tables.c2, &tables.cHalf, &tables.twC, &tables.twD }) {
                packed.insert(packed.end(), part->begin(), part->end());
            }
            _impl->factoredSplit[a] = tables.split;
            _impl->factoredTables[a].Resize(packed.size());
            _impl->factoredTables[a].Upload(packed.data(), packed.size());
        }
#else
        ThrowGpuUnavailable();
#endif
    }
}

namespace {

/*
 * The 3D path used to split the volume along Z and run FDHT3D/DHT3DCuda on
 * each rank's slab, then Allgatherv the slabs. That cannot work: a 3D
 * transform couples every Z-plane to every other, so it needs a global
 * transpose (all-to-all) between the per-axis passes, and the per-rank call
 * still used the full extents, reading past the end of the slab on every rank
 * but the last. Until a real distributed transform exists, refuse to run under
 * more than one process rather than return a wrong answer.
 */
void RejectMultiProcess(const MpiContext& mpi)
{
    if (mpi.size > 1) {
        throw std::runtime_error(
            "RapiDHT: running under " + std::to_string(mpi.size)
            + " MPI processes is not supported yet; the distributed 3D transform "
              "is not implemented. Run with a single process.");
    }
}

} // namespace

template <typename T>
void HartleyTransform<T>::ForwardTransform(T* data)
{
    PROFILE_FUNCTION();

    RejectMultiProcess(QueryMpi());

    const bool is1D = (Height() == 0 && Depth() == 0);
    const bool is2D = (Height() > 0 && Depth() == 0);

    switch (_mode) {
        case Modes::CPU:
            if (is1D) {
                FDHT1D(data);
            } else if (is2D) {
                FDHT2D(data);
            } else {
                FDHT3D(data);
            }
            break;
        case Modes::GPU:
            if (is1D) {
                DHT1DCuda(data);
            } else if (is2D) {
                DHT2DCuda(data);
            } else {
                DHT3DCuda(data);
            }
            break;
        case Modes::RFFT:
            // The constructor rejects RFFT for anything but 1D.
            RealFFT1D(data);
            break;
        case Modes::GPUFactored:
            FactoredOnHost(data, false);
            break;
    }
}

template <typename T>
void HartleyTransform<T>::InverseTransform(T* data)
{
    PROFILE_FUNCTION();

    // The Hartley transform is its own inverse up to the 1/N below.
    ForwardTransform(data);

    size_t totalSize = Width();
    if (Height() > 0) {
        totalSize *= Height();
    }
    if (Depth() > 0) {
        totalSize *= Depth();
    }

    const T scale = static_cast<T>(1.0 / static_cast<double>(totalSize));
    for (size_t i = 0; i < totalSize; ++i) {
        data[i] *= scale;
    }
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
void HartleyTransform<T>::DHT2DCuda(T*, bool)
{
    ThrowGpuUnavailable();
}

template <typename T>
void HartleyTransform<T>::DHT3DCuda(T*, bool)
{
    ThrowGpuUnavailable();
}

template <typename T>
void HartleyTransform<T>::DHT1DOnDevice(T*, T*)
{
    ThrowGpuUnavailable();
}

template <typename T>
void HartleyTransform<T>::DHT2DOnDevice(T*, T*, bool)
{
    ThrowGpuUnavailable();
}

template <typename T>
void HartleyTransform<T>::DHT3DOnDevice(T*, T*, bool)
{
    ThrowGpuUnavailable();
}

template <typename T>
void HartleyTransform<T>::FactoredOnDevice(T*, bool)
{
    ThrowGpuUnavailable();
}

template <typename T>
void HartleyTransform<T>::FactoredOnHost(T*, bool)
{
    ThrowGpuUnavailable();
}

#endif

template <typename T>
void HartleyTransform<T>::TransformOnDevice(T* deviceInOut, T* deviceScratch, bool separable)
{
    if (_mode == Modes::GPUFactored) {
        FactoredOnDevice(deviceInOut, separable);
        return;
    }
    if (_mode != Modes::GPU) {
        throw std::invalid_argument(
            "Device-resident transforms require a device mode (Modes::GPU or Modes::GPUFactored); this object was built for another mode.");
    }

    const bool is1D = (Height() == 0 && Depth() == 0);
    const bool is2D = (Height() > 0 && Depth() == 0);

    if (is1D) {
        DHT1DOnDevice(deviceInOut, deviceScratch);
    } else if (is2D) {
        DHT2DOnDevice(deviceInOut, deviceScratch, separable);
    } else {
        DHT3DOnDevice(deviceInOut, deviceScratch, separable);
    }
}

template <typename T>
void HartleyTransform<T>::CheckResidentVolume(const DeviceVolume<T>& volume) const
{
    // The mode has to come first. _impl exists only for the device modes, and
    // the callers read _impl->scratchB right after this returns.
    if (!IsDeviceMode()) {
        throw std::invalid_argument(
            "Device-resident transforms require a device mode (Modes::GPU or Modes::GPUFactored); this object was built for another mode.");
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
}

template <typename T>
void HartleyTransform<T>::ForwardTransform(DeviceVolume<T>& volume)
{
    PROFILE_FUNCTION();

    // This has to come first. _impl exists only for Modes::GPU, and the call
    // below reads _impl->scratchB while evaluating its own arguments -- before
    // any check inside the callee can run. Validating the mode there instead
    // meant dereferencing a null _impl on the way to the diagnostic.
    CheckResidentVolume(volume);

#ifdef RAPIDHT_WITH_CUDA
    // scratchB is the working buffer; the volume itself plays the part that
    // the staging buffer plays on the host path, which is why that one is
    // never allocated for resident-only use.
    // scratchB is empty in Modes::GPUFactored, which needs none.
    TransformOnDevice(static_cast<T*>(volume.DeviceData()), _impl->scratchB.Data());
#else
    ThrowGpuUnavailable();
#endif
}

template <typename T>
void HartleyTransform<T>::ForwardSeparable(DeviceVolume<T>& volume)
{
    PROFILE_FUNCTION();

    CheckResidentVolume(volume);

#ifdef RAPIDHT_WITH_CUDA
    TransformOnDevice(static_cast<T*>(volume.DeviceData()), _impl->scratchB.Data(), true);
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
