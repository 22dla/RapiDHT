/*
 * Project: RapiDHT
 * File: rapidht.cpp
 * Brief: Реализация ND-преобразований Хартли (CPU/OpenMP и GPU/CUDA), транспонирования и Bracewell.
 * Author: Волков Евгений Александрович, volkov22dla@yandex.ru
 */

// rapidht.h must come first: it pulls in the generated rapidht_config.h that
// defines RAPIDHT_WITH_CUDA / RAPIDHT_WITH_MPI used by the guards below.
#include "rapidht.h"
#include "utilities.h"

#include <complex>
#include <omp.h>

#ifdef RAPIDHT_WITH_CUDA
#include "dev_array.h"
#include "kernel.h"
#include <cublas_v2.h>
#endif

#ifdef RAPIDHT_WITH_MPI
#include <mpi.h>
#endif

namespace RapiDHT {

namespace {

/// Rank/size of MPI_COMM_WORLD, or the trivial single-process values when MPI
/// is either disabled at build time or not initialised by the host program.
struct MpiContext {
    int rank = 0;
    int size = 1;
    bool active = false;
};

MpiContext QueryMpi()
{
    MpiContext ctx;
#ifdef RAPIDHT_WITH_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        ctx.active = true;
        MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);
    }
#endif
    return ctx;
}

void MpiBarrier(const MpiContext& ctx)
{
#ifdef RAPIDHT_WITH_MPI
    if (ctx.active) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
#else
    (void)ctx;
#endif
}

#ifdef RAPIDHT_WITH_MPI
/*
 * Maps the element type onto its MPI datatype. Previously MPI_DOUBLE was
 * hardcoded inside this class template, so HartleyTransform<float> told MPI
 * that each element was eight bytes wide: the gather read past the end of the
 * send buffer and scattered the result over twice the intended receive area.
 */
template <typename T>
struct MpiDatatype;

template <>
struct MpiDatatype<float> {
    static MPI_Datatype value() { return MPI_FLOAT; }
};

template <>
struct MpiDatatype<double> {
    static MPI_Datatype value() { return MPI_DOUBLE; }
};
#endif

/// Exact power-of-two test. The previous check compared ceil(log2(n)) with
/// floor(log2(n)), which relies on floating point being exact at the boundary.
constexpr bool IsPowerOfTwo(size_t n) noexcept
{
    return n != 0 && (n & (n - 1)) == 0;
}

[[noreturn]] void ThrowGpuUnavailable()
{
    throw std::runtime_error(
        "RapiDHT was built without CUDA support (RAPIDHT_WITH_CUDA=OFF): "
        "Modes::GPU is unavailable. Reconfigure with -DRAPIDHT_WITH_CUDA=ON.");
}

} // namespace

bool IsGpuAvailable() noexcept
{
#ifdef RAPIDHT_WITH_CUDA
    int count = 0;
    // Reports the failure through its return value rather than a sticky error,
    // so an absent or too-old driver just yields false.
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

/* ------------------------------ DeviceVolume ------------------------------ */

template <typename T>
struct DeviceVolume<T>::Impl {
#ifdef RAPIDHT_WITH_CUDA
    dev_array<T> storage;
    size_t count = 0;
#endif
};

template <typename T>
DeviceVolume<T>::DeviceVolume(size_t count)
{
#ifdef RAPIDHT_WITH_CUDA
    if (count == 0) {
        throw std::invalid_argument("DeviceVolume: size must be positive.");
    }
    _impl = std::make_unique<Impl>();
    _impl->storage.resize(count);
    _impl->count = count;
#else
    (void)count;
    throw std::runtime_error(
        "RapiDHT was built without CUDA support (RAPIDHT_WITH_CUDA=OFF): "
        "DeviceVolume is unavailable. Reconfigure with -DRAPIDHT_WITH_CUDA=ON.");
#endif
}

template <typename T>
DeviceVolume<T>::~DeviceVolume() = default;

template <typename T>
DeviceVolume<T>::DeviceVolume(DeviceVolume&&) noexcept = default;

template <typename T>
DeviceVolume<T>& DeviceVolume<T>::operator=(DeviceVolume&&) noexcept = default;

template <typename T>
void DeviceVolume<T>::Upload(const T* host)
{
#ifdef RAPIDHT_WITH_CUDA
    if (host == nullptr) {
        throw std::invalid_argument("DeviceVolume::Upload: host pointer is null.");
    }
    _impl->storage.set(host, _impl->count);
#else
    (void)host;
#endif
}

template <typename T>
void DeviceVolume<T>::Download(T* host) const
{
#ifdef RAPIDHT_WITH_CUDA
    if (host == nullptr) {
        throw std::invalid_argument("DeviceVolume::Download: host pointer is null.");
    }
    _impl->storage.get(host, _impl->count);
#else
    (void)host;
#endif
}

template <typename T>
size_t DeviceVolume<T>::Size() const noexcept
{
#ifdef RAPIDHT_WITH_CUDA
    return _impl ? _impl->count : 0;
#else
    return 0;
#endif
}

template <typename T>
void* DeviceVolume<T>::DeviceData() const noexcept
{
#ifdef RAPIDHT_WITH_CUDA
    return _impl ? static_cast<void*>(const_cast<T*>(_impl->storage.getData())) : nullptr;
#else
    return nullptr;
#endif
}

/* ---------------------------- HartleyTransform ---------------------------- */

/*
 * Opaque device-side state. Declared in the public header, defined only here,
 * so that no CUDA type ever appears in an installed header. In a CPU-only
 * build the struct is empty and the owning pointer is never allocated.
 */
template <typename T>
struct HartleyTransform<T>::Impl {
#ifdef RAPIDHT_WITH_CUDA
    std::array<dev_array<T>, static_cast<size_t>(Direction::Count)> transformMatrices;

    /*
     * Working buffers, allocated once with the object rather than on every
     * call. They used to be locals inside each DHT*Cuda method, so a 512^3
     * transform allocated and released two 512 MiB regions, and created and
     * destroyed two CUDA streams, every single time it ran.
     *
     * Holding them costs twice the volume in device memory for the lifetime of
     * the object, which is the usual bargain for a transform plan.
     */
    dev_array<T> scratchA;
    dev_array<T> scratchB;
#endif
};

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

        matrices[static_cast<size_t>(Direction::Y)].resize(Width() * Width());
        matrices[static_cast<size_t>(Direction::X)].resize(Height() * Height());
        matrices[static_cast<size_t>(Direction::Z)].resize(Depth() * Depth());

        InitializeHartleyMatrix(matrices[static_cast<size_t>(Direction::X)].getData(), Height());
        InitializeHartleyMatrix(matrices[static_cast<size_t>(Direction::Y)].getData(), Width());
        InitializeHartleyMatrix(matrices[static_cast<size_t>(Direction::Z)].getData(), Depth());

        // Sized for the whole volume, which is the largest any of the 1D, 2D
        // or 3D paths asks for.
        const size_t totalElements = Width() * (Height() == 0 ? size_t { 1 } : Height())
                                   * (Depth() == 0 ? size_t { 1 } : Depth());
        _impl->scratchA.resize(totalElements);
        _impl->scratchB.resize(totalElements);
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

template <typename T>
void HartleyTransform<T>::BitReverse(std::vector<size_t>& indices)
{
    PROFILE_FUNCTION();

    if (indices.empty()) {
        return;
    }

    const size_t n = indices.size();
    const int kLog2n = static_cast<int>(std::log2(n));

    indices[0] = 0;
    for (size_t j = 1; j < n; ++j) {
        size_t reversed = 0;
        size_t temp = j;
        for (int i = 0; i < kLog2n; ++i) {
            if (temp & 1) {
                // size_t{1}, not 1: shifting an int is undefined past 31 bits.
                reversed |= size_t { 1 } << (kLog2n - 1 - i);
            }
            temp >>= 1;
        }
        indices[j] = reversed;
    }
}

template <typename T>
void HartleyTransform<T>::BuildTwiddleTable(Direction direction)
{
    PROFILE_FUNCTION();

    const size_t n = Length(direction);
    auto& table = _twiddles[static_cast<size_t>(direction)];
    table.clear();

    // Sizes that FDHT1D would reject need no table; it throws on its own.
    if (n < 4 || !IsPowerOfTwo(n)) {
        return;
    }

    const T kPi = static_cast<T>(std::acos(-1));
    table.resize(n);

    // Mirrors the loop structure of FDHT1D: stage s uses m2 = 2^(s-1) and
    // indices j in [1, m2/2), with angle j*pi/m2.
    for (size_t s = 2; (size_t(1) << s) <= n; ++s) {
        const size_t m2 = size_t(1) << (s - 1);
        const size_t m4 = m2 / 2;
        for (size_t j = 1; j < m4; ++j) {
            const T angle = static_cast<T>(j) * kPi / static_cast<T>(m2);
            table[m2 + j] = Twiddle { std::cos(angle), std::sin(angle) };
        }
    }
}

template <typename T>
void HartleyTransform<T>::Series1D(T* data, Direction direction)
{
    PROFILE_FUNCTION();

    if (data == nullptr) {
        throw std::invalid_argument("The pointer to image is null.");
    }

    size_t M1 = 0, M2 = 0;
    switch (direction) {
        case Direction::Y:
            M1 = Height();
            M2 = (Depth() == 0 ? 1 : Depth());
            break;
        case Direction::X:
            M1 = Width();
            M2 = (Depth() == 0 ? 1 : Depth());
            break;
        case Direction::Z:
            M1 = Width();
            M2 = Height();
            break;
        default:
            throw std::invalid_argument("Invalid direction");
    }

    if (_mode == Modes::CPU) {
#pragma omp parallel for
        for (int i = 0; i < M1; ++i) {
            for (size_t j = 0; j < M2; ++j) {
                auto index = AxisIndex(0, i, j, direction);
                FDHT1D(data + index, direction);
            }
        }
        return;
    }
    // if (_mode == Modes::RFFT) {
    // #pragma omp parallel for
    //	for (int i = 0; i < Width(); ++i) {
    //		RealFFT1D(image_ptr + i * Height(), direction);
    //	}
    //	return;
    // }
}

template <typename T>
void HartleyTransform<T>::BracewellTransform2DCPU(T* image_ptr)
{
    PROFILE_FUNCTION();

    int W = Width();
    int H = Height();

    std::vector<T> result(W * H, T(0));

    // collapse(N) требует идеально вложенных циклов: объявления переносим внутрь
#pragma omp parallel for collapse(2)
    for (int y = 0; y < W; ++y) {
        for (int x = 0; x < H; ++x) {
            const int ym = (y > 0) ? (W - y) : 0;
            const int xm = (x > 0) ? (H - x) : 0;

            const T A = image_ptr[LinearIndex(y, x, 0)];
            const T B = image_ptr[LinearIndex(y, xm, 0)]; // flip X
            const T C = image_ptr[LinearIndex(ym, x, 0)]; // flip Y
            const T D = image_ptr[LinearIndex(ym, xm, 0)]; // flip both

            result[LinearIndex(y, x, 0)] = (A + B + C - D) / static_cast<T>(2);
        }
    }

    std::copy(result.begin(), result.end(), image_ptr);
}

template <typename T>
void HartleyTransform<T>::BracewellTransform3DCPU(T* volumePtr)
{
    PROFILE_FUNCTION();
    int W = Width();
    int H = Height();
    int D = Depth();

    std::vector<T> result(W * H * D, T(0));

    // collapse(N) требует идеально вложенных циклов: объявления переносим внутрь
#pragma omp parallel for collapse(3)
    for (int y = 0; y < W; ++y) {
        for (int x = 0; x < H; ++x) {
            for (int z = 0; z < D; ++z) {
                const int ym = (y > 0) ? (W - y) : 0;
                const int xm = (x > 0) ? (H - x) : 0;
                const int zm = (z > 0) ? (D - z) : 0;

                const T A = volumePtr[LinearIndex(ym, x, z)]; // flip X
                const T B = volumePtr[LinearIndex(y, xm, z)]; // flip Y
                const T C = volumePtr[LinearIndex(y, x, zm)]; // flip Z
                const T D_ = volumePtr[LinearIndex(ym, xm, zm)]; // flip all

                result[LinearIndex(y, x, z)] = (A + B + C - D_) / static_cast<T>(2);
            }
        }
    }

    std::copy(result.begin(), result.end(), volumePtr);
}

template <typename T>
void HartleyTransform<T>::FDHT1D(T* data, Direction direction)
{
    if (data == nullptr) {
        throw std::invalid_argument("The pointer to vector is null.");
    }

    const size_t n = Length(direction);

    // No lower-bound check: n is size_t, so "n < 0" was always false.
    if (!IsPowerOfTwo(n)) {
        throw std::invalid_argument("FDHT1D: length must be a power of two, got "
                                    + std::to_string(n) + ".");
    }

    // временный буфер
    std::vector<T> vec(n);

    // собрать данные в буфер
    for (size_t idx = 0; idx < n; ++idx) {
        vec[idx] = data[AxisIndex(idx, 0, 0, direction)];
    }

    for (size_t i = 1; i < n; ++i) {
        auto j = BitReversedIndex(direction, i);
        if (j > i) {
            std::swap(vec[i], vec[j]);
        }
    }

    // FHT for 1rd axis
    const auto kLog2n = static_cast<size_t>(std::log2(n));
    const Twiddle* twiddles = _twiddles[static_cast<size_t>(direction)].data();

    // Main cicle
    for (size_t s = 1; s <= kLog2n; ++s) {
        const auto m = size_t(1) << s;
        const auto m2 = m / 2;
        const auto m4 = m / 4;

        // Hoisted out of the loop over r: the factors depend on the stage and
        // on j, never on the block, so the inner loops below only ever read
        // from the table built in the constructor.
        const Twiddle* stage = twiddles + m2;

        for (size_t r = 0; r <= n - m; r = r + m) {
            for (size_t j = 1; j < m4; ++j) {
                const size_t k = m2 - j;
                const auto u = vec[r + m2 + j];
                const auto v = vec[r + m2 + k];
                const T cosVal = stage[j].cosine;
                const T sinVal = stage[j].sine;
                vec[r + m2 + j] = u * cosVal + v * sinVal;
                vec[r + m2 + k] = u * sinVal - v * cosVal;
            }
            for (size_t j = 0; j < m2; ++j) {
                const auto u = vec[r + j];
                const auto v = vec[r + j + m2];
                vec[r + j] = u + v;
                vec[r + j + m2] = u - v;
            }
        }
    }

    // записать обратно
    for (size_t idx = 0; idx < n; ++idx) {
        data[AxisIndex(idx, 0, 0, direction)] = vec[idx];
    }
}

template <typename T>
void HartleyTransform<T>::FDHT2D(T* image_ptr)
{
    PROFILE_FUNCTION();

    if (image_ptr == nullptr) {
        throw std::invalid_argument("FDHT2D: the pointer to image is null.");
    }

    Series1D(image_ptr, Direction::X);
    Series1D(image_ptr, Direction::Y);

    BracewellTransform2DCPU(image_ptr);
}

template <typename T>
void HartleyTransform<T>::FDHT3D(T* volume_ptr)
{
    PROFILE_FUNCTION();

    if (volume_ptr == nullptr) {
        throw std::invalid_argument("FDHT3D: the pointer to volume is null.");
    }

    // 1D transforms along X, Y, Z dimensions
    Series1D(volume_ptr, Direction::Y);
    Series1D(volume_ptr, Direction::X);
    Series1D(volume_ptr, Direction::Z);
    // Bracewell 3D
    BracewellTransform3DCPU(volume_ptr);
}

template <typename T>
void HartleyTransform<T>::RealFFT1D(T* vec, Direction direction)
{
    PROFILE_FUNCTION();

    if (vec == nullptr) {
        throw std::invalid_argument("RealFFT1D: the pointer to vector is null.");
    }

    // No lower-bound check: Length() returns size_t, so "< 0" was always false.
    if (!IsPowerOfTwo(Length(direction))) {
        throw std::invalid_argument("RealFFT1D: length must be a power of two, got "
                                    + std::to_string(Length(direction)) + ".");
    }

    // RealFFT
    std::vector<std::complex<T>> x(Length(direction));
    for (size_t i = 0; i < Length(direction); i++) {
        x[i] = std::complex<T>(vec[i], 0);
    }
    size_t k = Length(direction);
    size_t n;
    const T thetaT = static_cast<T>(std::acos(-1)) / Length(direction);
    std::complex<T> phiT = std::complex<T>(std::cos(thetaT), -std::sin(thetaT)), TTT;
    while (k > 1) {
        n = k;
        k >>= 1;
        phiT = phiT * phiT;
        TTT = 1.0L;
        for (size_t l = 0; l < k; l++) {
            for (size_t a = l; a < Length(direction); a += n) {
                size_t b = a + k;
                std::complex<T> t = x[a] - x[b];
                x[a] += x[b];
                x[b] = t * TTT;
            }
            TTT *= phiT;
        }
    }
    // Decimate, reusing the table the constructor already builds.
    //
    // This used to reverse the bits inline with the classic 32-bit sequence of
    // masked shifts, but on a size_t the final "b << 16" keeps the bits that
    // the 32-bit version discards, and the following shift does not remove
    // them. Above n = 65536 the result exceeded the array: at n = 262144 it
    // reached index 12885164031, which segfaulted. The precomputed table is
    // both correct and already paid for.
    for (size_t a = 1; a < Length(direction); ++a) {
        const size_t b = BitReversedIndex(direction, a);
        if (b > a) {
            std::swap(x[a], x[b]);
        }
    }

    // The Hartley transform is Re(X) - Im(X), not Re(X): cas(t) = cos(t) + sin(t)
    // while the Fourier kernel is cos(t) - i*sin(t).
    for (size_t i = 0; i < Length(direction); i++) {
        vec[i] = x[i].real() - x[i].imag();
    }
}

#ifdef RAPIDHT_WITH_CUDA

template <typename T>
void HartleyTransform<T>::DHT1DCuda(T* h_x)
{
    PROFILE_FUNCTION();

    // Buffers live in Impl and are allocated once with the object, not on
    // every call.
    _impl->scratchA.set(h_x, Width());
    DHT1DOnDevice(_impl->scratchA.getData(), _impl->scratchB.getData());
    _impl->scratchA.get(h_x, Width());
}

template <typename T>
void HartleyTransform<T>::DHT1DOnDevice(T* deviceInOut, T* deviceScratch)
{
    PROFILE_FUNCTION();

    VectorMatrixMultiplication(_impl->transformMatrices[static_cast<size_t>(Direction::Y)].getData(),
        deviceInOut, deviceScratch, Width());

    // The multiply cannot write in place, so the answer lands in the scratch
    // buffer and has to come back to satisfy this method's contract.
    CUDA_CHECK(cudaMemcpy(deviceInOut, deviceScratch, Width() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
void HartleyTransform<T>::DHT2DCuda(T* h_X)
{
    PROFILE_FUNCTION();

    const size_t sliceSize = Width() * Height();

    _impl->scratchA.set(h_X, sliceSize);
    DHT2DOnDevice(_impl->scratchA.getData(), _impl->scratchB.getData());
    _impl->scratchA.get(h_X, sliceSize);
}

template <typename T>
void HartleyTransform<T>::DHT2DOnDevice(T* deviceInOut, T* deviceScratch)
{
    PROFILE_FUNCTION();

    const size_t sliceSize = Width() * Height();
    T* d_X = deviceInOut;
    T* d_Y = deviceScratch;

    // The slice is Height() rows of Width() elements, so the first pass runs
    // along the fast axis and must use the Width()-sized matrix -- that is
    // Direction::Y, per InitializeHartleyMatrix in the constructor.
    //
    // These two were the other way round, which made the inner dimension of
    // the multiply disagree with the size of the matrix: on any non-square
    // extent the kernel read past the end of the transform matrix, which is
    // why 8x4 and 16x8 produced stable garbage while 4x4 happened to work.
    MatrixMultiplication(d_X, _impl->transformMatrices[static_cast<size_t>(Direction::Y)].getData(),
        d_Y, Height(), Width(), Width());
    MatrixTranspose(d_Y, d_X, Height(), Width());

    MatrixMultiplication(d_X, _impl->transformMatrices[static_cast<size_t>(Direction::X)].getData(),
        d_Y, Width(), Height(), Height());
    MatrixTranspose(d_Y, d_X, Width(), Height());

    // Without this the GPU produced the separable transform while the CPU
    // produced the true multidimensional one: the two backends computed
    // different functions for every extent except 1D.
    BracewellTransform2D(d_X, d_Y, static_cast<int>(Width()), static_cast<int>(Height()));

    // The correction reads mirrored points and so cannot write in place; bring
    // the answer back to satisfy this method's contract.
    CUDA_CHECK(cudaMemcpy(d_X, d_Y, sliceSize * sizeof(T), cudaMemcpyDeviceToDevice));
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
void HartleyTransform<T>::DHT3DCuda(T* h_X)
{
    PROFILE_FUNCTION();

    const size_t totalSize = Width() * Height() * Depth();

    _impl->scratchA.set(h_X, totalSize);
    DHT3DOnDevice(_impl->scratchA.getData(), _impl->scratchB.getData());
    _impl->scratchA.get(h_X, totalSize);
}

template <typename T>
void HartleyTransform<T>::DHT3DOnDevice(T* deviceInOut, T* deviceScratch)
{
    PROFILE_FUNCTION();

    auto W = Width();
    auto H = Height();
    auto D = Depth();

    T* d_X = deviceInOut;
    T* d_Y = deviceScratch;

    cublasHandle_t handle;
    cublasCreate(&handle);
    const T alpha = 1.0;
    const T beta = 0.0;

    // -------------------------------
    // Приводим к column major
    // -------------------------------
    // One launch instead of one per slice. Profiling 512^3 showed 1024 geam
    // launches per transform against 3 GEMMs, with the device idle for 83% of
    // the wall time waiting between them.
    MatrixTransposeBatched(d_X, d_Y, static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(D));

    // -------------------------------
    // 1D Hartley along Y (batched GEMM)
    // -------------------------------
    {
        int m = H;
        int n = W;
        int k = W;
        int lda = H;
        int ldb = W;
        int ldc = H;

        long long int strideA = H * W;
        long long int strideB = 0;
        long long int strideC = H * W;

        int batchCount = D;

        CublasGemmStridedBatched<T>::call(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_Y, lda,
            strideA, _impl->transformMatrices[(size_t)Direction::Y].getData(), ldb, strideB,
            &beta, d_X, ldc, strideC, batchCount);
    }

    // Транспонируем
    MatrixTransposeBatched(d_X, d_Y, static_cast<int>(W), static_cast<int>(H),
        static_cast<int>(D));

    // -------------------------------
    // 1D Hartley along X (batched GEMM)
    // -------------------------------
    {
        int m = W;
        int n = H;
        int k = H;
        int lda = W;
        int ldb = H;
        int ldc = W;

        long long int strideA = H * W;
        long long int strideB = 0;
        long long int strideC = H * W;

        int batchCount = D;

        CublasGemmStridedBatched<T>::call(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_Y, lda,
            strideA, _impl->transformMatrices[(size_t)Direction::X].getData(), ldb, strideB,
            &beta, d_X, ldc, strideC, batchCount);
    }

    // Swap the Y and Z axes, rather than transposing each slice.
    //
    // At this point the volume is back in its natural layout, x fastest:
    //   idx = x + W*y + W*H*z
    // but the batched multiply below asks for H batches of a W x D matrix with
    // leading dimension W, that is
    //   idx = x + W*z + W*D*y
    // so Y and Z have to change places. The per-slice geam that used to sit
    // here transposed within each slice instead, which is a different
    // permutation entirely -- and one that happens to coincide only when the
    // extents are equal, which is why even the cubic case came out wrong.
    //
    // transpose_YZ_cuda does exactly this and was already written, instantiated
    // and never called.
    transpose_YZ_cuda(d_X, d_Y, static_cast<int>(W), static_cast<int>(H),
        static_cast<int>(D));

    {
        int m = W;
        int n = D;
        int k = D;
        int lda = W;
        int ldb = D;
        int ldc = W;

        long long int strideA = D * W;
        long long int strideB = 0;
        long long int strideC = D * W;

        int batchCount = H;

        CublasGemmStridedBatched<T>::call(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_Y, lda,
            strideA, _impl->transformMatrices[(size_t)Direction::Z].getData(), ldb, strideB,
            &beta, d_X, ldc, strideC, batchCount);
    }

    // Swap Y and Z back, restoring the natural layout. The volume is currently
    // (x, z, y), so it is the same operation applied with D and H exchanged.
    // permute_ZXY_simple used to be called here; it produces a third layout
    // again, which neither the correction below nor the copy back expects.
    transpose_YZ_cuda(d_X, d_Y, static_cast<int>(W), static_cast<int>(D),
        static_cast<int>(H));

    // -------------------------------
    // Bracewell 3D
    // -------------------------------
    // This call used to be commented out, which left the GPU computing the
    // separable transform while the CPU computed the true multidimensional
    // one. Writes into d_X because the correction reads mirrored points and
    // cannot share its input and output buffer.
    BracewellTransform3D(d_Y, d_X, static_cast<int>(W), static_cast<int>(H),
        static_cast<int>(D));

    // The result is in d_X, which is deviceInOut, as this method promises.

    cublasDestroy(handle);
    cudaDeviceSynchronize();
}

#else // !RAPIDHT_WITH_CUDA

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

#endif // RAPIDHT_WITH_CUDA

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
    TransformOnDevice(static_cast<T*>(volume.DeviceData()), _impl->scratchB.getData());
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
