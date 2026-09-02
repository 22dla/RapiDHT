/*
 * Project: RapiDHT
 * File: include/rapidht/transform.h
 * Brief: The public API of the discrete Hartley transform, 1D/2D/3D, CPU and GPU.
 * Author: Volkov Evgeny Aleksandrovich, volkov22dla@yandex.ru
 */

#ifndef RAPIDHT_TRANSFORM_H
#define RAPIDHT_TRANSFORM_H

#include <rapidht/config_flags.h>
#include <rapidht/device_volume.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace RapiDHT {

/// Axis of a transform. The names are historical: Y runs along the fastest
/// varying index, X along the next, Z along the slowest.
enum class Direction : size_t { Y = 0,
    X = 1,
    Z = 2,
    Count };

/// Which backend carries out the transform.
enum class Modes { CPU, ///< OpenMP, fast Hartley transform.
    GPU, ///< CUDA, dense matrices through cuBLAS. Needs RAPIDHT_WITH_CUDA.
    RFFT ///< Real FFT, 1D only, for comparison.
};

/**
 * @brief The discrete Hartley transform of a 1D, 2D or 3D array.
 *
 * This computes the **true multidimensional** transform,
 *
 *     H(a,b,c) = sum over i,j,k of x(i,j,k) * cas(2*pi*(ia/W + jb/H + kc/D))
 *
 * where cas(t) = cos(t) + sin(t). That is not the same thing as applying a 1D
 * transform along each axis in turn: the separable product and the true
 * transform coincide only in 1D. The 2D and 3D paths therefore end with a
 * Bracewell correction, on both backends.
 *
 * Conventions, all deliberate:
 *  - the forward transform is unnormalised; InverseTransform applies the 1/N;
 *  - storage is row-major, index = i + Width*(j + Height*k);
 *  - every extent must be a power of two;
 *  - a 1D transform is (W, 0, 0) and a 2D one is (W, H, 0).
 *
 * The object owns whatever the backend needs -- bit-reversal tables and
 * twiddles on the CPU, transform matrices and scratch buffers on the device --
 * so it is worth keeping one around and reusing it, the way an FFT plan is
 * reused. It is move-only for the same reason.
 *
 * @tparam T Element type: float or double, the only two instantiated.
 */
template <typename T>
class HartleyTransform {
public:
    HartleyTransform() = delete;

    /**
     * @brief Builds a transform for one fixed extent and backend.
     *
     * All the per-extent setup happens here rather than per call.
     *
     * @param width Extent along the fastest axis; must be a power of two.
     * @param height Extent along the middle axis, or 0 for a 1D transform.
     * @param depth Extent along the slowest axis, or 0 for 1D and 2D.
     * @param mode Backend to use.
     *
     * @throws std::invalid_argument If width is zero, if depth is given
     *         without height, or if Modes::RFFT is asked for in 2D or 3D,
     *         where it is not implemented.
     * @throws std::runtime_error If Modes::GPU is asked for in a build without
     *         the CUDA backend.
     */
    HartleyTransform(size_t width, size_t height, size_t depth, Modes mode);

    // Declared here and defined in the translation unit where Impl is
    // complete. Copying is disabled because the object may own device
    // memory, which cannot be duplicated meaningfully.
    ~HartleyTransform();
    HartleyTransform(const HartleyTransform&) = delete;
    HartleyTransform& operator=(const HartleyTransform&) = delete;
    HartleyTransform(HartleyTransform&&) noexcept;
    HartleyTransform& operator=(HartleyTransform&&) noexcept;

    /**
     * @brief Transforms an array in host memory, in place, unnormalised.
     *
     * On Modes::GPU this uploads the data, transforms it and downloads it
     * again on every call. For anything that applies more than one operation,
     * prefer the DeviceVolume overload below and pay for the transfer once.
     *
     * @param data Input/output, Width()*Height()*Depth() elements, row-major.
     */
    void ForwardTransform(T* data);

    /**
     * @brief The inverse: a forward transform followed by scaling by 1/N.
     * @param data Input/output, left holding the reconstructed data.
     */
    void InverseTransform(T* data);

    /**
     * @brief Transforms a volume already resident on the device, in place.
     *
     * No host/device copying takes place, so a pipeline that applies several
     * operations pays for the transfer once rather than per call. Requires
     * Modes::GPU and a volume whose size matches this transform's extent.
     *
     * @param volume Input/output, left holding the transformed data.
     */
    void ForwardTransform(DeviceVolume<T>& volume);

    /**
     * @brief The in-place inverse of the above, on device-resident data.
     * @param volume Input/output, left holding the transformed data.
     */
    void InverseTransform(DeviceVolume<T>& volume);

    constexpr size_t Width() const noexcept { return _dims[static_cast<size_t>(Direction::Y)]; }
    constexpr size_t Height() const noexcept { return _dims[static_cast<size_t>(Direction::X)]; }
    constexpr size_t Depth() const noexcept { return _dims[static_cast<size_t>(Direction::Z)]; }

    /// Linear offset of element (i, j, k) in row-major storage.
    size_t LinearIndex(size_t i, size_t j, size_t k) const
    {
        return i + Width() * (j + Height() * k); // row-major
    }

    /// Linear offset of the point that sits at `idxAlongAxis` along
    /// `direction`, with the other two coordinates fixed.
    size_t AxisIndex(size_t idxAlongAxis, size_t fixed1, size_t fixed2, Direction direction) const
    {
        switch (direction) {
            case Direction::Y:
                return LinearIndex(idxAlongAxis, fixed1, fixed2);
            case Direction::X:
                return LinearIndex(fixed1, idxAlongAxis, fixed2);
            case Direction::Z:
                return LinearIndex(fixed1, fixed2, idxAlongAxis);
            default:
                throw std::invalid_argument("Invalid direction");
        }
    }

    /**
     * @brief Returns the length of the specified direction.
     * @param direction Direction (X, Y, Z) to query.
     * @return Length along the specified direction.
     */
    constexpr size_t Length(Direction direction) const noexcept { return _dims[static_cast<size_t>(direction)]; }

    /**
     * @brief Returns the bit-reversed index for the given index and direction.
     * @param direction Direction (X, Y, Z) for the index.
     * @param index Original index.
     * @return Bit-reversed index.
     */
    inline size_t BitReversedIndex(Direction direction, size_t index) const noexcept
    {
        return _bitReversedIndices[static_cast<size_t>(direction)][index];
    }

private:
    /* ------------------------- ND Transforms ------------------------- */

    /**
     * @brief Performs a 1D Fast Hartley Transform on the given vector along the specified direction.
     * @param vector Pointer to the input/output data array.
     * @param direction Direction along which to perform the transform.
     */
    void FDHT1D(T* vector, Direction direction = Direction::Y);

    /**
     * @brief Performs a 2D Fast Hartley Transform on the given image.
     * @param image Pointer to the input/output 2D data array.
     */
    void FDHT2D(T* image);

    /**
     * @brief Performs a 3D Fast Hartley Transform on the given data.
     * @param data Pointer to the input/output 3D data array.
     */
    void FDHT3D(T* data);

    /*
     * GPU entry points. Declared unconditionally so that the class layout and
     * interface do not depend on RAPIDHT_WITH_CUDA -- otherwise a consumer
     * compiled with a different setting than the library would silently
     * disagree about this type. They are only defined, and only ever called,
     * in a CUDA-enabled build.
     */

    /**
     * @brief Performs a 1D Hartley Transform using CUDA matrix-vector multiplication.
     * @param hostData Pointer to the input/output data vector, Width() elements long.
     */
    void DHT1DCuda(T* hostData);

    /**
     * @brief Performs a 2D Hartley Transform using CUDA matrix-matrix multiplication.
     * @param image Pointer to the input/output 2D data array.
     */
    void DHT2DCuda(T* image);

    /**
     * @brief Performs a 3D Hartley Transform using CUDA matrix-matrix multiplication.
     * @param image Pointer to the input/output 3D data array.
     */
    void DHT3DCuda(T* data);

    /*
     * The device-side halves of the three above, working on memory that is
     * already there. The host-pointer versions are now a copy in, one of
     * these, and a copy out; the resident overloads call them directly.
     *
     * Each leaves its result in `deviceInOut` and may use `deviceScratch`,
     * which must hold at least as many elements.
     */
    void DHT1DOnDevice(T* deviceInOut, T* deviceScratch);
    void DHT2DOnDevice(T* deviceInOut, T* deviceScratch);
    void DHT3DOnDevice(T* deviceInOut, T* deviceScratch);

    /// Dispatches to the right rank, and rejects anything but Modes::GPU.
    void TransformOnDevice(T* deviceInOut, T* deviceScratch);

    /**
     * @brief Performs a 1D Real Fourier Transform along the specified direction.
     * @param vector Pointer to the input/output data array.
     * @param direction Direction along which to perform the transform.
     */
    void RealFFT1D(T* vector, Direction direction = Direction::Y);

    /**
     * @brief Performs a series of 1D transforms along the given direction.
     * @param image Pointer to the input/output data array.
     * @param direction Direction along which to perform the series of transforms.
     */
    void Series1D(T* image, Direction direction);

    /**
     * @brief Computes bit-reversed indices for FFT.
     * @param indices Pointer to the vector of indices to fill.
     */
    static void BitReverse(std::vector<size_t>& indices);

    /**
     * @brief Performs the 2D Hartley Transform on the CPU using Bracewell's algorithm.
     * @param imagePtr Pointer to the input/output 2D data array.
     */
    void BracewellTransform2DCPU(T* imagePtr);

    /**
     * @brief Performs the 3D Hartley Transform on the CPU using Bracewell's algorithm.
     * @param imagePtr Pointer to the input/output 3D data array.
     */
    void BracewellTransform3DCPU(T* volumePtr);

    /**
     * @brief Fills _twiddles for the given direction.
     * @param direction Axis whose length determines the table size.
     */
    void BuildTwiddleTable(Direction direction);

    std::array<size_t, static_cast<size_t>(Direction::Count)> _dims { };
    std::array<std::vector<size_t>, static_cast<size_t>(Direction::Count)> _bitReversedIndices;

    Modes _mode = Modes::CPU;

    /*
     * Butterfly twiddle factors, precomputed once per axis.
     *
     * FDHT1D used to evaluate std::cos and std::sin inside its innermost loop.
     * The argument there depends only on the stage and the butterfly index, so
     * the same handful of values was recomputed for every block of every stage
     * and again on every call. Measured on the benchmark suite, that
     * trigonometry accounted for roughly 70% of the run time.
     *
     * Layout: stage s occupies [2^(s-1) .. 2^s), so the factor for index j of
     * stage s lives at _twiddles[m2 + j] where m2 = 2^(s-1). That wastes the
     * first slot and costs n entries in total, against n/2 for a tightly
     * packed layout, in exchange for index arithmetic that stays trivial.
     */
    struct Twiddle {
        T cosine;
        T sine;
    };
    std::array<std::vector<Twiddle>, static_cast<size_t>(Direction::Count)> _twiddles;

    /*
     * Device-side state lives behind an opaque pointer so that this header
     * never names a CUDA type. Without it, <cuda_runtime.h> reached every
     * consumer of the library through device_array.h, forcing them to have a
     * CUDA toolkit installed even to use the CPU backend.
     *
     * The member is present in every configuration, keeping the class layout
     * independent of RAPIDHT_WITH_CUDA. In a CPU-only build Impl is simply
     * empty and the pointer stays null.
     */
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace RapiDHT

#endif // RAPIDHT_TRANSFORM_H
