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
#include <rapidht/integer_offset.h>

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
    RFFT, ///< Real FFT, 1D only, for comparison.
    /// CUDA, experimental (R-4): every axis as two small dense multiplies in
    /// shared memory, in place, with no device memory beyond the data itself.
    /// Extents up to 2048 per axis. Needs RAPIDHT_WITH_CUDA.
    GPUFactored
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
namespace internal {
struct DeviceAccess; // R-5: the internal entry points benchmarks and tests reach (src/internal)
} // namespace internal

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
     * @brief The separable Hartley transform: a 1D DHT along each axis in
     *        turn, with no Bracewell correction. In place, unnormalised.
     *
     * This is a different function from ForwardTransform in 2D and 3D (they
     * coincide in 1D). It is still its own inverse up to a factor of N, and it
     * is what convolution with an even kernel needs: if g is even along every
     * axis, g(i,j,k) = g(-i,j,k) = g(i,-j,k) = g(i,j,-k) (indices mod the
     * extent), then
     *
     *     S(f * g) = S(f) . S(g)       (circular convolution, pointwise product)
     *
     * so the correction pass and its extra buffer can be skipped entirely.
     * See EvenKernelConvolution in <rapidht/convolution.h>.
     *
     * Implemented for every mode. On Modes::GPU it is the true transform's
     * multiplies without the Bracewell kernel; in 3D one device-to-device copy
     * takes its place, because three out-of-place multiplies leave the result
     * in the scratch buffer.
     *
     * @param data Input/output, Width()*Height()*Depth() elements, row-major.
     */
    void ForwardSeparable(T* data);

    /**
     * @brief The separable transform of a volume already resident on the
     *        device, in place. Requires Modes::GPU and a matching size, as
     *        ForwardTransform(DeviceVolume&) does.
     * @param volume Input/output, left holding the separable spectrum.
     */
    void ForwardSeparable(DeviceVolume<T>& volume);

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

    /*
     * Integer voxels (R-5). U is std::uint8_t or std::uint16_t, the only two
     * instantiated; T, the spectrum, stays float or double. The integers are
     * read as they are -- a uint16 volume crosses the bus and sits in device
     * memory at two bytes a voxel, not four -- and converted inside the first
     * pass; the inverse writes them from inside its last one. The offset
     * policy (<rapidht/integer_offset.h>) decides the integer the passes work
     * around; it changes rounding only, never the result's meaning.
     *
     * The DeviceVolume overloads need Modes::GPUFactored. The host-pointer
     * overloads work on every mode: on Modes::GPUFactored they upload the
     * integers as integers; on the others they convert on the host and run
     * the floating-point path, with the same offset arithmetic.
     */

    /**
     * @brief The transform of an integer volume on the device into a spectrum.
     *
     * out = H(in), unnormalised, as ForwardTransform(DeviceVolume&) gives for
     * the same values in floating point. `in` is only read. Device memory:
     * the two volumes and nothing else (no scratch volume on this mode).
     *
     * @param in Integer voxels, Width()*Height()*Depth() of them.
     * @param out Receives the spectrum; the same size.
     * @param offset Integer worked around; the default samples every 64th line.
     * @throws std::invalid_argument Unless the mode is Modes::GPUFactored and
     *         both sizes match.
     */
    template <typename U>
    void ForwardTransform(const DeviceVolume<U>& in, DeviceVolume<T>& out,
        IntegerOffset offset = IntegerOffset::Default());

    /**
     * @brief The separable transform (ForwardSeparable) of an integer volume
     *        on the device. Arguments and requirements as above.
     */
    template <typename U>
    void ForwardSeparable(const DeviceVolume<U>& in, DeviceVolume<T>& out,
        IntegerOffset offset = IntegerOffset::Default());

    /**
     * @brief The inverse of a spectrum on the device into integer voxels.
     *
     * Each voxel is rint(x / N) of the inverse transform, clamped to the range
     * of U. A spectrum of integers comes back exactly: in float, the forward
     * transform with the default policy and this inverse with MeanOfAll()
     * restore every voxel of the uint8 and uint16 volumes tested, 128^3 to
     * 1024^3. An offset far from the mean (NoOffset() on a uint16 tomogram)
     * can miss voxels by one.
     *
     * @param in The spectrum. Used as work space: it does not hold the
     *        spectrum afterwards.
     * @param out Receives the voxels; the same size.
     * @param offset Integer worked around. MeanOfAll() (the default) and
     *        MeanOfEvery() read it off H(0) as rint(H(0) / N), at no cost.
     * @param saturated If not null, receives the number of voxels clamped to
     *        the range of U (NaN counts, and gives 0).
     * @throws std::invalid_argument Unless the mode is Modes::GPUFactored and
     *         both sizes match.
     */
    template <typename U>
    void InverseTransform(DeviceVolume<T>& in, DeviceVolume<U>& out,
        IntegerOffset offset = IntegerOffset::MeanOfAll(), size_t* saturated = nullptr);

    /**
     * @brief The transform of integer voxels in host memory into a spectrum
     *        in host memory. Every mode.
     * @param in Integer voxels, Width()*Height()*Depth() of them; only read.
     * @param out Receives the spectrum; the same count, not overlapping `in`.
     * @param offset As for the DeviceVolume overload.
     */
    template <typename U>
    void ForwardTransform(const U* in, T* out, IntegerOffset offset = IntegerOffset::Default());

    /// The separable transform of integer voxels in host memory. Every mode.
    template <typename U>
    void ForwardSeparable(const U* in, T* out, IntegerOffset offset = IntegerOffset::Default());

    /**
     * @brief The inverse of a spectrum in host memory into integer voxels in
     *        host memory. Every mode; `in` is only read.
     *
     * Rounding, clamping, `offset` and `saturated` as for the DeviceVolume
     * overload.
     */
    template <typename U>
    void InverseTransform(const T* in, U* out, IntegerOffset offset = IntegerOffset::MeanOfAll(),
        size_t* saturated = nullptr);

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
    void DHT2DCuda(T* image, bool separable = false);

    /**
     * @brief Performs a 3D Hartley Transform using CUDA matrix-matrix multiplication.
     * @param image Pointer to the input/output 3D data array.
     */
    void DHT3DCuda(T* data, bool separable = false);

    /*
     * The device-side halves of the three above, working on memory that is
     * already there. The host-pointer versions are now a copy in, one of
     * these, and a copy out; the resident overloads call them directly.
     *
     * Each leaves its result in `deviceInOut` and may use `deviceScratch`,
     * which must hold at least as many elements. `separable` skips the
     * Bracewell correction (ForwardSeparable); in 1D there is none to skip.
     */
    void DHT1DOnDevice(T* deviceInOut, T* deviceScratch);
    void DHT2DOnDevice(T* deviceInOut, T* deviceScratch, bool separable = false);
    void DHT3DOnDevice(T* deviceInOut, T* deviceScratch, bool separable = false);

    /// Dispatches to the right rank, and rejects anything but Modes::GPU.
    void TransformOnDevice(T* deviceInOut, T* deviceScratch, bool separable = false);

    /// Rejects a DeviceVolume this object cannot transform: wrong mode, wrong
    /// size or no allocation. Shared by the resident overloads.
    void CheckResidentVolume(const DeviceVolume<T>& volume) const;

    /// True for both device modes, Modes::GPU and Modes::GPUFactored.
    bool IsDeviceMode() const noexcept { return _mode == Modes::GPU || _mode == Modes::GPUFactored; }

    /// Modes::GPUFactored (R-4): every present axis by the factored kernel,
    /// then, unless `separable`, the Bracewell correction -- all in place on
    /// device memory.
    void FactoredOnDevice(T* deviceData, bool separable);

    /// The host-pointer version: staging upload, FactoredOnDevice, download.
    void FactoredOnHost(T* hostData, bool separable);

    /// The passes of FactoredOnDevice with the first present one reading
    /// `in` (TIn = T: the same pointer as `data`, in place, as before R-5,
    /// `offset` null; TIn = std::uint8_t / std::uint16_t: integers minus the
    /// offset the policy gives, not written) and the rest in place on `data`.
    /// Defined in backend_cuda.cpp.
    template <typename TIn>
    void FactoredPasses(const TIn* in, T* data, const IntegerOffset* offset, bool separable);

    /**
     * R-5: the raw-pointer form of ForwardTransform(const DeviceVolume<U>&,
     * ...), for internal::DeviceAccess (benchmarks, tests): the
     * factored transform of an integer volume already on the device, TIn =
     * std::uint8_t or std::uint16_t, into `out`. The integer `offset` gives
     * (a value or a mean, computed on the device) is subtracted from every
     * voxel as the first pass reads it and returned to the zero-frequency
     * coefficient at the end, so `out` holds the transform of the input
     * itself. Modes::GPUFactored only; reached through internal::DeviceAccess.
     */
    template <typename TIn>
    void FactoredFromIntegers(
        const TIn* deviceIn, DeviceVolume<T>& out, const IntegerOffset& offset, bool separable);

    /// R-5 step 4.2: the passes of the inverse of an integer volume -- the
    /// offset taken out of H(0), the passes in place on `data`, the last one
    /// writing TOut to `out`. Defined in backend_cuda.cpp.
    template <typename TOut>
    void FactoredInversePasses(
        T* data, TOut* out, const IntegerOffset& offset, size_t* saturated, bool separable);

    /**
     * R-5 step 4.2, the raw-pointer form of InverseTransform(DeviceVolume<T>&,
     * DeviceVolume<U>&, ...), also separable: the inverse transform of the
     * spectrum in `spectrum` into integers, TOut = std::uint8_t or
     * std::uint16_t, written to `deviceOut`: each voxel rint(x / N + offset)
     * clamped to the type. The offset (a value, or the rounded mean read off
     * H(0) for the mean policies) is taken out of H(0) first and added back
     * as the voxels are written. `spectrum` is used as the work buffer and
     * does not hold the spectrum afterwards. *saturated, if given, receives
     * the number of clamped voxels. Modes::GPUFactored only.
     */
    template <typename TOut>
    void FactoredToIntegers(DeviceVolume<T>& spectrum, TOut* deviceOut, const IntegerOffset& offset,
        size_t* saturated, bool separable);

    /// The checks of the integer overloads: the mode (`device`: Modes::GPUFactored
    /// only), the sizes, the pointers. `what` names the entry point.
    void CheckIntegerCall(const char* what, size_t inSize, const void* inData, size_t outSize,
        const void* outData, bool device) const;

    /// The host-pointer integer overloads, for every mode.
    template <typename U>
    void ForwardFromIntegersOnHost(const U* in, T* out, const IntegerOffset& offset, bool separable);
    template <typename U>
    void InverseToIntegersOnHost(const T* in, U* out, const IntegerOffset& offset, size_t* saturated);

    /**
     * R-6 step 1.2: circular convolution on the device, in place on `data`,
     * with `spectrum` = H(g) / N (same extent). `pair`: any kernel; otherwise
     * g is even along every axis. The passes z, y, then the fused x pass
     * (transform, product, transform), then y, z -- five over the volume.
     * With `prepared`, `data` is first set to a copy of *prepared, which
     * FactoredConvolvePrepare has already taken through z and y (a bank of
     * kernels shares those two passes). Where the fused pass does not apply
     * (double, short x) the forward transform is completed, the product
     * taken over the whole spectrum, and the transform run again.
     * Modes::GPUFactored only; reached through internal::DeviceAccess.
     */
    void FactoredConvolve(DeviceVolume<T>& data, const DeviceVolume<T>& spectrum, bool pair,
        const DeviceVolume<T>* prepared);

    /// R-6 step 1.2: the z and y passes of FactoredConvolve, in place.
    void FactoredConvolvePrepare(DeviceVolume<T>& data);

    /// The passes of FactoredConvolve on raw device pointers (`prepared` may
    /// be null); with `prepareOnly`, only the z and y passes. Defined in
    /// backend_cuda.cpp.
    void FactoredConvolvePasses(T* data, const T* spectrum, bool pair, const T* prepared, bool prepareOnly);

    friend struct internal::DeviceAccess;

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
     * @brief Fast Hartley butterflies on a block of lines held side by side.
     *
     * buffer holds Length(direction) rows of `lanes` values, row idx being
     * element idx of every line in the block. Each butterfly is then a short
     * contiguous loop across the lines, which the compiler vectorises, and the
     * strided gather from the volume happens once per block, not per element.
     * @param buffer Length(direction) * lanes values, transformed in place.
     * @param lanes Number of lines in the block.
     * @param direction Axis whose length, bit reversal and twiddles to use.
     */
    void FDHTLanes(T* buffer, size_t lanes, Direction direction) const;

    /**
     * @brief Throws std::invalid_argument unless every non-zero extent is a
     *        power of two, as the CPU butterflies require.
     *
     * Series1D runs FDHT1D inside an OpenMP parallel region, where an escaping
     * exception ends in std::terminate instead of reaching the caller. Checking
     * all axes here, before the first pass, turns that into an ordinary error
     * and leaves the caller's data untouched.
     * @param caller Name of the public entry point, for the message.
     */
    void RequirePowerOfTwoExtents(const char* caller) const;

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

    std::array<size_t, static_cast<size_t>(Direction::Count)> _dims {};
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
