/*
 * Project: RapiDHT
 * File: rapidht.h
 * Brief: Публичный API дискретного преобразования Хартли (1D/2D/3D), CPU/GPU режимы.
 * Author: Волков Евгений Александрович, volков22dla@yandex.ru
 */

#ifndef RAPIDHT_H
#define RAPIDHT_H

#include "dev_array.h"
#include <array>
#include <vector>

namespace RapiDHT {
enum class Direction : size_t { Y = 0,
    X = 1,
    Z = 2,
    Count };
enum class Modes { CPU,
    GPU,
    RFFT };

template <typename T>
class HartleyTransform {
public:
    HartleyTransform() = delete;

    /**
     * @brief Constructs a HartleyTransform object with specified dimensions and mode.
     * @param width Width of the 3D data.
     * @param height Height of the 3D data.
     * @param depth Depth of the 3D data.
     * @param mode Transformation mode (CPU, GPU, RFFT).
     */
    HartleyTransform(size_t width, size_t height, size_t depth, Modes mode);

    /**
     * @brief Performs the forward Hartley transform on the input data.
     * @param data Pointer to the input/output data array.
     */
    void ForwardTransform(T* data);

    /**
     * @brief Performs the inverse Hartley transform on the input data.
     * @param data Pointer to the input/output data array.
     */
    void InverseTransform(T* data);
    static void GpuMatrixMultiply111(T* A, T* B, T* C, int n);
    static void GpuMatrixMultiplyInt(const uint8_t* A, const uint8_t* B, uint32_t* C, int n);

    constexpr size_t Width() const noexcept { return _dims[static_cast<size_t>(Direction::Y)]; }
    constexpr size_t Height() const noexcept { return _dims[static_cast<size_t>(Direction::X)]; }
    constexpr size_t Depth() const noexcept { return _dims[static_cast<size_t>(Direction::Z)]; }

    // Функция для получения линейного индекса в зависимости от направления
    size_t LinearIndex(size_t i, size_t j, size_t k) const
    {
        return i + Width() * (j + Height() * k); // row-major
    }

    // Функция для получения индекса вдоль конкретной оси
    size_t AxisIndex(size_t idx_along_axis, size_t fixed1, size_t fixed2, Direction direction) const
    {
        switch (direction) {
            case Direction::Y:
                return LinearIndex(idx_along_axis, fixed1, fixed2);
            case Direction::X:
                return LinearIndex(fixed1, idx_along_axis, fixed2);
            case Direction::Z:
                return LinearIndex(fixed1, fixed2, idx_along_axis);
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

    // static void Process3DDataWithHartley(std::vector<float>& h_data, int N);

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

    /**
     * @brief Performs a 1D Hartley Transform using CUDA matrix-vector multiplication.
     * @param hX Pointer to the input data vector.
     * @param length Length of the vector.
     */
    void DHT1DCuda(T* hX);

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
     * @brief Initializes the kernel used for 1D Hartley transform.
     * @param kernel Pointer to the kernel vector to initialize.
     * @param height Height of the 1D transform.
     */
    [[deprecated]] static void InitializeKernelHost(std::vector<T>& kernel, size_t height);

    /**
     * @brief Computes the 1D Hartley transform using a given kernel.
     * @param a Input vector.
     * @param kernel Precomputed kernel vector.
     * @return Transformed 1D vector.
     */
    static std::vector<T> DHT1D(const std::vector<T>& a, const std::vector<T>& kernel);

    /**
     * @brief Transposes a 2D vector.
     * @tparam T Type of elements in the vector.
     * @param image Pointer to the 2D vector to transpose.
     */
    static void Transpose(std::vector<std::vector<T>>& image);

    /**
     * @brief Performs a simple transpose of a 2D array in place.
     * @param image Pointer to the 2D array.
     * @param width Width of the array.
     * @param height Height of the array.
     */
    static void TransposeSimple(T* image, size_t width, size_t height);

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

    std::array<size_t, static_cast<size_t>(Direction::Count)> _dims {};
    std::array<std::vector<size_t>, static_cast<size_t>(Direction::Count)> _bitReversedIndices;

    Modes _mode = Modes::CPU;

    std::array<std::vector<T>, static_cast<size_t>(Direction::Count)> _hTransformMatrices;
    std::array<dev_array<T>, static_cast<size_t>(Direction::Count)> _dTransformMatrices;
};
} // namespace RapiDHT

#endif // RAPIDHT_H
