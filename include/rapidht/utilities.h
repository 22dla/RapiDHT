/*
 * Project: RapiDHT
 * File: include/rapidht/utilities.h
 * Brief: Helpers around the transform: profiling, test data, argument parsing.
 * Author: Volkov Evgeny Aleksandrovich, volkov22dla@yandex.ru
 *
 * Nothing here is needed to use the library; these are the conveniences the
 * examples, tests and benchmarks share.
 */

#ifndef RAPIDHT_UTILITIES_H
#define RAPIDHT_UTILITIES_H

#include <rapidht/transform.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint> // SIZE_MAX
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits> // std::numeric_limits
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace RapiDHT {

/**
 * @brief Times the enclosing function and prints the result on the way out.
 *
 * Expands to nothing unless the library was configured with
 * -DENABLE_PROFILING=ON, so the instrumentation can stay in the sources
 * without costing a release build anything.
 */
#ifdef DEBUG
#define PROFILE_FUNCTION() Profiler __profiler(__FUNCTION__)
#else
#define PROFILE_FUNCTION()
#endif

/// Scope timer behind PROFILE_FUNCTION. Prints on destruction.
class Profiler {
public:
    Profiler(const std::string& functionName):
        _functionName(functionName), _startTime(std::chrono::high_resolution_clock::now()) { }

    ~Profiler()
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto microseconds
            = std::chrono::duration_cast<std::chrono::microseconds>(endTime - _startTime).count();

        // Scale to whichever unit keeps the number readable.
        std::string unit = "mics";
        double duration = static_cast<double>(microseconds);
        if (duration > 1000.0) {
            duration /= 1000.0;
            unit = "ms";
        }
        if (duration > 1000.0) {
            duration /= 1000.0;
            unit = "s";
        }

        std::cout << std::setw(10) << std::fixed << std::setprecision(3) << duration << " " << unit
                  << "\t|\t"
                  << std::setw(30) << std::left << _functionName
                  << std::endl;
    }

private:
    std::string _functionName;
    std::chrono::high_resolution_clock::time_point _startTime;
};

/// Extent and backend an example or benchmark was asked to run, as produced by
/// ParseArgs.
struct LoadingConfig {
    size_t width = 1 << 3;
    size_t height = 1;
    size_t depth = 1;
    Modes mode = Modes::CPU;

    void print() const
    {
        std::cout << "width=" << width
                  << " height=" << height
                  << " depth=" << depth
                  << " mode="
                  << (mode == RapiDHT::Modes::CPU ? "CPU" : mode == RapiDHT::Modes::GPU ? "GPU"
                                                                                        : "RFFT")
                  << std::endl;
    }
};

enum class FillMode {
    Random, ///< Uniform integers in [0, 255], cast to T.
    Sequential ///< 0, 1, 2, ... in storage order.
};

/**
 * @brief Builds a flat array of the given extent, filled for testing.
 *
 * @tparam T Element type.
 * @param sizes Extent along each dimension.
 * @param mode How to fill it; random by default.
 * @return A vector whose length is the product of `sizes`.
 *
 * @throws std::invalid_argument If `sizes` is empty or any extent is zero.
 * @throws std::overflow_error If the product does not fit in a size_t.
 */
template <typename T>
std::vector<T> MakeData(std::initializer_list<size_t> sizes, FillMode mode = FillMode::Random)
{
    if (sizes.size() == 0) {
        throw std::invalid_argument("Sizes list cannot be empty");
    }

    size_t totalSize = std::accumulate(sizes.begin(), sizes.end(), size_t { 1 },
        [](size_t acc, size_t val) {
            if (val == 0)
                throw std::invalid_argument("Dimension size cannot be zero");
            // Checked before multiplying rather than after: the overflowed
            // product would be a perfectly ordinary-looking number.
            if (acc > SIZE_MAX / val)
                throw std::overflow_error("Size overflow");
            return acc * val;
        });

    std::vector<T> data(totalSize);

    if (mode == FillMode::Random) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, 255);

        std::for_each(data.begin(), data.end(),
            [&](T& x) { x = static_cast<T>(dist(gen)); });
    } else if (mode == FillMode::Sequential) {
        std::iota(data.begin(), data.end(), T { 0 });
    }

    return data;
}

/**
 * @brief Prints a 1D array to standard output.
 * @param data Array to print.
 * @param length Number of elements.
 */
template <typename T>
void PrintData1d(const T* data, int length)
{
    for (int idx = 0; idx < length; ++idx) {
        std::cout << std::fixed << std::setprecision(2) << data[idx] << "\t";
    }
    std::cout << std::endl;
}

/**
 * @brief Prints a 2D array as a table.
 * @param data Row-major array.
 * @param width Number of columns.
 * @param height Number of rows.
 */
template <typename T>
void PrintData2d(const T* data, int width, int height)
{
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << data[i * width + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Prints a 3D array one layer at a time.
 *
 * The three `*Max` arguments cap how much of each axis is shown, so a corner of
 * a large volume can be inspected without flooding the terminal.
 *
 * @param data Row-major volume, index = j + width * (i + height * l).
 */
template <typename T>
void PrintData3d(const T* data, int width, int height, int depth,
    int widthMax = std::numeric_limits<int>::max(),
    int heightMax = std::numeric_limits<int>::max(),
    int depthMax = std::numeric_limits<int>::max())
{
    auto N = (widthMax < width) ? widthMax : width;
    auto M = (heightMax < height) ? heightMax : height;
    auto L = (depthMax < depth) ? depthMax : depth;

    for (int l = 0; l < L; ++l) {
        std::cout << "Layer " << l << ":\n";
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                int idx = l * width * height + i * width + j;
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << data[idx] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

/**
 * @brief The same, for a volume in column-major order.
 *
 * Useful while debugging the GPU path, which hands cuBLAS column-major slices.
 */
template <typename T>
void PrintData3dColumnMajor(const T* data, int width, int height, int depth)
{
    for (int l = 0; l < depth; ++l) {
        std::cout << "Layer " << l << ":\n";
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int idx = i + j * height + l * width * height;
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << data[idx] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

/**
 * @brief Writes a matrix out as semicolon-separated CSV.
 *
 * @param matrix Row-major matrix.
 * @param width Number of columns.
 * @param height Number of rows.
 * @param filePath Destination, overwritten if it exists.
 * @throws std::runtime_error If the file cannot be opened for writing.
 */
template <typename T>
void WriteMatrixToCsv(const T* matrix, const size_t width,
    const size_t height, const std::string& filePath)
{
    std::ofstream outputFile(filePath);
    if (!outputFile) {
        throw std::runtime_error("Failed to open file for writing");
    }

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            outputFile << matrix[i * width + j];
            if (j < width - 1)
                outputFile << ";";
        }
        outputFile << "\n";
    }
}

/**
 * @brief Builds a nested-vector volume of smooth synthetic values.
 *
 * @param n Extent of the middle dimension.
 * @param m Extent of the innermost dimension.
 * @param l Extent of the outermost dimension.
 */
template <typename T>
std::vector<std::vector<std::vector<T>>> MakeData3dVecVecVec(int n, int m, int l)
{
    const double kPi = std::acos(-1);
    std::vector<std::vector<std::vector<T>>> data(l);

    for (int j1 = 0; j1 < l; ++j1) {
        data[j1].resize(n);
        for (int j2 = 0; j2 < n; ++j2) {
            data[j1][j2].resize(m);
            for (int j3 = 0; j3 < m; ++j3) {
                data[j1][j2][j3] = static_cast<T>(n + std::cos(j1 / kPi) - std::sin(std::cos(j2)) + std::tan(j3) + 2 + l) / m;
            }
        }
    }
    return data;
}

/**
 * @brief Prints the interval between two time points, with a label.
 * @tparam Duration Unit to report in; seconds unless given.
 */
template <typename Duration = std::chrono::seconds>
inline void ShowElapsedTime(const std::chrono::high_resolution_clock::time_point& start,
    const std::chrono::high_resolution_clock::time_point& finish, const std::string& message)
{
    Duration elapsed = std::chrono::duration_cast<Duration>(finish - start);
    std::cout << message << ":\t" << elapsed.count() << " "
              << (std::is_same<Duration, std::chrono::seconds>::value           ? "sec"
                     : std::is_same<Duration, std::chrono::milliseconds>::value ? "ms"
                     : std::is_same<Duration, std::chrono::microseconds>::value ? "us"
                                                                                : "units")
              << std::endl;
}

/**
 * @brief Reports how far two arrays have drifted apart, as max and RMS error.
 *
 * Prints its verdict and returns nothing: this is for the examples, which are
 * read by a person. Tests assert against the reference transform instead.
 *
 * @param tolerance Absolute bound on the largest difference, or 0 to derive
 *        one from T's precision and the scale of the data. The old fixed
 *        default of 1e-9 was an absolute bound on values of any magnitude in
 *        any precision, so a float round trip over a 16^3 volume reported a
 *        mismatch at 5e-4 -- which is simply what float costs there.
 */
template <typename T>
void CompareData(const std::vector<T>& original, const std::vector<T>& transformed, double tolerance = 0.0)
{
    if (original.size() != transformed.size()) {
        std::cerr << "Error: sizes differ!" << std::endl;
        return;
    }

    if (tolerance <= 0.0) {
        double scale = 1.0;
        for (const T& value : original) {
            scale = std::max(scale, static_cast<double>(std::abs(value)));
        }
        tolerance = 100.0 * static_cast<double>(std::numeric_limits<T>::epsilon()) * scale;
    }

    double maxDifference = 0.0;
    double l2Norm = 0.0;

    for (size_t i = 0; i < original.size(); ++i) {
        double difference = std::abs(original[i] - transformed[i]);
        maxDifference = std::max(maxDifference, difference);
        l2Norm += difference * difference;
    }

    l2Norm = std::sqrt(l2Norm / original.size());

    std::cout << "Max difference: " << maxDifference << std::endl;
    std::cout << "L2 norm of difference: " << l2Norm << std::endl;

    if (maxDifference < tolerance) {
        std::cout << "Transform verified: data matches within tolerance." << std::endl;
    } else {
        std::cout << "Transform mismatch: data differs beyond tolerance." << std::endl;
    }
}

/// Parses an extent written as "NxM" or "NxMxK".
std::vector<size_t> ParseDims(const std::string& str);

/// Parses a backend name: "CPU", "GPU" or "RFFT", case-sensitive.
Modes ParseDevice(const char* device);

/**
 * @brief Reads the extent and backend out of a command line.
 *
 * Expects `NxM[xK] [device]`. Throws rather than exiting, so the caller keeps
 * the chance to print its own usage message -- which means it also throws when
 * given no arguments at all, and callers that want defaults should check argc
 * before asking.
 */
LoadingConfig ParseArgs(int argc, char** argv);

} // namespace RapiDHT

#endif // RAPIDHT_UTILITIES_H
