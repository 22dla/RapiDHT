/*
 * Project: RapiDHT
 * File: utilities.h
 * Brief: Вспомогательные утилиты: профилирование, генерация данных, парсинг аргументов.
 * Author: Волков Евгений Александрович, volkov22dla@yandex.ru
 */

#ifndef UTILITIES_H
#define UTILITIES_H

#include "rapidht.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace RapiDHT {

#ifdef DEBUG
#define PROFILE_FUNCTION() Profiler __profiler(__FUNCTION__)
#else
#define PROFILE_FUNCTION()
#endif

class Profiler {
public:
    Profiler(const std::string& functionName):
        _functionName(functionName), _startTime(std::chrono::high_resolution_clock::now()) { }
    ~Profiler()
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - _startTime).count();

        // Человекочитаемые единицы времени
        std::string unit = "mics";
        double duration = static_cast<double>(duration_us);
        if (duration > 1000.0) {
            duration /= 1000.0;
            unit = "ms";
        }
        if (duration > 1000.0) {
            duration /= 1000.0;
            unit = "s";
        }

        // Форматированный вывод: время и имя функции
        std::cout << std::setw(10) << std::fixed << std::setprecision(3) << duration << " " << unit
                  << "\t|\t"
                  << std::setw(30) << std::left << _functionName
                  << std::endl;
    }

private:
    std::string _functionName;
    std::chrono::high_resolution_clock::time_point _startTime;
};

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
    Random,
    Sequential
};
/**
 * @brief Генерирует массив данных с указанными размерами и записывает его в вектор.
 *
 * Режимы заполнения:
 *  - FillMode::Random     — случайные значения в диапазоне [0, 255].
 *  - FillMode::Sequential — последовательные значения, начиная с 0.
 *
 * @tparam T тип элементов массива (например, int, float, double).
 * @param sizes список размеров по каждому измерению.
 * @param mode  режим заполнения (по умолчанию FillMode::Random).
 * @return std::vector<T> вектор со сгенерированными данными, длиной равной произведению размеров.
 *
 * @throws std::invalid_argument если sizes пуст или содержит нулевое измерение.
 * @throws std::overflow_error  если произведение размеров превышает максимальный размер size_t.
 */
template <typename T>
std::vector<T> MakeData(std::initializer_list<size_t> sizes, FillMode mode = FillMode::Random)
{
    if (sizes.size() == 0) {
        throw std::invalid_argument("Sizes list cannot be empty");
    }

    // Общий объём данных = произведение размеров
    size_t total_size = std::accumulate(sizes.begin(), sizes.end(), size_t { 1 },
        [](size_t acc, size_t val) {
            if (val == 0)
                throw std::invalid_argument("Dimension size cannot be zero");
            if (acc > SIZE_MAX / val)
                throw std::overflow_error("Size overflow");
            return acc * val;
        });

    std::vector<T> data(total_size);

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
 * @brief Печатает одномерный массив с фиксированной точностью.
 *
 * @tparam T тип данных массива.
 * @param data указатель на массив данных.
 * @param length длина массива.
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
 * @brief Печатает двумерный массив в табличном виде.
 *
 * @tparam T тип данных массива.
 * @param data указатель на массив (row-major).
 * @param width количество столбцов.
 * @param height количество строк.
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

template <typename T>
void PrintData3d(const T* data, int width, int height, int depth, int width_max = std::numeric_limits<int>::max(),
    int height_max = std::numeric_limits<int>::max(), int depth_max = std::numeric_limits<int>::max())
{
    auto N = (width_max < width) ? width_max : width;
    auto M = (height_max < height) ? height_max : height;
    auto L = (depth_max < depth) ? depth_max : depth;

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

template <typename T>
void PrintData3dColumnMajor(const T* data, int width, int height, int depth)
{
    for (int l = 0; l < depth; ++l) {
        std::cout << "Layer " << l << ":\n";
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                // Правильный индекс для column-major хранения
                int idx = i + j * height + l * width * height;
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << data[idx] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

/**
 * @brief Записывает матрицу в CSV-файл.
 *
 * @tparam T тип элементов матрицы.
 * @param matrix указатель на данные матрицы (row-major).
 * @param width количество столбцов.
 * @param height количество строк.
 * @param file_path путь к CSV-файлу для записи.
 * @throws std::runtime_error если файл не удалось открыть для записи.
 */
template <typename T>
void WriteMatrixToCsv(const T* matrix, const size_t width,
    const size_t height, const std::string& file_path)
{
    std::ofstream output_file(file_path);
    if (!output_file) {
        throw std::runtime_error("Failed to open file for writing");
    }

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            output_file << matrix[i * width + j];
            if (j < width - 1)
                output_file << ";";
        }
        output_file << "\n";
    }
}

/**
 * @brief Генерирует трёхмерный вектор в формате std::vector<std::vector<std::vector<T>>>
 * с синтетическими данными для тестов.
 *
 * @tparam T тип данных.
 * @param n размер по второй размерности.
 * @param m размер по третьей размерности.
 * @param l размер по первой размерности.
 * @return std::vector<std::vector<std::vector<T>>> сгенерированные данные.
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
 * @brief Выводит в консоль разницу между двумя отметками времени с сообщением.
 *
 * @param start время начала.
 * @param finishTime время окончания.
 * @param finish поясняющее сообщение.
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

// Утилита для сравнения массивов с выводом метрик
template <typename T>
void CompareData(const std::vector<T>& original, const std::vector<T>& transformed, double tolerance = 1e-9)
{
    if (original.size() != transformed.size()) {
        std::cerr << "Error: sizes differ!" << std::endl;
        return;
    }

    // Метрики отклонения
    double max_diff = 0.0;
    double l2_norm = 0.0;

    for (size_t i = 0; i < original.size(); ++i) {
        double diff = std::abs(original[i] - transformed[i]);
        max_diff = std::max(max_diff, diff);
        l2_norm += diff * diff;
    }

    l2_norm = std::sqrt(l2_norm / original.size());

    std::cout << "Max difference: " << max_diff << std::endl;
    std::cout << "L2 norm of difference: " << l2_norm << std::endl;

    if (max_diff < tolerance) {
        std::cout << "Transform verified: data matches within tolerance." << std::endl;
    } else {
        std::cout << "Transform mismatch: data differs beyond tolerance." << std::endl;
    }
}

// Разбор строки размеров вида "NxM[xK]"
std::vector<size_t> ParseDims(const std::string& str);

// Разбор типа устройства выполнения
Modes ParseDevice(const char* device);

LoadingConfig ParseArgs(int argc, char** argv);

} // namespace RapiDHT

#endif // !UTILITIES_H
