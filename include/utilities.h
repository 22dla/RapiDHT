#ifndef UTILITIES_H
#define UTILITIES_H

#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <execution>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <sstream>
#include "rapidht.h"

namespace RapiDHT {

#ifdef DEBUG
#define PROFILE_FUNCTION() Profiler __profiler(__FUNCTION__)
#else
#define PROFILE_FUNCTION()
#endif

class Profiler {
public:
	Profiler(const std::string& functionName) :
		m_functionName(functionName), m_startTime(std::chrono::high_resolution_clock::now()) {}
	~Profiler() {
		auto endTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - m_startTime).count();
		std::cout << m_functionName << " took " << duration << " microseconds" << std::endl;
	}
private:
	std::string m_functionName;
	std::chrono::high_resolution_clock::time_point m_startTime;
};

struct LoadingConfig {
	size_t width = 1 << 3;
	size_t height = 1;
	size_t depth = 1;
	Modes mode = Modes::CPU;

	void print() const {
		std::cout << "width=" << width
			<< " height=" << height
			<< " depth=" << depth
			<< " mode="
			<< (mode == RapiDHT::Modes::CPU ? "CPU" :
				mode == RapiDHT::Modes::GPU ? "GPU" : "RFFT")
			<< std::endl;
	}
};

enum class FillMode {
	Random,
	Sequential
};
/**
 * @brief Создает вектор данных с указанными размерами и заполняет его в одном из режимов.
 *
 * Режимы заполнения:
 *  - FillMode::Random     — случайные числа в диапазоне [0, 255].
 *  - FillMode::Sequential — последовательные значения, начиная с 0.
 *
 * @tparam T Тип элементов вектора (например, int, float).
 * @param sizes Список размеров по каждой размерности.
 * @param mode  Режим заполнения (по умолчанию FillMode::Random).
 * @return std::vector<T> Вектор данных размером, равным произведению всех размеров.
 *
 * @throws std::invalid_argument Если sizes пуст или один из размеров равен 0.
 * @throws std::overflow_error  Если произведение размеров превышает допустимый размер size_t.
 */
template <typename T>
std::vector<T> MakeData(std::initializer_list<size_t> sizes, FillMode mode = FillMode::Random) {
	if (sizes.size() == 0) {
		throw std::invalid_argument("Sizes list cannot be empty");
	}

	// Вычисляем общий размер
	size_t total_size = std::accumulate(sizes.begin(), sizes.end(), size_t{ 1 },
		[](size_t acc, size_t val) {
		if (val == 0) throw std::invalid_argument("Dimension size cannot be zero");
		if (acc > SIZE_MAX / val) throw std::overflow_error("Size overflow");
		return acc * val;
	});

	std::vector<T> data(total_size);

	if (mode == FillMode::Random) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dist(0, 255);

		std::for_each(std::execution::par, data.begin(), data.end(),
			[&](T& x) { x = static_cast<T>(dist(gen)); });
	} else if (mode == FillMode::Sequential) {
		std::iota(data.begin(), data.end(), T{ 0 });
	}

	return data;
}

/**
 * @brief Выводит одномерный массив в консоль с форматированием.
 *
 * @tparam T Тип данных массива.
 * @param data Указатель на данные массива.
 * @param length Длина массива.
 */
template<typename T>
void PrintData1d(const T* data, int length) {
	for (int idx = 0; idx < length; ++idx) {
		std::cout << std::fixed << std::setprecision(2) << data[idx] << "\t";
	}
	std::cout << std::endl;
}

/**
 * @brief Выводит двумерный массив в консоль с форматированием.
 *
 * @tparam T Тип данных массива.
 * @param data Указатель на данные массива.
 * @param width Количество строк.
 * @param height Количество столбцов.
 */
template<typename T>
void PrintData2d(const T* data, int width, int height) {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			std::cout << std::fixed << std::setprecision(2) << data[i * width + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << std::endl;
}

/**
 * @brief Сохраняет матрицу в CSV файл.
 *
 * @tparam T Тип данных матрицы.
 * @param matrix Указатель на данные матрицы.
 * @param width Количество строк.
 * @param height Количество столбцов.
 * @param file_path Путь к CSV файлу для записи.
 * @throws std::runtime_error если файл не удалось открыть для записи.
 */
template<typename T>
void WriteMatrixToCsv(const T* matrix, const size_t width,
	const size_t height, const std::string& file_path) {
	std::ofstream output_file(file_path);
	if (!output_file) {
		throw std::runtime_error("Failed to open file for writing");
	}

	for (size_t i = 0; i < height; ++i) {
		for (size_t j = 0; j < width; ++j) {
			output_file << matrix[i * width + j];
			if (j < width - 1) output_file << ";";
		}
		output_file << "\n";
	}
}

/**
 * @brief Создает трёхмерный вектор std::vector<std::vector<std::vector<T>>>
 * и заполняет его значениями по формуле.
 *
 * @tparam T Тип данных.
 * @param n Размер второй размерности.
 * @param m Размер третьей размерности.
 * @param l Размер первой размерности.
 * @return std::vector<std::vector<std::vector<T>>> Трёхмерный вектор.
 */
template <typename T>
std::vector<std::vector<std::vector<T>>> MakeData3dVecVecVec(int n, int m, int l) {
	const double kPi = std::acos(-1);
	std::vector<std::vector<std::vector<T>>> data(l);

	for (int j1 = 0; j1 < l; ++j1) {
		data[j1].resize(n);
		for (int j2 = 0; j2 < n; ++j2) {
			data[j1][j2].resize(m);
			for (int j3 = 0; j3 < m; ++j3) {
				data[j1][j2][j3] = static_cast<T>(n + std::cos(j1 / kPi)
					- std::sin(std::cos(j2)) + std::tan(j3) + 2 + l) / m;
			}
		}
	}
	return data;
}

/**
 * @brief Выводит на консоль время выполнения с сообщением.
 *
 * @param startTime Время начала.
 * @param finishTime Время окончания.
 * @param message Сообщение для отображения.
 */
inline void ShowTime(double startTime, double finishTime, std::string message) {
	std::cout << message << ":\t" << finishTime - startTime << " sec" << std::endl;
}

// Функция для проверки совпадения массивов
template <typename T>
void CompareData(const std::vector<T>& original, const std::vector<T>& transformed, double tolerance = 1e-9) {
	if (original.size() != transformed.size()) {
		std::cerr << "Error: sizes differ!" << std::endl;
		return;
	}

	// Считаем максимальное отклонение
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

// Парсинг строки вида "NxM[xK]"
std::vector<size_t> ParseDims(const std::string& str);

// Определение режима вычисления
Modes ParseDevice(const char* device);

LoadingConfig ParseArgs(int argc, char** argv);


} // namespace RapiDHT

#endif // !UTILITIES_H