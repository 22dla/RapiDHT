/*
 * Project: RapiDHT
 * File: utilities.h
 * Brief: Вспомогательные утилиты: профилирование, генерация данных, парсинг аргументов.
 * Author: Волков Евгений Александрович, volkov22dla@yandex.ru
 */

#ifndef UTILITIES_H
#define UTILITIES_H

#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <execution>
#include <fstream>
#include <iomanip>
#include <cmath>
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
		_functionName(functionName), _startTime(std::chrono::high_resolution_clock::now()) {}
	~Profiler() {
		auto endTime = std::chrono::high_resolution_clock::now();
		auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(endTime - _startTime).count();

		// �������������� ����� ������
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

		// ��������������� �����: ������� �����, ����� ��� �������
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
 * @brief ������� ������ ������ � ���������� ��������� � ��������� ��� � ����� �� �������.
 *
 * ������ ����������:
 *  - FillMode::Random     � ��������� ����� � ��������� [0, 255].
 *  - FillMode::Sequential � ���������������� ��������, ������� � 0.
 *
 * @tparam T ��� ��������� ������� (��������, int, float).
 * @param sizes ������ �������� �� ������ �����������.
 * @param mode  ����� ���������� (�� ��������� FillMode::Random).
 * @return std::vector<T> ������ ������ ��������, ������ ������������ ���� ��������.
 *
 * @throws std::invalid_argument ���� sizes ���� ��� ���� �� �������� ����� 0.
 * @throws std::overflow_error  ���� ������������ �������� ��������� ���������� ������ size_t.
 */
template <typename T>
std::vector<T> MakeData(std::initializer_list<size_t> sizes, FillMode mode = FillMode::Random) {
	if (sizes.size() == 0) {
		throw std::invalid_argument("Sizes list cannot be empty");
	}

	// ��������� ����� ������
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
 * @brief ������� ���������� ������ � ������� � ���������������.
 *
 * @tparam T ��� ������ �������.
 * @param data ��������� �� ������ �������.
 * @param length ����� �������.
 */
template<typename T>
void PrintData1d(const T* data, int length) {
	for (int idx = 0; idx < length; ++idx) {
		std::cout << std::fixed << std::setprecision(2) << data[idx] << "\t";
	}
	std::cout << std::endl;
}

/**
 * @brief ������� ��������� ������ � ������� � ���������������.
 *
 * @tparam T ��� ������ �������.
 * @param data ��������� �� ������ �������.
 * @param width ���������� �����.
 * @param height ���������� ��������.
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
 * @brief ��������� ������� � CSV ����.
 *
 * @tparam T ��� ������ �������.
 * @param matrix ��������� �� ������ �������.
 * @param width ���������� �����.
 * @param height ���������� ��������.
 * @param file_path ���� � CSV ����� ��� ������.
 * @throws std::runtime_error ���� ���� �� ������� ������� ��� ������.
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
 * @brief ������� ��������� ������ std::vector<std::vector<std::vector<T>>>
 * � ��������� ��� ���������� �� �������.
 *
 * @tparam T ��� ������.
 * @param n ������ ������ �����������.
 * @param m ������ ������� �����������.
 * @param l ������ ������ �����������.
 * @return std::vector<std::vector<std::vector<T>>> ��������� ������.
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
 * @brief ������� �� ������� ����� ���������� � ����������.
 *
 * @param startTime ����� ������.
 * @param finishTime ����� ���������.
 * @param message ��������� ��� �����������.
 */
inline void ShowTime(double startTime, double finishTime, std::string message) {
	std::cout << message << ":\t" << finishTime - startTime << " sec" << std::endl;
}

// ������� ��� �������� ���������� ��������
template <typename T>
void CompareData(const std::vector<T>& original, const std::vector<T>& transformed, double tolerance = 1e-9) {
	if (original.size() != transformed.size()) {
		std::cerr << "Error: sizes differ!" << std::endl;
		return;
	}

	// ������� ������������ ����������
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

// ������� ������ ���� "NxM[xK]"
std::vector<size_t> ParseDims(const std::string& str);

// ����������� ������ ����������
Modes ParseDevice(const char* device);

LoadingConfig ParseArgs(int argc, char** argv);


} // namespace RapiDHT

#endif // !UTILITIES_H