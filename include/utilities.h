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

#ifdef DEBUG
#define PROFILE_FUNCTION() Profiler __profiler(__FUNCTION__)
#else
#define PROFILE_FUNCTION()
#endif

class Profiler
{
public:
	Profiler(const std::string& functionName) :
		m_functionName(functionName), m_startTime(std::chrono::high_resolution_clock::now()) {}
	~Profiler()
	{
		auto endTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - m_startTime).count();
		std::cout << m_functionName << " took " << duration << " microseconds" << std::endl;
	}
private:
	std::string m_functionName;
	std::chrono::high_resolution_clock::time_point m_startTime;
};

/**
 * @brief ������� ������ ������ � ���������� ��������� � ��������� ���������� �������.
 *
 * @tparam T ��� ������ (��������, int, double).
 * @param sizes ������ �������� �� ������ �����������.
 * @return std::vector<T> ������ ������, ������ �������� ����� ������������ ���� ��������.
 * @throws std::invalid_argument ���� sizes ���� ��� ���� �� �������� ����� 0.
 * @throws std::overflow_error ���� ������������ �������� ��������� ������ size_t.
 */
template <typename T>
std::vector<T> make_data(std::initializer_list<size_t> sizes)
{
	if (sizes.size() == 0) {
		throw std::invalid_argument("Sizes list cannot be empty");
	}

	// ��������� ����� ������, �������� �� ������������
	size_t total_size = std::accumulate(sizes.begin(), sizes.end(), size_t{ 1 },
		[](size_t acc, size_t val) {
		if (val == 0) throw std::invalid_argument("Dimension size cannot be zero");
		if (acc > SIZE_MAX / val) throw std::overflow_error("Size overflow");
		return acc * val;
	});

	std::vector<T> data(total_size);

	// ����������� ��������� ��������� �����
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(0, 255);

	// ������������ ����������
	std::for_each(std::execution::par, data.begin(), data.end(),
		[&](T& x) { x = static_cast<T>(dist(gen)); });

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
void print_data_1d(const T* data, int length)
{
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
 * @param rows ���������� �����.
 * @param cols ���������� ��������.
 */
template<typename T>
void print_data_2d(const T* data, int rows, int cols)
{
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			std::cout << std::fixed << std::setprecision(2) << data[i * cols + j] << " ";
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
 * @param rows ���������� �����.
 * @param cols ���������� ��������.
 * @param file_path ���� � CSV ����� ��� ������.
 * @throws std::runtime_error ���� ���� �� ������� ������� ��� ������.
 */
template<typename T>
void write_matrix_to_csv(const T* matrix, const size_t rows,
	const size_t cols, const std::string& file_path)
{
	std::ofstream output_file(file_path);
	if (!output_file) {
		throw std::runtime_error("Failed to open file for writing");
	}

	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			output_file << matrix[i * cols + j];
			if (j < cols - 1) output_file << ";";
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
std::vector<std::vector<std::vector<T>>> make_data_3d_vec_vec_vec(int n, int m, int l)
{
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
inline void show_time(double startTime, double finishTime, std::string message)
{
	std::cout << message << ":\t" << finishTime - startTime << " sec" << std::endl;
}

#endif // !UTILITIES_H