#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <utilities.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <filesystem>

// Функция для загрузки изображений и создания 3D данных
std::vector<std::vector<std::vector<uint8_t>>> loadImagesTo3DArray(const std::string& folderPath) {
	std::vector<cv::Mat> images;
	for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
		if (entry.is_regular_file() && (entry.path().extension() == ".jpg"
			|| entry.path().extension() == ".png"
			|| entry.path().extension() == ".bmp")) {
			cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

			if (img.empty()) {
				std::cerr << "Error loading: " << entry.path().string() << std::endl;
				continue;
			}

			images.push_back(img);
		}
	}

	if (images.empty()) {
		std::cerr << "No images for loading." << std::endl;
		return {};
	}

	int numImages = images.size();
	int height = images[0].rows;
	int width = images[0].cols;

	std::vector<std::vector<std::vector<uint8_t>>> image3D(numImages, std::vector<std::vector<uint8_t>>(height, std::vector<uint8_t>(width)));
	for (int i = 0; i < numImages; ++i) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				image3D[i][y][x] = images[i].at<uint8_t>(y, x);
			}
		}
	}

	return image3D;
}

template<typename T>
void printData1D(const T* data, int length) {

	for (size_t idx = 0; idx < length; ++idx) {
		std::cout << std::fixed << std::setprecision(2) << data[idx] << "\t";
	}
	std::cout << std::endl;
}

template<typename T>
void printData2D(const T* data, int _rows, int _cols) {

	for (size_t i = 0; i < _rows; ++i) {
		for (size_t j = 0; j < _cols; ++j) {
			std::cout << std::fixed << std::setprecision(2) << data[i * _cols + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << std::endl;
}

template<typename T>
void writeMatrixToCSV(const T* matrix, const size_t rows,
	const size_t cols, const std::string& file_path) {
	std::ofstream output_file(file_path);
	if (!output_file) {
		throw std::runtime_error("Failed to open file for writing");
	}

	// Write matrix elements to file
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			//output_file << std::fixed << std::setprecision(2) << matrix[i * cols + j];
			output_file << matrix[i * cols + j];

			if (j < cols - 1) {
				output_file << ";";
			}
		}
		output_file << "\n";
	}

	output_file.close();
}

template <typename T>
std::vector<std::vector<std::vector<T>>> makeData3DArray(
	int n, int m, int l) {
	const double kPi = std::acos(-1);
	std::vector<std::vector<std::vector<T>>> data(l);

	for (size_t j1 = 0; j1 < l; ++j1) {
		data[j1].resize(n);
		for (size_t j2 = 0; j2 < n; ++j2) {
			data[j1][j2].resize(m);
			for (size_t j3 = 0; j3 < m; ++j3) {
				data[j1][j2][j3] = static_cast<T>(n + std::cos(j1 / kPi)
					- std::sin(std::cos(j2)) + std::tan(j3) + 2 + l) / m;
			}
		}
	}
	return data;
}

template <typename T>
std::vector<T> makeData(std::initializer_list<size_t> sizes) {
	PROFILE_FUNCTION();
	size_t num_dims = sizes.size();
	std::vector<size_t> dim_sizes(sizes);
	for (size_t i = 0; i < num_dims; ++i) {
		if (dim_sizes[i] < 0) {
			throw std::invalid_argument("Invalid size");
		}
	}
	std::vector<T> data(1);
	for (int i = 0; i < num_dims; ++i) {
		data.resize(data.size() * dim_sizes[i]);
	}
	// fill massive with random values
#pragma omp parallel for
	for (int idx = 0; idx < data.size(); ++idx) {
		/*data[idx] = static_cast<T>(static_cast<double>(dim_sizes[0]) + std::cos(std::asin(0.1) / (static_cast<double>(idx) + 1)) -
			std::sin(std::cos(static_cast<double>(idx) / static_cast<double>(dim_sizes[0]))) +
			std::tan(static_cast<double>(idx * dim_sizes[0])) + static_cast<double>(2 + idx)) / static_cast<double>((dim_sizes[0] * dim_sizes[0]));*/
		data[idx] = rand() % 256; // Случайное число от 0 до 255
	}
	//std::iota(data.begin(), data.end(), 0);

	return data;
}

void showTime(double startTime, double finishTime, std::string message) {
	std::cout << message + ":\t" << finishTime - startTime << " sec" << std::endl;
}

std::optional<RapiDHT::Modes> parseMode(const std::map<std::string, std::string>& args) {
	auto getMode = [](const std::string& device) {
		if (device == "CPU") {
			return RapiDHT::Modes::CPU;
		}
		else if (device == "GPU") {
			return RapiDHT::Modes::GPU;
		}
		else if (device == "RFFT") {
			return RapiDHT::Modes::RFFT;
		}
		else {
			throw std::invalid_argument("Error: device must be either CPU, GPU or RFFT");
		}
	};

	auto it = args.find("--mode");
	if (it != args.end()) {
		auto mode = getMode(it->second);
		return mode;
	}
	return {};
}

void printUsage(const std::string& programName) {
	std::cout << "Usage: " << programName
		<< " --rows <number> --mode <CPU|GPU|RFFT> [--cols <number>] [--depth <number>] [...]"
		<< std::endl;
}

std::map<std::string, std::string> parseCommandLine(int argc, char* argv[]) {
	std::map<std::string, std::string> args;
	for (int i = 1; i < argc - 1; i += 2) {
		args[argv[i]] = argv[i + 1];
	}
	return args;
}

std::optional<size_t> parseSize(const std::map<std::string, std::string>& args, const std::string& arg) {

	auto getSize = [](const std::string& str) {
		return std::stoul(str);
	};

	auto it = args.find(arg);
	if (it != args.end()) {
		auto arg_size_t = getSize(it->second);
		return arg_size_t;
	}
	return {};
}


template void printData1D(const int* data, int length);
template void printData1D(const double* data, int length);
template void printData2D(const int* data, int rows, int cols);
template void printData2D(const double* data, int rows, int cols);
template void writeMatrixToCSV(const double* matrix, const size_t rows,
	const size_t cols, const std::string& file_path);
template void writeMatrixToCSV(const int* matrix, const size_t rows,
	const size_t cols, const std::string& file_path);
template std::vector<int> makeData(std::initializer_list<size_t> sizes);
template std::vector<double> makeData(std::initializer_list<size_t> size);
template std::vector<std::vector<std::vector<int>>> makeData3DArray(int n, int m, int l);
template std::vector<std::vector<std::vector<double>>> makeData3DArray(int n, int m, int l);
