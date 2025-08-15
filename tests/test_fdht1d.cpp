#include <rapidht.h>
#include <utilities.h>
#include <iostream>
#include <cmath>
#include <numeric>
#include <cstring>

using namespace RapiDHT;

int main(int argc, char** argv) {
	size_t width = static_cast<size_t>(pow(2, 3));
	auto mode = RapiDHT::Modes::GPU;

	// If arguments is parced then exactly one argument is required
	if (argc >= 2) {
		width = std::atoi(argv[1]);

		if (argc >= 3) {
			auto device = argv[2];

			if (!strcmp(device, "CPU")) {
				mode = RapiDHT::Modes::CPU;
			} else if (!strcmp(device, "GPU")) {
				mode = RapiDHT::Modes::GPU;
			} else if (!strcmp(device, "RFFT")) {
				mode = RapiDHT::Modes::RFFT;
			} else {
				std::cerr << "Error: device must be either CPU, GPU or RFFT" << std::endl;
				return 1;
			}
		}
		if (argc >= 4) {
			std::cerr << "Usage: " << argv[0] << " rows" << std::endl;
			return 1;
		}
	}

	// ---- Создание данных ----
	auto original_data = make_data<double>({ width });
	auto transformed_data = original_data;

	print_data_1d(original_data.data(), width);

	// ---- Засекаем время ----
	auto start_time = std::chrono::high_resolution_clock::now();

	// ---- Преобразование Хартли ----
	RapiDHT::HartleyTransform ht(width, 0, 0, mode);
	ht.ForwardTransform(transformed_data.data());
	ht.InverseTransform(transformed_data.data());

	print_data_1d(transformed_data.data(), width);

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end_time - start_time;
	show_time(0, elapsed.count(), "Common time");

	// ---- Подсчёт ошибки ----
	double sum_sqr = std::transform_reduce(
		transformed_data.begin(), transformed_data.end(),
		original_data.begin(), 0.0, std::plus<>(),
		[](double x, double y) { return (x - y) * (x - y); }
	);

	std::cout << "Error:\t" << std::sqrt(sum_sqr) << std::endl;
	return 0;
}
