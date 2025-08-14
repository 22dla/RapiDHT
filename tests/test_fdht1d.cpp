#include <rapidht.h>
#include <utilities.h>
#include <iostream>
#include <cmath>
#include <numeric>
#include <cstring>

int main(int argc, char** argv) {
	size_t rows = static_cast<size_t>(pow(2, 14));
	RapiDHT::Modes mode = RapiDHT::GPU;

	// If arguments is parced then exactly one argument is required
	if (argc >= 2) {
		rows = std::atoi(argv[1]);

		if (argc >= 3) {
			auto device = argv[2];

			if (!strcmp(device, "CPU")) {
				mode = RapiDHT::CPU;
			} else if (!strcmp(device, "GPU")) {
				mode = RapiDHT::GPU;
			} else if (!strcmp(device, "RFFT")) {
				mode = RapiDHT::RFFT;
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
	auto original_data = make_data<double>({ rows });
	auto transformed_data = original_data;

	//print_data_1d(original_data.data(), rows);

	// ---- Засекаем время ----
	auto start_time = std::chrono::high_resolution_clock::now();

	// ---- Преобразование Хартли ----
	RapiDHT::HartleyTransform ht(rows, 0, 0, mode);
	ht.ForwardTransform(transformed_data.data());
	ht.InverseTransform(transformed_data.data());

	//print_data_1d(transformed_data.data(), rows);

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
