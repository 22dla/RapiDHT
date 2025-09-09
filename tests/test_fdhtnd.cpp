#include <iostream>
#include "rapidht.h"
#include "utilities.h"
#include <cmath>
#include <cstring>

using namespace RapiDHT;

int main(int argc, char** argv) {
	// Define global 3D array sizes
	size_t width = static_cast<int>(pow(2, 10));
	size_t height = width;
	size_t images_num = 1;
	auto mode = RapiDHT::Modes::CPU;

	// If arguments are parsed then exactly two arguments are required
	if (argc >= 2) {
		if (argc >= 3) {
			width = std::atoi(argv[1]);
			height = width;
			images_num = std::atoi(argv[2]);
			if (argc >= 4) {
				auto device = argv[3];
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
		} else {
			std::cerr << "Usage: " << argv[0] << " rows images_num" << std::endl;
			return 1;
		}
	}

	auto common_start = std::chrono::high_resolution_clock::now();

	std::cout << "making data...";
	auto making_start = std::chrono::high_resolution_clock::now();
	auto a3 = MakeData<double>({ width, height, images_num });
	auto making_finish = std::chrono::high_resolution_clock::now();
	auto making_time = std::chrono::duration_cast<std::chrono::milliseconds>(making_finish - making_start);
	std::cout << "time:\t" << making_time.count() / 1000.0 << std::endl;

	std::cout << "HT calculation...";
	auto calculation_start = std::chrono::high_resolution_clock::now();
	HartleyTransform<double> ht(-1, height, 0, mode);
	for (int i = 0; i < images_num; ++i) {
		//PrintData2d(ptr + i * height * width, width, height);
		//ht.ForwardTransform(ptr + i * height * width);
		ht.ForwardTransform(a3.data());
		//PrintData2d(ptr + i * height * width, width, height);
	}
	auto calculation_finish = std::chrono::high_resolution_clock::now();
	auto calculation_time = std::chrono::duration_cast<std::chrono::milliseconds>(calculation_finish - calculation_start);
	std::cout << "time:\t" << calculation_time.count() / 1000.0 << std::endl;

	auto common_finish = std::chrono::high_resolution_clock::now();
	auto common_time = std::chrono::duration_cast<std::chrono::milliseconds>(common_finish - common_start);
	std::cout << "common time:\t" << common_time.count() / 1000.0 << std::endl;
	return 0;
}
