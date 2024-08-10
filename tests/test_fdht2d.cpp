#include <iostream>
#include <rapidht.h>
#include <utilities.h>
#include <cmath>
#include <cstring>
#include <numeric>

int main(int argc, char** argv) {
	// Define global 3D array sizes
	int rows = static_cast<int>(pow(2, 13));
	int cols = rows;
	RapiDHT::Modes mode = RapiDHT::GPU;

	// If arguments are parced then exactly two arguments are required
	if (argc >= 2) {
		if (argc >= 3) {
			rows = std::atoi(argv[1]);
			cols = std::atoi(argv[2]);
			if (argc >= 4) {
				auto device = argv[3];
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
		} else {
			std::cerr << "Usage: " << argv[0] << " rows cols" << std::endl;
			return 1;
		}
	}

	auto a2_1 = make_data<double>({ rows, cols });
	auto a2_2(a2_1);

	double common_start, common_finish;
	common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

	auto ptr = a2_1.data();

	//print_data_2d(ptr, rows, cols);

	RapiDHT::HartleyTransform ht(rows, cols, 0, mode);
	ht.ForwardTransform(ptr);
	ht.InverseTransform(ptr);

	//print_data_2d(ptr, rows, cols);

	common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	show_time(common_start, common_finish, "Common time");

	double sum_sqr = std::transform_reduce(
		a2_1.begin(), a2_1.end(), a2_2.begin(), 0.0, std::plus<>(),
		[](double x, double y) { return (x - y) * (x - y); }
	);
	std::cout << "Error:\t" << std::sqrt(sum_sqr) << std::endl;
	return 0;
}
