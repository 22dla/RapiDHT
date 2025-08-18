#include <rapidht.h>
#include <utilities.h>

using namespace RapiDHT;

int main(int argc, char** argv) {
	// ---- Обработка аргументов ----
	auto cfg = ParseArgs(argc, argv);
	cfg.width = 1 << 10;
	cfg.height = 1 << 5;
	cfg.depth = 1 << 6;
	cfg.mode = Modes::CPU;

	auto width = cfg.width;
	auto height = cfg.height;
	auto depth = cfg.depth;
	auto mode = cfg.mode;

	auto common_start = std::chrono::high_resolution_clock::now();

	std::cout << "making data...";
	auto making_start = std::chrono::high_resolution_clock::now();
	auto a3 = MakeData<double>({ width, height, depth });
	auto making_finish = std::chrono::high_resolution_clock::now();
	auto making_time = std::chrono::duration_cast<std::chrono::milliseconds>(making_finish - making_start);
	std::cout << "time:\t" << making_time.count() / 1000.0 << std::endl;

	std::cout << "HT calculation...";
	auto calculation_start = std::chrono::high_resolution_clock::now();
	RapiDHT::HartleyTransform ht(width, height, depth, mode);
	ht.ForwardTransform(a3.data());
	auto calculation_finish = std::chrono::high_resolution_clock::now();
	auto calculation_time = std::chrono::duration_cast<std::chrono::milliseconds>(calculation_finish - calculation_start);
	std::cout << "time:\t" << calculation_time.count() / 1000.0 << std::endl;

	auto common_finish = std::chrono::high_resolution_clock::now();
	auto common_time = std::chrono::duration_cast<std::chrono::milliseconds>(common_finish - common_start);
	std::cout << "common time:\t" << common_time.count() / 1000.0 << std::endl;
	return 0;
}
