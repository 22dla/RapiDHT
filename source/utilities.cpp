#include "utilities.h"

namespace RapiDHT {

// ������� ������ ���� "NxM[xK]"
std::vector<size_t> ParseDims(const std::string& str) {
	std::vector<size_t> dims;
	std::stringstream ss(str);
	std::string item;
	while (std::getline(ss, item, 'x')) {
		dims.push_back(std::stoul(item));
	}
	return dims;
}

// ����������� ����������
Modes ParseDevice(const char* device) {
	if (!strcmp(device, "CPU")) return Modes::CPU;
	if (!strcmp(device, "GPU")) {
#ifndef USE_CUDA
		// Fallback to CPU if CUDA is not available
		return Modes::CPU;
#else
		return Modes::GPU;
#endif
	}
	if (!strcmp(device, "RFFT")) return Modes::RFFT;
	throw std::runtime_error("Error: device must be CPU, GPU or RFFT");
}

LoadingConfig ParseArgs(int argc, char** argv) {
	LoadingConfig cfg;

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " NxM[xK] [device]" << std::endl;
		exit(1);
	}

	auto dims = ParseDims(argv[1]);
	if (dims.size() == 1) {
		cfg.width = dims[0];
		cfg.height = 1;
		cfg.depth = 1;
	} else if (dims.size() == 2) {
		cfg.width = dims[0];
		cfg.height = dims[1];
		cfg.depth = 1;
	} else if (dims.size() == 3) {
		cfg.width = dims[0];
		cfg.height = dims[1];
		cfg.depth = dims[2];
	} else {
		throw std::runtime_error("Error: dimensions must be N, NxM or NxMxK");
	}

	if (argc >= 3) {
		cfg.mode = ParseDevice(argv[2]);
	}

	return cfg;
}

} // namespace RapiDHT
