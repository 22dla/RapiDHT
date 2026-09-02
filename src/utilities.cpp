#include <rapidht/utilities.h>

#include <cstring> // strcmp -- previously pulled in transitively via <mpi.h>

namespace RapiDHT {

std::vector<size_t> ParseDims(const std::string& str)
{
    std::vector<size_t> dims;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, 'x')) {
        dims.push_back(std::stoul(item));
    }
    return dims;
}

Modes ParseDevice(const char* device)
{
    if (!strcmp(device, "CPU"))
        return Modes::CPU;
    if (!strcmp(device, "GPU"))
        return Modes::GPU;
    if (!strcmp(device, "RFFT"))
        return Modes::RFFT;
    throw std::runtime_error("Error: device must be CPU, GPU or RFFT");
}

LoadingConfig ParseArgs(int argc, char** argv)
{
    LoadingConfig cfg;

    // Throws rather than calling exit(): this lives in the library, and a
    // caller that mis-parses its own arguments should get the chance to
    // print its own usage message.
    if (argc < 2) {
        throw std::runtime_error(
            std::string("Usage: ") + (argc > 0 ? argv[0] : "rapidht") + " NxM[xK] [device]");
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
