/*
 * Project: RapiDHT
 * File: examples/fdht3d.cpp
 * Brief: Forward then inverse 3D transform, reporting the round-trip error.
 *
 * Run with no arguments for the defaults, or as: fdht3d NxMxK [CPU|GPU]
 */

#include <rapidht/transform.h>
#include <rapidht/utilities.h>

using namespace RapiDHT;

int main(int argc, char** argv)
{
    LoadingConfig cfg;
    cfg.width = 1 << 2;
    cfg.height = 1 << 3;
    cfg.depth = 1 << 4;

    if (argc > 1) {
        cfg = ParseArgs(argc, argv);
    } else {
        // Exercise the backend this build actually has. IsGpuAvailable rather
        // than kCudaEnabled: the latter only says the backend was compiled in,
        // and a machine can carry the toolkit and no card at all.
        cfg.mode = IsGpuAvailable() ? Modes::GPU : Modes::CPU;
    }
    cfg.print();

    auto makingStart = std::chrono::high_resolution_clock::now();
    auto originalData = MakeData<float>({ cfg.width, cfg.height, cfg.depth }, FillMode::Sequential);
    auto transformedData = originalData;
    auto makingFinish = std::chrono::high_resolution_clock::now();
    ShowElapsedTime<std::chrono::milliseconds>(makingStart, makingFinish, "Making time");

    auto commonStart = std::chrono::high_resolution_clock::now();

    HartleyTransform<float> ht(cfg.width, cfg.height, cfg.depth, cfg.mode);
    ht.ForwardTransform(transformedData.data());
    ht.InverseTransform(transformedData.data());

    auto commonFinish = std::chrono::high_resolution_clock::now();
    ShowElapsedTime<std::chrono::milliseconds>(commonStart, commonFinish, "Common time");

    CompareData(originalData, transformedData);
    return 0;
}
