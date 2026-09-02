/*
 * Project: RapiDHT
 * File: examples/fdht2d.cpp
 * Brief: Forward then inverse 2D transform, reporting the round-trip error.
 *
 * Run with no arguments for the defaults, or as: fdht2d NxM [CPU|GPU]
 */

#include <rapidht/transform.h>
#include <rapidht/utilities.h>

using namespace RapiDHT;

int main(int argc, char** argv)
{
    LoadingConfig cfg;
    cfg.width = 1 << 5;
    cfg.height = 1 << 10;
    if (argc > 1) {
        cfg = ParseArgs(argc, argv);
    }
    cfg.print();

    auto originalData = MakeData<double>({ cfg.width, cfg.height }, FillMode::Random);
    auto transformedData = originalData;

    auto startTime = std::chrono::high_resolution_clock::now();

    HartleyTransform<double> ht(cfg.width, cfg.height, 0, cfg.mode);
    ht.ForwardTransform(transformedData.data());
    ht.InverseTransform(transformedData.data());

    auto endTime = std::chrono::high_resolution_clock::now();
    ShowElapsedTime<std::chrono::milliseconds>(startTime, endTime, "Common time");

    CompareData(originalData, transformedData);
    return 0;
}
