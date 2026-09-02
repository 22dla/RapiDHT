/*
 * Project: RapiDHT
 * File: examples/fdht1d.cpp
 * Brief: Forward then inverse 1D transform, reporting the round-trip error.
 *
 * Run with no arguments for the defaults, or as: fdht1d N [CPU|GPU|RFFT]
 */

#include <rapidht/transform.h>
#include <rapidht/utilities.h>

using namespace RapiDHT;

int main(int argc, char** argv)
{
    LoadingConfig cfg;
    cfg.width = 1 << 10;
    if (argc > 1) {
        cfg = ParseArgs(argc, argv);
    }
    cfg.print();

    auto originalData = MakeData<double>({ cfg.width });
    auto transformedData = originalData;

    PrintData1d(originalData.data(), static_cast<int>(cfg.width));

    auto startTime = std::chrono::high_resolution_clock::now();

    HartleyTransform<double> ht(cfg.width, 0, 0, cfg.mode);
    ht.ForwardTransform(transformedData.data());
    ht.InverseTransform(transformedData.data());

    PrintData1d(transformedData.data(), static_cast<int>(cfg.width));

    auto endTime = std::chrono::high_resolution_clock::now();
    ShowElapsedTime<std::chrono::milliseconds>(startTime, endTime, "Common time");

    double sumOfSquares = std::transform_reduce(
        transformedData.begin(), transformedData.end(),
        originalData.begin(), 0.0, std::plus<>(),
        [](double x, double y) { return (x - y) * (x - y); });

    std::cout << "Error:\t" << std::sqrt(sumOfSquares) << std::endl;
    return 0;
}
