#include <rapidht.h>
#include <utilities.h>
#include <iostream>
#include <cmath>
#include <numeric>
#include <cstring>
#include <chrono>

int main(int argc, char** argv)
{
    // ---- Размеры массива ----
    size_t width = static_cast<size_t>(pow(2, 12));
    size_t height = width;
    auto mode = RapiDHT::Modes::GPU;

    // ---- Обработка аргументов ----
    if (argc >= 3) {
        width = std::atoi(argv[1]);
        height = std::atoi(argv[2]);

        if (argc >= 4) {
            const char* device = argv[3];
            if (!strcmp(device, "CPU")) {
                mode = RapiDHT::Modes::CPU;
            }
            else if (!strcmp(device, "GPU")) {
                mode = RapiDHT::Modes::GPU;
            }
            else if (!strcmp(device, "RFFT")) {
                mode = RapiDHT::Modes::RFFT;
            }
            else {
                std::cerr << "Error: device must be either CPU, GPU or RFFT" << std::endl;
                return 1;
            }
        }
        if (argc >= 5) {
            std::cerr << "Usage: " << argv[0] << " rows cols [device]" << std::endl;
            return 1;
        }
    }
    else if (argc == 2) {
        std::cerr << "Usage: " << argv[0] << " rows cols [device]" << std::endl;
        return 1;
    }

    // ---- Создание данных ----
    auto original_data = make_data<double>({ width, height });
    auto transformed_data = original_data;
    //print_data_2d(original_data.data(), width, height);

    // ---- Засекаем время ----
    auto start_time = std::chrono::high_resolution_clock::now();

    // ---- Преобразование Хартли ----
    RapiDHT::HartleyTransform ht(width, height, 0, mode);
    ht.ForwardTransform(transformed_data.data());
    ht.InverseTransform(transformed_data.data());

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    show_time(0, elapsed.count(), "Common time");

    //print_data_2d(transformed_data.data(), width, height);
    // ---- Подсчёт ошибки ----
    double sum_sqr = std::transform_reduce(
        transformed_data.begin(), transformed_data.end(),
        original_data.begin(), 0.0, std::plus<>(),
        [](double x, double y) { return (x - y) * (x - y); }
    );

    std::cout << "Error:\t" << std::sqrt(sum_sqr) << std::endl;
    return 0;
}
