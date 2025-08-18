#include <rapidht.h>
#include <utilities.h>

using namespace RapiDHT;

int main(int argc, char** argv) {
	// ---- Обработка аргументов ----
	auto cfg = ParseArgs(argc, argv);
	cfg.width = 1 << 10;

	auto width = cfg.width;
	auto height = cfg.height;
	auto mode = cfg.mode;

	// ---- Создание данных ----
	auto original_data = MakeData<double>({ width });
	auto transformed_data = original_data;

	PrintData1d(original_data.data(), width);

	// ---- Засекаем время ----
	auto start_time = std::chrono::high_resolution_clock::now();

	// ---- Преобразование Хартли ----
	HartleyTransform ht(width, 0, 0, mode);
	ht.ForwardTransform(transformed_data.data());
	ht.InverseTransform(transformed_data.data());

	PrintData1d(transformed_data.data(), width);

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end_time - start_time;
	ShowTime(0, elapsed.count(), "Common time");

	// ---- Подсчёт ошибки ----
	double sum_sqr = std::transform_reduce(
		transformed_data.begin(), transformed_data.end(),
		original_data.begin(), 0.0, std::plus<>(),
		[](double x, double y) { return (x - y) * (x - y); }
	);

	std::cout << "Error:\t" << std::sqrt(sum_sqr) << std::endl;
	return 0;
}
