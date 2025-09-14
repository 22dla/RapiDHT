#include "rapidht.h"
#include "utilities.h"

using namespace RapiDHT;

int main(int argc, char** argv) {
	// ---- Обработка аргументов ----
	//auto cfg = ParseArgs(argc, argv);
	LoadingConfig cfg;
	cfg.width = 1 << 5;
	cfg.height = 1 << 10;
	//cfg.height = cfg.width;
	cfg.mode = Modes::CPU;

	auto width = cfg.width;
	auto height = cfg.height;
	auto mode = cfg.mode;

	cfg.print();

	// ---- Создание данных ----
	auto original_data = MakeData<double>({width, height}, FillMode::Random);

	//original_data = {183.00, 248.00, 80.00,	 203.00, 129.00, 200.00, 31.00,	 5.00,	 19.00,	 196.00, 186.00,
	//				 167.00, 107.00, 245.00, 39.00,	 156.00, 199.00, 126.00, 103.00, 181.00, 185.00, 16.00,
	//				 234.00, 125.00, 218.00, 86.00,	 118.00, 151.00, 72.00,	 232.00, 206.00, 141.00};

	auto transformed_data = original_data;
	//PrintData2d(original_data.data(), width, height);

	// ---- Засекаем время ----
	auto start_time = std::chrono::high_resolution_clock::now();

	// ---- Преобразование Хартли ----
	HartleyTransform<double> ht(width, height, 0, mode);
	ht.ForwardTransform(transformed_data.data());
	//PrintData2d(transformed_data.data(), width, height);

	ht.InverseTransform(transformed_data.data());
	//PrintData2d(transformed_data.data(), width, height);

	auto end_time = std::chrono::high_resolution_clock::now();
	ShowElapsedTime<std::chrono::milliseconds>(start_time, end_time, "Common time");

	//PrintData2d(transformed_data.data(), width, height);
	// ---- Подсчёт ошибки ----
	CompareData(original_data, transformed_data);
	return 0;
}
