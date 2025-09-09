#include "rapidht.h"
#include "utilities.h"

using namespace RapiDHT;

int main(int argc, char** argv) {
	// ---- Обработка аргументов ----
	//auto cfg = ParseArgs(argc, argv);
	LoadingConfig cfg;
	cfg.width = 1 << 9;
	cfg.height = cfg.width;
	cfg.depth = cfg.width;
	//cfg.height = 4;
	//cfg.depth = 8;
	cfg.mode = Modes::CPU;

	cfg.print();

	auto width = cfg.width;
	auto height = cfg.height;
	auto depth = cfg.depth;
	auto mode = cfg.mode;

	// ---- Создание данных ----
	auto making_start = std::chrono::high_resolution_clock::now();
	auto original_data = MakeData<float>({width, height, depth});
	//auto transformed_data = original_data;
	auto making_finish = std::chrono::high_resolution_clock::now();
	ShowElapsedTime<std::chrono::milliseconds>(making_start, making_finish, "Making time");

	// ---- Засекаем время ----
	auto common_start = std::chrono::high_resolution_clock::now();

	// ---- Преобразование Хартли ----
	HartleyTransform<float> ht(width, height, depth, mode);
	ht.ForwardTransform(original_data.data());
	//PrintData3d(transformed_data.data(), width, height, depth);
	ht.InverseTransform(original_data.data());

	auto common_finish = std::chrono::high_resolution_clock::now();
	ShowElapsedTime<std::chrono::seconds>(common_start, common_finish, "Common time");

	// ---- Подсчёт ошибки ----
	//CompareData(original_data, transformed_data);
	return 0;
}
