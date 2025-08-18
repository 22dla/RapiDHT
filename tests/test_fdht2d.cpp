#include <rapidht.h>
#include <utilities.h>

using namespace RapiDHT;

int main(int argc, char** argv) {
	// ---- Обработка аргументов ----
	auto cfg = ParseArgs(argc, argv);
	cfg.width = 1 << 12;
	cfg.height = 1 << 12;
	cfg.mode = Modes::CPU;

	auto width = cfg.width;
	auto height = cfg.height;
	auto mode = cfg.mode;

	cfg.print();

	// ---- Создание данных ----
	auto original_data = MakeData<double>({ width, height }, FillMode::Random);
	auto transformed_data = original_data;
	//PrintData2d(original_data.data(), width, height);

	// ---- Засекаем время ----
	

	// ---- Преобразование Хартли ----
	HartleyTransform ht(width, height, 0, mode);
	auto start_time = std::chrono::high_resolution_clock::now();
	ht.ForwardTransform(transformed_data.data());
	//PrintData2d(transformed_data.data(), width, height);
	ht.InverseTransform(transformed_data.data());

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end_time - start_time;
	ShowTime(0, elapsed.count(), "Common time");

	//PrintData2d(transformed_data.data(), width, height);
	// ---- Подсчёт ошибки ----
	CompareData(original_data, transformed_data);
	return 0;
}
