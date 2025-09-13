## RapiDHT — Discrete Hartley Transform (CPU/GPU)

RapiDHT — библиотека и набор тестов для дискретного преобразования Хартли в 1D/2D/3D режимах:
- **CPU (OpenMP)**: быстрая реализация FDHT через разложение на 1D и транспонирования
- **GPU (CUDA)**: матричные умножения и транспонирования ядрами CUDA; частично используется **cuBLAS**, частично - кастомные **cuda**-ядра
- **RFFT**: вычисление через реализацию на основе вещественного FFT

### Требования
- CMake 3.18+
- C++17-компилятор
- CUDA Toolkit (для GPU-режима) и драйвер NVIDIA
- MPI (для распределённой 3D обработки)
- GoogleTest (включён как сабмодуль/вендор в `3dparty`)

### Сборка
#### Debug:
```bash
cmake -S . -B build 
cmake --build build
```
#### Release:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

По умолчанию устанавливаемая библиотека: `coreht`.

### Опции
- `ENABLE_PROFILING` — включает простой профайлер функций (макрос `PROFILE_FUNCTION()`)

### Запуск тестов
```bash
cd build
ctest -C Release --output-on-failure
```
Тестовые бинарники также собираются в `build/tests` и могут запускаться напрямую.

### Использование API
```cpp
#include "rapidht.h"
#include "utilities.h"

using namespace RapiDHT;

// 2D пример
size_t W = 256, H = 256;
auto mode = Modes::GPU; // или CPU/RFFT

std::vector<double> data = MakeData<double>({ W, H });

HartleyTransform ht(W, H, 0, mode);
ht.ForwardTransform(data.data());
ht.InverseTransform(data.data());
```

### Структура проекта
- `include/` — публичные заголовки (`rapidht.h`, `utilities.h`, `kernel.h`, `dev_array.h`)
- `source/` — реализации CPU/GPU (`rapidht.cpp`, `kernel.cu`)
- `tests/` — тесты (GoogleTest) и утилитарные примеры
- `3dparty/` — сторонние зависимости (GoogleTest, FFTW для экспериментов)

### Лицензирование и авторство
См. лицензии в директориях `3dparty/*` и корневые файлы проекта.
