## RapiDHT — Discrete Hartley Transform (CPU/GPU)

RapiDHT — библиотека и набор тестов для дискретного преобразования Хартли в 1D/2D/3D режимах:
- **CPU (OpenMP)**: быстрая реализация FDHT через разложение на 1D и транспонирования
- **GPU (CUDA)**: матричные умножения и транспонирования ядрами CUDA; планируется миграция на **cuBLAS**
- **RFFT**: вычисление через реализацию на основе вещественного FFT

### Требования
- CMake 3.18+
- C++17-компилятор
- CUDA Toolkit (для GPU-режима) и драйвер NVIDIA
- MPI (для распределённой 3D обработки)
- GoogleTest (включён как сабмодуль/вендор в `3dparty`)

### Сборка
```bash
cmake -S . -B build -DENABLE_PROFILING=ON
cmake --build build --config Release
```

По умолчанию устанавливаемая библиотека: `coreht`.

### Опции
- `ENABLE_PROFILING` — включает простой профайлер функций (макрос `PROFILE_FUNCTION()`)

### Запуск тестов
```bash
ctest --test-dir build -C Release --output-on-failure
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

### О GPU-реализации
Текущая реализация использует собственные CUDA-ядра для умножений и транспонирований. В ближайших версиях планируется перенос на **cuBLAS** (`cublasDgemm`, `cublasDgemv`, `cublasDgemmStridedBatched`) для ускорения и лучшего использования ресурсов GPU.

### Структура проекта
- `include/` — публичные заголовки (`rapidht.h`, `utilities.h`, `kernel.h`, `dev_array.h`)
- `source/` — реализации CPU/GPU (`rapidht.cpp`, `kernel.cu`)
- `tests/` — тесты (GoogleTest) и утилитарные примеры
- `3dparty/` — сторонние зависимости (GoogleTest, FFTW для экспериментов)

### Лицензирование и авторство
См. лицензии в директориях `3dparty/*` и корневые файлы проекта.
