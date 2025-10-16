## RapiDHT — Discrete Hartley Transform (CPU/GPU)

RapiDHT is a library and a set of tests for performing the Discrete Hartley Transform in 1D/2D/3D modes:
- **CPU (OpenMP)**: FDHT implementation through 1D decomposition and transpositions
- **GPU (CUDA)**: matrix multiplications and transpositions using CUDA kernels; partially uses **cuBLAS**, partially custom **cuda** kernels
- **RFFT**: computation via real-valued FFT implementation

### Requirements
- CMake 3.18+
- C++17 compiler
- CUDA Toolkit (for GPU mode) and NVIDIA driver
- MPI (for distributed 3D processing)
- GoogleTest (included as a submodule/vendor in `3dparty`)

### Build
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

The default installed library is: `coreht`.

### Options
- `ENABLE_PROFILING` — enables a simple function profiler (macro `PROFILE_FUNCTION()`)

### Running Tests
```bash
cd build
ctest -C Release --output-on-failure
```
Test binaries are also built in `build/tests` and can be run directly.

### API Usage Example
```cpp
#include "rapidht.h"
#include "utilities.h"

using namespace RapiDHT;

// 2D example
size_t W = 256, H = 256;
auto mode = Modes::GPU; // or CPU/RFFT

std::vector<double> data = MakeData<double>({ W, H });

HartleyTransform ht(W, H, 0, mode);
ht.ForwardTransform(data.data());
ht.InverseTransform(data.data());
```

### Project Structure
- `include/` — public headers (`rapidht.h`, `utilities.h`, `kernel.h`, `dev_array.h`)
- `source/` — CPU/GPU implementations (`rapidht.cpp`, `kernel.cu`)
- `tests/` — tests (GoogleTest) and utility examples
- `3dparty/` — third-party dependencies (GoogleTest, FFTW for experiments)

### License & Authors
See licenses in `3dparty/*` directories and the project root files.
