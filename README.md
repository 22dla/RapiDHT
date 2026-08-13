## RapiDHT — Discrete Hartley Transform (CPU/GPU)

RapiDHT is a library and a set of tests for performing the Discrete Hartley Transform in 1D/2D/3D modes:
- **CPU (OpenMP)**: FDHT implementation through 1D decomposition and transpositions
- **GPU (CUDA)**: matrix multiplications and transpositions using CUDA kernels; partially uses **cuBLAS**, partially custom **cuda** kernels
- **RFFT**: computation via real-valued FFT implementation

### Requirements
Always required:
- CMake 3.18+
- C++17 compiler with OpenMP
- GoogleTest (vendored in `3dparty`)

Optional, only when the corresponding backend is enabled:
- CUDA Toolkit 11+ and an NVIDIA driver — for `RAPIDHT_WITH_CUDA=ON`
- An MPI implementation — for `RAPIDHT_WITH_MPI=ON`

### Build
The default configuration is **CPU-only** and needs neither CUDA nor MPI.

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
#### With the GPU and/or MPI backends:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DRAPIDHT_WITH_CUDA=ON -DRAPIDHT_WITH_MPI=ON
cmake --build build --config Release
```

The default installed library is: `coreht`.

### Options
| Option | Default | Description |
| --- | --- | --- |
| `RAPIDHT_WITH_CUDA` | `OFF` | Build the CUDA/cuBLAS GPU backend. When `OFF`, constructing a transform with `Modes::GPU` throws `std::runtime_error`. |
| `RAPIDHT_WITH_MPI` | `OFF` | Build the MPI-distributed 3D backend. When `OFF`, 3D transforms run single-process. |
| `RAPIDHT_BUILD_TESTS` | `ON` | Build the test executables. |
| `ENABLE_PROFILING` | `OFF` | Enable a simple function profiler (macro `PROFILE_FUNCTION()`). |

Enabled backends are recorded in the generated header `rapidht_config.h`
(`RAPIDHT_WITH_CUDA` / `RAPIDHT_WITH_MPI`) and exposed to user code as the
`constexpr bool` flags `RapiDHT::kCudaEnabled` and `RapiDHT::kMpiEnabled`.
GPU tests skip themselves automatically in a CPU-only build.

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
