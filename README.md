## RapiDHT — Discrete Hartley Transform (CPU/GPU)

[![CI](https://github.com/22dla/RapiDHT/actions/workflows/ci.yml/badge.svg)](https://github.com/22dla/RapiDHT/actions/workflows/ci.yml)

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

With `RAPIDHT_WITH_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES` is left to CMake, which
derives a default from `nvcc`. Pass `-DCMAKE_CUDA_ARCHITECTURES=native` to target
the card in the build machine, or an explicit list such as `"80;86"`. Note that
`native` needs a GPU present at configure time, so it cannot be the default.

Enabled backends are recorded in the generated header `rapidht_config.h`
(`RAPIDHT_WITH_CUDA` / `RAPIDHT_WITH_MPI`) and exposed to user code as the
`constexpr bool` flags `RapiDHT::kCudaEnabled` and `RapiDHT::kMpiEnabled`.
GPU tests skip themselves automatically in a CPU-only build.

### Benchmarks
Off by default, because enabling them makes the configure step fetch Google
Benchmark:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRAPIDHT_BUILD_BENCHMARKS=ON
cmake --build build --config Release
./build/benchmarks/bench_transform
```
If FFTW is installed and visible to `pkg-config`, an `FFTW_DHT` baseline is
built alongside. Note that in 2D/3D FFTW computes the *separable* transform,
whereas RapiDHT computes the true multidimensional one and pays for an extra
Bracewell pass — so the multidimensional comparison is not like for like.

Indicative figures, double precision, 2 threads on a shared cloud VM
(13th Gen Core i7-1360P), median of 7 repetitions. Absolute numbers say more
about that VM than about the library; the ratios are the point. "Before" is
the state prior to precomputing the butterfly twiddle factors:

| case | before | after | speedup |
| --- | --- | --- | --- |
| 1D 1 024 | 35.1 µs | 9.0 µs | 3.90× |
| 1D 16 384 | 822.8 µs | 242.1 µs | 3.40× |
| 1D 262 144 | 17 404 µs | 4 988 µs | 3.49× |
| 2D 128² | 426.2 µs | 208.2 µs | 2.05× |
| 2D 512² | 10 019 µs | 4 991 µs | 2.01× |
| 3D 32³ | 965.9 µs | 614.6 µs | 1.57× |
| 3D 64³ | 10 170 µs | 6 223 µs | 1.63× |

`FDHT1D` used to call `std::cos` and `std::sin` inside its innermost butterfly
loop; the factors are now built once per axis in the constructor. The gain
falls off with dimension because the multidimensional paths spend a growing
share of their time in the Bracewell correction and in strided memory access
rather than in the butterflies — those are the next things to look at.

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
- `include/` — public headers (`rapidht.h`, `utilities.h`). These never name a
  CUDA type, so consuming the library requires no CUDA toolkit even when the
  GPU backend is compiled in, and the class layout does not depend on
  `RAPIDHT_WITH_CUDA`.
- `source/` — implementations plus internal headers not meant for consumers
  (`rapidht.cpp`, `kernel.cu`, `kernel.h`, `dev_array.h`)
- `tests/` — tests (GoogleTest) and utility examples
- `3dparty/` — vendored third-party dependencies (GoogleTest)

### License & Authors
RapiDHT is released under the MIT License — see [LICENSE](LICENSE).

Author: Evgeny A. Volkov.

The vendored GoogleTest in `3dparty/googletest` carries its own BSD-3-Clause
licence, reproduced in `3dparty/googletest/LICENSE`.
