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

Measured on 6 cores at 4.4 GHz with an RTX 3060 Ti, median of 7 repetitions.

![3D benchmark](docs/images/benchmark-3d.svg)

**3D volumes, which is what the library is for.** `GPU−PCIe` subtracts the
host/device round trip, which the current API performs on every call:

| | CPU | GPU | GPU−PCIe | speedup vs CPU |
| --- | --- | --- | --- | --- |
| f32 128³ | 15 392 µs | 4 803 µs | 2 832 µs | 3.2× (5.4× net) |
| f32 256³ | 410 724 µs | 51 712 µs | 36 584 µs | 7.9× (11.2× net) |
| f32 512³ | 3 861 066 µs | 434 567 µs | 312 945 µs | 8.9× (12.3× net) |
| f64 128³ | 24 759 µs | 14 444 µs | 10 604 µs | 1.7× (2.3× net) |
| f64 256³ | 470 233 µs | 202 110 µs | 171 917 µs | 2.3× (2.7× net) |
| f64 512³ | 4 432 536 µs | 2 442 008 µs | 2 201 164 µs | 1.8× (2.0× net) |

The GPU backend wins on volumes, and the margin grows with size. Its dense
`cas` matrix is only `W × W` per axis and is reused once per line, so the work
becomes a batched GEMM — the case GPUs are built for — rather than the
memory-bound matrix-vector product a single long 1D transform degenerates into.

**Precision dominates on consumer hardware.** A GeForce runs FP64 at a fraction
of its FP32 rate, so moving to `float` speeds the GPU up by 3–5.6× while barely
moving the CPU (1.15×). Anyone using this backend for volumes should be in
single precision.

**Only 3D is worth offloading.** In 2D the GPU is slower than the CPU in double
and roughly at parity in float; in 1D it is 66–80× slower, and the `n × n`
matrix makes large sizes impossible outright.

For reference, FFTW `FFTW_DHT` in double takes 11 503 / 125 314 / 1 360 737 µs
on the same volumes — about 3× faster than this CPU backend, and about 3×
slower than this GPU backend in float. Note that FFTW computes the separable
transform in 2D/3D while RapiDHT computes the true multidimensional one and
pays for an extra Bracewell pass, so it is doing less work.

Two openings visible in the same table. The CPU backend gains almost nothing
from `float` (1.15× where 2× is available), which points at butterflies that do
not vectorise. And its throughput falls from 85 to 36 M points/s between 128³
and 256³, exactly where the volume stops fitting in the 18 MiB L3.

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
