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

### Keeping data on the device
The host-pointer overloads upload, transform and download on every call. For a
512³ volume that round trip costs about three times as much as the transform,
so a pipeline applying more than one operation should hold the volume on the
card instead:

```cpp
HartleyTransform<float> ht(512, 512, 512, Modes::GPU);
DeviceVolume<float> volume(512ull * 512 * 512);

volume.Upload(data.data());     // once
ht.ForwardTransform(volume);    // in place, no transfer
ht.InverseTransform(volume);    // still no transfer
volume.Download(data.data());   // once
```

`DeviceVolume` owns its allocation, is move-only, and names no CUDA type, so
this header still needs no CUDA toolkit to include. Constructing one in a build
without CUDA throws.

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

**3D volumes, which is what the library is for.** `GPU` copies the data across
the bus on every call; `Resident` keeps it on the card, as `DeviceVolume` does:

| | CPU | GPU | Resident | vs CPU |
| --- | --- | --- | --- | --- |
| f32 128³ | 15 025 µs | 2 595 µs | **806 µs** | 18.6× |
| f32 256³ | 412 630 µs | 19 858 µs | **5 047 µs** | **81.8×** |
| f32 512³ | 3 815 926 µs | 177 130 µs | **58 117 µs** | **65.7×** |
| f64 128³ | 26 856 µs | 11 117 µs | 7 292 µs | 3.7× |
| f64 256³ | 476 088 µs | 136 434 µs | 107 829 µs | 4.4× |
| f64 512³ | 4 422 303 µs | 1 903 826 µs | 1 673 060 µs | 2.6× |

Three things to read off that table.

**The transform itself is 65× faster than the CPU backend, and the bus was
hiding it.** At 512³ the resident figure of 58.1 ms matches the 58.1 ms that a
profiler attributes to the kernels, so nothing but the transform is being
measured. The dense `cas` matrix is only `W × W` per axis and is reused once per
line, which turns the work into a batched GEMM — the case GPUs are built for —
rather than the memory-bound matrix-vector product a single long 1D transform
degenerates into. The GEMMs run at 10.5 TFLOP/s against a 16.2 TFLOP/s peak.

**Precision decides the outcome on consumer hardware.** A GeForce runs FP64 at a
fraction of its FP32 rate, so single precision is 9–29× faster on the device
while barely moving the CPU. Anyone using this backend for volumes should be in
`float`.

**2D now pays off too, 3D emphatically.** In 1D the GPU remains far slower: the
axis is the whole signal, so the matrix is `n × n` and the approach collapses.

### Against cuFFT

cuFFT computes the Fourier transform, but for real input the Hartley transform
follows by one linear pass, `Re(X) − Im(X)`, so cuFFT plus an `O(N)` conversion
computes exactly what this library computes. That makes it the baseline that
matters. At 512³ in single precision, both working on data already resident on
the device, and both measured in the same run so the figures are directly
comparable (which is why the 57.4 ms below differs by 1% from the 58.1 ms in
the table above, taken from a separate run):

| | time | arithmetic | achieved | extra device memory |
| --- | --- | --- | --- | --- |
| RapiDHT, matrix | 57.4 ms | 412 GFLOP | 7.2 TFLOP/s | **515 MiB** |
| cuFFT + conversion | **11.9 ms** | ~9 GFLOP | 0.76 TFLOP/s | 1 028 MiB |

**cuFFT is 4.8× faster on the transform.** No amount of framing changes that,
and it is the number to quote.

Two things sit alongside it. The matrix path performs about 46× more arithmetic
yet finishes only 4.8× behind, because it is compute-bound and reaches 44% of
the card while an FFT is bandwidth-bound and reaches around 5%. And it needs
**half the extra device memory**: 515 MiB of scratch and matrices against
cuFFT's 1 028 MiB of spectrum and workspace, measured through
`cufftGetSize3d`. On this 8 GiB card that is the difference between a largest
cubic volume of about 997³ and about 871³.

Whether the arithmetic disadvantage or the memory advantage decides depends on
the hardware and on the surrounding pipeline, neither of which this table
covers. A GEMM maps onto tensor cores and an FFT does not, so the gap is a
property of the card as much as of the algorithm; and the transform is only one
stage of a filtering pipeline, where a Hartley spectrum multiplies by an even
kernel with a real Hadamard product rather than a complex one.

For reference, FFTW `FFTW_DHT` in double takes 11 316 / 127 101 / 1 386 526 µs
on the same volumes. That comparison carries three caveats: FFTW here is
single-threaded, it is in double against our float, and in 2D/3D it computes the
separable transform while RapiDHT computes the true multidimensional one and
pays for an extra Bracewell pass. The honest reading is that a GPU beats a
single CPU core, not that this beats FFTW.

Two openings remain. The CPU backend gains almost nothing from `float`, which
points at butterflies that do not vectorise. And its throughput falls from 85 to
36 M points/s between 128³ and 256³, exactly where the volume stops fitting in
the 18 MiB L3.

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
