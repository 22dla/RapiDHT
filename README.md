<div align="center">

# RapiDHT

**Discrete Hartley Transform for 1D, 2D and 3D — on CPU and GPU**

[![CI](https://github.com/22dla/RapiDHT/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/22dla/RapiDHT/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-optional-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

A C++17 library that computes the *true multidimensional* Hartley transform,
built for the 3D volumes that come out of core tomography.

</div>

---

## ⚡ Quick start

```bash
git clone https://github.com/22dla/RapiDHT.git && cd RapiDHT
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

That is the CPU build — it needs no CUDA toolkit and no MPI.

```cpp
#include <rapidht/transform.h>
#include <vector>

using namespace RapiDHT;

std::vector<double> volume(256 * 256 * 256);
// ... fill it ...

HartleyTransform<double> ht(256, 256, 256, Modes::CPU);
ht.ForwardTransform(volume.data());
ht.InverseTransform(volume.data());   // back to where you started
```

For volumes on the GPU, keep the data on the card between operations:

```cpp
HartleyTransform<float> ht(512, 512, 512, Modes::GPU);
DeviceVolume<float> gpu(512ull * 512 * 512);

gpu.Upload(volume.data());     // once
ht.ForwardTransform(gpu);      // in place, no transfer
ht.InverseTransform(gpu);      // still no transfer
gpu.Download(volume.data());   // once
```

At 512³ the round trip across the bus costs about three times the transform, so
this is not a micro-optimisation.

---

## ✨ What it does

| | |
| --- | --- |
| 🧊 **True multidimensional** | Not the separable product of 1D transforms — the per-axis passes are followed by the Bracewell correction, and the tests check this against a directly evaluated definition |
| 🖥️ **Three backends** | `CPU` (OpenMP butterflies), `GPU` (CUDA, dense matrices through cuBLAS), `RFFT` (real FFT, 1D only) |
| 📌 **Device-resident data** | `DeviceVolume` keeps a volume on the card so a pipeline pays for the transfer once |
| 🧩 **Optional everything** | CUDA and MPI are off by default; the public headers name no CUDA type, so consuming the library needs no toolkit |
| 🎯 **Both precisions** | `float` and `double`; on consumer GPUs `float` is 9–29× faster |
| ✅ **Verified, not asserted** | 36 tests, including comparison against an `O(N²)` reference and analytically known transform pairs |

---

## 🏗️ Architecture

A transform along one axis of a volume is a large batch of short 1D transforms.
The GPU backend exploits exactly that: the dense `cas` matrix is only `W × W`,
small enough to stay resident, and it is reused once per line — so the work
becomes a batched GEMM rather than the memory-bound matrix-vector product a
single long 1D transform degenerates into.

```
ForwardTransform
├── CPU   ─ per-axis FDHT1D (precomputed twiddles) ─ Bracewell correction
├── GPU   ─ per-axis batched GEMM (cuBLAS) ─ transposes ─ Bracewell correction
└── RFFT  ─ real FFT, then Re(X) − Im(X)                       [1D only]
```

```
include/rapidht/    public headers — no CUDA type appears here
src/                implementation
    internal/       headers that are not installed
tests/              GoogleTest suites, registered individually with CTest
examples/           standalone demo programs
benchmarks/         Google Benchmark harness, FFTW and cuFFT baselines
docs/               benchmark report and figures
```

---

## 🔧 Building

### Requirements

**Always:** CMake 3.18+, a C++17 compiler with OpenMP, and the vendored
GoogleTest in `3dparty`.

**Only when the matching backend is enabled:** a CUDA Toolkit 11.2+ with an
NVIDIA driver, or an MPI implementation.

### Options

| Option | Default | What it does |
| --- | --- | --- |
| `RAPIDHT_WITH_CUDA` | `OFF` | Builds the GPU backend. With it off, `Modes::GPU` throws |
| `RAPIDHT_WITH_MPI` | `OFF` | Builds the MPI-distributed 3D backend |
| `RAPIDHT_BUILD_TESTS` | `ON` | Builds the test executables |
| `RAPIDHT_BUILD_BENCHMARKS` | `OFF` | Fetches Google Benchmark and builds the harness |
| `ENABLE_PROFILING` | `OFF` | Enables the `PROFILE_FUNCTION()` timer |

```bash
# with the GPU backend
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRAPIDHT_WITH_CUDA=ON
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `all-major`: one real architecture per
generation the toolkit supports, so the binary runs natively on any of them.
Pass `native` for a faster build targeting only the card in this machine —
that needs a GPU present at configure time, which is why it cannot be the
default.

> **Your host compiler has to be one `nvcc` accepts.** Each CUDA release caps
> the GCC version it will take: 12.4 stops at GCC 13, and Ubuntu's
> `nvidia-cuda-toolkit` package quietly ships its own `g++` of that version.
> Left alone, `nvcc` uses it, so `kernels.cu` gets built by one GCC and every
> `.cpp` by another — two ABIs in one binary. The build then fails at link with
> an undefined `__cxa_call_terminate`, and only in Release, since the reference
> lives in a cold section that `-O2` creates and `-O0` does not.
>
> This project therefore points `nvcc` at `CMAKE_CXX_COMPILER`, so a mismatch
> is a configure error naming the versions rather than a link error two minutes
> in. If you hit it, either build everything with a compiler the toolkit
> accepts:
>
> ```bash
> cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRAPIDHT_WITH_CUDA=ON \
>       -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13
> ```
>
> or install a newer toolkit from NVIDIA's repository instead of the
> distribution package, which is what CI does.

Which backends were compiled in is recorded in the generated
`rapidht_config.h` and exposed as `RapiDHT::kCudaEnabled` and
`RapiDHT::kMpiEnabled`. Whether a device is actually *present* is a separate
question, answered by `RapiDHT::IsGpuAvailable()`.

---

## 📊 Performance

At 512³ in single precision, with the volume resident on the device:

| | time | vs CPU |
| --- | ---: | ---: |
| CPU, 6 cores | 3 816 ms | — |
| **GPU, resident** | **58 ms** | **65.7×** |
| cuFFT + conversion | 12 ms | 320× |

The GPU backend is 65× the CPU one and 4.8× behind cuFFT, while needing half
cuFFT's extra device memory. The full picture, including profiling and an
extrapolation to datacentre hardware, is in **[docs/benchmarks.md](docs/benchmarks.md)**.

---

## 🧪 Tests

```bash
ctest --test-dir build --output-on-failure
```

Cases needing a GPU skip themselves through `IsGpuAvailable()`, so the suite is
green on a machine with the toolkit and no card.

The suite checks the transform against a direct `O(N²)` evaluation of the
definition, and against pairs that hold analytically: an impulse maps to a
constant, a constant to a scaled impulse, applying the transform twice scales by
N, Parseval's identity, and linearity. Round-trip tests alone would pass for any
invertible operation, which is how three wrong-result bugs survived in the GPU
backend until this suite arrived.

---

## 📐 Conventions

The forward transform is **unnormalised**: applying it twice scales the input by
N, and `InverseTransform` is the forward transform followed by a `1/N` scaling.

Data is stored **row-major with the first axis fastest**:
`index = i + Width·(j + Height·k)`.

Extents must be **powers of two** — `FDHT1D` rejects anything else. `height == 0`
means a 1D transform, `depth == 0` means 2D.

---

## 📄 License

MIT — see [LICENSE](LICENSE). Author: Evgeny A. Volkov.

The vendored GoogleTest in `3dparty/googletest` carries its own BSD-3-Clause
licence.
