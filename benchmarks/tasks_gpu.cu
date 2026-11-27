/*
 * Project: RapiDHT
 * File: benchmarks/tasks_gpu.cu
 * Brief: R-5 stage 5 -- what the end-to-end tasks add around the transforms.
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cufft_common.cuh"
#include "tasks_gpu.h"
// The LTO-IR of the cuFFT callbacks (cufft_int_lto_callback.cu).
#include "cufft_int_lto_fatbin.h"

namespace rapidht_bench {

std::vector<double> GaussianTaps(double sigma)
{
    const int r = static_cast<int>(std::lround(3.0 * sigma));
    std::vector<double> c(static_cast<size_t>(r) + 1);
    double sum = 0.0;
    for (int d = 0; d <= r; ++d) {
        c[static_cast<size_t>(d)] = std::exp(-0.5 * d * d / (sigma * sigma));
        sum += (d == 0 ? 1.0 : 2.0) * c[static_cast<size_t>(d)];
    }
    for (auto& v : c) {
        v /= sum;
    }
    return c;
}

std::vector<float> GaussianAxisSpectrum(const std::vector<double>& taps, size_t n)
{
    const size_t r = taps.size() - 1;
    if (2 * r + 1 > n) {
        throw std::runtime_error("GaussianAxisSpectrum: the kernel is longer than the axis");
    }
    const double pi = std::acos(-1.0);
    std::vector<float> g(n);
    for (size_t k = 0; k < n; ++k) {
        double s = taps[0];
        for (size_t d = 1; d <= r; ++d) {
            // cos(2 pi k d / n) with k d reduced mod n first: exact argument.
            s += 2.0 * taps[d] * std::cos(2.0 * pi * static_cast<double>((k * d) % n) / static_cast<double>(n));
        }
        g[k] = static_cast<float>(s);
    }
    return g;
}

namespace {

unsigned Blocks(size_t count, unsigned block) { return static_cast<unsigned>((count + block - 1) / block); }

/// Rounded and clamped to TOut, as the integer inverse converts (rint, ties to even).
template <typename TOut>
__device__ __forceinline__ TOut ToInteger(float v)
{
    const float r = rintf(v);
    const float top = sizeof(TOut) == 1 ? 255.0f : 65535.0f;
    return static_cast<TOut>(r >= 0.0f ? (r <= top ? r : top) : 0.0f);
}

/// Four consecutive x a thread, as float4; sy[y] sz[z] once per thread.
__global__ void SeparableProduct(float4* __restrict__ data, const float4* __restrict__ sx,
    const float* __restrict__ sy, const float* __restrict__ sz, int quadsPerLine, int H)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= quadsPerLine) {
        return;
    }
    const int y = blockIdx.y;
    const int z = blockIdx.z;
    const float yz = sy[y] * sz[z];
    const size_t i = static_cast<size_t>(quadsPerLine) * (y + static_cast<size_t>(H) * z) + q;
    const float4 s = sx[q];
    float4 v = data[i];
    v.x *= s.x * yz;
    v.y *= s.y * yz;
    v.z *= s.z * yz;
    v.w *= s.w * yz;
    data[i] = v;
}

/// The half spectrum times the same separable table (kx <= W/2 indexes sx directly).
__global__ void ComplexBySeparable(cufftComplex* __restrict__ spectrum, const float* __restrict__ sx,
    const float* __restrict__ sy, const float* __restrict__ sz, int half, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= half) {
        return;
    }
    const int y = blockIdx.y;
    const int z = blockIdx.z;
    const float s = sx[x] * sy[y] * sz[z];
    const size_t i = static_cast<size_t>(half) * (y + static_cast<size_t>(H) * z) + x;
    cufftComplex c = spectrum[i];
    c.x *= s;
    c.y *= s;
    spectrum[i] = c;
}

dim3 LineGrid(size_t perLine, unsigned block, int H, int D)
{
    return dim3(Blocks(perLine, block), static_cast<unsigned>(H), static_cast<unsigned>(D));
}

template <typename T>
T* DeviceCopy(const std::vector<T>& host, const char* what)
{
    T* device = nullptr;
    Check(cudaMalloc(&device, host.size() * sizeof(T)), what);
    Check(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    return device;
}

} // namespace

void MultiplySeparable(float* deviceData, const float* deviceSx, const float* deviceSy, const float* deviceSz,
    int width, int height, int depth)
{
    const int D = depth > 0 ? depth : 1;
    const int quads = width / 4;
    const unsigned block = 64;
    SeparableProduct<<<LineGrid(static_cast<size_t>(quads), block, height, D), block>>>(
        reinterpret_cast<float4*>(deviceData), reinterpret_cast<const float4*>(deviceSx), deviceSy, deviceSz, quads,
        height);
    Check(cudaGetLastError(), "SeparableProduct launch");
    Check(cudaDeviceSynchronize(), "SeparableProduct");
}

// ---------------------------------------------------------------------------
// T2 through cuFFT
// ---------------------------------------------------------------------------

struct CufftGaussianFilter::Impl {
    int width = 0, height = 0, depth = 0;
    IntInput type = IntInput::U16;
    size_t half = 0; ///< complex coefficients a line of the half spectrum
    size_t spectrumBytes = 0, workBytes = 0;
    cufftHandle forward = 0, inverse = 0;
    void* work = nullptr;
    cufftComplex* spectrum = nullptr;
    float* sx = nullptr;
    float* sy = nullptr;
    float* sz = nullptr;
    float* params = nullptr; ///< { 1/N, offset }
    unsigned long long* sum = nullptr;

    ~Impl()
    {
        if (forward != 0) {
            cufftDestroy(forward);
        }
        if (inverse != 0) {
            cufftDestroy(inverse);
        }
        cudaFree(work);
        cudaFree(spectrum);
        cudaFree(sx);
        cudaFree(sy);
        cudaFree(sz);
        cudaFree(params);
        cudaFree(sum);
    }
};

CufftGaussianFilter::CufftGaussianFilter(int width, int height, int depth, IntInput type,
    const std::vector<float>& sx, const std::vector<float>& sy, const std::vector<float>& sz):
    _impl(std::make_unique<Impl>())
{
    Impl& s = *_impl;
    s.width = width;
    s.height = height;
    s.depth = depth > 0 ? depth : 1;
    s.type = type;
    s.half = static_cast<size_t>(width) / 2 + 1;
    const size_t count = static_cast<size_t>(width) * height * s.depth;
    s.spectrumBytes = s.half * height * s.depth * sizeof(cufftComplex);
    Check(cudaMalloc(&s.spectrum, s.spectrumBytes), "cudaMalloc(spectrum)");
    s.sx = DeviceCopy(sx, "upload sx");
    s.sy = DeviceCopy(sy, "upload sy");
    s.sz = DeviceCopy(sz, "upload sz");
    const float params[2] = { static_cast<float>(1.0 / static_cast<double>(count)), 0.0f };
    Check(cudaMalloc(&s.params, sizeof(params)), "cudaMalloc(params)");
    Check(cudaMemcpy(s.params, params, sizeof(params), cudaMemcpyHostToDevice), "upload params");
    Check(cudaMalloc(&s.sum, sizeof(unsigned long long)), "cudaMalloc(sum)");

    Check(cufftCreate(&s.forward), "cufftCreate");
    Check(cufftCreate(&s.inverse), "cufftCreate");
    Check(cufftSetAutoAllocation(s.forward, 0), "cufftSetAutoAllocation");
    Check(cufftSetAutoAllocation(s.inverse, 0), "cufftSetAutoAllocation");
    void* loadInfo = s.params + 1; // the load callback reads the offset alone
    void* storeInfo = s.params;
    const bool u8 = type == IntInput::U8;
    Check(cufftXtSetJITCallback(s.forward, u8 ? "rapidht_load_u8_lto" : "rapidht_load_u16_lto", rapidht_lto_fatbin,
              sizeof(rapidht_lto_fatbin), CUFFT_CB_LD_REAL, &loadInfo),
        "cufftXtSetJITCallback(load)");
    Check(cufftXtSetJITCallback(s.inverse, u8 ? "rapidht_store_u8_offset_lto" : "rapidht_store_u16_offset_lto",
              rapidht_lto_fatbin, sizeof(rapidht_lto_fatbin), CUFFT_CB_ST_REAL, &storeInfo),
        "cufftXtSetJITCallback(store)");
    size_t forwardBytes = 0, inverseBytes = 0;
    Check(cufftMakePlan3d(s.forward, s.depth, height, width, CUFFT_R2C, &forwardBytes), "cufftMakePlan3d(R2C)");
    Check(cufftMakePlan3d(s.inverse, s.depth, height, width, CUFFT_C2R, &inverseBytes), "cufftMakePlan3d(C2R)");
    s.workBytes = std::max<size_t>(std::max(forwardBytes, inverseBytes), 1);
    Check(cudaMalloc(&s.work, s.workBytes), "cudaMalloc(work)");
    Check(cufftSetWorkArea(s.forward, s.work), "cufftSetWorkArea");
    Check(cufftSetWorkArea(s.inverse, s.work), "cufftSetWorkArea");
}

CufftGaussianFilter::~CufftGaussianFilter() = default;

void CufftGaussianFilter::Apply(const void* deviceIn, void* deviceOut)
{
    Impl& s = *_impl;
    IntegerMeanOffset(deviceIn, s.type, static_cast<size_t>(s.width), static_cast<size_t>(s.height) * s.depth, 64,
        s.sum, s.params + 1);
    Check(cufftExecR2C(s.forward, static_cast<cufftReal*>(const_cast<void*>(deviceIn)), s.spectrum), "cufftExecR2C");
    const unsigned block = 128;
    ComplexBySeparable<<<LineGrid(s.half, block, s.height, s.depth), block>>>(
        s.spectrum, s.sx, s.sy, s.sz, static_cast<int>(s.half), s.height);
    Check(cudaGetLastError(), "ComplexBySeparable launch");
    Check(cufftExecC2R(s.inverse, s.spectrum, static_cast<cufftReal*>(deviceOut)), "cufftExecC2R");
    Check(cudaDeviceSynchronize(), "CufftGaussianFilter::Apply");
}

size_t CufftGaussianFilter::ExtraDeviceBytes() const
{
    const Impl& s = *_impl;
    return s.spectrumBytes + s.workBytes;
}

// ---------------------------------------------------------------------------
// T2 in space
// ---------------------------------------------------------------------------

namespace {

constexpr int kMaxRadius = 6;

struct Taps {
    float c[kMaxRadius + 1];
};

/*
 * Along x: a block a line, the line in shared memory (W <= 1024, a power of
 * two, so the wrap is a mask), four outputs a thread read and written as
 * quads.
 */
template <typename TIn, typename TIn4, int R>
__global__ void SpatialX(const TIn4* __restrict__ in, float4* __restrict__ out, int W, Taps taps)
{
    __shared__ float row[1024];
    const size_t line = blockIdx.x;
    const int quads = W / 4;
    const TIn4* src = in + line * quads;
    for (int q = threadIdx.x; q < quads; q += blockDim.x) {
        const TIn4 v = src[q];
        row[4 * q] = static_cast<float>(v.x);
        row[4 * q + 1] = static_cast<float>(v.y);
        row[4 * q + 2] = static_cast<float>(v.z);
        row[4 * q + 3] = static_cast<float>(v.w);
    }
    __syncthreads();
    const int mask = W - 1;
    for (int q = threadIdx.x; q < quads; q += blockDim.x) {
        float acc[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int x = 4 * q + j;
            float a = taps.c[0] * row[x];
#pragma unroll
            for (int d = 1; d <= R; ++d) {
                a += taps.c[d] * (row[(x - d) & mask] + row[(x + d) & mask]);
            }
            acc[j] = a;
        }
        out[line * quads + q] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
}

template <typename TOut>
__device__ __forceinline__ TOut Store(float v)
{
    if constexpr (sizeof(TOut) == sizeof(float)) {
        return v;
    } else {
        return ToInteger<TOut>(v);
    }
}

/*
 * Along y or z: a thread a column (consecutive threads, consecutive x, so
 * every step reads a coalesced row), walking it with the 2R + 1 values of the
 * window in registers. Safe in place: the value read ahead, at p + 1 + R, is
 * never one already written, except past the end, where the wrap needs the
 * first R values of the column -- kept from the start.
 */
template <typename TOut, int R>
__global__ void SpatialColumn(const float* data, TOut* out, int W, int columns, size_t otherStride, size_t step,
    int L, Taps taps)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= columns) {
        return;
    }
    const size_t base = static_cast<size_t>(c % W) + static_cast<size_t>(c / W) * otherStride;
    const float* col = data + base;
    TOut* dst = out + base;
    float w[2 * R + 1];
#pragma unroll
    for (int i = 0; i <= 2 * R; ++i) {
        const int p = (i - R + L) % L;
        w[i] = col[static_cast<size_t>(p) * step];
    }
    float head[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        head[i] = w[R + i];
    }
    auto at = [&](int p) {
        float a = taps.c[0] * w[R];
#pragma unroll
        for (int d = 1; d <= R; ++d) {
            a += taps.c[d] * (w[R - d] + w[R + d]);
        }
        dst[static_cast<size_t>(p) * step] = Store<TOut>(a);
    };
    auto shift = [&](float next) {
#pragma unroll
        for (int i = 0; i < 2 * R; ++i) {
            w[i] = w[i + 1];
        }
        w[2 * R] = next;
    };
    int p = 0;
    for (; p < L - R - 1; ++p) {
        const float next = col[static_cast<size_t>(p + 1 + R) * step];
        at(p);
        shift(next);
    }
#pragma unroll
    for (int t = 0; t < R; ++t) {
        at(p + t);
        shift(head[t]);
    }
    at(p + R);
}

template <int R>
void LaunchSpatial(const void* in, IntInput type, float* buffer, void* out, int W, int H, int D, const Taps& taps)
{
    const size_t lines = static_cast<size_t>(H) * D;
    if (type == IntInput::U8) {
        SpatialX<std::uint8_t, uchar4, R><<<static_cast<unsigned>(lines), 128>>>(
            static_cast<const uchar4*>(in), reinterpret_cast<float4*>(buffer), W, taps);
    } else {
        SpatialX<std::uint16_t, ushort4, R><<<static_cast<unsigned>(lines), 128>>>(
            static_cast<const ushort4*>(in), reinterpret_cast<float4*>(buffer), W, taps);
    }
    Check(cudaGetLastError(), "SpatialX launch");
    const unsigned block = 128;
    const size_t plane = static_cast<size_t>(W) * H;
    // y in place: columns (x, z)
    SpatialColumn<float, R><<<Blocks(static_cast<size_t>(W) * D, block), block>>>(
        buffer, buffer, W, W * D, plane, static_cast<size_t>(W), H, taps);
    Check(cudaGetLastError(), "SpatialColumn(y) launch");
    // z into the integers: columns (x, y)
    if (type == IntInput::U8) {
        SpatialColumn<std::uint8_t, R><<<Blocks(plane, block), block>>>(
            buffer, static_cast<std::uint8_t*>(out), W, W * H, static_cast<size_t>(W), plane, D, taps);
    } else {
        SpatialColumn<std::uint16_t, R><<<Blocks(plane, block), block>>>(
            buffer, static_cast<std::uint16_t*>(out), W, W * H, static_cast<size_t>(W), plane, D, taps);
    }
    Check(cudaGetLastError(), "SpatialColumn(z) launch");
}

} // namespace

struct SpatialGaussian::Impl {
    int width = 0, height = 0, depth = 0;
    IntInput type = IntInput::U16;
    int radius = 0;
    Taps taps {};
    float* buffer = nullptr;
    size_t bufferBytes = 0;

    ~Impl() { cudaFree(buffer); }
};

SpatialGaussian::SpatialGaussian(int width, int height, int depth, IntInput type, const std::vector<double>& taps):
    _impl(std::make_unique<Impl>())
{
    Impl& s = *_impl;
    s.width = width;
    s.height = height;
    s.depth = depth > 0 ? depth : 1;
    s.type = type;
    s.radius = static_cast<int>(taps.size()) - 1;
    if (s.radius != 3 && s.radius != 6) {
        throw std::runtime_error("SpatialGaussian: sigma 1 or 2 only (radius 3 or 6)");
    }
    if (width > 1024 || width < 4 || 2 * s.radius + 1 > std::min(height, s.depth)) {
        throw std::runtime_error("SpatialGaussian: unsupported extent");
    }
    for (int d = 0; d <= s.radius; ++d) {
        s.taps.c[d] = static_cast<float>(taps[static_cast<size_t>(d)]);
    }
    s.bufferBytes = static_cast<size_t>(width) * height * s.depth * sizeof(float);
    Check(cudaMalloc(&s.buffer, s.bufferBytes), "cudaMalloc(spatial buffer)");
}

SpatialGaussian::~SpatialGaussian() = default;

void SpatialGaussian::Apply(const void* deviceIn, void* deviceOut)
{
    Impl& s = *_impl;
    if (s.radius == 3) {
        LaunchSpatial<3>(deviceIn, s.type, s.buffer, deviceOut, s.width, s.height, s.depth, s.taps);
    } else {
        LaunchSpatial<6>(deviceIn, s.type, s.buffer, deviceOut, s.width, s.height, s.depth, s.taps);
    }
    Check(cudaDeviceSynchronize(), "SpatialGaussian::Apply");
}

size_t SpatialGaussian::ExtraDeviceBytes() const { return _impl->bufferBytes; }

// ---------------------------------------------------------------------------
// T3: radial bins
// ---------------------------------------------------------------------------

namespace {

constexpr int kLinesPerFold = 16;

/*
 * One block takes kLinesPerFold lines at a time; a thread takes a run of
 * consecutive x of a line (coalesced across the warp), adds the values of one
 * bin in a register and flushes to a float bin in shared memory when the bin
 * changes. Every kLinesPerFold lines the float bins are folded into double
 * ones, so a float bin never sums more than a few thousand values; the block
 * adds its double bins to the global ones once, at the end.
 */
template <typename Value>
__global__ void RadialBins(Value value, int lineLength, int W, int H, int D, long long longest, double* bins)
{
    __shared__ float partial[kRadialBins];
    __shared__ double total[kRadialBins];
    for (int i = threadIdx.x; i < kRadialBins; i += blockDim.x) {
        partial[i] = 0.0f;
        total[i] = 0.0;
    }
    __syncthreads();
    const int lines = H * D;
    const int perThread = (lineLength + blockDim.x - 1) / blockDim.x;
    const int x0 = threadIdx.x * perThread;
    const int x1 = min(x0 + perThread, lineLength);
    for (int first = blockIdx.x * kLinesPerFold; first < lines; first += gridDim.x * kLinesPerFold) {
        const int last = min(first + kLinesPerFold, lines);
        for (int line = first; line < last; ++line) {
            const long long ky = ScaledFrequency(line % H, H, longest);
            const long long kz = ScaledFrequency(line / H, D, longest);
            const long long yz = ky * ky + kz * kz;
            int runBin = -1;
            float run = 0.0f;
            for (int x = x0; x < x1; ++x) {
                const long long kx = ScaledFrequency(x, W, longest);
                const int b = RadialBin(kx * kx + yz, longest);
                if (b != runBin) {
                    if (runBin >= 0) {
                        atomicAdd(&partial[runBin], run);
                    }
                    runBin = b;
                    run = 0.0f;
                }
                run += value(line, x);
            }
            if (runBin >= 0) {
                atomicAdd(&partial[runBin], run);
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < kRadialBins; i += blockDim.x) {
            total[i] += partial[i];
            partial[i] = 0.0f;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < kRadialBins; i += blockDim.x) {
        if (total[i] != 0.0) {
            atomicAdd(&bins[i], total[i]);
        }
    }
}

struct HartleySquare {
    const float* data;
    int W;
    __device__ float operator()(int line, int x) const
    {
        const float h = data[static_cast<size_t>(line) * W + x];
        return h * h;
    }
};

/// |F|^2 of the half spectrum, twice where a column stands for its mirror; the
/// zero frequency gets offset * N back first, in double.
struct HalfSpectrumSquare {
    const cufftComplex* spectrum;
    int half, W;
    const float* offset;
    double count;
    __device__ float operator()(int line, int x) const
    {
        const cufftComplex c = spectrum[static_cast<size_t>(line) * half + x];
        if (line == 0 && x == 0) {
            const double re = static_cast<double>(c.x) + static_cast<double>(*offset) * count;
            return static_cast<float>(re * re + static_cast<double>(c.y) * c.y);
        }
        const float p = c.x * c.x + c.y * c.y;
        return (x == 0 || 2 * x == W) ? p : 2.0f * p;
    }
};

template <typename Value>
void LaunchRadialBins(const Value& value, int lineLength, int W, int H, int D, double* deviceBins, double* hostBins)
{
    int device = 0, sms = 0;
    Check(cudaGetDevice(&device), "cudaGetDevice");
    Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "cudaDeviceGetAttribute");
    const long long longest = std::max({ W, H, D });
    const size_t lines = static_cast<size_t>(H) * D;
    const unsigned grid = static_cast<unsigned>(
        std::min<size_t>((lines + kLinesPerFold - 1) / kLinesPerFold, static_cast<size_t>(sms) * 8));
    Check(cudaMemsetAsync(deviceBins, 0, kRadialBins * sizeof(double)), "cudaMemsetAsync(bins)");
    RadialBins<<<grid, 128>>>(value, lineLength, W, H, D, longest, deviceBins);
    Check(cudaGetLastError(), "RadialBins launch");
    Check(cudaMemcpy(hostBins, deviceBins, kRadialBins * sizeof(double), cudaMemcpyDeviceToHost), "download bins");
}

} // namespace

void RadialPowerHartley(const float* deviceData, int width, int height, int depth, double* deviceBins,
    double* hostBins)
{
    const int D = depth > 0 ? depth : 1;
    LaunchRadialBins(HartleySquare { deviceData, width }, width, width, height, D, deviceBins, hostBins);
}

struct CufftPowerProfile::Impl {
    int width = 0, height = 0, depth = 0;
    IntInput type = IntInput::U16;
    size_t spectrumBytes = 0, workBytes = 0;
    cufftHandle plan = 0;
    cufftComplex* spectrum = nullptr;
    float* offset = nullptr;
    unsigned long long* sum = nullptr;
    double* bins = nullptr;

    ~Impl()
    {
        if (plan != 0) {
            cufftDestroy(plan);
        }
        cudaFree(spectrum);
        cudaFree(offset);
        cudaFree(sum);
        cudaFree(bins);
    }
};

CufftPowerProfile::CufftPowerProfile(int width, int height, int depth, IntInput type):
    _impl(std::make_unique<Impl>())
{
    Impl& s = *_impl;
    s.width = width;
    s.height = height;
    s.depth = depth > 0 ? depth : 1;
    s.type = type;
    s.spectrumBytes = (static_cast<size_t>(width) / 2 + 1) * height * s.depth * sizeof(cufftComplex);
    Check(cudaMalloc(&s.spectrum, s.spectrumBytes), "cudaMalloc(spectrum)");
    Check(cudaMalloc(&s.offset, sizeof(float)), "cudaMalloc(offset)");
    Check(cudaMemset(s.offset, 0, sizeof(float)), "cudaMemset(offset)");
    Check(cudaMalloc(&s.sum, sizeof(unsigned long long)), "cudaMalloc(sum)");
    Check(cudaMalloc(&s.bins, kRadialBins * sizeof(double)), "cudaMalloc(bins)");
    Check(cufftCreate(&s.plan), "cufftCreate");
    void* info = s.offset;
    Check(cufftXtSetJITCallback(s.plan, type == IntInput::U8 ? "rapidht_load_u8_lto" : "rapidht_load_u16_lto",
              rapidht_lto_fatbin, sizeof(rapidht_lto_fatbin), CUFFT_CB_LD_REAL, &info),
        "cufftXtSetJITCallback(load)");
    Check(cufftMakePlan3d(s.plan, s.depth, height, width, CUFFT_R2C, &s.workBytes), "cufftMakePlan3d(R2C)");
}

CufftPowerProfile::~CufftPowerProfile() = default;

void CufftPowerProfile::Apply(const void* deviceIn, double* hostBins)
{
    Impl& s = *_impl;
    IntegerMeanOffset(deviceIn, s.type, static_cast<size_t>(s.width), static_cast<size_t>(s.height) * s.depth, 64,
        s.sum, s.offset);
    Check(cufftExecR2C(s.plan, static_cast<cufftReal*>(const_cast<void*>(deviceIn)), s.spectrum), "cufftExecR2C");
    const int half = s.width / 2 + 1;
    const double count = static_cast<double>(s.width) * s.height * s.depth;
    LaunchRadialBins(HalfSpectrumSquare { s.spectrum, half, s.width, s.offset, count }, half, s.width, s.height,
        s.depth, s.bins, hostBins);
}

size_t CufftPowerProfile::ExtraDeviceBytes() const
{
    const Impl& s = *_impl;
    return s.spectrumBytes + s.workBytes;
}

} // namespace rapidht_bench
