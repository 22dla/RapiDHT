/*
 * Project: RapiDHT
 * File: benchmarks/stride_probe.cu
 * Brief: How fast the memory moves line sets along y and along z, by shape.
 *
 *   ./stride_probe            (512^3 float, about a minute)
 *
 * R-4, toward P2+. At 512^3 the z pass is bound by its loads and stores alone
 * (the io-only mode of the kernel, RAPIDHT_R4_ABLATE=io until step 5.1: 3.62 ms
 * against 3.10 along y, the arithmetic 2.52 ms hidden under it), and handing
 * neighbouring blocks different rows made every axis slower
 * (RAPIDHT_R4_ORDER=across, 2026-11-20_*_r4_zord_*). Before the
 * kernel is reshaped, this probe asks which shape of line set the memory
 * moves fastest along z. A block takes kLanesX consecutive x and kRowsY
 * consecutive rows of the other lane axis -- kLanesX * kRowsY lines of n
 * values -- reads them into registers and writes them back unchanged, batch
 * after batch, neighbouring x groups to neighbouring blocks as in the real
 * kernel. Shapes:
 *   16 x 1  the real kernel today: 64 bytes contiguous per row of a line set;
 *   32 x 1  128 bytes contiguous: half the DRAM rows and pages per byte;
 *   16 x 2  64 bytes twice, 2 KiB apart: the same page, a different DRAM row;
 *           helps if address translation is what costs, not if DRAM rows are;
 *   64 x 1  256 bytes contiguous.
 * Each shape is timed along z (rows 1 MiB apart) and along y (2 KiB apart),
 * with 1, 2 and 4 blocks to an SM; the real kernel runs 1. The pass is the
 * identity, so the volume stays valid; nothing is checked but the time.
 *
 *   ./stride_probe int        (R-5 step 1.1, 512^3, about two minutes)
 *
 * R-5, step 1.1: the first pass of a transform of an integer volume reads
 * uint8 / uint16 and writes float into another buffer. Along z a batch of L
 * lines reads L * 2 bytes contiguous per row for uint16, L for uint8 -- half or
 * a quarter of the float case, where 32 lines (128 bytes) were needed to move
 * at the speed of a copy. Mode "int" times exactly that access: L = 16, 32, 64,
 * 128 consecutive lines, read whole (as the real kernel reads a batch before it
 * computes), converted, written as float; along z, y and x; 1 and 2 blocks to
 * an SM. float input is timed the same way for reference. The floor is a copy
 * with the same traffic (cudaMemcpy of half of it: 0.75 GiB for uint16, 0.625
 * for uint8, 1 GiB for float).
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

#define PROBE_CHECK(call)                                                            \
    do {                                                                             \
        const cudaError_t probeError = (call);                                       \
        if (probeError != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(probeError)); \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

constexpr int kThreads = 256;

/// One batch: kLanesX x kRowsY lines of kN values. Element (k, row, x) of the
/// batch sits at base + row * otherStride + k * axisStride + x.
template <int kN, int kLanesX, int kRowsY>
__global__ void __launch_bounds__(kThreads) MoveLines(
    float* data, size_t axisStride, size_t otherStride, int groupsX, int batches)
{
    constexpr int kQuadsX = kLanesX / 4;
    constexpr int kItems = kN * kRowsY * kQuadsX / kThreads;
    static_assert(kItems * kThreads == kN * kRowsY * kQuadsX, "whole float4 items per thread");
    float4 v[kItems];
    for (int batch = static_cast<int>(blockIdx.x); batch < batches; batch += static_cast<int>(gridDim.x)) {
        const int xg = batch % groupsX;
        const int og = batch / groupsX;
        float* base = data + static_cast<size_t>(og) * kRowsY * otherStride + static_cast<size_t>(xg) * kLanesX;
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
            // x fastest, then the rows of one plane, then k: consecutive
            // threads read contiguous bytes.
            const int item = static_cast<int>(threadIdx.x) + i * kThreads;
            const int q = item % kQuadsX;
            const int r = item / kQuadsX;
            const int row = r % kRowsY;
            const int k = r / kRowsY;
            v[i] = *reinterpret_cast<const float4*>(
                base + static_cast<size_t>(row) * otherStride + static_cast<size_t>(k) * axisStride + q * 4);
        }
        __syncthreads(); // as the real kernel: the batch is in before it goes out
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
            const int item = static_cast<int>(threadIdx.x) + i * kThreads;
            const int q = item % kQuadsX;
            const int r = item / kQuadsX;
            const int row = r % kRowsY;
            const int k = r / kRowsY;
            *reinterpret_cast<float4*>(
                base + static_cast<size_t>(row) * otherStride + static_cast<size_t>(k) * axisStride + q * 4)
                = v[i];
        }
    }
}

struct Timing {
    float min = 0.0f;
    float median = 0.0f;
};

template <typename Launch>
Timing Time(Launch launch, int repetitions)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    PROBE_CHECK(cudaEventCreate(&start));
    PROBE_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 3; ++i) {
        launch();
    }
    PROBE_CHECK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < repetitions; ++i) {
        PROBE_CHECK(cudaEventRecord(start));
        launch();
        PROBE_CHECK(cudaEventRecord(stop));
        PROBE_CHECK(cudaEventSynchronize(stop));
        float t = 0.0f;
        PROBE_CHECK(cudaEventElapsedTime(&t, start, stop));
        ms.push_back(t);
    }
    PROBE_CHECK(cudaGetLastError());
    PROBE_CHECK(cudaEventDestroy(start));
    PROBE_CHECK(cudaEventDestroy(stop));
    std::sort(ms.begin(), ms.end());
    return { ms.front(), ms[ms.size() / 2] };
}

template <int kN, int kLanesX, int kRowsY>
void Probe(float* data, int sms)
{
    constexpr size_t n = kN;
    const int groupsX = kN / kLanesX;
    // One batch per (x group, row group): along z the rows are y (stride n),
    // along y they are z (stride n*n); either way the batches cover all n*n
    // lines once.
    const int batches = groupsX * (kN / kRowsY);
    struct Axis {
        const char* name;
        size_t axisStride;
        size_t otherStride;
    };
    const Axis axes[] = { { "z", n * n, n }, { "y", n, n * n } };
    for (const Axis& axis : axes) {
        std::printf("  %2d x %d   %s", kLanesX, kRowsY, axis.name);
        for (const int perSm : { 1, 2, 4 }) {
            int fit = 0;
            PROBE_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, MoveLines<kN, kLanesX, kRowsY>, kThreads, 0));
            if (fit < perSm) {
                std::printf("   %d/SM        -     ", perSm);
                continue;
            }
            const unsigned grid = static_cast<unsigned>(std::min(batches, sms * perSm));
            const Timing t = Time(
                [&] {
                    MoveLines<kN, kLanesX, kRowsY><<<grid, kThreads>>>(
                        data, axis.axisStride, axis.otherStride, groupsX, batches);
                },
                20);
            std::printf("   %d/SM %6.3f (%6.3f)", perSm, t.min, t.median);
        }
        std::printf("\n");
    }
}

/*
 * R-5 step 1.1. A batch is L lines of kN values of the transformed axis.
 * The contiguous direction of memory is the lane (x) along z and y, the
 * transformed axis itself along x; a "quad" is four neighbours in it, read as
 * one uchar4 / ushort4 / float4 and written as one float4. A batch is read
 * whole into registers and then written -- in parts of at most kMaxItems quads
 * a thread when it does not fit (128 lines of 512).
 *
 *   element (c, s) of a batch is at base + c * 1 + s * strideS, c contiguous,
 *   along z / y: c = lane (L of them), s = k (kN, stride n*n or n);
 *   along x:     c = k (kN, stride 1), s = lane (L rows, stride n).
 */
template <typename TIn>
struct Quad;
template <>
struct Quad<unsigned char> {
    using Type = uchar4;
};
template <>
struct Quad<unsigned short> {
    using Type = ushort4;
};
template <>
struct Quad<float> {
    using Type = float4;
};

constexpr int kMaxItems = 32;

template <typename TIn, int kC, int kS>
__global__ void __launch_bounds__(kThreads) ConvertLines(const TIn* __restrict__ in, float* __restrict__ out,
    size_t strideS, size_t groupStrideA, int groupsA, size_t groupStrideB, int batches)
{
    using Q = typename Quad<TIn>::Type;
    constexpr int kQuadsC = kC / 4;
    constexpr int kTotal = kQuadsC * kS / kThreads; // quads a thread moves per batch
    static_assert(kTotal * kThreads == kQuadsC * kS, "whole quads per thread");
    constexpr int kItems = kTotal < kMaxItems ? kTotal : kMaxItems;
    constexpr int kParts = kTotal / kItems;
    static_assert(kParts * kItems == kTotal, "whole parts");
    float4 v[kItems];
    for (int batch = static_cast<int>(blockIdx.x); batch < batches; batch += static_cast<int>(gridDim.x)) {
        // Neighbouring batches to neighbouring blocks: the A groups (along the
        // contiguous lanes, or rows along x) first, as the real kernel does.
        const size_t base = static_cast<size_t>(batch % groupsA) * groupStrideA
                          + static_cast<size_t>(batch / groupsA) * groupStrideB;
        for (int part = 0; part < kParts; ++part) {
#pragma unroll
            for (int i = 0; i < kItems; ++i) {
                const int item = static_cast<int>(threadIdx.x) + (part * kItems + i) * kThreads;
                const int q = item % kQuadsC;
                const int sIndex = item / kQuadsC;
                const Q x = *reinterpret_cast<const Q*>(in + base + static_cast<size_t>(sIndex) * strideS + q * 4);
                v[i] = make_float4(static_cast<float>(x.x), static_cast<float>(x.y), static_cast<float>(x.z),
                    static_cast<float>(x.w));
            }
            __syncthreads(); // the part is in before it goes out
#pragma unroll
            for (int i = 0; i < kItems; ++i) {
                const int item = static_cast<int>(threadIdx.x) + (part * kItems + i) * kThreads;
                const int q = item % kQuadsC;
                const int sIndex = item / kQuadsC;
                *reinterpret_cast<float4*>(out + base + static_cast<size_t>(sIndex) * strideS + q * 4) = v[i];
            }
        }
    }
}

template <typename TIn, int kN, int kL>
void ProbeConvert(const TIn* in, float* out, int sms)
{
    constexpr size_t n = kN;
    const int groups = kN / kL;
    struct Axis {
        const char* name;
        bool alongX;
        size_t strideS, groupStrideA, groupStrideB;
        int groupsA;
    };
    // z: lanes are x (groups of L), batches over (x group, y); y: over (x group, z);
    // x: lanes are y rows (groups of L), batches over (y group, z).
    const Axis axes[] = {
        { "z", false, n * n, kL, n, groups },
        { "y", false, n, kL, n * n, groups },
        { "x", true, n, kL * n, n * n, groups },
    };
    const int batches = groups * kN;
    for (const Axis& axis : axes) {
        std::printf("  %3d   %s", kL, axis.name);
        for (const int perSm : { 1, 2 }) {
            int fit = 0;
            if (axis.alongX) {
                PROBE_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, ConvertLines<TIn, kN, kL>, kThreads, 0));
            } else {
                PROBE_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, ConvertLines<TIn, kL, kN>, kThreads, 0));
            }
            if (fit < perSm) {
                std::printf("   %d/SM        -        ", perSm);
                continue;
            }
            const unsigned grid = static_cast<unsigned>(std::min(batches, sms * perSm));
            const Timing t = Time(
                [&] {
                    if (axis.alongX) {
                        ConvertLines<TIn, kN, kL><<<grid, kThreads>>>(
                            in, out, axis.strideS, axis.groupStrideA, axis.groupsA, axis.groupStrideB, batches);
                    } else {
                        ConvertLines<TIn, kL, kN><<<grid, kThreads>>>(
                            in, out, axis.strideS, axis.groupStrideA, axis.groupsA, axis.groupStrideB, batches);
                    }
                },
                20);
            std::printf("   %d/SM %6.3f (%6.3f)", perSm, t.min, t.median);
        }
        std::printf("\n");
    }
}

template <typename TIn>
void ProbeConvertType(const char* name, const void* in, float* out, float* scratch, size_t count, int sms)
{
    constexpr int kN = 512;
    const size_t traffic = count * (sizeof(TIn) + sizeof(float));
    const Timing copy = Time(
        [&] { PROBE_CHECK(cudaMemcpyAsync(scratch, out, traffic / 2, cudaMemcpyDeviceToDevice)); }, 20);
    std::printf("%s -> float, %.3f GiB of traffic\n", name, static_cast<double>(traffic) / (1u << 30));
    std::printf("  floor: copy with the same traffic   %6.3f (%6.3f)\n", copy.min, copy.median);
    std::printf("  lines axis  blocks to an SM\n");
    const TIn* typed = static_cast<const TIn*>(in);
    ProbeConvert<TIn, kN, 16>(typed, out, sms);
    ProbeConvert<TIn, kN, 32>(typed, out, sms);
    ProbeConvert<TIn, kN, 64>(typed, out, sms);
    ProbeConvert<TIn, kN, 128>(typed, out, sms);
}

int ConvertMode(const cudaDeviceProp& prop)
{
    constexpr size_t kN = 512;
    const size_t count = kN * kN * kN;
    void* in = nullptr;
    float* out = nullptr;
    float* scratch = nullptr; // the far side of the floor's copy
    PROBE_CHECK(cudaMalloc(&in, count * sizeof(float)));
    PROBE_CHECK(cudaMalloc(&out, count * sizeof(float)));
    PROBE_CHECK(cudaMalloc(&scratch, count * sizeof(float)));
    PROBE_CHECK(cudaMemset(in, 0, count * sizeof(float)));
    PROBE_CHECK(cudaMemset(out, 0, count * sizeof(float)));
    std::printf("%s, %d SMs; %zu^3, read integers, write float to another buffer; ms, min (median) of 20\n",
        prop.name, prop.multiProcessorCount, kN);
    ProbeConvertType<unsigned short>("uint16", in, out, scratch, count, prop.multiProcessorCount);
    ProbeConvertType<unsigned char>("uint8", in, out, scratch, count, prop.multiProcessorCount);
    ProbeConvertType<float>("float (reference)", in, out, scratch, count, prop.multiProcessorCount);
    PROBE_CHECK(cudaFree(in));
    PROBE_CHECK(cudaFree(out));
    PROBE_CHECK(cudaFree(scratch));
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    constexpr int kN = 512;
    const size_t count = static_cast<size_t>(kN) * kN * kN;
    int device = 0;
    cudaDeviceProp prop {};
    PROBE_CHECK(cudaGetDevice(&device));
    PROBE_CHECK(cudaGetDeviceProperties(&prop, device));
    if (argc > 1 && std::string(argv[1]) == "int") {
        return ConvertMode(prop);
    }
    float* a = nullptr;
    float* b = nullptr;
    PROBE_CHECK(cudaMalloc(&a, count * sizeof(float)));
    PROBE_CHECK(cudaMalloc(&b, count * sizeof(float)));
    PROBE_CHECK(cudaMemset(a, 0, count * sizeof(float)));
    PROBE_CHECK(cudaMemset(b, 0, count * sizeof(float)));

    std::printf("%s, %d SMs; %d^3 float, one pass reads and writes 512 MiB; ms, min (median) of 20\n", prop.name,
        prop.multiProcessorCount, kN);
    const Timing copy = Time([&] { PROBE_CHECK(cudaMemcpyAsync(b, a, count * sizeof(float), cudaMemcpyDeviceToDevice)); }, 20);
    std::printf("  copy (cudaMemcpy, the floor)   %6.3f (%6.3f)\n", copy.min, copy.median);
    std::printf("  lanes    axis  blocks to an SM\n");
    Probe<kN, 16, 1>(a, prop.multiProcessorCount);
    Probe<kN, 32, 1>(a, prop.multiProcessorCount);
    Probe<kN, 16, 2>(a, prop.multiProcessorCount);
    Probe<kN, 64, 1>(a, prop.multiProcessorCount);

    PROBE_CHECK(cudaFree(a));
    PROBE_CHECK(cudaFree(b));
    return 0;
}
