/*
 * Project: RapiDHT
 * File: benchmarks/stride_probe.cu
 * Brief: How fast the memory moves line sets along y and along z, by shape.
 *
 *   ./stride_probe            (512^3 float, about a minute)
 *
 * R-4, toward P2+. At 512^3 the z pass is bound by its loads and stores alone
 * (RAPIDHT_R4_ABLATE=io: 3.62 ms against 3.10 along y, the arithmetic 2.52 ms
 * hidden under it), and handing neighbouring blocks different rows made every
 * axis slower (RAPIDHT_R4_ORDER=across, 2026-11-20_*_r4_zord_*). Before the
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
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
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

} // namespace

int main()
{
    constexpr int kN = 512;
    const size_t count = static_cast<size_t>(kN) * kN * kN;
    int device = 0;
    cudaDeviceProp prop {};
    PROBE_CHECK(cudaGetDevice(&device));
    PROBE_CHECK(cudaGetDeviceProperties(&prop, device));
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
