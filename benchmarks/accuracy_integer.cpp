/*
 * Project: RapiDHT
 * File: benchmarks/accuracy_integer.cpp
 * Brief: How accurate is a float transform of an integer tomogram, and against what?
 *
 *   ./accuracy_integer 128 256 512                         white integer noise, n^3
 *   ./accuracy_integer --synthetic cylinder u16|u8 128 512 a synthetic tomogram, n^3
 *   ./accuracy_integer --raw FILE WxHxD u16|u8 [--crop X0 Y0 Z0 W H D]
 *                                                          a volume from a file
 *
 * R-5 of docs/BACKLOG.md, first question: would integer arithmetic beat the
 * error of the float transform on uint8 / uint16 volumes? Integer arithmetic
 * cannot compute the transform exactly -- cas(2 pi k n / N) is irrational, so
 * it would be fixed point -- and it can only help if the float error matters.
 * This program puts the float error next to two floors that no arithmetic
 * inside the transform can go below:
 *
 *   - quantisation of the input: a volume rounded to integers carries noise of
 *     RMS 1/sqrt(12) LSB per voxel, which reaches every coefficient as
 *     sqrt(N) / sqrt(12). In the units used here (error / sqrt(N), "LSB") that
 *     is 0.289 whatever the size.
 *   - the float output: the exact spectrum rounded to float. A transform that
 *     returns float cannot be more accurate than this, however it computes.
 *
 * Inputs. White integer noise, the case with the most information per voxel:
 *   uint8 full   uniform 0..255
 *   uint16 full  uniform 0..65535
 *   uint16 CT    30000 + uniform -2000..2000: a narrow band far from zero, the
 *                way a reconstructed tomogram sits in its range
 * White noise is not what a tomogram looks like: its spectrum is dominated by a
 * few large low frequencies (sample against air, the container wall, cupping),
 * which is where the float error can grow (docs/R5_PLAN.md, risk R2). Hence two
 * more sources (R-5 step 0.4):
 *   --synthetic cylinder  a sample cylinder along z in air, grains as random
 *                balls on a pore matrix in the band 30000 +- 2000, cupping (the
 *                sample darker toward its axis), Gaussian noise; deterministic.
 *                u8 is the same volume divided by 256.
 *   --raw        a real volume, raw little-endian integers, x fastest, as
 *                tools/r5_tiff_to_raw.py writes them; --crop takes a box of it
 *                (the box, not the file, must be a power of two per axis).
 *
 * Each is transformed raw, and with an integer offset subtracted first, by
 * the policies of R-5 step 0.5, cheapest first: half the range (no pass over
 * the data), a known level (here the histogram mode, standing for air or the
 * matrix), the mean of every 512th and every 64th line, the rounded mean (one
 * full reduction). The subtraction is exact, and it changes only the
 * zero-frequency coefficient, by offset * N, which is added back in double
 * before comparing. Errors are taken over the non-zero frequencies; the zero
 * frequency is reported on its own. Under each row: the largest error by band
 * of frequency with the worst coefficient, and the round trip -- the float
 * spectrum transformed back, rounded, compared voxel by voxel with the input.
 *
 * Three rows a policy: cuFFT and GPUFactored on the float input with the
 * offset subtracted on the host, and (R-5 step 3.2) "R-5 kernel": the
 * integers themselves on the device, the first pass reading them, the offset
 * resolved by the library's own policy (internal::IntegerOffset -- the means
 * reduced on the device) and returned to H(0) in float.
 * tools/r5_accuracy_table.py turns the report into a table and a verdict.
 *
 * The reference is the CPU backend in double: for these inputs every partial
 * sum is far below 2^53, so it is exact to well below the floors above.
 */

#include <rapidht/device_volume.h>
#include <rapidht/transform.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "internal/device_access.h"
#include "cufft_baseline.h"

namespace {

using RapiDHT::DeviceVolume;
using RapiDHT::HartleyTransform;
using RapiDHT::Modes;

/// An integer volume held as doubles (exact), with what the report needs.
struct Volume {
    std::string name;
    size_t width = 0, height = 0, depth = 0;
    double halfRange = 0.0; ///< the offset a format suggests without looking at the data
    std::vector<double> data;

    size_t Count() const { return width * height * depth; }
};

/// SplitMix64: reproducible, and the same on every platform.
struct SplitMix64 {
    uint64_t state;
    uint64_t Next()
    {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    /// Uniform in [0, 1).
    double Uniform() { return static_cast<double>(Next() >> 11) * 0x1.0p-53; }
    /// Standard normal (Box-Muller; one value of the pair is dropped).
    double Normal()
    {
        const double u = 1.0 - Uniform(); // (0, 1]
        const double v = Uniform();
        return std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v);
    }
};

/// White integers in [base, base + span), n^3.
Volume WhiteNoise(size_t n, const char* name, int32_t base, int32_t span, double halfRange)
{
    Volume v { name, n, n, n, halfRange, std::vector<double>(n * n * n) };
    SplitMix64 rng { 2463534242u };
    for (double& value : v.data) {
        value = static_cast<double>(base + static_cast<int32_t>(rng.Next() % static_cast<uint64_t>(span)));
    }
    return v;
}

/*
 * A synthetic tomogram of a porous sample, uint16 levels (u8: divided by 256):
 *
 *   air       outside a cylinder of radius 0.45 n along z: level 6000
 *   pores     inside it: level 28500
 *   grains    random balls of radius n/40 .. n/14 placed until they cover about
 *             half the cylinder (overlaps allowed): level 31500
 *   cupping   inside the cylinder the level drops toward the axis by up to 4 %
 *             (quadratically in r: 1 - 0.04 (1 - r^2 / R^2))
 *   noise     Gaussian, sigma 300 (u16), everywhere
 *
 * then rounded and clamped to the type. The band of the sample is about
 * 27 400 .. 32 100, the "30000 +- 2000" of the plan; air gives the sharp edge
 * of the sample, the largest low-frequency coefficients.
 */
Volume SyntheticCylinder(size_t n, bool u8)
{
    const std::string name = std::string("synthetic cylinder, ") + (u8 ? "uint8" : "uint16");
    Volume v { name, n, n, n, u8 ? 128.0 : 32768.0, std::vector<double>(n * n * n) };
    const double centre = 0.5 * static_cast<double>(n) - 0.5;
    const double radius = 0.45 * static_cast<double>(n);
    std::vector<uint8_t> grain(n * n * n, 0);

    // Balls until their nominal volume is half the cylinder's.
    SplitMix64 rng { 0x5EED0004u };
    const double target = 0.5 * 3.141592653589793 * radius * radius * static_cast<double>(n);
    double placed = 0.0;
    while (placed < target) {
        const double r = static_cast<double>(n) * (1.0 / 40.0 + rng.Uniform() * (1.0 / 14.0 - 1.0 / 40.0));
        // Centre uniformly inside the cylinder (rejection on the disc).
        double cx = 0.0, cy = 0.0;
        do {
            cx = centre + (2.0 * rng.Uniform() - 1.0) * radius;
            cy = centre + (2.0 * rng.Uniform() - 1.0) * radius;
        } while ((cx - centre) * (cx - centre) + (cy - centre) * (cy - centre) > radius * radius);
        const double cz = rng.Uniform() * static_cast<double>(n);
        const long x0 = std::max(0L, static_cast<long>(std::floor(cx - r)));
        const long x1 = std::min(static_cast<long>(n) - 1, static_cast<long>(std::ceil(cx + r)));
        const long y0 = std::max(0L, static_cast<long>(std::floor(cy - r)));
        const long y1 = std::min(static_cast<long>(n) - 1, static_cast<long>(std::ceil(cy + r)));
        const long z0 = std::max(0L, static_cast<long>(std::floor(cz - r)));
        const long z1 = std::min(static_cast<long>(n) - 1, static_cast<long>(std::ceil(cz + r)));
        for (long z = z0; z <= z1; ++z) {
            for (long y = y0; y <= y1; ++y) {
                for (long x = x0; x <= x1; ++x) {
                    const double dx = x - cx, dy = y - cy, dz = z - cz;
                    if (dx * dx + dy * dy + dz * dz <= r * r) {
                        grain[static_cast<size_t>(x) + n * (static_cast<size_t>(y) + n * static_cast<size_t>(z))] = 1;
                    }
                }
            }
        }
        placed += 4.0 / 3.0 * 3.141592653589793 * r * r * r;
    }

    SplitMix64 noise { 0x5EED0005u };
    const double maximum = u8 ? 255.0 : 65535.0;
    for (size_t z = 0; z < n; ++z) {
        for (size_t y = 0; y < n; ++y) {
            for (size_t x = 0; x < n; ++x) {
                const size_t i = x + n * (y + n * z);
                const double dx = static_cast<double>(x) - centre, dy = static_cast<double>(y) - centre;
                const double rr = (dx * dx + dy * dy) / (radius * radius);
                double level = 6000.0;
                if (rr <= 1.0) {
                    level = (grain[i] != 0 ? 31500.0 : 28500.0) * (1.0 - 0.04 * (1.0 - rr));
                }
                level += 300.0 * noise.Normal();
                if (u8) {
                    level /= 256.0;
                }
                v.data[i] = std::clamp(std::round(level), 0.0, maximum);
            }
        }
    }
    return v;
}

/// 64-bit file positions: fseeko / ftello on POSIX, their _i64 forms on Windows.
int Seek(FILE* file, unsigned long long at, int whence)
{
#ifdef _WIN32
    return _fseeki64(file, static_cast<long long>(at), whence);
#else
    return fseeko(file, static_cast<off_t>(at), whence);
#endif
}

unsigned long long Tell(FILE* file)
{
#ifdef _WIN32
    return static_cast<unsigned long long>(_ftelli64(file));
#else
    return static_cast<unsigned long long>(ftello(file));
#endif
}

/// A box of a raw volume of file extent fw x fh x fd, element size 1 or 2.
Volume ReadRaw(const std::string& path, size_t fw, size_t fh, size_t fd, bool u8, size_t x0, size_t y0, size_t z0,
    size_t w, size_t h, size_t d)
{
    if (x0 + w > fw || y0 + h > fh || z0 + d > fd) {
        throw std::runtime_error("--crop box lies outside the file's extent");
    }
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        throw std::runtime_error("cannot open " + path);
    }
    const size_t bytes = u8 ? 1 : 2;
    if (Seek(file, 0, SEEK_END) != 0 || Tell(file) != static_cast<unsigned long long>(fw) * fh * fd * bytes) {
        std::fclose(file);
        throw std::runtime_error(path + ": size does not match the extent and type given");
    }
    std::string name = path.substr(path.find_last_of('/') + 1);
    if (w != fw || h != fh || d != fd) {
        name += ", box at (" + std::to_string(x0) + ", " + std::to_string(y0) + ", " + std::to_string(z0) + ")";
    }
    Volume v { name, w, h, d, u8 ? 128.0 : 32768.0, std::vector<double>(w * h * d) };
    std::vector<unsigned char> line(w * bytes);
    for (size_t z = 0; z < d; ++z) {
        for (size_t y = 0; y < h; ++y) {
            const unsigned long long at = ((static_cast<unsigned long long>(z0 + z) * fh + (y0 + y)) * fw + x0) * bytes;
            if (Seek(file, at, SEEK_SET) != 0 || std::fread(line.data(), 1, line.size(), file) != line.size()) {
                std::fclose(file);
                throw std::runtime_error(path + ": read failed");
            }
            double* out = v.data.data() + w * (y + h * z);
            for (size_t x = 0; x < w; ++x) {
                // Little-endian, whatever the host is.
                out[x] = u8 ? line[x] : static_cast<double>(line[2 * x] | (line[2 * x + 1] << 8));
            }
        }
    }
    std::fclose(file);
    return v;
}

std::vector<float> OnDevice(const std::vector<float>& input, const Volume& v, bool cufft)
{
    std::vector<float> output(input.size());
    if (cufft) {
        float* device = nullptr;
        if (cudaMalloc(&device, input.size() * sizeof(float)) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
        try {
            rapidht_bench::CufftHartley plan(
                static_cast<int>(v.width), static_cast<int>(v.height), static_cast<int>(v.depth));
            cudaMemcpy(device, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
            plan.Forward(device);
            cudaMemcpy(output.data(), device, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
        } catch (...) {
            cudaFree(device);
            throw;
        }
        cudaFree(device);
        return output;
    }
    HartleyTransform<float> transform(v.width, v.height, v.depth, Modes::GPUFactored);
    DeviceVolume<float> volume(input.size());
    volume.Upload(input.data());
    transform.ForwardTransform(volume);
    volume.Download(output.data());
    return output;
}

/// R-5 step 3.2: the integers on the device through the R-5 kernel with the
/// library's offset policy; the result is the transform of the input itself.
template <typename TIn>
std::vector<float> IntegerOnDevice(const Volume& v, const RapiDHT::internal::IntegerOffset& policy)
{
    const size_t count = v.Count();
    std::vector<TIn> host(count);
    for (size_t i = 0; i < count; ++i) {
        host[i] = static_cast<TIn>(v.data[i]);
    }
    TIn* in = nullptr;
    if (cudaMalloc(&in, count * sizeof(TIn)) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
    }
    std::vector<float> output(count);
    try {
        cudaMemcpy(in, host.data(), count * sizeof(TIn), cudaMemcpyHostToDevice);
        HartleyTransform<float> transform(v.width, v.height, v.depth, Modes::GPUFactored);
        DeviceVolume<float> out(count);
        RapiDHT::internal::DeviceAccess::ForwardFromIntegers(transform, in, out, policy);
        out.Download(output.data());
    } catch (...) {
        cudaFree(in);
        throw;
    }
    cudaFree(in);
    return output;
}

struct Error {
    double rms = 0.0; ///< over k != 0, in LSB: RMS(error) / sqrt(N)
    double largest = 0.0; ///< over k != 0, in LSB
    double dc = 0.0; ///< |error| at k = 0, in LSB
};

/// `actual` is the transform of (input - offset); `reference` that of the input.
Error Measure(const std::vector<float>& actual, double offset, const std::vector<double>& reference)
{
    const double n = static_cast<double>(reference.size());
    const double unit = std::sqrt(n);
    Error e;
    e.dc = std::fabs(static_cast<double>(actual[0]) + offset * n - reference[0]) / unit;
    double sum = 0.0;
    for (size_t i = 1; i < reference.size(); ++i) {
        const double error = std::fabs(static_cast<double>(actual[i]) - reference[i]);
        sum += error * error;
        if (!(error <= e.largest)) {
            e.largest = error;
        }
    }
    e.rms = std::sqrt(sum / (n - 1.0)) / unit;
    e.largest /= unit;
    return e;
}

/// The float output floor: the exact spectrum rounded to float.
Error OutputFloor(const std::vector<double>& reference)
{
    std::vector<float> rounded(reference.begin(), reference.end());
    return Measure(rounded, 0.0, reference);
}

void Row(const char* label, const Error& e)
{
    std::printf("  %-34s rms %9.3e   largest %9.3e   zero frequency %9.3e\n", label, e.rms, e.largest, e.dc);
}

std::string Extent(const Volume& v)
{
    if (v.width == v.height && v.height == v.depth) {
        return std::to_string(v.width) + "^3";
    }
    return std::to_string(v.width) + "x" + std::to_string(v.height) + "x" + std::to_string(v.depth);
}

/*
 * Where the worst coefficients are (R-5 step 0.5). Frequencies are folded,
 * k' = min(k, n - k) per axis, and the radius is rho = |(kx'/W, ky'/H, kz'/D)|,
 * from 0 (the zero frequency) to sqrt(3)/2. The largest error is reported per
 * band of rho, and the worst coefficient with its frequency and the size of its
 * exact value -- on a tomogram the worst errors sit on the few large
 * low-frequency coefficients, whose float rounding alone exceeds the rest.
 */
void ByFrequency(const std::vector<float>& actual, const std::vector<double>& reference, const Volume& v)
{
    const double bands[] = { 1.0 / 64.0, 1.0 / 16.0, 1.0 / 4.0 };
    double worstInBand[4] = { 0.0, 0.0, 0.0, 0.0 };
    double worst = -1.0;
    size_t worstAt = 0;
    const size_t W = v.width, H = v.height, D = v.depth;
    for (size_t z = 0; z < D; ++z) {
        const double fz = static_cast<double>(std::min(z, D - z)) / static_cast<double>(D);
        for (size_t y = 0; y < H; ++y) {
            const double fy = static_cast<double>(std::min(y, H - y)) / static_cast<double>(H);
            const size_t row = W * (y + H * z);
            for (size_t x = 0; x < W; ++x) {
                const size_t i = row + x;
                if (i == 0) {
                    continue;
                }
                const double error = std::fabs(static_cast<double>(actual[i]) - reference[i]);
                const double fx = static_cast<double>(std::min(x, W - x)) / static_cast<double>(W);
                const double rho = std::sqrt(fx * fx + fy * fy + fz * fz);
                int band = 3;
                while (band > 0 && rho < bands[band - 1]) {
                    --band;
                }
                worstInBand[band] = std::max(worstInBand[band], error);
                if (error > worst) {
                    worst = error;
                    worstAt = i;
                }
            }
        }
    }
    const double unit = std::sqrt(static_cast<double>(v.Count()));
    const size_t kx = worstAt % W, ky = (worstAt / W) % H, kz = worstAt / (W * H);
    std::printf("      largest by |k|/k_max: <1/64 %8.2e  <1/16 %8.2e  <1/4 %8.2e  >=1/4 %8.2e;"
                " worst at k=(%zu,%zu,%zu), |H|/sqrt(N) %.3g\n",
        worstInBand[0] / unit, worstInBand[1] / unit, worstInBand[2] / unit, worstInBand[3] / unit,
        std::min(kx, W - kx), std::min(ky, H - ky), std::min(kz, D - kz), std::fabs(reference[worstAt]) / unit);
}

/*
 * The round trip on the float path (a first look at criterion C4 of R-5): the
 * float spectrum through the same transform again -- the Hartley transform is
 * its own inverse up to 1/N -- then / N and + offset in double, rounded to the
 * nearest integer. Counts the voxels that do not come back exactly; the
 * largest |x' - x| before rounding says how far from 0.5 that is.
 */
void RoundTrip(const std::vector<float>& spectrum, double offset, const Volume& v, bool cufft)
{
    const std::vector<float> back = OnDevice(spectrum, v, cufft);
    const double scale = 1.0 / static_cast<double>(v.Count());
    size_t mismatched = 0;
    double largest = 0.0;
    for (size_t i = 0; i < back.size(); ++i) {
        const double x = static_cast<double>(back[i]) * scale + offset;
        largest = std::max(largest, std::fabs(x - v.data[i]));
        if (std::round(x) != v.data[i]) {
            ++mismatched;
        }
    }
    std::printf("      round trip: %zu of %zu voxels not restored (%.3g %%), largest |x' - x| %.3e\n", mismatched,
        back.size(), 100.0 * static_cast<double>(mismatched) / static_cast<double>(back.size()), largest);
}

/// R-5 step 4.2: `spectrum` (the transform of the input itself) back into
/// integers on the device, compared voxel by voxel; printed as RoundTrip does,
/// with the clamped count.
template <typename TOut>
void IntegerRoundTrip(const std::vector<float>& spectrum, const Volume& v)
{
    const size_t count = v.Count();
    TOut* out = nullptr;
    if (cudaMalloc(&out, count * sizeof(TOut)) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
    }
    std::vector<TOut> back(count);
    size_t saturated = 0;
    try {
        HartleyTransform<float> transform(v.width, v.height, v.depth, Modes::GPUFactored);
        DeviceVolume<float> work(count);
        work.Upload(spectrum.data());
        RapiDHT::internal::DeviceAccess::InverseToIntegers(
            transform, work, out, RapiDHT::internal::IntegerOffset::MeanOfAll(), &saturated);
        cudaMemcpy(back.data(), out, count * sizeof(TOut), cudaMemcpyDeviceToHost);
    } catch (...) {
        cudaFree(out);
        throw;
    }
    cudaFree(out);
    size_t mismatched = 0;
    double largest = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double d = std::fabs(static_cast<double>(back[i]) - v.data[i]);
        largest = std::max(largest, d);
        mismatched += d != 0.0 ? 1 : 0;
    }
    std::printf("      round trip: %zu of %zu voxels not restored (%.3g %%), largest |x' - x| %.3e"
                " (into integers on the device, %zu clamped)\n",
        mismatched, count, 100.0 * static_cast<double>(mismatched) / static_cast<double>(count), largest, saturated);
}

/// The mean over every `step`-th line along x (lines y + H*z = 0, step, ...),
/// rounded: the sampled-mean offset policies of step 0.5.
double SampledMean(const Volume& v, size_t step)
{
    double sum = 0.0;
    size_t count = 0;
    for (size_t line = 0; line < v.height * v.depth; line += step) {
        const double* x = v.data.data() + v.width * line;
        for (size_t i = 0; i < v.width; ++i) {
            sum += x[i];
        }
        count += v.width;
    }
    return std::round(sum / static_cast<double>(count));
}

/// A level the user may know without a pass over the data, such as air or the
/// matrix: here the mode of the histogram, in bins of 64 for uint16 (1 for
/// uint8), at the centre of its bin.
double HistogramMode(const Volume& v)
{
    const bool u8 = v.halfRange < 256.0;
    const size_t bin = u8 ? 1 : 64;
    std::vector<size_t> counts((u8 ? 256 : 65536) / bin, 0);
    for (double x : v.data) {
        ++counts[static_cast<size_t>(x) / bin];
    }
    const size_t mode = static_cast<size_t>(std::max_element(counts.begin(), counts.end()) - counts.begin());
    return static_cast<double>(mode * bin + bin / 2);
}

void Run(const Volume& v)
{
    const size_t count = v.Count();
    double mean = 0.0, low = v.data[0], high = v.data[0];
    for (double x : v.data) {
        mean += x;
        low = std::min(low, x);
        high = std::max(high, x);
    }
    mean /= static_cast<double>(count);

    std::vector<double> reference = v.data;
    HartleyTransform<double>(v.width, v.height, v.depth, Modes::CPU).ForwardTransform(reference.data());

    std::printf("%s, %s (mean %.1f, range %.0f..%.0f); errors / sqrt(N), in LSB of the input\n", Extent(v).c_str(),
        v.name.c_str(), mean, low, high);
    std::printf("  %-34s rms %9.3e   (input rounding: 1/sqrt(12))\n", "floor: quantisation of the input",
        1.0 / std::sqrt(12.0));
    Row("floor: exact spectrum stored as float", OutputFloor(reference));

    // The offset policies of step 0.5, cheapest first (docs/R5_PLAN.md, rule
    // for the default policy). Every one is printed, equal values included,
    // so that tools/r5_accuracy_table.py finds each policy on every input.
    using RapiDHT::internal::IntegerOffset;
    struct Policy {
        const char* name;
        double offset;
        IntegerOffset library; ///< the same policy as the library resolves it
    };
    const double level = HistogramMode(v);
    const Policy policies[] = {
        { "raw", 0.0, IntegerOffset::NoOffset() },
        { "minus half range", v.halfRange, IntegerOffset::Of(v.halfRange) },
        { "minus known level", level, IntegerOffset::Of(level) },
        { "minus mean of every 512th line", SampledMean(v, 512), IntegerOffset::MeanOfEvery(512) },
        { "minus mean of every 64th line", SampledMean(v, 64), IntegerOffset::MeanOfEvery(64) },
        { "minus rounded mean", std::round(mean), IntegerOffset::MeanOfAll() },
    };
    std::printf("  offsets:");
    for (const Policy& p : policies) {
        std::printf(" %s = %.0f;", p.name, p.offset);
    }
    std::printf("\n");

    for (const Policy& p : policies) {
        std::vector<float> single(count);
        for (size_t i = 0; i < count; ++i) {
            single[i] = static_cast<float>(v.data[i] - p.offset); // exact: integers below 2^24
        }
        for (const bool cufft : { true, false }) {
            const std::string label = std::string(cufft ? "cuFFT + conversion, " : "GPUFactored, ") + p.name;
            const std::vector<float> spectrum = OnDevice(single, v, cufft);
            Row(label.c_str(), Measure(spectrum, p.offset, reference));
            ByFrequency(spectrum, reference, v);
            RoundTrip(spectrum, p.offset, v, cufft);
        }
        // The R-5 kernel: H(0) already holds the offset, so it is measured
        // with none.
        const std::vector<float> spectrum = v.halfRange < 256.0 ? IntegerOnDevice<uint8_t>(v, p.library)
                                                                : IntegerOnDevice<uint16_t>(v, p.library);
        Row((std::string("R-5 kernel, ") + p.name).c_str(), Measure(spectrum, 0.0, reference));
        ByFrequency(spectrum, reference, v);
        // Back into integers by the inverse of R-5 step 4.2 (the mean read
        // off H(0), the voxels rounded and clamped on the device): criterion
        // C4 on this input.
        if (v.halfRange < 256.0) {
            IntegerRoundTrip<uint8_t>(spectrum, v);
        } else {
            IntegerRoundTrip<uint16_t>(spectrum, v);
        }
    }
    std::fflush(stdout);
}

bool IsPowerOfTwo(size_t n)
{
    return n >= 2 && (n & (n - 1)) == 0;
}

size_t ParseSize(const char* text)
{
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::runtime_error(std::string("not a number: ") + text);
    }
    return static_cast<size_t>(value);
}

/// "WxHxD" -> three sizes.
void ParseExtent(const std::string& text, size_t& w, size_t& h, size_t& d)
{
    const size_t a = text.find('x'), b = text.find('x', a + 1);
    if (a == std::string::npos || b == std::string::npos) {
        throw std::runtime_error("extent must be WxHxD, got " + text);
    }
    w = ParseSize(text.substr(0, a).c_str());
    h = ParseSize(text.substr(a + 1, b - a - 1).c_str());
    d = ParseSize(text.substr(b + 1).c_str());
}

bool ParseType(const std::string& text)
{
    if (text != "u8" && text != "u16") {
        throw std::runtime_error("type must be u8 or u16, got " + text);
    }
    return text == "u8";
}

void CheckTransformExtent(size_t w, size_t h, size_t d)
{
    if (!IsPowerOfTwo(w) || !IsPowerOfTwo(h) || !IsPowerOfTwo(d)) {
        throw std::runtime_error("the transformed extent must be a power of two per axis");
    }
}

int Usage(const char* program)
{
    std::fprintf(stderr,
        "usage: %s EDGE [EDGE ...]                                white noise, EDGE^3\n"
        "       %s --synthetic cylinder u16|u8 EDGE [EDGE ...]    synthetic tomogram, EDGE^3\n"
        "       %s --raw FILE WxHxD u16|u8 [--crop X0 Y0 Z0 W H D]\n"
        "EDGE and the transformed extent: powers of two.\n",
        program, program, program);
    return 2;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        return Usage(argv[0]);
    }
    try {
        const std::string mode = argv[1];
        if (mode == "--synthetic") {
            if (argc < 5 || std::string(argv[2]) != "cylinder") {
                return Usage(argv[0]);
            }
            const bool u8 = ParseType(argv[3]);
            for (int i = 4; i < argc; ++i) {
                const size_t n = ParseSize(argv[i]);
                CheckTransformExtent(n, n, n);
                Run(SyntheticCylinder(n, u8));
            }
        } else if (mode == "--raw") {
            if (argc != 5 && argc != 12) {
                return Usage(argv[0]);
            }
            size_t fw = 0, fh = 0, fd = 0;
            ParseExtent(argv[3], fw, fh, fd);
            const bool u8 = ParseType(argv[4]);
            size_t box[6] = { 0, 0, 0, fw, fh, fd };
            if (argc == 12) {
                if (std::string(argv[5]) != "--crop") {
                    return Usage(argv[0]);
                }
                for (int i = 0; i < 6; ++i) {
                    box[i] = ParseSize(argv[6 + i]);
                }
            }
            CheckTransformExtent(box[3], box[4], box[5]);
            Run(ReadRaw(argv[2], fw, fh, fd, u8, box[0], box[1], box[2], box[3], box[4], box[5]));
        } else {
            for (int i = 1; i < argc; ++i) {
                const size_t n = ParseSize(argv[i]);
                CheckTransformExtent(n, n, n);
                Run(WhiteNoise(n, "uint8 full range, uniform 0..255", 0, 256, 128.0));
                Run(WhiteNoise(n, "uint16 full range, uniform 0..65535", 0, 65536, 32768.0));
                Run(WhiteNoise(n, "uint16 CT-like band, 30000 +- 2000", 28000, 4001, 32768.0));
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
