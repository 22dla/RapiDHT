/*
 * Project: RapiDHT
 * File: tests/test_convolution.cpp
 * Brief: The separable transform and the convolutions (even kernels, any
 *        kernel; CPU and device), checked against direct evaluation of their
 *        definitions.
 */

#include <rapidht/convolution.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_support.h"

using namespace RapiDHT;
using namespace rapidht_test;

namespace {

struct Extent {
    size_t width;
    size_t height;
    size_t depth;
};

// Small, because the references are O(N^2). Non-square on purpose: a swapped
// axis cannot hide behind equal extents.
const Extent kExtents[] = {
    { 16, 0, 0 },
    { 8, 4, 0 },
    { 4, 16, 0 },
    { 4, 4, 4 },
    { 8, 4, 2 },
    { 2, 8, 16 },
};

std::string Describe(const Extent& e)
{
    return std::to_string(e.width) + "x" + std::to_string(e.height) + "x" + std::to_string(e.depth);
}

size_t Mirror(size_t i, size_t n) { return i == 0 ? 0 : n - i; }

/// Separable DHT by definition: a direct 1D DHT along each axis in turn.
std::vector<double> ReferenceSeparableDht(std::vector<double> data, const Dims& d)
{
    const double kTwoPi = 2.0 * std::acos(-1.0);
    const size_t extents[3] = { d.width, d.height, d.depth };
    for (int axis = 0; axis < 3; ++axis) {
        const size_t n = extents[axis];
        if (n == 1) {
            continue;
        }
        std::vector<double> out(data.size(), 0.0);
        for (size_t k = 0; k < d.depth; ++k) {
            for (size_t j = 0; j < d.height; ++j) {
                for (size_t i = 0; i < d.width; ++i) {
                    const size_t idx[3] = { i, j, k };
                    double sum = 0.0;
                    for (size_t m = 0; m < n; ++m) {
                        size_t src[3] = { i, j, k };
                        src[axis] = m;
                        const double phase = kTwoPi * static_cast<double>(idx[axis] * m) / static_cast<double>(n);
                        sum += data[d.Index(src[0], src[1], src[2])] * Cas(phase);
                    }
                    out[d.Index(i, j, k)] = sum;
                }
            }
        }
        data.swap(out);
    }
    return data;
}

// ReferenceCircularConvolution (the definition) is in test_support.h.

/// An anisotropic Gaussian in the wrapped layout: even along every axis.
std::vector<double> MakeEvenKernel(const Dims& d)
{
    std::vector<double> g(d.Total());
    for (size_t k = 0; k < d.depth; ++k) {
        for (size_t j = 0; j < d.height; ++j) {
            for (size_t i = 0; i < d.width; ++i) {
                const double x = static_cast<double>(std::min(i, d.width - i));
                const double y = static_cast<double>(std::min(j, d.height - j));
                const double z = static_cast<double>(std::min(k, d.depth - k));
                g[d.Index(i, j, k)] = std::exp(-(x * x / 3.0 + y * y / 2.0 + z * z / 5.0));
            }
        }
    }
    return g;
}

/// A kernel that is even but not separable and not smooth: the average of a
/// fixed signal over all eight axis flips. Guards against a test that only
/// passes because the Gaussian happens to factorise.
std::vector<double> MakeRoughEvenKernel(const Dims& d)
{
    const std::vector<double> s = MakeSignal(d.Total());
    std::vector<double> g(d.Total(), 0.0);
    for (size_t k = 0; k < d.depth; ++k) {
        for (size_t j = 0; j < d.height; ++j) {
            for (size_t i = 0; i < d.width; ++i) {
                double sum = 0.0;
                for (int flip = 0; flip < 8; ++flip) {
                    const size_t a = (flip & 1) ? Mirror(i, d.width) : i;
                    const size_t b = (flip & 2) ? Mirror(j, d.height) : j;
                    const size_t c = (flip & 4) ? Mirror(k, d.depth) : k;
                    sum += s[d.Index(a, b, c)];
                }
                g[d.Index(i, j, k)] = sum / 8.0;
            }
        }
    }
    return g;
}

} // namespace

// ---------------------------------------------------------------------------
// The separable transform
// ---------------------------------------------------------------------------

TEST(Separable, MatchesDefinition)
{
    for (const auto& e : kExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(d.Total());

        std::vector<double> actual = input;
        HartleyTransform<double> ht(e.width, e.height, e.depth, Modes::CPU);
        ht.ForwardSeparable(actual.data());

        ExpectClose(actual, ReferenceSeparableDht(input, d), 1e-12);
    }
}

// B-102: the device path is the true transform's multiplies without the
// Bracewell kernel (plus a copy in 3D). Checked against the definition, not
// against the CPU, so that both backends answer to the same reference.
TEST(Separable, MatchesDefinition_GPU)
{
    SKIP_IF_NO_GPU();
    for (const auto& e : kExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(d.Total());
        const auto expected = ReferenceSeparableDht(input, d);

        for (const Modes mode : { Modes::GPU, Modes::GPUFactored }) {
            SCOPED_TRACE(mode == Modes::GPU ? "GPU" : "GPUFactored");
            HartleyTransform<double> ht(e.width, e.height, e.depth, mode);

            std::vector<double> fromHost = input;
            ht.ForwardSeparable(fromHost.data());
            ExpectClose(fromHost, expected, 1e-9);

            DeviceVolume<double> volume(d.Total());
            volume.Upload(input.data());
            ht.ForwardSeparable(volume);
            std::vector<double> fromResident(d.Total());
            volume.Download(fromResident.data());
            ExpectClose(fromResident, expected, 1e-9);
        }
    }
}

// The resident overload shares the checks of ForwardTransform(DeviceVolume&),
// which must run before anything touches the device state of a CPU object.
TEST(Separable, ResidentRejectsCpuModeAndWrongSize_GPU)
{
    SKIP_IF_NO_GPU();
    DeviceVolume<double> volume(64);

    HartleyTransform<double> cpu(4, 4, 4, Modes::CPU);
    EXPECT_THROW(cpu.ForwardSeparable(volume), std::invalid_argument);

    HartleyTransform<double> gpu(8, 4, 4, Modes::GPU);
    EXPECT_THROW(gpu.ForwardSeparable(volume), std::invalid_argument);
}

TEST(Separable, AppliedTwiceScalesByN)
{
    for (const auto& e : kExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto input = MakeSignal(d.Total());

        std::vector<double> data = input;
        HartleyTransform<double> ht(e.width, e.height, e.depth, Modes::CPU);
        ht.ForwardSeparable(data.data());
        ht.ForwardSeparable(data.data());

        std::vector<double> expected = input;
        for (auto& v : expected) {
            v *= static_cast<double>(d.Total());
        }
        ExpectClose(data, expected, 1e-12);
    }
}

TEST(Separable, CoincidesWithTrueTransformIn1D)
{
    const auto input = MakeSignal(64);
    std::vector<double> separable = input;
    HartleyTransform<double> ht(64, 0, 0, Modes::CPU);
    ht.ForwardSeparable(separable.data());
    ExpectClose(separable, ReferenceDht(input, Dims::Of(64, 0, 0)), 1e-12);
}

TEST(Separable, DiffersFromTrueTransformIn3D)
{
    // A guard on the guard: if these coincided, the tests above could not tell
    // a missing Bracewell pass from a present one.
    const Dims d = Dims::Of(4, 4, 4);
    const auto input = MakeSignal(d.Total());
    std::vector<double> separable = input;
    HartleyTransform<double> ht(4, 4, 4, Modes::CPU);
    ht.ForwardSeparable(separable.data());
    const auto truth = ReferenceDht(input, d);

    double worst = 0.0;
    for (size_t n = 0; n < truth.size(); ++n) {
        worst = std::max(worst, std::fabs(truth[n] - separable[n]));
    }
    EXPECT_GT(worst, 1e-6);
}

// ---------------------------------------------------------------------------
// Convolution with even kernels
// ---------------------------------------------------------------------------

TEST(EvenConvolution, MatchesDirectConvolutionGaussian)
{
    for (const auto& e : kExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto f = MakeSignal(d.Total());
        const auto g = MakeEvenKernel(d);

        std::vector<double> actual = f;
        EvenKernelConvolution<double> conv(e.width, e.height, e.depth);
        conv.SetKernel(g.data());
        conv.Apply(actual.data());

        ExpectClose(actual, ReferenceCircularConvolution(f, g, d), 1e-12);
    }
}

TEST(EvenConvolution, MatchesDirectConvolutionRoughKernel)
{
    for (const auto& e : kExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto f = MakeSignal(d.Total());
        const auto g = MakeRoughEvenKernel(d);

        std::vector<double> actual = f;
        EvenKernelConvolution<double> conv(e.width, e.height, e.depth);
        conv.SetKernel(g.data());
        conv.Apply(actual.data());

        ExpectClose(actual, ReferenceCircularConvolution(f, g, d), 1e-12);
    }
}

TEST(EvenConvolution, FloatIsAccurate)
{
    const Extent e { 8, 4, 2 };
    const Dims d = Dims::Of(e.width, e.height, e.depth);
    const auto f = MakeSignal(d.Total());
    const auto g = MakeEvenKernel(d);
    const auto expected = ReferenceCircularConvolution(f, g, d);

    std::vector<float> data(f.begin(), f.end());
    std::vector<float> kernel(g.begin(), g.end());
    EvenKernelConvolution<float> conv(e.width, e.height, e.depth);
    conv.SetKernel(kernel.data());
    conv.Apply(data.data());

    ExpectClose(std::vector<double>(data.begin(), data.end()), expected, 1e-5);
}

TEST(EvenConvolution, KernelReusedAcrossCalls)
{
    const Extent e { 4, 4, 4 };
    const Dims d = Dims::Of(e.width, e.height, e.depth);
    const auto g = MakeEvenKernel(d);
    EvenKernelConvolution<double> conv(e.width, e.height, e.depth);
    conv.SetKernel(g.data());

    for (int round = 0; round < 3; ++round) {
        std::vector<double> f = MakeSignal(d.Total());
        for (auto& v : f) {
            v += round;
        }
        const auto expected = ReferenceCircularConvolution(f, g, d);
        conv.Apply(f.data());
        ExpectClose(f, expected, 1e-12);
    }
}

TEST(EvenConvolution, DeltaKernelIsIdentity)
{
    const Dims d = Dims::Of(16, 8, 4);
    std::vector<double> delta(d.Total(), 0.0);
    delta[0] = 1.0;
    const auto f = MakeSignal(d.Total());

    std::vector<double> actual = f;
    EvenKernelConvolution<double> conv(16, 8, 4);
    conv.SetKernel(delta.data());
    conv.Apply(actual.data());
    ExpectClose(actual, f, 1e-12);
}

TEST(EvenConvolution, RejectsKernelThatIsNotEven)
{
    const Dims d = Dims::Of(8, 4, 2);
    std::vector<double> shifted(d.Total(), 0.0);
    shifted[d.Index(1, 0, 0)] = 1.0; // a delta off the origin: odd along X
    EXPECT_FALSE(IsEvenPerAxis(shifted.data(), 8, 4, 2));

    EvenKernelConvolution<double> conv(8, 4, 2);
    EXPECT_THROW(conv.SetKernel(shifted.data()), std::invalid_argument);
}

TEST(EvenConvolution, ApplyWithoutKernelThrows)
{
    std::vector<double> data(16, 1.0);
    EvenKernelConvolution<double> conv(16, 0, 0);
    EXPECT_THROW(conv.Apply(data.data()), std::logic_error);
}

TEST(EvenConvolution, RejectsMatrixGpuMode)
{
    // The device convolutions run on the factored transform only.
    EXPECT_THROW(EvenKernelConvolution<double>(8, 8, 8, Modes::GPU), std::exception);
    EXPECT_THROW(EvenKernelBank<double>(8, 8, 8, Modes::GPU), std::exception);
    EXPECT_THROW(CircularConvolution<double>(8, 8, 8, Modes::GPU), std::exception);
}

// ---------------------------------------------------------------------------
// A bank of even kernels
// ---------------------------------------------------------------------------

TEST(EvenKernelBank, MatchesDirectConvolutionForEveryKernel)
{
    for (const auto& e : kExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto f = MakeSignal(d.Total());
        std::vector<double> delta(d.Total(), 0.0);
        delta[0] = 1.0;
        const std::vector<std::vector<double>> kernels = { MakeEvenKernel(d), MakeRoughEvenKernel(d), delta };

        EvenKernelBank<double> bank(e.width, e.height, e.depth);
        for (size_t k = 0; k < kernels.size(); ++k) {
            EXPECT_EQ(bank.AddKernel(kernels[k].data()), k);
        }
        std::vector<double> out(kernels.size() * d.Total());
        const auto input = f;
        bank.Apply(f.data(), out.data());
        EXPECT_EQ(f, input); // the field is left alone

        for (size_t k = 0; k < kernels.size(); ++k) {
            const std::vector<double> got(out.begin() + k * d.Total(), out.begin() + (k + 1) * d.Total());
            ExpectClose(got, ReferenceCircularConvolution(f, kernels[k], d), 1e-12);
        }
    }
}

TEST(EvenKernelBank, RejectsOddKernelAndEmptyBank)
{
    const Dims d = Dims::Of(8, 4, 2);
    EvenKernelBank<double> bank(8, 4, 2);
    std::vector<double> f(d.Total(), 1.0), out(d.Total());
    EXPECT_THROW(bank.Apply(f.data(), out.data()), std::logic_error);
    std::vector<double> shifted(d.Total(), 0.0);
    shifted[1] = 1.0;
    EXPECT_THROW(bank.AddKernel(shifted.data()), std::invalid_argument);
    EXPECT_EQ(bank.KernelCount(), 0u);
}

TEST(EvenKernelBank, RejectsNonPowerOfTwoExtentUpFront)
{
    // Would otherwise throw inside an OpenMP region and terminate the process.
    EXPECT_THROW(EvenKernelBank<double>(12, 16, 16), std::invalid_argument);
    EXPECT_THROW(EvenKernelConvolution<double>(16, 6, 0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Convolution with any kernel (R-6)
// ---------------------------------------------------------------------------

TEST(CircularConvolution, MatchesDirectConvolution)
{
    for (const auto& e : kExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto f = MakeSignal(d.Total());
        const auto g = MakeConvolutionKernel(d, false);

        std::vector<double> actual = f;
        CircularConvolution<double> conv(e.width, e.height, e.depth);
        conv.SetKernel(g.data());
        conv.Apply(actual.data());

        ExpectClose(actual, ReferenceCircularConvolution(f, g, d), 1e-12);
    }
}

TEST(CircularConvolution, ApplyWithoutKernelThrows)
{
    std::vector<double> data(16, 1.0);
    CircularConvolution<double> conv(16, 0, 0);
    EXPECT_THROW(conv.Apply(data.data()), std::logic_error);
    EXPECT_THROW(conv.SetKernel(nullptr), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// The convolutions on the device (R-6 step 1.2)
// ---------------------------------------------------------------------------

namespace {

// Where the fused x pass runs (float, x >= 16: 3D, 2D, 1D, a degenerate y)
// and where the device path falls back to transform, product, transform
// (x < 16 in float; double always).
const Extent kDeviceExtents[] = {
    { 32, 16, 8 },
    { 16, 8, 4 },
    { 16, 2, 16 },
    { 64, 32, 0 },
    { 32, 0, 0 },
    { 8, 4, 2 },
};

/// One device convolution of `f` with `g` through the host pointer and
/// through a DeviceVolume; both must agree with the definition.
template <typename T, typename Conv>
void ExpectDeviceConvolution(Conv& conv, const std::vector<double>& f, const std::vector<double>& expected,
    double tolerance)
{
    std::vector<T> host(f.begin(), f.end());
    conv.Apply(host.data());
    ExpectClose(std::vector<double>(host.begin(), host.end()), expected, tolerance);

    std::vector<T> resident(f.begin(), f.end());
    DeviceVolume<T> volume(resident.size());
    volume.Upload(resident.data());
    conv.Apply(volume);
    volume.Download(resident.data());
    ExpectClose(std::vector<double>(resident.begin(), resident.end()), expected, tolerance);
}

template <typename T>
void ExpectDeviceConvolutionsMatchDefinition(double tolerance)
{
    for (const auto& e : kDeviceExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto f = MakeSignal(d.Total());
        {
            SCOPED_TRACE("any kernel");
            const auto g = MakeConvolutionKernel(d, false);
            const std::vector<T> kernel(g.begin(), g.end());
            CircularConvolution<T> conv(e.width, e.height, e.depth, Modes::GPUFactored);
            conv.SetKernel(kernel.data());
            ExpectDeviceConvolution<T>(conv, f, ReferenceCircularConvolution(f, g, d), tolerance);
        }
        {
            SCOPED_TRACE("even kernel");
            const auto g = MakeRoughEvenKernel(d);
            const std::vector<T> kernel(g.begin(), g.end());
            EvenKernelConvolution<T> conv(e.width, e.height, e.depth, Modes::GPUFactored);
            conv.SetKernel(kernel.data());
            ExpectDeviceConvolution<T>(conv, f, ReferenceCircularConvolution(f, g, d), tolerance);
        }
    }
}

} // namespace

TEST(DeviceConvolution, MatchesDefinition_Float_GPU)
{
    SKIP_IF_NO_GPU();
    ExpectDeviceConvolutionsMatchDefinition<float>(2e-5);
}

TEST(DeviceConvolution, MatchesDefinition_Double_GPU)
{
    SKIP_IF_NO_GPU();
    ExpectDeviceConvolutionsMatchDefinition<double>(1e-11);
}

// A volume of many batches in every pass, against the CPU convolutions in
// double (fast transforms, not the O(N^2) definition).
TEST(DeviceConvolution, AgreesWithCpuOnLargerVolume_GPU)
{
    SKIP_IF_NO_GPU();
    const size_t W = 128, H = 64, D = 32;
    const Dims d = Dims::Of(W, H, D);
    const auto f = MakeSignal(d.Total());
    for (const bool even : { false, true }) {
        SCOPED_TRACE(even ? "even kernel" : "any kernel");
        const auto g = MakeConvolutionKernel(d, even);
        std::vector<double> expected = f;
        std::vector<float> actual(f.begin(), f.end());
        const std::vector<float> kernel(g.begin(), g.end());
        if (even) {
            EvenKernelConvolution<double> cpu(W, H, D);
            cpu.SetKernel(g.data());
            cpu.Apply(expected.data());
            EvenKernelConvolution<float> gpu(W, H, D, Modes::GPUFactored);
            gpu.SetKernel(kernel.data());
            gpu.Apply(actual.data());
        } else {
            CircularConvolution<double> cpu(W, H, D);
            cpu.SetKernel(g.data());
            cpu.Apply(expected.data());
            CircularConvolution<float> gpu(W, H, D, Modes::GPUFactored);
            gpu.SetKernel(kernel.data());
            gpu.Apply(actual.data());
        }
        ExpectClose(std::vector<double>(actual.begin(), actual.end()), expected, 2e-5);
    }
}

TEST(DeviceConvolution, BankMatchesDefinition_GPU)
{
    SKIP_IF_NO_GPU();
    for (const auto& e : kDeviceExtents) {
        SCOPED_TRACE(Describe(e));
        const Dims d = Dims::Of(e.width, e.height, e.depth);
        const auto f = MakeSignal(d.Total());
        std::vector<double> delta(d.Total(), 0.0);
        delta[0] = 1.0;
        const std::vector<std::vector<double>> kernels = { MakeEvenKernel(d), MakeRoughEvenKernel(d), delta };

        EvenKernelBank<float> bank(e.width, e.height, e.depth, Modes::GPUFactored);
        for (const auto& k : kernels) {
            const std::vector<float> kf(k.begin(), k.end());
            bank.AddKernel(kf.data());
        }
        EXPECT_EQ(bank.KernelCount(), kernels.size());
        const std::vector<float> field(f.begin(), f.end());
        std::vector<float> out(kernels.size() * d.Total());
        bank.Apply(field.data(), out.data());
        for (size_t k = 0; k < kernels.size(); ++k) {
            const std::vector<double> got(out.begin() + k * d.Total(), out.begin() + (k + 1) * d.Total());
            ExpectClose(got, ReferenceCircularConvolution(f, kernels[k], d), 2e-5);
        }
    }
}

TEST(DeviceConvolution, ResidentApplyNeedsDeviceModeAndKernel_GPU)
{
    SKIP_IF_NO_GPU();
    DeviceVolume<float> volume(16 * 8 * 4);
    EvenKernelConvolution<float> cpu(16, 8, 4);
    EXPECT_THROW(cpu.Apply(volume), std::invalid_argument);
    CircularConvolution<float> gpu(16, 8, 4, Modes::GPUFactored);
    EXPECT_THROW(gpu.Apply(volume), std::logic_error);
    const Dims d = Dims::Of(16, 8, 4);
    const auto g = MakeConvolutionKernel(d, false);
    const std::vector<float> kernel(g.begin(), g.end());
    gpu.SetKernel(kernel.data());
    DeviceVolume<float> wrong(16 * 8 * 2);
    EXPECT_THROW(gpu.Apply(wrong), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Linear convolution (B-103)
// ---------------------------------------------------------------------------

namespace {

struct LinearCase {
    size_t width, height, depth; // the volume, any size
    size_t kw, kh, kd; // the kernel, odd (0 along an absent axis)
};

// Not powers of two, kernels wider than the volume, a 1-voxel axis, and
// anisotropic kernels: every way the padding could be too short.
const LinearCase kLinearCases[] = {
    { 13, 0, 0, 5, 0, 0 },
    { 3, 0, 0, 9, 0, 0 },
    { 7, 5, 0, 3, 5, 0 },
    { 5, 9, 0, 1, 3, 0 },
    { 6, 5, 3, 3, 3, 5 },
    { 4, 1, 7, 5, 1, 3 },
};

std::string Describe(const LinearCase& c)
{
    return std::to_string(c.width) + "x" + std::to_string(c.height) + "x" + std::to_string(c.depth) + " * "
         + std::to_string(c.kw) + "x" + std::to_string(c.kh) + "x" + std::to_string(c.kd);
}

/// A compact kernel, centre in the middle: symmetric along every axis, or not.
std::vector<double> MakeCompactKernel(const LinearCase& c, bool symmetric)
{
    const Dims k = Dims::Of(c.kw, c.kh, c.kd);
    std::vector<double> g(k.Total());
    for (size_t z = 0; z < k.depth; ++z) {
        for (size_t y = 0; y < k.height; ++y) {
            for (size_t x = 0; x < k.width; ++x) {
                const double dx = static_cast<double>(x) - static_cast<double>(k.width / 2);
                const double dy = static_cast<double>(y) - static_cast<double>(k.height / 2);
                const double dz = static_cast<double>(z) - static_cast<double>(k.depth / 2);
                double v = std::exp(-(dx * dx / 4.0 + dy * dy / 3.0 + dz * dz / 5.0));
                if (!symmetric) {
                    // Still positive everywhere, so normalising stays well posed.
                    v *= 1.0 + 0.1 * dx - 0.05 * dy + 0.02 * dz * dx;
                }
                g[k.Index(x, y, z)] = v;
            }
        }
    }
    return g;
}

/// The definition: out(x) = sum_t c(x-t) f(x-t) g(t) / [sum_t c(x-t) g(t)],
/// zeros outside the volume; no certainty means c = 1, no division unless
/// `normalized`.
std::vector<double> ReferenceLinearConvolution(const std::vector<double>& f, const std::vector<double>& g,
    const LinearCase& c, bool normalized, const std::vector<double>* certainty = nullptr)
{
    const Dims d = Dims::Of(c.width, c.height, c.depth);
    const Dims k = Dims::Of(c.kw, c.kh, c.kd);
    const long long r[3] = { static_cast<long long>(k.width / 2), static_cast<long long>(k.height / 2),
        static_cast<long long>(k.depth / 2) };
    const long long n[3] = { static_cast<long long>(d.width), static_cast<long long>(d.height),
        static_cast<long long>(d.depth) };
    std::vector<double> out(d.Total(), 0.0);
    for (long long z = 0; z < n[2]; ++z) {
        for (long long y = 0; y < n[1]; ++y) {
            for (long long x = 0; x < n[0]; ++x) {
                double num = 0.0;
                double den = 0.0;
                for (long long tz = -r[2]; tz <= r[2]; ++tz) {
                    for (long long ty = -r[1]; ty <= r[1]; ++ty) {
                        for (long long tx = -r[0]; tx <= r[0]; ++tx) {
                            const long long sx = x - tx, sy = y - ty, sz = z - tz;
                            if (sx < 0 || sy < 0 || sz < 0 || sx >= n[0] || sy >= n[1] || sz >= n[2]) {
                                continue;
                            }
                            const size_t s = d.Index(static_cast<size_t>(sx), static_cast<size_t>(sy),
                                static_cast<size_t>(sz));
                            const double w = g[k.Index(static_cast<size_t>(tx + r[0]),
                                static_cast<size_t>(ty + r[1]), static_cast<size_t>(tz + r[2]))];
                            const double cert = certainty ? (*certainty)[s] : 1.0;
                            num += cert * f[s] * w;
                            den += cert * w;
                        }
                    }
                }
                out[d.Index(static_cast<size_t>(x), static_cast<size_t>(y), static_cast<size_t>(z))]
                    = !normalized ? num : (den > 0.0 ? num / den : 0.0); // 0: the kernel covers no trusted voxel
            }
        }
    }
    return out;
}

template <typename T>
std::vector<double> RunLinear(const LinearCase& c, Edges edges, const std::vector<double>& f,
    const std::vector<double>& g, Modes mode = Modes::CPU, const std::vector<double>* certainty = nullptr)
{
    const std::vector<T> kernel(g.begin(), g.end());
    std::vector<T> data(f.begin(), f.end());
    std::vector<T> out(data.size());
    LinearConvolution<T> conv(c.width, c.height, c.depth, c.kw, c.kh, c.kd, edges, mode);
    conv.SetKernel(kernel.data());
    if (certainty) {
        const std::vector<T> cert(certainty->begin(), certainty->end());
        conv.Apply(data.data(), cert.data(), out.data());
    } else {
        conv.Apply(data.data(), out.data());
    }
    return std::vector<double>(out.begin(), out.end());
}

/// 0 on a slab and a scattered few voxels, 1 elsewhere, 0.5 on one voxel.
std::vector<double> MakeCertainty(const LinearCase& c)
{
    const Dims d = Dims::Of(c.width, c.height, c.depth);
    std::vector<double> cert(d.Total(), 1.0);
    for (size_t n = 0; n < cert.size(); ++n) {
        if (n % 7 == 3 || d.width - 1 == n % d.width) {
            cert[n] = 0.0;
        }
    }
    cert[0] = 0.5;
    return cert;
}

} // namespace

TEST(LinearConvolution, ZeroEdgesMatchDefinition)
{
    for (const auto& c : kLinearCases) {
        SCOPED_TRACE(Describe(c));
        const Dims d = Dims::Of(c.width, c.height, c.depth);
        const auto f = MakeSignal(d.Total());
        for (const bool symmetric : { true, false }) {
            SCOPED_TRACE(symmetric ? "symmetric kernel" : "asymmetric kernel");
            const auto g = MakeCompactKernel(c, symmetric);
            const auto expected = ReferenceLinearConvolution(f, g, c, false);
            ExpectClose(RunLinear<double>(c, Edges::Zero, f, g), expected, 1e-12);
            ExpectClose(RunLinear<float>(c, Edges::Zero, f, g), expected, 2e-6);
        }
    }
}

TEST(LinearConvolution, NormalizedEdgesMatchDefinition)
{
    for (const auto& c : kLinearCases) {
        SCOPED_TRACE(Describe(c));
        const Dims d = Dims::Of(c.width, c.height, c.depth);
        const auto f = MakeSignal(d.Total());
        const auto g = MakeCompactKernel(c, false);
        const auto expected = ReferenceLinearConvolution(f, g, c, true);
        ExpectClose(RunLinear<double>(c, Edges::Normalized, f, g), expected, 1e-11);
        ExpectClose(RunLinear<float>(c, Edges::Normalized, f, g), expected, 1e-5);

        // The point of normalising: a constant stays constant up to the border.
        const std::vector<double> constant(d.Total(), 2.5);
        ExpectClose(RunLinear<double>(c, Edges::Normalized, constant, MakeCompactKernel(c, true)), constant, 1e-12);
    }
}

TEST(LinearConvolution, CertaintyMatchesDefinitionAndHidesUntrustedVoxels)
{
    for (const auto& c : kLinearCases) {
        SCOPED_TRACE(Describe(c));
        const Dims d = Dims::Of(c.width, c.height, c.depth);
        const auto f = MakeSignal(d.Total());
        const auto g = MakeCompactKernel(c, true);
        const auto cert = MakeCertainty(c);
        const auto expected = ReferenceLinearConvolution(f, g, c, true, &cert);
        const auto actual = RunLinear<double>(c, Edges::Normalized, f, g, Modes::CPU, &cert);
        ExpectClose(actual, expected, 1e-11);
        ExpectClose(RunLinear<float>(c, Edges::Normalized, f, g, Modes::CPU, &cert), expected, 1e-5);

        // Garbage where the certainty is 0 must not change anything.
        std::vector<double> poisoned = f;
        for (size_t n = 0; n < poisoned.size(); ++n) {
            if (cert[n] == 0.0) {
                poisoned[n] = 1e6;
            }
        }
        ExpectClose(RunLinear<double>(c, Edges::Normalized, poisoned, g, Modes::CPU, &cert), actual, 1e-12);
    }
}

TEST(LinearConvolution, CertaintyCoveringNothingGivesZero)
{
    // One trusted voxel at the left end; a 3-tap kernel cannot reach it from
    // the right half, so the denominator there is 0 and so is the output.
    const LinearCase c { 8, 0, 0, 3, 0, 0 };
    std::vector<double> cert(8, 0.0);
    cert[0] = 1.0;
    const std::vector<double> f(8, 4.0);
    const auto out = RunLinear<double>(c, Edges::Normalized, f, { 1.0, 2.0, 1.0 }, Modes::CPU, &cert);
    EXPECT_NEAR(out[0], 4.0, 1e-12);
    EXPECT_NEAR(out[1], 4.0, 1e-12);
    for (size_t i = 2; i < 8; ++i) {
        EXPECT_EQ(out[i], 0.0) << "at " << i;
    }
}

TEST(LinearConvolution, PadsToThePowerOfTwoThatAvoidsWrapAround)
{
    LinearConvolution<float> a(500, 300, 0, 21, 3, 0);
    EXPECT_EQ(a.PaddedWidth(), 512u); // 500 + 10
    EXPECT_EQ(a.PaddedHeight(), 512u); // 300 + 1 > 256
    EXPECT_EQ(a.PaddedDepth(), 0u);
    LinearConvolution<float> b(3, 0, 0, 9, 0, 0);
    EXPECT_EQ(b.PaddedWidth(), 16u); // 2r + 1 = 9 > 3 + 4
    LinearConvolution<float> e(64, 64, 64, 1, 1, 1);
    EXPECT_EQ(e.PaddedWidth(), 64u); // a 1-tap kernel needs no padding
    EXPECT_EQ(e.Size(), 64u * 64u * 64u);
    EXPECT_EQ(e.KernelSize(), 1u);
}

TEST(LinearConvolution, AppliesInPlaceAndRejectsBadArguments)
{
    const LinearCase c { 6, 5, 3, 3, 3, 5 };
    const Dims d = Dims::Of(c.width, c.height, c.depth);
    const auto f = MakeSignal(d.Total());
    const auto g = MakeCompactKernel(c, false);
    std::vector<double> data = f;
    LinearConvolution<double> conv(c.width, c.height, c.depth, c.kw, c.kh, c.kd);
    EXPECT_THROW(conv.Apply(data.data(), data.data()), std::logic_error);
    conv.SetKernel(g.data());
    conv.Apply(data.data(), data.data());
    ExpectClose(data, ReferenceLinearConvolution(f, g, c, false), 1e-12);
    EXPECT_THROW(conv.Apply(f.data(), f.data(), data.data()), std::logic_error); // certainty, Zero edges

    EXPECT_THROW(LinearConvolution<double>(8, 8, 0, 4, 3, 0), std::invalid_argument); // even extent
    EXPECT_THROW(LinearConvolution<double>(8, 0, 0, 3, 3, 0), std::invalid_argument); // absent axis
    EXPECT_THROW(LinearConvolution<double>(0, 0, 0, 1, 0, 0), std::invalid_argument);
    EXPECT_THROW(LinearConvolution<double>(8, 0, 4, 3, 0, 3), std::invalid_argument);
    EXPECT_THROW(LinearConvolution<double>(8, 8, 8, 3, 3, 3, Edges::Zero, Modes::GPU), std::invalid_argument);
}

TEST(LinearConvolution, MatchesDefinitionOnDevice_GPU)
{
    SKIP_IF_NO_GPU();
    // Padded to 32 x 16 x 8 and 64 x 32: the fused x pass runs.
    const LinearCase cases[] = { { 27, 11, 5, 5, 3, 3 }, { 50, 20, 0, 9, 7, 0 } };
    for (const auto& c : cases) {
        SCOPED_TRACE(Describe(c));
        const Dims d = Dims::Of(c.width, c.height, c.depth);
        const auto f = MakeSignal(d.Total());
        for (const bool symmetric : { true, false }) {
            SCOPED_TRACE(symmetric ? "symmetric kernel" : "asymmetric kernel");
            const auto g = MakeCompactKernel(c, symmetric);
            ExpectClose(RunLinear<float>(c, Edges::Zero, f, g, Modes::GPUFactored),
                ReferenceLinearConvolution(f, g, c, false), 2e-5);
            const auto cert = MakeCertainty(c);
            ExpectClose(RunLinear<float>(c, Edges::Normalized, f, g, Modes::GPUFactored, &cert),
                ReferenceLinearConvolution(f, g, c, true, &cert), 2e-5);
        }
    }
}
