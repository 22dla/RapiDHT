/*
 * Project: RapiDHT
 * File: benchmarks/cufft_baseline.h
 * Brief: The 2D and 3D Hartley transform computed through cuFFT, as a baseline.
 *
 * This is the competitor the matrix formulation has to answer for. cuFFT
 * computes the Fourier transform, but for real input the Hartley transform
 * follows from it by a single linear pass:
 *
 *     X(k) = sum x(n) * (cos - i*sin)   =>   Re(X) - Im(X) = sum x(n) * cas
 *
 * so "cuFFT plus an O(N) conversion" is a legitimate way to compute exactly
 * what this library computes, and the obvious thing a reviewer will ask about.
 *
 * Benchmark scaffolding, not part of the library: it lives here so that the
 * public headers stay free of CUDA, and it is float-only because that is the
 * case being compared. 2D images were added for the full table of R-4 step 5.2.
 */

#ifndef RAPIDHT_BENCH_CUFFT_BASELINE_H
#define RAPIDHT_BENCH_CUFFT_BASELINE_H

#include <cstddef>
#include <memory>

namespace rapidht_bench {

class CufftHartley {
public:
    /// Plans an R2C transform for a W x H x D volume, W contiguous; depth == 0
    /// is a W x H image.
    CufftHartley(int width, int height, int depth);
    ~CufftHartley();

    CufftHartley(const CufftHartley&) = delete;
    CufftHartley& operator=(const CufftHartley&) = delete;

    /// Transforms a device-resident real volume in place, leaving the true
    /// multidimensional Hartley transform in it.
    void Forward(float* deviceInOut);

    /// Only the R2C transform into the internal spectrum buffer, no Hartley
    /// conversion. The lower bound of what cuFFT spends on this volume: the
    /// R-4 track measures it to know how much of the baseline is cuFFT itself
    /// and how much the conversion pass.
    void ForwardSpectrumOnly(const float* deviceIn);

    /// Device memory this baseline needs beyond the caller's volume: the
    /// complex spectrum plus whatever cuFFT asks for as workspace.
    size_t ExtraDeviceBytes() const;

    /// The two parts of ExtraDeviceBytes.
    size_t SpectrumBytes() const;
    size_t WorkspaceBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/*
 * R-5 step 0.2: the same Hartley transform through cuFFT, but from an integer
 * volume (uint8 or uint16, as a tomograph writes it) into a float one. cuFFT has
 * no integer input type, so there are three ways to feed it, and the strongest
 * of them is the baseline R-5 has to beat:
 *
 *   Convert         a kernel converts the input into the float output buffer,
 *                   R2C reads it from there (one extra pass over the volume);
 *   LegacyCallback  a load callback reads the integers inside cuFFT's first
 *                   kernel (CUFFT_CB_LD_REAL; needs the static cuFFT and
 *                   separate compilation, so it lives in its own shared library,
 *                   rapidht_cufft_callback, that hides that cuFFT);
 *   LtoCallback     the same callback, compiled to LTO-IR and linked into the
 *                   plan by cuFFT at run time (cufftXtSetJITCallback, CUDA 12.6
 *                   Update 2 and later), with the ordinary shared cuFFT.
 *
 * Every variant subtracts `offset` from each voxel as it reads it -- exact in
 * float for integers of up to 24 bits -- and adds offset * N back to the zero
 * coefficient in the Hartley pass, at no cost. The result is the transform of
 * the input itself. The input is never written. Extra device memory is the
 * same for all three: the half spectrum and cuFFT's workspace, besides the
 * caller's input and float output.
 */
enum class IntInput { U8,
    U16 };
enum class IntLoad { Convert,
    LegacyCallback,
    LtoCallback };

class CufftIntHartley {
public:
    virtual ~CufftIntHartley() = default;

    /// deviceIn: W*H*D integers of the type given at construction;
    /// deviceOut: W*H*D floats. Synchronises before it returns.
    virtual void Forward(const void* deviceIn, float* deviceOut) = 0;

    /// The half spectrum plus cuFFT's workspace.
    virtual size_t ExtraDeviceBytes() const = 0;
};

/// Convert and LtoCallback; LegacyCallback is refused here (MakeCufftIntCallback).
/// Throws std::runtime_error when cuFFT refuses the plan or the memory.
std::unique_ptr<CufftIntHartley> MakeCufftIntHartley(
    int width, int height, int depth, IntInput input, IntLoad load, double offset);

/// LegacyCallback. Defined in cufft_int_callback.cu, which only targets that
/// link rapidht_cufft_callback have.
std::unique_ptr<CufftIntHartley> MakeCufftIntCallback(
    int width, int height, int depth, IntInput input, double offset);

} // namespace rapidht_bench

#endif // RAPIDHT_BENCH_CUFFT_BASELINE_H
