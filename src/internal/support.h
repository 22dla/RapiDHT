/*
 * Project: RapiDHT
 * File: src/internal/support.h
 * Brief: Small helpers shared between the translation units of the library.
 *
 * Not installed and not part of the public API.
 */

#ifndef RAPIDHT_INTERNAL_SUPPORT_H
#define RAPIDHT_INTERNAL_SUPPORT_H

#include <rapidht/config_flags.h>

#include <cstddef>
#include <stdexcept>

#ifdef RAPIDHT_WITH_MPI
#include <mpi.h>
#endif

namespace RapiDHT {
namespace internal {

/// Exact power-of-two test. The obvious alternative, comparing ceil(log2(n))
/// with floor(log2(n)), relies on floating point being exact at the boundary.
constexpr bool IsPowerOfTwo(size_t n) noexcept
{
    return n != 0 && (n & (n - 1)) == 0;
}

[[noreturn]] inline void ThrowGpuUnavailable()
{
    throw std::runtime_error(
        "RapiDHT was built without CUDA support (RAPIDHT_WITH_CUDA=OFF): "
        "Modes::GPU is unavailable. Reconfigure with -DRAPIDHT_WITH_CUDA=ON.");
}

/// Rank and size of MPI_COMM_WORLD, or the trivial single-process values when
/// MPI is either disabled at build time or not initialised by the host program.
struct MpiContext {
    int rank = 0;
    int size = 1;
    bool active = false;
};

inline MpiContext QueryMpi()
{
    MpiContext ctx;
#ifdef RAPIDHT_WITH_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        ctx.active = true;
        MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);
    }
#endif
    return ctx;
}

inline void MpiBarrier(const MpiContext& ctx)
{
#ifdef RAPIDHT_WITH_MPI
    if (ctx.active) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
#else
    (void)ctx;
#endif
}

#ifdef RAPIDHT_WITH_MPI
/*
 * Maps the element type onto its MPI datatype. MPI_DOUBLE used to be hardcoded
 * inside the class template, so HartleyTransform<float> told MPI that each
 * element was eight bytes wide: the gather read past the end of the send buffer
 * and scattered the result over twice the intended receive area.
 */
template <typename T>
struct MpiDatatype;

template <>
struct MpiDatatype<float> {
    static MPI_Datatype value() { return MPI_FLOAT; }
};

template <>
struct MpiDatatype<double> {
    static MPI_Datatype value() { return MPI_DOUBLE; }
};
#endif

} // namespace internal
} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_SUPPORT_H
