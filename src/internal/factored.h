/*
 * Project: RapiDHT
 * File: src/internal/factored.h
 * Brief: R-4 -- the DHT of length N = N1 * N2 as two small dense multiplies.
 *
 * The matrix GPU path multiplies by a dense N x N cas matrix per axis: 2N
 * operations per element, 1024 at N = 512, which no tuning brings near cuFFT.
 * Cooley-Tukey written as a matrix factorisation keeps the multiplies but
 * shrinks them. With n = N2*n1 + n2 and k = k1 + N1*k2 the phase splits as
 * 2*pi*n*k/N = A + B, A = 2*pi*n1*k1/N1, B = 2*pi*n2*k/N, and
 * cas(A + B) = cos(B) cas(A) + sin(B) cas(-A) gives
 *
 *   step 1  Y[k1][n2] = sum_n1 C1[k1][n1] x[N2*n1 + n2]          DHT of length N1
 *   step 2  H[k1 + N1*k2] = sum_n2 Y[k1][n2] cos(B) + Y[-k1][n2] sin(B)
 *
 * Step 2 couples k1 with -k1, which the Fourier version does not have to do.
 * Taking the pairs (k1, N1 - k1) together turns it into a twiddle, a DHT of
 * length N2 per column and a recombination:
 *
 *   P = d*Y[k1] + c*Y[-k1],  Q = c*Y[k1] - d*Y[-k1],
 *       c = cas(T)/2, d = cas(-T)/2, T = 2*pi*n2*k1/N          (twiddle)
 *   DP = C2 . P,  DQ = C2 . Q                                  (step 2)
 *   H[ k1 + N1*k2] = DP[k2]  + DQ[-k2]
 *   H[-(k1 + N1*k2)] = DP[-k2] - DQ[k2]                        (recombination)
 *
 * P goes into column k1 and Q into column N1 - k1, so the pairs pack into the
 * N1 columns exactly. Column 0 needs no twiddle. Column N1/2 is its own mirror
 * and takes a half-shifted matrix instead: H[N1/2 + N1*k2] = sum_n2
 * cas(2*pi*n2*(k2 + 1/2)/N2) Y[N1/2][n2].
 *
 * Cost per element: 2*N1 + 2*N2 multiply-adds -- 96 at 512 = 32 * 16, against
 * 1024 for the dense matrix -- plus the twiddle. Everything here works on a
 * block of `lanes` lines held side by side, element idx of line l at
 * [idx * lanes + l], which is the layout of the CPU backend's blocks and of
 * the shared-memory tile the CUDA kernel uses.
 *
 * This header holds the tables, shared by both, and a plain host reference of
 * the kernel's steps, which the tests check against the O(N^2) definition.
 */

#ifndef RAPIDHT_INTERNAL_FACTORED_H
#define RAPIDHT_INTERNAL_FACTORED_H

#include <cstddef>
#include <vector>

namespace RapiDHT {
namespace internal {

/// N = n1 * n2 with n1 >= n2, both powers of two, n1 = 2^ceil(log2(N)/2).
struct FactoredSplit {
    size_t n = 1;
    size_t n1 = 1;
    size_t n2 = 1;
};

/// Throws std::invalid_argument unless n is a power of two.
FactoredSplit SplitLength(size_t n);

/// Everything the two steps multiply by, computed in double and stored as T.
template <typename T>
struct FactoredTables {
    FactoredSplit split;
    std::vector<T> c1; ///< [k1][n1] cas(2 pi n1 k1 / N1), N1 x N1
    std::vector<T> c2; ///< [k2][n2] cas(2 pi n2 k2 / N2), N2 x N2
    std::vector<T> cHalf; ///< [k2][n2] cas(2 pi n2 (k2 + 1/2) / N2), for column N1/2
    std::vector<T> twC; ///< [k1][n2] cas(T)/2 for k1 in [0, N1/2), T = 2 pi n2 k1 / N
    std::vector<T> twD; ///< [k1][n2] cas(-T)/2, same shape
};

template <typename T>
FactoredTables<T> MakeFactoredTables(size_t n);

/**
 * @brief The DHT of every line in a block, in place, by the factored steps.
 * @param tile   n * lanes values, element idx of line l at [idx * lanes + l].
 * @param work   scratch of the same size.
 * @param lanes  number of lines in the block.
 */
template <typename T>
void FactoredDhtLanes(T* tile, T* work, size_t lanes, const FactoredTables<T>& tables);

} // namespace internal
} // namespace RapiDHT

#endif // RAPIDHT_INTERNAL_FACTORED_H
