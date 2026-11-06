/*
 * Project: RapiDHT
 * File: src/factored.cpp
 * Brief: R-4 tables and the host reference of the factored DHT steps.
 *
 * The reference is written step by step in the order the CUDA kernel takes
 * them, not for speed: its job is to pin the index arithmetic down against the
 * definition before any of it runs on a device.
 */

#include "internal/factored.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace RapiDHT {
namespace internal {

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

double Cas(double t)
{
    return std::cos(t) + std::sin(t);
}

} // namespace

FactoredSplit SplitLength(size_t n)
{
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("SplitLength: length must be a power of two, got " + std::to_string(n) + ".");
    }
    size_t log2n = 0;
    while ((size_t { 1 } << log2n) < n) {
        ++log2n;
    }
    FactoredSplit split;
    split.n = n;
    split.n1 = size_t { 1 } << ((log2n + 1) / 2);
    split.n2 = n / split.n1;
    return split;
}

template <typename T>
FactoredTables<T> MakeFactoredTables(size_t n)
{
    FactoredTables<T> t;
    t.split = SplitLength(n);
    const size_t n1 = t.split.n1;
    const size_t n2 = t.split.n2;
    const double N = static_cast<double>(n);

    t.c1.resize(n1 * n1);
    for (size_t k = 0; k < n1; ++k) {
        for (size_t j = 0; j < n1; ++j) {
            // k*j reduced mod n1 first: the angle stays small and exact.
            const double phase = kTwoPi * static_cast<double>((k * j) % n1) / static_cast<double>(n1);
            t.c1[k * n1 + j] = static_cast<T>(Cas(phase));
        }
    }
    t.c2.resize(n2 * n2);
    t.cHalf.resize(n2 * n2);
    for (size_t k = 0; k < n2; ++k) {
        for (size_t j = 0; j < n2; ++j) {
            const double phase = kTwoPi * static_cast<double>((k * j) % n2) / static_cast<double>(n2);
            t.c2[k * n2 + j] = static_cast<T>(Cas(phase));
            const double half = kTwoPi * static_cast<double>(j) * (static_cast<double>(k) + 0.5)
                              / static_cast<double>(n2);
            t.cHalf[k * n2 + j] = static_cast<T>(Cas(half));
        }
    }
    const size_t pairs = n1 / 2 == 0 ? 1 : n1 / 2;
    t.twC.resize(pairs * n2);
    t.twD.resize(pairs * n2);
    for (size_t k1 = 0; k1 < pairs; ++k1) {
        for (size_t j = 0; j < n2; ++j) {
            const double phase = kTwoPi * static_cast<double>(j * k1) / N;
            t.twC[k1 * n2 + j] = static_cast<T>(0.5 * Cas(phase));
            t.twD[k1 * n2 + j] = static_cast<T>(0.5 * Cas(-phase));
        }
    }
    return t;
}

template <typename T>
void FactoredDhtLanes(T* tile, T* work, size_t lanes, const FactoredTables<T>& tables)
{
    const size_t n = tables.split.n;
    const size_t n1 = tables.split.n1;
    const size_t n2 = tables.split.n2;
    if (n == 1) {
        return; // the DHT of one value is that value
    }
    auto at = [lanes](T* base, size_t index, size_t lane) -> T& { return base[index * lanes + lane]; };

    // Step 1: Y[k1][n2] = sum_n1 C1[k1][n1] x[N2*n1 + n2], into work.
    for (size_t k1 = 0; k1 < n1; ++k1) {
        for (size_t j2 = 0; j2 < n2; ++j2) {
            for (size_t l = 0; l < lanes; ++l) {
                T sum = 0;
                for (size_t j1 = 0; j1 < n1; ++j1) {
                    sum += tables.c1[k1 * n1 + j1] * at(tile, n2 * j1 + j2, l);
                }
                at(work, k1 * n2 + j2, l) = sum;
            }
        }
    }

    // Twiddle the pairs (k1, N1 - k1), 0 < k1 < N1/2, in place in work:
    // P into column k1, Q into column N1 - k1.
    for (size_t k1 = 1; k1 < n1 / 2; ++k1) {
        const size_t m1 = n1 - k1;
        for (size_t j2 = 0; j2 < n2; ++j2) {
            const T c = tables.twC[k1 * n2 + j2];
            const T d = tables.twD[k1 * n2 + j2];
            for (size_t l = 0; l < lanes; ++l) {
                const T yp = at(work, k1 * n2 + j2, l);
                const T ym = at(work, m1 * n2 + j2, l);
                at(work, k1 * n2 + j2, l) = d * yp + c * ym;
                at(work, m1 * n2 + j2, l) = c * yp - d * ym;
            }
        }
    }

    // Step 2: a DHT of length N2 down every column, into tile; the
    // half-shifted matrix for column N1/2.
    for (size_t k1 = 0; k1 < n1; ++k1) {
        const T* matrix = (n1 >= 2 && k1 == n1 / 2) ? tables.cHalf.data() : tables.c2.data();
        for (size_t k2 = 0; k2 < n2; ++k2) {
            for (size_t l = 0; l < lanes; ++l) {
                T sum = 0;
                for (size_t j2 = 0; j2 < n2; ++j2) {
                    sum += matrix[k2 * n2 + j2] * at(work, k1 * n2 + j2, l);
                }
                at(tile, k1 * n2 + k2, l) = sum;
            }
        }
    }

    // Recombination into output order k = k1 + N1*k2, into work.
    for (size_t k2 = 0; k2 < n2; ++k2) {
        const size_t mk2 = (n2 - k2) % n2;
        for (size_t l = 0; l < lanes; ++l) {
            at(work, n1 * k2, l) = at(tile, k2, l); // column 0
            if (n1 >= 2) {
                at(work, n1 / 2 + n1 * k2, l) = at(tile, (n1 / 2) * n2 + k2, l);
            }
        }
        for (size_t k1 = 1; k1 < n1 / 2; ++k1) {
            const size_t m1 = n1 - k1;
            for (size_t l = 0; l < lanes; ++l) {
                const T dpK = at(tile, k1 * n2 + k2, l);
                const T dpM = at(tile, k1 * n2 + mk2, l);
                const T dqK = at(tile, m1 * n2 + k2, l);
                const T dqM = at(tile, m1 * n2 + mk2, l);
                at(work, k1 + n1 * k2, l) = dpK + dqM;
                // -(k1 + N1*k2) mod N = (N1 - k1) + N1*(N2 - 1 - k2)
                at(work, m1 + n1 * (n2 - 1 - k2), l) = dpM - dqK;
            }
        }
    }

    for (size_t i = 0; i < n * lanes; ++i) {
        tile[i] = work[i];
    }
}

template struct FactoredTables<float>;
template struct FactoredTables<double>;
template FactoredTables<float> MakeFactoredTables<float>(size_t);
template FactoredTables<double> MakeFactoredTables<double>(size_t);
template void FactoredDhtLanes<float>(float*, float*, size_t, const FactoredTables<float>&);
template void FactoredDhtLanes<double>(double*, double*, size_t, const FactoredTables<double>&);

} // namespace internal
} // namespace RapiDHT
