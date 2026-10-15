/*
 * Project: RapiDHT
 * File: include/rapidht/minkowski.h
 * Brief: Minkowski functionals of 3D binary volumes and their local densities,
 *        from the 2x2x2 configurations around each lattice vertex.
 * Author: Volkov Evgeny Aleksandrovich, volkov22dla@yandex.ru
 */

#ifndef RAPIDHT_MINKOWSKI_H
#define RAPIDHT_MINKOWSKI_H

#include <cstddef>
#include <cstdint>

namespace RapiDHT {

/**
 * @brief The four Minkowski functionals of a body in 3D, in voxel units.
 *
 * In the Steiner normalisation V(X + eps B) = V + S eps + M eps^2 + (4 pi / 3) chi eps^3:
 *  - volume        V,   voxel^3;
 *  - surface       S,   voxel^2;
 *  - meanCurvature M = integral of (1/r1 + 1/r2) / 2 over the surface, voxel;
 *  - euler         chi, dimensionless (the integral of the Gaussian curvature is 4 pi chi).
 */
struct MinkowskiValues {
    double volume = 0.0;
    double surface = 0.0;
    double meanCurvature = 0.0;
    double euler = 0.0;
};

/**
 * @brief Contribution of one lattice vertex, given the 8 voxels around it.
 *
 * The foreground is the union of closed unit cubes, one per set voxel (Michielsen &
 * De Raedt, Phys. Rep. 347 (2001) 461): with n_c cubes, n_f faces, n_e edges and
 * n_v vertices of that cubical complex,
 *
 *     V = n_c,  S = -6 n_c + 2 n_f,  M = pi (3 n_c - 2 n_f + n_e)  (= 2 pi x mean breadth),
 *     chi = -n_c + n_f - n_e + n_v,
 *
 * and a cell shared by k vertices is split equally between them (cube 1/8, face 1/4,
 * edge 1/2). Summing the contributions over all vertices gives the exact functionals
 * of the union of cubes. Closed cubes mean the foreground is 26-connected and the
 * background 6-connected.
 *
 * The configuration index is sum_i 2^i s_i over the voxels at offsets
 * (i & 1, (i >> 1) & 1, (i >> 2) & 1) along (width, height, depth) from the vertex,
 * s_i = 1 for the foreground; this is formula (2.30) of the thesis. The returned
 * values times (8, 4, 8/pi, 8) reproduce table 2 there: 8 M0, 24 M2 = 4 S,
 * 24 M3 = 8 M / pi, 8 M4 = 8 chi, M2 and M3 being normalised as S/6 and M/(3 pi).
 *
 * On a digitised smooth surface S and M are biased: for isotropic orientations both
 * come out about 3/2 of the true value (the cubes' surface follows the staircase).
 * V and chi have no such bias.
 *
 * @param configuration 0..255.
 * @throws std::out_of_range for a larger value.
 */
MinkowskiValues MinkowskiConfigurationWeights(unsigned configuration);

/// @brief What lies outside the volume.
enum class MinkowskiBoundary {
    Periodic, ///< The volume tiles space: the functionals are per period cell.
    Empty, ///< Background: the functionals of the body as it is, edges included.
};

/**
 * @brief Global Minkowski functionals of a binary volume.
 *
 * @param volume Width*Height*Depth voxels, idx = i + W * (j + H * k), as for
 *        HartleyTransform; any nonzero value is foreground.
 * @param width, height, depth Extents; all must be at least 2 for Periodic, 1 for Empty.
 */
MinkowskiValues MinkowskiFunctionals(const uint8_t* volume, size_t width, size_t height, size_t depth,
    MinkowskiBoundary boundary = MinkowskiBoundary::Periodic);

/// @brief Where the density fields are sampled.
enum class DensityCentring {
    /// Index (i,j,k) holds the vertex at the corner (i+1/2, j+1/2, k+1/2) between voxels
    /// i..i+1, j..j+1, k..k+1: the raw contributions, half a voxel off the voxel grid.
    Vertex,
    /// Index (i,j,k) holds the mean of the 8 vertex contributions at the corners of voxel
    /// (i,j,k). Centred on the voxels, so the densities commute with every symmetry of the
    /// voxel grid (axis flips, permutations, the central inversion) applied to the volume.
    Voxel,
};

/**
 * @brief Local densities of the four Minkowski functionals, periodic boundary.
 *
 * Writes four planes of Width*Height*Depth values each, in the order volume, surface,
 * mean curvature, Euler characteristic, every plane in the input's layout. Each plane
 * sums to the corresponding MinkowskiFunctionals(..., Periodic) value. Convolving a
 * plane with a window (EvenKernelConvolution) gives the field of that functional in the
 * window, the construction of B-202.
 *
 * @param out 4 * Width*Height*Depth elements.
 * @tparam T float or double.
 */
template <typename T>
void MinkowskiDensities(const uint8_t* volume, size_t width, size_t height, size_t depth, T* out,
    DensityCentring centring = DensityCentring::Voxel);

} // namespace RapiDHT

#endif // RAPIDHT_MINKOWSKI_H
