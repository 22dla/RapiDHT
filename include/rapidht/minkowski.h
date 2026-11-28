/*
 * Project: RapiDHT
 * File: include/rapidht/minkowski.h
 * Brief: Minkowski functionals of 3D binary volumes and their local densities,
 *        from the 2x2x2 configurations around each lattice vertex; the surface
 *        normal tensor from the gradient of the smoothed body.
 * Author: Volkov Evgeny Aleksandrovich, volkov22dla@yandex.ru
 */

#ifndef RAPIDHT_MINKOWSKI_H
#define RAPIDHT_MINKOWSKI_H

#include <rapidht/convolution.h>
#include <rapidht/filters.h>

#include <cstddef>
#include <cstdint>
#include <vector>

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
 * @brief Which digital topology the Euler characteristic counts (B-206).
 *
 * The other three functionals are always those of the union of closed cubes.
 */
enum class EulerAdjacency {
    /// The union of closed cubes: foreground 26-adjacent, background 6-adjacent.
    Cube26,
    /**
     * The voxel centres as the vertices of the Freudenthal (Kuhn) triangulation of the
     * lattice along the main diagonal (1, 1, 1): every 2x2x2 cell is split into the six
     * tetrahedra 0 -> e_a -> e_a + e_b -> (1, 1, 1), and chi = vertices - edges +
     * triangles - tetrahedra of the simplices whose vertices are all foreground. Foreground
     * and background are then both 14-adjacent (neighbours along e_a, e_a + e_b and
     * (1, 1, 1)), so chi of a body equals chi of its complement on the periodic lattice,
     * which Cube26 does not give. It does not cure the bias on nearly touching grains
     * (see MinkowskiConfigurationWeights; B-206): on Boolean models of balls it is as
     * biased as Cube26. Not invariant under axis flips: the diagonal is fixed.
     */
    Kuhn14,
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
 * V has no such bias. chi is exact for the union of cubes, but that union's topology
 * differs from the continuum's where grains nearly touch: two balls closer than about
 * 1.5 voxels touch through edges and corners in many places along the gap, each place
 * after the first a spurious tunnel (chi down to -12 instead of 2 for one such pair of
 * radius 16). Such pairs get rarer as the resolution grows and the tunnels per pair
 * more numerous, so the bias stays: on Boolean models of balls chi is 0.23-0.39 per
 * ball below the exact value at volume fractions 0.2-0.5 for R = 4 to 16 voxels,
 * 0.08-0.15 with the foreground 6-adjacent (chi of the complement), 0.25-0.46 with
 * EulerAdjacency::Kuhn14 (B-204, B-206, tools/b204_miles.py).
 *
 * With EulerAdjacency::Kuhn14 only the Euler weight differs: the simplices of the cell
 * with all vertices set, each weighted (-1)^dimension over the number of cells sharing
 * it (a vertex 8, an axis edge 4, a face diagonal and a face triangle 2, the rest 1).
 *
 * @param configuration 0..255.
 * @throws std::out_of_range for a larger value.
 */
MinkowskiValues MinkowskiConfigurationWeights(unsigned configuration, EulerAdjacency adjacency = EulerAdjacency::Cube26);

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
    MinkowskiBoundary boundary = MinkowskiBoundary::Periodic, EulerAdjacency adjacency = EulerAdjacency::Cube26);

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
 * window; MinkowskiFields does this for a bank of windows.
 *
 * @param out 4 * Width*Height*Depth elements.
 * @tparam T float or double.
 */
template <typename T>
void MinkowskiDensities(const uint8_t* volume, size_t width, size_t height, size_t depth, T* out,
    DensityCentring centring = DensityCentring::Voxel, EulerAdjacency adjacency = EulerAdjacency::Cube26);

/// @brief Shape of the window over which the local functionals are taken.
enum class WindowShape {
    /// 1 at the voxels whose centre lies within the radius, 0 elsewhere. The radius must be
    /// below half of every extent, or the ball would overlap its own periodic image.
    Ball,
    /// exp(-|x|^2 / (2 r^2)) with the radius as the standard deviation, summed over the
    /// periodic images, so any radius is allowed and the window stays smooth and even.
    Gaussian,
};

/**
 * @brief A window in the wrapped layout (centre at index 0), even along every axis.
 * @param width, height, depth Extents, as for HartleyTransform.
 * @throws std::invalid_argument for a non-positive radius, or a Ball that does not fit.
 */
template <typename T>
std::vector<T> MakeWindow(size_t width, size_t height, size_t depth, WindowShape shape, double radius);

/// @brief Whether a field holds the windowed sum of a density or its window average.
enum class WindowNormalisation {
    Sum, ///< The functional of the part of the body inside the window (weighted, for Gaussian).
    Mean, ///< The same divided by the window's total weight: per-voxel densities, e.g. the
          ///< local volume fraction, comparable across window sizes.
};

/**
 * @brief Local Minkowski functionals on a bank of windows (B-202), periodic boundary.
 *
 * The four voxel-centred densities (MinkowskiDensities) are each convolved with every
 * window through EvenKernelBank: 4 (1 + K) separable transforms for K windows, not the
 * 22 or 256 convolutions of a configuration-by-configuration approach. With an even
 * window and voxel centring every field commutes with the symmetries of the voxel grid,
 * the central inversion included.
 *
 * Output layout: 4 * K fields of Size() values, functional-major,
 * out[(f * K + s) * Size() + n] for functional f (volume, surface, mean curvature,
 * Euler characteristic) and window s, in the order of the radii.
 *
 * Usage:
 * @code
 *   MinkowskiFields<float> mf(64, 64, 64, WindowShape::Ball, { 2, 4, 8 });
 *   std::vector<float> out(mf.FieldCount() * mf.Size());
 *   mf.Compute(solid.data(), out.data());
 * @endcode
 *
 * @tparam T float or double. Extents must be powers of two (HartleyTransform).
 */
template <typename T>
class MinkowskiFields {
public:
    MinkowskiFields(size_t width, size_t height, size_t depth, WindowShape shape, const std::vector<double>& radii,
        WindowNormalisation normalisation = WindowNormalisation::Mean,
        EulerAdjacency adjacency = EulerAdjacency::Cube26);

    /// @brief Fills out with FieldCount() fields of the binary volume (nonzero = foreground).
    void Compute(const uint8_t* volume, T* out);

    size_t ScaleCount() const noexcept { return _windowWeights.size(); }
    size_t FieldCount() const noexcept { return 4 * ScaleCount(); }
    size_t Size() const noexcept { return _size; }

    /// @brief Total weight of window s (the voxel count, for a Ball).
    double WindowWeight(size_t s) const { return _windowWeights.at(s); }

private:
    size_t _width;
    size_t _height;
    size_t _depth;
    size_t _size;
    WindowNormalisation _normalisation;
    EulerAdjacency _adjacency;
    EvenKernelBank<T> _bank;
    std::vector<double> _windowWeights;
    std::vector<T> _densities;
};

/**
 * @brief A symmetric 3x3 tensor: the surface normal tensor integral of n (x) n dS.
 *
 * Its trace is the surface area, and for an isotropic structure it is S/3 times the
 * identity; the ratio of its smallest to its largest eigenvalue is the usual anisotropy
 * index (1 for isotropic, 0 for a stack of plates). karambola's W1^{0,2} is a third of it.
 */
struct SurfaceTensor {
    double xx = 0.0;
    double yy = 0.0;
    double zz = 0.0;
    double xy = 0.0;
    double xz = 0.0;
    double yz = 0.0;

    double Trace() const noexcept { return xx + yy + zz; }
};

/**
 * @brief The surface normal tensor of a binary volume (B-205): integral of n (x) n dS,
 *        globally and as a density field, periodic boundary.
 *
 * On the union of cubes every normal is an axis, so that tensor would be diagonal
 * whatever the body's orientation; instead the body is smoothed, psi = G_sigma * 1_X,
 * and the surface measure and normal are read off its gradient, |grad psi| dV -> dS and
 * grad psi / |grad psi| -> n as sigma -> 0:
 *
 *     W = integral of grad psi (x) grad psi / |grad psi| dV.
 *
 * The gradient is the convolution with the derivative of a Gaussian (physical sigma,
 * truncated at 4 sigma, scaled so that a linear ramp has exactly its slope), three
 * CircularConvolution calls. For a flat interface W is exact: a slab with normal z gives
 * 2 A on zz and 0 elsewhere; a digitised slab tilted to (1, 1, 0) comes out within 1e-4.
 * On a sphere of radius R the estimate is low by about (sigma / R)^2 (R = 16 voxels,
 * sigma = 1.5: 0.96 %); below a voxel the staircase shows (+1 % at sigma = 0.7), so
 * sigma of 1 to 1.5 voxels is the sensible range. The trace is an estimate of the
 * surface area without the 3/2 bias of MinkowskiFunctionals.
 *
 * Densities: six planes in the order xx, yy, zz, xy, xz, yz, each voxel holding its
 * share of the integral (the integrand times the voxel's volume), so every plane sums
 * to that component of Compute; convolve them with a window (EvenKernelBank) for local
 * anisotropy fields.
 *
 * @tparam T float or double. 3D only; extents must be powers of two.
 */
template <typename T>
class SurfaceNormalTensor {
public:
    /**
     * @param sigma Smoothing in the unit of `spacing`; positive.
     * @param spacing Voxel edge lengths (anisotropic voxels allowed).
     * @param mode Modes::CPU or Modes::GPUFactored, for the convolutions.
     * @throws std::invalid_argument on a 1D or 2D extent, a non-power of two, a bad
     *         sigma or spacing, or another mode.
     */
    SurfaceNormalTensor(size_t width, size_t height, size_t depth, double sigma, Spacing spacing = {},
        Modes mode = Modes::CPU);

    /// @brief The integral of n (x) n dS of the body (nonzero = foreground), in units of spacing^2.
    SurfaceTensor Compute(const uint8_t* volume);

    /// @brief The six density planes (6 * Size() values, xx, yy, zz, xy, xz, yz).
    void Densities(const uint8_t* volume, T* out);

    size_t Size() const noexcept { return _size; }

private:
    size_t _size;
    double _voxelVolume;
    CircularConvolution<T> _dx;
    CircularConvolution<T> _dy;
    CircularConvolution<T> _dz;
    std::vector<T> _gradient; ///< three planes: d/dx, d/dy, d/dz of psi
    std::vector<T> _densities; ///< Compute's scratch
};

} // namespace RapiDHT

#endif // RAPIDHT_MINKOWSKI_H
