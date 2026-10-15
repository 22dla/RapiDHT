/*
 * Project: RapiDHT
 * File: tests/test_minkowski.cpp
 * Brief: Minkowski functionals and densities, checked against two independent
 *        references -- tables 1 and 2 of the thesis, and a direct count of the
 *        cells of the cubical complex -- and against bodies with known values.
 */

#include <rapidht/convolution.h>
#include <rapidht/minkowski.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace RapiDHT;

namespace {

const double kPi = std::acos(-1.0);

struct Volume {
    size_t w, h, d;
    std::vector<uint8_t> v;
    Volume(size_t w_, size_t h_, size_t d_):
        w(w_), h(h_), d(d_), v(w_ * h_ * d_, 0) { }
    uint8_t& at(size_t i, size_t j, size_t k) { return v[i + w * (j + h * k)]; }
    uint8_t at(size_t i, size_t j, size_t k) const { return v[i + w * (j + h * k)]; }
};

/// Voxels whose centres (i + 0.5 - n/2 ...) satisfy a predicate of the centred coordinates.
template <typename F>
Volume Shape(size_t n, F inside)
{
    Volume vol(n, n, n);
    const double c = n / 2.0;
    for (size_t k = 0; k < n; ++k) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < n; ++i) {
                vol.at(i, j, k) = inside(i + 0.5 - c, j + 0.5 - c, k + 0.5 - c) ? 1 : 0;
            }
        }
    }
    return vol;
}

MinkowskiValues Global(const Volume& vol, MinkowskiBoundary b)
{
    return MinkowskiFunctionals(vol.v.data(), vol.w, vol.h, vol.d, b);
}

/*
 * Reference 1: the thesis (Volkov, 2025), table 1 (configuration indices by symmetry
 * class IC) and table 2 (weights 8 M0, 24 M2, 24 M3, 8 M4 per class). Transcribed as
 * printed, with one correction: index 8, a single voxel, is missing from class 1 in
 * table 1 and belongs there.
 */
struct ThesisClass {
    std::vector<unsigned> members;
    std::array<int, 4> weights;
};

const std::vector<ThesisClass>& ThesisTables()
{
    static const std::vector<ThesisClass> t = {
        { { 0 }, { 0, 0, 0, 0 } },
        { { 1, 2, 4, 8 /* missing in the thesis */, 16, 32, 64, 128 }, { 1, 3, 3, 1 } },
        { { 3, 5, 10, 12, 17, 34, 48, 68, 80, 136, 160, 192 }, { 2, 4, 2, 0 } },
        { { 6, 9, 18, 20, 33, 40, 65, 72, 96, 130, 132, 144 }, { 2, 6, 2, -2 } },
        { { 24, 36, 66, 129 }, { 2, 6, 6, -6 } },
        { { 7, 11, 13, 14, 19, 21, 35, 42, 49, 50, 69, 76, 81, 84, 112, 138, 140, 162, 168, 176, 196, 200, 208, 224 },
            { 3, 5, 1, -1 } },
        { { 25, 26, 28, 37, 38, 44, 52, 56, 67, 70, 74, 82, 88, 98, 100, 131, 133, 137, 145, 152, 161, 164, 193, 194 },
            { 3, 7, 1, -3 } },
        { { 22, 41, 73, 97, 104, 134, 146, 148 }, { 3, 9, -3, -1 } },
        { { 15, 51, 85, 170, 204, 240 }, { 4, 4, 0, 0 } },
        { { 23, 43, 77, 113, 142, 178, 212, 232 }, { 4, 6, 0, -2 } },
        { { 27, 29, 39, 46, 53, 58, 71, 78, 83, 92, 114, 116, 139, 141, 163, 172, 177, 184, 197, 202, 209, 216, 226,
              228 },
            { 4, 6, 0, -2 } },
        { { 30, 45, 54, 57, 75, 86, 89, 99, 101, 106, 108, 120, 135, 147, 149, 154, 156, 166, 169, 180, 198, 201, 210,
              225 },
            { 4, 8, -4, 0 } },
        { { 60, 90, 102, 153, 165, 195 }, { 4, 8, -4, 0 } },
        { { 105, 150 }, { 4, 12, -12, 4 } },
        { { 107, 109, 121, 151, 158, 182, 214, 233 }, { 5, 9, -9, 3 } },
        { { 61, 62, 91, 94, 103, 110, 118, 122, 124, 155, 157, 167, 173, 181, 185, 188, 199, 203, 211, 217, 218, 227,
              229, 230 },
            { 5, 7, -5, 1 } },
        { { 31, 47, 55, 59, 79, 87, 93, 115, 117, 143, 171, 174, 179, 186, 205, 206, 213, 220, 234, 236, 241, 242, 244,
              248 },
            { 5, 5, -1, -1 } },
        { { 126, 189, 219, 231 }, { 6, 6, -6, 2 } },
        { { 111, 123, 125, 159, 183, 190, 215, 222, 235, 237, 246, 249 }, { 6, 6, -6, 2 } },
        { { 63, 95, 119, 175, 187, 207, 221, 238, 243, 245, 250, 252 }, { 6, 4, -2, 0 } },
        { { 127, 191, 223, 239, 247, 251, 253, 254 }, { 7, 3, -3, 1 } },
        { { 255 }, { 8, 0, 0, 0 } },
    };
    return t;
}

/*
 * Reference 2: count the cells of the union of closed cubes directly, in global
 * coordinates (doubled, so a cell is identified by its centre), and apply
 * Michielsen & De Raedt's formulas. Nothing is shared with the library's per-vertex
 * bookkeeping. Background outside the volume.
 */
MinkowskiValues CellCount(const Volume& vol)
{
    using Cell = std::tuple<long, long, long>;
    std::set<Cell> cells[4]; // by dimension
    for (size_t k = 0; k < vol.d; ++k) {
        for (size_t j = 0; j < vol.h; ++j) {
            for (size_t i = 0; i < vol.w; ++i) {
                if (!vol.at(i, j, k)) {
                    continue;
                }
                // Every cell of the cube [i,i+1]x[j,j+1]x[k,k+1]: each doubled coordinate is
                // 2i, 2i+1 (open interval) or 2i+2; the dimension is the count of odd ones.
                for (long a = 0; a < 3; ++a) {
                    for (long b = 0; b < 3; ++b) {
                        for (long c = 0; c < 3; ++c) {
                            const int dim = (a == 1) + (b == 1) + (c == 1);
                            cells[dim].insert({ 2 * static_cast<long>(i) + a, 2 * static_cast<long>(j) + b,
                                2 * static_cast<long>(k) + c });
                        }
                    }
                }
            }
        }
    }
    const double nv = static_cast<double>(cells[0].size());
    const double ne = static_cast<double>(cells[1].size());
    const double nf = static_cast<double>(cells[2].size());
    const double nc = static_cast<double>(cells[3].size());
    return { nc, -6 * nc + 2 * nf, kPi * (3 * nc - 2 * nf + ne), -nc + nf - ne + nv };
}

/// Apply one of the 48 symmetries of the cube (axis permutation, then flips) to a configuration.
unsigned Transform(unsigned config, const std::array<int, 3>& perm, const std::array<int, 3>& flip)
{
    unsigned out = 0;
    for (int bit = 0; bit < 8; ++bit) {
        if (!((config >> bit) & 1u)) {
            continue;
        }
        const int p[3] = { bit & 1, (bit >> 1) & 1, (bit >> 2) & 1 };
        int q[3];
        for (int a = 0; a < 3; ++a) {
            q[a] = p[perm[a]] ^ flip[a];
        }
        out |= 1u << (q[0] | (q[1] << 1) | (q[2] << 2));
    }
    return out;
}

void ExpectNear(const MinkowskiValues& a, const MinkowskiValues& b, double tol)
{
    EXPECT_NEAR(a.volume, b.volume, tol);
    EXPECT_NEAR(a.surface, b.surface, tol);
    EXPECT_NEAR(a.meanCurvature, b.meanCurvature, tol);
    EXPECT_NEAR(a.euler, b.euler, tol);
}

} // namespace

TEST(MinkowskiWeights, ReproduceThesisTables)
{
    std::set<unsigned> seen;
    for (size_t ic = 0; ic < ThesisTables().size(); ++ic) {
        const auto& cls = ThesisTables()[ic];
        for (unsigned n : cls.members) {
            EXPECT_TRUE(seen.insert(n).second) << "index " << n << " listed twice";
            const MinkowskiValues w = MinkowskiConfigurationWeights(n);
            EXPECT_NEAR(8 * w.volume, cls.weights[0], 1e-12) << "IC " << ic << ", index " << n;
            EXPECT_NEAR(4 * w.surface, cls.weights[1], 1e-12) << "IC " << ic << ", index " << n;
            EXPECT_NEAR(8 * w.meanCurvature / kPi, cls.weights[2], 1e-12) << "IC " << ic << ", index " << n;
            EXPECT_NEAR(8 * w.euler, cls.weights[3], 1e-12) << "IC " << ic << ", index " << n;
        }
    }
    EXPECT_EQ(seen.size(), 256u);
}

TEST(MinkowskiWeights, ThesisClassesAreTheOrbitsOfTheFullCubeGroup)
{
    // 22 classes need reflections as well as rotations: rotations alone give 23
    // (one chiral pair). The thesis says "up to rotations"; the classes are right.
    std::array<int, 3> perm = { 0, 1, 2 };
    std::vector<std::pair<std::array<int, 3>, std::array<int, 3>>> group;
    do {
        for (int f = 0; f < 8; ++f) {
            group.push_back({ perm, { f & 1, (f >> 1) & 1, (f >> 2) & 1 } });
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
    ASSERT_EQ(group.size(), 48u);

    for (const auto& cls : ThesisTables()) {
        std::set<unsigned> orbit;
        for (const auto& g : group) {
            orbit.insert(Transform(cls.members.front(), g.first, g.second));
        }
        EXPECT_EQ(orbit, std::set<unsigned>(cls.members.begin(), cls.members.end()));
    }
}

TEST(MinkowskiWeights, InvariantUnderCubeSymmetries)
{
    std::array<int, 3> perm = { 0, 1, 2 };
    do {
        for (int f = 0; f < 8; ++f) {
            const std::array<int, 3> flip = { f & 1, (f >> 1) & 1, (f >> 2) & 1 };
            for (unsigned n = 0; n < 256; ++n) {
                ExpectNear(MinkowskiConfigurationWeights(n), MinkowskiConfigurationWeights(Transform(n, perm, flip)),
                    1e-12);
            }
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
}

TEST(MinkowskiWeights, RejectsIndexAbove255)
{
    EXPECT_THROW(MinkowskiConfigurationWeights(256), std::out_of_range);
}

TEST(MinkowskiFunctionals, MatchCellCountOnRandomVolumes)
{
    std::mt19937 rng(7);
    for (double density : { 0.1, 0.3, 0.5, 0.7, 0.9 }) {
        Volume vol(7, 5, 6);
        std::bernoulli_distribution bit(density);
        for (auto& x : vol.v) {
            x = bit(rng) ? 1 : 0;
        }
        ExpectNear(Global(vol, MinkowskiBoundary::Empty), CellCount(vol), 1e-9);
    }
}

TEST(MinkowskiFunctionals, BoxIsExact)
{
    Volume vol(16, 12, 10);
    for (size_t k = 2; k < 6; ++k) {
        for (size_t j = 3; j < 10; ++j) {
            for (size_t i = 1; i < 11; ++i) {
                vol.at(i, j, k) = 1;
            }
        }
    }
    // 10 x 7 x 4: V = abc, S = 2(ab + bc + ca), M = pi (a + b + c), chi = 1.
    const MinkowskiValues expected { 280.0, 276.0, kPi * 21.0, 1.0 };
    ExpectNear(Global(vol, MinkowskiBoundary::Empty), expected, 1e-9);
    ExpectNear(Global(vol, MinkowskiBoundary::Periodic), expected, 1e-9); // does not touch the faces
}

TEST(MinkowskiFunctionals, EulerCharacteristicOfKnownTopologies)
{
    const size_t n = 40;
    auto r = [](double x, double y, double z) { return std::sqrt(x * x + y * y + z * z); };
    const Volume ball = Shape(n, [&](double x, double y, double z) { return r(x, y, z) < 12; });
    const Volume torus = Shape(n, [](double x, double y, double z) {
        const double rho = std::sqrt(x * x + y * y) - 10;
        return rho * rho + z * z < 16;
    });
    const Volume shell = Shape(n, [&](double x, double y, double z) { return r(x, y, z) < 15 && r(x, y, z) > 8; });
    const Volume twoBalls = Shape(n, [&](double x, double y, double z) { return r(x - 9, y, z) < 7 || r(x + 9, y, z) < 7; });
    // A solid cube pierced by three orthogonal tunnels that cross at the centre: the
    // level-1 Menger sponge cell, genus 5 (not 3: the crossing adds two loops), chi = -4.
    const Volume pierced = Shape(n, [](double x, double y, double z) {
        const bool cube = std::max({ std::fabs(x), std::fabs(y), std::fabs(z) }) < 12;
        const bool tunnel = std::max(std::fabs(x), std::fabs(y)) < 4 || std::max(std::fabs(y), std::fabs(z)) < 4
                         || std::max(std::fabs(x), std::fabs(z)) < 4;
        return cube && !tunnel;
    });

    for (auto b : { MinkowskiBoundary::Empty, MinkowskiBoundary::Periodic }) {
        EXPECT_NEAR(Global(ball, b).euler, 1.0, 1e-12);
        EXPECT_NEAR(Global(torus, b).euler, 0.0, 1e-12);
        EXPECT_NEAR(Global(shell, b).euler, 2.0, 1e-12); // a sphere bounds the cavity: 1 + 1
        EXPECT_NEAR(Global(twoBalls, b).euler, 2.0, 1e-12);
        EXPECT_NEAR(Global(pierced, b).euler, -4.0, 1e-12);
    }
}

TEST(MinkowskiFunctionals, ForegroundIs26Connected)
{
    Volume corner(4, 4, 4);
    corner.at(1, 1, 1) = corner.at(2, 2, 2) = 1; // touch at one point
    EXPECT_NEAR(Global(corner, MinkowskiBoundary::Empty).euler, 1.0, 1e-12);
    Volume edge(4, 4, 4);
    edge.at(1, 1, 1) = edge.at(2, 2, 1) = 1; // touch along an edge
    EXPECT_NEAR(Global(edge, MinkowskiBoundary::Empty).euler, 1.0, 1e-12);
    Volume apart(5, 4, 4);
    apart.at(1, 1, 1) = apart.at(3, 1, 1) = 1;
    EXPECT_NEAR(Global(apart, MinkowskiBoundary::Empty).euler, 2.0, 1e-12);
}

TEST(MinkowskiFunctionals, PeriodicWrapsAround)
{
    const size_t n = 12;
    Volume full(n, n, n);
    std::fill(full.v.begin(), full.v.end(), uint8_t { 1 });
    ExpectNear(Global(full, MinkowskiBoundary::Periodic), { double(n * n * n), 0.0, 0.0, 0.0 }, 1e-9);

    // A rod along z through the whole period is a solid torus (circle x disk): chi = 0,
    // and its surface is the lateral surface only.
    Volume rod(n, n, n);
    for (size_t k = 0; k < n; ++k) {
        for (size_t j = 4; j < 7; ++j) {
            for (size_t i = 3; i < 8; ++i) {
                rod.at(i, j, k) = 1;
            }
        }
    }
    const MinkowskiValues p = Global(rod, MinkowskiBoundary::Periodic);
    EXPECT_NEAR(p.volume, 5.0 * 3 * n, 1e-9);
    EXPECT_NEAR(p.surface, 2.0 * (5 + 3) * n, 1e-9);
    EXPECT_NEAR(p.meanCurvature, kPi * n, 1e-9); // 4 convex edges of length n: 4 * n * (pi/2) / 2
    EXPECT_NEAR(p.euler, 0.0, 1e-12);
    EXPECT_NEAR(Global(rod, MinkowskiBoundary::Empty).euler, 1.0, 1e-12); // cut open: a box
}

TEST(MinkowskiFunctionals, DigitalBallBias)
{
    // Documents the bias stated in the header: S and M about 3/2 of the smooth values.
    const double radius = 14;
    const Volume ball = Shape(40, [&](double x, double y, double z) { return x * x + y * y + z * z < radius * radius; });
    const MinkowskiValues m = Global(ball, MinkowskiBoundary::Empty);
    EXPECT_NEAR(m.volume / (4.0 / 3.0 * kPi * radius * radius * radius), 1.0, 0.02);
    EXPECT_NEAR(m.surface / (4 * kPi * radius * radius), 1.5, 0.05);
    EXPECT_NEAR(m.meanCurvature / (4 * kPi * radius), 1.5, 0.05);
}

TEST(MinkowskiFunctionals, RejectsBadInput)
{
    std::vector<uint8_t> v(8, 0);
    EXPECT_THROW(MinkowskiFunctionals(nullptr, 2, 2, 2), std::invalid_argument);
    EXPECT_THROW(MinkowskiFunctionals(v.data(), 8, 1, 1, MinkowskiBoundary::Periodic), std::invalid_argument);
    EXPECT_NO_THROW(MinkowskiFunctionals(v.data(), 8, 1, 1, MinkowskiBoundary::Empty));
    std::vector<float> out(4 * 8);
    EXPECT_THROW(MinkowskiDensities(v.data(), 2, 2, 2, static_cast<float*>(nullptr)), std::invalid_argument);
    EXPECT_THROW(MinkowskiDensities(v.data(), 8, 1, 1, out.data()), std::invalid_argument);
}

namespace {

Volume RandomVolume(size_t w, size_t h, size_t d, unsigned seed)
{
    std::mt19937 rng(seed);
    std::bernoulli_distribution bit(0.45);
    Volume vol(w, h, d);
    for (auto& x : vol.v) {
        x = bit(rng) ? 1 : 0;
    }
    return vol;
}

template <typename T>
std::array<double, 4> PlaneSums(const std::vector<T>& out, size_t plane)
{
    std::array<double, 4> s {};
    for (int p = 0; p < 4; ++p) {
        for (size_t n = 0; n < plane; ++n) {
            s[p] += out[p * plane + n];
        }
    }
    return s;
}

} // namespace

TEST(MinkowskiDensities, SumToTheGlobalFunctionals)
{
    const Volume vol = RandomVolume(9, 6, 7, 3);
    const size_t plane = vol.v.size();
    const MinkowskiValues g = Global(vol, MinkowskiBoundary::Periodic);
    for (auto c : { DensityCentring::Vertex, DensityCentring::Voxel }) {
        std::vector<double> d(4 * plane);
        MinkowskiDensities(vol.v.data(), vol.w, vol.h, vol.d, d.data(), c);
        auto s = PlaneSums(d, plane);
        EXPECT_NEAR(s[0], g.volume, 1e-9);
        EXPECT_NEAR(s[1], g.surface, 1e-9);
        EXPECT_NEAR(s[2], g.meanCurvature, 1e-9);
        EXPECT_NEAR(s[3], g.euler, 1e-9);

        std::vector<float> f(4 * plane);
        MinkowskiDensities(vol.v.data(), vol.w, vol.h, vol.d, f.data(), c);
        for (size_t n = 0; n < f.size(); ++n) {
            ASSERT_NEAR(f[n], d[n], 1e-6);
        }
    }
}

TEST(MinkowskiDensities, VertexLayoutHoldsConfigurationWeights)
{
    const Volume vol = RandomVolume(5, 4, 3, 11);
    const size_t plane = vol.v.size();
    std::vector<double> d(4 * plane);
    MinkowskiDensities(vol.v.data(), vol.w, vol.h, vol.d, d.data(), DensityCentring::Vertex);
    for (size_t k = 0; k < vol.d; ++k) {
        for (size_t j = 0; j < vol.h; ++j) {
            for (size_t i = 0; i < vol.w; ++i) {
                unsigned config = 0;
                for (unsigned bit = 0; bit < 8; ++bit) {
                    config |= unsigned(vol.at((i + (bit & 1)) % vol.w, (j + ((bit >> 1) & 1)) % vol.h,
                                  (k + ((bit >> 2) & 1)) % vol.d))
                           << bit;
                }
                const MinkowskiValues w = MinkowskiConfigurationWeights(config);
                const size_t n = i + vol.w * (j + vol.h * k);
                EXPECT_EQ(d[n], w.volume);
                EXPECT_EQ(d[n + plane], w.surface);
                EXPECT_EQ(d[n + 2 * plane], w.meanCurvature);
                EXPECT_EQ(d[n + 3 * plane], w.euler);
            }
        }
    }
}

TEST(MinkowskiDensities, VoxelCentringCommutesWithGridSymmetries)
{
    // Inversion (all three flips), one flip, and a transposition of two equal axes.
    const size_t n = 8;
    const Volume vol = RandomVolume(n, n, n, 5);
    const size_t plane = vol.v.size();
    using Map = std::array<size_t, 3> (*)(size_t, size_t, size_t);
    const Map maps[] = {
        [](size_t i, size_t j, size_t k) { return std::array<size_t, 3> { 7 - i, 7 - j, 7 - k }; },
        [](size_t i, size_t j, size_t k) { return std::array<size_t, 3> { i, 7 - j, k }; },
        [](size_t i, size_t j, size_t k) { return std::array<size_t, 3> { j, i, k }; },
    };
    std::vector<double> d(4 * plane), dt(4 * plane);
    MinkowskiDensities(vol.v.data(), n, n, n, d.data());
    for (Map m : maps) {
        Volume t(n, n, n);
        for (size_t k = 0; k < n; ++k) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    const auto q = m(i, j, k);
                    t.at(q[0], q[1], q[2]) = vol.at(i, j, k);
                }
            }
        }
        MinkowskiDensities(t.v.data(), n, n, n, dt.data());
        for (int p = 0; p < 4; ++p) {
            for (size_t k = 0; k < n; ++k) {
                for (size_t j = 0; j < n; ++j) {
                    for (size_t i = 0; i < n; ++i) {
                        const auto q = m(i, j, k);
                        ASSERT_NEAR(dt[p * plane + q[0] + n * (q[1] + n * q[2])], d[p * plane + i + n * (j + n * k)],
                            1e-12)
                            << "plane " << p;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Windows and local functionals (B-202)
// ---------------------------------------------------------------------------

namespace {

size_t WrappedIndex(long i, size_t n) { return static_cast<size_t>(((i % long(n)) + long(n)) % long(n)); }

/// Brute force: the voxel-centred densities summed over the voxels of a ball around each voxel.
std::vector<double> BallSumReference(const Volume& vol, double radius)
{
    const size_t plane = vol.v.size();
    std::vector<double> dens(4 * plane), out(4 * plane, 0.0);
    MinkowskiDensities(vol.v.data(), vol.w, vol.h, vol.d, dens.data(), DensityCentring::Voxel);
    const long R = static_cast<long>(std::ceil(radius));
    for (size_t k = 0; k < vol.d; ++k) {
        for (size_t j = 0; j < vol.h; ++j) {
            for (size_t i = 0; i < vol.w; ++i) {
                for (long c = -R; c <= R; ++c) {
                    for (long b = -R; b <= R; ++b) {
                        for (long a = -R; a <= R; ++a) {
                            if (double(a * a + b * b + c * c) > radius * radius) {
                                continue;
                            }
                            const size_t m = WrappedIndex(long(i) + a, vol.w)
                                           + vol.w * (WrappedIndex(long(j) + b, vol.h) + vol.h * WrappedIndex(long(k) + c, vol.d));
                            for (int p = 0; p < 4; ++p) {
                                out[p * plane + i + vol.w * (j + vol.h * k)] += dens[p * plane + m];
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

} // namespace

TEST(MinkowskiWindows, BallIsEvenAndHasTheRightVoxelCount)
{
    const auto w = MakeWindow<double>(16, 8, 8, WindowShape::Ball, 3.2);
    EXPECT_TRUE(IsEvenPerAxis(w.data(), 16, 8, 8));
    size_t count = 0;
    for (long c = -3; c <= 3; ++c) {
        for (long b = -3; b <= 3; ++b) {
            for (long a = -3; a <= 3; ++a) {
                count += a * a + b * b + c * c <= 3.2 * 3.2;
            }
        }
    }
    double sum = 0.0;
    for (double v : w) {
        sum += v;
    }
    EXPECT_EQ(sum, double(count));
}

TEST(MinkowskiWindows, RejectsBadRadius)
{
    EXPECT_THROW(MakeWindow<double>(16, 8, 8, WindowShape::Ball, 4.0), std::invalid_argument); // not < 8/2
    EXPECT_NO_THROW(MakeWindow<double>(16, 8, 8, WindowShape::Ball, 3.99));
    EXPECT_THROW(MakeWindow<double>(16, 8, 8, WindowShape::Ball, 0.0), std::invalid_argument);
    EXPECT_THROW(MakeWindow<double>(16, 8, 8, WindowShape::Gaussian, -1.0), std::invalid_argument);
    EXPECT_NO_THROW(MakeWindow<double>(16, 8, 8, WindowShape::Gaussian, 20.0)); // periodic images summed
}

TEST(MinkowskiWindows, GaussianIsThePeriodicSumOfImages)
{
    const double r = 5.0; // wider than a quarter period: the images matter
    const auto w = MakeWindow<double>(16, 8, 8, WindowShape::Gaussian, r);
    EXPECT_TRUE(IsEvenPerAxis(w.data(), 16, 8, 8, 1e-14));
    // Total weight of the periodised window = the integral-free lattice sum of one Gaussian,
    // which for r >> 1 is (sqrt(2 pi) r)^3 to many digits (Poisson summation).
    double sum = 0.0;
    for (double v : w) {
        sum += v;
    }
    EXPECT_NEAR(sum / std::pow(std::sqrt(2 * kPi) * r, 3), 1.0, 1e-12);
    // One value against a direct sum over images.
    double direct = 0.0;
    for (long a = -20; a <= 20; ++a) {
        for (long b = -20; b <= 20; ++b) {
            for (long c = -20; c <= 20; ++c) {
                const double x = 3 + 16.0 * a, y = 6 + 8.0 * b, z = 1 + 8.0 * c;
                direct += std::exp(-(x * x + y * y + z * z) / (2 * r * r));
            }
        }
    }
    EXPECT_NEAR(w[3 + 16 * (6 + 8 * 1)], direct, 1e-12);
}

TEST(MinkowskiFields, BallSumMatchesBruteForce)
{
    const Volume vol = RandomVolume(16, 8, 8, 21);
    const size_t plane = vol.v.size();
    const std::vector<double> radii = { 1.5, 2.5, 3.2 };
    MinkowskiFields<double> mf(16, 8, 8, WindowShape::Ball, radii, WindowNormalisation::Sum);
    ASSERT_EQ(mf.FieldCount(), 12u);
    std::vector<double> out(mf.FieldCount() * plane);
    mf.Compute(vol.v.data(), out.data());

    MinkowskiFields<float> mff(16, 8, 8, WindowShape::Ball, radii, WindowNormalisation::Sum);
    std::vector<float> outf(mff.FieldCount() * plane);
    mff.Compute(vol.v.data(), outf.data());

    for (size_t s = 0; s < radii.size(); ++s) {
        const auto ref = BallSumReference(vol, radii[s]);
        for (size_t f = 0; f < 4; ++f) {
            for (size_t n = 0; n < plane; ++n) {
                const size_t at = (f * radii.size() + s) * plane + n;
                ASSERT_NEAR(out[at], ref[f * plane + n], 1e-9) << "functional " << f << ", radius " << radii[s];
                ASSERT_NEAR(outf[at], ref[f * plane + n], 2e-3) << "float, functional " << f;
            }
        }
    }
}

TEST(MinkowskiFields, MeanOfFullVolumeIsUnitVolumeFraction)
{
    Volume full(8, 8, 8);
    std::fill(full.v.begin(), full.v.end(), uint8_t { 1 });
    for (auto shape : { WindowShape::Ball, WindowShape::Gaussian }) {
        MinkowskiFields<double> mf(8, 8, 8, shape, { 1.0, 2.5 });
        std::vector<double> out(mf.FieldCount() * mf.Size());
        mf.Compute(full.v.data(), out.data());
        for (size_t f = 0; f < 4; ++f) {
            for (size_t n = 0; n < 2 * mf.Size(); ++n) {
                ASSERT_NEAR(out[f * 2 * mf.Size() + n], f == 0 ? 1.0 : 0.0, 1e-12);
            }
        }
    }
}

TEST(MinkowskiFields, SpatialSumIsWindowWeightTimesGlobalFunctional)
{
    const Volume vol = RandomVolume(8, 16, 8, 4);
    const MinkowskiValues g = Global(vol, MinkowskiBoundary::Periodic);
    const double globals[4] = { g.volume, g.surface, g.meanCurvature, g.euler };
    MinkowskiFields<double> mf(8, 16, 8, WindowShape::Gaussian, { 1.0, 3.0, 9.0 }, WindowNormalisation::Sum);
    std::vector<double> out(mf.FieldCount() * mf.Size());
    mf.Compute(vol.v.data(), out.data());
    for (size_t f = 0; f < 4; ++f) {
        for (size_t s = 0; s < 3; ++s) {
            double sum = 0.0;
            for (size_t n = 0; n < mf.Size(); ++n) {
                sum += out[(f * 3 + s) * mf.Size() + n];
            }
            EXPECT_NEAR(sum, mf.WindowWeight(s) * globals[f], 1e-8 * mf.WindowWeight(s) * (1 + std::fabs(globals[f])));
        }
    }
}

TEST(MinkowskiFields, CommuteWithCentralInversion)
{
    const size_t n = 16;
    const Volume vol = RandomVolume(n, n, n, 9);
    Volume inv(n, n, n);
    for (size_t k = 0; k < n; ++k) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < n; ++i) {
                inv.at(n - 1 - i, n - 1 - j, n - 1 - k) = vol.at(i, j, k);
            }
        }
    }
    for (auto shape : { WindowShape::Ball, WindowShape::Gaussian }) {
        MinkowskiFields<double> mf(n, n, n, shape, { 2.0, 4.0 });
        std::vector<double> a(mf.FieldCount() * mf.Size()), b(a.size());
        mf.Compute(vol.v.data(), a.data());
        mf.Compute(inv.v.data(), b.data());
        for (size_t c = 0; c < mf.FieldCount(); ++c) {
            for (size_t k = 0; k < n; ++k) {
                for (size_t j = 0; j < n; ++j) {
                    for (size_t i = 0; i < n; ++i) {
                        const size_t p = c * mf.Size() + i + n * (j + n * k);
                        const size_t q = c * mf.Size() + (n - 1 - i) + n * ((n - 1 - j) + n * (n - 1 - k));
                        ASSERT_NEAR(b[q], a[p], 1e-10) << "field " << c;
                    }
                }
            }
        }
    }
}

TEST(MinkowskiFields, RejectsBadSetup)
{
    EXPECT_THROW(MinkowskiFields<double>(16, 16, 16, WindowShape::Ball, {}), std::invalid_argument);
    EXPECT_THROW(MinkowskiFields<double>(16, 16, 16, WindowShape::Ball, { 8.0 }), std::invalid_argument);
    EXPECT_THROW(MinkowskiFields<double>(12, 16, 16, WindowShape::Ball, { 2.0 }), std::invalid_argument); // not a power of two
}
