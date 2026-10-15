/*
 * Project: RapiDHT
 * File: src/minkowski.cpp
 * Brief: Minkowski functionals and their densities from 2x2x2 configurations.
 */

#include <rapidht/minkowski.h>
#include <rapidht/utilities.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace RapiDHT {

namespace {

constexpr double kPi = 3.14159265358979323846;

/// Voxel offset of bit i of a configuration index.
constexpr int Offset(int bit, int axis) { return (bit >> axis) & 1; }

/// Bit of the voxel at offset (x, y, z).
constexpr int Bit(int x, int y, int z) { return x | (y << 1) | (z << 2); }

/*
 * The weights are derived, not tabulated: count the cells of the cubical complex
 * that touch the vertex, each weighted by the share of it that belongs to the vertex.
 * The thesis table they reproduce lives in the tests as independent data.
 */
MinkowskiValues Derive(unsigned configuration)
{
    auto set = [configuration](int x, int y, int z) { return (configuration >> Bit(x, y, z)) & 1u; };

    int cubes = 0;
    for (int bit = 0; bit < 8; ++bit) {
        cubes += static_cast<int>((configuration >> bit) & 1u);
    }
    int faces = 0; // quarter faces at the vertex: present if either voxel beside it is
    int edges = 0; // half edges at the vertex: present if any of the 4 voxels around it is
    for (int axis = 0; axis < 3; ++axis) {
        const int u = (axis + 1) % 3;
        const int v = (axis + 2) % 3;
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                int lo[3] = { 0, 0, 0 };
                lo[u] = a;
                lo[v] = b;
                int hi[3] = { lo[0], lo[1], lo[2] };
                hi[axis] = 1;
                faces += static_cast<int>(set(lo[0], lo[1], lo[2]) | set(hi[0], hi[1], hi[2]));
            }
        }
        for (int side = 0; side < 2; ++side) {
            unsigned any = 0;
            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    int p[3] = { 0, 0, 0 };
                    p[axis] = side;
                    p[u] = a;
                    p[v] = b;
                    any |= set(p[0], p[1], p[2]);
                }
            }
            edges += static_cast<int>(any);
        }
    }
    const double c = cubes / 8.0;
    const double f = faces / 4.0;
    const double e = edges / 2.0;
    const double vertex = configuration != 0 ? 1.0 : 0.0;
    return { c, -6.0 * c + 2.0 * f, kPi * (3.0 * c - 2.0 * f + e), -c + f - e + vertex };
}

const std::array<MinkowskiValues, 256>& Table()
{
    static const std::array<MinkowskiValues, 256> table = [] {
        std::array<MinkowskiValues, 256> t {};
        for (unsigned n = 0; n < 256; ++n) {
            t[n] = Derive(n);
        }
        return t;
    }();
    return table;
}

void CheckExtents(const uint8_t* volume, size_t width, size_t height, size_t depth, size_t minimum,
    const char* caller)
{
    if (volume == nullptr) {
        throw std::invalid_argument(std::string(caller) + ": the pointer to volume is null.");
    }
    if (width < minimum || height < minimum || depth < minimum) {
        throw std::invalid_argument(std::string(caller) + ": every extent must be at least "
                                    + std::to_string(minimum) + " (this is a 3D routine).");
    }
}

/// Configuration index at vertex (i,j,k) of the periodic lattice: voxels i..i+1 etc., wrapped.
unsigned PeriodicConfiguration(const uint8_t* vol, size_t W, size_t H, size_t D, size_t i, size_t j, size_t k)
{
    const size_t xs[2] = { i, i + 1 == W ? 0 : i + 1 };
    const size_t ys[2] = { j, j + 1 == H ? 0 : j + 1 };
    const size_t zs[2] = { k, k + 1 == D ? 0 : k + 1 };
    unsigned index = 0;
    for (int bit = 0; bit < 8; ++bit) {
        const size_t n = xs[Offset(bit, 0)] + W * (ys[Offset(bit, 1)] + H * zs[Offset(bit, 2)]);
        index |= static_cast<unsigned>(vol[n] != 0) << bit;
    }
    return index;
}

} // namespace

MinkowskiValues MinkowskiConfigurationWeights(unsigned configuration)
{
    if (configuration > 255) {
        throw std::out_of_range("MinkowskiConfigurationWeights: the configuration index must be 0..255.");
    }
    return Table()[configuration];
}

MinkowskiValues MinkowskiFunctionals(const uint8_t* volume, size_t width, size_t height, size_t depth,
    MinkowskiBoundary boundary)
{
    PROFILE_FUNCTION();

    const auto& table = Table();
    const size_t W = width;
    const size_t H = height;
    const size_t D = depth;

    // Integer counts per configuration, then one weighted sum: exact and order-independent.
    std::array<long long, 256> histogram {};
    if (boundary == MinkowskiBoundary::Periodic) {
        CheckExtents(volume, W, H, D, 2, "MinkowskiFunctionals");
        for (size_t k = 0; k < D; ++k) {
            for (size_t j = 0; j < H; ++j) {
                for (size_t i = 0; i < W; ++i) {
                    ++histogram[PeriodicConfiguration(volume, W, H, D, i, j, k)];
                }
            }
        }
    } else {
        CheckExtents(volume, W, H, D, 1, "MinkowskiFunctionals");
        // Vertices -1..N-1 along each axis; voxels outside the volume are background.
        auto at = [&](long long i, long long j, long long k) -> unsigned {
            if (i < 0 || j < 0 || k < 0 || i >= static_cast<long long>(W) || j >= static_cast<long long>(H)
                || k >= static_cast<long long>(D)) {
                return 0;
            }
            return volume[static_cast<size_t>(i) + W * (static_cast<size_t>(j) + H * static_cast<size_t>(k))] != 0;
        };
        for (long long k = -1; k < static_cast<long long>(D); ++k) {
            for (long long j = -1; j < static_cast<long long>(H); ++j) {
                for (long long i = -1; i < static_cast<long long>(W); ++i) {
                    unsigned index = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        index |= at(i + Offset(bit, 0), j + Offset(bit, 1), k + Offset(bit, 2)) << bit;
                    }
                    ++histogram[index];
                }
            }
        }
    }

    MinkowskiValues sum;
    for (unsigned n = 0; n < 256; ++n) {
        const double count = static_cast<double>(histogram[n]);
        sum.volume += count * table[n].volume;
        sum.surface += count * table[n].surface;
        sum.meanCurvature += count * table[n].meanCurvature;
        sum.euler += count * table[n].euler;
    }
    return sum;
}

template <typename T>
void MinkowskiDensities(const uint8_t* volume, size_t width, size_t height, size_t depth, T* out,
    DensityCentring centring)
{
    PROFILE_FUNCTION();

    CheckExtents(volume, width, height, depth, 2, "MinkowskiDensities");
    if (out == nullptr) {
        throw std::invalid_argument("MinkowskiDensities: the pointer to out is null.");
    }
    const auto& table = Table();
    const size_t W = width;
    const size_t H = height;
    const size_t D = depth;
    const size_t plane = W * H * D;
    const long long depthCount = static_cast<long long>(D);

    if (centring == DensityCentring::Vertex) {
#pragma omp parallel for
        for (long long kk = 0; kk < depthCount; ++kk) {
            const size_t k = static_cast<size_t>(kk);
            for (size_t j = 0; j < H; ++j) {
                for (size_t i = 0; i < W; ++i) {
                    const MinkowskiValues& w = table[PeriodicConfiguration(volume, W, H, D, i, j, k)];
                    const size_t n = i + W * (j + H * k);
                    out[n] = static_cast<T>(w.volume);
                    out[n + plane] = static_cast<T>(w.surface);
                    out[n + 2 * plane] = static_cast<T>(w.meanCurvature);
                    out[n + 3 * plane] = static_cast<T>(w.euler);
                }
            }
        }
        return;
    }

    // Voxel (i,j,k) has corners at vertices i-1..i, j-1..j, k-1..k of the Vertex layout.
    // Sum the 8 configuration weights in double, then divide once.
#pragma omp parallel for
    for (long long kk = 0; kk < depthCount; ++kk) {
        const size_t k = static_cast<size_t>(kk);
        const size_t ks[2] = { k == 0 ? D - 1 : k - 1, k };
        for (size_t j = 0; j < H; ++j) {
            const size_t js[2] = { j == 0 ? H - 1 : j - 1, j };
            for (size_t i = 0; i < W; ++i) {
                const size_t is[2] = { i == 0 ? W - 1 : i - 1, i };
                MinkowskiValues s;
                for (int corner = 0; corner < 8; ++corner) {
                    const MinkowskiValues& w = table[PeriodicConfiguration(volume, W, H, D, is[Offset(corner, 0)],
                        js[Offset(corner, 1)], ks[Offset(corner, 2)])];
                    s.volume += w.volume;
                    s.surface += w.surface;
                    s.meanCurvature += w.meanCurvature;
                    s.euler += w.euler;
                }
                const size_t n = i + W * (j + H * k);
                out[n] = static_cast<T>(s.volume / 8.0);
                out[n + plane] = static_cast<T>(s.surface / 8.0);
                out[n + 2 * plane] = static_cast<T>(s.meanCurvature / 8.0);
                out[n + 3 * plane] = static_cast<T>(s.euler / 8.0);
            }
        }
    }
}

template void MinkowskiDensities<float>(const uint8_t*, size_t, size_t, size_t, float*, DensityCentring);
template void MinkowskiDensities<double>(const uint8_t*, size_t, size_t, size_t, double*, DensityCentring);

namespace {

size_t AtLeastOne(size_t n) { return n == 0 ? size_t { 1 } : n; }

/// Wrapped displacement of index i on a period of n: min(i, n - i).
double Wrapped(size_t i, size_t n) { return static_cast<double>(std::min(i, n - i)); }

/// exp(-x^2 / (2 r^2)) summed over the periodic images x + m n, as a function of the index.
/// Computed for i <= n/2 and mirrored, so the table is exactly even.
std::vector<double> PeriodicGaussian(size_t n, double r)
{
    std::vector<double> g(n, 0.0);
    const long images = static_cast<long>(std::ceil(12.0 * r / static_cast<double>(n))) + 1;
    for (size_t i = 0; i <= n / 2; ++i) {
        double sum = 0.0;
        for (long m = -images; m <= images; ++m) {
            const double x = static_cast<double>(i) + static_cast<double>(m) * static_cast<double>(n);
            sum += std::exp(-x * x / (2.0 * r * r));
        }
        g[i] = sum;
        g[(n - i) % n] = sum;
    }
    return g;
}

} // namespace

template <typename T>
std::vector<T> MakeWindow(size_t width, size_t height, size_t depth, WindowShape shape, double radius)
{
    if (!(radius > 0.0)) {
        throw std::invalid_argument("MakeWindow: the radius must be positive.");
    }
    const size_t W = width;
    const size_t H = AtLeastOne(height);
    const size_t D = AtLeastOne(depth);
    std::vector<T> window(W * H * D);

    if (shape == WindowShape::Ball) {
        for (size_t n : { W, H, D }) {
            if (n > 1 && !(radius < static_cast<double>(n) / 2.0)) {
                throw std::invalid_argument("MakeWindow: a Ball window needs a radius below half of every extent; "
                                            "use a Gaussian window or a larger volume.");
            }
        }
        for (size_t k = 0; k < D; ++k) {
            const double z = Wrapped(k, D);
            for (size_t j = 0; j < H; ++j) {
                const double y = Wrapped(j, H);
                for (size_t i = 0; i < W; ++i) {
                    const double x = Wrapped(i, W);
                    window[i + W * (j + H * k)] = x * x + y * y + z * z <= radius * radius ? T(1) : T(0);
                }
            }
        }
        return window;
    }

    const auto gx = PeriodicGaussian(W, radius);
    const auto gy = PeriodicGaussian(H, radius);
    const auto gz = PeriodicGaussian(D, radius);
    for (size_t k = 0; k < D; ++k) {
        for (size_t j = 0; j < H; ++j) {
            for (size_t i = 0; i < W; ++i) {
                window[i + W * (j + H * k)] = static_cast<T>(gx[i] * gy[j] * gz[k]);
            }
        }
    }
    return window;
}

template <typename T>
MinkowskiFields<T>::MinkowskiFields(size_t width, size_t height, size_t depth, WindowShape shape,
    const std::vector<double>& radii, WindowNormalisation normalisation):
    _width(width),
    _height(height),
    _depth(depth),
    _size(width * height * depth),
    _normalisation(normalisation),
    _bank(width, height, depth, Modes::CPU)
{
    if (radii.empty()) {
        throw std::invalid_argument("MinkowskiFields: give at least one window radius.");
    }
    for (double r : radii) {
        const std::vector<T> window = MakeWindow<T>(width, height, depth, shape, r);
        double weight = 0.0;
        for (T v : window) {
            weight += static_cast<double>(v);
        }
        _bank.AddKernel(window.data());
        _windowWeights.push_back(weight);
    }
}

template <typename T>
void MinkowskiFields<T>::Compute(const uint8_t* volume, T* out)
{
    PROFILE_FUNCTION();

    if (out == nullptr) {
        throw std::invalid_argument("MinkowskiFields::Compute: the pointer to out is null.");
    }
    _densities.resize(4 * _size);
    MinkowskiDensities(volume, _width, _height, _depth, _densities.data(), DensityCentring::Voxel);

    const size_t K = ScaleCount();
    for (size_t f = 0; f < 4; ++f) {
        T* block = out + f * K * _size;
        _bank.Apply(_densities.data() + f * _size, block);
        if (_normalisation == WindowNormalisation::Mean) {
            for (size_t s = 0; s < K; ++s) {
                const T scale = static_cast<T>(1.0 / _windowWeights[s]);
                T* field = block + s * _size;
                for (size_t n = 0; n < _size; ++n) {
                    field[n] *= scale;
                }
            }
        }
    }
}

template std::vector<float> MakeWindow<float>(size_t, size_t, size_t, WindowShape, double);
template std::vector<double> MakeWindow<double>(size_t, size_t, size_t, WindowShape, double);
template class MinkowskiFields<float>;
template class MinkowskiFields<double>;

} // namespace RapiDHT
