/*
 * Project: RapiDHT
 * File: python/rapidht_ext.cpp
 * Brief: nanobind bindings on numpy arrays: Minkowski functionals, densities, fields and
 *        the surface normal tensor; the Hartley transforms, the convolutions and the
 *        kernels in physical units (B-601, second part).
 *
 * Thin on purpose. Argument checking, dtype conversion and output allocation live in
 * python/rapidht/__init__.py; this file only maps arrays onto the C++ calls. A numpy
 * array of shape (n0, n1, n2) in C order is the C++ volume with width = n2,
 * height = n1, depth = n0 (the last axis is the fastest in both); (n0, n1) is
 * width = n1, height = n0, depth = 0, and (n0,) is width = n0. The extents arrive
 * here already in C++ order, (width, height, depth) with 0 for an absent axis.
 */

#include <rapidht/convolution.h>
#include <rapidht/filters.h>
#include <rapidht/minkowski.h>
#include <rapidht/transform.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace RapiDHT;

namespace {

using Volume = nb::ndarray<const uint8_t, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
template <typename T>
using Out = nb::ndarray<T, nb::c_contig, nb::device::cpu>;

struct Extent {
    size_t width, height, depth;
};

Extent ExtentOf(const Volume& v) { return { v.shape(2), v.shape(1), v.shape(0) }; }

template <typename T>
void CheckOut(const Out<T>& out, size_t expected, const char* caller)
{
    if (out.size() != expected) {
        throw std::invalid_argument(std::string(caller) + ": output has the wrong size.");
    }
}

EulerAdjacency Adjacency(bool kuhn) { return kuhn ? EulerAdjacency::Kuhn14 : EulerAdjacency::Cube26; }

nb::dict ToDict(const MinkowskiValues& m)
{
    nb::dict d;
    d["volume"] = m.volume;
    d["surface"] = m.surface;
    d["mean_curvature"] = m.meanCurvature;
    d["euler"] = m.euler;
    return d;
}

/// Any shape, C order, on the CPU: the arrays the transforms and convolutions take.
template <typename T>
using Array = nb::ndarray<T, nb::c_contig, nb::device::cpu>;
template <typename T>
using ConstArray = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;
template <typename T>
using NumpyArray = nb::ndarray<nb::numpy, T>;

size_t AtLeastOne(size_t n) { return n == 0 ? size_t { 1 } : n; }

template <typename A>
void CheckSize(const A& a, size_t expected, const char* caller)
{
    if (a.size() != expected) {
        throw std::invalid_argument(std::string(caller) + ": the array has " + std::to_string(a.size())
                                    + " elements, expected " + std::to_string(expected) + ".");
    }
}

/// A numpy array that owns `values`, in numpy order (depth, height, width), absent axes left out.
template <typename T>
NumpyArray<T> ToNumpy(std::vector<T>&& values, size_t width, size_t height, size_t depth)
{
    std::vector<size_t> shape;
    if (depth != 0) {
        shape.push_back(depth);
    }
    if (height != 0) {
        shape.push_back(height);
    }
    shape.push_back(width);
    auto* heap = new std::vector<T>(std::move(values));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return NumpyArray<T>(heap->data(), shape.size(), shape.data(), owner);
}

/// The transform and its element count, for the size checks.
template <typename T>
struct Transform {
    HartleyTransform<T> transform;
    size_t size;
    Transform(size_t w, size_t h, size_t d): transform(w, h, d, Modes::CPU), size(w * AtLeastOne(h) * AtLeastOne(d)) { }
};

/// A circular convolution: the separable path for a kernel even along every axis, the
/// true transform for any other.
template <typename T>
struct Circular {
    std::unique_ptr<EvenKernelConvolution<T>> even;
    std::unique_ptr<CircularConvolution<T>> general;
    size_t size;
};

template <typename T>
void BindTyped(nb::module_& m, const char* suffix)
{
    const std::string sfx(suffix);
    const auto release = nb::call_guard<nb::gil_scoped_release>();

    using Tr = Transform<T>;
    nb::class_<Tr>(m, ("_Transform_" + sfx).c_str())
        .def(nb::init<size_t, size_t, size_t>())
        .def(
            "forward",
            [](Tr& t, Array<T> a) {
                CheckSize(a, t.size, "dht");
                t.transform.ForwardTransform(a.data());
            },
            release)
        .def(
            "inverse",
            [](Tr& t, Array<T> a) {
                CheckSize(a, t.size, "idht");
                t.transform.InverseTransform(a.data());
            },
            release)
        .def(
            "separable",
            [](Tr& t, Array<T> a) {
                CheckSize(a, t.size, "separable_dht");
                t.transform.ForwardSeparable(a.data());
            },
            release);

    using Circ = Circular<T>;
    nb::class_<Circ>(m, ("_Circular_" + sfx).c_str())
        .def("__init__",
            [](Circ* self, size_t w, size_t h, size_t d, ConstArray<T> kernel) {
                // Everything that can throw before the placement new, which must not.
                Circ c;
                c.size = w * AtLeastOne(h) * AtLeastOne(d);
                CheckSize(kernel, c.size, "CircularConvolution");
                if (IsEvenPerAxis(kernel.data(), w, h, d, 0.0)) {
                    c.even = std::make_unique<EvenKernelConvolution<T>>(w, h, d, Modes::CPU);
                    c.even->SetKernel(kernel.data(), 0.0);
                } else {
                    c.general = std::make_unique<CircularConvolution<T>>(w, h, d, Modes::CPU);
                    c.general->SetKernel(kernel.data());
                }
                new (self) Circ(std::move(c));
            })
        .def_prop_ro("separable", [](const Circ& c) { return static_cast<bool>(c.even); })
        .def(
            "apply",
            [](Circ& c, Array<T> a) {
                CheckSize(a, c.size, "CircularConvolution");
                if (c.even) {
                    c.even->Apply(a.data());
                } else {
                    c.general->Apply(a.data());
                }
            },
            release);

    using Lin = LinearConvolution<T>;
    nb::class_<Lin>(m, ("_Linear_" + sfx).c_str())
        .def("__init__",
            [](Lin* self, size_t w, size_t h, size_t d, size_t kw, size_t kh, size_t kd, bool normalized,
                ConstArray<T> kernel) {
                Lin c(w, h, d, kw, kh, kd, normalized ? Edges::Normalized : Edges::Zero, Modes::CPU);
                CheckSize(kernel, c.KernelSize(), "LinearConvolution");
                c.SetKernel(kernel.data());
                new (self) Lin(std::move(c));
            })
        .def(
            "apply",
            [](Lin& c, ConstArray<T> a, Array<T> out) {
                CheckSize(a, c.Size(), "LinearConvolution");
                CheckSize(out, c.Size(), "LinearConvolution");
                c.Apply(a.data(), out.data());
            },
            release)
        .def(
            "apply_certainty",
            [](Lin& c, ConstArray<T> a, ConstArray<T> certainty, Array<T> out) {
                CheckSize(a, c.Size(), "LinearConvolution");
                CheckSize(certainty, c.Size(), "LinearConvolution");
                CheckSize(out, c.Size(), "LinearConvolution");
                c.Apply(a.data(), certainty.data(), out.data());
            },
            release)
        .def_prop_ro("padded", [](const Lin& c) {
            return std::array<size_t, 3> { c.PaddedWidth(), c.PaddedHeight(), c.PaddedDepth() };
        });

    m.def(("_gaussian_" + sfx).c_str(), [](int ndim, double sigma, double sx, double sy, double sz, double truncate) {
        CompactKernel<T> k = GaussianKernel<T>(ndim, sigma, Spacing { sx, sy, sz }, truncate);
        return ToNumpy(std::move(k.values), k.width, k.height, k.depth);
    });
    m.def(("_ball_" + sfx).c_str(), [](int ndim, double radius, double sx, double sy, double sz, bool normalise) {
        CompactKernel<T> k = BallKernel<T>(ndim, radius, Spacing { sx, sy, sz }, normalise);
        return ToNumpy(std::move(k.values), k.width, k.height, k.depth);
    });
    m.def(("_butterworth_" + sfx).c_str(),
        [](size_t w, size_t h, size_t d, double cutoff, int order, double sx, double sy, double sz) {
            return ToNumpy(ButterworthKernel<T>(w, h, d, cutoff, order, Spacing { sx, sy, sz }), w, h, d);
        });
    m.def(("_wrap_" + sfx).c_str(),
        [](ConstArray<T> kernel, size_t kw, size_t kh, size_t kd, size_t w, size_t h, size_t d) {
            CompactKernel<T> k { std::vector<T>(kernel.data(), kernel.data() + kernel.size()), kw, kh, kd };
            return ToNumpy(WrapKernel(k, w, h, d), w, h, d);
        });

    using Snt = SurfaceNormalTensor<T>;
    nb::class_<Snt>(m, ("_SurfaceTensor_" + sfx).c_str())
        .def("__init__",
            [](Snt* self, size_t w, size_t h, size_t d, double sigma, double sx, double sy, double sz) {
                new (self) Snt(w, h, d, sigma, Spacing { sx, sy, sz }, Modes::CPU);
            })
        .def(
            "compute",
            [](Snt& t, const Volume& v) {
                CheckSize(v, t.Size(), "SurfaceNormalTensor");
                const SurfaceTensor r = t.Compute(v.data());
                return std::array<double, 6> { r.xx, r.yy, r.zz, r.xy, r.xz, r.yz };
            },
            release)
        .def(
            "densities",
            [](Snt& t, const Volume& v, Out<T> out) {
                CheckSize(v, t.Size(), "SurfaceNormalTensor");
                CheckOut(out, 6 * t.Size(), "SurfaceNormalTensor");
                t.Densities(v.data(), out.data());
            },
            release);

    m.def((std::string("_densities_") + suffix).c_str(),
        [](const Volume& v, Out<T> out, bool voxel, bool kuhn) {
            const Extent e = ExtentOf(v);
            CheckOut(out, 4 * e.width * e.height * e.depth, "minkowski_densities");
            MinkowskiDensities<T>(v.data(), e.width, e.height, e.depth, out.data(),
                voxel ? DensityCentring::Voxel : DensityCentring::Vertex, Adjacency(kuhn));
        },
        nb::call_guard<nb::gil_scoped_release>());

    m.def((std::string("_window_") + suffix).c_str(),
        [](size_t n0, size_t n1, size_t n2, bool gaussian, double radius, Out<T> out) {
            CheckOut(out, n0 * n1 * n2, "make_window");
            const auto w
                = MakeWindow<T>(n2, n1, n0, gaussian ? WindowShape::Gaussian : WindowShape::Ball, radius);
            std::copy(w.begin(), w.end(), out.data());
        });

    using Fields = MinkowskiFields<T>;
    nb::class_<Fields>(m, (std::string("_MinkowskiFields_") + suffix).c_str())
        .def(
            "__init__",
            [](Fields* self, size_t n0, size_t n1, size_t n2, bool gaussian, const std::vector<double>& radii,
                bool mean, bool kuhn) {
                new (self) Fields(n2, n1, n0, gaussian ? WindowShape::Gaussian : WindowShape::Ball, radii,
                    mean ? WindowNormalisation::Mean : WindowNormalisation::Sum, Adjacency(kuhn));
            })
        .def(
            "compute",
            [](Fields& self, const Volume& v, Out<T> out) {
                const Extent e = ExtentOf(v);
                if (e.width * e.height * e.depth != self.Size()) {
                    throw std::invalid_argument("MinkowskiFields: the volume does not have the shape given at construction.");
                }
                CheckOut(out, self.FieldCount() * self.Size(), "MinkowskiFields");
                self.Compute(v.data(), out.data());
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def("window_weight", &Fields::WindowWeight)
        .def_prop_ro("scale_count", &Fields::ScaleCount);
}

} // namespace

NB_MODULE(_rapidht, m)
{
    m.doc() = "RapiDHT core bindings; use the rapidht package, not this module.";

    m.def("configuration_weights", [](unsigned n, bool kuhn) {
        const MinkowskiValues w = MinkowskiConfigurationWeights(n, Adjacency(kuhn));
        return std::vector<double> { w.volume, w.surface, w.meanCurvature, w.euler };
    });

    m.def(
        "minkowski_functionals",
        [](const Volume& v, bool periodic, bool kuhn) {
            const Extent e = ExtentOf(v);
            MinkowskiValues r;
            {
                nb::gil_scoped_release release;
                r = MinkowskiFunctionals(v.data(), e.width, e.height, e.depth,
                    periodic ? MinkowskiBoundary::Periodic : MinkowskiBoundary::Empty, Adjacency(kuhn));
            }
            return ToDict(r);
        });

    BindTyped<float>(m, "f32");
    BindTyped<double>(m, "f64");
}
