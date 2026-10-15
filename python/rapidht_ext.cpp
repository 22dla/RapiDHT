/*
 * Project: RapiDHT
 * File: python/rapidht_ext.cpp
 * Brief: nanobind bindings: Minkowski functionals, densities and fields on numpy arrays.
 *
 * Thin on purpose. Argument checking, dtype conversion and output allocation live in
 * python/rapidht/__init__.py; this file only maps arrays onto the C++ calls. A numpy
 * array of shape (n0, n1, n2) in C order is the C++ volume with width = n2,
 * height = n1, depth = n0 (the last axis is the fastest in both).
 */

#include <rapidht/minkowski.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
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

nb::dict ToDict(const MinkowskiValues& m)
{
    nb::dict d;
    d["volume"] = m.volume;
    d["surface"] = m.surface;
    d["mean_curvature"] = m.meanCurvature;
    d["euler"] = m.euler;
    return d;
}

template <typename T>
void BindTyped(nb::module_& m, const char* suffix)
{
    m.def((std::string("_densities_") + suffix).c_str(),
        [](const Volume& v, Out<T> out, bool voxel) {
            const Extent e = ExtentOf(v);
            CheckOut(out, 4 * e.width * e.height * e.depth, "minkowski_densities");
            MinkowskiDensities<T>(v.data(), e.width, e.height, e.depth, out.data(),
                voxel ? DensityCentring::Voxel : DensityCentring::Vertex);
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
                bool mean) {
                new (self) Fields(n2, n1, n0, gaussian ? WindowShape::Gaussian : WindowShape::Ball, radii,
                    mean ? WindowNormalisation::Mean : WindowNormalisation::Sum);
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

    m.def("configuration_weights", [](unsigned n) {
        const MinkowskiValues w = MinkowskiConfigurationWeights(n);
        return std::vector<double> { w.volume, w.surface, w.meanCurvature, w.euler };
    });

    m.def(
        "minkowski_functionals",
        [](const Volume& v, bool periodic) {
            const Extent e = ExtentOf(v);
            MinkowskiValues r;
            {
                nb::gil_scoped_release release;
                r = MinkowskiFunctionals(v.data(), e.width, e.height, e.depth,
                    periodic ? MinkowskiBoundary::Periodic : MinkowskiBoundary::Empty);
            }
            return ToDict(r);
        });

    BindTyped<float>(m, "f32");
    BindTyped<double>(m, "f64");
}
