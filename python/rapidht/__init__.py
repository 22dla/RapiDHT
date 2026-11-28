"""RapiDHT from Python (CPU): the multidimensional Hartley transform, convolutions through
it, kernels in physical units, and Minkowski functionals, fields and the surface normal
tensor of 3D binary volumes.

    import numpy as np, rapidht
    x = np.random.rand(64, 32, 16).astype(np.float32)
    h = rapidht.dht(x)                              # true 3D DHT, unnormalised; idht(h) == x
    g = rapidht.gaussian_kernel(3, sigma=2.0, spacing=(2.5, 1.0, 1.0))   # thick slices
    smooth = rapidht.LinearConvolution(x.shape, g, edges="normalized")(x)
    solid = np.random.rand(64, 64, 64) < 0.4
    rapidht.minkowski_functionals(solid)            # {'volume': ..., 'surface': ..., ...}
    mf = rapidht.MinkowskiFields(solid.shape, radii=(2, 4, 8))
    fields = mf(solid)                              # (4, 3, 64, 64, 64): functional, scale, volume
    rapidht.surface_normal_tensor(solid, sigma=1.5) # 3 x 3, integral of n n dS

Axes and spacings are in numpy order: shape (n0, n1, n2) is depth, height, width, and a
spacing is (s0, s1, s2) for those axes. Results are numpy arrays; torch.from_numpy (and
np.asarray of a CPU tensor) share their memory, so nothing is copied on the way to or
from PyTorch.

Conventions (see include/rapidht/minkowski.h for the derivations):

- Foreground = nonzero voxels = closed unit cubes, so the foreground is 26-connected and
  the background 6-connected. For a pore space checked for percolation with 6-connectivity,
  pass the solid.
- Functionals in voxel units, Steiner normalisation: volume, surface, mean curvature
  (integral of (1/r1 + 1/r2)/2), Euler characteristic. Surface and mean curvature of a
  digitised smooth surface come out about 3/2 of the smooth values.
- Volumes are periodic unless stated otherwise; the field and window routines need every
  extent to be a power of two.
- Densities and fields are voxel-centred by default, so they commute with np.flip,
  transpositions and the central inversion applied to the volume.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import _rapidht

__all__ = [
    "FUNCTIONALS",
    "CircularConvolution",
    "HartleyTransform",
    "LinearConvolution",
    "MinkowskiFields",
    "SurfaceNormalTensor",
    "ball_kernel",
    "butterworth_kernel",
    "configuration_weights",
    "dht",
    "gaussian_kernel",
    "idht",
    "make_window",
    "minkowski_densities",
    "minkowski_functionals",
    "separable_dht",
    "surface_normal_tensor",
    "wrap_kernel",
]

FUNCTIONALS = ("volume", "surface", "mean_curvature", "euler")

_TYPED = {np.dtype(np.float32): "f32", np.dtype(np.float64): "f64"}


def _volume(volume) -> np.ndarray:
    a = np.asarray(volume)
    if a.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {a.shape}")
    if a.dtype == np.bool_:
        a = a.view(np.uint8)
    elif a.dtype != np.uint8:
        a = a != 0
        a = a.view(np.uint8)
    return np.ascontiguousarray(a)


def _suffix(dtype) -> str:
    try:
        return _TYPED[np.dtype(dtype)]
    except KeyError:
        raise ValueError(f"dtype must be float32 or float64, got {np.dtype(dtype)}") from None


def _choice(value: str, options: Sequence[str], name: str) -> str:
    if value not in options:
        raise ValueError(f"{name} must be one of {tuple(options)}, got {value!r}")
    return value


def configuration_weights() -> np.ndarray:
    """(256, 4): contribution of one lattice vertex per 2x2x2 configuration, columns FUNCTIONALS.

    Bit i of the index is the voxel at offset (i & 1, i >> 1 & 1, i >> 2 & 1) along the
    C++ (width, height, depth) axes, i.e. numpy axes (2, 1, 0).
    """
    return np.array([_rapidht.configuration_weights(n) for n in range(256)])


def minkowski_functionals(volume, boundary: str = "periodic") -> dict[str, float]:
    """Global functionals; boundary 'periodic' (per period cell) or 'empty' (background outside)."""
    _choice(boundary, ("periodic", "empty"), "boundary")
    return _rapidht.minkowski_functionals(_volume(volume), boundary == "periodic")


def minkowski_densities(volume, centring: str = "voxel", dtype=np.float32) -> np.ndarray:
    """(4, *shape) local densities; each plane sums to the periodic global functional.

    centring 'voxel': mean of the 8 corner contributions of each voxel (symmetric);
    'vertex': the contribution of the corner at index + 1/2 (raw, half a voxel off).
    """
    _choice(centring, ("voxel", "vertex"), "centring")
    v = _volume(volume)
    out = np.empty((4, *v.shape), dtype=dtype)
    getattr(_rapidht, "_densities_" + _suffix(dtype))(v, out, centring == "voxel")
    return out


def make_window(shape: Sequence[int], window: str, radius: float, dtype=np.float64) -> np.ndarray:
    """A window in the wrapped layout (centre at index 0), even along every axis.

    'ball': 1 at voxel centres within radius (radius < half of every extent);
    'gaussian': exp(-|x|^2 / 2 r^2) summed over the periodic images (any radius).
    """
    _choice(window, ("ball", "gaussian"), "window")
    n0, n1, n2 = (int(s) for s in shape)
    out = np.empty((n0, n1, n2), dtype=dtype)
    getattr(_rapidht, "_window_" + _suffix(dtype))(n0, n1, n2, window == "gaussian", float(radius), out)
    return out


class MinkowskiFields:
    """Local Minkowski functionals over a bank of windows, periodic boundary.

    MinkowskiFields(shape, radii, window='ball'|'gaussian', normalisation='mean'|'sum',
    dtype=float32). Calling it on a volume of that shape returns (4, len(radii), *shape):
    functional (FUNCTIONALS order), window, voxel. 'sum' gives the functionals of the part
    of the body inside each window; 'mean' divides by the window weight (local volume
    fraction, surface per voxel, ...), comparable across window sizes.

    The transform is set up once per instance; reuse it for many volumes of one shape.
    """

    def __init__(
        self,
        shape: Sequence[int],
        radii: Sequence[float],
        window: str = "ball",
        normalisation: str = "mean",
        dtype=np.float32,
    ):
        _choice(window, ("ball", "gaussian"), "window")
        _choice(normalisation, ("mean", "sum"), "normalisation")
        self.shape = tuple(int(s) for s in shape)
        if len(self.shape) != 3:
            raise ValueError(f"expected a 3D shape, got {self.shape}")
        self.radii = tuple(float(r) for r in radii)
        self.window = window
        self.normalisation = normalisation
        self.dtype = np.dtype(dtype)
        cls = getattr(_rapidht, "_MinkowskiFields_" + _suffix(dtype))
        self._impl = cls(*self.shape, window == "gaussian", list(self.radii), normalisation == "mean")

    @property
    def window_weights(self) -> np.ndarray:
        """Total weight of each window (its voxel count, for a ball)."""
        return np.array([self._impl.window_weight(s) for s in range(len(self.radii))])

    def __call__(self, volume) -> np.ndarray:
        v = _volume(volume)
        if v.shape != self.shape:
            raise ValueError(f"volume shape {v.shape} differs from {self.shape} given at construction")
        out = np.empty((4, len(self.radii), *self.shape), dtype=self.dtype)
        self._impl.compute(v, out)
        return out

    def __repr__(self) -> str:
        return (
            f"MinkowskiFields(shape={self.shape}, radii={self.radii}, window={self.window!r}, "
            f"normalisation={self.normalisation!r}, dtype={self.dtype.name})"
        )


# ---------------------------------------------------------------------------------------
# The transform, the convolutions and the kernels (B-601, second part)
# ---------------------------------------------------------------------------------------


def _extent(shape: Sequence[int]) -> tuple[int, int, int]:
    """numpy shape -> C++ (width, height, depth), 0 for an absent axis."""
    s = tuple(int(n) for n in shape)
    if len(s) == 1:
        return s[0], 0, 0
    if len(s) == 2:
        return s[1], s[0], 0
    if len(s) == 3:
        return s[2], s[1], s[0]
    raise ValueError(f"expected a 1D, 2D or 3D shape, got {s}")


def _spacing(spacing, ndim: int) -> tuple[float, float, float]:
    """numpy-order spacing (one per axis, or one number for all) -> C++ (x, y, z)."""
    if spacing is None:
        return 1.0, 1.0, 1.0
    if np.isscalar(spacing):
        return (float(spacing),) * 3
    sp = tuple(float(v) for v in spacing)
    if len(sp) != ndim:
        raise ValueError(f"expected {ndim} spacings, one per axis, got {len(sp)}")
    sp = sp[::-1] + (1.0,) * (3 - ndim)
    return sp[0], sp[1], sp[2]


def _float_dtype(a, dtype):
    if dtype is not None:
        return np.dtype(dtype)
    a = np.asarray(a)
    return a.dtype if a.dtype in _TYPED else np.dtype(np.float64)


def _working_copy(a, dtype) -> np.ndarray:
    """A C-contiguous copy of a in dtype: the transforms work in place."""
    return np.array(a, dtype=dtype, order="C", copy=True)


class HartleyTransform:
    """The true multidimensional DHT of 1D, 2D or 3D arrays of one shape (powers of two).

    H(k) = sum_x a(x) cas(2 pi k.x / n), cas = cos + sin; unnormalised, the inverse applies
    1/N. `separable` is the 1D transform along every axis in turn, without the Bracewell
    pass: a different function in 2D and 3D, the one even-kernel convolution needs. Set up
    once per shape and reuse, like an FFT plan. Each call returns a new array.
    """

    def __init__(self, shape: Sequence[int], dtype=np.float64):
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype)
        self._impl = getattr(_rapidht, "_Transform_" + _suffix(self.dtype))(*_extent(self.shape))

    def _run(self, a, method: str) -> np.ndarray:
        if np.shape(a) != self.shape:
            raise ValueError(f"array shape {np.shape(a)} differs from {self.shape} given at construction")
        out = _working_copy(a, self.dtype)
        getattr(self._impl, method)(out)
        return out

    def forward(self, a) -> np.ndarray:
        return self._run(a, "forward")

    def inverse(self, a) -> np.ndarray:
        return self._run(a, "inverse")

    def separable(self, a) -> np.ndarray:
        return self._run(a, "separable")

    def __repr__(self) -> str:
        return f"HartleyTransform(shape={self.shape}, dtype={self.dtype.name})"


def dht(a, dtype=None) -> np.ndarray:
    """The true multidimensional DHT of a (1D, 2D or 3D, powers of two), unnormalised."""
    return HartleyTransform(np.shape(a), _float_dtype(a, dtype)).forward(a)


def idht(a, dtype=None) -> np.ndarray:
    """The inverse DHT: dht(a) / N."""
    return HartleyTransform(np.shape(a), _float_dtype(a, dtype)).inverse(a)


def separable_dht(a, dtype=None) -> np.ndarray:
    """The 1D DHT along every axis, no Bracewell pass; its own inverse up to N."""
    return HartleyTransform(np.shape(a), _float_dtype(a, dtype)).separable(a)


class CircularConvolution:
    """Periodic convolution of arrays of one shape (powers of two) with a fixed kernel.

    The kernel has the array's shape, in the wrapped layout: its centre at index 0 (see
    wrap_kernel). A kernel even along every axis takes the separable transform and a plain
    product (`separable` is then True), any other the true transform and the pair product.
    out(x) = sum_y a(y) kernel(x - y), indices modulo the shape.
    """

    def __init__(self, kernel, dtype=None):
        k = np.asarray(kernel)
        self.shape = k.shape
        self.dtype = _float_dtype(k, dtype)
        cls = getattr(_rapidht, "_Circular_" + _suffix(self.dtype))
        self._impl = cls(*_extent(self.shape), np.ascontiguousarray(k, dtype=self.dtype))

    @property
    def separable(self) -> bool:
        return self._impl.separable

    def __call__(self, a) -> np.ndarray:
        if np.shape(a) != self.shape:
            raise ValueError(f"array shape {np.shape(a)} differs from the kernel's {self.shape}")
        out = _working_copy(a, self.dtype)
        self._impl.apply(out)
        return out


class LinearConvolution:
    """Linear convolution of arrays of any shape with a compact kernel, by zero padding.

    LinearConvolution(shape, kernel, edges='zero'|'normalized', dtype). The kernel has odd
    extents with its centre in the middle (gaussian_kernel, ball_kernel) and the array's
    number of axes; out(x) = sum_t a(x - t) kernel(t), the output of the array's shape --
    scipy.ndimage.convolve(a, kernel, mode='constant'). 'normalized' divides by the kernel's
    weight inside the array (or on the certainty), so a constant stays constant up to the
    border; meant for non-negative smoothing kernels. Calling it with certainty=c (weights
    in [0, 1], 0 for voxels not to trust) gives ((c a) * k) / (c * k); where nothing
    trusted is in reach the output is 0.
    """

    def __init__(self, shape: Sequence[int], kernel, edges: str = "zero", dtype=None):
        _choice(edges, ("zero", "normalized"), "edges")
        self.shape = tuple(int(n) for n in shape)
        k = np.asarray(kernel)
        if k.ndim != len(self.shape):
            raise ValueError(f"the kernel has {k.ndim} axes, the arrays {len(self.shape)}")
        self.edges = edges
        self.dtype = _float_dtype(k, dtype)
        cls = getattr(_rapidht, "_Linear_" + _suffix(self.dtype))
        self._impl = cls(
            *_extent(self.shape), *_extent(k.shape), edges == "normalized", np.ascontiguousarray(k, dtype=self.dtype)
        )

    @property
    def padded_shape(self) -> tuple[int, ...]:
        """The power-of-two shape of the circular convolution underneath, numpy order."""
        w, h, d = self._impl.padded
        return tuple(n for n in (d, h, w) if n != 0)

    def __call__(self, a, certainty=None) -> np.ndarray:
        if np.shape(a) != self.shape:
            raise ValueError(f"array shape {np.shape(a)} differs from {self.shape} given at construction")
        data = np.ascontiguousarray(a, dtype=self.dtype)
        out = np.empty(self.shape, dtype=self.dtype)
        if certainty is None:
            self._impl.apply(data, out)
        else:
            if self.edges != "normalized":
                raise ValueError("a certainty needs edges='normalized'")
            if np.shape(certainty) != self.shape:
                raise ValueError(f"certainty shape {np.shape(certainty)} differs from {self.shape}")
            self._impl.apply_certainty(data, np.ascontiguousarray(certainty, dtype=self.dtype), out)
        return out


def gaussian_kernel(ndim: int, sigma: float, spacing=None, truncate: float = 4.0, dtype=np.float64) -> np.ndarray:
    """Compact sampled Gaussian exp(-|x|^2 / 2 sigma^2), x in the unit of `spacing`, sum 1.

    Reaches ceil(truncate sigma / spacing_a) voxels from the centre along each axis, so on
    a grid with thick slices it is thin across them. For LinearConvolution.
    """
    sx, sy, sz = _spacing(spacing, int(ndim))
    return getattr(_rapidht, "_gaussian_" + _suffix(dtype))(int(ndim), float(sigma), sx, sy, sz, float(truncate))


def ball_kernel(ndim: int, radius: float, spacing=None, normalise: bool = True, dtype=np.float64) -> np.ndarray:
    """Compact ball: the voxels whose centre is within radius (unit of `spacing`).

    normalise=True divides by their count (a moving average), False leaves the 0/1
    indicator (a moving sum).
    """
    sx, sy, sz = _spacing(spacing, int(ndim))
    return getattr(_rapidht, "_ball_" + _suffix(dtype))(int(ndim), float(radius), sx, sy, sz, bool(normalise))


def butterworth_kernel(shape: Sequence[int], cutoff: float, order: int, spacing=None, dtype=np.float64) -> np.ndarray:
    """Butterworth low-pass 1 / (1 + (|f| / cutoff)^(2 order)) as a wrapped kernel of `shape`.

    f in cycles per unit of `spacing`; the kernel's Hartley spectrum is exactly that
    response, so CircularConvolution applies it exactly. Shape: powers of two.
    """
    sx, sy, sz = _spacing(spacing, len(shape))
    fn = getattr(_rapidht, "_butterworth_" + _suffix(dtype))
    return fn(*_extent(shape), float(cutoff), int(order), sx, sy, sz)


def wrap_kernel(kernel, shape: Sequence[int]) -> np.ndarray:
    """A compact kernel (odd extents, centre in the middle) in the wrapped layout of `shape`.

    Tap t goes to index t mod n, taps that land together are summed: the layout
    CircularConvolution takes.
    """
    k = np.asarray(kernel)
    dtype = _float_dtype(k, None)
    if k.ndim != len(shape):
        raise ValueError(f"the kernel has {k.ndim} axes, the shape {len(shape)}")
    fn = getattr(_rapidht, "_wrap_" + _suffix(dtype))
    return fn(np.ascontiguousarray(k, dtype=dtype), *_extent(k.shape), *_extent(shape))


def _tensor_matrix(c) -> np.ndarray:
    """C++ (xx, yy, zz, xy, xz, yz), x the fastest axis -> 3 x 3 in numpy axis order."""
    xx, yy, zz, xy, xz, yz = c
    return np.array([[zz, yz, xz], [yz, yy, xy], [xz, xy, xx]])


class SurfaceNormalTensor:
    """The surface normal tensor, integral of n n dS, of 3D binary volumes of one shape.

    From the gradient of the body smoothed by a Gaussian of `sigma` (unit of `spacing`):
    exact on flat interfaces, low by about (sigma / R)^2 on a sphere of radius R; sigma
    of 1 to 1.5 voxels is the sensible range. Periodic; the shape must be powers of two.
    Calling it gives the 3 x 3 tensor in numpy axis order (its trace is the surface area,
    S/3 times the identity if isotropic; karambola's W1^{0,2} is a third of it);
    `densities` gives (3, 3, *shape), each voxel's share of the integral.
    """

    def __init__(self, shape: Sequence[int], sigma: float = 1.0, spacing=None, dtype=np.float32):
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3:
            raise ValueError(f"expected a 3D shape, got {self.shape}")
        self.dtype = np.dtype(dtype)
        cls = getattr(_rapidht, "_SurfaceTensor_" + _suffix(self.dtype))
        self._impl = cls(*_extent(self.shape), float(sigma), *_spacing(spacing, 3))

    def _volume_of_shape(self, volume) -> np.ndarray:
        v = _volume(volume)
        if v.shape != self.shape:
            raise ValueError(f"volume shape {v.shape} differs from {self.shape} given at construction")
        return v

    def __call__(self, volume) -> np.ndarray:
        return _tensor_matrix(self._impl.compute(self._volume_of_shape(volume)))

    def densities(self, volume) -> np.ndarray:
        v = self._volume_of_shape(volume)
        planes = np.empty((6, *self.shape), dtype=self.dtype)
        self._impl.densities(v, planes)
        xx, yy, zz, xy, xz, yz = planes
        return np.stack([np.stack([zz, yz, xz]), np.stack([yz, yy, xy]), np.stack([xz, xy, xx])])


def surface_normal_tensor(volume, sigma: float = 1.0, spacing=None) -> np.ndarray:
    """The 3 x 3 integral of n n dS of a 3D binary volume (see SurfaceNormalTensor)."""
    return SurfaceNormalTensor(np.shape(volume), sigma, spacing, dtype=np.float64)(volume)
