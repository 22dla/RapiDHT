"""RapiDHT from Python: Minkowski functionals of 3D binary volumes, their local densities
and their fields over a bank of windows (CPU).

    import numpy as np, rapidht
    solid = np.random.rand(64, 64, 64) < 0.4
    rapidht.minkowski_functionals(solid)            # {'volume': ..., 'surface': ..., ...}
    mf = rapidht.MinkowskiFields(solid.shape, radii=(2, 4, 8))
    fields = mf(solid)                              # (4, 3, 64, 64, 64): functional, scale, volume

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
    "MinkowskiFields",
    "configuration_weights",
    "make_window",
    "minkowski_densities",
    "minkowski_functionals",
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
