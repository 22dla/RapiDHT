"""The Python bindings against oracles written in numpy alone.

- Global functionals: the cells of the union of closed cubes on the periodic lattice counted
  with array rolls, then Michielsen & De Raedt's formulas -- no configuration table.
- Fields: the library's densities convolved with a numpy-built window by np.fft.
"""

import numpy as np
import pytest

import rapidht


def random_volume(shape, p=0.45, seed=0):
    return np.random.default_rng(seed).random(shape) < p


def cell_count_functionals(x: np.ndarray) -> dict[str, float]:
    """Periodic: a cell of the lattice is present if any voxel containing it is."""
    x = x.astype(bool)

    def any_of(axes):
        acc = np.zeros_like(x)
        for k in range(1 << len(axes)):
            sub = tuple(ax for i, ax in enumerate(axes) if k >> i & 1)
            acc |= np.roll(x, -1, axis=sub) if sub else x
        return acc

    nc = x.sum()
    nf = sum(any_of((a,)).sum() for a in range(3))  # face between a voxel and its +a neighbour
    ne = sum(any_of(p).sum() for p in ((0, 1), (0, 2), (1, 2)))  # edge shared by 4 voxels
    nv = any_of((0, 1, 2)).sum()
    return {
        "volume": float(nc),
        "surface": float(-6 * nc + 2 * nf),
        "mean_curvature": float(np.pi * (3 * nc - 2 * nf + ne)),
        "euler": float(-nc + nf - ne + nv),
    }


def ball_window(shape, radius):
    d = [np.minimum(np.arange(n), n - np.arange(n)) for n in shape]
    z, y, x = np.meshgrid(*d, indexing="ij")
    return (z**2 + y**2 + x**2 <= radius**2).astype(float)


def fft_convolve(f, g):
    return np.real(np.fft.ifftn(np.fft.fftn(f) * np.fft.fftn(g)))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("shape", [(8, 8, 8), (4, 8, 16), (6, 5, 7)])
def test_functionals_match_cell_count(shape, seed):
    x = random_volume(shape, p=0.2 + 0.2 * seed, seed=seed)
    got = rapidht.minkowski_functionals(x)
    want = cell_count_functionals(x)
    for k in rapidht.FUNCTIONALS:
        assert got[k] == pytest.approx(want[k], abs=1e-9), k


def test_known_bodies():
    n = 32
    c = np.indices((n, n, n)) + 0.5 - n / 2
    r = np.sqrt((c**2).sum(0))
    rho = np.sqrt(c[1] ** 2 + c[2] ** 2) - 9
    torus = rho**2 + c[0] ** 2 < 9
    shell = (r < 12) & (r > 6)
    box = np.zeros((n, n, n), bool)
    box[2:6, 3:10, 1:11] = True  # 4 x 7 x 10
    assert rapidht.minkowski_functionals(torus, "empty")["euler"] == pytest.approx(0)
    assert rapidht.minkowski_functionals(shell, "empty")["euler"] == pytest.approx(2)
    m = rapidht.minkowski_functionals(box, "empty")
    assert (m["volume"], m["surface"], m["euler"]) == pytest.approx((280, 276, 1))
    assert m["mean_curvature"] == pytest.approx(np.pi * 21)


def test_configuration_weights_match_thesis_columns():
    w = rapidht.configuration_weights()
    assert w.shape == (256, 4)
    assert w[1] * [8, 4, 8 / np.pi, 8] == pytest.approx([1, 3, 3, 1])  # one voxel
    assert w[105] * [8, 4, 8 / np.pi, 8] == pytest.approx([4, 12, -12, 4])  # tetrahedral four
    assert w[255] * [8, 4, 8 / np.pi, 8] == pytest.approx([8, 0, 0, 0])


@pytest.mark.parametrize("centring", ["voxel", "vertex"])
def test_densities_sum_to_functionals(centring):
    x = random_volume((8, 6, 10), seed=3)
    d = rapidht.minkowski_densities(x, centring=centring, dtype=np.float64)
    assert d.shape == (4, 8, 6, 10)
    g = rapidht.minkowski_functionals(x)
    assert d.sum(axis=(1, 2, 3)) == pytest.approx([g[k] for k in rapidht.FUNCTIONALS], abs=1e-9)


def test_voxel_densities_commute_with_inversion_and_transposition():
    x = random_volume((8, 8, 8), seed=5)
    d = rapidht.minkowski_densities(x, dtype=np.float64)
    inv = rapidht.minkowski_densities(np.flip(x), dtype=np.float64)
    np.testing.assert_allclose(inv, np.flip(d, axis=(1, 2, 3)), atol=1e-12)
    tr = rapidht.minkowski_densities(x.transpose(2, 0, 1), dtype=np.float64)
    np.testing.assert_allclose(tr, d.transpose(0, 3, 1, 2), atol=1e-12)


def test_window_matches_numpy():
    np.testing.assert_array_equal(rapidht.make_window((8, 16, 8), "ball", 3.2), ball_window((8, 16, 8), 3.2))
    shape, sigma = (16, 8, 8), 1.5
    g = rapidht.make_window(shape, "gaussian", sigma)
    # Separable sum over periodic images, per axis.
    axes = [
        np.exp(-((np.arange(n)[:, None] + n * np.arange(-5, 6)[None, :]) ** 2) / (2 * sigma**2)).sum(1) for n in shape
    ]
    np.testing.assert_allclose(g, np.einsum("i,j,k->ijk", *axes), rtol=1e-12)
    np.testing.assert_allclose(g, np.flip(np.roll(g, -1, axis=(0, 1, 2)), axis=(0, 1, 2)))  # even


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-9), (np.float32, 2e-3)])
def test_fields_match_fft_convolution(dtype, tol):
    shape, radii = (16, 8, 8), (1.5, 2.5, 3.2)
    x = random_volume(shape, seed=7)
    d = rapidht.minkowski_densities(x, dtype=np.float64)
    for normalisation in ("sum", "mean"):
        mf = rapidht.MinkowskiFields(shape, radii, normalisation=normalisation, dtype=dtype)
        out = mf(x)
        assert out.shape == (4, 3, *shape) and out.dtype == dtype
        for s, r in enumerate(radii):
            w = ball_window(shape, r)
            for f in range(4):
                want = fft_convolve(d[f], w)
                if normalisation == "mean":
                    want /= w.sum()
                np.testing.assert_allclose(out[f, s], want, atol=tol * max(1.0, np.abs(want).max()))
        np.testing.assert_allclose(mf.window_weights, [ball_window(shape, r).sum() for r in radii])


def test_mean_volume_field_is_local_fraction_of_full_volume():
    mf = rapidht.MinkowskiFields((8, 8, 8), (1.0, 3.0), window="gaussian", dtype=np.float64)
    out = mf(np.ones((8, 8, 8), bool))
    np.testing.assert_allclose(out[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(out[1:], 0.0, atol=1e-12)


def test_any_nonzero_is_foreground():
    x = random_volume((8, 8, 8), seed=1)
    a = rapidht.minkowski_functionals(x)
    b = rapidht.minkowski_functionals(np.where(x, 7, 0).astype(np.int32))
    assert a == b


def test_errors_are_python_exceptions():
    with pytest.raises(ValueError):
        rapidht.minkowski_functionals(np.zeros((8, 8), bool))
    with pytest.raises(ValueError):
        rapidht.MinkowskiFields((12, 16, 16), (2.0,))  # not a power of two
    with pytest.raises(ValueError):
        rapidht.MinkowskiFields((16, 16, 16), (8.0,))  # ball does not fit
    with pytest.raises(ValueError):
        rapidht.MinkowskiFields((16, 16, 16), (2.0,), window="cube")
    mf = rapidht.MinkowskiFields((8, 8, 8), (2.0,))
    with pytest.raises(ValueError):
        mf(np.zeros((8, 8, 16), bool))
    with pytest.raises(ValueError):
        rapidht.minkowski_densities(np.zeros((8, 8, 8), bool), dtype=np.int32)
