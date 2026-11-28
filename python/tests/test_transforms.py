"""The transform, convolutions, kernels and surface tensor bindings against numpy oracles.

- DHT: Re(F) - Im(F) of np.fft.fftn, which is sum x cas(2 pi k.x / n) by definition.
- Separable DHT: the same along one axis at a time.
- Convolutions: products of np.fft spectra; the linear one on a zero-padded copy.
- Butterworth: its response evaluated on np.fft.fftfreq, the frequency in cycles per unit.
"""

import numpy as np
import pytest

import rapidht

SHAPES = [(16,), (8, 4), (4, 16), (8, 4, 2), (2, 8, 16)]


def signal(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def numpy_dht(x, axes=None):
    f = np.fft.fftn(x, axes=axes)
    return f.real - f.imag


def numpy_separable_dht(x):
    for ax in range(x.ndim):
        x = numpy_dht(x, axes=(ax,))
    return x


def circular_convolve(a, k):
    return np.real(np.fft.ifftn(np.fft.fftn(a) * np.fft.fftn(k)))


def linear_convolve_same(a, k):
    """sum_t a(x - t) k(t), t centred, zeros outside: full linear convolution, centre slice."""
    full = [n + m - 1 for n, m in zip(a.shape, k.shape)]
    axes = tuple(range(a.ndim))
    out = np.real(np.fft.ifftn(np.fft.fftn(a, full, axes) * np.fft.fftn(k, full, axes), axes=axes))
    return out[tuple(slice(m // 2, m // 2 + n) for n, m in zip(a.shape, k.shape))]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype, tol", [(np.float64, 1e-12), (np.float32, 2e-5)])
def test_dht_matches_numpy_fft(shape, dtype, tol):
    x = signal(shape)
    scale = np.abs(numpy_dht(x)).max()
    assert np.allclose(rapidht.dht(x.astype(dtype)), numpy_dht(x), atol=tol * scale, rtol=0)
    assert rapidht.dht(x.astype(dtype)).dtype == dtype
    assert np.allclose(rapidht.idht(rapidht.dht(x)), x, atol=1e-12)
    assert np.allclose(rapidht.separable_dht(x), numpy_separable_dht(x), atol=1e-12 * scale, rtol=0)


def test_transform_object_is_reused_and_leaves_its_input():
    t = rapidht.HartleyTransform((8, 4, 2), dtype=np.float32)
    x = signal((8, 4, 2)).astype(np.float32)
    before = x.copy()
    for _ in range(2):
        assert np.allclose(t.forward(x), numpy_dht(x), atol=1e-4)
    assert np.array_equal(x, before)
    with pytest.raises(ValueError):
        t.forward(np.zeros((8, 4, 4)))
    with pytest.raises(ValueError):
        rapidht.dht(np.zeros(12))  # not a power of two


@pytest.mark.parametrize("shape", SHAPES)
def test_circular_convolution_matches_fft_for_any_and_even_kernels(shape):
    a = signal(shape, 1)
    k = signal(shape, 2)
    conv = rapidht.CircularConvolution(k)
    assert not conv.separable
    assert np.allclose(conv(a), circular_convolve(a, k), atol=1e-11)

    even = k.copy()
    for ax in range(len(shape)):
        even = even + np.roll(np.flip(even, axis=ax), 1, axis=ax)  # g(-i) + g(i), mod n
    conv = rapidht.CircularConvolution(even)
    assert conv.separable
    assert np.allclose(conv(a), circular_convolve(a, even), atol=1e-10)


@pytest.mark.parametrize(
    "shape, kshape",
    [((13,), (5,)), ((3,), (9,)), ((7, 5), (3, 5)), ((6, 5, 3), (5, 3, 3)), ((4, 1, 7), (3, 1, 5))],
)
def test_linear_convolution_matches_padded_fft(shape, kshape):
    a = signal(shape, 3)
    k = np.abs(signal(kshape, 4)) + 0.1  # positive, for the normalised case
    assert np.allclose(rapidht.LinearConvolution(shape, k)(a), linear_convolve_same(a, k), atol=1e-11)

    norm = rapidht.LinearConvolution(shape, k, edges="normalized")
    weight = linear_convolve_same(np.ones(shape), k)
    assert np.allclose(norm(a), linear_convolve_same(a, k) / weight, atol=1e-11)

    c = (np.arange(np.prod(shape)).reshape(shape) % 3 != 0).astype(float)
    expected = linear_convolve_same(a * c, k) / linear_convolve_same(c, k)
    got = norm(a, certainty=c)
    finite = linear_convolve_same(c, k) > 1e-9
    assert np.allclose(got[finite], expected[finite], atol=1e-10)
    assert np.all(got[~finite] == 0.0)


def test_linear_convolution_padding_and_errors():
    conv = rapidht.LinearConvolution((300, 500), np.ones((3, 21)))
    assert conv.padded_shape == (512, 512)
    with pytest.raises(ValueError):
        rapidht.LinearConvolution((8, 8), np.ones((4, 3)))  # even extent
    with pytest.raises(ValueError):
        rapidht.LinearConvolution((8, 8), np.ones(3))  # axes differ
    with pytest.raises(ValueError):
        conv(np.zeros((300, 500)), certainty=np.ones((300, 500)))  # certainty, zero edges


def test_gaussian_kernel_in_physical_units_numpy_order():
    # Axis 0 has slices 2.5 thick: the kernel reaches ceil(8 / 2.5) = 4 voxels there.
    g = rapidht.gaussian_kernel(3, 2.0, spacing=(2.5, 0.5, 1.0))
    assert g.shape == (9, 33, 17)
    assert g.sum() == pytest.approx(1.0, abs=1e-14)
    z, y, x = np.meshgrid(*[(np.arange(n) - n // 2) * s for n, s in zip(g.shape, (2.5, 0.5, 1.0))], indexing="ij")
    expected = np.exp(-(x**2 + y**2 + z**2) / 8.0)
    assert np.allclose(g, expected / expected.sum(), atol=1e-15)
    assert rapidht.gaussian_kernel(1, 1.0).shape == (9,)
    assert rapidht.gaussian_kernel(2, 1.0, spacing=2.0, truncate=3.0).shape == (5, 5)  # ceil(3 / 2) = 2


def test_ball_kernel_counts_voxel_centres():
    b = rapidht.ball_kernel(3, 4.0, spacing=(2.0, 1.5, 1.0), normalise=False)
    assert b.shape == (5, 5, 9)
    z, y, x = np.meshgrid(*[(np.arange(n) - n // 2) * s for n, s in zip(b.shape, (2.0, 1.5, 1.0))], indexing="ij")
    assert np.array_equal(b, (x**2 + y**2 + z**2 <= 16.0 + 1e-9).astype(float))
    assert rapidht.ball_kernel(2, 3.0).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("shape, spacing", [((16, 8, 4), (2.0, 0.5, 1.0)), ((8, 16), (1.0, 0.7)), ((32,), (2.0,))])
def test_butterworth_spectrum_is_its_response(shape, spacing):
    cutoff, order = 0.2, 2
    k = rapidht.butterworth_kernel(shape, cutoff, order, spacing=spacing)
    f = np.meshgrid(*[np.fft.fftfreq(n, d) for n, d in zip(shape, spacing)], indexing="ij")
    response = 1.0 / (1.0 + (sum(fi**2 for fi in f) / cutoff**2) ** order)
    assert np.allclose(numpy_dht(k), response, atol=1e-12)


def test_wrap_kernel_matches_numpy_placement():
    k = signal((3, 5, 7), 5)
    wrapped = rapidht.wrap_kernel(k, (8, 4, 16))
    expected = np.zeros((8, 4, 16))
    for idx in np.ndindex(k.shape):
        target = tuple((i - m // 2) % n for i, m, n in zip(idx, k.shape, (8, 4, 16)))
        expected[target] += k[idx]  # the 5 taps along axis 1 overlap in a period of 4
    assert np.allclose(wrapped, expected, atol=1e-15)


def test_surface_normal_tensor_numpy_axis_order():
    # A slab normal to numpy axis 0, slices 2 thick: two interfaces of (16 * 0.5) by (32 * 1).
    v = np.zeros((64, 16, 32), dtype=bool)
    v[20:40] = True
    t = rapidht.surface_normal_tensor(v, sigma=1.5, spacing=(2.0, 0.5, 1.0))
    expected = np.zeros((3, 3))
    expected[0, 0] = 2.0 * (16 * 0.5) * (32 * 1.0)
    assert np.allclose(t, expected, atol=1e-8 * expected[0, 0])

    snt = rapidht.SurfaceNormalTensor(v.shape, sigma=1.5, spacing=(2.0, 0.5, 1.0))
    d = snt.densities(v)
    assert d.shape == (3, 3, 64, 16, 32)
    assert np.allclose(d.sum(axis=(2, 3, 4)), expected, atol=1e-4 * expected[0, 0])
    assert np.allclose(d, np.swapaxes(d, 0, 1))


def test_torch_tensors_go_in_and_out_without_copies():
    torch = pytest.importorskip("torch")
    x = torch.randn(8, 4, 2, dtype=torch.float64)
    h = rapidht.dht(x)
    assert np.allclose(h, numpy_dht(x.numpy()), atol=1e-12)
    back = torch.from_numpy(h)
    assert back.data_ptr() == h.ctypes.data
