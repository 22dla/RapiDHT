"""Float32 model of the factored DHT in NumPy, to tell the algorithm from the kernels (R-4a).

The GPU path splits each axis as N = N1*N2 (N1 = 2^ceil(log2 N / 2)), sums over n1 with the
N1 x N1 cas matrix, applies the twiddles and sums over n2. This script does the same steps in
float32, pass by pass (z, y, x; separable, no Bracewell pass), and compares with a float64
reference built from numpy.fft. If the model shows the same worst points as accuracy_gpu3d,
the error belongs to the algorithm on that input, not to a kernel.

    python3 tools/r4_factored_emu.py 256 lcg       # the tests' LCG noise
    python3 tools/r4_factored_emu.py 256 white     # PCG64 noise on the same 2^-23 grid

At 256^3 it needs about 2 GiB of memory; 512^3 does not fit a small machine.
"""

import sys

import numpy as np


def lcg_noise(n: int) -> np.ndarray:
    """The tests' MakeNoise, vectorised by jumping the LCG ahead: s -> A_k s + C_k (mod 2^32)."""
    count = n**3
    mask = np.uint64(0xFFFFFFFF)
    a = np.array([1664525], np.uint64)
    c = np.array([1013904223], np.uint64)
    while a.size < count:
        am, cm = a[-1], c[-1]
        a = np.concatenate([a, (am * a) & mask])
        c = np.concatenate([c, (am * c + cm) & mask])
    seeds = (a[:count] * np.uint64(2463534242) + c[:count]) & mask
    return ((seeds >> np.uint64(8)).astype(np.float64) / 8388608.0 - 1.0).reshape(n, n, n)


def white_noise(n: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 1 << 24, size=(n, n, n)).astype(np.float64) / 8388608.0 - 1.0


def dht64(a: np.ndarray, axis: int) -> np.ndarray:
    f = np.fft.fft(a, axis=axis)
    return f.real - f.imag


def factored(a: np.ndarray, axis: int, dt) -> np.ndarray:
    """One factored 1D DHT along axis, every intermediate in dt."""
    n = a.shape[axis]
    n1 = 1 << ((n.bit_length() - 1 + 1) // 2)
    n2 = n // n1
    a = np.moveaxis(a, axis, -1).astype(dt)
    shape = a.shape[:-1]
    x = a.reshape(*shape, n1, n2)  # index n = n2 + N2*n1
    j = np.arange(n1)
    ph = 2 * np.pi * np.outer(j, j) / n1
    y = np.einsum("kj,...jm->...km", (np.cos(ph) + np.sin(ph)).astype(dt), x)  # Y[k1][n2]
    ym = y[..., (-j) % n1, :]  # Y[-k1][n2]
    t = 2 * np.pi * np.arange(n2)[None, :] * j[:, None] / n
    ct, st = np.cos(t).astype(dt), np.sin(t).astype(dt)
    p = y * ct + ym * st
    q = ym * ct - y * st
    b = 2 * np.pi * np.outer(np.arange(n2), np.arange(n2)) / n2
    h = np.einsum("...km,mq->...kq", p, np.cos(b).astype(dt)) + np.einsum("...km,mq->...kq", q, np.sin(b).astype(dt))
    return np.moveaxis(np.swapaxes(h, -1, -2).reshape(*shape, n), -1, axis)  # k = k1 + N1*k2


def report(label: str, actual: np.ndarray, ref: np.ndarray, rms: float) -> None:
    err = np.abs(actual.astype(np.float64) - ref) / rms
    z, y, x = np.unravel_index(err.argmax(), err.shape)
    print(f"  {label:<22} max {err.max():.2e} at (x {x}, y {y}, z {z})   p99.999 {np.quantile(err, 0.99999):.2e}")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    kind = sys.argv[2] if len(sys.argv) > 2 else "lcg"
    v = lcg_noise(n) if kind == "lcg" else white_noise(n)
    ref = dht64(dht64(dht64(v, 0), 1), 2)
    rms = float(np.sqrt((ref**2).mean()))
    f32, f64 = np.float32, np.float64
    print(f"{n}^3, {kind} noise, separable, errors relative to the RMS of the double spectrum")
    report("all passes float32", factored(factored(factored(v, 0, f32), 1, f32), 2, f32), ref, rms)
    for name, dts in (
        ("only z in float32", (f32, f64, f64)),
        ("only y in float32", (f64, f32, f64)),
        ("only x in float32", (f64, f64, f32)),
    ):
        a = v
        for axis, dt in enumerate(dts):
            a = factored(a, axis, dt).astype(f64)
        report(name, a, ref, rms)


if __name__ == "__main__":
    main()
