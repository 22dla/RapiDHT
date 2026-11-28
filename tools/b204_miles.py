#!/usr/bin/env python3
"""B-204: Minkowski functionals of a Boolean model of balls against Miles' formulas.

A Boolean model: germs of a Poisson process of intensity lam, each the centre of a ball
of radius R; the body is their union. Its functional densities per unit volume are known
in closed form (Miles 1976; Mecke & Wagner 1991). With the ball's volume v = 4/3 pi R^3,
surface s = 4 pi R^2 and integral mean curvature m = 4 pi R (of (1/r1 + 1/r2)/2, the
convention of RapiDHT), and q = exp(-lam v):

    V_V   = 1 - q
    S_V   = lam s q
    M_V   = (lam m - pi^2/32 lam^2 s^2) q
    chi_V = (lam - lam^2 m s / (4 pi) + pi/384 lam^3 s^3) q

This script draws realisations on a periodic L^3 grid (a voxel is foreground when its
centre is within R of a germ, minimum image), measures the global functionals with
rapidht.minkowski_functionals (periodic) and prints, as mean +- standard error over the
realisations:

    V      measured V_V / Miles (the expectation is exactly 1 - q: each voxel centre is
           covered with that probability),
    S      measured S_V / Miles,
    M      measured M_V / (lam m q), the first term of Miles' M_V; Miles gives
           1 - pi^2/32 lam s^2 / m, printed beside it,
    chi26  (measured chi_V - Miles) / (lam q), the error per germ, as a share of a
           lone ball's chi = 1; the library's chi, foreground 26-adjacent,
    chi6   the same with the foreground 6-adjacent: on the 3-torus that is the library's
           chi of the complement (the background then 26-adjacent).

M_V and chi_V change sign as the density grows, so a plain ratio would blow up there;
dividing by the first term keeps every column finite. S and M of a digitised smooth
surface are biased by about 3/2 (see include/rapidht/minkowski.h); the terms of the
intersections between balls are biased differently, which is what the table shows.

Result (2026-11-28, B-204): V agrees at every R and density; S is 1.3-1.55 of Miles;
chi is biased with either adjacency -- at mid densities by -0.4 to -0.65 per germ with the
foreground 26-adjacent (the library's), -0.1 to -0.27 with it 6-adjacent -- and the bias
does not shrink from R = 8 to R = 16 voxels (nor, on the same germs, from R/h = 4 to 32):
nearly touching balls are joined through edges and corners in many places at once.
This is the effect Ohser, Nagel and Schladitz describe for discretised Boolean models
(Image Anal. Stereol. 22, 2003). The library's chi is right for the voxel set -- it
equals an independent count of cells (python/tests) -- it is the voxel set's topology
that differs from the continuum's.

    python3 tools/b204_miles.py [--reps 8] [--seed 1]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import rapidht


def miles(lam: float, radius: float) -> dict[str, float]:
    v = 4.0 / 3.0 * math.pi * radius**3
    s = 4.0 * math.pi * radius**2
    m = 4.0 * math.pi * radius
    q = math.exp(-lam * v)
    return {
        "volume": 1.0 - q,
        "surface": lam * s * q,
        "mean_curvature": (lam * m - math.pi**2 / 32.0 * lam**2 * s**2) * q,
        "euler": (lam - lam**2 * m * s / (4.0 * math.pi) + math.pi / 384.0 * lam**3 * s**3) * q,
    }


def boolean_model(size: int, lam: float, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Union of balls around Poisson germs on a periodic size^3 grid of unit voxels."""
    body = np.zeros((size, size, size), dtype=bool)
    count = rng.poisson(lam * size**3)
    germs = rng.uniform(0.0, size, size=(count, 3))
    reach = int(math.ceil(radius)) + 1
    offsets = np.arange(-reach, reach + 1)
    for g in germs:
        # Voxel centres at integer + 1/2; the ones near the germ, wrapped.
        base = np.floor(g).astype(int)
        idx = [(base[a] + offsets) % size for a in range(3)]
        d = [(base[a] + offsets + 0.5) - g[a] for a in range(3)]
        r2 = d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2
        inside = r2 <= radius * radius
        body[np.ix_(idx[0], idx[1], idx[2])] |= inside
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    # Grid sizes that hold some hundreds of balls at the lowest density.
    cases = [(r, size, phi) for r, size in ((4.0, 128), (8.0, 128), (16.0, 256)) for phi in (0.05, 0.2, 0.5, 0.8)]
    print(f"{args.reps} realisations each; mean +- standard error")
    header = ("R", "L", "phi", "germs", "V / Miles", "S / Miles", "M / (lam m q)", "Miles", "dchi26", "dchi6")
    print("{:>5} {:>4} {:>5} {:>6} {:>17} {:>17} {:>17} {:>7} {:>17} {:>17}".format(*header))
    for radius, size, phi in cases:
        v = 4.0 / 3.0 * math.pi * radius**3
        s = 4.0 * math.pi * radius**2
        m = 4.0 * math.pi * radius
        lam = -math.log(1.0 - phi) / v
        q = math.exp(-lam * v)
        expected = miles(lam, radius)
        cell = float(size**3)
        cols: dict[str, list[float]] = {"V": [], "S": [], "M": [], "chi": [], "chi6": []}
        for _ in range(args.reps):
            body = boolean_model(size, lam, radius, rng)
            mf = rapidht.minkowski_functionals(body, boundary="periodic")
            chi6 = rapidht.minkowski_functionals(~body, boundary="periodic")["euler"]
            cols["V"].append(mf["volume"] / cell / expected["volume"])
            cols["S"].append(mf["surface"] / cell / expected["surface"])
            cols["M"].append(mf["mean_curvature"] / cell / (lam * m * q))
            cols["chi"].append((mf["euler"] / cell - expected["euler"]) / (lam * q))
            cols["chi6"].append((chi6 / cell - expected["euler"]) / (lam * q))

        def cell_text(values: list[float]) -> str:
            a = np.array(values)
            se = a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else float("nan")
            return f"{a.mean():8.3f} +- {se:<5.3f}"

        miles_m = 1.0 - math.pi**2 / 32.0 * lam * s * s / m
        print(
            f"{radius:>5.0f} {size:>4} {phi:>5.2f} {lam * cell:>6.0f} {cell_text(cols['V']):>17} "
            f"{cell_text(cols['S']):>17} {cell_text(cols['M']):>17} {miles_m:>7.3f} {cell_text(cols['chi']):>17} {cell_text(cols['chi6']):>17}"
        )


if __name__ == "__main__":
    main()
