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
centre is within R of a germ, minimum image) and prints, as mean +- standard error over the
realisations:

    V      measured V_V / Miles (the expectation is exactly 1 - q: each voxel centre is
           covered with that probability),
    S      measured S_V / Miles,
    M      measured M_V / (lam m q), the first term of Miles' M_V; Miles gives
           1 - pi^2/32 lam s^2 / m, printed beside it,
    nerve  (chi of the union of the balls themselves - Miles) / germs: chi from the Cech
           nerve (nerve theorem: sum over k of (-1)^k times the number of (k+1)-sets of
           balls with a common point, i.e. a minimal enclosing ball of radius <= R) --
           an independent check of the formula and of the model, with no voxels in it,
    26, 6, 14   (digital chi - nerve chi) / germs, the digitisation error per ball on the
           same realisation: the library's chi (foreground 26-adjacent), the foreground
           6-adjacent (on the 3-torus the library's chi of the complement), and
           EulerAdjacency::Kuhn14 (adjacency="kuhn14", both 14-adjacent).

S and M of a digitised smooth surface are biased by about 3/2 (see
include/rapidht/minkowski.h); the terms of the intersections between balls are biased
differently, which is what the table shows.

Result (2026-11-28, B-204, B-206): V agrees at every R and density, and so does the nerve:
Miles' chi and the model are right. Digital chi is not, with any of the three adjacencies,
and the error does not shrink from R = 8 to R = 16 voxels (nor, on the same germs, from
R/h = 4 to 32). The cause is pairs of balls closer than about 1.5 voxels: their digital
images touch through edges and corners in many places along the gap, every place after the
first a spurious tunnel -- one such pair alone gives chi down to -12 instead of 2 at
R = 16. Such pairs get rarer as h / R, the tunnels per pair more numerous as R / h, so the
bias stays. No local adjacency can repair it: the gap is below the voxel. The library's chi
is right for the voxel set -- it equals independent counts of cells (python/tests) and of
Freudenthal simplices -- it is the voxel set's topology that differs from the balls'.

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


def boolean_model(size: int, radius: float, germs: np.ndarray) -> np.ndarray:
    """Union of balls around the germs on a periodic size^3 grid of unit voxels."""
    body = np.zeros((size, size, size), dtype=bool)
    reach = math.ceil(radius) + 1
    offsets = np.arange(-reach, reach + 1)
    for g in germs:
        # Voxel centres at integer + 1/2; the ones near the germ, wrapped.
        base = np.floor(g).astype(int)
        idx = [(base[a] + offsets) % size for a in range(3)]
        d = [(base[a] + offsets + 0.5) - g[a] for a in range(3)]
        r2 = d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2
        body[np.ix_(idx[0], idx[1], idx[2])] |= r2 <= radius * radius
    return body


def miniball_radius(points: np.ndarray) -> float:
    """Radius of the smallest ball enclosing a few points (Welzl's algorithm)."""

    def circumscribed(support: list[np.ndarray]) -> tuple[np.ndarray, float]:
        if not support:
            return np.zeros(3), -1.0
        if len(support) == 1:
            return support[0], 0.0
        a = np.array([p - support[0] for p in support[1:]])
        y = np.linalg.solve(a @ a.T, 0.5 * np.sum(a * a, axis=1))
        centre = support[0] + a.T @ y
        return centre, float(np.linalg.norm(centre - support[0]))

    def welzl(rest: list[np.ndarray], support: list[np.ndarray]) -> tuple[np.ndarray, float]:
        if not rest or len(support) == 4:
            return circumscribed(support)
        centre, radius = welzl(rest[:-1], support)
        if radius >= 0.0 and np.linalg.norm(rest[-1] - centre) <= radius + 1e-9:
            return centre, radius
        return welzl(rest[:-1], [*support, rest[-1]])

    return welzl(list(points), [])[1]


def nerve_euler(germs: np.ndarray, radius: float, size: int) -> int:
    """chi of the union of the balls on the 3-torus, from the Cech nerve (2R < size / 2)."""
    n = len(germs)

    def offset(i: int, j: int) -> np.ndarray:
        d = germs[j] - germs[i]
        return d - size * np.round(d / size)

    later = []
    for i in range(n):
        d = germs - germs[i]
        d -= size * np.round(d / size)
        close = np.nonzero(np.einsum("ij,ij->i", d, d) < 4.0 * radius * radius)[0]
        later.append({int(j) for j in close if j > i})
    chi, sign = n, -1
    level = [((i,), later[i]) for i in range(n)]
    while level:
        grown = []
        for simplex, candidates in level:
            for j in sorted(candidates):
                vertices = (*simplex, j)
                points = np.array([np.zeros(3)] + [offset(vertices[0], v) for v in vertices[1:]])
                if miniball_radius(points) <= radius:
                    grown.append((vertices, candidates & later[j]))
        chi += sign * len(grown)
        sign = -sign
        level = grown
    return chi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    # L = 16 R: the same number of balls in every row, and the minimum image of every
    # pair of overlapping balls unique.
    cases = [(r, size, phi) for r, size in ((4.0, 64), (8.0, 128), (16.0, 256)) for phi in (0.05, 0.2, 0.5, 0.8)]
    print(f"{args.reps} realisations each; mean +- standard error; the chi columns are per ball")
    header = ("R", "L", "phi", "germs", "V / Miles", "S / Miles", "M / (lam m q)", "Miles", "nerve - Miles")
    header += ("26 - nerve", "6 - nerve", "14 - nerve")
    print(("{:>5} {:>4} {:>5} {:>6}" + " {:>16}" * 3 + " {:>6}" + " {:>16}" * 4).format(*header))
    for radius, size, phi in cases:
        v = 4.0 / 3.0 * math.pi * radius**3
        s = 4.0 * math.pi * radius**2
        m = 4.0 * math.pi * radius
        lam = -math.log(1.0 - phi) / v
        q = math.exp(-lam * v)
        expected = miles(lam, radius)
        cell = float(size**3)
        cols: dict[str, list[float]] = {k: [] for k in ("V", "S", "M", "nerve", "26", "6", "14")}
        for _ in range(args.reps):
            germs = rng.uniform(0.0, size, size=(rng.poisson(lam * cell), 3))
            body = boolean_model(size, radius, germs)
            mf = rapidht.minkowski_functionals(body, boundary="periodic")
            balls = max(len(germs), 1)
            exact = nerve_euler(germs, radius, size)
            cols["V"].append(mf["volume"] / cell / expected["volume"])
            cols["S"].append(mf["surface"] / cell / expected["surface"])
            cols["M"].append(mf["mean_curvature"] / cell / (lam * m * q))
            cols["nerve"].append((exact - expected["euler"] * cell) / balls)
            cols["26"].append((mf["euler"] - exact) / balls)
            cols["6"].append((rapidht.minkowski_functionals(~body)["euler"] - exact) / balls)
            cols["14"].append((rapidht.minkowski_functionals(body, adjacency="kuhn14")["euler"] - exact) / balls)

        def cell_text(values: list[float]) -> str:
            a = np.array(values)
            se = a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else float("nan")
            return f"{a.mean():7.3f} +- {se:<5.3f}"

        miles_m = 1.0 - math.pi**2 / 32.0 * lam * s * s / m
        texts = [cell_text(cols[k]) for k in ("V", "S", "M")] + [f"{miles_m:6.3f}"]
        texts += [cell_text(cols[k]) for k in ("nerve", "26", "6", "14")]
        print(f"{radius:>5.0f} {size:>4} {phi:>5.2f} {lam * cell:>6.0f} " + " ".join(texts))


if __name__ == "__main__":
    main()
