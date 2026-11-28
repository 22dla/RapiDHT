#!/usr/bin/env python3
"""B-107: how filtering a tomogram before thresholding moves the boundaries of the pore space.

Ground truth: a periodic Boolean model of solid balls (radius R, solid fraction phi) in a box
of physical size Lx x Ly x Lz. It is "scanned" on a grid of anisotropic voxels -- slices dz
thick, pixels dx = dy = 1 -- the way a tomogram with thick slices is (dz / dx = 2.7 is the
ratio behind the anisotropy question of B-401):

    grey = PSF * (partial-volume fraction of solid in each voxel) + noise,

the PSF an isotropic Gaussian in physical units. Each filter below is applied, the result is
thresholded half-way between the two phases (0.5, both phase values being known), and the
segmentation is compared with the truth:

    dphi     solid fraction of the segmentation minus that of the truth,
    shift    mean signed distance of the segmented solid's boundary voxels from the true
             surface (exact signed distance to the union of balls, physical units; > 0 is
             outside the solid, i.e. the solid grew),
    rms      root mean square of that distance,
    aniso    W_zz / ((W_xx + W_yy) / 2) of the surface normal tensor (RapiDHT
             SurfaceNormalTensor, physical spacing): 1 for the isotropic truth; > 1 means
             more surface facing z, i.e. structures flattened across the slices.

Filters (all periodic, through RapiDHT unless noted):

    none          the scan itself,
    gauss2d       Gaussian in each slice only, sigma_xy in physical units (the slice-wise
                  practice),
    gauss3d       Gaussian with the same physical sigma along every axis (spacing-aware),
    gauss3d-vox   Gaussian of sigma voxels along every axis, the spacing ignored: physically
                  2.7 times wider along z,
    pm3d          Perona-Malik anisotropic diffusion, spacing-aware (numpy),
    nlm2d, nlm3d  non-local means slice by slice / in 3D (skimage; skipped without it).

Result (2026-11-28, 3 realisations of 32 x 64 x 64 voxels, R = 5, solid fraction 0.6, PSF 1,
noise 0.15, filter sigma 1.5; shifts relative to the truth row):

    dz/dx = 2.7      dphi    shift   aniso      dz/dx = 1      dphi    shift   aniso
    truth            0       0       1.01       truth          0       0       1.00
    none            +0.007  +0.04    1.02       none          +0.010  -0.00    1.01
    gauss2d         +0.027  +0.07    1.13       gauss2d       +0.031  +0.08    1.22
    gauss3d         +0.036  +0.12    1.02       gauss3d       +0.040  +0.11    1.02
    gauss3d-vox     +0.079  +0.39    0.65       (= gauss3d)
    pm3d            +0.081  +0.27    1.01       pm3d          +0.083  +0.22    1.02
    nlm2d           +0.014  +0.03    1.06       nlm2d         +0.018  +0.05    1.12
    nlm3d           +0.024  +0.12    0.93       nlm3d         +0.021  +0.06    1.02

Every filter followed by the threshold grows the solid (curvature: narrow pores close
first). Slice-wise filters add surface facing z (+12 to +22 %: structures flattened across
the slices). A 3D Gaussian of sigma voxels on thick slices, the spacing ignored, does the
opposite and three times as strongly: 35 % less surface facing z, structures drawn out along
z -- a candidate mechanism for the k33 / k11 of B-401. Only filters in physical units leave
the anisotropy alone (skimage's NLM in 3D works in voxels, hence its 0.93).

The true distance is analytic, so numpy and rapidht are enough; skimage adds the NLM rows.
The "truth" row is the voxel-centre segmentation of the truth itself: its shift (boundary
voxel centres sit inside the solid by part of a voxel) and aniso (the grid's own) are the
baselines the filters are compared with. Run from an environment with rapidht, e.g.

    ~/projects/hno-porous/.venv/bin/python tools/b107_boundary_shift.py [--reps 3]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import rapidht

try:
    from skimage.restoration import denoise_nl_means
except ImportError:  # the NLM rows are skipped
    denoise_nl_means = None


def boolean_model(shape, spacing, radius, phi, rng):
    """Germs of a periodic Boolean model of balls in the physical box of the grid."""
    box = np.array([n * s for n, s in zip(shape, spacing)])
    lam = -math.log(1.0 - phi) / (4.0 / 3.0 * math.pi * radius**3)
    return rng.uniform(0.0, 1.0, size=(rng.poisson(lam * box.prod()), 3)) * box, box


def signed_distance(points, germs, box, radius):
    """Exact signed distance to the union of balls, periodic (negative inside the solid)."""
    best = np.full(points.shape[0], np.inf)
    for g in germs:
        d = points - g
        d -= box * np.round(d / box)
        best = np.minimum(best, np.sqrt(np.einsum("ij,ij->i", d, d)))
    return best - radius


def scan(shape, spacing, germs, box, radius, sub=2):
    """Partial-volume solid fraction per voxel, from sub^3 samples in each voxel."""
    frac = np.zeros(shape)
    offsets = (np.arange(sub) + 0.5) / sub
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).reshape(-1, 3)
    for o0 in offsets:
        for o1 in offsets:
            for o2 in offsets:
                p = (grid + np.array([o0, o1, o2])) * np.array(spacing)
                frac += (signed_distance(p, germs, box, radius) <= 0.0).reshape(shape)
    return frac / sub**3


def centres(shape, spacing):
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).reshape(-1, 3)
    return (grid + 0.5) * np.array(spacing)


def gaussian(volume, sigma, spacing, axes=(0, 1, 2)):
    """Periodic Gaussian smoothing with sigma in physical units along `axes` only."""
    sp = [s if a in axes else 1e9 for a, s in enumerate(spacing)]  # a huge spacing: one tap
    k = rapidht.gaussian_kernel(3, sigma, spacing=sp)
    return rapidht.CircularConvolution(rapidht.wrap_kernel(k, volume.shape))(volume)


def perona_malik(volume, spacing, kappa, steps, dt=0.1):
    """Explicit Perona-Malik diffusion, g = exp(-(|grad| / kappa)^2), periodic, spacing-aware."""
    u = volume.copy()
    for _ in range(steps):
        flux = np.zeros_like(u)
        for axis, h in enumerate(spacing):
            fwd = (np.roll(u, -1, axis) - u) / h
            bwd = (u - np.roll(u, 1, axis)) / h
            flux += (np.exp(-((fwd / kappa) ** 2)) * fwd - np.exp(-((bwd / kappa) ** 2)) * bwd) / h
        u += dt * min(spacing) ** 2 * flux
    return u


def measure(segmented, truth_fraction, sd_centres, spacing, tensor):
    solid = segmented
    # Boundary voxels of the segmented solid: solid with a non-solid 6-neighbour.
    boundary = np.zeros_like(solid)
    for axis in range(3):
        for shift in (1, -1):
            boundary |= solid & ~np.roll(solid, shift, axis)
    d = sd_centres.reshape(solid.shape)[boundary]
    w = tensor(solid)
    return {
        "dphi": solid.mean() - truth_fraction,
        "shift": float(d.mean()),
        "rms": float(np.sqrt((d**2).mean())),
        "aniso": w[0, 0] / ((w[1, 1] + w[2, 2]) / 2.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dz", type=float, default=2.7, help="slice thickness over pixel size")
    parser.add_argument("--radius", type=float, default=5.0, help="ball radius, pixels")
    parser.add_argument("--phi", type=float, default=0.6, help="solid fraction")
    parser.add_argument("--psf", type=float, default=1.0, help="PSF sigma, pixels")
    parser.add_argument("--noise", type=float, default=0.15, help="noise sd, phase contrast 1")
    parser.add_argument("--sigma", type=float, default=1.5, help="filter sigma, pixels")
    args = parser.parse_args()

    # numpy order: axis 0 is z (the slices), axes 1, 2 are y, x.
    shape = (32, 64, 64)
    spacing = (args.dz, 1.0, 1.0)
    rng = np.random.default_rng(args.seed)
    # The tensor's own smoothing at least a voxel along every axis (B-205: below a voxel the
    # staircase shows), so one slice thickness.
    tensor = rapidht.SurfaceNormalTensor(shape, sigma=max(spacing), spacing=spacing, dtype=np.float64)

    rows: dict[str, list[dict[str, float]]] = {}
    for _ in range(args.reps):
        germs, box = boolean_model(shape, spacing, args.radius, args.phi, rng)
        frac = scan(shape, spacing, germs, box, args.radius)
        sd = signed_distance(centres(shape, spacing), germs, box, args.radius)
        truth = sd.reshape(shape) <= 0.0
        grey = gaussian(frac, args.psf, spacing) + rng.normal(0.0, args.noise, shape)

        filtered = {
            "truth": truth.astype(float),
            "none": grey,
            "gauss2d": gaussian(grey, args.sigma, spacing, axes=(1, 2)),
            "gauss3d": gaussian(grey, args.sigma, spacing),
            "gauss3d-vox": gaussian(grey, args.sigma, (1.0, 1.0, 1.0)),
            "pm3d": perona_malik(grey, spacing, kappa=0.3, steps=40),
        }
        if denoise_nl_means is not None:
            s = args.noise  # known here; a real scan would estimate it
            filtered["nlm2d"] = np.stack(
                [denoise_nl_means(sl, h=0.8 * s, sigma=s, patch_size=5, patch_distance=6) for sl in grey]
            )
            filtered["nlm3d"] = denoise_nl_means(grey, h=0.8 * s, sigma=s, patch_size=3, patch_distance=4)
        for name, f in filtered.items():
            rows.setdefault(name, []).append(measure(f >= 0.5, truth.mean(), sd, spacing, tensor))

    print(
        f"{args.reps} realisations of {shape} voxels, dz/dx = {args.dz}, R = {args.radius}, "
        f"phi = {args.phi}, PSF {args.psf}, noise {args.noise}, filter sigma {args.sigma} (pixels)"
    )
    print(f"{'filter':>12} {'dphi':>16} {'shift':>16} {'rms':>16} {'aniso z/xy':>16}")
    for name, rs in rows.items():
        cells = []
        for key in ("dphi", "shift", "rms", "aniso"):
            a = np.array([r[key] for r in rs])
            se = a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else float("nan")
            cells.append(f"{a.mean():8.4f} +- {se:<5.4f}")
        print(f"{name:>12} " + " ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
