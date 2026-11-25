#!/usr/bin/env python3
"""Stack a folder of 16-bit TIFF slices into one raw volume, for R-5 (docs/R5_PLAN.md).

    python3 tools/r5_tiff_to_raw.py SRC_DIR OUT.raw [--crop X0 Y0 Z0 W H D]

Why raw: accuracy_integer and the benchmarks read a volume with one fread, in
the library's layout (x fastest, then y, then slice), with no TIFF reader in
C++. Why a sidecar: plan rule 10 compares accuracy runs only on files with the
same SHA-256, and the voxel geometry (93.3 x 93.3 x 250 um for the Korneev
columns) does not survive in a bare raw file.

Reads the slices in name order. Every slice must be an uncompressed,
single-strip, single-channel 16-bit TIFF of the same size -- that is what the
Korneev columns are (1024 x 1024, data at offset ~2.5 KB); anything else is
refused rather than guessed. Standard library and numpy only.

Writes OUT.raw (uint16, little-endian) and OUT.json (extent, dtype, slice
range, value statistics, SHA-256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys

import numpy as np

TAG_WIDTH, TAG_HEIGHT, TAG_BITS, TAG_COMPRESSION = 256, 257, 258, 259
TAG_STRIP_OFFSETS, TAG_SAMPLES, TAG_STRIP_COUNTS, TAG_SAMPLE_FORMAT = 273, 277, 279, 339


def read_slice(path: pathlib.Path) -> np.ndarray:
    """One uncompressed 16-bit TIFF slice as a (height, width) uint16 array."""
    data = path.read_bytes()
    if data[:2] not in (b"II", b"MM"):
        sys.exit(f"{path}: not a TIFF")
    bo = "<" if data[:2] == b"II" else ">"
    (ifd,) = struct.unpack(bo + "I", data[4:8])
    (count,) = struct.unpack(bo + "H", data[ifd : ifd + 2])
    tags: dict[int, list[int]] = {}
    sizes = {3: 2, 4: 4}
    for i in range(count):
        entry = data[ifd + 2 + 12 * i : ifd + 14 + 12 * i]
        tag, kind, n = struct.unpack(bo + "HHI", entry[:8])
        if kind not in sizes:
            continue
        fmt = bo + ("H" if kind == 3 else "I") * n
        raw = entry[8 : 8 + sizes[kind] * n]
        if sizes[kind] * n > 4:  # the values live elsewhere
            (at,) = struct.unpack(bo + "I", entry[8:12])
            raw = data[at : at + sizes[kind] * n]
        tags[tag] = list(struct.unpack(fmt, raw))
    width, height = tags[TAG_WIDTH][0], tags[TAG_HEIGHT][0]
    if tags.get(TAG_BITS, [0])[0] != 16 or tags.get(TAG_COMPRESSION, [1])[0] != 1:
        sys.exit(f"{path}: need uncompressed 16-bit data, got tags {tags}")
    if tags.get(TAG_SAMPLES, [1])[0] != 1 or tags.get(TAG_SAMPLE_FORMAT, [1])[0] != 1:
        sys.exit(f"{path}: need one unsigned-integer channel")
    if len(tags[TAG_STRIP_OFFSETS]) != 1:
        sys.exit(f"{path}: {len(tags[TAG_STRIP_OFFSETS])} strips; only single-strip slices are supported")
    start = tags[TAG_STRIP_OFFSETS][0]
    nbytes = width * height * 2
    if start + nbytes > len(data):
        sys.exit(f"{path}: strip runs past the end of the file")
    return np.frombuffer(data, dtype=bo + "u2", count=width * height, offset=start).reshape(height, width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src", type=pathlib.Path, help="folder of .tif slices")
    parser.add_argument("out", type=pathlib.Path, help="output .raw (a .json is written beside it)")
    parser.add_argument("--crop", type=int, nargs=6, metavar=("X0", "Y0", "Z0", "W", "H", "D"))
    args = parser.parse_args()

    slices = sorted(p for p in args.src.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    if not slices:
        sys.exit(f"{args.src}: no .tif files")
    first = read_slice(slices[0])
    full = (first.shape[1], first.shape[0], len(slices))
    x0, y0, z0, w, h, d = args.crop if args.crop else (0, 0, 0, *full)
    if x0 + w > full[0] or y0 + h > full[1] or z0 + d > full[2]:
        sys.exit(f"crop {args.crop} does not fit in the volume {full}")

    sha = hashlib.sha256()
    lo, hi, total, zeros = 65535, 0, 0, 0
    with open(args.out, "wb") as raw:
        for k, path in enumerate(slices[z0 : z0 + d]):
            img = read_slice(path)
            if img.shape != first.shape:
                sys.exit(f"{path}: {img.shape} differs from {first.shape}")
            block = np.ascontiguousarray(img[y0 : y0 + h, x0 : x0 + w], dtype="<u2")
            raw.write(block.tobytes())
            sha.update(block.tobytes())
            lo, hi = min(lo, int(block.min())), max(hi, int(block.max()))
            total += int(block.sum(dtype=np.uint64))
            zeros += int((block == 0).sum())
            if k % 100 == 0:
                print(f"  slice {k}/{d}: {path.name}", file=sys.stderr)

    count = w * h * d
    meta = {
        "source": str(args.src.resolve()),
        "slices": [slices[z0].name, slices[z0 + d - 1].name],
        "extent_xyz": [w, h, d],
        "crop_origin_xyz": [x0, y0, z0],
        "full_extent_xyz": list(full),
        "dtype": "uint16, little-endian, x fastest, then y, then slice",
        "bytes": count * 2,
        "sha256": sha.hexdigest(),
        "min": lo,
        "max": hi,
        "mean": total / count,
        "zero_fraction": zeros / count,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
