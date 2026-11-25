"""Figures of the R-5 track (docs/R5_PLAN.md).

    python3 tools/r5_figures.py [RUN.json ...]      # writes docs/images/r5-budget.svg

r5-budget: time of "uint16 -> Hartley spectrum" at 512^3, on the device and
from the host, from r5_measure.sh runs. Every number is a median of one
bench_transform JSON; with several files given (or found), each row is taken
from the newest file that has it -- the device rows from one run (they are
compared with each other), the bus rows (H2D/, R5_BUS=1) possibly from another.

Segments are differences of rows of the same run, so they add up to the
measured totals:
  conversion uint16 -> float   cuFFTInt - cuFFT, IntConvertFactored - FactoredResident
  Hartley pass of cuFFT        cuFFT - cuFFT_R2C
  cuFFT with the LTO callback  cuFFTIntLto - (Hartley pass)
The only hatched segment is the target of R-5 (section 4 of the plan), which
is an estimate until stage 2 measures it.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SIZE = "3D/512x512x512"
TARGET_MS = 8.8  # R-5 target, section 4 of docs/R5_PLAN.md: 8.6-9.0 ms


def medians(path: pathlib.Path) -> dict[str, float]:
    """run_name -> median in ms."""
    unit = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
    out = {}
    for b in json.loads(path.read_text(encoding="utf-8")).get("benchmarks", []):
        if b.get("aggregate_name") == "median" and not b.get("error_occurred"):
            out[b["run_name"]] = b["real_time"] * unit[b.get("time_unit", "ns")]
    return out


def collect(files: list[pathlib.Path], names: list[str]) -> tuple[dict[str, float], pathlib.Path]:
    """The rows `names` from the newest file that has all of them."""
    for path in sorted(files, reverse=True):
        rows = medians(path)
        if all(n in rows for n in names):
            return {n: rows[n] for n in names}, path
    sys.exit(f"no run has all of {names}")


def main() -> None:
    files = [pathlib.Path(a) for a in sys.argv[1:]] or sorted((ROOT / "benchmarks" / "results").glob("*_r5_*.json"))
    dev_names = [f"cuFFT/f32/{SIZE}", f"cuFFT_R2C/f32/{SIZE}", f"FactoredResident/f32/{SIZE}",
                 f"cuFFTInt/u16/{SIZE}", f"cuFFTIntLto/u16/{SIZE}", f"IntConvertFactored/u16/{SIZE}"]
    bus_names = [f"H2D/u16/pageable/{SIZE}", f"H2D/f32/pageable/{SIZE}",
                 f"H2D/u16/pinned/{SIZE}", f"H2D/f32/pinned/{SIZE}"]
    d, dev_file = collect(files, dev_names)
    b, bus_file = collect(files, bus_names)
    cufft, r2c, ours = d[dev_names[0]], d[dev_names[1]], d[dev_names[2]]
    hart = cufft - r2c
    conv_cufft = d[dev_names[3]] - cufft
    conv_ours = d[dev_names[5]] - ours
    lto_core = d[dev_names[4]] - hart

    # (label, [(kind, ms, measured)])
    dev = [
        ("cuFFT: перевод в float + R2C\n+ перевод в Хартли", [("conv", conv_cufft, True), ("core", r2c, True), ("hart", hart, True)]),
        ("cuFFT: R2C с колбэком LTO\n+ перевод в Хартли", [("core", lto_core, True), ("hart", hart, True)]),
        ("RapiDHT сейчас: перевод в float\n+ факторизованный путь", [("conv", conv_ours, True), ("core", ours, True)]),
        ("RapiDHT, цель R-5: первый проход\nчитает uint16 (оценка)", [("core", TARGET_MS, False)]),
    ]
    short = ["cuFFT: перевод + R2C + Хартли", "cuFFT: колбэк LTO + Хартли", "RapiDHT сейчас: перевод + путь",
             "RapiDHT, цель R-5 (оценка)"]
    pageable_u16, pageable_f32 = b[bus_names[0]], b[bus_names[1]]
    host = [("RapiDHT сейчас: вход float с хоста", [("bus", pageable_f32, True), ("core", ours, True)])] + [
        (short[i], [("bus", pageable_u16, True)] + segs) for i, (_, segs) in enumerate(dev)
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    colours = {"bus": "#257de0", "conv": "#f47230", "core": "#00b894", "hart": "#f2a600"}
    txt, txt2, grid = "#191919", "#666666", "#e6e6e6"
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.8), gridspec_kw={"height_ratios": [4, 5]})
    device_max = max(sum(ms for _, ms, _ in segs) for _, segs in dev)
    host_max = max(sum(ms for _, ms, _ in segs) for _, segs in host)
    for ax, rows, title, limit in [
        (axes[0], dev, "Объём uint16 уже на устройстве, мс", device_max * 1.12),
        (axes[1], host, "Объём на хосте: перенос + преобразование, мс", host_max * 1.1),
    ]:
        for i, (lab, segs) in enumerate(rows):
            y = len(rows) - 1 - i
            x = 0.0
            for kind, ms, measured in segs:
                ax.barh(y, ms, left=x, height=0.56, color=colours[kind], edgecolor="white", linewidth=2,
                        hatch=None if measured else "///")
                if ms > limit * 0.12:
                    ax.text(x + ms / 2, y, f"{ms:.1f}".replace(".", ","), ha="center", va="center", color=txt,
                            fontsize=9, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
                x += ms
            ax.text(x + limit * 0.01, y, f"{x:.1f}".replace(".", ","), va="center", color=txt, fontsize=9)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", color=txt, fontsize=9)
        ax.set_xlim(0, limit)
        ax.set_yticks([])
        ax.set_title(title, loc="left", color=txt, fontsize=10.5)
        ax.xaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(txt2)
        ax.tick_params(colors=txt2)
    handles = [Patch(facecolor=colours["bus"], label="перенос с хоста, обычная память"),
               Patch(facecolor=colours["conv"], label="перевод uint16 → float"),
               Patch(facecolor=colours["core"], label="преобразование (cuFFT R2C или проходы RapiDHT)"),
               Patch(facecolor=colours["hart"], label="перевод спектра в Хартли"),
               Patch(facecolor="white", edgecolor=txt2, hatch="///", label="оценка, не замер")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.27, 0.08, 1, 1))
    out = ROOT / "docs" / "images" / "r5-budget.svg"
    fig.savefig(out, metadata={"Date": None})
    print(f"wrote {out}\n  device rows: {dev_file.name}\n  bus rows:    {bus_file.name}")
    print(f"  pinned memory, for reference: u16 {b[bus_names[2]]:.1f} ms, f32 {b[bus_names[3]]:.1f} ms")


if __name__ == "__main__":
    main()
