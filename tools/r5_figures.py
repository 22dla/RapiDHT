"""Figures of the R-5 track (docs/R5_PLAN.md).

    python3 tools/r5_figures.py [RUN.json ...]      # writes docs/images/r5-budget.svg
    python3 tools/r5_figures.py tasks RUN.json      # writes docs/images/r5-tasks.svg (step 5.1)
    python3 tools/r5_figures.py scaling RUN.json    # writes docs/images/r5-scaling.svg (step 6.3)

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
The last bar is the library as shipped since step 6.1: the first pass reads
uint16 itself, default offset policy (IntFactoredEvery64). Until step 6.3 it
was the hatched target estimate of section 4 of the plan.

r5-tasks (step 5.1): the end-to-end tasks T1-T3 at 512^3 uint16 from one
R5_TASKS=1 run, on the device and from pinned host memory: this library (the
faster of its variants), cuFFT the strongest way (LTO callbacks) and, for the
filter at sigma 1 and 2, the spatial filter. Every bar is a measured median.

r5-scaling (step 6.3): time per voxel of uint16 -> spectrum and spectrum ->
uint16 from 128^3 to 1024^3, from one R5_LARGE=1 run: this library (integer
passes, and with a conversion kernel) against cuFFT (LTO callbacks, and with
a conversion kernel). A missing point is a variant that does not fit.
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
                 f"cuFFTInt/u16/{SIZE}", f"cuFFTIntLto/u16/{SIZE}", f"IntConvertFactored/u16/{SIZE}",
                 f"IntFactoredEvery64/u16/{SIZE}"]
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
        ("RapiDHT: перевод в float\n+ факторизованный путь", [("conv", conv_ours, True), ("core", ours, True)]),
        ("RapiDHT: первый проход\nчитает uint16", [("core", d[dev_names[6]], True)]),
    ]
    short = ["cuFFT: перевод + R2C + Хартли", "cuFFT: колбэк LTO + Хартли", "RapiDHT: перевод + путь",
             "RapiDHT: проход читает uint16"]
    pageable_u16, pageable_f32 = b[bus_names[0]], b[bus_names[1]]
    host = [("RapiDHT: вход float с хоста", [("bus", pageable_f32, True), ("core", ours, True)])] + [
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
               Patch(facecolor=colours["hart"], label="перевод спектра в Хартли")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.27, 0.08, 1, 1))
    out = ROOT / "docs" / "images" / "r5-budget.svg"
    fig.savefig(out, metadata={"Date": None})
    print(f"wrote {out}\n  device rows: {dev_file.name}\n  bus rows:    {bus_file.name}")
    print(f"  pinned memory, for reference: u16 {b[bus_names[2]]:.1f} ms, f32 {b[bus_names[3]]:.1f} ms")


def tasks(path: pathlib.Path) -> None:
    rows = medians(path)
    size = "3D/512x512x512"
    # (label, {implementation: [candidate families]}, middle of the name)
    tasks_ = [
        ("Т1 спектр", "", {"ours": ["Task1/IntFactored"], "cufft": ["Task1/cuFFTLto"]}),
        ("Т2 фильтр, σ = 1", "/s1", {"ours": ["Task2/Hartley", "Task2/HartleySeparable"], "cufft": ["Task2/cuFFTLto"],
                                    "spatial": ["Task2/Spatial"]}),
        ("Т2 фильтр, σ = 2", "/s2", {"ours": ["Task2/Hartley", "Task2/HartleySeparable"], "cufft": ["Task2/cuFFTLto"],
                                    "spatial": ["Task2/Spatial"]}),
        ("Т2 фильтр, σ = 4", "/s4", {"ours": ["Task2/Hartley", "Task2/HartleySeparable"], "cufft": ["Task2/cuFFTLto"]}),
        ("Т3 спектр мощности", "", {"ours": ["Task3/Hartley", "Task3/HartleySeparable"], "cufft": ["Task3/cuFFTLto"]}),
    ]
    impls = [("ours", "RapiDHT (быстрейший вариант)", "#00b894"), ("cufft", "cuFFT, колбэки LTO", "#257de0"),
             ("spatial", "пространственный фильтр", "#f2a600")]

    def best(families: list[str], middle: str, where: str) -> float | None:
        found = [rows[n] for f in families if (n := f"{f}/u16{middle}/{where}/{size}") in rows]
        return min(found) if found else None

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    txt, txt2, grid = "#191919", "#666666", "#e6e6e6"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    height = 0.26
    for ax, where, title in [(axes[0], "device", "Объём uint16 на устройстве, мс"),
                             (axes[1], "pinned", "С хоста, закреплённая память, мс\n(Т2 — и обратно)")]:
        limit = 0.0
        for i, (label, middle, families) in enumerate(tasks_):
            y0 = len(tasks_) - 1 - i
            present = [(key, name, colour) for key, name, colour in impls if key in families]
            for j, (key, _, colour) in enumerate(present):
                ms = best(families[key], middle, where)
                if ms is None:
                    continue
                y = y0 + (len(present) - 1) / 2 * height - j * height
                ax.barh(y, ms, height=height * 0.92, color=colour, edgecolor="white", linewidth=1)
                ax.text(ms, y, f" {ms:.1f}".replace(".", ","), va="center", color=txt, fontsize=8.5)
                limit = max(limit, ms)
            if where == "device":
                ax.text(-0.01, y0, label, transform=ax.get_yaxis_transform(), ha="right", va="center", color=txt,
                        fontsize=9.5)
        ax.set_xlim(0, limit * 1.15)
        ax.set_yticks([])
        ax.set_title(title, loc="left", color=txt, fontsize=10.5)
        ax.xaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(txt2)
        ax.tick_params(colors=txt2)
    handles = [Patch(facecolor=colour, label=name) for _, name, colour in impls]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.14, 0.07, 1, 1))
    out = ROOT / "docs" / "images" / "r5-tasks.svg"
    fig.savefig(out, metadata={"Date": None})
    print(f"wrote {out}\n  rows: {path.name}")


def scaling(path: pathlib.Path) -> None:
    rows = medians(path)
    sizes = [("128³", "128x128x128", 128**3), ("256³", "256x256x256", 256**3), ("512³", "512x512x512", 512**3),
             ("1024²×512", "1024x1024x512", 1024 * 1024 * 512), ("1024³", "1024x1024x1024", 1024**3)]
    panels = [
        ("uint16 → спектр Хартли (float)", [
            ("IntFactoredEvery64", "RapiDHT: проход читает uint16", "#00b894", "-"),
            ("IntConvertFactored", "RapiDHT: ядро перевода + путь", "#00b894", "--"),
            ("cuFFTIntLto", "cuFFT: колбэк загрузки LTO", "#257de0", "-"),
            ("cuFFTInt", "cuFFT: ядро перевода", "#257de0", "--")]),
        ("спектр → uint16 (обратное)", [
            ("IntInverse", "RapiDHT: проход пишет uint16", "#00b894", "-"),
            ("IntInverseConvert", "RapiDHT: путь + ядро перевода", "#00b894", "--"),
            ("cuFFTIntInvLto", "cuFFT: колбэк сохранения LTO", "#257de0", "-"),
            ("cuFFTIntInv", "cuFFT: ядро перевода", "#257de0", "--")]),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    txt, txt2, grid = "#191919", "#666666", "#e6e6e6"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (title, lines) in zip(axes, panels):
        for family, label, colour, style in lines:
            xs, ys = [], []
            for i, (_, name, count) in enumerate(sizes):
                ms = rows.get(f"{family}/u16/3D/{name}")
                if ms is not None:
                    xs.append(i)
                    ys.append(ms * 1e6 / count)  # ns per voxel
            ax.plot(xs, ys, style, color=colour, marker="o", markersize=4, linewidth=1.8, label=label)
        ax.set_xticks(range(len(sizes)), [s[0] for s in sizes])
        ax.set_title(title, loc="left", color=txt, fontsize=10.5)
        ax.yaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(txt2)
        ax.tick_params(colors=txt2)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("нс на воксель (меньше — лучше)", color=txt2)
    axes[0].set_ylim(bottom=0)
    fig.tight_layout()
    out = ROOT / "docs" / "images" / "r5-scaling.svg"
    fig.savefig(out, metadata={"Date": None})
    print(f"wrote {out}\n  rows: {path.name}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "tasks":
        tasks(pathlib.Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "scaling":
        scaling(pathlib.Path(sys.argv[2]))
    else:
        main()
