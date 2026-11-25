"""Draws the figures of docs/R4_PLAN.md and docs/benchmarks.md from the measurement files.

    python3 tools/r4_figures.py          # writes docs/images/r4-history.svg, r4-budget.svg, r4-scaling.svg

Why a script: the figures were drawn by hand once (2026-11-13) and then went
five steps out of date. Here the R-4 points are read from the result files of
scripts/r4_measure.sh, so redrawing after a step is one command; the points
from before R-4's protocol existed (matrices .. v2.3) are fixed numbers with
their source noted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
IMAGES = ROOT / "docs" / "images"

# The palette of the first figures, so old and new read as one series.
INK, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, GREEN, ORANGE, RED = "#2a78d6", "#1baf7a", "#eda100", "#eb6834"

# Before scripts/r4_measure.sh: 512^3 f32 times as printed in the 2026-11-13
# figure (benchmarks/results/2026-11-06_*.json, 2026-11-13_*_factored_v2.3.json).
EARLY = [
    ("матрицы\n(до R-4)", 44.2),
    ("v1\nразложение\nN = N₁·N₂", 50.0),
    ("v2\nрегистровый\nтайл 4×4", 28.6),
    ("v2.2\nзагрузка\nпачками", 25.3),
    ("v2.3\nконвейер,\nбез конфликтов", 18.0),
]

# R-4 steps: label and the "B" file of the step's measurement.
STEPS = [
    ("2.4\nширина пачки\nпо осям", "2026-11-15_evgeny_r4_s24_b2"),
    ("3.3\nБрейсуэлл\nвнутри x", "2026-11-15_evgeny_r4_s33_b"),
    ("4Б\nшаги на месте\n(гибрид)", "2026-11-15_evgeny_r4_s4bh"),
    ("4А\nповорот\nв шаге 1", "2026-11-19_evgeny_r4_s4as"),
    ("4Д\nразмеры\nконстантами", "2026-11-20_evgeny_r4_s4ds"),
    ("4.Y\nось y:\n16 линий", "2026-11-20_evgeny_r4_s4y"),
    ("4.Z\nось z:\n32 линии", "2026-11-20_evgeny_r4_s4z"),
    ("4.XG\nось x:\nадреса групп", "2026-11-25_evgeny_r4_xg_b2"),
]

# Step 5.2: the full table, every size against both cuFFT baselines.
FULL_TABLE = "2026-11-25_evgeny_r4_s52_full"
SIZES_3D = [64, 128, 256, 512, 1024]
SIZES_2D = [512, 1024, 2048]

COPY_FLOOR_MS = 2.6  # one pass over 512^3 at the speed of a copy (2.62 ms, step 0.2), as the plan rounds it


def bench_ms(stem: str, name: str) -> float:
    """Median real time of `name` (e.g. 'FactoredResident/f32/3D/512x512x512') in ms."""
    data = json.loads((RESULTS / f"{stem}.json").read_text(encoding="utf-8"))
    for b in data["benchmarks"]:
        if b.get("aggregate_name") == "median" and b["run_name"] == name:
            scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0}[b["time_unit"]]
            return b["real_time"] * scale
    raise KeyError(f"{name} not in {stem}.json")


def baseline_ms(name: str) -> float:
    """cuFFT from the latest step that measured it: sweeps may leave it out (R4_NO_CUFFT=1)."""
    for _, stem in reversed(STEPS):
        try:
            return bench_ms(stem, name)
        except KeyError:
            continue
    raise KeyError(f"{name} in none of the steps")


def trace_ms(stem: str, size: str = "n=512") -> dict[str, float]:
    """Per-axis minimum over the calls of the trace (the protocol since 2026-11-19)."""
    times: dict[str, list[float]] = {}
    for line in (RESULTS / f"{stem}_trace.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith(size + " ") and "[RapiDHT trace]" in line:
            for axis, value in re.findall(r"axis-([xyz])\(\w+\) ([0-9.]+) ms", line):
                times.setdefault(axis, []).append(float(value))
    return {axis: min(v) for axis, v in times.items()}


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def history(cufft_ms: float) -> None:
    labels = [label for label, _ in EARLY] + [label for label, _ in STEPS]
    values = [v for _, v in EARLY] + [bench_ms(stem, "FactoredResident/f32/3D/512x512x512") for _, stem in STEPS]
    fig, ax = plt.subplots(figsize=(10.2, 4.4), facecolor=PAPER)
    style(ax)
    xs = range(len(values))
    ax.plot(xs, values, color=BLUE, linewidth=2, marker="o", markersize=5, zorder=3)
    for x, v in zip(xs, values):
        ax.annotate(
            f"{v:.1f}".replace(".", ","),
            (x, v),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            color=INK,
        )
    ax.axhline(cufft_ms, color=INK, linestyle=(0, (6, 3.6)), linewidth=1)
    ax.text(
        len(EARLY) - 0.4,
        cufft_ms + 0.6,
        f"cuFFT + перевод в Хартли: {cufft_ms:.1f}".replace(".", ","),
        ha="left",
        fontsize=8,
        color=INK,
    )
    ax.axhline(3 * COPY_FLOOR_MS, color=MUTED, linestyle=(0, (1, 3)), linewidth=1)
    ax.text(
        len(EARLY) - 0.4,
        3 * COPY_FLOOR_MS - 2.2,
        "три прохода на пределе шины: 7,8",
        ha="left",
        fontsize=8,
        color=MUTED,
    )
    ax.axvline(len(EARLY) - 0.5, color=GRID, linewidth=1)
    ax.text(len(EARLY) - 0.4, 49, "план R-4 →", fontsize=8, color=MUTED)
    ax.set_xticks(list(xs), labels, fontsize=7.5, color=MUTED)
    ax.set_ylim(0, 55)
    ax.set_ylabel("время преобразования, мс", color=MUTED, fontsize=9)
    ax.set_title(
        "Факторизованный путь на GPU: 512 × 512 × 512, float, данные на устройстве", color=INK, fontsize=10, loc="left"
    )
    fig.tight_layout()
    fig.savefig(IMAGES / "r4-history.svg", facecolor=PAPER, metadata={"Date": None})
    plt.close(fig)


def budget(cufft_ms: float, r2c_ms: float) -> None:
    stem = STEPS[-1][1]
    traced = trace_ms(stem)
    # The trace runs at a locked 1665 MHz, the benchmark at boost: scale the
    # passes so that they add up to the benchmark's time.
    total = bench_ms(stem, "FactoredResident/f32/3D/512x512x512")
    scale = total / sum(traced.values())
    now = [traced["x"] * scale, traced["y"] * scale, traced["z"] * scale, 0.0]
    rows = [
        ("v2.2 (06.11)", [8.6, 6.9, 6.9, 2.6]),  # 2026-11-13 figure
        ("v2.3 (13.11)", [4.0, 5.1, 6.1, 2.6]),  # step 0.2 trace
        (STEPS[-1][0].split("\n")[0] + " (20.11)", now),
        ("цель П2", [3.9, 3.4, 3.4, 0.0]),
        ("предел шины,\nтри прохода", [COPY_FLOOR_MS, COPY_FLOOR_MS, COPY_FLOOR_MS, 0.0]),
    ]
    colors = [BLUE, GREEN, ORANGE, RED]
    names = ["ось x (с Брейсуэллом с шага 3.3)", "ось y", "ось z", "проход Брейсуэлла"]
    fig, ax = plt.subplots(figsize=(10.2, 4.6), facecolor=PAPER)
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ys = list(range(len(rows) + 1))[::-1]
    for y, (label, parts) in zip(ys, rows):
        left = 0.0
        for value, color in zip(parts, colors):
            if value > 0:
                ax.barh(y, value, left=left, color=color, edgecolor=PAPER, linewidth=2, height=0.62)
                ax.text(
                    left + value / 2,
                    y,
                    f"{value:.1f}".replace(".", ","),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#ffffff",
                )
            left += value
        ax.text(left + 0.3, y, f"{left:.1f} мс".replace(".", ","), va="center", fontsize=8, color=INK)
    y = ys[-1]
    ax.barh(y, r2c_ms, color="#b9b8b2", edgecolor=PAPER, linewidth=2, height=0.62)
    ax.barh(y, cufft_ms - r2c_ms, left=r2c_ms, color="#dcdbd5", edgecolor=PAPER, linewidth=2, height=0.62)
    ax.text(r2c_ms / 2, y, f"R2C {r2c_ms:.1f}".replace(".", ","), ha="center", va="center", fontsize=8, color=INK)
    ax.text(r2c_ms + (cufft_ms - r2c_ms) / 2, y, "перевод", ha="center", va="center", fontsize=7, color=INK)
    ax.text(cufft_ms + 0.3, y, f"{cufft_ms:.1f} мс".replace(".", ","), va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, [label for label, _ in rows] + ["cuFFT +\nперевод"], fontsize=8, color=MUTED)
    ax.set_xlim(0, 27)
    ax.set_xlabel("время, мс", color=MUTED, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, names, loc="lower right", fontsize=8, frameon=False, labelcolor=MUTED)
    ax.set_title(
        "Время по проходам: 512 × 512 × 512, float, данные на устройстве (последний шаг — трасса, приведённая к бенчмарку)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(IMAGES / "r4-budget.svg", facecolor=PAPER, metadata={"Date": None})
    plt.close(fig)


def scaling() -> None:
    """Our time over cuFFT's, per size: below 1 is faster. Two pairs, like against like:
    the true transform against cuFFT + conversion, the separable one against bare R2C."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), facecolor=PAPER, gridspec_kw={"width_ratios": [5, 3]})
    pairs = [
        ("FactoredResident", "cuFFT", "преобразование Хартли / cuFFT + перевод", BLUE, -13),
        ("FactoredResidentSeparable", "cuFFT_R2C", "сепарабельное / cuFFT R2C без перевода", ORANGE, 7),
    ]  # the last number: where the value labels go, points above the marker
    for ax, rank, sizes in ((axes[0], "3D", SIZES_3D), (axes[1], "2D", SIZES_2D)):
        style(ax)
        for ours, theirs, label, color, offset in pairs:
            xs, ys = [], []
            for i, n in enumerate(sizes):
                case = f"f32/{rank}/" + "x".join([str(n)] * (3 if rank == "3D" else 2))
                try:
                    ys.append(bench_ms(FULL_TABLE, f"{ours}/{case}") / bench_ms(FULL_TABLE, f"{theirs}/{case}"))
                    xs.append(i)
                except KeyError:  # cuFFT does not fit 1024^3 on 8 GiB
                    continue
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label, zorder=3)
            for x, v in zip(xs, ys):
                ax.annotate(f"{v:.2f}".replace(".", ","), (x, v), textcoords="offset points", xytext=(0, offset),
                            ha="center", fontsize=8, color=color)
        ax.axhline(1.0, color=INK, linestyle=(0, (6, 3.6)), linewidth=1)
        names = [f"{n}³" if rank == "3D" else f"{n}²" for n in sizes]
        ax.set_xticks(range(len(sizes)), names, fontsize=8, color=MUTED)
        ax.set_xlim(-0.5, len(sizes) - 0.5)
        ax.set_ylim(0, 3.0)
        ax.set_title("объём" if rank == "3D" else "изображение", color=INK, fontsize=9, loc="left")
        if rank == "3D":
            ax.text(len(sizes) - 1, 0.25, "cuFFT\nне помещается\nв 8 ГБ", ha="center", fontsize=7.5, color=MUTED)
    axes[0].set_ylabel("время RapiDHT / время cuFFT", color=MUTED, fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False, labelcolor=MUTED)
    fig.suptitle("Факторизованный путь против cuFFT, float, данные на устройстве (ниже 1 — быстрее cuFFT)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(IMAGES / "r4-scaling.svg", facecolor=PAPER, metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    plt.rcParams["svg.hashsalt"] = "rapidht-r4"  # stable ids: re-running without new data changes nothing
    plt.rcParams["svg.fonttype"] = "path"
    cufft = baseline_ms("cuFFT/f32/3D/512x512x512")
    r2c = baseline_ms("cuFFT_R2C/f32/3D/512x512x512")
    history(cufft)
    budget(cufft, r2c)
    scaling()
    print(f"written: {IMAGES / 'r4-history.svg'}, {IMAGES / 'r4-budget.svg'}, {IMAGES / 'r4-scaling.svg'} (cuFFT {cufft:.2f} ms)")


if __name__ == "__main__":
    main()
