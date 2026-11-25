"""Figures of the R-5 track (docs/R5_PLAN.md).

    python3 tools/r5_figures.py      # writes docs/images/r5-budget.svg

r5-budget: time of "uint16 -> Hartley spectrum" at 512^3, on the device and
from the host. Hatched segments are the plan's estimates; steps 0.2, 0.3 and
0.6 replace them with measurements (edit the numbers below and set measured=True).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
C = {"bus": "#257de0", "conv": "#f47230", "core": "#00b894", "hart": "#f2a600"}
TXT, TXT2, GRID = "#191919", "#666666", "#e6e6e6"

# (label, [(kind, ms, measured)])
dev = [
    ("cuFFT: перевод в float + R2C\n+ перевод в Хартли", [("conv", 1.9, False), ("core", 8.11, True), ("hart", 2.62, True)]),
    ("cuFFT: R2C с колбэком загрузки\n+ перевод в Хартли", [("core", 7.5, False), ("hart", 2.62, True)]),
    ("RapiDHT сейчас: перевод в float\n+ факторизованный путь", [("conv", 1.9, False), ("core", 9.34, True)]),
    ("RapiDHT, цель R-5: первый проход\nчитает uint16", [("core", 8.8, False)]),
]
short = ["cuFFT: перевод + R2C + Хартли", "cuFFT: колбэк + Хартли", "RapiDHT сейчас: перевод + путь", "RapiDHT, цель R-5"]
host = [("RapiDHT сейчас: вход float с хоста", [("bus", 58.0, False), ("core", 9.34, True)])] + [
    (short[i], [("bus", 29.0, False)] + segs) for i, (lab, segs) in enumerate(dev)
]

fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.6), gridspec_kw={"height_ratios": [4, 5]})
for ax, rows, title in [
    (axes[0], dev, "Объём uint16 уже на устройстве, мс"),
    (axes[1], host, "Объём на хосте: перенос + преобразование, мс"),
]:
    for i, (lab, segs) in enumerate(rows):
        y = len(rows) - 1 - i
        x = 0.0
        for kind, ms, measured in segs:
            ax.barh(y, ms, left=x, height=0.56, color=C[kind], edgecolor="white", linewidth=2,
                    hatch=None if measured else "///")
            if ms > (2.2 if ax is axes[0] else 6):
                ax.text(x + ms / 2, y, f"{ms:.1f}".replace(".", ","), ha="center", va="center",
                        color=TXT, fontsize=9, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
            x += ms
        ax.text(x + 0.4 if ax is axes[0] else x + 1.0, y, f"{x:.1f}".replace(".", ","), va="center", color=TXT, fontsize=9)
        ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", color=TXT, fontsize=9)
    ax.set_yticks([])
    ax.set_title(title, loc="left", color=TXT, fontsize=10.5)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(TXT2)
    ax.tick_params(colors=TXT2)
axes[0].set_xlim(0, 14.5)
axes[1].set_xlim(0, 74)
handles = [Patch(facecolor=C["bus"], label="перенос с хоста"),
           Patch(facecolor=C["conv"], label="перевод uint16 → float"),
           Patch(facecolor=C["core"], label="преобразование (cuFFT R2C или проходы RapiDHT)"),
           Patch(facecolor=C["hart"], label="перевод спектра в Хартли"),
           Patch(facecolor="white", edgecolor=TXT2, hatch="///", label="оценка, не замер")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.tight_layout(rect=(0.27, 0.08, 1, 1))
import pathlib
out = pathlib.Path(__file__).resolve().parents[1] / "docs" / "images" / "r5-budget.svg"
fig.savefig(out)
print(f"wrote {out}")
