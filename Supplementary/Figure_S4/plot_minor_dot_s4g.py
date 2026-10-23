"""Render Supplementary Figure S4g: full-Wu minor marker dot plot."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG4 = ROOT.parents[1] / "Figure_4"
DATA = FIG4 / "data/figure4_kl_data.npz"
OUT = ROOT / "outputs/panels"
PAGE = (11.69, 8.27)
DPI = 300
DOT_MAX_AREA = 90
DOT_POWER = 1.5
RED = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#fee0d2", "#fc9272", "#ef3b2c", "#99000d"])
MAJOR_SHORT = {"Cancer Epithelial": "Cancer epi.", "Normal Epithelial": "Normal epi.",
               "Endothelial": "Endothelial", "CAFs": "CAFs", "PVL": "PVL",
               "T-cells": "T/NK", "B-cells": "B cells", "Plasmablasts": "Plasma", "Myeloid": "Myeloid"}


def main():
    z = np.load(DATA, allow_pickle=True)
    labels = z["labels"].astype(str).tolist()
    parents = z["parents"].astype(str).tolist()
    genes = z["genes"].astype(str).tolist()
    counts = z["counts"].astype(int)
    pct = z["marker_detection"].astype(float)
    scaled = z["marker_scaled"].astype(float)
    assert len(labels) == 29 and len(genes) == 58 and pct.shape == scaled.shape == (29, 58)
    assert counts.sum() + 72 == 69_391
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.grid": False, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=PAGE, facecolor="white")
    fig.text(.018, .982, "g", fontsize=11, fontweight="bold", va="top")
    fig.text(.044, .982, "Reference-derived marker expression across all 29 full-Wu minor identities",
             fontsize=9, va="top")
    ax = fig.add_axes([.218, .155, .650, .695])
    yy, xx = np.indices((29, 58))
    ax.scatter(xx.ravel(), yy.ravel(), s=(DOT_MAX_AREA * (pct / 100) ** DOT_POWER).ravel(),
               c=scaled.ravel(), cmap=RED, norm=Normalize(0, 1), edgecolors="#777777",
               linewidths=.22, rasterized=False)
    ax.set_xlim(-.6, 57.6)
    ax.set_ylim(28.6, -.6)
    ax.set_xticks(range(58), genes, rotation=90, fontsize=5.2)
    ax.set_yticks(range(29), [f"{lab}  (n={n:,})" for lab, n in zip(labels, counts)], fontsize=7.0)
    ax.tick_params(length=0, pad=2)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    start = 0
    while start < len(labels):
        stop = start + 1
        while stop < len(labels) and parents[stop] == parents[start]:
            stop += 1
        left, right = 2 * start - .3, 2 * stop - .7
        ax.plot([left, left, right, right], [1.015, 1.045, 1.045, 1.015],
                transform=ax.get_xaxis_transform(), color="#444444", lw=.55, clip_on=False)
        ax.text((left + right) / 2, 1.052, MAJOR_SHORT[parents[start]],
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5)
        if stop < len(labels):
            ax.axhline(stop - .5, color="#dddddd", lw=.5)
        start = stop
    legend = fig.add_axes([.890, .590, .096, .210])
    legend.axis("off")
    legend.text(0, 1, "Expressing cells", fontsize=6.5, va="top")
    for i, p in enumerate((20, 40, 60, 80, 100)):
        y = .82 - i * .145
        legend.scatter([.18], [y], s=DOT_MAX_AREA * (p / 100) ** DOT_POWER,
                       color="#999999", edgecolors="#777777", linewidths=.22)
        legend.text(.42, y, f"{p}%", va="center", fontsize=6)
    legend.set_xlim(0, 1)
    legend.set_ylim(0, 1)
    cax = fig.add_axes([.900, .450, .020, .125])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=RED), cax=cax, ticks=(0, .5, 1))
    cb.ax.tick_params(labelsize=5.5, length=2, pad=2)
    fig.text(.887, .424, "Mean expression\nscaled per gene", fontsize=6.2, ha="left", va="top")
    fig.text(.218, .018,
             "Two saved Wu reference-derived markers per identity; shared genes may repeat. "
             "Dot size: cells with expression >0. Color: mean log1p(10k counts/cell), including zeros, scaled per gene.",
             fontsize=6.3, va="bottom")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"S4_g_minor_marker_dotplot.{ext}", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Rendered S4g; Python {sys.version_info.major}.{sys.version_info.minor}")


if __name__ == "__main__":
    main()
