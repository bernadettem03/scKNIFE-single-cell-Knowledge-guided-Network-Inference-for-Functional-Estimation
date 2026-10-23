"""Render the final A4 Figure 4a–k from saved, validated figure inputs.

Panels a–d are high-resolution crops of the approved compiled artwork. The
remaining panels are redrawn from the same saved cell-level inputs; no
inferential analysis is performed here.
"""
from pathlib import Path
import subprocess
import tempfile

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image
import anndata as ad

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
SINGLER = WORKSPACE / "May_2026_clean_version_redo/inhouse_rerun/scknife_run/figures/celltype/singleR/data"
EXPRESSION = WORKSPACE / "May_2026_clean_version_redo/inhouse_rerun/primary_analysis/data/adata_final_infercnv.h5ad"
SOURCE = DATA / "Figure_4_compiled_aj_source.pdf"

# A4 portrait. All positions are millimetres from the bottom-left corner.
PAGE = (210.0, 297.0)
DPI = 300
SOURCE_DPI = 450
PANEL_LETTER_PT = 7.0
SMALL_PT = 4.3
MICRO_PT = 3.55
PURPLE = LinearSegmentedColormap.from_list("marker_purple", ["#eeeeee", "#b39ddb", "#54278f"])
MAJOR_ORDER = ["Cancer Epithelial", "Normal Epithelial", "Endothelial", "CAFs", "PVL",
               "T-cells", "B-cells", "Plasmablasts", "Myeloid"]
MAJOR_COLORS = {"Cancer Epithelial": "#ED7925", "Normal Epithelial": "#FDB366",
                "Endothelial": "#AEC7E8", "CAFs": "#8C564B", "PVL": "#009E73",
                "T-cells": "#008C95", "B-cells": "#9467BD", "Plasmablasts": "#CC79A7",
                "Myeloid": "#E377C2", "Ambiguous": "#FFF0C2"}
BOX = {"a": (6, 227, 130, 63), "b": (139, 227, 65, 63),
       "c": (6, 171, 69, 53), "d": (78, 171, 126, 53),
       "e": (6, 78, 90, 90), "f": (99, 78, 67, 90),
       "g": (169, 78, 35, 90), "h": (6, 6, 45, 69),
       "i": (54, 6, 45, 69), "j": (102, 6, 45, 69),
       "k": (150, 6, 54, 69)}
# Bounds in the frozen page rendered at 180 dpi (1336 × 1890 pixels).
# The panel letters are deliberately excluded and redrawn consistently.
OLD_CROPS = {"ab": (40, 40, 1322, 472),
             "c": (56, 501, 480, 811), "d": (495, 495, 1316, 814)}


def mm_axes(fig, box):
    x, y, w, h = box
    return fig.add_axes([x / PAGE[0], y / PAGE[1], w / PAGE[0], h / PAGE[1]])


def label(fig, letter, title=None):
    x, y, w, h = BOX[letter]
    fig.text(x / PAGE[0], (y + h) / PAGE[1], letter, fontsize=PANEL_LETTER_PT,
             fontweight="bold", va="top")
    if title:
        fig.text((x + 5) / PAGE[0], (y + h) / PAGE[1], title,
                 fontsize=SMALL_PT, va="top")


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def fit_image(fig, image, box):
    x, y, w, h = box
    ratio = image.width / image.height
    if w / h > ratio:
        actual_h = h
        actual_w = h * ratio
    else:
        actual_w = w
        actual_h = w / ratio
    ax = mm_axes(fig, (x + (w - actual_w) / 2, y + (h - actual_h) / 2,
                       actual_w, actual_h))
    ax.imshow(image, interpolation="lanczos")
    ax.axis("off")


def approved_artwork(fig):
    with tempfile.TemporaryDirectory(prefix="figure4_crops_") as td:
        stem = Path(td) / "source"
        subprocess.run(["pdftoppm", "-f", "1", "-l", "1", "-r", str(SOURCE_DPI),
                        "-png", "-singlefile", str(SOURCE), str(stem)], check=True)
        with Image.open(stem.with_suffix(".png")) as page:
            x_scale, y_scale = page.width / 1336, page.height / 1890
            for letter, coords in OLD_CROPS.items():
                x0, y0, x1, y1 = coords
                crop = page.crop((round(x0 * x_scale), round(y0 * y_scale),
                                  round(x1 * x_scale), round(y1 * y_scale)))
                if letter == "ab":
                    fit_image(fig, crop, (6, 227, 198, 63))
                else:
                    x, y, w, h = BOX[letter]
                    fit_image(fig, crop, (x + 3, y + 1, w - 3, h - 6))
                    label(fig, letter)
    # The approved source artwork counts six people in its three displayed
    # two-patient groups. P3 is drawn separately but also has Bx and surgery.
    # Replace only the outdated footer annotation; preserve the data artwork.
    footer = mm_axes(fig, (14, 230.5, 64, 8.2))
    footer.set_facecolor("white")
    footer.set_xticks([])
    footer.set_yticks([])
    for spine in footer.spines.values():
        spine.set_visible(False)
    footer.text(.27, .70, "Displayed groups", ha="center", va="center", fontsize=SMALL_PT)
    footer.text(.27, .25, "3 pairs · 6 patients", ha="center", va="center", fontsize=SMALL_PT)
    footer.text(.76, .70, "Biopsy–surgery pairs", ha="center", va="center", fontsize=SMALL_PT)
    footer.text(.76, .25, "7 patients (P1–P7)", ha="center", va="center", fontsize=SMALL_PT)


def load_data():
    with h5py.File(DATA / "fig4_plot_data.h5") as h:
        cell_ids = h["shared/cell_id"].asstr()[:]
        score = h["panel_e/cnv_score"][:]
        status = h["panel_e/cnv_status"].asstr()[:]
        subtype = h["panel_g/subtype_correlation_matrix"][:]
        subtype_names = h["panel_g/subtype_correlation_subtypes"].asstr()[:]
        patients = np.array([f"P{i}" for i in range(1, 8)])
    with h5py.File(DATA / "fig4_cd_final.h5") as h:
        xy = h["umap"][:]
    assert len(cell_ids) == len(score) == len(status) == len(xy) == 69_391
    assert subtype.shape == (7, 6)
    z = np.load(DATA / "figure4_kl_data.npz", allow_pickle=True)
    assert len(z["labels"]) == 29 and len(z["genes"]) == 58
    assert z["marker_detection"].shape == z["marker_scaled"].shape == (29, 58)
    pred = pd.read_csv(SINGLER / "singler_full_minor_predictions.csv.gz",
                       usecols=["cell_id", "pruned_label"]).set_index("cell_id")
    assert pred.index.is_unique and len(pred) == len(cell_ids)
    missing = pd.Index(cell_ids).difference(pred.index)
    assert missing.empty
    minor = pred.pruned_label.reindex(cell_ids).fillna("Ambiguous").to_numpy(dtype=str)
    assert sum(minor == "Ambiguous") == 72
    a = ad.read_h5ad(EXPRESSION, backed="r")
    try:
        genes = ["EPCAM", "DCN", "C1QA"]
        assert pd.Index(cell_ids).difference(a.obs_names).empty
        expression = a[cell_ids, genes].layers["log1p_norm"][:]
        expression = expression.toarray() if hasattr(expression, "toarray") else np.asarray(expression)
    finally:
        a.file.close()
    assert expression.shape == (69_391, 3) and np.isfinite(expression).all()
    return xy, score, status, subtype, subtype_names, patients, z, minor, expression


def minor_colors(z):
    labels, parents = z["labels"].astype(str), z["parents"].astype(str)
    palette = {"Ambiguous": MAJOR_COLORS["Ambiguous"]}
    for parent in MAJOR_ORDER:
        own = [lab for lab, major in zip(labels, parents) if major == parent]
        for i, lab in enumerate(own):
            base = np.asarray(to_rgb(MAJOR_COLORS[parent]))
            alpha = .70 + .30 * i / max(1, len(own) - 1)
            palette[lab] = to_hex(alpha * base + (1 - alpha))
    return palette


def minor_umap(fig, xy, minor, z, palette):
    label(fig, "e", "Full-Wu minor SingleR")
    ax = mm_axes(fig, (7, 97, 57, 60))
    order = np.random.default_rng(42).permutation(len(minor))
    ax.scatter(xy[order, 0], xy[order, 1], c=[palette[minor[i]] for i in order],
               s=.13, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    clean(ax)
    labels = z["labels"].astype(str).tolist()
    key = mm_axes(fig, (65, 89, 30, 70))
    key.set_xlim(0, 1)
    key.set_ylim(29.5, -.5)
    key.axis("off")
    for i, name in enumerate(labels):
        key.scatter([.05], [i], s=5.5, color=palette[name], linewidths=0)
        key.text(.13, i, name, fontsize=3.65, va="center")
    fig.text(7 / PAGE[0], 88 / PAGE[1], "H-derived UMAP · 69,391 cells", fontsize=SMALL_PT)
    fig.text(7 / PAGE[0], 83 / PAGE[1], "29 identities; 72 ambiguous calls", fontsize=SMALL_PT)


def cnv_umap(fig, xy, score):
    label(fig, "f", "CNV score")
    ax = mm_axes(fig, (100, 92, 56, 64))
    lo, hi = np.quantile(score, [.02, .98])
    im = ax.scatter(xy[:, 0], xy[:, 1], c=score, cmap="viridis", vmin=lo, vmax=hi,
                    s=.07, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    clean(ax)
    cbax = mm_axes(fig, (157.5, 96, 1.9, 50))
    cb = fig.colorbar(im, cax=cbax)
    cb.set_ticks(np.linspace(lo, hi, 4))
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    cb.ax.tick_params(labelsize=SMALL_PT, length=1, pad=.5)


def cnv_violin(fig, score, status):
    label(fig, "g", "CNV status")
    ax = mm_axes(fig, (172, 94, 29, 61))
    parts = ax.violinplot([score[status == "normal"], score[status == "tumor"]],
                          positions=(0, 1), widths=.7, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], ("#56B4E9", "#D55E00")):
        body.set_facecolor(color)
        body.set_edgecolor("#444444")
        body.set_linewidth(.2)
        body.set_alpha(.9)
    parts["cmedians"].set_color("#222222")
    parts["cmedians"].set_linewidth(.5)
    ax.set_xticks((0, 1), ("Normal", "Tumor"), fontsize=SMALL_PT)
    ax.tick_params(axis="y", labelsize=SMALL_PT, length=1, pad=.5)
    ax.spines[["top", "right"]].set_visible(False)


def gene_umap(fig, letter, xy, values, gene):
    label(fig, letter, gene)
    x, y, w, h = BOX[letter]
    ax = mm_axes(fig, (x + 1, y + 7, w - 2, h - 15))
    ax.scatter(xy[:, 0], xy[:, 1], c="#dddddd", s=.055, linewidths=0,
               rasterized=True)
    positive = np.flatnonzero(values > 0)
    positive = positive[np.argsort(values[positive], kind="stable")]
    vmax = max(float(np.quantile(values, .98)), 1e-6)
    ax.scatter(xy[positive, 0], xy[positive, 1], c=values[positive],
               cmap=PURPLE, vmin=0, vmax=vmax, s=.08, linewidths=0,
               rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    clean(ax)


def subtype_heatmap(fig, matrix, names, patients):
    label(fig, "k", "TNBC subtype")
    ax = mm_axes(fig, (157, 13, 40, 51))
    limit = np.ceil(np.abs(matrix).max() * 10) / 10
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit,
                   interpolation="none", aspect="auto")
    ax.set_xticks(range(6), names, fontsize=MICRO_PT)
    ax.xaxis.tick_top()
    ax.set_yticks(range(7), patients, fontsize=MICRO_PT)
    ax.tick_params(length=0, pad=.5)
    for i in range(7):
        for j in range(6):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=SMALL_PT, color="white" if abs(matrix[i, j]) > .30 else "black")
    for spine in ax.spines.values():
        spine.set_linewidth(.25)
    cbax = mm_axes(fig, (199, 16, 1.7, 44))
    cb = fig.colorbar(im, cax=cbax)
    cb.ax.tick_params(labelsize=SMALL_PT, length=1, pad=.5)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.grid": False, "savefig.facecolor": "white"})
    xy, score, status, subtype, names, patients, z, minor, expression = load_data()
    fig = plt.figure(figsize=(PAGE[0] / 25.4, PAGE[1] / 25.4), facecolor="white")
    approved_artwork(fig)
    palette = minor_colors(z)
    minor_umap(fig, xy, minor, z, palette)
    cnv_umap(fig, xy, score)
    cnv_violin(fig, score, status)
    for i, (letter, gene) in enumerate(zip("hij", ("EPCAM", "DCN", "C1QA"))):
        gene_umap(fig, letter, xy, expression[:, i], gene)
    subtype_heatmap(fig, subtype, names, patients)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"Figure_4_main_panel.{ext}", dpi=DPI, facecolor="white")
    plt.close(fig)
    print("Rendered one-page A4 Figure 4a–k")


if __name__ == "__main__":
    main()
