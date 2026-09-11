#!/usr/bin/env python3
"""Plotting-only renderers and exporters for manuscript Figure 4."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import tempfile
import warnings
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = MODULE_ROOT / "data"
OUTPUT_ROOT = MODULE_ROOT / "outputs"
PANEL_OUTPUT_ROOT = OUTPUT_ROOT / "panels"
PLOT_DATA = DATA_ROOT / "fig4_plot_data.h5"
PATIENT_MAPPING_DATA = DATA_ROOT / "patient_id_mapping.csv"
DATA_MANIFEST_PATH = DATA_ROOT / "manifest.json"
PANEL_LABELS = tuple("BCDEF")
OUTPUT_DPI = 300
EXPECTED_CELLS = 69_391

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
})
warnings.filterwarnings("ignore", category=FutureWarning)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_figure4_inputs() -> dict:
    """Validate the private plot bundle and public lineage declarations."""
    for path in (PLOT_DATA, PATIENT_MAPPING_DATA, DATA_MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Figure 4 input: {path.relative_to(MODULE_ROOT)}")
    manifest = json.loads(DATA_MANIFEST_PATH.read_text())
    if manifest.get("format_version") != 2 or set(manifest.get("panels", {})) != set(PANEL_LABELS):
        raise RuntimeError("Figure 4 data manifest must declare exactly panels B-F using format version 2")
    mapping = list(csv.DictReader(PATIENT_MAPPING_DATA.open(newline="")))
    if [row.get("display_id") for row in mapping] != [f"P{i}" for i in range(1, 8)]:
        raise RuntimeError("Invalid Figure 4 public patient display mapping")
    required = {"shared", "panel_a", "panel_b", "panel_c", "panel_d", "panel_e", "panel_f", "panel_g"}
    with h5py.File(PLOT_DATA, "r") as bundle:
        missing = sorted(required.difference(bundle.keys()))
        cells = len(bundle["shared/cell_id"])
        umap_shape = tuple(bundle["shared/umap"].shape)
        schema = str(bundle.attrs.get("schema_version", "unknown"))
    if missing or cells != EXPECTED_CELLS or umap_shape != (EXPECTED_CELLS, 2) or schema != "2.1":
        raise RuntimeError(f"Invalid Figure 4 plot bundle: missing={missing}, cells={cells}, umap={umap_shape}, schema={schema}")
    return {
        "schema_version": schema,
        "cells": cells,
        "panels": list(PANEL_LABELS),
        "rendering": "HDF5 + NumPy/Pandas + Matplotlib; no analysis or preprocessing",
        "python": platform.python_version(),
        "matplotlib": mpl.__version__,
    }

def draw_cnv_status_violin(ax, values, positions, labels, colors, tick_size, text_size, max_score, show_note=True):
    violin = ax.violinplot(values, positions=positions, widths=0.58, showmeans=False, showmedians=True, showextrema=True, bw_method=0.20)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color); body.set_edgecolor("#333333"); body.set_alpha(0.78); body.set_linewidth(0.65)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        violin[key].set_color("#222222"); violin[key].set_linewidth(0.70)
    margin = max(0.35, 0.22 * (positions[-1] - positions[0]))
    ax.set_xlim(positions[0] - margin, positions[-1] + margin)
    ax.set_ylim(-0.0002, float(max_score) * 1.08)
    ax.set_xticks(positions, labels, fontsize=tick_size)
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color); tick.set_fontweight("bold")
    for position, group in zip(positions, values):
        ax.text(position, float(max_score) * 1.035, f"n={len(group):,}", ha="center", va="top", fontsize=tick_size)
    ax.set_xlabel("Final CNV-cluster-derived status", fontsize=text_size)
    ax.set_ylabel("CNV cluster score", fontsize=text_size)
    if show_note:
        ax.text(0.5, 0.02, "Scores are cluster-derived", transform=ax.transAxes, ha="center", va="bottom", fontsize=tick_size)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(False)
    return violin

def make_figure4_main_panel(panel_label=None):
    # Exporting the final outcomes: mixed raster/vector Figure 4.
    # Run this section after the validated main compilation cell above.

    from pathlib import Path
    import json, platform
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np

    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "pdf.use14corefonts": False})

    import csv, h5py, tempfile

    FIG4_DPI = 300
    FIG4_LABEL_PT, FIG4_TEXT_PT, FIG4_TICK_PT = 8.5, 6.2, 4.8
    FIG4_UMAP_TITLE_PT, FIG4_POINT_SIZE, FIG4_ALPHA = 4.0, 0.22, 0.80
    FIG4_SOURCE_RECTS = {"C": (12, 229, 70, 55), "F": (86, 175, 112, 109), "E": (12, 175, 70, 48), "D": (12, 95, 144, 72), "B": (160, 95, 38, 72)}
    if panel_label is None:
        FIG4_W_MM, FIG4_H_MM = 210, 297
        FIG4_LAYOUT_RECTS = FIG4_SOURCE_RECTS
    elif panel_label in FIG4_SOURCE_RECTS:
        _, _, FIG4_W_MM, FIG4_H_MM = FIG4_SOURCE_RECTS[panel_label]
        FIG4_LAYOUT_RECTS = {panel_label: (0, 0, FIG4_W_MM, FIG4_H_MM)}
    else:
        raise ValueError(f"Unknown Figure 4 panel: {panel_label}")
    FIG4_RECTS = FIG4_SOURCE_RECTS

    mix_root = MODULE_ROOT
    data_path = mix_root / "data/fig4_plot_data.h5"
    patient_mapping_path = mix_root / "data/patient_id_mapping.csv"
    with patient_mapping_path.open(newline="") as handle:
        patient_display = {row["patient_id"]: row["display_id"] for row in csv.DictReader(handle)}
    mix_patient_order = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
    if [patient_display.get(p) for p in mix_patient_order] != [f"P{i}" for i in range(1, 8)]:
        raise RuntimeError("Invalid local Figure 4 patient display mapping")

    def rs(x):
        return np.asarray(x.asstr()[:], dtype=str)

    with h5py.File(data_path, "r") as f:
        schema = str(f.attrs.get("schema_version", "unknown"))
        u = np.asarray(f["shared/umap"][:], np.float32)
        ct = rs(f["panel_a/celltype"])
        hm = np.asarray(f["panel_c/heatmap"][:], np.float32)
        hg = rs(f["panel_c/h_cluster"])
        bd = np.asarray(f["panel_c/boundaries"][:], np.int64)
        mg = rs(f["panel_c/marker_groups"])
        tracks = {"Patient": rs(f["panel_c/patient"]), "Site": rs(f["panel_c/site"]), "Metastatic": rs(f["panel_c/metastatic"]), "Library": rs(f["panel_c/library"]), "scKNIFE DE cell type": rs(f["panel_c/scknife_final_celltype"])}
        cnv_status = rs(f["panel_e/cnv_status"])
        cnv_score = np.asarray(f["panel_e/cnv_score"][:], np.float32)
        marker_expression = np.asarray(f["panel_f/marker_expression"][:], np.float32)
        marker_genes = rs(f["panel_f/marker_genes"]).tolist()
        marker_h_clusters = rs(f["panel_f/marker_h_clusters"]).tolist()
        marker_labels = rs(f["panel_f/marker_labels"]).tolist()
        subtype_patient = rs(f["panel_g/patient"]).tolist()
        subtype_call = rs(f["panel_g/subtype"]).tolist()
        subtype_correlation_matrix = np.asarray(f["panel_g/subtype_correlation_matrix"][:], np.float32)
        subtype_score_subtypes = rs(f["panel_g/subtype_correlation_subtypes"]).tolist()

    # Panel E uses the same H-graph-derived UMAP as the H-cluster panels.
    mix_h_xy = u
    mix_h_score = cnv_score
    if mix_h_xy.shape != (len(mix_h_score), 2) or not np.isfinite(mix_h_xy).all() or not np.isfinite(mix_h_score).all():
        raise RuntimeError("Invalid H-graph CNV-score UMAP inputs")

    if schema != "2.1" or u.shape != (69391, 2) or hm.shape != (120, 69391) or marker_expression.shape != (69391, 8):
        raise RuntimeError("Invalid local Figure 4 plotting package")
    if marker_genes != ["KRT7", "IL2RG", "PDGFRB", "ELF3", "EGFL7", "KRT5", "KRT14", "MS4A6A"] or marker_h_clusters != ["H7", "H2", "H1", "H6", "H9", "H11", "H11", "H4"]:
        raise RuntimeError("Marker identity/order changed")
    if subtype_correlation_matrix.shape != (7, 6) or subtype_score_subtypes != ["IM", "BL1", "MSL", "M", "BL2", "LAR"] or not np.isfinite(subtype_correlation_matrix).all():
        raise RuntimeError("Invalid TNBCtype subtype-correlation inputs")
    if set(cnv_status) != {"normal", "tumor"} or not np.isfinite(cnv_score).all():
        raise RuntimeError("Invalid local CNV inputs")
    if not np.array_equal(bd, np.flatnonzero(hg[1:] != hg[:-1]) + 1):
        raise RuntimeError("Invalid H-cluster boundaries")
    mix_e_order = ["normal", "tumor"]
    mix_e_labels = ["Normal", "Tumor"]
    mix_e_colors = ["#56B4E9", "#D55E00"]
    mix_e_values = [cnv_score[cnv_status == status] for status in mix_e_order]
    if any(len(values) < 2 for values in mix_e_values): raise RuntimeError("Insufficient CNV-score values for error bars")
    mix_e_means = np.array([np.mean(values) for values in mix_e_values], dtype=np.float64)
    mix_e_sem = np.array([np.std(values, ddof=1) / np.sqrt(len(values)) for values in mix_e_values], dtype=np.float64)


    ctc = {"IFN-response epithelial":"#F57F20", "Epithelial cycling":"#F57F20", "Epithelial":"#F57F20", "Epithelial basal":"#F57F20", "Epithelial luminal":"#F57F20", "Epithelial stress/EMT":"#F57F20", "Epithelial progenitor":"#F57F20", "CAFs":"#1878B6", "Endothelial":"#6BAED6", "T cells":"#9DD189", "Macrophages":"#FCBB78"}
    hlist = [f"H{i}" for i in range(12)]
    hcolors = ["#009E73", "#9A42E8", "#E69F00", "#21B6C7", "#D62728", "#8C564B", "#AEC7E8", "#D66DB3", "#B6BD61", "#FFBB78", "#FA8F8F", "#2B83BA"]
    specs = [("Patient", mix_patient_order, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]), ("Site", ["Bx", "Surgery", "Recurrence"], ["#377EB8", "#E41A1C", "#984EA3"]), ("Metastatic", ["Non-metastatic", "Metastatic"], ["#4DAF4A", "#E41A1C"]), ("Library", ["Library1", "Library2", "Library3", "Library4"], ["#1F78B4", "#FFB000", "#33A02C", "#E31A1C"]), ("scKNIFE DE cell type", ["IFN-response epithelial", "Epithelial cycling", "Epithelial progenitor", "Epithelial basal", "Epithelial luminal", "CAFs", "Endothelial", "T cells", "Macrophages"], ["#E41A1C", "#FF7F00", "#33A02C", "#1F78B4", "#6A3D9A", "#A65628", "#17BECF", "#66A61E", "#E7298A"])]

    def mm_box(x, y, w, h):
        return [x / FIG4_W_MM, y / FIG4_H_MM, w / FIG4_W_MM, h / FIG4_H_MM]

    out = mix_root / "outputs"
    MIX_DPI = FIG4_DPI
    mix_out = out
    mix_tempdir = tempfile.TemporaryDirectory(prefix="figure_4_mixed_")
    mix_assets = Path(mix_tempdir.name)

    def mix_inside(rect, left, bottom, width, height):
        x, y, rw, rh = rect
        return mm_box(x + left * rw, y + bottom * rh, width * rw, height * rh)

    def mix_heading(fig, key, title, size=FIG4_LABEL_PT):
        rect = FIG4_LAYOUT_RECTS[key]
        fig.text((rect[0] + .8) / FIG4_W_MM, (rect[1] + rect[3] - 1) / FIG4_H_MM, f"{key}  {title}", ha="left", va="top", fontsize=size, fontweight="bold")
        return rect

    def mix_clean_axis(ax):
        ax.grid(False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(.4); ax.spines["bottom"].set_linewidth(.4)
        ax.tick_params(labelsize=FIG4_TICK_PT, width=.4, length=1.5, pad=1)

    def mix_vector_colorbar(fig, rect, cmap, vmin, vmax, label="", ticks=None):
        cax = fig.add_axes(rect)
        lo, hi = float(vmin), float(vmax)
        if not hi > lo: hi = lo + 1e-6
        edges = np.linspace(lo, hi, 129)
        for start, end in zip(edges[:-1], edges[1:]):
            cax.add_patch(mpl.patches.Rectangle((0, start), 1, end - start, facecolor=cmap((start - lo) / (hi - lo)), edgecolor="none"))
        cax.set_xlim(0, 1); cax.set_ylim(lo, hi); cax.set_xticks([])
        cax.set_yticks(np.linspace(lo, hi, 5) if ticks is None else ticks)
        cax.tick_params(labelsize=FIG4_TICK_PT, length=1, pad=1, width=.3)
        for spine in cax.spines.values():
            spine.set_visible(True); spine.set_linewidth(.25); spine.set_color("black")
        if label: cax.set_ylabel(label, fontsize=FIG4_TICK_PT, labelpad=1)

    def mix_save_umap(path, width_mm, height_mm, *, colors=None, values=None, vmin=None, vmax=None, xlim=None, ylim=None):
        image_fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=MIX_DPI, facecolor="white")
        image_ax = image_fig.add_axes([0, 0, 1, 1])
        if values is None:
            image_ax.scatter(u[:, 0], u[:, 1], s=FIG4_POINT_SIZE, alpha=FIG4_ALPHA, color=colors, linewidths=0)
        else:
            image_ax.scatter(u[:, 0], u[:, 1], c=values, cmap=mix_marker_cmap, vmin=vmin, vmax=vmax, s=FIG4_POINT_SIZE, alpha=FIG4_ALPHA, linewidths=0)
        image_ax.set_xlim(xlim); image_ax.set_ylim(ylim); image_ax.set_axis_off()
        image_fig.savefig(path, format="png", dpi=MIX_DPI, facecolor="white", edgecolor="white", pad_inches=0)
        plt.close(image_fig)

    def mix_save_score_umap(path, width_mm, height_mm):
        image_fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=MIX_DPI, facecolor="white")
        image_ax = image_fig.add_axes([0, 0, 1, 1])
        image_ax.scatter(mix_h_xy[:, 0], mix_h_xy[:, 1], c=mix_h_score, cmap=mix_h_cmap, vmin=mix_h_vmin, vmax=mix_h_vmax, s=FIG4_POINT_SIZE, alpha=FIG4_ALPHA, linewidths=0)
        image_ax.set_xlim(mix_h_xlim); image_ax.set_ylim(mix_h_ylim); image_ax.set_axis_off()
        image_fig.savefig(path, format="png", dpi=MIX_DPI, facecolor="white", edgecolor="white", pad_inches=0)
        plt.close(image_fig)

    mix_umap_extent_x, mix_umap_extent_y = np.ptp(u[:, 0]), np.ptp(u[:, 1])
    mix_c_xlim = (u[:, 0].min() - .05 * mix_umap_extent_x, u[:, 0].max() + .05 * mix_umap_extent_x)
    mix_c_ylim = (u[:, 1].min() - .05 * mix_umap_extent_y, u[:, 1].max() + .05 * mix_umap_extent_y)
    mix_d_xlim = (u[:, 0].min() - .02 * mix_umap_extent_x, u[:, 0].max() + .02 * mix_umap_extent_x)
    mix_d_ylim = (u[:, 1].min() - .02 * mix_umap_extent_y, u[:, 1].max() + .02 * mix_umap_extent_y)
    mix_marker_cmap = mpl.colors.LinearSegmentedColormap.from_list("mixed_marker_grey_to_purple", ["#E6E6E6", "#B39DDB", "#54278F"])

    mix_h_center = np.mean(mix_h_xy, axis=0)
    mix_h_half = max(np.ptp(mix_h_xy[:, 0]), np.ptp(mix_h_xy[:, 1])) * .53
    mix_h_xlim = (mix_h_center[0] - mix_h_half, mix_h_center[0] + mix_h_half)
    mix_h_ylim = (mix_h_center[1] - mix_h_half, mix_h_center[1] + mix_h_half)
    mix_h_vmin, mix_h_vmax = np.quantile(mix_h_score, [.02, .98])
    mix_h_cmap = mpl.colormaps["viridis"]
    mix_h_png = mix_assets / "Figure_4_panel_E_CNV_score_UMAP.png"
    mix_save_score_umap(mix_h_png, FIG4_SOURCE_RECTS["E"][2] * .56, FIG4_RECTS["E"][3] * .82)

    mix_c_png = mix_assets / "Figure_4_panel_C_UMAP.png"
    mix_save_umap(mix_c_png, FIG4_SOURCE_RECTS["C"][2] * .60, FIG4_RECTS["C"][3] * .74, colors=np.array([ctc.get(label, "#999999") for label in ct], dtype=object), xlim=mix_c_xlim, ylim=mix_c_ylim)
    mix_d_pngs = []
    for gene, values in zip(marker_genes, marker_expression.T):
        mix_vmin, mix_vmax = np.percentile(values, [2, 98])
        if not mix_vmax > mix_vmin: mix_vmax = mix_vmin + 1e-6
        mix_d_png = mix_assets / f"Figure_4_panel_D_{gene}_UMAP.png"
        mix_save_umap(mix_d_png, FIG4_SOURCE_RECTS["D"][2] * .150, FIG4_RECTS["D"][3] * .310, values=values, vmin=mix_vmin, vmax=mix_vmax, xlim=mix_d_xlim, ylim=mix_d_ylim)
        mix_d_pngs.append(mix_d_png)

    fig = plt.figure(figsize=(FIG4_W_MM / 25.4, FIG4_H_MM / 25.4), facecolor="white")

    if panel_label in (None, "C"):
        # C: PNG UMAP artwork with vector legend.
        rect = mix_heading(fig, "C", "DE-based final cell type")
        ax = fig.add_axes(mix_inside(rect, .04, .08, .60, .82))
        ax.imshow(imread(mix_c_png), aspect="auto", interpolation="none", extent=[mix_c_xlim[0], mix_c_xlim[1], mix_c_ylim[0], mix_c_ylim[1]], origin="upper")
        ax.set_xlim(mix_c_xlim); ax.set_ylim(mix_c_ylim); ax.set_xticks([]); ax.set_yticks([])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for spine in (ax.spines["left"], ax.spines["bottom"]): spine.set_visible(True); spine.set_linewidth(.45); spine.set_color("black")
        c_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgewidth=0, markersize=4, label=label) for label, color in ctc.items()]
        c_legend_ax = fig.add_axes(mix_inside(rect, .68, .08, .30, .82))
        c_legend_ax.set_axis_off()
        c_legend_ax.legend(handles=c_handles, frameon=False, loc="upper left", ncol=1, fontsize=FIG4_TICK_PT, handlelength=.7, handletextpad=.25, columnspacing=.30, labelspacing=.10, borderaxespad=0)

    if panel_label in (None, "F"):
        # F: dense heatmap and metadata strips remain rasterized by design.
        rect = mix_heading(fig, "F", "All-cell H-cluster marker expression")
        left, bottom, width, height = .13, .25, .73, .57
        mix_de_order = ["IFN-response epithelial", "Epithelial cycling", "Epithelial progenitor", "Epithelial basal", "Epithelial luminal", "CAFs", "Endothelial", "T cells", "Macrophages"]
        mix_de_colors = ["#E41A1C", "#FF7F00", "#33A02C", "#1F78B4", "#6A3D9A", "#A65628", "#17BECF", "#66A61E", "#E7298A"]
        mix_de_color_by_label = dict(zip(mix_de_order, mix_de_colors))
        if set(tracks["scKNIFE DE cell type"]) != set(mix_de_order):
            raise RuntimeError("scKNIFE DE cell-type categories do not match the declared legend")
        for row, (label, cats, colors) in enumerate(specs):
            ax = fig.add_axes(mix_inside(rect, left, bottom + height + .018 + (len(specs) - 1 - row) * .026, width, .021))
            if label == "scKNIFE DE cell type":
                cats, colors = mix_de_order, mix_de_colors
            code = {value: index for index, value in enumerate(cats)}
            ax.imshow(np.array([code[value] for value in tracks[label]])[None, :], aspect="auto", interpolation="nearest", cmap=mpl.colors.ListedColormap(colors), vmin=0, vmax=max(1, len(colors) - 1), rasterized=True)
            for boundary in bd: ax.axvline(boundary - .5, color="black", linewidth=.20)
            ax.set_xticks([]); ax.set_yticks([]); ax.text(-.008, .5, label, transform=ax.transAxes, ha="right", va="center", fontsize=FIG4_TICK_PT); ax.set_frame_on(False)
        hcode = {value: index for index, value in enumerate(hlist)}
        ax = fig.add_axes(mix_inside(rect, left, bottom + height + .018, width, .021))
        ax.imshow(np.array([hcode[value] for value in hg])[None, :], aspect="auto", cmap=mpl.colors.ListedColormap(hcolors), vmin=0, vmax=11, rasterized=True)
        for boundary in bd: ax.axvline(boundary - .5, color="black", linewidth=.25)
        ax.set_xticks([]); ax.set_yticks([]); ax.text(-.008, .5, "H cluster", transform=ax.transAxes, ha="right", va="center", fontsize=FIG4_TICK_PT); ax.set_frame_on(False)
        ax = fig.add_axes(mix_inside(rect, left, bottom, width, height))
        ax.imshow(hm, aspect="auto", interpolation="nearest", cmap="bwr", vmin=-3, vmax=3, rasterized=True)
        for boundary in bd: ax.axvline(boundary - .5, color="black", linewidth=.30)
        for boundary in range(10, 120, 10): ax.axhline(boundary - .5, color="black", linewidth=.25)
        ax.set_xticks([]); ax.set_yticks([]); ax.text(-.008, .5, "Top 10 ranked genes per H cluster", transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=FIG4_TICK_PT)
        source_ax = fig.add_axes(mix_inside(rect, .865, bottom, .012, height))
        source_ax.imshow(np.array([hcode[value] for value in mg])[:, None], aspect="auto", cmap=mpl.colors.ListedColormap(hcolors), vmin=0, vmax=11, rasterized=True); source_ax.set_axis_off()
        mix_vector_colorbar(fig, mix_inside(rect, .886, bottom + .12, .017, height * .60), mpl.colormaps["bwr"], -3, 3, label="Scaled expression")
        legend_ax = fig.add_axes(mix_inside(rect, .13, .01, .77, .20)); legend_ax.axis("off")
        patient_handles = [Patch(facecolor=color, edgecolor="none", label=patient_display[label]) for label, color in zip(specs[0][1], specs[0][2])]
        annotation_handles = [Patch(facecolor=color, edgecolor="none", label=label) for _, cats, colors in specs[1:4] for label, color in zip(cats, colors)]
        de_handles = [Patch(facecolor=mix_de_color_by_label[label], edgecolor="none", label=label) for label in mix_de_order]
        legend = legend_ax.legend(handles=patient_handles, title="Patient", loc="upper left", bbox_to_anchor=(0, 1), ncol=2, frameon=False, fontsize=FIG4_TICK_PT, title_fontsize=FIG4_TICK_PT, handlelength=.9, columnspacing=.50, labelspacing=.18, borderaxespad=0); legend_ax.add_artist(legend)
        legend = legend_ax.legend(handles=annotation_handles, title="Sample annotations", loc="upper left", bbox_to_anchor=(.31, 1), ncol=2, frameon=False, fontsize=FIG4_TICK_PT, title_fontsize=FIG4_TICK_PT, handlelength=.9, columnspacing=.42, labelspacing=.18, borderaxespad=0); legend_ax.add_artist(legend)
        legend_ax.legend(handles=de_handles, title="scKNIFE DE cell type", loc="upper left", bbox_to_anchor=(.66, 1), ncol=1, frameon=False, fontsize=FIG4_TICK_PT, title_fontsize=FIG4_TICK_PT, handlelength=.9, labelspacing=.18, borderaxespad=0)

    if panel_label in (None, "D"):
        # D: PNG UMAP artwork with vector titles and colour bars.
        rect = mix_heading(fig, "D", "Canonical H-cluster marker expression")
        mix_positions = [(.020, .535), (.262, .535), (.504, .535), (.746, .535), (.020, .095), (.262, .095), (.504, .095), (.746, .095)]
        for gene, cluster, label, values, png_path, loc in zip(marker_genes, marker_h_clusters, marker_labels, marker_expression.T, mix_d_pngs, mix_positions):
            ax = fig.add_axes(mix_inside(rect, loc[0], loc[1], .150, .310))
            ax.imshow(imread(png_path), aspect="auto", interpolation="none", extent=[mix_d_xlim[0], mix_d_xlim[1], mix_d_ylim[0], mix_d_ylim[1]], origin="upper")
            ax.set_title(f"{gene}\n{label} ({cluster})", fontsize=FIG4_UMAP_TITLE_PT, fontweight="bold", pad=1)
            ax.set_xlim(mix_d_xlim); ax.set_ylim(mix_d_ylim); ax.set_xticks([]); ax.set_yticks([])
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            for spine in (ax.spines["left"], ax.spines["bottom"]): spine.set_visible(True); spine.set_linewidth(.45); spine.set_color("black")
            mix_vmin, mix_vmax = np.percentile(values, [2, 98])
            if not mix_vmax > mix_vmin: mix_vmax = mix_vmin + 1e-6
            mix_vector_colorbar(fig, mix_inside(rect, loc[0] + .161, loc[1] + .032, .008, .235), mix_marker_cmap, mix_vmin, mix_vmax, ticks=np.linspace(mix_vmin, mix_vmax, 4))

    if panel_label in (None, "E"):
        # E: H-graph CNV-score UMAP plus a descriptive status summary.
        rect = mix_heading(fig, "E", "CNV score")
        ax_umap = fig.add_axes(mix_inside(rect, .04, .11, .48, .74))
        ax_umap.imshow(imread(mix_h_png), aspect="auto", interpolation="none", extent=[mix_h_xlim[0], mix_h_xlim[1], mix_h_ylim[0], mix_h_ylim[1]], origin="upper")
        ax_umap.set_xlim(mix_h_xlim); ax_umap.set_ylim(mix_h_ylim)
        ax_umap.set_xticks([]); ax_umap.set_yticks([]); ax_umap.set_xlabel("UMAP1", fontsize=FIG4_TEXT_PT, labelpad=1); ax_umap.set_ylabel("UMAP2", fontsize=FIG4_TEXT_PT, labelpad=1)
        ax_umap.spines["top"].set_visible(False); ax_umap.spines["right"].set_visible(False)
        for spine in (ax_umap.spines["left"], ax_umap.spines["bottom"]): spine.set_visible(True); spine.set_linewidth(.45); spine.set_color("black")
        mix_vector_colorbar(fig, mix_inside(rect, .56, .18, .035, .60), mix_h_cmap, mix_h_vmin, mix_h_vmax, label="CNV score")
        ax_violin = fig.add_axes(mix_inside(rect, .68, .14, .29, .68))
        draw_cnv_status_violin(
            ax_violin, mix_e_values, [0, 1], mix_e_labels, mix_e_colors,
            FIG4_TICK_PT, FIG4_TICK_PT, cnv_score.max(), show_note=False,
        )
        ax_violin.set_xlabel("")


    if panel_label in (None, "B"):
        # B: vector patient-level subtype-correlation heatmap.
        rect = mix_heading(fig, "B", "TNBC subtype", size=FIG4_TEXT_PT)
        subtype_map = {patient: call for patient, call in zip(subtype_patient, subtype_call)}
        correlation_limit = np.ceil(np.max(np.abs(subtype_correlation_matrix)) * 10) / 10
        score_cmap = mpl.colormaps["RdBu_r"]; score_norm = mpl.colors.Normalize(vmin=-correlation_limit, vmax=correlation_limit)
        ax = fig.add_axes(mix_inside(rect, .07, .13, .78, .74))
        for ypos, patient in enumerate(mix_patient_order):
            for xpos, score in enumerate(subtype_correlation_matrix[ypos]):
                ax.add_patch(mpl.patches.Rectangle((xpos - .5, ypos - .5), 1, 1, facecolor=score_cmap(score_norm(score)), edgecolor="none"))
                ax.text(xpos, ypos, f"{score:.2f}", ha="center", va="center", fontsize=FIG4_TICK_PT, color="white" if abs(score) > .30 else "black")
        ax.set_xlim(-.5, len(subtype_score_subtypes) - .5); ax.set_ylim(len(mix_patient_order) - .5, -.5)
        ax.set_xticks(np.arange(len(subtype_score_subtypes)), subtype_score_subtypes, fontsize=FIG4_TICK_PT); ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=1)
        ax.set_yticks(np.arange(len(mix_patient_order)), [patient_display[p] for p in mix_patient_order], fontsize=FIG4_TICK_PT); ax.tick_params(axis="y", length=0, pad=1); ax.set_xlabel(""); ax.set_ylabel("")
        for spine in ax.spines.values(): spine.set_visible(True); spine.set_color("black"); spine.set_linewidth(.5)
        mix_vector_colorbar(fig, mix_inside(rect, .89, .13, .035, .74), score_cmap, -correlation_limit, correlation_limit, label="Correlation to subtype centroid")
        ax.text(.5, -.22, "Outlined cell: assigned subtype; positive correlation = stronger support", transform=ax.transAxes, ha="center", va="top", fontsize=FIG4_TICK_PT)

    mix_tempdir.cleanup()
    return fig


def export_figure4_panel(figure, label: str) -> dict:
    """Export one authoritative main panel as 300-DPI PNG and vector PDF."""
    if label not in PANEL_LABELS:
        raise ValueError(f"Unknown Figure 4 panel: {label}")
    PANEL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix in ("png", "pdf"):
        path = PANEL_OUTPUT_ROOT / f"Figure_4_{label}.{suffix}"
        figure.savefig(path, format=suffix, dpi=OUTPUT_DPI, facecolor="white", bbox_inches="tight", pad_inches=0.04)
        paths[suffix] = path
    return paths


def write_panel_export_manifest() -> Path:
    """Write the complete portable checksum manifest for panels B-F."""
    panels = {}
    for label in PANEL_LABELS:
        formats = {}
        for suffix in ("png", "pdf"):
            path = PANEL_OUTPUT_ROOT / f"Figure_4_{label}.{suffix}"
            if not path.is_file():
                raise FileNotFoundError(f"Missing Figure 4 panel export: {path.name}")
            formats[suffix] = {
                "path": str(path.relative_to(MODULE_ROOT.parent)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        panels[label] = formats
    payload = {
        "figure": "Figure 4",
        "panel_order": list(PANEL_LABELS),
        "formats": ["png", "pdf"],
        "png_dpi": OUTPUT_DPI,
        "note": "Standalone exports generated by the authoritative plotting-only main notebook.",
        "panels": panels,
    }
    path = PANEL_OUTPUT_ROOT / "Figure_4_panel_export_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def export_figure4_main(figure) -> dict:
    """Export the compiled Figure 4 page without changing the approved layout."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix in ("png", "pdf"):
        path = OUTPUT_ROOT / f"Figure_4_main_panel.{suffix}"
        figure.savefig(path, format=suffix, dpi=OUTPUT_DPI, facecolor="white")
        paths[suffix] = path
    return paths
