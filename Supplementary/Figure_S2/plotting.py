"""Supplementary Figure 2 plotting functions."""
from pathlib import Path
import hashlib, json, re, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
DATASET_ORDER=["bassez_c1","bassez_c2","wu_tnbc"]
DATASET_LABEL={"bassez_c1":"Bassez C1","bassez_c2":"Bassez C2","wu_tnbc":"Wu TNBC"}
CATEGORY_PALETTE=["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9","#D55E00","#F0E442","#6B6B6B","#8DD3C7","#BEBADA","#FB8072","#80B1D3","#FDB462","#B3DE69","#FCCDE5","#BC80BD","#CCEBC5","#FFED6F","#1B9E77","#7570B3"]
OUTPUT_DPI=300
# Supplementary Figure 2 — separate A4 mixed raster/vector compilation and export.
# Uses raster derivatives of the validated frozen Figure 2 coordinates and labels; no clustering or embedding is run here.

from pathlib import Path
import hashlib
import json
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from IPython.display import display

mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "pdf.use14corefonts": False})

SUPP_PAGE_W_MM, SUPP_PAGE_H_MM = 210, 297
SUPP_LABEL_PT = 7
SUPP_TITLE_PT = 7
SUPP_TICK_PT = 7
SUPP_NOTE_PT = 7
SUPP_LEGEND_PT = 7
SUPP_LEGEND_MARKER_PT = 3.5
SUPP_DATASET_ORDER = DATASET_ORDER.copy()
SUPP_DATA_DIR = Path(__file__).resolve().parent / "data" / "no_match_umaps"
SUPP_MANIFEST_PATH = SUPP_DATA_DIR / "no_match_umap_manifest.json"
SUPP_UMAP_METHODS = [("expimap", "ExpiMap"), ("slalom", "Slalom"), ("spectra", "SPECTRA")]
SUPP_UMAP_FILES = {(slug, dataset): SUPP_DATA_DIR / f"{slug}_{dataset}_points.png" for slug, _ in SUPP_UMAP_METHODS for dataset in SUPP_DATASET_ORDER}
SUPP_W_DATA_DIR = Path(__file__).resolve().parent / "data" / "pooled_w"
SUPP_CONCORDANCE_PATH = SUPP_W_DATA_DIR / "w_pooled_concordance_metrics.csv"
SUPP_CONCORDANCE_MANIFEST_PATH = SUPP_W_DATA_DIR / "w_pooled_concordance_manifest.json"
SUPP_TSNE_PATHS = [
    SUPP_W_DATA_DIR / "w_pooled_tsne_dataset_panels.png",
    SUPP_W_DATA_DIR / "w_pooled_tsne_node_kinds_corrected.png",
]
SUPP_CONCORDANCE_DISPLAY_W_MM = 98
SUPP_CONCORDANCE_X_MM = 5
SUPP_CONCORDANCE_Y_MM = 16
SUPP_TSNE_DISPLAY_W_MM = 96
SUPP_TSNE_X_MM = 109
SUPP_TSNE_Y_MM = 10
SUPP_TSNE_GAP_MM = 4
if not SUPP_MANIFEST_PATH.is_file():
    raise FileNotFoundError(f"Missing no-match UMAP manifest: {SUPP_MANIFEST_PATH}")
if not SUPP_CONCORDANCE_PATH.is_file():
    raise FileNotFoundError(f"Missing pooled concordance data: {SUPP_CONCORDANCE_PATH}")
if not SUPP_CONCORDANCE_MANIFEST_PATH.is_file():
    raise FileNotFoundError(f"Missing pooled concordance manifest: {SUPP_CONCORDANCE_MANIFEST_PATH}")
for tsne_path in SUPP_TSNE_PATHS:
    if not tsne_path.is_file():
        raise FileNotFoundError(f"Missing pooled W-analysis t-SNE panel: {tsne_path}")
SUPP_MANIFEST = json.loads(SUPP_MANIFEST_PATH.read_text())
if SUPP_MANIFEST.get("schema_version") != "no_match_umap_raster_v3":
    raise RuntimeError(f"Unexpected no-match manifest schema: {SUPP_MANIFEST.get('schema_version')}")
if SUPP_MANIFEST.get("source_mode") != "validated_frozen_figure2_bundle":
    raise RuntimeError("Supplementary Figure 2 requires derivatives of the validated frozen Figure 2 bundle")
expected_output_keys = {f"{slug}/{dataset}" for slug, _ in SUPP_UMAP_METHODS for dataset in SUPP_DATASET_ORDER}
if set(SUPP_MANIFEST.get("outputs", {})) != expected_output_keys:
    raise RuntimeError("No-match manifest does not contain exactly the nine expected method/cohort panels")

def supp_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for slug, _ in SUPP_UMAP_METHODS:
    for dataset in SUPP_DATASET_ORDER:
        key = f"{slug}/{dataset}"
        entry = SUPP_MANIFEST["outputs"][key]
        raster_path = SUPP_UMAP_FILES[(slug, dataset)]
        table_path = Path(__file__).resolve().parents[2] / "Figure_2" / "data" / f"{slug}_{dataset}_umap.csv.gz"
        if not raster_path.is_file():
            raise FileNotFoundError(f"Missing packaged frozen raster: {raster_path}")
        if supp_sha256(raster_path) != entry["combined_png_sha256"]:
            raise RuntimeError(f"Manifest checksum mismatch for {raster_path}")
        if supp_sha256(table_path) != entry["source_sha256"]:
            raise RuntimeError(f"Frozen source-table checksum mismatch for {table_path}")
        frame = pd.read_csv(table_path, usecols=["leiden_cluster", "author_cell_type"], dtype=str)
        if len(frame) != entry["cell_count"]:
            raise RuntimeError(f"Frozen cell count mismatch for {key}")
        if set(frame["leiden_cluster"]) != set(entry["leiden_colors"]):
            raise RuntimeError(f"Frozen Leiden labels mismatch for {key}")
        if set(frame["author_cell_type"]) != set(entry["author_colors"]):
            raise RuntimeError(f"Frozen author labels mismatch for {key}")

def supp_natural_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]

def supp_rect(x_mm, y_mm, width_mm, height_mm):
    return [
        x_mm / SUPP_PAGE_W_MM,
        y_mm / SUPP_PAGE_H_MM,
        width_mm / SUPP_PAGE_W_MM,
        height_mm / SUPP_PAGE_H_MM,
    ]

def supp_frozen_legend_specs(slug, dataset):
    key = f"{slug}/{dataset}"
    entry = SUPP_MANIFEST["outputs"][key]
    leiden_order = sorted(entry["leiden_colors"], key=lambda value: int(value))
    author_order = sorted(entry["author_colors"], key=supp_natural_key)
    cluster_handles = [
        Line2D(
            [], [], marker="o", linestyle="", markersize=SUPP_LEGEND_MARKER_PT,
            markerfacecolor=entry["leiden_colors"][label],
            markeredgecolor="none", label=label,
        )
        for label in leiden_order
    ]
    author_handles = [
        Line2D(
            [], [], marker="o", linestyle="", markersize=SUPP_LEGEND_MARKER_PT,
            markerfacecolor=entry["author_colors"][label],
            markeredgecolor="none", label=label,
        )
        for label in author_order
    ]
    return cluster_handles, author_handles

def supp_overlay_vector_content(fig, title_rect, legend_rect, slug, dataset):
    cluster_handles, author_handles = supp_frozen_legend_specs(slug, dataset)
    title_axis = fig.add_axes(title_rect, frameon=False, zorder=10)
    title_axis.set_axis_off()
    title_axis.text(0.25, 0.50, "Leiden", ha="center", va="center", fontsize=SUPP_TITLE_PT)
    title_axis.text(0.75, 0.50, "Author cell type", ha="center", va="center", fontsize=SUPP_TITLE_PT)

    legend_axis = fig.add_axes(legend_rect, frameon=False, zorder=10)
    legend_axis.set_axis_off()
    legend_common = {
        "frameon": False,
        "fontsize": SUPP_LEGEND_PT,
        "handlelength": 0.55,
        "handletextpad": 0.20,
        "borderaxespad": 0,
        "borderpad": 0,
        "labelspacing": 0.08,
        "columnspacing": 0.45,
        "markerscale": 1.0,
    }
    cluster_legend = legend_axis.legend(
        handles=cluster_handles,
        loc="upper left",
        bbox_to_anchor=(0.00, 1.00),
        ncol=2 if len(cluster_handles) > 8 else 1,
        **legend_common,
    )
    legend_axis.add_artist(cluster_legend)
    legend_axis.legend(
        handles=author_handles,
        loc="upper left",
        bbox_to_anchor=(0.52, 1.00),
        ncol=1,
        **legend_common,
    )

def draw_supp_nomatch_umaps(fig):
    x0_mm, y_top_mm = 15, 214
    cell_width_mm, cell_height_mm = 58, 40
    plot_height_mm, legend_height_mm, title_height_mm = 18, 18, 4
    col_gap_mm, row_gap_mm = 4, 4
    for col_index, dataset in enumerate(SUPP_DATASET_ORDER):
        center_x = x0_mm + col_index * (cell_width_mm + col_gap_mm) + cell_width_mm / 2
        fig.text(
            center_x / SUPP_PAGE_W_MM,
            259 / SUPP_PAGE_H_MM,
            DATASET_LABEL[dataset],
            ha="center",
            va="bottom",
            fontsize=SUPP_TITLE_PT,
        )
    for row_index, (slug, method) in enumerate(SUPP_UMAP_METHODS):
        y_mm = y_top_mm - row_index * (cell_height_mm + row_gap_mm)
        fig.text(
            11 / SUPP_PAGE_W_MM,
            (y_mm + cell_height_mm / 2) / SUPP_PAGE_H_MM,
            method,
            ha="center",
            va="center",
            rotation=90,
            fontsize=SUPP_TITLE_PT,
        )
        for col_index, dataset in enumerate(SUPP_DATASET_ORDER):
            x_mm = x0_mm + col_index * (cell_width_mm + col_gap_mm)
            image_rect = supp_rect(
                x_mm, y_mm + legend_height_mm, cell_width_mm, plot_height_mm
            )
            title_rect = supp_rect(
                x_mm, y_mm + legend_height_mm + plot_height_mm, cell_width_mm, title_height_mm
            )
            legend_rect = supp_rect(
                x_mm, y_mm, cell_width_mm, legend_height_mm
            )
            ax = fig.add_axes(image_rect, zorder=0)
            ax.imshow(plt.imread(SUPP_UMAP_FILES[(slug, dataset)]), aspect="auto", interpolation="none")
            ax.axis("off")
            supp_overlay_vector_content(fig, title_rect, legend_rect, slug, dataset)


def draw_supp_concordance(fig):
    metrics = pd.read_csv(SUPP_CONCORDANCE_PATH)
    order = ["Bassez_A", "Bassez_B", "Shiao", "Wu"]
    labels = {"Bassez_A":"Bassez C1", "Bassez_B":"Bassez C2", "Shiao":"Shiao", "Wu":"Wu TNBC"}
    matrix = np.eye(len(order), dtype=float)
    for row in metrics.loc[metrics.record_type.eq("pair")].itertuples():
        i, j = order.index(row.dataset_a), order.index(row.dataset_b)
        matrix[i, j] = matrix[j, i] = float(row.linear_cka)
    cmap, norm = plt.get_cmap("viridis"), mpl.colors.Normalize(0, 1)
    fig.text(7/SUPP_PAGE_W_MM, 75/SUPP_PAGE_H_MM, "Cross-dataset W concordance", fontsize=SUPP_TITLE_PT, fontweight="bold")
    ax = fig.add_axes(supp_rect(7, 30, 37, 37))
    for i in range(4):
        for j in range(4):
            value = matrix[i, j]
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor=cmap(norm(value)), edgecolor="white", linewidth=.35))
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=5.2, color="white" if value >= .58 else "black")
    ax.set(xlim=(-.5,3.5), ylim=(3.5,-.5), xlabel="Dataset", ylabel="Dataset")
    ax.set_xticks(range(4), [labels[d] for d in order], rotation=35, ha="right", fontsize=5.2)
    ax.set_yticks(range(4), [labels[d] for d in order], fontsize=5.2)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    cax = fig.add_axes(supp_rect(46, 30, 2, 37))
    color_steps = 100
    for step in range(color_steps):
        cax.add_patch(Rectangle((0, step/color_steps), 1, 1/color_steps, facecolor=cmap((step+.5)/color_steps), edgecolor="none"))
    cax.set(xlim=(0,1), ylim=(0,1), ylabel="Linear CKA")
    cax.set_xticks([]); cax.set_yticks(np.linspace(0,1,6)); cax.tick_params(labelsize=5.2, length=1.5)
    cax.yaxis.set_label_position("right"); cax.yaxis.tick_right(); cax.yaxis.label.set_size(5.2)
    table = fig.add_axes(supp_rect(53, 28, 48, 40), frameon=False); table.set_axis_off()
    table.set_title("Within-dataset factor diagnostics", loc="left", fontsize=SUPP_TITLE_PT, fontweight="bold")
    headers = ("Dataset", "Factors", "Eff. rank", "Med. |r|", "P95 |r|", "Condition")
    xpos = (0, .24, .44, .65, .82, 1.0)
    for x, value in zip(xpos, headers):
        table.text(x, .84, value, ha="left" if x == 0 else "right", fontsize=4.5, fontweight="bold", transform=table.transAxes)
    table.plot((0,1),(.78,.78), color="#555555", linewidth=.5, transform=table.transAxes)
    stats = metrics.loc[metrics.record_type.eq("dataset")].set_index("dataset_a")
    for y, dataset in zip((.64,.49,.34,.19), order):
        row = stats.loc[dataset]
        values = (labels[dataset], f"{int(row.n_factors)}", f"{row.effective_rank:.1f} ({100*row.effective_rank_fraction:.0f}%)", f"{row.median_abs_factor_correlation:.3f}", f"{row.p95_abs_factor_correlation:.3f}", f"{row.condition_number:.2f}")
        for x, value in zip(xpos, values): table.text(x, y, value, ha="left" if x == 0 else "right", fontsize=4.5, transform=table.transAxes)
    table.add_patch(Rectangle((0,.01),1,.10,facecolor="#F1F5F4",edgecolor="#8DA6A0",linewidth=.5,transform=table.transAxes))
    table.text(.5,.06,"Concordant W structure with limited factor redundancy",ha="center",va="center",fontsize=4.5,fontweight="bold",color="#244B45",transform=table.transAxes)
    fig.text(7/SUPP_PAGE_W_MM, 20/SUPP_PAGE_H_MM, "CKA quantifies shared node geometry; effective rank and factor correlations quantify within-run redundancy.", fontsize=4.5, color="#444444")


def draw_supp_tsne_panels(fig):
    y_mm = SUPP_TSNE_Y_MM
    for tsne_path in SUPP_TSNE_PATHS:
        tsne_image = plt.imread(tsne_path)
        image_height_px, image_width_px = tsne_image.shape[:2]
        display_height_mm = SUPP_TSNE_DISPLAY_W_MM * image_height_px / image_width_px
        ax = fig.add_axes(
            supp_rect(SUPP_TSNE_X_MM, y_mm, SUPP_TSNE_DISPLAY_W_MM, display_height_mm),
            zorder=0,
        )
        ax.imshow(tsne_image, aspect="equal", interpolation="none")
        ax.axis("off")
        y_mm += display_height_mm + SUPP_TSNE_GAP_MM



def supp_raster_asset(path, role, display_width_mm):
    image_width_px = plt.imread(path).shape[1]
    return {
        "path": path.as_posix(),
        "sha256": supp_sha256(path),
        "role": role,
        "display_width_mm": display_width_mm,
        "effective_dpi": image_width_px / (display_width_mm / 25.4),
    }


def make_figure2_supplementary():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(
        12 / SUPP_PAGE_W_MM,
        285 / SUPP_PAGE_H_MM,
        "Supplementary Figure 2 | Frozen no-match UMAPs, pooled W-space t-SNE panels, and concordance",
        ha="left",
        va="bottom",
        fontsize=SUPP_LABEL_PT,
    )
    draw_supp_nomatch_umaps(fig)
    draw_supp_concordance(fig)
    draw_supp_tsne_panels(fig)
    return fig
