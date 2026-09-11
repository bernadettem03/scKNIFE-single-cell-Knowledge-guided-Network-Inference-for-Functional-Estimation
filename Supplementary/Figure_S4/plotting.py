#!/usr/bin/env python3
"""Plotting-only renderers and exporters for Supplementary Figure 4."""
from __future__ import annotations

from pathlib import Path
import hashlib
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = MODULE_ROOT
# Figure S4 plotting setup — staged numerical data only; no analysis is recomputed.
import json
import anndata as ad
import scanpy as sc
from matplotlib.lines import Line2D

S4_DATA = NOTEBOOK_DIR / "data/fig4_supplementary_plot_data.h5"
S4_OUT = NOTEBOOK_DIR / "outputs/panels"
S4_COMPILED = NOTEBOOK_DIR / "outputs/Figure_S4"
S4_MANIFEST = NOTEBOOK_DIR / "outputs/Figure_S4_manifest.json"
S4_OUT.mkdir(parents=True, exist_ok=True)

# ─── Figure S4 parameters (user-editable) ───────────────────────────
S4_WIDTH_IN = 11.0
S4_HEIGHT_IN = 16.0
S4_PANEL_TITLE_PT = 10
S4_AXIS_TITLE_PT = 8
S4_TICK_PT = 6.5
S4_LEGEND_PT = 6.2
S4_GENE_PT = 5.8
S4_UMAP_POINT_SIZE = 0.20
S4_UMAP_ALPHA = 0.90
S4_DOT_MIN_SIZE = 3.0
S4_DOT_MAX_SIZE = 48.0
S4_DPI = 300
S4_RASTERIZE_UMAPS = True
S4_EDITABLE_SVG_TEXT = True
# ─────────────────────────────────────────────────────────────────────

mpl.rcParams.update({"svg.fonttype": "none" if S4_EDITABLE_SVG_TEXT else "path"})

S4_CT_ORDER = [
    "Epithelial", "Epithelial_Luminal_Progenitor", "Epithelial_Luminal_Mature",
    "Epithelial_cycling", "Basal_Myoepithelial", "CAFs", "Macrophages",
    "T_cells", "B_cells", "Plasma_cells", "Endothelial", "pDC",
]
S4_CT_LABEL = {
    "Epithelial": "Epithelial", "Epithelial_Luminal_Progenitor": "Luminal Progenitor",
    "Epithelial_Luminal_Mature": "Luminal Mature", "Epithelial_cycling": "Cycling",
    "Basal_Myoepithelial": "Basal/Myoepithelial", "CAFs": "CAFs",
    "Macrophages": "Macrophages", "T_cells": "T cells", "B_cells": "B cells",
    "Plasma_cells": "Plasma cells", "Endothelial": "Endothelial", "pDC": "pDC",
}
S4_CT_COLORS = {
    "Epithelial": "#1f77b4", "Epithelial_Luminal_Progenitor": "#ff7f0e",
    "Epithelial_Luminal_Mature": "#2ca02c", "Epithelial_cycling": "#d62728",
    "Basal_Myoepithelial": "#9467bd", "CAFs": "#8c564b",
    "Macrophages": "#e377c2", "T_cells": "#7f7f7f", "B_cells": "#bcbd22",
    "Plasma_cells": "#17becf", "Endothelial": "#aec7e8", "pDC": "#ffbb78",
}
S4_CNV_CLUSTER_COLORS = [
    "#023fa5", "#7d87b9", "#bec1d4", "#d6bcc0", "#bb7784", "#8e063b",
    "#4a6fe3", "#8595e1", "#b5bbe3", "#e6afb9", "#e07b91", "#d33f6a",
    "#11c638", "#8dd593", "#c6dec7", "#ead3c6", "#f0b98d", "#ef9708",
    "#0fcfc0", "#9cded6", "#d5eae7", "#f3e1eb", "#f6c4e1", "#f79cd4",
    "#7f7f7f", "#c7c7c7",
]
S4_CNV_STATUS_ORDER = ["provisional_normal_CNV_low", "provisional_tumor_CNV_high"]
S4_CNV_STATUS_LABEL = {
    "provisional_normal_CNV_low": "CNV-low (provisional)",
    "provisional_tumor_CNV_high": "CNV-high shortlist",
}
S4_CNV_STATUS_COLORS = {
    "provisional_normal_CNV_low": "#bdbdbd",
    "provisional_tumor_CNV_high": "#7b3294",
}

def s4_strings(values):
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values])

if not S4_DATA.is_file():
    raise FileNotFoundError(S4_DATA)

with h5py.File(S4_DATA, "r") as bundle:
    s4_atlas_xy = np.asarray(bundle["atlas/umap"][:], dtype=np.float32)
    s4_atlas_celltype = s4_strings(bundle["atlas/celltype"][:])
    s4_composition_counts = np.asarray(bundle["atlas/composition_counts"][:], dtype=np.int64)
    s4_sample_names = s4_strings(bundle["atlas/patient_site"][:])
    s4_sample_counts = np.asarray(bundle["atlas/patient_site_counts"][:], dtype=np.int64)
    s4_qc_xy = np.asarray(bundle["cnv_qc/umap"][:], dtype=np.float32)
    s4_qc_cluster = s4_strings(bundle["cnv_qc/cnv_leiden"][:])
    s4_qc_celltype = s4_strings(bundle["cnv_qc/celltype"][:])
    s4_provisional_xy = np.asarray(bundle["cnv_provisional/umap"][:], dtype=np.float32)
    s4_provisional_status = s4_strings(bundle["cnv_provisional/status"][:])
    s4_marker_celltypes = s4_strings(bundle["marker_dotplots/celltypes"][:])
    s4_marker_genes = s4_strings(bundle["marker_dotplots/genes"][:])
    s4_marker_owner = s4_strings(bundle["marker_dotplots/gene_owner"][:])
    s4_marker_mean = np.asarray(bundle["marker_dotplots/mean_log1p"][:], dtype=np.float32)
    s4_marker_fraction = np.asarray(bundle["marker_dotplots/fraction_expressing"][:], dtype=np.float32)

assert s4_atlas_xy.shape == s4_qc_xy.shape == s4_provisional_xy.shape == (69_391, 2)
assert s4_sample_counts.shape == (15, 12)
assert s4_marker_mean.shape == s4_marker_fraction.shape == (12, 59)
assert s4_marker_celltypes.tolist() == S4_CT_ORDER

s4_sample_pct = 100 * s4_sample_counts / s4_sample_counts.sum(axis=1, keepdims=True)
s4_overall_pct = 100 * s4_composition_counts / s4_composition_counts.sum()
s4_status_counts = {status: int(np.sum(s4_provisional_status == status)) for status in S4_CNV_STATUS_ORDER}
assert sum(s4_status_counts.values()) == 69_391
assert s4_status_counts == {
    "provisional_normal_CNV_low": 27_421,
    "provisional_tumor_CNV_high": 41_970,
}

def s4_plot_adata(xy, values, categories, color_key, palette):
    obs = pd.DataFrame({color_key: pd.Categorical(values, categories=categories, ordered=True)})
    result = ad.AnnData(X=np.empty((len(obs), 0), dtype=np.float32), obs=obs)
    result.obsm["X_umap"] = xy
    result.uns[f"{color_key}_colors"] = np.asarray(palette, dtype=str)
    return result

s4_atlas_adata = s4_plot_adata(
    s4_atlas_xy, s4_atlas_celltype, S4_CT_ORDER, "celltype",
    [S4_CT_COLORS[key] for key in S4_CT_ORDER],
)
s4_qc_cluster_order = [str(index) for index in range(26)]
s4_qc_cluster_adata = s4_plot_adata(
    s4_qc_xy, s4_qc_cluster, s4_qc_cluster_order, "cnv_leiden", S4_CNV_CLUSTER_COLORS,
)
s4_qc_celltype_adata = s4_plot_adata(
    s4_qc_xy, s4_qc_celltype, S4_CT_ORDER, "celltype",
    [S4_CT_COLORS[key] for key in S4_CT_ORDER],
)
s4_provisional_adata = s4_plot_adata(
    s4_provisional_xy, s4_provisional_status, S4_CNV_STATUS_ORDER, "cnv_status",
    [S4_CNV_STATUS_COLORS[key] for key in S4_CNV_STATUS_ORDER],
)

def s4_clean_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def s4_panel_title(ax, label, title):
    ax.set_title(f"{label}  {title}", loc="left", fontsize=S4_PANEL_TITLE_PT, fontweight="bold", pad=5)

def s4_save(fig, stem):
    """Export one panel without changing or closing the live figure."""
    outputs = {}
    for suffix in ("png", "pdf"):
        path = S4_OUT / f"{stem}.{suffix}"
        fig.savefig(path, format=suffix, dpi=S4_DPI, bbox_inches="tight", facecolor="white")
        outputs[suffix] = str(path.relative_to(NOTEBOOK_DIR))
    return outputs

def s4_celltype_handles():
    return [Line2D([0], [0], marker="o", linestyle="none", markersize=3.2,
                   markerfacecolor=S4_CT_COLORS[key], markeredgecolor="none",
                   label=S4_CT_LABEL[key]) for key in S4_CT_ORDER]

def draw_s4_t(ax):
    x = np.arange(len(s4_sample_names)); bottom = np.zeros(len(x))
    for index, celltype in enumerate(S4_CT_ORDER):
        values = s4_sample_pct[:, index]
        ax.bar(x, values, bottom=bottom, width=.78, color=S4_CT_COLORS[celltype], linewidth=0)
        bottom += values
    ax.set_xticks(x, [name.replace("|", " ") for name in s4_sample_names], rotation=48, ha="right")
    ax.set_ylim(0, 100); ax.set_ylabel("% cells"); s4_clean_axis(ax)
    s4_panel_title(ax, "T", "Cell composition by patient and site")

def draw_s4_r(ax):
    y = np.arange(len(S4_CT_ORDER))
    ax.barh(y, s4_overall_pct, color=[S4_CT_COLORS[key] for key in S4_CT_ORDER], height=.76)
    ax.set_yticks(y, [S4_CT_LABEL[key] for key in S4_CT_ORDER]); ax.invert_yaxis()
    ax.set_xlabel("% cells"); s4_clean_axis(ax); s4_panel_title(ax, "R", "Cell-type composition")

def draw_s4_s(ax):
    x = np.arange(len(s4_marker_genes))
    for row, celltype in enumerate(S4_CT_ORDER):
        sizes = S4_DOT_MIN_SIZE + S4_DOT_MAX_SIZE * np.clip(s4_marker_fraction[row], 0, 1)
        ax.scatter(x, np.full(len(x), row), s=sizes, color=S4_CT_COLORS[celltype], linewidths=0)
    ax.set_xticks(x, s4_marker_genes, rotation=90)
    ax.set_yticks(np.arange(len(S4_CT_ORDER)), [S4_CT_LABEL[key] for key in S4_CT_ORDER])
    ax.invert_yaxis(); ax.set_axisbelow(True); ax.grid(axis="x", color="#eeeeee", linewidth=.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    s4_panel_title(ax, "S", "Canonical marker expression")

def draw_s4_scanpy(ax, adata, color_key, title):
    collection_start = len(ax.collections)
    sc.pl.umap(
        adata, color=color_key, ax=ax, show=False, frameon=False,
        size=S4_UMAP_POINT_SIZE, alpha=S4_UMAP_ALPHA, legend_loc=None,
    )
    if S4_RASTERIZE_UMAPS:
        for collection in ax.collections[collection_start:]:
            collection.set_rasterized(True)
    ax.set_xlabel("UMAP1", fontsize=S4_AXIS_TITLE_PT); ax.set_ylabel("UMAP2", fontsize=S4_AXIS_TITLE_PT)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    s4_panel_title(ax, title.split(" ", 1)[0], title.split(" ", 1)[1])

def draw_s4_q(ax): draw_s4_scanpy(ax, s4_atlas_adata, "celltype", "Q Cell-type UMAP")
def draw_s4_n(ax): draw_s4_scanpy(ax, s4_qc_cluster_adata, "cnv_leiden", "N CNV clusters")
def draw_s4_p(ax): draw_s4_scanpy(ax, s4_qc_celltype_adata, "celltype", "P Cell type")
def draw_s4_m(ax): draw_s4_scanpy(ax, s4_provisional_adata, "cnv_status", "M CNV shortlist")

PANEL_ORDER = tuple("TRSQNPM")
PANEL_STEMS = {
    "T": "S4_T_patient_site_composition",
    "R": "S4_R_overall_composition",
    "S": "S4_S_canonical_marker_expression",
    "Q": "S4_Q_celltype_umap",
    "N": "S4_N_cnv_clusters",
    "P": "S4_P_cnv_celltype",
    "M": "S4_M_cnv_shortlist",
}
PANEL_FIGSIZES = {"T": (10.2, 4.5), "R": (5.2, 4.5), "S": (12.5, 5.2), "Q": (6.0, 6.0), "N": (6.0, 6.0), "P": (6.0, 6.0), "M": (6.0, 6.0)}
PANEL_DRAWERS = {"T": draw_s4_t, "R": draw_s4_r, "S": draw_s4_s, "Q": draw_s4_q, "N": draw_s4_n, "P": draw_s4_p, "M": draw_s4_m}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_s4_inputs():
    """Return a compact audit after validating the frozen S4 plotting bundle."""
    if not S4_DATA.is_file():
        raise FileNotFoundError(S4_DATA)
    return {
        "source": str(S4_DATA.relative_to(MODULE_ROOT)),
        "source_sha256": _sha256(S4_DATA),
        "cells": int(len(s4_atlas_xy)),
        "specimens": int(len(s4_sample_names)),
        "markers": int(len(s4_marker_genes)),
        "cnv_clusters": int(len(s4_qc_cluster_order)),
        "panel_order": list(PANEL_ORDER),
        "analysis_recomputed": False,
    }


def _add_panel_legend(ax, label):
    if label in ("Q", "P"):
        ax.legend(handles=s4_celltype_handles(), frameon=False, ncol=2, fontsize=S4_LEGEND_PT,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Cell type")
    elif label == "N":
        handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=3,
                          markerfacecolor=color, markeredgecolor="none", label=cluster)
                   for cluster, color in zip(s4_qc_cluster_order, S4_CNV_CLUSTER_COLORS)]
        ax.legend(handles=handles, frameon=False, ncol=2, fontsize=S4_LEGEND_PT,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), title="CNV cluster")
    elif label == "M":
        handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=4,
                          markerfacecolor=S4_CNV_STATUS_COLORS[key], markeredgecolor="none",
                          label=f"{S4_CNV_STATUS_LABEL[key]}: {s4_status_counts[key]:,}")
                   for key in S4_CNV_STATUS_ORDER]
        ax.legend(handles=handles, frameon=False, fontsize=S4_LEGEND_PT,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), title="CNV status")


def render_s4_panel(label):
    """Render one authoritative S4 panel directly from frozen plot-ready data."""
    if label not in PANEL_ORDER:
        raise ValueError(f"Unknown Supplementary Figure 4 panel: {label}")
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZES[label])
    PANEL_DRAWERS[label](ax)
    _add_panel_legend(ax, label)
    return fig


def export_s4_panel(fig, label):
    if label not in PANEL_ORDER:
        raise ValueError(label)
    return s4_save(fig, PANEL_STEMS[label])


def render_s4_compiled():
    """Render the approved T/R/S/Q/N/P/M composite."""
    fig = plt.figure(figsize=(S4_WIDTH_IN, S4_HEIGHT_IN), facecolor="white")
    grid = fig.add_gridspec(4, 12, height_ratios=[1.05, 1.45, .86, .74],
                            left=.065, right=.965, bottom=.035, top=.955, hspace=.52, wspace=1.35)
    ax_t=fig.add_subplot(grid[0,:8]);ax_r=fig.add_subplot(grid[0,8:]);ax_s=fig.add_subplot(grid[1,:])
    ax_q=fig.add_subplot(grid[2,0:3]);ax_n=fig.add_subplot(grid[2,3:6]);ax_p=fig.add_subplot(grid[2,6:9]);ax_m=fig.add_subplot(grid[2,9:12])
    draw_s4_t(ax_t);draw_s4_r(ax_r);draw_s4_s(ax_s);draw_s4_q(ax_q);draw_s4_n(ax_n);draw_s4_p(ax_p);draw_s4_m(ax_m)
    legend_q=fig.add_subplot(grid[3,0:3]);legend_n=fig.add_subplot(grid[3,3:6]);legend_p=fig.add_subplot(grid[3,6:9]);legend_m=fig.add_subplot(grid[3,9:12])
    for axis in (legend_q,legend_n,legend_p,legend_m):axis.axis("off")
    legend_q.legend(handles=s4_celltype_handles(),title="Q cell type",frameon=False,fontsize=S4_LEGEND_PT,title_fontsize=S4_LEGEND_PT,loc="upper left")
    cluster_handles=[Line2D([0],[0],marker="o",linestyle="none",markersize=2.6,markerfacecolor=color,markeredgecolor="none",label=cluster) for cluster,color in zip(s4_qc_cluster_order,S4_CNV_CLUSTER_COLORS)]
    legend_n.legend(handles=cluster_handles,title="N cluster",frameon=False,ncol=2,fontsize=S4_LEGEND_PT,title_fontsize=S4_LEGEND_PT,loc="upper left")
    legend_p.legend(handles=s4_celltype_handles(),title="P cell type",frameon=False,fontsize=S4_LEGEND_PT,title_fontsize=S4_LEGEND_PT,loc="upper left")
    status_handles=[Line2D([0],[0],marker="o",linestyle="none",markersize=3.2,markerfacecolor=S4_CNV_STATUS_COLORS[key],markeredgecolor="none",label=S4_CNV_STATUS_LABEL[key]) for key in S4_CNV_STATUS_ORDER]
    legend_m.legend(handles=status_handles,title="M CNV status",frameon=False,fontsize=S4_LEGEND_PT,title_fontsize=S4_LEGEND_PT,loc="upper left")
    fig.suptitle("Figure S4. Cell-type annotation and CNV quality control",fontsize=12,fontweight="bold",y=.985)
    return fig


def export_s4_compiled(fig):
    outputs={}
    for suffix in ("png","pdf"):
        path=S4_COMPILED.with_suffix("."+suffix);fig.savefig(path,format=suffix,dpi=S4_DPI,facecolor="white");outputs[suffix]=str(path.relative_to(NOTEBOOK_DIR))
    return outputs


def write_s4_manifest(panel_outputs, compiled_outputs):
    manifest={
        "notebook":"Figure_S4.ipynb","mode":"plot_only","analysis_recomputed":False,
        "source":str(S4_DATA.relative_to(NOTEBOOK_DIR)),"source_sha256":_sha256(S4_DATA),
        "panel_order":list(PANEL_ORDER),
        "fields":{"T":["atlas/patient_site","atlas/patient_site_counts"],"R":["atlas/composition_counts"],"S":["marker_dotplots/genes","marker_dotplots/fraction_expressing"],"Q":["atlas/umap","atlas/celltype"],"N":["cnv_qc/umap","cnv_qc/cnv_leiden"],"P":["cnv_qc/umap","cnv_qc/celltype"],"M":["cnv_provisional/umap","cnv_provisional/status"]},
        "renderers":{"T":"matplotlib","R":"matplotlib","S":"matplotlib","Q":"scanpy","N":"scanpy","P":"scanpy","M":"scanpy"},
        "export":{"svg_text":"editable" if S4_EDITABLE_SVG_TEXT else "paths","rasterized_panels":["Q","N","P","M"] if S4_RASTERIZE_UMAPS else [],"raster_dpi":S4_DPI,"umap_point_size":S4_UMAP_POINT_SIZE,"umap_alpha":S4_UMAP_ALPHA},
        "individual_outputs":panel_outputs,"compiled_outputs":compiled_outputs,
        "checks":{"cells":69391,"specimens":15,"markers":59,"cnv_clusters":26,"cnv_low":27421,"cnv_high":41970},
    }
    S4_MANIFEST.write_text(json.dumps(manifest,indent=2)+"\n")
    return S4_MANIFEST
