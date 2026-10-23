"""Single-page Supplementary Figure S6 panels A–E from checksummed inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotting as legacy

MM = 1 / 25.4
PANEL_ORDER = tuple("ABCDE")
INPUTS = ("status_four_set_membership.csv", "status_pathway_scatter.csv",
          "status_pathway_enrichment.csv", "status_lcms_scknife_coverage.csv",
          "carnitine_activity_umap.csv.gz")
NODES = (("L_carnitine", "C", "Myeloid L-carnitine activity"),
         ("L_octanoylcarnitine", "D", "Myeloid L-octanoylcarnitine activity"),
         ("octanoyl_CoA", "E", "Myeloid octanoyl-CoA activity"))

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.grid": False, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42,
                     "svg.fonttype": "none"})


def load_data(root: Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    records = {item["path"]: item for item in json.loads((root / "manifest.json").read_text())["files"]}
    data = {}
    for name in INPUTS:
        payload = (root / name).read_bytes()
        record = records[name]
        if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise RuntimeError(f"S6 input checksum mismatch: {name}")
        data[name] = pd.read_csv(root / name)
    return data


def _title(container, label: str, title: str) -> None:
    container.text(.02, .975, f"{label}  {title}", ha="left", va="top",
                   fontsize=9, fontweight="bold")


def _coverage(container, data: dict) -> None:
    mapping = data["status_four_set_membership.csv"]
    coverage = data["status_lcms_scknife_coverage.csv"]
    assert len(mapping) == 33 and int(mapping.mapped_to_humangem.sum()) == 31
    assert len(coverage) == 140 and coverage.pathway.nunique() == 140
    counts = coverage.groupby(["lcms", "scknife"]).size().to_dict()
    assert counts == {(False, True): 111, (True, False): 2, (True, True): 27}
    categories = [("Shared", 27, (True, True)),
                  ("scKNIFE only", 111, (False, True)),
                  ("LC–MS only", 2, (True, False))]
    grid = container.add_gridspec(2, 2, left=.17, right=.94, top=.82,
                                  bottom=.15, width_ratios=[.28, .72],
                                  height_ratios=[.66, .34], hspace=.04, wspace=.07)
    ax_bar = container.add_subplot(grid[0, 1])
    ax_sets = container.add_subplot(grid[1, 0])
    ax_matrix = container.add_subplot(grid[1, 1], sharex=ax_bar)
    x = np.arange(3)
    bars = ax_bar.bar(x, [c[1] for c in categories], width=.58,
                      color=["#6B4C9A", "#CC79A7", "#666666"])
    for bar, (_, count, _) in zip(bars, categories):
        ax_bar.text(bar.get_x() + bar.get_width()/2, count + 2, str(count),
                    ha="center", fontsize=7)
    ax_bar.set_ylim(0, 126)
    ax_bar.set_ylabel("Pathways")
    ax_bar.set_xticks([])
    ax_sets.barh([0, 1], [29, 138], color=["#666666", "#CC79A7"], height=.5)
    ax_sets.set_yticks([0, 1], ["LC–MS", "scKNIFE"])
    ax_sets.invert_xaxis()
    ax_sets.invert_yaxis()
    ax_sets.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_sets.spines[["left", "bottom"]].set_visible(False)
    for xi, (_, _, membership) in enumerate(categories):
        for yi, present in enumerate(membership):
            ax_matrix.scatter(xi, yi, s=45, facecolor=("#666666" if yi == 0 else "#CC79A7") if present else "white",
                              edgecolor="#AAAAAA", linewidth=.55, zorder=3)
        if all(membership):
            ax_matrix.plot([xi, xi], [0, 1], color="#333333", lw=1.2, zorder=2)
    ax_matrix.set_yticks([])
    ax_matrix.set_ylim(1.4, -.4)
    ax_matrix.set_xticks(x, [c[0] for c in categories], fontsize=7)
    ax_matrix.tick_params(axis="x", length=0)
    ax_matrix.spines[:].set_visible(False)
    _title(container, "A", "Metastatic-status pathway coverage")


def _enrichment(container, data: dict) -> None:
    legacy._draw_status_pathway_scatter(container, data, "B", show_subtitle=False)


def _umap_data(data: dict):
    frame = data["carnitine_activity_umap.csv.gz"]
    assert len(frame) == 69391 and frame.cell_id.is_unique
    mask = frame.celltype.eq("Myeloid") & frame.Timepoint.isin(("Bx", "Surgery"))
    myeloid = frame.loc[mask]
    assert len(myeloid) == 6014 and myeloid[[node for node, _, _ in NODES]].notna().all().all()
    xy = frame[["UMAP1", "UMAP2"]].to_numpy(float)
    middle = xy.mean(axis=0)
    half = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) * .54
    return frame, myeloid, middle, half


def _draw_umap_axis(container, ax, frame, myeloid, middle, half, node, letter, title):
    scores = myeloid[node].to_numpy(float)
    lo, hi = np.percentile(scores, [2, 98])
    ax.scatter(frame.UMAP1, frame.UMAP2, s=.07, c="#D9D9D9",
               alpha=.4, linewidths=0, rasterized=True)
    sc = ax.scatter(myeloid.UMAP1, myeloid.UMAP2, c=scores,
                    s=.55, cmap="viridis", vmin=lo, vmax=hi,
                    linewidths=0, rasterized=True)
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=7.3, fontweight="bold")
    ax.set_xlim(middle[0] - half, middle[0] + half)
    ax.set_ylim(middle[1] - half, middle[1] + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False)
    cb = container.colorbar(sc, ax=ax, fraction=.035, pad=.01)
    cb.ax.tick_params(labelsize=5.5, length=2)


def _umap(container, data: dict) -> None:
    frame, myeloid, middle, half = _umap_data(data)
    axes = container.subplots(1, 3)
    for ax, (node, letter, title) in zip(axes, NODES):
        _draw_umap_axis(container, ax, frame, myeloid, middle, half, node, letter, title)
    container.subplots_adjust(left=.02, right=.92, top=.92, bottom=.11, wspace=.20)


def _single_umap(container, data: dict, letter: str) -> None:
    frame, myeloid, middle, half = _umap_data(data)
    node, _, title = next(item for item in NODES if item[1] == letter)
    ax = container.subplots(1, 1)
    _draw_umap_axis(container, ax, frame, myeloid, middle, half, node, letter, title)
    container.subplots_adjust(left=.08, right=.90, top=.88, bottom=.08)


DRAW = {"A": _coverage, "B": _enrichment,
        **{letter: (lambda c, d, l=letter: _single_umap(c, d, l)) for letter in "CDE"}}


def make_panel(label: str, data: dict):
    if label not in DRAW:
        raise ValueError(label)
    size = {"A": (180, 95), "B": (180, 115),
            "C": (100, 100), "D": (100, 100), "E": (100, 100)}[label]
    fig = plt.figure(figsize=(size[0]*MM, size[1]*MM), facecolor="white")
    DRAW[label](fig, data)
    return fig


def make_page(data: dict, page: int = 1):
    if page != 1:
        raise ValueError(page)
    fig = plt.figure(figsize=(297*MM, 210*MM), facecolor="white")
    rows = fig.subfigures(2, 1, height_ratios=[.54, .46], hspace=.06)
    a, b = rows[0].subfigures(1, 2, width_ratios=[1, 1], wspace=.055)
    _coverage(a, data)
    _enrichment(b, data)
    _umap(rows[1], data)
    return fig
