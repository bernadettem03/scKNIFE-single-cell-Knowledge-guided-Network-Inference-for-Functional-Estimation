"""Render the approved Figure 4 SingleR panels from checked, staged inputs."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image

GENES = ("EPCAM", "DCN", "RGS5", "CD2", "MZB1", "C1QA")
N_CELLS = 69_391
PURPLE = LinearSegmentedColormap.from_list("marker_purple", ["#ffffff", "#56227f"])


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(fig, target: Path, dpi: int = 300) -> None:
    fig.savefig(target.with_suffix(".png"), dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _marker_panel(data: Path, output: Path) -> None:
    with h5py.File(data / "overlays.h5", "r") as handle:
        xy = handle["umap"][:]
        values = handle["marker_expression"][:]
        vmax = handle["marker_vmax"][:]
        genes = tuple(g.decode() for g in handle["marker_genes"][:])
    assert xy.shape == (N_CELLS, 2) and values.shape == (N_CELLS, len(GENES))
    assert genes == GENES
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.6), constrained_layout=False)
    fig.subplots_adjust(left=.06, right=.98, top=.93, bottom=.08, hspace=.22, wspace=.12)
    fig.text(.05, .982, "F", fontsize=18, fontweight="bold", va="top")
    fig.text(.14, .982, "Canonical lineage-marker expression", fontsize=13, va="top")
    for index, ax in enumerate(axes.flat):
        v = values[:, index]
        order = np.argsort(v, kind="stable")
        ax.scatter(xy[:, 0], xy[:, 1], s=.08, c="#dddddd", rasterized=True, linewidths=0)
        positive = order[v[order] > 0]
        ax.scatter(xy[positive, 0], xy[positive, 1], s=.12, c=v[positive], cmap=PURPLE,
                   vmin=0, vmax=float(vmax[index]), rasterized=True, linewidths=0)
        ax.set_title(genes[index], fontsize=12, fontweight="bold", pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_aspect("equal", adjustable="box")
    _save(fig, output)


def _score_panel(data: Path, output: Path, manifest: dict) -> None:
    image = Image.open(data / "G_score_body.png").convert("RGB")
    fig = plt.figure(figsize=(11.2, 5.7), facecolor="white")
    ax = fig.add_axes([.015, .06, .97, .76])
    ax.imshow(image, interpolation="nearest")
    ax.axis("off")
    fig.text(.025, .985, "G", fontsize=18, fontweight="bold", va="top")
    fig.text(.07, .985, "SingleR major-label relative scores", fontsize=13, va="top")
    for index, (label, color) in enumerate(manifest["patient_legend"]):
        x = .09 + index * .125
        fig.add_artist(plt.Rectangle((x, .885), .018, .026, transform=fig.transFigure,
                                     facecolor=color, edgecolor="none"))
        fig.text(x+.023, .898, label, fontsize=9, va="center")
    fig.text(.04, .02, "Scores are within-cell range-scaled and cubed; not probabilities.", fontsize=9)
    _save(fig, output)


def _compose(panels: Path, target: Path) -> None:
    # One 210 × 390 mm journal canvas; each embedded source plot is used without
    # changing its data or plotted coordinates. Text added here remains vector.
    fig = plt.figure(figsize=(210/25.4, 390/25.4), facecolor="white")
    mm = lambda box: [box[0]/210, box[1]/390, box[2]/210, box[3]/390]
    layout = {
        "C": (8, 300, 95, 80),
        "E": (9, 213, 95, 82),
        "F": (108, 266, 94, 115),
        "B": (125, 205, 65, 62),
        "D": (8, 113, 194, 89),
        "G": (8, 9, 194, 99),
    }
    for letter, box in layout.items():
        ax = fig.add_axes(mm(box))
        ax.imshow(Image.open(panels / f"Figure_4_{letter}.png"))
        ax.axis("off")
        if letter in {"C", "D"}:
            fig.text((box[0]+1)/210, (box[1]+box[3]-2)/390, letter,
                     fontsize=17, fontweight="bold", va="top", ha="left")
    fig.savefig(target.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(target.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def render_package(root: str | Path | None = None) -> dict:
    root = Path(root or Path(__file__).resolve().parent).resolve()
    is_git = root.name == "Figure_4"
    data = root / ("data/singler" if is_git else "data/Figure_4/singler")
    panels = root / ("outputs/panels" if is_git else "figs/individual/Figure_4")
    main = root / ("outputs/Figure_4_main_panel" if is_git else "figs/panel/Figure_4/Figure_4_main_panel")
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["cells"] == N_CELLS and tuple(manifest["marker_genes"]) == GENES
    for name, checksum in manifest["packaged_sha256"].items():
        if name == "manifest.json":
            continue  # A manifest cannot contain a stable hash of itself.
        assert _sha(data / name) == checksum, f"Staged input changed: {name}"
    panels.mkdir(parents=True, exist_ok=True)
    main.parent.mkdir(parents=True, exist_ok=True)
    for letter in ("B", "C", "D", "E"):
        source_name = f"{letter}_source" if letter in "BE" else letter
        for suffix in ("png", "pdf"):
            shutil.copy2(data / f"{source_name}.{suffix}", panels / f"Figure_4_{letter}.{suffix}")
    _marker_panel(data, panels / "Figure_4_F")
    _score_panel(data, panels / "Figure_4_G", manifest)
    _compose(panels, main)
    if is_git:
        shutil.copy2(main.with_suffix(".pdf"), root / "outputs/Figure_4.pdf")
        shutil.copy2(main.with_suffix(".png"), root / "outputs/Figure_4_preview.png")
    audit = {
        "cells": N_CELLS,
        "panel_order": ["B", "C", "D", "E", "F", "G"],
        "singleR_reference": "Wu et al. breast cancer atlas; major labels",
        "embedding": manifest["embedding"],
        "cnv": manifest["cnv"],
        "marker_genes": list(GENES),
        "score_scale": manifest["score_scale"],
        "patient_display_ids": [item[0] for item in manifest["patient_legend"]],
        "panels_sha256": {f"{letter}.{suffix}": _sha(panels / f"Figure_4_{letter}.{suffix}")
                          for letter in "BCDEFG" for suffix in ("png", "pdf")},
        "main_sha256": {suffix: _sha(main.with_suffix("."+suffix)) for suffix in ("png", "pdf")},
    }
    (panels / "Figure_4_panel_export_manifest.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(f"Figure 4 ready: {main.with_suffix('.png').relative_to(root)}")
    return audit


if __name__ == "__main__":
    render_package()
