"""Figure 4 mixed vector/raster renderer; no biological analysis is run here."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

# Figure parameters (all display choices are centralized here).
PAGE_MM = (210, 390)
LAYOUT_MM = {"C": (8, 300, 95, 80), "F": (108, 265, 94, 116),
             "E": (8, 210, 95, 84), "B": (125, 210, 65, 54),
             "D": (8, 115, 194, 88), "G": (8, 8, 194, 100)}
DPI = 300
PANEL_PT, TITLE_PT, AXIS_PT, SMALL_PT = 10, 8, 6, 5
C_POINT_AREA, C_ALPHA = .60, .95
F_BACKGROUND_AREA, F_POSITIVE_AREA = .12, .24
E_POINT_AREA, E_ALPHA = .22, .80
SEED = 42
N_CELLS = 69_391
GENES = ("EPCAM", "DCN", "RGS5", "CD2", "MZB1", "C1QA")
F_NAMES = ("Epithelial", "CAF", "PVL", "T cells", "Plasmablasts", "Myeloid")
MAJOR = ("Epithelial", "Endothelial", "CAFs", "PVL",
         "T-cells", "B-cells", "Plasmablasts", "Myeloid")
MAJOR_COLORS = ("#D55E00", "#0072B2", "#7A4E3A", "#009E73",
                "#008C95", "#6F4AA8", "#B23A73", "#CC3F93")
PURPLE = mcolors.LinearSegmentedColormap.from_list("marker", ["#ffffff", "#56227f"])
DOT_RED = mcolors.LinearSegmentedColormap.from_list(
    "reference_red", ["#ffffff", "#fee0d2", "#fc9272", "#ef3b2c", "#99000d"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": SMALL_PT,
                     "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none", "figure.facecolor": "white"})


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _axes(fig, page, box, part):
    x, y, w, h = box
    l, b, rw, rh = part
    return fig.add_axes([(x + l*w)/page[0], (y + b*h)/page[1],
                         rw*w/page[0], rh*h/page[1]])


def _text(fig, page, box, px, py, value, **kw):
    x, y, w, h = box
    fig.text((x+px*w)/page[0], (y+py*h)/page[1], value, **kw)


def _heading(fig, page, box, letter, title):
    _text(fig, page, box, .015, .985, f"{letter}  {title}",
          va="top", fontsize=PANEL_PT, fontweight="bold")


def _clean_umap(ax, xlabel=True):
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(.45)
    if xlabel:
        ax.set_xlabel("UMAP 1", fontsize=AXIS_PT, labelpad=1)
        ax.set_ylabel("UMAP 2", fontsize=AXIS_PT, labelpad=1)
    ax.set_aspect("equal", adjustable="box")


def _limits(xy):
    mid = xy.mean(axis=0)
    half = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))*.54
    return (mid[0]-half, mid[0]+half), (mid[1]-half, mid[1]+half)


def _vector_colorbar(fig, page, box, part, cmap, lo, hi, title, ticks=5):
    ax = _axes(fig, page, box, part)
    for i in range(64):
        y0 = lo+(hi-lo)*i/64
        ax.add_patch(Rectangle((0, y0), 1, (hi-lo)/64,
                               facecolor=cmap((i+.5)/64), edgecolor="none"))
    ax.set_xlim(0, 1); ax.set_ylim(lo, hi); ax.set_xticks([])
    ax.set_yticks(np.linspace(lo, hi, ticks)); ax.tick_params(labelsize=SMALL_PT, length=1, pad=1)
    ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
    ax.set_ylabel(title, fontsize=SMALL_PT, labelpad=2)
    for spine in ax.spines.values(): spine.set_linewidth(.35)


def _panel_c(fig, page, box, data, manifest):
    _heading(fig, page, box, "C", "SingleR cell-level broad labels")
    xy, labels = data["xy"], data["major"]
    palette = dict(zip(MAJOR, MAJOR_COLORS))
    palette["Ambiguous"] = "#8C8C8C"
    ax = _axes(fig, page, box, (.05, .13, .72, .74))
    order = np.random.default_rng(SEED).permutation(len(labels))
    ax.scatter(xy[order, 0], xy[order, 1], s=C_POINT_AREA,
               c=[palette[label] for label in labels[order]], alpha=C_ALPHA,
               linewidths=0, rasterized=True)
    ax.set_xlim(*data["xlim"]); ax.set_ylim(*data["ylim"])
    _clean_umap(ax)
    ax.legend(handles=[Patch(facecolor=palette[label], label=label)
                       for label in (*MAJOR, "Ambiguous") if label in set(labels)],
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
              fontsize=SMALL_PT, borderaxespad=0, handlelength=1)
    _text(fig, page, box, .05, .025, "Saved H-derived UMAP · cell-level SingleR predictions after pruning",
          fontsize=SMALL_PT, va="bottom")


def _panel_f(fig, page, box, data, manifest):
    _heading(fig, page, box, "F", "Lineage-marker expression")
    xy = data["xy"]
    for i, (gene, celltype) in enumerate(zip(GENES, F_NAMES)):
        col, row = i % 2, i // 2
        part = (.02 + col*.50, .66-row*.305, .46, .275)
        ax = _axes(fig, page, box, part)
        v = data["marker"][:, i]
        ax.scatter(xy[:, 0], xy[:, 1], s=F_BACKGROUND_AREA, c="#dedede",
                   linewidths=0, rasterized=True)
        positive = np.argsort(v, kind="stable")
        positive = positive[v[positive] > 0]
        ax.scatter(xy[positive, 0], xy[positive, 1], s=F_POSITIVE_AREA,
                   c=v[positive], cmap=PURPLE, vmin=0, vmax=float(data["vmax"][i]),
                   linewidths=0, rasterized=True)
        ax.set_xlim(*data["xlim"]); ax.set_ylim(*data["ylim"])
        _clean_umap(ax, xlabel=False)
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.set_title(f"{celltype} · {gene}", fontsize=AXIS_PT, pad=1)


def _panel_e(fig, page, box, data, manifest):
    _heading(fig, page, box, "E", "CNV score")
    xy, score, status = data["xy"], data["cnv"], data["cnv_status"]
    vmin, vmax = np.quantile(score, [.02, .98])
    ax = _axes(fig, page, box, (.04, .13, .47, .73))
    ax.scatter(xy[:, 0], xy[:, 1], c=score, cmap="viridis", vmin=vmin, vmax=vmax,
               s=E_POINT_AREA, alpha=E_ALPHA, linewidths=0, rasterized=True)
    ax.set_xlim(*data["xlim"]); ax.set_ylim(*data["ylim"])
    _clean_umap(ax)
    _vector_colorbar(fig, page, box, (.535, .24, .032, .53),
                     plt.get_cmap("viridis"), vmin, vmax, "CNV score")
    axv = _axes(fig, page, box, (.68, .19, .29, .64))
    groups = [score[status == x] for x in ("normal", "tumor")]
    assert tuple(map(len, groups)) == (27421, 41970)
    violin = axv.violinplot(groups, positions=[0, 1], widths=.60,
                            showmeans=False, showmedians=True, showextrema=True,
                            bw_method=.20)
    for body, color in zip(violin["bodies"], ("#56B4E9", "#D55E00")):
        body.set_facecolor(color); body.set_edgecolor("#333333")
        body.set_alpha(.78); body.set_linewidth(.65)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        violin[key].set_color("#222222"); violin[key].set_linewidth(.7)
    axv.set_xticks([0, 1], ["Normal", "Tumor"], fontsize=SMALL_PT)
    for label, color in zip(axv.get_xticklabels(), ("#56B4E9", "#D55E00")):
        label.set_color(color); label.set_fontweight("bold")
    # The vector colour bar immediately to the left carries the numeric score
    # scale; repeating ticks here would collide at print size.
    axv.set_yticks([])
    axv.set_xlim(-.35, 1.35); axv.set_ylim(-.0002, float(score.max())*1.08)
    axv.spines[["top", "right"]].set_visible(False)


def _panel_b(fig, page, box, data, manifest):
    _heading(fig, page, box, "B", "TNBC subtype")
    values = data["subtype_matrix"]
    limit = np.ceil(np.abs(values).max()*10)/10
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(-limit, limit)
    ax = _axes(fig, page, box, (.08, .19, .77, .66))
    for i in range(7):
        for j in range(6):
            value = values[i, j]
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1,
                                   facecolor=cmap(norm(value)), edgecolor="none"))
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=SMALL_PT,
                    color="white" if abs(value) > .30 else "black")
        call = data["subtype_calls"][i]
        if call in data["subtype_names"]:
            j = list(data["subtype_names"]).index(call)
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False,
                                   edgecolor="black", linewidth=.85))
    ax.set_xlim(-.5, 5.5); ax.set_ylim(6.5, -.5)
    ax.set_xticks(range(6), data["subtype_names"], fontsize=SMALL_PT)
    ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=1)
    ax.set_yticks(range(7), data["subtype_patients"], fontsize=SMALL_PT)
    ax.tick_params(axis="y", length=0, pad=1)
    for spine in ax.spines.values(): spine.set_linewidth(.4)
    _vector_colorbar(fig, page, box, (.89, .21, .04, .62), cmap,
                     -limit, limit, "Correlation")
    _text(fig, page, box, .09, .045, "Outlined cell: assigned subtype", fontsize=SMALL_PT)


def _panel_d(fig, page, box, data, manifest):
    _heading(fig, page, box, "D", "Wu-reference marker expression by SingleR broad label")
    values = data["dot"]
    ax = _axes(fig, page, box, (.145, .23, .72, .51))
    for pos in range(1, 41):
        subset = values[values.column_position == pos].set_index("predicted_major").loc[list(MAJOR)]
        pct = subset.pct_expressing.to_numpy()
        ax.scatter(np.full(8, pos-1), np.arange(8), s=190*(pct/100)**1.5,
                   c=subset.scaled_mean.to_numpy(), cmap=DOT_RED, vmin=0, vmax=1,
                   edgecolors="#777777", linewidths=.3)
    ax.set_xlim(-.6, 39.6); ax.set_ylim(7.6, -.6)
    ax.set_yticks(range(8), [f"{lab} (n={data['major_counts'][lab]:,})" for lab in MAJOR], fontsize=SMALL_PT)
    selected = values.drop_duplicates("column_position").sort_values("column_position")
    ax.set_xticks(range(40), selected.gene.tolist(), rotation=90, fontsize=SMALL_PT)
    ax.tick_params(length=0, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    for i, label in enumerate(MAJOR):
        left, right = i*5-.3, i*5+4.3
        ax.plot([left, left, right, right], [1.02, 1.07, 1.07, 1.02],
                transform=ax.get_xaxis_transform(), color="black", lw=.45, clip_on=False)
        ax.text(i*5+2, 1.10, label, transform=ax.get_xaxis_transform(),
                rotation=35, ha="left", va="bottom", fontsize=SMALL_PT)
    legend = _axes(fig, page, box, (.89, .44, .10, .25))
    legend.set_xlim(0, 1); legend.set_ylim(0, 1); legend.axis("off")
    legend.text(.5, .98, "Expressing\ncells (%)", ha="center", va="top", fontsize=SMALL_PT)
    for i, pct in enumerate((20, 40, 60, 80, 100)):
        legend.scatter(.13+i*.18, .48, s=190*(pct/100)**1.5,
                       c="#999999", edgecolors="#777777", linewidths=.3)
        legend.text(.13+i*.18, .12, str(pct), ha="center", fontsize=SMALL_PT)
    _vector_colorbar(fig, page, box, (.90, .22, .025, .18), DOT_RED, 0, 1,
                     "Scaled mean", ticks=3)
    _text(fig, page, box, .15, .035,
          "Dot size: fraction expressing >0 · colour: mean log expression, scaled per gene · descriptive support",
          fontsize=SMALL_PT)


def _panel_g(fig, page, box, data, manifest):
    _heading(fig, page, box, "G", "SingleR broad-label relative scores")
    labels = data["score_major"]
    centers = [np.flatnonzero(labels == lab).mean() for lab in MAJOR]
    scores = data["score_matrix"]
    xpart = (.12, .23, .80, .46)
    ax = _axes(fig, page, box, xpart)
    ax.imshow(scores, aspect="auto", interpolation="nearest", cmap="viridis",
              vmin=0, vmax=1, rasterized=True)
    ax.set_yticks(range(8), MAJOR, fontsize=SMALL_PT)
    ax.set_xticks(centers, MAJOR, rotation=35, ha="right", fontsize=SMALL_PT)
    ax.tick_params(length=0, pad=1)
    for edge in np.flatnonzero(labels[1:] != labels[:-1])+.5:
        ax.axvline(edge, color="white", linewidth=.25)
    palette = dict(zip(MAJOR, MAJOR_COLORS))
    legend_map = dict(manifest["patient_legend"])
    tracks = (("Predicted broad", labels, palette),
              ("Original cluster", data["score_cluster"],
               {i: plt.get_cmap("turbo")(i/20) for i in range(21)}),
              ("Patient", data["score_patient"], legend_map))
    for row, (title, items, palette) in enumerate(tracks):
        strip = _axes(fig, page, box, (.12, .80-row*.043, .80, .025))
        rgba = np.asarray([mcolors.to_rgba(palette[item]) for item in items], dtype="float32")
        strip.imshow(rgba[None, :, :], aspect="auto", interpolation="nearest", rasterized=True)
        strip.set_xticks([]); strip.set_yticks([])
        strip.set_ylabel(title, fontsize=SMALL_PT, rotation=0, ha="right", va="center", labelpad=2)
    for i in range(1, 8):
        x = .15 + (i-1)*.105
        fig.add_artist(Rectangle(((box[0]+x*box[2])/page[0],
                                  (box[1]+.91*box[3])/page[1]),
                                 .012, .007, transform=fig.transFigure,
                                 facecolor=legend_map[f"P{i}"], edgecolor="none"))
        _text(fig, page, box, x+.018, .918, f"P{i}", fontsize=SMALL_PT, va="center")
    _vector_colorbar(fig, page, box, (.95, .29, .015, .40),
                     plt.get_cmap("viridis"), 0, 1, "Relative score", ticks=3)
    _text(fig, page, box, .12, .025,
          "All 69,391 cells · ordered by predicted broad label, H cluster and cell ID · within-cell scaled and cubed, not probability",
          fontsize=SMALL_PT)


DRAW = {"B": _panel_b, "C": _panel_c, "D": _panel_d,
        "E": _panel_e, "F": _panel_f, "G": _panel_g}


def _save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("."+ext), dpi=DPI, facecolor="white")
    plt.close(fig)


def render_package(root: str | Path | None = None):
    root = Path(root or Path(__file__).resolve().parent).resolve()
    is_git = root.name == "Figure_4"
    stage = root / ("data/singler" if is_git else "data/Figure_4/singler")
    panels = root / ("outputs/panels" if is_git else "figs/individual/Figure_4")
    main = root / ("outputs/Figure_4_main_panel" if is_git else "figs/panel/Figure_4/Figure_4_main_panel")
    manifest = json.loads((stage / "manifest.json").read_text())
    for name, checksum in manifest["packaged_sha256"].items():
        if name != "manifest.json":
            assert _sha(stage / name) == checksum, name
    assert manifest["cells"] == N_CELLS and tuple(manifest["score_rows"]) == MAJOR
    with h5py.File(stage / "overlays.h5", "r") as f:
        decode = lambda name: np.asarray(f[name].asstr()[:], dtype=str)
        data = {"xy": f["umap"][:], "cnv": f["cnv_score"][:],
                "cnv_status": decode("cnv_status"), "marker": f["marker_expression"][:],
                "vmax": f["marker_vmax"][:], "major": decode("singler_major"),
                "cluster_major": decode("singler_cluster_major"),
                "cluster_id": f["singler_cluster_id"][:],
                "score_matrix": f["score_matrix"][:], "score_major": decode("score_major"),
                "score_cluster": f["score_cluster"][:], "score_patient": decode("score_patient"),
                "subtype_matrix": f["subtype_matrix"][:],
                "subtype_names": decode("subtype_names"),
                "subtype_patients": decode("subtype_patients"),
                "subtype_calls": decode("subtype_calls")}
        assert tuple(decode("marker_genes")) == GENES
    assert data["xy"].shape == (N_CELLS, 2)
    assert data["marker"].shape == (N_CELLS, len(GENES))
    assert data["score_matrix"].shape == (8, N_CELLS)
    assert list(data["subtype_patients"]) == [f"P{i}" for i in range(1, 8)]
    assert set(data["major"]) == set(MAJOR)
    assert set(data["cluster_major"]) <= set(MAJOR) | {"Ambiguous"}
    assert set(data["cluster_id"]) == set(range(21))
    assert all(len(set(data["cluster_major"][data["cluster_id"] == cluster])) == 1
               for cluster in range(21))
    assert set(data["score_major"]) == set(MAJOR)
    data["xlim"], data["ylim"] = _limits(data["xy"])
    data["major_counts"] = {lab: int(np.count_nonzero(data["score_major"] == lab)) for lab in MAJOR}
    data["dot"] = pd.read_csv(stage / "D_values.csv")
    assert len(data["dot"]) == 320
    panels.mkdir(parents=True, exist_ok=True)
    main.parent.mkdir(parents=True, exist_ok=True)
    for letter in "BCDEFG":
        _, _, w, h = LAYOUT_MM[letter]
        fig = plt.figure(figsize=(w/25.4, h/25.4), facecolor="white")
        DRAW[letter](fig, (w, h), (0, 0, w, h), data, manifest)
        _save(fig, panels / f"Figure_4_{letter}")
    fig = plt.figure(figsize=(PAGE_MM[0]/25.4, PAGE_MM[1]/25.4), facecolor="white")
    for letter in "BCDEFG":
        DRAW[letter](fig, PAGE_MM, LAYOUT_MM[letter], data, manifest)
    _save(fig, main)
    if is_git:
        for suffix, name in (("png", "Figure_4_preview.png"),
                             ("pdf", "Figure_4.pdf"), ("svg", "Figure_4.svg")):
            shutil.copy2(main.with_suffix("."+suffix), root / "outputs" / name)
    audit = {"cells": N_CELLS, "panel_order": list("BCDEFG"), "panels": list("BCDEFG"), "formats": ["png", "pdf", "svg"],
             "rendering": "matplotlib fallback; dense UMAP and score images rasterized inside editable PDF/SVG",
             "python": platform.python_version(), "matplotlib": matplotlib.__version__,
             "C_point_area_pt2": C_POINT_AREA, "C_alpha": C_ALPHA,
             "C_call_level": "full Wu-atlas SingleR major label per cell after default pruning; cancer/normal epithelial collapsed for broad display",
             "F_labels": dict(zip(GENES, F_NAMES)),
             "B_patient_alignment": "Subtype rows sorted by original patient ID to P1-P7 before display",
             "main_sha256": {ext: _sha(main.with_suffix("."+ext)) for ext in ("png", "pdf", "svg")}}
    (panels / "Figure_4_panel_export_manifest.json").write_text(json.dumps(audit, indent=2)+"\n")
    print("Figure 4 ready:", main.with_suffix(".pdf").relative_to(root))
    return audit


if __name__ == "__main__":
    render_package()
