from pathlib import Path
import json, math, platform, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patches

# ─── Figure parameters (user-editable) ───
FIGURE_WIDTH_MM = 457
FIGURE_HEIGHT_MM = 508
PANEL_WIDTH_IN = 7.1
PANEL_HEIGHT_IN = 5.3
TEXT_SIZE_PT = 8
AXIS_TITLE_SIZE_PT = 9
PANEL_LABEL_SIZE_PT = 10
TICK_LABEL_SIZE_PT = 7
LEGEND_SIZE_PT = 7
UMAP_POINT_SIZE = 0.18
UMAP_ALPHA = 0.78
OUTPUT_DPI = 300
RASTERIZED = True
LOW_PERCENTILE = 2
HIGH_PERCENTILE = 98
CELLCHAT_COLOR = "#F8766D"
SCKNIFE_COLOR = "#00BA38"
SCATTER_COLOR = "#4DBBD5"
NETWORK_COLOR = "#808080"
NETWORK_REFERENCE_AXES_WIDTH_IN = 5.5025
NETWORK_EDGE_WIDTH_MIN_PT = 0.5
NETWORK_EDGE_WIDTH_MAX_PT = 5.0
NETWORK_NODE_SIZE_PT2 = 900
NETWORK_NODE_BORDER_WIDTH_PT = 1.0
NETWORK_ARROW_SIZE_PT = 10
NETWORK_AXIS_LIMIT = 1.24
NETWORK_MAJOR_LABEL_SIZE_PT = 7.0
NETWORK_MINOR_LABEL_SIZE_PT = 4.2
HEATMAP_LABEL_SIZE_PT = 4.5
HEATMAP_GRID_WIDTH_PT = 0.2
HEATMAP_SMALL_LABEL_SIZE_PT = 5.2
EVIDENCE_MAJOR_COLOR = "#0072B2"
EVIDENCE_MINOR_COLOR = "#CC79A7"
STORY_PATHWAYS = ["COLLAGEN", "FN1", "THBS", "SPP1", "PERIOSTIN", "PDGF"]
TOP_PAIR_COUNT = 5
PAIR_HIGHLIGHT_COLOR = "#6B3E2E"
PAIR_OTHER_COLOR = "#B8B8B8"
PAIR_BAR_HEIGHT = .52
PAIR_LABEL_Y_OFFSET = -.31
PAIR_SCORE_MARGIN = 1.20
PAIR_PANEL_SPACING = .30
UMAP_CMAP = "viridis"
SPECIMEN_CMAP = "RdBu_r"
SPECIMEN_Z_LIMIT = 2.0
EF_CMAP_COLORS = ("#F2FAF9", "#B8E1DD", "#5AB4AC", "#1F7893", "#0B2E4F")
RANDOM_SEED = 42
# ─────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DATA = ROOT / ("data" if ROOT.name == "Figure_7" else "data/Figure_7")
INDIVIDUAL = ROOT / ("outputs/panels" if ROOT.name == "Figure_7" else "figs/individual/Figure_7")
COMPILED = ROOT / ("outputs" if ROOT.name == "Figure_7" else "figs/panel/Figure_7")
INDIVIDUAL.mkdir(parents=True, exist_ok=True)
COMPILED.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": TEXT_SIZE_PT, "axes.titlesize": PANEL_LABEL_SIZE_PT,
    "axes.labelsize": AXIS_TITLE_SIZE_PT, "xtick.labelsize": TICK_LABEL_SIZE_PT,
    "ytick.labelsize": TICK_LABEL_SIZE_PT, "legend.fontsize": LEGEND_SIZE_PT,
    "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
np.random.seed(RANDOM_SEED)
EF_CMAP = mpl.colors.LinearSegmentedColormap.from_list("teal_navy", EF_CMAP_COLORS)

MAJOR_ORDER = [
    "Epithelial", "Endothelial", "CAFs", "PVL", "T-cells", "B-cells",
    "Plasmablasts", "Myeloid",
]
MAJOR_COLORS = {
    "Epithelial": "#D55E00", "Endothelial": "#0072B2", "CAFs": "#7A4E3A",
    "PVL": "#009E73", "T-cells": "#008C95", "B-cells": "#6F4AA8",
    "Plasmablasts": "#B23A73", "Myeloid": "#CC3F93",
}
MINOR_ORDER = [
    "Cancer Basal SC", "Cancer Cycling", "Cancer Her2 SC", "Cancer LumA SC",
    "Cancer LumB SC", "Luminal Progenitors", "Mature Luminal", "Myoepithelial",
    "CAFs MSC iCAF-like", "CAFs myCAF-like", "Cycling PVL", "PVL Differentiated",
    "PVL Immature", "Endothelial ACKR1", "Endothelial CXCL12",
    "Endothelial Lymphatic LYVE1", "Endothelial RGS5", "B cells Memory",
    "B cells Naive", "Cycling T-cells", "NK cells", "NKT cells", "T cells CD4+",
    "T cells CD8+", "Cycling_Myeloid", "DCs", "Macrophage", "Monocyte",
    "Plasmablasts",
]
MINOR_COLORS = {
    "Cancer Basal SC": "#8C2D04", "Cancer Cycling": "#D94801", "Cancer Her2 SC": "#F16913",
    "Cancer LumA SC": "#FD8D3C", "Cancer LumB SC": "#FDAE6B", "Luminal Progenitors": "#FDD0A2",
    "Mature Luminal": "#E6550D", "Myoepithelial": "#A63603", "CAFs MSC iCAF-like": "#9C755F",
    "CAFs myCAF-like": "#6B3E2E", "Cycling PVL": "#006D2C", "PVL Differentiated": "#31A354",
    "PVL Immature": "#74C476", "Endothelial ACKR1": "#08519C", "Endothelial CXCL12": "#3182BD",
    "Endothelial Lymphatic LYVE1": "#6BAED6", "Endothelial RGS5": "#9ECAE1",
    "B cells Memory": "#54278F", "B cells Naive": "#756BB1", "Cycling T-cells": "#006D6F",
    "NK cells": "#238B8E", "NKT cells": "#41B6C4", "T cells CD4+": "#7FCDBB",
    "T cells CD8+": "#225EA8", "Cycling_Myeloid": "#7A0177", "DCs": "#AE017E",
    "Macrophage": "#DD3497", "Monocyte": "#F768A1", "Plasmablasts": "#B23A73",
}
PANEL_TITLES = {
    "A": "Analysis overview", "B": "Filtered pathway overlap", "C": "Filtered LR-pair overlap",
    "D": "Selected pathway enrichment: major and minor", "E": "Convergent matrix-associated patterns",
    "F": "Complementary SPP1 signaling", "G": "Specimen-level transcript support",
    "H": "COL6A3 activity", "I": "CD44 expression",
    "J": "Filtered log10 score concordance",
}
MANIFEST = json.loads((DATA / "manifest.json").read_text())
pathway = pd.read_csv(DATA / "pathway_overlap.csv")
lr_overlap = pd.read_csv(DATA / "lr_harmonized_overlap.csv")
strength_correlation = pd.read_csv(DATA / "strength_correlation_filtered.csv")
strength_statistics = pd.read_csv(DATA / "strength_correlation_statistics.csv").set_index("scale")
collagen_major_matrix = pd.read_csv(DATA / "collagen_major_communication_matrix.csv", index_col=0).loc[MAJOR_ORDER, MAJOR_ORDER]
collagen_minor_matrix = pd.read_csv(DATA / "collagen_minor_communication_matrix_lr_q05.csv", index_col=0).loc[MINOR_ORDER, MINOR_ORDER]
collagen_interactions = pd.read_csv(DATA / "collagen_top25_lr_pairs.csv")
umap_scores = pd.read_parquet(DATA / "figure4_umap_activity_expression.parquet")
specimen_support = pd.read_csv(DATA / "specimen_expression_support_matrix_spp1.csv")
pathway_evidence = pd.read_csv(DATA / "pathway_evidence_major_minor.csv")
story_matrices = {
    name: pd.read_csv(DATA / f"{name.lower()}_minor_communication_matrix_lr_q05.csv", index_col=0).loc[MINOR_ORDER, MINOR_ORDER]
    for name in ["COLLAGEN", "FN1", "THBS", "SPP1"]
}

print(
    f"Loaded Figure 7 bundle: {len(umap_scores):,} cells; "
    f"{len(MAJOR_ORDER)} major and {len(MINOR_ORDER)} minor network groups"
)
if not all(MANIFEST["checks"].values()):
    raise ValueError("Figure 7 data bundle failed extraction checks")


def clean_axis(ax, keep_left=True, keep_bottom=True):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_visible(keep_bottom)

def panel_title(ax, label):
    ax.set_title(f"{label}  {PANEL_TITLES[label]}", loc="left", fontweight="bold", fontsize=PANEL_LABEL_SIZE_PT, pad=8)

def draw_workflow(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        (0.03, 0.58, 0.25, 0.23, "CellChat", "Overlap/concordance\ncomparator (B–C, J)", CELLCHAT_COLOR),
        (0.03, 0.18, 0.25, 0.23, "scKNIFE", "Primary signaling\nanalysis", SCKNIFE_COLOR),
        (0.38, 0.38, 0.25, 0.23, "Assessment", "Agreement +\npathway evidence", "#6C757D"),
        (0.72, 0.58, 0.25, 0.23, "Comparison", "Sensitivity checks\n(B–C, J)", "#0072B2"),
        (0.72, 0.18, 0.25, 0.23, "Matrix + SPP1", "Saved scKNIFE +\ntranscript support", "#8C2D04"),
    ]
    for x, y, w, h, head, body, color in boxes:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015", facecolor="white", edgecolor=color, linewidth=1.8)
        ax.add_patch(box); ax.text(x+w/2, y+h*0.67, head, ha="center", va="center", weight="bold", color=color)
        ax.text(x+w/2, y+h*0.30, body, ha="center", va="center", fontsize=TICK_LABEL_SIZE_PT)
    for start, end in [((.28,.695),(.38,.515)),((.28,.295),(.38,.495)),((.63,.52),(.72,.695)),((.63,.47),(.72,.295))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#4D4D4D"))
    ax.text(.03,.04,"69,319 minor-tier cells · pooled scKNIFE predictions",fontsize=TICK_LABEL_SIZE_PT)
    panel_title(ax, "A")

def draw_venn(ax, counts, label, subtitle):
    left, shared, right = counts
    total_left, total_right = left + shared, right + shared
    max_total = max(total_left, total_right)
    r_left = 0.26 * math.sqrt(total_left / max_total)
    r_right = 0.26 * math.sqrt(total_right / max_total)
    cx_left, cx_right, cy = 0.39, 0.59, 0.49
    ax.add_patch(patches.Circle((cx_left, cy), r_left, facecolor=CELLCHAT_COLOR, alpha=.55, edgecolor="#222222", lw=1))
    ax.add_patch(patches.Circle((cx_right, cy), r_right, facecolor=SCKNIFE_COLOR, alpha=.55, edgecolor="#222222", lw=1))
    ax.text(cx_left-r_left*.65, cy, f"{left}", ha="center", va="center", fontsize=AXIS_TITLE_SIZE_PT)
    ax.text((cx_left+cx_right)/2, cy, f"{shared}", ha="center", va="center", fontsize=AXIS_TITLE_SIZE_PT, weight="bold")
    ax.text(cx_right+r_right*.55, cy, f"{right}", ha="center", va="center", fontsize=AXIS_TITLE_SIZE_PT)
    left_label = "CellChat\nLR p<0.05" if label in {"B", "C"} else "CellChat"
    right_label = "scKNIFE\nBH q<0.05" if label in {"B", "C"} else "scKNIFE"
    ax.text(.20, .12, left_label, ha="center", weight="bold", fontsize=LEGEND_SIZE_PT)
    ax.text(.80, .12, right_label, ha="center", weight="bold", fontsize=LEGEND_SIZE_PT)
    jaccard = shared / (left + shared + right)
    ax.text(.5, .91, f"{subtitle}\nJaccard = {jaccard:.3f}", ha="center", va="top", fontsize=TEXT_SIZE_PT)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal"); ax.axis("off"); panel_title(ax, label)

def draw_network(ax, matrix, order, colors, label, label_size):
    graph = nx.DiGraph()
    graph.add_nodes_from(order)
    top = matrix.stack().sort_values(ascending=False)
    top = top.loc[top > 0].head(40)
    for (source, target), weight in top.items(): graph.add_edge(source, target, weight=float(weight))
    pos = nx.circular_layout(graph)
    weights = np.array([data["weight"] for _,_,data in graph.edges(data=True)])
    lo, hi = np.percentile(weights, [LOW_PERCENTILE, HIGH_PERCENTILE]); denom=max(hi-lo,np.finfo(float).eps)
    axes_width_in = ax.get_window_extent().width / ax.figure.dpi
    geometry_scale = min(1.0, axes_width_in / NETWORK_REFERENCE_AXES_WIDTH_IN)
    widths = geometry_scale * (
        NETWORK_EDGE_WIDTH_MIN_PT
        + (NETWORK_EDGE_WIDTH_MAX_PT - NETWORK_EDGE_WIDTH_MIN_PT)
        * np.clip((weights-lo)/denom,0,1)
    )
    nx.draw_networkx_nodes(graph,pos,node_color=[colors[x] for x in graph.nodes()],node_size=NETWORK_NODE_SIZE_PT2*geometry_scale**2,edgecolors="white",linewidths=NETWORK_NODE_BORDER_WIDTH_PT*geometry_scale,ax=ax)
    nx.draw_networkx_labels(graph,pos,font_size=label_size,font_weight="bold",ax=ax)
    nx.draw_networkx_edges(graph,pos,width=widths,alpha=.55,arrows=True,arrowsize=NETWORK_ARROW_SIZE_PT*geometry_scale,connectionstyle="arc3,rad=0.08",edge_color=NETWORK_COLOR,ax=ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-NETWORK_AXIS_LIMIT, NETWORK_AXIS_LIMIT)
    ax.set_ylim(-NETWORK_AXIS_LIMIT, NETWORK_AXIS_LIMIT)
    ax.axis("off"); panel_title(ax,label)

def draw_heatmap(ax):
    matrix = collagen_minor_matrix.to_numpy()
    vmin, vmax = float(matrix.min()), float(matrix.max())
    image = ax.pcolormesh(
        np.arange(matrix.shape[1] + 1), np.arange(matrix.shape[0] + 1), matrix,
        cmap=EF_CMAP, vmin=vmin, vmax=vmax, shading="flat", edgecolors="white",
        linewidth=HEATMAP_GRID_WIDTH_PT, rasterized=False,
    )
    ax.set_xlim(0, matrix.shape[1]); ax.set_ylim(matrix.shape[0], 0); ax.set_aspect("auto")
    ax.set_xticks(np.arange(len(MINOR_ORDER)) + .5, MINOR_ORDER, rotation=90)
    ax.set_yticks(np.arange(len(MINOR_ORDER)) + .5, MINOR_ORDER)
    ax.tick_params(labelsize=HEATMAP_LABEL_SIZE_PT, length=1.5)
    ax.set_xlabel("Receiver"); ax.set_ylabel("Sender")
    colorbar = plt.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Summed weighted score (LR q<0.05)")
    colorbar.solids.set_rasterized(False)
    panel_title(ax, "E")

def draw_lollipop(ax):
    agg=collagen_interactions.sort_values("weighted_score").copy()
    agg["pair"]=agg["ligand"] + " → " + agg["receptor"]
    colors=EF_CMAP(np.linspace(.20,.95,len(agg)))
    ax.barh(np.arange(len(agg)),agg.weighted_score,color=colors,height=.78,edgecolor="none")
    maximum=float(agg.weighted_score.max())
    ax.set_xlim(-.38*maximum,1.05*maximum)
    ax.set_yticks([])
    for row,label in enumerate(agg.pair):
        ax.text(-.36*maximum,row,label,ha="left",va="center",fontsize=TICK_LABEL_SIZE_PT)
    ax.set_xlabel("Summed weighted score")
    clean_axis(ax); panel_title(ax,"F")


def draw_pathway_evidence(ax):
    for tier, offset, marker, color in [("major", -.14, "o", EVIDENCE_MAJOR_COLOR),
                                         ("minor", .14, "D", EVIDENCE_MINOR_COLOR)]:
        table = pathway_evidence.loc[pathway_evidence.tier.eq(tier)].set_index("pathway").loc[STORY_PATHWAYS]
        y = np.arange(len(table)) + offset
        ax.scatter(table.observed_to_null, y, marker=marker, color=color, s=35, label=tier.capitalize(), zorder=3)
        for value, ypos in zip(table.observed_to_null, y):
            ax.text(value + .035, ypos, f"{value:.2f}×", va="center", fontsize=TICK_LABEL_SIZE_PT, color=color)
    ax.axvline(1, color="#777777", ls="--", lw=.8)
    ax.set_yticks(np.arange(len(STORY_PATHWAYS)), STORY_PATHWAYS)
    ax.set_ylim(len(STORY_PATHWAYS)-.5, -.6)
    ax.set_xlim(.9, 4.25)
    ax.set_xlabel("Observed / permutation-null mean pathway score")
    ax.legend(frameon=False, loc="upper right", title="All shown: pathway BH q<0.05")
    clean_axis(ax)
    panel_title(ax, "D")
    ax.text(1, 1.02, "Six selected candidates; not a top-six ranking", ha="right", transform=ax.transAxes, fontsize=TICK_LABEL_SIZE_PT)


def draw_pathway_heatmap(ax, pathway_name, show_y=True):
    values = np.log1p(story_matrices[pathway_name].to_numpy())
    mesh = ax.pcolormesh(np.arange(30), np.arange(30), values, cmap=EF_CMAP,
                        vmin=float(values.min()), vmax=float(values.max()), shading="flat",
                        linewidth=HEATMAP_GRID_WIDTH_PT, edgecolors="white", rasterized=False)
    ax.set_xlim(0, 29); ax.set_ylim(29, 0)
    ax.set_xticks(np.arange(29)+.5, MINOR_ORDER, rotation=90)
    ax.set_yticks(np.arange(29)+.5, MINOR_ORDER if show_y else [""]*29)
    ax.tick_params(labelsize=HEATMAP_SMALL_LABEL_SIZE_PT, length=1)
    ax.set_xlabel("Receiver", fontsize=TEXT_SIZE_PT)
    ax.set_ylabel("Sender" if show_y else "", fontsize=TEXT_SIZE_PT)
    n_keys = int(pathway_evidence.loc[pathway_evidence.tier.eq("minor") & pathway_evidence.pathway.eq(pathway_name), "n_retained_lr_keys"].iloc[0])
    ax.set_title(f"{pathway_name} · {n_keys} LR keys (q<0.05)", loc="left", fontsize=AXIS_TITLE_SIZE_PT, fontweight="bold")
    bar = ax.figure.colorbar(mesh, ax=ax, location="top", fraction=.035, pad=.07, aspect=30)
    bar.set_label("log1p(sum weight)", fontsize=TICK_LABEL_SIZE_PT, labelpad=3)
    bar.ax.tick_params(labelsize=TICK_LABEL_SIZE_PT, length=2)
    bar.solids.set_rasterized(False)
    return mesh


def top_pathway_pairs(pathway_name):
    """Rank all 29×29 saved entries, including self-pairs; never preselect a cell type."""
    matrix = story_matrices[pathway_name]
    table = matrix.rename_axis(index="source", columns="target").stack().rename("weighted_score").reset_index()
    table = table.loc[table.weighted_score.gt(0)].sort_values(
        ["weighted_score", "source", "target"], ascending=[False, True, True], kind="stable"
    ).head(TOP_PAIR_COUNT).reset_index(drop=True)
    table["highlight"] = table.source.eq("CAFs myCAF-like") & table.target.eq("Cancer Basal SC")
    return table


def draw_pathway_top_pairs(ax, pathway_name):
    table = top_pathway_pairs(pathway_name)
    colors = np.where(table.highlight, PAIR_HIGHLIGHT_COLOR, PAIR_OTHER_COLOR)
    positions = np.arange(len(table))
    bars = ax.barh(positions, table.weighted_score, height=PAIR_BAR_HEIGHT, color=colors, edgecolor="none")
    display = {"CAFs myCAF-like": "myCAF", "Cancer Basal SC": "basal-labelled epithelial",
               "Cancer Cycling": "cycling-labelled epithelial"}
    for row, bar in zip(table.itertuples(), bars):
        label = f"{display.get(row.source, row.source)} → {display.get(row.target, row.target)}"
        ax.text(0, row.Index + PAIR_LABEL_Y_OFFSET, label, va="bottom", ha="left",
                fontsize=TICK_LABEL_SIZE_PT, color=PAIR_HIGHLIGHT_COLOR if row.highlight else "#444444",
                fontweight="bold" if row.highlight else "normal")
        ax.annotate(f"{row.weighted_score:,.1f}", (row.weighted_score, row.Index), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=TICK_LABEL_SIZE_PT)
    ax.set_ylim(len(table)-.5, -.85)
    ax.set_xlim(0, float(table.weighted_score.max())*PAIR_SCORE_MARGIN)
    ax.set_yticks([])
    ax.set_xlabel("Summed scKNIFE weight (linear)", fontsize=TEXT_SIZE_PT)
    ax.set_title(f"{pathway_name} · top {TOP_PAIR_COUNT} sender–receiver pairs\nPooled LR-key BH q<0.05", loc="left",
                 fontsize=AXIS_TITLE_SIZE_PT, fontweight="bold", pad=12)
    clean_axis(ax, keep_left=False)


def draw_matrix_triptych(ax):
    fig = ax.figure
    spec = ax.get_subplotspec().subgridspec(1, 3, wspace=PAIR_PANEL_SPACING)
    ax.remove()
    for i, name in enumerate(["COLLAGEN", "FN1", "THBS"]):
        child = fig.add_subplot(spec[0, i])
        if name == "COLLAGEN":
            draw_pathway_heatmap(child, name)
        else:
            draw_pathway_top_pairs(child, name)
        if i == 0:
            child.text(-.02, 1.30, f"E  {PANEL_TITLES['E']}", transform=child.transAxes,
                       fontsize=PANEL_LABEL_SIZE_PT, fontweight="bold", ha="left")


def draw_spp1(ax):
    draw_pathway_heatmap(ax, "SPP1")
    panel_title(ax, "F")

def draw_strength_correlation(ax):
    x = np.log10(strength_correlation["cellchat_prob"].to_numpy())
    y = np.log10(strength_correlation["scknife_score"].to_numpy())
    summary = strength_statistics.loc["log10"]
    slope, intercept = np.polyfit(x, y, 1)
    order = np.argsort(strength_correlation["normalized_lr_key"].astype(str).to_numpy(), kind="stable")
    ax.scatter(x[order], y[order], s=4, alpha=.25, color=SCATTER_COLOR, edgecolors="none", rasterized=RASTERIZED)
    extent = np.array([x.min(), x.max()])
    ax.plot(extent, slope * extent + intercept, "--", color="#555555", lw=1.1)
    ax.text(
        .02, .98,
        f"n = {len(x):,}\nPearson r = {summary.pearson_r:.3f}\nSpearman ρ = {summary.spearman_r:.3f}",
        transform=ax.transAxes, va="top", fontsize=TICK_LABEL_SIZE_PT,
    )
    ax.set_xlabel("log10(CellChat probability)")
    ax.set_ylabel("log10(scKNIFE weighted score)")
    clean_axis(ax); panel_title(ax, "J")

def draw_umap(ax, column, label, colorbar_label):
    values=umap_scores[column].to_numpy(); lo,hi=np.percentile(values,[LOW_PERCENTILE,HIGH_PERCENTILE])
    order=np.argsort(values); x=umap_scores["umap_1"].to_numpy()[order]; y=umap_scores["umap_2"].to_numpy()[order]
    plotted=np.clip(values[order],lo,hi)
    points=ax.scatter(x,y,c=plotted,cmap=UMAP_CMAP,s=UMAP_POINT_SIZE,alpha=UMAP_ALPHA,linewidths=0,rasterized=RASTERIZED)
    plt.colorbar(points,ax=ax,fraction=.046,pad=.04,label=colorbar_label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Figure 4 UMAP 1"); ax.set_ylabel("Figure 4 UMAP 2"); clean_axis(ax); panel_title(ax,label)

SPECIMEN_FEATURE_ORDER = [
    "COL1A1 in myCAF", "COL1A2 in myCAF", "COL6A3 in myCAF",
    "CD44 in basal-labelled", "ITGAV in basal-labelled",
    "ITGB8 in basal-labelled", "SDC4 in basal-labelled",
    "SPP1 in macrophage", "CD44 in myCAF",
]


def _specimen_layout(table):
    collection_order = {"Bx": 0, "Surgery": 1, "Recurrence": 2}
    metadata = table.drop_duplicates("specimen").copy()
    metadata["patient_rank"] = metadata.display_id.str.removeprefix("P").astype(int)
    metadata["collection_rank"] = metadata.collection.map(collection_order)
    metadata = metadata.sort_values(["patient_rank", "collection_rank", "specimen"]).reset_index(drop=True)
    specimen_order = metadata.specimen.tolist()
    x_lookup = {name: i for i, name in enumerate(specimen_order)}
    y_lookup = {name: i for i, name in enumerate(SPECIMEN_FEATURE_ORDER)}
    plotted = table.copy()
    plotted["x"] = plotted.specimen.map(x_lookup)
    plotted["y"] = plotted.feature_label.map(y_lookup)
    return metadata, plotted


def _decorate_specimen_axis(ax, metadata):
    status_colors = {"Non-metastatic": "#4DAF4A", "Metastatic": "#E41A1C"}
    site_colors = {"Bx": "#377EB8", "Surgery": "#E41A1C", "Recurrence": "#984EA3"}
    for row in metadata.itertuples():
        if not bool(row.eligible):
            ax.axvspan(row.Index - .48, row.Index + .48, color="#EFEFEF", zorder=0)
        ax.add_patch(patches.Rectangle((row.Index-.45, -.82), .9, .16, color=status_colors[row.metastatic_status], ec="none"))
        ax.add_patch(patches.Rectangle((row.Index-.45, -.57), .9, .16, color=site_colors[row.collection], ec="none"))
    ax.text(-1.55, -.74, "Status", ha="left", va="center", fontsize=TICK_LABEL_SIZE_PT)
    ax.text(-1.55, -.49, "Site", ha="left", va="center", fontsize=TICK_LABEL_SIZE_PT)

    starts = [(display_id, int(group.index.min()), int(group.index.max())) for display_id, group in metadata.groupby("display_id", sort=False)]
    for display_id, start, stop in starts:
        ax.text((start + stop) / 2, -1.01, display_id, ha="center", va="center", fontsize=6.5, fontweight="bold")
        if stop < len(metadata)-1:
            ax.axvline(stop + .5, color="#BDBDBD", lw=.6, zorder=1)

    tick_labels = {"Bx": "Bx", "Surgery": "Surg.", "Recurrence": "Rec."}
    ax.set_xticks(np.arange(len(metadata)), [tick_labels[value] for value in metadata.collection], rotation=55, ha="right")
    ax.set_yticks(np.arange(len(SPECIMEN_FEATURE_ORDER)), SPECIMEN_FEATURE_ORDER)
    ax.set_xlim(-1.65, len(metadata)-.5); ax.set_ylim(len(SPECIMEN_FEATURE_ORDER)-.45, -1.15)
    clean_axis(ax, keep_left=False, keep_bottom=True)
    ax.tick_params(axis="y", length=0)
    ax.text(.995, 1.01, "green: non-metastatic   red: metastatic", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2)


def draw_specimen_support(ax):
    metadata, table = _specimen_layout(specimen_support)
    table["mean_z"] = table.groupby("feature_label")["mean_log1p"].transform(
        lambda values: (values - values.mean()) / max(values.std(ddof=0), np.finfo(float).eps)
    ).clip(-SPECIMEN_Z_LIMIT, SPECIMEN_Z_LIMIT)

    _decorate_specimen_axis(ax, metadata)
    eligible = table.n_cells.ge(10) & table.mean_z.notna()
    points = ax.scatter(
        table.loc[eligible, "x"], table.loc[eligible, "y"],
        s=18 + 1.8 * table.loc[eligible, "pct_detected"],
        c=table.loc[eligible, "mean_z"], cmap=SPECIMEN_CMAP,
        vmin=-SPECIMEN_Z_LIMIT, vmax=SPECIMEN_Z_LIMIT,
        edgecolors="#333333", linewidths=.25, rasterized=False, zorder=3,
    )
    missing = table.loc[~eligible]
    ax.scatter(missing.x, missing.y, marker="x", s=34, color="#777777", linewidths=1.0, zorder=4)

    colorbar = plt.colorbar(points, ax=ax, fraction=.025, pad=.02)
    colorbar.set_label("Row z-score\n(mean log1p)")
    colorbar.solids.set_rasterized(False)
    handles = [ax.scatter([], [], s=18 + 1.8*pct, facecolor="white", edgecolor="#333333", linewidth=.4, label=f"{pct}%") for pct in (25, 50, 100)]
    handles.insert(0, ax.scatter([], [], s=18, facecolor="white", edgecolor="#333333", linewidth=.4, label="0%"))
    ax.legend(handles=handles, title="Cells detected", frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(.5, -0.30), columnspacing=.8, handletextpad=.2)
    ax.text(.5, -.34, "Gray: <10 myCAF or basal cells; ×: row compartment <10 cells", transform=ax.transAxes,
            ha="center", fontsize=TICK_LABEL_SIZE_PT)
    panel_title(ax, "G")

def draw_panel(label, ax):
    if label=="A": draw_workflow(ax)
    elif label=="B":
        c=pathway.status.value_counts(); draw_venn(ax,(int(c.cellchat_only),int(c.shared),int(c.scknife_only)),"B",f"{int(c.shared)} shared pathways")
    elif label=="C":
        c=lr_overlap.status.value_counts(); draw_venn(ax,(int(c.get("cellchat_only",0)),int(c.get("shared",0)),int(c.get("scknife_only",0))),"C",f"{int(c.get('shared',0))} shared LR pairs")
    elif label=="D": draw_pathway_evidence(ax)
    elif label=="E": draw_matrix_triptych(ax)
    elif label=="F": draw_spp1(ax)
    elif label=="G": draw_specimen_support(ax)
    elif label=="H": draw_umap(ax,"COL6A3_activity","H","Inferred activity")
    elif label=="I": draw_umap(ax,"CD44_expression","I","log1p-normalized expression")
    elif label=="J": draw_strength_correlation(ax)
    else: raise KeyError(label)

def make_figure7_main_panel(label):
    width = 16 if label == "E" else PANEL_WIDTH_IN
    height = 7.4 if label in {"E", "F", "G"} else PANEL_HEIGHT_IN
    fig,ax=plt.subplots(figsize=(width,height)); draw_panel(label,ax); fig.tight_layout(); return fig

def _save_live_vector_svg(fig,path,bbox):
    fig.savefig(path,format="svg",dpi=OUTPUT_DPI,bbox_inches=bbox)

def save_individual(label, fig):
    png=INDIVIDUAL/f"Figure_7_{label}.png"; pdf=INDIVIDUAL/f"Figure_7_{label}.pdf"; svg=INDIVIDUAL/f"Figure_7_{label}.svg"
    fig.savefig(png,format="png",dpi=OUTPUT_DPI,bbox_inches="tight")
    fig.savefig(pdf,format="pdf",dpi=OUTPUT_DPI,bbox_inches="tight")
    fig.savefig(svg,format="svg",dpi=OUTPUT_DPI,bbox_inches="tight")
    return png,pdf,svg
