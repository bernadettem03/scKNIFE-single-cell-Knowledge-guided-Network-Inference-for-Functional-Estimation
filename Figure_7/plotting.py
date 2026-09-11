from pathlib import Path
import json, math, platform, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap

# ─── Figure parameters (user-editable) ───
FIGURE_WIDTH_MM = 356
FIGURE_HEIGHT_MM = 490
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
HEATMAP_VALUE_FORMAT = ".0f"
HEATMAP_ANNOTATION_SIZE = 5
SCATTER_COLOR = "#4DBBD5"
CELLCHAT_COLOR = "#F8766D"
SCKNIFE_COLOR = "#00BA38"
NETWORK_COLOR = "#808080"
NETWORK_REFERENCE_AXES_WIDTH_IN = 5.5025
NETWORK_EDGE_WIDTH_MIN_PT = 0.5
NETWORK_EDGE_WIDTH_MAX_PT = 5.0
NETWORK_NODE_SIZE_PT2 = 900
NETWORK_NODE_BORDER_WIDTH_PT = 1.0
NETWORK_ARROW_SIZE_PT = 10
VEGF_CMAP = LinearSegmentedColormap.from_list("white_red", ["#FFFFFF", "#B2182B"])
UMAP_CMAP = "viridis"
RANDOM_SEED = 42
# ─────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INDIVIDUAL = ROOT / "outputs/panels"
COMPILED = ROOT / "outputs"
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

CELL_ORDER = [
    "Epithelial progenitor", "CAFs", "T cells", "IFN-response epithelial",
    "Macrophages", "Epithelial luminal", "Endothelial", "Epithelial cycling", "Epithelial basal",
]
CELL_COLORS = {
    "Epithelial progenitor": "#CC79A7", "CAFs": "#D55E00", "T cells": "#9467BD",
    "IFN-response epithelial": "#8C564B", "Macrophages": "#009E73",
    "Epithelial luminal": "#E69F00", "Endothelial": "#9ECAE1",
    "Epithelial cycling": "#0072B2", "Epithelial basal": "#7F7F7F",
}
PANEL_TITLES = {
    "A": "Analysis overview", "B": "Pathway overlap", "C": "Ligand–receptor overlap",
    "D": "Signaling-strength agreement", "E": "VEGF communication network",
    "F": "VEGF sender–receiver signaling", "G": "VEGF ligand–receptor pairs",
    "H": "KDR activity", "I": "Hypoxia activity", "J": "FLT1 activity",
}
MANIFEST = json.loads((DATA / "manifest.json").read_text())
pathway = pd.read_csv(DATA / "pathway_overlap.csv")
lr_overlap = pd.read_csv(DATA / "lr_harmonized_overlap.csv")
strength = pd.read_csv(DATA / "strength_correlation.csv.gz")
vegf_matrix = pd.read_csv(DATA / "vegf_communication_matrix.csv", index_col=0).loc[CELL_ORDER, CELL_ORDER]
vegf_interactions = pd.read_csv(DATA / "vegf_top_interactions.csv.gz")
umap_scores = pd.read_parquet(DATA / "h_graph_umap_scores.parquet")

print(f"Loaded Figure 7 bundle: {len(umap_scores):,} cells; {len(strength):,} matched signaling observations")
display(pd.DataFrame(MANIFEST["checks"].items(), columns=["check", "passed"]))
signaling_statistics = pd.read_csv(DATA / "signaling_statistics.csv").iloc[0]


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
        (0.03, 0.58, 0.25, 0.23, "CellChat", "Probability\nLR/pathway calls", CELLCHAT_COLOR),
        (0.03, 0.18, 0.25, 0.23, "SCKNIFE", "Weighted activity\n9 DE cell types", SCKNIFE_COLOR),
        (0.38, 0.38, 0.25, 0.23, "Harmonization", "Cell types +\nLR identifiers", "#6C757D"),
        (0.72, 0.58, 0.25, 0.23, "Concordance", "Overlap +\nstrength", "#0072B2"),
        (0.72, 0.18, 0.25, 0.23, "VEGF focus", "Networks +\nH-UMAP overlays", "#A50F15"),
    ]
    for x, y, w, h, head, body, color in boxes:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015", facecolor="white", edgecolor=color, linewidth=1.8)
        ax.add_patch(box); ax.text(x+w/2, y+h*0.67, head, ha="center", va="center", weight="bold", color=color)
        ax.text(x+w/2, y+h*0.30, body, ha="center", va="center", fontsize=TICK_LABEL_SIZE_PT)
    for start, end in [((.28,.695),(.38,.515)),((.28,.295),(.38,.495)),((.63,.52),(.72,.695)),((.63,.47),(.72,.295))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#4D4D4D"))
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
    ax.text(cx_left-r_left*.2, cy-r_left-0.08, "CellChat", ha="center", weight="bold")
    ax.text(cx_right+r_right*.15, cy-r_right-0.08, "SCKNIFE", ha="center", weight="bold")
    jaccard = shared / (left + shared + right)
    ax.text(.5, .91, f"{subtitle}\nJaccard = {jaccard:.3f}", ha="center", va="top", fontsize=TEXT_SIZE_PT)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal"); ax.axis("off"); panel_title(ax, label)

def draw_strength(ax):
    raw_x = strength["cellchat_prob"].to_numpy(); raw_y = strength["scknife_score"].to_numpy()
    x = np.log10(raw_x); y = np.log10(raw_y)
    pearson_r=float(signaling_statistics.log_pearson_r); spearman_r=float(signaling_statistics.log_spearman_rho)
    raw_pearson_r=float(signaling_statistics.raw_pearson_r); raw_spearman_r=float(signaling_statistics.raw_spearman_rho)
    order = np.argsort(strength["normalized_lr_key"].astype(str).to_numpy())
    ax.scatter(x[order], y[order], s=5, alpha=.28, color=SCATTER_COLOR, edgecolors="none", rasterized=False)
    slope=float(signaling_statistics.log_linear_slope); intercept=float(signaling_statistics.log_linear_intercept); xx = np.array([x.min(), x.max()])
    ax.plot(xx, slope*xx+intercept, color="#666666", ls="--", lw=1.3)
    ax.text(.02,.98,f"Log10: Pearson r={pearson_r:.3f}; Spearman ρ={spearman_r:.3f}\nRaw: Pearson r={raw_pearson_r:.3f}; Spearman ρ={raw_spearman_r:.3f}\nn={len(x):,}", transform=ax.transAxes, va="top", fontsize=TICK_LABEL_SIZE_PT)
    ax.set_xlabel("log10(CellChat probability)"); ax.set_ylabel("log10(SCKNIFE weighted score)")
    clean_axis(ax); panel_title(ax, "D")

def draw_network(ax):
    graph = nx.DiGraph()
    top = vegf_matrix.stack().sort_values(ascending=False)
    top = top.loc[top > 0].head(40)
    for (source, target), weight in top.items(): graph.add_edge(source, target, weight=float(weight))
    graph.add_nodes_from(CELL_ORDER); pos = nx.circular_layout(graph)
    weights = np.array([data["weight"] for _,_,data in graph.edges(data=True)])
    lo, hi = np.percentile(weights, [LOW_PERCENTILE, HIGH_PERCENTILE]); denom=max(hi-lo,np.finfo(float).eps)
    axes_width_in = ax.get_window_extent().width / ax.figure.dpi
    geometry_scale = min(1.0, axes_width_in / NETWORK_REFERENCE_AXES_WIDTH_IN)
    widths = geometry_scale * (
        NETWORK_EDGE_WIDTH_MIN_PT
        + (NETWORK_EDGE_WIDTH_MAX_PT - NETWORK_EDGE_WIDTH_MIN_PT)
        * np.clip((weights-lo)/denom,0,1)
    )
    nx.draw_networkx_nodes(graph,pos,node_color=[CELL_COLORS[x] for x in graph.nodes()],node_size=NETWORK_NODE_SIZE_PT2*geometry_scale**2,edgecolors="white",linewidths=NETWORK_NODE_BORDER_WIDTH_PT*geometry_scale,ax=ax)
    nx.draw_networkx_labels(graph,pos,font_size=7,font_weight="bold",ax=ax)
    nx.draw_networkx_edges(graph,pos,width=widths,alpha=.55,arrows=True,arrowsize=NETWORK_ARROW_SIZE_PT*geometry_scale,connectionstyle="arc3,rad=0.08",edge_color=NETWORK_COLOR,ax=ax)
    ax.axis("off"); panel_title(ax,"E")

def draw_heatmap(ax):
    matrix = vegf_matrix.to_numpy()
    vmax=np.percentile(matrix,HIGH_PERCENTILE)
    image = ax.pcolormesh(np.arange(matrix.shape[1]+1),np.arange(matrix.shape[0]+1),matrix,cmap=VEGF_CMAP,vmin=0,vmax=vmax,shading="flat",edgecolors="white",linewidth=.35,rasterized=False)
    ax.set_xlim(0,matrix.shape[1]); ax.set_ylim(matrix.shape[0],0); ax.set_aspect("auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value=matrix[row,col]
            text_color="white" if value > .55*vmax else "#222222"
            ax.text(col+.5,row+.5,format(value,HEATMAP_VALUE_FORMAT),ha="center",va="center",fontsize=HEATMAP_ANNOTATION_SIZE,color=text_color)
    ax.set_xticks(np.arange(len(CELL_ORDER))+.5, CELL_ORDER, rotation=90)
    ax.set_yticks(np.arange(len(CELL_ORDER))+.5, CELL_ORDER)
    ax.set_xlabel("Receiver"); ax.set_ylabel("Sender")
    plt.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Weighted score")
    panel_title(ax,"F")

def draw_lollipop(ax):
    agg=(vegf_interactions.assign(pair=lambda d:d.ligand.astype(str)+" → "+d.receptor.astype(str)).groupby("pair",as_index=False).weighted_score.sum().nlargest(8,"weighted_score").sort_values("weighted_score"))
    colors=plt.cm.viridis(np.linspace(.20,.90,len(agg)))
    ax.barh(np.arange(len(agg)),agg.weighted_score,color=colors,height=.78,edgecolor="none")
    ax.set_yticks(np.arange(len(agg)),agg.pair)
    ax.set_xlabel("Summed weighted score"); ax.set_xlim(left=0)
    clean_axis(ax); panel_title(ax,"G")

def draw_umap(ax, column, label, colorbar_label):
    values=umap_scores[column].to_numpy(); lo,hi=np.percentile(values,[LOW_PERCENTILE,HIGH_PERCENTILE])
    order=np.argsort(values); x=umap_scores["umap_1"].to_numpy()[order]; y=umap_scores["umap_2"].to_numpy()[order]
    plotted=np.clip(values[order],lo,hi)
    points=ax.scatter(x,y,c=plotted,cmap=UMAP_CMAP,s=UMAP_POINT_SIZE,alpha=UMAP_ALPHA,linewidths=0,rasterized=RASTERIZED)
    plt.colorbar(points,ax=ax,fraction=.046,pad=.04,label=colorbar_label)
    ax.set_xlabel("H-UMAP 1"); ax.set_ylabel("H-UMAP 2"); clean_axis(ax); panel_title(ax,label)

def draw_panel(label, ax):
    if label=="A": draw_workflow(ax)
    elif label=="B":
        c=pathway.status.value_counts(); draw_venn(ax,(int(c.cellchat_only),int(c.shared),int(c.scknife_only)),"B","79 shared pathways")
    elif label=="C":
        c=lr_overlap.status.value_counts(); draw_venn(ax,(int(c.cellchat_only),int(c.shared),int(c.scknife_only)),"C","551 shared LR pairs")
    elif label=="D": draw_strength(ax)
    elif label=="E": draw_network(ax)
    elif label=="F": draw_heatmap(ax)
    elif label=="G": draw_lollipop(ax)
    elif label=="H": draw_umap(ax,"KDR_activity","H","KDR activity")
    elif label=="I": draw_umap(ax,"HALLMARK_HYPOXIA_activity","I","HALLMARK_HYPOXIA activity")
    elif label=="J": draw_umap(ax,"FLT1_activity","J","FLT1 activity")
    else: raise KeyError(label)

def make_figure7_main_panel(label):
    fig,ax=plt.subplots(figsize=(PANEL_WIDTH_IN,PANEL_HEIGHT_IN)); draw_panel(label,ax); fig.tight_layout(); return fig

def _save_live_vector_svg(fig,path,bbox):
    fig.savefig(path,format="svg",bbox_inches=bbox)

def save_individual(label, fig):
    png=INDIVIDUAL/f"Figure_7_{label}.png"; pdf=INDIVIDUAL/f"Figure_7_{label}.pdf"; svg=INDIVIDUAL/f"Figure_7_{label}.svg"
    fig.savefig(png,format="png",dpi=OUTPUT_DPI,bbox_inches="tight")
    fig.savefig(pdf,format="pdf",bbox_inches="tight")
    return png,pdf,svg