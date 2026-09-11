"""Figure 2 plotting functions using frozen, checksummed tables."""
from pathlib import Path
from functools import lru_cache
import hashlib, importlib.util, json, math, re, sys
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display

# ─── Figure parameters (user-editable) ───────────────────────────────
FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans"]
PANEL_LABEL_SIZE_PT = 10; AXIS_TITLE_SIZE_PT = 9; TICK_LABEL_SIZE_PT = 7
LEGEND_SIZE_PT = 7; ANNOTATION_SIZE_PT = 7; PAGE_TITLE_SIZE_PT = 12
COMPARATOR_POINT_SIZE = 0.18; SCKNIFE_POINT_SIZE = 0.10; UMAP_ALPHA = 0.75
HALLMARK_DIAGNOSTIC_POINT_SIZE = 3.0; HALLMARK_DIAGNOSTIC_ALPHA = 0.67
RASTERIZED = True; OUTPUT_DPI = 300
A4_LANDSCAPE_FIGSIZE = (11.69, 8.27); OVERVIEW_FIGSIZE = (11.69, 6.2); METRIC_HEATMAP_FIGSIZE = (10.5, 3.3); TUNING_FIGSIZE = (11, 8); METRIC_CMAP = "RdBu_r"
# ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":FONT_FAMILY,"pdf.fonttype":42,"ps.fonttype":42,"axes.grid":False})
# Restrained two-colour gradients: darker tones echo the benchmark palette.
HALLMARK_BGNMF_DARK="#7A1F3D"; HALLMARK_SCANPY_DARK="#1F5D70"
HALLMARK_BGNMF_DOT_DARK="#55162B"; HALLMARK_SCANPY_DOT_DARK="#16414E"
HALLMARK_BGNMF_CMAP=LinearSegmentedColormap.from_list("hallmark_bgnmf",["#F5E7EB",HALLMARK_BGNMF_DOT_DARK])
HALLMARK_SCANPY_CMAP=LinearSegmentedColormap.from_list("hallmark_scanpy",["#E4F0F2",HALLMARK_SCANPY_DOT_DARK])
METHOD_ORDER=["scKNIFE","SPECTRA","ExpiMap","Slalom"]
METHOD_SLUG={"scKNIFE":"scknife","SPECTRA":"spectra","ExpiMap":"expimap","Slalom":"slalom"}
METHOD_LABEL={value:key for key,value in METHOD_SLUG.items()}
METHOD_COLORS={"scKNIFE":"#A46BBE","SPECTRA":"#63B8D0","ExpiMap":"#E76F70","Slalom":"#43AAA5"}
DATASET_ORDER=["bassez_c1","bassez_c2","wu_tnbc"]; SCKNIFE_DATASET_ORDER=["bassez_c1","bassez_c2","shiao","wu_tnbc"]
DATASET_LABEL={"bassez_c1":"Bassez C1","bassez_c2":"Bassez C2","wu_tnbc":"Wu TNBC","shiao":"Shiao"}
UMAP_COLUMNS=["cell_id","umap_1","umap_2","leiden_cluster","author_cell_type"]
CATEGORY_PALETTE=["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9","#D55E00","#F0E442","#6B6B6B","#8DD3C7","#BEBADA","#FB8072","#80B1D3","#FDB462","#B3DE69","#FCCDE5","#BC80BD","#CCEBC5","#FFED6F","#1B9E77","#7570B3"]
DATA_DIR=(Path(__file__).resolve().parent/"data").resolve()
if not DATA_DIR.is_dir(): raise FileNotFoundError("Figure_2/data is missing")
manifest=json.loads((DATA_DIR/"manifest.json").read_text())
if manifest.get("format_version") != 2: raise RuntimeError("Figure 2 requires manifest format_version 2")
expected_umaps={*(f"{method}_{dataset}_umap.csv.gz" for method in ("expimap","slalom","spectra") for dataset in DATASET_ORDER),*(f"scknife_{dataset}_umap.csv.gz" for dataset in SCKNIFE_DATASET_ORDER)}
expected_inputs=expected_umaps|{"runtime.csv","knowledge_feature_counts.csv","agreement_metrics_from_frozen_labels.csv","tuning_sweeps.csv","tuning_final_parameters.csv","tahoe_plot_data.h5"}
manifest_names={entry["name"] for entry in manifest["files"]}
if not expected_inputs.issubset(manifest_names): raise RuntimeError(f"Manifest inventory is missing inputs: {sorted(expected_inputs-manifest_names)}")
def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
validation_rows=[]
for entry in manifest["files"]:
    source=DATA_DIR/entry["name"]
    if not source.is_file(): raise FileNotFoundError(source)
    if source.stat().st_size!=entry["bytes"] or sha256(source)!=entry["sha256"]: raise RuntimeError(f"Size/checksum mismatch: {source.name}")
    if source.name.endswith("_umap.csv.gz"):
        frame=pd.read_csv(source,dtype={"cell_id":str,"leiden_cluster":str,"author_cell_type":str})
        if list(frame.columns)!=UMAP_COLUMNS: raise ValueError(f"{source.name}: invalid columns")
        if frame[UMAP_COLUMNS].isna().any().any() or frame.cell_id.duplicated().any(): raise ValueError(f"{source.name}: invalid rows")
        if not np.isfinite(frame[["umap_1","umap_2"]].to_numpy()).all(): raise ValueError(f"{source.name}: non-finite UMAP coordinates")
        validation_rows.append({"file":source.name,"cells":len(frame),"clusters":frame.leiden_cluster.nunique(),"author_types":frame.author_cell_type.nunique()})
        del frame
print(f"Validated {len(manifest['files'])} inputs and {len(validation_rows)} UMAP schemas/checksums (matplotlib fallback; cnsplots unavailable in Python 3.9 environment).")
display(pd.DataFrame(validation_rows))
TAHOE_BUNDLE = DATA_DIR / "tahoe_plot_data.h5"
TAHOE_EXPECTED = {"FAS": 64064, "TP53": 64064, "BIRC5": 64064, "ICAM1": 64064, "PIK3CA": 60743, "PIK3R1": 60743, "RPTOR": 60743}
tahoe = h5py.File(TAHOE_BUNDLE, "r")
if set(tahoe) != set(TAHOE_EXPECTED): raise RuntimeError("Unexpected Tahoe Hallmark bundle groups")

@lru_cache(maxsize=16)
def load_umap(method, dataset):
    return pd.read_csv(DATA_DIR/f"{method}_{dataset}_umap.csv.gz", dtype={"cell_id":str,"leiden_cluster":str,"author_cell_type":str})

runtime_df=pd.read_csv(DATA_DIR/"runtime.csv")
runtime_df["method"]=runtime_df.method.replace({"bgNMF":"scKNIFE","Spectra":"SPECTRA"})
runtime_df["hours"]=pd.to_numeric(runtime_df.seconds,errors="raise")/3600
capacity_df=pd.read_csv(DATA_DIR/"knowledge_feature_counts.csv")
capacity_order=[("SPECTRA","Benchmarking"),("ExpiMap","Benchmarking"),("Slalom","Benchmarking"),("scKNIFE","Benchmarking graph"),("scKNIFE","Full in-house graph")]
capacity_df=capacity_df.set_index(["method","scope"]).loc[capacity_order].reset_index()
agreement_df=pd.read_csv(DATA_DIR/"agreement_metrics.csv")
hallmark_correlations=pd.read_csv(DATA_DIR/"hallmark_correlations.csv")
tuning_df=pd.read_csv(DATA_DIR/"tuning_sweeps.csv")
tuning_final_df=pd.read_csv(DATA_DIR/"tuning_final_parameters.csv")
tuning_selections=pd.read_csv(DATA_DIR/"tuning_selections.csv")
_data_manifest=manifest

def clean_axis(ax):
    ax.grid(False); ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=TICK_LABEL_SIZE_PT)
def draw_workflow(ax):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off"); ax.set_title("A  Benchmarking workflow",loc="left",fontsize=PANEL_LABEL_SIZE_PT,fontweight="bold")
    stages=[(.02,"Cohorts","Bassez C1 · Bassez C2\nWu TNBC · Shiao*","#EAF2F8"),(.27,"Methods","scKNIFE · SPECTRA\nExpiMap · Slalom","#F4ECF7"),(.52,"Frozen results","UMAP coordinates\nLeiden + author labels","#E8F6F3"),(.77,"Evaluation","Runtime · capacity\nARI · AMI · NMI","#FEF5E7")]
    width,height,y=.20,.55,.22
    for index,(x,heading,detail,color) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle="round,pad=0.012,rounding_size=0.018",linewidth=1,edgecolor="#4D4D4D",facecolor=color)); ax.text(x+width/2,y+.37,heading,ha="center",va="center",fontsize=AXIS_TITLE_SIZE_PT,fontweight="bold"); ax.text(x+width/2,y+.18,detail,ha="center",va="center",fontsize=ANNOTATION_SIZE_PT,linespacing=1.4)
        if index<len(stages)-1: ax.add_patch(FancyArrowPatch((x+width+.008,y+height/2),(stages[index+1][0]-.008,y+height/2),arrowstyle="-|>",mutation_scale=11,linewidth=1.1,color="#555555"))
    ax.text(.02,.06,"*Shiao is displayed for scKNIFE only; common-dataset metrics use shared cells from Bassez C1, Bassez C2, and Wu TNBC.",fontsize=ANNOTATION_SIZE_PT,color="#555555")
def draw_runtime(ax):
    datasets=["Bassez c1","Bassez c2"]; x=np.arange(len(datasets)); width=.78/len(METHOD_ORDER)
    for index,method in enumerate(METHOD_ORDER):
        rows=runtime_df[runtime_df.method.eq(method)].set_index("dataset").reindex(datasets); values=rows.hours.to_numpy(); offsets=x+(index-(len(METHOD_ORDER)-1)/2)*width
        bars=ax.bar(offsets,values,width*.90,color=METHOD_COLORS[method],edgecolor="#333333",linewidth=.5,label=method)
        for bar,hours in zip(bars,values):
            minutes=int(round(float(hours)*60)); ax.annotate(f"{minutes//60}h {minutes%60:02d}m",(bar.get_x()+bar.get_width()/2,hours),xytext=(0,4),textcoords="offset points",ha="center",fontsize=ANNOTATION_SIZE_PT)
    ax.set_xticks(x,datasets); ax.set_ylabel("Training time (hours)",fontsize=AXIS_TITLE_SIZE_PT); ax.set_ylim(0,runtime_df.hours.max()*1.22); ax.set_title("B  Training runtime at 20k cells",loc="left",fontsize=PANEL_LABEL_SIZE_PT,fontweight="bold"); clean_axis(ax); ax.legend(loc="upper center",bbox_to_anchor=(.5,-.16),ncol=2,frameon=False,fontsize=LEGEND_SIZE_PT)
def draw_capacity(ax):
    labels=["SPECTRA","ExpiMap","Slalom","scKNIFE\nbenchmark","scKNIFE\nfull in-house"]; colors=[METHOD_COLORS["SPECTRA"],METHOD_COLORS["ExpiMap"],METHOD_COLORS["Slalom"],"#C7A4D4",METHOD_COLORS["scKNIFE"]]; x=np.arange(len(capacity_df))
    bars=ax.bar(x,capacity_df["count"],color=colors,edgecolor="#333333",linewidth=.6,width=.62); ax.set_yscale("log"); ax.set_xticks(x,labels); ax.set_ylabel("Prior-knowledge features (log scale)",fontsize=AXIS_TITLE_SIZE_PT); ax.set_title("C  Prior-knowledge capacity",loc="left",fontsize=PANEL_LABEL_SIZE_PT,fontweight="bold")
    for bar,value in zip(bars,capacity_df["count"]): ax.annotate(f"{int(value):,}",(bar.get_x()+bar.get_width()/2,value),xytext=(0,4),textcoords="offset points",ha="center",fontsize=ANNOTATION_SIZE_PT,fontweight="bold")
    ax.text(0,-.24,"Comparators: gene sets/factors · scKNIFE: unified graph nodes",transform=ax.transAxes,fontsize=ANNOTATION_SIZE_PT,color="#555555"); clean_axis(ax)
def make_overview_page():
    fig=plt.figure(figsize=OVERVIEW_FIGSIZE,layout="constrained"); grid=fig.add_gridspec(2,2,height_ratios=[.36,.64]); draw_workflow(fig.add_subplot(grid[0,:])); draw_runtime(fig.add_subplot(grid[1,0])); draw_capacity(fig.add_subplot(grid[1,1])); fig.suptitle("Figure 2 — benchmarking overview",fontsize=PAGE_TITLE_SIZE_PT,fontweight="bold"); return fig
METRIC_SPECS=[("ari","ARI"),("ami","AMI"),("nmi","NMI")]
def make_metric_heatmaps():
    shared_n=agreement_df.groupby("dataset").n_shared.first().to_dict(); fig,axes=plt.subplots(1,3,figsize=METRIC_HEATMAP_FIGSIZE,sharey=True,constrained_layout=True)
    for panel,(ax,(metric,label)) in enumerate(zip(axes,METRIC_SPECS),start=1):
        matrix=agreement_df.pivot(index="method",columns="dataset",values=metric).reindex(index=METHOD_ORDER,columns=DATASET_ORDER); image=ax.imshow(matrix.to_numpy(),cmap=METRIC_CMAP,vmin=0,vmax=1,aspect="auto"); ax.set_xticks(np.arange(len(DATASET_ORDER)),[f"{DATASET_LABEL[d]}\nn={shared_n[d]:,}" for d in DATASET_ORDER]); ax.set_yticks(np.arange(len(METHOD_ORDER)),METHOD_ORDER); ax.set_title(f"D{panel}  {label}",loc="left",fontsize=PANEL_LABEL_SIZE_PT,fontweight="bold"); ax.tick_params(labelsize=TICK_LABEL_SIZE_PT)
        for row in range(len(METHOD_ORDER)):
            for col in range(len(DATASET_ORDER)):
                value=matrix.iloc[row,col]; rgba=plt.get_cmap(METRIC_CMAP)(value); luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]; ax.text(col,row,f"{value:.3f}",ha="center",va="center",fontsize=ANNOTATION_SIZE_PT,color="black" if luminance>.58 else "white",fontweight="bold")
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.colorbar(image,ax=axes,location="right",fraction=.025,pad=.02,label="Agreement score"); return fig
for maker in (make_overview_page,make_metric_heatmaps):
    fig=maker(); display(fig); plt.close(fig)

TUNING_DATASET_ORDER=["Bassez C1","Bassez C2","Wu TNBC","Shiao"]
TUNING_PANEL_COLOR="#365F70"; TUNING_SECONDARY_COLOR="#C26D3A"; TUNING_SELECTED_COLOR="#D73027"; PRODUCTION_COLOR="#111111"

def _format_parameter(value):
    if value==0: return "0"
    if abs(value)<.01: return f"{value:.0e}"
    return f"{value:g}"

def _mark_choices(ax,frame,x_column,y_column="silhouette",categorical_x=False):
    selected=frame[frame.tuning_selected]; production=frame[frame.production_used]
    for subset,marker,color,size,zorder in ((selected,"*",TUNING_SELECTED_COLOR,150,6),(production,"D",PRODUCTION_COLOR,48,7)):
        if subset.empty: continue
        row=subset.iloc[0]; x=(frame.index.get_loc(row.name) if categorical_x else row[x_column]); ax.scatter([x],[row[y_column]],marker=marker,s=size,color=color,edgecolor="white",linewidth=.5,zorder=zorder)

def make_tuning_page(dataset,page_number):
    frame=tuning_df[tuning_df.dataset.eq(dataset)].copy(); final=tuning_final_df.set_index("dataset").loc[dataset]
    fig,axes=plt.subplots(2,2,figsize=TUNING_FIGSIZE,layout="constrained")

    kdf=frame[frame.phase.eq("k")].sort_values("k").set_index(pd.Index(range(sum(frame.phase.eq("k")))))
    axes[0,0].plot(kdf.k,kdf.silhouette,color=TUNING_PANEL_COLOR,marker="o",linewidth=1.5,markersize=4); _mark_choices(axes[0,0],kdf,"k")
    axes[0,0].set_xlabel("Latent factors (k)",fontsize=AXIS_TITLE_SIZE_PT); axes[0,0].set_ylabel("Silhouette",fontsize=AXIS_TITLE_SIZE_PT); axes[0,0].set_title(f"E{page_number}a  k sweep",loc="left",fontsize=PANEL_LABEL_SIZE_PT,fontweight="bold"); clean_axis(axes[0,0])

    joint=frame[frame.phase.eq("joint_k_lambda")].copy(); kvals=sorted(joint.k.unique()); lvals=sorted(joint.lambda_reg.unique())
    matrix=joint.pivot(index="k",columns="lambda_reg",values="silhouette").reindex(index=kvals,columns=lvals)
    global_joint=tuning_df[tuning_df.phase.eq("joint_k_lambda")].silhouette; image=axes[0,1].imshow(matrix.to_numpy(),cmap="RdBu_r",vmin=global_joint.min(),vmax=global_joint.max(),aspect="auto")
    axes[0,1].set_xticks(range(len(lvals)),[_format_parameter(v) for v in lvals]); axes[0,1].set_yticks(range(len(kvals)),[f"{int(v)}" for v in kvals]); axes[0,1].set_xlabel("Graph regularization (lambda)",fontsize=AXIS_TITLE_SIZE_PT); axes[0,1].set_ylabel("k",fontsize=AXIS_TITLE_SIZE_PT); axes[0,1].set_title(f"E{page_number}b  Joint k × lambda",loc="left",fontsize=PANEL_LABEL_SIZE_PT,fontweight="bold")
    for row,kval in enumerate(kvals):
        for col,lval in enumerate(lvals):
            value=matrix.loc[kval,lval]
            if pd.notna(value):
                rgba=plt.get_cmap("RdBu_r")((value-global_joint.min())/(global_joint.max()-global_joint.min())); luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]; axes[0,1].text(col,row,f"{value:.3f}",ha="center",va="center",fontsize=ANNOTATION_SIZE_PT,color="black" if luminance>.58 else "white")
    for subset,marker,color,size,zorder in ((joint[joint.tuning_selected],"*",TUNING_SELECTED_COLOR,180,6),(joint[joint.production_used],"D",PRODUCTION_COLOR,55,7)):
        if not subset.empty:
            choice=subset.iloc[0]; axes[0,1].scatter([lvals.index(choice.lambda_reg)],[kvals.index(choice.k)],marker=marker,s=size,color=color,edgecolor="white",linewidth=.6,zorder=zorder)
    for spine in axes[0,1].spines.values(): spine.set_visible(False)
    fig.colorbar(image,ax=axes[0,1],fraction=.046,pad=.04,label="Silhouette")

    bw=frame[frame.phase.eq("beta_W")].sort_values("beta_W").reset_index(drop=True); ax=axes[1,0]; secondary=ax.twinx()
    ax.plot(bw.beta_W,bw.silhouette,color=TUNING_PANEL_COLOR,marker="o",linewidth=1.5,markersize=4,label="Silhouette"); secondary.plot(bw.beta_W,bw.sparsity_W,color=TUNING_SECONDARY_COLOR,marker="s",linestyle="--",linewidth=1.2,markersize=3,label="W sparsity"); secondary.axhspan(.35,.55,color="#D9EAD3",alpha=.55,label="Target sparsity")
    _mark_choices(ax,bw,"beta_W"); ax.set_xscale("symlog",linthresh=1e-4,linscale=.8); ax.set_xticks(bw.beta_W,[_format_parameter(v) for v in bw.beta_W]); ax.set_xlabel("W sparsity penalty (beta_W)",fontsize=AXIS_TITLE_SIZE_PT); ax.set_ylabel("Silhouette",fontsize=AXIS_TITLE_SIZE_PT,color=TUNING_PANEL_COLOR); secondary.set_ylabel("W sparsity",fontsize=AXIS_TITLE_SIZE_PT,color=TUNING_SECONDARY_COLOR); ax.set_title(f"E{page_number}c  beta_W sweep",loc="left",fontsize=PANEL_LABEL_SIZE_PT,fontweight="bold"); clean_axis(ax); secondary.grid(False); secondary.spines["top"].set_visible(False); secondary.tick_params(labelsize=TICK_LABEL_SIZE_PT)

    bh=frame[frame.phase.eq("beta_H")].sort_values("beta_H").reset_index(drop=True); axes[1,1].plot(bh.beta_H,bh.silhouette,color=TUNING_PANEL_COLOR,marker="o",linewidth=1.5,markersize=4); _mark_choices(axes[1,1],bh,"beta_H"); axes[1,1].set_xscale("log"); axes[1,1].set_xticks(bh.beta_H,[_format_parameter(v) for v in bh.beta_H]); axes[1,1].set_xlabel("H sparsity penalty (beta_H)",fontsize=AXIS_TITLE_SIZE_PT); axes[1,1].set_ylabel("Silhouette",fontsize=AXIS_TITLE_SIZE_PT); axes[1,1].set_title(f"E{page_number}d  beta_H sweep",loc="left",fontsize=PANEL_LABEL_SIZE_PT,fontweight="bold"); clean_axis(axes[1,1])

    handles=[Line2D([0],[0],marker="*",color="none",markerfacecolor=TUNING_SELECTED_COLOR,markeredgecolor="white",markersize=12,label="Tuning-selected"),Line2D([0],[0],marker="D",color="none",markerfacecolor=PRODUCTION_COLOR,markeredgecolor="white",markersize=6,label="Production-used")]
    fig.legend(handles=handles,loc="outside lower center",ncol=2,frameon=False,fontsize=LEGEND_SIZE_PT)
    final_text=f"Production: k={int(final.k)}, lambda={_format_parameter(final.lambda_reg)}, beta_W={_format_parameter(final.beta_W)}, beta_H={_format_parameter(final.beta_H)}"
    if dataset=="Shiao": final_text += " · Joint-grid optimum was k=40, lambda=1; production used k=80, lambda=1"
    fig.suptitle(f"E{page_number}  {dataset} — scKNIFE hyperparameter tuning\n{final_text}",fontsize=PAGE_TITLE_SIZE_PT,fontweight="bold")
    return fig

for page_number,dataset in enumerate(TUNING_DATASET_ORDER,start=1):
    fig=make_tuning_page(dataset,page_number); display(fig); plt.close(fig)

def natural_key(value):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(value))]

all_author_types = set()
for name in sorted(expected_umaps):
    values = pd.read_csv(DATA_DIR / name, usecols=["author_cell_type"], dtype=str).author_cell_type.dropna().unique()
    all_author_types.update(values)

def canonical_author_label(label):
    compact = re.sub(r"[._\s-]+", "", str(label)).lower()
    aliases = {
        "bcell": "B cell", "bcells": "B cell", "tcell": "T cell", "tcells": "T cell",
        "cancercell": "Cancer epithelial", "cancerepithelial": "Cancer epithelial",
        "endothelialcell": "Endothelial", "endothelial": "Endothelial",
        "fibroblast": "Fibroblast", "fibroblasts": "Fibroblast", "mastcell": "Mast cell",
        "myeloid": "Myeloid", "myeloidcell": "Myeloid", "normalepithelial": "Normal epithelial",
        "epithelial": "Epithelial", "plasmablasts": "Plasmablasts", "pvl": "PVL", "pdc": "pDC", "cafs": "CAFs",
    }
    return aliases.get(compact, str(label))

canonical_types = sorted({canonical_author_label(label) for label in all_author_types}, key=natural_key)
if len(canonical_types) > len(CATEGORY_PALETTE):
    raise ValueError("Author cell-type palette is too short")
canonical_colors = {label: CATEGORY_PALETTE[index] for index, label in enumerate(canonical_types)}
AUTHOR_COLORS = {label: canonical_colors[canonical_author_label(label)] for label in all_author_types}


def categorical_scatter(ax, frame, column, title, point_size, colors=None):
    categories = sorted(frame[column].astype(str).unique(), key=natural_key)
    color_map = colors or {category: CATEGORY_PALETTE[index % len(CATEGORY_PALETTE)] for index, category in enumerate(categories)}
    for category in categories:
        selected = frame[column].astype(str).eq(category)
        ax.scatter(frame.loc[selected, "umap_1"], frame.loc[selected, "umap_2"], s=point_size,
                   alpha=UMAP_ALPHA, color=color_map[category], label=category, linewidths=0, rasterized=RASTERIZED)
    ax.set_title(title, fontsize=PANEL_LABEL_SIZE_PT); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, markerscale=5, fontsize=LEGEND_SIZE_PT)
    ax.set_aspect("equal", adjustable="datalim")


def make_comparator_page(method):
    fig, axes = plt.subplots(3, 2, figsize=A4_LANDSCAPE_FIGSIZE, layout="constrained")
    for row_index, dataset in enumerate(DATASET_ORDER):
        frame = load_umap(method, dataset)
        ari = agreement_df.loc[(agreement_df.method.eq(METHOD_LABEL[method])) & (agreement_df.dataset.eq(dataset)), "ari"].iloc[0]
        categorical_scatter(axes[row_index, 0], frame, "leiden_cluster",
                            f"{DATASET_LABEL[dataset]} — Leiden (ARI={ari:.3f}, {frame.leiden_cluster.nunique()} clusters)", COMPARATOR_POINT_SIZE)
        categorical_scatter(axes[row_index, 1], frame, "author_cell_type",
                            f"{DATASET_LABEL[dataset]} — author cell types ({frame.author_cell_type.nunique()} types)", COMPARATOR_POINT_SIZE, AUTHOR_COLORS)
    fig.suptitle(f"Figure 2 — {METHOD_LABEL[method]} frozen UMAPs", fontsize=PAGE_TITLE_SIZE_PT, fontweight="bold")
    return fig


def make_scknife_page():
    fig, axes = plt.subplots(2, 2, figsize=A4_LANDSCAPE_FIGSIZE, layout="constrained")
    for ax, dataset in zip(axes.flat, SCKNIFE_DATASET_ORDER):
        frame = load_umap("scknife", dataset)
        categorical_scatter(ax, frame, "leiden_cluster",
                            f"{DATASET_LABEL[dataset]} — Leiden ({frame.leiden_cluster.nunique()} clusters)", SCKNIFE_POINT_SIZE)
    fig.suptitle("Figure 2 — scKNIFE frozen UMAPs", fontsize=PAGE_TITLE_SIZE_PT, fontweight="bold")
    return fig


# Tahoe Hallmark plots — local compact numerical inputs only.
def tahoe_density_scatter(ax, x, y, cmap, point_size=HALLMARK_DIAGNOSTIC_POINT_SIZE, alpha=HALLMARK_DIAGNOSTIC_ALPHA):
    mask=np.isfinite(x)&np.isfinite(y); x,y=np.asarray(x)[mask],np.asarray(y)[mask]
    hist,xe,ye=np.histogram2d(x,y,bins=120); xi=np.clip(np.searchsorted(xe,x,side="right")-1,0,118); yi=np.clip(np.searchsorted(ye,y,side="right")-1,0,118); density=hist[xi,yi]; order=density.argsort()
    ax.scatter(x[order],y[order],c=density[order],cmap=cmap,s=point_size,alpha=alpha,edgecolors="none",rasterized=True); return x,y
def plot_tahoe_hallmark(driver_gene):
    group=tahoe[driver_gene]; x=group["expression"][:]; scores=[group["bgnmf_score"][:],group["scanpy_score"][:]]
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),dpi=150)
    for ax,y,cmap,color,ylabel,title in zip(axes,scores,[HALLMARK_BGNMF_CMAP,HALLMARK_SCANPY_CMAP],[HALLMARK_BGNMF_DARK,HALLMARK_SCANPY_DARK],["bGNMF Activity Score","Scanpy score_genes"],["bGNMF Factor Score","Scanpy Standard Score"]):
        xp,yp=tahoe_density_scatter(ax,x,y,cmap); slope,intercept=np.polyfit(xp,yp,1); line=np.linspace(xp.min(),xp.max(),200); ax.plot(line,slope*line+intercept,color=color,linewidth=2,linestyle="--")
        score_label = "scKNIFE" if "bGNMF" in title else "Scanpy"
        corr_row = hallmark_correlations.loc[(hallmark_correlations.gene.eq(driver_gene)) & (hallmark_correlations.score.eq(score_label))].iloc[0]
        ax.set(xlabel=f"{driver_gene} Expression (MAGIC)", ylabel=f"{ylabel} ({group.attrs['pathway_label']})", title=f"{title}\nPearson r = {corr_row.pearson_r:.3f} | Spearman rho = {corr_row.spearman_rho:.3f}"); clean_axis(ax)
    fig.suptitle(f"Hallmark: {group.attrs['pathway_label']} | {group.attrs['drug']}-treated cells | Driver gene: {driver_gene}",fontweight="bold",y=1.02); fig.tight_layout(); plt.show(); return {"driver_gene":driver_gene,"n":len(x)}

# Final A4 portrait mixed raster/vector compilation and export.
# Dense UMAP/Hallmark point clouds are temporary 300-DPI PNG assets;
# axes, labels, legends, annotations, lines, bars, and heatmap cells remain vector.
from pathlib import Path
import json, platform, tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter
from IPython.display import display
import numpy as np

# ─── Figure 2 editable parameters ───────────────────────────────────────────
PAGE_W_MM, PAGE_H_MM, OUTPUT_DPI = 210, 297, 300
FIG2_FONT_FAMILY = 'DejaVu Sans'
FIG2_FONT_FALLBACK = 'DejaVu Sans'
R_LABEL, R_TITLE, R_TICK, R_NOTE = 7, 7, 7, 7
R_POINT = .055
R_HALLMARK_POINT, R_HALLMARK_ALPHA = 4.5, .85
R_UMAP_LEGEND = 7
REF_B_COLORS = ['#CABCA2', '#C5AD7D', '#BEA890', '#416879', '#416879']
REF_C_COLORS = {'scKNIFE':'#4F8F3A', 'SPECTRA':'#F4C678', 'ExpiMap':'#A31F2A', 'Slalom':'#BFDC80'}

mpl.rcParams.update({
    'font.family': [FIG2_FONT_FAMILY, FIG2_FONT_FALLBACK],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'pdf.use14corefonts': False,
})


def mm_rect(x, y, w, h):
    return [x / PAGE_W_MM, y / PAGE_H_MM, w / PAGE_W_MM, h / PAGE_H_MM]


def panel_title(ax, label, title):
    ax.text(0, 1.035, f'{label}  {title}', transform=ax.transAxes, ha='left', va='bottom',
            fontsize=R_LABEL, fontweight='bold', clip_on=False)


def _square_limits(frame, pad=.04):
    x = np.asarray(frame['umap_1'], dtype=float)
    y = np.asarray(frame['umap_2'], dtype=float)
    x_span, y_span = np.ptp(x), np.ptp(y)
    span = max(x_span, y_span)
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    span *= 1 + pad
    x_mid, y_mid = np.mean(x), np.mean(y)
    return (x_mid - span / 2, x_mid + span / 2), (y_mid - span / 2, y_mid + span / 2)


def _hallmark_limits(x, y, pad=.04):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    x_span = max(np.ptp(x), 1e-9); y_span = max(np.ptp(y), 1e-9)
    return (x.min() - pad * x_span, x.max() + pad * x_span), (y.min() - pad * y_span, y.max() + pad * y_span)


def save_umap_asset(frame, column, colors, path, xlim, ylim, width_mm, height_mm):
    image_fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=OUTPUT_DPI, facecolor='white')
    image_ax = image_fig.add_axes([0, 0, 1, 1])
    cats = sorted(frame[column].astype(str).unique(), key=natural_key)
    palette = colors or {value: CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)] for i, value in enumerate(cats)}
    for value in cats:
        take = frame[column].astype(str).eq(value)
        image_ax.scatter(frame.loc[take, 'umap_1'], frame.loc[take, 'umap_2'], s=R_POINT,
                          color=palette[value], alpha=UMAP_ALPHA, linewidths=0)
    image_ax.set_xlim(xlim); image_ax.set_ylim(ylim); image_ax.set_aspect('equal', adjustable='box'); image_ax.set_axis_off()
    image_fig.savefig(path, format='png', dpi=OUTPUT_DPI, facecolor='white', edgecolor='white', pad_inches=0)
    plt.close(image_fig)


def place_umap_asset(ax, path, xlim, ylim, title):
    ax.imshow(imread(path), extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='upper',
              aspect='auto', interpolation='none')
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=R_TITLE, pad=1)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def runtime_celltype_legend(ax, frame, colors):
    ax.axis('off')
    cats = sorted(frame['author_cell_type'].astype(str).unique(), key=natural_key)
    handles = [Line2D([], [], marker='o', linestyle='', markersize=2.8,
                      markerfacecolor=colors[value], markeredgecolor='none', label=value)
               for value in cats]
    ax.legend(handles=handles, loc='upper left', ncol=1, frameon=False,
              fontsize=R_UMAP_LEGEND, handlelength=.45, handletextpad=.20,
              borderaxespad=0, labelspacing=.10, columnspacing=.20, markerscale=1.0)


def draw_runtime_workflow(ax):
    ax.axis('off'); panel_title(ax, 'A', 'Benchmarking workflow')
    stages = [('Cohorts', 'Bassez C1/C2\nWu TNBC · Shiao', '#EAF2F8'),
              ('Methods', 'scKNIFE · SPECTRA\nExpiMap · Slalom', '#F4ECF7'),
              ('Frozen results', 'UMAP · Leiden\nauthor labels', '#E8F6F3'),
              ('Evaluation', 'Runtime · capacity\nARI · AMI · NMI', '#FEF5E7')]
    xs = [.01, .26, .51, .76]
    for i, (heading, detail, color) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((xs[i], .24), .21, .50, boxstyle='round,pad=.008,rounding_size=.015',
                                    linewidth=.7, edgecolor='#555', facecolor=color))
        ax.text(xs[i] + .105, .55, heading, ha='center', va='center', fontsize=R_TITLE, fontweight='bold')
        ax.text(xs[i] + .105, .38, detail, ha='center', va='center', fontsize=R_NOTE, linespacing=1.25)
        if i < 3:
            ax.add_patch(FancyArrowPatch((xs[i] + .215, .49), (xs[i + 1] - .006, .49), arrowstyle='-|>',
                                         mutation_scale=7, linewidth=.7, color='#666'))
    ax.text(.01, .08, 'Shared-cell comparison on frozen coordinates and labels', fontsize=R_NOTE, color='#555')


def draw_runtime_bars(ax):
    datasets = ['Bassez c1', 'Bassez c2']; x = np.arange(2); width = .18
    for i, method in enumerate(METHOD_ORDER):
        values = runtime_df[runtime_df.method.eq(method)].set_index('dataset').reindex(datasets).hours.to_numpy()
        bars = ax.bar(x + (i - 1.5) * width, values, width * .9, color=REF_C_COLORS[method],
                      edgecolor='#444', linewidth=.3, label=method)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + .25, f'{value:.1f}', ha='center', fontsize=R_NOTE, rotation=90)
    ax.set_xticks(x, ['Bassez C1', 'Bassez C2'], fontsize=R_TICK); ax.tick_params(axis='y', labelsize=R_TICK, pad=1)
    ax.set_ylabel('Hours', fontsize=R_TICK, labelpad=1); ax.set_ylim(0, runtime_df.hours.max() * 1.25); panel_title(ax, 'C', 'Training runtime')
    ax.legend(loc='upper left', bbox_to_anchor=(.01, .98), ncol=2, frameon=False, fontsize=R_NOTE,
              columnspacing=.5, handlelength=.8, borderaxespad=0)
    clean_axis(ax)


def _broken_mark(ax, position):
    size = .018; kwargs = {'transform': ax.transAxes, 'color': '#111111', 'clip_on': False, 'linewidth': .65}
    if position == 'bottom':
        ax.plot((-size, +size), (-size, +size), **kwargs); ax.plot((1 - size, 1 + size), (-size, +size), **kwargs)
    else:
        ax.plot((-size, +size), (1 - size, 1 + size), **kwargs); ax.plot((1 - size, 1 + size), (1 - size, 1 + size), **kwargs)


def draw_runtime_capacity_three_tier(fig, x, y, width, height):
    lower = fig.add_axes(mm_rect(x, y, width, height * .30))
    middle = fig.add_axes(mm_rect(x, y + height * .34, width, height * .28), sharex=lower)
    upper = fig.add_axes(mm_rect(x, y + height * .66, width, height * .34), sharex=lower)
    values = capacity_df['count'].to_numpy(); labels = ['SP', 'Ex', 'Sl', 'scK\nbench', 'scK\nfull']; colors = REF_B_COLORS
    for ax in (lower, middle, upper):
        ax.bar(range(5), values, color=colors, edgecolor='#3A2D29', linewidth=.55, width=.64)
        ax.set_xlim(-.6, 4.6); clean_axis(ax); ax.tick_params(axis='y', labelsize=R_TICK, pad=1)
    lower.set_ylim(0, 220); middle.set_ylim(700, 1100); upper.set_ylim(20000, 42000)
    comma_fmt = FuncFormatter(lambda value, _: f'{int(value):,}')
    for ax, ticks in ((lower, [0, 100, 200]), (middle, [700, 850, 1000]), (upper, [20000, 30000, 40000])):
        ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(comma_fmt)
    lower.set_xticks(range(5), labels, fontsize=R_TICK)
    for ax in (middle, upper): ax.tick_params(axis='x', bottom=False, labelbottom=False)
    lower.set_ylabel('Features', fontsize=R_TICK, labelpad=1); panel_title(upper, 'B', 'Prior knowledge')
    def label_visible(ax, ymin, ymax):
        offset = (ymax - ymin) * .045
        for index, value in enumerate(values):
            if ymin <= value <= ymax:
                ax.text(index, value + offset, f'{value:,}', ha='center', va='bottom', fontsize=R_NOTE,
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .90, 'pad': .55})
    label_visible(lower, 0, 220); label_visible(middle, 700, 1100); label_visible(upper, 20000, 42000)
    for ax in (lower, middle, upper):
        ax.spines['top'].set_visible(False); ax.spines['bottom'].set_visible(False)
    lower.spines['bottom'].set_visible(True)
    _broken_mark(lower, 'top'); _broken_mark(middle, 'bottom'); _broken_mark(middle, 'top'); _broken_mark(upper, 'bottom')


def runtime_metrics(ax):
    rows = [(d, m) for d in DATASET_ORDER for m in METHOD_ORDER]
    values = np.array([[agreement_df.loc[agreement_df.dataset.eq(d) & agreement_df.method.eq(m), v].iloc[0]
                        for v in ('ari', 'nmi', 'ami')] for d, m in rows])
    cmap = mpl.colormaps[METRIC_CMAP]
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.add_patch(Rectangle((col - .5, row - .5), 1, 1, facecolor=cmap(values[row, col]), edgecolor='none'))
    ax.set_xlim(-.5, values.shape[1] - .5); ax.set_ylim(values.shape[0] - .5, -.5)
    ax.set_xticks(range(3), ['ARI', 'NMI', 'AMI'], fontsize=R_TICK)
    ax.set_yticks(range(12), [m for d, m in rows], fontsize=R_TICK); ax.tick_params(axis='y', pad=1)
    for boundary in (3.5, 7.5): ax.axhline(boundary, color='white', linewidth=1.2)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]; rgba = cmap(value); lum = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            ax.text(col, row, f'{value:.2f}', ha='center', va='center', fontsize=R_NOTE, color='black' if lum > .58 else 'white')
    panel_title(ax, 'E', 'Agreement metrics')
    ax.text(0, 1.01, 'row blocks: Bassez C1 · Bassez C2 · Wu TNBC', transform=ax.transAxes, fontsize=R_NOTE, ha='left', va='bottom')
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('#000000'); spine.set_linewidth(.55)
    clean_axis(ax)
    return cmap, 0.0, 1.0


def vector_colorbar(fig, rect, cmap, vmin, vmax, label=''):
    cax = fig.add_axes(rect); edges = np.linspace(vmin, vmax, 129)
    for start, end in zip(edges[:-1], edges[1:]):
        cax.add_patch(Rectangle((0, start), 1, end - start, facecolor=cmap((start - vmin) / (vmax - vmin)), edgecolor='none'))
    cax.set_xlim(0, 1); cax.set_ylim(vmin, vmax); cax.set_xticks([]); cax.set_yticks(np.linspace(vmin, vmax, 5))
    cax.tick_params(labelsize=R_TICK, length=1, pad=1, width=.3)
    for spine in cax.spines.values():
        spine.set_visible(True); spine.set_linewidth(.25); spine.set_color('black')
    if label: cax.set_ylabel(label, fontsize=R_TICK, labelpad=1)


def save_hallmark_asset(expression, score, cmap, path, xlim, ylim, width_mm, height_mm):
    image_fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=OUTPUT_DPI, facecolor='white')
    image_ax = image_fig.add_axes([0, 0, 1, 1])
    xp, yp = tahoe_density_scatter(image_ax, expression, score, cmap, point_size=R_HALLMARK_POINT, alpha=R_HALLMARK_ALPHA)
    image_ax.set_xlim(xlim); image_ax.set_ylim(ylim); image_ax.set_axis_off()
    image_fig.savefig(path, format='png', dpi=OUTPUT_DPI, facecolor='white', edgecolor='white', pad_inches=0)
    plt.close(image_fig)
    return xp, yp


def draw_tahoe_compiled(fig, x, y, width, height, gene, label, asset_dir, asset_records):
    group = tahoe[gene]; expression = group['expression'][:]; drug = str(group.attrs['drug'])
    drug_label = 'Doxo' if 'Doxorubicin' in drug else ('Afi' if 'Afatinib' in drug else drug)
    for index, (score, cmap, line_color, title) in enumerate(((group['bgnmf_score'][:], HALLMARK_BGNMF_CMAP, HALLMARK_BGNMF_DARK, 'scKNIFE'),
                                                               (group['scanpy_score'][:], HALLMARK_SCANPY_CMAP, HALLMARK_SCANPY_DARK, 'Scanpy'))):
        xlim, ylim = _hallmark_limits(expression, score)
        asset_path = asset_dir / f'Figure_2_{gene}_{title}_Hallmark_points.png'
        xp, yp = save_hallmark_asset(expression, score, cmap, asset_path, xlim, ylim, width / 2 - 2, height)
        asset_records.append({'filename': asset_path.name, 'role': 'Hallmark point cloud', 'dpi': OUTPUT_DPI})
        ax = fig.add_axes(mm_rect(x + index * (width / 2 + 2), y, width / 2 - 2, height))
        ax.imshow(imread(asset_path), extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='upper', aspect='auto', interpolation='none')
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        slope, intercept = np.polyfit(xp, yp, 1); line = np.linspace(xp.min(), xp.max(), 100)
        ax.plot(line, slope * line + intercept, color=line_color, linestyle='--', linewidth=.6)
        corr_row = hallmark_correlations.loc[(hallmark_correlations.gene.eq(gene)) & (hallmark_correlations.score.eq("scKNIFE" if title == "scKNIFE" else "Scanpy"))].iloc[0]; pearson_value = corr_row.pearson_r; spearman_value = corr_row.spearman_rho
        ax.text(.03, .97, f'Pearson r = {pearson_value:.2f}\nSpearman ρ = {spearman_value:.2f}', transform=ax.transAxes,
                ha='left', va='top', fontsize=R_NOTE, color='#333333', bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.78, 'pad':.7})
        ax.set_title(f'{label} · {drug_label} | {title}', fontsize=R_TITLE, pad=1)
        ax.set_xlabel(gene, fontsize=R_TICK, labelpad=1); ax.set_ylabel('Score', fontsize=R_TICK, labelpad=1)
        ax.tick_params(labelsize=R_NOTE, pad=1); clean_axis(ax)


def make_figure2_runtime_preview():
    fig = plt.figure(figsize=(PAGE_W_MM / 25.4, PAGE_H_MM / 25.4), facecolor='white')
    asset_records = []
    with tempfile.TemporaryDirectory(prefix='figure_2_raster_') as temp_dir:
        asset_dir = Path(temp_dir)
        draw_runtime_workflow(fig.add_axes(mm_rect(7, 238, 91, 52)))
        draw_runtime_capacity_three_tier(fig, 108, 238, 34, 52)
        draw_runtime_bars(fig.add_axes(mm_rect(150, 238, 53, 52)))
        umap_x, umap_w, umap_gap = 7, 125, 3; umap_each = (umap_w - 3 * umap_gap) / 4
        cluster_y, cluster_h = 190, 34; celltype_y, celltype_h = 147, 34; legend_y, legend_h = 84, 54
        for i, dataset in enumerate(SCKNIFE_DATASET_ORDER):
            x = umap_x + i * (umap_each + umap_gap); frame = load_umap('scknife', dataset); xlim, ylim = _square_limits(frame)
            cluster_path = asset_dir / f'Figure_2_{dataset}_clustering_UMAP.png'
            save_umap_asset(frame, 'leiden_cluster', None, cluster_path, xlim, ylim, umap_each, cluster_h)
            asset_records.append({'filename': cluster_path.name, 'role': 'UMAP point cloud', 'dpi': OUTPUT_DPI})
            cluster_ax = fig.add_axes(mm_rect(x, cluster_y, umap_each, cluster_h))
            place_umap_asset(cluster_ax, cluster_path, xlim, ylim, DATASET_LABEL[dataset])
            if i == 0: panel_title(cluster_ax, 'D', 'Clustering')
            celltype_path = asset_dir / f'Figure_2_{dataset}_cell_type_UMAP.png'
            save_umap_asset(frame, 'author_cell_type', AUTHOR_COLORS, celltype_path, xlim, ylim, umap_each, celltype_h)
            asset_records.append({'filename': celltype_path.name, 'role': 'UMAP point cloud', 'dpi': OUTPUT_DPI})
            celltype_ax = fig.add_axes(mm_rect(x, celltype_y, umap_each, celltype_h))
            place_umap_asset(celltype_ax, celltype_path, xlim, ylim, DATASET_LABEL[dataset])
            if i == 0: panel_title(celltype_ax, 'F', 'Cell type')
            runtime_celltype_legend(fig.add_axes(mm_rect(x, legend_y, umap_each, legend_h)), frame, AUTHOR_COLORS)
        metric_ax = fig.add_axes(mm_rect(163, 135, 37, 79)); metric_cmap, metric_vmin, metric_vmax = runtime_metrics(metric_ax)
        vector_colorbar(fig, mm_rect(202, 141, 1.5, 65), metric_cmap, metric_vmin, metric_vmax)
        draw_tahoe_compiled(fig, 19, 50, 80, 34, 'TP53', 'G  TP53', asset_dir, asset_records)
        draw_tahoe_compiled(fig, 111, 50, 80, 34, 'PIK3R1', 'H  PIK3R1', asset_dir, asset_records)
    return fig, asset_records




FIGURE2_PANEL_SIZES_MM = {
    "A": (96, 57), "B": (38, 57), "C": (57, 57), "D": (130, 39),
    "E": (62, 86), "F": (130, 103), "G": (84, 40), "H": (84, 40),
}


def make_figure2_panel(label):
    # Plot one authoritative Figure 2 panel directly from validated data.
    global PAGE_W_MM, PAGE_H_MM
    if label not in FIGURE2_PANEL_SIZES_MM:
        raise ValueError(f"Unknown Figure 2 panel: {label}")
    previous_size = (PAGE_W_MM, PAGE_H_MM)
    PAGE_W_MM, PAGE_H_MM = FIGURE2_PANEL_SIZES_MM[label]
    width, height = PAGE_W_MM, PAGE_H_MM
    fig = plt.figure(
        figsize=(width / 25.4, height / 25.4),
        facecolor="white",
    )
    asset_records = []
    try:
        with tempfile.TemporaryDirectory(prefix=f"figure_2_panel_{label}_") as temp_dir:
            asset_dir = Path(temp_dir)
            if label == "A":
                draw_runtime_workflow(fig.add_axes(mm_rect(2, 3, width - 4, height - 6)))
            elif label == "B":
                draw_runtime_capacity_three_tier(fig, 2, 3, width - 4, height - 6)
            elif label == "C":
                draw_runtime_bars(fig.add_axes(mm_rect(3, 4, width - 6, height - 8)))
            elif label in {"D", "F"}:
                margin, gap = 2.0, 2.0
                each = (width - 2 * margin - 3 * gap) / 4
                for index, dataset in enumerate(SCKNIFE_DATASET_ORDER):
                    x = margin + index * (each + gap)
                    frame = load_umap("scknife", dataset)
                    xlim, ylim = _square_limits(frame)
                    if label == "D":
                        image_path = asset_dir / f"{dataset}_clustering.png"
                        save_umap_asset(frame, "leiden_cluster", None, image_path, xlim, ylim, each, height - 6)
                        axis = fig.add_axes(mm_rect(x, 3, each, height - 6))
                        place_umap_asset(axis, image_path, xlim, ylim, DATASET_LABEL[dataset])
                        if index == 0:
                            panel_title(axis, "D", "Clustering")
                    else:
                        plot_height = height * .43
                        image_path = asset_dir / f"{dataset}_cell_type.png"
                        save_umap_asset(frame, "author_cell_type", AUTHOR_COLORS, image_path, xlim, ylim, each, plot_height)
                        axis = fig.add_axes(mm_rect(x, height - plot_height - 3, each, plot_height))
                        place_umap_asset(axis, image_path, xlim, ylim, DATASET_LABEL[dataset])
                        if index == 0:
                            panel_title(axis, "F", "Cell type")
                        runtime_celltype_legend(fig.add_axes(mm_rect(x, 2, each, height - plot_height - 7)), frame, AUTHOR_COLORS)
            elif label == "E":
                metric_grid = fig.add_gridspec(
                    1, 2, width_ratios=(1, 0.055), wspace=0.12,
                    left=0.28, right=0.88, bottom=0.07, top=0.92,
                )
                metric_axis = fig.add_subplot(metric_grid[0, 0])
                metric_cmap, metric_vmin, metric_vmax = runtime_metrics(metric_axis)
                metric_colorbar_axis = fig.add_subplot(metric_grid[0, 1])
                metric_colorbar = fig.colorbar(
                    mpl.cm.ScalarMappable(
                        norm=mpl.colors.Normalize(metric_vmin, metric_vmax), cmap=metric_cmap,
                    ),
                    cax=metric_colorbar_axis,
                    ticks=np.linspace(metric_vmin, metric_vmax, 5),
                )
                metric_colorbar.ax.yaxis.set_ticks_position("right")
                metric_colorbar.ax.tick_params(labelsize=R_TICK, length=1, pad=1, width=.3)
            elif label in {"G", "H"}:
                gene = {"G": "TP53", "H": "PIK3R1"}[label]
                lineage = _data_manifest["panels"][label]
                expected_selectors = {gene, f"{gene}/expression", f"{gene}/bgnmf_score", f"{gene}/scanpy_score"}
                declared_selectors = set(lineage["inputs"][0].get("selectors", []))
                if lineage.get("title") != gene or declared_selectors != expected_selectors:
                    raise RuntimeError(f"Panel {label} lineage does not match plotted gene {gene}")
                draw_tahoe_compiled(fig, 3, 3, width - 6, height - 6, gene, f"{label}  {gene}", asset_dir, asset_records)
        return fig
    finally:
        PAGE_W_MM, PAGE_H_MM = previous_size

PANEL_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "panels"

def export_figure2_panel(fig, label):
    """Export one live Figure 2 panel as publication PNG and vector PDF."""
    if label not in FIGURE2_PANEL_SIZES_MM:
        raise ValueError(label)
    PANEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("png", "pdf"):
        path = PANEL_OUTPUT_DIR / f"Figure_2_{label}.{suffix}"
        fig.savefig(path, format=suffix, dpi=OUTPUT_DPI, facecolor="white", bbox_inches="tight", pad_inches=0.04)
        outputs[suffix] = path
    return outputs
