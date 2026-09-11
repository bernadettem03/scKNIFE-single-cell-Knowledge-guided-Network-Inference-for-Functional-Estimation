"""Plotting-only renderer for Supplementary Figure 2 no-match UMAPs."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

MODULE_ROOT=Path(__file__).resolve().parent
DATASET_ORDER=["bassez_c1","bassez_c2","wu_tnbc"]
DATASET_LABEL={"bassez_c1":"Bassez C1","bassez_c2":"Bassez C2","wu_tnbc":"Wu TNBC"}
METHODS=[("expimap","ExpiMap"),("slalom","Slalom"),("spectra","SPECTRA")]
PAGE_W_MM,PAGE_H_MM=210,297
OUTPUT_DPI=300
LABEL_PT=7
TITLE_PT=7
LEGEND_PT=7
LEGEND_MARKER_PT=3.5
DATA_DIR=MODULE_ROOT/"data/no_match_umaps"
MANIFEST_PATH=DATA_DIR/"no_match_umap_manifest.json"
UMAP_FILES={(method,dataset):DATA_DIR/f"{method}_{dataset}_points.png" for method,_ in METHODS for dataset in DATASET_ORDER}
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":7,"axes.grid":False,"pdf.fonttype":42,"svg.fonttype":"none"})

def _sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()

def _natural(value):return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)",str(value))]
def _rect(x,y,w,h):return [x/PAGE_W_MM,y/PAGE_H_MM,w/PAGE_W_MM,h/PAGE_H_MM]

def validate_inputs():
 manifest=json.loads(MANIFEST_PATH.read_text())
 if manifest.get("schema_version")!="no_match_umap_raster_v3" or manifest.get("source_mode")!="validated_frozen_figure2_bundle":raise RuntimeError("Invalid S2 UMAP manifest")
 expected={f"{method}/{dataset}" for method,_ in METHODS for dataset in DATASET_ORDER}
 if set(manifest.get("outputs",{}))!=expected:raise RuntimeError("S2 manifest must contain exactly nine method/dataset UMAPs")
 source_root=MODULE_ROOT.parents[1]/"Figure_2/data"
 for method,_ in METHODS:
  for dataset in DATASET_ORDER:
   key=f"{method}/{dataset}";entry=manifest["outputs"][key];raster=UMAP_FILES[(method,dataset)];table=source_root/f"{method}_{dataset}_umap.csv.gz"
   if not raster.is_file() or _sha(raster)!=entry["combined_png_sha256"]:raise RuntimeError(f"S2 raster mismatch: {key}")
   if not table.is_file() or _sha(table)!=entry["source_sha256"]:raise RuntimeError(f"S2 source mismatch: {key}")
   frame=pd.read_csv(table,usecols=["leiden_cluster","author_cell_type"],dtype=str)
   if len(frame)!=entry["cell_count"] or set(frame.leiden_cluster)!=set(entry["leiden_colors"]) or set(frame.author_cell_type)!=set(entry["author_colors"]):raise RuntimeError(f"S2 labels mismatch: {key}")
 return {"datasets":[DATASET_LABEL[x] for x in DATASET_ORDER],"methods":[x[1] for x in METHODS],"panels":9,"plotting_only":True}

def _legend_specs(manifest,method,dataset):
 entry=manifest["outputs"][f"{method}/{dataset}"];leiden=sorted(entry["leiden_colors"],key=lambda x:int(x));author=sorted(entry["author_colors"],key=_natural)
 clusters=[Line2D([],[],marker="o",linestyle="",markersize=LEGEND_MARKER_PT,markerfacecolor=entry["leiden_colors"][x],markeredgecolor="none",label=x) for x in leiden]
 authors=[Line2D([],[],marker="o",linestyle="",markersize=LEGEND_MARKER_PT,markerfacecolor=entry["author_colors"][x],markeredgecolor="none",label=x) for x in author]
 return clusters,authors

def _overlay(fig,title_rect,legend_rect,manifest,method,dataset):
 clusters,authors=_legend_specs(manifest,method,dataset);title=fig.add_axes(title_rect,frameon=False);title.set_axis_off();title.text(.25,.5,"Leiden",ha="center",va="center",fontsize=TITLE_PT);title.text(.75,.5,"Author cell type",ha="center",va="center",fontsize=TITLE_PT)
 legend=fig.add_axes(legend_rect,frameon=False);legend.set_axis_off();common={"frameon":False,"fontsize":LEGEND_PT,"handlelength":.55,"handletextpad":.20,"borderaxespad":0,"borderpad":0,"labelspacing":.08,"columnspacing":.45,"markerscale":1.0}
 first=legend.legend(handles=clusters,loc="upper left",bbox_to_anchor=(0,1),ncol=2 if len(clusters)>8 else 1,**common);legend.add_artist(first);legend.legend(handles=authors,loc="upper left",bbox_to_anchor=(.52,1),ncol=1,**common)

def make_figure_s2():
 audit=validate_inputs();manifest=json.loads(MANIFEST_PATH.read_text());fig=plt.figure(figsize=(8.27,11.69),facecolor="white")
 fig.text(12/PAGE_W_MM,285/PAGE_H_MM,"Supplementary Figure 2 | Frozen no-match UMAPs: Bassez C1, Bassez C2, and Wu TNBC",ha="left",va="bottom",fontsize=LABEL_PT)
 x0,y_top=15,170;cell_w,cell_h=58,40;plot_h,legend_h,title_h=18,18,4;col_gap,row_gap=4,4
 for col,dataset in enumerate(DATASET_ORDER):fig.text((x0+col*(cell_w+col_gap)+cell_w/2)/PAGE_W_MM,214/PAGE_H_MM,DATASET_LABEL[dataset],ha="center",va="bottom",fontsize=TITLE_PT,fontweight="bold")
 for row,(method,label) in enumerate(METHODS):
  y=y_top-row*(cell_h+row_gap);fig.text(11/PAGE_W_MM,(y+cell_h/2)/PAGE_H_MM,label,ha="center",va="center",rotation=90,fontsize=TITLE_PT,fontweight="bold")
  for col,dataset in enumerate(DATASET_ORDER):
   x=x0+col*(cell_w+col_gap);ax=fig.add_axes(_rect(x,y+legend_h,cell_w,plot_h));ax.imshow(plt.imread(UMAP_FILES[(method,dataset)]),aspect="auto",interpolation="none");ax.axis("off");_overlay(fig,_rect(x,y+legend_h+plot_h,cell_w,title_h),_rect(x,y,cell_w,legend_h),manifest,method,dataset)
 return fig
