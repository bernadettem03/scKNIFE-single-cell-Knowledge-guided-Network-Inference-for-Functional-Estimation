"""Plotting-only renderer for Supplementary Figure 1."""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

MODULE_ROOT=Path(__file__).resolve().parent
DATA_ROOT=MODULE_ROOT/"data"
POOLED_ROOT=DATA_ROOT/"pooled_w"
MANIFEST_PATH=DATA_ROOT/"manifest.json"
METRICS_PATH=POOLED_ROOT/"w_pooled_concordance_metrics.csv"
NODE_KINDS_PATH=POOLED_ROOT/"w_pooled_tsne_node_kinds_corrected.png"
POOLED_CLUSTERS_PATH=POOLED_ROOT/"w_pooled_tsne_dataset_panels.png"
PAGE_W_MM,PAGE_H_MM=210,297
OUTPUT_DPI=300
LABEL_PT=7
TITLE_PT=7
TICK_PT=5.2
NOTE_PT=4.5
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":7,"axes.grid":False,"pdf.fonttype":42,"svg.fonttype":"none"})

def _sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()

def _rect(x,y,w,h):return [x/PAGE_W_MM,y/PAGE_H_MM,w/PAGE_W_MM,h/PAGE_H_MM]

def validate_inputs():
 manifest=json.loads(MANIFEST_PATH.read_text())
 if manifest.get("schema_version")!="supplementary_figure_s1_v1" or manifest.get("panel_order")!=list("ABC"):
  raise RuntimeError("Invalid Supplementary Figure 1 manifest")
 for name,entry in manifest["assets"].items():
  path=POOLED_ROOT/name
  if not path.is_file() or path.stat().st_size!=entry["bytes"] or _sha(path)!=entry["sha256"]:
   raise RuntimeError(f"S1 asset mismatch: {name}")
 metrics=pd.read_csv(METRICS_PATH)
 if set(metrics.record_type)!={"dataset","pair"}:raise RuntimeError("Invalid concordance metrics")
 return {"panels":list("ABC"),"assets":len(manifest["assets"]),"plotting_only":True}

def _draw_concordance(fig):
 metrics=pd.read_csv(METRICS_PATH)
 order=["Bassez_A","Bassez_B","Shiao","Wu"]
 labels={"Bassez_A":"Bassez C1","Bassez_B":"Bassez C2","Shiao":"Shiao","Wu":"Wu TNBC"}
 matrix=np.eye(4,dtype=float)
 for row in metrics.loc[metrics.record_type.eq("pair")].itertuples():
  i,j=order.index(row.dataset_a),order.index(row.dataset_b);matrix[i,j]=matrix[j,i]=float(row.linear_cka)
 cmap,norm=plt.get_cmap("viridis"),mpl.colors.Normalize(0,1)
 fig.text(10/PAGE_W_MM,270/PAGE_H_MM,"A  Cross-dataset W concordance",fontsize=TITLE_PT,fontweight="bold")
 ax=fig.add_axes(_rect(12,207,50,50))
 for i in range(4):
  for j in range(4):
   value=matrix[i,j];ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=cmap(norm(value)),edgecolor="white",linewidth=.35));ax.text(j,i,f"{value:.3f}",ha="center",va="center",fontsize=TICK_PT,color="white" if value>=.58 else "black")
 ax.set(xlim=(-.5,3.5),ylim=(3.5,-.5),xlabel="Dataset",ylabel="Dataset");ax.set_xticks(range(4),[labels[d] for d in order],rotation=35,ha="right",fontsize=TICK_PT);ax.set_yticks(range(4),[labels[d] for d in order],fontsize=TICK_PT);ax.tick_params(length=0)
 for spine in ax.spines.values():spine.set_visible(False)
 cax=fig.add_axes(_rect(65,207,3,50))
 for step in range(100):cax.add_patch(Rectangle((0,step/100),1,.01,facecolor=cmap((step+.5)/100),edgecolor="none"))
 cax.set(xlim=(0,1),ylim=(0,1),ylabel="Linear CKA");cax.set_xticks([]);cax.set_yticks(np.linspace(0,1,6));cax.tick_params(labelsize=TICK_PT,length=1.5);cax.yaxis.set_label_position("right");cax.yaxis.tick_right();cax.yaxis.label.set_size(TICK_PT)
 table=fig.add_axes(_rect(80,203,118,58),frameon=False);table.set_axis_off();table.set_title("Within-dataset factor diagnostics",loc="left",fontsize=TITLE_PT,fontweight="bold")
 headers=("Dataset","Factors","Eff. rank","Med. |r|","P95 |r|","Condition");xpos=(0,.24,.44,.65,.82,1.0)
 for x,value in zip(xpos,headers):table.text(x,.84,value,ha="left" if x==0 else "right",fontsize=NOTE_PT,fontweight="bold",transform=table.transAxes)
 table.plot((0,1),(.78,.78),color="#555555",linewidth=.5,transform=table.transAxes);stats=metrics.loc[metrics.record_type.eq("dataset")].set_index("dataset_a")
 for y,dataset in zip((.64,.49,.34,.19),order):
  row=stats.loc[dataset];values=(labels[dataset],f"{int(row.n_factors)}",f"{row.effective_rank:.1f} ({100*row.effective_rank_fraction:.0f}%)",f"{row.median_abs_factor_correlation:.3f}",f"{row.p95_abs_factor_correlation:.3f}",f"{row.condition_number:.2f}")
  for x,value in zip(xpos,values):table.text(x,y,value,ha="left" if x==0 else "right",fontsize=NOTE_PT,transform=table.transAxes)
 table.add_patch(Rectangle((0,.01),1,.10,facecolor="#F1F5F4",edgecolor="#8DA6A0",linewidth=.5,transform=table.transAxes));table.text(.5,.06,"Concordant W structure with limited factor redundancy",ha="center",va="center",fontsize=NOTE_PT,fontweight="bold",color="#244B45",transform=table.transAxes)
 fig.text(12/PAGE_W_MM,196/PAGE_H_MM,"CKA quantifies shared node geometry; effective rank and factor correlations quantify within-run redundancy.",fontsize=NOTE_PT,color="#444444")

def _draw_raster(fig,path,label,title,y):
 fig.text(10/PAGE_W_MM,(y+69)/PAGE_H_MM,f"{label}  {title}",fontsize=TITLE_PT,fontweight="bold")
 image=plt.imread(path);height=190*image.shape[0]/image.shape[1]
 ax=fig.add_axes(_rect(10,y,190,height));ax.imshow(image,aspect="equal",interpolation="none");ax.axis("off")

def make_figure_s1():
 validate_inputs();fig=plt.figure(figsize=(8.27,11.69),facecolor="white")
 fig.text(12/PAGE_W_MM,285/PAGE_H_MM,"Supplementary Figure 1 | Cross-dataset W concordance and pooled W-space structure",ha="left",va="bottom",fontsize=LABEL_PT)
 _draw_concordance(fig)
 _draw_raster(fig,NODE_KINDS_PATH,"B","t-SNE node kinds across independently normalized W blocks",119)
 _draw_raster(fig,POOLED_CLUSTERS_PATH,"C","t-SNE pooled clusters across independently normalized W blocks",36)
 return fig
