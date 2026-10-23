"""Build all Figure 7 panels A–O from retained scientific tables.

The approved prior page is a visual reference only. No SVG/PDF artwork is read.
This script does not rerun scKNIFE, CellChat, an embedding, or a statistical test.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import Normalize
from matplotlib import patheffects
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
DATA = ROOT / "data"
KNOWN = DATA / "known_new"
STEM = ROOT / "outputs/Figure_7_main_panel"
SUBMISSION = WORKSPACE / "May_2026_clean_version_redo/Figures/submission/figs/panel/Figure_7"
FINAL = WORKSPACE / "May_2026_clean_version_redo/Final_writeUps_18thSept/Final Figs /Main/Fig_07.pdf"
PAGE_PT = (595.276, 841.89)
DPI = 300
PATIENTS = [f"P{i}" for i in range(1, 8)]
PATIENT_COLORS = dict(zip(PATIENTS, ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#FFD92F", "#E5C494", "#B3B3B3"]))
STAGE_COLORS = ("#E69F00", "#56B4E9")
MAJOR_ORDER = ["Epithelial", "Endothelial", "CAFs", "PVL", "T-cells", "B-cells", "Plasmablasts", "Myeloid"]
MAJOR_COLORS = {"Epithelial": "#3E78BD", "Endothelial": "#56B4D8", "CAFs": "#845348", "PVL": "#E79B12", "T-cells": "#2C9D6E", "B-cells": "#D957A2", "Plasmablasts": "#9850B4", "Myeloid": "#D35A22"}
TEXT = 6.0
SMALL = 4.4
TITLE = 7.0
SCATTER = 0.22
UMAP_GRAY = "#e6e6e6"
UMAP_CMAP = "viridis"

# A4 portrait, normalized figure coordinates. Each panel is placed once here.
LAYOUT = {
    "a": {"axes": (.05, .795, .38, .155), "title": (.045, .968)},
    "b": {"axes": (.485, .865, .16, .08)},
    "c": {"axes": (.485, .790, .16, .07)},
    "d": {"axes": (.705, .815, .24, .13), "title": (.705, .968)},
    "e": {"axes": (.050, .560, .20, .21), "title": (.045, .785)},
    "f": {"axes": (.310, .560, .39, .21), "title": (.310, .785)},
    "g": {"axes": (.800, .610, .14, .13), "title": (.800, .765),
          "colorbar": (.948, .610, .006, .13)},
    "h": {"axes": (.050, .345, .16, .11), "title": (.045, .480),
          "colorbar": (.218, .350, .006, .10)},
    "i": {"axes": (.285, .345, .16, .11), "title": (.280, .480),
          "colorbar": (.453, .350, .006, .10)},
    "j": {"axes": (.520, .345, .16, .11), "title": (.515, .480),
          "colorbar": (.688, .350, .006, .10)},
    "k": {"axes": (.755, .345, .16, .11), "title": (.750, .480),
          "colorbar": (.923, .350, .006, .10)},
    "l": {"axes": (.075, .185, .225, .11), "title": (.055, .317)},
    "m": {"axes": (.375, .185, .225, .11), "title": (.355, .317)},
    "n": {"axes": (.675, .185, .225, .11), "title": (.655, .317)},
    "o": {"axes": (.250, .035, .69, .12), "title": (.055, .173)},
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": TEXT, "axes.labelsize": 5.7,
    "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})


def title(fig, letter, name, x, y, size=TITLE):
    fig.text(x, y, f"{letter}  {name}", fontsize=size, fontweight="bold", ha="left", va="top")


def axes(fig, rect, off=False):
    ax = fig.add_axes(rect)
    ax.grid(False)
    if off:
        ax.axis("off")
    return ax


def load_data():
    pathway = pd.read_csv(DATA / "pathway_overlap.csv")
    lr = pd.read_csv(DATA / "lr_harmonized_overlap.csv")
    shared = pd.read_csv(DATA / "strength_correlation_filtered.csv")
    major = pd.read_csv(KNOWN / "PDGF_major_matrix.csv", index_col=0).loc[MAJOR_ORDER, MAJOR_ORDER]
    minor = pd.read_csv(KNOWN / "PDGF_minor_matrix.csv", index_col=0)
    block = pd.read_csv(KNOWN / "PDGF_minor_endothelial_PVL_block.csv", index_col=0)
    top = pd.read_csv(KNOWN / "PDGF_minor_top10_connections.csv")
    umap = pd.read_parquet(KNOWN / "PDGF_umap_activity_expression.parquet")
    expression = pd.read_csv(KNOWN / "PDGF_major_expression.csv")
    paired = pd.read_csv(KNOWN / "PDGF_major_paired.csv")
    screen = pd.read_csv(KNOWN / "longitudinal_screen_major_tested.csv")
    assert pathway.status.value_counts().to_dict() == {"cellchat_only": 44, "shared": 32, "scknife_only": 33}
    assert lr.status.value_counts().to_dict() == {"cellchat_only": 194, "shared": 225, "scknife_only": 260}
    assert len(shared) == 1917 and (shared.cellchat_prob > 0).all() and (shared.scknife_score > 0).all()
    assert major.shape == (8, 8) and np.isclose(major.loc["Endothelial", "PVL"], 109.33530005939414)
    assert minor.shape == (29, 29) and block.equals(minor.loc[block.index, block.columns])
    assert np.isclose(block.loc["Endothelial RGS5", "PVL Immature"], 602.3858471232165)
    assert len(top) == 10 and tuple(top.loc[0, ["source", "target"]]) == ("Endothelial RGS5", "PVL Immature")
    assert len(umap) == 69319 and umap.cell_id.is_unique
    assert len(screen) == 2080 and not screen.q_bh.lt(.05).any()
    selected = screen.query("tier=='major' and pathway=='PDGF' and source=='Endothelial' and target=='PVL'")
    assert len(selected) == 1 and np.isclose(selected.p_exact.iloc[0], .109375)
    assert np.isclose(selected.q_bh.iloc[0], .6318347509113001)
    return dict(pathway=pathway, lr=lr, shared=shared, major=major, minor=minor,
                block=block, top=top, umap=umap, expression=expression, paired=paired,
                screen=screen, selected=selected.iloc[0])


def panel_a(fig):
    title(fig, "a", "", *LAYOUT["a"]["title"])
    ax = axes(fig, LAYOUT["a"]["axes"], off=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(patches.Circle((.07,.76),.055,facecolor="#fbd9d7",edgecolor="#d66c67",lw=.8))
    ax.add_patch(patches.Circle((.26,.76),.055,facecolor="#cbeee1",edgecolor="#4aab91",lw=.8))
    ax.annotate("",(.19,.76),(.13,.76),arrowprops={"arrowstyle":"->","lw":.8,"color":"#777"})
    ax.text(.165,.89,"ligand × receptor",ha="center",fontsize=5.5)
    ax.text(.165,.62,"sender        receiver",ha="center",fontsize=4.5,color="#555")
    for row in range(6):
        for col in range(6):
            color=plt.cm.Greens(.18+.7*((row+2*col)%8)/8)
            ax.add_patch(patches.Rectangle((.39+col*.026,.66+row*.026),.024,.024,facecolor=color,edgecolor="white",lw=.1))
    ax.text(.465,.60,"cell × cell score matrix",ha="center",fontsize=4.6)
    xx=np.linspace(0,1,60); yy=np.exp(-((xx-.52)/.19)**2)
    ax.plot(.69+.22*xx,.69+.15*yy,color="#6caa95",lw=.8)
    ax.fill_between(.69+.22*xx,.69,.69+.15*yy,color="#ceeadd")
    ax.axvline(.69+.22*.87,ymin=.69,ymax=.87,color="#ce6966",lw=.8)
    ax.text(.80,.60,"shuffle → null / FDR",ha="center",fontsize=4.6)
    pts=np.array([[.08,.19],[.19,.30],[.30,.18],[.42,.32],[.54,.18]])
    for i,j in ((0,1),(1,2),(1,3),(3,4)):
        ax.plot(pts[[i,j],0],pts[[i,j],1],color="#b3b3b3",lw=.7)
    ax.scatter(pts[:,0],pts[:,1],s=[21,32,25,44,25],c=["#5073b7","#9cc7d8","#dcad9d","#3c9977","#9070ab"],zorder=2)
    ax.text(.26,.05,"network centrality",ha="center",fontsize=4.6)
    for row in range(3):
        for col in range(4):
            ax.add_patch(patches.Rectangle((.63+col*.04,.18+row*.035),.037,.032,
                facecolor=plt.cm.RdYlBu(.15+.7*((row*4+col)%5)/5),edgecolor="white",lw=.1))
    ax.text(.69,.05,"communication strength",ha="center",fontsize=4.6)


def venn(fig, rect, counts, letter):
    ax=axes(fig,rect,off=True); ax.set_xlim(0,1);ax.set_ylim(0,1)
    left,shared,right=counts
    ax.add_patch(patches.Circle((.42,.54),.30,facecolor="#55bddd",alpha=.76,lw=.5,edgecolor="#333"))
    ax.add_patch(patches.Circle((.61,.54),.30,facecolor="#ed6d85",alpha=.76,lw=.5,edgecolor="#333"))
    for x,n in ((.28,left),(.515,shared),(.75,right)):
        ax.text(x,.54,str(n),ha="center",va="center",fontsize=7.2)
    ax.text(.33,.14,"CellChat",ha="center",fontsize=5.8)
    ax.text(.73,.14,"scKNIFE",ha="center",fontsize=5.8)
    title(fig,letter,"",rect[0],rect[1]+rect[3]+.006)


def panel_d(fig, shared):
    rect=LAYOUT["d"]["axes"]
    ax=axes(fig,rect)
    x=np.log10(shared.cellchat_prob.to_numpy(float)); y=np.log10(shared.scknife_score.to_numpy(float))
    ax.scatter(x,y,s=.7,c="#55b9d5",alpha=.75,rasterized=True,linewidths=0)
    slope,intercept=np.polyfit(x,y,1)
    xx=np.array([x.min(),x.max()]);ax.plot(xx,slope*xx+intercept,color="#777",lw=.7,ls="--")
    ax.set_xlabel("log10(CellChat probability)",fontsize=5.2)
    ax.set_ylabel("log10(scKNIFE score)",fontsize=5.2)
    ax.tick_params(length=2,pad=2)
    ax.text(.72,.15,"n = 1,917\nPearson r = 0.732\nSpearman ρ = 0.746",transform=ax.transAxes,fontsize=4.8,ha="left")
    assert np.isclose(np.corrcoef(x,y)[0,1],.7315651849696532,atol=1e-6)
    title(fig,"d","",*LAYOUT["d"]["title"])


def panel_e(fig, major, umap):
    rect=LAYOUT["e"]["axes"]
    ax=axes(fig,rect,off=True);ax.set_xlim(0,1);ax.set_ylim(0,1)
    title(fig,"e","",*LAYOUT["e"]["title"])
    pos={"Epithelial":(.51,.86),"Endothelial":(.82,.74),"CAFs":(.93,.46),"PVL":(.75,.18),
         "T-cells":(.49,.08),"B-cells":(.24,.15),"Plasmablasts":(.08,.45),"Myeloid":(.17,.75)}
    edges=major.stack().sort_values(ascending=False).head(15)
    for (source,target),score in reversed(list(edges.items())):
        x0,y0=pos[source];x1,y1=pos[target]
        color="#be1e2d" if (source,target)==("Endothelial","PVL") else "#909090"
        width=2.4 if color=="#be1e2d" else .35+1.5*np.sqrt(float(score)/float(edges.max()))
        if source==target:
            ax.add_patch(patches.Arc((x0+.045,y0+.045),.12,.10,theta1=20,theta2=315,color=color,lw=width,alpha=.8))
        else:
            ax.add_patch(patches.FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",
                mutation_scale=4,connectionstyle="arc3,rad=.14",color=color,lw=width,alpha=.75,
                shrinkA=7,shrinkB=7,zorder=1))
    counts=umap.major_celltype.value_counts()
    for name,(x,y) in pos.items():
        size=20+30*np.sqrt(counts[name]/counts.max())
        ax.scatter(x,y,s=size,color=MAJOR_COLORS[name],edgecolor="white",linewidth=.5,zorder=3)
        ha="left" if x<.5 else "right"
        dx=.035 if x<.5 else -.035
        ax.text(x+dx,y,name,ha=ha,va="center",fontsize=4.6)


def panel_f_g(fig, minor, block):
    vals=np.log1p(minor.to_numpy(float)); vmax=np.log1p(float(minor.to_numpy().max()))
    f_rect=LAYOUT["f"]["axes"]
    ax=axes(fig,f_rect)
    ax.imshow(vals,cmap="YlOrRd",vmin=0,vmax=vmax,origin="upper",aspect="auto",interpolation="nearest",rasterized=False)
    labels=[x.replace("Endothelial ","Endo ").replace("CAFs ","CAF ").replace("Cancer ","Ca ").replace("Lymphatic ","ly. ").replace("Differentiated","diff.").replace("Immature","imm.") for x in minor.index]
    ax.set_xticks(range(29),labels,rotation=90,fontsize=4.0)
    ax.set_yticks(range(29),labels,fontsize=4.0)
    ax.tick_params(length=0,pad=1)
    ax.set_xlabel("Receiver",fontsize=5.2);ax.set_ylabel("Sender",fontsize=5.2)
    ax.add_patch(patches.Rectangle((9.5,12.5),3,4,fill=False,edgecolor="#4b7c96",lw=.9,ls="--"))
    title(fig,"f","",*LAYOUT["f"]["title"])
    g_rect=LAYOUT["g"]["axes"]
    bx=axes(fig,g_rect)
    bx.imshow(np.log1p(block.to_numpy(float)),cmap="YlOrRd",vmin=0,vmax=vmax,origin="upper",aspect="auto",interpolation="nearest")
    bx.set_xticks(range(3),["PVL\nCyc.","PVL\nDiff.","PVL\nImm."],fontsize=4.1)
    bx.set_yticks(range(4),["Endo ACKR1","Endo CXCL12","Endo ly. LYVE1","Endo RGS5"],fontsize=4.1)
    bx.tick_params(length=0,pad=1)
    for i in range(4):
        for j in range(3):
            v=block.iloc[i,j]; bx.text(j,i,f"{v:.1f}",ha="center",va="center",fontsize=4.1,
                color="white" if np.log1p(v)>vmax*.60 else "#343434",fontweight="bold")
    bx.add_patch(patches.Rectangle((1.5,2.5),1,1,fill=False,edgecolor="#5d0000",lw=1.0))
    title(fig,"g","",*LAYOUT["g"]["title"])
    # The zoom is an exact labelled slice, not a thresholded second matrix.
    f_right=f_rect[0]+f_rect[2]
    g_left=g_rect[0]
    fig.lines.extend([mpl.lines.Line2D([f_right,g_left],[.745,.740],transform=fig.transFigure,color="#7ba3af",ls="--",lw=.5),
                      mpl.lines.Line2D([f_right,g_left],[.600,.610],transform=fig.transFigure,color="#7ba3af",ls="--",lw=.5)])
    cax=axes(fig,LAYOUT["g"]["colorbar"]); cb=fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0,vmax),cmap="YlOrRd"),cax=cax)
    cb.set_ticks([0,vmax]);cb.set_ticklabels(["0",f"{minor.to_numpy().max():.0f}"]);cb.ax.tick_params(labelsize=3.7,length=1)


def panel_umaps(fig, umap):
    panels=[("h","PDGFB activity","PDGFB_activity"),
            ("i","PDGFB expression","PDGFB_expression"),
            ("j","PDGFRB activity","PDGFRB_activity"),
            ("k","PDGFRB expression","PDGFRB_expression")]
    xx=umap.umap_1.to_numpy(float);yy=umap.umap_2.to_numpy(float)
    centroids=umap.loc[umap.major_celltype.isin(["Endothelial","PVL"])].groupby("major_celltype")[["umap_1","umap_2"]].median()
    for label,name,column in panels:
        ax=axes(fig,LAYOUT[label]["axes"])
        value=umap[column].to_numpy(float)
        positive=value>0
        hi=float(np.quantile(value[positive],.98))
        lo=float(np.quantile(value[positive],.02))
        ax.scatter(xx,yy,s=SCATTER,c=UMAP_GRAY,linewidths=0,rasterized=True)
        image=ax.scatter(xx[positive],yy[positive],s=SCATTER,c=np.clip(value[positive],lo,hi),
                         cmap=UMAP_CMAP,vmin=lo,vmax=hi,linewidths=0,rasterized=True)
        ax.set_xlim(xx.min()-.5,xx.max()+.5);ax.set_ylim(yy.min()-.5,yy.max()+.5)
        ax.set_aspect("equal",adjustable="box");ax.axis("off")
        halo=[patheffects.withStroke(linewidth=1.3,foreground="white")]
        for celltype,text_name in (("Endothelial","Endo"),("PVL","PVL")):
            point=centroids.loc[celltype]
            ax.text(float(point.umap_1),float(point.umap_2),text_name,
                    fontsize=4.8,fontweight="bold",color="#222",path_effects=halo)
        cax=axes(fig,LAYOUT[label]["colorbar"]);cb=fig.colorbar(image,cax=cax)
        cb.ax.tick_params(labelsize=4.0,length=1,pad=1)
        title(fig,label,name,*LAYOUT[label]["title"],size=6.0)
    fig.text(.50,.331,"UMAP 1",fontsize=4.8,ha="center")
    fig.text(.030,.40,"UMAP 2",fontsize=4.8,rotation=90,va="center")


def patient_frames(expression, paired):
    expression=expression.query("tier=='major' and stage in ['Bx','Surgery']")
    out=[]
    for label,name,gene,celltype in (("l","Endothelial PDGFB expression","PDGFB","Endothelial"),
                                     ("m","PVL PDGFRB expression","PDGFRB","PVL")):
        frame=expression.query("gene==@gene and celltype==@celltype").pivot(
            index="display_id",columns="stage",values="mean_log1p").loc[PATIENTS,["Bx","Surgery"]]
        assert frame.shape==(7,2) and np.isfinite(frame.to_numpy()).all()
        out.append((label,name,frame))
    selected=paired.query("tier=='major' and pathway=='PDGF' and source=='Endothelial' and target=='PVL'")
    assert len(selected)==7 and selected.eligible.all()
    frame=selected.set_index("display_id").loc[PATIENTS,["score_bx","score_surgery"]]
    frame.columns=["Bx","Surgery"]
    assert (frame.Surgery<frame.Bx).sum()==6 and frame.loc["P1","Surgery"]>frame.loc["P1","Bx"]
    out.append(("n","Endothelial→PVL PDGF score",frame))
    return out


def violin_density(values):
    """Positive-support KDE with smooth tails beyond the observed values."""
    values = np.asarray(values, dtype=float)
    kde = gaussian_kde(values)
    bandwidth = float(kde.factor * values.std(ddof=1))
    minimum, maximum = float(values.min()), float(values.max())
    lower = max(0.0, minimum - 1.5 * bandwidth)
    upper = maximum + 1.5 * bandwidth
    coords = np.linspace(lower, upper, 256)
    density = kde(coords)
    # The density is tapered only outside the observed range. The zero-width
    # endpoints avoid Matplotlib's flat caps at the observed extrema.
    below = coords < minimum
    above = coords > maximum
    density[below] *= np.sin(0.5 * np.pi * (coords[below] - lower) / (minimum - lower)) ** 2
    density[above] *= np.sin(0.5 * np.pi * (upper - coords[above]) / (upper - maximum)) ** 2
    density[0] = density[-1] = 0.0
    half_width = 0.16 * density / density.max()
    assert coords[0] >= 0 and half_width[0] == half_width[-1] == 0
    return coords, half_width


def panel_violins(fig, expression, paired):
    frames=patient_frames(expression,paired)
    for label,name,frame in frames:
        title(fig,label,name,*LAYOUT[label]["title"],size=6.2)
        ax=axes(fig,LAYOUT[label]["axes"])
        vals=[frame.Bx.to_numpy(float),frame.Surgery.to_numpy(float)]
        supports=[]
        for position,(group,color) in enumerate(zip(vals,STAGE_COLORS)):
            coords,half_width=violin_density(group)
            ax.fill_betweenx(coords,position-half_width,position+half_width,
                             facecolor=color,edgecolor="none",alpha=.20)
            ax.plot(position-half_width,coords,color=color,alpha=.55,lw=.4)
            ax.plot(position+half_width,coords,color=color,alpha=.55,lw=.4)
            supports.append((coords[0],coords[-1]))
        for patient in PATIENTS:
            pair=frame.loc[patient,["Bx","Surgery"]].to_numpy(float)
            ax.plot([0,1],pair,color=PATIENT_COLORS[patient],lw=.65,
                    ls=":" if patient=="P1" else "-",zorder=3)
            ax.scatter([0,1],pair,color=PATIENT_COLORS[patient],s=6,lw=0,zorder=4)
        for j,group in enumerate(vals):
            med=float(np.median(group))
            ax.plot([j-.11,j+.11],[med,med],color="#202020",lw=.8,zorder=5)
        lo=min(x[0] for x in supports);hi=max(x[1] for x in supports);pad=.05*(hi-lo)
        ax.set_ylim(max(0.0,lo-pad),hi+pad);ax.set_xlim(-.32,1.18)
        ax.set_xticks([0,1],["Bx","Surgery"]);ax.tick_params(length=2,pad=1,labelsize=5)
        ax.set_ylabel("Mean log1p" if label in "lm" else "Absolute score",fontsize=5)
        if label=="m":
            ax.set_xlabel("PVL",fontsize=5,labelpad=1)
    fig.text(.055,.299,"Patients:",fontsize=4.7,va="center")
    for j,patient in enumerate(PATIENTS):
        fig.text(.110+j*.035,.299,patient,color=PATIENT_COLORS[patient],fontsize=4.7,fontweight="bold",va="center")
    fig.text(.370,.299,"P1 dotted",fontsize=4.7,va="center")
    fig.text(.655,.299,"P=0.1094 · BH q=0.6318",fontsize=5.4,va="center")


def panel_o(fig, top):
    title(fig,"o","",*LAYOUT["o"]["title"])
    ax=axes(fig,LAYOUT["o"]["axes"])
    labels=(top.source+" → "+top.target).tolist()
    ypos=np.arange(len(top))
    colors=["#bf2034"]+["#bcbcbc"]*9
    ax.barh(ypos,top.score,color=colors,height=.74)
    ax.set_yticks(ypos,labels,fontsize=5.0)
    ax.invert_yaxis();ax.tick_params(axis="y",length=0,pad=2)
    ax.tick_params(axis="x",labelsize=5.0,length=2)
    ax.set_xlim(0,700)
    ax.set_xlabel("Pooled scKNIFE PDGF pathway score",fontsize=5)


def make_figure(data):
    fig=plt.figure(figsize=(PAGE_PT[0]/72,PAGE_PT[1]/72),facecolor="white")
    panel_a(fig)
    venn(fig,LAYOUT["b"]["axes"],(44,32,33),"b")
    venn(fig,LAYOUT["c"]["axes"],(194,225,260),"c")
    panel_d(fig,data["shared"])
    panel_e(fig,data["major"],data["umap"])
    panel_f_g(fig,data["minor"],data["block"])
    panel_umaps(fig,data["umap"])
    panel_violins(fig,data["expression"],data["paired"])
    panel_o(fig,data["top"])
    fig.text(.942,.013,"Figure 7",ha="right",fontsize=5.4,color="#333")
    return fig


def render():
    data=load_data()
    fig=make_figure(data)
    STEM.parent.mkdir(parents=True,exist_ok=True)
    for extension in ("pdf","svg","png"):
        fig.savefig(STEM.with_suffix("."+extension),format=extension,dpi=DPI,
                    bbox_inches=None,pad_inches=0,facecolor="white")
    plt.close(fig)
    SUBMISSION.mkdir(parents=True,exist_ok=True)
    for extension in ("pdf","svg","png"):
        shutil.copyfile(STEM.with_suffix("."+extension),SUBMISSION/f"Figure_7_main_panel.{extension}")
    FINAL.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(STEM.with_suffix(".pdf"),FINAL)
    source_files=[DATA/"pathway_overlap.csv",DATA/"lr_harmonized_overlap.csv",DATA/"strength_correlation_filtered.csv",
                  KNOWN/"PDGF_major_matrix.csv",KNOWN/"PDGF_minor_matrix.csv",
                  KNOWN/"PDGF_minor_endothelial_PVL_block.csv",KNOWN/"PDGF_minor_top10_connections.csv",
                  KNOWN/"PDGF_umap_activity_expression.parquet",KNOWN/"PDGF_major_expression.csv",
                  KNOWN/"PDGF_major_paired.csv",KNOWN/"longitudinal_screen_major_tested.csv"]
    manifest={"figure":"Figure 7 A–O","active_panels":list("ABCDEFGHIJKLMNO"),
              "renderer":"Matplotlib: all panels drawn from saved tables; dense UMAP/scatter layers rasterized",
              "layout":"balanced A4 portrait rows: A–D; E–G; H–K; L–N; O",
              "source_layout":None,"source_artwork_used":False,
              "source_tables":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
              "selected_n":{"p_exact":.109375,"q_bh":.6318347509113001,"screen_size":2080},
              "violin_density":"Gaussian KDE, 1.5 bandwidth tails, tapered to zero, nonnegative support",
              "formats":{ext:{"sha256":hashlib.sha256(STEM.with_suffix("."+ext).read_bytes()).hexdigest(),
                               "bytes":STEM.with_suffix("."+ext).stat().st_size} for ext in ("pdf","svg","png")},
              "python_matplotlib":mpl.__version__}
    for dest in (ROOT/"outputs/Figure_7_current_export_manifest.json",ROOT/"outputs/Figure_7_main_export_manifest.json",
                 SUBMISSION/"Figure_7_current_export_manifest.json",SUBMISSION/"Figure_7_main_export_manifest.json"):
        dest.write_text(json.dumps(manifest,indent=2)+"\n")
    with Image.open(STEM.with_suffix(".png")) as image:
        assert image.width>=2400 and image.height>=3400


if __name__=="__main__":
    render()
