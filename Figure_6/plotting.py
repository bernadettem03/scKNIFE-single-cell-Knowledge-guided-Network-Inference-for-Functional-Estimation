#!/usr/bin/env python3
"""Publication plotting for biology-first Figure 6 panels A–G."""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# Figure parameters: one editable source of truth.
MAIN_WIDTH_MM = 180
MAIN_HEIGHT_MM = 410
SUPPLEMENT_WIDTH_MM = 180
SUPPLEMENT_HEIGHT_MM = 650
STANDALONE_WIDTH_MM = 112
STANDALONE_HEIGHT_MM = 82
TEXT_SIZE_PT = 8
AXIS_TITLE_SIZE_PT = 9
PANEL_LABEL_SIZE_PT = 10
TICK_LABEL_SIZE_PT = 7
LEGEND_SIZE_PT = 7
LINE_WIDTH = 1.0
POINT_SIZE = 28
UPSET_DOT_SIZE = 36
UPSET_EMPTY_DOT_SIZE = 34
UPSET_BAR_WIDTH = 0.62
UPSET_SET_BAR_HEIGHT = 0.42
UPSET_CONNECTOR_WIDTH = 1.4
BOX_WIDTH = 0.38
BOX_ALPHA = 0.18
OUTPUT_DPI = 300
TOP_PATHWAYS = 10
NOMINAL_P_CUTOFF = 0.05
COLORS = {
    "biopsy": "#377EB8",
    "surgery": "#E41A1C",
    "nonmetastatic": "#4DAF4A",
    "metastatic": "#E41A1C",
    "pathway": "#3B6FB6",
    "lcms": "#D55E00",
    "gpa_specific": "#009E73",
    "scKNIFE": "#CC79A7",
    "mapped": "#4C78A8",
    "unmatched": "#B8B8B8",
    "dark": "#303030",
    "gray": "#6A6A6A",
    "light": "#D9D9D9",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": TEXT_SIZE_PT,
    "axes.titlesize": AXIS_TITLE_SIZE_PT,
    "axes.labelsize": AXIS_TITLE_SIZE_PT,
    "xtick.labelsize": TICK_LABEL_SIZE_PT,
    "ytick.labelsize": TICK_LABEL_SIZE_PT,
    "legend.fontsize": LEGEND_SIZE_PT,
    "axes.grid": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

def _mm(value: float) -> float:
    return value / 25.4

def _clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.8, length=3)

def _wrap(labels, width=34):
    return [textwrap.fill(str(label), width=width) for label in labels]

def _panel_title(container, label: str, title: str, subtitle: str | None = None) -> None:
    container.suptitle(
        f"{label}  {title}", x=0.01, y=0.995, ha="left", va="top",
        fontsize=PANEL_LABEL_SIZE_PT, fontweight="bold", color=COLORS["dark"],
    )
    if subtitle:
        container.text(
            0.01, 0.945, subtitle, ha="left", va="top",
            fontsize=TICK_LABEL_SIZE_PT, color=COLORS["gray"],
        )

def _validated_integrated_pathway_scatter(data,scatter_file,enrichment_file,membership_file,contrast,expected_rows,expected_discoveries):
    table=data[scatter_file].copy().sort_values("rank")
    enrichment=data[enrichment_file].copy()
    membership=data[membership_file].copy()
    scatter_required={"rank","pathway","enrichment_factor","neg_log10_p","radius","label_top5","coordinate_source"}
    enrichment_required={"candidate_pathway","pathway_BH_FDR_gamma","contrast"}
    membership_required={"candidate_pathway","mapped_to_humangem","human_gem_pathway"}
    if not scatter_required.issubset(table.columns):raise RuntimeError(f"{scatter_file} schema changed")
    if not enrichment_required.issubset(enrichment.columns):raise RuntimeError(f"{enrichment_file} schema changed")
    if set(membership.columns)!=membership_required:raise RuntimeError(f"{membership_file} schema changed")
    if len(table)!=expected_rows or table["pathway"].nunique()!=expected_rows:raise RuntimeError(f"{scatter_file} requires {expected_rows} unique pathways")
    if len(enrichment)!=expected_rows or enrichment["candidate_pathway"].nunique()!=expected_rows:raise RuntimeError(f"{enrichment_file} requires {expected_rows} unique pathways")
    if len(membership)!=expected_rows or membership["candidate_pathway"].nunique()!=expected_rows:raise RuntimeError(f"{membership_file} requires {expected_rows} unique source pathways")
    scatter_paths=set(table["pathway"].astype(str));enrichment_paths=set(enrichment["candidate_pathway"].astype(str));upset_paths=set(membership["candidate_pathway"].astype(str))
    if scatter_paths!=enrichment_paths or scatter_paths!=upset_paths:
        raise RuntimeError(f"{contrast} Panel B and UpSet pathway sets differ: scatter={len(scatter_paths)}, enrichment={len(enrichment_paths)}, upset={len(upset_paths)}")
    if set(enrichment["contrast"].astype(str))!={contrast}:raise RuntimeError(f"{enrichment_file} contrast changed")
    if set(table.loc[table["label_top5"],"rank"].astype(int))!=set(range(1,6)):raise RuntimeError(f"{scatter_file} top-five labels changed")
    discoveries=int(enrichment["pathway_BH_FDR_gamma"].lt(.05).sum())
    if discoveries!=expected_discoveries:raise RuntimeError(f"{contrast} pathway BH-FDR discoveries changed: {discoveries}")
    return table,discoveries

def _draw_treatment_pathway_scatter(container, data: Path, label: str = "B") -> None:
    table,discoveries=_validated_integrated_pathway_scatter(data,"surgery_pathway_scatter.csv","surgery_pathway_enrichment.csv","treatment_four_set_membership.csv","surgery_vs_biopsy",22,3)
    radius=table["radius"].to_numpy(dtype=float); rmin,rmax=radius.min(),radius.max()
    sizes=24+96*(radius-rmin)/(rmax-rmin)
    cmap=LinearSegmentedColormap.from_list("pathway_yellow_red",["#FFF200","#FFA000","#FF1300"])
    ax=container.subplots(1,1)
    nominal_cutoff_y = -np.log10(NOMINAL_P_CUTOFF)
    ax.axhline(nominal_cutoff_y, color=COLORS["gray"], linestyle="--", linewidth=LINE_WIDTH, alpha=.8, zorder=1)
    ax.text(.02, nominal_cutoff_y, r"Nominal $p = 0.05$", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=TICK_LABEL_SIZE_PT, color=COLORS["gray"])
    scatter=ax.scatter(table["enrichment_factor"],table["neg_log10_p"],s=sizes,c=table["neg_log10_p"],cmap=cmap,edgecolor="white",linewidth=.45,alpha=.92,zorder=3)
    offsets={1:(-12,-2),2:(-8,-5),3:(-8,4),4:(-10,7),5:(-12,-10)}
    for row in table.loc[table["label_top5"]].itertuples(index=False):
        dx,dy=offsets[int(row.rank)]
        ax.annotate(row.pathway,(row.enrichment_factor,row.neg_log10_p),xytext=(dx,dy),textcoords="offset points",ha="right" if dx<0 else "left",va="bottom",fontsize=5.5,color=COLORS["dark"])
    ax.set_xlabel("Enrichment factor"); ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_xlim(0,max(1.86,table["enrichment_factor"].max()*1.06)); ax.set_ylim(-.03,table["neg_log10_p"].max()*1.10)
    _clean_axis(ax); ax.grid(False,which="both")
    colorbar=container.colorbar(scatter,ax=ax,fraction=.045,pad=.035); colorbar.set_label(r"$-\log_{10}(p)$",fontsize=TICK_LABEL_SIZE_PT); colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE_PT); colorbar.solids.set_rasterized(False)
    legend_values=[.5,.9,1.3]
    handles=[ax.scatter([],[],s=24+96*(value-rmin)/(rmax-rmin),facecolor="#F4A300",edgecolor="white",linewidth=.45) for value in legend_values]
    ax.legend(handles,[f"{value:.1f}" for value in legend_values],title=r"$\sqrt{\mathrm{enrichment}}$",frameon=False,fontsize=5.5,title_fontsize=5.5,loc="lower right",borderaxespad=.2,labelspacing=.7)
    _panel_title(container,label,"Treatment-associated pathway enrichment",f"{len(table)} candidates shared with A; top five labeled.\n{discoveries} pass pathway BH-FDR < 0.05.")
    container.subplots_adjust(left=.13,right=.84,top=.78,bottom=.17)

def _draw_robust_trajectories(container,data:Path,label:str="D")->None:
    sel=data["robust_tier3_trajectory_selection.csv"].copy().sort_values("display_order")
    vals=data["robust_tier3_trajectory_values.csv.gz"].copy()
    expected=["heparan_sulfate_degradation_product_18","VI3NeuAc_nLc6Cer","2_phosphoglycolate"]
    if sel["metabolite"].tolist()!=expected or not sel["tier"].eq(3).all():raise RuntimeError("Panel D mechanism-aligned Tier-3 selection changed")
    if set(vals["PatientID"])!={"CF/30117"} or set(vals["Timepoint"])!={"Bx","Surgery","Recurrence"}:raise RuntimeError("Panel D requires P3 Bx-Surgery-Recurrence cells")
    axes=np.atleast_1d(container.subplots(1,3)).ravel()
    order=["Bx","Surgery","Recurrence"];palette={"Bx":"#377EB8","Surgery":"#E41A1C","Recurrence":"#984EA3"}
    def fmt_q(value):
        return f"{value:.3f}" if value>=.001 else f"{value:.1e}"
    for idx,(ax,c) in enumerate(zip(axes,sel.itertuples(index=False))):
        frame=vals.loc[vals["metabolite"].eq(c.metabolite)]
        sns.violinplot(data=frame,x="Timepoint",y="activity",order=order,hue="Timepoint",hue_order=order,palette=palette,legend=False,inner=None,cut=0,linewidth=.6,ax=ax)
        sns.boxplot(data=frame,x="Timepoint",y="activity",order=order,width=.16,showfliers=False,boxprops={"facecolor":"white","alpha":.82},medianprops={"color":"black"},ax=ax)
        counts=frame.groupby("Timepoint").size().reindex(order)
        ax.set_xticks(range(3),[f"{tp}\nn={int(counts[tp]):,}" for tp in order])
        ax.set_title(f"{c.display_name}\n{c.celltype} • global rank {int(c.global_rank)}",loc="left",fontsize=7,fontweight="bold",pad=2)
        ax.text(.98,.98,f"S–Bx δ={c.p4_tx_rank_biserial_effect:+.3f}, q={fmt_q(c.p4_tx_global_contrast_BH_FDR)}\nR–S δ={c.p4_rebound_rank_biserial_effect:+.3f}, q={fmt_q(c.p4_rebound_global_contrast_BH_FDR)}",transform=ax.transAxes,ha="right",va="top",fontsize=5.2,color=COLORS["dark"],bbox={"facecolor":"white","edgecolor":"none","alpha":.78,"pad":1.2})
        ax.set_xlabel("");ax.set_ylabel("Inferred scKNIFE\nmetabolite-node activity" if idx==0 else "");ax.tick_params(axis="x",labelsize=6)
        _clean_axis(ax)
    _panel_title(container,label,"Mechanism-aligned longitudinal restoration","P3: Bx → Surgery → Recurrence; strict Tier-3 inferred metabolite nodes.")
    container.text(.99,.01,"Cell-level exploratory distributions from one recurrence patient; the six-patient cohort independently supplies Bx–Surgery discovery evidence",ha="right",va="bottom",fontsize=5.5,color=COLORS["gray"])
    container.subplots_adjust(left=.08,right=.99,top=.72,bottom=.23,wspace=.28)

MAIN_VALIDATION_METABOLITES = ("threonine",)

MAIN_CROSSMODAL_UMAP_SPECS = {
    "2_aminoethylphosphonate": "2-Aminoethylphosphonate",
    "Glutaryl_Carnitine": "Glutarylcarnitine",
    "malonate": "Malonate",
}

def _draw_status_pathway_scatter(container,data:Path,label:str="SB")->None:
    table,discoveries=_validated_integrated_pathway_scatter(data,"status_pathway_scatter.csv","status_pathway_enrichment.csv","status_four_set_membership.csv","metastatic_vs_nonmetastatic",34,0)
    radius=table["radius"].to_numpy(dtype=float); rmin,rmax=radius.min(),radius.max(); sizes=24+96*(radius-rmin)/(rmax-rmin)
    cmap=LinearSegmentedColormap.from_list("status_pathway_yellow_red",["#FFF200","#FFA000","#FF1300"])
    ax=container.subplots(1,1)
    nominal_cutoff_y = -np.log10(NOMINAL_P_CUTOFF)
    ax.axhline(nominal_cutoff_y, color=COLORS["gray"], linestyle="--", linewidth=LINE_WIDTH, alpha=.8, zorder=1)
    ax.text(.02, nominal_cutoff_y, r"Nominal $p = 0.05$", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=TICK_LABEL_SIZE_PT, color=COLORS["gray"])
    scatter=ax.scatter(table["enrichment_factor"],table["neg_log10_p"],s=sizes,c=table["neg_log10_p"],cmap=cmap,edgecolor="white",linewidth=.45,alpha=.92,zorder=3)
    offsets={1:(-10,-2),2:(-8,-7),3:(-8,7),4:(8,-10),5:(-10,-8)}
    for row in table.loc[table["label_top5"]].itertuples(index=False):
        dx,dy=offsets[int(row.rank)]; ax.annotate(row.pathway,(row.enrichment_factor,row.neg_log10_p),xytext=(dx,dy),textcoords="offset points",ha="right" if dx<0 else "left",va="bottom",fontsize=5.5,color=COLORS["dark"])
    ax.set_xlabel("Enrichment factor"); ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_xlim(0,max(2.45,table["enrichment_factor"].max()*1.05)); ax.set_ylim(-.03,table["neg_log10_p"].max()*1.10)
    _clean_axis(ax); ax.grid(False,which="both")
    colorbar=container.colorbar(scatter,ax=ax,fraction=.045,pad=.035); colorbar.set_label(r"$-\log_{10}(p)$",fontsize=TICK_LABEL_SIZE_PT); colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE_PT); colorbar.solids.set_rasterized(False)
    legend_values=[.75,1.00,1.25,1.50]
    handles=[ax.scatter([],[],s=24+96*(value-rmin)/(rmax-rmin),facecolor="#F4A300",edgecolor="white",linewidth=.45) for value in legend_values]
    ax.legend(handles,[f"{value:.2f}" for value in legend_values],title=r"$\sqrt{\mathrm{enrichment}}$",frameon=False,fontsize=5.5,title_fontsize=5.5,loc="lower right",borderaxespad=.2,labelspacing=.55)
    _panel_title(container,label,"Metastatic-status pathway enrichment",f"{len(table)} candidates shared with SA; top five labeled.\n{discoveries} pass pathway BH-FDR < 0.05.")
    container.subplots_adjust(left=.13,right=.84,top=.78,bottom=.17)

CORRELATION_CELLTYPE_ORDER = ["Epithelial", "Epithelial cycling", "CAFs", "Macrophages", "T cells", "Endothelial"]

CORRELATION_CELLTYPE_COLORS = {"Epithelial":"#1f77b4","Epithelial cycling":"#d62728","CAFs":"#8c564b","Macrophages":"#e377c2","T cells":"#7f7f7f","Endothelial":"#aec7e8"}

FOUR_SET_ORDER = ["MEIA-Task", "MEIA-Pathway", "GPR-GPA", "LC-MS"]

FOUR_SET_COLORS = {"MEIA-Task":"#009E73","MEIA-Pathway":"#CC79A7","GPR-GPA":"#D55E00","LC-MS":"#666666"}

CORRELATION_POINT_SIZE = 9

CORRELATION_POINT_ALPHA = .62

def _draw_four_set_upset(container, data, label, filename, title):
    mapping=data[filename].copy()
    required={"candidate_pathway","mapped_to_humangem","human_gem_pathway"}
    if set(mapping.columns)!=required: raise RuntimeError(f"Unexpected LC-MS mapping schema for {label}")
    mapping["mapped_to_humangem"]=mapping["mapped_to_humangem"].astype(bool)
    expected={
        "A":{"source_total":22,"mapped_sources":18,"mapped_unique":14,"unmapped":4,"harmonized_total":18},
        "SA":{"source_total":34,"mapped_sources":31,"mapped_unique":27,"unmapped":3,"harmonized_total":30},
    }[label]
    mapped_targets={
        target.strip()
        for targets in mapping.loc[mapping["mapped_to_humangem"],"human_gem_pathway"].dropna().astype(str)
        for target in targets.split("|") if target.strip()
    }
    observed={
        "source_total":len(mapping),
        "mapped_sources":int(mapping["mapped_to_humangem"].sum()),
        "mapped_unique":len(mapped_targets),
        "unmapped":int((~mapping["mapped_to_humangem"]).sum()),
        "harmonized_total":len(mapped_targets)+int((~mapping["mapped_to_humangem"]).sum()),
    }
    if observed!=expected: raise RuntimeError(f"{label} LC-MS mapping counts {observed} != {expected}")
    meia_gpr_only=138-expected["mapped_unique"]
    set_sizes={"MEIA-Task":218,"MEIA-Pathway":138,"GPR-GPA":138,"LC-MS":expected["harmonized_total"]}
    relationships=[
        ("MEIA-Task\nfeatures",218,(True,False,False,False)),
        ("MEIA-Pathway ↔\nGPR-GPA",meia_gpr_only,(False,True,True,False)),
        ("LC-MS mapped",expected["mapped_unique"],(False,True,True,True)),
        ("LC-MS unmapped",expected["unmapped"],(False,False,False,True)),
    ]
    grid=container.add_gridspec(2,2,height_ratios=[.62,.38],width_ratios=[.36,.64],left=.23,right=.98,top=.79,bottom=.24,hspace=.06,wspace=.10)
    ax_top=container.add_subplot(grid[0,1]);ax_sets=container.add_subplot(grid[1,0]);ax_matrix=container.add_subplot(grid[1,1],sharex=ax_top)
    xpos=np.arange(len(relationships));counts=[x[1] for x in relationships];bars=ax_top.bar(xpos,counts,color=COLORS["dark"],width=UPSET_BAR_WIDTH)
    for bar,count in zip(bars,counts):ax_top.text(bar.get_x()+bar.get_width()/2,count+max(counts)*.025,str(count),ha="center",va="bottom",fontsize=TICK_LABEL_SIZE_PT)
    ax_top.set_ylabel("Count");ax_top.set_xticks([]);ax_top.set_ylim(0,max(counts)*1.18);_clean_axis(ax_top)
    ypos=np.arange(len(FOUR_SET_ORDER));ax_sets.barh(ypos,[set_sizes[x] for x in FOUR_SET_ORDER],color=[FOUR_SET_COLORS[x] for x in FOUR_SET_ORDER],height=UPSET_SET_BAR_HEIGHT)
    ax_sets.set_yticks(ypos,FOUR_SET_ORDER);ax_sets.invert_xaxis();ax_sets.invert_yaxis();ax_sets.set_xlabel("Source feature/pathway count");_clean_axis(ax_sets)
    for tick,key in zip(ax_sets.get_yticklabels(),FOUR_SET_ORDER):tick.set_color(FOUR_SET_COLORS[key])
    for x,(_,_,membership) in enumerate(relationships):
        active=[]
        for y,key in enumerate(FOUR_SET_ORDER):
            on=membership[y];ax_matrix.scatter(x,y,s=UPSET_DOT_SIZE if on else UPSET_EMPTY_DOT_SIZE,color=FOUR_SET_COLORS[key] if on else COLORS["light"],edgecolors="none",zorder=3)
            if on:active.append(y)
        if len(active)>1:ax_matrix.plot([x,x],[min(active),max(active)],color=COLORS["dark"],lw=UPSET_CONNECTOR_WIDTH,zorder=2)
    ax_matrix.set_ylim(len(FOUR_SET_ORDER)-.45,-.45);ax_matrix.set_xticks(xpos,[x[0] for x in relationships],rotation=24,ha="right");ax_matrix.set_yticks(ypos,[""]*len(ypos));ax_matrix.tick_params(axis="y",length=0);ax_matrix.spines[:].set_visible(False);ax_matrix.grid(False)
    _panel_title(container,label,title,f"LC-MS: {expected['source_total']} pathways → {expected['harmonized_total']} harmonized\n({expected['mapped_unique']} mapped + {expected['unmapped']} unmapped); MEIA: 218 tasks + 138 pathways.")

_compiled_dir=Path("figs/panel/Figure_6_blank_corrected").resolve()

_individual_dir=Path("figs/individual/Figure_6_blank_corrected").resolve()

_review_export_labels={"A","B","SA","SB"}

_blank_corrected_export_paths=[]

ENTITY_SCATTER_POINT_SIZE=28

ENTITY_SCATTER_ALPHA=.72

ENTITY_IDENTITY_LINE_WIDTH=.9

MEIA_ENTITY_LIMITS=(.30,1.00)

GPA_ENTITY_LIMITS=(.15,1.00)

def _validated_entity_concordance(data,filename,kind):
    table=data[filename].copy()
    required={"feature","entity","resolution","comparison","n_cells","pearson","spearman"}
    if set(table.columns)!=required or not np.isfinite(table[["pearson","spearman"]].to_numpy(float)).all():raise RuntimeError(f"{filename} schema or values changed")
    if table["feature"].duplicated().any() or not table["n_cells"].eq(69390).all():raise RuntimeError(f"{filename} entity/cell counts changed")
    if kind=="meia":
        if len(table)!=356 or table["resolution"].value_counts().to_dict()!={"Task":218,"Pathway":138} or set(table["comparison"])!={"MEIA vs direct scKNIFE"}:raise RuntimeError("Main C requires 218 tasks + 138 pathways")
        expected={"pearson":.9290342115273988,"spearman":.9270746908214075}
    else:
        if len(table)!=138 or set(table["resolution"])!={"Pathway"} or set(table["comparison"])!={"GPR-GPA-specific vs direct scKNIFE"}:raise RuntimeError("Supplementary SJ requires 138 Human-GEM pathways")
        expected={"pearson":.8433539746026706,"spearman":.8407887377370387}
    if any(not np.isclose(float(table[metric].median()),value,atol=1e-5) for metric,value in expected.items()):raise RuntimeError(f"{filename} reference medians changed")
    return table

def _draw_entity_concordance(container,data,label,kind):
    if kind=="meia":
        table=_validated_entity_concordance(data,"meia_vs_scknife_entity_correlations.csv","meia");limits=MEIA_ENTITY_LIMITS
        layers=(("Task","#0072B2","scCellFie tasks (n=218)"),("Pathway","#D55E00","Human-GEM pathways (n=138)"))
        title="MEIA preserves matched scKNIFE activity";subtitle="356 entities (218 tasks + 138 pathways); each entity correlated across 69,390 cells"
    else:
        table=_validated_entity_concordance(data,"gpr_gpa_vs_scknife_entity_correlations.csv","gpa");limits=GPA_ENTITY_LIMITS
        layers=(("Pathway","#009E73","Human-GEM pathways (n=138)"),)
        title="GPR-GPA–scKNIFE concordance";subtitle="138 Human-GEM pathways; each correlated across 69,390 cells"
    ax=container.subplots(1,1)
    for resolution,color,legend_label in layers:
        frame=table.loc[table["resolution"].eq(resolution)]
        ax.scatter(frame["pearson"],frame["spearman"],s=ENTITY_SCATTER_POINT_SIZE,alpha=ENTITY_SCATTER_ALPHA,color=color,edgecolors="none",label=legend_label)
    ax.plot(limits,limits,color=COLORS["gray"],linestyle="--",linewidth=ENTITY_IDENTITY_LINE_WIDTH,zorder=1)
    ax.set_xlim(limits);ax.set_ylim(limits);ax.set_aspect("equal",adjustable="box")
    ax.set_xlabel("Entity-wise Pearson correlation" if kind=="meia" else "Pathway-wise Pearson correlation")
    ax.set_ylabel("Entity-wise Spearman correlation" if kind=="meia" else "Pathway-wise Spearman correlation")
    _clean_axis(ax);ax.legend(frameon=False,fontsize=LEGEND_SIZE_PT,loc="lower right")
    _panel_title(container,label,title,subtitle);container.subplots_adjust(left=.15,right=.94,top=.80,bottom=.17)

def _validated_robust_context(data):
    assignments=data["robust_metabolite_pathway_assignments.csv"].copy()
    contexts=data["robust_metabolite_pathway_context_counts.csv"].copy().sort_values("display_order")
    expected={"Fatty-acid oxidation and carnitine handling":38,"No specific Human-GEM pathway":13,"Glycoconjugate and sphingolipid remodeling":10,"Amino-acid metabolism":3,"Glycerophospholipid metabolism":2,"Nucleotide and pyrimidine metabolism":1,"Glycolysis / gluconeogenesis":1}
    if len(assignments)!=68 or assignments["metabolite"].nunique()!=68:raise RuntimeError("Panel E requires 68 unique Tier-3 metabolites")
    if contexts.set_index("pathway_context")["metabolite_count"].to_dict()!=expected or contexts["metabolite_count"].sum()!=68:raise RuntimeError("Panel E pathway-context counts changed")
    return contexts

def _draw_pathway_context_pie(container,data,label="E"):
    contexts=_validated_robust_context(data)
    ax=container.subplots(1,1)
    short={"Fatty-acid oxidation and carnitine handling":"Fatty-acid oxidation/carnitine","No specific Human-GEM pathway":"No specific Human-GEM pathway","Glycoconjugate and sphingolipid remodeling":"Glycoconjugate/sphingolipid","Amino-acid metabolism":"Amino-acid metabolism","Glycerophospholipid metabolism":"Glycerophospholipid metabolism","Nucleotide and pyrimidine metabolism":"Nucleotide/pyrimidine metabolism","Glycolysis / gluconeogenesis":"Glycolysis/gluconeogenesis"}
    colors=["#0072B2","#999999","#009E73","#E69F00","#CC79A7","#56B4E9","#D55E00"]
    def pct(value):return f"{value:.1f}%" if value>=8 else ""
    wedges,_,_=ax.pie(contexts["metabolite_count"],colors=colors,startangle=90,counterclock=False,radius=.78,center=(-.62,0),autopct=pct,pctdistance=.73,textprops={"fontsize":TICK_LABEL_SIZE_PT,"color":"white","fontweight":"bold"},wedgeprops={"width":.46,"edgecolor":"white","linewidth":.8})
    ax.text(-.62,0,"n=68",ha="center",va="center",fontsize=8,fontweight="bold",color=COLORS["dark"])
    labels=[f"{short[row.pathway_context]} — {int(row.metabolite_count)} ({row.percentage:.1f}%)" for row in contexts.itertuples(index=False)]
    ax.legend(wedges,labels,loc="center left",bbox_to_anchor=(.50,.48),frameon=False,fontsize=5.2,handlelength=.85,handletextpad=.4,labelspacing=.55)
    ax.set_xlim(-1.48,2.58);ax.set_ylim(-1.05,1.05);ax.set_aspect("equal");ax.axis("off")
    _panel_title(container,label,"Tier-3 pathway context","68 unique inferred metabolite nodes; each counted once.")
    container.text(.99,.01,"Human-GEM metabolite–reaction–pathway context; generic exchange, transport, isolated and miscellaneous groups excluded.",ha="right",va="bottom",fontsize=5.2,color=COLORS["gray"])
    container.subplots_adjust(left=.02,right=.99,top=.72,bottom=.12)

def _draw_filtering_cascade(container,data,label="Supplementary candidate"):
    funnel=data["robust_metabolite_filtering_funnel.csv"].copy().sort_values("display_order")
    if funnel["count"].tolist()!=[20650,4607,599,70,68,3]:raise RuntimeError("Filtering-cascade counts changed")
    ax=container.subplots(1,1);y=np.arange(len(funnel));counts=funnel["count"].to_numpy(float)
    ax.plot(counts,y,color="#0072B2",lw=1.3,zorder=1);ax.scatter(counts,y,s=34,color="#0072B2",edgecolor="white",linewidth=.6,zorder=2)
    for ypos,count in zip(y,funnel["count"]):ax.text(.93,ypos,f"{int(count):,}",transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=TICK_LABEL_SIZE_PT,fontweight="bold",color=COLORS["dark"])
    ax.set_xscale("log");ax.set_xlim(2,50000);ax.set_yticks(y,["Tested pairs","Replicated treatment","Recurrence rebound","Baseline-like recurrence","Unique Tier-3 metabolites","Shown in main D"]);ax.invert_yaxis();ax.set_xlabel("Count (log scale)");_clean_axis(ax)
    _panel_title(container,label,"Robust inferred-metabolite filtering cascade","Metabolite–cell-type pairs are distinguished from unique inferred metabolite nodes.")
    container.subplots_adjust(left=.30,right=.96,top=.73,bottom=.20)

def _draw_homocysteine_screen(container,data,label="F"):
    cohort=data["sulfur_cohort_patient_values.csv"].copy();order=['CAFs','Endothelial','Epithelial','Macrophages','T cells']
    if len(cohort)!=315 or cohort.score_label.nunique()!=9 or set(cohort.cell_type_analysis)!=set(order):raise RuntimeError("Panel F cohort table changed")
    stats=_sulfur_treatment_statistics(data);screen=stats.query("cohort=='All 7' and score_label=='MEIA task | Homocysteine degradation'").set_index('cell_type').loc[order].reset_index()
    expected={'CAFs':(5,.296875,.66796875),'Endothelial':(4,.9375,.9811046511627907),'Epithelial':(6,.109375,.328125),'Macrophages':(7,.015625,.28125),'T cells':(4,.8125,.9811046511627907)}
    for row in screen.itertuples(index=False):
        n,p,q=expected[row.cell_type]
        if int(row.n_decrease)!=n or not np.isclose(row.p,p) or not np.isclose(row.q45,q):raise RuntimeError(f"Panel F statistics changed for {row.cell_type}")
    ax=container.subplots(1,1);y=np.arange(5);ax.axhspan(2.65,3.35,color='#F0F0F0',zorder=0);ax.scatter(screen.n_decrease,y,s=55,color='#0072B2',edgecolor='white',linewidth=.6,zorder=3)
    for x,ypos,delta,p,q in zip(screen.n_decrease,y,screen.median_delta,screen.p,screen.q45):ax.text(x+.1,ypos,f'{int(x)}/7  Δ̃={delta:.2e}; p={p:.3f}; q45={q:.3f}',va='center',fontsize=6.2,color='#0072B2')
    ax.axvline(3.5,color='#777777',lw=.8,ls='--');ax.set(xlim=(2.5,8.45),xticks=range(3,8),yticks=range(5),yticklabels=order,xlabel='Patients with lower task score at Surgery');_clean_axis(ax)
    _panel_title(container,label,"Homocysteine degradation across cell types","Exact paired Wilcoxon tests across seven patient medians; BH across the original 45 tests.")
    container.text(.99,.01,"Macrophages: 7/7 decrease, nominal p=0.0156, q45=0.281; directionally consistent but not FDR-significant.",ha="right",va="bottom",fontsize=5.2,color=COLORS["gray"])
    container.subplots_adjust(left=.22,right=.97,top=.72,bottom=.22)

def _draw_macrophage_tasks(container,data,label="G"):
    cohort=data["sulfur_cohort_patient_values.csv"].copy();stats=_sulfur_treatment_statistics(data);primary=stats.query("cohort=='All 7'").set_index(['score_label','cell_type']);sensitivity=stats.query("cohort=='Without P3'").set_index(['score_label','cell_type'])
    tasks=['MEIA task | Homocysteine degradation','MEIA task | Glutathione synthesis'];axes=np.atleast_1d(container.subplots(1,2)).ravel()
    for idx,(ax,task) in enumerate(zip(axes,tasks)):
        frame=cohort.query("score_label==@task and cell_type_analysis=='Macrophages'").sort_values('PatientID');boxes=ax.boxplot([frame.Bx,frame.Surgery],positions=[0,1],widths=.42,patch_artist=True,showfliers=False,medianprops={'color':'#222222'})
        for patch,color in zip(boxes['boxes'],['#377EB8','#E41A1C']):patch.set_facecolor(color);patch.set_alpha(.2)
        jitter=np.linspace(-.06,.06,7)
        for j,(_,row) in enumerate(frame.iterrows()):
            color='#E41A1C' if row.metastatic_status=='Metastatic' else '#4DAF4A';marker='*' if bool(row.is_longitudinal_case) else 'o';ax.plot([jitter[j],1+jitter[j]],[row.Bx,row.Surgery],color=color,lw=.9);ax.scatter([jitter[j],1+jitter[j]],[row.Bx,row.Surgery],s=68 if marker=='*' else 38,marker=marker,c=['#377EB8','#E41A1C'],edgecolor=color,zorder=3)
        a=primary.loc[(task,'Macrophages')];b=sensitivity.loc[(task,'Macrophages')];ax.set_title(f"{task.replace('MEIA task | ','')}\n{int(a.n_decrease)}/7↓: p={a.p:.4f}, q45={a.q45:.3f}\nWithout P3 {int(b.n_decrease)}/6↓: p={b.p:.4f}",fontsize=6.3,fontweight='bold',pad=2);ax.set_xticks([0,1],['Bx\nn=7','Surgery\nn=7']);ax.set_ylabel('Patient-median compositional MEIA score' if idx==0 else '');_clean_axis(ax)
    _panel_title(container,label,"Macrophage metabolic tasks","Paired Bx–Surgery; seven patients plus leave-P3-out sensitivity.")
    container.text(.99,.01,"Exact paired Wilcoxon; BH across 45 treatment tests. Star=P3; line color=metastatic status. Neither task passes q<0.05.",ha="right",va="bottom",fontsize=5.2,color=COLORS["gray"])
    container.subplots_adjust(left=.10,right=.99,top=.70,bottom=.20,wspace=.27)

def _draw_selenium_lcms(container,data,label="H"):
    values=data["sulfur_lcms_feature_values.csv"].copy();evidence=data["sulfur_lcms_feature_evidence.csv"].iloc[0];paired=values.pivot(index='PatientID_clean',columns='Timepoint',values='log2_intensity').reset_index()
    if len(paired)!=7 or not ((paired.Surgery-paired.Biopsy)<0).all() or bool(evidence.has_putative_native_match):raise RuntimeError("Panel H LC-MS evidence changed")
    ax=container.subplots(1,1);arrays=[paired.Biopsy.to_numpy(),paired.Surgery.to_numpy()];boxes=ax.boxplot(arrays,positions=[0,1],widths=.34,patch_artist=True,showfliers=False,medianprops={'color':'#222222','linewidth':1.3},boxprops={'edgecolor':'#555555','linewidth':.8},whiskerprops={'color':'#555555','linewidth':.8},capprops={'color':'#555555','linewidth':.8})
    for patch,color in zip(boxes['boxes'],['#377EB8','#E41A1C']):patch.set_facecolor(color);patch.set_alpha(.22)
    jitter=np.linspace(-.075,.075,7)
    for i,row in paired.iterrows():ax.plot([jitter[i],1+jitter[i]],[row.Biopsy,row.Surgery],color='#666666',lw=.9);ax.scatter([jitter[i],1+jitter[i]],[row.Biopsy,row.Surgery],s=34,c=['#377EB8','#E41A1C'],edgecolor='white',zorder=3)
    ax.set_xticks([0,1],['Biopsy','Surgery']);ax.set_ylabel('Specimen-level log2 intensity');_clean_axis(ax)
    ax.text(.03,.05,f"7/7 Surgery < Biopsy\nlog2FC={evidence.log2_fold_change:.3f}; p={evidence.nominal_p_value:.5f}; BH q={evidence.joint_BH_FDR:.4f}",transform=ax.transAxes,fontsize=6,bbox={'facecolor':'white','edgecolor':'#BBBBBB'})
    _panel_title(container,label,"Paired LC–MS m/z 215.0260","Accurate-mass candidate: Se-adenosyl-L-selenohomocysteine ([M−2H]²⁻).")
    container.text(.99,.01,f"No native MS/MS or standard; median technical CV={evidence.median_technical_replicate_CV_percent:.1f}%; supporting context only.",ha="right",va="bottom",fontsize=5.2,color=COLORS["gray"])
    container.subplots_adjust(left=.18,right=.96,top=.70,bottom=.20)

def _draw_ng8_graph(container,data,label="I"):
    import networkx as nx
    from adjustText import adjust_text
    import matplotlib.patheffects as patheffects
    nodes=data["sulfur_ng8_nodes.csv"].copy();edges=data["sulfur_ng8_edges.csv"].copy();anchors=data["sulfur_ng8_anchors.csv"].copy();graph=nx.from_pandas_edgelist(edges,'source','target');nx.set_node_attributes(graph,nodes.set_index('node').to_dict('index'))
    if (len(graph),len(edges),nx.number_connected_components(graph))!=(35,42,1) or len(anchors)==0:raise RuntimeError("Panel I NG8 topology changed")
    palette={'Gene':'#0072B2','Metabolite':'#009E73','Reaction':'#D55E00','Task / pathway':'#CC79A7','GO term / gene set':'#7F3C8D','CellChat entity':'#56B4E9','Other':'#777777'};position=nx.spring_layout(graph,seed=42,k=.95/np.sqrt(35),iterations=1200,weight=None);ax=container.subplots(1,1)
    nx.draw_networkx_edges(graph,position,ax=ax,edge_color='#8E8E8E',width=.8,alpha=.62)
    for kind,color in palette.items():
        selected=[node for node in graph if graph.nodes[node]['display_type']==kind]
        nx.draw_networkx_nodes(graph,position,nodelist=selected,ax=ax,node_color=color,node_size=[135 if graph.nodes[node]['is_connector'] else 205 for node in selected],edgecolors='#555555',linewidths=.7,alpha=.7)
    ax.scatter(*zip(position['selenocystathionine']),s=390,facecolor='none',edgecolor='#D7301F',linewidth=2,zorder=4);texts=[]
    for node in sorted(graph):
        x,y=position[node];text=ax.text(x,y,str(graph.nodes[node]['name']).replace('_',' '),fontsize=5.0,fontweight='bold' if graph.nodes[node]['is_ng8'] else 'normal',ha='center',va='center');text.set_path_effects([patheffects.withStroke(linewidth=2,foreground='white')]);texts.append(text)
    adjust_text(texts,ax=ax,prevent_crossings=True,iter_lim=1800,arrowprops={'arrowstyle':'-','color':'#999999','lw':.5});ax.axis('off')
    _panel_title(container,label,"NG8 entity subgraph of the scKNIFE knowledge backbone","20 NG8 nodes plus 15 minimum connectors; topology is biochemical context, not node-level statistical evidence.")
    handles=[mpl.lines.Line2D([0],[0],marker='o',linestyle='',markerfacecolor=color,markeredgecolor='#555555',markersize=5,label=kind) for kind,color in palette.items() if kind!='Other'];ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,fontsize=5.5)
    container.subplots_adjust(left=.03,right=.98,top=.75,bottom=.04)
    container.figure.canvas.draw();renderer=container.figure.canvas.get_renderer();boxes=[text.get_window_extent(renderer) for text in texts];overlaps=sum(boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i+1,len(boxes)))
    if overlaps:raise RuntimeError(f"Panel {label} has {overlaps} label overlaps")

def _draw_review_s1(container,data,label="S1"):
    patient=data["review_whole_profile_patient_summary.csv"].copy();overall=data["review_whole_profile_concordance_summary.csv"].copy()
    if len(patient)!=14 or patient.groupby("metric").PatientID.nunique().to_dict()!={"pearson":7,"spearman":7}:raise RuntimeError("S1 patient summaries changed")
    axes=np.atleast_1d(container.subplots(1,2)).ravel();order=[f"P{i}" for i in range(1,8)];colors={"spearman":"#E69F00","pearson":"#0072B2"}
    for ax,metric,title,symbol in zip(axes,["spearman","pearson"],["Patient Spearman robustness","Patient Pearson robustness"],["ρ","r"]):
        f=patient.loc[patient.metric.eq(metric)].set_index("patient_display").reindex(order);y=np.arange(7)
        low=f["median"].to_numpy()-f["q25"].to_numpy();high=f["q75"].to_numpy()-f["median"].to_numpy()
        ax.errorbar(f["median"],y,xerr=np.vstack([low,high]),fmt="o",color=colors[metric],ecolor=colors[metric],elinewidth=1.2,capsize=2.5,markersize=5)
        med=float(overall.loc[overall.scope.eq("overall")&overall.metric.eq(metric),"median"].iloc[0]);ax.axvline(med,color=COLORS["gray"],ls="--",lw=.9)
        ax.set_yticks(y,order);ax.invert_yaxis();ax.set_xlim(.80,1.00);ax.set_xlabel("Patient median cell-wise correlation");ax.set_title(title,loc="left",fontsize=8,fontweight="bold")
        ax.text(.02,.03,f"Patient medians: {f['median'].min():.3f}–{f['median'].max():.3f}\nOverall median {symbol}={med:.3f}",transform=ax.transAxes,ha="left",va="bottom",fontsize=5.8)
        _clean_axis(ax)
    axes[0].set_ylabel("Patient");axes[1].set_ylabel("")
    _panel_title(container,label,"Patient robustness of MEIA–scKNIFE concordance","Dots are patient medians; intervals are within-patient cell IQRs across 356 entities.")
    container.text(.99,.01,"Descriptive technical concordance; no inferential p-values.",ha="right",va="bottom",fontsize=5.4,color=COLORS["gray"])
    container.subplots_adjust(left=.08,right=.98,top=.72,bottom=.22,wspace=.25)

def _draw_su3_tasks(container,data,label="SU3"):
    cells=data["sulfur_p3_all_score_cells.csv.gz"].copy()
    stats=data["sulfur_p3_task_statistics.csv"].copy()
    sites=["Bx","Surgery","Recurrence"];celltypes=["Macrophages","T cells","CAFs","Endothelial","Epithelial"]
    if len(stats)!=135 or cells.score_label.nunique()!=9 or not cells.groupby(["score_label","cell_type_analysis"]).Site.nunique().eq(3).all():raise RuntimeError("SU3 frozen audit changed")
    lookup=stats.set_index(["comparison","label","ct"]);tasks=["MEIA task | Homocysteine degradation","MEIA task | Glutathione synthesis"]
    colors={"Bx":"#377EB8","Surgery":"#E41A1C","Recurrence":"#984EA3"};axes=np.asarray(container.subplots(2,5)).reshape(2,5)
    def probability(value):return "<4.9e−324" if value==0 else f"{value:.2e}"
    for row_index,score in enumerate(tasks):
        for column_index,celltype in enumerate(celltypes):
            ax=axes[row_index,column_index];frame=cells.query("score_label==@score and cell_type_analysis==@celltype");values=[frame.loc[frame.Site.eq(site),"score"].to_numpy() for site in sites]
            violin=ax.violinplot(values,range(3),widths=.7,showextrema=False)
            for body,site in zip(violin["bodies"],sites):body.set_facecolor(colors[site]);body.set_edgecolor(colors[site]);body.set_alpha(.2)
            ax.boxplot(values,range(3),widths=.28,patch_artist=True,showfliers=False,medianprops={"color":"#222222","linewidth":1.2},boxprops={"facecolor":"white","edgecolor":"#666666","linewidth":.8},whiskerprops={"color":"#666666","linewidth":.7},capprops={"color":"#666666","linewidth":.7})
            means=[np.mean(value) for value in values];ax.plot(range(3),means,color="#333333",marker="D",ms=3,lw=.8);stat=lookup.loc[("Recurrence vs Surgery",score,celltype)]
            ax.set_title(f"{celltype}\nS→R {stat.change:+.1f}%; d={stat.d:+.2f}; q135={probability(stat.q135)}",fontsize=7.2,fontweight="bold");ax.set_xticks(range(3),[f"{site}\nn={len(value):,}" for site,value in zip(sites,values)],fontsize=6.5);ax.spines[["top","right"]].set_visible(False);ax.grid(False)
            if column_index==0:ax.set_ylabel("Cell-level activity score")
        axes[row_index,0].text(-.5,.5,score.replace("MEIA task | ",""),transform=axes[row_index,0].transAxes,rotation=90,ha="center",va="center",fontsize=8,fontweight="bold")
    container.suptitle(f"{label}  P3 homocysteine-degradation and glutathione-synthesis trajectories",x=.01,y=.985,ha="left",fontweight="bold",fontsize=PANEL_LABEL_SIZE_PT)
    container.text(.5,.018,"Violin width=density; white box=IQR; line=median; whiskers=1.5×IQR; diamond/line=mean; box outliers hidden.",ha="center",fontsize=6.5,color="#555555")
    container.text(.5,.004,"Welch tests; BH across original 135 tests. One patient; cells are pseudoreplicates.",ha="center",fontsize=6.5,color="#555555");container.subplots_adjust(left=.07,right=.995,top=.86,bottom=.14,hspace=.72,wspace=.38)


def _sulfur_treatment_statistics(data):
    stats=data["sulfur_cohort_treatment_statistics.csv"].copy()
    if len(stats)!=90: raise RuntimeError("Frozen sulfur treatment statistics changed")
    return stats

MAIN_INPUTS={
"A":["treatment_four_set_membership.csv"],"B":["surgery_pathway_scatter.csv","surgery_pathway_enrichment.csv","treatment_four_set_membership.csv"],"C":["meia_vs_scknife_entity_correlations.csv"],
"D":["robust_tier3_trajectory_selection.csv","robust_tier3_trajectory_values.csv.gz"],
"E":["robust_metabolite_pathway_assignments.csv","robust_metabolite_pathway_context_counts.csv"],
"F":["sulfur_cohort_patient_values.csv","sulfur_cohort_treatment_statistics.csv"],
"G":["sulfur_lcms_feature_values.csv","sulfur_lcms_feature_evidence.csv"],
"H":["sulfur_ng8_nodes.csv","sulfur_ng8_edges.csv","sulfur_ng8_anchors.csv"]}
SUPPLEMENT_INPUTS={"SA":["status_four_set_membership.csv"],"SB":["status_pathway_scatter.csv","status_pathway_enrichment.csv","status_four_set_membership.csv"],"SJ":["gpr_gpa_vs_scknife_entity_correlations.csv"],
"S1":["review_whole_profile_patient_summary.csv","review_whole_profile_concordance_summary.csv"],"SFUNNEL":["robust_metabolite_filtering_funnel.csv"],
"SDIRECTION":["sulfur_cohort_patient_values.csv","sulfur_cohort_treatment_statistics.csv"],"SU3":["sulfur_p3_all_score_cells.csv.gz","sulfur_p3_task_statistics.csv"]}

def load_tables(data_root, inputs):
    root=Path(data_root);return {name:pd.read_csv(root/name) for name in sorted(set(sum(inputs.values(),[])))}

def make_figure6_main_panel(label,data):
    fig=plt.figure(figsize=((14.5,8.8) if label=="H" else (_mm(112),_mm(88))),facecolor="white")
    {"A":lambda:_draw_four_set_upset(fig,data,"A","treatment_four_set_membership.csv","Treatment four-set metabolic overlap"),"B":lambda:_draw_treatment_pathway_scatter(fig,data,"B"),"C":lambda:_draw_entity_concordance(fig,data,"C","meia"),"D":lambda:_draw_robust_trajectories(fig,data,"D"),"E":lambda:_draw_pathway_context_pie(fig,data,"E"),"F":lambda:_draw_macrophage_tasks(fig,data,"F"),"G":lambda:_draw_selenium_lcms(fig,data,"G"),"H":lambda:_draw_ng8_graph(fig,data,"H")}[label]()
    return fig

def make_figure6_supplementary_panel(label,data):
    fig=plt.figure(figsize=(_mm(150),_mm(96)),facecolor="white")
    {"SA":lambda:_draw_four_set_upset(fig,data,"SA","status_four_set_membership.csv","Metastatic-status four-set metabolic overlap"),"SB":lambda:_draw_status_pathway_scatter(fig,data,"SB"),"SJ":lambda:_draw_entity_concordance(fig,data,"SJ","gpa"),"S1":lambda:_draw_review_s1(fig,data,"S1"),"SFUNNEL":lambda:_draw_filtering_cascade(fig,data,"SFUNNEL"),"SDIRECTION":lambda:_draw_homocysteine_screen(fig,data,"SDIRECTION"),"SU3":lambda:_draw_su3_tasks(fig,data,"SU3")}[label]()
    return fig

def make_figure6_compiled(data):
    fig=plt.figure(figsize=(_mm(297),_mm(380)),facecolor="white");rows=fig.subfigures(4,1,height_ratios=[.86,.98,.90,1.45],hspace=.035);top=rows[0].subfigures(1,3,wspace=.045)
    _draw_four_set_upset(top[0],data,"A","treatment_four_set_membership.csv","Treatment four-set metabolic overlap");_draw_treatment_pathway_scatter(top[1],data,"B");_draw_entity_concordance(top[2],data,"C","meia");_draw_robust_trajectories(rows[1],data,"D")
    evidence=rows[2].subfigures(1,3,width_ratios=[.95,1.62,.93],wspace=.045);_draw_pathway_context_pie(evidence[0],data,"E");_draw_macrophage_tasks(evidence[1],data,"F");_draw_selenium_lcms(evidence[2],data,"G");_draw_ng8_graph(rows[3],data,"H");return fig

def make_figure6_supplementary(data):
    fig=plt.figure(figsize=(_mm(297),_mm(480)),facecolor="white");rows=fig.subfigures(4,1,height_ratios=[1.02,.83,.90,1.82],hspace=.035);top=rows[0].subfigures(1,3,wspace=.045)
    _draw_four_set_upset(top[0],data,"SA","status_four_set_membership.csv","Metastatic-status four-set metabolic overlap");_draw_status_pathway_scatter(top[1],data,"SB");_draw_entity_concordance(top[2],data,"SJ","gpa");_draw_review_s1(rows[1],data,"S1")
    middle=rows[2].subfigures(1,2,width_ratios=[.44,.56],wspace=.055);_draw_filtering_cascade(middle[0],data,"SFUNNEL");_draw_homocysteine_screen(middle[1],data,"SDIRECTION");_draw_su3_tasks(rows[3],data,"SU3");return fig

def save_figure(fig, stem, dpi=300):
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ("png","pdf","svg"):fig.savefig(stem.with_suffix("."+suffix),dpi=dpi,bbox_inches=None,pad_inches=0)
