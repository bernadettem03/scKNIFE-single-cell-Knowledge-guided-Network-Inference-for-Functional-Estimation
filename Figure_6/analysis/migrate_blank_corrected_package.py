#!/usr/bin/env python3
"""Build the public Figure 6/S6 plotting packages from the validated working notebook."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_ind, wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
WORKSPACE = REPO.parents[1]
SOURCE_NOTEBOOK = WORKSPACE / "May_2026_clean_version_redo/Figures/submission/Figure_6_blank_corrected.ipynb"
SOURCE_DATA = SOURCE_NOTEBOOK.parent / "data/Figure_6_blank_corrected"
FIG6 = REPO / "Figure_6"
S6 = REPO / "Supplementary/Figure_S6"

MAIN_FILES = [
    "treatment_four_set_membership.csv", "surgery_pathway_scatter.csv", "surgery_pathway_enrichment.csv",
    "meia_vs_scknife_entity_correlations.csv", "robust_tier3_trajectory_selection.csv",
    "robust_tier3_trajectory_values.csv.gz", "robust_metabolite_pathway_assignments.csv",
    "robust_metabolite_pathway_context_counts.csv", "sulfur_cohort_patient_values.csv",
    "sulfur_lcms_feature_values.csv", "sulfur_lcms_feature_evidence.csv",
    "sulfur_ng8_nodes.csv", "sulfur_ng8_edges.csv", "sulfur_ng8_anchors.csv",
]
SUPPLEMENT_FILES = [
    "status_four_set_membership.csv", "status_pathway_scatter.csv", "status_pathway_enrichment.csv",
    "gpr_gpa_vs_scknife_entity_correlations.csv", "review_whole_profile_patient_summary.csv",
    "review_whole_profile_concordance_summary.csv", "robust_metabolite_filtering_funnel.csv",
    "sulfur_cohort_patient_values.csv", "sulfur_p3_all_score_cells.csv.gz",
]


def paired_rank_biserial(delta):
    delta = np.asarray(delta, dtype=float)
    nonzero = delta[~np.isclose(delta, 0)]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def derive_statistics(destination):
    cohort = pd.read_csv(SOURCE_DATA / "sulfur_cohort_patient_values.csv")
    celltypes = ["Macrophages", "T cells", "CAFs", "Endothelial", "Epithelial"]
    rows = []
    for cohort_name, keep in [("All 7", np.ones(len(cohort), bool)),
                              ("Without P3", ~cohort.is_longitudinal_case.astype(bool))]:
        for score in cohort.score_label.drop_duplicates():
            for celltype in celltypes:
                frame = cohort.loc[keep].query("score_label == @score and cell_type_analysis == @celltype")
                delta = (frame.Surgery - frame.Bx).to_numpy()
                test = wilcoxon(delta, alternative="two-sided", method="exact")
                rows.append(dict(cohort=cohort_name, score_label=score, cell_type=celltype,
                                 n=len(delta), n_decrease=int((delta < 0).sum()),
                                 n_increase=int((delta > 0).sum()), median_delta=float(np.median(delta)),
                                 rank_biserial=paired_rank_biserial(delta), p=float(test.pvalue)))
    stats = pd.DataFrame(rows)
    for cohort_name in ("All 7", "Without P3"):
        mask = stats.cohort.eq(cohort_name)
        stats.loc[mask, "q45"] = multipletests(stats.loc[mask, "p"], method="fdr_bh")[1]
    assert len(stats) == 90
    stats.to_csv(destination / "sulfur_cohort_treatment_statistics.csv", index=False)

    cells = pd.read_csv(SOURCE_DATA / "sulfur_p3_all_score_cells.csv.gz")
    comparisons = [("Surgery", "Bx"), ("Recurrence", "Surgery"), ("Recurrence", "Bx")]
    rows = []
    for score in cells.score_label.drop_duplicates():
        for celltype in celltypes:
            frame = cells.query("score_label == @score and cell_type_analysis == @celltype")
            for group_a, group_b in comparisons:
                a = frame.loc[frame.Site.eq(group_a), "score"].to_numpy()
                b = frame.loc[frame.Site.eq(group_b), "score"].to_numpy()
                t_stat, p_value = ttest_ind(a, b, equal_var=False)
                pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                rows.append(dict(label=score, ct=celltype, comparison=f"{group_a} vs {group_b}",
                                 n_a=len(a), n_b=len(b), change=(a.mean()/b.mean()-1)*100,
                                 d=(a.mean()-b.mean())/pooled_sd, welch_t=t_stat, p=p_value))
    p3 = pd.DataFrame(rows)
    assert len(p3) == 135
    p3["q135"] = multipletests(p3.p, method="fdr_bh")[1]
    p3.to_csv(destination / "sulfur_p3_task_statistics.csv", index=False)


def stage_data(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.iterdir():
        if old.is_file():
            old.unlink()
    for name in names:
        shutil.copy2(SOURCE_DATA / name, directory / name)
    derive_statistics(directory)
    manifest = []
    for path in sorted(directory.glob("*")):
        manifest.append({"path": path.name, "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (directory / "manifest.json").write_text(json.dumps({"files": manifest}, indent=2) + "\n")
    (directory / "README.md").write_text(
        "# Figure 6 plot-ready data\n\nChecksummed inputs for the public plotting notebook. "
        "Inferential tables are frozen by `../analysis/derive_blank_corrected_statistics.py`; "
        "the notebook performs visualization only.\n")


def source_plot_code():
    notebook = json.loads(SOURCE_NOTEBOOK.read_text())
    source = notebook["cells"][5]["source"]
    return "".join(source) if isinstance(source, list) else source


def build_plotting_module():
    source = source_plot_code()
    tree = ast.parse(source)
    keep_functions = {
        "_mm", "_clean_axis", "_wrap", "_panel_title",
        "_validated_integrated_pathway_scatter", "_draw_treatment_pathway_scatter",
        "_draw_status_pathway_scatter", "_draw_robust_trajectories", "_draw_four_set_upset",
        "_validated_entity_concordance", "_draw_entity_concordance", "_validated_robust_context",
        "_draw_pathway_context_pie", "_draw_filtering_cascade", "_draw_homocysteine_screen",
        "_draw_macrophage_tasks", "_draw_selenium_lcms", "_draw_ng8_graph", "_draw_review_s1",
        "_draw_su3_tasks",
    }
    chunks = ["\n".join(source.splitlines()[:68])]
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.lineno > 68:
            chunks.append(ast.get_source_segment(source, node))
        elif isinstance(node, ast.FunctionDef) and node.name in keep_functions:
            chunks.append(ast.get_source_segment(source, node))
    module = "\n\n".join(chunks)
    module = module.replace("from scipy.stats import pearsonr, spearmanr\n", "")
    # Public renderers consume frozen inferential tables.
    start = module.index("def _sulfur_treatment_statistics") if "def _sulfur_treatment_statistics" in module else -1
    if start >= 0:
        end = module.index("\ndef _draw_homocysteine_screen", start)
        module = module[:start] + "def _sulfur_treatment_statistics(data):\n    stats=data['sulfur_cohort_treatment_statistics.csv'].copy()\n    if len(stats)!=90: raise RuntimeError('Frozen sulfur treatment statistics changed')\n    return stats\n\n" + module[end+1:]
    module = module.replace("stats=_sulfur_treatment_statistics(cohort)", "stats=_sulfur_treatment_statistics(data)")
    # Replace SU3 inferential block with frozen statistics lookup.
    old_a = module.index("def _draw_su3_tasks")
    old_b = len(module)
    su3 = '''def _draw_su3_tasks(container,data,label="SU3"):
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
            ax.set_title(f"{celltype}\\nS→R {stat.change:+.1f}%; d={stat.d:+.2f}; q135={probability(stat.q135)}",fontsize=7.2,fontweight="bold");ax.set_xticks(range(3),[f"{site}\\nn={len(value):,}" for site,value in zip(sites,values)],fontsize=6.5);ax.spines[["top","right"]].set_visible(False);ax.grid(False)
            if column_index==0:ax.set_ylabel("Cell-level activity score")
        axes[row_index,0].text(-.5,.5,score.replace("MEIA task | ",""),transform=axes[row_index,0].transAxes,rotation=90,ha="center",va="center",fontsize=8,fontweight="bold")
    container.suptitle(f"{label}  P3 homocysteine-degradation and glutathione-synthesis trajectories",x=.01,y=.985,ha="left",fontweight="bold",fontsize=PANEL_LABEL_SIZE_PT)
    container.text(.5,.018,"Violin width=density; white box=IQR; line=median; whiskers=1.5×IQR; diamond/line=mean; box outliers hidden.",ha="center",fontsize=6.5,color="#555555")
    container.text(.5,.004,"Welch tests; BH across original 135 tests. One patient; cells are pseudoreplicates.",ha="center",fontsize=6.5,color="#555555");container.subplots_adjust(left=.07,right=.995,top=.86,bottom=.14,hspace=.72,wspace=.38)
'''
    module = module[:old_a] + su3
    module += '''

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
'''
    if any(term in module for term in ("pearsonr(", "spearmanr(", "multipletests(", "ttest_ind(", "wilcoxon(")):
        raise RuntimeError("Public plotting module still contains inferential calls")
    return module


def notebook(title, supplementary=False):
    labels = list(("SA", "SB", "SJ", "S1", "SFUNNEL", "SDIRECTION", "SU3") if supplementary else tuple("ABCDEFGH"))
    inputs = "SUPPLEMENT_INPUTS" if supplementary else "MAIN_INPUTS"
    panel_fun = "make_figure6_supplementary_panel" if supplementary else "make_figure6_main_panel"
    compiled_fun = "make_figure6_supplementary" if supplementary else "make_figure6_compiled"
    compiled_name = "Figure_S6" if supplementary else "Figure_6"
    cells = [
        {"cell_type":"markdown","metadata":{},"source":f"# {title}\n\nLive, repository-relative plotting from checksummed plot-ready data. Statistical derivation is isolated under `Figure_6/analysis/`."},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":"%matplotlib inline\nfrom pathlib import Path\nimport json\nimport hashlib\nimport matplotlib.pyplot as plt\nfrom plotting import *\nROOT=Path.cwd(); DATA_ROOT=ROOT/'data'; OUTPUT_ROOT=ROOT/'outputs'; PANEL_ROOT=OUTPUT_ROOT/'panels'\n"},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":f"DATA=load_tables(DATA_ROOT,{inputs})\nprint(f'Loaded {{len(DATA)}} checksummed plot-ready tables from data/')"},
        {"cell_type":"markdown","metadata":{},"source":"## Standalone panels"},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":f"""panel_manifest={{'figure':'{compiled_name.replace('_',' ')}','panel_order':{labels!r},'formats':['png','pdf'],'png_dpi':300,'note':'Standalone exports generated by the authoritative plotting-only notebook.','panels':{{}}}}
for label in {labels!r}:
    fig={panel_fun}(label,DATA)
    save_figure(fig,PANEL_ROOT/f'{compiled_name}_{{label}}')
    panel_manifest['panels'][label]={{}}
    for ext in ('png','pdf'):
        path=PANEL_ROOT/f'{compiled_name}_{{label}}.{{ext}}'; rel=path.relative_to(ROOT.parent if "{compiled_name}"=="Figure_6" else ROOT.parents[1])
        panel_manifest['panels'][label][ext]={{'path':str(rel),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
    plt.show();plt.close(fig)
(PANEL_ROOT/'{compiled_name}_panel_export_manifest.json').write_text(json.dumps(panel_manifest,indent=2)+\'\\n\')"""},
        {"cell_type":"markdown","metadata":{},"source":"## Compiled vector figure"},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":f"fig={compiled_fun}(DATA)\nsave_figure(fig,OUTPUT_ROOT/'{compiled_name}')\nplt.show();plt.close(fig)"},
    ]
    return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.9"}},"nbformat":4,"nbformat_minor":5}


def main():
    stage_data(FIG6 / "data", MAIN_FILES)
    stage_data(S6 / "data", SUPPLEMENT_FILES)
    plotting = build_plotting_module()
    (FIG6 / "plotting.py").write_text(plotting)
    (S6 / "plotting.py").write_text(plotting)
    (FIG6 / "Figure_6.ipynb").write_text(json.dumps(notebook("Figure 6 — blank-corrected metabolic evidence story"), indent=1) + "\n")
    (S6 / "Figure_S6.ipynb").write_text(json.dumps(notebook("Supplementary Figure S6 — robustness and longitudinal evidence", True), indent=1) + "\n")


if __name__ == "__main__":
    main()
