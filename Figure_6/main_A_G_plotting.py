"""Reproducible Figure 6 A–T renderer from staged patient-level tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import plotting as legacy
import restored_figure6_plotting as restored
from pooled_concordance import draw_single_concordance

PATIENTS = ("P1", "P2", "P3", "P4",
            "P5", "P6", "P7")
PATIENT_COLORS = plt.cm.Set2(np.linspace(0, 1, len(PATIENTS)))
STATUS_COLORS = {"Metastatic": "#7B3294", "Non-metastatic": "#008837"}
TIME_COLORS = {"Bx": "#377EB8", "Surgery": "#E41A1C"}
RECURRENCE_COLOR = "#D55E00"
RECURRENCE_TITLES = {"J": "Myeloid peroxisomal carnitine shuttle"}
COMPACT_TITLES = {
    "J": "Myeloid shuttle",
    "K": "L-carnitine", "L": "L-octanoylcarnitine", "M": "octanoyl-CoA",
}
COMPACT_NODE_TITLES = {"D": "Selenocystathionine\nactivity score",
                       "F": "L-cystathionine\nactivity score"}
DISPLAY_TO_SOURCE = {
    "A": "A", "B": "B", "C": "C", "D": "D",
    "E": "E", "F": "E", "G": "E", "H": "F",
    "I": "G", "J": "G", "K": "H", "L": "L",
    "M": "M", "N": "N", "O": "L", "P": "L",
    "Q": "p3_lipid_cells", "R": "p3_lipid_cells",
    "S": "p3_lipid_cells", "T": "lipid_change_12",
}
LIPID_TITLES = {
    "L": "Epithelial fatty acid activation · cytosolic",
    "M": "Epithelial fatty acid activation · ER",
    "N": "Myeloid peroxisomal carnitine shuttle",
}
PURINE_DISPLAY = {"adenine": "E", "guanine": "F", "deoxyinosine": "G"}
ASPARTATE_DISPLAY = {"3_sulfinoalanine": "I", "S_2_aminobutanoate": "J"}

FIGURE_FONT_SIZE = 6

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FIGURE_FONT_SIZE,
    "axes.titlesize": FIGURE_FONT_SIZE, "axes.labelsize": FIGURE_FONT_SIZE,
    "xtick.labelsize": FIGURE_FONT_SIZE, "ytick.labelsize": FIGURE_FONT_SIZE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def load_data(root: Path) -> dict[str, pd.DataFrame]:
    data = restored.load_data(root)
    for name in ("fig6_main_patient_values.csv", "fig6_main_patient_stats.csv",
                 "fig6_gh_p3_recurrence.csv",
                 "lcms_node_pathway_screen.csv", "lcms_node_pathway_membership.csv",
                 "recurrence_patient_values.csv", "recurrence_panel_summary.csv",
                 "carnitine_patient_values.csv", "carnitine_panel_summary.csv",
                 "carnitine_focused_family.csv", "carnitine_p3_trajectory.csv",
                 "lipid_pathway_patient_values.csv", "lipid_pathway_summary.csv",
                 "lipid_pathway_specificity.csv", "lipid_trajectory_patient_values.csv",
                 "lipid_trajectory_summary.csv", "p3_lipid_cell_scores.csv.gz",
                 "carnitine_activity_umap.csv.gz", "lipid_change_12_patient_values.csv",
                 "lipid_change_12_summary.csv"):
        data[name] = pd.read_csv(Path(root) / name)
    data["carnitine_graph_subset.json"] = json.loads((Path(root) / "carnitine_graph_subset.json").read_text())
    pairs, stats = data["fig6_main_patient_values.csv"], data["fig6_main_patient_stats.csv"]
    expected = {("D", "Myeloid", "selenocystathionine"),
                ("E", "Myeloid", "adenine"), ("E", "Myeloid", "guanine"),
                ("E", "Myeloid", "deoxyinosine"),
                ("F", "Myeloid", "L_cystathionine"),
                ("G", "Myeloid", "3_sulfinoalanine"),
                ("G", "Myeloid", "S_2_aminobutanoate"),
                ("H", "Myeloid", "methylglyoxal")}
    assert set(zip(stats.panel, stats.celltype, stats.node)) == expected
    assert len(pairs) == 56 and set(pairs.PatientID) == set(PATIENTS)
    assert stats.patients.eq(7).all()
    p3_nodes = data["fig6_gh_p3_recurrence.csv"]
    assert set(zip(p3_nodes.panel, p3_nodes.node)) == {
        ("G", "3_sulfinoalanine"), ("G", "S_2_aminobutanoate"),
        ("H", "methylglyoxal")}
    assert p3_nodes.PatientID.eq(PATIENTS[2]).all()
    assert p3_nodes.recurrence_cells.eq(227).all()
    assert len(data["lcms_node_pathway_screen.csv"]) == 1800
    assert data["lcms_node_pathway_screen.csv"].global_node_cell_BH_q.min() > .05
    recurrence = data["recurrence_patient_values.csv"]
    rebound = data["recurrence_panel_summary.csv"]
    assert rebound.panel.tolist() == list("HIJ")
    assert len(recurrence) == 21 and recurrence.groupby("panel").PatientID.nunique().eq(7).all()
    assert recurrence.loc[recurrence.PatientID.eq("P3"), "recurrence_median"].notna().all()
    assert recurrence.loc[~recurrence.PatientID.eq("P3"), "recurrence_median"].isna().all()
    carnitine = data["carnitine_panel_summary.csv"]
    assert carnitine.panel.tolist() == list("KLM")
    assert carnitine.patients_lower_at_surgery.between(0, 7).all()
    assert len(data["carnitine_patient_values.csv"]) == 21
    status_pairs = data["carnitine_patient_values.csv"]
    assert status_pairs.groupby("PatientID").Status.nunique().eq(1).all()
    assert status_pairs.loc[status_pairs.Status.eq("Metastatic"), "PatientID"].nunique() == 4
    assert status_pairs.loc[status_pairs.Status.eq("Non-metastatic"), "PatientID"].nunique() == 3
    assert len(data["carnitine_focused_family.csv"]) == 60
    assert data["carnitine_p3_trajectory.csv"].patient_median.notna().all()
    assert len(data["lipid_pathway_patient_values.csv"]) == 420
    assert len(data["lipid_pathway_summary.csv"]) == 60
    assert len(data["lipid_pathway_specificity.csv"]) == 42
    assert len(data["lipid_change_12_patient_values.csv"]) == 504
    assert len(data["lipid_change_12_summary.csv"]) == 72
    assert len(data["p3_lipid_cell_scores.csv.gz"]) == 8918
    assert len(data["carnitine_activity_umap.csv.gz"]) == 69391
    lipid = data["lipid_trajectory_patient_values.csv"]
    lipid_stats = data["lipid_trajectory_summary.csv"]
    assert lipid_stats.panel.tolist() == list("LMN")
    assert len(lipid) == 21 and lipid.groupby("panel").PatientID.nunique().eq(7).all()
    assert lipid.loc[lipid.PatientID.eq(PATIENTS[2]), "recurrence_median"].notna().all()
    assert lipid.loc[~lipid.PatientID.eq(PATIENTS[2]), "recurrence_median"].isna().all()
    return data


def _patient_legend(container) -> None:
    handles = [Line2D([0], [0], color=PATIENT_COLORS[i], marker="o", markersize=4,
                      linewidth=1, label=f"P{i+1}") for i in range(len(PATIENTS))]
    container.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
                     bbox_to_anchor=(.5, .045), fontsize=6.5)


def _paired_node_boxes(ax, frame: pd.DataFrame) -> None:
    """Use one biopsy/surgery box style throughout the D–K node panels."""
    boxes = ax.boxplot(
        [frame.biopsy_median.to_numpy(), frame.surgery_median.to_numpy()],
        positions=[0, 1], widths=.27, patch_artist=True, showfliers=False,
        whis=(0, 100), medianprops={"color": "#333333", "linewidth": 1},
        whiskerprops={"color": "#777777", "linewidth": .7},
        capprops={"color": "#777777", "linewidth": .7},
        boxprops={"edgecolor": "#666666", "linewidth": .7},
    )
    for patch, timepoint in zip(boxes["boxes"], ("Bx", "Surgery")):
        patch.set_facecolor(TIME_COLORS[timepoint])
        patch.set_alpha(.17)


def _paired_node_patients(ax, frame: pd.DataFrame, compact: bool) -> None:
    """Keep each patient's color and mark style consistent across D–K."""
    offsets = np.linspace(-.055, .055, len(PATIENTS))
    for index, patient in enumerate(PATIENTS):
        row = frame.loc[patient]
        x = [offsets[index], 1 + offsets[index]]
        y = [row.biopsy_median, row.surgery_median]
        ax.plot(x, y, color=PATIENT_COLORS[index],
                linewidth=.8 if compact else 1, alpha=.82, zorder=2)
        ax.scatter(x, y, s=12 if compact else 22,
                   color=PATIENT_COLORS[index], edgecolors="white",
                   linewidths=.3, zorder=3)


def _paired_axis(ax, pairs: pd.DataFrame, stat: pd.Series, title: str,
                 compact: bool = False, show_ylabel: bool = True) -> None:
    frame = pairs.loc[pairs.node.eq(stat.node) & pairs.celltype.eq(stat.celltype)]
    frame = frame.set_index("PatientID").loc[list(PATIENTS)]
    _paired_node_boxes(ax, frame)
    _paired_node_patients(ax, frame, compact)
    observed = frame[["biopsy_median", "surgery_median"]].to_numpy()
    low, high = float(observed.min()), float(observed.max())
    span = high - low
    ax.set_ylim(low - .04 * span, high + .25 * span)
    ax.set_xlim(-.34, 1.34)
    ax.set_xticks([0, 1], ["Bx", "Surg."] if compact else ["Biopsy", "Surgery"])
    ax.set_title(title, loc="left", fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Inferred scKNIFE node activity")
        if compact:
            ax.yaxis.labelpad = 0
            ax.yaxis.label.set_fontsize(7)
    if not compact:
        ax.text(.98, .97, f"{int(stat.patients_lower_at_surgery)}/7 lower at surgery\n"
                f"Median paired Δ = {stat.median_paired_delta:+.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95, "pad": 2},
                zorder=10)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)


def draw_node(container, label: str, pairs: pd.DataFrame, stats: pd.DataFrame,
              with_legend: bool = False, compact: bool = False,
              display_letter: str = "") -> None:
    node, title = {"D": ("selenocystathionine", "Myeloid selenocystathionine"),
                   "F": ("L_cystathionine", "Myeloid L-cystathionine")}[label]
    ax = container.subplots(1, 1)
    stat = stats.loc[stats.node.eq(node)].iloc[0]
    _paired_axis(ax, pairs, stat, title,
                 compact=compact, show_ylabel=not compact or label == "D")
    if compact and label == "D":
        ax.set_ylabel("Inferred node activity score", fontsize=7)
    container.text(.025, .985,
                   f"{display_letter or label}  {COMPACT_NODE_TITLES[label] if compact else title}",
                   ha="left", va="top", fontsize=7 if compact else 9, fontweight="bold")
    if not compact:
        container.text(.025, .88, "Biopsy → surgery after nACT; n=7",
                       ha="left", va="top", fontsize=6.5, color="#555555")
    ax.set_title("", loc="left")
    ax.set_title(f"{int(stat.patients_lower_at_surgery)}/7 lower · "
                 f"P={stat.paired_wilcoxon_two_sided_unadjusted_p:.4f}",
                 fontsize=5.2 if compact else 7.2, pad=3)
    container.subplots_adjust(left=(.28 if label == "D" else .20) if compact else .24,
                              right=.96, top=.71 if compact else .74,
                              bottom=.24 if compact else .27)
    if with_legend:
        _patient_legend(container)


def draw_purine_boxplots(container, pairs: pd.DataFrame, stats: pd.DataFrame,
                         compact: bool = False, selected_node: str | None = None) -> None:
    """Paired patient-median boxplots for the three purine nodes, E–G."""
    nodes = (("adenine", "Adenine"), ("guanine", "Guanine"),
             ("deoxyinosine", "Deoxyinosine"))
    if selected_node is not None:
        nodes = tuple(item for item in nodes if item[0] == selected_node)
        assert len(nodes) == 1
    axes = np.atleast_1d(container.subplots(1, len(nodes), sharey=True))
    for index, (ax, (node, name)) in enumerate(zip(axes, nodes)):
        frame = pairs.loc[pairs.panel.eq("E") & pairs.node.eq(node)]
        frame = frame.set_index("PatientID").loc[list(PATIENTS)]
        stat = stats.loc[stats.panel.eq("E") & stats.node.eq(node)].iloc[0]
        _paired_node_boxes(ax, frame)
        _paired_node_patients(ax, frame, compact)
        ax.set_title((f"{int(stat.patients_lower_at_surgery)}/7 lower · "
                      f"P={stat.paired_wilcoxon_two_sided_unadjusted_p:.4f}")
                     if compact and selected_node else
                     (f"{name}\n{int(stat.patients_lower_at_surgery)}/7 lower · "
                      f"P={stat.paired_wilcoxon_two_sided_unadjusted_p:.4f}"),
                     fontsize=5.2 if compact else 7.2, pad=3)
        if not (compact and selected_node):
            ax.text(-.10, 1.19, PURINE_DISPLAY[node], transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8 if compact else 10,
                    fontweight="bold", clip_on=False)
        ax.set_xlim(-.34, 1.34)
        ax.set_xticks([0, 1], ["Bx", "Surg."] if compact else ["Biopsy", "Surgery"])
        ax.set_ylim(0, .65)
        ax.tick_params(axis="both", labelsize=5.5 if compact else 7, length=2)
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        if index == 0 and (not compact or node == "adenine"):
            ax.set_ylabel("Inferred node activity score", fontsize=6 if compact else 8)
        elif compact and node != "adenine":
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)
    container.text(.025, .985,
                   f"{PURINE_DISPLAY[nodes[0][0]]}  {nodes[0][1]}\nactivity score"
                   if compact and selected_node else "Myeloid purine nodes",
                   ha="left", va="top",
                   fontsize=7 if compact else 9, fontweight="bold")
    if not compact and selected_node is None:
        container.text(.025, .89, "Seven paired patient medians · shared node-score scale",
                       ha="left", va="top", fontsize=7, color="#555555")
    container.subplots_adjust(left=(.25 if selected_node == "adenine" else .13)
                              if compact else (.23 if selected_node else .09),
                              right=.99 if compact else .98,
                              top=.71 if compact else .71,
                              bottom=.24 if compact else .23,
                              wspace=.24 if compact else .17)
    if not compact and selected_node is not None:
        _patient_legend(container)


def draw_ik_paired(container, label: str, pairs: pd.DataFrame,
                   stats: pd.DataFrame, compact: bool = False,
                   selected_node: str | None = None) -> None:
    """Show paired biopsy and surgery boxplots for the I–K Myeloid nodes."""
    nodes = {"G": (("3_sulfinoalanine", "3-sulfinoalanine"),
                   ("S_2_aminobutanoate", "(S)-2-aminobutanoate")),
             "H": (("methylglyoxal", "Methylglyoxal"),)}[label]
    if selected_node is not None:
        nodes = tuple(item for item in nodes if item[0] == selected_node)
        assert len(nodes) == 1
    axes = container.subplots(1, len(nodes))
    axes = np.atleast_1d(axes)
    for index, (ax, (node, name)) in enumerate(zip(axes, nodes)):
        frame = pairs.loc[pairs.panel.eq(label) & pairs.node.eq(node)]
        frame = frame.set_index("PatientID").loc[list(PATIENTS)]
        stat = stats.loc[stats.panel.eq(label) & stats.node.eq(node)].iloc[0]
        _paired_node_boxes(ax, frame)
        _paired_node_patients(ax, frame, compact)
        observed = np.r_[frame.biopsy_median.to_numpy(),
                         frame.surgery_median.to_numpy()]
        low, high = float(observed.min()), float(observed.max())
        span = high - low
        ax.set_ylim(low - .10 * span, high + .20 * span)
        ax.set_xlim(-.34, 1.34)
        ax.set_xticks([0, 1], ["Bx", "Surg."] if compact else
                      ["Biopsy", "Surgery"])
        display = ASPARTATE_DISPLAY[node] if label == "G" else "K"
        ax.set_title((f"{int(stat.patients_lower_at_surgery)}/7 lower · "
                      f"P={stat.paired_wilcoxon_two_sided_unadjusted_p:.4f}")
                     if compact and selected_node else
                     (f"{name}\n{int(stat.patients_lower_at_surgery)}/7 lower · "
                      f"P={stat.paired_wilcoxon_two_sided_unadjusted_p:.4f}"),
                     fontsize=5.2 if compact else 7.2, pad=3)
        if not (compact and selected_node):
            ax.text(-.10, 1.19, display, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8 if compact else 10,
                    fontweight="bold", clip_on=False)
        ax.tick_params(labelsize=5.4 if compact else 7, length=2)
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        if index == 0 and (not compact or selected_node == "3_sulfinoalanine"):
            ax.set_ylabel("Inferred node activity score", fontsize=6 if compact else 8)
        elif compact and selected_node:
            ax.set_ylabel("")
    titles = {"G": ("Myeloid 3-sulfinoalanine" if selected_node == "3_sulfinoalanine"
                    else "Myeloid (S)-2-aminobutanoate" if selected_node
                    else "Myeloid aspartate-linked nodes"),
              "H": "Myeloid methylglyoxal"}
    compact_title = f"{ASPARTATE_DISPLAY[nodes[0][0]] if label == 'G' else 'K'}  {nodes[0][1]}\nactivity score"
    container.text(.025, .985, compact_title if compact and selected_node else titles[label],
                   ha="left", va="top", fontsize=7 if compact else 9,
                   fontweight="bold")
    if not compact:
        container.text(.025, .89, "Seven paired patient medians",
                       ha="left", va="top", fontsize=6.5, color="#555555")
    container.subplots_adjust(left=(.26 if selected_node == "3_sulfinoalanine" else .18)
                              if compact else (.23 if selected_node else .12),
                              right=.98, top=.71 if compact else .70,
                              bottom=.24 if compact else .23,
                              wspace=.31 if label == "G" else .15)
    if not compact and selected_node is not None:
        _patient_legend(container)


def draw_recurrence(container, label: str, data: dict[str, pd.DataFrame],
                    compact: bool = False, display_letter: str = "") -> None:
    """Seven paired treatment medians and one highlighted P3 recurrence trajectory."""
    frame = data["recurrence_patient_values.csv"]
    frame = frame.loc[frame.panel.eq(label)].set_index("PatientID").loc[list(PATIENTS)]
    stat = data["recurrence_panel_summary.csv"].set_index("panel").loc[label]
    ax = container.subplots(1, 1)
    boxes = ax.boxplot(
        [frame.biopsy_median.to_numpy(), frame.surgery_median.to_numpy()],
        positions=[0, 1], widths=.24, patch_artist=True, showfliers=False,
        whis=(0, 100), medianprops={"color": "#333333", "linewidth": 1.2},
        whiskerprops={"color": "#777777", "linewidth": .8},
        capprops={"color": "#777777", "linewidth": .8},
        boxprops={"edgecolor": "#666666", "linewidth": .8},
    )
    for patch, timepoint in zip(boxes["boxes"], ("Bx", "Surgery")):
        patch.set_facecolor(TIME_COLORS[timepoint])
        patch.set_alpha(.16)
        patch.set_zorder(0)
    for patient in PATIENTS:
        row = frame.loc[patient]
        p3 = patient == "P3"
        color = RECURRENCE_COLOR if p3 else "#AAB4BE"
        ax.plot([0, 1], [row.biopsy_median, row.surgery_median],
                color=color, linewidth=1.6 if p3 else .8,
                alpha=1 if p3 else .75, zorder=3 if p3 else 1)
        ax.scatter([0, 1], [row.biopsy_median, row.surgery_median],
                   s=25 if p3 else 12, color=color, edgecolors="white",
                   linewidths=.45, zorder=4 if p3 else 2)
        if p3:
            ax.plot([1, 2], [row.surgery_median, row.recurrence_median],
                    color=color, linewidth=1.8, zorder=4)
            ax.scatter([2], [row.recurrence_median], s=34, color=color,
                       edgecolors="white", linewidths=.5, zorder=5)
    observed = np.r_[frame.biopsy_median.to_numpy(),
                     frame.surgery_median.to_numpy(),
                     frame.loc["P3", "recurrence_median"]]
    low, high = float(observed.min()), float(observed.max())
    span = max(high - low, abs(high) * .02)
    ax.set_ylim(low - .08 * span, high + .18 * span)
    ax.set_xlim(-.18, 2.18)
    ax.set_xticks([0, 1, 2], ["Bx", "Surg.", "Rec.\n(P3)"] if compact else
                  ["Biopsy", "Surgery", "Recurrence\n(P3)"])
    if not compact or label == "H":
        ax.set_ylabel("Relative MEIA score")
        if compact:
            ax.yaxis.labelpad = 0
            ax.yaxis.label.set_fontsize(7)
    if not compact:
        ax.text(.98, .97,
                f"{int(stat.patients_in_treatment_direction)}/7 paired; p={stat.paired_wilcoxon_two_sided_p:.4f}\n"
                f"screen BH q={stat.treatment_BH_FDR:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .94, "pad": 2})
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    container.text(.025, .985, f"{display_letter or label}  {COMPACT_TITLES[label] if compact else RECURRENCE_TITLES[label]}",
                   ha="left", va="top", fontsize=7 if compact else 8,
                   fontweight="bold", wrap=True)
    if not compact:
        container.text(.025, .88, "P3 recurrence shown in orange",
                       ha="left", va="top", fontsize=6.5, color="#555555")
    ax.set_title("")
    container.subplots_adjust(left=(.40 if label == "H" else .32) if compact else .22, right=.97,
                              top=.84 if compact else .73, bottom=.24 if compact else .25)


def draw_lipid_pair(container, label: str, data: dict,
                    compact: bool = False) -> None:
    """Seven independent patient medians at biopsy and surgery."""
    frame = data["lipid_trajectory_patient_values.csv"]
    frame = frame.loc[frame.panel.eq(label)].set_index("PatientID").loc[list(PATIENTS)]
    stat = data["lipid_trajectory_summary.csv"].set_index("panel").loc[label]
    ax = container.subplots(1, 1)
    boxes = ax.boxplot([frame.biopsy_median.to_numpy(), frame.surgery_median.to_numpy()],
                       positions=[0, 1], widths=.26, patch_artist=True,
                       showfliers=False, whis=(0, 100),
                       medianprops={"color": "#333333", "linewidth": 1.1},
                       whiskerprops={"color": "#777777", "linewidth": .75},
                       capprops={"color": "#777777", "linewidth": .75},
                       boxprops={"edgecolor": "#666666", "linewidth": .75})
    for patch, site in zip(boxes["boxes"], ("Bx", "Surgery")):
        patch.set_facecolor(TIME_COLORS[site]); patch.set_alpha(.18)
    for patient in PATIENTS:
        row = frame.loc[patient]
        is_p3 = patient == PATIENTS[2]
        color = RECURRENCE_COLOR if is_p3 else "#AAB4BE"
        ax.plot([0, 1], [row.biopsy_median, row.surgery_median], color=color,
                lw=1.55 if is_p3 else .8, alpha=1 if is_p3 else .8, zorder=2)
        ax.scatter([0, 1], [row.biopsy_median, row.surgery_median],
                   color=color, s=24 if is_p3 else 13, edgecolors="white",
                   linewidths=.4, zorder=3)
    observed = frame[["biopsy_median", "surgery_median"]].to_numpy(float)
    lo, hi = float(observed.min()), float(observed.max())
    span = max(hi-lo, abs(hi)*.02)
    ax.set_ylim(lo-.09*span, hi+.18*span)
    ax.set_xlim(-.3, 1.3)
    ax.set_xticks([0, 1], ["Bx", "Surg."] if compact else ["Biopsy", "Surgery"])
    if not compact:
        ax.set_ylabel("Relative MEIA pathway score", fontsize=8)
    direction = "higher" if stat.median_paired_change > 0 else "lower"
    ax.set_title(f"{int(stat.patients_in_treatment_direction)}/7 {direction} · "
                 f"P={stat.paired_wilcoxon_two_sided_p:.4f}",
                 fontsize=6.1 if compact else 7.2, pad=3)
    ax.grid(False); ax.spines[["top", "right"]].set_visible(False)
    container.text(.025, .985, f"{label}  {LIPID_TITLES[label]}",
                   ha="left", va="top", fontsize=7 if compact else 8,
                   fontweight="bold", wrap=True)
    if not compact:
        container.text(.025, .89,
                       f"Seven paired patient medians · screen BH q={stat.treatment_BH_FDR:.3f}",
                       ha="left", va="top", fontsize=6.3, color="#555555")
    container.subplots_adjust(left=(.25 if label == "L" else .16) if compact else .22, right=.98,
                              top=.72 if compact else .72, bottom=.24)


def draw_p3_lipid_violins(container, data: dict, compact: bool = False,
                         selected: str | None = None) -> None:
    """Descriptive single-cell distributions for P3's three selected pathways."""
    frame = data["p3_lipid_cell_scores.csv.gz"]
    medians = data["lipid_trajectory_patient_values.csv"]
    medians = medians.loc[medians.PatientID.eq(PATIENTS[2])].set_index("panel")
    labels = {"Q": "L", "R": "M", "S": "N"}
    shown = [selected] if selected else list(labels)
    axes = np.atleast_1d(container.subplots(1, len(shown))).ravel()
    names = {"Q": "Epithelial cytosolic\nFA activation activity",
             "R": "Epithelial ER\nFA activation activity",
             "S": "Myeloid peroxisomal\ncarnitine shuttle activity"}
    sites = ("Bx", "Surgery", "Recurrence")
    colors = (TIME_COLORS["Bx"], TIME_COLORS["Surgery"], RECURRENCE_COLOR)
    for ax, display in zip(axes, shown):
        label = labels[display]
        subset = frame.loc[frame.panel.eq(label)]
        values = [subset.loc[subset.timepoint.eq(site), "score"].to_numpy(float)
                  for site in sites]
        assert all(len(group) > 0 and np.isfinite(group).all() for group in values)
        violins = ax.violinplot(values, positions=np.arange(3), widths=.65,
                                showmeans=False, showmedians=False, showextrema=False)
        for body, color in zip(violins["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(.36)
        observed_medians = np.array([np.median(group) for group in values])
        expected = medians.loc[label, ["biopsy_median", "surgery_median",
                                       "recurrence_median"]].to_numpy(float)
        assert np.allclose(observed_medians, expected, atol=1e-12)
        ax.plot(range(3), observed_medians, color="#202020", linewidth=1.1,
                alpha=.85, zorder=4)
        ax.scatter(range(3), observed_medians, s=17, color="#202020", zorder=5)
        ax.set_xticks(range(3), [f"Bx\nn={len(values[0]):,}",
                                   f"Surg.\nn={len(values[1]):,}",
                                   f"Rec.\nn={len(values[2]):,}"])
        if selected:
            container.text(.02, .99, f"{display}  {names[display]}",
                           ha="left", va="top", fontweight="bold",
                           fontsize=6.4 if compact else 8)
        else:
            ax.set_title(f"{display}  {names[display]}", loc="left",
                         fontweight="bold", fontsize=6.4 if compact else 8, pad=3)
        ax.tick_params(labelsize=5.7 if compact else 7, length=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    axes[0].set_ylabel("Relative MEIA pathway score", fontsize=6 if compact else 8)
    if not selected:
        container.text(.015, .99, "P3 inferred pathway activity scores",
                       ha="left", va="top", fontsize=7.2 if compact else 9,
                       fontweight="bold")
    container.text(.015, .85, "P3 · specimen medians · descriptive" if selected else
                   "One patient · lines connect specimen medians · descriptive",
                   ha="left", va="top", fontsize=6.1 if compact else 7,
                   color="#555555")
    container.subplots_adjust(left=.075 if not selected else .20, right=.985,
                              top=.68 if not selected else .70, bottom=.23, wspace=.32)


def draw_carnitine_composite(container, data: dict, compact: bool = False,
                            selected: str | None = None) -> None:
    """Frozen atlas localization and paired patient medians of one carnitine node."""
    frame = data["carnitine_activity_umap.csv.gz"]
    myeloid = frame.loc[frame.celltype.eq("Myeloid") &
                        frame.Timepoint.isin(("Bx", "Surgery"))]
    assert len(myeloid) == 6014
    node = "L_octanoylcarnitine"
    scores = myeloid[node].to_numpy(float)
    low, high = np.percentile(scores, [2, 98])
    ax = container.add_axes([.015, .20, .52, .57] if selected is None else
                            [.10, .20, .78, .57]) if selected != "P" else None
    if ax is not None:
        ax.scatter(frame.UMAP1, frame.UMAP2, s=.07, c="#D9D9D9",
                   alpha=.4, linewidths=0, rasterized=True)
        scatter = ax.scatter(myeloid.UMAP1, myeloid.UMAP2, c=scores, s=.45,
                             cmap="viridis", vmin=low, vmax=high,
                             linewidths=0, rasterized=True)
        xy = frame[["UMAP1", "UMAP2"]].to_numpy(float)
        middle = xy.mean(axis=0)
        half = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) * .54
        ax.set_xlim(middle[0]-half, middle[0]+half)
        ax.set_ylim(middle[1]-half, middle[1]+half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
        colorbar = container.colorbar(scatter, ax=ax, fraction=.04, pad=.005)
        colorbar.ax.tick_params(labelsize=4.5 if compact else 6, length=1)
    pair_ax = container.add_axes([.66, .22, .32, .50] if selected is None else
                                 [.22, .22, .58, .50]) if selected != "O" else None
    if pair_ax is not None:
        stats = data["carnitine_panel_summary.csv"]
        stat = stats.loc[stats.panel.eq("L")].iloc[0]
        pairs = data["carnitine_patient_values.csv"]
        pairs = pairs.loc[pairs.panel.eq("L")].set_index("PatientID").loc[list(PATIENTS)]
        boxes = pair_ax.boxplot([pairs.biopsy_median, pairs.surgery_median],
                             positions=[0, 1], widths=.3, patch_artist=True,
                             showfliers=False, whis=(0, 100))
        for patch, color in zip(boxes["boxes"], (TIME_COLORS["Bx"], TIME_COLORS["Surgery"])):
            patch.set_facecolor(color); patch.set_alpha(.2)
        for index, patient in enumerate(PATIENTS):
            row = pairs.loc[patient]
            color = PATIENT_COLORS[index]
            pair_ax.plot([0, 1], [row.biopsy_median, row.surgery_median],
                     color=color, lw=1, zorder=3)
            pair_ax.scatter([0, 1], [row.biopsy_median, row.surgery_median],
                        s=10, color=color, edgecolors="white", linewidths=.3, zorder=4)
        pair_ax.set_xticks([0, 1], ["Bx", "Surg."])
        pair_ax.set_xlim(-.3, 1.3)
        if selected == "P":
            pair_ax.set_ylabel("Inferred node activity score")
        pair_ax.tick_params(labelsize=5.2 if compact else 7, length=2)
        pair_ax.spines[["top", "right"]].set_visible(False)
        pair_ax.set_title(f"7/7 lower · P={stat.paired_wilcoxon_two_sided_nominal_p:.4f}",
                      fontsize=5 if compact else 6.5, pad=2)
    if selected != "P":
        container.text(.02, .99, "O  L-octanoylcarnitine UMAP",
                       ha="left", va="top", fontsize=7 if compact else 9,
                       fontweight="bold")
    if selected != "O":
        container.text(.66 if selected is None else .02, .99,
                       "P  Myeloid\nL-octanoylcarnitine activity",
                       ha="left", va="top", fontsize=7 if compact else 9,
                       fontweight="bold")
    container.text(.02, .86, "Atlas localization  ·  paired patient medians"
                   if selected is None else "Inferred node activity score",
                   ha="left", va="top", fontsize=7 if compact else 9,
                   color="#555555")
    container.text(.05, .08, "UMAP color: 2nd–98th percentile" if selected != "P" else
                   "Seven paired patient medians",
                   fontsize=5.1 if compact else 6.5, color="#555555")


def draw_paired_lipid_heatmap(container, data: dict, compact: bool = False) -> None:
    """Twelve pathway × six Wu-major compartment paired-change summary."""
    frame = data["lipid_change_12_summary.csv"]
    pathways = ["Fatty_acid_activation_cytosolic",
                "Fatty_acid_activation_endoplasmic_reticular",
                "Fatty_acid_metabolism", "Fatty_acid_biosynthesis",
                "Fatty_acid_elongation_even_chain", "Fatty_acid_oxidation",
                "Carnitine_shuttle_peroxisomal", "Glycerolipid_metabolism",
                "Glycerophospholipid_metabolism", "Sphingolipid_metabolism",
                "Ether_lipid_metabolism", "Cholesterol_metabolism"]
    names = ["Fatty acid activation · cytosolic", "Fatty acid activation · ER",
             "Fatty acid metabolism", "Fatty acid biosynthesis",
             "Fatty acid elongation", "Fatty acid oxidation",
             "Peroxisomal carnitine shuttle", "Glycerolipid metabolism",
             "Glycerophospholipid metabolism", "Sphingolipid metabolism",
             "Ether lipid metabolism", "Cholesterol metabolism"]
    celltypes = ["CAF", "Endothelial", "Epithelial", "Myeloid", "PVL", "T cells"]
    lookup = frame.set_index(["pathway", "celltype"])
    assert lookup.index.is_unique and len(lookup) == 72
    values = np.array([[lookup.loc[(pathway, ct), "median_paired_log2_ratio"]
                        for ct in celltypes] for pathway in pathways])
    counts = np.array([[lookup.loc[(pathway, ct), "patients_in_median_direction"]
                        for ct in celltypes] for pathway in pathways])
    limit = max(.1, np.ceil(np.max(np.abs(values))*20)/20)
    ax = container.add_axes([.32, .13, .56, .70])
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(12), names)
    ax.set_xticks(range(6), celltypes)
    ax.tick_params(length=0, labelsize=5.6 if compact else 7)
    for index in (0, 1, 5, 6):
        ax.get_yticklabels()[index].set_fontweight("bold")
    for i in range(12):
        for j in range(6):
            count = int(counts[i, j])
            direction = "↑" if values[i, j] > 0 else "↓"
            nominal_p = float(lookup.loc[(pathways[i], celltypes[j]),
                                         "paired_wilcoxon_two_sided_nominal_p"])
            star = "*" if nominal_p < .05 else ""
            ax.text(j, i, f"{count}{direction}{star}", ha="center", va="center",
                    fontsize=6 if compact else 6.8,
                    fontweight="bold" if count >= 6 else "normal",
                    alpha=1 if count >= 6 else .72,
                    color="white" if abs(values[i, j]) > .6*limit else "#202020")
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for boundary in (1.5, 6.5):
        ax.axhline(boundary, color="#555555", linewidth=.7, zorder=5)
    for row, column in ((0, 2), (1, 2), (0, 3), (1, 3), (5, 3), (6, 3)):
        ax.add_patch(Rectangle((column-.46, row-.46), .92, .92,
                               fill=False, edgecolor="#252525", linewidth=1,
                               zorder=6))
    ax.spines[:].set_visible(False)
    cb_ax = container.add_axes([.91, .20, .018, .56])
    colorbar = container.colorbar(image, cax=cb_ax)
    colorbar.set_label("Median paired log₂(Surg./Bx)",
                       fontsize=6 if compact else 8)
    colorbar.ax.tick_params(labelsize=5.5 if compact else 7, length=2)
    container.text(.025, .99, "T  Paired lipid pathway activity changes across cell types",
                   ha="left", va="top", fontsize=7.2 if compact else 9,
                   fontweight="bold")
    container.text(.025, .93, "Red ↑ higher at surgery · blue ↓ lower · number = patients of seven · * nominal paired P<0.05",
                   ha="left", va="top", fontsize=6 if compact else 7,
                   color="#555555")
    container.text(.025, .875, "Outlined: epithelial activation (L–M); myeloid activation, oxidation and shuttle (N)",
                   ha="left", va="top", fontsize=6 if compact else 7,
                   color="#555555")


def draw_carnitine_node(container, label: str, data: dict,
                        with_legend: bool = False, compact: bool = False,
                        display_letter: str = "") -> None:
    titles = {"K": "Myeloid L-carnitine", "L": "Myeloid L-octanoylcarnitine",
              "M": "Myeloid octanoyl-CoA"}
    stat = data["carnitine_panel_summary.csv"].set_index("panel").loc[label]
    frame = data["carnitine_patient_values.csv"].loc[
        lambda x: x.panel.eq(label)].set_index("PatientID").loc[list(PATIENTS)]
    ax = container.subplots(1, 1)
    patient_medians = [frame.biopsy_median.to_numpy(), frame.surgery_median.to_numpy()]
    boxes = ax.boxplot(patient_medians, positions=[0, 1],
                       widths=.26, patch_artist=True, showfliers=False, whis=(0, 100),
                       medianprops={"color": "#333333", "linewidth": 1.2},
                       whiskerprops={"color": "#777777", "linewidth": .8},
                       capprops={"color": "#777777", "linewidth": .8},
                       boxprops={"edgecolor": "#666666", "linewidth": .8}, zorder=2)
    for patch, timepoint in zip(boxes["boxes"], ("Bx", "Surgery")):
        patch.set_facecolor(TIME_COLORS[timepoint]); patch.set_alpha(.2)
    for i, patient in enumerate(PATIENTS):
        row = frame.loc[patient]
        is_p3 = patient == PATIENTS[2]
        color = RECURRENCE_COLOR if is_p3 else "#AAB4BE"
        x = [np.linspace(-.06, .06, 7)[i], 1 + np.linspace(-.06, .06, 7)[i]]
        ax.plot(x, [row.biopsy_median, row.surgery_median], color=color,
                lw=1.5 if is_p3 else .75, zorder=2)
        ax.scatter(x, [row.biopsy_median, row.surgery_median], s=23 if is_p3 else 16,
                   color=color, edgecolors="white", linewidths=.4, zorder=3)
    vals = frame[["biopsy_median", "surgery_median"]].to_numpy()
    low, high = float(vals.min()), float(vals.max())
    span = high - low
    ax.set_ylim(low - .06*span, high + .3*span)
    ax.set_xlim(-.25, 1.25)
    ax.set_xticks([0, 1], ["Bx", "Surg."] if compact else ["Biopsy", "Surgery"])
    if not compact or label == "L":
        ax.set_ylabel("Node score" if compact else "Inferred scKNIFE node activity")
        if compact:
            ax.yaxis.labelpad = 0
            ax.yaxis.label.set_fontsize(7)
    if not compact:
        ax.text(.98, .97, f"{int(stat.patients_lower_at_surgery)}/7 lower; nominal p={stat.paired_wilcoxon_two_sided_nominal_p:.4f}\n"
                f"focused BH q={stat.focused_60_test_BH_q:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .94, "pad": 2})
    container.text(.025, .985, f"{display_letter or label}  {COMPACT_TITLES[label] if compact else titles[label]}",
                   ha="left", va="top", fontsize=7 if compact else 8, fontweight="bold")
    if not compact:
        container.text(.025, .88, "Seven patient medians per timepoint", ha="left", va="top",
                       fontsize=6.5, color="#555555")
    container.subplots_adjust(left=(.40 if label == "K" else .32) if compact else .22, right=.97,
                              top=.84 if compact else .74, bottom=.24 if compact else .25)
    if with_legend:
        container.text(.52, .06, "Orange = P3", ha="center", fontsize=6,
                       color=RECURRENCE_COLOR)


def draw_carnitine_graph(container, data: dict, display_letter: str = "M") -> None:
    """Draw the L-carnitine-centered subset of the frozen scKNIFE graph."""
    graph = data["carnitine_graph_subset.json"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert len(nodes) == 24 and len(graph["edges"]) == 76
    assert graph["center"] == "L_carnitine"
    assert {node["kind"] for node in nodes.values()} == {
        "pathway", "reaction", "gene", "metabolite"}
    reaction_by_id = {nodes[node]["original_id"]: node for node in graph["reactions"]}
    reaction_positions = {
        "MAR02894": (.36, .74), "MAR02780": (.50, .76),
        "MAR03034": (.64, .74), "MAR03033": (.73, .59),
        "MAR03035": (.64, .39), "MAR03037": (.50, .34),
        "MAR03029": (.36, .39), "MAR03027": (.27, .59),
    }
    pos = {
        graph["center"]: (.50, .55),
        graph["pathway"]: (.48, .96),
        "Carnitine_shuttle_cytosolic": (.16, .96),
        "Carnitine_shuttle_endoplasmic_reticular": (.74, .96),
        "Carnitine_shuttle_mitochondrial": (.94, .96),
        "CPT1A": (.07, .81), "CPT1B": (.07, .65),
        "CPT1C": (.07, .49), "CPT2": (.07, .33),
        "CROT": (.93, .81), "SLC25A20": (.93, .65),
        "SLC25A29": (.93, .49), "CRAT": (.93, .33),
        "CoA": (.20, .12), "L_octanoylcarnitine": (.67, .12),
        "octanoyl_CoA": (.85, .12),
    }
    pos.update({reaction_by_id[reaction]: xy
                for reaction, xy in reaction_positions.items()})
    assert set(pos) == set(nodes)
    ax = container.subplots(1, 1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    edge_style = {
        "gene-metabolite": ("#C8CED2", .55, .30, ":"),
        "gene-reaction": ("#6B8C9C", .90, .57, "-"),
        "pathway-reaction": ("#92718C", 1.05, .56, "-"),
        "metabolite-reaction": ("#BE954D", 1.12, .70, "-"),
    }
    for kind in edge_style:
        for edge in graph["edges"]:
            if edge["kind"] != kind:
                continue
            a, b = pos[edge["source"]], pos[edge["target"]]
            color, width, alpha, style = edge_style[kind]
            if kind == "metabolite-reaction" and graph["center"] in (edge["source"], edge["target"]):
                color, width, alpha, style = "#A56A16", 1.8, .93, "-"
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
                    linewidth=width, alpha=alpha, linestyle=style, zorder=1)
    for node, (x, y) in pos.items():
        info = nodes[node]
        kind = info["kind"]
        if kind == "pathway":
            label = {"Carnitine_shuttle_peroxisomal": "Peroxisomal shuttle",
                     "Carnitine_shuttle_cytosolic": "Cytosolic shuttle",
                     "Carnitine_shuttle_endoplasmic_reticular": "ER shuttle",
                     "Carnitine_shuttle_mitochondrial": "Mitochondrial shuttle"}[node]
            color, edge_color, size = "#E5C8DB", "#8D6C87", 6.8
        elif kind == "reaction":
            label = info["original_id"]
            color, edge_color, size = "#C5DDEB", "#7895A4", 6.9
        elif kind == "gene":
            label = node
            color, edge_color, size = "#E3E7EB", "#8D969D", 7
        else:
            label = node.replace("_", "-")
            color = "#E9B85B" if node == graph["center"] else "#F3D58D" if node != "CoA" else "#E9E9E9"
            edge_color, size = "#9D6F28", 9 if node == graph["center"] else 7.2
        ax.text(x, y, label, ha="center", va="center", fontsize=size,
                fontweight="bold" if kind in ("pathway", "reaction") or
                node in ("L_carnitine", "L_octanoylcarnitine", "octanoyl_CoA")
                else "normal",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": color,
                      "edgecolor": edge_color, "linewidth": .8}, zorder=5)
    container.text(.02, .98, f"{display_letter}  L-carnitine-centered scKNIFE graph subset",
                   ha="left", va="top", fontsize=9, fontweight="bold")
    container.text(.02, .92, "Eight direct L-carnitine reactions in the peroxisomal shuttle; surrounding pathways, genes, and selected metabolites.",
                   ha="left", va="top", fontsize=7, color="#555555")
    key = [Line2D([0], [0], color=edge_style[k][0], linewidth=1.5,
                  linestyle=edge_style[k][3], label=label)
           for k, label in (("pathway-reaction", "Pathway–reaction"),
                            ("metabolite-reaction", "Reaction–metabolite"),
                            ("gene-reaction", "Reaction–gene"),
                            ("gene-metabolite", "Gene–metabolite"))]
    container.legend(handles=key, loc="lower center", bbox_to_anchor=(.5, .076),
                     ncol=4, frameon=False, fontsize=6.4)
    container.text(.5, .025,
                   "All 76 edges among 24 selected nodes are retained. Dark spokes link L-carnitine directly to reactions; lines do not imply flux direction.",
                   ha="center", va="bottom", fontsize=6.5, color="#555555")
    container.subplots_adjust(left=.03, right=.97, top=.865, bottom=.18)



def draw_lipid_pathway_summary(container, data: dict, display_letter: str = "M",
                               compact: bool = False) -> None:
    """Spectra-style surgery specificity across seven patients and six cell types."""
    celltypes = ["CAF", "Endothelial", "Epithelial", "Myeloid", "PVL", "T cells"]
    frame = data["lipid_pathway_specificity.csv"]
    assert len(frame) == 42 and set(frame.PatientID) == set(PATIENTS)
    assert set(frame.celltype) == set(celltypes)
    assert frame.pathways_averaged.eq(10).all() and frame.timepoint.eq("Surgery").all()
    lookup = frame.set_index(["PatientID", "celltype"])
    assert lookup.index.is_unique
    values = np.array([[lookup.loc[(patient, ct), "mean_pathway_z"]
                        for ct in celltypes] for patient in PATIENTS], dtype=float)
    assert values.shape == (7, 6) and np.isfinite(values).all()
    assert np.allclose(values.mean(axis=1), 0, atol=1e-10)
    means = values.mean(axis=0)
    plot_left, plot_width = (.34, .40) if compact else (.31, .43)
    plot_bottom, plot_height = (.32, .58) if compact else (.34, .58)
    ax = container.add_axes([plot_left, plot_bottom, plot_width, plot_height])
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(PATIENTS)), [f"P{i + 1}" for i in range(len(PATIENTS))])
    ax.set_xticks([])
    ax.tick_params(length=0, labelsize=7)
    ax.set_xticks(np.arange(-.5, len(celltypes), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(PATIENTS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bottom_ax = container.add_axes([plot_left, plot_bottom - .15, plot_width, .13])
    bottom_ax.bar(np.arange(len(celltypes)), means, width=.7, color="#202020")
    bottom_ax.axhline(0, color="#666666", linewidth=.5)
    bottom_ax.set_xlim(-.5, len(celltypes) - .5)
    bottom_limit = max(.6, float(np.ceil(np.max(np.abs(means)) * 10) / 10))
    bottom_ax.set_ylim(-bottom_limit, bottom_limit)
    bottom_ax.set_yticks([-.5, 0, .5])
    bottom_ax.set_xticks(range(len(celltypes)), celltypes)
    bottom_ax.tick_params(axis="both", labelsize=5.8, length=1.5, pad=1)
    for tick in bottom_ax.get_xticklabels():
        tick.set_rotation(90)
        tick.set_ha("center")
        tick.set_va("top")
    bottom_ax.grid(False)
    for spine in bottom_ax.spines.values():
        spine.set_visible(False)

    cax = container.add_axes([plot_left + plot_width + .025,
                              .105 if compact else .12, .17, .022])
    colorbar = container.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_label("Mean of 10 pathway z-scores", fontsize=6.2, labelpad=2)
    colorbar.ax.tick_params(labelsize=5.8, length=1.5, pad=1)
    container.text(plot_left, .965,
                   f"{display_letter}  Lipid pathway cell-type specificity (surgery)",
                   ha="left", va="top", fontsize=8, fontweight="bold")
    container.text(plot_left - .07, plot_bottom - .02, "Mean z-score",
                   ha="right", va="top", fontsize=6.2)


def _set_uniform_font_size(fig):
    """Apply the six-point manuscript typography after all artists exist."""
    fig.canvas.draw()
    for artist in fig.findobj(match=mpl.text.Text):
        artist.set_fontsize(FIGURE_FONT_SIZE)
    return fig


def make_panel(label: str, data: dict[str, pd.DataFrame]):
    return _set_uniform_font_size(_make_panel(label, data))


def _make_panel(label: str, data: dict[str, pd.DataFrame]):
    source = DISPLAY_TO_SOURCE[label]
    if label in "AB":
        return legacy.make_figure6_main_panel(label, data)
    if label == "C":
        fig = plt.figure(figsize=(3.9, 3.5), facecolor="white")
        draw_single_concordance(fig, data, "C", "Task")
        return fig
    if label in ("D", "H"):
        fig = plt.figure(figsize=(3.5, 3.5), facecolor="white")
        draw_node(fig, source, data["fig6_main_patient_values.csv"],
                  data["fig6_main_patient_stats.csv"], with_legend=True,
                  display_letter=label)
        return fig
    if label in ("I", "J", "K"):
        fig = plt.figure(figsize=(3.5, 3.5), facecolor="white")
        selected = {"I": "3_sulfinoalanine", "J": "S_2_aminobutanoate", "K": "methylglyoxal"}[label]
        draw_ik_paired(fig, source, data["fig6_main_patient_values.csv"],
                       data["fig6_main_patient_stats.csv"], selected_node=selected)
        return fig
    if label in ("E", "F", "G"):
        fig = plt.figure(figsize=(3.5, 3.5), facecolor="white")
        args = (data["fig6_main_patient_values.csv"],
                data["fig6_main_patient_stats.csv"])
        selected = {"E": "adenine", "F": "guanine", "G": "deoxyinosine"}[label]
        draw_purine_boxplots(fig, *args, selected_node=selected)
        return fig
    if label in "LMN":
        fig = plt.figure(figsize=(3.5, 3.5), facecolor="white")
        draw_lipid_pair(fig, label, data)
        return fig
    if label in "OP":
        fig = plt.figure(figsize=(5, 3.5), facecolor="white")
        draw_carnitine_composite(fig, data, selected=label)
        return fig
    if label in "QRS":
        fig = plt.figure(figsize=(3.5, 3.5), facecolor="white")
        draw_p3_lipid_violins(fig, data, selected=label)
        return fig
    if label == "T":
        fig = plt.figure(figsize=(8.3, 5), facecolor="white")
        draw_paired_lipid_heatmap(fig, data)
        return fig
    if source in "HIJ":
        fig = plt.figure(figsize=(3.2, 3.5), facecolor="white")
        draw_recurrence(fig, source, data, display_letter=label)
        return fig
    if source in "KLM":
        fig = plt.figure(figsize=(3.2, 3.5), facecolor="white")
        draw_carnitine_node(fig, source, data, with_legend=True,
                            display_letter=label)
        return fig
    if source == "lipid_summary":
        fig = plt.figure(figsize=(11.7, 5.8), facecolor="white")
        draw_lipid_pathway_summary(fig, data, display_letter=label)
        return fig
    raise ValueError(label)


def make_compiled(data: dict[str, pd.DataFrame]):
    fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4), facecolor="white")
    rows = fig.subfigures(6, 1, height_ratios=[1.15, .91, .97, 1.32, 1.46, .16], hspace=.035)
    top = rows[0].subfigures(1, 3, width_ratios=[1, 1, 1], wspace=.045)
    legacy._draw_four_set_upset(top[0], data, "A", "treatment_four_set_membership.csv",
                                "Treatment four-set metabolic overlap")
    legacy._draw_treatment_pathway_scatter(top[1], data, "B")
    draw_single_concordance(top[2], data, "C", "Task")
    top[0]._suptitle.set_text("A  Treatment overlap")
    top[1]._suptitle.set_text("B  Treatment pathways")
    top[0].axes[1].set_yticklabels(["MEIA-Task", "MEIA-Path", "GPR-GPA", "LC-MS"])
    for label in top[0].texts:
        if label is not top[0]._suptitle:
            label.set_text("21 LC–MS terms: 18 mapped → 14 targets\n3 unmapped; MEIA: 218 tasks + 138 pathways")
            label.set_fontsize(5.6)
    for label in top[1].texts:
        if label is not top[1]._suptitle:
            label.set_text("21 LC–MS treatment candidates\n5 pass Holm-adjusted Gamma P")
            label.set_fontsize(6)
    for label in top[2].texts:
        if label.get_text().startswith("C  Task score concordance"):
            label.set_text("C  Task concordance")
    pairs, stats = data["fig6_main_patient_values.csv"], data["fig6_main_patient_stats.csv"]
    for row_index, displays in ((1, "DEFGHI"), (2, "JKLMN")):
        response_row = rows[row_index].subfigures(1, len(displays), wspace=.025)
        for subfig, display in zip(response_row, displays):
            source = DISPLAY_TO_SOURCE[display]
            if display in ("D", "H"):
                draw_node(subfig, source, pairs, stats, compact=True,
                          display_letter=display)
            elif display in "EFG":
                selected = {"E": "adenine", "F": "guanine",
                            "G": "deoxyinosine"}[display]
                draw_purine_boxplots(subfig, pairs, stats, compact=True,
                                     selected_node=selected)
            elif display in "IJK":
                selected = {"I": "3_sulfinoalanine", "J": "S_2_aminobutanoate",
                            "K": "methylglyoxal"}[display]
                draw_ik_paired(subfig, source, pairs, stats, compact=True,
                               selected_node=selected)
            else:
                draw_lipid_pair(subfig, display, data, compact=True)
    handles = [Line2D([0], [0], color=PATIENT_COLORS[i], marker="o", markersize=4,
                      linewidth=1, label=f"P{i+1}") for i in range(len(PATIENTS))]
    lipid_row = rows[3].subfigures(1, 5,
                                  width_ratios=[1.25, .95, 1.1, 1.1, 1.1],
                                  wspace=.045)
    for subfig, label in zip(lipid_row, "OPQRS"):
        if label in "OP":
            draw_carnitine_composite(subfig, data, compact=True, selected=label)
        else:
            draw_p3_lipid_violins(subfig, data, compact=True, selected=label)
    draw_paired_lipid_heatmap(rows[4], data, compact=True)
    legend_ax = rows[5].subplots(1, 1)
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", bbox_to_anchor=(.5, .5),
                     ncol=7, frameon=False, fontsize=6.5)
    return _set_uniform_font_size(fig)


export = restored.export


def export_compiled(fig, stem: Path) -> None:
    """Export the compiled figure on an uncropped A4 page."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix("." + extension), dpi=300,
                    facecolor="white", bbox_inches=None, pad_inches=0)
