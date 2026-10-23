#!/usr/bin/env python3
"""Legacy and shared S6 plotting helpers; active A–J layout is in companion.py."""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

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


def _validated_integrated_pathway_scatter(data,scatter_file,enrichment_file,membership_file,contrast):
    table=data[scatter_file].copy().sort_values("rank")
    enrichment=data[enrichment_file].copy()
    membership=data[membership_file].copy()
    scatter_required={"rank","pathway","enrichment_factor","neg_log10_p","radius","label_top5","coordinate_source"}
    enrichment_required={"candidate_pathway","pathway_BH_FDR_gamma","contrast"}
    membership_required={"candidate_pathway","mapped_to_humangem","human_gem_pathway"}
    if not scatter_required.issubset(table.columns):raise RuntimeError(f"{scatter_file} schema changed")
    if not enrichment_required.issubset(enrichment.columns):raise RuntimeError(f"{enrichment_file} schema changed")
    if set(membership.columns)!=membership_required:raise RuntimeError(f"{membership_file} schema changed")
    expected_rows=len(enrichment)
    if expected_rows<1 or any(len(x)!=expected_rows or x.nunique()!=expected_rows for x in (table["pathway"],enrichment["candidate_pathway"],membership["candidate_pathway"])):raise RuntimeError(f"{contrast} pathway rows are missing or duplicated")
    scatter_paths=set(table["pathway"].astype(str));enrichment_paths=set(enrichment["candidate_pathway"].astype(str));upset_paths=set(membership["candidate_pathway"].astype(str))
    if scatter_paths!=enrichment_paths or scatter_paths!=upset_paths:
        raise RuntimeError(f"{contrast} Panel B and UpSet pathway sets differ: scatter={len(scatter_paths)}, enrichment={len(enrichment_paths)}, upset={len(upset_paths)}")
    if set(enrichment["contrast"].astype(str))!={contrast}:raise RuntimeError(f"{enrichment_file} contrast changed")
    if set(table.loc[table["label_top5"],"rank"].astype(int))!=set(range(1,6)):raise RuntimeError(f"{scatter_file} top-five labels changed")
    discoveries=int(enrichment["pathway_BH_FDR_gamma"].lt(.05).sum())
    return table,discoveries


def _draw_treatment_pathway_scatter(container, data: Path, label: str = "B") -> None:
    table,discoveries=_validated_integrated_pathway_scatter(data,"surgery_pathway_scatter.csv","surgery_pathway_enrichment.csv","treatment_four_set_membership.csv","surgery_vs_biopsy")
    radius=table["radius"].to_numpy(dtype=float); rmin,rmax=radius.min(),radius.max()
    sizes=24+96*(radius-rmin)/(rmax-rmin)
    cmap=LinearSegmentedColormap.from_list("pathway_yellow_red",["#FFF200","#FFA000","#FF1300"])
    ax=container.subplots(1,1)
    nominal_cutoff_y = -np.log10(NOMINAL_P_CUTOFF)
    ax.axhline(nominal_cutoff_y, color=COLORS["gray"], linestyle="--", linewidth=LINE_WIDTH, alpha=.8, zorder=1)
    ax.text(.02, nominal_cutoff_y, r"Nominal $p = 0.05$", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=TICK_LABEL_SIZE_PT, color=COLORS["gray"])
    scatter=ax.scatter(table["enrichment_factor"],table["neg_log10_p"],s=sizes,c=table["neg_log10_p"],cmap=cmap,edgecolor="white",linewidth=.45,alpha=.92,zorder=3)
    short={"Selenoamino acid metabolism":"Selenoamino acids","Purine metabolism":"Purines","Methionine and cysteine metabolism":"Methionine/cysteine","Aspartate and asparagine metabolism":"Aspartate/asparagine","Glycolysis and Gluconeogenesis":"Glycolysis"}
    top5=table.loc[table["label_top5"]]
    for row in top5.itertuples(index=False):
        ax.annotate(str(int(row.rank)),(row.enrichment_factor,row.neg_log10_p),xytext=(5,5),textcoords="offset points",fontsize=6,fontweight="bold",color=COLORS["dark"])
    ax.text(.02,.98,"\n".join(f"{int(row.rank)}  {short.get(row.pathway,row.pathway)}" for row in top5.itertuples(index=False)),transform=ax.transAxes,ha="left",va="top",fontsize=5.2,linespacing=1.35,color=COLORS["dark"],bbox={"facecolor":"white","edgecolor":"none","alpha":.78})
    ax.set_xlabel("Enrichment factor"); ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_xlim(0,max(1.86,table["enrichment_factor"].max()*1.06)); ax.set_ylim(-.03,table["neg_log10_p"].max()*1.10)
    _clean_axis(ax); ax.grid(False,which="both")
    colorbar=container.colorbar(scatter,ax=ax,fraction=.045,pad=.035); colorbar.set_label(r"$-\log_{10}(p)$",fontsize=TICK_LABEL_SIZE_PT); colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE_PT); colorbar.solids.set_rasterized(False)
    legend_values=[.5,.9,1.3]
    handles=[ax.scatter([],[],s=24+96*(value-rmin)/(rmax-rmin),facecolor="#F4A300",edgecolor="white",linewidth=.45) for value in legend_values]
    ax.legend(handles,[f"{value:.1f}" for value in legend_values],title=r"$\sqrt{\mathrm{enrichment}}$",frameon=False,fontsize=5.5,title_fontsize=5.5,loc="lower right",borderaxespad=.2,labelspacing=.7)
    _panel_title(container,label,"Treatment-associated pathway enrichment",f"{len(table)} candidates shared with A; top five labeled.\n{discoveries} pass Holm-adjusted Gamma P < 0.05.")
    container.subplots_adjust(left=.13,right=.84,top=.78,bottom=.17)


def _draw_pathway_panel(container, data: Path, label: str, contrast: str) -> None:
    filename = "surgery_pathway_enrichment.csv" if contrast == "surgery_vs_biopsy" else "status_pathway_enrichment.csv"
    table = data[filename].copy().sort_values(
        ["permutation_adjusted_gamma_p", "candidate_pathway"]
    ).head(TOP_PATHWAYS).sort_values("permutation_adjusted_gamma_p", ascending=False)
    values = -np.log10(table["permutation_adjusted_gamma_p"].clip(lower=np.finfo(float).tiny))
    sizes = 25 + 8 * table["Hits.sig"].astype(float)
    y = np.arange(len(table))
    ax = container.subplots(1, 1)
    scatter = ax.scatter(
        values, y, s=sizes, c=table["permutation_adjusted_gamma_p"],
        cmap="viridis_r", edgecolor="white", linewidth=0.55, zorder=3,
    )
    ax.set_yticks(y, _wrap(table["candidate_pathway"], 33))
    ax.set_xlabel(r"Candidate pathway score, $-\log_{10}(P_{\mathrm{gamma}})$")
    ax.set_ylabel("")
    ax.set_xlim(left=0)
    _clean_axis(ax)
    colorbar = container.colorbar(scatter, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label(r"Permutation-adjusted $P_{\mathrm{gamma}}$", fontsize=TICK_LABEL_SIZE_PT)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE_PT)
    total = len(data[filename])
    title = "Treatment pathway enrichment" if contrast == "surgery_vs_biopsy" else "Metastatic-status pathway enrichment"
    subtitle = (
        f"Top {TOP_PATHWAYS} integrated Mummichog candidates;\n"
        f"{int(data[filename].pathway_BH_FDR_gamma.lt(.05).sum())}/{total} pathways pass Holm-adjusted Gamma P < 0.05."
        if contrast == "surgery_vs_biopsy"
        else f"Top {TOP_PATHWAYS} integrated Mummichog candidates; {int(data[filename].pathway_BH_FDR_gamma.lt(.05).sum())}/{total} pathways pass Holm-adjusted Gamma P < 0.05."
    )
    if contrast == "surgery_vs_biopsy":
        _panel_title(container, label, title)
        ax.text(
            0.0, 1.02, f"Top {TOP_PATHWAYS} candidates; {int(data[filename].pathway_BH_FDR_gamma.lt(.05).sum())}/{total} pass Holm-adjusted Gamma P",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=TICK_LABEL_SIZE_PT, color=COLORS["gray"],
        )
    else:
        _panel_title(container, label, title, subtitle)
    left_margin = 0.34 if container.get_figure().get_figwidth() > 5.5 else 0.42
    top_margin = 0.79 if contrast == "surgery_vs_biopsy" else 0.86
    container.subplots_adjust(left=left_margin, right=0.90, top=top_margin, bottom=0.17)


def _paired_metabolite_axis(ax: plt.Axes, values: pd.DataFrame, stats: pd.Series, show_ylabel: bool) -> None:
    pivot = values.pivot(index="PatientID_clean", columns="Timepoint", values="log2_intensity")
    for xpos, (timepoint, color) in enumerate([
        ("Biopsy", COLORS["biopsy"]), ("Surgery", COLORS["surgery"]),
    ]):
        ax.boxplot(
            [pivot[timepoint].dropna().to_numpy()], positions=[xpos], widths=BOX_WIDTH,
            patch_artist=True, showfliers=False, whis=1.5, zorder=0,
            boxprops={"facecolor": mpl.colors.to_rgba(color, BOX_ALPHA), "edgecolor": color, "linewidth": LINE_WIDTH},
            medianprops={"color": COLORS["dark"], "linewidth": LINE_WIDTH + 0.4},
            whiskerprops={"color": color, "linewidth": LINE_WIDTH},
            capprops={"color": color, "linewidth": LINE_WIDTH},
        )
    for _, row in pivot.iterrows():
        ax.plot([0, 1], [row["Biopsy"], row["Surgery"]], color=COLORS["light"], lw=LINE_WIDTH, zorder=1)
        ax.scatter(0, row["Biopsy"], s=POINT_SIZE, color=COLORS["biopsy"], edgecolor="white", linewidth=0.45, zorder=2)
        ax.scatter(1, row["Surgery"], s=POINT_SIZE, color=COLORS["surgery"], edgecolor="white", linewidth=0.45, zorder=2)
    ax.set_xticks([0, 1], ["Biopsy", "Surgery"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_title(
        f"{stats.name}\n"
        f"log2FC={stats.log2_fold_change:+.2f}; FDR={stats.joint_BH_FDR:.3f}",
        fontsize=AXIS_TITLE_SIZE_PT, pad=5,
    )
    ax.set_ylabel("Specimen log2 intensity" if show_ylabel else "")
    _clean_axis(ax)


def _draw_treatment_metabolites(container, data: Path, label: str = "C") -> None:
    stats = data["selected_metabolite_statistics.csv"].copy()
    values = data["selected_metabolite_values.csv"].copy()
    stats = stats.loc[stats["panel"] == label].set_index("display_name")
    values = values.loc[values["panel"] == label]
    order = ["Selenoamino acid metabolism\nm/z 174.9854 • RT 1689.69 s", "Aspartate/asparagine metabolism\nm/z 250.0920 • RT 262.79 s", "Glycolysis/gluconeogenesis\nm/z 246.9429 • RT 242.71 s"]
    axes = container.subplots(1, 3)
    for index, (ax, name) in enumerate(zip(axes, order)):
        _paired_metabolite_axis(ax, values.loc[values["display_name"] == name], stats.loc[name], index == 0)
    _panel_title(
        container, label, "Treatment pathway-driving LC–MS features",
        "Seven paired patients; blank-corrected pathway-driving features.",
    )
    container.text(
        0.99, 0.02,
        "Observed biopsy-to-surgery changes; not a causal treatment estimate.",
        ha="right", va="bottom", fontsize=TICK_LABEL_SIZE_PT, color=COLORS["gray"],
    )
    container.subplots_adjust(left=0.08, right=0.99, top=0.70, bottom=0.18, wspace=0.32)


def _draw_pairwise_lcms_upsets(container, summary: pd.Series, label: str) -> None:
    total = int(summary["lcms_source_pathways"])
    level1_shared = int(summary["lcms_level1_only"] + summary["lcms_level1_level2"])
    level2_shared = int(summary["lcms_level2_only"] + summary["lcms_level1_level2"])
    comparisons = [
        ("LC-MS vs Level 1", "Level 1", level1_shared, COLORS["gpa_specific"]),
        ("LC-MS vs Level 2", "Level 2", level2_shared, COLORS["scKNIFE"]),
    ]
    if total != 54 or level1_shared != 48 or level2_shared != 28:
        raise RuntimeError(
            f"Panel A requires LC-MS/Level 1/Level 2 counts 54/48/28; found {total}/{level1_shared}/{level2_shared}"
        )
    outer = container.add_gridspec(
        1, 2, left=0.13, right=0.99, top=0.78, bottom=0.16, wspace=0.42
    )
    for index, (comparison_title, level_label, shared, shared_color) in enumerate(comparisons):
        unmatched = total - shared
        pair = outer[index].subgridspec(2, 1, height_ratios=[0.64, 0.36], hspace=0.04)
        ax_top = container.add_subplot(pair[0])
        ax_matrix = container.add_subplot(pair[1])
        x = np.arange(2)
        counts = [shared, unmatched]
        ax_top.bar(x, counts, color=[shared_color, COLORS["light"]], width=UPSET_BAR_WIDTH)
        ax_top.text(
            0, shared + total * 0.025, f"{shared}/{total} matched ({100 * shared / total:.1f}%)",
            ha="center", va="bottom", fontsize=TICK_LABEL_SIZE_PT, fontweight="bold",
        )
        ax_top.text(
            1, unmatched + total * 0.025, str(unmatched), ha="center", va="bottom",
            fontsize=TICK_LABEL_SIZE_PT, fontweight="bold",
        )
        ax_top.set_title(comparison_title, fontsize=AXIS_TITLE_SIZE_PT, fontweight="bold", pad=3)
        ax_top.set_ylabel("Intersection size")
        ax_top.set_xticks([])
        ax_top.set_ylim(0, total * 1.22)
        _clean_axis(ax_top)
        rows = np.array([1, 0])
        memberships = [(True, True), (True, False)]
        for xpos, membership in enumerate(memberships):
            ax_matrix.scatter(
                np.repeat(xpos, 2), rows, s=UPSET_EMPTY_DOT_SIZE, facecolor="white",
                edgecolor=COLORS["light"], linewidth=LINE_WIDTH, zorder=1,
            )
            active_rows = rows[np.asarray(membership, dtype=bool)]
            ax_matrix.scatter(
                np.repeat(xpos, len(active_rows)), active_rows, s=UPSET_DOT_SIZE,
                color=shared_color if xpos == 0 else COLORS["gray"], zorder=3,
            )
            if len(active_rows) > 1:
                ax_matrix.plot(
                    [xpos, xpos], [active_rows.min(), active_rows.max()],
                    color=shared_color, lw=UPSET_CONNECTOR_WIDTH, zorder=2,
                )
        ax_matrix.set_xlim(-0.55, 1.55)
        ax_matrix.set_ylim(-0.5, 1.5)
        ax_matrix.set_xticks([])
        ax_matrix.set_yticks(rows, [f"LC-MS (n={total})", f"{level_label} match (n={shared})"])
        ax_matrix.tick_params(axis="y", length=0, labelsize=TICK_LABEL_SIZE_PT)
        ax_matrix.spines[:].set_visible(False)
    _panel_title(
        container, label, "LC-MS overlap with GPA evidence",
        "Pairwise LC-MS source-pathway matching; raw p < 0.05.",
    )


def _draw_three_set_upset(
    container, data: Path, label: str, summary_filename: str, title: str
) -> None:
    summary = data[summary_filename].copy().iloc[0]
    treatment_panel = "level1_pathways" in summary.index
    if treatment_panel:
        _draw_pairwise_lcms_upsets(container, summary, label)
        return
    if treatment_panel:
        level1_lcms = int(summary["lcms_level1_only"] + summary["lcms_level1_level2"])
        level2_lcms = int(summary["lcms_level2_only"] + summary["lcms_level1_level2"])
        set_specs = [
            ("LC-MS enrichment", int(summary["lcms_source_pathways"]), COLORS["lcms"]),
            ("Level 1: HG mechanistic", level1_lcms, COLORS["gpa_specific"]),
            ("Level 2: GPA-G/S + scCellFie", level2_lcms, COLORS["scKNIFE"]),
        ]
        intersection_specs = [
            ("lcms_only", (True, False, False)),
            ("lcms_level1_only", (True, True, False)),
            ("lcms_level2_only", (True, False, True)),
            ("lcms_level1_level2", (True, True, True)),
        ]
    else:
        set_specs = [
            ("LC-MS", int(summary["lcms_unique_pathways"]), COLORS["lcms"]),
            ("GPA-specific", int(summary["gpa_specific_pathways"]), COLORS["gpa_specific"]),
            ("scKNIFE", int(summary["scKNIFE_pathways"]), COLORS["scKNIFE"]),
        ]
        intersection_specs = [
            ("lcms_only", (True, False, False)),
            ("gpa_specific_only", (False, True, False)),
            ("lcms_gpa_specific_only", (True, True, False)),
            ("lcms_gpa_specific_scKNIFE", (True, True, True)),
            ("gpa_specific_scKNIFE_only", (False, True, True)),
        ]
    intersections = sorted(
        [
            (int(summary[column]), membership)
            for column, membership in intersection_specs
            if int(summary[column]) > 0
        ],
        key=lambda item: item[0],
        reverse=True,
    )

    grid = container.add_gridspec(
        2, 2, height_ratios=[0.62, 0.38], width_ratios=[0.37, 0.63],
        left=0.26, right=0.98, top=0.79,
        bottom=0.30 if treatment_panel else 0.16,
        hspace=0.08, wspace=0.08,
    )
    ax_top = container.add_subplot(grid[0, 1])
    ax_sets = container.add_subplot(grid[1, 0])
    ax_matrix = container.add_subplot(grid[1, 1])

    x = np.arange(len(intersections))
    counts = [count for count, _ in intersections]
    ax_top.bar(x, counts, color=COLORS["dark"], width=UPSET_BAR_WIDTH)
    for xpos, count in zip(x, counts):
        ax_top.text(
            xpos, count + max(max(counts) * 0.025, 0.8), str(count),
            ha="center", va="bottom", fontweight="bold",
        )
    ax_top.set_ylabel("Intersection size")
    ax_top.set_xticks([])
    ax_top.set_ylim(0, max(counts) * 1.18)
    _clean_axis(ax_top)

    rows = np.array([2, 1, 0])
    ax_sets.barh(
        rows, [size for _, size, _ in set_specs],
        color=[color for _, _, color in set_specs], height=UPSET_SET_BAR_HEIGHT,
    )
    ax_sets.set_yticks(rows, [label_ for label_, _, _ in set_specs])
    ax_sets.invert_xaxis()
    ax_sets.set_xlabel("Set size")
    _clean_axis(ax_sets)

    for xpos, (_, membership) in enumerate(intersections):
        ax_matrix.scatter(
            np.repeat(xpos, len(rows)), rows, s=UPSET_EMPTY_DOT_SIZE,
            facecolor="white", edgecolor=COLORS["light"], linewidth=LINE_WIDTH, zorder=1,
        )
        active_rows = rows[np.asarray(membership, dtype=bool)]
        ax_matrix.scatter(
            np.repeat(xpos, len(active_rows)), active_rows, s=UPSET_DOT_SIZE,
            color=COLORS["dark"], zorder=3,
        )
        if len(active_rows) > 1:
            ax_matrix.plot(
                [xpos, xpos], [active_rows.min(), active_rows.max()],
                color=COLORS["dark"], lw=UPSET_CONNECTOR_WIDTH, zorder=2,
            )
    ax_matrix.set_xlim(-0.55, len(intersections) - 0.45)
    ax_matrix.set_ylim(-0.5, 2.5)
    ax_matrix.set_xticks([])
    ax_matrix.set_yticks([])
    ax_matrix.spines[:].set_visible(False)

    _panel_title(container, label, title, "Source-pathway overlap; raw p < 0.05, no direction filter." if treatment_panel else None)


def _paired_lcms_axis(ax: plt.Axes, values: pd.DataFrame, stats: pd.Series, show_ylabel: bool) -> None:
    wide = (
        values.pivot(index="PatientID_clean", columns="Timepoint", values="log2_intensity")
        .dropna(subset=["Biopsy", "Surgery"])
        .sort_index()
    )
    if len(wide) != 7:
        raise RuntimeError(f"Paired LC-MS panels require seven patients for {stats.name}")
    boxplot = ax.boxplot(
        [wide["Biopsy"], wide["Surgery"]], positions=[0, 1], widths=.34,
        patch_artist=True, showfliers=False,
        boxprops={"edgecolor": COLORS["dark"]}, medianprops={"color": COLORS["dark"]},
    )
    for patch, color in zip(boxplot["boxes"], ["#377EB8", "#E41A1C"]):
        patch.set_facecolor(color)
        patch.set_alpha(.72)
    for _, row in wide.iterrows():
        ax.plot(
            [0, 1], [row["Biopsy"], row["Surgery"]], color=COLORS["gray"],
            marker="o", ms=2.8, lw=.7, alpha=.75,
        )
    ax.set_xticks([0, 1], ["Biopsy", "Surgery"], rotation=18)
    ax.set_title(stats.name, loc="left", fontsize=7, fontweight="bold")
    ax.set_ylabel("LC–MS log2 intensity" if show_ylabel else "")
    ax.text(
        .02, .03,
        f"log2FC={stats.log2_fold_change:+.2f}\nq={stats.joint_BH_FDR:.3g}",
        transform=ax.transAxes, fontsize=5.8,
    )
    _clean_axis(ax)


def _draw_paired_lcms_panel(
    container, data: Path, label: str, selected: list[str], title: str, subtitle: str,
) -> None:
    stats = data["selected_metabolite_statistics.csv"].query(
        "contrast == 'surgery_vs_biopsy' and display_name in @selected"
    ).copy()
    values = data["selected_metabolite_values.csv"].query(
        "contrast == 'surgery_vs_biopsy' and display_name in @selected"
    ).copy()
    if set(stats["display_name"]) != set(selected):
        raise RuntimeError(f"Panel {label} is missing a selected paired LC-MS metabolite")
    stats = stats.set_index("display_name")
    axes = np.atleast_1d(container.subplots(1, 2)).ravel()
    for index, (ax, name) in enumerate(zip(axes, selected)):
        _paired_lcms_axis(
            ax, values.loc[values["display_name"].eq(name)], stats.loc[name], index == 0,
        )
    _panel_title(container, label, title, subtitle)
    container.subplots_adjust(left=.08, right=.995, top=.72, bottom=.22, wspace=.32)


def _draw_status_metabolites(container, data: Path, label: str = "SC") -> None:
    _draw_paired_lcms_panel(
        container, data, label, ["Retinol", "Fucose 1-phosphate"],
        "Additional paired biopsy-versus-surgery LC–MS metabolites",
        "Seven paired patients; patient-blocked Status + Timepoint model; neither candidate passes joint BH-FDR < 0.05.",
    )


def _draw_robust_cascade(container, data: Path, label: str = "D") -> None:
    table = data["robust_filtering_cascade.csv"].copy().sort_values("stage_order")
    labels = ["Tested", "Tier 1\nReplicated response", "Tier 2\nRecurrence rebound", "Tier 3\nBaseline restoration"]
    colors = ["#BDBDBD", "#56B4E9", "#E69F00", "#009E73"]
    ax = container.subplots(1, 1)
    x = np.arange(len(table))
    ax.bar(x, table["pair_count"], color=colors, width=0.68)
    offset = table["pair_count"].max() * 0.025
    for xpos, value in zip(x, table["pair_count"]):
        ax.text(xpos, value + offset, f"{int(value):,}", ha="center", va="bottom", fontsize=TICK_LABEL_SIZE_PT, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cell-type–metabolite pairs")
    ax.set_ylim(0, table["pair_count"].max() * 1.16)
    _clean_axis(ax)
    _panel_title(container, label, "Recurrence tiers")
    container.text(
        0.99, 0.02, "Cells are the statistical unit",
        ha="right", va="bottom", fontsize=TICK_LABEL_SIZE_PT, color=COLORS["gray"],
    )
    container.subplots_adjust(left=0.17, right=0.98, top=0.77, bottom=0.22)


def _draw_robust_trajectories(container,data:Path,label:str="D")->None:
    sel=data["robust_tier3_trajectory_selection.csv"].copy().sort_values("display_order")
    vals=data["robust_tier3_trajectory_values.csv.gz"].copy()
    expected=["tRNA_containing_6_isopentenyladenosine","selenocystathionine","2_phosphoglycolate"]
    if sel["metabolite"].tolist()!=expected or not sel["tier"].eq(3).all() or not sel["minor_tier"].eq(3).all():raise RuntimeError("Panel D reviewed major/minor Tier-3 selection changed")
    if set(vals["PatientID"])!={"P3"} or set(vals["Timepoint"])!={"Bx","Surgery","Recurrence"}:raise RuntimeError("Panel D requires P3 Bx-Surgery-Recurrence cells")
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
        ax.set_title(f"{c.display_name}\n{c.celltype} / {c.celltype_minor} • rank {int(c.global_rank)}",loc="left",fontsize=7,fontweight="bold",pad=2)
        ax.text(.98,.98,f"S–Bx δ={c.p4_tx_rank_biserial_effect:+.3f}, q={fmt_q(c.p4_tx_global_contrast_BH_FDR)}\nR–S δ={c.p4_rebound_rank_biserial_effect:+.3f}, q={fmt_q(c.p4_rebound_global_contrast_BH_FDR)}",transform=ax.transAxes,ha="right",va="top",fontsize=5.2,color=COLORS["dark"],bbox={"facecolor":"white","edgecolor":"none","alpha":.78,"pad":1.2})
        ax.set_xlabel("");ax.set_ylabel("Inferred scKNIFE\nmetabolite-node activity" if idx==0 else "");ax.tick_params(axis="x",labelsize=6)
        _clean_axis(ax)
    _panel_title(container,label,"Reviewed-label longitudinal restoration","P3: Bx → Surgery → Recurrence; strict major and covered-minor Tier-3 inferred nodes.")
    container.text(.99,.01,"Cell-level exploratory distributions from one recurrence patient; the six-patient cohort independently supplies Bx–Surgery discovery evidence",ha="right",va="bottom",fontsize=5.5,color=COLORS["gray"])
    container.subplots_adjust(left=.08,right=.99,top=.72,bottom=.23,wspace=.28)


MAIN_VALIDATION_METABOLITES = ("threonine",)


def _draw_selected_lcms(container,data:Path,label:str="F")->None:
    paired=data["robust_four_lcms_paired_abundance.csv"].copy()
    effects=data["robust_lcms_treatment_effects.csv"].copy()
    cells=data["robust_dhu_cf30117_t_cells.csv.gz"].copy()
    paired=paired.loc[paired["scKNIFE_metabolite"].isin(MAIN_VALIDATION_METABOLITES)].copy()
    effects=effects.loc[effects["scKNIFE_metabolite"].isin(MAIN_VALIDATION_METABOLITES)].sort_values("display_order")
    if set(effects["scKNIFE_metabolite"]) != set(MAIN_VALIDATION_METABOLITES): raise RuntimeError("Panel F is missing corrected L-threonine")
    axes=np.atleast_1d(container.subplots(1,2,width_ratios=[.82,1.18])).ravel()
    stat=next(effects.itertuples(index=False)); frame=paired.loc[paired["scKNIFE_metabolite"].eq(stat.scKNIFE_metabolite)]
    wide=frame.pivot(index="PatientID",columns="Timepoint",values="log2_intensity").sort_index(); ax=axes[0]
    boxplot=ax.boxplot([wide["Bx"],wide["Surgery"]],positions=[0,1],widths=.36,patch_artist=True,showfliers=False,boxprops={"edgecolor":COLORS["dark"]},medianprops={"color":COLORS["dark"]})
    for patch,color in zip(boxplot["boxes"],["#377EB8","#E41A1C"]): patch.set_facecolor(color); patch.set_alpha(.72)
    for _,row in wide.iterrows(): ax.plot([0,1],[row["Bx"],row["Surgery"]],color=COLORS["gray"],marker="o",ms=2.8,lw=.7,alpha=.75)
    ax.set_xticks([0,1],["Bx","Surgery"],rotation=18); ax.set_title("LC–MS • seven paired patients",loc="left",fontweight="bold",fontsize=7)
    ax.set_ylabel("LC–MS log2 intensity"); _clean_axis(ax)
    ax.text(.02,.03,f"log2FC={stat.log2_fold_change:+.2f}\nq={stat.joint_BH_FDR:.3g}",transform=ax.transAxes,fontsize=5.8)
    order=["Bx","Surgery","Recurrence"]; palette={"Bx":"#377EB8","Surgery":"#E41A1C","Recurrence":"#984EA3"}; ax=axes[1]
    sns.violinplot(data=cells,x="Timepoint",y="activity",order=order,hue="Timepoint",hue_order=order,palette=palette,legend=False,inner=None,cut=0,width=.58,linewidth=.6,ax=ax)
    sns.boxplot(data=cells,x="Timepoint",y="activity",order=order,width=.16,showfliers=False,boxprops={"facecolor":"white","alpha":.8},medianprops={"color":"black"},ax=ax)
    ax.set_title("scKNIFE • P3 T cells",loc="left",fontweight="bold",fontsize=7); ax.set_xlabel(""); ax.set_ylabel("Inferred activity"); ax.tick_params(axis="x",labelsize=6); _clean_axis(ax)
    medians=cells.groupby("Timepoint")["activity"].median().reindex(order)
    ax.text(.98,.03,"Medians: "+", ".join(f"{tp} {value:.2f}" for tp,value in medians.items()),transform=ax.transAxes,ha="right",fontsize=5.5)
    _panel_title(container,label,"L-threonine across modalities","Paired blank-corrected LC–MS response and biopsy–surgery–recurrence scKnife activity; LC–MS q=0.0535 (exploratory).")
    container.subplots_adjust(left=.07,right=.995,top=.72,bottom=.22,wspace=.34)


MAIN_CROSSMODAL_UMAP_SPECS = {
    "2_aminoethylphosphonate": "2-Aminoethylphosphonate",
    "Glutaryl_Carnitine": "Glutarylcarnitine",
    "malonate": "Malonate",
}


def _draw_crossmodal_umaps(container,data:Path,label:str="G")->None:
    table=data["robust_activity_umap.csv.gz"].copy()
    axes=np.atleast_1d(container.subplots(1,3)).ravel()
    for ax,(score_column,display_name) in zip(axes,MAIN_CROSSMODAL_UMAP_SPECS.items()):
        frame=table.sort_values(score_column,kind="stable")
        values=frame[score_column].to_numpy(dtype=float)
        vmin,vmax=np.percentile(values,[2,98])
        if not vmax>vmin: vmax=vmin+1e-6
        ax.scatter(table["UMAP1"],table["UMAP2"],c="#D9D9D9",s=.14,alpha=.42,linewidths=0,rasterized=True,zorder=1)
        points=ax.scatter(frame["UMAP1"],frame["UMAP2"],c=values,cmap="Spectral_r",vmin=vmin,vmax=vmax,s=.24,alpha=.78,linewidths=0,rasterized=True,zorder=2)
        ax.set_title(f"{display_name}\nAll cells (n={len(frame):,})",loc="left",fontsize=7,fontweight="bold",pad=2)
        ax.set_xticks([]);ax.set_yticks([]);ax.tick_params(bottom=False,left=False);ax.grid(False)
        for spine in ax.spines.values(): spine.set_visible(False)
        colorbar=container.colorbar(points,ax=ax,fraction=.035,pad=.025)
        colorbar.set_label("Inferred activity",fontsize=TICK_LABEL_SIZE_PT)
        colorbar.ax.tick_params(labelsize=5.5,length=1,width=.35,pad=1)
    _panel_title(container,label,"Whole-atlas scKNIFE activity","Grey atlas underlay; all cells colored by activity using Spectral_r (independent 2nd–98th percentile scales).")
    container.subplots_adjust(left=.025,right=.995,top=.72,bottom=.08,wspace=.24)


def _draw_status_pathway_scatter(container,data:Path,label:str="SB",show_subtitle:bool=True)->None:
    table,discoveries=_validated_integrated_pathway_scatter(data,"status_pathway_scatter.csv","status_pathway_enrichment.csv","status_four_set_membership.csv","metastatic_vs_nonmetastatic")
    radius=table["radius"].to_numpy(dtype=float); rmin,rmax=radius.min(),radius.max(); sizes=24+96*(radius-rmin)/(rmax-rmin)
    cmap=LinearSegmentedColormap.from_list("status_pathway_yellow_red",["#FFF200","#FFA000","#FF1300"])
    ax=container.subplots(1,1)
    nominal_cutoff_y = -np.log10(NOMINAL_P_CUTOFF)
    ax.axhline(nominal_cutoff_y, color=COLORS["gray"], linestyle="--", linewidth=LINE_WIDTH, alpha=.8, zorder=1)
    if show_subtitle:
        ax.text(.98, nominal_cutoff_y, r"Nominal $p = 0.05$", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=TICK_LABEL_SIZE_PT, color=COLORS["gray"])
    scatter=ax.scatter(table["enrichment_factor"],table["neg_log10_p"],s=sizes,c=table["neg_log10_p"],cmap=cmap,edgecolor="white",linewidth=.45,alpha=.92,zorder=3)
    top5=table.loc[table["label_top5"]]
    for row in top5.itertuples(index=False):ax.annotate(str(int(row.rank)),(row.enrichment_factor,row.neg_log10_p),xytext=(5,5),textcoords="offset points",fontsize=6,fontweight="bold",color=COLORS["dark"])
    short={"Glycosylphosphatidylinositol(GPI)-anchor biosynthesis":"GPI-anchor biosynthesis","Glycolysis and Gluconeogenesis":"Glycolysis/gluconeogenesis"}
    ax.text(.02,.98,"\n".join(f"{int(row.rank)}  {short.get(row.pathway,row.pathway)}" for row in top5.itertuples(index=False)),transform=ax.transAxes,ha="left",va="top",fontsize=5.1,linespacing=1.3,color=COLORS["dark"],bbox={"facecolor":"white","edgecolor":"none","alpha":1.0})
    ax.set_xlabel("Enrichment factor"); ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_xlim(0,max(2.45,table["enrichment_factor"].max()*1.05)); ax.set_ylim(-.03,table["neg_log10_p"].max()*1.10)
    _clean_axis(ax); ax.grid(False,which="both")
    colorbar=container.colorbar(scatter,ax=ax,fraction=.045,pad=.035); colorbar.set_label(r"$-\log_{10}(p)$",fontsize=TICK_LABEL_SIZE_PT); colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE_PT); colorbar.solids.set_rasterized(False)
    legend_values=[.75,1.00,1.25,1.50]
    handles=[ax.scatter([],[],s=24+96*(value-rmin)/(rmax-rmin),facecolor="#F4A300",edgecolor="white",linewidth=.45) for value in legend_values]
    ax.legend(handles,[f"{value:.2f}" for value in legend_values],title=r"$\sqrt{\mathrm{enrichment}}$",frameon=False,fontsize=5.5,title_fontsize=5.5,loc="lower right",borderaxespad=.2,labelspacing=.55)
    subtitle = f"{len(table)} LC–MS candidate pathways; top five labeled.\n{discoveries} pass Holm-adjusted Gamma P < 0.05." if show_subtitle else None
    _panel_title(container,label,"Metastatic-status pathway enrichment",subtitle)
    container.subplots_adjust(left=.13,right=.84,top=.78,bottom=.17)


def _draw_supplement_top2_lcms(container, data: Path, label: str = "SD") -> None:
    _draw_paired_lcms_panel(
        container, data, label, ["Isorhamnetin", "L-threonine"],
        "Top paired biopsy-versus-surgery LC–MS metabolites",
        "Seven paired patients; corrected named candidates (Isorhamnetin FDR-supported; L-threonine exploratory).",
    )



def _standalone(label:str,data:Path):
    dims={"A":(112,82),"B":(112,112),"C":(180,78),"D":(80,72),"E":(180,82),"F":(180,82),"G":(180,76),"SA":(112,82),"SB":(112,112),"SC":(135,82),"SD":(180,82),"SE":(180,105),"SF":(180,100)}
    w,h=dims[label];fig=plt.figure(figsize=(_mm(w),_mm(h)),facecolor="white")
    drawers={"A":lambda c:_draw_three_set_upset(c,data,"A","treatment_upset_summary.csv","LC-MS pathway support"),"B":lambda c:_draw_treatment_pathway_scatter(c,data,"B"),"C":lambda c:_draw_treatment_metabolites(c,data,"C"),"D":lambda c:_draw_robust_cascade(c,data,"D"),"E":lambda c:_draw_robust_trajectories(c,data,"E"),"F":lambda c:_draw_selected_lcms(c,data,"F"),"G":lambda c:_draw_crossmodal_umaps(c,data,"G"),"SA":lambda c:_draw_three_set_upset(c,data,"SA","status_upset_summary.csv","Metastatic-status pathway overlap"),"SB":lambda c:_draw_status_pathway_scatter(c,data,"SB"),"SC":lambda c:_draw_status_metabolites(c,data,"SC"),"SD":lambda c:_draw_supplement_top2_lcms(c,data,"SD")}
    drawers[label](fig);return fig


def make_figure6_main_panel(label:str,data:Path=Path("data/Figure_6"))->plt.Figure:
    if label not in "ABCDEFG": raise ValueError(f"Unknown Figure 6 main panel {label!r}")
    return _standalone(label,data)


def make_figure6_supplementary_panel(label:str,data:Path=Path("data/Figure_6"))->plt.Figure:
    if label not in {"SA","SB","SC","SD"}: raise ValueError(f"Unknown Figure 6 supplementary panel {label!r}")
    return _standalone(label,data)


def make_figure6_compiled(data:Path)->plt.Figure:
    data=data;fig=plt.figure(figsize=(_mm(MAIN_WIDTH_MM),_mm(MAIN_HEIGHT_MM)),facecolor="white")
    rows=fig.subfigures(5,1,height_ratios=[1.02,.72,.86,.66,.90],hspace=.055)
    top=rows[0].subfigures(1,2,width_ratios=[.45,.55],wspace=.06)
    _draw_three_set_upset(top[0],data,"A","treatment_upset_summary.csv","LC-MS pathway support");_draw_treatment_pathway_scatter(top[1],data,"B")
    _draw_treatment_metabolites(rows[1],data,"C")
    middle=rows[2].subfigures(1,2,width_ratios=[.32,.68],wspace=.055);_draw_robust_cascade(middle[0],data,"D");_draw_robust_trajectories(middle[1],data,"E")
    _draw_selected_lcms(rows[3],data,"F");_draw_crossmodal_umaps(rows[4],data,"G")
    return fig


def make_figure6_supplementary(data: Path) -> plt.Figure:
    figure=plt.figure(figsize=(_mm(SUPPLEMENT_WIDTH_MM),_mm(SUPPLEMENT_HEIGHT_MM)),facecolor="white")
    rows=figure.subfigures(4,1,height_ratios=[.72,.92,.72,.72],hspace=.065)
    _draw_three_set_upset(rows[0],data,"SA","status_upset_summary.csv","Metastatic-status pathway overlap")
    _draw_status_pathway_scatter(rows[1],data,"SB")
    _draw_status_metabolites(rows[2],data,"SC")
    _draw_supplement_top2_lcms(rows[3],data,"SD")
    return figure


# Native four-set UpSet and effect-correlation renderers for the MEIA extension.
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
    if mapping.candidate_pathway.duplicated().any() or observed["source_total"]<1 or observed["mapped_sources"]+observed["unmapped"]!=observed["source_total"]:raise RuntimeError(f"{label} LC-MS mapping is inconsistent")
    expected=observed
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
    ax_sets.set_yticks(ypos,FOUR_SET_ORDER);ax_sets.invert_xaxis();ax_sets.invert_yaxis();ax_sets.set_xlabel("");_clean_axis(ax_sets)
    for tick,key in zip(ax_sets.get_yticklabels(),FOUR_SET_ORDER):tick.set_color(FOUR_SET_COLORS[key])
    for x,(_,_,membership) in enumerate(relationships):
        active=[]
        for y,key in enumerate(FOUR_SET_ORDER):
            on=membership[y];ax_matrix.scatter(x,y,s=UPSET_DOT_SIZE if on else UPSET_EMPTY_DOT_SIZE,color=FOUR_SET_COLORS[key] if on else COLORS["light"],edgecolors="none",zorder=3)
            if on:active.append(y)
        if len(active)>1:ax_matrix.plot([x,x],[min(active),max(active)],color=COLORS["dark"],lw=UPSET_CONNECTOR_WIDTH,zorder=2)
    ax_matrix.set_ylim(len(FOUR_SET_ORDER)-.45,-.45);ax_matrix.set_xticks(xpos,["Task","MEIA/GPR","Mapped","Unmapped"],rotation=45,ha="right");ax_matrix.tick_params(axis="x",labelsize=5.5);ax_matrix.set_yticks(ypos,[""]*len(ypos));ax_matrix.tick_params(axis="y",length=0);ax_matrix.spines[:].set_visible(False);ax_matrix.grid(False)
    _panel_title(container,label,title,f"LC-MS: {expected['source_total']} pathways → {expected['harmonized_total']} harmonized\n({expected['mapped_unique']} mapped + {expected['unmapped']} unmapped); MEIA: 218 tasks + 138 pathways.")


def _format_corr_p(value): return "p < 1 × 10⁻³⁰⁰" if value==0 else f"p = {value:.2e}"

def _draw_effect_correlation(container,data,label,filename,contrast,title,xlabel,ylabel,expected_n):
    table=data[filename].copy();panel=table[table["contrast"].eq(contrast)].copy()
    if len(panel)!=expected_n or set(panel["cell_type"])!=set(CORRELATION_CELLTYPE_ORDER):raise RuntimeError(f"{label} correlation rows/cell types invalid")
    rho,rho_p=spearmanr(panel["x"],panel["y"]);r,r_p=pearsonr(panel["x"],panel["y"])
    ax=container.subplots(1,1)
    for ct in CORRELATION_CELLTYPE_ORDER:
        q=panel[panel["cell_type"].eq(ct)];ax.scatter(q["x"],q["y"],s=CORRELATION_POINT_SIZE,alpha=CORRELATION_POINT_ALPHA,color=CORRELATION_CELLTYPE_COLORS[ct],edgecolors="none",rasterized=True,label=ct)
    values=np.r_[panel["x"].to_numpy(),panel["y"].to_numpy()];lo,hi=float(values.min()),float(values.max());pad=max((hi-lo)*.05,.05);limits=(lo-pad,hi+pad)
    ax.plot(limits,limits,"--",lw=LINE_WIDTH,color=COLORS["gray"],zorder=1);ax.axhline(0,color=COLORS["light"],lw=.6,zorder=0);ax.axvline(0,color=COLORS["light"],lw=.6,zorder=0)
    ax.set_xlim(limits);ax.set_ylim(limits);ax.set_aspect("equal",adjustable="box");ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);_clean_axis(ax)
    ax.text(.04,.96,f"Spearman ρ = {rho:.3f} ({_format_corr_p(rho_p)})\nPearson r = {r:.3f} ({_format_corr_p(r_p)})\nn = {len(panel):,}",transform=ax.transAxes,ha="left",va="top",fontsize=TICK_LABEL_SIZE_PT,bbox=dict(boxstyle="round,pad=.3",fc="white",ec="#BBBBBB",alpha=.9))
    ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),frameon=False,fontsize=LEGEND_SIZE_PT)
    subtitle={"surgery_vs_biopsy":"Surgery − Biopsy","metastatic_vs_nonmetastatic":"Metastatic − Non-metastatic"}[contrast]
    _panel_title(container,label,title,subtitle);container.subplots_adjust(left=.14,right=.77,top=.82,bottom=.16)

def _standalone(label,data):
    dims={"A":(112,92),"B":(112,112),"C":(180,78),"D":(80,72),"E":(180,82),"F":(180,82),"G":(180,76),"H":(112,92),"I":(112,92),"SA":(112,92),"SB":(112,112),"SC":(135,82),"SD":(180,82),"SE":(112,92),"SF":(112,92),"SG":(112,92),"SH":(112,92)}
    w,h=dims[label];fig=plt.figure(figsize=(_mm(w),_mm(h)),facecolor="white")
    drawers={"A":lambda c:_draw_four_set_upset(c,data,"A","treatment_four_set_membership.csv","Treatment four-set metabolic overlap"),"B":lambda c:_draw_treatment_pathway_scatter(c,data,"B"),"C":lambda c:_draw_treatment_metabolites(c,data,"C"),"D":lambda c:_draw_robust_cascade(c,data,"D"),"E":lambda c:_draw_robust_trajectories(c,data,"E"),"F":lambda c:_draw_selected_lcms(c,data,"F"),"G":lambda c:_draw_crossmodal_umaps(c,data,"G"),"H":lambda c:_draw_effect_correlation(c,data,"H","gpr_gpa_vs_scknife_effects.csv","surgery_vs_biopsy","GPR-GPA versus scKnife pathway activity","GPR-GPA Cohen’s d","scKnife pathway Cohen’s d",828),"I":lambda c:_draw_effect_correlation(c,data,"I","gpr_gpa_vs_scknife_effects.csv","metastatic_vs_nonmetastatic","GPR-GPA versus scKnife pathway activity","GPR-GPA Cohen’s d","scKnife pathway Cohen’s d",828),"SA":lambda c:_draw_four_set_upset(c,data,"SA","status_four_set_membership.csv","Metastatic-status four-set metabolic overlap"),"SB":lambda c:_draw_status_pathway_scatter(c,data,"SB"),"SC":lambda c:_draw_status_metabolites(c,data,"SC"),"SD":lambda c:_draw_supplement_top2_lcms(c,data,"SD"),"SE":lambda c:_draw_effect_correlation(c,data,"SE","meia_vs_scknife_effects.csv","surgery_vs_biopsy","MEIA versus matched scKnife activity","MEIA Cohen’s d","Matched scKnife activity Cohen’s d",2136),"SF":lambda c:_draw_effect_correlation(c,data,"SF","meia_vs_scknife_effects.csv","metastatic_vs_nonmetastatic","MEIA versus matched scKnife activity","MEIA Cohen’s d","Matched scKnife activity Cohen’s d",2136),"SG":lambda c:_draw_effect_correlation(c,data,"SG","meia_pathway_vs_gpr_gpa_effects.csv","surgery_vs_biopsy","MEIA-Pathway versus GPR-GPA","MEIA-Pathway Cohen’s d","GPR-GPA Cohen’s d",828),"SH":lambda c:_draw_effect_correlation(c,data,"SH","meia_pathway_vs_gpr_gpa_effects.csv","metastatic_vs_nonmetastatic","MEIA-Pathway versus GPR-GPA","MEIA-Pathway Cohen’s d","GPR-GPA Cohen’s d",828)}
    drawers[label](fig);return fig

def make_figure6_main_panel(label,data=Path("data/Figure_6")):
    if label not in set("ABCDEFGHI"):raise ValueError(label)
    return _standalone(label,data)

def make_figure6_supplementary_panel(label,data=Path("data/Figure_6")):
    if label not in {"SA","SB","SC","SD","SE","SF","SG","SH"}:raise ValueError(label)
    return _standalone(label,data)

def make_figure6_compiled(data):
    fig=plt.figure(figsize=(_mm(MAIN_WIDTH_MM),_mm(MAIN_HEIGHT_MM)),facecolor="white");rows=fig.subfigures(6,1,height_ratios=[1.02,.72,.86,.66,.90,1.0],hspace=.05)
    top=rows[0].subfigures(1,2,width_ratios=[.45,.55],wspace=.06);_draw_four_set_upset(top[0],data,"A","treatment_four_set_membership.csv","Treatment four-set metabolic overlap");_draw_treatment_pathway_scatter(top[1],data,"B")
    _draw_treatment_metabolites(rows[1],data,"C");middle=rows[2].subfigures(1,2,width_ratios=[.32,.68],wspace=.055);_draw_robust_cascade(middle[0],data,"D");_draw_robust_trajectories(middle[1],data,"E");_draw_selected_lcms(rows[3],data,"F");_draw_crossmodal_umaps(rows[4],data,"G")
    corr=rows[5].subfigures(1,2,wspace=.08);_draw_effect_correlation(corr[0],data,"H","gpr_gpa_vs_scknife_effects.csv","surgery_vs_biopsy","GPR-GPA versus scKnife pathway activity","GPR-GPA Cohen’s d","scKnife pathway Cohen’s d",828);_draw_effect_correlation(corr[1],data,"I","gpr_gpa_vs_scknife_effects.csv","metastatic_vs_nonmetastatic","GPR-GPA versus scKnife pathway activity","GPR-GPA Cohen’s d","scKnife pathway Cohen’s d",828);return fig

def make_figure6_supplementary(data):
    fig=plt.figure(figsize=(_mm(SUPPLEMENT_WIDTH_MM),_mm(SUPPLEMENT_HEIGHT_MM)),facecolor="white");rows=fig.subfigures(6,1,height_ratios=[.72,.92,.72,.72,1,1],hspace=.05)
    _draw_four_set_upset(rows[0],data,"SA","status_four_set_membership.csv","Metastatic-status four-set metabolic overlap");_draw_status_pathway_scatter(rows[1],data,"SB");_draw_status_metabolites(rows[2],data,"SC");_draw_supplement_top2_lcms(rows[3],data,"SD")
    c1=rows[4].subfigures(1,2,wspace=.08);_draw_effect_correlation(c1[0],data,"SE","meia_vs_scknife_effects.csv","surgery_vs_biopsy","MEIA versus matched scKnife activity","MEIA Cohen’s d","Matched scKnife activity Cohen’s d",2136);_draw_effect_correlation(c1[1],data,"SF","meia_vs_scknife_effects.csv","metastatic_vs_nonmetastatic","MEIA versus matched scKnife activity","MEIA Cohen’s d","Matched scKnife activity Cohen’s d",2136)
    c2=rows[5].subfigures(1,2,wspace=.08);_draw_effect_correlation(c2[0],data,"SG","meia_pathway_vs_gpr_gpa_effects.csv","surgery_vs_biopsy","MEIA-Pathway versus GPR-GPA","MEIA-Pathway Cohen’s d","GPR-GPA Cohen’s d",828);_draw_effect_correlation(c2[1],data,"SH","meia_pathway_vs_gpr_gpa_effects.csv","metastatic_vs_nonmetastatic","MEIA-Pathway versus GPR-GPA","MEIA-Pathway Cohen’s d","GPR-GPA Cohen’s d",828);return fig

def _render_extension_panel(label,filename,registry,panel_group):
    lineage=DATA_MANIFEST[panel_group][label];_verify_staged_file(filename);table=pd.read_csv(DATA_ROOT/filename);tables={filename:table};_audit_panel_data(label,tables,panel_group);registry[label]=tables;fig=_standalone(label,tables)
    for suffix in ("png","pdf"):fig.savefig(_individual_dir/f"Figure_6_{label}.{suffix}",format=suffix,dpi=OUTPUT_DPI,facecolor="white",bbox_inches="tight",pad_inches=_standalone_export_pad_inches)
    _save_live_vector_svg(fig,_individual_dir/f"Figure_6_{label}.svg","tight",list(fig.axes),list(fig.texts),pad_inches=_standalone_export_pad_inches)
    _panel_manifest[label]={"mode":"standalone_data_driven_figure","renderer":"matplotlib_fallback","matplotlib_version":mpl.__version__,"axis_count":len(fig.axes),"saved_in_individual_notebook_cell":True,"data_lineage_kind":lineage["kind"],"intermediate_inputs":lineage["inputs"],"data_extractor":lineage["extractor"],"extraction_mode":lineage["extraction_mode"]}
    (_individual_dir/'Figure_6_panel_export_manifest.json').write_text(json.dumps(_panel_manifest,indent=2)+'\n');plt.figure(fig.number);plt.show();plt.close(fig)

# Final Figure 6 storyline dispatch (A–F main; SA–SI supplementary).
def _draw_treatment_metabolites_relabel(container,data,label="D"):
    copied={name:table.copy() for name,table in data.items()}
    copied["selected_metabolite_statistics.csv"]["panel"]=copied["selected_metabolite_statistics.csv"]["panel"].replace({"C":label})
    copied["selected_metabolite_values.csv"]["panel"]=copied["selected_metabolite_values.csv"]["panel"].replace({"C":label})
    _draw_treatment_metabolites(container,copied,label)

def _standalone(label,data):
    dims={"A":(112,92),"B":(112,112),"C":(112,92),"D":(180,125),"E":(180,82),"F":(180,82),"SA":(112,92),"SB":(112,112),"SC":(112,92),"SD":(180,76),"SE":(180,82),"SF":(112,92),"SG":(112,92),"SH":(112,92),"SI":(112,92)}
    w,h=dims[label];fig=plt.figure(figsize=(_mm(w),_mm(h)),facecolor="white")
    drawers={
      "A":lambda c:_draw_four_set_upset(c,data,"A","treatment_four_set_membership.csv","Treatment four-set metabolic overlap"),
      "B":lambda c:_draw_treatment_pathway_scatter(c,data,"B"),
      "C":lambda c:_draw_effect_correlation(c,data,"C","gpr_gpa_vs_scknife_effects.csv","surgery_vs_biopsy","GPR-GPA versus scKnife pathway activity","GPR-GPA Cohen’s d","scKnife pathway Cohen’s d",828),
      "D":lambda c:_draw_treatment_metabolites_relabel(c,data,"D"),
      "E":lambda c:_draw_robust_trajectories(c,data,"E"),"F":lambda c:_draw_selected_lcms(c,data,"F"),
      "SA":lambda c:_draw_four_set_upset(c,data,"SA","status_four_set_membership.csv","Metastatic-status four-set metabolic overlap"),
      "SB":lambda c:_draw_status_pathway_scatter(c,data,"SB"),
      "SC":lambda c:_draw_effect_correlation(c,data,"SC","gpr_gpa_vs_scknife_effects.csv","metastatic_vs_nonmetastatic","GPR-GPA versus scKnife pathway activity","GPR-GPA Cohen’s d","scKnife pathway Cohen’s d",828),
      "SD":lambda c:_draw_crossmodal_umaps(c,data,"SD"),"SE":lambda c:_draw_supplement_top2_lcms(c,data,"SE"),
      "SF":lambda c:_draw_effect_correlation(c,data,"SF","meia_vs_scknife_effects.csv","surgery_vs_biopsy","MEIA versus matched scKnife activity","MEIA Cohen’s d","Matched scKnife activity Cohen’s d",2136),
      "SG":lambda c:_draw_effect_correlation(c,data,"SG","meia_vs_scknife_effects.csv","metastatic_vs_nonmetastatic","MEIA versus matched scKnife activity","MEIA Cohen’s d","Matched scKnife activity Cohen’s d",2136),
      "SH":lambda c:_draw_effect_correlation(c,data,"SH","meia_pathway_vs_gpr_gpa_effects.csv","surgery_vs_biopsy","MEIA-Pathway versus GPR-GPA","MEIA-Pathway Cohen’s d","GPR-GPA Cohen’s d",828),
      "SI":lambda c:_draw_effect_correlation(c,data,"SI","meia_pathway_vs_gpr_gpa_effects.csv","metastatic_vs_nonmetastatic","MEIA-Pathway versus GPR-GPA","MEIA-Pathway Cohen’s d","GPR-GPA Cohen’s d",828),
    }
    drawers[label](fig);return fig

def make_figure6_main_panel(label,data=Path("data/Figure_6")):
    if label not in set("ABCDEF"):raise ValueError(label)
    return _standalone(label,data)

def make_figure6_supplementary_panel(label,data=Path("data/Figure_6")):
    if label not in {"SA","SB","SC","SD","SE","SF","SG","SH","SI"}:raise ValueError(label)
    return _standalone(label,data)

def make_figure6_compiled(data):
    fig=plt.figure(figsize=(_mm(MAIN_WIDTH_MM),_mm(405)),facecolor="white");rows=fig.subfigures(5,1,height_ratios=[1.02,.95,1.40,.86,.66],hspace=.05)
    top=rows[0].subfigures(1,2,width_ratios=[.45,.55],wspace=.06);_draw_four_set_upset(top[0],data,"A","treatment_four_set_membership.csv","Treatment four-set metabolic overlap");_draw_treatment_pathway_scatter(top[1],data,"B")
    _draw_effect_correlation(rows[1],data,"C","gpr_gpa_vs_scknife_effects.csv","surgery_vs_biopsy","GPR-GPA versus scKnife pathway activity","GPR-GPA Cohen’s d","scKnife pathway Cohen’s d",828)
    _draw_treatment_metabolites_relabel(rows[2],data,"D");_draw_robust_trajectories(rows[3],data,"E");_draw_selected_lcms(rows[4],data,"F");return fig

def make_figure6_supplementary(data):
    fig=plt.figure(figsize=(_mm(SUPPLEMENT_WIDTH_MM),_mm(750)),facecolor="white");rows=fig.subfigures(7,1,height_ratios=[.72,.92,.95,.90,.72,1,1],hspace=.045)
    _draw_four_set_upset(rows[0],data,"SA","status_four_set_membership.csv","Metastatic-status four-set metabolic overlap");_draw_status_pathway_scatter(rows[1],data,"SB")
    _draw_effect_correlation(rows[2],data,"SC","gpr_gpa_vs_scknife_effects.csv","metastatic_vs_nonmetastatic","GPR-GPA versus scKnife pathway activity","GPR-GPA Cohen’s d","scKnife pathway Cohen’s d",828)
    _draw_crossmodal_umaps(rows[3],data,"SD");_draw_supplement_top2_lcms(rows[4],data,"SE")
    c1=rows[5].subfigures(1,2,wspace=.08);_draw_effect_correlation(c1[0],data,"SF","meia_vs_scknife_effects.csv","surgery_vs_biopsy","MEIA versus matched scKnife activity","MEIA Cohen’s d","Matched scKnife activity Cohen’s d",2136);_draw_effect_correlation(c1[1],data,"SG","meia_vs_scknife_effects.csv","metastatic_vs_nonmetastatic","MEIA versus matched scKnife activity","MEIA Cohen’s d","Matched scKnife activity Cohen’s d",2136)
    c2=rows[6].subfigures(1,2,wspace=.08);_draw_effect_correlation(c2[0],data,"SH","meia_pathway_vs_gpr_gpa_effects.csv","surgery_vs_biopsy","MEIA-Pathway versus GPR-GPA","MEIA-Pathway Cohen’s d","GPR-GPA Cohen’s d",828);_draw_effect_correlation(c2[1],data,"SI","meia_pathway_vs_gpr_gpa_effects.csv","metastatic_vs_nonmetastatic","MEIA-Pathway versus GPR-GPA","MEIA-Pathway Cohen’s d","GPR-GPA Cohen’s d",828);return fig

def _render_panel_files(label,filenames,registry,panel_group):
    lineage=DATA_MANIFEST[panel_group][label];tables={}
    for filename in filenames:
        _verify_staged_file(filename);tables[filename]=pd.read_csv(DATA_ROOT/filename)
    _audit_panel_data(label,tables,panel_group);registry[label]=tables;fig=_standalone(label,tables)
    for suffix in ("png","pdf"):fig.savefig(_individual_dir/f"Figure_6_{label}.{suffix}",format=suffix,dpi=OUTPUT_DPI,facecolor="white",bbox_inches="tight",pad_inches=_standalone_export_pad_inches)
    _save_live_vector_svg(fig,_individual_dir/f"Figure_6_{label}.svg","tight",list(fig.axes),list(fig.texts),pad_inches=_standalone_export_pad_inches)
    _panel_manifest[label]={"mode":"standalone_data_driven_figure","renderer":"matplotlib_fallback","matplotlib_version":mpl.__version__,"axis_count":len(fig.axes),"saved_in_individual_notebook_cell":True,"data_lineage_kind":lineage["kind"],"intermediate_inputs":lineage["inputs"],"data_extractor":lineage["extractor"],"extraction_mode":lineage["extraction_mode"]}
    (_individual_dir/'Figure_6_panel_export_manifest.json').write_text(json.dumps(_panel_manifest,indent=2)+'\n');plt.figure(fig.number);plt.show();plt.close(fig)

from pooled_concordance import draw_pooled_concordance as _draw_pooled_concordance
# Blank-corrected notebook contract: retain every inline plot and additionally
# export the two submission composites plus the corrected A/B/SA/SB review panels.
_compiled_dir=Path("figs/panel/Figure_6_adduct_corrected_v1").resolve()
_individual_dir=Path("figs/individual/Figure_6_adduct_corrected_v1").resolve()
_compiled_dir.mkdir(parents=True,exist_ok=True);_individual_dir.mkdir(parents=True,exist_ok=True)
_review_export_labels={"A","B","C","SA","SB"}
_blank_corrected_export_paths=[]

def _save_blank_corrected_figure(fig,output_dir,stem,bbox_inches="tight",pad_inches=.04,include_svg=False):
    saved=[]
    suffixes=("png","pdf","svg") if include_svg else ("png","pdf")
    for suffix in suffixes:
        path=output_dir/f"{stem}.{suffix}"
        kwargs={"format":suffix,"facecolor":"white","bbox_inches":bbox_inches,"pad_inches":pad_inches}
        kwargs["dpi"]=OUTPUT_DPI if stem.endswith("_C") or suffix=="png" else mpl.rcParams["savefig.dpi"]
        fig.savefig(path,**kwargs);saved.append(path);_blank_corrected_export_paths.append(path)
    print("Exported:",", ".join(str(path.relative_to(ROOT)) for path in saved))
    return saved

def _render_extension_panel(label,filename,registry,panel_group):
 lineage=DATA_MANIFEST[panel_group][label];_verify_staged_file(filename);table=pd.read_csv(DATA_ROOT/filename,sep="\t" if filename.endswith(".tsv") else ",");tables={filename:table};_audit_panel_data(label,tables,panel_group);registry[label]=tables;fig=_standalone(label,tables)
 if label in _review_export_labels:_save_blank_corrected_figure(fig,_individual_dir,f"Figure_6_adduct_corrected_v1_{label}",bbox_inches=None if label=="C" else "tight",pad_inches=0 if label=="C" else .04,include_svg=(label=="C"))
 plt.figure(fig.number);plt.show();plt.close(fig)
def _render_panel_files(label,filenames,registry,panel_group):
 tables={}
 for filename in filenames:_verify_staged_file(filename);tables[filename]=pd.read_csv(DATA_ROOT/filename,sep="\t" if filename.endswith(".tsv") else ",")
 _audit_panel_data(label,tables,panel_group);registry[label]=tables;fig=_standalone(label,tables)
 if label in _review_export_labels:_save_blank_corrected_figure(fig,_individual_dir,f"Figure_6_adduct_corrected_v1_{label}",bbox_inches=None if label=="C" else "tight",pad_inches=0 if label=="C" else .04,include_svg=(label=="C"))
 plt.figure(fig.number);plt.show();plt.close(fig)

def _draw_panel_d_graph_guided(container,data,label="D"):
    seeds=data["panel_d_graph_seeds.csv"].copy().sort_values("display_order")
    enrichment=data["panel_d_meia_joint_enrichment.csv"].copy().sort_values("contrast_order")
    meia_features=data["panel_d_meia_tcell_reversal.csv"].copy()
    gpr_features=data["panel_d_gpr_context_reversal.csv"].copy()
    cells=data["panel_d_mechanism_cell_scores.csv.gz"].copy()
    patients=data["panel_d_mechanism_patient_medians.csv"].copy()
    tests=data["panel_d_mechanism_patient_tests.csv"].copy()
    if (len(seeds),len(enrichment),len(meia_features),len(gpr_features),len(cells),len(patients),len(tests))!=(6,2,29,14,14552,28,6):raise RuntimeError("Panel D patient-aware table dimensions changed")
    if cells.groupby("method")["cell_id"].nunique().to_dict()!={"GPR_GPA":7276,"MEIA":7276}:raise RuntimeError("Panel D requires 7,276 Bx/Surgery T cells per method")
    patient_order=["P1","P2","P3","P4","P5","P6","P7"]
    patient_colors=dict(zip(patient_order,sns.color_palette("Set2",n_colors=7).as_hex()))
    site_colors={"Bx":"#377EB8","Surgery":"#E41A1C"};status_colors={"Non-metastatic":"#4DAF4A","Metastatic":"#E41A1C"}
    outer=container.subfigures(2,1,height_ratios=[.25,.75],hspace=.025)
    axg=outer[0].subplots();axg.set_axis_off();axg.set_xlim(0,1);axg.set_ylim(-.12,1.25)
    def box(x,y,w,h,text,face,edge,fs=5.7):
        patch=mpl.patches.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.018",facecolor=face,edgecolor=edge,linewidth=.9)
        axg.add_patch(patch);axg.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=COLORS["dark"],linespacing=1.10)
    box(.025,.23,.26,.48,"DPAGT1 · PMANM · MAN1PT2 · ACGAM2E\nGDP-mannose · UDP-galactose","#F2F2F2",COLORS["gray"],5.35)
    box(.37,.25,.20,.44,"Nucleotide- /\namino-sugar supply","#FFF3CD","#B58A00",5.7)
    outputs=[("N/O-glycan + GPI","#009E73"),("Glycosphingolipid","#0072B2"),("Glycosaminoglycans","#CC79A7")]
    for (txt,col),y in zip(outputs,[.66,.39,.12]):box(.66,y,.22,.18,txt,mpl.colors.to_rgba(col,.10),col,5.25)
    arrow=dict(arrowstyle="-|>",lw=.9,color=COLORS["gray"],mutation_scale=7)
    axg.annotate("",xy=(.365,.47),xytext=(.29,.47),arrowprops=arrow)
    for y in (.75,.48,.21):axg.annotate("",xy=(.655,y),xytext=(.575,.47),arrowprops=arrow)
    tr=enrichment.loc[enrichment["contrast"].eq("surgery_vs_biopsy")].iloc[0]
    box(.90,.25,.085,.44,f"MEIA\n14/29\nOR 7.54\nq=6.46×10⁻⁶","#FFF5F5","#D55E00",5.1)

    parts=outer[1].subfigures(1,3,width_ratios=[.34,.46,.20],wspace=.045)
    rng=np.random.default_rng(42)
    meia_cells=cells.loc[cells["method"].eq("MEIA")].copy();meia_patients=patients.loc[patients["method"].eq("MEIA")].copy();meia_tests=tests.loc[tests["method"].eq("MEIA")].copy()
    def cell_box(ax,groups,colors,positions):
        arrays=[g["mechanism_score"].to_numpy(float) for g in groups]
        bp=ax.boxplot(arrays,positions=positions,widths=.48,patch_artist=True,showfliers=False,whis=1.5,zorder=2,
            medianprops={"color":COLORS["dark"],"linewidth":1.05},whiskerprops={"linewidth":.8},capprops={"linewidth":.8})
        for patch,col in zip(bp["boxes"],colors):patch.set_facecolor(mpl.colors.to_rgba(col,.24));patch.set_edgecolor(col);patch.set_linewidth(.9)
        for xpos,g,col in zip(positions,groups,colors):
            jitter=rng.normal(xpos,.065,len(g));ax.scatter(jitter,g["mechanism_score"],s=.30,color=col,alpha=.028,linewidth=0,rasterized=True,zorder=1)
        ax.set_yscale("symlog",linthresh=1,linscale=.70);ax.axhline(0,color=COLORS["gray"],linewidth=.65,linestyle="--",zorder=0);_clean_axis(ax)

    axl=parts[0].subplots();site_groups=[meia_cells.loc[meia_cells["Site"].eq(x)] for x in ["Bx","Surgery"]]
    cell_box(axl,site_groups,[site_colors["Bx"],site_colors["Surgery"]],[0,1])
    piv=meia_patients.pivot(index="patient_display",columns="Site",values="patient_median").loc[patient_order]
    for pid,row in piv.iterrows():
        col=patient_colors[pid];axl.plot([0,1],[row["Bx"],row["Surgery"]],color=col,linewidth=.85,alpha=.9,zorder=4)
        axl.scatter([0,1],[row["Bx"],row["Surgery"]],s=26,color=col,edgecolor="white",linewidth=.45,zorder=5)
        axl.text(1.045,row["Surgery"],pid,fontsize=4.5,color=col,va="center")
    t=meia_tests.loc[meia_tests["comparison"].eq("surgery_vs_biopsy")].iloc[0]
    axl.set_xticks([0,1],["Biopsy\nn=6,178 cells","Surgery\nn=1,098 cells"]);axl.set_xlim(-.38,1.32);axl.set_ylabel("Integrated MEIA mechanism activity\n(equal-feature z mean; symlog)");axl.set_title("Longitudinal T-cell activity",fontweight="bold",pad=4)
    axl.text(.03,.97,f"5/7 patients decrease\nmedian Δ={t.patient_effect:+.2f}; exact paired p={t.p_value:.3f}\ncellwise d={t.cellwise_cohens_d:+.2f} (descriptive)",transform=axl.transAxes,ha="left",va="top",fontsize=5.0,bbox={"facecolor":"white","edgecolor":"none","alpha":.82,"pad":1.0})

    status_axes=parts[1].subplots(1,2,sharey=True)
    for ax,site in zip(status_axes,["Bx","Surgery"]):
        site_cells=meia_cells.loc[meia_cells["Site"].eq(site)];site_patients=meia_patients.loc[meia_patients["Site"].eq(site)]
        order=["Non-metastatic","Metastatic"];groups=[site_cells.loc[site_cells["metastatic_status"].eq(x)] for x in order]
        cell_box(ax,groups,[status_colors[x] for x in order],[0,1])
        for xpos,status in enumerate(order):
            pp=site_patients.loc[site_patients["metastatic_status"].eq(status)].sort_values("patient_display")
            offsets=np.linspace(-.075,.075,len(pp))
            for off,row in zip(offsets,pp.itertuples(index=False)):
                col=patient_colors[row.patient_display];ax.scatter(xpos+off,row.patient_median,s=27,color=col,edgecolor="white",linewidth=.45,zorder=5);ax.text(xpos+off+.025,row.patient_median,row.patient_display,fontsize=4.2,color=col,va="bottom")
        tt=meia_tests.loc[meia_tests["comparison"].eq("metastatic_vs_nonmetastatic")&meia_tests["site"].eq(site)].iloc[0]
        counts=[len(g) for g in groups];ax.set_xticks([0,1],[f"Non-meta\nn={counts[0]:,}",f"Meta\nn={counts[1]:,}"]);ax.set_xlim(-.38,1.38);ax.set_title("Biopsy" if site=="Bx" else "Surgery",fontweight="bold",pad=4)
        ax.text(.50,.97,f"patient p={tt.p_value:.2f}\ncellwise d={tt.cellwise_cohens_d:+.2f}",transform=ax.transAxes,ha="center",va="top",fontsize=5.0,bbox={"facecolor":"white","edgecolor":"none","alpha":.82,"pad":.8})
    status_axes[0].set_ylabel("Mechanism activity (symlog)");parts[1].suptitle("Metastatic-status convergence",fontsize=7,fontweight="bold",y=.98)

    axc=parts[2].subplots();gt=tests.loc[tests["method"].eq("GPR_GPA")].copy()
    row_order=[("surgery_vs_biopsy","paired","Surgery−Bx"),("metastatic_vs_nonmetastatic","Bx","Meta−non\nat biopsy"),("metastatic_vs_nonmetastatic","Surgery","Meta−non\nat surgery")]
    for y,(comparison,site,textlabel) in enumerate(row_order):
        row=gt.loc[gt["comparison"].eq(comparison)&gt["site"].eq(site)].iloc[0];value=float(row.cellwise_cohens_d);col="#D55E00" if value<0 else "#0072B2"
        axc.plot([0,value],[y,y],color=col,linewidth=1.8,zorder=1);axc.scatter(value,y,s=42,color=col,edgecolor="white",linewidth=.5,zorder=2);axc.text(value,y+.20,f"d={value:+.2f}\np={row.p_value:.2f}",ha="center",va="bottom",fontsize=4.7)
    axc.axvline(0,color=COLORS["gray"],linewidth=.7,linestyle="--");axc.set_yticks(range(3),[x[2] for x in row_order]);axc.invert_yaxis();axc.set_xlim(-1.65,1.75);axc.set_xlabel("Cellwise Cohen's d");axc.set_title("GPR-rule context",fontweight="bold",pad=4);_clean_axis(axc)
    axc.text(.5,.02,"14 pathways\npatient exact p shown",transform=axc.transAxes,ha="center",va="bottom",fontsize=4.8,color=COLORS["gray"])

    _panel_title(container,label,"T-cell glycoconjugate activity: longitudinal decrease and postoperative convergence","Boxes/points: all T cells; large labelled dots: P1–P7 medians; patient-level exact tests")
    container.text(.99,.004,"Cell distributions and Cohen's d are descriptive because cells are nested and unequal across patients; GPR-GPA is contextual, not an independent discovery.",ha="right",va="bottom",fontsize=4.7,color=COLORS["gray"])
    parts[0].subplots_adjust(left=.21,right=.94,top=.82,bottom=.24);parts[1].subplots_adjust(left=.14,right=.98,top=.77,bottom=.24,wspace=.24);parts[2].subplots_adjust(left=.32,right=.96,top=.82,bottom=.24)

def _draw_treatment_metabolites_relabel(container,data,label="D"):
    _draw_panel_d_graph_guided(container,data,label)

# Final matched-entity concordance integration from the mechanism analysis.
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
    mapping=data["robust_metabolite_broad_entity_mapping.tsv"].copy()
    counts=data["robust_metabolite_broad_entity_counts.csv"].copy().sort_values("rank_by_abundance")
    required={"metabolite","display_name","supporting_major_celltypes","exact_human_gem_pathways","primary_broader_entity","secondary_broader_entities","mapping_basis_rule_id","mapping_confidence","manual_review_flag","manual_review_note"}
    if len(mapping)!=157 or mapping["metabolite"].nunique()!=157 or not required<=set(mapping):raise RuntimeError("Panel E requires a complete auditable map of 157 unique Tier-3 metabolites")
    if len(counts)!=14 or counts["unique_metabolites"].sum()!=157:raise RuntimeError("Panel E requires 14 broader biochemical families covering all 157 metabolites")
    if counts[["T cells","Epithelial","CAF","Endothelial"]].to_numpy().sum()!=158:raise RuntimeError("Panel E cell-type support changed")
    if counts["shown_in_panel"].sum()!=8 or set(mapping["mapping_confidence"])-{"high","moderate","low"}:raise RuntimeError("Panel E mapping contract changed")
    return mapping,counts

def _draw_pathway_context_pie(container,data,label="E"):
    mapping,counts=_validated_robust_context(data);top=counts.loc[counts.shown_in_panel].copy();other=counts.loc[~counts.shown_in_panel]
    celltypes=["T cells","Epithelial","CAF","Endothelial"]
    aggregate={"primary_broader_entity":"Other curated families","unique_metabolites":int(other.unique_metabolites.sum()),**{celltype:int(other[celltype].sum()) for celltype in celltypes}}
    plot=pd.concat([top,pd.DataFrame([aggregate])],ignore_index=True)
    short={
      "Glycerophospholipids and lysophospholipids":"Glycerophospholipids / LPCs",
      "Xenobiotics, pharmaceuticals and environmental chemicals":"Drugs / xenobiotics",
      "Fatty acids and beta-oxidation intermediates":"Fatty acids / β-oxidation",
      "Glycosaminoglycans, glycosphingolipids and glycans":"Glycans / glycosphingolipids",
      "Vitamins, cofactors, retinoids and isoprenoids":"Vitamins / cofactors / retinoids",
      "Aromatic amino acids and biogenic amines":"Aromatic AAs / biogenic amines",
      "Nucleotides, nucleic acids and RNA modification":"Nucleotides / RNA",
      "Peptides, proteins and post-translational intermediates":"Peptides / proteins",
      "Other amino acids, sulfur and one-carbon metabolism":"Sulfur / one-carbon AAs",
    }
    matrix=plot[celltypes].to_numpy(float);labels=[short.get(x,x) for x in plot.primary_broader_entity]
    grid=container.add_gridspec(1,2,width_ratios=[4.1,.82],left=.40,right=.98,top=.70,bottom=.17,wspace=.08)
    ax=container.add_subplot(grid[0]);ax_total=container.add_subplot(grid[1],sharey=ax)
    cmap=LinearSegmentedColormap.from_list("broad_entity_teal",["#F4F7F7","#A7D8D0","#00796B"])
    im=ax.imshow(matrix,aspect="auto",cmap=cmap,vmin=0,vmax=max(1,matrix.max()),interpolation="nearest")
    for y in range(len(plot)):
      for x in range(len(celltypes)):
       value=int(matrix[y,x]);ax.text(x,y,str(value) if value else "·",ha="center",va="center",fontsize=5.2,color="white" if value>matrix.max()*.55 else COLORS["dark"],fontweight="bold" if value else "normal")
    ax.set_xticks(range(4),["T","Epi","CAF","Endo"],fontsize=5.2);ax.xaxis.tick_top();ax.tick_params(axis="x",length=0,pad=2)
    ax.set_yticks(range(len(plot)),labels,fontsize=5.0);ax.tick_params(axis="y",length=0,pad=2)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.grid(False)
    totals=plot.unique_metabolites.to_numpy(int);y=np.arange(len(plot));ax_total.barh(y,totals,color="#4D4D4D",height=.56)
    for ypos,value in zip(y,totals):ax_total.text(value+.35,ypos,str(value),ha="left",va="center",fontsize=5.2,fontweight="bold",color=COLORS["dark"])
    ax_total.set_xlim(0,max(totals)*1.22);ax_total.set_xticks([]);ax_total.tick_params(axis="y",left=False,labelleft=False);_clean_axis(ax_total)
    ax_total.set_title("Total",fontsize=5.2,pad=3)
    low=int(mapping.manual_review_flag.astype(bool).sum())
    _panel_title(container,label,"Broader biochemical entities","157 Tier-3 metabolites; reviewed major cell types.")
    container.text(.01,.01,f"Seven largest + sulfur family; other six; exact mappings and {low} flag{'s' if low!=1 else ''} in TSV.",ha="left",va="bottom",fontsize=4.3,color=COLORS["gray"])

def _draw_filtering_cascade(container,data,label="Supplementary candidate"):
    funnel=data["robust_metabolite_filtering_funnel.csv"].copy().sort_values("display_order")
    if funnel["count"].tolist()!=[20650,5247,649,158,157,3]:raise RuntimeError("Filtering-cascade counts changed")
    ax=container.subplots(1,1);y=np.arange(len(funnel));counts=funnel["count"].to_numpy(float)
    ax.plot(counts,y,color="#0072B2",lw=1.3,zorder=1);ax.scatter(counts,y,s=34,color="#0072B2",edgecolor="white",linewidth=.6,zorder=2)
    for ypos,count in zip(y,funnel["count"]):ax.text(.93,ypos,f"{int(count):,}",transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=TICK_LABEL_SIZE_PT,fontweight="bold",color=COLORS["dark"])
    ax.set_xscale("log");ax.set_xlim(2,50000);ax.set_yticks(y,["Tested pairs","Replicated treatment","Recurrence rebound","Baseline-like recurrence","Unique Tier-3 metabolites","Shown in main D"]);ax.invert_yaxis();ax.set_xlabel("Count (log scale)");_clean_axis(ax)
    _panel_title(container,label,"Robust inferred-metabolite filtering cascade","Metabolite–cell-type pairs are distinguished from unique inferred metabolite nodes.")
    container.subplots_adjust(left=.30,right=.96,top=.73,bottom=.20)

def _sulfur_treatment_statistics(table):
    from scipy.stats import wilcoxon,rankdata
    from statsmodels.stats.multitest import multipletests
    celltypes=['Epithelial','Endothelial','Macrophage','T cells','CAF','PVL'];labels=table.score_label.drop_duplicates();rows=[]
    def rb(delta):
        nonzero=delta[~np.isclose(delta,0)];ranks=rankdata(np.abs(nonzero));return float((ranks[nonzero>0].sum()-ranks[nonzero<0].sum())/ranks.sum()) if len(nonzero) else 0.
    for cohort,keep in [('All 7',np.ones(len(table),bool)),('Without P3',~table.is_longitudinal_case.astype(bool))]:
        for score in labels:
            for celltype in celltypes:
                frame=table.loc[keep].query('score_label==@score and cell_type_analysis==@celltype');delta=(frame.Surgery-frame.Bx).to_numpy();test=wilcoxon(delta,alternative='two-sided',method='exact')
                rows.append({'cohort':cohort,'score_label':score,'cell_type':celltype,'n':len(delta),'n_decrease':int((delta<0).sum()),'n_increase':int((delta>0).sum()),'median_delta':float(np.median(delta)),'rank_biserial':rb(delta),'p':float(test.pvalue)})
    stats=pd.DataFrame(rows)
    for cohort in ('All 7','Without P3'):
        mask=stats.cohort.eq(cohort);stats.loc[mask,'q54']=multipletests(stats.loc[mask,'p'],method='fdr_bh')[1]
    return stats

def _draw_homocysteine_screen(container,data,label="F"):
    cohort=data["sulfur_cohort_patient_values.csv"].copy();order=['Epithelial','Endothelial','Macrophage','T cells','CAF','PVL']
    if len(cohort)!=378 or cohort.score_label.nunique()!=9 or set(cohort.cell_type_analysis)!=set(order):raise RuntimeError("Reviewed-label sulfur sensitivity table changed")
    stats=data["sulfur_cohort_treatment_statistics.csv"].copy()
    if len(stats)!=108 or set(stats.cell_type)!=set(order):raise RuntimeError("Reviewed-label sulfur statistics changed")
    screen=stats.query("cohort=='All 7' and score_label=='MEIA task | Homocysteine degradation'").set_index('cell_type').loc[order].reset_index()
    ax=container.subplots(1,1);y=np.arange(len(order));ax.scatter(screen.n_decrease,y,s=55,color='#0072B2',edgecolor='white',linewidth=.6,zorder=3)
    for x,ypos,delta,p,q in zip(screen.n_decrease,y,screen.median_delta,screen.p,screen.q54):ax.text(x+.08,ypos,f'{int(x)}/7; Δ̃={delta:.1e}; p={p:.2f}; q={q:.2f}',va='center',fontsize=5.5,color='#0072B2')
    ax.axvline(3.5,color='#777777',lw=.8,ls='--');ax.set(xlim=(-.2,9.8),xticks=range(0,8),yticks=range(len(order)),yticklabels=order,xlabel='Patients with lower task score at Surgery');_clean_axis(ax)
    _panel_title(container,label,"Homocysteine-degradation sensitivity","Reviewed major labels; exact paired Wilcoxon across seven patient medians; BH across 54 tests.")
    container.text(.99,.01,"Sensitivity view only; the prior macrophage-specific direction is not retained under reviewed labels.",ha="right",va="bottom",fontsize=5.2,color=COLORS["gray"])
    container.subplots_adjust(left=.22,right=.97,top=.72,bottom=.22)

def _draw_caf_selenocystathionine(container,data,label="F"):
    values=data["caf_selenocystathionine_patient_medians.csv"].copy()
    stats=data["caf_selenocystathionine_patient_statistics.csv"].iloc[0]
    paired=values.pivot(index=["PatientID","Status"],columns="Timepoint",values="median_score").reset_index().sort_values("PatientID").reset_index(drop=True)
    if len(paired)!=7 or paired[["Bx","Surgery"]].isna().any().any():raise RuntimeError("Panel F requires seven paired CAF patients")
    ax=container.subplots(1,1)
    boxes=ax.boxplot([paired.Bx,paired.Surgery],positions=[0,1],widths=.34,patch_artist=True,showfliers=False,medianprops={"color":"#222222","linewidth":1.3})
    for patch,color in zip(boxes["boxes"],["#377EB8","#E41A1C"]):patch.set_facecolor(color);patch.set_alpha(.22)
    jitter=np.linspace(-.075,.075,len(paired))
    for i,row in paired.iterrows():
        color="#D55E00" if row.Status=="Metastatic" else "#009E73"
        ax.plot([jitter[i],1+jitter[i]],[row.Bx,row.Surgery],color=color,lw=.9)
        ax.scatter([jitter[i],1+jitter[i]],[row.Bx,row.Surgery],s=34,c=["#377EB8","#E41A1C"],edgecolor=color,zorder=3)
    ax.set_xticks([0,1],["Biopsy","Surgery"]);ax.set_ylabel("Patient-median inferred CAF activity");_clean_axis(ax)
    _panel_title(container,label,"CAF selenocystathionine activity",f"{int(stats.patients_decreased)}/7 lower at surgery; exact paired p={stats.exact_two_sided_wilcoxon_p:.3f}.")
    container.text(.99,.01,"From D; colors indicate metastatic status. Exploratory activity estimate, not chemical abundance or a patient-level discovery.",ha="right",va="bottom",fontsize=5.2,color=COLORS["gray"])
    container.subplots_adjust(left=.12,right=.98,top=.70,bottom=.20)

def _draw_selenium_lcms(container,data,label="H"):
    values=data["sulfur_lcms_feature_values.csv"].copy();evidence=data["sulfur_lcms_feature_evidence.csv"].iloc[0];paired=values.pivot(index='PatientID_clean',columns='Timepoint',values='log2_intensity').reset_index()
    if len(paired)!=7 or not ((paired.Surgery-paired.Biopsy)<0).all() or bool(evidence.has_putative_native_match):raise RuntimeError("Panel H LC-MS evidence changed")
    ax=container.subplots(1,1);arrays=[paired.Biopsy.to_numpy(),paired.Surgery.to_numpy()];boxes=ax.boxplot(arrays,positions=[0,1],widths=.34,patch_artist=True,showfliers=False,medianprops={'color':'#222222','linewidth':1.3},boxprops={'edgecolor':'#555555','linewidth':.8},whiskerprops={'color':'#555555','linewidth':.8},capprops={'color':'#555555','linewidth':.8})
    for patch,color in zip(boxes['boxes'],['#377EB8','#E41A1C']):patch.set_facecolor(color);patch.set_alpha(.22)
    jitter=np.linspace(-.075,.075,7)
    for i,row in paired.iterrows():ax.plot([jitter[i],1+jitter[i]],[row.Biopsy,row.Surgery],color='#666666',lw=.9);ax.scatter([jitter[i],1+jitter[i]],[row.Biopsy,row.Surgery],s=34,c=['#377EB8','#E41A1C'],edgecolor='white',zorder=3)
    ax.set_xticks([0,1],['Biopsy','Surgery']);ax.set_ylabel('Specimen-level log2 intensity');_clean_axis(ax)
    ax.text(.03,.05,f"7/7 Surgery < Biopsy\nlog2FC={evidence.log2_fold_change:.3f}; p={evidence.nominal_p_value:.5f}; BH q={evidence.joint_BH_FDR:.4f}",transform=ax.transAxes,fontsize=6,bbox={'facecolor':'white','edgecolor':'#BBBBBB'})
    _panel_title(container,label,"Paired LC–MS m/z 215.0260","Selenium-compatible mass only; identity unconfirmed.")
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

def _draw_review_d1(container,data,label="D1"):
    selected=data["review_selected_mechanisms.csv"].copy().sort_values("mechanism_number")
    effects=data["review_pathway_four_view_effects_long.csv"].copy()
    nodes=data["review_graph_display_nodes.csv"].copy();edges=data["review_graph_display_edges.csv"].copy()
    expected=["Linoleate_metabolism","Ether_lipid_metabolism"]
    if selected["pathway"].tolist()!=expected or not selected["selected"].all():raise RuntimeError("D1 selected mechanisms changed")
    frame=effects.loc[effects["pathway"].isin(expected)].copy()
    if frame.groupby("pathway").size().to_dict()!={x:4 for x in expected}:raise RuntimeError("D1 requires four views per pathway")
    columns=[("MEIA","surgery_vs_biopsy","MEIA\nSurgery–Bx"),("GPR_GPA_specific","surgery_vs_biopsy","GPR-GPA\nSurgery–Bx"),("MEIA","metastatic_vs_nonmetastatic","MEIA\nMeta–non-meta"),("GPR_GPA_specific","metastatic_vs_nonmetastatic","GPR-GPA\nMeta–non-meta")]
    axes=container.subplots(1,3,gridspec_kw={"width_ratios":[1.65,1,1]})
    matrix=np.array([[frame.loc[frame.pathway.eq(pathway)&frame.method.eq(method)&frame.contrast.eq(contrast),"cohens_d"].iloc[0] for method,contrast,_ in columns] for pathway in expected])
    ax=axes[0];vmax=1.4;im=ax.imshow(matrix,aspect="auto",cmap="RdBu_r",vmin=-vmax,vmax=vmax,rasterized=True)
    ax.set_xticks(range(4),[x[2] for x in columns],fontsize=6);robust_labels=[f"{name}\nrank {int(row.robust_order)}; min LOPO {row.minimum_lopo_sign_retention:.2f}" for name,row in zip(["Linoleate metabolism","Ether-lipid metabolism"],selected.itertuples(index=False))];ax.set_yticks(range(2),robust_labels,fontsize=6.3)
    for i in range(2):
        for j in range(4):ax.text(j,i,f"{matrix[i,j]:+.2f}",ha="center",va="center",fontsize=6.3,color="white" if abs(matrix[i,j])>.75 else COLORS["dark"],fontweight="bold")
    ax.tick_params(length=0);[sp.set_visible(False) for sp in ax.spines.values()]
    cb=container.colorbar(im,ax=ax,fraction=.035,pad=.025);cb.set_label("Cell-level Cohen’s d",fontsize=7);cb.ax.tick_params(labelsize=6,length=2)
    motifs=[(axes[1],expected[0],"Linoleate motif","Linoleate","MAR02455\nCYP family","9(10)-EpOME"),(axes[2],expected[1],"Ether-lipid motif","Plasmenyl-PE","MAR07603\nPLA2/PLB1/PLAAT","Fatty-acid pool")]
    for axis,pathway,title,left,middle,right in motifs:
        if pathway not in set(nodes.pathway) or pathway not in set(edges.pathway):raise RuntimeError(f"Missing graph context for {pathway}")
        axis.set_xlim(0,1);axis.set_ylim(0,1);axis.axis("off");axis.set_title(title,fontsize=7,fontweight="bold",pad=3)
        xs=[.13,.50,.87];labels=[left,middle,right];colors=["#DDEBF7","#FCE4D6","#E2F0D9"]
        for x0,txt,color in zip(xs,labels,colors):axis.text(x0,.52,txt,ha="center",va="center",fontsize=5.8,bbox={"boxstyle":"round,pad=.32","facecolor":color,"edgecolor":COLORS["gray"],"linewidth":.7})
        axis.annotate("",xy=(.39,.52),xytext=(.25,.52),arrowprops={"arrowstyle":"->","color":COLORS["gray"],"lw":.9});axis.annotate("",xy=(.76,.52),xytext=(.63,.52),arrowprops={"arrowstyle":"->","color":COLORS["gray"],"lw":.9})
        axis.text(.5,.13,"Human-GEM context; gene families collapsed",ha="center",va="center",fontsize=5.2,color=COLORS["gray"])
    _panel_title(container,label,"Robust MEIA/GPR-GPA lipid mechanisms","Two deterministic nonredundant pathways; four-view effects and graph context.")
    container.text(.99,.01,"Cell-level effects are exploratory; leave-one-patient-out sign retention is an influence diagnostic, not patient-level inference.",ha="right",va="bottom",fontsize=5.4,color=COLORS["gray"])
    container.subplots_adjust(left=.12,right=.98,top=.74,bottom=.22,wspace=.34)

def _draw_review_e1(container,data,label="E1"):
    stats=data["selected_metabolite_statistics.csv"].copy();stats=stats.loc[stats.panel.eq("C")].copy()
    values=data["selected_metabolite_values.csv"].copy();values=values.loc[values.panel.eq("C")].copy()
    order=["pos_174.9854_1689.69","neg_246.9429_242.71"]
    stats=stats.loc[stats.feature.isin(order)].copy();values=values.loc[values.feature.isin(order)].copy()
    if stats.feature.tolist()!=order:stats=stats.set_index("feature").loc[order].reset_index()
    if stats.feature.tolist()!=order or values.groupby("feature").PatientID_clean.nunique().to_dict()!={x:7 for x in order}:raise RuntimeError("E1 paired LC-MS inputs changed")
    pmap={"P1":"P1","P2":"P2","P3":"P3","P4":"P4","P5":"P5","P6":"P6","P7":"P7"};pcolors=dict(zip([f"P{i}" for i in range(1,8)],sns.color_palette("Set2",7)))
    axes=np.atleast_1d(container.subplots(1,2)).ravel()
    for idx,(ax,row) in enumerate(zip(axes,stats.itertuples(index=False))):
        f=values.loc[values.feature.eq(row.feature)].copy();f["patient_display"]=f.PatientID.map(pmap)
        wide=f.pivot(index="patient_display",columns="Timepoint",values="log2_intensity").reindex([f"P{i}" for i in range(1,8)])
        for patient,r in wide.iterrows():ax.plot([0,1],[r.Biopsy,r.Surgery],color=pcolors[patient],lw=.8,alpha=.85,zorder=1);ax.scatter([0,1],[r.Biopsy,r.Surgery],color=pcolors[patient],s=13,zorder=2,edgecolor="white",linewidth=.3)
        ax.boxplot([wide.Biopsy.to_numpy(),wide.Surgery.to_numpy()],positions=[0,1],widths=.35,showfliers=False,patch_artist=True,boxprops={"facecolor":"white","edgecolor":COLORS["dark"],"linewidth":.7},medianprops={"color":COLORS["dark"],"linewidth":1},whiskerprops={"color":COLORS["gray"],"linewidth":.6},capprops={"color":COLORS["gray"],"linewidth":.6})
        title=row.display_name.split("\n")[0];ax.set_title(title,loc="left",fontsize=7,fontweight="bold",pad=2);ax.set_xticks([0,1],["Biopsy","Surgery"]);ax.set_xlabel("");ax.set_ylabel("log2 intensity" if idx==0 else "")
        ax.text(.98,.98,f"m/z {row.mz:.4f}; RT {row.rt_seconds:.2f} s\nlog2FC={row.log2_fold_change:+.3f}; q={row.joint_BH_FDR:.4f}",transform=ax.transAxes,ha="right",va="top",fontsize=5.3,bbox={"facecolor":"white","edgecolor":"none","alpha":.78,"pad":1})
        _clean_axis(ax)
    handles=[mpl.lines.Line2D([0],[0],marker="o",color=pcolors[f"P{i}"],lw=0,markersize=4,label=f"P{i}") for i in range(1,8)];axes[-1].legend(handles=handles,frameon=False,ncol=4,fontsize=5.3,loc="lower right")
    _panel_title(container,label,"Patient-paired LC-MS pathway drivers","Seven paired patients; two unannotated blank-corrected features; patient-blocked statistics.")
    container.subplots_adjust(left=.07,right=.99,top=.74,bottom=.19,wspace=.25)

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

def _review_standalone(label,data):
    dims={"D1":(180,92),"E1":(180,84),"S1":(180,82),"G1":(180,125)}
    w,h=dims[label];fig=plt.figure(figsize=(_mm(w),_mm(h)),facecolor="white")
    {"D1":_draw_review_d1,"E1":_draw_review_e1,"S1":_draw_review_s1,"G1":_draw_panel_d_graph_guided}[label](fig,data,label)
    return fig

def make_figure6_candidate_main(data):
    fig=plt.figure(figsize=(_mm(MAIN_WIDTH_MM),_mm(380)),facecolor="white");rows=fig.subfigures(4,1,height_ratios=[1.05,.83,1.0,.88],hspace=.05)
    top=rows[0].subfigures(1,2,width_ratios=[.45,.55],wspace=.06);_draw_four_set_upset(top[0],data,"A","treatment_four_set_membership.csv","Treatment four-set metabolic overlap");_draw_treatment_pathway_scatter(top[1],data,"B")
    _draw_pooled_concordance(rows[1],data,"C");_draw_review_d1(rows[2],data,"D1");_draw_review_e1(rows[3],data,"E1");return fig

def make_figure6_candidate_supplementary(data):
    fig=plt.figure(figsize=(_mm(SUPPLEMENT_WIDTH_MM),_mm(540)),facecolor="white");rows=fig.subfigures(5,1,height_ratios=[.78,.92,.80,.72,.72],hspace=.045)
    _draw_four_set_upset(rows[0],data,"SA","status_four_set_membership.csv","Metastatic-status four-set metabolic overlap")
    _draw_status_pathway_scatter(rows[1],data,"SB")
    _draw_entity_concordance(rows[2],data,"SJ","gpa");_draw_review_s1(rows[3],data,"S1");_draw_robust_trajectories(rows[4],data,"D");return fig

def _standalone(label,data):
    dims={"A":(112,92),"B":(112,112),"C":(112,88),"D":(180,82),"E":(112,88),"F":(180,88),"G":(112,88),"H":(180,120),"SA":(112,92),"SB":(112,112),"SJ":(112,92)}
    w,h=dims[label];fig=plt.figure(figsize=(_mm(w),_mm(h)),facecolor="white")
    drawers={
      "A":lambda c:_draw_four_set_upset(c,data,"A","treatment_four_set_membership.csv","Treatment four-set metabolic overlap"),
      "B":lambda c:_draw_treatment_pathway_scatter(c,data,"B"),
      "C":lambda c:_draw_pooled_concordance(c,data,"C"),
      "D":lambda c:_draw_robust_trajectories(c,data,"D"),
      "E":lambda c:_draw_pathway_context_pie(c,data,"E"),
      "F":lambda c:_draw_caf_selenocystathionine(c,data,"F"),
      "G":lambda c:_draw_selenium_lcms(c,data,"G"),
      "H":lambda c:_draw_ng8_graph(c,data,"H"),
      "SA":lambda c:_draw_four_set_upset(c,data,"SA","status_four_set_membership.csv","Metastatic-status four-set metabolic overlap"),
      "SB":lambda c:_draw_status_pathway_scatter(c,data,"SB"),
      "SJ":lambda c:_draw_entity_concordance(c,data,"SJ","gpa"),
    }
    drawers[label](fig);return fig

def make_figure6_main_panel(label,data=Path("data/Figure_6")):
    if label not in set("ABCDEFGH"):raise ValueError(label)
    return _standalone(label,data)

def make_figure6_supplementary_panel(label,data=Path("data/Figure_6")):
    if label not in {"SA","SB","SJ"}:raise ValueError(label)
    return _standalone(label,data)

def make_figure6_compiled(data):
    # Wide tall page. Four rows preserve the requested story hierarchy without horizontal crowding.
    fig=plt.figure(figsize=(_mm(297),_mm(380)),facecolor="white")
    rows=fig.subfigures(4,1,height_ratios=[.86,.98,.90,1.45],hspace=.035)
    top=rows[0].subfigures(1,3,width_ratios=[1,1,1],wspace=.045)
    _draw_four_set_upset(top[0],data,"A","treatment_four_set_membership.csv","Treatment four-set metabolic overlap")
    _draw_treatment_pathway_scatter(top[1],data,"B")
    _draw_pooled_concordance(top[2],data,"C")
    _draw_robust_trajectories(rows[1],data,"D")
    evidence=rows[2].subfigures(1,3,width_ratios=[.95,1.62,.93],wspace=.045)
    _draw_pathway_context_pie(evidence[0],data,"E")
    _draw_caf_selenocystathionine(evidence[1],data,"F")
    _draw_selenium_lcms(evidence[2],data,"G")
    _draw_ng8_graph(rows[3],data,"H")
    return fig

def _draw_su3_tasks(container,data,label="SU3"):
    cells=data["sulfur_p3_all_score_cells.csv.gz"].copy()
    sites=["Bx","Surgery","Recurrence"]
    celltypes=["Epithelial","Endothelial","Macrophage","T cells","CAF","PVL"]
    if cells.score_label.nunique()!=9 or not cells.groupby(["score_label","cell_type_analysis"]).Site.nunique().eq(3).all():raise RuntimeError("SU3 coverage changed")
    stats=data["sulfur_p3_task_statistics.csv"].copy()
    if len(stats)!=162:raise RuntimeError("SU3 requires the frozen reviewed-label 162-test family")
    lookup=stats.set_index(["comparison","label","ct"])
    tasks=["MEIA task | Homocysteine degradation","MEIA task | Glutathione synthesis"]
    colors={"Bx":"#377EB8","Surgery":"#E41A1C","Recurrence":"#984EA3"}
    axes=np.asarray(container.subplots(2,6)).reshape(2,6)
    def probability(value):return "<4.9e−324" if value==0 else f"{value:.2e}"
    for row_index,score in enumerate(tasks):
        for column_index,celltype in enumerate(celltypes):
            ax=axes[row_index,column_index];frame=cells.query("score_label==@score and cell_type_analysis==@celltype");values=[frame.loc[frame.Site.eq(site),"score"].to_numpy() for site in sites]
            violin=ax.violinplot(values,range(3),widths=.7,showextrema=False)
            for body,site in zip(violin["bodies"],sites):body.set_facecolor(colors[site]);body.set_edgecolor(colors[site]);body.set_alpha(.2)
            ax.boxplot(values,range(3),widths=.28,patch_artist=True,showfliers=False,medianprops={"color":"#222222","linewidth":1.2},boxprops={"facecolor":"white","edgecolor":"#666666","linewidth":.8},whiskerprops={"color":"#666666","linewidth":.7},capprops={"color":"#666666","linewidth":.7})
            means=[np.mean(value) for value in values];ax.plot(range(3),means,color="#333333",marker="D",ms=3,lw=.8)
            stat=lookup.loc[("Recurrence vs Surgery",score,celltype)]
            ax.set_title(celltype,fontsize=7.0,fontweight="bold",pad=15)
            ax.text(.5,1.01,f"S→R {stat.change:+.0f}% | d={stat.d:+.1f}\nq={probability(stat.q162)}",transform=ax.transAxes,ha="center",va="bottom",fontsize=4.7,fontweight="bold")
            ax.set_xticks(range(3),[f"{site}\nn={len(value):,}" for site,value in zip(sites,values)],fontsize=6.5);ax.spines[["top","right"]].set_visible(False);ax.grid(False)
            if column_index==0:ax.set_ylabel("Cell-level activity score")
        axes[row_index,0].text(-.5,.5,score.replace("MEIA task | ",""),transform=axes[row_index,0].transAxes,rotation=90,ha="center",va="center",fontsize=8,fontweight="bold")
    container.suptitle(f"{label}  P3 homocysteine-degradation and glutathione-synthesis trajectories",x=.01,y=.985,ha="left",fontweight="bold",fontsize=PANEL_LABEL_SIZE_PT)
    container.text(.5,.018,"Violin width=density; white box=IQR; line=median; whiskers=1.5×IQR; diamond/line=mean; box outliers hidden.",ha="center",fontsize=6.5,color="#555555")
    container.text(.5,.004,"Welch tests; BH across reviewed-label 162 tests. One patient; cells are pseudoreplicates.",ha="center",fontsize=6.5,color="#555555")
    container.subplots_adjust(left=.07,right=.995,top=.86,bottom=.14,hspace=.72,wspace=.38)


def make_figure6_supplementary(data):
    # The four paired LC-MS features follow the seven existing context panels.
    fig=plt.figure(figsize=(_mm(297),_mm(590)),facecolor="white")
    rows=fig.subfigures(5,1,height_ratios=[1.02,.83,.90,1.82,1.28],hspace=.035)
    top=rows[0].subfigures(1,3,width_ratios=[1,1,1],wspace=.045)
    _draw_four_set_upset(top[0],data,"A","status_four_set_membership.csv","Metastatic-status four-set metabolic overlap")
    _draw_status_pathway_scatter(top[1],data,"B")
    _draw_entity_concordance(top[2],data,"C","gpa")
    _draw_review_s1(rows[1],data,"D")
    middle=rows[2].subfigures(1,2,width_ratios=[.44,.56],wspace=.055)
    _draw_filtering_cascade(middle[0],data,"E")
    _draw_homocysteine_screen(middle[1],data,"F")
    _draw_su3_tasks(rows[3],data,"G")
    _draw_moved_lcms(rows[4],data)
    return fig




MAIN_INPUTS={
"A":["treatment_four_set_membership.csv"],
"B":["surgery_pathway_scatter.csv","surgery_pathway_enrichment.csv","treatment_four_set_membership.csv"],
"C":["meia_vs_scknife_pooled_correlations.csv","meia_vs_scknife_pooled_density.csv.gz"],
"D":["robust_tier3_trajectory_selection.csv","robust_tier3_trajectory_values.csv.gz"],
"E":["robust_metabolite_broad_entity_mapping.tsv","robust_metabolite_broad_entity_counts.csv"],
"F":["caf_selenocystathionine_patient_medians.csv","caf_selenocystathionine_patient_statistics.csv"],
"G":["sulfur_lcms_feature_values.csv","sulfur_lcms_feature_evidence.csv"],
"H":["sulfur_ng8_nodes.csv","sulfur_ng8_edges.csv","sulfur_ng8_anchors.csv"]}
SUPPLEMENT_INPUTS={
"A":["status_four_set_membership.csv"],
"B":["status_pathway_scatter.csv","status_pathway_enrichment.csv","status_four_set_membership.csv"],
"C":["gpr_gpa_vs_scknife_entity_correlations.csv"],
"D":["review_whole_profile_patient_summary.csv","review_whole_profile_concordance_summary.csv"],
"E":["robust_metabolite_filtering_funnel.csv"],
"F":["sulfur_cohort_patient_values.csv","sulfur_cohort_treatment_statistics.csv"],
"G":["sulfur_p3_all_score_cells.csv.gz","sulfur_p3_task_statistics.csv"],
"H":["cohort_lcms_patient_changes.csv"],
"I":["cohort_lcms_patient_changes.csv"],
"J":["cohort_lcms_patient_changes.csv"],
"K":["cohort_lcms_patient_changes.csv"]}

def _bridge_module():
    import importlib.util
    path=Path(__file__).resolve().parents[2]/"Figure_6/patient_bridge_plotting.py"
    spec=importlib.util.spec_from_file_location("figure6_patient_bridge",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _draw_moved_lcms(container,data):
    bridge=_bridge_module()
    bridge.panel_d(container,data,label="",plot_letters=("H","I","J","K"))

def _draw_moved_lcms_single(container,data,label):
    bridge=_bridge_module()
    feature=bridge.LCMS["HIJK".index(label)]
    frame=data["cohort_lcms_patient_changes.csv"].loc[lambda x:x.feature.eq(feature)]
    row=frame.iloc[0]
    family={"H":"Methionine/cysteine association","I":"Methionine/cysteine association",
            "J":"Purine association","K":"Selenoamino acid association"}[label]
    lower=int(frame.delta.lt(0).sum())
    direction=f"{lower}/7 lower" if lower>=4 else f"{7-lower}/7 higher"
    ax=container.subplots(1,1)
    bridge.paired_boxplot(ax,frame,"biopsy_log2_intensity","surgery_log2_intensity",
        ylabel="LC–MS log₂ intensity",title=f"{label}  m/z {row.mz:.4f} · RT {row.rt_seconds:.2f} s",
        detail=f"{family}\nq={row.treatment_q:.3g}; CV={row.technical_cv_percent:.0f}%; {direction}",detail_side="above")
    container.text(.98,.02,"Candidate pathway association; identity unconfirmed. ○ metastatic; □ non-metastatic.",
                   ha="right",fontsize=6.2,color="#555555")
    container.subplots_adjust(left=.18,right=.96,top=.72,bottom=.22)

def load_tables(data_root, inputs):
    import hashlib
    import json
    root = Path(data_root)
    manifest = {entry["path"]: entry for entry in json.loads((root / "manifest.json").read_text())["files"]}
    tables = {}
    for name in sorted(set(sum(inputs.values(), []))):
        path = root / name
        payload = path.read_bytes()
        record = manifest[name]
        if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Supplementary Figure S6 checksum mismatch: {name}")
        tables[name] = pd.read_csv(path, sep="\t" if name.endswith(".tsv") else ",")
    return tables

def save_figure(fig, stem, dpi=300):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=dpi, bbox_inches=None, pad_inches=0)

def make_figure6_supplementary_panel(label, data):
    if label not in SUPPLEMENT_INPUTS:
        raise ValueError(label)
    height = 174 if label == "G" else 96
    fig = plt.figure(figsize=(_mm(230 if label in "HIJK" else 150), _mm(height)), facecolor="white")
    {
        "A": lambda: _draw_four_set_upset(fig, data, "A", "status_four_set_membership.csv", "Metastatic-status four-set metabolic overlap"),
        "B": lambda: _draw_status_pathway_scatter(fig, data, "B"),
        "C": lambda: _draw_entity_concordance(fig, data, "C", "gpa"),
        "D": lambda: _draw_review_s1(fig, data, "D"),
        "E": lambda: _draw_filtering_cascade(fig, data, "E"),
        "F": lambda: _draw_homocysteine_screen(fig, data, "F"),
        "G": lambda: _draw_su3_tasks(fig, data, "G"),
        "H": lambda: _draw_moved_lcms_single(fig,data,"H"),
        "I": lambda: _draw_moved_lcms_single(fig,data,"I"),
        "J": lambda: _draw_moved_lcms_single(fig,data,"J"),
        "K": lambda: _draw_moved_lcms_single(fig,data,"K"),
    }[label]()
    return fig
