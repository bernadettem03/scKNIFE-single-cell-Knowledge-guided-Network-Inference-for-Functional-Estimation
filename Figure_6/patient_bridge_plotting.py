"""Paired patient boxplots for corrected LC-MS, inferred scKNIFE and PDGF."""
from __future__ import annotations

import numpy as np

PATIENTS = [f"P{i}" for i in range(1, 8)]
LCMS = ["pos_104.9996_205.94", "neg_188.9273_269.18", "neg_178.9518_209.60", "pos_174.9854_1689.69"]
NODES = [("CAF", "H2S"), ("CAF", "homocysteine"), ("CAF", "L_cystathionine"),
         ("Endothelial", "adenine"), ("Endothelial", "deoxyinosine")]
BX_COLOR = "#377EB8"
SURGERY_COLOR = "#D55E00"
INK = "#30373B"
PATIENT_LINE = "#8D969B"


def paired_boxplot(ax, frame, biopsy, surgery, *, ylabel, title, detail="", detail_side="left", highlight=None):
    """Box/IQR/min-max and all paired biological replicates, in fixed patient order."""
    ordered = frame.set_index("patient_display").loc[PATIENTS]
    bx = ordered[biopsy].to_numpy(float)
    surg = ordered[surgery].to_numpy(float)
    assert len(bx) == len(surg) == 7 and np.isfinite(bx).all() and np.isfinite(surg).all()
    assert set(ordered.Status) == {"Metastatic", "Non-metastatic"}
    offsets = np.linspace(-.075, .075, 7)
    for i, (patient, row) in enumerate(ordered.iterrows()):
        x = offsets[i]
        ax.plot([x, 1+x], [row[biopsy], row[surgery]], color=PATIENT_LINE,
                lw=.9, alpha=.72, zorder=1)
    bp = ax.boxplot([bx, surg], positions=[0, 1], widths=.34, whis=(0, 100),
                    patch_artist=True, showfliers=False, zorder=2,
                    boxprops={"edgecolor": INK, "linewidth": .8},
                    whiskerprops={"color": INK, "linewidth": .8},
                    capprops={"color": INK, "linewidth": .8},
                    medianprops={"color": INK, "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], [BX_COLOR, SURGERY_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(.35)
    for i, (patient, row) in enumerate(ordered.iterrows()):
        marker = "o" if row.Status == "Metastatic" else "s"
        ax.scatter([offsets[i], 1+offsets[i]], [row[biopsy], row[surgery]],
                   marker=marker, s=20, facecolor="white", edgecolor=INK,
                   linewidth=.85, zorder=4)
    extent = max(np.ptp(np.r_[bx, surg]), .05)
    bottom = min(np.r_[bx, surg]) - extent*.15
    top = max(np.r_[bx, surg]) + extent*.22
    ax.set_ylim(bottom, top)
    ax.set_xlim(-.32, 1.32)
    ax.set_xticks([0, 1], ["Biopsy", "Surgery"])
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, loc="left", fontsize=7.6, fontweight="bold",
                 pad=32 if detail_side == "above" else 8)
    if detail:
        if detail_side == "above":
            ax.text(.0, 1.02, detail, transform=ax.transAxes, fontsize=6.1,
                    color="#4C555A", ha="left", va="bottom")
        else:
            x, align = (.98, "right") if detail_side == "right" else (.02, "left")
            ax.text(x, .97, detail, transform=ax.transAxes, fontsize=6.1,
                    color="#4C555A", ha=align, va="top")
    if highlight:
        patient, label = highlight
        row = ordered.loc[patient]
        x = 1 + offsets[PATIENTS.index(patient)]
        ax.annotate(label, (x, row[surgery]), xytext=(4, 4), textcoords="offset points",
                    fontsize=6.3, fontweight="bold", color=INK, zorder=5)
    ax.tick_params(axis="both", labelsize=6.5, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def panel_d(container, data, label="D", plot_letters=None):
    lcms = data["cohort_lcms_patient_changes.csv"]
    axes = np.asarray(container.subplots(2, 2)).ravel()
    family = {"pos_104.9996_205.94": "Methionine/cysteine association",
              "neg_188.9273_269.18": "Methionine/cysteine association",
              "neg_178.9518_209.60": "Purine association",
              "pos_174.9854_1689.69": "Selenoamino acid association"}
    for index, (ax, feature) in enumerate(zip(axes, LCMS)):
        frame = lcms.loc[lcms.feature.eq(feature)]
        row = frame.iloc[0]
        decreases = int(frame.delta.lt(0).sum())
        direction = f"{decreases}/7 lower" if decreases >= 4 else f"{7-decreases}/7 higher"
        paired_boxplot(ax, frame, "biopsy_log2_intensity", "surgery_log2_intensity",
            ylabel="LC–MS log₂ intensity", title=f"{plot_letters[index] + '  ' if plot_letters else ''}m/z {row.mz:.4f} · RT {row.rt_seconds:.2f} s",
            detail=f"{family[feature]}\nq={row.treatment_q:.3g}; CV={row.technical_cv_percent:.0f}%; {direction}",
            detail_side="above")
    if label:
        container.suptitle(f"{label}  Corrected LC–MS features across seven paired patients",
                           x=.02, ha="left", fontsize=10, fontweight="bold")
    container.text(.99, .01, "Candidate pathway associations; identities unconfirmed. All 14 specimen values observed.  ○ metastatic; □ non-metastatic.",
                   ha="right", va="bottom", fontsize=6.2, color="#555555")
    container.subplots_adjust(left=.12, right=.98, top=.85, bottom=.13, hspace=.62, wspace=.34)


def panel_e(container, data, label="E", plot_letters=None):
    sck = data["cohort_scknife_patient_changes.csv"]
    bridge = data["endothelial_pdgf_patient_alignment.csv"]
    axes = np.asarray(container.subplots(2, 3)).ravel()
    names = {"H2S": "H₂S", "homocysteine": "Homocysteine",
             "L_cystathionine": "L-cystathionine", "adenine": "Adenine", "deoxyinosine": "Deoxyinosine"}
    for index, (ax, (cell, node)) in enumerate(zip(axes[:5], NODES)):
        frame = sck.loc[sck.celltype.eq(cell) & sck.node.eq(node)]
        tier = int(frame.tier.iloc[0])
        decreases = int(frame.delta.lt(0).sum())
        paired_boxplot(ax, frame, "biopsy_median", "surgery_median",
            ylabel="Inferred activity", title=f"{plot_letters[index] + '  ' if plot_letters else ''}{cell} · {names[node]} · Tier {tier}",
            detail=f"{decreases}/7 patient medians lower",
            highlight=("P2", "P2 ↑") if node == "H2S" else None)
    pdgf = bridge.loc[bridge.node.eq("adenine")]
    assert len(pdgf) == 7 and pdgf.PatientID.nunique() == 7
    paired_boxplot(axes[5], pdgf, "pdgf_biopsy_log2_score", "pdgf_surgery_log2_score",
        ylabel="log₂ PDGF score", title=f"{plot_letters[5] + '  ' if plot_letters else ''}Endothelial→PVL · PDGF",
        detail="6/7 paired scores lower", highlight=("P1", "P1 ↑"))
    container.suptitle(f"{label + '  ' if label else ''}Inferred cell-type activity and Figure 7 PDGF context",
                       x=.02, ha="left", fontsize=10, fontweight="bold")
    container.text(.99, .01, "Boxes and points summarize seven patients. scKNIFE panels use patient cell medians; PDGF is a distinct communication score.  ○ metastatic; □ non-metastatic.",
                   ha="right", va="bottom", fontsize=6.2, color="#555555")
    container.subplots_adjust(left=.08, right=.99, top=.85, bottom=.14, hspace=.66, wspace=.46)


def panel_h(container, data, label="H", plot_letters=None):
    """P3 cell-score distributions with connected medians for nested CAF labels."""
    frame = data["p3_caf_major_minor_trajectories.csv"]
    cells = data["p3_caf_major_minor_cell_values.csv.gz"]
    nodes = [
        ("tRNA_containing_6_isopentenyladenosine", "tRNA containing\n6-isopentenyladenosine"),
        ("selenocystathionine", "Selenocystathionine"),
        ("2_phosphoglycolate", "2-phosphoglycolate"),
    ]
    timepoints = ["Bx", "Surgery", "Recurrence"]
    groups = [
        ("Major CAF", "#234A69", "-", -.16),
        ("Reviewed myCAF-like s5", "#B66A2B", "--", .16),
    ]
    axes = np.asarray(container.subplots(1, 3)).ravel()
    for index, (ax, (node, title)) in enumerate(zip(axes, nodes)):
        observed = frame.loc[frame.metabolite.eq(node)]
        node_cells = cells.loc[cells.metabolite.eq(node)]
        for group, color, style, offset in groups:
            series = observed.loc[observed.group.eq(group)].set_index("Timepoint").loc[timepoints]
            assert len(series) == 3 and series.PatientID.eq("P3").all()
            samples = [node_cells.loc[node_cells.group.eq(group) & node_cells.Timepoint.eq(t), "activity"].to_numpy(float)
                       for t in timepoints]
            assert [len(v) for v in samples] == series.n_cells.tolist()
            assert all(np.all(v > 0) for v in samples)
            x = np.arange(3) + offset
            violins = ax.violinplot([np.log10(v) for v in samples], positions=x, widths=.28,
                                    showmeans=False, showmedians=False, showextrema=False)
            for body in violins["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_linewidth(.6)
                body.set_alpha(.24)
            ax.plot(x, np.log10(series.median_activity.to_numpy(float)), linestyle=style,
                    marker="o", markersize=3.8, linewidth=1.7, color=color, label=group, zorder=4)
        ax.set_title(f"{plot_letters[index] + '  ' if plot_letters else ''}{title}", fontsize=8.2, fontweight="bold", pad=9)
        ax.set_xticks(range(3), ["Biopsy", "Surgery", "Recurrence"])
        ax.set_xlim(-.48, 2.48)
        ax.set_ylabel("log₁₀ inferred activity per cell", fontsize=7)
        ax.tick_params(axis="both", labelsize=6.6, length=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        values = np.log10(node_cells.activity.to_numpy(float))
        spread = max(np.ptp(values), .03)
        ax.set_ylim(values.min() - .04*spread, values.max() + .06*spread)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.35), ncol=2, frameon=False, fontsize=6.6)
    container.suptitle(f"{label + '  ' if label else ''}P3 CAF inferred-activity trajectories", x=.02, ha="left",
                       fontsize=10, fontweight="bold")
    container.text(.99, .01,
        "Violins: log₁₀ cell scores; lines: log₁₀ of medians. One patient (P3); myCAF-like s5 is nested within major CAF. Cells at Bx/Surgery/Recurrence: major 154/317/790; subset 142/92/210.",
        ha="right", va="bottom", fontsize=6.0, color="#555555")
    container.subplots_adjust(left=.075, right=.99, top=.78, bottom=.30, wspace=.32)
