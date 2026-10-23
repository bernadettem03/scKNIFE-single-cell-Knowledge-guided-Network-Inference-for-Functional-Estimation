"""Plot-only renderer for the two-angle Figure 6 package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patient_bridge_plotting import paired_boxplot

STATUS = {"Metastatic": "#E41A1C", "Non-metastatic": "#4DAF4A"}
SITE = {"Biopsy": "#377EB8", "Surgery": "#E41A1C", "Recurrence": "#984EA3"}
PATIENT_ORDER = [f"P{i}" for i in range(1, 8)]
FEATURE_ORDER = ["pos_133.0811_1285.72", "pos_258.2420_159.36"]
FAMILY_SHORT = {
    "Aromatic amino acids and biogenic amines": "Aromatic AA / amines",
    "Fatty acids and beta-oxidation intermediates": "Fatty acids / beta oxidation",
    "Glycerophospholipids and lysophospholipids": "Glycerophospholipids",
    "Glycosaminoglycans, glycosphingolipids and glycans": "Glycans / glycosphingolipids",
    "Nucleotides, nucleic acids and RNA modification": "Nucleotides / RNA",
    "Vitamins, cofactors, retinoids and isoprenoids": "Vitamins / cofactors",
    "Xenobiotics, pharmaceuticals and environmental chemicals": "Xenobiotics",
    "Other families": "Other families",
}


def load_data(root: Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    manifest = json.loads((root / "two_angle_manifest.json").read_text())
    out = {}
    for name, evidence in manifest["files"].items():
        payload = (root / name).read_bytes()
        if len(payload) != evidence["bytes"] or hashlib.sha256(payload).hexdigest() != evidence["sha256"]:
            raise RuntimeError(f"Figure 6 data checksum mismatch: {name}")
        out[name.removesuffix(".csv")] = pd.read_csv(root / name)
    return out


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2, "axes.titlesize": 9,
        "axes.labelsize": 7.2, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def title(ax, letter, heading, subtitle=""):
    ax.set_title(f"{letter}  {heading}", loc="left", fontweight="bold", fontsize=9, pad=10)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, va="bottom", fontsize=6.3, color="#555555")


def panel_a(ax, d):
    patients = d["two_angle_feature_values"].drop_duplicates("PatientID").sort_values("patient_display")
    for idx, row in enumerate(patients.itertuples(index=False)):
        y = 6 - idx
        ax.plot([0, 1], [y, y], color=STATUS[row.Status], lw=2, alpha=.8)
        ax.scatter([0, 1], [y, y], s=36, color=[SITE["Biopsy"], SITE["Surgery"]], edgecolors=STATUS[row.Status], linewidths=1.4, zorder=3)
        ax.text(-.15, y, row.patient_display, va="center", ha="right", fontweight="bold")
        if row.patient_display == "P3":
            ax.scatter([1.8], [y], marker="D", s=31, color=SITE["Recurrence"], zorder=3)
            ax.plot([1, 1.8], [y, y], ls=":", lw=1, color="#777777")
    ax.set_xlim(-.42, 2.2); ax.set_ylim(-.65, 6.8)
    ax.set_xticks([0, 1, 1.8], ["Biopsy", "Surgery", "P3 recurrence"])
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    handles = [mpl.lines.Line2D([], [], color=color, lw=2.5, label=f"{status} ({4 if status == 'Metastatic' else 3})") for status, color in STATUS.items()]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=6.3)
    title(ax, "A", "Paired TNBC tissue sampling", "Patient is the biological replicate")


def panel_b(ax, d):
    frame = d["two_angle_treatment_pathways"].sort_values("pathway_BH_FDR_gamma", ascending=False)
    labels = frame.candidate_pathway.str.replace(" and ", " / ", regex=False)
    y = np.arange(len(frame))
    ax.barh(y, -np.log10(frame.pathway_BH_FDR_gamma), color="#377EB8", alpha=.82, height=.65)
    ax.set_yticks(y, labels)
    ax.set_xlabel("−log₁₀ pathway BH q")
    for yy, q in zip(y, frame.pathway_BH_FDR_gamma):
        ax.text(-np.log10(q)+.025, yy, f"q={q:.3f}", va="center", fontsize=6.2)
    ax.set_xlim(0, 1.75)
    title(ax, "B", "Treatment-associated pathways", "Five corrected LC–MS pathway discoveries")


def _feature_label(feature):
    return "m/z 133.0811" if feature == FEATURE_ORDER[0] else "m/z 258.2420"


def panel_c(container, d, label="C", plot_letters=None):
    axes = container.subplots(1, 2)
    values = d["two_angle_feature_changes"]
    summary = d["two_angle_feature_summary"].set_index("feature")
    for index, (ax, feature) in enumerate(zip(axes, FEATURE_ORDER)):
        frame = values.loc[values.feature.eq(feature)]
        s = summary.loc[feature]
        paired_boxplot(ax, frame, "Biopsy", "Surgery", ylabel="LC–MS log₂ intensity",
            title=f"{plot_letters[index] + '  ' if plot_letters else ''}{_feature_label(feature)}",
            detail=f"7/7 lower; treatment q={s.treatment_q:.4f}\nCV={s.technical_cv_percent:.1f}%")
    container.suptitle(f"{label + '  ' if label else ''}Paired response of two unidentified LC–MS features",
                       x=.02, ha="left", fontsize=9, fontweight="bold")
    container.text(.99, .01, "Boxes summarize seven patients; each line joins one biopsy and surgery specimen.  ○ metastatic; □ non-metastatic.",
                   ha="right", va="bottom", fontsize=6.1, color="#555555")
    container.subplots_adjust(left=.12, right=.99, top=.78, bottom=.19, wspace=.32)


def panel_d(container, d, label="D", plot_letters=None):
    axes = container.subplots(1, 2, sharex=True)
    values = d["two_angle_feature_changes"]
    summary = d["two_angle_feature_summary"].set_index("feature")
    for index, (ax, feature) in enumerate(zip(axes, FEATURE_ORDER)):
        frame = values.loc[values.feature.eq(feature)]
        for j, status in enumerate(["Non-metastatic", "Metastatic"]):
            subset = frame.loc[frame.Status.eq(status)].sort_values("patient_display")
            offsets = np.linspace(-.09, .09, len(subset))
            ax.scatter(np.full(len(subset), j)+offsets, subset.delta, s=37, color=STATUS[status], edgecolor="white", lw=.5, zorder=3)
            ax.plot([j-.16, j+.16], [subset.delta.median()]*2, color="#222222", lw=1.4)
            for offset, row in zip(offsets, subset.itertuples(index=False)):
                ax.text(j+offset, row.delta+.055, row.patient_display, ha="center", va="bottom", fontsize=5.3)
        s = summary.loc[feature]
        ax.axhline(0, color="#888888", lw=.7)
        ax.set_xticks([0, 1], ["Non-meta\nn=3", "Meta\nn=4"])
        ax.set_title(f"{plot_letters[index] + '  ' if plot_letters else ''}{_feature_label(feature)}", loc="left", fontsize=8, fontweight="bold")
        ax.text(.02, .97, f"Status model p={s.status_p:.4f}\nΔ exact p={s.delta_exact_two_sided_p:.4f}", transform=ax.transAxes, fontsize=5.8, va="top")
    axes[0].set_ylabel("Surgery − biopsy, log₂ intensity")
    container.suptitle(f"{label + '  ' if label else ''}Exploratory response size by metastatic status", x=.02, ha="left", fontsize=9, fontweight="bold")


def panel_e(ax, d):
    frame = d["two_angle_status_pathways"].sort_values("pathway_BH_FDR_gamma", ascending=False)
    y = np.arange(len(frame))
    ax.scatter(-np.log10(frame.pathway_BH_FDR_gamma), y, s=55, color="#D55E00", zorder=3)
    ax.hlines(y, 0, -np.log10(frame.pathway_BH_FDR_gamma), color="#D8B4A4", lw=2)
    ax.set_yticks(y, frame.candidate_pathway.str.replace(" and ", " / ", regex=False))
    ax.set_xlabel("−log₁₀ pathway BH q")
    ax.set_xlim(0, 1.47)
    ax.axvline(-np.log10(.05), ls="--", color="#777777", lw=.7)
    for yy, q in zip(y, frame.pathway_BH_FDR_gamma):
        ax.text(-np.log10(q)+.02, yy, f"q={q:.3f}", va="center", fontsize=6.2)
    title(ax, "E", "Metastatic-status pathway candidates", "Exploratory; no pathway passes BH q<0.05")


def panel_f(ax, d):
    frame = d["two_angle_scknife_family_map"]
    cells = [x for x in ["Epithelial", "CAF", "Endothelial", "T cells"] if x in set(frame.celltype)]
    families = [x for x in FAMILY_SHORT if x in set(frame.display_family)]
    values = np.full((len(families), len(cells)), np.nan)
    counts = np.zeros(values.shape, int)
    for i, family in enumerate(families):
        for j, cell in enumerate(cells):
            row = frame.loc[frame.display_family.eq(family) & frame.celltype.eq(cell)]
            if len(row):
                values[i, j] = row.fraction_patients_decreased.iloc[0]
                counts[i, j] = int(row.n_nodes.iloc[0])
    cmap = mpl.colormaps["RdBu_r"].copy(); cmap.set_bad("#F2F2F2")
    im = ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cells)), cells)
    ax.set_yticks(range(len(families)), [FAMILY_SHORT[x] for x in families])
    ax.tick_params(length=0)
    for i in range(len(families)):
        for j in range(len(cells)):
            if counts[i, j]:
                ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=6, color="white" if abs(values[i, j]-.5)>.33 else "#222222")
    cb = ax.figure.colorbar(im, ax=ax, fraction=.035, pad=.02)
    cb.set_label("Mean fraction of patients lower at surgery", fontsize=6.3)
    cb.ax.tick_params(labelsize=5.5)
    title(ax, "F", "Patient-balanced scKNIFE response map", "158 reviewed Tier-3 pairs; color is mean patient fraction; cells show node counts")


def panel_g(ax, d):
    frame = d["two_angle_p3_medians"]
    time = ["Bx", "Surgery", "Recurrence"]
    labels = {"2_phosphoglycolate": "2-phosphoglycolate", "selenocystathionine": "Selenocystathionine", "tRNA_containing_6_isopentenyladenosine": "tRNA with i6A"}
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for color, (node, subset) in zip(colors, frame.groupby("metabolite", sort=True)):
        ordered = subset.set_index("Timepoint").loc[time]
        ax.plot(range(3), ordered.median_activity, marker="o", color=color, lw=1.7, markersize=4, label=labels.get(node, node))
    ax.set_xticks(range(3), ["Biopsy", "Surgery", "Recurrence"])
    ax.set_ylabel("P3 CAF median inferred activity")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    title(ax, "G", "P3 CAF trajectory", "One patient; inferred activity, not LC–MS abundance")


def panel_h(ax, d):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        (.08, .72, .34, .17, "Corrected LC–MS\n5 treatment pathways"),
        (.58, .72, .34, .17, "Paired tissue response\n2 unidentified features"),
        (.58, .39, .34, .17, "Status-associated\nresponse amplitude"),
        (.08, .39, .34, .17, "scKNIFE cell-type\nactivity context"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(mpl.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.018", facecolor="#F4F7F8", edgecolor="#62757C", lw=.9))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=7)
    ax.annotate("", xy=(.57, .80), xytext=(.43, .80), arrowprops={"arrowstyle":"-", "lw":1, "ls":"--", "color":"#888888"})
    ax.annotate("", xy=(.75, .57), xytext=(.75, .70), arrowprops={"arrowstyle":"->", "lw":1.1, "color":"#4A5960"})
    ax.annotate("", xy=(.43, .47), xytext=(.57, .47), arrowprops={"arrowstyle":"-", "lw":1, "ls":"--", "color":"#888888"})
    ax.text(.5, .30, "No confirmed molecular match across assays", ha="center", fontsize=6.5, color="#555555")
    ax.text(.5, .18, "Outcome association is exploratory (4 vs 3 patients); P3 recurrence is a case view.", ha="center", fontsize=6.3, color="#555555")
    title(ax, "H", "Evidence and open links")


def make_panel(label: str, d: dict[str, pd.DataFrame]):
    style()
    fig = plt.figure(figsize=(7.1, 4.5 if label in {"C", "D", "F"} else 3.7), facecolor="white", layout="constrained")
    if label in {"C", "D"}:
        {"C": panel_c, "D": panel_d}[label](fig, d)
    else:
        ax = fig.add_subplot(111)
        {"A":panel_a,"B":panel_b,"E":panel_e,"F":panel_f,"G":panel_g,"H":panel_h}[label](ax,d)
    return fig


def make_compiled(d: dict[str, pd.DataFrame]):
    style()
    fig = plt.figure(figsize=(11.7, 14.8), facecolor="white", layout="constrained")
    sub = fig.subfigures(4, 2, height_ratios=[.9, 1.15, 1, 1], wspace=.10, hspace=.12)
    for label, container in zip("ABCDEFGH", sub.flat):
        if label in {"C", "D"}:
            {"C":panel_c,"D":panel_d}[label](container,d)
        else:
            ax=container.add_subplot(111)
            {"A":panel_a,"B":panel_b,"E":panel_e,"F":panel_f,"G":panel_g,"H":panel_h}[label](ax,d)
    return fig


def export(fig, stem: Path):
    stem=Path(stem);stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix("."+ext), dpi=300, facecolor="white")
