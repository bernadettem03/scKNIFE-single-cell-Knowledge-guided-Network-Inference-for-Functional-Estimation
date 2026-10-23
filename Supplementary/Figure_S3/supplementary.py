"""A4 publication layout for Supplementary Figure 3, using frozen plot data."""
from __future__ import annotations

from pathlib import Path
import json
import platform
import hashlib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import leaves_list, linkage
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parents[1] / "Figure_3/data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure parameters (user editable)
OUTPUT_DPI = 300
MM = 1 / 25.4
A4_LANDSCAPE_IN = (297 * MM, 210 * MM)
TEXT_SIZE_PT = 6
PANEL_LABEL_SIZE_PT = TEXT_SIZE_PT
TITLE_SIZE_PT = TEXT_SIZE_PT
NOTE_SIZE_PT = TEXT_SIZE_PT
TICK_SIZE_PT = TEXT_SIZE_PT
HEATMAP_TOP_N = 10
HEATMAP_PADJ = 0.05
HEATMAP_LABEL_LIMIT = 38
HEATMAP_CONTEXT_TASKS = ("Glutathionate synthesis", "Trafficking to endosomes (Clathrin Vesicles)")
HEATMAP_CELL_TYPE_LABELS = {
    "B_cell": "B cell",
    "Cancer_cell": "Tumor/epithelial",
    "Endothelial_cell": "Endothelial cell",
    "Fibroblast": "Fibroblast",
    "Myeloid_cell": "Myeloid cell",
    "T_cell": "T cell",
}
CANDIDATES_PER_COMPARTMENT = 4
MIN_CONCORDANCE = 0.70
FAV_COLOR, UNFAV_COLOR = "#0072B2", "#D55E00"
UMAP_BACKGROUND_SIZE = 0.075
UMAP_OVERLAY_SIZE = 0.24
UMAP_BACKGROUND_ALPHA = 0.30
UMAP_OVERLAY_ALPHA = 0.84
UMAP_BACKGROUND_COLOR = "#D9D9D9"
UMAP_CMAP = "Spectral_r"
UMAP_RASTERIZED = True
GO_COHORT_COLORS = {"Bassez A": "#0072B2", "Bassez B": "#D55E00"}
GO_FOREST_PAD = 1.0
GO_TERM_SIZE_PT = 5.0
GO_GROUP_SIZE_PT = 5.3


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "Carlito",
        "font.size": TEXT_SIZE_PT,
        "axes.titlesize": TEXT_SIZE_PT,
        "axes.labelsize": TEXT_SIZE_PT,
        "xtick.labelsize": TEXT_SIZE_PT,
        "ytick.labelsize": TEXT_SIZE_PT,
        "legend.fontsize": TEXT_SIZE_PT,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": OUTPUT_DPI,
        "savefig.dpi": OUTPUT_DPI,
    })


def _short_go_term(term: str, limit: int = 72) -> str:
    term = term.rsplit(" (GO:", 1)[0]
    replacements = {
        "positive regulation of ": "pos. regulation of ",
        "negative regulation of ": "neg. regulation of ",
        "CD4-positive": "CD4+",
        "CD25-positive": "CD25+",
        "alpha-beta": "alpha-beta",
    }
    for old, new in replacements.items():
        term = term.replace(old, new)
    return term if len(term) <= limit else term[: limit - 1].rstrip() + "…"


def _draw_go_patient_forest(label_ax, axes) -> None:
    path = DATA_DIR / "bassez_go_patient_interaction_display.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    cohort_order = ["Bassez A", "Bassez B"]
    rows = (
        table.loc[table.cohort.eq("Bassez A")]
        .sort_values(["cell_type", "display_rank_within_celltype"])
        [["cell_type", "cell_type_label", "go_term", "display_rank_within_celltype"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    group_order = rows.cell_type.drop_duplicates().tolist()
    row_y: dict[tuple[str, str], float] = {}
    group_y: dict[str, float] = {}
    y = 0.0
    for cell_type in group_order:
        group_rows = rows.loc[rows.cell_type.eq(cell_type)]
        group_y[cell_type] = y + (len(group_rows) - 1) / 2
        for record in group_rows.itertuples(index=False):
            row_y[(record.cell_type, record.go_term)] = y
            y += 1.0
        y += GO_FOREST_PAD

    finite_intervals = table.loc[
        table.testable,
        ["limma_ci_low", "limma_ci_high"],
    ].to_numpy(dtype=float)
    finite_intervals = finite_intervals[np.isfinite(finite_intervals)]
    x_limit = float(np.ceil((np.max(np.abs(finite_intervals)) + 0.2) * 2) / 2)
    x_limit = max(x_limit, 1.0)

    for cohort, ax in zip(cohort_order, axes):
        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8, zorder=0)
        for record in rows.itertuples(index=False):
            match = table.loc[
                table.cell_type.eq(record.cell_type)
                & table.go_term.eq(record.go_term)
                & table.cohort.eq(cohort)
            ]
            if len(match) != 1:
                raise RuntimeError(
                    f"Expected one GO display row for {cohort}, {record.cell_type}, {record.go_term}"
                )
            result = match.iloc[0]
            row = row_y[(record.cell_type, record.go_term)]
            if bool(result.testable):
                effect = float(result.limma_interaction_log2_ne_minus_e)
                low = float(result.limma_ci_low)
                high = float(result.limma_ci_high)
                ax.errorbar(
                    effect, row, xerr=[[effect - low], [high - effect]],
                    fmt="o", markersize=3.8, capsize=2.2, capthick=0.8,
                    color=GO_COHORT_COLORS[cohort], ecolor=GO_COHORT_COLORS[cohort],
                    elinewidth=1.0, markeredgecolor="white", markeredgewidth=0.45,
                    zorder=3,
                )
                if pd.notna(result.global_bh_q) and result.global_bh_q < 0.05:
                    offset = 0.10 if effect >= 0 else -0.10
                    ax.text(
                        effect + offset, row, "*", color="#111111", fontsize=TEXT_SIZE_PT,
                        fontweight="bold", ha="left" if effect >= 0 else "right",
                        va="center", zorder=4,
                    )
            elif result.n_e > 0 and result.n_ne > 0:
                effect = float(result.median_interaction_ne_minus_e)
                ax.plot(
                    effect, row, marker="D", markersize=3.7, linestyle="none",
                    markerfacecolor="white", markeredgecolor=GO_COHORT_COLORS[cohort],
                    markeredgewidth=1.1, zorder=3,
                )
                ax.annotate(
                    "NT", (effect, row), xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=TEXT_SIZE_PT, color="#444444",
                )
            else:
                ax.text(0, row, "NA", ha="center", va="center", fontsize=TICK_SIZE_PT, color="#777777")

        ax.set_xlim(-x_limit, x_limit)
        ax.set_ylim(y - 0.5, -0.8)
        ax.set_xticks(np.arange(-x_limit, x_limit + 0.01, 1.0))
        ax.tick_params(axis="x", labelsize=TICK_SIZE_PT, length=2)
        ax.set_xlabel("")
        ax.set_title(cohort, fontsize=TITLE_SIZE_PT, fontweight="bold", pad=5)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False, length=0)

    label_ax.set_axis_off()
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(y - 0.5, -0.8)
    for record in rows.itertuples(index=False):
        first_in_group = record.display_rank_within_celltype == 1
        if first_in_group:
            counts = []
            for cohort in cohort_order:
                match = table.loc[
                    table.cohort.eq(cohort) & table.cell_type.eq(record.cell_type)
                ].iloc[0]
                counts.append(f"{cohort[-1]} E/NE={int(match.n_e)}/{int(match.n_ne)}")
            label_ax.text(
                0.0, row_y[(record.cell_type, record.go_term)] - 0.63,
                f"{record.cell_type_label}  ({'; '.join(counts)})",
                ha="left", va="bottom", fontsize=GO_GROUP_SIZE_PT,
                fontweight="bold", color="#222222",
            )
        label_ax.text(
            0.015, row_y[(record.cell_type, record.go_term)],
                _short_go_term(record.go_term, limit=36),
                ha="left", va="center", fontsize=GO_TERM_SIZE_PT, color="#333333",
        )
    axes[0].tick_params(axis="y", labelleft=False, labelright=False)
    axes[1].tick_params(axis="y", labelleft=False, labelright=False)
    for cell_type in group_order:
        center = group_y[cell_type]
        for ax in axes:
            ax.axhline(center + 1.5 + GO_FOREST_PAD / 2, color="#B8B8B8", linewidth=0.45)
    axes[1].set_ylabel("")
    label_ax.text(
        0.0, 1.03, "Patient-level GO treatment-response interaction",
        transform=label_ax.transAxes, ha="left", va="bottom",
        fontsize=PANEL_LABEL_SIZE_PT, fontweight="bold",
    )


def _short_task(task: str, limit: int = 43) -> str:
    return task if len(task) <= limit else task[:limit - 1].rstrip() + "…"


def _candidate_dotplot(ax, letter: str, title: str, ranked: pd.DataFrame,
                       metric: str, directions: tuple[str, str]) -> None:
    blocks = [ranked.loc[ranked.cell_type.eq(cell_type)].head(CANDIDATES_PER_COMPARTMENT)
              for cell_type in ("Myeloid", "Tumor/epithelial", "T cell")]
    data = pd.concat(blocks, ignore_index=True).iloc[::-1].reset_index(drop=True)
    data["label"] = data.cell_type.str.replace("Tumor/epithelial", "Tumor", regex=False) + " · " + data.task.map(_short_task)
    direction = data["delta_group_b_vs_a"] if "delta_group_b_vs_a" in data else data["delta_group_b_vs_a__Bassez A"]
    color = np.where(direction.gt(0), UNFAV_COLOR, FAV_COLOR)
    size = 16 + 75 * data[metric].clip(0, 1)
    concordance = data.patient_concordance_rate if "patient_concordance_rate" in data else data.mean_concordance
    yy = np.arange(len(data))
    ax.scatter(concordance, yy, s=size, c=color, edgecolor="white", linewidth=0.45)
    chosen = data.selected.to_numpy(bool)
    ax.scatter(concordance.to_numpy()[chosen], yy[chosen], s=size.to_numpy()[chosen] + 22,
               facecolor="none", edgecolor="#111111", linewidth=0.8)
    ax.axvline(MIN_CONCORDANCE, color="#777777", linewidth=0.7, linestyle="--")
    ax.set_xlim(0.45, 1.02)
    ax.set_yticks(yy, data.label, fontsize=TEXT_SIZE_PT)
    ax.set_xlabel("Patient directional concordance", fontsize=TEXT_SIZE_PT)
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=TEXT_SIZE_PT, fontweight="bold")
    ax.tick_params(axis="x", labelsize=TEXT_SIZE_PT)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(False)


def render_s3a():
    """One A4 landscape page containing all four supplementary panels."""
    _style()
    rnr = pd.read_csv(DATA_DIR / "rnr_ranked_candidates.csv")
    ene = pd.read_csv(DATA_DIR / "ene_ranked_candidates.csv")
    fig = plt.figure(figsize=A4_LANDSCAPE_IN, facecolor="white")
    axes = [fig.add_axes([0.19, 0.505, 0.09, 0.370]),
            fig.add_axes([0.45, 0.505, 0.09, 0.370])]
    _candidate_dotplot(axes[0], "A", "Clinical R/NR", rnr, "abs_delta", ("R", "NR"))
    _candidate_dotplot(axes[1], "B", "Clonotype E/NE", ene, "min_abs_delta", ("E", "NE"))
    ax_labels = fig.add_axes([0.56, 0.495, 0.14, 0.380])
    ax_a = fig.add_axes([0.715, 0.495, 0.115, 0.380])
    ax_b = fig.add_axes([0.85, 0.495, 0.115, 0.380], sharey=ax_a)
    _draw_go_patient_forest(ax_labels, (ax_a, ax_b))
    ax_a.set_title("C  Bassez A", fontsize=TITLE_SIZE_PT, fontweight="bold", pad=5)
    ax_b.set_title("D  Bassez B", fontsize=TITLE_SIZE_PT, fontweight="bold", pad=5)
    heatmap_audits = {}
    for cohort, x in (("Bassez A", 0.05), ("Bassez B", 0.53)):
        heatmap_audits[cohort] = render_s3a_heatmap(fig, cohort, x)
    fig.text(0.05, 0.975, "Supplementary Figure 3A · candidate concordance, metabolic tasks and patient-level GO interaction",
             ha="left", va="top", fontsize=TEXT_SIZE_PT, fontweight="bold")
    fig.text(0.05, 0.945, "Blue: higher R or E · orange: higher NR or NE · outlined points: selected candidates",
             ha="left", va="top", fontsize=TEXT_SIZE_PT)
    fig.text(0.84, 0.435, "NE − E difference in patient log2(On/Pre)",
             ha="center", va="center", fontsize=NOTE_SIZE_PT)
    fig.text(0.05, 0.397, "Pooled-cell metabolic-task log2(On/Pre) · top 10 per group + G/H tasks",
             ha="left", va="bottom", fontsize=PANEL_LABEL_SIZE_PT, fontweight="bold")
    fig.text(0.05, 0.018,
             "GO: limma estimate and 95% CI · * global BH q<0.05 · diamond: descriptive median (NT) · NA: group absent",
             ha="left", va="bottom", fontsize=TEXT_SIZE_PT)
    return fig, heatmap_audits


def _prepare_heatmap(de: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, dict]:
    """Select existing significant task values without re-estimation."""
    frame = de.copy()
    frame["label"] = frame.expansion_group + "_" + frame.cell_type
    significant = frame.loc[frame.adj_p_value < HEATMAP_PADJ]
    grouped = significant.groupby(["feature", "expansion_group"]).log2FC.mean()
    top_e = grouped.xs("E", level="expansion_group").nlargest(top_n).index.tolist()
    top_ne = grouped.xs("NE", level="expansion_group").nlargest(top_n).index.tolist()
    selected = top_e + top_ne if top_n == 50 else list(dict.fromkeys(top_e + top_ne))
    if top_n == HEATMAP_TOP_N:
        selected = list(dict.fromkeys(selected + [task for task in HEATMAP_CONTEXT_TASKS
                                                 if task in frame.feature.values]))
    pivot = (frame.loc[frame.feature.isin(selected)]
             .pivot_table(index="feature", columns="label", values="log2FC", aggfunc="first")
             .fillna(0).reindex(selected))
    return pivot, {"matrix_rows": len(pivot), "matrix_columns": len(pivot.columns),
                   "e_tasks": top_e, "ne_tasks": top_ne, "columns": pivot.columns.tolist()}


def _display_heatmap_label(label: str) -> str:
    return label if len(label) <= HEATMAP_LABEL_LIMIT else label[:HEATMAP_LABEL_LIMIT - 1].rstrip() + "…"


def render_s3a_heatmap(fig, cohort: str, x: float):
    """Draw one compact cohort heatmap on the shared A4 S3A page."""
    _style()
    code = cohort[-1]
    de = pd.read_csv(DATA_DIR / f"de/bassez_{code}_knife.csv")
    full_pivot, full_audit = _prepare_heatmap(de, 50)
    frozen = json.loads((DATA_DIR / "Figure_3_bassez_heatmap_audit.json").read_text())[code]
    for key in ("matrix_rows", "matrix_columns", "e_tasks", "ne_tasks"):
        if full_audit[key] != frozen[key]:
            raise RuntimeError(f"Bassez {code} heatmap selection changed: {key}")
    if set(full_audit["columns"]) != set(frozen["columns"]):
        raise RuntimeError(f"Bassez {code} heatmap columns changed")
    pivot, audit = _prepare_heatmap(de, HEATMAP_TOP_N)
    pivot = pivot.reindex(columns=frozen["columns"])
    row_order = leaves_list(linkage(pivot.to_numpy(), method="average", metric="euclidean"))
    col_order = leaves_list(linkage(pivot.to_numpy().T, method="average", metric="euclidean"))
    pivot = pivot.iloc[row_order, col_order]
    ax = fig.add_axes([x + 0.015, 0.145, 0.145, 0.220])
    limit = float(np.max(np.abs(pivot.to_numpy())))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
                      interpolation="nearest", rasterized=True)
    ax.set_yticks(np.arange(len(pivot)), [_display_heatmap_label(task) for task in pivot.index],
                  fontsize=4.8)
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", length=0, pad=2)
    display_columns = []
    for label in pivot.columns:
        group, cell_type = label.split("_", 1)
        display_columns.append(f"{group} · {HEATMAP_CELL_TYPE_LABELS[cell_type]}")
    ax.set_xticks(np.arange(len(pivot.columns)), display_columns, rotation=90, fontsize=5.2)
    ax.tick_params(axis="x", length=0, pad=2)
    letter = "E" if cohort == "Bassez A" else "F"
    ax.set_title(f"{letter}  {cohort}", loc="left", fontsize=TEXT_SIZE_PT, fontweight="bold", pad=3)
    colorbar_ax = fig.add_axes([x + 0.001, 0.19, 0.006, 0.125])
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="vertical")
    colorbar.ax.tick_params(labelsize=4.4, length=1, pad=1)
    colorbar.set_label("log2(On/Pre)", fontsize=4.7, labelpad=1)
    return {"rows": len(pivot), "columns": len(pivot.columns),
            "row_order": pivot.index.tolist(), "selected_e": audit["e_tasks"],
            "selected_ne": audit["ne_tasks"], "context_tasks": [task for task in HEATMAP_CONTEXT_TASKS
                                                        if task in pivot.index]}


def _clean_umap_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_task_umap(ax, background, overlay, row, letter: str):
    bg = background.loc[
        background.cohort.eq(row.cohort) & background.condition.eq(row.condition)
    ]
    points = overlay.loc[overlay.map_id.eq(row.map_id)].sort_values("log10_score_plus_epsilon")
    if bg.empty or points.empty:
        raise RuntimeError(f"Empty UMAP panel for {row.map_id}")
    norm = Normalize(float(row.color_low), float(row.color_high))
    ax.scatter(
        bg.umap_1, bg.umap_2, s=UMAP_BACKGROUND_SIZE,
        color=UMAP_BACKGROUND_COLOR, alpha=UMAP_BACKGROUND_ALPHA,
        linewidths=0, rasterized=UMAP_RASTERIZED,
    )
    ax.scatter(
        points.umap_1, points.umap_2, c=points.log10_score_plus_epsilon,
        cmap=UMAP_CMAP, norm=norm, s=UMAP_OVERLAY_SIZE,
        alpha=UMAP_OVERLAY_ALPHA, linewidths=0, rasterized=UMAP_RASTERIZED,
    )
    ax.set_title(
        f"{letter}  Main {row.main_panel} · {row.cohort} · {row.cell_type}\n{row.condition} · {row.display_task}",
        loc="left", fontsize=TITLE_SIZE_PT, fontweight="bold", pad=3,
    )
    ax.text(
        0.5, -0.055,
        f"{row.patient_count_label} · {len(points):,} target cells",
        transform=ax.transAxes, ha="center", va="top", fontsize=NOTE_SIZE_PT,
    )
    _clean_umap_axis(ax)
    return ScalarMappable(norm=norm, cmap=UMAP_CMAP)


def render_s3b():
    """Seven frozen-embedding task UMAPs corresponding to main E--J."""
    _style()
    background = pd.read_parquet(DATA_DIR / "task_umap_background.parquet")
    overlay = pd.read_parquet(DATA_DIR / "task_umap_activity.parquet")
    metadata = pd.read_csv(DATA_DIR / "task_umap_metadata.csv")

    baseline = metadata.loc[metadata.figure_kind.eq("baseline")].sort_values("display_order")
    treated = metadata.loc[
        metadata.figure_kind.eq("longitudinal") & metadata.condition_order.eq(1)
    ].sort_values("display_order")
    selected = pd.concat([baseline, treated], ignore_index=True)
    expected_panels = ["E", "F", "F", "G", "H", "I", "J"]
    if len(selected) != 7 or selected.main_panel.tolist() != expected_panels:
        raise RuntimeError(
            f"Expected current E, F, F, G, H, I, J UMAPs; found {selected.main_panel.tolist()}"
        )

    fig = plt.figure(figsize=A4_LANDSCAPE_IN, facecolor="white")
    outer = fig.add_gridspec(
        2, 1, height_ratios=(1, 1), left=0.045, right=0.97,
        bottom=0.12, top=0.91, hspace=0.43,
    )
    top = outer[0].subgridspec(1, 3, wspace=0.18)
    bottom = outer[1].subgridspec(1, 4, wspace=0.16)
    axes = [fig.add_subplot(top[0, index]) for index in range(3)]
    axes += [fig.add_subplot(bottom[0, index]) for index in range(4)]

    for ax, letter, row in zip(axes, "ABCDEFG", selected.itertuples(index=False)):
        sm = _draw_task_umap(ax, background, overlay, row, letter)
        colorbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.15, aspect=24)
        colorbar.set_label("log10(activity + 1e−10)", fontsize=NOTE_SIZE_PT, labelpad=1)
        colorbar.ax.tick_params(labelsize=TICK_SIZE_PT, length=2)

    fig.text(0.045, 0.965, "Supplementary Figure 3B · frozen-embedding task localization",
             ha="left", va="top", fontsize=TEXT_SIZE_PT, fontweight="bold")
    return fig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_and_export() -> dict:
    _style()
    s3a, heatmap_audits = render_s3a()
    a_pdf = OUTPUT_DIR / "Figure_3_S3_A_final.pdf"
    a_png = OUTPUT_DIR / "Figure_3_S3_A_final.png"
    s3a.savefig(a_pdf, dpi=OUTPUT_DPI, facecolor="white")
    s3a.savefig(a_png, dpi=OUTPUT_DPI, facecolor="white")
    plt.close(s3a)
    s3b = render_s3b()
    b_pdf = OUTPUT_DIR / "Figure_3_S3_B_final.pdf"
    b_png = OUTPUT_DIR / "Figure_3_S3_B_final.png"
    s3b.savefig(b_pdf, dpi=OUTPUT_DPI, facecolor="white")
    s3b.savefig(b_png, dpi=OUTPUT_DPI, facecolor="white")
    plt.close(s3b)

    final_files = {a_pdf, a_png, b_pdf, b_png}
    for pattern in ("Figure_3_S3_A_final_page_*", "Figure_3_S3_[AB]_final.*"):
        for old in OUTPUT_DIR.glob(pattern):
            if old.is_file() and old not in final_files:
                old.unlink()
    old_manifest = OUTPUT_DIR / "Figure_3_supplementary_manifest.json"
    if old_manifest.exists():
        old_manifest.unlink()

    manifest = {
        "official_supplementary": ["S3_A_final", "S3_B_final"],
        "font": {"family": "Carlito", "size_pt": TEXT_SIZE_PT, "calibri_compatible_substitute": True},
        "S3_A_final": {"pdf": str(a_pdf.relative_to(ROOT)), "png": str(a_png.relative_to(ROOT)),
                       "pages": 1, "size_mm": [297, 210],
                       "layout": {"top_row": ["A", "B", "C: Bassez A GO", "D: Bassez B GO"],
                                  "bottom_row": ["E: Bassez A heatmap", "F: Bassez B heatmap"]},
                       "heatmap_audit": heatmap_audits},
        "S3_B_final": {"pdf": str(b_pdf.relative_to(ROOT)), "size_mm": [297, 210],
                       "png": str(b_png.relative_to(ROOT)), "pages": 1,
                       "panels": ["A main E", "B-C main F in Bassez A/B", "D main G",
                                  "E main H", "F main I", "G main J"]},
        "renderer": {"python": platform.python_version(), "matplotlib": mpl.__version__,
                     "seaborn": sns.__version__, "cnsplots": "unavailable; matplotlib fallback",
                     "umap_coordinates": "frozen; no embedding recomputed"},
    }
    for key, pdf, png in (("S3_A_final", a_pdf, a_png), ("S3_B_final", b_pdf, b_png)):
        manifest[key]["sha256_pdf"] = _sha256(pdf)
        manifest[key]["sha256_png"] = _sha256(png)
    (ROOT / "Figure_3_supplementary_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(render_and_export(), indent=2))
