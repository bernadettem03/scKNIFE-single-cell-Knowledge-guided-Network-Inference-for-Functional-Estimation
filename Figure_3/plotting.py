from pathlib import Path
from typing import Mapping
from dataclasses import replace
from itertools import combinations
import itertools
from IPython.display import display
import base64, hashlib, json, platform, sys, textwrap
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
PANEL_OUTPUT_DIR = OUTPUT_DIR / 'panels'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure parameters (user editable)
MM = 1 / 25.4
OUTPUT_DPI = 300
RANDOM_SEED = 42
BOOTSTRAP_REPLICATES = 5000
MIN_CELLS_PER_PATIENT_CELLTYPE = 20
MIN_GROUP_PATIENTS_RNR = 4
MIN_GROUP_PATIENTS_ENE = 2
MIN_CONCORDANCE = 0.70
TOP_DOTPLOT_PER_COMPARTMENT = 4
COMPARTMENTS = ("Myeloid", "Tumor/epithelial", "T cell")
FAV_COLOR, UNFAV_COLOR = "#0072B2", "#D55E00"
COHORT_COLORS = {"Bassez A": "#009E73", "Bassez B": "#CC79A7"}
COHORT_MARKERS = {"Bassez A": "o", "Bassez B": "s"}
PANEL_LABEL_SIZE_PT, TITLE_SIZE_PT = 10, 8
AXIS_TITLE_SIZE_PT, TICK_SIZE_PT, NOTE_SIZE_PT = 7, 5.2, 4.6
MAIN_WIDTH_MM, MAIN_HEIGHT_MM = 180, 180
AUDIT_HEIGHT_MM = 245
A4_PORTRAIT_WIDTH_MM, A4_PORTRAIT_HEIGHT_MM = 210, 297
A4_MARGIN_MM, A4_GUTTER_MM = 8, 5
A4_LAYOUT_ROW_GAP_MM = 5
UPSET_BAR_WIDTH = 0.62
UPSET_MATRIX_POINT_SIZE = 13
UPSET_MATRIX_LINE_WIDTH = 0.85
UPSET_INACTIVE_COLOR = "#D9D9D9"
UPSET_ACTIVE_COLOR = "#303030"
UPSET_LEGEND_SIZE_PT = 4.0
UPSET_CELLTYPE_COLORS = {
    "B cell":"#bcbd22", "B_cell":"#bcbd22", "Bcell":"#bcbd22",
    "Cancer cell":"#1f77b4", "Cancer_cell":"#1f77b4", "Epithelial":"#1f77b4",
    "Endothelial cell":"#aec7e8", "Endothelial_cell":"#aec7e8", "Endothelial":"#aec7e8",
    "Fibroblast":"#8c564b", "Fibroblasts":"#8c564b",
    "Myeloid cell":"#e377c2", "Myeloid_cell":"#e377c2", "myeloid":"#e377c2",
    "T cell":"#7f7f7f", "T_cell":"#7f7f7f", "Tcell":"#7f7f7f",
    "Mast cell":"#ff7f0e", "mastcell":"#ff7f0e",
    "PVL":"#9467bd",
}
ACTIVITY_AXIS_LABEL = "Patient median normalized Method 5 activity (%)"
ACTIVITY_AXIS_TICKS = 4
TREATMENT_HEIGHT_MM = DELTA_HEIGHT_MM = 72
MIN_CELLS_PER_PAIRED_TIMEPOINT = 20
PRE_BOX_ALPHA, ON_BOX_ALPHA = 0.20, 0.42
BOX_WIDTH, PATIENT_POINT_SIZE, PATIENT_POINT_EDGE_WIDTH = 0.56, 10, 0.60
PAIR_GAP, BRACKET_LINE_WIDTH = 0.72, 0.65
BRACKET_HEIGHT_FRACTION, BRACKET_OFFSET_FRACTION = 0.025, 0.055
DELTA_POINT_SIZE, DELTA_POINT_EDGE_WIDTH, DELTA_JITTER_WIDTH = 15, 0.70, 0.09
DELTA_MEDIAN_HALF_WIDTH, DELTA_MEDIAN_LINE_WIDTH = 0.20, 1.6
DELTA_AXIS_ROUNDING, DELTA_ZERO_LINE_WIDTH = 0.5, 0.65
DELTA_ZERO_LINE_COLOR = "#777777"
UMAP_BASELINE_HEIGHT_MM, UMAP_LONGITUDINAL_HEIGHT_MM = 118, 224
UMAP_BACKGROUND_SIZE, UMAP_OVERLAY_SIZE = 0.075, 0.24
UMAP_BACKGROUND_ALPHA, UMAP_OVERLAY_ALPHA = 0.30, 0.82
UMAP_BACKGROUND_COLOR, UMAP_CMAP = "#D9D9D9", "Spectral_r"
UMAP_LOW_QUANTILE, UMAP_HIGH_QUANTILE = 0.02, 0.98
UMAP_RASTERIZED = True
UMAP_RASTER_DPI = 300  # raster point layers inside PDF/SVG
FINAL_S3_TITLE_SIZE_PT = 15
# GO heatmap parameters (same selection and clustermap grammar as metabolic tasks)
BASSEZ_GO_HEATMAP_TOP_N = 25
BASSEZ_GO_HEATMAP_PADJ = 0.05
BASSEZ_GO_COMPILED_ROW_FONT_PT = 3.6
FINAL_S3_A_PAGE_PT = (1152, 828)  # 16 x 11.5 inches
S3A_FINAL_SIZE_IN = (16, 11.5)
S3A_GRID_HEIGHT_RATIOS = (0.92, 1.08)
S3A_GRID_HSPACE = 0.16
S3A_PAGE_TITLE_Y = 0.985
S3A_TOP_SECTION_TITLE_Y = 0.925
S3A_BOTTOM_SECTION_TITLE_Y = 0.49
S3A_CANDIDATE_LEFT = 0.17
S3A_SOURCE_BOX_WIDTH_MM, S3A_SOURCE_BOX_HEIGHT_MM = 356, 82
FINAL_S3_B_PAGE_PT = (1152, 1584)  # 16 x 22 inches
S3B_GRID_TOP = 0.91
S3B_GRID_RIGHT = 0.97
S3B_PAGE_TITLE_Y = 0.985
S3B_TOP_PANEL_TITLE_Y = 0.925
S3B_BOTTOM_PANEL_TITLE_Y = 0.465
UMAP_TITLE_SIZE_PT = TITLE_SIZE_PT - 1.1
UMAP_NOTE_SIZE_PT = UMAP_COLORBAR_SIZE_PT = NOTE_SIZE_PT
RAW_SCORE_FLOOR = np.finfo(np.float32).tiny
EPSILON = 1e-10
DISPLAY_TRANSFORMATION = "log10(Method 5 score + 1e-10)"
LITERATURE_PRIORITY = {
    "Myeloid": ["Glycogen degradation", "Glucose to lactate conversion", "Arginine degradation", "Krebs cycle - oxidative decarboxylation of pyruvate"],
    "Tumor/epithelial": ["Arachidonate synthesis", "Palmitate synthesis", "Synthesis of palmitoyl-CoA", "Methionine degradation"],
    "T cell": ["Recognition of misfolded protein", "Heme synthesis", "Methionine degradation", "Oxidative phosphorylation via succinate-coenzyme Q oxidoreductase (COMPLEX II)"],
}

def _style():
    mpl.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
        "axes.grid":False, "axes.spines.top":False, "axes.spines.right":False,
        "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none", "figure.dpi":OUTPUT_DPI})
_style()

def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def _cliffs_delta(group_b, group_a):
    b = np.asarray(group_b, float); a = np.asarray(group_a, float)
    return float((np.greater.outer(b, a).sum() - np.less.outer(b, a).sum()) / (len(a)*len(b)))

manifest = json.loads((DATA_DIR / "input_manifest.json").read_text())
for relative, record in manifest.items():
    path = DATA_DIR / relative
    assert path.is_file() and _sha(path) == record["sha256"], relative
print(f"Validated {len(manifest)} staged inputs")


ARACHIDONATE = "Arachidonate synthesis"
ERAD = "Recognition of misfolded protein"
GLYCOGEN = "Glycogen degradation"

TREATMENT_PANELS = [
    {
        "association_id": "BASSEZ_A_MYELOID_GLUTATHIONE",
        "cohort": "Bassez A", "comparison": "Pre→On", "on_timepoint": "On",
        "cell_type": "Myeloid", "task": "Glutathionate synthesis",
        "display_task": "Glutathione synthesis", "group_labels": ("E", "NE"),
        "expected_patients": (5, 7),
    },
    {
        "association_id": "BASSEZ_B_TCELL_GLUTAMINE",
        "cohort": "Bassez B", "comparison": "Pre→On", "on_timepoint": "On",
        "cell_type": "T cell", "task": "Gluconeogenesis from Glutamine",
        "display_task": "Gluconeogenesis from glutamine", "group_labels": ("E", "NE"),
        "expected_patients": (3, 3),
    },
    {
        "association_id": "SHIAO_RTPD1_FIBRO_ALANINE",
        "cohort": "Shiao", "comparison": "Base→RTPD1", "on_timepoint": "RTPD1",
        "cell_type": "Fibroblast", "task": "Synthesis of alanine from glutamine",
        "display_task": "Alanine synthesis from glutamine", "group_labels": ("R", "NR"),
        "expected_patients": (8, 4),
    },
    {
        "association_id": "SHIAO_PD1_TUMOR_BILIRUBIN",
        "cohort": "Shiao", "comparison": "Base→PD1", "on_timepoint": "PD1",
        "cell_type": "Tumor/epithelial", "task": "Synthesis of bilirubin",
        "display_task": "Bilirubin synthesis", "group_labels": ("R", "NR"),
        "expected_patients": (10, 6),
    },
]

EVIDENCE = [
    {
        "association_id": "RNR_TUMOR_ARACHIDONATE", "panel": "S3_A", "endpoint": "R/NR",
        "cohort_scope": "Shiao", "cell_type": "Tumor/epithelial", "task": ARACHIDONATE,
        "direction": "higher in NR", "statistical_rank": 30,
        "evidence_tier": "Direct TNBC mechanism",
        "primary_citation": "Yu et al., Immunity (2025)",
        "citation_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11981829/",
        "rationale": "Therapy-resistant TNBC cells synthesize and export arachidonic-acid-rich lipids that promote immune suppression.",
        "limitation": "The inferred task does not measure arachidonate flux, export, or a specific eicosanoid branch.",
        "cross_endpoint_convergence": False,
    },
    {
        "association_id": "RNR_TCELL_ERAD", "panel": "E", "endpoint": "R/NR",
        "cohort_scope": "Shiao", "cell_type": "T cell", "task": ERAD,
        "direction": "higher in NR", "statistical_rank": 27,
        "evidence_tier": "Close immune mechanism",
        "primary_citation": "Ma et al., Cell Metabolism (2019)",
        "citation_url": "https://pubmed.ncbi.nlm.nih.gov/31031094/",
        "rationale": "ER-stress/XBP1 signaling is a demonstrated driver of dysfunctional, exhausted CD8 T cells in tumors.",
        "limitation": "The ERAD task is narrower than the unfolded-protein response and may reflect adaptive proteostasis.",
        "cross_endpoint_convergence": True,
    },
    {
        "association_id": "ENE_MYELOID_GLYCOGEN", "panel": "F", "endpoint": "E/NE",
        "cohort_scope": "Bassez A + B", "cell_type": "Myeloid", "task": GLYCOGEN,
        "direction": "higher in E", "statistical_rank": 3,
        "evidence_tier": "Close macrophage mechanism",
        "primary_citation": "Pahwa et al. (2024); Ma et al., Nature Communications (2020)",
        "citation_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11624281/; https://www.nature.com/articles/s41467-020-15636-8",
        "rationale": "Macrophage glycogenolysis supports inflammatory cytokine secretion, phagocytosis, and macrophage activation.",
        "limitation": "This evidence is not specific to TNBC or to clonotype expansion and does not establish clinical response.",
        "cross_endpoint_convergence": False,
    },
    {
        "association_id": "ENE_TCELL_ERAD", "panel": "S3_A", "endpoint": "E/NE",
        "cohort_scope": "Bassez A + B", "cell_type": "T cell", "task": ERAD,
        "direction": "higher in NE", "statistical_rank": 2,
        "evidence_tier": "Close immune mechanism",
        "primary_citation": "Ma et al., Cell Metabolism (2019)",
        "citation_url": "https://pubmed.ncbi.nlm.nih.gov/31031094/",
        "rationale": "ER-stress/XBP1 signaling is a demonstrated driver of dysfunctional, exhausted CD8 T cells in tumors.",
        "limitation": "E/NE is a clonotype-expansion endpoint, not clinical response; the ERAD score is not a direct exhaustion assay.",
        "cross_endpoint_convergence": True,
    },
]


PANEL_A_SUMMARY = pd.read_csv(DATA_DIR / "panel_a_overlap_summary.csv")
V8_PATHS = {name: DATA_DIR / name for name in (
 "eligible_patient_scores.csv.gz", "rnr_full_screen.csv", "ene_full_screen.csv",
 "rnr_ranked_candidates.csv", "ene_ranked_candidates.csv",
 "rnr_selected_tasks.csv", "ene_selected_tasks.csv",
)}
def prepare_v8_data(root=None, force=False):
    return V8_PATHS


def _upset(ax, root, selected_comparison=None):
    ax.set_axis_off()
    comparisons = PANEL_A_SUMMARY.comparison.drop_duplicates().tolist()
    if selected_comparison is not None:
        if selected_comparison not in comparisons:
            raise ValueError(f"Unknown method-overlap comparison: {selected_comparison}")
        comparisons = [selected_comparison]
    panel_titles = {'Bassez B · E': 'A  Bassez B · E', 'Bassez B · NE': 'B  Bassez B · NE', 'Shiao · R': 'C  Shiao · R', 'Shiao · NR': 'D  Shiao · NR'}
    intersections = ['Shared', 'scKNIFE only', 'scCellFie only']
    memberships = np.array([[True, True, False], [True, False, True]])
    x = np.arange(len(intersections))
    displayed_celltypes = PANEL_A_SUMMARY.cell_type.drop_duplicates().tolist()
    missing_colors = sorted(set(displayed_celltypes) - set(UPSET_CELLTYPE_COLORS))
    if missing_colors:
        raise ValueError(f'Missing UpSet cell-type colors: {missing_colors}')
    legend_handles = [Patch(facecolor=UPSET_CELLTYPE_COLORS[cell_type], label=cell_type) for cell_type in displayed_celltypes]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=len(legend_handles), frameon=False, fontsize=UPSET_LEGEND_SIZE_PT, handlelength=0.9, columnspacing=0.75, borderaxespad=0)
    for (index, comparison) in enumerate(comparisons):
        container = ax.inset_axes([0.02, 0.0, 0.96, 0.8] if len(comparisons) == 1 else [0.006 + index * 0.249, 0.0, 0.236, 0.8])
        container.set_axis_off()
        block = PANEL_A_SUMMARY.loc[PANEL_A_SUMMARY.comparison.eq(comparison)].copy()
        totals = block[intersections].sum().astype(int)
        union_total = int(totals.sum())
        container.text(0, 1.01, panel_titles[comparison], transform=container.transAxes, ha='left', va='bottom', fontsize=TITLE_SIZE_PT, fontweight='bold')
        container.text(0, 0.91, f'Union = {union_total:,} cell type–task pairs', transform=container.transAxes, ha='left', va='bottom', fontsize=NOTE_SIZE_PT, color='#444444')
        bar_ax = container.inset_axes([0.24, 0.4, 0.74, 0.46])
        bottom = np.zeros(len(intersections), dtype=float)
        for (_, row) in block.iterrows():
            values = row[intersections].to_numpy(dtype=float)
            bar_ax.bar(x, values, bottom=bottom, width=UPSET_BAR_WIDTH, color=UPSET_CELLTYPE_COLORS[row.cell_type], edgecolor='white', linewidth=0.35)
            bottom += values
        for (xpos, total) in zip(x, totals):
            bar_ax.annotate(f'{int(total):,}', (xpos, total), xytext=(0, 1.5), textcoords='offset points', ha='center', va='bottom', fontsize=NOTE_SIZE_PT)
        bar_ax.set_xticks([])
        bar_ax.set_ylabel('Intersection size', fontsize=NOTE_SIZE_PT, labelpad=1)
        bar_ax.tick_params(axis='y', labelsize=NOTE_SIZE_PT, width=0.5, length=2, pad=1)
        bar_ax.margins(y=0.16)
        bar_ax.grid(False)
        for spine in ('top', 'right'):
            bar_ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            bar_ax.spines[spine].set_linewidth(0.5)
        matrix_ax = container.inset_axes([0.24, 0.02, 0.74, 0.28])
        for column in range(len(intersections)):
            active = np.flatnonzero(memberships[:, column])
            if len(active) > 1:
                matrix_ax.plot([column, column], [active.min(), active.max()], color=UPSET_ACTIVE_COLOR, linewidth=UPSET_MATRIX_LINE_WIDTH, zorder=1)
            for row_index in range(2):
                matrix_ax.scatter(column, row_index, s=UPSET_MATRIX_POINT_SIZE, color=UPSET_ACTIVE_COLOR if memberships[row_index, column] else UPSET_INACTIVE_COLOR, edgecolors='none', zorder=2)
        matrix_ax.set_xlim(-0.45, len(intersections) - 0.55)
        matrix_ax.set_ylim(-0.45, 1.45)
        matrix_ax.invert_yaxis()
        matrix_ax.set_xticks(x, ['Shared', 'scKNIFE\nonly', 'scCellFie\nonly'], fontsize=NOTE_SIZE_PT)
        matrix_ax.set_yticks([0, 1], ['scKNIFE', 'scCellFie'], fontsize=NOTE_SIZE_PT)
        matrix_ax.tick_params(length=0, pad=1)
        matrix_ax.set_title('Set membership', loc='left', fontsize=NOTE_SIZE_PT, pad=1)
        for spine in matrix_ax.spines.values():
            spine.set_visible(False)

UMAP_PATHS = {'background': DATA_DIR / 'task_umap_background.parquet', 'overlay': DATA_DIR / 'task_umap_activity.parquet', 'metadata': DATA_DIR / 'task_umap_metadata.csv', 'audit': DATA_DIR / 'task_umap_validation.csv', 'render_audit': DATA_DIR / 'task_umap_render_audit.csv'}

PLOT_PATHS = {'evidence': DATA_DIR / 'literature_supported_tasks.csv', 'effects': DATA_DIR / 'literature_supported_effects_reference.csv', 'treatment_patient': DATA_DIR / 'treatment_patient_scores.csv.gz', 'treatment_statistics': DATA_DIR / 'treatment_panel_statistics.csv'}

def _short_task(task, width=35):
    return task if len(task) <= width else task[:width - 1] + '…'

def _dotplot(ax, letter, title, ranked, metric, direction_labels):
    blocks = []
    for cell_type in COMPARTMENTS:
        block = ranked.loc[ranked.cell_type.eq(cell_type)].head(TOP_DOTPLOT_PER_COMPARTMENT).copy()
        blocks.append(block)
    data = pd.concat(blocks, ignore_index=True)
    data['label'] = data.cell_type.str.replace('Tumor/epithelial', 'Tumor', regex=False) + ' · ' + data.task.map(_short_task)
    data = data.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    direction_value = data['delta_group_b_vs_a'] if 'delta_group_b_vs_a' in data else data['delta_group_b_vs_a__Bassez A']
    colors = np.where(direction_value.gt(0), UNFAV_COLOR, FAV_COLOR)
    sizes = 16 + 75 * data[metric].clip(0, 1)
    ax.scatter(data.patient_concordance_rate if 'patient_concordance_rate' in data else data.mean_concordance, y, s=sizes, c=colors, edgecolor='white', lw=0.45)
    chosen = data.selected.to_numpy(bool)
    x_values = (data.patient_concordance_rate if 'patient_concordance_rate' in data else data.mean_concordance).to_numpy()
    ax.scatter(x_values[chosen], y[chosen], s=sizes[chosen] + 22, facecolor='none', edgecolor='#111111', lw=0.8)
    ax.axvline(MIN_CONCORDANCE, color='#777777', lw=0.7, ls='--')
    ax.set_xlim(0.45, 1.02)
    ax.set_yticks(y, data.label, fontsize=NOTE_SIZE_PT)
    ax.set_xlabel('Patient directional concordance', fontsize=AXIS_TITLE_SIZE_PT)
    ax.set_title(f'{letter}  {title}', loc='left', fontsize=TITLE_SIZE_PT, fontweight='bold')
    ax.text(0.02, 0.98, f'Blue: higher {direction_labels[0]}  ·  orange: higher {direction_labels[1]}  ·  ring: selected', transform=ax.transAxes, ha='left', va='top', fontsize=NOTE_SIZE_PT, color='#555555')
    ax.tick_params(axis='x', labelsize=TICK_SIZE_PT)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.grid(False)

def _box_panel(ax, letter, cell_type, task, patient, cohorts, group_labels, screen, endpoint_title, compact=False, show_ylabel=True):
    (arrays, positions, colors, xticks, xticklabels) = ([], [], [], [], [])
    position = 0
    rng = np.random.default_rng(RANDOM_SEED + ord(letter))
    for cohort in cohorts:
        block = patient.loc[patient.cohort.eq(cohort) & patient.cell_type.eq(cell_type) & patient.task.eq(task)]
        for (status, label, color) in zip(('Favorable', 'Unfavorable'), group_labels, (FAV_COLOR, UNFAV_COLOR)):
            values = block.loc[block.status.eq(status), 'log2_median_score'].to_numpy()
            arrays.append(values)
            positions.append(position)
            colors.append(color)
            ax.scatter(position + rng.uniform(-0.12, 0.12, len(values)), values, s=10, facecolor='white', edgecolor=color, lw=0.6, zorder=3)
            position += 1
        xticks.append(position - 1.5)
        xticklabels.append(cohort.replace('Bassez ', 'B'))
        position += 0.8
    boxes = ax.boxplot(arrays, positions=positions, widths=0.58, patch_artist=True, showfliers=False, medianprops={'color': '#222222', 'linewidth': 0.8}, whiskerprops={'linewidth': 0.65}, capprops={'linewidth': 0.65}, boxprops={'linewidth': 0.65})
    for (box, color) in zip(boxes['boxes'], colors):
        box.set_facecolor(color)
        box.set_alpha(0.38)
    ax.set_xticks(xticks, xticklabels, fontsize=TICK_SIZE_PT)
    ax.set_ylabel('Patient median log2 score' if show_ylabel else '', fontsize=AXIS_TITLE_SIZE_PT)
    title_width = 20 if compact else 28
    wrapped_task = '\n'.join(textwrap.wrap(task, width=title_width, break_long_words=False))
    ax.set_title(f'{letter}  {cell_type} · {endpoint_title}\n{wrapped_task}', loc='left', fontsize=TITLE_SIZE_PT - (1.5 if compact else 1), fontweight='bold', pad=3)
    if len(cohorts) == 1:
        row = screen.loc[screen.cell_type.eq(cell_type) & screen.task.eq(task)].iloc[0]
        summary = f'δ={row.delta_group_b_vs_a:+.2f}; p={row.exact_permutation_p:.2g}; q={row.exact_permutation_q:.2g}'
    else:
        parts = []
        for cohort in cohorts:
            row = screen.loc[screen.cohort.eq(cohort) & screen.cell_type.eq(cell_type) & screen.task.eq(task)].iloc[0]
            parts.append(f"{cohort.replace('Bassez ', 'B')}: δ={row.delta_group_b_vs_a:+.2f}, q={row.exact_permutation_q:.2g}")
        summary = ('\n' if compact else '  ').join(parts)
    ax.text(0.5, -0.24 if compact else -0.2, summary, transform=ax.transAxes, ha='center', fontsize=NOTE_SIZE_PT - (0.3 if compact else 0))
    ax.tick_params(axis='y', labelsize=TICK_SIZE_PT)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.grid(False)

def _save_v8(fig, stem):
    result = {}
    for extension in ('png', 'pdf', ):
        path = stem.with_suffix('.' + extension)
        fig.savefig(path, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
        result[extension] = str(path)
    plt.close(fig)
    return result

def _evidence_summary(ax, evidence, effects):
    labels = ['Tumor · arachidonate synthesis · R/NR', 'T cell · ERAD/misfolded protein · R/NR', 'Myeloid · glycogen degradation · E/NE', 'T cell · ERAD/misfolded protein · E/NE']
    y = np.arange(4)[::-1]
    for (yy, (_, item)) in zip(y, evidence.iterrows()):
        block = effects.loc[effects.association_id.eq(item.association_id)]
        concordance = block.patient_concordance_rate.mean()
        color = UNFAV_COLOR if 'higher in N' in item.direction else FAV_COLOR
        ax.scatter(concordance, yy, s=70, color=color, edgecolor='white', lw=0.6, zorder=3)
        ax.text(1.01, yy, item.evidence_tier.replace(' mechanism', ''), va='center', ha='right', transform=ax.get_yaxis_transform(), fontsize=NOTE_SIZE_PT, color='#555555')
    ax.axvline(0.7, color='#888888', ls='--', lw=0.7)
    ax.set_xlim(0.6, 1.02)
    ax.set_yticks(y, labels, fontsize=NOTE_SIZE_PT + 0.2)
    ax.set_xlabel('Patient directional concordance', fontsize=7)
    ax.set_title('B  Literature-supported associations', loc='left', fontsize=TITLE_SIZE_PT, fontweight='bold')
    ax.text(0.01, 1.01, 'Blue: higher favorable  ·  orange: higher unfavorable', transform=ax.transAxes, fontsize=NOTE_SIZE_PT, color='#555555', va='bottom')
    ax.text(0.01, -0.31, 'Repeated T-cell ERAD = cross-endpoint convergence (distinct endpoints; not clinical replication).', transform=ax.transAxes, fontsize=NOTE_SIZE_PT, color='#555555')
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='x', labelsize=TICK_SIZE_PT)

def _forest(ax, letter, title, effects, endpoint):
    data = effects.loc[effects.endpoint.eq(endpoint)].copy()
    if endpoint == 'R/NR':
        data['label'] = data.cell_type.str.replace('Tumor/epithelial', 'Tumor', regex=False) + ' · ' + data.task
        palette = {'Shiao': '#333333'}
        markers = {'Shiao': 'o'}
    else:
        data['label'] = data.cell_type + ' · ' + data.task
        (palette, markers) = (COHORT_COLORS, COHORT_MARKERS)
    labels = list(dict.fromkeys(data.label))
    y_base = {label: len(labels) - 1 - i for (i, label) in enumerate(labels)}
    offsets = {'Shiao': 0, 'Bassez A': 0.11, 'Bassez B': -0.11}
    for (_, row) in data.iterrows():
        yy = y_base[row.label] + offsets[row.cohort]
        value = row.delta_group_b_vs_a
        ax.errorbar(value, yy, xerr=[[value - row.ci_low], [row.ci_high - value]], fmt=markers[row.cohort], color=palette[row.cohort], ecolor=palette[row.cohort], capsize=2, ms=4, label=row.cohort)
        ax.text(0.99, yy + 0.13, f'q={row.exact_permutation_q:.2g}', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=NOTE_SIZE_PT)
    (handles, legend_labels) = ax.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    if len(unique) > 1:
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=NOTE_SIZE_PT, ncol=2, loc='upper right', borderaxespad=0.1)
    ax.axvline(0, color='#555555', lw=0.7)
    ax.set_xlim(-1.05, 1.05)
    ax.set_yticks([y_base[label] for label in labels], ['\n'.join(textwrap.wrap(label, 35)) for label in labels], fontsize=NOTE_SIZE_PT)
    ax.set_xlabel("Cliff's δ (NR − R)" if endpoint == 'R/NR' else "Cliff's δ (NE − E)", fontsize=7)
    ax.set_title(f'{letter}  {title}', loc='left', fontsize=TITLE_SIZE_PT, fontweight='bold')
    ax.tick_params(axis='x', labelsize=TICK_SIZE_PT)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)

def _format_activity_axis(ax, show_ylabel=True):
    """Show raw normalized percentages at the existing log2 coordinates."""
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=ACTIVITY_AXIS_TICKS, prune='both'))

    def percentage_label(log2_score, _position):
        percentage = 100.0 * np.exp2(log2_score)
        if percentage >= 0.01:
            return f'{percentage:.3g}'
        return f'{percentage:.1e}'
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(percentage_label))
    ax.set_ylabel(ACTIVITY_AXIS_LABEL if show_ylabel else '', fontsize=7, labelpad=0)

def _box(ax, letter, patient, effects, association_id, endpoint_title):
    effect = effects.loc[effects.association_id.eq(association_id)]
    item = effect.iloc[0]
    cohorts = list(effect.cohort)
    (arrays, positions, colors, tick_labels) = ([], [], [], [])
    pos = 0.0
    rng = np.random.default_rng(RANDOM_SEED + ord(letter))
    for cohort in cohorts:
        block = patient.loc[patient.cohort.eq(cohort) & patient.cell_type.eq(item.cell_type) & patient.task.eq(item.task)]
        short_cohort = cohort.replace('Bassez ', 'B')
        endpoint_groups = ('R', 'NR') if endpoint_title == 'R/NR' else ('E', 'NE')
        for ((status, color), group_label) in zip((('Favorable', FAV_COLOR), ('Unfavorable', UNFAV_COLOR)), endpoint_groups):
            values = block.loc[block.status.eq(status), 'log2_median_score'].to_numpy()
            arrays.append(values)
            positions.append(pos)
            colors.append(color)
            tick_labels.append(group_label if cohort == 'Shiao' else f'{short_cohort}\n{group_label}')
            ax.scatter(pos + rng.uniform(-0.11, 0.11, len(values)), values, s=10, facecolor='white', edgecolor=color, lw=0.6, zorder=3)
            pos += 1
        pos += 0.65
    boxes = ax.boxplot(arrays, positions=positions, widths=0.55, patch_artist=True, showfliers=False, medianprops={'color': '#222222', 'linewidth': 0.8}, whiskerprops={'linewidth': 0.65}, capprops={'linewidth': 0.65}, boxprops={'linewidth': 0.65})
    for (box, color) in zip(boxes['boxes'], colors):
        box.set_facecolor(color)
        box.set_alpha(0.38)
    ax.set_xticks(positions, tick_labels, fontsize=TICK_SIZE_PT)
    _format_activity_axis(ax)
    task = 'ERAD / recognition of misfolded protein' if item.task == ERAD else item.task
    ax.set_title(f'{letter}  {item.cell_type} · {endpoint_title}\n' + '\n'.join(textwrap.wrap(task, 25)), loc='left', fontsize=TITLE_SIZE_PT - 1.2, fontweight='bold', pad=3)
    stats = []
    for (_, row) in effect.iterrows():
        prefix = '' if len(effect) == 1 else row.cohort.replace('Bassez ', 'B') + ': '
        stats.append(f'{prefix}δ={row.delta_group_b_vs_a:+.2f}, p={row.exact_permutation_p:.2g}, q={row.exact_permutation_q:.2g}')
    ax.text(0.5, -0.25, '\n'.join(stats), transform=ax.transAxes, ha='center', fontsize=NOTE_SIZE_PT)
    cell_p = item.cell_wilcox_p
    cell_q = item.cell_wilcox_q
    p_text = '<1×10⁻³⁰⁰' if cell_p == 0 else f'{cell_p:.2g}'
    q_text = '<1×10⁻³⁰⁰' if cell_q == 0 else f'{cell_q:.2g}'
    ax.text(0.5, -0.36, f'Cell screen*: p={p_text}, q={q_text}', transform=ax.transAxes, ha='center', fontsize=NOTE_SIZE_PT, color='#555555')
    ax.tick_params(axis='y', labelsize=TICK_SIZE_PT)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

def _format_adjusted_p(value):
    if value == 0 or value < 1e-300:
        return '<1e−300'
    if value >= 0.001:
        return f'{value:.3g}' + (' (ns)' if value >= 0.05 else '')
    return f'{value:.1e}'.replace('e-', 'e−')

def _add_de_bracket(ax, left, right, y, height, label):
    ax.plot([left, left, right, right], [y, y + height, y + height, y], color='#555555', lw=BRACKET_LINE_WIDTH, clip_on=False)
    ax.text((left + right) / 2, y + height, f'q={label}', ha='center', va='bottom', fontsize=NOTE_SIZE_PT, color='#444444')

def _treatment_box(ax, letter, patient, statistics, association_id):
    item = next((x for x in TREATMENT_PANELS if x['association_id'] == association_id))
    block = patient.loc[patient.association_id.eq(association_id)].copy()
    stat = statistics.loc[statistics.association_id.eq(association_id)].copy()
    positions = [0.0, 1.0, 2.0 + PAIR_GAP, 3.0 + PAIR_GAP]
    statuses = ('Favorable', 'Unfavorable')
    colors = (FAV_COLOR, UNFAV_COLOR)
    arrays = []
    tick_labels = []
    point_records = []
    rng = np.random.default_rng(RANDOM_SEED + ord(letter))
    for (group_index, (status, group_label, color)) in enumerate(zip(statuses, item['group_labels'], colors)):
        group = block.loc[block.status.eq(status)].copy()
        wide = group.pivot(index='patient', columns='timepoint_order', values='log2_median_score')
        if list(wide.columns) != [0, 1] or wide.isna().any().any():
            raise RuntimeError(f'Incomplete paired values for {association_id}/{group_label}')
        jitter = rng.uniform(-0.1, 0.1, len(wide))
        group_positions = positions[group_index * 2:group_index * 2 + 2]
        for (timepoint_order, position) in enumerate(group_positions):
            values = wide[timepoint_order].to_numpy()
            arrays.append(values)
            timepoint = group.loc[group.timepoint_order.eq(timepoint_order), 'timepoint'].iloc[0]
            tick_labels.append(f'{group_label}\n{timepoint}')
            point_records.append((position, jitter, values, color))
    boxes = ax.boxplot(arrays, positions=positions, widths=BOX_WIDTH, patch_artist=True, showfliers=False, medianprops={'color': '#222222', 'linewidth': 0.8}, whiskerprops={'linewidth': 0.65}, capprops={'linewidth': 0.65}, boxprops={'linewidth': 0.65})
    for (index, (box, color)) in enumerate(zip(boxes['boxes'], colors * 2)):
        box.set_facecolor(colors[index // 2])
        box.set_alpha(PRE_BOX_ALPHA if index % 2 == 0 else ON_BOX_ALPHA)
    for (position, jitter, values, color) in point_records:
        ax.scatter(position + jitter, values, s=PATIENT_POINT_SIZE, facecolor='white', edgecolor=color, lw=PATIENT_POINT_EDGE_WIDTH, zorder=3)
    values = block.log2_median_score.to_numpy()
    (low, high) = (float(values.min()), float(values.max()))
    span = max(high - low, 0.5)
    bracket_y = high + BRACKET_OFFSET_FRACTION * span
    bracket_height = BRACKET_HEIGHT_FRACTION * span
    for (group_index, group_label) in enumerate(item['group_labels']):
        q_value = float(stat.loc[stat.group_label.eq(group_label), 'de_adj_p_value'].iloc[0])
        (left, right) = positions[group_index * 2:group_index * 2 + 2]
        _add_de_bracket(ax, left, right, bracket_y, bracket_height, _format_adjusted_p(q_value))
    ax.set_ylim(low - 0.08 * span, bracket_y + 0.2 * span)
    ax.set_xticks(positions, tick_labels, fontsize=TICK_SIZE_PT)
    _format_activity_axis(ax)
    display_cell_type = 'Tumor' if item['cell_type'] == 'Tumor/epithelial' else item['cell_type']
    ax.set_title(f"{letter}  {item['cohort']} · {display_cell_type}\n" + '\n'.join(textwrap.wrap(f"{item['comparison']} · {item['display_task']}", 28)), loc='left', fontsize=TITLE_SIZE_PT - 1.2, fontweight='bold', pad=3)
    interaction_p = float(stat.patient_interaction_p.iloc[0])
    ax.text(0.5, -0.27, f'Exact between-group Δ p={interaction_p:.2g}', transform=ax.transAxes, ha='center', fontsize=NOTE_SIZE_PT)
    ax.tick_params(axis='y', labelsize=TICK_SIZE_PT)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.grid(False)

def _treatment_figure(paths):
    patient = pd.read_csv(paths['treatment_patient'])
    statistics = pd.read_csv(paths['treatment_statistics'])
    fig = plt.figure(figsize=(MAIN_WIDTH_MM * MM, TREATMENT_HEIGHT_MM * MM))
    grid = fig.add_gridspec(1, 4, left=0.055, right=0.992, bottom=0.25, top=0.78, wspace=0.58)
    for (axis, letter, item) in zip([fig.add_subplot(grid[0, index]) for index in range(4)], 'ABCD', TREATMENT_PANELS):
        _treatment_box(axis, letter, patient, statistics, item['association_id'])
    fig.text(0.5, 0.035, 'Paired points are patient medians (≥20 cells at both timepoints) · q = DE BH-adjusted pooled-cell Wilcoxon p, descriptive only · Bassez E/NE denotes clonotype expansion', ha='center', fontsize=NOTE_SIZE_PT, color='#555555')
    return fig

def _patient_delta_table(patient, association_id):
    """Return one paired treatment-minus-baseline delta per eligible patient."""
    block = patient.loc[patient.association_id.eq(association_id)].copy()
    identity = ['patient', 'status']
    wide = block.pivot(index=identity, columns='timepoint_order', values='log2_median_score')
    if list(wide.columns) != [0, 1] or wide.isna().any().any():
        raise RuntimeError(f'Incomplete paired patient values for {association_id}')
    delta = wide.rename(columns={0: 'baseline', 1: 'treatment'}).reset_index()
    delta['delta_log2'] = delta.treatment - delta.baseline
    return delta

def _supplementary_patient_deltas(paths):
    """Patient-replicate concordance view for longitudinal panels I--L."""
    patient = pd.read_csv(paths['treatment_patient'])
    statistics = pd.read_csv(paths['treatment_statistics'])
    delta_tables = {item['association_id']: _patient_delta_table(patient, item['association_id']) for item in TREATMENT_PANELS}
    max_abs = max((float(frame.delta_log2.abs().max()) for frame in delta_tables.values()))
    axis_limit = np.ceil(max_abs / DELTA_AXIS_ROUNDING) * DELTA_AXIS_ROUNDING
    (fig, axes) = plt.subplots(1, 4, sharey=True, figsize=(MAIN_WIDTH_MM * MM, DELTA_HEIGHT_MM * MM))
    fig.subplots_adjust(left=0.065, right=0.992, bottom=0.26, top=0.77, wspace=0.32)
    for (axis, letter, main_panel, item) in zip(axes, 'ABCD', 'IJKL', TREATMENT_PANELS):
        association_id = item['association_id']
        delta = delta_tables[association_id]
        stat = statistics.loc[statistics.association_id.eq(association_id)]
        rng = np.random.default_rng(RANDOM_SEED + ord(letter))
        for (x_position, (status, group_label, color)) in enumerate(zip(('Favorable', 'Unfavorable'), item['group_labels'], (FAV_COLOR, UNFAV_COLOR))):
            values = delta.loc[delta.status.eq(status), 'delta_log2'].to_numpy()
            expected_n = item['expected_patients'][x_position]
            if len(values) != expected_n:
                raise RuntimeError(f'Unexpected patient count for {association_id}/{group_label}: {len(values)} != {expected_n}')
            jitter = rng.uniform(-DELTA_JITTER_WIDTH, DELTA_JITTER_WIDTH, len(values))
            axis.scatter(x_position + jitter, values, s=DELTA_POINT_SIZE, facecolor='white', edgecolor=color, linewidth=DELTA_POINT_EDGE_WIDTH, zorder=3)
            median = float(np.median(values))
            axis.plot([x_position - DELTA_MEDIAN_HALF_WIDTH, x_position + DELTA_MEDIAN_HALF_WIDTH], [median, median], color=color, linewidth=DELTA_MEDIAN_LINE_WIDTH, solid_capstyle='round', zorder=4)
            direction = 1 if median > 0 else -1
            concordant = int((np.sign(values) == direction).sum())
            arrow = '↑' if direction > 0 else '↓'
            axis.text(x_position, -0.17, f'{concordant}/{len(values)} {arrow}', transform=axis.get_xaxis_transform(), ha='center', va='top', fontsize=NOTE_SIZE_PT, color=color)
        axis.axhline(0, color=DELTA_ZERO_LINE_COLOR, linewidth=DELTA_ZERO_LINE_WIDTH, linestyle='--', zorder=1)
        axis.set_xlim(-0.52, 1.52)
        axis.set_ylim(-axis_limit, axis_limit)
        axis.set_xticks((0, 1), item['group_labels'], fontsize=TICK_SIZE_PT)
        axis.tick_params(axis='y', labelsize=TICK_SIZE_PT)
        display_cell_type = 'Tumor' if item['cell_type'] == 'Tumor/epithelial' else item['cell_type']
        title = f"{letter}  Main {main_panel} · {item['cohort']} · {display_cell_type}\n" + '\n'.join(textwrap.wrap(f"{item['comparison']} · {item['display_task']}", 27))
        axis.text(0, 1.17, title, transform=axis.transAxes, ha='left', va='bottom', fontsize=TITLE_SIZE_PT - 1.2, fontweight='bold', clip_on=False)
        interaction_p = float(stat.patient_interaction_p.iloc[0])
        axis.text(0, 1.04, f'Exact between-group Δ p={interaction_p:.3g} (nominal)', transform=axis.transAxes, ha='left', va='bottom', fontsize=NOTE_SIZE_PT, color='#444444', clip_on=False)
        for spine in ('top', 'right'):
            axis.spines[spine].set_visible(False)
        axis.grid(False)
    axes[0].set_ylabel('Treatment − baseline\nΔ log2(patient-median Method 5 activity)', fontsize=NOTE_SIZE_PT)
    fig.text(0.5, 0.035, 'One point per paired patient (≥20 target cells/timepoint) · colored bar = group median Δ · arrows report concordance with the median direction · exact two-sided Mann–Whitney test compares patient Δ between groups; nominal p, no multiplicity correction · Bassez E/NE denotes clonotype expansion', ha='center', fontsize=NOTE_SIZE_PT, color='#555555')
    return fig

def _main(root, paths):
    v8 = prepare_v8_data(root, force=False)
    patient = pd.read_csv(v8['eligible_patient_scores.csv.gz'])
    effects = pd.read_csv(paths['effects'])
    treatment_patient = pd.read_csv(paths['treatment_patient'])
    treatment_statistics = pd.read_csv(paths['treatment_statistics'])
    fig = plt.figure(figsize=(MAIN_WIDTH_MM * MM, 165 * MM))
    outer = fig.add_gridspec(3, 1, height_ratios=(0.82, 0.52, 0.68), hspace=0.42, left=0.055, right=0.992, bottom=0.06, top=0.985)
    _upset(fig.add_subplot(outer[0]), root)
    baseline = outer[1].subgridspec(1, 2, wspace=0.38)
    for (ax, spec) in zip([fig.add_subplot(baseline[0, j]) for j in range(2)], [('E', 'RNR_TCELL_ERAD', 'R/NR'), ('F', 'ENE_MYELOID_GLYCOGEN', 'E/NE')]):
        _box(ax, spec[0], patient, effects, spec[1], spec[2])
    longitudinal = outer[2].subgridspec(1, 4, wspace=0.58)
    for (axis, letter, item) in zip([fig.add_subplot(longitudinal[0, j]) for j in range(4)], 'GHIJ', TREATMENT_PANELS):
        _treatment_box(axis, letter, treatment_patient, treatment_statistics, item['association_id'])
    fig.text(0.5, 0.008, 'Patient medians are the biological replicate · pooled-cell Wilcoxon/BH values are descriptive · Bassez E/NE denotes clonotype expansion, not clinical response', ha='center', fontsize=NOTE_SIZE_PT, color='#555555')
    return fig

def _save_full_canvas(fig, stem, close=True):
    """Save without tight cropping so the reserved lower composition area remains blank."""
    result = {}
    for extension in ('png', 'pdf', ):
        path = stem.with_suffix('.' + extension)
        fig.savefig(path, dpi=OUTPUT_DPI, facecolor='white')
        result[extension] = str(path)
    if close:
        plt.close(fig)
    return result

def _supplementary_evidence(paths):
    evidence = pd.read_csv(paths['evidence'])
    effects = pd.read_csv(paths['effects'])
    fig = plt.figure(figsize=(MAIN_WIDTH_MM * MM, 62 * MM))
    axis = fig.add_axes([0.07, 0.22, 0.915, 0.68])
    _evidence_summary(axis, evidence, effects)
    fig.text(0.5, 0.025, 'Patient-level evidence from the frozen v8 screen · all BH q ≥ 0.05 (exploratory) · Bassez E/NE is not clinical response', ha='center', fontsize=NOTE_SIZE_PT, color='#555555')
    return fig

def _supplementary_unbiased_audit(root):
    data = prepare_v8_data(root, force=False)
    patient = pd.read_csv(data['eligible_patient_scores.csv.gz'])
    rnr_screen = pd.read_csv(data['rnr_full_screen.csv'])
    ene_screen = pd.read_csv(data['ene_full_screen.csv'])
    ranked_rnr = pd.read_csv(data['rnr_ranked_candidates.csv'])
    ranked_ene = pd.read_csv(data['ene_ranked_candidates.csv'])
    selected_rnr = pd.read_csv(data['rnr_selected_tasks.csv'])
    selected_ene = pd.read_csv(data['ene_selected_tasks.csv'])
    fig = plt.figure(figsize=(MAIN_WIDTH_MM * MM, AUDIT_HEIGHT_MM * MM))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.5, 1.24, 1.65], hspace=0.48, left=0.06, right=0.985, bottom=0.06, top=0.985)
    _upset(fig.add_subplot(outer[0]), root)
    selection = outer[1].subgridspec(1, 2, wspace=0.42)
    _dotplot(fig.add_subplot(selection[0]), 'B', 'Clinical R/NR candidates', ranked_rnr, 'abs_delta', ('R', 'NR'))
    _dotplot(fig.add_subplot(selection[1]), 'C', 'Clonotype E/NE candidates', ranked_ene, 'min_abs_delta', ('E', 'NE'))
    boxes = outer[2].subgridspec(2, 3, hspace=0.72, wspace=0.32)
    for (index, (letter, cell_type)) in enumerate(zip('FGH', COMPARTMENTS)):
        task = selected_rnr.loc[selected_rnr.cell_type.eq(cell_type), 'task'].iloc[0]
        axis = fig.add_subplot(boxes[0, index])
        _box_panel(axis, letter, cell_type, task, patient, ('Shiao',), ('R', 'NR'), rnr_screen, 'R/NR')
        _format_activity_axis(axis)
    for (index, (letter, cell_type)) in enumerate(zip('IJK', COMPARTMENTS)):
        task = selected_ene.loc[selected_ene.cell_type.eq(cell_type), 'task'].iloc[0]
        axis = fig.add_subplot(boxes[1, index])
        _box_panel(axis, letter, cell_type, task, patient, ('Bassez A', 'Bassez B'), ('E', 'NE'), ene_screen, 'E/NE')
        _format_activity_axis(axis)
    fig.text(0.5, 0.018, 'Points are patient medians · exact two-sided permutation tests · BH within endpoint screen · Bassez E/NE is not clinical response · logarithmic spacing; ticks show raw activity (%)', ha='center', fontsize=NOTE_SIZE_PT, color='#555555')
    return fig

def _clean_umap_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def _draw_background(ax, background):
    ax.scatter(background.umap_1, background.umap_2, s=UMAP_BACKGROUND_SIZE, color=UMAP_BACKGROUND_COLOR, alpha=UMAP_BACKGROUND_ALPHA, linewidths=0, rasterized=UMAP_RASTERIZED)

def _draw_baseline_map(ax, background, overlay, row, letter):
    bg = background.loc[background.cohort.eq(row.cohort) & background.condition.eq(row.condition)]
    points = overlay.loc[overlay.map_id.eq(row.map_id)].sort_values('log10_score_plus_epsilon')
    norm = Normalize(row.color_low, row.color_high)
    _draw_background(ax, bg)
    ax.scatter(points.umap_1, points.umap_2, c=points.log10_score_plus_epsilon, cmap=UMAP_CMAP, norm=norm, s=UMAP_OVERLAY_SIZE, alpha=UMAP_OVERLAY_ALPHA, linewidths=0, rasterized=UMAP_RASTERIZED)
    task_title = '\n'.join(textwrap.wrap(f'{row.condition} · {row.display_task}', 31))
    ax.set_title(f'{letter}  {row.main_panel} · {row.cohort} · {row.cell_type}\n{task_title}', loc='left', fontsize=UMAP_TITLE_SIZE_PT, fontweight='bold', pad=3)
    ax.text(0.5, -0.08, f'{row.patient_count_label} · {int(row.n_overlay_cells):,} colored cells', transform=ax.transAxes, ha='center', va='top', fontsize=UMAP_NOTE_SIZE_PT)
    _clean_umap_axis(ax)
    return ScalarMappable(norm=norm, cmap=UMAP_CMAP)

def supplementary_baseline_umaps(paths: Mapping[str, Path]):
    background = pd.read_parquet(paths['background'])
    overlay = pd.read_parquet(paths['overlay'])
    metadata = pd.read_csv(paths['metadata']).query("figure_kind == 'baseline'").sort_values('display_order')
    if len(metadata) != 6:
        raise RuntimeError('Expected six baseline task-UMAP maps')
    (fig, axes) = plt.subplots(2, 3, figsize=(180 * MM, UMAP_BASELINE_HEIGHT_MM * MM))
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.12, top=0.94, wspace=0.24, hspace=0.48)
    for (ax, letter, row) in zip(axes.flat, 'ABCDEF', metadata.itertuples(index=False)):
        sm = _draw_baseline_map(ax, background, overlay, row, letter)
        cb = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.035, pad=0.13, aspect=25)
        cb.set_label('log10(score + 1e−10)', fontsize=UMAP_COLORBAR_SIZE_PT, labelpad=1)
        cb.ax.tick_params(labelsize=UMAP_COLORBAR_SIZE_PT, length=2)
    fig.text(0.5, 0.018, 'Frozen embeddings · grey = all endpoint-annotated cells in condition · color = eligible target-cell activity · descriptive localization only; compare colors within association', ha='center', fontsize=UMAP_NOTE_SIZE_PT, color='#555555')
    return fig

def _draw_activity_facet(ax, background, points, row, status, group_label, condition, title):
    bg = background.loc[background.cohort.eq(row.cohort) & background.condition.eq(condition)]
    selected = points.loc[points.condition.eq(condition) & points.status.eq(status)].sort_values('log10_score_plus_epsilon')
    if selected.empty:
        raise RuntimeError(f'Empty longitudinal facet: {row.association_id}/{group_label}/{condition}')
    norm = Normalize(row.color_low, row.color_high)
    _draw_background(ax, bg)
    ax.scatter(selected.umap_1, selected.umap_2, c=selected.log10_score_plus_epsilon, cmap=UMAP_CMAP, norm=norm, s=UMAP_OVERLAY_SIZE, alpha=UMAP_OVERLAY_ALPHA, linewidths=0, rasterized=UMAP_RASTERIZED)
    ax.set_title(title, loc='left', fontsize=UMAP_TITLE_SIZE_PT, fontweight='bold', pad=2)
    ax.text(0.5, -0.07, f'n={selected.patient.nunique()} patients · {len(selected):,} cells', transform=ax.transAxes, ha='center', va='top', fontsize=UMAP_NOTE_SIZE_PT)
    _clean_umap_axis(ax)
    return ScalarMappable(norm=norm, cmap=UMAP_CMAP)

def _draw_categorical_facet(ax, background, points, row, labels):
    conditions = sorted(points.condition.unique(), key=lambda value: 0 if value in {'Base', 'Pre'} else 1)
    bg = background.loc[background.cohort.eq(row.cohort) & background.condition.isin(conditions)]
    if points.cell_id.duplicated().any():
        raise RuntimeError(f'Duplicated cells in categorical facet: {row.association_id}')
    _draw_background(ax, bg)
    for (status, label, color) in zip(('Favorable', 'Unfavorable'), labels, (FAV_COLOR, UNFAV_COLOR)):
        selected = points.loc[points.status.eq(status)]
        ax.scatter(selected.umap_1, selected.umap_2, s=UMAP_OVERLAY_SIZE, color=color, alpha=UMAP_OVERLAY_ALPHA, linewidths=0, rasterized=UMAP_RASTERIZED)
    ax.set_title(f'{labels[0]} vs {labels[1]} · both timepoints', loc='left', fontsize=UMAP_TITLE_SIZE_PT, fontweight='bold', pad=2)
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=3.4, color=color, label=label) for (label, color) in zip(labels, (FAV_COLOR, UNFAV_COLOR))]
    ax.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=UMAP_NOTE_SIZE_PT, handletextpad=0.25, columnspacing=0.7)
    counts = []
    for (status, label) in zip(('Favorable', 'Unfavorable'), labels):
        selected = points.loc[points.status.eq(status)]
        counts.append(f'{label}: n={selected.patient.nunique()}, {len(selected):,} cells')
    ax.text(0.5, -0.07, ' · '.join(counts), transform=ax.transAxes, ha='center', va='top', fontsize=UMAP_NOTE_SIZE_PT)
    _clean_umap_axis(ax)

def supplementary_longitudinal_umaps(paths: Mapping[str, Path]):
    background = pd.read_parquet(paths['background'])
    overlay = pd.read_parquet(paths['overlay'])
    metadata = pd.read_csv(paths['metadata']).query("figure_kind == 'longitudinal'")
    panels = list('IJKL')
    if len(metadata) != 8 or sorted(metadata.main_panel.unique()) != panels:
        raise RuntimeError('Expected two staged conditions for each longitudinal panel I-L')
    (fig, axes) = plt.subplots(4, 5, figsize=(180 * MM, UMAP_LONGITUDINAL_HEIGHT_MM * MM))
    fig.subplots_adjust(left=0.035, right=0.992, bottom=0.045, top=0.965, wspace=0.13, hspace=0.7)
    for (row_index, (letter, panel)) in enumerate(zip('ABCD', panels)):
        group_metadata = metadata.loc[metadata.main_panel.eq(panel)].sort_values('condition_order')
        (baseline_row, treatment_row) = tuple(group_metadata.itertuples(index=False))
        points = overlay.loc[overlay.association_id.eq(baseline_row.association_id)].copy()
        labels = ('R', 'NR') if baseline_row.cohort == 'Shiao' else ('E', 'NE')
        conditions = (baseline_row.condition, treatment_row.condition)
        row_title = f'{letter}  {panel} · {baseline_row.cohort} · {baseline_row.cell_type} · {baseline_row.display_task}'
        axes[row_index, 0].text(0, 1.16, row_title, transform=axes[row_index, 0].transAxes, fontsize=UMAP_TITLE_SIZE_PT, fontweight='bold', ha='left', va='bottom', clip_on=False)
        sm = None
        for (column, (status, group_label, condition)) in enumerate((('Favorable', labels[0], conditions[0]), ('Favorable', labels[0], conditions[1]), ('Unfavorable', labels[1], conditions[0]), ('Unfavorable', labels[1], conditions[1]))):
            sm = _draw_activity_facet(axes[row_index, column], background, points, baseline_row, status, group_label, condition, f'{group_label} · {condition}')
        _draw_categorical_facet(axes[row_index, 4], background, points, baseline_row, labels)
        cb = fig.colorbar(sm, ax=axes[row_index, :4].tolist(), orientation='horizontal', fraction=0.018, pad=0.085, aspect=58)
        cb.set_label('log10(score + 1e−10) · shared across four activity maps', fontsize=UMAP_COLORBAR_SIZE_PT, labelpad=1)
        cb.ax.tick_params(labelsize=UMAP_COLORBAR_SIZE_PT, length=2)
    fig.text(0.5, 0.009, 'Frozen embeddings · activity facets use eligible target cells from one group/timepoint · fifth map pools both retained timepoints and colors target cells by E/NE or R/NR · descriptive only', ha='center', fontsize=UMAP_NOTE_SIZE_PT, color='#555555')
    return fig

def renderer_metadata() -> dict:
    return {'python': platform.python_version(), 'matplotlib': mpl.__version__, 'cnsplots': 'unavailable; established matplotlib fallback', 'umap_coordinates': 'frozen X_umap; no embedding recomputed', 'activity_transformation': DISPLAY_TRANSFORMATION, 'activity_colormap': UMAP_CMAP, 'display_epsilon': EPSILON, 'color_clipping': 'association-specific q02-q98 shared across four group-by-timepoint activity facets', 'longitudinal_layout': 'four tasks x five maps; fifth categorical map pools both retained timepoints'}

def supplementary_unbiased_audit_compact(root):
    data = prepare_v8_data(root, force=False)
    patient = pd.read_csv(data['eligible_patient_scores.csv.gz'])
    rnr_screen = pd.read_csv(data['rnr_full_screen.csv'])
    ene_screen = pd.read_csv(data['ene_full_screen.csv'])
    ranked_rnr = pd.read_csv(data['rnr_ranked_candidates.csv'])
    ranked_ene = pd.read_csv(data['ene_ranked_candidates.csv'])
    selected_rnr = pd.read_csv(data['rnr_selected_tasks.csv'])
    selected_ene = pd.read_csv(data['ene_selected_tasks.csv'])
    fig = plt.figure(figsize=(MAIN_WIDTH_MM * MM, 188 * MM))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.82, 1, 1], hspace=0.63, left=0.065, right=0.985, bottom=0.07, top=0.97)
    top = outer[0].subgridspec(1, 2, wspace=0.42)
    _dotplot(fig.add_subplot(top[0]), 'A', 'Clinical R/NR candidates', ranked_rnr, 'abs_delta', ('R', 'NR'))
    _dotplot(fig.add_subplot(top[1]), 'B', 'Clonotype E/NE candidates', ranked_ene, 'min_abs_delta', ('E', 'NE'))
    row_specs = (('CDE', selected_rnr, rnr_screen, ('Shiao',), ('R', 'NR'), 'R/NR'), ('FGH', selected_ene, ene_screen, ('Bassez A', 'Bassez B'), ('E', 'NE'), 'E/NE'))
    for (row_index, (letters, selected, screen, cohorts, labels, endpoint)) in enumerate(row_specs, start=1):
        grid = outer[row_index].subgridspec(1, 3, wspace=0.35)
        for (index, (letter, cell_type)) in enumerate(zip(letters, COMPARTMENTS)):
            task = selected.loc[selected.cell_type.eq(cell_type), 'task'].iloc[0]
            axis = fig.add_subplot(grid[index])
            _box_panel(axis, letter, cell_type, task, patient, cohorts, labels, screen, endpoint)
            _format_activity_axis(axis)
    fig.text(0.5, 0.018, 'Complete endpoint-specific audit · one median per patient · exact two-sided permutation tests · BH within endpoint screen · Bassez E/NE denotes clonotype expansion', ha='center', fontsize=NOTE_SIZE_PT, color='#555555')
    return fig

def render_main_method_overlap():
    (fig, ax) = plt.subplots(figsize=(MAIN_WIDTH_MM * MM, 68 * MM))
    fig.subplots_adjust(left=0.055, right=0.992, bottom=0.12, top=0.96)
    _upset(ax, ROOT)
    return fig

def render_main_baseline(letter, association_id, endpoint_title):
    patient_frame = pd.read_csv(DATA_DIR / 'eligible_patient_scores.csv.gz')
    effects_frame = pd.read_csv(PLOT_PATHS['effects'])
    (fig, ax) = plt.subplots(figsize=(86 * MM, 62 * MM))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.2, top=0.84)
    _box(ax, letter, patient_frame, effects_frame, association_id, endpoint_title)
    return fig

def render_main_longitudinal(letter, association_id):
    patient_frame = pd.read_csv(PLOT_PATHS['treatment_patient'])
    statistics_frame = pd.read_csv(PLOT_PATHS['treatment_statistics'])
    (fig, ax) = plt.subplots(figsize=(44 * MM, 68 * MM))
    fig.subplots_adjust(left=0.25, right=0.97, bottom=0.18, top=0.82)
    _treatment_box(ax, letter, patient_frame, statistics_frame, association_id)
    return fig


def render_main_method_overlap_single(letter):
    comparisons = dict(zip("ABCD", PANEL_A_SUMMARY.comparison.drop_duplicates().tolist()))
    if letter not in comparisons:
        raise ValueError(letter)
    fig, ax = plt.subplots(figsize=(48 * MM, 68 * MM))
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.12, top=0.96)
    _upset(ax, ROOT, selected_comparison=comparisons[letter])
    return fig

def render_all_main_panels():
    figures = {letter: render_main_method_overlap_single(letter) for letter in "ABCD"}
    figures.update({
      "A-D": render_main_method_overlap(),
      "E": render_main_baseline("E", "RNR_TCELL_ERAD", "R/NR"),
      "F": render_main_baseline("F", "ENE_MYELOID_GLYCOGEN", "E/NE"),
    })
    for letter, item in zip("GHIJ", TREATMENT_PANELS):
        figures[letter] = render_main_longitudinal(letter, item["association_id"])
    return figures

def compile_main_figure():
    return _main(ROOT, PLOT_PATHS)

def export_figure(fig, stem):
    outputs={}
    for suffix in ("png","pdf"):
        path=OUTPUT_DIR/f"{stem}.{suffix}"
        fig.savefig(path,format=suffix,dpi=OUTPUT_DPI,bbox_inches="tight",facecolor="white")
        outputs[suffix]=path
    return outputs
