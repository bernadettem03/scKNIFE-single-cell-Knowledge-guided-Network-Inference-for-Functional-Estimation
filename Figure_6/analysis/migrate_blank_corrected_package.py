#!/usr/bin/env python3
"""Build the public Figure 6/S6 plotting packages from the validated working notebook."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
from itertools import combinations, product
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
ROBUST_METABOLITE_ROOT = (
    WORKSPACE / "May_2026_clean_version_redo/inhouse_rerun/scknife_run/data/robust_metabolite_analysis"
)
METABOLOMICS_ROOT = WORKSPACE / "metabolomics_all_analaysi_part2/data"
FIG6 = REPO / "Figure_6"
S6 = REPO / "Supplementary/Figure_S6"

MAIN_FILES = [
    'meia_vs_scknife_pooled_correlations.csv', 'meia_vs_scknife_pooled_density.csv.gz', 'meia_vs_scknife_pooled_provenance.json',
    "treatment_four_set_membership.csv", "surgery_pathway_scatter.csv", "surgery_pathway_enrichment.csv",
    "meia_vs_scknife_entity_correlations.csv", "robust_metabolite_pathway_assignments.csv",
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

TIER3_ALL_SPECS = [
    ("2E_heptenoyl_CoA", "(2E)-heptenoyl-CoA", "CAFs"),
    ("Glutaryl_Carnitine", "Glutarylcarnitine", "CAFs"),
    ("heparan_sulfate_degradation_product_18", "Heparan sulfate product 18", "T cells"),
    ("VI3NeuAc_nLc6Cer", "VI3NeuAc-nLc6Cer", "T cells"),
]
TIER3_SELECTED = {"2E_heptenoyl_CoA", "Glutaryl_Carnitine"}
TIER3_P3 = "CF/30117"
TIER3_2AEP_METABOLITE = "2_aminoethylphosphonate"
TIER3_2AEP_FEATURE = "neg_124.0035_397.74"


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



def exact_signflip(differences):
    values = np.asarray(differences, dtype=float)
    observed = float(values.mean())
    null = np.asarray([
        (values * np.asarray(signs)).mean()
        for signs in product((-1, 1), repeat=len(values))
    ])
    return observed, float(np.mean(np.abs(null) >= abs(observed) - 1e-12))


def exact_status_test(frame):
    ordered = frame.sort_values("PatientID").reset_index(drop=True)
    if len(ordered) != 6 or ordered.Status.value_counts().to_dict() != {
        "Metastatic": 3, "Non-metastatic": 3,
    }:
        raise RuntimeError("Tier-3 status comparison requires a complete 3-vs-3 discovery cohort")
    values = ordered.activity.to_numpy(dtype=float)
    actual = ordered.Status.eq("Metastatic").to_numpy()
    observed = float(values[actual].mean() - values[~actual].mean())
    null = []
    for indices in combinations(range(6), 3):
        mask = np.zeros(6, dtype=bool)
        mask[list(indices)] = True
        null.append(float(values[mask].mean() - values[~mask].mean()))
    p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    wins = sum(left > right for left in values[actual] for right in values[~actual])
    losses = sum(left < right for left in values[actual] for right in values[~actual])
    return observed, p_value, float((wins - losses) / 9)


def derive_two_node_tier3_evidence(destination):
    metadata_path = ROBUST_METABOLITE_ROOT / "intermediate/cell_metadata.csv.gz"
    scores_path = ROBUST_METABOLITE_ROOT / "intermediate/metabolite_activity_scores.parquet"
    tiers_path = ROBUST_METABOLITE_ROOT / "tables/metabolite_celltype_rebound_tiers.csv"
    if not all(path.exists() for path in (metadata_path, scores_path, tiers_path)):
        raise FileNotFoundError("Validated robust-metabolite source inputs are missing")

    specs = pd.DataFrame(
        TIER3_ALL_SPECS, columns=["metabolite", "display_name", "celltype"]
    )
    specs["display_order"] = np.arange(1, len(specs) + 1)
    metadata = pd.read_csv(metadata_path)
    scores = pd.read_parquet(scores_path, columns=["cell_id", *specs.metabolite])
    if scores.cell_id.astype(str).tolist() != metadata.cell_id.astype(str).tolist():
        raise RuntimeError("Tier-3 score and metadata cell orders differ")
    tiers = pd.read_csv(tiers_path)
    tier_rows = specs.merge(tiers, on=["metabolite", "celltype"], validate="one_to_one")
    if not tier_rows.tier.eq(3).all() or not tier_rows.tier3_baseline_like_recurrence.all():
        raise RuntimeError("Tier-3 node classification changed")

    patient_parts = []
    p3_parts = []
    for spec in specs.itertuples(index=False):
        frame = metadata[["cell_id", "PatientID", "Timepoint", "Status", "celltype"]].copy()
        frame["activity"] = scores[spec.metabolite].to_numpy(dtype=float)
        frame = frame.loc[frame.celltype.eq(spec.celltype)].dropna(subset=["activity"])
        medians = (
            frame.groupby(["PatientID", "Timepoint", "Status", "celltype"], as_index=False)
            .agg(activity=("activity", "median"), n_cells=("activity", "size"))
        )
        medians["metabolite"] = spec.metabolite
        medians["display_name"] = spec.display_name
        medians["display_order"] = spec.display_order
        patient_parts.append(medians)
        if spec.metabolite in TIER3_SELECTED:
            p3 = frame.loc[
                frame.PatientID.eq(TIER3_P3)
                & frame.Timepoint.isin(["Bx", "Surgery", "Recurrence"]),
                ["cell_id", "PatientID", "Timepoint", "Status", "celltype", "activity"],
            ].copy()
            p3["metabolite"] = spec.metabolite
            p3["display_name"] = spec.display_name
            p3["display_order"] = spec.display_order
            p3_parts.append(p3)

    patient_activity = pd.concat(patient_parts, ignore_index=True)
    p3_cells = pd.concat(p3_parts, ignore_index=True)
    discovery = patient_activity.loc[
        patient_activity.PatientID.ne(TIER3_P3)
        & patient_activity.Timepoint.isin(["Bx", "Surgery"])
    ].copy()
    if not discovery.groupby("metabolite").PatientID.nunique().eq(6).all():
        raise RuntimeError("Tier-3 discovery cohort coverage changed")

    treatment_rows = []
    status_rows = []
    status_values = []
    for spec in specs.itertuples(index=False):
        current = discovery.loc[discovery.metabolite.eq(spec.metabolite)]
        wide = current.pivot(
            index=["PatientID", "Status"], columns="Timepoint", values="activity"
        ).reset_index()
        wide["delta"] = wide.Surgery - wide.Bx
        effect, p_value = exact_signflip(wide.delta)
        treatment_rows.append({
            "metabolite": spec.metabolite,
            "treatment_effect": effect,
            "treatment_median_delta": float(wide.delta.median()),
            "treatment_exact_p": p_value,
            "treatment_n_decrease": int(wide.delta.lt(0).sum()),
        })
        wide["activity"] = (wide.Bx + wide.Surgery) / 2
        status_effect, status_p, cliffs_delta = exact_status_test(wide)
        status_rows.append({
            "metabolite": spec.metabolite,
            "status_effect": status_effect,
            "status_cliffs_delta": cliffs_delta,
            "status_exact_p": status_p,
        })
        wide["metabolite"] = spec.metabolite
        status_values.append(wide[["PatientID", "Status", "activity", "metabolite"]])

    treatment = pd.DataFrame(treatment_rows)
    treatment["treatment_BH_FDR_4"] = multipletests(
        treatment.treatment_exact_p, method="fdr_bh"
    )[1]
    status_stats = pd.DataFrame(status_rows)
    status_stats["status_BH_FDR_4"] = multipletests(
        status_stats.status_exact_p, method="fdr_bh"
    )[1]
    statistics = (
        specs.merge(treatment, on="metabolite", validate="one_to_one")
        .merge(status_stats, on="metabolite", validate="one_to_one")
        .merge(
            tier_rows[[
                "metabolite", "celltype", "p4_tx_rank_biserial_effect",
                "p4_tx_global_contrast_BH_FDR", "p4_rebound_rank_biserial_effect",
                "p4_rebound_global_contrast_BH_FDR", "p4_baseline_rank_biserial_effect",
                "p4_baseline_global_contrast_BH_FDR",
            ]],
            on=["metabolite", "celltype"], validate="one_to_one",
        )
    )

    selected_stats = statistics.loc[statistics.metabolite.isin(TIER3_SELECTED)].copy()
    selected_stats = selected_stats.sort_values("display_order")
    selected_patients = patient_activity.loc[
        patient_activity.metabolite.isin(TIER3_SELECTED)
        & patient_activity.Timepoint.isin(["Bx", "Surgery"])
    ].copy()
    selected_status = pd.concat(status_values, ignore_index=True)
    selected_status = selected_status.loc[selected_status.metabolite.isin(TIER3_SELECTED)]
    selected_patients = selected_patients.merge(
        selected_status.rename(columns={"activity": "status_activity"}),
        on=["PatientID", "Status", "metabolite"], how="left", validate="many_to_one",
    )
    selected_patients["is_discovery"] = selected_patients.PatientID.ne(TIER3_P3)

    if len(selected_stats) != 2 or len(selected_patients) != 28:
        raise RuntimeError("Two-node frozen table dimensions changed")
    p3_coverage = p3_cells.groupby("metabolite").Timepoint.nunique()
    if len(p3_coverage) != 2 or not p3_coverage.eq(3).all():
        raise RuntimeError("Two-node P3 timepoint coverage changed")
    expected = selected_stats.set_index("metabolite")
    if not np.isclose(expected.loc["2E_heptenoyl_CoA", "treatment_BH_FDR_4"], .125):
        raise RuntimeError("(2E)-heptenoyl-CoA treatment q4 changed")
    if not np.isclose(expected.loc["Glutaryl_Carnitine", "treatment_BH_FDR_4"], .125):
        raise RuntimeError("Glutarylcarnitine treatment q4 changed")
    if not np.allclose(selected_stats.status_BH_FDR_4, .8):
        raise RuntimeError("Two-node status q4 changed")

    selected_patients.to_csv(destination / "tier3_two_node_patient_values.csv", index=False)
    p3_cells.to_csv(destination / "tier3_two_node_p3_cells.csv.gz", index=False)
    selected_stats.to_csv(destination / "tier3_two_node_statistics.csv", index=False)


def derive_2aep_status_evidence(destination):
    metadata_path = ROBUST_METABOLITE_ROOT / "intermediate/cell_metadata.csv.gz"
    scores_path = ROBUST_METABOLITE_ROOT / "intermediate/metabolite_activity_scores.parquet"
    tiers_path = ROBUST_METABOLITE_ROOT / "tables/metabolite_celltype_rebound_tiers.csv"
    representatives_path = ROBUST_METABOLITE_ROOT / "tables/lcms_tier3_crossmatch_representatives.csv"
    lcms_stats_path = (
        METABOLOMICS_ROOT
        / "primary/contrasts/metastatic_vs_nonmetastatic/supplementary_feature_statistics.csv"
    )
    lcms_matrix_path = METABOLOMICS_ROOT / "primary/shared/specimen_log2_intensities_negative.csv"
    lcms_clinical_path = METABOLOMICS_ROOT / "clinical_specimen_metadata.csv"
    inputs = (
        metadata_path, scores_path, tiers_path, representatives_path,
        lcms_stats_path, lcms_matrix_path, lcms_clinical_path,
    )
    if not all(path.exists() for path in inputs):
        raise FileNotFoundError("Validated 2-aminoethylphosphonate source inputs are missing")

    metadata = pd.read_csv(metadata_path)
    tiers = pd.read_csv(tiers_path)
    tier3_pairs = tiers.loc[tiers.tier.eq(3), ["celltype", "metabolite"]].drop_duplicates()
    if len(tier3_pairs) != 70 or tier3_pairs.metabolite.nunique() != 68:
        raise RuntimeError("Tier-3 70-pair/68-metabolite screen changed")
    metabolites = sorted(tier3_pairs.metabolite.unique())
    scores = pd.read_parquet(scores_path, columns=["cell_id", *metabolites])
    if scores.cell_id.astype(str).tolist() != metadata.cell_id.astype(str).tolist():
        raise RuntimeError("Tier-3 score and metadata cell orders differ")

    screen_rows = []
    patient_parts = []
    for pair in tier3_pairs.itertuples(index=False):
        frame = metadata[["cell_id", "PatientID", "Timepoint", "Status", "celltype"]].copy()
        frame["activity"] = scores[pair.metabolite].to_numpy(dtype=float)
        frame = frame.loc[
            frame.celltype.eq(pair.celltype)
            & frame.PatientID.ne(TIER3_P3)
            & frame.Timepoint.isin(["Bx", "Surgery"])
        ].dropna(subset=["activity"])
        patient = (
            frame.groupby(["PatientID", "Status", "Timepoint"], as_index=False).activity.median()
            .groupby(["PatientID", "Status"], as_index=False).activity.mean()
        )
        effect, p_value, cliffs_delta = exact_status_test(patient)
        screen_rows.append({
            "metabolite": pair.metabolite,
            "celltype": pair.celltype,
            "effect": effect,
            "cliffs_delta": cliffs_delta,
            "exact_p": p_value,
        })
        patient["metabolite"] = pair.metabolite
        patient["celltype"] = pair.celltype
        patient_parts.append(patient)

    screen = pd.DataFrame(screen_rows)
    screen["BH_FDR_70"] = multipletests(screen.exact_p, method="fdr_bh")[1]
    if len(screen) != 70:
        raise RuntimeError("Tier-3 status screen did not produce 70 tests")
    target_stats = screen.loc[
        screen.metabolite.eq(TIER3_2AEP_METABOLITE) & screen.celltype.eq("T cells")
    ]
    target_patients = pd.concat(patient_parts, ignore_index=True).loc[
        lambda x: x.metabolite.eq(TIER3_2AEP_METABOLITE) & x.celltype.eq("T cells")
    ].copy()
    if len(target_stats) != 1 or target_patients.Status.value_counts().to_dict() != {
        "Metastatic": 3, "Non-metastatic": 3,
    }:
        raise RuntimeError("T-cell 2-aminoethylphosphonate discovery coverage changed")
    target_stats = target_stats.iloc[0]

    representatives = pd.read_csv(representatives_path)
    representative = representatives.loc[
        representatives.scKNIFE_metabolite.eq(TIER3_2AEP_METABOLITE)
    ]
    if len(representative) != 1 or representative.Feature.iloc[0] != TIER3_2AEP_FEATURE:
        raise RuntimeError("2-aminoethylphosphonate LC-MS representative changed")
    representative = representative.iloc[0]

    lcms_matrix = pd.read_csv(lcms_matrix_path)
    intensity = lcms_matrix.loc[lcms_matrix.Feature.eq(TIER3_2AEP_FEATURE)]
    if len(intensity) != 1:
        raise RuntimeError("2-aminoethylphosphonate LC-MS feature row changed")
    intensity = intensity.iloc[0]
    lcms_values = pd.read_csv(lcms_clinical_path)[
        ["SpecimenID", "PatientID", "Timepoint", "Status"]
    ].copy()
    lcms_values["log2_intensity"] = lcms_values.SpecimenID.map(intensity.to_dict())
    if not lcms_values.log2_intensity.notna().all():
        raise RuntimeError("2-aminoethylphosphonate LC-MS coverage changed")
    lcms_patients = (
        lcms_values.groupby(["PatientID", "Status"], as_index=False).log2_intensity.mean()
    )
    if lcms_patients.Status.value_counts().to_dict() != {
        "Metastatic": 4, "Non-metastatic": 3,
    }:
        raise RuntimeError("2-aminoethylphosphonate LC-MS patient groups changed")
    lcms_stats = pd.read_csv(lcms_stats_path)
    lcms_stat = lcms_stats.loc[lcms_stats.Feature.eq(TIER3_2AEP_FEATURE)]
    if len(lcms_stat) != 1:
        raise RuntimeError("2-aminoethylphosphonate LC-MS status statistic changed")
    lcms_stat = lcms_stat.iloc[0]

    patient_values = pd.concat([
        target_patients[["PatientID", "Status", "activity"]]
        .rename(columns={"activity": "value"})
        .assign(layer="scKNIFE | T cells", value_definition="Mean of Bx/Surgery patient-median activity"),
        lcms_patients[["PatientID", "Status", "log2_intensity"]]
        .rename(columns={"log2_intensity": "value"})
        .assign(layer="LC-MS | negative mode", value_definition="Mean Bx/Surgery specimen log2 intensity"),
    ], ignore_index=True)
    statistics = pd.DataFrame([
        {
            "layer": "scKNIFE | T cells", "effect": float(target_stats.effect),
            "effect_name": "Metastatic-minus-non-metastatic mean activity",
            "cliffs_delta": float(target_stats.cliffs_delta),
            "p_value": float(target_stats.exact_p), "fdr": float(target_stats.BH_FDR_70),
            "fdr_family": "BH across 70 Tier-3 metabolite-cell-type pairs",
            "n_nonmetastatic": 3, "n_metastatic": 3,
        },
        {
            "layer": "LC-MS | negative mode", "effect": float(lcms_stat.log2_fold_change),
            "effect_name": "Model log2FC, Metastatic versus Non-metastatic",
            "cliffs_delta": np.nan, "p_value": float(lcms_stat.nominal_p_value),
            "fdr": float(lcms_stat.joint_BH_FDR),
            "fdr_family": "Joint LC-MS feature BH family",
            "n_nonmetastatic": 3, "n_metastatic": 4,
        },
    ])
    annotation = pd.DataFrame([{
        "scKNIFE_metabolite": TIER3_2AEP_METABOLITE,
        "display_name": representative.display_name,
        "Feature": TIER3_2AEP_FEATURE,
        "Polarity": representative.Polarity,
        "mz": float(lcms_stat.mz), "rt_seconds": float(lcms_stat.rt),
        "compound": representative.compound, "formula": representative.formula,
        "inchikey": representative.inchikey,
        "native_score": float(representative.native_score),
        "identity_confidence": representative.identity_confidence,
        "confidence_label": representative.confidence_label,
        "specimens_observed_before_imputation": int(lcms_stat.specimens_observed_before_imputation),
        "specimens_imputed": int(lcms_stat.specimens_imputed),
    }])

    expected = statistics.set_index("layer")
    if not np.isclose(expected.loc["scKNIFE | T cells", "effect"], 0.029879, atol=5e-7):
        raise RuntimeError("2-aminoethylphosphonate scKNIFE effect changed")
    if not np.isclose(expected.loc["scKNIFE | T cells", "p_value"], .9):
        raise RuntimeError("2-aminoethylphosphonate scKNIFE p value changed")
    if not np.isclose(expected.loc["scKNIFE | T cells", "fdr"], 1.0):
        raise RuntimeError("2-aminoethylphosphonate scKNIFE q70 changed")
    if not np.isclose(expected.loc["LC-MS | negative mode", "effect"], 1.025204, atol=5e-7):
        raise RuntimeError("2-aminoethylphosphonate LC-MS log2FC changed")
    if int(annotation.specimens_imputed.iloc[0]) != 0 or not np.isclose(annotation.native_score.iloc[0], 32.83):
        raise RuntimeError("2-aminoethylphosphonate annotation audit changed")

    screen.to_csv(destination / "tier3_status_screen_70.csv", index=False)
    patient_values.to_csv(destination / "tier3_2aep_status_patient_values.csv", index=False)
    statistics.to_csv(destination / "tier3_2aep_status_statistics.csv", index=False)
    annotation.to_csv(destination / "tier3_2aep_annotation_evidence.csv", index=False)


def stage_data(directory, names, include_two_node=False, include_2aep=False):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(SOURCE_DATA / name, directory / name)
    derive_statistics(directory)
    if include_two_node:
        derive_two_node_tier3_evidence(directory)
    if include_2aep:
        derive_2aep_status_evidence(directory)
    manifest = []
    manifest_files = [
        path for path in directory.glob("*")
        if path.name not in {"manifest.json", "README.md"}
    ]
    for path in sorted(manifest_files):
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
        "_draw_status_pathway_scatter", "_draw_four_set_upset",
        "_validated_entity_concordance", "_draw_entity_concordance", "_validated_robust_context",
        "_draw_pathway_context_pie", "_draw_filtering_cascade", "_draw_homocysteine_screen",
        "_draw_macrophage_tasks", "_draw_selenium_lcms", "_draw_ng8_graph", "_draw_review_s1",
        "_draw_su3_tasks",
    }
    header_end = next(node.lineno for node in tree.body if isinstance(node, ast.FunctionDef))
    chunks = ["\n".join(source.splitlines()[:header_end - 1])]
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.lineno >= header_end:
            chunks.append(ast.get_source_segment(source, node))
        elif isinstance(node, ast.FunctionDef) and node.name in keep_functions:
            chunks.append(ast.get_source_segment(source, node))
    module = "\n\n".join(chunks)
    module = module.replace("from figure_6_pooled_concordance import draw_pooled_concordance as _draw_pooled_concordance\n", "")
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

TIER3_TWO_NODE_ORDER = ["2E_heptenoyl_CoA", "Glutaryl_Carnitine"]
TIER3_SITE_COLORS = {"Bx":"#377EB8", "Surgery":"#E41A1C", "Recurrence":"#984EA3"}
TIER3_STATUS_COLORS = {"Non-metastatic":"#4DAF4A", "Metastatic":"#E41A1C"}
TIER3_STANDALONE_WIDTH_MM = 270
TIER3_STANDALONE_HEIGHT_MM = 90
TIER3_BOX_WIDTH = .34
TIER3_POINT_SIZE = 24
TIER3_PAIR_LINE_WIDTH = .8
TIER3_ANNOTATION_SIZE = 4.8
TIER3_GROUP_TITLE_SIZE = 7


def _draw_two_node_evidence(container, data, label="D"):
    patients = data["tier3_two_node_patient_values.csv"].copy()
    p3_cells = data["tier3_two_node_p3_cells.csv.gz"].copy()
    stats = (data["tier3_two_node_statistics.csv"].copy()
             .sort_values("display_order").set_index("metabolite").loc[TIER3_TWO_NODE_ORDER]
             .reset_index())
    if len(stats) != 2 or set(stats.celltype) != {"CAFs"}:
        raise RuntimeError("Panel D two-node specification changed")
    if len(patients) != 28 or patients.metabolite.nunique() != 2:
        raise RuntimeError("Panel D patient-median table changed")
    discovery = patients.loc[patients.is_discovery.astype(bool)].copy()
    if not discovery.groupby("metabolite").PatientID.nunique().eq(6).all():
        raise RuntimeError("Panel D discovery coverage changed")
    if len(p3_cells) != 3892 or not p3_cells.groupby("metabolite").Timepoint.nunique().eq(3).all():
        raise RuntimeError("Panel D P3 cell table changed")
    expected = stats.set_index("metabolite")
    if not np.allclose(expected.treatment_BH_FDR_4, .125):
        raise RuntimeError("Panel D treatment q4 changed")
    if not np.allclose(expected.status_BH_FDR_4, .8):
        raise RuntimeError("Panel D status q4 changed")

    axes = np.atleast_1d(container.subplots(1, 6)).ravel()
    time_order = ["Bx", "Surgery", "Recurrence"]
    status_order = ["Non-metastatic", "Metastatic"]

    for index, spec in enumerate(stats.itertuples(index=False)):
        ax = axes[index]
        frame = discovery.loc[discovery.metabolite.eq(spec.metabolite)]
        wide = frame.pivot(index="PatientID", columns="Timepoint", values="activity").dropna().sort_index()
        values = [wide.Bx.to_numpy(), wide.Surgery.to_numpy()]
        boxes = ax.boxplot(values, positions=[0, 1], widths=TIER3_BOX_WIDTH,
            patch_artist=True, showfliers=False, manage_ticks=False,
            medianprops={"color":COLORS["dark"], "linewidth":1.2},
            boxprops={"edgecolor":COLORS["gray"], "linewidth":.8},
            whiskerprops={"color":COLORS["gray"], "linewidth":.7},
            capprops={"color":COLORS["gray"], "linewidth":.7})
        for patch, color in zip(boxes["boxes"], [TIER3_SITE_COLORS["Bx"], TIER3_SITE_COLORS["Surgery"]]):
            patch.set_facecolor(color); patch.set_alpha(.22)
        offsets = np.linspace(-.05, .05, len(wide))
        for offset, (_, row) in zip(offsets, wide.iterrows()):
            ax.plot([offset, 1 + offset], [row.Bx, row.Surgery], color=COLORS["gray"],
                    lw=TIER3_PAIR_LINE_WIDTH, alpha=.78, zorder=2)
            ax.scatter([offset, 1 + offset], [row.Bx, row.Surgery], s=TIER3_POINT_SIZE,
                       c=[TIER3_SITE_COLORS["Bx"], TIER3_SITE_COLORS["Surgery"]],
                       edgecolor="white", linewidth=.45, zorder=3)
        ax.text(.03, .04,
            f"{int(spec.treatment_n_decrease)}/6↓; Δmean={spec.treatment_effect:+.3f}\\n"
            f"p={spec.treatment_exact_p:.4f}; q₄={spec.treatment_BH_FDR_4:.3f}",
            transform=ax.transAxes, fontsize=TIER3_ANNOTATION_SIZE, va="bottom",
            bbox={"boxstyle":"round,pad=.24", "facecolor":"white",
                  "edgecolor":"#BDBDBD", "alpha":.94})
        ax.set_xticks([0, 1], ["Bx", "Surgery"])
        ax.set_ylabel("Patient-median\\ninferred activity" if index == 0 else "")
        ax.set_title(f"{spec.display_name}\\nCAFs", loc="left", fontsize=6.2,
                     fontweight="bold", pad=2)
        _clean_axis(ax)

    def fmt_q(value):
        return f"{value:.3f}" if value >= .001 else f"{value:.1e}"

    for offset, spec in enumerate(stats.itertuples(index=False), start=2):
        ax = axes[offset]
        frame = p3_cells.loc[p3_cells.metabolite.eq(spec.metabolite)]
        arrays = [frame.loc[frame.Timepoint.eq(site), "activity"].to_numpy(dtype=float)
                  for site in time_order]
        violins = ax.violinplot(arrays, positions=np.arange(3), widths=.72,
            showmeans=False, showmedians=False, showextrema=False)
        for body, site in zip(violins["bodies"], time_order):
            body.set_facecolor(TIER3_SITE_COLORS[site]); body.set_edgecolor(TIER3_SITE_COLORS[site]); body.set_alpha(.26)
        boxes = ax.boxplot(arrays, positions=np.arange(3), widths=.18,
            patch_artist=True, showfliers=False, manage_ticks=False,
            medianprops={"color":COLORS["dark"], "linewidth":1.1},
            boxprops={"edgecolor":COLORS["dark"], "linewidth":.7},
            whiskerprops={"color":COLORS["dark"], "linewidth":.65},
            capprops={"color":COLORS["dark"], "linewidth":.65})
        for patch in boxes["boxes"]:
            patch.set_facecolor("white"); patch.set_alpha(.82)
        note = (f"S−Bx δ={spec.p4_tx_rank_biserial_effect:+.3f}; q={fmt_q(spec.p4_tx_global_contrast_BH_FDR)}\\n"
                f"R−S δ={spec.p4_rebound_rank_biserial_effect:+.3f}; q={fmt_q(spec.p4_rebound_global_contrast_BH_FDR)}\\n"
                f"R−Bx δ={spec.p4_baseline_rank_biserial_effect:+.3f}; q={fmt_q(spec.p4_baseline_global_contrast_BH_FDR)}")
        ax.text(.98, .98, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=TIER3_ANNOTATION_SIZE,
                bbox={"facecolor":"white", "edgecolor":"none", "alpha":.82, "pad":1.0})
        counts = [len(values) for values in arrays]
        short_sites = {"Bx":"Bx", "Surgery":"Surg.", "Recurrence":"Rec."}
        ax.set_xticks(range(3), [f"{short_sites[site]}\\nn={count:,}" for site, count in zip(time_order, counts)])
        ax.tick_params(axis="x", labelsize=5.5)
        ax.set_ylabel("Cell-level\\ninferred activity" if offset == 2 else "")
        ax.set_title(f"{spec.display_name}\\nCAFs", loc="left", fontsize=6.2,
                     fontweight="bold", pad=2)
        _clean_axis(ax)

    for offset, spec in enumerate(stats.itertuples(index=False), start=4):
        ax = axes[offset]
        frame = (discovery.loc[discovery.metabolite.eq(spec.metabolite),
                               ["PatientID", "Status", "status_activity"]]
                 .drop_duplicates("PatientID"))
        arrays = [frame.loc[frame.Status.eq(status), "status_activity"].to_numpy(dtype=float)
                  for status in status_order]
        boxes = ax.boxplot(arrays, positions=[0, 1], widths=TIER3_BOX_WIDTH,
            patch_artist=True, showfliers=False, manage_ticks=False,
            medianprops={"color":COLORS["dark"], "linewidth":1.2},
            boxprops={"edgecolor":COLORS["gray"], "linewidth":.8},
            whiskerprops={"color":COLORS["gray"], "linewidth":.7},
            capprops={"color":COLORS["gray"], "linewidth":.7})
        for patch, status in zip(boxes["boxes"], status_order):
            patch.set_facecolor(TIER3_STATUS_COLORS[status]); patch.set_alpha(.22)
        for xpos, (status, values) in enumerate(zip(status_order, arrays)):
            jitter = np.linspace(-.045, .045, len(values))
            ax.scatter(np.full(len(values), xpos) + jitter, values, s=TIER3_POINT_SIZE,
                       color=TIER3_STATUS_COLORS[status], edgecolor="white",
                       linewidth=.45, zorder=3)
        ax.text(.03, .04,
            f"M−NM={spec.status_effect:+.3f}; δ={spec.status_cliffs_delta:+.3f}\\n"
            f"p={spec.status_exact_p:.3f}; q₄={spec.status_BH_FDR_4:.3f}",
            transform=ax.transAxes, fontsize=TIER3_ANNOTATION_SIZE, va="bottom",
            bbox={"boxstyle":"round,pad=.24", "facecolor":"white",
                  "edgecolor":"#BDBDBD", "alpha":.94})
        ax.set_xticks([0, 1], ["Non-meta\\nn=3", "Meta\\nn=3"])
        ax.set_ylabel("Patient status score" if offset == 4 else "")
        ax.set_title(f"{spec.display_name}\\nCAFs", loc="left", fontsize=6.2,
                     fontweight="bold", pad=2)
        _clean_axis(ax)

    container.text(.17, .805, "Six-patient discovery: Bx → Surgery",
                   ha="center", va="top", fontsize=TIER3_GROUP_TITLE_SIZE, fontweight="bold")
    container.text(.50, .805, "P3 longitudinal trajectory",
                   ha="center", va="top", fontsize=TIER3_GROUP_TITLE_SIZE, fontweight="bold")
    container.text(.83, .805, "Discovery-cohort metastatic status",
                   ha="center", va="top", fontsize=TIER3_GROUP_TITLE_SIZE, fontweight="bold")
    _panel_title(container, label, "Two CAF Tier-3 inferred-metabolite programs",
                 "Treatment decrease, single-patient recurrence rebound and status context.")
    container.text(.5, .035,
        "Treatment: exact two-sided sign-flip tests; status: exact two-sided 3-vs-3 label permutation; BH across the original four-node family.",
        ha="center", fontsize=5.2, color=COLORS["gray"])
    container.text(.5, .012,
        "P3 cell distributions are descriptive; cells are not independent patients. Inferred activity is neither abundance nor flux.",
        ha="center", fontsize=5.2, color=COLORS["gray"])
    container.subplots_adjust(left=.060, right=.995, top=.67, bottom=.24, wspace=.58)


def _sulfur_treatment_statistics(data):
    stats=data["sulfur_cohort_treatment_statistics.csv"].copy()
    if len(stats)!=90: raise RuntimeError("Frozen sulfur treatment statistics changed")
    return stats

S2_STATUS_ORDER = ["Non-metastatic", "Metastatic"]
S2_STATUS_COLORS = {"Non-metastatic":"#4DAF4A", "Metastatic":"#E41A1C"}
S2_STANDALONE_WIDTH_MM = 180
S2_STANDALONE_HEIGHT_MM = 90
S2_BOX_WIDTH = .34
S2_POINT_SIZE = 30
S2_ANNOTATION_SIZE = 6.4
S2_AXIS_TITLE_SIZE = 8
S2_TICK_SIZE = 7


def _draw_2aep_status(container, data, label="S2"):
    patients = data["tier3_2aep_status_patient_values.csv"].copy()
    stats = data["tier3_2aep_status_statistics.csv"].copy().set_index("layer")
    evidence = data["tier3_2aep_annotation_evidence.csv"].copy()
    layers = ["scKNIFE | T cells", "LC-MS | negative mode"]
    if len(patients) != 13 or set(patients.layer) != set(layers):
        raise RuntimeError("S2 patient-level evidence changed")
    if len(stats) != 2 or len(evidence) != 1:
        raise RuntimeError("S2 frozen statistics or annotation evidence changed")
    if patients.loc[patients.layer.eq(layers[0]), "Status"].value_counts().to_dict() != {
        "Non-metastatic": 3, "Metastatic": 3,
    }:
        raise RuntimeError("S2 scKNIFE group sizes changed")
    if patients.loc[patients.layer.eq(layers[1]), "Status"].value_counts().to_dict() != {
        "Metastatic": 4, "Non-metastatic": 3,
    }:
        raise RuntimeError("S2 LC-MS group sizes changed")

    axes = np.atleast_1d(container.subplots(1, 2)).ravel()
    titles = ["S2a  scKNIFE | T cells", "S2b  LC-MS | negative mode"]
    ylabels = ["Patient-mean inferred activity", "Patient-mean log2 intensity"]
    for index, (ax, layer, title, ylabel) in enumerate(zip(axes, layers, titles, ylabels)):
        frame = patients.loc[patients.layer.eq(layer)]
        arrays = [
            frame.loc[frame.Status.eq(status)].sort_values("PatientID").value.to_numpy(dtype=float)
            for status in S2_STATUS_ORDER
        ]
        boxes = ax.boxplot(
            arrays, positions=[0, 1], widths=S2_BOX_WIDTH, patch_artist=True,
            showfliers=False, manage_ticks=False,
            medianprops={"color":COLORS["dark"], "linewidth":1.3},
            boxprops={"edgecolor":COLORS["gray"], "linewidth":.8},
            whiskerprops={"color":COLORS["gray"], "linewidth":.7},
            capprops={"color":COLORS["gray"], "linewidth":.7},
        )
        for patch, status in zip(boxes["boxes"], S2_STATUS_ORDER):
            patch.set_facecolor(S2_STATUS_COLORS[status]); patch.set_alpha(.22)
        for xpos, (status, values) in enumerate(zip(S2_STATUS_ORDER, arrays)):
            jitter = np.linspace(-.05, .05, len(values))
            ax.scatter(
                np.full(len(values), xpos) + jitter, values, s=S2_POINT_SIZE,
                color=S2_STATUS_COLORS[status], edgecolor="white", linewidth=.5, zorder=3,
            )
        stat = stats.loc[layer]
        if index == 0:
            annotation = (
                f"M−NM={stat.effect:+.3f}; Cliff's δ={stat.cliffs_delta:+.3f}\\n"
                f"exact p={stat.p_value:.3f}; BH q₇₀={stat.fdr:.3f}"
            )
        else:
            annotation = (
                f"model log2FC={stat.effect:+.3f}\\n"
                f"nominal p={stat.p_value:.3f}; joint BH q={stat.fdr:.3f}"
            )
        ax.text(
            .97, .95, annotation, transform=ax.transAxes, ha="right", va="top",
            fontsize=S2_ANNOTATION_SIZE,
            bbox={"boxstyle":"round,pad=.28", "facecolor":"white",
                  "edgecolor":"#BDBDBD", "alpha":.94},
        )
        ax.set_xticks(
            [0, 1],
            [f"Non-metastatic\\nn={len(arrays[0])}", f"Metastatic\\nn={len(arrays[1])}"],
        )
        ax.set_ylabel(ylabel, fontsize=S2_AXIS_TITLE_SIZE)
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=17)
        ax.text(
            0, 1.01, "One biopsy/surgery-average value per patient",
            transform=ax.transAxes, fontsize=S2_TICK_SIZE, color=COLORS["gray"],
            fontstyle="italic", va="bottom",
        )
        ax.tick_params(labelsize=S2_TICK_SIZE)
        _clean_axis(ax)

    audit = evidence.iloc[0]
    _panel_title(
        container, label, "2-Aminoethylphosphonate status candidate",
        "Directionally concordant but statistically unsupported across scKNIFE and LC-MS.",
    )
    container.text(
        .5, .045,
        f"LC-MS {audit.Feature} • m/z {audit.mz:.4f} • RT {audit.rt_seconds:.2f} s • native-library score {audit.native_score:.2f} • 0 imputed specimens.",
        ha="center", fontsize=6.2, color=COLORS["gray"],
    )
    container.text(
        .5, .018,
        "Putative native MS/MS library match (MSI level-2 candidate pending mirror/manual review); not metabolite or biomarker validation.",
        ha="center", fontsize=6.2, color=COLORS["gray"],
    )
    container.subplots_adjust(left=.075, right=.985, top=.67, bottom=.25, wspace=.36)


MAIN_INPUTS={
"A":["treatment_four_set_membership.csv"],"B":["surgery_pathway_scatter.csv","surgery_pathway_enrichment.csv","treatment_four_set_membership.csv"],"C":['meia_vs_scknife_pooled_correlations.csv', 'meia_vs_scknife_pooled_density.csv.gz'],
"D":["tier3_two_node_patient_values.csv","tier3_two_node_p3_cells.csv.gz","tier3_two_node_statistics.csv"],
"E":["robust_metabolite_pathway_assignments.csv","robust_metabolite_pathway_context_counts.csv"],
"F":["sulfur_cohort_patient_values.csv","sulfur_cohort_treatment_statistics.csv"],
"G":["sulfur_lcms_feature_values.csv","sulfur_lcms_feature_evidence.csv"],
"H":["sulfur_ng8_nodes.csv","sulfur_ng8_edges.csv","sulfur_ng8_anchors.csv"]}
SUPPLEMENT_INPUTS={"SA":["status_four_set_membership.csv"],"SB":["status_pathway_scatter.csv","status_pathway_enrichment.csv","status_four_set_membership.csv"],"SJ":["gpr_gpa_vs_scknife_entity_correlations.csv"],
"S1":["review_whole_profile_patient_summary.csv","review_whole_profile_concordance_summary.csv"],"SFUNNEL":["robust_metabolite_filtering_funnel.csv"],
"SDIRECTION":["sulfur_cohort_patient_values.csv","sulfur_cohort_treatment_statistics.csv"],"SU3":["sulfur_p3_all_score_cells.csv.gz","sulfur_p3_task_statistics.csv"],
"S2":["tier3_2aep_status_patient_values.csv","tier3_2aep_status_statistics.csv","tier3_2aep_annotation_evidence.csv"]}

def load_tables(data_root, inputs):
    root=Path(data_root);return {name:pd.read_csv(root/name) for name in sorted(set(sum(inputs.values(),[])))}

def make_figure6_main_panel(label,data):
    fig=plt.figure(figsize=((_mm(TIER3_STANDALONE_WIDTH_MM),_mm(TIER3_STANDALONE_HEIGHT_MM)) if label=="D" else ((14.5,8.8) if label=="H" else (_mm(112),_mm(88)))),facecolor="white")
    {"A":lambda:_draw_four_set_upset(fig,data,"A","treatment_four_set_membership.csv","Treatment four-set metabolic overlap"),"B":lambda:_draw_treatment_pathway_scatter(fig,data,"B"),"C":lambda:_draw_pooled_concordance(fig,data,"C"),"D":lambda:_draw_two_node_evidence(fig,data,"D"),"E":lambda:_draw_pathway_context_pie(fig,data,"E"),"F":lambda:_draw_macrophage_tasks(fig,data,"F"),"G":lambda:_draw_selenium_lcms(fig,data,"G"),"H":lambda:_draw_ng8_graph(fig,data,"H")}[label]()
    return fig

def make_figure6_supplementary_panel(label,data):
    fig=plt.figure(figsize=((_mm(S2_STANDALONE_WIDTH_MM),_mm(S2_STANDALONE_HEIGHT_MM)) if label=="S2" else (_mm(150),_mm(96))),facecolor="white")
    {"SA":lambda:_draw_four_set_upset(fig,data,"SA","status_four_set_membership.csv","Metastatic-status four-set metabolic overlap"),"SB":lambda:_draw_status_pathway_scatter(fig,data,"SB"),"SJ":lambda:_draw_entity_concordance(fig,data,"SJ","gpa"),"S1":lambda:_draw_review_s1(fig,data,"S1"),"SFUNNEL":lambda:_draw_filtering_cascade(fig,data,"SFUNNEL"),"SDIRECTION":lambda:_draw_homocysteine_screen(fig,data,"SDIRECTION"),"SU3":lambda:_draw_su3_tasks(fig,data,"SU3"),"S2":lambda:_draw_2aep_status(fig,data,"S2")}[label]()
    return fig

def make_figure6_compiled(data):
    fig=plt.figure(figsize=(_mm(297),_mm(380)),facecolor="white");rows=fig.subfigures(4,1,height_ratios=[.86,.98,.90,1.45],hspace=.035);top=rows[0].subfigures(1,3,wspace=.045)
    _draw_four_set_upset(top[0],data,"A","treatment_four_set_membership.csv","Treatment four-set metabolic overlap");_draw_treatment_pathway_scatter(top[1],data,"B");_draw_pooled_concordance(top[2],data,"C");_draw_two_node_evidence(rows[1],data,"D")
    evidence=rows[2].subfigures(1,3,width_ratios=[.95,1.62,.93],wspace=.045);_draw_pathway_context_pie(evidence[0],data,"E");_draw_macrophage_tasks(evidence[1],data,"F");_draw_selenium_lcms(evidence[2],data,"G");_draw_ng8_graph(rows[3],data,"H");return fig

def make_figure6_supplementary(data):
    fig=plt.figure(figsize=(_mm(297),_mm(570)),facecolor="white");rows=fig.subfigures(5,1,height_ratios=[1.02,.83,.90,1.82,.85],hspace=.035);top=rows[0].subfigures(1,3,wspace=.045)
    _draw_four_set_upset(top[0],data,"SA","status_four_set_membership.csv","Metastatic-status four-set metabolic overlap");_draw_status_pathway_scatter(top[1],data,"SB");_draw_entity_concordance(top[2],data,"SJ","gpa");_draw_review_s1(rows[1],data,"S1")
    middle=rows[2].subfigures(1,2,width_ratios=[.44,.56],wspace=.055);_draw_filtering_cascade(middle[0],data,"SFUNNEL");_draw_homocysteine_screen(middle[1],data,"SDIRECTION");_draw_su3_tasks(rows[3],data,"SU3");_draw_2aep_status(rows[4],data,"S2");return fig

def save_figure(fig, stem, dpi=300):
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ("png","pdf","svg"):fig.savefig(stem.with_suffix("."+suffix),dpi=dpi,bbox_inches=None,pad_inches=0)
'''
    if any(term in module for term in ("pearsonr(", "spearmanr(", "multipletests(", "ttest_ind(", "wilcoxon(")):
        raise RuntimeError("Public plotting module still contains inferential calls")
    module += "\n\ndef _draw_pooled_concordance(container,data,label=\"C\"):\n    from pooled_concordance import draw_pooled_concordance\n    return draw_pooled_concordance(container,data,label)\n"
    return module


def notebook(title, supplementary=False):
    labels = list(("SA", "SB", "SJ", "S1", "SFUNNEL", "SDIRECTION", "SU3", "S2") if supplementary else tuple("ABCDEFGH"))
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
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":f"fig={compiled_fun}(DATA)\nsave_figure(fig,OUTPUT_ROOT/'{compiled_name}')\ncompiled_png=OUTPUT_ROOT/'{compiled_name}.png'\npreview_png=OUTPUT_ROOT/'{compiled_name}_preview.png'\npreview_png.write_bytes(compiled_png.read_bytes())\nassert preview_png.read_bytes()==compiled_png.read_bytes()\nplt.show();plt.close(fig)"},
    ]
    return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.9"}},"nbformat":4,"nbformat_minor":5}


def main():
    shutil.copy2(SOURCE_NOTEBOOK.parent / "figure_6_pooled_concordance.py", FIG6 / "pooled_concordance.py")
    stage_data(FIG6 / "data", MAIN_FILES, include_two_node=True)
    stage_data(S6 / "data", SUPPLEMENT_FILES, include_2aep=True)
    plotting = build_plotting_module()
    (FIG6 / "plotting.py").write_text(plotting)
    (S6 / "plotting.py").write_text(plotting)
    (FIG6 / "Figure_6.ipynb").write_text(json.dumps(notebook("Figure 6 — blank-corrected metabolic evidence story"), indent=1) + "\n")
    (S6 / "Figure_S6.ipynb").write_text(json.dumps(notebook("Supplementary Figure S6 — robustness and longitudinal evidence", True), indent=1) + "\n")


if __name__ == "__main__":
    main()
