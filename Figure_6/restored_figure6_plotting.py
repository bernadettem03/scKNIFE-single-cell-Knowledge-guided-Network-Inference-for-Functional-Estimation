"""Corrected old Figure 6 flow with patient-level LC–MS ending."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import plotting as legacy
import two_angle_plotting as clinical
import patient_bridge_plotting as bridge

OLD_INPUTS = [
    "treatment_four_set_membership.csv", "surgery_pathway_scatter.csv",
    "surgery_pathway_enrichment.csv", "meia_vs_scknife_pooled_correlations.csv",
    "meia_vs_scknife_pooled_density.csv.gz", "robust_tier3_trajectory_selection.csv",
    "robust_tier3_trajectory_values.csv.gz", "robust_metabolite_broad_entity_mapping.tsv",
    "robust_metabolite_broad_entity_counts.csv", "status_pathway_scatter.csv",
    "status_pathway_enrichment.csv", "status_four_set_membership.csv",
]


def load_data(root: Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    old = {
        name: pd.read_csv(root / name, sep="\t" if name.endswith(".tsv") else ",")
        for name in OLD_INPUTS
    }
    new = clinical.load_data(root)
    import hashlib
    import json
    manifest = json.loads((root / "patient_bridge_manifest.json").read_text())
    patient = {}
    for name, evidence in manifest["files"].items():
        payload = (root / name).read_bytes()
        if len(payload) != evidence["bytes"] or hashlib.sha256(payload).hexdigest() != evidence["sha256"]:
            raise RuntimeError(f"Patient bridge checksum mismatch: {name}")
        patient[name] = pd.read_csv(root / name)
    if len(old["surgery_pathway_enrichment.csv"]) != 21:
        raise RuntimeError("Expected corrected 21-pathway treatment package")
    if len(old["status_pathway_enrichment.csv"]) != 33:
        raise RuntimeError("Expected corrected 33-pathway status package")
    if len(old["robust_metabolite_broad_entity_mapping.tsv"]) != 157:
        raise RuntimeError("Reviewed broad-entity map changed")
    return {**old, **new, **patient}


def make_panel(label: str, data: dict[str, pd.DataFrame]):
    if label in "ABC":
        if label in "AB":
            return legacy.make_figure6_main_panel(label, data)
        fig = plt.figure(figsize=(legacy._mm(180), legacy._mm(78)), facecolor="white")
        legacy._draw_pooled_concordance(fig, data, "", plot_letters=("C", "D"))
        return fig
    if label == "D":
        fig = plt.figure(figsize=(legacy._mm(230), legacy._mm(145)), facecolor="white")
        bridge.panel_e(fig, data, label="", plot_letters=("E", "F", "G", "H", "I", "J"))
        return fig
    if label in "EF":
        fig = plt.figure(figsize=(legacy._mm(180), legacy._mm(100)), facecolor="white")
        if label == "E":
            clinical.panel_c(fig, data, label="", plot_letters=("K", "L"))
        else:
            clinical.panel_d(fig, data, label="", plot_letters=("M", "N"))
        return fig
    if label == "G":
        fig = plt.figure(figsize=(legacy._mm(230), legacy._mm(105)), facecolor="white")
        bridge.panel_h(fig, data, "", plot_letters=("O", "P", "Q"))
        return fig
    raise ValueError(label)


def make_compiled(data: dict[str, pd.DataFrame]):
    fig = plt.figure(figsize=(legacy._mm(297), legacy._mm(425)), facecolor="white")
    rows = fig.subfigures(5, 1, height_ratios=[1.00, 1.20, .78, .82, .95], hspace=.07)
    top = rows[0].subfigures(1, 3, width_ratios=[1, 1, 1], wspace=.045)
    legacy._draw_four_set_upset(top[0], data, "A", "treatment_four_set_membership.csv", "Treatment four-set metabolic overlap")
    legacy._draw_treatment_pathway_scatter(top[1], data, "B")
    legacy._draw_pooled_concordance(top[2], data, "", plot_letters=("C", "D"))
    bridge.panel_e(rows[1], data, "", plot_letters=("E", "F", "G", "H", "I", "J"))
    clinical.panel_c(rows[2], data, label="", plot_letters=("K", "L"))
    clinical.panel_d(rows[3], data, label="", plot_letters=("M", "N"))
    bridge.panel_h(rows[4], data, "", plot_letters=("O", "P", "Q"))
    return fig


def export(fig, stem: Path):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix("." + extension), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.04)
