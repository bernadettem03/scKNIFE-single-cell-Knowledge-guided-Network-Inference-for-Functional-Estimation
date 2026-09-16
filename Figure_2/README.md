# Figure 2 — benchmarking and external validation

Submission legend: [LEGEND.txt](LEGEND.txt).

This directory reproduces the main Figure 2 from frozen, plot-ready inputs.
The public `Figure_2.ipynb` will be plotting-only. Computation of agreement
metrics and correlation annotations is isolated under `analysis/`.

## Panels

- A: benchmarking workflow.
- B: prior-knowledge capacity.
- C: training runtime.
- D: scKNIFE clustering across four cohorts.
- E: clustering agreement with author labels.
- F: author cell-type labels on the same embeddings.
- G: TP53/Doxorubicin Hallmark comparison.
- H: PIK3R1/Afatinib Hallmark comparison.

## Layout

- `Figure_2.ipynb`: migrated authoritative notebook; it will be reduced to the
  plotting-only main workflow after baseline validation.
- `analysis/`: deterministic derivation of small annotation tables.
- `data/`: frozen inputs required by the main figure.
- `outputs/`: approved PDF and PNG preview.

Pooled W-space diagnostics are maintained under `../Supplementary/Figure_S1/`;
no-match comparator UMAPs are maintained under `../Supplementary/Figure_S2/`.

## Standalone panel outputs

The plotting notebook writes each authoritative main panel as a 300-DPI PNG and
a vector PDF under `outputs/panels/`. `Figure_2_panel_export_manifest.json`
records the expected panel order, file sizes, and SHA-256 checksums.
