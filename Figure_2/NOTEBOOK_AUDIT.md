# Figure 2 notebook audit

Source: `source_snapshot/Figures/submission/Figure_2.ipynb`

The source notebook is scientifically organized but is not yet suitable as the
public plotting-only notebook. It combines main and supplementary figures,
repeats standalone-panel export code, and calculates several annotations at
render time.

## Retain in the main plotting workflow

- Execution contract, manifest validation, style constants, and checksums.
- Loading of frozen runtime, capacity, agreement, UMAP, and Tahoe tables.
- Workflow, bar-chart, heatmap, UMAP, and Hallmark rendering functions.
- Main A4 composition and PDF/PNG export.
- Interpretation and provenance notes.

## Move to `analysis/`

- ARI, AMI, and NMI calculation from frozen cluster and author labels.
- Pearson and Spearman correlation calculation for TP53 and PIK3R1.
- Hyperparameter optimum/selection logic.

The plotting notebook will consume the resulting checked CSV annotations.

## Move to Supplementary Figure S2

- The complete no-match UMAP grid.
- Pooled W-space CKA and factor diagnostics.
- Pooled W-space t-SNE panels.
- Supplementary A4 composition and export.

## Remove from the public notebook

- Eight near-identical standalone A–H export cells.
- Previously executed cell outputs and transient raster assets.
- Duplicate setup/import blocks and development-only diagnostics.
- SVG export from the GitHub workflow; editable SVG remains a Zenodo asset.

## Acceptance boundary

The current copied notebook and approved PDF/PNG files are a checksum-identical
baseline. Refactoring is accepted only if the main and supplementary rendered
pages remain visually equivalent and all displayed numerical annotations match
the frozen source outputs.

