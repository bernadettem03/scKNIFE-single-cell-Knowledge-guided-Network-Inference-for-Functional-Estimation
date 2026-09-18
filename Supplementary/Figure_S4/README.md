# Supplementary Figure 4 — cell-type annotation and CNV quality control

Submission legend: [LEGEND.txt](LEGEND.txt).

Supplementary Figure 4 is a plotting-only workflow built from frozen, plot-ready arrays. It does not rerun UMAP, cell annotation, CNV inference, CNV clustering, or marker selection.

## Caption

**Supplementary Figure 4 | Cell-type annotation and CNV quality control.** **a,** Cell-type composition by patient and collection site. **b,** Overall cell-type composition. **c,** Canonical marker-expression dot plot. **d,** Cell-type UMAP. **e,** CNV-cluster UMAP. **f,** Cell-type labels on the CNV-QC UMAP. **g,** Provisional CNV-low and CNV-high shortlist.

## Reproducibility layout

- `Figure_S4.ipynb`: compact plotting-only entrypoint with inline panels.
- `plotting.py`: input validation, panel renderers, compilation, exports, and manifest creation.
- `data/fig4_supplementary_plot_data.h5`: frozen arrays for 69,391 cells, 15 specimens, 59 markers, and 26 CNV clusters.
- `analysis/`: retained full-render provenance using the historical T/R/S/Q/N/P/M identifiers; analytical derivations are owned by `../../Figure_4/analysis/`.
- `outputs/panels/`: standalone a–g PNG and PDF files.
- `outputs/Figure_S4.png` and `.pdf`: compiled figure; `Figure_S4_manifest.json` records lineage and output paths.

The current publication mapping is a←T, b←R, c←S, d←Q, e←N, f←P and g←M. The legacy identifiers are retained only in the provenance notebook and manifest mapping.

## Input contract

The HDF5 bundle supplies atlas UMAP coordinates and labels, specimen-level composition counts, canonical-marker summaries, CNV-QC coordinates and clusters, and provisional CNV status. The plotting module validates expected dimensions, category order, and the fixed CNV-low/CNV-high cell counts before rendering. Percentages are deterministic display transformations of frozen counts.

## Run

From this directory, in the declared repository environment:

```bash
jupyter nbconvert --to notebook --execute --inplace Figure_S4.ipynb --ExecutePreprocessor.timeout=900
```

The workflow uses the established Wong-inspired cell-type palette, rasterizes dense UMAP layers inside vector PDFs, and exports PNG previews at 300 DPI.
