# Figure 4 notebook audit

## Public plotting contract

`Figure_4.ipynb` is a compact plotting-only entrypoint. It imports `figure4_editable.py`, validates frozen inputs under `data/singler/`, renders B–G and compiles the 210 × 390 mm main layout. The prior DE heatmap and six-marker panel are no longer current Figure 4 panels; their analytical inputs remain as provenance.

## Analysis boundary

UMAP construction, H clustering, SingleR assignment, CNV inference, differential expression and TNBCtype assignment are never run by the public notebook. The renderer checks 69,391 matched cells, shared saved H-derived coordinates, staged source hashes, and exact marker names. SingleR relative scores are not probabilities.

## Data and distribution

The large legacy `fig4_plot_data.h5` bundle remains private and unpublished, is restored with `download_data.py`, and is explicitly Git-ignored. Current portable plot inputs are in `data/singler/`; the patient strip legend uses only P1–P7. Supplementary Figure 4 remains under `Supplementary/Figure_S4/`.

## Acceptance boundary

Panels B–G and the compiled figure render inline, export as PNG/PDF/SVG, contain no absolute paths, and pass manifest/checksum validation without the private bundle. Dense UMAP/heatmap data are embedded images; PDF/SVG text, legends, axes, B/D plot elements and E's violin remain editable. The old B matrix-row/P-label mismatch is corrected by explicit patient-ID alignment.
