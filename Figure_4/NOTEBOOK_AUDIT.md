# Figure 4 notebook audit

## Public plotting contract

`Figure_4.ipynb` is a compact plotting-only entrypoint. It imports reusable renderers from `plotting.py`, validates the frozen Figure 4 bundle, renders panels B–F independently, and compiles the approved main-page layout.

## Analysis boundary

UMAP construction, H clustering, CNV inference, differential expression, marker selection, and TNBCtype assignment are never run by the public notebook or plotting module. Their provenance remains under `analysis/`; the root workflow consumes only the validated frozen bundle.

## Data and distribution

The large `fig4_plot_data.h5` bundle remains private and unpublished, is restored with `download_data.py`, and is explicitly Git-ignored. Public mappings use P1–P7 display identifiers. Supplementary Figure 4 remains under `Supplementary/Figure_S4/`.

## Acceptance boundary

Panels B–F and the compiled figure must render inline, export as PNG/PDF, retain the approved mixed raster/vector appearance, contain no absolute paths, and pass manifest/checksum validation without tracking the private bundle.
