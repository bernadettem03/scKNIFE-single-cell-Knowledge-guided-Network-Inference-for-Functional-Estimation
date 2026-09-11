# Figure 6 — Blank-corrected metabolomics and inferred activity

The blank-corrected workflow is canonical. Main and supplementary notebooks render precomputed enrichment, cross-modal, trajectory, and metabolite tables.

## Contents

- `Figure_6.ipynb`: public figure entrypoint.
- `plotting.py`: reusable rendering functions (created during notebook extraction).
- `analysis/`: reproducible analytical derivations and provenance.
- `data/`: frozen plot-ready inputs.
- `outputs/`: approved PDF and PNG preview.

All paths must remain repository-relative. Large editable outputs remain in the private unpublished archive.

## Standalone panel outputs

The plotting notebook writes each authoritative main panel as a 300-DPI PNG and
a vector PDF under `outputs/panels/`. `Figure_6_panel_export_manifest.json`
records the expected panel order, file sizes, and SHA-256 checksums.
