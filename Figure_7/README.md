# Figure 7 — Cell–cell signaling concordance and VEGF biology

The workflow renders frozen signaling, concordance, VEGF, and H-factor inputs without rerunning CellChat or scKNIFE.

## Contents

- `Figure_7.ipynb`: public figure entrypoint.
- `plotting.py`: reusable rendering functions (created during notebook extraction).
- `analysis/`: reproducible analytical derivations and provenance.
- `data/`: frozen plot-ready inputs.
- `outputs/`: approved PDF and PNG preview.

All paths must remain repository-relative. Large editable outputs remain in the private unpublished archive.

## Standalone panel outputs

The plotting notebook writes each authoritative main panel as a 300-DPI PNG and
a vector PDF under `outputs/panels/`. `Figure_7_panel_export_manifest.json`
records the expected panel order, file sizes, and SHA-256 checksums.
