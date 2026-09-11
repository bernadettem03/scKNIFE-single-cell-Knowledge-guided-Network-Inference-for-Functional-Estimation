# Figure 3 — External biological benchmarking

Figure 3 version 2 is canonical. Patient-level inference and candidate selection are retained under `analysis/`; the root notebook will render frozen results only.

## Contents

- `Figure_3.ipynb`: public figure entrypoint.
- `plotting.py`: reusable rendering functions (created during notebook extraction).
- `analysis/`: reproducible analytical derivations and provenance.
- `data/`: frozen plot-ready inputs.
- `outputs/`: approved PDF and PNG preview.

All paths must remain repository-relative. Large editable outputs remain in the private unpublished archive.

## Standalone panel outputs

The plotting notebook writes each authoritative main panel as a 300-DPI PNG and
a vector PDF under `outputs/panels/`. `Figure_3_panel_export_manifest.json`
records the expected panel order, file sizes, and SHA-256 checksums.
