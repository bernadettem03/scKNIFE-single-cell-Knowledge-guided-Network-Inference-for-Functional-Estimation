# Figure 4 — H-derived embedding and Wu SingleR annotation

Submission legend: [LEGEND.txt](LEGEND.txt).

The public workflow renders checked, frozen inputs. It does not recompute embeddings, clustering, SingleR predictions, CNV or TNBCtype assignments. `figure4_editable.py` validates staged inputs and renders six final panels; `Figure_4.ipynb` is the executed entrypoint.

## Contents

- `Figure_4.ipynb`: public figure entrypoint.
- `figure4_editable.py`: final B–G mixed raster/vector renderer. Earlier rendering scripts are retained as provenance.
- `analysis/`: reproducible analytical derivations and provenance.
- `data/singler/`: checked cell-aligned plot arrays, deidentified metadata and SHA manifest; standalone C/D source plots are included.
- `outputs/`: approved PNG preview and editable-text PDF/SVG.

All paths must remain repository-relative. Large editable outputs remain in the private unpublished archive.

## Standalone panel outputs

The notebook writes each panel and the 210 × 390 mm main figure as PNG/PDF/SVG. UMAP scatters and G's 69,391-cell score matrix/annotation strips are rasterized within the PDF/SVG; all fonts, axes and legends remain editable. B and D plot elements and E's violin are vector. Panel C uses full-atlas SingleR major calls assigned per cell after default pruning (0.6 pt² points; alpha 0.95), with Cancer Epithelial and Normal Epithelial collapsed to one broad Epithelial display label. D shows the same eight cell-level broad groups across 40 Wu-reference marker positions, and G shows corresponding broad relative scores. F shows six diagnostic lineage markers on the same H-derived coordinates. D and F are descriptive support, not independent validation of every cell-level call. The staged subtype matrix is explicitly aligned to P1–P7 before drawing B, correcting the former file-order/display-order mismatch. `Figure_4_panel_export_manifest.json` records version and hashes. SingleR relative scores are not probabilities.
