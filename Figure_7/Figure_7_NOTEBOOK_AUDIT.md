# Figure 7 code-generation audit

The active plot-only notebooks call the same `render_current.py` renderer. All 15 panels are constructed from saved tables. The 69,319 UMAP coordinates are read without recomputation; the Figure 4 source contains 69,391 cells, with the 72-cell signaling-table exclusion documented in `data/known_new/PDGF_umap_manifest.json`.

The renderer asserts overlap counts 44/32/33 and 194/225/260, 1,917 shared positive score records, the 8×8 and 29×29 PDGF matrices, exact 4×3 zoom equality, the rank-one Endothelial RGS5→PVL Immature connection, seven eligible patient pairs and the 2,080-test screen. It checks selected P=0.109375 and BH q=0.6318347509. The six violin densities use 1.5-bandwidth nonnegative tapered support; the seven plotted dots per stage retain the observed values. Outputs and hashes are recorded in `outputs/Figure_7_current_export_manifest.json`; no frozen artwork is used.
