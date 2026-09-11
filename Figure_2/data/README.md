# Figure 2 plot-ready data

Frozen benchmarking and Tahoe plot tables, with validated frozen no-match UMAP rasters for Supplementary Figure 2

## Panel lineage

| Panel | Title | Kind | Extraction mode | Packaged intermediate inputs |
|---|---|---|---|---|
| A | Benchmarking workflow | schematic | schematic | None (schematic) |
| B | Prior knowledge | data | staged_frozen | `knowledge_feature_counts.csv` |
| C | Training runtime | data | staged_frozen | `runtime.csv` |
| D | Clustering | data | staged_frozen | `scknife_bassez_c1_umap.csv.gz`<br>`scknife_bassez_c2_umap.csv.gz`<br>`scknife_shiao_umap.csv.gz`<br>`scknife_wu_tnbc_umap.csv.gz` |
| E | Agreement metrics | data | staged_frozen | `scknife_bassez_c1_umap.csv.gz`<br>`scknife_bassez_c2_umap.csv.gz`<br>`scknife_wu_tnbc_umap.csv.gz`<br>`spectra_bassez_c1_umap.csv.gz`<br>`spectra_bassez_c2_umap.csv.gz`<br>`spectra_wu_tnbc_umap.csv.gz`<br>`expimap_bassez_c1_umap.csv.gz`<br>`expimap_bassez_c2_umap.csv.gz`<br>`expimap_wu_tnbc_umap.csv.gz`<br>`slalom_bassez_c1_umap.csv.gz`<br>`slalom_bassez_c2_umap.csv.gz`<br>`slalom_wu_tnbc_umap.csv.gz`<br>`agreement_metrics_from_frozen_labels.csv` |
| F | Cell type | data | staged_frozen | `scknife_bassez_c1_umap.csv.gz`<br>`scknife_bassez_c2_umap.csv.gz`<br>`scknife_shiao_umap.csv.gz`<br>`scknife_wu_tnbc_umap.csv.gz` |
| G | TP53 | data | staged_frozen | `tahoe_plot_data.h5` (`TP53`, `TP53/expression`, `TP53/bgnmf_score`, `TP53/scanpy_score`) |
| H | PIK3R1 | data | staged_frozen | `tahoe_plot_data.h5` (`PIK3R1`, `PIK3R1/expression`, `PIK3R1/bgnmf_score`, `PIK3R1/scanpy_score`) |

## Auxiliary packaged inputs

- `tuning_sweeps.csv` — benchmark tuning diagnostics
- `tuning_final_parameters.csv` — final benchmark parameter record
- `upstream_provenance.md` — retained upstream provenance notes
- `supplementary_umap_no_match/no_match_umap_manifest.json` — Supplementary Figure 2 frozen-raster provenance, checksums, and colours
- `supplementary_umap_no_match/expimap_bassez_c1_points.png` — Supplementary Figure 2 validated frozen no-match UMAP raster
- `supplementary_umap_no_match/expimap_bassez_c2_points.png` — Supplementary Figure 2 validated frozen no-match UMAP raster
- `supplementary_umap_no_match/expimap_wu_tnbc_points.png` — Supplementary Figure 2 validated frozen no-match UMAP raster
- `supplementary_umap_no_match/slalom_bassez_c1_points.png` — Supplementary Figure 2 validated frozen no-match UMAP raster
- `supplementary_umap_no_match/slalom_bassez_c2_points.png` — Supplementary Figure 2 validated frozen no-match UMAP raster
- `supplementary_umap_no_match/slalom_wu_tnbc_points.png` — Supplementary Figure 2 validated frozen no-match UMAP raster
- `supplementary_umap_no_match/spectra_bassez_c1_points.png` — Supplementary Figure 2 validated frozen no-match UMAP raster
- `supplementary_umap_no_match/spectra_bassez_c2_points.png` — Supplementary Figure 2 validated frozen no-match UMAP raster
- `supplementary_umap_no_match/spectra_wu_tnbc_points.png` — Supplementary Figure 2 validated frozen no-match UMAP raster
- `supplementary_w_pooled/w_pooled_concordance_metrics.csv` — Supplementary Figure 2 pooled W-space input
- `supplementary_w_pooled/w_pooled_concordance_manifest.json` — Supplementary Figure 2 pooled W-space input
- `supplementary_w_pooled/w_pooled_tsne_dataset_panels.png` — Supplementary Figure 2 pooled W-space input
- `supplementary_w_pooled/w_pooled_tsne_node_kinds_corrected.png` — Supplementary Figure 2 pooled W-space input

The authoritative machine-readable mapping is `manifest.json` under `panels`.
All hashes and source provenance are recorded in that manifest.
