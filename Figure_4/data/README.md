# Figure 4 plot-ready data

Frozen HDF5 arrays for the authoritative mixed raster/vector renderer

## Panel lineage

| Panel | Title | Kind | Extraction mode | Packaged intermediate inputs |
|---|---|---|---|---|
| B | TNBC subtype | data | staged_frozen | `fig4_plot_data.h5` (`panel_g/patient`, `panel_g/subtype`, `panel_g/subtype_correlation_matrix`, `panel_g/subtype_correlation_subtypes`)<br>`patient_id_mapping.csv` |
| C | DE-based final cell type | data | staged_frozen | `fig4_plot_data.h5` (`shared/umap`, `panel_a/scknife_final_celltype`) |
| D | Canonical H-cluster marker expression | data | staged_frozen | `fig4_plot_data.h5` (`shared/umap`, `panel_f/marker_expression`, `panel_f/marker_genes`, `panel_f/marker_h_clusters`, `panel_f/marker_labels`) |
| E | CNV score | data | staged_frozen | `fig4_plot_data.h5` (`shared/umap`, `panel_e/cnv_status`, `panel_e/cnv_score`) |
| F | All-cell H-cluster marker expression | data | staged_frozen | `fig4_plot_data.h5` (`panel_c/heatmap`, `panel_c/display_genes`, `panel_c/h_cluster`, `panel_c/boundaries`, `panel_c/marker_groups`, `panel_c/patient`, `panel_c/site`, `panel_c/metastatic`, `panel_c/library`, `panel_c/scknife_final_celltype`) |

## Auxiliary packaged inputs

- `upstream_provenance.md` — retained upstream provenance notes
- `patient_metadata.csv` — de-identified patient-level clinical metadata for Figure 4

The authoritative machine-readable mapping is `manifest.json` under `panels`.
All hashes and source provenance are recorded in that manifest.
