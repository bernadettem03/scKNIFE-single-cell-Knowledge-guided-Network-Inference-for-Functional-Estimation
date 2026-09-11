# Figure 7 plot-ready data

Current nine-cell-type SCKNIFE signaling rerun, CellChat comparison, and matched H-graph UMAP overlays

## Panel lineage

| Panel | Title | Kind | Extraction mode | Packaged intermediate inputs |
|---|---|---|---|---|
| A | CellChat and SCKNIFE signaling comparison workflow | schematic | schematic | None (schematic) |
| B | Pathway overlap | data | derived_and_staged | `pathway_overlap.csv` |
| C | Harmonized ligand-receptor overlap | data | derived_and_staged | `lr_harmonized_overlap.csv` |
| D | CellChat-SCKNIFE signaling-strength agreement | data | derived_and_staged | `strength_correlation.csv.gz` |
| E | VEGF cell-type communication network | data | derived_and_staged | `vegf_communication_matrix.csv` |
| F | VEGF sender-receiver communication | data | derived_and_staged | `vegf_communication_matrix.csv` |
| G | Aggregated VEGF ligand-receptor scores | data | derived_and_staged | `vegf_top_interactions.csv.gz` |
| H | KDR activity on the H-graph UMAP | data | derived_and_staged | `h_graph_umap_scores.parquet` |
| I | HALLMARK_HYPOXIA activity on the H-graph UMAP | data | derived_and_staged | `h_graph_umap_scores.parquet` |
| J | FLT1 activity on the H-graph UMAP | data | derived_and_staged | `h_graph_umap_scores.parquet` |

The authoritative machine-readable mapping is `manifest.json` under `panels`.
All hashes and source provenance are recorded in that manifest.
