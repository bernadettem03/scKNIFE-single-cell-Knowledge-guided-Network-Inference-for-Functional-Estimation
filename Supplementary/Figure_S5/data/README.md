# Figure 5 plot-ready data

Canonical corrected 43,574-cell epithelial scKNIFE W/H run

## Panel lineage

| Panel | Title | Kind | Extraction mode | Packaged intermediate inputs |
|---|---|---|---|---|
| A | Full decay-selected W-node enrichment after cell-type graph-node exclusion | data | canonical_result_copy | `inhouse_epithelial_W_graph.parquet`<br>`node_metadata.csv` |
| B | H-factor activity by retained epithelial cell type, ordered by factor-group cluster | data | canonical_result_copy | `inhouse_epithelial_W_graph.parquet`<br>`node_metadata.csv`<br>`epithelial_H_cell_metadata.parquet` |
| C | K=8 W-node definitions ordered by program cluster after cell-type graph-node exclusion | data | canonical_result_copy | `inhouse_epithelial_W_graph.parquet`<br>`node_metadata.csv` |
| D | W-program to shared W/H factor-group to epithelial-cell-type heatmaps | data | canonical_result_copy | `inhouse_epithelial_W_graph.parquet`<br>`node_metadata.csv`<br>`epithelial_H_cell_metadata.parquet` |
| E | W-program to shared W/H factor-group to epithelial-cell-type association network | data | canonical_result_copy | `inhouse_epithelial_W_graph.parquet`<br>`node_metadata.csv`<br>`epithelial_H_cell_metadata.parquet` |
| F | Reconstructed K=8 program activity by epithelial cell type | data | canonical_result_copy | `inhouse_epithelial_W_graph.parquet`<br>`node_metadata.csv`<br>`epithelial_H_cell_metadata.parquet` |
| G | Program activity across cells ordered by epithelial cell type | data | canonical_result_copy | `inhouse_epithelial_W_graph.parquet`<br>`node_metadata.csv`<br>`epithelial_H_cell_metadata.parquet` |
| H | Literature-prioritized reconstructed graph-node scores across epithelial cell types | data | canonical_result_copy | `inhouse_epithelial_W_graph.parquet`<br>`node_metadata.csv`<br>`epithelial_H_cell_metadata.parquet` |

## Auxiliary packaged inputs

- `upstream_provenance.md` — retained upstream provenance notes

The authoritative machine-readable mapping is `manifest.json` under `panels`.
All hashes and source provenance are recorded in that manifest.
