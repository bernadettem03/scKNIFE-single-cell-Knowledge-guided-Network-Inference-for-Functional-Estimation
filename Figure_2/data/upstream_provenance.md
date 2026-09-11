# Figure 2 plot-ready data

This directory contains the complete numerical input bundle used by
`../../fig2.ipynb`. The notebook reads only these packaged files: it does not
access the upstream analysis directories or use pre-rendered plot images.

All provenance paths below are relative to the project root
`source_snapshot/`. They identify where the packaged data were
taken from or derived from; they are not runtime dependencies of the notebook.

## Packaged files and provenance

| Packaged input | Content and derivation | Upstream project source |
|---|---|---|
| `expimap_*_umap.csv.gz`, `slalom_*_umap.csv.gz`, `spectra_*_umap.csv.gz` | Frozen cell IDs, author cell types, Leiden clusters, and UMAP coordinates for Bassez C1, Bassez C2, and Wu TNBC. Comparator factor matrices were standardized, projected by PCA, L2-normalized, clustered with cosine-neighbor Leiden, and embedded with UMAP using seed 42. | Factor matrices: `benchmarking_may/nmf_methods_output/all_factor/data/{method}_factor_cell_data_{dataset}.csv`; selected clustering settings: `benchmarking_may/nmf_methods_output/all_factor/results/ari_evaluation_nomatch.csv`. |
| `scknife_bassez_c1_umap.csv.gz` | Frozen scKNIFE coordinates and labels exported without recomputing the embedding. | `benchmarking_may/bassez_cohort_A_final_run/data/clustering/bassez_A_clustered_with_H.h5ad` (`author_celltype`). |
| `scknife_bassez_c2_umap.csv.gz` | Frozen scKNIFE coordinates and labels exported without recomputing the embedding. | `benchmarking_may/bassez_cohort_B_final_run/data/clustering/bassez_B_clustered_with_H.h5ad` (`cellType`). |
| `scknife_wu_tnbc_umap.csv.gz` | Frozen scKNIFE coordinates and labels exported without recomputing the embedding. | `benchmarking_may/wu_final_run/data/clustering/wu_clustered_with_H.h5ad` (`author_celltype`). |
| `scknife_shiao_umap.csv.gz` | Frozen scKNIFE coordinates and labels exported without recomputing the embedding. | `benchmarking_may/shiao_final_run/data/clustering/shiao_clustered_with_H.h5ad` (`author_celltype`). |
| `agreement_metrics_from_frozen_labels.csv` | Frozen reference ARI, AMI, and NMI values. These were calculated from `author_cell_type` and `leiden_cluster` on the shared-cell intersection of the packaged UMAP tables. The notebook recalculates and checks these values. | Derived from the packaged `*_umap.csv.gz` files above; no separate raw dataset is required. |
| `tuning_sweeps.csv` | Compact extraction of the single-seed `k`, joint `k`–`lambda`, `beta_W`, and `beta_H` tuning results for Bassez C1, Bassez C2, Wu TNBC, and Shiao. | The corresponding `benchmarking_may/{cohort}_final_run/data/tuning/` directories: `tuning_raw_*`, `tuning_summary_*`, `joint_k_lambda_grid_*`, and `best_params_*.json`. Cohort directories are `bassez_cohort_A_final_run`, `bassez_cohort_B_final_run`, `wu_final_run`, and `shiao_final_run`. |
| `tuning_final_parameters.csv` | Four production-used parameter configurations summarized from the final cohort runs. | The same four cohort `data/tuning/` directories plus the parameters used by their final-run configurations. |
| `runtime.csv` | Consolidated wall-clock training durations, in seconds, from the 20,000-cell Bassez C1 and Bassez C2 benchmark runs. `fig2.ipynb` converts seconds to hours and maps the internal `bgNMF` name to `scKNIFE`. | Atomic measurements from the method runs under `benchmarking_may/`; this compact table is the retained source for the figure rather than a copy of raw execution logs. |
| `knowledge_feature_counts.csv` | Comparator gene-set/factor counts and scKNIFE unified-graph node counts used in the capacity panel. | Comparator counts come from their benchmarking input configurations. The scKNIFE benchmarking count (22,951 nodes) comes from `benchmarking_may/bassez_cohort_A_final_run/`; the full in-house count (39,247 nodes) comes from `inhouse_rerun/scknife_run/`. |
| `tahoe_plot_data.h5` | Compact expression, scKNIFE/bGNMF activity, and Scanpy scores for seven Tahoe Hallmark comparisons. It supports all seven standalone comparisons and the TP53/PIK3R1 compiled panels. | Rebuilt from the original Tahoe expression, pipeline, activity-score, and MSigDB inputs; the 1.9 GB expression H5AD, 5.2 GB pipeline H5AD, and 20 GB activity matrix are not required at notebook runtime. |
| `manifest.json` | Package description, row counts, byte sizes, and SHA-256 checksums for all 19 numerical inputs. | Generated for this self-contained Figure 2 data package. |

Dataset slugs map as follows: `bassez_c1` = Bassez cohort A,
`bassez_c2` = Bassez cohort B, and `wu_tnbc` = Wu TNBC. Shiao is included
for scKNIFE and hyperparameter tuning but is not part of the common four-method
agreement comparison.

## UMAP input lineage

The packaged `*_umap.csv.gz` files are plot-ready cell-level tables, not raw
or normalized expression matrices. Their common schema is `cell_id`,
`umap_1`, `umap_2`, `leiden_cluster`, and `author_cell_type`.

```text
expression H5AD
  -> method-specific cell-by-factor representation
  -> Leiden clustering and UMAP
  -> packaged *_umap.csv.gz
  -> fig2.ipynb
```

### SPECTRA, ExpiMap, and Slalom

The immediate inputs for the nine comparator UMAPs are the cell-by-factor
tables in `benchmarking_may/nmf_methods_output/all_factor/data/`:

```text
{method}_factor_cell_data_bassez_c1.csv
{method}_factor_cell_data_bassez_c2.csv
{method}_factor_cell_data_wu_tnbc.csv
```

Each table contains cell IDs, author annotations in `groups`, numeric factor
scores, and dataset metadata such as `timepoint` or `cancer`. The scores were
extracted from these method-specific representations:

| Method | Cell-by-factor representation | Current CSV columns, including index and metadata |
|---|---|---|
| SPECTRA | `adata.obsm["SPECTRA_cell_scores"]`; names from `adata.uns["SPECTRA_overlap"].index` | Bassez C1: 148; Bassez C2: 151; Wu TNBC: 153 |
| ExpiMap | `adata.obsm["X_cvae"]`, produced by `intr_cvae.get_latent(mean=False, only_active=True)` | Bassez C1: 431; Bassez C2: 943; Wu TNBC: 581 |
| Slalom | `FA.getX()`, also stored as `adata.obsm["X_slalom"]` | Bassez C1: 113; Bassez C2: 121; Wu TNBC: 100 |

The exporter excludes `groups` and other nonnumeric metadata from the
embedding matrix. It standardizes numeric factor columns, applies PCA and L2
normalization, constructs cosine neighbors, and calculates Leiden and UMAP
with seed 42. PCA dimensions, neighbor counts, and Leiden resolutions come
from `benchmarking_may/nmf_methods_output/all_factor/results/ari_evaluation_nomatch.csv`.

The three surviving SPECTRA method-output H5ADs are:

```text
PRIVATE_ARCHIVE_ROOT/home_tnbc/bgnmf_benchmarking/spectra/
  adata_with_spectra_cohort1.h5ad
  adata_with_spectra_cohort2.h5ad
  adata_with_spectra_wu_tnbc.h5ad
```

Archived ExpiMap and Slalom implementations document their latent-score
extraction at:

```text
PRIVATE_ARCHIVE_ROOT/methodValidation/knief_github/TNBC-Project/
  Benchmarking/sota_methods/expimap/expiMap.py
  Benchmarking/sota_methods/slalom/slalom_training.py
```

The trained ExpiMap and Slalom model directories/H5ADs referenced by the
legacy analysis are not retained beside the active benchmark. Their factor
CSVs are therefore the authoritative frozen Figure 2 inputs. SPECTRA has the
stronger recoverable upstream chain because all three method-output H5ADs
survive. Shiao has no SPECTRA, ExpiMap, or Slalom UMAP in this package.

### scKNIFE

The extractor does not recalculate scKNIFE UMAPs. It copies `obsm["X_umap"]`,
`obs["leiden_clusters_str"]`, author cell types, and `obs_names` from the
clustered H5ADs listed above. Those runs ultimately use:

| Dataset | Original expression input | Main preprocessing |
|---|---|---|
| Bassez C1 | `PRIVATE_ARCHIVE_ROOT/methodValidation/Bassez_dataset/1863-counts_cells_cohort1.h5ad` plus `1872-BIOKEY_metaData_cohort1_web.csv` | TNBC subset, at least 200 detected genes, top 7,000 genes by mean expression, log-CPM |
| Bassez C2 | `PRIVATE_ARCHIVE_ROOT/methodValidation/Bassez_dataset/1867-counts_cells_cohort2.h5ad` plus `1871-BIOKEY_metaData_cohort2_web.csv` | Same TNBC, cell, gene, and normalization filters |
| Wu TNBC | `PRIVATE_ARCHIVE_ROOT/methodValidation/TNBC Data Wu 2021.h5ad` | Metadata from `obs`, at least 200 detected genes, top 7,000 genes, log-CPM |
| Shiao | `PRIVATE_ARCHIVE_ROOT/Complete_Analysis/scRNA_Our_Patients/ref_dataset/GSE246613_PembroRT_immune_R100_final.h5ad` and `GSE246613_PembroRT_non_immune_cells.h5ad` | Common-gene alignment, immune/non-immune merge, paired-patient subset, filtering, log-CPM |

These processed expression matrices are factorized by scKNIFE; downstream
clustering stores the final coordinates in the clustered H5ADs. Consequently,
the comparator UMAPs are regenerated from frozen factor CSVs, while the
scKNIFE UMAP CSVs preserve already-computed coordinates.

## Reproducibility boundary

`fig2.ipynb` regenerates the bar charts, heatmaps, UMAP scatter panels, tuning
plots, and agreement metrics from this directory. It does **not** rerun
scKNIFE, SPECTRA, ExpiMap, or Slalom from raw expression matrices. Therefore,
this is a self-contained plotting and evaluation package based on frozen
analysis results, not the complete raw-data benchmarking pipeline. Complete
end-to-end regeneration of the legacy ExpiMap and Slalom factor tables would
require rerunning their archived training workflows.
