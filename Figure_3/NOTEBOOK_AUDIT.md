# Figure 3 notebook audit

## Public plotting contract

`Figure_3.ipynb` is the authoritative version-2 plotting entrypoint. It loads finalized method-overlap summaries, patient-level statistics, selected candidates, longitudinal summaries, and frozen plotting coordinates. It renders main panels A–J and exports the compiled figure without running inference.

## Analysis boundary

Exact permutation tests, bootstrap confidence intervals, Cliff’s delta, multiple-testing correction, candidate ranking, method-overlap derivation, longitudinal tests, and biological-evidence table generation remain under `analysis/`. Pooled-cell GO results are descriptive; patient-level replication is the inferential unit.

## Supplementary and exclusions

Supplementary heatmaps and supporting pages are maintained under `Supplementary/Figure_S3/`. Historical versions, staging notebooks, duplicated export cells, caches, and absolute source paths are excluded.

## Acceptance boundary

Panels A–J must render inline from declared plot-ready inputs, export as PNG/PDF, preserve approved labels and numerical annotations, and pass repository checksum and plotting-only validation.
