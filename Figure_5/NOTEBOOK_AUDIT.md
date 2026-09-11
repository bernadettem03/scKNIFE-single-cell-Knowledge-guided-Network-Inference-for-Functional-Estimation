# Figure 5 notebook audit

## Public plotting contract

`Figure_5.ipynb` loads the frozen K=8 assignments and approved plot-ready tables, renders authoritative main panels A, C, E, G, and H, and exports the approved composite. Rendering helpers live in `plotting.py`.

## Analysis boundary

Consensus clustering over candidate K values, K selection, component assignment, decay-based node selection, W enrichment, W×H reconstruction, patient summaries, sensitivity analysis, and derived-table creation remain under `analysis/`.

## Supplementary and exclusions

Supplementary node dictionaries, sensitivity results, and panels B, D, E, F, and G are maintained under `Supplementary/Figure_S5/`. Obsolete main-panel B/D/F exports, historical notebooks, staging files, and builder utilities are excluded.

## Acceptance boundary

Only main panels A/C/E/G/H may be emitted by the root notebook. Their standalone PNG/PDF files and the compiled figure must match the approved renderer and pass checksum and plotting-only validation.
