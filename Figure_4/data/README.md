# Figure 4 plot inputs

The approved main Figure 4 has panels a–k. `../plot_figure4_one_page.py` reconstructs the data panels from saved plot inputs; it does not refit SingleR, infer CNVs, or repeat differential analysis. Its spacing and annotations can differ from the approved PDF.

| Panels | Saved inputs used by the final layout |
|---|---|
| a–d | `Figure_4_compiled_aj_source.pdf`, the approved source artwork cropped into the final page. |
| e | `fig4_cd_final.h5` for UMAP coordinates, `singler_minor_predictions.csv.gz` for frozen minor calls, and `figure4_kl_data.npz` for the 29-label key. |
| f–g | Reviewer-only `fig4_plot_data.h5` for saved CNV scores and status, aligned to the UMAP coordinates. |
| h–j | `figure4_hij_expression.npz` for cell-aligned, saved log1p-normalized EPCAM, DCN, and C1QA values. |
| k | Reviewer-only `fig4_plot_data.h5` for the seven-patient subtype-correlation matrix. |

The two small plot-ready files for panels e and h–j were extracted without changing values from the frozen full-Wu SingleR minor predictions and final expression matrix. Their cell IDs are checked against `fig4_plot_data.h5` during rendering. The HDF5 file is intentionally absent from public Git; a separate reviewer-only bundle has been prepared for submission. Its expected size and SHA-256 are recorded in `../../private_archive_manifest.tsv`.

Other files in this directory support the saved broad-label Figure 4 views and Supplementary Figure S4. The approved final page is `../outputs/Figure_4.pdf`.
