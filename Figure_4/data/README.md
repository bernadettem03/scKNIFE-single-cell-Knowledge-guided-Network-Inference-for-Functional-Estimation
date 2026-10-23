# Figure 4 plot inputs

The main Figure 4 has panels a–k. `../plot_figure4_one_page.py` preserves the submitted page layout and draws the patient-aligned subtype panel from saved public values. It does not refit SingleR, infer CNVs, or repeat differential analysis.

| Panels | Saved inputs used by the final layout |
|---|---|
| a–j | `Figure_4_layout_base.pdf`, the submitted vector page with a blank panel-k region. |
| e–j plot values | `fig4_cd_final.h5`, `singler_minor_predictions.csv.gz`, `figure4_kl_data.npz`, `singler/overlays.h5` and `figure4_hij_expression.npz` retain the saved values represented in the layout base. |
| k | Public `singler/overlays.h5` supplies the seven-patient subtype-correlation matrix and its P1–P7 row labels, used directly by the renderer. |

The two small plot-ready files for panels e and h–j were extracted without changing values from the frozen full-Wu SingleR minor predictions and final expression matrix. Their cell IDs are checked against each other during rendering. The saved CNV values in `singler/overlays.h5` match the original plot bundle exactly; its subtype rows contain the same values sorted and labeled P1–P7. The renderer checks that the saved UMAP coordinates match `fig4_cd_final.h5`.

Other files in this directory support the saved broad-label Figure 4 views and Supplementary Figure S4. The approved final page is `../outputs/Figure_4.pdf`.

`fig4_plot_data.h5` is a cleaned public reference containing CNV scores, status and subtype values; the final renderer uses `singler/overlays.h5` for panel k.
