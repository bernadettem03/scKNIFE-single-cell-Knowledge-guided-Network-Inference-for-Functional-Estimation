# Figure 6 notebook audit

- Public entrypoint: Figure_6.ipynb
- Main panel contract: A–H
- Canvas: 297 × 380 mm
- Inputs: repository-relative files under data/
- Statistics: frozen outside plotting by analysis/migrate_blank_corrected_package.py
- Output formats: PNG, PDF and SVG; PDF/SVG use editable text
- Rendering stack: matplotlib fallback under Python 3.9 (validated source preserved; cnsplots not introduced)
- Protected working source: May_2026_clean_version_redo/Figures/submission/Figure_6_blank_corrected.ipynb (not modified)
- Supplementary split: Supplementary/Figure_S6/Figure_S6.ipynb

The public notebook contains no model fitting, hypothesis testing, multiple-testing correction, or absolute storage paths.

## 22 September 2026 panel-C update

This targeted revision supersedes the original migration's statement that the working source was not modified: panel C was updated in both the working and public notebooks. Only the C and compiled-main notebook displays were refreshed; standalone non-C displays and all supplementary outputs were retained. Pooled coefficients and full-data bin counts are derived in `analysis/derive_pooled_score_concordance.py`; the plotting notebook does not recompute them. The two branches share byte-identical renderer code, numerical inputs and standalone C PNGs. PNG/PDF/SVG exports retain rasterized density and vector text. No analysis refit, cell-level inferential P value, new package installation, git commit or push was performed.

## 23 September 2026 fitted lines

Dotted charcoal lines show ordinary least-squares fits of scKNIFE scores on MEIA scores, computed separately from all matched task or pathway observations. These are descriptive fits; no confidence bands or inferential P values are shown. Existing correlations, density-bin counts and non-C panels are unchanged.
