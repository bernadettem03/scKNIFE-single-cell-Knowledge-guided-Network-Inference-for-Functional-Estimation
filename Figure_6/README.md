# Figure 6 — Blank-corrected metabolic evidence story

Submission legend: [LEGEND.txt](LEGEND.txt).

Current compiled figure: [PNG](outputs/Figure_6.png) | [PDF](outputs/Figure_6.pdf) | [GitHub preview](outputs/Figure_6_preview.png). The preview is refreshed as an exact byte-for-byte copy of the compiled PNG on every render.

Figure_6.ipynb is the authoritative plotting-only entrypoint for main panels A–H. It regenerates every standalone panel and the compiled 297 × 380 mm figure from repository-relative, checksummed inputs.

Story order: A–C establish cross-platform overlap, blank-corrected LC–MS pathway enrichment, and MEIA/direct-scKNIFE concordance; D integrates discovery-cohort treatment, P3 recurrence and metastatic-status evidence for two CAF Tier-3 nodes; E summarizes the metabolic pathway context of 68 robust inferred metabolites; F localizes paired macrophage homocysteine-degradation and glutathione-synthesis task changes; G supplies directionally concordant, accurate-mass-only LC–MS support; H shows the NG8 knowledge-backbone subgraph as biochemical context.

Inferential calculations are isolated under analysis/. plotting.py only validates frozen tables and renders figures. PDF and SVG retain editable vector text; PNG files are review previews.

## Panel C revision — 22 September 2026

Pooled matched-score concordance with MEIA activity on x and direct scKNIFE activity on y. The two density plots separately include all 218 tasks (15,127,020 cell–task pairs; Pearson r=0.8981, Spearman ρ=0.9170) and all 138 pathways (9,575,820 cell–pathway pairs; r=0.8934, ρ=0.9684), across the same 69,390 matched cells. Both score matrices retain the existing separate task/pathway compositional normalization. Axes are linear and retain the full observed ranges; colour shows cell–feature pairs per bin on a shared logarithmic count scale. Dotted charcoal lines show ordinary least-squares fits of scKNIFE scores on MEIA scores, computed separately from all matched task or pathway observations. These are descriptive fits; no confidence bands or inferential P values are shown. All observations contribute to the bins and coefficients, without downsampling or axis trimming. These pooled coefficients include between-feature differences and are not typical individual-feature correlations. Shared-input concordance is descriptive technical agreement, not independent biological validation or measured flux.

[Panel C PNG](outputs/panels/Figure_6_C.png) | [PDF](outputs/panels/Figure_6_C.pdf) | [SVG](outputs/panels/Figure_6_C.svg).

Numerical inputs: [pooled coefficients](data/meia_vs_scknife_pooled_correlations.csv) and [full-data density counts](data/meia_vs_scknife_pooled_density.csv.gz). The [provenance record](data/meia_vs_scknife_pooled_provenance.json) stores input hashes, alignment checks, feature identities and software versions. `analysis/derive_pooled_score_concordance.py` derives these frozen inputs from the existing score matrices; `pooled_concordance.py` only validates and renders them. The original `meia_vs_scknife_entity_correlations.csv` remains supporting data and is not used to draw the new C. Supplementary Figure S6 is unchanged.
