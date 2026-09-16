# Caption coherence audit — 16 September 2026

## Scope

The authoritative legends under `May_2026_clean_version_redo/Methods_section/plots description` were checked against the plotting modules, executed notebooks, frozen data tables, panel manifests, and compiled outputs in this repository. `validate_repository.py` passed for all 12 plotting notebooks before and after the documentation revision.

## Resolved documentation issues

- Added synchronized `LEGEND.txt` files beside every main and supplementary plotting workflow.
- Aligned legends with the current panel contracts, including Figure 6 A–H and Supplementary Figure S6 SA/SB/SJ/S1/SFUNNEL/SDIRECTION/SU3.
- Converted Figure 2 runtime reporting from seconds to hours to match the plotted axis.
- Condensed Figure 6 and S6 while retaining result-defining counts, effects, tests, correction families, and interpretation boundaries.
- Preserved the former exhaustive Figure 6 and S6 captions in dated full-statistics audit files outside this repository.
- Corrected repository prose that referred to 11 notebooks, Figure 6 A–F, or a nonexistent Figure 6 derivation script.

## Open plot-code issue

The Figure 6/S6 UpSet renderer contains the MEIA-Pathway/GPR-GPA-only relationship twice. This produces a repeated 124 bar in main Figure 6 and a repeated 111 bar in S6. The mathematically correct exclusive intersections, stated once in the legends, are:

- Treatment: 218 MEIA-task only; 124 MEIA-pathway/GPR-GPA only; 14 shared with mapped LC–MS; 4 unmapped LC–MS.
- Metastatic status: 218 MEIA-task only; 111 MEIA-pathway/GPR-GPA only; 27 shared with mapped LC–MS; 3 unmapped LC–MS.

The renderer and approved figures were not changed because this revision was explicitly limited to descriptions.

## Interpretation safeguards

- Patients or specimens are biological replicates; pooled-cell statistics are descriptive.
- The P3 recurrence analysis contains one patient and is a mechanistic illustration only.
- MEIA, GPR-GPA, and scKNIFE scores are inferred activities, not abundance or flux.
- Mummichog pathway enrichment is not directional.
- Accurate-mass MS1 candidates are not confirmed metabolites without native MS/MS and authentic-standard validation.
