# scKNIFE manuscript figure package

This repository contains the approved plots, plotting code, legends, and plot-ready inputs for main Figures 2–7 and Supplementary Figures S1–S6 of the 23 October 2026 portal manuscript draft. It is a figure reproduction package, not the scKNIFE model implementation or the complete analysis history.

## Organization

- `Figure_2/` through `Figure_7/`: main figure notebooks, plotting modules, packaged inputs, and exports.
- `Supplementary/Figure_S1/` through `Figure_S6/`: supplementary pages and their inputs.
- `data_manifest.tsv` and `checksums.sha256`: inventories for the public package.
- `SOURCE_PROVENANCE.md`: the local source snapshot and portal-draft identifiers used for this release.

The manuscript's Figure 1 is conceptual artwork and is outside this package. The final portal PDFs and raw study deposits are not copied here. The scKNIFE implementation is maintained separately at <https://github.com/bernadettem03/scKNIFE>.

## Reproducing plots

Create the environment with `conda env create -f environment.yml`, activate it, and run `python validate_repository.py`. Figure notebooks use repository-relative plot-ready inputs and save final outputs under their own `outputs/` folders. The retained notebook outputs let readers inspect the figures without rerunning them.

Figure 4a–k is reproduced from files included here. Run `python Figure_4/plot_figure4_one_page.py` to regenerate its PDF, PNG and SVG; `Figure_4/Figure_4.ipynb` runs the same renderer. The included vector layout base preserves panels a–j from the 23 October portal figure. Panel k is drawn from the patient-aligned TNBC subtype values in the public `data/singler/overlays.h5`; no private plotting bundle is needed.

Raw study data are deposited separately: single-cell data under ENA PRJEB108811 and LC–MS data under MassIVE MSV000103682. This repository contains only selected processed values required for the figures; model fitting and inferential derivations are documented in the manuscript and its supplemental tables.

Patient labels in the public package use P1–P7. The original specimen-to-pseudonym crosswalk remains outside this repository. No publication license or DOI is asserted here.
