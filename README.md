# scKNIFE manuscript figure package

This repository contains the approved plots, plotting code, plot-ready inputs, and supplemental tables for main Figures 2–7 and Supplementary Figures S1–S6 of the 23 October 2026 portal manuscript draft. It is a figure reproduction package, not the scKNIFE model implementation or the complete analysis history.

## Organization

- `Figure_2/` through `Figure_7/`: main figure notebooks, plotting modules, packaged inputs, and exports.
- `Supplementary/Figure_S1/` through `Figure_S6/`: supplementary pages and their inputs.
- [Tables/](Tables/README.md): Supplemental Tables 1–7 in Excel and TSV formats.
- `data_manifest.tsv` and `checksums.sha256`: inventories for the public package.
- `SOURCE_PROVENANCE.md`: the local source snapshot and portal-draft identifiers used for this release.

The manuscript's Figure 1 is conceptual artwork and is outside this package. The final portal PDFs and raw study deposits are not copied here. The scKNIFE implementation is maintained separately at <https://github.com/bernadettem03/scKNIFE>.

## Reproducing plots

Create the environment with `conda env create -f environment.yml`, activate it, and run `python validate_repository.py`. Figure notebooks use repository-relative plot-ready inputs and save final outputs under their own `outputs/` folders. The retained notebook outputs let readers inspect the figures without rerunning them.

The Figure 4 plotting bundle `Figure_4/data/fig4_plot_data.h5` is included for public reproduction. Run `python Figure_4/plot_figure4_one_page.py` from the repository root, or execute `Figure_4/Figure_4.ipynb`. Panels e–k are rendered from saved numerical inputs; panels a–d use the included approved source artwork. This reproduces the figure composition from saved results, not the upstream inference. The submitted PDF remains the layout reference.

Raw study data are deposited separately: single-cell data under ENA PRJEB108811 and LC–MS data under MassIVE MSV000103682. This repository contains selected processed values for the figures and supplemental tables; model fitting and inferential derivations are documented in the manuscript and its supplemental tables.

Patient labels in the public package use P1–P7. The original specimen-to-pseudonym crosswalk remains outside this repository. No publication license or DOI is asserted here.
