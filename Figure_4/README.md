# Figure 4

Run `python Figure_4/plot_figure4_one_page.py` from the repository root, or execute `Figure_4.ipynb` from this directory. All required plotting inputs are included in `data/`; no separate archive is needed. Install the root `environment.yml`, including Poppler for rendering the source PDF.

The script writes `outputs/Figure_4_main_panel.pdf` and `.png`. Panels e–k are redrawn from saved data; panels a–d are cropped from the included source artwork. This reconstructs the final a–k composition without rerunning upstream analyses. The approved `outputs/Figure_4.pdf` remains the layout reference; spacing can differ.

The cleaned `data/fig4_plot_data.h5` contains only aligned cell identifiers, CNV scores/status, and the seven-row subtype-correlation matrix and subtype names. It contains no patient-ID fields or patient-ID metadata, original specimen crosswalk, or internal server paths. Rows in the subtype matrix are displayed as P1–P7 by the plotting code. Cell identifiers retain technical library/sample prefixes for alignment with the other public inputs.

`figure4_editable.py` and `plotting.py` describe an earlier broad-label layout and are not the entry points for this cleaned final-layout bundle.
