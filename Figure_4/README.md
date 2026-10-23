# Figure 4

`outputs/Figure_4.pdf` is the one-page Figure 4a–k in the submitted page layout. `LEGEND.txt` gives its panel definitions. `plot_figure4_one_page.py` uses the included vector layout base for panels a–j and redraws panel k from the patient-aligned public subtype matrix.

Run `python plot_figure4_one_page.py` here, or execute `Figure_4.ipynb`. The public `data/singler/overlays.h5` supplies the saved CNV scores, status and P1–P7 subtype correlations. The minor-label and expression inputs for panels e and h–j are also packaged under `data/` for inspection.

The notebook and script both produce the a–k layout. They do not rerun SingleR, CNV inference or subtype assignment. The layout base contains a blank panel-k region, so only the corrected patient-aligned panel is displayed in the final outputs.
