"""Render the current Figure 6 A–T from frozen, plot-ready tables only."""
from pathlib import Path
import shutil

import matplotlib.pyplot as plt

from main_A_G_plotting import load_data, make_compiled, make_panel, export, export_compiled


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
SUBMISSION = WORKSPACE / "May_2026_clean_version_redo/Figures/submission"
FINAL = WORKSPACE / "May_2026_clean_version_redo/Final_writeUps_18thSept/Final Figs /Main/Fig_06.pdf"


def render(root: Path, data_root: Path, compiled: Path, panel_root: Path) -> None:
    data = load_data(data_root)
    fig = make_compiled(data)
    export_compiled(fig, compiled)
    plt.close(fig)
    for label in "ABCDEFGHIJKLMNOPQRST":
        fig = make_panel(label, data)
        export(fig, panel_root / f"Figure_6_{label}")
        plt.close(fig)


def main() -> None:
    render(ROOT, ROOT / "data", ROOT / "outputs/Figure_6", ROOT / "outputs/panels")
    render(SUBMISSION, SUBMISSION / "data/Figure_6_adduct_corrected_v1",
           SUBMISSION / "figs/panel/Figure_6_adduct_corrected_v1/Fig_06_final",
           SUBMISSION / "figs/individual/Figure_6")
    source = ROOT / "outputs/Figure_6.pdf"
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, FINAL)
    for extension in ("pdf", "png"):
        source = SUBMISSION / f"figs/panel/Figure_6_adduct_corrected_v1/Fig_06_final.{extension}"
        shutil.copy2(source, source.with_name(f"Figure_6_adduct_corrected_v1_main.{extension}"))


if __name__ == "__main__":
    main()
