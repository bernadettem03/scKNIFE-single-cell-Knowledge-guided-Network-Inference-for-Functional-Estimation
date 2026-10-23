"""Render the submitted Figure 4a–k layout from packaged public inputs.

The vector base contains the submitted a–j artwork and a blank panel-k region.
Panel k is drawn from the saved, patient-aligned subtype correlation matrix.
No model fitting, annotation, CNV inference or subtype assignment is run here.
"""
from pathlib import Path

import fitz
import h5py
import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"
BASE = DATA / "Figure_4_layout_base.pdf"
OVERLAYS = DATA / "singler" / "overlays.h5"
EMBEDDING = DATA / "fig4_cd_final.h5"
PATIENTS = [f"P{i}" for i in range(1, 8)]
SUBTYPES = ["IM", "BL1", "MSL", "M", "BL2", "LAR"]
COLOR_LIMIT = 0.6
# PDF coordinates are measured from the top-left of the submitted A4 page.
GRID_X, GRID_Y = 354.0, 555.4
CELL_W, CELL_H = 22.0, 28.53
DPI = 300


def _text_center(page, label, x, baseline, *, size=8.0, color=(0, 0, 0)):
    width = fitz.get_text_length(label, fontname="helv", fontsize=size)
    page.insert_text((x - width / 2, baseline), label, fontname="helv",
                     fontsize=size, color=color)


def _load_values():
    with h5py.File(OVERLAYS, "r") as saved, h5py.File(EMBEDDING, "r") as embedding:
        assert np.array_equal(saved["umap"][:], embedding["umap"][:])
        assert saved["cnv_score"].shape == saved["cnv_status"].shape == (69_391,)
        patients = list(saved["subtype_patients"].asstr()[:])
        names = list(saved["subtype_names"].asstr()[:])
        values = saved["subtype_matrix"][:]
    assert patients == PATIENTS, patients
    assert names == SUBTYPES, names
    assert values.shape == (7, 6) and np.isfinite(values).all()
    minor_ids = pd.read_csv(DATA / "singler_minor_predictions.csv.gz",
                            usecols=["cell_id"])["cell_id"].to_numpy()
    with np.load(DATA / "figure4_hij_expression.npz", allow_pickle=False) as saved:
        assert np.array_equal(minor_ids, saved["cell_id"])
        assert saved["expression"].shape == (69_391, 3)
    return values


def _draw_subtype_panel(page, values):
    page.insert_text((327.2, 545.5), "k", fontname="hebo", fontsize=10.5)
    cmap = matplotlib.colormaps["RdBu_r"]
    for column, name in enumerate(SUBTYPES):
        _text_center(page, name, GRID_X + (column + 0.5) * CELL_W, 552.5, size=8.2)
    for row, patient in enumerate(PATIENTS):
        cy = GRID_Y + (row + 0.5) * CELL_H
        page.insert_text((344.1, cy + 2.9), patient, fontname="helv", fontsize=8.1)
        for column, value in enumerate(values[row]):
            x0 = GRID_X + column * CELL_W
            y0 = GRID_Y + row * CELL_H
            rgb = cmap((float(value) + COLOR_LIMIT) / (2 * COLOR_LIMIT))[:3]
            page.draw_rect(fitz.Rect(x0, y0, x0 + CELL_W, y0 + CELL_H),
                           color=(1, 1, 1), fill=rgb, width=0.2)
            color = (1, 1, 1) if abs(float(value)) > 0.30 else (0, 0, 0)
            _text_center(page, f"{value:.2f}", x0 + CELL_W / 2,
                         cy + 3.0, size=7.7, color=color)
    page.draw_rect(fitz.Rect(GRID_X, GRID_Y, GRID_X + 6 * CELL_W,
                             GRID_Y + 7 * CELL_H), color=(0, 0, 0), width=0.35)
    bar_x0, bar_x1 = 491.7, 497.1
    bar_y0, bar_y1 = 556.9, 743.0
    for i in range(128):
        y0 = bar_y0 + (bar_y1 - bar_y0) * i / 128
        y1 = bar_y0 + (bar_y1 - bar_y0) * (i + 1) / 128
        value = COLOR_LIMIT * (1 - 2 * (i + 0.5) / 128)
        rgb = cmap((value + COLOR_LIMIT) / (2 * COLOR_LIMIT))[:3]
        page.draw_rect(fitz.Rect(bar_x0, y0, bar_x1, y1),
                       color=rgb, fill=rgb, width=0)
    page.draw_rect(fitz.Rect(bar_x0, bar_y0, bar_x1, bar_y1),
                   color=(0, 0, 0), width=0.3)
    for value in (0.6, 0.3, 0.0, -0.3, -0.6):
        y = bar_y0 + (COLOR_LIMIT - value) / (2 * COLOR_LIMIT) * (bar_y1 - bar_y0)
        page.insert_text((500.2, y + 2.6), f"{value:.1f}",
                         fontname="helv", fontsize=7.4)
    page.insert_text((528.0, 693.0), "Correlation to subtype centroid",
                     fontname="helv", fontsize=7.2, rotate=90)


def main():
    values = _load_values()
    OUTPUT.mkdir(exist_ok=True)
    doc = fitz.open(BASE)
    assert len(doc) == 1 and round(doc[0].rect.width) == 595
    page = doc[0]
    _draw_subtype_panel(page, values)
    doc.save(OUTPUT / "Figure_4.pdf", garbage=0, deflate=False)
    page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72), alpha=False).save(
        OUTPUT / "Figure_4.png")
    (OUTPUT / "Figure_4.svg").write_text(page.get_svg_image(text_as_path=False))
    doc.close()
    print("Rendered submitted-layout Figure 4a–k from public inputs")


if __name__ == "__main__":
    main()
