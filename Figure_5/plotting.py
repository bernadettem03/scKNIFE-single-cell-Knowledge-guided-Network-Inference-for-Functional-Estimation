"""Plot-only access to the approved Figure 5 render state.

The reproducible analytical workflow is retained under ``analysis/``. This module
never reconstructs H/W matrices, clusters factors, selects nodes, or calculates
statistics. It checksum-validates and displays the approved panel renderings.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from IPython.display import Image, display
from PIL import Image as PILImage
ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'outputs'; PANELS=OUTPUT/'panels'; MANIFEST=ROOT/'approved_render_manifest.json'
PANEL_ORDER=tuple(json.loads(MANIFEST.read_text())['composite_order'])
def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def validate_assets():
    manifest=json.loads(MANIFEST.read_text())
    for relative,expected in manifest['sha256'].items():
        path=ROOT/relative
        if not path.is_file(): raise FileNotFoundError(path)
        observed=_sha256(path)
        if observed != expected: raise RuntimeError(f'Checksum mismatch for {relative}: {observed}')
    size=tuple(PILImage.open(OUTPUT/'Figure_5.png').size)
    if size != tuple(manifest['composite_pixels']): raise RuntimeError(f'Unexpected Figure 5 canvas: {size}')
    return manifest
def show_panel(letter,width=900):
    letter=letter.upper()
    if letter not in PANEL_ORDER: raise ValueError(f'Expected one of {PANEL_ORDER}')
    path=PANELS/f'Figure_5_{letter}.png'; display(Image(filename=str(path),width=width)); return path
def show_figure(width=900):
    validate_assets(); path=OUTPUT/'Figure_5.png'; display(Image(filename=str(path),width=width)); return path
