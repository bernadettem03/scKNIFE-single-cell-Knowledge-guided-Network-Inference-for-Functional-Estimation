"""Plot-only access to the approved Supplementary Figure 5 render state.

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
PANEL_ORDER=tuple('ABCDEF')  # Legacy grouped standalone exports.
COMPOSITE_ORDER=tuple('ABCDEFGHIJK')
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
    if tuple(manifest['canvas_mm']) != (210,297): raise RuntimeError('Figure S5 must use A4 portrait pages')
    if manifest['composite_order'] != list(COMPOSITE_ORDER):
        raise RuntimeError('Unexpected Figure S5 display panel order')
    if [page['panels'] for page in manifest['pages']] != [list(COMPOSITE_ORDER)]:
        raise RuntimeError('Unexpected Figure S5 panel placement')
    for page in manifest['pages']:
        path=OUTPUT/'Figure_S5.png'
        with PILImage.open(path) as image: size=tuple(image.size)
        if size != tuple(page['pixels']): raise RuntimeError(f'Unexpected Figure S5 page canvas: {path}: {size}')
    return manifest
def show_panel(letter,width=900):
    letter=letter.upper()
    if letter not in PANEL_ORDER: raise ValueError(f'Expected one of {PANEL_ORDER}')
    path=PANELS/f'Figure_5_supplementary_{letter}.png'; display(Image(filename=str(path),width=width)); return path
def show_figure(page=1,width=900):
    validate_assets()
    if page != 1: raise ValueError('Figure S5 has one page')
    path=OUTPUT/'Figure_S5.png'; display(Image(filename=str(path),width=width)); return path
