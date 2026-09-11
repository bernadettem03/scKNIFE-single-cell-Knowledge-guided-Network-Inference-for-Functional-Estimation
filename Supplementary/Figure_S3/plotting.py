"""Plot-only access to the approved Supplementary Figure 3 pages.

S3B is the canonical 2x2 Bassez metabolic-task/GO heatmap composition. Statistical
selection and clustering provenance are retained under ``analysis/`` and ``data/``.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from IPython.display import Image, display
from PIL import Image as PILImage
ROOT=Path(__file__).resolve().parent; OUTPUT=ROOT/'outputs'; MANIFEST=ROOT/'approved_render_manifest.json'
def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def validate_s3b():
    manifest=json.loads(MANIFEST.read_text()); path=OUTPUT/'Figure_S3_B.png'
    if _sha256(path) != manifest['sha256']['outputs/Figure_S3_B.png']: raise RuntimeError('S3B checksum mismatch')
    size=tuple(PILImage.open(path).size)
    if size != tuple(manifest['s3b_pixels']): raise RuntimeError(f'Unexpected S3B canvas: {size}')
    return manifest
def show_page(page,width=1000):
    page=page.upper(); path=OUTPUT/f'Figure_S3_{page}.png'
    if page not in {'A','B','C'} or not path.exists(): raise ValueError('Expected an available S3 page: A, B, or C')
    if page=='B': validate_s3b()
    display(Image(filename=str(path),width=width)); return path.relative_to(ROOT)
