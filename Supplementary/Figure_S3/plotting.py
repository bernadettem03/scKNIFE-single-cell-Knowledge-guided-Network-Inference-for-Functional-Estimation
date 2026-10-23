"""Plot-only access to the two approved Supplementary Figure 3 pages."""
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
def validate_page(page):
    page=page.upper()
    if page not in {'A','B'}: raise ValueError('Expected S3 page A or B')
    manifest=json.loads(MANIFEST.read_text()); path=OUTPUT/f'Figure_3_S3_{page}_final.png'
    if _sha256(path) != manifest['sha256'][f'outputs/{path.name}']: raise RuntimeError(f'S3{page} checksum mismatch')
    size=tuple(PILImage.open(path).size)
    if size != tuple(manifest['page_pixels']): raise RuntimeError(f'Unexpected S3{page} canvas: {size}')
    return manifest
def show_page(page,width=1000):
    page=page.upper(); validate_page(page)
    path=OUTPUT/f'Figure_3_S3_{page}_final.png'
    display(Image(filename=str(path),width=width)); return path.relative_to(ROOT)
