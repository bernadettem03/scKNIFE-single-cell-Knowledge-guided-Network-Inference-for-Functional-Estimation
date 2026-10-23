#!/usr/bin/env python3
"""Validate the portal-aligned, plotting-only public figure package."""
from __future__ import annotations
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parent
MAIN = range(2, 8)
SUPP = range(1, 7)
PRIVATE = ROOT / "Figure_4/data/fig4_plot_data.h5"
ORIGINAL_ID = re.compile(rb"(?:CF|CN|CV)/[0-9]{5}|11F2023/[0-9]{5}")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def order(path: str, key: str, expected: str, errors: list[str]) -> None:
    file = ROOT / path
    if not file.is_file():
        errors.append(f"missing manifest: {path}")
        return
    observed = json.loads(file.read_text()).get(key)
    if observed != list(expected):
        errors.append(f"panel order in {path}: {observed!r}; expected {list(expected)!r}")


def main() -> int:
    errors: list[str] = []
    for n in MAIN:
        for relative in (f"Figure_{n}/Figure_{n}.ipynb", f"Figure_{n}/LEGEND.txt", f"Figure_{n}/outputs/Figure_{n}.pdf"):
            if not (ROOT / relative).is_file():
                errors.append(f"missing main figure file: {relative}")
    for n in SUPP:
        for relative in (f"Supplementary/Figure_S{n}/Figure_S{n}.ipynb", f"Supplementary/Figure_S{n}/LEGEND.txt"):
            if not (ROOT / relative).is_file():
                errors.append(f"missing supplementary figure file: {relative}")
        if n != 3 and not (ROOT / f"Supplementary/Figure_S{n}/outputs/Figure_S{n}.pdf").is_file():
            errors.append(f"missing supplementary PDF: S{n}")
    s3 = ROOT / "Supplementary/Figure_S3/outputs"
    expected_s3 = {f"Figure_3_S3_{letter}_final.{ext}" for letter in "AB" for ext in ("pdf", "png")}
    if not s3.is_dir() or {p.name for p in s3.iterdir() if p.is_file()} != expected_s3:
        errors.append("S3 must contain exactly the approved A and B PDF/PNG pages")
    s3_notebook = ROOT / "Supplementary/Figure_S3/Figure_S3.ipynb"
    if s3_notebook.is_file():
        s3_text = "\n".join(c.source for c in nbformat.read(s3_notebook, 4).cells if c.cell_type == "markdown")
        if "A–D across the top" not in s3_text or "heatmaps E–F below" not in s3_text:
            errors.append("S3 notebook description does not match approved panel layout")
    if (ROOT / "Figure_4/FIGURE4_CD_UPDATE.md").exists():
        errors.append("obsolete Figure 4 update note remains in public package")
    expression_input = ROOT / "Figure_4/data/figure4_hij_expression.npz"
    if expression_input.is_file():
        with np.load(expression_input, allow_pickle=False) as values:
            if (list(values["genes"]) != ["EPCAM", "DCN", "C1QA"]
                    or values["cell_id"].shape != (69391,)
                    or values["expression"].shape != (69391, 3)):
                errors.append("invalid Figure 4 plot-ready expression input")
    else:
        errors.append("missing Figure 4 plot-ready expression input")
    order("Figure_6/outputs/panels/Figure_6_panel_export_manifest.json", "panel_order", "ABCDEFGHIJKLMNOPQRST", errors)
    order("Figure_7/outputs/Figure_7_main_export_manifest.json", "active_panels", "ABCDEFGHIJKLMNO", errors)
    order("Supplementary/Figure_S4/outputs/Figure_S4_manifest.json", "panel_order", "abcdefg", errors)
    order("Supplementary/Figure_S6/outputs/Figure_S6_export_manifest.json", "panel_order", "ABCDE", errors)
    for directory, stem, labels in (
        (ROOT / "Figure_6/outputs/panels", "Figure_6", "ABCDEFGHIJKLMNOPQRST"),
        (ROOT / "Supplementary/Figure_S6/outputs/panels", "Figure_S6", "ABCDE"),
    ):
        for label in labels:
            for ext in ("pdf", "png"):
                if not (directory / f"{stem}_{label}.{ext}").is_file():
                    errors.append(f"missing {stem} panel {label}.{ext}")
    for n in MAIN:
        p = ROOT / f"Figure_{n}/Figure_{n}.ipynb"
        if p.is_file():
            nb = nbformat.read(p, 4)
            outputs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])]
            if any(o.output_type == "error" for o in outputs):
                errors.append(f"saved notebook error: Figure {n}")
            if not any(any(k.startswith("image/") for k in o.get("data", {})) for o in outputs):
                errors.append(f"no inline image: Figure {n}")
    for n in SUPP:
        p = ROOT / f"Supplementary/Figure_S{n}/Figure_S{n}.ipynb"
        if p.is_file() and n != 3:
            nb = nbformat.read(p, 4)
            outputs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])]
            if any(o.output_type == "error" for o in outputs):
                errors.append(f"saved notebook error: S{n}")
            if not any(any(k.startswith("image/") for k in o.get("data", {})) for o in outputs):
                errors.append(f"no inline image: S{n}")
    checksum_file = ROOT / "checksums.sha256"
    listed: set[str] = set()
    if checksum_file.is_file():
        for line in checksum_file.read_text().splitlines():
            hash_value, rel = line.split("  ", 1)
            listed.add(rel)
            p = ROOT / rel
            if not p.is_file() or digest(p) != hash_value:
                errors.append(f"checksum mismatch: {rel}")
    else:
        errors.append("missing checksums.sha256")
    for p in ROOT.rglob("*"):
        if not p.is_file() or p == PRIVATE:
            continue
        if any(part in {".git", "__pycache__", ".ipynb_checkpoints"} for part in p.relative_to(ROOT).parts):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel not in {"checksums.sha256", "data_manifest.tsv"} and rel not in listed:
            errors.append(f"unlisted public file: {rel}")
        if p.stat().st_size >= 100_000_000:
            errors.append(f"file exceeds GitHub 100 MB limit: {rel}")
        if p.suffix in {".csv", ".tsv", ".json", ".md", ".txt", ".py", ".ipynb", ".yml", ".yaml", ".R", ".sh", ".gz"}:
            data = p.read_bytes()
            if p.suffix == ".gz":
                try:
                    data = gzip.decompress(data)
                except OSError:
                    pass
            if ORIGINAL_ID.search(data):
                errors.append(f"original specimen ID in {rel}")
            local_roots = (b"/" + b"storage/bernadettem/", b"/" + b"home/bernadettem/")
            if any(root in data for root in local_roots):
                errors.append(f"local absolute path in {rel}")
        if any(part in {"analysis", "review", "__pycache__"} or part.startswith("archive_legacy") or "pre_adduct" in part for part in p.relative_to(ROOT).parts):
            errors.append(f"historical or analysis-core file present: {rel}")
    if errors:
        print("VALIDATION FAILED")
        for error in errors[:100]:
            print("-", error)
        if len(errors) > 100:
            print(f"... {len(errors)-100} additional errors")
        return 1
    print("VALIDATION PASSED: portal-aligned Figures 2–7 and S1–S6; public files checksummed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
