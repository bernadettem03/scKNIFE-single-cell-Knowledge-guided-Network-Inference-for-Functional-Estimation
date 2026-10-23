#!/usr/bin/env python3
"""Refresh local export hashes and the public package inventories."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return any(part == ".git" or part == "__pycache__" or part == "analysis" or part == "review" or part.startswith("archive_legacy") or "pre_adduct" in part for part in rel.parts)


def resolve(manifest: Path, value: str) -> Path | None:
    if Path(value).is_absolute():
        return None
    for base in (manifest.parent, ROOT, manifest.parent.parent, manifest.parent.parent.parent):
        candidate = (base / value).resolve()
        if candidate.is_file() and candidate != manifest and candidate.is_relative_to(ROOT) and not excluded(candidate):
            return candidate
    return None


def refresh_json_manifests() -> None:
    for manifest in sorted(ROOT.rglob("*.json")):
        if excluded(manifest) or "manifest" not in manifest.name.lower():
            continue
        obj = json.loads(manifest.read_text())
        changes = [0]
        def walk(value, key=None):
            if isinstance(value, dict):
                candidate = value.get("path", key)
                if isinstance(candidate, str) and isinstance(value.get("sha256"), str):
                    p = resolve(manifest, candidate)
                    if p:
                        current = sha(p)
                        if value["sha256"] != current:
                            value["sha256"] = current
                            changes[0] += 1
                        if "bytes" in value:
                            value["bytes"] = p.stat().st_size
                if isinstance(value.get("sha256"), dict):
                    for rel, old in value["sha256"].items():
                        p = resolve(manifest, rel)
                        if p and old != sha(p):
                            value["sha256"][rel] = sha(p)
                            changes[0] += 1
                for child_key, child in value.items():
                    walk(child, child_key)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(obj)
        if changes[0]:
            manifest.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    refresh_json_manifests()
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and not excluded(p))
    with (ROOT / "data_manifest.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("path", "bytes", "sha256", "storage"))
        for p in files:
            if "data" in p.relative_to(ROOT).parts or (
                p.relative_to(ROOT).parts[0] == "Tables" and p.suffix in {".xlsx", ".tsv"}
            ):
                writer.writerow((p.relative_to(ROOT).as_posix(), p.stat().st_size, sha(p), "git"))
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and not excluded(p) and p.name != "checksums.sha256")
    (ROOT / "checksums.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in files))
    print(f"Inventoried {len(files)} public files")

if __name__ == "__main__":
    main()
