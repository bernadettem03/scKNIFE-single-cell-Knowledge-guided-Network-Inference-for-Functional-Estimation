#!/usr/bin/env python3
"""Validate the review-ready figure-centric manuscript repository."""
from __future__ import annotations
import ast,hashlib,json,re,sys
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parent
EXPECTED=[*[ROOT/f"Figure_{n}"/f"Figure_{n}.ipynb" for n in range(2,8)],*[ROOT/"Supplementary"/f"Figure_S{n}"/f"Figure_S{n}.ipynb" for n in range(2,7)]]
FORBIDDEN_CALLS={"pearsonr","spearmanr","mannwhitneyu","multipletests","permutation_test","consensus_sweep","fit","fit_transform"}
MAIN_PANEL_SETS={2:list("ABCDEFGH"),3:list("ABCDEFGHIJ"),4:list("BCDEF"),5:list("ACEGH"),6:list("ABCDEF"),7:list("ABCDEFGHIJ")}
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def calls(source):
 source="\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
 try:tree=ast.parse(source)
 except SyntaxError:return {"<syntax-error>"}
 names=set()
 for node in ast.walk(tree):
  if isinstance(node,ast.Call):
   fn=node.func;names.add(fn.id if isinstance(fn,ast.Name) else (fn.attr if isinstance(fn,ast.Attribute) else ""))
 return names
def main():
 errors=[]
 for p in EXPECTED:
  if not p.is_file():errors.append(f"missing notebook: {p.relative_to(ROOT)}");continue
  nb=nbformat.read(p,4);code="\n".join(c.source for c in nb.cells if c.cell_type=="code")
  outputs=[o for c in nb.cells if c.cell_type=="code" for o in c.get("outputs",[])]
  displays=[o for o in outputs if o.output_type in {"display_data","execute_result"} and any(k.startswith("image/") for k in o.get("data",{}))]
  if not displays:errors.append(f"no inline generated figure output: {p.relative_to(ROOT)}")
  if any(o.output_type=="error" for o in outputs):errors.append(f"stored execution error: {p.relative_to(ROOT)}")
  if "approved_preview" in code or "reproduced_preview" in code:errors.append(f"preview wrapper in {p.relative_to(ROOT)}")
  if "/storage/" in p.read_text():errors.append(f"absolute storage path in notebook: {p.relative_to(ROOT)}")
  bad=calls(code)&FORBIDDEN_CALLS
  if bad:errors.append(f"analysis calls {sorted(bad)} in {p.relative_to(ROOT)}")
 for p in [*ROOT.glob("Figure_*/plotting.py"),*ROOT.glob("Supplementary/Figure_*/plotting.py")]:
  bad=calls(p.read_text())&FORBIDDEN_CALLS
  if bad:errors.append(f"analysis calls {sorted(bad)} in {p.relative_to(ROOT)}")
 for number,labels in MAIN_PANEL_SETS.items():
  panel_dir=ROOT/f"Figure_{number}"/"outputs"/"panels"
  manifest_path=panel_dir/f"Figure_{number}_panel_export_manifest.json"
  if not manifest_path.is_file():errors.append(f"missing panel manifest: {manifest_path.relative_to(ROOT)}");continue
  manifest=json.loads(manifest_path.read_text())
  if manifest.get("panel_order")!=labels:errors.append(f"incorrect panel order: {manifest_path.relative_to(ROOT)}")
  expected={f"Figure_{number}_{label}.{ext}" for label in labels for ext in ("png","pdf")}
  actual={q.name for q in panel_dir.iterdir() if q.suffix in {".png",".pdf"} and q.name!= "Figure_3_A_D.png" and q.name!= "Figure_3_A_D.pdf"}
  for name in sorted(expected-actual):errors.append(f"missing panel export: Figure_{number}/outputs/panels/{name}")
  for name in sorted(actual-expected):errors.append(f"unexpected panel export: Figure_{number}/outputs/panels/{name}")
  for label in labels:
   for ext in ("png","pdf"):
    item=manifest.get("panels",{}).get(label,{}).get(ext,{})
    q=ROOT/item.get("path","")
    if not q.is_file():errors.append(f"manifest path missing: Figure {number} panel {label} {ext}");continue
    if q.stat().st_size!=item.get("bytes") or sha(q)!=item.get("sha256"):errors.append(f"panel manifest mismatch: {q.relative_to(ROOT)}")
    if ext=="png" and q.read_bytes()[:8]!=b"\x89PNG\r\n\x1a\n":errors.append(f"invalid PNG: {q.relative_to(ROOT)}")
    if ext=="pdf" and q.read_bytes()[:5]!=b"%PDF-":errors.append(f"invalid PDF: {q.relative_to(ROOT)}")
 with (ROOT/"checksums.sha256").open() as f:
  for line in f:
   digest,rel=line.rstrip().split("  ",1);p=ROOT/rel
   if not p.is_file() or sha(p)!=digest:errors.append(f"checksum mismatch: {rel}")
 for p in ROOT.rglob("*"):
  if p.is_file() and ".git" not in p.parts and p.stat().st_size>=100_000_000:errors.append(f"GitHub-size file: {p.relative_to(ROOT)}")
 if errors:print("VALIDATION FAILED\n"+"\n".join(f"- {e}" for e in errors));return 1
 print(f"VALIDATION PASSED: {len(EXPECTED)} executed live plotting notebooks")
 return 0
if __name__=="__main__":sys.exit(main())
