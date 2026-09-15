#!/usr/bin/env python3
"""Validate the review-ready figure-centric manuscript repository."""
from __future__ import annotations
import ast,hashlib,json,re,sys
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parent
EXPECTED=[*[ROOT/f"Figure_{n}"/f"Figure_{n}.ipynb" for n in range(2,8)],*[ROOT/"Supplementary"/f"Figure_S{n}"/f"Figure_S{n}.ipynb" for n in range(1,7)]]
FORBIDDEN_CALLS={"pearsonr","spearmanr","mannwhitneyu","multipletests","permutation_test","consensus_sweep","fit","fit_transform"}
MAIN_PANEL_SETS={2:list("ABCDEFGH"),3:list("ABCDEFGHIJ"),4:list("BCDEF"),5:list("ACEGH"),6:list("ABCDEFGH"),7:list("ABCDEFGHIJ")}
REQUIRED_AUDITS=(2,3,4,5,7)
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
 for number in range(2,8):
  plotting=ROOT/f"Figure_{number}"/"plotting.py"
  if not plotting.is_file():errors.append(f"missing plotting module: {plotting.relative_to(ROOT)}")
 for number in range(1,7):
  plotting=ROOT/"Supplementary"/f"Figure_S{number}"/"plotting.py"
  if not plotting.is_file():errors.append(f"missing supplementary plotting module: {plotting.relative_to(ROOT)}")
 for number in REQUIRED_AUDITS:
  audit=ROOT/f"Figure_{number}"/"NOTEBOOK_AUDIT.md"
  if not audit.is_file():errors.append(f"missing notebook audit: {audit.relative_to(ROOT)}")
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
  module_text=p.read_text()
  bad=calls(module_text)&FORBIDDEN_CALLS
  if bad:errors.append(f"analysis calls {sorted(bad)} in {p.relative_to(ROOT)}")
  if "/storage/" in module_text:errors.append(f"absolute storage path in plotting module: {p.relative_to(ROOT)}")
 s4_manifest_path=ROOT/"Supplementary/Figure_S4/outputs/Figure_S4_manifest.json"
 if not s4_manifest_path.is_file():errors.append("missing Supplementary Figure S4 output manifest")
 else:
  s4_manifest=json.loads(s4_manifest_path.read_text())
  if s4_manifest.get("panel_order")!=list("TRSQNPM"):errors.append("incorrect Supplementary Figure S4 panel order")
  for section in ("individual_outputs","compiled_outputs"):
   def check_paths(value):
    if isinstance(value,dict):
     for nested in value.values():check_paths(nested)
    elif isinstance(value,str) and not (ROOT/"Supplementary/Figure_S4"/value).is_file():errors.append(f"missing S4 output: {value}")
   check_paths(s4_manifest.get(section,{}))
 s1=ROOT/"Supplementary/Figure_S1"
 s2=ROOT/"Supplementary/Figure_S2"
 if not (s1/"data/pooled_w").is_dir():errors.append("S1 pooled-W data directory is missing")
 if (s2/"data/pooled_w").exists():errors.append("pooled-W content remains assigned to S2")
 if not (s2/"data/no_match_umaps").is_dir():errors.append("S2 no-match UMAP directory is missing")
 s1_text=(s1/"plotting.py").read_text() if (s1/"plotting.py").is_file() else ""
 s2_text=(s2/"plotting.py").read_text() if (s2/"plotting.py").is_file() else ""
 for title in ("Cross-dataset W concordance","t-SNE node kinds across independently normalized W blocks","t-SNE pooled clusters across independently normalized W blocks"):
  if title not in s1_text:errors.append(f"S1 panel title missing: {title}")
 if "POOLED_ROOT" in s2_text or "concordance" in s2_text.lower():errors.append("S2 plotting module still contains pooled-W rendering")
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
