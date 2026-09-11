#!/usr/bin/env python3
"""Restore checksummed private manuscript assets without publishing them."""
from __future__ import annotations
import argparse, csv, hashlib, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/"private_archive_manifest.tsv"
def sha256(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--archive-dir",type=Path,required=True,help="Extracted private archive root (the directory containing Figure_*/).")
 ap.add_argument("--verify-only",action="store_true")
 args=ap.parse_args(); failures=[]
 with MANIFEST.open(newline="") as f:
  for row in csv.DictReader(f,delimiter="\t"):
   src=args.archive_dir/row["archive_path"]; dst=ROOT/row["repository_path"]
   if not src.is_file() or sha256(src)!=row["sha256"]: failures.append(str(src)); continue
   if not args.verify_only:
    dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    if sha256(dst)!=row["sha256"]: failures.append(str(dst))
 if failures: raise SystemExit("Missing or invalid assets:\n"+"\n".join(failures))
 print("Private assets verified" if args.verify_only else "Private assets restored and verified")
if __name__=="__main__": main()
