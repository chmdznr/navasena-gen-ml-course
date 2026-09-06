#!/usr/bin/env python3
"""Gate cheat sheet Modul 01 (HTML + PDF A4 satu halaman, 10 kartu). Mirrors 05_rag/tools/validate_cheatsheet.py."""
import re, subprocess, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bahasa_rules import check

ROOT = pathlib.Path(__file__).resolve().parent.parent
html = (ROOT / "ml-fundamentals-cheatsheet.html").read_text(encoding="utf-8")
pdf = ROOT / "ml-fundamentals-cheatsheet.pdf"
errs = []

for marker in "①②③④⑤⑥⑦⑧⑨⑩":
    if marker not in html:
        errs.append(f"kartu {marker} hilang")

low = html.lower()
for kw in ["leakage", "baseline", "cross-validation", "interval prediksi", "bootstrap", "silhouette",
           "early stopping", "class_weight", "log1p", "warm-up", "sejajar sumbu feature"]:
    if kw not in low:
        errs.append(f"coverage gap: '{kw}'")

for stale in ["titik awal", "10-50x", "10–50x", "Turbo", "kotak-kotak", "Berliku"]:
    if stale in html:
        errs.append(f"stale wording: {stale!r}")
errs += check(re.sub(r"<[^>]+>", " ", html), limit=0)

if not pdf.exists():
    errs.append("PDF belum dibuat")
else:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    if not m or int(m.group(1)) != 1:
        errs.append(f"PDF bukan 1 halaman: {m.group(1) if m else '?'}")

if errs:
    print("CHEATSHEET VALIDATION FAILED:"); [print("  -", e) for e in errs]; sys.exit(1)
print("CHEATSHEET OK: 10 kartu, istilah spec tercakup, PDF 1 halaman.")
