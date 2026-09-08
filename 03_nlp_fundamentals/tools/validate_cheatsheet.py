#!/usr/bin/env python3
"""Gate cheat sheet Modul 03 (HTML + PDF A4 satu halaman). Kartu ditandai judul <h3>, bukan simbol berlingkar."""
import re, subprocess, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bahasa_rules import check
from freshness import tertinggal

ROOT = pathlib.Path(__file__).resolve().parent.parent
html = (ROOT / "nlp-fundamentals-cheatsheet.html").read_text(encoding="utf-8")
pdf = ROOT / "nlp-fundamentals-cheatsheet.pdf"
errs = []

n_cards = html.count('class="card')
if n_cards < 12:
    errs.append(f"hanya {n_cards} kartu (mau >= 12)")

low = html.lower()
for kw in ["stopword", "lemmatization", "pos tagging", "ner", "sentiment", "naive bayes",
           "bag-of-words", "kebocoran data", "baseline", "tf-idf", "embedding",
           "cosine similarity", "subword", "indobert", "cudf", "cuml", "minhash", "warm-up"]:
    if kw not in low:
        errs.append(f"coverage gap: '{kw}'")

for stale in ["Spam vs Ham", "spam vs ham", "Module 0", "_id.ipynb", "_en.ipynb", "10-100x", "10–100x"]:
    if stale in html:
        errs.append(f"stale wording: {stale!r}")
errs += check(re.sub(r"<[^>]+>", " ", html), limit=0)

if (e := tertinggal(pdf, ROOT / "nlp-fundamentals-cheatsheet.html")):
    errs.append(e + " (Chrome headless --print-to-pdf)")
if pdf.exists():
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    if not m or int(m.group(1)) != 1:
        errs.append(f"PDF bukan 1 halaman: {m.group(1) if m else '?'}")

if errs:
    print("CHEATSHEET VALIDATION FAILED:"); [print("  -", e) for e in errs]; sys.exit(1)
print(f"CHEATSHEET OK: {n_cards} kartu, istilah spec tercakup, PDF 1 halaman.")
