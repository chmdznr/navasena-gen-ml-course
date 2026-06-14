#!/usr/bin/env python3
"""Gate for the reworked Module 05 RAG cheat sheet (HTML + single-page A4 PDF, 8 concept cards).
Mirrors 04_llm/tools/validate_cheatsheet.py."""
import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
html = (ROOT / "rag-fundamentals-cheatsheet.html").read_text(encoding="utf-8")
pdf = ROOT / "rag-fundamentals-cheatsheet.pdf"
errs = []

# Scope / framing updated to the 8-notebook concept arc.
if not re.search(r"[Dd]elapan notebook", html):
    errs.append("'Delapan notebook' scope framing missing")
if "(8 Notebook)" not in html:
    errs.append("Quick-Ref table not labelled '(8 Notebook)'")

# Eight numbered concept cards present.
for marker in "①②③④⑤⑥⑦⑧":
    if marker not in html:
        errs.append(f"missing card marker {marker}")

# Stale stub content removed.
for stale in ["TinyLlama", "all-MiniLM-L6-v2", "faiss-gpu", "Module 06", "Mod 06", "k=1"]:
    if stale in html:
        errs.append(f"stale stub content present: {stale!r}")

# Reworked-concept coverage.
low = html.lower()
for kw in ["docling", "bge-reranker", "ragas", "faithfulness", "faiss", "cited_labels", "fastapi", "rewriting", "nim"]:
    if kw not in low:
        errs.append(f"coverage gap: '{kw}' missing")

if not pdf.exists():
    errs.append("cheatsheet PDF not generated")

if errs:
    print("CHEATSHEET VALIDATION FAILED:"); [print("  -", e) for e in errs]; sys.exit(1)
print("CHEATSHEET OK: delapan-notebook framing, cards ①–⑧, RAG stack covered, PDF present.")
