#!/usr/bin/env python3
"""Gate for the reworked Module 05 RAG slides. Checks SOURCE invariants + that the PDF built.
Mirrors 04_llm/tools/validate_slides.py. Exit 0 = pass, 1 = fail."""
import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEX = ROOT / "slides" / "module05_slides.tex"
PDF = ROOT / "slides" / "module05_slides.pdf"
LOG = ROOT / "slides" / "module05_slides.log"

src = TEX.read_text(encoding="utf-8")
errs = []

# 1. Code-free deck — no lstlisting blocks remain.
n_code = len(re.findall(r"\\begin\{lstlisting\}", src))
if n_code:
    errs.append(f"{n_code} lstlisting block(s) present (want 0 — slides are concept/visualization only)")

# 2. Stale stub framing removed (old 16-frame TinyLlama deck).
for stale in ["TinyLlama", "all-MiniLM-L6-v2", "Module 06", "Module~06", "faiss-gpu"]:
    if stale in src:
        errs.append(f"stale stub framing still present: {stale!r}")

# 3. Reworked-concept markers present (the 8-notebook arc, concept-driven).
markers = {
    "Docling": "Docling",
    "HybridChunker": "HybridChunker",
    "reranker": "bge-reranker-v2-m3",
    "RAGAS": "RAGAS",
    "NIM": "NIM",
    "multilingual embed": "multilingual",
    "FastAPI deploy": "FastAPI",
    "/ask endpoint": "/ask",
    "citations": "[S",
    "conversational": "rewriting",
    "FAISS scale": "HNSW",
}
for label, needle in markers.items():
    if needle not in src:
        errs.append(f"missing concept marker [{label}]: {needle!r}")

# 4. Ten concept acts present.
for n in range(1, 11):
    if not re.search(rf"\\acttitle\{{{n}\}}", src):
        errs.append(f"missing act divider: \\acttitle{{{n}}}")

# 5. PDF built + page count from log.
if not PDF.exists():
    errs.append("module05_slides.pdf not built")
pages = None
if LOG.exists():
    m = re.search(r"Output written on .*\((\d+) pages?", LOG.read_text(errors="ignore"))
    if m:
        pages = int(m.group(1))
if pages is None:
    errs.append("could not read page count from .log (build the deck first)")
elif pages < 36:
    errs.append(f"only {pages} pages (want >= 36 for a full concept deck)")

if errs:
    print("SLIDES VALIDATION FAILED:")
    [print("  -", e) for e in errs]
    sys.exit(1)
print(f"SLIDES OK: 0 code blocks, {pages} pages, 10 concept acts, reworked-RAG framing.")
