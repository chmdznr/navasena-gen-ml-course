#!/usr/bin/env python3
"""Gate for the reworked Module 04 slides. Checks SOURCE invariants + that the PDF built."""
import re, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
TEX = ROOT / "slides" / "module04_slides.tex"
PDF = ROOT / "slides" / "module04_slides.pdf"
LOG = ROOT / "slides" / "module04_slides.log"
src = TEX.read_text(encoding="utf-8")
errs = []

# 1. No Python/bash code blocks remain.
n_code = len(re.findall(r"\\begin\{lstlisting\}", src))
if n_code: errs.append(f"{n_code} lstlisting block(s) still present (want 0)")

# 2. Framing updated 5 -> 8 notebooks.
if re.search(r"5\s+[Nn]otebook", src): errs.append("'5 notebook' framing still present")
if "8 notebook" not in src.lower(): errs.append("'8 notebook' framing missing")

# 3. New acts present, summary renumbered to 10.
for needle in [r"\\acttitle\{8\}\{SPESIALISASI", r"\\acttitle\{9\}\{DEPLOY", r"\\acttitle\{10\}"]:
    if not re.search(needle, src): errs.append(f"missing act marker: {needle}")

# 4. transformers pin tightened.
if ">=4.53,<5" not in src and "4.53" not in src: errs.append("transformers pin not tightened to >=4.53,<5")
if "4.44" in src: errs.append("stale transformers pin 4.44 still present")

# 5. Dead command removed.
if "\\newcommand{\\sectiontitle}" in src: errs.append("dead \\sectiontitle command not removed")

# 6. PDF built + page count from log.
if not PDF.exists(): errs.append("module04_slides.pdf not built")
pages = None
if LOG.exists():
    m = re.search(r"Output written on .*\((\d+) pages?", LOG.read_text(errors="ignore"))
    if m: pages = int(m.group(1))
if pages is None: errs.append("could not read page count from .log (build it first)")
elif pages < 44: errs.append(f"only {pages} pages (want >= 44)")

if errs:
    print("SLIDES VALIDATION FAILED:"); [print("  -", e) for e in errs]; sys.exit(1)
print(f"SLIDES OK: 0 code blocks, {pages} pages, acts 8/9/10 present, 8-notebook framing.")
