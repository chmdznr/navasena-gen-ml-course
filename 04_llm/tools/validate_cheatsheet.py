#!/usr/bin/env python3
"""Gate for the reworked Module 04 cheat sheet (HTML + PDF cover 8 notebooks)."""
import re, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
html = (ROOT / "llm-fundamentals-cheatsheet.html").read_text(encoding="utf-8")
pdf  = ROOT / "llm-fundamentals-cheatsheet.pdf"
errs = []
if re.search(r"[Ll]ima notebook|\(5 Notebook\)", html): errs.append("stale 5-notebook framing still present")
if "Delapan notebook" not in html: errs.append("'Delapan notebook' scope missing")
if "(8 Notebook)" not in html: errs.append("Quick-Ref table not updated to 8 Notebook")
for needle in ["⑥", "⑦", "⑧"]:
    if needle not in html: errs.append(f"missing new card marker {needle}")
for kw in ["SLM", "specialist", "GGUF", "distillation", "merge"]:
    if kw.lower() not in html.lower(): errs.append(f"coverage gap: '{kw}' missing")
if "Qwen3B" in html: errs.append("stale 'Qwen3B' (should be Qwen2.5-3B)")
if not pdf.exists(): errs.append("cheatsheet PDF not generated")
if errs:
    print("CHEATSHEET VALIDATION FAILED:"); [print("  -", e) for e in errs]; sys.exit(1)
print("CHEATSHEET OK: 8-notebook framing, cards 6/7/8 present, SLM/deploy covered.")
