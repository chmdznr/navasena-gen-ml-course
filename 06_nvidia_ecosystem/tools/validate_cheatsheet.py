#!/usr/bin/env python3
"""Gate for the reworked Module 06 (NVIDIA Ecosystem) cheat sheet -- 5-notebook v2 arc.
Checks nvidia-ecosystem-cheatsheet.html (+ the single-page A4 PDF beside it).
Five concept cards (①–⑤) map 1:1 to the five notebooks nb01-nb05.
Mirrors 05_rag/tools/validate_cheatsheet.py. Exit 0 = pass, 1 = fail.
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML_PATH = ROOT / "nvidia-ecosystem-cheatsheet.html"
PDF_PATH = ROOT / "nvidia-ecosystem-cheatsheet.pdf"


def main() -> int:
    if not HTML_PATH.exists():
        print(f"CHEATSHEET VALIDATION FAILED:\n  - {HTML_PATH} not found")
        return 1
    html = HTML_PATH.read_text(encoding="utf-8")
    low = html.lower()
    errs: list[str] = []

    # 1. Scope / framing updated to the 5-notebook arc (nb01-nb05).
    if "lima notebook" not in low:
        errs.append("'lima notebook' scope framing missing (5 notebooks nb01-05)")
    if "(5 Notebook)" not in html:
        errs.append("Quick-Ref table not labelled '(5 Notebook)'")
    if "Module 06" not in html:
        errs.append("'Module 06' module label missing")

    # 2. Five numbered concept cards present (① per notebook nb01..nb05).
    for marker in "①②③④⑤":
        if marker not in html:
            errs.append(f"missing card glyph {marker}")
    # 5-notebook arc: cards ①–⑤ required; an optional ⑥ is allowed (nb04 safety+guardrails+
    # privacy+grounding is dense enough to split). A 7th+ glyph signals leftover from the old arc.
    for stale_glyph in "⑦⑧⑨":
        if stale_glyph in html:
            errs.append(f"stale card glyph {stale_glyph} present (5-notebook arc -> ①–⑤, optional ⑥)")

    # 3. Stale stub content removed (deleted v1/old-companion + cross-module leakage).
    #    Note: NIM / TensorRT / NeMo are LEGIT M06 topics -- never blanket-forbid them.
    #    Matched case-INSENSITIVELY (against `low`): a stale token in ANY casing is still
    #    stale -- e.g. 'tinyllama', 'TinyLlama', 'module 05', 'Module 05' must all be caught.
    for stale in ["phi-2", "gpt2", "nvidia-tensorrt", "ner_en_bert", "megatrongpt",
                  "tinyllama", "cudf", "module 05", "mod 05",
                  "docling", "ragas", "bge-reranker", "skalabel",
                  "delapan notebook", "sembilan notebook", "8 notebook", "9 notebook"]:
        if stale in low:
            errs.append(f"stale stub content present: {stale!r}")

    # 4. Reworked-concept coverage (front NVIDIA stack + back Trustworthy-AI/guardrails).
    for kw in ["nim", "quantization", "nemoguard", "fairness", "grounding", "pii", "fastapi",
               "nemotron", "tensor core", "model card", "uu pdp", "capstone", "trustworthy"]:
        if kw not in low:
            errs.append(f"coverage gap: '{kw}' missing")

    # 5. The single-page A4 PDF must sit beside the HTML.
    if not PDF_PATH.exists():
        errs.append("cheatsheet PDF not generated (expected nvidia-ecosystem-cheatsheet.pdf next to the HTML)")

    if errs:
        print("CHEATSHEET VALIDATION FAILED:")
        for e in errs:
            print("  -", e)
        return 1
    print(
        "CHEATSHEET OK: lima-notebook framing, cards ①–⑤ (5 Notebook), "
        "NVIDIA + Trustworthy-AI stack covered, PDF present."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
