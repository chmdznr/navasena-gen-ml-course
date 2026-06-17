#!/usr/bin/env python3
"""Gate for the reworked Module 06 (NVIDIA Ecosystem) slides -- 5-notebook v2 arc.
Checks SOURCE invariants on slides/module06_slides.tex + that the PDF built.
Mirrors 05_rag/tools/validate_slides.py. Exit 0 = pass, 1 = fail.
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEX = ROOT / "slides" / "module06_slides.tex"
PDF = ROOT / "slides" / "module06_slides.pdf"
LOG = ROOT / "slides" / "module06_slides.log"


def main() -> int:
    if not TEX.exists():
        print(f"SLIDES VALIDATION FAILED:\n  - {TEX} not found")
        return 1
    src = TEX.read_text(encoding="utf-8")
    low = src.lower()
    errs: list[str] = []

    # 1. Code-free deck -- no lstlisting / verbatim blocks (M04/M05 standard).
    n_code = len(re.findall(r"\\begin\{lstlisting\}", src)) + len(re.findall(r"\\begin\{verbatim\}", src))
    if n_code:
        errs.append(f"{n_code} code block(s) present (lstlisting/verbatim) -- want 0 (concept/visualization only)")

    # 2. Footline bold module label must be the M06 label.
    if r"\textbf{NVIDIA Ecosystem}" not in src:
        errs.append(r"footline bold label '\textbf{NVIDIA Ecosystem}' not found")

    # 3. Concept act dividers -- the 5-notebook v2 deck has at least 6 acts.
    acts = sorted(int(m) for m in re.findall(r"\\acttitle\{(\d+)\}", src))
    if len(acts) < 6:
        errs.append(f"only {len(acts)} \\acttitle act divider(s) (want >= 6)")

    # 4. v2 concept markers (case-insensitive). 'nim' uses a word boundary so 'minim'/'minimal'
    #    don't trigger false positives; 'nemoguard' accepts the spaced 'NeMo Guard' spelling.
    def has_word(needle: str) -> bool:
        return re.search(rf"\b{re.escape(needle)}\b", low) is not None

    markers = {
        "Nemotron generator": lambda: "nemotron" in low,
        "NVIDIA NIM": lambda: has_word("nim"),
        "Triton (serving baseline)": lambda: "triton" in low,
        "NemoGuard rails": lambda: ("nemoguard" in low) or ("nemo guard" in low),
        "fairness": lambda: "fairness" in low,
        "grounding": lambda: "grounding" in low,
    }
    for label, fn in markers.items():
        if not fn():
            errs.append(f"missing concept marker [{label}]")

    # 5. Stale stub framing removed. (TensorRT / NeMo / Triton are LEGIT M06 topics -- never blanket-forbid.)
    for stale in ["phi-2", "gpt2", "nvidia-tensorrt", "ner_en_bert", "megatrongpt",
                  "tinyllama", "cudf", "module 05", "module~05", "mod 05",
                  "delapan notebook", "sembilan notebook", "8 notebook", "9 notebook", "skalabel"]:
        if stale in low:
            errs.append(f"stale stub framing still present: {stale!r}")

    # 6. PDF built + page count from log.
    if not PDF.exists():
        errs.append("module06_slides.pdf not built")
    pages = None
    if LOG.exists():
        m = re.search(r"Output written on .*\((\d+) pages?", LOG.read_text(errors="ignore"))
        if m:
            pages = int(m.group(1))
    if pages is None:
        errs.append("could not read page count from .log (build the deck first)")
    elif pages < 30:
        errs.append(f"only {pages} pages (want >= 30 for a full concept deck)")

    if errs:
        print("SLIDES VALIDATION FAILED:")
        for e in errs:
            print("  -", e)
        return 1
    print(
        f"SLIDES OK: 0 code blocks, {pages} pages, {len(acts)} concept acts, "
        f"NVIDIA Ecosystem footline, v2 markers (nemotron/nim/triton/nemoguard/fairness/grounding)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
