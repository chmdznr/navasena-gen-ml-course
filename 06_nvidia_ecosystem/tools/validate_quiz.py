#!/usr/bin/env python3
"""Gate for the reworked Module 06 (NVIDIA Ecosystem) quiz -- 5-notebook v2 arc.
Parses the inline `const QUIZ = {...};` JSON from nvidia-ecosystem-quiz.html and checks invariants.
Mirrors 05_rag/tools/validate_quiz.py. Exit 0 = pass, 1 = fail.
"""
import re
import json
import sys
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML_PATH = ROOT / "nvidia-ecosystem-quiz.html"


def _extract_quiz_object(html: str) -> str | None:
    """Return the `const QUIZ = {...}` object literal via brace-balanced scan.

    A non-greedy `\\{.*?\\}` regex truncates at the first '};' substring, which can
    sit *inside* a question/explanation string -- silently validating a partial quiz.
    Scan from the first '{' after the anchor and balance braces instead so the whole
    object is always captured.
    """
    m = re.search(r"const\s+QUIZ\s*=\s*", html)
    if not m:
        return None
    i = html.find("{", m.end())
    if i == -1:
        return None
    depth = 0
    in_str = False   # inside a double-quoted JSON string
    esc = False      # previous char was a backslash escape
    for j in range(i, len(html)):
        c = html[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return html[i : j + 1]
    return None  # unbalanced braces


def main() -> int:
    if not HTML_PATH.exists():
        print(f"QUIZ VALIDATION FAILED:\n  - {HTML_PATH} not found")
        return 1
    html = HTML_PATH.read_text(encoding="utf-8")
    errs: list[str] = []

    payload = _extract_quiz_object(html)
    if payload is None:
        print("QUIZ VALIDATION FAILED:\n  - QUIZ payload not found (expected `const QUIZ = {...};`)")
        return 1
    try:
        quiz = json.loads(payload)
    except json.JSONDecodeError as e:
        print(f"QUIZ VALIDATION FAILED:\n  - QUIZ payload is not valid JSON -- {e}")
        return 1

    # Envelope: {"module":"06","title":"NVIDIA Ecosystem","questions":[...]}.
    if quiz.get("module") != "06":
        errs.append(f"envelope module is {quiz.get('module')!r} (want '06')")
    if quiz.get("title") != "NVIDIA Ecosystem":
        errs.append(f"envelope title is {quiz.get('title')!r} (want 'NVIDIA Ecosystem')")

    qs = quiz.get("questions", [])
    n = len(qs)
    if not (28 <= n <= 34):
        errs.append(f"{n} questions (want 28-34)")

    for i, q in enumerate(qs, 1):
        if len(q.get("options", [])) != 4:
            errs.append(f"Q{i}: not 4 options")
        if not isinstance(q.get("answer"), int) or not (0 <= q["answer"] <= 3):
            errs.append(f"Q{i}: bad answer index (want int 0..3)")
        if not str(q.get("explanation", "")).strip():
            errs.append(f"Q{i}: empty explanation")
        # Pure-concept quiz: every question's code must be null/empty.
        if q.get("code") not in (None, "", "null"):
            errs.append(f"Q{i}: has code (must be pure-concept, code=null)")
        # No stray category/notebook keys on the question objects.
        for stray in ("category", "notebook", "nb"):
            if stray in q:
                errs.append(f"Q{i}: stray key {stray!r} present (use comment dividers, not JSON keys)")

    # Header count string must match question count AND use the 'murni konsep' wording.
    mh = re.search(r"<p>(\d+)\s+soal", html)
    if not mh:
        errs.append("header 'N soal' string not found")
    elif int(mh.group(1)) != n:
        errs.append(f"header says {mh.group(1)} soal but {n} questions")
    if "murni konsep" not in html:
        errs.append("header phrase 'murni konsep' missing (want '<N> soal pilihan ganda · murni konsep')")
    if "Module 06" not in html:
        errs.append("'Module 06' module label missing from the page")

    # Answer distribution must not be degenerate: no single index may dominate.
    if qs:
        dist = Counter(q.get("answer") for q in qs)
        worst = max(dist.values())
        if worst > 12:
            errs.append(f"degenerate answer distribution {dict(dist)} -- index appears {worst}x (max ~12)")

    # Stale stub framing must be gone (deleted v1/old-companion content + cross-module leakage).
    blob = json.dumps(quiz, ensure_ascii=False).lower()
    for stale in ["phi-2", "gpt2", "nvidia-tensorrt", "ner_en_bert", "megatrongpt",
                  "tinyllama", "cudf", "module 05", "delapan notebook", "sembilan notebook",
                  "8 notebook", "9 notebook", "skalabel"]:
        if stale in blob:
            errs.append(f"stale stub content present: {stale!r}")

    # Coverage: the 5-notebook v2 arc must appear across the questions (case-insensitive).
    #   nb01 GPU/precision/quant, nb02 serving, nb03 fairness/governance,
    #   nb04 safety/privacy [module identity], nb05 capstone.
    coverage = [
        "nim", "quantization", "triton", "dynamo",
        "fairness", "equalized odds", "grounding", "pii",
        "model card", "fastapi", "capstone", "trustworthy",
        "uu pdp", "nemoguard", "content safety", "topic control",
    ]
    for kw in coverage:
        if kw not in blob:
            errs.append(f"coverage gap: '{kw}' not found in any question")

    if errs:
        print("QUIZ VALIDATION FAILED:")
        for e in errs:
            print("  -", e)
        return 1
    print(
        f"QUIZ OK: {n} pure-concept questions, envelope module 06 / NVIDIA Ecosystem, "
        f"header matches '{n} soal ... murni konsep', balanced answers, 5-notebook v2 coverage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
