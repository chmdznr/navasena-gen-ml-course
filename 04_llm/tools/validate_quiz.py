#!/usr/bin/env python3
"""Gate for the reworked Module 04 quiz. Parses the inline QUIZ JSON and checks invariants."""
import re, json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML = (ROOT / "llm-fundamentals-quiz.html").read_text(encoding="utf-8")
errs = []

m = re.search(r"const QUIZ = (\{.*?\});", HTML, re.S)
if not m:
    print("QUIZ payload not found"); sys.exit(1)
quiz = json.loads(m.group(1))
qs = quiz["questions"]
n = len(qs)
if not (28 <= n <= 32): errs.append(f"{n} questions (want 28-32)")

for i, q in enumerate(qs, 1):
    if len(q.get("options", [])) != 4: errs.append(f"Q{i}: not 4 options")
    if not isinstance(q.get("answer"), int) or not (0 <= q["answer"] <= 3): errs.append(f"Q{i}: bad answer index")
    if not q.get("explanation", "").strip(): errs.append(f"Q{i}: empty explanation")
    if q.get("code") not in (None, "", "null"): errs.append(f"Q{i}: has code (must be pure-concept, code=null)")

# Header count string must match question count.
mh = re.search(r"<p>(\d+)\s+soal", HTML)
if not mh: errs.append("header 'N soal' string not found")
elif int(mh.group(1)) != n: errs.append(f"header says {mh.group(1)} soal but {n} questions")

# Coverage sanity: 05-07 topics present somewhere in the payload.
blob = json.dumps(quiz, ensure_ascii=False).lower()
for kw in ["specialist", "generalist", "gguf", "distillation", "merge"]:
    if kw not in blob: errs.append(f"coverage gap: '{kw}' not found in any question")

if errs:
    print("QUIZ VALIDATION FAILED:"); [print("  -", e) for e in errs]; sys.exit(1)
print(f"QUIZ OK: {n} pure-concept questions, header matches, SLM/deploy covered.")
