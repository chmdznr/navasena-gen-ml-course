#!/usr/bin/env python3
"""Gate for the reworked Module 02 DL quiz. Parses the inline QUIZ JSON and checks invariants.
Mirrors 04_llm/tools/validate_quiz.py."""
import re, json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bahasa_rules import check

ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML = (ROOT / "dl-fundamentals-quiz.html").read_text(encoding="utf-8")
errs = []

m = re.search(r"const QUIZ = (\{.*?\});", HTML, re.S)
if not m:
    print("QUIZ payload not found"); sys.exit(1)
quiz = json.loads(m.group(1))
qs = quiz["questions"]
n = len(qs)
if not (28 <= n <= 34):
    errs.append(f"{n} questions (want 28-34)")

for i, q in enumerate(qs, 1):
    if len(q.get("options", [])) != 4:
        errs.append(f"Q{i}: not 4 options")
    if not isinstance(q.get("answer"), int) or not (0 <= q["answer"] <= 3):
        errs.append(f"Q{i}: bad answer index")
    if not q.get("explanation", "").strip():
        errs.append(f"Q{i}: empty explanation")
    if q.get("code") not in (None, "", "null"):
        errs.append(f"Q{i}: has code (must be pure-concept, code=null)")

# Header count string must match question count.
mh = re.search(r"<p>(\d+)\s+soal", HTML)
if not mh:
    errs.append("header 'N soal' string not found")
elif int(mh.group(1)) != n:
    errs.append(f"header says {mh.group(1)} soal but {n} questions")

# Stale stub framing must be gone.
blob = json.dumps(quiz, ensure_ascii=False).lower()
for stale in ["tinyllama", "all-minilm-l6-v2", "module 06", "indexflatl2 mengukur jarak euclidean"]:
    if stale in blob:
        errs.append(f"stale stub content present: {stale!r}")

# Coverage: the 8-notebook concept arc must appear across the questions.
for kw in ["learning rate", "overfitting", "validation", "dropout", "softmax", "relu", "konvolusi", "transfer learning", "hidden state", "vanishing", "mixed precision", "tensorrt", "latent", "generator"]:
    if kw not in blob:
        errs.append(f"coverage gap: '{kw}' not found in any question")

# Aturan bahasa spec §4 ditegakkan pada SELURUH HTML (skeleton + payload), bukan
# hanya pada blob JSON: footer/header di luar payload pernah lolos karenanya.
errs += [f"bahasa: {e}" for e in check(re.sub(r"<[^>]+>", " ", HTML), limit=0)]

# Bias panjang: opsi benar tidak boleh sistematis lebih panjang/pendek dari pengecoh.
import statistics
cor = [len(q["options"][q["answer"]]) for q in qs]
inc = [len(o) for q in qs for i, o in enumerate(q["options"]) if i != q["answer"]]
ratio = statistics.mean(cor) / statistics.mean(inc)
if not 0.85 <= ratio <= 1.15:
    errs.append(f"length bias: mean correct/incorrect = {ratio:.2f} (want 0.85–1.15)")
# Per soal: opsi terpanjang / terpendek <= 1,2, dan opsi benar jarang jadi yang terpanjang.
n_longest = 0
for i, q in enumerate(qs, 1):
    lens = [len(o) for o in q["options"]]
    r = max(lens) / min(lens)
    if r > 1.2:
        errs.append(f"Q{i}: rasio panjang opsi {r:.2f} > 1.2")
    if lens[q["answer"]] == max(lens):
        n_longest += 1
if n_longest > 9:
    errs.append(f"opsi benar terpanjang di {n_longest} soal (maks 9)")

from collections import Counter
dist = Counter(q["answer"] for q in qs)
if max(dist.values()) > n // 2:
    errs.append(f"answer index skew: {dict(dist)}")

if errs:
    print("QUIZ VALIDATION FAILED:"); [print("  -", e) for e in errs]; sys.exit(1)
print(f"QUIZ OK: {n} pure-concept questions, header matches, M02 concept coverage.")
