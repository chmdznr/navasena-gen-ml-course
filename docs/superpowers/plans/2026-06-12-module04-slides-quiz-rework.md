# Module 04 Slides + Quiz Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the Module 04 Beamer deck (`module04_slides.tex`) to be code-free and illustration-driven across all 8 notebooks (00–07), and rework the quiz (`llm-fundamentals-quiz.html`) into ~30 pure-concept MCQ covering all 8 notebooks.

**Architecture:** Two independent deliverables sharing one spec. (1) Slides: surgically replace each of the 10 Python `lstlisting` blocks with a TikZ/matplotlib illustration, append two new acts (SPESIALISASI nb05+06, DEPLOY nb07), renumber the summary act to 10, and fix the "5 notebook" framing. (2) Quiz: replace the inline `QUIZ` JSON payload with a new pure-concept set and update hardcoded counts. Two Python validator scripts act as the "tests" — written first (they fail on the current files), then made to pass.

**Tech Stack:** XeLaTeX + Beamer (custom "Navasena Dark" theme), TikZ, pgfplots, matplotlib→PDF figures, single-file HTML+JS quiz. Build via `04_llm/slides/build.sh`. Figure/validator python: `/Users/chmdznr/work/kemendag/sip/docling/bin/python` (matplotlib 3.10.8).

**Language constraint (HARD, applies to every Indonesian string written):** *"Bahasa Indonesia untuk Engineering yang luwes/fleksibel — istilah serapan Inggris yang umum dipakai tidak diterjemahkan; frasa/kalimat tidak ambigu (tidak aneh didengar)."* Keep untranslated: deploy, fine-tune, adapter, quantize, embedding, attention, token, prompt, system prompt, ChatML, GGUF, merge, batch, padding, streaming, runtime, throughput, generalist, specialist, dst.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `04_llm/tools/validate_slides.py` | Gate: slides source has 0 Python/bash code blocks, "8 notebook" present, acts 8/9/10 present, pin `>=4.53,<5`, PDF builds, page count ≥ 44 | Create |
| `04_llm/tools/validate_quiz.py` | Gate: quiz `QUIZ` JSON valid, ~30 questions, every Q has 4 options + answer 0–3 + non-empty explanation + `code:null`, header count matches, SLM/deploy keywords present | Create |
| `04_llm/slides/figures/gen_nb00_loss.py` | matplotlib: TinyGPT loss curve 3.68→0.06 → `nb00_loss.pdf` | Create |
| `04_llm/slides/figures/gen_slm_charts.py` | matplotlib: `slm_accuracy.pdf` (grouped bar 78/72/95), `slm_quadrant.pdf` (VRAM×accuracy scatter), `slm_gallery.pdf` (base→specialist 57→100/0→100/12→100) | Create |
| `04_llm/slides/module04_slides.tex` | The deck — remove 10 code blocks, add illustrations, 2 new acts, renumber, fix framing | Modify |
| `04_llm/llm-fundamentals-quiz.html` | The quiz — replace `QUIZ` payload (line 61), update count string (line 52) | Modify |

**Build/verify commands (used throughout):**
```bash
# Build slides (figures + xelatex x2). Use docling python for matplotlib.
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm/slides
PYTHON=/Users/chmdznr/work/kemendag/sip/docling/bin/python ./build.sh
# Validate
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm
/Users/chmdznr/work/kemendag/sip/docling/bin/python tools/validate_slides.py
/Users/chmdznr/work/kemendag/sip/docling/bin/python tools/validate_quiz.py
```

**TikZ house idiom reference** (read before authoring diagrams): the Q/K/V frame (`module04_slides.tex` ~line 393), the LoRA stack frame (~line 965), and the "Alur Besar" pipeline (~line 252). Colors available: `nvgreen #76B900`, `nvlightgreen #A3D944`, `nvcard #2D2D44`, `nvdarkcard #23233A`, `nvgray #AAAACC`, `nvred #EF5350`, `nvorange #FF6F00`, `nvblue #42A5F5`, `nvwhite`. Frames with code use `[fragile]`; once code is gone, drop `[fragile]` (not needed for TikZ/figures, though harmless).

---

## Task 0: Validators (the "tests")

**Files:**
- Create: `04_llm/tools/validate_slides.py`
- Create: `04_llm/tools/validate_quiz.py`

- [ ] **Step 1: Write `validate_slides.py`**

```python
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
```

- [ ] **Step 2: Write `validate_quiz.py`**

```python
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
```

- [ ] **Step 3: Run both — expect FAIL on current files**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm
/Users/chmdznr/work/kemendag/sip/docling/bin/python tools/validate_slides.py; echo "exit=$?"
/Users/chmdznr/work/kemendag/sip/docling/bin/python tools/validate_quiz.py; echo "exit=$?"
```
Expected: slides FAIL (10 code blocks, "5 notebook", no act 8/9/10). quiz FAIL (16 questions, 3 have code, no SLM keywords).

- [ ] **Step 4: Commit**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course
git add 04_llm/tools/validate_slides.py 04_llm/tools/validate_quiz.py
git commit -m "test(module04): validator slides (bebas kode, 8 notebook) + quiz (murni konsep)"
```

---

## Task 1: Slides Act 3 (nb00) — replace 2 code blocks with illustrations

**Files:** Modify `04_llm/slides/module04_slides.tex` frames at ~L609 (`Self-Attention dalam Beberapa Baris`) and ~L627 (`Latih \& Hasilkan Teks`). Depends on `nb00_loss.pdf` from Task 6 for the loss curve — **author the frame now referencing `figures/nb00_loss.pdf`; the PDF is generated in Task 6 (build runs figures first).**

- [ ] **Step 1: Replace the `Self-Attention dalam Beberapa Baris` frame body**

Remove `[fragile]` + the `lstlisting`. New body = a **causal-mask attention-weight heatmap** (4×4 grid) with a one-line recap. Content spec:
- Title stays `Self-Attention: Menimbang Kata yang Relevan`.
- Left: TikZ 4×4 grid for the sentence tokens `["Ibu", "kota", "Indonesia", "adalah"]`. Cells on/below diagonal shaded `nvgreen` with opacity ∝ a plausible weight (e.g. row "adalah" attends strongly to "Indonesia"); cells above the diagonal filled `nvdarkcard` with a small "✕" = "tidak boleh lihat masa depan".
- Right: 3 bullets — (1) tiap kata bikin **Query**, tiap kata punya **Key** + **Value**; (2) skor = Q·Kᵀ ÷ √dim → `softmax` → bobot; (3) **causal mask** menutup kata di kanan supaya model menebak kiri-ke-kanan.
- Caption: `Inilah "diskusi kelompok" dari Act 2 — sekarang sebagai bobot, bukan kode.`

- [ ] **Step 2: Replace the `Latih \& Hasilkan Teks` frame body**

Remove `[fragile]` + `lstlisting`. New body = **before/after panel + loss curve**. Content spec:
- Title: `Latih TinyGPT: dari Acak ke Kalimat`.
- Left column: two stacked boxes. Top box (`nvred` border) labeled **Sebelum latih (step 0)** containing gibberish `kMKhcrvwf qzx...`. Bottom box (`nvgreen` border) labeled **Sesudah latih (step 3000)** containing a coherent Indonesian sample, e.g. `saya suka belajar jaringan saraf tiruan`.
- Right column: `\includegraphics[width=\linewidth]{figures/nb00_loss.pdf}` (loss 3.68→0.06) + caption `Tugasnya cuma: tebak token berikutnya. Loss turun = model makin paham.` + a small note `TinyGPT ~0.81 juta parameter — prinsip yang sama dengan GPT raksasa, beda skala.`

- [ ] **Step 3: Verify partial — code count drops, compiles**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm/slides
PYTHON=/Users/chmdznr/work/kemendag/sip/docling/bin/python ./build.sh 2>&1 | tail -3
grep -c "begin{lstlisting}" module04_slides.tex   # expect 8 (was 10)
```
Expected: build succeeds (note: `nb00_loss.pdf` exists once Task 6 done; if doing Task 1 first, temporarily comment the `\includegraphics` or run Task 6 first — recommended order: do Task 6 before Task 1). 8 code blocks remain.

- [ ] **Step 4: Commit**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course
git add 04_llm/slides/module04_slides.tex
git commit -m "feat(module04 slides): Act 3 bebas kode — heatmap attention + panel sebelum/sesudah latih"
```

---

## Task 2: Slides Act 4 (nb01) — replace 3 code blocks

**Files:** Modify `module04_slides.tex` frames ~L665 (`Chat Template`), ~L726 (`Zero/Few-shot`), ~L793 (`Chatbot dengan Gradio`).

- [ ] **Step 1: `Chat Template \& System Prompt` → ChatML diagram**

Remove `[fragile]`/code. TikZ flow: three role cards (`system` `nvgray`, `user` `nvgreen`) as a `{role, content}` list → arrow → a "kotak ChatML" showing the special-token wrapping `<|im_start|>system … <|im_end|>` → arrow → model. Bullets: (1) model chat dilatih dengan format khusus (ChatML); (2) `apply_chat_template` menyusun peran system/user persis seperti saat training; (3) jangan rangkai string `User: ...` manual.

- [ ] **Step 2: `Prompt Engineering: Zero-shot \& Few-shot` → two prompt cards**

Remove code. Two side-by-side TikZ cards: **Zero-shot** (cuma pertanyaan) vs **Few-shot** (2 contoh `teks → label` lalu item baru). Caption: `Few-shot = kasih contoh di prompt; model meniru pola tanpa training ulang (in-context learning).`

- [ ] **Step 3: `Chatbot dengan Gradio` → chatbot UI mockup**

Remove code. TikZ mockup of a chat window: a header bar, two user bubbles (`nvgreen`, right) + two assistant bubbles (`nvcard`, left), an input box at the bottom. Bullets: (1) bungkus model jadi fungsi `reply(pesan, riwayat)`; (2) Gradio `ChatInterface` bikin UI chat instan yang bisa di-`share`. (Concept only — name the tool, show no code.)

- [ ] **Step 4: Verify + commit**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm/slides
PYTHON=/Users/chmdznr/work/kemendag/sip/docling/bin/python ./build.sh 2>&1 | tail -3
grep -c "begin{lstlisting}" module04_slides.tex   # expect 5
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course
git add 04_llm/slides/module04_slides.tex
git commit -m "feat(module04 slides): Act 4 bebas kode — diagram ChatML, kartu zero/few-shot, mockup chatbot"
```

---

## Task 3: Slides Act 5 (nb02) — replace 2 code blocks

**Files:** Modify `module04_slides.tex` frames ~L864 (`4-bit NF4 Quantization`), ~L900 (`Profiling, Batching \& Streaming`).

- [ ] **Step 1: `4-bit NF4 Quantization` → weight-shrink diagram + T4 trap callout**

Remove code. Left: TikZ — one weight as a full 16-bit bar (`2 byte`) shrinking by arrow to a quarter-size 4-bit bar (`0,5 byte`), labeled `4× lebih kecil`. Right: a `nvred`/`nvorange` **callout box "Jebakan T4"**: `compute_dtype WAJIB float16` — `bfloat16 crash di T4 (compute 7.5 < 8.0)`. Bullets: NF4 menyimpan bobot dalam 4 bit → 7B muat di T4.

- [ ] **Step 2: `Profiling, Batching \& Streaming` → streaming-vs-waiting timeline**

Remove code. TikZ: two horizontal timelines. Top **"Tunggu penuh"**: long blank bar then the whole answer pops at the end. Bottom **"Streaming"**: tokens appear one-by-one along the bar. Side bullets: batching = proses banyak prompt sekaligus (decoder-only → `padding kiri`); streaming = tampilkan token saat diproduksi (UX lebih enak).

- [ ] **Step 3: Verify + commit**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm/slides
PYTHON=/Users/chmdznr/work/kemendag/sip/docling/bin/python ./build.sh 2>&1 | tail -3
grep -c "begin{lstlisting}" module04_slides.tex   # expect 3
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course
git add 04_llm/slides/module04_slides.tex
git commit -m "feat(module04 slides): Act 5 bebas kode — visual quantization bobot + timeline streaming"
```

---

## Task 4: Slides Act 6 (nb03) — replace LoRAConfig code

**Files:** Modify `module04_slides.tex` frame ~L1037 (`QLoRA dalam Kode`).

- [ ] **Step 1: `QLoRA dalam Kode` → "anatomi QLoRA" visual (rename frame)**

Rename frame title to `Anatomi QLoRA: Apa yang Dilatih`. Remove `[fragile]`/code. TikZ stack: big greyed/locked block = **base 4-bit (dibekukan)**, with thin `nvgreen` **adapter LoRA** clipped onto the attention layers; a pie/bar showing **~1,2% parameter dilatih** (18,4 jt dari 1,56 M... use "18,4 juta dari 1,56 miliar"). Bullets reuse the sticky-note analogy: (1) base dibekukan + di-quantize 4-bit; (2) cuma adapter kecil yang dilatih; (3) hasilnya disimpan terpisah (beberapa MB), bisa ditempel ulang. Keep one small fact line: `Qwen2.5-1.5B, 18 contoh, ~2 menit di T4.`

- [ ] **Step 2: Verify + commit**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm/slides
PYTHON=/Users/chmdznr/work/kemendag/sip/docling/bin/python ./build.sh 2>&1 | tail -3
grep -c "begin{lstlisting}" module04_slides.tex   # expect 2
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course
git add 04_llm/slides/module04_slides.tex
git commit -m "feat(module04 slides): Act 6 bebas kode — anatomi QLoRA (base beku + adapter ~1,2%)"
```

---

## Task 5: Slides Act 7 (nb04) — replace NIM judge code

**Files:** Modify `module04_slides.tex` frame ~L1143 (`LLM-as-Judge: Lokal \& Cloud`).

- [ ] **Step 1: `LLM-as-Judge` → self-preference-bias cartoon**

Remove `[fragile]`/code. TikZ: left — a model handing **its own essay** to itself, giving `5/5` (label `bias self-preference`). Right — the same essay handed to a **big independent cloud judge** (NVIDIA NIM), giving an honest score. Bullets: (1) LLM-as-judge = pakai LLM lain untuk menilai (reference-free, fleksibel); (2) model yang menilai output keluarganya sendiri cenderung memihak; (3) judge cloud besar yang independen lebih tepercaya. One fact line: `Lokal: Qwen3B · Cloud: NVIDIA NIM (build.nvidia.com), OpenAI-compatible.`

- [ ] **Step 2: Verify + commit**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm/slides
PYTHON=/Users/chmdznr/work/kemendag/sip/docling/bin/python ./build.sh 2>&1 | tail -3
grep -c "begin{lstlisting}" module04_slides.tex   # expect 1 (only the bash setup left)
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course
git add 04_llm/slides/module04_slides.tex
git commit -m "feat(module04 slides): Act 7 bebas kode — kartun self-preference bias (judge lokal vs cloud)"
```

---

## Task 6: Figures — matplotlib generators (dark theme → PDF)

**Files:** Create `04_llm/slides/figures/gen_nb00_loss.py` and `04_llm/slides/figures/gen_slm_charts.py`. Follow `gen_model_scale.py` idiom (BG `#1A1A2E`, GREEN `#76B900`, WHITE `#FFFFFF`, hardcoded data, `plt.savefig(..., facecolor=BG)`).

- [ ] **Step 1: `gen_nb00_loss.py`**

```python
"""Figure: kurva loss TinyGPT (3.68 -> 0.06), tema gelap. Output: figures/nb00_loss.pdf"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BG, GREEN, WHITE, GRAY = "#1A1A2E", "#76B900", "#FFFFFF", "#AAAACC"
steps  = [0, 100, 300, 600, 1000, 1500, 2000, 3000]
loss   = [3.681, 1.9, 0.708, 0.118, 0.09, 0.075, 0.068, 0.062]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(steps, loss, color=GREEN, lw=2.4, marker="o", ms=4)
ax.annotate("3,68 (acak)", (0, 3.681), color=WHITE, fontsize=8, xytext=(40, 3.2))
ax.annotate("0,06 (paham)", (3000, 0.062), color=GREEN, fontsize=8, xytext=(1700, 0.5))
ax.set_xlabel("Langkah training", color=WHITE, fontsize=9)
ax.set_ylabel("Loss (makin kecil makin baik)", color=WHITE, fontsize=9)
for s in ax.spines.values(): s.set_color(GRAY)
ax.tick_params(colors=GRAY, labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "nb00_loss.pdf"), facecolor=BG)
```

- [ ] **Step 2: `gen_slm_charts.py` (3 PDFs)**

```python
"""SLM charts (nb05/06), tema gelap.
Output: figures/slm_accuracy.pdf, figures/slm_quadrant.pdf, figures/slm_gallery.pdf"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
BG, GREEN, LGREEN, WHITE, GRAY, RED = "#1A1A2E", "#76B900", "#A3D944", "#FFFFFF", "#AAAACC", "#EF5350"
HERE = os.path.dirname(__file__)
def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color(GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)

# (1) Field-accuracy head-to-head: base 0.6B, generalist 3B, specialist 0.6B
fig, ax = plt.subplots(figsize=(4.4, 3.0)); fig.patch.set_facecolor(BG); style(ax)
names = ["base 0.6B", "generalist 3B", "specialist 0.6B"]
acc   = [78, 72, 95]; colors = [GRAY, RED, GREEN]
ax.bar(names, acc, color=colors)
for i, v in enumerate(acc): ax.text(i, v+1.5, f"{v}%", ha="center", color=WHITE, fontsize=9)
ax.set_ylim(0, 105); ax.set_ylabel("Field-accuracy", color=WHITE, fontsize=9)
ax.set_title("Specialist 0.6B menang (sempit tapi dalam)", color=WHITE, fontsize=9)
plt.setp(ax.get_xticklabels(), color=WHITE, fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(HERE, "slm_accuracy.pdf"), facecolor=BG); plt.close()

# (2) VRAM x accuracy quadrant (the real SLM win = memory)
fig, ax = plt.subplots(figsize=(4.4, 3.0)); fig.patch.set_facecolor(BG); style(ax)
pts = {"base 0.6B": (0.54, 78, GRAY), "specialist 0.6B": (0.54, 95, GREEN), "generalist 3B": (1.99, 72, RED)}
for name, (x, y, c) in pts.items():
    ax.scatter(x, y, s=120, color=c); ax.annotate(name, (x, y), color=WHITE, fontsize=8, xytext=(6, 4), textcoords="offset points")
ax.set_xlabel("VRAM (GB) — kiri lebih hemat", color=WHITE, fontsize=9)
ax.set_ylabel("Field-accuracy (%)", color=WHITE, fontsize=9)
ax.set_title("Hemat memory 3,7x + akurasi tertinggi", color=WHITE, fontsize=9)
ax.set_xlim(0, 2.3); ax.set_ylim(60, 100); plt.tight_layout()
plt.savefig(os.path.join(HERE, "slm_quadrant.pdf"), facecolor=BG); plt.close()

# (3) Task gallery: base -> specialist jump for 3 tasks
fig, ax = plt.subplots(figsize=(4.6, 3.0)); fig.patch.set_facecolor(BG); style(ax)
tasks = ["Intent", "Sentiment+Aspect", "Domain FAQ"]
base = [57, 0, 12]; spec = [100, 100, 100]
x = np.arange(len(tasks)); w = 0.38
ax.bar(x - w/2, base, w, color=GRAY, label="base 0.6B")
ax.bar(x + w/2, spec, w, color=GREEN, label="specialist 0.6B")
ax.set_xticks(x); ax.set_xticklabels(tasks, color=WHITE, fontsize=8)
ax.set_ylim(0, 112); ax.set_ylabel("Akurasi (%)", color=WHITE, fontsize=9)
ax.set_title("Satu resep QLoRA, banyak specialist 100%", color=WHITE, fontsize=9)
leg = ax.legend(facecolor=BG, edgecolor=GRAY, fontsize=8, labelcolor=WHITE)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "slm_gallery.pdf"), facecolor=BG); plt.close()
```

- [ ] **Step 3: Generate + verify 4 PDFs exist**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm/slides
/Users/chmdznr/work/kemendag/sip/docling/bin/python figures/gen_nb00_loss.py
/Users/chmdznr/work/kemendag/sip/docling/bin/python figures/gen_slm_charts.py
ls -1 figures/nb00_loss.pdf figures/slm_accuracy.pdf figures/slm_quadrant.pdf figures/slm_gallery.pdf
```
Expected: all 4 PDFs listed.

- [ ] **Step 4: Commit**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course
git add 04_llm/slides/figures/gen_nb00_loss.py 04_llm/slides/figures/gen_slm_charts.py 04_llm/slides/figures/*.pdf
git commit -m "feat(module04 slides): generator chart matplotlib — loss TinyGPT + 3 chart SLM"
```

> **Recommended order:** do Task 6 before Tasks 1 and 8 (they `\includegraphics` these PDFs). If a slides task is built before its figure exists, the build errors on a missing file.

---

## Task 7: Slides Act 8 — SPESIALISASI (nb05 + nb06), NEW

**Files:** Modify `module04_slides.tex` — insert AFTER Act 7's last frame (after the LLM-as-Judge frame, ~L1175) and BEFORE the current Ringkasan act (`\acttitle{8}` at ~L1179).

- [ ] **Step 1: Insert act divider + 3 content frames**

Insert:
```latex
\acttitle{8}{SPESIALISASI}{SLM: kecil tapi cakap (Notebook 05 \& 06)}
```
Then **Frame 8a — `SLM: Spesialis, Bukan Model Bodoh`:** TikZ teacher→student distillation arrow (model raksasa → model kecil). Bullets: (1) SLM = "sempit tapi dalam" — fokus satu tugas; (2) distillation: model kecil belajar dari output model raksasa → SLM sekarang bisa lebih pintar dari raksasa dulu; (3) lebih murah, hemat memory, bisa jalan on-device (data tak keluar) → cocok untuk masa depan agentic (banyak sub-tugas kecil).

**Frame 8b — `Kecil Mengalahkan Besar` (nb05):** two figures side by side: `\includegraphics[width=\linewidth]{figures/slm_accuracy.pdf}` and `figures/slm_quadrant.pdf`. Caption: `Specialist 0.6B (95%) mengalahkan generalist 3B (72%) di tugas ekstraksi JSON — 5× lebih kecil.` Honest note box: `Kemenangan paling andal = memory (3,7× lebih hemat VRAM). Throughput di T4 ~sama — kita jujur soal ini.`

**Frame 8c — `Satu Resep, Banyak Specialist` (nb06):** left TikZ "pabrik" (satu kotak resep QLoRA di tengah, tiga dataset kecil masuk → tiga SLM specialist keluar: Intent, Sentiment+Aspect, Domain FAQ); right `\includegraphics[width=\linewidth]{figures/slm_gallery.pdf}`. Caption: `Resep yang sama, dataset beda → tiap tugas lompat ke 100%. Sekali kuasai resepnya, kamu bisa cetak specialist sendiri.`

- [ ] **Step 2: Verify + commit**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm/slides
PYTHON=/Users/chmdznr/work/kemendag/sip/docling/bin/python ./build.sh 2>&1 | tail -3
grep -c "acttitle{8}{SPESIALISASI" module04_slides.tex   # expect 1
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course
git add 04_llm/slides/module04_slides.tex
git commit -m "feat(module04 slides): Act 8 SPESIALISASI baru — SLM specialist kalahkan generalist (nb05/06)"
```

---

## Task 8: Slides Act 9 — DEPLOY ke Edge (nb07), NEW

**Files:** Modify `module04_slides.tex` — insert AFTER Act 8 (Task 7) and BEFORE the Ringkasan act.

- [ ] **Step 1: Insert act divider + 2 content frames**

```latex
\acttitle{9}{DEPLOY}{Kirim specialist ke lapangan (Notebook 07)}
```
**Frame 9a — `Pipeline Deployment ke Edge`:** TikZ left→right flow `adapter → merge fp16 → GGUF → quantize (Q8_0/Q4_K_M) → push HuggingFace → device (Ollama)`. A `nvred` callout **"Jebakan #1: merge ke 4-bit"**: salah = adapter di-merge ke base 4-bit → NaN → output `GGGG...`; benar = reload base fp16 dulu, baru merge. Fact line: `GGUF: bf16 1,2 GB · Q8_0 0,6 GB (near-lossless) · Q4_K_M 0,4 GB.`

**Frame 9b — `Satu Kode, Banyak Target` + pohon keputusan:** left — one app code box (concept, no real code: just `client = OpenAI(base_url=...)`-as-label) fanning to 3 targets: Ollama (lokal/edge), vLLM (cloud), NVIDIA NIM. Note: semua OpenAI-compatible → pindah target cukup ganti `base_url`. Right — decision tree: `1 user / edge → Ollama/llama.cpp/TensorRT`; `banyak user / cloud → vLLM/NIM`. Tie-in line: `Langkah hands-on lengkap ada di Runbook Deployment (PDF terpisah).`

- [ ] **Step 2: Verify + commit**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm/slides
PYTHON=/Users/chmdznr/work/kemendag/sip/docling/bin/python ./build.sh 2>&1 | tail -3
grep -c "acttitle{9}{DEPLOY" module04_slides.tex   # expect 1
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course
git add 04_llm/slides/module04_slides.tex
git commit -m "feat(module04 slides): Act 9 DEPLOY baru — pipeline edge + kode portabel (nb07)"
```

---

## Task 9: Slides — global fixes (renumber summary, 8-notebook table, pin, dead code, T4 checklist)

**Files:** Modify `module04_slides.tex`: title slide (~L156), Ringkasan act (`\acttitle{8}`→`{10}` ~L1179), summary table (~L1182-1206), setup frame (~L1208 incl. the bash `lstlisting` ~L1222), closing (~L1237); dead `\sectiontitle` def (~L109).

- [ ] **Step 1: Renumber the Ringkasan act 8 → 10 and reword**

`\acttitle{8}{Ringkasan \& Lanjutan}{Lima notebook, satu perjalanan}` → `\acttitle{10}{Ringkasan \& Lanjutan}{Delapan notebook, satu perjalanan}`.

- [ ] **Step 2: Rebuild the summary table to 8 rows**

Replace `Ringkasan: 5 Notebook Modul 04` → `Ringkasan: 8 Notebook Modul 04`, and the tabular rows with nb00–07:
```
00  BANGUN        TinyGPT (PyTorch)            transformer dari nol
01  PAKAI         Qwen2.5-3B-Instruct          chat template, prompt, klasifikasi
02  PRODUKSIKAN   Mistral-7B (4-bit)           quantization, batching, streaming
03  ADAPT         Qwen2.5-1.5B (QLoRA)         fine-tune adapter ~1%
04  UKUR          Qwen2.5 1.5B vs 3B           metrik, signifikansi, LLM-as-judge
05  SPESIALISASI  Qwen3-0.6B vs SmolLM3-3B     specialist kalahkan generalist
06  GALLERY       Qwen3-0.6B                   satu resep, banyak specialist
07  DEPLOY        Qwen3-0.6B (GGUF)            merge, quantize, Ollama di edge
```
(Adjust the `tabular` column spec if needed so the 4 columns fit; keep the `\rowcolor`/header styling.)

- [ ] **Step 3: Replace the bash setup `lstlisting` with a T4 checklist + tighten pin**

In `Setup T4 \& Lanjut ke Module 05`, remove `[fragile]` + the bash `lstlisting`. Replace with a checklist (itemize with `\checkmark`):
- `transformers>=4.53,<5` (Qwen3/SmolLM3 butuh ≥4.53; v5 bikin `apply_chat_template` rusak)
- 4-bit di T4: `compute_dtype=float16` (bf16 crash)
- `apply_chat_template(..., return_dict=True)` lalu `generate(**inputs)`
- batching decoder-only: `padding_side="left"`
- pakai model non-gated; cek GPU dengan `nvidia-smi`
Keep the RAG teaser (hallucination + knowledge cutoff → Module 05).

- [ ] **Step 4: Title slide + closing 5→8**

Title footer (~L156): `5 notebook` → `8 notebook`; extend tagline to `Bangun → Pakai → Produksikan → Adapt → Ukur → Spesialisasi → Deploy`. Closing `[plain]` frame: update any "5 notebook" / "Lima notebook" to 8 / Delapan.

- [ ] **Step 5: Remove dead `\sectiontitle` command** (~L109-111 block).

- [ ] **Step 6: Verify + commit**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm/slides
PYTHON=/Users/chmdznr/work/kemendag/sip/docling/bin/python ./build.sh 2>&1 | tail -3
grep -c "begin{lstlisting}" module04_slides.tex   # expect 0
grep -c "5 notebook\|Lima notebook" module04_slides.tex   # expect 0
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course
git add 04_llm/slides/module04_slides.tex
git commit -m "feat(module04 slides): Act 10 ringkasan 8 notebook + checklist T4 (pin >=4.53,<5), buang kode mati"
```

---

## Task 10: Slides — full build + validator pass + visual spot-check

**Files:** none (verification only).

- [ ] **Step 1: Clean build**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm/slides
PYTHON=/Users/chmdznr/work/kemendag/sip/docling/bin/python ./build.sh 2>&1 | tail -5
```
Expected: ends "Done! Output: module04_slides.pdf"; log shows `Output written ... (>=44 pages)`.

- [ ] **Step 2: Run validator**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm
/Users/chmdznr/work/kemendag/sip/docling/bin/python tools/validate_slides.py
```
Expected: `SLIDES OK: 0 code blocks, NN pages, acts 8/9/10 present, 8-notebook framing.`

- [ ] **Step 3: Visual spot-check** — render the new/changed pages to PNG and eyeball diagrams + Indonesian phrasing.

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm/slides
pdftoppm -png -r 110 module04_slides.pdf /tmp/sl && ls /tmp/sl-*.png | tail -14
```
Read a sample (Act 3 heatmap, Act 8 SLM charts, Act 9 deploy pipeline, summary table). Fix any overflow/awkward text.

- [ ] **Step 4: Commit any spot-check fixes**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course
git add 04_llm/slides/module04_slides.tex 04_llm/slides/module04_slides.pdf
git commit -m "fix(module04 slides): perbaikan tata letak/bahasa dari spot-check + rebuild PDF"
```

---

## Task 11: Quiz — replace payload with ~30 pure-concept questions

**Files:** Modify `04_llm/llm-fundamentals-quiz.html` — line 52 (`16 soal` → `30 soal`) and line 61 (`const QUIZ = {...};`).

Schema per question: `{"q": str, "code": null, "options": [4 str], "answer": int(0-3), "explanation": str}`. ALL `code` = `null`. Indonesian per kaidah.

- [ ] **Step 1: Update the header count (line 52)**

`<p>16 soal pilihan ganda · gaya sertifikasi · pilih satu jawaban</p>` → `<p>30 soal pilihan ganda · gaya sertifikasi · pilih satu jawaban</p>`

- [ ] **Step 2: Replace the `QUIZ` payload (line 61) with the 30-question set**

Keep these 13 existing (set their `code` to `null` where present — none of the kept ones had code): Q1 NLP-vs-LLM, Q2 transformer-vs-RNN, Q3 params, Q4 tokenization, Q5 embedding, Q6 Q/K/V analogy, Q7 autoregressive, Q8 temperature, Q11 7B-OOM, Q12 LoRA, Q13 QLoRA, Q15 ROUGE/BLEU/BERTScore, Q16 significance.

Add/convert these 17 (full content — stem · 4 options [correct marked] · explanation). Author them in this exact form:

1. **O projection (nb00):** "Setelah Q, K, V menghasilkan campuran berbobot, apa peran proyeksi Output (O)?" — ["O menghapus token yang tak penting", *"O merangkum hasil tiap head jadi satu representasi baru untuk layer berikutnya"*, "O menambah jumlah parameter saja", "O menerjemahkan ke bahasa lain"] · expl: "Q/K/V menghasilkan campuran informasi; proyeksi O menggabungkan hasil semua head menjadi satu vektor keluaran — 'merangkum hasil diskusi' sebelum diteruskan."
2. **Causal mask (nb00):** "Apa fungsi causal mask (segitiga) pada self-attention?" — [*"Menutup token di kanan supaya tiap kata tak melihat masa depan (paksa prediksi kiri-ke-kanan)"*, "Mempercepat GPU", "Menghapus token jarang", "Mengubah token jadi gambar"] · expl: "Causal mask menyetel skor ke arah token masa depan jadi −∞ sebelum softmax, sehingga model hanya boleh memakai konteks kiri saat menebak token berikutnya."
3. **KV-cache (nb00/02):** "Saat LLM menulis token demi token, kenapa KV-cache mempercepat?" — [*"Menyimpan Key & Value token sebelumnya agar tak dihitung ulang tiap langkah"*, "Menyimpan jawaban akhir di disk", "Mengecilkan ukuran model", "Mengganti GPU dengan CPU"] · expl: "Tanpa cache, tiap token baru menghitung ulang K/V seluruh konteks. KV-cache menyimpannya sehingga tiap langkah hanya menghitung token terbaru."
4. **ChatML concept (nb01) [converted]:** "Kenapa model chat perlu format ChatML (`apply_chat_template`), bukan string biasa?" — ["Supaya model lebih kecil", "Supaya tak butuh GPU", *"Karena model instruct dilatih dengan format token khusus (peran system/user); template menyusun persis seperti saat training"*, "Karena string manual dilarang Python"] · expl: "Model instruct belajar dari format ChatML (`<|im_start|>` dst). `apply_chat_template` menyusun pesan + system prompt persis format itu — hasilnya jauh lebih baik daripada string mentah."
5. **Zero vs few-shot (nb01):** "Beda zero-shot dan few-shot prompting?" — [*"Few-shot memberi beberapa contoh di prompt agar model meniru pola, tanpa training ulang"*, "Few-shot melatih ulang seluruh model", "Zero-shot butuh GPU lebih besar", "Keduanya sama persis"] · expl: "Zero-shot langsung bertanya; few-shot menaruh beberapa contoh input→output di prompt (in-context learning) sehingga model meniru polanya — tanpa mengubah bobot."
6. **float16-on-T4 concept (nb02) [converted]:** "Saat quantization 4-bit di Colab T4, kenapa compute-nya harus float16, bukan bfloat16?" — ["float16 paling akurat dari semua tipe", *"T4 (compute 7.5) tidak mendukung bfloat16 — bfloat16 crash; float16 aman"*, "float16 membuat model lebih besar", "Bebas, sama saja"] · expl: "GPU T4 punya compute capability 7.5 (< 8.0) sehingga tidak mendukung bfloat16. Menyetel compute ke bfloat16 menyebabkan error; gunakan float16."
7. **Left padding (nb02):** "Kenapa batching pada model decoder-only butuh padding di kiri (`padding_side='left'`)?" — [*"Agar posisi token terakhir (tempat generasi mulai) sejajar untuk semua prompt; padding kanan merusak generasi"*, "Agar model lebih kecil", "Karena GPU hanya baca dari kiri", "Supaya output berbahasa Indonesia"] · expl: "Model decoder-only melanjutkan dari token terakhir. Dengan panjang prompt beragam, padding di kiri membuat posisi akhir sejajar; padding kanan menaruh token padding di posisi generasi → rusak."
8. **RAG bridge (nb02):** "Kenapa LLM butuh RAG?" — ["Supaya model lebih cepat", *"Model bisa halusinasi & punya knowledge cutoff; RAG memberi 'buku contekan' dokumen relevan ke prompt"*, "Supaya tak butuh tokenizer", "Supaya parameter bertambah"] · expl: "LLM bisa mengarang (halusinasi) dan tak tahu data terbaru/privat (knowledge cutoff). RAG mengambil dokumen relevan lalu menyisipkannya ke prompt agar jawaban akurat."
9. **Prove-learning / disable-adapter concept (nb03) [converted]:** "Bagaimana membuktikan model fine-tuned benar-benar belajar hal baru, memakai satu model yang sama?" — ["Memuat dua model terpisah", *"Matikan adapter sementara untuk dapat jawaban BASE, lalu nyalakan lagi untuk jawaban FINE-TUNED — bandingkan"*, "Hapus model dari memori", "Latih ulang dari nol"] · expl: "Dengan mematikan adapter sementara kita mendapat perilaku base; menyalakannya memberi perilaku fine-tuned. Membandingkan keduanya dari satu model membuktikan adapter-lah yang menambah pengetahuan baru."
10. **What QLoRA trains (nb03/05):** "Pada fine-tuning QLoRA, apa yang sebenarnya dilatih?" — [*"Hanya adapter kecil (~1–2% parameter); base dibekukan & di-quantize 4-bit"*, "Seluruh parameter model", "Hanya tokenizer", "Hanya layer terakhir penuh"] · expl: "QLoRA membekukan base (dalam 4-bit) dan hanya melatih matriks adapter LoRA yang kecil — sekitar 1–2% dari total parameter — jauh lebih murah & cepat."
11. **Specialist beats generalist (nb05):** "Kenapa specialist 0.6B bisa mengalahkan generalist 3B pada tugas ekstraksi JSON?" — [*"'Sempit tapi dalam' — fine-tune fokus satu tugas mengalahkan model umum yang lebih besar di tugas itu"*, "Karena 0.6B selalu lebih pintar", "Karena 3B rusak", "Karena 0.6B pakai internet"] · expl: "SLM yang di-fine-tune untuk satu tugas spesifik bisa lebih akurat di tugas itu daripada generalist yang jauh lebih besar — spesialisasi mengalahkan generalisasi pada cakupan sempit."
12. **Real SLM win = memory (nb05):** "Keunggulan SLM yang paling andal di notebook ini?" — ["Selalu jauh lebih cepat", *"Hemat memory — 3,7× lebih sedikit VRAM (kecepatan ternyata ~sama di T4)"*, "Selalu lebih akurat dari model apa pun", "Tak butuh fine-tuning"] · expl: "Kemenangan SLM yang dijamin di sini adalah akurasi-pada-tugas + memory (0,54 GB vs 1,99 GB ≈ 3,7× lebih hemat). Throughput batch di T4 ternyata ~sama — kita jujur soal ini."
13. **Distillation (nb05):** "Apa itu distillation, dan kenapa membuat model kecil sekarang bisa lebih pintar dari model raksasa dulu?" — [*"Model kecil belajar dari output model raksasa, mewarisi kemampuannya"*, "Menghapus separuh layer model besar", "Menggabungkan dua model jadi satu file", "Menerjemahkan model ke bahasa lain"] · expl: "Distillation melatih model kecil meniru output model raksasa, sehingga SLM modern bisa melampaui LLM lama yang jauh lebih besar."
14. **One recipe many specialists (nb06):** "Kenapa satu resep QLoRA bisa menghasilkan specialist untuk banyak tugas berbeda (klasifikasi, sentiment, FAQ)?" — [*"Prosedur fine-tune-nya sama dan dipakai-ulang; yang berubah hanya dataset kecil tiap tugas"*, "Karena tiap tugas butuh model raksasa", "Karena resepnya berubah total tiap tugas", "Karena base model sudah tahu semuanya"] · expl: "Yang dipakai-ulang adalah resep QLoRA (langkah & konfigurasi); yang berbeda hanya data tugas. Maka satu recipe bisa 'mencetak' banyak specialist."
15. **Merge fp16 not 4-bit (nb07):** "Kenapa adapter LoRA harus di-merge ke base fp16, bukan base 4-bit?" — [*"Merge ke base 4-bit menyuntikkan NaN → output rusak ('GGGG'); reload base fp16 dulu baru merge"*, "Karena 4-bit lebih lambat", "Karena fp16 lebih kecil dari 4-bit", "Tidak ada bedanya"] · expl: "Menggabungkan adapter ke bobot 4-bit yang sudah ter-quantize merusak nilai (NaN) sehingga model menghasilkan gibberish. Aturan emas: reload base fp16, baru attach & merge adapter, lalu export GGUF."
16. **GGUF + edge tools (nb07):** "Apa itu GGUF dan tool mana yang cocok untuk edge (Jetson Orin Nano 8GB)?" — [*"GGUF = format file model universal untuk Ollama/llama.cpp; edge pakai Ollama/llama.cpp/TensorRT, sedangkan vLLM/NIM untuk datacenter"*, "GGUF format gambar; semua tool sama saja", "GGUF hanya untuk cloud", "Jetson 8GB paling cocok pakai vLLM"] · expl: "GGUF adalah satu format file yang dibaca Ollama & llama.cpp. Untuk perangkat edge 8 GB pakai Ollama/llama.cpp/TensorRT; vLLM/NIM adalah tool kelas datacenter yang tak muat di edge."
17. **Portable OpenAI API (nb07):** "Kenapa kode aplikasi bisa pindah dari Ollama ke vLLM ke NVIDIA NIM hanya dengan ganti beberapa baris?" — [*"Semuanya OpenAI-compatible; cukup ganti base_url (dan api_key/model)"*, "Karena semua memakai model yang sama persis", "Karena kode otomatis menulis ulang dirinya", "Karena GGUF berisi kode aplikasi"] · expl: "Ollama, vLLM, dan NIM semua mengekspos OpenAI-compatible API. Maka kode client sama; cukup ganti `base_url` (plus `api_key`/`model`) untuk pindah target."

Final order in the payload: interleave by notebook for a natural 00→07 flow (intro/00 → 01 → 02 → 03 → 04 → 05 → 06 → 07), ~30 total.

- [ ] **Step 3: Validate**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm
/Users/chmdznr/work/kemendag/sip/docling/bin/python tools/validate_quiz.py
```
Expected: `QUIZ OK: 30 pure-concept questions, header matches, SLM/deploy covered.`

- [ ] **Step 4: Commit**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course
git add 04_llm/llm-fundamentals-quiz.html
git commit -m "feat(module04 quiz): rework 30 soal murni konsep (00-07) — SLM specialist, gallery, deploy"
```

---

## Task 12: Quiz — render check (scoring works, one correct each)

**Files:** none (verification only).

- [ ] **Step 1: Headless sanity (no browser needed — validator already parsed JSON). Confirm exactly one correct per question and scoring math.**

```bash
cd /Users/chmdznr/work/navasena/navasena-gen-ml-course/04_llm
/Users/chmdznr/work/kemendag/sip/docling/bin/python - <<'PY'
import re, json
h=open("llm-fundamentals-quiz.html").read()
q=json.loads(re.search(r"const QUIZ = (\{.*?\});", h, re.S).group(1))
qs=q["questions"]
print("total:", len(qs))
assert all(isinstance(x["answer"],int) and 0<=x["answer"]<=3 for x in qs), "bad answer idx"
assert all(len(x["options"])==4 for x in qs), "not 4 options"
assert all(x.get("code") is None for x in qs), "code present somewhere"
print("all questions: 4 options, exactly one answer index, code=null — OK")
PY
```
Expected: total 30, "OK".

- [ ] **Step 2: Optional visual** — open the quiz in a browser (`open llm-fundamentals-quiz.html`) and click through 2–3 questions to confirm lock/reveal/scorebar work with the new payload. (Manual; engine unchanged, so low risk.)

- [ ] **Step 3: No commit** (verification only; any fix loops back to Task 11).

---

## Task 13: Finish — update memory + finishing-a-development-branch

**Files:** Update `~/.claude/projects/-Users-chmdznr-work-navasena-navasena-gen-ml-course/memory/module04-llm-redesign-3-notebooks.md` "Pending" line (slides/quiz now cover 8 notebooks; cheatsheet still stale).

- [ ] **Step 1: Update the memory "Pending" note** to record that slides + quiz are now reworked to cover nb00–07 (code-free slides + pure-concept quiz), leaving only the cheatsheet + Drive sync outstanding.

- [ ] **Step 2: Run the finishing-a-development-branch skill** — verify both validators pass, present options (merge/PR/keep/discard). Given this repo's master-based Colab workflow, expect a direct commit + push to master after final validation.

- [ ] **Step 3: Send the rebuilt slides PDF to the user** for review (`SendUserFile module04_slides.pdf`).

---

## Self-Review

**Spec coverage:**
- Remove all 10 code blocks → Tasks 1–5 (9 Python) + Task 9 step 3 (bash). ✓
- Each removed block → illustration → Tasks 1–5 specify each replacement. ✓
- Two new acts (SPESIALISASI nb05+06, DEPLOY nb07) → Tasks 7, 8. ✓
- Renumber summary to Act 10, 8-row table → Task 9. ✓
- Pin `>=4.53,<5`, remove dead `\sectiontitle`, 5→8 framing → Task 9. ✓
- Charts use verified numbers → Task 6 (78/72/95, 3.7×, 57→100/0→100/12→100, loss 3.68→0.06). ✓
- Quiz ~30 pure-concept across nb00–07, counts updated, code=null → Tasks 11–12. ✓
- Language kaidah → stated as HARD constraint in header + all authored Indonesian strings. ✓
- Validators as gates → Task 0. ✓
- Cheatsheet/Drive out of scope → not tasked (noted in Task 13). ✓

**Placeholder scan:** No "TBD/TODO". Diagram tasks give explicit content specs (labels, numbers, analogy text) rather than pre-rendered TikZ — acceptable for creative LaTeX authored with compile-iterate; the validator + visual spot-check (Task 10) catch defects. Validator + figure + quiz code is complete and literal.

**Type/consistency:** Act numbering consistent — existing acts 1–7 unchanged, new acts `\acttitle{8}{SPESIALISASI}`, `\acttitle{9}{DEPLOY}`, old Ringkasan `\acttitle{8}`→`{10}`. Validator checks for `{8}{SPESIALISASI`, `{9}{DEPLOY`, `{10}`. Figure filenames consistent between Task 6 (creates `nb00_loss.pdf`, `slm_accuracy.pdf`, `slm_quadrant.pdf`, `slm_gallery.pdf`) and Tasks 1/7 (`\includegraphics` the same names). Quiz schema (`q/code/options/answer/explanation`) matches the existing payload exactly.

**Execution order note:** Do **Task 0 → Task 6 (figures) → Tasks 1–5 → Tasks 7–9 → Task 10 → Tasks 11–12 → Task 13.** Figures (Task 6) must precede any frame that `\includegraphics` them. All slides tasks edit one file (`module04_slides.tex`) → execute sequentially, not in parallel.
