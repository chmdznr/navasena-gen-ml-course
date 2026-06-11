# Module 04 — Slides + Quiz Rework Design

**Date:** 2026-06-12
**Status:** Approved (structure + decisions confirmed by user)
**Scope:** Rework `04_llm/slides/module04_slides.tex` and `04_llm/llm-fundamentals-quiz.html` so both cover the full 8-notebook arc (00–07), with slides shifted from code-centric to explanation-and-illustration, and the quiz reworked to pure-concept.

---

## Goal

Transform the Module 04 lecture deck and quiz from a 5-notebook (00–04), code-heavy form into an **8-notebook (00–07), explanation-and-illustration-first** form aimed at high-school (SMA) graduates.

- **Slides:** remove ALL Python code; every removed block is replaced by a diagram/illustration. Add coverage for nb 05 (SLM specialist), 06 (task gallery), 07 (edge deployment).
- **Quiz:** rework into ~28–30 pure-concept multiple-choice questions covering all 8 notebooks; no code snippets in questions.

## Constraints (hard)

1. **Language kaidah (verbatim):** *"pastikan menggunakan Bahasa Indonesia untuk Engineering yang luwes/fleksibel, artinya istilah-istilah serapan dalam bahasa Inggris yang memang umum dipakai itu tidak perlu ditranslasikan. pastikan frasa atau kalimat yang dipakai itu tidak ambigu (aneh didengar)."* → keep common English loan-words untranslated (deploy, fine-tune, adapter, quantize, embedding, attention, token, prompt, GGUF, merge, batch, streaming, runtime, throughput, system prompt, dst.); fix any awkward/ambiguous phrasing; avoid calques.
2. **Audience = lulusan SMA** — explanations simple, analogy-driven, no math derivations.
3. **Slides contain zero Python code** — concepts conveyed via TikZ/matplotlib diagrams, tables, callouts, analogies.
4. **Quiz = pure concept** — no code-read questions; single-answer MCQ (preserve the existing self-scoring engine).
5. **Preserve existing infrastructure:** Beamer "Navasena Dark" theme + custom commands (`\acttitle`) + footline; the quiz's single-file interactive self-scoring HTML engine + NVIDIA dark theme.
6. **Free-tier Colab T4 context** stays accurate: `transformers>=4.53,<5` (Qwen3/SmolLM3 need ≥4.53; v5 breaks things), `float16` compute on T4 (bf16 crashes), non-gated models default.
7. **Illustration tooling:** chosen per-diagram (user delegated). Default mapping — flow/concept diagrams → inline TikZ (vector, theme-matched); data charts (benchmark bars, scatter) → matplotlib → PDF in `slides/figures/` with a generator script; reserve mermaid→PNG (`mmdc`, scale 2–3) only if a diagram is too gnarly for clean TikZ.

## Source of truth

Content/numbers/analogies come from the verified notebooks 00–07 and the audit (workflow `wf_1c26019d-775`). Key reusable analogies already written in the notebooks:
- Attention = "menimbang kata mana yang paling relevan"; multi-head = "banyak sudut perhatian paralel".
- LoRA/QLoRA = sticky-note pada model raksasa yang dibekukan; latih hanya ~1% parameter.
- SLM = "spesialis sempit tapi dalam, bukan generalis serba-bisa"; distillation = "model kecil belajar dari output model raksasa".
- Deploy = "kirim pekerja terlatih ke lapangan"; HF Hub = "jembatan distribusi".
- Evaluation = "'kelihatannya bagus' bukan bukti"; self-preference bias = "murid menilai ujiannya sendiri".

Verified numbers to put on charts:
- nb00 TinyGPT: 808,995 params (~0.81M); loss 3.68→0.06 over 3000 steps; vocab 35.
- nb02: 7B fp16 ~14.5 GB (OOM) vs 4-bit NF4 ~3.6 GB (fits); Mistral-7B 4-bit peak ~7.1 GB.
- nb03: 1.18% trainable (18.4M/1.56B); ~2 min train; loss 0.68.
- nb04: ROUGE-L 0.775 vs 0.790; BLEU 56.8 vs 60.8; BERTScore 0.982 vs 0.986; perplexity 69.8 vs 54.4; paired t-test p=0.13 (NOT significant); Cohen's d 0.52.
- nb05: field-accuracy base 78% / generalist(3B) 72% / **specialist(0.6B) 95%**; memory 0.54 GB vs 1.99 GB (**3.7×**); throughput ~1.0× (honest "sama").
- nb06: base→specialist 57→100, 0→100, 12→100; VRAM 0.6B 2.43 / 1.7B 3.24 / 3B 3.86 GB; throughput 150/145/119 tok/s.
- nb07: GGUF bf16 1.2 GB / Q8_0 0.61 GB / Q4_K_M 0.38 GB; ~128 tok/s on T4 (caveat T4≠Jetson).

---

## Part A — Slides (`module04_slides.tex`)

Existing deck: 36 frames, 8 acts (Title, Act1 Apa&Mengapa, Act2 Bagaimana Bekerja, Act3 BANGUN nb00, Act4 PAKAI nb01, Act5 PRODUKSIKAN nb02, Act6 ADAPT nb03, Act7 UKUR nb04, Act8 Ringkasan).

### A.1 Replace the 10 Python code blocks with illustrations

| # | Current code block (location) | Replacement (no code) |
|---|---|---|
| 1 | Act3 self-attention (~L611) | 4×4 causal-mask **attention-weight heatmap** (upper triangle blacked out) + 1-line recap of Q·Kᵀ/√d → softmax → weighted V. TikZ. |
| 2 | Act3 training loop (~L640) | **Before/after panel** (gibberish kiri → kalimat Indonesia koheren kanan) + **loss curve** 3.68→0.06. matplotlib chart + TikZ panel. |
| 3 | Act4 chat-template (~L678) | **ChatML diagram**: list `{role, content}` (system+user) → dibungkus jadi string token spesial → model. TikZ. |
| 4 | Act4 few-shot (~L739) | **Dua kartu prompt**: zero-shot vs few-shot (in-context learning, tanpa retraining). TikZ. |
| 5 | Act4 Gradio (~L806) | **Mockup UI chatbot** (gelembung user/assistant), level konsep. TikZ. |
| 6 | Act5 BitsAndBytesConfig (~L877) | **"Apa yang dilakukan quantization pada satu weight"** (bar 16-bit → bar 4-bit seperempatnya) + callout box jebakan T4 (`compute_dtype=float16`, bf16 crash). TikZ. |
| 7 | Act5 streaming (~L912) | **Timeline streaming vs menunggu** (UX). TikZ. |
| 8 | Act6 LoRAConfig (~L1050) | Perkuat **stack LoRA/QLoRA** yang sudah ada + visual "~1.2% parameter dilatih"; pertahankan analogi sticky-note. TikZ. |
| 9 | Act7 NIM judge (~L1156) | **LLM-as-judge + self-preference bias** (judge lokal menilai esainya sendiri vs judge cloud independen). TikZ. |
| 10 | Act8 pip/bash setup (~L1222) | **Checklist aturan aman T4** (bullets, bukan kode); pin diperbarui ke `transformers>=4.53,<5`. |

> Removing code keeps the same frames but swaps their body to an illustration; frame count for Acts 1–8 stays ~36.

### A.2 Add two SLM acts (~9 new frames)

**Act 8 — SPESIALISASI (nb05 + nb06)** — divider `\acttitle{8}{SPESIALISASI}{SLM: kecil tapi cakap}` then:
- **Frame 9a — "SLM itu spesialis, bukan model bodoh":** tesis "sempit tapi dalam" + analogi distillation (model kecil belajar dari output model raksasa). Konsep "many small repetitive sub-tasks / agentic future". TikZ teacher→student.
- **Frame 9b — nb05 head-to-head:** **grouped bar** field-accuracy (base 78 / generalist-3B 72 / **specialist-0.6B 95**) + **quadrant VRAM×accuracy** (kemenangan nyata = memory 3.7× lebih hemat). Catatan jujur: throughput ~sama di T4. matplotlib (2 charts).
- **Frame 9c — nb06 "satu resep → banyak spesialis":** **diagram pabrik/percetakan** (satu resep QLoRA di tengah, tiga dataset kecil masuk → tiga SLM spesialis keluar) + **bar lonjakan** base→specialist (57→100, 0→100, 12→100). TikZ factory + matplotlib bars.

**Act 9 — DEPLOY ke Edge (nb07, level konsep)** — divider `\acttitle{9}{DEPLOY}{Kirim spesialis ke lapangan}` then:
- **Frame 10a — pipeline deployment:** flow kiri→kanan `adapter → merge fp16 → GGUF → quantize (Q8_0/Q4_K_M) → push HF → device (Ollama)`, dengan **jebakan merge-on-4bit** sebagai callout salah-vs-benar (4-bit → output "GGGG"). TikZ.
- **Frame 10b — kode aplikasi portabel + pohon keputusan:** satu kode aplikasi, ganti `base_url` → Ollama/vLLM/NIM (OpenAI-compatible); **decision tree** edge vs datacenter (Ollama/llama.cpp/TensorRT untuk edge; vLLM/NIM untuk cloud). Tunjuk ke runbook PDF untuk hands-on. TikZ.

> Note: the existing "Act 8: Ringkasan" is pushed to the end and renumbered. SPESIALISASI and DEPLOY are inserted before the summary. Final `\acttitle` ordering: …Act7 UKUR → **Act8 SPESIALISASI** → **Act9 DEPLOY** → **Act10 Ringkasan** (the old Ringkasan divider's number changes 8→10).

### A.3 Global fixes
- Title slide, summary table, and closing slide: "5 notebook" → **"8 notebook"**; tagline extended to include Spesialisasi + Deploy.
- Rebuild the **Ringkasan** table to 8 rows (nb00–07: verb/topic/model), reflecting Qwen3-0.6B/SmolLM3-3B for 05–07.
- Setup/closing: `transformers>=4.53,<5`; keep float16/return_dict guidance.
- Remove dead `\sectiontitle` command.
- Optionally include the existing-but-unused `figures/llm_architecture.png` (or drop the dead asset) — builder's discretion.

### A.4 Build & verify (slides)
- Compile via `slides/build.sh` (xelatex). Must succeed with no errors and no leftover code blocks.
- Render 4–6 representative pages to PNG to confirm diagrams + Indonesian phrasing.
- Target ~45 frames, zero `lstlisting` Python.

---

## Part B — Quiz (`llm-fundamentals-quiz.html`)

Preserve the single-file interactive self-scoring HTML engine + NVIDIA dark theme. Only the `QUIZ` JSON payload and the hardcoded counts change.

### B.1 Blueprint (~28–30 single-answer MCQ, pure concept, Indonesian per kaidah)

**Reuse (reworded, concept):** the 13 existing concept questions (NLP-vs-LLM, transformer-vs-RNN, params, tokenization, embeddings, Q/K/V analogy, autoregressive, temperature, 7B-OOM math, LoRA, QLoRA, ROUGE/BLEU/BERTScore, significance).

**Convert to code-free concept:** Q9 chat-template (why ChatML/`apply_chat_template` vs raw string), Q10 `compute_dtype=float16` on T4 (why not bf16), Q14 `disable_adapter` (why toggle adapter to prove learning).

**Add (~12 new), from audit quiz-seeds:**
- nb00: causal-mask purpose; √d scaling; residual+LayerNorm role. (pick 2)
- nb01: float16-not-bf16 on T4 (concept); zero-shot vs few-shot.
- nb02: left-padding for decoder-only batching; allocated vs reserved vs peak; RAG-bridge motivation (hallucination + knowledge cutoff). (pick 2)
- core: KV-cache; the "O" output projection (its role after Q/K/V). (pick 1–2)
- nb05: kenapa specialist 0.6B mengalahkan generalist 3B (narrow-but-deep); kemenangan SLM yang paling andal = memory (bukan kecepatan); distillation; QLoRA melatih ~1.67%. (pick 3)
- nb06: satu resep → banyak spesialis (apa yang dipakai-ulang vs apa yang berubah per task); kenapa base 0% di aspect-sentiment; trade-off ukuran (VRAM vs throughput). (pick 2)
- nb07: kenapa merge ke base fp16 bukan 4-bit; apa itu GGUF & kenapa untuk edge; Q8_0 vs Q4_K_M; kode portabel OpenAI-compatible (ganti 3 baris); tool edge vs datacenter. (pick 3)
- SLM-vs-LLM trade-off konseptual ("kapan pilih SLM spesialis").

Each question: 1 stem + 4 options + exactly 1 correct + a short `explain` string (kaidah-compliant). Distractors plausible, not trick.

### B.2 Engine updates
- Replace the inline `QUIZ` array with the new ~28–30 items.
- Update hardcoded count strings (header subtitle, scorebar `/16`, footer) to the new total. (Summary uses dynamic `total`, so it self-updates.)

### B.3 Verify (quiz)
- Open headless (or parse the inline JSON) to confirm: valid JSON; every question has exactly one option flagged correct; count matches header/scorebar; scoring runs.

---

## Part C — Out of scope (flagged, not done here)
- `llm-fundamentals-cheatsheet.html/.pdf` rework (still describes 5 notebooks) — separate future task; note as tech debt.
- Google Drive sync.
- Notebook content changes (notebooks are verified; this is assets-only).

## Acceptance criteria
- [ ] Slides compile clean (xelatex), ~45 frames, **zero Python code**, 2 new SLM acts present, summary says 8 notebooks, pin `>=4.53,<5`.
- [ ] Every former code frame now shows a diagram/illustration/table conveying the same idea.
- [ ] Quiz = ~28–30 pure-concept MCQ across nb00–07, no code snippets, counts updated, self-scoring works, one correct answer each.
- [ ] All new/changed Indonesian text follows the language kaidah (serapan untranslated, no awkward/ambiguous phrasing).
- [ ] Charts use verified numbers from the notebooks.
