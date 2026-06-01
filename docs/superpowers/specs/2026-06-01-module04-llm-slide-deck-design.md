# Design Spec: Module 04 (LLM) Slide Deck

> **Tanggal**: 2026-06-01
> **Status**: Approved (brainstorming) → siap build
> **Scope**: Buat slide deck `04_llm/slides/module04_slides.tex` mengikuti template Module 03 (Navasena dark, xelatex, 16:9), kedalaman "konsep + internal", bahasa BI primary.

## Tujuan

Module 04 punya 1 notebook (`01_llm_basics.ipynb`) yang kini sudah ada theory cells. Slide deck ini melengkapi notebook dengan penjelasan konseptual — termasuk cara kerja LLM (tokenization, transformer, attention) yang tidak dibahas di notebook — agar instruktur punya bahan presentasi yang koheren dengan notebook.

## Template & Konvensi (warisan Module 03)

- `\documentclass[aspectratio=169,t]{beamer}`, compiler **xelatex**, font **FiraSans**.
- Palet Navasena dark: `nvbg #1A1A2E`, `nvcard #2D2D44`, `nvgreen #76B900`, `nvlightgreen #A3D944`, `nvgray #AAAACC`, `nvred #EF5350`, `nvorange #FF6F00`, `nvblue #42A5F5`.
- Custom command `\acttitle{N}{Judul}{Subjudul}` untuk act divider, `\sectiontitle{}{}`.
- Footline: **"LLM Fundamentals"** \hfill frame X/Y. **TANPA** NCA-GENL / NVIDIA branding.
- `\title{LLM Fundamentals}`, `\subtitle{Navasena Gen-ML Course}`, `\author{Navasena}`.
- Preamble disalin VERBATIM dari `module03_slides.tex` (hanya ganti judul modul + footer string).

## Struktur (≈21 frame: title + 5 act divider + 16 konten + closing)

**Title frame**

**Act 1 — Apa itu LLM?** (3 konten)
1. Definisi: **LLM** vs **NLP**, LLM = jenis model NLP berbasis **transformer** dilatih data masif.
2. Skala & evolusi: n-gram → RNN → Transformer; ukuran parameter; contoh GPT/Llama/OPT. (figure: `gen_model_scale.py`)
3. Aplikasi LLM nyata: chatbot, code gen, translation, summarization, QA.

**Act 2 — Cara Kerja LLM** (4 konten, inti konseptual)
4. **Tokenization**: teks → token (subword) → ID. (TikZ inline contoh)
5. **Embedding & Transformer**: token → vektor → transformer blocks ×N. (figure: `llm_architecture.mmd`)
6. **Attention** intuitif: model "memperhatikan" kata relevan. (TikZ inline word-to-word)
7. **Autoregressive generation** + sampling: prediksi next-token; `temperature`/`top_p`/`do_sample`. (figure: `gen_sampling.py`)

**Act 3 — Memakai LLM dengan HuggingFace** (4 konten)
8. Ekosistem **HuggingFace**: `transformers` + model hub + model card.
9. **`pipeline`** vs load manual (`AutoModel` + `AutoTokenizer`). (listing code)
10. Model **gated** & login HF (`huggingface-cli login`, Llama-2).
11. **Text classification**: `AutoModelForSequenceClassification` + `pipeline("sentiment-analysis")`, label & score.

**Act 4 — Aplikasi LLM** (3 konten)
12. **Chatbot** dengan **Gradio** (`gr.ChatInterface`).
13. **Few-shot learning**: contoh dalam prompt tanpa training; struktur prompt. (TikZ inline)
14. **Model evaluation**: ukur kualitas output (prompt/response/length).

**Act 5 — Ringkasan & Lanjutan** (2 konten + closing)
15. Ringkasan Modul 04: tabel recap topik/library/tujuan.
16. Setup & tools: model kecil (opt-350m, TinyLlama, bert-tiny), float16, GPU T4, requirements; lalu jembatan ke **Module 05 RAG** (LLM bisa halusinasi & punya knowledge cutoff → RAG).

**Closing frame** (terima kasih / Navasena).

## Figures

| File | Tipe | Isi |
|------|------|-----|
| `figures/llm_architecture.mmd` | Mermaid | text → tokenizer → embeddings → transformer ×N → logits → next token |
| `figures/autoregressive.mmd` | Mermaid | loop generasi token-by-token (opsional bila muat) |
| `figures/gen_sampling.py` | Matplotlib | efek `temperature` pada distribusi softmax (peaked vs flat) |
| `figures/gen_model_scale.py` | Matplotlib | pertumbuhan jumlah parameter model (skala log) |
| (inline TikZ) | — | tokenization contoh, attention word-to-word, struktur few-shot prompt |

Render: `mmdc -i x.mmd -o x.png -s 3 -b transparent`; matplotlib via docling python (`/Users/chmdznr/work/kemendag/sip/docling/bin/python`, sudah punya matplotlib + sklearn). Output `figures/*.png` / `*.pdf`; `.tex` mereferensi `figures/...`.

## Build

- `04_llm/slides/build.sh` = salinan Module 03: (1) jalankan `figures/gen_*.py`, (2) konversi `figures/*.mmd` → png (mmdc -s 3), (3) 2-pass xelatex.
- Pre-clean: `find . -maxdepth 1 -name "*.aux" -delete` dsb sebelum compile (zsh nullglob).
- Target: **0 overfull box** (hook `latex-overfull-guard.sh` block ≥2pt hbox & semua vbox overflow).
- Gitignore sudah cover `*.aux *.log *.nav *.out *.snm *.toc *.vrb`; commit hanya `.tex`, `build.sh`, `figures/gen_*.py`, `figures/*.mmd`, dan `*.pdf` final + figure PNG/PDF (ikut pola Module 03 yang commit figure outputs).

## Out of scope

- Slide deck Module 05/06 (berikutnya).
- Mengubah notebook Module 04 (sudah selesai theory cells).
- Cheat sheet PDF.

## Risiko & mitigasi

- **Overfull box** (umum di 16:9 kolom sempit): pakai `\resizebox`/`\parbox{\dimexpr\linewidth-8pt}`, font lebih kecil, atau split frame. Iterasi compile→fix sampai bersih.
- **Akurasi konseptual** (attention/transformer untuk SMA): jaga intuitif, hindari notasi matematis berat; review akurasi sebelum finalisasi.
- **LaTeX dari draft paralel**: semua frame pakai HANYA command yang terdefinisi di preamble; assembly + compile-fix dilakukan manual.
