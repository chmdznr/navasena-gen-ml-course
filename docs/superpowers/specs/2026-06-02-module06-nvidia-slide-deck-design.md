# Design Spec: Module 06 (NVIDIA Ecosystem) Slide Deck

> **Tanggal**: 2026-06-02
> **Status**: Approved (brainstorming) → siap build
> **Scope**: Satu slide deck `06_nvidia_ecosystem/slides/module06_slides.tex` (~22 frame, 3 act) mengikuti template Module 03/04/05. BI primary. **Deck terakhir** — melengkapi semua modul punya slide.

## Tujuan

Module 06 punya 3 notebook (basic GPU, NeMo demo, NVIDIA RAG), semuanya sudah ada theory cells. Slide deck menyatukan jadi presentasi ekosistem NVIDIA. Karena GPU/CUDA/TensorRT sudah disinggung di deck Module 02 dan RAG di Module 05, deck ini menekankan yang DISTINCTIVE: **NeMo** sebagai inti; Act GPU & RAG diberi sudut "tooling NVIDIA", bukan mengajar ulang dasar.

## Template & Konvensi (warisan Module 03/04/05)

- `\documentclass[aspectratio=169,t]{beamer}`, xelatex, FiraSans, palet Navasena dark.
- `\acttitle{N}{Judul}{Subjudul}`, footline **"NVIDIA Ecosystem"** (tanpa NCA-GENL).
- `\title{NVIDIA Ecosystem}`, `\subtitle{Navasena Gen-ML Course}`, `\author{Navasena}`.
- Preamble disalin VERBATIM dari Module 05.

### LaTeX gotchas (wajib — diterapkan di depan)
- Frame `lstlisting` HARUS `[fragile]`.
- Tabel kolom sempit: `\setlength{\tabcolsep}{3pt}` + lebar `p{}` < linewidth.
- TikZ: `scale=` + `transform shape`; nama node TANPA titik desimal (index integer); JANGAN `\resizebox` diagram tinggi.
- Target **0 overfull box**.

## Struktur (≈22 frame: title + 3 act divider + 16 konten + ringkasan + closing)

**Title** — Module 06 / NVIDIA Ecosystem / "GPU, NeMo, dan RAG yang dipercepat".

**Act 1 — GPU, CUDA & Optimasi** (5 konten, sudut tooling)
1. Stack NVIDIA — **Hardware** (T4/A100/H100) → **CUDA** → **Library** (cuDNN/cuBLAS/TensorRT) → **Framework** → **Tools** (NeMo/Triton). figure `nvidia_stack`.
2. Kenapa GPU (recap singkat) — paralelisme; cocok matrix ops deep learning.
3. Benchmark — model **microsoft/phi-2**, CPU vs GPU; mixed precision.
4. **Mixed Precision** — FP16 + FP32 (Tensor Cores), hemat memori + cepat.
5. **TensorRT** — optimasi inference (layer fusion, precision calibration); **TF-TRT** 2--5x lebih cepat (TikZ).

**Act 2 — NVIDIA NeMo** (inti, 6 konten)
6. Apa itu **NeMo** — toolkit open-source conversational AI; banyak pre-trained model.
7. Peta Tugas NeMo — figure `nemo_tasks` (6 task fan-out).
8. **NER** & **Question Answering** — `ner_en_bert`, `qa_squadv1.1_bertbase`.
9. **Text Classification** & **Punctuation/Capitalization** — `punctuation_en_bert`.
10. **Machine Translation** & **TTS** — `nmt_en_zh`, FastPitch + HiFiGAN.
11. Catatan — demo GPU-heavy & konseptual; ASR di-import tapi tak didemokan.

**Act 3 — NVIDIA-Accelerated RAG** (4 konten)
12. RAG di atas Stack NVIDIA — embedding + LLM di GPU; `faiss-gpu`.
13. Arsitektur `NvidiaRAGSystem` — embedding (GPU) + FAISS + LLM `gpt2`; `cudf` untuk dokumen.
14. CPU (Module 05) vs GPU (Module 06) — perbandingan; faiss-gpu, presisi.
15. Alur Demo — retrieve → augment → generate (`demonstrate_rag()`).
16. Catatan akurasi — `MegatronGPTModel` diimpor tapi LLM yang dipakai `gpt2`; FAISS dibangun di CPU meski install faiss-gpu.

**Wrap** — Ringkasan ekosistem (table layer) + closing "akhir course".

## Figures

| File | Tipe | Isi |
|------|------|-----|
| `figures/nvidia_stack.mmd` | Mermaid | layered: Hardware → CUDA → Library → Framework → Tools (bottom-up) |
| `figures/nemo_tasks.mmd` | Mermaid | NeMo → 6 task (NER, QA, Classification, Punctuation, MT, TTS) |
| (inline TikZ) | — | TensorRT optimasi (model → fuse/calibrate → faster); CPU vs GPU RAG |

Render: mermaid via `mmdc -s 3 -b transparent` → `figures/*.png`. Tema gelap.

## Build

`06_nvidia_ecosystem/slides/build.sh` = salinan Module 05. Pre-clean aux, 2-pass xelatex. Commit: `.tex`, `build.sh`, `figures/*.mmd`, figure png, final PDF. (Bila tak ada `gen_*.py`, langkah figure python di build.sh otomatis dilewati.)

## Out of scope

- Mengubah notebook Module 06 (sudah ada theory cells).

## Risiko & mitigasi

- **Overlap** dengan deck Module 02 (GPU) & 05 (RAG) — mitigasi: Act 1/3 fokus sudut NVIDIA-tooling, eksplisit "lihat Module 02/05 untuk dasar".
- **Akurasi NeMo**: nama model & kenyataan demo (ASR import tapi tak dipakai; gpt2 bukan Megatron; FAISS CPU) — review akurasi terhadap notebook.
- Overfull: patuhi gotchas.
