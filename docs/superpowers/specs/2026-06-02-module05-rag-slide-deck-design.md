# Design Spec: Module 05 (RAG) Slide Deck

> **Tanggal**: 2026-06-02
> **Status**: Approved (brainstorming) → siap build
> **Scope**: Satu slide deck `05_rag/slides/module05_slides.tex` (~18 frame, 4 act) mengikuti template Module 03/04. BI primary.

## Tujuan

Module 05 punya 1 notebook (`01_rag_fundamentals.ipynb`) yang sudah ada intro + theory cells. Slide deck melengkapi notebook dengan penjelasan konseptual RAG yang koheren, jadi bahan presentasi instruktur.

## Template & Konvensi (warisan Module 03/04)

- `\documentclass[aspectratio=169,t]{beamer}`, xelatex, FiraSans, palet Navasena dark.
- `\acttitle{N}{Judul}{Subjudul}`, footline **"RAG Fundamentals"** (tanpa NCA-GENL).
- `\title{RAG Fundamentals}`, `\subtitle{Navasena Gen-ML Course}`, `\author{Navasena}`.
- Preamble disalin VERBATIM dari Module 04.

### LaTeX gotchas (wajib — dari Module 02/04)
- Frame dengan `lstlisting` HARUS `[fragile]`.
- Tabel kolom sempit: `\setlength{\tabcolsep}{3pt}` + lebar `p{}` < linewidth.
- TikZ: `scale=` + `every node/.style={transform shape}`; JANGAN `\resizebox` untuk diagram tinggi. Nama node TANPA titik desimal (pakai index integer).
- Target **0 overfull box**.

## Struktur (≈18 frame: title + 4 act divider + 12 konten + ringkasan + closing)

**Title** — Module 05 / RAG Fundamentals / "Retrieval-Augmented Generation".

**Act 1 — Apa itu RAG?** (3 konten)
1. Masalah LLM — **halusinasi** (mengarang) + **knowledge cutoff** (tak tahu kejadian baru).
2. Solusi RAG — **Retrieval** (cari konteks) + **Generation** (LLM jawab dari konteks); definisi.
3. Analogi — ujian buka buku / "buku contekan"; LLM dibekali fakta dulu.

**Act 2 — Komponen RAG** (4 konten)
4. Embeddings & **SentenceTransformer** — teks → vektor; makna mirip → vektor dekat (`all-MiniLM-L6-v2`). figure `embedding_space`.
5. Similarity Search & **FAISS** — `IndexFlatL2` (jarak Euclidean), cari top-k vektor terdekat cepat.
6. Knowledge Base — kumpulan dokumen/fakta yang di-embed jadi index.
7. Generator LLM — **TinyLlama** menyusun jawaban akhir (float32, sampling temperature/top_p).

**Act 3 — Pipeline RAG** (4 konten)
8. Alur End-to-End — **retrieve → augment → generate** (diagram `rag_pipeline`).
9. Retrieve — query → embed → cari top-k konteks di FAISS.
10. Augment — sisipkan konteks ke dalam **prompt** bersama pertanyaan (TikZ).
11. Generate — LLM jawab BERDASARKAN konteks; kurangi halusinasi.

**Act 4 — Praktik & Lanjutan** (1 konten + wrap)
12. Kode Sistem RAG — ringkasan fungsi `retrieve_relevant_context()` + `ask_question()` (notebook).
13. Tips & Batasan — ukuran chunk, nilai `k`, kualitas embedding, konteks bisa salah.
14. Ringkasan Modul 05 (table) + Jembatan ke **Module 06** (RAG dipercepat GPU/NVIDIA).

**Closing** — Terima Kasih.

## Figures

| File | Tipe | Isi |
|------|------|-----|
| `figures/rag_pipeline.mmd` | Mermaid | query → embed → FAISS search → konteks → augment prompt → LLM → jawaban |
| `figures/gen_embedding_space.py` | Matplotlib | scatter 2D: kalimat semantically-similar mengelompok (ilustratif, seed tetap) |
| (inline TikZ) | — | augment (query + konteks → prompt), top-k retrieval |

Render: matplotlib via docling python → `figures/*.pdf`; mermaid via `mmdc -s 3 -b transparent`. Tema gelap (#1A1A2E, aksen #76B900).

## Build

`05_rag/slides/build.sh` = salinan Module 04. Pre-clean aux, 2-pass xelatex. Commit: `.tex`, `build.sh`, `figures/gen_*.py`, `figures/*.mmd`, figure outputs, final PDF.

## Out of scope

- Mengubah notebook Module 05 (sudah ada theory cells).
- Deck Module 06.

## Risiko & mitigasi

- Notebook RAG memakai float32 (bukan float16) untuk stabilitas CPU — slide harus akurat soal ini (jangan klaim float16).
- Konteks retrieval bisa salah → jelaskan RAG bukan jaminan benar 100%.
- Overfull: patuhi gotchas LaTeX.
