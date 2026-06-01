# Design Spec: Theory Cells untuk Module 04 / 05 / 06

> **Tanggal**: 2026-06-01
> **Status**: Approved (brainstorming) → siap masuk implementation plan
> **Scope**: Tambah *theory markdown cells* ke notebook Module 04 (LLM), 05 (RAG), 06 (NVIDIA Ecosystem), mengikuti pattern yang sudah terbentuk di Module 03.

## Tujuan

Module 03 sudah punya scaffolding edukatif yang baik (theory cells per-section, intro, Ringkasan). Module 04/05/06 belum — sebagian besar hanya berisi code cells tanpa penjelasan konsep. Tujuan revisi ini: menyamakan kualitas pedagogis Module 04/05/06 dengan Module 03, **tanpa mengubah satu pun code cell**.

## Style & Pedagogical Contract (warisan Module 03)

Setiap theory cell yang ditambahkan WAJIB mengikuti kontrak berikut:

- **Bahasa Indonesia primary**, istilah teknis Inggris **di-bold** saat pertama muncul.
- Target audience: *"siswa SMA/sederajat dengan basic Python"*.
- Cell pendek (±250–550 char): heading → definisi satu kalimat → 2–5 bullet yang menyebut teknik + library/model yang dipakai.
- Setiap notebook **dibuka** dengan cell intro `# 📚 Module XX: <Judul>` berisi:
  - definisi singkat topik
  - bagian "## Apa yang akan kita pelajari?" (numbered list)
  - baris target audience
- Setiap notebook **ditutup** dengan `## Ringkasan`: tabel recap (`# | Topik | Library | Tujuan`) + pointer ke module berikutnya.

### Realign dua cell Inggris ke BI

Per keputusan bahasa (BI primary di seluruh modul), dua intro yang saat ini berbahasa Inggris di-realign ke BI (framing dipertahankan, hanya diterjemahkan + diformat ke template):

1. Module 04 cell 0 — "NLP vs LLMs".
2. Module 06 `02_nvidia_nemo_demo` cell 0 — "Nvidia NeMo: A Powerful Toolkit...".

## Constraint: markdown-only, code tidak disentuh

- HANYA insert/replace markdown cells. TIDAK mengubah, menjalankan, atau menyusun-ulang code cell.
- Module 05: intro markdown **di-prepend** di index 0 (di atas `pip install` yang ada) — menambah struktur tanpa memindah code.
- Validasi otomatis tiap notebook: hash semua code-cell `source` sebelum & sesudah edit harus identik.

## Per-Notebook Cell Plan

### Module 04 — `04_llm/01_llm_basics.ipynb`
State: 16 cells (1 md / 15 code) → target ±8 md.

| Posisi | Cell baru/realign | Isi |
|---|---|---|
| cell 0 | realign intro | BI: NLP vs **LLM**, perbedaan scope/data/kemampuan/aplikasi + "Apa yang akan dipelajari" |
| sebelum §1 (inference) | theory | **LLM inference** dengan model kecil (OPT-350m / TinyLlama), kenapa float16 |
| sebelum §2 (pipeline) | theory | HuggingFace **`pipeline`** (high-level) vs load model+tokenizer manual |
| sebelum §3 (classification) | theory | **Text classification** dengan transformer pre-trained |
| sebelum §4 (chatbot) | theory | **Chatbot** sederhana: prompt + conversation loop |
| sebelum §5 (few-shot) | theory | **Few-shot learning** — beri contoh dalam prompt, tanpa training |
| sebelum §6 (evaluation) | theory | **Model evaluation** — metrik kualitas generasi |
| akhir | Ringkasan | tabel recap + pointer ke Module 05 (RAG) |

### Module 05 — `05_rag/01_rag_fundamentals.ipynb`
State: 9 cells (1 md stub / 8 code) → target ±7 md. **Paling banyak kerja struktural** (belum ada title cell).

| Posisi | Cell baru | Isi |
|---|---|---|
| prepend index 0 | 📚 intro | Apa itu **RAG** = **Retrieval** + **Generation**; kenapa (LLM halusinasi, knowledge cutoff); alur ringkas + target audience |
| sebelum embedding load | theory | **Embeddings** & **SentenceTransformer** (`all-MiniLM-L6-v2`) — teks → vektor |
| sebelum FAISS index | theory | **FAISS** — index vektor untuk similarity search cepat (`IndexFlatL2`) |
| sebelum retrieve fn | theory | Fungsi retrieval: query → embed → top-k konteks |
| sebelum RAG test | theory | Loop **RAG**: retrieve → augment prompt → generate |
| akhir | Ringkasan | tabel recap + pointer ke Module 06 |

Catatan: stub markdown 9-char "inference" (cell 7) diganti dengan theory cell yang proper.

### Module 06 — NVIDIA Ecosystem (3 notebook)

**`01_nvidia_ecosystem_basic.ipynb`** — 3 cells (0 md) → ±4 md.
- 📚 intro: ekosistem **GPU**/**CUDA** NVIDIA untuk deep learning.
- theory sebelum GPU check: `nvidia-smi`, `torch.cuda`, memory.
- Ringkasan.

**`02_nvidia_nemo_demo.ipynb`** — 28 cells (13 md, header pendek) → di-*enrich*.
- Realign intro (cell 0) ke BI.
- Expand header pendek jadi 1–3 baris theory: NER, QnA, Text Classification, Punctuation & Capitalization, Machine Translation, TTS.
- Fix typo "avilable" → "available" (cell 12).
- Tambah Ringkasan.

**`03_nvidia_rag.ipynb`** — 10 cells (3 md) → ±6 md.
- 📚 intro: RAG di atas stack NVIDIA (GPU-accelerated, `faiss-gpu`).
- theory per section.
- Ringkasan. Catat: GPU-heavy.

## Proses & Safety

1. Backup tiap notebook ke `/tmp/` sebelum edit (`cp nb.ipynb /tmp/<name>.bak.ipynb`).
2. Edit via Python script: load JSON → insert markdown cells **reverse index order** (index tertinggi dulu) → `json.dump` write back.
3. Helper `md(source)` → markdown cell dict (lihat handoff doc).
4. Validasi per notebook: re-parse JSON valid + assert hash semua code-cell `source` tidak berubah.
5. Commit per-module — 3 commit: `feat(module04)`, `feat(module05)`, `feat(module06)` — granularity sama seperti Module 03.

## Catatan diff

`json.dump` akan reformat seluruh file → git diff terlihat banyak insertion/deletion noise. Ini normal (didokumentasikan di handoff). Konten code fungsional tetap sama (dijamin oleh validasi hash).

## Out of scope (tidak dikerjakan sesi ini)

- Slide deck Module 04/05/06 (kerjaan terpisah berikutnya).
- Restrukturisasi / menjalankan / memperbaiki code cell.
- Cheat sheet PDF, mermaid diagram, quiz.
