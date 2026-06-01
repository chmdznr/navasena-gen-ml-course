# QC Report (qc-laporan): Materi Modul 01–06

> **Tanggal**: 2026-06-02 · **Metode**: pakem qc-laporan 6-check (Factual, Consistency, Flow, Narasi, Redundancy, Completeness) + deterministic LaTeX/structural checks.
> **Scope**: materi yang dibangun sesi ini — 6 slide deck (`.tex`), 6 cheat sheet (HTML), 6 quiz (HTML). Notebook = source of truth untuk factual-check.
> **Hasil**: **0 temuan block-level**, **11 temuan info-level** (polish konsistensi & kejelasan). Materi dinilai siap pakai; temuan info bersifat penyempurnaan.

## Deterministic checks (otomatis)

| Check | Hasil |
|---|---|
| Kompilasi 6 deck (2-pass xelatex) | ✓ semua bersih, 0 error |
| Overfull box (≥2pt hbox / vbox) | ✓ 0 di semua deck |
| Undefined ref `??` di PDF | ✓ 0 di semua 6 PDF |
| `\ref`/`\cite`/`\label` (risiko `??`) | ✓ tidak dipakai (deck tak punya cross-ref) |
| Placeholder (TODO/TBD/FIXME/lorem) | ✓ 0 (1 hit = kata "placeholder" di handoff, false positive) |
| Inventory | ✓ tiap modul: 1 deck + 1 cheatsheet + 1 quiz |
| Footer/theme/branding | ✓ konsisten Navasena, 0 NCA-GENL |

## Temuan per modul (6-check)

### Module 01 — Machine Learning Fundamentals
- **Factual** — 2 info:
  - `[info · cheatsheet]` Pipeline "Alur": "scaling untuk model berbasis jarak (KNN/SVM)" menyiratkan scaling HANYA untuk model jarak, padahal tabel di cheatsheet & deck menandai Logistic Regression & SVM Linear juga butuh scaling. → Longgarkan: "scaling untuk model yang sensitif skala fitur (KNN, SVM, Logistic Regression)".
  - `[info · cheatsheet]` Card GPU: notasi wildcard "sklearn.* → cuml.*" bisa disalahartikan sebagai find-replace string. → Selaraskan ke frasa deck: "ganti baris import (sklearn → cuml), kode fit/predict identik".
- **Consistency / Flow / Narasi / Redundancy / Completeness** — ✓ clean.

### Module 02 — Deep Learning Fundamentals
- **Consistency** — 1 info:
  - `[info · cheatsheet]` Card Sequential menampilkan 1 hidden layer (128→10), deck Slide 4 menampilkan 2 (128→64→10, ~109K params). Keduanya valid tapi beda antar-artefak. → Tambah `Dense(64, relu)` di cheatsheet agar identik dengan deck.
- **Narasi** — 1 info:
  - `[info · cheatsheet]` Komentar kode `Embedding(vocab_size, 64)  # panjang dari input (batch,200)` kabur. → Ganti: `Embedding(vocab_size, 64, input_length=200)  # kata→vektor 64-dim; panjang sequence 200 (padding)`.
- **Factual / Flow / Redundancy / Completeness** — ✓ clean.

### Module 03 — NLP Fundamentals
- **Factual** — 2 info:
  - `[info · cheatsheet]` Card POS: "Penn-Treebank ID" agak longgar; tagset nlp-id (NNP/VB/NN) ≠ label UD coarse (NOUN/VERB) yang dipakai di contoh ilustratif deck. → Longgarkan klaim + beri catatan bahwa label slide bersifat UD/ilustratif sedangkan output nlp-id pakai NN/VB.
  - `[info · cheatsheet]` `Lemmatizer().lemmatize()` menerima/mengembalikan STRING kalimat (lowercase), bukan list token. → Perjelas contoh I/O di cheatsheet.
- **Consistency** — 1 info:
  - `[info · deck/cheatsheet]` Contoh stopword ID berbeda antar lokasi (slide 7 "yang, di, dari, dan"; slide 9 "yang, ini, untuk"; cheatsheet "yang, di, untuk"). → Seragamkan ke satu set.
- **Flow / Narasi / Redundancy / Completeness** — ✓ clean.

### Module 04 — LLM Fundamentals
- **Consistency** — 1 info:
  - `[info · cheatsheet/deck]` Format prompt few-shot beda: deck pakai `Q:`/`A:` (3 contoh), cheatsheet pakai `Question:`/`Answer:`. Notebook asli pakai `Question:`/`Answer:`. → Samakan label (ikuti notebook).
- **Factual / Flow / Narasi / Redundancy / Completeness** — ✓ clean.

### Module 05 — RAG Fundamentals
- **Consistency** — 1 info:
  - `[info · cheatsheet]` Heading "RAG = Retrieve + Augment + Generate" mencampur akronim dengan langkah pipeline (akronim sebenarnya **R**etrieval-**A**ugmented **G**eneration). → Ganti jadi "Pipeline RAG: Retrieve → Augment → Generate" agar akronim & pipeline tak tertukar.
- **Factual / Flow / Narasi / Redundancy / Completeness** — ✓ clean.

### Module 06 — NVIDIA Ecosystem
- **Narasi** — 1 info:
  - `[info · cheatsheet]` TensorRT disebut "alat" di card tapi "Library" di glossary. → Seragamkan istilah (mis. "library/SDK optimasi inference") di kedua tempat + selaras deck.
- **Redundancy** — 1 info:
  - `[info · cheatsheet]` Pipeline bar menambah node "Aplikasi: benchmark, NeMo task, RAG" yang tak ada di stack deck. → Perjelas bahwa "Aplikasi" adalah output di atas stack, bukan lapisan ke-6.
- **Factual / Consistency / Flow / Completeness** — ✓ clean.

## Summary

- **Block-level (wajib): 0.**
- **Info-level (pertimbangkan): 11** — mayoritas konsistensi lintas-artefak (cheatsheet vs deck) & kejelasan pedagogis. Tidak ada informasi yang salah-fatal.
- Distribusi: M01=2, M02=2, M03=3, M04=1, M05=1, M06=2.
- Kategori paling sering: **consistency** (5) & **factual-nuance** (4); narasi (2), redundancy (1). Flow & completeness bersih di semua modul.
- Catatan: deck adalah referensi yang sudah di-QC; sebagian besar fix diarahkan ke cheatsheet/quiz agar selaras deck. Beberapa item menyentuh deck (M03 stopword, M04 few-shot label) — opsional.
