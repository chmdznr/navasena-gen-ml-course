# Design Spec: Module 02 (Deep Learning) Slide Deck

> **Tanggal**: 2026-06-01
> **Status**: Approved (brainstorming) → siap build
> **Scope**: Satu slide deck `02_deep_learning_fundamentals/slides/module02_slides.tex` (~33 frame, 5 act) mengikuti template Module 03/04. BI primary, istilah teknis DL lebih banyak English.

## Tujuan

Module 02 punya 5 notebook besar yang sudah direvisi (markdown lengkap, struktur "Bagian N", section "Peran NVIDIA" + Kesimpulan). Slide deck menyatukan kelima topik jadi satu presentasi modul yang koheren; tiap act memetakan ke satu notebook sehingga deck juga jadi peta baca.

## Template & Konvensi (warisan Module 03/04)

- `\documentclass[aspectratio=169,t]{beamer}`, xelatex, FiraSans, palet Navasena dark.
- `\acttitle{N}{Judul}{Subjudul}`, footline **"Deep Learning Fundamentals"** (tanpa NCA-GENL).
- `\title{Deep Learning Fundamentals}`, `\subtitle{Navasena Gen-ML Course}`, `\author{Navasena}`.
- Preamble disalin VERBATIM dari Module 03/04.

### LaTeX gotchas (wajib dipatuhi — dari Module 04)
- Frame dengan `lstlisting` HARUS `\begin{frame}[fragile]{...}`.
- Tabel di kolom sempit: `\setlength{\tabcolsep}{3pt}` + lebar kolom `p{}` jumlah < linewidth.
- TikZ vertikal: `scale=` + `every node/.style={transform shape}`; JANGAN `\resizebox{\linewidth}{!}` untuk diagram tinggi (height membengkak → overfull vbox).
- Target **0 overfull box** (hook `latex-overfull-guard.sh`).

## Struktur (≈33 frame: title + 5 act divider + 24 konten + ringkasan + lanjutan + closing)

**Title** — Module 02 / Deep Learning Fundamentals / "Neural Networks dengan TensorFlow".

**Act 1 — Neural Network Pertama** (nb01) — 5 konten
1. Apa itu Neural Network? — neuron, **layer**, **weights**, intuitif (diagram `nn_layers`).
2. Dataset & Normalisasi — **Fashion MNIST**, kenapa normalisasi pixel ke 0-1.
3. Membangun Model — `Sequential`, `Dense`, `Flatten`; jumlah **parameter**.
4. Training & Loss — `compile` (optimizer/loss/metrics), **epoch**, kurva **loss**/accuracy.
5. Evaluasi — accuracy test, **confusion matrix**, prediksi probabilitas per kelas.

**Act 2 — Aktivasi & Optimizer** (nb02) — 5 konten
6. Kenapa Perlu Aktivasi — **non-linearity**; tanpa aktivasi = tetap linear.
7. Fungsi Aktivasi Utama — **ReLU**, **Sigmoid**, **Tanh**, **LeakyReLU** (figure `activations`).
8. Softmax untuk Klasifikasi — **logits** → probabilitas (jumlah = 1).
9. Panduan Memilih Aktivasi — ReLU default hidden, Sigmoid output biner, Softmax multiclass.
10. Optimizer — **SGD**, **Adam**, **RMSprop**; strategi update **gradient**.

**Act 3 — CNN & Transfer Learning** (nb03) — 5 konten
11. Apa itu CNN — **convolution**, **filter/kernel**, **pooling** (diagram `cnn_flow`).
12. Membangun CNN — `Conv2D`, `MaxPool`, **CIFAR-10**, perubahan dimensi.
13. Apa yang Dipelajari CNN — **feature maps**, hierarki fitur (tepi → bentuk → objek).
14. Transfer Learning — **ResNet50** pretrained, **freeze** layer, tambah head baru (TikZ freeze/train).
15. CNN dari Nol vs Transfer Learning — perbandingan akurasi & efisiensi.

**Act 4 — RNN & LSTM** (nb04) — 5 konten
16. Data Berurutan & RNN — **sequential data**, **hidden state**, "memori" (TikZ RNN unroll).
17. RNN: Prediksi Sinus — praktik regresi deret waktu.
18. Masalah RNN → LSTM — **vanishing gradient**; **gates** (forget/input/output).
19. LSTM: Sentimen IMDB — **embedding**, **padding** panjang review.
20. Variasi RNN — **SimpleRNN** vs **LSTM** vs **GRU**.

**Act 5 — Akselerasi GPU NVIDIA** (nb05) — 4 konten
21. Kenapa DL Butuh GPU — paralelisme ribuan core, operasi matriks.
22. Benchmark CPU vs GPU — training & inference speedup (figure `gpu_benchmark`).
23. Mixed Precision Training — **FP16**/**FP32**, hemat memori + lebih cepat.
24. NVIDIA Ecosystem & TensorRT — **CUDA**, **cuDNN**, **TensorRT** optimasi inference.

**Wrap** — 25. Ringkasan Modul 02 (tabel 5 topik). 26. Lanjutan: Module 03 (NLP). Closing "Terima Kasih".

## Figures

| File | Tipe | Isi |
|------|------|-----|
| `figures/gen_activations.py` | Matplotlib | kurva ReLU/Sigmoid/Tanh/LeakyReLU di satu axis (tema gelap) |
| `figures/gen_gpu_benchmark.py` | Matplotlib | bar CPU vs GPU (training + inference), speedup ilustratif |
| `figures/nn_layers.mmd` | Mermaid | input layer → hidden layer(s) → output layer |
| `figures/cnn_flow.mmd` | Mermaid | image → Conv → Pool → Conv → Pool → Flatten → Dense → class |
| (inline TikZ) | — | RNN unrolled (hidden state), softmax bars, transfer-learning freeze vs train |

Render: matplotlib via docling python → `figures/*.pdf`; mermaid via `mmdc -s 3 -b transparent` → `figures/*.png`. Output gelap (facecolor #1A1A2E, aksen #76B900).

## Build

`02_deep_learning_fundamentals/slides/build.sh` = salinan Module 03/04. Pre-clean aux (zsh nullglob), 2-pass xelatex. Commit: `.tex`, `build.sh`, `figures/gen_*.py`, `figures/*.mmd`, figure outputs (`.png`/`.pdf`), final PDF.

## Out of scope

- Mengubah notebook Module 02 (sudah direvisi).
- Deck Module 05/06.
- Cheat sheet PDF.

## Risiko & mitigasi

- **Deck panjang (33 frame)** → jaga tiap frame padat (bullet pendek, footnotesize), banyak two-column.
- **Akurasi** (CNN/RNN/LSTM untuk pemula): jaga intuitif; review akurasi terhadap kode notebook.
- **Overfull**: patuhi gotchas LaTeX di atas; iterasi compile→fix sampai bersih.
