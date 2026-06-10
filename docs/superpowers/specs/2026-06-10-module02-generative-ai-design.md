# Design: Materi Generative AI untuk Module 02 (Deep Learning Fundamentals)

**Tanggal**: 2026-06-10
**Status**: Approved (user, 2026-06-10)

## Latar Belakang & Tujuan

Module 02 saat ini 100% diskriminatif: klasifikasi (NN, CNN), prediksi sekuensial
(RNN/LSTM), dan benchmark GPU. Peserta kemudian lompat ke Module 04 yang langsung
membahas LLM (model generatif untuk teks) tanpa pernah diperkenalkan konsep
"neural network yang menghasilkan sesuatu".

**Tujuan**: menambah materi survey generative AI di Module 02 sebagai jembatan
konseptual — peserta melihat dan melatih sendiri model yang *menghasilkan* data
baru, bukan hanya mengklasifikasi/memprediksi. Saat ketemu LLM di Module 04,
mereka sudah punya mental model "generator yang belajar distribusi data".

## Keputusan Desain (sudah disepakati user)

1. **Scope**: survey generatif luas dalam SATU notebook — autoencoder → GAN →
   peta dunia generatif modern (diffusion, GPT). Bukan deep-dive GAN saja.
2. **Porsi training**: autoencoder dilatih penuh (~10 epoch, ~2-3 menit, stabil);
   DCGAN dilatih versi mini (~10-15 epoch, ~10 menit di T4) — hasil digit kasar
   tapi terlihat "lahir dari noise". Diffusion/LLM teori saja.
   Total runtime target: ~15 menit di Colab T4.
3. **Artefak ikut di-update SEMUA**: slide deck (+Act 6), cheatsheet (+kartu
   konsep & glossary), quiz (+3 soal).
4. **Dataset**: MNIST (bukan Fashion-MNIST) — digit recognizable lebih cepat
   dengan epoch sedikit, paling klasik untuk DCGAN. Seed di-fix supaya hasil
   reproducible.

## Kaidah Bahasa (WAJIB untuk semua konten baru)

Gunakan Bahasa Indonesia untuk Engineering yang luwes/fleksibel:

- Istilah serapan bahasa Inggris yang memang umum dipakai engineer TIDAK perlu
  ditranslasikan (contoh: training, layer, latent space, noise, epoch, loss,
  generator, discriminator, sampling, output).
- Frasa/kalimat tidak boleh ambigu atau aneh didengar (hindari over-translation
  seperti "ruang laten", "pelatihan berlawanan", "kebisingan acak").
- Konsisten dengan gaya notebook M02 yang sudah ada: sapaan "kamu", theory cells
  markdown di antara code cells, analogi sehari-hari.

## Komponen

### 1. Notebook baru — `02_deep_learning_fundamentals/06_generative_ai_intro.ipynb`

Arc cerita: **"dari mengenali → mencipta"**. Target Colab T4, TensorFlow/Keras
(konsisten dengan stack M02).

| Bagian | Isi | Training? |
|---|---|---|
| Intro | Diskriminatif vs generatif — selama ini model menebak label; sekarang model menghasilkan data baru. Bridge eksplisit dari notebook 01-05. | — |
| Setup | Pattern standar M02: `!nvidia-smi`, cek GPU TensorFlow | — |
| A. Autoencoder | Encoder → latent space → decoder di MNIST. Latih ~10 epoch (~2-3 menit). Visual: rekonstruksi digit + interpolasi latent space (morphing antar digit) + sampling acak dari latent → hasil blur → motivasi alami menuju GAN | ✅ penuh |
| B. GAN (DCGAN) | Analogi pemalsu vs polisi. DCGAN Keras di MNIST, latih ~10-15 epoch (~10 menit). Grid digit hasil generator tiap beberapa epoch — noise pelan-pelan jadi angka | ✅ mini |
| C. Peta dunia generatif | Teori saja: kenapa GAN rewel (mode collapse, singkat) → VAE → diffusion (Stable Diffusion) → autoregressive transformer (GPT) — jembatan eksplisit ke Module 04 LLM | — |
| Penutup | Ringkasan + teaser "di Module 04 kamu akan pakai model generatif untuk teks" | — |

Catatan implementasi:
- Seed fixed (`tf.random.set_seed` + numpy) untuk reproducibility.
- DCGAN pakai arsitektur kecil ala tutorial Keras resmi (Conv2DTranspose
  generator, Conv2D discriminator) yang sudah terbukti konvergen di MNIST
  dengan epoch sedikit.
- Konten di-draft via workflow (draft → review akurasi ∥ gaya → sintesis),
  lalu di-apply dengan guard hash code-cell seperti pola sesi sebelumnya.

### 2. Slide deck — Act 6 "Generative AI" (~5 frame baru)

File: `02_deep_learning_fundamentals/slides/module02_slides.tex` (29 → ~34 frame).
Ikuti template Navasena dark + semua gotcha LaTeX terdokumentasi (frame lstlisting
wajib `[fragile]`, tabcolsep 3pt di kolom sempit, TikZ pakai `scale=` +
`transform shape`, nama node TikZ tanpa titik desimal, mermaid portrait di-size
by `height=`).

1. `\acttitle{6}{Generative AI}{...}` — divider
2. Diskriminatif vs Generatif (TikZ perbandingan dua arah panah)
3. Autoencoder & latent space (figure interpolasi, matplotlib dark)
4. GAN: adversarial game (diagram generator↔discriminator + grid hasil DCGAN)
5. Peta generatif modern: GAN → VAE → Diffusion → GPT, menuju Module 04

Figures baru di `slides/figures/`:
- `gen_latent_interp.py` → `latent_interp.pdf` (matplotlib dark)
- `gen_gan_grid.py` → `gan_grid.pdf` (pre-generated saat development, BUKAN
  latih ulang saat build — simpan hasil sebagai data/gambar statis)
- `gan_architecture.mmd` → `gan_architecture.png` (mmdc -s 3)
- Update `build.sh` untuk men-chain generator baru.

### 3. Cheat sheet — `dl-fundamentals-cheatsheet.{html,pdf}`

- Tambah 1 kartu konsep "Generative AI (AE & GAN)" di grid.
- Tambah 3-4 istilah glossary: generator, discriminator, latent space, diffusion.
- WAJIB tetap 1 halaman A4 — kalau penuh, kompres kartu lain sedikit.
- Render ulang via pipeline headless Chrome yang sama.

### 4. Quiz — `dl-fundamentals-quiz.html`

- Tambah 3 soal generatif (total 15): konsep diskriminatif-vs-generatif,
  peran generator/discriminator, posisi diffusion/GPT di peta generatif.
- Pakai assembler/template yang sama; answer key diverifikasi reviewer akurasi.
- Grading diuji ulang fungsional via puppeteer (pattern terdokumentasi:
  node + Chrome executablePath + file://, BUKAN Playwright MCP).

## Urutan Kerja

1. Notebook (konten via workflow draft → review akurasi ∥ gaya → apply dengan
   guard hash code-cell) — commit.
2. Slide deck Act 6 + figures + build 2-pass xelatex, 0 overfull — commit.
3. Cheatsheet + quiz, verifikasi 1 halaman + uji grading — commit.
4. QC akhir: compile bersih, konsistensi lintas-artefak (notebook↔deck↔
   cheatsheet↔quiz), kaidah bahasa (0 over-translation).

## Kriteria Sukses

- Notebook jalan end-to-end di Colab T4 dalam ~15 menit, AE menghasilkan
  rekonstruksi + interpolasi yang jelas, DCGAN menghasilkan digit recognizable
  (walau kasar).
- Deck compile 0 error / 0 overfull; mapping notebook↔act tetap 1:1 (6 act,
  6 notebook).
- Cheatsheet tetap 1 halaman; quiz 15 soal semua ter-grade benar.
- Seluruh teks baru lolos cek kaidah bahasa (istilah teknis tidak dipaksakan
  diterjemahkan, kalimat tidak ambigu).

## Di Luar Scope

- VAE hands-on (hanya disebut di teori).
- Diffusion model hands-on (teori + gambar saja).
- Perubahan pada notebook 01-05 M02.
- Video tutorial script.
