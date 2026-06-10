# Spec: Notebook "NLP on Steroids" — NLP Klasik di GPU dengan NVIDIA RAPIDS

**Tanggal:** 2026-06-11
**Modul:** 03_nlp_fundamentals (Day 03, bootcamp NCA-GENL Batch 4)
**Status:** Disetujui (brainstorming 2026-06-11)

## Tujuan

Notebook hands-on resmi baru untuk Day 03 yang menunjukkan bahwa seluruh alur NLP
klasik yang dipelajari di notebook 01 (preprocessing → vektorisasi → klasifikasi)
dapat dijalankan di GPU menggunakan stack NVIDIA RAPIDS — tanpa mengubah konsep,
hanya mengganti hardware. Sekaligus menjadi jembatan kurikulum ke Day 04 (LLM)
dan Day 06 (ekosistem NVIDIA).

## Keputusan desain (hasil klarifikasi)

| Aspek | Keputusan |
|---|---|
| Posisi | Notebook hands-on resmi, dijalankan 22 peserta di Colab T4 |
| Bahasa | Dual: `02_nlp_on_steroids_id.ipynb` + `02_nlp_on_steroids_en.ipynb` |
| Cakupan | Preprocessing (cuDF/nvtext) + vektorisasi & ML (cuML) + teaser NeMo Curator |
| Dataset | Wikipedia ID (HF `wikimedia/wikipedia`, config `20231101.id`, streaming, sampel ±150k paragraf) untuk preprocessing & dedup; SmSA IndoNLU (raw GitHub, URL di-pin ke commit) untuk klasifikasi sentimen |
| Paket | Lengkap: notebook ID+EN, 3 frame slide + 1 figure, +3 soal quiz (15→18), +2 kartu cheatsheet |
| Pendekatan | C — "Sulap dulu, bedah kemudian" (zero-code-change dulu, lalu API native) |

## Struktur notebook (5 babak)

### Babak 0 — Setup & sanity check
- `!nvidia-smi`, verifikasi `import cudf, cuml` (asumsi: pre-installed di Colab GPU runtime).
- Sel fallback `pip install` disiapkan dalam bentuk komentar, hanya dipakai jika asumsi gagal.

### Babak 1 — Sulap: kode lama, GPU baru (zero code change)
- Kendala teknis: `cudf.pandas` harus dimuat sebelum `import pandas`, sehingga
  demo before/after tidak bisa dalam satu kernel.
- Mekanisme: tulis script pipeline kecil (load SmSA → TF-IDF → LogisticRegression,
  pola sama dengan notebook 01) via `%%writefile pipeline.py`, lalu:
  - `!python pipeline.py` → baseline CPU (pandas + sklearn)
  - `!python -m cudf.pandas -m cuml.accel pipeline.py` → GPU, nol perubahan kode
- Output yang ditunjukkan: akurasi sama, waktu eksekusi jauh berbeda.

### Babak 2 — Bedah nvtext: preprocessing skala Wikipedia
- Load sampel ±150k paragraf Wikipedia ID (streaming agar tidak download penuh).
- Operasi nvtext via accessor `.str` cuDF: `tokenize`, `token_count`, `ngrams`,
  normalisasi — masing-masing dengan benchmark berpasangan vs pandas (`%%time`).

### Babak 3 — Dedup near-duplicate dengan MinHash
- Temukan paragraf nyaris-duplikat di sampel Wikipedia menggunakan minhash GPU.
- Framing: ini langkah nyata penyiapan korpus training LLM → pengait ke Babak 5.

### Babak 4 — cuML native: TF-IDF + klasifikasi sentimen SmSA
- `cuml.feature_extraction.text.TfidfVectorizer` + `cuml.LogisticRegression`.
- Bandingkan akurasi & waktu vs sklearn (versi API-asli dari Babak 1).
- SmSA di-load dari URL raw GitHub IndoNLU yang di-pin ke commit hash.

### Babak 5 — Teaser NeMo Curator + jembatan kurikulum
- Markdown: skala 150k paragraf (notebook ini) vs miliaran dokumen (NeMo Curator)
  untuk penyiapan data training LLM.
- Sambungan eksplisit ke Day 04 (LLM) dan Day 06 (ekosistem NVIDIA).
- Ringkasan + 2–3 latihan mandiri.

## Materi pendukung

### Slides (`03_nlp_fundamentals/slides/module03_slides.tex`)
- 3 frame baru di bagian akhir:
  1. "RAPIDS: NLP Klasik di GPU" — posisi RAPIDS di stack NVIDIA
  2. "Zero Code Change" — snippet `python -m cudf.pandas`
  3. "Seberapa Cepat?" — bar chart speedup
- Figure baru: `slides/figures/gen_rapids_speedup.py` → `rapids_speedup.pdf`
  (tema gelap mengikuti template figure course; angka diisi dari hasil
  smoke-test Colab asli, bukan klaim marketing).
- Recompile dengan XeLaTeX (bukan PDFLaTeX).

### Quiz (`nlp-fundamentals-quiz.html`)
- +3 soal (15 → 18): konsep zero-code-change, kapan GPU menguntungkan
  (ukuran data / overhead transfer), peran minhash dedup.
- Pola penulisan mengikuti Q13–Q15; test Puppeteer yang ada diperbarui untuk 18 soal.

### Cheatsheet (`nlp-fundamentals-cheatsheet.html`)
- +2 kartu: "RAPIDS untuk NLP" (cuDF/nvtext/cuML) dan "Zero Code Change"
  (`%load_ext cudf.pandas`, `cuml.accel`).
- Regenerate PDF.

## Risiko & mitigasi

| Risiko | Mitigasi |
|---|---|
| cuML/`cuml.accel` ternyata tidak pre-installed di Colab | Smoke-test di Colab T4 sebelum hari-H (wajib); sel fallback pip |
| Download Wikipedia lambat saat 22 peserta serentak | Streaming + cap 150k paragraf; cadangan: pre-sample ke parquet di Google Drive course |
| API minhash berubah antar versi cuDF | Smoke-test menentukan versi; kode defensif dengan pesan error jelas |
| URL SmSA berubah | Pin ke commit hash di URL raw GitHub |

## Urutan pengerjaan & kriteria selesai

1. Notebook ID → notebook EN (terjemahan setia, bukan parafrase).
2. Smoke-test di Colab T4 oleh instruktur; hasil benchmark dicatat.
3. Figure & slide "Seberapa Cepat?" diisi angka asli hasil smoke-test → recompile.
4. Quiz +3 soal, cheatsheet +2 kartu, test Puppeteer diperbarui dan lulus.
5. Commit per komponen (notebook / slides / quiz+cheatsheet), tanpa footer atribusi.
6. Upload ke folder Google Drive course.

Selesai berarti: kedua notebook jalan penuh di Colab T4 tanpa error, slide
ter-compile, test Puppeteer quiz lulus, semua file ter-upload ke Drive.
