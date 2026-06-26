# Design: Penyelesaian Slide Deck Sertifikasi NVIDIA NCA-GENL

**Tanggal:** 2026-06-26
**Status:** Disetujui (siap eksekusi)
**File target:** `docs/nca_genl_slides/nca_genl_slides.tex`

## Konteks

Sudah ada draft deck (kemungkinan dari Gemini) berisi 31 frame dalam Bahasa
Indonesia, terorganisir menjadi 7 "Act" yang memetakan blueprint resmi ujian
NCA-GENL. Bobot domain pada draft sudah **terverifikasi cocok** dengan halaman
resmi NVIDIA:

| Domain | Bobot resmi |
|---|---|
| Core Machine Learning & AI Knowledge | 30% |
| Software Development | 24% |
| Experimentation | 22% |
| Data Analysis & Visualization | 14% |
| Trustworthy AI | 10% |

Masalah: draft **gagal compile** (banyak `&` tidak di-escape di judul frame,
`\acttitle{}`, `\node{}`, `\subtitle{}`) sehingga PDF hanya terbentuk sebagian.
Selain itu ada beberapa klaim faktual yang tidak akurat / tidak resmi.

Audiens: peserta bootcamp yang akan mengikuti ujian sertifikasi
(exam-prep depth).

## Sumber otoritatif (sudah di-index)

1. Halaman resmi NVIDIA NCA-GENL (blueprint, format, harga, validitas, kursus DLI).
2. PDF referensi "Section 1: Core ML & AI Knowledge" (586 halaman) — verifikasi
   klaim teknis Domain 1.

## Lingkup pekerjaan

### A. Perbaikan compile LaTeX (mekanis)
- Escape semua `&` mentah (≈20 kemunculan) di judul/teks non-tabular → `\&`.
- Konversi sisa artefak markdown (`*kernel*`, emphasis `*...*`) ke LaTeX
  (`\emph{}` / `\texttt{}`).
- Compile bersih dengan `xelatex` (compiler resmi repo), 0 error, semua frame
  ter-render.

### B. QC konten & koreksi faktual
Diverifikasi terhadap sumber otoritatif:
- **Jumlah soal:** "50 Soal" → **"50–60 soal pilihan ganda"** (3 lokasi: title
  slide, tabel format, dan teks terkait); sesuaikan estimasi waktu per soal.
- **Skor kelulusan:** hapus klaim "70% / minimal 35 dari 50". NVIDIA tidak
  mempublikasikan passing score. Ganti dengan catatan jujur:
  *"NVIDIA tidak mempublikasikan skor kelulusan resmi; fokus pada penguasaan
  seluruh domain."*
- **Prasyarat:** sesuaikan ke teks resmi — "pemahaman dasar tentang generative
  AI dan LLM" (bukan "tidak ada prasyarat").
- Spot-check klaim teknis Domain 1 (transformer/attention, tipe ML, RLHF)
  terhadap PDF referensi; koreksi bila ada kekeliruan.
- Sanity-check klaim Domain 2–5 terhadap daftar topik resmi.

### C. Penyelarasan house style
- Verifikasi macro `\acttitle`, `\takeaway`, `\certnote` konsisten dengan
  konvensi module06.
- Pastikan tidak ada tabel/teks low-contrast (riwayat module06 pernah ada baris
  tabel tak terbaca).

### D. Tambahan nilai exam-prep
- Tambah harga ($125), validitas (2 tahun), recert via retake ke slide format.
- Tambah slide **"Sumber Belajar Resmi"**: memetakan tiap domain ke kursus
  NVIDIA DLI (nama, durasi, biaya) dari blueprint.

### E. Build, verifikasi, deliver
- Hasilkan PDF final; konfirmasi jumlah frame + tidak ada overfull box serius;
  kirim PDF ke user; commit.

## Di luar lingkup (YAGNI)
- Tidak membuat cheatsheet/handout terpisah (bisa menyusul bila diminta).
- Tidak mengubah arsitektur 7-Act (sudah benar).
- Tidak menerjemahkan ke bahasa lain.

## Kriteria sukses
- `xelatex` compile 0 error.
- Semua klaim faktual cocok dengan halaman resmi NVIDIA.
- Tidak ada angka passing score yang dibuat-buat.
- Deck terbaca jelas dengan house style Navasena.
