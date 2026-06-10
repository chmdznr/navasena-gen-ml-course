# Design: Revamp Module 03 NLP — Bug Fix + Jembatan "Kata ke Vektor"

**Tanggal**: 2026-06-10
**Status**: Approved (user, 2026-06-10 — "oke, aku setuju, silakan eksekusi")
**Dasar**: audit 3-agen atas M03 (notebook EN+ID, deck, cheatsheet, quiz) + analisis kurikulum M03→M04→M05.

## Masalah

1. **Notebook EN gagal di Colab 2026**: NLTK data names usang (`punkt` → harus `punkt_tab`; `averaged_perceptron_tagger` → `_eng`).
2. **Klasifikasi EN rusak metodologis**: fitur dari FreqDist satu kalimat Section 1 (bukan corpus movie_reviews) + tanpa shuffle → test set 100% negatif, akurasi menyesatkan.
3. **Gap kurikulum**: tidak ada subword tokenization (dipakai M04) dan embeddings/vector similarity (fondasi M05) di mana pun sebelum modul yang memakainya.
4. **Asimetri EN/ID**: task aplikatif (NER/sentiment/klasifikasi) hanya di EN; notebook ID berhenti di preprocessing; IndoBERT/HF absen.
5. **Klaim Sastrawi salah** di deck (terdaftar sbg library notebook ID padahal tidak dipakai) + technical debt notebook ID (cell duplikat, print-as-markdown, klaim lemmatizer "memperhatikan konteks").

## Keputusan Batas Materi (disepakati user)

- **M03 = representasi teks**: kata → token (termasuk subword/BPE) → vektor (TF-IDF → embeddings → similarity). Hands-on.
- **M04 = arsitektur & pemakaian**: attention, transformer block, generasi/sampling (teori deck Act 2) + pemakaian LLM. TIDAK dipindah.
- Benturan diredakan dengan reframing 2 frame M04 jadi recap/pendalaman (lihat E).

## Kaidah Bahasa (WAJIB, sama dgn paket M02)

Bahasa Indonesia engineering luwes; istilah serapan umum TIDAK diterjemahkan (token, embedding, similarity, vector, pipeline, subword, dst). Larangan: "ruang laten", "kebisingan", "pembangkit", "pembeda", "pelatihan berlawanan", "penyemat/penyematan". Kalimat tidak ambigu. Sapaan "kamu", gaya konsisten notebook existing.

## Komponen

### A. Fix notebook EN (`03_nlp_fundamentals/01_nlp_fundamentals_en.ipynb`)

1. Setup: `punkt`→`punkt_tab`, `averaged_perceptron_tagger`→`averaged_perceptron_tagger_eng`; tambah `wordcloud` ke pip install.
2. Klasifikasi: `random.seed(42)` + `random.shuffle(documents)`; fitur dari top-2000 kata movie_reviews (`nltk.FreqDist(movie_reviews.words())`), bukan `fdist` Section 1; +1 markdown menamai pendekatan "bag-of-words features" sebagai representasi vektor sederhana (jembatan ke materi vektor di notebook ID).
3. Word frequency: stopword + punctuation removal sebelum FreqDist & wordcloud (+1 kalimat md).
4. Catatan md TextBlob hanya akurat untuk English.
5. Hapus komentar sisa `# prompt: ...`.

### B. Fix notebook ID (`03_nlp_fundamentals/01_nlp_fundamentals_id.ipynb`)

1. Cell duplikat Section 7 (re-import + re-analisis ~3KB): dirampingkan — hapus duplikasi, reuse objek yang sudah ada.
2. Cell tips `print("""...""")` → markdown.
3. Klaim lemmatizer "dengan memperhatikan konteks" → "berbasis kamus & aturan (per kata, tanpa konteks kalimat)".
4. Ringkasan: Sastrawi bukan "library utama" → nlp-id; Sastrawi disebut sebagai opsi historis stemming (catatan: tidak di-maintain sejak ~2016).
5. Contoh motivasi phrase tokenization disesuaikan ke frasa nomina (mis. "Universitas Indonesia") sesuai perilaku nlp-id.
6. Catatan setelah pip install: nlp-id mem-pin dependency exact → Colab bisa minta restart runtime.
7. Md pengakuan pipeline tidak menormalkan slang ("Gw suka bgt...") → teaser ke section IndoBERT.

### C. Section baru notebook ID (inti penambahan)

- **Section 8 — "Dari Kata ke Vektor"** (±8 cell): TF-IDF (`TfidfVectorizer`) pada kalimat Indonesia → cosine similarity gagal menangkap sinonim → `sentence-transformers` `paraphrase-multilingual-MiniLM-L12-v2` → heatmap similarity (sinonim berdekatan) → mini semantic search 10 dokumen fakta Indonesia (cosine + argsort, TANPA FAISS) → md jembatan eksplisit ke Module 05 RAG.
- **Section 9 — "Tokenisasi ala LLM (Subword)"** (±3 cell): `AutoTokenizer` (gpt2 + multilingual) vs tokenizer kata pada kata berimbuhan ("mempelajari" → subword) + perbandingan jumlah token EN vs ID kalimat semakna + md soal implikasi biaya per-token; jembatan eksplisit ke Module 04.
- **Section 10 — "Sentiment Bahasa Indonesia dengan IndoBERT"** (±3 cell): `transformers.pipeline("sentiment-analysis")` model Indonesia siap pakai dari HF Hub (ungated; verifikasi ketersediaan saat implementasi, kandidat: `w11wo/indonesian-roberta-base-sentiment-classifier`), diuji pada teks gaul yang sudah ada di notebook → menutup asimetri EN/ID.
- Update "Apa yang akan kita pelajari?" + Ringkasan sesuai section baru.

### D. Fix deck M03 (`03_nlp_fundamentals/slides/module03_slides.tex`) + Act baru

Fixes: hapus Sastrawi dari daftar library notebook ID & dari `pip install` frame Setup; deskripsi Module 06 di frame Lanjutan ("RAG" → TensorRT/NeMo/GPU); disclaimer tagset (UD ilustratif vs nlp-id NNP/VB/NN); reword klaim imbuhan-menyulitkan-tokenisasi → konteks lemmatization; catatan kecil spaCy melabeli kota sbg GPE.
Act baru: **Act 5 "Dari Kata ke Vektor"** (±4 frame: divider; BoW/TF-IDF→embedding; subword tokenization; IndoBERT & ekosistem Indonesia + jembatan M04/M05). Act Ringkasan lama dinomori ulang jadi Act 6. Recompile 0 overfull baru.

### E. Reframe 2 frame deck M04 (`04_llm/slides/module04_slides.tex`)

1. Frame "Tokenization: Teks Menjadi Angka" → framing recap: "Ingat dari Module 03 ..." (konten inti dipertahankan).
2. Frame "Embedding & Transformer" → bedakan eksplisit sentence embedding statis (yang dilihat peserta di M03) vs contextual embedding di dalam transformer.
Recompile M04, tidak menambah overfull.

### F. Cheatsheet + quiz M03

- Cheatsheet: +2 kartu (Dari Kata ke Vektor; Subword & IndoBERT) + 3-4 glossary (embedding, cosine similarity, subword/BPE, TF-IDF); WAJIB tetap 1 halaman.
- Quiz: +3 soal (vektor/similarity, subword tokenization, IndoBERT) → 15 soal; header & scorebar di-update; uji grading puppeteer.

## Validasi

- Smoke-run lokal via docling env (`uv pip install nltk textblob wordcloud sentence-transformers` — preseden install ada): EN end-to-end versi mini; ID hanya CELL BARU (TF-IDF/ST/AutoTokenizer/IndoBERT) — JANGAN install nlp-id ke docling env (pin exact numpy/sklearn bisa merusak env user); cell nlp-id cukup validasi sintaks.
- Source format rule notebook (newline guard) seperti paket M02.
- Deck: 2-pass xelatex, tidak ada overfull baru (M03 baseline saat ini bersih; M04 cek baseline dulu).
- Cheatsheet 1 halaman; quiz uji fungsional puppeteer.
- QC akhir: grep kaidah bahasa, konsistensi istilah lintas-artefak, branding.

## Di Luar Scope

- Memindah teori attention/transformer ke M03 (tetap di M04).
- Training word2vec/GloVe dari nol; fine-tuning IndoBERT; FAISS di M03 (itu porsi M05).
- Restruktur ulang pembagian EN/ID (EN tetap ada; hanya diperbaiki).
- Lokalisasi knowledge base M05 (dicatat sebagai kandidat pekerjaan terpisah).
- Menghapus `03_nlp_fundamentals/llm_architecture.mmd` (orphan, untracked — menunggu keputusan user).

## Kriteria Sukses

- Notebook EN jalan end-to-end di Colab (NLTK names benar), klasifikasi memakai shuffle + fitur movie_reviews.
- Notebook ID: 3 section baru jalan; tidak ada cell duplikat; klaim akurat.
- Deck M03 ter-update (fix + Act baru) 0 overfull baru; M04 2 frame reframed, compile bersih.
- Cheatsheet 1 halaman; quiz 15 soal lolos uji puppeteer.
- 0 anti-pattern bahasa; konsistensi istilah lintas-artefak.
