# Module 03 NLP Revamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Perbaiki bug notebook EN, rapikan notebook ID, tambah jembatan "kata ke vektor" (TF-IDF→embedding→similarity, subword tokenization, IndoBERT) di M03, reframe 2 frame M04, update deck/cheatsheet/quiz M03.

**Architecture:** Edit notebook via Python JSON scripts dengan newline-format guard (pelajaran paket M02). Section baru hanya di notebook ID. Deck M03 dapat Act 5 baru ("Dari Kata ke Vektor"), Act Ringkasan jadi Act 6. Smoke-test lokal pakai docling env (+`uv pip install nltk textblob wordcloud sentence-transformers`); cell nlp-id TIDAK dijalankan lokal (pin dependency-nya berbahaya bagi env) — cukup validasi sintaks.

**Tech Stack:** NLTK 3.9 (punkt_tab), sklearn TfidfVectorizer, sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2), transformers AutoTokenizer + pipeline, XeLaTeX Beamer, headless Chrome, puppeteer.

**Spec:** `docs/superpowers/specs/2026-06-10-module03-nlp-revamp-design.md`

**Kaidah bahasa:** Indonesia luwes; istilah English umum TIDAK diterjemahkan. Larangan tambahan modul ini: "penyemat/penyematan" (pakai "embedding"), "kemiripan kosinus" (pakai "cosine similarity").

**Env:**
- `PY=/Users/chmdznr/work/kemendag/sip/docling/bin/python`; `uv` di `/Users/chmdznr/.local/bin/uv`
- Sebelum Task 1: `uv pip install --python $PY nltk textblob wordcloud sentence-transformers` (JANGAN install nlp-id!)
- Newline guard (jalankan setelah SETIAP edit notebook): semua source line kecuali terakhir berakhir `"\n"`, terakhir tidak; tidak ada line berisi `\n` di tengah.
- LaTeX: hapus `*.aux` dulu, 2-pass xelatex, hook overfull-guard aktif (≥2pt block).

---

### Task 1: Fix notebook EN

**Files:**
- Modify: `03_nlp_fundamentals/01_nlp_fundamentals_en.ipynb`

- [ ] **Step 1: Baca notebook, petakan index cell** (download NLTK ada di cell ~2; klasifikasi cell ~17-23; word freq cell ~14-16; wordcloud ~15-16; sentiment md ~11).

- [ ] **Step 2: Terapkan edit via Python script:**
  a. Cell download NLTK: ganti `'punkt'`→`'punkt_tab'`, `'averaged_perceptron_tagger'`→`'averaged_perceptron_tagger_eng'`. Pertahankan `wordnet` dan tambah `'stopwords'` + `'movie_reviews'` bila belum diunduh di situ (cek dulu di mana movie_reviews diunduh).
  b. Cell pip install pertama: tambah `wordcloud`.
  c. Section klasifikasi:
     - Setelah pembuatan `documents`, tambah:
       ```python
       import random
       random.seed(42)
       random.shuffle(documents)
       ```
     - Ganti basis fitur: buang pemakaian `fdist` (FreqDist kalimat Section 1). Tambah sebelum `document_features`:
       ```python
       all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
       word_features = [w for (w, _) in all_words.most_common(2000)]
       ```
       dan ubah `document_features` agar iterasi `word_features` (pola NLTK Book):
       ```python
       def document_features(document):
           document_words = set(document)
           features = {}
           for word in word_features:
               features[f'contains({word})'] = (word in document_words)
           return features
       ```
     - Tambah 1 markdown cell sebelum `document_features` yang menamai pendekatan ini:
       ```
       ### Bag-of-Words: Teks sebagai Vektor Fitur

       Classifier tidak bisa membaca teks mentah — kita ubah setiap review menjadi
       **vektor fitur**: daftar `contains(kata) = True/False` untuk 2000 kata
       terpopuler di corpus. Pendekatan ini disebut **bag-of-words**: urutan kata
       diabaikan, yang dihitung hanya kehadiran kata. Sederhana, tapi inilah bentuk
       paling awal dari ide besar NLP modern: **merepresentasikan teks sebagai
       vektor angka**. (Di notebook bahasa Indonesia, Section 8 melanjutkan ide ini
       sampai ke embedding.)
       ```
  d. Word frequency: sebelum `FreqDist`, filter token: lowercase, `isalpha()`, dan buang stopwords English (`from nltk.corpus import stopwords`); wordcloud pakai token terfilter. +1 kalimat di markdown section-nya: "Tanpa membuang stopword dan tanda baca, daftar kata terpopuler hanya berisi 'the', 'of', dan koma — jadi kita bersihkan dulu."
  e. Markdown section Sentiment: tambah kalimat "Catatan: TextBlob hanya akurat untuk teks **English** — untuk bahasa Indonesia, lihat Section 10 di notebook bahasa Indonesia (IndoBERT)."
  f. Hapus semua komentar `# prompt: ...` (sisa Colab AI).

- [ ] **Step 3: Validasi**: JSON valid; newline guard 0 pelanggaran; ast.parse semua code cell (join kosong); grep `punkt'` tanpa `_tab` → 0.

- [ ] **Step 4: Smoke-run end-to-end versi mini** di docling env (setelah `uv pip install nltk textblob wordcloud`): gabung code cells, matplotlib Agg + plt.show no-op, dan untuk klasifikasi batasi `documents = documents[:300]` + `word_features` top-500 agar cepat. NLTK download akan jalan (punkt_tab dll). Expect selesai tanpa exception, akurasi tercetak masuk akal (0.6-0.9, BUKAN aneh). Timeout 600000 ms.

- [ ] **Step 5: Commit**
```bash
git add 03_nlp_fundamentals/01_nlp_fundamentals_en.ipynb
git commit -m "fix(module03): repair EN notebook (NLTK punkt_tab, classification methodology, stopword-filtered wordfreq)"
```

---

### Task 2: Fix + perluas notebook ID (3 section baru)

**Files:**
- Modify: `03_nlp_fundamentals/01_nlp_fundamentals_id.ipynb`

- [ ] **Step 1: Baca notebook, petakan cell** (pip install cell ~1; phrase tokenization md ~11; lemmatization md ~19; tips print ~25; cell duplikat ~26; ringkasan ~27).

- [ ] **Step 2: Fixes:**
  a. Cell duplikat Section 7: hapus re-import & re-inisialisasi (Tokenizer/StopWord/Lemmatizer sudah ada); ramping jadi satu loop analisis yang reuse objek; bagian "topic analysis" hardcoded dihapus ATAU diberi 1 kalimat md penjelasan — pilih hapus jika tidak menambah nilai.
  b. Cell tips `print("""...""")` → markdown cell dengan konten sama (rapikan jadi bullet).
  c. Klaim lemmatization: "dengan memperhatikan konteks" → "berbasis kamus dan aturan — per kata, tanpa melihat konteks kalimat".
  d. Ringkasan: nlp-id = library utama; Sastrawi disebut sebagai opsi stemming historis + catatan "tidak lagi aktif di-maintain (rilis terakhir ~2016)".
  e. Phrase tokenization md: ganti contoh motivasi "tidak enak" → frasa nomina: "PhraseTokenizer mengelompokkan frasa nomina seperti 'Universitas Indonesia' atau 'kecerdasan buatan' menjadi satu token — penting agar entitas multi-kata tidak terpecah."
  f. Setelah cell pip install: tambah md "⚠️ nlp-id mengunci versi dependency tertentu — kalau Colab menampilkan pesan untuk **restart runtime** setelah install, lakukan (Runtime > Restart session), lalu lanjut dari cell berikutnya."
  g. Setelah output pipeline pada teks gaul: tambah md "Perhatikan: kata gaul seperti 'gw' dan 'bgt' TIDAK dinormalkan oleh pipeline ini — pipeline klasik memang rapuh terhadap bahasa informal. Di Section 10 kamu akan lihat model modern yang jauh lebih tahan slang."

- [ ] **Step 3: Tambah Section 8 — "Dari Kata ke Vektor" (8 cell, sebelum Ringkasan):**

Cell 8a (md):
```
## 8. Dari Kata ke Vektor — Jembatan ke Dunia LLM

Sejauh ini kita MENGHITUNG kata: frekuensi, stopword, bentuk dasar. Tapi komputer
tidak paham "makna" dari hitungan. Cara modern: ubah teks menjadi **vektor angka**
sedemikian rupa sehingga teks yang maknanya mirip menghasilkan vektor yang
berdekatan. Kita naik tangga tiga anak tangga:

1. **Bag-of-Words / TF-IDF** — vektor berbasis hitungan kata (klasik)
2. **Embedding** — vektor makna hasil belajar dari jutaan kalimat (modern)
3. **Semantic search** — aplikasi nyata: mencari dokumen berdasarkan MAKNA query

Anak tangga ketiga ini adalah fondasi RAG yang akan kamu bangun di Module 05.
```

Cell 8b (code):
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

kalimat = [
    "Saya suka makan nasi goreng",
    "Saya gemar menyantap nasi goreng",   # makna ~sama dgn kalimat 1, kata beda
    "Nasi goreng adalah makanan favorit saya",
    "Harga BBM naik lagi minggu ini",     # topik beda total
]

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(kalimat)
print("Ukuran matriks TF-IDF:", X_tfidf.shape)
print("Kosakata:", list(vectorizer.get_feature_names_out()))
```

Cell 8c (code):
```python
sim_tfidf = cosine_similarity(X_tfidf)
label = [f"K{i+1}" for i in range(len(kalimat))]
print("Cosine similarity antar kalimat (TF-IDF):")
print(pd.DataFrame(sim_tfidf.round(2), index=label, columns=label))
print()
print(f"K1 vs K2 (sinonim, kata beda): {sim_tfidf[0, 1]:.2f}  <-- rendah!")
print(f"K1 vs K3 (kata banyak sama)  : {sim_tfidf[0, 2]:.2f}")
```

Cell 8d (md):
```
### Kenapa TF-IDF "buta" terhadap sinonim?

K1 ("suka makan") dan K2 ("gemar menyantap") maknanya hampir identik, tapi skor
similarity-nya rendah — karena TF-IDF hanya mencocokkan **kata yang persis sama**.
Bagi TF-IDF, "suka" dan "gemar" adalah dua dimensi yang tidak berhubungan.

Solusinya: **embedding** — model neural network yang dilatih pada jutaan kalimat
sehingga belajar bahwa "suka"≈"gemar" dan "makan"≈"menyantap". Hasilnya: satu
kalimat → satu vektor (di model ini 384 angka) yang menangkap MAKNA.
```

Cell 8e (code):
```python
!pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer

model_embedding = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
emb = model_embedding.encode(kalimat)
print("Bentuk embedding:", emb.shape)  # (4, 384): 4 kalimat, masing-masing 384 angka
```

Cell 8f (code):
```python
import seaborn as sns
import matplotlib.pyplot as plt

sim_emb = cosine_similarity(emb)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.heatmap(sim_tfidf, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=label, yticklabels=label, ax=axes[0], vmin=0, vmax=1)
axes[0].set_title("TF-IDF: K1-K2 dianggap beda")
sns.heatmap(sim_emb, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=label, yticklabels=label, ax=axes[1], vmin=0, vmax=1)
axes[1].set_title("Embedding: K1-K2 terdeteksi semakna")
plt.tight_layout()
plt.show()
```

Cell 8g (md + intro semantic search):
```
### Mini Semantic Search: Mesin Pencari Berbasis Makna

Sekarang aplikasi nyatanya. Kita punya 10 "dokumen" fakta tentang Indonesia.
Diberi sebuah pertanyaan, kita cari dokumen yang paling relevan — BUKAN dengan
mencocokkan kata, melainkan dengan membandingkan vektor embedding-nya.
```

Cell 8h (code):
```python
dokumen = [
    "Gunung Kerinci adalah gunung berapi tertinggi di Indonesia",
    "Candi Borobudur dibangun pada masa dinasti Syailendra",
    "Komodo adalah kadal terbesar di dunia yang hidup di Nusa Tenggara",
    "Batik Indonesia diakui UNESCO sebagai warisan budaya dunia",
    "Danau Toba terbentuk dari letusan gunung api purba",
    "Rendang berasal dari Sumatera Barat dan dimasak berjam-jam",
    "Ibu kota Nusantara dibangun di Kalimantan Timur",
    "Raja Ampat di Papua terkenal dengan keanekaragaman hayati lautnya",
    "Angklung adalah alat musik bambu khas Jawa Barat",
    "Sungai Kapuas adalah sungai terpanjang di Indonesia",
]

emb_dokumen = model_embedding.encode(dokumen)

query = "puncak paling tinggi di Indonesia"   # tidak ada kata 'gunung' / 'Kerinci'!
emb_query = model_embedding.encode([query])

skor = cosine_similarity(emb_query, emb_dokumen)[0]
peringkat = skor.argsort()[::-1]

print(f"Query: {query!r}\n")
print("3 dokumen paling relevan:")
for i in peringkat[:3]:
    print(f"  {skor[i]:.2f}  {dokumen[i]}")
```

Cell 8i (md):
```
Perhatikan: query "puncak paling tinggi" menemukan dokumen "gunung berapi
tertinggi" — **tanpa satu pun kata yang sama**. Itulah kekuatan semantic search.

Kamu baru saja membangun inti dari **RAG (Retrieval-Augmented Generation)** dalam
~10 baris: encode dokumen → encode query → cosine similarity → ambil top-k.
Di Module 05, pola yang persis sama dipakai lagi — bedanya dokumen disimpan di
**FAISS** (index khusus agar pencarian tetap cepat untuk jutaan dokumen), dan
hasil pencariannya diberikan ke LLM untuk menyusun jawaban.
```
(8a-8i = 9 cell: 4 md + 5 code — gunakan ini, bukan "8 cell".)

- [ ] **Step 4: Tambah Section 9 — "Tokenisasi ala LLM" (3 cell):**

Cell 9a (md):
```
## 9. Tokenisasi ala LLM — Subword

Di Section 1, token = kata. Tapi LLM modern (GPT, BERT, Llama) memakai
**subword tokenization** (BPE/WordPiece): kata dipecah menjadi potongan yang
lebih kecil dari kata, tapi lebih besar dari huruf.

Kenapa? Dengan kosakata terbatas (~30-50 ribu subword), model bisa menulis
**kata apa pun** — termasuk kata yang belum pernah dilihat saat training.
Untuk bahasa Indonesia yang kaya imbuhan (me-, di-, -kan, -nya...), ini
relevan sekali: "mempelajari", "dipelajari", "pelajaran" berbagi potongan
"pelajar" yang sama.
```

Cell 9b (code):
```python
from transformers import AutoTokenizer

tok_gpt2 = AutoTokenizer.from_pretrained("gpt2")
tok_multi = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

kata = "mempelajari"
print(f"GPT-2        : {tok_gpt2.tokenize(kata)}")
print(f"Multilingual : {tok_multi.tokenize(kata)}")

kalimat_id = "Saya sedang mempelajari kecerdasan buatan di kelas ini"
kalimat_en = "I am learning artificial intelligence in this class"

n_id = len(tok_gpt2(kalimat_id)["input_ids"])
n_en = len(tok_gpt2(kalimat_en)["input_ids"])
print(f"\nJumlah token GPT-2 — kalimat Indonesia: {n_id}, kalimat Inggris semakna: {n_en}")
```

Cell 9c (md):
```
Dua pengamatan penting:

1. **GPT-2 memecah kata Indonesia jadi banyak potongan kecil** — tokenizer-nya
   dilatih dominan pada teks English. Akibat praktisnya: kalimat Indonesia
   "lebih mahal" di API LLM yang menagih per token. Tokenizer multilingual
   jauh lebih hemat untuk bahasa Indonesia.
2. Mulai Module 04, setiap kali kamu bertemu istilah "token" (misalnya
   `max_new_tokens`), yang dimaksud adalah **subword seperti ini** — bukan
   kata utuh hasil `word_tokenize`.
```

- [ ] **Step 5: Tambah Section 10 — "Sentiment Bahasa Indonesia dengan IndoBERT" (3 cell):**

SEBELUM menulis cell, verifikasi model tersedia & ungated:
`curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/models/w11wo/indonesian-roberta-base-sentiment-classifier` → 200 berarti OK. Fallback bila 4xx: `mdhugol/indonesia-bert-sentiment-classification` (cek dengan cara sama; sesuaikan nama label di md bila fallback dipakai).

Cell 10a (md):
```
## 10. Sentiment Analysis Bahasa Indonesia dengan IndoBERT

Di notebook English, sentiment dianalisis dengan TextBlob — yang tidak paham
bahasa Indonesia. Cara industri Indonesia melakukannya di era sekarang:
model transformer (keluarga **IndoBERT**) yang sudah di-fine-tune pada data
sentimen berbahasa Indonesia, tinggal dipakai lewat `pipeline` Hugging Face —
persis pola transfer learning yang kamu lihat di Module 02 (ResNet).
```

Cell 10b (code):
```python
from transformers import pipeline

analisis_sentimen = pipeline(
    "sentiment-analysis",
    model="w11wo/indonesian-roberta-base-sentiment-classifier",
)

uji = [
    "Gw suka bgt sama teknologi NLP nih",          # bahasa gaul dari Section 1!
    "Pelayanan restoran itu lambat dan mengecewakan",
    "Filmnya biasa saja, tidak jelek tapi tidak bagus juga",
]
for teks in uji:
    hasil = analisis_sentimen(teks)[0]
    print(f"{hasil['label']:>8} ({hasil['score']:.2f})  {teks}")
```

Cell 10c (md):
```
Tiga hal yang layak dicatat:

1. Kalimat gaul "Gw suka bgt..." — yang gagal dinormalkan pipeline klasik kita
   di Section 1 — **langsung dikenali positif**. Model transformer belajar dari
   teks nyata (termasuk bahasa informal), jadi jauh lebih tahan slang.
2. Model juga mengenali kelas **netral** — bukan sekadar positif/negatif.
3. Ekosistem NLP Indonesia lebih luas dari ini: **IndoNLU** (benchmark),
   **IndoLEM**, **NusaBERT** — semuanya tersedia di Hugging Face Hub.

Di Module 04, `pipeline` yang sama persis menjadi pintu masuk kamu ke LLM.
```

- [ ] **Step 6: Update cell "Apa yang akan kita pelajari?"** — tambah 3 butir (vektor & semantic search; subword tokenization; sentiment IndoBERT), dan **Ringkasan** — tambah ringkasan 3 section + ganti narasi penutup agar menyebut jembatan ke Module 04 (token & pipeline) dan Module 05 (embedding & semantic search).

- [ ] **Step 7: Validasi**: JSON valid; newline guard 0; ast.parse (join kosong) semua code cell; grep anti-pattern bahasa (+ "penyemat", "kemiripan kosinus") → 0.

- [ ] **Step 8: Smoke-run CELL BARU SAJA** di docling env (sklearn/transformers/torch sudah ada; sentence-transformers di-install Task 0): jalankan kode 8b,8c,8e(tanpa pip line),8f,8h,9b,10b berurutan dengan Agg backend. Model downloads: MiniLM (~470MB) + IndoBERT (~500MB) + 2 tokenizer — sekali saja, acceptable. Expect: similarity TF-IDF K1-K2 RENDAH (<0.3) dan embedding K1-K2 TINGGI (>0.6); semantic search top-1 = dokumen Gunung Kerinci; sentiment 3 teks = positive/negative/neutral (atau setara). Kalau ekspektasi konten tidak terpenuhi (mis. top-1 salah dokumen), sesuaikan contoh kalimat — JANGAN biarkan klaim md tidak cocok dengan output nyata. Timeout 600000 ms.

- [ ] **Step 9: Commit**
```bash
git add 03_nlp_fundamentals/01_nlp_fundamentals_id.ipynb
git commit -m "feat(module03): add kata-ke-vektor, subword tokenization, IndoBERT sections + cleanup ID notebook"
```

---

### Task 3: Deck M03 — fixes + Act 5 "Dari Kata ke Vektor"

**Files:**
- Modify: `03_nlp_fundamentals/slides/module03_slides.tex` (+ recompile .pdf)

- [ ] **Step 1: Fixes (baca frame terkait dulu, lalu edit):**
  a. Frame "Notebook 01 (ID)" (~l.561-573): hapus baris `Sastrawi — stemming Indonesia` dari daftar library; pastikan nlp-id tercantum.
  b. Frame "Setup & Tools" (~l.622): `!pip install Sastrawi nlp-id ...` → hapus `Sastrawi`.
  c. Frame "Lanjutan" (~l.644): deskripsi Module 06 "(TensorRT, NeMo, RAG)" → "(TensorRT, NeMo, GPU optimization)".
  d. Frame POS Tagging (~l.348-369): tambah satu baris kecil `{\color{nvgray}\scriptsize Catatan: label di slide memakai tagset UD (ilustratif); nlp-id memakai NNP/VB/NN/IN.}`
  e. ~l.300: reword klaim imbuhan — dari konteks tokenization ke "imbuhan membuat lemmatization/stemming Indonesia lebih menantang".
  f. Frame NER (~l.421): tambah catatan kecil `{\color{nvgray}\scriptsize spaCy melabeli kota/negara sebagai GPE (bukan LOC).}`

- [ ] **Step 2: Renumber Act lama**: `\acttitle{5}{Ringkasan \& Lanjutan}{...}` → `\acttitle{6}{Ringkasan \& Lanjutan}{...}`.

- [ ] **Step 3: Sisipkan Act 5 baru SEBELUM acttitle Ringkasan:**

```latex
% ============================================================
% ACT 5: DARI KATA KE VEKTOR
% ============================================================
\acttitle{5}{Dari Kata ke Vektor}{Jembatan menuju LLM (Module 04) dan RAG (Module 05)}

\begin{frame}{Dari Menghitung Kata ke Memahami Makna}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      {\color{nvgreen}\textbf{TF-IDF / Bag-of-Words}}
      \begin{itemize}
        \item Vektor = hitungan kata
        \item ``suka'' dan ``gemar'' dianggap TIDAK berhubungan
        \item Baseline klasik, murah, cepat
      \end{itemize}
    \end{column}
    \begin{column}{0.48\textwidth}
      {\color{nvorange}\textbf{Embedding}}
      \begin{itemize}
        \item Vektor makna, belajar dari jutaan kalimat
        \item Sinonim $\to$ vektor berdekatan
        \item \texttt{sentence-transformers} multilingual
      \end{itemize}
    \end{column}
  \end{columns}
  \vspace{0.5em}
  \begin{center}
    {\color{nvlightgreen}\small Kedekatan diukur dengan \textbf{cosine similarity} --- semantic search: cari dokumen berdasarkan MAKNA query $\Rightarrow$ inti RAG di Module 05}
  \end{center}
\end{frame}

\begin{frame}[fragile]{Subword Tokenization: Token ala LLM}
  \begin{lstlisting}[language=Python]
tok = AutoTokenizer.from_pretrained("gpt2")
tok.tokenize("mempelajari")
# ['m', 'empel', 'aj', 'ari']  (potongan subword!)
  \end{lstlisting}
  \begin{itemize}
    \item Kosakata terbatas ($\sim$30--50 ribu subword) bisa menulis kata APA PUN
    \item Tokenizer English-centric memecah kata Indonesia jadi lebih banyak token $\to$ kalimat Indonesia ``lebih mahal'' di API per-token
    \item Mulai Module 04: ``token'' = subword ini, bukan kata utuh
  \end{itemize}
\end{frame}

\begin{frame}[fragile]{Sentiment Indonesia dengan IndoBERT}
  \begin{lstlisting}[language=Python]
sentimen = pipeline("sentiment-analysis",
    model="w11wo/indonesian-roberta-base-sentiment-classifier")
sentimen("Gw suka bgt sama teknologi NLP nih")
# [{'label': 'positive', 'score': 0.98}]
  \end{lstlisting}
  \begin{itemize}
    \item Transformer fine-tuned untuk bahasa Indonesia --- tahan bahasa gaul
    \item Pola transfer learning yang sama dengan ResNet di Module 02
    \item Ekosistem: IndoBERT, IndoNLU, IndoLEM, NusaBERT (Hugging Face Hub)
  \end{itemize}
  \vspace{0.3em}
  {\color{nvlightgreen}\small $\Rightarrow$ \texttt{pipeline} yang sama menjadi pintu masuk ke LLM di Module 04}
\end{frame}
```
CATATAN: token hasil `tok.tokenize("mempelajari")` di listing HARUS diganti dengan hasil NYATA dari smoke-run Task 2 Step 8 (jangan mengarang); demikian pula label/score IndoBERT.

- [ ] **Step 4: Update frame overview "Notebook 01 (ID)"** agar mencantumkan 3 section baru (vektor & semantic search, subword, IndoBERT), dan frame Ringkasan bila ia mendaftar topik.

- [ ] **Step 5: Compile 2-pass + verifikasi**: hapus aux; `grep -c Overfull` — baseline M03 bersih, jadi harus 0; page count naik (~23 → ~28). Inspeksi visual halaman Act 5 (PNG render + Read).

- [ ] **Step 6: Commit**
```bash
git add 03_nlp_fundamentals/slides/module03_slides.tex 03_nlp_fundamentals/slides/module03_slides.pdf
git commit -m "feat(slides): Act 5 'Dari Kata ke Vektor' + accuracy fixes for Module 03 deck"
```

---

### Task 4: Reframe 2 frame deck M04 jadi recap

**Files:**
- Modify: `04_llm/slides/module04_slides.tex` (+ recompile .pdf)

- [ ] **Step 1: Cek baseline overfull M04**: compile dulu TANPA edit, catat jumlah `Overfull` (agar bisa membedakan overfull baru vs pre-existing).

- [ ] **Step 2: Frame "Tokenization: Teks Menjadi Angka"**: baca frame; tambahkan di awal body: `{\color{nvgray}\small Ingat dari Module 03: LLM memakai \textbf{subword tokenization} --- kamu sudah melihat ``mempelajari'' dipecah jadi potongan subword.}` dan sesuaikan kalimat pembuka frame agar nadanya recap, bukan pengenalan pertama. Konten inti frame dipertahankan.

- [ ] **Step 3: Frame "Embedding \& Transformer"**: tambahkan bullet pembeda:
  `\item Bedakan: di Module 03 kamu memakai \textbf{sentence embedding statis} (satu kalimat $\to$ satu vektor); di DALAM transformer, setiap token punya \textbf{contextual embedding} yang berubah mengikuti kalimatnya ("bisa" dalam ``bisa ular'' vs ``saya bisa'')`
  (sesuaikan format itemize frame; contoh "bisa" boleh diganti contoh ambigu lain yang sudah ada di deck bila ada).

- [ ] **Step 4: Compile 2-pass; overfull TIDAK bertambah dari baseline Step 1; page count tetap. Inspeksi visual 2 frame.**

- [ ] **Step 5: Commit**
```bash
git add 04_llm/slides/module04_slides.tex 04_llm/slides/module04_slides.pdf
git commit -m "refactor(slides): reframe M04 tokenization & embedding frames as recap of Module 03"
```

---

### Task 5: Cheatsheet + quiz M03

**Files:**
- Modify: `03_nlp_fundamentals/nlp-fundamentals-cheatsheet.html` (+ re-render .pdf)
- Modify: `03_nlp_fundamentals/nlp-fundamentals-quiz.html`

- [ ] **Step 1: Cheatsheet — 2 kartu baru** (ikuti struktur kartu existing; konten HTML-escaped seperlunya):

```html
<div class="card"><h3>Dari Kata ke Vektor</h3><ul><li>TF-IDF: vektor hitungan kata — buta sinonim ("suka" != "gemar")</li><li>Embedding: vektor makna; sinonim -&gt; vektor berdekatan</li><li>cosine_similarity(emb_query, emb_dokumen) -&gt; semantic search</li><li>Inti RAG di Module 05 (di sana ditambah FAISS)</li></ul><div class="when">▸ Fondasi retrieval &amp; RAG</div></div>
```
```html
<div class="card"><h3>Subword &amp; IndoBERT</h3><ul><li>LLM memakai subword (BPE/WordPiece): "mempelajari" dipecah jadi potongan</li><li>Kalimat Indonesia lebih banyak token di tokenizer English-centric</li><li>IndoBERT via pipeline("sentiment-analysis") — tahan bahasa gaul</li><li>Ekosistem ID: IndoNLU, IndoLEM, NusaBERT</li></ul><div class="when">▸ Jembatan langsung ke Module 04 (LLM)</div></div>
```

  Glossary +4 (ikuti format & gaya kapitalisasi entri existing — huruf besar di awal, titik di akhir): **Embedding** — Vektor angka yang merepresentasikan makna teks; **Cosine similarity** — Ukuran kedekatan dua vektor (1 = searah/semakna); **Subword** — Potongan kata yang menjadi satuan token LLM (BPE/WordPiece); **TF-IDF** — Bobot kata berdasarkan frekuensi di dokumen vs keseluruhan corpus.
  Update teks scope bila ia mendaftar topik. Re-render headless Chrome; **WAJIB 1 halaman** (kalau lebih: kecilkan font 0.5px dulu, lalu footer margin — pola M02).

- [ ] **Step 2: Quiz +3 soal** (append ke const QUIZ via json round-trip; verifikasi 12 soal lama deep-equal; update header & scorebar 12→15):

```json
{
 "q": "TF-IDF memberi skor similarity RENDAH untuk 'Saya suka makan nasi goreng' vs 'Saya gemar menyantap nasi goreng', padahal maknanya hampir sama. Kenapa?",
 "code": "",
 "options": [
  "Karena kedua kalimat memiliki jumlah kata yang berbeda",
  "Karena TF-IDF hanya mencocokkan kata yang persis sama, sedangkan 'suka'/'gemar' dan 'makan'/'menyantap' adalah kata berbeda",
  "Karena TF-IDF tidak bisa memproses bahasa Indonesia",
  "Karena kalimatnya terlalu pendek untuk dihitung TF-IDF"
 ],
 "answer": 1,
 "explanation": "Slide 'Dari Menghitung Kata ke Memahami Makna': TF-IDF merepresentasikan teks sebagai hitungan kata, sehingga sinonim dianggap dimensi yang tidak berhubungan. Embedding menyelesaikan ini karena dilatih pada jutaan kalimat sehingga sinonim menghasilkan vektor berdekatan. TF-IDF tetap bisa memproses bahasa apa pun dan tidak bergantung pada panjang/jumlah kata."
}
```
```json
{
 "q": "Kenapa LLM modern memakai subword tokenization (BPE/WordPiece), bukan token per kata utuh?",
 "code": "tok.tokenize(\"mempelajari\")\n# hasil: beberapa potongan subword",
 "options": [
  "Supaya teks menjadi lebih pendek dan hemat memori",
  "Karena komputer tidak bisa menyimpan kata yang panjang",
  "Agar dengan kosakata terbatas (puluhan ribu subword) model tetap bisa menulis kata apa pun, termasuk kata yang tidak pernah dilihat saat training",
  "Karena subword membuat model selalu lebih akurat daripada token kata"
 ],
 "answer": 2,
 "explanation": "Slide 'Subword Tokenization: Token ala LLM': kosakata ~30-50 ribu subword cukup untuk merangkai kata apa pun, termasuk kata baru dan kata berimbuhan. Subword justru sering membuat teks jadi LEBIH BANYAK token (bukan lebih pendek), dan tidak ada jaminan 'selalu lebih akurat'."
}
```
```json
{
 "q": "Untuk sentiment analysis teks bahasa Indonesia yang mengandung bahasa gaul ('Gw suka bgt...'), pendekatan mana yang paling tepat menurut materi?",
 "code": "",
 "options": [
  "TextBlob, karena library-nya paling sederhana",
  "Menghapus semua kata gaul terlebih dahulu, lalu memakai TextBlob",
  "Model transformer yang di-fine-tune untuk bahasa Indonesia (keluarga IndoBERT) via pipeline Hugging Face",
  "Menerjemahkan teks ke bahasa Inggris secara manual lalu memakai TextBlob"
 ],
 "answer": 2,
 "explanation": "Slide 'Sentiment Indonesia dengan IndoBERT': TextBlob hanya akurat untuk English, sedangkan model keluarga IndoBERT dilatih pada teks Indonesia nyata (termasuk bahasa informal) sehingga tahan slang dan mengenali kelas netral. Menghapus kata gaul atau menerjemahkan manual justru membuang sinyal sentimen."
}
```

- [ ] **Step 3: Uji puppeteer** (pola M02: require puppeteer dari mermaid-cli + executablePath Chrome bila perlu; struktur DOM quiz M03 = template yang sama: `.q`/`.opt`/`correct`/`wrong`/`#correct`): 15 soal; klik salah di Q13 → wrong + correct ter-highlight; klik benar Q14-15 → correct + skor 2.

- [ ] **Step 4: Commit**
```bash
git add 03_nlp_fundamentals/nlp-fundamentals-cheatsheet.html 03_nlp_fundamentals/nlp-fundamentals-cheatsheet.pdf 03_nlp_fundamentals/nlp-fundamentals-quiz.html
git commit -m "feat(module03): cheatsheet vektor/subword cards + 3 quiz questions (15 total)"
```

---

### Task 6: QC akhir, handoff, push

- [ ] **Step 1: QC lintas-artefak:**
```bash
# Bahasa (4 artefak M03 + deck M04)
grep -riE 'ruang laten|kebisingan|pembangkit|pelatihan berlawanan|jaringan pembeda|penyemat|kemiripan kosinus' \
  03_nlp_fundamentals/01_nlp_fundamentals_en.ipynb 03_nlp_fundamentals/01_nlp_fundamentals_id.ipynb \
  03_nlp_fundamentals/slides/module03_slides.tex 03_nlp_fundamentals/nlp-fundamentals-cheatsheet.html \
  03_nlp_fundamentals/nlp-fundamentals-quiz.html 04_llm/slides/module04_slides.tex && echo "ADA ANTI-PATTERN" || echo "bahasa OK"
# Konsistensi istilah kunci (harus muncul konsisten: cosine similarity, subword, embedding, IndoBERT)
grep -c 'cosine similarity\|subword\|IndoBERT' 03_nlp_fundamentals/01_nlp_fundamentals_id.ipynb 03_nlp_fundamentals/slides/module03_slides.tex 03_nlp_fundamentals/nlp-fundamentals-cheatsheet.html
# Sastrawi tidak lagi diklaim sbg library notebook (boleh tersisa sbg catatan historis di notebook ID)
grep -n 'Sastrawi' 03_nlp_fundamentals/slides/module03_slides.tex
# Deck M03 0 overfull; M04 tidak bertambah dari baseline; cheatsheet 1 halaman
grep -c Overfull 03_nlp_fundamentals/slides/module03_slides.log || echo 0
mdls -name kMDItemNumberOfPages 03_nlp_fundamentals/nlp-fundamentals-cheatsheet.pdf
# Branding
grep -ci 'NCA-GENL\|Achmad' 03_nlp_fundamentals/01_nlp_fundamentals_id.ipynb 03_nlp_fundamentals/01_nlp_fundamentals_en.ipynb || echo "branding OK"
```

- [ ] **Step 2: Handoff** — tambah section "### Update Jun 10 — Revamp Module 03 NLP" di `docs/handoffs/2026-06-01-session-handoff.md`: bug fix EN (punkt_tab + metodologi klasifikasi), 3 section baru notebook ID (+nomor commit), Act 5 deck M03, reframe 2 frame M04, cheatsheet/quiz 15 soal, batas materi M03=representasi vs M04=arsitektur, catatan model HF yang dipakai (sentence-transformers multilingual + IndoBERT w11wo) dan bahwa notebook BELUM diuji ulang penuh di Colab oleh user.

- [ ] **Step 3: Commit handoff + push + `git status -sb` sinkron.**

---

## Catatan Eksekutor

1. JANGAN `pip install nlp-id` ke docling env (pin exact numpy/sklearn — merusak env user). Cell nlp-id divalidasi sintaks saja; eksekusinya domain Colab.
2. Snippet hasil di deck (token subword, label IndoBERT) HARUS dari output smoke-run nyata, bukan karangan.
3. Task 3 bergantung output smoke-run Task 2 (snippet nyata) — kerjakan setelah Task 2. Task 4 independen (boleh kapan saja). Task 5 setelah Task 3 (referensi judul slide di explanation quiz).
4. Newline guard WAJIB setiap selesai mengedit notebook (bug paket M02 jangan terulang).
5. Hook latex-overfull-guard aktif; jangan bypass.
