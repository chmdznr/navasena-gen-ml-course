#!/usr/bin/env python3
"""Bangun 03_nlp_fundamentals/01_nlp_fundamentals.ipynb — hasil gabungan notebook _en dan _id.

Struktur: Bagian A (bersihkan) -> B (analisis) -> C (nilai) -> D (kata ke makna) -> Latihan -> Ringkasan.
Jalankan: python build_nb01.py
"""
import json, pathlib, sys

TOOLS = pathlib.Path(__file__).resolve()
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbedit import _cell, save  # noqa: E402

C = []
def md(t): C.append(_cell("markdown", t.strip("\n")))
def code(t): C.append(_cell("code", t.strip("\n")))

# ---------------------------------------------------------------- header
md(r"""
# NLP Fundamentals — Dari Teks Mentah ke Vektor Makna

**Modul 3: NLP Fundamentals** | Notebook 1 dari 2

---

## Tujuan Pembelajaran

Setelah menyelesaikan notebook ini, kamu akan mampu:

1. **Membersihkan teks Indonesia** — tokenisasi, stopword, lemmatisasi dengan `nlp-id`
2. **Menandai kelas kata (POS tagging)** dan mengenali frasa multi-kata
3. **Menganalisis frekuensi kata** dan memvisualkannya sebagai word cloud
4. **Mengenali entitas bernama (NER)** — nama orang, organisasi, lokasi
5. **Menganalisis sentimen** — membandingkan pendekatan leksikon (TextBlob) dan transformer (IndoBERT)
6. **Melatih classifier teks** dengan Naive Bayes tanpa membocorkan data uji
7. **Mengubah teks menjadi vektor** — TF-IDF, embedding, dan semantic search
8. **Menjelaskan tokenisasi subword** — arti kata "token" yang akan kamu temui di Modul 04

**Estimasi sesi:** ± 60 menit (termasuk membaca)

---

## Kenapa NLP butuh perlakuan khusus?

Bahasa manusia tidak rapi seperti tabel angka. Kalimat panjangnya berbeda-beda,
satu kata bisa punya banyak makna, dan bahasa Indonesia menambah tantangannya
sendiri: imbuhan berlapis (`me-`, `di-`, `-kan`, `-nya`), kata serapan, singkatan,
dan bahasa gaul. Notebook ini menempuh perjalanan dari teks mentah sampai ke
vektor makna — jalur yang sama yang dipakai LLM di Modul 04 dan RAG di Modul 05.
""")

# ---------------------------------------------------------------- setup
md(r"""
## Setup

Semua library dipasang di satu tempat supaya runtime cukup di-restart sekali.
Versi di-pin agar hasilnya tetap sama saat kamu mengulang notebook ini bulan depan.
""")

code(r"""
# Instalasi (± 2 menit). Versi di-pin agar hasilnya tetap sama saat notebook diulang.
# nlp-id >= 0.1.22 mengunci huggingface-hub==1.14.0, dan huggingface-hub 1.x
# hanya kompatibel dengan transformers 5.x. Memakai transformers 4.x di sini
# membuat resolusi dependency GAGAL, bukan sekadar memberi peringatan.
PAKET = ' '.join([
    "nlp-id>=0.1.23,<0.2", "textblob>=0.19,<1", "wordcloud>=1.9,<2",
    "sentence-transformers>=5,<6", "transformers>=5,<6",
])
!pip install -q {PAKET}
!python -m spacy download en_core_web_sm -q
""")

md(r"""
⚠️ Kalau Colab menampilkan tombol **RESTART SESSION** setelah sel di atas selesai, tekan tombolnya,
lalu lanjutkan dari sel berikutnya (tidak perlu mengulang instalasi).
""")

code(r"""
import os, re, string, random
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)
sns.set_style("whitegrid")

# QUICK=1 -> notebook dieksekusi cepat (subset data lebih kecil) untuk verifikasi lokal.
QUICK = bool(os.environ.get("QUICK"))
print("Mode QUICK aktif." if QUICK else "Mode penuh.")
""")

code(r"""
import nltk

for paket in ["punkt_tab", "averaged_perceptron_tagger_eng", "wordnet", "stopwords", "movie_reviews"]:
    nltk.download(paket, quiet=True)
print("Data NLTK siap.")
""")

code(r"""
from nlp_id.tokenizer import Tokenizer, PhraseTokenizer
from nlp_id.postag import PosTag
from nlp_id.stopword import StopWord
from nlp_id.lemmatizer import Lemmatizer

tokenizer = Tokenizer()
phrase_tokenizer = PhraseTokenizer()
postagger = PosTag()
stopword = StopWord()
lemmatizer = Lemmatizer()

STOPWORD_ID = set(stopword.get_stopword())
print(f"nlp-id siap. Jumlah stopword Indonesia: {len(STOPWORD_ID)}")
""")

# ================================================================ BAGIAN A
md(r"""
⚠️ Sel di atas memunculkan `InconsistentVersionWarning` dari scikit-learn. Penyebabnya:
model POS tagger bawaan `nlp-id` disimpan dengan scikit-learn versi lama, lalu dimuat
dengan versi yang lebih baru. Peringatan ini muncul setiap kali dan **tidak menghentikan
notebook** — keluaran POS tagging tetap wajar. Sebutkan saja supaya tidak dikira ada
yang rusak.
""")

md(r"""
---

# Bagian A — Membersihkan Teks

Model machine learning tidak bisa menelan teks mentah. Empat section berikut
membangun pipeline pembersihan yang dipakai ulang sepanjang notebook.
""")

# --- 1
md(r"""
## 1. Pipeline Pra-pemrosesan Teks

Urutan yang lazim dipakai:

1. **Huruf kecil** — `NLP` dan `nlp` jangan dihitung sebagai dua kata berbeda
2. **Buang angka & tanda baca** — jarang membawa makna untuk analisis kata
3. **Rapikan spasi** — teks nyata penuh spasi ganda dan baris baru
4. **Tokenisasi** — pecah jadi kata
5. **Buang stopword** — kata umum yang muncul di mana-mana
6. **Lemmatisasi** — kembalikan kata ke bentuk dasar

Urutan langkah 5 dan 6 bukan aturan mati. Kita membuang stopword **sebelum**
lemmatisasi karena daftar stopword `nlp-id` sudah berisi bentuk permukaan yang
umum (`yang`, `di`, `adalah`), jadi mencocokkannya lebih akurat sebelum kata diubah.
""")

code(r'''
def bersihkan(teks):
    """Pipeline pra-pemrosesan untuk teks Indonesia: teks mentah -> daftar token bersih."""
    teks = teks.lower()
    teks = teks.translate(str.maketrans("", "", string.punctuation))
    teks = re.sub(r"\d+", "", teks)
    teks = " ".join(teks.split())

    token = tokenizer.tokenize(teks)
    token = [t for t in token if t not in STOPWORD_ID]
    token = [lemmatizer.lemmatize(t) for t in token]
    return token
''')

code(r"""
contoh_teks = [
    "Pemrosesan Bahasa Alami (NLP) adalah bidang ilmu komputer dan kecerdasan buatan "
    "yang berfokus pada interaksi antara komputer dan bahasa manusia.",
    "Gw suka bgt sama teknologi NLP nih, keren banget aplikasinya!",
    "Para peneliti sedang mengembangkan algoritma baru untuk meningkatkan akurasi.",
]

for i, teks in enumerate(contoh_teks, 1):
    print(f"\nContoh {i}")
    print("  Asli   :", " ".join(teks.split()))
    print("  Bersih :", bersihkan(teks))
""")

md(r"""
Lihat contoh 2: kata gaul `gw` dan `bgt` **tidak** dinormalkan. Pipeline klasik
memang rapuh terhadap bahasa informal — daftar stopword dan kamus lemma disusun
dari teks baku. Kalimat yang sama akan kita uji lagi di Section 7 dengan model
transformer, dan hasilnya berbeda jauh.
""")

# --- 2
md(r"""
## 2. Penandaan Kelas Kata (POS Tagging)

**POS tagging** (*part-of-speech tagging*) menandai setiap token dengan kelas
katanya: kata benda, kata kerja, kata sifat, dan seterusnya. Gunanya: memilih
hanya kata benda sebagai kandidat topik, atau memisahkan kata kerja untuk
analisis tindakan.

`PosTag` dari `nlp-id` menerima **satu string**, bukan daftar token — jadi hasil
tokenisasi kita gabungkan kembali dengan spasi.
""")

code(r"""
kalimat_pos = "Saya sedang belajar pemrosesan bahasa alami di universitas."

kalimat_rapi = " ".join(tokenizer.tokenize(kalimat_pos))
tag_hasil = postagger.get_pos_tag(kalimat_rapi)

print("Hasil POS tagging:")
for token, tag in tag_hasil:
    print(f"  {token:<15} {tag}")
""")

md(r"""
Singkatan tag yang muncul di keluaran `nlp-id`: `NN` kata benda, `VB` kata kerja,
`PR` kata ganti, `ADV` keterangan, `IN` preposisi, `SYM` tanda baca.

Dua hal yang layak dilihat pada hasil di atas. Pertama, "pemrosesan bahasa alami"
ditandai per kata — padahal tiga kata itu satu istilah, dan Section berikutnya
menangani persoalan itu. Kedua, `alami` ditandai `VB` padahal di kalimat ini ia
kata sifat. POS tagger adalah model statistik, bukan kamus: ia menebak, dan
kadang tebakannya meleset.
""")

# --- 3
md(r"""
## 3. Tokenisasi Frasa

Kalimat bukan sekadar kumpulan kata lepas. "Universitas Indonesia" adalah satu
nama; memecahnya jadi "universitas" + "indonesia" menghilangkan maknanya.
`PhraseTokenizer` mengelompokkan frasa nomina menjadi satu token.
""")

code(r"""
kalimat_frasa = "Universitas Indonesia adalah kampus terbaik di Jakarta."

print("Tokenisasi kata  :", tokenizer.tokenize(kalimat_frasa))
print("Tokenisasi frasa :", phrase_tokenizer.tokenize(kalimat_frasa))
""")

# --- 4
md(r"""
## 4. Stopword dan Lemmatisasi

**Stopword** adalah kata yang muncul di hampir semua dokumen sehingga nyaris tidak
membedakan apa pun: `yang`, `di`, `dari`, `dan`, `adalah`. Membuangnya memperkecil
dimensi fitur dan memusatkan analisis pada kata yang membawa isi.

**Lemmatisasi** mengembalikan kata ke bentuk dasarnya berdasarkan kamus dan aturan
morfologi — per kata, tanpa melihat konteks kalimat. Bedanya dengan **stemming**:
stemming memotong imbuhan secara mekanis (`mempelajari` bisa jadi `pelajar` atau
bahkan potongan yang bukan kata), sedangkan lemmatisasi mengembalikan kata yang
benar-benar ada di kamus. Lemmatisasi lebih akurat, tetapi lebih lambat.
""")

code(r"""
print("20 stopword Indonesia pertama:")
print(sorted(STOPWORD_ID)[:20])
print(f"\nTotal stopword: {len(STOPWORD_ID)}")
""")

code(r"""
kata_uji = ["memakan", "bermain", "berlari", "menulis", "membaca",
            "mempelajari", "mengembangkan", "bekerja", "berjalan", "kebersihan"]

print("Hasil lemmatisasi:")
for kata in kata_uji:
    print(f"  {kata:<15} -> {lemmatizer.lemmatize(kata)}")
""")

# ================================================================ BAGIAN B
md(r"""
---

# Bagian B — Menganalisis Teks

Teks sudah bersih. Sekarang kita gali isinya: kata apa yang dominan, dan
entitas apa saja yang disebut.
""")

# --- 5
md(r"""
## 5. Frekuensi Kata dan Word Cloud

Menghitung kata yang paling sering muncul adalah cara tercepat menebak topik
sebuah kumpulan dokumen. Syaratnya satu: **teks harus dibersihkan dulu**. Tanpa
membuang stopword, sepuluh kata teratas dokumen apa pun hanya akan berisi
`yang`, `di`, `dan`.
""")

code(r"""
dokumen_contoh = [
    "Teknologi NLP semakin berkembang di Indonesia.",
    "Para peneliti terus mengembangkan model bahasa yang lebih baik.",
    "Aplikasi berbasis NLP sangat membantu dalam pekerjaan sehari-hari.",
    "Preprocessing teks adalah langkah penting dalam NLP.",
    "Banyak perusahaan menggunakan NLP untuk analisis sentimen.",
    "Model bahasa Indonesia dilatih dari korpus berita dan Wikipedia.",
    "Analisis sentimen membantu perusahaan memahami keluhan pelanggan.",
]

token_semua = []
for teks in dokumen_contoh:
    token_semua.extend(bersihkan(teks))

frekuensi = Counter(token_semua)
print(f"Total token setelah dibersihkan : {len(token_semua)}")
print(f"Token unik                      : {len(frekuensi)}")
print("\n10 kata paling sering muncul:")
for kata, n in frekuensi.most_common(10):
    print(f"  {kata:<15} {n}")
""")

code(r"""
df_frekuensi = pd.DataFrame(frekuensi.most_common(12), columns=["kata", "frekuensi"])

plt.figure(figsize=(9, 5))
sns.barplot(x="frekuensi", y="kata", data=df_frekuensi, color="#76b900")
plt.title("Kata paling sering muncul (setelah pembersihan)")
plt.xlabel("Frekuensi")
plt.ylabel("")
plt.tight_layout()
plt.show()
""")

code(r"""
from wordcloud import WordCloud

wc = WordCloud(width=900, height=450, background_color="white",
               colormap="summer", random_state=42).generate(" ".join(token_semua))

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word cloud — ukuran kata sebanding dengan frekuensinya")
plt.tight_layout()
plt.show()
""")

md(r"""
Word cloud enak dipandang, tetapi mata sulit membandingkan ukuran huruf secara
akurat. Untuk mengambil keputusan, bar chart di atas lebih bisa dipercaya;
word cloud lebih cocok sebagai ilustrasi.
""")

# --- 6
md(r"""
## 6. Pengenalan Entitas Bernama (NER)

**NER** (*named entity recognition*) menemukan dan mengelompokkan entitas yang
disebut dalam teks: nama orang, organisasi, lokasi, tanggal, nilai uang. Ini
tulang punggung banyak sistem nyata — ekstraksi data dari dokumen kontrak,
pemantauan nama perusahaan di berita, pengisian formulir otomatis.

Kita pakai **spaCy** dengan model Inggris `en_core_web_sm`, karena model spaCy
untuk bahasa Indonesia belum tersedia secara resmi.
""")

code(r"""
import spacy

nlp_en = spacy.load("en_core_web_sm")

teks_ner = ("Nvidia was founded by Jensen Huang in 1993 and is headquartered in Santa Clara, "
            "California. The company announced a partnership with Google Cloud last March.")

doc = nlp_en(teks_ner)

print(f"{'entitas':<22} {'label':<10} penjelasan")
print("-" * 60)
for ent in doc.ents:
    print(f"{ent.text:<22} {ent.label_:<10} {spacy.explain(ent.label_)}")
""")

md(r"""
Perhatikan bahwa `Nvidia` dilabeli **GPE** (wilayah geopolitik), bukan `ORG`.
Model kecil `en_core_web_sm` memang sering keliru pada nama perusahaan yang mirip
nama tempat. Ini pengingat berguna: keluaran NER adalah tebakan model, dan untuk
dipakai di produksi ia perlu diperiksa, bukan langsung dipercaya.

**Bagaimana dengan bahasa Indonesia?** Ada tiga jalur yang lazim dipakai:

1. Model multibahasa spaCy `xx_ent_wiki_sm` — cakupannya luas, akurasinya sedang
2. Model NER Indonesia di Hugging Face Hub, misalnya keluarga **IndoBERT** yang
   sudah di-*fine-tune* untuk NER
3. `PhraseTokenizer` dari Section 3 sebagai pendekatan sederhana — ia menangkap
   frasa nomina, meskipun tidak memberi label jenis entitas

Pola pemakaian model Hugging Face akan kamu lihat di Section 7 berikut ini.
""")

# ================================================================ BAGIAN C
md(r"""
---

# Bagian C — Menilai Teks

Dari menggambarkan teks, kita naik ke memberi penilaian: apa nada kalimat ini,
dan masuk kategori apa dokumen ini.
""")

# --- 7
md(r"""
## 7. Analisis Sentimen: Leksikon vs Transformer

**Analisis sentimen** menentukan apakah sebuah teks bernada positif, negatif,
atau netral. Ada dua generasi pendekatan, dan membandingkannya langsung adalah
cara tercepat memahami kenapa NLP pindah ke transformer.

**Generasi 1 — leksikon.** TextBlob menyimpan kamus kata beserta skor
sentimennya, lalu menjumlahkan skor kata yang muncul. Cepat, transparan, tanpa
GPU. Keluarannya dua angka:

- `polarity` ∈ [-1, 1] — negatif sampai positif
- `subjectivity` ∈ [0, 1] — faktual sampai opini
""")

code(r"""
from textblob import TextBlob

kalimat_en = [
    "I love natural language processing!",
    "The service was slow and disappointing.",
    "The meeting has been rescheduled to Thursday.",
    "The plot was not good.",
]

print(f"{'polarity':>9} {'subjectivity':>13}  kalimat")
print("-" * 78)
for kalimat in kalimat_en:
    b = TextBlob(kalimat).sentiment
    print(f"{b.polarity:>9.2f} {b.subjectivity:>13.2f}  {kalimat}")
""")

md(r"""
Empat kalimat pertama ditangani dengan benar, termasuk `"The plot was not good."`
TextBlob memang punya aturan negasi sederhana: kata `not` membalik skor kata
**tepat sesudahnya**.

Batasnya baru terlihat kalau negasinya berjarak, atau kalau kalimatnya sarkastis.
Jalankan sel berikut dan perhatikan ketiga skornya.
""")

code(r"""
gagal = [
    "I would not call this a great movie.",          # panning, tapi negasinya berjarak
    "Nobody would say this restaurant is bad.",      # pujian berbentuk negasi
    "Oh great, another meeting that could have been an email.",   # sarkasme
]

print(f"{'polarity':>9}  kalimat")
print("-" * 74)
for kalimat in gagal:
    print(f"{TextBlob(kalimat).sentiment.polarity:>+9.2f}  {kalimat}")
""")

md(r"""
Ketiganya meleset, dan dua di antaranya meleset dengan yakin:

- `"I would not call this a great movie."` adalah kritik, tetapi diberi skor
  **positif kuat**. `not` berjarak tiga kata dari `great`, di luar jangkauan
  aturan negasinya, jadi yang terhitung tinggal `great`.
- `"Nobody would say this restaurant is bad."` justru pujian, tetapi diberi skor
  **negatif kuat** — `bad` terhitung apa adanya, dan `nobody` tidak dianggap pembalik.
- Sarkasme meleset sepenuhnya, karena isyaratnya ada di konteks, bukan di kata.

Polanya satu: leksikon menjumlahkan skor kata dan hanya mengenali pola negasi
yang sangat dekat. Begitu makna bergantung pada susunan kalimat, ia kehilangan jejak.

Masalah kedua lebih dekat ke kita: TextBlob **hanya punya kamus bahasa Inggris**.
Coba kalimat Indonesia dan hasilnya nol — bukan karena netral, melainkan karena
tidak ada satu pun katanya yang dikenali.
""")

code(r"""
kalimat_id = "Pelayanan restoran itu lambat dan mengecewakan"
print(f"TextBlob pada kalimat Indonesia -> polarity = {TextBlob(kalimat_id).sentiment.polarity:.2f}")
print("Nol di sini berarti 'tidak ada kata yang dikenali', bukan 'netral'.")
""")

md(r"""
**Generasi 2 — transformer.** Cara industri Indonesia mengerjakan ini sekarang:
model keluarga **IndoBERT** yang sudah di-*fine-tune* pada data sentimen berbahasa
Indonesia, dipanggil lewat `pipeline` Hugging Face. Polanya persis transfer
learning yang kamu lihat di Modul 02 dengan ResNet: model besar yang sudah
dilatih orang lain, tinggal dipakai.
""")

code(r"""
from transformers import pipeline

analisis_sentimen = pipeline(
    "sentiment-analysis",
    model="w11wo/indonesian-roberta-base-sentiment-classifier",
)

uji_id = [
    "Gw suka bgt sama teknologi NLP nih",            # kalimat gaul dari Section 1
    "Pelayanan restoran itu lambat dan mengecewakan",
    "Rapat dijadwalkan ulang ke hari Kamis",
]

print(f"{'label':>8} {'skor':>6}  kalimat")
print("-" * 60)
for teks in uji_id:
    hasil = analisis_sentimen(teks)[0]
    print(f"{hasil['label']:>8} {hasil['score']:>6.2f}  {teks}")
""")

md(r"""
Tiga hal yang layak dicatat:

1. Kalimat gaul "Gw suka bgt..." — yang gagal dinormalkan pipeline klasik di
   Section 1 — **langsung dikenali positif**. Model transformer dilatih dari teks
   nyata termasuk bahasa informal, jadi jauh lebih tahan slang.
2. Model mengenali kelas **netral**, bukan sekadar positif/negatif.
3. Ekosistem NLP Indonesia lebih luas dari satu model ini: **IndoNLU** (benchmark),
   **IndoLEM**, dan **NusaBERT** semuanya tersedia di Hugging Face Hub.

Harganya: model transformer butuh unduhan ratusan MB dan jauh lebih lambat di CPU.
Untuk penyaringan kasar jutaan dokumen, leksikon masih punya tempat.
""")

# --- 8
md(r"""
## 8. Klasifikasi Teks dengan Naive Bayes

Sentimen di Section 7 memakai model siap pakai. Sekarang kita **melatih classifier
sendiri** dari nol, dengan algoritma paling klasik untuk teks: **Naive Bayes**.

Datanya `movie_reviews` bawaan NLTK — 2.000 review film Inggris berlabel `pos`/`neg`.

**Cara kerjanya.** Setiap dokumen diubah jadi **bag-of-words**: daftar
`contains(kata) = True/False` untuk sekumpulan kata yang kita pilih sebagai fitur.
Urutan kata dibuang total; yang dihitung hanya kehadiran. Lalu Teorema Bayes
menghitung P(kelas | fitur) dan kelas dengan probabilitas tertinggi menang.
""")

code(r"""
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier

dokumen = [(list(movie_reviews.words(fid)), kategori)
           for kategori in movie_reviews.categories()
           for fid in movie_reviews.fileids(kategori)]

random.seed(42)
random.shuffle(dokumen)

N_TEST = 400
dok_test, dok_train = dokumen[:N_TEST], dokumen[N_TEST:]
print(f"Dokumen train : {len(dok_train)}")
print(f"Dokumen test  : {len(dok_test)}")
""")

md(r"""
### Satu langkah yang gampang salah

Fitur kita adalah "2.000 kata paling sering muncul". Pertanyaannya: **sering muncul
di data yang mana?**

Kalau daftar kata itu dihitung dari **seluruh korpus**, informasi dari data test
sudah bocor ke dalam desain fitur sebelum model dilatih — namanya **kebocoran data**
(*data leakage*), dan akurasi yang keluar jadi terlalu optimistis. Daftar fitur
harus dihitung dari **data train saja**, seperti di sel berikut.

Aturan yang sama berlaku untuk semua langkah yang "belajar" dari data:
`TfidfVectorizer`, `StandardScaler`, seleksi fitur. Semuanya di-`fit` pada train,
lalu diterapkan ke test.
""")

code(r"""
# PENTING: kosakata fitur dihitung HANYA dari data train.
kata_train = (w.lower() for dok, _ in dok_train for w in dok)
fitur_kata = [w for w, _ in nltk.FreqDist(kata_train).most_common(2000)]

def ekstrak_fitur(dokumen_kata):
    # Ubah satu dokumen menjadi bag-of-words biner atas 2.000 kata fitur.
    ada = set(w.lower() for w in dokumen_kata)
    return {f"contains({w})": (w in ada) for w in fitur_kata}

set_train = [(ekstrak_fitur(d), k) for d, k in dok_train]
set_test = [(ekstrak_fitur(d), k) for d, k in dok_test]
print(f"Jumlah fitur per dokumen: {len(fitur_kata)}")
""")

code(r"""
classifier = NaiveBayesClassifier.train(set_train)

akurasi = nltk.classify.accuracy(classifier, set_test)
baseline = max(Counter(k for _, k in dok_test).values()) / len(dok_test)

print(f"Akurasi Naive Bayes    : {akurasi:.1%}")
print(f"Baseline kelas mayoritas: {baseline:.1%}")
print(f"Selisih terhadap baseline: {akurasi - baseline:+.1%}")
""")

md(r"""
Selalu bandingkan dengan **baseline kelas mayoritas** — akurasi 85% terdengar bagus
sampai kamu sadar 84% dokumen memang satu kelas. Korpus `movie_reviews` seimbang
(1.000 lawan 1.000), jadi baseline-nya mendekati 50%; angka persisnya bergantung
pada bagaimana 400 dokumen test kebetulan terbagi, dan itu yang dicetak sel di atas.
""")

code(r"""
print("Fitur paling informatif:")
classifier.show_most_informative_features(10)
""")

md(r"""
Baca hasilnya begini: `neg : pos = 12.0 : 1.0` berarti kata itu 12 kali lebih
sering muncul di review negatif daripada positif. Inilah keunggulan Naive Bayes —
keputusannya bisa dibaca manusia, tidak seperti neural network.

Kelemahannya juga terlihat dari sini: karena urutan kata dibuang, "not good"
dan "good" memberi sinyal yang sama. Persoalan yang akan diselesaikan oleh
representasi di Bagian D.
""")

# ================================================================ BAGIAN D
md(r"""
---

# Bagian D — Dari Kata ke Makna

Sejauh ini kita selalu **menghitung kata**. Bagian ini mengganti hitungan dengan
**vektor makna** — dan di situlah jembatan menuju Modul 04 dan Modul 05.
""")

# --- 9
md(r"""
## 9. Dari Kata ke Vektor: TF-IDF, Embedding, Semantic Search

Komputer tidak paham "makna" dari hitungan kata. Cara modern: ubah teks jadi
**vektor angka** sedemikian rupa sehingga teks yang maknanya mirip menghasilkan
vektor yang berdekatan. Kita naik tiga anak tangga.

**Anak tangga 1 — TF-IDF.** Perbaikan atas bag-of-words: kata yang sering muncul
di satu dokumen tapi jarang di dokumen lain diberi bobot tinggi, karena kata
seperti itulah yang membedakan.
""")

code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

kalimat = [
    "Saya suka makan nasi goreng",
    "Aku gemar menyantap nasi goreng",       # makna ~sama dengan K1, kata berbeda
    "Nasi goreng adalah makanan favorit saya",
    "Harga BBM naik lagi minggu ini",         # topik berbeda total
]
label = [f"K{i+1}" for i in range(len(kalimat))]

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(kalimat)
print("Ukuran matriks TF-IDF:", X_tfidf.shape)
print("Kosakata:", list(vectorizer.get_feature_names_out()))
""")

code(r"""
sim_tfidf = cosine_similarity(X_tfidf)

print("Cosine similarity antar kalimat (TF-IDF):")
print(pd.DataFrame(sim_tfidf.round(2), index=label, columns=label))
print()
print(f"K1 vs K2 (sinonim, kata berbeda) : {sim_tfidf[0, 1]:.2f}  <- rendah")
print(f"K1 vs K3 (banyak kata sama)      : {sim_tfidf[0, 2]:.2f}")
""")

md(r"""
### Kenapa TF-IDF buta terhadap sinonim?

K1 ("suka makan") dan K2 ("gemar menyantap") maknanya hampir identik, tetapi
skornya rendah — TF-IDF hanya mencocokkan **kata yang persis sama**. Baginya
"suka" dan "gemar" adalah dua dimensi yang tidak berhubungan sama sekali.

**Anak tangga 2 — embedding.** Model neural network yang dilatih pada jutaan
kalimat sampai belajar bahwa "suka" ≈ "gemar". Hasilnya: satu kalimat menjadi
satu vektor (384 angka di model ini) yang menangkap makna, bukan ejaan.
""")

code(r"""
from sentence_transformers import SentenceTransformer

model_embedding = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
emb = model_embedding.encode(kalimat)
print("Bentuk embedding:", emb.shape, "-> 4 kalimat, masing-masing 384 angka")
""")

code(r"""
sim_emb = cosine_similarity(emb)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.heatmap(sim_tfidf, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=label, yticklabels=label, ax=axes[0], vmin=0, vmax=1)
axes[0].set_title("TF-IDF: K1 dan K2 dianggap berbeda")
sns.heatmap(sim_emb, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=label, yticklabels=label, ax=axes[1], vmin=0, vmax=1)
axes[1].set_title("Embedding: K1 dan K2 terdeteksi semakna")
plt.tight_layout()
plt.show()

print(f"K1 vs K2 — TF-IDF {sim_tfidf[0, 1]:.2f}  ->  embedding {sim_emb[0, 1]:.2f}")
""")

md(r"""
**Anak tangga 3 — semantic search.** Aplikasi nyatanya. Kita punya sepuluh
"dokumen" fakta tentang Indonesia. Diberi pertanyaan, kita cari dokumen paling
relevan — bukan dengan mencocokkan kata, melainkan dengan membandingkan vektornya.
""")

code(r"""
dokumen_id = [
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

emb_dokumen = model_embedding.encode(dokumen_id)

for query in ["puncak paling tinggi di Indonesia", "makanan khas Sumatera"]:
    skor = cosine_similarity(model_embedding.encode([query]), emb_dokumen)[0]
    peringkat = skor.argsort()[::-1]
    print(f"\nQuery: {query!r}")
    for i in peringkat[:3]:
        print(f"  {skor[i]:.2f}  {dokumen_id[i]}")
""")

md(r"""
Query "puncak paling tinggi" menemukan dokumen "gunung berapi tertinggi" —
**tanpa satu pun kata yang sama**. Itulah kekuatan semantic search, dan itu
mustahil dilakukan TF-IDF.

Kamu baru saja membangun inti **RAG (Retrieval-Augmented Generation)** dalam
sekitar sepuluh baris: encode dokumen, encode query, hitung cosine similarity,
ambil top-k. Di Modul 05 pola yang persis sama dipakai lagi — bedanya dokumen
disimpan di **FAISS** agar pencarian tetap cepat untuk jutaan dokumen, dan hasil
pencariannya diserahkan ke LLM untuk menyusun jawaban.
""")

# --- 10
md(r"""
## 10. Tokenisasi ala LLM: Subword

Di Section 1, satu token sama dengan satu kata. LLM modern (GPT, BERT, Llama)
memakai **tokenisasi subword** (BPE/WordPiece): kata dipecah menjadi potongan
yang lebih kecil dari kata tetapi lebih besar dari huruf.

Kenapa? Dengan kosakata terbatas (sekitar 30–50 ribu subword), model tetap bisa
menuliskan **kata apa pun** — termasuk kata yang belum pernah dilihat saat
training. Untuk bahasa Indonesia yang kaya imbuhan, ini relevan sekali:
`mempelajari`, `dipelajari`, dan `pelajaran` berbagi potongan yang sama.
""")

code(r"""
from transformers import AutoTokenizer

tok_gpt2 = AutoTokenizer.from_pretrained("gpt2")
tok_multi = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

for kata in ["mempelajari", "kebersihan"]:
    print(f"{kata!r}")
    print(f"   GPT-2        : {tok_gpt2.tokenize(kata)}")
    print(f"   Multilingual : {tok_multi.tokenize(kata)}")
""")

code(r"""
pasangan = [
    ("Saya sedang mempelajari kecerdasan buatan di kelas ini",
     "I am learning artificial intelligence in this class"),
    ("Pemerintah mengumumkan kebijakan baru minggu depan",
     "The government announced a new policy next week"),
]

print(f"{'tokenizer':<14} {'Indonesia':>10} {'Inggris':>9} {'rasio':>7}")
print("-" * 44)
for nama, tok in [("GPT-2", tok_gpt2), ("Multilingual", tok_multi)]:
    n_id = sum(len(tok(a)["input_ids"]) for a, _ in pasangan)
    n_en = sum(len(tok(b)["input_ids"]) for _, b in pasangan)
    print(f"{nama:<14} {n_id:>10} {n_en:>9} {n_id / n_en:>6.2f}x")
""")

md(r"""
Dua pengamatan:

1. **GPT-2 memecah kata Indonesia jadi lebih banyak potongan** daripada kalimat
   Inggris yang semakna, karena tokenizer-nya dilatih dominan pada teks Inggris.
   Akibat praktisnya langsung terasa di dompet: API LLM menagih per token, jadi
   kalimat Indonesia lebih mahal. Tokenizer multibahasa biasanya menekan rasio itu.
2. Mulai Modul 04, setiap kali kamu bertemu istilah "token" — misalnya
   `max_new_tokens` — yang dimaksud adalah subword seperti ini, bukan kata utuh
   hasil `tokenizer.tokenize()` di Section 1.
""")

# ---------------------------------------------------------------- latihan
md(r"""
---

## 🏋️ Latihan

1. Pada `bersihkan()` di Section 1, **balik urutannya**: lemmatisasi dulu, baru buang
   stopword. Jalankan pada `contoh_teks` dan bandingkan hasilnya dengan versi asli.
   Token mana yang berubah nasibnya, dan kenapa urutan itu berpengaruh?
2. Di Section 8, ubah `most_common(2000)` menjadi `most_common(300)`, latih ulang
   classifier, lalu bandingkan akurasinya. Apakah fitur yang lebih sedikit membuat
   model lebih buruk? Jelaskan satu–dua kalimat.
3. Di Section 9, tambahkan satu kalimat baru ke daftar `kalimat` yang **semakna
   dengan K4** tetapi memakai kata yang sama sekali berbeda (misalnya soal harga
   bahan bakar). Hitung ulang kedua similarity, lalu tunjukkan berapa selisih skor
   TF-IDF dan embedding untuk pasangan itu.
4. Di Section 10, ganti `bert-base-multilingual-cased` dengan tokenizer model
   Indonesia (`indobenchmark/indobert-base-p1`) dan hitung ulang tabel rasionya.
   Apakah tokenizer khusus Indonesia lebih hemat token dibanding yang multibahasa?

**Petunjuk untuk tugas 2:** `fitur_kata`, `set_train`, dan `set_test` semuanya harus
dibangun ulang — mengganti angka saja tidak cukup, karena ketiganya sudah terlanjur
dihitung dengan 2.000 kata.
""")

# ---------------------------------------------------------------- ringkasan
md(r"""
## Ringkasan

| Bagian | Section | Yang dikuasai | Library |
|---|---|---|---|
| A | 1 | Pipeline pembersihan teks Indonesia | `nlp-id` |
| A | 2 | Penandaan kelas kata (POS tagging) | `nlp-id` |
| A | 3 | Tokenisasi frasa multi-kata | `nlp-id` |
| A | 4 | Stopword dan lemmatisasi | `nlp-id` |
| B | 5 | Frekuensi kata dan word cloud | `nltk`, `wordcloud` |
| B | 6 | Pengenalan entitas bernama (NER) | `spaCy` |
| C | 7 | Sentimen: leksikon vs transformer | `textblob`, `transformers` |
| C | 8 | Klasifikasi Naive Bayes tanpa kebocoran data | `nltk` |
| D | 9 | TF-IDF, embedding, semantic search | `scikit-learn`, `sentence-transformers` |
| D | 10 | Tokenisasi subword ala LLM | `transformers` |

**Tiga hal yang paling penting dibawa pulang:**

1. **Pembersihan menentukan segalanya.** Tanpa membuang stopword, analisis frekuensi
   apa pun hanya akan memunculkan `yang` dan `di`.
2. **Apa pun yang belajar dari data harus di-`fit` pada train saja.** Daftar fitur,
   scaler, vectorizer — semuanya. Melanggarnya menghasilkan akurasi palsu.
3. **Hitungan kata tidak sama dengan makna.** TF-IDF buta terhadap sinonim;
   embedding tidak. Perbedaan itulah yang membuat RAG mungkin.

**Catatan library:** `nlp-id` adalah pilihan utama untuk bahasa Indonesia dan masih
aktif dirawat. `Sastrawi` masih sering dikutip untuk stemming, tetapi rilis
terakhirnya tahun 2016 — perlakukan sebagai referensi historis, bukan pilihan baru.

**Langkah selanjutnya:**

- **Notebook 02 (NLP on Steroids)** — semua yang barusan kamu kerjakan, dijalankan di
  GPU dengan NVIDIA RAPIDS, pada korpus 150 ribu paragraf Wikipedia
- **Modul 04 (LLM)** — istilah "token" dari Section 10 dan `pipeline` dari Section 7
  menjadi pintu masukmu
- **Modul 05 (RAG)** — pola embedding + semantic search dari Section 9 dipakai lagi,
  kali ini dengan FAISS
""")

nb = {
    "cells": C,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = ROOT / "01_nlp_fundamentals.ipynb"
save(nb, out)
print(f"OK: {out} — {len(C)} sel "
      f"(md={sum(1 for c in C if c['cell_type']=='markdown')}, "
      f"code={sum(1 for c in C if c['cell_type']=='code')})")
