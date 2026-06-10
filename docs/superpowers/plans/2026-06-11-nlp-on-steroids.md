# NLP on Steroids (RAPIDS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Notebook hands-on dual-bahasa "NLP on Steroids" (NLP klasik di GPU dengan RAPIDS) untuk Day 03, plus slide, quiz, dan cheatsheet pendukung.

**Architecture:** Kedua notebook (ID+EN) dibangun dari satu generator Python bilingual (sumber kebenaran tunggal, cell didefinisikan sebagai pasangan ID/EN). Materi pendukung diedit langsung pada file yang ada. Angka benchmark mengalir dari smoke-test Colab → `benchmark_results.json` → figure slide.

**Tech Stack:** RAPIDS (cuDF/nvtext/cuML, pre-installed Colab), HF `datasets` (streaming Wikipedia ID), SmSA IndoNLU (raw GitHub TSV), XeLaTeX beamer, matplotlib dark-theme.

**Spec:** `docs/superpowers/specs/2026-06-11-nlp-on-steroids-design.md`

---

## Catatan untuk pelaksana

- Semua path relatif terhadap root repo `/Users/chmdznr/work/navasena/navasena-gen-ml-course`.
- Commit TANPA footer atribusi Claude (config user).
- Mesin lokal (Mac) tidak punya GPU NVIDIA — notebook TIDAK bisa dieksekusi lokal. Validasi lokal = struktur JSON + lint visual. Eksekusi penuh = smoke-test Colab T4 (Task 6, gerbang manual oleh instruktur).
- API minhash cuDF berubah antar versi → kode notebook memakai helper defensif (sudah ada di generator).

---

### Task 1: Verifikasi & pin URL dataset SmSA

**Files:** tidak ada (hasil dipakai Task 2)

- [ ] **Step 1: Resolve commit hash master IndoNLU**

```bash
git ls-remote https://github.com/IndoNLP/indonlu.git master
```

Expected: satu baris `<sha40>\trefs/heads/master`. Catat `<sha40>`.

- [ ] **Step 2: Verifikasi kedua URL TSV dengan hash ter-pin**

```bash
SHA=<sha40 dari step 1>
curl -sI "https://raw.githubusercontent.com/IndoNLP/indonlu/$SHA/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv" | head -1
curl -sI "https://raw.githubusercontent.com/IndoNLP/indonlu/$SHA/dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv" | head -1
```

Expected: `HTTP/2 200` untuk keduanya. Jika 404, cek nama path di repo IndoNLU (`dataset/` listing) dan sesuaikan; lalu substitusikan URL final ke variabel `SMSA_TRAIN_URL`/`SMSA_VALID_URL` di generator Task 2.

### Task 2: Generator bilingual + kedua notebook

**Files:**
- Create: `/tmp/gen_steroids_nb.py` (build tool, tidak di-commit)
- Create: `03_nlp_fundamentals/02_nlp_on_steroids_id.ipynb`
- Create: `03_nlp_fundamentals/02_nlp_on_steroids_en.ipynb`

- [ ] **Step 1: Tulis generator** — `/tmp/gen_steroids_nb.py`. `M(id,en)` = markdown, `C(id,en=None)` = code (EN default sama dengan ID). Ganti `{SHA}` dengan hash dari Task 1.

```python
#!/usr/bin/env python3
"""Generate 02_nlp_on_steroids_{id,en}.ipynb (Module 03, RAPIDS)."""
import json, pathlib

OUT = pathlib.Path("/Users/chmdznr/work/navasena/navasena-gen-ml-course/03_nlp_fundamentals")
SHA = "{SHA}"  # dari Task 1
TRAIN = f"https://raw.githubusercontent.com/IndoNLP/indonlu/{SHA}/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv"
VALID = f"https://raw.githubusercontent.com/IndoNLP/indonlu/{SHA}/dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv"

def M(id_, en): return ("markdown", id_, en)
def C(id_, en=None): return ("code", id_, en or id_)

CELLS = [
# ============ BABAK 0 — SETUP ============
M(f"""# NLP on Steroids 💉 — NLP Klasik di GPU dengan NVIDIA RAPIDS

**Modul 03 · Notebook 02 · Bootcamp NCA-GENL**

Di notebook 01 kita belajar alur NLP klasik: *preprocessing → vektorisasi → klasifikasi*.
Di notebook ini kita jalankan alur yang **sama persis** — tapi di **GPU**, dengan stack **NVIDIA RAPIDS**:

| Library | Peran |
|---|---|
| **cuDF** | pandas versi GPU (DataFrame) |
| **nvtext** | operasi teks GPU di dalam cuDF (tokenize, n-gram, minhash) |
| **cuML** | scikit-learn versi GPU (TF-IDF, Logistic Regression, dll.) |

> ⚠️ **Wajib GPU**: Runtime → Change runtime type → **T4 GPU**, baru jalankan sel di bawah.

**Yang akan kamu kuasai:**
1. Menjalankan kode pandas/sklearn lamamu di GPU **tanpa mengubah satu baris pun**
2. Preprocessing ±150 ribu paragraf Wikipedia dalam hitungan detik
3. Deduplikasi *near-duplicate* dengan MinHash — langkah nyata penyiapan korpus LLM
4. TF-IDF + klasifikasi sentimen dengan API native cuML""",
f"""# NLP on Steroids 💉 — Classic NLP on the GPU with NVIDIA RAPIDS

**Module 03 · Notebook 02 · NCA-GENL Bootcamp**

In notebook 01 we learned the classic NLP flow: *preprocessing → vectorization → classification*.
In this notebook we run the **exact same flow** — on the **GPU**, using the **NVIDIA RAPIDS** stack:

| Library | Role |
|---|---|
| **cuDF** | GPU version of pandas (DataFrame) |
| **nvtext** | GPU text ops inside cuDF (tokenize, n-gram, minhash) |
| **cuML** | GPU version of scikit-learn (TF-IDF, Logistic Regression, etc.) |

> ⚠️ **GPU required**: Runtime → Change runtime type → **T4 GPU** before running the cells below.

**What you will master:**
1. Running your existing pandas/sklearn code on the GPU **without changing a single line**
2. Preprocessing ±150k Wikipedia paragraphs in seconds
3. Near-duplicate deduplication with MinHash — a real step in LLM corpus preparation
4. TF-IDF + sentiment classification with the native cuML API"""),

C("""# Cek GPU — harus muncul Tesla T4
!nvidia-smi""",
"""# Check the GPU — you should see a Tesla T4
!nvidia-smi"""),

C("""# RAPIDS sudah pre-installed di runtime GPU Colab — tinggal import.
# Kalau sel ini error, jalankan sel fallback di bawahnya.
import cudf, cuml
print("cuDF :", cudf.__version__)
print("cuML :", cuml.__version__)""",
"""# RAPIDS comes pre-installed on Colab GPU runtimes — just import it.
# If this cell fails, run the fallback cell below.
import cudf, cuml
print("cuDF :", cudf.__version__)
print("cuML :", cuml.__version__)"""),

C("""# FALLBACK — jalankan HANYA jika import di atas gagal (butuh ±5 menit):
# !pip install -q --extra-index-url=https://pypi.nvidia.com cudf-cu12 cuml-cu12
pass""",
"""# FALLBACK — run ONLY if the import above failed (takes ±5 min):
# !pip install -q --extra-index-url=https://pypi.nvidia.com cudf-cu12 cuml-cu12
pass"""),

# ============ BABAK 1 — SULAP ============
M("""## Babak 1 — Sulap: Kode Lama, GPU Baru 🎩

Klaim NVIDIA: kode pandas + sklearn yang sudah kamu tulis bisa langsung lari di GPU, **nol perubahan kode**, lewat dua "akselerator":

- `cudf.pandas` — membajak `import pandas`, semua operasi dialihkan ke cuDF (GPU); yang tidak didukung otomatis *fallback* ke pandas asli.
- `cuml.accel` — hal yang sama untuk estimator scikit-learn → cuML.

**Caranya bukan mengubah kode, tapi mengubah cara MENJALANKAN-nya.** Karena akselerator harus aktif *sebelum* `import pandas`, kita tulis pipeline sebagai script lalu jalankan dua kali: normal (CPU) dan lewat akselerator (GPU).

Data: **SmSA** dari benchmark IndoNLU — ulasan berbahasa Indonesia berlabel sentimen (positive/negative/neutral), dataset yang sama yang disebut di notebook 01.""",
"""## Act 1 — The Magic Trick: Old Code, New GPU 🎩

NVIDIA's claim: the pandas + sklearn code you already wrote can run on the GPU with **zero code changes**, via two "accelerators":

- `cudf.pandas` — hijacks `import pandas`; every operation is routed to cuDF (GPU); unsupported ops automatically *fall back* to real pandas.
- `cuml.accel` — the same idea for scikit-learn estimators → cuML.

**The trick is not changing the code, but changing how you RUN it.** Because the accelerator must be active *before* `import pandas`, we write the pipeline as a script and run it twice: normally (CPU) and through the accelerator (GPU).

Data: **SmSA** from the IndoNLU benchmark — Indonesian reviews labeled with sentiment (positive/negative/neutral), the same dataset mentioned in notebook 01."""),

C(f"""# Unduh SmSA (URL di-pin ke commit agar stabil)
!wget -q "{TRAIN}" -O smsa_train.tsv
!wget -q "{VALID}" -O smsa_valid.tsv
!wc -l smsa_train.tsv smsa_valid.tsv""",
f"""# Download SmSA (URL pinned to a commit for stability)
!wget -q "{TRAIN}" -O smsa_train.tsv
!wget -q "{VALID}" -O smsa_valid.tsv
!wc -l smsa_train.tsv smsa_valid.tsv"""),

C('''%%writefile pipeline.py
# Pipeline klasik notebook 01: TF-IDF + Logistic Regression.
# TIDAK ADA satu baris pun kode GPU di file ini.
import time
t_all = time.perf_counter()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv("smsa_train.tsv", sep="\\t", names=["text", "label"])
valid = pd.read_csv("smsa_valid.tsv", sep="\\t", names=["text", "label"])
# Gandakan data train 20x: simulasi beban skala produksi (~220 ribu dokumen)
train = pd.concat([train] * 20, ignore_index=True)

t0 = time.perf_counter()
vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
Xtr = vec.fit_transform(train.text)
Xva = vec.transform(valid.text)
clf = LogisticRegression(max_iter=200)
clf.fit(Xtr, train.label)
acc = accuracy_score(valid.label, clf.predict(Xva))
print(f"akurasi={acc:.4f}  fit+predict={time.perf_counter()-t0:.1f}s  total={time.perf_counter()-t_all:.1f}s")''',
'''%%writefile pipeline.py
# The classic notebook-01 pipeline: TF-IDF + Logistic Regression.
# There is NOT a single line of GPU code in this file.
import time
t_all = time.perf_counter()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv("smsa_train.tsv", sep="\\t", names=["text", "label"])
valid = pd.read_csv("smsa_valid.tsv", sep="\\t", names=["text", "label"])
# Duplicate train 20x: simulates a production-scale workload (~220k documents)
train = pd.concat([train] * 20, ignore_index=True)

t0 = time.perf_counter()
vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
Xtr = vec.fit_transform(train.text)
Xva = vec.transform(valid.text)
clf = LogisticRegression(max_iter=200)
clf.fit(Xtr, train.label)
acc = accuracy_score(valid.label, clf.predict(Xva))
print(f"accuracy={acc:.4f}  fit+predict={time.perf_counter()-t0:.1f}s  total={time.perf_counter()-t_all:.1f}s")'''),

C("""# Baseline: CPU murni (pandas + sklearn biasa)
!python pipeline.py""",
"""# Baseline: pure CPU (plain pandas + sklearn)
!python pipeline.py"""),

C("""# Sulapnya: file yang SAMA, dijalankan lewat akselerator GPU
!python -m cudf.pandas -m cuml.accel pipeline.py""",
"""# The magic: the SAME file, run through the GPU accelerators
!python -m cudf.pandas -m cuml.accel pipeline.py"""),

M("""### Apa yang barusan terjadi?

- **Akurasi sama** → hasil komputasinya identik secara statistik.
- **Waktu beda jauh** → bagian pandas (load, concat, string) lari di cuDF, estimator sklearn yang didukung lari di cuML.
- Operasi yang belum didukung GPU otomatis **fallback ke CPU** — kode tidak pernah crash karena ini.

> 💡 **Kapan trik ini layak?** Saat data cukup besar. Untuk data kecil, waktu transfer CPU↔GPU lebih mahal daripada komputasinya sendiri — GPU justru bisa lebih lambat. Ini pola umum di semua komputasi GPU.""",
"""### What just happened?

- **Same accuracy** → the computation is statistically identical.
- **Very different time** → the pandas parts (load, concat, strings) ran on cuDF; supported sklearn estimators ran on cuML.
- Operations not yet supported on GPU automatically **fall back to CPU** — your code never crashes because of this.

> 💡 **When is this trick worth it?** When the data is big enough. For small data, CPU↔GPU transfer costs more than the computation itself — the GPU can actually be slower. This is a universal pattern in GPU computing."""),

# ============ BABAK 2 — NVTEXT ============
M("""## Babak 2 — Bedah nvtext: Preprocessing Skala Wikipedia 🔬

Sulap selesai; sekarang kita pakai API GPU-nya **secara sadar**. cuDF punya modul teks (*nvtext*) dengan operasi yang pandas **tidak punya sama sekali**: `tokenize`, `token_count`, `ngrams_tokenize`, `minhash`.

Korpusnya kita naikkan kelas: **Wikipedia Bahasa Indonesia** — kita ambil ±150 ribu paragraf via streaming (tanpa download dump penuh).""",
"""## Act 2 — Dissecting nvtext: Wikipedia-Scale Preprocessing 🔬

Magic over; now we use the GPU API **deliberately**. cuDF has a text module (*nvtext*) with operations pandas **simply does not have**: `tokenize`, `token_count`, `ngrams_tokenize`, `minhash`.

We also upgrade the corpus: **Indonesian Wikipedia** — we take ±150k paragraphs via streaming (no full-dump download)."""),

C("""!pip install -q datasets

from datasets import load_dataset

# Streaming: ambil 20 ribu artikel pertama, pecah jadi paragraf
ds = load_dataset("wikimedia/wikipedia", "20231101.id", split="train", streaming=True)
paragraphs = []
for i, art in enumerate(ds):
    if i >= 20_000:
        break
    paragraphs += [p for p in art["text"].split("\\n") if len(p) > 80]

print(f"{len(paragraphs):,} paragraf terkumpul")""",
"""!pip install -q datasets

from datasets import load_dataset

# Streaming: take the first 20k articles, split into paragraphs
ds = load_dataset("wikimedia/wikipedia", "20231101.id", split="train", streaming=True)
paragraphs = []
for i, art in enumerate(ds):
    if i >= 20_000:
        break
    paragraphs += [p for p in art["text"].split("\\n") if len(p) > 80]

print(f"{len(paragraphs):,} paragraphs collected")"""),

C("""import time
import pandas as pd
import cudf

pdf = pd.Series(paragraphs, name="text")     # CPU
gdf = cudf.Series(paragraphs, name="text")   # GPU

BENCH = {}  # hasil benchmark, dipakai untuk ringkasan & slide

def bench(group, label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    BENCH.setdefault(group, {})[label] = dt
    print(f"{group:12s} | {label:6s} | {dt:7.3f} s")
    return out""",
None),

C("""# --- Tokenisasi + hitung token ---
n_cpu = bench("tokenize", "CPU", lambda: pdf.str.split().map(len).sum())
n_gpu = bench("tokenize", "GPU", lambda: gdf.str.token_count().sum())
print(f"total token: CPU={n_cpu:,} GPU={n_gpu:,}")""",
"""# --- Tokenization + token count ---
n_cpu = bench("tokenize", "CPU", lambda: pdf.str.split().map(len).sum())
n_gpu = bench("tokenize", "GPU", lambda: gdf.str.token_count().sum())
print(f"total tokens: CPU={n_cpu:,} GPU={n_gpu:,}")"""),

C("""# --- Frekuensi kata (top-15) ---
top_cpu = bench("wordcount", "CPU", lambda: pdf.str.lower().str.split().explode().value_counts().head(15))
top_gpu = bench("wordcount", "GPU", lambda: gdf.str.lower().str.tokenize().value_counts().head(15))
print(top_gpu)""",
"""# --- Word frequency (top-15) ---
top_cpu = bench("wordcount", "CPU", lambda: pdf.str.lower().str.split().explode().value_counts().head(15))
top_gpu = bench("wordcount", "GPU", lambda: gdf.str.lower().str.tokenize().value_counts().head(15))
print(top_gpu)"""),

C("""# --- Bigram (n-gram dengan n=2) ---
def bigram_cpu():
    toks = pdf.str.lower().str.split()
    return toks.map(lambda t: ["_".join(p) for p in zip(t, t[1:])]).explode().value_counts().head(10)

def bigram_gpu():
    return gdf.str.lower().str.ngrams_tokenize(2, separator="_").value_counts().head(10)

bench("bigram", "CPU", bigram_cpu)
big = bench("bigram", "GPU", bigram_gpu)
print(big)""",
None),

M("""### Skor sementara

Jalankan sel di bawah untuk rekap. Pola yang harus kamu lihat: makin "per-karakter" operasinya (tokenize, n-gram), makin besar kemenangan GPU — karena ribuan core mengerjakan ribuan string sekaligus.""",
"""### Halftime score

Run the cell below for a recap. The pattern to notice: the more "per-character" the operation (tokenize, n-grams), the bigger the GPU win — thousands of cores chew thousands of strings at once."""),

C("""import json

print(f"{'operasi':12s} {'CPU (s)':>9s} {'GPU (s)':>9s} {'speedup':>9s}")
for op, r in BENCH.items():
    if "CPU" in r and "GPU" in r:
        print(f"{op:12s} {r['CPU']:9.3f} {r['GPU']:9.3f} {r['CPU']/r['GPU']:8.1f}x")

# Simpan untuk figure slide (instruktur: download file ini setelah smoke-test)
with open("benchmark_results.json", "w") as f:
    json.dump(BENCH, f, indent=2)""",
"""import json

print(f"{'operation':12s} {'CPU (s)':>9s} {'GPU (s)':>9s} {'speedup':>9s}")
for op, r in BENCH.items():
    if "CPU" in r and "GPU" in r:
        print(f"{op:12s} {r['CPU']:9.3f} {r['GPU']:9.3f} {r['CPU']/r['GPU']:8.1f}x")

# Saved for the slide figure (instructor: download this file after the smoke test)
with open("benchmark_results.json", "w") as f:
    json.dump(BENCH, f, indent=2)"""),

# ============ BABAK 3 — MINHASH DEDUP ============
M("""## Babak 3 — Berburu Kembar: Dedup dengan MinHash 👯

Korpus mentah penuh duplikat dan *near-duplicate* (artikel template, boilerplate, copy-paste). Untuk training LLM, duplikat = model menghafal + komputasi terbuang. Masalahnya: membandingkan semua pasangan dari 150 ribu paragraf = ±11 **miliar** perbandingan. Tidak mungkin.

**MinHash** memotong jalan: setiap dokumen diringkas jadi *sidik jari* kecil; dokumen yang mirip menghasilkan sidik jari yang sama dengan probabilitas tinggi. Kita cukup mengelompokkan sidik jari yang identik — dari miliaran perbandingan jadi satu `groupby`.""",
"""## Act 3 — Twin Hunting: Dedup with MinHash 👯

Raw corpora are full of duplicates and *near-duplicates* (template articles, boilerplate, copy-paste). For LLM training, duplicates = memorization + wasted compute. The problem: comparing every pair among 150k paragraphs = ±11 **billion** comparisons. Not happening.

**MinHash** is the shortcut: each document is reduced to a small *fingerprint*; similar documents produce identical fingerprints with high probability. We just group identical fingerprints — billions of comparisons collapse into one `groupby`."""),

C("""import cudf

# API minhash cuDF sedikit berbeda antar versi — helper ini mencoba keduanya.
def minhash_signature(series, n_hashes=16, width=5):
    try:  # cuDF >= 24.12
        import cupy as cp
        a = cudf.Series(cp.random.RandomState(42).randint(1, 2**31, n_hashes), dtype="uint32")
        b = cudf.Series(cp.random.RandomState(7).randint(0, 2**31, n_hashes), dtype="uint32")
        return series.str.minhash(0, a=a, b=b, width=width)
    except TypeError:  # API lama: minhash(seeds, width)
        seeds = cudf.Series(range(n_hashes), dtype="uint32")
        return series.str.minhash(seeds, width=width)

sig = minhash_signature(gdf.str.lower())
# Sidik jari = gabungan seluruh nilai minhash jadi satu string kunci
fingerprint = sig.list.astype("str").list.join("-")
df = cudf.DataFrame({"text": gdf, "fp": fingerprint})
dup_counts = df.groupby("fp").size()
dup_groups = dup_counts[dup_counts > 1]
print(f"{len(dup_groups):,} kelompok near-duplicate ditemukan")""",
"""import cudf

# The cuDF minhash API differs slightly across versions — this helper tries both.
def minhash_signature(series, n_hashes=16, width=5):
    try:  # cuDF >= 24.12
        import cupy as cp
        a = cudf.Series(cp.random.RandomState(42).randint(1, 2**31, n_hashes), dtype="uint32")
        b = cudf.Series(cp.random.RandomState(7).randint(0, 2**31, n_hashes), dtype="uint32")
        return series.str.minhash(0, a=a, b=b, width=width)
    except TypeError:  # old API: minhash(seeds, width)
        seeds = cudf.Series(range(n_hashes), dtype="uint32")
        return series.str.minhash(seeds, width=width)

sig = minhash_signature(gdf.str.lower())
# Fingerprint = all minhash values joined into one key string
fingerprint = sig.list.astype("str").list.join("-")
df = cudf.DataFrame({"text": gdf, "fp": fingerprint})
dup_counts = df.groupby("fp").size()
dup_groups = dup_counts[dup_counts > 1]
print(f"{len(dup_groups):,} near-duplicate groups found")"""),

C("""# Intip satu kelompok kembar terbesar
biggest_fp = dup_groups.sort_values(ascending=False).index[0]
twins = df[df.fp == biggest_fp].text.to_pandas()
for t in twins.head(3):
    print("—", t[:160], "...")""",
"""# Peek at the largest twin group
biggest_fp = dup_groups.sort_values(ascending=False).index[0]
twins = df[df.fp == biggest_fp].text.to_pandas()
for t in twins.head(3):
    print("—", t[:160], "...")"""),

M("""> 🧠 Yang barusan kamu lakukan — minhash + groupby di GPU — adalah versi mini dari **fuzzy deduplication** yang dijalankan **NVIDIA NeMo Curator** saat menyiapkan korpus training LLM. Bedanya cuma skala: kita 150 ribu paragraf di 1 GPU, Curator miliaran dokumen di ratusan GPU.""",
"""> 🧠 What you just did — minhash + groupby on a GPU — is a mini version of the **fuzzy deduplication** that **NVIDIA NeMo Curator** runs when preparing LLM training corpora. The only difference is scale: us, 150k paragraphs on 1 GPU; Curator, billions of documents on hundreds of GPUs."""),

# ============ BABAK 4 — CUML NATIVE ============
M("""## Babak 4 — cuML Native: TF-IDF + Sentimen, Tanpa Sulap 🛠️

Di Babak 1 GPU bekerja diam-diam di balik `cuml.accel`. Sekarang kita panggil cuML **langsung** — perhatikan API-nya sengaja dibuat kembar dengan scikit-learn, jadi yang kamu pelajari di notebook 01 dipakai lagi 1:1.""",
"""## Act 4 — Native cuML: TF-IDF + Sentiment, No Magic 🛠️

In Act 1 the GPU worked silently behind `cuml.accel`. Now we call cuML **directly** — note the API is deliberately a twin of scikit-learn, so everything from notebook 01 transfers 1:1."""),

C("""import time
import cudf
from cuml.feature_extraction.text import TfidfVectorizer as GpuTfidf
from cuml.linear_model import LogisticRegression as GpuLogReg

train_g = cudf.read_csv("smsa_train.tsv", sep="\\t", names=["text", "label"])
valid_g = cudf.read_csv("smsa_valid.tsv", sep="\\t", names=["text", "label"])
ytr = train_g.label.astype("category").cat.codes.astype("float32")
yva = valid_g.label.astype("category").cat.codes.astype("float32")

t0 = time.perf_counter()
gvec = GpuTfidf(max_features=20_000)
Xtr = gvec.fit_transform(train_g.text)
Xva = gvec.transform(valid_g.text)
gclf = GpuLogReg(max_iter=200)
gclf.fit(Xtr, ytr)
acc_gpu = (gclf.predict(Xva) == yva).mean()
t_gpu = time.perf_counter() - t0
print(f"cuML  : akurasi={float(acc_gpu):.4f}  waktu={t_gpu:.2f}s")""",
"""import time
import cudf
from cuml.feature_extraction.text import TfidfVectorizer as GpuTfidf
from cuml.linear_model import LogisticRegression as GpuLogReg

train_g = cudf.read_csv("smsa_train.tsv", sep="\\t", names=["text", "label"])
valid_g = cudf.read_csv("smsa_valid.tsv", sep="\\t", names=["text", "label"])
ytr = train_g.label.astype("category").cat.codes.astype("float32")
yva = valid_g.label.astype("category").cat.codes.astype("float32")

t0 = time.perf_counter()
gvec = GpuTfidf(max_features=20_000)
Xtr = gvec.fit_transform(train_g.text)
Xva = gvec.transform(valid_g.text)
gclf = GpuLogReg(max_iter=200)
gclf.fit(Xtr, ytr)
acc_gpu = (gclf.predict(Xva) == yva).mean()
t_gpu = time.perf_counter() - t0
print(f"cuML  : accuracy={float(acc_gpu):.4f}  time={t_gpu:.2f}s")"""),

C("""# Pembanding apel-ke-apel di CPU (data sama, tanpa duplikasi 20x)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_p = pd.read_csv("smsa_train.tsv", sep="\\t", names=["text", "label"])
valid_p = pd.read_csv("smsa_valid.tsv", sep="\\t", names=["text", "label"])

t0 = time.perf_counter()
vec = TfidfVectorizer(max_features=20_000)
Xtr = vec.fit_transform(train_p.text)
Xva = vec.transform(valid_p.text)
clf = LogisticRegression(max_iter=200)
clf.fit(Xtr, train_p.label)
acc_cpu = accuracy_score(valid_p.label, clf.predict(Xva))
t_cpu = time.perf_counter() - t0
print(f"sklearn: akurasi={acc_cpu:.4f}  waktu={t_cpu:.2f}s")
print(f"speedup: {t_cpu/t_gpu:.1f}x")""",
"""# Apples-to-apples CPU comparison (same data, without the 20x duplication)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_p = pd.read_csv("smsa_train.tsv", sep="\\t", names=["text", "label"])
valid_p = pd.read_csv("smsa_valid.tsv", sep="\\t", names=["text", "label"])

t0 = time.perf_counter()
vec = TfidfVectorizer(max_features=20_000)
Xtr = vec.fit_transform(train_p.text)
Xva = vec.transform(valid_p.text)
clf = LogisticRegression(max_iter=200)
clf.fit(Xtr, train_p.label)
acc_cpu = accuracy_score(valid_p.label, clf.predict(Xva))
t_cpu = time.perf_counter() - t0
print(f"sklearn: accuracy={acc_cpu:.4f}  time={t_cpu:.2f}s")
print(f"speedup: {t_cpu/t_gpu:.1f}x")"""),

M("""### Native vs accel — kapan pakai yang mana?

| | `cuml.accel` (Babak 1) | API native (Babak 4) |
|---|---|---|
| Perubahan kode | nol | import diganti |
| Kontrol | terbatas (otomatis) | penuh (dtype, memori GPU) |
| Cocok untuk | migrasi cepat kode lama | pipeline GPU baru dari awal |

Pada dataset sekecil SmSA (±11 ribu), speedup-nya tipis — sekali lagi: **GPU menang di skala**, dan itulah mengapa Babak 2 memakai 150 ribu paragraf.""",
"""### Native vs accel — when to use which?

| | `cuml.accel` (Act 1) | Native API (Act 4) |
|---|---|---|
| Code changes | zero | swap the imports |
| Control | limited (automatic) | full (dtypes, GPU memory) |
| Best for | quickly migrating old code | new GPU pipelines from scratch |

On a dataset as small as SmSA (±11k), the speedup is thin — once more: **GPUs win at scale**, which is exactly why Act 2 used 150k paragraphs."""),

# ============ BABAK 5 — PENUTUP ============
M("""## Babak 5 — Naik Kelas: NeMo Curator & Jembatan ke LLM 🌉

Semua yang kamu kerjakan hari ini adalah miniatur dari pipeline data LLM sungguhan:

| Hari ini (1× T4, 150 ribu paragraf) | Produksi (NeMo Curator, ratusan GPU, miliaran dokumen) |
|---|---|
| `tokenize`, `normalize` di nvtext | language ID, cleaning, filtering kualitas |
| MinHash + groupby (Babak 3) | fuzzy dedup MinHash-LSH terdistribusi |
| TF-IDF + LogReg (Babak 4) | classifier kualitas dokumen |

**Peta library NVIDIA untuk NLP** — yang sudah dan akan kamu temui:
- **RAPIDS (cuDF/nvtext/cuML)** — hari ini ✅
- **NeMo + NeMo Curator** — framework training LLM & kurasi data → **Day 06**
- **TensorRT-LLM, Triton** — inference cepat → **Day 06**
- **HF Transformers di GPU** — **Day 04**, besok kita mulai masuk dunia LLM

### Latihan mandiri
1. Ganti korpus Babak 2 dengan 40 ribu artikel (`if i >= 40_000`). Apakah speedup GPU membesar atau mengecil? Mengapa?
2. Di Babak 3, perkecil `width` minhash menjadi 3. Apa efeknya terhadap jumlah kelompok duplikat, dan mengapa?
3. Di Babak 4, tambahkan `ngram_range=(1, 2)` pada kedua vectorizer. Bandingkan akurasi dan waktunya.""",
"""## Act 5 — Leveling Up: NeMo Curator & the Bridge to LLMs 🌉

Everything you did today is a miniature of a real LLM data pipeline:

| Today (1× T4, 150k paragraphs) | Production (NeMo Curator, hundreds of GPUs, billions of docs) |
|---|---|
| `tokenize`, `normalize` in nvtext | language ID, cleaning, quality filtering |
| MinHash + groupby (Act 3) | distributed MinHash-LSH fuzzy dedup |
| TF-IDF + LogReg (Act 4) | document quality classifiers |

**NVIDIA's NLP library map** — what you've met and will meet:
- **RAPIDS (cuDF/nvtext/cuML)** — today ✅
- **NeMo + NeMo Curator** — LLM training & data curation framework → **Day 06**
- **TensorRT-LLM, Triton** — fast inference → **Day 06**
- **HF Transformers on GPU** — **Day 04**, tomorrow we enter the LLM world

### Exercises
1. Change the Act 2 corpus to 40k articles (`if i >= 40_000`). Does the GPU speedup grow or shrink? Why?
2. In Act 3, reduce the minhash `width` to 3. What happens to the number of duplicate groups, and why?
3. In Act 4, add `ngram_range=(1, 2)` to both vectorizers. Compare accuracy and time."""),
]

def build(lang):
    cells = []
    for kind, id_, en in CELLS:
        src = id_ if lang == "id" else en
        lines = src.splitlines(keepends=True)
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": lines})
    return {"cells": cells,
            "metadata": {"accelerator": "GPU",
                         "colab": {"gpuType": "T4", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}

for lang in ("id", "en"):
    path = OUT / f"02_nlp_on_steroids_{lang}.ipynb"
    path.write_text(json.dumps(build(lang), ensure_ascii=False, indent=1))
    print("wrote", path)
```

- [ ] **Step 2: Jalankan generator**

```bash
python3 /tmp/gen_steroids_nb.py
```

Expected: `wrote .../02_nlp_on_steroids_id.ipynb` dan `..._en.ipynb`.

- [ ] **Step 3: Validasi struktur kedua notebook**

```bash
python3 - <<'EOF'
import json
for lang in ("id", "en"):
    p = f"03_nlp_fundamentals/02_nlp_on_steroids_{lang}.ipynb"
    nb = json.load(open(p))
    assert nb["nbformat"] == 4
    n_md = sum(c["cell_type"] == "markdown" for c in nb["cells"])
    n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
    assert all("".join(c["source"]).strip() for c in nb["cells"]), "sel kosong!"
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
    print(f"{p}: OK ({n_md} md + {n_code} code)")
EOF
```

Expected: kedua file `OK` dengan jumlah sel identik (ID vs EN).

- [ ] **Step 4: Review isi ID vs EN sejajar** — buka kedua file, spot-check 3 sel acak: EN harus terjemahan setia (bukan parafrase), kode identik kecuali komentar/label print.

- [ ] **Step 5: Commit**

```bash
git add 03_nlp_fundamentals/02_nlp_on_steroids_id.ipynb 03_nlp_fundamentals/02_nlp_on_steroids_en.ipynb
git commit -m "feat(module03): notebook NLP on Steroids — NLP klasik di GPU dengan RAPIDS (ID+EN)"
```

### Task 3: Slides — 3 frame baru + figure speedup

**Files:**
- Create: `03_nlp_fundamentals/slides/figures/gen_rapids_speedup.py`
- Modify: `03_nlp_fundamentals/slides/module03_slides.tex` (sisipkan sebelum `% Slide 17 — Next Steps`)

- [ ] **Step 1: Tulis figure script** (angka awal = estimasi, ditandai jelas; diganti angka asli di Task 7)

```python
"""
Generate rapids_speedup.pdf — speedup GPU vs CPU operasi NLP (Module 03 slides).
ANGKA dari benchmark_results.json hasil smoke-test Colab T4 notebook
02_nlp_on_steroids. Sebelum smoke-test: nilai ESTIMASI (jangan dipakai final).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "rapids_speedup.pdf")

# ── Data (kunci: operasi → speedup CPU/GPU) ────────────────────────────────
# >>> ESTIMASI SEMENTARA — WAJIB diganti angka benchmark_results.json (Task 7) <<<
SPEEDUP = {
    "tokenize":   8.0,
    "word count": 6.0,
    "bigram":     10.0,
    "TF-IDF + LogReg": 4.0,
}

BG_COLOR  = "#1A1A2E"
BAR_COLOR = "#76B900"  # navasena green

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

ops = list(SPEEDUP)
vals = [SPEEDUP[o] for o in ops]
bars = ax.barh(ops, vals, color=BAR_COLOR, height=0.55)
ax.bar_label(bars, fmt="%.1fx", color="white", fontsize=12, padding=4)
ax.axvline(1, color="#E0E0E0", lw=1, ls="--")
ax.text(1.05, -0.45, "CPU = 1x", color="#E0E0E0", fontsize=9)

ax.set_xlabel("Speedup vs CPU (lebih besar = lebih cepat)", color="white")
ax.tick_params(colors="white", labelsize=11)
for spine in ax.spines.values():
    spine.set_color("#444466")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUTPUT_PATH, facecolor=BG_COLOR)
print("wrote", OUTPUT_PATH)
```

- [ ] **Step 2: Jalankan & cek output**

```bash
python3 03_nlp_fundamentals/slides/figures/gen_rapids_speedup.py
```

Expected: `wrote .../rapids_speedup.pdf`, file muncul.

- [ ] **Step 3: Sisipkan 3 frame** ke `module03_slides.tex`, tepat SEBELUM baris `% Slide 17 — Next Steps` (gunakan Edit; sesuaikan nomor komentar slide setelahnya TIDAK perlu diubah — cukup beri label 16b/16c/16d agar tidak mengganggu penomoran lama):

```latex
% Slide 16b — RAPIDS: NLP Klasik di GPU
\begin{frame}{NLP on Steroids: NVIDIA RAPIDS}
  \begin{columns}[T]
    \begin{column}{0.52\textwidth}
      \textcolor{nvgreen}{\textbf{Stack RAPIDS untuk NLP}}
      \begin{itemize}
        \footnotesize
        \item \textbf{cuDF} — pandas versi GPU
        \item \textbf{nvtext} — tokenize, n-gram, minhash di GPU
        \item \textbf{cuML} — scikit-learn versi GPU
        \item Pre-installed di Colab (runtime GPU)
      \end{itemize}
    \end{column}
    \begin{column}{0.44\textwidth}
      \textcolor{nvorange}{\textbf{Kapan GPU menang?}}
      \begin{itemize}
        \footnotesize
        \item Data besar (puluhan ribu dokumen ke atas)
        \item Operasi string yang seragam \& masif
        \item Overhead transfer CPU$\to$GPU tertutupi
      \end{itemize}
    \end{column}
  \end{columns}
  \vspace{0.6em}
  \centering\footnotesize\textcolor{nvgray}{Hands-on: notebook \texttt{02\_nlp\_on\_steroids}}
\end{frame}

% Slide 16c — Zero Code Change
\begin{frame}[fragile]{Zero Code Change: cudf.pandas \& cuml.accel}
  \begin{lstlisting}
# CPU: pandas + sklearn biasa
python pipeline.py

# GPU: file SAMA, nol perubahan kode
python -m cudf.pandas -m cuml.accel pipeline.py
  \end{lstlisting}
  \vspace{0.4em}
  \begin{itemize}
    \item \texttt{cudf.pandas}: operasi pandas dialihkan ke GPU, sisanya \textit{fallback} otomatis ke CPU
    \item \texttt{cuml.accel}: estimator scikit-learn $\to$ cuML di GPU
    \item Akurasi identik — yang berubah hanya \textbf{waktu}
  \end{itemize}
\end{frame}

% Slide 16d — Seberapa Cepat?
\begin{frame}{Seberapa Cepat? (Colab T4, 150 ribu paragraf Wikipedia ID)}
  \begin{center}
    \includegraphics[width=0.88\textwidth]{figures/rapids_speedup.pdf}
  \end{center}
  \footnotesize\textcolor{nvgray}{Diukur langsung di notebook \texttt{02\_nlp\_on\_steroids} — angka di mesinmu bisa sedikit berbeda.}
\end{frame}

```

- [ ] **Step 4: Build slides**

```bash
cd 03_nlp_fundamentals/slides && bash build.sh
```

Expected: berakhir `=== Done! Output: module03_slides.pdf ===` tanpa error. Jika error `Undefined control sequence` cek typo; jika error lstlisting cek frame `[fragile]`.

- [ ] **Step 5: Inspeksi visual 3 slide baru** — buka PDF, cek slide 17–19 (posisi baru): teks tidak overflow, figure terbaca.

- [ ] **Step 6: Commit**

```bash
git add 03_nlp_fundamentals/slides/module03_slides.tex 03_nlp_fundamentals/slides/module03_slides.pdf 03_nlp_fundamentals/slides/figures/gen_rapids_speedup.py 03_nlp_fundamentals/slides/figures/rapids_speedup.pdf
git commit -m "feat(module03): 3 slide RAPIDS (stack, zero-code-change, speedup) + figure"
```

### Task 4: Quiz +3 soal (15 → 18)

**Files:**
- Modify: `03_nlp_fundamentals/nlp-fundamentals-quiz.html` (JSON satu-baris `const QUIZ = {...};` + header baris 52–53)

- [ ] **Step 1: Sisipkan soal via script** (JSON-aware, aman untuk file satu-baris)

```bash
python3 - <<'EOF'
import json

PATH = "03_nlp_fundamentals/nlp-fundamentals-quiz.html"
NEW = [
 {"q": "Apa yang dilakukan perintah `python -m cudf.pandas -m cuml.accel pipeline.py` terhadap script pandas/sklearn biasa?",
  "code": None,
  "options": [
    "Mengompilasi script menjadi binary CUDA agar tidak butuh Python lagi",
    "Menjalankan script tanpa modifikasi: operasi pandas/sklearn yang didukung dialihkan ke GPU (cuDF/cuML), sisanya fallback otomatis ke CPU",
    "Mengubah semua DataFrame menjadi numpy array agar lebih hemat memori",
    "Memaksa seluruh operasi jalan di GPU dan error jika ada yang tidak didukung"],
  "answer": 1,
  "explanation": "Notebook 02 Babak 1: cudf.pandas dan cuml.accel adalah mode akselerasi zero-code-change — kode tidak diubah, operasi yang didukung lari di GPU, dan yang belum didukung otomatis fallback ke CPU sehingga tidak pernah crash karena ini."},
 {"q": "Pada benchmark notebook 02, kapan pemakaian GPU (cuDF/cuML) justru BISA lebih lambat daripada CPU?",
  "code": None,
  "options": [
    "Saat data berukuran kecil, karena overhead transfer CPU↔GPU lebih mahal daripada komputasinya sendiri",
    "Saat teks mengandung bahasa Indonesia, karena GPU hanya mendukung bahasa Inggris",
    "Saat memakai TF-IDF, karena TF-IDF tidak bisa diparalelkan",
    "GPU selalu lebih cepat dalam segala kondisi"],
  "answer": 0,
  "explanation": "Pola yang ditekankan di Babak 1 dan 4: GPU menang di skala. Untuk data kecil (mis. SmSA ±11 ribu dokumen) waktu transfer data ke GPU bisa melebihi penghematan komputasinya."},
 {"q": "Mengapa MinHash dipakai saat menyiapkan korpus training LLM (seperti di notebook 02 Babak 3 dan NeMo Curator)?",
  "code": None,
  "options": [
    "Untuk mengenkripsi dokumen agar data training tidak bocor",
    "Untuk memampatkan dokumen sehingga muat di memori GPU",
    "Untuk menemukan dokumen near-duplicate secara efisien lewat sidik jari hash, tanpa membandingkan semua pasangan dokumen",
    "Untuk menghitung frekuensi kata lebih cepat daripada value_counts"],
  "answer": 2,
  "explanation": "Babak 3: membandingkan semua pasangan dari 150 ribu paragraf = miliaran perbandingan. MinHash meringkas tiap dokumen jadi sidik jari; dokumen mirip menghasilkan sidik jari sama dengan probabilitas tinggi, sehingga dedup cukup dengan satu groupby."},
]

html = open(PATH).read()
key = "const QUIZ = "
i = html.find(key) + len(key)
obj, end = json.JSONDecoder().raw_decode(html[i:])
assert len(obj["questions"]) == 15, f"ekspektasi 15 soal, dapat {len(obj['questions'])}"
obj["questions"] += NEW
html = html[:i] + json.dumps(obj, ensure_ascii=False) + html[i + end:]
html = html.replace("15 soal pilihan ganda", "18 soal pilihan ganda")
html = html.replace("</b> / 15", "</b> / 18")
open(PATH, "w").write(html)
print("OK: 18 soal")
EOF
```

Expected: `OK: 18 soal`.

- [ ] **Step 2: Validasi hasil**

```bash
python3 - <<'EOF'
import json
html = open("03_nlp_fundamentals/nlp-fundamentals-quiz.html").read()
i = html.find("const QUIZ = ") + len("const QUIZ = ")
quiz, _ = json.JSONDecoder().raw_decode(html[i:])
assert len(quiz["questions"]) == 18
for n, q in enumerate(quiz["questions"], 1):
    assert len(q["options"]) == 4 and q["answer"] in range(4) and q["explanation"], f"soal {n} cacat"
assert "18 soal pilihan ganda" in html and "</b> / 18" in html
print("VALID: 18 soal, header sinkron")
EOF
```

Expected: `VALID: 18 soal, header sinkron`.

- [ ] **Step 3: Cek manual di browser** — buka file, scroll ke Soal 16–18, jawab beberapa, pastikan grading & skor total /18 benar.

- [ ] **Step 4: Commit**

```bash
git add 03_nlp_fundamentals/nlp-fundamentals-quiz.html
git commit -m "feat(module03): 3 soal quiz RAPIDS (zero-code-change, skala GPU, minhash) — total 18"
```

### Task 5: Cheatsheet +2 kartu + regenerate PDF

**Files:**
- Modify: `03_nlp_fundamentals/nlp-fundamentals-cheatsheet.html` (HTML satu-baris, kartu di dalam `<div class="grid">`)
- Regenerate: `03_nlp_fundamentals/nlp-fundamentals-cheatsheet.pdf`

- [ ] **Step 1: Temukan anchor kartu terakhir**

```bash
grep -o 'when">▸ [^<]*' 03_nlp_fundamentals/nlp-fundamentals-cheatsheet.html | tail -1
```

Catat teks `<div class="when">▸ ...` kartu terakhir; kartu baru disisipkan SETELAH `</div></div>` penutup kartu itu (tepat sebelum `</div>` penutup grid).

- [ ] **Step 2: Sisipkan 2 kartu** (via Edit, old_string = penutup kartu terakhir dari Step 1 + `</div>` grid; new_string menambahkan dua kartu ini sebelum penutup grid):

```html
<div class="card"><h3>RAPIDS untuk NLP</h3><ul><li>cuDF = pandas di GPU; cuML = scikit-learn di GPU</li><li>nvtext (dalam cuDF): tokenize, token_count, ngrams_tokenize, minhash</li><li>Pre-installed di Colab runtime GPU — tinggal import cudf, cuml</li><li>MinHash + groupby = dedup near-duplicate ala NeMo Curator</li></ul><div class="codeblock">s = cudf.Series(teks); s.str.tokenize(); s.str.ngrams_tokenize(2, separator=&quot;_&quot;)</div><div class="when">▸ preprocessing teks skala besar (100 ribu+ dokumen)</div></div><div class="card"><h3>Zero Code Change (GPU)</h3><ul><li>cudf.pandas: kode pandas lama lari di GPU tanpa diubah</li><li>cuml.accel: estimator sklearn dialihkan ke cuML</li><li>Operasi tak didukung → fallback otomatis ke CPU (tidak crash)</li><li>Data kecil bisa lebih lambat — GPU menang di skala</li></ul><div class="codeblock">python -m cudf.pandas -m cuml.accel pipeline.py</div><div class="when">▸ migrasi cepat pipeline pandas/sklearn ke GPU</div></div>
```

- [ ] **Step 3: Validasi jumlah kartu & render**

```bash
grep -o '<h3>[^<]*</h3>' 03_nlp_fundamentals/nlp-fundamentals-cheatsheet.html | tail -4
```

Expected: dua judul terakhir = `RAPIDS untuk NLP` dan `Zero Code Change (GPU)`. Lalu buka di browser, cek layout grid tidak rusak.

- [ ] **Step 4: Regenerate PDF**

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --print-to-pdf="$PWD/03_nlp_fundamentals/nlp-fundamentals-cheatsheet.pdf" --no-pdf-header-footer "file://$PWD/03_nlp_fundamentals/nlp-fundamentals-cheatsheet.html"
```

Expected: PDF ter-update (cek `ls -la`, timestamp baru; buka dan pastikan 2 kartu baru ada). Jika Chrome tidak ada di path itu, pakai Edge: `/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge`.

- [ ] **Step 5: Commit**

```bash
git add 03_nlp_fundamentals/nlp-fundamentals-cheatsheet.html 03_nlp_fundamentals/nlp-fundamentals-cheatsheet.pdf
git commit -m "feat(module03): 2 kartu cheatsheet RAPIDS (stack NLP GPU, zero-code-change)"
```

### Task 6: Smoke-test Colab T4 — GERBANG MANUAL ⛔

**Files:** tidak ada (eksekusi oleh instruktur di Colab)

- [ ] **Step 1: Push & siapkan link Colab**

```bash
git push origin master
```

Link untuk instruktur (format sama dengan notebook lain course ini):
`https://colab.research.google.com/github/chmdznr/navasena-gen-ml-course/blob/master/03_nlp_fundamentals/02_nlp_on_steroids_id.ipynb`

- [ ] **Step 2: Instruktur menjalankan SEMUA sel** notebook ID di Colab T4, atas-ke-bawah, dan mencatat:
  - `import cudf, cuml` sukses tanpa fallback pip? (verifikasi asumsi pre-installed)
  - Sintaks `python -m cudf.pandas -m cuml.accel` jalan? (jika `-m` ganda tidak didukung, ubah sel jadi dua baris env-var sesuai pesan error, regenerasi notebook, ulangi)
  - Helper minhash masuk cabang API yang mana?
  - Waktu total eksekusi notebook (target < 15 menit)
  - **Download `benchmark_results.json`** dari Colab → simpan ke `/tmp/benchmark_results.json`

- [ ] **Step 3: Jika ada sel gagal** — perbaiki di generator `/tmp/gen_steroids_nb.py`, jalankan ulang generator (kedua bahasa), commit fix dengan pesan `fix(module03): <masalah> di notebook NLP on Steroids`, ulangi smoke-test. Ulangi sampai bersih.

- [ ] **Step 4: Smoke-test cepat notebook EN** — jalankan minimal Babak 0–1 untuk memastikan tidak ada beda struktural.

### Task 7: Isi angka benchmark asli ke figure + rebuild slides

**Files:**
- Modify: `03_nlp_fundamentals/slides/figures/gen_rapids_speedup.py` (dict `SPEEDUP`)

- [ ] **Step 1: Hitung speedup dari hasil smoke-test**

```bash
python3 - <<'EOF'
import json
b = json.load(open("/tmp/benchmark_results.json"))
for op, r in b.items():
    if "CPU" in r and "GPU" in r:
        print(f'    "{op}": {r["CPU"]/r["GPU"]:.1f},')
EOF
```

- [ ] **Step 2: Update dict `SPEEDUP`** di `gen_rapids_speedup.py` dengan output Step 1 + entri `"TF-IDF + LogReg"` dari catatan Babak 4 smoke-test; hapus komentar `>>> ESTIMASI SEMENTARA ... <<<`.

- [ ] **Step 3: Rebuild & commit**

```bash
python3 03_nlp_fundamentals/slides/figures/gen_rapids_speedup.py
cd 03_nlp_fundamentals/slides && bash build.sh && cd ../..
git add 03_nlp_fundamentals/slides/figures/gen_rapids_speedup.py 03_nlp_fundamentals/slides/figures/rapids_speedup.pdf 03_nlp_fundamentals/slides/module03_slides.pdf
git commit -m "fix(module03): figure speedup RAPIDS pakai angka asli smoke-test Colab T4"
git push origin master
```

### Task 8: Upload ke Google Drive course

**Files:** tidak ada perubahan repo

- [ ] **Step 1: Upload via workflow Drive yang sudah ada** (folder bootcamp Batch 4 — ID folder di memory `bootcamp-batch-4-materials-and-participants`):
  - `02_nlp_on_steroids_id.ipynb`, `02_nlp_on_steroids_en.ipynb`
  - `module03_slides.pdf` (replace, pertahankan file ID agar link peserta tidak berubah)
  - `nlp-fundamentals-quiz.html`, `nlp-fundamentals-cheatsheet.html`, `nlp-fundamentals-cheatsheet.pdf` (replace)

- [ ] **Step 2: Verifikasi** — buka link Drive sebagai peserta (incognito), pastikan file terbaru terlihat dan link Colab notebook baru bisa dibuka.

---

## Definition of Done

- [ ] Kedua notebook jalan penuh di Colab T4 tanpa error (smoke-test Task 6 lulus)
- [ ] Slide ter-compile dengan figure berisi angka benchmark asli
- [ ] Quiz 18 soal valid + grading benar di browser
- [ ] Cheatsheet 2 kartu baru, PDF ter-regenerate
- [ ] Semua ter-push ke GitHub & ter-upload ke Drive
