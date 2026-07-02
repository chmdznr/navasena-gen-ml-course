# Ide Final Capstone Project untuk Bootcamp NCA-GENL (Navasena)

Dokumen ini berisi **11 ide capstone project** (10 utama plus 1 bonus) untuk peserta bootcamp NCA-GENL.
Semua ide sudah disaring dengan tiga syarat keras:

1. **Selaras dengan materi bootcamp.** Setiap project memakai skill dari Modul 01 sampai 06, dan sebisa
   mungkin me-reuse notebook yang sudah dipelajari, bukan skill baru dari nol.
2. **Dataset benar-benar tersedia.** Setiap dataset sudah dicek lewat pencarian web: URL publiknya aktif,
   lisensinya jelas, dan ukurannya masuk akal. Tidak ada dataset fiktif.
3. **Cukup dijalankan di Google Colab Tesla T4 (16 GB).** Tidak ada training yang butuh GPU di atas T4.
   QLoRA untuk model di bawah 3B (4-bit, fp16) muat di T4; generasi LLM besar di-offload ke **NVIDIA NIM**
   (free tier); embedding, reranker, RAG, CNN transfer learning, dan classical ML semuanya muat.

> **Cara dokumen ini dibuat:** 28 kandidat ide di-generate lintas 7 tema. Setiap kandidat lalu di-fact-check
> ketersediaan datasetnya lewat pencarian web, kemudian dinilai apakah benar layak di T4 dan cocok dengan
> kurikulum. 11 ide terkuat lolos. Catatan dataset yang sengaja dibuang ada di bagian akhir.

---

## Ringkasan (untuk memilih cepat)

| # | Judul | Tingkat | Modul inti | Dataset | Bahasa ID |
|---|-------|---------|-----------|---------|:---------:|
| 1 | Audit Adil: deteksi bias sentimen antar kategori produk | Menengah | M06 + M03/M04/M01 | PRDECT-ID | Ya |
| 2 | Spesialis NL-to-SQL (dengan verifikasi eksekusi) | Menengah | M04 + M06 | sql-create-context | Tidak* |
| 3 | Tanya-Dokter Grounded (RAG kesehatan + disclaimer) | Menengah | M05 + M06/M03 | Alodokter QA | Ya |
| 4 | IDK-RAG: asisten yang tahu kapan bilang "tidak tahu" | Menengah | M05 + M06/M03/M04 | IDK-MRC | Ya |
| 5 | Wartawan Mini: spesialis ringkasan & judul berita | Mahir | M04 + M03/M06 | IndoSum | Ya |
| 6 | JDIH-Asisten: Ask-My-Regulation hukum ketenagakerjaan | Menengah | M05 + M03/M06/M01 | UU 13/2003 (JDIH BPK) | Ya |
| 7 | Deteksi fraud pada data tidak seimbang | Menengah | M01 | Credit Card Fraud (ULB) | Tidak* |
| 8 | Pengenal motif batik + Grad-CAM | Menengah | M02 + M06 | Indonesian Batik Motifs | Ya |
| 9 | Estimator harga mobil bekas Indonesia | **Pemula** | M01 | Used Car Listings ID | Ya |
| 10 | Asisten triase kesehatan (spesialis + edge GGUF) | Mahir | M04 + M03/M06 | Alodokter QA | Ya |
| 11 *(bonus)* | Radar emosi ulasan: klasik vs IndoBERT + dashboard | Menengah | M03 + M01/M02 | PRDECT-ID | Ya |

\* Project #2 (NL-to-SQL) dan #7 (fraud) memakai dataset non-Indonesia, dan ini disengaja: keyword SQL memang
berbahasa Inggris, sedangkan untuk fraud tidak ada dataset Indonesia dengan kualitas setara. Keduanya tetap
sangat relevan dari sisi skill.

**Sebaran tingkat:** 1 Pemula, 7 Menengah, 3 Mahir, jadi ada jalur masuk untuk peserta di semua level.
**9 dari 11 memakai dataset Indonesia.** Semua dataset berlisensi publik (CC0, CC-BY, MIT, Apache, DbCL, atau
domain publik).

---

## 1. Audit Adil: deteksi bias sentimen LLM antar kategori produk *(Trustworthy AI)*

- **Modul:** M06 (inti) plus M03, M04, M01. **Tingkat:** Menengah.
- **Ide singkat:** Meng-audit sebuah classifier sentimen/emosi untuk soal keadilan (fairness) antar kategori
  produk dan kelompok rating bintang, memakai review Tokopedia asli. Peserta menghitung *demographic parity*,
  *equalized-odds*, dan *disparate impact*, lalu memeriksa kecenderungan stereotip dengan LLM-judge NIM, dan
  menerbitkan **Model Card plus checklist etika**. Ini kandidat yang paling pas dengan jalur Trustworthy AI di M06.
- **Dataset:** PRDECT-ID (Indonesian Product Reviews Dataset for Emotions). Lisensi MIT (mirror GitHub) atau
  CC-BY-4.0 (Mendeley), ukuran sekitar 1.2 MB CSV.
  URL: <https://github.com/rhiosutoyo/PRDECT-ID-Indonesian-Product-Reviews-Dataset>
- **Yang dibangun:** Satu notebook Colab. Latih classifier baseline (default TF-IDF + LogReg, opsional fine-tune
  IndoBERT), bentuk grup audit dari kolom asli (29 kategori dan bin rating), hitung tabel metrik fairness per
  grup lengkap dengan uji signifikansi, lewatkan sekitar 300 review ke bias-judge NIM (Nemotron), lalu hasilkan
  `Model_Card.md`, `AI_ETHICS_CHECKLIST.md`, dan sorotan kategori paling bias beserta rekomendasi mitigasinya.
- **Kenapa cukup di T4:** Semua bagian ringan. Classifier-nya TF-IDF + LogReg (jalan di CPU) atau IndoBERT-base
  sekitar 110M parameter yang muat enteng di T4. Metrik fairness murni numpy/sklearn, dan bias-judge di-offload
  ke NIM. Datasetnya cuma satu CSV.
- **Titik mulai:** Pakai ulang fungsi `demographic_parity()` dan `equalized_odds()` dari
  `06_nvidia_ecosystem/03_fairness_and_governance.ipynb`. Bentuk grup audit dari kolom **Category** dan bin
  **Customer Rating**; jangan mengarang kolom gender karena memang tidak ada di data. Pasang ambang n minimum
  untuk kategori yang datanya sedikit, lalu jalankan `bias_judge()` Nemotron pada sampel sekitar 300 review.
- **Stretch goal:** Tambahkan classifier IndoBERT-base hasil fine-tune, lalu bandingkan: apakah model yang lebih
  kuat justru memperbesar atau malah memperkecil selisih error antar kategori?

## 2. Spesialis NL-to-SQL: ubah pertanyaan bahasa alami jadi query SQL yang bisa dijalankan

- **Modul:** M04 (inti) plus M06. **Tingkat:** Menengah.
- **Ide singkat:** Fine-tune **Qwen3-1.7B (QLoRA)** menjadi spesialis text-to-SQL, dengan andalan berupa
  **verifikasi eksekusi**: query hasil model benar-benar dijalankan di SQLite in-memory untuk mengukur
  kesetaraan hasil (*execution-equivalence*), bukan sekadar kecocokan teks, lalu diuji pada skema bisnis
  Indonesia buatan sendiri. Project ini mengajarkan satu pelajaran berharga: teks yang mirip belum tentu benar.
- **Dataset:** `b-mc2/sql-create-context`. Lisensi CC-BY-4.0, ukuran 21.8 MB.
  URL: <https://huggingface.co/datasets/b-mc2/sql-create-context>
- **Yang dibangun:** Satu notebook. Format instruksi `schema + pertanyaan -> SQL`, QLoRA fine-tune Qwen3-1.7B,
  lalu bangun skrip verifikasi yang menjalankan SQL prediksi dan SQL gold di SQLite in-memory dan membandingkan
  hasilnya tanpa peduli urutan baris (base vs spesialis). Uji pada skema penjualan Indonesia buatan sendiri, lalu merge
  dan GGUF menjadi demo "SQL copilot" lokal.
- **Kenapa cukup di T4:** QLoRA 4-bit Qwen3-1.7B (di bawah 3B) dengan compute fp16 muat lega di T4, puncak
  pemakaian VRAM jauh di bawah 8 GB. Dataset 21.8 MB, verifikasi eksekusi SQLite jalan di CPU, dan batas-atas
  NIM Llama-3.3-70B yang opsional di-offload ke cloud.
- **Titik mulai:** Fork `04_llm/05_slm_specialist.ipynb`. Pisahkan data (split) **sebelum** disubsample ke
  15-20k, format chat template dengan `enable_thinking=False`, lalu QLoRA Qwen3-1.7B (nf4, fp16,
  paged_adamw_8bit, r=16). Bangun skrip verifikasi eksekusi: jalankan `CREATE TABLE` ke sqlite3, lalu jalankan
  SQL gold dan prediksi pakai try/except plus timeout, dan bandingkan kedua hasilnya. Tulis skema Indonesia kecil yang
  **sudah berisi baris INSERT** supaya demonya mengembalikan angka nyata. Merge ke fp16, lalu GGUF, lalu Ollama.
- **Stretch goal:** Pakai database SQLite berisi dari dataset **Spider** untuk mengukur akurasi eksekusi pada
  database terisi, bukan sekadar kesetaraan pada skema kosong.

## 3. Tanya-Dokter Grounded: RAG info kesehatan dengan sitasi & disclaimer medis

> **PENTING:** Ini artefak edukasi, **bukan** alat diagnosis. Setiap jawaban wajib ditutup dengan kalimat
> "informasi ini bukan pengganti konsultasi dokter".

- **Modul:** M05 (inti) plus M06, M03. **Tingkat:** Menengah.
- **Ide singkat:** Ubah 250 ribu lebih pasang tanya-jawab dokter Indonesia menjadi asisten info kesehatan yang
  **me-retrieve jawaban dokter asli dan menyitirnya**, bukan berhalusinasi. Fokus utama project ini adalah
  pengukuran Trustworthy AI: seberapa besar grounding, reranking, abstention, dan guardrail menekan klaim medis
  yang tak berdasar, dibandingkan baseline tanpa RAG.
- **Dataset:** `agufsamudra/alodokter-qna`. Lisensi Apache-2.0, CSV 2.83 GB (di-stream dan disampel, jangan
  di-load penuh). URL: <https://huggingface.co/datasets/agufsamudra/alodokter-qna>
- **Yang dibangun:** Satu notebook. Stream, bersihkan, dan sampel sekitar 30k tanya-jawab; embed dengan
  multilingual-MiniLM ke FAISS-CPU; retrieve dua tahap plus bge-reranker-v2-m3; generasi NIM yang **wajib
  menyitir** jawaban hasil retrieve; rail NeMo-Guardrails self-check yang memaksa disclaimer dan memblokir
  saran dosis; evaluasi RAGAS faithfulness; plus demo FastAPI. Laporannya membandingkan jawaban yang grounded
  dan yang tanpa RAG pada 25 pertanyaan.
- **Kenapa cukup di T4:** Murni inference dan retrieval, tanpa training. MiniLM, FAISS-CPU, dan bge-reranker-v2-m3
  semuanya muat di 16 GB; sementara generasi, RAGAS judge, dan guardrail self-check di-offload ke NIM. Satu-satunya
  jebakan adalah CSV 2.83 GB, jadi di-stream dan disampel, jangan pernah di-load penuh.
- **Titik mulai:** Anonimkan kolom `doctor_name`, buang jawaban yang lebih dari 2000 karakter, dan sampel
  sekitar 30k. Tiap dokumen retrieve berisi title plus answer; embed ke FAISS-CPU (IndexFlatIP, ternormalisasi)
  seperti di M05 nb05; ambil FAISS top-50, lalu bge-reranker top-5. Buat system prompt NIM yang memaksa sitasi
  plus menambahkan disclaimer, lalu bungkus dengan NeMo Guardrails self-check (M06 nb04). Jalankan RAGAS
  faithfulness pada 25 pertanyaan (grounded vs tanpa RAG), lalu expose lewat FastAPI `/ask` (M05 nb08 plus
  `RagEngine` di M06 nb05).
- **Stretch goal:** Tambahkan tabel audit guardrail yang terukur, misalnya berapa persen prompt yang meminta
  dosis berhasil diblokir.

## 4. IDK-RAG: asisten Indonesia yang tahu kapan bilang "tidak tahu"

- **Modul:** M05 (inti) plus M06, M03, M04. **Tingkat:** Menengah.
- **Ide singkat:** Sistem RAG yang seluruh alasan keberadaannya adalah **abstention yang terkalibrasi**: ia
  menjawab hanya saat bukti mendukung, dan secara eksplisit bilang "maaf, informasi itu tidak ada dalam dokumen"
  jika tidak. Dibangun dan dievaluasi di **IDK-MRC**, dataset Indonesia yang sengaja berisi pertanyaan yang bisa
  dijawab sekaligus yang tidak bisa dijawab. Dengan begitu peserta bisa mengukur kualitas jawaban sekaligus
  seberapa sering sistem nekat menjawab pertanyaan mustahil, yaitu failure mode yang justru diabaikan oleh
  kebanyakan demo RAG.
- **Dataset:** `rifkiaputri/idk-mrc`. Lisensi CC-BY-4.0, korpus kecil (sekitar 3.659 paragraf).
  URL: <https://huggingface.co/datasets/rifkiaputri/idk-mrc>
- **Yang dibangun:** Satu notebook RAG yang dikontrol oleh tingkat keyakinan (skor reranker plus cek "bisa
  dijawab?" sebelum generasi), dievaluasi pada split test idk-mrc: EM/F1 untuk pertanyaan yang bisa dijawab,
  plus akurasi abstention untuk yang tidak bisa dijawab, ditambah kurva precision vs coverage, dan demo FastAPI
  `/ask` yang mengembalikan "tidak tahu" pada pertanyaan di luar korpus.
- **Kenapa cukup di T4:** Tanpa training. Korpusnya mungil sehingga embed selesai dalam hitungan detik; FAISS-CPU
  plus bge-reranker-v2-m3 sepele di 16 GB; generasi di-offload ke NIM (kalaupun mau inference Qwen 4-bit lokal,
  itu juga muat).
- **Titik mulai:** Angkat pipeline capstone nb07 (MiniLM -> FAISS-CPU -> bge-reranker -> NIM Llama-3.3-70B), lalu
  ganti korpusnya dengan paragraf konteks IDK-MRC. Bentuk test set dengan menelusuri `qas[]` dan menyimpan flag
  `is_impossible`. Tambahkan gerbang abstention dua tahap (abstain kalau skor reranker tertinggi di bawah ambang),
  tuning ambangnya di split VAL, lalu laporkan hasilnya di split TEST. Bungkus dengan FastAPI nb08.
- **Stretch goal:** Bandingkan gerbang berbasis ambang reranker dengan sinyal abstention kedua, yaitu cek
  *self-consistency* bertipe "apakah saya yakin?".

## 5. Wartawan Mini: spesialis ringkasan & judul berita Indonesia *(IndoSum)*

- **Modul:** M04 (inti) plus M03, M06. **Tingkat:** Mahir.
- **Ide singkat:** Fine-tune **Qwen3-1.7B (QLoRA)** menjadi spesialis yang membaca artikel berita Indonesia lalu
  menghasilkan ringkasan abstraktif 2-3 kalimat plus satu judul, dievaluasi dengan ROUGE/BERTScore dan diadu
  langsung melawan model raksasa Llama-3.3-70B via NIM. Tujuannya membuktikan bahwa model kecil yang terfokus
  bisa menyaingi model raksasa di satu domain spesifik. Ini tugas generatif betulan, bukan sekadar klasifikasi.
- **Dataset:** IndoSum (Indonesian Text Summarization Benchmark). Lisensi CC-BY-SA-4.0 (data) dan Apache-2.0
  (kode). URL: <https://github.com/kata-ai/indosum>
- **Yang dibangun:** Satu notebook. Parse IndoSum fold 1, format instruksi "ringkas plus beri judul", QLoRA
  fine-tune, evaluasi ROUGE/BERTScore untuk base vs spesialis vs NIM-70B dengan uji signifikansi berpasangan,
  merge dan GGUF, lalu demo tempel-artikel yang langsung mengeluarkan ringkasan plus judul.
- **Kenapa cukup di T4:** QLoRA 4-bit Qwen3-1.7B pada seq_len 1024 memakai sekitar 5-7 GB. Evaluasinya
  (ROUGE/BLEU/BERTScore plus uji-t scipy) ringan dan sudah ada di M04 nb04, judge 70B di-offload ke NIM, dan
  datasetnya cuma puluhan MB.
- **Titik mulai:** Unduh IndoSum fold 1, lalu format tiap artikel sebagai `prompt='Ringkas dalam 2-3 kalimat dan
  beri satu judul:\n<artikel ~1024 token>'` dengan `target='Judul: ...\nRingkasan: ...'`. Pakai ulang kerangka
  QLoRA dari M04 nb05 (Qwen3-1.7B, 4-bit, fp16, gradient checkpointing, batch 1 plus grad-accum). Pakai ulang sel
  evaluasi nb04, tapi ganti BERTScore ke model multilingual/IndoBERT. Skor base vs spesialis vs NIM Llama-3.3-70B
  pada test fold, lalu tutup dengan demo merge -> GGUF dari nb07.
- **Stretch goal:** Coba SmolLM3-3B 4-bit sebagai spesialis yang lebih besar (fp16 plus gradient checkpointing
  plus batch 1), lalu bandingkan trade-off antara ukuran dan kualitasnya.

## 6. JDIH-Asisten: Ask-My-Regulation untuk hukum ketenagakerjaan (sitasi Pasal)

- **Modul:** M05 (inti) plus M03, M06, M01. **Tingkat:** Menengah.
- **Ide singkat:** Asisten Ask-My-Document atas korpus regulasi resmi dari **JDIH BPK**, fokus ke hukum
  ketenagakerjaan (UU 13/2003 plus klaster Cipta Kerja). **Docling** mem-parse PDF regulasi sambil menjaga
  struktur Pasal/Ayat, sehingga tiap jawaban bisa menyitir pasal persis asalnya, dan akan abstain bila tidak ada
  pasal yang relevan. Korpusnya domain publik (peraturan perundang-undangan tidak ber-hak-cipta), jadi
  lisensinya sebersih mungkin.
- **Dataset:** UU 13/2003 Ketenagakerjaan plus klaster Cipta Kerja (PDF resmi JDIH BPK). Domain publik (Pasal 42
  UU 28/2014: peraturan perundang-undangan tidak ada hak cipta).
  URL: <https://peraturan.bpk.go.id/Details/43013/uu-no-13-tahun-2003>
- **Yang dibangun:** Satu notebook. Ingest PDF regulasi dengan Docling menjadi chunk ber-tag Pasal -> FAISS,
  hybrid retrieve plus rerank, jawaban dari NIM dengan sitasi Pasal yang wajib, evaluasi RAGAS
  faithfulness/answer-relevancy pada sekitar 30 pertanyaan ketenagakerjaan buatan tangan, demo FastAPI `/ask`,
  dan laporan 4 halaman yang menyoroti citation-grounding serta abstention.
- **Kenapa cukup di T4:** Semua pekerjaan lokal ringan: parsing Docling (CPU plus RapidOCR), embedding MiniLM,
  FAISS-CPU atas beberapa lusin PDF, dan bge-reranker-v2-m3. Generasi jawaban dan RAGAS judge di-offload ke NIM,
  dan korpusnya kecil (UU 13/2003 sekitar 79 halaman).
- **Titik mulai:** Fork nb07: ganti URL-fetch arXiv dengan parsing Docling untuk PDF UU 13/2003
  (`peraturan.bpk.go.id/Download/31128`). Di langkah HybridChunker, tambahkan satu field metadata dengan
  regex untuk menangkap heading "Pasal N (ayat (m))" terdekat, supaya `source_label` ikut memunculkan nomor
  Pasal. Sisanya pertahankan apa adanya (MiniLM -> FAISS -> bge-reranker -> NIM dengan prompt sitasi wajib plus
  abstain bila kosong). Port RAGAS dari nb04 untuk sekitar 30 pertanyaan, lalu pakai ulang FastAPI nb08.
  Kerjakan dulu jalur paling sederhana, yaitu satu UU saja.
- **Stretch goal:** Tambahkan amandemen Cipta Kerja (UU 6/2023) dan lacak Pasal mana yang masih berlaku dan mana
  yang sudah dicabut, agar sitasi tidak pernah memunculkan teks yang sudah dicabut.

## 7. Deteksi transaksi fraud pada data tidak seimbang *(imbalanced learning end-to-end)*

- **Modul:** M01. **Tingkat:** Menengah.
- **Ide singkat:** Bangun pipeline deteksi fraud kartu kredit pada data yang sangat tidak seimbang (hanya 0,172%
  fraud), dengan fokus pada teknik yang benar-benar penting: class weight, threshold tuning, SMOTE/undersampling,
  dan pemilihan metrik yang tepat (PR-AUC, recall@precision, dan metrik berbasis biaya). Project ini mengajarkan
  jebakan "akurasi yang menyesatkan" dan trade-off antara precision dan recall pada kasus fintech nyata yang
  biaya kesalahannya tinggi. Capstone classical ML yang sangat pas.
- **Dataset:** Credit Card Fraud Detection (mlg-ulb / ULB-Worldline). Lisensi Database Contents License (DbCL)
  v1.0, ukuran sekitar 144 MB. URL: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
- **Yang dibangun:** Satu notebook. EDA yang membuktikan jebakan akurasi -> baseline LogReg -> eksperimen
  resampling (SMOTE/undersample/class_weight, **hanya di dalam fold training**) -> bandingkan XGBoost dan
  IsolationForest lewat PR-AUC/recall/precision plus confusion matrix -> threshold tuning berbasis biaya ->
  pemilihan model final -> demo skoring satu transaksi. Laporannya berisi kurva PR, tabel trade-off, dan
  rekomendasi operasional.
- **Kenapa cukup di T4:** Classical ML murni di CPU atas sekitar 285k baris numerik yang sudah ter-PCA. CSV
  sekitar 144 MB load dalam hitungan detik, dan tidak ada model yang menyentuh VRAM. Justru T4 belum
  dimanfaatkan penuh.
- **Titik mulai:** Load `creditcard.csv` dari mirror parquet HuggingFace (tanpa login) atau dari OpenML id 1597.
  Buat EDA jebakan-akurasi dalam 2 sel (model dummy "selalu bukan-fraud" mencapai akurasi 99,83%, tapi nol fraud
  tertangkap). Lakukan stratified split, lalu baseline LogReg. Rangkai `imblearn.Pipeline` supaya SMOTE atau
  undersampling **hanya** terjadi di dalam fold training, untuk mencegah leakage. Evaluasi dengan
  `precision_recall_curve` / `average_precision` plus recall pada precision tetap. Iterasi class_weight vs SMOTE
  vs undersample lintas LogReg/RF/XGBoost, tambahkan IsolationForest sebagai pembanding unsupervised, lalu pilih
  threshold dengan meminimalkan fungsi biaya fraud.
- **Stretch goal:** Tambahkan kalibrasi probabilitas (Platt/isotonic) dan sapuan cost-curve untuk menunjukkan
  bagaimana titik operasi optimal bergeser saat rasio antara biaya fraud dan biaya review berubah.

## 8. Pengenal motif batik Nusantara dengan Grad-CAM *(ResNet50 transfer learning)*

- **Modul:** M02 (inti) plus M06. **Tingkat:** Menengah.
- **Ide singkat:** Classifier 15 kelas motif batik (ResNet50 transfer learning) yang memakai **Grad-CAM** untuk
  menyorot bagian kain mana yang paling menentukan prediksi, misalnya lingkaran kawung, diagonal parang, atau
  awan megamendung. Jadi project ini sekaligus mengklasifikasi tekstil warisan budaya dan menjelaskan
  keputusannya secara visual; cocok untuk demo museum atau edukasi, dan datasetnya berlisensi CC0.
- **Dataset:** Indonesian Batik Motifs (Corak App). Lisensi CC0 (domain publik), ukuran sekitar 156 MB.
  URL: <https://www.kaggle.com/datasets/alfanme/indonesian-batik-motifs-corak-app>
- **Yang dibangun:** Satu notebook. Training transfer learning (ResNet50, opsional MobileNetV2), metrik per
  kelas, overlay heatmap Grad-CAM, opsional benchmark latency ONNX/TensorRT, dan demo Gradio yang menyebut nama
  motif sambil menampilkan heatmap "di bagian mana ia melihat". Laporannya membandingkan dua backbone dan
  mendaftar motif yang paling sering tertukar.
- **Kenapa cukup di T4:** Sekitar 156 MB gambar 224x224, dan ResNet50/MobileNet transfer learning (batch 32,
  fp16) memakai sekitar 3-5 GB. Grad-CAM cuma satu forward dan backward pass, dan benchmark ONNX/TensorRT yang
  opsional itu jalan di T4 itu sendiri.
- **Titik mulai:** Mulai dari M02 nb03 (`03_cat_dog_classifier_resnet.ipynb`), ganti loader CIFAR-10 dengan
  `image_dataset_from_directory` atas folder batik (224x224, batch 32, `validation_split=0.2`, `.map`
  preprocess_input, `.prefetch`). Bangun `ResNet50(include_top=False)` -> GAP -> Dropout -> `Dense(15, softmax)`
  dengan backbone beku, latih sekitar 10-15 epoch (mixed_float16). Laporkan precision/recall per kelas plus
  confusion matrix, lalu tambahkan Grad-CAM (GradientTape, bobot channel dari GAP, ReLU, resize plus overlay) dan
  `gr.Interface(inputs=gr.Image, outputs=[gr.Label, gr.Image])`.
- **Stretch goal:** Pakai ulang resep tf2onnx plus tensorrt 10.x yang sudah ter-pin di M02 nb05 untuk benchmark
  latency ONNX -> TensorRT versus PyTorch di T4.

## 9. Estimator harga mobil bekas Indonesia *(regresi plus demo interaktif, jalur masuk Pemula)*

- **Modul:** M01. **Tingkat:** **Pemula**.
- **Ide singkat:** Latih model regresi untuk memprediksi harga mobil bekas Indonesia dari listing nyata (merek,
  tahun, kilometer, transmisi, lokasi), lalu bandingkan Linear/Ridge vs RandomForest vs XGBoost dan analisis
  fitur mana yang paling menentukan harga. Domainnya sangat dekat dengan keseharian orang Indonesia, dan
  outputnya langsung berguna sebagai alat "cek harga wajar".
- **Dataset:** Used Car Listings in Indonesia (`used_car.csv`). Lisensi CC0 (domain publik), ukuran sekitar 25 KB
  zip (sekitar 1-2,5 ribu baris).
  URL: <https://www.kaggle.com/datasets/indraputra21/used-car-listings-in-indonesia>
- **Yang dibangun:** Satu notebook. Cleaning (parse harga Rupiah dan kilometer, log-transform target, tangani
  outlier) -> feature engineering (umur mobil, encoding) -> bandingkan 3 model atau lebih dengan RMSE/MAE/R2 plus
  cross-validation -> feature importance dan analisis error per segmen -> fungsi `estimasi_harga(spek)` lewat
  ipywidgets yang menampilkan rentang prediksi. Laporan singkatnya berisi tabel perbandingan model dan insight
  pasar.
- **Kenapa cukup di T4:** Regresi tabular murni atas zip 25 KB, selesai di CPU dalam hitungan detik. Akselerasi
  cuML yang opsional (M01 nb10) cuma bonus.
- **Titik mulai:** Tarik `used_car.csv` lewat Kaggle API. Parse harga dan kilometer ke numerik dengan regex,
  log-transform harga, lalu hitung `car_age = tahun_sekarang - year`, turunkan merek dari nama mobil, dan encode
  merek/transmisi/lokasi. Latih Linear/Ridge/RandomForest/XGBoost dalam satu loop dengan `cross_val_score` 5-fold
  yang melaporkan RMSE/MAE/R2. Plot `feature_importances_` dari XGBoost, lakukan analisis residual per segmen
  harga, lalu bungkus model terbaik di `estimasi_harga(spek)` dengan rentang dari kuantil residual cross-validation.
- **Stretch goal:** Jalankan ulang RandomForest di bawah cuML (M01 nb10) untuk menunjukkan speedup di T4, dan
  tambahkan dataset penjualan mobil Indonesia yang lebih kaya untuk analisis per segmen yang lebih dalam.

## 10. Asisten triase kesehatan: spesialis tanya-jawab kesehatan Indonesia plus edge GGUF

> **PENTING:** Ini artefak edukasi, **bukan** alat triase sungguhan. Output wajib ditutup dengan disclaimer dan
> blok "kapan harus ke dokter".

- **Modul:** M04 (inti) plus M03, M06. **Tingkat:** Mahir.
- **Ide singkat:** *Instruction-tune* **Qwen3-1.7B (QLoRA)** menjadi spesialis yang menjawab pertanyaan kesehatan
  awam dalam Bahasa Indonesia dengan struktur ala dokter, dilatih dari tanya-jawab Alodokter asli. Sisi
  keamanannya: setiap jawaban wajib diakhiri disclaimer plus triase red-flag "kapan harus ke dokter", jadi ini
  spesialis yang sadar batas, bukan chatbot generik. Lalu di-deploy offline di edge lewat Ollama GGUF, untuk
  klinik yang tidak punya internet.
- **Dataset:** `agufsamudra/alodokter-qna`. Lisensi Apache-2.0.
  URL: <https://huggingface.co/datasets/agufsamudra/alodokter-qna>
- **Yang dibangun:** Satu notebook. Load dan bersihkan Alodokter QA (anonimkan `doctor_name`, filter panjang
  jawaban) -> format instruksi dengan kerangka disclaimer yang wajib -> QLoRA Qwen3-1.7B -> bandingkan base vs
  spesialis -> LLM-judge via NIM untuk menilai keamanan dan faktualitas -> merge fp16 -> GGUF Q4_K_M -> demo
  Ollama/Gradio. Laporannya berisi contoh sebelum dan sesudah, skor judge, serta analisis kasus di mana model
  justru **tidak boleh** menjawab (batas keamanan).
- **Kenapa cukup di T4:** QLoRA 4-bit Qwen3-1.7B dengan compute fp16 memakai sekitar 5-9 GB di T4, masih di bawah
  plafon 3B. Konversi GGUF jalan di CPU (`GGML_CUDA=OFF`), LLM-judge di-offload ke NIM, dan dataset disubsample
  ke sekitar 10k.
- **Titik mulai:** Load `agufsamudra/alodokter-qna` dengan pandas, langsung anonimkan `doctor_name` dan buang
  jawaban yang lebih dari sekitar 2000 karakter, lalu subsample sekitar 10k. Bungkus tiap pasangan dalam template
  instruksi yang kerangka outputnya **selalu** menambahkan disclaimer plus blok red-flag, supaya aspek keamanan
  ikut dilatih, bukan sekadar diharapkan muncul. QLoRA Qwen3-1.7B dengan compute fp16 (**bukan** bf16), r=16, 1-2
  epoch. Pakai ulang jalur merge fp16 plus llama.cpp GGUF Q4_K_M plus demo Ollama dari M04. Bingkai laporannya
  secara tegas sebagai artefak edukasi, bukan alat triase yang siap pakai.
- **Stretch goal:** Tambahkan rail NeMo Guardrails self-check yang memaksa disclaimer dan memblokir saran dosis,
  lalu ukur berapa banyak output tak aman yang berhasil ditangkapnya.

## 11. *(Bonus)* Radar emosi ulasan Tokopedia: klasik vs IndoBERT plus dashboard penjual

- **Modul:** M03 (inti) plus M01, M02. **Tingkat:** Menengah.
- **Ide singkat:** Fine-tune **IndoBERT-base** untuk mengklasifikasi review Tokopedia ke 5 emosi, lalu adu
  langsung dengan baseline TF-IDF + SVM, dan kemas semuanya dalam dashboard Gradio "seller insight" yang
  memetakan emosi per kategori produk serta menandai review marah atau sedih sebagai komplain prioritas. Project
  ini memberi penjual UMKM alat untuk membaca suara pelanggan, lebih dari sekadar rating bintang, sekaligus
  menunjukkan perbandingan antara baseline klasik dan transformer.
- **Dataset:** PRDECT-ID. Lisensi CC-BY-4.0, ukuran 1.26 MB.
  URL: <https://data.mendeley.com/datasets/574v66hf2v/1>
- **Yang dibangun:** Satu notebook. EDA -> baseline klasik (TF-IDF + LinearSVC, macro-F1 plus confusion matrix)
  -> fine-tune IndoBERT (`AutoModelForSequenceClassification`) -> confusion matrix dan F1 per emosi plus analisis
  error -> dashboard Gradio seller insight yang memetakan emosi per kategori dan menandai Anger/Sadness sebagai
  komplain prioritas. Laporannya 4-6 halaman membandingkan klasik vs transformer.
- **Kenapa cukup di T4:** IndoBERT-base (sekitar 125M parameter) yang di-fine-tune pada max_len 128, batch 16,
  fp16 memakai di bawah 4 GB dan rampung 3 epoch dalam hitungan menit. Baseline TF-IDF + SVM jalan di CPU, dan
  datasetnya cuma satu CSV 1.26 MB.
- **Titik mulai:** `pd.read_csv` dari URL raw GitHub (URL-encode spasi pada nama file), EDA kolom Emotion dan
  Category, label-encode 5 emosi, lalu stratified split. Bangun baseline dulu (TfidfVectorizer plus LinearSVC,
  macro-F1 plus `confusion_matrix`), baru fine-tune `indobenchmark/indobert-base-p1` dengan `num_labels=5`,
  memakai ulang pola Trainer dari M04 (fp16, batch 16, 3 epoch), tapi ganti ke `DataCollatorWithPadding` dan
  tambahkan `compute_metrics` untuk macro-F1. Tandai secara eksplisit bahwa pergantian head ke
  `AutoModelForSequenceClassification` ini tidak diajarkan di kelas. Terakhir, bungkus inference di `gr.Interface`.
- **Stretch goal:** Tambahkan perbandingan klasifikasi emosi antara model fine-tuned dan zero-shot NIM-LLM, untuk
  menunjukkan di mana model kecil yang terspesialisasi mengalahkan few-shot prompting.

---

## Catatan untuk instruktur

### Saran rubrik penilaian (bisa disesuaikan)
- **Fungsionalitas end-to-end (30%):** notebook jalan dari data sampai model/sistem sampai demo, tanpa error fatal.
- **Ketepatan metrik dan evaluasi (25%):** memakai metrik yang benar untuk masalahnya, misalnya PR-AUC (bukan
  akurasi) untuk #7; faithfulness dan abstention untuk RAG; ROUGE plus uji signifikansi untuk #5.
- **Trustworthy AI dan keamanan (15%):** ada disclaimer, abstention, sitasi, audit bias, atau guardrail sesuai
  konteks.
- **Kualitas laporan dan komunikasi (20%):** analisis error yang jujur, keterbatasan yang diakui, dan insight,
  bukan cuma angka.
- **Demo dan reproduktibilitas (10%):** demo interaktifnya jalan, dan orang lain bisa menjalankan ulang di Colab T4.

### Pembagian berdasarkan minat peserta
- **Suka data dan bisnis (tanpa GPU berat):** #7 fraud, #9 harga mobil, #1 audit bias.
- **Suka computer vision:** #8 batik plus Grad-CAM.
- **Suka LLM dan fine-tuning:** #2 NL-to-SQL, #5 ringkasan berita, #10 triase kesehatan.
- **Suka sistem RAG dan search:** #3 RAG kesehatan, #4 IDK-RAG, #6 JDIH regulasi.
- **Suka NLP klasik plus transformer:** #11 radar emosi.

### Dataset yang sengaja DIHINDARI (jebakan capstone)
- **MAFINDO / Cek-Klaim-ID hoax:** datasetnya di HuggingFace bersifat gated (butuh approval manual), sehingga
  bisa menghambat capstone yang waktunya cuma 1-2 minggu. Dataset hoax terbuka yang jadi alternatif punya label
  yang kurang bersih.
- **"Indonesian agriculture instruction set":** ini dataset fiktif, tidak ada yang benar-benar tersedia; ide
  agri-LLM-nya pun sekitar 90% cuma menyalin M04 nb07.
- **PRDECT-ID metadata -> sentiment classifier:** datanya bocor sehingga jadi terlalu mudah (F1 mencapai 0,989
  karena rating membocorkan label); kalau ratingnya dibuang, malah hampir tak berguna (F1 0,715 vs baseline
  0,686). Karena itu PRDECT-ID hanya dipakai untuk task yang layak, yaitu #1 audit bias dan #11 klasifikasi emosi.
- **Skill yang tidak diajarkan:** NER redaction (alignment label subword), kalibrasi INT8 ONNX/TensorRT, dan
  retrieval hybrid BM25 sengaja dibuang atau ditandai eksplisit, karena tidak pernah masuk silabus.

> **Catatan teknis penting:** dua project (#11 IndoBERT, dan sebagian #1) butuh
> `AutoModelForSequenceClassification` yang **tidak** persis diajarkan di M04 (M04 fokus pada
> `AutoModelForCausalLM`). Ini disengaja sebagai tantangan capstone yang ringan: peserta harus membaca
> dokumentasi sendiri, dan langkah ini sudah ditandai eksplisit di bagian titik mulai.
