#!/usr/bin/env python3
"""Build the Modul 03 NLP Fundamentals quiz HTML — 30 concept questions (code=null).
Sebaran: bagian A 6, bagian B 4, bagian C 8, bagian D 8, notebook 02 (RAPIDS) 4.
Run: python build_quiz.py  ->  ../nlp-fundamentals-quiz.html
"""
import json, pathlib

# (pertanyaan, [4 opsi], indeks jawaban, penjelasan). Opsi ditulis sepanjang ± sama.
Q = [
    # --- Bagian A: membersihkan teks (6) ---
    ("Kenapa teks diubah ke huruf kecil sebelum dihitung frekuensinya?",
     ["Agar model tidak perlu mempelajari aturan kapitalisasi bahasa Indonesia yang rumit",
      "Agar `NLP` dan `nlp` dihitung sebagai satu kata, bukan dua dimensi fitur terpisah",
      "Agar ukuran berkas korpus mengecil sehingga proses tokenisasi berjalan lebih cepat",
      "Agar tokenizer bisa mengenali batas kalimat tanpa bantuan tanda baca apa pun lagi"], 1,
     "Setiap variasi ejaan yang tidak disatukan menjadi satu kolom fitur tambahan, dan tiap kolom harus dipelajari model dari datanya sendiri. Menyatukan NLP dan nlp berarti bukti yang ada tidak terbagi. Ukuran berkas dan deteksi kalimat bukan alasannya."),

    ("Sebuah tim membuang seluruh stopword sebelum menganalisis sentimen ulasan. Risiko terbesarnya apa?",
     ["Jumlah token menyusut sehingga model kehilangan kemampuan mengenali topik dokumen",
      "Kata `tidak` ikut terbuang, sehingga `tidak enak` dan `enak` jadi tidak terbedakan",
      "Daftar stopword bahasa Indonesia belum lengkap sehingga sebagian kata umum tertinggal",
      "Stopword yang dibuang membuat lemmatisasi berikutnya menghasilkan bentuk dasar keliru"], 1,
     "Banyak daftar stopword memuat kata negasi seperti `tidak`. Untuk analisis sentimen, membuangnya menghapus justru informasi yang paling menentukan. Stopword bukan daftar universal — isinya harus disesuaikan dengan tugasnya."),

    ("Apa perbedaan mendasar antara stemming dan lemmatisasi?",
     ["Stemming bekerja per kalimat dengan konteks, lemmatisasi bekerja per kata tanpa konteks",
      "Stemming memotong imbuhan mekanis, lemmatisasi memakai kamus dan aturan morfologi",
      "Stemming hanya tersedia untuk bahasa Inggris, lemmatisasi tersedia untuk semua bahasa",
      "Stemming memerlukan model neural, lemmatisasi cukup memakai pencocokan pola sederhana"], 1,
     "Stemming memangkas imbuhan tanpa kamus sehingga hasilnya boleh jadi bukan kata; lemmatisasi berbasis kamus sehingga keluarannya selalu kata yang benar. Keduanya sama-sama bekerja per kata; ketersediaan bahasa bukan pembedanya."),

    ("Kenapa `PhraseTokenizer` dipakai sebelum analisis frekuensi kata?",
     ["Supaya entitas multi-kata seperti `kecerdasan buatan` tidak terpecah jadi dua token",
      "Supaya frekuensi dihitung per kalimat, bukan per kata, sehingga hasilnya lebih stabil",
      "Supaya stopword bisa dikenali lebih akurat karena konteks frasanya ikut diperhitungkan",
      "Supaya tokenizer bisa memberi label jenis entitas pada tiap frasa yang ditemukannya"], 0,
     "Memecah `Universitas Indonesia` menjadi dua kata umum menghilangkan maknanya. PhraseTokenizer menjaga frasa nomina tetap utuh. Ia tidak memberi label jenis entitas — untuk itu diperlukan NER."),

    ("Sebuah kalimat sudah di-POS tagging. Pemakaian yang paling masuk akal untuk hasilnya apa?",
     ["Mengambil kata benda saja sebagai kandidat topik, karena topik jarang berupa kata sambung",
      "Mengurutkan kata berdasarkan kelasnya agar kalimat menjadi lebih mudah dibaca oleh model",
      "Mengganti setiap kata dengan label kelasnya supaya kosakata fitur menjadi jauh lebih kecil",
      "Menghapus semua kata kerja karena kata kerja tidak pernah membawa informasi topik apa pun"], 0,
     "POS tag dipakai untuk menyaring. Mengambil kata benda saja adalah cara umum mencari kandidat topik. Mengurutkan ulang kata merusak kalimat, dan membuang seluruh kata kerja menghilangkan informasi tindakan yang sering berguna."),

    ("Notebook membuang stopword lebih dulu, baru melakukan lemmatisasi. Alasan urutan itu apa?",
     ["Lemmatisasi selalu gagal kalau daftar tokennya masih memuat kata umum berfrekuensi tinggi",
      "Daftar stopword berisi bentuk permukaan umum, jadi cocoknya lebih akurat sebelum diubah",
      "Membuang stopword setelah lemmatisasi akan membuat jumlah token akhir menjadi tidak stabil",
      "Lemmatisasi mengubah huruf besar menjadi kecil, sehingga stopword tidak lagi bisa dikenali"], 1,
     "Daftar stopword `nlp-id` disusun dari bentuk kata yang lazim muncul, bukan bentuk dasarnya. Mencocokkannya sebelum lemmatisasi karena itu lebih akurat. Urutan ini pilihan yang beralasan, bukan keharusan mutlak."),

    # --- Bagian B: menganalisis teks (4) ---
    ("Sepuluh kata teratas sebuah dokumen berbahasa Indonesia adalah `yang`, `di`, `dan`, `adalah`. Kesimpulan yang tepat?",
     ["Dokumen tersebut kemungkinan besar membahas topik umum yang tidak spesifik ke bidang apa pun",
      "Pembersihan teks belum dijalankan, jadi daftar itu belum memberi informasi apa pun soal isi",
      "Korpusnya terlalu kecil, jadi kata bermakna belum sempat muncul dengan frekuensi yang memadai",
      "Tokenizer yang dipakai keliru, karena tokenizer benar akan otomatis melewati kata penghubung"], 1,
     "Daftar seperti itu akan muncul untuk dokumen berbahasa Indonesia apa pun karena stopword belum dibuang. Bukan sifat dokumennya, bukan pula ukuran korpus atau kesalahan tokenizer — tokenizer memang tidak bertugas menyaring stopword."),

    ("Kapan bar chart lebih tepat dipakai daripada word cloud untuk menyajikan frekuensi kata?",
     ["Saat hasilnya dipakai untuk mengambil keputusan, karena panjang batang bisa dibandingkan",
      "Saat jumlah kata unik sangat banyak, karena word cloud hanya sanggup menampilkan sedikit kata",
      "Saat teksnya berbahasa Indonesia, karena word cloud tidak menangani imbuhan dengan semestinya",
      "Saat frekuensi antar kata berdekatan, karena word cloud selalu gagal menampilkan kata langka"], 0,
     "Mata manusia buruk membandingkan luas huruf, dan tata letak word cloud acak sehingga tidak membawa informasi. Panjang batang bisa dibandingkan dengan akurat. Word cloud tetap berguna untuk menarik perhatian, bukan untuk memutuskan."),

    ("Apa yang membedakan NER dari tokenisasi frasa?",
     ["NER bekerja pada teks berbahasa Inggris, tokenisasi frasa dirancang khusus untuk Indonesia",
      "NER memberi label jenis entitas, sedangkan tokenisasi frasa hanya menyatukan kata jadi satu",
      "NER memerlukan pembersihan teks lebih dulu, sedangkan tokenisasi frasa memakai teks mentah",
      "NER menghasilkan vektor makna per entitas, tokenisasi frasa menghasilkan hitungan frekuensi"], 1,
     "Keduanya menemukan satuan multi-kata, tetapi hanya NER yang memberi label seperti PERSON, ORG, atau LOC. Label itulah yang membuat keluaran NER bisa langsung dipetakan ke kolom basis data."),

    ("Sebuah tim perlu menarik nama perusahaan dan tanggal dari ribuan dokumen kontrak berbahasa Indonesia. Pilihan yang paling tepat?",
     ["spaCy dengan model `en_core_web_sm`, karena model Inggris tetap akurat untuk nama diri",
      "Model NER berbasis IndoBERT di Hugging Face, dengan biaya unduhan besar dan butuh GPU",
      "Daftar kata kunci manual, karena nama perusahaan dan tanggal polanya sudah cukup baku",
      "TF-IDF pada seluruh dokumen, lalu ambil istilah dengan bobot tertinggi sebagai entitasnya"], 1,
     "Untuk teks Indonesia, model NER yang dilatih pada bahasa Indonesia memberi akurasi terbaik, dengan biaya unduhan dan komputasi. Model Inggris tidak dilatih pada konteks Indonesia, dan TF-IDF tidak mengenali jenis entitas."),

    # --- Bagian C: menilai teks (8) ---
    ("TextBlob mengembalikan `polarity = 0.0` untuk sebuah kalimat berbahasa Indonesia. Tafsiran yang benar?",
     ["Kalimat itu benar-benar netral, karena skor nol memang menandakan tidak ada muatan emosi",
      "Tidak ada kata yang dikenali kamusnya, jadi nol di sini berarti tidak tahu, bukan netral",
      "Kalimat itu terlalu pendek, sehingga TextBlob belum punya cukup kata untuk menghitung skor",
      "Kalimatnya mengandung negasi, dan negasi selalu membuat skor polaritas kembali menjadi nol"], 1,
     "Kamus TextBlob hanya berbahasa Inggris. Kalimat Indonesia menghasilkan nol karena tidak ada kata yang dikenali, dan model tidak membedakannya dari netral sungguhan. Ini kegagalan diam yang tidak memunculkan peringatan apa pun."),

    ("TextBlob menilai `The plot was not good` dengan benar (-0,35) tetapi `I would not call this a great movie` keliru (+0,80). Apa penjelasannya?",
     ["Kalimat kedua lebih panjang, dan skor leksikon memang menjadi tidak stabil di kalimat panjang",
      "Aturan negasi leksikon hanya membalik kata tepat sesudah `not`, sedangkan di kalimat kedua jaraknya tiga kata",
      "Kata `great` punya skor jauh lebih tinggi daripada `good`, sehingga skor akhirnya ikut terangkat",
      "Kalimat kedua memuat kata `call` yang tidak ada di kamus, jadi skornya dihitung tanpa negasi"], 1,
     "Leksikon bukan tidak bisa menangani negasi sama sekali: `not good` dibalik dengan benar karena kata sasarannya berdekatan. Yang gagal adalah negasi berjarak, karena `not` tidak lagi menjangkau `great`. Sarkasme gagal karena alasan serupa: isyaratnya ada di konteks, bukan di kata."),

    ("Kalimat `Gw suka bgt sama teknologi NLP nih` gagal dinormalkan pipeline klasik, tetapi dikenali positif oleh IndoBERT. Penjelasannya apa?",
     ["Transformer menormalkan bahasa gaul ke bentuk baku lebih dulu sebelum menilai sentimennya",
      "Transformer dilatih dari teks nyata termasuk bahasa informal, jadi ragam itu ada di datanya",
      "Transformer memakai daftar kata gaul tambahan yang disediakan terpisah oleh pembuat modelnya",
      "Transformer mengabaikan kata yang tidak dikenal, lalu menilai dari sisa kata yang masih baku"], 1,
     "Bedanya bukan kecerdasan model, melainkan asal pengetahuannya: kamus leksikon disusun dari teks baku, sedangkan transformer dilatih dari teks nyata. Tokenisasi subword juga membuat kata tak dikenal tetap terwakili sebagian."),

    ("Sebuah tim harus menyaring sepuluh juta ulasan berbahasa Inggris di server tanpa GPU. Pendekatan sentimen yang paling masuk akal?",
     ["Transformer, karena akurasinya lebih tinggi dan itu selalu lebih penting daripada kecepatan",
      "Leksikon, karena murah dan transparan, lalu transformer untuk kasus yang perlu diputuskan",
      "Transformer dengan ukuran batch sangat besar, supaya beban komputasinya tersebar lebih merata",
      "Leksikon saja untuk semuanya, karena transformer tidak akan pernah jalan tanpa perangkat GPU"], 1,
     "Berlapis adalah pola yang lazim di produksi: leksikon menyaring murah, transformer memutuskan sisanya. Transformer tetap bisa jalan di CPU, hanya lambat, jadi menyebutnya mustahil juga tidak tepat."),

    ("Apa yang hilang saat sebuah dokumen diubah menjadi representasi bag-of-words?",
     ["Frekuensi kemunculan kata, karena yang dicatat hanya ada atau tidak adanya kata tersebut",
      "Urutan kata, sehingga `not good` dan `good` memberi sinyal yang sama kepada classifier",
      "Kata langka, karena bag-of-words hanya menyimpan kata yang paling sering muncul di korpus",
      "Panjang dokumen, sehingga dokumen panjang dan pendek tidak lagi bisa dibedakan modelnya"], 1,
     "Bag-of-words membuang urutan kata sepenuhnya. Akibatnya negasi hilang: `not good` dan `good` menghasilkan sinyal serupa. Frekuensi bisa dipertahankan, dan pemilihan kata fitur adalah keputusan terpisah."),

    ("Kenapa Naive Bayes masih diajarkan meskipun akurasinya kalah dari model modern?",
     ["Karena asumsi kebebasan antar katanya terbukti benar untuk data teks berbahasa apa pun",
      "Karena keputusannya bisa dibaca manusia lewat daftar kata yang paling membedakan kelas",
      "Karena ia satu-satunya algoritma yang bisa dilatih tanpa memerlukan data yang sudah dilabeli",
      "Karena ia menangani urutan kata dengan benar, berbeda dari model bag-of-words pada umumnya"], 1,
     "Nilai utamanya keterbacaan: daftar fitur informatif bisa dibawa ke rapat. Asumsi kebebasan antar kata jelas salah tetapi tetap bekerja; ia tetap butuh data berlabel dan tetap memakai bag-of-words."),

    ("Kosakata 2.000 kata fitur dihitung dari seluruh korpus, baru datanya dipecah train dan test. Apa yang terjadi?",
     ["Tidak ada masalah, karena yang dihitung hanya frekuensi kata dan bukan bobot model apa pun",
      "Terjadi kebocoran data: data test ikut menentukan fitur, jadi akurasinya terlalu optimistis",
      "Model akan gagal dilatih, karena jumlah fitur menjadi tidak konsisten antara train dan test",
      "Akurasi menjadi terlalu rendah, karena kata khas data test ikut mengaburkan pola data train"], 1,
     "Seleksi fitur adalah langkah yang belajar dari data, jadi ia harus di-fit pada train saja. Melibatkan test membuat model mengintip sebelum diuji. Kesalahan ini tidak memunculkan error, hanya angka yang menyenangkan."),

    ("Sebuah classifier mencapai akurasi 94% pada data yang 95% isinya satu kelas. Kesimpulan yang tepat?",
     ["Model itu sangat baik, karena akurasi di atas 90% jarang tercapai pada masalah klasifikasi teks",
      "Model itu lebih buruk daripada menebak kelas terbanyak, yang sudah mencapai 95% tanpa model",
      "Akurasi tidak bisa dinilai, karena metrik akurasi memang tidak berlaku untuk data tidak seimbang",
      "Model itu mengalami kebocoran data, karena akurasi setinggi itu mustahil dicapai tanpa bocoran"], 1,
     "Baseline kelas mayoritas di sini 95%, jadi 94% berarti model lebih buruk daripada menebak. Akurasi tetap bisa dihitung, hanya tidak boleh dibaca tanpa baseline. Angka tinggi belum tentu berarti ada kebocoran."),

    # --- Bagian D: dari kata ke makna (8) ---
    ("Apa peran komponen IDF dalam TF-IDF?",
     ["Menormalkan panjang dokumen supaya dokumen panjang tidak otomatis mendapat bobot lebih besar",
      "Menekan bobot kata yang muncul di banyak dokumen, menaikkan bobot kata yang jarang muncul",
      "Menghitung berapa kali sebuah kata muncul dalam satu dokumen yang sedang sedang diproses",
      "Mengelompokkan kata bermakna mirip supaya sinonim mendapat bobot yang kurang lebih setara"], 1,
     "IDF menilai seberapa jarang sebuah kata di seluruh koleksi. Kata yang ada di mana-mana dapat bobot mendekati nol — mirip fungsi daftar stopword, tetapi menyesuaikan diri dengan korpusnya. Menghitung kemunculan di satu dokumen adalah tugas TF."),

    ("`Saya suka makan nasi goreng` dan `Aku gemar menyantap nasi goreng` mendapat skor TF-IDF rendah. Kenapa?",
     ["Kedua kalimat terlalu pendek, sehingga bobot TF-IDF-nya belum sempat terbentuk dengan stabil",
      "TF-IDF hanya mencocokkan kata yang persis sama, jadi `suka` dan `gemar` dianggap tak terkait",
      "Kata `nasi` dan `goreng` terlalu sering muncul, sehingga bobot IDF keduanya menjadi sangat kecil",
      "Kedua kalimat memakai kata ganti berbeda, dan kata ganti selalu mendominasi perhitungan bobot"], 1,
     "Bagi TF-IDF, `suka` dan `gemar` adalah dua dimensi yang sama sekali tidak berhubungan. Yang tersisa untuk dicocokkan hanya `nasi` dan `goreng`. Panjang kalimat dan kata ganti bukan penyebab utamanya."),

    ("Sebuah model embedding menghasilkan vektor 384 angka. Apa yang benar tentang keluaran itu?",
     ["Panjang vektornya bergantung pada jumlah kata, jadi kalimat panjang menghasilkan vektor lebih besar",
      "Panjang vektornya tetap 384 berapa pun panjang kalimatnya, jadi kalimat bisa saling dibandingkan",
      "Tiap angka mewakili satu kata dalam kalimat, jadi kalimat maksimalnya hanya boleh 384 kata saja",
      "Tiap angka adalah frekuensi satu kata dari kosakata 384 kata yang dipilih model saat pelatihan"], 1,
     "Keluaran embedding berdimensi tetap, tidak bergantung panjang masukan. Justru keseragaman itu yang membuat kalimat panjang dan pendek bisa dibandingkan. Angkanya bukan frekuensi kata dan tidak dipetakan satu-satu ke kata."),

    ("Kenapa kemiripan dua embedding diukur dengan cosine similarity, bukan jarak biasa?",
     ["Karena cosine similarity selalu menghasilkan angka antara nol dan satu, sehingga lebih mudah dibaca",
      "Karena cosine mengukur sudut, bukan panjang, jadi dokumen panjang dan pendek adil dibandingkan",
      "Karena menghitung sudut jauh lebih cepat daripada menghitung jarak pada vektor berdimensi tinggi",
      "Karena jarak biasa tidak terdefinisi untuk vektor dengan dimensi lebih dari tiga di ruang embedding"], 1,
     "Panjang vektor sering mencerminkan panjang dokumen, dan kita tidak ingin itu memengaruhi kemiripan makna. Mengukur sudut membuat keduanya adil dibandingkan. Jarak Euclid tetap terdefinisi di dimensi berapa pun."),

    ("Query `puncak paling tinggi di Indonesia` menemukan dokumen `Gunung Kerinci adalah gunung berapi tertinggi di Indonesia`. Apa yang membuatnya mungkin?",
     ["Pencocokan kata dasar, karena `tinggi` dan `tertinggi` punya bentuk dasar yang sama setelah lemma",
      "Perbandingan vektor makna, jadi kemiripan terdeteksi meski kata yang dipakai hampir semua beda",
      "Perluasan query otomatis, karena model menambahkan sinonim ke query sebelum pencarian dijalankan",
      "Pembobotan TF-IDF, karena kata `Indonesia` muncul di keduanya dan bobotnya paling tinggi di korpus"], 1,
     "Pencarian dilakukan dengan membandingkan embedding, bukan mencocokkan kata. Karena itu dokumen tanpa kata yang sama pun bisa ditemukan. Tidak ada perluasan query, dan TF-IDF justru tidak akan menemukannya."),

    ("Apa yang ditambahkan RAG di Modul 05 dibanding semantic search yang dibangun di modul ini?",
     ["Model embedding yang jauh lebih besar, sehingga kemiripan makna terdeteksi dengan lebih akurat",
      "Indeks FAISS agar pencarian tetap cepat di data besar, dan LLM untuk menyusun jawabannya",
      "Pembersihan teks yang lebih ketat, karena dokumen harus bebas stopword sebelum bisa diindeks",
      "Pengukuran kemiripan dengan jarak Euclid, yang lebih tepat dipakai untuk pencarian dokumen"], 1,
     "Pola intinya sama: encode, bandingkan, ambil top-k. Yang bertambah adalah indeks agar pencarian tetap cepat untuk jutaan dokumen, dan LLM di ujung untuk menyusun jawaban. Ukuran model dan metrik jaraknya bukan pembeda utamanya."),

    ("Kenapa LLM memakai tokenisasi subword dan bukan tokenisasi per kata?",
     ["Karena subword membuat jumlah token satu kalimat menjadi jauh lebih sedikit daripada per kata",
      "Karena kosakata terbatas tetap bisa menulis kata apa pun, termasuk yang belum pernah dilihat",
      "Karena subword mempertahankan urutan kata, sedangkan tokenisasi per kata selalu mengabaikannya",
      "Karena subword secara otomatis mengembalikan tiap kata ke bentuk dasarnya seperti lemmatisasi"], 1,
     "Dengan sekitar 30–50 ribu subword, model bisa menyusun kata apa pun dari potongan yang dikenal — termasuk nama, istilah baru, dan salah ketik. Jumlah tokennya justru lebih banyak, bukan lebih sedikit."),

    ("Kalimat Indonesia menghasilkan lebih banyak token GPT-2 daripada kalimat Inggris yang semakna. Konsekuensi praktisnya apa?",
     ["Kualitas jawaban model untuk bahasa Indonesia pasti lebih rendah daripada untuk bahasa Inggris",
      "Biaya pemakaian API menjadi lebih mahal untuk isi yang sama, karena tagihannya dihitung per token",
      "Kalimat Indonesia harus dipotong lebih pendek, karena batas konteks model dihitung dalam kata",
      "Model perlu dilatih ulang dari awal sebelum bisa memproses kalimat berbahasa Indonesia sama sekali"], 1,
     "Tokenizer GPT-2 dilatih dominan pada teks Inggris sehingga kata Indonesia terpecah lebih halus. Karena API menagih per token, isi yang sama menjadi lebih mahal. Batas konteks juga dihitung dalam token, bukan kata."),

    # --- Notebook 02: NLP di GPU (4) ---
    ("Apa yang dilakukan `python -m cudf.pandas -m cuml.accel pipeline.py`?",
     ["Menerjemahkan berkas Python menjadi kode CUDA lebih dulu, lalu menjalankan hasil terjemahannya",
      "Mengaktifkan akselerator sebelum `import pandas` dijalankan, jadi operasinya dialihkan ke GPU",
      "Menyalin seluruh isi DataFrame ke memori GPU di awal, lalu menjalankan skripnya seperti biasa",
      "Menjalankan skrip dua kali, sekali di CPU dan sekali di GPU, lalu membandingkan kedua hasilnya"], 1,
     "Akselerator harus aktif sebelum pandas diimpor, karena ia mengambil alih modulnya. Operasi yang belum didukung GPU otomatis kembali ke CPU, sehingga kodenya tidak crash. Tidak ada penerjemahan ke CUDA maupun eksekusi ganda."),

    ("Hasil benchmark menunjukkan cuML lebih lambat daripada scikit-learn pada dataset kecil. Pemeriksaan pertama yang paling tepat?",
     ["Menaikkan ukuran batch, karena GPU baru menunjukkan keunggulannya pada batch yang besar",
      "Memastikan sel warm-up jalan, karena pemanggilan CUDA pertama membayar biaya inisialisasi",
      "Mengganti tipe data menjadi float64, karena cuML memerlukan presisi ganda agar bekerja optimal",
      "Menambah jumlah iterasi pelatihan, karena hasil GPU baru stabil setelah banyak epoch berjalan"], 1,
     "Pemanggilan CUDA pertama membayar inisialisasi context dan kompilasi kernel, dan itu bukan kecepatan komputasi sesungguhnya. Aturan emasnya: jangan pernah mengukur pemanggilan pertama. Data kecil memang bisa membuat GPU kalah, tetapi warm-up diperiksa lebih dulu."),

    ("Kenapa MinHash dipakai untuk mencari near-duplicate di korpus 150 ribu paragraf?",
     ["Karena MinHash menghitung kemiripan tiap pasangan dokumen jauh lebih cepat daripada cosine",
      "Karena tiap dokumen diringkas jadi sidik jari, sehingga pencarian berubah jadi pengelompokan",
      "Karena MinHash mengurutkan dokumen berdasarkan panjangnya, sehingga duplikat pasti berdekatan",
      "Karena MinHash membuang dokumen pendek lebih dulu, dan duplikat biasanya berupa teks pendek"], 1,
     "Membandingkan semua pasangan berarti sekitar 11 miliar perbandingan. MinHash mengubah masalah `cari yang mirip` menjadi `kelompokkan yang sama`, dan yang kedua jauh lebih murah. Ia tidak mempercepat perbandingan pasangan satu per satu."),

    ("Operasi apa dari `nvtext` yang tidak punya padanan langsung di pandas?",
     ["`str.lower()`, karena pandas belum menyediakan konversi huruf kecil untuk kolom bertipe teks",
      "`str.ngrams_tokenize()`, karena pembentukan n-gram di pandas harus disusun manual dengan map",
      "`str.split()`, karena pandas hanya bisa memecah string berdasarkan satu karakter pemisah saja",
      "`str.replace()`, karena penggantian pola di pandas tidak mendukung ekspresi reguler sama sekali"], 1,
     "nvtext menyediakan operasi teks yang memang lahir di GPU, termasuk pembentukan n-gram dan minhash. Di pandas, bigram harus disusun manual lewat zip dan map. Lower, split, dan replace semuanya sudah ada di pandas."),
]
# Sebar indeks jawaban supaya tidak selalu B: rotasi deterministik per soal.
ROT = [1, 0, 2, 3, 1, 2, 0, 3, 2, 1, 3, 0, 1, 2, 0, 3, 2, 1, 0, 3, 1, 2, 3, 0, 2, 1, 3, 0, 2, 1]
questions = []
for (q, opts, a, e), target in zip(Q, ROT):
    opts = list(opts)
    correct = opts.pop(a)
    opts.insert(target, correct)
    questions.append({"q": q, "code": None, "options": opts, "answer": target, "explanation": e})

payload = {"module": "03", "title": "NLP Fundamentals", "questions": questions}
N = len(questions)
assert N == 30, N
assert all(len(x["options"]) == 4 for x in questions), "every question needs 4 options"
assert all(0 <= x["answer"] <= 3 for x in questions)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKEL_HEAD = """<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NLP Fundamentals — Quiz</title>
<style>
:root{--bg:#1A1A2E;--card:#2D2D44;--green:#76B900;--lgreen:#A3D944;--white:#fff;
--gray:#AAAACC;--red:#EF5350;--dark:#23233A;}
*{box-sizing:border-box;}
body{font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
margin:0;background:var(--bg);color:var(--white);line-height:1.45;}
.wrap{max-width:820px;margin:0 auto;padding:18px 16px 60px;}
header{background:linear-gradient(135deg,var(--green),#5a8c00);border-radius:12px;
padding:16px 20px;margin-bottom:14px;}
header .mod{font-size:12px;letter-spacing:2px;text-transform:uppercase;opacity:.9;}
header h1{margin:2px 0 0;font-size:24px;}
header p{margin:4px 0 0;font-size:13px;opacity:.95;}
.scorebar{position:sticky;top:0;z-index:5;background:var(--dark);border:1px solid var(--card);
border-radius:10px;padding:8px 14px;margin-bottom:14px;display:flex;justify-content:space-between;
align-items:center;font-size:13px;}
.scorebar b{color:var(--lgreen);}
.q{background:var(--card);border-radius:10px;padding:14px 16px;margin-bottom:12px;
border-left:4px solid var(--green);}
.q .num{color:var(--green);font-weight:700;font-size:12px;}
.q .text{font-size:15px;font-weight:600;margin:4px 0 8px;}
.opt{display:block;width:100%;text-align:left;background:var(--dark);color:var(--white);
border:1.5px solid #3a3a55;border-radius:8px;padding:9px 12px;margin:6px 0;font-size:14px;
cursor:pointer;transition:.12s;}
.opt:hover:not(.locked){border-color:var(--green);}
.opt .lab{display:inline-block;width:20px;font-weight:700;color:var(--gray);}
.opt.correct{background:rgba(118,185,0,.18);border-color:var(--green);}
.opt.correct .lab{color:var(--green);}
.opt.wrong{background:rgba(239,83,80,.16);border-color:var(--red);}
.opt.wrong .lab{color:var(--red);}
.opt.locked{cursor:default;}
.expl{display:none;font-size:13px;color:var(--gray);background:var(--bg);
border-radius:6px;padding:8px 11px;margin-top:8px;border-left:3px solid var(--lgreen);}
.expl.show{display:block;}
.expl b{color:var(--lgreen);}
.summary{display:none;background:var(--card);border-radius:12px;padding:20px;text-align:center;
margin-top:8px;border:2px solid var(--green);}
.summary.show{display:block;}
.summary .big{font-size:34px;font-weight:800;color:var(--green);}
.summary .msg{font-size:15px;color:var(--gray);margin:6px 0 14px;}
.btn{background:var(--green);color:#0d0d0d;border:none;border-radius:8px;padding:10px 22px;
font-size:14px;font-weight:700;cursor:pointer;}
.btn:hover{background:var(--lgreen);}
footer{text-align:center;color:var(--gray);font-size:11px;margin-top:18px;opacity:.7;}
</style></head>
<body><div class="wrap">
<header><div class="mod">Modul 03 · Quiz Latihan</div><h1>NLP Fundamentals</h1>
<p>__N__ soal pilihan ganda · murni konsep · pilih satu jawaban</p></header>
<div class="scorebar"><span>Terjawab: <b id="answered">0</b> / __N__</span>
<span>Benar: <b id="correct">0</b></span></div>
<div id="quiz"></div>
<div class="summary" id="summary"><div class="big" id="finalscore"></div>
<div class="msg" id="finalmsg"></div><button class="btn" onclick="location.reload()">Ulangi Quiz</button></div>
<footer>Navasena Gen-ML Course · Modul 03 NLP Fundamentals — Quiz</footer>
</div>
<script>
const QUIZ = __JSON__;
window.QUIZ = QUIZ;
const LAB = ["A","B","C","D"];
let answered = 0, correct = 0;
const total = QUIZ.questions.length;
const quizEl = document.getElementById("quiz");
QUIZ.questions.forEach((item, qi) => {
  const card = document.createElement("div"); card.className = "q";
  const num = document.createElement("div"); num.className = "num"; num.textContent = "Soal " + (qi+1);
  card.appendChild(num);
  const text = document.createElement("div"); text.className = "text"; text.textContent = item.q;
  card.appendChild(text);
  if (item.code) { const pre = document.createElement("pre"); pre.textContent = item.code; card.appendChild(pre); }
  const expl = document.createElement("div"); expl.className = "expl";
  item.options.forEach((opt, oi) => {
    const b = document.createElement("button"); b.className = "opt";
    const lab = document.createElement("span"); lab.className = "lab"; lab.textContent = LAB[oi] + ".";
    b.appendChild(lab); b.appendChild(document.createTextNode(" " + opt));
    b.onclick = () => {
      if (card.dataset.locked) return;
      card.dataset.locked = "1";
      const opts = card.querySelectorAll(".opt");
      opts.forEach(o => o.classList.add("locked"));
      opts[item.answer].classList.add("correct");
      if (oi === item.answer) { correct++; } else { b.classList.add("wrong"); }
      answered++;
      document.getElementById("answered").textContent = answered;
      document.getElementById("correct").textContent = correct;
      expl.classList.add("show");
      if (answered === total) showSummary();
    };
    card.appendChild(b);
  });
  const eb = document.createElement("b"); eb.textContent = "Penjelasan: ";
  expl.appendChild(eb); expl.appendChild(document.createTextNode(item.explanation));
  card.appendChild(expl);
  quizEl.appendChild(card);
});
function showSummary(){
  const pct = Math.round(correct/total*100);
  const s = document.getElementById("summary"); s.classList.add("show");
  document.getElementById("finalscore").textContent = "Skor: " + correct + " / " + total + " (" + pct + "%)";
  let msg = pct>=80 ? "Bagus, pemahaman kamu solid." : pct>=60 ? "Lumayan. Ulas lagi materi yang masih salah." : "Pelajari lagi modulnya, lalu coba ulangi.";
  document.getElementById("finalmsg").textContent = msg;
  s.scrollIntoView({behavior:"smooth"});
}
</script></body></html>"""

html = SKEL_HEAD.replace("__N__", str(N)).replace("__JSON__", json.dumps(payload, ensure_ascii=False))
out = ROOT / "nlp-fundamentals-quiz.html"
out.write_text(html, encoding="utf-8")
# cek bias panjang opsi benar vs salah
import statistics
cor = [len(x["options"][x["answer"]]) for x in questions]
inc = [len(o) for x in questions for i, o in enumerate(x["options"]) if i != x["answer"]]
mc, mi = statistics.mean(cor), statistics.mean(inc)
print(f"wrote {out} with {N} questions; mean len correct={mc:.0f} incorrect={mi:.0f} ratio={mc/mi:.2f}")
from collections import Counter
print("answer index distribution:", dict(sorted(Counter(x["answer"] for x in questions).items())))
