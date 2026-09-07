#!/usr/bin/env python3
"""Build the Module 01 ML Fundamentals quiz HTML — 30 concept questions (code=null).
Sebaran: alur kerja & metrik 8, regresi & time series 5, klasifikasi 7, ensemble 5,
clustering 3, GPU 2. Mirrors 05_rag/tools/build_quiz.py.
Run: python build_quiz.py  ->  ../ml-fundamentals-quiz.html
"""
import json, pathlib

# (pertanyaan, [4 opsi], indeks jawaban, penjelasan). Opsi ditulis sepanjang ± sama supaya tidak ada bias panjang.
Q = [
    # --- Alur kerja & metrik (8) ---
    ("Rekanmu menyaring spam dengan daftar aturan if/else: 'ada kata GRATIS', 'banyak huruf kapital'. Apa yang membedakan pendekatan machine learning dari cara ini?",
     ["Aturannya sama persis, hanya ditulis ulang dalam Python supaya lebih rapi dan cepat",
      "Polanya dipelajari dari ribuan contoh email berlabel, bukan ditulis tangan satu per satu",
      "Tidak butuh data sama sekali, cukup definisi spam yang jelas dari pemilik produk",
      "Baru bisa dipakai kalau daftar aturannya sudah lengkap dan diuji oleh manusia dulu"], 1,
     "Inti ML: aturan dipelajari dari contoh berlabel, bukan ditulis satu per satu oleh programmer."),
    ("Seorang peserta mengukur akurasi model pada data yang sama dengan data train-nya, hasilnya 99%. Apa masalahnya?",
     ["Tidak ada masalah; 99% pada data mana pun sudah cukup untuk menyatakan model layak dipakai",
      "Angka itu tidak menunjukkan kemampuan pada data baru; model bisa saja menghafal data train",
      "Akurasi bukan metrik yang sah untuk klasifikasi, seharusnya selalu memakai R² atau MAE",
      "Data train seharusnya lebih kecil dari data test supaya angkanya tidak terlalu tinggi"], 1,
     "Skor di data train hanya menunjukkan seberapa baik model mengikuti data yang sudah dilihat. Generalisasi diukur di data test yang terpisah."),
    ("Kamu memprediksi apakah nasabah akan membuka deposito sebelum bank meneleponnya. Di data ada kolom 'lama panggilan'. Mengapa kolom ini berbahaya kalau ikut dipakai?",
     ["Satuannya detik, padahal kolom lain memakai menit, sehingga skalanya tidak sebanding",
      "Nilainya baru lengkap setelah panggilan berakhir, padahal prediksi dibutuhkan sebelum menelepon",
      "Kolom numerik tidak boleh dicampur dengan kolom kategori dalam satu model klasifikasi",
      "Kolom itu berkorelasi dengan umur nasabah sehingga membuat model menjadi bias usia"], 1,
     "Ini leakage: model memakai informasi yang belum tersedia saat prediksi dibutuhkan, sehingga evaluasinya terlalu optimistis."),
    ("Kapan scaling (menyamakan skala nilai feature) benar-benar mengubah hasil model?",
     ["Pada semua model tanpa kecuali, karena algoritma apa pun bekerja lebih baik dengan angka kecil",
      "Pada model berbasis jarak seperti KNN dan SVM; decision tree tidak terpengaruh skala",
      "Hanya pada decision tree, karena ambang pemisahnya dihitung dari nilai yang sudah diskalakan",
      "Tidak pernah; scaling hanya kebiasaan lama yang dipertahankan supaya kode terlihat rapi"], 1,
     "Model yang menghitung jarak atau kombinasi linear peka terhadap skala. Tree membandingkan nilai dengan ambang per feature, jadi skala tidak mengubah hasil."),
    ("Skor model di data train sangat tinggi, di data validation jauh lebih rendah. Kondisi apa ini dan langkah apa yang masuk akal?",
     ["Underfitting; tambah kompleksitas model atau tambah feature supaya polanya tertangkap",
      "Overfitting; kurangi kompleksitas model, tambah data, atau pakai regularisasi",
      "Leakage; buang data validation dan ukur ulang langsung di data test yang tersisa",
      "Kondisi normal; laporkan skor train karena itu yang mencerminkan kemampuan model"], 1,
     "Gap besar train vs validation adalah tanda overfitting: model mengikuti detail dan noise data train."),
    ("Kamu memilih kedalaman decision tree terbaik dari 1 sampai 15. Cara mana yang tidak mencemari penilaian akhir?",
     ["Coba semua kedalaman, pilih skor tertinggi di data test, lalu laporkan skor test yang sama",
      "Pilih dengan cross-validation di data train, lalu ukur sekali di data test di akhir",
      "Pilih kedalaman terbesar, karena pohon yang lebih dalam selalu menangkap pola lebih lengkap",
      "Pilih kedalaman 1 supaya cepat, karena kedalaman hampir tidak mempengaruhi hasil"], 1,
     "Kalau data test dipakai untuk memilih, skor test jadi terlalu optimistis. Cross-validation di data train menjaga test tetap 'baru'."),
    ("Model memprediksi 'tidak beli' untuk semua orang dan akurasinya 89%. Apa yang bisa disimpulkan?",
     ["Model sudah bagus karena akurasi di atas 80% umumnya dianggap layak untuk produksi",
      "Belum tentu bagus: 89% mungkin sama dengan baseline kelas mayoritas, model belum menambah apa pun",
      "Model perlu lebih banyak feature numerik supaya akurasinya bisa naik melewati 90%",
      "Model terlalu rumit dan perlu disederhanakan supaya tidak menghafal data train-nya"], 1,
     "Baseline kelas mayoritas adalah pembanding minimal. Kalau model tidak melampauinya, akurasi tinggi itu kosong."),
    ("Nilai R² sebuah model regresi adalah 0. Apa artinya?",
     ["Model sempurna, karena error kuadratnya nol pada semua baris data evaluasi",
      "Model tidak lebih baik daripada selalu memprediksi rata-rata target di data evaluasi",
      "Semua prediksi model bernilai nol karena koefisiennya belum dilatih dengan benar",
      "Target di data evaluasi tidak punya variasi sehingga metriknya tidak bisa dihitung"], 1,
     "R² membandingkan error model dengan error prediksi rata-rata: 1 = prediksi tepat di data itu, 0 = sama dengan prediksi rata-rata, negatif = lebih buruk."),
    # --- Regresi & time series (5) ---
    ("Pada model harga rumah ŷ = 2,5·luas + 50, apa arti angka 2,5?",
     ["Harga rumah paling murah yang ada di data train setelah dibagi seratus",
      "Setiap tambahan satu satuan luas menaikkan prediksi harga sebesar 2,5",
      "Jumlah rumah rata-rata per kelurahan yang dipakai untuk melatih model",
      "Error rata-rata prediksi model pada data test dalam satuan harga"], 1,
     "Koefisien (kemiringan) menyatakan perubahan prediksi per satu satuan feature. 50 adalah intercept, prediksi saat luas = 0."),
    ("Model A dan B punya MAE sama, tetapi RMSE model A jauh lebih besar. Apa yang bisa disimpulkan tentang model A?",
     ["Model A lebih akurat karena RMSE yang besar menandakan model menangkap variasi data",
      "Model A punya beberapa error sangat besar, karena RMSE menghukum error besar lewat kuadrat",
      "Model A memakai lebih banyak feature sehingga metriknya dihitung dengan cara berbeda",
      "Kedua model identik; perbedaan RMSE hanya efek pembulatan saat mencetak hasil"], 1,
     "RMSE mengkuadratkan error sebelum dirata-rata, sehingga beberapa kesalahan besar membuatnya melonjak meski MAE sama."),
    ("Setelah Lasso (regularisasi L1) dipakai, koefisien kolom 'iklan koran' menjadi tepat nol. Apa maknanya?",
     ["Kolom itu berisi nilai nol di semua baris sehingga tidak mungkin punya koefisien",
      "Lasso menilai kolom itu tidak cukup membantu dibanding penaltinya, jadi dibuang dari model",
      "Model gagal dilatih karena alpha terlalu kecil dan perlu dilatih ulang dari awal",
      "Kolom itu paling penting, dan nol menandakan model sudah yakin sepenuhnya padanya"], 1,
     "Penalti L1 mendorong koefisien yang kontribusinya kecil ke nol, sehingga Lasso sekaligus menyeleksi feature."),
    ("Kamu membagi data penjualan bulanan untuk evaluasi model time series. Cara split mana yang benar?",
     ["Acak 80/20 seperti data tabular biasa supaya distribusi train dan test seimbang",
      "Kronologis: latih di bulan-bulan awal, uji di 12 bulan terakhir yang belum dilihat",
      "Uji di bulan-bulan awal dan latih di bulan-bulan akhir supaya model belajar tren terbaru",
      "Tidak perlu split; model time series dievaluasi cukup dari seberapa halus kurvanya"], 1,
     "Urutan waktu penting. Mengacak data membuat model 'melihat masa depan' saat dilatih."),
    ("Model SARIMA kamu punya MAPE 6%. Sebelum bangga, pembanding apa yang paling wajar dicek?",
     ["Akurasi model klasifikasi pada data yang sama supaya ada angka kedua untuk dibandingkan",
      "Baseline seasonal naive: tebak bulan ini sama dengan bulan yang sama tahun lalu",
      "R² dari linear regression sederhana terhadap nomor urut bulan sebagai satu-satunya feature",
      "Tidak perlu pembanding; MAPE di bawah 10% sudah pasti berarti model layak dipakai"], 1,
     "Untuk data musiman, seasonal naive sering sudah cukup bagus. Model layak kalau mengalahkannya."),
    # --- Klasifikasi (7) ---
    ("Logistic regression memberi probabilitas 0,62 untuk seorang nasabah. Dengan threshold 0,5 prediksinya apa, dan apa yang terjadi kalau threshold dinaikkan ke 0,7?",
     ["Ya; tetap Ya, karena probabilitas 0,62 tidak berubah saat threshold digeser",
      "Ya; menjadi Tidak, karena 0,62 berada di bawah threshold baru 0,7",
      "Tidak; menjadi Ya, karena threshold yang lebih tinggi membuat model lebih longgar",
      "Tidak bisa ditentukan tanpa melihat akurasi model pada seluruh data test"], 1,
     "Threshold mengubah probabilitas menjadi kelas. Menaikkan threshold membuat model lebih jarang bilang 'Ya': precision naik, recall turun."),
    ("Di sistem deteksi penipuan kartu kredit, kesalahan mana yang biasanya paling mahal, dan metrik mana yang menyorotnya?",
     ["False positive, karena transaksi sah yang tertahan membuat nasabah kesal; metriknya precision",
      "False negative, karena penipuan yang lolos adalah kerugian langsung; metriknya recall",
      "True negative, karena transaksi normal yang terlalu banyak menurunkan akurasi",
      "True positive, karena setiap penipuan yang tertangkap harus diverifikasi manual; metriknya F1"], 1,
     "Penipuan yang tidak tertangkap (false negative) adalah kerugian langsung. Recall mengukur berapa banyak kasus positif yang tertangkap."),
    ("Mengapa F1 memakai rata-rata harmonik precision dan recall, bukan rata-rata biasa?",
     ["Karena rata-rata harmonik lebih mudah dihitung oleh library dan hasilnya selalu lebih besar",
      "Karena rata-rata harmonik menghukum ketidakseimbangan: precision tinggi tidak menutupi recall rendah",
      "Karena precision dan recall selalu bernilai sama sehingga jenis rata-ratanya tidak penting",
      "Karena rata-rata biasa hanya berlaku untuk metrik regresi seperti MAE dan RMSE"], 1,
     "Rata-rata harmonik condong ke nilai yang lebih kecil, jadi F1 tinggi hanya kalau keduanya tinggi."),
    ("Pada plot dua feature (umur dan gaji), batas keputusan sebuah model hanya terdiri dari garis horizontal dan vertikal. Model apa yang paling mungkin?",
     ["KNN, karena tetangga terdekat dikelompokkan dalam kotak-kotak berukuran sama",
      "Decision tree, karena tiap simpul membandingkan satu feature dengan satu ambang",
      "SVM kernel RBF, karena kernel membuat batas tegak lurus pada tiap sumbu feature",
      "Logistic regression, karena koefisiennya memisahkan tiap feature secara terpisah"], 1,
     "Setiap simpul decision tree memecah data berdasarkan satu feature, sehingga batasnya sejajar sumbu feature."),
    ("KNN dengan K=1: akurasi train 100%, akurasi CV rendah. KNN dengan K=99: akurasi train dan CV sama-sama rendah. Apa yang terjadi?",
     ["K=1 underfit karena terlalu sedikit tetangga; K=99 overfit karena terlalu banyak yang dihitung",
      "K=1 overfit karena mengikuti tiap titik; K=99 underfit karena batasnya terlalu rata",
      "Keduanya overfit; jumlah tetangga tidak mempengaruhi kompleksitas batas keputusan",
      "Keduanya sudah optimal; selisih train dan CV adalah hal wajar untuk model berbasis jarak"], 1,
     "K kecil membuat prediksi mengikuti noise; K besar merata-ratakan terlalu banyak tetangga sehingga pola lokal hilang."),
    ("Kamu memakai KNN pada kolom umur (18–60) dan gaji (15.000–150.000) tanpa scaling. Apa akibatnya?",
     ["Tidak ada; KNN menghitung jarak per feature secara terpisah lalu memberi suara sama rata",
      "Jarak didominasi selisih gaji; umur hampir tidak berpengaruh pada tetangga yang terpilih",
      "Umur mendominasi karena kolom pertama selalu diberi bobot lebih besar oleh algoritma",
      "Model gagal dilatih karena KNN menolak feature dengan rentang lebih dari sepuluh ribu"], 1,
     "Jarak Euclidean menjumlahkan selisih kuadrat tiap feature. Feature dengan rentang besar menenggelamkan yang lain tanpa scaling."),
    ("Dalam SVM, titik data mana yang menentukan posisi batas keputusan?",
     ["Semua titik dengan bobot yang sama, karena SVM meminimalkan total error seperti regresi",
      "Support vector: titik yang berada tepat di margin atau melewatinya",
      "Titik yang paling jauh dari batas, karena mereka menentukan lebar margin maksimum",
      "Titik dengan nilai feature terbesar, karena mereka mendominasi perhitungan kernel"], 1,
     "Batas SVM ditentukan oleh titik-titik terdekat ke margin. Titik lain bisa dipindah tanpa mengubah garisnya."),
    # --- Ensemble (5) ---
    ("Kamu menggabungkan tiga model lewat voting, tetapi ketiganya logistic regression dengan feature yang sama. Mengapa voting hampir tidak membantu?",
     ["Voting hanya bekerja untuk decision tree, karena model lain tidak menghasilkan suara diskrit",
      "Ketiga model membuat kesalahan yang sama, jadi suara mereka tidak saling mengoreksi",
      "Voting membutuhkan minimal lima model supaya suara mayoritas bisa dihitung dengan sah",
      "Logistic regression tidak punya probabilitas sehingga soft voting tidak bisa dijalankan"], 1,
     "Ensemble membantu kalau kesalahan anggotanya tidak seragam. Model yang mirip membuat kesalahan yang mirip pula."),
    ("Dalam bootstrap yang dipakai bagging, sampel untuk tiap model diambil dengan cara apa?",
     ["Mengambil separuh data secara berurutan dari atas supaya tiap model melihat bagian berbeda",
      "Mengambil sampel acak dengan pengembalian, sehingga satu baris bisa terpilih lebih dari sekali",
      "Membuat data sintetis baru di antara titik-titik yang ada supaya jumlah datanya bertambah",
      "Mengurutkan data berdasarkan target lalu memotongnya menjadi beberapa bagian sama besar"], 1,
     "Bootstrap = sampling dengan pengembalian dari data yang ada. Ada baris yang muncul berulang dan ada yang tidak terambil."),
    ("Apa yang membedakan boosting dari bagging?",
     ["Boosting melatih model paralel dari sampel acak, bagging melatih model berurutan satu per satu",
      "Boosting melatih berurutan, tiap model fokus memperbaiki kesalahan sebelumnya; bagging paralel",
      "Tidak ada bedanya; keduanya nama lain untuk Random Forest dengan jumlah pohon berbeda",
      "Boosting hanya untuk regresi, sedangkan bagging hanya untuk klasifikasi biner"], 1,
     "Bagging mengurangi variance dengan merata-ratakan model independen; boosting mengurangi bias dengan memperbaiki kesalahan bertahap."),
    ("Saat melatih XGBoost dengan early stopping, AUC validation berhenti membaik di iterasi 180 dari 1000. Apa tindakan yang tepat?",
     ["Paksa sampai 1000 iterasi, karena lebih banyak pohon selalu menghasilkan model lebih akurat",
      "Berhenti di sekitar iterasi 180; iterasi berikutnya cenderung overfitting atau sia-sia",
      "Ulangi dengan learning rate lebih besar tanpa data validation supaya lebih cepat selesai",
      "Hapus data validation dan pakai seluruh data untuk melatih supaya AUC-nya naik lagi"], 1,
     "Early stopping memakai data validation untuk menemukan titik berhenti sebelum model mulai menghafal data train."),
    ("Feature importance XGBoost menunjukkan 'status menikah' sebagai kontributor terbesar untuk prediksi penghasilan. Kesimpulan mana yang tepat?",
     ["Menikah menyebabkan penghasilan naik, karena model sudah mengukur pengaruhnya secara langsung",
      "Feature itu paling banyak dipakai model untuk memisahkan kelas; ini bukan hubungan sebab-akibat",
      "Feature lain tidak berguna dan sebaiknya dibuang supaya model lebih sederhana dan cepat",
      "Model salah karena hasilnya tidak intuitif; feature importance seharusnya menyorot pendidikan"], 1,
     "Feature importance mengukur kontribusi dalam membangun pohon, bukan kausalitas."),
    # --- Clustering (3) ---
    ("Dalam satu iterasi k-means, apa yang berubah dan apa yang tetap?",
     ["Titik data bergerak mendekati pusat terdekat, sedangkan posisi pusat cluster tetap",
      "Titik data diam; keanggotaan cluster dan posisi pusat yang diperbarui",
      "Semuanya tetap sampai iterasi terakhir, lalu pusat dan titik dipindah bersamaan",
      "Jumlah titik data berubah karena outlier dibuang di setiap iterasi"], 1,
     "K-means menugaskan tiap titik ke pusat terdekat, memindahkan pusat ke rata-rata anggotanya, dan mengulang sampai stabil."),
    ("Kamu mengelompokkan pelanggan grosir memakai kolom belanja plus kolom 'Channel' (1 = restoran, 2 = retail) yang ikut di-scaling. Silhouette tertinggi di K=2 dan clusternya hampir sama dengan Channel. Apa yang sebenarnya terjadi?",
     ["Clustering menemukan segmen belanja baru yang kebetulan sejalan dengan jenis pelanggan",
      "Kolom kategori ikut dihitung sebagai jarak, sehingga cluster hanya meniru kolom Channel",
      "K=2 memang selalu paling tepat untuk data pelanggan karena hanya ada dua jenis toko",
      "Scaling membuat semua kolom setara, jadi hasilnya valid dan bisa langsung dilaporkan"], 1,
     "Kolom kategori bukan jumlah belanja; setelah scaling ia memisahkan data dengan tajam. Pakai kolom belanja saja dan simpan Channel untuk mengecek hasil."),
    ("Data belanja pelanggan sangat miring: beberapa pelanggan belanja puluhan kali lipat dari yang lain. Sebelum k-means, langkah apa yang membantu?",
     ["Buang semua pelanggan besar supaya sisa datanya seragam dan cluster jadi rapi",
      "Transformasi log (log1p) lalu scaling, supaya jarak tidak didominasi pelanggan raksasa",
      "Kalikan semua nilai dengan 100 supaya selisih antar pelanggan biasa terlihat lebih jelas",
      "Ubah tiap kolom belanja menjadi kategori ya/tidak supaya k-means bekerja seperti tree"], 1,
     "Log menekan rentang nilai ekstrem sehingga jarak antar pelanggan biasa tetap bermakna."),
    # --- GPU (2) ---
    ("Kamu membandingkan cuML dan scikit-learn pada 20.000 baris data. GPU ternyata lebih lambat. Penjelasan mana yang paling masuk akal?",
     ["GPU-nya bermasalah, karena GPU seharusnya selalu lebih cepat berapa pun ukuran datanya",
      "Data terlalu kecil: overhead memindahkan data dan menyiapkan kernel GPU melebihi kerja hitungnya",
      "cuML memakai algoritma yang berbeda dari scikit-learn sehingga hasilnya tidak bisa dibandingkan",
      "GPU hanya berguna untuk deep learning, bukan untuk algoritma machine learning klasik"], 1,
     "GPU unggul pada operasi paralel berukuran besar. Pada data kecil, overhead transfer dan warm-up mendominasi."),
    ("Setelah melatih Random Forest di CPU (sklearn) dan GPU (cuML), akurasinya beda 0,001. Apa yang bisa disimpulkan?",
     ["GPU membuat model lebih pintar karena bisa memeriksa lebih banyak kombinasi pemisah",
      "GPU hanya mempercepat komputasi; selisih sekecil itu wajar dari detail perhitungan floating point",
      "Salah satu model pasti rusak, karena algoritma yang sama harus memberi akurasi identik",
      "cuML memakai data train yang berbeda karena bootstrap di GPU tidak bisa diberi random_state"], 1,
     "Akselerasi GPU mengubah kecepatan, bukan kemampuan model. Selisih kecil berasal dari detail numerik."),
]

# Sebar indeks jawaban supaya tidak selalu B: rotasi deterministik per soal.
ROT = [1, 0, 2, 3, 1, 2, 0, 3, 2, 1, 3, 0, 1, 2, 0, 3, 2, 1, 0, 3, 1, 2, 3, 0, 2, 1, 3, 0, 2, 1]
questions = []
for (q, opts, a, e), target in zip(Q, ROT):
    opts = list(opts)
    correct = opts.pop(a)
    opts.insert(target, correct)
    questions.append({"q": q, "code": None, "options": opts, "answer": target, "explanation": e})

payload = {"module": "01", "title": "Machine Learning Fundamentals", "questions": questions}
N = len(questions)
assert N == 30, N
assert all(len(x["options"]) == 4 for x in questions), "every question needs 4 options"
assert all(0 <= x["answer"] <= 3 for x in questions)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKEL_HEAD = """<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Machine Learning Fundamentals — Quiz</title>
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
<header><div class="mod">Module 01 · Quiz Latihan</div><h1>Machine Learning Fundamentals</h1>
<p>__N__ soal pilihan ganda · murni konsep · pilih satu jawaban</p></header>
<div class="scorebar"><span>Terjawab: <b id="answered">0</b> / __N__</span>
<span>Benar: <b id="correct">0</b></span></div>
<div id="quiz"></div>
<div class="summary" id="summary"><div class="big" id="finalscore"></div>
<div class="msg" id="finalmsg"></div><button class="btn" onclick="location.reload()">Ulangi Quiz</button></div>
<footer>Navasena Gen-ML Course · Module 01 Machine Learning Fundamentals — Quiz</footer>
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
out = ROOT / "ml-fundamentals-quiz.html"
out.write_text(html, encoding="utf-8")
# cek bias panjang opsi benar vs salah
import statistics
cor = [len(x["options"][x["answer"]]) for x in questions]
inc = [len(o) for x in questions for i, o in enumerate(x["options"]) if i != x["answer"]]
mc, mi = statistics.mean(cor), statistics.mean(inc)
print(f"wrote {out} with {N} questions; mean len correct={mc:.0f} incorrect={mi:.0f} ratio={mc/mi:.2f}")
from collections import Counter
print("answer index distribution:", dict(sorted(Counter(x["answer"] for x in questions).items())))
