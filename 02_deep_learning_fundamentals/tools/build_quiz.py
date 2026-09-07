#!/usr/bin/env python3
"""Build the Modul 02 DL Fundamentals quiz HTML — 30 concept questions (code=null).
Sebaran: belajar & evaluasi 8, aktivasi/optimizer 4, CNN/transfer 6, RNN/LSTM 4, GPU/TensorRT 4, generatif 4. Mirrors 05_rag/tools/build_quiz.py.
Run: python build_quiz.py  ->  ../dl-fundamentals-quiz.html
"""
import json, pathlib

# (pertanyaan, [4 opsi], indeks jawaban, penjelasan). Opsi ditulis sepanjang ± sama.
Q = [
    # --- Belajar & evaluasi (8) ---
    ("Loss training turun terus sampai epoch 30, tetapi loss validation mulai naik sejak epoch 8. Apa yang terjadi dan apa langkah yang wajar?",
     ["Underfitting; tambah layer dan latih sampai epoch 60 agar loss validation ikut turun",
      "Ada indikasi overfitting; pakai early stopping pada loss validation, pulihkan bobot terbaik",
      "Learning rate terlalu kecil; naikkan sepuluh kali lipat agar keduanya turun bersama",
      "Data validation rusak; ganti dengan data test supaya kurva validationnya halus"], 1,
     "Loss train terus membaik sementara loss validation memburuk: ini indikasi overfitting. Early stopping dapat membatasi training dan memulihkan bobot terbaik jika dikonfigurasi demikian (di Keras: restore_best_weights=True). Dropout dapat dicoba sebagai regularisasi, tetapi hasilnya perlu dievaluasi ulang. Data test tetap untuk evaluasi akhir."),
    ("Saat gradient descent, loss melonjak naik-turun dan akhirnya menjadi NaN. Penyebab paling mungkin?",
     ["Batch size terlalu kecil sehingga gradient menjadi terlalu halus dan tidak bergerak",
      "Learning rate terlalu besar: langkahnya melompati dasar lembah dan meledak",
      "Terlalu banyak epoch, model kehabisan pola baru yang bisa dipelajari dari data",
      "Fungsi aktivasi ReLU mematikan semua neuron sehingga loss tidak terdefinisi"], 1,
     "Learning rate yang terlalu besar membuat langkah melewati minimum dan makin jauh; loss meledak. Solusinya turunkan learning rate."),
    ("Kamu melatih Fashion-MNIST dengan 60.000 gambar dan batch size 64. Apa arti 'satu epoch'?",
     ["Satu kali pembaruan bobot memakai 64 gambar yang dipilih acak dari data train",
      "Satu putaran penuh melewati seluruh 60.000 gambar train, sekitar 938 pembaruan bobot",
      "Satu kali evaluasi model pada data validation setelah bobot diperbarui",
      "Satu detik waktu training, berapa pun jumlah gambar yang sempat diproses"], 1,
     "Epoch = seluruh data train dilewati sekali; dengan batch 64 itu berarti 60.000/64 ≈ 938 langkah pembaruan."),
    ("Kamu membandingkan tiga arsitektur (kecil, sedang, besar) dan mau memilih satu untuk dilaporkan. Cara mana yang tidak mencemari angka akhir?",
     ["Latih ketiganya, ukur di data test, pilih yang tertinggi, laporkan angka test itu",
      "Latih ketiganya, bandingkan di data validation, ukur pemenangnya sekali di data test",
      "Pilih yang paling besar karena kapasitas lebih tinggi hampir pasti lebih akurat",
      "Pilih yang paling kecil supaya training cepat, lalu laporkan akurasi trainnya"], 1,
     "Test set hanya dipakai sekali untuk model yang sudah dipilih; kalau dipakai memilih, skornya terlalu optimistis."),
    ("Mengapa pixel gambar dibagi 255 sebelum masuk ke neural network?",
     ["Supaya file gambar menjadi lebih kecil dan proses membaca data dari disk lebih cepat",
      "Supaya masukan berada di rentang 0–1 dan gradient descent lebih stabil",
      "Karena Keras menolak masukan bertipe integer dan hanya menerima nilai di bawah satu",
      "Supaya gambar menjadi hitam-putih dan jumlah kelas keluaran ikut berkurang"], 1,
     "Normalisasi input menyamakan skala masukan sehingga bobot awal dan learning rate bekerja dengan baik."),
    ("Apa yang sebenarnya dilakukan backpropagation?",
     ["Menjalankan data dari belakang ke depan supaya model membaca gambar secara terbalik",
      "Menghitung gradient loss terhadap tiap bobot, dari output ke layer sebelumnya",
      "Menghapus neuron yang salah memprediksi dan menggantinya dengan neuron baru yang acak",
      "Mengulang epoch terakhir dengan learning rate lebih kecil sampai loss berhenti berubah"], 1,
     "Backprop menghitung gradient loss terhadap tiap bobot dengan menelusuri perhitungan dari output ke layer sebelumnya; optimizer yang memperbarui bobotnya."),
    ("Model 10 kelas mencapai akurasi test 87%. Confusion matrix menunjukkan Coat dan Pullover paling sering saling tertukar. Kesimpulan mana yang tepat?",
     ["Model rusak karena akurasi di bawah 90% tidak layak untuk klasifikasi gambar",
      "Kesalahan menumpuk di dua kelas yang memang mirip; akurasi total menyembunyikannya",
      "Kelas Coat harus dihapus dari data supaya akurasi keseluruhan naik ke angka sempurna",
      "Confusion matrix tidak relevan untuk 10 kelas dan hanya berlaku untuk klasifikasi biner"], 1,
     "Confusion matrix menunjukkan di mana model salah; kelas yang secara visual mirip wajar tertukar dan itu informasi untuk perbaikan."),
    ("Dropout 0,5 dipasang di layer tersembunyi. Apa yang terjadi saat training dan saat prediksi?",
     ["Saat training sebagian aktivasi ditolkan secara acak; saat prediksi semuanya aktif",
      "Saat training semua neuron aktif; saat prediksi separuh neuron dimatikan supaya lebih cepat",
      "Separuh data train dibuang permanen supaya model tidak menghafal contoh yang sama",
      "Separuh bobot dibulatkan ke nol setelah training selesai agar model lebih kecil"], 0,
     "Dropout hanya aktif saat training: sebagian aktivasi ditolkan secara acak agar model tidak bergantung pada jalur tertentu. Saat inferensi semua neuron dipakai."),
    # --- Aktivasi & optimizer (4) ---
    ("Kamu menumpuk lima layer Dense tanpa fungsi aktivasi. Apa kemampuan model ini?",
     ["Setara dengan satu layer linear: tumpukan transformasi linear tetap linear",
      "Lima kali lebih kuat dari satu layer karena tiap layer menambah pola baru",
      "Bisa memisahkan pola apa pun asalkan jumlah neuron per layer cukup besar",
      "Tidak bisa dilatih sama sekali karena gradient descent membutuhkan fungsi aktivasi"], 0,
     "Tanpa nonlinearitas, komposisi layer linear tetap linear. Aktivasi seperti ReLU yang memberi kemampuan memodelkan pola melengkung."),
    ("Network 20 layer dengan aktivasi sigmoid di semua layer belajar sangat lambat; bobot layer awal hampir tidak berubah. Apa penyebabnya?",
     ["Vanishing gradient: turunan sigmoid ≤ 0,25, dikalikan berulang jadi hampir nol",
      "Exploding gradient: sigmoid memperbesar gradient tiap layer sampai bobot awal tak terhingga",
      "Data terlalu kecil untuk 20 layer, jadi bobot awal tidak pernah menerima contoh yang cukup",
      "Sigmoid hanya bekerja pada layer keluaran, sehingga layer tengah diabaikan oleh optimizer"], 0,
     "Turunan sigmoid kecil; perkalian berantai lewat banyak layer membuat gradient di layer awal menyusut. ReLU mengurangi masalah ini."),
    ("Layer keluaran untuk klasifikasi 10 kelas memakai softmax. Apa yang dijamin softmax pada keluarannya?",
     ["Sepuluh nilai 0–1 yang berjumlah tepat 1, bisa dibaca sebagai probabilitas kelas",
      "Sepuluh nilai yang bisa negatif atau positif, mencerminkan skor mentah tiap kelas",
      "Satu nilai antara 0 dan 1 yang menyatakan peluang gambar termasuk kelas pertama",
      "Sepuluh nilai biner 0 atau 1, satu untuk tiap kelas, tanpa informasi keyakinan"], 0,
     "Softmax menormalkan skor menjadi distribusi probabilitas atas semua kelas; kelas dengan nilai terbesar menjadi prediksi."),
    ("Apa perbedaan praktis Adam dibanding SGD biasa?",
     ["Adam menyesuaikan besar langkah tiap parameter dari riwayat gradient, konvergen lebih cepat",
      "Adam memakai learning rate lebih besar sehingga selalu mencapai loss paling rendah",
      "Adam mengubah arsitektur model dengan menambah neuron secara otomatis saat loss stagnan",
      "Adam hanya berfungsi untuk data gambar, sedangkan SGD untuk data tabular dan teks"], 0,
     "Adam mengombinasikan momentum dan penyesuaian skala per parameter; itu membuatnya andal sebagai pilihan awal, bukan jaminan hasil terbaik."),
    # --- CNN & transfer learning (6) ---
    ("Mengapa CNN memakai filter kecil (misalnya 3×3) yang digeser ke seluruh gambar, bukan Dense yang terhubung ke semua pixel?",
     ["Filter yang sama dipakai di banyak posisi: pola lokal dicari dengan parameter lebih sedikit",
      "Karena gambar tidak bisa diratakan menjadi vektor, jadi Dense tidak bisa menerima pixel",
      "Karena filter 3×3 melihat seluruh gambar sekaligus sehingga konteks globalnya lengkap",
      "Karena Dense hanya bekerja pada gambar hitam-putih, sedangkan konvolusi mendukung warna"], 0,
     "Konvolusi memakai bobot yang sama pada setiap posisi dan mengolah wilayah lokal. Karena itu, model tidak perlu mempelajari bobot terpisah untuk setiap lokasi gambar."),
    ("Apa yang ditunjukkan sebuah feature map di layer konvolusi pertama?",
     ["Peta respons satu filter di tiap posisi gambar, misalnya tepi vertikal atau horizontal",
      "Probabilitas tiap kelas untuk gambar itu sebelum melewati layer softmax di akhir",
      "Salinan gambar asli yang sudah dinormalisasi dan diperkecil ke ukuran 3×3 pixel",
      "Daftar bobot layer Dense terakhir yang diurutkan dari yang paling berpengaruh"], 0,
     "Feature map adalah keluaran satu filter; di layer awal biasanya berupa detektor tepi, tekstur, atau warna sederhana."),
    ("Apa fungsi pooling (misalnya max pooling 2×2) di CNN?",
     ["Mengecilkan feature map dan membuat deteksi tahan pergeseran kecil, ambil nilai terbesar tiap blok",
      "Menggandakan ukuran feature map supaya detail halus gambar tidak hilang di layer berikutnya",
      "Mengubah gambar berwarna menjadi hitam-putih supaya jumlah channel berkurang menjadi satu",
      "Menormalkan nilai pixel ke rentang 0–1 sebelum masuk ke layer konvolusi berikutnya"], 0,
     "Pooling merangkum blok kecil menjadi satu nilai; ukuran turun, komputasi hemat, dan posisi persis feature menjadi kurang penting."),
    ("Kamu punya 5.000 foto produk dan ingin classifier gambar yang bagus dalam satu sore. Pendekatan mana yang paling masuk akal?",
     ["Transfer learning: backbone pre-trained ImageNet dibekukan, latih layer atas dengan datamu",
      "Latih CNN dari nol dengan 50 layer, karena arsitektur dalam pasti mengalahkan model pre-trained",
      "Pakai Dense besar tanpa konvolusi supaya semua pixel diperhitungkan tanpa kehilangan detail",
      "Kumpulkan dulu satu juta foto tambahan, karena CNN tidak bisa belajar dari data sebanyak itu"], 0,
     "Dengan data terbatas, feature umum dari ImageNet (tepi, tekstur, bentuk) bisa dipakai ulang; hanya kepala classifier yang dilatih."),
    ("ResNet50 pre-trained ImageNet dipakai untuk CIFAR-10 berukuran 32×32 dan hasilnya tidak sebaik yang diharapkan. Penjelasan mana yang paling masuk akal?",
     ["Feature ImageNet dipelajari dari gambar ±224×224; di 32×32 detailnya hampir tak ada",
      "ResNet50 hanya bisa mengenali 1.000 kelas ImageNet dan menolak dataset dengan 10 kelas",
      "Transfer learning selalu lebih buruk dari CNN dari nol pada dataset yang lebih kecil",
      "CIFAR-10 adalah gambar hitam-putih sehingga tiga channel warna ResNet tidak terpakai"], 0,
     "Kecocokan resolusi dan domain menentukan manfaat transfer learning; gambar sangat kecil membuat feature pre-trained kurang relevan."),
    ("Augmentasi seperti RandomFlip dan RandomRotation dipasang sebagai layer di dalam model. Apa efeknya saat training dan saat prediksi?",
     ["Saat training tiap batch melihat variasi acak; saat prediksi layer ini tidak mengubah gambar",
      "Saat training gambar tetap asli; saat prediksi gambar diputar acak supaya ujiannya berat",
      "Jumlah gambar di disk bertambah dua kali lipat karena setiap gambar disimpan versi terbaliknya",
      "Model menjadi lebih kecil karena gambar yang mirip digabungkan menjadi satu contoh"], 0,
     "Layer augmentasi Keras aktif hanya saat training, menghasilkan variasi on-the-fly tanpa menambah file; saat inferensi ia pass-through."),
    # --- RNN/LSTM (4) ---
    ("Apa yang dibawa 'hidden state' pada RNN dari satu langkah waktu ke langkah berikutnya?",
     ["Ringkasan masukan sebelumnya, sehingga prediksi kini bergantung pada konteks yang lewat",
      "Salinan persis seluruh masukan sebelumnya, sehingga memorinya tumbuh seiring panjang urutan",
      "Label jawaban benar dari langkah sebelumnya, supaya model bisa mengoreksi dirinya sendiri",
      "Learning rate yang menyesuaikan diri di tiap langkah waktu berdasarkan error terakhir"], 0,
     "Hidden state adalah memori ringkas berukuran tetap yang diperbarui tiap langkah; itu yang membedakan RNN dari Dense biasa."),
    ("SimpleRNN gagal menangkap hubungan antara kata pertama dan kata ke-80 dalam sebuah ulasan. Mengapa, dan apa yang membantu?",
     ["Gradien menyusut lewat puluhan langkah (vanishing); gerbang LSTM/GRU menjaga info lebih lama",
      "Ulasan terlalu pendek untuk RNN; tambahkan padding sampai 1.000 token supaya konteksnya cukup",
      "SimpleRNN hanya membaca kata terakhir; ganti dengan Dense yang membaca semua kata sekaligus",
      "Learning rate terlalu besar; kecilkan sepuluh kali supaya kata pertama tidak terlupakan"], 0,
     "Di urutan panjang gradient RNN biasa menghilang; gerbang LSTM/GRU mengatur apa yang disimpan dan dilupakan sehingga konteks jauh bisa bertahan."),
    ("Apa peran layer Embedding pada model teks?",
     ["Mengubah indeks kata menjadi vektor padat yang dipelajari model saat training",
      "Menghapus kata umum seperti 'dan' dan 'yang' supaya model fokus pada kata penting",
      "Menerjemahkan ulasan ke bahasa Inggris lebih dulu sebelum diproses LSTM",
      "Mengurutkan kata berdasarkan frekuensinya lalu membuang yang jarang muncul"], 0,
     "Embedding memetakan tiap token ke vektor kontinu yang ikut dilatih; itu representasi masukan untuk RNN/LSTM."),
    ("Untuk memprediksi nilai deret waktu berikutnya, baseline paling sederhana yang wajar dibandingkan dengan RNN adalah?",
     ["Memprediksi bahwa nilai berikutnya sama dengan nilai terakhir yang diketahui",
      "Memprediksi angka nol untuk semua langkah supaya errornya mudah dihitung",
      "Memakai rata-rata seluruh deret termasuk data masa depan yang belum terjadi",
      "Menjalankan LSTM dengan satu neuron saja lalu menyebutnya baseline"], 0,
     "Baseline 'nilai terakhir' (naive) sering mengejutkan kuatnya; RNN layak dipakai kalau mengalahkannya."),
    # --- GPU/TensorRT (4) ---
    ("Mengapa training deep learning cocok untuk GPU?",
     ["Kerjanya didominasi perkalian matriks besar yang bisa dipecah jadi ribuan operasi",
      "GPU punya memori jauh lebih besar dari RAM sehingga seluruh dataset bisa dimuat sekaligus",
      "GPU menjalankan Python lebih cepat sehingga loop training per epoch lebih singkat",
      "GPU menghitung gradient secara simbolik sehingga backprop tidak perlu dilakukan"], 0,
     "Matmul dan konvolusi adalah operasi paralel masif; ribuan core GPU mengerjakannya bersamaan."),
    ("Model kecil (dua layer Dense, batch 32, data 1.000 baris) dilatih di GPU ternyata tidak lebih cepat dari CPU. Penjelasan yang tepat?",
     ["Kerja per langkah terlalu kecil; overhead pindah data dan kernel GPU melebihinya",
      "GPU Colab sedang rusak, karena GPU seharusnya selalu lebih cepat untuk model apa pun",
      "TensorFlow tidak mendukung GPU untuk layer Dense, hanya untuk layer konvolusi",
      "Batch 32 terlalu besar untuk GPU sehingga otomatis dipecah menjadi batch 1"], 0,
     "Keuntungan GPU muncul saat pekerjaannya besar; pada model kecil overhead mendominasi dan CPU bisa setara atau lebih cepat."),
    ("Mixed precision (FP16 untuk sebagian komputasi) dipakai saat training. Apa manfaat dan syaratnya?",
     ["Bisa mempercepat dan menghemat memori pada workload yang sesuai; layer keluaran float32",
      "Model menjadi dua kali lebih akurat karena dihitung dengan dua presisi sekaligus",
      "Semua bobot menjadi integer sehingga modelnya bisa dijalankan di mikrokontroler",
      "Training menjadi deterministik sehingga hasilnya selalu identik di setiap run"], 0,
     "Mixed precision memakai FP16 dan FP32 untuk bagian perhitungan yang berbeda; percepatannya bergantung pada operasi dan hambatan komputasinya, jadi hasilnya diukur. Pada contoh nb05, layer output disetel ke float32."),
    ("Kapan mengubah model Keras menjadi engine TensorRT (lewat ONNX) layak dilakukan?",
     ["Saat model final akan melayani banyak inferensi di GPU NVIDIA dan latensi per prediksi penting",
      "Saat masih bereksperimen dengan arsitektur, supaya tiap percobaan training lebih cepat",
      "Saat data train bertambah, karena TensorRT melatih ulang model secara otomatis di GPU",
      "Saat model akan dijalankan di laptop tanpa GPU, karena TensorRT mengganti CPU menjadi GPU virtual"], 0,
     "TensorRT mengoptimasi inferensi (fusi layer, presisi rendah) untuk model yang sudah dilatih; ia bukan alat training dan butuh GPU NVIDIA."),
    # --- Generatif (4) ---
    ("Apa perbedaan mendasar model diskriminatif dan generatif?",
     ["Diskriminatif memetakan masukan ke label; generatif membuat contoh data baru",
      "Diskriminatif bekerja pada gambar, sedangkan generatif hanya pada teks dan suara",
      "Diskriminatif dilatih tanpa label, generatif membutuhkan label untuk setiap contoh",
      "Diskriminatif selalu lebih besar dan lebih lambat dibanding model generatif"], 0,
     "Classifier menjawab 'kelas apa'; model generatif memodelkan bagaimana data terbentuk sehingga bisa membuat sampel baru."),
    ("Autoencoder dilatih merekonstruksi digit MNIST lewat bottleneck 32 dimensi. Apa arti 'latent space' di sini?",
     ["Ruang ringkasan 32 dimensi tiap gambar; titik berdekatan jadi digit mirip",
      "Daftar 784 pixel asli yang disimpan ulang tanpa perubahan di tengah model",
      "Label 0–9 yang disisipkan ke dalam model supaya rekonstruksinya lebih akurat",
      "Learning rate dan batch size yang dipelajari sendiri oleh encodernya"], 0,
     "Latent space adalah representasi terkompresi; interpolasi antar titik di ruang ini menghasilkan transisi halus antar digit."),
    ("Dalam GAN, generator dan discriminator dilatih bersama. Apa peran masing-masing?",
     ["Generator membuat gambar palsu dari noise; discriminator menilai asli atau palsu",
      "Generator memberi label pada gambar; discriminator menghapus gambar yang labelnya salah",
      "Generator dan discriminator adalah dua salinan model yang sama, dilatih pada data berbeda",
      "Generator mengompresi gambar; discriminator mendekompresinya kembali menjadi gambar asli"], 0,
     "Dua pemain berlawanan: generator berusaha menipu, discriminator berusaha membedakan; keseimbangannya yang menghasilkan gambar realistis."),
    ("Setelah 15 epoch, DCGAN menghasilkan digit yang semuanya terlihat seperti angka 1. Fenomena ini disebut apa?",
     ["Mode collapse: generator menemukan satu keluaran yang menipu discriminator dan berhenti bervariasi",
      "Overfitting discriminator: discriminator menghafal data train sehingga generator tidak bisa belajar apa pun",
      "Vanishing gradient pada decoder: gradient tidak sampai ke layer awal generator sehingga keluarannya seragam",
      "Konvergensi sempurna: GAN memang seharusnya menghasilkan satu digit terbaik setelah cukup epoch"], 0,
     "Mode collapse adalah kegagalan khas GAN: keragaman keluaran hilang meski discriminator tertipu."),
]

# Sebar indeks jawaban supaya tidak selalu B: rotasi deterministik per soal.
ROT = [1, 0, 2, 3, 1, 2, 0, 3, 2, 1, 3, 0, 1, 2, 0, 3, 2, 1, 0, 3, 1, 2, 3, 0, 2, 1, 3, 0, 2, 1]
questions = []
for (q, opts, a, e), target in zip(Q, ROT):
    opts = list(opts)
    correct = opts.pop(a)
    opts.insert(target, correct)
    questions.append({"q": q, "code": None, "options": opts, "answer": target, "explanation": e})

payload = {"module": "02", "title": "Deep Learning Fundamentals", "questions": questions}
N = len(questions)
assert N == 30, N
assert all(len(x["options"]) == 4 for x in questions), "every question needs 4 options"
assert all(0 <= x["answer"] <= 3 for x in questions)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKEL_HEAD = """<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deep Learning Fundamentals — Quiz</title>
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
<header><div class="mod">Modul 02 · Quiz Latihan</div><h1>Deep Learning Fundamentals</h1>
<p>__N__ soal pilihan ganda · murni konsep · pilih satu jawaban</p></header>
<div class="scorebar"><span>Terjawab: <b id="answered">0</b> / __N__</span>
<span>Benar: <b id="correct">0</b></span></div>
<div id="quiz"></div>
<div class="summary" id="summary"><div class="big" id="finalscore"></div>
<div class="msg" id="finalmsg"></div><button class="btn" onclick="location.reload()">Ulangi Quiz</button></div>
<footer>Navasena Gen-ML Course · Modul 02 Deep Learning Fundamentals — Quiz</footer>
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
out = ROOT / "dl-fundamentals-quiz.html"
out.write_text(html, encoding="utf-8")
# cek bias panjang opsi benar vs salah
import statistics
cor = [len(x["options"][x["answer"]]) for x in questions]
inc = [len(o) for x in questions for i, o in enumerate(x["options"]) if i != x["answer"]]
mc, mi = statistics.mean(cor), statistics.mean(inc)
print(f"wrote {out} with {N} questions; mean len correct={mc:.0f} incorrect={mi:.0f} ratio={mc/mi:.2f}")
from collections import Counter
print("answer index distribution:", dict(sorted(Counter(x["answer"] for x in questions).items())))
