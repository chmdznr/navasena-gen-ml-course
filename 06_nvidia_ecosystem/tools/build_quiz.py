#!/usr/bin/env python3
"""Build the reworked Module 06 NVIDIA Ecosystem quiz HTML — ~31 pure-concept questions (code=null),
covering the 5-notebook concept arc (nb01-05): NVIDIA stack front (nb01-02), Trustworthy AI back (nb03-05).
Weighted toward nb04/nb05 (safety/guardrails/privacy + guarded deploy = module identity).
Mirrors 05_rag/tools/build_quiz.py.
Run: python build_quiz.py  ->  ../nvidia-ecosystem-quiz.html
"""
import json, pathlib
from collections import Counter

# Each: (question, [4 options], answer_index, explanation). All pure-concept (code=null).
# Answer key is HAND-BALANCED across indices 0/1/2/3; see assert below.
Q = [
    # ── nb01 · GPU, Precision & Optimasi Inferensi ──────────────────
    ("Kenapa GPU jauh lebih cepat daripada CPU untuk inferensi LLM?",
     ["GPU punya clock per-core yang jauh lebih tinggi daripada CPU",
      "Inti LLM adalah perkalian matriks yang sangat paralel, dan GPU punya ribuan core untuk mengerjakannya serentak",
      "GPU menyimpan seluruh model di cache L1 sehingga tak perlu RAM",
      "CPU tidak bisa menjalankan operasi floating-point sama sekali"], 1,
     "Inti tiap layer transformer adalah perkalian matriks (matmul) yang 'embarrassingly parallel': tiap elemen hasil dihitung independen. GPU punya ribuan core sederhana yang mengerjakannya serentak, sementara CPU hanya punya belasan core."),
    ("Apa fungsi Tensor Core pada GPU NVIDIA, dan apa syaratnya menyala?",
     ["Unit hardware khusus matmul yang hanya aktif pada precision rendah seperti FP16/FP8, bukan FP32",
      "Cache khusus yang mempercepat akses memori untuk semua precision",
      "Core yang hanya dipakai saat training, tidak untuk inference",
      "Unit yang menambah jumlah CUDA core secara dinamis"], 0,
     "Tensor Core mengerjakan perkalian-akumulasi matriks kecil langsung di silikon, tetapi hanya untuk data precision rendah (FP16/BF16/FP8/INT8). FP32 jalan di CUDA core biasa. Maka FP16 bukan sekadar hemat memori — ia menyalakan Tensor Core."),
    ("Dari benchmark T4: memori bobot FP32 6.18 GB vs FP16 3.10 GB. Apa kesimpulannya?",
     ["FP16 separuh memori (2.0x lebih hemat) karena 16 bit = separuh dari 32 bit",
      "FP16 boros memori karena butuh konversi tambahan",
      "FP32 dan FP16 memakai memori yang sama, bedanya hanya kecepatan",
      "FP16 hanya hemat saat model di bawah 1 miliar parameter"], 0,
     "16 bit adalah separuh dari 32 bit, jadi memori bobot turun 2.0x (6.18 -> 3.10 GB) persis sesuai teori. FP16 adalah default deploy paling aman: kualitas hampir identik FP32, tapi separuh memori dan jauh lebih cepat."),
    ("Apa beda quantization 4-bit untuk inference (nb01) dengan QLoRA di Modul 04?",
     ["QLoRA memakai 8-bit, sedangkan inference quant memakai 4-bit",
      "Tidak ada beda; keduanya istilah yang sama persis",
      "Inference quant murni kompresi bobot untuk deploy (tanpa training/adapter); QLoRA meng-quantize agar bisa melatih adapter LoRA (mengubah perilaku)",
      "Inference quant melatih ulang seluruh model, QLoRA tidak"], 2,
     "Konfigurasi quantization-nya mirip, tapi niatnya beda. QLoRA (M04): quantize agar muat, lalu latih adapter LoRA untuk MENGUBAH perilaku (training). nb01: model sudah jadi, kita hanya ingin MENJALANKANNYA dengan memori lebih kecil (inference) — tanpa training, tanpa adapter."),
    ("Apa beda mendasar TensorRT dengan TensorRT-LLM?",
     ["TensorRT-LLM hanya jalan di CPU, TensorRT hanya di GPU",
      "TensorRT adalah versi lama yang sudah ditinggalkan TensorRT-LLM",
      "Keduanya identik, hanya beda nama dagang",
      "TensorRT = compiler inference umum (CNN/ViT dll); TensorRT-LLM = khusus LLM dengan paged KV-cache & in-flight batching"], 3,
     "TensorRT mengoptimalkan model apa pun lewat fusion + kalibrasi precision + auto-tuning kernel. TensorRT-LLM adalah cabang khusus LLM yang menambah paged KV-cache, in-flight (continuous) batching, dan FP8/INT4 — fitur yang dibutuhkan generasi token autoregresif."),

    # ── nb02 · Serving & Deploy: Triton -> Dynamo -> NIM ────────────
    ("Apa beda 'menjalankan model' dan 'menyajikan (serving) model'?",
     ["Menjalankan = di GPU, menyajikan = di CPU",
      "Menjalankan = satu prompt satu jawaban untuk satu pemakai; menyajikan = melayani banyak pengguna konkuren dengan batching, scaling, API standar",
      "Menyajikan selalu lebih lambat dan tidak dipakai di produksi",
      "Keduanya sama; 'serving' hanya istilah pemasaran"], 1,
     "model.generate(...) menjalankan model: satu prompt, satu jawaban, satu pemakai. Serving menambahkan lapisan yang melayani ratusan pengguna konkuren — antre & gabung request (batching), skala replika, satu API standar, observability — supaya GPU mahal tidak menganggur."),
    ("Apa konsep 'dynamic batching' di Triton Inference Server?",
     ["Menjalankan beberapa model berbeda secara bergiliran",
      "Membagi satu request besar menjadi banyak request kecil",
      "Server menunggu sepersekian detik untuk mengumpulkan beberapa request lalu memprosesnya sekaligus, sehingga GPU terisi penuh dan throughput naik",
      "Mengubah ukuran batch training secara otomatis saat fine-tuning"], 2,
     "Bagai kasir yang menunggu sebentar mengumpulkan pesanan lalu memproses sekaligus: memproses 8 request bersama hampir secepat memproses 1 di GPU. Hasilnya throughput melonjak dengan tambahan latency kecil. config.pbtxt mengatur tuas berapa lama server menunggu sebelum membentuk batch."),
    ("Bagaimana posisi Triton dan Dynamo dalam tangga serving NVIDIA?",
     ["Triton = baseline serving (diuji sertifikasi); Dynamo = penerus fokus LLM datacenter, flagship GTC 2025",
      "Dynamo = baseline lama; Triton = penerus terbaru",
      "Keduanya hanya untuk training, bukan serving",
      "Triton untuk gambar, Dynamo untuk teks — keduanya tak berkaitan"], 0,
     "Triton adalah baseline serving yang diuji di sertifikasi (dynamic batching, multi-framework, metrics). Dynamo adalah penerusnya yang diumumkan di GTC 2025 — sering disebut 'Dynamo-Triton' — berfokus pada LLM skala datacenter (disaggregated serving, KV-cache aware routing, multi-node)."),
    ("Apa keistimewaan NVIDIA NIM sebagai cara men-deploy model?",
     ["NIM hanya bisa jalan di Google Colab",
      "NIM mewajibkan kita menulis config.pbtxt dan mem-build engine sendiri",
      "NIM adalah format model baru yang menggantikan ONNX",
      "Model sudah dibungkus optimasi+serving (TensorRT-LLM di dalam) dan disajikan sebagai microservice berprotokol OpenAI /v1 — kita cukup memanggilnya"], 3,
     "NIM (NVIDIA Inference Microservices) membungkus seluruh tumpukan optimasi + serving menjadi layanan yang berbicara protokol OpenAI /v1. Kita tidak mem-build engine, tidak menulis config.pbtxt, tidak mengurus container — cukup memanggilnya seperti memanggil OpenAI."),
    ("Streaming jawaban (stream=True) paling tepat dipahami sebagai apa?",
     ["Trik kompresi yang memperkecil ukuran jawaban",
      "Fitur lapisan serving (bukan fitur model): server meneruskan token begitu dihasilkan, memangkas time-to-first-token",
      "Fitur internal model yang hanya ada di Nemotron",
      "Mode yang membuat jawaban lebih akurat"], 1,
     "Model selalu menghasilkan token satu per satu (autoregresif). Server memilih menahannya sampai lengkap (stream=False) atau meneruskannya seketika (stream=True) — itulah 'efek mengetik'. Ini bagian kontrak OpenAI /v1, jadi kodenya identik lintas backend (NIM, vLLM, Ollama)."),

    # ── nb03 · Fairness, Bias & Tata Kelola ─────────────────────────
    ("Lima pilar Trustworthy AI versi NVIDIA mencakup yang mana?",
     ["Speed, Accuracy, Cost, Latency, Scale",
      "Fairness, Transparency, Accountability, Safety, Privacy",
      "Training, Tuning, Testing, Tracing, Tagging",
      "Tokenization, Embedding, Retrieval, Ranking, Generation"], 1,
     "Lima pilar: Nondiscrimination/Fairness, Transparency, Accountability, Safety, dan Privacy & Security. nb03 mengerjakan tiga pertama (audit offline); nb04 mengerjakan Safety & Privacy (rail runtime)."),
    ("Mengapa fairness diaudit 'offline', bukan sebagai rail runtime per pesan?",
     ["Karena audit offline lebih murah daripada apa pun",
      "Karena regulator melarang audit runtime",
      "Karena fairness diuji pada ribuan keputusan sekaligus dengan data historis di meja audit, bukan pada satu permintaan saat tombol kirim ditekan",
      "Karena model tidak bisa diukur saat sedang berjalan"], 2,
     "Fairness, Transparency, dan Accountability adalah pekerjaan audit offline: diukur pada dataset historis sebelum sistem dipercaya, lalu diulang berkala. Tidak ada 'rail fairness' yang menyala tiap permintaan — bias hanya tampak pada agregat ribuan keputusan, bukan satu per satu. Safety & Privacy-lah yang jadi rail runtime."),
    ("Kenapa akurasi total yang tinggi tidak menjamin model itu adil?",
     ["Karena akurasi selalu salah dihitung pada dataset kecil",
      "Karena akurasi total hanya berlaku untuk data numerik",
      "Akurasi selalu berbanding terbalik dengan fairness",
      "Akurasi menjumlahkan semua kelompok jadi satu angka, sehingga bisa menyembunyikan model yang hampir selalu salah pada satu kelompok"], 3,
     "Model bisa berakurasi 88-92% total dan tetap diam-diam tidak adil: salahnya menumpuk di satu kelompok. Akurasi total menyembunyikannya karena menjumlahkan semua kelompok; fairness metrics membongkarnya dengan memecah per kelompok."),
    ("Apa beda demographic parity dan equalized odds, dan kenapa equalized odds lebih ketat?",
     ["Keduanya sama; equalized odds hanya nama lain demographic parity",
      "Demographic parity bersyarat pada kebenaran, equalized odds tidak",
      "Demographic parity = apakah tingkat persetujuan sama antar kelompok (disparitas < 0.1 = cukup adil); equalized odds menuntut keadilan bersyarat kebenaran (gap TPR/FPR pada yang sama-sama layak), jadi lebih sulit ditipu",
      "Equalized odds mengabaikan label kebenaran sepenuhnya"], 2,
     "Demographic parity membandingkan tingkat hasil positif antar kelompok (mis. P(setuju|Pria) vs P(setuju|Wanita)); ambang ajar disparitas < 0.1 = cukup adil. Tapi ia bisa ditipu dengan menyetujui orang tidak layak demi menyamakan angka. Equalized odds lebih ketat: di antara yang sama-sama layak (TPR) dan sama-sama tidak layak (FPR), peluangnya harus sama antar kelompok."),
    ("Kapan memakai fairness metrics deterministik vs LLM-judge untuk mendeteksi ketidakadilan?",
     ["Selalu pakai LLM-judge karena lebih modern",
      "Metrik deterministik untuk keputusan terstruktur berisiko tinggi (kredit, rekrutmen); LLM-judge untuk teks bebas skala besar — perlakukan keluarannya sebagai sinyal, bukan vonis",
      "Selalu pakai metrik deterministik; LLM-judge tidak pernah berguna",
      "Keduanya wajib dipakai bersamaan pada setiap kasus tanpa kecuali"], 1,
     "Metrik deterministik (numpy): input sama -> angka sama, bisa dipertanggungjawabkan di hadapan auditor, nol biaya — cocok untuk keputusan terstruktur berisiko tinggi. LLM-judge: fleksibel untuk teks bebas tak terstruktur, tapi probabilistik & bisa keliru — perlakukan sebagai sinyal dan validasi dengan tinjauan manusia."),
    ("Apa fungsi 'Model Card' sebagai artefak tata kelola?",
     ["Sertifikat lisensi komersial untuk menjual model",
      "File konfigurasi yang mempercepat inferensi model",
      "Kunci API yang dibutuhkan untuk memanggil model",
      "'Label gizi' model: apa modelnya, data apa yang dipakai, untuk apa ia dimaksudkan, dan apa batasannya — memenuhi pilar Transparency"], 3,
     "Model Card adalah dokumen transparansi (dipopulerkan Google, kini standar de facto di Hugging Face) yang menjelaskan model, datanya, tujuan penggunaan, dan terutama BATASANNYA. Pasangannya, AI Ethics Checklist (ditandatangani sebelum rilis), memenuhi pilar Accountability."),

    # ── nb04 · Safety, Guardrails & Privasi: NemoGuard NYATA ───────
    ("Apa itu NemoGuard, dan bagaimana ia disajikan?",
     ["Sebuah library Python yang di-install lokal untuk memfilter kata",
      "Model khusus penjaga yang dilatih NVIDIA, disajikan sebagai NIM di endpoint yang sama dengan generator — satu API key, satu base_url, beda model=",
      "Sebuah firewall jaringan untuk melindungi server GPU",
      "Fitur bawaan tiap LLM yang aktif secara otomatis"], 1,
     "NVIDIA tidak menyuruh kita menulis filter keamanan dengan tumpukan if. Mereka melatih model khusus penjaga (NemoGuard) dan menyajikannya sebagai NIM di endpoint yang sama persis dengan generator: satu NVIDIA_API_KEY, satu base_url, cukup ganti argumen model=."),
    ("Model content-safety (nvidia/llama-3.1-nemoguard-8b-content-safety) bekerja berdasarkan apa?",
     ["Daftar kata terlarang statis yang dicocokkan persis",
      "Pencarian web untuk tiap pesan masuk",
      "Taksonomi Aegis berisi 23 kategori bahaya (S1..S23) — system prompt = kebijakan + daftar kategori, output JSON {'User Safety':'safe'/'unsafe', ...}",
      "Skor sentimen positif/negatif sederhana"], 2,
     "Content Safety adalah classifier yang dilatih di atas Llama-3.1-8B mengikuti taksonomi Aegis: 23 kategori bahaya (S1..S23) seperti kekerasan, senjata, ujaran kebencian. System prompt-nya berisi kebijakan + daftar kategori; ia membaca percakapan (dibungkus penanda BEGIN/END CONVERSATION) dan mengembalikan JSON {'User Safety': ...}."),
    ("Pertanyaan apa yang dijawab 'topic control' (yang BERBEDA dari content safety)?",
     ["Apakah pesan ini ditulis dalam bahasa yang benar?",
      "Apakah pesan ini berbahaya?",
      "Apakah pesan ini terlalu panjang untuk diproses?",
      "Apakah pesan ini di dalam lingkup yang boleh dilayani asisten ini? (on-topic / off-topic)"], 3,
     "Content safety menjawab 'apakah ini berbahaya?'. Topic control menjawab pertanyaan berbeda: 'apakah ini di dalam lingkup?'. Chatbot HR tak seharusnya memberi saran saham atau diagnosis penyakit — walau semua itu 'aman'. Model membaca definisi topik di system prompt (wajib diakhiri baris restriksi) lalu memvonis on-topic/off-topic."),
    ("Kenapa di tier gratis, jailbreak ditangani lewat 'self-check' classifier langsung (Nemotron-nano), bukan model nvidia/nemoguard-jailbreak-detect?",
     ["Karena nemoguard-jailbreak-detect tidak ada lagi",
      "Karena jailbreak-detect bukan model chat (Random Forest di endpoint /v1/classify), sehingga tak bisa dipanggil pola chat.completions; self-check LLM mengerjakan keputusan yang sama lewat jalur free-tier",
      "Karena self-check selalu 100% akurat",
      "Karena jailbreak tidak berbahaya sehingga tak perlu model serius"], 1,
     "nvidia/nemoguard-jailbreak-detect bukan model chat — ia Random Forest di atas embedding 768-dim lewat endpoint khusus /v1/classify, tak bisa dipanggil dengan chat.completions. Maka kita pakai self-check: minta Nemotron-nano (non-reasoning) memvonis Yes/No — pola sama dengan content-safety, bisa jalan di free-tier dan transparan untuk belajar."),
    ("Mengapa Nemotron-3-nano perlu 'reasoning dimatikan' saat dipakai sebagai rail penjaga?",
     ["Karena reasoning membuat jawaban lebih lambat dikirim ke jaringan",
      "Karena reasoning hanya boleh untuk training",
      "Ia reasoning model: tanpa dimatikan ia mengeluarkan blok <think>...</think> yang mengotori output yang harus berupa satu kata seperti 'unsafe' / 'on-topic'",
      "Karena reasoning menghabiskan kuota GPU lokal"], 2,
     "Nemotron-3-nano adalah reasoning model: secara default ia mengeluarkan blok <think>...</think> sebelum jawaban — boros token dan fatal untuk rail yang mengharap output satu kata (Yes/No, safe/unsafe, on-topic). Reasoning dimatikan lewat extra_body chat_template_kwargs.enable_thinking=False pada setiap panggilan."),
    ("Di NIM untuk Nemotron, cara yang BEKERJA untuk mematikan reasoning adalah?",
     ["Parameter request extra_body chat_template_kwargs.enable_thinking=False",
      "Menempel token /no_think di awal prompt",
      "Menyetel temperature=0",
      "Menambahkan kalimat 'jangan berpikir' di system prompt"], 0,
     "Di endpoint NIM untuk model ini, token prompt /no_think DIABAIKAN. Satu-satunya cara yang bekerja adalah parameter request extra_body={'chat_template_kwargs': {'enable_thinking': False}} — mekanik paling NVIDIA-spesifik di modul ini, ditampilkan inline (bukan disembunyikan di helper)."),
    ("Mengapa PII masking memakai regex deterministik, bukan LLM?",
     ["Karena regex lebih pintar memahami konteks daripada LLM",
      "Karena LLM tidak bisa membaca angka sama sekali",
      "Karena regex bisa dilatih ulang, LLM tidak",
      "Karena kontrol kepatuhan butuh perilaku cepat, andal, dapat diprediksi — pola PII ([NIK]/[PHONE]/[ACCOUNT]) bisa dideteksi pasti tanpa biaya/halusinasi LLM"], 3,
     "PII masking adalah kontrol kepatuhan, jadi harus deterministik: regex mengenali pola PII Indonesia dan menggantinya dengan placeholder [NIK]/[PHONE]/[ACCOUNT]. Urutan pola penting — NIK (16 digit) dicek sebelum rekening (10-15 digit) agar tak salah-klaim. Cepat, andal, tanpa biaya/halusinasi LLM."),
    ("Dua angka kunci UU PDP No. 27/2022 yang wajib diingat untuk sertifikasi adalah?",
     ["Notifikasi kebocoran <= 24 jam; denda <= 5% pendapatan",
      "Notifikasi kebocoran <= 30 hari; denda tetap Rp 1 miliar",
      "Notifikasi kebocoran <= 72 jam; denda administratif <= 2% pendapatan tahunan",
      "Notifikasi kebocoran <= 7 hari; denda <= 10% laba bersih"], 2,
     "UU PDP No. 27/2022 (setara GDPR Indonesia): kebocoran data pribadi wajib dilaporkan ke otoritas & subjek data paling lambat 72 jam, dengan denda administratif hingga 2% pendapatan tahunan. Itu sebabnya PII masking bukan 'nice to have' melainkan kontrol kepatuhan."),
    ("Apa fungsi rail 'grounding' dan apa sifatnya yang berbeda dari rail lain?",
     ["Ia memblokir konten berbahaya seperti content-safety",
      "Ia mempercepat retrieval konteks RAG",
      "Ia mengenkripsi jawaban sebelum dikirim",
      "Ia menjaga kejujuran (anti-halusinasi): memastikan jawaban bersandar pada dokumen & tiap klaim bisa ditelusuri lewat sitasi [n] yang diverifikasi"], 3,
     "Grounding tidak memblokir konten berbahaya — ia rail transparansi yang menjaga kejujuran: jawaban harus bersandar pada konteks sumber, tiap klaim ditandai sitasi [n]. Lalu diverifikasi: sitasi di luar konteks atau klaim tanpa sitasi = sinyal halusinasi -> ditolak/ditandai."),

    # ── nb05 · Capstone: Deploy /ask Berpagar ──────────────────────
    ("Pada capstone /ask, urutan rail yang benar dalam pipeline adalah?",
     ["Generate dulu, lalu baru cek semua rail di akhir",
      "Input rails (jailbreak -> content-safety -> topic-control) -> RAG generate (NIM) -> output rails (grounding -> content-safety jawaban -> PII masking)",
      "PII masking dulu, lalu generate, tanpa cek safety sama sekali",
      "Hanya satu rail content-safety di tengah pipeline"], 1,
     "Tiap POST /ask mengalir: input rails (self-check jailbreak -> content-safety -> topic-control) menyaring SEBELUM menyentuh generator; lalu RAG retrieve + generate (NIM); lalu output rails (grounding -> content-safety pada jawaban -> PII masking). Fail fast: begitu satu input rail menolak, generator tak pernah dipanggil."),
    ("Pada layanan FastAPI /ask, apa peran validasi Pydantic?",
     ["Memvalidasi bentuk request: request tanpa field 'question' ditolak HTTP 422 sebelum menyentuh model — lapis pertahanan termurah",
      "Mempercepat inferensi model NIM",
      "Menyimpan jawaban ke database otomatis",
      "Menjalankan rail content-safety menggantikan NemoGuard"], 0,
     "Pydantic mendefinisikan skema request. FastAPI memvalidasinya otomatis: request cacat (mis. tanpa 'question') ditolak HTTP 422 SEBELUM menyentuh model NIM mana pun. Validasi adalah lapis pertahanan termurah — menolak input rusak tanpa membayar satu panggilan API pun."),
    ("Untuk apa field 'rails' (jejak rail yang aktif) dikembalikan & disimpan tiap /ask?",
     ["Untuk mempercepat permintaan berikutnya",
      "Untuk menghitung tagihan API ke pengguna",
      "Jejak audit: bukti rail mana yang aktif/memblokir, disimpan ke log untuk pilar Accountability & Transparency (dan kepatuhan UU PDP)",
      "Untuk menyimpan jawaban agar bisa di-cache"], 2,
     "Setiap rail yang aktif dicatat dan dikembalikan bersama jawaban ({answer, sources, rails, blocked}). Field rails inilah jejak audit — bukti rail mana yang aktif/memblokir — yang disimpan ke log untuk Accountability & Transparency, sejalan kewajiban audit UU PDP."),
    ("Bagaimana TestClient FastAPI dipakai menguji service tanpa menyalakan uvicorn?",
     ["Ia memanggil endpoint dalam proses (rail tetap benar-benar memanggil NIM), memverifikasi 200 grounded+cited, jailbreak/off-topic blocked, request cacat 422",
      "Ia hanya mengecek sintaks Python tanpa menjalankan rail apa pun",
      "Ia menjalankan server di port 8000 lalu menutupnya",
      "Ia mengganti NIM dengan jawaban palsu agar tak perlu API key"], 0,
     "TestClient memanggil endpoint DALAM proses — tak perlu uvicorn berjalan — tetapi rail-nya benar-benar memanggil NIM. Empat skenario kunci diverifikasi: pertanyaan AI valid -> 200 grounded+bersitasi+blocked=False; jailbreak -> blocked=True; off-topic -> blocked=True; request tanpa question -> 422."),
    ("Kenapa di capstone, mengganti generator (mis. ke model spesialis M04 via Ollama) tidak menyentuh logika rail?",
     ["Karena rail ditulis ulang otomatis tiap ganti model",
      "Karena model spesialis kebetulan punya rail bawaan sendiri",
      "Karena rail hanya aktif untuk model NVIDIA",
      "Tema 'satu client, banyak backend': semua backend bicara kontrak OpenAI /v1, jadi ganti generator = ganti base_url + model; keenam rail tetap utuh"], 3,
     "Karena semua backend (NIM hosted, Ollama lokal, engine TensorRT Jetson) berbicara kontrak OpenAI /v1 yang sama, mengganti generator hanya berarti mengganti base_url + model — tanpa menyentuh logika rail sama sekali. Keenam rail tetap utuh; yang berubah hanya tiga baris konfigurasi."),
    ("Apa pesan besar capstone tentang posisi Trustworthy AI dalam sistem AI?",
     ["Trustworthy AI cukup didokumentasikan di akhir, tak perlu dijalankan",
      "Trustworthy AI bukan fitur tambahan, melainkan lapisan runtime yang membungkus model dan keputusannya bisa diaudit",
      "Trustworthy AI hanya relevan untuk model di atas 70B parameter",
      "Trustworthy AI menggantikan kebutuhan akan serving dan optimasi"], 1,
     "Capstone membuktikan 'AI yang bisa dipercaya' bukan fitur tambahan, melainkan lapisan runtime yang membungkus model — input rails + RAG + output rails — yang berjalan di SETIAP permintaan dan keputusannya (field rails) bisa diaudit. Inilah titik temu Software Development (deploy/serving) dan Trustworthy AI yang ditanyakan ujian."),
]

questions = [{"q": q, "code": None, "options": opts, "answer": a, "explanation": e} for (q, opts, a, e) in Q]
payload = {"module": "06", "title": "NVIDIA Ecosystem", "questions": questions}
N = len(questions)
assert all(len(x["options"]) == 4 for x in questions), "every question needs 4 options"
assert all(0 <= x["answer"] <= 3 for x in questions)
assert all(x["code"] is None for x in questions), "every question must be code=null (pure-concept)"

# Answer distribution must be roughly even; at most one index may appear ~10x.
_dist = Counter(x["answer"] for x in questions)
assert max(_dist.values()) <= 11, f"answer distribution too skewed: {dict(_dist)}"

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKEL_HEAD = """<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NVIDIA Ecosystem — Quiz</title>
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
<header><div class="mod">Module 06 · Quiz Latihan</div><h1>NVIDIA Ecosystem</h1>
<p>__N__ soal pilihan ganda · murni konsep · pilih satu jawaban</p></header>
<div class="scorebar"><span>Terjawab: <b id="answered">0</b> / __N__</span>
<span>Benar: <b id="correct">0</b></span></div>
<div id="quiz"></div>
<div class="summary" id="summary"><div class="big" id="finalscore"></div>
<div class="msg" id="finalmsg"></div><button class="btn" onclick="location.reload()">Ulangi Quiz</button></div>
<footer>Navasena Gen-ML Course · Module 06 NVIDIA Ecosystem — Quiz</footer>
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
  let msg = pct>=80 ? "Hebat! Pemahaman kamu solid." : pct>=60 ? "Lumayan — ulas lagi materi yang masih salah." : "Ayo pelajari lagi modulnya, lalu coba ulangi.";
  document.getElementById("finalmsg").textContent = msg;
  s.scrollIntoView({behavior:"smooth"});
}
</script></body></html>"""

html = SKEL_HEAD.replace("__N__", str(N)).replace("__JSON__", json.dumps(payload, ensure_ascii=False))
out = ROOT / "nvidia-ecosystem-quiz.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out} with {N} pure-concept questions")
print(f"answer distribution: {dict(sorted(Counter(x['answer'] for x in questions).items()))}")
