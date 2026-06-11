# Redesign Modul 04 — LLM Fundamentals: "Bangun → Pakai → Skalakan"

- **Tanggal:** 2026-06-11
- **Modul:** `04_llm/`
- **Status:** Design (disetujui user, menunggu review spec)
- **Penulis:** Achmad Zaenuri

---

## 1. Konteks & Masalah

Notebook `04_llm/01_llm_basics.ipynb` saat ini menganchor seluruh demo pada model yang
tidak layak:

- **`facebook/opt-350m`** — base model 2022 yang tidak pernah di-instruction-tune. Tidak bisa
  "menjawab", hanya melanjutkan teks. Output nyata dari run terakhir: *"A large language model
  is a set of rules... But it is not a set of rules. A large language model is not a set of
  rules..."* (looping, inkoheren). Dipakai untuk basic generation, chatbot, few-shot, dan
  evaluation — sehingga **seluruh tulang punggung notebook menghasilkan teks ngawur**.
- **`prajjwal1/bert-tiny`** — head klasifikasi acak (belum dilatih) → `LABEL_0/LABEL_1` skor ~0.5.
- **`TinyLlama-1.1B`** (hasil perbaikan QC sebelumnya) — satu-satunya yang koheren, tapi hanya
  dipakai untuk demo "Who are you?" lalu notebook balik ke opt-350m.

Bukti kapasitas terbuang: pada **Colab T4 biasa (bukan High-RAM)**, run penuh hanya memakai
**GPU RAM 5.3 / 15.0 GB** dan **System RAM 4.7 / 12.7 GB**. T4 sanggup menjalankan model
instruct 3B (fp16) dan 7B (4-bit) yang menghasilkan output koheren.

Selain itu, notebook sama sekali **tidak mengajarkan cara kerja Transformer** — langsung "pakai
library" tanpa peserta pernah melihat attention/transformer dari dalam.

## 2. Tujuan & Non-Tujuan

### Tujuan
1. Output demo **koheren & meyakinkan** — pakai model instruct kapabel yang memanfaatkan T4.
2. Mengajarkan **cara kerja LLM dari nol** (attention, transformer block, training) secara
   hands-on, bukan sekadar konsep di slide.
3. Mengajarkan **teknik produksi** (quantization, memory profiling, batching, streaming) sebagai
   jembatan kuat ke Modul 05 (RAG).
4. **Nol friksi lisensi** — semua model runnable bersifat non-gated (tanpa approval HuggingFace),
   penting untuk 21 peserta bootcamp.
5. Semua jalan di **Colab T4 16GB biasa**, reproducible.

### Non-Tujuan (YAGNI)
- Tidak menyelaraskan slides/quiz/cheatsheet di iterasi ini (deliverables: **notebook dulu**;
  aset pendukung menyusul di sesi terpisah setelah notebook tervalidasi di Colab).
- Tidak memakai model gated (Llama/Gemma) untuk jalur runnable — hanya sebagai **konsep**.
- Tidak melatih LLM skala besar; notebook 00 hanya melatih GPT mungil sebagai ilustrasi.
- Tidak mengejar kompatibilitas `transformers` v5 (sengaja di-pin `<5` — lihat §6).

## 3. Batasan Lingkungan (verified)

| Aspek | Nilai |
|-------|-------|
| GPU | NVIDIA Tesla T4, 16 GB (~15 GB usable), compute capability **7.5 → TIDAK ada bf16** |
| System RAM | ~12.7 GB (Colab free) |
| Runtime | Colab free, Python 3.12 |
| transformers | **pin `>=4.44,<5`** (wajib — lihat §6) |

## 4. Arsitektur: Tiga Notebook (Build → Use → Scale)

| # | Notebook | Peran | Model utama | Library |
|---|----------|-------|-------------|---------|
| **00** | `00_transformer_from_scratch.ipynb` *(baru)* | **BANGUN** | bikin sendiri | PyTorch murni (tanpa `transformers`) |
| **01** | `01_llm_basics.ipynb` *(rombak total)* | **PAKAI** | `Qwen/Qwen2.5-3B-Instruct` | transformers, gradio |
| **02** | `02_llm_production.ipynb` *(baru)* | **SKALAKAN** | `mistralai/Mistral-7B-Instruct-v0.3` | transformers, bitsandbytes, accelerate |

**Arc pedagogis (rasa sakit → obat):** 00 menunjukkan training mahal/lambat → 01 memakai model
jadi; 01 mentok di 7B & lambat & tanpa streaming → 02 quantization/batching/streaming; 02
menunjukkan model tetap halusinasi & punya knowledge cutoff → Modul 05 (RAG).

## 5. Strategi Model (verified 2026-06, T4 16GB, non-gated)

| Slot | Model | Mode | VRAM (perkiraan konservatif) | Lisensi | Fit T4 |
|------|-------|------|------------------------------|---------|--------|
| Chat utama | `Qwen/Qwen2.5-3B-Instruct` | fp16 | ~8 GB | Qwen Research (non-komersial) | ✅ |
| Ringan/fallback | `Qwen/Qwen2.5-1.5B-Instruct` | fp16 | ~4.5 GB | Apache-2.0 | ✅ |
| 7B showcase | `mistralai/Mistral-7B-Instruct-v0.3` | 4-bit NF4 | ~5.5 GB | Apache-2.0 | ✅ |
| (counter-example) | Mistral-7B fp16 | fp16 | ~16 GB | — | ❌ OOM = pelajaran |
| Klasifikasi (acak) | `prajjwal1/bert-tiny` | fp16 | <0.1 GB | open | ✅ |
| Klasifikasi (terlatih) | `distilbert-base-uncased-finetuned-sst-2-english` | fp16 | ~0.3 GB | open | ✅ |
| Zero-shot | `facebook/bart-large-mnli` | fp16 | ~1.6 GB | MIT | ✅ |
| Gated (konsep saja) | `google/gemma-2-2b-it` | tidak dijalankan | — | Gemma terms | n/a |

**Catatan lisensi (sebutkan di kelas):** Qwen2.5-**3B** memakai *Qwen Research License*
(non-komersial) — aman untuk edukasi, tapi flag batasannya bila peserta mau produksi komersial.
Qwen2.5-1.5B & 7B + Mistral-7B = Apache-2.0 (bebas). Dipilih 3B sebagai chat utama karena
**multilingual terbaik (Bahasa Indonesia masuk akal)** untuk kelas Indonesia.

**Budget VRAM — load satu model pada satu waktu** (`del model; torch.cuda.empty_cache()` antar
bagian). Resident terbesar: Mistral-7B fp16 (~16 GB → OOM, ini demonya) atau 4-bit (~5.5 GB,
nyaman). Semua aman di ~15 GB usable.

## 6. Guardrail Teknis (verified — wajib dipatuhi implementasi)

1. **Pin `transformers>=4.44,<5`** di tiap sel install. Alasan terverifikasi: (a) v5 punya regresi
   OOM bnb 4-bit (HF issue #43032); (b) v5 mengubah kontrak `apply_chat_template` & backend
   tokenizer (memecah `bert-tiny`, dll). 4.x stabil membuat ketiga notebook jalan sesuai materi.
   Jika peserta sempat memakai v5 di sesi yang sama: **Runtime → Restart**, lalu Run all.
2. **4-bit di T4 — `bnb_4bit_compute_dtype=torch.float16` (BUKAN bfloat16).** bf16 **CRASH** di
   T4 (compute 7.5 < 8.0). Ini kegagalan #1 quantization di T4.
   ```python
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,      # hemat ~0.4 GB
       bnb_4bit_compute_dtype=torch.float16 # WAJIB fp16 di T4
   )
   model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config,
                                                device_map="auto")
   ```
   Pakai `quantization_config=` (arg `load_in_4bit=True` langsung sudah deprecated).
3. **Model bf16-native (Qwen, Mistral) → load `torch_dtype=torch.float16`** di T4 (bukan
   bfloat16). Nilai bobot aman di rentang fp16; standar di Colab T4.
4. **Chat inference (pola robust lintas-versi):**
   ```python
   messages = [{"role": "system", "content": "..."},
               {"role": "user", "content": "..."}]
   inputs = tokenizer.apply_chat_template(
       messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
   ).to(model.device)
   out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7,
                        top_p=0.9, pad_token_id=tokenizer.eos_token_id)
   ```
   `return_dict=True` + `generate(**inputs)` menghindari error `'shape'` di transformers baru dan
   membawa `attention_mask`.
5. **Batching decoder-only → `tokenizer.padding_side="left"`** + set `pad_token` bila kosong.
6. **Device-agnostic:** `.to(model.device)`, bukan hardcode `"cuda"`.
7. **`gpu_mem()` helper bersama:**
   ```python
   def gpu_mem(tag=""):
       a = torch.cuda.memory_allocated()/1e9; r = torch.cuda.memory_reserved()/1e9
       print(f"[{tag}] allocated={a:.2f} GB | reserved={r:.2f} GB")
   ```

## 7. Konvensi Bersama (ketiga notebook)

- Markdown **Bahasa Indonesia**; kode & komentar **English** (gaya kursus).
- Struktur: sel install ber-pin → cek GPU (`!nvidia-smi` + `torch.cuda`) → `gpu_mem()` helper →
  isi materi → tabel **Ringkasan** → **jembatan** ke notebook berikut.
- **Ship dengan output bersih** (peserta menjalankan sendiri di Colab).
- Setiap notebook self-contained (tidak bergantung file lokal selain yang di-download/embed).
- Header "Apa yang akan kita pelajari?" di awal tiap notebook.

## 8. Desain Per-Notebook

### 8.1 `00_transformer_from_scratch.ipynb` — BANGUN
**Tujuan belajar:** memahami attention & transformer dengan membangunnya dari nol, lalu melatih
GPT mungil dan melihatnya "belajar" bahasa. **Hanya PyTorch** (tanpa `transformers`).

**Korpus:** teks Bahasa Indonesia kecil yang self-contained (~beberapa KB, public-domain —
mis. kumpulan pantun/prosa pendek) di-embed sebagai string di notebook. **Char-level** (vocab
kecil, tanpa kompleksitas BPE; BPE disebut konsep saja).

**Konfigurasi tiny GPT (target ~1–3 jt parameter):** `block_size≈128`, `n_layer≈4`, `n_head≈4`,
`n_embd≈128–192`, dropout 0.1. Training: AdamW, beberapa ribu langkah, ~menit di T4 (atau CPU).

**Sel/bagian:**
1. Intro + gambaran besar (teks → token → embedding → transformer → prediksi token berikutnya).
2. **Tokenization dari nol** — bangun vocab char-level dari korpus, `encode`/`decode`, split
   train/val, fungsi `get_batch`.
3. **Embedding dari nol** — `nn.Embedding` token + positional; tunjukkan shape `(B, T, C)`.
4. **Self-attention dari nol** — scaled dot-product manual di contoh kecil (Q·Kᵀ/√d → softmax →
   ·V), lalu `Head` sebagai `nn.Module` dengan W_q/W_k/W_v.
5. **Causal mask + multi-head** — `tril` mask token masa depan; `MultiHeadAttention` gabung head.
6. **Transformer block** — LayerNorm (pre-norm), `FeedForward` (Linear→GELU→Linear), koneksi
   residual.
7. **Rakit GPT mungil** — stack `n_layer` block + final LayerNorm + `lm_head`; cetak jumlah
   parameter.
8. **Training loop** — next-token char-level; cetak train/val loss berkala; kurva loss.
9. **Generate** — sampling autoregresif (`generate` buatan sendiri); bandingkan output sebelum vs
   sesudah training (acak → mirip bahasa Indonesia).
10. Ringkasan + jembatan: "inilah yang dilakukan GPT/Qwen, tapi miliaran param + data internet →
    selanjutnya kita pakai yang sudah dilatih (notebook 01)."

### 8.2 `01_llm_basics.ipynb` — PAKAI *(rombak total)*
**Tujuan belajar:** memakai model instruct pretrained dengan benar; output koheren di semua sel.
Anchor **Qwen2.5-3B-Instruct** (fp16); fallback disebut **Qwen2.5-1.5B-Instruct**.

**Sel/bagian:**
1. Setup — install ber-pin, cek GPU, `gpu_mem()`.
2. **Load model instruct dengan benar** — Qwen2.5-3B fp16, `device_map="auto"`; cetak footprint
   VRAM via `gpu_mem()`; sebut fallback 1.5B untuk runtime lambat.
3. **Chat template + system prompt** — `apply_chat_template` (system+user) per pola §6.4; kontras
   dengan string `"User:..."` mentah; jelaskan ChatML (`<|im_start|>`).
4. **Parameter generasi** — `temperature`, `top_p`, `repetition_penalty`, `max_new_tokens` vs
   `max_length`; demo deterministik (greedy) vs kreatif (sampling).
5. **Prompt engineering** — zero-shot; **few-shot yang kini benar** (model menjawab "Rome");
   steering perilaku via system prompt.
6. **Klasifikasi teks** — `bert-tiny` (head acak → LABEL_0/1 ~0.5, tampilkan kualitatif karena
   non-deterministik) **vs** `distilbert-sst2` (terlatih → POSITIVE/NEGATIVE); plus **zero-shot**
   `bart-large-mnli` dengan `candidate_labels` custom (selalu lewat pipeline `zero-shot-
   classification`, jangan ekspos label NLI mentah).
7. **Chatbot Gradio** — `gr.ChatInterface` memakai Qwen2.5-3B (koheren); format `messages`
   modern (`type="messages"`), bukan tuples.
8. Ringkasan + jembatan: "7B tak muat fp16 di T4, generasi lambat, tak ada streaming → notebook 02."

### 8.3 `02_llm_production.ipynb` — SKALAKAN *(baru)*
**Tujuan belajar:** teknik produksi/scaling di T4. Anchor **Mistral-7B-Instruct-v0.3** (4-bit).

**Sel/bagian:**
1. Setup — install ber-pin + `bitsandbytes` + `accelerate`; cek GPU; `gpu_mem()`.
2. **Masalahnya** — jelaskan 7B fp16 butuh ~16 GB → **OOM di T4**, dengan **perhitungan**
   (7.25B × 2 byte ≈ 14.5 GB bobot + KV/aktivasi ≈ ~16 GB). **JANGAN picu OOM sungguhan**:
   memicu CUDA OOM nyata bisa merusak context & memaksa restart runtime (mematahkan "Run all").
   Tampilkan ukuran bobot fp16 secara teoretis, lalu pivot ke 4-bit sebagai solusinya.
3. **Quantization 4-bit** — `BitsAndBytesConfig` per §6.2 (tekankan `compute_dtype=float16`); load
   Mistral-7B 4-bit (~5.5 GB); generate jawaban koheren. "7B di Colab gratis!"
4. **GPU memory profiling** — `memory_allocated/reserved`, `max_memory_allocated`; bandingkan
   footprint; higiene `del model; torch.cuda.empty_cache()`.
5. **Batching** — banyak prompt dalam satu `generate`; ukur throughput (tokens/sec) tunggal vs
   batch; `padding_side="left"` + `attention_mask`.
6. **Streaming token** — `TextStreamer` (cetak saat generate) & `TextIteratorStreamer` (threaded,
   untuk aplikasi).
7. **Jembatan → RAG (Modul 05)** — LLM halusinasi + knowledge cutoff; quantization/batching/
   streaming membuat serving feasible, tapi jawaban benar/terkini butuh retrieval → RAG.

## 9. Deliverables & Urutan Kerja

1. **Notebook dulu** (urutan implementasi): 00 → 01 → 02. Tiap notebook **smoke-test di Colab T4
   biasa** sebelum dianggap selesai (commit + push agar bisa diakses via
   `colab.research.google.com/github/...`).
2. Setelah ketiga notebook tervalidasi: perbarui **slides / quiz / cheatsheet** agar selaras
   (sesi terpisah).
3. **Google Drive sync ditahan** sampai user mengonfirmasi run penuh ketiga notebook lulus.

## 10. Rencana Validasi

- Setiap notebook: Run all dari atas di **T4 biasa** (non-High-RAM), nol error, output koheren.
- Notebook 00: tiny GPT loss turun & generate menghasilkan teks makin mirip bahasa.
- Notebook 01: Qwen2.5-3B menjawab koheren (termasuk prompt Indonesia); klasifikasi menunjukkan
  kontras acak-vs-terlatih.
- Notebook 02: Mistral-7B 4-bit load tanpa OOM; profiling menunjukkan ~5.5 GB; batching & streaming
  jalan.
- Konfirmasi GPU RAM puncak < ~15 GB.

## 11. Risiko & Mitigasi

| Risiko | Mitigasi |
|--------|----------|
| Qwen2.5-3B fp16 + model lain resident bersamaan → OOM | Load satu model/waktu; `del`+`empty_cache` antar bagian |
| `compute_dtype=bfloat16` tak sengaja terpakai | Guardrail §6.2 eksplisit; review tiap sel quantization |
| transformers v5 terinstal | Pin `<5` + instruksi restart runtime |
| VRAM estimasi meleset ±1–2 GB (bukan benchmark live) | Validasi nyata saat smoke-test Colab; angka di materi = panduan, bukan klaim |
| Korpus notebook 00 terlalu besar/lambat | Char-level + korpus ~beberapa KB + tiny config; target training menit |
| Lisensi Qwen2.5-3B non-komersial | Disebut eksplisit di materi; alternatif Phi-3.5-mini (MIT) bila perlu |

## 12. Pertanyaan Terbuka

- Korpus persis untuk notebook 00 (teks Indonesia public-domain mana). Default: embed string kecil
  self-contained; finalisasi saat implementasi.
- Apakah perlu menampilkan visualisasi heatmap attention di notebook 00 (nice-to-have, tidak wajib).
