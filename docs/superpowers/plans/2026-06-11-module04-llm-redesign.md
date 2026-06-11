# Modul 04 LLM Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rombak Modul 04 jadi tiga notebook "Bangun → Pakai → Skalakan" yang jalan di Colab T4 16GB biasa dengan output koheren & non-gated.

**Architecture:** `00_transformer_from_scratch` (PyTorch murni: bangun attention + latih tiny GPT) → `01_llm_basics` (rombak: Qwen2.5-3B-Instruct, prompting, klasifikasi, chatbot) → `02_llm_production` (Mistral-7B 4-bit, profiling, batching, streaming, jembatan RAG). Verifikasi lokal via `validate_notebook.py`; verifikasi nyata via smoke-test Colab T4.

**Tech Stack:** PyTorch, HuggingFace transformers (`>=4.44,<5`), bitsandbytes, accelerate, gradio, Google Colab T4.

**Spec:** `docs/superpowers/specs/2026-06-11-module04-llm-redesign-design.md`

---

## Catatan Domain (BACA DULU)

- **Tidak ada pytest.** "Test" = (1) `validate_notebook.py` (lokal: JSON valid + `py_compile` tiap sel kode), lalu (2) **Run all di Colab T4 biasa** (manual, oleh user) di tiap checkpoint.
- **Tidak bisa run di laptop** (tak ada GPU). Jangan klaim notebook "berhasil run" dari lokal — hanya validasi statis + Colab.
- **Bahasa:** markdown Indonesia, kode/komentar English.
- **Membangun notebook:** tiap notebook ditulis sekali sebagai file `.ipynb` JSON lengkap via `Write` (semua sel sudah disiapkan di plan ini), lalu `validate_notebook.py`, lalu commit. Per-notebook = satu unit deliverable yang bisa diuji sendiri.
- **Guardrail wajib** (dari spec §6): pin `transformers>=4.44,<5`; di T4 `bnb_4bit_compute_dtype=torch.float16` (bf16 CRASH); load bobot `torch_dtype=torch.float16`; `apply_chat_template(..., return_dict=True)` + `generate(**inputs)`; batching `padding_side="left"`; `.to(model.device)`.
- **Ship output bersih** (tiap code cell `outputs: []`, `execution_count: null`).
- **Format ipynb:** `nbformat=4, nbformat_minor=5`. Tiap sel beri `id` unik (mis. `c00-01`).

---

## File Structure

- Create: `04_llm/tools/validate_notebook.py` — validator lokal (JSON + py_compile).
- Create: `04_llm/00_transformer_from_scratch.ipynb` — notebook BANGUN.
- Modify (overwrite): `04_llm/01_llm_basics.ipynb` — rombak total notebook PAKAI.
- Create: `04_llm/02_llm_production.ipynb` — notebook SKALAKAN.
- (Tak disentuh di iterasi ini: `slides/`, `*-quiz.html`, `*-cheatsheet.*` — menyusul.)

Helper JSON-builder yang dipakai semua task notebook (jalankan sekali di kepala implementasi, atau tulis sel manual):
```python
import json
def md(id, src):   return {"id":id,"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)}
def code(id, src): return {"id":id,"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}
def nb(cells):     return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
# json.dump(nb([...]), open(path,"w"), indent=1, ensure_ascii=False)
```

---

## Task 0: Validator lokal

**Files:**
- Create: `04_llm/tools/validate_notebook.py`

- [ ] **Step 1: Tulis validator**

```python
#!/usr/bin/env python3
"""Validate a course notebook: valid JSON + every code cell compiles.
Magics/shell lines (starting with ! or %) are treated as `pass`."""
import ast, json, re, sys

def validate(path):
    nb = json.load(open(path, encoding="utf-8"))
    assert nb.get("nbformat") == 4, "nbformat must be 4"
    errs = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        clean = "\n".join("pass" if re.match(r"\s*[!%]", ln) else ln
                          for ln in src.split("\n"))
        try:
            ast.parse(clean)
        except SyntaxError as e:
            errs.append(f"  cell {i}: {e}")
    if errs:
        print(f"FAIL {path}\n" + "\n".join(errs)); return False
    n = sum(c["cell_type"] == "code" for c in nb["cells"])
    print(f"OK   {path}  ({len(nb['cells'])} cells, {n} code)"); return True

if __name__ == "__main__":
    ok = all(validate(p) for p in sys.argv[1:])
    sys.exit(0 if ok else 1)
```

- [ ] **Step 2: Smoke-test validator on the current (pre-redesign) notebook**

Run: `python 04_llm/tools/validate_notebook.py 04_llm/01_llm_basics.ipynb`
Expected: `OK   04_llm/01_llm_basics.ipynb  (...)`

- [ ] **Step 3: Commit**

```bash
git add 04_llm/tools/validate_notebook.py
git commit -m "chore(module04): tambah validate_notebook.py (JSON + py_compile gate)"
```

---

## PHASE A — `00_transformer_from_scratch.ipynb` (BANGUN)

Bangun seluruh notebook 00 dengan sel-sel berikut (urut). Hanya `torch` (TANPA `transformers`).
Hyperparam tiny GPT: `block_size=64, batch_size=32, n_embd=128, n_head=4, n_layer=4, dropout=0.1, lr=3e-4, max_iters=3000`.

### Task A1: Header + setup + korpus + tokenizer

**Files:** Create `04_llm/00_transformer_from_scratch.ipynb`

- [ ] **Step 1: Buat sel-sel berikut**

`md` cell `c00-00`:
```markdown
# 📗 Module 04 · Notebook 00 — Transformer from Scratch

Sebelum memakai LLM siap pakai, kita **bangun mesinnya dari nol**: tokenizer, embedding,
**self-attention**, transformer block, lalu rakit **GPT mungil** dan **latih** sampai ia
"belajar" pola bahasa Indonesia. Semua pakai **PyTorch murni** (tanpa `transformers`).

## Apa yang akan kita pelajari?
1. Tokenization & embedding dari nol
2. **Self-attention** (scaled dot-product) — inti Transformer
3. Causal mask + multi-head attention
4. Transformer block (LayerNorm, GELU, feed-forward, residual)
5. Rakit GPT mungil & latih (next-token prediction)
6. Generate teks & lihat ia belajar

> Ini "GPT" yang sama secara prinsip dengan raksasa modern — bedanya skala & data.
```

`code` cell `c00-01` (setup):
```python
# PyTorch murni — tidak perlu transformers di notebook ini
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Hyperparameter tiny GPT (cukup kecil untuk dilatih dalam hitungan menit di T4)
batch_size = 32      # banyak sequence paralel per langkah
block_size = 64      # panjang konteks maksimum (token)
n_embd = 128         # dimensi embedding
n_head = 4           # jumlah attention head
n_layer = 4          # jumlah transformer block
dropout = 0.1
learning_rate = 3e-4
max_iters = 3000
eval_interval = 300
eval_iters = 100
```

`md` cell `c00-02`:
```markdown
## 1. Korpus & Tokenization (char-level)

Model tidak membaca huruf — ia membaca **angka**. Tokenizer mengubah teks ↔ ID.
Di sini kita pakai **char-level** (tiap karakter = 1 token) agar sederhana; LLM nyata
memakai **subword (BPE)**. Kita latih di korpus kecil berbahasa Indonesia.
```

`code` cell `c00-03` (corpus + tokenizer). Embed korpus Indonesia self-contained (~2-3 KB; perbanyak/variasikan agar model punya cukup pola — implementer boleh menambah paragraf bertema AI/ML/Indonesia, public-domain/original):
```python
text = """Kecerdasan buatan mengubah cara manusia bekerja dan belajar.
Model bahasa besar dilatih dari teks dalam jumlah masif di internet.
Sebuah model belajar memprediksi kata berikutnya dari konteks sebelumnya.
Semakin banyak data dan parameter, semakin kaya pola yang bisa ditangkap.
Transformer memproses kata secara paralel lewat mekanisme attention.
Attention membuat model menimbang kata mana yang paling relevan.
Dengan begitu model memahami konteks kalimat yang panjang sekalipun.
Pembelajaran mesin adalah fondasi dari kecerdasan buatan modern.
Jaringan saraf tiruan terinspirasi dari cara kerja otak manusia.
Data yang berkualitas sama pentingnya dengan ukuran model itu sendiri.
"""
# Gandakan & sambung agar korpus cukup panjang untuk melatih pola char-level.
text = (text + "\n") * 60

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

print("panjang korpus (char):", len(text))
print("ukuran vocab:", vocab_size)
print("contoh encode 'data':", encode("data"))

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
```

- [ ] **Step 2: Validate (akan diulang tiap task — notebook tumbuh)**

Run: `python 04_llm/tools/validate_notebook.py 04_llm/00_transformer_from_scratch.ipynb`
Expected: `OK ...`

### Task A2: Embedding + self-attention (single head)

- [ ] **Step 1: Tambah sel**

`md` cell `c00-04`:
```markdown
## 2. Embedding & Self-Attention

Tiap token ID → **vektor** (token embedding). Kita tambah **positional embedding** agar model
tahu urutan. Lalu inti Transformer: **self-attention** — tiap posisi menghitung Query, Key,
Value; skor `Q·Kᵀ/√d` → softmax → rata-rata berbobot atas Value.
```

`code` cell `c00-05` (intuisi scaled dot-product pada contoh kecil):
```python
# Intuisi attention pada satu contoh kecil (B=1, T=4, C=2)
torch.manual_seed(0)
B, T, C = 1, 4, 2
x = torch.randn(B, T, C)
q = k = v = x                      # sederhana: Q=K=V=x untuk ilustrasi
wei = q @ k.transpose(-2, -1) * C ** -0.5   # (B,T,T) skor kemiripan
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))   # causal: tak boleh lihat masa depan
wei = F.softmax(wei, dim=-1)
print("bobot attention (tiap baris jumlahnya 1, segitiga bawah):")
print(wei[0])
out = wei @ v
print("output shape:", out.shape)
```

`code` cell `c00-06` (Head sebagai Module):
```python
class Head(nn.Module):
    """Satu attention head (causal)."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v                                         # (B,T,head_size)
```

- [ ] **Step 2: Validate** → `OK`

### Task A3: Multi-head + feed-forward + Block

- [ ] **Step 1: Tambah sel**

`md` cell `c00-07`:
```markdown
## 3. Multi-Head Attention & Transformer Block

Beberapa head berjalan paralel (tiap head "memperhatikan" hal berbeda), lalu disatukan.
Satu **transformer block** = multi-head attention + feed-forward, dibungkus **LayerNorm**
(pre-norm) dan **koneksi residual** agar gradien mengalir mulus.
```

`code` cell `c00-08`:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: pre-norm + residual."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

- [ ] **Step 2: Validate** → `OK`

### Task A4: Rakit GPT + hitung parameter

- [ ] **Step 1: Tambah sel**

`md` cell `c00-09`:
```markdown
## 4. Rakit GPT Mungil

Tumpuk `n_layer` block + LayerNorm akhir + `lm_head` (proyeksi ke vocab). Tambah method
`generate` untuk sampling autoregresif.
```

`code` cell `c00-10`:
```python
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding_table(idx)                       # (B,T,C)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                    # (B,T,vocab)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]                         # potong ke konteks
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                              # token terakhir
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = TinyGPT().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"jumlah parameter: {n_params:,} (~{n_params/1e6:.2f} jt)")
```

- [ ] **Step 2: Validate** → `OK`

### Task A5: Generate sebelum latih + training loop + generate sesudah

- [ ] **Step 1: Tambah sel**

`md` cell `c00-11`:
```markdown
## 5. Sebelum Dilatih → Latih → Sesudah

Model yang baru diinisialisasi menghasilkan **acak**. Kita latih (next-token prediction),
lalu generate lagi — perhatikan teks berubah dari acak menjadi mirip pola bahasa Indonesia.
```

`code` cell `c00-12` (generate sebelum training):
```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("=== SEBELUM training (acak) ===")
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
```

`code` cell `c00-13` (training loop):
```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it:4d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

`code` cell `c00-14` (generate sesudah training):
```python
print("=== SESUDAH training ===")
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
```

`md` cell `c00-15` (ringkasan + jembatan):
```markdown
## Ringkasan & Jembatan

Kita membangun Transformer dari nol: tokenization → embedding → **self-attention** →
multi-head → block → GPT, lalu **melatihnya** hingga loss turun dan output makin mirip bahasa.

| Komponen | Fungsi |
|----------|--------|
| Token + positional embedding | teks → vektor + info urutan |
| Self-attention | timbang token relevan (Q·Kᵀ/√d → softmax → V) |
| Causal mask | cegah melihat token masa depan |
| Multi-head | banyak "sudut perhatian" paralel |
| Transformer block | attention + FFN + LayerNorm + residual |

GPT raksasa (GPT-4, Qwen, Llama) = **prinsip yang sama**, tapi miliaran parameter + data
internet + ribuan GPU. **Notebook 01**: kita pakai salah satu yang sudah dilatih.
```

- [ ] **Step 2: Validate** → `OK`

### Task A6: Commit notebook 00

- [ ] **Step 1: Validate final**

Run: `python 04_llm/tools/validate_notebook.py 04_llm/00_transformer_from_scratch.ipynb`
Expected: `OK ... (16 cells, 9 code)`

- [ ] **Step 2: Commit**

```bash
git add 04_llm/00_transformer_from_scratch.ipynb
git commit -m "feat(module04): notebook 00 transformer from scratch (bangun+latih tiny GPT)"
```

### ✅ CHECKPOINT A (Colab) — user-run

- [ ] Push: `git push origin master`
- [ ] User buka `colab.research.google.com/github/chmdznr/navasena-gen-ml-course/blob/master/04_llm/00_transformer_from_scratch.ipynb`, **Run all** di T4 biasa.
- [ ] Verifikasi: loss train turun (mis. dari ~4 ke <2), output "sesudah" jelas lebih mirip teks Indonesia daripada "sebelum". Lapor bila error/aneh; perbaiki sebelum lanjut Phase B.

---

## PHASE B — `01_llm_basics.ipynb` (PAKAI, rombak total)

Overwrite file lama. Anchor **Qwen2.5-3B-Instruct** fp16.

### Task B1: Header + setup + load model

**Files:** Modify (overwrite) `04_llm/01_llm_basics.ipynb`

- [ ] **Step 1: Buat sel**

`md` `c01-00`:
```markdown
# 📘 Module 04 · Notebook 01 — Memakai LLM (Instruct Models)

Setelah membangun Transformer dari nol (notebook 00), sekarang kita **pakai model instruct
yang sudah dilatih** dan kapabel: **Qwen2.5-3B-Instruct** — open (non-gated), multilingual
(termasuk Indonesia), dan muat nyaman di Colab T4.

## Apa yang akan kita pelajari?
1. Load model instruct dengan benar (`float16`, `device_map`)
2. **Chat template** & system prompt (cara model chat sebenarnya dipanggil)
3. Parameter generasi (temperature, top_p, repetition_penalty)
4. Prompt engineering: zero-shot & few-shot
5. Klasifikasi teks: head terlatih vs acak + **zero-shot**
6. Chatbot sederhana (Gradio)
```

`code` `c01-01` (install — PIN penting):
```python
# PENTING: pin transformers < 5. transformers v5 memecah API lama (apply_chat_template)
# & tokenizer model kecil. Jika sempat memakai v5: Runtime -> Restart, lalu Run all.
!pip install -q "transformers>=4.44,<5" accelerate gradio
```

`code` `c01-02` (GPU + helper):
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

def gpu_mem(tag=""):
    if not torch.cuda.is_available():
        return
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    print(f"[{tag}] allocated={a:.2f} GB | reserved={r:.2f} GB")
```

`md` `c01-03`:
```markdown
## 1. Load Model Instruct dengan Benar

`Qwen2.5-3B-Instruct` bobotnya bf16; di T4 (tak ada bf16) kita load `torch_dtype=torch.float16`.
`device_map="auto"` menaruh model di GPU. Footprint ~6 GB bobot → muat dengan banyak sisa.

> Runtime lambat? Ganti ke `Qwen/Qwen2.5-1.5B-Instruct` (lebih ringan, ~3 GB).
```

`code` `c01-04` (load):
```python
model_name = "Qwen/Qwen2.5-3B-Instruct"   # fallback ringan: "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,   # T4 = tanpa bf16 -> pakai float16
    device_map="auto",
)
gpu_mem("setelah load Qwen2.5-3B")
```

- [ ] **Step 2: Validate** → `OK`

### Task B2: Chat template + helper chat

- [ ] **Step 1: Tambah sel**

`md` `c01-05`:
```markdown
## 2. Chat Template & System Prompt

Model chat dilatih dengan format khusus (Qwen pakai **ChatML**: `<|im_start|>`). Jangan susun
string mentah — pakai `tokenizer.apply_chat_template` dengan peran `system`/`user`.
`return_dict=True` + `generate(**inputs)` = pola robust (bawa `attention_mask`, anti error versi baru).
```

`code` `c01-06` (chat helper + demo):
```python
def chat(user_msg, system_msg="You are a helpful assistant.", max_new_tokens=256,
         temperature=0.7, top_p=0.9):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature and temperature > 0:                # sampling
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:                                              # greedy (deterministik)
        gen_kwargs.update(do_sample=False)
    out = model.generate(**inputs, **gen_kwargs)
    # ambil hanya token baru (setelah prompt)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)

print(chat("Jelaskan apa itu large language model dalam 3 kalimat."))
```

- [ ] **Step 2: Validate** → `OK`

### Task B3: Parameter generasi + prompt engineering

- [ ] **Step 1: Tambah sel**

`md` `c01-07`:
```markdown
## 3. Parameter Generasi

- `temperature` rendah → fokus/deterministik; tinggi → kreatif/acak.
- `top_p` (nucleus) → ambil token dari massa probabilitas teratas.
- `repetition_penalty` → kurangi pengulangan.
- `max_new_tokens` → jumlah token BARU (bukan total panjang).
```

`code` `c01-08`:
```python
prompt = "Tulis satu kalimat motivasi untuk peserta bootcamp AI."
print("--- temperature 0.2 (fokus) ---")
print(chat(prompt, temperature=0.2))
print("\n--- temperature 1.0 (kreatif) ---")
print(chat(prompt, temperature=1.0))
```

`md` `c01-09`:
```markdown
## 4. Prompt Engineering: Zero-shot & Few-shot

**Zero-shot**: minta langsung. **Few-shot**: beri beberapa contoh pola di prompt; model meniru
tanpa dilatih ulang. Dengan model kapabel, few-shot kini benar-benar mengikuti pola.
```

`code` `c01-10`:
```python
# Zero-shot
print("ZERO-SHOT:")
print(chat("Apa ibu kota Italia? Jawab singkat."))

# Few-shot: tunjukkan pola Q/A, model melanjutkan
few_shot = (
    "Question: What is the capital of France?\nAnswer: Paris.\n"
    "Question: What is the capital of Japan?\nAnswer: Tokyo.\n"
    "Question: What is the capital of Italy?\nAnswer:"
)
print("\nFEW-SHOT:")
print(chat(few_shot, system_msg="Continue the pattern. Answer with only the city name.",
           temperature=0.0))
```

- [ ] **Step 2: Validate** → `OK`

### Task B4: Klasifikasi (terlatih vs acak + zero-shot)

- [ ] **Step 1: Tambah sel**

`md` `c01-11`:
```markdown
## 5. Klasifikasi Teks

Kita bandingkan tiga pendekatan:
1. **`bert-tiny`** — head klasifikasi **acak** (belum dilatih) → `LABEL_0/LABEL_1` ~0.5.
2. **`distilbert-sst2`** — head **terlatih** untuk sentimen → `POSITIVE/NEGATIVE` yakin.
3. **`bart-large-mnli`** — **zero-shot**: kita beri label custom sendiri, satu model untuk apa saja.

Variabel yang berubah dari (1) ke (2): hanya **head-nya dilatih**.
```

`code` `c01-12`:
```python
texts = [
    "I love working with language models!",
    "This task is quite challenging.",
    "The results are impressive and amazing.",
]

# (1) bert-tiny: head acak -> LABEL_0/LABEL_1 ~0.5 (non-deterministik)
clf_untrained = pipeline("sentiment-analysis", model="prajjwal1/bert-tiny",
                         device=0 if torch.cuda.is_available() else -1)
# (2) distilbert SST-2: head terlatih -> POSITIVE/NEGATIVE
clf_trained = pipeline("sentiment-analysis",
                       model="distilbert-base-uncased-finetuned-sst-2-english",
                       device=0 if torch.cuda.is_available() else -1)

for label, clf in [("ACAK (bert-tiny)", clf_untrained), ("TERLATIH (distilbert)", clf_trained)]:
    print(f"\n=== {label} ===")
    for t in texts:
        r = clf(t)[0]
        print(f"  {r['label']:<9} {r['score']:.3f} | {t}")
```

`code` `c01-13` (zero-shot):
```python
# (3) Zero-shot: label ditentukan SAAT inferensi, bukan saat training
zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",
               device=0 if torch.cuda.is_available() else -1)
labels = ["teknologi", "olahraga", "kuliner", "politik"]
for t in ["Timnas menang 3-0 tadi malam.",
          "GPU baru ini mempercepat training model AI.",
          "Resep rendang autentik dari Padang."]:
    r = zsc(t, candidate_labels=labels)
    print(f"{r['labels'][0]:<10} {r['scores'][0]:.3f} | {t}")
```

- [ ] **Step 2: Validate** → `OK`

### Task B5: Chatbot Gradio + ringkasan + commit

- [ ] **Step 1: Tambah sel**

`md` `c01-14`:
```markdown
## 6. Chatbot Sederhana (Gradio)

`gr.ChatInterface` membuat UI chat siap pakai. Kini balasannya **koheren** karena memakai
Qwen2.5-3B. Format `messages` (role/content) adalah standar modern.
```

`code` `c01-15`:
```python
import gradio as gr

def respond(message, history):
    # history: list of {"role","content"} (format messages)
    msgs = [{"role": "system", "content": "You are a concise, helpful assistant. Reply in the user's language."}]
    msgs += history
    msgs.append({"role": "user", "content": message})
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                         temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

demo = gr.ChatInterface(respond, type="messages", title="Qwen2.5 Chatbot")
demo.launch(share=True)
```

`md` `c01-16`:
```markdown
## Ringkasan & Jembatan

| # | Topik | Inti |
|---|-------|------|
| 1 | Load instruct model | `float16` + `device_map` di T4 |
| 2 | Chat template | `apply_chat_template`, system prompt, ChatML |
| 3 | Parameter generasi | temperature/top_p/repetition_penalty |
| 4 | Prompt engineering | zero-shot & few-shot |
| 5 | Klasifikasi | head terlatih vs acak + zero-shot |
| 6 | Chatbot | Gradio `ChatInterface` |

Tapi: model **7B tak muat fp16** di T4, generasi terasa **lambat**, dan output muncul sekaligus
(tanpa **streaming**). **Notebook 02** menyelesaikan ini dengan teknik produksi.
```

- [ ] **Step 2: Validate** → `OK ... (17 cells, 9 code)`

- [ ] **Step 3: Commit**

```bash
git add 04_llm/01_llm_basics.ipynb
git commit -m "feat(module04): rombak 01_llm_basics ke Qwen2.5-3B (prompting, klasifikasi, chatbot)"
```

### ✅ CHECKPOINT B (Colab) — user-run

- [ ] Push, buka di Colab T4, Run all.
- [ ] Verifikasi: Qwen2.5-3B load tanpa OOM (`gpu_mem` ~6-8 GB); jawaban koheren (termasuk Indonesia); few-shot menjawab "Roma/Rome"; klasifikasi menunjukkan kontras acak vs terlatih; zero-shot memberi label benar; Gradio muncul link & balas koheren. Lapor bila ada masalah sebelum Phase C.

---

## PHASE C — `02_llm_production.ipynb` (SKALAKAN)

Anchor **Mistral-7B-Instruct-v0.3** (4-bit).

### Task C1: Header + setup

**Files:** Create `04_llm/02_llm_production.ipynb`

- [ ] **Step 1: Buat sel**

`md` `c02-00`:
```markdown
# 📙 Module 04 · Notebook 02 — LLM untuk Produksi (di T4 gratis)

Model 7B tak muat `float16` di T4. Di sini kita pakai **teknik produksi** agar tetap jalan:
**4-bit quantization**, **memory profiling**, **batching**, dan **streaming**.

## Apa yang akan kita pelajari?
1. Kenapa 7B fp16 OOM di T4 (perhitungan)
2. **Quantization 4-bit** (bitsandbytes) → jalankan 7B di T4
3. **GPU memory profiling**
4. **Batching** untuk throughput
5. **Streaming** token
6. Jembatan ke RAG (Module 05)
```

`code` `c02-01` (install):
```python
# Pin transformers < 5 WAJIB: v5 punya regresi OOM bnb 4-bit (HF #43032).
!pip install -q "transformers>=4.44,<5" accelerate bitsandbytes
```

`code` `c02-02` (GPU + helper):
```python
import torch, time
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TextStreamer, TextIteratorStreamer)

def gpu_mem(tag=""):
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    print(f"[{tag}] allocated={a:.2f} GB | reserved={r:.2f} GB")

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

- [ ] **Step 2: Validate** → `OK`

### Task C2: OOM explanation + 4-bit load

- [ ] **Step 1: Tambah sel**

`md` `c02-03`:
```markdown
## 1. Kenapa 7B `float16` OOM di T4

Bobot saja = **7.25 miliar × 2 byte ≈ 14.5 GB**, plus KV-cache & aktivasi ≈ **~16 GB** →
melebihi 16 GB T4. (Kita **tidak** memicu OOM sungguhan — itu bisa merusak runtime; cukup hitung.)
```

`code` `c02-04` (perhitungan, tanpa load fp16):
```python
params_b = 7.25
for dtype, bytes_per in [("float16", 2), ("4-bit NF4", 0.5)]:
    gb = params_b * 1e9 * bytes_per / 1e9
    print(f"{dtype:10}: bobot ~{gb:.1f} GB  -> {'OOM di T4 (16GB)' if gb > 13 else 'muat di T4'}")
```

`md` `c02-05`:
```markdown
## 2. Quantization 4-bit (bitsandbytes)

NF4 menyimpan bobot dalam 4-bit (~1/4 memori). **Di T4 `bnb_4bit_compute_dtype` WAJIB
`torch.float16`** — `bfloat16` CRASH (T4 compute 7.5 < 8.0). Footprint ~5.5 GB → muat nyaman.
```

`code` `c02-06` (load 4-bit):
```python
prod_model = "mistralai/Mistral-7B-Instruct-v0.3"   # Apache-2.0, non-gated

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,   # WAJIB fp16 di T4 (bf16 crash)
)
tok = AutoTokenizer.from_pretrained(prod_model)
m4 = AutoModelForCausalLM.from_pretrained(
    prod_model, quantization_config=bnb_config, device_map="auto",
)
gpu_mem("Mistral-7B 4-bit")

messages = [{"role": "user", "content": "Sebutkan 3 manfaat AI untuk UMKM, singkat."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to(m4.device)
out = m4.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7,
                  pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

- [ ] **Step 2: Validate** → `OK`

### Task C3: Memory profiling + batching

- [ ] **Step 1: Tambah sel**

`md` `c02-07`:
```markdown
## 3. GPU Memory Profiling

`torch.cuda.memory_allocated` (terpakai sekarang) vs `memory_reserved` (di-cache PyTorch) vs
`max_memory_allocated` (puncak). Higiene: `del model; torch.cuda.empty_cache()` saat ganti model.
```

`code` `c02-08`:
```python
print(f"peak allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
gpu_mem("sekarang")
# contoh higiene (jangan jalankan kalau masih butuh m4):
# del m4; torch.cuda.empty_cache(); gpu_mem("setelah dibebaskan")
```

`md` `c02-09`:
```markdown
## 4. Batching untuk Throughput

Memproses banyak prompt sekaligus jauh lebih efisien daripada satu-satu. Untuk decoder-only,
WAJIB `padding_side="left"` agar posisi generasi benar.
```

`code` `c02-10`:
```python
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

prompts = [
    "Terjemahkan ke Inggris: Selamat pagi.",
    "Apa kepanjangan dari CPU?",
    "Beri satu tips belajar coding.",
]
batch_msgs = [[{"role": "user", "content": p}] for p in prompts]
texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
         for m in batch_msgs]
enc = tok(texts, return_tensors="pt", padding=True).to(m4.device)

t0 = time.time()
out = m4.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
dt = time.time() - t0
for i, p in enumerate(prompts):
    gen = out[i][enc["input_ids"].shape[1]:]
    print(f"\nQ: {p}\nA: {tok.decode(gen, skip_special_tokens=True).strip()}")
print(f"\n{len(prompts)} prompt dalam 1 batch: {dt:.1f}s")
```

- [ ] **Step 2: Validate** → `OK`

### Task C4: Streaming + jembatan + commit

- [ ] **Step 1: Tambah sel**

`md` `c02-11`:
```markdown
## 5. Streaming Token

Daripada menunggu seluruh jawaban, **streaming** menampilkan token saat diproduksi (UX lebih
baik). `TextStreamer` mencetak ke output; `TextIteratorStreamer` (threaded) untuk aplikasi.
```

`code` `c02-12`:
```python
msg = [{"role": "user", "content": "Ceritakan singkat sejarah Transformer dalam NLP."}]
inp = tok.apply_chat_template(msg, add_generation_prompt=True,
                              return_tensors="pt", return_dict=True).to(m4.device)
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
_ = m4.generate(**inp, max_new_tokens=200, do_sample=True, temperature=0.7,
                pad_token_id=tok.eos_token_id, streamer=streamer)
```

`code` `c02-13` (TextIteratorStreamer, threaded):
```python
from threading import Thread

it_streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
gen_kwargs = dict(**inp, max_new_tokens=120, do_sample=True, temperature=0.7,
                  pad_token_id=tok.eos_token_id, streamer=it_streamer)
Thread(target=m4.generate, kwargs=gen_kwargs).start()
for token in it_streamer:        # cocok untuk app/web: konsumsi token saat datang
    print(token, end="", flush=True)
print()
```

`md` `c02-14`:
```markdown
## Ringkasan & Jembatan ke RAG (Module 05)

| Teknik | Gunanya |
|--------|---------|
| Quantization 4-bit | jalankan 7B di GPU kecil |
| Memory profiling | ukur & jaga footprint |
| Batching | throughput lebih tinggi |
| Streaming | UX responsif |

Tapi model tetap bisa **berhalusinasi** dan punya **knowledge cutoff** — ia tak tahu data
terbaru/privat kita. **Module 05 (RAG)** memberi LLM "buku contekan": mengambil dokumen relevan
lalu menyuntikkannya ke prompt agar jawaban akurat & terkini.
```

- [ ] **Step 2: Validate** → `OK ... (15 cells, 8 code)`

- [ ] **Step 3: Commit**

```bash
git add 04_llm/02_llm_production.ipynb
git commit -m "feat(module04): notebook 02 produksi (4-bit quant, profiling, batching, streaming)"
```

### ✅ CHECKPOINT C (Colab) — user-run

- [ ] Push, buka di Colab T4, Run all.
- [ ] Verifikasi: Mistral-7B 4-bit load tanpa OOM (`gpu_mem` ~5.5 GB); generate koheren; profiling tampil; batching menjalankan 3 prompt sekaligus; streaming mencetak token bertahap. Lapor bila ada masalah.

---

## Penutup (setelah ketiga checkpoint hijau)

- [ ] Update README/CLAUDE.md bila perlu menyebut struktur 3-notebook (opsional).
- [ ] Aset pendukung (slides/quiz/cheatsheet) diselaraskan di sesi terpisah (di luar plan ini).
- [ ] Google Drive sync HANYA setelah user konfirmasi ketiga notebook lulus run penuh.

---

## Self-Review (diisi penulis plan)

**Spec coverage:** §8.1→Phase A, §8.2→Phase B, §8.3→Phase C, §6 guardrail→tertanam di tiap sel
(pin<5, compute_dtype fp16, return_dict, padding_side left, gpu_mem). §9 urutan & §10 validasi→
checkpoint Colab tiap fase. §11 risiko OOM→Task C2 tidak memicu OOM nyata. ✔ tidak ada gap.

**Placeholder scan:** korpus notebook 00 = teks nyata yang di-embed (boleh diperbanyak, bukan
TODO). Tidak ada "TBD/handle errors/dst". ✔

**Type consistency:** `gpu_mem()` sama di 01 & 02; `chat()` (01) konsisten; `Head/MultiHeadAttention/
Block/TinyGPT` (00) saling rujuk benar; `apply_chat_template(...return_dict=True)` + `generate(**inputs)`
konsisten di 01 & 02. ✔
