# Module 02 Generative AI Material Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menambah materi survey generative AI ke Module 02 — notebook hands-on (autoencoder penuh + DCGAN mini), Act 6 di slide deck, kartu cheatsheet, dan 3 soal quiz.

**Architecture:** Satu notebook baru `06_generative_ai_intro.ipynb` (TF/Keras, MNIST, Colab T4, ~15 menit) dengan arc "dari mengenali → mencipta". Figure deck dihasilkan dari training NYATA yang dijalankan sekali secara lokal (docling env punya TensorFlow), disimpan sebagai PDF statis — build.sh tidak melatih ulang. Artefak turunan (deck/cheatsheet/quiz) di-update mengikuti pola dan pipeline yang sudah terbentuk.

**Tech Stack:** TensorFlow/Keras, matplotlib (dark theme), mermaid (mmdc), XeLaTeX Beamer, headless Chrome (cheatsheet PDF), puppeteer (uji quiz).

**Spec:** `docs/superpowers/specs/2026-06-10-module02-generative-ai-design.md`

**Kaidah bahasa (berlaku untuk SEMUA teks baru):** Bahasa Indonesia engineering yang luwes — istilah serapan umum TIDAK diterjemahkan (training, layer, latent space, noise, epoch, loss, generator, discriminator, sampling, output). Kalimat tidak boleh ambigu/aneh didengar. DILARANG: "ruang laten", "kebisingan", "pembangkit", "pembeda", "pelatihan berlawanan".

**Env & commands penting:**
- Python: `PY=/Users/chmdznr/work/kemendag/sip/docling/bin/python` (tensorflow, torch, sklearn, matplotlib tersedia)
- Mermaid: `/opt/homebrew/bin/mmdc -i in.mmd -o out.png -s 3 -b transparent`
- Chrome headless: `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer --print-to-pdf=...`
- LaTeX: 2× `xelatex -interaction=nonstopmode -halt-on-error` (hapus `*.aux`/`*.log` dulu via `find . -maxdepth 1 -name "*.aux" -delete`)
- Hook `latex-overfull-guard.sh` aktif: overfull ≥2pt atau vbox overflow akan di-block.

---

### Task 1: Notebook `06_generative_ai_intro.ipynb`

**Files:**
- Create: `02_deep_learning_fundamentals/06_generative_ai_intro.ipynb`
- Create (sementara): `/tmp/build_nb06.py` (builder, tidak di-commit)

- [ ] **Step 1: Tulis builder script `/tmp/build_nb06.py`**

Builder membuat notebook dari nol (file baru — tidak perlu guard hash). Struktur cell berurutan seperti di bawah. Helper:

```python
import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.split("\n")}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.split("\n")}

cells = []
```

Isi cell (urutan WAJIB seperti ini). Teks markdown di bawah adalah konten final — boleh dipoles gaya minor, tapi istilah teknis dan struktur dipertahankan:

```python
# --- Cell 1 (md): judul + bridge ---
cells.append(md("""# Generative AI --- Saat Neural Network Mencipta, Bukan Hanya Mengenali

Sejauh ini di Module 02, semua model yang kamu bangun bersifat **diskriminatif**: diberi input, model menebak label atau angka.

- Notebook 01--02: mengklasifikasi gambar fashion ke 10 kategori
- Notebook 03: membedakan kucing vs anjing
- Notebook 04: memprediksi nilai berikutnya dari data berurutan

Di notebook ini kamu akan ketemu keluarga model yang berbeda: model **generatif** --- model yang *menghasilkan data baru*, bukan sekadar menebak label. Inilah fondasi dari teknologi seperti ChatGPT dan Stable Diffusion.

**Yang akan kamu lakukan:**
1. Melatih **autoencoder**: jaringan yang belajar memampatkan dan merekonstruksi gambar
2. Menjelajah **latent space**: melihat digit "bermorfing" dari satu angka ke angka lain
3. Melatih **GAN (DCGAN) mini**: melihat digit MNIST "lahir" dari noise acak
4. Memahami peta dunia generatif modern: VAE, diffusion, dan GPT

> **Prasyarat**: jalankan di Google Colab dengan GPU runtime (Runtime > Change runtime type > T4 GPU). Total waktu training sekitar 15 menit."""))

# --- Cell 2 (code): cek GPU ---
cells.append(code("!nvidia-smi"))

# --- Cell 3 (code): import + seed ---
cells.append(code("""import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)
print("GPU terdeteksi:", tf.config.list_physical_devices('GPU'))

# Seed di-fix supaya hasil kamu sama dengan yang dibahas di materi
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)"""))

# --- Cell 4 (md): Bagian A intro ---
cells.append(md("""## Bagian A --- Autoencoder: Belajar Memampatkan, Lalu Mencipta Ulang

Autoencoder adalah neural network yang tugasnya terdengar aneh: **merekonstruksi inputnya sendiri**. Strukturnya seperti jam pasir:

- **Encoder**: memampatkan gambar 28x28 (784 angka) menjadi vektor kecil, misalnya 32 angka
- **Latent space**: "ruang ringkasan" berukuran 32 dimensi itu --- esensi dari gambar
- **Decoder**: membangun ulang gambar 28x28 dari 32 angka tadi

Analogi: kamu memfoto sebuah lukisan, lalu mendeskripsikannya dalam satu kalimat (encoder), kemudian temanmu melukis ulang hanya dari kalimat itu (decoder). Kalau hasil lukisannya mirip, berarti kalimat ringkasannya bagus.

Kenapa ini penting untuk generative AI? Karena **decoder pada dasarnya adalah generator**: dia mengubah vektor angka menjadi gambar."""))

# --- Cell 5 (code): load MNIST ---
cells.append(code("""# MNIST: 70.000 gambar digit tulisan tangan 28x28 grayscale
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalisasi ke 0.0-1.0 (pola yang sama seperti notebook 01)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Data train:", x_train.shape)
print("Data test :", x_test.shape)"""))

# --- Cell 6 (code): lihat sample ---
cells.append(code("""fig, axes = plt.subplots(1, 10, figsize=(14, 2))
for i, ax in enumerate(axes):
    ax.imshow(x_train[i], cmap="gray")
    ax.set_title(y_train[i])
    ax.axis("off")
plt.suptitle("10 sample digit MNIST")
plt.show()"""))

# --- Cell 7 (md): arsitektur AE ---
cells.append(md("""### Membangun Autoencoder

Kita pakai `Dense` layer biasa (seperti notebook 01) supaya fokusnya ke *konsep*, bukan arsitektur:

- Encoder: 784 -> 128 -> **32** (latent)
- Decoder: 32 -> 128 -> 784 -> reshape ke 28x28

Loss-nya `binary_crossentropy` per pixel: seberapa beda gambar rekonstruksi dengan gambar asli. Perhatikan: **tidak ada label sama sekali** --- targetnya adalah inputnya sendiri (`x_train, x_train`). Ini disebut *self-supervised*."""))

# --- Cell 8 (code): build AE ---
cells.append(code("""LATENT_DIM = 32

encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(LATENT_DIM, activation="relu"),
], name="encoder")

decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(LATENT_DIM,)),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape((28, 28)),
], name="decoder")

autoencoder = tf.keras.Sequential([encoder, decoder], name="autoencoder")
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()"""))

# --- Cell 9 (code): train AE ---
cells.append(code("""EPOCHS_AE = 10  # ~2-3 menit di T4

history = autoencoder.fit(
    x_train, x_train,          # input = target: rekonstruksi diri sendiri
    epochs=EPOCHS_AE,
    batch_size=256,
    validation_data=(x_test, x_test),
)"""))

# --- Cell 10 (md): rekonstruksi ---
cells.append(md("""### Seberapa Bagus Rekonstruksinya?

Baris atas = gambar asli, baris bawah = hasil rekonstruksi setelah dipampatkan jadi 32 angka. Detailnya sedikit hilang (wajar --- kompresi 784 jadi 32, sekitar 24x lebih kecil), tapi digitnya masih jelas terbaca."""))

# --- Cell 11 (code): plot rekonstruksi ---
cells.append(code("""recon = autoencoder.predict(x_test[:10])

fig, axes = plt.subplots(2, 10, figsize=(14, 3))
for i in range(10):
    axes[0, i].imshow(x_test[i], cmap="gray"); axes[0, i].axis("off")
    axes[1, i].imshow(recon[i], cmap="gray"); axes[1, i].axis("off")
axes[0, 0].set_ylabel("Asli", rotation=0, labelpad=30)
axes[1, 0].set_ylabel("Rekonstruksi", rotation=0, labelpad=30)
plt.suptitle("Atas: asli | Bawah: rekonstruksi dari 32 angka latent")
plt.show()"""))

# --- Cell 12 (md): interpolasi latent ---
cells.append(md("""### Menjelajah Latent Space: Morphing Antar Digit

Ini bagian serunya. Setiap gambar kini punya "alamat" berupa 32 angka di latent space. Kalau kita ambil alamat digit A dan alamat digit B, lalu berjalan pelan-pelan dari A ke B *di dalam latent space* sambil men-decode setiap langkahnya --- kita akan melihat digit A **bermorfing** menjadi digit B.

Ini bukti bahwa latent space bukan sekadar kompresi: dia menyimpan *struktur* --- titik-titik di antara dua digit valid juga menghasilkan gambar yang masuk akal."""))

# --- Cell 13 (code): interpolasi ---
cells.append(code("""# Ambil satu digit '7' dan satu digit '2' dari test set
idx_a = int(np.where(y_test == 7)[0][0])
idx_b = int(np.where(y_test == 2)[0][0])

z_a = encoder(x_test[idx_a:idx_a + 1]).numpy()
z_b = encoder(x_test[idx_b:idx_b + 1]).numpy()

STEPS = 10
fig, axes = plt.subplots(1, STEPS, figsize=(15, 2))
for i, t in enumerate(np.linspace(0.0, 1.0, STEPS)):
    z = (1 - t) * z_a + t * z_b      # campuran linear dua "alamat"
    img = decoder(z).numpy()[0]
    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")
plt.suptitle("Interpolasi latent space: 7 bermorfing menjadi 2")
plt.show()"""))

# --- Cell 14 (md): sampling acak -> motivasi GAN ---
cells.append(md("""### Bisakah Kita Mencipta Digit Baru dari Nol?

Logikanya: kalau decoder bisa mengubah 32 angka menjadi gambar, kita tinggal kasih 32 angka **acak** dan dapat digit baru, kan? Coba kita buktikan."""))

# --- Cell 15 (code): random sampling AE ---
cells.append(code("""z_random = np.abs(np.random.normal(size=(10, LATENT_DIM))).astype("float32") * 3.0
imgs = decoder(z_random).numpy()

fig, axes = plt.subplots(1, 10, figsize=(14, 2))
for i, ax in enumerate(axes):
    ax.imshow(imgs[i], cmap="gray")
    ax.axis("off")
plt.suptitle("Decode dari titik acak di latent space: hasilnya tidak meyakinkan")
plt.show()"""))

# --- Cell 16 (md): kenapa gagal -> GAN ---
cells.append(md("""Hasilnya kabur dan banyak yang tidak berbentuk digit. Kenapa?

Autoencoder hanya dilatih untuk merekonstruksi gambar yang *ada* --- dia tidak pernah diminta memastikan bahwa **setiap** titik di latent space menghasilkan gambar yang bagus. Titik acak yang kita pilih kemungkinan besar jatuh di "daerah kosong" yang tidak pernah dilewati data training.

Untuk benar-benar *mencipta*, kita butuh model yang sejak awal dilatih menghasilkan data baru dari noise. Masuklah: **GAN**."""))

# --- Cell 17 (md): Bagian B intro GAN ---
cells.append(md("""## Bagian B --- GAN: Dua Jaringan yang Saling Berlomba

**GAN (Generative Adversarial Network)** terdiri dari dua jaringan yang dilatih sebagai lawan tanding:

- **Generator** = pemalsu lukisan: mengubah noise acak menjadi gambar palsu, berusaha menipu
- **Discriminator** = kurator galeri: melihat gambar dan menebak --- asli (dari dataset) atau palsu (dari generator)?

Keduanya dilatih bergantian. Setiap kali discriminator makin jago membedakan, generator dipaksa membuat pemalsuan yang lebih meyakinkan --- dan sebaliknya. Hasil akhir perlombaan ini: generator yang bisa menghasilkan gambar yang nyaris tidak bisa dibedakan dari asli.

Versi yang kita pakai adalah **DCGAN** (Deep Convolutional GAN): generator memakai `Conv2DTranspose` (kebalikan convolution --- memperbesar gambar), discriminator memakai `Conv2D` biasa seperti CNN di notebook 03."""))

# --- Cell 18 (code): siapkan data GAN ---
cells.append(code("""# GAN dengan output tanh bekerja di rentang -1..1, bukan 0..1
N_GAN = 20000  # subset supaya training ~10 menit di T4
x_gan = x_train[:N_GAN].reshape(-1, 28, 28, 1) * 2.0 - 1.0

BATCH_SIZE = 256
NOISE_DIM = 100

dataset = (tf.data.Dataset.from_tensor_slices(x_gan)
           .shuffle(N_GAN, seed=SEED)
           .batch(BATCH_SIZE))
print("Jumlah gambar untuk training GAN:", N_GAN)"""))

# --- Cell 19 (code): generator ---
cells.append(code("""def build_generator():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(7 * 7 * 128, use_bias=False, input_shape=(NOISE_DIM,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((7, 7, 128)),
        tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(1, 5, strides=2, padding="same",
                                        activation="tanh"),
    ], name="generator")

generator = build_generator()
generator.summary()"""))

# --- Cell 20 (code): discriminator ---
cells.append(code("""def build_discriminator():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 5, strides=2, padding="same",
                               input_shape=(28, 28, 1)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, 5, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1),   # logit: > 0 cenderung asli, < 0 cenderung palsu
    ], name="discriminator")

discriminator = build_discriminator()
discriminator.summary()"""))

# --- Cell 21 (md): training loop GAN ---
cells.append(md("""### Training Loop Adversarial

Beda dengan `model.fit()` biasa, GAN butuh training loop manual karena dua model dilatih bergantian dalam satu langkah:

1. Generator membuat gambar palsu dari noise
2. Discriminator menilai gambar asli DAN palsu; loss-nya = seberapa sering dia salah tebak
3. Generator di-update berdasarkan seberapa sukses dia menipu discriminator

Perhatikan dua optimizer terpisah --- masing-masing jaringan hanya meng-update weight miliknya sendiri."""))

# --- Cell 22 (code): train_step ---
cells.append(code("""bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_opt = tf.keras.optimizers.Adam(1e-4)
disc_opt = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(real_images):
    noise = tf.random.normal([tf.shape(real_images)[0], NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)

        real_logits = discriminator(real_images, training=True)
        fake_logits = discriminator(fake_images, training=True)

        # Discriminator: asli harus ditebak 1, palsu harus ditebak 0
        disc_loss = (bce(tf.ones_like(real_logits), real_logits) +
                     bce(tf.zeros_like(fake_logits), fake_logits))
        # Generator: sukses kalau yang palsu ditebak 1 (berhasil menipu)
        gen_loss = bce(tf.ones_like(fake_logits), fake_logits)

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    return gen_loss, disc_loss"""))

# --- Cell 23 (code): training loop + grid ---
cells.append(code("""EPOCHS_GAN = 15  # ~10 menit di T4; makin lama makin halus hasilnya

# Noise tetap: digit yang sama kita pantau perkembangannya antar epoch
fixed_noise = tf.random.normal([16, NOISE_DIM], seed=SEED)


def show_generated(epoch):
    imgs = generator(fixed_noise, training=False).numpy()
    imgs = (imgs + 1.0) / 2.0  # kembalikan dari -1..1 ke 0..1
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i, :, :, 0], cmap="gray")
        ax.axis("off")
    plt.suptitle(f"Hasil generator --- epoch {epoch}")
    plt.show()


for epoch in range(1, EPOCHS_GAN + 1):
    for batch in dataset:
        gen_loss, disc_loss = train_step(batch)
    print(f"Epoch {epoch:2d}/{EPOCHS_GAN} | gen_loss={gen_loss:.3f} "
          f"| disc_loss={disc_loss:.3f}")
    if epoch in (1, 5, 10, EPOCHS_GAN):
        show_generated(epoch)"""))

# --- Cell 24 (md): membaca hasil ---
cells.append(md("""### Membaca Hasilnya

Perhatikan progresnya: di epoch 1 hasilnya masih noise berbintik, lalu perlahan muncul coretan, dan di epoch 15 sebagian besar sudah berbentuk digit yang bisa dikenali --- **semuanya digit baru yang tidak ada di dataset**, lahir murni dari noise.

Dua catatan jujur tentang GAN:

- Hasil 15 epoch masih kasar. GAN "produksi" dilatih ratusan epoch dengan banyak trik kestabilan.
- GAN terkenal **rewel**: kadang generator menemukan satu-dua jenis digit yang selalu lolos, lalu berhenti belajar variasi lain (disebut *mode collapse*). Kalau hasil kamu didominasi digit yang itu-itu saja, kamu baru saja menyaksikannya sendiri.

Justru kerewelannya ini yang mendorong riset ke arsitektur generatif yang lebih stabil --- yang akan kita petakan sekarang."""))

# --- Cell 25 (md): Bagian C peta generatif ---
cells.append(md("""## Bagian C --- Peta Dunia Generative AI Modern

Autoencoder dan GAN adalah dua leluhur penting, tapi dunia generatif 2020-an dikuasai dua keluarga lain:

| Keluarga | Ide inti | Contoh terkenal |
|---|---|---|
| Autoencoder | Kompres lalu rekonstruksi; decoder = generator | Kompresi, denoising |
| **VAE** | Autoencoder yang latent space-nya DIATUR supaya bisa di-sampling | Generator wajah awal |
| **GAN** | Generator vs discriminator saling berlomba | Deepfake, StyleGAN |
| **Diffusion** | Belajar MEMBERSIHKAN noise bertahap; generate = denoise dari noise murni | Stable Diffusion, DALL-E, Midjourney |
| **Autoregressive Transformer** | Prediksi token berikutnya, berulang-ulang | GPT, ChatGPT, Llama |

Dua benang merah yang berlaku untuk semuanya:

1. **Semua generator belajar distribusi data** --- "seperti apa sih data yang wajar?" --- lalu sampling dari situ.
2. **Mencipta = transformasi dari sesuatu yang sederhana** (noise, atau teks sebelumnya) **menjadi data yang kompleks**, dipelajari dari jutaan contoh.

Yang menarik: ChatGPT pada dasarnya melakukan hal yang sama dengan DCGAN kamu barusan --- menghasilkan data baru yang meniru distribusi data training --- hanya saja datanya teks, dan arsitekturnya transformer."""))

# --- Cell 26 (md): penutup ---
cells.append(md("""## Ringkasan

Hari ini kamu sudah:

1. Melatih **autoencoder** dan membuktikan decoder bisa "melukis" dari 32 angka
2. Melihat **latent space** menyimpan struktur (interpolasi 7 -> 2 yang mulus)
3. Membuktikan kenapa AE biasa belum cukup untuk mencipta (sampling acak gagal)
4. Melatih **DCGAN** dan menyaksikan digit lahir dari noise lewat perlombaan generator vs discriminator
5. Memetakan keluarga generatif modern: VAE, diffusion, dan autoregressive transformer

**Menuju Module 04**: di sana kamu akan memakai model generatif untuk *teks* --- LLM seperti GPT. Bekal dari notebook ini: LLM juga generator yang belajar distribusi data; bedanya dia men-generate token demi token, bukan pixel."""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

out = "/Users/chmdznr/work/navasena/navasena-gen-ml-course/02_deep_learning_fundamentals/06_generative_ai_intro.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("OK:", out)
```

- [ ] **Step 2: Jalankan builder**

Run: `/Users/chmdznr/work/kemendag/sip/docling/bin/python /tmp/build_nb06.py`
Expected: `OK: .../06_generative_ai_intro.ipynb`

- [ ] **Step 3: Validasi JSON + syntax semua code cell**

```bash
PY=/Users/chmdznr/work/kemendag/sip/docling/bin/python
$PY - <<'EOF'
import json, ast
nb = json.load(open('02_deep_learning_fundamentals/06_generative_ai_intro.ipynb'))
n_code = 0
for c in nb['cells']:
    if c['cell_type'] == 'code':
        src = '\n'.join(c['source'])
        if src.strip().startswith('!'):
            continue
        ast.parse(src)
        n_code += 1
print(f"JSON valid, {n_code} code cell lolos syntax check")
EOF
```
Expected: `JSON valid, N code cell lolos syntax check` (tanpa exception)

- [ ] **Step 4: Smoke-run logika training secara lokal (CPU, versi mini)**

Gabungkan code cell jadi script dengan konstanta diperkecil, untuk membuktikan kode jalan end-to-end (bukan untuk kualitas hasil):

```bash
PY=/Users/chmdznr/work/kemendag/sip/docling/bin/python
$PY - <<'EOF'
import json, re
nb = json.load(open('02_deep_learning_fundamentals/06_generative_ai_intro.ipynb'))
srcs = ['\n'.join(c['source']) for c in nb['cells']
        if c['cell_type'] == 'code' and not '\n'.join(c['source']).strip().startswith('!')]
script = '\n\n'.join(srcs)
# Versi mini: 1 epoch, data kecil, tanpa window plot
script = script.replace('EPOCHS_AE = 10', 'EPOCHS_AE = 1')
script = script.replace('EPOCHS_GAN = 15', 'EPOCHS_GAN = 1')
script = script.replace('N_GAN = 20000', 'N_GAN = 2000')
script = ('import matplotlib; matplotlib.use("Agg")\n'
          'import matplotlib.pyplot as plt\nplt.show = lambda *a, **k: plt.close("all")\n'
          + script)
exec(script, {'__name__': '__main__'})
print('SMOKE RUN OK')
EOF
```
Expected: training AE 1 epoch + GAN 1 epoch jalan, output diakhiri `SMOKE RUN OK`. Boleh lambat (CPU) — tunggu sampai selesai (timeout Bash 600000 ms).

- [ ] **Step 5: Cek kaidah bahasa**

```bash
grep -c -iE 'ruang laten|kebisingan|pembangkit|pelatihan berlawanan|jaringan pembeda' \
  02_deep_learning_fundamentals/06_generative_ai_intro.ipynb || echo "0 anti-pattern"
```
Expected: `0 anti-pattern` (grep tidak menemukan apa pun)

- [ ] **Step 6: Review akurasi + gaya via subagent**

Dispatch satu subagent reviewer dengan prompt: "Review notebook `02_deep_learning_fundamentals/06_generative_ai_intro.ipynb` untuk (a) akurasi teknis klaim ML/GAN/diffusion, (b) konsistensi narasi dengan notebook M02 lain (sapaan 'kamu', analogi), (c) kaidah bahasa: istilah serapan umum tidak diterjemahkan, kalimat tidak ambigu. Laporkan temuan per cell, JANGAN mengedit." Terapkan perbaikan yang valid via Edit/NotebookEdit.

- [ ] **Step 7: Commit**

```bash
git add 02_deep_learning_fundamentals/06_generative_ai_intro.ipynb
git commit -m "feat(module02): add generative AI intro notebook (autoencoder + DCGAN survey)"
```

---

### Task 2: Figure deck dari training nyata

**Files:**
- Create: `02_deep_learning_fundamentals/slides/figures/train_generative_figs.py` (prefix `train_`, BUKAN `gen_`, supaya `build.sh` tidak melatih ulang saat build)
- Create (output): `slides/figures/latent_interp.pdf`, `slides/figures/gan_grid.pdf`
- Create: `02_deep_learning_fundamentals/slides/figures/gan_architecture.mmd` → `gan_architecture.png`

- [ ] **Step 1: Tulis `train_generative_figs.py`**

Arsitektur SAMA dengan notebook (supaya figure jujur merepresentasikan materi). Tema gelap mengikuti pola `gen_activations.py` (bg `#1A1A2E`, aksen `#76B900`):

```python
"""
Generate latent_interp.pdf dan gan_grid.pdf dari training NYATA
(AE 10 epoch + DCGAN 15 epoch di MNIST subset). Dijalankan SEKALI saat
development (CPU ~15-25 menit); output PDF di-commit. build.sh sengaja
tidak menjalankan file ini (prefix train_, bukan gen_).
    PYTHON=<docling-env> python3 figures/train_generative_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BG, FG, ACCENT = "#1A1A2E", "#EEEEEE", "#76B900"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x01 = x_train.astype("float32") / 255.0
x01_test = x_test.astype("float32") / 255.0

# ---------- Autoencoder (identik dgn notebook) ----------
LATENT_DIM = 32
encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(LATENT_DIM, activation="relu"),
])
decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(LATENT_DIM,)),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape((28, 28)),
])
ae = tf.keras.Sequential([encoder, decoder])
ae.compile(optimizer="adam", loss="binary_crossentropy")
ae.fit(x01, x01, epochs=10, batch_size=256, verbose=2)

idx_a = int(np.where(y_test == 7)[0][0])
idx_b = int(np.where(y_test == 2)[0][0])
z_a = encoder(x01_test[idx_a:idx_a + 1]).numpy()
z_b = encoder(x01_test[idx_b:idx_b + 1]).numpy()

STEPS = 8
fig, axes = plt.subplots(1, STEPS, figsize=(10, 1.6))
fig.patch.set_facecolor(BG)
for i, t in enumerate(np.linspace(0, 1, STEPS)):
    img = decoder((1 - t) * z_a + t * z_b).numpy()[0]
    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")
fig.suptitle("Interpolasi latent space: 7 → 2 (autoencoder, MNIST)",
             color=FG, fontsize=10)
fig.savefig(os.path.join(OUT, "latent_interp.pdf"),
            facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("latent_interp.pdf OK")

# ---------- DCGAN (identik dgn notebook) ----------
N_GAN, BATCH_SIZE, NOISE_DIM, EPOCHS_GAN = 20000, 256, 100, 15
x_gan = x01[:N_GAN].reshape(-1, 28, 28, 1) * 2.0 - 1.0
dataset = (tf.data.Dataset.from_tensor_slices(x_gan)
           .shuffle(N_GAN, seed=SEED).batch(BATCH_SIZE))

generator = tf.keras.Sequential([
    tf.keras.layers.Dense(7 * 7 * 128, use_bias=False, input_shape=(NOISE_DIM,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Reshape((7, 7, 128)),
    tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(1, 5, strides=2, padding="same", activation="tanh"),
])
discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 5, strides=2, padding="same", input_shape=(28, 28, 1)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, 5, strides=2, padding="same"),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1),
])

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_opt, disc_opt = tf.keras.optimizers.Adam(1e-4), tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(real):
    noise = tf.random.normal([tf.shape(real)[0], NOISE_DIM])
    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake = generator(noise, training=True)
        rl, fl = discriminator(real, training=True), discriminator(fake, training=True)
        d_loss = bce(tf.ones_like(rl), rl) + bce(tf.zeros_like(fl), fl)
        g_loss = bce(tf.ones_like(fl), fl)
    gen_opt.apply_gradients(zip(gt.gradient(g_loss, generator.trainable_variables),
                                generator.trainable_variables))
    disc_opt.apply_gradients(zip(dt.gradient(d_loss, discriminator.trainable_variables),
                                 discriminator.trainable_variables))


fixed_noise = tf.random.normal([8, NOISE_DIM], seed=SEED)
snapshots = {}
for epoch in range(1, EPOCHS_GAN + 1):
    for batch in dataset:
        train_step(batch)
    print("epoch", epoch, "selesai")
    if epoch in (1, 5, EPOCHS_GAN):
        snapshots[epoch] = (generator(fixed_noise, training=False).numpy() + 1) / 2

fig, axes = plt.subplots(3, 8, figsize=(10, 4.2))
fig.patch.set_facecolor(BG)
for row, (ep, imgs) in enumerate(sorted(snapshots.items())):
    for col in range(8):
        axes[row, col].imshow(imgs[col, :, :, 0], cmap="gray")
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(f"epoch {ep}", color=ACCENT, fontsize=9,
                            rotation=0, labelpad=28, va="center")
    axes[row, 0].axis("on")
    axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
    for s in axes[row, 0].spines.values():
        s.set_visible(False)
fig.suptitle("DCGAN di MNIST: noise → digit (noise input sama tiap baris)",
             color=FG, fontsize=10)
fig.savefig(os.path.join(OUT, "gan_grid.pdf"), facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("gan_grid.pdf OK")
```

- [ ] **Step 2: Jalankan (background — bisa 15-25 menit di CPU)**

Run (background): `cd 02_deep_learning_fundamentals/slides && /Users/chmdznr/work/kemendag/sip/docling/bin/python figures/train_generative_figs.py`
Expected: `latent_interp.pdf OK` lalu `gan_grid.pdf OK`. Sambil menunggu, Task 3 (mermaid) boleh dikerjakan.

- [ ] **Step 3: Inspeksi visual kedua PDF**

Konversi cepat ke PNG dan lihat: digit interpolasi harus bermorfing mulus; baris epoch 15 di gan_grid harus menampilkan mayoritas bentuk menyerupai digit (kasar tidak apa-apa, asal bukan noise murni). Kalau epoch 15 masih noise murni → naikkan EPOCHS_GAN ke 25 dan ulangi.

- [ ] **Step 4: Tulis `gan_architecture.mmd`**

```
flowchart LR
    Z(["noise z<br/>(100 angka acak)"]) --> G["Generator<br/>Conv2DTranspose"]
    G --> F["Gambar palsu<br/>28x28"]
    R["Gambar asli<br/>MNIST"] --> D{"Discriminator<br/>Conv2D"}
    F --> D
    D --> O(["Asli atau palsu?"])
    O -. "feedback: makin jago menipu" .-> G
    O -. "feedback: makin jago membedakan" .-> D
    style Z fill:#2D2D44,stroke:#76B900,color:#EEEEEE
    style G fill:#2D2D44,stroke:#76B900,color:#A3D944
    style F fill:#2D2D44,stroke:#FF6F00,color:#EEEEEE
    style R fill:#2D2D44,stroke:#42A5F5,color:#EEEEEE
    style D fill:#2D2D44,stroke:#76B900,color:#A3D944
    style O fill:#2D2D44,stroke:#76B900,color:#EEEEEE
```

- [ ] **Step 5: Render mermaid + cek orientasi**

Run: `/opt/homebrew/bin/mmdc -i 02_deep_learning_fundamentals/slides/figures/gan_architecture.mmd -o 02_deep_learning_fundamentals/slides/figures/gan_architecture.png -s 3 -b transparent`
Lalu cek dimensi: `sips -g pixelWidth -g pixelHeight .../gan_architecture.png`.
GOTCHA terdokumentasi: kalau hasilnya portrait/tall (height > width), di LaTeX nanti WAJIB size by `height=` bukan `width=`.

- [ ] **Step 6: Commit**

```bash
git add 02_deep_learning_fundamentals/slides/figures/train_generative_figs.py \
        02_deep_learning_fundamentals/slides/figures/latent_interp.pdf \
        02_deep_learning_fundamentals/slides/figures/gan_grid.pdf \
        02_deep_learning_fundamentals/slides/figures/gan_architecture.mmd \
        02_deep_learning_fundamentals/slides/figures/gan_architecture.png
git commit -m "feat(module02): generative AI figures from real AE/DCGAN training runs"
```

---

### Task 3: Slide deck Act 6 "Generative AI"

**Files:**
- Modify: `02_deep_learning_fundamentals/slides/module02_slides.tex` — sisipkan Act 6 SETELAH frame "NVIDIA Ecosystem & TensorRT" (berakhir sebelum baris 855) dan SEBELUM frame "Ringkasan Modul 02" (baris ~855); edit frame Ringkasan.

- [ ] **Step 1: Sisipkan 5 frame Act 6**

Sisipkan blok berikut sebelum `\begin{frame}{Ringkasan Modul 02}`. Ikuti gotcha terdokumentasi: TikZ pakai `scale=` + `every node/.style={transform shape}`, nama node TANPA titik desimal, tabel sempit pakai `\setlength{\tabcolsep}{3pt}`:

```latex
% ============================================================
% ACT 6: GENERATIVE AI
% ============================================================
\acttitle{6}{Generative AI}{Saat neural network mencipta, bukan hanya mengenali}

\begin{frame}{Diskriminatif vs Generatif}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      {\color{nvgreen}\textbf{Diskriminatif (Act 1--5)}}
      \begin{itemize}
        \item Input data $\to$ output \textbf{label/angka}
        \item ``Gambar ini kucing atau anjing?''
        \item Belajar \textbf{batas} antar kelas
      \end{itemize}
    \end{column}
    \begin{column}{0.48\textwidth}
      {\color{nvorange}\textbf{Generatif (Act 6)}}
      \begin{itemize}
        \item Noise/konteks $\to$ output \textbf{data baru}
        \item ``Buatkan gambar digit yang meyakinkan''
        \item Belajar \textbf{distribusi} data itu sendiri
      \end{itemize}
    \end{column}
  \end{columns}
  \vspace{0.5em}
  \begin{center}
    {\color{nvgray}\small Fondasi ChatGPT \& Stable Diffusion --- dan jembatan kita menuju Module 04 (LLM)}
  \end{center}
\end{frame}

\begin{frame}{Autoencoder \& Latent Space}
  \begin{itemize}
    \item Encoder memampatkan 784 pixel $\to$ \textbf{32 angka} (latent); decoder merekonstruksi
    \item Decoder pada dasarnya = \textbf{generator}: vektor angka $\to$ gambar
    \item Latent space menyimpan \textbf{struktur}: titik di antara dua digit juga valid
  \end{itemize}
  \begin{center}
    \includegraphics[width=0.85\linewidth]{figures/latent_interp.pdf}
  \end{center}
  {\color{nvgray}\small Tapi: sampling \emph{acak} dari latent space AE hasilnya kabur --- AE tidak dilatih untuk mencipta. Kita butuh GAN.}
\end{frame}

\begin{frame}{GAN: Generator vs Discriminator}
  \begin{columns}[T]
    \begin{column}{0.52\textwidth}
      \begin{center}
        % kalau PNG portrait: ganti ke height=0.62\textheight
        \includegraphics[width=\linewidth]{figures/gan_architecture.png}
      \end{center}
    \end{column}
    \begin{column}{0.44\textwidth}
      \begin{itemize}
        \item \textbf{Generator} = pemalsu: noise $\to$ gambar palsu
        \item \textbf{Discriminator} = kurator: asli atau palsu?
        \item Dilatih \textbf{bergantian} --- saling memaksa jadi lebih baik
        \item DCGAN: generator \texttt{Conv2DTranspose}, discriminator \texttt{Conv2D}
      \end{itemize}
    \end{column}
  \end{columns}
\end{frame}

\begin{frame}{DCGAN: Digit Lahir dari Noise}
  \begin{center}
    \includegraphics[width=0.8\linewidth]{figures/gan_grid.pdf}
  \end{center}
  \begin{itemize}
    \item Noise input \textbf{sama} di tiap baris --- yang berubah hanya kemampuan generator
    \item Epoch 1: noise berbintik $\to$ epoch 15: mayoritas berbentuk digit
    \item Catatan jujur: GAN rewel ---  \emph{mode collapse} (variasi macet) sering terjadi
  \end{itemize}
\end{frame}

\begin{frame}{Peta Generative AI Modern}
  \setlength{\tabcolsep}{3pt}
  {\small
  \begin{tabular}{p{0.2\linewidth}p{0.42\linewidth}p{0.28\linewidth}}
    {\color{nvgreen}\textbf{Keluarga}} & {\color{nvgreen}\textbf{Ide inti}} & {\color{nvgreen}\textbf{Contoh}} \\
    \hline
    VAE & latent space diatur supaya bisa di-sampling & generator wajah awal \\
    GAN & generator vs discriminator berlomba & StyleGAN, deepfake \\
    Diffusion & generate = denoise bertahap dari noise murni & Stable Diffusion, DALL-E \\
    Autoregressive & prediksi token berikutnya, berulang & GPT, ChatGPT, Llama \\
  \end{tabular}}
  \vspace{0.6em}
  \begin{itemize}
    \item Benang merah: semua \textbf{belajar distribusi data}, lalu sampling darinya
    \item ChatGPT $\approx$ DCGAN kamu: sama-sama generator --- bedanya token, bukan pixel
  \end{itemize}
  \vspace{0.3em}
  {\color{nvlightgreen}\small $\Rightarrow$ Module 04: hands-on dengan generator teks (LLM)}
\end{frame}
```

- [ ] **Step 2: Update frame "Ringkasan Modul 02"**

Baca frame Ringkasan (baris ~855) dan tambahkan satu butir di akhir daftar ringkasannya (sesuaikan format itemize yang ada):

```latex
\item \textbf{Generative AI}: autoencoder \& DCGAN --- network yang mencipta, jembatan ke LLM
```

- [ ] **Step 3: Compile 2-pass + verifikasi**

```bash
cd 02_deep_learning_fundamentals/slides
find . -maxdepth 1 -name "*.aux" -delete
xelatex -interaction=nonstopmode -halt-on-error module02_slides.tex
xelatex -interaction=nonstopmode -halt-on-error module02_slides.tex
grep -c "Overfull" module02_slides.log || echo "0 overfull"
mdls -name kMDItemNumberOfPages module02_slides.pdf
```
Expected: compile sukses, `0 overfull`, halaman naik dari 33 ke ~39-40.

- [ ] **Step 4: Inspeksi visual frame baru**

Render halaman Act 6 ke PNG (`sips` atau `pdftoppm`) dan periksa: figure tidak terpotong, teks tidak menabrak, theme konsisten. Perbaiki kalau ada masalah, recompile.

- [ ] **Step 5: Commit**

```bash
git add 02_deep_learning_fundamentals/slides/module02_slides.tex \
        02_deep_learning_fundamentals/slides/module02_slides.pdf
git commit -m "feat(slides): add Act 6 Generative AI to Module 02 deck (5 frames)"
```

---

### Task 4: Cheat sheet — kartu Generative AI + glossary

**Files:**
- Modify: `02_deep_learning_fundamentals/dl-fundamentals-cheatsheet.html`
- Regenerate: `02_deep_learning_fundamentals/dl-fundamentals-cheatsheet.pdf`

GOTCHA terdokumentasi: konten HTML ter-escape (`'` → `&#x27;`, `>` → `&gt;`) — saat mencari teks eksisting via Python, pakai bentuk escaped.

- [ ] **Step 1: Tambah kartu konsep (jadi kartu ke-11)**

Sisipkan SETELAH kartu terakhir di dalam `<div class="grid">`, ikuti struktur kartu eksisting persis:

```html
<div class="card"><h3>Generative AI: AE &amp; GAN</h3><ul><li>Model generatif menghasilkan DATA BARU, bukan menebak label</li><li>Autoencoder: encoder -&gt; latent space -&gt; decoder; decoder = generator</li><li>GAN: generator (pemalsu) vs discriminator (kurator) dilatih bergantian</li><li>Modern: diffusion (Stable Diffusion) &amp; autoregressive (GPT) -&gt; Module 04</li></ul><div class="when">▸ Fondasi ChatGPT &amp; Stable Diffusion</div></div>
```

- [ ] **Step 2: Tambah 4 entri glossary + update scope**

Di dalam `<div class="glossary">`, tambahkan (ikuti format `<div><b>Istilah</b> — definisi</div>` yang ada):

```html
<div><b>Generator</b> — network yang mengubah noise menjadi data baru (gambar/teks)</div>
<div><b>Discriminator</b> — network penilai pada GAN: asli atau palsu</div>
<div><b>Latent space</b> — ruang representasi terkompresi tempat data "diringkas"</div>
<div><b>Diffusion</b> — model generatif yang mencipta dengan denoising bertahap</div>
```

Update teks `<div class="scope">`: tambahkan di akhir kalimat scope `, plus intro generative AI (autoencoder &amp; GAN)` sebelum tanda titik penutup.

- [ ] **Step 3: Render ulang PDF + verifikasi 1 halaman**

```bash
cd 02_deep_learning_fundamentals
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new \
  --no-pdf-header-footer \
  --print-to-pdf="$(pwd)/dl-fundamentals-cheatsheet.pdf" \
  "file://$(pwd)/dl-fundamentals-cheatsheet.html"
mdls -name kMDItemNumberOfPages dl-fundamentals-cheatsheet.pdf
```
Expected: `kMDItemNumberOfPages = 1`. Kalau jadi 2 halaman: kecilkan `font-size` `.card` 0.5px ATAU pangkas 1 `<li>` dari kartu yang paling padat, render ulang sampai 1 halaman.

- [ ] **Step 4: Commit**

```bash
git add 02_deep_learning_fundamentals/dl-fundamentals-cheatsheet.html \
        02_deep_learning_fundamentals/dl-fundamentals-cheatsheet.pdf
git commit -m "feat(cheatsheet): add generative AI card + glossary to Module 02"
```

---

### Task 5: Quiz — 3 soal generatif (total 15)

**Files:**
- Modify: `02_deep_learning_fundamentals/dl-fundamentals-quiz.html` (const `QUIZ` baris ~61, header `<p>12 soal ...`)

- [ ] **Step 1: Append 3 soal ke array `QUIZ.questions`**

Edit via Python (JSON di dalam HTML — load, modify, dump kembali dengan `ensure_ascii=False`, pertahankan format satu baris seperti aslinya):

```json
{
 "q": "Apa perbedaan paling mendasar antara model diskriminatif dan model generatif?",
 "code": "",
 "options": [
  "Model diskriminatif hanya untuk gambar, model generatif hanya untuk teks",
  "Model diskriminatif memprediksi label dari input, model generatif menghasilkan data baru yang meniru distribusi data training",
  "Model generatif tidak memerlukan proses training sama sekali",
  "Model diskriminatif selalu lebih akurat daripada model generatif"
 ],
 "answer": 1,
 "explanation": "Slide Diskriminatif vs Generatif: model diskriminatif belajar batas antar kelas untuk memprediksi label, sedangkan model generatif belajar distribusi data itu sendiri lalu menghasilkan data baru. Keduanya bisa untuk gambar maupun teks, sama-sama butuh training, dan tidak ada yang otomatis 'lebih akurat'."
}
```

```json
{
 "q": "Pada arsitektur GAN, bagaimana generator dan discriminator dilatih?",
 "code": "gen_loss  = bce(tf.ones_like(fake_logits), fake_logits)\ndisc_loss = bce(tf.ones_like(real_logits), real_logits) + bce(tf.zeros_like(fake_logits), fake_logits)",
 "options": [
  "Generator dilatih dulu sampai selesai, baru discriminator dilatih terpisah",
  "Keduanya dilatih bergantian sebagai lawan: generator berusaha menipu, discriminator berusaha membedakan asli vs palsu",
  "Discriminator membuat gambar baru dan generator menilainya",
  "Keduanya memakai satu optimizer yang sama supaya loss-nya identik"
 ],
 "answer": 1,
 "explanation": "Slide GAN: Generator vs Discriminator menjelaskan keduanya dilatih bergantian dalam satu loop --- generator sukses jika gambar palsunya ditebak asli, discriminator sukses jika bisa membedakan. Perannya tidak terbalik, tidak dilatih terpisah-tuntas, dan masing-masing punya optimizer sendiri."
}
```

```json
{
 "q": "Stable Diffusion dan GPT sama-sama model generatif. Apa ide inti masing-masing keluarga?",
 "code": "",
 "options": [
  "Diffusion: denoising bertahap dari noise murni; GPT: memprediksi token berikutnya secara berulang",
  "Diffusion: memprediksi token berikutnya; GPT: denoising bertahap dari noise",
  "Keduanya memakai generator vs discriminator seperti GAN",
  "Keduanya adalah autoencoder biasa dengan latent space yang lebih besar"
 ],
 "answer": 0,
 "explanation": "Slide Peta Generative AI Modern: keluarga diffusion (Stable Diffusion, DALL-E) mencipta dengan membersihkan noise secara bertahap, sedangkan keluarga autoregressive (GPT) men-generate token demi token. Keduanya BUKAN GAN dan bukan autoencoder biasa."
}
```

- [ ] **Step 2: Update header jumlah soal**

Ganti `<p>12 soal pilihan ganda · gaya sertifikasi · pilih satu jawaban</p>` → `15 soal ...`. Cek juga apakah ada teks lain yang hardcode "12" (mis. skor "/12") — kalau logika pakai `QUIZ.questions.length`, tidak perlu diubah.

- [ ] **Step 3: Uji grading fungsional via puppeteer**

Pattern terdokumentasi: node + chromium bawaan mermaid-cli + `file://` (BUKAN Playwright MCP). Tulis `/tmp/quiz_test06.js`:

Struktur DOM quiz (SUDAH diverifikasi dari source): tiap soal = `<div class="q">` berisi button `.opt`; klik menambah kelas `correct` pada jawaban benar dan `wrong` pada pilihan salah; card terkunci via `dataset.locked`; skor live di `#answered`/`#correct`; summary `#finalscore` memakai `QUIZ.questions.length` (jadi "/15" otomatis benar).

```javascript
// Modul puppeteer dari instalasi mermaid-cli (pattern terdokumentasi di handoff)
const puppeteer = require('/opt/homebrew/lib/node_modules/@mermaid-js/mermaid-cli/node_modules/puppeteer');

(async () => {
  const browser = await puppeteer.launch({headless: 'new'});
  const page = await browser.newPage();
  await page.goto('file:///Users/chmdznr/work/navasena/navasena-gen-ml-course/02_deep_learning_fundamentals/dl-fundamentals-quiz.html');

  const n = await page.evaluate(() => window.QUIZ.questions.length);
  if (n !== 15) throw new Error(`Expected 15 questions, got ${n}`);

  // Soal 13 (index 12): klik jawaban SALAH -> harus dapat kelas 'wrong' + jawaban benar ter-highlight 'correct'
  const wrongCase = await page.evaluate(() => {
    const qi = 12;
    const ans = window.QUIZ.questions[qi].answer;
    const wrongIdx = (ans + 1) % 4;
    const opts = document.querySelectorAll('.q')[qi].querySelectorAll('.opt');
    opts[wrongIdx].click();
    return {clickedHasWrong: opts[wrongIdx].classList.contains('wrong'),
            answerHasCorrect: opts[ans].classList.contains('correct')};
  });
  if (!wrongCase.clickedHasWrong || !wrongCase.answerHasCorrect)
    throw new Error('Soal 13 (kasus salah): ' + JSON.stringify(wrongCase));

  // Soal 14-15 (index 13-14): klik jawaban BENAR -> kelas 'correct', skor naik
  for (const qi of [13, 14]) {
    const ok = await page.evaluate((qi) => {
      const ans = window.QUIZ.questions[qi].answer;
      const opts = document.querySelectorAll('.q')[qi].querySelectorAll('.opt');
      opts[ans].click();
      return opts[ans].classList.contains('correct');
    }, qi);
    if (!ok) throw new Error(`Soal ${qi + 1}: kelas correct tidak muncul`);
  }
  const score = await page.evaluate(() => document.getElementById('correct').textContent);
  if (score !== '2') throw new Error(`Skor live salah: expected 2, got ${score}`);
  console.log('QUIZ TEST PASS: 15 soal; kasus benar & salah ter-grade sesuai; skor live OK');
  await browser.close();
})();
```

Kalau path puppeteer mermaid-cli tidak ada, cari dengan: `find /opt/homebrew/lib/node_modules -maxdepth 4 -name puppeteer -type d 2>/dev/null | head -3` dan sesuaikan `require()`.

Run: `node /tmp/quiz_test06.js`
Expected: `QUIZ TEST PASS: 15 soal, 3 soal baru ter-grade benar`

- [ ] **Step 4: Commit**

```bash
git add 02_deep_learning_fundamentals/dl-fundamentals-quiz.html
git commit -m "feat(quiz): add 3 generative AI questions to Module 02 quiz (15 total)"
```

---

### Task 6: QC akhir, handoff, push

**Files:**
- Modify: `docs/handoffs/2026-06-01-session-handoff.md` (tambah update section)

- [ ] **Step 1: QC lintas-artefak**

Checklist (jalankan dan catat hasilnya):

```bash
# 1. Konsistensi istilah notebook<->deck<->cheatsheet<->quiz (harus sama: DCGAN, latent space, dst)
grep -o 'DCGAN\|latent space\|Conv2DTranspose' 02_deep_learning_fundamentals/06_generative_ai_intro.ipynb | sort | uniq -c
grep -o 'DCGAN\|latent space\|Conv2DTranspose' 02_deep_learning_fundamentals/slides/module02_slides.tex | sort | uniq -c
# 2. Kaidah bahasa di semua artefak baru/teredit
grep -riE 'ruang laten|kebisingan|pembangkit|pelatihan berlawanan|jaringan pembeda' \
  02_deep_learning_fundamentals/06_generative_ai_intro.ipynb \
  02_deep_learning_fundamentals/slides/module02_slides.tex \
  02_deep_learning_fundamentals/dl-fundamentals-cheatsheet.html \
  02_deep_learning_fundamentals/dl-fundamentals-quiz.html && echo "ADA ANTI-PATTERN" || echo "bahasa OK"
# 3. Deck: 0 overfull (dari log compile terakhir)
grep -c Overfull 02_deep_learning_fundamentals/slides/module02_slides.log || echo "0 overfull"
# 4. Cheatsheet tetap 1 halaman
mdls -name kMDItemNumberOfPages 02_deep_learning_fundamentals/dl-fundamentals-cheatsheet.pdf
# 5. Tidak ada branding terlarang
grep -ci 'NCA-GENL\|Achmad' 02_deep_learning_fundamentals/06_generative_ai_intro.ipynb || echo "branding OK"
```
Expected: bahasa OK, 0 overfull, 1 halaman, branding OK.

- [ ] **Step 2: Update handoff**

Tambah section di `docs/handoffs/2026-06-01-session-handoff.md`:

```markdown
### Update Jun 10 — Materi Generative AI Module 02 SELESAI
- Notebook baru `06_generative_ai_intro.ipynb`: AE penuh (10 epoch) + DCGAN mini (15 epoch, subset 20k) di MNIST, arc "dari mengenali → mencipta", jembatan ke Module 04.
- Deck Act 6 (5 frame), cheatsheet +1 kartu & 4 glossary, quiz 12→15 soal.
- Figure deck dari training NYATA via `slides/figures/train_generative_figs.py` (prefix train_ supaya build.sh tidak melatih ulang; PDF di-commit).
- Spec: `docs/superpowers/specs/2026-06-10-module02-generative-ai-design.md`. Plan: `docs/superpowers/plans/2026-06-10-module02-generative-ai.md`.
- Status M02: 6 notebook, 6 act deck — mapping 1:1 dipertahankan.
```

- [ ] **Step 3: Commit handoff + push semua**

```bash
git add docs/handoffs/2026-06-01-session-handoff.md
git commit -m "docs(handoff): record Module 02 generative AI material complete"
git push
git status   # harus clean, sinkron dengan origin
```

- [ ] **Step 4: Verifikasi akhir sebelum klaim selesai**

Jalankan ulang: deck PDF ada & ~39 halaman, cheatsheet 1 halaman, quiz test PASS, notebook smoke-run OK (bukti dari Task 1 Step 4). Laporkan semua bukti ke user — JANGAN klaim selesai tanpa output verifikasi.

---

## Catatan untuk Eksekutor

1. **Jangan jalankan ulang `train_generative_figs.py` saat build deck** — itu sebabnya prefix-nya `train_`, bukan `gen_`. `build.sh` hanya menjalankan `gen_*.py`.
2. **Smoke-run notebook (Task 1 Step 4) wajib lulus sebelum commit notebook** — ini satu-satunya bukti lokal bahwa kode TF jalan; validasi penuh 10+15 epoch terjadi saat figure run (Task 2) yang memakai arsitektur identik.
3. Kalau hook `latex-overfull-guard.sh` mem-block compile: perbaiki layout (kecilkan figure/font), jangan bypass hook.
4. Urutan task fleksibel sebagian: Task 2 Step 2 (training lokal, lama) bisa jalan background sambil mengerjakan Task 3 Step 1-2 (edit .tex), tapi compile (Task 3 Step 3) harus menunggu PDF figure selesai.
