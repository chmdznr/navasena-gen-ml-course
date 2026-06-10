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
