"""
Generate figures/overfit_curves.pdf dari run nyata.
Dua model pada 6000 sampel train Fashion-MNIST, 30 epoch:
  - overfit: Dense(512) -> Dense(512) -> Dense(10), tanpa regularisasi
  - dropout: sama, tapi dengan Dropout(0.5) di antara layer Dense besar
Dua panel: loss train vs validation untuk masing-masing model, memperlihatkan
gap yang membesar (overfit) vs gap kecil (dropout).
Run dari direktori slides/: python3 figures/gen_overfit_curves.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18,
                     'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)
# angka yang dikutip di slide harus sama tiap kali figur diregenerasi
tf.config.experimental.enable_op_determinism()

BG = "#1A1A2E"
TEXT = "white"
GRID = "#333355"
TRAIN_COLOR = "#42A5F5"
VAL_COLOR = "#76B900"

(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_small, y_small = x_train[:6000], y_train[:6000]

EPOCHS = 30


def build_model(use_dropout):
    layers = [tf.keras.Input(shape=(28, 28)),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(512, activation="relu")]
    if use_dropout:
        layers.append(tf.keras.layers.Dropout(0.5))
    layers.append(tf.keras.layers.Dense(512, activation="relu"))
    if use_dropout:
        layers.append(tf.keras.layers.Dropout(0.5))
    layers.append(tf.keras.layers.Dense(10, activation="softmax"))
    m = tf.keras.Sequential(layers)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def run(use_dropout):
    tf.random.set_seed(42)
    m = build_model(use_dropout)
    hist = m.fit(x_small, y_small, epochs=EPOCHS, batch_size=128,
                 validation_split=0.2, verbose=0)
    return hist.history


hist_overfit = run(use_dropout=False)
hist_dropout = run(use_dropout=True)

epochs = np.arange(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
fig.patch.set_facecolor(BG)

panels = [
    ("Tanpa Dropout (overfit)", hist_overfit, axes[0]),
    ("Dengan Dropout(0.5)", hist_dropout, axes[1]),
]

for title, h, ax in panels:
    ax.set_facecolor(BG)
    ax.plot(epochs, h["loss"], color=TRAIN_COLOR, linewidth=2.0, label="data train")
    ax.plot(epochs, h["val_loss"], color=VAL_COLOR, linewidth=2.0, label="data validation")
    gap = h["val_loss"][-1] - h["loss"][-1]
    ax.fill_between(epochs, h["loss"], h["val_loss"], color="#FF7043", alpha=0.15)
    ax.set_title(f"{title}\ngap akhir = {gap:.2f}", color=TEXT, fontweight="bold")
    ax.set_xlabel("epoch", color=TEXT)
    ax.set_ylabel("loss", color=TEXT)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower left", facecolor="#2A2A4E", edgecolor="#555577")
    for txt in leg.get_texts():
        txt.set_color(TEXT)

ymax = max(max(hist_overfit["val_loss"]), max(hist_dropout["val_loss"])) * 1.1
for _, _, ax in panels:
    ax.set_ylim(0, ymax)

fig.suptitle("Overfitting vs Dropout(0.5) — 6000 sampel train, 30 epoch",
             color=TEXT, y=1.0)
plt.tight_layout(pad=1.0)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overfit_curves.pdf")
fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=BG)
print(f"Saved: {output_path}")
print(f"overfit: final train_loss={hist_overfit['loss'][-1]:.4f} "
      f"val_loss={hist_overfit['val_loss'][-1]:.4f} gap={hist_overfit['val_loss'][-1]-hist_overfit['loss'][-1]:.4f}")
print(f"dropout: final train_loss={hist_dropout['loss'][-1]:.4f} "
      f"val_loss={hist_dropout['val_loss'][-1]:.4f} gap={hist_dropout['val_loss'][-1]-hist_dropout['loss'][-1]:.4f}")
