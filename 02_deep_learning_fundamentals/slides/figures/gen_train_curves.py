"""
Generate figures/train_curves.pdf dari run nyata (bukan angka karangan).
Model: Dense(128, relu) -> Dense(10, softmax) pada Fashion-MNIST, 8 epoch,
validation_split=0.1. Dua panel: loss dan accuracy (train vs validation),
epoch dengan val_loss terendah ditandai.
Run dari direktori slides/: python3 figures/gen_train_curves.py
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
MARK_COLOR = "#FFCA28"

(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=8, batch_size=128,
                     validation_split=0.1, verbose=0)

h = history.history
best_epoch = int(np.argmin(h["val_loss"]))  # 0-indexed
epochs = np.arange(1, len(h["loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
fig.patch.set_facecolor(BG)

# Panel 1: loss
ax = axes[0]
ax.set_facecolor(BG)
ax.plot(epochs, h["loss"], color=TRAIN_COLOR, linewidth=2.2, marker="o",
        markersize=4, label="data train")
ax.plot(epochs, h["val_loss"], color=VAL_COLOR, linewidth=2.2, marker="o",
        markersize=4, label="data validation")
ax.axvline(epochs[best_epoch], color=MARK_COLOR, linewidth=1.4, linestyle="--")
# label garis putus-putus: ditaruh di sudut kiri-bawah panel (area kosong),
# warnanya sama dengan garisnya supaya kaitannya jelas
ax.text(epochs[0] + 0.15, min(h["loss"]) + 0.005,
        f"garis kuning: val_loss terendah (epoch {epochs[best_epoch]})",
        color=MARK_COLOR, fontsize=13, fontweight="bold", va="bottom")
ax.set_title("Loss", color=TEXT, fontweight="bold")
ax.set_xlabel("epoch", color=TEXT)
ax.set_ylabel("loss", color=TEXT)

# Panel 2: accuracy
ax2 = axes[1]
ax2.set_facecolor(BG)
ax2.plot(epochs, h["accuracy"], color=TRAIN_COLOR, linewidth=2.2, marker="o",
         markersize=4, label="data train")
ax2.plot(epochs, h["val_accuracy"], color=VAL_COLOR, linewidth=2.2, marker="o",
         markersize=4, label="data validation")
ax2.axvline(epochs[best_epoch], color=MARK_COLOR, linewidth=1.4, linestyle="--")
ax2.set_title("Accuracy", color=TEXT, fontweight="bold")
ax2.set_xlabel("epoch", color=TEXT)
ax2.set_ylabel("accuracy", color=TEXT)

for a, loc in zip(axes, ["upper right", "lower right"]):
    a.tick_params(colors=TEXT)
    for spine in a.spines.values():
        spine.set_edgecolor("#444466")
    a.grid(True, color=GRID, linestyle="--", linewidth=0.6, zorder=0)
    a.set_axisbelow(True)
    leg = a.legend(loc=loc, facecolor="#2A2A4E", edgecolor="#555577")
    for txt in leg.get_texts():
        txt.set_color(TEXT)

fig.suptitle("Kurva pelatihan: Dense(128) pada Fashion-MNIST (8 epoch)",
             color=TEXT, y=1.0)
plt.tight_layout(pad=1.0)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_curves.pdf")
fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=BG)
print(f"Saved: {output_path}")
print(f"best_epoch={epochs[best_epoch]} val_loss={h['val_loss'][best_epoch]:.4f} "
      f"val_accuracy={h['val_accuracy'][best_epoch]:.4f} "
      f"final_train_acc={h['accuracy'][-1]:.4f}")
