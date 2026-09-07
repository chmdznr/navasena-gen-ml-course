"""
Generate figures/feature_maps.pdf — CNN kecil (Conv 8 -> Conv 16) dilatih
1 epoch pada MNIST, menampilkan 8 feature map layer pertama untuk satu
gambar input.
Run dari direktori slides/: python3 figures/gen_feature_maps.py
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

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]

# subset kecil untuk 1 epoch cepat di CPU
n_sub = 4000
x_sub, y_sub = x_train[:n_sub], y_train[:n_sub]

inputs = tf.keras.Input(shape=(28, 28, 1))
conv1 = tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same", name="conv1")(inputs)
pool1 = tf.keras.layers.MaxPooling2D()(conv1)
conv2 = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same", name="conv2")(pool1)
pool2 = tf.keras.layers.MaxPooling2D()(conv2)
flat = tf.keras.layers.Flatten()(pool2)
outputs = tf.keras.layers.Dense(10, activation="softmax")(flat)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(x_sub, y_sub, epochs=1, batch_size=64, verbose=0)

feature_model = tf.keras.Model(inputs, conv1)
idx = np.where(y_train == 3)[0][0]
sample = x_train[idx:idx + 1]
feature_maps = feature_model.predict(sample, verbose=0)[0]  # (28, 28, 8)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

axes[0].set_facecolor(BG)
axes[0].imshow(sample[0, :, :, 0], cmap="gray")
axes[0].set_title("Input (digit 3)", color=TEXT, fontweight="bold")
axes[0].axis("off")

for i in range(8):
    ax = axes[i + 1]
    ax.set_facecolor(BG)
    ax.imshow(feature_maps[:, :, i], cmap="viridis")
    ax.set_title(f"feature map {i+1}", color=TEXT)
    ax.axis("off")

fig.suptitle("Feature map layer conv1 (8 filter) — setelah 1 epoch", color=TEXT, fontweight="bold", y=0.995)
plt.tight_layout(pad=1.0)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_maps.pdf")
fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=BG)
print(f"Saved: {output_path}")
print(f"train_acc_1epoch={history.history['accuracy'][-1]:.4f} "
      f"train_loss_1epoch={history.history['loss'][-1]:.4f}")
