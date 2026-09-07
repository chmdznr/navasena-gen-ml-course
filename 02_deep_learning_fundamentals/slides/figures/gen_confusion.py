"""
Generate figures/confusion.pdf — confusion matrix dari model Dense(128)
(sama arsitektur dengan gen_train_curves.py) pada test set nyata
Fashion-MNIST. Judul memuat akurasi test yang benar-benar dihitung.
Run dari direktori slides/: python3 figures/gen_confusion.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score

tf.random.set_seed(42)
np.random.seed(42)

BG = "#1A1A2E"
TEXT = "white"

classes = ["T-shirt", "Celana", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=8, batch_size=128, validation_split=0.1, verbose=0)

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8.2, 6.6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.imshow(cm, cmap="Greens")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(classes, rotation=45, ha="right", color=TEXT, fontsize=9)
ax.set_yticklabels(classes, color=TEXT, fontsize=9)
ax.set_xlabel("Prediksi", color=TEXT, fontsize=12)
ax.set_ylabel("Aktual", color=TEXT, fontsize=12)
ax.set_title(f"Confusion Matrix — Fashion-MNIST test set (akurasi = {acc*100:.1f}%)",
             color=TEXT, fontsize=13, fontweight="bold", pad=12)
thresh = cm.max() / 2
for i in range(10):
    for j in range(10):
        if cm[i, j] > 0:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color=(TEXT if cm[i, j] > thresh else "#0d0d0d"))
for spine in ax.spines.values():
    spine.set_color(TEXT)
ax.tick_params(colors=TEXT)
fig.tight_layout()

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion.pdf")
fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {output_path}")
print(f"test_accuracy={acc:.4f}")
