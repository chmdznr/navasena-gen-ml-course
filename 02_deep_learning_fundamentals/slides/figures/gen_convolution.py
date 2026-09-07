"""
Generate figures/convolution.pdf — satu digit MNIST 28x28 dikonvolusi
dengan filter tepi vertikal dan filter tepi horizontal (angka filter
ditampilkan), beserta hasilnya.
Run dari direktori slides/: python3 figures/gen_convolution.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)

BG = "#1A1A2E"
TEXT = "white"

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
idx = np.where(y_train == 7)[0][0]  # digit "7" — kontras tepi vertikal/horizontal jelas
img = x_train[idx].astype("float32") / 255.0

vertical_filter = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype="float32")
horizontal_filter = np.array([[-1, -1, -1],
                               [0, 0, 0],
                               [1, 1, 1]], dtype="float32")


def convolve2d_valid(image, kernel):
    kh, kw = kernel.shape
    h, w = image.shape
    out = np.zeros((h - kh + 1, w - kw + 1), dtype="float32")
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
    return out


out_v = convolve2d_valid(img, vertical_filter)
out_h = convolve2d_valid(img, horizontal_filter)

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
fig.patch.set_facecolor(BG)

rows = [
    ("Filter tepi vertikal", vertical_filter, out_v),
    ("Filter tepi horizontal", horizontal_filter, out_h),
]

for r, (title, kernel, out) in enumerate(rows):
    ax_img = axes[r, 0]
    ax_img.set_facecolor(BG)
    ax_img.imshow(img, cmap="gray")
    ax_img.set_title("Gambar input (digit 7)" if r == 0 else "", color=TEXT, fontsize=11)
    ax_img.axis("off")

    ax_k = axes[r, 1]
    ax_k.set_facecolor(BG)
    ax_k.imshow(kernel, cmap="RdBu", vmin=-1, vmax=1)
    for (i, j), v in np.ndenumerate(kernel):
        ax_k.text(j, i, f"{v:.0f}", ha="center", va="center", color=TEXT,
                   fontsize=13, fontweight="bold")
    ax_k.set_title(title, color=TEXT, fontsize=11)
    ax_k.set_xticks([]); ax_k.set_yticks([])

    ax_out = axes[r, 2]
    ax_out.set_facecolor(BG)
    ax_out.imshow(out, cmap="gray")
    ax_out.set_title("Hasil konvolusi" if r == 0 else "", color=TEXT, fontsize=11)
    ax_out.axis("off")

fig.suptitle("Konvolusi manual: filter tepi vertikal vs horizontal", color=TEXT,
             fontsize=15, fontweight="bold", y=0.99)
plt.tight_layout(pad=1.0)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convolution.pdf")
fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=BG)
print(f"Saved: {output_path}")
print(f"digit index={idx} label={y_train[idx]} out_v shape={out_v.shape} out_h shape={out_h.shape}")
