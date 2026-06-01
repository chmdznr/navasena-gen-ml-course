"""
Ilustrasi "Ruang Embedding": kalimat dengan makna mirip dipetakan
menjadi titik yang berdekatan. Tema GELAP, output PDF vektor.

Jalankan:
    python gen_embedding_space.py
Output:
    figures/embedding_space.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# --------------------------------------------------------------------------
# Konfigurasi tema GELAP
# --------------------------------------------------------------------------
BG = "#1A1A2E"          # facecolor latar
FG = "#FFFFFF"          # warna teks utama
SEED = 7

np.random.seed(SEED)

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.edgecolor": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "font.size": 11,
})

# --------------------------------------------------------------------------
# Data hardcoded: 3 cluster topik, tiap titik punya pusat + sedikit jitter
# --------------------------------------------------------------------------
clusters = {
    "Hewan": {
        "color": "#FF6B6B",
        "center": (-3.2, 2.4),
        "points": [
            ("kucing", (-3.6, 2.9)),
            ("anjing", (-2.7, 2.7)),
            ("harimau", (-3.4, 1.8)),
            ("burung",  (-2.5, 2.1)),
        ],
    },
    "Teknologi": {
        "color": "#4ECDC4",
        "center": (3.0, 2.0),
        "points": [
            ("komputer", (3.4, 2.6)),
            ("server",   (2.5, 2.3)),
            ("algoritma",(3.5, 1.5)),
            ("jaringan", (2.6, 1.4)),
        ],
    },
    "Makanan": {
        "color": "#FFD93D",
        "center": (-0.2, -2.6),
        "points": [
            ("nasi goreng", (-0.6, -2.0)),
            ("rendang",     (0.5, -2.3)),
            ("sate",        (-0.8, -3.1)),
            ("bakso",       (0.4, -3.2)),
        ],
    },
}

# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

for label, info in clusters.items():
    color = info["color"]
    xs = [p[1][0] for p in info["points"]]
    ys = [p[1][1] for p in info["points"]]

    # Lingkaran samar yang membungkus cluster (penegas "berkelompok")
    cx, cy = np.mean(xs), np.mean(ys)
    rx = (max(xs) - min(xs)) / 2 + 1.0
    ry = (max(ys) - min(ys)) / 2 + 1.0
    halo = Ellipse((cx, cy), width=2 * rx, height=2 * ry,
                   facecolor=color, edgecolor=color,
                   alpha=0.10, lw=1.5, ls="--", zorder=1)
    ax.add_patch(halo)

    # Titik-titik kalimat
    ax.scatter(xs, ys, s=170, color=color, edgecolors=FG,
               linewidths=1.2, zorder=3, label=label)

    # Anotasi teks pendek tiap titik
    for word, (px, py) in info["points"]:
        ax.annotate(word, (px, py),
                    xytext=(6, 6), textcoords="offset points",
                    color=FG, fontsize=9.5, zorder=4)

    # Label nama cluster (sedikit di atas halo)
    ax.text(cx, cy + ry + 0.25, label, color=color,
            fontsize=13, fontweight="bold", ha="center", zorder=4)

# --------------------------------------------------------------------------
# Sumbu abstrak (dim-1, dim-2): sembunyikan angka, beri panah arah
# --------------------------------------------------------------------------
ax.set_xlim(-6.0, 6.0)
ax.set_ylim(-5.5, 5.0)
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

# Panah sumbu di pojok kiri bawah sebagai orientasi abstrak
ax.annotate("", xy=(-4.0, -4.8), xytext=(-5.6, -4.8),
            arrowprops=dict(arrowstyle="->", color=FG, lw=1.5))
ax.annotate("", xy=(-5.6, -3.4), xytext=(-5.6, -4.8),
            arrowprops=dict(arrowstyle="->", color=FG, lw=1.5))
ax.text(-4.0, -5.05, "dimensi-1", color=FG, fontsize=9,
        ha="center", va="top", style="italic")
ax.text(-5.85, -3.4, "dimensi-2", color=FG, fontsize=9,
        ha="right", va="center", rotation=90, style="italic")

ax.set_title("Ruang Embedding: Makna Mirip -> Berdekatan",
             color=FG, fontsize=16, fontweight="bold", pad=18)

legend = ax.legend(loc="upper left", framealpha=0.0, fontsize=11)
for text in legend.get_texts():
    text.set_color(FG)

# --------------------------------------------------------------------------
# Simpan ke PDF vektor (path RELATIF), TANPA plt.show()
# --------------------------------------------------------------------------
os.makedirs("figures", exist_ok=True)
out_path = "figures/embedding_space.pdf"
fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Tersimpan: {out_path}")
