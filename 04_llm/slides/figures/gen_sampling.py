"""Figure: efek temperature pada distribusi softmax (tema gelap untuk slide).

Output: figures/sampling.pdf (vektor, facecolor gelap).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# --- Tema gelap ---
BG = "#1A1A2E"       # facecolor figure / axes
GREEN = "#76B900"    # aksen hijau NVIDIA
WHITE = "#FFFFFF"    # teks / aksen putih

# --- Data tetap (hardcode, tanpa random) ---
# Logits contoh untuk ~6 kandidat token.
tokens = ["belajar", "makan", "tidur", "lari", "membaca", "menulis"]
logits = np.array([3.0, 2.0, 1.5, 0.5, 0.2, -0.5])

temperatures = [0.3, 0.7, 1.5]
labels = {0.3: "T=0.3 (peaked)", 0.7: "T=0.7 (sedang)", 1.5: "T=1.5 (flat)"}


def softmax(z):
    z = z - np.max(z)  # stabilitas numerik
    e = np.exp(z)
    return e / e.sum()


fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
fig.patch.set_facecolor(BG)

x = np.arange(len(tokens))

for ax, T in zip(axes, temperatures):
    probs = softmax(logits / T)

    ax.set_facecolor(BG)
    bars = ax.bar(x, probs, color=GREEN, edgecolor=WHITE, linewidth=0.8, width=0.7)

    # Anotasi nilai probabilitas di atas tiap bar
    for rect, p in zip(bars, probs):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.015,
            f"{p:.2f}",
            ha="center",
            va="bottom",
            color=WHITE,
            fontsize=8,
        )

    ax.set_title(labels[T], color=GREEN, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=40, ha="right", color=WHITE, fontsize=9)

    # Sumbu & label warna terang
    ax.tick_params(axis="y", colors=WHITE)
    for spine in ax.spines.values():
        spine.set_color(WHITE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color=WHITE, alpha=0.12, linewidth=0.6)

axes[0].set_ylabel("Probabilitas softmax", color=WHITE, fontsize=11)

fig.suptitle(
    "Efek Temperature pada Distribusi Sampling Token",
    color=WHITE,
    fontsize=15,
    fontweight="bold",
)
fig.text(
    0.5,
    0.005,
    "Temperature rendah -> tajam (deterministik) | tinggi -> rata (lebih acak)",
    ha="center",
    color=WHITE,
    fontsize=9,
    alpha=0.85,
)

fig.tight_layout(rect=[0, 0.03, 1, 0.94])

os.makedirs("figures", exist_ok=True)
fig.savefig(
    "figures/sampling.pdf",
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
plt.close(fig)
