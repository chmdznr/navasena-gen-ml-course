"""Figure: pertumbuhan jumlah parameter model bahasa (tema gelap untuk slide).

Output: figures/model_scale.pdf (vektor, facecolor gelap, sumbu y log).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# --- Tema gelap ---
BG = "#1A1A2E"       # facecolor figure / axes
GREEN = "#76B900"    # aksen hijau NVIDIA
WHITE = "#FFFFFF"    # teks / aksen putih

# --- Data tetap (hardcode) ---
# Jumlah parameter dinyatakan dalam JUTA (M).
models = ["BERT-base", "GPT-2", "Llama-2-70B", "GPT-3"]
params_millions = np.array([110.0, 1_500.0, 70_000.0, 175_000.0])
labels = ["110M", "1.5B", "70B", "175B"]

fig, ax = plt.subplots(figsize=(9, 5.2))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

x = np.arange(len(models))
bars = ax.bar(x, params_millions, color=GREEN, edgecolor=WHITE, linewidth=0.9, width=0.6)

# Anotasi ukuran di atas tiap bar
for rect, lab in zip(bars, labels):
    ax.text(
        rect.get_x() + rect.get_width() / 2.0,
        rect.get_height() * 1.15,
        lab,
        ha="center",
        va="bottom",
        color=WHITE,
        fontsize=11,
        fontweight="bold",
    )

# --- Sumbu y log ---
ax.set_yscale("log")
ax.set_ylim(10, 1_000_000)
ax.set_ylabel("Jumlah parameter (juta, skala log)", color=WHITE, fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(models, color=WHITE, fontsize=11)

ax.set_title(
    "Pertumbuhan Skala Model Bahasa",
    color=WHITE,
    fontsize=16,
    fontweight="bold",
    pad=14,
)

# Warna sumbu / grid terang
ax.tick_params(axis="y", colors=WHITE, which="both")
ax.tick_params(axis="x", colors=WHITE)
for spine in ax.spines.values():
    spine.set_color(WHITE)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=WHITE, alpha=0.15, linewidth=0.6, which="both")

fig.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig(
    "figures/model_scale.pdf",
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
plt.close(fig)
