"""
Generate figures/activations.pdf - empat fungsi aktivasi (ReLU, Sigmoid,
Tanh, LeakyReLU) di satu axis dengan tema gelap untuk slide.
Run from the directory yang berisi folder figures/:
    python3 gen_activations.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Output path RELATIF
OUTPUT_DIR = "figures"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "activations.pdf")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tema gelap
BG_COLOR = "#1A1A2E"
TEXT_COLOR = "white"
GRID_COLOR = "#333355"
AXIS_LINE_COLOR = "#888899"
PANEL_COLOR = "#2A2A4E"

# Warna kurva: aksen hijau NVIDIA + 3 warna lain
RELU_COLOR = "#76B900"   # aksen hijau
SIGMOID_COLOR = "#FF7043"
TANH_COLOR = "#42A5F5"
LEAKY_COLOR = "#FFCA28"

# Data hardcoded (rentang tetap)
LEAKY_ALPHA = 0.1
x = np.linspace(-5, 5, 400)

relu = np.maximum(0.0, x)
sigmoid = 1.0 / (1.0 + np.exp(-x))
tanh = np.tanh(x)
leaky = np.where(x >= 0, x, LEAKY_ALPHA * x)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

ax.plot(x, relu, color=RELU_COLOR, linewidth=2.4, label="ReLU", zorder=4)
ax.plot(x, sigmoid, color=SIGMOID_COLOR, linewidth=2.4, label="Sigmoid", zorder=4)
ax.plot(x, tanh, color=TANH_COLOR, linewidth=2.4, label="Tanh", zorder=4)
ax.plot(x, leaky, color=LEAKY_COLOR, linewidth=2.4,
        linestyle="--", label=f"LeakyReLU (α={LEAKY_ALPHA})", zorder=3)

# Garis sumbu di 0
ax.axhline(0, color=AXIS_LINE_COLOR, linewidth=1.0, zorder=2)
ax.axvline(0, color=AXIS_LINE_COLOR, linewidth=1.0, zorder=2)

# Judul & label
ax.set_title("Fungsi Aktivasi", color=TEXT_COLOR, fontsize=15,
             fontweight="bold", pad=12)
ax.set_xlabel("x", color=TEXT_COLOR, fontsize=11)
ax.set_ylabel("f(x)", color=TEXT_COLOR, fontsize=11)

# Styling axes (teks/grid putih)
ax.set_xlim(-5, 5)
ax.set_ylim(-1.6, 5.0)
ax.tick_params(axis="x", colors=TEXT_COLOR)
ax.tick_params(axis="y", colors=TEXT_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor("#444466")

# Grid
ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# Legend
ax.legend(loc="upper left", framealpha=0.3,
          facecolor=PANEL_COLOR, edgecolor="#555577",
          labelcolor=TEXT_COLOR, fontsize=10)

plt.tight_layout(pad=1.2)
plt.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {OUTPUT_PATH}")
