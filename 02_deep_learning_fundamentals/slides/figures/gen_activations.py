"""
Generate figures/activations.pdf — sigmoid, tanh, ReLU, Leaky ReLU beserta
turunannya (2 baris x 4 kolom). Daerah gradien ~0 ditandai oranye (saturasi
sigmoid/tanh, sisi negatif ReLU); sisi negatif Leaky ReLU ditandai ungu
karena gradiennya kecil tetapi bukan nol.
Run dari direktori slides/: python3 figures/gen_activations.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18,
                     'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})

BG = "#1A1A2E"
TEXT = "white"
GRID = "#333355"
FN_COLOR = "#76B900"
DERIV_COLOR = "#42A5F5"
FLAT_COLOR = "#FF7043"
SMALL_COLOR = "#9575CD"

LEAKY_ALPHA = 0.1
x = np.linspace(-5, 5, 400)

funcs = {
    "Sigmoid": (1.0 / (1.0 + np.exp(-x)), None),
    "Tanh": (np.tanh(x), None),
    "ReLU": (np.maximum(0.0, x), None),
    f"Leaky ReLU (α={LEAKY_ALPHA})": (np.where(x >= 0, x, LEAKY_ALPHA * x), None),
}

sigmoid = funcs["Sigmoid"][0]
tanh = funcs["Tanh"][0]
relu = funcs["ReLU"][0]
leaky = funcs[f"Leaky ReLU (α={LEAKY_ALPHA})"][0]

derivs = {
    "Sigmoid": sigmoid * (1 - sigmoid),
    "Tanh": 1 - tanh ** 2,
    "ReLU": np.where(x >= 0, 1.0, 0.0),
    f"Leaky ReLU (α={LEAKY_ALPHA})": np.where(x >= 0, 1.0, LEAKY_ALPHA),
}

flat_regions = {
    "Sigmoid": (x < -3) | (x > 3),
    "Tanh": (x < -2.5) | (x > 2.5),
    "ReLU": x < 0,
    f"Leaky ReLU (α={LEAKY_ALPHA})": np.zeros_like(x, dtype=bool),
}

# Sisi negatif Leaky ReLU: turunannya 0,1 — kecil, tetapi bukan nol. Ditandai
# dengan warna lain supaya tidak terbaca sebagai "mati" seperti ReLU.
small_regions = {
    "Sigmoid": np.zeros_like(x, dtype=bool),
    "Tanh": np.zeros_like(x, dtype=bool),
    "ReLU": np.zeros_like(x, dtype=bool),
    f"Leaky ReLU (α={LEAKY_ALPHA})": x < 0,
}

names = ["Sigmoid", "Tanh", "ReLU", f"Leaky ReLU (α={LEAKY_ALPHA})"]

fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.0))
fig.patch.set_facecolor(BG)

for col, name in enumerate(names):
    fn = funcs[name][0]
    d = derivs[name]
    flat = flat_regions[name]
    small = small_regions[name]

    ax_fn = axes[0, col]
    ax_fn.set_facecolor(BG)
    ax_fn.plot(x, fn, color=FN_COLOR, linewidth=2.2)
    ax_fn.fill_between(x, fn.min(), fn.max(),
                        where=flat, color=FLAT_COLOR, alpha=0.12, zorder=0)
    ax_fn.fill_between(x, fn.min(), fn.max(),
                        where=small, color=SMALL_COLOR, alpha=0.12, zorder=0)
    ax_fn.set_title(name, color=TEXT, fontweight="bold")
    ax_fn.axhline(0, color="#888899", linewidth=0.8)
    ax_fn.axvline(0, color="#888899", linewidth=0.8)

    ax_d = axes[1, col]
    ax_d.set_facecolor(BG)
    ax_d.plot(x, d, color=DERIV_COLOR, linewidth=2.2)
    ax_d.fill_between(x, 0, d.max() * 1.1, where=flat, color=FLAT_COLOR, alpha=0.15, zorder=0)
    ax_d.fill_between(x, 0, d.max() * 1.1, where=small, color=SMALL_COLOR, alpha=0.18, zorder=0)
    ax_d.axhline(0, color="#888899", linewidth=0.8)
    ax_d.set_xlabel("x", color=TEXT)

    for ax in (ax_fn, ax_d):
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.grid(True, color=GRID, linestyle="--", linewidth=0.5, zorder=-1)
        ax.set_axisbelow(True)
        ax.set_xlim(-5, 5)

axes[0, 0].set_ylabel("f(x)", color=TEXT)
axes[1, 0].set_ylabel("f'(x)", color=TEXT)

# Legend penjelas daerah gradien ~0
handles = [plt.Rectangle((0, 0), 1, 1, color=FLAT_COLOR, alpha=0.3),
           plt.Rectangle((0, 0), 1, 1, color=SMALL_COLOR, alpha=0.3)]
fig.legend(handles, ["daerah gradien ~0 (saturasi / mati)",
                     "gradient kecil, tidak nol"],
           loc="lower center", ncol=2, framealpha=0.3,
           facecolor="#2A2A4E", edgecolor="#555577",
           labelcolor=TEXT, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Fungsi Aktivasi dan Turunannya", color=TEXT,
             fontweight="bold", y=1.0)
plt.tight_layout(rect=(0, 0.05, 1, 0.95))

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "activations.pdf")
fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=BG)
print(f"Saved: {output_path}")
