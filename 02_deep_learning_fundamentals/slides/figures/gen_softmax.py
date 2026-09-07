"""
Generate figures/softmax.pdf — skor mentah 3 kelas diubah menjadi
probabilitas oleh softmax. Bar chart sebelum/sesudah, angka tertulis
di atas tiap bar.
Run dari direktori slides/: python3 figures/gen_softmax.py
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
RAW_COLOR = "#42A5F5"
PROB_COLOR = "#76B900"

np.random.seed(42)
classes = ["Kucing", "Anjing", "Burung"]
scores = np.array([2.0, 1.0, 0.1])  # skor mentah (logit)

exp_scores = np.exp(scores)
probs = exp_scores / exp_scores.sum()

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
fig.patch.set_facecolor(BG)

ax = axes[0]
ax.set_facecolor(BG)
bars = ax.bar(classes, scores, color=RAW_COLOR, zorder=3)
for rect, v in zip(bars, scores):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.05, f"{v:.1f}",
            ha="center", va="bottom", color=TEXT, fontweight="bold")
ax.set_title("Skor mentah (logit)", color=TEXT, fontweight="bold")
ax.set_ylim(0, 2.4)
ax.set_ylabel("skor", color=TEXT)

ax2 = axes[1]
ax2.set_facecolor(BG)
bars2 = ax2.bar(classes, probs, color=PROB_COLOR, zorder=3)
for rect, v in zip(bars2, probs):
    ax2.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.2f}",
             ha="center", va="bottom", color=TEXT, fontweight="bold")
ax2.set_title("Setelah softmax (total = 1.00)", color=TEXT, fontweight="bold")
ax2.set_ylabel("probabilitas", color=TEXT)
ax2.set_ylim(0, 1.05)

for ax_ in axes:
    ax_.tick_params(colors=TEXT)
    for spine in ax_.spines.values():
        spine.set_edgecolor("#444466")
    ax_.grid(True, axis="y", color=GRID, linestyle="--", linewidth=0.6, zorder=0)
    ax_.set_axisbelow(True)

fig.suptitle("Softmax: dari skor mentah ke probabilitas", color=TEXT, fontweight="bold", y=1.0)
plt.tight_layout(pad=1.0)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "softmax.pdf")
fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=BG)
print(f"Saved: {output_path}")
print("scores:", scores.tolist())
print("probs:", [round(float(p), 4) for p in probs])
