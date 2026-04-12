"""Generate ML Types taxonomy diagram using matplotlib."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

BG = "#1A1A2E"
GREEN = "#76B900"
LIGHT_GREEN = "#A3D944"
CARD = "#2D2D44"
WHITE = "#FFFFFF"

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

def box(x, y, w, h, text, color=CARD, border=GREEN, fontsize=11, bold=False):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=WHITE, fontsize=fontsize, fontweight=weight)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))

# Root
box(3.5, 4.8, 3, 0.8, "Machine Learning", color=GREEN, border=GREEN, fontsize=14, bold=True)

# Level 1
box(0.3, 3.0, 2.8, 0.8, "Supervised\nLearning", fontsize=11, bold=True)
box(3.6, 3.0, 2.8, 0.8, "Unsupervised\nLearning", fontsize=11, bold=True)
box(6.9, 3.0, 2.8, 0.8, "Reinforcement\nLearning", fontsize=11, bold=True)

# Arrows from root
arrow(4.2, 4.8, 1.7, 3.8)
arrow(5.0, 4.8, 5.0, 3.8)
arrow(5.8, 4.8, 8.3, 3.8)

# Level 2 - Supervised
box(0.0, 1.2, 1.8, 0.7, "Regresi\nPrediksi Angka", border=LIGHT_GREEN, fontsize=9)
box(2.0, 1.2, 1.8, 0.7, "Klasifikasi\nPrediksi Kategori", border=LIGHT_GREEN, fontsize=9)
arrow(1.2, 3.0, 0.9, 1.9)
arrow(2.2, 3.0, 2.9, 1.9)

# Level 2 - Unsupervised
box(3.5, 1.2, 1.8, 0.7, "Clustering\nCari Kelompok", border=LIGHT_GREEN, fontsize=9)
box(5.5, 1.2, 1.8, 0.7, "Dimensionality\nReduction", border=LIGHT_GREEN, fontsize=9)
arrow(4.5, 3.0, 4.4, 1.9)
arrow(5.5, 3.0, 6.4, 1.9)

# Level 2 - Reinforcement
box(7.2, 1.2, 2.2, 0.7, "Agent belajar\ndari reward", border=LIGHT_GREEN, fontsize=9)
arrow(8.3, 3.0, 8.3, 1.9)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_types.pdf")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out}")
