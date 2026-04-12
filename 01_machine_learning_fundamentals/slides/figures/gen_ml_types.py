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

fig, ax = plt.subplots(figsize=(12, 6.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

def box(x, y, w, h, text, color=CARD, border=GREEN, fontsize=11, bold=False):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.18",
                                    facecolor=color, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=WHITE, fontsize=fontsize, fontweight=weight)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))

# Root — centered horizontally at x=6
box(4.5, 5.5, 3.0, 0.8, "Machine Learning", color=GREEN, border=GREEN, fontsize=16, bold=True)

# Level 1 — y=3.2
box(0.4, 3.2, 3.0, 0.8, "Supervised\nLearning", fontsize=13, bold=True)
box(4.5, 3.2, 3.0, 0.8, "Unsupervised\nLearning", fontsize=13, bold=True)
box(8.6, 3.2, 3.0, 0.8, "Reinforcement\nLearning", fontsize=13, bold=True)

# Arrows from root to level 1
arrow(5.3, 5.5, 1.9, 4.0)
arrow(6.0, 5.5, 6.0, 4.0)
arrow(6.7, 5.5, 10.1, 4.0)

# Level 2 — y=0.8
# Supervised children
box(0.1, 0.8, 2.0, 0.8, "Regresi\nPrediksi Angka", border=LIGHT_GREEN, fontsize=11)
box(2.5, 0.8, 2.0, 0.8, "Klasifikasi\nPrediksi Kategori", border=LIGHT_GREEN, fontsize=11)
arrow(1.5, 3.2, 1.1, 1.6)
arrow(2.3, 3.2, 3.5, 1.6)

# Unsupervised children
box(4.3, 0.8, 2.0, 0.8, "Clustering\nCari Kelompok", border=LIGHT_GREEN, fontsize=11)
box(6.7, 0.8, 2.0, 0.8, "Dimensionality\nReduction", border=LIGHT_GREEN, fontsize=11)
arrow(5.3, 3.2, 5.3, 1.6)
arrow(6.7, 3.2, 7.7, 1.6)

# Reinforcement child
box(8.9, 0.8, 2.4, 0.8, "Agent belajar\ndari reward", border=LIGHT_GREEN, fontsize=11)
arrow(10.1, 3.2, 10.1, 1.6)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_types.pdf")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out}")
