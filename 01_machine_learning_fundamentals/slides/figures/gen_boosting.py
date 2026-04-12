"""Generate Boosting iterative process diagram using matplotlib."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

BG = "#1A1A2E"
WHITE = "#FFFFFF"

fig, ax = plt.subplots(figsize=(12, 3))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 3)
ax.axis("off")

models = [
    (1.0, "Model 1", "Banyak salah", "#EF5350"),
    (4.0, "Model 2", "Lebih baik", "#FF9800"),
    (7.0, "Model 3", "Makin baik", "#FFC107"),
    (10.0, "Model Final", "Akurat!", "#76B900"),
]

for x, title, subtitle, color in models:
    rect = mpatches.FancyBboxPatch((x - 0.8, 0.6), 1.6, 1.6,
                                    boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    text_color = "#000000" if color == "#FFC107" else WHITE
    ax.text(x, 1.65, title, ha="center", va="center",
            color=text_color, fontsize=11, fontweight="bold")
    ax.text(x, 1.15, subtitle, ha="center", va="center",
            color=text_color, fontsize=9)

# Arrows between models
arrow_labels = ["Fokus pada\nkesalahan", "Fokus pada\nsisa kesalahan", "..."]
for i, (x_start, label) in enumerate(zip([1.8, 4.8, 7.8], arrow_labels)):
    ax.annotate("", xy=(x_start + 1.4, 1.4), xytext=(x_start, 1.4),
                arrowprops=dict(arrowstyle="-|>", color=WHITE, lw=2))
    ax.text(x_start + 0.7, 2.5, label, ha="center", va="center",
            color="#888888", fontsize=8, style="italic")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boosting.pdf")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out}")
