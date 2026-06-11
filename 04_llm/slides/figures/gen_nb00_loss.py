"""Figure: kurva loss TinyGPT (3.68 -> 0.06), tema gelap. Output: figures/nb00_loss.pdf"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BG, GREEN, WHITE, GRAY = "#1A1A2E", "#76B900", "#FFFFFF", "#AAAACC"
steps  = [0, 100, 300, 600, 1000, 1500, 2000, 3000]
loss   = [3.681, 1.9, 0.708, 0.118, 0.09, 0.075, 0.068, 0.062]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(steps, loss, color=GREEN, lw=2.4, marker="o", ms=4)
ax.annotate("3,68 (acak)", (0, 3.681), color=WHITE, fontsize=8, xytext=(40, 3.2))
ax.annotate("0,06 (paham)", (3000, 0.062), color=GREEN, fontsize=8, xytext=(1700, 0.5))
ax.set_xlabel("Langkah training", color=WHITE, fontsize=9)
ax.set_ylabel("Loss (makin kecil makin baik)", color=WHITE, fontsize=9)
for s in ax.spines.values(): s.set_color(GRAY)
ax.tick_params(colors=GRAY, labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "nb00_loss.pdf"), facecolor=BG)
